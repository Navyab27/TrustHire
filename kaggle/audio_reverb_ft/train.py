"""Kaggle GPU job: make the voice-spoof detector robust to room reverberation and noise.

Measured locally (results/audio): Spectra-0 int8 EER on In-the-Wild is 1.25 % clean but 21 / 27.5 / 34 % with
RT60 = 0.25 / 0.4 / 0.7 s reverb (26.5 % of genuine voices flagged at RT60 0.4). WPE dereverberation did not help.

This job fine-tunes Spectra-0 (XLS-R-300M + ECAPA, Apache-2.0) on ASVspoof 2019 LA train with heavy
reverb / noise / codec augmentation, then evaluates the released and the fine-tuned model *in PyTorch on the
same GPU* on the identical stratified In-the-Wild subset used locally (shards 0 and 5, n=400, seed 0) under
clean / reverb / noise. ITW is never used for training. Outputs (in /kaggle/working):
    spectra0_ft.safetensors, spectra0_ft.onnx, results.json, train_log.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# one BLAS/OpenMP thread per process: thread pools inherited across fork() can segfault DataLoader workers
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "TrustHire"


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# ----------------------------------------------------------------------------- setup
if not REPO.exists():
    sh(f"git clone --depth 1 https://github.com/Navyab27/TrustHire.git {REPO}")
sh("pip install -q av pydantic-settings onnxruntime safetensors 'huggingface_hub>=0.34'")
sys.path.insert(0, str(REPO / "src"))
os.environ["TRUSTHIRE_DATA_DIR"] = str(TMP / "data")
os.environ["TRUSTHIRE_MODELS_DIR"] = str(TMP / "models")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from trusthire.audio.datasets import fetch_shards  # noqa: E402
from trusthire.common.augment import AUDIO_CONDITIONS, add_noise, reverb, telephone  # noqa: E402
from trusthire.evaluation.metrics import report  # noqa: E402

DEV = "cuda"
NB = 64600
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH = int(os.environ.get("BATCH", "12"))
LR = float(os.environ.get("LR", "3e-6"))
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ----------------------------------------------------------------------------- data
# ASVspoof 2019 LA (ODC-By): attached Kaggle mirror awsaf49/asvpoof-2019-dataset, else download from DataShare
_found = sorted(Path("/kaggle/input").rglob("ASVspoof2019_LA_cm_protocols"))
if _found:
    LA = _found[0].parent
else:
    LA = TMP / "LA"
    if not (LA / "ASVspoof2019_LA_train").exists():
        z = TMP / "LA.zip"
        sh(f"curl -fL --retry 5 -A 'Mozilla/5.0' -o {z} https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(TMP)
        z.unlink()
print("ASVspoof LA root:", LA, flush=True)


def protocol(split: str) -> pd.DataFrame:
    name = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt"}[split]
    rows = []
    for line in (LA / "ASVspoof2019_LA_cm_protocols" / name).read_text().splitlines():
        _, utt, _, _, key = line.split()
        rows.append({"path": str(LA / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt}.flac"), "spoof": int(key == "spoof")})
    return pd.DataFrame(rows)


HAVE_FFMPEG = subprocess.run("ffmpeg -version", shell=True, capture_output=True).returncode == 0
print("ffmpeg available:", HAVE_FFMPEG, flush=True)


def codec_ffmpeg(x: np.ndarray, fmt: str, bitrate: int) -> np.ndarray:
    """Lossy codec round trip in an ffmpeg subprocess. (PyAV inside forked DataLoader workers segfaulted.)"""
    enc = {"opus": ["-c:a", "libopus", "-f", "ogg"], "mp3": ["-c:a", "libmp3lame", "-f", "mp3"],
           "aac": ["-c:a", "aac", "-f", "adts"]}[fmt]
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    a = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                        *enc, "-b:a", str(bitrate), "pipe:1"], input=pcm, capture_output=True, timeout=30)
    b = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1",
                        "pipe:1"], input=a.stdout, capture_output=True, timeout=30)
    y = np.frombuffer(b.stdout, np.int16).astype(np.float32) / 32767
    if len(y) < len(x) // 2:
        return x
    return np.pad(y[: len(x)], (0, max(0, len(x) - len(y)))).astype(np.float32)


def tile(x: np.ndarray) -> np.ndarray:
    return x[:NB] if len(x) >= NB else np.tile(x, int(np.ceil(NB / max(1, len(x)))))[:NB]


def preemph(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]]).astype(np.float32)


class Train(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        rng = np.random.default_rng()
        x, _ = sf.read(r.path, dtype="float32")
        if len(x) > NB:
            s = rng.integers(0, len(x) - NB)
            x = x[s:s + NB]
        u = rng.random()
        # augmentation mix targeting the measured failure modes (reverb >> noise > codec)
        if u < 0.5:
            x = reverb(x, float(rng.uniform(0.15, 0.9)), rng=rng)
        if rng.random() < 0.4:
            x = add_noise(x, float(rng.uniform(5, 30)), rng, "pink" if rng.random() < 0.5 else "white")
        if HAVE_FFMPEG and rng.random() < 0.25:
            x = codec_ffmpeg(x, str(rng.choice(["opus", "mp3", "aac"])), int(rng.choice([12000, 16000, 24000, 32000])))
        if rng.random() < 0.1:
            x = telephone(x)
        x = (x * rng.uniform(0.3, 1.2)).clip(-1, 1).astype(np.float32)
        return torch.from_numpy(preemph(tile(x))), torch.tensor(1 - int(r.spoof))  # index 1 = bona fide


# ----------------------------------------------------------------------------- model
spec_dir = snapshot_download("lab260/spectra_0", allow_patterns=["model.py", "*.json", "model.safetensors"])
sys.path.insert(0, spec_dir)
from model import Spectra0Model  # noqa: E402


def load_released() -> nn.Module:
    return Spectra0Model.from_pretrained(spec_dir).to(DEV)


@torch.inference_mode()
def p_spoof(model: nn.Module, wavs: np.ndarray) -> np.ndarray:
    """Evaluation runs in fp32: the fine-tuned model overflowed fp16 (all-NaN outputs) in a previous run."""
    model.eval().float()
    out = []
    for i in range(0, len(wavs), 8):
        x = torch.from_numpy(np.stack([preemph(w) for w in wavs[i:i + 8]])).to(DEV)
        out.append(torch.softmax(model(x).float(), -1)[:, 0].cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- ITW evaluation (same subset as local)
fetch_shards("itw", shard_ids=[0, 5])
from trusthire.audio.eval import stratified_subset  # noqa: E402

utts = stratified_subset("itw", 400, 0)
y = np.array([u.label for u in utts])
EVAL_CONDS = ["clean", "reverb_0.25", "reverb_0.4", "reverb_0.7", "noise_snr10", "opus_16k"]
eval_x = {}
for c in EVAL_CONDS:
    rng = np.random.default_rng(0)
    eval_x[c] = np.stack([tile(AUDIO_CONDITIONS[c](u.audio, rng)) for u in utts])


def evaluate(model: nn.Module, tag: str) -> dict:
    res = {}
    for c in EVAL_CONDS:
        s = p_spoof(model, eval_x[c])
        n_nan = int(np.isnan(s).sum())
        if n_nan:
            print(f"[{tag}] {c}: {n_nan} NaN scores -> treated as P(spoof)=0.5", flush=True)
            s = np.nan_to_num(s, nan=0.5)
        r = report(y, s, bootstrap=300).as_dict()
        r["nan_scores"] = n_nan
        b = s[y == 0]
        r["genuine_flagged_at_0.5"] = float((b >= 0.5).mean())
        res[c] = r
        print(f"[{tag}] {c:12s} AUC={r['auc']:.4f} EER={100 * r['eer']:.2f}% genuine-flagged={100 * r['genuine_flagged_at_0.5']:.1f}%",
              flush=True)
    return res


results = {"subset": "In-the-Wild shards 0+5, stratified n=400, seed 0 (identical to results/audio)",
           "gpu": torch.cuda.get_device_name(0)}
model = load_released()
results["released"] = evaluate(model, "released")

# ----------------------------------------------------------------------------- fine-tune
train_df = protocol("train")
dev_df = protocol("dev").sample(2000, random_state=0)
w_bona = float((train_df.spoof == 1).sum() / max((train_df.spoof == 0).sum(), 1))
dl = DataLoader(Train(train_df), batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
for p in model.ssl_encoder.model.feature_extractor.parameters():  # freeze the CNN feature encoder
    p.requires_grad = False
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
total = EPOCHS * len(dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05)
scaler = torch.amp.GradScaler()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_bona], device=DEV))
log = []
t0 = time.time()
step = 0
best_state = None
try:
    for ep in range(EPOCHS):
        model.train()
        for x, t in dl:
            x, t = x.to(DEV, non_blocking=True), t.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x).float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0:
                log.append({"step": step, "epoch": ep, "loss": float(loss.detach()), "min": (time.time() - t0) / 60})
                print(f"step {step}/{total} loss {float(loss.detach()):.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(log).to_csv(WORK / "train_log.csv", index=False)
        torch.save(model.state_dict(), WORK / "ft_last.pt")  # persistent output: survives a later failure
except RuntimeError as e:  # e.g. a crashed DataLoader worker: keep the last completed checkpoint
    print("TRAINING INTERRUPTED:", e, flush=True)
    results["training_interrupted"] = str(e)[:300]
    if (WORK / "ft_last.pt").exists():
        model.load_state_dict(torch.load(WORK / "ft_last.pt", map_location=DEV))

results["finetuned"] = evaluate(model, "finetuned")
results["config"] = {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "train": "ASVspoof2019 LA train + reverb/noise/codec aug",
                     "minutes": (time.time() - t0) / 60}

# ----------------------------------------------------------------------------- export
from safetensors.torch import save_file  # noqa: E402

model.eval().float()
save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, str(WORK / "spectra0_ft.safetensors"))
dummy = torch.randn(1, NB, device=DEV) * 0.05
torch.onnx.export(model, (dummy,), str(WORK / "spectra0_ft.onnx"), input_names=["wav"], output_names=["logits"],
                  dynamic_axes={"wav": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
(WORK / "results.json").write_text(json.dumps(results, indent=2))
(WORK / "ft_last.pt").unlink(missing_ok=True)
print("DONE", json.dumps({c: {k: results[k][c]["eer"] for k in ("released", "finetuned")} for c in EVAL_CONDS}), flush=True)
