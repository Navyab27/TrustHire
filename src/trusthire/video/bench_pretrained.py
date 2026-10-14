"""Benchmark off-the-shelf face-deepfake / AI-generated-image classifiers on the custom recordings.

Visual task: AI-generated video (label 1) vs real camera footage (label 0). Each model sees K frames per video,
both as the full frame and as the 1.3x face crop; the video score is the mean P(fake). Reports per-model AUC
(and accuracy at 0.5) so the live video detector can be chosen on interview-like data.

    python -m trusthire.video.bench_pretrained [--frames 8]
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from trusthire.common.face import FaceStack, square_crop
from trusthire.common.media import iter_frames, probe
from trusthire.common.registry import models_dir
from trusthire.common.settings import REPO_ROOT
from trusthire.evaluation.dataset import custom_root, load_manifest
from trusthire.evaluation.metrics import report

HF_CANDIDATES = [
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "haywoodsloan/ai-image-detector-deploy",
    "Ateeqq/ai-vs-human-image-detector",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
    "umm-maybe/AI-image-detector",
    "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
]
FAKE_WORDS = ("fake", "ai", "artificial", "generated", "synthetic", "deepfake", "label_1")


def fake_indices(id2label: dict[int, str]) -> list[int]:
    idx = [
        i
        for i, n in id2label.items()
        if any(w in n.lower() for w in FAKE_WORDS) and "real" not in n.lower() and "human" not in n.lower()
    ]
    if not idx:
        raise ValueError(f"cannot tell which label is fake: {id2label}")
    return idx


class HFDetector:
    def __init__(self, repo: str):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        cache = str(models_dir() / "hf")
        self.proc = AutoImageProcessor.from_pretrained(repo, cache_dir=cache)
        self.model = AutoModelForImageClassification.from_pretrained(repo, cache_dir=cache).eval()
        self.fake = fake_indices({int(k): v for k, v in self.model.config.id2label.items()})
        self.labels = self.model.config.id2label

    @torch.inference_mode()
    def p_fake(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs_bgr]
        p = torch.softmax(self.model(**self.proc(images=rgb, return_tensors="pt")).logits, -1)
        return p[:, self.fake].sum(-1).numpy()


class OnnxVideoDetector:
    """The current live detectors (DeepfakeBench FF++ EfficientNet-B4 / Xception)."""

    def __init__(self, backbone: str):
        from trusthire.video.infer import VideoDeepfakeDetector

        self.det = VideoDeepfakeDetector(backbone, 3)

    def p_fake(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        return self.det.frame_scores(imgs_bgr)[0]


def sample_frames(path: str, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dur = probe(path).duration
    times = set(np.round(np.linspace(0.5, max(0.6, dur - 0.5), k), 1))
    fs = FaceStack.get()
    full, crops = [], []
    for t, f in iter_frames(path, 10.0, 960):
        if round(t, 1) in times:
            faces = fs.detector.detect(f, max_num=1)
            full.append(f)
            crops.append(square_crop(f, faces[0], 1.3, 256) if faces else f)
    return full, crops


def run(k: int = 8) -> dict[str, Any]:
    root = custom_root()
    man = load_manifest(root)
    man["y"] = (man.attack == "face_and_voice").astype(int)  # visual fakes; other attacks have real faces
    frames = {r.session_id: sample_frames(str(root / r.recording), k) for _, r in man.iterrows()}
    results: dict[str, Any] = {}
    cands: list[tuple[str, Any]] = [
        ("effnb4 (current)", lambda: OnnxVideoDetector("effnb4")),
        ("xception", lambda: OnnxVideoDetector("xception")),
    ]
    cands += [(m, (lambda m=m: HFDetector(m))) for m in HF_CANDIDATES]
    for name, make in cands:
        try:
            det = make()
        except Exception as e:
            print(f"{name}: load failed ({type(e).__name__}: {str(e)[:120]})")
            continue
        for view in ("crop", "full"):
            scores, n_imgs, t_inf = [], 0, 0.0
            for _, r in man.iterrows():
                imgs = frames[r.session_id][1 if view == "crop" else 0]
                t1 = time.perf_counter()
                scores.append(float(np.mean(det.p_fake(imgs))))
                t_inf += time.perf_counter() - t1
                n_imgs += len(imgs)
            s = np.array(scores)
            rep = report(man.y.to_numpy(), s, probabilistic=True)
            per_video = dict(zip(man.session_id, np.round(s, 3).tolist(), strict=True))
            results[f"{name}|{view}"] = {
                "model": name,
                "view": view,
                "auc": rep.auc,
                "eer": rep.eer,
                "acc_at_0.5": float(((s >= 0.5) == man.y).mean()),
                "mean_real": float(s[man.y == 0].mean()),
                "mean_ai": float(s[man.y == 1].mean()),
                "ms_per_image": 1e3 * t_inf / max(n_imgs, 1),
                "per_video": per_video,
            }
            print(
                f"{name:52s} {view:4s} AUC={rep.auc:.3f} acc@0.5={results[f'{name}|{view}']['acc_at_0.5']:.2f} "
                f"real={s[man.y == 0].mean():.2f} ai={s[man.y == 1].mean():.2f}",
                flush=True,
            )
        del det
    out = REPO_ROOT / "results" / "video"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pretrained_on_custom.json").write_text(json.dumps(results, indent=2))
    rows = sorted(results.values(), key=lambda r: -r["auc"])
    md = [
        "| Model | Input | AUC | Acc@0.5 | mean P(fake) real | mean P(fake) AI | ms/img |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['model']} | {r['view']} | {r['auc']:.3f} | {r['acc_at_0.5']:.2f} | {r['mean_real']:.2f} | "
        f"{r['mean_ai']:.2f} | {r['ms_per_image']:.0f} |"
        for r in rows
    ]
    note = (
        f"\n\nCustom recordings: {int(man.y.sum())} AI-generated vs {int((1 - man.y).sum())} real-camera videos, "
        f"{k} frames each. Small set (one real participant) — indicative only.\n"
    )
    (out / "pretrained_on_custom.md").write_text("\n".join(md) + note, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "pretrained_on_custom.csv", index=False)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    run(ap.parse_args().frames)
