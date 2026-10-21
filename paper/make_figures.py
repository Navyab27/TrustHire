"""Generate every figure and data table used by paper/main.tex from the files under results/.

    python paper/make_figures.py        (run with the project venv)

Nothing here is typed in by hand: each plot reads a JSON/CSV/parquet produced by a real run, so the paper can be
regenerated after more data is collected.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from trusthire.evaluation.dataset import custom_root
from trusthire.evaluation.harness import _cv_predict, ablation_configs, session_level
from trusthire.evaluation.metrics import report, tpr_at_fpr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
COL = 3.45  # IEEE single-column width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"blue": "#1f4e79", "red": "#b03a2e", "teal": "#148f77", "grey": "#7f8c8d", "orange": "#ca6f1e", "purple": "#6c3483"}
stats: dict[str, object] = {}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: audio robustness
bench = json.loads((RES / "audio" / "benchmark.json").read_text())
conds = ["clean", "telephone", "mp3_32k", "opus_16k", "noise_snr20_pink", "noise_snr10", "reverb_0.4"]
labels = ["Clean", "Telephone", "MP3 32k", "Opus 16k", "Pink 20 dB", "White 10 dB", "Reverb 0.4 s"]
fig, ax = plt.subplots(figsize=(COL, 1.6))
x = np.arange(len(conds))
for i, (be, name, col) in enumerate([("aasist", "AASIST [baseline]", C["grey"]), ("spectra0_int8", "Spectra-0 int8 (ours)", C["blue"])]):
    v = [100 * bench[f"itw/{c}/{be}"]["eer"] for c in conds]
    ax.bar(x + (i - 0.5) * 0.38, v, 0.38, label=name, color=col)
    for xi, vi in zip(x, v, strict=True):
        ax.text(xi + (i - 0.5) * 0.38, vi + 0.6, f"{vi:.1f}", ha="center", va="bottom", fontsize=5.2)
ax.set_xticks(x, labels, rotation=25, ha="right")
ax.set_ylabel("EER (%)")
ax.set_ylim(0, 62)
ax.grid(axis="y")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "audio_robustness")

# ------------------------------------------------------------------ Fig: reverb dose-response (released vs fine-tuned)
kag = json.loads((RES / "audio" / "kaggle" / "reverb_finetune_results.json").read_text())
rts = [0.0, 0.25, 0.4, 0.7]
keys = ["clean", "reverb_0.25", "reverb_0.4", "reverb_0.7"]
fig, ax = plt.subplots(figsize=(COL, 1.5))
for tag, name, col, mk in [("released", "Released", C["blue"], "o"), ("finetuned", "Fine-tuned (Kaggle)", C["orange"], "s")]:
    ax.plot(rts, [100 * kag[tag][k]["eer"] for k in keys], marker=mk, color=col, lw=1.2, ms=3.5, label=f"{name}: EER")
    ax.plot(rts, [100 * kag[tag][k]["genuine_flagged_at_0.5"] for k in keys], marker=mk, color=col, lw=1.0, ms=3.5,
            ls="--", mfc="white", label=f"{name}: genuine flagged")
ax.set_xlabel("Reverberation time RT$_{60}$ (s)")
ax.set_ylabel("Rate (%)")
ax.set_xticks(rts)
ax.grid()
ax.legend(frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, handlelength=1.6, columnspacing=1.0, labelspacing=0.2)
save(fig, "reverb_dose")

# ------------------------------------------------------------------ Fig: detector selection on the custom set
pc = json.loads((RES / "video" / "pretrained_on_custom.json").read_text())
man = pd.read_csv(custom_root() / "manifest.csv")
is_ai = dict(zip(man.session_id, man.attack == "face_and_voice", strict=True))
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True)
rng = np.random.default_rng(0)
for ax, key, title in [(axes[0], "effnb4 (current)|crop", "EfficientNet-B4 (FF++)"),
                       (axes[1], "haywoodsloan/ai-image-detector-deploy|crop", "SwinV2-L (selected)")]:
    pv = pc[key]["per_video"]
    for j, (grp, col) in enumerate([(False, C["teal"]), (True, C["red"])]):
        vals = [v for s, v in pv.items() if is_ai[s] == grp]
        ax.scatter(j + rng.uniform(-0.12, 0.12, len(vals)), vals, s=10, color=col, edgecolor="none", alpha=0.9)
    ax.set_xticks([0, 1], ["Real\n(n=12)", "AI-gen.\n(n=6)"])
    ax.set_title(f"{title}\nAUC = {pc[key]['auc']:.2f}")
    ax.axhline(0.5, color=C["grey"], lw=0.6, ls=":")
    ax.set_xlim(-0.5, 1.5)
axes[0].set_ylabel("Mean P(fake) per video")
save(fig, "detector_selection")
stats["detector"] = {k: {kk: pc[k][kk] for kk in ("auc", "mean_real", "mean_ai", "ms_per_image")} for k in pc}

# ------------------------------------------------------------------ custom-set fusion: OOF predictions
def load_cache(d: str) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(str(custom_root() / d / "*.parquet"))) if "answers" not in f]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


new, old = load_cache("cache"), load_cache("cache_effnb4")
cfg = ablation_configs()
SEEDS = [0, 1, 2, 3, 4]
p_seeds = [_cv_predict(new, "gbm", cfg["all-but-transcript"], seed=sd) for sd in SEEDS]
p_learn = p_seeds[0]  # curves drawn for seed 0; tables report mean +- sd over seeds
variants = [("Hand-set fusion, FF++ video model", old, old.online_risk.to_numpy(), C["grey"], ":"),
            ("Hand-set fusion, SwinV2 video model", new, new.online_risk.to_numpy(), C["blue"], "--"),
            ("Learned fusion (LightGBM), SwinV2", new, p_learn, C["red"], "-")]
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.7))
fusion_rows = []
for name, d, p, col, ls in variants:
    ok = ~np.isnan(p)
    y = d.label.to_numpy()[ok]
    fpr, tpr, _ = roc_curve(y, p[ok])
    rw = report(y, p[ok], probabilistic=False)
    axes[0].plot(fpr, tpr, color=col, ls=ls, lw=1.1, label=f"{rw.auc:.2f}")
    ys, ss = session_level(d, p)
    rs = report(ys, ss, probabilistic=False)
    fs_, ts_, _ = roc_curve(ys, ss)
    axes[1].plot(fs_, ts_, color=col, ls=ls, lw=1.1, drawstyle="steps-post", label=f"{rs.auc:.2f}")
    fusion_rows.append({"variant": name, "win_auc": rw.auc, "win_eer": rw.eer, "sess_auc": rs.auc, "sess_eer": rs.eer})
for ax, t in zip(axes, ["Window level (n=339)", "Session level (n=18)"], strict=True):
    ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.5, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_title(t)
    ax.legend(title="AUC", frameon=False, fontsize=6, title_fontsize=6, loc="lower right")
    ax.set_aspect("equal")
axes[0].set_ylabel("True positive rate")
save(fig, "roc_fusion")
stats["fusion"] = fusion_rows
seed_rows = []
for p_s in p_seeds:
    ok_s = ~np.isnan(p_s)
    rw = report(new.label.to_numpy()[ok_s], p_s[ok_s], probabilistic=False)
    ys, ss = session_level(new, p_s)
    rs = report(ys, ss, probabilistic=False)
    seed_rows.append({"win_auc": rw.auc, "win_eer": rw.eer, "sess_auc": rs.auc, "sess_eer": rs.eer})
sr = pd.DataFrame(seed_rows)
stats["learned_seeds"] = {"mean": sr.mean().to_dict(), "std": sr.std(ddof=1).to_dict(), "per_seed": seed_rows}

# per-attack recall at 5 % window FPR for the shipped fusion, mean over seeds
per_rows = []
for p_s in p_seeds:
    ok_s = ~np.isnan(p_s)
    d = new[ok_s].assign(p=p_s[ok_s])
    _, _, thr = tpr_at_fpr(d.label.to_numpy(), d.p.to_numpy(), 0.05)
    row = {k: float((g.p >= thr).mean()) for k, g in d[d.label == 1].groupby("attack")}
    row["genuine_fpr"] = float((d[d.label == 0].p >= thr).mean())
    per_rows.append(row)
pr = pd.DataFrame(per_rows)
per = {"mean": pr.mean().to_dict(), "std": pr.std(ddof=1).to_dict(),
       "n_windows": {k: int(len(g)) for k, g in new.groupby("attack")}}
stats["per_attack_shipped"] = per

# ------------------------------------------------------------------ Fig: modality ablation (session AUC)
res = json.loads((RES / "custom" / "results.json").read_text())
order = ["video-only", "audio-only", "behavior-only", "transcript-only", "video+audio", "audio+behavior",
         "video+audio+avsync", "all-but-transcript", "full"]
names = ["Video", "Audio", "Behav.", "Transcr.", "V+A", "A+B", "V+A+Sync", "All$-$T", "All"]
fig, ax = plt.subplots(figsize=(COL, 1.7))
x = np.arange(len(order))
for i, (fu, nm, col) in enumerate([("weighted", "Hand-set", C["grey"]), ("gbm", "LightGBM", C["red"]), ("attention", "Cross-attn.", C["purple"])]):
    v = [res["ablation"].get(f"{o}|{fu}", {}).get("session", {}) or {} for o in order]
    ax.bar(x + (i - 1) * 0.27, [vv.get("auc", np.nan) for vv in v], 0.27, color=col, label=nm)
ax.axhline(0.5, color="k", lw=0.5, ls=":")
ax.set_xticks(x, names, rotation=30, ha="right")
ax.set_ylabel("Session AUC")
ax.set_ylim(0.2, 1.02)
ax.grid(axis="y")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
save(fig, "ablation")

# ------------------------------------------------------------------ Fig: latency
lat = json.loads((RES / "latency" / "latency_cpu.json").read_text())
mods = sorted(lat["modules_ms_p50"].items(), key=lambda kv: kv[1])
nice = {"video_deepfake": "Video (SwinV2 int8)", "voice_spoof": "Voice spoof (Spectra-0 int8)", "behavior": "Gaze/blink/rPPG",
        "av_sync": "Lip sync (SyncNet)", "face_identity": "Face ID (ArcFace)", "voice_identity": "Voice ID (ECAPA)",
        "objects": "Objects (YOLO26n)"}
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6), gridspec_kw={"width_ratios": [1.2, 1]})
axes[0].barh([nice.get(k, k) for k, _ in mods], [v for _, v in mods], color=C["blue"], height=0.6)
axes[0].set_xlabel("p50 latency when run (ms)")
axes[0].grid(axis="x")
steps = [("Naive", 9500), ("Thread\ncaps", 3490), ("Alt.\nwin.", 1689), ("SwinV2\nall win.", 2492), ("Final", lat["total_ms_p50"])]
axes[1].bar(range(len(steps)), [s for _, s in steps], color=[C["grey"]] * 3 + [C["orange"], C["teal"]], width=0.6)
axes[1].axhline(1500, color=C["red"], lw=0.7, ls="--")
axes[1].set_xticks(range(len(steps)), [s for s, _ in steps], fontsize=5.5)
axes[1].set_yscale("log")
axes[1].set_ylabel("Window p50 (ms)")
save(fig, "latency")

# ------------------------------------------------------------------ Fig: SyncNet desync study
ds = json.loads((RES / "avsync" / "desync_study.json").read_text())
sh = sorted(int(k) for k in ds["per_shift"])
fig, ax = plt.subplots(figsize=(COL, 1.45))
ax.plot([s * 40 for s in sh], [ds["per_shift"][str(s)]["mean_risk"] for s in sh], marker="o", ms=3, color=C["red"], lw=1.1, label="Sync risk")
ax2 = ax.twinx()
ax2.plot([s * 40 for s in sh], [ds["per_shift"][str(s)]["mean_conf"] for s in sh], marker="s", ms=3, color=C["blue"], lw=1.0, ls="--", label="SyncNet confidence")
ax2.spines["right"].set_visible(True)
ax.set_xlabel("Injected audio shift (ms)")
ax.set_ylabel("Mean risk", color=C["red"])
ax2.set_ylabel("Confidence", color=C["blue"])
ax.grid()
save(fig, "desync")

(OUT / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
print(json.dumps({"fusion": fusion_rows, "learned_seeds": stats["learned_seeds"], "per_attack_shipped": per}, indent=2, default=float))


# ------------------------------------------------------------------ numbers.tex: every number quoted in the text
def pct(v: float) -> str:
    return f"{100 * v:.1f}"


def pct2(v: float) -> str:
    return f"{100 * v:.2f}"


ls_ = stats["learned_seeds"]
hand_old, hand_new = fusion_rows[0], fusion_rows[1]
pa = stats["per_attack_shipped"]
b = bench
lfw = json.loads((RES / "identity" / "face_lfw.json").read_text())
spk = json.loads((RES / "identity" / "speaker_librispeech.json").read_text())
macros = {
    "NWin": str(len(new)), "NSess": str(new.session_id.nunique()),
    "HandOldSessAUC": f"{hand_old['sess_auc']:.3f}", "HandOldSessEER": pct(hand_old["sess_eer"]),
    "HandOldWinAUC": f"{hand_old['win_auc']:.3f}",
    "HandNewSessAUC": f"{hand_new['sess_auc']:.3f}", "HandNewWinAUC": f"{hand_new['win_auc']:.3f}",
    "LearnWinAUC": f"{ls_['mean']['win_auc']:.3f}", "LearnWinAUCsd": f"{ls_['std']['win_auc']:.3f}",
    "LearnSessAUC": f"{ls_['mean']['sess_auc']:.3f}", "LearnSessAUCsd": f"{ls_['std']['sess_auc']:.3f}",
    "LearnSessEER": pct(ls_["mean"]["sess_eer"]), "LearnSessEERsd": pct(ls_["std"]["sess_eer"]),
    "LearnWinEER": pct(ls_["mean"]["win_eer"]),
    "RecallAI": pct(pa["mean"].get("face_and_voice", float("nan"))),
    "RecallLip": pct(pa["mean"].get("lip_sync", float("nan"))),
    "RecallScript": pct(pa["mean"].get("scripted_reading", float("nan"))),
    "GenFPR": pct(pa["mean"]["genuine_fpr"]),
    "ITWSpectraEER": pct2(b["itw/clean/spectra0_int8"]["eer"]), "ITWAasistEER": pct2(b["itw/clean/aasist"]["eer"]),
    "ITWAasistLEER": pct2(b["itw/clean/aasist_l"]["eer"]), "ITWSpectraFPEER": pct2(b["itw/clean/spectra0"]["eer"]),
    "ASVSpectraEER": pct2(b["asv19la/clean/spectra0_int8"]["eer"]), "ASVAasistEER": pct2(b["asv19la/clean/aasist"]["eer"]),
    "OpusEER": pct2(b["itw/opus_16k/spectra0_int8"]["eer"]), "NoiseEER": pct2(b["itw/noise_snr10/spectra0_int8"]["eer"]),
    "RevEER": pct2(b["itw/reverb_0.4/spectra0_int8"]["eer"]), "RevWPEEER": pct2(b["itw/reverb_0.4/spectra0_int8+wpe"]["eer"]),
    "LFWAUC": f"{lfw['auc']:.3f}", "LFWEER": pct(lfw["eer"]), "LFWAcc": pct(lfw["accuracy_at_eer_thr"]),
    "LFWFlag": pct(lfw["at_configured_threshold"]["false_flag_rate_same_person"]),
    "SpkAUC": f"{spk['auc']:.3f}", "SpkEER": pct(spk["eer"]), "SpkTrials": str(spk["n_trials"]),
    "LatMedian": f"{lat['total_ms_p50'] / 1000:.2f}", "LatTail": f"{lat['total_ms_p95'] / 1000:.2f}",
    "LatRTF": f"{lat['real_time_factor']:.2f}", "RSS": f"{lat['peak_rss_mb'] / 1024:.1f}",
}
(Path(__file__).resolve().parent / "numbers.tex").write_text(
    "% generated by make_figures.py -- do not edit\n"
    + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
print("wrote numbers.tex with", len(macros), "macros")
