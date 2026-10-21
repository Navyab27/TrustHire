"""Evaluation harness for the custom interview test set.

    python -m trusthire.evaluation.harness extract            # run the pipeline over every recording (cached)
    python -m trusthire.evaluation.harness evaluate           # ablations x fusers, grouped CV, tables + plots
    python -m trusthire.evaluation.harness evaluate --synthetic   # smoke run on SYNTHETIC data (separate folder)
    python -m trusthire.evaluation.harness train-final        # fit + calibrate fusers on all real data -> models/fusion

Protocol
  * Leave-participants-out cross-validation (GroupKFold by participant_id) so no person is in train and test.
  * Inside each training fold, 20 % of participants are held out to fit the calibrator (temperature / isotonic).
  * Window-level metrics on out-of-fold predictions; session-level metrics use the trust-score aggregation.
  * The operating threshold for "flag" metrics is chosen at 5 % FPR on genuine windows (out-of-fold).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

from trusthire.common.registry import models_dir
from trusthire.common.settings import REPO_ROOT
from trusthire.common.types import MODALITY_GROUPS
from trusthire.evaluation.dataset import (
    STRATA,
    custom_root,
    load_labels,
    load_manifest,
    validate,
    write_template,
)
from trusthire.evaluation.metrics import ece, report, tpr_at_fpr
from trusthire.fusion.features import MODS, mask_modalities
from trusthire.fusion.models import CalibratedFuser, Calibrator, build_fuser

log = logging.getLogger("harness")
FUSERS = ["weighted", "gbm", "attention"]


def ablation_configs() -> dict[str, tuple[str, ...]]:
    g = {k: tuple(m.value for m in v) for k, v in MODALITY_GROUPS.items()}
    cfgs: dict[str, tuple[str, ...]] = {
        "video-only": g["video"],
        "audio-only": g["audio"],
        "behavior-only": g["behavior"],
        "transcript-only": g["transcript"],
    }
    for a, b in itertools.combinations(["video", "audio", "behavior"], 2):
        cfgs[f"{a}+{b}"] = g[a] + g[b]
    cfgs["video+audio+avsync"] = g["video"] + g["audio"] + g["avsync"]
    cfgs["all-but-transcript"] = tuple(m for m in MODS if m not in g["transcript"])
    cfgs["full"] = MODS
    return cfgs


# ----------------------------------------------------------------------------- extraction
def extract(root: Path | None = None, force: bool = False) -> pd.DataFrame:
    import cv2

    from trusthire.common.media import load_audio
    from trusthire.fusion.features import row_from_outputs
    from trusthire.identity.infer import Enrollment, enroll_face_from_image, enroll_voice_from_audio
    from trusthire.pipeline.analyzer import AnalyzerConfig
    from trusthire.pipeline.offline import analyze_file
    from trusthire.transcript.nlp import read_resume

    root = root or custom_root()
    problems = validate(root)
    if problems:
        raise SystemExit("dataset problems:\n  " + "\n  ".join(problems))
    man = load_manifest(root)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    frames = []
    for _, r in man.iterrows():
        out = cache / f"{r.session_id}.parquet"
        if out.exists() and not force:
            frames.append(pd.read_parquet(out))
            continue
        enr = Enrollment()
        if r.id_photo:
            img = cv2.imread(str(root / r.id_photo))
            if img is not None:
                enr.face = enroll_face_from_image(img)
        if r.voice_sample:
            enr.voice = enroll_voice_from_audio(load_audio(root / r.voice_sample))
        resume = read_resume(root / r.resume) if r.resume else None
        log.info("analysing %s (%s)", r.session_id, r.attack)
        res = analyze_file(root / r.recording, enr, resume, AnalyzerConfig())
        labels = load_labels(root, r.session_id)
        rows = []
        for i, p in enumerate(res.points):
            y, kind = labels.label_at(p.t_start, p.t_end)
            row: dict[str, Any] = {
                "session_id": r.session_id,
                "participant_id": r.participant_id,
                "window": i,
                "t_start": p.t_start,
                "t_end": p.t_end,
                "label": y,
                "attack": kind,
                "session_attack": r.attack,
                "online_risk": p.risk,
                "latency_ms": p.latency_ms,
                "synthetic": False,
            }
            row.update(row_from_outputs(p.modules))
            row.update({k: r[k] for k in STRATA})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(out)
        ans = []
        for a in res.answers:  # answer-level rows for transcript/train.py
            y, kind = labels.label_at(a["t_start"], a["t_end"], min_overlap=0.5)
            ans.append(
                {
                    "session_id": r.session_id,
                    "participant_id": r.participant_id,
                    "t_start": a["t_start"],
                    "t_end": a["t_end"],
                    "label": y,
                    "attack": kind,
                    "text": a["text"],
                    "scripted_answer_score": a["scripted_score"],
                    "ai_text_prob": a["ai_text_prob"],
                    **{k: v for k, v in a["features"].items() if k != "ai_text_prob"},
                }
            )
        pd.DataFrame(ans).to_parquet(cache / f"{r.session_id}_answers.parquet")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------- evaluation
def _cv_predict(
    df: pd.DataFrame, method: str, mods: tuple[str, ...], seed: int = 0, cal_method: str | None = None
) -> np.ndarray:
    groups = df["participant_id"].to_numpy()
    n_groups = len(np.unique(groups))
    k = min(5, n_groups)
    oof = np.full(len(df), np.nan)
    data = mask_modalities(df, mods)
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=k).split(data, groups=groups):
        tr_groups = np.unique(groups[tr])
        cal_g = rng.choice(tr_groups, max(1, len(tr_groups) // 5), replace=False)
        cal = np.isin(groups[tr], cal_g)
        d_fit, d_cal = data.iloc[tr[~cal]], data.iloc[tr[cal]]
        y_fit, y_cal = d_fit.label.to_numpy(), d_cal.label.to_numpy()
        if len(np.unique(y_fit)) < 2:
            continue
        cm = cal_method or "temperature"  # isotonic on ~3 calibration videos creates ties and hurts ranking
        f = CalibratedFuser(build_fuser(method, MODS, seed), Calibrator(cm))
        f.fit(d_fit, y_fit, d_cal, y_cal)
        oof[te] = f.predict_proba(data.iloc[te])
    return oof


def session_level(df: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = df.assign(p=p).dropna(subset=["p"])
    ys, ss = [], []
    for _, g in d.groupby("session_id"):
        r = np.sort(g.p.to_numpy())
        k = max(1, int(np.ceil(0.1 * len(r))))
        ss.append(0.6 * r[-k:].mean() + 0.4 * r.mean())
        ys.append(int(g.label.max()))
    return np.array(ys), np.array(ss)


def evaluate(df: pd.DataFrame, out: Path, synthetic: bool, fusers: list[str] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    fusers = fusers or FUSERS
    results: dict[str, Any] = {
        "synthetic": synthetic,
        "n_windows": len(df),
        "n_sessions": int(df.session_id.nunique()),
        "n_participants": int(df.participant_id.nunique()),
        "ablation": {},
        "per_attack": {},
        "fairness": {},
        "calibration": {},
    }
    oof_store: dict[str, np.ndarray] = {}
    for cfg, mods in ablation_configs().items():
        for method in fusers:
            p = _cv_predict(df, method, mods)
            ok = ~np.isnan(p)
            y = df.label.to_numpy()[ok]
            if len(np.unique(y)) < 2:
                continue
            w = report(y, p[ok], bootstrap=200).as_dict()
            ys, ss = session_level(df, p)
            sess = report(ys, ss).as_dict() if len(np.unique(ys)) == 2 else None
            results["ablation"][f"{cfg}|{method}"] = {
                "config": cfg,
                "fuser": method,
                "window": w,
                "session": sess,
            }
            oof_store[f"{cfg}|{method}"] = p
            log.info("%-22s %-9s window AUC %.3f EER %.3f", cfg, method, w["auc"], w["eer"])
    # pick the best full-fusion model for the detailed analyses
    full = {k: v for k, v in results["ablation"].items() if v["config"] == "full"}
    best = max(full, key=lambda k: full[k]["window"]["auc"]) if full else None
    results["best_full"] = best
    if best:
        p = oof_store[best]
        ok = ~np.isnan(p)
        d = df[ok].assign(p=p[ok])
        genuine = d[d.label == 0]
        _, _, thr = tpr_at_fpr(d.label.to_numpy(), d.p.to_numpy(), 0.05)
        results["threshold_at_5fpr"] = thr
        for kind, g in d[d.label == 1].groupby("attack"):
            results["per_attack"][kind] = {"n_windows": len(g), "recall_at_5fpr": float((g.p >= thr).mean())}
        results["per_attack"]["genuine"] = {
            "n_windows": len(genuine),
            "fpr": float((genuine.p >= thr).mean()),
        }
        for col in [c for c in STRATA if c in d and d[c].astype(str).str.len().gt(0).any()]:
            rows = {}
            for val, g in d.groupby(d[col].astype(str)):
                entry: dict[str, Any] = {
                    "n": len(g),
                    "fpr_at_global_thr": float((g[g.label == 0].p >= thr).mean())
                    if (g.label == 0).any()
                    else None,
                }
                if g.label.nunique() == 2:
                    entry["auc"] = report(g.label.to_numpy(), g.p.to_numpy()).auc
                rows[val] = entry
            results["fairness"][col] = rows
        results["calibration"] = {"ece": ece(d.label.to_numpy(), d.p.to_numpy()), "fuser": best}
        _plots(df, oof_store, best, out)
    if "latency_ms" in df and df.latency_ms.notna().any():
        results["latency_ms"] = {
            "p50": float(df.latency_ms.median()),
            "p95": float(df.latency_ms.quantile(0.95)),
        }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _tables(results, out)
    return results


def _plots(df: pd.DataFrame, oof: dict[str, np.ndarray], best: str, out: Path) -> None:
    y_all = df.label.to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    method = best.split("|")[1]
    for cfg in ["video-only", "audio-only", "behavior-only", "video+audio", "full"]:
        k = f"{cfg}|{method}"
        if k not in oof:
            continue
        ok = ~np.isnan(oof[k])
        fpr, tpr, _ = roc_curve(y_all[ok], oof[k][ok])
        ax.plot(fpr, tpr, lw=1.6, label=cfg)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Window-level ROC ({method} fusion)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_ablation.png", dpi=160)
    plt.close(fig)
    # reliability diagram for the best model
    p = oof[best]
    ok = ~np.isnan(p)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p[ok], bins) - 1, 0, 9)
    conf = [p[ok][idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    acc = [y_all[ok][idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.plot(conf, acc, "o-", color="#1f4e79")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(f"Reliability ({best.replace('|', ', ')})")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=160)
    plt.close(fig)


def _tables(res: dict[str, Any], out: Path) -> None:
    banner = (
        "> **SYNTHETIC DATA — pipeline smoke test only. These numbers are NOT results.**\n\n"
        if res["synthetic"]
        else ""
    )
    rows = sorted(res["ablation"].values(), key=lambda r: (r["config"], r["fuser"]))
    md = [
        banner + "| Modalities | Fuser | AUC | EER | F1@EER | TPR@1%FPR | TPR@5%FPR | ECE | Session AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tex = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Modalities & Fuser & AUC & EER & F1 & TPR@1\% & TPR@5\% & ECE & Sess.\ AUC \\",
        r"\midrule",
    ]
    for r in rows:
        w, s = r["window"], r["session"]
        sa = f"{s['auc']:.3f}" if s else "–"
        md.append(
            f"| {r['config']} | {r['fuser']} | {w['auc']:.3f} | {100 * w['eer']:.1f}% | {w['f1_at_eer']:.3f} | "
            f"{w['tpr_at_fpr1']:.3f} | {w['tpr_at_fpr5']:.3f} | {w['ece']:.3f} | {sa} |"
        )
        tex.append(
            f"{r['config']} & {r['fuser']} & {w['auc']:.3f} & {100 * w['eer']:.1f} & {w['f1_at_eer']:.3f} & "
            f"{w['tpr_at_fpr1']:.3f} & {w['tpr_at_fpr5']:.3f} & {w['ece']:.3f} & {sa} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    if res.get("per_attack"):
        pa = [banner + "| Attack | Windows | Recall @ 5% FPR (genuine FPR) |", "|---|---|---|"]
        for k, v in res["per_attack"].items():
            pa.append(f"| {k} | {v['n_windows']} | {v.get('recall_at_5fpr', v.get('fpr', 0)):.3f} |")
        (out / "per_attack.md").write_text("\n".join(pa) + "\n", encoding="utf-8")
    if res.get("fairness"):
        fl = [banner + "| Factor | Value | n | AUC | FPR @ global threshold |", "|---|---|---|---|---|"]
        for col, vals in res["fairness"].items():
            for v, e in vals.items():
                auc = f"{e['auc']:.3f}" if e.get("auc") is not None else "–"
                fpr = f"{e['fpr_at_global_thr']:.3f}" if e.get("fpr_at_global_thr") is not None else "–"
                fl.append(f"| {col} | {v} | {e['n']} | {auc} | {fpr} |")
        (out / "fairness.md").write_text("\n".join(fl) + "\n", encoding="utf-8")


def train_final(df: pd.DataFrame, mods: tuple[str, ...] = MODS, methods: list[str] | None = None) -> None:
    if df["synthetic"].any():
        raise SystemExit("refusing to ship fusion models trained on synthetic data")
    groups = df.participant_id.to_numpy()
    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    cal = np.isin(groups, rng.choice(ug, max(1, len(ug) // 5), replace=False))
    data = mask_modalities(df, mods)
    for method in methods or FUSERS:
        f = CalibratedFuser(
            build_fuser(method, mods), Calibrator("temperature")  # rank-preserving; isotonic tied scores (see paper)
        )
        f.fit(data[~cal], df.label[~cal].to_numpy(), data[cal], df.label[cal].to_numpy())
        f.save(models_dir() / "fusion" / f"{method}.pkl")
        log.info("saved fusion/%s.pkl", method)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["template", "validate", "extract", "evaluate", "train-final"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fusers", nargs="*", default=None)
    ap.add_argument(
        "--mods", default="full", help="ablation config name used by train-final (e.g. all-but-transcript)"
    )
    a = ap.parse_args(argv)
    if a.cmd == "template":
        print("template written to", write_template())
        return 0
    if a.cmd == "validate":
        probs = validate()
        print("\n".join(probs) if probs else "dataset OK")
        return 1 if probs else 0
    if a.synthetic:
        from trusthire.evaluation.synthetic import make_sessions

        df = make_sessions(48, 40, seed=7)
        df["participant_id"] = df.session_id.str.slice(0, 5)  # 1 participant per session in the synthetic set
        evaluate(df, REPO_ROOT / "results" / "SYNTHETIC_DEMO", synthetic=True, fusers=a.fusers)
        return 0
    df = extract(force=a.force)
    if a.cmd == "extract":
        print(f"{len(df)} windows from {df.session_id.nunique()} sessions -> {custom_root() / 'cache'}")
        return 0
    if a.cmd == "evaluate":
        evaluate(df, REPO_ROOT / "results" / "custom", synthetic=False, fusers=a.fusers)
    elif a.cmd == "train-final":
        train_final(df, ablation_configs()[a.mods], a.fusers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
