"""Fill the paper-style report (docs/report/paper.md) from the result files under results/.

Every number in the generated report is read from a JSON/Markdown artefact produced by a real run. Sections
whose data does not exist yet are rendered as explicit "pending — run X" notes, never with placeholder numbers.
Synthetic smoke-test results are never included.

    python -m trusthire.evaluation.report
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trusthire.common.settings import REPO_ROOT

RES = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "report" / "paper.md"


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _md(p: Path) -> str | None:
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _pending(what: str, cmd: str) -> str:
    return f"> *Pending — {what}. Run `{cmd}`; this section is filled automatically from the results.*"


def audio_section() -> str:
    bench = _load(RES / "audio" / "benchmark.json")
    lat = _load(RES / "audio" / "latency.json") or {}
    if not bench:
        return _pending("voice anti-spoofing benchmark", "make bench")
    clean = [r for r in bench.values() if r["condition"] == "clean"]
    rows = [
        "| Model | Test set | n | AUC [95% CI] | EER % [95% CI] | TPR@1%FPR | CPU ms/utt |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(clean, key=lambda r: (r["dataset"], r["eer"])):
        rows.append(
            f"| {r['backend']} | {r['dataset']} | {r['n']} | {r['auc']:.4f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] | "
            f"{100 * r['eer']:.2f} [{100 * r['eer_ci'][0]:.1f}, {100 * r['eer_ci'][1]:.1f}] | {r['tpr_at_fpr1']:.3f} | "
            f"{lat.get(r['backend'], float('nan')):.0f} |"
        )
    rob = [r for r in bench.values() if r["condition"] != "clean"]
    txt = "\n".join(rows)
    cond = lat.get("_conditions") or {}
    if cond.get("on_battery"):
        txt += (
            f"\n\n*CPU ms/utt measured {cond.get('measured')} on battery power (CPU capped at "
            f"{cond.get('cpu_mhz'):.0f} MHz) — about 3x slower than on mains power; re-measure plugged in with "
            "`python -m trusthire.audio.eval --measure-latency`.*"
        )
    if rob:
        conds = sorted({r["condition"] for r in bench.values() if r["dataset"] == "itw"})
        bks = sorted(
            {r["backend"] for r in bench.values() if r["dataset"] == "itw" and r["condition"] != "clean"}
        )
        hdr = "| Condition | " + " | ".join(f"{b} EER %" for b in bks) + " |"
        lines = [hdr, "|---|" + "---|" * len(bks)]
        for c in conds:
            vals = [bench.get(f"itw/{c}/{b}") for b in bks]
            lines.append(f"| {c} | " + " | ".join(f"{100 * v['eer']:.2f}" if v else "–" for v in vals) + " |")
        txt += (
            "\n\n**Robustness (In-the-Wild subset, same utterances under each degradation):**\n\n"
            + "\n".join(lines)
        )
    return txt


def video_section() -> str:
    t = _md(RES / "video" / "benchmark.md")
    custom = _md(RES / "video" / "pretrained_on_custom.md")
    if custom:
        intro = (
            "**Detector selection on the custom recordings** (AI-generated vs real-camera video; the deployed "
            "detector is the int8 ONNX export of the top row):\n\n"
        )
        t = (t + "\n\n" if t else "") + intro + custom
    return t or _pending(
        "FF++ / Celeb-DF / DFDC video benchmark (datasets need access approval, see docs/DATASETS.md)",
        "python -m trusthire.video.eval --crops <crops.csv> --test-datasets ffpp celebdf",
    )


def identity_section() -> str:
    parts = []
    f = _load(RES / "identity" / "face_lfw.json")
    if f:
        c = f["at_configured_threshold"]
        parts.append(
            f"**Face (LFW test, {f['n_pairs']} pairs, SCRFD→ArcFace R50):** AUC {f['auc']:.4f}, EER {100 * f['eer']:.2f}% "
            f"(threshold cos {f['eer_threshold_sim']:.3f}), accuracy {100 * f['accuracy_at_eer_thr']:.2f}%. "
            f"At the deployed threshold cos={c['threshold_sim']:.2f}: same-person false flags "
            f"{100 * c['false_flag_rate_same_person']:.2f}%, impostors missed {100 * c['missed_impostor_rate']:.2f}%."
        )
    else:
        parts.append(_pending("LFW face verification", "python -m trusthire.identity.eval face"))
    s = _load(RES / "identity" / "speaker_librispeech.json")
    if s:
        c = s["at_configured_threshold"]
        parts.append(
            f"**Speaker (LibriSpeech test-clean, {s['n_trials']} trials, 4 s, ECAPA-TDNN):** AUC {s['auc']:.4f}, "
            f"EER {100 * s['eer']:.2f}%. At the deployed threshold cos={c['threshold_sim']:.2f}: same-speaker false flags "
            f"{100 * c['false_flag_rate_same_person']:.2f}%, impostors missed {100 * c['missed_impostor_rate']:.2f}%."
        )
    else:
        parts.append(_pending("speaker verification", "python -m trusthire.identity.eval speaker"))
    return "\n\n".join(parts)


def avsync_section() -> str:
    d = _load(RES / "avsync" / "desync_study.json")
    if not d:
        return _pending("AV desync study", "python -m trusthire.avsync.eval")
    lines = [
        "| Audio shift (frames) | Offset accuracy (±1 frame) | Mean SyncNet conf | Mean risk |",
        "|---|---|---|---|",
    ]
    for k, v in sorted(d["per_shift"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {k} ({int(k) * 40} ms) | {100 * v['offset_acc_1f']:.1f}% | {v['mean_conf']:.2f} | {v['mean_risk']:.3f} |"
        )
    txt = f"Videos: {len(d['videos'])}; windows: {d['n_windows']}.\n\n" + "\n".join(lines)
    if d.get("detect_desync_ge4"):
        r = d["detect_desync_ge4"]
        txt += f"\n\nDetecting |shift| ≥ 4 frames (160 ms) vs in-sync: AUC {r['auc']:.3f}, EER {100 * r['eer']:.1f}%."
    return txt


def latency_section() -> str:
    lat = _load(RES / "latency" / "latency_cpu.json")
    gpu = _load(RES / "latency" / "latency_gpu.json")
    parts = []
    for name, d in (("CPU", lat), ("GPU", gpu)):
        if not d:
            parts.append(
                _pending(
                    f"{name} latency",
                    "python -m trusthire.evaluation.latency"
                    + (" (TRUSTHIRE_PROFILE=gpu)" if name == "GPU" else ""),
                )
            )
            continue
        mods = ", ".join(
            f"{k} {v:.0f}" for k, v in sorted(d["modules_ms_p50"].items(), key=lambda kv: -kv[1])
        )
        battery = ", on battery" if d.get("on_battery") else ""
        gpu_mem = "" if d.get("gpu_mem_mb") is None else f", GPU memory {d['gpu_mem_mb']:.0f} MB"
        parts.append(
            f"**{name}** ({d['cpu']}, {d['logical_cpus']} threads{battery}): "
            f"window p50 **{d['total_ms_p50']:.0f} ms**, p95 {d['total_ms_p95']:.0f} ms for a {d['window_s']:.0f} s window "
            f"(hop {d['hop_s'] * 1000:.0f} ms; real-time factor {d['real_time_factor']:.2f}); "
            f"peak RSS {d['peak_rss_mb']:.0f} MB{gpu_mem}. Per-module p50 (ms, concurrent): {mods}."
        )
    return "\n\n".join(parts)


def custom_section() -> tuple[str, str, str]:
    r = _load(RES / "custom" / "results.json")
    if not r or r.get("synthetic"):
        p = _pending(
            "custom interview test set (30–50 recorded sessions; protocol in docs/DATA_COLLECTION.md)",
            "make eval-custom",
        )
        return p, p, p
    abl = _md(RES / "custom" / "ablation.md") or ""
    head = (
        f"{r['n_sessions']} sessions, {r['n_participants']} participants, {r['n_windows']} windows; "
        f"leave-participants-out CV. Best full-fusion model: `{r['best_full']}`, ECE {r['calibration']['ece']:.3f}.\n\n"
    )
    pa = _md(RES / "custom" / "per_attack.md") or ""
    fair = _md(RES / "custom" / "fairness.md") or ""
    return head + abl + "\n\n![ROC](../../results/custom/roc_ablation.png)", pa, fair


def build() -> Path:
    abl, per_attack, fairness = custom_section()
    tmpl = (REPO_ROOT / "docs" / "report" / "paper_template.md").read_text(encoding="utf-8")
    filled = (
        tmpl.replace("{{GENERATED}}", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
        .replace("{{AUDIO}}", audio_section())
        .replace("{{VIDEO}}", video_section())
        .replace("{{IDENTITY}}", identity_section())
        .replace("{{AVSYNC}}", avsync_section())
        .replace("{{LATENCY}}", latency_section())
        .replace("{{ABLATION}}", abl)
        .replace("{{PER_ATTACK}}", per_attack)
        .replace("{{FAIRNESS}}", fairness)
    )
    OUT.write_text(filled, encoding="utf-8")
    print("wrote", OUT)
    return OUT


if __name__ == "__main__":
    build()
