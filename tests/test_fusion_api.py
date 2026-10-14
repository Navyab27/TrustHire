"""Fusion models (on SYNTHETIC data — tests only), calibration, live fuser, report and API smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trusthire.common.types import Modality, ModuleOutput
from trusthire.evaluation.metrics import ece
from trusthire.evaluation.synthetic import make_sessions
from trusthire.fusion.features import mask_modalities, row_from_outputs
from trusthire.fusion.live import LiveFuser, session_trust
from trusthire.fusion.models import CalibratedFuser, Calibrator, WeightedLateFusion, build_fuser


@pytest.fixture(scope="module")
def syn():
    df = make_sessions(36, 30, seed=1)
    sess = df.session_id.unique()
    tr = df[df.session_id.isin(sess[::2])].reset_index(drop=True)
    te = df[df.session_id.isin(sess[1::2])].reset_index(drop=True)
    return tr, te


@pytest.mark.parametrize("method", ["weighted", "gbm", "xgboost", "attention"])
def test_fusers_learn(syn, method):
    from sklearn.metrics import roc_auc_score

    tr, te = syn
    f = CalibratedFuser(build_fuser(method, seed=0)).fit(tr, tr.label.to_numpy(), tr, tr.label.to_numpy())
    p = f.predict_proba(te)
    assert p.shape == (len(te),) and np.isfinite(p).all()
    assert roc_auc_score(te.label, p) > 0.80  # "learns" (chance = 0.5); exact value varies with column sampling
    c = f.contributions(te.head(5))
    assert c.shape[0] == 5


def test_missing_modalities_handled(syn):
    tr, te = syn
    f = CalibratedFuser(build_fuser("attention", seed=0)).fit(tr, tr.label.to_numpy())
    only_audio = mask_modalities(te, ["voice_spoof", "voice_identity"])
    p = f.predict_proba(only_audio)
    assert np.isfinite(p).all()


def test_prior_fusion_not_overconfident_but_red_flag_dominates():
    f = WeightedLateFusion()
    genuine = {
        m.value: ModuleOutput(module=m, score=0.1, confidence=0.9, t_start=0, t_end=3) for m in Modality
    }
    p_gen = 1 / (1 + np.exp(-f.predict_logit(pd.DataFrame([row_from_outputs(genuine)]))[0]))
    flagged = dict(genuine)
    flagged["face_identity"] = ModuleOutput(
        module=Modality.FACE_ID, score=0.98, confidence=0.9, t_start=0, t_end=3
    )
    p_flag = 1 / (1 + np.exp(-f.predict_logit(pd.DataFrame([row_from_outputs(flagged)]))[0]))
    assert 0.03 < p_gen < 0.25  # many agreeing modules must not compound into ~0
    assert p_flag > 0.65  # a single confident mismatch must raise an alert


def test_temperature_scaling_reduces_ece(rng):
    y = rng.integers(0, 2, 4000)
    logits = (y * 2 - 1) * 1.0 + rng.normal(0, 1, 4000)
    over = logits * 4  # overconfident model
    cal = Calibrator("temperature").fit(over, y)
    assert pytest.approx(2.0, abs=0.25) == cal.T  # true LLR is 2x, so 4x-overconfident logits need T=2
    assert ece(y, cal(over)) < ece(y, 1 / (1 + np.exp(-over)))
    iso = Calibrator("isotonic").fit(over, y)
    assert ece(y, iso(over)) < 0.05


def test_live_fuser_and_trust():
    lf = LiveFuser("weighted")
    outs = {
        "face_identity": ModuleOutput(
            module=Modality.FACE_ID,
            score=0.97,
            confidence=0.9,
            t_start=0,
            t_end=3,
            details={"sim_enrolled": 0.05},
        )
    }
    p = lf.fuse(outs, 0, 3)
    assert p.risk > 0.65 and any("enrolled" in a for a in p.alerts)
    tr = session_trust([p, p])
    assert tr["trust_score"] < 35


def test_report_pdf(tmp_path):
    from trusthire.explain import report, visual

    pts = [
        {"t": i * 1.5 + 1.5, "t_start": i * 1.5, "t_end": i * 1.5 + 3, "risk": r, "alerts": []}
        for i, r in enumerate([0.1, 0.2, 0.8, 0.3])
    ]
    figs = {
        "timeline": visual.timeline_figure(pts),
        "contributions": visual.contributions_figure({"a": 0.5, "b": -0.2}, "x"),
    }
    sess = {
        "id": "t",
        "candidate": "Test <Candidate>",
        "duration_s": 7.5,
        "trust": {"trust_score": 55.0},
        "module_stats": {"video_deepfake": {"mean": 0.3, "max": 0.8, "coverage": 1.0, "flagged": 1}},
        "alerts": [{"t": 4.5, "severity": "critical", "module": "video_deepfake", "message": "x & y"}],
        "answers": [{"t_start": 0, "t_end": 5, "text": "hello", "features": {}, "scripted_score": 0.2}],
    }
    out = report.build_pdf(sess, "Summary line.", "template", figs, [], tmp_path / "r.pdf")
    assert out.exists() and out.stat().st_size > 5000


def test_template_summary():
    from trusthire.explain.llm import build_evidence, summarize

    ev = build_evidence(
        {"trust_score": 80, "session_risk": 0.2, "peak_risk": 0.5}, [], {"video_deepfake": 0.2}, [], 60
    )
    text, prov = summarize(ev, provider="template")
    assert prov == "template" and "80" in text


def test_api_smoke(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from trusthire.api import app as appmod
    from trusthire.pipeline.session import SessionStore

    monkeypatch.setattr(appmod, "STORE", SessionStore(tmp_path))
    with TestClient(appmod.app) as c:
        assert c.get("/api/health").json()["ok"]
        s = c.post("/api/sessions", json={"candidate": "Ada"}).json()
        assert s["candidate"] == "Ada" and s["status"] == "created"
        assert [x["id"] for x in c.get("/api/sessions").json()] == [s["id"]]
        d = c.get(f"/api/sessions/{s['id']}").json()
        assert d["points"] == [] and d["alerts"] == []
        assert c.get("/api/sessions/nope").status_code == 404
        r = c.post(
            f"/api/sessions/{s['id']}/resume",
            files={"file": ("cv.txt", b"Built a Rust compiler at Acme Corp in 2022.")},
        )
        assert r.status_code == 200 and r.json()["chunks"] >= 1
        assert c.post(f"/api/sessions/{s['id']}/question", json={}).status_code == 409
        assert c.delete(f"/api/sessions/{s['id']}").json()["deleted"]


def test_relative_sync_baseline():
    """Constant A/V skew (e.g. Bluetooth headset) is tolerated; a sudden change or an extreme skew is not."""
    from trusthire.avsync.infer import sync_risk
    from trusthire.pipeline.analyzer import SessionAnalyzer

    a = SessionAnalyzer.__new__(SessionAnalyzer)  # only the sync state is needed
    a._sync_offsets = []

    def mk(off: int, conf: float = 7.0) -> ModuleOutput:
        return ModuleOutput(
            module=Modality.AVSYNC,
            score=sync_risk(off, conf),
            confidence=1.0,
            t_start=0,
            t_end=3,
            details={"offset_frames": off, "sync_conf": conf},
        )

    assert mk(6).score > 0.7  # absolute rule alone would flag a constant 240 ms skew
    for _ in range(3):
        a._relative_sync(mk(6))
    assert a._relative_sync(mk(6)).score < 0.1
    assert a._relative_sync(mk(1)).score >= 0.45  # 5-frame jump vs baseline
    assert a._relative_sync(mk(13)).score > 0.95  # > 320 ms is suspicious regardless of baseline
