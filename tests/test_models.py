"""Module tests that run the real pretrained models on the real SyncNet example clip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import requires_models
from trusthire.common.types import Modality, Window
from trusthire.video.aiimg import onnx_paths


def _window(frames, audio, i0: int, idx: int = 0) -> Window:
    return Window(
        idx,
        i0 / 25,
        i0 / 25 + 3,
        frames[i0 : i0 + 75],
        [(i0 + k) / 25 for k in range(75)],
        audio[i0 * 640 : (i0 + 75) * 640],
    )


@requires_models
def test_face_detect_and_arcface(example_clip):
    from trusthire.common.face import FaceStack, interpolated_faces

    frames, _, _ = example_clip
    fs = FaceStack.get()
    f0 = fs.detector.detect(frames[0])
    f1 = fs.detector.detect(frames[200])
    assert len(f0) == 1 and f0[0].score > 0.6 and f0[0].kps.shape == (5, 2)
    sim = float(fs.arcface.embed(frames[0], f0[0]) @ fs.arcface.embed(frames[200], f1[0]))
    assert sim > 0.6  # same person across the clip
    cache = {"faces:0": f0, "faces:10": f1}
    track = interpolated_faces(cache, 11)
    assert track[5] is not None and np.allclose(track[5].bbox, (f0[0].bbox + f1[0].bbox) / 2)


@requires_models
def test_face_detect_none_on_blank():
    from trusthire.common.face import FaceStack

    assert FaceStack.get().detector.detect(np.zeros((240, 320, 3), np.uint8)) == []


@requires_models
@pytest.mark.parametrize("backbone", ["effnb4", "xception"])
def test_video_detector_on_real_clip(example_clip, backbone):
    from trusthire.video.infer import VideoDeepfakeDetector

    frames, audio, _ = example_clip
    out = VideoDeepfakeDetector(backbone, 3).process(_window(frames, audio, 0))
    assert out.module == Modality.VIDEO and 0 <= out.score <= 1 and out.confidence > 0.5
    assert len(out.details["frame_probs"]) == 3


@requires_models
def test_video_no_face_is_missing():
    from trusthire.video.infer import VideoDeepfakeDetector

    w = Window(
        0,
        0,
        3,
        [np.zeros((240, 320, 3), np.uint8)] * 10,
        [i / 25 for i in range(10)],
        np.zeros(48000, np.float32),
    )
    out = VideoDeepfakeDetector("effnb4", 3).process(w)
    assert out.confidence == 0 and out.details["missing"]


def test_temporal_heads_shapes():
    from trusthire.video.model import build_temporal_head

    for kind in ("bilstm", "transformer"):
        h = build_temporal_head(kind, 64)
        x = torch.randn(3, 8, 64)
        mask = torch.ones(3, 8, dtype=torch.bool)
        mask[0, 5:] = False
        assert h(x, mask).shape == (3,)


@requires_models
def test_onnx_parity_effnb4():
    import onnxruntime as ort

    from trusthire.common.registry import ensure
    from trusthire.video.infer import ensure_onnx
    from trusthire.video.model import build_frame_model

    m = build_frame_model("effnb4", ensure("video.effnb4"))
    x = torch.randn(2, 3, 256, 256)
    with torch.inference_mode():
        ref = m(x)[0].numpy()
    got = ort.InferenceSession(str(ensure_onnx("effnb4")), providers=["CPUExecutionProvider"]).run(
        None, {"image": x.numpy()}
    )[0]
    assert np.abs(ref - got).max() < 1e-2


@requires_models
def test_aasist_backend_probabilities():
    from trusthire.audio.infer import AASISTBackend

    p = AASISTBackend("aasist").p_spoof(
        (np.random.default_rng(0).standard_normal((2, 64600)) * 0.05).astype(np.float32)
    )
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_tile_and_preemphasis():
    from trusthire.audio.infer import preemphasis, tile

    x = np.arange(10, dtype=np.float32)
    assert len(tile(x, 25)) == 25 and tile(x, 25)[10] == 0
    assert len(tile(np.arange(100, dtype=np.float32), 64)) == 64
    y = preemphasis(np.ones(5, np.float32))
    assert y[0] == 1 and np.allclose(y[1:], 0.03)


@requires_models
def test_syncnet_tracks_audio_shift(example_clip):
    """Shifting the audio by k frames must move the estimated offset by -k (reference: offset 3 on this clip)."""
    from trusthire.avsync.infer import SyncNetScorer
    from trusthire.common.face import FaceStack, syncnet_crop

    frames, audio, _ = example_clip
    fs = FaceStack.get()
    crops = [syncnet_crop(f, fs.detector.detect(f)[0]) for f in frames[:150]]
    sc = SyncNetScorer(vshift=8, max_stacks=24)
    base, conf, _ = sc.offsets(crops, audio[: 150 * 640])
    shifted, _, _ = sc.offsets(crops, np.roll(audio, 4 * 640)[: 150 * 640])
    assert base == 3 and conf > 3
    assert shifted == base - 4


def test_sync_risk_monotone():
    from trusthire.avsync.infer import sync_risk

    assert sync_risk(0, 8.0) < 0.05
    assert sync_risk(0, 0.5) > 0.9
    assert sync_risk(8, 6.0) > sync_risk(2, 6.0)


@requires_models
def test_identity_detects_person_change(example_clip):
    from trusthire.identity.infer import Enrollment, IdentityMonitor, enroll_face_from_image

    frames, audio, _ = example_clip
    enr = Enrollment(face=enroll_face_from_image(frames[0]))
    mon = IdentityMonitor(enr)
    same = mon.face(_window(frames, audio, 30))
    assert same.score < 0.2 and same.details["sim_enrolled"] > 0.5
    # a different identity: use a random unit vector as the enrolled face
    other = np.random.default_rng(0).standard_normal(512)
    mon2 = IdentityMonitor(Enrollment(face=other / np.linalg.norm(other)))
    diff = mon2.face(_window(frames, audio, 30))
    assert diff.score > 0.8


@pytest.mark.skipif(
    not onnx_paths()[1].exists(),
    reason="AI-image detector not exported (run trusthire.video.aiimg.export)",
)
def test_aiimg_detector_real_clip_scores_real(example_clip):
    from trusthire.common.face import FaceStack, square_crop
    from trusthire.video.aiimg import AIImageDetector

    frames, _, _ = example_clip
    fs = FaceStack.get()
    crops = [square_crop(f, fs.detector.detect(f)[0], 1.3, 256) for f in (frames[0], frames[150])]
    p = AIImageDetector(quantized=True).p_fake(crops)
    assert p.shape == (2,) and (p < 0.5).all()  # genuine broadcast footage
