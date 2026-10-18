from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRUSTHIRE_SKIP_WARMUP", "1")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _have(rel: str) -> bool:
    from trusthire.common.registry import models_dir

    return (models_dir() / rel).exists()


requires_models = pytest.mark.skipif(
    not (_have("identity/buffalo_l") and _have("video/effnb4_best.pth") and _have("avsync/syncnet_v2.model")),
    reason="pretrained weights not downloaded (run `trusthire download-models`)",
)


@pytest.fixture(scope="session")
def example_clip():
    """Real talking-head clip shipped with SyncNet (12.7 s, 25 fps, 16 kHz audio)."""
    from trusthire.common.media import iter_frames, load_audio
    from trusthire.common.registry import ensure

    p = ensure("avsync.example")
    frames = [f for _, f in iter_frames(p, 25)]
    return frames, load_audio(p), p


@pytest.fixture
def rng():
    return np.random.default_rng(0)
