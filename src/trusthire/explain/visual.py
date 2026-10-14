"""Visual explanations: Grad-CAM on flagged face crops, occlusion saliency over audio spectrograms, plots."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from trusthire.common.registry import ensure
from trusthire.video.infer import preprocess
from trusthire.video.model import INPUT_SIZE, build_frame_model

_models: dict[str, nn.Module] = {}
_lock = threading.Lock()


class _LogitsOnly(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)[0]  # type: ignore[no-any-return]


def gradcam(crop_bgr: np.ndarray, backbone: str = "effnb4") -> tuple[np.ndarray, float]:
    """Grad-CAM (Selvaraju et al.) for the 'fake' logit. Returns (overlay BGR, P(fake))."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if backbone == "aiimg":
        return _gradcam_swin(crop_bgr)
    with _lock:
        if backbone not in _models:
            _models[backbone] = build_frame_model(backbone, ensure(f"video.{backbone}"))
        model = _models[backbone]
    size = INPUT_SIZE.get(backbone, 224)
    x = torch.from_numpy(preprocess([crop_bgr], size))
    wrapped = _LogitsOnly(model).eval()
    with torch.no_grad():
        p_fake = float(torch.softmax(wrapped(x), -1)[0, 1])
    with GradCAM(model=wrapped, target_layers=[model.gradcam_layer]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    img = cv2.resize(crop_bgr, (size, size))
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, color, 0.45, 0), p_fake


def _gradcam_swin(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Grad-CAM for the SwinV2 AI-image detector: last stage's tokens reshaped to an 8x8 map."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from transformers import AutoModelForImageClassification

    from trusthire.common.registry import models_dir
    from trusthire.video.aiimg import FAKE_INDEX, REPO, SIZE
    from trusthire.video.aiimg import preprocess as aipre

    with _lock:
        if "aiimg" not in _models:
            m = AutoModelForImageClassification.from_pretrained(
                REPO, cache_dir=str(models_dir() / "hf")
            ).eval()

            class _Logits(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.m = m

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.m(pixel_values=x).logits  # type: ignore[no-any-return]

            _models["aiimg"] = _Logits().eval()
        model = _models["aiimg"]
    x = torch.from_numpy(aipre([crop_bgr]))
    with torch.no_grad():
        p_fake = float(torch.softmax(model(x), -1)[0, FAKE_INDEX])
    # end of stage 3 (16x16 tokens): class-discriminative and finer than the 8x8 last stage; the last
    # block's post-norm output gave an all-zero map (checked against GradCAM++/EigenCAM)
    layer = model.m.swinv2.encoder.layers[-2].blocks[-1].layernorm_after  # type: ignore[union-attr,index]

    def reshape(t: torch.Tensor) -> torch.Tensor:  # (B, HW, C) -> (B, C, H, W)
        side = round(t.shape[1] ** 0.5)
        return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)

    with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(FAKE_INDEX)])[0]
    img = cv2.resize(crop_bgr, (SIZE, SIZE))
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, color, 0.45, 0), p_fake


def audio_occlusion(
    audio: np.ndarray, p_spoof_fn: Callable[[np.ndarray], np.ndarray], win_s: float = 0.25, sr: int = 16000
) -> tuple[np.ndarray, float]:
    """Model-agnostic saliency: silence each `win_s` slice and measure the drop in P(spoof).
    Positive values = the slice made the model think 'spoof'."""
    from trusthire.audio.infer import tile

    x = tile(audio)
    base = float(p_spoof_fn(x[None])[0])
    step = int(win_s * sr)
    batch = []
    for s in range(0, len(x), step):
        y = x.copy()
        y[s : s + step] = 0.0
        batch.append(y)
    probs = np.concatenate([p_spoof_fn(np.stack(batch[i : i + 8])) for i in range(0, len(batch), 8)])
    return base - probs, base


def saliency_figure(
    audio: np.ndarray, saliency: np.ndarray, p: float, title: str, win_s: float = 0.25, sr: int = 16000
) -> bytes:
    import librosa
    import librosa.display

    from trusthire.audio.infer import tile

    x = tile(audio)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=sr, n_mels=64, hop_length=160), ref=np.max)
    fig, ax = plt.subplots(2, 1, figsize=(7, 3.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    librosa.display.specshow(S, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax[0], cmap="magma")
    ax[0].set_title(f"{title}  (P(spoof)={p:.2f})", fontsize=9)
    t = np.arange(len(saliency)) * win_s
    ax[1].bar(t + win_s / 2, saliency, width=win_s * 0.9, color=np.where(saliency > 0, "#c0392b", "#7f8c8d"))
    ax[1].set_ylabel("Δ P(spoof)", fontsize=8)
    ax[1].set_xlabel("time (s)", fontsize=8)
    for i, v in enumerate(saliency):
        if v > 0.05:
            ax[0].axvspan(i * win_s, (i + 1) * win_s, color="cyan", alpha=0.15)
    fig.tight_layout()
    return _png(fig)


def timeline_figure(points: list[dict], threshold: float = 0.65) -> bytes:
    t = [p["t"] for p in points]
    r = [p["risk"] for p in points]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(t, r, color="#1f4e79", lw=1.6, label="fused risk")
    ax.fill_between(t, r, 0, color="#1f4e79", alpha=0.12)
    ax.axhline(threshold, color="#c0392b", ls="--", lw=1, label=f"alert threshold {threshold}")
    for p in points:
        if p.get("alerts"):
            ax.axvspan(p["t_start"], p["t_end"], color="#e74c3c", alpha=0.08, lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("risk")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def contributions_figure(contrib: dict[str, float], title: str) -> bytes:
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]))[-10:]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 0.8))
    ax.barh(
        [k for k, _ in items],
        [v for _, v in items],
        color=["#c0392b" if v > 0 else "#27ae60" for _, v in items],
    )
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _png(fig)


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
