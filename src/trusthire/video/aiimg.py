"""AI-generated-face detector: SwinV2-L image classifier (haywoodsloan/ai-image-detector-deploy, Apache-2.0).

Chosen on the custom recordings (results/video/pretrained_on_custom.md): AUC 1.000 for AI-generated vs real-camera
videos, mean P(fake) 0.00 on real webcam footage (incl. re-compressed copies) vs 0.99 on text-to-video avatars, where
the FF++-trained EfficientNet-B4 was inverted (AUC 0.000 on face crops). Exported to ONNX and int8-quantised
(MatMul/Gemm weights) for CPU; the fp32 graph is used on GPU.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from trusthire.common.ort import make_session
from trusthire.common.registry import models_dir

REPO = "haywoodsloan/ai-image-detector-deploy"
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FAKE_INDEX = 0  # id2label {0: artificial, 1: real}


def onnx_paths() -> tuple[Path, Path]:
    d = models_dir() / "video" / "aiimg"
    return d / "aiimg_swinv2.onnx", d / "aiimg_swinv2.int8.onnx"


def export() -> Path:
    """Export the HF checkpoint to ONNX (+ int8). Parity-checked against PyTorch."""
    import torch
    from transformers import AutoModelForImageClassification

    from trusthire.common.export import export_onnx

    fp32, int8 = onnx_paths()
    if not fp32.exists():
        model = AutoModelForImageClassification.from_pretrained(
            REPO, cache_dir=str(models_dir() / "hf")
        ).eval()

        class Logits(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.m(pixel_values=x).logits  # type: ignore[no-any-return]

        export_onnx(
            Logits(), (torch.randn(2, 3, SIZE, SIZE),), fp32, ["image"], ["logits"], opset=17, atol=5e-3
        )
    if not int8.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]
        )
    return fp32


def preprocess(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    x = (
        np.stack(
            [
                cv2.cvtColor(cv2.resize(i, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
                for i in imgs_bgr
            ]
        ).astype(np.float32)
        / 255.0
    )
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2))


class AIImageDetector:
    def __init__(self, quantized: bool | None = None):
        import onnxruntime as ort

        export()
        fp32, int8 = onnx_paths()
        gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        self.path = fp32 if (gpu if quantized is None else not quantized) else int8
        self.sess = make_session(self.path)
        self._lock = threading.Lock()

    def p_fake(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        with self._lock:
            logits = self.sess.run(None, {"image": preprocess(imgs_bgr)})[0]
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return np.asarray(p[:, FAKE_INDEX])
