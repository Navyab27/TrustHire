"""Per-window video deepfake inference: face crops -> frame CNN (ONNX) -> temporal aggregation."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from trusthire.common.face import Face, anchor_indices, detect_cached, square_crop
from trusthire.common.ort import make_session
from trusthire.common.registry import ensure, models_dir
from trusthire.common.types import Modality, ModuleOutput, Window, missing
from trusthire.video.model import INPUT_SIZE, MEAN, STD, build_frame_model, build_temporal_head

log = logging.getLogger(__name__)


def preprocess(crops_bgr: list[np.ndarray], size: int) -> np.ndarray:
    x: np.ndarray = np.stack(
        [cv2.cvtColor(cv2.resize(c, (size, size)), cv2.COLOR_BGR2RGB) for c in crops_bgr]
    ).astype(np.float32)
    x = (x / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def onnx_path(backbone: str) -> Path:
    return models_dir() / "video" / f"{backbone}.onnx"


def ensure_onnx(backbone: str) -> Path:
    p = onnx_path(backbone)
    if not p.exists():
        from trusthire.common.export import export_onnx
        from trusthire.common.registry import BY_KEY

        # registry checkpoints (DeepfakeBench) or a locally trained one from `trusthire.video.train`
        trained = models_dir() / "video" / f"{backbone.split('.')[0]}_trained.pth"
        w = (
            trained
            if trained.exists()
            else ensure(f"video.{backbone}")
            if f"video.{backbone}" in BY_KEY
            else None
        )
        if w is None:
            raise FileNotFoundError(
                f"no weights for video backbone {backbone!r}; train it with trusthire.video.train"
            )
        model = build_frame_model(backbone, w, pretrained_timm=False)
        size = INPUT_SIZE.get(backbone, 224)
        export_onnx(model, (torch.randn(2, 3, size, size),), p, ["image"], ["logits", "embedding"])
    return p


def pick_face(faces: list[Face]) -> Face | None:
    return faces[0] if faces else None  # largest (detector sorts by size)


class VideoDeepfakeDetector:
    def __init__(self, backbone: str = "effnb4", frames_per_window: int = 5, temporal: str | None = None):
        self.backbone = backbone
        self.k = frames_per_window
        self._lock = threading.Lock()
        self.temporal: torch.nn.Module | None = None
        self.temporal_kind = "topk-mean"
        self.aiimg = None
        if backbone == "aiimg":  # SwinV2-L AI-generated-image detector (see video/aiimg.py)
            from trusthire.video.aiimg import SIZE, AIImageDetector

            self.aiimg = AIImageDetector()
            self.size = SIZE
            return
        self.size = INPUT_SIZE.get(backbone, 224)
        self.sess = make_session(ensure_onnx(backbone))
        head_path = models_dir() / "video" / f"temporal_{temporal or 'transformer'}_{backbone}.pt"
        if head_path.exists():
            ck = torch.load(head_path, map_location="cpu", weights_only=False)
            self.temporal = build_temporal_head(ck["kind"], ck["in_dim"])
            self.temporal.load_state_dict(ck["state_dict"])
            self.temporal.eval()
            self.temporal_kind = ck["kind"]

    def frame_scores(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        if self.aiimg is not None:
            return self.aiimg.p_fake(crops), None
        x = preprocess(crops, self.size)
        with self._lock:
            logits, emb = self.sess.run(None, {"image": x})
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return p[:, 1], emb

    def process(self, w: Window) -> ModuleOutput:
        t0 = time.perf_counter()
        if not w.frames:
            return missing(Modality.VIDEO, w.t_start, w.t_end, "no video")
        anchors = anchor_indices(len(w.frames))
        # one crop -> the window's middle anchor; otherwise k evenly spaced (subset of the shared anchors)
        idxs = [anchors[len(anchors) // 2]] if self.k == 1 else anchor_indices(len(w.frames), self.k)
        crops, sizes, used = [], [], []
        for i in idxs:
            f = pick_face(detect_cached(w.cache, int(i), w.frames[i]))
            if f is None:
                continue
            crops.append(square_crop(w.frames[i], f, 1.3, self.size))
            sizes.append(f.size)
            used.append(int(i))
        if not crops:
            return missing(Modality.VIDEO, w.t_start, w.t_end, "no face")
        probs, emb = self.frame_scores(crops)
        if self.temporal is not None and emb is not None:
            with torch.inference_mode():
                score = float(torch.sigmoid(self.temporal(torch.from_numpy(emb)[None])).item())
        else:
            k = max(1, len(probs) // 2)
            score = float(np.sort(probs)[-k:].mean())  # top-half mean: robust to a few clean frames
        coverage = len(crops) / len(idxs)
        quality = float(np.clip(np.median(sizes) / 112.0, 0.2, 1.0))  # tiny faces are unreliable
        conf = float(np.clip(coverage * quality * (1 - 0.5 * np.std(probs)), 0.0, 1.0))
        w.cache["video_crops"] = list(zip(used, crops, probs.tolist(), strict=True))
        return ModuleOutput(
            module=Modality.VIDEO,
            score=score,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={
                "frame_probs": [round(float(p), 4) for p in probs],
                "backbone": self.backbone,
                "temporal": self.temporal_kind,
                "face_px": round(float(np.median(sizes)), 1),
            },
        )
