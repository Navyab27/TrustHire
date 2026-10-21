"""Grad-CAM panel for the paper. Uses AI-generated avatars only (no real participant's face is published)."""

from pathlib import Path

import cv2
import numpy as np

from trusthire.evaluation.dataset import custom_root
from trusthire.explain.visual import gradcam
from trusthire.video.bench_pretrained import sample_frames

OUT = Path(__file__).resolve().parent / "figures"
tiles = []
for sid in ("aivideos_aivideo1", "aivideos_aivideo3"):
    crop = sample_frames(str(custom_root() / "recordings" / f"{sid}.mp4"), 3)[1][1]
    overlay, p = gradcam(crop, "aiimg")
    img = cv2.resize(crop, (256, 256))
    for t in (img, overlay):
        t = t.copy()
        tiles.append(t)
    cv2.putText(tiles[-1], f"P(fake)={p:.2f}", (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
sep = np.full((256, 6, 3), 255, np.uint8)
row = np.hstack([tiles[0], sep, tiles[1], sep, sep, tiles[2], sep, tiles[3]])
cv2.imwrite(str(OUT / "gradcam.png"), row)
print("wrote", OUT / "gradcam.png", row.shape)
