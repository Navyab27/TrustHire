"""Runtime settings (env vars prefixed TRUSTHIRE_, or .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from trusthire.common.registry import models_dir

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRUSTHIRE_", env_file=".env", extra="ignore")

    profile: Literal["cpu", "gpu"] = "cpu"
    models_dir: Path = models_dir()
    data_dir: Path = REPO_ROOT / "data"
    sessions_dir: Path = REPO_ROOT / "data" / "sessions"

    # chunker
    window_s: float = 3.0
    hop_s: float = 1.5  # 50 % overlap
    analysis_fps: float = 25.0  # SyncNet needs 25 fps

    # per-module model choices (overridable)
    # aiimg = SwinV2-L AI-generated-image detector: AUC 1.000 vs 0.000 for effnb4 on the custom recordings
    video_backbone: Literal["aiimg", "effnb4", "xception"] = "aiimg"
    video_frames_per_window: int = 3  # subset of the 5 shared face anchors (gpu profile: 8)
    audio_model: Literal[
        "aasist", "aasist_l", "spectra0", "spectra0_int8", "spectra0_ft", "spectra0_ft_int8"
    ] = "spectra0_int8"  # see results/audio
    whisper_model: str = "small"  # gpu profile: large-v3-turbo
    whisper_compute_type: str = "int8"
    face_det_size: int = 320
    yolo_model: str = "yolo26n.pt"

    # thresholds
    face_match_threshold: float = 0.30  # ArcFace cosine; below → different person
    voice_match_threshold: float = 0.25  # ECAPA cosine
    alert_risk_threshold: float = 0.65

    # Module scheduling: (stride, offset) in windows. With 50 % overlap, stride 2 still tiles the whole
    # timeline with contiguous 3 s windows (0, 2, 4, ...), so no evidence is lost — only redundant overlap.
    # Heavy modules are staggered across even/odd windows to balance CPU load. GPU profile: all (1, 0).
    module_schedule: dict[str, tuple[int, int]] = {
        "video_deepfake": (2, 0),  # SwinV2 detector is ~0.5 s/crop on CPU int8
        "voice_spoof": (2, 0),
        "face_identity": (2, 1),
        "behavior": (1, 0),
        "av_sync": (2, 1),
        "voice_identity": (2, 1),
        "objects": (2, 1),
    }

    # fusion
    # gbm = LightGBM trained on the custom recordings (all modules except transcript); falls back to the
    # documented prior (weighted) when models/fusion/gbm.pkl is absent
    fusion_method: Literal["weighted", "gbm", "attention"] = "gbm"

    # LLM summary
    llm_provider: Literal["anthropic", "ollama", "none"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    ollama_model: str = "llama3.2"

    # infra
    redis_url: str = "redis://localhost:6379/0"
    use_celery: bool = False
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    enable_transcript: bool = True

    def model_path(self, rel: str) -> Path:
        return self.models_dir / rel


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.profile == "gpu":
        s.module_schedule = {k: (1, 0) for k in s.module_schedule}
        s.video_frames_per_window = max(s.video_frames_per_window, 8)
        if s.whisper_model == "small":
            s.whisper_model = "large-v3-turbo"
            s.whisper_compute_type = "float16"
    s.sessions_dir.mkdir(parents=True, exist_ok=True)
    return s
