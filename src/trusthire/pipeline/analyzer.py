"""Session analyzer: runs every branch on each window in parallel and fuses the results.

Window ─┬─ video deepfake ────┐
        ├─ voice spoof ───────┤
        ├─ AV sync ───────────┤
        ├─ face identity ─────┼─► fusion ─► FusedPoint (+ alerts) ─► callbacks (WebSocket / storage)
        ├─ voice identity ────┤
        ├─ behaviour/liveness/objects
        └─ transcript (async; updates earlier points when an answer has been analysed)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from trusthire.common.face import anchor_indices, detect_cached
from trusthire.common.settings import Settings, get_settings
from trusthire.common.types import FusedPoint, Modality, ModuleOutput, Window
from trusthire.fusion.live import LiveFuser
from trusthire.identity.infer import Enrollment, IdentityMonitor

log = logging.getLogger(__name__)
CTX_S = 64600 / 16000  # audio context for anti-spoofing / speaker models


class SharedModels:
    """Heavy stateless models, loaded once per process and shared by all sessions."""

    _lock = threading.Lock()
    _inst: SharedModels | None = None

    def __init__(self, s: Settings):
        from trusthire.audio.infer import VoiceSpoofDetector
        from trusthire.avsync.infer import AVSyncDetector
        from trusthire.common.face import FaceStack
        from trusthire.video.infer import VideoDeepfakeDetector

        t = time.perf_counter()
        FaceStack.get()
        # the SwinV2 detector costs ~0.5 s/crop on CPU int8 -> one crop per window on the CPU profile
        k = 1 if (s.video_backbone == "aiimg" and s.profile == "cpu") else s.video_frames_per_window
        self.video = VideoDeepfakeDetector(s.video_backbone, k)
        self.audio = VoiceSpoofDetector(s.audio_model)
        self.avsync = AVSyncDetector(max_stacks=8 if s.profile == "cpu" else 32)
        log.info("shared models ready in %.1fs", time.perf_counter() - t)

    @classmethod
    def get(cls, s: Settings | None = None) -> SharedModels:
        with cls._lock:
            if cls._inst is None:
                cls._inst = SharedModels(s or get_settings())
            return cls._inst


@dataclass
class AnalyzerConfig:
    enabled: set[str] = field(default_factory=lambda: {m.value for m in Modality})
    transcript: bool = True
    synchronous_transcript: bool = False


class SessionAnalyzer:
    def __init__(
        self,
        enrollment: Enrollment | None = None,
        resume_text: str | None = None,
        on_point: Callable[[FusedPoint], None] | None = None,
        on_update: Callable[[int, FusedPoint], None] | None = None,
        on_transcript: Callable[[object], None] | None = None,
        config: AnalyzerConfig | None = None,
        settings: Settings | None = None,
    ):
        from trusthire.behavior.infer import BehaviorMonitor

        self.s = settings or get_settings()
        self.cfg = config or AnalyzerConfig()
        self.models = SharedModels.get(self.s)
        self.identity = IdentityMonitor(
            enrollment or Enrollment(), self.s.face_match_threshold, self.s.voice_match_threshold
        )
        self.behavior = BehaviorMonitor(landmark_stride=2 if self.s.profile == "gpu" else 3)
        self.fuser = LiveFuser(self.s.fusion_method, self.s.alert_risk_threshold)
        self.points: list[FusedPoint] = []
        self.outputs: list[dict[str, ModuleOutput]] = []
        self.on_point = on_point
        self.on_update = on_update
        self.on_transcript = on_transcript
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="branch")
        self._audio_buf = np.zeros(0, np.float32)
        self._audio_t0 = 0.0
        self._fed_until = 0.0
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self._last_out: dict[str, ModuleOutput] = {}
        self._sync_offsets: list[int] = []
        self.transcript = None
        if self.cfg.transcript and self.s.enable_transcript and self._on(Modality.SCRIPTED):
            from trusthire.transcript.infer import TranscriptMonitor

            self.transcript = TranscriptMonitor(
                self.s.whisper_model,
                self.s.whisper_compute_type,
                resume_text,
                on_result=self._on_answer,
                synchronous=self.cfg.synchronous_transcript,
            )

    def _on(self, m: Modality) -> bool:
        return m.value in self.cfg.enabled

    def _relative_sync(self, o: ModuleOutput) -> ModuleOutput:
        """Re-score lip-sync against this session's own baseline offset.

        Genuine setups often have a constant A/V skew (capture pipelines, Bluetooth headsets add 150-250 ms);
        what indicates manipulation is a *changing* offset or low sync confidence. The baseline is the median
        offset of the session's confident windows (needs >= 3; until then the absolute offset is used).
        """
        from trusthire.avsync.infer import sync_risk

        off, conf = o.details.get("offset_frames"), o.details.get("sync_conf")
        if off is None or conf is None:
            return o
        if conf >= 4.0:
            self._sync_offsets.append(int(off))
        if len(self._sync_offsets) < 3:
            return o
        base = int(np.median(self._sync_offsets[-40:]))
        # absolute guard: > 320 ms skew stays suspicious even if it has been there since the start
        abs_term = float(np.clip((abs(int(off)) - 8) / 4.0, 0.0, 1.0)) * min(1.0, float(conf) / 3.0)
        score = max(sync_risk(int(off) - base, float(conf)), abs_term)
        return o.model_copy(update={"score": score, "details": {**o.details, "baseline_offset": base}})

    def _due(self, m: Modality, w: Window) -> bool:
        if not self._on(m):
            return False
        stride, offset = self.s.module_schedule.get(m.value, (1, 0))
        return stride <= 1 or w.index % stride == offset % stride

    # ------------------------------------------------------------------ audio context
    def _update_audio(self, w: Window) -> np.ndarray:
        """Append the not-yet-seen part of the window's audio; return the CTX_S seconds ending at w.t_end."""
        sr = w.sample_rate
        buf_end = self._audio_t0 + self._audio_buf.size / sr
        if self._audio_buf.size and (w.t_start > buf_end + 0.25 or w.t_end < buf_end - 0.25):
            self._audio_buf = np.zeros(0, np.float32)  # stream gap / restart: don't stitch across it
        if self._audio_buf.size == 0:
            self._audio_t0 = w.t_start
            self._audio_buf = w.audio.copy()
            new = w.audio
        else:
            buf_end = self._audio_t0 + self._audio_buf.size / sr
            skip = round((buf_end - w.t_start) * sr)
            new = w.audio[max(skip, 0) :]
            self._audio_buf = np.concatenate([self._audio_buf, new])
        keep = int(8 * sr)
        if self._audio_buf.size > keep:
            self._audio_t0 += (self._audio_buf.size - keep) / sr
            self._audio_buf = self._audio_buf[-keep:]
        if self.transcript is not None and new.size:
            self.transcript.push(new, w.t_end - new.size / sr)
        end = round((w.t_end - self._audio_t0) * sr)
        return self._audio_buf[max(0, end - int(CTX_S * sr)) : end]

    # ------------------------------------------------------------------ main entry
    def process_window(self, w: Window) -> FusedPoint:
        from trusthire.audio.infer import speech_ratio

        t0 = time.perf_counter()
        ctx = self._update_audio(w)
        # shared face detections on anchor frames (parallel; ORT runs release the GIL)
        if w.frames:
            list(
                self.pool.map(lambda i: detect_cached(w.cache, i, w.frames[i]), anchor_indices(len(w.frames)))
            )
        sr_ratio = speech_ratio(w.audio)
        jobs: dict[str, object] = {}
        if self._due(Modality.VIDEO, w):
            jobs["video"] = self.pool.submit(self.models.video.process, w)
        if self._due(Modality.AUDIO, w):
            jobs["audio"] = self.pool.submit(self.models.audio.process, w, ctx)
        if self._due(Modality.AVSYNC, w):
            jobs["avsync"] = self.pool.submit(self.models.avsync.process, w, sr_ratio)
        if self._due(Modality.FACE_ID, w):
            jobs["face"] = self.pool.submit(self.identity.face, w)
        if self._due(Modality.VOICE_ID, w):
            jobs["voice"] = self.pool.submit(self.identity.voice, w, sr_ratio, ctx)
        if self._on(Modality.BEHAVIOR) or self._on(Modality.LIVENESS) or self._on(Modality.OBJECTS):
            jobs["behavior"] = self.pool.submit(self.behavior.process, w, self._due(Modality.OBJECTS, w))
        outs: dict[str, ModuleOutput] = {}
        for name, fut in jobs.items():
            try:
                res = fut.result()  # type: ignore[attr-defined]
            except Exception:  # one failing branch must not kill the session; it simply counts as missing
                log.exception("branch %s failed on window %d", name, w.index)
                continue
            if isinstance(res, tuple):
                for o in res:
                    if o is not None and self._on(o.module):
                        outs[o.module.value] = o
            elif res is not None:
                outs[res.module.value] = res
        if Modality.AVSYNC.value in outs:
            outs[Modality.AVSYNC.value] = self._relative_sync(outs[Modality.AVSYNC.value])
        # modules not scheduled this window: carry the previous (adjacent, contiguous) window's evidence
        for m in Modality:
            if (
                m.value not in outs
                and self._on(m)
                and m.value in self._last_out
                and m not in (Modality.SCRIPTED, Modality.RESUME)
            ):
                prev = self._last_out[m.value]
                outs[m.value] = prev.model_copy(
                    update={"details": {**prev.details, "carried": True}, "latency_ms": 0.0}
                )
        self._last_out.update({k: v for k, v in outs.items() if not v.details.get("carried")})
        with self._lock:
            # carry the latest transcript-derived outputs forward if they overlap this window
            for m in (Modality.SCRIPTED, Modality.RESUME):
                tr_out = self._transcript_output_for(m, w.t_start, w.t_end)
                if tr_out is not None:
                    outs[m.value] = tr_out
            point = self.fuser.fuse(outs, w.t_start, w.t_end, index=len(self.points))
            self.points.append(point)
            self.outputs.append(outs)
        point.latency_ms = (time.perf_counter() - t0) * 1e3
        self.latencies.append(point.latency_ms)
        if self.on_point:
            self.on_point(point)
        return point

    # ------------------------------------------------------------------ transcript callbacks
    def _transcript_output_for(self, m: Modality, t0: float, t1: float) -> ModuleOutput | None:
        if self.transcript is None:
            return None
        for r in reversed(self.transcript.results):
            o = r.scripted if m == Modality.SCRIPTED else r.resume
            if o is not None and o.t_start < t1 and o.t_end > t0:
                return o
        return None

    def _on_answer(self, r: object) -> None:
        from trusthire.transcript.infer import AnswerResult

        assert isinstance(r, AnswerResult)
        with self._lock:
            for i, (p, outs) in enumerate(zip(self.points, self.outputs, strict=True)):
                if p.t_start < r.t_end and p.t_end > r.t_start:
                    outs[Modality.SCRIPTED.value] = r.scripted
                    if r.resume is not None:
                        outs[Modality.RESUME.value] = r.resume
                    np_ = self.fuser.fuse(outs, p.t_start, p.t_end, index=i)
                    self.points[i] = np_
                    if self.on_update:
                        self.on_update(i, np_)
        if self.on_transcript:
            self.on_transcript(r)

    def mark_question(self, t: float) -> None:
        if self.transcript is not None:
            self.transcript.mark_question(t)

    def finish(self) -> None:
        if self.transcript is not None:
            self.transcript.flush(wait=True)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.behavior.close()
        if self.transcript is not None:
            self.transcript.close()
