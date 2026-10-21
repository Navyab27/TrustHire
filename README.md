# TrustHire — multimodal fake-interview detection

TrustHire watches a remote interview (live over WebRTC, or an uploaded recording) and builds a
**per-timestamp risk timeline** from ten evidence modules:

| Video | Audio | Audio-visual | Identity | Behaviour | Transcript |
|---|---|---|---|---|---|
| face-forgery CNN (EfficientNet-B4) | synthetic-voice detector (Spectra-0 int8) | lip-sync (SyncNet) | face re-verification (ArcFace) + voiceprint (ECAPA-TDNN) | gaze / second-screen, blink, rPPG pulse liveness, YOLO26 objects | Whisper → fillers, pause regularity, prosody, AI-text; resume RAG + NLI |

Scores are fused (confidence-weighted log-odds / LightGBM / cross-modal attention, calibrated) into a trust
score, alerts and an explainability report (Grad-CAM, audio saliency, SHAP/attention attributions, LLM summary → PDF).

> TrustHire produces **evidence for a human reviewer, not verdicts.** Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Quick start (one command)

```bash
make demo          # Linux/macOS
.\make demo        # Windows PowerShell / cmd (no GNU make needed)
```

This creates the environment with **uv** (Python 3.12, venv in `~/.trusthire/venv`), downloads ~1.9 GB of
pretrained weights to `~/.trusthire/models`, builds the React dashboard, starts the API on
<http://127.0.0.1:8000> and opens the browser. Requirements: Python ≥ 3.10 (any; uv fetches 3.12), Node ≥ 20.

Then:
1. **Live interview** → enter a name → *Create* (allow camera/mic) → enroll an ID photo (upload or webcam snapshot),
   record an 8 s voiceprint, optionally upload the resume → *Start interview*. Press **Q** after each question.
2. **Upload recording** → choose an mp4/webm (+ optional ID photo / resume) → watch progress → open results.
3. **Sessions** → timeline, per-window module evidence, alerts, transcript analysis → *Generate report* → PDF.

`TRUSTHIRE_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` set (or `ant auth login`) gives Claude-written
summaries; without a key the report uses a deterministic template; `ollama` uses a local model.

## Measured results (real runs, this repo — see `results/` and [docs/report/paper.md](docs/report/paper.md))

| What | Result |
|---|---|
| Voice anti-spoofing, **In-the-Wild** (n=400, cross-domain) | Spectra-0 int8 **EER 1.25 %**, AASIST (brief's baseline) 42.5 % |
| …under Opus 16 kbps / white noise 10 dB | 3.0 % / 7.5 % EER |
| Voice anti-spoofing, ASVspoof19 LA eval (n=308) | Spectra-0 0.0 %, AASIST 1.6 % EER |
| Face verification, LFW 1000 pairs (end-to-end SCRFD→ArcFace) | AUC 0.986, EER 3.0 %, 0 % impostors accepted at the deployed threshold |
| AV sync on the SyncNet reference clip | offset 3 / conf 9.8 (reference 3 / 10.0); injected ±k-frame shifts recovered exactly |
| Live latency, laptop CPU (i5-1235U, no GPU) | **1.69 s** p50 per 3 s window (hop 1.5 s) |
| **Team's recorded videos (18, pilot)** — learned fusion, grouped CV | session AUC 0.714 → **0.938 ± 0.019**, EER 46.8 % → 16.1 ± 5.7 % (5 seeds); AI-video detector AUC 0.000 → **1.000** ([details](docs/CUSTOM_RESULTS.md)) |

Pending because they need data you must request: video deepfake benchmarks (FF++, Celeb-DF, DFDC, LAV-DF,
DeepSpeak) and the **custom interview test set** ablations — see below.

## Repository

```
src/trusthire/     python package (see docs/ARCHITECTURE.md for the module map + diagram)
frontend/          React 19 + TypeScript + Vite 8 + Tailwind 4 + Recharts 3 dashboard
configs/           Hydra configs for training
notebooks/         Kaggle/Colab training notebooks + results notebook
docker/            Dockerfiles (CPU api/worker, GPU inference, frontend) — docker-compose.yml at the root
docs/              PLAN, PHASES, ARCHITECTURE, MODEL_CARDS, DATASETS, DATA_COLLECTION, LIMITATIONS, report/
results/           benchmark outputs (JSON / Markdown / plots) produced by real runs
tests/             pytest (60+ tests; model tests run the real weights on a real clip)
```

## Common tasks

| Task | Command |
|---|---|
| Install env + frontend deps | `make setup` |
| Download weights (`TRUSTHIRE_PROFILE=gpu` for the GPU set) | `make models` |
| Dev mode (Vite HMR on :5173 + API reload) | `make dev` |
| Tests / lint / types | `make test` · `make lint` · `make typecheck` |
| Public benchmarks (audio, LFW, AV-sync, latency) | `make bench` |
| Custom test set → ablations → report | `make eval-custom` (protocol: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)) |
| Regenerate the paper from results | `make report` |
| Docker (redis + api + celery worker + nginx frontend; `--profile gpu` adds CUDA inference) | `make docker-up` |
| Offline CLI analysis | `trusthire analyze interview.mp4 --id-photo id.jpg --resume cv.pdf --report` |

## GPU profile

`TRUSTHIRE_PROFILE=gpu` switches to: every module on every window, 8 face crops per window, ArcFace R100,
SCRFD at 640 px, Whisper large-v3-turbo fp16, ONNX Runtime TensorRT/CUDA execution providers with fp16
engines (`docker/Dockerfile.gpu`). Measure with `TRUSTHIRE_PROFILE=gpu python -m trusthire.evaluation.latency`.

## Training (Kaggle / Colab)

Heavy training is not run on the laptop. `notebooks/train_video_kaggle.ipynb` (FF++ → Celeb-DF; EfficientNet-B4,
Xception, ConvNeXt-V2, DINOv2; BiLSTM vs transformer temporal heads) and `notebooks/train_audio_kaggle.ipynb`
(ASVspoof19 LA; WavLM / XLS-R fine-tuning, AASIST, RawNet2) call `trusthire.video.train` / `trusthire.audio.train`.
Drop the resulting `*_trained.pth` into `~/.trusthire/models/<module>/` and the live system uses them.

## Windows notes

* Smart App Control may block some unsigned DLLs; `parselmouth` (Praat) is optional — pitch falls back to librosa YIN.
* Keep the venv and model cache outside OneDrive (defaults already do).
* Denormal flushing is enabled for ONNX Runtime and torch (40× faster CNNs on CPU).

## Licences

Code: Apache-2.0. Model weights carry their own licences — InsightFace and DeepfakeBench are **non-commercial**,
Ultralytics YOLO is **AGPL-3.0**; see [docs/MODEL_CARDS.md](docs/MODEL_CARDS.md).
