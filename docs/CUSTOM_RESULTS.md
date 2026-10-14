# Results on the team's recorded videos (pilot set, 2026-10-14)

**Data** (`KeyZama10/TrustHire/Training Data`, imported with `scripts/import_recordings.py`): 18 videos, 339
analysis windows — 5 genuine webcam clips + 2 re-encoded copies of two of them (480p / 1440p), 3 scripted-reading
clips (same participant, different room), 2 audio-mismatch (dubbed) clips, 6 fully AI-generated talking heads.
One real participant only. Re-encoded copies are grouped with their originals in cross-validation, so no model is
ever tested on a copy of a video it was trained on. **Small, single-participant set: treat as a pilot.**

## What changed and what it bought (all numbers from real runs; files in `results/`)

| Change | Evidence |
|---|---|
| Video detector EfficientNet-B4 (FF++, 2019) → **SwinV2-L AI-image detector** (int8 ONNX) | on these videos: AUC 0.000 (inverted) → **1.000**; real clips P(fake) ≤ 0.008, AI clips ≥ 0.966 (`results/video/pretrained_on_custom.md`) |
| Hand-set fusion → **LightGBM fusion learned on these videos** (all modules except transcript) | session AUC 0.688 → **0.935**, EER 46.8 % → **14.3 %** (grouped CV, `results/custom/results.json`) |

Per-video trust (0–100) with the hand-set fusion — old video detector vs new detector:

| Video type | old detector | new detector |
|---|---|---|
| genuine (7 incl. re-encoded) | 65–94 | **94.8–98.7** |
| AI-generated (6) | 1–9 | **0.0–2.1** |
| audio mismatch (2) | 90.9 / 35.8 | 97.6 / 51.4 → learned fusion: caught (93 % of windows at 5 % FPR) |
| scripted reading (3) | 82–95 | 97.5–98.2 → **still missed** |

Learned fusion, cross-validated, at the threshold giving 5 % false alarms on genuine windows: AI-generated
windows caught 92 %, audio-mismatch 93 %, scripted reading 20 %; genuine windows flagged 0.9 %.

## What did not work (also measured)

* **Scripted-reading detection** — no module separates the 3 scripted clips from genuine ones: the hand-set
  fluency rules overlap completely (0.59–0.89 vs 0.30–0.91), and the RoBERTa AI-text detector rates many *genuine*
  spoken answers 0.95–0.999 "AI-written". Gaze is slightly steadier when reading (0.4–2.4 % off-axis vs 1–23 %)
  but not separable at n = 3. Needs ~15–20 scripted sessions from several people.
* **Voice-model fine-tuning for reverberation** (Kaggle T4, ASVspoof 2019 LA + reverb/noise/codec augmentation,
  `results/audio/kaggle/reverb_finetune_results.json`): EER improved slightly under reverb (28.75 → 26.0 % at
  RT60 0.4 s) but genuine voices flagged at P>0.5 rose from 26.5 % to 73.5 % — not shipped.
* **Transcript features in fusion** made every combination worse on this data (transcript-only AUC 0.33), so the
  shipped fusion excludes them; they are still shown on the dashboard as evidence.

## Caveats

* One real participant, two rooms: "behaviour" features may partly encode the room, not the attack.
* The AI clips come from text-to-video generators; real-time face swaps (DeepFaceLive) are not in this set.
* The fused models are trained on these 18 videos; the cross-validated numbers above are the honest estimate,
  not the in-sample behaviour of the shipped model.
