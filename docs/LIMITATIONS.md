# Limitations, fairness and risks

Grounded in what we measured (see `results/`) and in what we could not measure.

## Evidence gaps

0. **Custom-set results are a pilot** (18 videos, one real participant) — see `docs/CUSTOM_RESULTS.md`. Scripted
   reading is not detected (20 % of windows at 5 % FPR); the learned fusion is trained on this small set.

1. **Video: the FF++ EfficientNet-B4 was inverted on our webcam data** (AUC 0.000) and was replaced by a SwinV2
   AI-image detector (AUC 1.000 on our 18 videos). Public face-swap benchmarks (FF++, Celeb-DF) are still pending.
   *Previously:* **Video deepfake numbers are pending.** FF++, Celeb-DF, DFDC, LAV-DF and DeepSpeak require access approval;
   the shipped EfficientNet-B4 / Xception weights are DeepfakeBench's FF++ models whose published cross-dataset
   AUC on Celeb-DF is only ~0.7–0.75. Real-time face-swap tools (DeepFaceLive) produce artefacts that differ
   from FF++; expect domain shift until the detector is fine-tuned on interview-like data (DeepSpeak, own set).
2. **The fusion layer ships with a hand-set prior** (documented weights). Learned fusers (LightGBM, cross-modal
   attention) and their calibration need the labelled custom test set; until then risk values are *not*
   calibrated probabilities.
3. **Custom test set size.** 30–50 sessions from ~10 participants gives wide confidence intervals; we report
   bootstrap CIs and leave-participants-out CV, but per-attack and per-stratum numbers will be noisy.

## Measured weaknesses

* **Voice anti-spoofing — room reverberation is the biggest measured weakness.** On the same 400 In-the-Wild
  utterances, Spectra-0 int8 EER is 1.25 % clean, 2.0 % telephone band, 2.25 % MP3-32k, 3.0 % Opus-16k,
  7.5 % at 10 dB white noise, but **27.5 % with RT60 = 0.4 s reverb**, where 26.5 % of *genuine* voices exceed
  P(spoof)=0.5 (5 % clean). A WPE dereverberation front-end did **not** help (26.5 % EER, measured). Mitigations
  shipped: the module's confidence drops with estimated SNR (noise), and fusion treats voice-spoof as one
  signal among many; the real fix is fine-tuning with room-impulse-response augmentation (`audio.train`
  includes it) — needs a GPU and ASVspoof train data. Until then a reverberant room can raise false
  synthetic-voice alerts. AASIST (the brief's baseline) fails out of domain everywhere (42–57 % EER).
* **Enrollment-based channel calibration was rejected** on purpose: the candidate controls enrollment, so a
  voice cloned already at enrollment would calibrate itself away.
* **False positives on genuine but heavily compressed media:** on the SyncNet example clip (old broadcast
  footage) one window scored 0.71 synthetic-voice, and rPPG reported a "weak pulse" in 4/9 windows. Fusion kept
  the session at 96/100 trust, but per-module alerts fire.
* **Face verification end-to-end:** 96.9 % LFW accuracy through our 320-px detector + alignment (EER 3.0 %),
  below the ~99.8 % reported for ArcFace on the canonical aligned protocol; at the deployed threshold 3.6 % of
  same-person LFW pairs would be flagged (0 % impostors accepted). Webcam angles, glasses and lighting add more.
* **CPU latency:** 1.69 s p50 per 3 s window on an i5-1235U laptop (hop budget 1.5 s); the live worker drops the
  oldest queued window when behind. The < 1.5 s target needs a GPU (or fewer modules).
* **Speaker verification:** ECAPA thresholds were tuned on VoxCeleb-style speech; see
  `results/identity/speaker_librispeech.json` for the measured operating point on 4 s segments.

## Known blind spots by design

* **AI-text detection on transcripts** is weak: detectors are trained on written text, and Whisper punctuation
  differs from LLM prose; it carries low weight and never alerts on its own.
* **Reading detection** relies on gaze sweeps and fluency; candidates with teleprompter-style setups near the
  camera, or naturally fluent speakers, are hard cases.
* **rPPG** is unreliable under compression, motion, dark skin tones in poor light, and makeup — low weight,
  info-level alerts only.
* **Adaptive attackers** who know the detectors (e.g. adding noise, using lip-synced puppets driven by real
  audio, replaying genuine enrollment audio) are not evaluated.

## Fairness

The harness reports AUC and false-flag rate per stratum (lighting, accent, webcam, bitrate, gender, skin tone,
glasses). Specific concerns to check before any use:
* ASR-derived features (fillers, pause regularity, response latency) are **accent- and language-dependent**;
  non-native speakers may read as "scripted". Whisper `small` errors are higher for some accents.
* Face detection/verification error rates differ across skin tones and lighting; rPPG degrades on darker skin
  under low light.
* Neurodivergent candidates may show atypical gaze and blink behaviour; behaviour alerts must never be used alone.

## Governance

TrustHire is decision support. Every alert links to its evidence; the report states that it is not a
determination of fraud. Obtain consent from candidates, disclose the monitoring, keep a human in the loop and
provide an appeal path. Several model licences are non-commercial (InsightFace, DeepfakeBench) or AGPL
(Ultralytics).
