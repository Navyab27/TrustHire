| Model | Input | AUC | Acc@0.5 | mean P(fake) real | mean P(fake) AI | ms/img |
|---|---|---|---|---|---|---|
| haywoodsloan/ai-image-detector-deploy | crop | 1.000 | 1.00 | 0.00 | 0.99 | 2377 |
| haywoodsloan/ai-image-detector-deploy | full | 1.000 | 1.00 | 0.01 | 1.00 | 2333 |
| prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 | full | 0.944 | 0.61 | 0.68 | 1.00 | 433 |
| Ateeqq/ai-vs-human-image-detector | crop | 0.889 | 0.33 | 1.00 | 1.00 | 405 |
| Ateeqq/ai-vs-human-image-detector | full | 0.875 | 0.33 | 1.00 | 1.00 | 404 |
| umm-maybe/AI-image-detector | full | 0.861 | 0.67 | 0.15 | 0.27 | 477 |
| prithivMLmods/Deep-Fake-Detector-v2-Model | crop | 0.833 | 0.67 | 0.43 | 0.60 | 505 |
| umm-maybe/AI-image-detector | crop | 0.736 | 0.78 | 0.27 | 0.42 | 589 |
| effnb4 (current) | full | 0.708 | 0.67 | 0.31 | 0.47 | 132 |
| prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 | crop | 0.611 | 0.72 | 0.04 | 0.16 | 389 |
| xception | full | 0.264 | 0.67 | 0.39 | 0.32 | 194 |
| dima806/deepfake_vs_real_image_detection | full | 0.250 | 0.67 | 0.08 | 0.00 | 511 |
| xception | crop | 0.139 | 0.39 | 0.45 | 0.21 | 194 |
| prithivMLmods/Deep-Fake-Detector-v2-Model | full | 0.097 | 0.33 | 0.87 | 0.82 | 531 |
| dima806/deepfake_vs_real_image_detection | crop | 0.042 | 0.22 | 0.95 | 0.58 | 517 |
| effnb4 (current) | crop | 0.000 | 0.50 | 0.45 | 0.06 | 123 |

Custom recordings: 6 AI-generated vs 12 real-camera videos, 8 frames each. Small set (one real participant) — indicative only.
