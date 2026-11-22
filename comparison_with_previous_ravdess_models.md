# RAVDESS: CREMA-D-initialised wav2vec2 vs previous RAVDESS models

Repeat exploratory comparison: RAVDESS test actors 16, 17, 19 and 24 (112 clips) were already examined in the previous RAVDESS comparison, so this is not a fresh blind benchmark. All settings and the epoch were fixed on RAVDESS validation before this single test evaluation. Scores are not comparable with CREMA-D test scores.

- Git commit: `ee1a38ae0f9cc46cc5a0dd6c85ec8930d319367e`
- Initialisation: `D:\SER-CAT\work\crema_d\wav2vec2_experiments\wav2vec2_base_seed_20261108\best_model.pt`
- Initialisation SHA-256: `e7efd80277e616d9c8efc70ea581bffc824bbfb42d1156319c5bd7f94f0c4f8d`
- Transfer settings: encoder LR 1e-06, head LR 1e-05, max 15 epochs, patience 3, batch 4 x accumulation 4, class-weighted loss, frozen conv feature encoder
- Selected epochs (RAVDESS validation): transfer 12, base 10
- RAVDESS validation macro F1 of the CREMA-D model before any RAVDESS training: 0.1962
- CREMA-D validation retention (702 clips, no CREMA-D test used): macro F1 0.7946 before -> 0.7244 after

## RAVDESS validation (actors 07, 08, 12, 13)

| Model | Macro F1 | Accuracy | Clips |
|---|---|---|---|
| MFCC + SVM | 0.6413 | 0.6607 | 112 |
| wav2vec2 from base (previous) | 0.6864 | 0.6964 | 112 |
| wav2vec2 from CREMA-D (transfer) | 0.7206 | 0.7321 | 112 |

**MFCC + SVM: per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.931 | 0.844 | 0.885 | 32 |
| happiness | 0.676 | 0.719 | 0.697 | 32 |
| neutral | 0.345 | 0.625 | 0.444 | 16 |
| sadness | 0.700 | 0.438 | 0.538 | 32 |

**MFCC + SVM: confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 27 | 1 | 4 | 0 |
| happiness | 2 | 23 | 6 | 1 |
| neutral | 0 | 1 | 10 | 5 |
| sadness | 0 | 9 | 9 | 14 |

**wav2vec2 from base (previous): per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.938 | 0.938 | 0.938 | 32 |
| happiness | 0.950 | 0.594 | 0.731 | 32 |
| neutral | 0.357 | 0.938 | 0.517 | 16 |
| sadness | 0.778 | 0.438 | 0.560 | 32 |

**wav2vec2 from base (previous): confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 30 | 1 | 0 | 1 |
| happiness | 2 | 19 | 9 | 2 |
| neutral | 0 | 0 | 15 | 1 |
| sadness | 0 | 0 | 18 | 14 |

**wav2vec2 from CREMA-D (transfer): per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.833 | 0.938 | 0.882 | 32 |
| happiness | 0.760 | 0.594 | 0.667 | 32 |
| neutral | 0.517 | 0.938 | 0.667 | 16 |
| sadness | 0.818 | 0.562 | 0.667 | 32 |

**wav2vec2 from CREMA-D (transfer): confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 30 | 2 | 0 | 0 |
| happiness | 6 | 19 | 4 | 3 |
| neutral | 0 | 0 | 15 | 1 |
| sadness | 0 | 4 | 10 | 18 |

## RAVDESS test, repeat exploratory (actors 16, 17, 19, 24)

| Model | Macro F1 | Accuracy | Clips |
|---|---|---|---|
| MFCC + SVM | 0.4932 | 0.5179 | 112 |
| wav2vec2 from base (previous) | 0.6327 | 0.6429 | 112 |
| wav2vec2 from CREMA-D (transfer) | 0.6936 | 0.7054 | 112 |

**MFCC + SVM: per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.659 | 0.844 | 0.740 | 32 |
| happiness | 0.435 | 0.312 | 0.364 | 32 |
| neutral | 0.364 | 0.500 | 0.421 | 16 |
| sadness | 0.500 | 0.406 | 0.448 | 32 |

**MFCC + SVM: confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 27 | 2 | 3 | 0 |
| happiness | 13 | 10 | 4 | 5 |
| neutral | 0 | 0 | 8 | 8 |
| sadness | 1 | 11 | 7 | 13 |

**wav2vec2 from base (previous): per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.683 | 0.875 | 0.767 | 32 |
| happiness | 0.632 | 0.375 | 0.471 | 32 |
| neutral | 0.560 | 0.875 | 0.683 | 16 |
| sadness | 0.667 | 0.562 | 0.610 | 32 |

**wav2vec2 from base (previous): confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 28 | 1 | 1 | 2 |
| happiness | 11 | 12 | 3 | 6 |
| neutral | 1 | 0 | 14 | 1 |
| sadness | 1 | 6 | 7 | 18 |

**wav2vec2 from CREMA-D (transfer): per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.711 | 0.844 | 0.771 | 32 |
| happiness | 0.867 | 0.406 | 0.553 | 32 |
| neutral | 0.533 | 1.000 | 0.696 | 16 |
| sadness | 0.793 | 0.719 | 0.754 | 32 |

**wav2vec2 from CREMA-D (transfer): confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 27 | 1 | 3 | 1 |
| happiness | 11 | 13 | 3 | 5 |
| neutral | 0 | 0 | 16 | 0 |
| sadness | 0 | 1 | 8 | 23 |
