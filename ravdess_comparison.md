# RAVDESS four-class comparison: MFCC + SVM vs wav2vec2-base

These scores measure generalisation to four unseen RAVDESS actors (16, 17, 19, 24; 112 clips) and compare only the two models trained on this RAVDESS split. They are not comparable with the CREMA-D test scores.

- Git commit: `c3295ec0b1c412c2d6e87d71002e6b6bc9140a21`
- Split seed 20261122; train actors ['01', '02', '03', '04', '05', '06', '09', '10', '11', '14', '15', '18', '20', '21', '22', '23'], validation ['07', '08', '12', '13'], test ['16', '17', '19', '24']
- SVM selected by validation: scaler=robust, kernel=rbf, C=3.0, gamma=0.01, class_weight=balanced
- wav2vec2 selected epoch (validation): 10
- Primary metric: macro F1 (neutral has half as many clips as each other class)
- Data: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391. Zenodo record 1188976. License: CC BY-NC-SA 4.0.

## Validation (actors 07, 08, 12, 13)

| Model | Macro F1 | Accuracy | Clips |
|---|---|---|---|
| MFCC + SVM | 0.6413 | 0.6607 | 112 |
| wav2vec2-base | 0.6864 | 0.6964 | 112 |

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

**wav2vec2-base: per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.938 | 0.938 | 0.938 | 32 |
| happiness | 0.950 | 0.594 | 0.731 | 32 |
| neutral | 0.357 | 0.938 | 0.517 | 16 |
| sadness | 0.778 | 0.438 | 0.560 | 32 |

**wav2vec2-base: confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 30 | 1 | 0 | 1 |
| happiness | 2 | 19 | 9 | 2 |
| neutral | 0 | 0 | 15 | 1 |
| sadness | 0 | 0 | 18 | 14 |

## Test, evaluated once (actors 16, 17, 19, 24)

| Model | Macro F1 | Accuracy | Clips |
|---|---|---|---|
| MFCC + SVM | 0.4932 | 0.5179 | 112 |
| wav2vec2-base | 0.6327 | 0.6429 | 112 |

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

**wav2vec2-base: per class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.683 | 0.875 | 0.767 | 32 |
| happiness | 0.632 | 0.375 | 0.471 | 32 |
| neutral | 0.560 | 0.875 | 0.683 | 16 |
| sadness | 0.667 | 0.562 | 0.610 | 32 |

**wav2vec2-base: confusion matrix (rows = true, columns = predicted)**

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 28 | 1 | 1 | 2 |
| happiness | 11 | 12 | 3 | 6 |
| neutral | 1 | 0 | 14 | 1 |
| sadness | 1 | 6 | 7 | 18 |
