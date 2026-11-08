# MFCC + SVM Baseline Results

**Run date:** 2026-11-08  
**Target:** CREMA-D actor-intended emotion labels  
**Evaluation:** actor-disjoint test set, seed `20261108`

## Result

The selected validation model was an RBF SVM with `C=1.0`, class-balanced
weights, and a `StandardScaler` in the training pipeline. It reached **0.695
macro F1 on validation**. On the held-out speakers, it reached **0.623 macro
F1** and **0.625 accuracy**. The project target is approximately 0.70 macro
F1, so this baseline did not meet that target on the test set.

| Actor-intended label | Precision | Recall | F1 | Test clips |
|---|---:|---:|---:|---:|
| Anger | 0.773 | 0.714 | 0.743 | 196 |
| Happiness | 0.500 | 0.607 | 0.548 | 196 |
| Neutral | 0.475 | 0.449 | 0.462 | 167 |
| Sadness | 0.775 | 0.704 | 0.738 | 196 |
| **Macro average** | **0.631** | **0.619** | **0.623** | **755** |

Confusion matrix (rows are actual labels, columns are predicted labels):

| Actual / Predicted | Anger | Happiness | Neutral | Sadness |
|---|---:|---:|---:|---:|
| Anger | 140 | 42 | 13 | 1 |
| Happiness | 37 | 119 | 30 | 10 |
| Neutral | 3 | 60 | 75 | 29 |
| Sadness | 1 | 17 | 40 | 138 |

The largest confusions were happiness predicted as anger or neutral, and
neutral predicted as happiness. Sadness was also often predicted as neutral.
These results suggest the simple clip-level MFCC summaries separate anger and
sadness better than happiness and neutral for the held-out actors.

## Method

- Usable clips: **4,899** after excluding the all-zero recording and leaving
  disgust/fear outside the first four-class model.
- Split: 64 training actors (3,442 clips), 13 validation actors (702 clips),
  and 14 test actors (755 clips). An actor appears in only one partition.
- Audio: mono, 16 kHz.
- Features: 13 MFCC coefficients computed with 25 ms frames and 10 ms hops;
  each clip is summarized by per-coefficient mean and standard deviation,
  producing 26 features.
- Model selection: linear and RBF SVMs, `C` values 0.1, 1, and 10; feature
  scaling is fitted inside each training pipeline. The selection metric was
  validation macro F1.
- Test evaluation: performed once after selecting RBF, `C=1.0`. Test results
  were not used to choose settings.
- Package versions used for the run are recorded in
  [`../ml/requirements-lock.txt`](../ml/requirements-lock.txt).

## Interpretation and next step

This is a modest classical baseline, not a production claim. It predicts the
emotion actors were instructed to portray in short acted clips. It does not
show that the model can detect a speaker's internal feelings or generalize to
spontaneous customer-support speech. Frustration is not a CREMA-D label and
was not predicted.

Keep this test result fixed. If improving the model, use training and
validation data for decisions; evaluate any substantially revised final model
with a newly planned held-out protocol rather than repeatedly tuning against
these same test scores.

Generated model, cache, and machine-readable result files are under the ignored
`work/crema_d/baseline_results/` directory.
