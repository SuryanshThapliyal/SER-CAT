# MFCC + SVM validation tuning

**Status:** The tuned candidate was selected on validation data and evaluated
once on the existing held-out test speakers after its settings were locked.

## Protocol

This experiment keeps the existing CREMA-D actor-disjoint train, validation,
and test partitions and the existing 26 MFCC summary features. It compares
`StandardScaler` and `RobustScaler`, linear and RBF SVMs, class-balanced
weights, and `C` values 1, 3, and 10. RBF models additionally use `gamma`
values `scale`, 0.01, 0.03, and 0.1. The script selects only by validation
macro F1. The training split is used for fitting; the validation split is used
for selection. Test examples were not scored or used to choose settings. After
locking the selected candidate, the script evaluated it once on the existing
test actors and wrote the result separately from the original baseline report.

The validation-selected candidate was RobustScaler + RBF SVM with `C=3` and
`gamma=0.03`. It scored **0.7041 validation macro F1** (accuracy 0.7037) on
702 clips from 13 validation actors. Per-class F1 was 0.7224 for anger, 0.6590
for happiness, 0.6873 for neutral, and 0.7477 for sadness. This modest
validation result is not evidence that the model meets the target on unseen
speakers.

## Held-out test result

On the 755 test clips from 14 held-out actors, the tuned candidate achieved
**0.6277 macro F1** and **0.6305 accuracy**. Per-class F1 was 0.7500 for anger,
0.5514 for happiness, 0.4634 for neutral, and 0.7461 for sadness.

| Model | Validation macro F1 | Held-out test macro F1 | Held-out test accuracy |
|---|---:|---:|---:|
| Original StandardScaler + RBF SVM (`C=1`) | 0.6950 | 0.6226 | 0.6252 |
| Tuned RobustScaler + RBF SVM (`C=3`, `gamma=0.03`) | 0.7041 | 0.6277 | 0.6305 |

The tuned candidate is higher by about 0.0051 macro F1 on this split. This is a
small descriptive difference, not evidence of a reliable or statistically
significant improvement. The target of approximately 0.70 macro F1 was reached
on validation, but not on the held-out test set.

Confusion matrix for the tuned candidate (rows are actual labels; columns are
predicted labels in anger, happiness, neutral, sadness order):

| Actual / Predicted | Anger | Happiness | Neutral | Sadness |
|---|---:|---:|---:|---:|
| Anger | 138 | 43 | 14 | 1 |
| Happiness | 31 | 118 | 34 | 13 |
| Neutral | 2 | 57 | 76 | 32 |
| Sadness | 1 | 14 | 37 | 144 |

The complete metrics for all 30 candidates, the selected settings, and the
validation classification report are stored in the ignored
`work/crema_d/baseline_results/validation_results.json`. The tuned test report
is saved as the ignored `work/crema_d/baseline_results/tuned_test_results.json`;
the original `test_results.json` remains the historical baseline result and
was not overwritten. The script refuses to overwrite the tuned test report.

## Test-set boundary

The historical result in [`baseline-results.md`](baseline-results.md) belongs
to the original StandardScaler, RBF, `C=1` model. Both model results are now
recorded separately. Do not make additional model or feature choices based on
this test set. A future model comparison or substantial revision needs a newly
planned held-out-speaker protocol before making another generalization claim.
