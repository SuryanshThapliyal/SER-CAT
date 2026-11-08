# MFCC + SVM validation tuning

**Status:** Validation-only model selection; the tuned candidate has not been
evaluated on the held-out test speakers.

## Protocol

This experiment keeps the existing CREMA-D actor-disjoint train, validation,
and test partitions and the existing 26 MFCC summary features. It compares
`StandardScaler` and `RobustScaler`, linear and RBF SVMs, class-balanced
weights, and `C` values 1, 3, and 10. RBF models additionally use `gamma`
values `scale`, 0.01, 0.03, and 0.1. The script selects only by validation
macro F1. The training split is used for fitting; the validation split is used
for selection. Test examples are not scored or used to choose settings.

The validation-selected candidate was RobustScaler + RBF SVM with `C=3` and
`gamma=0.03`. It scored **0.7041 validation macro F1** (accuracy 0.7037) on
702 clips from 13 validation actors. Per-class F1 was 0.7224 for anger, 0.6590
for happiness, 0.6873 for neutral, and 0.7477 for sadness. This modest
validation result is not evidence that the model meets the target on unseen
speakers.

## Result

The complete metrics for all 30 candidates, the selected settings, and the
validation classification report are stored in the ignored
`work/crema_d/baseline_results/validation_results.json`; the selected model is
saved as `best_svm.joblib` in the same directory. The run did not evaluate the
test actors, and the historical `test_results.json` was left untouched.

## Test-set boundary

The historical held-out test result in [`baseline-results.md`](baseline-results.md)
belongs to the original StandardScaler, RBF, `C=1` model. It remains unchanged
and must not be used to tune this search. The tuned candidate has not been
evaluated on that test set. A fresh held-out-speaker protocol is needed before
making a final generalization claim for a revised model.
