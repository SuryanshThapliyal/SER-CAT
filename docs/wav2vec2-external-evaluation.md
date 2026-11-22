# Wav2vec 2.0 external evaluation and improvement plan

## What we observed

The wav2vec 2.0 website option reportedly predicted anger at over 93% for
each of four TESS demo clips. The clips represent anger, happiness, neutral,
and sadness, but all are the word `back` spoken by the same older talker (OAF).
This is a useful warning about transfer to a different corpus, not an
accuracy estimate: four clips from one speaker cannot establish cross-speaker
performance.

The complete TESS collection contains 2,800 recordings: 200 words spoken by
two actresses for seven emotions. The four emotions shared with this project
can be evaluated, but even all of those TESS recordings still cover only two
speakers. Treat the result as an exploratory cross-corpus diagnostic, not a
final speaker-independent benchmark. The collection is available from the
[official Borealis TESS record](https://doi.org/10.5683/SP2/E8H2MF) under the
license shown on that record. Preserve its attribution and license terms.

For a more speaker-diverse manual check, we prepared
[`RAVDESS_8speaker_4emotion_manual_samples.zip`](../work/crema_d/external_test_samples/RAVDESS_8speaker_4emotion_manual_samples.zip)
from the official RAVDESS speech archive. It contains 64 WAVs: eight speakers
(four male and four female), four shared emotions, two utterances per speaker
and emotion, and 16 clips per class. Its `samples.csv` gives the actor-intended
label for checking website predictions. This small, balanced sample is still
exploratory and must not be treated as a final accuracy benchmark. See the
archive's README for source citation and license.

## Run the inference-only diagnostic

Place the full TESS WAV folder anywhere below:

```text
work/crema_d/external_test_samples/TESS/
```

The evaluator scans recursively, keeps anger, happiness, neutral, and sadness,
and skips the other TESS emotions. It uses the website's `librosa` decode path
and the same local wav2vec checkpoint. It reports accuracy, macro F1,
per-class precision/recall/F1, a confusion matrix, speaker count, and each
clip's scores. It does not train, tune, or change a checkpoint.

```powershell
.\.venv\Scripts\python.exe ml\evaluate_tess_wav2vec2.py `
  --audio-dir "work\crema_d\external_test_samples\TESS\<extracted-folder>" `
  --device cpu
```

The local four-clip demo can be scored with the same command by pointing
`--audio-dir` to its current folder. The report will explicitly mark a set
under 100 clips as demo-only. A full TESS result will still be marked as
limited in speaker diversity. CPU inference is supported but may take a long
time; GPU inference is optional for this evaluation, not a training step.

## How to improve the model after the diagnostic

Do not fine-tune on these four demo clips or use them to choose settings. The
next experiment is now a separate RAVDESS-only comparison: train both the
MFCC + SVM and wav2vec 2.0 on the same RAVDESS training actors, select on the
same validation actors, and compare once on the same held-out RAVDESS actors.
This isolates model performance on a new dataset; it is not a combined
CREMA-D/RAVDESS training run. Follow
[`ravdess-two-model-gpu-handoff.md`](ravdess-two-model-gpu-handoff.md) for the
fixed split, class mapping, and instructions. TESS can remain a separate
cross-corpus diagnostic, with its two-speaker limitation stated.

Before any new training, define a new actor-disjoint or corpus-aware protocol:

1. Keep the existing CREMA-D checkpoint and its reported test result intact as
   the historical baseline.
2. Add RAVDESS speech to the CREMA-D training data; document the four-class
   mapping and excluded RAVDESS emotions.
3. Assign speakers or entire corpora to train, validation, and final test
   before training. Use validation only for model/setting choices.
4. Fine-tune using the training portion, choose the checkpoint using
   validation macro F1, then evaluate once on the new untouched test portion.
5. Report macro F1 as the primary metric, with accuracy, per-class metrics,
   confusion matrix, class supports, and split details alongside it.

Fine-tuning wav2vec 2.0 requires a GPU for a practical training run. No GPU
training has been started as part of the TESS diagnostic preparation. A full
TESS inference evaluation does not mathematically require a GPU, but it is
likely impractical on this PC: an attempted CPU-only parity run was stopped
after wav2vec inference proved extremely slow. The evaluator supports CPU and
CUDA, so run the full evaluation on the GPU computer if available; this is
inference only and does not retrain the model.

## Interpretation of macro F1

Macro F1 is the arithmetic mean of the four per-class F1 scores, giving each
emotion equal weight. It is the primary comparison metric because it exposes
poor performance on an emotion even if other classes have more examples.
Accuracy is the fraction of all clips classified correctly. The confusion
matrix and class supports show which classes are being confused and how much
evidence each class contributes. None of these metrics should be interpreted
as calibrated confidence or as a measure of a person's internal emotional
state.
