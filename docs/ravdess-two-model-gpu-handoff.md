# GPU handoff: train and compare both models on RAVDESS

## Goal

Train a four-emotion wav2vec 2.0 model and an MFCC + SVM model using the same
RAVDESS speech clips and the same actor-disjoint split. Return validation and
held-out test scores for both models so their performance can be compared on
the same dataset.

This is a RAVDESS-only experiment. Keep the existing CREMA-D models, reports,
and scores unchanged. Do not combine CREMA-D and RAVDESS in this experiment.

## First: pull the current GitHub repository

Before downloading or processing the dataset, get the current project code
from GitHub on the GPU computer. In the existing clone, run:

```powershell
git pull --ff-only origin main
git rev-parse HEAD
```

If the repository is not cloned yet, clone the project from its GitHub URL
first, then record the commit hash. Use this current version of the README,
training code, model label order, and this handoff. Do not start from an older
copy of the training scripts. Report the commit hash with the results.

The existing `ml/train_baseline.py` and `ml/train_wav2vec2.py` are
CREMA-D-specific. Use them as references for feature extraction, training,
metrics, and model architecture, but create separate RAVDESS scripts and
outputs. Do not alter the existing CREMA-D outputs or checkpoints.

## Dataset to download

Download only the official **audio-only speech** archive:

- `Audio_Speech_Actors_01-24.zip`
- 1,440 WAV files; approximately 208.5 MB on the official record
- Do not download the song archive or video/audio-video files

Official source: [RAVDESS on Zenodo](https://zenodo.org/records/1188976).
Retain its attribution and CC BY-NC-SA 4.0 license with any redistributed
files or reports.

The owner already has a 64-clip manual sample pack for trying individual
website uploads. **Do not train or evaluate on that 64-clip pack.** It is a
subset of RAVDESS and would duplicate recordings from the full archive.

## Four-class target and expected counts

Keep only the four RAVDESS speech emotion codes that map to the project's
existing label order (`anger`, `happiness`, `neutral`, `sadness`):

| RAVDESS emotion code | Project label | Count in complete speech archive |
|---|---|---:|
| 01 | neutral | 96 |
| 03 | happiness | 192 |
| 04 | sadness | 192 |
| 05 | anger | 192 |
| **Total included** |  | **672** |

Exclude code 02 (calm), 06 (fear), 07 (disgust), and 08 (surprise); those do
not map cleanly to our current four labels. Keep the audio-only modality 03,
speech vocal channel 01, and WAV files. The filename fields are
`modality-channel-emotion-intensity-statement-repetition-actor.wav`.

Before training, create an auditable manifest under `work/ravdess/` with the
source path, actor ID, original emotion code, mapped label, and split. Confirm
there are exactly 672 included clips with the counts above, all files exist
and decode to finite non-silent audio, and every actor is in exactly one
split.

## Fixed actor-disjoint split

Use the same exact split for both models. RAVDESS actor IDs are gender-balanced
by parity (odd IDs male, even IDs female). This fixed allocation keeps both
sexes in each partition and avoids choosing a split after seeing scores:

| Split | Actor IDs | Actors | Included clips | Neutral | Happiness | Sadness | Anger |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | 01, 02, 03, 04, 05, 06, 09, 10, 11, 14, 15, 18, 20, 21, 22, 23 | 16 | 448 | 64 | 128 | 128 | 128 |
| Validation | 07, 08, 12, 13 | 4 | 112 | 16 | 32 | 32 | 32 |
| Test | 16, 17, 19, 24 | 4 | 112 | 16 | 32 | 32 | 32 |

The allocation was generated with seed `20261122`, stratified by odd/even
actor IDs. Write the exact allocation into a split-summary JSON before model
training. Neither model may train on validation/test actors. Do not change
this split in response to either model's scores.

## Train both models

Implement separate scripts and place all outputs under `work/ravdess/`, for
example:

- `ml/build_ravdess_manifest.py`
- `ml/train_ravdess_svm.py`
- `ml/train_ravdess_wav2vec2.py`
- `ml/evaluate_ravdess_test.py`

Use the same manifest, four-class labels, and train/validation/test actors for
both models.

### MFCC + SVM

- Use the existing baseline feature definition: resample to 16 kHz mono,
  extract 13 MFCCs, summarize each with its mean and standard deviation for 26
  values per clip.
- Use the existing scaler and SVM validation search as a starting point:
  standard/robust scaling, linear/RBF kernels, `C` values 1/3/10, and the
  existing RBF gamma candidates. Use balanced class weights.
- Choose settings only by validation macro F1. Save the selected model and
  validation report before evaluating test once.

### wav2vec 2.0

- Fine-tune `facebook/wav2vec2-base` from the pinned pretrained base weights,
  not from the already CREMA-D-fine-tuned checkpoint.
- Use the project's 16 kHz mono preprocessing, the same four output labels,
  and the existing mean-pooling classifier approach. Freeze the convolutional
  feature extractor and use class-weighted loss to account for fewer neutral
  samples.
- Select the best epoch using validation macro F1 with early stopping. Save
  the best checkpoint and validation history in a new RAVDESS run directory.
- Use the manual in-code wav2vec feature-extractor settings from
  `train_wav2vec2_fixed.py` if the pinned Hugging Face revision has no
  `preprocessor_config.json`.

Wav2vec fine-tuning requires a CUDA GPU in this project. The SVM training and
MFCC extraction are CPU-only; they can run on the same machine without GPU
acceleration.

## Test boundary and comparison

Validation is for all choices: SVM scaler/kernel/parameters and wav2vec
checkpoint epoch. After both model setups are fixed, evaluate each model once
on the same 112 test clips from actors 16, 17, 19, and 24. Do not alter a model
or rerun test evaluation to improve its score.

Use macro F1 as the primary metric because neutral has half as many clips as
each other target class. Also report accuracy, per-class precision/recall/F1
and support, and a confusion matrix. Include validation and test results in
separate tables. State that these scores measure generalization to four
unseen RAVDESS actors and compare only the two models trained on this RAVDESS
split. Do not directly compare them as if they were the existing CREMA-D test
scores.

## Return to the project owner

Return:

- the GitHub commit hash and the new RAVDESS scripts;
- the complete manifest and fixed split-summary JSON;
- SVM validation and one-time test reports plus selected settings;
- wav2vec validation history, best checkpoint, and one-time test report;
- for each model: macro F1, accuracy, per-class metrics/supports, and
  confusion matrices on validation and test;
- training logs, exact commands, package versions, and GPU name/VRAM;
- a short note confirming that no actor crosses splits and that the 64-clip
  manual pack and original CREMA-D test files were not used.

The full RAVDESS archive is downloaded by the GPU machine's operator; it does
not need to be copied from the project owner's PC. The best wav2vec checkpoint
is a large output and should be transferred separately from GitHub.
