# GPU handoff: fine-tune the CREMA-D wav2vec model on RAVDESS

## Goal

Continue training the existing wav2vec 2.0 checkpoint fine-tuned on CREMA-D,
using only the RAVDESS training actors. The purpose is to test whether the
CREMA-D-learned emotion model transfers better to RAVDESS than the previous
RAVDESS run that started from the generic pretrained wav2vec base.

This is a transfer-learning experiment. Do **not** initialize from the generic
`facebook/wav2vec2-base` checkpoint, and do not use the previous RAVDESS
fine-tuned checkpoint as initialization. Start from the CREMA-D `best_model.pt`
listed below. Save every output in a new run directory; keep all earlier
checkpoints and reports unchanged.

## First: pull the current GitHub repository

On the GPU computer, first update the project clone:

```powershell
git pull --ff-only origin main
git rev-parse HEAD
```

If there is no clone yet, clone the project from its GitHub URL, then record
the commit hash. The project owner will send this handoff separately if it has
not yet been pushed.

The RAVDESS training scripts returned from the previous run are currently
provided separately under the project's `scripts/` folder. The previous GPU
run report's commit hash does not include those scripts. Review them and place
or adapt them in `ml/` before running. The existing `ml/train_wav2vec2.py` is
CREMA-D-specific; do not overwrite it.

## Files needed

1. **CREMA-D fine-tuned checkpoint:**
   `work/crema_d/wav2vec2_experiments/wav2vec2_base_seed_20261108/best_model.pt`
   Copy this file separately from GitHub; it is large and gitignored.
2. **RAVDESS full speech archive:** download
   `Audio_Speech_Actors_01-24.zip` from the
   [official Zenodo record](https://zenodo.org/records/1188976). Download the
   audio-only speech archive (1,440 WAVs, about 208.5 MB), not the song or video
   archives. Extract to `work/ravdess/data/Actor_01/` through `Actor_24/`.
3. **RAVDESS manifest and split:** use the reviewed files from the previous
   run or regenerate them from the complete archive using the scripts provided
   separately. Expected files:
   `work/ravdess/manifest_ravdess_speech_4class.csv` and
   `work/ravdess/split_summary_ravdess_seed_20261122.json`.
4. The CREMA-D WAV files are not needed to fine-tune on RAVDESS. If they are
   already available on the GPU machine, the CREMA-D validation partition may
   be scored as a **retention monitor** to check for catastrophic forgetting.
   Do not load or evaluate CREMA-D test actors for this experiment.

The previous 64-clip RAVDESS manual upload pack is not needed. It is drawn
from the same RAVDESS corpus; its clips are already included in the full
manifest and fall in train, validation, or test according to their actors.

## RAVDESS data and split

Use the same four output labels and actor-disjoint split as the previous
RAVDESS comparison:

| Split | Actor IDs | Included clips |
|---|---|---:|
| Train | 01, 02, 03, 04, 05, 06, 09, 10, 11, 14, 15, 18, 20, 21, 22, 23 | 448 |
| Validation | 07, 08, 12, 13 | 112 |
| Test | 16, 17, 19, 24 | 112 |

Use only the four mapped classes: code 01 neutral, 03 happy -> happiness, 04
sad -> sadness, and 05 angry -> anger. Exclude calm, fearful, disgust, and
surprise. No actor or clip from validation/test may enter fine-tuning.

The current target-class training distribution is 128 anger, 128 happiness,
64 neutral, and 128 sadness. Keep the existing class-weighted loss. Preserve
the existing RAVDESS SVM and pretrained-base wav2vec reports as comparison
baselines.

## Transfer-fine-tuning requirements

Create a separate script, for example
`ml/finetune_cremad_wav2vec2_on_ravdess.py`, that:

1. Loads the CREMA-D `best_model.pt` state dict into the same four-class model
   architecture using strict key matching.
2. Verifies model ID/revision, sample rate, and label order before loading.
   Keep label order `anger`, `happiness`, `neutral`, `sadness`.
3. Resamples RAVDESS audio from 48 kHz mono to 16 kHz with the same
   polyphase-resampling and feature-extractor settings as the previous runs.
4. Continues training the encoder and classifier head on the 448 RAVDESS
   training clips. Keep the convolutional feature extractor frozen and use
   class-weighted cross-entropy. Use a lower learning rate than the base-model
   run (for example encoder `1e-6`, head `1e-5`), a maximum of 15 epochs, and
   patience 3. If exploring multiple learning rates, select only by validation
   macro F1 and record the full search.
5. Saves the best validation checkpoint and validation history under a new
   directory, such as
   `work/ravdess/wav2vec2_experiments/cremad_init_ravdess_seed_20261122/`.
   Include the initialization checkpoint path/hash and transfer settings in
   the report. Never overwrite the base-initialized RAVDESS checkpoint.

Fine-tuning wav2vec requires a CUDA GPU in this project. The SVM does not need
to be retrained for this experiment: its RAVDESS validation/test scores are
already available for comparison. The wav2vec training script must refuse to
fall back silently to CPU.

## Evaluation boundary

Use RAVDESS validation macro F1 for early stopping and any learning-rate
selection. The earlier RAVDESS test actors and results have already been
examined, so a new score on those same 112 clips is a **repeat exploratory
comparison**, not a fresh blind benchmark. If performing one comparison,
freeze all settings using validation first, evaluate the test once, label the
result with this caveat, and do not tune or rerun based on the test score.
For a new independent final claim, collect a new labeled test corpus with
multiple speakers before training and keep it untouched.

If CREMA-D validation audio is present, report its macro F1 before and after
RAVDESS fine-tuning as a retention check. Do not use the CREMA-D test split.

## Return these results

- Git commit hash and the new transfer-fine-tuning script.
- CREMA-D initialization checkpoint path and SHA-256 hash.
- RAVDESS validation history, selected epoch, and best checkpoint.
- RAVDESS validation metrics, and the repeat exploratory RAVDESS test metrics
  if evaluated: macro F1, accuracy, per-class precision/recall/F1/support, and
  confusion matrix.
- Comparison table with the previous RAVDESS-from-base wav2vec model and the
  existing RAVDESS SVM on the same split.
- Optional CREMA-D validation retention metrics; no CREMA-D test metrics.
- Training log, exact command, package versions, GPU model, and VRAM.

The RAVDESS wav2vec test checkpoint from the previous run is not present in
the files currently copied back. It is not needed to initialize this run; the
CREMA-D `best_model.pt` is the required starting checkpoint. The old RAVDESS
test report is sufficient to show the previous model's scores.
