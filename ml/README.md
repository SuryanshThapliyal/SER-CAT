# ML baseline

The initial model predicts CREMA-D actor-intended labels: anger, happiness,
sadness, and neutral. Listener `VoiceVote` annotations are not training targets.

## Environment setup

Use Python 3.12 or newer. From the project root (`D:\dl\pbl`), create the
environment and install the pinned dependencies with `uv`:

```powershell
python -m venv .venv
uv --cache-dir work\uv_cache pip install --python .venv\Scripts\python.exe -r ml\requirements.txt
```

Alternatively, activate the environment and use pip:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt
```

## Training workflow

Run the training script from the project root:

```powershell
.\.venv\Scripts\python.exe ml\train_baseline.py
```

It checks the actor-disjoint manifest, extracts MFCCs and summarizes each clip
as 26 values (13 means plus 13 standard deviations), then compares standard
and robust scaling with linear and RBF SVMs. It tries `C` values 1, 3, and 10;
RBF candidates also try `gamma` values `scale`, 0.01, 0.03, and 0.1. It chooses
the candidate with the best validation macro F1 and saves it with validation
results. It does not evaluate the test set by default.

After model settings are locked, run the held-out test evaluation once:

```powershell
.\.venv\Scripts\python.exe ml\train_baseline.py --evaluate-test
```

The selected model, validation report, and cached features are stored under
the ignored `work/crema_d/` directory. The original test result is stored in
`baseline_results/test_results.json`; the tuned candidate's one-time result is
stored separately in `baseline_results/tuned_test_results.json`. The script
refuses to overwrite the tuned test report to prevent repeating evaluation on
the same held-out speakers.

The original test result and its limits are documented in
[`docs/baseline-results.md`](../docs/baseline-results.md). The validation-only
tuning comparison is documented in [`docs/model-tuning.md`](../docs/model-tuning.md).

## wav2vec 2.0 experiment

The next model experiment fine-tunes a pinned pretrained wav2vec 2.0 base
encoder on the training actors and uses validation actors for early stopping.
It never reads the held-out test actors. See
[`docs/wav2vec2-experiment.md`](../docs/wav2vec2-experiment.md) for the setup
and GPU-machine instructions. `ml/train_wav2vec2.py` deliberately exits when
CUDA is unavailable; do not run it on a CPU-only machine.

To score the saved wav2vec checkpoint on the original test split, run
`ml/evaluate_wav2vec2_test.py`. It performs inference only and uses CUDA when
available, otherwise CPU. Follow
[`docs/wav2vec2-test-handoff.md`](../docs/wav2vec2-test-handoff.md) for the
files and command. This is an exploratory result because the same test actors
were previously used for SVM evaluation; do not tune from this score.

The fixed split keeps each actor in only one partition. The manifest marks the
all-zero recording as excluded and includes a reviewed correction for the
malformed intensity suffix. Do not use the test scores to change features or
model settings.
