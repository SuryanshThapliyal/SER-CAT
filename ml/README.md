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

The test report, selected model, and cached features are stored under the
ignored `work/crema_d/` directory. The script refuses to overwrite an existing
test report to reduce accidental repeated tuning against the test set.

The original test result and its limits are documented in
[`docs/baseline-results.md`](../docs/baseline-results.md). The validation-only
tuning comparison is documented in [`docs/model-tuning.md`](../docs/model-tuning.md).

The fixed split keeps each actor in only one partition. The manifest marks the
all-zero recording as excluded and includes a reviewed correction for the
malformed intensity suffix. Do not use the test scores to change features or
model settings.
