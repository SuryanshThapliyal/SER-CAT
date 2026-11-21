# GPU computer handoff: CREMA-D wav2vec 2.0 experiment

Give this file, the small set of project files listed below, and the CREMA-D
files to the person or coding assistant on the GPU computer. This document is
written so they can run the experiment without relying on this chat.

## Objective

Fine-tune the pinned pretrained `facebook/wav2vec2-base` checkpoint for the
four CREMA-D actor-intended emotion labels: `anger`, `happiness`, `neutral`,
and `sadness`. Compare the resulting **validation macro F1** with the tuned
MFCC + SVM validation macro F1 of `0.7041`.

This is a validation experiment only. Do not load, evaluate, tune against, or
report a new score for the test actors. The existing test split has already
been inspected for prior classical models, and this project is reserving a
fresh final evaluation protocol for later.

## Files to copy to the GPU computer

Create a project folder with this layout (the folder name itself can differ):

```text
speech-emotion-project/
  ml/
    train_wav2vec2.py
    requirements-wav2vec.txt
  work/
    crema_d/
      manifest_actor_intended.csv
      split_summary_seed_20261108.json
      data/
        AudioWAV/
          ... all CREMA-D WAV files referenced by the manifest ...
```

Copy these items from the existing project:

- `ml/train_wav2vec2.py`
- `ml/requirements-wav2vec.txt`
- `work/crema_d/manifest_actor_intended.csv`
- `work/crema_d/split_summary_seed_20261108.json`
- The complete `work/crema_d/data/AudioWAV/` directory
- This handoff document, `docs/wav2vec2-gpu-handoff.md`, for instructions

Keep the directory structure and filenames exactly as shown. The manifest
contains paths relative to `work/crema_d/`. Do not regenerate or edit the
manifest, labels, actor assignments, split summary, or audio files. The
all-zero clip is excluded in the reviewed manifest, and the reviewed filename
correction is already reflected there.

You do not need to copy the web frontend, `node_modules`, the original
`.venv`, `work/uv_cache`, cached MFCC features, or prior experiment outputs.
Do not send the dataset publicly; transfer it only through a method permitted
by its terms and the project owner.

## GPU requirement and environment

This run requires an NVIDIA GPU with a working CUDA-enabled PyTorch build.
About 8 GB of VRAM is recommended for the default batch size. The training
script intentionally exits if CUDA is unavailable; do not run training on CPU.
The first successful run downloads the pinned model checkpoint (about 380 MB)
from Hugging Face, so the machine also needs internet access for that download.

Use Python 3.12 or newer. Install PyTorch using the official selector for the
GPU machine and its driver: <https://pytorch.org/get-started/locally/>. Choose
the **CUDA** build, not a CPU-only build. Then, from the project root, install
the remaining pinned dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
uv pip install --python .venv\Scripts\python.exe -r ml\requirements-wav2vec.txt
```

If `uv` is not available, activate the environment and use pip:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements-wav2vec.txt
```

### Linux or macOS shell (NVIDIA CUDA is Linux/Windows in this workflow)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements-wav2vec.txt
```

Install the CUDA-enabled PyTorch build before installing the requirements
file. The requirements file intentionally does not pin PyTorch because the
correct CUDA wheel depends on the GPU machine. Check CUDA before training:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Proceed only when `CUDA available: True` and a GPU name is printed. If it says
`False`, stop and fix the CUDA/PyTorch environment; do not substitute CPU
training.

## Run the experiment

Start with the script defaults (maximum 10 epochs, early-stopping patience 3,
batch size 4, gradient accumulation 4, encoder learning rate `1e-5`, and
classification-head learning rate `1e-4`). From the project root:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe ml\train_wav2vec2.py
```

### Linux shell

```bash
.venv/bin/python ml/train_wav2vec2.py
```

The script prints the GPU name and epoch progress. It saves outputs under:

```text
work/crema_d/wav2vec2_experiments/wav2vec2_base_seed_20261108/
  best_model.pt
  validation_results.json
```

The JSON contains the best validation macro F1, accuracy, per-class metrics,
confusion matrix, epoch history, package versions, GPU name, VRAM, elapsed
time, and an explicit `test_evaluated: false` field. The checkpoint is the
model from the best validation macro F1 epoch.

### If GPU memory runs out

Do not change model architecture, split, labels, learning rates, or evaluation
rules to work around memory. Retry with a smaller batch size and a unique run
name so the failed attempt remains distinguishable. For example:

```powershell
.\.venv\Scripts\python.exe ml\train_wav2vec2.py --batch-size 2 --run-name wav2vec2_base_seed_20261108_bs2
```

On Linux:

```bash
.venv/bin/python ml/train_wav2vec2.py --batch-size 2 --run-name wav2vec2_base_seed_20261108_bs2
```

If batch size 2 still runs out of memory, try batch size 1 with another unique
run name. Record the arguments used for the successful run. Do not delete or
overwrite an existing experiment folder.

## Instructions for the coding assistant on the GPU computer

1. Read this entire handoff file before running commands.
2. Confirm the expected files exist and the project root contains the layout
   above. Do not alter the data, manifest, split, labels, or experiment code
   unless a concrete runtime error requires a code fix.
3. Check that a CUDA-enabled PyTorch build sees the GPU. If CUDA is unavailable,
   stop and report the exact error; do not run on CPU.
4. Install dependencies as documented and run the script once with defaults.
5. If there is an out-of-memory error, retry only by lowering batch size to 2,
   then 1 if necessary, using a unique run name each time. Report all attempts.
6. Do not inspect or evaluate test actors, do not run any test-evaluation
   command, and do not claim a test score. Do not choose new model settings
   based on test results.
7. Do not modify the original project on the owner's machine. If a code fix is
   unavoidable, preserve the original file, provide a clear diff and reason,
   and return the modified script separately for review.
8. When training completes, verify that `validation_results.json` exists and
   that it reports `test_evaluated` as `false`.

## Return these items to the project owner

Copy back or send:

- The complete `validation_results.json` from the successful run.
- `best_model.pt` if practical (large file; keep it with the result if the
  owner wants to use the model later).
- The GPU model and VRAM amount.
- The exact command and any non-default arguments used.
- The terminal log, especially epoch metrics, warnings, and errors.
- Any code changes as a separate diff; do not silently replace project files.

The result will be reviewed against the SVM's validation macro F1 of `0.7041`.
A validation improvement is useful development evidence, not a final unbiased
test estimate. No final test evaluation is authorized by this experiment.
