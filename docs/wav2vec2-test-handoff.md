# wav2vec 2.0 test-only handoff

Use this after the wav2vec checkpoint has already been trained. This is an
inference-only run: **do not train, fine-tune, choose an epoch, or change model
settings.**

## Which split this evaluates

The script uses the original actor-disjoint CREMA-D split, seed `20261108`:

- Training: 64 actors, 3,442 included clips
- Validation: 13 actors, 702 included clips
- Test: **14 held-out actors, 755 included clips**

The wav2vec checkpoint was trained on the original training actors and its
epoch was selected on the original validation actors. The script loads only
the test audio for predictions. It checks the split summary and refuses to
overwrite an existing test report.

**Caveat:** these same test actors were already used to evaluate the SVM.
Therefore this wav2vec result is an exploratory comparison, not a fresh,
independent final benchmark. Do not use its score to tune or select models.

## What to provide to the other computer

Get the code from the GitHub commit that includes
`ml/evaluate_wav2vec2_test.py` and this handoff file. The following data and
checkpoint are ignored by Git and must be copied separately unless the GPU
computer still has them from the training run:

1. `best_model.pt` from the wav2vec training run. The local repacked checkpoint
   is at:
   `work/crema_d/wav2vec2_experiments/wav2vec2_base_seed_20261108/best_model.pt`
2. `work/crema_d/manifest_actor_intended.csv`
3. `work/crema_d/split_summary_seed_20261108.json`
4. `work/crema_d/data/AudioWAV/` containing the CREMA-D WAV files. If needed,
   transfer [AudioWAV.zip](../work/crema_d/data/AudioWAV.zip) and extract it so
   the resulting directory is exactly `work/crema_d/data/AudioWAV/`.

Preserve those paths. The manifest references audio paths relative to
`work/crema_d/`. Do not send or use the temporary fresh-split files; this run
uses only seed `20261108`.

## Environment and command

CUDA is optional for this evaluation. The script uses CUDA when available and
otherwise runs inference on CPU, which can take longer. This does not train
the model. Make sure PyTorch, Transformers, NumPy, SciPy, SoundFile, and
scikit-learn are installed. The wav2vec dependencies are listed in
`ml/requirements-wav2vec.txt`; PyTorch should already be installed from the
training environment, or installed with an appropriate CPU/CUDA build.

From the project root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe ml\evaluate_wav2vec2_test.py
```

The default checkpoint path is the one listed above. If it is elsewhere, use
an explicit path:

```powershell
.\.venv\Scripts\python.exe ml\evaluate_wav2vec2_test.py `
  --checkpoint "D:\path\to\best_model.pt"
```

The report is written to:

```text
work/crema_d/wav2vec2_experiments/wav2vec2_base_seed_20261108/test_results.json
```

It includes macro F1, accuracy, per-class precision/recall/F1, confusion
matrix, checkpoint epoch, split seed, device, and the exploratory-result caveat.
The script refuses to overwrite the report, helping ensure the test is run
once.

## Return to the project owner

Send back `test_results.json` and the terminal log. Include whether inference
used CPU or GPU and the GPU model/VRAM if applicable. Do not run the command a
second time, change settings based on the result, or report the number as an
independent benchmark without the caveat above.
