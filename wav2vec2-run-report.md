# wav2vec2 GPU run: what was done

Run on 22 Nov 2026 by Navya, following `docs/wav2vec2-gpu-handoff.md`.

## Result

| | wav2vec2-base (this run) | MFCC + SVM (tuned) |
|---|---|---|
| Validation macro F1 | **0.7946** | 0.7041 |
| Validation accuracy | 0.7977 | n/a |

- Validation clips: 702 (13 actors). Train clips: 3,442 (64 actors).
- `test_evaluated: false`. No test actors were loaded, evaluated or used for any decision.
- Best epoch: 5 of 10. Early stopping triggered after epoch 8 (patience 3).
- Training time: 802 s (about 13.4 min).

### Per-class (validation, best epoch)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| anger | 0.771 | 0.945 | 0.849 | 182 |
| happiness | 0.951 | 0.643 | 0.767 | 182 |
| neutral | 0.695 | 0.846 | 0.763 | 156 |
| sadness | 0.837 | 0.764 | 0.799 | 182 |

### Confusion matrix (rows = true, columns = predicted)

| | anger | happiness | neutral | sadness |
|---|---|---|---|---|
| anger | 172 | 3 | 4 | 3 |
| happiness | 33 | 117 | 22 | 10 |
| neutral | 10 | 0 | 132 | 14 |
| sadness | 8 | 3 | 32 | 139 |

Main errors: happiness predicted as anger (33) or neutral (22); sadness predicted as neutral (32).

### Training loss by epoch

| Epoch | Train loss | Epoch time (s) |
|---|---|---|
| 1 | 1.218 | 116.7 |
| 2 | 0.862 | 79.2 |
| 3 | 0.692 | 78.2 |
| 4 | 0.592 | 82.0 |
| 5 | 0.546 | 117.9 |
| 6 | 0.484 | 112.6 |
| 7 | 0.447 | 104.5 |
| 8 | 0.424 | 109.0 |

Per-epoch validation macro F1 is in `validation_results.json` (`history`).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB VRAM
- Python 3.14.3, torch 2.11.0+cu128, transformers 4.48.3 (other packages from `ml/requirements-wav2vec.txt`)
- Model: `facebook/wav2vec2-base` at pinned revision `bfb1337a27fc4a418743753d6c965436844ab0a2`

## Exact command and settings

```powershell
.\.venv\Scripts\python.exe ml\train_wav2vec2_fixed.py --batch-size 4 --run-name wav2vec2_base_seed_20261108
```

All script defaults were used: max 10 epochs, patience 3, batch size 4, gradient accumulation 4, encoder LR 1e-5, head LR 1e-4, weight decay 0.01, seed 20261108, fp16 on. Batch size 4 fit in 8 GB, so no out-of-memory retries were needed.

## Steps followed

1. Cloned `SuryanshThapliyal/SER-CAT` (main) to `D:\SER-CAT`.
2. Placed the files exactly as the handoff layout says:
   - `work/crema_d/manifest_actor_intended.csv`
   - `work/crema_d/split_summary_seed_20261108.json`
   - `work/crema_d/data/AudioWAV/` (from `AudioWAV.zip`)
3. Checked the data before training. Nothing in it was edited:
   - All 7,442 manifest rows have their audio file present.
   - The label counts per split match the split summary exactly (train 3,442 / validation 702 / test 755 included).
   - No actor appears in more than one split.
   - The `1040_ITH_SAD_X.wav` filename and the excluded silent clip `1076_MTI_SAD_XX.wav` are as described.
4. Created a fresh `.venv` and installed CUDA PyTorch (cu128) and the pinned requirements. Confirmed `torch.cuda.is_available()` was True.
5. First attempt failed before training started (see code fix below). No results folder was created, so the default run name was reused.
6. Second attempt trained to completion with defaults.

## Code fix (the one change)

**Problem:** `Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", revision=bfb1337...)` fails with a 404. That pinned commit has no `preprocessor_config.json`, which was added to the Hugging Face repo in a later commit. Error:

```
OSError: facebook/wav2vec2-base does not appear to have a file named preprocessor_config.json.
```

**Fix:** the feature extractor is built in code with the same values as the current upstream `preprocessor_config.json`: feature_size 1, sampling_rate 16000, padding_value 0.0, padding right, do_normalize True, return_attention_mask False. Model weights and config still load from the pinned revision.

- The original `ml/train_wav2vec2.py` was **not modified**.
- The fixed copy is `ml/train_wav2vec2_fixed.py`; the diff is in `train_wav2vec2_fix.diff`.
- This is the only code change.

You may want to pin the revision that contains `preprocessor_config.json`, or keep the in-code extractor, in the repo.

## Warnings in the log (harmless)

- `FutureWarning: torch.cuda.amp.autocast(...) is deprecated` and the same for `GradScaler`. These are deprecation notices from the newer torch version and did not affect training.
- "under 8 GB VRAM" warning. The card reports exactly 8.0 GB, and batch size 4 fit.

## Files returned

| File | Notes |
|---|---|
| `validation_results.json` | Full results, history, versions, `test_evaluated: false` |
| `train_wav2vec2_base_seed_20261108_20261122_115632.txt` | Terminal log of the successful run |
| `train_wav2vec2_fix.diff` | The code change |
| `best_model.pt` | Best-epoch checkpoint (large; sent separately if needed) |

## Not done (per the handoff)

- No test-set evaluation and no test score.
- No hyperparameter changes or extra runs.
- No changes to the manifest, split, labels or audio.

## Question for Suryansh

Is this single default validation run everything you needed? Or do you want more, such as multiple seeds, tuning, or a test evaluation once you have set up the final protocol?
