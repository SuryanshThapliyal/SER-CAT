# wav2vec 2.0 emotion experiment plan

**Status:** Training script prepared; GPU training has not been run.

## Why this experiment

The current SVM receives 26 clip-level MFCC statistics. wav2vec 2.0 instead
starts from raw speech and a pretrained speech encoder, then learns an emotion
classification head from the actor-intended CREMA-D labels. The base checkpoint
was pretrained on English speech and expects 16 kHz audio; it was not pretrained
to predict emotions. Fine-tuning on CREMA-D is therefore required for this task.
The method may improve validation performance, but that is an empirical question,
not a guarantee.

The experiment uses `facebook/wav2vec2-base`, pinned to revision
`bfb1337a27fc4a418743753d6c965436844ab0a2`. The checkpoint is about 380 MB.
The convolutional feature extractor is frozen to reduce overfitting on the
limited labeled data; the transformer encoder and new four-class head are
fine-tuned. The training loss uses class weights computed from training labels.

## Data and model-selection boundary

Use the existing actor-disjoint training split (64 actors, 3,442 clips) to fit
the model and validation split (13 actors, 702 clips) for epoch selection and
early stopping. The script never reads the test split. It saves the best
validation checkpoint and the epoch-by-epoch validation metrics under
`work/crema_d/wav2vec2_experiments/`.

The validation split has already been used for classical model selection, so
the wav2vec 2.0 validation result is development evidence, not a final unbiased
estimate. The existing test actors have also been evaluated for the two
classical models. Do not use either set to make further model changes. After
the wav2vec experiment is locked, define a new actor-held-out evaluation before
making a final generalization claim.

## Hardware and setup

This script requires a CUDA GPU and exits if CUDA is unavailable; it will not
silently train on CPU. A GPU with about 8 GB of VRAM is recommended for the
default batch size of 4 with FP16. Reduce `--batch-size` to 2 or 1 if the GPU
runs out of memory. Training a base encoder is a substantial workload; no
training or checkpoint download has been started on the current machine.

The GPU machine needs the project code and these local dataset files:

- `work/crema_d/data/AudioWAV/`
- `work/crema_d/manifest_actor_intended.csv`
- `work/crema_d/split_summary_seed_20261108.json`

Install a CUDA-enabled PyTorch build compatible with that machine's CUDA driver
using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the project dependencies and run:

```powershell
uv pip install --python .venv\Scripts\python.exe -r ml\requirements-wav2vec.txt
.\.venv\Scripts\python.exe ml\train_wav2vec2.py
```

The first run downloads the pinned pretrained checkpoint from Hugging Face.
Default settings are 10 maximum epochs, patience 3, batch size 4, gradient
accumulation 4, encoder learning rate `1e-5`, and classification-head learning
rate `1e-4`. Only validation metrics are produced. Do not run this command on
the current non-GPU machine; send back `validation_results.json` and the GPU
model/VRAM details so the experiment can be reviewed before any test protocol
is planned.

## References

- Baevski et al., [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html), NeurIPS 2020.
- Hugging Face, [wav2vec2-base model card](https://huggingface.co/facebook/wav2vec2-base) and [audio classification guide](https://huggingface.co/docs/transformers/tasks/audio_classification).
