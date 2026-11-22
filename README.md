# SER-CAT

Speech Emotion Recognition project focused on learning and applying software engineering practices alongside an ML prototype.

## Prototype scope

The first version will analyze uploaded speech recordings, estimate supported vocal emotion categories, and present the result with model scores and visual summaries. It is a research prototype: its output is an estimate, not a definitive reading of a person or a mental-health diagnosis.

Live streaming, speaker identification/diarization, and automatic personalization of customer-support responses are deferred.

## Project documents

- [Software Requirements Specification](docs/requirements.md)
- [Proposed Architecture and Design](docs/architecture.md)
- [CREMA-D Dataset Inspection](docs/crema-d-inspection.md)
- [MFCC + SVM Baseline Results](docs/baseline-results.md)
- [Validation tuning experiment](docs/model-tuning.md)
- [Exploratory audio observations](docs/exploratory-audio-observations.md)
- [wav2vec 2.0 experiment plan](docs/wav2vec2-experiment.md)
- [wav2vec 2.0 test-only handoff](docs/wav2vec2-test-handoff.md)

## Current dataset findings

The first model predicts actor-intended emotion labels from CREMA-D: anger, happiness, sadness, and neutral. Listener `VoiceVote` ratings are not the training target. Frustration is deferred because CREMA-D does not label it. The initial held-out-speaker MFCC + SVM baseline scored 0.623 macro F1; details and limitations are in the baseline report.

## Local setup

The backend API is in `backend/` and the React frontend is in `frontend/`. ML environment setup is documented in `ml/README.md`. Start the API from the project root:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --env-file .env --reload
```

The API offers both **MFCC + SVM** and **wav2vec 2.0** in the website model
selector. PyTorch is installed separately as a CPU build for this GPU-free
localhost setup using the [official PyTorch 2.11.0 install instructions](https://pytorch.org/get-started/previous-versions/);
wav2vec inference may be slower on CPU. Its checkpoint and the SVM artifact
stay under ignored `work/crema_d/` paths. The wav2vec model loads the first
time that option is selected; the first load may fetch the pinned model
configuration from Hugging Face, while learned weights come from the local
checkpoint.

In another terminal, start the frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite. Its development proxy forwards `/api` requests to `http://127.0.0.1:8000`. The API loads the SVM artifact from `work/crema_d/baseline_results/` and the wav2vec checkpoint from `work/crema_d/wav2vec2_experiments/wav2vec2_base_seed_20261108/`; copy `.env.example` to `.env` to configure either path. The website lets you choose which model analyzes each upload. Wav2vec loads on its first request and defaults to CPU through `SER_WAV2VEC_DEVICE=cpu`. `GET /api/v1/health` reports both model states and `POST /api/v1/analyze` accepts multipart fields `file` and `model`. WAV and FLAC decode through SoundFile; MP3 decoding depends on the available decoder backend. The API returns relative model scores, not calibrated probabilities. The frontend uses system fonts and does not load external assets. Keep credentials and machine-specific settings in `.env`; `.env` is ignored by Git.

## Data and generated files

Do not commit downloaded datasets, recordings, model weights, training outputs, or scratch inspection artifacts. The CREMA-D download used during inspection is kept under the ignored `work/` directory.
