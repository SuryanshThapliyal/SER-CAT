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
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt
uvicorn backend.app.main:app --reload
```

In another terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. Its development proxy forwards `/api` requests to `http://127.0.0.1:8000`. The API expects the locally trained model under `work/crema_d/baseline_results/`; set `SER_MODEL_PATH` to override it. `GET /api/v1/health` reports service/model state and `POST /api/v1/analyze` accepts a multipart field named `file`. WAV and FLAC decode through SoundFile; MP3 decoding depends on the available decoder backend. The API returns relative model scores, not calibrated probabilities. The frontend uses system fonts and does not load external assets. When local configuration is needed, copy `.env.example` to `.env`; keep credentials and machine-specific settings in `.env`. The `.env` file is ignored by Git.

## Data and generated files

Do not commit downloaded datasets, recordings, model weights, training outputs, or scratch inspection artifacts. The CREMA-D download used during inspection is kept under the ignored `work/` directory.
