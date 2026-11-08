# SER-CAT

Speech Emotion Recognition project focused on learning and applying software engineering practices alongside an ML prototype.

## Prototype scope

The first version will analyze uploaded speech recordings, estimate supported vocal emotion categories, and present the result with model scores and visual summaries. It is a research prototype: its output is an estimate, not a definitive reading of a person or a mental-health diagnosis.

Live streaming, speaker identification/diarization, and automatic personalization of customer-support responses are deferred.

## Project documents

- [Software Requirements Specification](docs/requirements.md)
- [Proposed Architecture and Design](docs/architecture.md)
- [CREMA-D Dataset Inspection](docs/crema-d-inspection.md)

## Current dataset findings

The first model will predict actor-intended emotion labels from CREMA-D: anger, happiness, sadness, and neutral. Listener `VoiceVote` ratings are not the training target. Frustration is deferred because CREMA-D does not label it. See the dataset inspection report for the silent clip exclusion and speaker-disjoint evaluation requirements.

## Local setup

Application code and dependency setup have not been added yet. When local configuration is needed, copy `.env.example` to `.env`; keep credentials and machine-specific settings in `.env`. The `.env` file is ignored by Git.

## Data and generated files

Do not commit downloaded datasets, recordings, model weights, training outputs, or scratch inspection artifacts. The CREMA-D download used during inspection is kept under the ignored `work/` directory.
