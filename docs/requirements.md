# Software Requirements Specification (SRS)

## 1. Purpose

This document defines the requirements for a research-oriented speech emotion recognition prototype. The prototype will analyze uploaded speech recordings and present estimated vocal emotions with confidence information and visual summaries.

The longer-term motivation is to explore how emotion-aware systems might help personalize AI customer-support responses. That application is future work; this prototype will detect and report emotion estimates only. It will not generate or personalize customer-support responses.

## 2. Goals and scope

### 2.1 In scope

- Upload and analyze an audio recording.
- Support a selected set of common audio formats, including MP3. The precise supported-format list will be finalized during design and implementation.
- Handle recordings of different durations up to five minutes.
- For recordings longer than five minutes, analyze only the first five minutes and tell the user that the recording was truncated for analysis.
- Present emotion estimates, model confidence, and a report with graphs.
- Evaluate the model on a documented dataset and held-out evaluation data, aiming for approximately **0.70 macro F1**. This is a target, not a guaranteed result.
- Handle corrupted, unreadable, or unsupported audio with a clear error message.

### 2.2 Out of scope for this prototype

- Live audio streaming or real-time chunk-by-chunk graph updates. These may be considered after the uploaded-file prototype is complete.
- Speaker identification, diarization, or analysis of different speakers separately.
- Mental-health diagnosis or other clinical conclusions.
- Perfect or definitive determination of a person's internal emotional state.
- Automated personalization or generation of customer-support responses.

## 3. Intended users and use

The intended users are people conducting research or exploring a business use case for speech emotion analysis. A user supplies a speech recording and reviews the system's estimated emotions, confidence information, and visual report.

The system reports estimates based on vocal audio. Its output must not be presented as certain knowledge of what a speaker feels.

## 4. Emotion labels and prediction target

The first model will predict the **emotion the actor was instructed to portray** in the dataset. This is an operational dataset label; it does not establish what the actor actually felt or what every listener would perceive.

For the initial CREMA-D experiment, use four actor-intended categories: **anger, happiness, sadness, and neutral**. CREMA-D does not provide a frustration label, so frustration is deferred. Do not relabel anger as frustration. Disgust and fear are present in CREMA-D but are outside the first model's target set.

Use the emotion encoded in the reviewed dataset manifest/filename as the target. The crowd-sourced `VoiceVote` annotations describe what raters perceived and shall not replace the actor-intended training target. Exclude the confirmed all-zero sad-labeled recording from training and document it in the data-quality manifest.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall let a user select and upload an audio file for analysis. |
| FR-2 | The system shall accept MP3 and a small set of other common audio formats selected during design. |
| FR-3 | The system shall reject unsupported or corrupted/unreadable files and show a useful error message. |
| FR-4 | The system shall analyze recordings up to five minutes long. |
| FR-5 | If a recording is longer than five minutes, the system shall analyze only its first five minutes and notify the user. |
| FR-6 | The system shall produce estimates for the supported emotion labels and expose model confidence in the report. |
| FR-7 | The system shall display an analysis report with graphs summarizing the result. The exact graph types will be specified during design. |
| FR-8 | The system shall communicate that emotion outputs are model estimates and not a diagnosis or definitive assessment of a person. |
| FR-9 | The system shall detect empty or near-silent audio and report that no usable speech was detected instead of assigning an emotion. |

## 6. Quality and evaluation requirements

### 6.1 Model evaluation

- **Primary target:** approximately **0.70 macro F1** across the four supported actor-intended categories on held-out-speaker evaluation data.
- Report the dataset, label definitions/mappings, data split method, and evaluation results so the score can be interpreted and reproduced.
- Include per-class results and a confusion matrix alongside macro F1 where practical. Accuracy alone is not the target because class imbalance can hide poor performance on less common emotions.
- Results are dependent on the dataset and evaluation setup; the target is not a promise of performance on arbitrary real-world recordings.

### 6.2 Processing and usability

- The prototype is file-based, so live-stream latency requirements do not apply.
- The system should provide understandable progress or status feedback during analysis, especially for longer recordings. The exact response-time target will be selected after the model and expected hardware are known.
- Errors for unsupported, corrupted, or unreadable files should explain what the user can do next where possible.

## 7. Constraints and assumptions

- The first prototype analyzes uploaded files only; streaming is deferred.
- Analysis is limited to the first five minutes of any longer file.
- The four initial emotion labels are decided; supported audio formats, model choice, and graph design remain design decisions.
- The model predicts dataset actor-intended categories, not a person's confirmed internal feeling or the emotion every listener will perceive.
- Model performance will be measured against a defined dataset and speaker-disjoint held-out split, not assumed to generalize to every speaker, recording condition, or customer-support interaction.

## 8. Acceptance criteria

The prototype is acceptable when:

1. A user can upload a valid file in each documented supported format and receive an analysis report.
2. The report displays the supported emotion estimates, confidence information, and graphs.
3. A recording longer than five minutes is limited to its first five minutes and the user is informed.
4. Unsupported, corrupted, or unreadable files produce a clear error rather than an unhandled failure.
5. Evaluation uses actor-intended labels and held-out speakers; the dataset, labels, exclusions, and method are documented; macro F1 is reported and a score around 0.70 is the project target.
6. The interface makes clear that the output is an estimate of vocal emotion, not a mental-health diagnosis or definitive reading of a person.
7. Empty or near-silent input returns a no-usable-speech message and no emotion prediction.

## 9. Open decisions for design

- Whether a later dataset can support a separately trained frustration category.
- Which common audio formats beyond MP3 will be supported.
- Which model and baseline will be compared, and what hardware will be used for evaluation.
- Which graphs best communicate the recording-level results.
- What file-analysis progress and response-time expectations are reasonable for the selected model and hardware.
- Whether optional transcription belongs in the prototype requirements or should remain a later extension.
