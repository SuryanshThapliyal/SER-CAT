# CREMA-D Dataset Inspection

**Status:** Initial local audit
**Dataset inspected:** Audio-only CREMA-D WAV archive and official processed rating tables
**Inspection date:** 2026-10-29

This is a dataset-quality and label audit, not a model result. No model was trained. The dataset was downloaded to `work/crema_d/` for this inspection and is not part of the application source.

## 1. Dataset overview

CREMA-D contains **7,442 short English audio clips from 91 actors**, recorded from 12 sentence prompts. The actor-intended categories are anger, disgust, fear, happy, neutral, and sad. Clips also encode acted intensity. The dataset paper describes audio, video, and multimodal human ratings; the local package inspected here contains audio WAV files and processed rating tables. [CREMA-D paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/) · [Official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D)

The inspected WAVs are approximately **1.27–5.01 seconds** long (median **2.50 seconds**, mean **2.54 seconds**). These are short utterances, not five-minute conversational recordings. They can support a short-clip model experiment but do not demonstrate behavior on long customer-support calls.

## 2. File and audio checks

| Check | Result |
|---|---:|
| WAV files in archive | 7,442 |
| File headers inspected | 7,442, including manual inspection of one unusual filename |
| Sample rate | 16 kHz for inspected files |
| Channels | Mono |
| Sample format | 16-bit PCM |
| Completely silent files (normalized RMS below 0.0001) | 1 |
| Files with a low-amplitude onset/offset over 250 ms at a 0.001 amplitude threshold | 2, including the completely silent file |
| Frustration-labeled files | 0 |

The fully silent file is `1076_MTI_SAD_XX.wav`: its waveform is all zeros, although the filename encodes intended **sad**. Its processed voice vote is **neutral**. A production training manifest should explicitly detect and exclude or quarantine this file; the application should return “no usable speech detected” for an empty or near-silent user upload rather than assigning an emotion.

One valid WAV is named `1040_ITH_SAD_X.wav`, with a one-character intensity suffix where the usual suffix is `XX`. The official `summaryTable.csv` refers to it as `1040_ITH_SAD_XX`. The file decodes as mono 16 kHz 16-bit PCM, so this is a **filename/metadata inconsistency**, not a decode failure. Join audio to annotations using a reviewed manifest and flag unmatched names; do not rely on an unvalidated filename parser.

`1064_IEO_DIS_MD.wav` has about 1.225 seconds below the chosen low-amplitude threshold at its start. That threshold is only a diagnostic; it does not prove the segment is silence. Inspect it with a voice activity detector or listening before trimming. Never treat silence itself as neutral or as an emotion label, and do not blindly strip all pauses because pauses can carry prosodic information.

## 3. Actor-intended label counts

Labels parsed from the clip names (the emotion the actor was directed to perform):

| Intended label | Clips |
|---|---:|
| Anger | 1,271 |
| Disgust | 1,271 |
| Fear | 1,271 |
| Happy | 1,271 |
| Neutral | 1,087 |
| Sad | 1,271 |
| **Total** | **7,442** |

The target categories **anger, happiness, sadness, and neutral** all exist as intended labels. **Frustration does not.** Do not rename anger as frustration: those are different categories and this dataset does not validate that mapping.

## 4. Human-perceived voice labels are different

The official processed `summaryTable.csv` has a `VoiceVote` field for the most-selected audio-only human rating. Across 7,442 clips, the table contains:

| Voice vote | Clips |
|---|---:|
| Neutral | 3,897 |
| Anger | 986 |
| Fear | 645 |
| Disgust | 547 |
| Sadness | 370 |
| Happiness | 353 |
| Multiple tied labels | 644 |

Comparing the actor-directed category in the filename with a **single-label** voice vote gives an exact match on **3,099 / 7,442 clips (41.6%)**. This audit calculation is not necessarily identical to the paper’s recognition metric. Per intended category, the exact match rates were anger 60.6%, neutral 95.7%, fear 32.0%, disgust 27.0%, happiness 26.0%, and sadness 16.4%.

This is a major target-definition decision. If the project predicts the actor’s intended performance label, it can use the balanced filename categories, but that target is not the same as what listeners perceive. If the project predicts perceived vocal emotion, the voice votes are more relevant, but the labels are strongly imbalanced and **644 clips have ties**. The project must choose and document one target before training. A reasonable next step is to compare (a) intended labels and (b) high-consensus voice-vote labels, while excluding ambiguous/tied ratings in the latter experiment and reporting how many examples remain. The CREMA-D paper reports that intended emotion recognition from audio alone is challenging and that rater disagreement is meaningful, not simply annotation corruption. [CREMA-D paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)

## 5. Leakage and evaluation plan

- Split by **actor ID**, never by random clip. All clips from one actor must stay in exactly one of train, validation, or test. Random clip splitting can leak a person’s voice into both training and test data and overstate generalization. [IEMOCAP evaluation study](https://sail.usc.edu/publications/files/Antoniou-ICASSP2023.pdf)
- CREMA-D uses only 12 sentence prompts. A speaker-disjoint split prevents actor overlap, but does not prove generalization to new wording. Record this limitation; consider a sentence-held-out diagnostic if the split allows it.
- Choose the speaker groups with a fixed seed, inspect per-class counts and actor demographics in each partition, then freeze the split before model tuning. Do not use the sorted-actor split in the scratch inspection output as the project’s final split.
- Do not report only one macro-F1 number. Report per-class precision/recall/F1, a confusion matrix, the split protocol, the target definition (intended vs perceived), and the excluded/ambiguous examples.
- A held-out-speaker score on this acted, short-utterance dataset is not a guarantee of performance on spontaneous customer speech, other languages, microphones, or recording conditions.

## 6. Recommendation

CREMA-D is large enough for the first controlled experiment and has enough actors to make a speaker-independent split feasible. It is **not a clean answer to the project’s target by itself**: frustration is absent, perceived labels differ substantially from intended labels, the perceived labels are imbalanced, and one silent file is mislabeled by its intended filename category.

Before training, decide whether the prototype’s target is **actor-intended expression** or **human-perceived vocal emotion**. For the project purpose (estimating what emotion a listener might hear), test perceived voice-vote labels with an explicit agreement threshold and retain neutral as a category; keep ambiguous votes out of hard-label training or evaluate a soft-label approach separately. Remove the confirmed all-zero file from emotion training and keep a manifest explaining why.

## 7. Scratch artifacts

The original archive, extracted WAVs, source rating tables, and inspection scripts are under `work/crema_d/`. They are scratch/data artifacts, not application code. They should not be committed to Git. The dataset release is associated with the Open Database License; review the official terms before redistribution or use in a business setting. [Official repository and license information](https://github.com/CheyneyComputerScience/CREMA-D)
