# Exploratory audio observations

**Recorded:** 2026-11-22

**Purpose:** Track informal user trials that may reveal out-of-distribution
behavior. These observations are not a labeled evaluation set and were not
used to tune the models.

| Audio supplied | Format checked | User-reported context or label | Model output reported |
|---|---|---|---|
| AI-generated voice intended to sound angry | MP3 and WAV conversion of the same clip | Intended to sound angry | Happiness about 67% for both formats |
| Historical speech identified by the user as a Hitler speech | WAV | No verified emotion label assigned | Anger about 67%, happiness about 24%; other scores not recorded |
| Four TESS clips: `back` acted as angry, happy, neutral, and sad by talker OAF | WAV | TESS filename labels | Wav2vec 2.0 predicted anger above 93% for all four, as reported by the user |

The TESS clips are four different acted emotions from one speaker saying one
word. Their outputs indicate a possible cross-corpus generalization problem,
but four clips cannot estimate accuracy or explain its cause. The complete
TESS corpus has more examples but only two speakers; see
[`wav2vec2-external-evaluation.md`](wav2vec2-external-evaluation.md) for a
careful inference-only evaluation plan.

The user-provided AI and historical clips were not retained in the repository.
Their ground-truth emotion labels, source details, and full score vectors were
not independently verified. The similar MP3 and WAV outputs for the AI clip
make a format-decoding difference less likely for that clip, but do not
establish that formats behave identically in general.

The models were trained on acted CREMA-D speech and predict actor-intended
labels. These examples differ from the training data, so the outputs may
reflect domain mismatch. The website's percentages are normalized relative
model scores, not calibrated probabilities, and should not be treated as
reliable judgments of a speaker's internal emotion or the meaning of a speech.

Do not include these observations in training, hyperparameter selection, or
reported accuracy/F1 calculations. Any future evaluation needs a labeled set
and a protocol kept separate from model tuning.
