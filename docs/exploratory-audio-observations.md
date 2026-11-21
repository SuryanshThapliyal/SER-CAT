# Exploratory audio observations

**Recorded:** 2026-11-20  
**Purpose:** Note informal user trials that illustrate possible out-of-distribution behavior. These observations are not a labeled evaluation set and were not used to tune the model.

| Audio supplied | Format checked | User’s expectation or context | Model output reported |
|---|---|---|---|
| AI-generated voice intended to sound angry | MP3 and a WAV conversion of the same clip | The clip was intended to sound angry | Happiness about 67% in both formats; the user reported the same scores for MP3 and WAV. |
| Historical speech identified by the user as a Hitler speech | WAV | Historical political speech; no verified emotion label was assigned | Anger about 67%, happiness about 24%; other scores were not recorded. |

These are qualitative observations only. The files were not retained in the repository, and no ground-truth labels, controlled conversion procedure, exact source details, or full score vectors were recorded. The similar MP3 and WAV outputs for the first clip make a format-decoding difference less likely for that clip, but do not establish that MP3 and WAV behave identically in general.

The model was trained on acted CREMA-D speech and predicts four actor-intended categories from clip-level MFCC summaries. These user-provided examples differ from that training data, so the predictions may reflect domain mismatch. The displayed percentages are normalized relative model scores, not calibrated probabilities, and should not be interpreted as reliable judgments of a speaker's internal emotion or the historical meaning of a speech.

Do not include these examples in model training, hyperparameter selection, or reported accuracy/F1 calculations. A meaningful evaluation would require a deliberately collected set with defined labels and a protocol kept separate from model tuning.
