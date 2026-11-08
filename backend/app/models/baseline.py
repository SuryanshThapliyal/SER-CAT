"""Inference adapter for the trained MFCC + SVM baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
import joblib


LABELS = ("anger", "happiness", "neutral", "sadness")
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_MFCC = 13
N_MELS = 40


class EmotionPredictor(Protocol):
    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]: ...


class MfccSvmPredictor:
    """Loads a trained pipeline once and returns normalized class scores."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run ml/train_baseline.py first."
            )
        self._model = joblib.load(model_path)

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=N_MFCC,
            n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        features = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1)))
        decision = np.asarray(self._model.decision_function(features[None, :]))
        # SVC decision values are margins, not calibrated probabilities. Softmax
        # provides displayable relative scores; it must not be called confidence.
        if decision.ndim != 2 or decision.shape[1] != len(LABELS):
            raise ValueError("The loaded model returned an unexpected score shape.")
        shifted = decision[0] - np.max(decision[0])
        relative = np.exp(shifted)
        relative /= relative.sum()
        scores = {label: float(score) for label, score in zip(LABELS, relative)}
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError("The model returned non-finite scores.")
        return scores
