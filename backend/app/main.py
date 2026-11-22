"""FastAPI service for uploaded-file emotion analysis."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from threading import Lock

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .models.baseline import LABELS, SAMPLE_RATE, EmotionPredictor, MfccSvmPredictor
from .models.wav2vec2 import Wav2Vec2Predictor


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 300
ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVM_MODEL_PATH = Path(os.getenv(
    "SER_SVM_MODEL_PATH",
    os.getenv(
        "SER_MODEL_PATH",
        PROJECT_ROOT
        / "work" / "crema_d" / "baseline_results" / "best_svm.joblib",
    ),
))
WAV2VEC2_MODEL_PATH = Path(os.getenv(
    "SER_WAV2VEC2_MODEL_PATH",
    PROJECT_ROOT
    / "work" / "crema_d" / "wav2vec2_experiments"
    / "wav2vec2_base_seed_20261108" / "best_model.pt",
))
WAV2VEC2_DEVICE = os.getenv("SER_WAV2VEC_DEVICE", "cpu").strip().lower()

MODEL_DISPLAY_NAMES = {"svm": "MFCC + SVM", "wav2vec2": "wav2vec 2.0"}
MODEL_NAMES = frozenset(MODEL_DISPLAY_NAMES)

app = FastAPI(title="SER-CAT API", version="0.2.0")
predictors: dict[str, EmotionPredictor] = {}
predictor_lock = Lock()
logger = logging.getLogger(__name__)


def get_predictor(name: str) -> EmotionPredictor:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    cached = predictors.get(name)
    if cached is not None:
        return cached
    with predictor_lock:
        cached = predictors.get(name)
        if cached is not None:
            return cached
        if name == "svm":
            loaded: EmotionPredictor = MfccSvmPredictor(SVM_MODEL_PATH)
        else:
            loaded = Wav2Vec2Predictor(WAV2VEC2_MODEL_PATH, device=WAV2VEC2_DEVICE)
        predictors[name] = loaded
        return loaded


@app.on_event("startup")
def load_default_model() -> None:
    # Keep the existing SVM available immediately. The larger wav2vec model is
    # loaded lazily on its first request so API startup stays quick.
    try:
        get_predictor("svm")
    except FileNotFoundError:
        # Health remains available for setup diagnostics; analysis returns 503.
        pass


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    svm_ready = "svm" in predictors
    wav2vec_loaded = "wav2vec2" in predictors
    wav2vec_available = WAV2VEC2_MODEL_PATH.is_file()
    return {
        "status": "ok",
        "model": "ready" if svm_ready else "unavailable",
        "models": {
            "svm": "ready" if svm_ready else "unavailable",
            "wav2vec2": (
                "ready" if wav2vec_loaded
                else "available" if wav2vec_available
                else "unavailable"
            ),
        },
    }


@app.post("/api/v1/analyze")
async def analyze(
    file: UploadFile = File(...),
    model: str = Form(default="svm"),
) -> dict[str, object]:
    if model not in MODEL_NAMES:
        raise HTTPException(422, "Choose either the svm or wav2vec2 model.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload a WAV, FLAC, or MP3 audio file.")
    try:
        predictor = await run_in_threadpool(get_predictor, model)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as exc:
        logger.exception("Could not load requested emotion model: %s", model)
        if model == "wav2vec2":
            message = (
                "The wav2vec 2.0 model is unavailable. Check its checkpoint path, "
                "install PyTorch and the wav2vec dependencies, and confirm the "
                "configured inference device is available."
            )
        else:
            message = "The SVM model is unavailable. Run ml/train_baseline.py first."
        raise HTTPException(503, message) from exc

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file exceeds the 25 MB upload limit.")
    try:
        # Decode at most one second beyond the analysis limit so truncation can
        # be detected even for formats without reliable container metadata.
        audio, source_rate = librosa.load(
            io.BytesIO(raw), sr=None, mono=True, duration=MAX_DURATION_SECONDS + 1
        )
    except Exception as exc:
        raise HTTPException(422, "The audio could not be decoded. Check the file and try WAV, FLAC, or MP3.") from exc
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise HTTPException(422, "No usable audio was found in the file.")
    if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) <= 1e-4:
        raise HTTPException(422, "No usable speech was detected because the audio is silent or too quiet.")

    decoded_duration = len(audio) / source_rate
    truncated = decoded_duration > MAX_DURATION_SECONDS
    audio = audio[: int(MAX_DURATION_SECONDS * source_rate)]
    analyzed_duration = len(audio) / source_rate
    try:
        scores = await run_in_threadpool(predictor.predict, audio, source_rate)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("Requested model failed during inference: %s", model)
        raise HTTPException(500, "The selected model could not analyze this recording.") from exc
    prediction = max(scores, key=scores.get)
    return {
        "prediction": prediction,
        "scores": scores,
        "score_type": "relative_model_score",
        "model": model,
        "model_name": MODEL_DISPLAY_NAMES[model],
        "inference_device": getattr(predictor, "device_name", "CPU"),
        "labels": list(LABELS),
        "duration_seconds": float(analyzed_duration),
        "truncated": truncated,
        "sample_rate_hz": SAMPLE_RATE,
        "notice": "Scores are relative model outputs, not calibrated probabilities. Estimates are not a definitive reading of feelings or a diagnosis.",
    }
