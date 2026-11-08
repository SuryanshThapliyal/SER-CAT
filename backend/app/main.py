"""FastAPI service for uploaded-file emotion analysis."""

from __future__ import annotations

import io
import os
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .models.baseline import LABELS, SAMPLE_RATE, MfccSvmPredictor


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 300
ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3"}
MODEL_PATH = Path(os.getenv(
    "SER_MODEL_PATH",
    Path(__file__).resolve().parents[2]
    / "work" / "crema_d" / "baseline_results" / "best_svm.joblib",
))

app = FastAPI(title="SER-CAT API", version="0.1.0")
predictor: MfccSvmPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    try:
        predictor = MfccSvmPredictor(MODEL_PATH)
    except FileNotFoundError:
        # Keep health available for setup diagnostics; analyze returns a clear 503.
        predictor = None


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ready" if predictor else "unavailable"}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload a WAV, FLAC, or MP3 audio file.")
    if predictor is None:
        raise HTTPException(503, "The model is unavailable. Set up the local CREMA-D model first.")

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
    scores = predictor.predict(audio, source_rate)
    prediction = max(scores, key=scores.get)
    return {
        "prediction": prediction,
        "scores": scores,
        "score_type": "relative_model_score",
        "labels": list(LABELS),
        "duration_seconds": float(analyzed_duration),
        "truncated": truncated,
        "sample_rate_hz": SAMPLE_RATE,
        "notice": "Estimates reflect vocal patterns and are not a definitive reading of a person's feelings or a diagnosis.",
    }
