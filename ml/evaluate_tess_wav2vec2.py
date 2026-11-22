"""Exploratory cross-corpus evaluation of wav2vec 2.0 on TESS.

This script scores only the four labels shared with the CREMA-D checkpoint.
It does not train or tune the model. TESS has only two speakers, so its result
is an external diagnostic, not a speaker-diverse generalization benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from backend.app.models.baseline import LABELS
from backend.app.models.wav2vec2 import Wav2Vec2Predictor


DEFAULT_AUDIO_DIR = PROJECT_ROOT / "work" / "crema_d" / "external_test_samples" / "TESS"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "work"
    / "crema_d"
    / "wav2vec2_experiments"
    / "wav2vec2_base_seed_20261108"
    / "best_model.pt"
)
MAX_DURATION_SECONDS = 300
TESS_FILENAME = re.compile(
    r"^(?P<speaker>OAF|YAF)_.+_(?P<emotion>angry|happy|neutral|sad|disgust|fear|ps)\.wav$",
    re.IGNORECASE,
)
EMOTION_MAP = {
    "angry": "anger",
    "anger": "anger",
    "happy": "happiness",
    "happiness": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--manifest", type=Path, help="Optional CSV with filename, emotion, speaker columns")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, help="JSON report path (default: <audio-dir>/wav2vec2_four_class_evaluation.json)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def collect_rows(audio_dir: Path, manifest_path: Path | None) -> tuple[list[dict[str, str]], int]:
    if manifest_path is not None:
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            source_rows = list(csv.DictReader(stream))
        rows: list[dict[str, str]] = []
        skipped = 0
        for row in source_rows:
            filename = (row.get("filename") or row.get("file") or "").strip()
            raw_label = (row.get("emotion") or row.get("label") or "").strip().lower()
            label = EMOTION_MAP.get(raw_label)
            if not filename or label is None:
                skipped += 1
                continue
            rows.append({
                "filename": filename,
                "emotion": label,
                "speaker": (row.get("speaker") or "unknown").strip(),
            })
        return rows, skipped

    rows = []
    skipped = 0
    for path in sorted(audio_dir.rglob("*.wav")):
        match = TESS_FILENAME.match(path.name)
        if match is None:
            skipped += 1
            continue
        label = EMOTION_MAP.get(match.group("emotion").lower())
        if label is None:
            skipped += 1
            continue
        rows.append({
            "filename": str(path.relative_to(audio_dir)),
            "emotion": label,
            "speaker": match.group("speaker").upper(),
        })
    return rows, skipped


def resolve_audio(audio_dir: Path, filename: str) -> Path:
    candidate = (audio_dir / filename).resolve()
    root = audio_dir.resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate

    # TESS archives are sometimes wrapped in an extra directory. Resolve by
    # basename only when exactly one matching file exists.
    matches = list(audio_dir.rglob(Path(filename).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely locate {filename!r} below {audio_dir}")
    return matches[0]


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {args.audio_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.manifest is not None and not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in this PyTorch environment.")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)

    rows, skipped_count = collect_rows(args.audio_dir, args.manifest)
    if not rows:
        raise SystemExit("No WAV files with the four supported emotions were found.")
    for row in rows:
        resolve_audio(args.audio_dir, row["filename"])
    output_path = args.output or args.audio_dir / "wav2vec2_four_class_evaluation.json"

    print(f"Scoring {len(rows)} clips from {len({r['speaker'] for r in rows})} speaker(s) on {args.device}.", flush=True)
    print("Inference only: this script does not train, tune, or select a checkpoint.", flush=True)
    predictor = Wav2Vec2Predictor(args.checkpoint, device=args.device)
    predictions: list[str] = []
    targets: list[str] = []
    per_clip: list[dict[str, Any]] = []
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        path = resolve_audio(args.audio_dir, row["filename"])
        audio, sample_rate = librosa.load(
            path,
            sr=None,
            mono=True,
            duration=MAX_DURATION_SECONDS + 1,
        )
        audio = audio[: int(MAX_DURATION_SECONDS * sample_rate)]
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ValueError(f"Empty or invalid audio after decoding: {path}")
        if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) <= 1e-4:
            raise ValueError(f"Silent or too-quiet audio: {path}")

        scores = predictor.predict(audio, sample_rate)
        prediction = max(scores, key=scores.get)
        targets.append(row["emotion"])
        predictions.append(prediction)
        per_clip.append({
            "filename": row["filename"],
            "speaker": row["speaker"],
            "true_label": row["emotion"],
            "predicted_label": prediction,
            "scores": scores,
        })
        if index % 100 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} clips.", flush=True)

    score = {
        "experiment": "wav2vec2_tess_four_class_external_evaluation",
        "status": "demo_only_small_sample" if len(rows) < 100 else "exploratory_limited_speaker_diversity",
        "interpretation": (
            "This is not a final speaker-diverse benchmark. The four-clip sample is too small."
            if len(rows) < 100
            else "External diagnostic only: TESS has two speakers, so results do not establish broad speaker generalization."
        ),
        "dataset": "TESS",
        "target_labels": list(LABELS),
        "sample_count": len(rows),
        "speaker_count": len({row["speaker"] for row in rows}),
        "skipped_unmapped_files": skipped_count,
        "checkpoint": str(args.checkpoint),
        "device": predictor.device_name,
        "metrics": {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(LABELS), average="macro", zero_division=0)),
            "per_class": classification_report(
                targets,
                predictions,
                labels=list(LABELS),
                target_names=list(LABELS),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
                targets, predictions, labels=list(LABELS)
            ).tolist(),
        },
        "predictions": per_clip,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {score['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {score['metrics']['macro_f1']:.4f}")
    print(f"Saved report to {output_path}")
    print(score["interpretation"])


if __name__ == "__main__":
    main()
