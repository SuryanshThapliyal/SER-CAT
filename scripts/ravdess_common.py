"""Shared constants and checks for the RAVDESS four-class experiment.

Everything for this experiment lives under ``work/ravdess/``. The CREMA-D
files under ``work/crema_d/`` are never read or written by the RAVDESS scripts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAVDESS_ROOT = PROJECT_ROOT / "work" / "ravdess"
DATA_DIR = RAVDESS_ROOT / "data"
MANIFEST_PATH = RAVDESS_ROOT / "manifest_ravdess_speech_4class.csv"
SPLIT_SUMMARY_PATH = RAVDESS_ROOT / "split_summary_ravdess_seed_20261122.json"
SVM_DIR = RAVDESS_ROOT / "svm"
WAV2VEC2_DIR = RAVDESS_ROOT / "wav2vec2_experiments"
DEFAULT_WAV2VEC2_RUN = "wav2vec2_base_ravdess_seed_20261122"

ARCHIVE_NAME = "Audio_Speech_Actors_01-24.zip"
ARCHIVE_SOURCE = "https://zenodo.org/records/1188976"
LICENSE = "CC BY-NC-SA 4.0"
ATTRIBUTION = (
    "Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of "
    "Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. "
    "https://doi.org/10.1371/journal.pone.0196391. Zenodo record 1188976."
)

# Same label order as the CREMA-D models.
LABELS = ("anger", "happiness", "neutral", "sadness")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_SEED = 20261122

# RAVDESS emotion code -> project label. Codes 02 calm, 06 fear, 07 disgust,
# 08 surprise are excluded.
EMOTION_CODE_TO_LABEL = {"01": "neutral", "03": "happiness", "04": "sadness", "05": "anger"}
EMOTION_CODE_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

# Fixed actor allocation from the handoff (seed 20261122, stratified by
# odd = male / even = female actor IDs). Do not change after seeing scores.
SPLIT_ACTORS = {
    "train": ("01", "02", "03", "04", "05", "06", "09", "10", "11", "14",
              "15", "18", "20", "21", "22", "23"),
    "validation": ("07", "08", "12", "13"),
    "test": ("16", "17", "19", "24"),
}
ACTOR_TO_SPLIT = {actor: split for split, actors in SPLIT_ACTORS.items() for actor in actors}

EXPECTED_TOTAL_FILES = 1440
EXPECTED_INCLUDED_TOTAL = 672
EXPECTED_LABEL_COUNTS = {"neutral": 96, "happiness": 192, "sadness": 192, "anger": 192}
EXPECTED_SPLIT_COUNTS = {
    "train": {"actors": 16, "clips": 448, "neutral": 64, "happiness": 128, "sadness": 128, "anger": 128},
    "validation": {"actors": 4, "clips": 112, "neutral": 16, "happiness": 32, "sadness": 32, "anger": 32},
    "test": {"actors": 4, "clips": 112, "neutral": 16, "happiness": 32, "sadness": 32, "anger": 32},
}


def check_split_definition() -> None:
    all_actors = [actor for actors in SPLIT_ACTORS.values() for actor in actors]
    if sorted(all_actors) != [f"{i:02d}" for i in range(1, 25)]:
        raise ValueError("Split must assign each of actors 01-24 exactly once.")
    for split, actors in SPLIT_ACTORS.items():
        if len(actors) != EXPECTED_SPLIT_COUNTS[split]["actors"]:
            raise ValueError(f"Wrong actor count in {split}.")
        sexes = {int(actor) % 2 for actor in actors}
        if sexes != {0, 1}:
            raise ValueError(f"{split} must contain both male (odd) and female (even) actors.")


def resolve_audio_path(relative_path: str) -> Path:
    root = RAVDESS_ROOT.resolve()
    path = (RAVDESS_ROOT / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes work/ravdess: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_included_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load included manifest rows and re-verify every split rule."""
    if not MANIFEST_PATH.is_file() or not SPLIT_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "RAVDESS manifest or split summary missing. Run ml/build_ravdess_manifest.py first."
        )
    check_split_definition()
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = read_json(SPLIT_SUMMARY_PATH)

    actor_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        actor_splits[row["actor_id"]].add(row["split"])
        if ACTOR_TO_SPLIT[row["actor_id"]] != row["split"]:
            raise ValueError(f"Manifest split differs from the fixed allocation: {row['file']}")
    leaks = [actor for actor, splits in actor_splits.items() if len(splits) != 1]
    if leaks:
        raise ValueError(f"Actors occur in more than one split: {leaks}")

    included = [row for row in rows if row["include_in_model"].strip().lower() == "true"]
    if len(included) != EXPECTED_INCLUDED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_INCLUDED_TOTAL} included clips, found {len(included)}")
    for split in SPLIT_NAMES:
        split_rows = [row for row in included if row["split"] == split]
        counts = Counter(row["label"] for row in split_rows)
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(split_rows) != expected["clips"] or any(counts[label] != expected[label] for label in LABELS):
            raise ValueError(f"{split} counts {dict(counts)} differ from the handoff table.")
    for row in included:
        if row["label"] not in LABELS:
            raise ValueError(f"Unexpected label {row['label']!r}: {row['file']}")
    return included, summary


def rows_for(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]
