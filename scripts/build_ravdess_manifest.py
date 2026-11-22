"""Build the auditable RAVDESS four-class manifest and fixed split summary.

Input: the official audio-only speech archive (Audio_Speech_Actors_01-24.zip)
extracted under work/ravdess/data/ (Actor_01/ ... Actor_24/).

Output:
  work/ravdess/manifest_ravdess_speech_4class.csv   (all 1,440 speech files)
  work/ravdess/split_summary_ravdess_seed_20261122.json

Every one of the 1,440 files is listed with its parsed filename fields; the
672 neutral/happy/sad/angry clips are marked include_in_model=true. Included
clips are decoded and must be finite and non-silent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from ravdess_common import (
    ACTOR_TO_SPLIT,
    ARCHIVE_NAME,
    ARCHIVE_SOURCE,
    ATTRIBUTION,
    DATA_DIR,
    EMOTION_CODE_NAMES,
    EMOTION_CODE_TO_LABEL,
    EXPECTED_INCLUDED_TOTAL,
    EXPECTED_LABEL_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL_FILES,
    LABELS,
    LICENSE,
    MANIFEST_PATH,
    RAVDESS_ROOT,
    SPLIT_ACTORS,
    SPLIT_NAMES,
    SPLIT_SEED,
    SPLIT_SUMMARY_PATH,
    check_split_definition,
)

NAME_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.IGNORECASE)
COLUMNS = [
    "file", "actor_id", "actor_sex", "modality_code", "vocal_channel_code",
    "emotion_code", "emotion_name", "intensity_code", "statement_code",
    "repetition_code", "label", "split", "include_in_model", "exclusion_reason",
    "sample_rate_hz", "channels", "duration_seconds", "peak_abs_amplitude",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RAVDESS_ROOT / "raw" / ARCHIVE_NAME,
                        help="Downloaded archive, only used to record its SHA-256.")
    args = parser.parse_args()

    if MANIFEST_PATH.exists() or SPLIT_SUMMARY_PATH.exists():
        raise SystemExit(f"Refusing to overwrite an existing manifest/split summary in {RAVDESS_ROOT}.")
    check_split_definition()
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Extract {ARCHIVE_NAME} into {DATA_DIR} first.")

    wav_paths = sorted(DATA_DIR.rglob("*.wav"))
    if len(wav_paths) != EXPECTED_TOTAL_FILES:
        raise SystemExit(f"Expected {EXPECTED_TOTAL_FILES} WAV files under {DATA_DIR}, found {len(wav_paths)}.")

    rows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, path in enumerate(wav_paths, start=1):
        match = NAME_PATTERN.match(path.name)
        if not match:
            raise SystemExit(f"Unexpected filename: {path}")
        if path.name in seen_names:
            raise SystemExit(f"Duplicate filename: {path.name}")
        seen_names.add(path.name)
        modality, channel, emotion, intensity, statement, repetition, actor = match.groups()
        if modality != "03" or channel != "01":
            raise SystemExit(f"Not an audio-only speech file (modality 03, channel 01): {path.name}")
        if path.parent.name != f"Actor_{actor}":
            raise SystemExit(f"File {path.name} is in {path.parent.name}, expected Actor_{actor}.")
        if actor not in ACTOR_TO_SPLIT:
            raise SystemExit(f"Unknown actor {actor}: {path.name}")

        label = EMOTION_CODE_TO_LABEL.get(emotion, "")
        include = bool(label)
        row: dict[str, object] = {
            "file": path.relative_to(RAVDESS_ROOT).as_posix(),
            "actor_id": actor,
            "actor_sex": "male" if int(actor) % 2 else "female",
            "modality_code": modality,
            "vocal_channel_code": channel,
            "emotion_code": emotion,
            "emotion_name": EMOTION_CODE_NAMES.get(emotion, "unknown"),
            "intensity_code": intensity,
            "statement_code": statement,
            "repetition_code": repetition,
            "label": label,
            "split": ACTOR_TO_SPLIT[actor],
            "include_in_model": "true" if include else "false",
            "exclusion_reason": "" if include else "emotion_outside_four_class_target",
            "sample_rate_hz": "", "channels": "", "duration_seconds": "", "peak_abs_amplitude": "",
        }
        if include:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            if audio.size == 0 or not np.all(np.isfinite(audio)):
                raise SystemExit(f"Empty or non-finite audio: {path}")
            peak = float(np.max(np.abs(audio)))
            if peak <= 1e-8:
                raise SystemExit(f"Silent (all-zero) included audio: {path}")
            row.update({
                "sample_rate_hz": sample_rate,
                "channels": audio.shape[1],
                "duration_seconds": round(audio.shape[0] / sample_rate, 6),
                "peak_abs_amplitude": round(peak, 6),
            })
        rows.append(row)
        if index % 200 == 0 or index == len(wav_paths):
            print(f"Checked {index}/{len(wav_paths)} files")

    included = [row for row in rows if row["include_in_model"] == "true"]
    label_counts = Counter(row["label"] for row in included)
    if len(included) != EXPECTED_INCLUDED_TOTAL or dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise SystemExit(f"Included counts {dict(label_counts)} differ from the handoff table.")

    split_summary: dict[str, object] = {}
    for split in SPLIT_NAMES:
        split_rows = [row for row in included if row["split"] == split]
        counts = Counter(row["label"] for row in split_rows)
        actors = sorted({row["actor_id"] for row in split_rows})
        expected = EXPECTED_SPLIT_COUNTS[split]
        if (actors != sorted(SPLIT_ACTORS[split]) or len(split_rows) != expected["clips"]
                or any(counts[label] != expected[label] for label in LABELS)):
            raise SystemExit(f"{split}: {len(split_rows)} clips, {dict(counts)} differ from the handoff table.")
        split_summary[split] = {
            "actor_ids": actors,
            "actors": len(actors),
            "male_actors": sum(int(a) % 2 == 1 for a in actors),
            "female_actors": sum(int(a) % 2 == 0 for a in actors),
            "all_speech_files": sum(1 for row in rows if row["split"] == split),
            "included_four_class_files": len(split_rows),
            "included_label_counts": {label: counts[label] for label in LABELS},
        }

    RAVDESS_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    sample_rates = Counter(row["sample_rate_hz"] for row in included)
    channels = Counter(row["channels"] for row in included)
    durations = [float(row["duration_seconds"]) for row in included]
    summary = {
        "dataset": "RAVDESS audio-only speech",
        "archive": ARCHIVE_NAME,
        "archive_sha256": sha256_of(args.archive) if args.archive.is_file() else None,
        "source": ARCHIVE_SOURCE,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "target_labels": list(LABELS),
        "emotion_code_to_label": EMOTION_CODE_TO_LABEL,
        "excluded_emotion_codes": {"02": "calm", "06": "fearful", "07": "disgust", "08": "surprised"},
        "split_seed": SPLIT_SEED,
        "split_method": (
            "Fixed actor allocation from docs/ravdess-two-model-gpu-handoff.md, generated with seed "
            "20261122 and stratified by odd (male) / even (female) actor IDs; every actor is in "
            "exactly one split. Allocation was fixed before any model was trained."
        ),
        "speech_files_total": len(rows),
        "included_total": len(included),
        "included_label_counts": {label: label_counts[label] for label in LABELS},
        "excluded_total": len(rows) - len(included),
        "included_audio": {
            "sample_rates_hz": {str(k): v for k, v in sample_rates.items()},
            "channels": {str(k): v for k, v in channels.items()},
            "duration_seconds_min": round(min(durations), 3),
            "duration_seconds_mean": round(sum(durations) / len(durations), 3),
            "duration_seconds_max": round(max(durations), 3),
            "all_finite_and_non_silent": True,
        },
        "split_summary": split_summary,
        "manual_64_clip_sample_pack_used": False,
        "crema_d_files_used": False,
        "files": {"manifest": MANIFEST_PATH.name},
    }
    SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Included {len(included)} of {len(rows)} speech files: {dict(label_counts)}")
    for split in SPLIT_NAMES:
        info = split_summary[split]
        print(f"{split}: actors {info['actor_ids']} -> {info['included_four_class_files']} clips "
              f"{info['included_label_counts']}")
    print(f"Sample rates {dict(sample_rates)}, channels {dict(channels)}")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Saved split summary: {SPLIT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
