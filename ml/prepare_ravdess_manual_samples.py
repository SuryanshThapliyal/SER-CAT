"""Build a balanced multi-speaker RAVDESS sample pack for manual SER checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ZIP = (
    PROJECT_ROOT
    / "work"
    / "crema_d"
    / "external_test_samples"
    / "RAVDESS"
    / "source"
    / "Audio_Speech_Actors_01-24.zip"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "work"
    / "crema_d"
    / "external_test_samples"
    / "RAVDESS"
    / "manual_test_set"
)
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "work"
    / "crema_d"
    / "external_test_samples"
    / "RAVDESS_8speaker_4emotion_manual_samples.zip"
)
ACTORS = (1, 4, 7, 10, 13, 16, 19, 22)
EMOTIONS = {
    "01": "neutral",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
}
STATEMENTS = ("01", "02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_zip.is_file():
        raise SystemExit(f"RAVDESS speech archive not found: {args.source_zip}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output folder is not empty: {args.output_dir}")
    if args.archive.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {args.archive}")

    selected: list[tuple[str, str, str, str]] = []
    with zipfile.ZipFile(args.source_zip) as source:
        for source_name in source.namelist():
            if not source_name.lower().endswith(".wav"):
                continue
            stem = Path(source_name).stem
            fields = stem.split("-")
            if len(fields) != 7:
                continue
            modality, channel, emotion_id, intensity, statement, repetition, actor = fields
            if (
                modality == "03"
                and channel == "01"
                and emotion_id in EMOTIONS
                and intensity == "01"
                and statement in STATEMENTS
                and repetition == "01"
                and int(actor) in ACTORS
            ):
                selected.append((source_name, emotion_id, statement, actor))

        expected_count = len(ACTORS) * len(EMOTIONS) * len(STATEMENTS)
        if len(selected) != expected_count:
            raise SystemExit(
                f"Expected {expected_count} balanced WAVs, found {len(selected)}. "
                "Check that the official RAVDESS speech archive is complete."
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, str]] = []
        for source_name, emotion_id, statement, actor in sorted(selected):
            filename = Path(source_name).name
            target = args.output_dir / filename
            target.write_bytes(source.read(source_name))
            manifest_rows.append({
                "filename": filename,
                "emotion": EMOTIONS[emotion_id],
                "speaker": f"Actor_{int(actor):02d}",
                "actor_id": actor,
                "statement_id": statement,
                "source_filename": filename,
                "source_dataset": "RAVDESS",
            })

    manifest_path = args.output_dir / "samples.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = """# RAVDESS manual test sample

This pack contains 64 speech WAV clips selected from the official RAVDESS
speech archive. It includes eight actors (four odd-numbered male actors and
four even-numbered female actors), all four emotions shared with this project,
and both RAVDESS sentences. Each actor/emotion/sentence combination appears
once at normal intensity. The set is balanced: 16 clips per emotion.

Emotion codes map as follows: 01 neutral, 03 happy -> happiness, 04 sad ->
sadness, and 05 angry -> anger. The filename and `samples.csv` preserve the
actor-intended label. Upload the WAVs to the website and compare the predicted
label with that manifest. These are acted RAVDESS labels, not measurements of
the actors' internal feelings. This small sample is exploratory, not a
benchmark; do not use it to tune the model or report accuracy as a final score.

Source: Livingstone and Russo, RAVDESS, https://zenodo.org/records/1188976
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International. Retain this attribution and license if you share the files.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(args.output_dir.iterdir()):
            package.write(path, arcname=f"RAVDESS_manual_test_set/{path.name}")

    print(f"Created {len(manifest_rows)} WAV clips at {args.output_dir}")
    print(f"Created sample archive: {args.archive}")
    print("Counts: 16 each for anger, happiness, neutral, and sadness; 8 speakers; 2 sentences.")


if __name__ == "__main__":
    main()
