"""Evaluate a saved wav2vec2 checkpoint once on the original CREMA-D test split.

This script performs inference only. It does not train, select an epoch, or
read test predictions into any model-selection step. The original test split
was previously used for SVM evaluation, so its wav2vec2 result is exploratory.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from train_wav2vec2 import (
    CREMA_ROOT,
    EXPERIMENTS_DIR,
    LABELS,
    LABEL_TO_ID,
    MANIFEST_PATH,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    SPLIT_SUMMARY_PATH,
    CremaEmotionDataset,
    build_classifier,
    calculate_metrics,
    make_collate_fn,
    resolve_audio_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    EXPERIMENTS_DIR / "wav2vec2_base_seed_20261108" / "best_model.pt"
)
DEFAULT_OUTPUT = (
    EXPERIMENTS_DIR / "wav2vec2_base_seed_20261108" / "test_results.json"
)


def read_test_rows(
    manifest_path: Path,
    split_summary_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not split_summary_path.is_file():
        raise FileNotFoundError(f"Split summary not found: {split_summary_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    actor_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        actor_splits[row["actor_id"]].add(row["split"])
    overlaps = [actor for actor, splits in actor_splits.items() if len(splits) != 1]
    if overlaps:
        raise ValueError(f"Actors occur in multiple partitions: {overlaps}")

    test_rows = [
        row for row in rows
        if row["split"] == "test"
        and row["include_in_initial_model"].strip().lower() == "true"
    ]
    expected = summary["split_summary"]["test"]["included_four_class_files"]
    expected_actors = summary["split_summary"]["test"]["actors"]
    if len(test_rows) != expected:
        raise ValueError(f"Expected {expected} included test clips; found {len(test_rows)}")
    if len({row["actor_id"] for row in test_rows}) != expected_actors:
        raise ValueError("Test actor count differs from the split summary.")
    for row in test_rows:
        if row["actor_intended_label"] not in LABEL_TO_ID:
            raise ValueError(f"Unexpected test label in {row['file']}")
        if row["all_zero_signal"].strip().lower() == "true":
            raise ValueError(f"An all-zero recording is included: {row['file']}")
        resolve_audio_path(row["file"])
    return test_rows, summary


def make_feature_extractor() -> Wav2Vec2FeatureExtractor:
    # The pinned checkpoint revision predates preprocessor_config.json. These
    # values match the wav2vec2-base feature-extractor settings used in training.
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        padding_side="right",
        do_normalize=True,
        return_attention_mask=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split-summary", type=Path, default=SPLIT_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    for path in (args.checkpoint, args.manifest, args.split_summary):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    if args.output.exists():
        raise SystemExit(
            f"Refusing to overwrite an existing test report: {args.output}. "
            "Do not rerun evaluation on the same test split."
        )
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    test_rows, split_summary = read_test_rows(args.manifest, args.split_summary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_id") != MODEL_ID:
        raise ValueError("Checkpoint model ID differs from the expected model.")
    if checkpoint.get("model_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint model revision differs from the expected revision.")
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Checkpoint emotion label order differs from the expected order.")
    if checkpoint.get("sample_rate_hz") != SAMPLE_RATE:
        raise ValueError("Checkpoint sample rate differs from the expected sample rate.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    hardware: dict[str, Any] = {"device": str(device)}
    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        hardware = {
            "device": torch.cuda.get_device_name(device),
            "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        }
    print(f"Inference device: {hardware['device']}")
    print(f"Scoring {len(test_rows)} test clips from {len({r['actor_id'] for r in test_rows})} actors")

    dataset = CremaEmotionDataset(test_rows, make_feature_extractor())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=make_collate_fn(torch),
    )
    model = build_classifier(
        torch,
        Wav2Vec2Model,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics = calculate_metrics(
        torch,
        model,
        loader,
        device,
        amp=use_cuda and not args.no_fp16,
    )

    result = {
        "experiment": "wav2vec2_base_original_test_inference",
        "target_definition": "actor_intended_emotion",
        "target_labels": list(LABELS),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "split_seed": int(split_summary["split_seed"]),
        "split_method": split_summary["split_method"],
        "test_actor_count": len({row["actor_id"] for row in test_rows}),
        "test_sample_count": len(test_rows),
        "test_evaluated": True,
        "training_or_model_selection_used_test": False,
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "selected_epoch": int(checkpoint["best_epoch"]),
        "hardware": hardware,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "note": (
            "Exploratory result: these original test actors were previously used "
            "for SVM evaluation. Do not use this score to tune or select models."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test macro-F1={metrics['macro_f1']:.4f}")
    print(f"Saved test report: {args.output}")


if __name__ == "__main__":
    main()
