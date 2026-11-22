"""Fine-tune a pretrained wav2vec 2.0 encoder for CREMA-D emotion labels.

This experiment uses only the existing training and validation actors. It
requires a CUDA GPU by default and deliberately refuses to fall back to CPU.
The held-out test actors are never loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREMA_ROOT = PROJECT_ROOT / "work" / "crema_d"
MANIFEST_PATH = CREMA_ROOT / "manifest_actor_intended.csv"
SPLIT_SUMMARY_PATH = CREMA_ROOT / "split_summary_seed_20261108.json"
EXPERIMENTS_DIR = CREMA_ROOT / "wav2vec2_experiments"

LABELS = ("anger", "happiness", "neutral", "sadness")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SAMPLE_RATE = 16_000
MODEL_ID = "facebook/wav2vec2-base"
# Pin the pretrained weights/config revision for reproducibility.
MODEL_REVISION = "bfb1337a27fc4a418743753d6c965436844ab0a2"


def read_split_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Copy the reviewed CREMA-D "
            "manifest and audio files under work/crema_d/."
        )
    if not SPLIT_SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Split summary not found: {SPLIT_SUMMARY_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_summary = json.loads(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    actor_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        actor_splits[row["actor_id"]].add(row["split"])
    overlaps = [actor for actor, splits in actor_splits.items() if len(splits) != 1]
    if overlaps:
        raise ValueError(f"Actors leak across data partitions: {overlaps}")

    eligible = [
        row for row in rows
        if row["include_in_initial_model"].strip().lower() == "true"
    ]
    for row in eligible:
        if row["actor_intended_label"] not in LABEL_TO_ID:
            raise ValueError(f"Unexpected label {row['actor_intended_label']!r}: {row['file']}")
        if row["all_zero_signal"].strip().lower() == "true":
            raise ValueError(f"All-zero audio must be excluded: {row['file']}")

    train_rows = [row for row in eligible if row["split"] == "train"]
    validation_rows = [row for row in eligible if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("Both training and validation splits must contain eligible clips.")
    # No test rows are returned, decoded, or evaluated in this experiment.
    return train_rows, validation_rows, split_summary


def resolve_audio_path(relative_path: str) -> Path:
    root = CREMA_ROOT.resolve()
    path = (CREMA_ROOT / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes the dataset directory: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def read_mono_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Empty or invalid decoded audio: {path}")
    if float(np.max(np.abs(audio))) <= 1e-8:
        raise ValueError(f"No usable signal in included audio: {path}")
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            up=SAMPLE_RATE // divisor,
            down=int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    return np.asarray(audio, dtype=np.float32)


class CremaEmotionDataset:
    def __init__(self, rows: list[dict[str, str]], feature_extractor: Any) -> None:
        self.rows = rows
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio = read_mono_audio(resolve_audio_path(row["file"]))
        values = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=False,
        ).input_values[0]
        return {
            "input_values": values,
            "label": LABEL_TO_ID[row["actor_intended_label"]],
        }


def make_collate_fn(torch: Any):
    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [torch.as_tensor(item["input_values"], dtype=torch.float32) for item in samples]
        lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0.0,
        )
        labels = torch.tensor([item["label"] for item in samples], dtype=torch.long)
        return {"input_values": padded, "input_lengths": lengths, "labels": labels}

    return collate


def build_classifier(torch: Any, Wav2Vec2Model: Any, *, model_id: str, revision: str):
    class Wav2VecEmotionClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Wav2Vec2Model.from_pretrained(model_id, revision=revision)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(hidden_size, len(LABELS))
            torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.classifier.bias)
            # Keep the low-level convolutional feature extractor fixed on this
            # relatively small labeled dataset; fine-tune the transformer and head.
            self.encoder.feature_extractor._freeze_parameters()

        def forward(self, input_values: Any, input_lengths: Any) -> Any:
            # wav2vec2-base uses GroupNorm and was pretrained without attention
            # masks. We pad with zeros and use true lengths only for mean pooling.
            hidden = self.encoder(input_values=input_values).last_hidden_state
            output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
            output_lengths = output_lengths.clamp(min=1, max=hidden.shape[1])
            time_steps = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            mask = time_steps < output_lengths[:, None]
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / output_lengths.unsqueeze(-1).to(hidden.dtype)
            return self.classifier(self.dropout(pooled))

    return Wav2VecEmotionClassifier()


def calculate_metrics(torch: Any, model: Any, loader: Any, device: Any, amp: bool) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_values"].to(device, non_blocking=True)
            lengths = batch["input_lengths"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(inputs, lengths)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(len(LABELS)), average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=range(len(LABELS)),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
        "sample_count": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="wav2vec2_base_seed_20261108")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA mixed precision.")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs, patience, and batch size must be positive.")
    output_dir = EXPERIMENTS_DIR / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Run directory already contains files: {output_dir}. Choose a new --run-name "
            "so prior experiment outputs remain available."
        )

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except ImportError as exc:
        raise SystemExit(
            "Install the wav2vec dependencies and a CUDA-enabled PyTorch build first. "
            "See ml/README.md."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU was not detected. This training script intentionally will not "
            "fall back to CPU. Run it on the GPU-equipped machine."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"CUDA device: {torch.cuda.get_device_name(device)} ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 8:
        print("Warning: under 8 GB VRAM; reduce --batch-size to 2 or 1 if memory runs out.")

    train_rows, validation_rows, split_summary = read_split_rows()
    train_counts = Counter(row["actor_intended_label"] for row in train_rows)
    missing = set(LABELS) - set(train_counts)
    if missing:
        raise ValueError(f"Training split is missing target labels: {sorted(missing)}")
    expected_val = split_summary["split_summary"]["validation"]["included_four_class_files"]
    if len(validation_rows) != expected_val:
        raise ValueError("Validation row count differs from the reviewed split summary.")

    # FIX: the pinned revision has no preprocessor_config.json (it was added in a
    # later commit), so from_pretrained() fails. Build the extractor with the
    # same settings as the upstream facebook/wav2vec2-base preprocessor_config.json.
    # Model weights/config remain pinned to MODEL_REVISION.
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        padding_side="right",
        do_normalize=True,
        return_attention_mask=False,
    )
    train_dataset = CremaEmotionDataset(train_rows, feature_extractor)
    validation_dataset = CremaEmotionDataset(validation_rows, feature_extractor)
    collate_fn = make_collate_fn(torch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_classifier(
        torch,
        Wav2Vec2Model,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)
    class_weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * train_counts[label]) for label in LABELS],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
            {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(total_updates * 0.1))

    def lr_scale(step: int) -> float:
        if step < warmup_updates:
            return float(step + 1) / warmup_updates
        return max(0.0, float(total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp = not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch: int | None = None
    best_validation_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_model.pt"
    results_path = output_dir / "validation_results.json"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device, non_blocking=True)
            lengths = batch["input_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            group_start = (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(inputs, lengths)
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss_sum += float(loss.detach().item())
            scaler.scale(loss / group_size).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        validation_metrics = calculate_metrics(torch, model, validation_loader, device, amp)
        epoch_result = {
            "epoch": epoch,
            "training_loss": loss_sum / len(train_loader),
            "validation": validation_metrics,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch}/{args.epochs} loss={epoch_result['training_loss']:.4f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "labels": list(LABELS),
                    "sample_rate_hz": SAMPLE_RATE,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        results = {
            "experiment": "wav2vec2_base_finetune",
            "target_definition": "actor_intended_emotion",
            "target_labels": list(LABELS),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "split_seed": int(split_summary["split_seed"]),
            "split_method": split_summary["split_method"],
            "train_actor_count": len({row["actor_id"] for row in train_rows}),
            "train_sample_count": len(train_rows),
            "validation_actor_count": len({row["actor_id"] for row in validation_rows}),
            "validation_sample_count": len(validation_rows),
            "test_evaluated": False,
            "training_label_counts": dict(sorted(train_counts.items())),
            "selected_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_metrics": best_validation_metrics,
            "history": history,
            "training_config": {
                "epochs_max": args.epochs,
                "early_stopping_patience": args.patience,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "encoder_learning_rate": args.encoder_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
                "class_weighted_cross_entropy": True,
                "convolutional_feature_extractor_frozen": True,
                "mixed_precision_fp16": amp,
                "seed": args.seed,
            },
            "hardware": {
                "device": torch.cuda.get_device_name(device),
                "vram_gb": round(vram_gb, 2),
            },
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "checkpoint": str(checkpoint_path),
            "note": "Validation-only selection. Test actors were not loaded or evaluated.",
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    print(f"Best validation macro-F1={best_macro_f1:.4f}")
    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved validation history: {results_path}")
    print("No test actors were loaded. Do not use the held-out test set to tune this model.")


if __name__ == "__main__":
    main()
