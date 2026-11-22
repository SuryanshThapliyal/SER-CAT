"""Fine-tune wav2vec 2.0 base on the fixed RAVDESS four-class split.

Starts from the pinned pretrained facebook/wav2vec2-base weights (NOT the
CREMA-D fine-tuned checkpoint). Uses the same 16 kHz mono preprocessing,
label order, mean-pooling classifier, frozen convolutional feature encoder
and class-weighted loss as ml/train_wav2vec2.py. The best epoch is chosen by
validation macro F1 with early stopping. Test actors are never loaded.
Requires a CUDA GPU; it will not fall back to CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from collections import Counter
from importlib.metadata import version
from typing import Any

import numpy as np

from ravdess_common import (
    DEFAULT_WAV2VEC2_RUN,
    LABELS,
    MANIFEST_PATH,
    WAV2VEC2_DIR,
    load_included_rows,
    resolve_audio_path,
    rows_for,
)
from train_wav2vec2 import (
    LABELS as CREMA_LABELS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    build_classifier,
    calculate_metrics,
    make_collate_fn,
    read_mono_audio,
)

if tuple(CREMA_LABELS) != tuple(LABELS):
    raise RuntimeError("wav2vec2 label order differs from the RAVDESS label order.")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def make_feature_extractor(Wav2Vec2FeatureExtractor: Any) -> Any:
    # The pinned revision has no preprocessor_config.json; these are the
    # upstream wav2vec2-base settings, as in train_wav2vec2_fixed.py.
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        padding_side="right",
        do_normalize=True,
        return_attention_mask=False,
    )


class RavdessEmotionDataset:
    def __init__(self, rows: list[dict[str, str]], feature_extractor: Any) -> None:
        self.rows = rows
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio = read_mono_audio(resolve_audio_path(row["file"]))
        values = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_attention_mask=False
        ).input_values[0]
        return {"input_values": values, "label": LABEL_TO_ID[row["label"]]}


def require_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU was not detected. This script will not train on CPU.")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_WAV2VEC2_RUN)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs, patience and batch size must be positive.")
    output_dir = WAV2VEC2_DIR / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Run directory already has files: {output_dir}. Use a new --run-name.")

    import torch
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = require_cuda(torch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        gpu_name = torch.cuda.get_device_name(device)
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    else:
        gpu_name, vram_gb = str(device), 0.0
    print(f"CUDA device: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    rows, split_summary = load_included_rows()
    train_rows, validation_rows = rows_for(rows, "train"), rows_for(rows, "validation")
    train_counts = Counter(row["label"] for row in train_rows)
    print(f"train: {len(train_rows)} clips {dict(sorted(train_counts.items()))}; "
          f"validation: {len(validation_rows)} clips (test not loaded)")

    feature_extractor = make_feature_extractor(Wav2Vec2FeatureExtractor)
    collate_fn = make_collate_fn(torch)
    pin = device.type == "cuda"
    train_loader = DataLoader(RavdessEmotionDataset(train_rows, feature_extractor),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=pin, collate_fn=collate_fn)
    validation_loader = DataLoader(RavdessEmotionDataset(validation_rows, feature_extractor),
                                   batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=pin, collate_fn=collate_fn)

    model = build_classifier(torch, Wav2Vec2Model, model_id=MODEL_ID, revision=MODEL_REVISION).to(device)
    class_weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * train_counts[label]) for label in LABELS],
        dtype=torch.float32, device=device,
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
    amp = (not args.no_fp16) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, Any]] = []
    best_macro_f1, best_epoch, best_metrics = -1.0, None, None
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
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(inputs, lengths)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=class_weights)
            loss_sum += float(loss.detach().item())
            scaler.scale(loss / group_size).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        metrics = calculate_metrics(torch, model, validation_loader, device, amp)
        history.append({
            "epoch": epoch,
            "training_loss": loss_sum / len(train_loader),
            "validation": metrics,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        })
        print(f"epoch={epoch}/{args.epochs} loss={history[-1]['training_loss']:.4f} "
              f"validation_macro_f1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1, best_epoch, best_metrics = metrics["macro_f1"], epoch, metrics
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "labels": list(LABELS),
                "sample_rate_hz": SAMPLE_RATE,
                "best_epoch": epoch,
                "dataset": "ravdess",
                "split_seed": split_summary["split_seed"],
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        results = {
            "experiment": "ravdess_wav2vec2_base_finetune",
            "dataset": "RAVDESS audio-only speech, four classes",
            "initialised_from": f"{MODEL_ID}@{MODEL_REVISION} (pretrained base, not the CREMA-D checkpoint)",
            "target_labels": list(LABELS),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "manifest": str(MANIFEST_PATH),
            "split_seed": split_summary["split_seed"],
            "split_method": split_summary["split_method"],
            "train_actor_ids": sorted({row["actor_id"] for row in train_rows}),
            "validation_actor_ids": sorted({row["actor_id"] for row in validation_rows}),
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(validation_rows),
            "training_label_counts": dict(sorted(train_counts.items())),
            "test_evaluated": False,
            "selected_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_metrics": best_metrics,
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
                "class_weights": dict(zip(LABELS, [round(float(w), 4) for w in class_weights.tolist()])),
                "convolutional_feature_extractor_frozen": True,
                "pooling": "masked mean over time",
                "mixed_precision_fp16": amp,
                "seed": args.seed,
            },
            "hardware": {"device": gpu_name, "vram_gb": round(vram_gb, 2)},
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

    print(f"Best validation macro-F1={best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved validation history: {results_path}")
    print("Test actors were not loaded.")


if __name__ == "__main__":
    main()
