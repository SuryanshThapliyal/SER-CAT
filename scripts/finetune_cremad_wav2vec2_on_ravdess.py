"""Transfer-fine-tune the CREMA-D wav2vec2 checkpoint on RAVDESS (train actors only).

Initialisation: the CREMA-D fine-tuned best_model.pt (NOT the generic
facebook/wav2vec2-base weights and NOT the earlier RAVDESS checkpoint).
The state dict is loaded with strict key matching after checking model
id/revision, sample rate and label order.

Training: the 448 RAVDESS training clips (actors 01-06, 09-11, 14, 15, 18,
20-23) with the same 16 kHz polyphase resampling and feature-extractor
settings as before, frozen convolutional feature encoder, class-weighted
cross-entropy, lower learning rates (encoder 1e-6, head 1e-5), max 15 epochs,
patience 3. The best epoch is chosen by RAVDESS validation macro F1 only.

Retention monitor (optional): if the CREMA-D audio is present, the CREMA-D
VALIDATION split (702 clips) is scored with the initial checkpoint and with
the selected RAVDESS checkpoint. CREMA-D test actors are never loaded.

RAVDESS test actors are never loaded here. Requires a CUDA GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ravdess_common import LABELS, MANIFEST_PATH, PROJECT_ROOT, WAV2VEC2_DIR, load_included_rows, rows_for
from train_ravdess_wav2vec2 import RavdessEmotionDataset, make_feature_extractor, require_cuda
import train_wav2vec2 as crema
from train_wav2vec2 import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, build_classifier, calculate_metrics, make_collate_fn

DEFAULT_RUN = "cremad_init_ravdess_seed_20261122"
DEFAULT_INIT = PROJECT_ROOT / "work" / "crema_d" / "wav2vec2_experiments" / "wav2vec2_base_seed_20261108" / "best_model.pt"
PREVIOUS_RAVDESS_RUN = WAV2VEC2_DIR / "wav2vec2_base_ravdess_seed_20261122"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_init_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                          ("labels", list(LABELS)), ("sample_rate_hz", SAMPLE_RATE)):
        if checkpoint.get(key) != expected:
            raise SystemExit(f"Initial checkpoint {key}={checkpoint.get(key)!r}, expected {expected!r}.")
    if checkpoint.get("dataset") == "ravdess":
        raise SystemExit("This is a RAVDESS checkpoint; the transfer run must start from the CREMA-D checkpoint.")
    return checkpoint


def crema_validation_loader(torch: Any, DataLoader: Any, feature_extractor: Any, batch_size: int):
    """CREMA-D validation rows only (read_split_rows never returns test rows)."""
    try:
        _, validation_rows, _ = crema.read_split_rows()
        for row in validation_rows:
            crema.resolve_audio_path(row["file"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"CREMA-D retention monitor skipped: {exc}")
        return None, 0
    dataset = crema.CremaEmotionDataset(validation_rows, feature_extractor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=make_collate_fn(torch))
    return loader, len(validation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--skip-crema-retention", action="store_true")
    args = parser.parse_args()

    output_dir = WAV2VEC2_DIR / args.run_name
    if output_dir.resolve() == PREVIOUS_RAVDESS_RUN.resolve():
        raise SystemExit("Refusing to write into the base-initialised RAVDESS run directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Run directory already has files: {output_dir}. Use a new --run-name.")
    if not args.init_checkpoint.is_file():
        raise SystemExit(f"CREMA-D checkpoint not found: {args.init_checkpoint}")

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

    print(f"Hashing initial checkpoint {args.init_checkpoint}")
    init_sha256 = sha256_of(args.init_checkpoint)
    init_checkpoint = load_init_checkpoint(torch, args.init_checkpoint)
    print(f"Initial checkpoint SHA-256 {init_sha256} (CREMA-D epoch {init_checkpoint.get('best_epoch')})")

    rows, split_summary = load_included_rows()
    train_rows, validation_rows = rows_for(rows, "train"), rows_for(rows, "validation")
    train_counts = Counter(row["label"] for row in train_rows)
    print(f"RAVDESS train: {len(train_rows)} clips {dict(sorted(train_counts.items()))}; "
          f"validation: {len(validation_rows)} clips (test not loaded)")

    feature_extractor = make_feature_extractor(Wav2Vec2FeatureExtractor)
    collate_fn = make_collate_fn(torch)
    pin = device.type == "cuda"
    train_loader = DataLoader(RavdessEmotionDataset(train_rows, feature_extractor), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=pin, collate_fn=collate_fn)
    validation_loader = DataLoader(RavdessEmotionDataset(validation_rows, feature_extractor),
                                   batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=pin, collate_fn=collate_fn)

    model = build_classifier(torch, Wav2Vec2Model, model_id=MODEL_ID, revision=MODEL_REVISION)
    model.load_state_dict(init_checkpoint["state_dict"], strict=True)
    model.to(device)
    amp = (not args.no_fp16) and device.type == "cuda"

    crema_loader, crema_count = (None, 0)
    if not args.skip_crema_retention:
        crema_loader, crema_count = crema_validation_loader(torch, DataLoader, feature_extractor, args.batch_size)
    retention: dict[str, Any] = {"evaluated": crema_loader is not None, "split": "CREMA-D validation only",
                                 "clips": crema_count, "crema_test_used": False}
    if crema_loader is not None:
        print(f"Scoring CREMA-D validation ({crema_count} clips) BEFORE RAVDESS fine-tuning")
        retention["before"] = calculate_metrics(torch, model, crema_loader, device, amp)
        print(f"CREMA-D validation macro-F1 before={retention['before']['macro_f1']:.4f}")

    initial_ravdess = calculate_metrics(torch, model, validation_loader, device, amp)
    print(f"RAVDESS validation macro-F1 at initialisation (epoch 0)={initial_ravdess['macro_f1']:.4f}")

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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, Any]] = [{"epoch": 0, "training_loss": None, "validation": initial_ravdess,
                                      "note": "CREMA-D checkpoint before any RAVDESS training"}]
    best_macro_f1, best_epoch, best_metrics = -1.0, None, None
    epochs_without_improvement = 0
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_model.pt"
    results_path = output_dir / "validation_results.json"
    started = time.perf_counter()

    def write_results(final: bool) -> None:
        results = {
            "experiment": "ravdess_wav2vec2_transfer_from_cremad",
            "dataset": "RAVDESS audio-only speech, four classes",
            "initialised_from": {
                "checkpoint": str(args.init_checkpoint),
                "sha256": init_sha256,
                "source": "CREMA-D fine-tuned wav2vec2-base (validation macro F1 0.7946, epoch "
                          f"{init_checkpoint.get('best_epoch')})",
                "strict_state_dict_load": True,
            },
            "target_labels": list(LABELS),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "manifest": str(MANIFEST_PATH),
            "split_seed": split_summary["split_seed"],
            "train_actor_ids": sorted({row["actor_id"] for row in train_rows}),
            "validation_actor_ids": sorted({row["actor_id"] for row in validation_rows}),
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(validation_rows),
            "training_label_counts": dict(sorted(train_counts.items())),
            "test_evaluated": False,
            "selected_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_metrics": best_metrics,
            "validation_macro_f1_at_initialisation": initial_ravdess["macro_f1"],
            "history": history,
            "learning_rate_search": "single configuration (encoder 1e-6, head 1e-5) from the handoff; no search",
            "training_config": {
                "epochs_max": args.epochs,
                "early_stopping_patience": args.patience,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "encoder_learning_rate": args.encoder_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": "linear warmup 10% then linear decay",
                "class_weighted_cross_entropy": True,
                "class_weights": dict(zip(LABELS, [round(float(w), 4) for w in class_weights.tolist()])),
                "convolutional_feature_extractor_frozen": True,
                "pooling": "masked mean over time",
                "resampling": "scipy resample_poly 48 kHz -> 16 kHz, mono mean",
                "mixed_precision_fp16": amp,
                "seed": args.seed,
            },
            "crema_d_retention": retention,
            "hardware": {"device": gpu_name, "vram_gb": round(vram_gb, 2)},
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "checkpoint": str(checkpoint_path),
            "complete": final,
            "note": "RAVDESS validation used for early stopping only. RAVDESS and CREMA-D test actors were not loaded.",
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

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
        history.append({"epoch": epoch, "training_loss": loss_sum / len(train_loader), "validation": metrics,
                        "epoch_seconds": round(time.perf_counter() - epoch_started, 2)})
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
                "initialisation": "cremad_finetuned",
                "init_checkpoint_sha256": init_sha256,
                "split_seed": split_summary["split_seed"],
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        write_results(final=False)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    if crema_loader is not None:
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"], strict=True)
        print(f"Scoring CREMA-D validation AFTER RAVDESS fine-tuning (epoch {best_epoch})")
        retention["after"] = calculate_metrics(torch, model, crema_loader, device, amp)
        retention["macro_f1_change"] = retention["after"]["macro_f1"] - retention["before"]["macro_f1"]
        print(f"CREMA-D validation macro-F1 before={retention['before']['macro_f1']:.4f} "
              f"after={retention['after']['macro_f1']:.4f}")
    write_results(final=True)
    print(f"Best RAVDESS validation macro-F1={best_macro_f1:.4f} at epoch {best_epoch} "
          f"(at initialisation: {initial_ravdess['macro_f1']:.4f})")
    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved validation history: {results_path}")
    print("RAVDESS and CREMA-D test actors were not loaded.")


if __name__ == "__main__":
    main()
