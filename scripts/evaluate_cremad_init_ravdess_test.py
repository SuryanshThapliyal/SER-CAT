"""Repeat exploratory RAVDESS test for the CREMA-D-initialised transfer model.

Scores the selected checkpoint from ml/finetune_cremad_wav2vec2_on_ravdess.py
ONCE on the same 112 RAVDESS test clips (actors 16, 17, 19, 24) and builds a
comparison with the existing RAVDESS SVM and the base-initialised wav2vec2.

Caveat: these test actors and results were already examined in the previous
RAVDESS comparison, so this is a repeat exploratory comparison, not a fresh
blind benchmark. Nothing is trained or tuned here, and the script refuses to
run if its test report already exists.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any

from evaluate_ravdess_test import markdown_section, summarise
from finetune_cremad_wav2vec2_on_ravdess import DEFAULT_RUN, PREVIOUS_RAVDESS_RUN
from ravdess_common import LABELS, PROJECT_ROOT, RAVDESS_ROOT, SVM_DIR, WAV2VEC2_DIR, load_included_rows, read_json, rows_for
from train_ravdess_wav2vec2 import RavdessEmotionDataset, make_feature_extractor
from train_wav2vec2 import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, build_classifier, calculate_metrics, make_collate_fn

CAVEAT = (
    "Repeat exploratory comparison: RAVDESS test actors 16, 17, 19 and 24 (112 clips) were already "
    "examined in the previous RAVDESS comparison, so this is not a fresh blind benchmark. All settings "
    "and the epoch were fixed on RAVDESS validation before this single test evaluation. Scores are not "
    "comparable with CREMA-D test scores."
)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_dir = WAV2VEC2_DIR / args.run_name
    checkpoint_path = run_dir / "best_model.pt"
    validation_path = run_dir / "validation_results.json"
    test_path = run_dir / "test_results.json"
    comparison_md = run_dir / "comparison_with_previous_ravdess_models.md"
    comparison_json = run_dir / "comparison_with_previous_ravdess_models.json"
    for path in (checkpoint_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    for path in (test_path, comparison_md, comparison_json):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}: this test has already been run once.")
    transfer_validation = read_json(validation_path)
    if not transfer_validation.get("complete"):
        raise SystemExit("Transfer training did not finish; validation_results.json is not marked complete.")

    rows, split_summary = load_included_rows()
    test_rows = rows_for(rows, "test")
    test_actors = sorted({row["actor_id"] for row in test_rows})
    print(f"Scoring {len(test_rows)} RAVDESS test clips from actors {test_actors} (repeat exploratory)")

    import torch
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset") != "ravdess" or checkpoint.get("initialisation") != "cremad_finetuned":
        raise SystemExit("Checkpoint is not the CREMA-D-initialised RAVDESS transfer model.")
    for key, expected in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                          ("labels", list(LABELS)), ("sample_rate_hz", SAMPLE_RATE)):
        if checkpoint.get(key) != expected:
            raise SystemExit(f"Checkpoint {key} differs from the expected value.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    hardware = {"device": "cpu"}
    if use_cuda:
        hardware = {"device": torch.cuda.get_device_name(device),
                    "vram_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3), 2)}
    print(f"Inference device: {hardware['device']}")

    started = time.perf_counter()
    loader = DataLoader(RavdessEmotionDataset(test_rows, make_feature_extractor(Wav2Vec2FeatureExtractor)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda,
                        collate_fn=make_collate_fn(torch))
    model = build_classifier(torch, Wav2Vec2Model, model_id=MODEL_ID, revision=MODEL_REVISION)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metrics = calculate_metrics(torch, model, loader, device, amp=use_cuda)
    commit = git_commit()
    report = {
        "experiment": "ravdess_wav2vec2_transfer_from_cremad_test",
        "checkpoint": str(checkpoint_path),
        "init_checkpoint_sha256": checkpoint.get("init_checkpoint_sha256"),
        "selected_epoch": int(checkpoint["best_epoch"]),
        "test_actor_ids": test_actors,
        "test_sample_count": len(test_rows),
        "metrics": metrics,
        "test_evaluated": True,
        "training_or_model_selection_used_test": False,
        "repeat_exploratory_test": True,
        "caveat": CAVEAT,
        "hardware": hardware,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "evaluation_seconds": round(time.perf_counter() - started, 2),
        "git_commit": commit,
    }
    test_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Transfer model test macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")

    # ---- comparison with the existing RAVDESS models (same split) -----------
    svm_val = read_json(SVM_DIR / "validation_results.json")
    svm_test = read_json(SVM_DIR / "test_results.json")
    base_val = read_json(PREVIOUS_RAVDESS_RUN / "validation_results.json")
    base_test = read_json(PREVIOUS_RAVDESS_RUN / "test_results.json")
    names = ("MFCC + SVM", "wav2vec2 from base (previous)", "wav2vec2 from CREMA-D (transfer)")
    validation = dict(zip(names, (summarise(svm_val["selected_validation_metrics"]),
                                  summarise(base_val["best_validation_metrics"]),
                                  summarise(transfer_validation["best_validation_metrics"]))))
    test = dict(zip(names, (summarise(svm_test["metrics"]), summarise(base_test["metrics"]), summarise(metrics))))
    retention = transfer_validation.get("crema_d_retention", {})
    comparison = {
        "git_commit": commit,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_seed": split_summary["split_seed"],
        "transfer_selected_epoch": int(checkpoint["best_epoch"]),
        "base_selected_epoch": base_val.get("selected_epoch"),
        "svm_selected_settings": svm_val["selected_settings"],
        "init_checkpoint_sha256": checkpoint.get("init_checkpoint_sha256"),
        "validation": validation,
        "test_repeat_exploratory": test,
        "crema_d_validation_retention": {
            k: (retention[k]["macro_f1"] if isinstance(retention.get(k), dict) else retention.get(k))
            for k in ("evaluated", "clips", "before", "after", "macro_f1_change")
        },
        "caveat": CAVEAT,
    }
    comparison_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    cfg = transfer_validation["training_config"]
    md = [
        "# RAVDESS: CREMA-D-initialised wav2vec2 vs previous RAVDESS models", "",
        CAVEAT, "",
        f"- Git commit: `{commit}`",
        f"- Initialisation: `{transfer_validation['initialised_from']['checkpoint']}`",
        f"- Initialisation SHA-256: `{checkpoint.get('init_checkpoint_sha256')}`",
        f"- Transfer settings: encoder LR {cfg['encoder_learning_rate']}, head LR {cfg['head_learning_rate']}, "
        f"max {cfg['epochs_max']} epochs, patience {cfg['early_stopping_patience']}, batch {cfg['batch_size']} "
        f"x accumulation {cfg['gradient_accumulation_steps']}, class-weighted loss, frozen conv feature encoder",
        f"- Selected epochs (RAVDESS validation): transfer {int(checkpoint['best_epoch'])}, "
        f"base {base_val.get('selected_epoch')}",
        f"- RAVDESS validation macro F1 of the CREMA-D model before any RAVDESS training: "
        f"{transfer_validation['validation_macro_f1_at_initialisation']:.4f}",
    ]
    if retention.get("evaluated"):
        md.append(f"- CREMA-D validation retention ({retention['clips']} clips, no CREMA-D test used): macro F1 "
                  f"{retention['before']['macro_f1']:.4f} before -> {retention['after']['macro_f1']:.4f} after")
    else:
        md.append("- CREMA-D validation retention: not evaluated (CREMA-D audio not available)")
    md.append("")
    md += markdown_section("RAVDESS validation (actors 07, 08, 12, 13)", validation)
    md += markdown_section("RAVDESS test, repeat exploratory (actors 16, 17, 19, 24)", test)
    comparison_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {test_path}")
    print(f"Saved {comparison_md}")


if __name__ == "__main__":
    main()
