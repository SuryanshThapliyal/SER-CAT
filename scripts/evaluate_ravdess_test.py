"""Evaluate BOTH fixed RAVDESS models once on the 112 held-out test clips.

Run only after ml/train_ravdess_svm.py and ml/train_ravdess_wav2vec2.py have
finished and their settings/epoch are fixed by validation. Inference only:
nothing is trained or selected here. The script refuses to run if either test
report already exists, so the test set is scored once.

Writes:
  work/ravdess/svm/test_results.json
  work/ravdess/wav2vec2_experiments/<run>/test_results.json
  work/ravdess/ravdess_comparison.json and ravdess_comparison.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ravdess_common import (
    ATTRIBUTION,
    DEFAULT_WAV2VEC2_RUN,
    LABELS,
    LICENSE,
    PROJECT_ROOT,
    RAVDESS_ROOT,
    SVM_DIR,
    WAV2VEC2_DIR,
    load_included_rows,
    read_json,
    resolve_audio_path,
    rows_for,
)
from train_baseline import evaluate_predictions as svm_metrics
from train_baseline import extract_mfcc_features
from train_ravdess_wav2vec2 import RavdessEmotionDataset, make_feature_extractor
from train_wav2vec2 import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, build_classifier, calculate_metrics, make_collate_fn

COMPARISON_JSON = RAVDESS_ROOT / "ravdess_comparison.json"
COMPARISON_MD = RAVDESS_ROOT / "ravdess_comparison.md"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarise(metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = metrics["per_class"]
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics.get("accuracy", per_class.get("accuracy")),
        "per_class": {
            label: {key: per_class[label][key] for key in ("precision", "recall", "f1-score", "support")}
            for label in LABELS
        },
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def markdown_section(title: str, results: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| Model | Macro F1 | Accuracy | Clips |", "|---|---|---|---|"]
    for name, m in results.items():
        support = sum(m["per_class"][label]["support"] for label in LABELS)
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {int(support)} |")
    for name, m in results.items():
        lines += ["", f"**{name}: per class**", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for label in LABELS:
            c = m["per_class"][label]
            lines.append(f"| {label} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {int(c['support'])} |")
        lines += ["", f"**{name}: confusion matrix (rows = true, columns = predicted)**", "",
                  "| | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
        for label, row in zip(LABELS, m["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_WAV2VEC2_RUN)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    svm_model_path = SVM_DIR / "best_svm.joblib"
    svm_validation_path = SVM_DIR / "validation_results.json"
    svm_test_path = SVM_DIR / "test_results.json"
    run_dir = WAV2VEC2_DIR / args.run_name
    w2v_checkpoint_path = run_dir / "best_model.pt"
    w2v_validation_path = run_dir / "validation_results.json"
    w2v_test_path = run_dir / "test_results.json"

    for path in (svm_model_path, svm_validation_path, w2v_checkpoint_path, w2v_validation_path):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}. Train both models first.")
    for path in (svm_test_path, w2v_test_path, COMPARISON_JSON, COMPARISON_MD):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}: the RAVDESS test has already been run once.")

    rows, split_summary = load_included_rows()
    test_rows = rows_for(rows, "test")
    test_actors = sorted({row["actor_id"] for row in test_rows})
    print(f"Scoring {len(test_rows)} test clips from actors {test_actors}")
    svm_validation = read_json(svm_validation_path)
    w2v_validation = read_json(w2v_validation_path)
    commit = git_commit()

    # ---- MFCC + SVM -------------------------------------------------------
    started = time.perf_counter()
    X = np.vstack([extract_mfcc_features(resolve_audio_path(row["file"])) for row in test_rows])
    y = np.asarray([row["label"] for row in test_rows], dtype=str)
    svm = joblib.load(svm_model_path)
    predictions = svm.predict(X)
    svm_test = svm_metrics(y, predictions)
    svm_test["accuracy"] = float(np.mean(predictions == y))
    svm_test["sample_count"] = int(len(y))
    svm_report = {
        "experiment": "ravdess_mfcc_svm_test",
        "selected_settings": svm_validation["selected_settings"],
        "test_actor_ids": test_actors,
        "test_sample_count": len(test_rows),
        "metrics": svm_test,
        "test_evaluated": True,
        "training_or_model_selection_used_test": False,
        "evaluation_seconds": round(time.perf_counter() - started, 2),
        "git_commit": commit,
        "note": "One-time held-out test on 4 unseen RAVDESS actors; settings were fixed by validation.",
    }
    svm_test_path.write_text(json.dumps(svm_report, indent=2), encoding="utf-8")
    print(f"SVM test macro-F1={svm_test['macro_f1']:.4f} accuracy={svm_test['accuracy']:.4f}")

    # ---- wav2vec 2.0 ------------------------------------------------------
    import torch
    from torch.utils.data import DataLoader
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    checkpoint = torch.load(w2v_checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset") != "ravdess":
        raise SystemExit("Checkpoint is not the RAVDESS-trained model.")
    for key, expected in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                          ("labels", list(LABELS)), ("sample_rate_hz", SAMPLE_RATE)):
        if checkpoint.get(key) != expected:
            raise SystemExit(f"Checkpoint {key} differs from the expected value.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    hardware = {"device": "cpu"}
    if use_cuda:
        hardware = {
            "device": torch.cuda.get_device_name(device),
            "vram_gb": round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3), 2),
        }
    print(f"wav2vec2 inference device: {hardware['device']}")
    started = time.perf_counter()
    loader = DataLoader(RavdessEmotionDataset(test_rows, make_feature_extractor(Wav2Vec2FeatureExtractor)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=use_cuda, collate_fn=make_collate_fn(torch))
    model = build_classifier(torch, Wav2Vec2Model, model_id=MODEL_ID, revision=MODEL_REVISION)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    w2v_test = calculate_metrics(torch, model, loader, device, amp=use_cuda and not args.no_fp16)
    w2v_report = {
        "experiment": "ravdess_wav2vec2_base_test",
        "checkpoint": str(w2v_checkpoint_path),
        "selected_epoch": int(checkpoint["best_epoch"]),
        "test_actor_ids": test_actors,
        "test_sample_count": len(test_rows),
        "metrics": w2v_test,
        "test_evaluated": True,
        "training_or_model_selection_used_test": False,
        "hardware": hardware,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "evaluation_seconds": round(time.perf_counter() - started, 2),
        "git_commit": commit,
        "note": "One-time held-out test on 4 unseen RAVDESS actors; epoch was fixed by validation.",
    }
    w2v_test_path.write_text(json.dumps(w2v_report, indent=2), encoding="utf-8")
    print(f"wav2vec2 test macro-F1={w2v_test['macro_f1']:.4f} accuracy={w2v_test['accuracy']:.4f}")

    # ---- comparison ---------------------------------------------------------
    validation = {
        "MFCC + SVM": summarise(svm_validation["selected_validation_metrics"]),
        "wav2vec2-base": summarise(w2v_validation["best_validation_metrics"]),
    }
    test = {"MFCC + SVM": summarise(svm_test), "wav2vec2-base": summarise(w2v_test)}
    statement = (
        "These scores measure generalisation to four unseen RAVDESS actors (16, 17, 19, 24; 112 clips) "
        "and compare only the two models trained on this RAVDESS split. They are not comparable "
        "with the CREMA-D test scores."
    )
    comparison = {
        "dataset": "RAVDESS audio-only speech, four classes",
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "git_commit": commit,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_seed": split_summary["split_seed"],
        "split_actor_ids": {s: split_summary["split_summary"][s]["actor_ids"] for s in ("train", "validation", "test")},
        "svm_selected_settings": svm_validation["selected_settings"],
        "wav2vec2_selected_epoch": int(checkpoint["best_epoch"]),
        "validation": validation,
        "test": test,
        "statement": statement,
    }
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    s = svm_validation["selected_settings"]
    md = [
        "# RAVDESS four-class comparison: MFCC + SVM vs wav2vec2-base", "",
        statement, "",
        f"- Git commit: `{commit}`",
        f"- Split seed {split_summary['split_seed']}; train actors {comparison['split_actor_ids']['train']}, "
        f"validation {comparison['split_actor_ids']['validation']}, test {comparison['split_actor_ids']['test']}",
        f"- SVM selected by validation: scaler={s['scaler']}, kernel={s['kernel']}, C={s['C']}, gamma={s['gamma']}, class_weight=balanced",
        f"- wav2vec2 selected epoch (validation): {int(checkpoint['best_epoch'])}",
        f"- Primary metric: macro F1 (neutral has half as many clips as each other class)",
        f"- Data: {ATTRIBUTION} License: {LICENSE}.", "",
    ]
    md += markdown_section("Validation (actors 07, 08, 12, 13)", validation)
    md += markdown_section("Test, evaluated once (actors 16, 17, 19, 24)", test)
    COMPARISON_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {svm_test_path}")
    print(f"Saved {w2v_test_path}")
    print(f"Saved {COMPARISON_JSON} and {COMPARISON_MD}")


if __name__ == "__main__":
    main()
