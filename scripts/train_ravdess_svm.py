"""Train the MFCC + SVM model on the fixed RAVDESS split (validation only).

Uses the existing baseline feature definition and search space from
ml/train_baseline.py (16 kHz mono, 13 MFCCs, mean + std = 26 features;
standard/robust scaling, linear/RBF, C 1/3/10, the existing RBF gammas,
balanced class weights). Settings are chosen only by validation macro F1.
Test actors are never featurised or scored here; use ml/evaluate_ravdess_test.py
once after both models are fixed. CPU only.
"""

from __future__ import annotations

import json
import platform
import time
from collections import Counter

import joblib
import numpy as np

from ravdess_common import (
    LABELS,
    MANIFEST_PATH,
    SVM_DIR,
    load_included_rows,
    resolve_audio_path,
    rows_for,
)
from train_baseline import (
    FRAME_LENGTH,
    HOP_LENGTH,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    SCALERS,
    SVM_C_VALUES,
    SVM_GAMMA_VALUES,
    SVM_KERNELS,
    TARGET_LABELS,
    extract_mfcc_features,
    package_versions,
    train_and_select_with_validation,
)

if tuple(TARGET_LABELS) != tuple(LABELS):
    raise RuntimeError("Baseline label order differs from the RAVDESS label order.")


def featurise(rows: list[dict[str, str]], name: str) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    features = []
    for index, row in enumerate(rows, start=1):
        features.append(extract_mfcc_features(resolve_audio_path(row["file"])))
        if index % 100 == 0 or index == len(rows):
            print(f"{name}: MFCC features {index}/{len(rows)} ({time.perf_counter() - started:.1f}s)")
    return {
        "X": np.vstack(features),
        "y": np.asarray([row["label"] for row in rows], dtype=str),
        "actors": np.asarray([row["actor_id"] for row in rows], dtype=str),
    }


def main() -> None:
    SVM_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SVM_DIR / "best_svm.joblib"
    validation_path = SVM_DIR / "validation_results.json"
    if model_path.exists() or validation_path.exists():
        raise SystemExit(f"Refusing to overwrite the existing RAVDESS SVM in {SVM_DIR}.")

    rows, split_summary = load_included_rows()
    train_rows, validation_rows = rows_for(rows, "train"), rows_for(rows, "validation")
    print(f"train: {len(train_rows)} clips, validation: {len(validation_rows)} clips (test not loaded)")

    train_data = featurise(train_rows, "train")
    validation_data = featurise(validation_rows, "validation")
    model, settings, candidates = train_and_select_with_validation(train_data, validation_data)
    selected = next(c["validation"] for c in candidates if c["settings"] == settings)
    selected["accuracy"] = float(np.mean(model.predict(validation_data["X"]) == validation_data["y"]))
    selected["sample_count"] = int(len(validation_data["y"]))

    joblib.dump(model, model_path)
    results = {
        "experiment": "ravdess_mfcc_svm",
        "dataset": "RAVDESS audio-only speech, four classes",
        "target_labels": list(LABELS),
        "split_seed": split_summary["split_seed"],
        "split_method": split_summary["split_method"],
        "manifest": str(MANIFEST_PATH),
        "train_actor_ids": sorted(set(train_data["actors"].tolist())),
        "validation_actor_ids": sorted(set(validation_data["actors"].tolist())),
        "train_sample_count": int(len(train_data["y"])),
        "validation_sample_count": int(len(validation_data["y"])),
        "training_label_counts": dict(sorted(Counter(train_data["y"].tolist()).items())),
        "validation_label_counts": dict(sorted(Counter(validation_data["y"].tolist()).items())),
        "selected_settings": settings,
        "selected_validation_metrics": selected,
        "candidate_count": len(candidates),
        "candidate_results": candidates,
        "search_space": {
            "scalers": list(SCALERS),
            "kernels": list(SVM_KERNELS),
            "C": list(SVM_C_VALUES),
            "rbf_gamma": list(SVM_GAMMA_VALUES),
            "class_weight": "balanced",
            "selection_metric": "validation macro F1 (first candidate kept on ties)",
        },
        "feature_config": {
            "sample_rate_hz": SAMPLE_RATE,
            "frame_length_samples": FRAME_LENGTH,
            "hop_length_samples": HOP_LENGTH,
            "n_mfcc": N_MFCC,
            "n_mels": N_MELS,
            "per_clip_summary": "mean and standard deviation for each MFCC",
            "feature_count": 2 * N_MFCC,
        },
        "package_versions": package_versions(),
        "python_version": platform.python_version(),
        "model_artifact": str(model_path),
        "test_evaluated": False,
    }
    validation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected settings: {settings}")
    print(f"Selected validation macro-F1={selected['macro_f1']:.4f} accuracy={selected['accuracy']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved validation report: {validation_path}")
    print("Test actors were not loaded.")


if __name__ == "__main__":
    main()
