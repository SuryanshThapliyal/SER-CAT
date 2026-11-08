"""Train and evaluate the CREMA-D actor-intended MFCC + SVM baseline.

The actor-disjoint split and manifest live in the ignored local directory
``work/crema_d``. Model selection uses training and validation actors only.
The test set is evaluated only when explicitly requested with ``--evaluate-test``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREMA_ROOT = PROJECT_ROOT / "work" / "crema_d"
MANIFEST_PATH = CREMA_ROOT / "manifest_actor_intended.csv"
SPLIT_SUMMARY_PATH = CREMA_ROOT / "split_summary_seed_20261108.json"
RESULTS_DIR = CREMA_ROOT / "baseline_results"
FEATURE_CACHE_PATH = CREMA_ROOT / "mfcc13_mean_std_v1.npz"

TARGET_LABELS = ("anger", "happiness", "neutral", "sadness")
SPLIT_NAMES = ("train", "validation", "test")
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_MFCC = 13
N_MELS = 40
FEATURE_CACHE_VERSION = 1
SVM_C_VALUES = (0.1, 1.0, 10.0)
SVM_KERNELS = ("linear", "rbf")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    """Load eligible rows and enforce target, split, and speaker-group rules."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run the CREMA-D manifest "
            "builder first and keep the dataset under work/crema_d/."
        )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise ValueError("The CREMA-D manifest is empty.")

    actor_splits: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        actor = row["actor_id"]
        split = row["split"]
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split {split!r} for actor {actor}.")
        actor_splits[actor].add(split)
    leaking_actors = [actor for actor, splits in actor_splits.items() if len(splits) != 1]
    if leaking_actors:
        raise ValueError(f"Actors occur in more than one split: {leaking_actors}")

    rows = [
        row for row in all_rows
        if row["include_in_initial_model"].strip().lower() == "true"
    ]
    if not rows:
        raise ValueError("No usable rows are marked for the initial model.")

    for row in rows:
        if row["actor_intended_label"] not in TARGET_LABELS:
            raise ValueError(
                f"Unexpected target label {row['actor_intended_label']!r} "
                f"in {row['file']}"
            )
        if row["all_zero_signal"].strip().lower() == "true":
            raise ValueError(f"All-zero audio must be excluded: {row['file']}")

    present = {row["actor_intended_label"] for row in rows}
    if present != set(TARGET_LABELS):
        raise ValueError(f"Expected labels {TARGET_LABELS}; found {sorted(present)}")

    return rows


def resolve_audio_path(row: dict[str, str]) -> Path:
    """Resolve and validate the manifest's path relative to the dataset root."""
    root = CREMA_ROOT.resolve()
    path = (CREMA_ROOT / Path(row["file"])).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes the dataset directory: {row['file']}")
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    return path


def feature_cache_key(rows: list[dict[str, str]], manifest_path: Path) -> str:
    """Hash feature settings, manifest, and source-file metadata for cache safety."""
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    config = {
        "cache_version": FEATURE_CACHE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "frame_length": FRAME_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "summary": "per-coefficient mean and standard deviation",
    }
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    for row in rows:
        path = resolve_audio_path(row)
        stat = path.stat()
        digest.update(f"{row['file']}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def extract_mfcc_features(audio_path: Path) -> np.ndarray:
    """Return 26 MFCC summary features for a single audio recording."""
    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Empty or invalid decoded audio: {audio_path}")
    if float(np.max(np.abs(audio))) <= 1e-8:
        raise ValueError(f"No usable signal in included audio: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    feature_vector = np.concatenate(
        (np.mean(mfcc, axis=1), np.std(mfcc, axis=1))
    ).astype(np.float32)
    if feature_vector.shape != (2 * N_MFCC,):
        raise ValueError(
            f"Expected {2 * N_MFCC} features, got {feature_vector.shape} "
            f"for {audio_path}"
        )
    if not np.all(np.isfinite(feature_vector)):
        raise ValueError(f"Non-finite MFCC features: {audio_path}")
    return feature_vector


def build_feature_sets(
    rows: list[dict[str, str]],
    manifest_path: Path = MANIFEST_PATH,
    rebuild: bool = False,
) -> dict[str, dict[str, Any]]:
    """Extract or load cached features and partition them by fixed actor split."""
    cache_key = feature_cache_key(rows, manifest_path)
    cached_key: str | None = None
    if FEATURE_CACHE_PATH.is_file() and not rebuild:
        with np.load(FEATURE_CACHE_PATH, allow_pickle=False) as cache:
            cached_key = str(cache["cache_key"].item())
            if cached_key == cache_key:
                print(f"Loading cached MFCC features: {FEATURE_CACHE_PATH}")
                features = cache["features"].copy()
                labels = cache["labels"].astype(str).tolist()
                actors = cache["actors"].astype(str).tolist()
                splits = cache["splits"].astype(str).tolist()
                paths = cache["paths"].astype(str).tolist()
            else:
                features, labels, actors, splits, paths = extract_all_features(rows)
    else:
        features, labels, actors, splits, paths = extract_all_features(rows)

    if not FEATURE_CACHE_PATH.is_file() or rebuild or cached_key != cache_key:
        FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            FEATURE_CACHE_PATH,
            cache_key=np.asarray(cache_key),
            features=features,
            labels=np.asarray(labels, dtype="<U16"),
            actors=np.asarray(actors, dtype="<U8"),
            splits=np.asarray(splits, dtype="<U12"),
            paths=np.asarray(paths, dtype="<U64"),
        )
        print(f"Saved MFCC feature cache: {FEATURE_CACHE_PATH}")

    sets: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        mask = np.asarray(splits) == split_name
        split_labels = np.asarray(labels, dtype=str)[mask]
        if split_labels.size == 0:
            raise ValueError(f"No usable samples in {split_name} split.")
        sets[split_name] = {
            "X": features[mask],
            "y": split_labels,
            "actors": np.asarray(actors, dtype=str)[mask],
            "paths": np.asarray(paths, dtype=str)[mask],
        }
        missing = set(TARGET_LABELS) - set(split_labels.tolist())
        if missing:
            raise ValueError(f"Missing labels in {split_name}: {sorted(missing)}")
        print(
            f"{split_name}: {len(split_labels)} clips, "
            f"{len(set(sets[split_name]['actors'].tolist()))} actors, "
            f"labels={dict(sorted(Counter(split_labels.tolist()).items()))}"
        )

    return sets


def extract_all_features(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    actors: list[str] = []
    splits: list[str] = []
    paths: list[str] = []
    total = len(rows)
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        path = resolve_audio_path(row)
        features.append(extract_mfcc_features(path))
        labels.append(row["actor_intended_label"])
        actors.append(row["actor_id"])
        splits.append(row["split"])
        paths.append(row["file"])
        if index % 250 == 0 or index == total:
            elapsed = time.perf_counter() - started
            print(f"Extracted MFCC features for {index}/{total} clips ({elapsed:.1f}s)")
    return np.vstack(features), labels, actors, splits, paths


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=TARGET_LABELS, average="macro")),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
            target_names=TARGET_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": list(TARGET_LABELS),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
        ).tolist(),
    }


def train_and_select_with_validation(
    train_data: dict[str, Any],
    validation_data: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Select SVM kernel and C using validation macro F1 only."""
    candidates: list[dict[str, Any]] = []
    selected_model: Pipeline | None = None
    selected_settings: dict[str, Any] | None = None
    best_score = -1.0

    for kernel in SVM_KERNELS:
        for c_value in SVM_C_VALUES:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel=kernel,
                            C=c_value,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            started = time.perf_counter()
            model.fit(train_data["X"], train_data["y"])
            predictions = model.predict(validation_data["X"])
            metrics = evaluate_predictions(validation_data["y"], predictions)
            settings = {
                "kernel": kernel,
                "C": c_value,
                "class_weight": "balanced",
                "feature": f"MFCC{N_MFCC} mean+std",
            }
            candidate = {
                "settings": settings,
                "validation": metrics,
                "fit_and_validation_seconds": round(time.perf_counter() - started, 3),
            }
            candidates.append(candidate)
            print(
                f"validation {kernel=} {c_value=}: "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            # Keeping the first candidate on a tie makes the tie-break stable
            # in the order defined by SVM_KERNELS and SVM_C_VALUES.
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                selected_model = model
                selected_settings = settings

    if selected_model is None or selected_settings is None:
        raise RuntimeError("No SVM candidate was trained.")
    return selected_model, selected_settings, candidates


def package_versions() -> dict[str, str]:
    names = ("librosa", "numpy", "scikit-learn", "soundfile", "scipy", "numba")
    installed: dict[str, str] = {}
    for name in names:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = "not installed"
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore and rebuild the local MFCC feature cache.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the selected validation model once on held-out test actors.",
    )
    args = parser.parse_args()

    split_summary = read_json(SPLIT_SUMMARY_PATH)
    split_seed = int(split_summary["split_seed"])
    rows = load_manifest(args.manifest)
    print(
        f"Loaded {len(rows)} usable clips for {TARGET_LABELS}; "
        f"actor-disjoint split seed={split_seed}"
    )

    feature_sets = build_feature_sets(
        rows,
        manifest_path=args.manifest,
        rebuild=args.rebuild_features,
    )
    best_model, best_settings, candidates = train_and_select_with_validation(
        feature_sets["train"], feature_sets["validation"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / "best_svm.joblib"
    joblib.dump(best_model, model_path)
    validation_results = {
        "target_definition": "actor_intended_emotion",
        "target_labels": list(TARGET_LABELS),
        "listener_voice_votes_used_as_targets": False,
        "split_seed": split_seed,
        "split_method": split_summary["split_method"],
        "selected_settings": best_settings,
        "selected_validation_metrics": next(
            candidate["validation"]
            for candidate in candidates
            if candidate["settings"] == best_settings
        ),
        "candidate_results": candidates,
        "feature_config": {
            "sample_rate_hz": SAMPLE_RATE,
            "frame_length_samples": FRAME_LENGTH,
            "hop_length_samples": HOP_LENGTH,
            "n_mfcc": N_MFCC,
            "n_mels": N_MELS,
            "per_clip_summary": "mean and standard deviation for each MFCC",
            "feature_count": 2 * N_MFCC,
        },
        "training_counts": dict(sorted(Counter(feature_sets["train"]["y"].tolist()).items())),
        "validation_counts": dict(sorted(Counter(feature_sets["validation"]["y"].tolist()).items())),
        "package_versions": package_versions(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "model_artifact": str(model_path),
        "test_evaluated": False,
    }
    validation_path = RESULTS_DIR / "validation_results.json"
    validation_path.write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    selected_validation_f1 = validation_results["selected_validation_metrics"]["macro_f1"]
    print(f"Selected validation macro-F1={selected_validation_f1:.4f}")
    print(f"Saved selected model: {model_path}")
    print(f"Saved validation results: {validation_path}")

    if not args.evaluate_test:
        print("Test actors were not evaluated. Add --evaluate-test after settings are locked.")
        return

    test_results_path = RESULTS_DIR / "test_results.json"
    if test_results_path.exists():
        raise SystemExit(
            f"Refusing to overwrite {test_results_path}; the held-out test result "
            "has already been recorded. Start a new documented experiment if needed."
        )

    test_started = time.perf_counter()
    test_data = feature_sets["test"]
    test_predictions = best_model.predict(test_data["X"])
    test_metrics = evaluate_predictions(test_data["y"], test_predictions)
    test_results = {
        "target_definition": "actor_intended_emotion",
        "target_labels": list(TARGET_LABELS),
        "split_seed": split_seed,
        "selected_settings": best_settings,
        "test_sample_count": int(len(test_data["y"])),
        "test_actor_count": int(len(set(test_data["actors"].tolist()))),
        "metrics": test_metrics,
        "evaluation_seconds": round(time.perf_counter() - test_started, 3),
        "note": "Held-out test result; do not use it to select model settings.",
    }
    test_results_path.write_text(json.dumps(test_results, indent=2), encoding="utf-8")
    validation_results["test_evaluated"] = True
    validation_path.write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    print(f"TEST macro-F1={test_metrics['macro_f1']:.4f}")
    print(f"Saved final test results: {test_results_path}")


if __name__ == "__main__":
    main()
