"""Generate and seal E1 probabilities without access to target labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from vultriage.data import load_config, sha256, stable_bucket
from vultriage.domain import clipped_weight_summary, crossfit_density_ratio


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError("Metadata positions are not contiguous and ordered")
    return rows


def role_positions(
    metadata: list[dict[str, str]], positions: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train: list[int] = []
    validation: list[int] = []
    calibration: list[int] = []
    train_end = int(config["source_partition"]["train_end"])
    validation_end = int(config["source_partition"]["model_validation_end"])
    for position in positions:
        bucket = stable_bucket(
            metadata[int(position)]["commit_id"], config["split_salt"], 100
        )
        if bucket < train_end:
            train.append(int(position))
        elif bucket < validation_end:
            validation.append(int(position))
        else:
            calibration.append(int(position))
    return (
        np.asarray(train, dtype=np.int32),
        np.asarray(validation, dtype=np.int32),
        np.asarray(calibration, dtype=np.int32),
    )


def labels_for_positions(
    archive_positions: np.ndarray, archive_labels: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    order = np.argsort(archive_positions)
    sorted_positions = archive_positions[order]
    indices = np.searchsorted(sorted_positions, positions)
    if (indices >= len(sorted_positions)).any() or not np.array_equal(
        sorted_positions[indices], positions
    ):
        raise ValueError("Requested a label not present in the source-only archive")
    return archive_labels[order][indices].astype(int, copy=False)


def classifier(
    alpha: float, seed: int, class_weight: dict[int, float], config: dict[str, Any]
) -> SGDClassifier:
    grid = config["sgd_grid"]
    return SGDClassifier(
        loss=grid["loss"],
        penalty=grid["penalty"],
        alpha=float(alpha),
        class_weight=class_weight,
        random_state=int(seed),
        max_iter=int(grid["epochs"]),
        tol=None,
        average=True,
    )


def balanced_class_weight(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Detector training requires both vulnerability classes")
    return {label: float(len(labels) / (2 * counts[label])) for label in (0, 1)}


def fit_detector(
    features: sparse.csr_matrix,
    train_positions: np.ndarray,
    train_labels: np.ndarray,
    validation_positions: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> tuple[SGDClassifier, float, float]:
    weights = balanced_class_weight(train_labels)
    best_alpha: float | None = None
    best_score = -np.inf
    for alpha in config["sgd_grid"]["alpha"]:
        model = classifier(float(alpha), seed, weights, config)
        model.fit(features[train_positions], train_labels)
        probability = model.predict_proba(features[validation_positions])[:, 1]
        score = float(average_precision_score(validation_labels, probability))
        if score > best_score or (
            score == best_score
            and (best_alpha is None or float(alpha) > best_alpha)
        ):
            best_alpha = float(alpha)
            best_score = score
    assert best_alpha is not None
    final_positions = np.concatenate([train_positions, validation_positions])
    final_labels = np.concatenate([train_labels, validation_labels])
    model = classifier(
        best_alpha, seed, balanced_class_weight(final_labels), config
    )
    model.fit(features[final_positions], final_labels)
    return model, best_alpha, best_score


def official_roles(
    metadata: list[dict[str, str]], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train: list[int] = []
    validation: list[int] = []
    calibration: list[int] = []
    test: list[int] = []
    cutoff = int(
        config["official_validation_calibration_split"]["model_validation_end"]
    )
    salt = config["official_validation_calibration_split"]["salt"]
    for position, row in enumerate(metadata):
        if row["origin_split"] == "train":
            train.append(position)
        elif row["origin_split"] == "test":
            test.append(position)
        elif stable_bucket(row["commit_id"], salt, 100) < cutoff:
            validation.append(position)
        else:
            calibration.append(position)
    return tuple(
        np.asarray(values, dtype=np.int32)
        for values in (train, validation, calibration, test)
    )


def run_official(
    output: Path,
    features: sparse.csr_matrix,
    metadata: list[dict[str, str]],
    source_dir: Path,
    config: dict[str, Any],
) -> None:
    archive = np.load(source_dir / "official.npz")
    archive_positions = archive["positions"]
    archive_labels = archive["labels"]
    train, validation, calibration, test = official_roles(metadata, config)
    for seed in config["seeds"]:
        started = time.time()
        train_labels = labels_for_positions(archive_positions, archive_labels, train)
        validation_labels = labels_for_positions(
            archive_positions, archive_labels, validation
        )
        model, alpha, score = fit_detector(
            features,
            train,
            train_labels,
            validation,
            validation_labels,
            int(seed),
            config,
        )
        calibration_probability = model.predict_proba(features[calibration])[:, 1]
        test_probability = model.predict_proba(features[test])[:, 1]
        path = output / "predictions" / "official" / f"seed-{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            calibration_positions=calibration,
            calibration_p_vulnerable=calibration_probability.astype(np.float32),
            target_positions=test,
            target_p_vulnerable=test_probability.astype(np.float32),
        )
        atomic_json(
            path.with_suffix(".json"),
            {
                "track": "official_chronological",
                "seed": int(seed),
                "selected_alpha": alpha,
                "source_validation_pr_auc": score,
                "elapsed_seconds": time.time() - started,
                "target_labels_accessed": False,
            },
        )
        log(f"official seed={seed} complete")


def run_project(
    group: str,
    output: Path,
    features: sparse.csr_matrix,
    metadata: list[dict[str, str]],
    source_dir: Path,
    config: dict[str, Any],
) -> None:
    archive = np.load(source_dir / f"{group}.npz")
    archive_positions = archive["positions"].astype(np.int32)
    archive_labels = archive["labels"].astype(int)
    train, validation, calibration = role_positions(
        metadata, archive_positions, config
    )
    target = np.asarray(
        [
            position
            for position, row in enumerate(metadata)
            if row["project_group"] == group
        ],
        dtype=np.int32,
    )
    row_ids = [metadata[int(position)]["row_id"] for position in np.concatenate([calibration, target])]
    domain_labels = np.concatenate(
        [np.zeros(len(calibration), dtype=int), np.ones(len(target), dtype=int)]
    )
    domain_features = features[np.concatenate([calibration, target])]

    density_seed = int(config["seeds"][0])
    density = crossfit_density_ratio(
        domain_features,
        row_ids,
        domain_labels,
        folds=int(config["density_ratio"]["crossfit_folds"]),
        alpha=float(config["density_ratio"]["domain_alpha"]),
        epochs=int(config["density_ratio"]["domain_epochs"]),
        seed=density_seed,
        salt=f"vultriage-domain-v1|{group}",
    )
    n_calibration = len(calibration)
    sensitivity = {
        str(clip): {
            "calibration": clipped_weight_summary(
                density.raw_ratios[:n_calibration], float(clip)
            ),
            "target": clipped_weight_summary(
                density.raw_ratios[n_calibration:], float(clip)
            ),
        }
        for clip in config["density_ratio"]["candidate_upper_clips"]
    }
    log(f"project={group} cross-fitted domain ratios complete")

    for seed in config["seeds"]:
        started = time.time()
        train_labels = labels_for_positions(archive_positions, archive_labels, train)
        validation_labels = labels_for_positions(
            archive_positions, archive_labels, validation
        )
        model, alpha, score = fit_detector(
            features,
            train,
            train_labels,
            validation,
            validation_labels,
            int(seed),
            config,
        )
        calibration_probability = model.predict_proba(features[calibration])[:, 1]
        target_probability = model.predict_proba(features[target])[:, 1]
        path = output / "predictions" / group / f"seed-{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            calibration_positions=calibration,
            calibration_p_vulnerable=calibration_probability.astype(np.float32),
            calibration_raw_ratio=density.raw_ratios[:n_calibration],
            calibration_domain_probability=density.target_probabilities[:n_calibration].astype(np.float32),
            target_positions=target,
            target_p_vulnerable=target_probability.astype(np.float32),
            target_raw_ratio=density.raw_ratios[n_calibration:],
            target_domain_probability=density.target_probabilities[n_calibration:].astype(np.float32),
        )
        atomic_json(
            path.with_suffix(".json"),
            {
                "track": "project_disjoint",
                "target_group": group,
                "seed": int(seed),
                "selected_alpha": alpha,
                "source_validation_pr_auc": score,
                "domain_diagnostics": density.diagnostics,
                "density_estimator_seed": density_seed,
                "density_ratios_shared_across_head_seeds": True,
                "weight_sensitivity": sensitivity,
                "elapsed_seconds": time.time() - started,
                "target_vulnerability_labels_accessed": False,
            },
        )
        log(f"project={group} seed={seed} complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    cache_metadata = json.loads(
        (args.feature_cache / "metadata.json").read_text(encoding="utf-8")
    )
    input_hashes = json.loads(
        (args.inputs / "hashes.json").read_text(encoding="utf-8")
    )
    expected_config_hash = cache_metadata["config_sha256"]
    if sha256(args.config) != expected_config_hash:
        raise RuntimeError("Experiment config differs from feature-cache config")
    if sha256(args.feature_cache / "features.npz") != cache_metadata["features_sha256"]:
        raise RuntimeError("Feature cache hash verification failed")
    if sha256(args.inputs / "metadata.csv.gz") != input_hashes["metadata_sha256"]:
        raise RuntimeError("E1 metadata hash verification failed")
    for name, expected in input_hashes["source_label_sha256"].items():
        path = args.inputs / "source_labels" / f"{name}.npz"
        if sha256(path) != expected:
            raise RuntimeError(f"Source-label package hash verification failed: {name}")
    features = sparse.load_npz(args.feature_cache / "features.npz").tocsr()
    metadata = load_metadata(args.inputs / "metadata.csv.gz")
    if features.shape[0] != len(metadata):
        raise ValueError("Feature and metadata row counts differ")
    source_dir = args.inputs / "source_labels"

    atomic_json(
        args.output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "target_label_vault_argument_present": False,
        },
    )
    run_official(args.output, features, metadata, source_dir, config)
    for group in config["target_groups"]:
        run_project(group, args.output, features, metadata, source_dir, config)

    prediction_files = sorted((args.output / "predictions").rglob("*"))
    prediction_files = [path for path in prediction_files if path.is_file()]
    seal = {
        "experiment_id": args.output.name,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(args.config),
        "feature_cache_sha256": sha256(args.feature_cache / "features.npz"),
        "input_metadata_sha256": sha256(args.inputs / "metadata.csv.gz"),
        "input_hash_manifest_sha256": sha256(args.inputs / "hashes.json"),
        "prediction_runner_sha256": sha256(Path(__file__)),
        "domain_module_sha256": sha256(
            Path(__file__).parent / "vultriage" / "domain.py"
        ),
        "prediction_files": {
            path.relative_to(args.output).as_posix(): sha256(path)
            for path in prediction_files
        },
        "evaluation_labels_accessed": False,
    }
    atomic_json(args.output / "prediction_seal.json", seal)
    log(f"sealed {len(prediction_files)} prediction artifacts")


if __name__ == "__main__":
    main()
