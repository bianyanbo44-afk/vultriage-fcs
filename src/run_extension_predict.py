"""Generate sealed extension-v2 predictions without target labels.

The runner handles the frozen hashing-SGD and CodeBERT-logistic detectors.  It
only receives source labels from the target-specific source package; the
DiverseVul label vault is deliberately not an argument and is never opened.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from vultriage.codebert import fit_logistic_head_scaled
from vultriage.data import load_config, sha256
from vultriage.domain import clipped_weight_summary, crossfit_density_ratio


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError(f"positions are not contiguous in {path}")
    return rows


def balanced_class_weight(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if (counts == 0).any():
        raise ValueError("detector training requires both classes")
    return {label: float(len(labels) / (2 * counts[label])) for label in (0, 1)}


def fit_hashing_detector(
    features: Any,
    train_positions: np.ndarray,
    train_labels: np.ndarray,
    validation_positions: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> tuple[Any, float, float]:
    grid = config["sgd_grid"]
    weights = balanced_class_weight(train_labels)
    best_alpha: float | None = None
    best_score = -np.inf
    for alpha in grid["alpha"]:
        model = SGDClassifier(
            loss=grid["loss"],
            penalty=grid["penalty"],
            alpha=float(alpha),
            class_weight=weights,
            random_state=int(seed),
            max_iter=int(grid["epochs"]),
            tol=None,
            average=True,
        )
        model.fit(features[train_positions], train_labels)
        score = float(
            average_precision_score(
                validation_labels,
                model.predict_proba(features[validation_positions])[:, 1],
            )
        )
        if score > best_score or (
            score == best_score and (best_alpha is None or float(alpha) > best_alpha)
        ):
            best_alpha, best_score = float(alpha), score
    assert best_alpha is not None
    final_positions = np.concatenate([train_positions, validation_positions])
    final_labels = np.concatenate([train_labels, validation_labels])
    final = SGDClassifier(
        loss=grid["loss"],
        penalty=grid["penalty"],
        alpha=best_alpha,
        class_weight=balanced_class_weight(final_labels),
        random_state=int(seed),
        max_iter=int(grid["epochs"]),
        tol=None,
        average=True,
    )
    final.fit(features[final_positions], final_labels)
    return final, best_alpha, best_score


def load_hashing_features(path: Path, expected_rows: int | None = None) -> Any:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("labels_used") is not False or metadata.get("labels_serialized", False):
        raise ValueError("hashing cache is not label-free")
    if expected_rows is not None and int(metadata["rows"]) != expected_rows:
        raise ValueError("hashing cache row count differs from metadata")
    if sha256(path / "features.npz") != metadata["features_sha256"]:
        raise ValueError("hashing feature hash mismatch")
    return sparse.load_npz(path / "features.npz").tocsr()


def load_codebert_embeddings(path: Path, manifest: Path) -> tuple[np.memmap, dict[str, Any]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("labels_used") is not False:
        raise ValueError("CodeBERT cache is not label-free")
    if metadata.get("manifest_sha256") != sha256(manifest):
        raise ValueError("CodeBERT manifest hash mismatch")
    shape = tuple(int(value) for value in metadata["shape"])
    embedding_path = path / "embeddings.f32"
    if sha256(embedding_path) != metadata["embeddings_sha256"]:
        raise ValueError("CodeBERT embedding hash mismatch")
    return np.memmap(embedding_path, mode="r", dtype=np.float32, shape=shape), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=("hashing", "codebert"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-features", type=Path)
    parser.add_argument("--target-features", type=Path)
    parser.add_argument("--codebert-cache", type=Path)
    parser.add_argument("--codebert-manifest", type=Path)
    parser.add_argument(
        "--groups",
        nargs="+",
        help="optional subset of frozen target groups for independent parallel runs",
    )
    parser.add_argument(
        "--seed-workers",
        type=int,
        default=1,
        help="parallel independent CodeBERT seed fits; outputs remain seed-addressed",
    )
    parser.add_argument(
        "--reuse-codebert-seeds",
        action="store_true",
        help=(
            "fit the deterministic liblinear CodeBERT head once per project and "
            "materialize identical predictions for all frozen seed addresses"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seed_workers < 1:
        raise ValueError("--seed-workers must be positive")
    if args.reuse_codebert_seeds and args.detector != "codebert":
        raise ValueError("--reuse-codebert-seeds is only valid for the CodeBERT detector")
    started = time.perf_counter()
    extension_config = load_config(args.config)
    inherit_path = Path(extension_config["detectors"]["hashing_sgd"]["inherit"])
    if not inherit_path.is_absolute():
        inherit_path = Path.cwd() / inherit_path
    inherited_config = load_config(inherit_path)
    # Extension-v2 freezes the inherited detector/density implementation while
    # keeping its own dataset, seed, risk, and support declarations.
    config = dict(inherited_config)
    for key in ("protocol_version", "seeds", "risk_budgets", "support_gate", "calibration_size_sensitivity", "detectors"):
        if key in extension_config:
            config[key] = extension_config[key]
    package_summary = json.loads(
        (args.inputs / "package_summary.json").read_text(encoding="utf-8")
    )
    source_rows = read_rows(args.source_metadata)
    target_rows = read_rows(args.target_metadata)
    source_ids = [row["row_id"] for row in source_rows]
    target_ids = [row["row_id"] for row in target_rows]
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("metadata row IDs are not unique")
    source_dir = args.inputs / "source_label_packages"
    target_dir = args.inputs / "target_position_packages"
    all_groups = list(package_summary["selected_project_groups"])
    groups = all_groups if args.groups is None else list(args.groups)
    if not groups or len(set(groups)) != len(groups) or not set(groups).issubset(set(all_groups)):
        raise ValueError("--groups must be a non-empty subset of the frozen target groups")
    if args.detector == "hashing":
        if args.source_features is None or args.target_features is None:
            raise ValueError("hashing detector needs source and target feature caches")
        # The source cache retains the original PrimeVul feature positions;
        # target-specific packages point into that cache after exact dedup.
        source_features = load_hashing_features(args.source_features)
        target_features = load_hashing_features(args.target_features, len(target_rows))
        feature_meta: dict[str, Any] = {
            "source_features_sha256": sha256(args.source_features / "features.npz"),
            "target_features_sha256": sha256(args.target_features / "features.npz"),
        }
    else:
        if args.codebert_cache is None or args.codebert_manifest is None:
            raise ValueError("CodeBERT detector needs embedding cache and manifest")
        combined, embedding_meta = load_codebert_embeddings(args.codebert_cache, args.codebert_manifest)
        expected_source = int(embedding_meta["source_rows"])
        if expected_source != len(source_rows) or int(embedding_meta["target_rows"]) != len(target_rows):
            raise ValueError("CodeBERT manifest split does not match extension metadata")
        source_features = combined[:expected_source]
        target_features = combined[expected_source:]
        feature_meta = {"codebert_cache_metadata_sha256": sha256(args.codebert_cache / "metadata.json")}

    args.output.mkdir(parents=True)
    atomic_json(
        args.output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "detector": args.detector,
            "target_label_vault_argument_present": False,
            "target_labels_accessed": False,
            "seed_workers": int(args.seed_workers if args.detector == "codebert" else 1),
            "seed_reuse_mode": (
                "deterministic_liblinear_replicates"
                if args.reuse_codebert_seeds
                else "independent"
            ),
            "seed_reuse_reference": int(config["seeds"][0]) if args.reuse_codebert_seeds else None,
            "selected_project_groups": groups,
            **feature_meta,
        },
    )
    for group in groups:
        package_path = source_dir / f"{group}.npz"
        target_package_path = target_dir / f"{group}.npz"
        package = np.load(package_path)
        target_package = np.load(target_package_path)
        source_positions = package["source_positions"].astype(np.int32)
        feature_positions = package["feature_positions"].astype(np.int32)
        labels = package["labels"].astype(int)
        role_codes = package["role_codes"].astype(np.int8)
        target_positions = target_package["target_positions"].astype(np.int32)
        train_mask = role_codes == 0
        validation_mask = role_codes == 1
        calibration_mask = role_codes == 2
        if not (train_mask.any() and validation_mask.any() and calibration_mask.any()):
            raise ValueError(f"source role is empty for {group}")
        # Hashing packages index the original sparse cache; CodeBERT packages
        # index the source-then-target embedding manifest.  Both packages keep
        # source_positions for provenance, but only the hashing cache has a
        # separate feature_positions array.
        if args.detector == "hashing":
            detector_positions = feature_positions
        else:
            detector_positions = source_positions
        train_features = detector_positions[train_mask]
        validation_features = detector_positions[validation_mask]
        calibration_features = detector_positions[calibration_mask]
        train_labels = labels[train_mask]
        validation_labels = labels[validation_mask]
        calibration_labels = labels[calibration_mask]
        calibration_x = source_features[calibration_features]
        target_x = target_features[target_positions]
        domain_x = sparse.vstack([calibration_x, target_x], format="csr") if args.detector == "hashing" else np.vstack([calibration_x, target_x])
        domain_ids = [source_ids[int(position)] for position in source_positions[calibration_mask]] + [target_ids[int(position)] for position in target_positions]
        domain_labels = np.concatenate(
            [
                np.zeros(calibration_x.shape[0], dtype=int),
                np.ones(target_x.shape[0], dtype=int),
            ]
        )
        density = crossfit_density_ratio(
            domain_x,
            domain_ids,
            domain_labels,
            folds=int(config["density_ratio"]["crossfit_folds"]),
            alpha=float(config["density_ratio"]["domain_alpha"]),
            epochs=int(config["density_ratio"]["domain_epochs"]),
            seed=int(config["seeds"][0]),
            salt=f"vultriage-extension-domain-v2|{args.detector}|{group}",
        )
        n_cal = int(calibration_x.shape[0])
        sensitivity = {
            str(clip): {
                "calibration": clipped_weight_summary(density.raw_ratios[:n_cal], float(clip)),
                "target": clipped_weight_summary(density.raw_ratios[n_cal:], float(clip)),
            }
            for clip in config["density_ratio"]["candidate_upper_clips"]
        }
        codebert_scaled = None
        if args.detector == "codebert":
            # Fit the frozen source-train scaler once per target project and
            # share the transformed matrices across technical seeds.
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            train_matrix = np.asarray(source_features[train_features], dtype=np.float32)
            validation_matrix = np.asarray(source_features[validation_features], dtype=np.float32)
            scaler.fit(train_matrix)
            codebert_scaled = (
                scaler.transform(train_matrix),
                scaler.transform(validation_matrix),
                scaler,
            )
        fitted_codebert: dict[int, tuple[Any, float, float, float]] = {}
        if args.detector == "codebert":
            train_scaled, validation_scaled, scaler = codebert_scaled

            def fit_one_codebert(seed_value: int) -> tuple[int, Any, float, float, float]:
                fit_started = time.perf_counter()
                selected = fit_logistic_head_scaled(
                    train_scaled,
                    train_labels,
                    validation_scaled,
                    validation_labels,
                    config["detectors"]["frozen_codebert"]["c_grid"],
                    int(seed_value),
                    scaler,
                )
                return (
                    int(seed_value),
                    selected.model,
                    float(selected.c_value),
                    float(selected.validation_pr_auc),
                    time.perf_counter() - fit_started,
                )

            seed_values = [int(seed) for seed in config["seeds"]]
            fit_seed_values = seed_values[:1] if args.reuse_codebert_seeds else seed_values
            worker_count = min(int(args.seed_workers), len(fit_seed_values))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for seed_value, model_value, c_value, score_value, fit_seconds in executor.map(
                    fit_one_codebert, fit_seed_values
                ):
                    fitted_codebert[seed_value] = (model_value, c_value, score_value, fit_seconds)
            if args.reuse_codebert_seeds:
                reference = fitted_codebert[seed_values[0]]
                for seed_value in seed_values[1:]:
                    fitted_codebert[seed_value] = reference
        for seed in config["seeds"]:
            fit_started = time.perf_counter()
            if args.detector == "hashing":
                model, parameter, validation_score = fit_hashing_detector(
                    source_features,
                    train_features,
                    train_labels,
                    validation_features,
                    validation_labels,
                    int(seed),
                    config,
                )
                fit_seconds = time.perf_counter() - fit_started
            else:
                # The resulting matrices are read-only inputs to the
                # unchanged liblinear candidate/final fits.
                assert codebert_scaled is not None
                model, parameter, validation_score, fit_seconds = fitted_codebert[int(seed)]
            calibration_probability = model.predict_proba(source_features[calibration_features])[:, 1]
            target_probability = model.predict_proba(target_features[target_positions])[:, 1]
            path = args.output / "predictions" / group / f"seed-{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_npz(
                path,
                calibration_positions=source_positions[calibration_mask],
                calibration_feature_positions=calibration_features,
                calibration_p_vulnerable=calibration_probability.astype(np.float32),
                calibration_raw_ratio=density.raw_ratios[:n_cal].astype(np.float32),
                calibration_domain_probability=density.target_probabilities[:n_cal].astype(np.float32),
                target_positions=target_positions,
                target_p_vulnerable=target_probability.astype(np.float32),
                target_raw_ratio=density.raw_ratios[n_cal:].astype(np.float32),
                target_domain_probability=density.target_probabilities[n_cal:].astype(np.float32),
            )
            atomic_json(
                path.with_suffix(".json"),
                {
                    "detector": args.detector,
                    "target_group": group,
                    "seed": int(seed),
                    "selected_parameter": float(parameter),
                    "source_validation_pr_auc": float(validation_score),
                    "density_estimator_seed": int(config["seeds"][0]),
                    "domain_diagnostics": density.diagnostics,
                    "weight_sensitivity": sensitivity,
                    "head_fit_seconds": float(fit_seconds),
                    "technical_seed_reused": bool(args.reuse_codebert_seeds),
                    "seed_reused_from": int(config["seeds"][0]) if args.reuse_codebert_seeds else None,
                    "target_vulnerability_labels_accessed": False,
                },
            )
            log(f"detector={args.detector} project={group} seed={seed} complete")

    prediction_files = sorted(path for path in (args.output / "predictions").rglob("*") if path.is_file())
    seal = {
        "experiment_id": args.output.name,
        "detector": args.detector,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": sha256(args.config),
        "source_metadata_sha256": sha256(args.source_metadata),
        "target_metadata_sha256": sha256(args.target_metadata),
        "source_package_summary_sha256": sha256(args.inputs / "package_summary.json"),
        "runner_sha256": sha256(Path(__file__)),
        "prediction_files": {path.relative_to(args.output).as_posix(): sha256(path) for path in prediction_files},
        "target_label_vault_argument_present": False,
        "target_vulnerability_labels_accessed": False,
        "seed_reuse_mode": (
            "deterministic_liblinear_replicates"
            if args.reuse_codebert_seeds
            else "independent"
        ),
        "seed_reuse_reference": int(config["seeds"][0]) if args.reuse_codebert_seeds else None,
        "selected_project_groups": groups,
    }
    atomic_json(args.output / "prediction_seal.json", seal)
    log(f"sealed {len(prediction_files)} prediction artifacts for {args.detector}")


if __name__ == "__main__":
    main()
