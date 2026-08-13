"""Run the preregistered CPU smoke experiment.

This script deliberately implements only E0: one hashing detector, the official
chronological track, and the frozen OpenSSL project-disjoint fold. It writes raw
probabilities before computing target metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

from vultriage.conformal import (
    conformal_sets,
    estimated_weight_support,
    mondrian_thresholds,
    weighted_conformal_sets,
)
from vultriage.data import iter_manifest, load_config, sha256, stable_bucket
from vultriage.metrics import triage_metrics


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def manifest_index(path: Path) -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in iter_manifest(path)}


def iter_selected_rows(
    data_dir: Path,
    index: dict[str, dict[str, str]],
    selected_row_ids: set[str],
) -> Iterator[tuple[dict[str, str], str]]:
    by_location = {
        (row["source_file"], int(row["line_number"])): row
        for row_id, row in index.items()
        if row_id in selected_row_ids
    }
    for filename in ("primevul_train.jsonl", "primevul_valid.jsonl", "primevul_test.jsonl"):
        path = data_dir / filename
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                manifest_row = by_location.get((filename, line_number))
                if manifest_row is None:
                    continue
                payload = json.loads(line)
                yield manifest_row, str(payload["func"])


def vectorizer_from_config(config: dict[str, Any]) -> HashingVectorizer:
    settings = config["hashing_vectorizer"]
    return HashingVectorizer(
        n_features=int(settings["n_features"]),
        ngram_range=tuple(settings["ngram_range"]),
        alternate_sign=bool(settings["alternate_sign"]),
        norm=settings["norm"],
        lowercase=bool(settings["lowercase"]),
        token_pattern=settings["token_pattern"],
        dtype=np.float32,
    )


def fit_epoch(
    model: SGDClassifier,
    vectorizer: HashingVectorizer,
    data_dir: Path,
    index: dict[str, dict[str, str]],
    train_ids: set[str],
    batch_size: int,
    first_epoch: bool,
) -> None:
    code_batch: list[str] = []
    label_batch: list[int] = []
    seen = 0
    for manifest_row, code in iter_selected_rows(data_dir, index, train_ids):
        code_batch.append(code)
        label_batch.append(int(manifest_row["target"]))
        if len(code_batch) >= batch_size:
            features = vectorizer.transform(code_batch)
            if first_epoch and seen == 0:
                model.partial_fit(features, np.asarray(label_batch), classes=np.array([0, 1]))
            else:
                model.partial_fit(features, np.asarray(label_batch))
            seen += len(code_batch)
            code_batch.clear()
            label_batch.clear()
    if code_batch:
        features = vectorizer.transform(code_batch)
        if first_epoch and seen == 0:
            model.partial_fit(features, np.asarray(label_batch), classes=np.array([0, 1]))
        else:
            model.partial_fit(features, np.asarray(label_batch))
        seen += len(code_batch)
    log(f"trained on {seen} rows")


def predict_rows(
    model: SGDClassifier,
    vectorizer: HashingVectorizer,
    data_dir: Path,
    index: dict[str, dict[str, str]],
    row_ids: set[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    code_batch: list[str] = []
    meta_batch: list[dict[str, str]] = []

    def flush() -> None:
        if not code_batch:
            return
        features = vectorizer.transform(code_batch)
        probabilities = model.predict_proba(features)
        for metadata, probability in zip(meta_batch, probabilities):
            output.append(
                {
                    "row_id": metadata["row_id"],
                    "target": int(metadata["target"]),
                    "project_group": metadata["project_group"],
                    "origin_split": metadata["origin_split"],
                    "p_safe": float(probability[0]),
                    "p_vulnerable": float(probability[1]),
                }
            )
        code_batch.clear()
        meta_batch.clear()

    for metadata, code in iter_selected_rows(data_dir, index, row_ids):
        code_batch.append(code)
        meta_batch.append(metadata)
        if len(code_batch) >= batch_size:
            flush()
    flush()
    output.sort(key=lambda row: row["row_id"])
    return output


def rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([row["target"] for row in rows], dtype=int)
    probabilities = np.asarray(
        [[row["p_safe"], row["p_vulnerable"]] for row in rows], dtype=float
    )
    return labels, probabilities


def official_validation_role(row: dict[str, str], config: dict[str, Any]) -> str:
    settings = config["official_validation_calibration_split"]
    bucket = stable_bucket(row["commit_id"], settings["salt"], 100)
    return "model_validation" if bucket < int(settings["model_validation_end"]) else "calibration"


def choose_alpha(
    data_dir: Path,
    index: dict[str, dict[str, str]],
    train_ids: set[str],
    validation_ids: set[str],
    config: dict[str, Any],
) -> float:
    vectorizer = vectorizer_from_config(config)
    grid = config["sgd_grid"]
    best_alpha = None
    best_pr_auc = -np.inf
    label_counts = defaultdict(int)
    for row_id in train_ids:
        label_counts[int(index[row_id]["target"])] += 1
    total = sum(label_counts.values())
    class_weight = {
        label: total / (2.0 * label_counts[label]) for label in (0, 1)
    }
    for alpha in grid["alpha"]:
        model = SGDClassifier(
            loss=grid["loss"],
            penalty=grid["penalty"],
            alpha=float(alpha),
            class_weight=class_weight,
            random_state=int(config["seeds"][0]),
            learning_rate="optimal",
            average=True,
        )
        for epoch in range(int(grid["epochs"])):
            fit_epoch(
                model,
                vectorizer,
                data_dir,
                index,
                train_ids,
                int(grid["batch_size"]),
                first_epoch=epoch == 0,
            )
        validation_rows = predict_rows(
            model,
            vectorizer,
            data_dir,
            index,
            validation_ids,
            int(grid["batch_size"]),
        )
        labels, probabilities = rows_to_arrays(validation_rows)
        score = float(average_precision_score(labels, probabilities[:, 1]))
        log(f"source-only alpha={alpha:g}, validation PR-AUC={score:.6f}")
        if score > best_pr_auc or (score == best_pr_auc and (best_alpha is None or alpha > best_alpha)):
            best_alpha = float(alpha)
            best_pr_auc = score
    assert best_alpha is not None
    log(f"selected alpha={best_alpha:g} on source-only validation")
    return best_alpha


def fit_final_model(
    data_dir: Path,
    index: dict[str, dict[str, str]],
    train_ids: set[str],
    alpha: float,
    config: dict[str, Any],
) -> tuple[SGDClassifier, HashingVectorizer]:
    grid = config["sgd_grid"]
    vectorizer = vectorizer_from_config(config)
    label_counts = defaultdict(int)
    for row_id in train_ids:
        label_counts[int(index[row_id]["target"])] += 1
    total = sum(label_counts.values())
    class_weight = {
        label: total / (2.0 * label_counts[label]) for label in (0, 1)
    }
    model = SGDClassifier(
        loss=grid["loss"],
        penalty=grid["penalty"],
        alpha=alpha,
        class_weight=class_weight,
        random_state=int(config["seeds"][0]),
        learning_rate="optimal",
        average=True,
    )
    for epoch in range(int(grid["epochs"])):
        log(f"final model epoch {epoch + 1}/{grid['epochs']}")
        fit_epoch(
            model,
            vectorizer,
            data_dir,
            index,
            train_ids,
            int(grid["batch_size"]),
            first_epoch=epoch == 0,
        )
    return model, vectorizer


def fit_domain_ratio(
    vectorizer: HashingVectorizer,
    data_dir: Path,
    index: dict[str, dict[str, str]],
    source_ids: set[str],
    target_ids: set[str],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    settings = config["density_ratio"]
    rng = random.Random(int(config["seeds"][0]))
    source_sample = sorted(source_ids)
    target_sample = sorted(target_ids)
    if len(source_sample) > len(target_sample):
        source_sample = rng.sample(source_sample, len(target_sample))
    elif len(target_sample) > len(source_sample):
        target_sample = rng.sample(target_sample, len(source_sample))
    selected_ids = set(source_sample) | set(target_sample)
    code: list[str] = []
    labels: list[int] = []
    ordered_ids: list[str] = []
    target_set = set(target_sample)
    for metadata, function in iter_selected_rows(data_dir, index, selected_ids):
        ordered_ids.append(metadata["row_id"])
        code.append(function)
        labels.append(int(metadata["row_id"] in target_set))
    features = vectorizer.transform(code)
    domain_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(settings["domain_alpha"]),
        class_weight="balanced",
        random_state=int(config["seeds"][0]),
        max_iter=int(settings["domain_epochs"]),
        tol=None,
    )
    domain_model.fit(features, np.asarray(labels))
    domain_probability = domain_model.predict_proba(features)[:, 1]
    domain_auc = float(average_precision_score(np.asarray(labels), domain_probability))
    clip = float(settings["selected_upper_clip"])
    probability_by_id = dict(zip(ordered_ids, domain_probability))
    ratio_by_id = {
        row_id: min(clip, max(1.0 / clip, probability / max(1e-8, 1.0 - probability)))
        for row_id, probability in probability_by_id.items()
    }

    missing = (source_ids | target_ids) - ratio_by_id.keys()
    if missing:
        missing_code: list[str] = []
        missing_ids: list[str] = []
        for metadata, function in iter_selected_rows(data_dir, index, set(missing)):
            missing_ids.append(metadata["row_id"])
            missing_code.append(function)
        probabilities = domain_model.predict_proba(vectorizer.transform(missing_code))[:, 1]
        for row_id, probability in zip(missing_ids, probabilities):
            ratio_by_id[row_id] = min(
                clip, max(1.0 / clip, probability / max(1e-8, 1.0 - probability))
            )
    return ratio_by_id, {
        "balanced_domain_average_precision": domain_auc,
        "training_source_n": len(source_sample),
        "training_target_n": len(target_sample),
        "clip": clip,
    }


def evaluate_track(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: dict[str, Any],
    calibration_weights: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    labels_calibration, probabilities_calibration = rows_to_arrays(calibration_rows)
    labels_test, probabilities_test = rows_to_arrays(test_rows)
    results: dict[str, Any] = {}
    for alpha_vulnerable in config["risk_budgets"]["vulnerable"]:
        for alpha_safe in config["risk_budgets"]["safe"]:
            alpha = {0: float(alpha_safe), 1: float(alpha_vulnerable)}
            key = f"av={alpha_vulnerable:g}|as={alpha_safe:g}"
            if calibration_weights is None:
                thresholds = mondrian_thresholds(
                    probabilities_calibration, labels_calibration, alpha
                )
                sets = conformal_sets(probabilities_test, thresholds)
                support = {"supported": True, "reasons": []}
            else:
                assert test_weights is not None
                decision = estimated_weight_support(
                    labels_calibration,
                    calibration_weights,
                    alpha,
                    minimum_total_ess=float(
                        config["support_rules"]["minimum_total_ess"]
                    ),
                    minimum_class_ess=float(
                        config["support_rules"]["minimum_class_ess"]
                    ),
                    class_ess_multiplier_over_alpha=float(
                        config["support_rules"]["class_ess_multiplier_over_alpha"]
                    ),
                )
                sets, _ = weighted_conformal_sets(
                    probabilities_calibration,
                    labels_calibration,
                    calibration_weights,
                    probabilities_test,
                    test_weights,
                    alpha,
                )
                if not decision.supported:
                    sets[:] = True
                support = {
                    "supported": decision.supported,
                    "reasons": list(decision.reasons),
                    "total_ess": decision.total_ess,
                    "class_ess": decision.class_ess,
                }
            metrics = triage_metrics(
                labels_test, probabilities_test[:, 1], sets
            )
            metrics["support"] = support
            metrics["alpha_vulnerable"] = alpha_vulnerable
            metrics["alpha_safe"] = alpha_safe
            results[key] = metrics
    return results


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], phase: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "phase",
            "row_id",
            "project_group",
            "origin_split",
            "p_safe",
            "p_vulnerable",
            "target",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({"phase": phase, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    index = manifest_index(args.manifest)
    log(f"loaded {len(index)} manifest rows")

    official_train = {
        row_id for row_id, row in index.items() if row["origin_split"] == "train"
    }
    official_validation = [
        row for row in index.values() if row["origin_split"] == "valid"
    ]
    official_model_validation = {
        row["row_id"]
        for row in official_validation
        if official_validation_role(row, config) == "model_validation"
    }
    official_calibration = {
        row["row_id"]
        for row in official_validation
        if official_validation_role(row, config) == "calibration"
    }
    official_test = {
        row_id for row_id, row in index.items() if row["origin_split"] == "test"
    }

    chosen_alpha = choose_alpha(
        args.data_dir,
        index,
        official_train,
        official_model_validation,
        config,
    )
    model, vectorizer = fit_final_model(
        args.data_dir,
        index,
        official_train | official_model_validation,
        chosen_alpha,
        config,
    )

    prediction_path = args.output / "predictions.csv"
    official_calibration_rows = predict_rows(
        model,
        vectorizer,
        args.data_dir,
        index,
        official_calibration,
        int(config["sgd_grid"]["batch_size"]),
    )
    official_test_rows = predict_rows(
        model,
        vectorizer,
        args.data_dir,
        index,
        official_test,
        int(config["sgd_grid"]["batch_size"]),
    )
    write_prediction_csv(prediction_path, official_calibration_rows, "official_calibration")
    write_prediction_csv(prediction_path, official_test_rows, "official_test")
    prediction_hash_before_metrics = sha256(prediction_path)
    official_metrics = evaluate_track(
        official_calibration_rows, official_test_rows, config
    )

    target_group = "openssl"
    target_ids = {
        row_id for row_id, row in index.items() if row["project_group"] == target_group
    }
    source_ids = set(index) - target_ids
    source_train_ids = {
        row_id for row_id in source_ids if index[row_id]["source_file"] != ""
        and stable_bucket(index[row_id]["commit_id"], config["split_salt"], 100)
        < int(config["source_partition"]["train_end"])
    }
    source_validation_ids = {
        row_id
        for row_id in source_ids
        if int(config["source_partition"]["train_end"])
        <= stable_bucket(index[row_id]["commit_id"], config["split_salt"], 100)
        < int(config["source_partition"]["model_validation_end"])
    }
    source_calibration_ids = source_ids - source_train_ids - source_validation_ids

    chosen_project_alpha = choose_alpha(
        args.data_dir,
        index,
        source_train_ids,
        source_validation_ids,
        config,
    )
    project_model, project_vectorizer = fit_final_model(
        args.data_dir,
        index,
        source_train_ids | source_validation_ids,
        chosen_project_alpha,
        config,
    )
    project_calibration_rows = predict_rows(
        project_model,
        project_vectorizer,
        args.data_dir,
        index,
        source_calibration_ids,
        int(config["sgd_grid"]["batch_size"]),
    )
    project_target_rows = predict_rows(
        project_model,
        project_vectorizer,
        args.data_dir,
        index,
        target_ids,
        int(config["sgd_grid"]["batch_size"]),
    )
    write_prediction_csv(prediction_path, project_calibration_rows, "openssl_source_calibration")
    write_prediction_csv(prediction_path, project_target_rows, "openssl_target")
    sealed_prediction_hash = sha256(prediction_path)
    ratio_by_id, domain_diagnostics = fit_domain_ratio(
        project_vectorizer,
        args.data_dir,
        index,
        source_calibration_ids,
        target_ids,
        config,
    )
    calibration_weights = np.asarray(
        [ratio_by_id[row["row_id"]] for row in project_calibration_rows], dtype=float
    )
    target_weights = np.asarray(
        [ratio_by_id[row["row_id"]] for row in project_target_rows], dtype=float
    )
    project_metrics = {
        "unweighted": evaluate_track(
            project_calibration_rows, project_target_rows, config
        ),
        "estimated_weight": evaluate_track(
            project_calibration_rows,
            project_target_rows,
            config,
            calibration_weights,
            target_weights,
        ),
        "domain_diagnostics": domain_diagnostics,
    }

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count": os.cpu_count(),
    }
    atomic_json(args.output / "environment.json", environment)
    atomic_json(
        args.output / "metrics.json",
        {
            "experiment_id": "exp-e0-cpu-smoke",
            "selected_alpha": {
                "official": chosen_alpha,
                "openssl": chosen_project_alpha,
            },
            "official": official_metrics,
            "openssl": project_metrics,
            "prediction_hash_after_official_before_metrics": prediction_hash_before_metrics,
            "sealed_prediction_hash": sealed_prediction_hash,
            "elapsed_seconds": time.time() - started,
        },
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    atomic_json(args.output / "artifact_hashes.json", artifact_hashes)
    log(f"completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
