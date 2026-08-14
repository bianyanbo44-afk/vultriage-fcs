"""Nested calibration-size sensitivity at the frozen primary budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_e1 import add_targets
from evaluate_extension_v2 import (
    gate_decision,
    label_vault,
    support_features,
    verify_prediction_seal,
)
from vultriage.conformal import conformal_sets, mondrian_thresholds, weighted_conformal_sets
from vultriage.data import load_config, sha256
from vultriage.metrics import triage_metrics


def effective_config(path: Path) -> dict:
    extension = load_config(path)
    inherit = Path(extension["detectors"]["hashing_sgd"]["inherit"])
    if not inherit.is_absolute():
        inherit = Path.cwd() / inherit
    config = dict(load_config(inherit))
    for key in ("protocol_version", "seeds", "risk_budgets", "support_gate", "calibration_size_sensitivity", "detectors"):
        if key in extension:
            config[key] = extension[key]
    return config


def rows(path: Path) -> list[dict[str, str]]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        result = list(csv.DictReader(stream))
    return result


def nested_indices(row_ids: list[str], fraction: float, repetition: int, salt: str, labels: np.ndarray) -> np.ndarray:
    keys = [hashlib.sha256(f"{salt}|{repetition}|{row_id}".encode()).digest() for row_id in row_ids]
    order = np.asarray(sorted(range(len(row_ids)), key=lambda index: keys[index]), dtype=np.int32)
    requested = max(int(np.ceil(float(fraction) * len(order))), 40)
    selected = list(order[: min(requested, len(order))])
    selected_set = set(int(x) for x in selected)
    for label in (0, 1):
        missing = max(0, 20 - int(np.sum(labels[selected] == label)))
        if missing:
            for index in order:
                if int(index) not in selected_set and int(labels[int(index)]) == label:
                    selected.append(int(index))
                    selected_set.add(int(index))
                    missing -= 1
                    if not missing:
                        break
        if missing:
            raise RuntimeError("calibration subset cannot retain 20 examples per class")
    return np.asarray(selected, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-root",
        type=Path,
        help="legacy parent containing hashing/ and codebert/ prediction seals",
    )
    parser.add_argument(
        "--hashing-predictions",
        type=Path,
        help="sealed hashing prediction directory",
    )
    parser.add_argument(
        "--codebert-predictions",
        type=Path,
        help="sealed CodeBERT prediction directory",
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--label-vault", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.predictions_root is not None:
        if args.hashing_predictions is not None or args.codebert_predictions is not None:
            parser.error("use either --predictions-root or the two detector-specific prediction paths")
        prediction_dirs = {
            "hashing": args.predictions_root / "hashing",
            "codebert": args.predictions_root / "codebert",
        }
    elif args.hashing_predictions is not None and args.codebert_predictions is not None:
        prediction_dirs = {
            "hashing": args.hashing_predictions,
            "codebert": args.codebert_predictions,
        }
    else:
        parser.error(
            "provide --predictions-root or both --hashing-predictions and --codebert-predictions"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    config = effective_config(args.config)
    package_summary_path = args.inputs / "package_summary.json"
    package_summary = json.loads(package_summary_path.read_text(encoding="utf-8"))
    groups = package_summary["selected_project_groups"]
    target_y = label_vault(args.label_vault, args.target_metadata)
    source_rows = rows(args.source_metadata)
    target_rows = rows(args.target_metadata)
    gate = joblib.load(args.gate / "support_gate.joblib")
    for detector in ("hashing", "codebert"):
        verify_prediction_seal(
            prediction_dirs[detector],
            args.config,
            args.source_metadata,
            args.target_metadata,
            package_summary_path,
            expected_detector=detector,
        )
    records: list[dict[str, object]] = []
    fractions = config["calibration_size_sensitivity"]["fractions"]
    repetitions = int(config["calibration_size_sensitivity"]["repetitions"])
    salt = config["calibration_size_sensitivity"]["nested_subsample_salt"]
    av, ass = 0.1, 0.2
    alpha = {0: ass, 1: av}
    for detector in ("hashing", "codebert"):
        for group in groups:
            package = np.load(args.inputs / "source_label_packages" / f"{group}.npz")
            source_labels = package["labels"].astype(int)
            source_positions = package["source_positions"].astype(int)
            source_order = np.argsort(source_positions)
            source_sorted = source_positions[source_order]
            expected_calibration_positions = source_positions[package["role_codes"].astype(np.int8) == 2]
            expected_target_positions = np.load(
                args.inputs / "target_position_packages" / f"{group}.npz"
            )["target_positions"].astype(np.int32)
            for seed in config["seeds"]:
                prediction = np.load(prediction_dirs[detector] / "predictions" / group / f"seed-{seed}.npz")
                metadata = json.loads((prediction_dirs[detector] / "predictions" / group / f"seed-{seed}.json").read_text(encoding="utf-8"))
                if metadata.get("detector") != detector or metadata.get("target_group") != group or int(metadata.get("seed", -1)) != int(seed):
                    raise RuntimeError(f"prediction metadata identity mismatch: {detector}/{group}/{seed}")
                if metadata.get("target_vulnerability_labels_accessed") is not False:
                    raise RuntimeError(f"prediction metadata admits target-label access: {detector}/{group}/{seed}")
                cal_pos = prediction["calibration_positions"].astype(int)
                target_pos = prediction["target_positions"].astype(int)
                if not np.array_equal(cal_pos, expected_calibration_positions):
                    raise RuntimeError(f"prediction calibration positions differ from frozen package for {group}")
                if not np.array_equal(target_pos, expected_target_positions):
                    raise RuntimeError(f"prediction target positions differ from frozen package for {group}")
                local = np.searchsorted(source_sorted, cal_pos)
                if np.any(local >= len(source_sorted)) or not np.array_equal(source_sorted[local], cal_pos):
                    raise RuntimeError(f"calibration position missing for {group}")
                cal_labels = source_labels[source_order[local]]
                target_prob = np.asarray(prediction["target_p_vulnerable"], dtype=float)
                target_sets_prob = np.column_stack([1.0 - target_prob, target_prob])
                target_weights_all = prediction["target_raw_ratio"].astype(float)
                cal_ratio_all = prediction["calibration_raw_ratio"].astype(float)
                cal_ids = [source_rows[int(pos)]["row_id"] for pos in cal_pos]
                for repetition in range(repetitions):
                    for fraction in fractions:
                        subset = nested_indices(cal_ids, float(fraction), repetition, salt, cal_labels)
                        subset_labels = cal_labels[subset]
                        subset_prob = np.column_stack([
                            1.0 - prediction["calibration_p_vulnerable"].astype(float)[subset],
                            prediction["calibration_p_vulnerable"].astype(float)[subset],
                        ])
                        unweighted = conformal_sets(target_sets_prob, mondrian_thresholds(subset_prob, subset_labels, alpha))
                        cal_w = np.clip(cal_ratio_all[subset], 0.05, 20.0)
                        target_w = np.clip(target_weights_all, 0.05, 20.0)
                        weighted, _ = weighted_conformal_sets(subset_prob, subset_labels, cal_w, target_sets_prob, target_w, alpha)
                        features, diagnostics = support_features(subset_labels, cal_w, target_w, float(metadata["domain_diagnostics"]["domain_auroc"]), 20.0)
                        full, support = gate_decision(weighted, features, diagnostics, alpha, gate, "full", config)
                        for method, sets, extra in (
                            ("unweighted_mondrian", unweighted, {}),
                            ("estimated_weight_no_gate_clip_20", weighted, {"weight_clip": 20.0}),
                            ("vultriage_full_gate_clip_20", full, {"support": support, "weight_clip": 20.0}),
                        ):
                            metrics = add_targets(
                                triage_metrics(target_y[target_pos], target_prob, sets),
                                av,
                                ass,
                            )
                            metrics["max_relative_violation"] = max(
                                metrics["vulnerable_violation"] / av,
                                metrics["safe_violation"] / ass,
                            )
                            records.append({"detector": detector, "target_group": group, "seed": int(seed), "repetition": repetition, "fraction": float(fraction), "method": method, **metrics, **extra})
    args.output.mkdir(parents=True)
    table = pd.DataFrame(records)
    table_path = args.output / "calibration_size_sensitivity.csv"
    table.to_csv(table_path, index=False)
    summary_metrics = [
        "vulnerable_miscoverage",
        "safe_miscoverage",
        "vulnerable_violation",
        "safe_violation",
        "max_relative_violation",
        "singleton_coverage",
        "review_load",
        "pr_auc",
        "brier",
    ]
    project = table.groupby(
        ["detector", "target_group", "fraction", "method"], as_index=False
    )[summary_metrics].mean()
    project_path = args.output / "calibration_size_project_summary.csv"
    project.to_csv(project_path, index=False)
    aggregate_rows: list[dict[str, object]] = []
    for key, subset in project.groupby(["detector", "fraction", "method"], sort=True):
        detector, fraction, method = key
        for metric in summary_metrics:
            values = subset[metric].to_numpy(float)
            aggregate_rows.append(
                {
                    "detector": detector,
                    "fraction": float(fraction),
                    "method": method,
                    "metric": metric,
                    "projects": len(values),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = args.output / "calibration_size_aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "primary_budget": {"alpha_vulnerable": av, "alpha_safe": ass},
        "fractions": fractions,
        "repetitions": repetitions,
        "minimum_per_class": 20,
        "rows": len(table),
        "table_sha256": sha256(table_path),
        "project_summary_rows": len(project),
        "project_summary_sha256": sha256(project_path),
        "aggregate_summary_rows": len(aggregate),
        "aggregate_summary_sha256": sha256(aggregate_path),
        "independent_unit": "target_project_group",
        "within_project_aggregation": "mean over technical seeds and deterministic subsample repetitions",
        "script_sha256": sha256(Path(__file__)),
        "label_vault_sha256": sha256(args.label_vault),
    }
    (args.output / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
