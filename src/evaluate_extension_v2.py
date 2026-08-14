"""Evaluate sealed extension-v2 predictions after label-vault release.

This evaluator is intentionally separate from prediction generation.  It first
checks every prediction seal and all frozen input hashes, then joins the
label-only DiverseVul vault and writes fold-level decisions.  All methods are
label-blind at prediction time; target labels are used only in this process.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from evaluate_e1 import (
    add_targets,
    matched_msp_sets,
    matched_score_sets,
    pooled_conformal_sets,
    probability_matrix,
    temperature_scale,
)
from vultriage.conformal import (
    conformal_sets,
    estimated_weight_support,
    kish_ess,
    mondrian_thresholds,
    weighted_conformal_sets,
)
from vultriage.data import load_config, sha256
from vultriage.metrics import triage_metrics
from vultriage.prom_adapter import prom_binary_adapter


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            output[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            output[key] = value
    return output


def operating_key(alpha_vulnerable: float, alpha_safe: float) -> str:
    return f"av{alpha_vulnerable:g}_as{alpha_safe:g}".replace(".", "p")


def label_vault(path: Path, target_metadata: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(target_metadata, "rt", encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    if len(rows) != len(metadata):
        raise ValueError("target label vault and metadata have different lengths")
    labels = np.empty(len(rows), dtype=np.int8)
    for position, (row, meta) in enumerate(zip(rows, metadata)):
        if int(row["position"]) != position or int(meta["position"]) != position:
            raise ValueError("target label positions are not contiguous")
        if row["row_id"] != meta["row_id"]:
            raise ValueError("target label row ID does not match target metadata")
        target = int(row["target"])
        if target not in (0, 1):
            raise ValueError("target labels must be binary")
        labels[position] = target
    return labels


def verify_prediction_seal(
    prediction_dir: Path,
    config_path: Path,
    source_metadata: Path,
    target_metadata: Path,
    package_summary: Path,
    expected_detector: str | None = None,
) -> dict[str, Any]:
    seal_path = prediction_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if expected_detector is not None and seal.get("detector") != expected_detector:
        raise RuntimeError(
            f"prediction seal detector mismatch: expected {expected_detector}, "
            f"observed {seal.get('detector')} in {prediction_dir}"
        )
    if seal.get("target_label_vault_argument_present") is not False or seal.get(
        "target_vulnerability_labels_accessed"
    ) is not False:
        raise RuntimeError(f"prediction seal admits target-label access: {prediction_dir}")
    if seal.get("config_sha256") != sha256(config_path):
        raise RuntimeError(f"config hash mismatch: {prediction_dir}")
    expected = {
        "source_metadata_sha256": sha256(source_metadata),
        "target_metadata_sha256": sha256(target_metadata),
        "source_package_summary_sha256": sha256(package_summary),
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"{key} mismatch in {prediction_dir}")
    runner_path = Path(__file__).with_name("run_extension_predict.py")
    if seal.get("runner_sha256") != sha256(runner_path):
        raise RuntimeError(f"runner hash mismatch in {prediction_dir}")
    expected_files = seal.get("prediction_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError(f"prediction seal has no file inventory: {prediction_dir}")
    actual_files = {
        path.relative_to(prediction_dir).as_posix()
        for path in (prediction_dir / "predictions").rglob("*")
        if path.is_file()
    }
    if set(expected_files) != actual_files:
        raise RuntimeError(f"prediction seal file inventory mismatch: {prediction_dir}")
    failures = []
    for relative, expected_hash in expected_files.items():
        path = prediction_dir / relative
        observed = sha256(path) if path.is_file() else None
        if observed != expected_hash:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"prediction seal verification failed: {failures[:3]}")
    return {
        "prediction_seal_sha256": sha256(seal_path),
        "detector": seal["detector"],
        "verified_files": len(expected_files),
    }


def support_features(
    calibration_labels: np.ndarray,
    calibration_weights: np.ndarray,
    target_weights: np.ndarray,
    domain_auroc: float,
    upper_clip: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    calibration_weights = np.asarray(calibration_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    clipped_calibration = np.clip(calibration_weights, 1.0 / upper_clip, upper_clip)
    clipped_target = np.clip(target_weights, 1.0 / upper_clip, upper_clip)
    class_ess = {
        label: kish_ess(clipped_calibration[calibration_labels == label])
        for label in (0, 1)
    }
    masses = {
        label: float(clipped_calibration[calibration_labels == label].sum())
        for label in (0, 1)
    }
    per_point_by_label = {
        label: clipped_target / (masses[label] + clipped_target) for label in (0, 1)
    }
    per_point = np.maximum.reduce([per_point_by_label[label] for label in (0, 1)])
    diagnostics = {
        "total_ess": kish_ess(clipped_calibration),
        "class_ess": class_ess,
        "maximum_infinity_mass": float(np.max(per_point)),
        "p99_infinity_mass": float(np.quantile(per_point, 0.99)),
        "domain_auroc": float(domain_auroc),
        "lower_clipping_fraction": float(np.mean(calibration_weights < 1.0 / upper_clip)),
        "upper_clipping_fraction": float(np.mean(calibration_weights > upper_clip)),
        "per_point_infinity_mass": per_point,
        "per_point_infinity_mass_by_label": per_point_by_label,
    }
    features = {
        "log1p_total_ess": float(np.log1p(diagnostics["total_ess"])),
        "log1p_safe_ess": float(np.log1p(class_ess[0])),
        "log1p_vulnerable_ess": float(np.log1p(class_ess[1])),
        "maximum_infinity_mass": diagnostics["maximum_infinity_mass"],
        "p99_infinity_mass": diagnostics["p99_infinity_mass"],
        "domain_auroc": diagnostics["domain_auroc"],
        "lower_clipping_fraction": diagnostics["lower_clipping_fraction"],
        "upper_clipping_fraction": diagnostics["upper_clipping_fraction"],
    }
    return features, diagnostics


def gate_decision(
    raw_sets: np.ndarray,
    features: dict[str, float],
    diagnostics: dict[str, Any],
    alpha: dict[int, float],
    gate: Any,
    mode: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config["support_rules"]
    class_ess = diagnostics["class_ess"]
    required = {
        label: max(
            float(settings["minimum_class_ess"]),
            ceil(float(settings["class_ess_multiplier_over_alpha"]) / alpha[label]),
        )
        for label in (0, 1)
    }
    ess_pass = diagnostics["total_ess"] >= float(settings["minimum_total_ess"]) and all(
        class_ess[label] >= required[label] for label in (0, 1)
    )
    infinity_pass = all(
        not np.any(
            np.asarray(diagnostics["per_point_infinity_mass_by_label"][label])
            > alpha[label]
        )
        for label in (0, 1)
    )
    if mode == "full":
        feature_order = [
            "log1p_total_ess",
            "log1p_safe_ess",
            "log1p_vulnerable_ess",
            "maximum_infinity_mass",
            "p99_infinity_mass",
            "domain_auroc",
            "lower_clipping_fraction",
            "upper_clipping_fraction",
        ]
        vector = np.asarray([[features[name] for name in feature_order]], dtype=float)
        finite = bool(np.isfinite(vector).all())
        probability = float(gate.predict_proba(vector)[0, 1]) if finite else float("nan")
        pass_gate = finite and probability < 0.5 and bool(np.any(raw_sets.sum(axis=1) == 1))
    elif mode == "ess_only":
        probability = float("nan")
        pass_gate = bool(ess_pass and np.any(raw_sets.sum(axis=1) == 1))
    elif mode == "infinity_only":
        probability = float("nan")
        pass_gate = bool(infinity_pass and np.any(raw_sets.sum(axis=1) == 1))
    else:
        raise ValueError(mode)
    adjusted = raw_sets.copy() if pass_gate else np.ones_like(raw_sets, dtype=bool)
    return adjusted, {
        "mode": mode,
        "passed": bool(pass_gate),
        "gate_probability": probability,
        "ess_pass": bool(ess_pass),
        "infinity_pass": bool(infinity_pass),
        "raw_has_singleton": bool(np.any(raw_sets.sum(axis=1) == 1)),
        "features": features,
        "diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"per_point_infinity_mass", "per_point_infinity_mass_by_label"}
        },
    }


def add_record(
    records: list[dict[str, Any]],
    detector: str,
    group: str,
    seed: int,
    method: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    sets: np.ndarray,
    alpha_vulnerable: float,
    alpha_safe: float,
    **extra: Any,
) -> dict[str, Any]:
    metrics = add_targets(
        triage_metrics(labels, probabilities, sets), alpha_vulnerable, alpha_safe
    )
    metrics["max_relative_violation"] = max(
        metrics["vulnerable_violation"] / alpha_vulnerable,
        metrics["safe_violation"] / alpha_safe,
    )
    record = {
        "detector": detector,
        "target_group": group,
        "seed": int(seed),
        "method": method,
        **metrics,
        **extra,
    }
    records.append(record)
    return record


def evaluate_fold(
    detector: str,
    group: str,
    seed: int,
    archive: Any,
    source_package: Any,
    target_labels: np.ndarray,
    config: dict[str, Any],
    gate: Any,
    metadata: dict[str, Any],
    decision_path: Path,
    expected_calibration_positions: np.ndarray,
    expected_target_positions: np.ndarray,
) -> list[dict[str, Any]]:
    cal_positions = archive["calibration_positions"].astype(int)
    target_positions = archive["target_positions"].astype(int)
    source_labels = source_package["labels"].astype(int)
    source_positions = source_package["source_positions"].astype(int)
    expected_calibration_positions = np.asarray(expected_calibration_positions, dtype=np.int32)
    expected_target_positions = np.asarray(expected_target_positions, dtype=np.int32)
    if not np.array_equal(cal_positions, expected_calibration_positions):
        raise ValueError(f"prediction calibration positions differ from frozen package for {group}")
    if not np.array_equal(target_positions, expected_target_positions):
        raise ValueError(f"prediction target positions differ from frozen package for {group}")
    if len(np.unique(cal_positions)) != len(cal_positions) or len(np.unique(target_positions)) != len(target_positions):
        raise ValueError(f"prediction positions are not unique for {group}")
    target_probability = np.asarray(archive["target_p_vulnerable"], dtype=float)
    if len(target_probability) != len(target_positions):
        raise ValueError(f"target probability length differs from target positions for {group}")
    if not np.isfinite(target_probability).all() or np.any((target_probability < 0) | (target_probability > 1)):
        raise ValueError(f"target probabilities are not finite probabilities for {group}")
    if np.any(cal_positions < source_positions.min()) or np.any(cal_positions > source_positions.max()):
        raise ValueError(f"invalid calibration positions for {group}")
    source_order = np.argsort(source_positions)
    sorted_positions = source_positions[source_order]
    local_indices = np.searchsorted(sorted_positions, cal_positions)
    if np.any(local_indices >= len(sorted_positions)) or not np.array_equal(sorted_positions[local_indices], cal_positions):
        raise ValueError(f"calibration positions are absent from source package for {group}")
    calibration_labels = source_labels[source_order[local_indices]]
    target_y = target_labels[target_positions].astype(int)
    cal_probability = np.asarray(archive["calibration_p_vulnerable"], dtype=float)
    if len(cal_probability) != len(cal_positions) or not np.isfinite(cal_probability).all() or np.any((cal_probability < 0) | (cal_probability > 1)):
        raise ValueError(f"calibration probabilities are invalid for {group}")
    cal_prob = probability_matrix(cal_probability)
    target_prob = probability_matrix(target_probability)
    temperature_prob, fitted_temperature = temperature_scale(cal_prob, calibration_labels, target_prob)
    raw_cal_ratio = archive["calibration_raw_ratio"].astype(float)
    raw_target_ratio = archive["target_raw_ratio"].astype(float)
    if len(raw_cal_ratio) != len(cal_positions) or len(raw_target_ratio) != len(target_positions):
        raise ValueError(f"density-ratio arrays differ from prediction positions for {group}")
    if not np.isfinite(raw_cal_ratio).all() or not np.isfinite(raw_target_ratio).all():
        raise ValueError(f"density-ratio arrays are not finite for {group}")
    domain_auroc = float(metadata["domain_diagnostics"]["domain_auroc"])
    if not np.isfinite(domain_auroc):
        raise ValueError(f"domain AUROC is not finite for {group}")
    records: list[dict[str, Any]] = []
    decisions: dict[str, np.ndarray] = {
        "target_positions": target_positions.astype(np.int32),
        "target_labels": target_y.astype(np.int8),
        "target_p_vulnerable": target_prob[:, 1].astype(np.float32),
    }
    # PROM-derived experts depend on one scalar alpha, not on the ordered
    # vulnerable/safe budget pair.  Cache each scalar call within a fold so
    # the 3x3 risk grid does not repeat the same local-neighbour search.
    prom_cache: dict[float, Any] = {}
    for av in config["risk_budgets"]["vulnerable"]:
        for ass in config["risk_budgets"]["safe"]:
            av, ass = float(av), float(ass)
            alpha = {0: ass, 1: av}
            key = operating_key(av, ass)
            thresholds = mondrian_thresholds(cal_prob, calibration_labels, alpha)
            unweighted_sets = conformal_sets(target_prob, thresholds)
            add_record(records, detector, group, seed, "unweighted_mondrian", target_y, target_prob[:, 1], unweighted_sets, av, ass)
            pooled = pooled_conformal_sets(cal_prob, calibration_labels, target_prob, min(av, ass))
            add_record(records, detector, group, seed, "pooled_split_conformal", target_y, target_prob[:, 1], pooled, av, ass)
            forced = np.zeros_like(target_prob, dtype=bool)
            forced[np.arange(len(target_prob)), target_prob.argmax(axis=1)] = True
            add_record(records, detector, group, seed, "forced_argmax", target_y, target_prob[:, 1], forced, av, ass)
            n_single = int(np.sum(unweighted_sets.sum(axis=1) == 1))
            msp = matched_msp_sets(target_prob, n_single)
            add_record(records, detector, group, seed, "msp_matched_unweighted_mondrian", target_y, target_prob[:, 1], msp, av, ass, matched_singleton_count=n_single)
            ent = add_record(records, detector, group, seed, "entropy_matched_unweighted_mondrian", target_y, target_prob[:, 1], msp, av, ass, matched_singleton_count=n_single, binary_ranking_equivalent_to_msp=True)
            temp_msp = matched_msp_sets(temperature_prob, n_single)
            add_record(records, detector, group, seed, "temperature_msp_matched_unweighted_mondrian", target_y, temperature_prob[:, 1], temp_msp, av, ass, matched_singleton_count=n_single, temperature=fitted_temperature)
            # The scalar PROM alpha is fixed conservatively for this budget cell.
            prom_alpha = min(av, ass)
            if prom_alpha not in prom_cache:
                prom_cache[prom_alpha] = prom_binary_adapter(
                    cal_prob,
                    calibration_labels,
                    target_prob,
                    alpha=prom_alpha,
                    seed=int(seed),
                )
            prom = prom_cache[prom_alpha]
            prom_methods = {
                "prom_derived_lac": prom.experts["lac"].prediction_sets,
                "prom_derived_topk": prom.experts["topk"].prediction_sets,
                "prom_derived_aps": prom.experts["aps"].prediction_sets,
                "prom_derived_raps": prom.experts["raps"].prediction_sets,
            }
            union_sets = np.ones_like(target_prob, dtype=bool)
            union_sets[prom.union.accepted] = False
            union_sets[prom.union.accepted, prom.predicted_labels[prom.union.accepted]] = True
            prom_methods["prom_derived_union"] = union_sets
            for prom_name, prom_sets in prom_methods.items():
                add_record(records, detector, group, seed, prom_name, target_y, target_prob[:, 1], prom_sets, av, ass, prom_scalar_alpha=min(av, ass), prom_expert_count=4)
            decisions[f"{key}__unweighted_mondrian"] = unweighted_sets.astype(np.uint8)
            decisions[f"{key}__msp_matched_unweighted_mondrian"] = msp.astype(np.uint8)

            for clip in config["density_ratio"]["candidate_upper_clips"]:
                clip = float(clip)
                cal_w = np.clip(raw_cal_ratio, 1.0 / clip, clip)
                target_w = np.clip(raw_target_ratio, 1.0 / clip, clip)
                weighted_sets, _ = weighted_conformal_sets(cal_prob, calibration_labels, cal_w, target_prob, target_w, alpha)
                add_record(records, detector, group, seed, f"estimated_weight_no_gate_clip_{clip:g}", target_y, target_prob[:, 1], weighted_sets, av, ass, weight_clip=clip)
                features, diagnostics = support_features(calibration_labels, cal_w, target_w, domain_auroc, clip)
                for mode, name in (("ess_only", "vultriage_ess_only"), ("infinity_only", "vultriage_infinity_only"), ("full", "vultriage_full_gate")):
                    adjusted, support = gate_decision(weighted_sets, features, diagnostics, alpha, gate, mode, config)
                    add_record(records, detector, group, seed, f"{name}_clip_{clip:g}", target_y, target_prob[:, 1], adjusted, av, ass, weight_clip=clip, support=support)
                if abs(clip - float(config["density_ratio"]["selected_upper_clip"])) < 1e-12:
                    n_gate_single = int(np.sum(adjusted.sum(axis=1) == 1))
                    msp_gate = matched_msp_sets(target_prob, n_gate_single)
                    add_record(records, detector, group, seed, "msp_matched_vultriage_full_gate", target_y, target_prob[:, 1], msp_gate, av, ass, matched_singleton_count=n_gate_single)
                    temp_gate = matched_msp_sets(temperature_prob, n_gate_single)
                    add_record(records, detector, group, seed, "temperature_msp_matched_vultriage_full_gate", target_y, temperature_prob[:, 1], temp_gate, av, ass, matched_singleton_count=n_gate_single, temperature=fitted_temperature)
                    decisions[f"{key}__vultriage_full_gate"] = adjusted.astype(np.uint8)
                    decisions[f"{key}__estimated_weight_no_gate"] = weighted_sets.astype(np.uint8)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = decision_path.with_suffix(decision_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **decisions)
    temporary.replace(decision_path)
    return records


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
    started = time.perf_counter()
    extension_config = load_config(args.config)
    inherit_path = Path(extension_config["detectors"]["hashing_sgd"]["inherit"])
    if not inherit_path.is_absolute():
        inherit_path = Path.cwd() / inherit_path
    inherited_config = load_config(inherit_path)
    config = dict(inherited_config)
    for key in ("protocol_version", "seeds", "risk_budgets", "support_gate", "calibration_size_sensitivity", "detectors"):
        if key in extension_config:
            config[key] = extension_config[key]
    package_summary = args.inputs / "package_summary.json"
    gate_seal = json.loads((args.gate / "gate_seal.json").read_text(encoding="utf-8"))
    if gate_seal.get("config_sha256") != sha256(args.config):
        raise RuntimeError("support gate was not fit under the frozen extension config")
    gate_model = joblib.load(args.gate / "support_gate.joblib")
    seals: dict[str, Any] = {}
    for detector in ("hashing", "codebert"):
        seals[detector] = verify_prediction_seal(
            prediction_dirs[detector],
            args.config,
            args.source_metadata,
            args.target_metadata,
            package_summary,
            expected_detector=detector,
        )
    labels = label_vault(args.label_vault, args.target_metadata)
    source_metadata_rows = read_rows(args.source_metadata)
    target_metadata_rows = read_rows(args.target_metadata)
    package_summary_payload = json.loads(package_summary.read_text(encoding="utf-8"))
    groups = list(package_summary_payload["selected_project_groups"])
    records: list[dict[str, Any]] = []
    args.output.mkdir(parents=True)
    for detector in ("hashing", "codebert"):
        prediction_dir = prediction_dirs[detector]
        for group in groups:
            source_package = np.load(args.inputs / "source_label_packages" / f"{group}.npz")
            target_package = np.load(args.inputs / "target_position_packages" / f"{group}.npz")
            expected_calibration_positions = source_package["source_positions"][source_package["role_codes"] == 2].astype(np.int32)
            expected_target_positions = target_package["target_positions"].astype(np.int32)
            for seed in config["seeds"]:
                path = prediction_dir / "predictions" / group / f"seed-{seed}.npz"
                meta_path = path.with_suffix(".json")
                if not path.is_file() or not meta_path.is_file():
                    raise RuntimeError(f"missing prediction cell: {detector}/{group}/{seed}")
                archive = np.load(path)
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("detector") != detector or metadata.get("target_group") != group or int(metadata.get("seed", -1)) != int(seed):
                    raise RuntimeError(f"prediction metadata identity mismatch: {detector}/{group}/{seed}")
                if metadata.get("target_vulnerability_labels_accessed") is not False:
                    raise RuntimeError(f"prediction metadata admits target-label access: {detector}/{group}/{seed}")
                records.extend(
                    evaluate_fold(
                        detector,
                        group,
                        int(seed),
                        archive,
                        source_package,
                        labels,
                        config,
                        gate_model,
                        metadata,
                        args.output / "decisions" / detector / group / f"seed-{seed}.npz",
                        expected_calibration_positions,
                        expected_target_positions,
                    )
                )
    flat = [flatten(record) for record in records]
    fieldnames = sorted({key for record in flat for key in record})
    metrics_path = args.output / "fold_seed_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    decision_files = sorted(path for path in (args.output / "decisions").rglob("*.npz") if path.is_file())
    result = {
        "experiment_id": args.output.name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_seals": seals,
        "prediction_directories": {
            detector: str(prediction_dirs[detector]) for detector in ("hashing", "codebert")
        },
        "target_groups": groups,
        "seeds": [int(seed) for seed in config["seeds"]],
        "risk_budgets": config["risk_budgets"],
        "evaluation_label_vault_sha256": sha256(args.label_vault),
        "target_metadata_sha256": sha256(args.target_metadata),
        "metric_rows": len(records),
        "metrics_sha256": sha256(metrics_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "decision_files": {path.relative_to(args.output).as_posix(): sha256(path) for path in decision_files},
        "target_labels_joined_after_prediction_seal": True,
        "platform": platform.platform(),
    }
    atomic_json(args.output / "evaluation_manifest.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError(f"positions are not contiguous in {path}")
    return rows


if __name__ == "__main__":
    main()
