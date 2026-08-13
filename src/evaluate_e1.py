"""Verify sealed E1 predictions, then join labels and evaluate triage methods."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from vultriage.conformal import (
    conformal_sets,
    estimated_weight_support,
    mondrian_thresholds,
    weighted_conformal_sets,
)
from vultriage.data import load_config, sha256
from vultriage.metrics import triage_metrics


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_seal(
    prediction_dir: Path, config_path: Path, input_hash_path: Path, label_vault: Path
) -> dict[str, Any]:
    seal_path = prediction_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in seal["prediction_files"].items():
        path = prediction_dir / relative
        observed = sha256(path) if path.exists() else None
        if observed != expected:
            failures.append(
                {"file": relative, "expected": expected, "observed": observed}
            )
    if failures:
        raise RuntimeError(f"Prediction seal verification failed: {failures[:3]}")
    if sha256(config_path) != seal["config_sha256"]:
        raise RuntimeError("Evaluation config differs from sealed prediction config")
    frozen_inputs = json.loads(input_hash_path.read_text(encoding="utf-8"))
    if sha256(label_vault) != frozen_inputs["evaluation_label_vault_sha256"]:
        raise RuntimeError("Evaluation label vault hash verification failed")
    if sha256(input_hash_path) != seal["input_hash_manifest_sha256"]:
        raise RuntimeError("E1 input hash manifest differs from prediction seal")
    return {
        "prediction_seal_sha256": sha256(seal_path),
        "verified_files": len(seal["prediction_files"]),
        "prediction_generation_accessed_evaluation_labels": bool(
            seal["evaluation_labels_accessed"]
        ),
    }


def probability_matrix(p_vulnerable: np.ndarray) -> np.ndarray:
    p = np.asarray(p_vulnerable, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid vulnerable probabilities")
    return np.column_stack([1.0 - p, p])


def pooled_conformal_sets(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    target_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scores = 1.0 - calibration_probabilities[
        np.arange(len(calibration_labels)), calibration_labels
    ]
    level = min(1.0, ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    threshold = float(np.quantile(scores, level, method="higher"))
    return (1.0 - target_probabilities) <= threshold


def matched_msp_sets(
    target_probabilities: np.ndarray, requested_singletons: int
) -> np.ndarray:
    """Match a singleton count using target scores but no target labels."""

    probabilities = np.asarray(target_probabilities, dtype=float)
    n = len(probabilities)
    k = max(0, min(int(requested_singletons), n))
    sets = np.ones((n, 2), dtype=bool)
    if k == 0:
        return sets
    confidence = probabilities.max(axis=1)
    return matched_score_sets(probabilities, confidence, k)


def matched_score_sets(
    probabilities: np.ndarray, acceptance_scores: np.ndarray, requested_singletons: int
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.asarray(acceptance_scores, dtype=float)
    if scores.shape != (len(probabilities),):
        raise ValueError("Acceptance scores and probabilities must align")
    n = len(probabilities)
    k = max(0, min(int(requested_singletons), n))
    sets = np.ones((n, 2), dtype=bool)
    if k == 0:
        return sets
    selected = np.argsort(-scores, kind="stable")[:k]
    forced = probabilities.argmax(axis=1)
    sets[selected] = False
    sets[selected, forced[selected]] = True
    return sets


def prom_compatible_lac_credibility(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    target_probabilities: np.ndarray,
) -> np.ndarray:
    """Label-conditional LAC p-value for the forced predicted label.

    This implements a reproducible Prom-compatible global-calibration expert.
    It is not the official Prom ensemble, which additionally uses local feature
    weighting, four nonconformity functions, and majority voting.
    """

    predicted = target_probabilities.argmax(axis=1)
    credibility = np.empty(len(target_probabilities), dtype=float)
    for label in (0, 1):
        calibration_scores = 1.0 - calibration_probabilities[
            calibration_labels == label, label
        ]
        indices = np.flatnonzero(predicted == label)
        if len(calibration_scores) == 0:
            raise ValueError(f"Prom-compatible calibration class {label} is empty")
        test_scores = 1.0 - target_probabilities[indices, label]
        ordered = np.sort(calibration_scores)
        greater_or_equal = len(ordered) - np.searchsorted(
            ordered, test_scores, side="left"
        )
        credibility[indices] = (greater_or_equal + 1.0) / (len(ordered) + 1.0)
    return credibility


def temperature_scale(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    target_probabilities: np.ndarray,
) -> tuple[np.ndarray, float]:
    epsilon = 1e-7
    p_cal = np.clip(calibration_probabilities[:, 1], epsilon, 1.0 - epsilon)
    logits_cal = np.log(p_cal / (1.0 - p_cal))

    def objective(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        scaled = 1.0 / (1.0 + np.exp(-logits_cal / temperature))
        return float(
            -np.mean(
                calibration_labels * np.log(np.clip(scaled, epsilon, 1.0))
                + (1 - calibration_labels)
                * np.log(np.clip(1.0 - scaled, epsilon, 1.0))
            )
        )

    fitted = minimize_scalar(
        objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded"
    )
    temperature = float(np.exp(fitted.x))
    p_target = np.clip(target_probabilities[:, 1], epsilon, 1.0 - epsilon)
    logits_target = np.log(p_target / (1.0 - p_target))
    scaled_target = 1.0 / (1.0 + np.exp(-logits_target / temperature))
    return probability_matrix(scaled_target), temperature


def support_adjusted_sets(
    raw_sets: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_weights: np.ndarray,
    target_weights: np.ndarray,
    alpha: dict[int, float],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config["support_rules"]
    decision = estimated_weight_support(
        calibration_labels,
        calibration_weights,
        alpha,
        minimum_total_ess=float(settings["minimum_total_ess"]),
        minimum_class_ess=float(settings["minimum_class_ess"]),
        class_ess_multiplier_over_alpha=float(
            settings["class_ess_multiplier_over_alpha"]
        ),
    )
    adjusted = raw_sets.copy()
    reasons = list(decision.reasons)
    per_point_unsupported = np.zeros(len(target_weights), dtype=bool)
    for label in (0, 1):
        mass = float(calibration_weights[calibration_labels == label].sum())
        infinity_mass = target_weights / (mass + target_weights)
        per_point_unsupported |= infinity_mass > alpha[label]
    positive_neighbours = {
        label: int(np.sum(calibration_weights[calibration_labels == label] > 0))
        for label in (0, 1)
    }
    for label in (0, 1):
        if positive_neighbours[label] < int(settings["minimum_positive_neighbours"]):
            reasons.append(f"class_{label}_positive_neighbours<{settings['minimum_positive_neighbours']}")
    if per_point_unsupported.any():
        reasons.append("test_point_infinity_mass_exceeds_budget")
    if reasons:
        adjusted[:] = True
    if np.all(adjusted.sum(axis=1) != 1):
        reasons.append("no_singleton_outputs")
        adjusted[:] = True
    return adjusted, {
        "supported": not reasons,
        "global_reasons": reasons,
        "total_ess": decision.total_ess,
        "class_ess": decision.class_ess,
        "positive_neighbours": positive_neighbours,
        "per_point_unsupported": int(per_point_unsupported.sum()),
        "per_point_unsupported_rate": float(per_point_unsupported.mean()),
    }


def add_targets(
    metrics: dict[str, Any], alpha_vulnerable: float, alpha_safe: float
) -> dict[str, Any]:
    output = dict(metrics)
    vulnerable_difference = float(metrics["vulnerable_miscoverage"] - alpha_vulnerable)
    safe_difference = float(metrics["safe_miscoverage"] - alpha_safe)
    output.update(
        {
            "alpha_vulnerable": alpha_vulnerable,
            "alpha_safe": alpha_safe,
            "vulnerable_signed_target_difference": vulnerable_difference,
            "safe_signed_target_difference": safe_difference,
            "vulnerable_violation": max(0.0, vulnerable_difference),
            "safe_violation": max(0.0, safe_difference),
            "absolute_target_difference_sum": abs(vulnerable_difference)
            + abs(safe_difference),
        }
    )
    return output


def evaluate_one(
    track: str,
    group: str,
    seed: int,
    archive: Any,
    labels: np.ndarray,
    config: dict[str, Any],
    decision_path: Path,
) -> list[dict[str, Any]]:
    calibration_positions = archive["calibration_positions"].astype(int)
    target_positions = archive["target_positions"].astype(int)
    calibration_labels = labels[calibration_positions].astype(int)
    target_labels = labels[target_positions].astype(int)
    calibration_probabilities = probability_matrix(
        archive["calibration_p_vulnerable"]
    )
    target_probabilities = probability_matrix(archive["target_p_vulnerable"])
    temperature_probabilities, fitted_temperature = temperature_scale(
        calibration_probabilities, calibration_labels, target_probabilities
    )
    prom_credibility = prom_compatible_lac_credibility(
        calibration_probabilities, calibration_labels, target_probabilities
    )
    records: list[dict[str, Any]] = []
    decisions: dict[str, np.ndarray] = {
        "target_positions": target_positions.astype(np.int32),
        "target_labels": target_labels.astype(np.int8),
        "target_p_vulnerable": target_probabilities[:, 1].astype(np.float32),
    }

    for alpha_vulnerable in config["risk_budgets"]["vulnerable"]:
        for alpha_safe in config["risk_budgets"]["safe"]:
            alpha_vulnerable = float(alpha_vulnerable)
            alpha_safe = float(alpha_safe)
            alpha = {0: alpha_safe, 1: alpha_vulnerable}
            common = {
                "track": track,
                "target_group": group,
                "seed": int(seed),
                "alpha_vulnerable": alpha_vulnerable,
                "alpha_safe": alpha_safe,
            }

            thresholds = mondrian_thresholds(
                calibration_probabilities, calibration_labels, alpha
            )
            unweighted_sets = conformal_sets(target_probabilities, thresholds)
            unweighted = add_targets(
                triage_metrics(
                    target_labels, target_probabilities[:, 1], unweighted_sets
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            records.append(
                {**common, "method": "unweighted_mondrian", **unweighted}
            )
            operating_key = (
                f"av{alpha_vulnerable:g}_as{alpha_safe:g}".replace(".", "p")
            )
            decisions[f"{operating_key}__unweighted_mondrian"] = (
                unweighted_sets.astype(np.uint8)
            )

            pooled_sets = pooled_conformal_sets(
                calibration_probabilities,
                calibration_labels,
                target_probabilities,
                min(alpha_vulnerable, alpha_safe),
            )
            pooled = add_targets(
                triage_metrics(
                    target_labels, target_probabilities[:, 1], pooled_sets
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            records.append({**common, "method": "pooled_split_conformal", **pooled})

            forced_sets = np.zeros_like(target_probabilities, dtype=bool)
            forced_sets[
                np.arange(len(target_probabilities)), target_probabilities.argmax(axis=1)
            ] = True
            forced = add_targets(
                triage_metrics(
                    target_labels, target_probabilities[:, 1], forced_sets
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            records.append({**common, "method": "forced_argmax", **forced})

            unweighted_singletons = int(
                np.sum(unweighted_sets.sum(axis=1) == 1)
            )
            msp_unweighted_sets = matched_msp_sets(
                target_probabilities, unweighted_singletons
            )
            msp_unweighted = add_targets(
                triage_metrics(
                    target_labels,
                    target_probabilities[:, 1],
                    msp_unweighted_sets,
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            msp_unweighted["matched_singleton_count"] = unweighted_singletons
            records.append(
                {
                    **common,
                    "method": "msp_matched_unweighted_mondrian",
                    **msp_unweighted,
                }
            )
            entropy_unweighted = dict(msp_unweighted)
            entropy_unweighted["binary_ranking_equivalent_to_msp"] = True
            records.append(
                {
                    **common,
                    "method": "entropy_matched_unweighted_mondrian",
                    **entropy_unweighted,
                }
            )
            temperature_unweighted_sets = matched_msp_sets(
                temperature_probabilities, unweighted_singletons
            )
            temperature_unweighted = add_targets(
                triage_metrics(
                    target_labels,
                    temperature_probabilities[:, 1],
                    temperature_unweighted_sets,
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            temperature_unweighted["temperature"] = fitted_temperature
            records.append(
                {
                    **common,
                    "method": "temperature_msp_matched_unweighted_mondrian",
                    **temperature_unweighted,
                }
            )
            prom_unweighted_sets = matched_score_sets(
                target_probabilities,
                prom_credibility,
                unweighted_singletons,
            )
            prom_unweighted = add_targets(
                triage_metrics(
                    target_labels,
                    target_probabilities[:, 1],
                    prom_unweighted_sets,
                ),
                alpha_vulnerable,
                alpha_safe,
            )
            prom_unweighted.update(
                {
                    "prom_variant": "global_label_conditional_lac_credibility",
                    "official_prom_ensemble": False,
                }
            )
            records.append(
                {
                    **common,
                    "method": "prom_compatible_lac_matched_unweighted_mondrian",
                    **prom_unweighted,
                }
            )
            decisions[
                f"{operating_key}__msp_matched_unweighted_mondrian"
            ] = msp_unweighted_sets.astype(np.uint8)
            decisions[
                f"{operating_key}__temperature_msp_matched_unweighted_mondrian"
            ] = temperature_unweighted_sets.astype(np.uint8)
            decisions[
                f"{operating_key}__prom_compatible_lac_matched_unweighted_mondrian"
            ] = prom_unweighted_sets.astype(np.uint8)

            if track == "official_chronological":
                continue

            raw_calibration_ratio = archive["calibration_raw_ratio"].astype(float)
            raw_target_ratio = archive["target_raw_ratio"].astype(float)
            selected_clip = float(
                config["density_ratio"]["selected_upper_clip"]
            )
            for clip in config["density_ratio"]["candidate_upper_clips"]:
                clip = float(clip)
                calibration_weights = np.clip(
                    raw_calibration_ratio, 1.0 / clip, clip
                )
                target_weights = np.clip(raw_target_ratio, 1.0 / clip, clip)
                raw_weighted_sets, _ = weighted_conformal_sets(
                    calibration_probabilities,
                    calibration_labels,
                    calibration_weights,
                    target_probabilities,
                    target_weights,
                    alpha,
                )
                raw_weighted = add_targets(
                    triage_metrics(
                        target_labels,
                        target_probabilities[:, 1],
                        raw_weighted_sets,
                    ),
                    alpha_vulnerable,
                    alpha_safe,
                )
                records.append(
                    {
                        **common,
                        "method": f"estimated_weight_no_support_clip_{clip:g}",
                        **raw_weighted,
                    }
                )
                adjusted_sets, support = support_adjusted_sets(
                    raw_weighted_sets,
                    calibration_labels,
                    calibration_weights,
                    target_weights,
                    alpha,
                    config,
                )
                adjusted = add_targets(
                    triage_metrics(
                        target_labels,
                        target_probabilities[:, 1],
                        adjusted_sets,
                    ),
                    alpha_vulnerable,
                    alpha_safe,
                )
                adjusted["support"] = support
                records.append(
                    {
                        **common,
                        "method": f"vultriage_clip_{clip:g}",
                        **adjusted,
                    }
                )
                if clip == selected_clip:
                    decisions[
                        f"{operating_key}__estimated_weight_no_support"
                    ] = raw_weighted_sets.astype(np.uint8)
                    decisions[f"{operating_key}__vultriage"] = (
                        adjusted_sets.astype(np.uint8)
                    )
                    requested_singletons = int(
                        np.sum(adjusted_sets.sum(axis=1) == 1)
                    )
                    msp_sets = matched_msp_sets(
                        target_probabilities, requested_singletons
                    )
                    msp = add_targets(
                        triage_metrics(
                            target_labels,
                            target_probabilities[:, 1],
                            msp_sets,
                        ),
                        alpha_vulnerable,
                        alpha_safe,
                    )
                    msp["matched_singleton_count"] = requested_singletons
                    records.append(
                        {**common, "method": "msp_matched_vultriage", **msp}
                    )
                    decisions[f"{operating_key}__msp_matched_vultriage"] = (
                        msp_sets.astype(np.uint8)
                    )
                    entropy = dict(msp)
                    entropy["binary_ranking_equivalent_to_msp"] = True
                    records.append(
                        {**common, "method": "entropy_matched_vultriage", **entropy}
                    )
                    temperature_sets = matched_msp_sets(
                        temperature_probabilities, requested_singletons
                    )
                    temperature_metrics = add_targets(
                        triage_metrics(
                            target_labels,
                            temperature_probabilities[:, 1],
                            temperature_sets,
                        ),
                        alpha_vulnerable,
                        alpha_safe,
                    )
                    temperature_metrics["temperature"] = fitted_temperature
                    records.append(
                        {
                            **common,
                            "method": "temperature_msp_matched_vultriage",
                            **temperature_metrics,
                        }
                    )
                    decisions[
                        f"{operating_key}__temperature_msp_matched_vultriage"
                    ] = temperature_sets.astype(np.uint8)
                    prom_sets = matched_score_sets(
                        target_probabilities,
                        prom_credibility,
                        requested_singletons,
                    )
                    prom_metrics = add_targets(
                        triage_metrics(
                            target_labels,
                            target_probabilities[:, 1],
                            prom_sets,
                        ),
                        alpha_vulnerable,
                        alpha_safe,
                    )
                    prom_metrics.update(
                        {
                            "prom_variant": "global_label_conditional_lac_credibility",
                            "official_prom_ensemble": False,
                        }
                    )
                    records.append(
                        {
                            **common,
                            "method": "prom_compatible_lac_matched_vultriage",
                            **prom_metrics,
                        }
                    )
                    decisions[
                        f"{operating_key}__prom_compatible_lac_matched_vultriage"
                    ] = prom_sets.astype(np.uint8)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = decision_path.with_suffix(decision_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **decisions)
    temporary.replace(decision_path)
    return records


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            output[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-vault", type=Path, required=True)
    parser.add_argument("--input-hashes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    seal_verification = verify_seal(
        args.predictions,
        args.config,
        args.input_hashes,
        args.label_vault,
    )
    if seal_verification["prediction_generation_accessed_evaluation_labels"]:
        raise RuntimeError("Prediction seal admits evaluation-label access")
    labels = np.load(args.label_vault)["labels"].astype(int)
    config = load_config(args.config)
    records: list[dict[str, Any]] = []

    for path in sorted((args.predictions / "predictions" / "official").glob("seed-*.npz")):
        seed = int(path.stem.split("-")[-1])
        records.extend(
            evaluate_one(
                "official_chronological",
                "official",
                seed,
                np.load(path),
                labels,
                config,
                args.output / "decisions" / "official" / f"seed-{seed}.npz",
            )
        )
    for group in config["target_groups"]:
        for path in sorted((args.predictions / "predictions" / group).glob("seed-*.npz")):
            seed = int(path.stem.split("-")[-1])
            records.extend(
                evaluate_one(
                    "project_disjoint",
                    group,
                    seed,
                    np.load(path),
                    labels,
                    config,
                    args.output / "decisions" / group / f"seed-{seed}.npz",
                )
            )

    metrics_path = args.output / "fold_seed_metrics.csv"
    flat = [flatten(record) for record in records]
    fieldnames = sorted({key for record in flat for key in record})
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    decision_files = sorted((args.output / "decisions").rglob("*.npz"))
    result = {
        "experiment_id": args.predictions.name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_verification": seal_verification,
        "evaluation_label_vault_sha256": sha256(args.label_vault),
        "metric_rows": len(records),
        "metrics_sha256": sha256(metrics_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "conformal_module_sha256": sha256(
            Path(__file__).parent / "vultriage" / "conformal.py"
        ),
        "metrics_module_sha256": sha256(
            Path(__file__).parent / "vultriage" / "metrics.py"
        ),
        "decision_files": {
            path.relative_to(args.output).as_posix(): sha256(path)
            for path in decision_files
        },
    }
    atomic_json(args.output / "evaluation_manifest.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
