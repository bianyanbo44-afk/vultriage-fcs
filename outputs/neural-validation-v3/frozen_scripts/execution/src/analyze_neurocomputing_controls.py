"""Post-review allocation and source-only cost controls for frozen evaluation-v2.

The new selectors never accept target labels.  Target labels enter only the
retrospective action evaluation after the selectors have been constructed.
This is an exploratory audit of already observed outcomes, not preregistration
or independent confirmation.  Original experiment artifacts remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from functools import lru_cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

from analyze_neurocomputing_revision import action_counts


OPERATING_KEY = "av0p1_as0p2__"
REFERENCES = {
    "unweighted_mondrian": "unweighted_mondrian",
    "estimated_weight_no_gate": "weighted",
    "vultriage_full_gate": "gate",
}
COSTS = (10, 50)
FP_COST = 1.0
REVIEW_COST = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probability_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "Invalid probability vector")
    require(not ((values < 0) | (values > 1)).any(), "Probability outside [0, 1]")
    return values


def automation_count(value: int, n: int) -> int:
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
            "Automation count must be an integer")
    require(0 <= value <= n, "Automation count outside cohort")
    return int(value)


def prediction_sets(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values)
    require(values.shape == (n, 2) and np.isin(values, [0, 1]).all(), "Invalid prediction sets")
    return values.astype(bool)


def action_codes(sets: np.ndarray) -> np.ndarray:
    sets = np.asarray(sets)
    require(sets.ndim == 2 and sets.shape[1] == 2, "Invalid prediction sets")
    sets = prediction_sets(sets, len(sets))
    codes = np.full(len(sets), 2, dtype=np.uint8)
    single = sets.sum(axis=1) == 1
    codes[single & sets[:, 0]] = 0
    codes[single & sets[:, 1]] = 1
    return codes


def global_msp_matched(probability: np.ndarray, count: int) -> np.ndarray:
    """Reproduce original evaluator MSP, including safe argmax ties at p=0.5."""
    probability = probability_vector(probability)
    count = automation_count(count, len(probability))
    confidence = np.maximum(probability, 1.0 - probability)
    retained = np.argsort(-confidence, kind="stable")[:count]
    sets = np.ones((len(probability), 2), dtype=bool)
    sets[retained] = False
    sets[retained, (probability[retained] > 0.5).astype(int)] = True
    return sets


def class_side_matched(probability: np.ndarray, safe_count: int, vulnerable_count: int) -> np.ndarray:
    """Match both action counts with disjoint low/high probability tails.

    This is an asymmetric two-threshold selector over ALL points.  Its labels
    need not agree with p=0.5 argmax.  One stable ascending order is used for
    both tails, so a probability tie can never select a row on both sides.
    """
    probability = probability_vector(probability)
    safe_count = automation_count(safe_count, len(probability))
    vulnerable_count = automation_count(vulnerable_count, len(probability))
    require(safe_count + vulnerable_count <= len(probability), "Action counts exceed cohort")
    order = np.argsort(probability, kind="stable")
    safe = order[:safe_count]
    vulnerable = order[len(order) - vulnerable_count:] if vulnerable_count else order[:0]
    sets = np.ones((len(probability), 2), dtype=bool)
    sets[safe] = (True, False)
    sets[vulnerable] = (False, True)
    return sets


def fit_source_isotonic(probability: np.ndarray, source_labels: np.ndarray) -> IsotonicRegression:
    """Fit only the sealed SOURCE calibration scores and source labels."""
    probability = probability_vector(probability)
    source_labels = np.asarray(source_labels)
    require(len(probability) > 0 and source_labels.shape == probability.shape
            and np.isin(source_labels, [0, 1]).all(), "Invalid source calibration labels")
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                              out_of_bounds="clip").fit(probability, source_labels.astype(int))


def source_cost_matched(eta: np.ndarray, count: int, miss_cost: float,
                        false_alert_cost: float = FP_COST) -> np.ndarray:
    """Choose the cheaper label, then retain exactly count lowest-cost rows.

    eta is a score calibrated on source labels only.  A tied action cost chooses
    auto-vulnerable; equal acceptance costs use stable target archive row order.
    Review cost is constant across equal-volume policies and does not affect
    their ranking.  No target outcome is an argument to this function.
    """
    eta = probability_vector(eta)
    count = automation_count(count, len(eta))
    require(not isinstance(miss_cost, (bool, np.bool_))
            and not isinstance(false_alert_cost, (bool, np.bool_))
            and np.isfinite(miss_cost) and np.isfinite(false_alert_cost)
            and miss_cost > 0 and false_alert_cost > 0, "Invalid action costs")
    safe_cost = miss_cost * eta
    vulnerable_cost = false_alert_cost * (1.0 - eta)
    vulnerable_label = vulnerable_cost <= safe_cost
    order = np.argsort(np.minimum(safe_cost, vulnerable_cost), kind="stable")
    retained = order[:count]
    sets = np.ones((len(eta), 2), dtype=bool)
    sets[retained] = False
    sets[retained, vulnerable_label[retained].astype(int)] = True
    return sets


def build_controls(probability: np.ndarray, references: dict[str, np.ndarray],
                   source_eta: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Construct controls from frozen probabilities and label-free policy sets."""
    probability = probability_vector(probability)
    require(set(references) == set(REFERENCES), "Incomplete reference policies")
    decisions = {name: prediction_sets(sets, len(probability)).copy()
                 for name, sets in references.items()}
    for name, suffix in REFERENCES.items():
        codes = action_codes(decisions[name])
        count = int(np.sum(codes != 2))
        decisions[f"msp_matched_{suffix}"] = global_msp_matched(probability, count)
        decisions[f"class_matched_{suffix}"] = class_side_matched(
            probability, int(np.sum(codes == 0)), int(np.sum(codes == 1)))
        if source_eta is not None and suffix in ("weighted", "gate"):
            source_eta = probability_vector(source_eta)
            require(source_eta.shape == probability.shape, "Source-calibrated scores do not align")
            for miss_cost in COSTS:
                decisions[f"source_iso_cost{miss_cost}_matched_{suffix}"] = source_cost_matched(
                    source_eta, count, miss_cost)
    decisions["forced_argmax"] = global_msp_matched(probability, len(probability))
    decisions["all_review"] = np.ones((len(probability), 2), dtype=bool)
    return decisions


def verified_path(root: Path, relative: str, expected_hash: str) -> tuple[Path, str]:
    root = root.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Sealed path escapes input root: {relative}")
    require(path.is_file(), f"Missing sealed artifact: {path}")
    observed = sha256(path)
    require(observed == str(expected_hash).upper(), f"Sealed hash mismatch: {path}")
    return path, observed


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_inputs(evaluation: Path, public_root: Path, predictions_root: Path,
                         inputs: Path, config_path: Path, runner_path: Path) -> dict[str, Any]:
    """Verify every source package, prediction NPZ/JSON, and decision archive."""
    manifest_path = evaluation / "evaluation_manifest.json"
    manifest = json_file(manifest_path)
    public_manifest = public_root / "evaluation-v2/evaluation_manifest.json"
    require(sha256(manifest_path) == sha256(public_manifest), "Public/original evaluation manifests differ")
    require(manifest.get("target_labels_joined_after_prediction_seal") is True,
            "Original evaluation lacks post-seal label provenance")
    groups, seeds = manifest["target_groups"], manifest["seeds"]
    require(len(groups) == len(set(groups)) == 24 and len(seeds) == len(set(seeds)) == 5,
            "Unexpected frozen project or technical-seed cohort")
    metrics_path, metrics_hash = verified_path(public_root, "evaluation-v2/fold_seed_metrics.csv",
                                               manifest["metrics_sha256"])
    verified_path(evaluation, "fold_seed_metrics.csv", metrics_hash)
    summary_path = inputs / "package_summary.json"
    summary_hash = sha256(summary_path)
    summary = json_file(summary_path)
    require(summary["selected_project_groups"] == groups, "Source/evaluation project cohorts differ")
    require(summary["extension_config_sha256"] == sha256(config_path), "Frozen config differs")
    source_metadata, source_hash = verified_path(inputs, "source_metadata.csv.gz", summary["source_metadata_sha256"])
    target_metadata, target_hash = verified_path(inputs, "target_metadata.csv.gz", summary["target_metadata_sha256"])
    require(target_hash == manifest["target_metadata_sha256"], "Evaluation target metadata differs")
    _, source_vault_hash = verified_path(inputs, "source_labels.csv.gz", summary["source_label_vault_sha256"])
    package_records = summary["folds"]
    require(len(package_records) == 24 and {r["target_group"] for r in package_records} == set(groups),
            "Incomplete source fold inventory")
    package_hashes, package_paths = {}, {}
    for record in package_records:
        group = record["target_group"]
        require(record["target_position_package_contains_labels"] is False,
                "Target-position package declares label access")
        for kind in ("source_label_package", "target_position_package"):
            filename = Path(record[kind]).name
            require(filename == f"{group}.npz", "Unexpected source fold artifact identity")
            directory = "source_label_packages" if kind == "source_label_package" else "target_position_packages"
            relative = f"{directory}/{filename}"
            path, digest = verified_path(inputs, relative, record[f"{kind}_sha256"])
            package_hashes[relative] = digest
            package_paths[(group, kind)] = path
    prediction_hashes, seal_receipts, prediction_paths = {}, {}, {}
    for detector, final_directory in (("hashing", "hashing-v8"), ("codebert", "codebert-v2")):
        root = predictions_root / final_directory
        seal_path = root / "prediction_seal.json"
        seal_hash = sha256(seal_path)
        require(seal_hash == manifest["prediction_seals"][detector]["prediction_seal_sha256"],
                f"Original prediction seal differs for {detector}")
        seal = json_file(seal_path)
        require(seal["detector"] == detector and seal["selected_project_groups"] == groups,
                "Prediction seal identity or cohort differs")
        require(seal.get("target_label_vault_argument_present") is False
                and seal.get("target_vulnerability_labels_accessed") is False,
                "Prediction seal declares target-label access")
        for field, observed in (("config_sha256", sha256(config_path)),
                                ("runner_sha256", sha256(runner_path)),
                                ("source_package_summary_sha256", summary_hash),
                                ("source_metadata_sha256", source_hash),
                                ("target_metadata_sha256", target_hash)):
            require(seal[field] == observed, f"Frozen prediction binding differs: {detector}/{field}")
        expected_inventory = {f"predictions/{group}/seed-{seed}.{suffix}"
                              for group in groups for seed in seeds for suffix in ("npz", "json")}
        require(set(seal["prediction_files"]) == expected_inventory, "Incomplete sealed prediction inventory")
        actual_inventory = {p.relative_to(root).as_posix() for p in (root / "predictions").rglob("*") if p.is_file()}
        require(actual_inventory == expected_inventory, "Unexpected prediction files outside seal")
        for relative, expected in sorted(seal["prediction_files"].items()):
            path, digest = verified_path(root, relative, expected)
            prediction_hashes[f"{detector}/{relative}"] = digest
            if path.suffix == ".npz":
                prediction_paths[(detector, path.parent.name, int(path.stem.split("-")[-1]))] = path
            else:
                metadata = json_file(path)
                require(metadata["detector"] == detector and metadata["target_group"] == path.parent.name
                        and int(metadata["seed"]) == int(path.stem.split("-")[-1]),
                        "Prediction metadata identity differs")
                require(metadata.get("target_vulnerability_labels_accessed") is False,
                        "Prediction metadata declares target-label access")
        expected_mode = "deterministic_liblinear_replicates" if detector == "codebert" else "independent"
        require(seal.get("seed_reuse_mode") == expected_mode, "Technical seed provenance differs")
        seal_receipts[detector] = dict(sha256=seal_hash, verified_files=len(expected_inventory),
                                       seed_reuse_mode=expected_mode)
    expected_decisions = {f"decisions/{detector}/{group}/seed-{seed}.npz"
                          for detector in ("hashing", "codebert") for group in groups for seed in seeds}
    require(set(manifest["decision_files"]) == expected_decisions, "Incomplete sealed decision inventory")
    actual_decisions = {p.relative_to(evaluation).as_posix() for p in (evaluation / "decisions").rglob("*") if p.is_file()}
    require(actual_decisions == expected_decisions, "Unexpected decision files outside manifest")
    decision_hashes = {relative: verified_path(evaluation, relative, expected)[1]
                       for relative, expected in sorted(manifest["decision_files"].items())}
    return dict(manifest=manifest, metrics_path=metrics_path, package_paths=package_paths,
                prediction_paths=prediction_paths,
                receipts=dict(original_evaluation_manifest_sha256=sha256(manifest_path),
                              original_metrics_sha256=metrics_hash, original_config_sha256=sha256(config_path),
                              original_prediction_runner_sha256=sha256(runner_path),
                              source_package_summary_sha256=summary_hash, source_metadata_sha256=sha256(source_metadata),
                              target_metadata_sha256=sha256(target_metadata), source_label_vault_sha256=source_vault_hash,
                              verified_source_fold_sha256=package_hashes, verified_prediction_seals=seal_receipts,
                              verified_prediction_file_sha256=prediction_hashes,
                              verified_decision_archive_sha256=decision_hashes))


def load_source_labels(source_path: Path, target_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as package:
        positions, labels, roles = package["source_positions"], package["labels"], package["role_codes"]
        require(positions.ndim == 1 and positions.dtype.kind in "iu"
                and len(np.unique(positions)) == len(positions) and (positions >= 0).all(),
                "Invalid source positions")
        require(labels.shape == positions.shape and np.isin(labels, [0, 1]).all(), "Invalid source labels")
        require(roles.shape == positions.shape and np.isin(roles, [0, 1, 2]).all(), "Invalid source role codes")
        calibration = roles == 2
        calibration_positions, calibration_labels = positions[calibration].copy(), labels[calibration].copy()
    with np.load(target_path, allow_pickle=False) as package:
        require("target_labels" not in package.files and "labels" not in package.files,
                "Target position package contains labels")
        target_positions = package["target_positions"].copy()
    require(target_positions.ndim == 1 and target_positions.dtype.kind in "iu"
            and len(np.unique(target_positions)) == len(target_positions) and (target_positions >= 0).all(),
            "Invalid target positions")
    return calibration_positions, calibration_labels, target_positions


@lru_cache(maxsize=16)
def adjacent_order_probabilities(n: int) -> np.ndarray:
    """Exact even-n empirical bootstrap mass of the middle adjacent ranks.

    Adapted from the independently validated numerical-audit pure helper at
    tmp/neurocomputing-review-v2/numerical-audit/empirical_median.py.  It is
    embedded here so published reproduction does not depend on tmp files.
    """
    require(isinstance(n, int) and not isinstance(n, bool) and n >= 2 and n % 2 == 0,
            "Exact median bootstrap requires an even sample size of at least two")
    middle = n // 2
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            left = ((i + 1) / n) ** middle - (i / n) ** middle
            right = ((n - j) / n) ** middle - ((n - j - 1) / n) ** middle
            weights[i, j] = comb(n, middle) * left * right
        for below in range(middle):
            for above in range(middle):
                equal = n - below - above
                coefficient = factorial(n) / (factorial(below) * factorial(equal) * factorial(above))
                weights[i, i] += coefficient * (i / n) ** below * (1 / n) ** equal * ((n - i - 1) / n) ** above
    require(bool(np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)),
            "Exact bootstrap mass does not sum to one")
    weights.setflags(write=False)
    return weights


def empirical_median_distribution(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "Invalid empirical bootstrap values")
    weights = adjacent_order_probabilities(len(values))
    ordered = np.sort(values)
    mass = {}
    for i in range(len(values)):
        for j in range(i, len(values)):
            value = float((ordered[i] + ordered[j]) / 2.0)
            mass[value] = mass.get(value, 0.0) + weights[i, j]
    support = np.asarray(sorted(mass), dtype=float)
    probabilities = np.asarray([mass[value] for value in support], dtype=float)
    require(bool(np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)),
            "Exact bootstrap distribution is not normalized")
    return support, probabilities


def empirical_median_interval(values: np.ndarray, lower: float = 0.025,
                              upper: float = 0.975) -> tuple[float, float]:
    require(0 <= lower <= upper <= 1, "Invalid bootstrap quantiles")
    support, probabilities = empirical_median_distribution(values)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return tuple(float(support[np.searchsorted(cumulative, q)]) for q in (lower, upper))


def paired_summary(frame: pd.DataFrame, outcome: str, baseline: str, candidate: str,
                   ) -> dict[str, Any]:
    subsets = [frame[frame.method == name] for name in (baseline, candidate)]
    require(all(not sub.target_group.duplicated().any() for sub in subsets), "Duplicate paired project")
    a, b = [sub.set_index("target_group")[outcome].sort_index() for sub in subsets]
    require(len(a) > 0 and set(a.index) == set(b.index), "Unmatched paired project cohorts")
    delta = (b.reindex(a.index) - a).to_numpy(float)
    require(np.isfinite(delta).all(), "Undefined paired endpoint")
    lower, upper = empirical_median_interval(delta)
    decreasing_is_better = outcome != "coverage"
    favorable = delta < -1e-12 if decreasing_is_better else delta > 1e-12
    unfavorable = delta > 1e-12 if decreasing_is_better else delta < -1e-12
    return dict(outcome=outcome, baseline=baseline, candidate=candidate, projects=len(delta),
                difference_definition="candidate_minus_baseline", favorable_direction="lower" if decreasing_is_better else "higher",
                median_difference=float(np.median(delta)), mean_difference=float(np.mean(delta)),
                ci_lower=lower, ci_upper=upper,
                favorable_projects=int(favorable.sum()), unfavorable_projects=int(unfavorable.sum()),
                ties=int((~favorable & ~unfavorable).sum()))


def controls_comparisons() -> list[tuple[str, str, int | None]]:
    pairs = []
    for reference, suffix in REFERENCES.items():
        pairs.extend([(f"msp_matched_{suffix}", reference, None),
                      (f"class_matched_{suffix}", reference, None)])
        if suffix in ("weighted", "gate"):
            pairs.extend((f"source_iso_cost{cost}_matched_{suffix}", reference, cost) for cost in COSTS)
    return pairs


def audit(evaluation: Path, public_root: Path, predictions_root: Path, inputs: Path,
          config_path: Path, runner_path: Path, output: Path) -> dict:
    require(not output.exists(), f"Audit output already exists: {output}")
    verified = verify_frozen_inputs(evaluation, public_root, predictions_root, inputs, config_path, runner_path)
    manifest = verified["manifest"]
    metric_frame = pd.read_csv(verified["metrics_path"])
    primary = metric_frame[(metric_frame.alpha_vulnerable == 0.1) & (metric_frame.alpha_safe == 0.2)]
    require(not primary.duplicated(["detector", "target_group", "seed", "method"]).any(), "Duplicate original primary metrics")
    primary = primary.set_index(["detector", "target_group", "seed", "method"])
    split_path = public_root / "contract-v1/fixed_split_projects.csv"
    split_frame = pd.read_csv(split_path)
    require(not split_frame.target_group.duplicated().any() and set(split_frame.target_group) == set(manifest["target_groups"]),
            "Contract split cohort differs")
    splits = split_frame.set_index("target_group").partition.to_dict()
    source_data = {group: load_source_labels(verified["package_paths"][(group, "source_label_package")],
                                          verified["package_paths"][(group, "target_position_package")])
                   for group in manifest["target_groups"]}
    rows, selection_rows, source_fit_rows, generated_hashes = [], [], [], {}
    output.mkdir(parents=True, exist_ok=False)
    for relative in sorted(manifest["decision_files"]):
        path = evaluation / relative
        detector, group = path.parts[-3:-1]
        seed = int(path.stem.split("-")[-1])
        cal_positions, cal_labels, target_positions = source_data[group]
        with np.load(verified["prediction_paths"][(detector, group, seed)], allow_pickle=False) as prediction:
            require(np.array_equal(prediction["calibration_positions"], cal_positions), "Source calibration position mismatch")
            require(np.array_equal(prediction["target_positions"], target_positions), "Target prediction position mismatch")
            cal_probability = probability_vector(prediction["calibration_p_vulnerable"])
            probability = probability_vector(prediction["target_p_vulnerable"])
            require(cal_probability.shape == cal_labels.shape and probability.shape == target_positions.shape,
                    "Frozen probability length mismatch")
        calibrator = fit_source_isotonic(cal_probability, cal_labels)
        eta = calibrator.predict(probability)
        source_fit_rows.append(dict(detector=detector, target_group=group, seed=seed,
                                    source_calibration_n=len(cal_labels), source_calibration_vulnerable=int(cal_labels.sum()),
                                    source_calibration_safe=int((cal_labels == 0).sum()),
                                    source_package_sha256=verified["receipts"]["verified_source_fold_sha256"][f"source_label_packages/{group}.npz"],
                                    prediction_npz_sha256=verified["receipts"]["verified_prediction_file_sha256"][f"{detector}/predictions/{group}/seed-{seed}.npz"],
                                    calibration_positions_sha256=array_sha256(cal_positions),
                                    calibration_labels_sha256=array_sha256(cal_labels),
                                    calibration_probability_sha256=array_sha256(cal_probability),
                                    target_positions_sha256=array_sha256(target_positions),
                                    target_probability_sha256=array_sha256(probability),
                                    x_knots=len(calibrator.X_thresholds_),
                                    x_thresholds_sha256=array_sha256(calibrator.X_thresholds_),
                                    y_thresholds_sha256=array_sha256(calibrator.y_thresholds_),
                                    calibrated_target_scores_sha256=array_sha256(eta), target_labels_in_fit=False,
                                    **{f"cost{cost}_ranking_sha256": array_sha256(np.argsort(np.minimum(cost * eta, 1.0 - eta), kind="stable"))
                                       for cost in COSTS}))
        with np.load(path, allow_pickle=False) as archive:
            require(np.array_equal(archive["target_positions"], target_positions), "Decision target position mismatch")
            require(np.array_equal(archive["target_p_vulnerable"], probability), "Prediction/decision probability mismatch")
            references = {name: archive[OPERATING_KEY + name] for name in REFERENCES}
            decisions = build_controls(probability, references, eta)
            require(np.array_equal(decisions["msp_matched_unweighted_mondrian"],
                                   archive[OPERATING_KEY + "msp_matched_unweighted_mondrian"]),
                    "Recomputed global MSP differs from frozen evaluator")
            # This is the first use of target labels; all selectors are now fixed.
            labels = archive["target_labels"]
            for name, sets in decisions.items():
                counts = action_counts(labels, sets)
                codes = action_codes(sets)
                non_argmax = ((codes == 0) & (probability > 0.5)) | ((codes == 1) & (probability <= 0.5))
                rows.append(dict(detector=detector, target_group=group, seed=seed, partition=splits[group], method=name,
                                 non_argmax_count=int(non_argmax.sum()), non_argmax_rate=float(non_argmax.mean()),
                                 action_sha256=array_sha256(codes), **counts))
                if name in REFERENCES:
                    original_method = {"estimated_weight_no_gate": "estimated_weight_no_gate_clip_20",
                                       "vultriage_full_gate": "vultriage_full_gate_clip_20"}.get(name, name)
                    original = primary.loc[(detector, group, seed, original_method)]
                    empty = sets.sum(axis=1) == 0
                    require(counts["singleton_count"] == int(original.singleton_count), "Original automation count mismatch")
                    require(counts["auto_fn"] + int(np.sum(empty & (labels == 1))) == int(original.vulnerable_miscoverage_count),
                            "Original vulnerable miscoverage mismatch")
                    require(counts["auto_fp"] + int(np.sum(empty & (labels == 0))) == int(original.safe_miscoverage_count),
                            "Original safe miscoverage mismatch")
            for baseline, reference, fixed_cost in controls_comparisons():
                candidate_codes, control_codes = action_codes(decisions[reference]), action_codes(decisions[baseline])
                ns, nv = int((candidate_codes == 0).sum()), int((candidate_codes == 1).sum())
                cs, cv = int((control_codes == 0).sum()), int((control_codes == 1).sum())
                require(ns + nv == cs + cv, "Control does not match automatic volume")
                if baseline.startswith("class_matched_"):
                    require((ns, nv) == (cs, cv), "Class-side control does not match allocation")
                selection_rows.append(dict(detector=detector, target_group=group, seed=seed,
                                           reference=reference, control=baseline, source_ranking_miss_cost=fixed_cost,
                                           reference_auto_safe=ns, reference_auto_vulnerable=nv,
                                           control_auto_safe=cs, control_auto_vulnerable=cv,
                                           matched_automatic_count=ns + nv, matched_review_count=len(labels) - ns - nv,
                                           exact_action_allocation=(ns, nv) == (cs, cv),
                                           safe_selected_overlap=int(((candidate_codes == 0) & (control_codes == 0)).sum()),
                                           vulnerable_selected_overlap=int(((candidate_codes == 1) & (control_codes == 1)).sum()),
                                           automatic_selected_overlap=int(((candidate_codes != 2) & (control_codes != 2)).sum()),
                                           actions_identical=bool(np.array_equal(candidate_codes, control_codes))))
        generated_path = output / relative
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(generated_path, target_positions=target_positions,
                            source_isotonic_eta=eta, source_isotonic_x=calibrator.X_thresholds_,
                            source_isotonic_y=calibrator.y_thresholds_,
                            **{name: sets.astype(np.uint8) for name, sets in decisions.items()})
        generated_hashes[relative] = sha256(generated_path)
    frame = pd.DataFrame(rows)
    keys = ["detector", "target_group", "seed", "method"]
    require(not frame.duplicated(keys).any(), "Duplicate derived metrics")
    require(frame.groupby(["detector", "method"]).target_group.nunique().eq(24).all(), "A project disappeared")
    require(frame.groupby(["detector", "target_group", "method"]).seed.nunique().eq(5).all(), "A technical seed disappeared")
    frame.to_csv(output / "action_metrics_seed.csv", index=False)
    numeric = [c for c in frame.columns if c not in [*keys, "partition", "action_sha256"]]
    projects = frame.groupby(["detector", "target_group", "partition", "method"], sort=True)[numeric].mean().reset_index()
    projects.to_csv(output / "action_metrics_project.csv", index=False)
    projects.groupby(["detector", "method"], sort=True)[numeric].median().reset_index().to_csv(output / "action_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "selection_invariants_seed.csv", index=False)
    pd.DataFrame(source_fit_rows).to_csv(output / "source_calibration_fits.csv", index=False)
    paired, cost_paired, cost_summaries = [], [], []
    for detector, subset in projects.groupby("detector", sort=True):
        for baseline, candidate, _ in controls_comparisons():
            for outcome in ("coverage", "auto_fn_rate", "auto_fp_rate", "review_load"):
                paired.append(dict(detector=detector, **paired_summary(subset, outcome, baseline, candidate)))
        for miss_cost in COSTS:
            cost_frame = subset.copy()
            cost_frame["cost"] = (miss_cost * cost_frame.auto_fn + FP_COST * cost_frame.auto_fp
                                   + REVIEW_COST * cost_frame.review_count) / cost_frame.n
            for method, by_method in cost_frame.groupby("method", sort=True):
                cost_summaries.append(dict(detector=detector, method=method, miss_cost=miss_cost,
                                           false_alert_cost=FP_COST, review_cost=REVIEW_COST,
                                           review_residual_error=0.0, projects=len(by_method),
                                           median_cost=float(by_method.cost.median()), mean_cost=float(by_method.cost.mean())))
            for baseline, candidate, fixed_cost in controls_comparisons():
                if fixed_cost is not None and fixed_cost != miss_cost:
                    continue
                cost_paired.append(dict(detector=detector, miss_cost=miss_cost, false_alert_cost=FP_COST,
                                        review_cost=REVIEW_COST, review_residual_error=0.0,
                                        **paired_summary(cost_frame, "cost", baseline, candidate)))
    pd.DataFrame(paired).to_csv(output / "matched_paired_comparisons.csv", index=False)
    pd.DataFrame(cost_paired).to_csv(output / "matched_cost_comparisons.csv", index=False)
    pd.DataFrame(cost_summaries).to_csv(output / "cost_summary.csv", index=False)
    receipts = dict(analysis_status="post-review retrospective analysis; target outcomes already known",
                    independent_validation=False, preregistered=False,
                    designed_date="2026-10-09", analysis_date="2026-10-10",
                    detector_refits=0, original_experiment_files_modified=False,
                    target_labels_used="retrospective action evaluation only; no selector or source calibrator input",
                    primary_budget={"alpha_vulnerable": 0.1, "alpha_safe": 0.2, "selected_upper_clip": 20},
                    cohorts={"detectors": ["hashing", "codebert"], "projects_per_detector": 24,
                             "technical_seed_addresses": manifest["seeds"], "seed_addresses_averaged_before_bootstrap": True},
                    controls={"global_msp": "stable descending max(p,1-p); original evaluator argmax tie at p=.5 is safe",
                              "class_side_matched": "stable ascending p; first k_safe and last k_vulnerable in one order; non-argmax labels permitted",
                              "source_isotonic": "increasing source-only isotonic regression, y in [0,1], out_of_bounds=clip",
                              "source_cost_matched": "label minimizing c_FN*eta vs c_FP*(1-eta); tied label is vulnerable; stable ascending min cost; exact reference automation count",
                              "source_calibration_transfer": "no claim that source calibration is valid on shifted target projects",
                              "free_volume_threshold_optimization": "deferred; not run in this bounded exact-workload audit"},
                    cost_scenarios={"miss": list(COSTS), "false_alert": FP_COST, "review": REVIEW_COST,
                                    "review_residual_error": 0.0, "interpretation": "fixed illustrative costs; no empirical review efficacy claim"},
                    bootstrap={"unit": "target project", "paired": True,
                               "method": "exact ordinary even-n empirical bootstrap median distribution; no random resampling",
                               "estimator": "median of candidate-minus-baseline project differences",
                               "quantile_definition": "smallest support point with CDF >= requested probability",
                               "interval": "pointwise percentile 95%; exploratory; not multiplicity-adjusted"},
                    script_sha256=sha256(Path(__file__)), contract_split_sha256=sha256(split_path),
                    runtime={"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                             "sklearn": sklearn.__version__, "platform": platform.platform()},
                    rows={"seed_metrics": len(frame), "project_metrics": len(projects), "paired_outcomes": len(paired),
                          "cost_comparisons": len(cost_paired), "source_fits": len(source_fit_rows)},
                    generated_decision_archive_sha256=generated_hashes, **verified["receipts"])
    receipts["output_table_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.csv"))}
    (output / "analysis_manifest.json").write_text(json.dumps(receipts, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, default=Path("public_results/extension-v2"))
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/preregistered_extension_v2.json"))
    parser.add_argument("--runner", type=Path, default=Path("src/run_extension_predict.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs/neurocomputing-controls-v2"))
    args = parser.parse_args()
    predictions_root = args.predictions_root or args.evaluation.parent / "predictions"
    inputs = args.inputs or args.evaluation.parent / "source-v2"
    receipts = audit(args.evaluation, args.public_root, predictions_root, inputs,
                     args.config, args.runner, args.output)
    print(json.dumps({"verified_decisions": len(receipts["verified_decision_archive_sha256"]),
                      "verified_prediction_files": len(receipts["verified_prediction_file_sha256"]),
                      "verified_source_fold_packages": len(receipts["verified_source_fold_sha256"]),
                      "rows": receipts["rows"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
