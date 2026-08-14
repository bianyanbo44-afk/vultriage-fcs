"""Project-level inference for the sealed extension-v2 evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from vultriage.data import load_config, sha256


OUTCOMES = (
    "max_relative_violation",
    "vulnerable_violation",
    "safe_violation",
    "singleton_coverage",
    "review_load",
)
LOWER_IS_BETTER = {
    "max_relative_violation": True,
    "vulnerable_violation": True,
    "safe_violation": True,
    "singleton_coverage": False,
    "review_load": True,
}
COMPARATORS = (
    "estimated_weight_no_gate_clip_20",
    "vultriage_ess_only_clip_20",
    "vultriage_infinity_only_clip_20",
    "vultriage_full_gate_clip_20",
    "prom_derived_union",
)
REQUIRED_METHODS = frozenset(("unweighted_mondrian", *COMPARATORS))
DETECTOR_METRICS = (
    "pr_auc",
    "brier",
    "ece_equal_mass_15",
    "aurc",
    "fnr_at_fpr_0_005",
    "error_detection_auroc",
    "precision",
    "recall",
    "f1",
    "f2",
    "mcc",
)


def effective_config(path: Path) -> dict[str, Any]:
    extension = load_config(path)
    inherit = Path(extension["detectors"]["hashing_sgd"]["inherit"])
    if not inherit.is_absolute():
        inherit = Path.cwd() / inherit
    config = dict(load_config(inherit))
    for key in (
        "protocol_version",
        "seeds",
        "risk_budgets",
        "support_gate",
        "calibration_size_sensitivity",
        "detectors",
    ):
        if key in extension:
            config[key] = extension[key]
    return config


def holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[int(index)]))
        result[int(index)] = running
    return result.tolist()


def paired_hodges_lehmann(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    walsh = [(values[i] + values[j]) / 2 for i in range(len(values)) for j in range(i, len(values))]
    return float(np.median(walsh)) if walsh else float("nan")


def bootstrap_interval(values: np.ndarray, replicates: int, seed: int, level: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    statistics = np.median(values[indices], axis=1)
    tail = (1.0 - level) / 2.0
    return tuple(float(x) for x in np.quantile(statistics, [tail, 1.0 - tail]))


def gate_pass_fail_bootstrap_interval(
    joined: pd.DataFrame, replicates: int, seed: int, level: float
) -> tuple[float, float]:
    """Bootstrap the project-level pass-minus-fail median difference.

    Gate status is fixed by the frozen rule; resampling is performed within
    the observed pass and fail project strata so that each replicate retains
    the estimand used in the reported descriptive comparison.
    """
    passed = joined.loc[joined["pass_all"], "max_relative_violation"].to_numpy(float)
    failed = joined.loc[~joined["pass_all"], "max_relative_violation"].to_numpy(float)
    if len(passed) == 0 or len(failed) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pass_indices = rng.integers(0, len(passed), size=(replicates, len(passed)))
    fail_indices = rng.integers(0, len(failed), size=(replicates, len(failed)))
    statistics = np.median(passed[pass_indices], axis=1) - np.median(failed[fail_indices], axis=1)
    tail = (1.0 - level) / 2.0
    return tuple(float(x) for x in np.quantile(statistics, [tail, 1.0 - tail]))


def load_and_check(metrics: Path, manifest: Path) -> pd.DataFrame:
    seal = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256(metrics) != seal["metrics_sha256"]:
        raise RuntimeError("metrics hash differs from evaluation manifest")
    frame = pd.read_csv(metrics)
    key = ["detector", "target_group", "seed", "alpha_vulnerable", "alpha_safe", "method"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate fold-level metric primary key")
    return frame


def validate_frame(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    expected_seeds = {int(seed) for seed in config["seeds"]}
    expected_vulnerable = {float(value) for value in config["risk_budgets"]["vulnerable"]}
    expected_safe = {float(value) for value in config["risk_budgets"]["safe"]}
    if set(frame["detector"].unique()) != {"hashing", "codebert"}:
        raise RuntimeError("evaluation does not contain exactly the two frozen detectors")
    if set(frame["alpha_vulnerable"].unique()) != expected_vulnerable or set(frame["alpha_safe"].unique()) != expected_safe:
        raise RuntimeError("evaluation risk grid differs from the frozen configuration")
    if set(frame["method"].unique()) < REQUIRED_METHODS:
        missing = sorted(REQUIRED_METHODS - set(frame["method"].unique()))
        raise RuntimeError(f"evaluation is missing required methods: {missing}")
    groups = set(frame["target_group"].unique())
    if len(groups) != 24:
        raise RuntimeError(f"expected 24 frozen target projects, observed {len(groups)}")
    keys = ["detector", "target_group", "alpha_vulnerable", "alpha_safe", "method"]
    for key, subset in frame.groupby(keys, sort=False):
        observed = {int(seed) for seed in subset["seed"]}
        if observed != expected_seeds or len(subset) != len(expected_seeds):
            raise RuntimeError(f"seed set mismatch for cell {key}: {sorted(observed)}")
    required_keys = ["detector", "target_group", "alpha_vulnerable", "alpha_safe"]
    for key, subset in frame.groupby(required_keys, sort=False):
        methods = set(subset["method"])
        if not REQUIRED_METHODS.issubset(methods):
            raise RuntimeError(f"method set mismatch for cell {key}")


def project_means(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    keys = ["detector", "target_group", "alpha_vulnerable", "alpha_safe", "method"]
    numeric = [name for name in OUTCOMES if name in frame.columns]
    validate_frame(frame, config)
    return frame.groupby(keys, as_index=False)[numeric].mean()


def detector_project_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one seed-averaged detector-performance row per target project."""
    selected = frame[
        (frame["method"] == "forced_argmax")
        & (frame["alpha_vulnerable"] == 0.1)
        & (frame["alpha_safe"] == 0.2)
    ]
    metrics = [name for name in DETECTOR_METRICS if name in selected.columns]
    if len(metrics) != len(DETECTOR_METRICS):
        missing = sorted(set(DETECTOR_METRICS) - set(metrics))
        raise RuntimeError(f"evaluation is missing detector metrics: {missing}")
    result = selected.groupby(["detector", "target_group"], as_index=False)[metrics].mean()
    if len(result) != 48:
        raise RuntimeError(f"expected 48 detector-project rows, observed {len(result)}")
    return result


def paired_comparisons(project: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in sorted(project["detector"].unique()):
        for av in sorted(project["alpha_vulnerable"].unique()):
            for ass in sorted(project["alpha_safe"].unique()):
                subset = project[
                    (project["detector"] == detector)
                    & (project["alpha_vulnerable"] == av)
                    & (project["alpha_safe"] == ass)
                ]
                baseline = subset[subset["method"] == "unweighted_mondrian"].set_index("target_group")
                for method in COMPARATORS:
                    candidate = subset[subset["method"] == method].set_index("target_group")
                    if set(baseline.index) != set(candidate.index):
                        raise RuntimeError(
                            f"baseline/candidate project mismatch for {detector}/{av}/{ass}/{method}"
                        )
                    common = sorted(baseline.index)
                    for outcome in OUTCOMES:
                        difference = candidate.loc[common, outcome].to_numpy(float) - baseline.loc[common, outcome].to_numpy(float)
                        nonzero = difference[difference != 0]
                        if len(nonzero) >= 10:
                            wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue)
                        else:
                            wilcoxon_p = float("nan")
                        lower = LOWER_IS_BETTER[outcome]
                        wins = int(np.sum(difference < 0 if lower else difference > 0))
                        losses = int(np.sum(difference > 0 if lower else difference < 0))
                        ties = int(np.sum(difference == 0))
                        sign_p = float(binomtest(wins, wins + losses, p=0.5).pvalue) if wins + losses else float("nan")
                        low, high = bootstrap_interval(
                            difference,
                            int(config["bootstrap"]["replicates"]),
                            20260814 + len(rows),
                            float(config["bootstrap"]["confidence_level"]),
                        )
                        rows.append({
                            "detector": detector,
                            "alpha_vulnerable": av,
                            "alpha_safe": ass,
                            "baseline": "unweighted_mondrian",
                            "method": method,
                            "outcome": outcome,
                            "projects": len(common),
                            "nonzero_projects": len(nonzero),
                            "median_paired_difference": float(np.median(difference)),
                            "hodges_lehmann_paired_shift": paired_hodges_lehmann(difference),
                            "bootstrap_median_ci_lower": low,
                            "bootstrap_median_ci_upper": high,
                            "wins": wins,
                            "ties": ties,
                            "losses": losses,
                            "exact_sign_p": sign_p,
                            "wilcoxon_exact_p": wilcoxon_p,
                        })
    result = pd.DataFrame(rows)
    result["holm_adjusted_wilcoxon_p"] = np.nan
    if len(result):
        for _, indices in result.groupby(["detector", "alpha_vulnerable", "alpha_safe", "outcome"]).groups.items():
            valid = [int(index) for index in indices if np.isfinite(result.at[index, "wilcoxon_exact_p"])]
            adjusted = holm_adjust([float(result.at[index, "wilcoxon_exact_p"]) for index in valid])
            for index, value in zip(valid, adjusted):
                result.at[index, "holm_adjusted_wilcoxon_p"] = value
    return result


def gate_discrimination(frame: pd.DataFrame, project: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selected = frame[
        (frame["method"] == "vultriage_full_gate_clip_20")
        & (frame["alpha_vulnerable"] == 0.1)
        & (frame["alpha_safe"] == 0.2)
    ].copy()
    raw = project[
        (project["method"] == "estimated_weight_no_gate_clip_20")
        & (project["alpha_vulnerable"] == 0.1)
        & (project["alpha_safe"] == 0.2)
    ][["detector", "target_group", "max_relative_violation"]]
    selected["support_json"] = selected["support"].map(json.loads)
    selected["gate_pass_seed"] = selected["support_json"].map(lambda item: bool(item["passed"]))
    selected["gate_probability"] = selected["support_json"].map(lambda item: float(item["gate_probability"]))
    rows: list[dict[str, Any]] = []
    for detector in sorted(selected["detector"].unique()):
        seed_frame = selected[selected["detector"] == detector]
        grouped = seed_frame.groupby("target_group").agg(
            pass_all=("gate_pass_seed", "all"),
            pass_rate=("gate_pass_seed", "mean"),
            gate_probability=("gate_probability", "mean"),
        ).reset_index()
        joined = grouped.merge(
            raw[raw["detector"] == detector].drop(columns="detector"),
            on="target_group",
            how="left",
            validate="one_to_one",
        )
        # A project is considered passed only if every technical seed passed.
        y = (joined["max_relative_violation"] > 0.5).astype(int).to_numpy()
        score = joined["gate_probability"].to_numpy(float)
        auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
        auprc = float(average_precision_score(y, score)) if y.sum() else float("nan")
        pass_values = joined.loc[joined["pass_all"], "max_relative_violation"].to_numpy(float)
        fail_values = joined.loc[~joined["pass_all"], "max_relative_violation"].to_numpy(float)
        median_difference = float(np.median(pass_values) - np.median(fail_values)) if len(pass_values) and len(fail_values) else float("nan")
        ci_lower, ci_upper = gate_pass_fail_bootstrap_interval(
            joined,
            int(config["bootstrap"]["replicates"]),
            20260814 + (0 if detector == "codebert" else 1),
            float(config["bootstrap"]["confidence_level"]),
        )
        rows.append({
            "detector": detector,
            "projects": len(joined),
            "passed_all_seed": int(joined["pass_all"].sum()),
            "pass_rate_mean": float(joined["pass_all"].mean()),
            "gate_auroc_severe_violation": auroc,
            "gate_auprc_severe_violation": auprc,
            "median_raw_violation_pass_minus_fail": median_difference,
            "median_raw_violation_pass_minus_fail_bootstrap_ci_lower": ci_lower,
            "median_raw_violation_pass_minus_fail_bootstrap_ci_upper": ci_upper,
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "project_pass_fail_definition": "all five technical seeds pass",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    frame = load_and_check(args.metrics, args.evaluation_manifest)
    config = effective_config(args.config)
    project = project_means(frame, config)
    detector_performance = detector_project_performance(frame)
    comparisons = paired_comparisons(project, config)
    gates = gate_discrimination(frame, project, config)
    project_path = args.output / "project_seed_averages.csv"
    comparison_path = args.output / "paired_project_comparisons.csv"
    gate_path = args.output / "gate_discrimination.csv"
    detector_path = args.output / "detector_project_performance.csv"
    project.to_csv(project_path, index=False)
    detector_performance.to_csv(detector_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    gates.to_csv(gate_path, index=False)
    summary = {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_sha256": sha256(args.metrics),
        "independent_unit": "target_project_group",
        "seed_handling": "five seeds averaged within project before inference",
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "project_rows": len(project),
        "detector_performance_rows": len(detector_performance),
        "comparison_rows": len(comparisons),
        "gate_rows": len(gates),
        "project_seed_averages_sha256": sha256(project_path),
        "detector_project_performance_sha256": sha256(detector_path),
        "paired_project_comparisons_sha256": sha256(comparison_path),
        "gate_discrimination_sha256": sha256(gate_path),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
