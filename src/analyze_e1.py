"""Preregistered project-level statistical analysis for E1."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from vultriage.data import sha256


PRIMARY_BASELINE = "unweighted_mondrian"
COMPARATORS = (
    "vultriage_clip_20",
    "estimated_weight_no_support_clip_20",
    "msp_matched_vultriage",
    "prom_compatible_lac_matched_vultriage",
)
OUTCOMES = (
    "vulnerable_violation",
    "safe_violation",
    "absolute_target_difference_sum",
    "singleton_coverage",
    "review_load",
)
LOWER_IS_BETTER = {
    "vulnerable_violation": True,
    "safe_violation": True,
    "absolute_target_difference_sum": True,
    "singleton_coverage": False,
    "review_load": True,
}


def paired_hodges_lehmann(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    walsh = [
        (differences[i] + differences[j]) / 2.0
        for i in range(len(differences))
        for j in range(i, len(differences))
    ]
    return float(np.median(walsh))


def project_bootstrap_interval(
    differences: np.ndarray, replicates: int, seed: int, level: float
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    statistics = np.median(differences[indices], axis=1)
    tail = (1.0 - level) / 2.0
    return tuple(float(value) for value in np.quantile(statistics, [tail, 1.0 - tail]))


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def support_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        (frame["track"] == "project_disjoint")
        & (frame["method"] == "vultriage_clip_20")
    ].copy()
    selected["supported"] = selected["support"].map(
        lambda value: bool(json.loads(value)["supported"])
    )
    rows = []
    for (alpha_vulnerable, alpha_safe), group in selected.groupby(
        ["alpha_vulnerable", "alpha_safe"]
    ):
        by_project = group.groupby("target_group").agg(
            supported=("supported", "all"),
            singleton_coverage=("singleton_coverage", "mean"),
            vulnerable_singleton_rate=("vulnerable_singleton_rate", "mean"),
            safe_singleton_rate=("safe_singleton_rate", "mean"),
        )
        rows.append(
            {
                "alpha_vulnerable": alpha_vulnerable,
                "alpha_safe": alpha_safe,
                "supported_projects": int(by_project["supported"].sum()),
                "total_projects": int(len(by_project)),
                "support_rate": float(by_project["supported"].mean()),
                "median_singleton_coverage_all_projects": float(
                    by_project["singleton_coverage"].median()
                ),
                "median_singleton_coverage_supported_projects": (
                    float(
                        by_project.loc[
                            by_project["supported"], "singleton_coverage"
                        ].median()
                    )
                    if by_project["supported"].any()
                    else float("nan")
                ),
                "supported_with_both_class_singletons": int(
                    (
                        by_project["supported"]
                        & (by_project["vulnerable_singleton_rate"] > 0)
                        & (by_project["safe_singleton_rate"] > 0)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha_vulnerable", "alpha_safe"])


def paired_comparisons(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    project_seed_mean = (
        frame[frame["track"] == "project_disjoint"]
        .groupby(
            [
                "target_group",
                "alpha_vulnerable",
                "alpha_safe",
                "method",
            ],
            as_index=False,
        )[list(OUTCOMES)]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for alpha_vulnerable in sorted(project_seed_mean["alpha_vulnerable"].unique()):
        for alpha_safe in sorted(project_seed_mean["alpha_safe"].unique()):
            operating = project_seed_mean[
                (project_seed_mean["alpha_vulnerable"] == alpha_vulnerable)
                & (project_seed_mean["alpha_safe"] == alpha_safe)
            ]
            baseline = operating[operating["method"] == PRIMARY_BASELINE].set_index(
                "target_group"
            )
            for method in COMPARATORS:
                candidate = operating[operating["method"] == method].set_index(
                    "target_group"
                )
                common = sorted(set(baseline.index) & set(candidate.index))
                for outcome in OUTCOMES:
                    difference = (
                        candidate.loc[common, outcome].to_numpy(float)
                        - baseline.loc[common, outcome].to_numpy(float)
                    )
                    nonzero = difference[difference != 0]
                    p_value = float("nan")
                    if len(nonzero) >= 10:
                        p_value = float(
                            wilcoxon(
                                difference,
                                alternative="two-sided",
                                zero_method="wilcox",
                                method="exact",
                            ).pvalue
                        )
                    favorable = difference < 0 if LOWER_IS_BETTER[outcome] else difference > 0
                    unfavorable = difference > 0 if LOWER_IS_BETTER[outcome] else difference < 0
                    wins = int(np.sum(favorable))
                    ties = int(np.sum(difference == 0))
                    losses = int(np.sum(unfavorable))
                    sign_p = (
                        float(binomtest(wins, wins + losses, p=0.5).pvalue)
                        if wins + losses
                        else float("nan")
                    )
                    lower, upper = project_bootstrap_interval(
                        difference,
                        int(config["bootstrap"]["replicates"]),
                        20260813,
                        float(config["bootstrap"]["confidence_level"]),
                    )
                    rows.append(
                        {
                            "alpha_vulnerable": alpha_vulnerable,
                            "alpha_safe": alpha_safe,
                            "baseline": PRIMARY_BASELINE,
                            "method": method,
                            "outcome": outcome,
                            "projects": len(common),
                            "nonzero_projects": len(nonzero),
                            "median_paired_difference": float(np.median(difference)),
                            "hodges_lehmann_paired_shift": paired_hodges_lehmann(
                                difference
                            ),
                            "bootstrap_median_ci_lower": lower,
                            "bootstrap_median_ci_upper": upper,
                            "favorable_direction": (
                                "lower" if LOWER_IS_BETTER[outcome] else "higher"
                            ),
                            "wins": wins,
                            "ties": ties,
                            "losses": losses,
                            "common_language_win_rate_excluding_ties": (
                                wins / (wins + losses) if wins + losses else float("nan")
                            ),
                            "exact_sign_p": sign_p,
                            "wilcoxon_exact_p": p_value,
                        }
                    )
    result = pd.DataFrame(rows)
    result["holm_adjusted_wilcoxon_p"] = np.nan
    # Frozen family: all four methods against the primary baseline within one
    # outcome x risk-budget setting.
    for _, indices in result.groupby(
        ["alpha_vulnerable", "alpha_safe", "outcome"]
    ).groups.items():
        valid = [index for index in indices if np.isfinite(result.at[index, "wilcoxon_exact_p"])]
        adjusted = holm_adjust([result.at[index, "wilcoxon_exact_p"] for index in valid])
        for index, value in zip(valid, adjusted):
            result.at[index, "holm_adjusted_wilcoxon_p"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    if sha256(args.metrics) != manifest["metrics_sha256"]:
        raise RuntimeError("Metric hash differs from evaluation manifest")
    frame = pd.read_csv(args.metrics)
    key = [
        "track",
        "target_group",
        "seed",
        "alpha_vulnerable",
        "alpha_safe",
        "method",
    ]
    if frame.duplicated(key).any():
        raise RuntimeError("Duplicate metric primary keys")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    support = support_summary(frame)
    comparisons = paired_comparisons(frame, config)
    support_path = args.output / "support_summary.csv"
    comparison_path = args.output / "paired_project_comparisons.csv"
    support.to_csv(support_path, index=False)
    comparisons.to_csv(comparison_path, index=False)

    selected = comparisons[
        (comparisons["alpha_vulnerable"] == 0.05)
        & (comparisons["alpha_safe"] == 0.10)
        & comparisons["method"].isin(
            ["vultriage_clip_20", "estimated_weight_no_support_clip_20"]
        )
    ]
    report = {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_sha256": sha256(args.metrics),
        "independent_unit": "target_project_group",
        "technical_repetitions": "five seeds averaged within project before inference",
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "comparison_rows": len(comparisons),
        "selected_operating_point": selected.to_dict(orient="records"),
        "support_summary_sha256": sha256(support_path),
        "paired_comparisons_sha256": sha256(comparison_path),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "selected_operating_point"}, sort_keys=True))


if __name__ == "__main__":
    main()
