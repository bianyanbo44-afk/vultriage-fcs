"""Build a label-free, machine-readable evidence package from sealed v2 outputs.

The builder only consumes evaluator/analysis summaries and sealed metadata.  It
never opens source code, the target label vault, SQLite indexes, embeddings, or
per-function prediction archives.  Every output is hashed in a manifest so the
paper can cite one immutable evidence surface.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vultriage.data import load_config, sha256


PRIMARY_AV = 0.10
PRIMARY_AS = 0.20
PRIMARY_METHODS = {
    "unweighted_mondrian",
    "pooled_split_conformal",
    "forced_argmax",
    "msp_matched_unweighted_mondrian",
    "entropy_matched_unweighted_mondrian",
    "temperature_msp_matched_unweighted_mondrian",
    "prom_derived_lac",
    "prom_derived_topk",
    "prom_derived_aps",
    "prom_derived_raps",
    "prom_derived_union",
    "estimated_weight_no_gate_clip_10",
    "vultriage_ess_only_clip_10",
    "vultriage_infinity_only_clip_10",
    "vultriage_full_gate_clip_10",
    "estimated_weight_no_gate_clip_20",
    "vultriage_ess_only_clip_20",
    "vultriage_infinity_only_clip_20",
    "vultriage_full_gate_clip_20",
    "estimated_weight_no_gate_clip_50",
    "vultriage_ess_only_clip_50",
    "vultriage_infinity_only_clip_50",
    "vultriage_full_gate_clip_50",
    "msp_matched_vultriage_full_gate",
    "temperature_msp_matched_vultriage_full_gate",
}
SUMMARY_COLUMNS = [
    "vulnerable_miscoverage",
    "safe_miscoverage",
    "vulnerable_violation",
    "safe_violation",
    "max_relative_violation",
    "singleton_coverage",
    "review_load",
]
PERFORMANCE_COLUMNS = ["pr_auc", "f1", "mcc", "brier", "error_detection_auroc"]


def parse_support(value: object) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return parsed if isinstance(parsed, dict) else None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bootstrap_median(values: np.ndarray, seed: int, replicates: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    medians = np.median(values[indices], axis=1)
    return tuple(float(value) for value in np.quantile(medians, [0.025, 0.975]))


def artifact(path: Path, repo_root: Path) -> dict[str, Any]:
    path = path.resolve()
    repo_root = repo_root.resolve()
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def project_primary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["alpha_vulnerable"].astype(float) == PRIMARY_AV)
        & (metrics["alpha_safe"].astype(float) == PRIMARY_AS)
    ].copy()
    grouped = (
        selected.groupby(["detector", "target_group", "method"], as_index=False)[
            SUMMARY_COLUMNS
        ]
        .mean()
    )
    return grouped


def support_status(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["alpha_vulnerable"].astype(float) == PRIMARY_AV)
        & (metrics["alpha_safe"].astype(float) == PRIMARY_AS)
    ].copy()
    selected["support_record"] = selected["support"].map(parse_support)
    selected["seed_pass"] = selected["support_record"].map(
        lambda item: bool(item["passed"]) if item is not None and "passed" in item else np.nan
    )
    rows: list[dict[str, Any]] = []
    for (detector, target_group, method), subset in selected.groupby(
        ["detector", "target_group", "method"], sort=True
    ):
        passed = subset["seed_pass"].dropna().astype(bool).to_numpy()
        if len(passed) == 0:
            continue
        rows.append(
            {
                "detector": detector,
                "target_group": target_group,
                "method": method,
                "support_recorded": True,
                "support_pass_all_seeds": bool(passed.all()),
                "support_pass_rate": float(passed.mean()),
                "seed_rows": int(len(passed)),
            }
        )
    return pd.DataFrame(rows)


def primary_method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    project = project_primary_metrics(metrics)
    status = support_status(metrics)
    rows: list[dict[str, Any]] = []
    for detector in sorted(project["detector"].unique()):
        for method in sorted(PRIMARY_METHODS):
            subset = project[(project["detector"] == detector) & (project["method"] == method)].copy()
            if len(subset) != 24:
                raise RuntimeError(f"primary method cell is not 24 projects: {detector}/{method}")
            row: dict[str, Any] = {
                "detector": detector,
                "method": method,
                "projects": int(len(subset)),
                "overall_attainment_projects": int(
                    np.sum(
                        (subset["vulnerable_violation"].to_numpy(float) == 0.0)
                        & (subset["safe_violation"].to_numpy(float) == 0.0)
                    )
                ),
            }
            for column in SUMMARY_COLUMNS:
                row[f"overall_median_{column}"] = float(subset[column].median())
            gate = status[(status["detector"] == detector) & (status["method"] == method)]
            row["support_recorded"] = bool(len(gate) == 24)
            if len(gate) == 24:
                supported_groups = set(gate.loc[gate["support_pass_all_seeds"], "target_group"])
                supported = subset[subset["target_group"].isin(supported_groups)]
                row["support_pass_projects"] = int(len(supported))
                row["supported_attainment_projects"] = int(
                    np.sum(
                        (supported["vulnerable_violation"].to_numpy(float) == 0.0)
                        & (supported["safe_violation"].to_numpy(float) == 0.0)
                    )
                )
                for column in SUMMARY_COLUMNS:
                    row[f"supported_median_{column}"] = (
                        float(supported[column].median()) if len(supported) else np.nan
                    )
            else:
                row["support_pass_projects"] = np.nan
                row["supported_attainment_projects"] = np.nan
                for column in SUMMARY_COLUMNS:
                    row[f"supported_median_{column}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def detector_performance_summary(performance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in sorted(performance["detector"].unique()):
        subset = performance[performance["detector"] == detector]
        for index, metric in enumerate(PERFORMANCE_COLUMNS):
            values = subset[metric].astype(float).to_numpy()
            low, high = bootstrap_median(values, 20260814 + index + (100 if detector == "codebert" else 0))
            rows.append(
                {
                    "detector": detector,
                    "metric": metric,
                    "projects": int(len(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "bootstrap_ci_lower": low,
                    "bootstrap_ci_upper": high,
                    "bootstrap_replicates": 10000,
                    "unit": "target project after five-seed averaging",
                }
            )
    return pd.DataFrame(rows)


def detector_paired_summary(performance: pd.DataFrame) -> pd.DataFrame:
    pivot = performance.pivot(index="target_group", columns="detector", values=PERFORMANCE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for index, metric in enumerate(PERFORMANCE_COLUMNS):
        values = (pivot[(metric, "codebert")] - pivot[(metric, "hashing")]).to_numpy(float)
        low, high = bootstrap_median(values, 20261814 + index)
        rows.append(
            {
                "metric": metric,
                "projects": int(len(values)),
                "codebert_minus_hashing_median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "bootstrap_ci_lower": low,
                "bootstrap_ci_upper": high,
                "codebert_wins": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "hashing_wins": int(np.sum(values < 0)),
                "bootstrap_replicates": 10000,
            }
        )
    return pd.DataFrame(rows)


def gate_summary(
    metrics: pd.DataFrame,
    project: pd.DataFrame,
    gate_disc: pd.DataFrame,
    gate_seal: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = support_status(metrics)
    raw = project[
        (project["detector"].isin(["hashing", "codebert"]))
        & (project["method"] == "estimated_weight_no_gate_clip_20")
    ]
    raw = raw[
        (raw["alpha_vulnerable"].astype(float) == PRIMARY_AV)
        & (raw["alpha_safe"].astype(float) == PRIMARY_AS)
    ]
    gate = status[status["method"] == "vultriage_full_gate_clip_20"]
    joined = raw.merge(
        gate[["detector", "target_group", "support_pass_all_seeds"]],
        on=["detector", "target_group"],
        how="left",
        validate="one_to_one",
    )
    joined["severe_raw_violation"] = joined["max_relative_violation"] > 0.5
    rows = [
        {
            "domain": "PrimeVul development",
            "detector": "frozen gate",
            "projects": int(gate_seal["projects"]),
            "units": int(gate_seal["development_rows"]),
            "severe_prevalence": float(gate_seal["severe_rows"] / gate_seal["development_rows"]),
            "auroc": float(gate_seal["crossfit_auroc"]),
            "auprc": float(gate_seal["crossfit_auprc"]),
            "support_pass_projects": np.nan,
            "pass_minus_fail_median": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    ]
    for detector in ("hashing", "codebert"):
        record = gate_disc.loc[gate_disc["detector"] == detector].iloc[0]
        sub = joined[joined["detector"] == detector]
        passed = sub.loc[sub["support_pass_all_seeds"], "max_relative_violation"].to_numpy(float)
        failed = sub.loc[~sub["support_pass_all_seeds"], "max_relative_violation"].to_numpy(float)
        rows.append(
            {
                "domain": "DiverseVul external",
                "detector": detector,
                "projects": int(record["projects"]),
                "units": int(record["projects"]),
                "severe_prevalence": float(sub["severe_raw_violation"].mean()),
                "auroc": float(record["gate_auroc_severe_violation"]),
                "auprc": float(record["gate_auprc_severe_violation"]),
                "support_pass_projects": int(record["passed_all_seed"]),
                "pass_minus_fail_median": float(record["median_raw_violation_pass_minus_fail"]),
                "ci_lower": float(record["median_raw_violation_pass_minus_fail_bootstrap_ci_lower"]),
                "ci_upper": float(record["median_raw_violation_pass_minus_fail_bootstrap_ci_upper"]),
            }
        )
        if len(passed) == 0 or len(failed) == 0:
            raise RuntimeError(f"gate pass/fail strata are incomplete for {detector}")
    return pd.DataFrame(rows), joined


def calibration_summary(project_path: Path, sensitivity_path: Path) -> pd.DataFrame:
    project = pd.read_csv(project_path)
    sensitivity = pd.read_csv(sensitivity_path)
    methods = ["unweighted_mondrian", "estimated_weight_no_gate_clip_20", "vultriage_full_gate_clip_20"]
    project = project[project["method"].isin(methods)].copy()
    rows: list[dict[str, Any]] = []
    for (detector, fraction, method), subset in project.groupby(
        ["detector", "fraction", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "detector": detector,
            "fraction": float(fraction),
            "method": method,
            "projects": int(subset["target_group"].nunique()),
            "median_max_relative_violation": float(subset["max_relative_violation"].median()),
            "median_singleton_coverage": float(subset["singleton_coverage"].median()),
            "both_budget_attainment_projects": int(
                np.sum(
                    (subset["vulnerable_violation"].to_numpy(float) == 0.0)
                    & (subset["safe_violation"].to_numpy(float) == 0.0)
                )
            ),
            "cell_pass_rate": np.nan,
            "all_cells_pass_projects": np.nan,
            "any_cell_pass_projects": np.nan,
        }
        if method == "vultriage_full_gate_clip_20":
            cell = sensitivity[
                (sensitivity["detector"] == detector)
                & (sensitivity["fraction"].astype(float) == float(fraction))
                & (sensitivity["method"] == method)
            ].copy()
            cell["passed"] = cell["support"].map(parse_support).map(
                lambda item: bool(item["passed"]) if item is not None else False
            )
            row["cell_pass_rate"] = float(cell["passed"].mean())
            project_pass = cell.groupby("target_group")["passed"].agg(["all", "any"])
            row["all_cells_pass_projects"] = int(project_pass["all"].sum())
            row["any_cell_pass_projects"] = int(project_pass["any"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def near_duplicate_summary(
    audit_path: Path, primary_summary_path: Path, primary_project: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    clean = pd.DataFrame(json.loads(primary_summary_path.read_text(encoding="utf-8")))
    methods = ["unweighted_mondrian", "estimated_weight_no_gate_clip_20", "vultriage_full_gate_clip_20"]
    rows: list[dict[str, Any]] = []
    for detector in ("hashing", "codebert"):
        main = primary_project[
            (primary_project["detector"] == detector)
            & (primary_project["method"].isin(methods))
        ]
        for method in methods:
            subset = main[main["method"] == method]
            rows.append(
                {
                    "cohort": "main_exact_deduplicated",
                    "detector": detector,
                    "method": method,
                    "projects": int(len(subset)),
                    "median_max_relative_violation": float(subset["max_relative_violation"].median()),
                    "median_singleton_coverage": float(subset["singleton_coverage"].median()),
                    "both_budget_attainment_projects": int(
                        np.sum(
                            (subset["vulnerable_violation"].to_numpy(float) == 0.0)
                            & (subset["safe_violation"].to_numpy(float) == 0.0)
                        )
                    ),
                }
            )
            item = clean[(clean["detector"] == detector) & (clean["method"] == method)].iloc[0]
            rows.append(
                {
                    "cohort": "near_duplicate_clean_sensitivity",
                    "detector": detector,
                    "method": method,
                    "projects": int(item["projects"]),
                    "median_max_relative_violation": float(item["median_max_relative_violation"]),
                    "median_singleton_coverage": float(item["median_singleton_coverage"]),
                    "both_budget_attainment_projects": int(item["both_budget_attainment_projects"]),
                }
            )
    audit_summary = {
        "candidate_pairs": int(audit["counts"]["candidate_pairs"]),
        "verified_candidates": int(audit["counts"]["verified_candidates"]),
        "flagged_pairs": int(audit["counts"]["flagged_pairs"]),
        "excluded_target_rows": int(audit["counts"]["excluded_target_rows"]),
        "retained_target_rows": int(audit["counts"]["retained_target_rows"]),
        "affected_projects": int(len(audit["counts"]["affected_projects"])),
        "threshold": float(audit["algorithm"]["flag_threshold"]),
        "candidate_generation_approximate": bool(audit["algorithm"]["candidate_generation_only_is_approximate"]),
        "reported_pair_verification_exact": bool(audit["algorithm"]["reported_pair_verification_is_exact"]),
    }
    return pd.DataFrame(rows), audit_summary


def output_record(path: Path, repo_root: Path, rows: int, columns: list[str]) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != rows:
        raise RuntimeError(f"unexpected output row count for {path.name}: {len(frame)} != {rows}")
    if list(frame.columns) != columns:
        raise RuntimeError(f"unexpected output schema for {path.name}")
    return {**artifact(path, repo_root), "rows": rows, "columns": columns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"evidence output already exists: {args.output}")
    repo_root = Path.cwd()
    root = args.root
    config = load_config(args.config)
    config_hash = sha256(args.config)
    evaluation_dir = root / "evaluation-v2"
    analysis_dir = root / "analysis-v2"
    calibration_dir = root / "calibration-size-v2"
    near_dir = root / "near-duplicate-sensitivity-v2"
    efficiency_dir = root / "efficiency-v2"
    output = args.output
    output.mkdir(parents=True)

    metrics_path = evaluation_dir / "fold_seed_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    project_path = analysis_dir / "project_seed_averages.csv"
    project = pd.read_csv(project_path)
    performance = pd.read_csv(analysis_dir / "detector_project_performance.csv")
    gate_disc = pd.read_csv(analysis_dir / "gate_discrimination.csv")
    gate_seal = json.loads((root / "gate-v1" / "gate_seal.json").read_text(encoding="utf-8"))

    required_seeds = [int(value) for value in config["detectors"]["hashing_sgd"]["seeds"]]
    if set(metrics["method"]) != PRIMARY_METHODS or len(metrics) != 54000:
        raise RuntimeError("sealed evaluation dimensions differ from the frozen v2 protocol")
    if set(metrics["detector"]) != {"hashing", "codebert"}:
        raise RuntimeError("detector set differs from the freeze")
    primary_project = project_primary_metrics(metrics)
    method_out = primary_method_summary(metrics)
    method_path = output / "primary_method_summary.csv"
    method_out.to_csv(method_path, index=False)

    perf_out = detector_performance_summary(performance)
    perf_path = output / "detector_performance_summary.csv"
    perf_out.to_csv(perf_path, index=False)
    paired_out = detector_paired_summary(performance)
    paired_path = output / "detector_paired_differences.csv"
    paired_out.to_csv(paired_path, index=False)

    weighting = pd.read_csv(analysis_dir / "paired_project_comparisons.csv")
    weighting = weighting[
        (weighting["alpha_vulnerable"].astype(float) == PRIMARY_AV)
        & (weighting["alpha_safe"].astype(float) == PRIMARY_AS)
        & (weighting["baseline"] == "unweighted_mondrian")
        & (weighting["method"] == "estimated_weight_no_gate_clip_20")
    ].copy()
    weighting_path = output / "primary_weighting_comparisons.csv"
    weighting.to_csv(weighting_path, index=False)

    gate_out, gate_projects = gate_summary(metrics, project, gate_disc, gate_seal)
    gate_path = output / "gate_summary.csv"
    gate_out.to_csv(gate_path, index=False)
    gate_projects_path = output / "gate_external_projects.csv"
    gate_projects.to_csv(gate_projects_path, index=False)

    calibration_out = calibration_summary(
        calibration_dir / "calibration_size_project_summary.csv",
        calibration_dir / "calibration_size_sensitivity.csv",
    )
    calibration_path = output / "calibration_summary.csv"
    calibration_out.to_csv(calibration_path, index=False)

    near_out, near_audit = near_duplicate_summary(
        root / "near-duplicate-v1" / "near_duplicate_summary.json",
        near_dir / "primary_sensitivity_summary.json",
        primary_project,
    )
    near_path = output / "near_duplicate_summary.csv"
    near_out.to_csv(near_path, index=False)
    near_audit_path = output / "near_duplicate_audit.json"
    write_json(near_audit_path, near_audit)

    efficiency_summary_path = output / "efficiency_summary.csv"
    efficiency = pd.read_csv(efficiency_dir / "detector_efficiency_summary.csv")
    efficiency.to_csv(efficiency_summary_path, index=False)
    embedding_metadata = json.loads(
        (root / "codebert-v1" / "embeddings-v1" / "metadata.json").read_text(encoding="utf-8")
    )
    dimensions = {
        "protocol_version": config["protocol_version"],
        "config_sha256": config_hash,
        "projects": int(metrics["target_group"].nunique()),
        "detectors": sorted(metrics["detector"].unique().tolist()),
        "seed_addresses": required_seeds,
        "main_metric_rows": int(len(metrics)),
        "project_mean_rows": int(len(primary_project)),
        "risk_budgets_vulnerable": [float(value) for value in config["risk_budgets"]["vulnerable"]],
        "risk_budgets_safe": [float(value) for value in config["risk_budgets"]["safe"]],
        "primary_operating_point": {"alpha_vulnerable": PRIMARY_AV, "alpha_safe": PRIMARY_AS},
        "codebert_revision": embedding_metadata["revision"],
        "codebert_seed_policy": "fit seed 13 once per project; copy deterministic head to five frozen seed addresses",
        "hashing_seed_policy": "five independently fitted technical seeds per project",
        "target_labels_accessed_by_builder": False,
    }
    dimensions_path = output / "study_dimensions.json"
    write_json(dimensions_path, dimensions)

    input_paths = {
        "config": args.config,
        "evaluation_manifest": evaluation_dir / "evaluation_manifest.json",
        "evaluation_metrics": metrics_path,
        "analysis_manifest": analysis_dir / "analysis_manifest.json",
        "project_means": project_path,
        "detector_project_performance": analysis_dir / "detector_project_performance.csv",
        "paired_project_comparisons": analysis_dir / "paired_project_comparisons.csv",
        "gate_discrimination": analysis_dir / "gate_discrimination.csv",
        "gate_seal": root / "gate-v1" / "gate_seal.json",
        "calibration_manifest": calibration_dir / "sensitivity_manifest.json",
        "calibration_project_summary": calibration_dir / "calibration_size_project_summary.csv",
        "calibration_sensitivity": calibration_dir / "calibration_size_sensitivity.csv",
        "near_duplicate_audit": root / "near-duplicate-v1" / "near_duplicate_summary.json",
        "near_duplicate_manifest": near_dir / "sensitivity_manifest.json",
        "near_duplicate_primary_summary": near_dir / "primary_sensitivity_summary.json",
        "efficiency_manifest": efficiency_dir / "efficiency_manifest.json",
        "efficiency_validation": root / "efficiency-validation-v2" / "efficiency_validation.json",
        "efficiency_summary": efficiency_dir / "detector_efficiency_summary.csv",
        "figure_manifest": root / "figures-v2" / "figure_manifest.json",
        "artifact_validation": root / "validation-v2" / "artifact_validation.json",
    }
    inputs = {name: artifact(path, repo_root) for name, path in input_paths.items()}

    output_schemas = {
        "primary_method_summary.csv": method_path,
        "detector_performance_summary.csv": perf_path,
        "detector_paired_differences.csv": paired_path,
        "primary_weighting_comparisons.csv": weighting_path,
        "gate_summary.csv": gate_path,
        "gate_external_projects.csv": gate_projects_path,
        "calibration_summary.csv": calibration_path,
        "near_duplicate_summary.csv": near_path,
        "efficiency_summary.csv": efficiency_summary_path,
    }
    outputs = {
        name: artifact(path, repo_root) | {"rows": int(len(pd.read_csv(path))), "columns": list(pd.read_csv(path).columns)}
        for name, path in output_schemas.items()
    }
    outputs["near_duplicate_audit.json"] = artifact(near_audit_path, repo_root)
    outputs["study_dimensions.json"] = artifact(dimensions_path, repo_root)
    manifest = {
        "protocol_version": config["protocol_version"],
        "status": "complete",
        "config_sha256": config_hash,
        "target_vulnerability_labels_accessed": False,
        "input_artifacts": inputs,
        "outputs": outputs,
        "counts": {
            "projects": int(metrics["target_group"].nunique()),
            "detectors": 2,
            "seeds": len(required_seeds),
            "main_metric_rows": len(metrics),
            "primary_method_rows": len(method_out),
            "detector_performance_rows": len(perf_out),
            "detector_paired_rows": len(paired_out),
            "primary_weighting_rows": len(weighting),
            "gate_rows": len(gate_out),
            "calibration_rows": len(calibration_out),
            "near_duplicate_rows": len(near_out),
            "efficiency_rows": len(efficiency),
        },
        "near_duplicate_audit": near_audit,
        "notes": [
            "All rows are project-level or frozen artifact summaries; no row-level code or target labels are exported.",
            "Support qualification is separate from empirical risk attainment; gate-fail review-only cases are not treated as informative risk successes.",
            "Cell counts across the 3 x 3 grid are descriptive and correlated within project.",
            "Runtime observations are hardware- and parallelization-specific; no speedup or energy claim is supported.",
        ],
    }
    manifest_path = output / "evidence_manifest.json"
    write_json(manifest_path, manifest)
    lines = [
        "# Extension-v2 Evidence Package",
        "",
        "Status: **complete**",
        "",
        f"- Projects/detectors/seeds: {dimensions['projects']} / {len(dimensions['detectors'])} / {len(required_seeds)}",
        f"- Main evaluation rows: {len(metrics):,}",
        f"- Primary method rows: {len(method_out)}",
        f"- Calibration sensitivity rows: {len(calibration_out)}",
        f"- Near-duplicate cohort: {near_audit['retained_target_rows']:,} retained, {near_audit['excluded_target_rows']:,} excluded",
        "- Target labels accessed by builder: false",
        "- CodeBERT seed policy: one deterministic seed-13 fit copied to frozen seed addresses",
        "- Efficiency interpretation: observational and hardware-specific; no direct speedup claim",
        "",
        "All CSV schemas and input/output hashes are recorded in `evidence_manifest.json`.",
    ]
    (output / "evidence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(manifest_path), "outputs": len(outputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
