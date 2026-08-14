"""Independently validate the finalized extension-v2 artifact chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from vultriage.data import load_config, sha256


EXPECTED_METHODS = {
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
PRIMARY_KEY = [
    "detector",
    "target_group",
    "seed",
    "alpha_vulnerable",
    "alpha_safe",
    "method",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(path: Path, expected: str, description: str) -> str:
    require(path.is_file(), f"missing {description}: {path}")
    observed = sha256(path)
    require(observed == str(expected).upper(), f"hash mismatch for {description}: {path}")
    return observed


def verify_prediction_seal(
    root: Path,
    detector: str,
    config_hash: str,
    expected_groups: list[str],
    expected_seeds: list[int],
) -> dict[str, Any]:
    seal_path = root / "predictions" / ("hashing-v8" if detector == "hashing" else "codebert-v2") / "prediction_seal.json"
    seal = read_json(seal_path)
    require(seal.get("detector") == detector, f"detector mismatch in {seal_path}")
    require(seal.get("config_sha256") == config_hash, f"config hash mismatch in {seal_path}")
    require(seal.get("target_label_vault_argument_present") is False, f"label-vault argument admitted by {seal_path}")
    require(seal.get("target_vulnerability_labels_accessed") is False, f"target labels admitted by {seal_path}")
    require(seal.get("selected_project_groups") == expected_groups, f"project order mismatch in {seal_path}")
    inventory = seal.get("prediction_files", {})
    expected_file_count = len(expected_groups) * len(expected_seeds) * 2
    require(len(inventory) == expected_file_count, f"expected {expected_file_count} prediction files for {detector}, observed {len(inventory)}")
    prediction_root = seal_path.parent
    for relative, expected_hash in inventory.items():
        verify_hash(prediction_root / relative, expected_hash, f"{detector} prediction file")
    if detector == "codebert":
        require(seal.get("seed_reuse_mode") == "deterministic_liblinear_replicates", "CodeBERT seed-reuse provenance is missing")
        require(int(seal.get("seed_reuse_reference")) == expected_seeds[0], "CodeBERT seed-reuse reference differs from the freeze")
    else:
        require(seal.get("seed_reuse_mode", "independent") == "independent", "hashing seed provenance is not independent")
    return {
        "path": str(seal_path),
        "sha256": sha256(seal_path),
        "prediction_files": len(inventory),
        "seed_reuse_mode": seal.get("seed_reuse_mode", "independent"),
    }


def verify_metric_frame(
    frame: pd.DataFrame,
    expected_groups: list[str],
    expected_seeds: list[int],
    vulnerable_budgets: list[float],
    safe_budgets: list[float],
) -> None:
    require(not frame.duplicated(PRIMARY_KEY).any(), "duplicate main metric primary key")
    require(set(frame["detector"]) == {"hashing", "codebert"}, "main metrics do not contain exactly two detectors")
    require(set(frame["target_group"]) == set(expected_groups), "main metrics project set differs from the freeze")
    require(set(frame["seed"].astype(int)) == set(expected_seeds), "main metrics seed set differs from the freeze")
    require(set(frame["alpha_vulnerable"].astype(float)) == set(vulnerable_budgets), "vulnerable budget grid differs from the freeze")
    require(set(frame["alpha_safe"].astype(float)) == set(safe_budgets), "safe budget grid differs from the freeze")
    require(set(frame["method"]) == EXPECTED_METHODS, "main metrics method set differs from the frozen evaluator")
    expected_rows = 2 * len(expected_groups) * len(expected_seeds) * len(vulnerable_budgets) * len(safe_budgets) * len(EXPECTED_METHODS)
    require(len(frame) == expected_rows, f"expected {expected_rows} main metric rows, observed {len(frame)}")
    for column in (
        "vulnerable_miscoverage",
        "safe_miscoverage",
        "singleton_coverage",
        "review_load",
        "pr_auc",
        "brier",
        "f1",
    ):
        values = frame[column].astype(float)
        require(values.notna().all() and values.between(0.0, 1.0).all(), f"invalid bounded metric: {column}")
    require(frame["mcc"].astype(float).between(-1.0, 1.0).all(), "MCC falls outside [-1, 1]")
    for column in ("vulnerable_violation", "safe_violation", "max_relative_violation"):
        values = frame[column].astype(float)
        require(values.notna().all() and (values >= 0.0).all(), f"invalid nonnegative metric: {column}")


def verify_manifest_table(manifest: dict[str, Any], table: Path, hash_field: str, row_field: str | None, expected_rows: int, description: str) -> pd.DataFrame:
    verify_hash(table, manifest[hash_field], description)
    frame = pd.read_csv(table)
    require(len(frame) == expected_rows, f"expected {expected_rows} rows in {description}, observed {len(frame)}")
    if row_field is not None:
        require(int(manifest[row_field]) == expected_rows, f"manifest row count mismatch for {description}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"validation output already exists: {args.output}")

    config = load_config(args.config)
    config_hash = sha256(args.config)
    package = read_json(args.root / "source-v2" / "package_summary.json")
    groups = list(package["selected_project_groups"])
    seeds = [int(seed) for seed in config["detectors"]["hashing_sgd"]["seeds"]]
    vulnerable = [float(value) for value in config["risk_budgets"]["vulnerable"]]
    safe = [float(value) for value in config["risk_budgets"]["safe"]]
    require(len(groups) == 24, f"expected 24 frozen projects, observed {len(groups)}")
    require(len(seeds) == 5, f"expected five frozen seeds, observed {len(seeds)}")
    require(package["extension_config_sha256"] == config_hash, "package/config hash mismatch")

    prediction_results = {
        detector: verify_prediction_seal(args.root, detector, config_hash, groups, seeds)
        for detector in ("hashing", "codebert")
    }

    evaluation_dir = args.root / "evaluation-v2"
    evaluation = read_json(evaluation_dir / "evaluation_manifest.json")
    metrics_path = evaluation_dir / "fold_seed_metrics.csv"
    verify_hash(metrics_path, evaluation["metrics_sha256"], "main evaluation metrics")
    metrics = pd.read_csv(metrics_path)
    verify_metric_frame(metrics, groups, seeds, vulnerable, safe)
    require(int(evaluation["metric_rows"]) == len(metrics), "evaluation manifest row count mismatch")
    require(evaluation.get("target_labels_joined_after_prediction_seal") is True, "evaluation does not attest post-seal label joining")
    require(evaluation.get("target_groups") == groups, "evaluation project order mismatch")
    require([int(seed) for seed in evaluation.get("seeds", [])] == seeds, "evaluation seed order mismatch")
    require(len(evaluation.get("decision_files", {})) == 2 * len(groups) * len(seeds), "decision archive inventory is incomplete")

    analysis_dir = args.root / "analysis-v2"
    analysis = read_json(analysis_dir / "analysis_manifest.json")
    expected_project_rows = 2 * len(groups) * len(vulnerable) * len(safe) * len(EXPECTED_METHODS)
    expected_comparison_rows = 2 * len(vulnerable) * len(safe) * 5 * 5
    verify_manifest_table(analysis, analysis_dir / "project_seed_averages.csv", "project_seed_averages_sha256", "project_rows", expected_project_rows, "project seed averages")
    verify_manifest_table(analysis, analysis_dir / "detector_project_performance.csv", "detector_project_performance_sha256", "detector_performance_rows", 2 * len(groups), "detector project performance")
    verify_manifest_table(analysis, analysis_dir / "paired_project_comparisons.csv", "paired_project_comparisons_sha256", "comparison_rows", expected_comparison_rows, "paired project comparisons")
    verify_manifest_table(analysis, analysis_dir / "gate_discrimination.csv", "gate_discrimination_sha256", "gate_rows", 2, "gate discrimination")
    require(analysis["metrics_sha256"] == evaluation["metrics_sha256"], "analysis/evaluation metric hash mismatch")

    calibration_dir = args.root / "calibration-size-v2"
    calibration = read_json(calibration_dir / "sensitivity_manifest.json")
    fractions = list(config["calibration_size_sensitivity"]["fractions"])
    repetitions = int(config["calibration_size_sensitivity"]["repetitions"])
    calibration_rows = 2 * len(groups) * len(seeds) * repetitions * len(fractions) * 3
    calibration_project_rows = 2 * len(groups) * len(fractions) * 3
    calibration_aggregate_rows = 2 * len(fractions) * 3 * 9
    verify_manifest_table(calibration, calibration_dir / "calibration_size_sensitivity.csv", "table_sha256", "rows", calibration_rows, "calibration-size sensitivity")
    verify_manifest_table(calibration, calibration_dir / "calibration_size_project_summary.csv", "project_summary_sha256", "project_summary_rows", calibration_project_rows, "calibration-size project summary")
    verify_manifest_table(calibration, calibration_dir / "calibration_size_aggregate_summary.csv", "aggregate_summary_sha256", "aggregate_summary_rows", calibration_aggregate_rows, "calibration-size aggregate summary")

    near_dir = args.root / "near-duplicate-sensitivity-v2"
    near = read_json(near_dir / "sensitivity_manifest.json")
    near_rows = 2 * len(groups) * len(seeds) * len(vulnerable) * len(safe) * 3
    near_project_rows = 2 * len(groups) * len(vulnerable) * len(safe) * 3
    verify_manifest_table(near, near_dir / "near_duplicate_fold_seed_metrics.csv", "metrics_sha256", "metric_rows", near_rows, "near-duplicate sensitivity metrics")
    verify_manifest_table(near, near_dir / "near_duplicate_project_seed_averages.csv", "project_means_sha256", None, near_project_rows, "near-duplicate project averages")
    verify_hash(near_dir / "primary_sensitivity_summary.json", near["primary_summary_sha256"], "near-duplicate primary summary")
    require(int(near["retained_target_rows"]) == 64329, "near-duplicate retained cohort differs from the sealed audit")
    require(int(near["excluded_target_rows"]) == 15026, "near-duplicate exclusion count differs from the sealed audit")
    require(near.get("primary_cohort_changed") is False, "near-duplicate sensitivity replaced the primary cohort")

    figure_dir = args.root / "figures-v2"
    figure_manifest = read_json(figure_dir / "figure_manifest.json")
    expected_assets = {
        f"{stem}.{suffix}"
        for stem in figure_manifest["figures"]
        for suffix in ("pdf", "png", "svg", "tiff")
    }
    require(set(figure_manifest.get("assets", {})) == expected_assets, "figure asset inventory is incomplete")
    for name, metadata in figure_manifest["assets"].items():
        path = figure_dir / name
        verify_hash(path, metadata["sha256"], f"figure asset {name}")
        require(path.stat().st_size == int(metadata["bytes"]), f"figure byte count mismatch: {name}")
    require(int(figure_manifest.get("raster_export_dpi", 0)) == 600, "figure raster DPI is not 600")
    expected_figure_sources = {
        "metrics": metrics_path,
        "project_means": analysis_dir / "project_seed_averages.csv",
        "gate_discrimination": analysis_dir / "gate_discrimination.csv",
        "gate_seal": args.root / "gate-v1" / "gate_seal.json",
        "calibration_project_summary": calibration_dir / "calibration_size_project_summary.csv",
    }
    require(
        set(figure_manifest.get("source_inputs", {})) == set(expected_figure_sources),
        "figure source-input inventory is incomplete",
    )
    for name, path in expected_figure_sources.items():
        metadata = figure_manifest["source_inputs"][name]
        verify_hash(path, metadata["sha256"], f"figure source input {name}")
        require(
            path.stat().st_size == int(metadata["bytes"]),
            f"figure source-input byte count mismatch: {name}",
        )
    expected_figure_data = {
        "fig2_gate_external_projects.csv",
        "fig2_gate_summary.csv",
        "fig3_primary_automation.csv",
        "fig4_risk_alignment.csv",
        "fig5_calibration_sensitivity.csv",
    }
    require(
        set(figure_manifest.get("data_assets", {})) == expected_figure_data,
        "figure source-data inventory is incomplete",
    )
    for name, metadata in figure_manifest["data_assets"].items():
        path = figure_dir / "data" / name
        verify_hash(path, metadata["sha256"], f"figure source-data asset {name}")
        require(
            path.stat().st_size == int(metadata["bytes"]),
            f"figure source-data byte count mismatch: {name}",
        )
    require(int(figure_manifest.get("rows_metrics", 0)) == len(metrics), "figure metrics row count mismatch")
    require(
        int(figure_manifest.get("rows_project_means", 0)) == expected_project_rows,
        "figure project-summary row count mismatch",
    )
    require(
        int(figure_manifest.get("rows_gate_discrimination", 0)) == 2,
        "figure gate-discrimination row count mismatch",
    )
    require(
        int(figure_manifest.get("rows_calibration_project_summary", 0))
        == calibration_project_rows,
        "figure calibration-summary row count mismatch",
    )

    efficiency_dir = args.root / "efficiency-v2"
    efficiency_manifest_path = efficiency_dir / "efficiency_manifest.json"
    efficiency_manifest = read_json(efficiency_manifest_path)
    efficiency_validation_dir = args.root / "efficiency-validation-v2"
    efficiency_validation_path = efficiency_validation_dir / "efficiency_validation.json"
    efficiency_validation = read_json(efficiency_validation_path)
    require(efficiency_validation.get("status") == "PASS", "independent efficiency validation did not pass")
    require(
        efficiency_validation.get("efficiency_manifest_sha256")
        == sha256(efficiency_manifest_path),
        "efficiency validation/manifest hash mismatch",
    )
    require(
        efficiency_validation.get("config_sha256") == config_hash
        and efficiency_manifest.get("config_sha256") == config_hash,
        "efficiency/config hash mismatch",
    )
    require(
        efficiency_validation.get("target_vulnerability_labels_accessed") is False
        and efficiency_manifest.get("target_vulnerability_labels_accessed") is False,
        "efficiency evidence admits target-label access",
    )
    require(
        int(efficiency_validation.get("projects", 0)) == len(groups)
        and [int(value) for value in efficiency_validation.get("seeds", [])] == seeds,
        "efficiency validation dimensions differ from the freeze",
    )
    require(
        int(efficiency_validation.get("executed_head_fit_rows", 0)) == 144
        and int(efficiency_validation.get("codebert_parallel_parts", 0)) == 4
        and int(efficiency_validation.get("summary_rows", 0)) == 2,
        "efficiency validation counts are incomplete",
    )
    verify_hash(
        Path(__file__).with_name("validate_extension_v2_efficiency.py"),
        efficiency_validation["validator_sha256"],
        "efficiency validator",
    )
    for path, hash_field, description in (
        (efficiency_dir / "executed_head_fits.csv", "executed_head_fits_sha256", "executed head fits"),
        (efficiency_dir / "codebert_part_runtime.csv", "codebert_part_runtime_sha256", "CodeBERT part runtime"),
        (efficiency_dir / "detector_efficiency_summary.csv", "detector_efficiency_summary_sha256", "detector efficiency summary"),
    ):
        verify_hash(path, efficiency_manifest[hash_field], description)
    require(
        efficiency_manifest.get("prediction_seals", {}).get("hashing")
        == prediction_results["hashing"]["sha256"]
        and efficiency_manifest.get("prediction_seals", {}).get("codebert")
        == prediction_results["codebert"]["sha256"],
        "efficiency audit/prediction-seal hash mismatch",
    )

    result = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "projects": len(groups),
        "detectors": ["hashing", "codebert"],
        "seeds": seeds,
        "risk_grid_cells": len(vulnerable) * len(safe),
        "methods": len(EXPECTED_METHODS),
        "main_metric_rows": len(metrics),
        "project_rows": expected_project_rows,
        "calibration_sensitivity_rows": calibration_rows,
        "near_duplicate_sensitivity_rows": near_rows,
        "near_duplicate_retained_rows": int(near["retained_target_rows"]),
        "near_duplicate_excluded_rows": int(near["excluded_target_rows"]),
        "prediction_seals": prediction_results,
        "evaluation_manifest_sha256": sha256(evaluation_dir / "evaluation_manifest.json"),
        "analysis_manifest_sha256": sha256(analysis_dir / "analysis_manifest.json"),
        "calibration_manifest_sha256": sha256(calibration_dir / "sensitivity_manifest.json"),
        "near_duplicate_manifest_sha256": sha256(near_dir / "sensitivity_manifest.json"),
        "figure_manifest_sha256": sha256(figure_dir / "figure_manifest.json"),
        "figure_assets": len(expected_assets),
        "figure_data_assets": len(expected_figure_data),
        "efficiency_manifest_sha256": sha256(efficiency_manifest_path),
        "efficiency_validation_sha256": sha256(efficiency_validation_path),
        "validator_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True)
    json_path = args.output / "artifact_validation.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Extension-v2 Artifact Validation",
        "",
        "- Status: **PASS**",
        f"- Projects: {result['projects']}",
        f"- Detectors: {', '.join(result['detectors'])}",
        f"- Seeds: {', '.join(str(value) for value in seeds)}",
        f"- Main metric rows: {result['main_metric_rows']:,}",
        f"- Project-level rows: {result['project_rows']:,}",
        f"- Calibration-size rows: {result['calibration_sensitivity_rows']:,}",
        f"- Near-duplicate sensitivity rows: {result['near_duplicate_sensitivity_rows']:,}",
        f"- Retained/excluded near-duplicate cohort rows: {result['near_duplicate_retained_rows']:,} / {result['near_duplicate_excluded_rows']:,}",
        f"- Figure/data assets: {result['figure_assets']} / {result['figure_data_assets']}",
        "- Efficiency audit: independently validated observational timing and memory evidence",
        f"- Full JSON record: `{json_path.name}`",
    ]
    (args.output / "artifact_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
