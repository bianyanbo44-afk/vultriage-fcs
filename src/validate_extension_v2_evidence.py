"""Independent validator for the extension-v2 paper evidence package.

This validator deliberately re-derives key dimensions and primary medians from
sealed upstream tables instead of importing the evidence builder.  It checks
that the package is summary-only, hash-complete, and consistent with the
frozen experiment before manuscript numbers are consumed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
BANNED_COLUMNS = {
    "row_id",
    "target",
    "label",
    "function",
    "source_code",
    "code",
    "prediction",
    "probability",
    "probabilities",
    "embedding",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_support(value: object) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import ast

        parsed = ast.literal_eval(text)
    return parsed if isinstance(parsed, dict) else None


def verify_file(path: Path, expected: str, description: str) -> None:
    require(path.is_file(), f"missing {description}: {path}")
    require(sha256(path) == str(expected).upper(), f"hash mismatch for {description}: {path}")


def verify_csv(path: Path, expected: dict[str, Any], description: str) -> pd.DataFrame:
    verify_file(path, expected["sha256"], description)
    frame = pd.read_csv(path)
    require(len(frame) == int(expected["rows"]), f"row count mismatch for {description}")
    require(list(frame.columns) == list(expected["columns"]), f"schema mismatch for {description}")
    lowered = {column.lower() for column in frame.columns}
    require(not lowered.intersection(BANNED_COLUMNS), f"sensitive column exported in {description}")
    return frame


def verify_primary_rederivation(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    selected = metrics[
        (metrics["alpha_vulnerable"].astype(float) == PRIMARY_AV)
        & (metrics["alpha_safe"].astype(float) == PRIMARY_AS)
    ].copy()
    require(len(selected) == 2 * 24 * 5 * len(PRIMARY_METHODS), "primary selection dimensions differ")
    grouped = selected.groupby(["detector", "target_group", "method"], as_index=False)[SUMMARY_COLUMNS].mean()
    for detector in ("hashing", "codebert"):
        for method in sorted(PRIMARY_METHODS):
            observed = grouped[(grouped["detector"] == detector) & (grouped["method"] == method)]
            packaged = summary[(summary["detector"] == detector) & (summary["method"] == method)]
            require(len(observed) == 24 and len(packaged) == 1, f"primary cell missing: {detector}/{method}")
            row = packaged.iloc[0]
            for column in SUMMARY_COLUMNS:
                expected = float(observed[column].median())
                actual = float(row[f"overall_median_{column}"])
                require(np.isclose(expected, actual, atol=1e-12, rtol=0), f"primary median mismatch: {detector}/{method}/{column}")
            expected_attainment = int(
                np.sum(
                    (observed["vulnerable_violation"].to_numpy(float) == 0.0)
                    & (observed["safe_violation"].to_numpy(float) == 0.0)
                )
            )
            require(int(row["overall_attainment_projects"]) == expected_attainment, f"attainment mismatch: {detector}/{method}")
    support = selected.copy()
    support["record"] = support["support"].map(parse_support)
    support["pass"] = support["record"].map(
        lambda item: bool(item["passed"]) if item is not None and "passed" in item else np.nan
    )
    for detector, method in (
        (detector, method)
        for detector in ("hashing", "codebert")
        for method in sorted(PRIMARY_METHODS)
    ):
        subset = support[(support["detector"] == detector) & (support["method"] == method)]
        packaged = summary[(summary["detector"] == detector) & (summary["method"] == method)].iloc[0]
        if subset["pass"].notna().any():
            project_pass = subset.groupby("target_group")["pass"].all()
            require(bool(packaged["support_recorded"]), f"support provenance dropped: {detector}/{method}")
            require(int(packaged["support_pass_projects"]) == int(project_pass.sum()), f"support count mismatch: {detector}/{method}")
        else:
            require(not bool(packaged["support_recorded"]), f"spurious support provenance: {detector}/{method}")


def verify_gate_summary(gate: pd.DataFrame, gate_disc: pd.DataFrame, gate_seal: dict[str, Any]) -> None:
    require(list(gate["domain"]) == ["PrimeVul development", "DiverseVul external", "DiverseVul external"], "gate row order mismatch")
    development = gate.iloc[0]
    require(np.isclose(float(development["auroc"]), float(gate_seal["crossfit_auroc"])), "development gate AUROC mismatch")
    require(np.isclose(float(development["auprc"]), float(gate_seal["crossfit_auprc"])), "development gate AUPRC mismatch")
    for detector in ("hashing", "codebert"):
        expected = gate_disc[gate_disc["detector"] == detector].iloc[0]
        observed = gate[(gate["domain"] == "DiverseVul external") & (gate["detector"] == detector)].iloc[0]
        for column, source in (
            ("auroc", "gate_auroc_severe_violation"),
            ("auprc", "gate_auprc_severe_violation"),
            ("pass_minus_fail_median", "median_raw_violation_pass_minus_fail"),
            ("ci_lower", "median_raw_violation_pass_minus_fail_bootstrap_ci_lower"),
            ("ci_upper", "median_raw_violation_pass_minus_fail_bootstrap_ci_upper"),
        ):
            require(np.isclose(float(observed[column]), float(expected[source]), atol=1e-12, rtol=0), f"gate summary mismatch: {detector}/{column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"validation output already exists: {args.output}")
    repo_root = Path.cwd().resolve()
    package = args.evidence.resolve()
    manifest_path = package / "evidence_manifest.json"
    manifest = read_json(manifest_path)
    config = load_config(args.config)
    config_hash = sha256(args.config)
    require(manifest.get("status") == "complete", "evidence package is not complete")
    require(manifest.get("config_sha256") == config_hash, "evidence/config hash mismatch")
    require(manifest.get("target_vulnerability_labels_accessed") is False, "evidence builder accessed target labels")
    for name, metadata in manifest["input_artifacts"].items():
        path = repo_root / Path(metadata["path"])
        verify_file(path, metadata["sha256"], f"input artifact {name}")
        require(path.stat().st_size == int(metadata["bytes"]), f"input byte count mismatch: {name}")
    outputs = manifest["outputs"]
    frames: dict[str, pd.DataFrame] = {}
    for name, metadata in outputs.items():
        path = repo_root / Path(metadata["path"])
        if name.endswith(".csv"):
            frames[name] = verify_csv(path, metadata, name)
        else:
            verify_file(path, metadata["sha256"], name)
    require(len(frames["primary_method_summary.csv"]) == 50, "primary method summary must contain 50 rows")
    require(len(frames["detector_performance_summary.csv"]) == 10, "detector performance summary must contain 10 rows")
    require(len(frames["detector_paired_differences.csv"]) == 5, "detector paired summary must contain 5 rows")
    require(len(frames["primary_weighting_comparisons.csv"]) == 10, "primary weighting summary must contain 10 rows")
    require(len(frames["gate_summary.csv"]) == 3, "gate summary must contain 3 rows")
    require(len(frames["gate_external_projects.csv"]) == 48, "gate project summary must contain 48 rows")
    require(len(frames["calibration_summary.csv"]) == 24, "calibration summary must contain 24 rows")
    require(len(frames["near_duplicate_summary.csv"]) == 12, "near-duplicate summary must contain 12 rows")
    require(len(frames["efficiency_summary.csv"]) == 2, "efficiency summary must contain 2 rows")

    root = args.root
    metrics = pd.read_csv(root / "evaluation-v2" / "fold_seed_metrics.csv")
    verify_primary_rederivation(metrics, frames["primary_method_summary.csv"])
    verify_gate_summary(
        frames["gate_summary.csv"],
        pd.read_csv(root / "analysis-v2" / "gate_discrimination.csv"),
        read_json(root / "gate-v1" / "gate_seal.json"),
    )
    dimensions = read_json(package / "study_dimensions.json")
    require(dimensions["projects"] == 24 and dimensions["main_metric_rows"] == 54000, "study dimensions mismatch")
    require(dimensions["codebert_seed_policy"].startswith("fit seed 13 once"), "CodeBERT seed reuse not disclosed")
    near_audit = read_json(package / "near_duplicate_audit.json")
    require(near_audit["candidate_pairs"] == 866094 and near_audit["verified_candidates"] == 866094, "near-duplicate candidate count mismatch")
    require(near_audit["flagged_pairs"] == 24606 and near_audit["excluded_target_rows"] == 15026 and near_audit["retained_target_rows"] == 64329, "near-duplicate audit counts mismatch")
    efficiency = frames["efficiency_summary.csv"]
    require(set(efficiency["detector"]) == {"hashing", "codebert"}, "efficiency detector set mismatch")
    require("comparison_boundary" in efficiency.columns, "efficiency interpretation boundary missing")
    require(efficiency["comparison_boundary"].astype(str).str.contains("not a hashing speedup estimate").any(), "efficiency speedup boundary missing")

    result = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "evidence_manifest_sha256": sha256(manifest_path),
        "projects": 24,
        "main_metric_rows": 54000,
        "csv_outputs": len(frames),
        "target_vulnerability_labels_accessed": False,
        "validator_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True)
    json_path = args.output / "evidence_validation.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "evidence_validation.md").write_text(
        "\n".join(
            [
                "# Extension-v2 Evidence Validation",
                "",
                "- Status: **PASS**",
                "- Projects/main rows: 24 / 54,000",
                f"- CSV outputs checked: {len(frames)}",
                "- Target-label access: false",
                f"- Evidence manifest: `{manifest_path.name}`",
                f"- Full JSON record: `{json_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
