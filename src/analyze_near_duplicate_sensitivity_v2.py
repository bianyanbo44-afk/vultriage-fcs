"""Re-evaluate sealed decisions after excluding cross-dataset near duplicates.

The primary extension-v2 cohort is unchanged. This sensitivity analysis reads
only the evaluator's post-seal decision archives and the label-free retained
row list produced by the near-duplicate audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_e1 import add_targets
from vultriage.data import load_config, sha256
from vultriage.metrics import triage_metrics


ARCHIVE_METHODS = {
    "unweighted_mondrian": "unweighted_mondrian",
    "estimated_weight_no_gate": "estimated_weight_no_gate_clip_20",
    "vultriage_full_gate": "vultriage_full_gate_clip_20",
}


def operating_key(alpha_vulnerable: float, alpha_safe: float) -> str:
    return f"av{alpha_vulnerable:g}_as{alpha_safe:g}".replace(".", "p")


def load_retained_positions(path: Path) -> dict[str, set[int]]:
    retained: dict[str, set[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            retained.setdefault(row["project_group"], set()).add(int(row["position"]))
    if not retained:
        raise RuntimeError("near-duplicate sensitivity cohort is empty")
    return retained


def verify_inputs(
    evaluation: Path,
    cohort: Path,
    near_duplicate_summary: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_manifest = json.loads(
        (evaluation / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    near_duplicate = json.loads(near_duplicate_summary.read_text(encoding="utf-8"))
    expected_cohort_hash = near_duplicate["artifacts"]["sensitivity_cohort"]["sha256"]
    if sha256(cohort) != expected_cohort_hash:
        raise RuntimeError("near-duplicate sensitivity cohort hash mismatch")
    for relative, expected_hash in evaluation_manifest["decision_files"].items():
        path = evaluation / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"decision archive hash mismatch: {relative}")
    return evaluation_manifest, near_duplicate


def evaluate_archive(
    detector: str,
    group: str,
    seed: int,
    archive: Any,
    retained_positions: set[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    positions = archive["target_positions"].astype(int)
    labels = archive["target_labels"].astype(int)
    probabilities = archive["target_p_vulnerable"].astype(float)
    if len(positions) != len(labels) or len(labels) != len(probabilities):
        raise RuntimeError(f"decision array length mismatch: {detector}/{group}/{seed}")
    mask = np.fromiter(
        (int(position) in retained_positions for position in positions),
        dtype=bool,
        count=len(positions),
    )
    observed_positions = set(int(value) for value in positions[mask])
    if observed_positions != retained_positions:
        raise RuntimeError(f"retained-position mismatch: {detector}/{group}/{seed}")
    records: list[dict[str, Any]] = []
    for alpha_vulnerable in config["risk_budgets"]["vulnerable"]:
        for alpha_safe in config["risk_budgets"]["safe"]:
            av = float(alpha_vulnerable)
            ass = float(alpha_safe)
            prefix = operating_key(av, ass)
            for archive_name, method in ARCHIVE_METHODS.items():
                key = f"{prefix}__{archive_name}"
                if key not in archive.files:
                    raise RuntimeError(f"missing decision array {key}: {detector}/{group}/{seed}")
                sets = archive[key].astype(bool)[mask]
                metrics = add_targets(
                    triage_metrics(labels[mask], probabilities[mask], sets), av, ass
                )
                metrics["max_relative_violation"] = max(
                    metrics["vulnerable_violation"] / av,
                    metrics["safe_violation"] / ass,
                )
                records.append(
                    {
                        "detector": detector,
                        "target_group": group,
                        "seed": int(seed),
                        "alpha_vulnerable": av,
                        "alpha_safe": ass,
                        "method": method,
                        "near_duplicate_sensitivity": True,
                        **metrics,
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--near-duplicate-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    evaluation_manifest, near_duplicate = verify_inputs(
        args.evaluation, args.cohort, args.near_duplicate_summary
    )
    config = load_config(args.config)
    retained = load_retained_positions(args.cohort)
    expected_groups = set(evaluation_manifest["target_groups"])
    if set(retained) != expected_groups:
        raise RuntimeError("sensitivity cohort project set differs from evaluation")
    expected_counts = near_duplicate["counts"]["retained_rows_by_project"]
    for group, positions in retained.items():
        if len(positions) != int(expected_counts[group]):
            raise RuntimeError(f"retained-row count mismatch for {group}")

    records: list[dict[str, Any]] = []
    for detector in ("hashing", "codebert"):
        for group in evaluation_manifest["target_groups"]:
            for seed in evaluation_manifest["seeds"]:
                path = args.evaluation / "decisions" / detector / group / f"seed-{seed}.npz"
                with np.load(path) as archive:
                    records.extend(
                        evaluate_archive(
                            detector,
                            group,
                            int(seed),
                            archive,
                            retained[group],
                            config,
                        )
                    )

    args.output.mkdir(parents=True)
    frame = pd.DataFrame(records)
    metrics_path = args.output / "near_duplicate_fold_seed_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    keys = [
        "detector",
        "target_group",
        "alpha_vulnerable",
        "alpha_safe",
        "method",
    ]
    numeric = [
        "vulnerable_violation",
        "safe_violation",
        "max_relative_violation",
        "singleton_coverage",
        "review_load",
        "pr_auc",
        "brier",
        "f1",
        "mcc",
    ]
    project = frame.groupby(keys, as_index=False)[numeric].mean()
    project_path = args.output / "near_duplicate_project_seed_averages.csv"
    project.to_csv(project_path, index=False)

    primary = project[
        (project["alpha_vulnerable"] == 0.1)
        & (project["alpha_safe"] == 0.2)
    ]
    primary_summary: list[dict[str, Any]] = []
    for detector in ("hashing", "codebert"):
        detector_rows = primary[primary["detector"] == detector]
        for method in ARCHIVE_METHODS.values():
            subset = detector_rows[detector_rows["method"] == method]
            primary_summary.append(
                {
                    "detector": detector,
                    "method": method,
                    "projects": int(len(subset)),
                    "median_max_relative_violation": float(
                        subset["max_relative_violation"].median()
                    ),
                    "median_singleton_coverage": float(
                        subset["singleton_coverage"].median()
                    ),
                    "both_budget_attainment_projects": int(
                        (
                            (subset["vulnerable_violation"] == 0)
                            & (subset["safe_violation"] == 0)
                        ).sum()
                    ),
                }
            )
    summary_path = args.output / "primary_sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(primary_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "analysis_scope": "exclude target rows with exact lexical-token-set Jaccard >= 0.90 to any PrimeVul row",
        "primary_cohort_changed": False,
        "post_prediction_label_bearing_decisions_used": True,
        "retained_target_rows": int(sum(len(value) for value in retained.values())),
        "excluded_target_rows": int(near_duplicate["counts"]["excluded_target_rows"]),
        "projects": len(retained),
        "detectors": ["hashing", "codebert"],
        "seeds": [int(seed) for seed in evaluation_manifest["seeds"]],
        "metric_rows": len(frame),
        "evaluation_manifest_sha256": sha256(args.evaluation / "evaluation_manifest.json"),
        "near_duplicate_summary_sha256": sha256(args.near_duplicate_summary),
        "cohort_sha256": sha256(args.cohort),
        "metrics_sha256": sha256(metrics_path),
        "project_means_sha256": sha256(project_path),
        "primary_summary_sha256": sha256(summary_path),
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
