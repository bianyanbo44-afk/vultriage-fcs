"""Build the detector-shift qualification registry and contract audit tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vultriage.data import sha256
from vultriage.knowledge_contract import (
    qualification_metrics,
    stable_project_partition,
)


def load_external_frame(metrics_path: Path, project_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    selected = metrics[
        (metrics["method"] == "vultriage_full_gate_clip_20")
        & (metrics["alpha_vulnerable"] == 0.1)
        & (metrics["alpha_safe"] == 0.2)
    ].copy()
    selected["support_record"] = selected["support"].map(json.loads)
    selected["gate_pass_seed"] = selected["support_record"].map(
        lambda item: bool(item["passed"])
    )
    selected["gate_probability_seed"] = selected["support_record"].map(
        lambda item: float(item["gate_probability"])
    )
    gate = selected.groupby(["detector", "target_group"], as_index=False).agg(
        supported=("gate_pass_seed", "all"),
        gate_probability=("gate_probability_seed", "mean"),
        seed_addresses=("seed", "nunique"),
    )
    project = pd.read_csv(project_path)
    raw = project[
        (project["method"] == "estimated_weight_no_gate_clip_20")
        & (project["alpha_vulnerable"] == 0.1)
        & (project["alpha_safe"] == 0.2)
    ][
        [
            "detector",
            "target_group",
            "max_relative_violation",
            "singleton_coverage",
        ]
    ]
    frame = gate.merge(
        raw,
        on=["detector", "target_group"],
        how="inner",
        validate="one_to_one",
    )
    frame["severe_raw_violation"] = frame["max_relative_violation"] > 0.5
    if len(frame) != 48 or frame["target_group"].nunique() != 24:
        raise RuntimeError("contract analysis requires two detectors and 24 projects")
    return frame.sort_values(["detector", "target_group"]).reset_index(drop=True)


def metric_record(
    detector: str, partition: str, subset: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    metric = qualification_metrics(
        subset["severe_raw_violation"].astype(int),
        subset["gate_probability"],
        subset["supported"],
    )
    return {
        "detector": detector,
        "partition": partition,
        **asdict(metric),
        "qualification_threshold": threshold,
        "qualified": metric.qualifies(threshold),
    }


def partition_stability(
    frame: pd.DataFrame,
    projects: list[str],
    prefix: str,
    partitions: int,
    registration_size: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    for index in range(partitions):
        registration, confirmation = stable_project_partition(
            projects, f"{prefix}{index}", registration_size
        )
        registration_set = set(registration)
        for detector, detector_frame in frame.groupby("detector"):
            register_frame = detector_frame[
                detector_frame["target_group"].isin(registration_set)
            ]
            confirm_frame = detector_frame[
                ~detector_frame["target_group"].isin(registration_set)
            ]
            register = qualification_metrics(
                register_frame["severe_raw_violation"].astype(int),
                register_frame["gate_probability"],
                register_frame["supported"],
            )
            confirm = qualification_metrics(
                confirm_frame["severe_raw_violation"].astype(int),
                confirm_frame["gate_probability"],
                confirm_frame["supported"],
            )
            score_rows.append(
                {
                    "partition_index": index,
                    "detector": detector,
                    "registration_score": register.qualification_score,
                    "confirmation_score": confirm.qualification_score,
                    "registration_qualified": register.qualifies(threshold),
                    "confirmation_qualified": confirm.qualifies(threshold),
                    "both_qualified": register.qualifies(threshold)
                    and confirm.qualifies(threshold),
                }
            )
    scores = pd.DataFrame(score_rows)
    rows: list[dict[str, Any]] = []
    for detector, subset in scores.groupby("detector"):
        valid = subset[
            np.isfinite(subset["registration_score"])
            & np.isfinite(subset["confirmation_score"])
        ]
        rows.append(
            {
                "detector": detector,
                "partitions_requested": partitions,
                "valid_partitions": len(valid),
                "registration_qualification_rate": float(
                    valid["registration_qualified"].mean()
                ),
                "both_halves_qualification_rate": float(
                    valid["both_qualified"].mean()
                ),
                "median_registration_score": float(
                    valid["registration_score"].median()
                ),
                "median_confirmation_score": float(
                    valid["confirmation_score"].median()
                ),
                "q10_registration_score": float(
                    valid["registration_score"].quantile(0.1)
                ),
                "q10_confirmation_score": float(
                    valid["confirmation_score"].quantile(0.1)
                ),
            }
        )
    return scores, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--project-means", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = config["registration"]
    frame = load_external_frame(args.metrics, args.project_means)
    projects = sorted(frame["target_group"].unique())
    registration, confirmation = stable_project_partition(
        projects,
        settings["split_salt"],
        int(settings["registration_projects"]),
    )
    registration_set = set(registration)
    split = pd.DataFrame(
        [
            {
                "target_group": project,
                "partition": "registration"
                if project in registration_set
                else "confirmation",
            }
            for project in projects
        ]
    )
    threshold = float(settings["qualification_threshold"])
    summary_rows: list[dict[str, Any]] = []
    registry: dict[str, bool] = {}
    for detector, detector_frame in frame.groupby("detector"):
        register_frame = detector_frame[
            detector_frame["target_group"].isin(registration_set)
        ]
        confirm_frame = detector_frame[
            ~detector_frame["target_group"].isin(registration_set)
        ]
        register_record = metric_record(
            detector, "registration", register_frame, threshold
        )
        confirm_record = metric_record(
            detector, "confirmation", confirm_frame, threshold
        )
        registry[detector] = bool(register_record["qualified"])
        summary_rows.extend([register_record, confirm_record])
    qualification = pd.DataFrame(summary_rows)
    confirmation_frame = frame[
        ~frame["target_group"].isin(registration_set)
    ].copy()
    confirmation_frame["registry_qualified"] = confirmation_frame["detector"].map(
        registry
    )
    confirmation_frame["contract_pass"] = (
        confirmation_frame["registry_qualified"] & confirmation_frame["supported"]
    )
    confirmation_frame["contract_action"] = np.where(
        confirmation_frame["contract_pass"], "eligible-for-singleton", "review-only"
    )
    stability_config = config["partition_stability"]
    stability_scores, stability_summary = partition_stability(
        frame,
        projects,
        stability_config["split_salt_prefix"],
        int(stability_config["deterministic_partitions"]),
        int(settings["registration_projects"]),
        threshold,
    )
    paths = {
        "fixed_split_projects.csv": split,
        "qualification_summary.csv": qualification,
        "confirmation_contract_decisions.csv": confirmation_frame,
        "partition_stability_scores.csv": stability_scores,
        "partition_stability_summary.csv": stability_summary,
    }
    for name, table in paths.items():
        table.to_csv(args.output / name, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": config["policy_version"],
        "analysis_status": config["analysis_status"],
        "config_sha256": sha256(args.config),
        "metrics_sha256": sha256(args.metrics),
        "project_means_sha256": sha256(args.project_means),
        "registration_projects": list(registration),
        "confirmation_projects": list(confirmation),
        "registry": registry,
        "qualification_threshold": threshold,
        "partition_stability_runs": int(
            stability_config["deterministic_partitions"]
        ),
        "outputs": {
            name: {"rows": len(table), "sha256": sha256(args.output / name)}
            for name, table in paths.items()
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output / "contract_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
