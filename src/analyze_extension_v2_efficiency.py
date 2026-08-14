"""Summarize sealed extension-v2 runtime and memory observations.

This is a provenance audit of measurements already written by the frozen
prediction and embedding jobs.  It does not rerun either detector and it does
not access the DiverseVul label vault.
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


SEEDS = [13, 37, 73, 101, 137]
EXPECTED_CODEBERT_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


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


def select_executed_fits(observations: pd.DataFrame, reference_seed: int) -> pd.DataFrame:
    """Return one row per head fit that was actually executed.

    Hashing fits every frozen seed.  The deterministic CodeBERT/liblinear
    branch executes only the reference seed and reuses that fitted head for the
    remaining technical seed addresses.
    """

    required = {
        "detector",
        "target_group",
        "seed",
        "head_fit_seconds",
        "technical_seed_reused",
        "seed_reused_from",
    }
    require(required.issubset(observations.columns), "head-fit observation schema is incomplete")
    hashing = observations.loc[observations["detector"] == "hashing"].copy()
    codebert = observations.loc[observations["detector"] == "codebert"].copy()
    require(not hashing.empty and not codebert.empty, "both detector observations are required")
    require((~hashing["technical_seed_reused"].astype(bool)).all(), "hashing unexpectedly reuses technical seeds")
    require(codebert["technical_seed_reused"].astype(bool).all(), "CodeBERT seed reuse is not fully recorded")
    require(
        (pd.to_numeric(codebert["seed_reused_from"]) == int(reference_seed)).all(),
        "CodeBERT seed-reuse reference differs from the frozen reference",
    )
    for _, group in codebert.groupby("target_group", sort=False):
        require(set(group["seed"].astype(int)) == set(SEEDS), "CodeBERT project does not contain all frozen seed addresses")
        for column in ("head_fit_seconds", "selected_parameter", "source_validation_pr_auc"):
            require(group[column].nunique(dropna=False) == 1, f"reused CodeBERT metadata differs within a project: {column}")
    codebert_executed = codebert.loc[codebert["seed"].astype(int) == int(reference_seed)].copy()
    result = pd.concat([hashing, codebert_executed], ignore_index=True)
    require((pd.to_numeric(result["head_fit_seconds"]) > 0.0).all(), "non-positive head-fit duration")
    result["execution_status"] = "executed"
    return result.sort_values(["detector", "target_group", "seed"]).reset_index(drop=True)


def head_fit_summary(frame: pd.DataFrame, detector: str) -> dict[str, float | int]:
    values = pd.to_numeric(frame.loc[frame["detector"] == detector, "head_fit_seconds"]).to_numpy(dtype=float)
    require(values.size > 0, f"no executed head fits for {detector}")
    return {
        "executed_head_fits": int(values.size),
        "head_fit_seconds_sum": float(values.sum()),
        "head_fit_seconds_median": float(np.median(values)),
        "head_fit_seconds_q25": float(np.quantile(values, 0.25)),
        "head_fit_seconds_q75": float(np.quantile(values, 0.75)),
        "head_fit_seconds_min": float(values.min()),
        "head_fit_seconds_max": float(values.max()),
    }


def collect_observations(
    prediction_root: Path,
    detector: str,
    groups: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    directory = prediction_root / ("hashing-v8" if detector == "hashing" else "codebert-v2")
    seal_path = directory / "prediction_seal.json"
    seal = read_json(seal_path)
    require(seal.get("detector") == detector, f"detector mismatch in {seal_path}")
    require(seal.get("selected_project_groups") == groups, f"project order mismatch in {seal_path}")
    require(seal.get("target_label_vault_argument_present") is False, f"label-vault argument admitted by {seal_path}")
    require(seal.get("target_vulnerability_labels_accessed") is False, f"target labels admitted by {seal_path}")
    inventory = seal.get("prediction_files", {})
    rows: list[dict[str, Any]] = []
    for group in groups:
        for seed in seeds:
            relative = f"predictions/{group}/seed-{seed}.json"
            require(relative in inventory, f"missing sealed metadata entry: {relative}")
            path = directory / relative
            verify_hash(path, inventory[relative], f"{detector} prediction metadata")
            metadata = read_json(path)
            require(metadata.get("detector") == detector, f"metadata detector mismatch: {path}")
            require(metadata.get("target_group") == group, f"metadata project mismatch: {path}")
            require(int(metadata.get("seed")) == seed, f"metadata seed mismatch: {path}")
            require(metadata.get("target_vulnerability_labels_accessed") is False, f"target labels admitted by {path}")
            rows.append(
                {
                    "detector": detector,
                    "target_group": group,
                    "seed": seed,
                    "head_fit_seconds": float(metadata["head_fit_seconds"]),
                    "selected_parameter": float(metadata["selected_parameter"]),
                    "source_validation_pr_auc": float(metadata["source_validation_pr_auc"]),
                    "technical_seed_reused": bool(metadata["technical_seed_reused"]),
                    "seed_reused_from": metadata.get("seed_reused_from"),
                    "metadata_relative_path": relative,
                    "metadata_sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows), seal, seal_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"efficiency output already exists: {args.output}")

    config = load_config(args.config)
    config_hash = sha256(args.config)
    package_path = args.root / "source-v2" / "package_summary.json"
    package = read_json(package_path)
    require(package.get("extension_config_sha256") == config_hash, "package/config hash mismatch")
    require(package.get("target_label_vault_copied_or_opened") is False, "source preparation reports target-label access")
    groups = list(package["selected_project_groups"])
    seeds = [int(value) for value in config["detectors"]["hashing_sgd"]["seeds"]]
    require(len(groups) == 24, f"expected 24 projects, observed {len(groups)}")
    require(seeds == SEEDS, f"seed order differs from the freeze: {seeds}")

    frames: list[pd.DataFrame] = []
    seals: dict[str, dict[str, Any]] = {}
    seal_paths: dict[str, Path] = {}
    for detector in ("hashing", "codebert"):
        frame, seal, seal_path = collect_observations(args.root / "predictions", detector, groups, seeds)
        frames.append(frame)
        seals[detector] = seal
        seal_paths[detector] = seal_path
    observations = pd.concat(frames, ignore_index=True)
    require(len(observations) == 2 * len(groups) * len(seeds), "observation inventory is incomplete")

    reference_seed = int(seals["codebert"]["seed_reuse_reference"])
    require(seals["codebert"].get("seed_reuse_mode") == "deterministic_liblinear_replicates", "CodeBERT seed provenance is missing")
    require(seals["hashing"].get("seed_reuse_mode", "independent") == "independent", "hashing seed provenance is not independent")
    executed = select_executed_fits(observations, reference_seed)
    require(len(executed) == len(groups) * (len(seeds) + 1), "executed head-fit count differs from the frozen seed policy")

    embedding_path = args.root / "codebert-v1" / "embeddings-v1" / "metadata.json"
    embedding = read_json(embedding_path)
    require(embedding.get("labels_used") is False, "CodeBERT embedding metadata reports label use")
    require(embedding.get("revision") == EXPECTED_CODEBERT_REVISION, "CodeBERT resolved revision mismatch")
    require(int(embedding["rows"]) == int(package["source_rows"]) + int(package["target_rows"]), "embedding row count mismatch")

    part_root = args.root / "predictions" / "codebert-v2-balanced-parts"
    part_paths = sorted(part_root.glob("part-*/prediction_seal.json"))
    require(len(part_paths) == 4, f"expected four CodeBERT part seals, observed {len(part_paths)}")
    part_records: list[dict[str, Any]] = []
    part_projects: list[str] = []
    for path in part_paths:
        part = read_json(path)
        require(part.get("detector") == "codebert", f"detector mismatch in {path}")
        require(part.get("target_vulnerability_labels_accessed") is False, f"target labels admitted by {path}")
        require(part.get("seed_reuse_mode") == "deterministic_liblinear_replicates", f"seed provenance mismatch in {path}")
        selected = list(part["selected_project_groups"])
        part_projects.extend(selected)
        part_records.append(
            {
                "part": path.parent.name,
                "projects": len(selected),
                "elapsed_seconds": float(part["elapsed_seconds"]),
                "prediction_files": len(part["prediction_files"]),
                "seal_relative_path": path.relative_to(args.root).as_posix(),
                "seal_sha256": sha256(path),
            }
        )
    require(part_projects == groups, "CodeBERT part project order/coverage differs from the freeze")
    part_frame = pd.DataFrame(part_records).sort_values("part").reset_index(drop=True)

    hashing_head = head_fit_summary(executed, "hashing")
    codebert_head = head_fit_summary(executed, "codebert")
    codebert_part_critical = float(part_frame["elapsed_seconds"].max())
    codebert_part_sum = float(part_frame["elapsed_seconds"].sum())
    embedding_seconds = float(embedding["elapsed_seconds"])
    rows = int(embedding["rows"])
    summary = pd.DataFrame(
        [
            {
                "detector": "hashing",
                "representation": "frozen hashing features",
                "representation_device": "cpu",
                "representation_rows": np.nan,
                "representation_wall_seconds": np.nan,
                "representation_rows_per_second": np.nan,
                "peak_gpu_allocated_bytes": np.nan,
                "peak_gpu_reserved_bytes": np.nan,
                "peak_host_rss_bytes": np.nan,
                **hashing_head,
                "prediction_pipeline_wall_seconds": float(seals["hashing"]["elapsed_seconds"]),
                "prediction_parallel_parts": 1,
                "prediction_aggregate_part_seconds": float(seals["hashing"]["elapsed_seconds"]),
                "target_rows": int(package["target_rows"]),
                "seed_policy": "five independently fitted technical seeds per project",
                "comparison_boundary": "observational runtime on the recorded host; memory was not instrumented for hashing",
            },
            {
                "detector": "codebert",
                "representation": f"{embedding['model']}@{embedding['revision']} mean-pooled embeddings",
                "representation_device": str(embedding["device"]),
                "representation_rows": rows,
                "representation_wall_seconds": embedding_seconds,
                "representation_rows_per_second": rows / embedding_seconds,
                "peak_gpu_allocated_bytes": int(embedding["peak_gpu_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(embedding["peak_gpu_reserved_bytes"]),
                "peak_host_rss_bytes": int(embedding["peak_host_rss_bytes"]),
                **codebert_head,
                "prediction_pipeline_wall_seconds": codebert_part_critical,
                "prediction_parallel_parts": len(part_frame),
                "prediction_aggregate_part_seconds": codebert_part_sum,
                "target_rows": int(package["target_rows"]),
                "seed_policy": f"one deterministic liblinear fit per project, copied to seed addresses {seeds}",
                "comparison_boundary": "four concurrent parts; critical-path and aggregate part times are reported separately and are not a hashing speedup estimate",
            },
        ]
    )

    args.output.mkdir(parents=True)
    observations_path = args.output / "executed_head_fits.csv"
    parts_path = args.output / "codebert_part_runtime.csv"
    summary_path = args.output / "detector_efficiency_summary.csv"
    executed.to_csv(observations_path, index=False)
    part_frame.to_csv(parts_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "protocol_version": "vultriage-extension-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "sealed observational efficiency audit",
        "config_sha256": config_hash,
        "package_summary_sha256": sha256(package_path),
        "prediction_seals": {detector: sha256(path) for detector, path in seal_paths.items()},
        "codebert_embedding_metadata_sha256": sha256(embedding_path),
        "codebert_part_seals": {record["seal_relative_path"]: record["seal_sha256"] for record in part_records},
        "executed_head_fit_rows": len(executed),
        "executed_head_fits_sha256": sha256(observations_path),
        "codebert_part_rows": len(part_frame),
        "codebert_part_runtime_sha256": sha256(parts_path),
        "summary_rows": len(summary),
        "detector_efficiency_summary_sha256": sha256(summary_path),
        "target_vulnerability_labels_accessed": False,
        "claims_supported": [
            "recorded CodeBERT embedding throughput and peak memory",
            "recorded detector head-fit durations under the frozen seed policies",
            "recorded hashing sequential wall time and CodeBERT four-part critical-path/aggregate wall time",
        ],
        "claims_not_supported": [
            "hardware-independent speedup",
            "energy efficiency",
            "direct wall-clock superiority between differently parallelized runs",
        ],
        "script_sha256": sha256(Path(__file__)),
    }
    manifest_path = args.output / "efficiency_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashing_row = summary.loc[summary["detector"] == "hashing"].iloc[0]
    codebert_row = summary.loc[summary["detector"] == "codebert"].iloc[0]
    markdown = [
        "# Extension-v2 Efficiency Audit",
        "",
        "- Analysis type: sealed observational audit; no model rerun and no target-label access.",
        f"- Hashing: {int(hashing_row['executed_head_fits'])} executed heads; median fit {hashing_row['head_fit_seconds_median']:.3f} s (IQR {hashing_row['head_fit_seconds_q25']:.3f}--{hashing_row['head_fit_seconds_q75']:.3f}); sequential prediction-pipeline wall time {hashing_row['prediction_pipeline_wall_seconds']:.3f} s.",
        f"- CodeBERT: {int(codebert_row['executed_head_fits'])} executed deterministic heads; median fit {codebert_row['head_fit_seconds_median']:.3f} s (IQR {codebert_row['head_fit_seconds_q25']:.3f}--{codebert_row['head_fit_seconds_q75']:.3f}).",
        f"- CodeBERT embedding: {rows:,} rows in {embedding_seconds:.3f} s ({rows / embedding_seconds:.2f} rows/s); peak allocated GPU memory {int(embedding['peak_gpu_allocated_bytes']):,} bytes; peak host RSS {int(embedding['peak_host_rss_bytes']):,} bytes.",
        f"- CodeBERT four-part execution: critical path {codebert_part_critical:.3f} s; aggregate part time {codebert_part_sum:.3f} s.",
        "- Interpretation boundary: the hashing job was sequential whereas CodeBERT used four concurrent parts. These wall-clock observations document resource cost but do not establish a hardware-independent speedup ratio.",
        f"- Manifest: `{manifest_path.name}`.",
    ]
    (args.output / "efficiency_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
