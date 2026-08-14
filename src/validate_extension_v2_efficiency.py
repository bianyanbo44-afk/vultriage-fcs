"""Independently validate the sealed extension-v2 efficiency audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from vultriage.data import load_config, sha256


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"efficiency validation output already exists: {args.output}")

    config = load_config(args.config)
    config_hash = sha256(args.config)
    seeds = [int(value) for value in config["detectors"]["hashing_sgd"]["seeds"]]
    package_path = args.root / "source-v2" / "package_summary.json"
    package = read_json(package_path)
    groups = list(package["selected_project_groups"])
    require(len(groups) == 24 and len(seeds) == 5, "frozen project/seed dimensions are incomplete")

    manifest_path = args.efficiency / "efficiency_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("config_sha256") == config_hash, "efficiency/config hash mismatch")
    require(manifest.get("analysis_type") == "sealed observational efficiency audit", "efficiency analysis type is not frozen")
    require(manifest.get("target_vulnerability_labels_accessed") is False, "efficiency audit admits target-label access")
    verify_hash(package_path, manifest["package_summary_sha256"], "package summary")

    source_seals = {
        "hashing": args.root / "predictions" / "hashing-v8" / "prediction_seal.json",
        "codebert": args.root / "predictions" / "codebert-v2" / "prediction_seal.json",
    }
    for detector, path in source_seals.items():
        verify_hash(path, manifest["prediction_seals"][detector], f"{detector} prediction seal")
        seal = read_json(path)
        require(seal.get("target_vulnerability_labels_accessed") is False, f"{detector} seal admits target-label access")

    embedding_path = args.root / "codebert-v1" / "embeddings-v1" / "metadata.json"
    verify_hash(embedding_path, manifest["codebert_embedding_metadata_sha256"], "CodeBERT embedding metadata")
    embedding = read_json(embedding_path)
    require(embedding.get("labels_used") is False, "CodeBERT embedding metadata reports label use")
    require(embedding.get("revision") == EXPECTED_CODEBERT_REVISION, "CodeBERT resolved revision mismatch")

    for relative, expected in manifest["codebert_part_seals"].items():
        path = args.root / Path(relative)
        verify_hash(path, expected, "CodeBERT part seal")
        part = read_json(path)
        require(part.get("target_vulnerability_labels_accessed") is False, f"CodeBERT part admits target-label access: {path}")

    fit_path = args.efficiency / "executed_head_fits.csv"
    part_path = args.efficiency / "codebert_part_runtime.csv"
    summary_path = args.efficiency / "detector_efficiency_summary.csv"
    verify_hash(fit_path, manifest["executed_head_fits_sha256"], "executed head-fit table")
    verify_hash(part_path, manifest["codebert_part_runtime_sha256"], "CodeBERT part runtime table")
    verify_hash(summary_path, manifest["detector_efficiency_summary_sha256"], "detector efficiency summary")
    fits = pd.read_csv(fit_path)
    parts = pd.read_csv(part_path)
    summary = pd.read_csv(summary_path)
    require(len(fits) == len(groups) * (len(seeds) + 1) == int(manifest["executed_head_fit_rows"]), "executed fit row count mismatch")
    require(len(parts) == 4 == int(manifest["codebert_part_rows"]), "CodeBERT part row count mismatch")
    require(len(summary) == 2 == int(manifest["summary_rows"]), "efficiency summary row count mismatch")
    require(set(fits["detector"]) == {"hashing", "codebert"}, "executed fit detector set mismatch")
    require((fits["detector"] == "hashing").sum() == len(groups) * len(seeds), "hashing executed-fit count mismatch")
    require((fits["detector"] == "codebert").sum() == len(groups), "CodeBERT executed-fit count mismatch")
    require(set(summary["detector"]) == {"hashing", "codebert"}, "efficiency summary detector set mismatch")
    require((pd.to_numeric(fits["head_fit_seconds"]) > 0).all(), "non-positive head-fit duration")
    require((pd.to_numeric(parts["elapsed_seconds"]) > 0).all(), "non-positive CodeBERT part duration")
    require((pd.to_numeric(summary["prediction_pipeline_wall_seconds"]) > 0).all(), "non-positive pipeline duration")
    require((pd.to_numeric(summary["target_rows"]) == int(package["target_rows"])).all(), "target row count mismatch")
    codebert = summary.loc[summary["detector"] == "codebert"].iloc[0]
    require(int(codebert["prediction_parallel_parts"]) == 4, "CodeBERT parallel-part count mismatch")
    require(abs(float(codebert["prediction_pipeline_wall_seconds"]) - float(parts["elapsed_seconds"].max())) < 1e-9, "CodeBERT critical-path time mismatch")
    require(abs(float(codebert["prediction_aggregate_part_seconds"]) - float(parts["elapsed_seconds"].sum())) < 1e-9, "CodeBERT aggregate part time mismatch")
    require("not a hashing speedup estimate" in str(codebert["comparison_boundary"]), "wall-clock comparison boundary is missing")
    unsupported = set(manifest.get("claims_not_supported", []))
    require("hardware-independent speedup" in unsupported and "energy efficiency" in unsupported, "unsupported efficiency claims are not explicit")

    result = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "projects": len(groups),
        "seeds": seeds,
        "executed_head_fit_rows": len(fits),
        "codebert_parallel_parts": len(parts),
        "summary_rows": len(summary),
        "efficiency_manifest_sha256": sha256(manifest_path),
        "validator_sha256": sha256(Path(__file__)),
        "target_vulnerability_labels_accessed": False,
    }
    args.output.mkdir(parents=True)
    json_path = args.output / "efficiency_validation.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Extension-v2 Efficiency Validation",
        "",
        "- Status: **PASS**",
        f"- Projects/seeds: {len(groups)} / {len(seeds)}",
        f"- Executed head-fit observations: {len(fits)}",
        f"- CodeBERT parallel-part records: {len(parts)}",
        "- Target-label access: false",
        "- Interpretation boundary: runtime is observational and hardware-specific; no cross-parallelization speedup claim is validated.",
        f"- Full JSON record: `{json_path.name}`.",
    ]
    (args.output / "efficiency_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
