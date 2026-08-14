"""Merge independently sealed, label-free prediction parts."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from vultriage.data import load_config, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", required=True, choices=("hashing", "codebert"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--package-summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_hash = sha256(args.config)
    source_hash = sha256(args.source_metadata)
    target_hash = sha256(args.target_metadata)
    package_hash = sha256(args.package_summary)
    package_summary = json.loads(args.package_summary.read_text(encoding="utf-8"))
    expected_groups = list(package_summary["selected_project_groups"])
    runner_hash = sha256(Path(__file__).with_name("run_extension_predict.py"))
    seen: set[str] = set()
    inventories: dict[str, str] = {}
    seed_reuse_modes: set[str] = set()
    seed_reuse_references: set[int | None] = set()
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    (args.output / "predictions").mkdir()
    for part in args.parts:
        seal = json.loads((part / "prediction_seal.json").read_text(encoding="utf-8"))
        if seal.get("detector") != args.detector:
            raise RuntimeError(f"detector mismatch in part {part}")
        if seal.get("config_sha256") != config_hash or seal.get("source_metadata_sha256") != source_hash or seal.get("target_metadata_sha256") != target_hash or seal.get("source_package_summary_sha256") != package_hash:
            raise RuntimeError(f"frozen input hash mismatch in part {part}")
        if seal.get("runner_sha256") != runner_hash:
            raise RuntimeError(f"runner hash mismatch in part {part}")
        if seal.get("target_label_vault_argument_present") is not False or seal.get("target_vulnerability_labels_accessed") is not False:
            raise RuntimeError(f"part admits target-label access: {part}")
        seed_reuse_modes.add(str(seal.get("seed_reuse_mode", "independent")))
        seed_reuse_references.add(seal.get("seed_reuse_reference"))
        groups = list(seal.get("selected_project_groups", []))
        if not groups or seen.intersection(groups):
            raise RuntimeError(f"duplicate or missing group declaration in part {part}")
        seen.update(groups)
        for relative, expected_hash in seal.get("prediction_files", {}).items():
            if not relative.startswith("predictions/"):
                raise RuntimeError(f"unexpected prediction path {relative}")
            source = part / relative
            if not source.is_file() or sha256(source) != expected_hash:
                raise RuntimeError(f"prediction hash mismatch in part {part}: {relative}")
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"duplicate prediction file {relative}")
            shutil.copy2(source, destination)
            inventories[relative] = expected_hash
    if seen != set(expected_groups):
        raise RuntimeError(f"merged groups differ from frozen package: {sorted(seen)}")
    if len(seed_reuse_modes) != 1 or len(seed_reuse_references) != 1:
        raise RuntimeError("merged parts disagree on seed-reuse provenance")
    seed_reuse_mode = next(iter(seed_reuse_modes))
    seed_reuse_reference = next(iter(seed_reuse_references))
    environment = {
        "detector": args.detector,
        "merged_parts": [str(part) for part in args.parts],
        "selected_project_groups": expected_groups,
        "config_sha256": config_hash,
        "source_metadata_sha256": source_hash,
        "target_metadata_sha256": target_hash,
        "source_package_summary_sha256": package_hash,
        "target_label_vault_argument_present": False,
        "target_labels_accessed": False,
        "seed_reuse_mode": seed_reuse_mode,
        "seed_reuse_reference": seed_reuse_reference,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = {
        "experiment_id": args.output.name,
        "detector": args.detector,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": config_hash,
        "source_metadata_sha256": source_hash,
        "target_metadata_sha256": target_hash,
        "source_package_summary_sha256": package_hash,
        "runner_sha256": runner_hash,
        "prediction_files": dict(sorted(inventories.items())),
        "target_label_vault_argument_present": False,
        "target_vulnerability_labels_accessed": False,
        "seed_reuse_mode": seed_reuse_mode,
        "seed_reuse_reference": seed_reuse_reference,
        "selected_project_groups": expected_groups,
        "merged_parts": [str(part) for part in args.parts],
    }
    (args.output / "prediction_seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"detector": args.detector, "groups": len(seen), "prediction_files": len(inventories), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
