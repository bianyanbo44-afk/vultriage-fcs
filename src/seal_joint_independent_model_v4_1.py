"""Bind both complete dataset decision seals before either target-label evaluation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_independent_model_v4_1 import verify_seal
from vultriage.data import sha256

ROOT = Path(__file__).resolve().parents[1]


def seal_joint(config: Path, output: Path) -> dict:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if cfg["protocol_version"] != "trace-independent-model-v4.1":
        raise ValueError("Expected the corrected v4.1 amendment")
    out = ROOT / cfg["paths"]["output_root"]
    expected = out / "joint_dataset_decision_seal.json"
    if output.resolve() != expected.resolve():
        raise ValueError("Joint dataset seal must have its protocol-derived lineage path")
    if output.exists():
        raise FileExistsError("Preserve an existing joint seal")
    records = {}
    actual = None
    for dataset in ("diversevul", "megavul"):
        path = out / "decisions" / dataset / "decision_seal.json"
        manifest = verify_seal(path)
        if (manifest["dataset"] != dataset or sha256(Path(manifest["config_path"])) != sha256(config)
                or manifest["all_planned_fits_present"] is not True
                or manifest["all_planned_attempts_accounted"] is not True
                or manifest["projects"] != cfg["target_rosters"][dataset]["projects"]
                or manifest["target_rows"] != cfg["target_rosters"][dataset]["rows"]):
            raise ValueError("Joint sealing requires both full rosters and all three genuine fit predictions")
        if actual is None:
            actual = manifest["actual_seeds"]
        elif actual != manifest["actual_seeds"]:
            raise ValueError("Both target datasets must use the identical genuine source fits")
        records[dataset] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                            "projects": manifest["projects"], "target_rows": manifest["target_rows"],
                            "prepared_vault_sha256": manifest["prepared_target_label_vault_sha256"],
                            "domain_failure_cells": manifest["domain_failure_cells"]}
    result = {"schema_version": "trace-independent-model-v4.1-joint-dataset-seal-1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(config),
              "protocol_sha256": cfg["protocol_sha256"], "actual_seeds": actual,
              "all_complete_target_rosters_sealed_before_evaluation": True,
              "target_label_vaults_opened_for_this_seal": False, "known_original_and_v3_outcomes": True,
              "datasets": records, "script_sha256": sha256(Path(__file__))}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".sha256").write_text(sha256(output) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / cfg["paths"]["output_root"] / "joint_dataset_decision_seal.json"
    result = seal_joint(args.config, output)
    print(json.dumps({"actual_seeds": result["actual_seeds"], "datasets": {d: {"rows": r["target_rows"], "projects": len(r["projects"])} for d, r in result["datasets"].items()}, "joint_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
