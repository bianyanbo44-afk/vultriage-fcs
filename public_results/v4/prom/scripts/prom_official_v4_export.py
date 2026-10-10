"""Build the actual small PROM v4 public aggregate subset and its inventory.

No target/function row data, token caches, checkpoints, private numerical
binaries, original datasets, or full official Git repository are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kbs-validation-v4/prom"
WORK = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT / "public-evidence")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(OUT.resolve()) or destination == OUT.resolve():
        raise ValueError("This delegated exporter writes only beneath the new PROM output directory")
    evaluation = json.loads((OUT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if not evaluation.get("verified_before_target_label_access") or evaluation.get("status") != "completed":
        raise ValueError("Completed sealed evaluation required for public export")
    for record in evaluation["aggregate_files"]:
        if sha(OUT / record["path"]) != record["sha256"]:
            raise ValueError("Aggregate evidence changed before export")
    destination.mkdir(parents=True, exist_ok=True)
    copies = {}
    aggregate_names = ("source_selection.csv", "source_selection_manifest.json", "label_free_action_counts.csv", "computation_receipts.csv",
                       "project_seed_metrics.csv", "project_seed_averages.csv", "policy_project_summary.csv", "paired_project_summary.csv",
                       "evaluation_manifest.json", "provenance.json", "binary_repair.json", "fidelity_smoke.json", "implementation_history.json", "prom_official_v4_report.md",
                       "execution_optimization.json", "literal_equivalence.json",
                       "primary_source_selected_summary.csv", "fixed_native_union_absolute_summary.csv", "primary_matched_paired_endpoints.csv",
                       "action_equivalence.csv", "primary_summary_manifest.json", "prom_official_v4_maintext_tables.md",
                       "portable_patch_roundtrip.json",
                       "decision_seal.json", "decision_seal.sha256")
    for name in aggregate_names:
        copies[Path("evidence") / name] = OUT / name
    for name in ("prom_official_v4.py", "prom_official_v4_setup.py", "prom_official_v4_recompute.py", "prom_official_v4_export.py", "prom_official_v4_summary.py", "prom_official_v4_portable_patches.py"):
        copies[Path("scripts") / name] = ROOT / "scripts" / name
    copies[Path("configs/prom_official_v4.json")] = ROOT / "configs/prom_official_v4.json"
    copies[Path("configs/prom_official_v4_execution.json")] = ROOT / "configs/prom_official_v4_execution.json"
    copies[Path("protocol/prom_official_v4.md")] = ROOT / "protocol/prom_official_v4.md"
    copies[Path("protocol/prom_official_v4_execution_amendment.md")] = ROOT / "protocol/prom_official_v4_execution_amendment.md"
    copies[Path("protocol/prom_official_v4_clearance.json")] = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom_protocol_clearance.json"
    copies[Path("protocol/prom_official_v4_execution_clearance.json")] = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom_execution_optimization_clearance.json"
    copies[Path("third_party/PROM/LICENSE")] = WORK / "official-source/LICENSE"
    copies[Path("third_party/PROM/src/LICENSE.txt")] = WORK / "official-source/src/LICENSE.txt"
    copies[Path("third_party/PROM/prom_official_v4_binary_repair.diff")] = WORK / "prom_official_v4_binary_repair.diff"
    copies[Path("third_party/PROM/prom_official_v4_pooled_count_optimization.diff")] = WORK / "prom_official_v4_pooled_count_optimization.diff"
    for name in ("binary_repair", "pooled_count_optimization"):
        copies[Path("third_party/PROM") / f"prom_official_v4_{name}_portable.diff"] = WORK / f"prom_official_v4_{name}_portable.diff"
    copies[Path("requirements-prom-official-v4.txt")] = WORK / "prom_official_v4_runtime_requirements.txt"
    copies[Path("evidence/aggregate_recomputation_receipt.json")] = OUT / "aggregate-recomputation/recomputation_receipt.json"
    public_qa = OUT / "public-copy-aggregate-qa/recomputation_receipt.json"
    if public_qa.is_file():
        receipt = json.loads(public_qa.read_text(encoding="utf-8"))
        if not receipt.get("passed") or receipt["implementation_sha256"] != sha(ROOT / "scripts/prom_official_v4.py"):
            raise ValueError("Public-copy aggregation QA receipt is stale or failed")
        copies[Path("evidence/public_copy_recomputation_receipt.json")] = public_qa
    # Include actual small source modules, never the Git checkout or benchmarks.
    for path in sorted((WORK / "official-source/src/prom").rglob("*.py")):
        if "tests" not in path.parts and "__pycache__" not in path.parts:
            copies[Path("third_party/PROM/src/prom") / path.relative_to(WORK / "official-source/src/prom")] = path
    for relative, source in copies.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    readme = """# Executed official PROM component evidence

The primary result is actual pinned official conformal-component execution with
an explicit binary one-hot repair, probability-space neighborhoods and source
alpha/expert selection. Prior TRACE outcomes were already known. This package
does not reproduce the complete official CWE case study or TRACE incremental
neural retraining. See evidence/prom_official_v4_report.md and the dated protocol.

PROM source is from Huanting Wang, Patrick Lenihan and Zheng Wang, *Enhancing
Deployment-Time Predictive Model Robustness for Code Analysis and Optimization*,
CGO 2025, DOI 10.1145/3696443.3708959, https://github.com/HuantWang/PROM,
commit f16b52772123064551b6450f10b308971c4cdd39. Original source and both original
license notices are retained under third_party/PROM; the separately recorded
two-line binary repair is not silently applied to that source snapshot.
The separate execution amendment sorts the complete duplicated global pool for
the exact same >= counts/denominator. Exact native/binary and edge-case tests,
the real source alpha grid, and all 43 direct target cells match before outcomes.
Both original source and the distinct repair/optimization diffs are retained.
Use the *_portable.diff files for LF patch review. The sealed archival diffs
retain their original Windows doubled-carriage-return text. The included
portable_patch_roundtrip receipt independently applies the clean patch hunks
and reconstructs the actual executed runtime bytes exactly, without changing
the original source, archival diff, decision seal or result.

## Recompute project summaries from the released aggregates

Install numpy==1.26.4 in your own environment, then run from this package root:

```
python scripts/prom_official_v4_recompute.py --input evidence/project_seed_metrics.csv --config configs/prom_official_v4.json --implementation scripts/prom_official_v4.py --expected evidence --output verification
```

The program loads only the executed aggregation function bodies and checks exact
CSV byte parity for project/fit averages, policy medians and paired 10,000-draw
project-bootstrap summaries. It needs no raw labels or predictions and performs
no model inference. Empty selective-error fields remain undefined.
The included receipts record byte-identical reconstruction in the original
NumPy 1.26.4 stack and from the materialized public copy under NumPy 2.2.6.

The source-selection and action-count files contain aggregates per source
configuration or target project/fit, not individual functions. Raw datasets,
source/target rows, target-label vaults, token arrays, model weights, individual
predictions/actions, smoke row arrays, private dependency binaries and the full
official Git repository are deliberately excluded. Public decision-seal hashes
bind those omitted author-side inputs, but cannot by themselves replay raw
decisions. Full raw replay requires independently obtained hash-matched inputs
and the existing TRACE code/workspace layout.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    files = [{"path": str(path.relative_to(destination)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "public_export_manifest.json"]
    forbidden = (".npz", ".npy", ".pt", ".pkl", ".bin", ".jsonl")
    if any(Path(row["path"]).suffix in forbidden or "/.git/" in row["path"] or "runtime-dependencies" in row["path"] for row in files):
        raise ValueError("Public export contains forbidden raw/binary artifacts")
    manifest = {"schema": "trace-prom-official-public-export-v4.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "included_files": files, "included_file_count": len(files), "included_bytes": sum(r["bytes"] for r in files),
                "evaluation_manifest_sha256": sha(OUT / "evaluation_manifest.json"),
                "raw_row_data_exported": False, "full_official_repository_exported": False,
                "scope": "Actual materialized aggregate/reproduction/source/license subset; hashes listed for every included file except this manifest itself"}
    (destination / "public_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "included_file_count": len(files), "included_bytes": manifest["included_bytes"], "raw_row_data_exported": False}, indent=2))


if __name__ == "__main__":
    main()
