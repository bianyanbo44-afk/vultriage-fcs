"""Expose all primary PROM tables and audit sealed three-action equality.

Reads the completed component aggregates and sealed decisions. Does not select
a new expert, alpha, model, project subset, control anchor, or target outcome.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom/runtime-dependencies"
if PRIVATE.is_dir():
    sys.path.insert(0, str(PRIVATE))
import numpy as np

from prom_official_v4_setup import sha, utc, write_json

OUT = ROOT / "outputs/kbs-validation-v4/prom"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    seal_path = OUT / "decision_seal.json"
    if sha(seal_path) != (OUT / "decision_seal.sha256").read_text(encoding="ascii").strip():
        raise ValueError("PROM seal checksum mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    evaluation = json.loads((OUT / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if evaluation["new_decision_seal_sha256"] != sha(seal_path):
        raise ValueError("Completed evaluation does not bind the current seal")
    for record in evaluation["aggregate_files"]:
        if sha(OUT / record["path"]) != record["sha256"]:
            raise ValueError("Aggregate evidence changed")
    equality = []
    for cell in seal["decisions"]:
        path = OUT / cell["archive"]
        if sha(path) != cell["archive_sha256"]:
            raise ValueError("Action archive changed")
        with np.load(path, allow_pickle=False) as a:
            original = a["action__official_prom_source_selected"]
            equality.append({**{k: cell[k] for k in ("dataset", "detector", "seed", "project")}, "n": len(original),
                             "selected_family": cell["source_selected_family"], "selected_alpha": cell["source_selected_alpha"],
                             "exact_same_three_actions_as_msp": bool(np.array_equal(original, a["action__msp_matched_prom"])),
                             "exact_same_three_actions_as_allocation": bool(np.array_equal(original, a["action__allocation_matched_prom"])),
                             "target_labels_used_for_equality_check": False})
    write_table("action_equivalence.csv", equality)
    policy = read("policy_project_summary.csv")
    fields = ("dataset", "detector", "method", "project_count", "weighted_domain_supported_projects", "median_review_fraction", "median_coverage",
              "median_auto_fn_rate", "median_auto_fp_rate", "median_error_detection_f1", "median_selective_error", "selective_error_defined_projects", "median_cost_10", "median_cost_50")
    primary = [{k: row[k] for k in fields} for row in policy if row["method"] == "official_prom_source_selected"]
    fixed_union = [{k: row[k] for k in fields} for row in policy if row["method"] == "official_prom_fixed_mixture"]
    if len(primary) != 6 or len(fixed_union) != 6:
        raise ValueError("All six detector/collection cells must be exposed")
    write_table("primary_source_selected_summary.csv", primary)
    write_table("fixed_native_union_absolute_summary.csv", fixed_union)
    paired = [row for row in read("paired_project_summary.csv") if row["candidate"] == "official_prom_source_selected"
              and row["baseline"] in ("msp_matched_prom", "allocation_matched_prom")]
    write_table("primary_matched_paired_endpoints.csv", paired)
    lines = ["# PROM component tables for the manuscript", "",
             "The fixed primary comparison uses source-selected LAC/APS/RAPS or their conservative union. Every actual fit selected LAC at alpha=0.50 using source validation F1 only. No target-best-F1 choice is reported. Values are project medians after actual-seed averaging. Error-detection F1 treats review as an error flag and forced-argmax misprediction as the positive event; it is not vulnerable-class classification F1.", "",
             "| Collection | Detector | Projects | Review | Automatic miss rate | False-alert rate | Error-detection F1 | Cost 10 | Cost 50 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary:
        lines.append(f"| {row['dataset']} | {row['detector']} | {row['project_count']} | {float(row['median_review_fraction']):.4f} | {float(row['median_auto_fn_rate']):.4f} | {float(row['median_auto_fp_rate']):.4f} | {float(row['median_error_detection_f1']):.4f} | {float(row['median_cost_10']):.4f} | {float(row['median_cost_50']):.4f} |")
    n_msp = sum(r["exact_same_three_actions_as_msp"] for r in equality)
    n_allocation = sum(r["exact_same_three_actions_as_allocation"] for r in equality)
    lines += ["", f"Exact sealed three-action equality with MSP holds in {n_msp}/{len(equality)} project/fit cells; equality with allocation-matched tails holds in {n_allocation}/{len(equality)}. This is an action-array observation in the complete executed grid. Zero paired intervals do not establish statistical equivalence or full-system superiority.", "",
              "The protocol also retains the fixed-alpha=0.10 native-case-study LAC/APS/RAPS conservative union as an absolute-endpoint lane. It does not replace source selection or receive a newly chosen control anchor.", "",
              "| Collection | Detector | Projects | Review | Automatic miss rate | False-alert rate | Error-detection F1 | Cost 10 | Cost 50 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in fixed_union:
        lines.append(f"| {row['dataset']} | {row['detector']} | {row['project_count']} | {float(row['median_review_fraction']):.4f} | {float(row['median_auto_fn_rate']):.4f} | {float(row['median_auto_fp_rate']):.4f} | {float(row['median_error_detection_f1']):.4f} | {float(row['median_cost_10']):.4f} | {float(row['median_cost_50']):.4f} |")
    lines += ["", "Top-k and the four-expert union remain secondary in the complete aggregate tables. Costs assume correct review and are illustrative; the saved training fits, representation extraction, official component selection/sealing and numerical verification have different runtime scopes. The synthetic eight-class GaussianNB feedback API demonstration is software evidence; the original official eight-CWE benchmark and TRACE neural incremental retraining remain unperformed.", ""]
    (OUT / "prom_official_v4_maintext_tables.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [OUT / name for name in ("action_equivalence.csv", "primary_source_selected_summary.csv", "fixed_native_union_absolute_summary.csv", "primary_matched_paired_endpoints.csv", "prom_official_v4_maintext_tables.md")]
    result = {"schema": "trace-prom-complete-primary-summary-v4.2", "created_at_utc": utc(), "all_six_detector_collection_cells_present": True,
              "new_expert_alpha_model_or_subset_selected": False, "target_vault_opened": False, "prior_outcomes_known": True,
              "action_equality_check_uses_target_labels": False, "project_fit_cells": len(equality), "msp_exact_equal_cells": n_msp,
              "allocation_exact_equal_cells": n_allocation, "paired_endpoint_rows": len(paired),
              "decision_seal_sha256": sha(seal_path), "evaluation_manifest_sha256": sha(OUT / "evaluation_manifest.json"),
              "artifacts": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in paths]}
    write_json(OUT / "primary_summary_manifest.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
