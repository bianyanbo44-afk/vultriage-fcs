"""Export aggregate-only capacity/feedback evidence without changing its seals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vultriage.data import sha256

TMP = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/capacity_feedback"
PRIVATE_COLUMNS = {"archive", "capacity_key", "feedback_key"}
FORBIDDEN_COLUMNS = {"row_id", "index", "position", "target_position", "target_positions", "source_position", "source_positions",
    "source_probability", "target_probability", "arrival_eta", "service_batch", "prediction_batch", "label", "labels", "target_indices", "fit_window_indices", "commit_id", "commit_ids"}
CSV_LEVELS = {
    "candidate_roster_lineage.csv": "inherited_candidate_project",
    "ordered_batch_summary.csv": "project_seed_batch_summary",
    "capacity_project_seed_metrics.csv": "project_seed_policy",
    "capacity_project_metrics.csv": "project_policy_actual_seeds_averaged",
    "capacity_paired_project_differences.csv": "paired_project",
    "capacity_paired_summary.csv": "paired_project_bootstrap_summary",
    "capacity_curve_plot_data.csv": "full_roster_policy_plot_summary",
    "feedback_project_seed_metrics.csv": "project_seed_policy",
    "feedback_project_metrics.csv": "project_policy_actual_seeds_averaged",
    "feedback_paired_project_differences.csv": "paired_project",
    "feedback_paired_summary.csv": "paired_project_bootstrap_summary",
    "feedback_curve_plot_data.csv": "full_roster_policy_plot_summary"
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def csv_aggregate(source: Path, destination: Path) -> dict:
    """Preserve every data token; remove only private archive/run-key columns."""
    with source.open(encoding="utf-8", newline="") as incoming, destination.open("x", encoding="utf-8", newline="") as outgoing:
        reader = csv.DictReader(incoming)
        need(not (set(reader.fieldnames) & FORBIDDEN_COLUMNS), "Per-function columns cannot be exported")
        fields = [name for name in reader.fieldnames if name not in PRIVATE_COLUMNS]
        writer = csv.DictWriter(outgoing, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        count = 0
        for row in reader:
            writer.writerow({name: row[name] for name in fields})
            count += 1
    return {"rows": count, "columns": fields, "dropped_private_reference_columns": sorted(set(reader.fieldnames) & PRIVATE_COLUMNS), "numeric_tokens_preserved": True}


def operational_summary(analysis: Path) -> dict:
    capacity = pd.read_csv(analysis / "capacity_project_metrics.csv")
    primary = capacity[(capacity.capacity_fraction == .2) & (capacity.miss_cost == 10) & (capacity.review_cost == .1) &
                       (capacity.review_fn_probability == 0) & (capacity.method == "trace_mandatory")]
    trace = []
    for (dataset, detector), group in primary.groupby(["dataset", "detector"], sort=True):
        trace.append({"dataset": dataset, "detector": detector, "projects_roster": len(group),
            "projects_feasible_all_actual_seeds": int(group.feasible_complete_roster.sum()),
            "median_unresolved_fraction": float(group.unresolved_fraction.median()),
            "median_required_review_fraction": float((group.required_review / group.n).median())})
    feedback = pd.read_csv(analysis / "feedback_project_metrics.csv")
    primary = feedback[(feedback.slots_per_batch == 16) & (feedback.delay_batches == 3) & (feedback.miss_cost == 10) &
                       (feedback.method == "source_iso_rolling")]
    history = []
    for (dataset, detector), group in primary.groupby(["dataset", "detector"], sort=True):
        history.append({"dataset": dataset, "detector": detector, "projects_roster": len(group),
            "projects_with_any_actual_seed_update": int((group.rolling_updates > 0).sum()),
            "median_batches": float(group.batches.median()), "median_arrived_reviewed_labels": float(group.arrived_reviewed_labels.median()),
            "median_rolling_updates_after_seed_averaging": float(group.rolling_updates.median())})
    return {"capacity_setting": {"fraction": .2, "miss_cost": 10, "review_cost": .1, "review_error": 0},
            "trace_capacity_feasibility": trace, "feedback_setting": {"slots_per_batch": 16, "delay_batches": 3, "miss_cost": 10, "review_cost": .1},
            "source_cost_feedback_history": history, "all_projects_retained": True}


def export(analysis: Path, output: Path) -> dict:
    need(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Public export must be new/empty")
    original_manifest_path = analysis / "analysis_manifest.json"
    original_hash = sha256(original_manifest_path)
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    need(original["status"] == "COMPLETE" and original_hash == "C8D2FE9E97918776570CA3E0C3A554061A742D7950FD0369985299525805611E", "Original completed comparison changed")
    expected = {record["path"]: record["sha256"] for record in original["artifacts"]}
    verification = json.loads((TMP / "postrun_verification.json").read_text(encoding="utf-8"))
    need(verification["status"] == "PASS" and verification["manifest_sha256"] == original_hash, "Postrun verification does not bind this analysis")
    output.mkdir(parents=True, exist_ok=True)
    schemas = {}
    for name, level in CSV_LEVELS.items():
        need(sha256(analysis / name) == expected[name], "Aggregate result changed after sealing")
        schemas[name] = {"level": level, **csv_aggregate(analysis / name, output / name)}
    copies = [(ROOT / "configs/capacity_feedback_v4.json", "config_capacity_feedback_v4.json"),
              (ROOT / "protocol/capacity_feedback_v4.md", "protocol_capacity_feedback_v4.md"),
              (analysis / "integrity_report.json", "integrity_report.json"),
              (analysis / "concise_results.json", "concise_results.json"),
              (TMP / "postrun_verification.json", "postrun_verification.json"),
              (TMP / "unit_test_receipt.json", "unit_test_receipt.json")]
    for source, name in copies:
        if source.parent == analysis:
            need(sha256(source) == expected[source.name], "Original JSON result changed after seal")
        shutil.copyfile(source, output / name)
    # Metadata-only timestamp availability evidence; no function, label or code is copied.
    raw = json.loads((TMP / "raw_timestamp_schema_audit.json").read_text(encoding="utf-8"))
    dump(output / "timestamp_field_summary.json", {"source_chronology_available": False, "diversevul_chronology_available": False,
        "raw_schema_audit_status": raw["status"], "raw_schema_audit_sha256": sha256(TMP / "raw_timestamp_schema_audit.json"),
        "releases": [{"release_file": Path(row["path"]).name, "records": row["records"], "date_or_time_field_record_counts": row["date_or_time_field_record_counts"],
                      "sha256": row["sha256"]} for row in raw["raw_sources"]],
        "megavul_prepared_timestamp_fields": ["commit_date", "date"], "megavul_ordering": "recorded commit-author dates; analyst decisions and feedback arrivals simulated"})
    dump(output / "primary_operational_summary.json", operational_summary(analysis))
    paragraph = ("With review limited to 20% of each project's functions, mandatory TRACE leaves unresolved work in all six MegaVul projects and most of the 38 DiverseVul projects; the capped source-cost and MSP policies complete every roster. "
        "Policy rankings depend on capacity: for fine-tuned CodeBERT on DiverseVul, source-cost selection minus MSP has median cost difference -0.0569 at 20% capacity but +0.0275 when the full roster is reviewable, while the MegaVul difference at 20% is +0.00926. "
        "Delayed reviewed-label updates provide no consistent improvement. At 16 slots and a three-batch delay, the median DiverseVul project has only three batches and no feedback arrives during its run; only 11–12 of 38 projects produce a source-calibration update for any seed. MegaVul offers longer histories but mixed cost changes. "
        "These are known-outcome benchmark simulations under miss10, false-alert1, review0.1 and correct-review assumptions in illustrative cost units; MegaVul follows recorded author dates and DiverseVul a synthetic order, with no dated source-before-target validation.")
    (output / "scientific_interpretation.md").write_text(paragraph + "\n", encoding="utf-8")
    (output / "README.md").write_text(paragraph + "\n\nAll fixed cost/capacity/delay settings, project and seed aggregates, paired differences and plot summaries are included. Per-function actions, positions, labels, probabilities, label-event ledgers and calibration arrays are excluded. The public manifest binds the complete private analysis and verification receipts.\n", encoding="utf-8")
    input_manifest = json.loads((analysis / "input_manifest.json").read_text(encoding="utf-8"))
    source_path = ROOT / "outputs/neural-validation-v3/inputs/source_rows.jsonl"
    freeze = json.loads((analysis / "frozen_protocol_receipt.json").read_text(encoding="utf-8"))
    code_paths = [ROOT / "src/capacity_feedback_v4.py", ROOT / "scripts/test_capacity_feedback_v4.py", ROOT / "scripts/check_capacity_feedback_v4.py", Path(__file__)]
    for record in freeze["files"]:
        need(sha256(Path(record["path"])) == record["sha256"], "Frozen config/implementation changed")
    need(sha256(code_paths[2]) == verification["auditor_script_sha256"], "Checker changed after verification")
    config = json.loads((ROOT / "configs/capacity_feedback_v4.json").read_text(encoding="utf-8"))
    detector_dirs = {"finetuned_codebert": "finetuned", "frozen_codebert": "frozen", "hashing_sgd": "hashing"}
    predictions, old_seals = [], []
    for dataset, spec in config["datasets"].items():
        seal_path = ROOT / spec["decision_seal_path"]
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        old_seals.append({"dataset": dataset, "path": relative(seal_path), "sha256": sha256(seal_path)})
        for detector, seeds in seal["actual_seeds"].items():
            for seed in seeds:
                metadata_path = ROOT / f"outputs/neural-validation-v3/predictions/{dataset}/{detector_dirs[detector]}/seed-{seed}/metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                predictions.append({"dataset": dataset, "detector": detector, "seed": seed,
                    "prediction_metadata_path": relative(metadata_path), "prediction_metadata_sha256": sha256(metadata_path),
                    **{key: metadata[key] for key in ("prediction_sha256", "source_rows_sha256", "target_rows_sha256", "config_sha256", "training_manifest_sha256", "checkpoint_sha256", "predictor_script_sha256")},
                    "source_token_manifest_sha256": metadata.get("source_token_manifest_sha256")})
    dump(output / "evidence_bindings.json", {"original_analysis_manifest_sha256": original_hash,
        "original_input_manifest_sha256": sha256(analysis / "input_manifest.json"),
        "original_capacity_decision_seal_sha256": sha256(analysis / "capacity_decision_seal.json"),
        "original_feedback_decision_seal_sha256": sha256(analysis / "feedback_decision_seal.json"),
        "postrun_verification_sha256": sha256(TMP / "postrun_verification.json"), "freeze_receipt_sha256": original["freeze_receipt_sha256"],
        "source_metadata": {"path": relative(source_path), "sha256": sha256(source_path), "exact_calibration_role": "calibration", "calibration_rows": 4400,
                            "source_role_counts": input_manifest["source_role_counts"]},
        "config_sha256": original["config_sha256"], "protocol_sha256": original["protocol_sha256"],
        "code": [{"path": relative(path), "sha256": sha256(path)} for path in code_paths],
        "original_decision_seals": old_seals, "original_prediction_bindings": predictions,
        "raw_timestamp_audit_sha256": sha256(TMP / "raw_timestamp_schema_audit.json"),
        "no_per_function_data_exported": True, "no_new_statistical_choices": True})
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "schema": schemas.get(path.name)}
             for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {"schema_version": "capacity-feedback-v4-public-aggregate-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_AGGREGATE_ONLY", "original_analysis_manifest_sha256": original_hash, "known_outcome_amendment": True,
        "settings_preserved": "all 8 capacity cost scenarios x 7 project capacities x 7 methods; all 2 feedback miss costs x 3 fixed slot caps x 3 delays x 6 methods",
        "statistical_unit": "project after actual seed averaging; datasets separate", "complete_rosters": {"diversevul": 38, "megavul": 6},
        "source_calibration_rows": 4400, "files": files,
        "excluded": ["per-function action/probability arrays", "batch-position membership", "reviewed-label event and fit-window ledgers", "label vaults", "source/target function code"],
        "export_helper_sha256": sha256(Path(__file__)), "no_original_artifacts_modified": True, "no_new_statistical_choices": True}
    dump(output / "capacity_feedback_v4_public_manifest.json", manifest)
    (output / "capacity_feedback_v4_public_manifest.sha256").write_text(sha256(output / "capacity_feedback_v4_public_manifest.json") + "\n", encoding="ascii")
    need(sha256(original_manifest_path) == original_hash, "Original analysis manifest was modified")
    return {"status": manifest["status"], "output": str(output), "files": len(files),
            "public_manifest_sha256": sha256(output / "capacity_feedback_v4_public_manifest.json"), "original_analysis_manifest_sha256": original_hash,
            "csv_schemas": {name: {"level": record["level"], "rows": record["rows"], "columns": len(record["columns"])} for name, record in schemas.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/kbs-validation-v4/capacity_feedback")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/capacity_feedback/public-aggregate")
    args = parser.parse_args()
    print(json.dumps(export(args.analysis.resolve(), args.output.resolve()), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
