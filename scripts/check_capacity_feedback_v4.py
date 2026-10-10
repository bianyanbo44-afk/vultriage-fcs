"""Cross-link v4 action/service archives to every released label and fit window.

This audit computes no new scientific comparison. It independently reconstructs
the label-only chronology from the immutable result files, checking that ledger
claims correspond to actually reviewed past rows and the fixed rolling window.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from capacity_feedback_v4 import REVIEW, BACKLOG, json_hash, require, source_calibration_view, read_rows, verify_output
from vultriage.data import sha256
from vultriage.neural_audit import cell_key


def key_for(values):
    return str(values[0]), str(values[1]), int(values[2]), str(values[3]), str(values[4])


def audit(output: Path) -> dict:
    common = verify_output(output)
    config = json.loads((ROOT / "configs/capacity_feedback_v4.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    require(sha256(ROOT / "configs/capacity_feedback_v4.json") == manifest["config_sha256"], "Config changed after analysis")
    events = pd.read_csv(output / "feedback_label_events.csv.gz")
    fits = pd.read_csv(output / "feedback_fit_ledger.csv.gz")
    run_plan = pd.read_csv(output / "feedback_decision_counts.csv")
    group_keys = ["dataset", "detector", "seed", "project", "feedback_key"]
    event_groups = {key_for(k): g for k, g in events.groupby(group_keys, sort=False)}
    fit_groups = {key_for(k): g for k, g in fits.groupby(group_keys, sort=False)} if len(fits) else {}
    source = read_rows(ROOT / config["source_rows_path"], source=True)
    targets, labels, original = {}, {}, {}
    for dataset, spec in config["datasets"].items():
        input_dir = ROOT / spec["input_dir"]
        targets[dataset] = read_rows(input_dir / "target_rows.jsonl", source=False)
        with np.load(input_dir / "target_label_vault.npz", allow_pickle=False) as vault:
            positions = vault["positions"]
            order = np.argsort(positions)
            require(np.array_equal(positions[order], np.arange(len(targets[dataset]))), "Auditor vault positions changed")
            labels[dataset] = vault["labels"][order]
        seal_path = ROOT / spec["decision_seal_path"]
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        original[dataset] = { (e["detector"], int(e["seed"]), e["project"]): seal_path.parent / e["archive"] for e in seal["decisions"] }
    detectors = {"finetuned_codebert": "finetuned", "frozen_codebert": "frozen", "hashing_sgd": "hashing"}
    source_cache = {}
    fits_checked, releases_checked, archive_runs, pending_checked = 0, 0, 0, 0
    for (dataset, detector, seed, project), cell in run_plan.groupby(["dataset", "detector", "seed", "project"], sort=True):
        seed = int(seed)
        model_id = dataset, detector, seed
        if model_id not in source_cache:
            metadata_path = ROOT / f"outputs/neural-validation-v3/predictions/{dataset}/{detectors[detector]}/seed-{seed}/metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(ROOT / metadata["prediction_path"], allow_pickle=False) as prediction:
                source_cache[model_id] = prediction["source_positions"], prediction["source_probability"]
        with np.load(original[dataset][detector, seed, project], allow_pickle=False) as sealed:
            calp, caly = source_calibration_view(source, *source_cache[model_id], sealed["calibration_positions"])
            p = sealed["target_probability"]
            memberships = sealed[cell_key(.1, .2, 20, "estimated_weight_no_gate")]
            trace_single = memberships.sum(axis=1) == 1
            trace_labels = np.where(memberships[:, 1], 1, 0)
        require(cell.archive.nunique() == 1, "A single cell points to multiple action archives")
        with np.load(output / cell.archive.iloc[0], allow_pickle=False) as archive:
            positions = archive["target_positions"]
            for plan in cell.to_dict("records"):
                key = plan["feedback_key"]
                run_id = dataset, detector, seed, project, key
                codes, service, prediction_batch = (archive[f"{key}__{name}"] for name in ("codes", "service_batch", "prediction_batch"))
                require(np.array_equal(service >= 0, codes == REVIEW), "Released review service is not a final review action")
                require(np.all(service[codes == REVIEW] >= prediction_batch[codes == REVIEW]), "A function was reviewed before prediction arrival")
                if plan["method"].startswith("trace_"):
                    require(np.array_equal(codes[trace_single], trace_labels[trace_single]), "Capacity forced or changed a TRACE singleton action")
                    require(np.isin(codes[~trace_single], [REVIEW, BACKLOG]).all(), "Unreviewed mandatory TRACE item was automatically labeled")
                else:
                    require(np.all(codes != BACKLOG), "Nonmandatory complete-roster policy lost a function")
                    require(np.array_equal(service[codes == REVIEW], prediction_batch[codes == REVIEW]), "Current-batch policy used unpriced backlog review")
                for batch in range(int(plan["batches"])):
                    require(np.sum(service == batch) <= int(plan["slots_per_batch"]), "Action archive violates hard batch slots")
                arrived = np.flatnonzero((service >= 0) & (service + int(plan["delay_batches"]) < int(plan["batches"])))
                delayed = np.flatnonzero((service >= 0) & (service + int(plan["delay_batches"]) >= int(plan["batches"])))
                group = event_groups.get(run_id)
                observed = set() if group is None else set(group["index"].astype(int))
                require(observed == set(arrived.tolist()), "Missing arrived labels or free labels from unreviewed/pending items")
                require(len(delayed) == int(plan["reviewed_labels_pending_at_end"]), "Pending reviewed feedback was silently drained")
                pending_checked += len(delayed)
                history = []
                if group is not None:
                    for event in group.to_dict("records"):
                        index = int(event["index"])
                        global_position = int(positions[index])
                        require(int(event["target_position"]) == global_position and event["row_id"] == targets[dataset][global_position]["row_id"], "Feedback identity differs from sealed row")
                        require(codes[index] == REVIEW and int(event["service_batch"]) == int(service[index]), "Feedback was not supplied by an actually reviewed item")
                        require(int(event["label"]) == int(labels[dataset][global_position]), "Released feedback differs from benchmark-label simulation")
                        history.append({"index": index, "label": int(event["label"]), "service_batch": int(event["service_batch"]), "arrival_batch": int(event["arrival_batch"])})
                        releases_checked += 1
                fit_group = fit_groups.get(run_id)
                if fit_group is not None:
                    require(plan["method"].endswith("rolling"), "Static policy updated target calibration")
                    for fit in fit_group.to_dict("records"):
                        batch = int(fit["batch"])
                        window = [event for event in history if event["arrival_batch"] <= batch][-config["feedback"]["rolling_update"]["maximum_arrived_window"]:]
                        indices = json.loads(fit["target_indices"])
                        require(indices == [event["index"] for event in window], "Fit window differs from the most recent arrived reviewed labels")
                        require(all(event["service_batch"] < batch and event["arrival_batch"] <= batch for event in window), "Fit contains future/current or not-yet-arrived labels")
                        require(json_hash(window) == fit["used_window_hash"], "Fit label-window content/identity hash mismatch")
                        outcomes = np.asarray([event["label"] for event in window], dtype=int)
                        require(len(window) >= 30 and np.sum(outcomes == 1) >= 5 and np.sum(outcomes == 0) >= 5, "Fit became eligible using unavailable feedback")
                        weights = np.concatenate([np.full(len(calp), 100.0 / len(calp)), np.ones(len(window))])
                        model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
                            np.concatenate([calp, p[indices]]), np.concatenate([caly, outcomes]), sample_weight=weights)
                        require(json_hash([model.X_thresholds_, model.y_thresholds_]) == fit["knots_sha256"], "Rolling knots cannot be reconstructed from source plus past reviewed labels")
                        fits_checked += 1
                require((0 if fit_group is None else len(fit_group)) == int(plan["rolling_updates"]), "Fit count differs from saved update receipts")
                archive_runs += 1
    require(archive_runs == len(run_plan) and releases_checked == len(events) and fits_checked == len(fits), "Audit omitted an unfavorable policy or fit")
    return {**common, "created_at_utc": datetime.now(timezone.utc).isoformat(), "crosslinked_policy_archives": archive_runs,
            "released_labels_verified_as_reviewed_past_items": releases_checked, "pending_reviewed_labels_retained": pending_checked,
            "rolling_fits_independently_reconstructed": fits_checked, "manifest_sha256": sha256(output / "analysis_manifest.json"),
            "auditor_script_sha256": sha256(Path(__file__)), "no_new_scientific_comparisons": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/capacity_feedback")
    parser.add_argument("--report", type=Path, default=ROOT / "tmp/neurocomputing-strengthening/kbs-v4/capacity_feedback/postrun_verification.json")
    args = parser.parse_args()
    result = audit(args.output.resolve())
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
