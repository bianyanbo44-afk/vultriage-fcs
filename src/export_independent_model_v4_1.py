"""Export only portable project/fit aggregates after independent reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruct_independent_model_v4_1 import digest, json_object

ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "project_seed_metrics.csv", "project_seed_averages.csv", "paired_project_summaries.csv",
    "domain_supported_paired_summaries.csv", "primary_paired_summary.csv", "project_medians.csv",
    "project_counts_positive_commits.csv", "detector_project_seed_metrics.csv", "detector_project_seed_averages.csv",
)
PUBLIC_CODE = (
    "src/independent_model_v4_1.py", "src/prepare_independent_model_v4_1.py", "src/encode_independent_model_v4_1.py",
    "src/train_independent_model_v4_1.py", "src/predict_independent_model_v4_1.py", "src/evaluate_independent_model_v4_1.py",
    "src/seal_joint_independent_model_v4_1.py", "src/reconstruct_independent_model_v4_1.py", "src/export_independent_model_v4_1.py",
    "src/audit_fits_independent_model_v4_1.py",
    "src/prepare_independent_model_v4.py", "src/audit_independent_model_v4.py",
    "configs/independent_model_v4_1.json", "protocol/independent_model_v4_1.md",
    "tests/test_independent_model_v4_1.py", "tests/test_reconstruct_independent_model_v4_1.py",
)


def export(config: Path, output: Path) -> dict:
    cfg = json_object(config)
    root = ROOT / cfg["paths"]["output_root"]
    independent_path = root / "independent-reconstruction/independent_reconstruction.json"
    independent = json_object(independent_path)
    if independent.get("all_datasets_match") is not True or independent.get("both_complete_seals_verified_before_vault_access") is not True:
        raise ValueError("Export requires the completed independent post-seal reconstruction")
    fit_audit_path = root / "independent_actual_fit_audit.json"
    fit_audit = json_object(fit_audit_path)
    if fit_audit.get("all_three_actual_fits_verified") is not True or fit_audit.get("config_sha256") != digest(config):
        raise ValueError("Export also requires independently verified actual checkpoints/fresh heads")
    if output.exists():
        raise FileExistsError("Preserve an existing public aggregate export")
    output.mkdir(parents=True)
    source_roles = []
    for role, info in cfg["source_roles"].items():
        source_roles.append(dict(role=role, n=info["rows"], n_vulnerable=info["labels"]["1"], n_safe=info["labels"]["0"],
                                 commits=info["commits"], prevalence=info["labels"]["1"] / info["rows"],
                                 source_role_full_rows_sha256=info["canonical_full_rows_sha256"]))
    pd.DataFrame(source_roles).to_csv(output / "source_role_counts.csv", index=False)
    shutil.copyfile(root / "source_cohort_change.csv", output / "source_cohort_change.csv")
    metadata = json_object(root / "metadata-audit/metadata_audit.json")
    availability = []
    for role, row in metadata["availability"].items():
        availability.append({"role_or_dataset": role, **{k: v for k, v in row.items() if k != "unique_qualified_repositories"}})
    pd.DataFrame(availability).to_csv(output / "metadata_availability_summary.csv", index=False)
    training = []
    checkpoints, head_initial_hashes = set(), set()
    encoder_initial = None
    for seed in cfg["fine_tuned_codet5"]["seeds"]:
        path = root / "training" / f"seed-{seed}" / "training_manifest.json"
        fit = json_object(path)
        if (fit["actual_independent_fit"] is not True or fit["source_rows_sha256"] != cfg["source_rows_sha256"]
                or fit["source_role_hashes"] != cfg["source_roles"] or fit["unused_pretrained_decoder_instantiated"] is not False
                or fit["config_sha256"] != digest(config) or fit["optimizer_steps"] != fit["planned_optimizer_steps"]):
            raise ValueError("Actual corrected fit identity/source/decoder/update evidence is inconsistent")
        checkpoint = path.parent / "best_model.pt"
        if digest(checkpoint) != fit["checkpoint_sha256"]:
            raise ValueError("Actual selected checkpoint bytes changed")
        if fit["checkpoint_sha256"] in checkpoints or fit["classifier_initial_tensor_hashes"]["weight"] in head_initial_hashes:
            raise ValueError("Checkpoints or random heads were copied across purported independent fits")
        checkpoints.add(fit["checkpoint_sha256"])
        head_initial_hashes.add(fit["classifier_initial_tensor_hashes"]["weight"])
        if encoder_initial is None:
            encoder_initial = fit["encoder_initial_tensor_hashes"]
        elif encoder_initial != fit["encoder_initial_tensor_hashes"]:
            raise ValueError("Fresh fits did not start from identical pinned public encoder tensors")
        for record in fit["executed_script_snapshots"]:
            if digest(ROOT / record["snapshot_path"]) != record["sha256"]:
                raise ValueError("A per-fit executed script snapshot changed")
        updates = fit["encoder_query_relative_updates"]
        training.append(dict(detector=cfg["detector"], seed=seed, selected_epoch=fit["selected_epoch"],
                             source_validation_ap=fit["source_validation_ap"], epochs_completed=len(fit["history"]),
                             optimizer_steps=fit["optimizer_steps"], encoder_parameters=fit["encoder_parameters"],
                             classifier_parameters=fit["classifier_parameters"], all_instantiated_trainable_parameters=fit["trainable_parameters"],
                             unused_decoder_instantiated=False, updated_encoder_blocks=len(updates),
                             minimum_query_relative_update=min(updates.values()), maximum_query_relative_update=max(updates.values()),
                             changed_encoder_tensors=fit["changed_encoder_tensor_count"], encoder_tensors=fit["encoder_tensor_count"],
                             wall_seconds=fit["wall_seconds"], peak_gpu_allocated_bytes=fit["peak_gpu_allocated_bytes"],
                             checkpoint_sha256=fit["checkpoint_sha256"], initial_head_weight_sha256=fit["classifier_initial_tensor_hashes"]["weight"],
                             training_manifest_sha256=digest(path), executed_train_script_sha256=fit["script_sha256"],
                             executed_common_script_sha256=fit["common_script_sha256"],
                             source_rows_sha256=fit["source_rows_sha256"], source_token_manifest_sha256=fit["source_token_manifest_sha256"],
                             base_model_manifest_sha256=fit["model_manifest_sha256"], model_revision=fit["model_revision"]))
    pd.DataFrame(training).to_csv(output / "training_fit_summary.csv", index=False)
    primary_policies, primary_pairs, primary_actions, detector_summary, datasets = [], [], [], [], {}
    for dataset in ("diversevul", "megavul"):
        evaluation = root / "evaluation" / dataset
        evaluation_manifest = json_object(evaluation / "evaluation_manifest.json")
        if evaluation_manifest["verified_before_target_label_access"] is not True:
            raise ValueError("Target metrics lack a verified decision boundary")
        for name, expected in evaluation_manifest["report_hashes"].items():
            if digest(evaluation / name) != expected:
                raise ValueError("A evaluated report changed before public export")
        seal = json_object(root / "decisions" / dataset / "decision_seal.json")
        folder = output / dataset
        folder.mkdir()
        for name in TABLES:
            shutil.copyfile(evaluation / name, folder / name)
        shutil.copyfile(root / "decisions" / dataset / "domain_fit_availability.csv", folder / "domain_fit_availability.csv")
        receipt = {"schema_version": "trace-independent-model-v4.1-public-dataset-receipt-1", "dataset": dataset,
                   "projects": seal["projects"], "target_rows": evaluation_manifest["target_rows"],
                   "n_vulnerable": evaluation_manifest["target_vulnerable"], "actual_seeds": cfg["fine_tuned_codet5"]["seeds"],
                   "all_projects_retained": True, "domain_supported_projects": seal["domain_supported_projects"][cfg["detector"]],
                   "domain_unsupported_projects": seal["domain_unsupported_projects"][cfg["detector"]],
                   "domain_failure_cells": seal["domain_failure_cells"], "final_training_failures": evaluation_manifest["final_training_failures"],
                   "target_roster_byte_sha256": digest(ROOT / cfg["paths"][f"{dataset}_inputs"] / "target_rows.jsonl"),
                   "target_label_vault_sha256": evaluation_manifest["target_label_vault_sha256"],
                   "decision_seal_sha256": digest(root / "decisions" / dataset / "decision_seal.json"),
                   "evaluation_manifest_sha256": digest(evaluation / "evaluation_manifest.json"),
                   "known_original_and_v3_outcomes": True, "no_fresh_unseen_outcome_claim": True,
                   "pointwise_intervals": "descriptive project bootstrap; no equivalence/population guarantee",
                   "mirror_note": "Fixed third-party MegaVul mirror; full official byte equivalence unknown" if dataset == "megavul" else "Official DiverseVul release; inherited full D38 cohort"}
        (folder / "dataset_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        datasets[dataset] = {"projects": seal["projects"], "target_rows": seal["target_rows"],
                             "domain_supported_projects": seal["domain_supported_projects"][cfg["detector"]],
                             "domain_unsupported_projects": seal["domain_unsupported_projects"][cfg["detector"]]}
        av, ass = cfg["risk_budgets"]["primary"]
        clip = cfg["weighting"]["primary_clip"]
        medians = pd.read_csv(evaluation / "project_medians.csv")
        primary_policies.append(medians[(medians["alpha_v"] == av) & (medians["alpha_s"] == ass) & (medians["clip"] == clip)])
        pairs = pd.read_csv(evaluation / "paired_project_summaries.csv")
        comparators = ["class_matched_weighted", "msp_matched_weighted", "source_iso_cost10_matched_weighted", "source_iso_cost50_matched_weighted",
                       "source_iso_cost10_free_volume", "source_iso_cost50_free_volume"]
        endpoints = ["auto_fn_rate", "auto_fp_rate", "coverage", "cost_10", "cost_50"]
        selected = pairs[(pairs["alpha_v"] == av) & (pairs["alpha_s"] == ass) & (pairs["clip"] == clip)
                         & pairs["baseline"].isin(comparators) & pairs["outcome"].isin(endpoints)].copy()
        selected["comparator_family"] = selected["baseline"].map(lambda v: "allocation_matched_tails" if v == "class_matched_weighted"
                                                                else "global_argmax_confidence" if v == "msp_matched_weighted" else "source_only_expected_cost")
        primary_pairs.append(selected)
        actions = pd.read_csv(evaluation / "project_seed_averages.csv")
        actions = actions[(actions["alpha_v"] == av) & (actions["alpha_s"] == ass) & (actions["clip"] == clip)]
        columns = ["dataset", "detector", "project", "method", "status", "actual_seed_count", "n", "n_vulnerable", "n_safe",
                   "auto_safe", "auto_vulnerable", "review_count", "coverage", "auto_fn", "auto_fp", "auto_fn_rate", "auto_fp_rate", "cost_10", "cost_50"]
        primary_actions.append(actions[columns])
        detector = pd.read_csv(evaluation / "detector_project_seed_averages.csv")
        numeric = [c for c in detector.select_dtypes(include=[np.number])]
        for cohort, eligible in [("all_projects", seal["projects"]), ("domain_supported_secondary", seal["domain_supported_projects"][cfg["detector"]])]:
            subset = detector[detector["project"].isin(eligible)]
            detector_summary.append(dict(dataset=dataset, detector=cfg["detector"], cohort=cohort, projects=len(subset), actual_seed_count=3,
                                         **{f"median_{c}": float(subset[c].median()) for c in numeric}))
    pd.concat(primary_policies, ignore_index=True).to_csv(output / "primary_policy_medians.csv", index=False)
    pd.concat(primary_pairs, ignore_index=True).to_csv(output / "primary_paired_comparators.csv", index=False)
    pd.concat(primary_actions, ignore_index=True).to_csv(output / "project_actions_primary.csv", index=False)
    pd.DataFrame(detector_summary).to_csv(output / "detector_summary.csv", index=False)
    reconstruction_report = {k: independent[k] for k in ["schema_version", "config_sha256", "joint_dataset_decision_seal_sha256", "both_complete_seals_verified_before_vault_access", "imports_implementation_metric_helpers", "all_datasets_match", "datasets", "script_sha256"]}
    (output / "independent_reconstruction_report.json").write_text(json.dumps(reconstruction_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(fit_audit_path, output / "independent_actual_fit_audit.json")
    readme = """# CodeT5 independent-family aggregates (corrected v4.1)

These are project/fit aggregates only. They contain no function text, individual
probabilities/actions/events, token arrays, target-label vaults, or model weights.
Three genuine fresh CodeT5 encoder fits used the corrected source roles; the
interrupted source-only attempt is excluded. The unused decoder was absent.

All D38 and M6 projects are retained. The source correction removed 213 functions
from 30 whole commits for known ABRT/libreport/mujs family or observed-CVE overlaps,
preserving surviving role assignments without resampling. Unknown events stay
unknown. Source timestamps remain incomplete: the strict chronological subset
was unavailable, with no substitute cutoff. Original/v3 target outcomes were
already known, so this is a dated amendment and no fresh-unseen/preregistration
claim is made. Source-screening differences prevent architecture-superiority
interpretations from descriptive cross-detector comparisons.

Project seed averages precede paired project-bootstrap inference. All-project
deployment results retain sealed review fallback for domain-fit failures;
supported-only results are secondary. Undefined fitted-set risk and selective
error remain undefined. Costs use fixed illustrative units, correct-review
assumption and source-only isotonic decisions; they do not measure analyst time.

The standalone `reconstruct_independent_model_v4_1.py --aggregate-root PATH`
verifies manifest hashes and independently reconstructs seed averages, count-cost
algebra and every descriptive pointwise project-bootstrap interval using only
these public files. Individual score/action metrics were separately reconstructed
against both verified local decision seals and original aligned vaults before
export. The export manifest records that independent verification receipt.

The retained `prepare_independent_model_v4.py` and
`audit_independent_model_v4.py` are imports supplying data/metadata utility
functions used by the executed v4.1 scripts. Old v4 fitting/inference entry points,
old protocol/config, all attempt-1 results and private caches are excluded from
the public aggregate release. Shared v3 policy helpers retain their original
bytes. No publication or push was performed by this exporter.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    members = {str(p.relative_to(output)).replace("\\", "/"): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {"schema_version": "trace-independent-model-v4.1-public-aggregates-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_version": cfg["protocol_version"], "detector": cfg["detector"], "model_id": cfg["fine_tuned_codet5"]["model_id"],
                "model_revision": cfg["fine_tuned_codet5"]["revision"], "actual_seeds": cfg["fine_tuned_codet5"]["seeds"],
                "config_sha256": digest(config), "protocol_sha256": cfg["protocol_sha256"],
                "source_rows_sha256": cfg["source_rows_sha256"], "source_roles": cfg["source_roles"],
                "corrected_amendment_freeze_sha256": digest(root / "amendment_freeze.json"),
                "joint_dataset_decision_seal_sha256": digest(root / "joint_dataset_decision_seal.json"),
                "independent_reconstruction_sha256": digest(independent_path),
                "independent_actual_fit_audit_sha256": digest(fit_audit_path),
                "metadata_audit_sha256": digest(root / "metadata-audit/metadata_audit.json"),
                "source_exclusion_ledger_sha256": digest(root / "source_exclusion_ledger.json"),
                "excluded_attempt_receipt_sha256": digest(ROOT / cfg["paths"]["excluded_attempt_receipt"]),
                "known_original_and_v3_outcomes": True, "architecture_superiority_claim": False,
                "strict_temporal_available": False, "complete_cve_event_independence_claim": False,
                "datasets": datasets, "analysis": {"bootstrap_draws": 10000, "bootstrap_seed": cfg["statistics"]["bootstrap_seed"],
                                                       "primary_budgets": cfg["risk_budgets"]["primary"], "primary_clip": cfg["weighting"]["primary_clip"],
                                                       "miss_costs": cfg["cost"]["miss"], "false_alert_cost": cfg["cost"]["false_alert"], "review_cost": cfg["cost"]["review"],
                                                       "unit": "project after averaging all actual technical fits; datasets separate", "interval": "pointwise descriptive percentile"},
                "excluded_private_content": ["function text", "individual predictions/actions/events", "model weights", "tokens", "label vaults", "attempt-1 outputs"],
                "required_code_whitelist": {name: digest(ROOT / name) for name in PUBLIC_CODE},
                "export_script_sha256": digest(Path(__file__)), "members": members}
    (output / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "aggregate_manifest.sha256").write_text(digest(output / "aggregate_manifest.json") + "\n", encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json_object(args.config)
    output = args.output or ROOT / cfg["paths"]["output_root"] / "aggregate-export"
    result = export(args.config, output)
    print(json.dumps({"datasets": {k: len(v["projects"]) for k, v in result["datasets"].items()}, "public_members": len(result["members"]),
                      "manifest_sha256": digest(output / "aggregate_manifest.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
