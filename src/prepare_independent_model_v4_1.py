"""Deterministic whole-commit family/CVE correction before fresh CodeT5 fits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_independent_model_v4 import strict_temporal_rule, summarize
from prepare_independent_model_v4 import ROOT, V3, read_rows, role_receipts, boundary_audit
from vultriage.data import sha256

OUT = ROOT / "outputs/kbs-validation-v4/independent_model"
ATTEMPT1 = OUT / "attempt-1"
ATTEMPT2 = OUT / "attempt-2"
CONFIG = ROOT / "configs/independent_model_v4_1.json"
PROTOCOL = ROOT / "protocol/independent_model_v4_1.md"


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve existing evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_attempt1() -> None:
    path = ATTEMPT1 / "excluded_source_only_attempt.json"
    if path.exists():
        return
    progress = json.loads((ATTEMPT1 / "training/seed-13/progress.json").read_text())
    history = json.loads((ATTEMPT1 / "training/seed-13/epoch_history.json").read_text())
    files = [p for p in ATTEMPT1.rglob("*") if p.is_file() and p.name != "excluded_source_only_attempt.json"]
    receipt = {"schema_version": "trace-independent-model-v4-excluded-source-only-attempt-1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "interrupted_and_excluded_from_all_model_results",
               "reason": "Metadata-only audit detected ABRT/libreport shared CVE/family and ghostscript-to-mujs source aliases; root authorized a stronger whole-commit correction",
               "stopped_by": "Ctrl-C, exit code 1; no training process remained", "completed_actual_fits": 0,
               "incomplete_seed": 13, "last_saved_progress": progress, "completed_source_epoch_history": history,
               "target_predictions_produced": False, "domain_decisions_produced": False, "target_vault_opened": False,
               "old_config_sha256": sha256(ROOT / "configs/independent_model_v4.json"),
               "old_protocol_sha256": sha256(ROOT / "protocol/independent_model_v4.md"),
               "public_base_manifest_path": str((OUT / "models/codet5-small-b1ee9570/model_manifest.json").relative_to(ROOT)),
               "public_base_manifest_sha256": sha256(OUT / "models/codet5-small-b1ee9570/model_manifest.json"),
               "checkpoint_reuse_for_corrected_fits_allowed": False,
               "files": {str(p.relative_to(ATTEMPT1)): sha256(p) for p in files}}
    write_json(path, receipt)
    (ATTEMPT1 / "excluded_source_only_attempt.sha256").write_text(sha256(path) + "\n", encoding="ascii")


def corrections(source: list[dict], metadata: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    original_events = {role: {cve for row in source if row["role"] == role for cve in metadata[row["row_id"]]["cve_ids"]}
                       for role in ("train", "model_validation", "calibration")}
    family_commits, triggers = set(), defaultdict(list)
    for row in source:
        meta = metadata[row["row_id"]]
        reasons = []
        if row["project"].casefold() == "abrt" or row["project_group"].casefold() == "abrt":
            reasons.append("original_ABRT_project")
        if any(repo.startswith("github.com/abrt/") for repo in meta["repositories"]):
            reasons.append("observed_ABRT_namespace_family")
        if "github.com/artifexsoftware/mujs" in meta["repositories"]:
            reasons.append("confirmed_ghostscript_to_mujs_alias")
        if reasons:
            family_commits.add(row["commit_id"])
            triggers[row["commit_id"]].append({"row_id": row["row_id"], "reasons": reasons})
    valid_commits = {row["commit_id"] for row in source if row["role"] == "model_validation"
                     and set(metadata[row["row_id"]]["cve_ids"]) & original_events["train"]}
    cal_commits = {row["commit_id"] for row in source if row["role"] == "calibration"
                   and set(metadata[row["row_id"]]["cve_ids"]) & (original_events["train"] | original_events["model_validation"])}
    kept, removed = [], []
    for row in source:
        meta = metadata[row["row_id"]]
        if row["commit_id"] in family_commits:
            stage, cves = "known_family_complete_commit", []
        elif row["commit_id"] in valid_commits:
            stage, cves = "validation_CVE_present_in_original_train", sorted(set(meta["cve_ids"]) & original_events["train"])
        elif row["commit_id"] in cal_commits:
            stage, cves = "calibration_CVE_present_in_original_train_or_validation", sorted(set(meta["cve_ids"]) & (original_events["train"] | original_events["model_validation"]))
        else:
            kept.append({**row, "original_position": row["position"], "position": len(kept)})
            continue
        removed.append({"original_position": row["position"], "row_id": row["row_id"], "commit_id": row["commit_id"],
                        "project": row["project"], "project_group": row["project_group"], "role": row["role"],
                        "source_label": row["label"], "stage": stage, "observed_overlapping_cves": cves,
                        "family_commit_triggers": triggers.get(row["commit_id"], [])})
    details = {"rule": "Complete global SHA closure; family first, then validation CVEs in ORIGINAL train, calibration CVEs in ORIGINAL train OR ORIGINAL validation; no role reassignment, class resampling or rebalancing",
               "family_projects": ["abrt"], "family_namespace_prefixes": ["github.com/abrt/"],
               "confirmed_alias_namespaces": ["github.com/artifexsoftware/mujs"],
               "family_commits": sorted(family_commits), "validation_event_commits": sorted(valid_commits),
               "calibration_event_commits": sorted(cal_commits),
               "original_role_observed_cve_counts": {k: len(v) for k, v in original_events.items()},
               "unknown_cve_rule": "Unobserved identifiers stay unknown; observed-disjointness is not complete CVE/event independence",
               "removed_by_stage_role": {stage: {role: {"rows": len([r for r in removed if r["stage"] == stage and r["role"] == role]),
                                                        "commits": len({r["commit_id"] for r in removed if r["stage"] == stage and r["role"] == role}),
                                                        "source_labels": dict(Counter(str(r["source_label"]) for r in removed if r["stage"] == stage and r["role"] == role))}
                                              for role in ("train", "model_validation", "calibration")}
                                          for stage in sorted({r["stage"] for r in removed})}}
    return kept, removed, details


def protocol_text(roles: dict) -> str:
    counts = "; ".join(f"{r}: {v['rows']} functions, {v['commits']} commits, {v['labels'].get('1',0)} vulnerable and {v['labels'].get('0',0)} safe" for r, v in roles.items())
    return f"""# TRACE independent-family amendment v4.1, 11 October 2026

This is an author-authorized metadata-boundary correction after original and v3
target outcomes were known. An initial CodeT5 source-only attempt at seed 13 was
interrupted after the metadata audit exposed known ABRT/libreport vulnerability
and family sharing plus ghostscript-to-mujs aliases. Its source losses and epoch-1
validation AP are known; its checkpoint and all software/input evidence are
retained under `outputs/kbs-validation-v4/independent_model/attempt-1` and excluded
from all reported detector results. No CodeT5 target prediction, domain decision,
or target evaluation was produced by that attempt. This correction is frozen
before three fresh fits and any CodeT5 target comparison. It is neither public
preregistration nor a fresh-unseen-outcome claim.

The question remains whether TRACE allocation, workload and illustrative cost
observations transfer to a separate encoder family under stronger observed
source-family/event screening. It is not an architecture-superiority experiment:
the source cohort differs from v3 and original/v3 results are already known.

## Deterministic source correction

Start from the exact v3.1 separated source rows. Remove every source function in
any complete global source commit named `abrt`, recovered under `github.com/abrt/`,
or recovered in the confirmed `github.com/artifexsoftware/mujs` namespace. This
acts on the known ABRT/libreport family and ghostscript alias rather than merely
disclaiming them. Then remove whole model-validation commits whose OBSERVED
explicit CVEs occur in ORIGINAL source train; remove whole calibration commits
whose observed explicit CVEs occur in ORIGINAL source train OR ORIGINAL source
model validation. Original-set priority is conservative: discarded earlier-role
commits still contribute their observed event identifiers to later-role screens.
The exact metadata audit and exclusion ledger bind every trigger and row.

Surviving roles remain assigned exactly as before. Do not sample replacement
negatives, rebalance classes, reassign commits, or relax the rule from any model
outcome or source class count. The old 4:1 maximum training sample is followed by
these whole-commit exclusions; final class ratios and natural held-out prevalence
are reported explicitly. Source rows are only reindexed, with original position
and row identity preserved. Frozen final roles: {counts}.

All original D38 and M6 target projects/functions remain byte-identical to their
previously screened metadata, with the same original target-label vaults and
candidate/exclusion receipts. Both datasets are reported separately. No new
target-label eligibility or favorable-subset selection is used.

## Fresh encoder fits

Use public `Salesforce/codet5-small` at Hugging Face commit
`b1ee9570c289f21b5922b9c768a1ce12957bf968`. Instantiate the complete T5EncoderModel
with six blocks and 512-dimensional masked mean features excluding special
tokens, dropout 0.1, and a random two-class linear head (normal weight std 0.02,
zero bias). The unused pretrained decoder is absent and not counted as trained.
Every instantiated encoder/head parameter is trainable and must receive finite
nonzero gradients; every block must show a nonzero selected-checkpoint query
update. Each fresh seed 13/37/73 reloads the original pinned public encoder and
independently initializes the head and training permutation. No interrupted
attempt checkpoint may be reused or counted as an actual completed fit.

Keep three epochs, maximum 512 tokens, micro-batch 4, effective batch 32,
unweighted cross entropy, AdamW learning rate 2e-5, weight decay 0.01, linear
warmup 0.1, gradient clip 1.0, bfloat16 autocast with float32 parameters and
optimizer states, gradient checkpointing, and the v3 128-row length windows.
Select the maximum natural source model-validation AP over those three epochs,
ties favoring the earlier epoch. Calibration and target labels select nothing.
Bind the fresh config, source/role/token/base hashes, actual checkpoint, script,
seed, tensor digests/updates, gradient evidence, runtime, memory, and failures.
The same source rows are used by all three seeds; these technical fits are not
independent data units. Root clearance reserves the single GPU slot.

## Fixed decisions and analysis

The analysis specification remains unchanged from the first v4 amendment and
v3 policy audit: all nine budget pairs, primary (0.10,0.20); clips 10/20/50,
primary 20; miss costs 10/50, false-alert cost 1, review cost 0.1, assuming correct
review in illustrative units. Domain fits use the final 512-dimensional features
and five folds grouped by global complete commit SHA, with balanced SGD alpha
1e-4 for five epochs and the existing fixed salt. Use only source calibration
labels for conformal sets and source-only isotonic expected costs.

Compare weighted and unweighted sets, global-confidence count matching, disjoint
tails at both automatic-label counts, source-only fixed/free-volume cost policies,
forced argmax, and all-review. Primary: paired project median automatic-miss rate
difference, weighted minus allocation-matched tails at primary budgets/clip.
Secondary: false alerts, coverage, miscoverage, selective error, relative budget
violation, cost, discrimination/calibration and all fixed sensitivities. Average
actual fits within project first. Keep every project, including sealed review
fallback for domain failures; failed fitted-set risk is undefined. Supported-only
analysis is secondary. Descriptive pointwise 95% project-bootstrap intervals use
10,000 draws and seed 20261010; zero-crossing intervals do not prove equivalence.
Costs do not measure analyst time or money.

Label-free predictions and complete project policy decisions for BOTH target
datasets must be sealed, with all planned fit attempts accounted, before the
evaluation command opens either original target-label vault. The seal documents
the no-new-target-tuning boundary, not retroactive blindness to known v3 results.
Retain all unfavorable outcomes, missing predictions, fit failures and refusals.

## What the stronger metadata screen establishes

Known ABRT/libreport and mujs source-family aliases and observed CVE overlaps are
removed by complete commit. Existing exact/identical lexical key and complete
SHA separation is rechecked. Missing CVE identifiers remain unknown, so complete
event independence is not certified. Recovered repository namespaces and known
family lists do not exhaust shared components, forks, or pretraining overlap.

The strict temporal cutoff remains the maximum actual commit timestamp over
EVERY retained source-role commit, usable only with complete unambiguous source
date coverage. Target timestamps must be available and strictly later. Source
dates are incomplete after this correction, so the strict subset is unavailable,
with a saved complete metadata-only roster and no substitute cutoff. CVE years,
file order, publication/release dates and naive guessed timestamps are not dates.
Do not relax chronology or existing project eligibility from results.

Complete the three fresh fits and both full target rosters once. Preserve the
excluded attempt and every original/v3 input, result and seal. The corrected
experiment strengthens known isolation while retaining its limits; it cannot
guarantee journal acceptance.
"""


def prepare() -> dict:
    record_attempt1()
    if ATTEMPT2.exists() or CONFIG.exists() or PROTOCOL.exists():
        raise FileExistsError("The correction must have new immutable identifiers")
    ATTEMPT2.mkdir(parents=True)
    original_cfg = json.loads((ROOT / "configs/independent_model_v4.json").read_text())
    source_path = ATTEMPT1 / "inputs/source_rows.jsonl"
    if sha256(source_path) != original_cfg["source_rows_sha256"]:
        raise ValueError("Archived initial source input changed")
    source = read_rows(source_path, source=True)
    old_audit_path = ATTEMPT1 / "metadata-audit/metadata_audit.json"
    old_audit = json.loads(old_audit_path.read_text())
    metadata_path = old_audit_path.parent / "source_metadata.jsonl"
    if sha256(metadata_path) != old_audit["report_hashes"][metadata_path.name]:
        raise ValueError("Initial retained source metadata changed")
    metadata = {row["row_id"]: row for row in [json.loads(line) for line in metadata_path.open(encoding="utf-8")]}
    kept, removed, details = corrections(source, metadata)
    inputs = ATTEMPT2 / "inputs"
    inputs.mkdir()
    corrected_source = inputs / "source_rows.jsonl"
    with corrected_source.open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    roles = role_receipts(kept)
    if [roles[r]["rows"] for r in ("train", "model_validation", "calibration")] != [15594, 6215, 4294]:
        raise ValueError("The deterministic correction counts differ from the cleared rule; preserve and investigate")
    PROTOCOL.write_text(protocol_text(roles), encoding="utf-8")
    cfg = {**original_cfg, "protocol_version": "trace-independent-model-v4.1", "protocol_path": str(PROTOCOL.relative_to(ROOT)),
           "protocol_sha256": sha256(PROTOCOL), "status": "Metadata-only whole-commit family/CVE correction; original/v3 outcomes and interrupted source-only attempt AP known; no CodeT5 target outcomes produced",
           "known_outcomes": "Original/v3 target outcomes and interrupted v4 source-only losses/AP known; no CodeT5 target prediction/evaluation before v4.1 freeze",
           "previous_config_sha256": sha256(ROOT / "configs/independent_model_v4.json"),
           "source_rows_sha256": sha256(corrected_source), "source_roles": roles,
           "source_role_correction": details,
           "source": {**original_cfg["source"], "post_sampling_correction": "Whole-commit known-family and observed-CVE priority exclusions; no new sampling/rebalancing/reassignment"},
           "repository_family_exclusions": {**original_cfg["repository_family_exclusions"], "libreport": ["abrt", "libreport"]},
           "qualified_family_exclusions": {"prefixes": ["github.com/abrt/"], "exact": ["github.com/artifexsoftware/mujs"]},
           "paths": {**original_cfg["paths"], "source_rows": str(corrected_source.relative_to(ROOT)),
                     "diversevul_inputs": str((inputs / "diversevul").relative_to(ROOT)),
                     "megavul_inputs": str((inputs / "megavul").relative_to(ROOT)),
                     "output_root": str(ATTEMPT2.relative_to(ROOT)),
                     "excluded_attempt_receipt": str((ATTEMPT1 / "excluded_source_only_attempt.json").relative_to(ROOT))}}
    write_json(CONFIG, cfg)
    targets = {}
    for dataset, old in [("diversevul", V3), ("megavul", V3 / "megavul")]:
        destination = inputs / dataset
        destination.mkdir()
        shutil.copyfile(old / "target_rows.jsonl", destination / "target_rows.jsonl")
        shutil.copyfile(old / "candidate_projects.csv", destination / "candidate_projects.csv")
        targets[dataset] = read_rows(destination / "target_rows.jsonl", source=False)
        previous = json.loads((old / "preparation_manifest.json").read_text())
        prep = {"schema_version": "trace-independent-model-v4.1-inherited-target-preparation-1", "protocol_version": cfg["protocol_version"],
                "dataset": dataset, "selected_projects": previous["selected_projects"],
                "candidate_projects": previous["candidate_projects"], "target_rows": len(targets[dataset]),
                "target_positive_from_known_historical_preparation": previous["target_positive"],
                "target_metadata_byte_identical_to_v3": True, "target_outputs_used_for_selection": False,
                "target_labels_opened_for_correction": False, "source_training_changed": True,
                "source_counts": {role: value["labels"] for role, value in roles.items()},
                "output_hashes": {"source_rows.jsonl": sha256(corrected_source), "target_rows.jsonl": sha256(destination / "target_rows.jsonl"),
                                  "candidate_projects.csv": sha256(destination / "candidate_projects.csv"),
                                  "target_label_vault.npz": previous["output_hashes"]["target_label_vault.npz"]},
                "original_target_label_vault_path": str((old / "target_label_vault.npz").relative_to(ROOT)),
                "original_preparation_manifest_path": str((old / "preparation_manifest.json").relative_to(ROOT)),
                "original_preparation_manifest_sha256": sha256(old / "preparation_manifest.json")}
        write_json(destination / "preparation_manifest.json", prep)
    with (ATTEMPT2 / "excluded_source_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["original_position", "row_id", "commit_id", "project", "project_group", "role", "source_label", "stage", "observed_overlapping_cves", "family_commit_triggers"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in removed:
            writer.writerow({**row, "observed_overlapping_cves": json.dumps(row["observed_overlapping_cves"]), "family_commit_triggers": json.dumps(row["family_commit_triggers"])})
    ledger = {"schema_version": "trace-independent-model-v4.1-source-exclusion-ledger-1", "config_sha256": sha256(CONFIG),
              "original_source_sha256": sha256(source_path), "corrected_source_sha256": sha256(corrected_source),
              "original_metadata_audit_sha256": sha256(old_audit_path), "source_roles_before": original_cfg["source_roles"],
              "source_roles_after": roles, "exclusion_rule": details, "removed_rows": len(removed),
              "removed_commits": len({r["commit_id"] for r in removed}), "target_outcomes_used": False,
              "row_ledger_sha256": sha256(ATTEMPT2 / "excluded_source_rows.csv")}
    write_json(ATTEMPT2 / "source_exclusion_ledger.json", ledger)
    audit_dir = ATTEMPT2 / "metadata-audit"
    audit_dir.mkdir()
    records = {"source": [{**metadata[r["row_id"]], "original_position": r["original_position"], "position": r["position"]} for r in kept]}
    for dataset in targets:
        records[dataset] = [json.loads(line) for line in (ATTEMPT1 / "metadata-audit" / f"{dataset}_metadata.jsonl").open(encoding="utf-8")]
    for name, rows in records.items():
        with (audit_dir / f"{name}_metadata.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    subsets = {role: [r for r in records["source"] if r["role"] == role] for role in ("train", "model_validation", "calibration")}
    subsets.update(source_all=records["source"], **{d: records[d] for d in targets})
    events = {name: {c for row in rows for c in row["cve_ids"]} for name, rows in subsets.items()}
    overlap = {f"{a}:{b}": sorted(events[a] & events[b]) for a, b in [("train", "model_validation"), ("train", "calibration"),
                                                                 ("model_validation", "calibration"), ("source_all", "diversevul"), ("source_all", "megavul")]}
    if any(overlap.values()):
        raise ValueError("Known event overlap remains after the explicit conservative correction")
    temporal = {d: strict_temporal_rule(records["source"], records[d]) for d in targets}
    with (audit_dir / "strict_temporal_roster.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["dataset", "project", "position", "row_id", "commit_id", "actual_commit_timestamp", "source_cutoff_timestamp", "strict_subset_available", "strict_temporal_eligible", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in targets:
            eligible = set(temporal[dataset]["eligible_row_ids"])
            for row in records[dataset]:
                writer.writerow(dict(dataset=dataset, project=row["project_group"], position=row["position"], row_id=row["row_id"], commit_id=row["commit_id"],
                                     actual_commit_timestamp=row["dates"][0] if len(row["dates"]) == 1 else "", source_cutoff_timestamp=temporal[dataset]["cutoff_timestamp"],
                                     strict_subset_available=temporal[dataset]["available"], strict_temporal_eligible=row["row_id"] in eligible, reason=temporal[dataset]["reason"]))
    audit = {"schema_version": "trace-independent-model-v4.1-corrected-metadata-audit-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "config_sha256": sha256(CONFIG), "source_rows_sha256": sha256(corrected_source), "target_vault_opened": False,
             "target_outcomes_compared": False, "original_metadata_audit_sha256": sha256(old_audit_path),
             "availability": {name: summarize(rows) for name, rows in subsets.items()}, "observed_identifier_overlap_after_correction": overlap,
             "strict_temporal_subsets": temporal, "event_independence_certified": False,
             "event_scope": "Known explicit-CVE overlap removed; unknown metadata stays unknown",
             "raw_dependencies": old_audit["raw_dependencies"], "script_sha256": sha256(Path(__file__)),
             "report_hashes": {p.name: sha256(p) for p in audit_dir.iterdir() if p.is_file()}}
    write_json(audit_dir / "metadata_audit.json", audit)
    (audit_dir / "metadata_audit.sha256").write_text(sha256(audit_dir / "metadata_audit.json") + "\n", encoding="ascii")
    known_limits = {"schema_version": "trace-independent-model-v4.1-known-family-event-correction-1", "config_sha256": sha256(CONFIG),
                    "excluded_attempt_sha256": sha256(ATTEMPT1 / "excluded_source_only_attempt.json"),
                    "source_exclusion_ledger_sha256": sha256(ATTEMPT2 / "source_exclusion_ledger.json"),
                    "known_families_removed": details, "observed_source_target_cve_overlap_after": {d: overlap[f"source_all:{d}"] for d in targets},
                    "source_sampling_changed": True, "target_rosters_changed": False, "unknown_event_independence_certified": False,
                    "target_outcomes_compared": False}
    write_json(ATTEMPT2 / "focused_family_event_findings.json", known_limits)
    (ATTEMPT2 / "focused_family_event_findings.sha256").write_text(sha256(ATTEMPT2 / "focused_family_event_findings.json") + "\n", encoding="ascii")
    checks = boundary_audit(kept, targets, cfg)
    paths = [CONFIG, PROTOCOL, corrected_source, Path(__file__), ROOT / "src/prepare_independent_model_v4.py",
             ATTEMPT2 / "source_exclusion_ledger.json", ATTEMPT2 / "excluded_source_rows.csv", audit_dir / "metadata_audit.json",
             ATTEMPT2 / "focused_family_event_findings.json", ATTEMPT1 / "excluded_source_only_attempt.json", old_audit_path,
             OUT / "models/codet5-small-b1ee9570/model_manifest.json"]
    paths += [inputs / dataset / filename for dataset in targets for filename in ("target_rows.jsonl", "candidate_projects.csv", "preparation_manifest.json")]
    freeze = {"schema_version": "trace-independent-model-v4.1-amendment-freeze-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256(CONFIG), "protocol_sha256": sha256(PROTOCOL), "source_rows_sha256": sha256(corrected_source),
              "source_roles": roles, "boundary_audit": checks, "source_exclusion_rule": details,
              "target_labels_opened": False, "new_model_target_outcomes_compared": False,
              "dependencies": [{"path": str(p.resolve()), "sha256": sha256(p), "role": "corrected_amendment_dependency"} for p in paths],
              "excluded_attempt_receipt_sha256": sha256(ATTEMPT1 / "excluded_source_only_attempt.json"),
              "prepared_target_vault_hashes_from_historical_receipts_only": {d: json.loads((inputs / d / "preparation_manifest.json").read_text())["output_hashes"]["target_label_vault.npz"] for d in targets}}
    write_json(ATTEMPT2 / "amendment_freeze.json", freeze)
    (ATTEMPT2 / "amendment_freeze.sha256").write_text(sha256(ATTEMPT2 / "amendment_freeze.json") + "\n", encoding="ascii")
    return freeze


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = prepare()
    print(json.dumps({k: result[k] for k in ["config_sha256", "protocol_sha256", "source_rows_sha256", "source_roles", "source_exclusion_rule", "boundary_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
