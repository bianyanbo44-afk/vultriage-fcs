"""Bind an amendment to the exact read-only v3 input roles without label-vault access."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from vultriage.data import sha256
from vultriage.neural_audit import canonical_commit_id
from vultriage.near_duplicate import lexical_token_set, token_set_sha256

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "outputs/neural-validation-v3/inputs"
CONFIG = ROOT / "configs/independent_model_v4.json"
PROTOCOL = ROOT / "protocol/independent_model_v4.md"
OUT = ROOT / "outputs/kbs-validation-v4/independent_model"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def read_rows(path: Path, *, source: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    ids = set()
    for position, row in enumerate(rows):
        if row["position"] != position or row["row_id"] in ids:
            raise ValueError("Input row positions or identities are invalid")
        ids.add(row["row_id"])
        if not source and any(k.casefold() in {"label", "labels", "target", "is_vul", "y"} for k in row):
            raise ValueError("Target input contains labels")
        if source and (row["role"] not in {"train", "model_validation", "calibration"}
                       or row["label"] not in {0, 1}):
            raise ValueError("Invalid source role/label")
    return rows


def role_receipts(rows: list[dict]) -> dict:
    result = {}
    for role in ("train", "model_validation", "calibration"):
        subset = [r for r in rows if r["role"] == role]
        digest = hashlib.sha256()
        for row in subset:
            digest.update(canonical_bytes(row))
        result[role] = {"rows": len(subset), "labels": dict(Counter(str(r["label"]) for r in subset)),
                        "commits": len({canonical_commit_id(r["commit_id"]) for r in subset}),
                        "canonical_full_rows_sha256": digest.hexdigest().upper()}
    return result


def boundary_audit(source: list[dict], targets: dict[str, list[dict]], previous: dict) -> dict:
    source_commits = defaultdict(set)
    for row in source:
        source_commits[row["role"]].add(canonical_commit_id(row["commit_id"]))
    roles = sorted(source_commits)
    role_overlap = {f"{a}:{b}": len(source_commits[a] & source_commits[b])
                    for i, a in enumerate(roles) for b in roles[i + 1:]}
    full_commits = set().union(*source_commits.values())
    keys = {r["exact_code_key"] for r in source}
    lexical = {token_set_sha256(lexical_token_set(r["text"])) for r in source}
    projects = {r["project_group"].casefold() for r in source}
    excluded = set(n.casefold() for names in previous["repository_family_exclusions"].values() for n in names)
    excluded.update(n.casefold() for n in previous["opaque_source_project_exclusions"])
    excluded.update(n.casefold() for n in json.loads((V3 / "preparation_manifest.json").read_text())["candidate_projects"])
    datasets = {}
    for name, rows in targets.items():
        datasets[name] = {
            "rows": len(rows), "projects": sorted({r["project_group"] for r in rows}),
            "commits": len({canonical_commit_id(r["commit_id"]) for r in rows}),
            "source_shared_commits": len(full_commits & {canonical_commit_id(r["commit_id"]) for r in rows}),
            "source_shared_exact_keys": len(keys & {r["exact_code_key"] for r in rows}),
            "source_shared_identical_lexical_keys": len(lexical & {r["token_set_sha256"] for r in rows}),
            "source_shared_named_projects": sorted(projects & {r["project_group"].casefold() for r in rows}),
            "qualified_repository_rows": sum(bool(r.get("repository_identity")) for r in rows),
        }
    checks = {"source_roles_shared_commits": role_overlap, "source_recorded_excluded_family_projects": sorted(projects & excluded),
              "datasets": datasets,
              "scope": "Executable known-family/name and exact-identity audit; unrecorded relations and pretraining/event/time overlap are not certified."}
    if any(role_overlap.values()) or projects & excluded:
        raise ValueError(f"Frozen source boundary failure: {checks}")
    if any(d["source_shared_commits"] or d["source_shared_exact_keys"] or d["source_shared_identical_lexical_keys"]
           or d["source_shared_named_projects"] for d in datasets.values()):
        raise ValueError(f"Frozen source/target boundary failure: {checks}")
    return checks


def create_config_and_receipt(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    previous_path = ROOT / "configs/neural_validation_v3.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    source_path = V3 / "source_rows.jsonl"
    source = read_rows(source_path, source=True)
    targets = {"diversevul": read_rows(V3 / "target_rows.jsonl", source=False),
               "megavul": read_rows(V3 / "megavul/target_rows.jsonl", source=False)}
    role_hashes = role_receipts(source)
    settings = {k: v for k, v in previous["fine_tuned_codebert"].items()
                if k not in {"model_id", "revision", "head", "pooling", "update"}}
    settings.update(model_id="Salesforce/codet5-small", revision="b1ee9570c289f21b5922b9c768a1ce12957bf968",
                    architecture="T5EncoderModel only; pretrained decoder unused and excluded",
                    pooling="masked mean final encoder hidden state excluding tokenizer special tokens",
                    head="dropout 0.1 and two-class linear head; normal(0,0.02) weights, zero bias",
                    update="all instantiated encoder and head parameters; no frozen layers; no decoder",
                    feature_dimensions=512, expected_encoder_blocks=6)
    cfg = {"protocol_version": "trace-independent-model-v4.0", "dated_local_amendment": "2026-10-11",
           "status": "Post-v3 cross-family amendment; original/v3 outcomes already known; not preregistration or new-unseen outcomes",
           "known_outcomes": "Original and v3 target results are known. CodeT5 target results have not been generated or compared before this freeze.",
           "protocol_sha256": sha256(PROTOCOL), "previous_config_sha256": sha256(previous_path),
           "detector": "finetuned_codet5_small", "fine_tuned_codet5": settings,
           "source_roles": role_hashes, "source_rows_sha256": sha256(source_path),
           "source": previous["source"], "repository_family_exclusions": previous["repository_family_exclusions"],
           "opaque_source_project_exclusions": previous["opaque_source_project_exclusions"],
           "family_exclusion_scope": previous["family_exclusion_scope"],
           "paths": {"source_rows": str(source_path.relative_to(ROOT)),
                     "diversevul_inputs": str(V3.relative_to(ROOT)),
                     "megavul_inputs": str((V3 / "megavul").relative_to(ROOT)),
                     "megavul_dataset_config": "configs/megavul_validation_v3.json",
                     "model_path": "outputs/kbs-validation-v4/independent_model/models/codet5-small-b1ee9570",
                     "output_root": str(output.relative_to(ROOT))},
           "target_rosters": {d: {"rows": len(rows), "projects": sorted({r["project_group"] for r in rows})}
                              for d, rows in targets.items()},
           "strict_temporal_subset": {"cutoff": "max actual timestamp across every retained source-role commit",
                                      "complete_source_date_coverage_required": True,
                                      "target_rule": "available actual commit timestamp strictly later than source maximum",
                                      "eligibility": {"minimum_total": 200, "minimum_vulnerable": 20, "minimum_safe": 20},
                                      "missing_source_dates": "unavailable; no substitute cutoff or temporal claim"},
           **{k: previous[k] for k in ["weighting", "risk_budgets", "selectors", "cost", "statistics"]},
           "stopping": "All three actual seeds and both complete rosters once; retain failures and unfavorable results; no target tuning",
           "sealing": "Bind all new predictions, full-roster decisions and final failures before evaluation opens original read-only vaults"}
    if CONFIG.exists():
        old = json.loads(CONFIG.read_text(encoding="utf-8"))
        if old != cfg:
            raise FileExistsError("Preserve an existing changed v4 config; do not overwrite a freeze")
    else:
        CONFIG.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dependencies = []
    for path, role in [(previous_path, "previous_config"), (CONFIG, "new_config"), (PROTOCOL, "dated_protocol"),
                       (source_path, "exact_source_rows"), (V3 / "target_rows.jsonl", "diversevul_target_rows"),
                       (V3 / "preparation_manifest.json", "diversevul_preparation"),
                       (V3 / "megavul/target_rows.jsonl", "megavul_target_rows"),
                       (V3 / "megavul/preparation_manifest.json", "megavul_preparation"),
                       (ROOT / "configs/megavul_validation_v3.json", "megavul_dataset_config"),
                       (Path(__file__), "preparation_implementation")]:
        dependencies.append({"path": str(path.resolve()), "role": role, "sha256": sha256(path)})
    receipt = {"schema_version": "trace-independent-model-v4-amendment-freeze-1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(CONFIG),
               "protocol_sha256": sha256(PROTOCOL), "source_rows_sha256": sha256(source_path),
               "source_roles": role_hashes, "dependencies": dependencies,
               "boundary_audit": boundary_audit(source, targets, previous),
               "target_labels_opened": False, "new_model_target_outcomes_compared": False,
               "known_outcomes": cfg["known_outcomes"],
               "target_vault_sha256_from_existing_preparation_only": {
                   d: json.loads((V3 / ("megavul/" if d == "megavul" else "") / "preparation_manifest.json").read_text())["output_hashes"]["target_label_vault.npz"]
                   for d in targets}}
    receipt_path = output / "amendment_freeze.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        for key in ["config_sha256", "protocol_sha256", "source_rows_sha256", "source_roles", "dependencies", "boundary_audit"]:
            if existing[key] != receipt[key]:
                raise ValueError(f"Existing amendment changed: {key}")
        return existing
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "amendment_freeze.sha256").write_text(sha256(receipt_path) + "\n", encoding="ascii")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = create_config_and_receipt(args.output)
    print(json.dumps({k: result[k] for k in ["config_sha256", "protocol_sha256", "source_rows_sha256", "source_roles", "boundary_audit"]}, indent=2))


if __name__ == "__main__":
    main()
