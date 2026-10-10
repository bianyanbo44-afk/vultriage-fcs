"""Metadata-only event/date/family audit on exact retained identities; no outcome comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import ijson

from prepare_independent_model_v4 import ROOT, V3, CONFIG, read_rows
from vultriage.data import sha256
from vultriage.extension_data import exact_code_key
from vultriage.neural_audit import canonical_commit_id

CVE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
DATE_FIELDS = ("commit_date", "author_date", "committer_date", "commit_timestamp")
CVE_FIELDS = ("cve_id", "cve_ids", "cve")


def actual_timestamp(value) -> int | None:
    """Accept timestamp/offset-qualified ISO values, never infer dates from years."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        stamp = int(value)
        return stamp if 0 < stamp < 4102444800 else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None
            return int(parsed.timestamp())
        except ValueError:
            return None
    return None


def cve_values(raw: dict) -> list[str]:
    result = set()
    for key in CVE_FIELDS:
        value = raw.get(key)
        if isinstance(value, str):
            result.update(m.group().upper() for m in CVE.finditer(value))
        elif isinstance(value, list):
            result.update(m.group().upper() for item in value if isinstance(item, str) for m in CVE.finditer(item))
    return sorted(result)


def repository_identity(value: str) -> str | None:
    parsed = urlparse(str(value))
    if not parsed.netloc:
        return None
    path = parsed.path.strip("/")
    if "/-/commit/" in path:
        namespace = path.split("/-/commit/", 1)[0]
    elif "/commit/" in path:
        namespace = path.split("/commit/", 1)[0]
    elif "/commits/" in path:
        namespace = path.split("/commits/", 1)[0]
    else:
        return None
    if not namespace or "/" not in namespace:
        return None
    return parsed.netloc.casefold() + "/" + namespace.removesuffix(".git").casefold()


def extract_metadata(raw: dict, *, provenance: str) -> dict:
    timestamps = {actual_timestamp(raw.get(k)) for k in DATE_FIELDS}
    timestamps.discard(None)
    repositories = set()
    for key in ("git_url", "commit_url", "commit_id"):
        value = raw.get(key)
        if isinstance(value, str):
            identity = repository_identity(value)
            if identity:
                repositories.add(identity)
    return {"dates": sorted(timestamps), "cve_ids": cve_values(raw), "repositories": sorted(repositories),
            "cve_message_mentions": sorted({m.group().upper() for k in ("message", "commit_msg")
                                             for m in CVE.finditer(str(raw.get(k, "")))}),
            "present_date_fields": [k for k in DATE_FIELDS if k in raw],
            "present_cve_fields": [k for k in CVE_FIELDS if k in raw],
            "provenance": [provenance]}


def merge_metadata(primary: dict, supplement: dict | None) -> dict:
    if supplement is None:
        return primary
    return {key: sorted(set(primary[key]) | set(supplement[key])) for key in primary}


def strict_temporal_rule(source_records: list[dict], target_records: list[dict]) -> dict:
    source_commits = defaultdict(set)
    for row in source_records:
        source_commits[row["commit_id"]].update(row["dates"])
    complete = bool(source_commits) and all(len(dates) == 1 for dates in source_commits.values())
    cutoff = max(next(iter(dates)) for dates in source_commits.values()) if complete else None
    eligible = [r["row_id"] for r in target_records if cutoff is not None and len(r["dates"]) == 1 and r["dates"][0] > cutoff]
    return {"available": complete, "cutoff_timestamp": cutoff,
            "cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat() if cutoff is not None else None,
            "source_commits": len(source_commits), "source_unique_dated_commits": sum(len(dates) == 1 for dates in source_commits.values()),
            "source_missing_date_commits": sum(not dates for dates in source_commits.values()),
            "source_conflicting_date_commits": sum(len(dates) > 1 for dates in source_commits.values()),
            "eligible_row_ids": eligible,
            "reason": "complete-source dates, strict target timestamp > source maximum" if complete
                      else "strict chronology unavailable: incomplete or conflicting actual source commit dates; no substitute cutoff"}


def summarize(rows: list[dict]) -> dict:
    commits = defaultdict(list)
    for row in rows:
        commits[row["commit_id"]].append(row)
    commit_dates = {commit: set(t for row in members for t in row["dates"]) for commit, members in commits.items()}
    dates = [next(iter(v)) for v in commit_dates.values() if len(v) == 1]
    cves = set(c for row in rows for c in row["cve_ids"])
    return {"rows": len(rows), "commits": len(commits),
            "rows_with_unique_actual_date": sum(len(r["dates"]) == 1 for r in rows),
            "commits_with_unique_actual_date": len(dates), "conflicting_date_commits": sum(len(v) > 1 for v in commit_dates.values()),
            "minimum_actual_date_utc": datetime.fromtimestamp(min(dates), timezone.utc).isoformat() if dates else None,
            "maximum_actual_date_utc": datetime.fromtimestamp(max(dates), timezone.utc).isoformat() if dates else None,
            "rows_with_explicit_cve_metadata": sum(bool(r["cve_ids"]) for r in rows), "unique_observed_cves": len(cves),
            "rows_with_cve_message_mentions": sum(bool(r["cve_message_mentions"]) for r in rows),
            "rows_with_qualified_repository_metadata": sum(bool(r["repositories"]) for r in rows),
            "unique_qualified_repositories": sorted({repo for row in rows for repo in row["repositories"]})}


def audit(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Preserve previous metadata-audit evidence")
    output.mkdir(parents=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = read_rows(V3 / "source_rows.jsonl", source=True)
    targets = {"diversevul": read_rows(V3 / "target_rows.jsonl", source=False),
               "megavul": read_rows(V3 / "megavul/target_rows.jsonl", source=False)}
    primary = {}
    schema_counts, raw_dependencies = {}, []
    paths = {"primevul_train.jsonl": Path(r"F:\CCFA\crc_cpvd_fcs\data\external\primevul_original\primevul_train.jsonl"),
             "primevul_valid.jsonl": Path(r"F:\CCFA\crc_cpvd_fcs\data\external\primevul_original\primevul_valid.jsonl"),
             "diversevul_20230702.json": Path(r"F:\CCFA\crc_cpvd_fcs\data\external\diversevul_original\diversevul_20230702.json")}
    prep = json.loads((V3 / "preparation_manifest.json").read_text())
    for filename, path in paths.items():
        expected = prep["input_hashes"]["diversevul" if filename.startswith("diverse") else filename]
        if sha256(path) != expected:
            raise ValueError(f"Historical raw metadata file changed: {filename}")
        raw_dependencies.append({"path": str(path), "sha256": expected, "role": "exact_retained_pointer_metadata"})
        cohort = targets["diversevul"] if filename.startswith("diverse") else source
        wanted = {r["line_number"]: r for r in cohort if r["source_file"] == filename}
        keys = Counter()
        matched = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = wanted.get(line_number)
                if row is None:
                    continue
                raw = json.loads(line)
                if canonical_commit_id(raw["commit_id"]) != row["commit_id"] or exact_code_key(raw["func"]) != row["exact_code_key"]:
                    raise ValueError("Raw metadata pointer does not resolve to the retained exact row identity")
                keys.update(raw.keys())
                primary[row["row_id"]] = extract_metadata(raw, provenance=f"{filename}:line:{line_number}")
                matched += 1
        if matched != len(wanted):
            raise ValueError("Some retained row pointers are unavailable")
        schema_counts[filename] = {"retained_rows": matched, "observed_field_counts": dict(keys),
                                   "native_actual_date_rows": sum(bool(primary[r["row_id"]]["dates"]) for r in wanted.values()),
                                   "native_explicit_cve_rows": sum(bool(primary[r["row_id"]]["cve_ids"]) for r in wanted.values())}
    mega_path = ROOT / "data/external/megavul_mirror_v3/megavul.json"
    mega_cfg = json.loads((ROOT / "configs/megavul_validation_v3.json").read_text())
    if sha256(mega_path) != mega_cfg["sha256"].upper():
        raise ValueError("The pinned MegaVul mirror metadata file changed")
    raw_dependencies.append({"path": str(mega_path), "sha256": mega_cfg["sha256"].upper(),
                             "role": "native_MegaVul_metadata_and_exact_full_SHA_supplement"})
    relevant_commits = {r["commit_id"] for r in source + targets["diversevul"]}
    supplemental = {}
    wanted_mega = {r["source_array_index"]: r for r in targets["megavul"]}
    total, matches = 0, 0
    with mega_path.open("rb") as stream:
        for array_index, raw in enumerate(ijson.items(stream, "item")):
            total += 1
            commit = str(raw.get("commit_hash", "")).casefold()
            if commit in relevant_commits:
                meta = extract_metadata(raw, provenance=f"pinned_MegaVul_exact_full_SHA:{commit}")
                supplemental[commit] = merge_metadata(supplemental.get(commit, meta), meta)
            row = wanted_mega.get(array_index)
            if row is not None:
                # Identity is checked with the mapping already frozen in v3; outcome values are not inspected here.
                if commit != row["commit_id"] or raw.get("git_url") is None:
                    raise ValueError("MegaVul pointer did not resolve to the frozen complete commit identity")
                primary[row["row_id"]] = extract_metadata(raw, provenance=f"megavul.json:pointer:/{array_index}")
                matches += 1
            if total % 10000 == 0:
                print(f"Metadata-only MegaVul scan: {total} raw records", flush=True)
    if matches != len(targets["megavul"]):
        raise ValueError("The complete MegaVul metadata roster was not recovered")
    records = {}
    for dataset, rows in {"source": source, **targets}.items():
        enriched = []
        for row in rows:
            meta = primary[row["row_id"]]
            if dataset != "megavul":
                meta = merge_metadata(meta, supplemental.get(row["commit_id"]))
            enriched.append({"position": row["position"], "row_id": row["row_id"], "dataset": row["dataset"],
                             "role": row.get("role", "target"), "project_group": row["project_group"],
                             "commit_id": row["commit_id"], "source_file": row["source_file"], "line_number": row["line_number"], **meta})
        records[dataset] = enriched
        with (output / f"{dataset}_metadata.jsonl").open("w", encoding="utf-8") as stream:
            for row in enriched:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    roles = {role: [r for r in records["source"] if r["role"] == role] for role in ("train", "model_validation", "calibration")}
    subsets = {**roles, "source_all": records["source"], **{d: records[d] for d in targets}}
    events = {d: {cve for row in rows for cve in row["cve_ids"]} for d, rows in subsets.items()}
    repositories = {d: {repo for row in rows for repo in row["repositories"]} for d, rows in subsets.items()}
    event_overlap = []
    names = list(subsets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a == "source_all" and b in roles or b == "source_all" and a in roles:
                continue
            event_overlap.append({"left": a, "right": b, "observed_shared_cves": sorted(events[a] & events[b]),
                                  "shared_qualified_repositories": sorted(repositories[a] & repositories[b]),
                                  "event_independence_certified": False,
                                  "interpretation": "Observed identifiers only; missing event metadata cannot certify disjointness"})
    temporal = {dataset: strict_temporal_rule(records["source"], records[dataset]) for dataset in targets}
    with (output / "strict_temporal_roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "project", "position", "row_id", "commit_id", "actual_commit_timestamp",
                                                        "source_cutoff_timestamp", "strict_subset_available", "strict_temporal_eligible", "reason"])
        writer.writeheader()
        for dataset in targets:
            eligible = set(temporal[dataset]["eligible_row_ids"])
            for row in records[dataset]:
                writer.writerow({"dataset": dataset, "project": row["project_group"], "position": row["position"], "row_id": row["row_id"],
                                 "commit_id": row["commit_id"], "actual_commit_timestamp": row["dates"][0] if len(row["dates"]) == 1 else "",
                                 "source_cutoff_timestamp": temporal[dataset]["cutoff_timestamp"], "strict_subset_available": temporal[dataset]["available"],
                                 "strict_temporal_eligible": row["row_id"] in eligible, "reason": temporal[dataset]["reason"]})
    report = {"schema_version": "trace-independent-model-v4-metadata-audit-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256(CONFIG), "source_rows_sha256": sha256(V3 / "source_rows.jsonl"),
              "target_outcomes_compared": False, "target_vault_opened": False,
              "raw_metadata_audit": "Raw files contain label fields; only commit/row identity, date, CVE, repository and message-mention fields used; no label values used for metadata rules",
              "native_schema_availability": schema_counts,
              "exact_SHA_MegaVul_supplement": {"raw_records_scanned": total, "matched_relevant_commits": len(supplemental),
                                              "ambiguity_rule": "Multiple distinct actual dates for one commit are unavailable; explicit CVE ids are unioned and provenance retained"},
              "availability": {name: summarize(rows) for name, rows in subsets.items()},
              "observed_identifier_overlap": event_overlap, "strict_temporal_subsets": temporal,
              "event_isolation_claim": "Not established; source native CVE metadata absent and exact-SHA supplemental coverage may be incomplete",
              "repository_family_claim": "Recorded exclusions and observed qualified identities only; no exhaustive fork/shared-component/pretraining guarantee",
              "raw_dependencies": raw_dependencies, "script_sha256": sha256(Path(__file__)),
              "report_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "metadata_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "metadata_audit.sha256").write_text(sha256(output / "metadata_audit.json") + "\n", encoding="ascii")
    text = ["# Independent-model v4 metadata audit", "", "No new target outcomes were compared; the original target-label vaults were not opened.", "",
            "Exact pointers/full-SHA joins recovered observed raw metadata. Missing or conflicting dates are unavailable, not estimated; CVE message mentions are separate from explicit CVE fields.", "",
            "| Role/dataset | Functions | Commits | Uniquely dated commits | Functions with explicit CVE metadata |", "|---|---:|---:|---:|---:|"]
    for name, value in report["availability"].items():
        text.append(f"| {name} | {value['rows']} | {value['commits']} | {value['commits_with_unique_actual_date']} | {value['rows_with_explicit_cve_metadata']} |")
    text.extend(["", "Strict chronology:"])
    for name, value in temporal.items():
        text.append(f"- {name}: {value['reason']}; source dated commits {value['source_unique_dated_commits']}/{value['source_commits']}; eligible functions {len(value['eligible_row_ids'])}.")
    text.extend(["", "Event separation is not certified from incomplete observed identifiers. Qualified namespaces and the known-family exclusion list do not prove exhaustive repository-family or pretraining independence.", ""])
    (output / "metadata_audit.md").write_text("\n".join(text), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/independent_model/metadata-audit")
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"availability": result["availability"], "strict_temporal_subsets": result["strict_temporal_subsets"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
