"""Freeze new project-disjoint cohorts before neural-v3 predictions exist.

Raw public data and the label vault stay local. The encoder runner receives only
source labels and label-free target records. The historical clone index is read
through SQLite's read-only mode; no original experiment is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import time
from urllib.parse import unquote
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from vultriage.data import sha256, stable_bucket
from vultriage.extension_data import exact_code_key, stable_id
from vultriage.near_duplicate import (
    deserialize_token_set, lexical_token_set, lsh_band_keys,
    minhash_coefficients, minhash_signature, token_set_sha256,
)


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aliases() -> dict[str, str]:
    result: dict[str, str] = {}
    for path, key in ((Path("configs/preregistered_experiment.json"), "target_groups"),
                      (Path("configs/preregistered_extension_v2.json"), "diversevul_target_selection")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        mapping = cfg[key] if key == "target_groups" else cfg[key]["project_aliases"]
        for group, names in mapping.items():
            for name in [group, *names]:
                result[name.strip().casefold()] = group.strip().casefold()
    return result


def project_group(name: str, lookup: dict[str, str]) -> str:
    name = name.strip().casefold()
    return lookup.get(name, name)


def rank_key(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def canonical_commit_id(value: str) -> str:
    decoded = unquote(str(value)).strip().casefold().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    matches = re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", decoded)
    if len(set(matches)) == 1:
        return matches[0]
    if len(set(matches)) > 1:
        return "ambiguous:" + decoded
    return decoded.rsplit("/", 1)[-1]


def complete_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def historical_commit_overlap(value: str, full: set[str], prefixes: set[str]) -> bool:
    return value in full or any(value.startswith(prefix) for prefix in prefixes)


class ReadOnlyCloneScreen:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.coefficients = minhash_coefficients()
        self.lexical_hashes = dict(self.connection.execute(
            "SELECT exact_code_key, token_set_sha256 FROM documents WHERE dataset='primevul'"))

    @lru_cache(maxsize=2048)
    def token_set(self, doc_id: int) -> frozenset[str]:
        blob = self.connection.execute(
            "SELECT token_blob FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
        return frozenset(deserialize_token_set(blob))

    def overlap(self, text: str) -> tuple[bool, str, int, float]:
        tokens = lexical_token_set(text)
        token_hash = token_set_sha256(tokens)
        size = len(tokens)
        if not size:
            return True, token_hash, size, 1.0
        signature = minhash_signature(tokens, self.coefficients)
        band_keys = lsh_band_keys(signature, 16, 8)
        values = ",".join("(?,?)" for _ in band_keys)
        parameters: list[object] = []
        for band, key in enumerate(band_keys):
            parameters.extend((band, key))
        parameters.extend((math.ceil(0.9 * size), math.floor(size / 0.9)))
        query = (f"WITH requested(band, band_key) AS (VALUES {values}) "
                 "SELECT DISTINCT d.doc_id FROM requested r "
                 "JOIN prime_bands b ON b.band=r.band AND b.band_key=r.band_key "
                 "JOIN documents d ON d.doc_id=b.doc_id "
                 "WHERE d.token_count BETWEEN ? AND ?")
        target = frozenset(tokens)
        maximum = 0.0
        for (doc_id,) in self.connection.execute(query, parameters):
            other = self.token_set(int(doc_id))
            intersection = len(target & other)
            union = size + len(other) - intersection
            similarity = intersection / union if union else 1.0
            maximum = max(maximum, similarity)
            if similarity >= 0.9:
                return True, token_hash, size, similarity
        return False, token_hash, size, maximum


def write_records(path: Path, records: list[dict]) -> None:
    for position, record in enumerate(records):
        record["position"] = position
    if path.exists():
        expected = hashlib.sha256()
        for record in records:
            expected.update((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        if sha256(path).lower() != expected.hexdigest():
            raise ValueError(f"Incomplete-run resume would change an existing record file: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    if args.output.exists() and not args.resume_incomplete:
        raise FileExistsError(args.output)
    if args.output.exists() and (args.output / "preparation_manifest.json").exists():
        raise FileExistsError("Completed preparation cannot be overwritten")
    lookup = aliases()
    candidates = sorted(project_group(row["project_group"], lookup)
                        for row in frozen["project_groups"]
                        if row.get("eligible") and not row.get("selected"))
    if len(candidates) != 62 or len(set(candidates)) != 62:
        raise ValueError("Expected 62 distinct frozen candidate projects")
    candidate_set = set(candidates)
    source_exclusions = set(candidate_set)
    source_exclusions.update(project_group(name, lookup) for names in cfg.get("repository_family_exclusions", {}).values() for name in names)
    source_exclusions.update(project_group(name, lookup) for name in cfg.get("opaque_source_project_exclusions", []))
    inputs = {"config": sha256(args.config), "historical_manifest": sha256(args.historical_manifest)}
    for filename, expected in frozen["primevul_input_sha256"].items():
        actual = sha256(args.primevul / filename)
        if actual.upper() != expected.upper():
            raise ValueError(f"Changed PrimeVul input: {filename}")
        inputs[filename] = actual
    inputs["diversevul"] = sha256(args.diversevul)
    if inputs["diversevul"].upper() != frozen["diversevul_sha256"].upper():
        raise ValueError("Changed DiverseVul input")
    expected_clone_hash = json.loads(args.clone_summary.read_text(encoding="utf-8"))["artifacts"]["work_index"]["sha256"]
    inputs["clone_index"] = sha256(args.clone_index)
    if inputs["clone_index"].upper() != expected_clone_hash.upper():
        raise ValueError("Changed historical clone index")
    args.output.mkdir(parents=True, exist_ok=args.resume_incomplete)
    screen = ReadOnlyCloneScreen(args.clone_index)
    first: dict[str, dict] = {}
    conflicts: set[str] = set()
    all_prime_keys: set[str] = set()
    all_prime_commits: set[str] = set()
    raw_prime_counts = Counter()
    for split in ("train", "valid", "test"):
        path = args.primevul / f"primevul_{split}.jsonl"
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            text = str(row["func"])
            key = exact_code_key(text)
            label = int(row["target"])
            if label not in (0, 1):
                raise ValueError("Nonbinary PrimeVul label")
            group = project_group(str(row["project"]), lookup)
            commit = canonical_commit_id(row["commit_id"])
            all_prime_keys.add(key)
            all_prime_commits.add(commit)
            raw_prime_counts[f"{split}_{label}"] += 1
            previous = first.get(key)
            if previous is not None:
                if previous["label"] != label:
                    conflicts.add(key)
                continue
            first[key] = dict(row_id=stable_id("primevul-v3", path.name, line_number, key),
                              dataset="primevul", source_file=path.name, line_number=line_number,
                              project=str(row["project"]), project_group=group,
                              commit_id=commit, exact_code_key=key, label=label,
                              official_split=split, text=text)
        log(f"Parsed PrimeVul {split}")
    retained = [row for key, row in first.items()
                if key not in conflicts and row["project_group"] not in source_exclusions and complete_sha(row["commit_id"])]
    natural_train = [row for row in retained if row["official_split"] == "train"]
    train_commits = {row["commit_id"] for row in natural_train}
    train_lexical = {screen.lexical_hashes[row["exact_code_key"]] for row in natural_train}
    source_audit = Counter(exact_conflict_keys=len(conflicts),
                          incomplete_source_commit_rows=sum(not complete_sha(row["commit_id"]) for row in first.values()),
                          excluded_candidate_or_family_rows=sum(row["project_group"] in source_exclusions for row in first.values()))
    valid: list[dict] = []
    valid_lexical: set[str] = set()
    for row in retained:
        if row["official_split"] != "valid":
            continue
        if row["commit_id"] in train_commits:
            source_audit["valid_train_commit_overlap"] += 1
            continue
        token_hash = screen.lexical_hashes[row["exact_code_key"]]
        if token_hash in train_lexical or token_hash in valid_lexical:
            source_audit["valid_lexical_duplicate"] += 1
            continue
        valid_lexical.add(token_hash)
        commit_key = row["commit_id"]
        row["role"] = ("model_validation" if stable_bucket(commit_key,
                       cfg["source"]["commit_split_salt"]) < 50 else "calibration")
        valid.append(row)
    positives = [row for row in natural_train if row["label"] == 1]
    negatives = sorted((row for row in natural_train if row["label"] == 0),
                       key=lambda row: rank_key(row["exact_code_key"], cfg["source"]["sampling_salt"]))
    sampled = positives + negatives[:4 * len(positives)]
    sampled.sort(key=lambda row: (row["source_file"], row["line_number"]))
    for row in sampled:
        row["role"] = "train"
    source = sampled + sorted(valid, key=lambda row: row["line_number"])
    source_counts = defaultdict(Counter)
    source_commits = set()
    role_commits = defaultdict(set)
    for row in source:
        source_counts[row["role"]][str(row["label"])] += 1
        source_commits.add(row["commit_id"])
        role_commits[row["role"]].add(row["commit_id"])
    for role in ("train", "model_validation", "calibration"):
        if min(source_counts[role].get("0", 0), source_counts[role].get("1", 0)) < 20:
            raise ValueError(f"Insufficient source classes in {role}")
    for a in role_commits:
        for b in role_commits:
            if a != b and role_commits[a] & role_commits[b]:
                raise ValueError("Source commit roles overlap")
    if {row["project_group"] for row in source} & source_exclusions:
        raise ValueError("Target project or excluded family found in source")
    write_records(args.output / "source_rows.jsonl", source)
    log(f"Frozen source roles: {dict(source_counts)}")
    # Retain global first occurrence and quarantine cross-group conflicts.
    diverse_first: dict[str, tuple[int, str]] = {}
    diverse_conflicts: set[str] = set()
    target_rows: dict[str, dict] = {}
    diverse_keys: set[str] = set()
    diverse_commits: set[str] = set()
    for line_number, line in enumerate(args.diversevul.open(encoding="utf-8"), 1):
        row = json.loads(line)
        if not row.get("func") or not row.get("project") or not row.get("commit_id"):
            continue
        key = exact_code_key(str(row["func"]))
        label = int(row["target"])
        commit = canonical_commit_id(row["commit_id"])
        diverse_keys.add(key)
        diverse_commits.add(commit)
        if key in all_prime_keys:
            continue
        group = project_group(str(row["project"]), lookup)
        previous = diverse_first.get(key)
        if previous is not None:
            if previous[0] != label:
                diverse_conflicts.add(key)
            continue
        diverse_first[key] = (label, group)
        if group not in candidate_set:
            continue
        target_rows[key] = dict(row_id=stable_id("diversevul-v3", args.diversevul.name, line_number, key),
                               dataset="diversevul", source_file=args.diversevul.name,
                               line_number=line_number, project=str(row["project"]), project_group=group,
                               commit_id=commit, exact_code_key=key, label=label, text=str(row["func"]))
    raw_counts = defaultdict(Counter)
    screened_counts = defaultdict(Counter)
    excluded_counts = defaultdict(Counter)
    screened: list[dict] = []
    historical_full = {value for value in all_prime_commits if complete_sha(value)}
    historical_prefixes = {value for value in all_prime_commits if re.fullmatch(r"[0-9a-f]{7,39}", value)}
    for number, (key, row) in enumerate(target_rows.items(), 1):
        group = row["project_group"]
        if key in diverse_conflicts:
            excluded_counts[group]["label_conflict"] += 1
            continue
        raw_counts[group][str(row["label"])] += 1
        if not complete_sha(row["commit_id"]):
            excluded_counts[group]["incomplete_commit_id"] += 1
            continue
        if historical_commit_overlap(row["commit_id"], historical_full, historical_prefixes):
            excluded_counts[group]["source_commit_overlap"] += 1
            continue
        overlap, token_hash, token_count, maximum = screen.overlap(row["text"])
        if overlap:
            excluded_counts[group]["source_near_clone"] += 1
            continue
        row["token_set_sha256"] = token_hash
        row["lexical_token_count"] = token_count
        row["maximum_candidate_jaccard"] = maximum
        screened.append(row)
        screened_counts[group][str(row["label"])] += 1
        if number % 1000 == 0:
            log(f"Screened target {number}/{len(target_rows)}; retained {len(screened)}")
    eligibility = cfg["cohort"]["post_screen_eligibility"]
    selected = {group for group in candidates
                if sum(screened_counts[group].values()) >= eligibility["minimum_total"]
                and screened_counts[group]["1"] >= eligibility["minimum_vulnerable"]
                and screened_counts[group]["0"] >= eligibility["minimum_safe"]}
    target = sorted((row for row in screened if row["project_group"] in selected),
                    key=lambda row: (row["project_group"], row["line_number"]))
    if len(selected) < 10:
        raise ValueError("Fewer than ten eligible screened external projects")
    labels = np.asarray([row.pop("label") for row in target], dtype=np.int8)
    write_records(args.output / "target_rows.jsonl", target)
    if any("label" in row or "target" in row for row in target):
        raise ValueError("Label leaked into target records")
    np.savez_compressed(args.output / "target_label_vault.npz", labels=labels,
                        row_ids=np.asarray([row["row_id"] for row in target]),
                        positions=np.arange(len(target), dtype=np.int64))
    np.savez_compressed(args.output / "historical_exclusions.npz",
                        prime_keys=np.asarray(sorted(all_prime_keys)),
                        diverse_keys=np.asarray(sorted(diverse_keys)),
                        prime_commits=np.asarray(sorted(all_prime_commits)),
                        diverse_commits=np.asarray(sorted(diverse_commits)))
    candidate_audit = []
    for group in candidates:
        positives_by_commit = Counter(row["commit_id"] for row in target
                                      if row["project_group"] == group
                                      and labels[row["position"]] == 1)
        candidate_audit.append(dict(project_group=group,
            raw_exact_safe=raw_counts[group]["0"], raw_exact_vulnerable=raw_counts[group]["1"],
            screened_safe=screened_counts[group]["0"], screened_vulnerable=screened_counts[group]["1"],
            selected=group in selected, source_near_clone=excluded_counts[group]["source_near_clone"],
            source_commit_overlap=excluded_counts[group]["source_commit_overlap"],
            incomplete_commit_id=excluded_counts[group]["incomplete_commit_id"],
            label_conflict=excluded_counts[group]["label_conflict"],
            positive_commits=len(positives_by_commit),
            maximum_positive_commit_count=max(positives_by_commit.values(), default=0)))
    with (args.output / "candidate_projects.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_audit[0]))
        writer.writeheader()
        writer.writerows(candidate_audit)
    output_hashes = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_version=cfg["protocol_version"], input_hashes=inputs,
                  source_counts=dict(source_counts), source_audit=dict(source_audit),
                  source_train_natural=len(natural_train), source_train_sampled=len(sampled),
                  candidate_projects=candidates, selected_projects=sorted(selected),
                  excluded_source_project_groups=sorted(source_exclusions),
                  source_commit_normalization="global unambiguous 40-hex SHA; query/fragment removed; incomplete source ids excluded",
                  target_commit_screen="all historical PrimeVul full SHA and short-prefix matches, not just retained source",
                  target_rows=len(target), target_positive=int(labels.sum()),
                  target_metadata_contains_labels=False, source_target_projects_disjoint=True,
                  source_roles_commit_disjoint=True, target_outputs_used_for_selection=False,
                  output_hashes=output_hashes, wall_seconds=time.perf_counter() - started,
                  preparation_script_sha256=sha256(Path(__file__)))
    write_json(args.output / "preparation_manifest.json", result)
    screen.connection.close()
    log(f"Preparation complete: {len(selected)} projects, {len(target)} rows, {int(labels.sum())} positives")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/neural_validation_v3.json"))
    parser.add_argument("--primevul", type=Path, required=True)
    parser.add_argument("--diversevul", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--clone-index", type=Path, required=True)
    parser.add_argument("--clone-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="Resume an interrupted preparation; completed manifests remain immutable.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
