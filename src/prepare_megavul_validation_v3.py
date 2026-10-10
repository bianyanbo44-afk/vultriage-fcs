"""Prepare the frozen second-source cohort without opening detector outputs.

The complete MegaVul array is streamed.  Historical exact/commit exclusions and
the PrimeVul clone index are reused read-only; DiverseVul lexical keys are built
from every nonempty function in its official raw release.  Target labels are
written only to the separate vault.  No source training cohort is changed.

Use --lexical-cache-path to share the raw-DiverseVul cache between explicitly
amended source freezes.  Its source/algorithm identity is independent of model
roles.  Partial caches commit every 2,000 raw rows and resume after interruption;
complete caches are opened read-only.  Every preparation writes a new immutable
source binding in its own preparation_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import ijson
import numpy as np

from prepare_neural_validation_v3 import (
    ReadOnlyCloneScreen, aliases, project_group, write_json, write_records,
)
from vultriage.data import sha256
from vultriage.extension_data import exact_code_key, stable_id
from vultriage.near_duplicate import lexical_token_set, token_set_sha256


FROZEN_ELIGIBILITY = {"minimum_total": 200, "minimum_vulnerable": 20, "minimum_safe": 20}
REQUIRED_HISTORY_ARRAYS = {"prime_keys", "diverse_keys", "prime_commits", "diverse_commits"}


def log(message: str, output: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (output / "preparation.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def normalized_commit(value: object) -> str:
    # Some historical IDs have ?w=1 or #diff fragments, not another commit.
    return str(value or "").strip().casefold().split("?", 1)[0].split("#", 1)[0]


def megavul_project(name: str, lookup: dict[str, str]) -> str:
    """Use the frozen basename + casefold rule, without fuzzy matching."""
    return project_group(name.strip().rsplit("/", 1)[-1], lookup)


@dataclass(frozen=True)
class RepositoryIdentity:
    """Separate the frozen candidate name from an exact repository identity."""
    candidate_group: str
    namespace: str
    host: str | None
    basis: str
    allowed: bool
    exclusion_reason: str | None


def commit_repository_namespace(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").casefold()
    except ValueError:
        return None, None
    if host not in {"github.com", "gitlab.com", "cgit.freedesktop.org"}:
        return None, host or None
    path = unquote(parsed.path)
    namespace = None
    for marker in ("/-/commit/", "/commit/", "/commits/", "/+/"):
        if marker in path:
            namespace = path.split(marker, 1)[0].strip("/").casefold()
            break
    if namespace and namespace.endswith(".git"):
        namespace = namespace[:-4]
    return (namespace if namespace and "/" in namespace else None), host


def repository_identity(
    name: str, git_url: object, lookup: dict[str, str],
    allowlist: dict[str, frozenset[str]],
) -> RepositoryIdentity:
    raw_name = name.strip().casefold()
    group = megavul_project(name, lookup)
    url_namespace, host = commit_repository_namespace(git_url)
    qualified_name = raw_name if "/" in raw_name else None
    namespace = url_namespace or qualified_name or raw_name
    basis = ("commit_url_namespace" if url_namespace else
             "qualified_repository_name" if qualified_name else "unqualified_dataset_name")
    permitted = allowlist.get(group)
    if permitted is None:
        return RepositoryIdentity(group, namespace, host, basis, True, None)
    identities = {v for v in (qualified_name, url_namespace) if v is not None}
    if not identities:
        return RepositoryIdentity(group, namespace, host, basis, False, "repository_identity_unresolved")
    if not identities.issubset(permitted):
        return RepositoryIdentity(group, namespace, host, basis, False, "repository_identity_not_allowlisted")
    return RepositoryIdentity(group, namespace, host, basis, True, None)


def repository_allowlist(cfg: dict[str, Any], candidates: list[str]) -> dict[str, frozenset[str]]:
    declared = cfg.get("repository_identity_allowlist")
    if not isinstance(declared, dict):
        raise ValueError("An explicit frozen repository_identity_allowlist is required")
    result = {}
    for group, values in declared.items():
        if group not in candidates or not isinstance(values, list) or not values:
            raise ValueError(f"Invalid repository identity allowlist group: {group}")
        if any(not isinstance(v, str) or "/" not in v.strip() for v in values):
            raise ValueError(f"Allowlisted identities must be exact qualified repository names: {group}")
        result[group] = frozenset(v.strip().casefold() for v in values)
    if result.get("core") != frozenset({"dovecot/core"}) or result.get("server") != frozenset({"mariadb/server"}):
        raise ValueError("Frozen core/server identities must remain dovecot/core and mariadb/server")
    return result


def commit_date(value: object) -> tuple[int | None, str | None]:
    if value is None or isinstance(value, bool):
        return None, None
    try:
        epoch = int(value)
        if isinstance(value, float) and value != epoch:
            return None, None
        date = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None, None
    return epoch, date


def map_megavul_row(
    raw: dict[str, Any], array_index: int, source: Path, lookup: dict[str, str],
    allowlist: dict[str, frozenset[str]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Map one original record; never substitute the patch for a positive."""
    if not isinstance(raw.get("is_vul"), bool):
        raise ValueError(f"Nonboolean MegaVul is_vul at array index {array_index}")
    label = int(raw["is_vul"])
    text = raw.get("func_before" if label else "func")
    if not isinstance(text, str) or not text.strip():
        return None, "missing_vulnerable_before" if label else "missing_nonvulnerable_function"
    project = raw.get("repo_name")
    commit = normalized_commit(raw.get("commit_hash"))
    if not isinstance(project, str) or not project.strip() or not commit:
        return None, "missing_project_or_commit"
    key = exact_code_key(text)
    epoch, date = commit_date(raw.get("commit_date"))
    identity = repository_identity(project, raw.get("git_url"), lookup, allowlist or {})
    return dict(
        row_id=stable_id("megavul-v3", source.name, array_index, key),
        dataset="megavul", source_file=source.name,
        # line_number is the 1-based array record locator, not a physical line.
        line_number=array_index + 1, source_array_index=array_index,
        source_pointer=f"/{array_index}", project=project,
        project_group=identity.candidate_group, commit_id=commit,
        repository_identity=identity.namespace, repository_host=identity.host,
        repository_identity_basis=identity.basis, repository_identity_allowed=identity.allowed,
        repository_identity_exclusion_reason=identity.exclusion_reason,
        exact_code_key=key, text=text, label=label,
        commit_date=epoch, date=date,
    ), None


@dataclass
class HistoricalSets:
    prime_keys: set[str]
    diverse_keys: set[str]
    prime_commits: set[str]
    diverse_commits: set[str]

    @classmethod
    def load(cls, path: Path) -> "HistoricalSets":
        with np.load(path, allow_pickle=False) as arrays:
            if not REQUIRED_HISTORY_ARRAYS.issubset(arrays.files):
                raise ValueError("Historical exclusion archive is missing required arrays")
            return cls(
                set(arrays["prime_keys"].tolist()),
                set(arrays["diverse_keys"].tolist()),
                {normalized_commit(v) for v in arrays["prime_commits"].tolist()} - {""},
                {normalized_commit(v) for v in arrays["diverse_commits"].tolist()} - {""},
            )


def expand_abbreviated_commits(
    historical: set[str], observed: set[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Conservatively exclude every full hash matching a stored Git abbreviation.

    This is commit-ID normalization, not project matching.  An ambiguous short
    ID excludes all matching observed hashes instead of selecting one silently.
    """
    result = set(historical)
    audit = []
    for short in sorted(v for v in historical if re.fullmatch(r"[0-9a-f]{7,39}", v)):
        matches = sorted(v for v in observed if len(v) == 40 and v.startswith(short))
        result.update(matches)
        audit.append(dict(historical_id=short, matching_full_hashes=matches))
    return result, audit


class DateAudit:
    def __init__(self) -> None:
        self.total = 0
        self.missing = 0
        self.years: Counter[str] = Counter()
        self.minimum: int | None = None
        self.maximum: int | None = None

    def add(self, row: dict[str, Any]) -> None:
        self.total += 1
        epoch = row["commit_date"]
        if epoch is None:
            self.missing += 1
            return
        self.years[row["date"][:4]] += 1
        self.minimum = epoch if self.minimum is None else min(self.minimum, epoch)
        self.maximum = epoch if self.maximum is None else max(self.maximum, epoch)

    def summary(self) -> dict[str, Any]:
        return dict(
            rows=self.total, missing_or_invalid_date=self.missing,
            years=dict(sorted(self.years.items())),
            minimum_utc=commit_date(self.minimum)[1], maximum_utc=commit_date(self.maximum)[1],
        )


def diverse_lexical_history(
    path: Path, output: Path, input_sha256: str, history: HistoricalSets | None,
    cache_path: Path | None = None, *, checkpoint_rows: int = 2000,
    build_only: bool = False,
) -> tuple[set[str], dict[str, Any]]:
    """Resume a transactional raw-data cache without using labels/model outputs."""
    cache = cache_path or output / "diverse_lexical_history.sqlite"
    receipt = output / "diverse_lexical_history_manifest.json"
    token_module = Path(__file__).parent / "vultriage" / "near_duplicate.py"
    canonical_module = Path(__file__).parent / "vultriage" / "extension_data.py"
    identity = dict(cache_schema_version="raw-diverse-lexical-sqlite-v1",
                    source_sha256=input_sha256,
                    lexical_implementation_sha256=sha256(token_module),
                    canonical_implementation_sha256=sha256(canonical_module))
    if checkpoint_rows < 1:
        raise ValueError("Cache checkpoint interval must be positive")
    progress = dict(processed_rows=0, complete=False, counts={})
    if cache.exists():
        try:
            with sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True) as probe:
                metadata = dict(probe.execute("SELECT key, value FROM metadata"))
        except sqlite3.OperationalError:
            # A hard-killed partial transaction can leave a hot rollback journal.
            # Recovery is allowed only for this owned cache, never the source index.
            if not cache.with_name(cache.name + "-journal").exists():
                raise
            with sqlite3.connect(cache.resolve().as_uri() + "?mode=rw", uri=True) as probe:
                metadata = dict(probe.execute("SELECT key, value FROM metadata"))
        if json.loads(metadata["identity"]) != identity:
            raise ValueError("Lexical cache has another raw source/algorithm identity; preserve it and choose a new cache path")
        progress = json.loads(metadata["progress"])
    complete_at_entry = progress["complete"]
    resumed_from = progress["processed_rows"]
    if not complete_at_entry:
        cache.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cache)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS lexical_keys(hash TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TABLE IF NOT EXISTS exact_keys(hash TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TABLE IF NOT EXISTS commit_ids(commit_id TEXT PRIMARY KEY) WITHOUT ROWID;"
            )
            connection.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)", (json.dumps(identity, sort_keys=True),))
            connection.execute("INSERT OR IGNORE INTO metadata VALUES('progress',?)", (json.dumps(progress),))
            connection.commit()
            counts = Counter(progress["counts"])
            last_processed = resumed_from
            log(f"Raw DiverseVul lexical cache: resume at {resumed_from} committed rows", output)
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line_number <= resumed_from:
                        continue
                    raw = json.loads(line)
                    counts["raw_rows"] += 1
                    commit = normalized_commit(raw.get("commit_id"))
                    if commit:
                        connection.execute("INSERT OR IGNORE INTO commit_ids VALUES(?)", (commit,))
                    text = raw.get("func")
                    if not isinstance(text, str) or not text.strip():
                        counts["empty_function_rows"] += 1
                    else:
                        key = exact_code_key(text)
                        token_hash = token_set_sha256(lexical_token_set(text))
                        connection.execute("INSERT OR IGNORE INTO exact_keys VALUES(?)", (key,))
                        connection.execute("INSERT OR IGNORE INTO lexical_keys VALUES(?)", (token_hash,))
                        counts["nonempty_function_rows"] += 1
                    last_processed = line_number
                    if line_number % checkpoint_rows == 0:
                        checkpoint = dict(processed_rows=line_number, complete=False, counts=dict(counts))
                        connection.execute("UPDATE metadata SET value=? WHERE key='progress'", (json.dumps(checkpoint),))
                        connection.commit()
                    if line_number % 10000 == 0:
                        log(f"Raw DiverseVul lexical history: {line_number} rows (durable checkpoint)", output)
            progress = dict(processed_rows=last_processed, complete=True, counts=dict(counts))
            connection.execute("UPDATE metadata SET value=? WHERE key='progress'", (json.dumps(progress),))
            connection.commit()
        finally:
            # Uncommitted keys and progress roll back together after interruption.
            connection.close()
    with sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        lexical_n = connection.execute("SELECT COUNT(*) FROM lexical_keys").fetchone()[0]
        exact_n = connection.execute("SELECT COUNT(*) FROM exact_keys").fetchone()[0]
        commit_n = connection.execute("SELECT COUNT(*) FROM commit_ids").fetchone()[0]
        if build_only:
            hashes = set()
            archive_coverage = {}
        else:
            hashes = {v for (v,) in connection.execute("SELECT hash FROM lexical_keys")}
            exact_keys = {v for (v,) in connection.execute("SELECT hash FROM exact_keys")}
            commits = {v for (v,) in connection.execute("SELECT commit_id FROM commit_ids")}
            archive_coverage = {}
            if history is not None:
                archive_coverage = dict(exact_keys_missing_from_frozen_archive=len(exact_keys - history.diverse_keys),
                                        commits_missing_from_frozen_archive=len(commits - history.diverse_commits))
                history.diverse_keys.update(exact_keys)
                history.diverse_commits.update(commits)
    result = dict(**identity, counts=progress["counts"], lexical_keys=lexical_n,
                  exact_keys=exact_n, commit_hashes=commit_n, cache_complete=True,
                  resumed_from_row=resumed_from, complete_cache_reused_read_only=complete_at_entry,
                  checkpoint_rows=checkpoint_rows, archive_coverage=archive_coverage,
                  cache_sha256=sha256(cache), cache_path=str(cache.resolve()), source_path=str(path.resolve()),
                  source_model_roles_used=False, raw_labels_used=False, detector_outputs_used=False)
    write_json(receipt, result)
    log(f"Completed raw DiverseVul lexical cache: {lexical_n} unique keys", output)
    return hashes, result


def qualifying_projects(
    candidates: list[str], counts: dict[str, Counter[str]], eligibility: dict[str, int],
) -> set[str]:
    return {group for group in candidates
            if counts[group]["0"] + counts[group]["1"] >= eligibility["minimum_total"]
            and counts[group]["1"] >= eligibility["minimum_vulnerable"]
            and counts[group]["0"] >= eligibility["minimum_safe"]}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    frozen_manifest_path = args.frozen_inputs / "preparation_manifest.json"
    frozen_exclusions_path = args.frozen_inputs / "historical_exclusions.npz"
    if not args.output.resolve().is_relative_to(args.frozen_inputs.resolve()) or args.output.resolve() == args.frozen_inputs.resolve():
        raise ValueError("MegaVul outputs must be in a child of the frozen inputs directory")
    if (args.output / "preparation_manifest.json").exists():
        raise FileExistsError("A completed MegaVul preparation is immutable")
    if args.output.exists() and not args.resume_incomplete:
        raise FileExistsError("Output already exists; use --resume-incomplete only for an interrupted preparation")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if cfg["eligibility"] != FROZEN_ELIGIBILITY:
        raise ValueError("MegaVul eligibility must remain at the frozen 200/20/20 rule")
    candidates = frozen["candidate_projects"]
    if len(candidates) != 62 or len(set(candidates)) != 62:
        raise ValueError("Expected all 62 distinct frozen candidate projects, not the selected subset")
    identity_allowlist = repository_allowlist(cfg, candidates)
    if not frozen["source_target_projects_disjoint"] or not frozen["source_roles_commit_disjoint"]:
        raise ValueError("The existing source preparation does not establish the frozen exclusions")
    lookup = aliases()
    candidate_set = set(candidates)
    args.output.mkdir(parents=True, exist_ok=args.resume_incomplete)
    log("Verifying frozen source inputs and complete MegaVul mirror hash", args.output)
    inputs = {"config": sha256(args.config),
              "frozen_preparation_manifest": sha256(frozen_manifest_path),
              "historical_exclusions": sha256(frozen_exclusions_path),
              "source_config": sha256(args.source_config),
              "megavul": sha256(args.megavul), "diversevul": sha256(args.diversevul),
              "clone_index": sha256(args.clone_index)}
    expected = {"historical_exclusions": frozen["output_hashes"]["historical_exclusions.npz"],
                "source_config": frozen["input_hashes"]["config"],
                "diversevul": frozen["input_hashes"]["diversevul"],
                "clone_index": frozen["input_hashes"]["clone_index"], "megavul": cfg["sha256"]}
    for key, expected_hash in expected.items():
        if inputs[key].casefold() != expected_hash.casefold():
            raise ValueError(f"Changed frozen input: {key}")
    if args.megavul.stat().st_size != cfg["bytes"]:
        raise ValueError("MegaVul byte count differs from the fixed mirror artifact")
    alias_paths = [Path("configs/preregistered_experiment.json"), Path("configs/preregistered_extension_v2.json")]
    for path in alias_paths:
        inputs[f"aliases_{path.name}"] = sha256(path)
    provenance_path = args.megavul.with_suffix(".provenance.json")
    provenance = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        inputs["megavul_acquisition_provenance"] = sha256(provenance_path)
        if provenance.get("sha256", "").casefold() != cfg["sha256"].casefold():
            raise ValueError("Acquisition provenance disagrees with the frozen mirror hash")
        if provenance.get("mirror_revision") != cfg["mirror_revision"]:
            raise ValueError("Acquisition provenance disagrees with the frozen mirror revision")
    history = HistoricalSets.load(frozen_exclusions_path)
    screen = ReadOnlyCloneScreen(args.clone_index)
    try:
        if history.prime_keys - screen.lexical_hashes.keys():
            raise ValueError("The read-only clone index does not cover all frozen PrimeVul exact keys")
        prime_lexical = set(screen.lexical_hashes.values())
        diverse_lexical, diverse_receipt = diverse_lexical_history(
            args.diversevul, args.output, inputs["diversevul"], history,
            getattr(args, "lexical_cache_path", None))
        log("Streaming complete MegaVul array and quarantining global label conflicts", args.output)
        first_labels: dict[str, int] = {}
        conflicts: set[str] = set()
        target_first: dict[str, dict[str, Any]] = {}
        observed_commits: set[str] = set()
        raw_rows = Counter()
        per_project_raw: dict[str, Counter[str]] = defaultdict(Counter)
        repositories: dict[str, set[str]] = defaultdict(set)
        observed_repositories: dict[str, set[str]] = defaultdict(set)
        identity_exclusions: dict[str, Counter[str]] = defaultdict(Counter)
        excluded_repositories: dict[str, Counter[str]] = defaultdict(Counter)
        identity_mappings: dict[str, dict[str, Any]] = {}
        raw_dates: dict[str, DateAudit] = defaultdict(DateAudit)
        all_dates = DateAudit()
        with args.megavul.open("rb") as stream:
            for array_index, raw in enumerate(ijson.items(stream, "item", use_float=True)):
                raw_rows["array_records"] += 1
                row, reason = map_megavul_row(raw, array_index, args.megavul, lookup, identity_allowlist)
                if row is None:
                    raw_rows[reason] += 1
                    if reason == "missing_project_or_commit":
                        # Known code/labels still participate in global conflict
                        # quarantine even if their metadata cannot locate a target.
                        text = raw["func_before" if raw["is_vul"] else "func"]
                        key = exact_code_key(text)
                        label = int(raw["is_vul"])
                        previous = first_labels.get(key)
                        if previous is None:
                            first_labels[key] = label
                        elif previous != label:
                            conflicts.add(key)
                    continue
                raw_rows[f"mapped_label_{row['label']}"] += 1
                observed_commits.add(row["commit_id"])
                all_dates.add(row)
                key = row["exact_code_key"]
                previous = first_labels.get(key)
                if previous is None:
                    first_labels[key] = row["label"]
                elif previous != row["label"]:
                    conflicts.add(key)
                group = row["project_group"]
                if group not in candidate_set:
                    raw_rows["outside_candidate_projects"] += 1
                    continue
                per_project_raw[group][str(row["label"])] += 1
                raw_dates[group].add(row)
                observed_repositories[group].add(row["project"])
                identity_mappings[row["project"]] = dict(
                    candidate_group=group, repository_identity=row["repository_identity"],
                    repository_host=row["repository_host"], basis=row["repository_identity_basis"],
                    allowed=row["repository_identity_allowed"],
                    exclusion_reason=row["repository_identity_exclusion_reason"],
                )
                if not row["repository_identity_allowed"]:
                    reason = row["repository_identity_exclusion_reason"]
                    identity_exclusions[group][reason] += 1
                    excluded_repositories[group][row["project"]] += 1
                    raw_rows[reason] += 1
                    continue
                repositories[group].add(row["project"])
                row.pop("repository_identity_exclusion_reason")
                # One observation per exact key across candidate projects;
                # conflicts from anywhere in the full array remain quarantined.
                if key in target_first:
                    per_project_raw[group]["within_megavul_exact_duplicate"] += 1
                else:
                    target_first[key] = row
                if (array_index + 1) % 10000 == 0:
                    log(f"MegaVul {array_index + 1} records; {len(target_first)} candidate exact keys", args.output)
        history.prime_commits, prime_short = expand_abbreviated_commits(history.prime_commits, observed_commits)
        history.diverse_commits, diverse_short = expand_abbreviated_commits(history.diverse_commits, observed_commits)
        raw_counts: dict[str, Counter[str]] = defaultdict(Counter)
        screened_counts: dict[str, Counter[str]] = defaultdict(Counter)
        excluded: dict[str, Counter[str]] = defaultdict(Counter)
        screened_dates: dict[str, DateAudit] = defaultdict(DateAudit)
        screened = []
        for number, (key, row) in enumerate(target_first.items(), 1):
            group = row["project_group"]
            reason = None
            if key in conflicts:
                reason = "label_conflict"
            else:
                raw_counts[group][str(row["label"])] += 1
                if key in history.prime_keys:
                    reason = "prime_exact_duplicate"
                elif key in history.diverse_keys:
                    reason = "diverse_exact_duplicate"
                elif row["commit_id"] in history.prime_commits:
                    reason = "prime_commit_overlap"
                elif row["commit_id"] in history.diverse_commits:
                    reason = "diverse_commit_overlap"
                else:
                    tokens = lexical_token_set(row["text"])
                    token_hash = token_set_sha256(tokens)
                    if not tokens:
                        reason = "empty_lexical_tokens"
                    elif token_hash in prime_lexical:
                        reason = "prime_lexical_duplicate"
                    elif token_hash in diverse_lexical:
                        reason = "diverse_lexical_duplicate"
                    else:
                        overlap, token_hash, token_count, maximum = screen.overlap(row["text"])
                        if overlap:
                            reason = "source_near_clone"
                        else:
                            row.update(token_set_sha256=token_hash, lexical_token_count=token_count,
                                       maximum_candidate_jaccard=maximum)
            if reason:
                excluded[group][reason] += 1
            else:
                screened.append(row)
                screened_counts[group][str(row["label"])] += 1
                screened_dates[group].add(row)
            if number % 1000 == 0:
                log(f"Screened MegaVul {number}/{len(target_first)} keys; retained {len(screened)}", args.output)
        selected = qualifying_projects(candidates, screened_counts, cfg["eligibility"])
        target = sorted((row for row in screened if row["project_group"] in selected),
                        key=lambda row: (row["project_group"], row["source_array_index"]))
        labels = np.asarray([row.pop("label") for row in target], dtype=np.int8)
        if any(set(row) & {"label", "target", "is_vul", "func_before", "func_after"} for row in target):
            raise ValueError("A target label or label-revealing original field entered model-facing metadata")
        if any(row["exact_code_key"] in history.prime_keys or row["exact_code_key"] in history.diverse_keys for row in target):
            raise ValueError("Historical exact-key overlap survived screening")
        if any(row["commit_id"] in history.prime_commits or row["commit_id"] in history.diverse_commits for row in target):
            raise ValueError("Historical commit overlap survived screening")
        write_records(args.output / "target_rows.jsonl", target)
        np.savez_compressed(args.output / "target_label_vault.npz", labels=labels,
                            row_ids=np.asarray([row["row_id"] for row in target], dtype="U24"),
                            positions=np.arange(len(target), dtype=np.int64))
        positive_commits: dict[str, Counter[str]] = defaultdict(Counter)
        for row, label in zip(target, labels, strict=True):
            if label:
                positive_commits[row["project_group"]][row["commit_id"]] += 1
        reasons = ["label_conflict", "prime_exact_duplicate", "diverse_exact_duplicate",
                   "prime_commit_overlap", "diverse_commit_overlap", "prime_lexical_duplicate",
                   "diverse_lexical_duplicate", "empty_lexical_tokens", "source_near_clone"]
        audit = []
        for group in candidates:
            count = screened_counts[group]
            failed = []
            if count["0"] + count["1"] < 200:
                failed.append("minimum_total")
            if count["1"] < 20:
                failed.append("minimum_vulnerable")
            if count["0"] < 20:
                failed.append("minimum_safe")
            audit.append(dict(
                project_group=group, raw_exact_safe=raw_counts[group]["0"],
                raw_exact_vulnerable=raw_counts[group]["1"],
                screened_safe=count["0"], screened_vulnerable=count["1"],
                selected=group in selected, source_near_clone=excluded[group]["source_near_clone"],
                source_commit_overlap=excluded[group]["prime_commit_overlap"] + excluded[group]["diverse_commit_overlap"],
                label_conflict=excluded[group]["label_conflict"],
                positive_commits=len(positive_commits[group]),
                maximum_positive_commit_count=max(positive_commits[group].values(), default=0),
                raw_candidate_safe=per_project_raw[group]["0"], raw_candidate_vulnerable=per_project_raw[group]["1"],
                within_megavul_exact_duplicate=per_project_raw[group]["within_megavul_exact_duplicate"],
                eligibility_failures=";".join(failed),
                matched_repositories=json.dumps(sorted(repositories[group]), separators=(",", ":")),
                observed_repositories=json.dumps(sorted(observed_repositories[group]), separators=(",", ":")),
                repository_identity_excluded_rows=sum(identity_exclusions[group].values()),
                repository_identity_excluded_repositories=json.dumps(dict(excluded_repositories[group]), separators=(",", ":")),
                **{reason: excluded[group][reason] for reason in reasons if reason not in {"label_conflict", "source_near_clone"}},
            ))
        with (args.output / "candidate_projects.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        write_json(args.output / "commit_date_distribution.json", dict(
            dataset="megavul", semantics="commit author time, not a newly established chronological holdout",
            all_mapped_records=all_dates.summary(),
            raw_candidate_records={g: raw_dates[g].summary() for g in candidates},
            screened_records={g: screened_dates[g].summary() for g in candidates},
            selected_projects=sorted(selected), temporal_validation_claim=False,
        ))
        log(f"Preparation complete: {len(selected)} qualifying projects, {len(target)} rows, {int(labels.sum())} positives", args.output)
        result = dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(), protocol_version=cfg["protocol_version"],
            input_hashes=inputs, source_protocol=dict(config_sha256=inputs["source_config"],
                preparation_manifest_sha256=inputs["frozen_preparation_manifest"],
                protocol_version=frozen["protocol_version"],
                source_fit_policy="reuse the newly frozen source fits; no fitting performed in preparation",
                source_rows_modified=False),
            source_counts=frozen["source_counts"], source_audit=frozen["source_audit"],
            source_train_natural=frozen["source_train_natural"], source_train_sampled=frozen["source_train_sampled"],
            candidate_projects=candidates, selected_projects=sorted(selected), eligibility=cfg["eligibility"],
            target_rows=len(target), target_positive=int(labels.sum()),
            target_metadata_contains_labels=False, source_target_projects_disjoint=True,
            source_roles_commit_disjoint=True, target_outputs_used_for_selection=False,
            raw_counts=dict(raw_rows), within_megavul_conflicting_exact_keys=len(conflicts),
            candidate_unique_exact_keys=len(target_first), screened_counts={g:dict(screened_counts[g]) for g in candidates},
            exclusion_counts={g:dict(excluded[g]) for g in candidates},
            repository_identity_allowlist={g:sorted(v) for g,v in identity_allowlist.items()},
            repository_identity_exclusion_counts={g:dict(identity_exclusions[g]) for g in candidates},
            repository_identity_mappings=identity_mappings,
            checks=dict(full_mirror_hash_verified=True, frozen_source_inputs_verified=True,
                all_62_candidates_used=True, prime_clone_index_covers_all_historical_keys=True,
                repository_identity_allowlist_applied=True,
                historical_exact_keys_excluded=True, historical_identical_lexical_keys_excluded=True,
                historical_commit_hashes_excluded=True, prime_candidate_jaccard_threshold=0.90,
                source_training_changed=False, target_detector_outputs_opened=False),
            commit_id_normalization=dict(query_and_fragment_removed=True,
                abbreviated_prime_ids=prime_short, abbreviated_diverse_ids=diverse_short,
                ambiguous_abbreviations_rule="exclude every matching observed full hash"),
            diverse_raw_lexical_history=diverse_receipt,
            provenance=dict(mirror=cfg["mirror"], mirror_revision=cfg["mirror_revision"],
                mirror_sha256=inputs["megavul"], mirror_bytes=args.megavul.stat().st_size,
                official_byte_equivalence="unknown", third_party_mirror=True,
                acquisition=provenance, full_array_used=True, mirror_test_split_used=False),
            input_paths=dict(megavul=str(args.megavul.resolve()), diversevul=str(args.diversevul.resolve()),
                historical_exclusions=str(frozen_exclusions_path.resolve()), clone_index=str(args.clone_index.resolve()),
                diverse_lexical_cache=diverse_receipt["cache_path"]),
            datasets_reported_separately=True, projects_pooled_across_datasets=False,
            chronological_holdout_claim=False,
            output_hashes={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()},
            wall_seconds=time.perf_counter() - started, preparation_script_sha256=sha256(Path(__file__)),
        )
        write_json(args.output / "preparation_manifest.json", result)
        return result
    finally:
        screen.connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/megavul_validation_v3.json"))
    parser.add_argument("--source-config", type=Path, default=Path("configs/neural_validation_v3.json"))
    parser.add_argument("--megavul", type=Path, default=Path("data/external/megavul_mirror_v3/megavul.json"))
    parser.add_argument("--frozen-inputs", type=Path, default=Path("outputs/neural-validation-v3/inputs"))
    parser.add_argument("--diversevul", type=Path, default=Path(r"F:\CCFA\crc_cpvd_fcs\data\external\diversevul_original\diversevul_20230702.json"))
    parser.add_argument("--clone-index", type=Path, default=Path(r"F:\CCFA\crc_cpvd_fcs\outputs\extension-v2\near-duplicate-v1\near_duplicate_work.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("outputs/neural-validation-v3/inputs/megavul"))
    parser.add_argument("--lexical-cache-path", type=Path,
                        help="Explicit source-independent raw-DiverseVul SQLite cache; partial builds resume, complete builds stay read-only.")
    parser.add_argument("--build-lexical-cache-only", action="store_true",
                        help="Build the durable raw-DiverseVul cache without reading a source model freeze or producing targets.")
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    if args.build_lexical_cache_only:
        if (args.output / "preparation_manifest.json").exists():
            raise FileExistsError("A completed target preparation remains immutable")
        args.output.mkdir(parents=True, exist_ok=True)
        diverse_lexical_history(args.diversevul, args.output, sha256(args.diversevul), None,
                               args.lexical_cache_path, build_only=True)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
