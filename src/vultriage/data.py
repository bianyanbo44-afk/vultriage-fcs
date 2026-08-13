"""PrimeVul metadata preparation and frozen split assignment."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


FILE_ORDER = (
    ("train", "primevul_train.jsonl"),
    ("valid", "primevul_valid.jsonl"),
    ("test", "primevul_test.jsonl"),
)


@dataclass(frozen=True)
class ManifestRecord:
    row_id: str
    origin_split: str
    source_file: str
    line_number: int
    project: str
    project_group: str
    commit_id: str
    target: int
    code_hash: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def alias_lookup(config: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, aliases in config["target_groups"].items():
        for alias in aliases:
            if alias in lookup:
                raise ValueError(f"Duplicate project alias in configuration: {alias}")
            lookup[alias] = group
    return lookup


def canonical_project(project: str, lookup: dict[str, str]) -> str:
    project = project.strip()
    return lookup.get(project, project)


def stable_bucket(value: str, salt: str, buckets: int = 100) -> int:
    payload = f"{salt}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets


def source_role(commit_id: str, config: dict[str, Any]) -> str:
    partition = config["source_partition"]
    bucket = stable_bucket(commit_id, config["split_salt"], 100)
    if bucket < int(partition["train_end"]):
        return "train"
    if bucket < int(partition["model_validation_end"]):
        return "model_validation"
    return "calibration"


def iter_jsonl_rows(data_dir: Path) -> Iterator[tuple[str, Path, int, dict[str, Any]]]:
    for split, filename in FILE_ORDER:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
                yield split, path, line_number, row


def _validated_record(
    split: str,
    path: Path,
    line_number: int,
    row: dict[str, Any],
    aliases: dict[str, str],
) -> ManifestRecord:
    target = int(row["target"])
    if target not in (0, 1):
        raise ValueError(f"Nonbinary target at {path}:{line_number}: {target}")
    project = str(row.get("project", "")).strip()
    commit_id = str(row.get("commit_id", "")).strip()
    function = str(row.get("func", ""))
    if not project or not commit_id or not function.strip():
        raise ValueError(f"Missing project, commit, or function at {path}:{line_number}")
    code_hash = str(row.get("hash", "")).strip()
    if not code_hash:
        normalized = "\n".join(line.rstrip() for line in function.splitlines()).strip()
        code_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    row_payload = f"{path.name}:{line_number}:{code_hash}"
    row_id = hashlib.sha256(row_payload.encode("utf-8")).hexdigest()[:24]
    return ManifestRecord(
        row_id=row_id,
        origin_split=split,
        source_file=path.name,
        line_number=line_number,
        project=project,
        project_group=canonical_project(project, aliases),
        commit_id=commit_id,
        target=target,
        code_hash=code_hash,
    )


def collect_deduplicated_records(
    data_dir: Path, config: dict[str, Any]
) -> tuple[list[ManifestRecord], dict[str, Any]]:
    aliases = alias_lookup(config)
    first_by_hash: dict[str, ManifestRecord] = {}
    conflicts: set[str] = set()
    duplicate_same_label = 0

    for split, path, line_number, row in iter_jsonl_rows(data_dir):
        record = _validated_record(split, path, line_number, row, aliases)
        previous = first_by_hash.get(record.code_hash)
        if previous is None:
            first_by_hash[record.code_hash] = record
        elif previous.target == record.target:
            duplicate_same_label += 1
        else:
            conflicts.add(record.code_hash)

    records = [
        record for code_hash, record in first_by_hash.items() if code_hash not in conflicts
    ]
    audit = {
        "raw_rows": len(first_by_hash) + duplicate_same_label,
        "unique_hashes_seen": len(first_by_hash),
        "same_label_duplicates_removed": duplicate_same_label,
        "conflicting_hashes_quarantined": len(conflicts),
        "retained_rows": len(records),
    }
    return records, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_manifest(records: Iterable[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in ManifestRecord.__dataclass_fields__.values()]
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="split_manifest.csv", fileobj=raw_stream, mode="wb", mtime=0
        ) as compressed_stream:
            with io.TextIOWrapper(
                compressed_stream, encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow(record.__dict__)


def summarize_records(
    records: list[ManifestRecord], config: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    official = Counter((record.origin_split, record.target) for record in records)
    targets: dict[str, Any] = {}
    eligibility = config["target_eligibility"]
    for group in config["target_groups"]:
        rows = [record for record in records if record.project_group == group]
        vulnerable = sum(record.target for record in rows)
        safe = len(rows) - vulnerable
        eligible = (
            len(rows) >= int(eligibility["minimum_total"])
            and vulnerable >= int(eligibility["minimum_vulnerable"])
            and safe >= int(eligibility["minimum_safe"])
        )
        targets[group] = {
            "total": len(rows),
            "vulnerable": vulnerable,
            "safe": safe,
            "eligible": eligible,
        }

    source_roles = Counter(
        source_role(record.commit_id, config) for record in records
    )
    return {
        "protocol_version": config["protocol_version"],
        "deduplication": audit,
        "official_split_counts": {
            split: {
                "safe": official[(split, 0)],
                "vulnerable": official[(split, 1)],
                "total": official[(split, 0)] + official[(split, 1)],
            }
            for split, _ in FILE_ORDER
        },
        "global_source_role_counts": dict(sorted(source_roles.items())),
        "target_groups": targets,
    }


def iter_manifest(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)
