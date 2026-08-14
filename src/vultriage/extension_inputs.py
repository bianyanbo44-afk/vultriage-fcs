"""Leakage-resistant input packaging for extension-v2 external folds."""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from vultriage.data import iter_jsonl_rows, stable_bucket
from vultriage.extension_data import exact_code_key


SOURCE_SPLIT_SALT = "vultriage-extension-source-v2"
SOURCE_ROLE_CODES = {"train": 0, "model_validation": 1, "calibration": 2}
_SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def source_role_v2(commit_id: str) -> str:
    """Assign the frozen 70/10/20 commit-level source partition."""

    bucket = stable_bucket(commit_id, SOURCE_SPLIT_SALT, 100)
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "model_validation"
    return "calibration"


def union_alias_lookup(
    v1_config: dict[str, Any], extension_config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Combine only the two frozen alias maps, preserving case sensitivity."""

    lookup: dict[str, str] = {}
    aliases_by_group: dict[str, list[str]] = {}
    mappings = (
        v1_config["target_groups"],
        extension_config["diversevul_target_selection"]["project_aliases"],
    )
    for mapping in mappings:
        for group, aliases in mapping.items():
            values = [str(group), *(str(alias) for alias in aliases)]
            group_values = aliases_by_group.setdefault(str(group), [])
            for alias in values:
                previous = lookup.get(alias)
                if previous is not None and previous != group:
                    raise ValueError(
                        f"Conflicting frozen project alias {alias!r}: "
                        f"{previous!r} versus {group!r}"
                    )
                lookup[alias] = str(group)
                if alias not in group_values:
                    group_values.append(alias)
    return lookup, {
        group: tuple(values) for group, values in aliases_by_group.items()
    }


def canonical_source_project(project: str, lookup: dict[str, str]) -> str:
    stripped = project.strip()
    return lookup.get(stripped, stripped)


def artifact_name(group: str) -> str:
    if not group or not _SAFE_ARTIFACT_NAME.fullmatch(group):
        raise ValueError(f"Unsafe target-group artifact name: {group!r}")
    return group


@dataclass(frozen=True)
class SourceRecord:
    first_position: int
    feature_position: int
    row_id: str
    dataset: str
    origin_split: str
    source_file: str
    line_number: int
    project: str
    project_group: str
    commit_id: str
    source_role: str
    target: int
    exact_code_key: str


def read_target_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate the already-frozen label-free target metadata."""

    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        forbidden = fields.intersection({"target", "label", "labels", "y"})
        if forbidden:
            raise ValueError(
                "Target metadata contains forbidden label columns: "
                + ", ".join(sorted(forbidden))
            )
        required = {
            "position",
            "row_id",
            "dataset",
            "source_file",
            "line_number",
            "project",
            "project_group",
            "commit_id",
            "exact_code_key",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(
                "Target metadata is missing columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError("Target manifest positions must be contiguous and ordered")
        row_id = row["row_id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate target row_id: {row_id}")
        seen_ids.add(row_id)
    return rows


def selected_target_groups(
    rows: list[dict[str, str]], target_summary: dict[str, Any]
) -> list[str]:
    selected = [str(value) for value in target_summary["selected_project_groups"]]
    if len(selected) != len(set(selected)):
        raise ValueError("Selected target groups are not unique")
    observed = {row["project_group"] for row in rows}
    expected = set(selected)
    if observed != expected:
        raise ValueError(
            f"Target manifest groups differ from frozen selection: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if int(target_summary["selected_rows"]) != len(rows):
        raise ValueError("Target manifest row count differs from frozen summary")
    return selected


class ExtensionSourceIndex:
    """SQLite index for exact source deduplication and cache-position alignment."""

    _COMMIT_INTERVAL = 10_000

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Extension source index already exists: {self.path}")
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.executescript(
            """
            CREATE TABLE cache_locations (
                source_file TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                feature_position INTEGER NOT NULL UNIQUE,
                row_id TEXT NOT NULL UNIQUE,
                project TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                target INTEGER NOT NULL CHECK (target IN (0, 1)),
                PRIMARY KEY(source_file, line_number)
            ) WITHOUT ROWID;
            CREATE TABLE records (
                exact_code_key TEXT PRIMARY KEY,
                first_position INTEGER NOT NULL,
                feature_position INTEGER NOT NULL UNIQUE,
                row_id TEXT NOT NULL UNIQUE,
                dataset TEXT NOT NULL,
                origin_split TEXT NOT NULL,
                source_file TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                project TEXT NOT NULL,
                project_group TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                source_role TEXT NOT NULL,
                conflicting_label INTEGER NOT NULL DEFAULT 0
                    CHECK (conflicting_label IN (0, 1))
            );
            CREATE TABLE labels (
                row_id TEXT PRIMARY KEY,
                target INTEGER NOT NULL CHECK (target IN (0, 1)),
                FOREIGN KEY(row_id) REFERENCES records(row_id)
            ) WITHOUT ROWID;
            CREATE INDEX records_first_position ON records(first_position);
            CREATE INDEX records_project_group ON records(project_group);
            """
        )

    def __enter__(self) -> ExtensionSourceIndex:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def index_feature_manifest(
        self, manifest_path: Path, cache_row_ids_path: Path
    ) -> dict[str, int]:
        row_count = 0
        with (
            gzip.open(manifest_path, "rt", encoding="utf-8", newline="") as stream,
            cache_row_ids_path.open("r", encoding="utf-8") as id_stream,
        ):
            reader = csv.DictReader(stream)
            required = {
                "row_id",
                "source_file",
                "line_number",
                "project",
                "commit_id",
                "target",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "PrimeVul feature manifest is missing columns: "
                    + ", ".join(sorted(missing))
                )
            for feature_position, pair in enumerate(
                zip_longest(reader, id_stream, fillvalue=None)
            ):
                row, cache_id = pair
                if row is None or cache_id is None:
                    raise ValueError(
                        "PrimeVul feature manifest and cache row IDs have different lengths"
                    )
                cache_id = cache_id.rstrip("\r\n")
                if cache_id != row["row_id"]:
                    raise ValueError(
                        f"PrimeVul cache row mismatch at position {feature_position}"
                    )
                target = int(row["target"])
                if target not in (0, 1):
                    raise ValueError("PrimeVul feature manifest contains a nonbinary target")
                self.connection.execute(
                    """
                    INSERT INTO cache_locations(
                        source_file, line_number, feature_position, row_id,
                        project, commit_id, target
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["source_file"],
                        int(row["line_number"]),
                        feature_position,
                        row["row_id"],
                        row["project"],
                        row["commit_id"],
                        target,
                    ),
                )
                row_count += 1
                if row_count % self._COMMIT_INTERVAL == 0:
                    self.connection.commit()
        self.connection.commit()
        return {"feature_manifest_rows": row_count, "cache_row_ids": row_count}

    def ingest_primevul(
        self, data_dir: Path, alias_lookup: dict[str, str]
    ) -> dict[str, int]:
        counts = Counter()
        valid_position = 0
        for origin_split, path, line_number, row in iter_jsonl_rows(data_dir):
            counts["raw_rows"] += 1
            try:
                target = int(row["target"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid target at {path}:{line_number}") from exc
            if target not in (0, 1):
                raise ValueError(f"Nonbinary target at {path}:{line_number}")
            project = str(row.get("project", "")).strip()
            commit_id = str(row.get("commit_id", "")).strip()
            function = str(row.get("func", ""))
            if not project or not commit_id or not function.strip():
                counts["missing_or_invalid_rows"] += 1
                continue
            location = self.connection.execute(
                """
                SELECT feature_position, row_id, project, commit_id, target
                FROM cache_locations WHERE source_file = ? AND line_number = ?
                """,
                (path.name, line_number),
            ).fetchone()
            if location is None:
                raise ValueError(
                    f"PrimeVul row is absent from frozen feature manifest: "
                    f"{path.name}:{line_number}"
                )
            feature_position, row_id, cached_project, cached_commit, cached_target = location
            if (project, commit_id, target) != (
                cached_project,
                cached_commit,
                int(cached_target),
            ):
                raise ValueError(
                    f"PrimeVul raw row differs from feature manifest at "
                    f"{path.name}:{line_number}"
                )
            key = exact_code_key(function)
            inserted = self.connection.execute(
                """
                INSERT OR IGNORE INTO records(
                    exact_code_key, first_position, feature_position, row_id,
                    dataset, origin_split, source_file, line_number, project,
                    project_group, commit_id, source_role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    valid_position,
                    int(feature_position),
                    row_id,
                    "primevul",
                    origin_split,
                    path.name,
                    line_number,
                    project,
                    canonical_source_project(project, alias_lookup),
                    commit_id,
                    source_role_v2(commit_id),
                ),
            )
            if inserted.rowcount == 1:
                self.connection.execute(
                    "INSERT INTO labels(row_id, target) VALUES (?, ?)",
                    (row_id, target),
                )
            else:
                previous_target = int(
                    self.connection.execute(
                        """
                        SELECT labels.target FROM records JOIN labels USING (row_id)
                        WHERE records.exact_code_key = ?
                        """,
                        (key,),
                    ).fetchone()[0]
                )
                if previous_target == target:
                    counts["same_label_duplicates_removed"] += 1
                else:
                    self.connection.execute(
                        """
                        UPDATE records SET conflicting_label = 1
                        WHERE exact_code_key = ?
                        """,
                        (key,),
                    )
            valid_position += 1
            counts["valid_rows"] += 1
            if valid_position % self._COMMIT_INTERVAL == 0:
                self.connection.commit()
        self.connection.commit()
        unused_locations = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM cache_locations
                WHERE NOT EXISTS (
                    SELECT 1 FROM records
                    WHERE records.feature_position = cache_locations.feature_position
                      AND records.conflicting_label = 0
                )
                """
            ).fetchone()[0]
        )
        conflicting = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM records WHERE conflicting_label = 1"
            ).fetchone()[0]
        )
        retained = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM records WHERE conflicting_label = 0"
            ).fetchone()[0]
        )
        counts.update(
            {
                "conflicting_keys_quarantined": conflicting,
                "retained_rows": retained,
                "feature_manifest_rows_not_retained": unused_locations,
            }
        )
        return dict(counts)

    def iter_records(self) -> Iterator[SourceRecord]:
        cursor = self.connection.execute(
            """
            SELECT first_position, feature_position, row_id, dataset, origin_split,
                   source_file, line_number, project, project_group, commit_id,
                   source_role, labels.target, exact_code_key
            FROM records JOIN labels USING (row_id)
            WHERE conflicting_label = 0
            ORDER BY first_position
            """
        )
        for row in cursor:
            yield SourceRecord(*row)


def _gzip_text_writer(path: Path, embedded_name: str):
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename=embedded_name, fileobj=raw, mode="wb", mtime=0
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def write_source_manifest_and_labels(
    records: Iterable[SourceRecord], metadata_path: Path, labels_path: Path
) -> tuple[int, list[SourceRecord]]:
    """Write aligned source metadata and labels without leaking labels into metadata."""

    metadata_fields = [
        "position",
        "feature_position",
        "row_id",
        "dataset",
        "origin_split",
        "source_file",
        "line_number",
        "project",
        "project_group",
        "commit_id",
        "source_role",
        "exact_code_key",
    ]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    retained: list[SourceRecord] = []
    metadata_raw, metadata_gzip, metadata_stream = _gzip_text_writer(
        metadata_path, "source_metadata.csv"
    )
    labels_raw, labels_gzip, labels_stream = _gzip_text_writer(
        labels_path, "source_labels.csv"
    )
    try:
        metadata_writer = csv.DictWriter(metadata_stream, fieldnames=metadata_fields)
        labels_writer = csv.DictWriter(
            labels_stream,
            fieldnames=["position", "feature_position", "row_id", "target"],
        )
        metadata_writer.writeheader()
        labels_writer.writeheader()
        for position, record in enumerate(records):
            row = record.__dict__.copy()
            target = row.pop("target")
            row.pop("first_position")
            row["position"] = position
            metadata_writer.writerow(row)
            labels_writer.writerow(
                {
                    "position": position,
                    "feature_position": record.feature_position,
                    "row_id": record.row_id,
                    "target": target,
                }
            )
            retained.append(record)
    finally:
        metadata_stream.close()
        labels_stream.close()
        metadata_gzip.close()
        labels_gzip.close()
        metadata_raw.close()
        labels_raw.close()
    return len(retained), retained


def write_fold_packages(
    *,
    records: list[SourceRecord],
    target_rows: list[dict[str, str]],
    selected_groups: list[str],
    aliases_by_group: dict[str, tuple[str, ...]],
    source_label_dir: Path,
    target_fold_dir: Path,
    sha256_file: Any,
) -> list[dict[str, Any]]:
    source_label_dir.mkdir(parents=True, exist_ok=False)
    target_fold_dir.mkdir(parents=True, exist_ok=False)
    project_groups = np.asarray([record.project_group for record in records])
    feature_positions = np.asarray(
        [record.feature_position for record in records], dtype=np.int32
    )
    source_positions = np.arange(len(records), dtype=np.int32)
    labels = np.asarray([record.target for record in records], dtype=np.int8)
    roles = np.asarray(
        [SOURCE_ROLE_CODES[record.source_role] for record in records], dtype=np.uint8
    )
    folds: list[dict[str, Any]] = []
    for group in selected_groups:
        name = artifact_name(group)
        include = project_groups != group
        included_positions = source_positions[include]
        included_features = feature_positions[include]
        included_labels = labels[include]
        included_roles = roles[include]
        source_package = source_label_dir / f"{name}.npz"
        np.savez_compressed(
            source_package,
            source_positions=included_positions,
            feature_positions=included_features,
            labels=included_labels,
            role_codes=included_roles,
        )
        target_positions = np.asarray(
            [
                int(row["position"])
                for row in target_rows
                if row["project_group"] == group
            ],
            dtype=np.int32,
        )
        target_package = target_fold_dir / f"{name}.npz"
        np.savez_compressed(target_package, target_positions=target_positions)
        counts: dict[str, dict[str, int]] = {}
        for role, code in SOURCE_ROLE_CODES.items():
            role_mask = included_roles == code
            role_labels = included_labels[role_mask]
            vulnerable = int(role_labels.sum())
            counts[role] = {
                "safe": int(len(role_labels) - vulnerable),
                "vulnerable": vulnerable,
                "total": int(len(role_labels)),
            }
        folds.append(
            {
                "target_group": group,
                "artifact_name": name,
                "frozen_aliases": list(aliases_by_group.get(group, (group,))),
                "source_rows": int(include.sum()),
                "excluded_source_rows": int((~include).sum()),
                "source_partition_counts": counts,
                "target_rows": int(len(target_positions)),
                "source_label_package": source_package.as_posix(),
                "source_label_package_sha256": sha256_file(source_package),
                "target_position_package": target_package.as_posix(),
                "target_position_package_sha256": sha256_file(target_package),
                "target_position_package_contains_labels": False,
            }
        )
    return folds
