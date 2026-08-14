"""Memory-bounded data preparation for the preregistered extension-v2 study."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from vultriage.data import iter_jsonl_rows


TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>|[-+*/%&|^~!<>=?:;,.(){}\[\]]"
)


def canonicalize_function(text: str) -> str:
    """Apply the exact v2 function-text canonicalization."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n")).strip()


def exact_code_key(text: str) -> str:
    return hashlib.sha256(canonicalize_function(text).encode("utf-8")).hexdigest()


def lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(canonicalize_function(text)))


def stable_id(*parts: object, length: int = 24) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def extension_alias_lookup(config: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    mapping = config["diversevul_target_selection"]["project_aliases"]
    for group, values in mapping.items():
        for value in values:
            if value in aliases:
                raise ValueError(f"Duplicate extension project alias: {value}")
            aliases[value] = group
    return aliases


def extension_project_group(project: str, aliases: dict[str, str]) -> str:
    stripped = project.strip()
    return aliases.get(stripped, stripped)


@dataclass(frozen=True)
class ExtensionRecord:
    """Small row locator; function text deliberately never survives parsing."""

    position: int
    row_id: str
    dataset: str
    source_file: str
    line_number: int
    project: str
    project_group: str
    commit_id: str
    target: int
    exact_code_key: str


def _record_from_row(
    path: Path,
    line_number: int,
    position: int,
    row: dict[str, Any],
    aliases: dict[str, str],
) -> ExtensionRecord | None:
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
        return None
    code_key = exact_code_key(function)
    return ExtensionRecord(
        position=position,
        row_id=stable_id("diversevul", path.name, line_number, code_key),
        dataset="diversevul",
        source_file=path.name,
        line_number=line_number,
        project=project,
        project_group=extension_project_group(project, aliases),
        commit_id=commit_id,
        target=target,
        exact_code_key=code_key,
    )


def iter_diversevul(path: Path, config: dict[str, Any]) -> Iterator[ExtensionRecord]:
    """Parse one JSONL row at a time and yield metadata-only records."""

    aliases = extension_alias_lookup(config)
    with path.open("r", encoding="utf-8") as stream:
        position = 0
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            record = _record_from_row(path, line_number, position, row, aliases)
            if record is None:
                continue
            yield record
            position += 1


class ExtensionIndex:
    """SQLite-backed exact-key index with bounded Python memory use."""

    _COMMIT_INTERVAL = 10_000

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Extension index already exists: {self.path}")
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.executescript(
            """
            CREATE TABLE prime_keys (
                exact_code_key TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE records (
                exact_code_key TEXT PRIMARY KEY,
                first_position INTEGER NOT NULL,
                row_id TEXT NOT NULL UNIQUE,
                dataset TEXT NOT NULL,
                source_file TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                project TEXT NOT NULL,
                project_group TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                conflicting_label INTEGER NOT NULL DEFAULT 0
                    CHECK (conflicting_label IN (0, 1))
            );
            CREATE TABLE labels (
                row_id TEXT PRIMARY KEY,
                target INTEGER NOT NULL CHECK (target IN (0, 1)),
                FOREIGN KEY(row_id) REFERENCES records(row_id)
            ) WITHOUT ROWID;
            CREATE INDEX records_project_group
                ON records(project_group, conflicting_label);
            CREATE INDEX records_first_position
                ON records(first_position);
            """
        )

    def __enter__(self) -> ExtensionIndex:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def close(self) -> None:
        self.connection.close()

    def index_primevul(self, data_dir: Path) -> dict[str, int]:
        counts = Counter()
        for _, _, _, row in iter_jsonl_rows(data_dir):
            function = str(row.get("func", ""))
            if not function.strip():
                counts["missing_function"] += 1
                continue
            key = exact_code_key(function)
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO prime_keys(exact_code_key) VALUES (?)", (key,)
            )
            counts["valid_rows"] += 1
            if cursor.rowcount == 0:
                counts["duplicate_text_key"] += 1
            if counts["valid_rows"] % self._COMMIT_INTERVAL == 0:
                self.connection.commit()
        self.connection.commit()
        counts["unique_text_keys"] = int(
            self.connection.execute("SELECT COUNT(*) FROM prime_keys").fetchone()[0]
        )
        return dict(counts)

    def ingest_diversevul(
        self, path: Path, config: dict[str, Any]
    ) -> dict[str, int]:
        aliases = extension_alias_lookup(config)
        raw_physical_rows = 0
        raw_valid_rows = 0
        missing_or_invalid_rows = 0
        cross_dataset_removed = 0
        same_label_duplicates = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                raw_physical_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
                record = _record_from_row(
                    path, line_number, raw_valid_rows, row, aliases
                )
                if record is None:
                    missing_or_invalid_rows += 1
                    continue
                raw_valid_rows += 1
                if self.connection.execute(
                    "SELECT 1 FROM prime_keys WHERE exact_code_key = ?",
                    (record.exact_code_key,),
                ).fetchone() is not None:
                    cross_dataset_removed += 1
                    continue
                inserted = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO records(
                        exact_code_key, first_position, row_id, dataset, source_file,
                        line_number, project, project_group, commit_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.exact_code_key,
                        record.position,
                        record.row_id,
                        record.dataset,
                        record.source_file,
                        record.line_number,
                        record.project,
                        record.project_group,
                        record.commit_id,
                    ),
                )
                if inserted.rowcount == 1:
                    self.connection.execute(
                        "INSERT INTO labels(row_id, target) VALUES (?, ?)",
                        (record.row_id, record.target),
                    )
                else:
                    previous_target = int(
                        self.connection.execute(
                            """
                            SELECT labels.target
                            FROM records JOIN labels USING (row_id)
                            WHERE records.exact_code_key = ?
                            """,
                            (record.exact_code_key,),
                        ).fetchone()[0]
                    )
                    if previous_target == record.target:
                        same_label_duplicates += 1
                    else:
                        self.connection.execute(
                            """
                            UPDATE records SET conflicting_label = 1
                            WHERE exact_code_key = ?
                            """,
                            (record.exact_code_key,),
                        )
                if raw_valid_rows % self._COMMIT_INTERVAL == 0:
                    self.connection.commit()
        self.connection.commit()
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
        return {
            "raw_physical_rows": raw_physical_rows,
            "raw_valid_rows": raw_valid_rows,
            "missing_or_invalid_rows": missing_or_invalid_rows,
            "cross_dataset_exact_overlaps_removed": cross_dataset_removed,
            "within_dataset_same_label_duplicates_removed": same_label_duplicates,
            "within_dataset_conflicting_keys_quarantined": conflicting,
            "retained_rows": retained,
        }

    def project_counts(self) -> Iterator[dict[str, int | str]]:
        cursor = self.connection.execute(
            """
            SELECT records.project_group, COUNT(*), SUM(labels.target),
                   COUNT(*) - SUM(labels.target)
            FROM records JOIN labels USING (row_id)
            WHERE records.conflicting_label = 0
            GROUP BY records.project_group
            ORDER BY records.project_group COLLATE NOCASE, records.project_group
            """
        )
        for group, total, vulnerable, safe in cursor:
            yield {
                "project_group": str(group),
                "total": int(total),
                "vulnerable": int(vulnerable),
                "safe": int(safe),
            }

    def iter_records(
        self, project_groups: Iterable[str] | None = None
    ) -> Iterator[ExtensionRecord]:
        params: tuple[str, ...] = ()
        where = "records.conflicting_label = 0"
        if project_groups is not None:
            groups = tuple(project_groups)
            if not groups:
                return
            placeholders = ",".join("?" for _ in groups)
            where += f" AND records.project_group IN ({placeholders})"
            params = groups
        cursor = self.connection.execute(
            f"""
            SELECT first_position, row_id, dataset, source_file, line_number,
                   project, project_group, commit_id, labels.target, exact_code_key
            FROM records JOIN labels USING (row_id)
            WHERE {where}
            ORDER BY first_position
            """,
            params,
        )
        for row in cursor:
            yield ExtensionRecord(*row)
