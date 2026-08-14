import csv
import gzip
import hashlib
import json
import sqlite3
from dataclasses import fields
from pathlib import Path

from prepare_extension_manifest import write_manifest_and_labels
from vultriage.extension_data import (
    ExtensionIndex,
    ExtensionRecord,
    canonicalize_function,
    exact_code_key,
    extension_alias_lookup,
    extension_project_group,
    iter_diversevul,
    lexical_tokens,
)


def config() -> dict:
    return {
        "diversevul_target_selection": {
            "project_aliases": {"linux": ["linux", "linux-2.6"]}
        }
    }


def record(position: int, text: str, target: int) -> ExtensionRecord:
    return ExtensionRecord(
        position=position,
        row_id=f"row-{position}",
        dataset="diversevul",
        source_file="d.json",
        line_number=position + 1,
        project="p",
        project_group="p",
        commit_id=f"c-{position}",
        target=target,
        exact_code_key=exact_code_key(text),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prime_row(function: str, target: int = 0) -> dict:
    return {
        "func": function,
        "target": target,
        "project": "prime",
        "commit_id": exact_code_key(function)[:12],
    }


def diverse_row(
    function: str, target: int, project: str = "linux", commit_id: str = "c"
) -> dict:
    return {
        "func": function,
        "target": target,
        "project": project,
        "commit_id": commit_id,
    }


def test_canonicalization_normalizes_only_frozen_whitespace_rules():
    assert canonicalize_function("  int x;  \r\n\treturn x;\t\r\n") == "int x;\n\treturn x;"
    assert exact_code_key("int x;\r\n") == exact_code_key("int x;\n")
    assert lexical_tokens("if (x >= 10) x++;") == (
        "if", "(", "x", ">=", "10", ")", "x", "++", ";"
    )


def test_extension_aliases_are_explicit_and_case_sensitive():
    aliases = extension_alias_lookup(config())
    assert extension_project_group("linux-2.6", aliases) == "linux"
    assert extension_project_group("Linux", aliases) == "Linux"


def test_records_are_metadata_only_and_stream_parser_discards_function(tmp_path):
    source = tmp_path / "diversevul.json"
    write_jsonl(source, [diverse_row("int very_large_body(void) { return 1; }", 1)])
    item = next(iter_diversevul(source, config()))
    assert "function" not in {field.name for field in fields(ExtensionRecord)}
    assert not hasattr(item, "function")
    assert item.exact_code_key == exact_code_key(
        "int very_large_body(void) { return 1; }"
    )


def test_sqlite_index_performs_exact_cross_dataset_dedup_on_disk(tmp_path):
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
    overlap_text = "int overlap(void) { return 0; }"
    for name, rows in {
        "primevul_train.jsonl": [prime_row(overlap_text), prime_row("int p(void);")],
        "primevul_valid.jsonl": [prime_row(overlap_text)],
        "primevul_test.jsonl": [],
    }.items():
        write_jsonl(prime_dir / name, rows)

    diverse = tmp_path / "diversevul.json"
    write_jsonl(
        diverse,
        [
            diverse_row(overlap_text, 1, commit_id="overlap"),
            diverse_row("int same(void);", 0, commit_id="same-first"),
            diverse_row("int same(void);\n", 0, commit_id="same-second"),
            diverse_row("int conflict(void);", 0, commit_id="conflict-first"),
            diverse_row("int conflict(void);", 1, commit_id="conflict-second"),
            diverse_row("int kept(void);", 1, project="linux-2.6", commit_id="kept"),
            diverse_row("", 0, commit_id="missing"),
        ],
    )
    database = tmp_path / "index.sqlite"
    with ExtensionIndex(database) as index:
        prime_audit = index.index_primevul(prime_dir)
        audit = index.ingest_diversevul(diverse, config())
        retained = list(index.iter_records())
        counts = list(index.project_counts())

    assert prime_audit == {
        "valid_rows": 3,
        "duplicate_text_key": 1,
        "unique_text_keys": 2,
    }
    assert audit == {
        "raw_physical_rows": 7,
        "raw_valid_rows": 6,
        "missing_or_invalid_rows": 1,
        "cross_dataset_exact_overlaps_removed": 1,
        "within_dataset_same_label_duplicates_removed": 1,
        "within_dataset_conflicting_keys_quarantined": 1,
        "retained_rows": 2,
    }
    assert [item.commit_id for item in retained] == ["same-first", "kept"]
    assert counts == [
        {"project_group": "linux", "total": 2, "vulnerable": 1, "safe": 1}
    ]

    connection = sqlite3.connect(database)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(records)")
        }
        label_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(labels)")
        }
    finally:
        connection.close()
    assert "function" not in columns
    assert "func" not in columns
    assert "target" not in columns
    assert label_columns == {"row_id", "target"}


def test_manifest_and_label_vault_are_streamed_in_identical_order(tmp_path):
    records = [record(8, "int first;", 0), record(2, "int second;", 1)]
    manifest = tmp_path / "manifest.csv.gz"
    labels = tmp_path / "labels.csv.gz"
    assert write_manifest_and_labels(iter(records), manifest, labels) == 2

    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    with gzip.open(labels, "rt", encoding="utf-8", newline="") as stream:
        label_rows = list(csv.DictReader(stream))

    assert "target" not in manifest_rows[0]
    assert [row["position"] for row in manifest_rows] == ["0", "1"]
    assert [row["position"] for row in label_rows] == ["0", "1"]
    assert [row["row_id"] for row in manifest_rows] == [
        row["row_id"] for row in label_rows
    ]
    assert [row["target"] for row in label_rows] == ["0", "1"]


def test_streamed_gzip_outputs_are_byte_reproducible(tmp_path):
    records = [record(0, "int first;", 0), record(1, "int second;", 1)]
    hashes = []
    for run in ("a", "b"):
        manifest = tmp_path / f"manifest-{run}.csv.gz"
        labels = tmp_path / f"labels-{run}.csv.gz"
        write_manifest_and_labels(iter(records), manifest, labels)
        hashes.append(
            (
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
                hashlib.sha256(labels.read_bytes()).hexdigest(),
            )
        )
    assert hashes[0] == hashes[1]
