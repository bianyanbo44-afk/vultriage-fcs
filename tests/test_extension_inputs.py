import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from prepare_extension_hashing_cache import build_hashing_cache
from vultriage.data import sha256, stable_bucket
from vultriage.extension_data import exact_code_key
from vultriage.extension_inputs import (
    SOURCE_ROLE_CODES,
    SOURCE_SPLIT_SALT,
    ExtensionSourceIndex,
    read_target_manifest,
    source_role_v2,
    union_alias_lookup,
    write_fold_packages,
    write_source_manifest_and_labels,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prime_row(function: str, target: int, project: str, commit_id: str) -> dict:
    return {
        "func": function,
        "target": target,
        "project": project,
        "commit_id": commit_id,
    }


def test_union_aliases_and_v2_source_roles_are_frozen():
    v1 = {"target_groups": {"linux": ["linux", "linux-2.6"]}}
    v2 = {
        "diversevul_target_selection": {
            "project_aliases": {"linux": ["linux", "linux-2.6"]}
        }
    }
    lookup, by_group = union_alias_lookup(v1, v2)
    assert lookup["linux-2.6"] == "linux"
    assert by_group["linux"] == ("linux", "linux-2.6")
    commit = "frozen-commit"
    bucket = stable_bucket(commit, SOURCE_SPLIT_SALT, 100)
    expected = (
        "train" if bucket < 70 else "model_validation" if bucket < 80 else "calibration"
    )
    assert source_role_v2(commit) == expected


def test_union_aliases_reject_conflicting_frozen_maps():
    v1 = {"target_groups": {"one": ["alias"]}}
    v2 = {
        "diversevul_target_selection": {
            "project_aliases": {"two": ["alias"]}
        }
    }
    with pytest.raises(ValueError, match="Conflicting frozen project alias"):
        union_alias_lookup(v1, v2)


def test_source_index_deduplicates_and_fold_package_excludes_alias_group(tmp_path):
    prime_dir = tmp_path / "prime"
    prime_dir.mkdir()
    rows_by_file = {
        "primevul_train.jsonl": [
            prime_row("int linux_a(void);", 1, "linux-2.6", "commit-a"),
            prime_row("int kept(void);", 0, "other", "commit-b"),
        ],
        "primevul_valid.jsonl": [
            prime_row("int kept(void);\n", 0, "other", "commit-c")
        ],
        "primevul_test.jsonl": [
            prime_row("int conflict(void);", 0, "other", "commit-d"),
            prime_row("int conflict(void);", 1, "other", "commit-e"),
        ],
    }
    cache_manifest_rows = []
    cache_ids = []
    position = 0
    for filename, raw_rows in rows_by_file.items():
        write_jsonl(prime_dir / filename, raw_rows)
        for line_number, row in enumerate(raw_rows, start=1):
            row_id = f"prime-{position}"
            cache_ids.append(row_id)
            cache_manifest_rows.append(
                {
                    "row_id": row_id,
                    "source_file": filename,
                    "line_number": line_number,
                    "project": row["project"],
                    "commit_id": row["commit_id"],
                    "target": row["target"],
                }
            )
            position += 1
    cache_manifest = tmp_path / "prime_manifest.csv.gz"
    write_gzip_csv(
        cache_manifest,
        ["row_id", "source_file", "line_number", "project", "commit_id", "target"],
        cache_manifest_rows,
    )
    cache_id_path = tmp_path / "row_ids.txt"
    cache_id_path.write_text("\n".join(cache_ids) + "\n", encoding="utf-8")

    database = tmp_path / "source.sqlite"
    with ExtensionSourceIndex(database) as index:
        alignment = index.index_feature_manifest(cache_manifest, cache_id_path)
        audit = index.ingest_primevul(
            prime_dir, {"linux": "linux", "linux-2.6": "linux"}
        )
        metadata = tmp_path / "source_metadata.csv.gz"
        labels = tmp_path / "source_labels.csv.gz"
        count, records = write_source_manifest_and_labels(
            index.iter_records(), metadata, labels
        )

    assert alignment == {"feature_manifest_rows": 5, "cache_row_ids": 5}
    assert audit["same_label_duplicates_removed"] == 1
    assert audit["conflicting_keys_quarantined"] == 1
    assert audit["retained_rows"] == 2
    assert audit["feature_manifest_rows_not_retained"] == 3
    assert count == 2
    with gzip.open(metadata, "rt", encoding="utf-8", newline="") as stream:
        metadata_rows = list(csv.DictReader(stream))
    with gzip.open(labels, "rt", encoding="utf-8", newline="") as stream:
        label_rows = list(csv.DictReader(stream))
    assert "target" not in metadata_rows[0]
    assert [row["row_id"] for row in metadata_rows] == [
        row["row_id"] for row in label_rows
    ]

    target_rows = [
        {"position": "0", "project_group": "linux"},
        {"position": "1", "project_group": "other-target"},
    ]
    folds = write_fold_packages(
        records=records,
        target_rows=target_rows,
        selected_groups=["linux"],
        aliases_by_group={"linux": ("linux", "linux-2.6")},
        source_label_dir=tmp_path / "source_packages",
        target_fold_dir=tmp_path / "target_packages",
        sha256_file=sha256,
    )
    package = np.load(tmp_path / "source_packages" / "linux.npz")
    assert package.files == [
        "source_positions",
        "feature_positions",
        "labels",
        "role_codes",
    ]
    assert package["labels"].tolist() == [0]
    assert folds[0]["excluded_source_rows"] == 1
    target_package = np.load(tmp_path / "target_packages" / "linux.npz")
    assert target_package.files == ["target_positions"]
    assert target_package["target_positions"].tolist() == [0]


def test_target_manifest_rejects_labels(tmp_path):
    manifest = tmp_path / "target.csv.gz"
    write_gzip_csv(
        manifest,
        [
            "position",
            "row_id",
            "dataset",
            "source_file",
            "line_number",
            "project",
            "project_group",
            "commit_id",
            "exact_code_key",
            "target",
        ],
        [],
    )
    with pytest.raises(ValueError, match="forbidden label columns"):
        read_target_manifest(manifest)


def test_extension_hashing_cache_is_label_free_and_checks_exact_keys(tmp_path):
    diverse = tmp_path / "diverse.json"
    functions = ["int one(void) { return 1; }", "int two(void) { return 2; }"]
    write_jsonl(
        diverse,
        [
            {"func": functions[0], "target": 1},
            {"func": "not selected", "target": 0},
            {"func": functions[1], "target": 0},
        ],
    )
    manifest = tmp_path / "target.csv.gz"
    fields = [
        "position",
        "row_id",
        "dataset",
        "source_file",
        "line_number",
        "project",
        "project_group",
        "commit_id",
        "exact_code_key",
    ]
    rows = [
        {
            "position": position,
            "row_id": f"target-{position}",
            "dataset": "diversevul",
            "source_file": diverse.name,
            "line_number": line_number,
            "project": "linux",
            "project_group": "linux",
            "commit_id": f"c-{position}",
            "exact_code_key": exact_code_key(function),
        }
        for position, (line_number, function) in enumerate(zip((1, 3), functions))
    ]
    write_gzip_csv(manifest, fields, rows)
    hashing_config = tmp_path / "hashing.json"
    hashing_config.write_text(
        json.dumps(
            {
                "hashing_vectorizer": {
                    "n_features": 32,
                    "ngram_range": [1, 2],
                    "alternate_sign": False,
                    "norm": "l2",
                    "lowercase": False,
                    "token_pattern": r"(?u)\b\w+\b",
                }
            }
        ),
        encoding="utf-8",
    )
    extension_config = tmp_path / "extension.json"
    extension_config.write_text(
        json.dumps(
            {
                "protocol_version": "vultriage-extension-v2",
                "detectors": {
                    "hashing_sgd": {"inherit": hashing_config.as_posix()}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "cache"
    metadata = build_hashing_cache(
        diversevul=diverse,
        manifest=manifest,
        hashing_config_path=hashing_config,
        extension_config_path=extension_config,
        output=output,
        batch_size=1,
    )
    matrix = sparse.load_npz(output / "features.npz")
    assert matrix.shape == (2, 32)
    assert (output / "row_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "target-0",
        "target-1",
    ]
    assert metadata["labels_used"] is False
    assert metadata["labels_serialized"] is False
    assert "target" not in json.dumps(metadata).lower()
    assert set(SOURCE_ROLE_CODES) == {"train", "model_validation", "calibration"}

    bad_manifest = tmp_path / "bad.csv.gz"
    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["exact_code_key"] = "0" * 64
    write_gzip_csv(bad_manifest, fields, bad_rows)
    with pytest.raises(ValueError, match="differs from frozen target manifest"):
        build_hashing_cache(
            diversevul=diverse,
            manifest=bad_manifest,
            hashing_config_path=hashing_config,
            extension_config_path=extension_config,
            output=tmp_path / "bad-cache",
            batch_size=2,
        )
