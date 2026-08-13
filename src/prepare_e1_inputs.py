"""Separate E1 metadata/source labels from the sealed evaluation label vault."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from vultriage.data import iter_manifest, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(iter_manifest(args.manifest))
    cache_ids = (args.feature_cache / "row_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    manifest_ids = [row["row_id"] for row in rows]
    if cache_ids != manifest_ids:
        raise ValueError("Feature cache and split manifest row order differ")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    metadata_path = args.output / "metadata.csv.gz"
    fields = ["position", "row_id", "origin_split", "project_group", "commit_id"]
    with gzip.open(metadata_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(rows):
            writer.writerow({field: position if field == "position" else row[field] for field in fields})

    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    vault_path = args.output / "evaluation_label_vault.npz"
    np.savez_compressed(vault_path, labels=labels)

    source_dir = args.output / "source_labels"
    source_dir.mkdir()
    for group in config["target_groups"]:
        positions = np.asarray(
            [index for index, row in enumerate(rows) if row["project_group"] != group],
            dtype=np.int32,
        )
        path = source_dir / f"{group}.npz"
        np.savez_compressed(path, positions=positions, labels=labels[positions])

    official_source_positions = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["origin_split"] in {"train", "valid"}
        ],
        dtype=np.int32,
    )
    np.savez_compressed(
        source_dir / "official.npz",
        positions=official_source_positions,
        labels=labels[official_source_positions],
    )
    hashes = {
        "manifest_sha256": sha256(args.manifest),
        "feature_cache_metadata_sha256": sha256(args.feature_cache / "metadata.json"),
        "metadata_sha256": sha256(metadata_path),
        "evaluation_label_vault_sha256": sha256(vault_path),
        "source_label_sha256": {
            path.stem: sha256(path) for path in sorted(source_dir.glob("*.npz"))
        },
        "metadata_contains_vulnerability_labels": False,
        "prediction_runner_receives_evaluation_vault": False,
    }
    (args.output / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(hashes, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
