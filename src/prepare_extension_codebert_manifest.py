"""Create a label-free, source-then-target manifest for CodeBERT extraction.

The source and target manifests are already frozen independently.  This file
only concatenates their public metadata so that one deterministic embedding
cache can be indexed by the target-fold packages without ever opening the
target-label vault.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from vultriage.data import sha256


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError(f"positions are not contiguous in {path}")
    return rows


def build_manifest(source_path: Path, target_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    source_rows = read_rows(source_path)
    target_rows = read_rows(target_path)
    fields = [
        "position",
        "row_id",
        "dataset",
        "source_file",
        "line_number",
        "project",
        "project_group",
        "commit_id",
    ]
    combined: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in source_rows + target_rows:
        row_id = row["row_id"]
        if row_id in seen:
            raise ValueError(f"duplicate row_id in CodeBERT manifest: {row_id}")
        seen.add(row_id)
        combined.append(
            {
                "position": str(len(combined)),
                "row_id": row_id,
                "dataset": row["dataset"],
                "source_file": row["source_file"],
                "line_number": row["line_number"],
                "project": row["project"],
                "project_group": row["project_group"],
                "commit_id": row["commit_id"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="combined_metadata.csv", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(combined)
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "vultriage-extension-v2",
        "source_metadata": str(source_path),
        "target_metadata": str(target_path),
        "source_metadata_sha256": sha256(source_path),
        "target_metadata_sha256": sha256(target_path),
        "manifest_sha256": sha256(output),
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "rows": len(combined),
        "dataset_order": ["primevul", "diversevul"],
        "labels_used": False,
        "target_labels_serialized": False,
    }
    output.with_name("manifest_metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.source, args.target, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
