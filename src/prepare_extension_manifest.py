"""Freeze the exact-deduplicated DiverseVul target cohort before modeling."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from vultriage.data import load_config, sha256
from vultriage.extension_data import ExtensionIndex, ExtensionRecord


def choose_targets(
    project_counts: Iterable[dict[str, int | str]], config: dict
) -> tuple[list[str], list[dict[str, object]]]:
    eligibility = config["diversevul_target_selection"]["eligibility"]
    rows: list[dict[str, object]] = []
    for source in project_counts:
        row: dict[str, object] = dict(source)
        row["eligible"] = (
            int(row["total"]) >= int(eligibility["minimum_total"])
            and int(row["vulnerable"]) >= int(eligibility["minimum_vulnerable"])
            and int(row["safe"]) >= int(eligibility["minimum_safe"])
        )
        rows.append(row)
    ranked = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            -int(row["vulnerable"]),
            -int(row["total"]),
            str(row["project_group"]).casefold(),
            str(row["project_group"]),
        ),
    )
    selected = [str(row["project_group"]) for row in ranked[:24]]
    selected_set = set(selected)
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected"] = row["project_group"] in selected_set
    ineligible = sorted(
        (row for row in rows if not row["eligible"]),
        key=lambda row: (
            str(row["project_group"]).casefold(),
            str(row["project_group"]),
        ),
    )
    return selected, ranked + ineligible


def write_manifest_and_labels(
    records: Iterator[ExtensionRecord], manifest_path: Path, labels_path: Path
) -> int:
    """Write aligned public metadata and sealed labels in one streaming pass."""

    manifest_fields = [
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
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        manifest_path.open("wb") as manifest_raw,
        gzip.GzipFile(
            filename="extension_manifest.csv",
            fileobj=manifest_raw,
            mode="wb",
            mtime=0,
        ) as manifest_gzip,
        io.TextIOWrapper(
            manifest_gzip, encoding="utf-8", newline=""
        ) as manifest_text,
        labels_path.open("wb") as label_raw,
        gzip.GzipFile(
            filename="extension_labels.csv",
            fileobj=label_raw,
            mode="wb",
            mtime=0,
        ) as label_gzip,
        io.TextIOWrapper(label_gzip, encoding="utf-8", newline="") as label_text,
    ):
        manifest_writer = csv.DictWriter(manifest_text, fieldnames=manifest_fields)
        label_writer = csv.DictWriter(
            label_text, fieldnames=["position", "row_id", "target"]
        )
        manifest_writer.writeheader()
        label_writer.writeheader()
        for position, record in enumerate(records):
            manifest_writer.writerow(
                {
                    "position": position,
                    "row_id": record.row_id,
                    "dataset": record.dataset,
                    "source_file": record.source_file,
                    "line_number": record.line_number,
                    "project": record.project,
                    "project_group": record.project_group,
                    "commit_id": record.commit_id,
                    "exact_code_key": record.exact_code_key,
                }
            )
            label_writer.writerow(
                {"position": position, "row_id": record.row_id, "target": record.target}
            )
            count += 1
    return count


def build_manifest(
    *,
    primevul_dir: Path,
    diversevul: Path,
    config_path: Path,
    manifest: Path,
    labels: Path,
    summary_path: Path,
    index_path: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    with ExtensionIndex(index_path) as index:
        prime_audit = index.index_primevul(primevul_dir)
        audit = index.ingest_diversevul(diversevul, config)
        selected, project_rows = choose_targets(index.project_counts(), config)
        selected_rows = write_manifest_and_labels(
            index.iter_records(selected), manifest, labels
        )
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(config_path),
        "diversevul_sha256": sha256(diversevul),
        "primevul_input_sha256": {
            path.name: sha256(path)
            for path in sorted(primevul_dir.glob("primevul_*.jsonl"))
        },
        "primevul_text_key_audit": prime_audit,
        "diversevul_deduplication": audit,
        "selected_project_groups": selected,
        "selected_rows": selected_rows,
        "project_groups": project_rows,
        "manifest_sha256": sha256(manifest),
        "label_vault_sha256": sha256(labels),
        "manifest_contains_target_labels": False,
        "selection_used_model_outputs": False,
        "index_backend": "sqlite",
        "index_contains_function_text": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primevul-dir", type=Path, required=True)
    parser.add_argument("--diversevul", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        help="Optional persistent SQLite audit index; defaults to a temporary file.",
    )
    args = parser.parse_args()
    if args.index is not None:
        summary = build_manifest(
            primevul_dir=args.primevul_dir,
            diversevul=args.diversevul,
            config_path=args.config,
            manifest=args.manifest,
            labels=args.labels,
            summary_path=args.summary,
            index_path=args.index,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="vultriage-extension-") as directory:
            summary = build_manifest(
                primevul_dir=args.primevul_dir,
                diversevul=args.diversevul,
                config_path=args.config,
                manifest=args.manifest,
                labels=args.labels,
                summary_path=args.summary,
                index_path=Path(directory) / "extension_index.sqlite",
            )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
