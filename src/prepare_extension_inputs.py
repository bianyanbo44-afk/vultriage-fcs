"""Build frozen source-fold packages for extension-v2 external targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from vultriage.data import load_config, sha256
from vultriage.extension_inputs import (
    SOURCE_ROLE_CODES,
    SOURCE_SPLIT_SALT,
    ExtensionSourceIndex,
    read_target_manifest,
    selected_target_groups,
    union_alias_lookup,
    write_fold_packages,
    write_source_manifest_and_labels,
)


def build_extension_inputs(
    *,
    primevul_dir: Path,
    primevul_manifest: Path,
    primevul_feature_cache: Path,
    v1_config_path: Path,
    extension_config_path: Path,
    target_manifest: Path,
    target_summary_path: Path,
    output: Path,
    index_path: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    v1_config = load_config(v1_config_path)
    extension_config = load_config(extension_config_path)
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
    if target_summary.get("config_sha256") != sha256(extension_config_path):
        raise ValueError("Target summary does not match the extension-v2 config")
    if target_summary.get("manifest_sha256") != sha256(target_manifest):
        raise ValueError("Target summary does not match the target manifest")
    target_rows = read_target_manifest(target_manifest)
    selected_groups = selected_target_groups(target_rows, target_summary)
    alias_lookup, aliases_by_group = union_alias_lookup(v1_config, extension_config)

    cache_row_ids = primevul_feature_cache / "row_ids.txt"
    cache_metadata = primevul_feature_cache / "metadata.json"
    if not cache_row_ids.is_file() or not cache_metadata.is_file():
        raise FileNotFoundError("PrimeVul feature cache is missing row IDs or metadata")
    cache_info = json.loads(cache_metadata.read_text(encoding="utf-8"))
    if cache_info.get("manifest_sha256") != sha256(primevul_manifest):
        raise ValueError("PrimeVul feature cache does not match its source manifest")
    if cache_info.get("labels_used") is not False:
        raise ValueError("PrimeVul feature cache is not marked label-free")

    with ExtensionSourceIndex(index_path) as index:
        alignment_audit = index.index_feature_manifest(
            primevul_manifest, cache_row_ids
        )
        source_audit = index.ingest_primevul(primevul_dir, alias_lookup)
        source_metadata = output / "source_metadata.csv.gz"
        source_labels = output / "source_labels.csv.gz"
        source_rows, retained_records = write_source_manifest_and_labels(
            index.iter_records(), source_metadata, source_labels
        )

    # Re-emit the target metadata rather than copying bytes from the sealed
    # manifest, so the package records an independent label-free artifact.
    copied_target_manifest = output / "target_metadata.csv.gz"
    with gzip.open(target_manifest, "rt", encoding="utf-8", newline="") as source:
        with gzip.open(copied_target_manifest, "wt", encoding="utf-8", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            writer.writerows(reader)
    folds = write_fold_packages(
        records=retained_records,
        target_rows=target_rows,
        selected_groups=selected_groups,
        aliases_by_group=aliases_by_group,
        source_label_dir=output / "source_label_packages",
        target_fold_dir=output / "target_position_packages",
        sha256_file=sha256,
    )
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": extension_config["protocol_version"],
        "extension_config_sha256": sha256(extension_config_path),
        "v1_config_sha256": sha256(v1_config_path),
        "source_split": {
            "salt": SOURCE_SPLIT_SALT,
            "buckets": {
                "train": [0, 69],
                "model_validation": [70, 79],
                "calibration": [80, 99],
            },
            "role_codes": SOURCE_ROLE_CODES,
        },
        "source_deduplication": source_audit,
        "source_feature_alignment": alignment_audit,
        "source_rows": source_rows,
        "source_metadata_sha256": sha256(source_metadata),
        "source_label_vault_sha256": sha256(source_labels),
        "source_metadata_contains_labels": False,
        "source_index_contains_function_text": False,
        "primevul_manifest_sha256": sha256(primevul_manifest),
        "primevul_feature_cache_metadata_sha256": sha256(cache_metadata),
        "primevul_feature_cache_row_ids_sha256": sha256(cache_row_ids),
        "target_rows": len(target_rows),
        "target_metadata_sha256": sha256(copied_target_manifest),
        "target_metadata_contains_labels": False,
        "target_label_vault_copied_or_opened": False,
        "separate_target_label_vault_sha256": target_summary.get(
            "label_vault_sha256"
        ),
        "selected_project_groups": selected_groups,
        "alias_matching_case_sensitive": True,
        "folds": folds,
    }
    summary_path = output / "package_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primevul-dir", type=Path, required=True)
    parser.add_argument("--primevul-manifest", type=Path, required=True)
    parser.add_argument("--primevul-feature-cache", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if args.index is not None:
        result = build_extension_inputs(
            primevul_dir=args.primevul_dir,
            primevul_manifest=args.primevul_manifest,
            primevul_feature_cache=args.primevul_feature_cache,
            v1_config_path=args.v1_config,
            extension_config_path=args.extension_config,
            target_manifest=args.target_manifest,
            target_summary_path=args.target_summary,
            output=args.output,
            index_path=args.index,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="vultriage-extension-source-") as temp:
            result = build_extension_inputs(
                primevul_dir=args.primevul_dir,
                primevul_manifest=args.primevul_manifest,
                primevul_feature_cache=args.primevul_feature_cache,
                v1_config_path=args.v1_config,
                extension_config_path=args.extension_config,
                target_manifest=args.target_manifest,
                target_summary_path=args.target_summary,
                output=args.output,
                index_path=Path(temp) / "source_index.sqlite",
            )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
