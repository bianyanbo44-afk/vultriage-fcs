"""Run the frozen extension-v2 cross-dataset near-duplicate sensitivity audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from vultriage.data import iter_jsonl_rows, load_config, sha256
from vultriage.extension_data import exact_code_key, stable_id
from vultriage.near_duplicate import (
    DEFAULT_BANDS,
    DEFAULT_ROWS_PER_BAND,
    MINHASH_SEED,
    AuditDocument,
    NearDuplicateIndex,
    PeakRSSMonitor,
    cpp_lexical_tokens,
    lexical_token_set,
    lsh_candidate_probability,
)


PAIR_FIELDS = (
    "target_row_id",
    "target_project",
    "target_project_group",
    "target_source_file",
    "target_line_number",
    "target_exact_code_key",
    "target_token_count",
    "prime_row_id",
    "prime_project",
    "prime_project_group",
    "prime_source_file",
    "prime_line_number",
    "prime_exact_code_key",
    "prime_token_count",
    "intersection_count",
    "union_count",
    "exact_jaccard",
    "minhash_agreement",
    "minhash_estimate",
)

EXCLUSION_FIELDS = (
    "position",
    "target_row_id",
    "target_project",
    "target_project_group",
    "target_source_file",
    "target_line_number",
    "target_exact_code_key",
    "target_token_count",
    "flagged_prime_pair_count",
    "maximum_exact_jaccard",
)


def _open_csv_gzip(path: Path, filename: str):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename=filename, fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def write_deterministic_csv_gzip(
    path: Path,
    filename: str,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        raw, compressed, text = _open_csv_gzip(temporary_path, filename)
        try:
            writer = csv.DictWriter(
                text, fieldnames=tuple(fieldnames), extrasaction="raise"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
        finally:
            text.close()
            compressed.close()
            raw.close()
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def read_manifest(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def _manifest_by_line(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for row in read_manifest(path):
        line_number = int(row["line_number"])
        if line_number in rows:
            raise ValueError(f"Duplicate DiverseVul line number in manifest: {line_number}")
        rows[line_number] = row
    return rows


def _file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(paths)}


def _validate_frozen_inputs(
    *,
    config_path: Path,
    config: dict,
    manifest_path: Path,
    manifest_summary_path: Path,
    exact_index_path: Path,
    primevul_dir: Path,
    diversevul_path: Path,
) -> tuple[dict, int, int]:
    near = config["datasets"]["deduplication"]["near_duplicate_audit"]
    if int(near["minhash_permutations"]) != 128:
        raise ValueError("Frozen protocol does not specify 128 MinHash permutations")
    if float(near["jaccard_flag_threshold"]) != 0.9:
        raise ValueError("Frozen protocol does not specify Jaccard threshold 0.90")
    summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))
    checks = {
        "config": (sha256(config_path), str(summary["config_sha256"])),
        "manifest": (sha256(manifest_path), str(summary["manifest_sha256"])),
        "diversevul": (sha256(diversevul_path), str(summary["diversevul_sha256"])),
    }
    for name, (observed, expected) in checks.items():
        if observed.upper() != expected.upper():
            raise ValueError(
                f"Frozen {name} hash mismatch: observed {observed}, expected {expected}"
            )
    prime_paths = sorted(primevul_dir.glob("primevul_*.jsonl"))
    if len(prime_paths) != 3:
        raise ValueError(f"Expected three PrimeVul JSONL files, found {len(prime_paths)}")
    observed_prime = _file_hashes(prime_paths)
    expected_prime = {
        str(name): str(value).upper()
        for name, value in summary["primevul_input_sha256"].items()
    }
    if observed_prime != expected_prime:
        raise ValueError("PrimeVul input hashes do not match the frozen manifest summary")

    uri = exact_index_path.resolve().as_uri() + "?mode=ro"
    exact_connection = sqlite3.connect(uri, uri=True)
    try:
        prime_unique = int(
            exact_connection.execute("SELECT COUNT(*) FROM prime_keys").fetchone()[0]
        )
    finally:
        exact_connection.close()
    selected_groups = tuple(summary["selected_project_groups"])
    manifest_rows = 0
    manifest_groups: set[str] = set()
    for row in read_manifest(manifest_path):
        manifest_rows += 1
        manifest_groups.add(str(row["project_group"]))
    if manifest_groups != set(selected_groups):
        raise ValueError("Manifest project groups differ from the frozen summary")
    selected_rows = manifest_rows
    if prime_unique != int(summary["primevul_text_key_audit"]["unique_text_keys"]):
        raise ValueError("PrimeVul key count differs from the frozen summary")
    if selected_rows != int(summary["selected_rows"]):
        raise ValueError("Selected target count differs from the frozen summary")
    return summary, prime_unique, selected_rows


def _prime_project_group(project: str, config: dict) -> str:
    aliases: dict[str, str] = {}
    for group, values in config["diversevul_target_selection"]["project_aliases"].items():
        for value in values:
            aliases[value] = group
    return aliases.get(project.strip(), project.strip())


def index_primevul(
    index: NearDuplicateIndex, primevul_dir: Path, config: dict
) -> dict[str, int]:
    counts = Counter()
    for _, path, line_number, row in iter_jsonl_rows(primevul_dir):
        counts["raw_rows"] += 1
        function = str(row.get("func", ""))
        project = str(row.get("project", "")).strip()
        commit_id = str(row.get("commit_id", "")).strip()
        if not function.strip():
            counts["missing_function_rows"] += 1
            continue
        if not project or not commit_id:
            counts["missing_metadata_rows_retained"] += 1
        key = exact_code_key(function)
        if not index.register_prime_key(key):
            counts["exact_duplicate_rows_skipped"] += 1
            continue
        tokens = lexical_token_set(function)
        index.add_document(
            AuditDocument(
                dataset="primevul",
                row_id=stable_id("primevul-near-v2", path.name, line_number, key),
                source_file=path.name,
                line_number=line_number,
                project=project,
                project_group=_prime_project_group(project, config),
                exact_code_key=key,
                tokens=tokens,
            )
        )
        counts["unique_documents_indexed"] += 1
    index.finish_documents()
    return dict(counts)


def index_diversevul_targets(
    index: NearDuplicateIndex, diversevul_path: Path, manifest_path: Path
) -> dict[str, int]:
    selected = _manifest_by_line(manifest_path)
    counts = Counter()
    with diversevul_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            counts["raw_rows_scanned"] += 1
            metadata = selected.pop(line_number, None)
            if metadata is None:
                continue
            row = json.loads(line)
            function = str(row.get("func", ""))
            key = exact_code_key(function)
            if key != metadata["exact_code_key"]:
                raise ValueError(
                    f"DiverseVul key mismatch at line {line_number}: {key} != "
                    f"{metadata['exact_code_key']}"
                )
            if str(row.get("project", "")).strip() != metadata["project"]:
                raise ValueError(f"DiverseVul project mismatch at line {line_number}")
            index.add_document(
                AuditDocument(
                    dataset="diversevul",
                    row_id=metadata["row_id"],
                    source_file=metadata["source_file"],
                    line_number=line_number,
                    project=metadata["project"],
                    project_group=metadata["project_group"],
                    exact_code_key=key,
                    tokens=lexical_token_set(function),
                )
            )
            counts["selected_documents_indexed"] += 1
    if selected:
        missing = sorted(selected)[:10]
        raise ValueError(f"Selected DiverseVul lines were not found: {missing}")
    index.finish_documents()
    return dict(counts)


def _pair_output_rows(index: NearDuplicateIndex) -> Iterator[dict[str, object]]:
    for row in index.iter_flagged_pairs():
        output = dict(row)
        output["exact_jaccard"] = f"{float(output['exact_jaccard']):.12f}"
        output["minhash_estimate"] = f"{float(output['minhash_estimate']):.12f}"
        yield {field: output[field] for field in PAIR_FIELDS}


def _write_sensitivity_outputs(
    *,
    index: NearDuplicateIndex,
    manifest_path: Path,
    exclusions_path: Path,
    cohort_path: Path,
) -> dict[str, object]:
    flagged = {str(row["target_row_id"]): row for row in index.iter_flagged_targets()}
    position_by_id = {
        row["row_id"]: row["position"] for row in read_manifest(manifest_path)
    }

    def exclusions() -> Iterator[dict[str, object]]:
        for row_id in sorted(flagged, key=lambda item: int(position_by_id[item])):
            source = dict(flagged[row_id])
            source["position"] = position_by_id[row_id]
            source["maximum_exact_jaccard"] = (
                f"{float(source['maximum_exact_jaccard']):.12f}"
            )
            yield {field: source[field] for field in EXCLUSION_FIELDS}

    exclusion_count = write_deterministic_csv_gzip(
        exclusions_path,
        "near_duplicate_exclusions.csv",
        EXCLUSION_FIELDS,
        exclusions(),
    )

    retained_row_ids = hashlib.sha256()
    project_counts: Counter[str] = Counter()
    manifest_fields: tuple[str, ...] | None = None

    def cohort() -> Iterator[dict[str, object]]:
        nonlocal manifest_fields
        for row in read_manifest(manifest_path):
            if manifest_fields is None:
                manifest_fields = tuple(row.keys())
            if row["row_id"] in flagged:
                continue
            retained_row_ids.update(row["row_id"].encode("ascii") + b"\n")
            project_counts[row["project_group"]] += 1
            yield row

    with gzip.open(manifest_path, "rt", encoding="utf-8", newline="") as stream:
        original_fields = tuple(csv.DictReader(stream).fieldnames or ())
    cohort_count = write_deterministic_csv_gzip(
        cohort_path,
        "near_duplicate_sensitivity_cohort.csv",
        original_fields,
        cohort(),
    )
    return {
        "excluded_target_rows": exclusion_count,
        "retained_target_rows": cohort_count,
        "retained_row_id_sequence_sha256": retained_row_ids.hexdigest().upper(),
        "retained_rows_by_project": dict(sorted(project_counts.items())),
    }


def _system_metadata() -> dict[str, object]:
    try:
        import psutil

        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        memory_bytes = int(psutil.virtual_memory().total)
        psutil_version = psutil.__version__
    except ImportError:
        cpu_physical = None
        cpu_logical = os.cpu_count()
        memory_bytes = None
        psutil_version = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "sqlite": sqlite3.sqlite_version,
        "numpy": np.__version__,
        "psutil": psutil_version,
        "cpu_physical_cores": cpu_physical,
        "cpu_logical_cores": cpu_logical,
        "host_memory_bytes": memory_bytes,
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    output_paths = (
        args.work_index,
        args.flagged_pairs,
        args.exclusions,
        args.sensitivity_cohort,
        args.summary,
    )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite audit outputs: {existing}")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    print("stage=input-validation status=started", file=sys.stderr, flush=True)
    manifest_summary, expected_prime, expected_target = _validate_frozen_inputs(
        config_path=args.config,
        config=config,
        manifest_path=args.manifest,
        manifest_summary_path=args.manifest_summary,
        exact_index_path=args.exact_index,
        primevul_dir=args.primevul_dir,
        diversevul_path=args.diversevul,
    )
    print(
        f"stage=input-validation status=complete prime={expected_prime} "
        f"target={expected_target}",
        file=sys.stderr,
        flush=True,
    )
    near = config["datasets"]["deduplication"]["near_duplicate_audit"]
    permutations = int(near["minhash_permutations"])
    threshold = float(near["jaccard_flag_threshold"])
    durations: dict[str, float] = {}
    audit_started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()

    with PeakRSSMonitor() as memory_monitor:
        with NearDuplicateIndex(
            args.work_index,
            permutations=permutations,
            bands=DEFAULT_BANDS,
            rows_per_band=DEFAULT_ROWS_PER_BAND,
            seed=MINHASH_SEED,
        ) as index:
            stage = time.perf_counter()
            print("stage=primevul-index status=started", file=sys.stderr, flush=True)
            prime_counts = index_primevul(index, args.primevul_dir, config)
            durations["primevul_index_seconds"] = time.perf_counter() - stage
            if index.document_count("primevul") != expected_prime:
                raise ValueError("Indexed PrimeVul unique count does not match frozen input")
            print(
                f"stage=primevul-index status=complete documents={expected_prime} "
                f"seconds={durations['primevul_index_seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            print("stage=diversevul-index status=started", file=sys.stderr, flush=True)
            target_counts = index_diversevul_targets(
                index, args.diversevul, args.manifest
            )
            durations["diversevul_index_seconds"] = time.perf_counter() - stage
            if index.document_count("diversevul") != expected_target:
                raise ValueError("Indexed target count does not match frozen input")
            print(
                f"stage=diversevul-index status=complete documents={expected_target} "
                f"seconds={durations['diversevul_index_seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            print("stage=candidate-generation status=started", file=sys.stderr, flush=True)
            candidate_batches = 0
            latest_candidates = 0
            for batch in index.generate_candidates(threshold=threshold):
                candidate_batches += 1
                latest_candidates = int(batch["total_candidates"])
                if candidate_batches % 5 == 0:
                    print(
                        f"stage=candidate-generation batches={candidate_batches} "
                        f"candidates={latest_candidates}",
                        file=sys.stderr,
                        flush=True,
                    )
            durations["candidate_generation_seconds"] = time.perf_counter() - stage
            print(
                f"stage=candidate-generation status=complete candidates={latest_candidates} "
                f"seconds={durations['candidate_generation_seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            print("stage=exact-verification status=started", file=sys.stderr, flush=True)
            flagged_pairs_seen = 0
            for verified in index.verify_candidates(threshold):
                flagged_pairs_seen += 1
                if flagged_pairs_seen % 10_000 == 0:
                    print(
                        f"stage=exact-verification flagged={flagged_pairs_seen} "
                        f"checked={verified['checked_candidates']}",
                        file=sys.stderr,
                        flush=True,
                    )
            durations["exact_verification_seconds"] = time.perf_counter() - stage
            if flagged_pairs_seen != index.flagged_pair_count():
                raise RuntimeError("Flagged-pair verification count mismatch")
            print(
                f"stage=exact-verification status=complete flagged={flagged_pairs_seen} "
                f"seconds={durations['exact_verification_seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            print("stage=output status=started", file=sys.stderr, flush=True)
            pair_count = write_deterministic_csv_gzip(
                args.flagged_pairs,
                "near_duplicate_flagged_pairs.csv",
                PAIR_FIELDS,
                _pair_output_rows(index),
            )
            sensitivity = _write_sensitivity_outputs(
                index=index,
                manifest_path=args.manifest,
                exclusions_path=args.exclusions,
                cohort_path=args.sensitivity_cohort,
            )
            durations["output_seconds"] = time.perf_counter() - stage
            print(
                f"stage=output status=complete seconds={durations['output_seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )

            database_counts = {
                "candidate_batches": candidate_batches,
                "candidate_pairs": index.candidate_count(),
                "targets_with_candidates": index.candidate_target_count(),
                "flagged_pairs": pair_count,
                "flagged_target_rows": index.flagged_target_count(),
            }
            affected_projects = index.affected_projects()
            token_counts = {
                "primevul": index.token_count_summary("primevul"),
                "diversevul_selected": index.token_count_summary("diversevul"),
            }
            schema_columns = {
                str(table): [
                    str(row[1])
                    for row in index.connection.execute(f"PRAGMA table_info({table})")
                ]
                for table in ("documents", "candidates", "flagged_pairs")
            }

    durations["total_seconds"] = time.perf_counter() - audit_started
    artifacts = {
        "flagged_pairs": {
            "path": str(args.flagged_pairs),
            "sha256": sha256(args.flagged_pairs),
            "bytes": args.flagged_pairs.stat().st_size,
        },
        "near_duplicate_exclusions": {
            "path": str(args.exclusions),
            "sha256": sha256(args.exclusions),
            "bytes": args.exclusions.stat().st_size,
        },
        "sensitivity_cohort": {
            "path": str(args.sensitivity_cohort),
            "sha256": sha256(args.sensitivity_cohort),
            "bytes": args.sensitivity_cohort.stat().st_size,
        },
        "work_index": {
            "path": str(args.work_index),
            "sha256": sha256(args.work_index),
            "bytes": args.work_index.stat().st_size,
            "public_release": False,
        },
    }
    source_files = {
        "module": Path(__file__).parent / "vultriage" / "near_duplicate.py",
        "cli": Path(__file__),
    }
    result: dict[str, object] = {
        "audit_version": "vultriage-extension-v2-near-duplicate-audit-v1",
        "created_at_utc": created_at,
        "status": "complete",
        "scope": {
            "comparison": "unique PrimeVul functions versus the frozen exact-deduplicated selected DiverseVul cohort",
            "primary_analysis_changed": False,
            "outcome_labels_read": False,
            "outcome_bearing_models_run": False,
        },
        "frozen_inputs": {
            "protocol_version": config["protocol_version"],
            "config_sha256": sha256(args.config),
            "manifest_summary_sha256": sha256(args.manifest_summary),
            "manifest_sha256": sha256(args.manifest),
            "exact_index_sha256": sha256(args.exact_index),
            "diversevul_sha256": sha256(args.diversevul),
            "primevul_input_sha256": _file_hashes(
                args.primevul_dir.glob("primevul_*.jsonl")
            ),
            "selected_project_groups": manifest_summary["selected_project_groups"],
        },
        "algorithm": {
            "tokenizer": "deterministic C/C++ identifier, pp-number, and longest-match operator tokens; whitespace, comments, and string/character literal contents omitted",
            "similarity_object": "set of unique case-sensitive lexical tokens",
            "minhash_permutations": permutations,
            "minhash_seed": MINHASH_SEED,
            "minhash_token_hash": "first 64 bits of SHA-256 reduced modulo 4294967291",
            "minhash_permutation_family": "deterministic affine universal hashes modulo 4294967291",
            "lsh_bands": DEFAULT_BANDS,
            "lsh_rows_per_band": DEFAULT_ROWS_PER_BAND,
            "lsh_candidate_probability_at_threshold": lsh_candidate_probability(
                threshold, DEFAULT_BANDS, DEFAULT_ROWS_PER_BAND
            ),
            "candidate_cardinality_bound": "min(|A|,|B|)/max(|A|,|B|) >= threshold",
            "flag_rule": "candidate pair has exact lexical-token-set Jaccard >= 0.90",
            "flag_threshold": threshold,
            "candidate_generation_only_is_approximate": True,
            "reported_pair_verification_is_exact": True,
            "known_boundary": "a true pair can be missed if none of its 16 eight-permutation bands collide; nominal collision probability at Jaccard 0.90 is reported, not treated as a guarantee",
        },
        "counts": {
            "primevul": prime_counts,
            "diversevul": target_counts,
            **database_counts,
            **sensitivity,
            "affected_projects": affected_projects,
            "affected_project_count": len(affected_projects),
        },
        "token_set_cardinality": token_counts,
        "artifacts": artifacts,
        "resource_record": {
            "durations_seconds": {
                key: round(value, 6) for key, value in durations.items()
            },
            "peak_host_rss_bytes": memory_monitor.peak_rss_bytes,
            "system": _system_metadata(),
        },
        "privacy_and_schema_check": {
            "function_text_exported": False,
            "target_labels_exported_or_read": False,
            "target_label_vault_accessed": False,
            "index_validation_used_metadata_only": True,
            "sqlite_columns": schema_columns,
        },
        "implementation_sha256": {
            name: sha256(path) for name, path in source_files.items()
        },
    }
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primevul-dir", type=Path, required=True)
    parser.add_argument("--diversevul", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--exact-index", type=Path, required=True)
    parser.add_argument("--work-index", type=Path, required=True)
    parser.add_argument("--flagged-pairs", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--sensitivity-cohort", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
