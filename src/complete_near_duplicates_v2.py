"""Complete near-duplicate LSH/verification from a preserved work index."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from audit_near_duplicates_v2 import (
    PAIR_FIELDS,
    _file_hashes,
    _system_metadata,
    _validate_frozen_inputs,
    _write_sensitivity_outputs,
    _pair_output_rows,
    write_deterministic_csv_gzip,
)
from vultriage.data import load_config, sha256
from vultriage.near_duplicate import (
    DEFAULT_BANDS,
    DEFAULT_ROWS_PER_BAND,
    MINHASH_SEED,
    NearDuplicateIndex,
    PeakRSSMonitor,
    lsh_candidate_probability,
)


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


def main() -> None:
    args = parse_args()
    paths = (
        args.flagged_pairs,
        args.exclusions,
        args.sensitivity_cohort,
        args.summary,
    )
    existing_outputs = [str(path) for path in paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite completion outputs: {existing_outputs}")
    config = load_config(args.config)
    manifest_summary, expected_prime, expected_target = _validate_frozen_inputs(
        config_path=args.config,
        config=config,
        manifest_path=args.manifest,
        manifest_summary_path=args.manifest_summary,
        exact_index_path=args.exact_index,
        primevul_dir=args.primevul_dir,
        diversevul_path=args.diversevul,
    )
    near = config["datasets"]["deduplication"]["near_duplicate_audit"]
    permutations = int(near["minhash_permutations"])
    threshold = float(near["jaccard_flag_threshold"])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    durations: dict[str, float] = {}
    print("stage=recovery-index-validation status=started", file=sys.stderr, flush=True)

    with PeakRSSMonitor() as monitor:
        with NearDuplicateIndex.open_existing(
            args.work_index,
            permutations=permutations,
            bands=DEFAULT_BANDS,
            rows_per_band=DEFAULT_ROWS_PER_BAND,
            seed=MINHASH_SEED,
        ) as index:
            observed_prime = index.document_count("primevul")
            observed_target = index.document_count("diversevul")
            if (observed_prime, observed_target) != (expected_prime, expected_target):
                raise ValueError(
                    f"Existing index counts {(observed_prime, observed_target)} do not match "
                    f"frozen {(expected_prime, expected_target)}"
                )
            print(
                f"stage=recovery-index-validation status=complete prime={observed_prime} "
                f"target={observed_target}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            batch_count = 0
            last_total = index.candidate_count()
            for batch in index.generate_candidates(threshold=threshold):
                batch_count += 1
                last_total = int(batch["total_candidates"])
                if batch_count % 5 == 0:
                    print(
                        f"stage=candidate-generation batches={batch_count} candidates={last_total}",
                        file=sys.stderr,
                        flush=True,
                    )
            durations["candidate_generation_seconds"] = time.perf_counter() - stage
            print(
                f"stage=candidate-generation status=complete candidates={last_total}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
            verified_candidates = 0
            flagged_pairs = 0
            for item in index.verify_candidates(threshold):
                verified_candidates = int(item["checked_candidates"])
                flagged_pairs += 1
                if flagged_pairs % 10_000 == 0:
                    print(
                        f"stage=exact-verification flagged={flagged_pairs} checked={verified_candidates}",
                        file=sys.stderr,
                        flush=True,
                    )
            # ``verify_candidates`` yields only flagged rows, so the last
            # yielded counter can be below the full candidate count when the
            # final candidates are non-flagged. All candidates are verified
            # when the generator completes successfully.
            verified_candidates = index.candidate_count()
            durations["exact_verification_seconds"] = time.perf_counter() - stage
            print(
                f"stage=exact-verification status=complete flagged={flagged_pairs}",
                file=sys.stderr,
                flush=True,
            )

            stage = time.perf_counter()
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
            counts = {
                "candidate_batches": batch_count,
                "candidate_pairs": index.candidate_count(),
                "targets_with_candidates": index.candidate_target_count(),
                "verified_candidates": verified_candidates,
                "flagged_pairs": pair_count,
                "flagged_target_rows": index.flagged_target_count(),
                "affected_projects": index.affected_projects(),
                **sensitivity,
            }
            token_counts = {
                "primevul": index.token_count_summary("primevul"),
                "diversevul_selected": index.token_count_summary("diversevul"),
            }
            sqlite_columns = {
                table: [
                    str(row[1])
                    for row in index.connection.execute(f"PRAGMA table_info({table})")
                ]
                for table in ("documents", "candidates", "flagged_pairs")
            }

    durations["total_seconds"] = time.perf_counter() - started
    artifacts = {
        "flagged_pairs": {"path": str(args.flagged_pairs), "sha256": sha256(args.flagged_pairs), "bytes": args.flagged_pairs.stat().st_size},
        "near_duplicate_exclusions": {"path": str(args.exclusions), "sha256": sha256(args.exclusions), "bytes": args.exclusions.stat().st_size},
        "sensitivity_cohort": {"path": str(args.sensitivity_cohort), "sha256": sha256(args.sensitivity_cohort), "bytes": args.sensitivity_cohort.stat().st_size},
        "work_index": {"path": str(args.work_index), "sha256": sha256(args.work_index), "bytes": args.work_index.stat().st_size, "public_release": False},
    }
    result = {
        "audit_version": "vultriage-extension-v2-near-duplicate-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "provenance": {
            "execution_mode": "completion from preserved full document index",
            "initial_wrapper_timeout": True,
            "initial_index_reused_without_rebuilding_or_model_access": True,
        },
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
            "primevul_input_sha256": _file_hashes(args.primevul_dir.glob("primevul_*.jsonl")),
            "selected_project_groups": manifest_summary["selected_project_groups"],
        },
        "algorithm": {
            "tokenizer": "deterministic C/C++ identifier, pp-number, and longest-match operator tokens; whitespace, comments, and string/character literal contents omitted",
            "similarity_object": "set of unique case-sensitive lexical tokens",
            "minhash_permutations": permutations,
            "minhash_seed": MINHASH_SEED,
            "lsh_bands": DEFAULT_BANDS,
            "lsh_rows_per_band": DEFAULT_ROWS_PER_BAND,
            "lsh_candidate_probability_at_threshold": lsh_candidate_probability(threshold, DEFAULT_BANDS, DEFAULT_ROWS_PER_BAND),
            "candidate_cardinality_bound": "min(|A|,|B|)/max(|A|,|B|) >= threshold",
            "flag_rule": "candidate pair has exact lexical-token-set Jaccard >= 0.90",
            "flag_threshold": threshold,
            "candidate_generation_only_is_approximate": True,
            "reported_pair_verification_is_exact": True,
        },
        "counts": counts,
        "token_set_cardinality": token_counts,
        "artifacts": artifacts,
        "resource_record": {"durations_seconds": {k: round(v, 6) for k, v in durations.items()}, "peak_host_rss_bytes": monitor.peak_rss_bytes, "system": _system_metadata()},
        "privacy_and_schema_check": {
            "function_text_exported": False,
            "target_labels_exported_or_read": False,
            "target_label_vault_accessed": False,
            "index_validation_used_metadata_only": True,
            "sqlite_columns": sqlite_columns,
        },
    }
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
