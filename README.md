# TRACE research reproducibility artifact

This versioned release contains current and frozen source code, original
and additional project aggregates, seven selected-fit receipts, figure
exports and source data, and compact post-review displays for TRACE.
Repository: https://github.com/bianyanbo44-afk/vultriage-fcs
Version: trace-validation-v3-2026-10-10
Record: https://github.com/bianyanbo44-afk/vultriage-fcs/releases/tag/trace-validation-v3-2026-10-10

The current TRACE manuscript, cover letter, author declarations and mock
reviews are excluded from this archive, as are raw benchmark functions,
weights, per-function predictions and decisions, and label vaults. The
repository separately retains its historical KBS manuscript sources.
REPOSITORY_METADATA.json describes the authors, source relations and
access route. No new open licence or DOI is assigned by this update.

Use Python 3.12 and install requirements-audit.txt for aggregate reconstruction.
Run from this archive root:

    python scripts/check_neurocomputing_revision.py --aggregate-only --analysis-only --report legacy_reconstruction.json
    python scripts/check_neural_aggregate_evidence.py --analysis paper/neurocomputing_revision/analysis/neural_validation_v3 --output neural_reconstruction.json --check-table-bytes
    python scripts/build_neural_display_additions.py --check

The first command deliberately skips excluded editable manuscript and PDF
checks as well as raw replay. The neural command independently reconstructs
the complete released paired and bootstrap evidence. No training is needed.
The historical 18-member release and additional 48-member release have separate
public manifests. The compact display additions have a third manifest.

requirements-validation-v3-lock.txt records direct numerical-library versions
collected after execution from the dedicated validation environment, with its
limits in reproducibility/validation_runtime_v3.json. Original analysis-library
versions remain in their unchanged historical manifest. Full detector replay
requires independently acquired hash-matched benchmark/model inputs and the
historical screening artifacts described by the protocols. Do not substitute
new splits while claiming exact raw replay. Original neural copied seed
addresses are not repetitions; additional CodeBERT and hashing each have three
actual fits, and the frozen reference has one head. The same selected objects
are shared across the two external collections, which must not be pooled.

ARCHIVE_MANIFEST.json records every payload file and SHA-256. DATA_DICTIONARY.md
defines principal endpoints, units, missing values, aggregation and provenance.
Repository publication provides access to the checked evidence. It does not
attest journal submission, acceptance or externally immutable experiment timing.
