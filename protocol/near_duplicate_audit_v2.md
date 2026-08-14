# Extension-v2 cross-dataset near-duplicate sensitivity audit

## Material Passport

- Artifact ID: `vultriage-extension-v2-near-duplicate-audit-v1`
- Type: deterministic data-leakage sensitivity audit
- Status: implementation fixed; empirical fields are populated from
  `outputs/extension-v2/near-duplicate-v1/near_duplicate_summary.json`
- Outcome access: none; the label vault is not accepted by the audit CLI
- Primary-analysis effect: none; exact deduplication remains the primary cohort

## Frozen scope

The audit compares the 226,582 unique PrimeVul function texts indexed during
the frozen exact-deduplication pass against the 79,355 DiverseVul rows in the
already frozen 24-project external-confirmation manifest. It does not inspect
prediction files, outcomes, detector outputs, or the target label vault. A
DiverseVul row flagged here is removed only in the separately named sensitivity
cohort. The exact-deduplicated primary cohort is unchanged.

## Token definition

The implementation canonicalizes line endings and horizontal trailing
whitespace using the already frozen exact-deduplication function, then performs
a deterministic C/C++ lexical scan. It retains case-sensitive identifiers,
preprocessing-number approximations, and longest-match C/C++ operators and
punctuators. It omits whitespace, comments, string literal contents, character
literal contents, and otherwise unrecognized bytes. Similarity is defined over
the set of unique tokens, so token order and frequency are not used.

This choice makes the protocol phrase "C/C++ lexical alphanumeric and operator
tokens" executable without adding a parser dependency. It also avoids making
comment prose or embedded strings the basis of a code-leakage flag. The exact
tokenizer behavior is covered by `tests/test_near_duplicate.py`.

## Candidate generation and exact flagging

Each token is mapped with the first 64 bits of SHA-256 reduced modulo
4,294,967,291. The 128 MinHash permutations are deterministic affine universal
hashes derived from the fixed seed
`vultriage-extension-v2|minhash|sha256-v1`. Signatures are divided into 16
bands of eight rows. SQLite stores document metadata, compressed token sets,
signatures, band keys, deduplicated candidate pairs, and verified pair counts;
it stores neither function bodies nor target labels.

The LSH stage is only a candidate generator. The token-cardinality upper bound
first removes pairs that cannot reach 0.90 Jaccard. Every remaining candidate
is decoded and checked using the exact set intersection and union counts. A
reported pair must have exact Jaccard similarity at least 0.90. Multiple band
collisions still produce one candidate pair because of a SQLite primary key.

The approximate boundary remains explicit: an exact near-duplicate can be
missed when none of its 16 bands collide. Under the standard independent
MinHash model, the nominal candidate probability is
`1 - (1 - 0.9^8)^16 = 0.9998774538341496` at the threshold. This probability is
reported as an algorithm characteristic, not as a deterministic recall
guarantee. Conversely, any observed signature agreement of at least 116/128
must contain a complete eight-row band by the pigeonhole principle and is
therefore generated as a candidate.

## Outputs

The real run writes the following files under
`outputs/extension-v2/near-duplicate-v1/`:

- `near_duplicate_flagged_pairs.csv.gz`: source locators, hashes, token counts,
  exact intersection/union/Jaccard, and MinHash agreement; no function text.
- `near_duplicate_exclusions.csv.gz`: one row per affected DiverseVul target.
- `near_duplicate_sensitivity_cohort.csv.gz`: the frozen public manifest with
  affected row IDs omitted and original positions preserved.
- `near_duplicate_summary.json`: input and artifact hashes, counts, affected
  projects, implementation hashes, timing, peak RSS, environment, and the
  ordered retained-row cohort hash.
- `near_duplicate_work.sqlite`: disk-backed private working index. It contains
  token sets but no function bodies or labels and is not a public-release file.

The summary's artifact SHA-256 and `retained_row_id_sequence_sha256` seal the
sensitivity cohort. The gzip writers set `mtime=0` and stable internal names so
the CSV artifacts are byte-reproducible for the same inputs and implementation.

## Reproduction command

```powershell
$env:PYTHONPATH='src'
python src\audit_near_duplicates_v2.py `
  --primevul-dir data\external\primevul_original `
  --diversevul data\external\diversevul_original\diversevul_20230702.json `
  --config configs\preregistered_extension_v2.json `
  --manifest outputs\extension-v2\manifest-v1\extension_manifest.csv.gz `
  --manifest-summary outputs\extension-v2\manifest-v1\manifest_summary.json `
  --exact-index outputs\extension-v2\manifest-v1\extension_index.sqlite `
  --work-index outputs\extension-v2\near-duplicate-v1\near_duplicate_work.sqlite `
  --flagged-pairs outputs\extension-v2\near-duplicate-v1\near_duplicate_flagged_pairs.csv.gz `
  --exclusions outputs\extension-v2\near-duplicate-v1\near_duplicate_exclusions.csv.gz `
  --sensitivity-cohort outputs\extension-v2\near-duplicate-v1\near_duplicate_sensitivity_cohort.csv.gz `
  --summary outputs\extension-v2\near-duplicate-v1\near_duplicate_summary.json
```

The CLI fails rather than overwriting an existing output. A clean independent
rerun therefore needs a new output directory.
