# Extension-v2 Experiment Run Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan, followed by run and validate
- Origin Date: 2026-08-14T02:10:51+08:00
- Verification Status: UNVERIFIED until every sealed command completes
- Version Label: vultriage_extension_v2_plan

## Experiment Overview

- **Title:** VulTriage external-validity and representation-transport study
- **Objective:** test the frozen triage layer on PrimeVul and exact-deduplicated
  DiverseVul with hashing-SGD and frozen CodeBERT detectors, a PROM-derived
  baseline, a prospectively frozen gate, ablations, calibration-size sensitivity,
  and efficiency accounting.
- **Primary hypotheses:** at the fixed 10% vulnerable / 20% safe point, at least
  one external project remains both-class automatable; gate-passing external
  projects have directionally lower raw pre-gate maximum class-specific
  violation than failing projects. These hypotheses may fail and are not
  success criteria for process completion.
- **Type:** ETL, model inference, statistical analysis, and reproducibility audit.

## Frozen Inputs

| Input | Path | Role |
|---|---|---|
| v2 protocol | `configs/preregistered_extension_v2.json` | all choices and hash anchor |
| PrimeVul | `data/external/primevul_original` | labeled source/development domain |
| DiverseVul | `data/external/diversevul_original/diversevul_20230702.json` | external confirmation domain |
| PrimeVul v1 metrics | `outputs/exp-e1-cpu-full-evaluation/fold_seed_metrics.csv` | gate development only |
| CodeBERT | `microsoft/codebert-base` revision `3b0952feddeffad0063f274080e3c23d75e7eb39` | frozen representation |
| PROM | repository commit `f16b52772123064551b6450f10b308971c4cdd39` | algorithm provenance only |

## Stage Commands and Seals

Every output directory is new. No stage overwrites or automatically retries a
failed run. The evaluator receives the label vault only after the corresponding
prediction seal verifies.

1. Build the stream-safe exact-deduplicated DiverseVul manifest, label vault,
   frozen project list, and hashes.
2. Build the PrimeVul v2 source-fold metadata and per-target label packages,
   excluding every frozen same-name/alias project group, then build the
   label-free DiverseVul hashing feature cache. The source partition uses
   `vultriage-extension-source-v2` with train 0--69, model validation 70--79,
   and calibration 80--99. This stage does not open the target label vault.
3. Run the cross-dataset 128-permutation MinHash audit and seal the flagged-pair
   sensitivity cohort.
4. Fit and serialize the PrimeVul-only support gate, including leave-one-project-
   out development predictions and artifact hashes.
5. Build label-free hashing feature caches for selected DiverseVul rows and
   extract frozen CodeBERT embeddings for the required PrimeVul/DiverseVul rows.
6. Generate and seal detector probabilities, density ratios, diagnostics, and
   efficiency records for both detectors and all five seeds.
7. Verify the prediction seal, then join target labels and evaluate all methods,
   budgets, ablations, and calibration-size cells.
8. Independently reproduce table keys, project aggregation, bootstrap intervals,
   exact tests, Holm adjustment, and figure-source hashes.

Completed input-stage artifacts (2026-08-13 UTC):

- `outputs/extension-v2/source-v2/package_summary.json`: 226,582 retained
  exact-deduplicated PrimeVul source rows and 24 target-specific folds;
  9,186 same-label duplicate keys were removed. Metadata and labels are
  separate, and each fold excludes its frozen target alias group.
- `outputs/extension-v2/hashing-target-v1/metadata.json`: 79,355 target rows,
  262,144 hashing features, 11,826,911 nonzero entries; labels were neither
  read into the cache nor serialized.
- These are input-preparation artifacts only. No detector probabilities,
  gate decisions, or target-label evaluation was run in this stage.

Exact command lines are written to `outputs/extension-v2/command_ledger.jsonl`
before each command starts and included in the final public manifest. Commands
may change only to correct implementation errors; any such change is appended to
`research/deviation_log.md` before rerunning and never erases the failed attempt.

## Monitoring

- ETL and CPU post-processing: 30-second liveness and output-growth checks;
  hard timeout 120 minutes per stage.
- CodeBERT extraction: 30-second liveness, row-count progress, host RSS, and GPU
  memory checks; hard timeout 12 hours per extraction stage.
- Training/evaluation matrix: 30-second liveness and completed-cell count;
  hard timeout 12 hours.
- The unrelated GPU process visible before this plan is not terminated or
  modified. GPU extraction begins only when enough free memory is available.

## Success Criteria

Process success means exit code zero; expected files are nonempty; primary keys
are unique; arrays and manifests align; labels are absent from prediction input;
all expected dataset-detector-project-seed-budget cells exist or have one logged
failure; hashes verify; metrics are finite where mathematically defined; and the
statistical audit reproduces the released tables. A favorable empirical result
is not a process success criterion.

## Statistical Validation

Projects, not rows, seeds, or budget cells, are independent units. Seeds are
averaged within project. The primary point is 0.10/0.20; the other eight points
form the frozen budget family. The analysis uses project bootstrap intervals,
exact sign tests, exact Wilcoxon only with at least ten nonzero project
differences, and Holm adjustment as specified in the protocol. The final ARS
validation record checks all 11 statistical fallacy classes and labels unresolved
limitations rather than converting them to positive conclusions.
