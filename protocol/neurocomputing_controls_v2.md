# Neurocomputing retrospective comparator controls v2

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: validation and authorized bounded retrospective analysis.
- Origin Date: 2026-10-09.
- Verification Status: source integrity, selector invariants, and deterministic re-execution are checked in the accompanying receipt.
- Version Label: neurocomputing_controls_v2.

This is a post-review analysis of frozen evaluation-v2 artifacts whose target
outcomes were already known. It is neither a preregistered experiment nor
independent validation. It adds no detector training and modifies no original
experiment file. Every comparator and both cost scenarios are retained,
including null or unfavorable results.

## Question and fixed scope

The existing global maximum-softmax-probability (MSP) comparator matches the
number of automatic decisions, but its argmax labels can allocate those
decisions differently between safe releases and vulnerable alerts. A smaller
automatic miss rate against that comparator can therefore reflect action
allocation rather than better selection of cases by relevance weighting.

This audit uses all 24 frozen DiverseVul projects, both hashing and frozen
CodeBERT scores, and all five technical seed addresses. It uses the existing
primary budgets `(alpha_vulnerable, alpha_safe) = (0.1, 0.2)` and weight clip
20. The reference policies are unweighted Mondrian, estimated weighting
without a gate, and the full frozen gate. Registration/confirmation membership
is carried as historical metadata and does not create independent confirmation
for these newly specified controls.

## Controls fixed without target labels

1. **Global MSP, exact automatic volume.** Rank by `max(p, 1-p)` in descending
   order, using stable target archive row order for ties. Retain exactly the
   reference's number of singleton decisions and label by argmax. The original
   frozen evaluator assigns `p=0.5` argmax ties to safe; this audit reproduces
   that convention and verifies the archived unweighted matched-MSP sets.

2. **Probability tails, exact action allocation.** Obtain `k_safe` and
   `k_vulnerable` solely from the frozen reference sets, never from target
   outcomes. Sort all target rows once by increasing `p(vulnerable)`, preserving
   archive order within probability ties. Assign the first `k_safe` rows safe,
   the last `k_vulnerable` rows vulnerable, and the remaining rows review.
   Require `k_safe+k_vulnerable <= n`. Using opposite ends of one order makes
   the selections disjoint even when a probability tie straddles both cutoffs.
   At tied low scores, earlier rows are safe; at tied high scores, later rows
   are vulnerable. This is an asymmetric two-threshold policy over all points:
   labels may disagree with the `0.5` argmax. It matches automatic safe count,
   automatic vulnerable count, total automation, and review count exactly.

3. **Source-only isotonic cost ranking, exact automatic volume.** Fit an
   increasing `sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1,
   out_of_bounds='clip')` using only the frozen source calibration probabilities
   and the corresponding source calibration labels. The held-out source roles
   are recovered from hashed source fold packages and their positions must
   match the sealed prediction archive exactly. Apply this source fit to target
   probabilities to obtain `eta`. Use the two fixed scenarios
   `(c_FN,c_FP,c_review)=(10,1,0.1)` and `(50,1,0.1)`, with review residual error
   zero. For each point, choose the label with smaller expected automatic cost:
   `safe_cost=c_FN*eta` and `vulnerable_cost=1-eta`. Tied action costs choose
   vulnerable. Rank by the smaller cost in ascending order, preserving target
   archive row order for ranking ties. Retain exactly the raw weighted or gated
   weighted policy's singleton count. A constant review cost cannot affect this
   fixed-volume ranking. Target labels do not enter fitting, cost choice,
   action labels, reference counts, or ranking. This comparator tests asymmetric
   cost-aware decisions at the same workload; it does not establish calibration
   of `eta` under source-to-target shift.

Every rejected row in a new selector is represented by the two-label set and
goes to review. Original empty sets are also evaluated as review. The original
miscoverage counts are reconciled as automatic wrong labels plus empty sets
excluding the true class; empty-set errors are not automatic releases.

Free-volume empirical threshold optimization is deferred. It is not part of
the reported exact-workload controls, and no threshold is selected from target
outcomes.

## Input verification and source lineage

Before deriving any selector, the analyzer verifies:

- the identical public and original evaluation manifests and original metric
  SHA-256;
- all 240 original decision NPZ files and the complete Cartesian project/seed
  inventory;
- the evaluation-bound final hashing-v8 and codebert-v2 prediction seals,
  including the config, original runner, source package summary, and source and
  target metadata hashes;
- all 480 sealed prediction files: 240 NPZ and 240 metadata JSON;
- every metadata identity and the sealed declarations of no target-label access;
- all 48 per-project source-label and target-position NPZ package hashes, plus
  the source-label vault and metadata hashes;
- exact source calibration and target position equality, and exact target
  probability equality between predictions and evaluation decisions.

The separate target label vault is not opened. Only already-sealed evaluation
decision labels enter retrospective evaluation, after new sets are fixed.
Source labels are read exclusively from calibration-role entries in source
packages. `source_calibration_fits.csv` records, per detector/project/seed, the
source package and prediction NPZ hashes; calibration position, label and
probability hashes; class counts; fitted isotonic knot hashes; target position
and probability hashes; calibrated target score hashes; and cost-ranking
hashes. Array digests include dtype, shape and contiguous array bytes. Frozen
float32 probabilities are converted exactly to float64 for calculations; fitted
knots and target `eta` are saved without reducing their precision.

Generated control archives contain target positions, source-fitted isotonic
knots, calibrated target scores, and all reference/control sets. They do not
copy source or target labels. `analysis_manifest.json` includes original input
seals, generated archive digests, and table digests. Archived CodeBERT seed
addresses are deterministic liblinear replicates, not independent neural
training runs; hashing seed provenance remains as recorded in the final seal.

## Endpoints and descriptive uncertainty

For every project and method, report automatic FN over all vulnerable cases,
automatic FP over all safe cases, automatic decision counts by side, review
count, coverage, and conditional errors. An undefined conditional error for a
review-only policy remains undefined. Non-argmax action counts and rates make
asymmetric relabeling visible. Both illustrative costs are evaluated as

`(c_FN*auto_FN + c_FP*auto_FP + c_review*review_count)/n`.

This cost model assumes perfect reviewed decisions and uses stated illustrative
relative costs; it is not a measurement of analyst review efficacy, time, or
monetary cost. Every matched comparator has the same review count as its
reference, so its fixed review-cost term cancels in paired differences.

Average the five technical seed addresses within each project before paired
analysis. Each detector then has 24 project differences, with projects equally
weighted. Report the median and mean of candidate-minus-comparator project
differences, favorable/unfavorable/tied project counts, and pointwise 95%
percentile intervals for the median. The final calculation uses the exact
ordinary empirical-bootstrap distribution of the two middle order statistics
for 24 draws from 24 equally weighted project differences, including the usual
arithmetic mean of the middle two values. Discrete percentile endpoints are
the smallest support point with cumulative mass at least 0.025 or 0.975. No
Monte Carlo draws or winner-selection rule are used. This pure calculation is
adapted from the separately validated numerical-audit helper and tested against
exhaustive resampling for small even cohorts, including tied observations.

These intervals are exploratory and pointwise, without multiplicity adjustment
or a claim of representative random sampling of software projects. All
comparators, endpoints, cost scenarios, ties, and project rows are retained.
Zero median differences or intervals containing zero do not establish
equivalence. Differences against allocation-matched tails address attribution
within this frozen benchmark and do not identify a causal effect of weighting.

## Eleven-item statistical fallacy scan

| Check | Disposition for this audit |
|---|---|
| Simpson's paradox | Report hashing and CodeBERT separately; retain project-level differences rather than pooling detector/project sample rows. |
| Ecological fallacy | The resampling and comparison unit is the project; no individual-function population inference is made from project medians. |
| Berkson's selection bias | Project inclusion was frozen for the benchmark; selected-project coverage does not imply population representativeness. |
| Collider bias | Allocation matching is a designed conditional diagnostic; it is not presented as a causal adjustment. |
| Base-rate neglect | Preserve `n_vulnerable`, `n_safe`, action-side counts, conditional errors, and per-item cost. |
| Regression to the mean | No extreme project or outcome subgroup is selected; every frozen project is retained. |
| Survivorship bias | Retain zero-automation and failed-gate projects; assert 24 projects and five technical addresses for every method. |
| Look-elsewhere effect | Save every fixed comparator and both cost scenarios; make no significance or winning-model claim. |
| Garden of forking paths | Explicit post-review status; target outcomes already known; describe all new rules and defer unrun free-volume optimization. |
| Correlation versus causation | Describe observed differences and attribution limitations; no claim of a causal mechanism or deployment guarantee. |
| Reverse causality | Neither retrospective source calibration nor matched target workload establishes temporal or causal evidence. |

## Reproduction and deliverables

From the sync repository, run its bundled Python or a compatible local Python:

```powershell
python src/analyze_neurocomputing_controls.py `
  --evaluation F:\CCFA\crc_cpvd_fcs\outputs\extension-v2\evaluation-v2
```

The source packages and final prediction directories are discovered as siblings
of `evaluation-v2`; they can also be supplied through `--inputs` and
`--predictions-root`. The output path defaults to
`outputs/neurocomputing-controls-v2` and must not already exist. Reproduction
uses a new output directory, never overwrites the originals, and checks exact
table and generated archive hashes against the first run.

The outputs are `action_metrics_seed.csv`, `action_metrics_project.csv`,
`action_summary.csv`, `selection_invariants_seed.csv`,
`source_calibration_fits.csv`, `matched_paired_comparisons.csv`,
`matched_cost_comparisons.csv`, `cost_summary.csv`, generated decision archives,
and the verification manifest. Test coverage includes all feasible allocation
counts, tied/disjoint tails, zero automation, invalid inputs, original argmax
ties, source-only calibration, optimal fixed-volume expected-cost selection,
exact empirical bootstrap distributions, project pairing, and tamper/path
verification. Results are descriptive even when integrity checks and
deterministic re-execution succeed.
