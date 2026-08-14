# VulTriage Extension-v2 Preregistration Amendment

Frozen locally: 2026-08-14 01:49:35 China Standard Time (UTC+08:00)  
Status: frozen before any extension-v2 outcome-bearing run  
Machine-readable protocol: `configs/preregistered_extension_v2.json`

Amended locally: 2026-08-14 02:10:51 China Standard Time (UTC+08:00)  
Amendment status: frozen before any extension-v2 model output; it closes
previously underspecified implementation choices and does not change datasets,
budgets, seeds, project eligibility, or outcomes.

## 1. Transparency boundary

This document is an amendment, not a claim that the entire study is prospectively
blind. PrimeVul E1 results were already visible when this amendment was written.
Their metric file has SHA-256
`1B6DFC08F9291D1978C1DC2627CEF37C22D8EF588CBA3EA78E9DF6D971D27181`.
PrimeVul is therefore a development and retrospective-diagnosis domain for the
new support-gate study. The confirmatory extension is the official DiverseVul
release after cross-dataset deduplication. DiverseVul labels may be counted for
the frozen class-size eligibility rule, but they will not be joined to model
outputs until probabilities, weights, gate states, and artifact hashes are
sealed.

The v1 budgets, seeds, target groups, and unfavorable outcomes remain unchanged.
No v1 result will be deleted or relabeled as prospective evidence.

## 2. Confirmatory questions

- **E2-RQ1 (model transport):** Does the triage layer retain its qualitative
  risk--coverage behavior for a frozen pretrained-code representation rather
  than only hashing features?
- **E2-RQ2 (dataset transport):** Does the primary 10% vulnerable / 20% safe
  operating point retain nonzero both-class automation on external DiverseVul
  projects after cross-dataset deduplication?
- **E2-RQ3 (gate validity):** On previously unseen DiverseVul projects, do
  gate-passing targets have lower *pre-gate* maximum class-specific violation
  than gate-failing targets?
- **E2-RQ4 (mechanism and cost):** Which weighting and support components alter
  violation, coverage, and compute cost, and how sensitive are they to source
  calibration size?

Positive support for E2-RQ3 requires directionally lower pre-gate violation and
a bootstrap interval that does not contradict the direction. A failed gate is
never credited with zero post-gate miscoverage: an all-review output makes that
quantity zero mechanically and is not evidence that the gate predicted risk.

## 3. Data and leakage controls

PrimeVul remains exactly as hashed in v1. DiverseVul must come from the official
Wagner Group release. Before any model run, the download URL, file size, SHA-256,
row schema, row counts, project counts, class counts, and license or redistribution
terms are recorded in the canonical acquisition audit
`research/diversevul_acquisition_audit.md`.

For every external target project, PrimeVul is the labeled source domain. Every
PrimeVul row whose exact or frozen-alias project group matches that target is
removed from source training, validation, and calibration. The remaining source
commits are assigned by SHA-256 buckets under salt
`vultriage-extension-source-v2`: 0--69 train, 70--79 model validation, and 80--99
calibration. The complete target project supplies unlabeled domain context. Its
labels remain in a separate vault and are not joined to predictions until the
prediction directory and its hashes are sealed.

Function text is canonicalized by line-ending normalization, removal of trailing
horizontal whitespace, and outer trim. SHA-256 of this text is the primary
duplicate key. Conflicting-label exact duplicates are quarantined. Every
DiverseVul function whose exact key appears anywhere in PrimeVul is excluded
from confirmatory evaluation. A 128-permutation token MinHash audit flags
cross-dataset pairs with estimated Jaccard similarity at least 0.90; exclusion
of flagged pairs is a sensitivity analysis because approximate matching errors
must not silently redefine the primary cohort.

Before target selection, the v1 aliases are reused exactly: `linux` and
`linux-2.6`; `ImageMagick` and `ImageMagick6`; `php` and `php-src`; `qemu`,
`qemu-kvm`, and `qemu_qemu`; and `FFmpeg` as `ffmpeg`. No additional alias is
inferred from capitalization or name similarity. Eligible DiverseVul groups
contain at least 200 functions, 20 vulnerable functions, and 20 safe functions
after exact deduplication. Eligible groups are ranked by vulnerable count, total
count, then normalized group name; all are used up to a maximum of 24. The
resulting list is frozen before model outcomes are evaluated and is never
replaced for poor performance.

## 4. Detectors

The first family is the v1 hashing representation and class-weighted SGD-log-loss
head, including the original five seeds. The second is
`microsoft/codebert-base`: final-layer token states are attention-mask averaged
after excluding special tokens, with 512-token truncation. The encoder is frozen.
A class-weighted L2 logistic head selects `C` from 0.01, 0.1, 1, and 10 using
source validation PR-AUC; ties select the smaller `C`. Five head seeds are 13,
37, 73, 101, and 137. The resolved Hugging Face revision is recorded before
embedding extraction.

The head uses a `StandardScaler` fitted on source-train embeddings only, followed
by scikit-learn `LogisticRegression` with `solver=liblinear`, `penalty=l2`,
`dual=False`, `fit_intercept=True`, `class_weight=balanced`, `max_iter=2000`,
and `tol=1e-6`. The selected `C` is refit on the union of source train and
model-validation rows after applying that source-train-fitted scaler; the
scaler is not refit on the union. Source calibration and target rows are then
transformed with the same frozen scaler. Encoder inference is float16 and
stored embeddings are float32.

This is a frozen CodeBERT detector, not LineVul. End-to-end LineVul is excluded
from the confirmatory matrix because the available 8 GB GPU cannot complete the
full dataset-by-project-by-five-seed design in a reproducible time budget. The
paper may claim transport across two representation families, not equivalence to
LineVul or state-of-the-art detector performance.

The label-free execution amendment documenting deterministic liblinear seed
materialization is recorded in
`research/extension_codebert_execution_amendment_v2.md`.

## 5. Methods and strong baseline

Required comparisons are forced argmax; matched-count MSP and entropy rejection;
temperature-scaled matched MSP; pooled split conformal; unweighted Mondrian;
estimated weighting without refusal; and support-gated VulTriage.

The complete v1 risk grid remains 3 x 3. The primary operating point remains
10% vulnerable / 20% safe because it was fixed and emphasized before this
amendment, not because of DiverseVul outcomes.

PROM is tied to commit `f16b52772123064551b6450f10b308971c4cdd39`, but the
frozen baseline name is **PROM-derived (binary, deterministic bug-fixed
adapter)**, never official PROM or an official reproduction. It preserves LAC,
TopK, APS, and RAPS score definitions while repairing the binary APS/RAPS
one-hot representation, computing the local empirical p-value independently
per test point, and supplying explicit seeds. Local Euclidean distances use the
sealed two-class probability simplex; all calibration points are neighbors below
200 examples and otherwise `floor(0.1 n)` are selected with stable index
tie-breaking. The comparison keeps the repository's strict `p > 1-alpha`,
no-plus-one empirical p-value, and union rejection semantics. For each asymmetric
budget cell, the adapter uses the conservative scalar
`min(alpha_vulnerable, alpha_safe)`; no target-label expert or alpha selection is
permitted. Four expert
outputs, their union, and LAC-only are reported. RAPS uses a deterministic 20%
tuning split and lambda grid 0.001, 0.01, 0.1, 0.2, 0.5, with the smaller lambda
winning ties.

## 6. Support gate development and confirmation

The v1 `positive_neighbours` check is removed. With reciprocal clipping, every
weight is strictly positive, so the count was neither local nor informative.
The manuscript must also remove any claim that it measured local-neighborhood
support.

Candidate gate diagnostics are total and class-specific Kish ESS, maximum and
99th-percentile test-point infinity mass, cross-fitted domain AUROC, and the
fractions clipped at the lower and upper bounds. Any learned scalar combination
or threshold is fitted only on PrimeVul source pseudo-targets. The frozen mapping
is then applied once to DiverseVul projects.

The exact frozen learner has one row per PrimeVul project and budget cell after
averaging the five hashing-detector seeds. Its binary development target is a
severe relative violation:
`max(vulnerable_violation/alpha_vulnerable,
safe_violation/alpha_safe) > 0.5` for the raw clip-20 weighted method. Features
are `log1p` total and class ESS; the project maximum and p99 of the per-target
maximum infinity mass over the two calibration labels; cross-fitted domain AUROC;
and lower/upper calibration-weight clipping fractions. A `StandardScaler` and L2
`LogisticRegression(C=1, solver=liblinear, class_weight=balanced,
max_iter=2000, tol=1e-6, random_state=20260814)` are fit with
leave-one-project-out cross-fitting for the development audit, then refit on all
PrimeVul development rows. The external gate passes only when predicted severe
violation probability is strictly below 0.5 and the raw set has at least one
singleton; equality, missing diagnostics, and nonfinite values fail closed. The
same fitted gate is applied without a detector feature to both detector families.

The primary validation target is maximum class-specific violation of the raw
estimated-weight method before gate refusal. Secondary targets are both-budget
attainment, raw singleton coverage, and conditional utility. If the gate fails
to predict lower external violation, the paper removes the predictive-gate
claim and calls it only a model-based conservative refusal heuristic, not a
validated risk predictor.

## 7. Ablations, calibration size, and efficiency

The fixed ablations are unweighted calibration, estimated weights without gate,
ESS-only gate, infinity-mass-only gate, the complete frozen gate, and clips 10,
20, and 50. Calibration-size sensitivity uses nested deterministic 25%, 50%,
75%, and 100% samples under salt `vultriage-calibration-size-v2`, with five
repetitions and at least 20 examples per class.

Efficiency records representation extraction, head fitting, calibration, and
prediction wall time; peak host RSS; peak allocated and reserved GPU memory;
artifact size; hardware; batch size; and sample count. Comparisons are descriptive
unless measurement conditions match exactly.

## 8. Statistics and stopping

The independent unit is the target project. Seeds are averaged within project.
Paired differences use 10,000 project-bootstrap resamples. Exact two-sided
Wilcoxon tests are emitted only with at least ten nonzero project differences;
binary attainment uses an exact sign test. Holm adjustment is applied within
each outcome, detector, and budget family. Gate discrimination reports AUROC,
AUPRC, and the bootstrapped median pass--fail difference, with uncertainty
emphasized for small project counts.

All eligible targets, methods, budgets, and five seeds complete or fail once
with a logged error. No project, seed, threshold, clip, or budget is added or
removed in response to the results. Negative findings change claims, never the
protocol.
