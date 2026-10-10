# Official PROM component comparison, 11 October 2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED (protocol recorded before execution)
- Version Label: trace-official-prom-component-v4.1

This is an author-authorized new retrospective comparison. All original TRACE
and neural-validation-v3 outcomes are already known. Fixing this protocol before
the new target-label join prevents tuning the new comparison to its results; it
does not make this an outcome-blind experiment or a public preregistration.
Executable choices are in `configs/prom_official_v4.json`. Root protocol
clearance receipt will bind the exact fixed protocol/config hashes before
evaluation.

The institutional record for the cited CGO 2025 paper (DOI
10.1145/3696443.3708959) explicitly links the authors' repository
https://github.com/HuantWang/PROM. We pin the sole public development branch at
commit `f16b52772123064551b6450f10b308971c4cdd39` (31 January 2025), record
retrieval receipts and license files, and retain an unchanged source checkout.
The top-level license is CC-BY-4.0; `src/LICENSE.txt` also contains an MIT notice
with the literal placeholder copyright holder. Both notices are retained.

The official vulnerability case study uses eight CWE classes, while current
TRACE predicts binary vulnerability. Its reporting chooses the largest error
detection F1 using target labels, and its feedback workflow samples confirmed
target mispredictions for model retraining. These stages do not align with a
deployable comparison on frozen binary predictions. We execute actual official
conformal components, with an explicit binary adaptation and source-only
selection, and separately run native multiclass fidelity checks. We do not call
this the complete official system or an unchanged case-study reproduction.

The existing source roles are unchanged: 4,400 calibration functions and 6,296
model-validation functions from disjoint canonical source commit groups.
Calibration labels fit the official score/top_k/cumulated_score/raps components;
model-validation labels select a scalar alpha and an expert or conservative
union by error-detection F1. LAC, APS, RAPS and their union are primary, matching
the official vulnerability case study. Top-k and the four-expert union are
secondary because the native vulnerability workflow comments top-k out.
The predetermined alpha grid is 0.10 through 0.90
in steps of 0.10. Ties favor less source review, then the frozen expert order,
then smaller alpha. Source model validation already selected the detector
checkpoints; reusing that role for this comparison is disclosed. No target label
selects the policy, alpha, expert, representation, subset, or cost.

Each component runs with `cv=prefit`, `include_last_label=True`, the actual
detector seed as its random state, and the two-column sealed probability vector
as its neighborhood feature. This differs from native CWE token-space features.
The only planned algorithmic binary repair expands sklearn's one-column binary
`label_binarize` result to its two explicit class indicators. A copied runtime
and exact unified diff record that repair; native multiclass outputs must match
the unchanged source exactly. Namespace bootstrap resolves the upstream mixed
`prom`/`src.prom` imports without changing source methods.

We retain the official implementation's globally pooled selected-neighbor
conformity scores, including repeated calibration scores contributed by multiple
test rows. This behavior differs from the frozen independent binary adapter's
per-row local p-values. We do not silently repair it. Actual official execution
produces scores, sets, and p-values; scalar-alpha acceptance requires a singleton
and p-value greater than `1-alpha`. Accepted functions retain the original
detector's argmax action. The primary union accepts only when LAC, APS and RAPS
all accept; the four-expert union is secondary. Selection uses the full source
model-validation role in one batch, and each target project forms its complete
deployment batch. The neighborhood pool is transductive: target features and
project batch membership affect it, but target labels are inaccessible.
Computational chunking would change this pool and is therefore prohibited.

Every existing v3 project and actual detector fit is included: three independent
fine-tuned fits, one frozen fit, and three hashing fits on each of DiverseVul and
MegaVul. PROM decisions are sealed before reopening each old target-label vault.
The original weighted primary decisions are read from their old seals without
regeneration. Controls fix PROM's automation count, additionally fix its two
automatic-label counts, or choose workload using the source isotonic cost map.
We retain fixed-alpha=0.10 expert and union results alongside source selection.

Report class-conditioned automatic misses and false alerts, review fraction,
automation fraction, selective error, action counts, and illustrative cost at
miss prices 10 and 50, false-alert price 1 and review price 0.1. These costs assume
correct review and do not measure analyst time or monetary savings. Average
actual detector seeds within each project before project-paired comparisons and
10,000 seeded project-bootstrap percentile intervals. Retain zero-automation
cases and undefined selective-error denominators explicitly. Pointwise intervals
are descriptive, do not establish equivalence, and do not support general
superiority over the full official PROM system.

Native source checks cover supported multiclass execution, binary failure and
repair, exact multiclass version parity, official selection/union behavior, and
the official feedback-index operation if it can run locally. A feedback-index
smoke or probability calibration is not incremental neural retraining. The
original neural case-study models/taxonomy, deployment-label expert reporting,
feedback acquisition/retraining and official Docker environment remain outside
this component comparison. Any missing or failed step is recorded explicitly.
