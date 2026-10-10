# TRACE independent-family amendment v4.1, 11 October 2026

This is an author-authorized metadata-boundary correction after original and v3
target outcomes were known. An initial CodeT5 source-only attempt at seed 13 was
interrupted after the metadata audit exposed known ABRT/libreport vulnerability
and family sharing plus ghostscript-to-mujs aliases. Its source losses and epoch-1
validation AP are known; its checkpoint and all software/input evidence are
retained under `outputs/kbs-validation-v4/independent_model/attempt-1` and excluded
from all reported detector results. No CodeT5 target prediction, domain decision,
or target evaluation was produced by that attempt. This correction is frozen
before three fresh fits and any CodeT5 target comparison. It is neither public
preregistration nor a fresh-unseen-outcome claim.

The question remains whether TRACE allocation, workload and illustrative cost
observations transfer to a separate encoder family under stronger observed
source-family/event screening. It is not an architecture-superiority experiment:
the source cohort differs from v3 and original/v3 results are already known.

## Deterministic source correction

Start from the exact v3.1 separated source rows. Remove every source function in
any complete global source commit named `abrt`, recovered under `github.com/abrt/`,
or recovered in the confirmed `github.com/artifexsoftware/mujs` namespace. This
acts on the known ABRT/libreport family and ghostscript alias rather than merely
disclaiming them. Then remove whole model-validation commits whose OBSERVED
explicit CVEs occur in ORIGINAL source train; remove whole calibration commits
whose observed explicit CVEs occur in ORIGINAL source train OR ORIGINAL source
model validation. Original-set priority is conservative: discarded earlier-role
commits still contribute their observed event identifiers to later-role screens.
The exact metadata audit and exclusion ledger bind every trigger and row.

Surviving roles remain assigned exactly as before. Do not sample replacement
negatives, rebalance classes, reassign commits, or relax the rule from any model
outcome or source class count. The old 4:1 maximum training sample is followed by
these whole-commit exclusions; final class ratios and natural held-out prevalence
are reported explicitly. Source rows are only reindexed, with original position
and row identity preserved. Frozen final roles: train: 15594 functions, 3013 commits, 3106 vulnerable and 12488 safe; model_validation: 6215 functions, 197 commits, 189 vulnerable and 6026 safe; calibration: 4294 functions, 214 commits, 197 vulnerable and 4097 safe.

All original D38 and M6 target projects/functions remain byte-identical to their
previously screened metadata, with the same original target-label vaults and
candidate/exclusion receipts. Both datasets are reported separately. No new
target-label eligibility or favorable-subset selection is used.

## Fresh encoder fits

Use public `Salesforce/codet5-small` at Hugging Face commit
`b1ee9570c289f21b5922b9c768a1ce12957bf968`. Instantiate the complete T5EncoderModel
with six blocks and 512-dimensional masked mean features excluding special
tokens, dropout 0.1, and a random two-class linear head (normal weight std 0.02,
zero bias). The unused pretrained decoder is absent and not counted as trained.
Every instantiated encoder/head parameter is trainable and must receive finite
nonzero gradients; every block must show a nonzero selected-checkpoint query
update. Each fresh seed 13/37/73 reloads the original pinned public encoder and
independently initializes the head and training permutation. No interrupted
attempt checkpoint may be reused or counted as an actual completed fit.

Keep three epochs, maximum 512 tokens, micro-batch 4, effective batch 32,
unweighted cross entropy, AdamW learning rate 2e-5, weight decay 0.01, linear
warmup 0.1, gradient clip 1.0, bfloat16 autocast with float32 parameters and
optimizer states, gradient checkpointing, and the v3 128-row length windows.
Select the maximum natural source model-validation AP over those three epochs,
ties favoring the earlier epoch. Calibration and target labels select nothing.
Bind the fresh config, source/role/token/base hashes, actual checkpoint, script,
seed, tensor digests/updates, gradient evidence, runtime, memory, and failures.
The same source rows are used by all three seeds; these technical fits are not
independent data units. Root clearance reserves the single GPU slot.

## Fixed decisions and analysis

The analysis specification remains unchanged from the first v4 amendment and
v3 policy audit: all nine budget pairs, primary (0.10,0.20); clips 10/20/50,
primary 20; miss costs 10/50, false-alert cost 1, review cost 0.1, assuming correct
review in illustrative units. Domain fits use the final 512-dimensional features
and five folds grouped by global complete commit SHA, with balanced SGD alpha
1e-4 for five epochs and the existing fixed salt. Use only source calibration
labels for conformal sets and source-only isotonic expected costs.

Compare weighted and unweighted sets, global-confidence count matching, disjoint
tails at both automatic-label counts, source-only fixed/free-volume cost policies,
forced argmax, and all-review. Primary: paired project median automatic-miss rate
difference, weighted minus allocation-matched tails at primary budgets/clip.
Secondary: false alerts, coverage, miscoverage, selective error, relative budget
violation, cost, discrimination/calibration and all fixed sensitivities. Average
actual fits within project first. Keep every project, including sealed review
fallback for domain failures; failed fitted-set risk is undefined. Supported-only
analysis is secondary. Descriptive pointwise 95% project-bootstrap intervals use
10,000 draws and seed 20261010; zero-crossing intervals do not prove equivalence.
Costs do not measure analyst time or money.

Label-free predictions and complete project policy decisions for BOTH target
datasets must be sealed, with all planned fit attempts accounted, before the
evaluation command opens either original target-label vault. The seal documents
the no-new-target-tuning boundary, not retroactive blindness to known v3 results.
Retain all unfavorable outcomes, missing predictions, fit failures and refusals.

## What the stronger metadata screen establishes

Known ABRT/libreport and mujs source-family aliases and observed CVE overlaps are
removed by complete commit. Existing exact/identical lexical key and complete
SHA separation is rechecked. Missing CVE identifiers remain unknown, so complete
event independence is not certified. Recovered repository namespaces and known
family lists do not exhaust shared components, forks, or pretraining overlap.

The strict temporal cutoff remains the maximum actual commit timestamp over
EVERY retained source-role commit, usable only with complete unambiguous source
date coverage. Target timestamps must be available and strictly later. Source
dates are incomplete after this correction, so the strict subset is unavailable,
with a saved complete metadata-only roster and no substitute cutoff. CVE years,
file order, publication/release dates and naive guessed timestamps are not dates.
Do not relax chronology or existing project eligibility from results.

Complete the three fresh fits and both full target rosters once. Preserve the
excluded attempt and every original/v3 input, result and seal. The corrected
experiment strengthens known isolation while retaining its limits; it cannot
guarantee journal acceptance.
