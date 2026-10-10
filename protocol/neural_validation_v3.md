# Prospective neural validation of TRACE

This local amendment is made on 10 October 2026, after the original study and
its retrospective control audit were available. It is a prospective design
freeze for new detector fits and previously unevaluated DiverseVul project
outcomes, not a public preregistration or a replacement for the frozen v2 study.
The executable choices are recorded in `configs/neural_validation_v3.json`.

## Questions and independence

The primary question is whether an apparent automatic-miss improvement of
weighted sets remains when both automatic-label counts are matched, for a
genuinely fine-tuned neural detector on additional external projects. The
secondary questions examine the accompanying false-alert trade-off, source-only
cost decisions, model discrimination and weighting sensitivity. Results may
support or contradict the earlier detector-specific observations.

The candidate cohort includes **all 62** originally eligible DiverseVul groups
outside the old 24-group evaluation. Candidate membership is inherited from
the existing label-count manifest, without model-output selection. The complete
candidate-project union is excluded from every PrimeVul source role. Exact
cross-dataset overlaps, conflicting keys and target commits shared with source
roles are excluded. A read-only copy of the existing 128-permutation,
16-band-by-8-row source clone index generates candidates; every flagged pair
is checked at exact lexical-token-set Jaccard 0.90. Candidate generation is
approximate, so the screen does not certify absence of every possible clone.
The original 200/20/20 label-count eligibility rule is applied after screening,
before any model prediction. Every candidate and exclusion count is retained.

Only the PrimeVul official train partition supplies detector training data.
The official valid partition is split by complete commit groups into model
validation and conformal calibration; training-overlapping commits and exact
or identical lexical-token keys are removed. Official test supplies no source
fit or selection data. Neither source file order nor this commit-hash split
is interpreted as a fresh temporal validation. Related before/after functions
in a commit are never assigned to different source roles.

## Actual neural replication

Three runs initialize the same pinned CodeBERT encoder and independent random
heads at seeds 13, 37 and 73. All encoder parameters receive gradient updates.
Training uses three fixed epochs, AdamW at 2e-5, maximum length 512, an effective
batch of 32 and bfloat16 autocast. Gradient checkpointing bounds GPU memory.
The same fixed 4:1 maximum negative-to-positive source sample is used for every
seed, with unweighted cross entropy; no second class-balancing correction is
applied. Model validation and calibration retain their natural class prevalence.
The sampling shift is disclosed rather than interpreted as calibrated target
probability. Source-only isotonic calibration is used by cost selectors.

Each run records trainable parameter counts, initial/final encoder tensor
digests and update norms, per-epoch loss and source-validation average precision,
the selected checkpoint, runtime and peak memory. The best of three epochs is
selected on model-validation average precision alone, ties favoring the earlier
epoch. There is no target-driven checkpoint, epoch, threshold or cohort choice.
A matched frozen-encoder reference has one actual logistic-head fit; it is not
called a multi-seed neural experiment. A hashing reference uses three actual
SGD fits on the same source roles.

## Decisions, seal and analysis

Target metadata and token caches contain no labels. The training/inference
entry point cannot accept the separate target-label vault. Model predictions,
domain ratios and fixed selector decisions are sealed before evaluation opens
that vault. Cross-fitting uses commit-group identifiers so related functions
cannot supply their own held-out domain score through the same commit.

Primary budgets remain alpha_v=0.10 and alpha_s=0.20, with all nine original
budget pairs and clips 10, 20 and 50 retained for secondary sensitivity. The
new source-only free-volume cost policy chooses the least expensive automatic
action only if its source-calibrated expected cost is below review cost. This
policy supplies an independently chosen workload alongside the inherited-count
diagnostic controls. It has no target-label threshold optimization.

The independent analysis unit is the project. Average actual training seeds
within project before forming paired differences. Report the paired median,
project-wise favorable/unfavorable/tied counts and descriptive pointwise 95%
intervals from 10,000 seeded project-bootstrap draws. Record vulnerable and
safe denominators, distinct positive commits and commit concentration. Do not
count functions, seed addresses or repeated predictions as independent units.
Intervals containing zero do not establish equivalence. Costs are illustrative
relative units assuming correct review, not an analyst-cost measurement.

## Completion boundary

Run the frozen candidate cohort and analyses once, retain all failures and
unfavorable results, and write the evidence back into the manuscript. A new
dataset requires its own outcome-blind amendment. Passing software checks or
completing additional experiments does not establish a journal acceptance
probability or justify a guaranteed-acceptance statement.
