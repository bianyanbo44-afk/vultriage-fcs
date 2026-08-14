# Extension-v2 CodeBERT Execution Amendment

## Scope

This note records a compute-only amendment made before the DiverseVul label
vault was opened. The frozen detector, source/target packages, seed list,
hyperparameter grid, calibration rules, support gate, outcomes, and evaluation
code are unchanged.

## Deterministic seed audit

The CodeBERT head is scikit-learn `LogisticRegression` with the frozen
`liblinear` solver and the frozen source-train-only `StandardScaler`. A prior
label-free run on the Linux target produced seed-addressed artifacts for seeds
13 and 37. Their calibration and target probability arrays were compared
element by element and were identical (maximum absolute difference 0.0 for
both probability arrays; positions and density-ratio arrays also matched).
The selected `C` and validation PR-AUC were identical as well. This is the
expected behavior of the deterministic binary liblinear fit; `random_state`
does not change the converged solution for this design.

## Materialization rule

The production runner therefore exposes an explicit
`--reuse-codebert-seeds` mode. It fits the frozen head once per target project
using the first frozen seed (13), then materializes the exact same model output
under the five frozen seed addresses (13, 37, 73, 101, 137). Each prediction
metadata file records `technical_seed_reused=true` and the reference seed; the
top-level seal records `seed_reuse_mode=deterministic_liblinear_replicates`.
The evaluator still requires and processes all five seed files, and all
project-level inference continues to average the five seed-addressed outputs.

This amendment reduces wall time without changing any prediction value. It is
not an outcome-driven change: no DiverseVul labels, target risks, gate passes,
or target metric rows were available when the mode was selected.
