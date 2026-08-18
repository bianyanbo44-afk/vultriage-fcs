# VulTriage reproducibility artifact

This repository contains the reproducibility artifact for the manuscript
**VulTriage: Auditing Deploy-or-Review Policies for Cross-Project Vulnerability
Detection**, prepared for *Information and Software Technology* (Elsevier).
The repository name and URL are historical project identifiers; the current
submission target is IST. No private user dataset or pre-existing user
manuscript is distributed here.

## Study

VulTriage converts a binary vulnerability detector into three deployment
actions: automatic safe, automatic vulnerable, or manual review. It combines
class-asymmetric weighted conformal sets, estimated source-to-target relevance
weights, and a frozen label-free support gate. A failed support check returns a
review set. Exact weighted-conformal guarantees are stated only under their
assumptions; all estimated-weight target results in this artifact are empirical.

The frozen extension-v2 study uses PrimeVul for retrospective gate development
and the official DiverseVul release for external evaluation. It evaluates a
hashing-SGD detector and a frozen CodeBERT encoder with a deterministic
liblinear head, five technical seed addresses, a 3 x 3 asymmetric risk grid,
and 25 recorded method labels. The independent unit is the target project.
The principal result is detector-conditional: estimated weighting improves
hashing-SGD risk alignment on the 24-project external cohort at the primary
budget, with a measured singleton-coverage cost, while the same weighting does
not improve CodeBERT overall. The support gate is reported as a descriptive,
detector-conditional diagnostic rather than a target-risk certificate.

## Current IST manuscript

The complete IST submission source and PDF are in
[`paper/ist_submission/`](paper/ist_submission/). The package includes the
Elsevier class, bibliography style, figures, figure source data, highlights,
and cover-letter text. Build it from that directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The older files in `paper/` that use `fcs.cls` are retained as historical
project scaffolding for provenance and are not the current submission.

## Reproduction

Acquire PrimeVul and DiverseVul separately from their official releases. Do
not commit either dataset, target-label vaults, embeddings, feature caches, or
per-function predictions to this repository. Install `requirements.txt` and
use `PYTHONPATH=src` for the commands below.

Preparation and analysis entry points:

```powershell
python src/audit_primevul.py --help
python src/prepare_splits.py --help
python src/prepare_feature_cache.py --help
python src/prepare_extension_manifest.py --help
python src/audit_near_duplicates_v2.py --help
python src/prepare_extension_inputs.py --help
python src/prepare_extension_hashing_cache.py --help
python src/prepare_extension_codebert_manifest.py --help
python src/fit_support_gate_v2.py --help
python src/run_extension_predict.py --help
python src/evaluate_extension_v2.py --help
python src/analyze_extension_v2.py --help
python src/analyze_calibration_sensitivity.py --help
python src/analyze_near_duplicate_sensitivity_v2.py --help
python src/make_extension_v2_figures.py --help
```

Prediction generation does not receive the target-label vault. The evaluator
verifies prediction seals before joining target labels. Public results contain
project-level and aggregate summaries, not the excluded row-level archives.
Run the tests with:

```powershell
$env:PYTHONPATH='src'
pytest -q
```

The immutable protocol is recorded in
`configs/preregistered_extension_v2.json`. The configuration SHA-256 is
`8AA39B0920D8CD2CFEFBF8C28109754F1B2DFA6049E17565122C6968E199AAD2`.

## Public snapshot boundary

The snapshot includes executable code, frozen configurations, tests, figure
source data, aggregate evaluation tables, evidence and efficiency summaries,
validation reports, and metadata-only cohort manifests. It excludes:

- raw PrimeVul and DiverseVul records;
- target-label vaults and label packages;
- CodeBERT weights, embeddings, and feature caches;
- private SQLite indexes; and
- per-function prediction or decision archives.

The generated `public_snapshot_manifest.json` inventories every staged public
file except the manifest itself. Validation manifests record the prediction
seals, row counts, hashes, and the no-target-label-access boundary. The public
artifact is intended to make the protocol auditable; reproducing the results
still requires obtaining the public datasets and rerunning the compute-heavy
stages locally.

## Integrity and claims boundary

The artifact does not claim state-of-the-art detector performance, a universal
support gate, a distribution-free target-risk guarantee for estimated weights,
a hardware-independent speedup, or an independent five-fit CodeBERT variance
estimate. Review-only gate failures are reported separately from supported
projects so mechanical zero violations are not counted as predictive success.

Public repository: <https://github.com/bianyanbo44-afk/vultriage-fcs>
