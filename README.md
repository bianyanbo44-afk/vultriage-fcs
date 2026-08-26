# VulTriage reproducibility artifact

This repository contains the reproducibility artifact for the manuscript
**VulTriage: Proof-Carrying Detector--Shift Knowledge Contracts for Selective
Vulnerability Triage**, prepared for *Knowledge-Based Systems* (Elsevier). The
repository name is a historical project identifier.
The KBS manuscript source is intentionally published as part of the
reproducibility artifact; no private user dataset, target-label vault, or
per-function prediction archive is distributed here.

## Study

VulTriage wraps a binary vulnerability detector with a proof-carrying
detector--shift knowledge contract. A versioned registry binds automation
authority to the detector, source domain, target domain, and policy version;
three-valued rules then map class-asymmetric conformal evidence to automatic
safe, automatic vulnerable, or manual-review actions. Automatic actions carry
decision certificates that are verified by complete contract re-execution.
The certificate attests consistent policy execution, not source-code safety or
prediction correctness.

The frozen extension-v2 study uses PrimeVul for retrospective gate development
and the official DiverseVul release for external evaluation. It evaluates a
hashing-SGD detector and a frozen CodeBERT encoder with a deterministic
liblinear head, five technical seed addresses, a 3 x 3 asymmetric risk grid,
and 25 recorded method labels. The independent unit is the target project.
The principal result is detector-conditional: hashing-SGD passes fixed
registration and confirmation and qualifies on both halves in 84.94% of 10,000
deterministic project partitions, while the same registry withholds CodeBERT
automation authority. Independently, estimated weighting improves hashing-SGD
risk alignment on the 24-project external cohort at the primary budget and on
202 of 216 project--budget cells, with a measured singleton-coverage cost.

## Current KBS manuscript

The KBS submission source and compiled PDF are in
[`paper/kbs_submission/`](paper/kbs_submission/). This package contains the
editable LaTeX source, bibliography, figures, figure-source data, highlights,
cover-letter text, and KBS submission checklist. The declaration-of-interest
DOCX is kept in the local submission package and is not published here. Build
it from that directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The earlier IST package in [`paper/ist_submission/`](paper/ist_submission/) is
retained as historical submission provenance only; it is not the current
submission version.

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
python src/analyze_knowledge_contract.py --help
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
The exploratory governance contract is recorded separately in
`configs/knowledge_contract_v1.json`; its SHA-256 is
`837795719587D25D6EDB0BC636D89AE51385C3643677771B5BD1F186CAD1C946`.

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
