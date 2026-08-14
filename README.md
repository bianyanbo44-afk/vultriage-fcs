# VulTriage-FCS

Reproducibility repository for an extension-v2 submission to *Frontiers of Computer Science*. The project studies auditable vulnerability-triage automation under cross-project distribution shift. No private user dataset or pre-existing user manuscript is used.

## Study

**VulTriage: Detector-Conditional Support for Auditable Cross-Project Vulnerability Triage**

VulTriage converts a binary vulnerability detector into three deployment actions: automatically safe, automatically vulnerable, or manual review. It combines class-asymmetric conformal calibration, estimated source-to-target density ratios, and a fail-closed support gate. Exact weighted-conformal theory is stated only under its assumptions; estimated-weight target results are empirical.

Extension-v2 retains the frozen PrimeVul development study and adds external confirmation on the official DiverseVul release. The frozen design selects 24 eligible project groups before model evaluation, evaluates hashing-SGD and a frozen CodeBERT encoder with one deterministic liblinear head replicated at five seed addresses, uses a 3 x 3 asymmetric risk grid, includes PROM-derived and conformal baselines, and reports exact-deduplication and lexical near-duplicate sensitivity analyses. The principal positive result is detector-conditional: weighting improves risk alignment for hashing-SGD but not for CodeBERT, while the support gate has negative development discrimination and mixed external discrimination. Extension-v2 numerical conclusions are authoritative only in a built snapshot containing `public_snapshot_manifest.json`; without that manifest, a checkout must be treated as staging rather than a finalized result release.

Public repository: <https://github.com/bianyanbo44-afk/vultriage-fcs>

## Public Snapshot Boundary

The public snapshot contains:

- frozen v1 and extension-v2 configurations;
- executable preparation, prediction, evaluation, analysis, and figure code;
- tests;
- the final paper and figure source data;
- aggregate evaluation and project-level analysis tables;
- metadata-only cohort manifests and near-duplicate audit artifacts;
- label-free CodeBERT extraction metadata with the immutable model revision;
- immutable artifact hashes and a generated `public_snapshot_manifest.json`.

The public snapshot never contains:

- raw PrimeVul or DiverseVul records;
- the DiverseVul target-label vault or any source/target label package;
- CodeBERT embeddings, sparse feature caches, checkpoints, or downloaded model weights;
- private SQLite indexes, including exact- and near-duplicate work databases;
- per-function prediction or decision archives, including `.npz` and `.npy` files;
- any file larger than the builder's public size limit (50 MB by default).

The builder copies v1 and extension-v2 results by an exact source-to-destination path map, forces the final `figures-v2` assets, includes aggregate `evidence-v2`, `evidence-validation-v2`, `efficiency-v2`, and `efficiency-validation-v2` records, verifies producer-manifest, figure, evidence, efficiency, and prediction-seal hashes, audits CSV schemas and JSON fields for row-level code or labels, and then performs a second denylist and size audit over the complete staged tree. The generated manifest inventories every staged public file except itself. A violation aborts the build before the destination is published.

## Reproduction

Acquire PrimeVul and DiverseVul separately from their official repositories. Do not commit either dataset to this repository. Place the releases under local ignored paths, install `requirements.txt`, and install PyTorch and Transformers separately when reproducing the frozen CodeBERT branch. Use `PYTHONPATH=src` for the commands below.

Development and extension preparation entry points:

```powershell
python src/audit_primevul.py --help
python src/prepare_splits.py --help
python src/prepare_feature_cache.py --help
python src/prepare_e1_inputs.py --help
python src/run_e1_predict.py --help
python src/evaluate_e1.py --help
python src/analyze_e1.py --help
python src/prepare_extension_manifest.py --help
python src/audit_near_duplicates_v2.py --help
python src/prepare_extension_inputs.py --help
python src/prepare_extension_hashing_cache.py --help
python src/prepare_extension_codebert_manifest.py --help
python src/prepare_codebert_embeddings.py --help
python src/fit_support_gate_v2.py --help
```

Sealed extension-v2 prediction, evaluation, and analysis entry points:

```powershell
python src/run_extension_predict.py --help
python src/merge_prediction_parts.py --help
python src/evaluate_extension_v2.py --help
python src/analyze_extension_v2.py --help
python src/analyze_calibration_sensitivity.py --help
python src/analyze_near_duplicate_sensitivity_v2.py --help
python src/make_extension_v2_figures.py --help
```

Prediction generation does not receive the target-label vault. The evaluator is a separate process that verifies the prediction seals before joining target labels. Public aggregate tables can be regenerated from the excluded sealed archives; those per-function archives remain local because of size and disclosure boundaries.

The frozen CodeBERT revision is `3b0952feddeffad0063f274080e3c23d75e7eb39`. A built snapshot records the matching label-free extraction provenance under `public_results/extension-v2/codebert-v1/embedding_metadata.json` without publishing the embedding matrix.

Run the tests with:

```powershell
$env:PYTHONPATH='src'
pytest -q
```

Build the manuscript from `paper/` with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Snapshot Builder

Do not run the builder until the finalized extension-v2 result path is available. It always writes a new destination and never updates an existing checkout in place.

```powershell
.\src\build_public_snapshot.ps1 `
  -SourceProject <path-to-final-extension-v2-source-project> `
  -ExtensionV2Results <path-to-final-hash-sealed-v2-results> `
  -Destination <new-public-snapshot-directory>
```

`-ExtensionV2Results` must be the finalized extension-v2 root produced by `scripts/run_extension_v2_completion.ps1`; it may contain private experiment directories, but the builder copies only the approved aggregate manifests, tables, and near-duplicate sensitivity artifacts at their frozen relative paths. Missing required v2 artifacts, forbidden file types, or an oversized public file cause a hard failure.

## Integrity Boundary

- Dataset acquisition is external and user-controlled; dataset licenses remain authoritative.
- Project selection, detector definitions, risk budgets, seeds, gate rules, and statistical procedures are frozen in `configs/preregistered_extension_v2.json`.
- Prediction seals precede target-label evaluation.
- Near-duplicate exclusion is a named sensitivity cohort and does not replace the primary exact-deduplicated analysis.
- Novelty wording is search-bounded; the manuscript does not claim first or state-of-the-art performance without direct evidence.
