# VulTriage-FCS

Isolated research project for a new submission to *Frontiers of Computer Science*. No pre-existing user dataset or manuscript is used.

## Paper

**VulTriage: Auditable Automation under Project Shift for Vulnerability Detection**

VulTriage converts a binary vulnerability detector into three deployment actions: automatically safe, automatically vulnerable, or manual review. It combines class-asymmetric conformal calibration, estimated source--target density ratios, and a label-free support gate. A failed support check returns the doubleton review set. Exact weighted-conformal theory is stated only under its assumptions; estimated-weight target results are empirical.

## Completed experiment

The frozen E1 experiment completed without a retry: 12 project-disjoint targets, five technical seeds, nine budget pairs, 17 method variants, 130 sealed prediction artifacts, and 9,495 metric rows. At the 10% vulnerable / 20% safe operating point, 10/12 projects pass the label-free support audit and retain 57.83% median singleton coverage. Across the full grid, estimated weighting reduces worst-class budget violation in 74/108 project--operating-point pairs. These are positive empirical results, not target-risk guarantees.

Final manuscript:

- `paper_rewriting_output/final_paper/paper.pdf`
- `paper_rewriting_output/final_paper/paper.docx`
- `paper_rewriting_output/final_paper/main.tex`
- `paper_rewriting_output/reports/2026-08-13/citation_verification_final.html`

Public repository: <https://github.com/bianyanbo44-afk/vultriage-fcs>

## Reproduction

Clone PrimeVul separately from <https://github.com/DLVulDet/PrimeVul>. The public repository intentionally excludes the dataset, feature cache, source-label packages, and per-function prediction/decision archives. Place the original PrimeVul JSONL files under `data/external/primevul_original/`, install `requirements.txt`, and run the frozen stages below with `PYTHONPATH=src`:

```powershell
python src/audit_primevul.py --help
python src/prepare_splits.py --help
python src/prepare_feature_cache.py --help
python src/prepare_e1_inputs.py --help
python src/run_e1_predict.py --help
python src/evaluate_e1.py --help
python src/analyze_e1.py --help
python src/make_paper_figures.py --help
```

The exact arguments and immutable hashes are recorded in the manifests under `public_results/` and the manuscript validation materials. Run the unit tests with `$env:PYTHONPATH='src'; pytest -q` on Windows PowerShell or `PYTHONPATH=src pytest -q` on POSIX systems.

Reproduction entry points:

- `configs/preregistered_experiment.json`
- `src/run_e1_predict.py`
- `src/evaluate_e1.py`
- `src/analyze_e1.py`
- `src/make_paper_figures.py`
- `outputs/exp-e1-cpu-full-evaluation/evaluation_manifest.json`
- `outputs/exp-e1-cpu-full-analysis-v2/analysis_manifest.json`

## Integrity boundary

- Public data: PrimeVul original release, MIT-licensed repository metadata.
- Local user data: excluded.
- Experimental numbers: generated only by the committed public-data pipeline and tied to immutable manifests/hashes.
- Novelty wording: search-bounded; the manuscript does not claim “first” or state-of-the-art performance.
