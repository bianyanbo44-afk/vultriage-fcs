# Executed official PROM component evidence

The primary result is actual pinned official conformal-component execution with
an explicit binary one-hot repair, probability-space neighborhoods and source
alpha/expert selection. Prior TRACE outcomes were already known. This package
does not reproduce the complete official CWE case study or TRACE incremental
neural retraining. See evidence/prom_official_v4_report.md and the dated protocol.

PROM source is from Huanting Wang, Patrick Lenihan and Zheng Wang, *Enhancing
Deployment-Time Predictive Model Robustness for Code Analysis and Optimization*,
CGO 2025, DOI 10.1145/3696443.3708959, https://github.com/HuantWang/PROM,
commit f16b52772123064551b6450f10b308971c4cdd39. Original source and both original
license notices are retained under third_party/PROM; the separately recorded
two-line binary repair is not silently applied to that source snapshot.
The separate execution amendment sorts the complete duplicated global pool for
the exact same >= counts/denominator. Exact native/binary and edge-case tests,
the real source alpha grid, and all 43 direct target cells match before outcomes.
Both original source and the distinct repair/optimization diffs are retained.
Use the *_portable.diff files for LF patch review. The sealed archival diffs
retain their original Windows doubled-carriage-return text. The included
portable_patch_roundtrip receipt independently applies the clean patch hunks
and reconstructs the actual executed runtime bytes exactly, without changing
the original source, archival diff, decision seal or result.

## Recompute project summaries from the released aggregates

Install numpy==1.26.4 in your own environment, then run from this package root:

```
python scripts/prom_official_v4_recompute.py --input evidence/project_seed_metrics.csv --config configs/prom_official_v4.json --implementation scripts/prom_official_v4.py --expected evidence --output verification
```

The program loads only the executed aggregation function bodies and checks exact
CSV byte parity for project/fit averages, policy medians and paired 10,000-draw
project-bootstrap summaries. It needs no raw labels or predictions and performs
no model inference. Empty selective-error fields remain undefined.
The included receipts record byte-identical reconstruction in the original
NumPy 1.26.4 stack and from the materialized public copy under NumPy 2.2.6.

The source-selection and action-count files contain aggregates per source
configuration or target project/fit, not individual functions. Raw datasets,
source/target rows, target-label vaults, token arrays, model weights, individual
predictions/actions, smoke row arrays, private dependency binaries and the full
official Git repository are deliberately excluded. Public decision-seal hashes
bind those omitted author-side inputs, but cannot by themselves replay raw
decisions. Full raw replay requires independently obtained hash-matched inputs
and the existing TRACE code/workspace layout.
