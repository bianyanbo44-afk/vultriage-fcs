# Extension-v2 Efficiency Audit

- Analysis type: sealed observational audit; no model rerun and no target-label access.
- Hashing: 120 executed heads; median fit 1.828 s (IQR 1.792--1.865); sequential prediction-pipeline wall time 236.485 s.
- CodeBERT: 24 executed deterministic heads; median fit 1584.160 s (IQR 1423.740--1811.780).
- CodeBERT embedding: 305,937 rows in 1521.771 s (201.04 rows/s); peak allocated GPU memory 368,445,440 bytes; peak host RSS 3,439,824,896 bytes.
- CodeBERT four-part execution: critical path 9915.957 s; aggregate part time 38511.854 s.
- Interpretation boundary: the hashing job was sequential whereas CodeBERT used four concurrent parts. These wall-clock observations document resource cost but do not establish a hardware-independent speedup ratio.
- Manifest: `efficiency_manifest.json`.
