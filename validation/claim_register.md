# Claim Register

| Claim ID | Manuscript claim | Type | Evidence anchors | Citation anchors | Strength | Boundary language |
|---|---|---|---|---|---|---|
| C1 | VulTriage provides an auditable deploy-or-review interface for class-asymmetric vulnerability triage under project shift. | System/method | E01, E04, E18 | C33--C36, C41--C52, C60 | Strong implementation claim | Say `under the evaluated protocol`; estimated weights do not inherit an oracle target-risk guarantee. |
| C2 | The final study is a preregistered, prediction-sealed dual-dataset evaluation with a PrimeVul development domain, 24 external DiverseVul projects, two detector tracks, five frozen seed addresses, nine budget pairs, and 25 method labels. | Study design | E01--E05 | C01--C12 | Strong artifact claim | CodeBERT uses one deterministic fit per project copied to five addresses; do not call those five independent fits. |
| C3 | At the primary 10%/20% operating point, estimated weighting improves hashing-SGD maximum relative violation on all 24 external projects, with median signed difference -0.8161 and interval `[-1.7219,-0.3174]`. | Primary empirical | E10--E11 | C45--C52, C60 | Strong project-level result | Restrict the conclusion to hashing-SGD and the evaluated external cohort. |
| C4 | The hashing improvement carries a median 15.9 percentage-point loss of singleton coverage across all 24 projects. | Primary trade-off | E12 | C33--C36, C45--C48 | Strong project-level result | Singleton coverage is automation volume, not accuracy or analyst productivity. |
| C5 | The same estimated-weight policy does not improve CodeBERT maximum relative violation overall. | Negative detector contrast | E06, E13 | C21--C28, C45--C52 | Strong bounded negative result | Do not pool detector tracks or claim model-agnostic effectiveness. |
| C6 | The frozen support gate is externally discriminative for hashing-SGD but not for CodeBERT, and its PrimeVul development AUROC is below chance. | Gate validity | E07--E09 | C41--C52, C60 | Strong detector-conditional result | The gate is a label-free diagnostic, not a detector-general support certificate or causal intervention. |
| C7 | Support qualification, singleton coverage, and label-revealed budget attainment are distinct; review-only gate failures create mechanical zero violations and must not count as predictive successes. | Interpretive safeguard | E14 | C33--C40, C45--C52 | Strong protocol interpretation | Do not report overall full-gate attainment without the supported-project denominator. |
| C8 | Calibration-size and near-duplicate sensitivity analyses preserve the positive hashing direction and the negative CodeBERT contrast. | Robustness | E15--E16 | C01--C12 | Strong descriptive sensitivity claim | Sensitivity analyses are not independent replications and do not cover all semantic duplicates. |
| C9 | The experiment is auditable and limits target-label leakage through frozen configuration, prediction seals, project-level inference, validation manifests, and a privacy-bounded public snapshot. | Reproducibility | E01--E05, E18 | C01--C12 | Strong artifact claim | Sealing and public summaries do not eliminate benchmark-label bias or reproduce excluded raw/intermediate data automatically. |
| C10 | Existing work already covers cross-project detectors, calibration, selective prediction, code-model deferral, and weighted conformal theory; this paper contributes their security-specific operational integration and detector-conditional external evaluation. | Novelty positioning | Literature dossier | C13--C60 | Search-bounded synthesis | No first-in-field, SOTA-detector, universal-validity, or theorem claim. |

## Claim hierarchy

- Headline: C1 + C3 + C4.
- Generalization boundary: C5 + C6.
- Interpretive safeguards: C7 + C8.
- Trustworthiness and positioning: C2 + C9 + C10.
