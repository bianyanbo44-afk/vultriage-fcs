# Claim Register

| Claim ID | Manuscript claim | Type | Evidence anchors | Citation anchors | Strength | Boundary language |
|---|---|---|---|---|---|---|
| C1 | VulTriage provides an auditable deploy-or-refuse protocol for class-asymmetric vulnerability triage under project shift. | System/method | E15--E17 | C33--C36, C41--C52, C60 | Strong implementation claim | `under the evaluated protocol`; no oracle guarantee for estimated weights. |
| C2 | The preregistered grid contains a broad positive automation region: at a 10% vulnerable budget, 10/12 projects pass support; at a 20% safe budget, those projects retain 57.83% median singleton coverage. | Primary empirical | E06--E09 | C01--C04 | Strong descriptive claim | Support-qualified, not guaranteed risk-satisfying; denominator and budget pair always stated. |
| C3 | Estimated weighting directionally improves maximum class-specific budget alignment in 74/108 project--operating-point pairs. | Secondary empirical | E10--E12 | C45--C52, C60 | Strong count, cautious interpretation | Say `decreases violation` or `risk alignment`; never `dominates` or `guarantees`. |
| C4 | Stricter vulnerable budgets reduce the number of projects for which the frozen support gate permits automation. | Primary empirical/mechanistic | E06 | C41--C52, C60 | Strong within-grid claim | Do not extrapolate below 1%, above 10%, or to arbitrary detectors/datasets. |
| C5 | The value of fail-closed triage is auditability and explicit refusal, obtained at a substantial automation cost relative to unweighted conformal prediction. | Trade-off | E12--E14 | C33--C36, C45--C48 | Strong bounded comparison | Never call all-review output a predictive win. |
| C6 | Existing work already covers project-transfer detectors, code calibration, generic code rejection, and weighted conformal theory; this paper contributes their security-specific operational integration and evaluation. | Novelty positioning | Literature dossier | C13--C40, C41--C60 | Search-bounded synthesis | No first-in-field claim. |
| C7 | The experiment is reproducible and limits target-label leakage through sealed prediction generation and project-level inference. | Reproducibility | E01--E05 | C01--C08 | Strong artifact claim | Sealing does not cure benchmark-label bias or prove external validity. |
| C8 | Findings are limited to a low-cost hashing/SGD detector on PrimeVul; stronger encoders and human-review outcomes remain future tests. | Limitation | E15 plus deviation log | C09--C12, C21--C24, C53--C54 | Mandatory limitation | No model-agnostic, analyst-productivity, or SOTA-detector claim. |

## Claim hierarchy

- **Headline:** C1 + C2.
- **Mechanistic support:** C3 + C4.
- **Trustworthiness:** C5 + C7.
- **Novelty and external-validity boundary:** C6 + C8.

