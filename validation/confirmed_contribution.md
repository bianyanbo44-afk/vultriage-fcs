# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | VulTriage operationalizes class-asymmetric risk budgets as an auditable weighted-set triage protocol with a frozen label-free support gate, and tests that protocol in a sealed PrimeVul-to-DiverseVul study that exposes when the result transfers across projects and when it remains detector-conditional. |
| Contribution type | New deployment system plus preregistered external empirical analysis. |
| One-sentence reviewer payoff | At the frozen 10% vulnerable and 20% safe budgets, relevance weighting reduces hashing-SGD's project-paired worst relative violation by 0.8161 on all 24 external projects, while making the 15.9 percentage-point automation cost and the failed CodeBERT transfer equally visible. |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | Cross-project vulnerability detectors face changes in code style, prevalence, dependencies, and project conventions, while missed vulnerabilities and false alarms carry asymmetric operational costs. |
| Specific gap | Detector-transfer and calibration work can improve scores, but deployment still lacks a reproducible decision layer that separates an automatic label, mandatory review, label-free support evidence, observed target risk, and the workload cost of refusal. |
| Concrete challenge | Target labels are unavailable at deployment; importance ratios are estimated rather than known; vulnerable examples are rare; calibration evidence may be insufficient for tight budgets; and a gate that refuses everything can appear safe unless coverage and supported-project attainment are reported separately. |
| Why prior work leaves it unresolved | CPVD, DAM2P, CSVD-TF, CD-VulD, and ZSVulD adapt detectors; temperature scaling calibrates confidence; selective prediction and answer-or-defer work formalize rejection; Prom assesses code predictions; weighted conformal and conformal-risk-control work supply relevant theory under stronger assumptions. The missing piece is the security-specific integration and external detector-conditional validation. |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | Construct class-conditional weighted conformal sets with separate vulnerable and safe budgets; estimate source-to-target relevance by cross-fitted domain odds; apply ESS, infinity-mass, domain-shift, and clipping diagnostics through a frozen gate; and route every failed gate to a doubleton review set. |
| Evidence required | A frozen protocol, development and external domains, at least two detector tracks, strong confidence/selective/conformal baselines, label sealing, project-level inference, calibration-size and near-duplicate sensitivity, efficiency accounting, and an explicit distinction among support, automation, and target-label attainment. |
| Evidence available | PrimeVul development plus 24 external DiverseVul projects; hashing-SGD and frozen-CodeBERT/liblinear tracks; five frozen seed addresses; nine budget pairs; 25 method labels; 54,000 main metric rows; 10,000-project bootstrap intervals; exact paired tests with Holm adjustment; independent validation manifests; the hashing primary effect of -0.8161 with 24/24 improvements; a -0.1587 singleton-coverage change; CodeBERT's -0.0052 null overall effect; gate AUROCs 0.35, 0.86, and 0.43; and calibration-size, near-duplicate, and efficiency audits. |
| Evidence missing | No fine-tuned transformer or data-flow detector track, no second independent external dataset beyond DiverseVul, no prospective analyst study, no private-repository validation, and no theorem converting estimated-weight support qualification into finite-sample target-risk control. |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | VulTriage implements asymmetric deploy-or-review triage; the final evaluation is frozen, sealed, dual-dataset, and project-level; estimated weighting improves hashing-SGD worst relative budget violation at the primary point on all 24 external projects; the gain costs 15.9 percentage points of singleton coverage; CodeBERT does not share the overall improvement; and the support gate is externally informative only for hashing-SGD in this study. |
| Claims to soften or avoid | Do not claim universal or finite-sample target-risk control with estimated weights; detector-general gate validity; state-of-the-art vulnerability detection; a hardware-independent speedup; analyst productivity; causal gate effects; or first-in-field priority. Do not count all-review mechanical zeros as predictive success. |
| Novelty risk | Reviewers may view weighted conformal prediction, selective prediction, Prom, or domain-adaptive vulnerability detection as prior art. The response is not component novelty: it is the frozen security deployment interface, explicit unsupported-target state, dual-dataset protocol, detector contrast, and audit trail. |
| Significance risk | The positive result is strongest for the lightweight detector and fails to transfer to CodeBERT. The paper treats this as a scientific boundary rather than hiding it: support and weighting are properties of a detector-shift pair, which is directly relevant to deployment governance. |
