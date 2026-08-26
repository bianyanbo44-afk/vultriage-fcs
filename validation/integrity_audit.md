# Final KBS Integrity Audit

- Audit date: 2026-08-26
- Target manuscript: `paper/kbs_submission/main.tex`
- Target PDF: `paper/kbs_submission/VulTriage_KBS_manuscript.pdf`
- Scientific scope: detector-conditional, auditable knowledge-guided decision support
- Verdict: PASS WITH NOTES

This audit records factual, computational, and packaging checks. It does not
predict editorial acceptance, replace peer review, or replace a professional
similarity screen such as iThenticate.

## Gate Summary

| Dimension | Result | Evidence |
|---|---|---|
| Reference identity | PASS | 28/28 BibTeX records verified through Crossref, OpenAlex, Semantic Scholar, or an explicit authoritative publisher/proceedings fallback. |
| Citation closure | PASS | All 28 bibliography entries are cited in the manuscript; zero orphan references and zero dangling citation keys. |
| Numerical traceability | PASS | Headline counts, paired effects, intervals, direction counts, gate metrics, coverage costs, ablations, sensitivity results, and efficiency values trace to the public CSV/JSON evidence package. |
| Code and configuration consistency | PASS | Methods match the frozen configuration and implementation for detector revisions, seed reuse, risk grid, weighting, support-gate features, and thresholds. |
| Automated tests | PASS | `61 passed, 1 skipped`, including citation-parser regression coverage. |
| LaTeX and PDF | PASS | The KBS manuscript compiles to 25 pages with no undefined references or overfull boxes; rendered pages show no clipping, overlap, blank page, or missing figure. |
| Public snapshot boundary | PASS | The builder enforces file hashes and deny-lists raw datasets, labels, databases, embeddings, checkpoints, predictions, and per-function decisions. |
| Originality screening | NOTE | Local process-language and phrase-risk checks found no blocking issue, but no claim is made that these substitute for the journal's similarity screening. |

## Key Numerical Cross-Checks

| Manuscript statement | Evidence result |
|---|---|
| Main metric rows | 54,000 |
| Project-seed rows | 10,800 |
| Paired comparisons | 450 |
| Calibration-size rows | 14,400 |
| Near-duplicate sensitivity rows | 6,480 |
| Hashing maximum-relative-violation difference | -0.8161119; interval [-1.7218935, -0.3174312]; 24/24 improving |
| Hashing singleton-coverage difference | -0.1587374 |
| CodeBERT maximum-relative-violation difference | -0.0051685; 12/12 split |
| Gate AUROC | PrimeVul 0.35; external hashing 0.8601399; external CodeBERT 0.4285714 |
| Full-gate supported projects | Hashing 13/24; CodeBERT 22/24 |
| Supported-project budget attainment | Hashing 2/13; CodeBERT 0/22 |

The manuscript reports rounded forms of these values. Mechanical zero
violations from review-only projects are not treated as predictive success.

## AI Research Failure-Mode Checklist

| Mode | Status | Evidence and boundary |
|---|---|---|
| 1. Implementation bug passing self-review | CLEAR | Unit tests, artifact validators, evidence validators, hash checks, and independently re-derived summary tables pass. This reduces but cannot mathematically eliminate latent implementation defects. |
| 2. Hallucinated citation | CLEAR | 28/28 reference identities verified; no orphan or dangling citations. |
| 3. Hallucinated experimental result | CLEAR | Every headline result listed above is present in saved CSV/JSON artifacts and is covered by manifest hashes or deterministic re-derivation. |
| 4. Shortcut reliance | CLEAR WITH LIMITATION | Cross-project evaluation, exact/near-duplicate sensitivity, ablations, and detector contrast test major shortcuts. The manuscript explicitly does not claim that all semantic shortcuts or benchmark biases are excluded. |
| 5. Bug reframed as insight | CLEAR | Negative development and CodeBERT results are retained as boundary evidence; no unexpected result is promoted as an unqualified discovery. |
| 6. Methodology fabrication | CLEAR | Methods, configuration, code paths, manifests, detector revisions, label-sealing order, seeds, and thresholds agree. |
| 7. Early frame-lock | CLEAR WITH LIMITATION | The paper is framed around the observed detector contrast and review cost, not a universal guarantee. Missing second-external-dataset, fine-tuned-detector, and analyst-study evidence remains explicit. |

No mode is `SUSPECTED`. No user override is used.

## Claim Boundaries Preserved

The manuscript does not claim state-of-the-art detector performance, a
distribution-free target-risk guarantee for estimated weights, a universal
support certificate, detector-general validity, independent five-fit CodeBERT
variance, hardware-independent speedup, or analyst productivity. The positive
conclusion is limited to the externally evaluated hashing detector-shift pair
and is reported together with its 15.9 percentage-point automation cost.

## Submission Notes

The technical artifact is ready for submission, subject to author-controlled
checks that cannot be inferred from code: the previous submission must be
closed, both authors must approve author order and CRediT roles, Yanbo Bian must
be the sole corresponding author in the portal, and the portal declarations
must match the manuscript. Editorial fit and acceptance remain decisions of the
journal.
