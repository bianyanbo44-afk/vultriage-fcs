# Cover letter

18 August 2026

Dear Editor,

Please consider our manuscript, **“VulTriage: Auditing Deploy-or-Review Policies for Cross-Project Vulnerability Detection,”** for publication in *Information and Software Technology*.

The manuscript addresses a software-engineering deployment problem that is not captured by detector ranking metrics alone: when should a cross-project vulnerability detector release an automatic label, and when should it defer to review under asymmetric security costs? VulTriage is a post-hoc decision layer that combines class-asymmetric weighted conformal sets with a label-free support gate. The paper evaluates the layer as an empirical software-engineering artifact rather than proposing another vulnerability detector.

The study uses a frozen, prediction-sealed external-evaluation protocol with PrimeVul used for retrospective gate development and 24 external DiverseVul projects. At the primary operating point, estimated weighting without the gate reduces hashing-SGD's project-paired maximum relative budget violation on all 24 external projects, while reducing singleton coverage by 15.9 percentage points. The same weighting does not improve overall risk alignment for CodeBERT, and the support-gate analysis shows the same detector-conditional boundary. We report project-level uncertainty, refusal cost, duplicate sensitivity, calibration-size sensitivity, resource observations, and a public reproducibility artifact. The manuscript therefore offers a bounded, auditable deployment-policy audit with explicit limitations rather than a universal risk guarantee.

The topic fits the journal's interests in empirical software engineering, software quality and assurance, software analytics, and software security. The manuscript is original, is not under consideration by another journal, and has not been published elsewhere. The authors received no specific funding and declare no competing interests. The manuscript includes a data/code availability statement, CRediT contribution statement, ethics statement, and declaration of generative-AI assistance.

Thank you for your consideration.

Sincerely,

Yanbo Bian (corresponding author)  
Computer Science and Technology Program, Weihai International College, Beijing Jiaotong University  
Email: 24722081@bjtu.edu.cn

Zhihai Wang  
School of Computer Science and Technology, Beijing Jiaotong University  
Email: zhhwang@bjtu.edu.cn
