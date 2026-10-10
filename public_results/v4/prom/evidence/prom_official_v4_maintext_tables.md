# PROM component tables for the manuscript

The fixed primary comparison uses source-selected LAC/APS/RAPS or their conservative union. Every actual fit selected LAC at alpha=0.50 using source validation F1 only. No target-best-F1 choice is reported. Values are project medians after actual-seed averaging. Error-detection F1 treats review as an error flag and forced-argmax misprediction as the positive event; it is not vulnerable-class classification F1.

| Collection | Detector | Projects | Review | Automatic miss rate | False-alert rate | Error-detection F1 | Cost 10 | Cost 50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diversevul | finetuned_codebert | 38 | 0.5064 | 0.1813 | 0.0000 | 0.2589 | 0.2026 | 0.7930 |
| diversevul | frozen_codebert | 38 | 0.5052 | 0.1880 | 0.0000 | 0.2495 | 0.1849 | 0.6959 |
| diversevul | hashing_sgd | 38 | 0.5080 | 0.2531 | 0.0000 | 0.2170 | 0.2339 | 0.9299 |
| megavul | finetuned_codebert | 6 | 0.5094 | 0.0654 | 0.0000 | 0.1786 | 0.0808 | 0.2030 |
| megavul | frozen_codebert | 6 | 0.5028 | 0.1098 | 0.0000 | 0.1443 | 0.1042 | 0.3177 |
| megavul | hashing_sgd | 6 | 0.5061 | 0.1515 | 0.0000 | 0.1597 | 0.1062 | 0.3167 |

Exact sealed three-action equality with MSP holds in 308/308 project/fit cells; equality with allocation-matched tails holds in 308/308. This is an action-array observation in the complete executed grid. Zero paired intervals do not establish statistical equivalence or full-system superiority.

The protocol also retains the fixed-alpha=0.10 native-case-study LAC/APS/RAPS conservative union as an absolute-endpoint lane. It does not replace source selection or receive a newly chosen control anchor.

| Collection | Detector | Projects | Review | Automatic miss rate | False-alert rate | Error-detection F1 | Cost 10 | Cost 50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diversevul | finetuned_codebert | 38 | 0.9897 | 0.0000 | 0.0000 | 0.1648 | 0.0991 | 0.0991 |
| diversevul | frozen_codebert | 38 | 0.9897 | 0.0000 | 0.0000 | 0.1632 | 0.0990 | 0.0990 |
| diversevul | hashing_sgd | 38 | 0.9893 | 0.0000 | 0.0000 | 0.1550 | 0.0990 | 0.0990 |
| megavul | finetuned_codebert | 6 | 0.9903 | 0.0000 | 0.0000 | 0.1039 | 0.0991 | 0.0991 |
| megavul | frozen_codebert | 6 | 0.9915 | 0.0000 | 0.0000 | 0.0855 | 0.0992 | 0.0992 |
| megavul | hashing_sgd | 6 | 0.9907 | 0.0000 | 0.0000 | 0.0968 | 0.0993 | 0.0993 |

Top-k and the four-expert union remain secondary in the complete aggregate tables. Costs assume correct review and are illustrative; the saved training fits, representation extraction, official component selection/sealing and numerical verification have different runtime scopes. The synthetic eight-class GaussianNB feedback API demonstration is software evidence; the original official eight-CWE benchmark and TRACE neural incremental retraining remain unperformed.
