"""Build compact post-review displays from the immutable v3 aggregate release.

No training, selector, evaluator, estimator or released table is changed.
The new tables expose existing assigned-action counts and the entire fixed grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/neurocomputing_revision"
RELEASE = PAPER / "analysis/neural_validation_v3"
DISPLAY = PAPER / "display_additions"
COLLECTIONS = {"diversevul": "DiverseVul", "megavul": "MegaVul"}
DETECTORS = {"finetuned_codebert": "Fine-tuned", "frozen_codebert": "Frozen", "hashing_sgd": "Hashing"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def outputs() -> dict[str, bytes]:
    manifest_bytes = (RELEASE / "public_analysis_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest["files_sha256"].items():
        require(digest((RELEASE / name).read_bytes()) == expected, f"Changed released member {name}")
    structure = pd.read_csv(RELEASE / "project_allocation_structure.csv")
    require(not structure.duplicated(["dataset", "detector", "project"]).any(), "Repeated structural identity")
    fractions = []
    for dataset, label in COLLECTIONS.items():
        group = structure[(structure.dataset == dataset) & (structure.detector == "finetuned_codebert")].copy()
        require(len(group) == (38 if dataset == "diversevul" else 6), "Incomplete primary action roster")
        require(np.allclose(group.weighted_auto_safe + group.weighted_auto_vulnerable + group.weighted_review, group.n), "Weighted action counts do not partition inputs")
        require(np.allclose(group.msp_auto_safe + group.msp_auto_vulnerable + group.weighted_review, group.n), "MSP workload matching differs")
        for policy in ("Weighted", "Matched MSP", "Matched tails"):
            safe = group.msp_auto_safe if policy == "Matched MSP" else group.weighted_auto_safe
            vulnerable = group.msp_auto_vulnerable if policy == "Matched MSP" else group.weighted_auto_vulnerable
            fractions.append(dict(dataset=dataset, policy=policy, projects=len(group),
                                  safe_fraction=float((safe / group.n).median()),
                                  vulnerable_fraction=float((vulnerable / group.n).median()),
                                  review_fraction=float((group.weighted_review / group.n).median()),
                                  paired_vulnerable_change_from_msp=(0.0 if policy == "Matched MSP" else
                                      float(((group.weighted_auto_vulnerable - group.msp_auto_vulnerable) / group.n).median()))))
    fractions_frame = pd.DataFrame(fractions)
    main = [r"\begin{table}[htbp]\centering\small",
            r"\caption{Assigned automatic-action fractions for fine-tuned CodeBERT at $(\alpha_{\mathrm v},\alpha_{\mathrm s})=(0.10,0.20)$ and clip 20. Each entry is a project median after averaging the three actual fits. All 38 DiverseVul and six MegaVul groups enter, including review fallback. Workload matching preserves review count; tail matching also preserves both automatic-label counts, so its fractions repeat the weighted fractions by construction. Separate marginal medians need not sum to one.}",
            r"\label{tab:assignedactions}",
            r"\begin{tabular}{@{}llrrr@{}}\toprule Collection & Policy & Auto safe & Auto vulnerable & Review\\\midrule"]
    for row in fractions_frame.itertuples():
        main.append(f"{COLLECTIONS[row.dataset]} & {row.policy} & {row.safe_fraction:.4f} & {row.vulnerable_fraction:.4f} & {row.review_fraction:.4f}" + r"\\")
    main.append(r"\bottomrule\end{tabular}\end{table}")

    sensitivity, cost_grid = [], []
    for dataset, label in COLLECTIONS.items():
        supported = pd.read_csv(RELEASE / dataset / "domain_supported_paired_summaries.csv")
        subset = supported[(supported.detector == "finetuned_codebert") &
                           supported.cohort.eq("domain_supported_secondary") &
                           supported.candidate.eq("estimated_weight_no_gate") &
                           (supported.baseline == "class_matched_weighted") & (supported.outcome == "auto_fn_rate")].copy()
        require(len(subset) == 27 and not subset.duplicated(["alpha_v", "alpha_s", "clip"]).any(), "Incomplete supported allocation grid")
        require(subset.projects.eq(32 if dataset == "diversevul" else 6).all(), "Changed supported roster")
        require((subset.ci_lower.le(0) & subset.ci_upper.ge(0)).all(), "An allocation interval excludes zero")
        sensitivity.extend(subset[["dataset", "alpha_v", "alpha_s", "clip", "projects", "median_difference",
                                   "ci_lower", "ci_upper", "reductions", "increases", "ties"]].to_dict("records"))
        paired = pd.read_csv(RELEASE / dataset / "paired_project_summaries.csv")
        for detector in DETECTORS:
            for cost in (10, 50):
                subset = paired[(paired.detector == detector) & (paired.baseline == f"source_iso_cost{cost}_free_volume") &
                                paired.cohort.eq("all_projects_deployment") &
                                paired.candidate.eq("estimated_weight_no_gate") &
                                (paired.outcome == f"cost_{cost}")].copy()
                require(len(subset) == 27 and not subset.duplicated(["alpha_v", "alpha_s", "clip"]).any(), "Incomplete cost sensitivity grid")
                require(subset[["median_difference", "ci_lower", "ci_upper"]].notna().all().all(), "Undefined cost grid endpoint")
                weighted = int(subset.ci_upper.lt(0).sum())
                free = int(subset.ci_lower.gt(0).sum())
                zero = int((subset.ci_lower.le(0) & subset.ci_upper.ge(0)).sum())
                require(weighted + free + zero == 27, "Cost interval categories do not partition the grid")
                cost_grid.append(dict(dataset=dataset, detector=detector, miss_cost=cost, cells=27,
                                      median_min=float(subset.median_difference.min()), median_max=float(subset.median_difference.max()),
                                      ci_favors_weighted=weighted, ci_favors_free_volume=free, ci_includes_zero=zero))
    sensitivity_frame = pd.DataFrame(sensitivity).sort_values(["dataset", "alpha_v", "alpha_s", "clip"])
    cost_frame = pd.DataFrame(cost_grid)
    supplement = [r"\begin{table}[H]\centering\small",
                  r"\caption{Complete supported fine-tuned allocation-miss sensitivity. Entries are paired weighted-minus-tail medians. Columns retain every fixed clip for all nine budget pairs, using 32 domain-available DiverseVul groups and all six MegaVul groups. All 27 pointwise 95\% project-bootstrap intervals in each collection include zero. These shared-cohort summaries are descriptive, and are neither independent replications nor equivalence tests. The accompanying CSV retains every interval and direction count.}",
                  r"\label{si:allocsensitivity}",
                  r"\begin{tabular}{@{}ccrrrrrr@{}}\toprule & & \multicolumn{3}{c}{DiverseVul} & \multicolumn{3}{c}{MegaVul}\\",
                  r"$\alpha_{\mathrm v}$ & $\alpha_{\mathrm s}$ & Clip 10 & Clip 20 & Clip 50 & Clip 10 & Clip 20 & Clip 50\\\midrule"]
    for av in (.01, .05, .10):
        for a_s in (.05, .10, .20):
            values = []
            for dataset in COLLECTIONS:
                for clip in (10, 20, 50):
                    row = sensitivity_frame[(sensitivity_frame.dataset == dataset) &
                                            np.isclose(sensitivity_frame.alpha_v, av) & np.isclose(sensitivity_frame.alpha_s, a_s) &
                                            sensitivity_frame["clip"].eq(clip)]
                    require(len(row) == 1, "Sensitivity display identity is ambiguous")
                    value = float(row.iloc[0].median_difference)
                    values.append(f"{0.0 if abs(value) < 5e-10 else value:.4f}")
            supplement.append(f"{av:.2f} & {a_s:.2f} & " + " & ".join(values) + r"\\")
    supplement.extend([r"\bottomrule\end{tabular}\end{table}", "", r"\begin{table}[H]\centering\small",
                       r"\caption{Complete free-volume cost sensitivity. Each row summarizes all nine budget pairs and three clips in the all-project deployment roster. Median ranges and counts of pointwise 95\% intervals describe weighted-minus-source-free-volume cost. W and F count intervals strictly favoring weighted and free-volume policies; Z counts intervals including zero. Every row contains 27 reused grid cells. Costs use correct review, false-alert cost one and review cost 0.1. This complete family does not preserve the primary setting's ordering throughout.}",
                       r"\label{si:costgrid}",
                       r"\begin{tabular}{@{}llrrrrrr@{}}\toprule Collection & Detector & $c_v$ & Min median & Max median & W & F & Z\\\midrule"])
    for row in cost_frame.itertuples():
        supplement.append(f"{COLLECTIONS[row.dataset]} & {DETECTORS[row.detector]} & {row.miss_cost} & {row.median_min:.4f} & {row.median_max:.4f} & {row.ci_favors_weighted} & {row.ci_favors_free_volume} & {row.ci_includes_zero}" + r"\\")
    supplement.append(r"\bottomrule\end{tabular}\end{table}")
    result = {
        "neural_allocation_display.tex": ("\n".join(main) + "\n").encode(),
        "neural_sensitivity_display.tex": ("\n".join(supplement) + "\n").encode(),
        "display_additions/action_fractions.csv": fractions_frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(),
        "display_additions/allocation_sensitivity.csv": sensitivity_frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(),
        "display_additions/cost_grid_summary.csv": cost_frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(),
    }
    additions_manifest = {"schema_version": "trace-neural-display-additions-1", "public_analysis_manifest_sha256": digest(manifest_bytes),
                          "source_script_sha256": digest(Path(__file__).read_bytes()),
                          "original_release_or_generated_tables_modified": False,
                          "files_sha256": {name: digest(data) for name, data in result.items()},
                          "scope": "Existing primary action counts and complete frozen sensitivity grids exposed for readability"}
    result["display_additions/manifest.json"] = (json.dumps(additions_manifest, indent=2, sort_keys=True) + "\n").encode()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Reconstruct display bytes without writing")
    args = parser.parse_args()
    files = outputs()
    for name, data in files.items():
        path = PAPER / name
        if args.check:
            require(path.read_bytes() == data, f"Changed display bytes {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(json.dumps({"status": "PASS", "mode": "check" if args.check else "build", "files": len(files),
                      "original_release_modified": False}, indent=2))


if __name__ == "__main__":
    main()
