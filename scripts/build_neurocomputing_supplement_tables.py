"""Generate editable supplementary tables from verified project aggregates."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/neurocomputing_revision"
CONTROL = PAPER / "analysis/controls"


def value(number, digits=3):
    if pd.isna(number):
        return "--"
    return f"{number:.{digits}f}"


def main():
    projects = pd.read_csv(CONTROL / "action_metrics_project.csv")
    cohort = projects[(projects.detector == "hashing")
                      & (projects.method == "estimated_weight_no_gate")]
    cohort = cohort.sort_values("target_group")
    assert len(cohort) == 24 and int(cohort.n.sum()) == 79355
    lines = [
        r"\begin{longtable}{@{}lrrrr@{}}",
        r"\caption{Complete eligible target cohort. Counts precede seed averaging and are identical across detector tracks and seed addresses. The fixed split is historical registry metadata.}\label{si:cohort}\\",
        r"\toprule Project & Functions & Vulnerable & Safe & Partition\\\midrule",
        r"\endfirsthead",
        r"\caption[]{Complete eligible target cohort (continued).}\\",
        r"\toprule Project & Functions & Vulnerable & Safe & Partition\\\midrule",
        r"\endhead",
    ]
    for row in cohort.itertuples():
        label = "Reg." if row.partition == "registration" else "Conf."
        lines.append(f"{row.target_group} & {int(row.n)} & {int(row.n_vulnerable)}"
                     f" & {int(row.n_safe)} & {label}" + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    names = {
        "unweighted_mondrian": "Unweighted sets",
        "estimated_weight_no_gate": "Weighted sets",
        "vultriage_full_gate": "Weighted + frozen gate",
        "msp_matched_weighted": "MSP, weighted workload",
        "msp_matched_gate": "MSP, gate workload",
        "msp_matched_unweighted_mondrian": "MSP, unweighted workload",
        "class_matched_weighted": "Tails, weighted allocation",
        "class_matched_gate": "Tails, gate allocation",
        "class_matched_unweighted_mondrian": "Tails, unweighted allocation",
        "source_iso_cost10_matched_weighted": "Source cost 10, weighted workload",
        "source_iso_cost50_matched_weighted": "Source cost 50, weighted workload",
        "source_iso_cost10_matched_gate": "Source cost 10, gate workload",
        "source_iso_cost50_matched_gate": "Source cost 50, gate workload",
        "forced_argmax": "Forced argmax",
        "all_review": "All review",
    }
    lines += [
        r"\begin{longtable}{@{}llrrrrr@{}}",
        r"\caption{Complete policy summaries. Entries are separate project medians, not a pooled confusion matrix. $n_e$ counts projects with defined selective error. Every coverage and class-error entry retains all 24 projects.}\label{si:allactions}\\",
        r"\toprule Track & Policy & Coverage & $F_v$ & $F_s$ & Sel. error & $n_e$\\\midrule",
        r"\endfirsthead",
        r"\caption[]{Complete policy summaries (continued).}\\",
        r"\toprule Track & Policy & Coverage & $F_v$ & $F_s$ & Sel. error & $n_e$\\\midrule",
        r"\endhead",
    ]
    for detector, track in [("hashing", "Hashing"), ("codebert", "CodeBERT")]:
        for index, method in enumerate(names):
            subset = projects[(projects.detector == detector) & (projects.method == method)]
            assert len(subset) == 24
            label = track if index == 0 else " "
            cells = [value(subset[column].median()) for column in [
                "coverage", "auto_fn_rate", "auto_fp_rate", "selective_error",
            ]]
            lines.append(label + " & " + names[method] + " & " + " & ".join(cells)
                         + f" & {subset.selective_error.notna().sum()}" + r"\\")
        if detector == "hashing":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    pairs = pd.read_csv(CONTROL / "matched_paired_comparisons.csv")
    lines += [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Raw weighted policy minus its allocation-matched tails. Counts are project directions for lower class-error rates, after seed averaging. Zero medians or zero percentile intervals do not imply identical individual decisions or zero future risk.}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule Track & Endpoint & Median & Lower/upper 95\% & Lower & Higher/tie\\\midrule",
    ]
    for detector, track in [("hashing", "Hashing"), ("codebert", "CodeBERT")]:
        for outcome, label in [("auto_fn_rate", "$F_v$"), ("auto_fp_rate", "$F_s$")]:
            row = pairs[(pairs.detector == detector) & (pairs.outcome == outcome)
                        & (pairs.baseline == "class_matched_weighted")
                        & (pairs.candidate == "estimated_weight_no_gate")].iloc[0]
            lines.append(track + " & " + label + " & " + value(row.median_difference, 4)
                         + " & $[" + value(row.ci_lower, 4) + "," + value(row.ci_upper, 4)
                         + "]$ & " + str(row.favorable_projects)
                         + f" & {row.unfavorable_projects}/{row.ties}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    target = PAPER / "supplementary_tables.tex"
    content = "\n".join(lines)
    if target.exists() and target.read_text(encoding="utf-8") == content:
        print("Unchanged: 24-project cohort, 30 policy summaries, and four allocation comparisons")
    else:
        target.write_text(content, encoding="utf-8")
        print("Wrote 24-project cohort, 30 policy summaries, and four allocation comparisons")


if __name__ == "__main__":
    main()
