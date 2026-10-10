"""Render complete project-level evidence for the revised operational audit."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operational", type=Path, default=Path("outputs/neurocomputing-revision-v2"))
    parser.add_argument("--controls", type=Path, default=Path("outputs/neurocomputing-controls-v2"))
    parser.add_argument("--output", type=Path, default=Path("paper/neurocomputing_revision/figures"))
    parser.add_argument("--audit-scripts", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.5,
    })
    action = pd.read_csv(args.controls / "action_metrics_project.csv")
    costs = pd.read_csv(args.operational / "cost_paired_sensitivity.csv")
    receipt = {"cohort": "all 24 eligible projects; technical seeds averaged",
               "controls_manifest_sha256": sha256(args.controls / "analysis_manifest.json"),
               "operational_manifest_sha256": sha256(args.operational / "analysis_manifest.json"),
               "figures": {}}

    def save(fig, name, contract):
        if args.audit_scripts:
            sys.path.insert(0, str(args.audit_scripts))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(
                fig, json_out=str(args.output / (name + ".alignment.json")),
                overlay_svg=str(args.output / (name + ".alignment.svg")),
                tolerance_pt=1.5, strict=True,
            )
        fig.savefig(args.output / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(args.output / (name + ".svg"), bbox_inches="tight")
        fig.savefig(args.output / (name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(args.output / (name + ".tiff"), dpi=600, bbox_inches="tight")
        plt.close(fig)
        receipt["figures"][name] = {
            "contract": contract,
            "sha256": {extension: sha256(args.output / (name + "." + extension))
                       for extension in ("pdf", "svg", "png", "tiff")},
        }

    # One question: does the miss advantage survive matched label allocation?
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), sharex="col", sharey="col")
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.88, wspace=.30, hspace=.65)
    frame = action[action.detector == "hashing"]
    weighted = frame[frame.method == "estimated_weight_no_gate"].set_index("target_group").sort_index()
    plot_rows = []
    for row_index, baseline in enumerate(["msp_matched_weighted", "class_matched_weighted"]):
        base = frame[frame.method == baseline].set_index("target_group").reindex(weighted.index)
        assert len(base) == len(weighted) == 24 and base.n.notna().all()
        for col_index, endpoint in enumerate(["auto_fn_rate", "auto_fp_rate"]):
            axis = axes[row_index, col_index]
            x, y = base[endpoint].to_numpy(), weighted[endpoint].to_numpy()
            axis.scatter(x, y, color="#286889", s=28, edgecolor="white", linewidth=.45, zorder=3)
            axis.set_xlabel("Global MSP" if row_index == 0 else "Allocation-matched tails")
            axis.set_ylabel("Weighted sets")
            axis.set_title("Automatic missed-vulnerability rate" if col_index == 0
                           else "Automatic false-alert rate", pad=10)
            axis.tick_params(labelbottom=True)
            axis.annotate(chr(97 + 2 * row_index + col_index), xy=(0, 1),
                          xycoords="axes fraction", xytext=(-13, 9),
                          textcoords="offset points", fontsize=11, fontweight="bold")
            for project, xv, yv in zip(weighted.index, x, y):
                plot_rows.append({"project": project, "baseline": baseline,
                                  "endpoint": endpoint, "baseline_rate": xv, "weighted_rate": yv})
    for column, endpoint in enumerate(["auto_fn_rate", "auto_fp_rate"]):
        data = [weighted[endpoint].to_numpy()]
        for baseline in ["msp_matched_weighted", "class_matched_weighted"]:
            data.append(frame[frame.method == baseline][endpoint].to_numpy())
        upper = max(np.max(values) for values in data) * 1.08
        for row in range(2):
            axis = axes[row, column]
            axis.plot([0, upper], [0, upper], color="#8B9299", ls="--", lw=.8, zorder=1)
            axis.set_xlim(-.012, upper)
            axis.set_ylim(-.012, upper)
    pd.DataFrame(plot_rows).to_csv(data_dir / "matched_tradeoff_projects.csv", index=False)
    save(fig, "fig_matched_tradeoff", {
        "archetype": "quantitative grid",
        "claim": "Coverage-only gain does not persist after action-allocation matching",
        "panels": {"a": "miss trade-off at matched workload", "b": "false-alert cost of the trade-off",
                   "c": "allocation control for missed vulnerabilities", "d": "allocation control for false alerts"},
        "projects_per_panel": 24,
    })

    # The price sequence tests economic boundaries across both detector tracks.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    fig.subplots_adjust(left=.11, right=.98, bottom=.23, top=.78, wspace=.24)
    styles = [("unweighted_mondrian", "#858B93", "Versus unweighted sets"),
              ("msp_matched_weighted", "#286889", "Versus workload-matched MSP")]
    for index, (detector, axis) in enumerate(zip(["hashing", "codebert"], axes)):
        subset = costs[(costs.detector == detector) & (costs.review_cost == .1)
                       & (costs.review_residual_error == 0)
                       & (costs.candidate == "estimated_weight_no_gate")]
        for baseline, color, label in styles:
            values = subset[subset.baseline == baseline].sort_values("miss_cost")
            assert values.miss_cost.tolist() == [1., 5., 10., 20., 50.]
            assert (values.miss_cost > 0).all(), "Log-scale costs must be positive"
            axis.plot(values.miss_cost, values.median_difference, "o-", color=color, ms=3.8, label=label)
            axis.fill_between(values.miss_cost, values.ci_lower, values.ci_upper,
                              color=color, alpha=.13, linewidth=0)
        axis.axhline(0, color="#8B9299", ls="--", lw=.8)
        axis.set_xscale("log")
        axis.set_xticks([1, 5, 10, 20, 50], labels=["1", "5", "10", "20", "50"])
        axis.set_xlabel("Missed-vulnerability cost\n(false-alert cost = 1)")
        axis.set_title("Hashing-SGD" if detector == "hashing" else "Frozen CodeBERT", pad=9)
        axis.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction",
                      xytext=(-13, 9), textcoords="offset points",
                      fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Paired cost difference\n(weighted - comparator)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.51, 1.015), ncol=1, frameon=False)
    costs.to_csv(data_dir / "cost_paired_sensitivity.csv", index=False)
    save(fig, "fig_cost_sensitivity", {
        "archetype": "quantitative grid",
        "claim": "Economic benefit changes with miss cost and detector representation",
        "panels": {"a": "hashing cost boundary", "b": "neural transfer contrast"},
        "uncertainty": "95% exact empirical bootstrap percentile interval",
        "projects_per_point": 24,
    })

    fig, axis = plt.subplots(figsize=(7.2, 2.1))
    fig.subplots_adjust(left=.02, right=.98, bottom=.06, top=.94)
    axis.set_xlim(0, 100); axis.set_ylim(0, 32); axis.axis("off")
    boxes = [(1, "Detector", "Frozen probabilities"),
             (26, "Calibration", "Class budgets + shift weights"),
             (51, "Set decision", "{0}, {1}, or review"),
             (76, "Audit guard", "Identity + support + log")]
    for x, title, body in boxes:
        axis.add_patch(FancyBboxPatch((x, 15), 22, 12, boxstyle="round,pad=.5",
                                     fc="#EDF3F6", ec="#8DA5B5", lw=.8))
        axis.text(x + 11, 23.5, title, ha="center", fontsize=10, fontweight="bold")
        axis.text(x + 11, 18.5, body, ha="center", va="center", fontsize=8.5)
    for x in [23, 48, 73]:
        axis.annotate("", xy=(x + 3, 21), xytext=(x + .3, 21),
                      arrowprops={"arrowstyle": "->", "color": "#687C88", "lw": 1})
    axis.text(50, 8.5, "Automatic safe  |  Automatic vulnerable  |  Human review",
              ha="center", fontsize=10)
    axis.text(50, 2.5, "Audit workload, label allocation, automatic errors, and cost separately",
              ha="center", fontsize=9, color="#4F5A63")
    save(fig, "fig_workflow", {
        "archetype": "schematic",
        "claim": "Policy replay and prediction outcomes are different checks",
        "data": "conceptual explanation only; no invented quantitative observations",
    })
    action.to_csv(data_dir / "action_metrics_project.csv", index=False)
    (args.output / "figure_manifest_v2.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print("Exported three figures with full source data, PDF/SVG/PNG/TIFF, and alignment receipts")


if __name__ == "__main__":
    main()
