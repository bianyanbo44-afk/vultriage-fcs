"""Finding-led capacity and delayed-feedback figures from complete v4 aggregates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from make_neural_validation_figures import (
    Auditors, drawing_geometry, measured_layout, pdf_geometry_and_glyphs,
    require, sha, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/kbs-validation-v4/capacity_feedback"
OUTPUT = ROOT / "paper/neurocomputing_revision/figures/v4"
DATASETS = ("diversevul", "megavul")
COUNTS = {"diversevul": 38, "megavul": 6}
COLORS = {"diversevul": "#236B8E", "megavul": "#B87827"}
LABELS = {"diversevul": "DiverseVul (38 projects)", "megavul": "MegaVul (6 projects)"}
SCENARIO = "m10_r0p1_fn0_fp0"
DETECTOR = "finetuned_codebert"


def configure():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "legend.frameon": False,
        "savefig.facecolor": "white", "text.usetex": False,
    })


def primary(frame):
    selected = frame[(frame.detector == DETECTOR) & (frame.scenario == SCENARIO)].copy()
    require(set(selected.dataset) == set(DATASETS), "Both complete dataset aggregates required")
    return selected


def setup(height_mm=91):
    fig, axes = plt.subplots(1, 2, figsize=(6.2992125984, height_mm / 25.4))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.225, top=0.80, wspace=0.59)
    for axis, letter in zip(axes, "ab"):
        axis.text(-0.18, 1.075, letter, transform=axis.transAxes, fontweight="bold", fontsize=11, va="baseline")
        axis.tick_params(length=3, width=0.6)
    legend = [Line2D([0], [0], color=COLORS[d], marker="o", markersize=4, label=LABELS[d]) for d in DATASETS]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.98), ncol=2, columnspacing=1.8)
    return fig, axes


def export_figure(fig, name, axes, ids, context, contract, rows=None):
    """Export this source's drawing and bind QA to its exact PDF and source."""
    output, auditors = context["output"], context["auditors"]
    path = output / name
    require(auditors.directory is not None, "Final figures require the external QA scripts")
    fig.canvas.draw()
    layout = measured_layout(fig, axes, ids)
    layout.update(row_groups=rows or [], column_groups=[])
    write_json(path.with_suffix(".alignment-layout.json"), layout)
    write_json(path.with_suffix(".geometry.json"), drawing_geometry(fig))
    alignment_module = auditors.module("audit_panel_alignment")
    alignment = alignment_module.require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=ids, row_groups=rows or [], column_groups=[],
        json_out=path.with_suffix(".alignment.json"),
        overlay_svg=path.with_suffix(".alignment.svg"), strict=True,
        require_panel_labels=len(axes) > 1, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
    )
    source_module = auditors.module("validate_figure")
    findings = source_module.validate_source(Path(__file__).read_text(encoding="utf-8"), "python")
    source_report = {
        "source_sha256": sha(Path(__file__)),
        "summary": source_module.summarize(findings),
        "findings": [asdict(row) for row in findings],
        "reviewed_warnings": [
            {"check_id": row.check_id,
             "reason": "The 160-mm figure matches the Neurocomputing manuscript text width; final PDF dimensions and glyph scale are checked below."}
            for row in findings if row.level == "WARN" and row.check_id == "FINAL-WIDTH"
        ],
    }
    write_json(path.with_suffix(".source-preflight.json"), auditors.scrub(source_report))
    require(source_report["summary"]["counts"]["FAIL"] == 0, "Plotting source preflight failed")
    require(all(row.check_id == "FINAL-WIDTH" for row in findings if row.level == "WARN"),
            "Plotting source has an unreviewed warning")
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib TRACE v4 aggregate figures"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    pdf_geometry, glyphs = pdf_geometry_and_glyphs(path.with_suffix(".pdf"))
    page_box = pdf_geometry["page_size_pt"]
    require(abs((page_box[2] - page_box[0]) / 72 * 25.4 - 160.0) < 0.01,
            "Figure physical width differs from its contract")
    write_json(path.with_suffix(".pdf-geometry.json"), pdf_geometry)
    pdf_module = auditors.module("audit_pdf_text")
    operator_audit = pdf_module.audit_pdf(path.with_suffix(".pdf").read_bytes(), minimum_pt=5)
    require(operator_audit["auditable"] and not operator_audit["below_minimum_count"],
            "Final PDF operator glyph audit failed")
    write_json(path.with_suffix(".glyph-audit.json"), {**glyphs, "operator_audit": operator_audit})
    collision_module = auditors.module("audit_figure_collisions")
    collision = collision_module.audit_pdf(path.with_suffix(".pdf"))
    collision["pdf"] = "figures/v4/" + name + ".pdf"
    write_json(path.with_suffix(".collision-audit.json"), auditors.scrub(collision))
    collision_module.write_overlay_pdf(path.with_suffix(".pdf"), path.with_suffix(".collision-audit.pdf"), collision["findings"])
    require(collision_module.exit_code(collision, strict=True) == 0,
            "Final PDF collision audit failed; repair and re-export")
    suffixes = [".alignment-layout.json", ".alignment.json", ".alignment.svg",
                ".geometry.json", ".source-preflight.json", ".pdf", ".svg", ".png",
                ".tiff", ".pdf-geometry.json", ".glyph-audit.json",
                ".collision-audit.json", ".collision-audit.pdf"]
    context["figures"][name] = {
        "contract": contract, "width_mm": 160.0,
        "height_mm": layout["figure"]["height_pt"] / 72 * 25.4,
        "alignment_verdict": alignment.get("verdict"), "pdf_glyphs": glyphs,
        "external_QA": {"source_preflight": "PASS; explicit 160-mm width reviewed",
                        "collision": "PASS (strict)"},
        "source_script_sha256": sha(Path(__file__)),
        "visual_review": "Pending inspection of these exact final exports",
        "files_sha256": {name + suffix: sha(path.with_suffix(suffix)) for suffix in suffixes},
        "source_csv_sha256": {member.name: sha(member) for member in (output / "data").glob(name + "*.csv")},
    }
    plt.close(fig)


def trace_workflow(context):
    """Show the comparator logic without representing illustrative outcomes."""
    fig, axis = plt.subplots(figsize=(6.2992125984, 3.9370078740))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set(xlim=(0, 160), ylim=(0, 100))
    axis.axis("off")
    ink, border = "#25323B", "#8A9CA8"

    def box(x, y, width, height, color):
        axis.add_patch(FancyBboxPatch((x, y), width, height,
                                     boxstyle="round,pad=0,rounding_size=1.2",
                                     facecolor=color, edgecolor=border, linewidth=0.8))

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                      mutation_scale=8, linewidth=0.8,
                                      color=border, shrinkA=0, shrinkB=1))

    box(4, 81, 152, 15, "#E7F1F6")
    axis.text(80, 91.4, "Fitted detector scores", ha="center", va="center",
              fontsize=11, fontweight="bold", color=ink)
    axis.text(80, 85.3, "CodeBERT  |  CodeT5 encoder  |  Frozen CodeBERT  |  Hashing-SGD",
              ha="center", va="center", color=ink)
    arrow((80, 81), (80, 76.5))
    box(4, 64, 152, 12, "#F0F2F4")
    axis.text(80, 71.9, "Same target roster and detector outputs within each comparison",
              ha="center", va="center", fontsize=10, color=ink)
    axis.text(80, 66.9, "DiverseVul: 38 groups     MegaVul: 6 groups     Collections analyzed separately",
              ha="center", va="center", color=ink)
    axis.plot([28, 132], [60.5, 60.5], color=border, linewidth=0.8)
    axis.plot([80, 80], [64, 60.5], color=border, linewidth=0.8)
    entries = [
        (4, "#F0F2F4", "1  Workload", ["Fix total automation", "Match confidence selection", "Selection + label allocation"]),
        (56, "#FCF1DE", "2  Allocation", ["Fix both automatic label counts", "Match probability tails", "Selection within label counts"]),
        (108, "#E7F1F6", "3  Operations", ["Choose review volume", "Impose capacity and delay", "Cost, backlog and feedback"]),
    ]
    for x, color, title, lines in entries:
        center = x + 24
        arrow((center, 60.5), (center, 55.5))
        box(x, 27, 48, 28, color)
        axis.text(center, 48.8, title, ha="center", va="center",
                  fontsize=10, fontweight="bold", color=ink)
        for y, line in zip((42.2, 36.8, 31.4), lines):
            axis.text(center, y, line, ha="center", va="center", color=ink)
        arrow((center, 27), (center, 22.5))
    axis.plot([28, 132], [22, 22], color=border, linewidth=0.8)
    arrow((80, 22), (80, 18.5))
    box(4, 4, 152, 14, "#E7F1F6")
    axis.text(80, 13.5, "Class errors  |  Unresolved workload  |  Cost per function",
              ha="center", va="center", fontsize=10, fontweight="bold", color=ink)
    axis.text(80, 7.7, "Paired project comparisons connect the decision changes to their consequences",
              ha="center", va="center", color=ink)
    export_figure(fig, "fig_trace_workflow_v4", [axis], ["workflow"], context,
                  context["contract"]["new_figures"]["fig_trace_workflow_v4"])


def capacity(context):
    curve = primary(pd.read_csv(BASE / "capacity_curve_plot_data.csv"))
    curve = curve[curve.method == "trace_mandatory"].copy()
    paired = primary(pd.read_csv(BASE / "capacity_paired_summary.csv"))
    paired = paired[(paired.candidate == "source_iso_cap") & (paired.baseline == "msp_cap")
                    & (paired.endpoint == "cost_complete_per_function")].copy()
    required = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    fig, axes = setup()
    for dataset in DATASETS:
        a = curve[curve.dataset == dataset].sort_values("capacity_fraction")
        b = paired[paired.dataset == dataset].sort_values("capacity_fraction")
        require(len(a) == len(b) == 7 and np.allclose(a.capacity_fraction, required)
                and np.allclose(b.capacity_fraction, required), "Complete fixed capacity grid required")
        require((a.projects_roster == COUNTS[dataset]).all() and (b.projects_defined == COUNTS[dataset]).all(), "Project denominator changed")
        x = 100 * a.capacity_fraction.to_numpy()
        y = 100 * a.projects_feasible.to_numpy() / a.projects_roster.to_numpy()
        axes[0].plot(x, y, marker="o", markersize=3.5, color=COLORS[dataset])
        bx = 100 * b.capacity_fraction.to_numpy()
        mid, low, high = [b[name].to_numpy() for name in ("median_paired_difference", "ci_lower", "ci_upper")]
        require(np.isfinite(mid).all() and np.isfinite(low).all() and np.isfinite(high).all(), "Undefined complete-roster paired costs")
        axes[1].plot(bx, mid, marker="o", markersize=3.5, color=COLORS[dataset])
        axes[1].fill_between(bx, low, high, color=COLORS[dataset], alpha=0.14, linewidth=0)
    axes[0].set_title("Completion at hard capacity", pad=13)
    axes[0].set_ylabel("Projects completing the roster (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_title("Cost-aware versus confidence", pad=13)
    axes[1].set_ylabel("Paired cost difference")
    axes[1].axhline(0, color="#70777F", linestyle="--", linewidth=0.7, zorder=0)
    for axis in axes:
        axis.set_xlabel("Review capacity (% of functions)", labelpad=8)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.set_xlim(-3, 103)
    curve.to_csv(OUTPUT / "data/fig_capacity_v4_completion.csv", index=False)
    paired.to_csv(OUTPUT / "data/fig_capacity_v4_cost_pairs.csv", index=False)
    export_figure(fig, "fig_capacity_v4", list(axes), list("ab"), context,
                  context["contract"]["new_figures"]["fig_capacity_v4"], rows=[["a", "b"]])


def feedback(context):
    projects = primary(pd.read_csv(BASE / "feedback_project_metrics.csv"))
    projects = projects[(projects.slots_per_batch == 16) & (projects.delay_batches == 3)].copy()
    paired = primary(pd.read_csv(BASE / "feedback_paired_summary.csv"))
    paired = paired[(paired.slots_per_batch == 16) & (paired.delay_batches == 3)
                    & (paired.candidate == "source_iso_rolling") & (paired.baseline == "source_iso_static")
                    & (paired.endpoint == "cost_complete_per_function")].copy()
    differences = []
    fig, axes = setup()
    for x, dataset in enumerate(DATASETS):
        current = projects[projects.dataset == dataset]
        pivot = current.pivot(index="project", columns="method", values="cost_complete_per_function")
        require(len(pivot) == COUNTS[dataset] and pivot[["source_iso_rolling", "source_iso_static"]].notna().all().all(), "Missing paired project")
        delta = pivot.source_iso_rolling - pivot.source_iso_static
        summary = paired[paired.dataset == dataset]
        require(len(summary) == 1, "Missing paired median interval")
        row = summary.iloc[0]
        require(np.isclose(np.median(delta), row.median_paired_difference, atol=1e-10), "Plot median differs from paired release")
        # Project-order offsets separate points without changing any outcome.
        jitter = np.linspace(-0.16, 0.16, len(delta))
        axes[0].scatter(x + jitter, delta, s=15, color=COLORS[dataset], alpha=0.65, linewidths=0)
        axes[0].plot([x, x], [row.ci_lower, row.ci_upper], color="#25323B", linewidth=1.2)
        axes[0].scatter([x], [row.median_paired_difference], marker="D", color="#25323B", s=27, zorder=4)
        history = current[current.method == "source_iso_rolling"].set_index("project")
        require(set(history.index) == set(pivot.index), "Feedback history roster differs")
        count = history.loc[pivot.index, "arrived_reviewed_labels"].to_numpy()
        axes[1].scatter(x + jitter, count, s=15, color=COLORS[dataset], alpha=0.65, linewidths=0)
        median = np.median(count)
        axes[1].plot([x - 0.10, x + 0.10], [median, median], color="#25323B", linewidth=1.8, zorder=4)
        for project, value, n in zip(pivot.index, delta, count):
            differences.append({"dataset": dataset, "project": project, "paired_cost_difference": value, "arrived_reviewed_labels": n})
    axes[0].set_title("Cost after delayed feedback", pad=13)
    axes[0].set_ylabel("Rolling minus static cost")
    axes[0].axhline(0, color="#70777F", linestyle="--", linewidth=0.7, zorder=0)
    axes[1].set_title("History available for updating", pad=13)
    axes[1].set_ylabel("Arrived reviewed labels")
    for axis in axes:
        axis.set_xticks([0, 1], ["DiverseVul", "MegaVul"])
        axis.set_xlim(-0.5, 1.5)
        axis.set_xlabel("Project collection", labelpad=8)
    pd.DataFrame(differences).to_csv(OUTPUT / "data/fig_feedback_v4_projects.csv", index=False)
    paired.to_csv(OUTPUT / "data/fig_feedback_v4_pairs.csv", index=False)
    export_figure(fig, "fig_feedback_v4", list(axes), list("ab"), context,
                  context["contract"]["new_figures"]["fig_feedback_v4"], rows=[["a", "b"]])


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-scripts", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, default=BASE,
                        help="Original completed analysis or the aggregate-only public capacity packet")
    args = parser.parse_args()
    BASE = args.aggregate_dir.resolve()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    (OUTPUT / "data").mkdir(exist_ok=True)
    configure()
    contract_path = OUTPUT / "figure_contract_v4.json"
    import json
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    public_manifest = BASE / "capacity_feedback_v4_public_manifest.json"
    manifest_path = public_manifest if public_manifest.is_file() else BASE / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path == public_manifest:
        require(manifest["status"] == "COMPLETE_AGGREGATE_ONLY"
                and manifest["complete_rosters"] == COUNTS, "Completed full-roster public aggregates required")
        expected = {row["path"]: row["sha256"] for row in manifest["files"]}
        original_analysis_sha = manifest["original_analysis_manifest_sha256"]
    else:
        require(manifest["status"] == "COMPLETE", "Completed scientific aggregates required")
        expected = {row["path"]: row["sha256"] for row in manifest["artifacts"]}
        original_analysis_sha = sha(manifest_path)
    for name in ("capacity_curve_plot_data.csv", "capacity_paired_summary.csv", "feedback_project_metrics.csv", "feedback_paired_summary.csv"):
        require(sha(BASE / name) == expected[name], "Aggregate source changed " + name)
    context = {"output": OUTPUT, "auditors": Auditors(args.audit_scripts), "figures": {}, "contract": contract}
    trace_workflow(context)
    capacity(context)
    feedback(context)
    write_json(OUTPUT / "figure_manifest_v4.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "schema_version": "trace-v4-capacity-feedback-figures-1",
        "source_script_sha256": sha(Path(__file__)), "helper_script_sha256": sha(ROOT / "src/make_neural_validation_figures.py"),
        "capacity_analysis_manifest_sha256": original_analysis_sha,
        "capacity_public_aggregate_manifest_sha256": sha(manifest_path) if manifest_path == public_manifest else None,
        "contract_sha256": sha(contract_path),
        "figures": context["figures"], "visual_inspection": "Pending original-resolution root inspection",
    })
    print("Exported capacity and delayed-feedback figures from verified full-roster aggregates.")


if __name__ == "__main__":
    main()
