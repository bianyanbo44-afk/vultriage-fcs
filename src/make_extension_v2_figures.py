"""Build deterministic figures from the sealed extension-v2 evaluation tables.

The script consumes only evaluator/analysis CSV files and support JSON fields;
it never opens the source code, the label vault, or prediction archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#3572B0"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_BLUE = "#DCEAF7"
LIGHT_ORANGE = "#FCE8BD"
LIGHT_GRAY = "#ECECEC"
RED = "#C44E52"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def export(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.9,
            facecolor=color,
            edgecolor="#4A4A4A",
        )
    )
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7.8, linespacing=1.2)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#4A4A4A"))


def figure_workflow(stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.02, 0.64), 0.17, 0.17, "Labeled source\ncalibration code", LIGHT_BLUE)
    rounded_box(ax, (0.02, 0.25), 0.17, 0.17, "Unlabeled target\nproject code", LIGHT_ORANGE)
    rounded_box(ax, (0.25, 0.45), 0.18, 0.20, "Cross-fitted domain\nclassifier and\nrelevance weights", "#F4F4F4")
    rounded_box(ax, (0.49, 0.64), 0.18, 0.17, "Class-asymmetric\nweighted quantiles", LIGHT_BLUE)
    rounded_box(
        ax,
        (0.49, 0.25),
        0.18,
        0.17,
        "Frozen support model\n(label-free diagnostics)",
        LIGHT_ORANGE,
    )
    rounded_box(ax, (0.72, 0.45), 0.10, 0.20, "Decision\ngate", "#F4F4F4")
    rounded_box(ax, (0.85, 0.64), 0.13, 0.17, "Singleton\nautomation", "#DDF2EA")
    rounded_box(ax, (0.85, 0.25), 0.13, 0.17, "Doubleton\nmanual review", LIGHT_GRAY)
    for start, end in [((0.19, 0.72), (0.25, 0.57)), ((0.19, 0.34), (0.25, 0.52)), ((0.43, 0.57), (0.49, 0.71)), ((0.43, 0.52), (0.49, 0.34)), ((0.67, 0.71), (0.72, 0.57)), ((0.67, 0.34), (0.72, 0.52)), ((0.82, 0.57), (0.85, 0.71)), ((0.82, 0.52), (0.85, 0.34))]:
        arrow(ax, start, end)
    ax.text(0.83, 0.67, "pass", ha="center", va="bottom", fontsize=7.0, color=GREEN)
    ax.text(0.83, 0.38, "fail", ha="center", va="top", fontsize=7.0, color=GRAY)
    ax.text(0.5, 0.94, "VulTriage: support is checked before automation", ha="center", weight="bold", fontsize=10.2)
    ax.text(0.5, 0.08, "A failed support check is an explicit review decision, not a forced label.", ha="center", fontsize=7.4, color="#444444")
    export(fig, stem)


def parse_support(value: object) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def support_summary(metrics: pd.DataFrame, project: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[metrics["method"] == "vultriage_full_gate_clip_20"].copy()
    gate["support_pass"] = gate["support"].map(lambda value: bool(parse_support(value).get("passed", False)))
    support = (
        gate.groupby(["detector", "target_group", "alpha_vulnerable", "alpha_safe"], as_index=False)
        .agg(supported=("support_pass", "all"), pass_rate=("support_pass", "mean"))
    )
    cov = project[project["method"] == "vultriage_full_gate_clip_20"][
        ["detector", "target_group", "alpha_vulnerable", "alpha_safe", "singleton_coverage"]
    ]
    return support.merge(cov, on=["detector", "target_group", "alpha_vulnerable", "alpha_safe"], how="left", validate="one_to_one")


def gate_external_projects(metrics: pd.DataFrame, project: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[
        (metrics["method"] == "vultriage_full_gate_clip_20")
        & (metrics["alpha_vulnerable"] == 0.10)
        & (metrics["alpha_safe"] == 0.20)
    ].copy()
    gate["support_pass"] = gate["support"].map(
        lambda value: bool(parse_support(value).get("passed", False))
    )
    gate["gate_probability"] = gate["support"].map(
        lambda value: float(parse_support(value).get("gate_probability", np.nan))
    )
    grouped = gate.groupby(["detector", "target_group"], as_index=False).agg(
        supported=("support_pass", "all"),
        gate_probability=("gate_probability", "mean"),
        seed_addresses=("seed", "nunique"),
    )
    raw = project[
        (project["method"] == "estimated_weight_no_gate_clip_20")
        & (project["alpha_vulnerable"] == 0.10)
        & (project["alpha_safe"] == 0.20)
    ][
        [
            "detector",
            "target_group",
            "max_relative_violation",
            "singleton_coverage",
        ]
    ]
    joined = grouped.merge(
        raw,
        on=["detector", "target_group"],
        how="left",
        validate="one_to_one",
    )
    joined["severe_raw_violation"] = joined["max_relative_violation"] > 0.5
    return joined.sort_values(["detector", "supported", "target_group"]).reset_index(drop=True)


def figure_gate_validity(
    metrics: pd.DataFrame,
    project: pd.DataFrame,
    gate_discrimination: pd.DataFrame,
    gate_seal: dict,
    stem: Path,
    data_dir: Path,
) -> None:
    external = gate_external_projects(metrics, project)
    external.to_csv(data_dir / "fig2_gate_external_projects.csv", index=False)
    rows = [
        {
            "domain": "PrimeVul development LOPO",
            "detector": "frozen gate",
            "projects": int(gate_seal["projects"]),
            "evaluation_rows": int(gate_seal["development_rows"]),
            "severe_prevalence": float(gate_seal["severe_rows"])
            / float(gate_seal["development_rows"]),
            "auroc": float(gate_seal["crossfit_auroc"]),
            "auprc": float(gate_seal["crossfit_auprc"]),
            "passed_projects": np.nan,
            "median_pass_minus_fail": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    ]
    for detector in ("hashing", "codebert"):
        record = gate_discrimination.loc[
            gate_discrimination["detector"] == detector
        ].iloc[0]
        projects = external[external["detector"] == detector]
        rows.append(
            {
                "domain": "DiverseVul external",
                "detector": detector,
                "projects": int(record["projects"]),
                "evaluation_rows": int(record["projects"]),
                "severe_prevalence": float(projects["severe_raw_violation"].mean()),
                "auroc": float(record["gate_auroc_severe_violation"]),
                "auprc": float(record["gate_auprc_severe_violation"]),
                "passed_projects": int(record["passed_all_seed"]),
                "median_pass_minus_fail": float(
                    record["median_raw_violation_pass_minus_fail"]
                ),
                "ci_lower": float(
                    record[
                        "median_raw_violation_pass_minus_fail_bootstrap_ci_lower"
                    ]
                ),
                "ci_upper": float(
                    record[
                        "median_raw_violation_pass_minus_fail_bootstrap_ci_upper"
                    ]
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "fig2_gate_summary.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.88, 1.1, 1.1], wspace=0.38)
    ax_auc = fig.add_subplot(grid[0, 0])
    ax_hash = fig.add_subplot(grid[0, 1])
    ax_code = fig.add_subplot(grid[0, 2], sharey=ax_hash)

    labels = ["PrimeVul\nLOPO", "DiverseVul\nHashing", "DiverseVul\nCodeBERT"]
    colors = [GRAY, BLUE, ORANGE]
    x = np.arange(3)
    auc_values = summary["auroc"].to_numpy(float)
    ax_auc.vlines(x, 0, auc_values, color=colors, linewidth=2.2, alpha=0.8)
    ax_auc.scatter(x, auc_values, color=colors, s=34, zorder=3, edgecolor="white", linewidth=0.5)
    ax_auc.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
    for xpos, value in zip(x, auc_values):
        ax_auc.text(xpos, value + 0.045, f"{value:.2f}", ha="center", fontsize=7.2)
    ax_auc.set_xticks(x, labels)
    ax_auc.set_ylim(0, 1.0)
    ax_auc.set_ylabel("Severe-violation AUROC")
    ax_auc.set_title("a  Gate AUROC", loc="left", weight="bold", fontsize=8.8)
    ax_auc.grid(axis="y", color="#E0E0E0", linewidth=0.55)
    ax_auc.set_axisbelow(True)
    ax_auc.spines[["top", "right"]].set_visible(False)

    ymax = max(1.0, float(external["max_relative_violation"].max()) * 1.08)
    for ax, detector, title, color in (
        (ax_hash, "hashing", "b  Hashing-SGD", BLUE),
        (ax_code, "codebert", "c  CodeBERT", ORANGE),
    ):
        sub = external[external["detector"] == detector]
        for xpos, status in enumerate((False, True)):
            values = np.sort(
                sub.loc[sub["supported"] == status, "max_relative_violation"].to_numpy(float)
            )
            offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.asarray([0.0])
            ax.scatter(
                xpos + offsets,
                values,
                color=color,
                s=22,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(xpos, q25, q75, color="#222222", linewidth=2.2, zorder=4)
                ax.hlines(median, xpos - 0.18, xpos + 0.18, color="#222222", linewidth=1.1, zorder=4)
        record = summary[summary["detector"] == detector].iloc[0]
        ax.axhline(0.5, color=RED, linestyle=":", linewidth=0.8)
        ax.text(
            0.02,
            0.98,
            f"pass-fail median difference\n{record['median_pass_minus_fail']:.2f} "
            f"[{record['ci_lower']:.2f}, {record['ci_upper']:.2f}]",
            transform=ax.transAxes,
            va="top",
            fontsize=6.6,
            color="#333333",
        )
        failed = int((~sub["supported"]).sum())
        passed = int(sub["supported"].sum())
        ax.set_xticks([0, 1], [f"Fail\n(n={failed})", f"Pass\n(n={passed})"])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left", weight="bold", fontsize=8.8)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    ax_hash.set_ylabel("Raw max relative violation")
    plt.setp(ax_code.get_yticklabels(), visible=False)
    fig.suptitle(
        "The frozen support gate is externally informative only for hashing-SGD",
        y=1.01,
        fontsize=10.0,
        weight="bold",
    )
    export(fig, stem)


def figure_automation(metrics: pd.DataFrame, project: pd.DataFrame, stem: Path, data_dir: Path) -> None:
    summary = support_summary(metrics, project)
    selected = summary[(summary["alpha_vulnerable"] == 0.10) & (summary["alpha_safe"] == 0.20)].copy()
    selected.to_csv(data_dir / "fig3_primary_automation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), sharey=False, gridspec_kw={"wspace": 0.4})
    for ax, detector, title in zip(axes, ("hashing", "codebert"), ("a  Hashing-SGD", "b  CodeBERT")):
        sub = selected[selected["detector"] == detector].sort_values(["supported", "singleton_coverage"], ascending=[True, True]).reset_index(drop=True)
        y = np.arange(len(sub))
        for idx, row in sub.iterrows():
            if row["supported"]:
                ax.plot(100 * row["singleton_coverage"], idx, "o", color=BLUE, markersize=4.4)
            else:
                ax.axhspan(idx - 0.38, idx + 0.38, color=LIGHT_GRAY, zorder=0)
                ax.text(1.0, idx, "review only", va="center", color=GRAY, style="italic", fontsize=7.0)
        supported = sub[sub["supported"]]
        if len(supported):
            median = float(supported["singleton_coverage"].median() * 100)
            ax.axvline(median, color=GREEN, linestyle="--", linewidth=1.0, label=f"Median: {median:.1f}%")
        ax.set_yticks(y, sub["target_group"].astype(str))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Singleton coverage (%)")
        ax.set_title(title, loc="left", weight="bold", fontsize=8.8)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if len(supported):
            ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        r"Primary operating point: $\alpha_{v}=0.10$, $\alpha_{s}=0.20$",
        y=1.01,
        fontsize=10.2,
        weight="bold",
    )
    export(fig, stem)


def figure_alignment(project: pd.DataFrame, stem: Path, data_dir: Path) -> None:
    baseline = project[project["method"] == "unweighted_mondrian"]
    weighted = project[project["method"] == "estimated_weight_no_gate_clip_20"]
    keys = ["detector", "target_group", "alpha_vulnerable", "alpha_safe"]
    joined = baseline.merge(weighted, on=keys, suffixes=("_baseline", "_weighted"), validate="one_to_one")
    joined["baseline_worst_violation"] = joined["max_relative_violation_baseline"]
    joined["weighted_worst_violation"] = joined["max_relative_violation_weighted"]
    joined.to_csv(data_dir / "fig4_risk_alignment.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25})
    limit = max(1.0, float(joined[["baseline_worst_violation", "weighted_worst_violation"]].to_numpy().max()) * 1.03)
    budget_colors = {0.01: "#A6A6A6", 0.05: ORANGE, 0.10: BLUE}
    for ax, detector, title in zip(axes, ("hashing", "codebert"), ("a  Hashing-SGD", "b  CodeBERT")):
        sub = joined[joined["detector"] == detector]
        colors = sub["alpha_vulnerable"].map(budget_colors).to_numpy()
        ax.scatter(sub["baseline_worst_violation"], sub["weighted_worst_violation"], c=colors, s=18, alpha=0.72, edgecolor="white", linewidth=0.3)
        ax.plot([0, limit], [0, limit], linestyle="--", color=GRAY, linewidth=0.8)
        difference = sub["weighted_worst_violation"] - sub["baseline_worst_violation"]
        ax.text(
            0.03,
            0.96,
            f"improved / tied / worsened\n{int((difference < 0).sum())} / "
            f"{int((difference == 0).sum())} / {int((difference > 0).sum())}",
            transform=ax.transAxes,
            va="top",
            fontsize=6.8,
            color="#333333",
        )
        ax.set_title(title, loc="left", weight="bold", fontsize=8.8)
        ax.set_xlabel("Unweighted max relative violation")
        ax.grid(color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Estimated-weight max relative violation")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=budget_colors[value],
            markeredgecolor="white",
            markersize=5,
            label=f"{int(value * 100)}%",
        )
        for value in (0.01, 0.05, 0.10)
    ]
    axes[0].legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.63, 1.0),
        title="Vulnerable-class budget",
        title_fontsize=6.7,
        frameon=False,
        fontsize=6.7,
    )
    fig.suptitle("Directional risk alignment across 24 projects and 9 budget pairs", y=1.02, fontsize=10.2, weight="bold")
    export(fig, stem)


def figure_calibration_sensitivity(
    calibration: pd.DataFrame,
    stem: Path,
    data_dir: Path,
) -> None:
    methods = ["unweighted_mondrian", "estimated_weight_no_gate_clip_20"]
    selected = calibration[calibration["method"].isin(methods)].copy()
    rows: list[dict[str, object]] = []
    for (detector, fraction, method), subset in selected.groupby(
        ["detector", "fraction", "method"], sort=True
    ):
        row: dict[str, object] = {
            "detector": detector,
            "fraction": float(fraction),
            "method": method,
            "projects": int(subset["target_group"].nunique()),
        }
        for outcome in ("max_relative_violation", "singleton_coverage"):
            values = subset[outcome].to_numpy(float)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            row[f"{outcome}_q25"] = float(q25)
            row[f"{outcome}_median"] = float(median)
            row[f"{outcome}_q75"] = float(q75)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "fig5_calibration_sensitivity.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.35),
        sharex=True,
        gridspec_kw={"hspace": 0.20, "wspace": 0.28},
    )
    method_style = {
        "unweighted_mondrian": (GRAY, "o", "Unweighted Mondrian"),
        "estimated_weight_no_gate_clip_20": (BLUE, "s", "Estimated weight (clip 20)"),
    }
    for column, detector in enumerate(("hashing", "codebert")):
        for row_index, outcome in enumerate(("max_relative_violation", "singleton_coverage")):
            ax = axes[row_index, column]
            for method in methods:
                sub = summary[
                    (summary["detector"] == detector) & (summary["method"] == method)
                ].sort_values("fraction")
                color, marker, label = method_style[method]
                x = 100 * sub["fraction"].to_numpy(float)
                median = sub[f"{outcome}_median"].to_numpy(float)
                q25 = sub[f"{outcome}_q25"].to_numpy(float)
                q75 = sub[f"{outcome}_q75"].to_numpy(float)
                if outcome == "singleton_coverage":
                    median, q25, q75 = 100 * median, 100 * q25, 100 * q75
                ax.fill_between(x, q25, q75, color=color, alpha=0.13, linewidth=0)
                ax.plot(
                    x,
                    median,
                    color=color,
                    marker=marker,
                    markersize=3.6,
                    linewidth=1.15,
                    label=label,
                )
            ax.grid(color="#E0E0E0", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([25, 50, 75, 100])
            if row_index == 0:
                label = "a" if column == 0 else "b"
                detector_title = "Hashing-SGD" if detector == "hashing" else "CodeBERT"
                ax.set_title(f"{label}  {detector_title}", loc="left", weight="bold", fontsize=8.8)
            else:
                ax.set_xlabel("Calibration fraction (%)")
    axes[0, 0].set_ylabel("Median max relative violation")
    axes[1, 0].set_ylabel("Median singleton coverage (%)")
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=6.7)
    fig.suptitle(
        "Hashing risk alignment persists across frozen calibration fractions",
        y=1.01,
        fontsize=10.0,
        weight="bold",
    )
    export(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--project-means", type=Path, required=True)
    parser.add_argument("--gate-discrimination", type=Path, required=True)
    parser.add_argument("--gate-seal", type=Path, required=True)
    parser.add_argument("--calibration-project-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics = pd.read_csv(args.metrics)
    project = pd.read_csv(args.project_means)
    gate_discrimination = pd.read_csv(args.gate_discrimination)
    gate_seal = json.loads(args.gate_seal.read_text(encoding="utf-8"))
    calibration = pd.read_csv(args.calibration_project_summary)
    figure_workflow(args.output / "fig1_workflow")
    figure_gate_validity(
        metrics,
        project,
        gate_discrimination,
        gate_seal,
        args.output / "fig2_gate_validity",
        data_dir,
    )
    figure_automation(metrics, project, args.output / "fig3_primary_automation", data_dir)
    figure_alignment(project, args.output / "fig4_risk_alignment", data_dir)
    figure_calibration_sensitivity(
        calibration,
        args.output / "fig5_calibration_sensitivity",
        data_dir,
    )
    stems = [
        "fig1_workflow",
        "fig2_gate_validity",
        "fig3_primary_automation",
        "fig4_risk_alignment",
        "fig5_calibration_sensitivity",
    ]
    assets = {}
    for stem in stems:
        for suffix in ("pdf", "png", "svg", "tiff"):
            path = args.output / f"{stem}.{suffix}"
            assets[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    data_assets = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(data_dir.glob("*.csv"))
    }
    source_inputs = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in {
            "metrics": args.metrics,
            "project_means": args.project_means,
            "gate_discrimination": args.gate_discrimination,
            "gate_seal": args.gate_seal,
            "calibration_project_summary": args.calibration_project_summary,
        }.items()
    }
    manifest = {
        "metrics": str(args.metrics),
        "project_means": str(args.project_means),
        "figures": stems,
        "assets": assets,
        "data_assets": data_assets,
        "source_inputs": source_inputs,
        "raster_export_dpi": 600,
        "tiff_compression": "lzw",
        "rows_metrics": int(len(metrics)),
        "rows_project_means": int(len(project)),
        "rows_gate_discrimination": int(len(gate_discrimination)),
        "rows_calibration_project_summary": int(len(calibration)),
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
