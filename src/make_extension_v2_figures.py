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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    edgecolor: str = "#4A4A4A",
    fontsize: float = 7.8,
    weight: str = "normal",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.9,
            facecolor=color,
            edgecolor=edgecolor,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        linespacing=1.15,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#4A4A4A"))


def figure_workflow(stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "Proof-carrying detector-shift knowledge contract",
        ha="center",
        weight="bold",
        fontsize=10.0,
    )
    ax.text(
        0.5,
        0.905,
        "Automation is an evidence-bound entitlement, not a raw confidence threshold",
        ha="center",
        fontsize=7.2,
        color="#4A4A4A",
    )

    stages = [
        (
            (0.015, 0.42),
            0.175,
            "1  EVIDENCE",
            "Shift weights\nAsymmetric sets\nSupport diagnostics",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            (0.225, 0.42),
            0.185,
            "2  REGISTRY",
            "Detector-shift identity\nQualification registry\n$Q \\geq 0.70$",
            "#E9F3EF",
            GREEN,
        ),
        (
            (0.445, 0.42),
            0.205,
            "3  POLICY",
            "K1  budget bound\nK2  evidence complete\nK3  pair qualified\nK4  singleton eligible",
            "#FFF4D8",
            ORANGE,
        ),
        (
            (0.685, 0.42),
            0.145,
            "4  CERTIFICATE",
            "Contract ID\nRule truth values\nEvidence digest",
            "#F1EEF7",
            "#725A9E",
        ),
    ]
    for (x, y), width, heading, body, fill, edge in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.37,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.15,
                facecolor=fill,
                edgecolor=edge,
            )
        )
        ax.text(
            x + width / 2,
            y + 0.315,
            heading,
            ha="center",
            va="center",
            fontsize=6.8,
            weight="bold",
            color="#262626",
        )
        ax.text(
            x + width / 2,
            y + 0.16,
            body,
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=1.35,
            color="#333333",
        )

    for start, end in [
        ((0.190, 0.605), (0.225, 0.605)),
        ((0.410, 0.605), (0.445, 0.605)),
        ((0.650, 0.605), (0.685, 0.605)),
        ((0.830, 0.605), (0.855, 0.605)),
    ]:
        arrow(ax, start, end)

    ax.text(0.922, 0.835, "5  Action", ha="center", fontsize=7.4, weight="bold")
    rounded_box(
        ax,
        (0.855, 0.59),
        0.135,
        0.17,
        "TRUE + valid\nsingleton\nAUTO LABEL",
        "#DDF2EA",
        edgecolor=GREEN,
        fontsize=6.7,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.855, 0.37),
        0.135,
        0.15,
        "FALSE / UNKNOWN\nor mismatch\nREVIEW",
        LIGHT_GRAY,
        edgecolor=GRAY,
        fontsize=6.6,
        weight="bold",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.225, 0.105),
            0.605,
            0.19,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=0.8,
            facecolor="#FAFAFA",
            edgecolor="#AAAAAA",
        )
    )
    ax.text(0.5275, 0.245, "Fail-closed invariants", ha="center", fontsize=7.2, weight="bold", color="#333333")
    for xpos, label in [
        (0.325, "Evidence loss\ncannot create\nauthority"),
        (0.5275, "Detector transfer\ncannot inherit\nauthority"),
        (0.730, "Automatic actions\ncarry verifiable\nproof objects"),
    ]:
        ax.text(
            xpos,
            0.165,
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            linespacing=1.18,
            color="#333333",
        )
    ax.text(
        0.5,
        0.035,
        "Learned evidence proposes an action; the versioned contract decides whether that action is permitted.",
        ha="center",
        fontsize=7.0,
        color="#444444",
    )
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
    qualification: pd.DataFrame,
    stability: pd.DataFrame,
    confirmation: pd.DataFrame,
    threshold: float,
    stem: Path,
    data_dir: Path,
) -> None:
    qualification.to_csv(data_dir / "fig2_gate_summary.csv", index=False)
    confirmation.to_csv(data_dir / "fig2_gate_external_projects.csv", index=False)
    stability.to_csv(data_dir / "fig2_partition_stability.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.48, 0.97], wspace=0.45)
    ax_fixed = fig.add_subplot(grid[0, 0])
    ax_stability = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[0, 2])

    detector_order = ["hashing", "codebert"]
    detector_labels = ["Hashing-SGD", "CodeBERT"]
    detector_colors = {"hashing": BLUE, "codebert": ORANGE}
    for xpos, detector in enumerate(detector_order):
        subset = qualification[qualification["detector"] == detector].set_index(
            "partition"
        )
        values = [
            float(subset.loc["registration", "qualification_score"]),
            float(subset.loc["confirmation", "qualification_score"]),
        ]
        ax_fixed.plot(
            [xpos - 0.10, xpos + 0.10],
            values,
            color=detector_colors[detector],
            linewidth=1.5,
            zorder=2,
        )
        ax_fixed.scatter(
            xpos - 0.10,
            values[0],
            s=46,
            facecolor="white",
            edgecolor=detector_colors[detector],
            linewidth=1.5,
            zorder=3,
        )
        ax_fixed.scatter(
            xpos + 0.10,
            values[1],
            s=46,
            facecolor=detector_colors[detector],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for offset, value in zip((-0.10, 0.10), values):
            ax_fixed.text(
                xpos + offset,
                value + 0.045,
                f"{value:.2f}",
                ha="center",
                fontsize=6.8,
                color="#333333",
            )
    ax_fixed.axhline(threshold, color=RED, linestyle="--", linewidth=1.0)
    ax_fixed.text(
        0.03,
        threshold + 0.025,
        r"registry threshold $Q=0.70$",
        fontsize=6.5,
        color=RED,
    )
    ax_fixed.set_xticks(range(2), detector_labels)
    ax_fixed.set_ylim(0, 1.04)
    ax_fixed.set_ylabel("Qualification score $Q$")
    ax_fixed.set_title("a  Fixed 12/12 split", loc="left", weight="bold", fontsize=8.8)
    ax_fixed.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_fixed.spines[["top", "right"]].set_visible(False)
    ax_fixed.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", linewidth=0, label="Registration"),
            Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", linewidth=0, label="Confirmation"),
        ],
        loc="lower left",
        fontsize=6.5,
        handletextpad=0.25,
    )

    for detector in detector_order:
        subset = stability[stability["detector"] == detector].copy()
        subset = subset[
            np.isfinite(subset["registration_score"])
            & np.isfinite(subset["confirmation_score"])
        ]
        joint = np.minimum(
            subset["registration_score"].to_numpy(float),
            subset["confirmation_score"].to_numpy(float),
        )
        x = np.linspace(0.0, 1.0, 401)
        survival = np.asarray([(joint >= value).mean() for value in x])
        ax_stability.plot(
            x,
            survival * 100,
            color=detector_colors[detector],
            linewidth=2.0,
        )
        threshold_rate = float((joint >= threshold).mean()) * 100
        ax_stability.scatter(
            [threshold],
            [threshold_rate],
            color=detector_colors[detector],
            s=34,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        y_offset = 5 if detector == "hashing" else 5
        ax_stability.text(
            threshold + 0.025,
            threshold_rate + y_offset,
            f"{threshold_rate:.2f}%",
            color=detector_colors[detector],
            fontsize=7.0,
            weight="bold",
        )
    ax_stability.axvline(threshold, color=RED, linestyle="--", linewidth=1.0)
    ax_stability.set_xlim(0.0, 1.0)
    ax_stability.set_ylim(-2, 104)
    ax_stability.set_xlabel(r"Joint split score $\min(Q_{reg}, Q_{conf})$")
    ax_stability.set_ylabel("Deterministic splits at or above score (%)")
    ax_stability.set_title("b  10,000 split stability", loc="left", weight="bold", fontsize=8.8)
    ax_stability.grid(color="#E5E5E5", linewidth=0.55)
    ax_stability.spines[["top", "right"]].set_visible(False)
    ax_stability.text(0.79, 54, "Hashing-SGD", color=BLUE, fontsize=7.0, weight="bold")
    ax_stability.text(0.12, 68, "CodeBERT", color=ORANGE, fontsize=7.0, weight="bold")

    hashing = confirmation[confirmation["detector"] == "hashing"].copy()
    eligible = hashing["contract_action"] == "eligible-for-singleton"
    severe = hashing["severe_raw_violation"].astype(bool)
    triage = np.asarray(
        [
            [int((eligible & ~severe).sum()), int((~eligible & ~severe).sum())],
            [int((eligible & severe).sum()), int((~eligible & severe).sum())],
        ]
    )
    cell_colors = [["#DDF2EA", "#F1F1F1"], ["#FCE8BD", "#DCEAF7"]]
    for row in range(2):
        for column in range(2):
            ax_matrix.add_patch(
                Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=cell_colors[row][column],
                    edgecolor="white",
                    linewidth=2.0,
                )
            )
            ax_matrix.text(
                column + 0.5,
                row + 0.42,
                str(triage[row, column]),
                ha="center",
                va="center",
                fontsize=17,
                weight="bold",
            )
            ax_matrix.text(
                column + 0.5,
                row + 0.72,
                "projects",
                ha="center",
                va="center",
                fontsize=6.4,
                color="#555555",
            )
    ax_matrix.set_xlim(0, 2)
    ax_matrix.set_ylim(2, 0)
    ax_matrix.set_aspect("equal")
    ax_matrix.set_xticks([0.5, 1.5], ["Eligible", "Review-only"])
    ax_matrix.set_yticks([0.5, 1.5], ["Non-severe", "Severe > 0.5"])
    ax_matrix.tick_params(length=0)
    ax_matrix.spines[:].set_visible(False)
    ax_matrix.set_title(
        "c  Hashing confirmation",
        loc="left",
        weight="bold",
        fontsize=8.8,
    )

    fig.suptitle(
        "The qualification firewall grants detector-specific automation authority",
        y=1.015,
        fontsize=9.6,
        weight="bold",
    )
    export(fig, stem)


def figure_risk_coverage(metrics: pd.DataFrame, project: pd.DataFrame, stem: Path, data_dir: Path) -> None:
    methods = {
        "unweighted_mondrian": "Unweighted",
        "estimated_weight_no_gate_clip_20": "Project-weighted",
        "pooled_split_conformal": "Pooled conformal",
        "prom_derived_aps": "PROM-APS",
    }
    selected = project[
        (project["alpha_vulnerable"] == 0.10)
        & (project["alpha_safe"] == 0.20)
        & (project["method"].isin(methods))
    ].copy()
    summary = (
        selected.groupby(["detector", "method"], as_index=False)
        .agg(
            max_relative_violation=("max_relative_violation", "median"),
            singleton_coverage=("singleton_coverage", "median"),
            projects=("target_group", "nunique"),
        )
    )
    summary["label"] = summary["method"].map(methods)
    summary["cohort"] = "all projects"

    external = gate_external_projects(metrics, project)
    controller_rows = []
    for detector in ("hashing",):
        supported = external[(external["detector"] == detector) & (external["supported"])]
        controller_rows.append(
            {
                "detector": detector,
                "method": "vultriage_full_gate_clip_20",
                "max_relative_violation": float(supported["max_relative_violation"].median()),
                "singleton_coverage": float(supported["singleton_coverage"].median()),
                "projects": int(len(supported)),
                "label": "Qualified contract (13/24)",
                "cohort": "supported projects",
            }
        )
    summary = pd.concat([summary, pd.DataFrame(controller_rows)], ignore_index=True)
    summary.to_csv(data_dir / "fig3_primary_automation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True, gridspec_kw={"wspace": 0.25})
    styles = {
        "forced_argmax": ("X", "#555555", 54),
        "unweighted_mondrian": ("o", GRAY, 42),
        "estimated_weight_no_gate_clip_20": ("o", BLUE, 50),
        "pooled_split_conformal": ("D", "#7A6F9B", 40),
        "prom_derived_aps": ("s", GREEN, 42),
        "vultriage_full_gate_clip_20": ("*", ORANGE, 100),
    }
    offsets = {
        "forced_argmax": (-6, 8),
        "unweighted_mondrian": (5, 7),
        "estimated_weight_no_gate_clip_20": (5, 8),
        "pooled_split_conformal": (5, 7),
        "prom_derived_aps": (0, 10),
        "vultriage_full_gate_clip_20": (0, 12),
    }
    ymax = max(1.0, float(summary["max_relative_violation"].max()) * 1.12)
    for ax, detector, title in zip(axes, ("hashing", "codebert"), ("a  Hashing-SGD", "b  CodeBERT")):
        sub = summary[summary["detector"] == detector]
        for row in sub.itertuples(index=False):
            marker, color, size = styles[row.method]
            ax.scatter(
                100 * row.singleton_coverage,
                row.max_relative_violation,
                marker=marker,
                color=color,
                s=size,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            dx, dy = offsets[row.method]
            horizontal_alignment = "center" if row.method in {"prom_derived_aps", "vultriage_full_gate_clip_20"} else "left"
            display_label = row.label
            if detector == "codebert":
                dx, dy = {
                    "forced_argmax": (-5, 10),
                    "unweighted_mondrian": (8, 20),
                    "estimated_weight_no_gate_clip_20": (8, -5),
                    "pooled_split_conformal": (-5, 12),
                    "prom_derived_aps": (0, 10),
                }[row.method]
                if row.method in {"forced_argmax", "pooled_split_conformal"}:
                    horizontal_alignment = "right"
                if row.method == "prom_derived_aps":
                    horizontal_alignment = "center"
            ax.annotate(
                display_label,
                (100 * row.singleton_coverage, row.max_relative_violation),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
                ha=horizontal_alignment,
            )
        unweighted = sub[sub["method"] == "unweighted_mondrian"].iloc[0]
        weighted = sub[sub["method"] == "estimated_weight_no_gate_clip_20"].iloc[0]
        if detector == "hashing":
            ax.annotate(
                "",
                xy=(100 * weighted["singleton_coverage"], weighted["max_relative_violation"]),
                xytext=(100 * unweighted["singleton_coverage"], unweighted["max_relative_violation"]),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "linewidth": 1.5, "mutation_scale": 10},
                zorder=3,
            )
        if detector == "hashing":
            ax.text(0.04, 0.94, "24/24 projects improve", transform=ax.transAxes, va="top", weight="bold", fontsize=7.2, color=BLUE)
        else:
            ax.text(0.04, 0.94, "paired transfer audit", transform=ax.transAxes, va="top", weight="bold", fontsize=7.2, color="#555555")
        ax.axhline(0.5, color=RED, linestyle=":", linewidth=0.8)
        ax.set_xlim(18, 103)
        ax.set_ylim(-0.04, ymax)
        ax.set_xlabel("Median singleton coverage (%)")
        ax.set_title(title, loc="left", weight="bold", fontsize=8.8)
        ax.grid(color="#E0E0E0", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median max relative violation")
    fig.suptitle(
        "Risk-coverage positioning favors project-aware weighting for hashing",
        y=1.01,
        fontsize=9.4,
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
        diagonal = np.linspace(0, limit, 200)
        ax.fill_between(diagonal, 0, diagonal, color="#DDF2EA", alpha=0.45, linewidth=0, zorder=0)
        ax.scatter(sub["baseline_worst_violation"], sub["weighted_worst_violation"], c=colors, s=19, alpha=0.72, edgecolor="white", linewidth=0.3)
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
    axes[0].text(0.64, 0.08, "improved", transform=axes[0].transAxes, fontsize=6.8, color="#35735E", weight="bold")
    fig.suptitle("Hashing improves risk alignment across the frozen operating grid", y=1.02, fontsize=9.5, weight="bold")
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
    parser.add_argument("--qualification-summary", type=Path, required=True)
    parser.add_argument("--partition-stability", type=Path, required=True)
    parser.add_argument("--confirmation-decisions", type=Path, required=True)
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
    qualification = pd.read_csv(args.qualification_summary)
    stability = pd.read_csv(args.partition_stability)
    confirmation = pd.read_csv(args.confirmation_decisions)
    thresholds = qualification["qualification_threshold"].dropna().unique()
    if len(thresholds) != 1:
        raise ValueError("qualification summary must contain one threshold")
    qualification_threshold = float(thresholds[0])
    figure_workflow(args.output / "fig1_workflow")
    figure_gate_validity(
        qualification,
        stability,
        confirmation,
        qualification_threshold,
        args.output / "fig2_gate_validity",
        data_dir,
    )
    figure_risk_coverage(metrics, project, args.output / "fig3_primary_automation", data_dir)
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
            "qualification_summary": args.qualification_summary,
            "partition_stability": args.partition_stability,
            "confirmation_decisions": args.confirmation_decisions,
        }.items()
    }
    script_path = Path(__file__)
    script_digest = sha256(script_path)
    script_label = "src/make_extension_v2_figures.py"
    figure_table_trace = [
        {
            "artifact_id": "fig1_workflow",
            "source_data": [
                {"path": script_label, "sha256": script_digest},
            ],
            "transformation": {
                "script": "src/make_extension_v2_figures.py::figure_workflow",
                "script_sha256": script_digest,
            },
            "caption_claim": "Automatic vulnerability labels require a qualified detector-shift identity, complete three-valued rule evidence, and a verifiable decision certificate.",
            "supported_manuscript_claims": [
                {
                    "claim_id": "K1",
                    "claim_text": "VulTriage represents automation authority as an executable proof-carrying knowledge contract.",
                }
            ],
            "limitations": [
                "The panel specifies executable contract semantics; empirical qualification is evaluated separately in Figure 2."
            ],
        },
        {
            "artifact_id": "fig2_gate_validity",
            "source_data": [
                {
                    "path": str(args.qualification_summary),
                    "sha256": sha256(args.qualification_summary),
                },
                {
                    "path": str(args.partition_stability),
                    "sha256": sha256(args.partition_stability),
                },
                {
                    "path": str(args.confirmation_decisions),
                    "sha256": sha256(args.confirmation_decisions),
                },
            ],
            "transformation": {
                "script": "src/make_extension_v2_figures.py::figure_gate_validity",
                "script_sha256": script_digest,
            },
            "caption_claim": "The qualification firewall registers hashing-SGD, withholds CodeBERT authority, and preserves strong hashing triage on the fixed confirmation half.",
            "supported_manuscript_claims": [
                {
                    "claim_id": "K2",
                    "claim_text": "Hashing-SGD exceeds the Q=0.70 registration threshold and both halves exceed it in 84.94% of deterministic 12/12 partitions."
                }
            ],
            "limitations": [
                "The contract analysis is an explicitly exploratory extension over the frozen external evidence."
            ],
        },
        {
            "artifact_id": "fig3_primary_automation",
            "source_data": [
                {"path": str(args.metrics), "sha256": sha256(args.metrics)},
                {"path": str(args.project_means), "sha256": sha256(args.project_means)},
            ],
            "transformation": {
                "script": "src/make_extension_v2_figures.py::figure_risk_coverage",
                "script_sha256": script_digest,
            },
            "caption_claim": "Project-aware weighting produces the favorable hashing risk-coverage shift, while only the registered hashing identity receives a qualified-controller operating point.",
            "supported_manuscript_claims": [
                {
                    "claim_id": "K3",
                    "claim_text": "Project-aware weighting improves hashing risk alignment on all 24 projects at the primary operating point."
                }
            ],
            "limitations": [
                "Points are project medians and the qualified-controller point is conditional on the registered hashing cohort."
            ],
        },
        {
            "artifact_id": "fig4_risk_alignment",
            "source_data": [
                {"path": str(args.project_means), "sha256": sha256(args.project_means)},
            ],
            "transformation": {
                "script": "src/make_extension_v2_figures.py::figure_alignment",
                "script_sha256": script_digest,
            },
            "caption_claim": "The hashing improvement extends across projects and asymmetric budget pairs, whereas CodeBERT does not show the same directional pattern.",
            "supported_manuscript_claims": [
                {
                    "claim_id": "K4",
                    "claim_text": "Hashing improves, ties, or worsens 202, 6, or 8 of 216 project-budget cells."
                }
            ],
            "limitations": [
                "The nine budget cells within a project are correlated and are summarized descriptively."
            ],
        },
        {
            "artifact_id": "fig5_calibration_sensitivity",
            "source_data": [
                {
                    "path": str(args.calibration_project_summary),
                    "sha256": sha256(args.calibration_project_summary),
                },
            ],
            "transformation": {
                "script": "src/make_extension_v2_figures.py::figure_calibration_sensitivity",
                "script_sha256": script_digest,
            },
            "caption_claim": "Hashing retains lower maximum relative violation at every calibration fraction, with an explicit singleton-coverage trade-off.",
            "supported_manuscript_claims": [
                {
                    "claim_id": "K5",
                    "claim_text": "The direction of the hashing risk-alignment result is preserved across calibration-size sensitivity analyses."
                }
            ],
            "limitations": [
                "Lines summarize project medians and ribbons summarize interquartile ranges across the 24 external projects."
            ],
        },
    ]
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
        "rows_qualification_summary": int(len(qualification)),
        "rows_partition_stability": int(len(stability)),
        "rows_confirmation_decisions": int(len(confirmation)),
        "figure_table_trace": figure_table_trace,
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
