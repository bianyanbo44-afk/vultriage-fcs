"""Generate submission figures from the sealed E1 metric artifacts.

The script is intentionally deterministic and exports the exact figure data
beside the graphics. It never reads raw function text or user-provided data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


BLUE = "#3572B0"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#7A7A7A"
LIGHT_BLUE = "#DCEAF7"
LIGHT_ORANGE = "#FCE8BD"
LIGHT_GRAY = "#ECECEC"
RED = "#C44E52"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
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
    for suffix in ("png", "pdf", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#4A4A4A",
    fontsize: float = 8.0,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.9,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#4A4A4A",
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.02, 0.63), 0.17, 0.18, "Labeled source\ncalibration code", LIGHT_BLUE, BLUE)
    rounded_box(ax, (0.02, 0.25), 0.17, 0.18, "Unlabeled target-\nproject code", LIGHT_ORANGE, ORANGE)
    rounded_box(ax, (0.25, 0.44), 0.18, 0.20, "Cross-fitted domain\nclassifier\n" + r"$\hat{w}(x)$", "#F4F4F4")
    rounded_box(ax, (0.48, 0.63), 0.18, 0.18, "Class-asymmetric\nweighted quantiles\n" + r"$\alpha_v,\,\alpha_s$", LIGHT_BLUE, BLUE)
    rounded_box(ax, (0.48, 0.25), 0.18, 0.18, "Support audit\nESS · neighbors ·\n" + r"$\infty$-mass", LIGHT_ORANGE, ORANGE)
    rounded_box(ax, (0.70, 0.44), 0.10, 0.18, "Decision\ngate", "#F4F4F4")
    rounded_box(ax, (0.84, 0.63), 0.15, 0.18, "Supported\n" + r"$\{\mathrm{safe}\}$" + " or\n" + r"$\{\mathrm{vuln}\}$", "#DDF2EA", GREEN, fontsize=7.5)
    rounded_box(ax, (0.84, 0.25), 0.15, 0.18, "Unsupported\n" + r"$\{\mathrm{safe},\mathrm{vuln}\}$" + "\nreview", LIGHT_GRAY, GRAY, fontsize=7.5)

    arrow(ax, (0.19, 0.72), (0.25, 0.57))
    arrow(ax, (0.19, 0.34), (0.25, 0.51))
    arrow(ax, (0.43, 0.57), (0.48, 0.70))
    arrow(ax, (0.43, 0.51), (0.48, 0.34))
    arrow(ax, (0.66, 0.72), (0.70, 0.58))
    arrow(ax, (0.66, 0.34), (0.70, 0.48))
    arrow(ax, (0.80, 0.57), (0.84, 0.72))
    arrow(ax, (0.80, 0.48), (0.84, 0.34))
    ax.text(0.82, 0.65, "pass", ha="center", va="bottom", fontsize=7.2, color=GREEN)
    ax.text(0.82, 0.38, "fail", ha="center", va="top", fontsize=7.2, color=GRAY)
    ax.text(0.5, 0.93, "VulTriage: support is checked before automation", ha="center", weight="bold", fontsize=10.5)
    ax.text(
        0.5,
        0.08,
        "Estimated target relevance is an empirical input; unsupported operating points fail closed to review.",
        ha="center",
        fontsize=7.6,
        color="#444444",
    )
    export(fig, output / "fig1_workflow")


def figure_support_region(support: pd.DataFrame, output: Path, data_dir: Path) -> None:
    support.to_csv(data_dir / "fig2_support_region.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"wspace": 0.40})

    count_data = (
        support.groupby("alpha_vulnerable", as_index=False)["supported_projects"]
        .max()
        .sort_values("alpha_vulnerable")
    )
    x = np.arange(len(count_data))
    bars = axes[0].bar(x, count_data["supported_projects"], width=0.62, color=[GRAY, ORANGE, BLUE])
    axes[0].set_xticks(x, [f"{100*a:.0f}%" for a in count_data["alpha_vulnerable"]])
    axes[0].set_xlabel("Vulnerable-class risk budget ($\\alpha_v$)")
    axes[0].set_ylabel("Support-qualified projects (of 12)")
    axes[0].set_ylim(0, 12.7)
    axes[0].set_yticks(np.arange(0, 13, 2))
    axes[0].set_title("a  Support expands with budget", loc="left", weight="bold", fontsize=8.8)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, count_data["supported_projects"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{int(value)}/12", ha="center", weight="bold")

    for alpha_v, color, marker, label in [
        (0.05, ORANGE, "s", "$\\alpha_v=5\\%$"),
        (0.10, BLUE, "o", "$\\alpha_v=10\\%$"),
    ]:
        sub = support[support["alpha_vulnerable"] == alpha_v].sort_values("alpha_safe")
        axes[1].plot(
            100 * sub["alpha_safe"],
            100 * sub["median_singleton_coverage_supported_projects"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )
    axes[1].annotate(
        "57.83%",
        xy=(20, 57.83190891041571),
        xytext=(16.2, 63.5),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.8},
        color=BLUE,
        weight="bold",
    )
    axes[1].text(12.5, 10.8, "$\\alpha_v=1\\%$: no project\npassed the support audit", ha="center", color=GRAY, fontsize=7.5)
    axes[1].set_xlabel("Safe-class risk budget ($\\alpha_s$)")
    axes[1].set_ylabel("Median singleton coverage\namong supported projects (%)")
    axes[1].set_xticks([5, 10, 20], ["5%", "10%", "20%"])
    axes[1].set_ylim(0, 70)
    axes[1].set_title("b  Supported targets retain automation", loc="left", weight="bold", fontsize=8.8)
    axes[1].grid(color="#DDDDDD", linewidth=0.6)
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    export(fig, output / "fig2_support_region")


def parse_support(value: str) -> bool:
    return bool(json.loads(value)["supported"])


def figure_project_automation(metrics: pd.DataFrame, output: Path, data_dir: Path) -> None:
    selected = metrics[
        (metrics["track"] == "project_disjoint")
        & (metrics["method"] == "vultriage_clip_20")
        & np.isclose(metrics["alpha_vulnerable"], 0.10)
        & np.isclose(metrics["alpha_safe"], 0.20)
    ].copy()
    selected["supported"] = selected["support"].map(parse_support)
    summary = (
        selected.groupby("target_group", as_index=False)
        .agg(
            supported=("supported", "all"),
            overall_mean=("singleton_coverage", "mean"),
            overall_min=("singleton_coverage", "min"),
            overall_max=("singleton_coverage", "max"),
            vulnerable_singleton=("vulnerable_singleton_rate", "mean"),
            safe_singleton=("safe_singleton_rate", "mean"),
        )
    )
    summary = summary.sort_values(["supported", "overall_mean"], ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(data_dir / "fig3_project_automation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    y = np.arange(len(summary))
    for idx, row in summary.iterrows():
        if row["supported"]:
            ax.hlines(idx, 100 * row["overall_min"], 100 * row["overall_max"], color="#B9B9B9", linewidth=1.4, zorder=1)
            ax.plot(100 * row["vulnerable_singleton"], idx, "o", color=ORANGE, markersize=5.2, zorder=3)
            ax.plot(100 * row["safe_singleton"], idx, "s", color=BLUE, markersize=4.8, zorder=3)
            ax.plot(100 * row["overall_mean"], idx, "|", color="#222222", markersize=9, markeredgewidth=1.4, zorder=4)
        else:
            ax.axhspan(idx - 0.34, idx + 0.34, color=LIGHT_GRAY, zorder=0)
            ax.text(2.0, idx, "review only", va="center", ha="left", color=GRAY, style="italic", fontsize=7.5)

    median_supported = float(summary.loc[summary["supported"], "overall_mean"].median() * 100)
    ax.axvline(median_supported, color=GREEN, linewidth=1.1, linestyle="--", label=f"Median overall coverage: {median_supported:.2f}%")
    ax.set_yticks(y, [name.replace("imagemagick", "ImageMagick").replace("tensorflow", "TensorFlow").replace("radare2", "radare2").replace("ffmpeg", "FFmpeg").replace("openssl", "OpenSSL").replace("tcpdump", "tcpdump").replace("chrome", "Chrome").replace("linux", "Linux").replace("qemu", "QEMU").replace("php", "PHP").replace("vim", "Vim").replace("gpac", "GPAC") for name in summary["target_group"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Automatic singleton rate / overall coverage (%)")
    ax.set_title("Support-qualified automation spans 10 of 12 unseen projects", loc="left", weight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        mpl.lines.Line2D([], [], color=ORANGE, marker="o", linestyle="None", label="Vulnerable-class singleton rate"),
        mpl.lines.Line2D([], [], color=BLUE, marker="s", linestyle="None", label="Safe-class singleton rate"),
        mpl.lines.Line2D([], [], color="#222222", marker="|", markersize=9, linestyle="None", label="Overall singleton coverage"),
        mpl.lines.Line2D([], [], color=GREEN, linestyle="--", label=f"Median overall coverage: {median_supported:.2f}%"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.53, -0.31))
    ax.invert_yaxis()
    export(fig, output / "fig3_project_automation")


def figure_risk_alignment(metrics: pd.DataFrame, output: Path, data_dir: Path) -> None:
    methods = ["unweighted_mondrian", "estimated_weight_no_support_clip_20"]
    selected = metrics[(metrics["track"] == "project_disjoint") & metrics["method"].isin(methods)].copy()
    aggregated = (
        selected.groupby(
            ["target_group", "alpha_vulnerable", "alpha_safe", "method"],
            as_index=False,
        )[["vulnerable_violation", "safe_violation", "singleton_coverage"]]
        .mean()
    )
    aggregated["maximum_violation"] = aggregated[["vulnerable_violation", "safe_violation"]].max(axis=1)
    wide = aggregated.pivot(
        index=["target_group", "alpha_vulnerable", "alpha_safe"],
        columns="method",
        values=["maximum_violation", "singleton_coverage"],
    ).reset_index()
    wide.columns = ["_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col for col in wide.columns]
    wide = wide.rename(
        columns={
            "maximum_violation_unweighted_mondrian": "unweighted_max_violation",
            "maximum_violation_estimated_weight_no_support_clip_20": "weighted_max_violation",
            "singleton_coverage_unweighted_mondrian": "unweighted_singleton_coverage",
            "singleton_coverage_estimated_weight_no_support_clip_20": "weighted_singleton_coverage",
        }
    )
    wide["violation_delta"] = wide["weighted_max_violation"] - wide["unweighted_max_violation"]
    wide["outcome"] = np.select(
        [wide["violation_delta"] < -1e-15, wide["violation_delta"] > 1e-15],
        ["improved", "worse"],
        default="tied",
    )
    wide.to_csv(data_dir / "fig4_risk_alignment.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0.09, 0.16, 0.55, 0.72])
    styles = {
        0.01: ("o", "1% vulnerable budget"),
        0.05: ("s", "5% vulnerable budget"),
        0.10: ("^", "10% vulnerable budget"),
    }
    for alpha_v, (marker, label) in styles.items():
        sub = wide[np.isclose(wide["alpha_vulnerable"], alpha_v)]
        colors = sub["outcome"].map({"improved": BLUE, "tied": GRAY, "worse": ORANGE})
        ax.scatter(
            100 * sub["unweighted_max_violation"],
            100 * sub["weighted_max_violation"],
            c=colors,
            marker=marker,
            s=28,
            alpha=0.82,
            linewidths=0.35,
            edgecolors="white",
            label=label,
        )
    bound = max(float(wide["unweighted_max_violation"].max()), float(wide["weighted_max_violation"].max())) * 100 + 1
    ax.plot([0, bound], [0, bound], color="#444444", linewidth=0.9, linestyle="--")
    ax.fill_between([0, bound], [0, bound], [0, 0], color=LIGHT_BLUE, alpha=0.45, zorder=0)
    ax.text(bound * 0.05, bound * 0.78, "lower violation\nafter weighting", color=BLUE, fontsize=7.5)
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Unweighted violation (percentage points)")
    ax.set_ylabel("Estimated-weight violation (percentage points)")
    fig.suptitle(
        "Estimated weighting improves risk alignment in 74/108 settings",
        x=0.09,
        y=0.97,
        ha="left",
        weight="bold",
        fontsize=10.2,
    )
    ax.grid(color="#E1E1E1", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    budget_legend = ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.02),
        title=r"Marker: $\alpha_v$",
        ncol=1,
    )
    ax.add_artist(budget_legend)
    outcome_handles = [
        mpl.lines.Line2D([], [], color=BLUE, marker="o", linestyle="None", label="Improved"),
        mpl.lines.Line2D([], [], color=GRAY, marker="o", linestyle="None", label="Tied"),
        mpl.lines.Line2D([], [], color=ORANGE, marker="o", linestyle="None", label="Worse"),
    ]
    ax.legend(
        handles=outcome_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.05, 0.56),
        title="Color: outcome",
    )
    fig.text(
        0.72,
        0.17,
        "74 improved\n7 tied\n27 worse",
        fontsize=10,
        weight="bold",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "pad": 4.0},
    )
    export(fig, output / "fig4_risk_alignment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--support-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    support = pd.read_csv(args.support_summary)
    figure_workflow(args.output)
    figure_support_region(support, args.output, data_dir)
    figure_project_automation(metrics, args.output, data_dir)
    figure_risk_alignment(metrics, args.output, data_dir)
    manifest = {
        "metrics_sha256": sha256(args.metrics),
        "support_summary_sha256": sha256(args.support_summary),
        "script_sha256": sha256(Path(__file__)),
        "figures": {},
    }
    for path in sorted(args.output.glob("fig*.*")):
        manifest["figures"][path.name] = sha256(path)
    (args.output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
