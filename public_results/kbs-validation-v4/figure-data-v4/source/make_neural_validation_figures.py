"""Render v3 workflow and complete-project figures with Python/matplotlib.

The workflow-only mode reads preparation receipts and never reads analysis,
predictions, decisions or a target-label vault. Quantitative mode accepts only
the hash-bound aggregate release. It does not fit models or calculate new
bootstrap intervals. Optional external QA modules are supplied explicitly by
--audit-scripts; measured geometry is exported in either mode.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/neurocomputing_revision"
WIDTH_MM = 160.0
DETECTORS = ("finetuned_codebert", "frozen_codebert", "hashing_sgd")
DETECTOR_LABELS = {
    "finetuned_codebert": "Fine-tuned CodeBERT",
    "frozen_codebert": "Frozen CodeBERT", "hashing_sgd": "Hashing-SGD",
}
SHORT_LABELS = {"finetuned_codebert": "Fine-tuned", "frozen_codebert": "Frozen", "hashing_sgd": "Hashing"}
ACTUAL_SEEDS = {"finetuned_codebert": [13, 37, 73], "frozen_codebert": [13], "hashing_sgd": [13, 37, 73]}
DATASETS = ("diversevul", "megavul")
DATASET_LABELS = {"diversevul": "D38: DiverseVul", "megavul": "M6: MegaVul"}
PROJECT_COUNTS = {"diversevul": 38, "megavul": 6}
WEIGHTED = "estimated_weight_no_gate"
ALLOCATION_BASELINES = ("msp_matched_weighted", "class_matched_weighted")
BASELINE_LABELS = {"msp_matched_weighted": "Global MSP", "class_matched_weighted": "Allocation tails"}
PALETTE = {"signal": "#236B8E", "neutral": "#70777F", "accent": "#B87827", "ink": "#25323B",
           "train": "#F0F2F4", "selection": "#FCF1DE", "calibration": "#E7F1F6", "target": "#F2EDF7"}
TOLERANCE_PT = 1.5
EXPECTED_SOURCE = {"train": (15620, 3124), "model_validation": (6296, 192), "calibration": (4400, 204)}


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.usetex": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.0, "legend.frameon": False,
        "axes.unicode_minus": True, "savefig.facecolor": "white",
    })


class Auditors:
    """Load QA only from an explicitly supplied directory, without recording it."""

    def __init__(self, directory: Path | None):
        self.directory = directory.resolve() if directory else None
        self.modules: dict[str, Any] = {}

    def module(self, name: str) -> Any:
        require(self.directory is not None, "External QA was not requested")
        if name not in self.modules:
            path = self.directory / (name + ".py")
            require(path.is_file(), "Missing external QA module: " + name)
            module_name = "_neural_figure_external_" + name
            spec = importlib.util.spec_from_file_location(module_name, path)
            require(spec is not None and spec.loader is not None, "Cannot load external QA module: " + name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.modules[name] = module
        return self.modules[name]

    def scrub(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.scrub(item) for item in value]
        if isinstance(value, str) and self.directory:
            return value.replace(str(self.directory), "<external-QA>").replace(self.directory.as_posix(), "<external-QA>")
        return value


def text_artists(fig: Any) -> list[Any]:
    texts = list(fig.texts)
    for axis in fig.axes:
        texts.extend(axis.texts)
        if axis.axison:
            texts.extend([axis.title, axis.xaxis.label, axis.yaxis.label,
                          axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()])
            texts.extend(axis.get_xticklabels())
            texts.extend(axis.get_yticklabels())
        if axis.get_legend() is not None:
            texts.extend(axis.get_legend().get_texts())
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    return [text for text in texts if text.get_visible() and text.get_text().strip()]


def measured_layout(fig: Any, axes: list[Any], ids: list[str]) -> dict[str, Any]:
    fig.canvas.draw()
    width_pt, height_pt = fig.get_size_inches() * 72
    panels = []
    for axis, panel_id in zip(axes, ids, strict=True):
        position = axis.get_position(original=False)
        panel = {"id": panel_id, "bbox_pt": [position.x0 * width_pt, position.y0 * height_pt,
                                               position.x1 * width_pt, position.y1 * height_pt]}
        spec = axis.get_subplotspec()
        if spec is not None:
            panel.update(grid_id="main-grid", row_start=spec.rowspan.start, row_stop=spec.rowspan.stop,
                         col_start=spec.colspan.start, col_stop=spec.colspan.stop)
        for text in axis.texts:
            if text.get_text().strip() == panel_id:
                pixels = text.get_transform().transform(text.get_position())
                point = fig.dpi_scale_trans.inverted().transform(pixels) * 72
                panel.update(panel_label=panel_id, panel_label_anchor_pt=point.tolist())
                break
        panels.append(panel)
    return {"schema_version": 1, "backend": "python-matplotlib",
            "figure": {"width_pt": width_pt, "height_pt": height_pt}, "panels": panels, "exemptions": []}


def own_alignment_gate(layout: dict[str, Any], row_groups: list[list[str]], column_groups: list[list[str]]) -> dict:
    panels = {panel["id"]: panel for panel in layout["panels"]}
    findings = []
    gutters: dict[str, list[float]] = {"row": [], "column": []}
    for direction, groups in (("row", row_groups), ("column", column_groups)):
        for group in groups:
            boxes = np.asarray([panels[name]["bbox_pt"] for name in group], dtype=float)
            dimensions = boxes[:, 2:] - boxes[:, :2]
            edges = (1, 3) if direction == "row" else (0, 2)
            for edge in edges:
                spread = float(np.ptp(boxes[:, edge]))
                if spread > TOLERANCE_PT:
                    findings.append({"panels": group, "check": "common-edge", "deviation_pt": spread})
            for dimension in (0, 1):
                spread = float(np.ptp(dimensions[:, dimension]))
                if spread > TOLERANCE_PT:
                    findings.append({"panels": group, "check": "equal-size", "deviation_pt": spread})
            require(all("panel_label_anchor_pt" in panels[name] for name in group), "Missing measured panel-label anchor")
            anchors = np.asarray([panels[name]["panel_label_anchor_pt"] for name in group])
            spread = float(np.ptp(anchors[:, 1 if direction == "row" else 0]))
            if spread > TOLERANCE_PT:
                findings.append({"panels": group, "check": "label-anchor", "deviation_pt": spread})
            ordered = boxes[np.argsort(boxes[:, 0] if direction == "row" else -boxes[:, 3])]
            for first, second in zip(ordered[:-1], ordered[1:], strict=True):
                gap = float(second[0] - first[2]) if direction == "row" else float(first[1] - second[3])
                gutters[direction].append(gap)
                require(gap > 0, "Rendered plot areas overlap")
    for direction, values in gutters.items():
        if len(values) > 1 and np.ptp(values) > TOLERANCE_PT:
            findings.append({"check": "repeated-" + direction + "-gutters", "deviation_pt": float(np.ptp(values))})
    require(not findings, "Strict rendered alignment gate failed: " + json.dumps(findings))
    return {"verdict": "PASS" if len(panels) > 1 else "NOT APPLICABLE", "strict": True,
            "tolerance_pt": TOLERANCE_PT, "findings": findings, "layout": layout}


def drawing_geometry(fig: Any) -> dict[str, Any]:
    renderer = fig.canvas.get_renderer()
    factor = 72 / fig.dpi
    texts = [{"text": text.get_text(), "font_size_pt": float(text.get_fontsize()),
              "bbox_pt": (np.asarray(text.get_window_extent(renderer).extents) * factor).tolist()}
             for text in text_artists(fig)]
    require(texts and min(row["font_size_pt"] for row in texts) >= 9, "A nominal figure font is below 9 pt")
    return {"schema_version": 1, "backend": "python-matplotlib", "coordinate_origin": "bottom-left",
            "figure_size_pt": (fig.get_size_inches() * 72).tolist(), "texts": texts,
            "plot_areas": [list(np.asarray(axis.get_window_extent(renderer).extents) * factor) for axis in fig.axes]}


def pdf_geometry_and_glyphs(path: Path) -> tuple[dict, dict]:
    import pymupdf

    def coordinates(value: Any) -> Any:
        if isinstance(value, (str, int, float)) or value is None:
            return value
        if hasattr(value, "x") and hasattr(value, "y"):
            return [float(value.x), float(value.y)]
        if all(hasattr(value, name) for name in ("x0", "y0", "x1", "y1")):
            return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
        if all(hasattr(value, name) for name in ("ul", "ur", "ll", "lr")):
            return [coordinates(getattr(value, name)) for name in ("ul", "ur", "ll", "lr")]
        return [coordinates(item) for item in value]

    with pymupdf.open(path) as document:
        require(len(document) == 1, "A figure PDF must contain one page")
        page = document[0]
        texts, sizes = [], []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        texts.append({"text": span["text"], "bbox_pt": list(span["bbox"]), "size_pt": span["size"]})
                        sizes.append(span["size"])
        require(sizes and min(sizes) >= 5, "Final PDF contains a glyph smaller than 5 pt")
        require(abs(page.rect.width / 72 * 25.4 - WIDTH_MM) < .05, "Exported figure width is not 160 mm")
        clipped = [text for text in texts if text["bbox_pt"][0] < -.5 or text["bbox_pt"][1] < -.5
                   or text["bbox_pt"][2] > page.rect.width + .5 or text["bbox_pt"][3] > page.rect.height + .5]
        require(not clipped, "Final PDF text is clipped")
        strokes = []
        for drawing in page.get_drawings():
            strokes.append({"type": drawing["type"], "bbox_pt": list(drawing["rect"]),
                            "width_pt": drawing.get("width"),
                            "items": [{"operator": item[0], "operands": [coordinates(value) for value in item[1:]]}
                                      for item in drawing["items"]]})
        return ({"backend": "final-PDF", "coordinate_origin": "top-left", "page_size_pt": list(page.rect),
                 "texts": texts, "drawings": strokes},
                {"verdict": "PASS", "auditable": True, "minimum_required_pt": 5,
                 "minimum_found_pt": min(sizes), "text_span_count": len(sizes), "clipped_text_count": 0})


def archive_workflow(output: Path) -> None:
    if output.resolve() != (PAPER / "figures").resolve():
        return
    archive = ROOT / "outputs/neural-validation-v3/pre_strengthening_manuscript/figures"
    archive.mkdir(parents=True, exist_ok=True)
    receipt = archive / "workflow_archive_receipt.json"
    if receipt.exists():
        return
    copied = {}
    old_files = list(output.glob("fig_workflow.*"))
    if (output / "figure_manifest_v2.json").exists():
        old_files.append(output / "figure_manifest_v2.json")
    for source in old_files:
        destination = archive / source.name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
            copied[source.name] = sha(destination)
    write_json(receipt, {"files_sha256": copied,
                        "scope": "Previous canonical workflow preserved before v3 replacement; no old quantitative figures replaced."})


def export_figure(fig: Any, name: str, axes: list[Any], ids: list[str], context: dict,
                  contract: dict, rows: list[list[str]] | None = None, columns: list[list[str]] | None = None) -> None:
    output, auditors = context["output"], context["auditors"]
    path = output / name
    if name == "fig_workflow":
        archive_workflow(output)
    fig.canvas.draw()
    layout = measured_layout(fig, axes, ids)
    layout.update(row_groups=rows or [], column_groups=columns or [])
    write_json(path.with_suffix(".alignment-layout.json"), layout)
    alignment = own_alignment_gate(layout, rows or [], columns or [])
    write_json(path.with_suffix(".alignment.json"), alignment)
    write_json(path.with_suffix(".geometry.json"), drawing_geometry(fig))
    external = {"requested": auditors.directory is not None, "collision": "NOT RUN", "source_preflight": "NOT RUN"}
    if auditors.directory:
        alignment_module = auditors.module("audit_panel_alignment")
        alignment = alignment_module.require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=ids, row_groups=rows or [], column_groups=columns or [],
            json_out=path.with_suffix(".alignment.json"), overlay_svg=path.with_suffix(".alignment.svg"),
            strict=True, require_panel_labels=len(axes) > 1, tolerance_pt=1.5, gutter_tolerance_pt=1.5)
        source_module = auditors.module("validate_figure")
        findings = source_module.validate_source(Path(__file__).read_text(encoding="utf-8"), "python")
        source_report = {"summary": source_module.summarize(findings), "findings": [asdict(row) for row in findings]}
        source_report["reviewed_warnings"] = [
            {"check_id": row.check_id,
             "reason": "Explicit 160-mm A4 text-width contract; actual PDF width and at-least-9-pt glyphs are verified after export."}
            for row in findings if row.level == "WARN" and row.check_id == "FINAL-WIDTH"]
        write_json(path.with_suffix(".source-preflight.json"), auditors.scrub(source_report))
        require(source_report["summary"]["counts"]["FAIL"] == 0, "Plotting source preflight failed")
        require(all(row.check_id == "FINAL-WIDTH" for row in findings if row.level == "WARN"),
                "Plotting source has an unreviewed warning")
        external["source_preflight"] = "PASS; explicit 160-mm width warning reviewed"
    # Fixed physical dimensions preserve the glyph scale in the manuscript.
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib v3 aggregate figure renderer"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    pdf_geometry, glyphs = pdf_geometry_and_glyphs(path.with_suffix(".pdf"))
    write_json(path.with_suffix(".pdf-geometry.json"), pdf_geometry)
    write_json(path.with_suffix(".glyph-audit.json"), glyphs)
    if auditors.directory:
        pdf_module = auditors.module("audit_pdf_text")
        operator_audit = pdf_module.audit_pdf(path.with_suffix(".pdf").read_bytes(), minimum_pt=5)
        require(operator_audit["auditable"] and not operator_audit["below_minimum_count"], "Final PDF operator glyph audit failed")
        write_json(path.with_suffix(".glyph-audit.json"), {**glyphs, "operator_audit": operator_audit})
        collision_module = auditors.module("audit_figure_collisions")
        collision = collision_module.audit_pdf(path.with_suffix(".pdf"))
        write_json(path.with_suffix(".collision-audit.json"), auditors.scrub(collision))
        collision_module.write_overlay_pdf(path.with_suffix(".pdf"), path.with_suffix(".collision-audit.pdf"), collision["findings"])
        require(collision_module.exit_code(collision, strict=True) == 0, "Final PDF collision audit failed; repair and re-export")
        external["collision"] = "PASS (strict)"
    else:
        write_json(path.with_suffix(".collision-audit.json"), {
            "verdict": "EXTERNAL AUDIT REQUIRED", "passed": False,
            "geometry": path.with_suffix(".pdf-geometry.json").name,
            "note": "Measured final PDF geometry and glyph/clipping checks are available; an unrun collision audit is not a pass."})
    artifact_suffixes = [".alignment-layout.json", ".alignment.json", ".geometry.json",
                        ".pdf", ".svg", ".png", ".tiff", ".pdf-geometry.json",
                        ".glyph-audit.json", ".collision-audit.json"]
    if auditors.directory:
        artifact_suffixes.extend([".alignment.svg", ".source-preflight.json", ".collision-audit.pdf"])
    if name == "fig_workflow":
        artifact_suffixes.append(".node-geometry.json")
    # Only receipts produced by this render are bound here. Visual reviews are
    # added after inspecting these exact exports; legacy receipts stay archived.
    plt.close(fig)
    context["figures"][name] = {
        "contract": contract, "width_mm": WIDTH_MM, "height_mm": layout["figure"]["height_pt"] / 72 * 25.4,
        "alignment_verdict": alignment.get("verdict"), "pdf_glyphs": glyphs, "external_QA": external,
        "visual_review": "Required at final physical size; automatic gates do not replace review.",
        "files_sha256": {(name + suffix): sha(path.with_suffix(suffix)) for suffix in artifact_suffixes},
        "source_csv_sha256": {member.name: sha(member) for member in (output / "data").glob(name + "*.csv")},
    }


def frozen_workflow_facts() -> tuple[dict, dict]:
    source_path = ROOT / "outputs/neural-validation-v3/inputs/preparation_manifest.json"
    second_path = source_path.parent / "megavul/preparation_manifest.json"
    source, second = read_json(source_path), read_json(second_path)
    counts = {}
    for role, expected in EXPECTED_SOURCE.items():
        values = source["source_counts"][role]
        counts[role] = (int(values["0"]) + int(values["1"]), int(values["1"]))
        require(counts[role] == expected, "Workflow source role differs from the frozen v3.1 counts: " + role)
    require(len(source["selected_projects"]) == 38 and len(second["selected_projects"]) == 6,
            "Workflow dataset roster differs from D38/M6")
    require(source["candidate_projects"] == second["candidate_projects"] and len(source["candidate_projects"]) == 62,
            "Workflow candidate roster differs from the frozen D62")
    require(second["source_protocol"]["preparation_manifest_sha256"] == sha(source_path), "Workflow source binding changed")
    return ({"source_counts": counts, "datasets": PROJECT_COUNTS, "actual_seeds": ACTUAL_SEEDS,
             "eligibility": {"minimum_total": 200, "minimum_vulnerable": 20, "minimum_safe": 20}},
            {"source_preparation_manifest": sha(source_path), "megavul_preparation_manifest": sha(second_path)})


def workflow(context: dict, facts: dict) -> None:
    fig, axis = plt.subplots(figsize=(WIDTH_MM / 25.4, 125 / 25.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set(xlim=(0, 160), ylim=(0, 125)); axis.axis("off")
    node_rows = []
    node_artists = {}

    def node(identifier, x, y, width, height, title, body, color):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.25,rounding_size=1.0",
                              facecolor=color, edgecolor="#8A9AA5", linewidth=.7)
        axis.add_patch(patch)
        title_artist = axis.text(x + width / 2, y + height - 4.1, title, ha="center", va="center",
                                 fontsize=10, fontweight="bold", color=PALETTE["ink"])
        artists = [title_artist]
        lines = body.split("\n")
        for index, line in enumerate(lines):
            artists.append(axis.text(x + width / 2, y + height - 9.0 - index * 4.0, line,
                                     ha="center", va="center", fontsize=9, color=PALETTE["ink"]))
        node_rows.append({"node": identifier, "x_mm": x, "y_mm": y, "width_mm": width,
                          "height_mm": height, "title": title, "body": body})
        node_artists[identifier] = (patch, artists)
        return artists

    def arrow(points):
        if len(points) > 2:
            axis.plot(*np.asarray(points[:-1]).T, color="#667A87", linewidth=.8, zorder=1)
        axis.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
                                      linewidth=.8, color="#667A87", shrinkA=0, shrinkB=0, zorder=2))

    for x, role, title, color in ((4, "train", "Source train", PALETTE["train"]),
                                  (57, "model_validation", "Model validation", PALETTE["selection"]),
                                  (110, "calibration", "Source calibration", PALETTE["calibration"])):
        total, positive = facts["source_counts"][role]
        node("source_calibration" if role == "calibration" else role, x, 104, 46, 19, title,
             f"{total:,} functions\n{positive:,} vulnerable", color)
        arrow([(x + 23, 103.5), (x + 23, 98.5)])
    node("detectors", 4, 78, 46, 20, "Source detector fits",
         "3 fine-tuned CodeBERT fits\n1 frozen CodeBERT head\n3 Hashing-SGD fits", PALETTE["train"])
    node("selection", 57, 78, 46, 20, "Model selection", "Highest source-valid AP\nSelect checkpoint / head", PALETTE["selection"])
    node("policy_calibration", 110, 78, 46, 20, "Policy calibration",
         "Class budgets + isotonic\nSeparate calibration role", PALETTE["calibration"])
    arrow([(56.5, 90), (50.5, 90)])
    node("eligibility", 4, 53, 46, 19, "Target eligibility",
         "DiverseVul 38; MegaVul 6\n200 total; 20 per class\nPublished-label counts", PALETTE["target"])
    node("unlabeled", 57, 53, 46, 19, "Label-free inputs", "Target code + metadata\nLabels in separate vault", PALETTE["target"])
    node("prediction", 110, 53, 46, 19, "Probabilities / weights", "Detector probabilities\nLabel-free domain fit", PALETTE["calibration"])
    arrow([(50.5, 62.5), (56.5, 62.5)]); arrow([(103.5, 62.5), (109.5, 62.5)])
    arrow([(27, 77.5), (27, 75), (133, 75), (133, 72.5)])
    arrow([(156.5, 88), (158.5, 88), (158.5, 39), (156.5, 39)])
    arrow([(133, 52.5), (133, 49.5)])
    node("policies", 4, 29, 152, 20, "Source-calibrated policies and comparison controls",
         "Weighted / unweighted sets | MSP / tails | Source isotonic | Forced argmax / all review\n"
         "Domain-fit failure: complete review\nLegacy gate reported only for original audit", PALETTE["calibration"])
    arrow([(40, 28.5), (40, 24.5)])
    node("seal", 4, 9, 73, 15, "Seal both datasets first",
         "Complete predictions + decisions\nBefore joining evaluation labels", PALETTE["target"])
    node("evaluate", 83, 9, 73, 15, "Evaluate after both seals", "Join held-out target labels\nReport datasets separately", PALETTE["train"])
    arrow([(77.5, 16.5), (82.5, 16.5)])
    axis.text(80, 2.6, "Published-label eligibility precedes label-free policy replay and sealed evaluation.",
              ha="center", va="center", fontsize=9, color=PALETTE["ink"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    factor = 72 / fig.dpi
    node_geometry = {}
    for identifier, (patch, artists) in node_artists.items():
        box = np.asarray(patch.get_window_extent(renderer).extents) * factor
        for artist in artists:
            text_box = np.asarray(artist.get_window_extent(renderer).extents) * factor
            require(text_box[0] >= box[0] and text_box[1] >= box[1]
                    and text_box[2] <= box[2] and text_box[3] <= box[3], "Workflow node text exceeds its box: " + identifier)
        node_geometry[identifier] = {"bbox_pt": box.tolist()}
    for row in (("train", "model_validation", "source_calibration"), ("detectors", "selection", "policy_calibration"),
                ("eligibility", "unlabeled", "prediction"), ("seal", "evaluate")):
        boxes = np.asarray([node_geometry[name]["bbox_pt"] for name in row])
        require(np.ptp(boxes[:, 1]) < TOLERANCE_PT and np.ptp(boxes[:, 3]) < TOLERANCE_PT,
                "Workflow node-row alignment failed")
        require(np.ptp(boxes[:, 2] - boxes[:, 0]) < TOLERANCE_PT, "Workflow equal-row widths failed")
    for column in (("train", "detectors", "eligibility"), ("model_validation", "selection", "unlabeled"),
                   ("source_calibration", "policy_calibration", "prediction")):
        boxes = np.asarray([node_geometry[name]["bbox_pt"] for name in column])
        require(np.ptp(boxes[:, 0]) < TOLERANCE_PT and np.ptp(boxes[:, 2]) < TOLERANCE_PT,
                "Workflow node-column alignment failed")
    write_json(context["output"] / "fig_workflow.node-geometry.json", {
        "backend": "python-matplotlib", "coordinate_origin": "bottom-left", "nodes": node_geometry,
        "strict_row_column_alignment": "PASS", "tolerance_pt": TOLERANCE_PT,
        "text_inside_nodes": True})
    pd.DataFrame(node_rows).to_csv(context["output"] / "data/fig_workflow_nodes.csv", index=False)
    pd.DataFrame([{"role": role, "functions": counts[0], "vulnerable_functions": counts[1]}
                  for role, counts in facts["source_counts"].items()]).to_csv(
        context["output"] / "data/fig_workflow_source_counts.csv", index=False)
    export_figure(fig, "fig_workflow", [axis], ["workflow"], context, {
        "archetype": "schematic-led composite", "claim": "Frozen model-selection, source-calibration and target-label boundaries are separate.",
        "uncertainty": "Not applicable: factual role/count schematic, no outcome estimates.",
        "facts": facts, "target_effects_accessed": False,
    })


class AggregateEvidence:
    """Read only the explicitly released aggregate members needed by figures."""

    def __init__(self, analysis: Path):
        self.root = analysis.resolve()
        self.manifest_path = self.root / "public_analysis_manifest.json"
        require(self.manifest_path.is_file(), "Quantitative figures require the complete aggregate export; use --workflow-only until it exists")
        self.manifest = read_json(self.manifest_path)
        require(self.manifest["schema_version"] == "trace-v3-aggregate-release-1", "Unrecognized aggregate release")
        require(self.manifest["no_raw_datasets_predictions_weights_or_vaults"] is True, "Aggregate-only release boundary is missing")
        self.input_hashes = {"public_analysis_manifest.json": sha(self.manifest_path)}
        self.receipts, self.rosters, self.failed, self.availability = {}, {}, {}, {}
        for dataset in DATASETS:
            receipt = self.json(dataset + "/dataset_receipt.json")
            require(receipt["projects"] == PROJECT_COUNTS[dataset] and receipt["actual_seeds"] == ACTUAL_SEEDS,
                    "Dataset project/actual-fit count does not match D38/M6: " + dataset)
            require(receipt["verified_before_target_label_access"] is True and receipt["all_planned_fits_present"] is True,
                    "Complete pre-label seals are not established: " + dataset)
            self.receipts[dataset] = receipt
            self.rosters[dataset] = tuple(sorted(receipt["selected_projects"]))
            require(len(self.rosters[dataset]) == PROJECT_COUNTS[dataset], "Incomplete selected project roster")
            availability = self.csv(dataset + "/domain_fit_availability.csv")
            require(not availability.duplicated(["detector", "seed", "project"]).any(), "Duplicate availability cell")
            availability["domain_available"] = availability["domain_available"].map(self.boolean)
            diagnostics = self.json(dataset + "/domain_cell_diagnostics.json")
            diagnostic_status = {(row["detector"], int(row["seed"]), row["project"]): row["domain_status"] == "ok"
                                 for row in diagnostics}
            require(len(diagnostic_status) == len(diagnostics) == len(availability), "Domain diagnostic grid is incomplete")
            for row in availability.itertuples():
                require(row.dataset == dataset and diagnostic_status.get((row.detector, int(row.seed), row.project)) == row.domain_available,
                        "Domain diagnostics and availability disagree")
            self.availability[dataset] = availability
            for detector in DETECTORS:
                values = availability[availability["detector"] == detector]
                require(len(values) == len(self.rosters[dataset]) * len(ACTUAL_SEEDS[detector]), "Availability omits a planned fit/project")
                require(set(values["project"]) == set(self.rosters[dataset]), "Availability project roster changed")
                require(all(tuple(sorted(group["seed"])) == tuple(ACTUAL_SEEDS[detector])
                            for _, group in values.groupby("project")), "Availability seed roster changed")
                supported = values.groupby("project")["domain_available"].all()
                self.failed[dataset, detector] = set(supported[~supported].index)
        self.actions = self.csv("primary_budget_project_actions.csv")
        self.paired = self.csv("primary_budget_paired_summaries.csv")
        for frame in (self.actions, self.paired):
            require(np.isclose(frame["alpha_v"], .10).all() and np.isclose(frame["alpha_s"], .20).all()
                    and np.isclose(frame["clip"], 20).all(), "Figure source is not the complete primary-budget/clip export")
            require(set(frame["dataset"]) == set(DATASETS) and set(frame["detector"]) == set(DETECTORS), "Figure source omits a dataset or detector")
        require(not self.actions.duplicated(["dataset", "detector", "project", "method"]).any(), "Duplicate project-action aggregate")

    @staticmethod
    def boolean(value: Any) -> bool:
        if value is True or str(value).casefold() == "true":
            return True
        if value is False or str(value).casefold() == "false":
            return False
        raise ValueError("Nonboolean domain availability")

    def member(self, name: str) -> Path:
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root) and path.suffix in {".csv", ".json"}, "Only declared aggregate CSV/JSON members may be read")
        expected = self.manifest["files_sha256"].get(name)
        require(expected is not None and path.is_file() and sha(path) == expected, "Unbound or changed aggregate member: " + name)
        self.input_hashes[name] = expected
        return path

    def json(self, name: str) -> Any:
        return read_json(self.member(name))

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.member(name))

    def action(self, dataset: str, detector: str, method: str) -> pd.DataFrame:
        frame = self.actions[(self.actions["dataset"] == dataset) & (self.actions["detector"] == detector)
                             & (self.actions["method"] == method)].set_index("project").sort_index()
        require(tuple(frame.index) == self.rosters[dataset], "A plotted policy omits or adds an eligible project")
        require(frame["actual_seed_count"].eq(len(ACTUAL_SEEDS[detector])).all(), "Incomplete actual-fit averaging")
        return frame

    def summary(self, dataset: str, detector: str, baseline: str, outcome: str) -> pd.Series:
        frame = self.paired[(self.paired["dataset"] == dataset) & (self.paired["detector"] == detector)
                            & (self.paired["baseline"] == baseline) & (self.paired["candidate"] == WEIGHTED)
                            & (self.paired["outcome"] == outcome) & (self.paired["cohort"] == "all_projects_deployment")]
        require(len(frame) == 1, "Expected one complete all-project paired summary")
        row = frame.iloc[0]
        require(row["projects"] == row["eligible_projects"] == row["all_projects"] == PROJECT_COUNTS[dataset], "Paired summary uses an incomplete project cohort")
        require(row["bootstrap_draws"] == 10000 and row["bootstrap_seed"] == 20261010, "Paired CI does not match the fixed 10,000 project bootstrap")
        require(row["review_fallback_projects"] == len(self.failed[dataset, detector]), "Paired fallback count disagrees with sealed availability")
        if np.isfinite([row["median_difference"], row["ci_lower"], row["ci_upper"]]).all():
            require(row["defined_projects"] == row["projects"] and row["ci_lower"] <= row["median_difference"] <= row["ci_upper"],
                    "Invalid complete-project paired estimate/interval")
        else:
            require("undefined" in str(row["status"]) or "no_eligible" in str(row["status"]), "Undefined paired geometry lacks a declared status")
        return row


def panel_letter(axis: Any, letter: str) -> None:
    axis.set_gid(letter)
    axis.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-13, 9),
                  textcoords="offset points", fontsize=11, fontweight="bold", annotation_clip=False)


def limits_from_intervals(rows: list[dict], scale: float = 1.0) -> tuple[float, float]:
    values = [0.0]
    for row in rows:
        values.extend(float(row[key]) * scale for key in ("ci_lower", "ci_upper") if np.isfinite(row[key]))
    lower, upper = min(values), max(values)
    span = max(upper - lower, .01 if scale == 1 else 1.0)
    return lower - .12 * span, upper + .12 * span


def interval_mark(axis: Any, row: dict | pd.Series, y: float, color: str, marker: str, scale: float = 1.0) -> None:
    center, lower, upper = (float(row[key]) * scale for key in ("median_difference", "ci_lower", "ci_upper"))
    if not np.isfinite([center, lower, upper]).all():
        axis.text(.02, y, "Undefined", transform=axis.get_yaxis_transform(), fontsize=9, va="center", color=PALETTE["neutral"])
        return
    axis.errorbar(center, y, xerr=np.asarray([[center - lower], [upper - center]]), fmt=marker,
                  color=color, ecolor=color, markersize=4.4, capsize=2, elinewidth=1.0, linewidth=1.0, zorder=3)


def allocation(evidence: AggregateEvidence, context: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, 151 / 25.4))
    fig.subplots_adjust(left=.16, right=.985, bottom=.22, top=.82, wspace=.40, hspace=.70)
    scatter_rows, forest_rows = [], []
    weighted = evidence.action("diversevul", "finetuned_codebert", WEIGHTED)
    upper = 1.0
    paired_data = []
    for baseline in ALLOCATION_BASELINES:
        base = evidence.action("diversevul", "finetuned_codebert", baseline)
        x, y = base["auto_fn_rate"].to_numpy(float) * 100, weighted["auto_fn_rate"].to_numpy(float) * 100
        require(np.isfinite(x).all() and np.isfinite(y).all(), "Scatter endpoint is undefined; no project may be silently dropped")
        require(((x >= 0) & (x <= 100)).all() and ((y >= 0) & (y <= 100)).all(), "Scatter miss-rate bounds are invalid")
        upper = max(upper, float(max(x.max(), y.max())) * 1.10)
        paired_data.append((baseline, base, x, y))
    failed = evidence.failed["diversevul", "finetuned_codebert"]
    for index, (baseline, base, x, y) in enumerate(paired_data):
        axis = axes[0, index]
        fallback = np.asarray([project in failed for project in weighted.index])
        axis.plot([0, upper], [0, upper], "--", color="#929CA4", linewidth=.8, zorder=1)
        axis.scatter(x[~fallback], y[~fallback], color=PALETTE["signal"], s=22, linewidth=.5,
                     edgecolor="white", zorder=3)
        axis.scatter(x[fallback], y[fallback], marker="x", color=PALETTE["accent"], s=33, linewidth=1.2, zorder=4)
        axis.set(xlim=(-.035 * upper, upper), ylim=(-.035 * upper, upper),
                 xlabel=BASELINE_LABELS[baseline] + " Fv (%)", ylabel="Weighted Fv (%)")
        axis.set_title("Matched workload" if index == 0 else "Matched label allocation", pad=9)
        axis.xaxis.set_major_locator(MaxNLocator(4)); axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=None))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=None))
        panel_letter(axis, chr(97 + index))
        for project, xv, yv in zip(weighted.index, x, y, strict=True):
            scatter_rows.append({"panel": chr(97 + index), "dataset": "diversevul", "detector": "finetuned_codebert",
                                 "project": project, "baseline": baseline, "baseline_Fv_percent": xv,
                                 "weighted_Fv_percent": yv, "review_fallback_present": project in failed,
                                 "actual_fit_count": 3, "alpha_v": .10, "alpha_s": .20, "clip": 20})
    fig.text(.56, .970, "Fine-tuned CodeBERT on all 38 D38 projects", ha="center", fontsize=10)
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=PALETTE["signal"], markeredgecolor="white", markersize=5),
               Line2D([], [], marker="x", linestyle="none", color=PALETTE["accent"], markersize=5)]
    fig.legend(handles, [f"All domain fits available ({38 - len(failed)}/38 projects)",
                         f"Contains domain-failure full review ({len(failed)}/38 projects)"],
               loc="upper center", bbox_to_anchor=(.57, .954), ncol=1, fontsize=9, handletextpad=.6)
    for dataset in DATASETS:
        for detector in DETECTORS:
            for baseline in ALLOCATION_BASELINES:
                row = evidence.summary(dataset, detector, baseline, "auto_fn_rate").to_dict()
                row.update(display_difference_percentage_points=row["median_difference"] * 100,
                           display_ci_lower_percentage_points=row["ci_lower"] * 100,
                           display_ci_upper_percentage_points=row["ci_upper"] * 100)
                forest_rows.append(row)
    xlim = limits_from_intervals(forest_rows, scale=100)
    for index, dataset in enumerate(DATASETS):
        axis = axes[1, index]
        axis.axvline(0, color="#929CA4", linestyle="--", linewidth=.8, zorder=1)
        for detector_index, detector in enumerate(DETECTORS):
            for baseline_index, baseline in enumerate(ALLOCATION_BASELINES):
                row = next(row for row in forest_rows if row["dataset"] == dataset and row["detector"] == detector and row["baseline"] == baseline)
                interval_mark(axis, row, 2 - detector_index + (.16 if baseline_index == 0 else -.16),
                              PALETTE["neutral"] if baseline_index == 0 else PALETTE["accent"],
                              "o" if baseline_index == 0 else "s", scale=100)
        axis.set_yticks([2, 1, 0], labels=[SHORT_LABELS[d] +
                       f"\nALL {PROJECT_COUNTS[dataset]}; S {PROJECT_COUNTS[dataset] - len(evidence.failed[dataset, d])}"
                       for d in DETECTORS])
        axis.set(xlim=xlim, ylim=(-.6, 2.6), xlabel="Weighted minus comparator Fv (pp)")
        axis.set_title(DATASET_LABELS[dataset] + f" (n = {PROJECT_COUNTS[dataset]})", pad=9)
        axis.xaxis.set_major_locator(MaxNLocator(4)); axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        panel_letter(axis, chr(99 + index))
    fig.legend([Line2D([], [], marker="o", color=PALETTE["neutral"], linewidth=1, markersize=4),
                Line2D([], [], marker="s", color=PALETTE["accent"], linewidth=1, markersize=4)],
               ["Versus global MSP", "Versus allocation tails"], loc="lower center", bbox_to_anchor=(.56, .098), ncol=2)
    for y, text in ((.088, "Paired medians; 95% project bootstrap CI (10,000 draws); actual fits averaged within project."),
                    (.062, "ALL = all projects; S = every actual domain fit available. Review fallback retained."),
                    (.036, "Zero automatic errors from complete review are not predictive success."),
                    (.010, "Fv = automatic misses / vulnerable functions; pp = percentage points. Datasets remain separate.")):
        fig.text(.5, y, text, ha="center", fontsize=9, color=PALETTE["ink"])
    pd.DataFrame(scatter_rows).to_csv(context["output"] / "data/fig_neural_allocation_scatter.csv", index=False)
    pd.DataFrame(forest_rows).to_csv(context["output"] / "data/fig_neural_allocation_forest.csv", index=False)
    export_figure(fig, "fig_neural_allocation", list(axes.flat), list("abcd"), context, {
        "archetype": "quantitative grid", "claim": "All-project paired evidence tests workload and allocation explanations of automatic misses.",
        "panels": {"a": "Fine-tuned D38 weighted vs MSP", "b": "Same projects vs allocation tails",
                   "c": "Three detectors, all D38 projects", "d": "Three detectors, all M6 projects"},
        "n": PROJECT_COUNTS, "interval": "Median paired project difference; 10,000 project-bootstrap percentile 95% CI",
        "display_units": "Scatter percent; forest percentage points", "review_fallback_project_counts": {
            dataset: {detector: len(evidence.failed[dataset, detector]) for detector in DETECTORS} for dataset in DATASETS},
        "supported_project_counts": {dataset: {detector: PROJECT_COUNTS[dataset] - len(evidence.failed[dataset, detector])
                                                for detector in DETECTORS} for dataset in DATASETS},
        "endpoint_scope": "Fv action-error comparisons do not assert equal empty/doubleton set composition or set miscoverage.",
    }, [["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]])


def costs(evidence: AggregateEvidence, context: dict) -> None:
    rows, project_rows = [], []
    for dataset in DATASETS:
        for detector in DETECTORS:
            weighted = evidence.action(dataset, detector, WEIGHTED)
            for miss_cost in (10, 50):
                baselines = (f"source_iso_cost{miss_cost}_matched_weighted", f"source_iso_cost{miss_cost}_free_volume", "all_review")
                for comparator_index, baseline in enumerate(baselines):
                    row = evidence.summary(dataset, detector, baseline, f"cost_{miss_cost}").to_dict()
                    row.update(miss_cost=miss_cost, comparator_index=comparator_index)
                    rows.append(row)
                    base = evidence.action(dataset, detector, baseline)
                    for project in evidence.rosters[dataset]:
                        project_rows.append({"dataset": dataset, "detector": detector, "project": project,
                                             "miss_cost": miss_cost, "baseline": baseline,
                                             "weighted_cost": weighted.loc[project, f"cost_{miss_cost}"],
                                             "baseline_cost": base.loc[project, f"cost_{miss_cost}"],
                                             "paired_cost_difference": weighted.loc[project, f"cost_{miss_cost}"] - base.loc[project, f"cost_{miss_cost}"],
                                             "review_fallback_present": project in evidence.failed[dataset, detector]})
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM / 25.4, 175 / 25.4), sharex=True)
    fig.subplots_adjust(left=.24, right=.98, bottom=.17, top=.84, hspace=.44)
    styles = [("#70777F", "o", "Source-isotonic: matched volume"),
              ("#B87827", "s", "Source-isotonic: free volume"),
              ("#25323B", "D", "All review: complete-review benchmark")]
    xlim = limits_from_intervals(rows)
    for index, (dataset, axis) in enumerate(zip(DATASETS, axes, strict=True)):
        axis.axvline(0, color="#929CA4", linestyle="--", linewidth=.8, zorder=1)
        labels = []
        for detector_index, detector in enumerate(DETECTORS):
            for cost_index, miss_cost in enumerate((10, 50)):
                center_y = 5 - detector_index * 2 - cost_index
                labels.append(SHORT_LABELS[detector] + f", {miss_cost}\nALL {PROJECT_COUNTS[dataset]}; "
                              + f"S {PROJECT_COUNTS[dataset] - len(evidence.failed[dataset, detector])}")
                for comparator_index, (color, marker, _) in enumerate(styles):
                    row = next(row for row in rows if row["dataset"] == dataset and row["detector"] == detector
                               and row["miss_cost"] == miss_cost and row["comparator_index"] == comparator_index)
                    interval_mark(axis, row, center_y + (.25 - comparator_index * .25), color, marker)
        axis.set_yticks([5, 4, 3, 2, 1, 0], labels=labels)
        axis.set(xlim=xlim, ylim=(-.65, 5.65), ylabel="Detector, miss cost")
        axis.set_title(DATASET_LABELS[dataset] + f" (n = {PROJECT_COUNTS[dataset]}; all projects)", pad=11)
        axis.tick_params(labelbottom=True)
        axis.xaxis.set_major_locator(MaxNLocator(5)); axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))
        panel_letter(axis, chr(97 + index))
    axes[1].set_xlabel("Weighted minus comparator cost (per function)")
    fig.legend([Line2D([], [], color=color, marker=marker, linewidth=1, markersize=4) for color, marker, _ in styles],
               [label for _, _, label in styles], loc="upper center", bbox_to_anchor=(.60, .985), ncol=1, handletextpad=.7)
    for y, text in ((.086, "Median paired cost difference; 95% project bootstrap CI (10,000 draws)."),
                    (.061, "ALL = all projects; S = every actual domain fit available; no supported-only filtering."),
                    (.036, "Miss costs 10/50; false-alert cost 1; review cost 0.1; correct review assumed."),
                    (.011, "Negative differences favor weighted. Domain-failure review fallback is retained.")):
        fig.text(.5, y, text, ha="center", fontsize=9, color=PALETTE["ink"])
    pd.DataFrame(rows).to_csv(context["output"] / "data/fig_neural_costs_forest.csv", index=False)
    pd.DataFrame(project_rows).to_csv(context["output"] / "data/fig_neural_costs_project_pairs.csv", index=False)
    export_figure(fig, "fig_neural_costs", list(axes), list("ab"), context, {
        "archetype": "quantitative grid", "claim": "All-project paired costs compare volume constraints and complete review separately within each dataset.",
        "panels": {"a": "D38 cost boundary", "b": "M6 cost boundary"}, "n": PROJECT_COUNTS,
        "miss_costs": [10, 50], "false_alert_cost": 1, "review_cost": .1, "review_accuracy": "Correct review assumed",
        "interval": "Median paired project difference; 10,000 project-bootstrap percentile 95% CI",
        "all_review_benchmark_included": True,
        "supported_project_counts": {dataset: {detector: PROJECT_COUNTS[dataset] - len(evidence.failed[dataset, detector])
                                                for detector in DETECTORS} for dataset in DATASETS},
    }, [], [["a", "b"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=PAPER / "analysis/neural_validation_v3")
    parser.add_argument("--output", type=Path, default=PAPER / "figures")
    parser.add_argument("--workflow-only", action="store_true", help="Render frozen workflow facts without opening aggregate target effects")
    parser.add_argument("--audit-scripts", type=Path, help="Optional explicitly provided external QA modules")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to((ROOT / "outputs/neural-validation-v3/inputs").resolve()),
            "Figure outputs cannot modify the frozen input area")
    require(not args.output.resolve().is_relative_to(args.analysis.resolve()), "Figure outputs cannot modify aggregate source evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    configure()
    facts, inputs = frozen_workflow_facts()
    context = {"output": args.output, "auditors": Auditors(args.audit_scripts), "figures": {}}
    workflow(context, facts)
    if not args.workflow_only:
        evidence = AggregateEvidence(args.analysis)
        inputs.update(evidence.input_hashes)
        allocation(evidence, context)
        costs(evidence, context)
    source_script_sha256 = sha(Path(__file__))
    public_analysis_manifest_sha256 = None if args.workflow_only else sha(evidence.manifest_path)
    if not args.workflow_only:
        require(public_analysis_manifest_sha256 == inputs["public_analysis_manifest.json"],
                "Aggregate-release manifest changed during figure rendering")
    receipt = {"schema_version": "trace-neural-figures-v3", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "backend": "Python/matplotlib", "source_script_sha256": source_script_sha256,
               "plotting_source_sha256": source_script_sha256,
               "public_analysis_manifest_sha256": public_analysis_manifest_sha256,
               "workflow_only": args.workflow_only, "aggregate_target_effects_opened": not args.workflow_only,
               "predictions_decisions_raw_datasets_or_vaults_opened": False,
               "input_hashes": inputs, "figures": context["figures"],
               "quantitative_status": "Pending actual aggregate export and authorization" if args.workflow_only else "Rendered from complete bound aggregate evidence",
               "statistical_unit": "Project within dataset; actual fits averaged; no dataset pooling or independent-fit replication claim"}
    write_json(args.output / "figure_manifest_v3.json", receipt)
    print(json.dumps({"exported_figures": list(context["figures"]), "workflow_only": args.workflow_only,
                      "aggregate_target_effects_opened": not args.workflow_only,
                      "external_QA_requested": args.audit_scripts is not None}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        if "--audit-scripts" in sys.argv:
            index = sys.argv.index("--audit-scripts") + 1
            if index < len(sys.argv):
                directory = Path(sys.argv[index]).resolve()
                message = message.replace(str(directory), "<external-QA>").replace(directory.as_posix(), "<external-QA>")
        print("Figure export blocked: " + message, file=sys.stderr)
        raise SystemExit(1) from None
