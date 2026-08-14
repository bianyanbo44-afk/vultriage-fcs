#!/usr/bin/env python3
"""Prepare a Pandoc-friendly LaTeX view without changing the PDF source."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REFERENCE_LABELS = {
    "fig:workflow": "Figure 1",
    "tab:projects": "Table 1",
    "tab:detector": "Table 2",
    "fig:gate": "Figure 2",
    "fig:automation": "Figure 3",
    "fig:alignment": "Figure 4",
    "fig:calibration": "Figure 5",
}


def extract_required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Missing {label} in LaTeX source")
    return match.group(1).strip()


def make_pandoc_front_matter(text: str) -> str:
    abstract = extract_required(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, "abstract"
    )
    keywords = extract_required(r"\\keywords\{([^{}]+)\}", text, "keywords")
    authors = re.findall(r"\\author\[[^]]+\]\{([^{}]+)\}", text)
    addresses = re.findall(r"\\address\[[^]]+\]\{([^{}]+)\}", text)
    emails = extract_required(r"\\corremail\{([^{}]+)\}", text, "corresponding emails")
    if len(authors) != 2 or len(addresses) != 2:
        raise ValueError("Expected exactly two FCS authors and two affiliations")

    # Keep the Word-only author block plain.  LibreOffice treats the FCS
    # math-style affiliation markers (for example ``$^{1,*}$``) as literal
    # glyph boxes during Pandoc conversion, so use stable text labels here.
    author_block = (
        "\\author{" + authors[0].replace("~", " ") + " (1) and "
        + authors[1].replace("~", " ") + " (2)\\\\\n"
        + "1. " + addresses[0] + "\\\\\n"
        + "2. " + addresses[1] + "\\\\\n"
        + "Co-corresponding authors: " + emails + "}\n"
        + "\\date{}\n"
    )

    text = re.sub(r"\\documentclass\[research\]\{fcs\}", r"\\documentclass[10pt]{article}", text, count=1)
    text = re.sub(r"\\makeatletter.*?\\makeatother\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"\\shorttitle\{[^{}]*\}\s*", "", text)
    text = re.sub(r"\\author\[[^]]+\]\{[^{}]+\}\s*", "", text)
    text = re.sub(r"\\address\[[^]]+\]\{[^{}]+\}\s*", "", text)
    text = re.sub(r"\\corremail\{[^{}]+\}\s*", "", text)
    text = re.sub(r"\\fcssetup\{.*?\n\}\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}\s*", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"\\keywords\{[^{}]+\}\s*", "", text, count=1)
    text = re.sub(
        r"(\\title\{[^{}]+\}\s*)",
        lambda match: match.group(1) + author_block,
        text,
        count=1,
    )

    # The journal table is intentionally compact in the PDF (two project
    # blocks per row).  A single four-column table is more reliable in Word:
    # it prevents long names from being split across the paired blocks.
    project_table = re.search(
        r"\\begin\{table\*\}.*?\\label\{tab:projects\}.*?\\end\{table\*\}",
        text,
        flags=re.DOTALL,
    )
    if project_table:
        rows = []
        row_pattern = re.compile(
            r"^\s*([^&]+?)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9.]+)"
            r"\s*&\s*([^&]+?)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9.]+)\\\\\s*$",
        )
        for line in project_table.group(0).splitlines():
            match = row_pattern.match(line)
            if match:
                left = match.group(1).strip()
                right = match.group(5).strip()
                rows.append(
                    "{} & {} & {} & {} \\\\".format(
                        left,
                        match.group(2),
                        match.group(3),
                        match.group(4),
                    )
                )
                rows.append(
                    "{} & {} & {} & {} \\\\".format(
                        right,
                        match.group(6),
                        match.group(7),
                        match.group(8),
                    )
                )
        if len(rows) == 24:
            simplified = (
                "\\begin{table}[t]\n\\centering\n"
                "\\caption{Frozen DiverseVul external project groups. Target rows are exact-deduplicated functions; vulnerability prevalence is reported only descriptively after label release.}\n"
                "\\label{tab:projects}\n\\small\n"
                "\\begin{tabular}{lrrr}\n\\toprule\n"
                "Project & Total & Vulnerable & Prev. (\\%)\\\\\n\\midrule\n"
                + "\n".join(rows)
                + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
            )
            text = text[: project_table.start()] + simplified + text[project_table.end() :]

    body_front_matter = (
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
        "\\noindent\\textbf{Keywords:} " + keywords + "\n"
    )
    return text.replace(r"\begin{document}", body_front_matter, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    text = make_pandoc_front_matter(text)
    for label, display in REFERENCE_LABELS.items():
        text = text.replace(rf"Figure~\ref{{{label}}}", display)
        text = text.replace(rf"Fig.~\ref{{{label}}}", display)
        text = text.replace(rf"Table~\ref{{{label}}}", display)
    text = re.sub(r"(\\includegraphics(?:\[[^]]*\])?\{[^}]+)\.pdf\}", r"\1.png}", text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
