#!/usr/bin/env python3
"""Prepare a Pandoc-friendly LaTeX view without changing the PDF source."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REFERENCE_LABELS = {
    "fig:workflow": "Figure 1",
    "tab:projects": "Table 1",
    "tab:methods": "Table 2",
    "fig:support": "Figure 2",
    "fig:projects": "Figure 3",
    "fig:alignment": "Figure 4",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
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
