#!/usr/bin/env python3
"""Add explicit page geometry to a Pandoc-produced DOCX for renderers."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.input, "r") as zin:
        document = ET.fromstring(zin.read("word/document.xml"))
        sect = document.find(f".//{{{W}}}sectPr")
        if sect is None:
            body = document.find(f".//{{{W}}}body")
            sect = ET.SubElement(body, f"{{{W}}}sectPr")
        for tag in ("pgSz", "pgMar", "cols", "docGrid"):
            node = sect.find(f"{{{W}}}{tag}")
            if node is not None:
                sect.remove(node)
        ET.SubElement(sect, f"{{{W}}}pgSz", {f"{{{W}}}w": "11906", f"{{{W}}}h": "16838"})
        ET.SubElement(sect, f"{{{W}}}pgMar", {
            f"{{{W}}}top": "1440", f"{{{W}}}right": "1440", f"{{{W}}}bottom": "1440", f"{{{W}}}left": "1440",
            f"{{{W}}}header": "720", f"{{{W}}}footer": "720", f"{{{W}}}gutter": "0",
        })
        ET.SubElement(sect, f"{{{W}}}cols", {f"{{{W}}}num": "1"})
        ET.SubElement(sect, f"{{{W}}}docGrid", {f"{{{W}}}linePitch": "360"})
        document_xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                payload = document_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
