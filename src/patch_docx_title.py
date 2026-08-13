#!/usr/bin/env python3
"""Flatten the custom title macro in a converted DOCX."""

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
        paragraphs = document.findall(f".//{{{W}}}body/{{{W}}}p")
        if paragraphs:
            title = "".join(node.text or "" for node in paragraphs[0].iter(f"{{{W}}}t"))
            title = title.replace(r"\method", "VulTriage")
            for child in list(paragraphs[0]):
                paragraphs[0].remove(child)
            run = ET.SubElement(paragraphs[0], f"{{{W}}}r")
            ET.SubElement(run, f"{{{W}}}t").text = title
        document_xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                payload = document_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
