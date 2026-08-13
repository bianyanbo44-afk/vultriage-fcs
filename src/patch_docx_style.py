#!/usr/bin/env python3
"""Normalize converted manuscript DOCX style colors to black."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.input, "r") as zin:
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                payload = zin.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    text = payload.decode("utf-8")
                    if 'Extension="png"' not in text:
                        text = text.replace(
                            "</Types>",
                            '<Default Extension="png" ContentType="image/png"/></Types>',
                        )
                    payload = text.encode("utf-8")
                elif item.filename.endswith(".xml"):
                    text = payload.decode("utf-8")
                    for color in ("0F4761", "0F243E", "4F81BD", "365F91", "4070A0", "40A070", "60A0B0", "06287E"):
                        text = text.replace(color, "000000").replace(color.lower(), "000000")
                    text = text.replace('w:themeColor="accent1"', 'w:themeColor="text1"')
                    payload = text.encode("utf-8")
                zout.writestr(item, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
