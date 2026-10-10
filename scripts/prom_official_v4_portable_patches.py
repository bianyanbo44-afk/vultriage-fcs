"""Render portable source diffs without altering immutable executed evidence.

The initial Windows text writer doubled carriage returns in archival diff
context lines. Those sealed files stay intact. Clean LF patch renditions are
independently applied here and must reconstruct each actual runtime bytewise.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom"
OUT = ROOT / "outputs/kbs-validation-v4/prom"


def sha(data):
    return hashlib.sha256(data).hexdigest().upper()


def normalized(data):
    return data.replace(b"\r\n", b"\n")


def apply_unified(source: bytes, patch: bytes) -> bytes:
    """Independent strict single-file unified-hunk application on LF text."""
    original = source.decode("utf-8").splitlines(keepends=True)
    lines = patch.decode("utf-8").splitlines(keepends=True)
    result, position, i = [], 0, 2
    while i < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n?", lines[i])
        if match is None:
            raise ValueError("Malformed portable patch hunk")
        start, old_count, _, new_count = match.groups()
        target = int(start) - 1
        if target < position:
            raise ValueError("Overlapping patch hunks")
        result.extend(original[position:target]); position = target
        removed = added = 0
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            marker, payload = line[:1], line[1:]
            if marker in (" ", "-"):
                if position >= len(original) or original[position] != payload:
                    raise ValueError("Patch context differs from actual pinned source")
                position += 1; removed += 1
            if marker in (" ", "+"):
                result.append(payload); added += 1
            elif marker != "-":
                raise ValueError("Unsupported patch marker")
            i += 1
        if removed != int(old_count or 1) or added != int(new_count or 1):
            raise ValueError("Portable patch hunk lengths differ")
    result.extend(original[position:])
    return "".join(result).encode("utf-8")


def main():
    stages = [
        ("binary_repair", WORK / "official-source/src/prom/prom_classification.py", WORK / "literal-binary-runtime/src/prom/prom_classification.py"),
        ("pooled_count_optimization", WORK / "literal-binary-runtime/src/prom/prom_classification.py", WORK / "binary-runtime/src/prom/prom_classification.py"),
    ]
    records = []
    for name, before, after in stages:
        source, expected = before.read_bytes(), after.read_bytes()
        first, second = normalized(source), normalized(expected)
        patch = "".join(difflib.unified_diff(first.decode("utf-8").splitlines(keepends=True),
                                           second.decode("utf-8").splitlines(keepends=True),
                                           fromfile="a/src/prom/prom_classification.py", tofile="b/src/prom/prom_classification.py")).encode("utf-8")
        path = WORK / f"prom_official_v4_{name}_portable.diff"
        path.write_bytes(patch)
        result = apply_unified(first, patch)
        ending = b"\r\n" if b"\r\n" in source else b"\n"
        restored = result.replace(b"\n", ending)
        if restored != expected:
            raise ValueError("Portable patch did not reconstruct executed runtime bytes")
        archival = WORK / f"prom_official_v4_{name}.diff"
        records.append({"stage": name, "portable_patch": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "portable_patch_sha256": sha(patch), "portable_patch_line_endings": "LF",
                        "before_sha256": sha(source), "expected_executed_after_sha256": sha(expected),
                        "independently_applied_restored_sha256": sha(restored), "byte_exact_runtime_roundtrip": True,
                        "archival_diff_sha256_preserved": sha(archival.read_bytes()),
                        "archival_doubled_carriage_returns": archival.read_bytes().count(b"\r\r\n")})
    receipt = {"schema": "trace-prom-portable-patch-delivery-qa-v4.2", "passed": True,
               "scope": "Post-evaluation source-diff text rendering only; no executed implementation, selector, seal, or result changed",
               "sealed_archival_diffs_modified": False, "executed_source_modified": False, "records": records}
    (OUT / "portable_patch_roundtrip.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
