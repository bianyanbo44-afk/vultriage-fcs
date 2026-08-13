"""Audit PrimeVul split metadata without loading user-owned data.

Usage:
    python src/audit_primevul.py data/external/primevul_original
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalized_cwes(value: Any) -> list[str]:
    if value is None:
        return []
    raw: Iterable[Any] = value if isinstance(value, list) else [value]
    result = []
    for item in raw:
        text = str(item).strip()
        if text and text != "-":
            result.append(text)
    return result


def audit_file(path: Path) -> dict[str, Any]:
    labels: Counter[int] = Counter()
    projects: Counter[str] = Counter()
    vulnerable_projects: Counter[str] = Counter()
    cwe_types: Counter[str] = Counter()
    commits: set[str] = set()
    hashes: set[str] = set()

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            target = int(row["target"])
            project = str(row.get("project", "")).strip()
            labels[target] += 1
            projects[project] += 1
            if target:
                vulnerable_projects[project] += 1
            commits.add(str(row.get("commit_id", "")))
            hashes.add(str(row.get("hash", "")))
            cwe_types[type(row.get("cwe")).__name__] += 1
            normalized_cwes(row.get("cwe"))

    return {
        "file": path.name,
        "sha256": sha256(path),
        "total": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
        "project_count": len(projects),
        "commit_count": len(commits),
        "unique_hash_count": len(hashes),
        "projects": sorted(projects),
        "top_projects": projects.most_common(20),
        "top_vulnerable_projects": vulnerable_projects.most_common(20),
        "cwe_value_types": dict(cwe_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("primevul_*.jsonl"))
    if not paths:
        raise SystemExit(f"No PrimeVul JSONL files found under {args.data_dir}")

    reports = [audit_file(path) for path in paths]
    overlaps = []
    for left_index, left in enumerate(reports):
        left_projects = set(left["projects"])
        for right in reports[left_index + 1 :]:
            right_projects = set(right["projects"])
            overlaps.append(
                {
                    "left": left["file"],
                    "right": right["file"],
                    "intersection": len(left_projects & right_projects),
                    "union": len(left_projects | right_projects),
                }
            )

    print(json.dumps({"files": reports, "project_overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()

