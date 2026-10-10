"""Recompute PROM v4 project averages/summaries from released aggregates only.

Loads only the executed aggregate_rows and write_csv function ASTs from the
included comparison source. No raw predictions, labels, official checkout,
model weights, pandas, or inference are needed for this bounded reproduction.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom/runtime-dependencies"
if PRIVATE.is_dir():
    sys.path.insert(0, str(PRIVATE))
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/kbs-validation-v4/prom/project_seed_metrics.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prom_official_v4.json")
    parser.add_argument("--implementation", type=Path, default=ROOT / "scripts/prom_official_v4.py")
    parser.add_argument("--expected", type=Path, default=ROOT / "outputs/kbs-validation-v4/prom")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kbs-validation-v4/prom/aggregate-recomputation")
    args = parser.parse_args()
    tree = ast.parse(args.implementation.read_text(encoding="utf-8"))
    wanted = {"aggregate_rows", "write_csv"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("Executed aggregation functions are not uniquely available")
    namespace = {"np": np, "defaultdict": defaultdict, "csv": csv, "Path": Path}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(args.implementation), "exec"), namespace)
    with args.input.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key, value in row.items():
            if value == "":
                row[key] = np.nan  # preserve undefined, rather than zero, denominators
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    averages, summaries, paired = namespace["aggregate_rows"](records, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, rows in (("project_seed_averages.csv", averages), ("policy_project_summary.csv", summaries), ("paired_project_summary.csv", paired)):
        path = args.output / name
        namespace["write_csv"](path, rows)
        expected = args.expected / name
        checks[name] = {"row_count": len(rows), "sha256": digest(path), "expected_sha256": digest(expected),
                        "exact_byte_match": path.read_bytes() == expected.read_bytes()}
    result = {"schema": "trace-prom-official-aggregate-recomputation-v4.1", "numpy_version": np.__version__,
              "scope": "Project/fit aggregates only; no raw labels/predictions or official-model inference",
              "implementation_sha256": digest(args.implementation), "input_sha256": digest(args.input),
              "config_sha256": digest(args.config), "checks": checks, "passed": all(v["exact_byte_match"] for v in checks.values())}
    (args.output / "recomputation_receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
