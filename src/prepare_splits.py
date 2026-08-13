"""Create the immutable, metadata-only PrimeVul split manifest.

This command does not train or evaluate a model. It intentionally excludes function
text from the output manifest while preserving source file/line pointers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vultriage.data import (
    collect_deduplicated_records,
    load_config,
    sha256,
    summarize_records,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    records, audit = collect_deduplicated_records(args.data_dir, config)
    write_manifest(records, args.manifest)
    summary = summarize_records(records, config, audit)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": args.config.as_posix(),
            "config_sha256": sha256(args.config),
            "manifest_path": args.manifest.as_posix(),
            "manifest_sha256": sha256(args.manifest),
            "input_sha256": {
                path.name: sha256(path)
                for path in sorted(args.data_dir.glob("primevul_*.jsonl"))
            },
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
