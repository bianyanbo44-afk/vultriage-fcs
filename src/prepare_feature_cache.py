"""Build a deterministic, label-free hashing-feature cache for PrimeVul."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from vultriage.data import iter_manifest, load_config, sha256


def vectorizer(config: dict) -> HashingVectorizer:
    settings = config["hashing_vectorizer"]
    return HashingVectorizer(
        n_features=int(settings["n_features"]),
        ngram_range=tuple(settings["ngram_range"]),
        alternate_sign=bool(settings["alternate_sign"]),
        norm=settings["norm"],
        lowercase=bool(settings["lowercase"]),
        token_pattern=settings["token_pattern"],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    manifest_rows = list(iter_manifest(args.manifest))
    row_ids = [row["row_id"] for row in manifest_rows]
    position_by_location = {
        (row["source_file"], int(row["line_number"])): position
        for position, row in enumerate(manifest_rows)
    }
    code_by_position: list[str | None] = [None] * len(manifest_rows)
    for filename in (
        "primevul_train.jsonl",
        "primevul_valid.jsonl",
        "primevul_test.jsonl",
    ):
        with (args.data_dir / filename).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                position = position_by_location.get((filename, line_number))
                if position is not None:
                    code_by_position[position] = str(json.loads(line)["func"])
    if any(code is None for code in code_by_position):
        raise RuntimeError("At least one manifest row was not recovered from JSONL")

    transform = vectorizer(load_config(args.config))
    blocks = []
    for start in range(0, len(code_by_position), args.batch_size):
        batch = code_by_position[start : start + args.batch_size]
        blocks.append(transform.transform(batch))
        print(f"vectorized {min(start + args.batch_size, len(code_by_position))}/{len(code_by_position)}", flush=True)
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    feature_path = args.output / "features.npz"
    row_id_path = args.output / "row_ids.txt"
    sparse.save_npz(feature_path, matrix, compressed=True)
    row_id_path.write_text("\n".join(row_ids) + "\n", encoding="utf-8")
    metadata = {
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "nonzero": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config),
        "features_sha256": sha256(feature_path),
        "row_ids_sha256": sha256(row_id_path),
        "labels_used": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
