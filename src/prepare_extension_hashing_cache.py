"""Build a deterministic label-free hashing cache for extension-v2 targets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import sparse

from prepare_feature_cache import vectorizer
from vultriage.data import load_config, sha256
from vultriage.extension_data import exact_code_key
from vultriage.extension_inputs import read_target_manifest


def iter_recovered_target_code(
    rows: list[dict[str, str]], diversevul_path: Path
) -> Iterator[tuple[dict[str, str], str]]:
    """Recover only function text from the raw file and verify its frozen key."""

    expected = {
        int(row["line_number"]): (position, row)
        for position, row in enumerate(rows)
    }
    if len(expected) != len(rows):
        raise ValueError("Target manifest contains duplicate raw line locations")
    recovered: list[str | None] = [None] * len(rows)
    with diversevul_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            found = expected.get(line_number)
            if found is None:
                continue
            position, row = found
            raw = json.loads(line)
            function = str(raw.get("func", ""))
            if exact_code_key(function) != row["exact_code_key"]:
                raise ValueError(
                    f"Function text differs from frozen target manifest at line "
                    f"{line_number}"
                )
            recovered[position] = function
    missing = [rows[index]["row_id"] for index, text in enumerate(recovered) if text is None]
    if missing:
        raise RuntimeError(f"Could not recover {len(missing)} target functions")
    for row, function in zip(rows, recovered):
        assert function is not None
        yield row, function


def build_hashing_cache(
    *,
    diversevul: Path,
    manifest: Path,
    hashing_config_path: Path,
    extension_config_path: Path,
    output: Path,
    batch_size: int,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output.exists():
        raise FileExistsError(output)
    rows = read_target_manifest(manifest)
    hashing_config = load_config(hashing_config_path)
    extension_config = load_config(extension_config_path)
    # The protocol freezes the inherited config by repository path. The
    # command is expected to be run from the repository root, matching the
    # paths recorded in the preregistration.
    inherited_path = Path(extension_config["detectors"]["hashing_sgd"]["inherit"])
    requested_path = hashing_config_path
    if inherited_path.as_posix() != requested_path.as_posix():
        raise ValueError("Hashing configuration path differs from the extension-v2 freeze")

    output.mkdir(parents=True)
    transform = vectorizer(hashing_config)
    blocks: list[sparse.csr_matrix] = []
    row_ids: list[str] = []
    batch_ids: list[str] = []
    batch_text: list[str] = []

    def flush_batch() -> None:
        if not batch_text:
            return
        block = transform.transform(batch_text).tocsr().astype(np.float32)
        blocks.append(block)
        row_ids.extend(batch_ids)
        batch_ids.clear()
        batch_text.clear()
        print(f"vectorized {len(row_ids)}/{len(rows)}", flush=True)

    for row, function in iter_recovered_target_code(rows, diversevul):
        batch_ids.append(row["row_id"])
        batch_text.append(function)
        if len(batch_text) == batch_size:
            flush_batch()
    flush_batch()
    if len(row_ids) != len(rows):
        raise RuntimeError("Hashing cache row count differs from target manifest")
    if blocks:
        matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    else:
        matrix = sparse.csr_matrix(
            (0, int(hashing_config["hashing_vectorizer"]["n_features"])),
            dtype=np.float32,
        )
    feature_path = output / "features.npz"
    row_id_path = output / "row_ids.txt"
    sparse.save_npz(feature_path, matrix, compressed=True)
    row_id_path.write_text("\n".join(row_ids) + "\n", encoding="utf-8")
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": extension_config["protocol_version"],
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "nonzero": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "batch_size": int(batch_size),
        "manifest_sha256": sha256(manifest),
        "diversevul_sha256": sha256(diversevul),
        "hashing_config_sha256": sha256(hashing_config_path),
        "extension_config_sha256": sha256(extension_config_path),
        "features_sha256": sha256(feature_path),
        "row_ids_sha256": sha256(row_id_path),
        "function_keys_verified": len(rows),
        "labels_used": False,
        "labels_serialized": False,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diversevul", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hashing-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    result = build_hashing_cache(
        diversevul=args.diversevul,
        manifest=args.manifest,
        hashing_config_path=args.hashing_config,
        extension_config_path=args.extension_config,
        output=args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
