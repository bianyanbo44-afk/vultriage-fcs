"""Extract frozen CodeBERT embeddings into a deterministic memmap cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from vultriage.codebert import MODEL_ID, MODEL_REVISION, masked_mean_pool
from vultriage.data import sha256


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for position, row in enumerate(rows):
        if int(row["position"]) != position:
            raise ValueError("manifest positions must be contiguous and ordered")
    return rows


def recover_code(
    rows: list[dict[str, str]], primevul_dir: Path, diversevul_path: Path
) -> Iterator[tuple[str, str]]:
    locations: dict[tuple[str, int], tuple[int, str]] = {}
    for position, row in enumerate(rows):
        location = (row["source_file"], int(row["line_number"]))
        if location in locations:
            raise ValueError(f"duplicate source location in manifest: {location}")
        locations[location] = (position, row["row_id"])
    code: list[str | None] = [None] * len(rows)
    paths = sorted(primevul_dir.glob("primevul_*.jsonl")) + [diversevul_path]
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                found = locations.get((path.name, line_number))
                if found is not None:
                    position, _ = found
                    code[position] = str(json.loads(line)["func"])
    if any(value is None for value in code):
        missing = [rows[index]["row_id"] for index, value in enumerate(code) if value is None]
        raise RuntimeError(f"could not recover {len(missing)} manifest rows")
    for row, text in zip(rows, code):
        assert text is not None
        yield row["row_id"], text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primevul-dir", type=Path, required=True)
    parser.add_argument("--diversevul", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    args = parser.parse_args()

    import psutil
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = manifest_rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("extension-v2 CodeBERT extraction requires verified CUDA")
    torch.manual_seed(20260814)
    torch.cuda.manual_seed_all(20260814)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, use_fast=True
    )
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    embedding_path = args.output / "embeddings.f32"
    embeddings = np.memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), hidden_size),
    )
    row_ids: list[str] = []
    batch_ids: list[str] = []
    batch_text: list[str] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    completed = 0

    def flush_batch() -> None:
        nonlocal completed, peak_rss
        if not batch_text:
            return
        tokens = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=int(args.maximum_tokens),
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = tokens.pop("special_tokens_mask").to(device)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model(**tokens).last_hidden_state
            pooled = masked_mean_pool(hidden, tokens["attention_mask"], special)
        block = pooled.float().cpu().numpy()
        embeddings[completed : completed + len(block)] = block
        row_ids.extend(batch_ids)
        completed += len(block)
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(f"embedded {completed}/{len(rows)}", flush=True)
        batch_ids.clear()
        batch_text.clear()

    for row_id, text in recover_code(rows, args.primevul_dir, args.diversevul):
        batch_ids.append(row_id)
        batch_text.append(text)
        if len(batch_text) == int(args.batch_size):
            flush_batch()
    flush_batch()
    embeddings.flush()
    row_id_path = args.output / "row_ids.txt"
    row_id_path.write_text("\n".join(row_ids) + "\n", encoding="utf-8")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": args.revision,
        "rows": len(rows),
        "hidden_size": hidden_size,
        "dtype": "float32",
        "shape": [len(rows), hidden_size],
        "pooling": "attention-mask mean excluding special tokens",
        "maximum_tokens": int(args.maximum_tokens),
        "batch_size": int(args.batch_size),
        "encoder_precision": "float16",
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_bytes": int(peak_rss),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "manifest_sha256": sha256(args.manifest),
        "embeddings_sha256": sha256(embedding_path),
        "row_ids_sha256": sha256(row_id_path),
        "labels_used": False,
    }
    # The extension manifest sidecar records the source/target boundary used
    # by the detector runner.  Copy only those label-free counts into the
    # embedding metadata so index alignment is checked before prediction.
    sidecar = args.manifest.with_name("manifest_metadata.json")
    if sidecar.is_file():
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        result["source_rows"] = int(sidecar_payload["source_rows"])
        result["target_rows"] = int(sidecar_payload["target_rows"])
        result["manifest_metadata_sha256"] = sha256(sidecar)
    (args.output / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
