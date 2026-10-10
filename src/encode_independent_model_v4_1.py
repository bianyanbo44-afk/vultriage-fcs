"""Tokenize the exact v3 roles with the pinned CodeT5 tokenizer; never open a vault."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from independent_model_v4_1 import ROOT, TARGET_LABEL_KEYS, config_and_freeze, verify_public_base, relative, log
from prepare_independent_model_v4_1 import read_rows, role_receipts
from vultriage.data import sha256


def encode_records(path: Path, output: Path, tokenizer, config_path: Path, model_path: Path, *, source: bool) -> dict:
    cfg, receipt = config_and_freeze(config_path)
    model_manifest = verify_public_base(model_path, cfg)
    rows = read_rows(path, source=source)
    role_hashes = role_receipts(rows) if source else None
    if source and (sha256(path) != cfg["source_rows_sha256"] or role_hashes != cfg["source_roles"]):
        raise ValueError("Source tokenization would change the frozen roles")
    if output.exists():
        raise FileExistsError(f"Preserve the existing token cache: {output}")
    output.mkdir(parents=True)
    n, width = len(rows), cfg["fine_tuned_codet5"]["maximum_tokens"]
    arrays = {"input_ids": np.lib.format.open_memmap(output / "input_ids.npy", mode="w+", dtype=np.uint16, shape=(n, width)),
              "attention_mask": np.lib.format.open_memmap(output / "attention_mask.npy", mode="w+", dtype=np.uint8, shape=(n, width)),
              "special_tokens_mask": np.lib.format.open_memmap(output / "special_tokens_mask.npy", mode="w+", dtype=np.uint8, shape=(n, width))}
    lengths, raw_lengths = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
    for start in range(0, n, 128):
        text = [r["text"] for r in rows[start:start + 128]]
        raw = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        raw_lengths[start:start + len(text)] = [len(v) for v in raw["input_ids"]]
        batch = tokenizer(text, max_length=width, truncation=True, padding="max_length", return_special_tokens_mask=True, return_tensors="np")
        for name, array in arrays.items():
            if name == "input_ids" and int(batch[name].max()) >= 65536:
                raise ValueError("Tokenizer vocabulary exceeds uint16")
            array[start:start + len(text)] = batch[name]
        lengths[start:start + len(text)] = batch["attention_mask"].sum(axis=1)
        if start % 4096 == 0:
            log(f"CodeT5 tokenize {path}: {start}/{n}")
    for array in arrays.values():
        array.flush()
    np.save(output / "lengths.npy", lengths)
    np.save(output / "raw_lengths.npy", raw_lengths)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            metadata = {k: v for k, v in row.items() if k != "text"}
            if not source and any(k.casefold() in TARGET_LABEL_KEYS for k in metadata):
                raise ValueError("Target token metadata leaks labels")
            stream.write(json.dumps(metadata, separators=(",", ":"), ensure_ascii=False) + "\n")
    result = {"schema_version": "trace-independent-model-v4.1-tokens-1", "rows": n, "maximum_tokens": width,
              "truncated": int((raw_lengths > width).sum()), "truncated_fraction": float((raw_lengths > width).mean()),
              "median_raw_tokens": float(np.median(raw_lengths)), "source_labels_present": source, "target_labels_present": False,
              "input_path": relative(path), "input_sha256": sha256(path), "config_sha256": sha256(config_path),
              "amendment_freeze_path": relative(receipt), "amendment_freeze_sha256": sha256(receipt),
              "model_id": model_manifest["model_id"], "model_revision": model_manifest["revision"],
              "model_manifest_path": relative(model_path / "model_manifest.json"), "model_manifest_sha256": sha256(model_path / "model_manifest.json"),
              "source_role_hashes": role_hashes, "tokenizer_class": type(tokenizer).__name__,
              "script_sha256": sha256(Path(__file__)),
              "hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "tokenization_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"CodeT5 token cache: {output}, n={n}, truncated={result['truncated']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg, _ = config_and_freeze(args.config)
    model = args.model_path or ROOT / cfg["paths"]["model_path"]
    output = args.output or ROOT / cfg["paths"]["output_root"] / "tokens"
    verify_public_base(model, cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    for name, path, source in [("source", ROOT / cfg["paths"]["source_rows"], True),
                               ("diversevul", ROOT / cfg["paths"]["diversevul_inputs"] / "target_rows.jsonl", False),
                               ("megavul", ROOT / cfg["paths"]["megavul_inputs"] / "target_rows.jsonl", False)]:
        encode_records(path, output / name, tokenizer, args.config, model, source=source)


if __name__ == "__main__":
    main()
