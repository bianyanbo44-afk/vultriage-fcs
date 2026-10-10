"""Tokenize source and label-free target records without opening target labels."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from vultriage.data import sha256

def encode_records(path: Path, output: Path, tokenizer, max_tokens: int, source: bool):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not source and any("label" in row or "target" in row for row in rows):
        raise ValueError("Target labels found in tokenization records")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    n = len(rows)
    arrays = {
        "input_ids": np.lib.format.open_memmap(output/"input_ids.npy",mode="w+",dtype=np.uint16,shape=(n,max_tokens)),
        "attention_mask": np.lib.format.open_memmap(output/"attention_mask.npy",mode="w+",dtype=np.uint8,shape=(n,max_tokens)),
        "special_tokens_mask": np.lib.format.open_memmap(output/"special_tokens_mask.npy",mode="w+",dtype=np.uint8,shape=(n,max_tokens))}
    lengths = np.empty(n,dtype=np.int32)
    raw_lengths = np.empty(n,dtype=np.int32)
    for start in range(0,n,128):
        text = [row["text"] for row in rows[start:start+128]]
        # Pre-truncation token counts are measured without retaining huge arrays.
        raw = tokenizer(text,add_special_tokens=True,truncation=False,padding=False)
        raw_lengths[start:start+len(text)] = [len(v) for v in raw["input_ids"]]
        batch = tokenizer(text,max_length=max_tokens,truncation=True,padding="max_length",
                          return_special_tokens_mask=True,return_tensors="np")
        for name, array in arrays.items():
            value = batch[name]
            if name == "input_ids" and int(value.max()) >= 65536:
                raise ValueError("Tokenizer vocabulary exceeds uint16")
            array[start:start+len(text)] = value
        lengths[start:start+len(text)] = batch["attention_mask"].sum(axis=1)
        if start % 4096 == 0:
            print(f"Tokenized {path.name}: {start}/{n}",flush=True)
    for array in arrays.values():
        array.flush()
    np.save(output/"lengths.npy",lengths)
    np.save(output/"raw_lengths.npy",raw_lengths)
    metadata = [{k:v for k,v in row.items() if k != "text"} for row in rows]
    if not source and any("label" in row for row in metadata):
        raise ValueError("Token metadata leaks labels")
    with (output/"rows.jsonl").open("w",encoding="utf-8") as stream:
        for row in metadata:
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    result=dict(rows=n,maximum_tokens=max_tokens,truncated=int((raw_lengths>max_tokens).sum()),
                truncated_fraction=float((raw_lengths>max_tokens).mean()),
                median_raw_tokens=float(np.median(raw_lengths)),
                source_labels_present=source,target_labels_present=False,
                input_sha256=sha256(path),
                hashes={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    (output/"tokenization_manifest.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model-path",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/neural_validation_v3.json"))
    parser.add_argument("--target-only",action="store_true")
    parser.add_argument("--source-only",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
    n=cfg["fine_tuned_codebert"]["maximum_tokens"]
    if not args.target_only:
        encode_records(args.inputs/"source_rows.jsonl",args.output/"source",tokenizer,n,True)
    if not args.source_only:
        encode_records(args.inputs/"target_rows.jsonl",args.output/"target",tokenizer,n,False)

if __name__=="__main__":
    main()
