"""Audit the seven shared detector fits across both label-free target datasets.

This additional production gate cannot receive a target-label vault.  It
checks source prediction equality and current artifact bindings rather than
relying on an existing-prediction message or counting dataset copies as fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_neural_validation_v3 import verify_seal

PLANNED = {"finetuned_codebert": [13, 37, 73],
           "hashing_sgd": [13, 37, 73], "frozen_codebert": [13]}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    if "vault" in path.name.casefold():
        raise ValueError("Prediction lineage gate never accepts label-vault inputs")
    return json.loads(path.read_text(encoding="utf-8"))


def local(name: str) -> Path:
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT) or "vault" in p.name.casefold():
        raise ValueError("Evidence reference is outside the workspace or is a label vault")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/neural-validation-v3/predictions")
    parser.add_argument("--diverse-seal", type=Path)
    parser.add_argument("--mega-seal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.diverse_seal) != bool(args.mega_seal):
        parser.error("Pass both dataset seals together, or neither")
    checks, evidence = [], []

    def require(value: bool, label: str) -> None:
        if not bool(value):
            raise ValueError(label)
        checks.append(label)

    config_hash = sha(ROOT / "configs/neural_validation_v3.json")
    mega_hash = sha(ROOT / "configs/megavul_validation_v3.json")
    source_path = ROOT / "outputs/neural-validation-v3/inputs/source_rows.jsonl"
    source_hash = sha(source_path)
    source = [json.loads(line) for line in source_path.open(encoding="utf-8")]
    positions = np.asarray([i for i, row in enumerate(source) if row["role"] != "train"], dtype=np.int64)
    source_token = ROOT / "outputs/neural-validation-v3/tokens/source/tokenization_manifest.json"
    source_token_hash = sha(source_token)
    require(read(source_token)["input_sha256"] == source_hash, "Shared source token cache binds source rows")
    identities = {(d, seed) for d, seeds in PLANNED.items() for seed in seeds}
    records = {}
    for dataset in ("diversevul", "megavul"):
        target = ROOT / ("outputs/neural-validation-v3/inputs/target_rows.jsonl" if dataset == "diversevul"
                         else "outputs/neural-validation-v3/inputs/megavul/target_rows.jsonl")
        target_hash = sha(target)
        rows = sum(1 for _ in target.open(encoding="utf-8"))
        found = {}
        for path in sorted((args.predictions / dataset).rglob("metadata.json")):
            meta = read(path)
            key = meta["detector"], int(meta["seed"])
            require(key not in found, f"Unique prediction metadata for {dataset}/{key}")
            require(meta["dataset"] == dataset and meta["target_labels_accessed"] is False,
                    f"Label-free dataset identity for {dataset}/{key}")
            require(meta["predictor_script_sha256"] == sha(ROOT / "src/predict_neural_validation_v3.py"),
                    f"Executed predictor implementation binding for {dataset}/{key}")
            require(meta["config_sha256"] == config_hash and meta["source_rows_sha256"] == source_hash
                    and meta["target_rows_sha256"] == target_hash
                    and meta.get("dataset_config_sha256") == (mega_hash if dataset == "megavul" else None),
                    f"Current config/source/target binding for {dataset}/{key}")
            training_path, checkpoint = local(meta["training_manifest_path"]), local(meta["checkpoint_path"])
            training = read(training_path)
            require(sha(training_path) == meta["training_manifest_sha256"]
                    and sha(checkpoint) == meta["checkpoint_sha256"] == training["checkpoint_sha256"],
                    f"Current fitted receipt and checkpoint for {dataset}/{key}")
            require(training["config_sha256"] == config_hash and training["seed"] == key[1]
                    and training["target_labels_accessed"] is False,
                    f"Source-only fitted identity for {dataset}/{key}")
            executed = ROOT / ("src/train_neural_validation_v3.py" if key[0] == "finetuned_codebert"
                               else "src/predict_neural_validation_v3.py")
            require(training["script_sha256"] == sha(executed), f"Executed source-fit implementation binding for {dataset}/{key}")
            if key[0] == "finetuned_codebert":
                updates = training["encoder_query_relative_updates"]
                require(training["changed_encoder_tensor_count"] == training["encoder_tensor_count"]
                        and len(updates) == 12 and all(np.isfinite(v) and v > 0 for v in updates.values())
                        and training["trainable_parameters"] == training["total_parameters"] > 0,
                        f"Full encoder training evidence for {dataset}/{key}")
            elif key[0] == "hashing_sgd":
                require(training["actual_independent_fit"] is True, f"Actual SGD fit for {dataset}/{key}")
            elif key[0] == "frozen_codebert":
                require(training["actual_head_fit_count"] == 1 and training["encoder_updated"] is False,
                        f"One actual frozen-encoder head for {dataset}/{key}")
            if key[0] != "finetuned_codebert":
                require(training["source_rows_sha256"] == source_hash, f"Reference source fit binding for {dataset}/{key}")
            if key[0] != "hashing_sgd":
                require(meta["source_token_manifest_sha256"] == source_token_hash,
                        f"Shared source token receipt for {dataset}/{key}")
                tp = local(meta["target_token_manifest_path"])
                require(sha(tp) == meta["target_token_manifest_sha256"] and read(tp)["input_sha256"] == target_hash,
                        f"Current target token receipt for {dataset}/{key}")
            prediction = local(meta["prediction_path"])
            require(sha(prediction) == meta["prediction_sha256"], f"Current prediction archive for {dataset}/{key}")
            with np.load(prediction, allow_pickle=False) as data:
                require(not any("label" in name.casefold() for name in data.files), f"No prediction-label array for {dataset}/{key}")
                require(np.array_equal(data["source_positions"], positions)
                        and np.array_equal(data["target_positions"], np.arange(rows)),
                        f"Complete source/target prediction positions for {dataset}/{key}")
                for name in ("source_probability", "target_probability"):
                    p = data[name]
                    require(p.ndim == 1 and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(),
                            f"Finite valid {name} for {dataset}/{key}")
                require(len(data["target_probability"]) == rows and len(data["source_probability"]) == len(positions),
                        f"Prediction vector lengths for {dataset}/{key}")
                arrays = {name: data[name].copy() for name in ("source_positions", "source_probability", "source_features")
                          if name in data.files}
            found[key] = (meta, arrays)
            evidence.append(dict(dataset=dataset, detector=key[0], seed=key[1],
                                 metadata_sha256=sha(path), prediction_sha256=meta["prediction_sha256"],
                                 training_manifest_sha256=meta["training_manifest_sha256"],
                                 checkpoint_sha256=meta["checkpoint_sha256"]))
        require(set(found) == identities, f"All seven actual fitted detector identities present for {dataset}")
        records[dataset] = found
    for key in sorted(identities):
        a, b = records["diversevul"][key], records["megavul"][key]
        fields = ("config_sha256", "source_rows_sha256", "training_manifest_path", "training_manifest_sha256",
                  "checkpoint_path", "checkpoint_sha256")
        require(all(a[0][f] == b[0][f] for f in fields), f"Exactly one shared fitted object across target datasets for {key}")
        require(set(a[1]) == set(b[1]) and all(np.array_equal(a[1][f], b[1][f]) for f in a[1]),
                f"Identical source predictions and representations across target datasets for {key}")
    for detector in ("finetuned_codebert", "hashing_sgd"):
        require(len({records["diversevul"][detector, seed][0]["checkpoint_sha256"]
                     for seed in PLANNED[detector]}) == 3,
                f"Three different actual trained checkpoints for {detector}")
    caches = [local(records[dataset]["hashing_sgd", 13][0]["feature_cache_path"]) for dataset in ("diversevul", "megavul")]
    csr = []
    for dataset, cache in zip(("diversevul", "megavul"), caches):
        m = read(cache / "manifest.json")
        require(m["source_rows_sha256"] == source_hash
                and sha(cache / "source_features.npz") == m["hashes"]["source_features.npz"],
                f"Current source CSR feature binding for {dataset}")
        csr.append(sparse.load_npz(cache / "source_features.npz").tocsr())
    require(csr[0].shape == csr[1].shape and all(np.array_equal(getattr(csr[0], k), getattr(csr[1], k))
                                               for k in ("data", "indices", "indptr")),
            "Identical source hashing CSR content across target datasets")
    seals = []
    if args.diverse_seal:
        d, m = verify_seal(args.diverse_seal), verify_seal(args.mega_seal)
        require(d["dataset"] == "diversevul" and m["dataset"] == "megavul", "Dataset seal identities")
        require(d["actual_seeds"] == m["actual_seeds"] == PLANNED, "Both seals use the same seven actual source fits")
        require(d["all_planned_fits_present"] and m["all_planned_fits_present"], "No unresolved or failed planned fit")
        require(d["source_rows_path"] == m["source_rows_path"] and d["config_path"] == m["config_path"],
                "Both seals reuse the shared source input and protocol")
        for path in (args.diverse_seal, args.mega_seal):
            seals.append(dict(dataset=read(path)["dataset"], sha256=sha(path)))
    report = dict(status="PASS", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  target_labels_accessed=False, check_count=len(checks), checks=checks,
                  config_sha256=config_hash, megavul_config_sha256=mega_hash,
                  source_rows_sha256=source_hash, source_token_manifest_sha256=source_token_hash,
                  actual_shared_fits=PLANNED, independent_fitted_object_count=7,
                  target_dataset_prediction_copies=14, evidence=evidence, seals=seals,
                  checker_script_sha256=sha(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "evidence")}, indent=2))


if __name__ == "__main__":
    main()
