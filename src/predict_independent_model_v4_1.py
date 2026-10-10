"""Predict complete existing target rosters with genuine new CodeT5 checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from independent_model_v4_1 import (ROOT, CodeT5Detector, TokenCache, atomic_json,
                                 config_and_freeze, log, predict, relative, set_seed, verify_public_base)
from prepare_independent_model_v4_1 import read_rows
from vultriage.data import sha256


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def run(args) -> None:
    cfg, freeze = config_and_freeze(args.config)
    verify_public_base(args.model_path, cfg)
    source = TokenCache(args.source_tokens, source=True, config_path=args.config)
    target = TokenCache(args.target_tokens, source=False, config_path=args.config)
    source_rows, target_rows = read_rows(args.source_rows, source=True), read_rows(args.target_rows, source=False)
    if len(source.rows) != len(source_rows) or len(target.rows) != len(target_rows):
        raise ValueError("Input row/cache lengths differ")
    if target.manifest["input_sha256"] != sha256(args.target_rows) or source.manifest["input_sha256"] != sha256(args.source_rows):
        raise ValueError("Inference input rows differ from the tokenized frozen inputs")
    if sorted({r["project_group"] for r in target_rows}) != cfg["target_rosters"][args.dataset]["projects"] or len(target_rows) != cfg["target_rosters"][args.dataset]["rows"]:
        raise ValueError("Inference must preserve the complete frozen target roster")
    positions = np.asarray([i for i, r in enumerate(source.rows) if r["role"] != "train"], dtype=np.int64)
    checkpoints = set()
    for seed in cfg["fine_tuned_codet5"]["seeds"]:
        training = args.training / f"seed-{seed}" / "training_manifest.json"
        checkpoint = training.parent / "best_model.pt"
        receipt = json.loads(training.read_text(encoding="utf-8"))
        checkpoint_hash = sha256(checkpoint)
        if (receipt["config_sha256"] != sha256(args.config) or receipt["checkpoint_sha256"] != checkpoint_hash
                or receipt["source_rows_sha256"] != sha256(args.source_rows)
                or receipt["source_token_manifest_sha256"] != sha256(source.path / "tokenization_manifest.json")
                or receipt.get("actual_independent_fit") is not True or receipt["seed"] != seed
                or receipt.get("unused_pretrained_decoder_instantiated") is not False):
            raise ValueError("Actual CodeT5 checkpoint/fit/source identity mismatch")
        if checkpoint_hash in checkpoints:
            raise ValueError("Copied checkpoints cannot be counted as independent seed fits")
        checkpoints.add(checkpoint_hash)
        folder = args.output / f"seed-{seed}"
        if folder.exists():
            raise FileExistsError(f"Preserve prior complete/partial prediction evidence: {folder}")
        folder.mkdir(parents=True)
        started = time.perf_counter()
        set_seed(seed)
        model = CodeT5Detector(args.model_path).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        source_p, source_features = predict(model, source, positions, embeddings=True)
        target_p, target_features = predict(model, target, np.arange(len(target.rows)), embeddings=True)
        archive = folder / "predictions.npz"
        atomic_npz(archive, source_positions=positions, source_probability=source_p, source_features=source_features,
                   target_positions=np.arange(len(target.rows), dtype=np.int64), target_probability=target_p, target_features=target_features)
        metadata = dict(schema_version="trace-independent-model-v4.1-prediction-1", detector=cfg["detector"], dataset=args.dataset,
                        seed=seed, config_sha256=sha256(args.config), protocol_sha256=cfg["protocol_sha256"],
                        dataset_config_sha256=sha256(args.dataset_config) if args.dataset_config else None,
                        prediction_sha256=sha256(archive), prediction_path=relative(archive),
                        source_rows_path=relative(args.source_rows), source_rows_sha256=sha256(args.source_rows),
                        target_rows_path=relative(args.target_rows), target_rows_sha256=sha256(args.target_rows),
                        training_manifest_path=relative(training), training_manifest_sha256=sha256(training),
                        checkpoint_path=relative(checkpoint), checkpoint_sha256=checkpoint_hash,
                        source_token_manifest_path=relative(source.path / "tokenization_manifest.json"), source_token_manifest_sha256=sha256(source.path / "tokenization_manifest.json"),
                        target_token_manifest_path=relative(target.path / "tokenization_manifest.json"), target_token_manifest_sha256=sha256(target.path / "tokenization_manifest.json"),
                        amendment_freeze_path=relative(freeze), amendment_freeze_sha256=sha256(freeze),
                        model_manifest_path=relative(args.model_path / "model_manifest.json"), model_manifest_sha256=sha256(args.model_path / "model_manifest.json"),
                        model_id=cfg["fine_tuned_codet5"]["model_id"], model_revision=cfg["fine_tuned_codet5"]["revision"], feature_dimensions=512,
                        target_labels_accessed=False, target_labels_used_for_selection=False,
                        probability_storage="float32", feature_storage="float32", source_roles_predicted=["model_validation", "calibration"],
                        predictor_script_sha256=sha256(Path(__file__)), common_script_sha256=sha256(ROOT / "src/independent_model_v4_1.py"),
                        target_rows=len(target.rows), complete_projects=cfg["target_rosters"][args.dataset]["projects"],
                        created_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        atomic_json(folder / "metadata.json", metadata)
        log(f"Sealed complete CodeT5 predictions {args.dataset}/{seed}: rows={len(target_p)}, seconds={metadata['wall_seconds']:.1f}")
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--dataset", choices=("diversevul", "megavul"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--target-rows", type=Path, required=True)
    parser.add_argument("--source-tokens", type=Path, required=True)
    parser.add_argument("--target-tokens", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.dataset == "megavul") != (args.dataset_config is not None):
        raise ValueError("MegaVul requires its inherited dataset config binding")
    run(args)


if __name__ == "__main__":
    main()
