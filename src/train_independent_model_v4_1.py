"""Three actual CodeT5 encoder/classifier fits; CLI accepts source tokens only."""
from __future__ import annotations

import argparse
import math
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from transformers import get_linear_schedule_with_warmup

from independent_model_v4_1 import (
    ROOT, CodeT5Detector, TokenCache, atomic_json, config_and_freeze,
    encoder_digests, length_batches, log, predict, relative, runtime, set_seed, verify_public_base, tensor_digest,
)
from vultriage.data import sha256


def checkpoint_updates(model: CodeT5Detector, initial_queries: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(torch.linalg.vector_norm(value.detach().cpu() - initial_queries[name]) /
                        torch.linalg.vector_norm(initial_queries[name]))
            for name, value in model.encoder.named_parameters() if name in initial_queries}


def fit_one(args, cache: TokenCache, cfg: dict, seed: int) -> dict:
    out = args.output / f"seed-{seed}"
    if out.exists():
        raise FileExistsError(f"Keep prior fit/attempt evidence; output must be new: {out}")
    out.mkdir(parents=True)
    scripts = [Path(__file__).resolve(), ROOT / "src/independent_model_v4_1.py",
               ROOT / "src/prepare_independent_model_v4_1.py", ROOT / "src/prepare_independent_model_v4.py",
               ROOT / "src/audit_independent_model_v4.py", ROOT / "src/vultriage/data.py",
               ROOT / "src/vultriage/neural_audit.py", ROOT / "src/vultriage/near_duplicate.py"]
    snapshot_root = out / "executed_scripts"
    snapshot_root.mkdir()
    executed_snapshots = []
    for script in scripts:
        destination = snapshot_root / script.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script, destination)
        if sha256(destination) != sha256(script):
            raise ValueError("An executed script snapshot differs from the live implementation")
        executed_snapshots.append({"workspace_script": relative(script), "snapshot_path": relative(destination),
                                   "sha256": sha256(destination)})
    started = time.perf_counter()
    cfg, receipt = config_and_freeze(args.config)
    base = verify_public_base(args.model_path, cfg)
    settings = cfg["fine_tuned_codet5"]
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model = CodeT5Detector(args.model_path).cuda()
    if settings["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if model.encoder.config.d_model != settings["feature_dimensions"] or len(model.encoder.encoder.block) != settings["expected_encoder_blocks"]:
        raise ValueError("Pinned encoder architecture differs from the protocol")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("The amendment requires every instantiated encoder/head parameter trainable")
    initial = encoder_digests(model)
    initial_head = {name: tensor_digest(value) for name, value in model.head.state_dict().items()}
    initial_queries = {name: value.detach().cpu().clone() for name, value in model.encoder.named_parameters()
                       if name.endswith("SelfAttention.q.weight")}
    if len(initial_queries) != settings["expected_encoder_blocks"]:
        raise ValueError("Expected one query tensor in every CodeT5 encoder block")
    train = np.asarray([i for i, r in enumerate(cache.rows) if r["role"] == "train"], dtype=np.int64)
    valid = np.asarray([i for i, r in enumerate(cache.rows) if r["role"] == "model_validation"], dtype=np.int64)
    micro = settings["micro_batch_size"]
    accumulation = settings["effective_batch_size"] // micro
    steps_per_epoch = math.ceil(len(length_batches(cache, train, micro, seed + 104729)) / accumulation)
    total_steps = steps_per_epoch * settings["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings["warmup_fraction"]), total_steps)
    best, best_epoch, global_step = -math.inf, None, 0
    history, gradient_evidence = [], None
    log(f"CodeT5 seed={seed}: train={len(train)}, validation={len(valid)}, encoder={sum(p.numel() for p in model.encoder.parameters())}, head={sum(p.numel() for p in model.head.parameters())}, steps={total_steps}")
    for epoch in range(1, settings["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        batches = length_batches(cache, train, micro, seed + 104729 * epoch)
        loss_sum, seen = 0.0, 0
        for begin in range(0, len(batches), accumulation):
            window = batches[begin:begin + accumulation]
            n_window = sum(len(batch) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            for positions in window:
                labels = torch.as_tensor(cache.labels[positions], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(**cache.batch(positions))
                    summed = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
                if not bool(torch.isfinite(summed)):
                    raise FloatingPointError("Nonfinite source training loss")
                (summed / n_window).backward()
                loss_sum += float(summed.detach())
                seen += len(positions)
            if gradient_evidence is None:
                gradient_evidence = {name: {"received_gradient": value.grad is not None,
                                           "finite": value.grad is not None and bool(torch.isfinite(value.grad).all()),
                                           "nonzero": value.grad is not None and bool(torch.any(value.grad != 0))}
                                     for name, value in model.named_parameters()}
                if not all(row["received_gradient"] and row["finite"] and row["nonzero"] for row in gradient_evidence.values()):
                    raise ValueError("Every instantiated encoder/head parameter must receive a finite nonzero training gradient")
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite source gradient norm")
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 25 == 0:
                elapsed = time.perf_counter() - started
                log(f"CodeT5 seed={seed} epoch={epoch} step={global_step}/{total_steps} mean_loss={loss_sum / seen:.4f} elapsed={elapsed:.0f}s")
                atomic_json(out / "progress.json", dict(seed=seed, epoch=epoch, step=global_step, total_steps=total_steps,
                            elapsed_seconds=elapsed, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            source_only=True, target_labels_accessed=False))
        val_p, _ = predict(model, cache, valid)
        ap = float(average_precision_score(cache.labels[valid], val_p))
        row = dict(epoch=epoch, train_cross_entropy=loss_sum / len(train), source_validation_ap=ap,
                   optimizer_steps=global_step, wall_seconds=time.perf_counter() - epoch_start)
        history.append(row)
        log(f"Completed CodeT5 seed {seed} epoch {epoch}: {row}")
        if ap > best:
            best, best_epoch = ap, epoch
            temporary = out / "best_model.pt.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(out / "best_model.pt")
        atomic_json(out / "epoch_history.json", history)
    model.load_state_dict(torch.load(out / "best_model.pt", map_location="cuda", weights_only=True))
    final = encoder_digests(model)
    final_head = {name: tensor_digest(value) for name, value in model.head.state_dict().items()}
    updates = checkpoint_updates(model, initial_queries)
    if len(updates) != settings["expected_encoder_blocks"] or any(not np.isfinite(v) or v <= 0 for v in updates.values()):
        raise ValueError("Every CodeT5 encoder block must have an actual nonzero update")
    if initial == final:
        raise ValueError("The selected encoder was not updated")
    checkpoint_hash = sha256(out / "best_model.pt")
    result = dict(schema_version="trace-independent-model-v4.1-actual-fit-1", detector=cfg["detector"], seed=seed,
                  actual_independent_fit=True, config_sha256=sha256(args.config),
                  protocol_sha256=cfg["protocol_sha256"], amendment_freeze_path=relative(receipt), amendment_freeze_sha256=sha256(receipt),
                  source_rows_sha256=cfg["source_rows_sha256"], source_role_hashes=cfg["source_roles"],
                  source_train_rows=len(train), source_validation_rows=len(valid),
                  executed_script_snapshots=executed_snapshots,
                  source_token_manifest_sha256=sha256(cache.path / "tokenization_manifest.json"),
                  script_sha256=sha256(Path(__file__)), common_script_sha256=sha256(ROOT / "src/independent_model_v4_1.py"),
                  selected_epoch=best_epoch, source_validation_ap=best, checkpoint_sha256=checkpoint_hash,
                  encoder_initial_tensor_hashes=initial, encoder_final_tensor_hashes=final, encoder_query_relative_updates=updates,
                  classifier_initial_tensor_hashes=initial_head, classifier_final_tensor_hashes=final_head,
                  changed_encoder_tensor_count=sum(initial[k] != final[k] for k in initial), encoder_tensor_count=len(initial),
                  unchanged_encoder_tensors=[k for k in initial if initial[k] == final[k]],
                  initial_gradient_evidence=gradient_evidence,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()), classifier_parameters=sum(p.numel() for p in model.head.parameters()),
                  unused_pretrained_decoder_instantiated=False, unused_pretrained_decoder_trainable_parameters=0,
                  model_class=type(model.encoder).__name__, model_id=base["model_id"], model_revision=base["revision"],
                  model_manifest_path=relative(args.model_path / "model_manifest.json"), model_manifest_sha256=sha256(args.model_path / "model_manifest.json"),
                  target_labels_accessed=False, target_inputs_accessed=False, history=history,
                  optimizer_steps=global_step, planned_optimizer_steps=total_steps, wall_seconds=time.perf_counter() - started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  runtime=runtime(), created_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out / "training_manifest.json", result)
    log(f"Completed actual CodeT5 fit {seed}: selected_epoch={best_epoch}, checkpoint={checkpoint_hash}, seconds={result['wall_seconds']:.1f}")
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def benchmark(args, cache: TokenCache, cfg: dict) -> dict:
    if args.output.exists():
        raise FileExistsError("Preserve prior GPU preflight evidence")
    args.output.mkdir(parents=True)
    set_seed(13)
    verify_public_base(args.model_path, cfg)
    torch.cuda.reset_peak_memory_stats()
    model = CodeT5Detector(args.model_path).cuda()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    positions = np.asarray([i for i, r in enumerate(cache.rows) if r["role"] == "train"], dtype=np.int64)
    positions = positions[np.argsort(cache.lengths[positions], kind="stable")[-cfg["fine_tuned_codet5"]["micro_batch_size"]:]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["fine_tuned_codet5"]["learning_rate"])
    started = time.perf_counter()
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**cache.batch(positions))
            loss = nn.functional.cross_entropy(logits.float(), torch.as_tensor(cache.labels[positions], device="cuda"))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("GPU preflight produced nonfinite source loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    result = dict(schema_version="trace-independent-model-v4.1-source-only-preflight-1", config_sha256=sha256(args.config),
                  source_token_manifest_sha256=sha256(cache.path / "tokenization_manifest.json"),
                  model_manifest_sha256=sha256(args.model_path / "model_manifest.json"),
                  maximum_tested_length=int(cache.lengths[positions].max()), micro_batch=len(positions),
                  seconds_per_micro_batch=(time.perf_counter() - started) / 4, losses=losses,
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()), classifier_parameters=sum(p.numel() for p in model.head.parameters()),
                  unused_pretrained_decoder_instantiated=False,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  runtime=runtime(), target_labels_accessed=False, target_inputs_accessed=False, not_a_reported_fit=True)
    atomic_json(args.output / "gpu_preflight.json", result)
    log(str(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokens", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    cfg, _ = config_and_freeze(args.config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen amendment requires CUDA bfloat16 support")
    cache = TokenCache(args.source_tokens, source=True, config_path=args.config)
    if args.benchmark_only:
        benchmark(args, cache, cfg)
        return
    seeds = args.seed or cfg["fine_tuned_codet5"]["seeds"]
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(set(cfg["fine_tuned_codet5"]["seeds"])):
        raise ValueError("Only the independently planned actual fit seeds are allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        try:
            fit_one(args, cache, cfg, seed)
        except Exception as error:
            folder = args.output / f"seed-{seed}"
            if folder.is_dir() and not (folder / "training_manifest.json").exists():
                atomic_json(folder / "attempt_failure.json", dict(detector=cfg["detector"], seed=seed,
                            config_sha256=sha256(args.config), source_rows_sha256=cfg["source_rows_sha256"],
                            stage="actual_fit", error=repr(error), traceback=traceback.format_exc(),
                            target_labels_accessed=False, created_at_utc=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
