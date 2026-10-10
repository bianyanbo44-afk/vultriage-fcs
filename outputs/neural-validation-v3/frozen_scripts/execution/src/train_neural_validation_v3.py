"""Three actual all-parameter CodeBERT fits using source labels only.

The CLI deliberately has no target data/label argument. Model selection uses
only the commit-separated source model-validation role. Target inference and
decision sealing are separate stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from transformers import AutoModel, get_linear_schedule_with_warmup

from vultriage.data import sha256


def log(message: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def atomic_json(path: Path, value: object):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    temp.replace(path)


class TokenCache:
    def __init__(self, path: Path, *, source: bool):
        self.path = path
        self.rows = [json.loads(line) for line in (path/"rows.jsonl").open(encoding="utf-8")]
        self.ids = np.load(path/"input_ids.npy", mmap_mode="r")
        self.mask = np.load(path/"attention_mask.npy", mmap_mode="r")
        self.special = np.load(path/"special_tokens_mask.npy", mmap_mode="r")
        self.lengths = np.load(path/"lengths.npy", mmap_mode="r")
        self.manifest = json.loads((path/"tokenization_manifest.json").read_text(encoding="utf-8"))
        for filename, digest in self.manifest["hashes"].items():
            if sha256(path/filename) != digest:
                raise ValueError(f"Changed token cache: {filename}")
        if not source and any("label" in row or "target" in row for row in self.rows):
            raise ValueError("Target token metadata contains labels")
        if source and not all("label" in row and row["role"] in
                                  ("train", "model_validation", "calibration") for row in self.rows):
            raise ValueError("Invalid source roles")
        self.labels = np.asarray([row["label"] for row in self.rows],dtype=np.int64) if source else None

    def batch(self, positions: np.ndarray):
        width = int(np.max(self.lengths[positions]))
        return {"input_ids": torch.as_tensor(np.asarray(self.ids[positions,:width],dtype=np.int64),device="cuda"),
                "attention_mask": torch.as_tensor(np.asarray(self.mask[positions,:width],dtype=np.int64),device="cuda"),
                "special_tokens_mask": torch.as_tensor(np.asarray(self.special[positions,:width],dtype=np.bool_),device="cuda")}


class CodeBertDetector(nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_path),local_files_only=True,
                                                add_pooling_layer=False,attn_implementation="sdpa")
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size,2)
        nn.init.normal_(self.head.weight,mean=0,std=self.encoder.config.initializer_range)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        hidden = self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        keep = attention_mask.bool() & ~special_tokens_mask
        count = keep.sum(dim=1)
        if bool((count==0).any()):
            raise ValueError("No nonspecial tokens")
        pooled = (hidden*keep.unsqueeze(-1)).sum(dim=1)/count.unsqueeze(-1)
        return self.head(self.dropout(pooled)), pooled


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def encoder_digests(model: CodeBertDetector) -> dict[str,str]:
    return {name:tensor_digest(value) for name,value in model.encoder.state_dict().items()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def length_batches(cache: TokenCache, positions: np.ndarray, size: int, seed: int):
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(positions)
    batches=[]
    for start in range(0,len(shuffled),128):
        window=shuffled[start:start+128]
        window=window[np.argsort(cache.lengths[window],kind="stable")]
        batches.extend(window[begin:begin+size] for begin in range(0,len(window),size))
    return [batches[i] for i in rng.permutation(len(batches))]


def predict(model: CodeBertDetector, cache: TokenCache, positions: np.ndarray,
            batch_size: int=8, *, embeddings: bool=False):
    model.eval()
    probability=np.empty(len(positions),dtype=np.float32)
    features=np.empty((len(positions),768),dtype=np.float32) if embeddings else None
    order=np.argsort(cache.lengths[positions],kind="stable")
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits, pooled=model(**cache.batch(positions[indices]))
            probability[indices]=torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy()
            if embeddings:
                features[indices]=pooled.float().cpu().numpy()
    return probability,features


def runtime():
    import transformers
    return dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cpu_threads=torch.get_num_threads(),platform=platform.platform())


def fit_one(args,cache: TokenCache,cfg: dict,seed: int):
    out=args.output/f"seed-{seed}"
    if (out/"training_manifest.json").exists():
        previous=json.loads((out/"training_manifest.json").read_text(encoding="utf-8"))
        if previous["config_sha256"]!=sha256(args.config) or sha256(out/"best_model.pt")!=previous["checkpoint_sha256"]:
            raise ValueError("Completed training seal mismatch")
        log(f"Verified completed seed {seed}; skip")
        return
    if out.exists():
        raise FileExistsError(f"Incomplete training exists; preserve and investigate: {out}")
    out.mkdir(parents=True)
    started=time.perf_counter()
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    settings=cfg["fine_tuned_codebert"]
    model=CodeBertDetector(args.model_path).cuda()
    if settings["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    train=np.asarray([i for i,row in enumerate(cache.rows) if row["role"]=="train"],dtype=np.int64)
    valid=np.asarray([i for i,row in enumerate(cache.rows) if row["role"]=="model_validation"],dtype=np.int64)
    initial=encoder_digests(model)
    initial_queries={name:value.detach().cpu().clone() for name,value in model.encoder.named_parameters()
                     if name.endswith("attention.self.query.weight")}
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    micro=settings["micro_batch_size"]
    accumulation=settings["effective_batch_size"]//micro
    steps_per_epoch=math.ceil(math.ceil(len(train)/micro)/accumulation)
    total_steps=steps_per_epoch*settings["epochs"]
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*settings["warmup_fraction"]),total_steps)
    best=-math.inf
    best_epoch=None
    history=[]
    global_step=0
    log(f"Seed {seed}: train={len(train)}, validation={len(valid)}, params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(1,settings["epochs"]+1):
        epoch_start=time.perf_counter()
        model.train()
        batches=length_batches(cache,train,micro,seed+104729*epoch)
        loss_sum=0.0
        for begin in range(0,len(batches),accumulation):
            window=batches[begin:begin+accumulation]
            n_window=sum(len(batch) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            for positions in window:
                labels=torch.as_tensor(cache.labels[positions],device="cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    logits,_=model(**cache.batch(positions))
                    summed=nn.functional.cross_entropy(logits.float(),labels,reduction="sum")
                if not bool(torch.isfinite(summed)):
                    raise FloatingPointError("Nonfinite training loss")
                (summed/n_window).backward()
                loss_sum+=float(summed.detach())
            nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            global_step+=1
            if global_step%25==0:
                log(f"seed={seed} epoch={epoch} step={global_step}/{total_steps} loss={loss_sum/max(1,sum(len(b) for b in batches[:begin+len(window)])):.4f} elapsed={time.perf_counter()-started:.0f}s")
                atomic_json(out/"progress.json",dict(seed=seed,epoch=epoch,step=global_step,total_steps=total_steps,
                    elapsed_seconds=time.perf_counter()-started,peak_gpu_allocated=torch.cuda.max_memory_allocated()))
        val_p,_=predict(model,cache,valid)
        ap=float(average_precision_score(cache.labels[valid],val_p))
        row=dict(epoch=epoch,train_cross_entropy=loss_sum/len(train),source_validation_ap=ap,
                 optimizer_steps=global_step,wall_seconds=time.perf_counter()-epoch_start)
        history.append(row)
        log(f"Completed seed {seed} epoch {epoch}: {json.dumps(row)}")
        if ap>best:
            best=ap
            best_epoch=epoch
            temp=out/"best_model.pt.tmp"
            torch.save(model.state_dict(),temp)
            temp.replace(out/"best_model.pt")
        atomic_json(out/"epoch_history.json",history)
    model.load_state_dict(torch.load(out/"best_model.pt",map_location="cuda",weights_only=True))
    final=encoder_digests(model)
    update_norms={name:float(torch.linalg.vector_norm(value.detach().cpu()-initial_queries[name])/
                              torch.linalg.vector_norm(initial_queries[name]))
                  for name,value in model.encoder.named_parameters() if name in initial_queries}
    if len(update_norms)!=12 or any(value<=0 for value in update_norms.values()):
        raise ValueError("Every encoder block must have a nonzero parameter update")
    if initial==final:
        raise ValueError("Encoder remained frozen")
    result=dict(seed=seed,config_sha256=sha256(args.config),source_token_manifest_sha256=sha256(cache.path/"tokenization_manifest.json"),
                script_sha256=sha256(Path(__file__)),selected_epoch=best_epoch,source_validation_ap=best,
                checkpoint_sha256=sha256(out/"best_model.pt"),encoder_initial_tensor_hashes=initial,
                encoder_final_tensor_hashes=final,encoder_query_relative_updates=update_norms,
                changed_encoder_tensor_count=sum(initial[k]!=final[k] for k in initial),
                encoder_tensor_count=len(initial),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),target_labels_accessed=False,
                target_inputs_accessed=False,history=history,runtime=runtime(),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.perf_counter()-started,
                completed_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/"training_manifest.json",result)
    log(f"Finished real encoder fit {seed}: epoch {best_epoch}, AP {best:.6f}")
    del model,optimizer,scheduler
    torch.cuda.empty_cache()


def benchmark(args,cache,cfg):
    set_seed(13)
    model=CodeBertDetector(args.model_path).cuda()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["fine_tuned_codebert"]["learning_rate"])
    positions=np.argsort(-cache.lengths,kind="stable")[:cfg["fine_tuned_codebert"]["micro_batch_size"]]
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits,_=model(**cache.batch(positions))
            loss=nn.functional.cross_entropy(logits.float(),torch.as_tensor(cache.labels[positions],device="cuda"))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    result=dict(worst_length=int(cache.lengths[positions].max()),micro_batch=len(positions),
                seconds_per_micro_batch=(time.perf_counter()-started)/4,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                runtime=runtime(),target_labels_accessed=False)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/"gpu_preflight.json",result)
    log(json.dumps(result))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-tokens",type=Path,required=True)
    parser.add_argument("--model-path",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/neural_validation_v3.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmark-only",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen protocol requires a CUDA GPU with bfloat16 support")
    cache=TokenCache(args.source_tokens,source=True)
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    if args.benchmark_only:
        benchmark(args,cache,cfg)
    else:
        for seed in cfg["fine_tuned_codebert"]["seeds"]:
            fit_one(args,cache,cfg,seed)

if __name__=="__main__":
    main()
