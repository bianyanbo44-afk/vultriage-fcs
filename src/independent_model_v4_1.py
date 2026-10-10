"""Shared, explicitly encoder-only CodeT5 utilities for the v4 amendment."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel

from prepare_independent_model_v4_1 import role_receipts
from vultriage.data import sha256

ROOT = Path(__file__).resolve().parents[1]
TARGET_LABEL_KEYS = frozenset({"label", "labels", "target", "target_label", "target_labels",
                              "is_vul", "is_vulnerable", "vulnerable", "y"})


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    absolute = path.resolve()
    if not absolute.is_relative_to(ROOT):
        raise ValueError("New model evidence must remain inside the workspace")
    return str(absolute.relative_to(ROOT))


def config_and_freeze(config_path: Path) -> tuple[dict, Path]:
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("protocol_version") != "trace-independent-model-v4.1":
        raise ValueError("Expected the dated independent-model v4 amendment")
    protocol = ROOT / "protocol/independent_model_v4_1.md"
    receipt_path = ROOT / cfg["paths"]["output_root"] / "amendment_freeze.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (sha256(config_path) != receipt["config_sha256"] or sha256(protocol) != receipt["protocol_sha256"]
            or receipt["protocol_sha256"] != cfg["protocol_sha256"]
            or sha256(receipt_path) != receipt_path.with_suffix(".sha256").read_text().strip()):
        raise ValueError("Amendment config/protocol/receipt was changed after freezing")
    for dependency in receipt["dependencies"]:
        if sha256(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError(f"Frozen amendment dependency changed: {dependency['role']}")
    return cfg, receipt_path


def verify_public_base(model_path: Path, cfg: dict) -> dict:
    manifest_path = model_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = cfg["fine_tuned_codet5"]
    if (manifest.get("model_id") != settings["model_id"] or manifest.get("revision") != settings["revision"]
            or manifest.get("public") is not True or manifest.get("gated") is not False):
        raise ValueError("Public model base differs from the pinned amendment")
    for filename, record in manifest["files"].items():
        path = model_path / filename
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Pinned public base file changed: {filename}")
    return manifest


class TokenCache:
    def __init__(self, path: Path, *, source: bool, config_path: Path):
        self.path = path.resolve()
        self.rows = [json.loads(line) for line in (path / "rows.jsonl").open(encoding="utf-8")]
        self.manifest = json.loads((path / "tokenization_manifest.json").read_text(encoding="utf-8"))
        cfg, _ = config_and_freeze(config_path)
        if (self.manifest.get("config_sha256") != sha256(config_path)
                or self.manifest.get("model_revision") != cfg["fine_tuned_codet5"]["revision"]
                or self.manifest.get("source_labels_present") is not source
                or self.manifest.get("target_labels_present") is not False):
            raise ValueError("Token cache has the wrong config/base/label boundary")
        for filename, digest in self.manifest["hashes"].items():
            if sha256(path / filename) != digest:
                raise ValueError(f"Changed token cache: {filename}")
        for position, row in enumerate(self.rows):
            if row.get("position") != position:
                raise ValueError("Token-cache rows are not aligned")
            if not source and any(k.casefold() in TARGET_LABEL_KEYS for k in row):
                raise ValueError("Target token metadata contains labels")
            if source and (row.get("role") not in ("train", "model_validation", "calibration") or row.get("label") not in (0, 1)):
                raise ValueError("Invalid source token-cache role/label")
        if source and (self.manifest["input_sha256"] != cfg["source_rows_sha256"]
                       or self.manifest["source_role_hashes"] != cfg["source_roles"]):
            raise ValueError("Source cache differs from exact frozen fitting roles")
        self.ids = np.load(path / "input_ids.npy", mmap_mode="r")
        self.mask = np.load(path / "attention_mask.npy", mmap_mode="r")
        self.special = np.load(path / "special_tokens_mask.npy", mmap_mode="r")
        self.lengths = np.load(path / "lengths.npy", mmap_mode="r")
        n = len(self.rows)
        width = cfg["fine_tuned_codet5"]["maximum_tokens"]
        if (self.ids.shape != (n, width) or self.mask.shape != self.ids.shape or self.special.shape != self.ids.shape
                or self.lengths.shape != (n,) or np.any(self.lengths <= 0) or np.any(self.lengths > width)):
            raise ValueError("Token-cache arrays are malformed")
        self.labels = np.asarray([r["label"] for r in self.rows], dtype=np.int64) if source else None

    def batch(self, positions: np.ndarray, device: str = "cuda") -> dict:
        positions = np.asarray(positions, dtype=np.int64)
        width = int(np.max(self.lengths[positions]))
        return {"input_ids": torch.as_tensor(np.asarray(self.ids[positions, :width], dtype=np.int64), device=device),
                "attention_mask": torch.as_tensor(np.asarray(self.mask[positions, :width], dtype=np.int64), device=device),
                "special_tokens_mask": torch.as_tensor(np.asarray(self.special[positions, :width], dtype=np.bool_), device=device)}


class CodeT5Detector(nn.Module):
    def __init__(self, model_path: Path | None = None, *, encoder: T5EncoderModel | None = None):
        super().__init__()
        if encoder is None:
            if model_path is None:
                raise ValueError("A pinned base path or test encoder is required")
            encoder = T5EncoderModel.from_pretrained(str(model_path), local_files_only=True, attn_implementation="eager")
        self.encoder = encoder
        if any("decoder" in name for name, _ in self.encoder.named_parameters()):
            raise ValueError("The unused pretrained decoder must not be instantiated")
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.d_model, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        keep = attention_mask.bool() & ~special_tokens_mask.bool()
        count = keep.sum(dim=1)
        if bool((count == 0).any()):
            raise ValueError("CodeT5 pooling requires a nonspecial token")
        pooled = (hidden * keep.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
        return self.head(self.dropout(pooled)), pooled


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest().upper()


def encoder_digests(model: CodeT5Detector) -> dict[str, str]:
    return {name: tensor_digest(value) for name, value in model.encoder.state_dict().items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def length_batches(cache: TokenCache, positions: np.ndarray, size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(positions)
    batches = []
    for start in range(0, len(shuffled), 128):
        window = shuffled[start:start + 128]
        window = window[np.argsort(cache.lengths[window], kind="stable")]
        batches.extend(window[begin:begin + size] for begin in range(0, len(window), size))
    return [batches[i] for i in rng.permutation(len(batches))]


def predict(model: CodeT5Detector, cache: TokenCache, positions: np.ndarray,
            batch_size: int = 8, *, embeddings: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probability = np.empty(len(positions), dtype=np.float32)
    features = np.empty((len(positions), model.encoder.config.d_model), dtype=np.float32) if embeddings else None
    order = np.argsort(cache.lengths[positions], kind="stable")
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, pooled = model(**cache.batch(positions[indices]))
            probability[indices] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            if embeddings:
                features[indices] = pooled.float().cpu().numpy()
    if not np.isfinite(probability).all() or (features is not None and not np.isfinite(features).all()):
        raise FloatingPointError("Nonfinite CodeT5 predictions/features")
    return probability, features


def runtime() -> dict:
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cuda=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cpu_threads=torch.get_num_threads(), platform=platform.platform())
