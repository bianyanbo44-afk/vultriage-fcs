"""Source-only independent verification of saved encoder/head tensors and fresh starts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from independent_model_v4_1 import ROOT, CodeT5Detector, config_and_freeze, tensor_digest, verify_public_base
from vultriage.data import sha256


def audit(config: Path, output: Path) -> dict:
    cfg, freeze = config_and_freeze(config)
    if output.exists():
        raise FileExistsError("Preserve independent actual-weight audit evidence")
    root = ROOT / cfg["paths"]["output_root"]
    base_path = ROOT / cfg["paths"]["model_path"]
    verify_public_base(base_path, cfg)
    torch.set_num_threads(4)
    checkpoints, heads, reports = set(), set(), []
    for seed in cfg["fine_tuned_codet5"]["seeds"]:
        manifest_path = root / "training" / f"seed-{seed}" / "training_manifest.json"
        fit = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = manifest_path.parent / "best_model.pt"
        if sha256(checkpoint) != fit["checkpoint_sha256"]:
            raise ValueError("Saved actual checkpoint differs from the fit receipt")
        # CPU initialization recreates the source runner's initial head. No CUDA
        # model/tensors or target inputs are created by this verification.
        torch.manual_seed(seed)
        model = CodeT5Detector(base_path).cpu()
        base_hashes = {name: tensor_digest(value) for name, value in model.encoder.state_dict().items()}
        initial_head = {name: tensor_digest(value) for name, value in model.head.state_dict().items()}
        if base_hashes != fit["encoder_initial_tensor_hashes"] or initial_head != fit["classifier_initial_tensor_hashes"]:
            raise ValueError("A fit did not start from the pinned public encoder and independently seeded random head")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any("decoder" in name for name in state):
            raise ValueError("The unused pretrained decoder appeared in the selected detector checkpoint")
        actual_encoder = {name.removeprefix("encoder."): tensor_digest(value) for name, value in state.items() if name.startswith("encoder.")}
        actual_head = {name.removeprefix("head."): tensor_digest(value) for name, value in state.items() if name.startswith("head.")}
        if actual_encoder != fit["encoder_final_tensor_hashes"] or actual_head != fit["classifier_final_tensor_hashes"]:
            raise ValueError("The declared final tensor digests do not describe the actual selected checkpoint")
        updates = {}
        for name, parameter in model.encoder.named_parameters():
            if name.endswith("SelfAttention.q.weight"):
                selected = state["encoder." + name]
                updates[name] = float(torch.linalg.vector_norm(selected - parameter.detach()) / torch.linalg.vector_norm(parameter.detach()))
        if len(updates) != 6 or any(v <= 0 for v in updates.values()):
            raise ValueError("An actual selected encoder block was not updated")
        if any(not np.isclose(value, fit["encoder_query_relative_updates"][name], rtol=1e-8, atol=1e-10) for name, value in updates.items()):
            raise ValueError("A reported encoder update norm differs from the actual saved tensors")
        if fit["checkpoint_sha256"] in checkpoints or initial_head["weight"] in heads:
            raise ValueError("A checkpoint or independently claimed random head was copied")
        checkpoints.add(fit["checkpoint_sha256"])
        heads.add(initial_head["weight"])
        selected = max(fit["history"], key=lambda row: (row["source_validation_ap"], -row["epoch"]))
        if selected["epoch"] != fit["selected_epoch"] or selected["source_validation_ap"] != fit["source_validation_ap"]:
            raise ValueError("Checkpoint selection did not follow fixed source AP/earlier tie rule")
        if sum(p.numel() for p in model.parameters()) != fit["total_parameters"] or not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Actual instantiated encoder/head parameter accounting differs")
        for snapshot in fit["executed_script_snapshots"]:
            if sha256(ROOT / snapshot["snapshot_path"]) != snapshot["sha256"] or sha256(ROOT / snapshot["workspace_script"]) != snapshot["sha256"]:
                raise ValueError("Executed script snapshot/current bytes differ from actual fit evidence")
        reports.append({"seed": seed, "actual_checkpoint_sha256": fit["checkpoint_sha256"], "selected_epoch": fit["selected_epoch"],
                        "source_validation_ap": fit["source_validation_ap"], "all_initial_encoder_tensors_match_pinned_public_base": True,
                        "seeded_initial_head_tensors_independently_recreated": True, "actual_saved_final_encoder_head_tensors_match_manifest": True,
                        "actual_query_relative_updates": updates, "unused_decoder_absent": True,
                        "actual_instantiated_parameters": sum(p.numel() for p in model.parameters()),
                        "executed_script_snapshots_verified": len(fit["executed_script_snapshots"]),
                        "changed_actual_encoder_tensors": sum(base_hashes[n] != actual_encoder[n] for n in base_hashes)})
        del model, state
    result = {"schema_version": "trace-independent-model-v4.1-independent-actual-fit-audit-1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(config),
              "amendment_freeze_sha256": sha256(freeze), "source_only": True, "gpu_used": False,
              "target_inputs_or_vaults_opened": False, "all_three_actual_fits_verified": True,
              "checkpoint_and_initial_random_head_hashes_distinct": True,
              "model_manifest_sha256": sha256(base_path / "model_manifest.json"),
              "fits": reports, "script_sha256": sha256(Path(__file__))}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".sha256").write_text(sha256(output) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/independent_model_v4_1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / cfg["paths"]["output_root"] / "independent_actual_fit_audit.json"
    result = audit(args.config, output)
    print(json.dumps({"all_three_actual_fits_verified": result["all_three_actual_fits_verified"], "fits": result["fits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
