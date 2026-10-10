"""Seal label-free CodeT5-v4 policies on the complete inherited rosters, then separately verify and evaluate them on CPU.

Examples (run with the repository's src directory on the Python import path)::

    python src/evaluate_independent_model_v4_1.py seal --config configs/independent_model_v4_1.json \
        --inputs outputs/neural-validation-v3/inputs --predictions outputs/kbs-validation-v4/independent_model/predictions/diversevul \
        --output outputs/kbs-validation-v4/independent_model/decisions/diversevul
    python src/evaluate_independent_model_v4_1.py evaluate --seal outputs/kbs-validation-v4/independent_model/decisions/diversevul/decision_seal.json \
        --target-label-vault outputs/neural-validation-v3/inputs/target_label_vault.npz \
        --output outputs/kbs-validation-v4/independent_model/evaluation/diversevul

For the second dataset, seal additionally accepts --dataset-config and
--source-rows; the shared source JSONL need not be copied to its input directory.
The seal subcommand deliberately has no target-label-vault argument.
Hashing feature caches are read from their prediction metadata; an explicit
--feature-cache must name that same directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse

from analyze_neurocomputing_controls import (
    action_codes, fit_source_isotonic, prediction_sets, probability_vector, verified_path,
)
from vultriage.conformal import kish_ess
from vultriage.data import sha256
from vultriage.domain import clipped_weight_summary
from vultriage.neural_audit import (
    DOMAIN_ALPHA, DOMAIN_EPOCHS, DOMAIN_FOLDS, DOMAIN_SALT, METHODS,
    DomainFitUnavailable, average_actual_seed_metrics, construct_decisions,
    canonical_commit_id, crossfit_grouped_density_ratio, detector_metrics, project_commit_counts,
    project_metrics, paired_project_summary,
)

WORKSPACE = Path(__file__).resolve().parents[1]
DETECTORS = ("finetuned_codet5_small",)
TARGET_LABEL_KEYS = frozenset({"label", "labels", "target", "target_label", "target_labels",
                             "is_vul", "is_vulnerable", "vulnerable", "y"})
EVIDENCE_FIELDS = ("training_manifest", "checkpoint", "amendment_freeze", "model_manifest")
TOKEN_EVIDENCE_FIELDS = ("source_token_manifest", "target_token_manifest")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(_clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _forbid_vault(path: Path) -> None:
    if "target_label_vault" in path.name.casefold():
        raise ValueError("seal must never open the target-label vault")


def _artifact(path: Path, role: str, expected_hash: str | None = None) -> dict:
    path = path.resolve()
    _forbid_vault(path)
    actual = sha256(path)
    if expected_hash is not None and actual != str(expected_hash).upper():
        raise ValueError(f"Artifact hash mismatch ({role}): {path}")
    return {"path": str(path), "role": role, "sha256": actual}


def _rows(path: Path, *, source: bool) -> list[dict]:
    _forbid_vault(path)
    records, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Every metadata row must be a JSON object")
            if not source and any(k.casefold() in TARGET_LABEL_KEYS for k in row):
                raise ValueError("Target metadata contains labels")
            row = {k: v for k, v in row.items() if k != "text"}
            if (row.get("position") != len(records) or not isinstance(row.get("position"), int)
                    or isinstance(row.get("position"), bool)):
                raise ValueError("Metadata positions must be consecutive input row indices")
            for field in ("row_id", "project_group", "commit_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing metadata identity: {field}")
            if row["row_id"] in ids:
                raise ValueError("Duplicate metadata row ID")
            ids.add(row["row_id"])
            if source:
                if row.get("role") not in ("train", "model_validation", "calibration"):
                    raise ValueError("Unknown source role")
                if row.get("label") not in (0, 1) or isinstance(row.get("label"), bool):
                    raise ValueError("Invalid source label")
            records.append(row)
    if not records:
        raise ValueError("Metadata cohort is empty")
    return records


def _verify_role_boundaries(source: list[dict], target: list[dict]) -> None:
    source_projects = {r["project_group"].strip().casefold() for r in source}
    target_projects = {r["project_group"].strip().casefold() for r in target}
    if source_projects & target_projects:
        raise ValueError("Source and target projects overlap")
    roles: dict[str, str] = {}
    for row in source:
        commit = canonical_commit_id(row["commit_id"])
        if commit in roles and roles[commit] != row["role"]:
            raise ValueError("Source commit appears in multiple roles")
        roles[commit] = row["role"]
    if set(roles) & {canonical_commit_id(r["commit_id"]) for r in target}:
        raise ValueError("Source and target commits overlap")


def _metadata_path(prediction: Path) -> Path:
    options = [prediction.with_suffix(".json"), prediction.with_suffix(".metadata.json")]
    folder_metadata = prediction.parent / "metadata.json"
    if folder_metadata.is_file():
        metadata = _json(folder_metadata)
        declared_name = Path(metadata.get("prediction_path", "predictions.npz")).name
        if declared_name == prediction.name:
            options.append(folder_metadata)
    found = [p for p in options if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Prediction needs exactly one adjacent metadata JSON: {prediction}")
    return found[0]


def _prediction_paths(values: list[Path]) -> list[Path]:
    result = []
    for value in values:
        if value.is_dir():
            for path in sorted(value.rglob("*.npz")):
                _forbid_vault(path)
                if (path.with_suffix(".json").is_file() or path.with_suffix(".metadata.json").is_file()
                        or (path.name == "predictions.npz" and (path.parent / "metadata.json").is_file())):
                    result.append(path.resolve())
        else:
            _forbid_vault(value)
            result.append(value.resolve())
    if not result or len(set(result)) != len(result):
        raise ValueError("Provide nonempty, distinct prediction archives")
    return sorted(result)


def _positions(values: np.ndarray, expected: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or len(values) != len(expected) or len(set(values.tolist())) != len(values)
            or not np.array_equal(np.sort(values), expected)):
        raise ValueError(f"{name} must cover each expected metadata position exactly once")
    order = np.argsort(values, kind="stable")
    return values[order].astype(np.int64), order


def _verify_training_manifest(value: dict, detector: str, seed: int, config_hash: str) -> None:
    if value.get("seed") != seed or value.get("config_sha256", "").upper() != config_hash:
        raise ValueError("Training identity/config does not match prediction metadata")
    if value.get("target_labels_accessed") is not False:
        raise ValueError("Training manifest must declare no target-label access")
    if detector == "finetuned_codet5_small":
        updates = value.get("encoder_query_relative_updates", {})
        initial, final = value.get("encoder_initial_tensor_hashes", {}), value.get("encoder_final_tensor_hashes", {})
        if (len(updates) != 6 or any(not np.isfinite(v) or v <= 0 for v in updates.values())
                or not initial or set(initial) != set(final) or initial == final
                or value.get("changed_encoder_tensor_count", 0) <= 0
                or value.get("trainable_parameters", 0) <= 0
                or value.get("trainable_parameters", 0) != value.get("total_parameters", -1)):
            raise ValueError("Training manifest does not demonstrate full encoder updating")
        if (value.get("schema_version") != "trace-independent-model-v4.1-actual-fit-1"
                or value.get("actual_independent_fit") is not True
                or value.get("model_class") != "T5EncoderModel"
                or value.get("model_id") != "Salesforce/codet5-small"
                or value.get("model_revision") != "b1ee9570c289f21b5922b9c768a1ce12957bf968"
                or value.get("unused_pretrained_decoder_instantiated") is not False
                or value.get("unused_pretrained_decoder_trainable_parameters") != 0
                or value.get("encoder_parameters", 0) <= 0
                or value.get("classifier_parameters") != 1026
                or value.get("encoder_parameters", 0) + value.get("classifier_parameters", 0) != value.get("total_parameters")
                or value.get("source_role_hashes", {}).get("train", {}).get("rows", 0) <= 0
                or value.get("optimizer_steps") != 3 * math.ceil(value.get("source_role_hashes", {}).get("train", {}).get("rows", 0) / 32)):
            raise ValueError("Training manifest does not establish the actual complete CodeT5 encoder-only fit")
        gradients = value.get("initial_gradient_evidence", {})
        if not gradients or not all(row.get("received_gradient") and row.get("finite") and row.get("nonzero")
                                    for row in gradients.values()):
            raise ValueError("The instantiated encoder/head parameters lack actual gradient evidence")


def _load_prediction(path: Path, *, workspace: Path, source: list[dict], target: list[dict],
                     dataset: str, hashes: dict, planned_seeds: dict[str, list[int]],
                     artifact_cache: dict[str, dict]) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    metadata_path = _metadata_path(path)
    metadata = _json(metadata_path)
    detector, seed = metadata.get("detector"), metadata.get("seed")
    if (detector not in DETECTORS or not isinstance(seed, int) or isinstance(seed, bool)
            or seed not in planned_seeds.get(detector, [])):
        raise ValueError("Unexpected detector/actual training seed; this amendment permits only actual CodeT5 seeds")
    if metadata.get("dataset") != dataset or metadata.get("target_labels_accessed") is not False:
        raise ValueError("Prediction dataset or target-label boundary declaration is invalid")
    for field, expected in hashes.items():
        actual = metadata.get(field)
        if expected is None:
            if actual is not None:
                raise ValueError(f"Unexpected prediction binding: {field}")
        elif not isinstance(actual, str) or actual.upper() != expected:
            raise ValueError(f"Prediction input binding mismatch: {field}")
    refs = [_artifact(path, "prediction", metadata.get("prediction_sha256", "")),
            _artifact(metadata_path, "prediction_metadata")]
    evidence = {}
    prefixes = EVIDENCE_FIELDS + (TOKEN_EVIDENCE_FIELDS if detector != "hashing_sgd" else ())
    for prefix in prefixes:
        relative, digest = metadata.get(f"{prefix}_path"), metadata.get(f"{prefix}_sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or not isinstance(digest, str):
            raise ValueError(f"Missing workspace-relative evidence binding: {prefix}")
        evidence_path = (workspace / relative).resolve()
        _forbid_vault(evidence_path)
        if not evidence_path.is_relative_to(workspace):
            raise ValueError("Prediction evidence escapes the workspace")
        cache_key = str(evidence_path)
        if cache_key not in artifact_cache:
            artifact_cache[cache_key] = _artifact(evidence_path, prefix, digest)
        if artifact_cache[cache_key]["sha256"] != digest.upper():
            raise ValueError("Contradictory hashes for shared prediction evidence")
        refs.append(artifact_cache[cache_key])
        evidence[prefix] = evidence_path
    if "prediction_path" in metadata:
        declared = Path(metadata["prediction_path"])
        if declared.is_absolute() or (workspace / declared).resolve() != path.resolve():
            raise ValueError("Prediction filename/path does not match its metadata binding")
    if detector == "hashing_sgd":
        relative = metadata.get("feature_cache_path")
        digest = metadata.get("feature_cache_manifest_sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or not isinstance(digest, str):
            raise ValueError("Hashing metadata must bind its workspace-relative feature cache")
        cache_path = (workspace / relative).resolve()
        if not cache_path.is_relative_to(workspace):
            raise ValueError("Hashing feature cache escapes the workspace")
        refs.append(_artifact(cache_path / "manifest.json", "hashing_feature_manifest", digest))
        metadata = {**metadata, "verified_feature_cache_path": str(cache_path)}
    training = _json(evidence["training_manifest"])
    _verify_training_manifest(training, detector, int(seed), hashes["config_sha256"])
    freeze = _json(evidence["amendment_freeze"])
    model_base = _json(evidence["model_manifest"])
    if (freeze.get("config_sha256") != hashes["config_sha256"]
            or freeze.get("source_rows_sha256") != hashes["source_rows_sha256"]
            or training.get("source_role_hashes") != freeze.get("source_roles")
            or training.get("amendment_freeze_sha256") != metadata.get("amendment_freeze_sha256")
            or training.get("model_manifest_sha256") != metadata.get("model_manifest_sha256")
            or model_base.get("revision") != training.get("model_revision")
            or metadata.get("model_revision") != training.get("model_revision")
            or metadata.get("feature_dimensions") != 512):
        raise ValueError("Prediction/base/role/amendment lineage is inconsistent")
    for filename, record in model_base["files"].items():
        refs.append(_artifact(evidence["model_manifest"].parent / filename, "pinned_public_base_file", record["sha256"]))
    if training.get("checkpoint_sha256", "").upper() != metadata["checkpoint_sha256"].upper():
        raise ValueError("Prediction checkpoint differs from the selected training checkpoint")
    if "source_rows_sha256" in training and str(training["source_rows_sha256"]).upper() != hashes["source_rows_sha256"]:
        raise ValueError("Training used a different source-row preparation")
    if detector != "hashing_sgd":
        for prefix, input_hash, source_labels in (
            ("source_token_manifest", hashes["source_rows_sha256"], True),
            ("target_token_manifest", hashes["target_rows_sha256"], False),
        ):
            token_manifest = _json(evidence[prefix])
            if (str(token_manifest.get("input_sha256", "")).upper() != input_hash
                    or token_manifest.get("source_labels_present") is not source_labels
                    or token_manifest.get("target_labels_present") is not False
                    or token_manifest.get("config_sha256") != hashes["config_sha256"]
                    or token_manifest.get("model_revision") != training.get("model_revision")):
                raise ValueError("Token cache input/label boundary does not match the prediction")
            if source_labels and token_manifest.get("source_role_hashes") != freeze["source_roles"]:
                raise ValueError("Token cache changed the exact source-role assignments")
            for filename, digest in token_manifest["hashes"].items():
                token_file = evidence[prefix].parent / filename
                cache_key = str(token_file.resolve())
                if cache_key not in artifact_cache:
                    artifact_cache[cache_key] = _artifact(token_file, "verified_token_cache_file", digest)
                refs.append(artifact_cache[cache_key])
        if ("source_token_manifest_sha256" in training
                and str(training["source_token_manifest_sha256"]).upper() != metadata["source_token_manifest_sha256"].upper()):
            raise ValueError("Training and prediction bind different source token caches")
    for digest, script_name in ((metadata.get("predictor_script_sha256"), "predict_independent_model_v4_1.py"),
                                (training.get("script_sha256"), "train_independent_model_v4_1.py" if detector == "finetuned_codet5_small"
                                 else "predict_independent_model_v4_1.py")):
        if digest is not None:
            refs.append(_artifact(workspace / "src" / script_name, "detector_implementation", digest))
    if metadata.get("common_script_sha256") != training.get("common_script_sha256"):
        raise ValueError("Fit and predictor used different CodeT5 implementations")
    refs.append(_artifact(workspace / "src/independent_model_v4_1.py", "codet5_shared_implementation", training["common_script_sha256"]))
    with np.load(path, allow_pickle=False) as archive:
        if any(k.casefold() in TARGET_LABEL_KEYS or k.casefold().endswith(("_labels", "_label")) for k in archive.files):
            raise ValueError("Prediction archive contains label arrays")
        required = {"source_positions", "source_probability", "target_positions", "target_probability"}
        if not required.issubset(archive.files):
            raise ValueError("Prediction archive is missing aligned probability arrays")
        arrays = {k: archive[k] for k in archive.files}
    expected_source = np.asarray([r["position"] for r in source if r["role"] != "train"], dtype=np.int64)
    source_positions, source_order = _positions(arrays["source_positions"], expected_source, "source_positions")
    target_positions, target_order = _positions(arrays["target_positions"], np.arange(len(target)), "target_positions")
    result = {"source_positions": source_positions, "target_positions": target_positions}
    for name, order in (("source", source_order), ("target", target_order)):
        probability = probability_vector(arrays[f"{name}_probability"])
        if len(probability) != len(order):
            raise ValueError("Prediction probabilities do not align to positions")
        result[f"{name}_probability"] = probability[order]
        feature_name = f"{name}_features"
        if detector != "hashing_sgd":
            if feature_name not in arrays:
                raise ValueError("Neural density weighting needs the final detector's features")
            features = arrays[feature_name]
            if features.shape != (len(order), 512) or not np.isfinite(features).all():
                raise ValueError("Neural features must be finite, aligned 512-dimensional CodeT5 encoder vectors")
            result[feature_name] = np.asarray(features[source_order if name == "source" else target_order], dtype=np.float32)
    return metadata, result, refs


def _hashing_cache(path: Path | None, source_n: int, target_n: int, hashes: dict) -> tuple[Any, Any, list[dict]]:
    if path is None:
        raise ValueError("Hashing predictions require --feature-cache")
    manifest_path = path / "manifest.json"
    manifest = _json(manifest_path)
    for field in ("source_rows_sha256", "target_rows_sha256"):
        if manifest.get(field, "").upper() != hashes[field]:
            raise ValueError(f"Hashing cache input binding mismatch: {field}")
    refs = [_artifact(manifest_path, "hashing_feature_manifest")]
    matrices = []
    for filename, n in (("source_features.npz", source_n), ("target_features.npz", target_n)):
        feature_path, digest = verified_path(path, filename, manifest["hashes"][filename])
        refs.append(_artifact(feature_path, "hashing_features", digest))
        matrix = sparse.load_npz(feature_path).tocsr()
        if matrix.shape[0] != n or not np.isfinite(matrix.data).all():
            raise ValueError("Hashing features do not match full input row positions")
        matrices.append(matrix)
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError("Hashing source and target dimensions differ")
    return matrices[0], matrices[1], refs


def _implementation_artifacts() -> list[dict]:
    files = [Path(__file__), WORKSPACE / "src/vultriage/neural_audit.py",
             WORKSPACE / "src/vultriage/conformal.py", WORKSPACE / "src/vultriage/domain.py",
             WORKSPACE / "src/vultriage/metrics.py", WORKSPACE / "src/vultriage/data.py",
             WORKSPACE / "src/analyze_neurocomputing_controls.py",
             WORKSPACE / "src/analyze_neurocomputing_revision.py",
             WORKSPACE / "src/independent_model_v4_1.py", WORKSPACE / "src/prepare_independent_model_v4_1.py",
             WORKSPACE / "src/encode_independent_model_v4_1.py", WORKSPACE / "src/train_independent_model_v4_1.py",
             WORKSPACE / "src/predict_independent_model_v4_1.py", WORKSPACE / "src/audit_independent_model_v4.py",
             WORKSPACE / "tests/test_independent_model_v4_1.py", WORKSPACE / "protocol/independent_model_v4_1.md"]
    return [_artifact(p, "audit_implementation") for p in files]


def _final_training_failures(path: Path | None, hashes: dict, planned: dict,
                             present: set[tuple[str, int]]) -> tuple[list[dict], list[dict]]:
    """Bind final failure receipts; pending/missing predictions are never failures."""
    if path is None:
        return [], []
    _forbid_vault(path)
    receipt = _json(path)
    if (str(receipt.get("config_sha256", "")).upper() != hashes["config_sha256"]
            or str(receipt.get("source_rows_sha256", "")).upper() != hashes["source_rows_sha256"]
            or receipt.get("target_labels_accessed") is not False):
        raise ValueError("Final failure receipts must bind this config/source and no target-label access")
    failures, identities = receipt.get("failures"), set()
    if not isinstance(failures, list):
        raise ValueError("Final failure receipts need an explicit failures list")
    for row in failures:
        if not isinstance(row, dict):
            raise ValueError("Malformed final training-failure receipt")
        detector, seed = row.get("detector"), row.get("seed")
        identity = detector, seed
        if (detector not in planned or not isinstance(seed, int) or isinstance(seed, bool)
                or seed not in planned[detector] or identity in present or identity in identities
                or row.get("status") != "failed" or not isinstance(row.get("error"), str)
                or not row["error"].strip() or not isinstance(row.get("stage"), str) or not row["stage"].strip()):
            raise ValueError("Failure must identify one absent planned fit with final failed status, stage and error")
        identities.add(identity)
    return failures, [_artifact(path, "final_training_failure_receipts")]


def seal(*, config: Path, inputs: Path, predictions: list[Path], output: Path,
         source_rows: Path | None = None, dataset_config: Path | None = None,
         feature_cache: Path | None = None, training_failures: Path | None = None,
         workspace: Path = WORKSPACE) -> dict:
    """Construct policies and bind every dependency without opening target labels."""
    workspace, inputs = workspace.resolve(), inputs.resolve()
    config_path, source_path, target_path = config.resolve(), (source_rows or inputs / "source_rows.jsonl").resolve(), inputs / "target_rows.jsonl"
    prep_path = inputs / "preparation_manifest.json"
    cfg, prep = _json(config_path), _json(prep_path)
    config_hash = sha256(config_path)
    if not str(cfg.get("protocol_version", "")).startswith("trace-independent-model-v4"):
        raise ValueError("Expected the dated independent-family amendment v4 protocol")
    initial_artifacts = [_artifact(config_path, "config"), _artifact(prep_path, "preparation_manifest"),
                         _artifact(source_path, "source_rows"), _artifact(target_path, "target_rows")]
    freeze_path = workspace / cfg["paths"]["output_root"] / "amendment_freeze.json"
    freeze = _json(freeze_path)
    if (freeze["config_sha256"] != config_hash or freeze["source_rows_sha256"] != initial_artifacts[2]["sha256"]
            or sha256(freeze_path) != freeze_path.with_suffix(".sha256").read_text().strip()):
        raise ValueError("The original v4 amendment freeze changed")
    initial_artifacts.extend([_artifact(freeze_path, "amendment_freeze"),
                              _artifact(workspace / "protocol/independent_model_v4_1.md", "dated_protocol", freeze["protocol_sha256"])])
    audit_root = workspace / cfg["paths"]["output_root"] / "metadata-audit"
    audit_path = audit_root / "metadata_audit.json"
    metadata_audit = _json(audit_path)
    if (metadata_audit.get("config_sha256") != config_hash
            or metadata_audit.get("source_rows_sha256") != initial_artifacts[2]["sha256"]
            or metadata_audit.get("target_vault_opened") is not False
            or metadata_audit.get("target_outcomes_compared") is not False
            or sha256(audit_path) != audit_path.with_suffix(".sha256").read_text().strip()):
        raise ValueError("The metadata-only independence/temporal audit is inconsistent")
    initial_artifacts.append(_artifact(audit_path, "independence_metadata_audit"))
    for filename, digest in metadata_audit["report_hashes"].items():
        initial_artifacts.append(_artifact(audit_root / filename, "metadata_audit_roster", digest))
    for record in metadata_audit["raw_dependencies"]:
        initial_artifacts.append(_artifact(Path(record["path"]), "raw_metadata_provenance", record["sha256"]))
    focused = workspace / cfg["paths"]["output_root"] / "focused_family_event_findings.json"
    initial_artifacts.append(_artifact(focused, "focused_observed_family_event_findings",
                                       focused.with_suffix(".sha256").read_text().strip()))
    ds_hash = None
    if dataset_config is not None:
        if not str(_json(dataset_config).get("protocol_version", "")).startswith("trace-second-source-validation-v3"):
            raise ValueError("Expected the separate frozen second-source v3 dataset protocol")
        ds_hash = sha256(dataset_config)
        initial_artifacts.append(_artifact(dataset_config, "dataset_config"))
    output_hashes = prep.get("output_hashes", {})
    for filename, actual in (("source_rows.jsonl", initial_artifacts[2]["sha256"]),
                             ("target_rows.jsonl", initial_artifacts[3]["sha256"])):
        if filename in output_hashes and str(output_hashes[filename]).upper() != actual:
            raise ValueError(f"Prepared input hash mismatch: {filename}")
    vault_hash = output_hashes.get("target_label_vault.npz")
    if not isinstance(vault_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", vault_hash):
        raise ValueError("Preparation manifest must bind the unopened target-label vault")
    source, target = _rows(source_path, source=True), _rows(target_path, source=False)
    _verify_role_boundaries(source, target)
    projects = sorted({r["project_group"] for r in target})
    if "selected_projects" in prep and set(prep["selected_projects"]) != set(projects):
        raise ValueError("Prepared selected projects do not match target metadata")
    datasets = {r.get("dataset") for r in target}
    if len(datasets) != 1 or next(iter(datasets)) not in ("diversevul", "megavul"):
        raise ValueError("Each seal must cover exactly one declared dataset")
    dataset = next(iter(datasets))
    if projects != cfg["target_rosters"][dataset]["projects"] or len(target) != cfg["target_rosters"][dataset]["rows"]:
        raise ValueError("The independent-family seal must preserve every inherited target project and function")
    if (dataset == "megavul") != (dataset_config is not None):
        raise ValueError("MegaVul requires its separate frozen dataset configuration")
    planned = {"finetuned_codet5_small": list(cfg["fine_tuned_codet5"]["seeds"])}
    bindings = {"config_sha256": config_hash, "dataset_config_sha256": ds_hash,
                "source_rows_sha256": initial_artifacts[2]["sha256"],
                "target_rows_sha256": initial_artifacts[3]["sha256"]}
    loaded, cache, identities, checkpoint_hashes, artifacts = [], {}, set(), set(), list(initial_artifacts)
    for path in _prediction_paths(predictions):
        metadata, arrays, refs = _load_prediction(
            path, workspace=workspace, source=source, target=target, dataset=dataset,
            hashes=bindings, planned_seeds=planned, artifact_cache=cache)
        identity = metadata["detector"], int(metadata["seed"])
        if identity in identities:
            raise ValueError("Duplicate actual detector fit; prediction copies are not seeds")
        if metadata["checkpoint_sha256"] in checkpoint_hashes:
            raise ValueError("Copied checkpoints cannot be counted as independent seed fits")
        checkpoint_hashes.add(metadata["checkpoint_sha256"])
        identities.add(identity)
        loaded.append((metadata, arrays))
        artifacts.extend(refs)
    final_failures, refs = _final_training_failures(training_failures, bindings, planned, identities)
    artifacts.extend(refs)
    hashing_source = hashing_target = None
    if any(detector == "hashing_sgd" for detector, _ in identities):
        bound_caches = {Path(m["verified_feature_cache_path"]) for m, _ in loaded if m["detector"] == "hashing_sgd"}
        if len(bound_caches) != 1:
            raise ValueError("Hashing fits bind different feature caches")
        bound_cache = next(iter(bound_caches))
        if feature_cache is not None and feature_cache.resolve() != bound_cache:
            raise ValueError("--feature-cache differs from the prediction-bound cache")
        feature_cache = bound_cache
        hashing_source, hashing_target, refs = _hashing_cache(feature_cache, len(source), len(target), bindings)
        artifacts.extend(refs)
    artifacts.extend(_implementation_artifacts())
    output = _new_directory(output)
    entries, failures, label_free_counts, availability_rows = [], [], [], []
    project_positions = {project: np.asarray([i for i, r in enumerate(target) if r["project_group"] == project], dtype=np.int64)
                         for project in projects}
    for metadata, arrays in loaded:
        detector, seed = metadata["detector"], int(metadata["seed"])
        s_positions = arrays["source_positions"]
        cal_indices = np.asarray([i for i, position in enumerate(s_positions) if source[int(position)]["role"] == "calibration"], dtype=np.int64)
        cal_positions = s_positions[cal_indices]
        cal_rows = [source[int(i)] for i in cal_positions]
        cal_labels = np.asarray([r["label"] for r in cal_rows], dtype=np.int64)
        cal_probability = arrays["source_probability"][cal_indices]
        if set(np.unique(cal_labels)) != {0, 1}:
            raise ValueError("Both classes are required in the source calibration role")
        cal_features = (hashing_source[cal_positions] if detector == "hashing_sgd" else arrays["source_features"][cal_indices])
        for project, positions in project_positions.items():
            print(f"seal {dataset} {detector} seed={seed} project={project}", flush=True)
            probability = arrays["target_probability"][positions]
            target_features = hashing_target[positions] if detector == "hashing_sgd" else arrays["target_features"][positions]
            # Exactly one source-label isotonic fit per project/actual seed, reused for every cost and budget.
            isotonic = fit_source_isotonic(cal_probability, cal_labels)
            eta = isotonic.predict(probability)
            project_rows = [target[int(i)] for i in positions]
            groups = [(r["project_group"], r["commit_id"]) for r in cal_rows + project_rows]
            features = (sparse.vstack([cal_features, target_features], format="csr") if sparse.issparse(cal_features)
                        else np.concatenate([cal_features, target_features], axis=0))
            domain = None
            domain_error = None
            try:
                domain = crossfit_grouped_density_ratio(
                    features, ["source:" + r["row_id"] for r in cal_rows] + ["target:" + r["row_id"] for r in project_rows],
                    np.asarray([0] * len(cal_rows) + [1] * len(project_rows)), groups, seed=seed)
            except DomainFitUnavailable as error:
                domain_error = str(error)
                failures.append({"dataset": dataset, "detector": detector, "seed": seed,
                                 "project": project, "stage": "grouped_domain_fit", "error": domain_error,
                                 "action": "review fallback for weighted-dependent policies; no inferential exclusion"})
            availability_rows.append({"dataset": dataset, "detector": detector, "seed": seed,
                                      "project": project, "domain_available": domain is not None,
                                      "error": domain_error})
            raw_cal = domain.raw_ratios[:len(cal_rows)] if domain is not None else None
            raw_target = domain.raw_ratios[len(cal_rows):] if domain is not None else None
            decisions, specs = construct_decisions(cal_probability, cal_labels, probability, eta, raw_cal, raw_target, cfg)
            folder = output / "decisions" / detector / f"seed-{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            basename = re.sub(r"[^a-zA-Z0-9_-]", "_", project) + "-" + hashlib.sha256(project.encode()).hexdigest()[:8]
            archive_path, meta_path = folder / f"{basename}.npz", folder / f"{basename}.json"
            archive_arrays = {"target_positions": positions, "target_probability": probability.astype(np.float32),
                              "source_eta": eta.astype(np.float64), "calibration_positions": cal_positions,
                              "isotonic_x": isotonic.X_thresholds_, "isotonic_y": isotonic.y_thresholds_, **decisions}
            if domain is not None:
                archive_arrays.update(domain_fold_ids=domain.fold_ids, domain_probability=domain.target_probabilities,
                                      raw_calibration_weights=raw_cal, raw_target_weights=raw_target)
            with archive_path.open("xb") as stream:
                np.savez_compressed(stream, **archive_arrays)
            clip_diagnostics = []
            if domain is not None:
                for clip in cfg["weighting"]["clips"]:
                    weights = np.clip(raw_cal, 1 / clip, clip)
                    clip_diagnostics.append({"clip": clip, "source": clipped_weight_summary(raw_cal, clip),
                                             "target": clipped_weight_summary(raw_target, clip),
                                             "class_ess": {str(label): kish_ess(weights[cal_labels == label]) for label in (0, 1)}})
            cell_meta = {"dataset": dataset, "detector": detector, "seed": seed, "project": project,
                         "target_rows": len(positions), "calibration_rows": len(cal_rows),
                         "calibration_class_counts": {str(label): int(np.sum(cal_labels == label)) for label in (0, 1)},
                         "domain_status": "ok" if domain is not None else "failed",
                         "domain_error": domain_error, "domain_diagnostics": domain.diagnostics if domain is not None else None,
                         "clip_diagnostics": clip_diagnostics, "decision_specs": specs,
                         "target_labels_accessed": False, "new_gate_fitted": False,
                         "tie_rules": {"msp": "argmax p=0.5 chooses safe; stable target position rank",
                                       "tails": "one stable ascending order; disjoint low/high tails",
                                       "cost": "automatic label cost tie chooses vulnerable; stable rank",
                                       "free_volume": "automatic cost strictly below review cost"}}
            _write_json(meta_path, cell_meta)
            entries.append({"project": project, "detector": detector, "seed": seed,
                            "archive": str(archive_path.relative_to(output)), "archive_sha256": sha256(archive_path),
                            "metadata": str(meta_path.relative_to(output)), "metadata_sha256": sha256(meta_path)})
            for spec in specs:
                codes = action_codes(decisions[spec["key"]])
                label_free_counts.append({"dataset": dataset, "detector": detector, "seed": seed,
                                          "project": project, **{k: v for k, v in spec.items() if k != "key"},
                                          "n": len(codes), "auto_safe": int(np.sum(codes == 0)),
                                          "auto_vulnerable": int(np.sum(codes == 1)),
                                          "review_count": int(np.sum(codes == 2))})
    pd.DataFrame(label_free_counts).to_csv(output / "label_free_decision_counts.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(output / "domain_fit_availability.csv", index=False)
    _write_json(output / "failures.json", {"domain_failures": failures, "final_training_failures": final_failures})
    actual = {detector: sorted(seed for d, seed in identities if d == detector) for detector in DETECTORS}
    actual = {detector: seeds for detector, seeds in actual.items() if seeds}
    availability = pd.DataFrame(availability_rows)
    supported_projects = {detector: [project for project in projects
                                    if availability[(availability["detector"] == detector)
                                                    & (availability["project"] == project)]["domain_available"].all()]
                          for detector in actual}
    missing = [{"detector": d, "seed": seed, "status": "missing_prediction_not_counted_as_fit"}
               for d, seeds in planned.items() for seed in seeds if (d, seed) not in identities]
    failed_identities = {(row["detector"], row["seed"]) for row in final_failures}
    for row in missing:
        if (row["detector"], row["seed"]) in failed_identities:
            row["status"] = "final_failed_fit_not_counted_as_prediction"
    accounted = all((row["detector"], row["seed"]) in failed_identities for row in missing)
    unique_artifacts = {r["path"]: r for r in artifacts}
    for record in unique_artifacts.values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Dependency changed while decisions were being sealed")
    manifest = {"schema_version": "trace-independent-model-v4.1-decision-seal-1", "protocol_version": cfg["protocol_version"],
                "dataset": dataset, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_labels_accessed": False, "new_gate_fitted": False,
                "known_original_and_v3_outcomes": True, "new_unseen_outcomes_claim": False,
                "dated_amendment_freeze_sha256": sha256(freeze_path),
                "independence_metadata_audit_sha256": sha256(audit_path),
                "strict_temporal_subset_available": metadata_audit["strict_temporal_subsets"][dataset]["available"],
                "event_independence_certified": False,
                "source_rows_path": str(source_path), "target_rows_path": str(target_path),
                "config_path": str(config_path), "dataset_config_path": str(dataset_config.resolve()) if dataset_config else None,
                "prepared_target_label_vault_sha256": vault_hash.upper(),
                "projects": projects, "target_rows": len(target), "source_rows": len(source),
                "planned_seeds": planned, "actual_seeds": actual, "missing_prediction_pairs": missing,
                "all_planned_fits_present": not missing, "domain_failure_cells": len(failures),
                "all_planned_attempts_accounted": accounted, "final_training_failures": final_failures,
                "domain_supported_projects": supported_projects,
                "domain_unsupported_projects": {d: [p for p in projects if p not in supported_projects[d]] for d in actual},
                "availability_rule": "all actual fitted seeds have a successful label-free grouped domain fit; fixed before target-label access",
                "domain_protocol": {"folds": DOMAIN_FOLDS,
                                    "grouping": "project/commit identities; shared global 40-hex SHA coalesced",
                                    "salt": DOMAIN_SALT,
                                    "alpha": DOMAIN_ALPHA, "epochs": DOMAIN_EPOCHS, "balanced_domains": True},
                "dependencies": sorted(unique_artifacts.values(), key=lambda r: r["path"]),
                "decisions": entries,
                "reports": {name: sha256(output / name) for name in ("label_free_decision_counts.csv", "domain_fit_availability.csv", "failures.json")},
                "runtime": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__},
                "interpretation": "Dataset-specific project audit; actual seeds are technical fits, not independent datasets."}
    manifest_path = output / "decision_seal.json"
    _write_json(manifest_path, manifest)
    (output / "decision_seal.sha256").write_text(sha256(manifest_path) + "\n", encoding="ascii")
    return manifest


def verify_seal(seal_path: Path) -> dict:
    """Verify all input, evidence, implementation and decision bytes before label access."""
    seal_path = seal_path.resolve()
    root = seal_path.parent
    expected = (root / "decision_seal.sha256").read_text(encoding="ascii").strip().upper()
    if sha256(seal_path) != expected:
        raise ValueError("Decision seal manifest hash mismatch")
    manifest = _json(seal_path)
    if (manifest.get("schema_version") != "trace-independent-model-v4.1-decision-seal-1"
            or manifest.get("target_labels_accessed") is not False or manifest.get("new_gate_fitted") is not False):
        raise ValueError("Invalid decision seal boundary")
    for record in manifest["dependencies"]:
        _artifact(Path(record["path"]), record["role"], record["sha256"])
    identities, availability = set(), {}
    projects = set(manifest["projects"])
    counts = {}
    for entry in manifest["decisions"]:
        key = entry["detector"], entry["seed"], entry["project"]
        if key in identities:
            raise ValueError("Duplicate sealed detector/seed/project cell")
        identities.add(key)
        counts.setdefault((entry["detector"], entry["seed"]), set()).add(entry["project"])
        verified_path(root, entry["archive"], entry["archive_sha256"])
        metadata_path, _ = verified_path(root, entry["metadata"], entry["metadata_sha256"])
        metadata = _json(metadata_path)
        if (metadata.get("project") != entry["project"] or metadata.get("seed") != entry["seed"]
                or metadata.get("detector") != entry["detector"] or metadata.get("dataset") != manifest["dataset"]
                or metadata.get("target_labels_accessed") is not False or metadata.get("new_gate_fitted") is not False
                or metadata.get("domain_status") not in ("ok", "failed")):
            raise ValueError("Sealed project metadata identity/availability is inconsistent")
        availability[key] = metadata["domain_status"] == "ok"
    expected_pairs = {(d, seed) for d, seeds in manifest["actual_seeds"].items() for seed in seeds}
    if set(counts) != expected_pairs or any(p != projects for p in counts.values()):
        raise ValueError("Incomplete sealed project/actual-seed grid")
    for detector, seeds in manifest["actual_seeds"].items():
        supported = {p for p in projects if all(availability[detector, seed, p] for seed in seeds)}
        if (set(manifest["domain_supported_projects"][detector]) != supported
                or set(manifest["domain_unsupported_projects"][detector]) != projects - supported):
            raise ValueError("Sealed supported-project roster differs from label-free fit availability")
    for name, digest in manifest["reports"].items():
        verified_path(root, name, digest)
    return manifest


def _load_target_vault(path: Path, manifest: dict, rows: list[dict]) -> np.ndarray:
    """Called only after verify_seal has returned successfully."""
    if sha256(path) != manifest["prepared_target_label_vault_sha256"]:
        raise ValueError("Target-label vault differs from the pre-prediction preparation seal")
    with np.load(path, allow_pickle=False) as archive:
        if not {"labels", "positions", "row_ids"}.issubset(archive.files):
            raise ValueError("Target-label vault schema is incomplete")
        positions, order = _positions(archive["positions"], np.arange(len(rows)), "vault positions")
        labels, ids = archive["labels"], archive["row_ids"]
        if labels.shape != positions.shape or ids.shape != positions.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Invalid aligned target-label vault")
        labels, ids = labels[order].astype(np.int64), ids[order].astype(str)
    if not np.array_equal(ids, np.asarray([r["row_id"] for r in rows], dtype=str)):
        raise ValueError("Target-label vault row IDs do not match the sealed metadata positions")
    return labels


def _paired_tables(averaged: pd.DataFrame, cfg: dict, manifest: dict) -> pd.DataFrame:
    outcomes = ("auto_fn_rate", "auto_fp_rate", "coverage", "selective_error",
                "vulnerable_miscoverage", "safe_miscoverage", "maximum_relative_violation", "cost_10", "cost_50")
    baselines = [m for m in METHODS if m != "estimated_weight_no_gate"]
    rows, bootstrap_cache = [], {}
    seed = int(cfg["statistics"]["bootstrap_seed"])
    for key, cell in averaged.groupby(["dataset", "detector", "alpha_v", "alpha_s", "clip"], sort=True):
        supported = manifest["domain_supported_projects"][key[1]]
        for cohort in ("all_projects_deployment", "domain_supported_secondary"):
            subset = cell if cohort == "all_projects_deployment" else cell[cell["project"].isin(supported)]
            projects = tuple(sorted(subset["project"].unique()))
            if projects and projects not in bootstrap_cache:
                bootstrap_cache[projects] = np.random.default_rng(seed).integers(0, len(projects), size=(10000, len(projects)))
            for baseline in baselines:
                for outcome in outcomes:
                    summary = paired_project_summary(
                        subset, outcome, baseline, "estimated_weight_no_gate", bootstrap_seed=seed,
                        draws=10000, bootstrap_indices=bootstrap_cache.get(projects),
                        include_review_fallback=cohort == "all_projects_deployment",
                        all_project_count=len(manifest["projects"]), domain_supported_projects=supported)
                    summary.update(dict(zip(["dataset", "detector", "alpha_v", "alpha_s", "clip"], key)))
                    summary["cohort"] = cohort
                    summary["primary"] = (cohort == "all_projects_deployment" and baseline == "class_matched_weighted"
                                          and outcome == "auto_fn_rate" and key[2:4] == tuple(cfg["risk_budgets"]["primary"])
                                          and key[4] == cfg["weighting"]["primary_clip"])
                    rows.append(summary)
    return pd.DataFrame(rows)


def evaluate(*, seal_path: Path, target_label_vault: Path, output: Path) -> dict:
    # Do not move any vault hash/open/load operation above this full verification.
    manifest = verify_seal(seal_path)
    if manifest.get("all_planned_attempts_accounted") is not True:
        raise ValueError("Planned fits remain unresolved: do not open target labels before all predictions or final failure receipts are sealed")
    cfg = _json(Path(manifest["config_path"]))
    joint_path = WORKSPACE / cfg["paths"]["output_root"] / "joint_dataset_decision_seal.json"
    joint = _json(joint_path)
    if (sha256(joint_path) != joint_path.with_suffix(".sha256").read_text().strip()
            or joint.get("schema_version") != "trace-independent-model-v4.1-joint-dataset-seal-1"
            or joint.get("config_sha256") != sha256(Path(manifest["config_path"]))
            or joint.get("target_label_vaults_opened_for_this_seal") is not False
            or joint.get("all_complete_target_rosters_sealed_before_evaluation") is not True
            or set(joint.get("datasets", {})) != {"diversevul", "megavul"}):
        raise ValueError("Both complete dataset decision seals must be bound before either target-vault evaluation")
    for dataset, record in joint["datasets"].items():
        sibling = WORKSPACE / record["path"]
        if sha256(sibling) != record["sha256"] or record["projects"] != cfg["target_rosters"][dataset]["projects"]:
            raise ValueError("A complete dataset seal changed after the joint pre-evaluation boundary")
        if sibling.resolve() != seal_path.resolve():
            verify_seal(sibling)
    if sha256(WORKSPACE / "src/seal_joint_independent_model_v4_1.py") != joint["script_sha256"]:
        raise ValueError("The joint-seal implementation changed before evaluation")
    if joint["datasets"][manifest["dataset"]]["sha256"] != sha256(seal_path):
        raise ValueError("This decision seal is absent from the frozen joint dataset receipt")
    target = _rows(Path(manifest["target_rows_path"]), source=False)
    labels = _load_target_vault(target_label_vault, manifest, target)
    output = _new_directory(output)
    root = seal_path.resolve().parent
    metrics, discrimination, project_counts = [], [], []
    for project in manifest["projects"]:
        positions = np.asarray([i for i, r in enumerate(target) if r["project_group"] == project], dtype=np.int64)
        project_counts.append({"dataset": manifest["dataset"], "project": project,
                               **project_commit_counts([target[int(i)] for i in positions], labels[positions])})
    for entry in manifest["decisions"]:
        meta = _json(root / entry["metadata"])
        with np.load(root / entry["archive"], allow_pickle=False) as archive:
            positions = archive["target_positions"]
            if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
                raise ValueError("Sealed decision positions are malformed")
            expected_positions = np.asarray([i for i, r in enumerate(target) if r["project_group"] == entry["project"]])
            if not np.array_equal(positions, expected_positions):
                raise ValueError("Sealed project decision positions do not match target metadata")
            y = labels[positions]
            probability, eta = probability_vector(archive["target_probability"]), probability_vector(archive["source_eta"])
            identity = {"dataset": manifest["dataset"], "detector": entry["detector"],
                        "seed": entry["seed"], "project": entry["project"]}
            discrimination.append({**identity, **detector_metrics(y, probability, eta)})
            for spec in meta["decision_specs"]:
                sets = prediction_sets(archive[spec["key"]], len(positions))
                values = project_metrics(y, probability, sets, spec["alpha_v"], spec["alpha_s"], cfg)
                if spec["status"] != "ok":
                    # A technical refusal is an observable deployment action,
                    # but its manufactured doubleton is not a fitted weighted
                    # prediction set and cannot earn zero budget violation.
                    for field in ("vulnerable_miscoverage", "safe_miscoverage", "maximum_relative_violation",
                                  "vulnerable_miscoverage_count", "safe_miscoverage_count"):
                        values[field] = np.nan
                metrics.append({**identity, **{k: v for k, v in spec.items() if k != "key"}, **values})
    frame = pd.DataFrame(metrics)
    averaged = average_actual_seed_metrics(frame, manifest["actual_seeds"])
    comparisons = _paired_tables(averaged, cfg, manifest)
    frame.to_csv(output / "project_seed_metrics.csv", index=False)
    averaged.to_csv(output / "project_seed_averages.csv", index=False)
    comparisons.to_csv(output / "paired_project_summaries.csv", index=False)
    comparisons[comparisons["cohort"] == "domain_supported_secondary"].to_csv(
        output / "domain_supported_paired_summaries.csv", index=False)
    comparisons[comparisons["primary"]].to_csv(output / "primary_paired_summary.csv", index=False)
    pd.DataFrame(project_counts).to_csv(output / "project_counts_positive_commits.csv", index=False)
    detector_frame = pd.DataFrame(discrimination)
    detector_frame.to_csv(output / "detector_project_seed_metrics.csv", index=False)
    detector_frame.groupby(["dataset", "detector", "project"], as_index=False).mean(numeric_only=True).drop(columns="seed").to_csv(
        output / "detector_project_seed_averages.csv", index=False)
    summary_keys = ["dataset", "detector", "alpha_v", "alpha_s", "clip", "method"]
    summary_groups = averaged.groupby(summary_keys, sort=True)
    medians = summary_groups.median(numeric_only=True)
    medians["projects"] = summary_groups.size()
    medians["policy_ok_projects"] = summary_groups["status"].agg(lambda values: int(values.eq("ok").sum()))
    medians["policy_failure_projects"] = medians["projects"] - medians["policy_ok_projects"]
    for column in averaged.select_dtypes(include=[np.number]).columns:
        if column not in summary_keys:
            medians[f"{column}_defined_projects"] = summary_groups[column].count()
    medians.reset_index().to_csv(output / "project_medians.csv", index=False)
    reports = sorted(p for p in output.iterdir() if p.is_file())
    result = {"schema_version": "trace-independent-model-v4.1-evaluation-1", "dataset": manifest["dataset"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "decision_seal_sha256": sha256(seal_path), "verified_before_target_label_access": True,
              "joint_dataset_decision_seal_sha256": sha256(joint_path),
              "target_label_vault_sha256": sha256(target_label_vault), "target_rows": len(target),
              "target_vulnerable": int(labels.sum()), "projects": len(project_counts),
              "actual_seeds": manifest["actual_seeds"], "all_planned_fits_present": manifest["all_planned_fits_present"],
              "all_planned_attempts_accounted": manifest["all_planned_attempts_accounted"],
              "missing_prediction_pairs": manifest["missing_prediction_pairs"],
              "final_training_failures": manifest["final_training_failures"],
              "domain_failure_cells": manifest["domain_failure_cells"],
              "domain_supported_projects": manifest["domain_supported_projects"],
              "domain_unsupported_projects": manifest["domain_unsupported_projects"],
              "rows": {"project_seed_metrics": len(frame), "project_seed_averages": len(averaged),
                       "paired_project_summaries": len(comparisons)},
              "report_hashes": {p.name: sha256(p) for p in reports},
              "statistics": {"unit": "equally weighted project after averaging actual fits",
                             "bootstrap_draws": 10000, "bootstrap_seed": cfg["statistics"]["bootstrap_seed"],
                             "interval": "pointwise descriptive percentile; no equivalence or population guarantee",
                             "availability_strata": "all-project deployment includes sealed review fallback; fixed supported-only secondary cohort",
                             "undefined_pairs": "retain full eligible count and report undefined summary; never drop target-outcome-selected projects",
                             "raw_set_failure": "unavailable weighted set miscoverage/violation is undefined; review fallback zero errors are operational, not predictive evidence"}}
    _write_json(output / "evaluation_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sealing = commands.add_parser("seal", help="construct and seal decisions without target labels")
    sealing.add_argument("--config", type=Path, required=True)
    sealing.add_argument("--inputs", type=Path, required=True)
    sealing.add_argument("--source-rows", type=Path)
    sealing.add_argument("--dataset-config", type=Path)
    sealing.add_argument("--predictions", type=Path, action="append", required=True)
    sealing.add_argument("--feature-cache", type=Path)
    sealing.add_argument("--training-failures", type=Path,
                         help="final failed fit receipts, bound to config/source SHA; pending fits cannot open the vault")
    sealing.add_argument("--output", type=Path, required=True)
    sealing.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    evaluation = commands.add_parser("evaluate", help="verify seal, then join the target-label vault")
    evaluation.add_argument("--seal", type=Path, required=True)
    evaluation.add_argument("--target-label-vault", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "seal":
        result = seal(config=args.config, inputs=args.inputs, source_rows=args.source_rows,
                      dataset_config=args.dataset_config, predictions=args.predictions,
                      feature_cache=args.feature_cache, training_failures=args.training_failures,
                      output=args.output, workspace=args.workspace_root)
        print(json.dumps({"dataset": result["dataset"], "sealed_project_seed_cells": len(result["decisions"]),
                          "actual_seeds": result["actual_seeds"], "domain_failure_cells": result["domain_failure_cells"],
                          "all_planned_fits_present": result["all_planned_fits_present"]}), flush=True)
    else:
        result = evaluate(seal_path=args.seal, target_label_vault=args.target_label_vault, output=args.output)
        print(json.dumps({"dataset": result["dataset"], "projects": result["projects"],
                          "actual_seeds": result["actual_seeds"], "rows": result["rows"]}), flush=True)


if __name__ == "__main__":
    main()
