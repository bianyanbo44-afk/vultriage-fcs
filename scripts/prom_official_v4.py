"""Execute pinned official PROM components on unchanged sealed TRACE v3 inputs.

seal has no target-label argument. evaluate verifies every dependency and action
archive before opening the existing separate target-label vaults. This is an
explicit binary/component adaptation, not the full official PROM system.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom"
OUT = ROOT / "outputs/kbs-validation-v4/prom"
DEPS = WORK / "runtime-dependencies"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from prom_official_v4_setup import bootstrap, sha, utc, write_json
from analyze_neurocomputing_controls import action_codes, class_side_matched, global_msp_matched, source_cost_matched

CONFIG = ROOT / "configs/prom_official_v4.json"
PROTOCOL = ROOT / "protocol/prom_official_v4.md"
CLEARANCE = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom_protocol_clearance.json"
EXECUTION_CLEARANCE = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom_execution_optimization_clearance.json"
FOLDER_MAP = {"finetuned_codebert": "finetuned", "frozen_codebert": "frozen", "hashing_sgd": "hashing"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def path_record(path: Path, role: str, expected: str | None = None) -> dict:
    path = path.resolve()
    actual = sha(path)
    if expected is not None and actual != expected.upper():
        raise ValueError(f"Changed {role}: {path}")
    try:
        name = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        name = str(path)
    return {"path": name, "role": role, "sha256": actual, "bytes": path.stat().st_size}


def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def array_digest(values: np.ndarray) -> str:
    a = np.ascontiguousarray(values)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest().upper()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Empty aggregate table")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for k, v in row.items()})


def checked_protocol() -> dict:
    cfg, receipt = load_json(CONFIG), load_json(CLEARANCE)
    if receipt.get("status") != "CLEARED":
        raise ValueError("Root protocol receipt does not clear this comparison")
    for name, expected in receipt["inputs_sha256"].items():
        path_record(ROOT / name, "approved protocol/config", expected)
    smoke = load_json(OUT / "fidelity_smoke.json")
    if (smoke["status"] != "passed" or not smoke["binary_aps_independent_mathematical_parity"]
            or not all(all(v.values()) for v in smoke["native_multiclass_exact_parity"].values())
            or smoke["target_labels_accessed"] is not False):
        raise ValueError("Native/binary exact parity gate is incomplete")
    execution = load_json(OUT / "execution_optimization.json")
    execution_receipt = load_json(EXECUTION_CLEARANCE)
    if execution_receipt.get("status") != "CLEARED" or execution["status"] != "passed" or execution["target_labels_accessed"]:
        raise ValueError("Exact execution-optimization gate is incomplete")
    for name, expected in execution_receipt["inputs_sha256"].items():
        path_record(ROOT / name, "approved execution amendment", expected)
    if (not all(c["exact_array_equal"] for c in execution["pooled_count_parity_cases"])
            or not all(all(all(values.values()) for values in row["exact_parity"].values()) for row in execution["native_and_binary_full_parity"].values())
            or not all(execution["nonfinite_probability_rejection"].values())):
        raise ValueError("Exact optimization parity or finite-input gate failed")
    repair = load_json(OUT / "binary_repair.json")
    path_record(WORK / "official-source/src/prom/prom_classification.py", "unchanged official source", repair["original_sha256"])
    path_record(WORK / "literal-binary-runtime/src/prom/prom_classification.py", "literal binary runtime", repair["runtime_sha256"])
    path_record(WORK / "binary-runtime/src/prom/prom_classification.py", "exact pooled-count runtime", execution["optimized_binary_runtime_sha256"])
    return cfg


class SealedProbabilityEstimator:
    """A no-fit bridge exposing the sealed detector's two probabilities.

    These probabilities also define the declared neighborhood feature; this
    bridge is not a trained replacement model and never accesses a row label.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 2

    def fit(self, *_args, **_kwargs):
        raise RuntimeError("Sealed detector predictions cannot be refitted")

    def predict_proba(self, features):
        a = np.asarray(features, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
            raise ValueError("Invalid sealed probability bridge features")
        if not np.allclose(a.sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Binary probabilities do not sum to one")
        return a

    def predict(self, features):
        return np.argmax(self.predict_proba(features), axis=1)


def features(probability):
    p = np.asarray(probability, dtype=np.float64)
    return np.column_stack((1.0 - p, p))


def actions(accepted, predicted):
    result = np.full(len(predicted), 2, dtype=np.uint8)
    result[np.asarray(accepted, dtype=bool)] = np.asarray(predicted, dtype=np.uint8)[accepted]
    return result


def accepted_by_alpha(outputs: dict, alphas: np.ndarray) -> dict:
    return {n: (v["sets"].sum(axis=1) == 1) & (v["pvalues"][:, None] > 1.0 - alphas[None, :]) for n, v in outputs.items()}


def f1_counts(rejected, wrong) -> dict:
    rejected, wrong = np.asarray(rejected, bool), np.asarray(wrong, bool)
    tp, fp = int(np.sum(rejected & wrong)), int(np.sum(rejected & ~wrong))
    fn, tn = int(np.sum(~rejected & wrong)), int(np.sum(~rejected & ~wrong))
    denominator = 2 * tp + fp + fn
    return {"error_detection_tp": tp, "error_detection_fp": fp, "error_detection_fn": fn, "error_detection_tn": tn,
            "error_detection_f1": 2 * tp / denominator if denominator else 0.0,
            "error_detection_precision": tp / (tp + fp) if tp + fp else np.nan,
            "error_detection_recall": tp / (tp + fn) if tp + fn else np.nan}


def choose_source(outputs: dict, source_labels, source_probability, cfg: dict, detector: str, seed: int):
    alpha = np.asarray(cfg["alpha_grid"], dtype=float)
    masks = accepted_by_alpha(outputs, alpha)
    masks["mixture"] = np.logical_and.reduce([masks[n] for n in cfg["primary_experts"]])
    wrong = (source_probability > 0.5).astype(int) != source_labels
    candidates, table = [], []
    primary_order = cfg["primary_experts"] + ["mixture"]
    for n in primary_order:
        for i, a in enumerate(alpha):
            values = f1_counts(~masks[n][:, i], wrong)
            row = {"detector": detector, "seed": seed, "family": n, "alpha": float(a), "source_rows": len(wrong),
                   "source_rejected": int(np.sum(~masks[n][:, i])), "source_review_fraction": float(np.mean(~masks[n][:, i])), **values}
            table.append(row)
            denominator = 2 * values["error_detection_tp"] + values["error_detection_fp"] + values["error_detection_fn"]
            exact_f1 = Fraction(2 * values["error_detection_tp"], denominator) if denominator else Fraction(0)
            key = (-exact_f1, row["source_rejected"], primary_order.index(n), float(a))
            candidates.append((key, n, i, row))
    _, name, index, best = min(candidates, key=lambda value: value[0])
    return {"family": name, "alpha": float(alpha[index]), "alpha_index": int(index), "source_result": best,
            "source_labels_only": True, "target_labels_used": False}, table


def read_source() -> list[dict]:
    rows = []
    with (ROOT / "outputs/neural-validation-v3/inputs/source_rows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append({k: row[k] for k in ("row_id", "position", "project_group", "commit_id", "role", "label")})
    for i, row in enumerate(rows):
        if int(row["position"]) != i or row["label"] not in (0, 1):
            raise ValueError("Source labels/positions do not match the existing sealed roles")
    commits = defaultdict(set)
    for row in rows:
        commits[row["commit_id"]].add(row["role"])
    if any(len(roles) > 1 for roles in commits.values()):
        raise ValueError("Source commit crosses roles")
    if {r: sum(x["role"] == r for x in rows) for r in ("calibration", "model_validation")} != {"calibration": 4400, "model_validation": 6296}:
        raise ValueError("Unexpected source role counts")
    return rows


def read_target(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if any(k in row for k in ("label", "target", "vulnerable", "vul", "is_vulnerable")):
                raise ValueError("Target metadata contains an outcome label")
            rows.append({k: row[k] for k in ("row_id", "position", "project_group", "commit_id")})
    if [int(r["position"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Target positions are not canonical")
    return rows


def verify_old_inputs(cfg: dict):
    # Existing official v3 verification also checks model/checkpoint/prediction
    # lineage and implementation bytes; it has no target-label read operation.
    from evaluate_neural_validation_v3 import verify_seal
    manifests, dependencies, target_rows = {}, {}, {}
    for dataset in cfg["datasets"]:
        seal_path = ROOT / cfg["old_decision_root"] / dataset / "decision_seal.json"
        print(f"verify existing v3 lineage: {dataset}", flush=True)
        manifest = verify_seal(seal_path)
        if manifest["actual_seeds"] != cfg["detectors"] or not manifest["all_planned_fits_present"]:
            raise ValueError("Existing fitted-seed grid is not the declared complete comparison")
        manifests[dataset] = manifest
        dependencies[str(seal_path)] = path_record(seal_path, "old decision seal")
        dependencies[str(seal_path.with_suffix(".sha256"))] = path_record(seal_path.with_suffix(".sha256"), "old seal checksum")
        for record in manifest["dependencies"]:
            path = resolve(record["path"])
            dependencies[str(path)] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "role": "old verified " + record["role"],
                                       "sha256": record["sha256"], "bytes": path.stat().st_size}
        target_path = ROOT / "outputs/neural-validation-v3/inputs" / ("megavul/target_rows.jsonl" if dataset == "megavul" else "target_rows.jsonl")
        rows = read_target(target_path)
        if sorted({r["project_group"] for r in rows}) != sorted(manifest["projects"]):
            raise ValueError("Target project roster changed")
        target_rows[dataset] = rows
    return manifests, dependencies, target_rows


def save_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def seal() -> dict:
    cfg = checked_protocol()
    final_path = OUT / "decision_seal.json"
    if final_path.exists():
        return verify_new_seal()
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    old, dependencies, target = verify_old_inputs(cfg)
    for path, role in ((CONFIG, "protocol config"), (PROTOCOL, "dated protocol"), (CLEARANCE, "root protocol clearance"),
                       (Path(__file__), "comparison implementation"), (ROOT / "scripts/prom_official_v4_setup.py", "setup implementation"),
                       (OUT / "provenance.json", "official provenance"), (OUT / "binary_repair.json", "binary repair receipt"),
                       (OUT / "fidelity_smoke.json", "native exact parity gate"), (OUT / "implementation_history.json", "implementation history"),
                       (OUT / "execution_optimization.json", "exact counted-operation parity"),
                       (ROOT / "protocol/prom_official_v4_execution_amendment.md", "dated execution amendment"),
                       (ROOT / "configs/prom_official_v4_execution.json", "fixed execution amendment config"),
                       (EXECUTION_CLEARANCE, "root execution amendment clearance"),
                       (WORK / "prom_official_v4_runtime_requirements.txt", "private numerical stack lock")):
        dependencies[str(path)] = path_record(path, role)
    for source in (WORK / "official-source", WORK / "literal-binary-runtime", WORK / "binary-runtime"):
        for path in sorted((source / "src/prom").rglob("*.py")):
            if "__pycache__" not in path.parts:
                dependencies[str(path)] = path_record(path, "actual official source" if source.name == "official-source" else "binary runtime source")
    for path in (WORK / "official-source/LICENSE", WORK / "official-source/src/LICENSE.txt", WORK / "prom_official_v4_binary_repair.diff",
                 WORK / "prom_official_v4_pooled_count_optimization.diff"):
        dependencies[str(path)] = path_record(path, "official license or declared diff")
    source = read_source()
    calibration_positions = np.asarray([i for i, row in enumerate(source) if row["role"] == "calibration"], dtype=np.int64)
    selection_positions = np.asarray([i for i, row in enumerate(source) if row["role"] == "model_validation"], dtype=np.int64)
    calibration_labels = np.asarray([source[int(i)]["label"] for i in calibration_positions])
    selection_labels = np.asarray([source[int(i)]["label"] for i in selection_positions])
    alpha = np.asarray(cfg["alpha_grid"], dtype=float)
    fixed_index = cfg["alpha_grid"].index(cfg["fixed_alpha"])
    MapieClassifier, _, version = bootstrap(WORK / "binary-runtime")
    estimator = SealedProbabilityEstimator()
    selections, source_table, cells, counts, timings = [], [], [], [], []
    log_path = OUT / "official_execution.log"
    with log_path.open("w", encoding="utf-8") as log:
        for detector, seeds in cfg["detectors"].items():
            for seed in seeds:
                print(f"source selection: {detector} seed={seed}", flush=True)
                fit_started = time.perf_counter()
                archives, source_probability, source_positions = {}, None, None
                for dataset in cfg["datasets"]:
                    p = ROOT / cfg["prediction_root"] / dataset / FOLDER_MAP[detector] / f"seed-{seed}" / "predictions.npz"
                    metadata = load_json(p.parent / "metadata.json")
                    if metadata["detector"] != detector or metadata["seed"] != seed or metadata["dataset"] != dataset or metadata["target_labels_accessed"]:
                        raise ValueError("Invalid sealed prediction identity/boundary")
                    dependencies[str(p)] = path_record(p, "sealed prediction", metadata["prediction_sha256"])
                    dependencies[str(p.parent / "metadata.json")] = path_record(p.parent / "metadata.json", "sealed prediction metadata")
                    if metadata["source_rows_sha256"] != sha(ROOT / cfg["source_rows"]):
                        raise ValueError("Prediction source role binding differs")
                    with np.load(p, allow_pickle=False) as a:
                        positions = a["source_positions"].astype(np.int64)
                        source_p = a["source_probability"].astype(np.float64)
                        target_positions, target_p = a["target_positions"].astype(np.int64), a["target_probability"].astype(np.float64)
                    if not np.array_equal(target_positions, np.arange(len(target[dataset]))):
                        raise ValueError("Prediction target row alignment changed")
                    if source_probability is None:
                        source_probability, source_positions = source_p, positions
                    elif not np.array_equal(source_p, source_probability) or not np.array_equal(positions, source_positions):
                        raise ValueError("Shared fitted source predictions differ between target datasets")
                    archives[dataset] = target_p
                expected_positions = np.sort(np.concatenate((calibration_positions, selection_positions)))
                if not np.array_equal(source_positions, expected_positions):
                    raise ValueError("Prediction source roles are not the exact declared source positions")
                cal = features(source_probability[np.searchsorted(source_positions, calibration_positions)])
                valid_p = source_probability[np.searchsorted(source_positions, selection_positions)]
                valid = features(valid_p)
                models, valid_outputs = {}, {}
                for name, method in cfg["experts"].items():
                    log.write(f"source {detector}/{seed}/{name}\n"); log.flush()
                    m = MapieClassifier(estimator=estimator, method=method, cv="prefit", classes=[0, 1], random_state=seed)
                    m.fit(cal, calibration_labels, task="dev")
                    m.prom_test_ncm(valid, task="dev")
                    predicted, sets, pvalues = m.predict(cal, valid, alpha=alpha, include_last_label=True)
                    if not np.array_equal(predicted, (valid_p > 0.5).astype(int)):
                        raise ValueError("Official sealed predictor changed its forced labels")
                    valid_outputs[name] = {"sets": np.asarray(sets, bool), "pvalues": np.asarray(pvalues, float)}
                    models[name] = m
                choice, table = choose_source(valid_outputs, selection_labels, valid_p, cfg, detector, seed)
                source_table.extend(table)
                selection = {"detector": detector, "seed": seed, **choice,
                             "source_probability_array_sha256": array_digest(source_probability),
                             "selection_positions_sha256": array_digest(selection_positions),
                             "calibration_positions_sha256": array_digest(calibration_positions)}
                selections.append(selection)
                valid_path = OUT / "source-selection" / detector / f"seed-{seed}.npz"
                save_npz(valid_path, {f"{n}_{f}": v for n, r in valid_outputs.items() for f, v in r.items()})
                source_seconds = time.perf_counter() - fit_started
                for dataset in cfg["datasets"]:
                    target_probability = archives[dataset]
                    entries = [e for e in old[dataset]["decisions"] if e["detector"] == detector and e["seed"] == seed]
                    for number, entry in enumerate(entries, 1):
                        started = time.perf_counter()
                        old_root = ROOT / cfg["old_decision_root"] / dataset
                        old_npz, old_metadata = old_root / entry["archive"], old_root / entry["metadata"]
                        dependencies[str(old_npz)] = path_record(old_npz, "old weighted decisions", entry["archive_sha256"])
                        dependencies[str(old_metadata)] = path_record(old_metadata, "old weighted metadata", entry["metadata_sha256"])
                        meta = load_json(old_metadata)
                        with np.load(old_npz, allow_pickle=False) as previous:
                            positions = previous["target_positions"].astype(np.int64)
                            probability = previous["target_probability"].astype(np.float64)
                            eta = previous["source_eta"].astype(np.float64)
                            spec = next(s for s in meta["decision_specs"] if s["method"] == "estimated_weight_no_gate" and s["alpha_v"] == 0.1 and s["alpha_s"] == 0.2 and s["clip"] == 20)
                            weighted = action_codes(previous[spec["key"]])
                        if (not np.array_equal(probability, target_probability[positions])
                                or any(target[dataset][int(i)]["project_group"] != entry["project"] for i in positions)):
                            raise ValueError("Old decisions and sealed target predictions are not aligned")
                        project_features = features(probability)
                        outputs = {}
                        for name, m in models.items():
                            m.prom_test_ncm(project_features, task="dev")
                            predicted, sets, pvalues = m.predict(cal, project_features, alpha=alpha, include_last_label=True)
                            outputs[name] = {"sets": np.asarray(sets, bool), "pvalues": np.asarray(pvalues, float)}
                        accepted = accepted_by_alpha(outputs, alpha)
                        accepted["mixture"] = np.logical_and.reduce([accepted[n] for n in cfg["primary_experts"]])
                        accepted["mixture4"] = np.logical_and.reduce([accepted[n] for n in cfg["experts"]])
                        predicted = (probability > 0.5).astype(np.uint8)
                        primary = actions(accepted[choice["family"]][:, choice["alpha_index"]], predicted)
                        selected_count = int(np.sum(primary != 2))
                        selected_safe, selected_vuln = int(np.sum(primary == 0)), int(np.sum(primary == 1))
                        policies = {"official_prom_source_selected": primary}
                        for name in cfg["primary_experts"] + ["mixture"]:
                            policies[f"official_prom_fixed_{name}"] = actions(accepted[name][:, fixed_index], predicted)
                        policies["official_prom_secondary_fixed_top_k"] = actions(accepted["top_k"][:, fixed_index], predicted)
                        policies["official_prom_secondary_fixed_mixture4"] = actions(accepted["mixture4"][:, fixed_index], predicted)
                        policies["msp_matched_prom"] = action_codes(global_msp_matched(probability, selected_count))
                        policies["allocation_matched_prom"] = action_codes(class_side_matched(probability, selected_safe, selected_vuln))
                        for cost in cfg["cost"]["miss"]:
                            fixed = action_codes(source_cost_matched(eta, selected_count, cost, cfg["cost"]["false_alert"]))
                            free = action_codes(source_cost_matched(eta, len(eta), cost, cfg["cost"]["false_alert"]))
                            free[np.minimum(cost * eta, cfg["cost"]["false_alert"] * (1 - eta)) >= cfg["cost"]["review"]] = 2
                            policies[f"source_iso_cost{cost}_matched_prom"] = fixed
                            policies[f"source_iso_cost{cost}_free_volume"] = free
                        policies.update(trace_weighted_primary=weighted, all_review=np.full(len(positions), 2, dtype=np.uint8), forced_argmax=predicted)
                        for name in ("msp_matched_prom", "source_iso_cost10_matched_prom", "source_iso_cost50_matched_prom"):
                            if int(np.sum(policies[name] != 2)) != selected_count:
                                raise AssertionError("Workload-matched control changed automation count")
                        if (int(np.sum(policies["allocation_matched_prom"] == 0)) != selected_safe
                                or int(np.sum(policies["allocation_matched_prom"] == 1)) != selected_vuln):
                            raise AssertionError("Allocation-matched control changed an automatic-label count")
                        relative = Path("decisions") / dataset / detector / f"seed-{seed}" / Path(entry["archive"]).name
                        destination = OUT / relative
                        arrays = {"target_positions": positions, **{f"action__{n}": v for n, v in policies.items()},
                                  **{f"raw__{n}__{f}": v for n, r in outputs.items() for f, v in r.items()}}
                        save_npz(destination, arrays)
                        cell = {"dataset": dataset, "detector": detector, "seed": seed, "project": entry["project"],
                                "archive": str(relative).replace("\\", "/"), "archive_sha256": sha(destination),
                                "n": len(positions), "methods": list(policies), "source_selected_family": choice["family"],
                                "source_selected_alpha": choice["alpha"], "weighted_domain_status": meta["domain_status"],
                                "official_transductive_batch": "complete target project", "wall_seconds": time.perf_counter() - started}
                        cells.append(cell)
                        for name, codes in policies.items():
                            counts.append({**{k: cell[k] for k in ("dataset", "detector", "seed", "project")}, "method": name,
                                           "n": len(codes), "auto_safe": int(np.sum(codes == 0)), "auto_vulnerable": int(np.sum(codes == 1)),
                                           "review_count": int(np.sum(codes == 2)), "weighted_domain_status": cell["weighted_domain_status"]})
                        write_json(OUT / "progress.json", {"updated_at_utc": utc(), "phase": "label-free sealing", "detector": detector,
                                                          "seed": seed, "dataset": dataset, "project": entry["project"], "completed_cells": len(cells), "target_labels_accessed": False})
                        if number == len(entries):
                            print(f"sealed {dataset} {detector} seed={seed}: {number} projects", flush=True)
                timings.append({"detector": detector, "seed": seed, "source_selection_seconds": source_seconds,
                                "all_datasets_seconds": time.perf_counter() - fit_started})
    write_csv(OUT / "source_selection.csv", source_table)
    write_json(OUT / "source_selection_manifest.json", {"selections": selections, "target_labels_accessed": False,
                                                         "candidate_family": cfg["primary_experts"] + ["mixture"], "alpha_grid": cfg["alpha_grid"]})
    write_csv(OUT / "label_free_action_counts.csv", counts)
    write_csv(OUT / "computation_receipts.csv", timings)
    literal_equivalence = compare_literal_outputs()
    dependencies[str(OUT / "literal_equivalence.json")] = path_record(OUT / "literal_equivalence.json", "direct-versus-optimized complete real-array parity")
    for path in sorted((OUT / "literal-execution-evidence").rglob("*")):
        if path.is_file():
            dependencies[str(path)] = path_record(path, "preserved literal execution evidence")
    source_records = [{"path": str(p.relative_to(OUT)).replace("\\", "/"), "sha256": sha(p)} for p in sorted((OUT / "source-selection").rglob("*.npz"))]
    result = {"schema": "trace-prom-official-component-decision-seal-v4.1", "created_at_utc": utc(),
              "protocol_version": cfg["protocol_version"], "official_commit": cfg["official_commit"], "official_module_version": version,
              "execution_version": "trace-official-prom-pooled-count-execution-v4.2", "literal_parity_passed": literal_equivalence["passed"],
              "target_labels_accessed": False, "previous_target_outcomes_known": True,
              "scope": "Actual pinned official conformal code with recorded binary/feature/source-selection adaptation; full official system not reproduced",
              "dependencies": list(dependencies.values()), "decisions": cells, "source_output_archives": source_records,
              "source_selections": selections, "actual_fits": cfg["detectors"], "planned_cells": sum(len(s["decisions"]) for s in old.values()),
              "target_rows_paths": {d: str((ROOT / "outputs/neural-validation-v3/inputs" / ("megavul/target_rows.jsonl" if d == "megavul" else "target_rows.jsonl")).relative_to(ROOT)).replace("\\", "/") for d in cfg["datasets"]},
              "target_vault_expected_sha256": {d: old[d]["prepared_target_label_vault_sha256"] for d in cfg["datasets"]},
              "reports": {p.name: sha(p) for p in (OUT / "source_selection.csv", OUT / "source_selection_manifest.json", OUT / "label_free_action_counts.csv", OUT / "computation_receipts.csv")},
              "wall_seconds": time.perf_counter() - start, "cpu_thread_limit": 1}
    if len(cells) != result["planned_cells"]:
        raise ValueError("Component decision grid is incomplete")
    write_json(final_path, result)
    (OUT / "decision_seal.sha256").write_text(sha(final_path) + "\n", encoding="ascii")
    write_json(OUT / "progress.json", {"phase": "label-free seal complete", "completed_cells": len(cells), "target_labels_accessed": False, "updated_at_utc": utc()})
    return result


def compare_literal_outputs() -> dict:
    """Compare preserved real direct-run arrays before any target outcome join."""
    literal = OUT / "literal-execution-evidence"
    execution = load_json(OUT / "execution_optimization.json")
    for record in execution["preserved_literal_files"]:
        path_record(literal / record["path"], "original direct-run evidence", record["sha256"])
    direct_tree = ast.parse((literal / "prom_official_v4_before_optimization.py").read_text(encoding="utf-8"))
    current_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    relevant = {"choose_source", "accepted_by_alpha", "features", "actions", "SealedProbabilityEstimator"}
    extract = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in relevant}
    if extract(direct_tree) != extract(current_tree):
        raise ValueError("Source-selection or action code changed during count optimization")
    comparisons = []
    for direct in sorted(literal.rglob("*.npz")):
        relative = direct.relative_to(literal)
        optimized = OUT / relative
        fields = {}
        with np.load(direct, allow_pickle=False) as a, np.load(optimized, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Direct/optimized archive schemas differ")
            for name in a.files:
                fields[name] = bool(np.array_equal(a[name], b[name], equal_nan=True))
            if not all(fields.values()):
                raise AssertionError(f"Literal/optimized real arrays differ: {relative}")
        comparisons.append({"path": str(relative).replace("\\", "/"), "array_count": len(fields), "all_arrays_exact_equal": True,
                            "direct_archive_sha256": sha(direct), "optimized_archive_sha256": sha(optimized)})
    source_count = sum(c["path"].startswith("source-selection/") for c in comparisons)
    cell_count = sum(c["path"].startswith("decisions/") for c in comparisons)
    if source_count != 1 or cell_count != execution["preserved_literal_target_cells"]:
        raise ValueError("Literal source/target evidence comparison is incomplete")
    result = {"schema": "trace-prom-real-literal-equivalence-v4.2", "created_at_utc": utc(), "passed": True,
              "source_selection_code_exact_ast_equal": True, "source_grid_archives": source_count, "source_validation_rows": 6296,
              "literal_target_cells": cell_count, "complete_global_pool_retained": True, "target_labels_accessed": False,
              "scope": "Every array in the real complete source alpha grid and all previously literal-executed target cells; no outcome comparison", "comparisons": comparisons}
    write_json(OUT / "literal_equivalence.json", result)
    return result


def verify_new_seal() -> dict:
    checked_protocol()
    path = OUT / "decision_seal.json"
    expected = (OUT / "decision_seal.sha256").read_text(encoding="ascii").strip().upper()
    if sha(path) != expected:
        raise ValueError("PROM decision seal checksum differs")
    manifest = load_json(path)
    if manifest["target_labels_accessed"] is not False or manifest.get("literal_parity_passed") is not True or len(manifest["decisions"]) != manifest["planned_cells"]:
        raise ValueError("PROM seal boundary/completeness differs")
    for record in manifest["dependencies"]:
        path_record(resolve(record["path"]), record["role"], record["sha256"])
    seen = set()
    for cell in manifest["decisions"]:
        key = tuple(cell[k] for k in ("dataset", "detector", "seed", "project"))
        if key in seen:
            raise ValueError("Duplicate official component cell")
        seen.add(key)
        path_record(OUT / cell["archive"], "sealed official actions", cell["archive_sha256"])
    for record in manifest["source_output_archives"]:
        path_record(OUT / record["path"], "source component output", record["sha256"])
    for name, digest in manifest["reports"].items():
        path_record(OUT / name, "label-free aggregate", digest)
    return manifest


def metric_row(labels, codes, forced) -> dict:
    y, a = np.asarray(labels, dtype=int), np.asarray(codes, dtype=int)
    if y.ndim != 1 or a.shape != y.shape or not np.isin(y, (0, 1)).all() or not np.isin(a, (0, 1, 2)).all():
        raise ValueError("Invalid aligned outcome/action arrays")
    n, nv, ns = len(y), int(np.sum(y == 1)), int(np.sum(y == 0))
    fn, fp = int(np.sum((a == 0) & (y == 1))), int(np.sum((a == 1) & (y == 0)))
    review, n_auto = int(np.sum(a == 2)), int(np.sum(a != 2))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {"n": n, "n_vulnerable": nv, "n_safe": ns, "auto_safe": int(np.sum(a == 0)), "auto_vulnerable": int(np.sum(a == 1)),
            "auto_fn": fn, "auto_fp": fp, "review_count": review, "coverage": n_auto / n, "review_fraction": review / n,
            "auto_fn_rate": ratio(fn, nv), "auto_fp_rate": ratio(fp, ns), "selective_error": ratio(fn + fp, n_auto),
            "cost_10": (10 * fn + fp + 0.1 * review) / n, "cost_50": (50 * fn + fp + 0.1 * review) / n,
            **f1_counts(a == 2, np.asarray(forced) != y)}


def aggregate_rows(rows: list[dict], cfg: dict):
    groups = defaultdict(list)
    identity = ("dataset", "detector", "project", "method")
    numeric = [key for key in rows[0] if key not in identity + ("seed", "weighted_domain_status", "source_selected_family", "source_selected_alpha")]
    for row in rows:
        groups[tuple(row[k] for k in identity)].append(row)
    averages = []
    for key, records in sorted(groups.items()):
        if sorted(int(r["seed"]) for r in records) != sorted(cfg["detectors"][key[1]]):
            raise ValueError("Repeated/missing fitted seed in project average")
        row = dict(zip(identity, key))
        row["actual_seed_count"] = len(records)
        row["all_weighted_domain_fits_supported"] = all(r["weighted_domain_status"] == "ok" for r in records)
        for name in numeric:
            values = np.asarray([float(r[name]) for r in records])
            row[name] = float(values.mean()) if np.isfinite(values).all() else np.nan
        averages.append(row)
    summaries, cohort_groups = [], defaultdict(list)
    for row in averages:
        cohort_groups[(row["dataset"], row["detector"], row["method"])].append(row)
    for (dataset, detector, method), records in sorted(cohort_groups.items()):
        row = {"dataset": dataset, "detector": detector, "method": method, "project_count": len(records),
               "weighted_domain_supported_projects": sum(r["all_weighted_domain_fits_supported"] for r in records)}
        for name in numeric:
            values = np.asarray([r[name] for r in records]); finite = values[np.isfinite(values)]
            row[f"median_{name}"] = float(np.median(finite)) if len(finite) else np.nan
        row["selective_error_defined_projects"] = sum(np.isfinite(r["selective_error"]) for r in records)
        summaries.append(row)
    pairings = [("official_prom_source_selected", name) for name in ("msp_matched_prom", "allocation_matched_prom", "source_iso_cost10_matched_prom",
                 "source_iso_cost50_matched_prom", "source_iso_cost10_free_volume", "source_iso_cost50_free_volume", "trace_weighted_primary", "all_review", "forced_argmax")]
    pairings.append(("official_prom_fixed_mixture", "official_prom_source_selected"))
    outcomes = ("auto_fn_rate", "auto_fp_rate", "review_fraction", "coverage", "selective_error", "cost_10", "cost_50", "error_detection_f1")
    rng = np.random.default_rng(cfg["statistics"]["bootstrap_seed"])
    paired = []
    for dataset in cfg["datasets"]:
        for detector in cfg["detectors"]:
            for candidate, baseline in pairings:
                first = {r["project"]: r for r in averages if r["dataset"] == dataset and r["detector"] == detector and r["method"] == candidate}
                second = {r["project"]: r for r in averages if r["dataset"] == dataset and r["detector"] == detector and r["method"] == baseline}
                if first.keys() != second.keys():
                    raise ValueError("Unmatched project rosters in a paired comparison")
                for outcome in outcomes:
                    projects = sorted(first)
                    differences = np.asarray([first[p][outcome] - second[p][outcome] for p in projects])
                    finite = np.isfinite(differences)
                    differences = differences[finite]
                    if len(differences):
                        samples = rng.integers(0, len(differences), size=(cfg["statistics"]["bootstrap_draws"], len(differences)))
                        boot = np.median(differences[samples], axis=1)
                        lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                        median = float(np.median(differences))
                    else:
                        median = lo = hi = np.nan
                    paired.append({"dataset": dataset, "detector": detector, "candidate": candidate, "baseline": baseline, "outcome": outcome,
                                   "difference_direction": "candidate minus baseline", "project_count": len(projects), "defined_project_count": len(differences),
                                   "median_difference": median, "ci_low": float(lo), "ci_high": float(hi),
                                   "n_negative": int(np.sum(differences < 0)), "n_positive": int(np.sum(differences > 0)), "n_tie": int(np.sum(differences == 0))})
    return averages, summaries, paired


def evaluate() -> dict:
    started = time.perf_counter()
    manifest = verify_new_seal()  # all byte/role/decision checks BEFORE the vault
    cfg = load_json(CONFIG)
    labels, targets, vaults = {}, {}, []
    for dataset in cfg["datasets"]:
        targets[dataset] = read_target(resolve(manifest["target_rows_paths"][dataset]))
        path = ROOT / "outputs/neural-validation-v3/inputs" / ("megavul/target_label_vault.npz" if dataset == "megavul" else "target_label_vault.npz")
        vaults.append(path_record(path, "existing target-label vault", manifest["target_vault_expected_sha256"][dataset]))
        with np.load(path, allow_pickle=False) as a:
            if not np.array_equal(a["positions"], np.arange(len(targets[dataset]))) or not np.array_equal(a["row_ids"].astype(str), [r["row_id"] for r in targets[dataset]]):
                raise ValueError("Outcome vault positions/row IDs differ from sealed metadata")
            labels[dataset] = a["labels"].astype(np.int64)
    rows = []
    for cell in manifest["decisions"]:
        identity = {k: cell[k] for k in ("dataset", "detector", "seed", "project")}
        with np.load(OUT / cell["archive"], allow_pickle=False) as a:
            positions = a["target_positions"].astype(int)
            y, forced = labels[cell["dataset"]][positions], a["action__forced_argmax"]
            for method in cell["methods"]:
                rows.append({**identity, "method": method, "weighted_domain_status": cell["weighted_domain_status"],
                             "source_selected_family": cell["source_selected_family"], "source_selected_alpha": cell["source_selected_alpha"],
                             **metric_row(y, a[f"action__{method}"], forced)})
    averages, summaries, paired = aggregate_rows(rows, cfg)
    write_csv(OUT / "project_seed_metrics.csv", rows)
    write_csv(OUT / "project_seed_averages.csv", averages)
    write_csv(OUT / "policy_project_summary.csv", summaries)
    write_csv(OUT / "paired_project_summary.csv", paired)
    result = {"schema": "trace-prom-official-component-evaluation-v4.1", "created_at_utc": utc(), "status": "completed",
              "official_commit": cfg["official_commit"], "scope": manifest["scope"], "all_previous_TRACE_outcomes_known": True,
              "execution_version": manifest["execution_version"], "literal_real_array_parity_passed_before_labels": True,
              "new_decision_seal_sha256": sha(OUT / "decision_seal.json"), "verified_before_target_label_access": True,
              "target_labels_used_for_selection": False, "project_seed_cells": len(manifest["decisions"]), "metric_rows": len(rows),
              "source_roles": {"calibration": 4400, "selection": 6296}, "actual_fits": cfg["detectors"],
              "cohorts": {d: {"projects": len({r["project"] for r in rows if r["dataset"] == d}), "functions": len(labels[d])} for d in cfg["datasets"]},
              "target_vaults": vaults, "primary_experts": cfg["primary_experts"], "secondary_experts": cfg["secondary_experts"],
              "statistics": cfg["statistics"], "cost": cfg["cost"], "whole_system_not_reproduced": cfg["whole_system_not_reproduced"],
              "full_neural_feedback_retraining_executed": False, "wall_seconds": time.perf_counter() - started,
              "aggregate_files": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in (OUT / "project_seed_metrics.csv", OUT / "project_seed_averages.csv", OUT / "policy_project_summary.csv", OUT / "paired_project_summary.csv")]}
    write_json(OUT / "evaluation_manifest.json", result)
    write_report(result, summaries, paired)
    write_json(OUT / "progress.json", {"phase": "evaluation complete", "status": "completed", "updated_at_utc": utc(), "project_seed_cells": len(manifest["decisions"]), "target_labels_used_for_selection": False})
    return result


def write_report(result: dict, summaries: list[dict], paired: list[dict]):
    lines = ["# Executed official PROM component comparison", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run", "- Origin Date: 2026-10-11",
             "- Verification Status: VERIFIED for executed software/lineage; scientific scope remains component adaptation", "- Version Label: trace-official-prom-component-v4.1", "",
             "This new retrospective comparison uses actual author PROM conformal code, pinned at `f16b52772123064551b6450f10b308971c4cdd39`. All earlier TRACE outcomes were already known. Alpha/expert decisions use source labels only and were sealed before the new outcome join.", "",
             f"Completed {result['project_seed_cells']} project/actual-fit cells and {result['metric_rows']} policy metric rows. DiverseVul has {result['cohorts']['diversevul']['projects']} projects/{result['cohorts']['diversevul']['functions']} functions; MegaVul has {result['cohorts']['megavul']['projects']} projects/{result['cohorts']['megavul']['functions']} functions. Each collection uses three fine-tuned fits, one frozen fit and three hashing fits on the unchanged source roles.", "",
             "The primary family is native-case-study LAC/APS/RAPS plus conservative union; top-k/four-expert union are secondary. A declared two-line one-hot repair enables binary APS/RAPS and preserves every checked native eight-class output exactly. Two-column detector probabilities replace native token-space neighborhood features; the upstream globally pooled, transductive neighborhood p-values are retained, with each complete project as its deployment batch.", "",
             "A separately recorded count optimization sorts the complete duplicated global pool and uses the unchanged >= comparison and total denominator. Exact native/binary, repeated-tie/edge-alpha and NaN/infinity tests pass. Every raw array in the real 6,296-row direct source grid and all 43 direct target cells matches the optimized execution exactly before target labels are joined. The original direct runtime/evidence and both diffs are preserved.", "",
             "Values below are project medians after averaging actual fitted seeds. Costs assume correct review at relative price 0.1, false alerts at 1, and misses at 10 or 50; they do not measure analyst time or monetary savings.", "",
             "| Collection | Detector | Policy | Automation | Miss rate | False-alert rate | Cost 10 | Cost 50 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["method"] in ("official_prom_source_selected", "allocation_matched_prom", "trace_weighted_primary", "source_iso_cost50_free_volume"):
            lines.append(f"| {r['dataset']} | {r['detector']} | {r['method']} | {r['median_coverage']:.4f} | {r['median_auto_fn_rate']:.4f} | {r['median_auto_fp_rate']:.4f} | {r['median_cost_10']:.4f} | {r['median_cost_50']:.4f} |")
    lines += ["", "Official component minus its control at the same automatic-safe and automatic-vulnerable counts:", "",
              "| Collection | Detector | Endpoint | Median difference | Pointwise 95% interval | Negative / positive / tied projects |", "|---|---|---|---:|---|---|"]
    for r in paired:
        if r["candidate"] == "official_prom_source_selected" and r["baseline"] == "allocation_matched_prom" and r["outcome"] in ("auto_fn_rate", "auto_fp_rate", "cost_10", "cost_50"):
            lines.append(f"| {r['dataset']} | {r['detector']} | {r['outcome']} | {r['median_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['n_negative']} / {r['n_positive']} / {r['n_tie']} |")
    lines += ["", "All unfavorable results, zero-automation cases, and unsupported old domain fits remain in the tables. Workload controls preserve the exact review count; allocation controls additionally preserve both automatic-label counts. Direct comparison with old weighted decisions includes their sealed full-review fallback. Domain availability is recorded in every cell and is not used to select a favorable subset.", "",
              "The native eight-class API software demonstrator executes official conformal prediction, union/evaluation and feedback-index selection, then refits a GaussianNB model on six acquired synthetic rows and evaluates a disjoint holdout. It is not the unchanged official benchmark or TRACE neural feedback training. The full published CWE case study, its original model/taxonomy and token features, deployment-label best-F1 reporting, full incremental neural retraining and original Docker environment remain unperformed. The documented tutorial requires unavailable TF/Keras/NNI, Linux paths, and a saved model absent from the repository inventory.", "",
              "These results support a bounded official-code component comparison. They do not establish superiority over the full official PROM system, statistical equivalence, or guaranteed journal acceptance.", ""]
    (OUT / "prom_official_v4_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify", "evaluate"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            result = {"seal": seal, "verify": verify_new_seal, "evaluate": evaluate}[args.command]()
    except Exception:
        name = f"{args.command}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        (OUT / name).write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(json.dumps({"command": args.command, "status": result.get("status", "completed"),
                      "cells": len(result.get("decisions", [])) or result.get("project_seed_cells"),
                      "target_labels_used_for_selection": False}, indent=2))


if __name__ == "__main__":
    main()
