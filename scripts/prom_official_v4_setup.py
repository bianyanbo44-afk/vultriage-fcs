"""Acquire provenance and verify a minimal binary repair of actual PROM code.

Only writes within the new PROM v4 workspace/output paths. It never opens TRACE
target labels. The pinned official checkout is kept byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import importlib
import importlib.util
import io
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import types
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/neurocomputing-strengthening/kbs-v4/prom"
OUT = ROOT / "outputs/kbs-validation-v4/prom"
COMMIT = "f16b52772123064551b6450f10b308971c4cdd39"
OFFICIAL = WORK / "official-source"
RUNTIME = WORK / "binary-runtime"
DEPENDENCIES = WORK / "runtime-dependencies"
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(OFFICIAL), *args], text=True, encoding="utf-8").strip()


def bootstrap(source: Path):
    """Resolve upstream mixed namespaces without rewriting its source code.

    The package __init__ eagerly imports irrelevant regression modules with
    mutually conflicting absolute namespaces. Both namespace names here point
    to the same real source package, and real module code executes unchanged.
    """
    sys.dont_write_bytecode = True
    for name in list(sys.modules):
        if name == "prom" or name.startswith("prom.") or name == "src.prom" or name.startswith("src.prom."):
            del sys.modules[name]
    package = types.ModuleType("prom")
    package.__path__ = [str(source / "src/prom")]
    package.__package__ = "prom"
    sys.modules["prom"] = package
    namespace = types.ModuleType("src")
    namespace.__path__ = [str(source / "src")]
    sys.modules["src"] = namespace
    sys.modules["src.prom"] = package
    classification = importlib.import_module("prom.prom_classification")
    utilities = importlib.import_module("prom.prom_util")
    version = importlib.import_module("prom._version").__version__
    assert Path(classification.__file__).resolve() == (source / "src/prom/prom_classification.py").resolve()
    return classification.MapieClassifier, utilities.Prom_utils, version


def acquire() -> dict:
    assert git("rev-parse", "HEAD") == COMMIT
    if git("diff", "--name-only", "HEAD"):
        raise ValueError("Pinned official source has tracked modifications")
    provenance = WORK / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    urls = {
        "institutional_record.html": "https://eprints.whiterose.ac.uk/id/eprint/221073/",
        "github_repository.json": "https://api.github.com/repos/HuantWang/PROM",
        "github_commit.json": f"https://api.github.com/repos/HuantWang/PROM/commits/{COMMIT}",
    }
    receipts = []
    for filename, url in urls.items():
        request = urllib.request.Request(url, headers={"User-Agent": "TRACE-official-PROM-v4-reproducibility"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
            status, final_url, content_type = response.status, response.url, response.headers.get("Content-Type")
        destination = provenance / filename
        destination.write_bytes(data)
        receipts.append({"url": url, "final_url": final_url, "http_status": status,
                         "content_type": content_type, "retrieved_at_utc": utc(),
                         "path": str(destination.relative_to(ROOT)), "bytes": len(data), "sha256": sha(destination)})
    record = (provenance / "institutional_record.html").read_text(encoding="utf-8")
    if "https://github.com/HuantWang/PROM" not in record or "10.1145/3696443.3708959" not in record:
        raise ValueError("Primary institutional record did not verify the DOI/repository link")
    tracked = git("ls-files").splitlines()
    snapshot = [{"path": name, "sha256": sha(OFFICIAL / name), "bytes": (OFFICIAL / name).stat().st_size}
                for name in tracked if (OFFICIAL / name).is_file()]
    native_inventory = git("ls-tree", "-r", "--name-only", "HEAD", "examples/case_study/BugD", "examples/case_study/Thread", "benchmark/Thread").splitlines()
    (provenance / "native_inventory.txt").write_text("\n".join(native_inventory) + "\n", encoding="utf-8")
    result = {"schema": "trace-prom-official-provenance-v4.1", "retrieved_at_utc": utc(),
              "paper_doi": "10.1145/3696443.3708959", "paper_repository_verified_by": urls["institutional_record.html"],
              "repository": "https://github.com/HuantWang/PROM", "commit": COMMIT,
              "commit_record": git("show", "-s", "--format=%H%n%aI%n%cI%n%s", "HEAD"),
              "branch": "development", "release_tags": "none returned by git ls-remote --tags",
              "retrieval": "depth-one filtered sparse checkout; unrelated benchmark blobs not downloaded",
              "top_level_license": "CC-BY-4.0", "nested_src_license": "MIT notice with literal YOUR NAME placeholder",
              "license_files": [{"path": str((OFFICIAL / p).relative_to(ROOT)), "sha256": sha(OFFICIAL / p)} for p in ("LICENSE", "src/LICENSE.txt")],
              "source_snapshot": snapshot, "fetch_receipts": receipts,
              "native_vulnerability_taxonomy": "8 CWE classes, not binary vulnerable/safe",
              "native_vulnerability_precomputed_prediction_archives": [p for p in native_inventory if p.startswith("examples/case_study/BugD/") and Path(p).suffix in (".npy", ".npz", ".pkl", ".pt", ".bin")],
              "native_tutorial_limitations": ["hard-coded Linux paths", "requires TensorFlow, Keras preprocessing, NNI and extra model imports", "documented pretrained tutorial model is not in the pinned repository inventory"],
              "target_labels_accessed": False}
    write_json(OUT / "provenance.json", result)
    return result


def prepare_runtime() -> dict:
    source = OFFICIAL / "src/prom/prom_classification.py"
    original = source.read_bytes()
    ending = b"\r\n" if b"\r\n" in original else b"\n"
    needle = ending.join([b"        y_true = label_binarize(", b"            y=y, classes=self.classes_", b"        )"])
    addition = ending.join([b"", b"        if len(self.classes_) == 2:",
                            b"            y_true = np.column_stack((1 - y_true[:, 0], y_true[:, 0]))"])
    if original.count(needle) != 1:
        raise ValueError("Binary repair locator is not unique in pinned source")
    repaired = original.replace(needle, needle + addition)
    if RUNTIME.exists():
        if (RUNTIME / "src/prom/prom_classification.py").read_bytes() != repaired:
            raise ValueError("Existing private runtime differs; refusing to overwrite it")
    else:
        shutil.copytree(OFFICIAL / "src", RUNTIME / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(OFFICIAL / "LICENSE", RUNTIME / "LICENSE")
        (RUNTIME / "src/prom/prom_classification.py").write_bytes(repaired)
    diff = "".join(difflib.unified_diff(original.decode("utf-8").splitlines(keepends=True),
                                     repaired.decode("utf-8").splitlines(keepends=True),
                                     fromfile="official-source/src/prom/prom_classification.py",
                                     tofile="binary-runtime/src/prom/prom_classification.py"))
    diff_path = WORK / "prom_official_v4_binary_repair.diff"
    diff_path.write_text(diff, encoding="utf-8")
    result = {"schema": "trace-prom-binary-repair-v4.1", "created_at_utc": utc(), "commit": COMMIT,
              "original_sha256": sha(source), "runtime_sha256": sha(RUNTIME / "src/prom/prom_classification.py"),
              "diff_path": str(diff_path.relative_to(ROOT)), "diff_sha256": sha(diff_path),
              "source_changes": "Two added executable lines: two-column one-hot only when exactly two classes are declared",
              "import_bootstrap": "Alias prom and src.prom namespaces to the actual source package, avoid unrelated eager imports; no source method replacement",
              "other_algorithm_changes": [], "target_labels_accessed": False}
    write_json(OUT / "binary_repair.json", result)
    return result


def low_level(source: Path, x_cal, y_cal, x_test, classifier, classes, seed=73, alphas=None):
    import numpy as np
    MapieClassifier, _, version = bootstrap(source)
    methods = {"lac": "score", "top_k": "top_k", "aps": "cumulated_score", "raps": "raps"}
    results = {}
    for name, method in methods.items():
        m = MapieClassifier(estimator=classifier, method=method, cv="prefit", classes=classes, random_state=seed)
        m.fit(x_cal, y_cal, task="bug" if len(classes) == 8 else "dev")
        m.prom_test_ncm(x_test, task="bug" if len(classes) == 8 else "dev")
        predicted, sets, pvalues = m.predict(x_cal, x_test, alpha=[0.1, 0.3, 0.5] if alphas is None else alphas, include_last_label=True)
        results[name] = {"predicted": np.asarray(predicted), "sets": np.asarray(sets),
                         "pvalues": np.asarray(pvalues), "scores": np.asarray(m.conformity_scores_),
                         "test_scores": np.asarray(m.test_conformity_scores_), "quantiles": np.asarray(m.quantiles_)}
    return results, version


def smoke() -> dict:
    import numpy as np
    import scipy
    import sklearn
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB

    started = time.perf_counter()
    output = OUT / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    x, y = make_classification(n_samples=1200, n_features=20, n_informative=16, n_redundant=0,
                               n_classes=8, n_clusters_per_class=1, class_sep=0.8, random_state=613)
    classifier = GaussianNB().fit(x[:600], y[:600])
    cal_x, cal_y, test_x, test_y = x[600:1000], y[600:1000], x[1000:1128], y[1000:1128]
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
        native, version = low_level(OFFICIAL, cal_x, cal_y, test_x, classifier, list(range(8)))
        adapted, _ = low_level(RUNTIME, cal_x, cal_y, test_x, classifier, list(range(8)))
    parity = {name: {field: bool(np.array_equal(native[name][field], adapted[name][field])) for field in native[name]} for name in native}
    if not all(all(fields.values()) for fields in parity.values()):
        raise AssertionError("Binary repair changes native multiclass outputs")
    np.savez_compressed(output / "native_multiclass_outputs.npz", **{f"{n}_{f}": a for n, r in native.items() for f, a in r.items()})

    xb, yb = make_classification(n_samples=650, n_features=8, n_informative=6, n_redundant=0, random_state=619)
    binary = GaussianNB().fit(xb[:250], yb[:250])
    binary_failures = {}
    MapieClassifier, _, _ = bootstrap(OFFICIAL)
    for name, method in {"lac": "score", "top_k": "top_k", "aps": "cumulated_score", "raps": "raps"}.items():
        try:
            m = MapieClassifier(estimator=binary, method=method, cv="prefit", classes=[0, 1], random_state=73)
            m.fit(xb[250:550], yb[250:550], task="dev")
            m.prom_test_ncm(xb[550:], task="dev")
            m.predict(xb[250:550], xb[550:], alpha=[0.1, 0.3, 0.5])
            binary_failures[name] = {"status": "passed_without_repair"}
        except Exception as exc:
            binary_failures[name] = {"status": "failed_without_repair", "exception": type(exc).__name__, "message": str(exc)}
    repaired, _ = low_level(RUNTIME, xb[250:550], yb[250:550], xb[550:], binary, [0, 1])
    np.savez_compressed(output / "repaired_binary_outputs.npz", **{f"{n}_{f}": a for n, r in repaired.items() for f, a in r.items()})

    # Real official helper selection/union/acquisition API on its native 8-class
    # task. Synthetic data is a software demonstrator, never benchmark evidence.
    _, Prom_utils, _ = bootstrap(OFFICIAL)
    native_methods = {"lac": ("score", True), "aps": ("cumulated_score", True), "raps": ("raps", True)}
    helper = Prom_utils(classifier, native_methods, task="bug")
    np.random.seed(73)
    with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
        predictions, sets, values = helper.conformal_prediction(cal_x, cal_y, test_x, test_y, significance_level=0.1)
        result = helper.evaluate_conformal_prediction(predictions, sets, values, classifier.predict(test_x), test_y, significance_level=0.1)
        train_indices, remaining = helper.incremental_learning(73, list(range(1000, 1128)), list(range(600)))
    acquired = sorted(set(train_indices) - set(range(600)))
    np.savez_compressed(output / "native_helper_outputs.npz", **{f"{n}_sets": np.asarray(sets[n]) for n in sets},
                        **{f"{n}_pvalues": np.asarray(values[n]) for n in values},
                        union_accepted=np.asarray(result[0], dtype=int), acquired=np.asarray(acquired), remaining=np.asarray(remaining))
    before = classifier.predict(x[1128:])
    retrained = GaussianNB().fit(x[train_indices], y[train_indices])
    after = retrained.predict(x[1128:])
    # An independent binary mathematical oracle verifies the repair's one-hot
    # score, without relying on the previous TRACE component reimplementation.
    MapieClassifier, _, _ = bootstrap(RUNTIME)
    m = MapieClassifier(estimator=binary, method="cumulated_score", cv="prefit", classes=[0, 1], random_state=73)
    m.fit(xb[250:550], yb[250:550], task="dev")
    probabilities = binary.predict_proba(xb[250:550])
    order = np.fliplr(np.argsort(probabilities, axis=1))
    rank = np.argmax(order == yb[250:550, None], axis=1)
    cumulative = np.cumsum(np.take_along_axis(probabilities, order, axis=1), axis=1)
    uniforms = np.random.RandomState(73).uniform(size=300)
    expected = cumulative[np.arange(300), rank] - uniforms * probabilities[np.arange(300), yb[250:550]]
    mathematical_parity = bool(np.array_equal(expected, m.conformity_scores_.ravel()))
    if not mathematical_parity:
        raise AssertionError("Repaired binary APS does not match explicit mathematical score")
    (output / "execution.log").write_text(logs.getvalue(), encoding="utf-8")
    result = {"schema": "trace-prom-official-fidelity-smoke-v4.1", "completed_at_utc": utc(),
              "status": "passed", "wall_seconds": time.perf_counter() - started,
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__, "official_module_version": version},
              "native_multiclass_exact_parity": parity, "native_binary_path": binary_failures,
              "repaired_binary_all_four_experts_executed": list(repaired), "binary_aps_independent_mathematical_parity": mathematical_parity,
              "native_api_demonstrator": {"data": "seeded synthetic eight-class data, not native benchmark data", "trained_model": "GaussianNB",
                                           "official_task": "bug", "official_helper_executed": ["conformal_prediction", "evaluate_conformal_prediction", "incremental_learning"],
                                           "feedback_acquired": len(acquired), "model_refit_executed": True,
                                           "disjoint_remaining_holdout": len(after), "holdout_predictions_changed": int(np.sum(before != after)),
                                           "claim_scope": "Native official supported API software smoke; not unchanged ae_tutorial.py or original benchmark reproduction"},
              "documented_full_tutorial_executed": False,
              "documented_full_tutorial_blockers": ["native Magni_utils.py imports unavailable TensorFlow/Keras/NNI at module level", "Linux absolute paths and missing documented pretrained tutorial checkpoint"],
              "current_TRACE_neural_incremental_retraining_executed": False, "target_labels_accessed": False,
              "artifacts": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(OUT / "fidelity_smoke.json", result)
    return result


def optimize() -> dict:
    """Record and verify an exact pooled-count acceleration before outcomes."""
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB
    import warnings

    before = WORK / "literal-binary-runtime"
    literal = OUT / "literal-execution-evidence"
    if (OUT / "decision_seal.json").exists() or (OUT / "evaluation_manifest.json").exists():
        raise ValueError("Execution optimization is only allowed before final seal/outcome join")
    initial_repair = json.loads((OUT / "binary_repair.json").read_text(encoding="utf-8"))
    if not before.exists():
        if sha(RUNTIME / "src/prom/prom_classification.py") != initial_repair["runtime_sha256"]:
            raise ValueError("Literal binary runtime differs from its original repair receipt")
        shutil.copytree(RUNTIME, before, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if not literal.exists():
        literal.mkdir(parents=True)
        for folder in ("decisions", "source-selection"):
            if (OUT / folder).exists():
                shutil.copytree(OUT / folder, literal / folder)
        for name in ("official_execution.log", "progress.json"):
            if (OUT / name).is_file():
                shutil.copy2(OUT / name, literal / name)
        shutil.copy2(Path(__file__), literal / "prom_official_v4_setup_before_optimization.py")
        shutil.copy2(ROOT / "scripts/prom_official_v4.py", literal / "prom_official_v4_before_optimization.py")
    preserved = [{"path": str(p.relative_to(literal)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(literal.rglob("*")) if p.is_file()]
    # Every preserved npz must be a real completed array archive, not a partial.
    for p in literal.rglob("*.npz"):
        with np.load(p, allow_pickle=False) as arrays:
            for key in arrays.files:
                _ = arrays[key]
    original_path = before / "src/prom/prom_classification.py"
    original = original_path.read_bytes()
    ending = b"\r\n" if b"\r\n" in original else b"\n"
    start = original.index(b"    def _compute_p_values(")
    finish = original.index(b"    def predict(", start)
    lines = [
        "    def _compute_p_values(self, y_pred_proba, conformity_scores_, selected_cal):",
        "        # Execution-equivalent count of the COMPLETE duplicated global pool.",
        "        cal_conformity_scores = conformity_scores_[selected_cal]",
        "        pool = np.asarray(cal_conformity_scores).reshape(-1)",
        "        scores = np.asarray(y_pred_proba)[:, 0]",
        "        if pool.size == 0:",
        "            return np.full(scores.shape, np.nan).tolist()",
        "        # NaNs fail >= comparisons but retain their original denominator.",
        "        sorted_pool = np.sort(pool[~np.isnan(pool)])",
        "        counts = len(sorted_pool) - np.searchsorted(sorted_pool, scores, side=\"left\")",
        "        counts[np.isnan(scores)] = 0",
        "        return (counts.astype(np.float64) / pool.size).tolist()",
        "", "", "",
    ]
    optimized = original[:start] + ending.join(line.encode("utf-8") for line in lines) + original[finish:]
    runtime_path = RUNTIME / "src/prom/prom_classification.py"
    runtime_path.write_bytes(optimized)
    diff = "".join(difflib.unified_diff(original.decode("utf-8").splitlines(keepends=True),
                                     optimized.decode("utf-8").splitlines(keepends=True),
                                     fromfile="literal-binary-runtime/src/prom/prom_classification.py",
                                     tofile="binary-runtime/src/prom/prom_classification.py"))
    diff_path = WORK / "prom_official_v4_pooled_count_optimization.diff"
    diff_path.write_text(diff, encoding="utf-8")
    Direct, _, _ = bootstrap(before)
    Optimized, _, _ = bootstrap(RUNTIME)
    rng = np.random.RandomState(20261011)
    cases = []
    for ncal, ntest, k in ((11, 1, 1), (30, 7, 4), (85, 24, 8), (200, 200, 20)):
        scores = rng.uniform(-0.1, 1.2, (ncal, 1))
        test = rng.uniform(-0.1, 1.2, (ntest, 1))
        neighbors = rng.randint(0, ncal, size=(ntest, k))
        cases.append((f"random_{ncal}_{ntest}_{k}", scores, test, neighbors))
    tied = np.array([[0.0], [0.1], [0.1], [0.5], [0.5], [1.0], [1.0]])
    boundary = np.array([[0.0], [0.1], [np.nextafter(0.5, -np.inf)], [0.5], [np.nextafter(0.5, np.inf)], [1.0]])
    repeats = np.array([[0, 1, 1, 3, 4, 5, 6]] * len(boundary))
    cases.append(("repeated_ties_and_nextafter", tied, boundary, repeats))
    nonfinite = np.array([[-np.inf], [0.0], [0.0], [np.inf], [np.nan]])
    nonfinite_test = np.array([[-np.inf], [0.0], [1.0], [np.inf], [np.nan]])
    cases.append(("nan_infinity_semantics", nonfinite, nonfinite_test, np.array([[0, 1, 1, 2, 3, 4]] * 5)))
    cases.append(("empty_pool_semantics", tied, boundary, np.empty((len(boundary), 0), dtype=int)))
    count_parity = []
    for name, calibration, tests, selected in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = np.asarray(Direct(classes=[0, 1])._compute_p_values(tests, calibration, selected))
            b = np.asarray(Optimized(classes=[0, 1])._compute_p_values(tests, calibration, selected))
        same = bool(np.array_equal(a, b, equal_nan=True))
        if not same:
            raise AssertionError(f"Complete-pool count parity failed: {name}")
        count_parity.append({"case": name, "exact_array_equal": same, "original_pool_denominator": int(calibration[selected].size), "test_rows": len(tests)})
    full_parity = {}
    for nclasses, alpha in ((8, [0.025, 0.1, 0.9, 0.975]), (2, [0.02, 0.1, 0.9, 0.98])):
        x, y = make_classification(n_samples=1200, n_features=20, n_informative=16, n_redundant=0,
                                   n_classes=nclasses, n_clusters_per_class=1, random_state=621 + nclasses)
        classifier = GaussianNB().fit(x[:600], y[:600])
        a, _ = low_level(before, x[600:1000], y[600:1000], x[1000:], classifier, list(range(nclasses)), alphas=alpha)
        b, _ = low_level(RUNTIME, x[600:1000], y[600:1000], x[1000:], classifier, list(range(nclasses)), alphas=alpha)
        parity = {name: {field: bool(np.array_equal(a[name][field], b[name][field])) for field in a[name]} for name in a}
        if not all(all(v.values()) for v in parity.values()):
            raise AssertionError("Native/binary optimized full component parity failed")
        full_parity[str(nclasses)] = {"allowed_edge_alphas": alpha, "exact_parity": parity}
    # The bridge used in the current run rejects nonfinite probabilities.
    bridge_source = ROOT / "scripts/prom_official_v4.py"
    import ast
    tree = ast.parse(bridge_source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SealedProbabilityEstimator"]
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(bridge_source), "exec"), namespace)
    rejected = {}
    for name, value in (("nan", np.nan), ("positive_infinity", np.inf), ("negative_infinity", -np.inf)):
        try:
            namespace["SealedProbabilityEstimator"]().predict_proba(np.array([[value, 1.0 - value]]))
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError("Current finite-probability guard admitted a nonfinite value")
    result = {"schema": "trace-prom-pooled-count-execution-amendment-v4.2", "created_at_utc": utc(), "status": "passed",
              "scientific_protocol_changed": False, "target_labels_accessed": False,
              "original_official_source_sha256": sha(OFFICIAL / "src/prom/prom_classification.py"),
              "literal_binary_runtime_path": str(original_path.relative_to(ROOT)), "literal_binary_runtime_sha256": sha(original_path),
              "optimized_binary_runtime_path": str(runtime_path.relative_to(ROOT)), "optimized_binary_runtime_sha256": sha(runtime_path),
              "diff_path": str(diff_path.relative_to(ROOT)), "diff_sha256": sha(diff_path),
              "original_binary_repair_receipt_sha256": sha(OUT / "binary_repair.json"),
              "original_fidelity_smoke_sha256": sha(OUT / "fidelity_smoke.json"),
              "preserved_literal_files": preserved, "preserved_literal_target_cells": len(list((literal / "decisions").rglob("*.npz"))),
              "pooled_count_parity_cases": count_parity, "native_and_binary_full_parity": full_parity,
              "nonfinite_probability_rejection": rejected,
              "mathematical_operation": "Complete repeated global pool: (number >= score) / original total pool size; exact sorted rank count with side=left",
              "remaining_before_outcome_join": "Compare every literal source-grid and target archive against the complete optimized seal; bind exact root execution-clearance hashes"}
    write_json(OUT / "execution_optimization.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire", "prepare", "smoke", "optimize"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        result = {"acquire": acquire, "prepare": prepare_runtime, "smoke": smoke, "optimize": optimize}[args.command]()
    except Exception:
        path = OUT / f"{args.command}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(json.dumps({"command": args.command, "status": result.get("status", "completed"), "target_labels_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
