"""Check v2 aggregate arithmetic, frozen-input lineage and paper transcription.

Default: replay source-only calibrators/selectors from separately held sealed
inputs and check all three compiled PDFs. --aggregate-only skips raw replay;
--analysis-only skips paper/PDF checks. Skipped scopes are reported explicitly.
Passing this retrospective reproducibility check is not independent validation.
Original inputs and generated analysis outputs are read-only.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import binomtest, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vultriage.empirical_bootstrap import percentile_interval

KEYS = ["detector", "target_group", "seed", "method"]
PKEYS = ["detector", "target_group", "partition", "method"]
COUNTS = ["n", "n_vulnerable", "n_safe", "auto_safe", "auto_vulnerable", "auto_fn",
          "auto_fp", "review_count", "review_vulnerable", "review_safe", "empty_count", "singleton_count"]
PAIRS = [("unweighted_mondrian", "estimated_weight_no_gate"), ("msp_matched_weighted", "estimated_weight_no_gate"),
         ("msp_matched_gate", "vultriage_full_gate"), ("forced_argmax", "vultriage_full_gate")]
SUFFIXES = {"unweighted_mondrian": "unweighted_mondrian", "estimated_weight_no_gate": "weighted", "vultriage_full_gate": "gate"}
POLICIES = {
    "unweighted_mondrian": "Unweighted sets", "estimated_weight_no_gate": "Weighted sets",
    "vultriage_full_gate": "Weighted + frozen gate", "msp_matched_weighted": "MSP, weighted workload",
    "msp_matched_gate": "MSP, gate workload", "msp_matched_unweighted_mondrian": "MSP, unweighted workload",
    "class_matched_weighted": "Tails, weighted allocation", "class_matched_gate": "Tails, gate allocation",
    "class_matched_unweighted_mondrian": "Tails, unweighted allocation",
    "source_iso_cost10_matched_weighted": "Source cost 10, weighted workload",
    "source_iso_cost50_matched_weighted": "Source cost 50, weighted workload",
    "source_iso_cost10_matched_gate": "Source cost 10, gate workload",
    "source_iso_cost50_matched_gate": "Source cost 50, gate workload",
    "forced_argmax": "Forced argmax", "all_review": "All review",
}


class Audit:
    def __init__(self):
        self.checks, self.skipped, self.details = [], [], {}

    def require(self, value, label):
        if not bool(value):
            raise ValueError(label)
        self.checks.append(label)

    def same(self, actual, expected, label):
        self.require(np.allclose(np.asarray(actual, float), np.asarray(expected, float),
                                 atol=1e-12, rtol=1e-11, equal_nan=True), label)


@lru_cache(maxsize=None)
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(base, relative):
    base, path = base.resolve(), (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes expected root: {relative}")
    return path


def booleans(values):
    values = pd.Series(values).astype(str).str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError("Non-boolean provenance field")
    return values.eq("true").to_numpy()


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


def same_frame(audit, a, b, keys, columns, label):
    try:
        assert_frame_equal(a[keys + columns].sort_values(keys).reset_index(drop=True),
                           b[keys + columns].sort_values(keys).reset_index(drop=True),
                           check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
    except AssertionError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    audit.require(True, label)


def controls(costs):
    result = []
    for candidate, suffix in SUFFIXES.items():
        result.extend((f"{p}_matched_{suffix}", candidate, None) for p in ["msp", "class"])
        if suffix in ["weighted", "gate"]:
            result.extend((f"source_iso_cost{c}_matched_{suffix}", candidate, c) for c in costs)
    return result


def cost(frame, row):
    return (row.miss_cost * frame.auto_fn + row.false_alert_cost * frame.auto_fp + row.review_cost * frame.review_count
            + row.review_residual_error * (row.miss_cost * frame.review_vulnerable + row.false_alert_cost * frame.review_safe)) / frame.n


def receipts(audit, data, public, revision, control):
    release = read_json(data / "public_analysis_manifest.json")
    audit.require(release["upstream_manifests_unchanged"] is True, "unchanged upstream analysis manifests")
    audit.require(release["excluded_nonaggregate_outputs"] == {"revision": ["running_examples.json"], "controls": ["decisions/**"]},
                  "per-function example and selector archives explicitly excluded")
    neural = data / "neural_validation_v3"
    actual = {p.relative_to(data).as_posix() for p in data.rglob("*")
              if p.is_file() and not p.is_relative_to(neural)}
    audit.require(actual == set(release["files_sha256"]) | {"public_analysis_manifest.json"}, "legacy public analysis exact file whitelist")
    if neural.exists():
        neural_release = read_json(neural / "public_analysis_manifest.json")
        audit.require(neural_release["schema_version"] == "trace-v3-aggregate-release-1"
                      and neural_release["no_raw_datasets_predictions_weights_or_vaults"] is True,
                      "new neural subtree has its separate aggregate-only release contract")
        neural_actual = {p.relative_to(neural).as_posix() for p in neural.rglob("*") if p.is_file()}
        audit.require(neural_actual == set(neural_release["files_sha256"]) | {"public_analysis_manifest.json"},
                      "new neural public analysis exact file whitelist")
        for relative, expected in neural_release["files_sha256"].items():
            audit.require(digest(safe_path(neural, relative)) == expected, f"new neural released-byte binding: {relative}")
        audit.details["neural_aggregate_scope"] = "Released-byte verification only here; every new fit average and paired interval is independently reconstructed by check_neural_aggregate_evidence.py."
    for relative, expected in release["files_sha256"].items():
        audit.require(digest(safe_path(data, relative)) == expected, f"public aggregate hash: {relative}")
    audit.require(not any(p.suffix.lower() in {".npz", ".gz"} for p in data.rglob("*")), "no raw archives in public analysis")
    for relative, expected in release["analysis_source_sha256"].items():
        audit.require(digest(safe_path(ROOT, relative)) == expected, f"public analysis source binding: {relative}")
    for sub, manifest, key in [("", revision, "output_sha256"), ("controls", control, "output_table_sha256")]:
        audit.require("analysis_manifest.json" not in manifest[key], f"{sub or 'revision'} no self-referential output digest")
        for name, expected in manifest[key].items():
            if name == "running_examples.json" and not sub:
                audit.require(not (data / name).exists(), "nonaggregate running examples excluded")
                continue
            audit.require(Path(name).name == name and name.endswith(".csv"), f"safe aggregate filename: {sub}/{name}")
            audit.require(digest(data / sub / name) == expected, f"upstream aggregate hash: {sub}/{name}")
    for relative, expected in [
        ("src/analyze_neurocomputing_revision.py", revision["script_sha256"]),
        ("src/analyze_neurocomputing_controls.py", control["script_sha256"]),
        ("src/vultriage/empirical_bootstrap.py", revision["empirical_bootstrap_helper_sha256"]),
    ]:
        audit.require(digest(ROOT / relative) == expected, f"execution-bound source hash: {relative}")
    audit.require(control["independent_validation"] is False and control["preregistered"] is False and control["detector_refits"] == 0,
                  "controls declared retrospective without detector refits")
    original = read_json(public / "evaluation-v2/evaluation_manifest.json")
    audit.require(digest(public / "evaluation-v2/evaluation_manifest.json") == revision["original_manifest_sha256"]
                  == control["original_evaluation_manifest_sha256"], "same sealed original evaluation manifest")
    audit.require(digest(public / "evaluation-v2/fold_seed_metrics.csv") == original["metrics_sha256"]
                  == revision["original_metrics_sha256"] == control["original_metrics_sha256"], "sealed original aggregate metrics")
    audit.require(original["target_labels_joined_after_prediction_seal"] is True, "original post-seal label-join declaration")
    for relative, expected in revision["input_sha256"].items():
        audit.require(digest(safe_path(public, relative)) == expected, f"auxiliary input hash: {relative}")
    contract = read_json(public / "contract-v1/contract_manifest.json")
    for name in ["qualification_summary.csv", "fixed_split_projects.csv"]:
        audit.require(digest(public / "contract-v1" / name) == contract["outputs"][name]["sha256"], f"contract lineage: {name}")
    seal = read_json(public / "gate-v1/gate_seal.json")
    audit.require(digest(public / "gate-v1/primevul_gate_development.csv") == seal["development_csv_sha256"], "gate-development lineage")
    audit.require(control["contract_split_sha256"] == revision["input_sha256"]["contract-v1/fixed_split_projects.csv"], "same contract split")
    audit.require(control["cohorts"]["technical_seed_addresses"] == original["seeds"]
                  and control["cohorts"]["seed_addresses_averaged_before_bootstrap"] is True, "technical-seed averaging declaration")
    n = len(control["cohorts"]["detectors"]) * len(original["target_groups"]) * len(original["seeds"])
    audit.require(len(original["decision_files"]) == revision["verified_decision_archives"] == n, "complete original decision inventory")
    audit.require(original["decision_files"] == revision["verified_archive_sha256"] == control["verified_decision_archive_sha256"],
                  "original decision hashes agree across three manifests")
    audit.require(len(control["verified_source_fold_sha256"]) == 2 * len(original["target_groups"]), "both source and target-position fold receipts")
    audit.require(len(control["verified_prediction_file_sha256"]) == 2 * n, "NPZ/metadata prediction inventory")
    audit.details.update(aggregate_files=len(release["files_sha256"]), projects=len(original["target_groups"]),
                         detectors=control["cohorts"]["detectors"], technical_seed_addresses=original["seeds"])
    return original


def action_metrics(audit, seed, projects, summary, original, label):
    audit.require(not seed.duplicated(KEYS).any() and not projects.duplicated(PKEYS).any(), f"{label}: unique policy rows")
    expected = {(d, g, int(s), m) for d in seed.detector.unique() for g in original["target_groups"] for s in original["seeds"] for m in seed.method.unique()}
    audit.require(set(map(tuple, seed[KEYS].itertuples(index=False, name=None))) == expected, f"{label}: complete policy cohort")
    a = seed[COUNTS].to_numpy(float)
    audit.require(np.isfinite(a).all() and (a >= 0).all() and np.array_equal(a, np.floor(a)), f"{label}: valid seeded integer counts")
    for actual, expected, endpoint in [
        (seed.n_vulnerable + seed.n_safe, seed.n, "true-class counts"),
        (seed.auto_safe + seed.auto_vulnerable + seed.review_count, seed.n, "action partition"),
        (seed.singleton_count, seed.auto_safe + seed.auto_vulnerable, "automatic count"),
        (seed.review_vulnerable + seed.review_safe, seed.review_count, "reviewed-class counts"),
        (seed.auto_fn + seed.auto_vulnerable - seed.auto_fp + seed.review_vulnerable, seed.n_vulnerable, "vulnerable confusion counts"),
        (seed.auto_fp + seed.auto_safe - seed.auto_fn + seed.review_safe, seed.n_safe, "safe confusion counts"),
    ]:
        audit.same(actual, expected, f"{label}: {endpoint}")
    audit.require((seed.auto_fn <= seed.auto_safe).all() and (seed.auto_fp <= seed.auto_vulnerable).all()
                  and (seed.empty_count <= seed.review_count).all(), f"{label}: error and empty-set bounds")
    for endpoint, value in {
        "coverage": ratio(seed.singleton_count, seed.n), "review_load": ratio(seed.review_count, seed.n),
        "auto_fn_rate": ratio(seed.auto_fn, seed.n_vulnerable), "auto_fp_rate": ratio(seed.auto_fp, seed.n_safe),
        "selective_error": ratio(seed.auto_fn + seed.auto_fp, seed.singleton_count),
        "selective_vulnerable_error": ratio(seed.auto_fn, seed.n_vulnerable - seed.review_vulnerable),
        "selective_safe_error": ratio(seed.auto_fp, seed.n_safe - seed.review_safe),
        "safe_release_contamination": ratio(seed.auto_fn, seed.auto_safe),
        "vulnerable_alert_precision": ratio(seed.auto_vulnerable - seed.auto_fp, seed.auto_vulnerable),
    }.items():
        audit.same(seed[endpoint], value, f"{label}: {endpoint} arithmetic, zero denominator stays NaN")
    numeric = [c for c in seed.select_dtypes(include="number").columns if c != "seed"]
    expected = seed.groupby(PKEYS, sort=True)[numeric].mean().reset_index()
    same_frame(audit, projects, expected, PKEYS, numeric, f"{label}: project technical-seed means")
    expected = projects.groupby(["detector", "method"], sort=True)[numeric].median().reset_index()
    same_frame(audit, summary, expected, ["detector", "method"], numeric, f"{label}: separate project medians")


def matching(audit, frame, seed, fits, control, original):
    identity = ["detector", "target_group", "seed"]
    audit.require(not frame.duplicated(identity + ["reference", "control"]).any(), "unique matched-control invariant rows")
    expected_pairs = {(c, b, m) for b, c, m in controls(control["cost_scenarios"]["miss"])}
    expected_ids = {(d, g, int(s)) for d in control["cohorts"]["detectors"] for g in original["target_groups"] for s in original["seeds"]}
    audit.require(set(map(tuple, frame[identity].itertuples(index=False, name=None))) == expected_ids, "complete invariant identity cohort")
    index = seed.set_index(KEYS)
    for key, sub in frame.groupby(identity, sort=True):
        observed = {(r.reference, r.control, None if pd.isna(r.source_ranking_miss_cost) else int(r.source_ranking_miss_cost)) for r in sub.itertuples()}
        audit.require(observed == expected_pairs, f"control inventory: {key}")
        for r in sub.itertuples():
            a, b = index.loc[(*key, r.reference)], index.loc[(*key, r.control)]
            audit.same([r.reference_auto_safe, r.reference_auto_vulnerable, r.control_auto_safe, r.control_auto_vulnerable],
                       [a.auto_safe, a.auto_vulnerable, b.auto_safe, b.auto_vulnerable], f"action counts: {key}/{r.control}")
            audit.same([a.singleton_count, a.review_count, r.matched_automatic_count, r.matched_review_count],
                       [b.singleton_count, b.review_count, a.singleton_count, a.review_count], f"exact matched workload: {key}/{r.control}")
            allocation = a.auto_safe == b.auto_safe and a.auto_vulnerable == b.auto_vulnerable
            audit.require(booleans([r.exact_action_allocation])[0] == allocation, f"allocation flag: {key}/{r.control}")
            if r.control.startswith("class_matched_"):
                audit.require(allocation, f"both label counts matched: {key}/{r.control}")
            audit.require(0 <= r.safe_selected_overlap <= min(a.auto_safe, b.auto_safe)
                          and 0 <= r.vulnerable_selected_overlap <= min(a.auto_vulnerable, b.auto_vulnerable)
                          and r.safe_selected_overlap + r.vulnerable_selected_overlap <= r.automatic_selected_overlap <= a.singleton_count,
                          f"overlap bounds: {key}/{r.control}")
            if booleans([r.actions_identical])[0]:
                audit.require(r.safe_selected_overlap == a.auto_safe and r.vulnerable_selected_overlap == a.auto_vulnerable,
                              f"identical-action overlap: {key}/{r.control}")
    audit.require(not fits.duplicated(identity).any()
                  and set(map(tuple, fits[identity].itertuples(index=False, name=None))) == expected_ids,
                  "one source isotonic map per identity, reused by both cost ratios")
    audit.require(not booleans(fits.target_labels_in_fit).any(), "source-fit no-target-label declaration")
    audit.same(fits.source_calibration_vulnerable + fits.source_calibration_safe, fits.source_calibration_n, "source-class counts")
    audit.require((fits.source_calibration_n > 0).all() and fits.x_knots.between(1, fits.source_calibration_n).all(), "source fit/knot counts")
    audit.require(all(fits[c].astype(str).str.fullmatch(r"[0-9A-F]{64}").all() for c in fits if c.endswith("_sha256")), "source-fit SHA-256 fields")
    for r in fits.itertuples():
        audit.require(r.source_package_sha256 == control["verified_source_fold_sha256"][f"source_label_packages/{r.target_group}.npz"]
                      and r.prediction_npz_sha256 == control["verified_prediction_file_sha256"][f"{r.detector}/predictions/{r.target_group}/seed-{r.seed}.npz"],
                      f"source/prediction fit binding: {r.detector}/{r.target_group}/{r.seed}")
    audit.details.update(selection_invariant_rows=len(frame), source_fit_receipts=len(fits))


def paired(audit, rows, project, label, new=False):
    for r in rows.itertuples():
        frame = project[project.detector == r.detector].copy()
        if r.outcome == "cost":
            frame["cost"] = cost(frame, r)
        a, b = [frame[frame.method == m].set_index("target_group")[r.outcome] for m in [r.baseline, r.candidate]]
        audit.require(a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index), f"{label}: paired cohort/{r.Index}")
        delta = (b.reindex(a.index) - a).to_numpy(float)
        audit.require(np.isfinite(delta).all() and len(delta) == r.projects, f"{label}: finite project endpoint/{r.Index}")
        audit.same([r.median_difference, r.ci_lower, r.ci_upper], [np.median(delta), *percentile_interval(delta)], f"{label}: exact interval/{r.Index}")
        low, high = int((delta < -1e-12).sum()), int((delta > 1e-12).sum())
        if new:
            direction = "higher" if r.outcome == "coverage" else "lower"
            audit.require(r.difference_definition == "candidate_minus_baseline" and r.favorable_direction == direction, f"{label}: estimator/direction/{r.Index}")
            favorable, unfavorable = (high, low) if direction == "higher" else (low, high)
            audit.same([r.mean_difference, r.favorable_projects, r.unfavorable_projects, r.ties],
                       [np.mean(delta), favorable, unfavorable, len(delta) - low - high], f"{label}: mean/directions/{r.Index}")
        else:
            p = binomtest(low, low + high).pvalue if low + high else np.nan
            audit.same([r.wins, r.losses, r.ties, r.descriptive_sign_p], [low, high, len(delta) - low - high, p], f"{label}: sign counts/test/{r.Index}")


def inventories(audit, revision, control, frames):
    ds, grid, scenarios = control["cohorts"]["detectors"], revision["cost_grid"], control["cost_scenarios"]
    inventory = [
        ("revision/action_paired_comparisons", ["detector", "outcome", "baseline", "candidate"],
         {(d, o, b, c) for d in ds for o in ["coverage", "auto_fn_rate", "auto_fp_rate"] for b, c in PAIRS}),
        ("controls/matched_paired_comparisons", ["detector", "outcome", "baseline", "candidate"],
         {(d, o, b, c) for d in ds for o in ["coverage", "auto_fn_rate", "auto_fp_rate", "review_load"] for b, c, _ in controls(scenarios["miss"])}),
    ]
    columns = ["detector", "miss_cost", "false_alert_cost", "review_cost", "review_residual_error", "baseline", "candidate"]
    inventory += [
        ("revision/cost_paired_sensitivity", columns,
         {(d, float(m), float(grid["false_alert"]), float(r), float(e), b, c)
          for d, m, r, e, (b, c) in itertools.product(ds, grid["miss"], grid["review"], grid["review_residual_error"], PAIRS)}),
        ("controls/matched_cost_comparisons", columns,
         {(d, float(m), float(scenarios["false_alert"]), float(scenarios["review"]), float(scenarios["review_residual_error"]), b, c)
          for d in ds for m in scenarios["miss"] for b, c, fixed in controls(scenarios["miss"]) if fixed is None or fixed == m}),
    ]
    for name, cols, expected in inventory:
        rows = frames[name]
        audit.require(set(map(tuple, rows[cols].itertuples(index=False, name=None))) == expected and len(rows) == len(expected), f"complete endpoint inventory: {name}")


def gate_stats(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    return [(rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0),
            np.mean((s - y) ** 2), y.mean(), np.mean((y - y.mean()) ** 2)]


def gates(audit, scores, summary, public, revision, original):
    audit.require(not scores.duplicated(["detector", "target_group"]).any()
                  and scores.groupby("detector").target_group.nunique().eq(len(original["target_groups"])).all(), "complete gate project cohort")
    audit.require(scores.score.between(0, 1).all() and scores.severe.isin([0, 1]).all(), "valid gate scores/labels")
    rng = np.random.default_rng(revision["seed"])
    for detector, sub in scores.groupby("detector", sort=True):
        r = summary[(summary.domain == "DiverseVul") & (summary.detector == detector)].iloc[0]
        y, s = sub.severe.to_numpy(int), sub.score.to_numpy(float)
        audit.same(r[["auroc", "brier", "prevalence", "prevalence_brier"]], gate_stats(y, s), f"gate metrics: {detector}")
        ix = rng.integers(0, len(sub), (revision["gate_bootstrap_replicates"], len(sub)))
        yy, ss = y[ix], s[ix]
        n1, n = yy.sum(axis=1), len(sub)
        valid = (n1 > 0) & (n1 < n)
        sums = (rankdata(ss[valid], axis=1) * yy[valid]).sum(axis=1)
        auc = (sums - n1[valid] * (n1[valid] + 1) / 2) / (n1[valid] * (n - n1[valid]))
        audit.same([r.auc_ci_lower, r.auc_ci_upper, r.valid_bootstrap_samples], [*np.quantile(auc, [.025, .975]), valid.sum()],
                   f"gate bootstrap intervals and valid denominator: {detector}")
    dev, r = pd.read_csv(public / "gate-v1/primevul_gate_development.csv"), summary[summary.domain == "PrimeVul crossfit"].iloc[0]
    audit.same(r[["auroc", "brier", "prevalence", "prevalence_brier"]], gate_stats(dev.severe_violation, dev.crossfit_severe_probability), "source cross-fit gate audit")
    audit.require(r.projects == dev.target_group.nunique() and r.rows == len(dev), "source gate denominators")


def raw_replay(audit, args, control, revision, fits, seeded):
    from analyze_neurocomputing_controls import (
        OPERATING_KEY, REFERENCES, action_codes, array_sha256, build_controls,
        fit_source_isotonic, load_source_labels, verify_frozen_inputs,
    )
    from analyze_neurocomputing_revision import action_counts
    v = verify_frozen_inputs(args.evaluation, args.public_root, args.predictions_root, args.inputs, args.config, args.runner)
    for field, value in v["receipts"].items():
        audit.require(value == control[field], f"local frozen-input lineage: {field}")
    audit.require(v["receipts"]["verified_decision_archive_sha256"] == revision["verified_archive_sha256"], "revision hashes match local original bytes")
    source = {g: load_source_labels(v["package_paths"][(g, "source_label_package")], v["package_paths"][(g, "target_position_package")]) for g in v["manifest"]["target_groups"]}
    expected = control["generated_decision_archive_sha256"]
    actual = {p.relative_to(args.controls_output).as_posix() for p in (args.controls_output / "decisions").rglob("*") if p.is_file()}
    audit.require(actual == set(expected), "complete local selector archive inventory")
    fit_index, metric_index, compared = fits.set_index(["detector", "target_group", "seed"]), seeded.set_index(KEYS), 0
    for relative, expected_hash in sorted(expected.items()):
        path = safe_path(args.controls_output, relative)
        audit.require(digest(path) == expected_hash, f"local selector archive hash: {relative}")
        detector, group = path.parts[-3:-1]
        seed = int(path.stem.split("-")[-1])
        fit, (cp, cy, tp) = fit_index.loc[(detector, group, seed)], source[group]
        with np.load(v["prediction_paths"][(detector, group, seed)], allow_pickle=False) as prediction:
            audit.require(np.array_equal(prediction["calibration_positions"], cp) and np.array_equal(prediction["target_positions"], tp), f"source-fit row lineage: {relative}")
            cal_p, p = prediction["calibration_p_vulnerable"].astype(float), prediction["target_p_vulnerable"].astype(float)
        calibrator = fit_source_isotonic(cal_p, cy)
        eta = calibrator.predict(p)
        arrays = {"calibration_positions_sha256": cp, "calibration_labels_sha256": cy,
                  "calibration_probability_sha256": cal_p, "target_positions_sha256": tp,
                  "target_probability_sha256": p, "x_thresholds_sha256": calibrator.X_thresholds_,
                  "y_thresholds_sha256": calibrator.y_thresholds_, "calibrated_target_scores_sha256": eta,
                  **{f"cost{c}_ranking_sha256": np.argsort(np.minimum(c * eta, 1 - eta), kind="stable") for c in control["cost_scenarios"]["miss"]}}
        audit.require(all(array_sha256(a) == fit[k] for k, a in arrays.items()), f"source-only fit and both rankings replay: {relative}")
        audit.same([fit.source_calibration_n, fit.source_calibration_vulnerable, fit.source_calibration_safe, fit.x_knots],
                   [len(cy), cy.sum(), (cy == 0).sum(), len(calibrator.X_thresholds_)], f"source-fit count replay: {relative}")
        with np.load(safe_path(args.evaluation, relative), allow_pickle=False) as archive:
            audit.require(np.array_equal(archive["target_positions"], tp) and np.array_equal(archive["target_p_vulnerable"], p), f"prediction/decision lineage: {relative}")
            references = {name: archive[OPERATING_KEY + name] for name in REFERENCES}
            rebuilt = build_controls(p, references, eta)  # Selectors fixed before reading target labels.
            labels = archive["target_labels"].copy()
        with np.load(path, allow_pickle=False) as stored:
            audit.require("labels" not in stored.files and "target_labels" not in stored.files, f"generated selector excludes labels: {relative}")
            audit.require(np.array_equal(stored["target_positions"], tp) and np.array_equal(stored["source_isotonic_eta"], eta), f"selector score lineage: {relative}")
            for method, sets in rebuilt.items():
                audit.require(np.array_equal(stored[method], sets), f"label-free selector replay: {relative}/{method}")
                codes, row = action_codes(sets), metric_index.loc[(detector, group, seed, method)]
                audit.require(array_sha256(codes) == row.action_sha256, f"aggregate action digest: {relative}/{method}")
                counts = action_counts(labels, sets)
                audit.same(row[list(counts)], list(counts.values()), f"raw action endpoints: {relative}/{method}")
                nonargmax = ((codes == 0) & (p > .5)) | ((codes == 1) & (p <= .5))
                audit.same([row.non_argmax_count, row.non_argmax_rate], [nonargmax.sum(), nonargmax.mean()], f"non-argmax allocation: {relative}/{method}")
                compared += 1
    audit.details["local_input_files_verified"] = {
        "original_decisions": len(v["receipts"]["verified_decision_archive_sha256"]),
        "predictions_npz_and_metadata": len(v["receipts"]["verified_prediction_file_sha256"]),
        "source_and_target_position_packages": len(v["receipts"]["verified_source_fold_sha256"]),
        "generated_selector_archives": len(expected), "reconstructed_policy_seed_rows": compared,
    }
    audit.details["source_fit_reconstruction"] = "Source-only isotonic maps and selectors replayed with the receipt-bound supplied implementation; no detector training. This is implementation replay, not independently written selector verification."


def table(source, label, end=r"\end{tabularx}"):
    return source.split(r"\label{" + label + "}", 1)[1].split(end, 1)[0]


def texrows(source):
    return [[cell.strip().removesuffix(r"\\").strip() for cell in line.split("&")] for line in source.splitlines() if "&" in line and line.rstrip().endswith(r"\\")]


def rounded(audit, actual, expected, digits, label):
    audit.same(actual, [float(f"{v:.{digits}f}") for v in expected], label)


def documents(audit, paper, frames, control):
    main, si, sitables, cover = [(paper / f).read_text(encoding="utf-8") for f in ["main.tex", "supplementary.tex", "supplementary_tables.tex", "cover_letter.tex"]]
    historical_displays = "\n".join([main, si, sitables])
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S).group(1)
    keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main, re.S).group(1).split(r"\sep")
    highlights = [s for s in (paper / "highlights.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    audit.require(len(abstract.split()) <= 250, "abstract <=250 whitespace-delimited words")
    audit.require(1 <= len(keywords) <= 7 and all(s.strip() for s in keywords), "one to seven keywords")
    audit.require(3 <= len(highlights) <= 5 and max(map(len, highlights)) <= 85, "three to five highlights, <=85 characters each")
    title = re.search(r"\\title\{([^}]+)\}", main).group(1)
    audit.require(title in cover and "10 October 2026" in cover, "cover uses current title and intended revision date")
    audit.require("supplementary" in main.lower() and "isotonic" in si and "empirical" in si, "SI cited and describes new controls/intervals")
    cited = set()
    for source in [main, si, sitables]:
        for group in re.findall(r"\\cite(?:[a-zA-Z]*)?\{([^}]+)\}", source):
            cited.update(k.strip() for k in group.split(","))
        for f in re.findall(r"\\input\{([^}]+)\}", source):
            audit.require((paper / (f if f.endswith(".tex") else f + ".tex")).is_file(), f"editable TeX input: {f}")
        for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
            audit.require((paper / f).is_file() or (paper / "figures" / f).is_file(), f"figure source: {f}")
    defined = set(re.findall(r"@\w+\{\s*([^,]+),", (paper / "references.bib").read_text(encoding="utf-8")))
    audit.require(cited <= defined, "main/SI citations defined (structural only)")
    names = {"Unweighted sets": "unweighted_mondrian", "Weighted sets": "estimated_weight_no_gate", "MSP matched to weighted": "msp_matched_weighted",
             "Weighted + frozen gate": "vultriage_full_gate", "MSP matched to gate": "msp_matched_gate"}
    det, found, summary = None, set(), frames["revision/action_summary"].set_index(["detector", "method"])
    for cells in texrows(table(historical_displays, "tab:actions")):
        if len(cells) != 6 or cells[1] not in names:
            continue
        if cells[0]:
            det = {"Hashing": "hashing", "CodeBERT": "codebert"}[cells[0]]
        method = names[cells[1]]
        rounded(audit, list(map(float, cells[2:])), summary.loc[(det, method)][["coverage", "auto_fn_rate", "auto_fp_rate", "selective_error"]], 3, f"main action table: {det}/{method}")
        found.add((det, method))
    audit.require(found == {(d, m) for d in control["cohorts"]["detectors"] for m in names.values()}, "all main action rows checked")
    gatesummary, found = frames["revision/gate_audit"], set()
    for cells in texrows(table(historical_displays, "tab:gate")):
        if len(cells) != 5 or cells[0] not in ["PrimeVul cross-fit", "DiverseVul hashing", "DiverseVul CodeBERT"]:
            continue
        r = (gatesummary[gatesummary.domain == "PrimeVul crossfit"].iloc[0] if cells[0].startswith("PrimeVul") else
             gatesummary[(gatesummary.domain == "DiverseVul") & (gatesummary.detector == ("hashing" if cells[0].endswith("hashing") else "codebert"))].iloc[0])
        audit.require(int(cells[1]) == r.projects, f"gate table cohort: {cells[0]}")
        rounded(audit, list(map(float, cells[2:])), r[["auroc", "brier", "prevalence_brier"]], 3, f"gate table: {cells[0]}")
        found.add(cells[0])
    audit.require(len(found) == 3, "all gate table rows checked")
    rows, det, found = texrows(table(historical_displays, "tab:costcontrols")), None, set()
    for i, cells in enumerate(rows):
        if len(cells) != 4 or cells[1] not in ["Global MSP", "Allocation-matched tails", "Source-calibrated cost"]:
            continue
        if cells[0]:
            det = {"Hashing": "hashing", "CodeBERT": "codebert"}[cells[0]]
        audit.require(rows[i + 1][1] == r"95\% interval", "cost interval follows point estimates")
        for j, miss in enumerate(control["cost_scenarios"]["miss"]):
            baseline = {"Global MSP": "msp_matched_weighted", "Allocation-matched tails": "class_matched_weighted", "Source-calibrated cost": f"source_iso_cost{miss}_matched_weighted"}[cells[1]]
            costs = frames["controls/matched_cost_comparisons"]
            r = costs[(costs.detector == det) & (costs.baseline == baseline) & (costs.candidate == "estimated_weight_no_gate") & (costs.miss_cost == miss)].iloc[0]
            rounded(audit, [float(cells[j + 2].strip("$"))], [r.median_difference], 4, f"cost estimate: {det}/{baseline}/{miss}")
            rounded(audit, [float(v) for v in re.findall(r"[+-]?\d+\.\d+", rows[i + 1][j + 2])], [r.ci_lower, r.ci_upper], 4, f"cost exact interval: {det}/{baseline}/{miss}")
            found.add((det, cells[1], miss))
    audit.require(len(found) == len(control["cohorts"]["detectors"]) * 3 * len(control["cost_scenarios"]["miss"]), "all main cost-control endpoints")
    projects = frames["controls/action_metrics_project"]
    cohort = projects[(projects.detector == "hashing") & (projects.method == "estimated_weight_no_gate")].set_index("target_group")
    found = set()
    for cells in texrows(table(sitables, "si:cohort", r"\end{longtable}")):
        if len(cells) == 5 and cells[0] in cohort.index:
            r = cohort.loc[cells[0]]
            audit.same(list(map(int, cells[1:4])), r[["n", "n_vulnerable", "n_safe"]], f"SI cohort counts: {cells[0]}")
            audit.require(cells[4] == ("Reg." if r.partition == "registration" else "Conf."), f"SI partition: {cells[0]}")
            found.add(cells[0])
    audit.require(found == set(cohort.index), "every SI cohort project")
    names, det, found = {v: k for k, v in POLICIES.items()}, None, set()
    for cells in texrows(table(sitables, "si:allactions", r"\end{longtable}")):
        if len(cells) != 7 or cells[1] not in names:
            continue
        if cells[0]:
            det = {"Hashing": "hashing", "CodeBERT": "codebert"}[cells[0]]
        method = names[cells[1]]
        sub = projects[(projects.detector == det) & (projects.method == method)]
        rounded(audit, [np.nan if s == "--" else float(s) for s in cells[2:6]], sub[["coverage", "auto_fn_rate", "auto_fp_rate", "selective_error"]].median(), 3, f"SI policy medians: {det}/{method}")
        audit.require(int(cells[6]) == sub.selective_error.notna().sum(), f"SI conditional denominator: {det}/{method}")
        found.add((det, method))
    audit.require(found == {(d, m) for d in control["cohorts"]["detectors"] for m in POLICIES}, "every SI policy and conditional denominator")
    audit.details.update(abstract_words=len(abstract.split()), keyword_count=len(keywords), highlight_lengths=list(map(len, highlights)), citation_keys=len(cited), manuscript_title=title)
    audit.details["source_sha256"] = {f: digest(paper / f) for f in ["main.tex", "cover_letter.tex", "supplementary.tex", "supplementary_tables.tex", "references.bib", "highlights.txt"]}


def pdfs(audit, paper):
    import pymupdf
    result = {}
    for name in ["main", "cover_letter", "supplementary"]:
        log, tex = (paper / f"{name}.log").read_text(encoding="utf-8", errors="replace"), (paper / f"{name}.tex").read_text(encoding="utf-8")
        audit.require(not re.search(r"Overfull|(?:Citation|Reference).*undefined|There were undefined|^! ", log, re.M), f"{name}: no overfull, undefined references or TeX errors")
        inputs = [paper / f"{name}.tex"]
        if name != "cover_letter":
            inputs += [paper / "references.bib", paper / "elsarticle.cls"]
        for f in re.findall(r"\\input\{([^}]+)\}", tex):
            inputs.append(paper / (f if f.endswith(".tex") else f + ".tex"))
        for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
            inputs.append(paper / f if (paper / f).is_file() else paper / "figures" / f)
        path = paper / f"{name}.pdf"
        audit.require(path.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in inputs), f"{name}: PDF newer than editable inputs")
        with pymupdf.open(path) as doc:
            audit.require(len(doc) > 0, f"{name}: PDF contains pages")
            if name == "cover_letter":
                audit.require(len(doc) == 1, "one-page cover letter")
            if name == "main":
                norm = lambda t: re.sub(r"\s+", "", unicodedata.normalize("NFKC", t)).casefold()
                audit.require(norm(audit.details["manuscript_title"]) in norm(doc[0].get_text()), "compiled PDF current manuscript title")
            result[name] = {"pages": len(doc), "sha256": digest(path)}
    audit.details["pdfs"] = result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = ROOT.parent / "crc_cpvd_fcs/outputs/extension-v2"
    for name, default in [
        ("paper", ROOT / "paper/neurocomputing_revision"), ("public-root", ROOT / "public_results/extension-v2"),
        ("evaluation", experiment / "evaluation-v2"), ("predictions-root", experiment / "predictions"), ("inputs", experiment / "source-v2"),
        ("controls-output", ROOT / "outputs/neurocomputing-controls-v2"), ("config", ROOT / "configs/preregistered_extension_v2.json"),
        ("runner", ROOT / "src/run_extension_predict.py"), ("report", ROOT / "tmp/neurocomputing-review-v2/reproducibility_checks.json"),
    ]:
        parser.add_argument("--" + name, type=Path, default=default)
    parser.add_argument("--aggregate-only", action="store_true", help="skip raw inputs and selector replay")
    parser.add_argument("--analysis-only", action="store_true", help="skip manuscript and PDFs during editing")
    parser.add_argument("--skip-pdf", action="store_true", help="check document sources but skip compiled PDFs")
    args = parser.parse_args()
    audit, data = Audit(), args.paper / "analysis"
    revision, control = read_json(data / "analysis_manifest.json"), read_json(data / "controls/analysis_manifest.json")
    original = receipts(audit, data, args.public_root, revision, control)
    frames = {}
    for label, directory in [("revision", data), ("controls", data / "controls")]:
        for p in directory.glob("*.csv"):
            frames[f"{label}/{p.stem}"] = pd.read_csv(p)
        action_metrics(audit, frames[f"{label}/action_metrics_seed"], frames[f"{label}/action_metrics_project"], frames[f"{label}/action_summary"], original, label)
    audit.require(len(frames["revision/action_metrics_seed"]) == revision["metric_rows"] and len(frames["revision/action_metrics_project"]) == revision["project_method_rows"], "revision receipt row counts")
    for field, filename in {"seed_metrics": "action_metrics_seed", "project_metrics": "action_metrics_project", "paired_outcomes": "matched_paired_comparisons", "cost_comparisons": "matched_cost_comparisons", "source_fits": "source_calibration_fits"}.items():
        audit.require(len(frames[f"controls/{filename}"]) == control["rows"][field], f"control receipt row count: {field}")
    rs, cs = frames["revision/action_metrics_seed"], frames["controls/action_metrics_seed"]
    audit.require(set(cs.method) == set(POLICIES), "all declared control policies present")
    shared, numeric = set(rs.method) & set(cs.method), [c for c in rs.select_dtypes(include="number").columns if c != "seed"]
    same_frame(audit, cs[cs.method.isin(shared)], rs[rs.method.isin(shared)], KEYS, numeric, "shared original/MSP endpoints agree across v2 analyses")
    qualification = pd.read_csv(args.public_root / "contract-v1/qualification_summary.csv")
    qualified = qualification[qualification.partition == "registration"].set_index("detector").qualified
    for d in control["cohorts"]["detectors"]:
        m = "vultriage_full_gate" if booleans([qualified.loc[d]])[0] else "all_review"
        expected = rs[(rs.detector == d) & (rs.method == m)].copy()
        expected["method"] = "registered_controller"
        same_frame(audit, rs[(rs.detector == d) & (rs.method == "registered_controller")], expected, KEYS, numeric, f"qualified-controller routing: {d}")
    matching(audit, frames["controls/selection_invariants_seed"], cs, frames["controls/source_calibration_fits"], control, original)
    inventories(audit, revision, control, frames)
    for key, new in [("revision/action_paired_comparisons", False), ("revision/cost_paired_sensitivity", False), ("controls/matched_paired_comparisons", True), ("controls/matched_cost_comparisons", True)]:
        paired(audit, frames[key], frames[key.split("/")[0] + "/action_metrics_project"], key, new)
    rp = frames["revision/action_metrics_project"].copy()
    rp["cost"] = (10 * rp.auto_fn + rp.auto_fp + .1 * rp.review_count) / rp.n
    same_frame(audit, frames["revision/nominal_cost_summary"], rp.groupby(["detector", "method"]).cost.median().reset_index(), ["detector", "method"], ["cost"], "nominal cost summary arithmetic")
    cp = frames["controls/action_metrics_project"]
    for r in frames["controls/cost_summary"].itertuples():
        sub = cp[(cp.detector == r.detector) & (cp.method == r.method)]
        values = cost(sub, r)
        audit.same([r.projects, r.median_cost, r.mean_cost], [len(sub), values.median(), values.mean()], f"control cost summary: {r.detector}/{r.method}/{r.miss_cost}")
    gates(audit, frames["revision/gate_project_scores"], frames["revision/gate_audit"], args.public_root, revision, original)
    if args.aggregate_only:
        audit.skipped.append("Raw-input bytes, source-calibrator/selector replay and raw action endpoints: aggregate-only mode.")
    else:
        raw_replay(audit, args, control, revision, frames["controls/source_calibration_fits"], cs)
    if args.analysis_only:
        audit.skipped.append("Manuscript, cover, highlights, SI transcription, editable dependencies and all PDFs: analysis-only mode.")
    else:
        documents(audit, args.paper, frames, control)
        if args.skip_pdf:
            audit.skipped.append("Main, cover and supplementary compiled PDFs: skip-pdf mode.")
        else:
            pdfs(audit, args.paper)
    report = {
        "scope": "V2 aggregate arithmetic, byte lineage, supplied-implementation replay, manuscript transcription and compilation",
        "status": "PASS_WITH_SKIPPED_CHECKS" if audit.skipped else "PASS",
        "limitations": [
            "Post-review analyses remain retrospective; passing checks/hashes do not create independent validation or preregistration.",
            "Byte identity and metadata declarations do not independently establish absence of all historical leakage or tuning channels.",
            "Median intervals use the receipt-bound helper, not an independently written bootstrap implementation.",
            "Public aggregates cannot reconstruct selectors without separately available source labels, sealed predictions and target evaluation labels.",
            "Citation keys and PDF compilation are structural; live reference verification and visual layout QA are separate tasks.",
        ],
        "skipped_checks": audit.skipped, "check_count": len(audit.checks), "checks": audit.checks, "details": audit.details,
        "checker_sha256": digest(Path(__file__)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["check_count"], "skipped_checks": audit.skipped,
                      "details": audit.details, "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
