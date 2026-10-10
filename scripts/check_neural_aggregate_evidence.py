"""Independently reconstruct v3 fit averages and every paired project interval.

This aggregate-only checker works without model weights, raw functions,
predictions or target-label vaults.  It resamples complete projects within
each dataset and preserves undefined endpoints and technical review fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = {"finetuned_codebert": [13, 37, 73], "hashing_sgd": [13, 37, 73], "frozen_codebert": [13]}
KEYS = ["dataset", "detector", "project", "alpha_v", "alpha_s", "clip", "method"]
DETECTOR_KEYS = ["dataset", "detector", "project"]
PAIR_KEYS = ["dataset", "detector", "alpha_v", "alpha_s", "clip", "cohort",
             "baseline", "candidate", "outcome"]
SUMMARY_KEYS = ["dataset", "detector", "alpha_v", "alpha_s", "clip", "method"]
CONFIG_HASHES = {
    "configs/neural_validation_v3.json": "6251B1C900AD6F1DDCC600AA402B72EC1A4DD90A9B150B6BF43B9777212CBC81",
    "configs/megavul_validation_v3.json": "FA031D1CC7747E1FCDABF2276F614743BA34157E1A6F6174C6B1D417B9439AA9",
}
COHORTS = ("all_projects_deployment", "domain_supported_secondary")
BOOTSTRAP_DRAWS = 10000
CORE_TABLES = (
    "project_seed_metrics", "project_seed_averages", "paired_project_summaries",
    "domain_supported_paired_summaries", "primary_paired_summary",
    "project_counts_positive_commits", "detector_project_seed_metrics",
    "detector_project_seed_averages", "project_medians", "domain_fit_availability",
    "label_free_decision_counts", "candidate_projects",
)


def ensure(value, name):
    if not bool(value):
        raise ValueError(name)


def protocol_from_configs(source: dict, second: dict) -> dict:
    """Independent schema, without importing the evaluator or selector code."""
    actual = {"finetuned_codebert": source["fine_tuned_codebert"]["seeds"],
              "hashing_sgd": source["fine_tuned_codebert"]["seeds"],
              "frozen_codebert": [13]}
    ensure(actual == ACTUAL, "Frozen protocol actual-fit identities")
    miss = tuple(source["cost"]["miss"])
    ensure(miss == (10, 50), "Frozen evaluation cost schema")
    methods = ("unweighted_mondrian", "estimated_weight_no_gate", "msp_matched_weighted",
               "class_matched_weighted",
               *(f"source_iso_cost{cost}_matched_weighted" for cost in miss),
               *(f"source_iso_cost{cost}_free_volume" for cost in miss),
               "forced_argmax", "all_review")
    return {
        "actual": actual, "alpha_v": tuple(source["risk_budgets"]["vulnerable"]),
        "alpha_s": tuple(source["risk_budgets"]["safe"]),
        "clips": tuple(source["weighting"]["clips"]), "methods": methods,
        "baselines": tuple(method for method in methods if method != "estimated_weight_no_gate"),
        "outcomes": ("auto_fn_rate", "auto_fp_rate", "coverage", "selective_error",
                     "vulnerable_miscoverage", "safe_miscoverage", "maximum_relative_violation",
                     *(f"cost_{cost}" for cost in miss)),
        "primary": tuple(source["risk_budgets"]["primary"]),
        "primary_clip": source["weighting"]["primary_clip"],
        "bootstrap_seed": source["statistics"]["bootstrap_seed"], "bootstrap_draws": BOOTSTRAP_DRAWS,
        "candidate_count": source["cohort"]["expected_candidates"],
        "training_epochs": source["fine_tuned_codebert"]["epochs"],
        "eligibility": {"diversevul": source["cohort"]["post_screen_eligibility"],
                        "megavul": second["eligibility"]},
    }


def expected_rosters(dataset: str, projects: tuple[str, ...], protocol: dict) -> dict:
    """Build complete identities independently of any observed metric groups."""
    policy = {(dataset, detector, project, av, ass, clip, method, seed)
              for detector, seeds in protocol["actual"].items()
              for project, av, ass, clip, method, seed in product(
                  projects, protocol["alpha_v"], protocol["alpha_s"], protocol["clips"],
                  protocol["methods"], seeds)}
    averaged = {row[:-1] for row in policy}
    fitted = {(dataset, detector, project, seed)
              for detector, seeds in protocol["actual"].items() for project, seed in product(projects, seeds)}
    detector_projects = {row[:-1] for row in fitted}
    pairs = {(dataset, detector, av, ass, clip, cohort, baseline, "estimated_weight_no_gate", outcome)
             for detector, av, ass, clip, cohort, baseline, outcome in product(
                 protocol["actual"], protocol["alpha_v"], protocol["alpha_s"], protocol["clips"],
                 COHORTS, protocol["baselines"], protocol["outcomes"])}
    primary = {row for row in pairs if row[2:4] == protocol["primary"]
               and row[4] == protocol["primary_clip"] and row[5] == COHORTS[0]
               and row[6] == "class_matched_weighted" and row[8] == "auto_fn_rate"}
    summary = {(dataset, detector, av, ass, clip, method)
               for detector, av, ass, clip, method in product(
                   protocol["actual"], protocol["alpha_v"], protocol["alpha_s"],
                   protocol["clips"], protocol["methods"])}
    return {
        "project_seed_metrics": (KEYS + ["seed"], policy),
        "label_free_decision_counts": (KEYS + ["seed"], policy),
        "project_seed_averages": (KEYS, averaged),
        "detector_project_seed_metrics": (DETECTOR_KEYS + ["seed"], fitted),
        "domain_fit_availability": (DETECTOR_KEYS + ["seed"], fitted),
        "detector_project_seed_averages": (DETECTOR_KEYS, detector_projects),
        "project_counts_positive_commits": (["dataset", "project"], {(dataset, p) for p in projects}),
        "paired_project_summaries": (PAIR_KEYS, pairs),
        "domain_supported_paired_summaries": (PAIR_KEYS, {row for row in pairs if row[5] == COHORTS[1]}),
        "primary_paired_summary": (PAIR_KEYS, primary),
        "project_medians": (SUMMARY_KEYS, summary),
    }


def exact_roster(frame: pd.DataFrame, keys: list[str], expected: set, label: str, require=ensure) -> None:
    require(set(keys).issubset(frame.columns), label + " identity columns")
    require(not frame[keys].isna().any().any(), label + " nonmissing identities")
    require(not frame.duplicated(keys).any(), label + " unique identities")
    observed = set(frame[keys].itertuples(index=False, name=None))
    missing, unexpected = expected - observed, observed - expected
    require(not missing and not unexpected,
            label + f" exact identities (missing={len(missing)}, unexpected={len(unexpected)})")


def boolean_column(frame: pd.DataFrame, name: str, label: str, require=ensure) -> None:
    require(name in frame and frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all(),
            label + " explicit Boolean " + name)


def validate_structure(dataset: str, receipt: dict, tables: dict[str, pd.DataFrame],
                       protocol: dict, require=ensure) -> dict:
    projects = receipt.get("selected_projects")
    require(isinstance(projects, list) and bool(projects)
            and all(isinstance(p, str) and p and p == p.strip() for p in projects)
            and len(set(projects)) == len(projects), dataset + " unique frozen selected-project receipt")
    eligible = tuple(sorted(projects))
    require(receipt.get("dataset") == dataset and receipt.get("actual_seeds") == protocol["actual"]
            and type(receipt.get("projects")) is int and receipt["projects"] == len(eligible),
            dataset + " frozen dataset/project/actual-fit receipt")
    input_hashes = receipt.get("input_hashes", {})
    config_name = ("configs/neural_validation_v3.json" if dataset == "diversevul"
                   else "configs/megavul_validation_v3.json")
    require(str(input_hashes.get("config", "")).upper() == CONFIG_HASHES[config_name],
            dataset + " frozen input config binding")
    if dataset == "megavul":
        require(str(input_hashes.get("source_config", "")).upper()
                == CONFIG_HASHES["configs/neural_validation_v3.json"], dataset + " shared frozen source config")
    statistics = receipt.get("statistics", {})
    require(statistics.get("bootstrap_draws") == protocol["bootstrap_draws"]
            and statistics.get("bootstrap_seed") == protocol["bootstrap_seed"],
            dataset + " frozen bootstrap receipt")

    ledger = tables["candidate_projects"]
    require({"project_group", "screened_safe", "screened_vulnerable", "selected"}.issubset(ledger.columns),
            dataset + " complete candidate ledger columns")
    require(ledger.project_group.map(lambda p: isinstance(p, str) and bool(p) and p == p.strip()).all()
            and not ledger.project_group.duplicated().any()
            and len(ledger) == receipt.get("candidate_count") == protocol["candidate_count"],
            dataset + " unique complete candidate ledger")
    boolean_column(ledger, "selected", dataset + " candidate ledger", require)
    screened = ledger[["screened_safe", "screened_vulnerable"]]
    require(np.isfinite(screened).all().all() and screened.ge(0).all().all()
            and screened.eq(np.floor(screened)).all().all(), dataset + " integer screened class counts")
    thresholds = protocol["eligibility"][dataset]
    qualifies = (ledger.screened_safe.ge(thresholds["minimum_safe"])
                 & ledger.screened_vulnerable.ge(thresholds["minimum_vulnerable"])
                 & screened.sum(axis=1).ge(thresholds["minimum_total"]))
    require(ledger.selected.equals(qualifies)
            and set(ledger.loc[qualifies, "project_group"]) == set(eligible),
            dataset + " every fixed-screen qualifying project retained")
    for name, (keys, expected) in expected_rosters(dataset, eligible, protocol).items():
        exact_roster(tables[name], keys, expected, dataset + " " + name, require)

    count = tables["project_counts_positive_commits"].set_index("project").sort_index()
    ledger_selected = ledger[ledger.selected].set_index("project_group").reindex(count.index)
    require(np.array_equal(count.n_safe, ledger_selected.screened_safe)
            and np.array_equal(count.n_vulnerable, ledger_selected.screened_vulnerable),
            dataset + " frozen screened project denominators")
    availability = tables["domain_fit_availability"]
    boolean_column(availability, "domain_available", dataset + " availability", require)
    supported_receipt = receipt.get("domain_supported_projects")
    unsupported_receipt = receipt.get("domain_unsupported_projects")
    require(isinstance(supported_receipt, dict) and set(supported_receipt) == set(protocol["actual"])
            and isinstance(unsupported_receipt, dict) and set(unsupported_receipt) == set(protocol["actual"]),
            dataset + " complete domain-support receipt identities")
    supported_by_detector = {}
    for detector in protocol["actual"]:
        cell = availability[availability.detector.eq(detector)]
        supported = tuple(sorted(cell.groupby("project").domain_available.all().loc[lambda s: s].index))
        for declared, expected, label in (
            (supported_receipt[detector], set(supported), "supported"),
            (unsupported_receipt[detector], set(eligible) - set(supported), "unsupported"),
        ):
            require(isinstance(declared, list) and len(set(declared)) == len(declared)
                    and set(declared) == expected, dataset + " exact sealed " + label + " roster " + detector)
        supported_by_detector[detector] = supported
    require(receipt.get("domain_failure_cells") == int((~availability.domain_available).sum()),
            dataset + " exact sealed domain-failure count")
    raw = tables["project_seed_metrics"]
    availability_lookup = availability.set_index(DETECTOR_KEYS + ["seed"]).domain_available.to_dict()
    available = pd.Series([availability_lookup[key] for key in raw[DETECTOR_KEYS + ["seed"]].itertuples(index=False, name=None)],
                          index=raw.index)
    weighted = raw.method.isin(protocol["methods"][1:6])
    expected_status = np.where(weighted & ~available, "domain_fit_failed_review_fallback", "ok")
    require("status" in raw and np.array_equal(raw.status, expected_status),
            dataset + " every planned policy retains its sealed availability/fallback status")
    pairs = tables["paired_project_summaries"]
    boolean_column(pairs, "primary", dataset + " pairs", require)
    pair_rosters = expected_rosters(dataset, eligible, protocol)
    exact_roster(pairs[pairs.primary], PAIR_KEYS, pair_rosters["primary_paired_summary"][1],
                 dataset + " exact primary comparison identities", require)
    require(pairs.bootstrap_draws.eq(protocol["bootstrap_draws"]).all()
            and pairs.bootstrap_seed.eq(protocol["bootstrap_seed"]).all(),
            dataset + " every paired cell fixed bootstrap draws/seed")
    require(receipt.get("rows", {}).get("project_seed_metrics") == len(raw)
            and receipt["rows"].get("project_seed_averages") == len(tables["project_seed_averages"])
            and receipt["rows"].get("paired_project_summaries") == len(pairs),
            dataset + " evaluation receipt exact complete table sizes")
    return {"projects": eligible, "supported": supported_by_detector,
            "expected_rows": {name: len(expected) for name, (_, expected) in pair_rosters.items()}}


def primary_budget(frame: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    return frame[frame.alpha_v.eq(protocol["primary"][0]) & frame.alpha_s.eq(protocol["primary"][1])
                 & frame["clip"].eq(protocol["primary_clip"])]


def verify_release_summaries(analysis: Path, tables: dict, protocol: dict, require, same_frame) -> dict:
    """Check published projections and summaries against the reconstructed cores."""
    combined = {name: pd.concat([tables[d][name] for d in ("diversevul", "megavul")], ignore_index=True)
                for name in ("project_seed_averages", "paired_project_summaries", "project_medians",
                             "project_counts_positive_commits", "domain_fit_availability")}
    for filename, core_name, keys, filter_primary in (
        ("target_project_counts", "project_counts_positive_commits", ["dataset", "project"], False),
        ("primary_budget_project_actions", "project_seed_averages", KEYS, True),
        ("primary_budget_paired_summaries", "paired_project_summaries", PAIR_KEYS, True),
    ):
        expected = primary_budget(combined[core_name], protocol) if filter_primary else combined[core_name]
        supplied = pd.read_csv(analysis / f"{filename}.csv")
        exact_roster(supplied, keys, set(expected[keys].itertuples(index=False, name=None)), filename, require)
        same_frame(expected, supplied, keys, [c for c in expected.columns if c not in keys],
                   "Independent combined projection " + filename)
    detector_rows, cost_rows = [], []
    primary_medians = primary_budget(combined["project_medians"], protocol)
    primary_pairs = primary_budget(combined["paired_project_summaries"], protocol)
    for dataset in ("diversevul", "megavul"):
        count = tables[dataset]["project_counts_positive_commits"]
        availability = tables[dataset]["domain_fit_availability"]
        for detector, seeds in protocol["actual"].items():
            detector_metrics = tables[dataset]["detector_project_seed_averages"]
            cell = detector_metrics[detector_metrics.detector.eq(detector)]
            supported = int(availability[availability.detector.eq(detector)].groupby("project").domain_available.all().sum())
            summary = dict(dataset=dataset, detector=detector, projects=len(cell), actual_fit_count=len(seeds),
                           functions=int(count.n.sum()), positives=int(count.n_vulnerable.sum()),
                           domain_supported_projects=supported)
            for column in ("pr_auc", "roc_auc", "brier", "ece_equal_mass_15", "source_iso_brier", "source_iso_ece_equal_mass_15"):
                for suffix, value in (("median", cell[column].median()),
                                      ("q1", cell[column].quantile(.25)), ("q3", cell[column].quantile(.75))):
                    summary[column + "_" + suffix] = value
            detector_rows.append(summary)
            policy = primary_medians[primary_medians.dataset.eq(dataset) & primary_medians.detector.eq(detector)].set_index("method")
            weighted = policy.loc["estimated_weight_no_gate"]
            for ratio, workload in product((10, 50), ("matched_weighted", "free_volume")):
                baseline = f"source_iso_cost{ratio}_{workload}"
                paired = primary_pairs[primary_pairs.dataset.eq(dataset) & primary_pairs.detector.eq(detector)
                                       & primary_pairs.baseline.eq(baseline) & primary_pairs.outcome.eq(f"cost_{ratio}")
                                       & primary_pairs.cohort.eq(COHORTS[0])].iloc[0]
                comparator = policy.loc[baseline]
                cost_rows.append(dict(dataset=dataset, detector=detector, miss_cost=ratio, workload=workload,
                                      baseline=baseline, weighted_cost=weighted[f"cost_{ratio}"],
                                      baseline_cost=comparator[f"cost_{ratio}"], weighted_coverage=weighted.coverage,
                                      baseline_coverage=comparator.coverage,
                                      **{c: paired[c] for c in ("median_difference", "ci_lower", "ci_upper",
                                                               "reductions", "increases", "ties", "status")}))
    summaries = {"detector_summary": pd.DataFrame(detector_rows), "cost_summary": pd.DataFrame(cost_rows)}
    for name, keys in (("detector_summary", ["dataset", "detector"]),
                       ("cost_summary", ["dataset", "detector", "miss_cost", "workload", "baseline"])):
        expected = summaries[name]
        supplied = pd.read_csv(analysis / f"{name}.csv")
        exact_roster(supplied, keys, set(expected[keys].itertuples(index=False, name=None)), name, require)
        same_frame(expected, supplied, keys, [c for c in expected.columns if c not in keys],
                   "Independent published summary " + name)
    summaries.update(primary_medians=primary_medians, primary_pairs=primary_pairs,
                     target_counts=combined["project_counts_positive_commits"])
    summaries["allocation_structure_summary"] = verify_allocation_structure(analysis, tables, protocol, require, same_frame)
    summaries["source_fit_summary"] = verify_source_fit_summary(analysis, protocol, require, same_frame)
    return summaries


def verify_source_fit_summary(analysis: Path, protocol: dict, require, same_frame) -> pd.DataFrame:
    expected, rows = {(detector, seed) for detector, seeds in protocol["actual"].items() for seed in seeds}, []
    supplied = pd.read_csv(analysis / "source_fit_summary.csv")
    exact_roster(supplied, ["detector", "seed"], expected, "Seven selected source-fit receipt identities", require)
    for detector, seed in sorted(expected):
        path = analysis / "training" / f"{detector}_seed-{seed}.json"
        receipt = read(path)
        require(receipt.get("seed") == seed and receipt.get("target_labels_accessed") is False
                and receipt.get("config_sha256") == CONFIG_HASHES["configs/neural_validation_v3.json"],
                "Source-only selected-fit protocol " + detector + "/" + str(seed))
        script = "train_neural_validation_v3.py" if detector == "finetuned_codebert" else "predict_neural_validation_v3.py"
        require(receipt.get("script_sha256") == sha(ROOT / "src" / script),
                "Selected-fit executed script receipt " + detector + "/" + str(seed))
        if detector == "finetuned_codebert":
            require(receipt.get("selected_epoch") in range(1, protocol["training_epochs"] + 1), "Selected fine-tuning epoch")
        elif detector == "hashing_sgd":
            require(receipt.get("selected_alpha") in (1e-6, 1e-5, 1e-4), "Selected source-only SGD grid alpha")
        else:
            require(receipt.get("selected_C") in (.01, .1, 1, 10), "Selected source-only logistic grid C")
        updates = list(receipt.get("encoder_query_relative_updates", {}).values())
        row = dict(detector=detector, seed=seed,
                   **{name: receipt.get(name, np.nan) for name in (
                       "selected_epoch", "selected_C", "selected_alpha", "trainable_parameters",
                       "source_validation_ap", "wall_seconds", "peak_gpu_allocated_bytes")},
                   changed_encoder_tensors=receipt.get("changed_encoder_tensor_count", 0),
                   query_update_min=min(updates) if updates else np.nan,
                   query_update_max=max(updates) if updates else np.nan,
                   checkpoint_sha256=receipt["checkpoint_sha256"], training_manifest_sha256=sha(path))
        rows.append(row)
    rebuilt = pd.DataFrame(rows)
    same_frame(rebuilt, supplied, ["detector", "seed"], [c for c in rebuilt.columns if c not in {"detector", "seed"}],
               "Independent source-fit receipt summary and selected parameters")
    return rebuilt


def verify_allocation_structure(analysis: Path, tables: dict, protocol: dict, require, same_frame) -> pd.DataFrame:
    """Replay label-free aggregate arithmetic, not excluded per-function equality."""
    manifest = read(analysis / "allocation_structure_manifest.json")
    require(manifest.get("target_labels_accessed") is False
            and manifest.get("selector_or_hyperparameter_modified") is False, "Sealed label-free action-structure scope")
    require(manifest.get("code_sha256") == sha(ROOT / "scripts/audit_label_free_allocation_v3.py"),
            "Label-free action-structure source receipt")
    for name, expected in manifest["reports_sha256"].items():
        require(PurePosixPath(name).name == name and name.endswith(".csv")
                and sha(analysis / name) == expected, "Bound label-free action-structure report " + name)
    for dataset in ("diversevul", "megavul"):
        require(manifest["decision_seals"][dataset] == read(analysis / dataset / "dataset_receipt.json")["decision_seal_sha256"],
                dataset + " action-structure decision seal")
    seed = pd.read_csv(analysis / "project_seed_allocation_structure.csv")
    averages = pd.read_csv(analysis / "project_allocation_structure.csv")
    expected_seed, expected_project = set(), set()
    for dataset in ("diversevul", "megavul"):
        projects = tuple(sorted(tables[dataset]["project_counts_positive_commits"].project))
        rosters = expected_rosters(dataset, projects, protocol)
        expected_seed.update(rosters["domain_fit_availability"][1])
        expected_project.update(rosters["detector_project_seed_averages"][1])
    keys = DETECTOR_KEYS + ["seed"]
    exact_roster(seed, keys, expected_seed, "Label-free action structure actual-fit grid", require)
    exact_roster(averages, DETECTOR_KEYS, expected_project, "Label-free action structure project grid", require)
    flags = ("domain_available", "exact_allocation_match", "no_automatic_safe_release", "full_review")
    for flag in flags:
        boolean_column(seed, flag, "Action-structure fit flags", require)
    availability = pd.concat([tables[d]["domain_fit_availability"] for d in ("diversevul", "megavul")], ignore_index=True)
    same_frame(availability, seed, keys, ["domain_available"], "Action-structure fixed label-free availability")
    counts = pd.concat([primary_budget(tables[d]["label_free_decision_counts"], protocol)
                        for d in ("diversevul", "megavul")], ignore_index=True)
    weighted = counts[counts.method.eq("estimated_weight_no_gate")].rename(
        columns={"auto_safe": "weighted_auto_safe", "auto_vulnerable": "weighted_auto_vulnerable",
                 "review_count": "weighted_review"})
    same_frame(weighted, seed, keys, ["n", "weighted_auto_safe", "weighted_auto_vulnerable", "weighted_review"],
               "Action-structure exact sealed weighted counts")
    msp = counts[counts.method.eq("msp_matched_weighted")].rename(
        columns={"auto_safe": "msp_auto_safe", "auto_vulnerable": "msp_auto_vulnerable"})
    same_frame(msp, seed, keys, ["msp_auto_safe", "msp_auto_vulnerable"], "Action-structure exact sealed MSP counts")
    for numerator, fraction in (("weighted_auto_safe", "weighted_safe_fraction"),
                                ("weighted_auto_vulnerable", "weighted_vulnerable_fraction"),
                                ("weighted_review", "weighted_review_fraction")):
        require(np.allclose(seed[fraction], seed[numerator] / seed.n, atol=1e-12, rtol=1e-11),
                "Independent action-structure fraction " + fraction)
    require(seed.allocation_disagreement_fraction.between(0, 1).all()
            and np.isclose(seed.allocation_disagreement_fraction * seed.n,
                           np.round(seed.allocation_disagreement_fraction * seed.n), atol=1e-9).all(),
            "Action-structure disagreement has complete-function integer denominator")
    require(seed.exact_allocation_match.equals(seed.allocation_disagreement_fraction.eq(0))
            and seed.no_automatic_safe_release.equals(seed.weighted_auto_safe.eq(0))
            and seed.full_review.equals(seed.weighted_review.eq(seed.n)), "Action-structure exact all-fit flag semantics")
    numeric = [c for c in seed.select_dtypes(include=[np.number]).columns if c != "seed"]
    rebuilt = seed.groupby(DETECTOR_KEYS, as_index=False)[numeric].agg(lambda s: s.mean(skipna=False))
    for flag in flags:
        rebuilt = rebuilt.merge(seed.groupby(DETECTOR_KEYS, as_index=False)[flag].all(),
                                on=DETECTOR_KEYS, validate="one_to_one")
    same_frame(rebuilt, averages, DETECTOR_KEYS, [c for c in rebuilt.columns if c not in DETECTOR_KEYS],
               "Independent action-structure actual-fit averaging and all-fit flags")
    rows = []
    for (dataset, detector), cell in rebuilt.groupby(["dataset", "detector"], sort=True):
        supported = cell[cell.domain_available]
        row = dict(dataset=dataset, detector=detector, projects=len(cell), supported_projects=len(supported))
        for flag, name in (("no_automatic_safe_release", "no_safe_release_projects"),
                           ("full_review", "full_review_projects"),
                           ("exact_allocation_match", "exactly_matched_policy_projects")):
            row[name] = int(cell[flag].sum())
            row["supported_" + name] = int(supported[flag].sum())
        for field, name in (("weighted_safe_fraction", "median_safe_fraction"),
                            ("weighted_vulnerable_fraction", "median_vulnerable_fraction"),
                            ("weighted_review_fraction", "median_review_fraction"),
                            ("allocation_disagreement_fraction", "median_allocation_disagreement")):
            row[name] = cell[field].median()
        row["median_vulnerable_infinity_fraction"] = supported.vulnerable_infinity_fraction.median()
        row["median_target_upper_clip_fraction"] = supported.target_upper_clip_fraction.median()
        rows.append(row)
    expected = pd.DataFrame(rows)
    supplied = pd.read_csv(analysis / "allocation_structure_summary.csv")
    exact_roster(supplied, ["dataset", "detector"], set(expected[["dataset", "detector"]].itertuples(index=False, name=None)),
                 "Action-structure dataset/detector summary", require)
    same_frame(expected, supplied, ["dataset", "detector"], [c for c in expected.columns if c not in {"dataset", "detector"}],
               "Independent supported/all action-equality and full-review counts")
    return expected


def display_number(value, places=4) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{0 if abs(value) < .5 * 10 ** -places else value:.{places}f}"


def display_interval(row: pd.Series) -> str:
    if not np.isfinite(row.median_difference):
        return "--"
    return (r"\begin{tabular}[c]{@{}r@{}}$" + display_number(row.median_difference) + r"$\\"
            + r"\scriptsize $[" + display_number(row.ci_lower) + ", " + display_number(row.ci_upper)
            + r"]$\end{tabular}")


def check_table_rows(text: str, label: str, expected_rows: list[str], require=ensure,
                     prefixes: set[str] | None = None) -> None:
    marker = r"\label{" + label + "}"
    require(text.count(marker) == 1, "Unique generated table label " + label)
    body = text.split(marker, 1)[1]
    stop = re.search(r"\\end\{(?:table|longtable)\}", body)
    require(stop is not None, "Closed generated table " + label)
    prefixes = {"DiverseVul", "MegaVul"} if prefixes is None else prefixes
    actual = [line.strip() for line in body[:stop.start()].splitlines()
              if line.strip().split(" & ", 1)[0] in prefixes]
    normalized = lambda rows: [" ".join(row.split()) for row in rows]
    actual, expected = normalized(actual), normalized(expected_rows)
    require(len(actual) == len(expected) and len(set(actual)) == len(actual) and set(actual) == set(expected),
            "Independent table transcription " + label)


def verify_table_transcription(paper: Path, summaries: dict, protocol: dict, require) -> None:
    main = (paper / "neural_validation_main_tables.tex").read_text(encoding="utf-8")
    supplement = (paper / "neural_validation_supplement_tables.tex").read_text(encoding="utf-8")
    collections = {"diversevul": "DiverseVul", "megavul": "MegaVul"}
    names = {"finetuned_codebert": "Fine-tuned", "frozen_codebert": "Frozen", "hashing_sgd": "Hashing"}
    action_rows, primary_rows, supported_rows, detector_rows, structure_rows = [], [], [], [], []
    for dataset, detector in product(collections, names):
        policy = summaries["primary_medians"]
        weighted = policy[policy.dataset.eq(dataset) & policy.detector.eq(detector)
                          & policy.method.eq("estimated_weight_no_gate")].iloc[0]
        action_rows.append(" & ".join([collections[dataset], names[detector],
                                      *[display_number(weighted[c]) for c in ("coverage", "auto_fn_rate", "auto_fp_rate", "selective_error")],
                                      f"{int(weighted.selective_error_defined_projects)}/{int(weighted.projects)}"]) + r"\\")
        pairs = summaries["primary_pairs"]
        def comparison(baseline, outcome, cohort):
            return pairs[pairs.dataset.eq(dataset) & pairs.detector.eq(detector)
                         & pairs.baseline.eq(baseline) & pairs.outcome.eq(outcome) & pairs.cohort.eq(cohort)].iloc[0]
        msp_v = comparison("msp_matched_weighted", "auto_fn_rate", COHORTS[0])
        msp_s = comparison("msp_matched_weighted", "auto_fp_rate", COHORTS[0])
        tail_v = comparison("class_matched_weighted", "auto_fn_rate", COHORTS[0])
        primary_rows.append(" & ".join([collections[dataset], names[detector],
                                       f"{int(tail_v.domain_supported_projects)}/{int(tail_v.all_projects)}",
                                       display_interval(msp_v), display_interval(msp_s), display_interval(tail_v)]) + r"\\[2pt]")
        tail_v, tail_s = (comparison("class_matched_weighted", endpoint, COHORTS[1])
                          for endpoint in ("auto_fn_rate", "auto_fp_rate"))
        rit = lambda row: f"{int(row.reductions)}/{int(row.increases)}/{int(row.ties)}"
        supported_rows.append(" & ".join([collections[dataset], names[detector], str(int(tail_v.projects)),
                                         display_interval(tail_v), rit(tail_v), display_interval(tail_s), rit(tail_s)]) + r"\\[2pt]")
        detector_summary = summaries["detector_summary"]
        row = detector_summary[detector_summary.dataset.eq(dataset) & detector_summary.detector.eq(detector)].iloc[0]
        detector_rows.append(" & ".join([collections[dataset], names[detector], str(int(row.actual_fit_count)),
                                        *[display_number(row[c]) for c in ("pr_auc_median", "roc_auc_median", "brier_median", "source_iso_brier_median")]]) + r"\\")
        structure = summaries["allocation_structure_summary"]
        row = structure[structure.dataset.eq(dataset) & structure.detector.eq(detector)].iloc[0]
        structure_rows.append(" & ".join([collections[dataset], names[detector],
                                         f"{int(row.supported_projects)}/{int(row.projects)}",
                                         str(int(row.full_review_projects)),
                                         f"{int(row.supported_exactly_matched_policy_projects)}/{int(row.exactly_matched_policy_projects)}",
                                         display_number(row.median_allocation_disagreement)]) + r"\\")
    for text, label, rows in ((main, "tab:newdetectors", detector_rows), (main, "tab:newprimary", primary_rows),
                              (main, "tab:newactions", action_rows), (supplement, "si:newsupported", supported_rows),
                              (supplement, "si:newstructure", structure_rows)):
        check_table_rows(text, label, rows, require)
    fit_rows = []
    for row in summaries["source_fit_summary"].itertuples(index=False):
        if row.detector == "finetuned_codebert":
            parameter = str(int(row.selected_epoch))
        elif row.detector == "frozen_codebert":
            parameter = f"$C={row.selected_C:g}$"
        else:
            exponent = int(round(np.log10(row.selected_alpha)))
            require(np.isclose(row.selected_alpha, 10. ** exponent, rtol=1e-12, atol=0),
                    "Source SGD alpha has exact displayed power of ten")
            parameter = r"$\alpha=10^{" + str(exponent) + "}$"
        fit_rows.append(" & ".join([names[row.detector], str(int(row.seed)), parameter,
                                   display_number(row.source_validation_ap), str(int(row.changed_encoder_tensors))]) + r"\\")
    check_table_rows(supplement, "si:newfits", fit_rows, require, set(names.values()))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "paper/neurocomputing_revision/analysis/neural_validation_v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-table-bytes", action="store_true")
    args = parser.parse_args()
    manifest_path = args.analysis / "public_analysis_manifest.json"
    manifest_sha256, checker_sha256 = sha(manifest_path), sha(Path(__file__))
    manifest = read(manifest_path)
    checks, details = [], {}

    def require(value, name):
        if not bool(value):
            raise ValueError(name)
        checks.append(name)

    def same_frame(a, b, keys, values, name):
        assert_frame_equal(a[keys + values].sort_values(keys).reset_index(drop=True),
                           b[keys + values].sort_values(keys).reset_index(drop=True),
                           check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
        checks.append(name)

    for name, expected in CONFIG_HASHES.items():
        require(sha(ROOT / name) == expected, "Frozen protocol bytes " + name)
    protocol = protocol_from_configs(read(ROOT / "configs/neural_validation_v3.json"),
                                     read(ROOT / "configs/megavul_validation_v3.json"))
    require(manifest.get("schema_version") == "trace-v3-aggregate-release-1"
            and manifest.get("no_raw_datasets_predictions_weights_or_vaults") is True,
            "Declared aggregate-only public release schema")
    require(set(manifest.get("datasets", {})) == {"diversevul", "megavul"},
            "Both datasets retained separately in public release")
    supplied_inventory = {p.relative_to(args.analysis).as_posix()
                          for p in args.analysis.rglob("*") if p.is_file()}
    require(supplied_inventory == set(manifest["files_sha256"]) | {"public_analysis_manifest.json"},
            "Exact manifest-bound public aggregate inventory")
    for name, expected in manifest["files_sha256"].items():
        member = PurePosixPath(name)
        require(not member.is_absolute() and member.as_posix() == name
                and not any(part in {"", ".", ".."} for part in name.split("/"))
                and not any(char in name for char in ("\\", ":", "\0")),
                "Safe public aggregate relative path " + name)
        path = (args.analysis / name).resolve()
        require(path.is_relative_to(args.analysis.resolve()) and path.suffix.lower() in {".csv", ".json"},
                "Aggregate-only safe released member " + name)
        require(sha(path) == expected, "Released bytes " + name)
    if args.check_table_bytes:
        for name, expected in manifest["table_source_sha256"].items():
            require(PurePosixPath(name).name == name and name.endswith(".tex"), "Safe generated table source " + name)
            require(sha(args.analysis.parents[1] / name) == expected, "Generated table bytes " + name)
    bootstrap_indices, interval_cache = {}, {}
    dataset_tables = {}
    for dataset in ("diversevul", "megavul"):
        base = args.analysis / dataset
        receipt = read(base / "dataset_receipt.json")
        tables = {name: pd.read_csv(base / f"{name}.csv") for name in CORE_TABLES}
        structure = validate_structure(dataset, receipt, tables, protocol, require)
        dataset_tables[dataset] = tables
        declared = manifest["datasets"][dataset]
        for field in ("projects", "target_rows", "target_vulnerable", "actual_seeds",
                      "decision_seal_sha256", "domain_failure_cells"):
            require(declared.get(field) == receipt.get(field), dataset + " public/dataset receipt " + field)
        raw, avg = tables["project_seed_metrics"], tables["project_seed_averages"]
        pairs, count = tables["paired_project_summaries"], tables["project_counts_positive_commits"]
        availability = tables["domain_fit_availability"]
        require(raw.dataset.eq(dataset).all() and receipt["actual_seeds"] == ACTUAL, dataset + " fixed identities")
        require(not raw.duplicated(KEYS + ["seed"]).any(), dataset + " unique metric cells")
        numeric = [c for c in raw.select_dtypes(include=[np.number]).columns if c not in KEYS + ["seed"]]
        group = raw.groupby(KEYS, sort=True, dropna=False)
        # Explicit count check prevents averaging available fits after a failure.
        expected_seed_sets = group.seed.agg(lambda s: tuple(sorted(s)))
        require(all(seeds == tuple(ACTUAL[key[1]]) for key, seeds in expected_seed_sets.items()), dataset + " complete actual-fit grid")
        reconstructed = group[numeric].agg(lambda s: s.mean(skipna=False)).reset_index()
        reconstructed["actual_seed_count"] = group.size().to_numpy()
        reconstructed["status"] = group.status.agg(lambda s: "ok" if s.eq("ok").all() else "policy_failure_present").to_numpy()
        same_frame(reconstructed, avg, KEYS, numeric + ["actual_seed_count", "status"], dataset + " independent actual-fit averaging")
        detect_seed = tables["detector_project_seed_metrics"]
        detect_avg = tables["detector_project_seed_averages"]
        detect_cols = [c for c in detect_seed.select_dtypes(include=[np.number]).columns if c != "seed"]
        rebuilt_detect = detect_seed.groupby(DETECTOR_KEYS, as_index=False)[detect_cols].mean()
        same_frame(rebuilt_detect, detect_avg, DETECTOR_KEYS, detect_cols, dataset + " detector fit averaging")
        require(len(count) == receipt["projects"] and int(count.n.sum()) == receipt["target_rows"]
                and int(count.n_vulnerable.sum()) == receipt["target_vulnerable"], dataset + " cohort denominators")
        require(np.array_equal(count.n, count.n_vulnerable + count.n_safe)
                and count.positive_commits.between(1, count.n_vulnerable).all()
                and count.largest_positive_commit_fraction.between(0, 1).all(), dataset + " positive-commit bounds")
        eligible = structure["projects"]
        for detector, seeds in ACTUAL.items():
            cell = availability[availability.detector == detector]
            require(len(cell) == len(eligible) * len(seeds) and not cell.duplicated(["project", "seed"]).any(),
                    dataset + " complete label-free availability " + detector)
            supported = tuple(sorted(cell.groupby("project").domain_available.all().loc[lambda s: s].index))
            require(list(supported) == sorted(receipt["domain_supported_projects"][detector]),
                    dataset + " sealed supported roster " + detector)
        same_frame(raw, tables["label_free_decision_counts"], KEYS + ["seed"],
                   ["n", "auto_safe", "auto_vulnerable", "review_count", "status"],
                   dataset + " every metric cell preserves label-free sealed actions")
        same_frame(pairs[pairs.primary], tables["primary_paired_summary"], PAIR_KEYS,
                   [name for name in pairs.columns if name not in PAIR_KEYS], dataset + " exact primary paired projection")
        same_frame(pairs[pairs.cohort.eq(COHORTS[1])], tables["domain_supported_paired_summaries"], PAIR_KEYS,
                   [name for name in pairs.columns if name not in PAIR_KEYS], dataset + " exact supported paired projection")
        summary_group = avg.groupby(SUMMARY_KEYS, sort=True)
        medians = summary_group.median(numeric_only=True)
        medians["projects"] = summary_group.size()
        medians["policy_ok_projects"] = summary_group.status.agg(lambda s: int(s.eq("ok").sum()))
        medians["policy_failure_projects"] = medians["projects"] - medians["policy_ok_projects"]
        for column in avg.select_dtypes(include=[np.number]).columns:
            if column not in SUMMARY_KEYS:
                medians[column + "_defined_projects"] = summary_group[column].count()
        medians = medians.reset_index()
        same_frame(medians, tables["project_medians"], SUMMARY_KEYS,
                   [name for name in medians.columns if name not in SUMMARY_KEYS],
                   dataset + " independent project medians and defined denominators")
        grouped = {key: frame for key, frame in avg.groupby(["dataset", "detector", "alpha_v", "alpha_s", "clip"], sort=True)}
        numerical = ["median_difference", "ci_lower", "ci_upper"]
        count_fields = ["projects", "defined_projects", "eligible_projects", "all_projects", "domain_supported_projects", "review_fallback_projects",
                        "reductions", "increases", "ties", "bootstrap_draws", "bootstrap_seed"]
        rows = []
        for _, row in pairs.iterrows():
            key = (row.dataset, row.detector, row.alpha_v, row.alpha_s, row["clip"])
            cell = grouped[key]
            supported = set(receipt["domain_supported_projects"][row.detector])
            if row.cohort == "domain_supported_secondary":
                cell = cell[cell.project.isin(supported)]
            else:
                require(row.cohort == "all_projects_deployment", dataset + " declared paired cohort")
            a = cell[cell.method == row.baseline].set_index("project").sort_index()
            b = cell[cell.method == row.candidate].set_index("project").sort_index()
            expected_projects = (tuple(sorted(supported)) if row.cohort == COHORTS[1] else eligible)
            require(tuple(a.index) == tuple(b.index) == expected_projects
                    and a.index.is_unique and b.index.is_unique,
                    dataset + " paired project identities")
            b = b.reindex(a.index)
            roster = tuple(a.index)
            delta = (b[row.outcome] - a[row.outcome]).to_numpy(float)
            failed = not a.status.eq("ok").all() or not b.status.eq("ok").all()
            rebuilt = {"projects": len(roster), "defined_projects": int(np.isfinite(delta).sum()),
                       "eligible_projects": len(roster), "all_projects": len(eligible),
                       "domain_supported_projects": len(set(roster) & supported),
                       "review_fallback_projects": len(set(roster) - supported),
                       "bootstrap_draws": protocol["bootstrap_draws"], "bootstrap_seed": protocol["bootstrap_seed"],
                       "reductions": 0, "increases": 0, "ties": 0,
                       "median_difference": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "status": "ok"}
            if not len(roster):
                rebuilt["status"] = "no_eligible_projects"
            elif failed and row.cohort != "all_projects_deployment":
                rebuilt["status"] = "incomplete_policy_failure_no_project_dropped"
            elif not np.isfinite(delta).all():
                rebuilt["status"] = "undefined_endpoint_no_project_dropped"
            else:
                rebuilt.update(median_difference=float(np.median(delta)), reductions=int((delta < -1e-12).sum()),
                               increases=int((delta > 1e-12).sum()), ties=int((np.abs(delta) <= 1e-12).sum()))
                if failed:
                    rebuilt["status"] = "deployment_review_fallback_included_not_selection_evidence"
                if roster not in bootstrap_indices:
                    bootstrap_indices[roster] = np.random.default_rng(protocol["bootstrap_seed"]).integers(
                        0, len(roster), size=(protocol["bootstrap_draws"], len(roster)))
                cache_key = (roster, hashlib.sha256(delta.tobytes()).hexdigest())
                if cache_key not in interval_cache:
                    interval_cache[cache_key] = (float(delta[0]), float(delta[0])) if np.all(delta == delta[0]) else tuple(
                        np.quantile(np.median(delta[bootstrap_indices[roster]], axis=1), [.025, .975]))
                rebuilt["ci_lower"], rebuilt["ci_upper"] = interval_cache[cache_key]
            for column in numerical:
                require(np.isclose(row[column], rebuilt[column], atol=1e-12, rtol=1e-11, equal_nan=True),
                        dataset + " paired estimate and bootstrap " + column)
            for column in count_fields:
                require(row[column] == rebuilt[column], dataset + " paired count " + column)
            require(row.status == rebuilt["status"], dataset + " paired availability/undefined status")
            rows.append(rebuilt)
        primary = pairs[pairs.primary]
        require(len(primary) == 3 and primary.cohort.eq("all_projects_deployment").all()
                and primary.baseline.eq("class_matched_weighted").all() and primary.outcome.eq("auto_fn_rate").all()
                and np.isclose(primary.alpha_v, .1).all() and np.isclose(primary.alpha_s, .2).all()
                and primary["clip"].eq(protocol["primary_clip"]).all(), dataset + " frozen primary endpoint")
        details[dataset] = dict(projects=len(eligible), target_rows=int(count.n.sum()), vulnerable=int(count.n_vulnerable.sum()),
                                project_seed_rows=len(raw), project_averages=len(avg), independently_rebuilt_paired_cells=len(rows),
                                actual_seeds=ACTUAL, domain_supported_projects=receipt["domain_supported_projects"],
                                expected_table_rows=structure["expected_rows"])
        print(dataset, "independent aggregate reconstruction PASS", len(rows), "paired cells", flush=True)
    summaries = verify_release_summaries(args.analysis, dataset_tables, protocol, require, same_frame)
    if args.check_table_bytes:
        verify_table_transcription(args.analysis.parents[1], summaries, protocol, require)
    require(sha(manifest_path) == manifest_sha256, "Public release manifest unchanged throughout reconstruction")
    require(sha(Path(__file__)) == checker_sha256, "Checker source unchanged throughout reconstruction")
    require({p.relative_to(args.analysis).as_posix() for p in args.analysis.rglob("*") if p.is_file()} == supplied_inventory,
            "Public release inventory unchanged throughout reconstruction")
    for name, expected in manifest["files_sha256"].items():
        require(sha(args.analysis / name) == expected, "Released bytes still current " + name)
    if args.check_table_bytes:
        for name, expected in manifest["table_source_sha256"].items():
            require(sha(args.analysis.parents[1] / name) == expected, "Generated table bytes still current " + name)
    report = dict(status="PASS", created_at_utc=datetime.now(timezone.utc).isoformat(), check_count=len(checks),
                  check_categories=sorted(set(checks)), details=details,
                  public_analysis_manifest_sha256=manifest_sha256,
                  checker_script_sha256=checker_sha256, unique_bootstrap_vectors=len(interval_cache),
                  config_sha256=CONFIG_HASHES["configs/neural_validation_v3.json"],
                  megavul_config_sha256=CONFIG_HASHES["configs/megavul_validation_v3.json"],
                  scope="Every released byte; exact planned project/fit/budget/clip/policy/comparator/outcome grids; actual-fit averages, project medians/defined counts, paired estimates/directions and all 10,000-draw project intervals; label-free structure aggregation, selected source-fit receipts, and generated main/supported/structure/source-fit table transcription; no raw label or prediction access.",
                  limitations=["Per-function action equality and infinity-mass inputs are not replayed from aggregate counts alone; their released label-free receipts are bound and their aggregate arithmetic/flags/summaries are independently checked."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "check_categories"}, indent=2))


if __name__ == "__main__":
    main()
