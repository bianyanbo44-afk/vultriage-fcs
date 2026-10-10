"""Label-boundary-safe policy and project summaries for neural validation v3.

Policy construction accepts source calibration labels only.  Target outcomes
are arguments exclusively to the post-seal metric functions below.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from itertools import product
from typing import Any, Sequence
from urllib.parse import unquote

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from analyze_neurocomputing_controls import (
    action_codes,
    class_side_matched,
    global_msp_matched,
    probability_vector,
    source_cost_matched,
)
from analyze_neurocomputing_revision import action_counts
from vultriage.conformal import weighted_conformal_sets
from vultriage.domain import CrossfitDensityRatio
from vultriage.metrics import equal_mass_ece

DOMAIN_FOLDS = 5
DOMAIN_ALPHA = 1e-4
DOMAIN_EPOCHS = 5
DOMAIN_SALT = "trace-neural-v3-project-commit-domain"
METHODS = (
    "unweighted_mondrian", "estimated_weight_no_gate", "msp_matched_weighted",
    "class_matched_weighted", "source_iso_cost10_matched_weighted",
    "source_iso_cost50_matched_weighted", "source_iso_cost10_free_volume",
    "source_iso_cost50_free_volume", "forced_argmax", "all_review",
)
WEIGHT_DEPENDENT = frozenset(METHODS[1:6])


class DomainFitUnavailable(ValueError):
    """A grouped complement cannot support a two-domain fit."""


@lru_cache(maxsize=262144)
def canonical_commit_id(value: str) -> str:
    """A global 40-hex commit identity, independent of URL/alias decorations."""
    if not isinstance(value, str):
        raise ValueError("Commit identity must be a string")
    cleaned = unquote(value.strip()).split("?", 1)[0].split("#", 1)[0].casefold()
    if re.fullmatch(r"[0-9a-f]{40}", cleaned):
        return cleaned
    matches = set(re.findall(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", cleaned))
    if len(matches) != 1:
        raise ValueError("Commit identity must resolve unambiguously to one global 40-hex SHA")
    return next(iter(matches))


def grouped_fold_ids(groups: Sequence[tuple[str, str]], *, folds: int = DOMAIN_FOLDS,
                     salt: str = DOMAIN_SALT) -> np.ndarray:
    if folds < 2:
        raise ValueError("At least two grouped domain folds are required")
    result = []
    for project, commit in groups:
        project, commit = str(project).strip().casefold(), canonical_commit_id(str(commit))
        if not project or not commit:
            raise ValueError("Domain grouping requires project and commit identities")
        # A commit is content-addressed globally.  Alias/fork spellings of the
        # same SHA must not place correlated functions on opposite fold sides.
        key = json.dumps(["global_commit", commit], separators=(",", ":"))
        digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
        result.append(int.from_bytes(digest[:8], "big") % folds)
    return np.asarray(result, dtype=np.int64)


def crossfit_grouped_density_ratio(features: Any, row_ids: Sequence[str],
                                  domain_labels: np.ndarray,
                                  groups: Sequence[tuple[str, str]], *, seed: int,
                                  folds: int = DOMAIN_FOLDS,
                                  alpha: float = DOMAIN_ALPHA,
                                  epochs: int = DOMAIN_EPOCHS,
                                  salt: str = DOMAIN_SALT) -> CrossfitDensityRatio:
    """Balanced logistic SGD, with every complete project/commit in one fold."""
    labels = np.asarray(domain_labels)
    n = len(row_ids)
    if (features.shape[0] != n or labels.shape != (n,) or len(groups) != n
            or not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}
            or len(set(row_ids)) != n):
        raise ValueError("Domain features, unique row IDs, labels and groups must align")
    if sparse.issparse(features):
        features = features.tocsr()
        finite = np.isfinite(features.data).all()
    else:
        features = np.asarray(features, dtype=np.float32)
        finite = np.isfinite(features).all()
    if not finite:
        raise ValueError("Domain features must be finite")
    groups = [(str(project).strip().casefold(), canonical_commit_id(str(commit)))
              for project, commit in groups]
    fold_ids = grouped_fold_ids(groups, folds=folds, salt=salt)
    probabilities = np.full(n, np.nan, dtype=float)
    records = []
    for fold in range(folds):
        held = np.flatnonzero(fold_ids == fold)
        if not len(held):
            raise DomainFitUnavailable(f"Grouped domain fold {fold} has no held-out rows")
        source = np.flatnonzero((fold_ids != fold) & (labels == 0))
        target = np.flatnonzero((fold_ids != fold) & (labels == 1))
        balanced_n = min(len(source), len(target))
        if not balanced_n:
            raise DomainFitUnavailable(f"Grouped fold {fold} has an empty training domain")
        rng = np.random.default_rng(int(seed) + 104729 * fold)
        if len(source) > balanced_n:
            source = rng.choice(source, balanced_n, replace=False)
        if len(target) > balanced_n:
            target = rng.choice(target, balanced_n, replace=False)
        train = np.concatenate([source, target])
        rng.shuffle(train)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha,
                              random_state=int(seed) + fold, max_iter=epochs, tol=None)
        try:
            model.fit(features[train], labels[train].astype(int))
            probabilities[held] = model.predict_proba(features[held])[:, 1]
        except (ValueError, FloatingPointError) as error:
            raise DomainFitUnavailable(f"Grouped fold {fold} numerical fit failed: {error}") from error
        records.append({
            "fold": fold, "balanced_training_per_domain": balanced_n,
            "held_source": int(np.sum(labels[held] == 0)),
            "held_target": int(np.sum(labels[held] == 1)),
            "held_groups": len({groups[int(i)] for i in held}),
            "training_groups": len({groups[int(i)] for i in train}),
            "effective_target_prior": 0.5, "prior_odds_correction": 1.0,
        })
    if not np.isfinite(probabilities).all():
        raise DomainFitUnavailable("Grouped cross-fit produced non-finite probabilities")
    bounded = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    ratios = bounded / (1 - bounded)
    return CrossfitDensityRatio(
        row_ids=tuple(row_ids), domain_labels=labels.astype(int), fold_ids=fold_ids,
        target_probabilities=probabilities, raw_ratios=ratios,
        diagnostics={"out_of_fold": True, "grouping": "project/commit identities; shared global SHA coalesced",
                     "fold_salt": salt, "crossfit_folds": folds,
                     "alpha": alpha, "epochs": epochs,
                     "domain_auroc": float(roc_auc_score(labels, probabilities)),
                     "domain_average_precision": float(average_precision_score(labels, probabilities)),
                     "source_n": int(np.sum(labels == 0)),
                     "target_n": int(np.sum(labels == 1)), "folds": records},
    )


def free_volume_cost_sets(eta: np.ndarray, miss_cost: float, *, false_alert_cost: float = 1,
                          review_cost: float = 0.1) -> np.ndarray:
    """Automate only when the cheapest automatic action costs strictly less."""
    eta = probability_vector(eta)
    if (isinstance(review_cost, (bool, np.bool_)) or not np.isfinite(review_cost)
            or review_cost < 0):
        raise ValueError("Review cost must be finite and nonnegative")
    # Reuse the fixed-volume selector's cost validation and label tie convention.
    full = source_cost_matched(eta, len(eta), miss_cost, false_alert_cost)
    automate = np.minimum(miss_cost * eta, false_alert_cost * (1 - eta)) < review_cost
    full[~automate] = True
    return full


def cell_key(alpha_v: float, alpha_s: float, clip: float, method: str) -> str:
    number = lambda value: format(float(value), ".12g").replace(".", "p")
    return f"av{number(alpha_v)}_as{number(alpha_s)}__clip{number(clip)}__{method}"


def construct_decisions(cal_probability: np.ndarray, cal_labels: np.ndarray,
                        target_probability: np.ndarray, source_eta: np.ndarray,
                        raw_cal_weights: np.ndarray | None,
                        raw_target_weights: np.ndarray | None,
                        config: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build every frozen grid cell without receiving target labels."""
    cal_probability, target_probability = probability_vector(cal_probability), probability_vector(target_probability)
    cal_labels = np.asarray(cal_labels)
    if cal_labels.shape != cal_probability.shape or not np.isin(cal_labels, [0, 1]).all():
        raise ValueError("Invalid source calibration labels")
    source_eta = probability_vector(source_eta)
    if source_eta.shape != target_probability.shape:
        raise ValueError("Source-calibrated target scores do not align")
    if (raw_cal_weights is None) != (raw_target_weights is None):
        raise ValueError("Both calibration and target weights are required together")
    available = raw_cal_weights is not None
    if available:
        raw_cal_weights = np.asarray(raw_cal_weights, dtype=float)
        raw_target_weights = np.asarray(raw_target_weights, dtype=float)
        if (raw_cal_weights.shape != cal_probability.shape or raw_target_weights.shape != target_probability.shape
                or not np.isfinite(raw_cal_weights).all() or not np.isfinite(raw_target_weights).all()
                or (raw_cal_weights <= 0).any() or (raw_target_weights <= 0).any()):
            raise ValueError("Positive finite density ratios must align")
    pcal = np.column_stack([1 - cal_probability, cal_probability])
    ptarget = np.column_stack([1 - target_probability, target_probability])
    fixed = {"forced_argmax": global_msp_matched(target_probability, len(target_probability)),
             "all_review": np.ones((len(target_probability), 2), dtype=bool)}
    for cost in config["cost"]["miss"]:
        fixed[f"source_iso_cost{int(cost)}_free_volume"] = free_volume_cost_sets(
            source_eta, cost, false_alert_cost=config["cost"]["false_alert"],
            review_cost=config["cost"]["review"])
    arrays, specs = {}, []
    for av, ass in product(config["risk_budgets"]["vulnerable"], config["risk_budgets"]["safe"]):
        alpha = {0: float(ass), 1: float(av)}
        # Unit weights retain the test-point infinity mass even when a class is
        # too small for a finite (n+1)(1-alpha) quantile.  The old threshold
        # helper caps at a finite maximum and is inappropriate at that boundary.
        unweighted = weighted_conformal_sets(
            pcal, cal_labels, np.ones(len(cal_labels)), ptarget,
            np.ones(len(target_probability)), alpha)[0]
        for clip in config["weighting"]["clips"]:
            weighted = (weighted_conformal_sets(
                pcal, cal_labels, np.clip(raw_cal_weights, 1 / clip, clip),
                ptarget, np.clip(raw_target_weights, 1 / clip, clip), alpha)[0]
                if available else np.ones_like(unweighted))
            codes = action_codes(weighted)
            count = int(np.sum(codes != 2))
            policies = {**fixed, "unweighted_mondrian": unweighted,
                        "estimated_weight_no_gate": weighted,
                        "msp_matched_weighted": global_msp_matched(target_probability, count),
                        "class_matched_weighted": class_side_matched(
                            target_probability, int(np.sum(codes == 0)), int(np.sum(codes == 1)))}
            for cost in config["cost"]["miss"]:
                policies[f"source_iso_cost{int(cost)}_matched_weighted"] = source_cost_matched(
                    source_eta, count, cost, config["cost"]["false_alert"])
            for method in METHODS:
                key = cell_key(av, ass, clip, method)
                arrays[key] = policies[method]
                specs.append({"key": key, "method": method, "alpha_v": float(av),
                              "alpha_s": float(ass), "clip": float(clip),
                              "status": ("domain_fit_failed_review_fallback"
                                         if not available and method in WEIGHT_DEPENDENT else "ok")})
    return arrays, specs


def project_metrics(labels: np.ndarray, probability: np.ndarray, sets: np.ndarray,
                    alpha_v: float, alpha_s: float, config: dict) -> dict:
    """Post-seal metrics, preserving the true-class automatic-error denominators."""
    result = action_counts(labels, sets)
    labels = np.asarray(labels, dtype=int)
    sets = np.asarray(sets, dtype=bool)
    result["doubleton_count"] = int(np.sum(sets.sum(axis=1) == 2))
    result["empty_rate"] = result["empty_count"] / len(labels)
    result["doubleton_rate"] = result["doubleton_count"] / len(labels)
    missing = ~sets[np.arange(len(labels)), labels]
    for label, name in ((1, "vulnerable"), (0, "safe")):
        mask = labels == label
        result[f"{name}_miscoverage_count"] = int(np.sum(missing[mask]))
        result[f"{name}_miscoverage"] = float(np.mean(missing[mask])) if mask.any() else np.nan
    result["maximum_relative_violation"] = (max(
        max(0, result["vulnerable_miscoverage"] - alpha_v) / alpha_v,
        max(0, result["safe_miscoverage"] - alpha_s) / alpha_s)
        if result["n_vulnerable"] and result["n_safe"] else np.nan)
    for cost in config["cost"]["miss"]:
        result[f"cost_{int(cost)}"] = (
            cost * result["auto_fn"] + config["cost"]["false_alert"] * result["auto_fp"]
            + config["cost"]["review"] * result["review_count"]) / len(labels)
    return result


def detector_metrics(labels: np.ndarray, probability: np.ndarray, source_eta: np.ndarray) -> dict:
    labels, probability = np.asarray(labels, dtype=int), probability_vector(probability)
    two_classes = np.unique(labels).size == 2
    return {"pr_auc": float(average_precision_score(labels, probability)) if two_classes else np.nan,
            "roc_auc": float(roc_auc_score(labels, probability)) if two_classes else np.nan,
            "brier": float(brier_score_loss(labels, probability)),
            "ece_equal_mass_15": equal_mass_ece(labels, probability),
            "source_iso_brier": float(brier_score_loss(labels, source_eta)),
            "source_iso_ece_equal_mass_15": equal_mass_ece(labels, source_eta)}


def project_commit_counts(rows: Sequence[dict], labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    if labels.shape != (len(rows),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Commit counts require aligned target labels")
    commits = np.asarray([canonical_commit_id(str(row["commit_id"])) for row in rows])
    positive = pd.Series(commits[labels == 1]).value_counts().to_numpy(dtype=float)
    npositive = int(np.sum(labels == 1))
    fractions = positive / npositive if npositive else np.asarray([], dtype=float)
    return {"n": len(rows), "n_vulnerable": npositive, "n_safe": len(rows) - npositive,
            "distinct_commits": len(set(commits)), "positive_commits": len(positive),
            "largest_positive_commit_functions": int(positive.max()) if len(positive) else 0,
            "largest_positive_commit_fraction": float(fractions.max()) if len(fractions) else np.nan,
            "top3_positive_commit_fraction": float(fractions[:3].sum()) if len(fractions) else np.nan,
            "positive_commit_hhi": float(np.square(fractions).sum()) if len(fractions) else np.nan,
            "effective_positive_commit_count": float(1 / np.square(fractions).sum()) if len(fractions) else np.nan}


def average_actual_seed_metrics(frame: pd.DataFrame, actual_seeds: dict[str, list[int]]) -> pd.DataFrame:
    keys = ["dataset", "detector", "project", "alpha_v", "alpha_s", "clip", "method"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate project/seed policy rows; copied seed addresses are not fits")
    rows = []
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in keys + ["seed"]]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        detector = key[keys.index("detector")]
        if set(group["seed"]) != set(actual_seeds[detector]):
            raise ValueError("Policy cells do not contain the same actual fitted seeds")
        row = dict(zip(keys, key))
        # An all-review seed has undefined selective error.  Averaging only the
        # other seeds would silently replace three actual fits with a subset.
        row.update(group[numeric].mean(skipna=False).to_dict())
        row["actual_seed_count"] = len(group)
        row["status"] = "ok" if group["status"].eq("ok").all() else "policy_failure_present"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_project_summary(frame: pd.DataFrame, outcome: str, baseline: str, candidate: str,
                           *, bootstrap_seed: int = 20261010, draws: int = 10000,
                           bootstrap_indices: np.ndarray | None = None,
                           include_review_fallback: bool = False,
                           all_project_count: int | None = None,
                           domain_supported_projects: Sequence[str] | None = None) -> dict:
    selected = frame[frame["method"].isin([baseline, candidate])]
    if selected.duplicated(["project", "method"]).any():
        raise ValueError("Paired inference requires one seed-averaged row per project and method")
    a = selected[selected["method"] == baseline].set_index("project").sort_index()
    b = selected[selected["method"] == candidate].set_index("project").sort_index()
    if set(a.index) != set(b.index):
        raise ValueError("Unmatched paired project cohorts")
    b = b.reindex(a.index)
    supported = set(domain_supported_projects) if domain_supported_projects is not None else set(a.index)
    result = {"outcome": outcome, "baseline": baseline, "candidate": candidate,
              "projects": len(a), "median_difference": np.nan, "ci_lower": np.nan,
              "ci_upper": np.nan, "reductions": 0, "increases": 0, "ties": 0,
              "defined_projects": 0, "status": "ok", "bootstrap_draws": draws,
              "bootstrap_seed": bootstrap_seed, "eligible_projects": len(a),
              "all_projects": len(a) if all_project_count is None else int(all_project_count),
              "domain_supported_projects": len(set(a.index) & supported),
              "review_fallback_projects": len(set(a.index) - supported)}
    if not len(a):
        result["status"] = "no_eligible_projects"
        return result
    delta = (b[outcome] - a[outcome]).to_numpy(dtype=float)
    result["defined_projects"] = int(np.isfinite(delta).sum())
    failed = "status" in a and (not a["status"].eq("ok").all() or not b["status"].eq("ok").all())
    if failed and not include_review_fallback:
        result["status"] = "incomplete_policy_failure_no_project_dropped"
        return result
    if not np.isfinite(delta).all():
        result["status"] = "undefined_endpoint_no_project_dropped"
        return result
    result["median_difference"] = float(np.median(delta))
    if failed:
        result["status"] = "deployment_review_fallback_included_not_selection_evidence"
    result["reductions"] = int(np.sum(delta < -1e-12))
    result["increases"] = int(np.sum(delta > 1e-12))
    result["ties"] = len(delta) - result["reductions"] - result["increases"]
    if np.all(delta == delta[0]):
        lower = upper = float(delta[0])
    else:
        if bootstrap_indices is None:
            bootstrap_indices = np.random.default_rng(bootstrap_seed).integers(
                0, len(delta), size=(draws, len(delta)))
        if bootstrap_indices.shape != (draws, len(delta)):
            raise ValueError("Bootstrap indices must resample complete projects")
        lower, upper = np.quantile(np.median(delta[bootstrap_indices], axis=1), [0.025, 0.975])
    result.update(ci_lower=float(lower), ci_upper=float(upper))
    return result
