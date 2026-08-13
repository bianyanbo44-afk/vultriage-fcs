"""Metrics for binary set-valued vulnerability triage."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def equal_mass_ece(
    labels: np.ndarray, vulnerable_probabilities: np.ndarray, bins: int = 15
) -> float:
    """Binary ECE using deterministic equal-mass bins."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(vulnerable_probabilities, dtype=float)
    if labels.shape != probabilities.shape or bins < 1:
        raise ValueError("ECE inputs are incompatible")
    order = np.argsort(probabilities, kind="stable")
    result = 0.0
    for indices in np.array_split(order, min(bins, len(order))):
        if len(indices):
            result += len(indices) / len(labels) * abs(
                float(labels[indices].mean() - probabilities[indices].mean())
            )
    return float(result)


def aurc(labels: np.ndarray, vulnerable_probabilities: np.ndarray) -> float:
    """Area under the forced-classification selective risk--coverage curve."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(vulnerable_probabilities, dtype=float)
    forced = (probabilities >= 0.5).astype(int)
    error = (forced != labels).astype(float)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    risk = np.cumsum(error[order]) / np.arange(1, len(labels) + 1)
    return float(risk.mean())


def fnr_at_fpr(
    labels: np.ndarray, vulnerable_probabilities: np.ndarray, maximum_fpr: float
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(vulnerable_probabilities, dtype=float)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    eligible = fpr <= maximum_fpr + np.finfo(float).eps
    return float(1.0 - np.max(tpr[eligible]))


def triage_metrics(
    labels: np.ndarray,
    vulnerable_probabilities: np.ndarray,
    prediction_sets: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    p_vulnerable = np.asarray(vulnerable_probabilities, dtype=float)
    sets = np.asarray(prediction_sets, dtype=bool)
    if labels.shape != p_vulnerable.shape or sets.shape != (len(labels), 2):
        raise ValueError("Metric arrays have incompatible shapes")
    forced = (p_vulnerable >= 0.5).astype(int)
    true_in_set = sets[np.arange(len(labels)), labels]
    set_sizes = sets.sum(axis=1)
    vulnerable_mask = labels == 1
    safe_mask = labels == 0
    singleton_mask = set_sizes == 1
    miscovered = ~true_in_set
    forced_error = forced != labels
    uncertainty = 1.0 - np.maximum(p_vulnerable, 1.0 - p_vulnerable)

    def mean_or_nan(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else float("nan")

    return {
        "n": int(len(labels)),
        "n_vulnerable": int(vulnerable_mask.sum()),
        "n_safe": int(safe_mask.sum()),
        "vulnerable_miscoverage_count": int(miscovered[vulnerable_mask].sum()),
        "safe_miscoverage_count": int(miscovered[safe_mask].sum()),
        "singleton_count": int(singleton_mask.sum()),
        "vulnerable_miscoverage": mean_or_nan(~true_in_set[vulnerable_mask]),
        "safe_miscoverage": mean_or_nan(~true_in_set[safe_mask]),
        "singleton_coverage": float((set_sizes == 1).mean()),
        "vulnerable_singleton_rate": mean_or_nan(singleton_mask[vulnerable_mask]),
        "safe_singleton_rate": mean_or_nan(singleton_mask[safe_mask]),
        "review_load": float((set_sizes != 1).mean()),
        "empty_rate": float((set_sizes == 0).mean()),
        "doubleton_rate": float((set_sizes == 2).mean()),
        "pr_auc": float(average_precision_score(labels, p_vulnerable)),
        "brier": float(brier_score_loss(labels, p_vulnerable)),
        "ece_equal_mass_15": equal_mass_ece(labels, p_vulnerable, bins=15),
        "aurc": aurc(labels, p_vulnerable),
        "fnr_at_fpr_0_005": fnr_at_fpr(labels, p_vulnerable, 0.005),
        "error_detection_auroc": (
            float(roc_auc_score(forced_error.astype(int), uncertainty))
            if np.unique(forced_error).size == 2
            else float("nan")
        ),
        "precision": float(precision_score(labels, forced, zero_division=0)),
        "recall": float(recall_score(labels, forced, zero_division=0)),
        "f1": float(f1_score(labels, forced, zero_division=0)),
        "f2": float(fbeta_score(labels, forced, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, forced)),
    }
