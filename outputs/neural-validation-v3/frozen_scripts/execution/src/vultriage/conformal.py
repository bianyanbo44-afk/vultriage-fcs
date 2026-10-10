"""Conformal set construction and fail-closed diagnostics.

The exact weighted routine implements the test-point mass at infinity from
weighted conformal prediction under covariate shift. Estimated weights remain an
empirical procedure unless their estimation error is separately justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected an (n, 2) probability matrix")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    return probabilities


def _higher_quantile(values: np.ndarray, level: float) -> float:
    if not 0 <= level <= 1:
        raise ValueError("Quantile level must lie in [0, 1]")
    if len(values) == 0:
        raise ValueError("Cannot take a quantile of an empty class")
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, min(len(ordered) - 1, ceil(level * len(ordered)) - 1))
    return float(ordered[index])


def mondrian_thresholds(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    alpha_by_class: dict[int, float],
) -> dict[int, float]:
    """Finite-sample class-conditional split-conformal thresholds."""

    probabilities = _validate_probabilities(calibration_probabilities)
    labels = np.asarray(calibration_labels, dtype=int)
    if labels.shape != (len(probabilities),):
        raise ValueError("Labels and probabilities have incompatible shapes")
    thresholds: dict[int, float] = {}
    for label in (0, 1):
        mask = labels == label
        scores = 1.0 - probabilities[mask, label]
        n_class = len(scores)
        if n_class == 0:
            raise ValueError(f"Calibration class {label} is empty")
        alpha = float(alpha_by_class[label])
        level = min(1.0, ceil((n_class + 1) * (1.0 - alpha)) / n_class)
        thresholds[label] = _higher_quantile(scores, level)
    return thresholds


def conformal_sets(
    test_probabilities: np.ndarray, thresholds: dict[int, float]
) -> np.ndarray:
    """Return a boolean (n, 2) matrix denoting included labels."""

    probabilities = _validate_probabilities(test_probabilities)
    result = np.zeros_like(probabilities, dtype=bool)
    for label in (0, 1):
        result[:, label] = (1.0 - probabilities[:, label]) <= thresholds[label]
    return result


def weighted_quantile_with_infinity(
    scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weight: float,
    level: float,
) -> float:
    """Quantile of calibration scores plus test-point mass at infinity."""

    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(calibration_weights, dtype=float)
    if scores.shape != weights.shape or scores.ndim != 1:
        raise ValueError("Scores and weights must be aligned one-dimensional arrays")
    if len(scores) == 0:
        raise ValueError("Weighted quantile needs at least one calibration score")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("Scores and weights must be finite")
    if (weights < 0).any() or test_weight < 0:
        raise ValueError("Weights must be nonnegative")
    total = float(weights.sum() + test_weight)
    if total <= 0:
        raise ValueError("Total weight must be positive")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order]) / total
    index = int(np.searchsorted(cumulative, level, side="left"))
    if index >= len(order):
        return float("inf")
    return float(scores[order[index]])


def weighted_conformal_sets(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_weights: np.ndarray,
    test_probabilities: np.ndarray,
    test_weights: np.ndarray,
    alpha_by_class: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Test-point-aware class-conditional weighted conformal sets.

    Returns `(sets, thresholds)`, where thresholds has shape `(n_test, 2)`.
    """

    calibration = _validate_probabilities(calibration_probabilities)
    test = _validate_probabilities(test_probabilities)
    labels = np.asarray(calibration_labels, dtype=int)
    weights = np.asarray(calibration_weights, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    if labels.shape != (len(calibration),) or weights.shape != labels.shape:
        raise ValueError("Calibration arrays have incompatible shapes")
    if test_weights.shape != (len(test),):
        raise ValueError("Test weights have incompatible shape")

    thresholds = np.empty((len(test), 2), dtype=float)
    sets = np.zeros((len(test), 2), dtype=bool)
    for label in (0, 1):
        mask = labels == label
        scores = 1.0 - calibration[mask, label]
        class_weights = weights[mask]
        level = 1.0 - float(alpha_by_class[label])
        if len(scores) == 0:
            raise ValueError(f"Calibration class {label} is empty")
        order = np.argsort(scores, kind="stable")
        ordered_scores = scores[order]
        cumulative_weights = np.cumsum(class_weights[order])
        calibration_weight_sum = float(cumulative_weights[-1])
        for index, test_weight in enumerate(test_weights):
            target_mass = level * (calibration_weight_sum + float(test_weight))
            threshold_index = int(
                np.searchsorted(cumulative_weights, target_mass, side="left")
            )
            threshold = (
                float("inf")
                if threshold_index >= len(ordered_scores)
                else float(ordered_scores[threshold_index])
            )
            thresholds[index, label] = threshold
            sets[index, label] = 1.0 - test[index, label] <= threshold
    return sets, thresholds


def kish_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("ESS weights must be a finite nonnegative vector")
    squared_sum = float(np.square(weights).sum())
    return 0.0 if squared_sum == 0 else float(weights.sum() ** 2 / squared_sum)


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reasons: tuple[str, ...]
    total_ess: float
    class_ess: dict[int, float]


def estimated_weight_support(
    calibration_labels: np.ndarray,
    calibration_weights: np.ndarray,
    alpha_by_class: dict[int, float],
    minimum_total_ess: float = 200.0,
    minimum_class_ess: float = 20.0,
    class_ess_multiplier_over_alpha: float = 2.0,
) -> SupportDecision:
    """Apply the preregistered ESS portion of the fail-closed rule."""

    labels = np.asarray(calibration_labels, dtype=int)
    weights = np.asarray(calibration_weights, dtype=float)
    if labels.shape != weights.shape:
        raise ValueError("Labels and weights must have the same shape")
    total_ess = kish_ess(weights)
    class_ess = {label: kish_ess(weights[labels == label]) for label in (0, 1)}
    reasons: list[str] = []
    if total_ess < minimum_total_ess:
        reasons.append(f"total_ess<{minimum_total_ess:g}")
    for label in (0, 1):
        required = max(
            minimum_class_ess,
            ceil(class_ess_multiplier_over_alpha / float(alpha_by_class[label])),
        )
        if class_ess[label] < required:
            reasons.append(f"class_{label}_ess<{required:g}")
    return SupportDecision(
        supported=not reasons,
        reasons=tuple(reasons),
        total_ess=total_ess,
        class_ess=class_ess,
    )
