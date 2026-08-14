"""Deterministic binary adapter derived from PROM's four experts.

This module is an independent, minimal reimplementation for sealed binary
probability artifacts.  It deliberately does not import PROM or MAPIE.  The
differences from the referenced PROM commit are documented in
``research/PROM_PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EXPERT_NAMES = ("lac", "topk", "aps", "raps")
RAPS_LAMBDA_GRID = (0.001, 0.01, 0.1, 0.2, 0.5)
RAPS_TUNING_FRACTION = 0.2


@dataclass(frozen=True)
class PromExpertResult:
    """Auditable outputs for one PROM-derived expert."""

    calibration_scores: np.ndarray
    test_scores: np.ndarray
    p_values: np.ndarray
    prediction_sets: np.ndarray
    singleton: np.ndarray
    credible: np.ndarray
    accepted: np.ndarray
    rejected: np.ndarray


@dataclass(frozen=True)
class PromUnionResult:
    """PROM's conservative mixture: accept only if every expert accepts."""

    accepted: np.ndarray
    rejected: np.ndarray


@dataclass(frozen=True)
class PromAdapterResult:
    """Complete output of :func:`prom_binary_adapter`."""

    alpha: float
    seed: int
    predicted_labels: np.ndarray
    neighbor_counts: Mapping[str, int]
    raps_tuning_size: int
    raps_calibration_size: int
    raps_k_reg: int
    raps_lambda: float
    experts: Mapping[str, PromExpertResult]
    union: PromUnionResult


def _validate_probabilities(probabilities: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_samples, 2)")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError(f"{name} rows must sum to one")

    # Canonicalize onto the binary simplex.  This makes probability-space
    # Euclidean neighbor ordering exact and independent of tiny sum drift.
    normalized = values / row_sums[:, np.newaxis]
    normalized[:, 0] = 1.0 - normalized[:, 1]
    return normalized


def _validate_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or len(raw) != n_samples:
        raise ValueError("calibration_labels must align with calibration_probabilities")
    if not np.isfinite(raw).all():
        raise ValueError("calibration_labels must contain only finite values")
    if not np.all(np.equal(raw, np.floor(raw))):
        raise ValueError("calibration_labels must contain integer class indices")
    encoded = raw.astype(np.int64)
    if not np.isin(encoded, (0, 1)).all():
        raise ValueError("calibration_labels must contain only 0 and 1")
    return encoded


def _binary_one_hot(labels: np.ndarray) -> np.ndarray:
    """Return an explicit two-column encoding, including for binary targets."""

    labels = np.asarray(labels, dtype=np.int64)
    return labels[:, np.newaxis] == np.array([0, 1], dtype=np.int64)


def _descending_order(probabilities: np.ndarray) -> np.ndarray:
    # PROM's descending ranking is produced by reversing the ascending
    # argsort.  Preserve that class-1-first tie rule for exact fidelity to
    # the audited upstream implementation; argmax remains class-0-first for
    # the detector's forced label outside this helper.
    ascending = np.argsort(probabilities, axis=1, kind="stable")
    return np.flip(ascending, axis=1)


def _expert_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    expert: str,
    *,
    uniforms: np.ndarray | None = None,
    raps_k_reg: int = 1,
    raps_lambda: float = 0.0,
) -> np.ndarray:
    """Compute one score per row for the supplied (possibly predicted) label."""

    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert: {expert}")
    one_hot = _binary_one_hot(labels)
    true_probabilities = np.sum(probabilities * one_hot, axis=1)
    if expert == "lac":
        return 1.0 - true_probabilities

    order = _descending_order(probabilities)
    sorted_one_hot = np.take_along_axis(one_hot, order, axis=1)
    zero_based_rank = np.argmax(sorted_one_hot, axis=1)
    if expert == "topk":
        return zero_based_rank.astype(np.float64)

    if uniforms is None:
        raise ValueError(f"{expert} requires explicit random uniforms")
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if uniforms.shape != (len(probabilities),):
        raise ValueError("uniforms must contain one value per probability row")
    if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("uniforms must lie in [0, 1)")

    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    score = cumulative[np.arange(len(probabilities)), zero_based_rank]
    score = score - uniforms * true_probabilities
    if expert == "raps":
        one_based_rank = zero_based_rank + 1
        score = score + raps_lambda * np.maximum(
            one_based_rank - int(raps_k_reg), 0
        )
    return score


def _higher_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """PROM-compatible finite-sample higher quantile.

    ``np.quantile(..., method="higher")`` is intentionally used instead of
    interpolating or silently substituting a different order-statistic rule.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("conformity scores must be nonempty and finite")
    quantile = ((len(values) + 1) * (1 - alpha)) / len(values)
    quantile = min(1.0, max(0.0, float(quantile)))
    return float(np.quantile(values, quantile, method="higher"))


def _rank_based_prediction_sets(
    probabilities: np.ndarray,
    maximum_zero_based_rank: int,
) -> np.ndarray:
    order = _descending_order(probabilities)
    rank = min(max(int(maximum_zero_based_rank), 0), 1)
    last_probability = probabilities[
        np.arange(len(probabilities)), order[:, rank]
    ]
    # PROM includes all labels tied with the boundary label.
    return probabilities >= last_probability[:, np.newaxis]


def _cumulative_prediction_sets(
    probabilities: np.ndarray,
    threshold: float,
    *,
    raps_k_reg: int | None = None,
    raps_lambda: float = 0.0,
) -> np.ndarray:
    order = _descending_order(probabilities)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    if raps_k_reg is not None:
        positions = np.arange(1, probabilities.shape[1] + 1)
        cumulative = cumulative + raps_lambda * np.maximum(
            positions[np.newaxis, :] - int(raps_k_reg), 0
        )

    reaches_threshold = cumulative >= threshold
    first_reaching_rank = np.argmax(reaches_threshold, axis=1)
    first_reaching_rank = np.where(
        reaches_threshold.any(axis=1),
        first_reaching_rank,
        probabilities.shape[1] - 1,
    )
    last_class = order[np.arange(len(probabilities)), first_reaching_rank]
    last_probability = probabilities[np.arange(len(probabilities)), last_class]
    return probabilities >= last_probability[:, np.newaxis]


def _split_raps_calibration(
    n_calibration: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tuning_size = max(1, int(np.ceil(RAPS_TUNING_FRACTION * n_calibration)))
    permutation = rng.permutation(n_calibration)
    return permutation[tuning_size:], permutation[:tuning_size]


def _tune_raps(
    tuning_probabilities: np.ndarray,
    tuning_labels: np.ndarray,
    alpha: float,
) -> tuple[int, float]:
    """Select PROM's RAPS parameters on a calibration-only tuning view."""

    ranks = _expert_scores(tuning_probabilities, tuning_labels, "topk")
    k_reg = int(_higher_conformal_quantile(ranks, alpha)) + 1

    best_lambda = float(RAPS_LAMBDA_GRID[0])
    best_mean_size = np.inf
    zero_uniforms = np.zeros(len(tuning_probabilities), dtype=np.float64)
    for candidate in RAPS_LAMBDA_GRID:
        scores = _expert_scores(
            tuning_probabilities,
            tuning_labels,
            "raps",
            uniforms=zero_uniforms,
            raps_k_reg=k_reg,
            raps_lambda=float(candidate),
        )
        threshold = _higher_conformal_quantile(scores, alpha)
        prediction_sets = _cumulative_prediction_sets(
            tuning_probabilities,
            threshold,
            raps_k_reg=k_reg,
            raps_lambda=float(candidate),
        )
        mean_size = float(prediction_sets.sum(axis=1).mean())
        if mean_size < best_mean_size:
            best_mean_size = mean_size
            best_lambda = float(candidate)
    return k_reg, best_lambda


def _neighbor_count(n_calibration: int) -> int:
    return n_calibration if n_calibration < 200 else int(np.floor(0.1 * n_calibration))


def _prepare_neighbor_index(
    calibration_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original_indices = np.arange(len(calibration_probabilities), dtype=np.int64)
    sorted_indices = np.lexsort((original_indices, calibration_probabilities[:, 1]))
    return calibration_probabilities[sorted_indices, 1], sorted_indices


def _nearest_indices_for_probability(
    sorted_values: np.ndarray,
    sorted_indices: np.ndarray,
    target_value: float,
    k: int,
) -> np.ndarray:
    """Exact binary-simplex kNN with deterministic original-index tie breaks."""

    n_calibration = len(sorted_values)
    if k == n_calibration:
        return np.arange(n_calibration, dtype=np.int64)

    insertion = int(np.searchsorted(sorted_values, target_value, side="left"))
    lower = max(0, insertion - k)
    upper = min(n_calibration, insertion + k)
    if upper - lower < k:
        if lower == 0:
            upper = k
        else:
            lower = n_calibration - k

    initial_distances = np.abs(sorted_values[lower:upper] - target_value)
    boundary_distance = float(np.partition(initial_distances, k - 1)[k - 1])

    # Expand through all rows no farther than the kth distance.  A small ULP
    # tolerance is required because subtracting target +/- distance can land
    # one representable value inside the exact endpoint (for example, 0.375 -
    # 0.35 versus 0.025).  Final ordering is by computed distance and original
    # row index, so the tolerance cannot change which k rows are retained.
    tolerance = 8.0 * np.finfo(np.float64).eps * max(
        1.0, abs(target_value), boundary_distance
    )
    value_lower = target_value - boundary_distance - tolerance
    value_upper = target_value + boundary_distance + tolerance
    candidate_lower = int(
        np.searchsorted(sorted_values, value_lower, side="left")
    )
    candidate_upper = int(
        np.searchsorted(sorted_values, value_upper, side="right")
    )
    candidate_indices = sorted_indices[candidate_lower:candidate_upper]
    candidate_distances = np.abs(
        sorted_values[candidate_lower:candidate_upper] - target_value
    )
    candidate_order = np.lexsort((candidate_indices, candidate_distances))
    selected = candidate_indices[candidate_order[:k]]
    if len(selected) != k:
        raise RuntimeError("failed to select the requested number of neighbors")
    return selected


def _nearest_calibration_indices(
    calibration_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    k: int,
) -> np.ndarray:
    """Materialized helper used by tests and small diagnostic runs."""

    sorted_values, sorted_indices = _prepare_neighbor_index(
        calibration_probabilities
    )
    return np.vstack(
        [
            _nearest_indices_for_probability(
                sorted_values, sorted_indices, float(probability[1]), k
            )
            for probability in target_probabilities
        ]
    )


def _local_empirical_p_values(
    calibration_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    calibration_scores: np.ndarray,
    target_scores: np.ndarray,
    k: int,
    neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute each test point's four local p-values without cross-row pooling."""

    if neighbor_indices is None:
        sorted_values, sorted_indices = _prepare_neighbor_index(
            calibration_probabilities
        )
    else:
        neighbors = np.asarray(neighbor_indices, dtype=np.int64)
        if neighbors.shape != (len(target_probabilities), int(k)):
            raise ValueError("neighbor_indices must align with target rows and k")
        if np.any(neighbors < 0) or np.any(neighbors >= len(calibration_probabilities)):
            raise ValueError("neighbor_indices contain an out-of-range row")
    p_values = np.empty_like(target_scores, dtype=np.float64)
    for row_index, probability in enumerate(target_probabilities):
        if neighbor_indices is None:
            row_neighbors = _nearest_indices_for_probability(
                sorted_values,
                sorted_indices,
                float(probability[1]),
                k,
            )
        else:
            row_neighbors = neighbors[row_index]
        # Preserve PROM's empirical >= comparison and its no-+1 convention.
        p_values[row_index] = np.mean(
            calibration_scores[row_neighbors] >= target_scores[row_index],
            axis=0,
        )
    return p_values


def _combine_expert_acceptance(
    accepted_by_expert: np.ndarray,
) -> PromUnionResult:
    accepted = np.logical_and.reduce(accepted_by_expert, axis=0)
    rejected = np.logical_or.reduce(~accepted_by_expert, axis=0)
    return PromUnionResult(accepted=accepted, rejected=rejected)


def _expert_acceptance(
    prediction_sets: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    singleton = np.asarray(prediction_sets, dtype=bool).sum(axis=1) == 1
    credible = np.asarray(p_values, dtype=np.float64) > 1.0 - alpha
    accepted = singleton & credible
    return singleton, credible, accepted, ~accepted


def prom_binary_adapter(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> PromAdapterResult:
    """Run the deterministic PROM-derived adapter on binary probabilities.

    ``alpha`` and ``seed`` are mandatory and fixed for the whole call.  The API
    intentionally has no target-label argument, so neither expert parameters
    nor acceptance thresholds can be selected using target outcomes.
    """

    if isinstance(alpha, bool) or not np.isscalar(alpha):
        raise ValueError("alpha must be a scalar strictly between zero and one")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an explicit nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")

    calibration = _validate_probabilities(
        calibration_probabilities, "calibration_probabilities"
    )
    target = _validate_probabilities(target_probabilities, "target_probabilities")
    labels = _validate_labels(calibration_labels, len(calibration))
    if len(calibration) < 2:
        raise ValueError("at least two calibration rows are required for RAPS")
    predicted_labels = np.argmax(target, axis=1).astype(np.int64)

    seed_streams = np.random.SeedSequence(seed).spawn(5)
    raps_calibration_indices, raps_tuning_indices = _split_raps_calibration(
        len(calibration), np.random.default_rng(seed_streams[0])
    )
    raps_calibration = calibration[raps_calibration_indices]
    raps_calibration_labels = labels[raps_calibration_indices]
    raps_k_reg, raps_lambda = _tune_raps(
        calibration[raps_tuning_indices],
        labels[raps_tuning_indices],
        alpha,
    )
    aps_calibration_uniforms = np.random.default_rng(seed_streams[1]).random(
        len(calibration)
    )
    aps_target_uniforms = np.random.default_rng(seed_streams[2]).random(len(target))
    raps_calibration_uniforms = np.random.default_rng(seed_streams[3]).random(
        len(raps_calibration)
    )
    raps_target_uniforms = np.random.default_rng(seed_streams[4]).random(len(target))

    calibration_probabilities_by_expert = {
        "lac": calibration,
        "topk": calibration,
        "aps": calibration,
        "raps": raps_calibration,
    }
    calibration_scores_by_expert: dict[str, np.ndarray] = {}
    target_scores_by_expert: dict[str, np.ndarray] = {}
    prediction_sets_by_expert: dict[str, np.ndarray] = {}

    lac_calibration_scores = _expert_scores(calibration, labels, "lac")
    lac_target_scores = _expert_scores(target, predicted_labels, "lac")
    lac_threshold = _higher_conformal_quantile(lac_calibration_scores, alpha)
    prediction_sets_by_expert["lac"] = target >= 1.0 - lac_threshold
    calibration_scores_by_expert["lac"] = lac_calibration_scores
    target_scores_by_expert["lac"] = lac_target_scores

    topk_calibration_scores = _expert_scores(calibration, labels, "topk")
    topk_target_scores = _expert_scores(target, predicted_labels, "topk")
    topk_threshold = int(
        _higher_conformal_quantile(topk_calibration_scores, alpha)
    )
    prediction_sets_by_expert["topk"] = _rank_based_prediction_sets(
        target, topk_threshold
    )
    calibration_scores_by_expert["topk"] = topk_calibration_scores
    target_scores_by_expert["topk"] = topk_target_scores

    aps_calibration_scores = _expert_scores(
        calibration,
        labels,
        "aps",
        uniforms=aps_calibration_uniforms,
    )
    aps_target_scores = _expert_scores(
        target,
        predicted_labels,
        "aps",
        uniforms=aps_target_uniforms,
    )
    aps_threshold = _higher_conformal_quantile(aps_calibration_scores, alpha)
    prediction_sets_by_expert["aps"] = _cumulative_prediction_sets(
        target, aps_threshold
    )
    calibration_scores_by_expert["aps"] = aps_calibration_scores
    target_scores_by_expert["aps"] = aps_target_scores

    raps_calibration_scores = _expert_scores(
        raps_calibration,
        raps_calibration_labels,
        "raps",
        uniforms=raps_calibration_uniforms,
        raps_k_reg=raps_k_reg,
        raps_lambda=raps_lambda,
    )
    raps_target_scores = _expert_scores(
        target,
        predicted_labels,
        "raps",
        uniforms=raps_target_uniforms,
        raps_k_reg=raps_k_reg,
        raps_lambda=raps_lambda,
    )
    raps_threshold = _higher_conformal_quantile(raps_calibration_scores, alpha)
    prediction_sets_by_expert["raps"] = _cumulative_prediction_sets(
        target,
        raps_threshold,
        raps_k_reg=raps_k_reg,
        raps_lambda=raps_lambda,
    )
    calibration_scores_by_expert["raps"] = raps_calibration_scores
    target_scores_by_expert["raps"] = raps_target_scores

    neighbor_counts: dict[str, int] = {}
    p_values_by_expert: dict[str, np.ndarray] = {}
    common_k = _neighbor_count(len(calibration))
    common_neighbors = _nearest_calibration_indices(calibration, target, common_k)
    for expert in ("lac", "topk", "aps"):
        expert_calibration = calibration_probabilities_by_expert[expert]
        neighbor_counts[expert] = common_k
        p_values_by_expert[expert] = _local_empirical_p_values(
            expert_calibration,
            target,
            calibration_scores_by_expert[expert],
            target_scores_by_expert[expert],
            common_k,
            neighbor_indices=common_neighbors,
        )
    del common_neighbors
    raps_k = _neighbor_count(len(raps_calibration))
    neighbor_counts["raps"] = raps_k
    raps_neighbors = _nearest_calibration_indices(raps_calibration, target, raps_k)
    p_values_by_expert["raps"] = _local_empirical_p_values(
        raps_calibration,
        target,
        calibration_scores_by_expert["raps"],
        target_scores_by_expert["raps"],
        raps_k,
        neighbor_indices=raps_neighbors,
    )
    del raps_neighbors

    expert_results: dict[str, PromExpertResult] = {}
    accepted_columns = []
    for expert in EXPERT_NAMES:
        prediction_sets = prediction_sets_by_expert[expert]
        p_values = p_values_by_expert[expert]
        singleton, credible, accepted, rejected = _expert_acceptance(
            prediction_sets, p_values, alpha
        )
        accepted_columns.append(accepted)
        expert_results[expert] = PromExpertResult(
            calibration_scores=calibration_scores_by_expert[expert],
            test_scores=target_scores_by_expert[expert],
            p_values=p_values,
            prediction_sets=prediction_sets,
            singleton=singleton,
            credible=credible,
            accepted=accepted,
            rejected=rejected,
        )

    union = _combine_expert_acceptance(np.vstack(accepted_columns))
    return PromAdapterResult(
        alpha=alpha,
        seed=seed,
        predicted_labels=predicted_labels,
        neighbor_counts=neighbor_counts,
        raps_tuning_size=len(raps_tuning_indices),
        raps_calibration_size=len(raps_calibration_indices),
        raps_k_reg=raps_k_reg,
        raps_lambda=raps_lambda,
        experts=expert_results,
        union=union,
    )
