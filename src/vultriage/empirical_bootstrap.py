"""Exact descriptive percentile intervals for an even-n empirical bootstrap.

This enumerates the distribution of the mean of the two central order
statistics. It removes Monte Carlo error, not small-sample uncertainty,
selection bias, or the assumption of exchangeable resampling units.
"""
from functools import lru_cache
from math import comb, factorial

import numpy as np


@lru_cache(maxsize=16)
def adjacent_order_probabilities(n: int) -> np.ndarray:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("An even sample size of at least two is required")
    m = n // 2
    weights = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            left = ((i + 1) / n) ** m - (i / n) ** m
            right = ((n - j) / n) ** m - ((n - j - 1) / n) ** m
            weights[i, j] = comb(n, m) * left * right
        for below in range(m):
            for above in range(m):
                equal = n - below - above
                coefficient = factorial(n) / (
                    factorial(below) * factorial(equal) * factorial(above)
                )
                weights[i, i] += (
                    coefficient * (i / n) ** below * (1 / n) ** equal
                    * ((n - i - 1) / n) ** above
                )
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ArithmeticError("Bootstrap order probabilities do not sum to one")
    weights.setflags(write=False)
    return weights


def distribution(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Empirical values must be a finite vector")
    weights = adjacent_order_probabilities(len(values))
    ordered = np.sort(values)
    mass = {}
    for i in range(len(values)):
        for j in range(i, len(values)):
            point = float((ordered[i] + ordered[j]) / 2)
            mass[point] = mass.get(point, 0) + weights[i, j]
    support = np.array(sorted(mass), dtype=float)
    probabilities = np.array([mass[point] for point in support], dtype=float)
    return support, probabilities


def percentile_interval(values, lower=.025, upper=.975) -> tuple[float, float]:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower <= upper <= 1")
    support, probability = distribution(values)
    cumulative = np.cumsum(probability)
    cumulative[-1] = 1
    return tuple(
        float(support[np.searchsorted(cumulative, quantile)])
        for quantile in (lower, upper)
    )
