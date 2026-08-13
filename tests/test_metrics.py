import numpy as np

from vultriage.metrics import aurc, equal_mass_ece, fnr_at_fpr, triage_metrics


def test_triage_metrics_count_singletons_and_miscoverage():
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.4, 0.4])
    sets = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=bool)
    metrics = triage_metrics(labels, probabilities, sets)
    assert metrics["singleton_coverage"] == 0.75
    assert metrics["review_load"] == 0.25
    assert metrics["vulnerable_miscoverage"] == 0.5
    assert metrics["safe_miscoverage"] == 0.0
    assert metrics["vulnerable_singleton_rate"] == 1.0
    assert metrics["safe_singleton_rate"] == 0.5


def test_secondary_metrics_are_bounded_and_perfect_ranking_has_zero_fnr():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.98, 0.99])
    assert 0 <= equal_mass_ece(labels, probabilities, bins=2) <= 1
    assert 0 <= aurc(labels, probabilities) <= 1
    assert fnr_at_fpr(labels, probabilities, maximum_fpr=0.005) == 0.0
