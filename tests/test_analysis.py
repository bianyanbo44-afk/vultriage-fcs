import numpy as np

from analyze_e1 import LOWER_IS_BETTER, holm_adjust, paired_hodges_lehmann


def test_holm_adjust_is_monotone_in_sorted_order():
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert all(0 <= value <= 1 for value in adjusted)
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= 0)


def test_paired_hodges_lehmann_uses_walsh_averages():
    assert paired_hodges_lehmann(np.array([1.0, 2.0, 3.0])) == 2.0


def test_metric_direction_distinguishes_coverage_from_review_load():
    assert LOWER_IS_BETTER["singleton_coverage"] is False
    assert LOWER_IS_BETTER["review_load"] is True
