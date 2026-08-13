import numpy as np
from scipy import sparse

from vultriage.domain import clipped_weight_summary, crossfit_density_ratio


def test_crossfit_density_ratio_is_complete_finite_and_deterministic():
    rng = np.random.default_rng(7)
    source = rng.normal(-0.5, 1.0, size=(60, 4))
    target = rng.normal(0.5, 1.0, size=(60, 4))
    features = sparse.csr_matrix(np.vstack([source, target]))
    row_ids = [f"row-{index}" for index in range(120)]
    labels = np.array([0] * 60 + [1] * 60)
    first = crossfit_density_ratio(
        features,
        row_ids,
        labels,
        folds=5,
        alpha=1e-3,
        epochs=5,
        seed=13,
        salt="test",
    )
    second = crossfit_density_ratio(
        features,
        row_ids,
        labels,
        folds=5,
        alpha=1e-3,
        epochs=5,
        seed=13,
        salt="test",
    )
    assert np.isfinite(first.raw_ratios).all()
    assert (first.raw_ratios > 0).all()
    assert np.allclose(first.target_probabilities, second.target_probabilities)
    assert first.diagnostics["out_of_fold"] is True


def test_clipped_weight_summary_obeys_bounds():
    summary = clipped_weight_summary(np.array([0.001, 1.0, 1000.0]), 20.0)
    assert summary["quantiles"]["min"] == 0.05
    assert summary["quantiles"]["max"] == 20.0
    assert summary["clipped_low"] == 1
    assert summary["clipped_high"] == 1
