import numpy as np
import pytest
from scipy.stats import norm

from analyze_ratio_diagnostics_v4 import class_sets, exact_ratio, loss_decomposition, target_density
from vultriage.conformal import weighted_conformal_sets


@pytest.mark.parametrize("seed", [13, 37, 73])
def test_vector_quantiles_match_independent_existing_scalar_routine(seed):
    rng = np.random.default_rng(seed)
    source = rng.choice([0.05, 0.2, 0.5, 0.7, 0.95], 80)
    labels = np.tile([0, 1], 40)
    target = rng.random(30)
    weights_source = rng.uniform(0.05, 20, 80)
    weights_target = np.concatenate([rng.uniform(0.05, 20, 29), [10000]])
    a, q = class_sets(source, labels, weights_source, target, weights_target, 0.2, 0.1)
    expected, expected_q = weighted_conformal_sets(np.column_stack([1 - source, source]), labels, weights_source,
                                                  np.column_stack([1 - target, target]), weights_target, {0: 0.2, 1: 0.1})
    np.testing.assert_array_equal(a, expected)
    np.testing.assert_allclose(q, expected_q)


@pytest.mark.parametrize("specification", ["Normal(0.5,1)", "Normal(1.5,1)", "0.5*Normal(-2,1)+0.5*Normal(2,1)"])
def test_analytic_ratio_reconstructs_target_density(specification):
    regime = {"target_distribution": specification, "target_conditional_delta": 0.0}
    x = np.linspace(-6, 6, 1000)
    np.testing.assert_allclose(norm.pdf(x) * exact_ratio(x, regime), target_density(x, regime), rtol=1e-12, atol=1e-14)


def test_fixed_loss_decomposition_has_known_normal_total_variation():
    x = np.linspace(-10, 10, 20001)
    regime = {"target_distribution": "Normal(1.5,1)", "target_conditional_delta": 0.0}
    result = loss_decomposition(x, regime, np.ones(len(x)))
    assert result["covariate_tv"] == pytest.approx(2 * norm.cdf(0.75) - 1, abs=1e-6)
    assert result["conditional_label_tv"] == 0
    assert result["fixed_risk_absolute_gap"] <= result["fixed_loss_bound"]
    oracle = loss_decomposition(x, regime, exact_ratio(x, regime))
    assert oracle["covariate_tv"] < 1e-12
    assert oracle["fixed_risk_absolute_gap"] < 1e-12


def test_oracle_ratio_does_not_repair_conditional_label_shift():
    x = np.linspace(-10, 10, 20001)
    regime = {"target_distribution": "Normal(1.5,1)", "target_conditional_delta": 1.0}
    result = loss_decomposition(x, regime, exact_ratio(x, regime))
    assert result["covariate_tv"] < 1e-12
    assert result["conditional_label_tv"] > 0.05
    assert result["fixed_risk_absolute_gap"] > 0.01
    assert result["fixed_risk_absolute_gap"] <= result["fixed_loss_bound"]


def test_weight_normalization_does_not_change_sets():
    score = np.linspace(0.01, 0.99, 80)
    labels = np.tile([0, 1], 40)
    target = np.linspace(0.05, 0.95, 50)
    source_w = np.linspace(0.1, 20, 80)
    target_w = np.linspace(0.05, 50, 50)
    a, q = class_sets(score, labels, source_w, target, target_w, 0.2, 0.1)
    b, r = class_sets(score, labels, source_w * 7, target, target_w * 7, 0.2, 0.1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(q, r)
