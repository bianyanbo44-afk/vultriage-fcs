import numpy as np

from vultriage.conformal import (
    conformal_sets,
    estimated_weight_support,
    kish_ess,
    mondrian_thresholds,
    weighted_conformal_sets,
    weighted_quantile_with_infinity,
)


def test_mondrian_sets_include_high_probability_true_classes():
    calibration_probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    labels = np.array([0, 0, 1, 1])
    thresholds = mondrian_thresholds(
        calibration_probabilities, labels, {0: 0.5, 1: 0.5}
    )
    sets = conformal_sets(np.array([[0.85, 0.15], [0.15, 0.85]]), thresholds)
    assert sets[0, 0]
    assert sets[1, 1]


def test_test_point_mass_can_force_infinite_threshold():
    threshold = weighted_quantile_with_infinity(
        scores=np.array([0.1, 0.2]),
        calibration_weights=np.array([1.0, 1.0]),
        test_weight=100.0,
        level=0.9,
    )
    assert np.isinf(threshold)


def test_uniform_weighted_sets_reduce_to_conservative_sets():
    calibration_probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    labels = np.array([0, 0, 1, 1])
    sets, thresholds = weighted_conformal_sets(
        calibration_probabilities,
        labels,
        np.ones(4),
        np.array([[0.7, 0.3]]),
        np.ones(1),
        {0: 0.5, 1: 0.5},
    )
    assert sets.shape == (1, 2)
    assert thresholds.shape == (1, 2)


def test_kish_ess_and_fail_closed_rule():
    assert kish_ess(np.ones(10)) == 10.0
    decision = estimated_weight_support(
        calibration_labels=np.array([0] * 30 + [1] * 30),
        calibration_weights=np.ones(60),
        alpha_by_class={0: 0.1, 1: 0.1},
        minimum_total_ess=50,
        minimum_class_ess=20,
        class_ess_multiplier_over_alpha=2,
    )
    assert decision.supported


def test_fail_closed_when_vulnerable_ess_is_too_small():
    decision = estimated_weight_support(
        calibration_labels=np.array([0] * 100 + [1] * 5),
        calibration_weights=np.ones(105),
        alpha_by_class={0: 0.1, 1: 0.05},
        minimum_total_ess=50,
        minimum_class_ess=20,
        class_ess_multiplier_over_alpha=2,
    )
    assert not decision.supported
    assert any(reason.startswith("class_1_ess") for reason in decision.reasons)

