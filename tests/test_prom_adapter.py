import inspect

import numpy as np
import pytest

from vultriage.prom_adapter import (
    EXPERT_NAMES,
    RAPS_LAMBDA_GRID,
    _binary_one_hot,
    _combine_expert_acceptance,
    _expert_acceptance,
    _expert_scores,
    _local_empirical_p_values,
    _nearest_calibration_indices,
    _neighbor_count,
    _descending_order,
    _higher_conformal_quantile,
    prom_binary_adapter,
)


def _binary_probabilities(positive_probabilities):
    positive = np.asarray(positive_probabilities, dtype=float)
    return np.column_stack((1.0 - positive, positive))


def test_binary_one_hot_always_has_two_columns():
    encoded = _binary_one_hot(np.array([0, 1, 1, 0]))
    assert encoded.shape == (4, 2)
    np.testing.assert_array_equal(
        encoded,
        np.array(
            [[True, False], [False, True], [False, True], [True, False]]
        ),
    )


def test_four_expert_scores_match_binary_golden_example():
    probabilities = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    )
    labels = np.array([0, 0, 1, 1])
    uniforms = np.array([0.0, 0.5, 0.25, 0.75])

    np.testing.assert_allclose(
        _expert_scores(probabilities, labels, "lac"),
        [0.2, 0.7, 0.6, 0.4],
    )
    np.testing.assert_array_equal(
        _expert_scores(probabilities, labels, "topk"),
        [0.0, 1.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        _expert_scores(
            probabilities, labels, "aps", uniforms=uniforms
        ),
        [0.8, 0.85, 0.9, 0.15],
    )
    np.testing.assert_allclose(
        _expert_scores(
            probabilities,
            labels,
            "raps",
            uniforms=uniforms,
            raps_k_reg=1,
            raps_lambda=0.2,
        ),
        [0.8, 1.05, 1.1, 0.15],
    )


def test_prom_tie_order_and_higher_quantile_match_audited_rules():
    # The audited PROM implementation reverses ascending argsort, so class 1
    # precedes class 0 when the binary probabilities are exactly tied.
    np.testing.assert_array_equal(
        _descending_order(np.array([[0.5, 0.5], [0.7, 0.3]])),
        np.array([[1, 0], [0, 1]]),
    )
    # n=3, alpha=.5 gives q=((n+1)(1-alpha))/n=2/3; ``higher`` selects the
    # third order value, unlike a rounded finite-sample rank of two.
    assert _higher_conformal_quantile(np.array([0.1, 0.2, 0.9]), 0.5) == 0.9


def test_neighbor_count_uses_all_below_200_and_floor_ten_percent_otherwise():
    assert _neighbor_count(199) == 199
    assert _neighbor_count(200) == 20
    assert _neighbor_count(209) == 20
    assert _neighbor_count(210) == 21


def test_probability_space_knn_breaks_exact_distance_ties_by_original_index():
    calibration = _binary_probabilities([0.2, 0.1, 0.3, 0.1])
    target = _binary_probabilities([0.1])
    neighbors = _nearest_calibration_indices(calibration, target, k=2)
    np.testing.assert_array_equal(neighbors, [[1, 3]])


def test_probability_space_knn_matches_binary_simplex_euclidean_selection():
    rng = np.random.default_rng(20260814)
    for n_calibration in (2, 3, 10, 200):
        for _ in range(10):
            # Discrete probabilities deliberately create duplicate values and
            # exact distance ties.
            calibration = _binary_probabilities(
                rng.integers(0, 41, size=n_calibration) / 40.0
            )
            target = _binary_probabilities(rng.integers(0, 41, size=7) / 40.0)
            for k in sorted({1, min(2, n_calibration), min(7, n_calibration)}):
                actual = _nearest_calibration_indices(calibration, target, k)
                expected = []
                for row in target:
                    # On the binary simplex the Euclidean distance is
                    # sqrt(2) * abs(delta p_1).  Use that exact ordering so
                    # mathematical ties are resolved only by original index,
                    # not by two-column floating-point cancellation.
                    distances = np.abs(calibration[:, 1] - row[1])
                    expected.append(
                        np.lexsort((np.arange(n_calibration), distances))[:k]
                    )
                np.testing.assert_array_equal(
                    np.sort(actual, axis=1), np.sort(np.vstack(expected), axis=1)
                )


def test_local_p_values_are_computed_per_test_point_without_pooling():
    calibration = _binary_probabilities(np.linspace(0.0, 1.0, 200))
    target = _binary_probabilities([0.025, 0.975])
    calibration_scores = np.linspace(0.0, 1.0, 200)
    target_scores = np.array([0.5, 0.5])

    p_values = _local_empirical_p_values(
        calibration,
        target,
        calibration_scores,
        target_scores,
        k=20,
    )
    np.testing.assert_array_equal(p_values, [0.0, 1.0])


def test_local_p_value_preserves_greater_equal_and_has_no_plus_one():
    calibration = _binary_probabilities([0.2, 0.8])
    target = _binary_probabilities([0.5])
    p_values = _local_empirical_p_values(
        calibration,
        target,
        calibration_scores=np.array([0.2, 0.5]),
        target_scores=np.array([0.5]),
        k=2,
    )
    # Exactly one of two calibration scores is >= the test score.  A +1
    # conformal correction would instead return 2/3.
    np.testing.assert_array_equal(p_values, [0.5])


def test_strict_credibility_boundary_and_union_rejection_semantics():
    singleton_sets = np.array([[True, False], [False, True]])
    singleton, credible, accepted, rejected = _expert_acceptance(
        singleton_sets,
        p_values=np.array([0.9, np.nextafter(0.9, 1.0)]),
        alpha=0.1,
    )
    np.testing.assert_array_equal(singleton, [True, True])
    np.testing.assert_array_equal(credible, [False, True])
    np.testing.assert_array_equal(accepted, [False, True])
    np.testing.assert_array_equal(rejected, [True, False])

    combined = _combine_expert_acceptance(
        np.array(
            [
                [True, True, False],
                [True, False, True],
                [True, True, True],
                [True, True, True],
            ]
        )
    )
    np.testing.assert_array_equal(combined.accepted, [True, False, False])
    np.testing.assert_array_equal(combined.rejected, [False, True, True])


def test_complete_adapter_is_seeded_auditable_and_returns_every_expert():
    rng = np.random.default_rng(90210)
    calibration = _binary_probabilities(rng.uniform(0.01, 0.99, size=250))
    labels = (np.arange(250) % 2).astype(int)
    target = _binary_probabilities([0.08, 0.35, 0.61, 0.93])

    first = prom_binary_adapter(
        calibration, labels, target, alpha=0.2, seed=37
    )
    second = prom_binary_adapter(
        calibration, labels, target, alpha=0.2, seed=37
    )
    different_seed = prom_binary_adapter(
        calibration, labels, target, alpha=0.2, seed=73
    )

    assert tuple(first.experts) == EXPERT_NAMES
    assert first.raps_tuning_size == 50
    assert first.raps_calibration_size == 200
    assert first.neighbor_counts == {
        "lac": 25,
        "topk": 25,
        "aps": 25,
        "raps": 20,
    }
    assert first.raps_k_reg in (1, 2)
    assert first.raps_lambda in RAPS_LAMBDA_GRID
    np.testing.assert_array_equal(first.predicted_labels, [0, 0, 1, 1])

    for expert in EXPERT_NAMES:
        left = first.experts[expert]
        right = second.experts[expert]
        np.testing.assert_array_equal(left.calibration_scores, right.calibration_scores)
        np.testing.assert_array_equal(left.test_scores, right.test_scores)
        np.testing.assert_array_equal(left.p_values, right.p_values)
        np.testing.assert_array_equal(left.prediction_sets, right.prediction_sets)
        np.testing.assert_array_equal(left.accepted, right.accepted)
        np.testing.assert_array_equal(left.rejected, ~left.accepted)
        assert left.prediction_sets.shape == (4, 2)
        assert left.p_values.shape == (4,)

    np.testing.assert_array_equal(first.union.accepted, second.union.accepted)
    np.testing.assert_array_equal(first.union.rejected, ~first.union.accepted)
    np.testing.assert_array_equal(
        first.union.rejected,
        np.logical_or.reduce(
            np.vstack([first.experts[name].rejected for name in EXPERT_NAMES]),
            axis=0,
        ),
    )
    assert not np.array_equal(
        first.experts["aps"].test_scores,
        different_seed.experts["aps"].test_scores,
    )


def test_public_api_requires_fixed_alpha_and_seed_and_has_no_target_labels():
    parameters = inspect.signature(prom_binary_adapter).parameters
    assert "target_labels" not in parameters
    assert parameters["alpha"].default is inspect.Parameter.empty
    assert parameters["seed"].default is inspect.Parameter.empty

    calibration = _binary_probabilities([0.1, 0.9])
    labels = np.array([0, 1])
    target = _binary_probabilities([0.5])
    with pytest.raises(ValueError, match="alpha"):
        prom_binary_adapter(calibration, labels, target, alpha=0.0, seed=13)
    with pytest.raises(ValueError, match="seed"):
        prom_binary_adapter(calibration, labels, target, alpha=0.1, seed=-1)
    with pytest.raises(TypeError):
        prom_binary_adapter(
            calibration,
            labels,
            target,
            alpha=0.1,
            seed=13,
            target_labels=np.array([0]),
        )


@pytest.mark.parametrize(
    "calibration,labels,target,error",
    [
        (np.array([[0.2, 0.3, 0.5]]), [0], [[0.5, 0.5]], "shape"),
        (np.array([[0.2, 0.7]]), [0], [[0.5, 0.5]], "sum"),
        (np.array([[0.5, 0.5], [0.4, 0.6]]), [0, 2], [[0.5, 0.5]], "0 and 1"),
    ],
)
def test_adapter_rejects_invalid_binary_inputs(calibration, labels, target, error):
    with pytest.raises(ValueError, match=error):
        prom_binary_adapter(
            calibration,
            np.asarray(labels),
            np.asarray(target),
            alpha=0.1,
            seed=13,
        )
