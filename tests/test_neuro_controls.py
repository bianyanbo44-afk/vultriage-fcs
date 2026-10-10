from collections import Counter
from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from analyze_neurocomputing_controls import (
    REFERENCES,
    action_codes,
    build_controls,
    class_side_matched,
    empirical_median_distribution,
    empirical_median_interval,
    fit_source_isotonic,
    global_msp_matched,
    paired_summary,
    sha256,
    source_cost_matched,
    verified_path,
)
from analyze_neurocomputing_revision import action_counts


@pytest.mark.parametrize("probability", [
    np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]),
    np.full(6, 0.8),
    np.array([0.9, 0.2, 0.9, 0.1, 0.6, 0.2]),
])
def test_all_feasible_class_allocations_are_exact_and_disjoint(probability):
    for safe in range(len(probability) + 1):
        for vulnerable in range(len(probability) - safe + 1):
            sets = class_side_matched(probability, safe, vulnerable)
            codes = action_codes(sets)
            assert np.sum(codes == 0) == safe
            assert np.sum(codes == 1) == vulnerable
            assert np.sum(codes == 2) == len(probability) - safe - vulnerable
            assert np.isin(sets.sum(axis=1), [1, 2]).all()
            np.testing.assert_array_equal(sets, class_side_matched(probability, safe, vulnerable))
            if safe and vulnerable:
                assert probability[codes == 0].max() <= probability[codes == 1].min()


def test_class_side_ties_share_one_order_and_allow_non_argmax_labels():
    sets = class_side_matched(np.full(5, 0.9), 2, 2)
    assert action_codes(sets).tolist() == [0, 0, 2, 1, 1]
    assert len(set(np.flatnonzero(action_codes(sets) == 0)) &
               set(np.flatnonzero(action_codes(sets) == 1))) == 0


def test_empty_selector_cohort_and_zero_automation_are_well_defined():
    assert class_side_matched(np.array([]), 0, 0).shape == (0, 2)
    assert global_msp_matched(np.array([]), 0).shape == (0, 2)
    assert source_cost_matched(np.array([]), 0, 10).shape == (0, 2)
    for sets in (class_side_matched(np.array([0.1, 0.9]), 0, 0),
                 global_msp_matched(np.array([0.1, 0.9]), 0),
                 source_cost_matched(np.array([0.1, 0.9]), 0, 10)):
        assert sets.all()


@pytest.mark.parametrize("count", [-1, 4, True, np.bool_(False), 1.0, "1"])
def test_invalid_automation_counts_are_rejected(count):
    for selector in (lambda: class_side_matched(np.array([0.1, 0.5, 0.9]), count, 0),
                     lambda: class_side_matched(np.array([0.1, 0.5, 0.9]), 0, count),
                     lambda: global_msp_matched(np.array([0.1, 0.5, 0.9]), count),
                     lambda: source_cost_matched(np.array([0.1, 0.5, 0.9]), count, 10)):
        with pytest.raises(ValueError):
            selector()


def test_infeasible_combined_class_allocation_is_rejected():
    with pytest.raises(ValueError, match="exceed"):
        class_side_matched(np.array([0.1, 0.5, 0.9]), 2, 2)


@pytest.mark.parametrize("probability", [[-0.01, 0.2], [0.2, 1.01], [0.2, np.nan],
                                         [np.inf, 0.2], [[0.1, 0.9]]])
def test_invalid_probabilities_are_rejected_by_each_selector(probability):
    for selector in (lambda: class_side_matched(np.asarray(probability), 1, 0),
                     lambda: global_msp_matched(np.asarray(probability), 1),
                     lambda: source_cost_matched(np.asarray(probability), 1, 10),
                     lambda: fit_source_isotonic(np.asarray(probability), np.array([0, 1]))):
        with pytest.raises(ValueError):
            selector()


def test_global_msp_reproduces_original_argmax_and_archive_order_ties():
    sets = global_msp_matched(np.array([0.5, 0.5, 0.1, 0.9]), 3)
    assert action_codes(sets).tolist() == [0, 2, 0, 1]


def test_isotonic_is_fit_on_source_labels_and_clips_outside_source_score_range():
    model = fit_source_isotonic(np.array([0.2, 0.3, 0.7, 0.8]), np.array([0, 0, 1, 1]))
    eta = model.predict(np.array([0.0, 0.25, 0.5, 0.9, 1.0]))
    np.testing.assert_allclose(eta, [0.0, 0.0, 0.5, 1.0, 1.0])
    assert (np.diff(eta) >= 0).all()


def test_isotonic_averages_tied_source_scores_without_using_target_outcomes():
    model = fit_source_isotonic(np.array([0.4, 0.4]), np.array([0, 1]))
    np.testing.assert_array_equal(model.predict([0.0, 0.4, 1.0]), [0.5, 0.5, 0.5])


@pytest.mark.parametrize("labels", [[0, 1.7], [0, 2], [0, np.nan], [[0, 1]], [0]])
def test_source_calibrator_rejects_malformed_source_labels(labels):
    with pytest.raises(ValueError):
        fit_source_isotonic(np.array([0.1, 0.9]), np.asarray(labels))


def test_empty_source_calibration_cannot_fit_a_model():
    with pytest.raises(ValueError):
        fit_source_isotonic(np.array([]), np.array([]))


@pytest.mark.parametrize("miss_cost", [1, 10, 50])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_source_cost_selector_solves_fixed_volume_expected_cost_problem(miss_cost, count):
    eta = np.array([0.0, 0.125, 0.75, 1.0])
    sets = source_cost_matched(eta, count, miss_cost)
    codes = action_codes(sets)
    assert np.sum(codes != 2) == count
    obtained = np.sum(miss_cost * eta[codes == 0]) + np.sum(1 - eta[codes == 1])
    # Independent exhaustive choice of a retained subset AND its assigned labels.
    feasible_costs = []
    for selected in combinations(range(len(eta)), count):
        for assigned_labels in product([0, 1], repeat=count):
            feasible_costs.append(sum(miss_cost * eta[row] if label == 0 else 1 - eta[row]
                                      for row, label in zip(selected, assigned_labels)))
    assert obtained == pytest.approx(min(feasible_costs))


def test_source_cost_ties_choose_vulnerable_and_keep_stable_first_rows():
    sets = source_cost_matched(np.full(4, 0.5), 2, 1)
    assert action_codes(sets).tolist() == [1, 1, 2, 2]


@pytest.mark.parametrize("miss_cost,false_alert_cost", [(0, 1), (-1, 1), (True, 1),
                                                         (1, False), (np.nan, 1),
                                                         (1, np.inf), (1, -1)])
def test_invalid_source_ranking_costs_are_rejected(miss_cost, false_alert_cost):
    with pytest.raises(ValueError):
        source_cost_matched(np.array([0.1, 0.9]), 1, miss_cost, false_alert_cost)


def test_build_controls_matches_original_workload_and_each_class_allocation():
    p = np.array([0.1, 0.9, 0.4, 0.8])
    references = {
        "unweighted_mondrian": np.array([[1, 0], [0, 1], [1, 1], [1, 1]]),
        "estimated_weight_no_gate": np.array([[1, 0], [1, 0], [0, 1], [0, 0]]),
        "vultriage_full_gate": np.ones((4, 2)),
    }
    before = {name: sets.copy() for name, sets in references.items()}
    decisions = build_controls(p, references, np.array([0.01, 0.3, 0.02, 0.1]))
    for name, suffix in REFERENCES.items():
        reference = action_codes(references[name])
        tail = action_codes(decisions[f"class_matched_{suffix}"])
        assert np.sum(tail == 0) == np.sum(reference == 0)
        assert np.sum(tail == 1) == np.sum(reference == 1)
        np.testing.assert_array_equal(references[name], before[name])
        for control_name, sets in decisions.items():
            if control_name.endswith(f"matched_{suffix}"):
                assert np.sum(action_codes(sets) != 2) == np.sum(reference != 2)
    assert "source_iso_cost10_matched_weighted" in decisions
    assert "source_iso_cost50_matched_gate" in decisions
    # Evaluating a different target label vector cannot mutate a fixed selector.
    fixed = decisions["class_matched_weighted"].copy()
    action_counts(np.array([0, 1, 0, 1]), fixed)
    action_counts(np.array([1, 0, 1, 0]), fixed)
    np.testing.assert_array_equal(fixed, decisions["class_matched_weighted"])


def test_build_controls_rejects_missing_policy_or_misaligned_source_scores():
    p = np.array([0.1, 0.9])
    refs = {name: np.ones((2, 2)) for name in REFERENCES}
    with pytest.raises(ValueError):
        build_controls(p, {"unweighted_mondrian": refs["unweighted_mondrian"]})
    with pytest.raises(ValueError):
        build_controls(p, refs, np.array([0.1]))


@pytest.mark.parametrize("values", [np.array([0.0, 1.0]), np.array([0.0, 1.0, 3.0, 3.0])])
def test_exact_bootstrap_distribution_matches_exhaustive_resampling(values):
    medians = Counter(float(np.median(values[list(indices)]))
                      for indices in product(range(len(values)), repeat=len(values)))
    support, masses = empirical_median_distribution(values)
    denominator = len(values) ** len(values)
    np.testing.assert_array_equal(support, sorted(medians))
    np.testing.assert_allclose(masses, [medians[value] / denominator for value in support], atol=1e-14, rtol=0)


def test_exact_interval_preserves_all_tied_projects_and_has_no_random_draw_error():
    assert empirical_median_interval(np.zeros(24)) == (0.0, 0.0)
    assert empirical_median_interval(np.arange(24)) == empirical_median_interval(np.arange(24)[::-1])


@pytest.mark.parametrize("values", [np.array([]), np.array([0.0]), np.array([0.0, 1.0, 2.0]),
                                    np.array([0.0, np.nan]), np.array([[0.0, 1.0]])])
def test_exact_bootstrap_rejects_invalid_or_odd_project_vectors(values):
    with pytest.raises(ValueError):
        empirical_median_interval(values)


@pytest.mark.parametrize("lower,upper", [(-0.1, 0.9), (0.1, 1.1), (0.9, 0.1), (np.nan, 1)])
def test_invalid_exact_quantiles_are_rejected(lower, upper):
    with pytest.raises(ValueError):
        empirical_median_interval(np.array([0.0, 1.0]), lower, upper)


def paired_frame():
    return pd.DataFrame([
        dict(method=method, target_group=group, auto_fn_rate=value, coverage=value)
        for method, values in (("baseline", [0.2, 0.2, 0.2, 0.2]),
                               ("candidate", [0.1, 0.2, 0.3, 0.3]))
        for group, value in zip(["d", "a", "c", "b"], values)
    ])


def test_paired_summary_uses_project_differences_and_endpoint_direction():
    frame = paired_frame()
    error = paired_summary(frame, "auto_fn_rate", "baseline", "candidate")
    coverage = paired_summary(frame.sample(frac=1, random_state=17), "coverage", "baseline", "candidate")
    assert error["projects"] == 4
    assert error["median_difference"] == pytest.approx(0.05)
    assert (error["favorable_projects"], error["unfavorable_projects"], error["ties"]) == (1, 2, 1)
    assert (coverage["favorable_projects"], coverage["unfavorable_projects"], coverage["ties"]) == (2, 1, 1)
    assert coverage["median_difference"] == error["median_difference"]


def test_paired_summary_rejects_seed_rows_instead_of_treating_them_as_independent_projects():
    frame = paired_frame()
    replicated = pd.concat([frame] * 5, ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate paired project"):
        paired_summary(replicated, "auto_fn_rate", "baseline", "candidate")
    averaged = replicated.groupby(["target_group", "method"], as_index=False).auto_fn_rate.mean()
    assert paired_summary(averaged, "auto_fn_rate", "baseline", "candidate")["projects"] == 4


def test_paired_summary_rejects_unmatched_projects_and_undefined_endpoints():
    frame = paired_frame()
    with pytest.raises(ValueError, match="Unmatched"):
        paired_summary(frame.iloc[1:], "auto_fn_rate", "baseline", "candidate")
    frame.loc[0, "auto_fn_rate"] = np.nan
    with pytest.raises(ValueError, match="Undefined"):
        paired_summary(frame, "auto_fn_rate", "baseline", "candidate")


def test_seal_verification_detects_tampering_and_prevents_path_escape(tmp_path):
    root = tmp_path / "sealed"
    root.mkdir()
    path = root / "artifact.npz"
    path.write_bytes(b"original frozen bytes")
    expected = sha256(path)
    assert verified_path(root, "artifact.npz", expected) == (path.resolve(), expected)
    path.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_path(root, "artifact.npz", expected)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside root")
    with pytest.raises(ValueError, match="escapes"):
        verified_path(root, "../outside.npz", sha256(outside))
