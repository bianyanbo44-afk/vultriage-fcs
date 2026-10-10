import numpy as np
import pytest

from analyze_neurocomputing_revision import action_counts, auc, matched_confidence


def test_empty_set_miscoverage_is_not_an_automated_false_negative():
    labels = np.array([1, 0, 1, 0])
    sets = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    counts = action_counts(labels, sets)
    assert counts['auto_fn'] == 1
    assert counts['auto_fp'] == 1
    assert counts['review_count'] == 2
    assert counts['review_vulnerable'] == 1
    assert counts['review_safe'] == 1
    assert counts['empty_count'] == 1
    assert counts['selective_error'] == 1
    assert counts['auto_fn_rate'] == .5


def test_review_only_does_not_receive_zero_selective_error():
    counts = action_counts(np.array([0, 1]), np.ones((2, 2)))
    assert counts['review_count'] == 2
    assert counts['auto_fn'] == counts['auto_fp'] == 0
    assert np.isnan(counts['selective_error'])
    assert np.isnan(counts['safe_release_contamination'])


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_confidence_control_matches_volume_without_labels(count):
    p = np.array([.1, .9, .3, .7])
    sets = matched_confidence(p, count)
    assert (sets.sum(1) == 1).sum() == count
    if count == 1:
        assert sets[0].tolist() == [True, False]
        assert sets[1].tolist() == [True, True]


def test_auroc_accounts_for_tied_scores():
    assert auc(np.array([0, 1, 0, 1]), np.ones(4)) == .5
    assert auc(np.array([0, 1]), np.array([.1, .9])) == 1
    assert np.isnan(auc(np.array([1, 1]), np.array([.1, .9])))


def test_exact_probability_tie_matches_original_argmax_convention():
    sets = matched_confidence(np.array([.5, .5]), 1)
    assert sets[0].tolist() == [True, False]
    assert sets[1].tolist() == [True, True]


def test_invalid_decision_shape_is_rejected():
    with pytest.raises(ValueError):
        action_counts(np.array([0, 1]), np.array([0, 1]))


@pytest.mark.parametrize('count', [-1, 5, 1.2, True])
def test_invalid_confidence_volume_is_rejected(count):
    with pytest.raises(ValueError):
        matched_confidence(np.array([.1, .9, .3, .7]), count)


@pytest.mark.parametrize('probability', [[-.1, .3], [.3, 1.1], [.3, np.nan], [[.1, .9]]])
def test_invalid_confidence_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        matched_confidence(np.array(probability), 1)


@pytest.mark.parametrize('labels,sets', [
    ([0, 1.7], [[1, 0], [0, 1]]),
    ([0, 1], [[1, 0], [0, np.nan]]),
    ([[0, 1]], [[1, 0], [0, 1]]),
    ([], np.empty((0, 2))),
])
def test_invalid_archive_values_are_not_silently_coerced(labels, sets):
    with pytest.raises(ValueError):
        action_counts(np.array(labels), np.array(sets))
