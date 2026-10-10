from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruct_independent_model_v4_1 import independent_metrics, independent_discrimination, independent_seed_averages


def test_empty_review_and_automatic_errors_use_different_denominators():
    labels = np.array([1, 1, 0, 0])
    sets = np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=bool)
    value = independent_metrics(labels, sets, .1, .2, dict(miss=[10, 50], false_alert=1, review=.1), "ok")
    assert value["coverage"] == .5 and value["auto_fn_rate"] == .5 and value["auto_fp_rate"] == .5
    assert value["vulnerable_miscoverage"] == 1 and value["safe_miscoverage"] == .5
    assert value["empty_count"] == 1 and value["review_count"] == 2
    assert value["maximum_relative_violation"] == 9 and value["cost_10"] == 2.8 and value["cost_50"] == 12.8


def test_failed_fitted_set_cannot_earn_zero_budget_violation():
    value = independent_metrics(np.array([0, 1]), np.ones((2, 2), dtype=bool), .1, .2,
                                dict(miss=[10, 50], false_alert=1, review=.1), "domain_unavailable_review_fallback")
    assert value["cost_10"] == .1 and value["auto_fn"] == 0
    assert np.isnan(value["maximum_relative_violation"]) and np.isnan(value["vulnerable_miscoverage"])


def test_tied_probabilities_have_threshold_group_ap_and_average_rank_auc():
    value = independent_discrimination(np.array([1, 0, 1, 0]), np.array([.9, .9, .2, .1]), np.array([.8, .8, .1, 0]))
    assert np.isclose(value["pr_auc"], 7 / 12) and value["roc_auc"] == .625


def test_undefined_seed_is_not_silently_dropped():
    rows = [dict(dataset="x", detector="codet5", project="p", alpha_v=.1, alpha_s=.2, clip=20,
                 method="m", seed=s, status="ok", selective_error=v) for s, v in [(13, .1), (37, np.nan), (73, .2)]]
    result = independent_seed_averages(pd.DataFrame(rows), [13, 37, 73])
    assert result.iloc[0]["actual_seed_count"] == 3 and np.isnan(result.iloc[0]["selective_error"])
