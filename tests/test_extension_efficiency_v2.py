import pandas as pd
import pytest

from analyze_extension_v2_efficiency import head_fit_summary, select_executed_fits


def observations(codebert_second_duration=9.0):
    rows = []
    for seed, seconds in ((13, 1.0), (37, 3.0)):
        rows.append(
            {
                "detector": "hashing",
                "target_group": "p",
                "seed": seed,
                "head_fit_seconds": seconds,
                "selected_parameter": 1e-5,
                "source_validation_pr_auc": 0.2,
                "technical_seed_reused": False,
                "seed_reused_from": None,
            }
        )
    for seed in (13, 37, 73, 101, 137):
        rows.append(
            {
                "detector": "codebert",
                "target_group": "p",
                "seed": seed,
                "head_fit_seconds": 9.0 if seed != 37 else codebert_second_duration,
                "selected_parameter": 0.01,
                "source_validation_pr_auc": 0.3,
                "technical_seed_reused": True,
                "seed_reused_from": 13,
            }
        )
    return pd.DataFrame(rows)


def test_seed_reuse_is_counted_as_one_executed_codebert_fit():
    executed = select_executed_fits(observations(), reference_seed=13)

    assert (executed["detector"] == "hashing").sum() == 2
    assert (executed["detector"] == "codebert").sum() == 1
    assert head_fit_summary(executed, "hashing")["head_fit_seconds_median"] == 2.0


def test_inconsistent_reused_codebert_metadata_is_rejected():
    with pytest.raises(RuntimeError, match="reused CodeBERT metadata differs"):
        select_executed_fits(observations(codebert_second_duration=10.0), reference_seed=13)
