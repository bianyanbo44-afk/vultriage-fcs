import numpy as np
import pandas as pd
from pathlib import Path

from analyze_extension_v2 import effective_config, gate_pass_fail_bootstrap_interval


def test_effective_config_inherits_seeds_and_bootstrap_settings():
    config = effective_config(Path("configs/preregistered_extension_v2.json"))
    assert config["seeds"] == [13, 37, 73, 101, 137]
    assert config["bootstrap"]["replicates"] == 10000
    assert config["risk_budgets"]["vulnerable"] == [0.01, 0.05, 0.1]


def test_gate_pass_fail_bootstrap_interval_is_deterministic_and_directional():
    joined = pd.DataFrame(
        {
            "pass_all": [True, True, True, False, False, False],
            "max_relative_violation": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        }
    )
    first = gate_pass_fail_bootstrap_interval(joined, 2000, 17, 0.95)
    second = gate_pass_fail_bootstrap_interval(joined, 2000, 17, 0.95)
    assert first == second
    assert first[1] < 0.0


def test_gate_pass_fail_bootstrap_interval_requires_both_strata():
    joined = pd.DataFrame(
        {"pass_all": [True, True], "max_relative_violation": [0.1, 0.2]}
    )
    lower, upper = gate_pass_fail_bootstrap_interval(joined, 100, 3, 0.95)
    assert np.isnan(lower)
    assert np.isnan(upper)
