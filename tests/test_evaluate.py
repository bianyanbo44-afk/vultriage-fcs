from pathlib import Path

import numpy as np

from evaluate_e1 import (
    evaluate_one,
    matched_score_sets,
    prom_compatible_lac_credibility,
    temperature_scale,
)


def test_prom_credibility_prefers_conforming_forced_predictions():
    calibration_probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    labels = np.array([0, 0, 1, 1])
    test = np.array([[0.95, 0.05], [0.55, 0.45]])
    credibility = prom_compatible_lac_credibility(
        calibration_probabilities, labels, test
    )
    assert credibility[0] > credibility[1]
    sets = matched_score_sets(test, credibility, requested_singletons=1)
    assert sets[0].sum() == 1
    assert sets[1].sum() == 2


def test_temperature_scaling_returns_valid_probabilities():
    calibration_probabilities = np.array(
        [[0.95, 0.05], [0.8, 0.2], [0.2, 0.8], [0.05, 0.95]]
    )
    labels = np.array([0, 0, 1, 1])
    test = np.array([[0.7, 0.3], [0.3, 0.7]])
    scaled, temperature = temperature_scale(
        calibration_probabilities, labels, test
    )
    assert temperature > 0
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert ((scaled >= 0) & (scaled <= 1)).all()


def test_evaluate_one_runs_project_track_and_writes_decisions(tmp_path: Path):
    calibration_positions = np.arange(0, 60)
    target_positions = np.arange(60, 80)
    labels = np.array(([0, 1] * 40), dtype=int)
    calibration_p = np.linspace(0.05, 0.95, 60)
    target_p = np.linspace(0.1, 0.9, 20)
    archive = {
        "calibration_positions": calibration_positions,
        "calibration_p_vulnerable": calibration_p,
        "calibration_raw_ratio": np.ones(60),
        "target_positions": target_positions,
        "target_p_vulnerable": target_p,
        "target_raw_ratio": np.ones(20),
    }
    config = {
        "risk_budgets": {"vulnerable": [0.1], "safe": [0.1]},
        "density_ratio": {
            "selected_upper_clip": 20.0,
            "candidate_upper_clips": [20.0],
        },
        "support_rules": {
            "minimum_total_ess": 10,
            "minimum_class_ess": 5,
            "class_ess_multiplier_over_alpha": 0.1,
            "minimum_positive_neighbours": 2,
        },
    }
    decision_path = tmp_path / "decisions.npz"
    records = evaluate_one(
        "project_disjoint",
        "synthetic",
        13,
        archive,
        labels,
        config,
        decision_path,
    )
    methods = {record["method"] for record in records}
    assert "vultriage_clip_20" in methods
    assert "prom_compatible_lac_matched_vultriage" in methods
    assert decision_path.exists()
