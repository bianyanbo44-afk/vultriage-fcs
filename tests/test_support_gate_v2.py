import numpy as np

from fit_support_gate_v2 import FEATURES, gate_model


def test_gate_model_contract_is_frozen_and_probabilistic():
    x = np.vstack([np.zeros(len(FEATURES)), np.ones(len(FEATURES))] * 4)
    y = np.asarray([0, 1] * 4)
    model = gate_model().fit(x, y)
    head = model.named_steps["model"]
    assert head.C == 1.0
    assert head.solver == "liblinear"
    assert head.class_weight == "balanced"
    assert head.random_state == 20260814
    probability = model.predict_proba(x)
    assert probability.shape == (8, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
