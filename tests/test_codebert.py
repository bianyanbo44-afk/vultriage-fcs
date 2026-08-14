import os

import numpy as np
import pytest

from vultriage.codebert import fit_logistic_head, masked_mean_pool


@pytest.mark.skipif(
    os.environ.get("VULTRIAGE_TEST_TORCH") != "1",
    reason="run in the verified CUDA environment with VULTRIAGE_TEST_TORCH=1",
)
def test_masked_mean_pool_excludes_padding_and_special_tokens():
    import torch
    hidden = torch.tensor(
        [[[100.0, 100.0], [2.0, 4.0], [4.0, 8.0], [200.0, 200.0]]]
    )
    attention = torch.tensor([[1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1]])
    pooled = masked_mean_pool(hidden, attention, special)
    assert torch.allclose(pooled, torch.tensor([[3.0, 6.0]]))


def test_logistic_head_selects_from_grid_and_emits_probabilities():
    train_x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    train_y = np.asarray([0, 0, 1, 1])
    valid_x = np.asarray([[-1.5], [-0.5], [0.5], [1.5]], dtype=np.float32)
    valid_y = np.asarray([0, 0, 1, 1])
    selected = fit_logistic_head(
        train_x, train_y, valid_x, valid_y, [1.0, 0.01, 0.1], seed=13
    )
    assert selected.c_value == 0.01
    probability = selected.model.predict_proba(valid_x)
    assert probability.shape == (4, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(selected.model.named_steps["scale"].mean_, train_x.mean(axis=0))
