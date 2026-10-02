import math

import numpy as np
import pytest
import torch

from rism.data import relevance_labels
from rism.losses import neural_ndcg
from rism.metrics import ndcg


def test_ndcg_hand_calculation_and_ties():
    relevance = torch.tensor([[0, 2, 1], [0, 0, 0]])
    scores = torch.ones(2, 3)
    expected = (3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3))
    assert ndcg(scores, relevance, 3).tolist() == pytest.approx([expected, 0.])
    assert ndcg(scores, relevance, 1).tolist() == [0., 0.]
    assert ndcg(torch.tensor([3., 2., 1.]), torch.tensor([2, 1, 0]), 5).item() == 1.


def test_relevance_thresholds_nan_and_zero():
    values = np.array([[1., .9, .8, .899, .799, np.nan], [0., 0., 0., 0., 0., np.nan]])
    np.testing.assert_array_equal(relevance_labels(values), [[2, 2, 1, 1, 0, 0], [0] * 6])
    # Normalize relative to the image maximum, not an absolute AP threshold.
    np.testing.assert_array_equal(relevance_labels(np.array([.5, .45, .4, np.nan])), [2, 2, 1, 0])
    with pytest.raises(ValueError):
        relevance_labels(np.array([np.inf, 0.]))


@pytest.mark.parametrize("k", [1, 3, 5])
def test_neural_ndcg_approximates_hard_ranking(k):
    scores = torch.tensor([[40., 10., 30., 20., 0.]], requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 0, 2]])
    loss = neural_ndcg(scores, labels, k=k)
    assert -loss.detach().item() == pytest.approx(ndcg(scores, labels, k).item(), abs=1e-3)


def test_neural_ndcg_gradients_and_truncation():
    scores = torch.tensor([[.1, .8, .3]], requires_grad=True)
    labels = torch.tensor([[2, 0, 1]])
    loss = neural_ndcg(scores, labels, k=1)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0
    assert loss.detach().item() != pytest.approx(neural_ndcg(scores, labels, k=3).detach().item())
    assert neural_ndcg(scores, torch.zeros_like(labels)).detach().item() == 0


def test_padding_including_interior_and_fully_padded_slates():
    scores = torch.tensor([[.2, .8, .5]], requires_grad=True)
    labels = torch.tensor([[1, 2, 0]])
    loss = neural_ndcg(scores, labels, k=2)
    padded_scores = torch.tensor([[.2, 999., .8, .5, -999.]], requires_grad=True)
    padded_labels = torch.tensor([[1, -1, 2, 0, -1]])
    padded_loss = neural_ndcg(padded_scores, padded_labels, k=2)
    assert loss.detach().item() == pytest.approx(padded_loss.detach().item(), abs=1e-6)
    padded_loss.backward()
    assert padded_scores.grad[0, 1].item() == 0
    assert padded_scores.grad[0, 4].item() == 0
    assert neural_ndcg(torch.zeros(1, 3), torch.full((1, 3), -1)).item() == 0
