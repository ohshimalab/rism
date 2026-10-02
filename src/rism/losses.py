"""NeuralSort with Sinkhorn scaling (Sec. 4.2)."""

import torch
from torch import Tensor
from torch.nn import functional as F


def sinkhorn_scaling(mat: Tensor, mask: Tensor | None = None,
                     tol: float = 1e-6, max_iter: int = 10) -> Tensor:
    """Approximately doubly normalize the unpadded block."""
    for _ in range(max_iter):
        if mask is not None:
            mat = mat.masked_fill(mask, 0.0)
        mat = mat / (mat.sum(dim=2, keepdim=True) + 1e-8)
        if mask is not None:
            mat = mat.masked_fill(mask, 0.0)
        mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-8)
        sums = mat.sum(dim=2)
        active = ~mask.all(dim=2) if mask is not None else torch.ones_like(sums, dtype=torch.bool)
        if (torch.abs(sums[active] - 1.0) < tol).all():
            break
    return mat


def deterministic_neural_sort(s: Tensor, tau: float, mask: Tensor) -> Tensor:
    """Return soft rank-to-item permutations; support padding anywhere in a slate."""
    if tau <= 0:
        raise ValueError("temperature must be positive")
    n = s.size(1)
    # Rank rows are compacted, unlike item columns. This also handles interior padding.
    valid_count = (~mask).sum(-1)
    rank_mask = torch.arange(n, device=s.device)[None, :] >= valid_count[:, None]
    safe_s = s.masked_fill(mask[:, :, None], 0.0)
    differences = torch.abs(safe_s - safe_s.transpose(1, 2))
    differences = differences.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)
    ranks = torch.arange(n, device=s.device, dtype=s.dtype) + 1
    scaling = valid_count.to(s.dtype)[:, None] + 1 - 2 * ranks[None, :]
    logits = scaling[:, :, None] * safe_s.transpose(1, 2) - differences.sum(-1)[:, None, :]
    blocked = rank_mask[:, :, None] | mask[:, None, :]
    logits = logits.masked_fill(blocked, torch.finfo(s.dtype).min)
    logits = logits.masked_fill(rank_mask[:, :, None], 0.0)
    return F.softmax(logits / tau, dim=-1).masked_fill(blocked, 0.0)


def neural_ndcg(y_pred: Tensor, y_true: Tensor, padded_value_indicator: int = -1,
                temperature: float = 1.0, powered_relevancies: bool = True,
                k: int | None = None) -> Tensor:
    """Negative batch mean approximate nDCG; y_true contains grades, never AP."""
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise ValueError("Expected matching (batch, candidates) matrices")
    k = y_true.shape[1] if k is None else k
    if k < 1:
        raise ValueError("k must be positive")
    mask = y_true == padded_value_indicator
    n = y_true.shape[1]
    rank_mask = torch.arange(n, device=y_pred.device)[None, :] >= (~mask).sum(-1)[:, None]
    pad_mat = rank_mask[:, :, None] | mask[:, None, :]
    permutation = deterministic_neural_sort(y_pred.unsqueeze(-1), temperature, mask)
    permutation = sinkhorn_scaling(permutation, pad_mat).masked_fill(pad_mat, 0.0)
    rels = y_true.to(y_pred.dtype).masked_fill(mask, 0.0)
    gains = (torch.pow(2.0, rels) - 1) if powered_relevancies else rels
    sorted_gains = (permutation @ gains.unsqueeze(-1)).squeeze(-1)
    discount = 1 / torch.log2(torch.arange(n, device=y_pred.device, dtype=y_pred.dtype) + 2)
    dcg = (sorted_gains[:, :k] * discount[:k]).sum(-1)
    ideal = (gains.sort(descending=True, dim=-1).values[:, :k] * discount[:k]).sum(-1)
    return -(dcg / ideal.clamp_min(1e-8)).mean()
