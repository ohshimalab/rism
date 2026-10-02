"""The shared hard-ranking metric for training, CV and test evaluation."""

import torch
from torch import Tensor


def ndcg(scores: Tensor, relevance: Tensor, k: int) -> Tensor:
    """Sec. 3.3 nDCG per image; ties favor the lower candidate index."""
    if scores.shape != relevance.shape or scores.ndim not in (1, 2) or k < 1:
        raise ValueError("Matching candidate vectors/matrices and positive k are required")
    scores = scores.unsqueeze(0) if scores.ndim == 1 else scores
    relevance = relevance.unsqueeze(0) if relevance.ndim == 1 else relevance
    rel = relevance.to(device=scores.device, dtype=torch.float32).clamp_min(0)
    width = min(k, scores.shape[-1])
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    gains = torch.pow(2.0, rel) - 1
    discount = 1 / torch.log2(torch.arange(width, device=scores.device).float() + 2)
    dcg = (gains.gather(-1, order)[:, :width] * discount).sum(-1)
    idcg = (gains.sort(dim=-1, descending=True).values[:, :width] * discount).sum(-1)
    return torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))
