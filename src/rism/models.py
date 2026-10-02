"""Four jointly trained retrievers sharing one LightningModule."""

from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from transformers import ViTModel

from .config import VARIANTS, validate_hparams
from .losses import neural_ndcg
from .metrics import ndcg


class Retriever(pl.LightningModule):
    """Shared implementation; variant selects feature extraction and fusion."""

    def __init__(self, vit: ViTModel, num_models: int, variant: str = "ours", *,
                 alpha: float = 0.5, k_loss: int = 51, lr: float = 1e-3,
                 nhead: int = 8, hidden_dim: int = 512,
                 dim_feedforward: int = 2048) -> None:
        super().__init__()
        validate_hparams(alpha, k_loss)
        if variant not in VARIANTS or num_models < 1:
            raise ValueError("Unknown variant or empty candidate pool")
        self.save_hyperparameters(ignore=["vit"])
        self.vit = vit
        self.variant = variant
        self.num_models = num_models
        self.alpha, self.k_loss, self.lr = alpha, k_loss, lr
        self.model_dim = vit.config.hidden_size
        self.model_embeddings = nn.Parameter(torch.rand(num_models, self.model_dim))
        size, patch = vit.config.image_size, vit.config.patch_size
        size = (size, size) if isinstance(size, int) else size
        patch = (patch, patch) if isinstance(patch, int) else patch
        self.num_patches = (size[0] // patch[0]) * (size[1] // patch[1])
        self.token_count = self.num_patches if variant in ("ours", "local_mlp") else 1
        # Sec. 4.1 / 6.1: positional embeddings on all tokens only for TE fusion.
        if variant in ("ours", "global_te"):
            if self.model_dim % nhead:
                raise ValueError("nhead must divide the ViT hidden size")
            self.pos_embedding = nn.Parameter(torch.randn(self.token_count + 1, self.model_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(self.model_dim, nhead,
                                               dim_feedforward, dropout=0.1,
                                               batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            input_dim = self.model_dim
        else:
            input_dim = (self.token_count + 1) * self.model_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.mse = nn.MSELoss(reduction="mean")

    def encode_image(self, images: Tensor) -> Tensor:
        """Return (B,P,D) patches or (B,1,D) CLS features."""
        tokens = self.vit(pixel_values=images).last_hidden_state
        return tokens[:, 1:] if self.variant in ("ours", "local_mlp") else tokens[:, :1]

    def _sequence(self, features: Tensor, embeddings: Tensor) -> Tensor:
        batch, tokens, dim = features.shape
        candidates = embeddings.shape[0]
        model_tokens = embeddings[None, :, None, :].expand(batch, -1, -1, -1)
        image_tokens = features[:, None, :, :].expand(-1, candidates, -1, -1)
        return torch.cat((model_tokens, image_tokens), dim=2).reshape(batch * candidates, tokens + 1, dim)

    def score_features(self, features: Tensor, model_embeddings: Tensor | None = None,
                       chunk_size: int | None = None) -> Tensor:
        """Score a replaceable candidate pool; chunking is identical in eval mode."""
        embeddings = self.model_embeddings if model_embeddings is None else model_embeddings
        if features.ndim != 3 or features.shape[1:] != (self.token_count, self.model_dim):
            raise ValueError("Image features have an incompatible shape")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.model_dim or not len(embeddings):
            raise ValueError("Expected a nonempty (candidates, hidden_size) embedding matrix")
        chunk_size = len(embeddings) if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        outputs = []
        for chunk in embeddings.split(chunk_size):
            sequence = self._sequence(features, chunk)
            if self.variant in ("ours", "global_te"):
                fused = self.encoder(sequence + self.pos_embedding)[:, 0]
            elif self.variant == "prior":
                # Sec. 5.2: concatenate global CLS features followed by model embedding.
                fused = sequence.flip(1).flatten(1)
            else:
                fused = sequence.flatten(1)
            outputs.append(self.mlp(fused).reshape(features.shape[0], len(chunk)))
        return torch.cat(outputs, dim=1)

    def forward(self, images: Tensor) -> Tensor:
        return self.score_features(self.encode_image(images))

    def model_token_attention(self, images: Tensor) -> Tensor:
        """Last TE layer attention, averaged over heads: (B,K,P) or (B,K,1).

        Call eval() for deterministic maps. MLP variants have no token attention.
        """
        if self.variant not in ("ours", "global_te"):
            raise NotImplementedError("Token attention requires a Transformer encoder")
        features = self.encode_image(images)
        sequence = self._sequence(features, self.model_embeddings) + self.pos_embedding
        for layer in self.encoder.layers[:-1]:
            sequence = layer(sequence)
        # Sec. 6.2; post-norm self-attention sees the previous layer output.
        _, weights = self.encoder.layers[-1].self_attn(
            sequence, sequence, sequence, need_weights=True, average_attn_weights=True)
        return weights[:, 0, 1:].reshape(images.shape[0], self.num_models, self.token_count)

    def calculate_loss(self, scores: Tensor, ap: Tensor, relevance: Tensor) -> dict[str, Tensor]:
        """Sec. 4.2: regress AP and rank the three-level relevance labels."""
        mse = self.mse(scores, ap)
        ranking = neural_ndcg(scores, relevance, k=self.k_loss, temperature=1.0)
        return {"loss": self.alpha * mse + (1 - self.alpha) * ranking,
                "mse": mse, "neuralndcg": ranking}

    def _shared_step(self, batch: dict[str, Any], prefix: str) -> Tensor:
        scores = self(batch["image"])
        terms = self.calculate_loss(scores, batch["ap"], batch["relevance"])
        for name, value in terms.items():
            self.log(f"{prefix}_{name}", value, on_step=False, on_epoch=True,
                     batch_size=len(scores), sync_dist=True)
        for k in (3, 5):
            self.log(f"{prefix}_ndcg_{k}", ndcg(scores.detach(), batch["relevance"], k).mean(),
                     on_step=False, on_epoch=True, batch_size=len(scores), sync_dist=True)
        return terms["loss"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return {"scores": self(batch["image"]).cpu(), "relevance": batch["relevance"].cpu(),
                "category": batch["category"]}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Sec. 5.1 learning rate; weight decay is not specified in the paper.
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
