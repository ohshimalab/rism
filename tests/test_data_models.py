import logging

import pytest
import torch

from rism.config import VARIANTS
from rism.data import ImageDataset, image_transform
from rism.models import Retriever


def test_dataset_grouping_and_labels(synthetic_data, caplog):
    with caplog.at_level(logging.INFO):
        dataset = ImageDataset(synthetic_data, image_size=32)
    assert len(dataset) == 12
    assert dataset.model_ids == list(range(5))
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["ap"].dtype == torch.float32
    assert sample["relevance"].tolist() == [2, 2, 1, 0, 0]
    assert "disagree with CSV Relevance Score" in caplog.text
    assert len(dataset.subset(originals_only=True)) == 8
    assert len(image_transform().transforms) == 3
    assert image_transform().transforms[0].size == (224, 224)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_shape_loss_chunking_and_attention(tiny_vit, variant, monkeypatch):
    model = Retriever(tiny_vit, 5, variant, nhead=2)
    images = torch.randn(2, 3, 32, 32)
    scores = model(images)
    assert scores.shape == (2, 5)
    assert ((scores >= 0) & (scores <= 1)).all()
    ap = torch.rand(2, 5)
    rel = torch.tensor([[2, 0, 1, 2, 0], [0, 1, 2, 0, 2]])
    terms = model.calculate_loss(scores, ap, rel)
    terms["loss"].backward()
    # Regression guard: perfect AP predictions and zero relevance have zero loss.
    # An implementation accidentally feeding AP into NeuralNDCG would be nonzero.
    zero = model.calculate_loss(ap, ap, torch.zeros_like(rel))
    assert zero["loss"].item() == 0.
    assert model.model_embeddings.grad.abs().sum() > 0
    assert model.vit.embeddings.patch_embeddings.projection.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        features = model.encode_image(images)
        full = model.score_features(features)
        assert torch.allclose(full, model.score_features(features, chunk_size=2), atol=1e-6)
        custom = torch.rand(7, model.model_dim)
        assert model.score_features(features, custom, chunk_size=3).shape == (2, 7)
        if variant in ("ours", "global_te"):
            attention = model.model_token_attention(images)
            assert attention.shape == (2, 5, 4 if variant == "ours" else 1)
            # Capture weights from the actual scoring forward pass. Grad mode
            # disables the encoder inference shortcut so the last MHA is observable.
            captured = []
            attention_layer = model.encoder.layers[-1].self_attn
            original_forward = attention_layer.forward
            def record(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                output, weights = original_forward(*args, **kwargs)
                captured.append(weights.detach())
                return output, weights
            monkeypatch.setattr(attention_layer, "forward", record)
            with torch.enable_grad():
                model(images)
            expected = captured[0][:, 0, 1:].reshape_as(attention)
            assert torch.allclose(attention, expected, atol=1e-6)
        else:
            with pytest.raises(NotImplementedError):
                model.model_token_attention(images)
    optimizer = model.configure_optimizers()
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 1e-4
