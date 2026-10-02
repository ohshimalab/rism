"""CPU/offline baseline equations and tiny-ViT workflow checks."""
import builtins
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner
from torch.utils.data import SequentialSampler

from rism.baselines import (SimSiam, _AllImageBatches, average_ranking,
                            load_backbone, nearest_image, simsiam_loss, weighted_average)
from rism.cli import app
from rism.data import ImageDataset
from rism.metrics import ndcg


def test_ni_ties_and_nan():
    refs = np.array([[1., 0.], [1., 0.], [0., 1.]])
    queries = np.array([[1., 0.], [0., 1.]])
    ap = np.array([[.3, np.nan], [.9, .7], [.6, .8]])
    np.testing.assert_array_equal(nearest_image(queries, refs, ap), [[.3, 0], [.6, .8]])


def test_wa_signed_all_references():
    refs = np.array([[1., 0.], [-.5, np.sqrt(.75)], [0., 1.]])
    queries = np.array([[1., 0.], [0., 1.]])
    ap = np.array([[.8, np.nan], [.4, .7], [.6, .9]])
    scores = weighted_average(queries, refs, ap)
    np.testing.assert_allclose(scores[0], [1.2, -.7])
    np.testing.assert_allclose(scores[1], (np.sqrt(.75) * ap[1] + ap[2]) / (np.sqrt(.75) + 1))
    with pytest.raises(ValueError, match="approximately zero"):
        weighted_average([[1., 0.]], [[1., 0.], [-1., 0.]], [[.8], [.4]])
    with pytest.raises(ValueError, match="approximately zero"):
        weighted_average([[1., 0.]], [[0., 1.]], [[.8]])


def test_ar_min_ties_and_nan():
    ap = np.array([[.9, .9, .2, np.nan], [np.nan, 0, 1, 1]])
    # Per-image ranks: [1,1,3,4] and [3,3,1,1].
    np.testing.assert_array_equal(average_ranking(ap), [-2, -2, -2, -2.5])
    np.testing.assert_array_equal(average_ranking([[.900000001, .9]]), [-1, -2])


def test_simsiam_loss_symmetric_stop_gradient(tiny_vit):
    torch.manual_seed(2)
    p0, z0, p1, z1 = [torch.randn(3, 8, requires_grad=True) for _ in range(4)]
    loss = simsiam_loss(p0, z0, p1, z1)
    expected = -.5 * (torch.nn.functional.cosine_similarity(p0, z1).mean()
                      + torch.nn.functional.cosine_similarity(p1, z0).mean())
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(loss, simsiam_loss(p1, z1, p0, z0))
    loss.backward()
    assert z0.grad is None and z1.grad is None
    assert p0.grad.abs().sum() > 0 and p1.grad.abs().sum() > 0
    model = SimSiam(tiny_vit)
    z, p = model(torch.randn(2, 3, 32, 32))
    assert not z.requires_grad and p.requires_grad
    p.square().mean().backward()
    assert model.projection.layers[0].weight.grad is not None
    assert tiny_vit.embeddings.cls_token.grad is not None


@pytest.mark.parametrize("count,size,expected", [(12, 5, [5, 5, 2]), (11, 5, [5, 6]), (12, 4, [4, 4, 4]),
                                                  (3, 2, [3]), (2, 32, [2])])
def test_all_images_in_batches(count, size, expected):
    sampler = _AllImageBatches(SequentialSampler(range(count)), size, drop_last=False)
    batches = list(sampler)
    assert [len(b) for b in batches] == expected
    assert len(sampler) == len(batches)
    assert [i for b in batches for i in b] == list(range(count))


def _deny_lightly(monkeypatch):
    original = builtins.__import__
    def denied(name, *args, **kwargs):
        if name == "lightly" or name.startswith("lightly."):
            raise ModuleNotFoundError("simulated missing lightly")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", denied)


def test_lightly_optional_error(monkeypatch, tiny_vit, synthetic_data, tmp_path):
    _deny_lightly(monkeypatch)
    with pytest.raises(ImportError, match=r"pip install rism\[baselines\]"):
        SimSiam(tiny_vit)
    result = CliRunner().invoke(app, ["train-simsiam", "--data-dir", str(synthetic_data),
                                     "--output-dir", str(tmp_path)])
    assert isinstance(result.exception, ImportError)
    assert "pip install rism[baselines]" in str(result.exception)


def test_ar_cli_without_checkpoint_or_lightly(synthetic_data, tmp_path, monkeypatch, caplog):
    _deny_lightly(monkeypatch)
    # Deliberately false CSV ranks must not become the predictions.
    path = synthetic_data / "train/train.csv"
    frame = pd.read_csv(path)
    frame["AP Rank"] = 99
    frame.to_csv(path, index=False)
    caplog.set_level("INFO")
    result = CliRunner().invoke(app, ["baselines", "--methods", "ar", "--data-dir", str(synthetic_data),
                                     "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert "60 rows disagree with CSV AP Rank" in caplog.text
    csv = pd.read_csv(tmp_path / "ar_test_results.csv")
    assert csv.columns.tolist() == ["category", "ndcg_3", "ndcg_5"]
    assert csv.category.tolist() == ["test-category-0", "test-category-1", "Average", "Standard Deviation"]
    assert not (tmp_path / "simsiam_features.npz").exists()
    test = ImageDataset(synthetic_data, "test", originals_only=True)
    ap = frame.pivot(index="Image Path", columns="Model ID", values="AP").fillna(0).to_numpy()
    scores = torch.tensor(np.broadcast_to(average_ranking(ap), (len(test), 5)).copy())
    labels = torch.stack([s["relevance"] for s in test.samples])
    for k in (3, 5):
        np.testing.assert_allclose(csv[f"ndcg_{k}"].iloc[:2], ndcg(scores, labels, k))
    missing = CliRunner().invoke(app, ["baselines"])
    assert missing.exit_code == 2 and "--simsiam-checkpoint is required" in missing.output


def test_simsiam_end_to_end(synthetic_data, local_vit, tmp_path, monkeypatch):
    output = tmp_path / "baselines"
    runner = CliRunner()
    args = ["--data-dir", str(synthetic_data), "--output-dir", str(output),
            "--batch-size", "5", "--num-workers", "0", "--accelerator", "cpu", "--devices", "1"]
    result = runner.invoke(app, ["train-simsiam", *args, "--vit-name", str(local_vit), "--epochs", "1"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    checkpoint = output / "simsiam_backbone.pth"
    vit, metadata = load_backbone(checkpoint)
    assert metadata["num_training_images"] == 12
    assert metadata["preprocessing"]["image_size"] == 32
    assert metadata["n_epochs"] == 1
    assert metadata["train_csv_sha256"] == hashlib.sha256((synthetic_data / "train/train.csv").read_bytes()).hexdigest()
    assert json.loads(checkpoint.with_suffix(".json").read_text()) == metadata
    assert (output / "lightning_logs/simsiam/metrics.csv").exists()
    saved = torch.load(checkpoint, weights_only=True)
    assert all(not key.startswith(("projection", "prediction")) for key in saved["state_dict"])
    from transformers import ViTModel
    original = ViTModel.from_pretrained(local_vit)
    assert not torch.equal(original.embeddings.cls_token, vit.embeddings.cls_token)
    def no_download(*args, **kwargs):
        raise AssertionError("Evaluation must never fetch pretrained weights")
    monkeypatch.setattr(ViTModel, "from_pretrained", no_download)
    _deny_lightly(monkeypatch)
    result = runner.invoke(app, ["baselines", *args, "--simsiam-checkpoint", str(checkpoint)])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    with np.load(output / "simsiam_features.npz") as cached:
        refs, queries = cached["references"], cached["queries"]
        assert refs.shape == (12, 32) and queries.shape == (2, 32)
        np.testing.assert_allclose(np.linalg.norm(refs, axis=1), 1, atol=1e-6)
        data = ImageDataset(synthetic_data, "train", image_size=32)
        assert cached["reference_paths"].tolist() == [s["image_path"] for s in data.samples]
        with torch.no_grad():
            cls = vit(pixel_values=torch.stack([data[i]["image"] for i in range(len(data))])).last_hidden_state[:, 0]
        np.testing.assert_allclose(refs, torch.nn.functional.normalize(cls, dim=1).numpy(), atol=1e-6)
    frame = pd.read_csv(synthetic_data / "train/train.csv")
    ap = frame.pivot(index="Image Path", columns="Model ID", values="AP").fillna(0).to_numpy()
    test = ImageDataset(synthetic_data, "test", originals_only=True)
    labels = torch.stack([s["relevance"] for s in test.samples])
    for method, scores in (("ni", nearest_image(queries, refs, ap)),
                           ("wa", weighted_average(queries, refs, ap)),
                           ("ar", np.broadcast_to(average_ranking(ap), (2, 5)).copy())):
        csv = pd.read_csv(output / f"{method}_test_results.csv")
        assert csv.category.tolist() == ["test-category-0", "test-category-1", "Average", "Standard Deviation"]
        for k in (3, 5):
            values = ndcg(torch.from_numpy(scores), labels, k).numpy()
            np.testing.assert_allclose(csv[f"ndcg_{k}"].iloc[:2], values)
            assert csv[f"ndcg_{k}"].iloc[2] == pytest.approx(values.mean())
            assert csv[f"ndcg_{k}"].iloc[3] == pytest.approx(values.std(ddof=0))
    cache = output / "simsiam_features.npz"
    before = cache.read_bytes()
    from rism.baselines import _Features
    def no_extraction(*args, **kwargs):
        raise AssertionError("Matching cache must skip extraction")
    with monkeypatch.context() as patch:
        patch.setattr(_Features, "predict_step", no_extraction)
        result = runner.invoke(app, ["baselines", *args, "--methods", "ni", "--simsiam-checkpoint", str(checkpoint)])
        assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert cache.read_bytes() == before
    # Changing a query source must invalidate the cache.
    from PIL import Image
    Image.new("RGB", (32, 32), "red").save(test.root / test.samples[0]["image_path"])
    result = runner.invoke(app, ["baselines", *args, "--methods", "ni", "--simsiam-checkpoint", str(checkpoint)])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert cache.read_bytes() != before
