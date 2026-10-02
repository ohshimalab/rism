"""Offline analysis workflows, including exported plots and resume boundaries."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner
from transformers import ViTConfig, ViTModel

from rism.alpha_sweep import alpha_sweep
from rism.attention import attention_maps
from rism.cli import app
from rism.config import DEFAULT_ALPHAS, DEFAULT_K_LOSSES, RunConfig
from rism.latency import latency
from rism.models import Retriever
from rism.selection import select_hparams
from rism.training import make_trainer


def checkpoint(path, vit, variant="ours"):
    model = Retriever(vit, 5, variant, nhead=2, hidden_dim=16, dim_feedforward=32)
    metadata = {"vit_config": vit.config.to_dict(), "model_kwargs": dict(model.hparams),
                "model_ids": list(range(5)), "alpha": .5, "k_loss": 51, "n_epochs": 1}
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return model


def test_attention_exports(synthetic_data, tiny_vit, tmp_path):
    # Tiny hidden size but the paper's 14x14 patch geometry. No downloads.
    config = tiny_vit.config.to_dict()
    config["image_size"] = 224
    path = tmp_path / "ours.pth"
    checkpoint(path, ViTModel(ViTConfig.from_dict(config)))
    output = tmp_path / "attention"
    runner = CliRunner()
    result = runner.invoke(app, ["attention-maps", "--checkpoint", str(path), "--data-dir", str(synthetic_data),
                                "--output-dir", str(output), "--model-ids", "3", "--model-ids", "0",
                                "--batch-size", "2", "--accelerator", "cpu", "--devices", "1"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    maps = output / "attention_maps"
    assert len(list(maps.glob("*.png"))) == 4
    npz = np.load(maps / "attention_weights.npz", allow_pickle=False)
    assert npz["attention_weights"].shape == (2, 5, 14, 14)
    assert npz["model_ids"].tolist() == list(range(5))
    assert npz["categories"].tolist() == ["test-category-0", "test-category-1"]
    with Image.open(maps / "test-category-0_image-0_model4.png") as image:
        assert image.size == (32, 32)
        rgb = np.asarray(image) / 255
    from rism.plotting import plt
    from torch.nn import functional as F
    raw = torch.from_numpy(npz["attention_weights"][0, 3])
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    up = F.interpolate(normalized[None, None], size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    expected = .45 * plt.get_cmap("inferno")(up.numpy())[..., :3] + .55 * np.array([0, 0, 80]) / 255
    np.testing.assert_allclose(rgb, expected, atol=1/255)
    assert (output / "attention_grid.pdf").stat().st_size > 1000
    assert (output / "attention_grid.png").exists()
    for variant, message in [("prior", "MLP variants"), ("global_te", "one image token")]:
        bad = tmp_path / f"{variant}.pth"
        checkpoint(bad, tiny_vit, variant)
        with pytest.raises(ValueError, match=message):
            attention_maps(bad, synthetic_data, output)
    with pytest.raises(ValueError, match="candidate range"):
        attention_maps(path, synthetic_data, output, model_ids=[5])
    with pytest.raises(ValueError, match="categories"):
        attention_maps(path, synthetic_data, output, categories=["absent"])


def test_latency_single_encoding(tiny_vit, tmp_path, monkeypatch):
    path = tmp_path / "ours.pth"
    checkpoint(path, tiny_vit)
    encode, score = Retriever.encode_image, Retriever.score_features
    encoding_calls, scoring_calls = [], []
    def record_encode(self, images):
        encoding_calls.append(images.shape)
        assert not torch.is_grad_enabled()
        assert not self.training
        assert images.dtype == torch.float32
        return encode(self, images)
    def record_score(self, features, model_embeddings=None, chunk_size=None):
        scoring_calls.append((len(model_embeddings), chunk_size, model_embeddings.data_ptr()))
        return score(self, features, model_embeddings, chunk_size)
    monkeypatch.setattr(Retriever, "encode_image", record_encode)
    monkeypatch.setattr(Retriever, "score_features", record_score)
    output = tmp_path / "latency"
    result = CliRunner().invoke(app, ["latency", "--checkpoint", str(path), "--output-dir", str(output),
                                     "--pool-sizes", "2", "--pool-sizes", "4", "--pool-sizes", "8",
                                     "--chunk-size", "4", "--warmup", "1", "--repeats", "2", "--device", "cpu"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    frame = pd.read_csv(output / "latency_ours.csv")
    assert frame.pool_size.tolist() == [2, 4, 8]
    assert frame.n_batches.tolist() == [1, 1, 2]
    assert frame.mean_ms.gt(0).all()
    assert frame.peak_memory_mib.isna().all()
    assert len(encoding_calls) == 9
    assert len(scoring_calls) == 9
    for start in (0, 3, 6):
        assert len({item[2] for item in scoring_calls[start:start + 3]}) == 1
    assert (output / "latency_ours.pdf").exists()
    assert (output / "latency_ours.png").exists()
    with pytest.raises(ValueError, match="positive"):
        latency(output, checkpoint=path, pool_sizes=[0])


def test_alpha_sweep_resume(synthetic_data, local_vit, tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"variant": "ours", "per_alpha": [
        {"alpha": .7, "k_loss": 5, "n_epochs": 1}, {"alpha": .2, "k_loss": 3, "n_epochs": 1}]}))
    output = tmp_path / "sweep"
    calls = []
    original = ViTModel.from_pretrained
    def fresh(name, **kwargs):
        calls.append(name)
        return original(name, **kwargs)
    monkeypatch.setattr(ViTModel, "from_pretrained", fresh)
    config = RunConfig(vit_name=str(local_vit), num_workers=0, batch_size=4,
                       nhead=2, accelerator="cpu", devices=1, max_epochs=1)
    args = ["alpha-sweep", "--hparams", str(selection), "--data-dir", str(synthetic_data),
            "--output-dir", str(output), "--vit-name", str(local_vit), "--num-workers", "0",
            "--batch-size", "4", "--nhead", "2", "--accelerator", "cpu", "--devices", "1", "--max-epochs", "1"]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    csv_path = output / "alpha_sweep_ours.csv"
    frame = pd.read_csv(csv_path)
    assert frame.alpha.tolist() == [.2, .7]
    assert frame.k_loss.tolist() == [3, 5]
    assert len(calls) == 2
    for row in frame.itertuples():
        folder = output / f"alpha_{row.alpha:g}"
        results = pd.read_csv(folder / "ours_test_results.csv").set_index("category")
        assert row.ndcg_3 == pytest.approx(results.loc["Average", "ndcg_3"])
        assert (folder / "lightning_logs/ours/final/metrics.csv").exists()
    before = csv_path.read_bytes()
    alpha_sweep(replace(config, num_workers=1, accelerator="auto", devices="auto"),
                selection, synthetic_data, output)
    assert csv_path.read_bytes() == before
    assert len(calls) == 2
    # Incomplete sweep resumes just the missing alpha.
    frame.iloc[:1].to_csv(csv_path, index=False)
    alpha_sweep(config, selection, synthetic_data, output)
    assert len(calls) == 3
    assert len(pd.read_csv(csv_path)) == 2
    result = runner.invoke(app, ["plot-alpha", "--csv-path", str(csv_path)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert (output / "alpha_sweep_ours.pdf").exists()
    assert (output / "alpha_sweep_ours.png").exists()
    with pytest.raises(ValueError, match="provenance"):
        alpha_sweep(replace(config, seed=1), selection, synthetic_data, output)
    with pytest.raises(ValueError, match="variant"):
        alpha_sweep(replace(config, variant="prior"), selection, synthetic_data, output)
    sample = synthetic_data / "test/test-category-0/images/image-0.png"
    Image.new("RGB", (32, 32), "red").save(sample)
    # Image bytes are intentionally outside the CSV-only resume identity.
    alpha_sweep(config, selection, synthetic_data, output)
    assert len(calls) == 3
    csv = synthetic_data / "test/test.csv"
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        alpha_sweep(config, selection, synthetic_data, output)


def test_expected_folds(tmp_path):
    path = tmp_path / "cv.csv"
    rows = [dict(variant="ours", fold=f, alpha=a, k_loss=k, n_epochs=1,
                 ndcg_3=.5, ndcg_5=.6) for a in DEFAULT_ALPHAS
            for k in DEFAULT_K_LOSSES for f in range(3)]
    pd.DataFrame(rows).to_csv(path, index=False)
    target = tmp_path / "selection.json"
    assert select_hparams(path, target, expected_folds=3)["n_folds"] == 3
    assert select_hparams(path, target, expected_folds=3)["expected_folds"] == 3
    with pytest.raises(ValueError, match="all 4 folds"):
        select_hparams(path, target, expected_folds=4)
    with pytest.raises(ValueError, match="Invalid CV fold"):
        select_hparams(path, target, expected_folds=2, allow_incomplete=True)
    result = CliRunner().invoke(app, ["select-hparams", "--cv-csv", str(path),
                                     "--output-dir", str(tmp_path), "--expected-folds", "3"])
    assert result.exit_code == 0, result.output


def test_logger_paths(tmp_path):
    config = RunConfig(accelerator="cpu", devices=1, num_workers=0)
    trainer = make_trainer(config, tmp_path, epochs=1)
    assert trainer.logger.log_dir.rstrip("/") == str(tmp_path / "lightning_logs/ours/final")
    prediction = make_trainer(config, tmp_path, epochs=1, prediction=True)
    assert prediction.logger is None
    assert not list(tmp_path.rglob("version_*"))


def test_latency_without_checkpoint_uses_local_weights(local_vit, tmp_path):
    path = latency(tmp_path / "fresh", vit_name=str(local_vit), nhead=2,
                   pool_sizes=[2], warmup=0, repeats=1, chunk_size=4, device="cpu")
    frame = pd.read_csv(path)
    assert frame.n_batches.tolist() == [1]
    assert frame.std_ms.tolist() == [0]
    assert frame.device_name.tolist() == ["cpu"]


def test_attention_tiny_geometry_and_grid_subset(synthetic_data, tiny_vit, tmp_path):
    path = tmp_path / "tiny.pth"
    checkpoint(path, tiny_vit)
    output = tmp_path / "tiny-maps"
    attention_maps(path, synthetic_data, output, categories=["test-category-1"],
                   batch_size=1, accelerator="cpu", devices=1)
    maps = output / "attention_maps"
    assert len(list(maps.glob("*.png"))) == 10
    with np.load(maps / "attention_weights.npz", allow_pickle=False) as raw:
        assert raw["attention_weights"].shape == (2, 5, 2, 2)
    assert (output / "attention_grid.pdf").exists()


def test_attention_grid_mixed_aspects(synthetic_data, tiny_vit, tmp_path, monkeypatch):
    import rism.attention as module
    path = tmp_path / "tiny.pth"
    checkpoint(path, tiny_vit)
    for index, width in enumerate((96, 48)):
        image = synthetic_data / f"test/test-category-{index}/images/image-0.png"
        Image.new("RGB", (width, 32), (30 * index, 0, 80)).save(image)
    save = module.save_figure
    def inspect_grid(fig, stem):
        fig.canvas.draw()
        axes = fig.axes
        assert all(not ax.get_frame_on() for ax in axes)
        assert [axes[i].get_ylabel() for i in (0, 2)] == ["Model 1", "Model 4"]
        bounds = [ax.get_window_extent() for ax in axes]
        assert bounds[0].width / bounds[0].height == pytest.approx(3)
        assert bounds[1].width / bounds[1].height == pytest.approx(1.5)
        # Whitespace between candidate rows must be small relative to image height.
        gap = bounds[0].y0 - bounds[2].y1
        assert 0 <= gap < .1 * bounds[0].height
        save(fig, stem)
    monkeypatch.setattr(module, "save_figure", inspect_grid)
    module.attention_maps(path, synthetic_data, tmp_path / "mixed-maps", model_ids=[0, 3],
                          batch_size=2, accelerator="cpu", devices=1)
