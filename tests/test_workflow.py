import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner
from transformers import ViTModel

from rism.cli import app
from rism.config import DEFAULT_ALPHAS, DEFAULT_K_LOSSES, RunConfig
from rism.cross_validation import category_splits, cross_validate
from rism.evaluation import category_summary
from rism.selection import select_hparams
from rism.training import BestEpoch, load_checkpoint


def test_category_splits():
    categories = [f"category-{i:02}" for i in range(20)]
    expected = sorted(categories)
    random.Random(42).shuffle(expected)
    splits = category_splits(reversed(categories))
    for f, split in enumerate(splits):
        assert split.eval_category == expected[f]
        assert split.val_category == expected[f - 1]
        assert len(split.optimization_categories) == 18
        assert set(split.optimization_categories) | {split.eval_category, split.val_category} == set(categories)
    assert splits[0].val_category == splits[-1].eval_category


def test_best_epoch_is_not_stopping_epoch():
    class FakeTrainer:
        sanity_checking = False
        current_epoch = 0
        callback_metrics = {}
    trainer = FakeTrainer()
    tracker = BestEpoch()
    for epoch, loss in enumerate([1., .4, .5, .4, .6]):
        trainer.current_epoch = epoch
        trainer.callback_metrics = {"val_loss": torch.tensor(loss)}
        tracker.on_validation_epoch_end(trainer, None)
    assert tracker.n_epochs == 2
    assert tracker.best_val_loss == pytest.approx(.4)


def test_selection_tie_break_completeness_and_epochs(tmp_path):
    rows = []
    for alpha, k, n5 in [(.8, 5, .7), (.5, 20, .8), (.5, 5, .8), (.1, 5, .7)]:
        for fold in range(20):
            rows.append(dict(variant="ours", fold=fold, alpha=alpha, k_loss=k,
                             ndcg_3=.75, ndcg_5=n5, n_epochs=1 if fold < 10 else 4))
    present = {(r["alpha"], r["k_loss"]) for r in rows}
    for alpha in DEFAULT_ALPHAS:
        for k in DEFAULT_K_LOSSES:
            if (alpha, k) not in present:
                rows.extend(dict(variant="ours", fold=f, alpha=alpha, k_loss=k,
                                 ndcg_3=.5, ndcg_5=.5, n_epochs=1) for f in range(20))
    path = tmp_path / "cv.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = select_hparams(path, tmp_path / "selection.json")
    assert (result["alpha"], result["k_loss"]) == (.5, 5)
    assert result["mean_epochs"] == 2.5
    assert result["n_epochs"] == 2
    assert len(result["per_alpha"]) == 11
    incomplete = pd.DataFrame(rows).query("fold != 0")
    incomplete.to_csv(path, index=False)
    with pytest.raises(ValueError, match="all 20 folds"):
        select_hparams(path, tmp_path / "selection.json")
    assert select_hparams(path, tmp_path / "selection.json", allow_incomplete=True)["n_folds"] == 19
    pd.concat([incomplete, incomplete.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        select_hparams(path, tmp_path / "selection.json", allow_incomplete=True)


def test_population_category_summary():
    categories = pd.DataFrame({"category": ["a", "b"], "ndcg_3": [.2, .8], "ndcg_5": [.4, .6]})
    result = category_summary(categories)
    assert result.iloc[-2].category == "Average"
    assert result.iloc[-1].category == "Standard Deviation"
    assert result.iloc[-1].ndcg_3 == pytest.approx(.3)
    assert result.iloc[-1].ndcg_5 == pytest.approx(.1)


def test_cli_end_to_end_and_resume(synthetic_data, local_vit, tmp_path, monkeypatch):
    calls = []
    original = ViTModel.from_pretrained
    def fresh(name, **kwargs):
        calls.append(name)
        return original(name, **kwargs)
    monkeypatch.setattr(ViTModel, "from_pretrained", fresh)
    runner = CliRunner()
    output = tmp_path / "outputs"
    common = ["--data-dir", str(synthetic_data), "--output-dir", str(output),
              "--vit-name", str(local_vit), "--accelerator", "cpu", "--devices", "1",
              "--num-workers", "0", "--batch-size", "4", "--nhead", "2", "--max-epochs", "2"]
    def invoke(args):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
        return result
    cv = ["cross-validate", *common, "--folds", "0", "--folds", "1", "--alphas", "0.5", "--k-loss", "5"]
    invoke(cv)
    csv = output / "ours_cv.csv"
    frame = pd.read_csv(csv)
    assert len(frame) == 2
    assert frame.n_epochs.between(1, 2).all()
    assert frame[["ndcg_3", "ndcg_5"]].ge(0).all().all()
    assert len(calls) == 4
    before = csv.read_bytes()
    invoke(cv)
    assert csv.read_bytes() == before
    assert len(calls) == 4
    execution_changed = RunConfig(vit_name=str(local_vit), num_workers=1, batch_size=4,
                                 nhead=2, accelerator="auto", devices="auto", max_epochs=2)
    cross_validate(execution_changed, synthetic_data, output,
                   folds=[0, 1], alphas=[.5], k_losses=[5])
    assert csv.read_bytes() == before
    assert len(calls) == 4
    # Step-1 sidecars included execution settings; they also remain resumable.
    sidecar = csv.with_suffix(".run.json")
    provenance = json.loads(sidecar.read_text())
    provenance["config"].update(num_workers=0, accelerator="cpu", devices=1)
    sidecar.write_text(json.dumps(provenance))
    cross_validate(execution_changed, synthetic_data, output,
                   folds=[0], alphas=[.5], k_losses=[5])
    assert len(calls) == 4
    for fold in (0, 1):
        for stage in ("inner", "outer"):
            assert (output / f"lightning_logs/ours/fold_{fold}/alpha_0.5_k_5/{stage}/metrics.csv").exists()
    assert not list(output.rglob("version_*"))
    invoke(["select-hparams", "--cv-csv", str(csv), "--output-dir", str(output), "--allow-incomplete"])
    hparams = output / "ours_hparams.json"
    selected = json.loads(hparams.read_text())
    assert selected["alpha"] == .5
    assert selected["k_loss"] == 5
    assert selected["per_alpha"][0]["n_folds"] == 2
    invoke(["train", *common, "--hparams", str(hparams)])
    checkpoint = output / "ours_checkpoint.pth"
    assert checkpoint.exists() and checkpoint.with_suffix(".json").exists()
    model, metadata = load_checkpoint(checkpoint)
    assert len(calls) == 5
    assert metadata["n_epochs"] == selected["n_epochs"]
    invoke(["evaluate", *common, "--checkpoint", str(checkpoint), "--hparams", str(hparams)])
    assert len(calls) == 5
    results = pd.read_csv(output / "ours_test_results.csv")
    assert results.columns.tolist() == ["category", "ndcg_3", "ndcg_5"]
    assert results.category.tolist()[-2:] == ["Average", "Standard Deviation"]
    categories = results.iloc[:-2]
    np.testing.assert_allclose(results.iloc[-1][["ndcg_3", "ndcg_5"]].astype(float),
                               categories[["ndcg_3", "ndcg_5"]].std(ddof=0))
    changed = RunConfig(vit_name=str(local_vit), num_workers=0, batch_size=4,
                        nhead=2, accelerator="cpu", devices=1, max_epochs=2, seed=7)
    with pytest.raises(ValueError, match="provenance differs"):
        cross_validate(changed, synthetic_data, output, folds=[0], alphas=[.5], k_losses=[5])


def test_failed_cv_is_not_silently_skipped(synthetic_data, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate model initialization failure")
    monkeypatch.setattr("rism.cross_validation.build_model", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        cross_validate(RunConfig(num_workers=0), synthetic_data, tmp_path / "failed",
                       folds=[0], alphas=[.5], k_losses=[5])
    assert not (tmp_path / "failed/ours_cv.csv").exists()
