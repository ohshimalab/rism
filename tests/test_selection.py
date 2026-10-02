"""Grid completeness and selection from parallel CV sources."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from rism.cli import app
from rism.config import DEFAULT_ALPHAS, DEFAULT_K_LOSSES, RunConfig
from rism.selection import select_hparams


def grid():
    return pd.DataFrame([
        dict(variant="ours", fold=f, alpha=a, k_loss=k, n_epochs=2,
             ndcg_3=.5, ndcg_5=.6)
        for a in DEFAULT_ALPHAS for k in DEFAULT_K_LOSSES for f in range(2)
    ])


def test_missing_grid_and_allow_incomplete(tmp_path):
    path = tmp_path / "cv.csv"
    frame = grid()
    frame = frame[~((frame.alpha == .3) & (frame.k_loss == 51))]
    frame = frame[~((frame.alpha == 1) & (frame.k_loss == 20))]
    frame.to_csv(path, index=False)
    target = tmp_path / "selection.json"
    with pytest.raises(ValueError, match=r"Missing paper grid.*\(0.3, 51\).*\(1.0, 20\)"):
        select_hparams(path, target, expected_folds=2)
    assert not target.exists()
    result = CliRunner().invoke(app, [
        "select-hparams", "--cv-csv", str(path), "--output-dir", str(tmp_path),
        "--expected-folds", "2", "--allow-incomplete",
    ])
    assert result.exit_code == 0, result.output
    assert len(json.loads((tmp_path / "ours_hparams.json").read_text())["configs"]) == 31


def test_merge_csvs_and_rounded_keys(tmp_path):
    frame = grid()
    paths = [tmp_path / "gpu0.csv", tmp_path / "gpu1.csv"]
    frame[frame.fold == 0].to_csv(paths[0], index=False)
    second = frame[frame.fold == 1].copy()
    second.loc[second.alpha == .3, "alpha"] += 1e-12
    second.to_csv(paths[1], index=False)
    result = CliRunner().invoke(app, [
        "select-hparams", "--cv-csv", str(paths[0]), "--cv-csv", str(paths[1]),
        "--output-dir", str(tmp_path), "--expected-folds", "2",
    ])
    assert result.exit_code == 0, result.output
    selected = json.loads((tmp_path / "ours_hparams.json").read_text())
    assert selected["source_csvs"] == [str(p.resolve()) for p in paths]
    assert len(selected["configs"]) == 33
    assert all(c["n_folds"] == 2 for c in selected["configs"])
    duplicate = frame.iloc[[0]].copy()
    duplicate["alpha"] += 1e-12
    pd.concat([second, duplicate]).to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        select_hparams(paths, tmp_path / "selection.json", expected_folds=2)


@pytest.mark.parametrize("field", ["seed", "train_csv_sha256", "categories"])
def test_merge_provenance(tmp_path, field):
    paths = [tmp_path / "gpu0.csv", tmp_path / "gpu1.csv"]
    frame = grid()
    provenance = dict(config=RunConfig().to_dict(), categories=["a", "b", "c"],
                      train_csv_sha256="original")
    for fold, path in enumerate(paths):
        frame[frame.fold == fold].to_csv(path, index=False)
        path.with_suffix(".run.json").write_text(json.dumps(provenance))
    other = json.loads(json.dumps(provenance))
    other["config"].update(num_workers=0, accelerator="cpu", devices=1)
    paths[1].with_suffix(".run.json").write_text(json.dumps(other))
    assert select_hparams(paths, tmp_path / "selection.json", expected_folds=2)["n_folds"] == 2
    if field == "seed":
        other["config"][field] = 99
    else:
        other[field] = "changed"
    paths[1].with_suffix(".run.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="provenance differs"):
        select_hparams(paths, tmp_path / "selection.json", expected_folds=2)
