import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

from rism.cli import app
from rism.tables import make_tables


def write_result(directory, method, n3, n5):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{method}_test_results.csv"
    pd.DataFrame({"category": ["z", "circle-glass-marble", "Average", "Standard Deviation"],
                  "ndcg_3": [n3, n3, n3, .1], "ndcg_5": [n5, n5, n5, .2]}).to_csv(path, index=False)
    return path


def test_tables_values_order_and_ties(tmp_path):
    inputs = tmp_path / "inputs"
    values = {"ours": (.9, .5), "prior": (.9, .6), "wa": (.8, .6),
              "ni": (.8, .4), "ar": (.7, .5), "global_te": (.75, .7), "local_mlp": (.85, .8)}
    for method, (n3, n5) in values.items():
        write_result(inputs, method, n3, n5)
    output = tmp_path / "tables"
    make_tables([inputs], output)
    t1 = pd.read_csv(output / "table1.csv", index_col=0)
    assert t1.index.tolist() == ["circle-glass-marble", "z", "Average", "Standard Deviation"]
    assert t1.columns.tolist() == [f"nDCG@{k} {m}" for k in (3, 5)
                                   for m in ("Ours", "Prior", "WA", "NI", "AR")]
    assert t1.loc["Average", "nDCG@3 Ours"] == .9
    assert t1.loc["Standard Deviation", "nDCG@5 AR"] == .2
    markdown = (output / "table1.md").read_text()
    assert '| circle-glass-marble | **0.900** | **0.900** | <u>0.800</u> | <u>0.800</u> | 0.700 | <u>0.500</u> | **0.600** | **0.600** | 0.400 | <u>0.500</u> |' in markdown
    average = next(line for line in markdown.splitlines() if line.startswith('| Average'))
    assert '**0.900**' in average and '<u>0.800</u>' in average
    std = next(line for line in markdown.splitlines() if line.startswith('| Standard Deviation'))
    assert '**' not in std and '<u>' not in std
    assert std == '| Standard Deviation | ' + ' | '.join(['0.100'] * 5 + ['0.200'] * 5) + ' |'
    assert 'Sphere Glass Marble' in markdown
    t2 = pd.read_csv(output / "table2.csv", index_col=0)
    assert t2.columns.tolist() == ["Prior", "Global+TE", "Local+MLP", "Ours"]
    assert t2.loc["Image Feature"].tolist() == ["Global", "Global", "Local", "Local"]
    assert t2.loc["Model–Image Fusion"].tolist() == ["MLP", "TE", "MLP", "TE"]
    assert t2.loc["nDCG@3"].astype(float).tolist() == [.9, .75, .85, .9]
    assert t2.loc["nDCG@5"].astype(float).tolist() == [.6, .7, .8, .5]
    assert '| nDCG@3 | 0.900 | 0.750 | 0.850 | 0.900 |' in (output / 'table2.md').read_text()


def test_missing_methods_multiple_dirs_and_explicit_override(tmp_path, caplog):
    a, b, output = tmp_path / 'a', tmp_path / 'b', tmp_path / 'out'
    write_result(a, 'ours', .123456, .6)
    prior = write_result(b, 'prior', .4, .7)
    with caplog.at_level(logging.WARNING):
        make_tables([a, b], output)
    assert 'Skipping missing method wa' in caplog.text
    frame = pd.read_csv(output / 'table1.csv')
    assert frame['nDCG@3 Ours'].iloc[0] == .123456
    assert len(frame.columns) == 5
    runner = CliRunner()
    result = runner.invoke(app, ['make-tables', '--results-dir', str(a), '--results-dir', str(b),
                                 '--output-dir', str(output)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ['make-tables', '--prior-results', str(prior), '--output-dir', str(output)])
    assert result.exit_code == 0, result.output
    assert pd.read_csv(output / 'table2.csv').columns.tolist() == ['Method', 'Prior']
    with pytest.raises(ValueError, match='At least one'):
        make_tables([], output)
    assert runner.invoke(app, ['make-tables', '--output-dir', str(output)]).exit_code != 0
    write_result(a, 'prior', .5, .8)
    with pytest.raises(ValueError, match='Multiple CSVs'):
        make_tables([a, b], output)
    make_tables([a, b], output, paths={'prior': prior})


def test_invalid_results_rejected(tmp_path):
    inputs = tmp_path / 'in'
    path = write_result(inputs, 'ours', .5, .6)
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match='summary rows'):
        make_tables([inputs], tmp_path / 'out')
    write_result(inputs, 'ours', .5, .6)
    prior = write_result(inputs, 'prior', .4, .7)
    frame = pd.read_csv(prior)
    frame.loc[0, 'category'] = 'different'
    frame.to_csv(prior, index=False)
    with pytest.raises(ValueError, match='different test categories'):
        make_tables([inputs], tmp_path / 'out')
