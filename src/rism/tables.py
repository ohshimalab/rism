"""Assemble paper tables from existing evaluations; never recompute results."""

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

METHODS = {"ours": "Ours", "prior": "Prior", "wa": "WA", "ni": "NI", "ar": "AR",
           "global_te": "Global+TE", "local_mlp": "Local+MLP"}
SUMMARY = ["Average", "Standard Deviation"]
NOTE = ('Dataset category `circle-glass-marble` is called "Sphere Glass Marble" in the paper. '
        'CSV values retain input precision. Markdown uses three decimals; in Table 1 category '
        'and Average rows, best (bold) and '
        'second-best (underlined) are distinct unrounded scores, with ties sharing marks.')


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["category", "ndcg_3", "ndcg_5"]
    if not set(required) <= set(frame.columns):
        raise ValueError(f"{path}: required columns are {required}")
    frame = frame[required]
    if frame.empty or frame.category.isna().any() or frame.category.duplicated().any():
        raise ValueError(f"{path}: empty, missing or duplicate categories")
    if not set(SUMMARY) <= set(frame.category) or len(frame) <= 2:
        raise ValueError(f"{path}: test categories and both summary rows are required")
    for metric in required[1:]:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    if not np.isfinite(frame[required[1:]].to_numpy()).all():
        raise ValueError(f"{path}: metrics must be finite")
    return frame.set_index("category")


def _markdown(frame: pd.DataFrame, groups: Sequence[Sequence[str]] = ()) -> str:
    lines = ["| " + " | ".join([str(frame.index.name), *frame.columns]) + " |",
             "| " + " | ".join(["---"] * (len(frame.columns) + 1)) + " |"]
    for label, row in frame.iterrows():
        cells = {column: f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                 for column, value in row.items()}
        for group in groups if label != "Standard Deviation" else ():
            ranks = sorted(set(row[list(group)]), reverse=True)
            for column in group:
                if row[column] == ranks[0]:
                    cells[column] = f"**{cells[column]}**"
                elif len(ranks) > 1 and row[column] == ranks[1]:
                    cells[column] = f"<u>{cells[column]}</u>"
        lines.append("| " + " | ".join([str(label), *cells.values()]) + " |")
    return "\n".join(lines) + "\n\n" + NOTE + "\n"


def make_tables(results_dirs: Sequence[Path | str], output_dir: Path | str, *,
                paths: Mapping[str, Path | str | None] | None = None) -> list[Path]:
    """Explicit method paths override directory discovery; missing methods are omitted."""
    paths = paths or {}
    frames = {}
    for method in METHODS:
        explicit = paths.get(method)
        candidates = ([Path(explicit)] if explicit is not None else
                      list(dict.fromkeys((Path(d) / f"{method}_test_results.csv").resolve()
                                         for d in results_dirs
                                         if (Path(d) / f"{method}_test_results.csv").is_file())))
        if len(candidates) > 1:
            raise ValueError(f"Multiple CSVs for {method}; specify --{method.replace('_', '-')}-results")
        if not candidates or not candidates[0].is_file():
            logging.warning("Skipping missing method %s", method)
            continue
        frames[method] = _read(candidates[0])
    if not frames:
        raise ValueError("At least one method result CSV is required")
    category_sets = [set(frame.index) - set(SUMMARY) for frame in frames.values()]
    if any(categories != category_sets[0] for categories in category_sets[1:]):
        raise ValueError("Method CSVs have different test categories")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(category_sets[0]) + SUMMARY
    groups = []
    table1 = pd.DataFrame(index=pd.Index(rows, name="Category"))
    for metric in (3, 5):
        group = []
        for method in ("ours", "prior", "wa", "ni", "ar"):
            if method in frames:
                column = f"nDCG@{metric} {METHODS[method]}"
                table1[column] = frames[method].loc[rows, f"ndcg_{metric}"].to_numpy()
                group.append(column)
        if group:
            groups.append(group)
    table2 = pd.DataFrame(index=pd.Index(
        ["Image Feature", "Model–Image Fusion", "nDCG@3", "nDCG@5"], name="Method"))
    for method, image, fusion in [("prior", "Global", "MLP"), ("global_te", "Global", "TE"),
                                  ("local_mlp", "Local", "MLP"), ("ours", "Local", "TE")]:
        if method in frames:
            table2[METHODS[method]] = [image, fusion, frames[method].loc["Average", "ndcg_3"],
                                     frames[method].loc["Average", "ndcg_5"]]
    written = []
    for number, frame, marks in [(1, table1, groups), (2, table2, [])]:
        csv_path, md_path = output / f"table{number}.csv", output / f"table{number}.md"
        frame.to_csv(csv_path)
        md_path.write_text(_markdown(frame, marks), encoding="utf-8")
        written.extend([csv_path, md_path])
    (output / "README.md").write_text(NOTE + "\n", encoding="utf-8")
    return written
