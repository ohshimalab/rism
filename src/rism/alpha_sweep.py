"""Sec. 6.3: full-training fits for each CV-selected fixed-alpha configuration.

Test data are used only for reporting, never for k_loss/epoch selection. The
plot's best-alpha marker describes test performance; it does not select a model.
Resume identity includes selection contents, result-affecting run settings, and
train/test CSV hashes. Worker/device settings may change on resume.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig, result_config, validate_hparams
from .evaluation import evaluate
from .plotting import STYLE, plt, save_figure
from .training import train

COLUMNS = ["alpha", "k_loss", "n_epochs", "ndcg_3", "ndcg_5"]


def _data_hash(data_dir: Path, split: str) -> str:
    return hashlib.sha256((data_dir / split / f"{split}.csv").read_bytes()).hexdigest()


def plot_alpha(csv_path: Path | str, output_dir: Path | str | None = None) -> Path:
    """Replot recorded category-average test metrics without training/evaluation."""
    path = Path(csv_path)
    frame = pd.read_csv(path, float_precision="round_trip").sort_values("alpha")
    if frame.empty or set(COLUMNS) - set(frame.columns):
        raise ValueError("Alpha CSV must contain completed sweep results")
    if not np.isfinite(frame[COLUMNS].to_numpy()).all() or frame.alpha.duplicated().any():
        raise ValueError("Alpha CSV contains nonfinite or duplicate results")
    if not frame.alpha.between(0, 1).all():
        raise ValueError("Alpha values must be in [0, 1]")
    best = frame.sort_values(["ndcg_3", "ndcg_5", "alpha"], ascending=[False, False, True]).iloc[0]
    stem = (Path(output_dir) if output_dir is not None else path.parent) / path.stem
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
        ax.plot(frame.alpha, frame.ndcg_3, "o-", color="#0072B2", label="nDCG@3")
        ax.plot(frame.alpha, frame.ndcg_5, "s--", color="#D55E00", label="nDCG@5")
        ax.axvline(best.alpha, color="0.45", ls=":", lw=1, label=f"Best test nDCG@3: α={best.alpha:g}")
        ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Hybrid-loss weight α", ylabel="Mean test nDCG")
        ax.legend(fontsize=8, loc="best")
        save_figure(fig, stem)
    return stem.with_suffix(".pdf")


def alpha_sweep(config: RunConfig, hparams: Path | str, data_dir: Path | str,
                output_dir: Path | str) -> Path:
    selected = json.loads(Path(hparams).read_text())
    if selected.get("variant") != config.variant:
        raise ValueError("Selection JSON variant differs from requested variant")
    if not isinstance(selected.get("per_alpha"), list) or not selected["per_alpha"]:
        raise ValueError("Selection JSON must contain a nonempty per_alpha list")
    choices = []
    for entry in selected["per_alpha"]:
        alpha = float(entry["alpha"])
        k, epochs = entry["k_loss"], entry["n_epochs"]
        if int(k) != k or int(epochs) != epochs:
            raise ValueError("CV-selected k_loss and n_epochs must be integers")
        validate_hparams(alpha, int(k), int(epochs))
        choices.append({"alpha": alpha, "k_loss": int(k), "n_epochs": int(epochs)})
    choices.sort(key=lambda entry: entry["alpha"])
    if len({r["alpha"] for r in choices}) != len(choices):
        raise ValueError("Selection JSON has duplicate alphas")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"alpha_sweep_{config.variant}.csv"
    provenance = {"config": result_config(config), "per_alpha": choices,
                  "train_sha256": _data_hash(Path(data_dir), "train"),
                  "test_sha256": _data_hash(Path(data_dir), "test"),
                  "weight_decay": 1e-4, "test_images": "original", "aggregation": "category_average"}
    sidecar = path.with_suffix(".run.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != provenance:
            raise ValueError("Resume provenance differs; use a new output directory")
    elif path.exists():
        raise ValueError("Cannot resume a CSV without its .run.json provenance")
    else:
        sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    completed = set()
    if path.exists():
        frame = pd.read_csv(path, float_precision="round_trip")
        if set(frame.columns) != set(COLUMNS) or not np.isfinite(frame[COLUMNS].to_numpy()).all():
            raise ValueError("Alpha CSV has missing columns or incomplete results")
        if frame.alpha.duplicated().any():
            raise ValueError("Alpha CSV contains duplicate alphas")
        for row in frame.itertuples():
            if {"alpha": row.alpha, "k_loss": row.k_loss, "n_epochs": row.n_epochs} not in choices:
                raise ValueError("Alpha CSV configuration differs from selection")
            if not 0 <= row.ndcg_3 <= 1.000001 or not 0 <= row.ndcg_5 <= 1.000001:
                raise ValueError("Alpha CSV contains invalid ranking scores")
            completed.add(float(row.alpha))
    for choice in choices:
        if choice["alpha"] in completed:
            continue
        directory = output_dir / f"alpha_{choice['alpha']}"
        checkpoint = train(config, data_dir, directory, alpha=choice["alpha"],
                           k_loss=choice["k_loss"], epochs=choice["n_epochs"])
        result_path = evaluate(checkpoint, data_dir, directory, config,
                               expected_hparams=(choice["alpha"], choice["k_loss"], choice["n_epochs"]))
        average = pd.read_csv(result_path).set_index("category").loc["Average"]
        row = {**choice, "ndcg_3": float(average.ndcg_3), "ndcg_5": float(average.ndcg_5)}
        if not np.isfinite([row["ndcg_3"], row["ndcg_5"]]).all():
            raise ValueError("Evaluation returned nonfinite test averages")
        new_file = not path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    plot_alpha(path)
    return path
