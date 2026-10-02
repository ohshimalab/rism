"""Nested leave-one-category-out training with incremental, resumable results."""

import csv
import gc
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightning.pytorch as pl
import pandas as pd
import torch

from .config import (DEFAULT_ALPHAS, DEFAULT_K_LOSSES, RunConfig, result_config,
                     result_provenance, validate_hparams)
from .data import ImageDataset
from .evaluation import evaluate_model
from .training import build_model, fit_model, image_size

CV_COLUMNS = ["variant", "fold", "eval_category", "val_category", "alpha", "k_loss",
              "n_epochs", "best_val_loss", "ndcg_3", "ndcg_5"]


@dataclass(frozen=True)
class CategorySplit:
    fold: int
    eval_category: str
    val_category: str
    optimization_categories: tuple[str, ...]


def category_splits(categories: Iterable[str], seed: int = 42) -> list[CategorySplit]:
    """Sec. 5.1: predecessor validation category in a seeded shuffled order."""
    categories = sorted(set(categories))
    if len(categories) < 3:
        raise ValueError("Nested CV requires at least three categories")
    random.Random(seed).shuffle(categories)
    return [CategorySplit(f, category, categories[f - 1], tuple(
        c for c in categories if c not in (category, categories[f - 1])))
        for f, category in enumerate(categories)]


def _resume_keys(path: Path, splits: list[CategorySplit], variant: str) -> set[tuple[int, float, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if set(CV_COLUMNS) - set(frame.columns):
        raise ValueError("CV CSV has missing columns")
    if frame[CV_COLUMNS].isna().any().any():
        raise ValueError("CV CSV contains incomplete rows")
    selected = frame[frame.variant == variant]
    if selected.duplicated(["fold", "alpha", "k_loss"]).any():
        raise ValueError("CV CSV contains duplicate result keys")
    for row in selected.itertuples():
        if row.fold != int(row.fold) or not 0 <= row.fold < len(splits):
            raise ValueError("CV CSV contains an invalid fold")
        split = splits[int(row.fold)]
        if (row.eval_category, row.val_category) != (split.eval_category, split.val_category):
            raise ValueError("CV CSV category order differs from this run")
    return {(int(r.fold), round(float(r.alpha), 10), int(r.k_loss)) for r in selected.itertuples()}


def cross_validate(config: RunConfig, data_dir: Path | str, output_dir: Path | str, *,
                   folds: Iterable[int] | None = None, alphas: Iterable[float] | None = None,
                   k_losses: Iterable[int] | None = None, csv_path: Path | str | None = None) -> Path:
    """Fit fresh inner and outer models for each requested fold/configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(csv_path) if csv_path is not None else output_dir / f"{config.variant}_cv.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ImageDataset(data_dir, "train")
    splits = category_splits(data.categories, config.seed)
    fold_ids = list(range(len(splits))) if folds is None else sorted(set(folds))
    if any(f < 0 or f >= len(splits) for f in fold_ids):
        raise ValueError("Requested fold is outside the category range")
    alphas = sorted(set(round(a, 10) for a in (alphas if alphas is not None else DEFAULT_ALPHAS)))
    k_losses = sorted(set(k_losses if k_losses is not None else DEFAULT_K_LOSSES))
    if not fold_ids or not alphas or not k_losses:
        raise ValueError("CV sweep must not be empty")
    for alpha in alphas:
        for k in k_losses:
            validate_hparams(alpha, k)
    # Protect partial results from accidental mixing of seeds/data/training settings.
    provenance = {"config": result_config(config), "categories": data.categories,
                  "train_csv_sha256": hashlib.sha256(
                      (Path(data_dir) / "train/train.csv").read_bytes()).hexdigest(),
                  "outer_images": "original", "epoch_selection": "lowest_val_loss",
                  "early_stopping_patience": 5, "weight_decay": 1e-4}
    sidecar = path.with_suffix(".run.json")
    if sidecar.exists():
        previous = result_provenance(json.loads(sidecar.read_text()))
        if previous != provenance:
            raise ValueError("Resume provenance differs; use a new output CSV")
    elif path.exists():
        raise ValueError("Cannot safely resume a CSV without its .run.json provenance")
    else:
        sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    complete = _resume_keys(path, splits, config.variant)
    for f in fold_ids:
        split = splits[f]
        for alpha in alphas:
            for k in k_losses:
                key = (f, alpha, k)
                if key in complete:
                    logging.info("Resume: skipping %s fold=%d alpha=%s k_loss=%d", config.variant, f, alpha, k)
                    continue
                logging.info("CV %s fold=%d alpha=%s k_loss=%d", config.variant, f, alpha, k)
                # Sec. 5.1: independent seeded construction for both training stages.
                pl.seed_everything(config.seed, workers=True)
                inner = build_model(config, data.num_models, alpha, k)
                log_dir = (output_dir / "lightning_logs" / config.variant / f"fold_{f}"
                           / f"alpha_{alpha}_k_{k}")
                size = image_size(inner)
                optimization = data.subset(split.optimization_categories, image_size=size)
                validation = data.subset([split.val_category], image_size=size)
                inner_trainer, best = fit_model(inner, optimization, config, output_dir,
                                               config.max_epochs, validation, log_dir=log_dir / "inner")
                if best is None or best.n_epochs < 1:
                    raise RuntimeError("No valid inner validation epoch was recorded")
                n_epochs, best_loss = best.n_epochs, best.best_val_loss
                del inner_trainer, inner, best
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pl.seed_everything(config.seed, workers=True)
                outer = build_model(config, data.num_models, alpha, k)
                training = data.subset((*split.optimization_categories, split.val_category), image_size=size)
                outer_trainer, _ = fit_model(outer, training, config, output_dir, n_epochs,
                                             log_dir=log_dir / "outer")
                # Documented choice: original outer images mirror the unaugmented test set.
                evaluation = data.subset([split.eval_category], originals_only=True, image_size=size)
                result = evaluate_model(outer, evaluation, config, output_dir).iloc[0]
                row = dict(zip(CV_COLUMNS, [config.variant, f, split.eval_category,
                           split.val_category, alpha, k, n_epochs, best_loss,
                           float(result.ndcg_3), float(result.ndcg_5)]))
                new_file = not path.exists()
                with path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CV_COLUMNS)
                    if new_file:
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                complete.add(key)
                del outer_trainer, outer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return path
