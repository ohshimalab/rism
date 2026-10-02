"""Deterministic selection from outer-fold ranking quality (Sec. 5.1)."""

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (DEFAULT_ALPHAS, DEFAULT_K_LOSSES, VARIANTS,
                     result_provenance, validate_hparams)


def select_hparams(csv_path: Path | str | Iterable[Path | str], output_path: Path | str, *,
                   variant: str = "ours", allow_incomplete: bool = False,
                   expected_folds: int = 20) -> dict[str, Any]:
    if expected_folds < 1:
        raise ValueError("expected_folds must be positive")
    paths = [Path(csv_path)] if isinstance(csv_path, (str, Path)) else [Path(p) for p in csv_path]
    if not paths:
        raise ValueError("Supply at least one CV CSV")
    provenance = None
    frames = []
    for path in paths:
        sidecar = path.with_suffix(".run.json")
        if sidecar.exists():
            current = result_provenance(json.loads(sidecar.read_text()))
            if provenance is not None and current != provenance:
                raise ValueError("CV provenance differs across source CSVs")
            provenance = current
        source = pd.read_csv(path)
        required = {"variant", "fold", "alpha", "k_loss", "n_epochs", "ndcg_3", "ndcg_5"}
        if required - set(source.columns):
            raise ValueError(f"Invalid CV CSV columns: {path}")
        frames.append(source)
    frame = pd.concat(frames, ignore_index=True)
    frame["alpha"] = frame.alpha.round(10)
    if frame.duplicated(["variant", "fold", "alpha", "k_loss"]).any():
        raise ValueError("Duplicate CV fold/configuration rows")
    required = {"variant", "fold", "alpha", "k_loss", "n_epochs", "ndcg_3", "ndcg_5"}
    if required - set(frame.columns) or variant not in VARIANTS:
        raise ValueError("Invalid variant or CV CSV columns")
    frame = frame[frame.variant == variant].copy()
    if frame.empty:
        raise ValueError(f"No CV rows for {variant}")
    numeric = ["fold", "alpha", "k_loss", "n_epochs", "ndcg_3", "ndcg_5"]
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("CV results must be finite")
    for row in frame.itertuples():
        if any(value != int(value) for value in (row.fold, row.k_loss, row.n_epochs)):
            raise ValueError("Fold, k_loss and n_epochs must be integers")
        if not 0 <= row.fold < expected_folds or not 0 <= row.ndcg_3 <= 1.000001 or not 0 <= row.ndcg_5 <= 1.000001:
            raise ValueError("Invalid CV fold or ranking score")
        validate_hparams(row.alpha, int(row.k_loss), int(row.n_epochs))
    if not allow_incomplete:
        present = set(zip(frame.alpha, frame.k_loss))
        missing = [(round(alpha, 10), k) for alpha in DEFAULT_ALPHAS for k in DEFAULT_K_LOSSES
                   if (round(alpha, 10), k) not in present]
        if missing:
            raise ValueError(f"Missing paper grid (alpha, k_loss) pairs: {missing}; use --allow-incomplete")
    rows = []
    for (alpha, k), group in frame.groupby(["alpha", "k_loss"], sort=True):
        if not allow_incomplete and set(group.fold) != set(range(expected_folds)):
            raise ValueError(f"alpha={alpha}, k_loss={k} requires all {expected_folds} folds; use --allow-incomplete")
        mean_epochs = float(group.n_epochs.mean())
        rows.append({"alpha": float(alpha), "k_loss": int(k),
                     "mean_ndcg_3": float(group.ndcg_3.mean()),
                     "mean_ndcg_5": float(group.ndcg_5.mean()),
                     "mean_epochs": mean_epochs, "n_epochs": max(1, round(mean_epochs)),
                     "n_folds": len(group)})
    def order(row: dict[str, Any]) -> tuple[float, float, float, int]:
        return (-row["mean_ndcg_3"], -row["mean_ndcg_5"], row["alpha"], row["k_loss"])
    ranked = sorted(rows, key=order)
    per_alpha = [min((r for r in rows if r["alpha"] == alpha), key=order)
                 for alpha in sorted({r["alpha"] for r in rows})]
    result = {"variant": variant, **ranked[0], "per_alpha": per_alpha,
              "configs": ranked, "allow_incomplete": allow_incomplete, "expected_folds": expected_folds,
              "source_csvs": [str(p.resolve()) for p in paths]}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def resolve_hparams(hparams: Path | str | None, alpha: float | None, k_loss: int | None,
                    epochs: int | None, variant: str) -> tuple[float, int, int]:
    """Require a complete explicit configuration or a compatible selection JSON."""
    if hparams is not None:
        selected = json.loads(Path(hparams).read_text())
        if selected["variant"] != variant:
            raise ValueError("Selection JSON variant differs from requested variant")
        values = (float(selected["alpha"]), int(selected["k_loss"]), int(selected["n_epochs"]))
        for explicit, chosen in zip((alpha, k_loss, epochs), values):
            if explicit is not None and explicit != chosen:
                raise ValueError("Explicit hyperparameters conflict with the selection JSON")
    else:
        if alpha is None or k_loss is None or epochs is None:
            raise ValueError("Supply --hparams or all of --alpha, --k-loss and --epochs")
        values = (alpha, k_loss, epochs)
    validate_hparams(*values)
    return values
