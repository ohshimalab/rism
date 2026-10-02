"""Fresh model construction, Lightning training, and portable checkpoints."""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import ViTConfig, ViTModel

from .config import RunConfig, validate_hparams
from .data import ImageDataset
from .models import Retriever


class BestEpoch(Callback):
    """Track the first epoch attaining the lowest inner validation loss."""

    def __init__(self) -> None:
        self.n_epochs = 0
        self.best_val_loss = math.inf

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        loss = float(trainer.callback_metrics["val_loss"])
        if not math.isfinite(loss):
            raise ValueError("Nonfinite validation loss")
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.n_epochs = trainer.current_epoch + 1


def build_model(config: RunConfig, num_models: int, alpha: float, k_loss: int) -> Retriever:
    """Every invocation loads a fresh pretrained ViT (Sec. 5.1)."""
    vit = ViTModel.from_pretrained(config.vit_name)
    return Retriever(vit, num_models, config.variant, alpha=alpha, k_loss=k_loss,
                     lr=config.lr, nhead=config.nhead)


def image_size(model: Retriever) -> int:
    size = model.vit.config.image_size
    if not isinstance(size, int):
        if size[0] != size[1]:
            raise ValueError("This preprocessing requires a square ViT input")
        size = size[0]
    return size


def loader(dataset: ImageDataset, config: RunConfig, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, persistent_workers=config.num_workers > 0)


def make_trainer(config: RunConfig, output_dir: Path | str, *, epochs: int,
                 callbacks: list[Callback] | None = None,
                 log_dir: Path | str | None = None, prediction: bool = False) -> pl.Trainer:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    trainer = pl.Trainer(
                      max_epochs=epochs, accelerator=config.accelerator, devices=config.devices,
                      callbacks=callbacks or [], default_root_dir=str(output_dir),
                      deterministic=True, enable_checkpointing=False,
                      logger=False if prediction else CSVLogger(
                          str(log_dir or Path(output_dir) / "lightning_logs" / config.variant / "final"),
                          name="", version=""),
                      enable_progress_bar=False, enable_model_summary=False,
                      num_sanity_val_steps=0, log_every_n_steps=1)
    # Keep result writing and evaluation complete under a single process. DDP prediction
    # does not return a combined result slate; reject it rather than report partial scores.
    if trainer.num_devices != 1:
        raise ValueError("These workflows require one device; use --devices 1 or a single GPU index")
    return trainer


def fit_model(model: Retriever, train_data: ImageDataset, config: RunConfig,
              output_dir: Path | str, epochs: int, val_data: ImageDataset | None = None,
              *, log_dir: Path | str | None = None
              ) -> tuple[pl.Trainer, BestEpoch | None]:
    model.train()
    tracker = BestEpoch() if val_data is not None else None
    callbacks = [] if tracker is None else [tracker, EarlyStopping(
        monitor="val_loss", patience=5, mode="min", min_delta=0.0,
        check_on_train_epoch_end=False, check_finite=True)]
    trainer = make_trainer(config, output_dir, epochs=epochs, callbacks=callbacks, log_dir=log_dir)
    trainer.fit(model, train_dataloaders=loader(train_data, config, shuffle=True),
                val_dataloaders=loader(val_data, config) if val_data is not None else None)
    return trainer, tracker


def train(config: RunConfig, data_dir: Path | str, output_dir: Path | str, *,
          alpha: float, k_loss: int, epochs: int) -> Path:
    """Sec. 5.1 final training on all images in all training categories."""
    validate_hparams(alpha, k_loss, epochs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(config.seed, workers=True)
    data = ImageDataset(data_dir, "train")
    model = build_model(config, data.num_models, alpha, k_loss)
    data = data.subset(image_size=image_size(model))
    trainer, _ = fit_model(model, data, config, output_dir, epochs)
    checkpoint = output_dir / f"{config.variant}_checkpoint.pth"
    metadata = {**config.to_dict(), "alpha": alpha, "k_loss": k_loss, "n_epochs": epochs,
                "model_ids": data.model_ids, "num_models": data.num_models,
                "vit_config": model.vit.config.to_dict(),
                "model_kwargs": dict(model.hparams),
                "preprocessing": {"image_size": image_size(model), "online_augmentation": False},
                "weight_decay": 1e-4,
                "train_csv_sha256": hashlib.sha256(
                    (Path(data_dir) / "train/train.csv").read_bytes()).hexdigest()}
    if trainer.is_global_zero:
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"state_dict": state, "metadata": metadata}, checkpoint)
        checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return checkpoint


def load_checkpoint(path: Path | str) -> tuple[Retriever, dict[str, Any]]:
    """Restore all weights locally; never fetch pretrained weights for evaluation."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    vit = ViTModel(ViTConfig.from_dict(metadata["vit_config"]))
    model = Retriever(vit, **metadata["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, metadata
