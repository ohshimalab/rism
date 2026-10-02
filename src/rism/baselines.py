"""Sec. 5.2 baselines; SimSiam training defaults follow Pham et al. (2025).

Reference rows follow ImageDataset's sorted Image Path order, candidates follow
ascending Model ID. Signed WA weights use every reference, without clipping.
"""
import hashlib
import json
import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from transformers import ViTConfig, ViTModel

from .config import RunConfig
from .data import ImageDataset
from .evaluation import category_summary
from .metrics import ndcg
from .training import image_size, loader, make_trainer


def _lightly():
    # AR and saved-backbone extraction work without the optional training extra.
    try:
        from lightly.loss import NegativeCosineSimilarity
        from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
        from lightly.transforms import SimSiamTransform
    except ImportError as exc:
        raise ImportError("SimSiam training requires lightly; pip install rism[baselines]") from exc
    return NegativeCosineSimilarity, SimSiamProjectionHead, SimSiamPredictionHead, SimSiamTransform


def simsiam_loss(p0, z0, p1, z1, criterion=None):
    """Symmetric prediction-to-target loss, stopping only the target branches."""
    if criterion is None:
        criterion = _lightly()[0]()
    return 0.5 * (criterion(p0, z1.detach()) + criterion(p1, z0.detach()))


class SimSiam(pl.LightningModule):
    def __init__(self, backbone: ViTModel, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        loss, projection, prediction, _ = _lightly()
        self.vit = backbone
        dim = backbone.config.hidden_size
        self.projection = projection(dim, 2048, dim)
        self.prediction = prediction(dim, 512, dim)
        self.criterion = loss()
        self.lr, self.weight_decay = lr, weight_decay

    def forward(self, images):
        z = self.projection(self.vit(pixel_values=images).last_hidden_state[:, 0])
        return z.detach(), self.prediction(z)

    def training_step(self, batch, batch_idx):
        x0, x1 = batch["image"]
        z0, p0 = self(x0)
        z1, p1 = self(x1)
        loss = simsiam_loss(p0, z0, p1, z1, self.criterion)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(x0))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class _AllImageBatches(BatchSampler):
    """Merge a final singleton into the previous batch for SimSiam BatchNorm."""
    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    return
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = super().__len__()
        return count - int(len(self.sampler) % self.batch_size == 1 and count > 1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_simsiam(config: RunConfig, data_dir: Path | str, output_dir: Path | str,
                  *, epochs: int = 30, weight_decay: float = 1e-4) -> Path:
    _, _, _, transform = _lightly()
    if epochs < 1 or config.batch_size < 2 or weight_decay < 0:
        raise ValueError("SimSiam requires epochs >= 1, batch_size >= 2 and weight_decay >= 0")
    pl.seed_everything(config.seed, workers=True)
    data = ImageDataset(data_dir, "train")
    if len(data) < 2:
        raise ValueError("SimSiam BatchNorm requires at least two training images")
    model = SimSiam(ViTModel.from_pretrained(config.vit_name), config.lr, weight_decay)
    size = image_size(model)
    data.transform = transform(input_size=size, normalize={
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)})
    batches = _AllImageBatches(RandomSampler(data), config.batch_size, drop_last=False)
    train_loader = DataLoader(data, batch_sampler=batches, num_workers=config.num_workers,
                              persistent_workers=config.num_workers > 0)
    output_dir = Path(output_dir)
    trainer = make_trainer(config, output_dir, epochs=epochs,
                           log_dir=output_dir / "lightning_logs" / "simsiam")
    model.train()
    trainer.fit(model, train_dataloaders=train_loader)
    metadata = {"method": "simsiam", "config": config.to_dict(), "n_epochs": epochs,
                "weight_decay": weight_decay, "model_ids": data.model_ids,
                "training_images": "original + augmented", "num_training_images": len(data),
                "vit_config": model.vit.config.to_dict(),
                "preprocessing": {"image_size": size, "features": "CLS", "l2_normalized": True},
                "heads": {"projection": [model.vit.config.hidden_size, 2048, model.vit.config.hidden_size],
                          "prediction": [model.vit.config.hidden_size, 512, model.vit.config.hidden_size]},
                "defaults_source": "Pham et al. 2025 ISMR reference (paper silent)",
                "train_csv_sha256": _sha256(data.root / "train.csv")}
    # Keep checkpoint metadata and its JSON sidecar identical (config keys
    # such as id2label may otherwise retain integer dictionary keys).
    metadata = json.loads(json.dumps(metadata))
    path = output_dir / "simsiam_backbone.pth"
    if trainer.is_global_zero:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.vit.state_dict().items()},
                    "vit_config": metadata["vit_config"], "metadata": metadata}, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load_backbone(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    vit = ViTModel(ViTConfig.from_dict(saved["vit_config"]))
    vit.load_state_dict(saved["state_dict"], strict=True)
    return vit.eval(), saved["metadata"]


def nearest_image(queries, references, ap):
    similarities = np.asarray(queries) @ np.asarray(references).T
    # NumPy argmax returns the first reference in a tie.
    return np.nan_to_num(np.asarray(ap), nan=0.0)[similarities.argmax(axis=1)]


def weighted_average(queries, references, ap):
    similarities = np.asarray(queries, dtype=np.float64) @ np.asarray(references, dtype=np.float64).T
    denominator = similarities.sum(axis=1, keepdims=True)
    if np.any(np.isclose(denominator, 0.0, rtol=0.0, atol=1e-8)):
        raise ValueError("WA cosine-weight denominator is approximately zero")
    return similarities @ np.nan_to_num(np.asarray(ap), nan=0.0) / denominator


def average_ranking(ap):
    ranks = pd.DataFrame(np.nan_to_num(np.asarray(ap), nan=0.0)).rank(
        axis=1, method="min", ascending=False).to_numpy()
    return -ranks.mean(axis=0)


def _rank_diagnostic(data):
    frame = pd.read_csv(data.root / "train.csv")
    if "AP Rank" not in frame:
        logging.info("train: CSV AP Rank absent; rank diagnostic unavailable")
        return
    computed = frame.groupby("Image Path", sort=False)["AP"].transform(
        lambda ap: ap.fillna(0).rank(method="min", ascending=False))
    logging.info("train: %d rows disagree with CSV AP Rank", int((computed != frame["AP Rank"]).sum()))


class _Features(pl.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def predict_step(self, batch, batch_idx):
        features = self.vit(pixel_values=batch["image"]).last_hidden_state[:, 0]
        if not torch.isfinite(features).all() or (features.norm(dim=1) <= 1e-12).any():
            raise ValueError("Backbone produced nonfinite or zero CLS features")
        return F.normalize(features, dim=1).cpu()


def _feature_identity(checkpoint, train, test):
    # CSVs bind row order and labels; cheap image stat signatures invalidate edits
    # without reading all JPEG bytes. These signatures are cache identity only.
    return {"version": 1, "checkpoint_sha256": _sha256(checkpoint),
            "csv_sha256": [_sha256(d.root / f"{split}.csv") for d, split in ((train, "train"), (test, "test"))],
            "images": [[str((d.root / s["image_path"]).resolve()),
                        (d.root / s["image_path"]).stat().st_size,
                        (d.root / s["image_path"]).stat().st_mtime_ns]
                       for d in (train, test) for s in d.samples]}


def _extract_features(checkpoint, train, test, config, output_dir):
    identity = json.dumps(_feature_identity(checkpoint, train, test), sort_keys=True)
    path = output_dir / "simsiam_features.npz"
    vit, metadata = load_backbone(checkpoint)
    if metadata["model_ids"] != train.model_ids:
        raise ValueError("Reference Model IDs differ from SimSiam training")
    if metadata["train_csv_sha256"] != _sha256(train.root / "train.csv"):
        raise ValueError("Training CSV differs from SimSiam checkpoint")
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["identity"].item()) == identity:
                refs, queries = cached["references"], cached["queries"]
                if (refs.shape == (len(train), vit.config.hidden_size)
                        and queries.shape == (len(test), vit.config.hidden_size)
                        and all(np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
                                for x in (refs, queries))):
                    logging.info("Using cached SimSiam CLS features: %s", path)
                    return refs, queries
    model = _Features(vit)
    trainer = make_trainer(config, output_dir, epochs=1, prediction=True)
    arrays = []
    for data in (train, test):
        batches = trainer.predict(model, dataloaders=loader(data.subset(image_size=image_size(model)), config))
        arrays.append(torch.cat(batches).numpy())
    refs, queries = arrays
    np.savez_compressed(path, references=refs, queries=queries, identity=np.array(identity),
                        reference_paths=np.array([s["image_path"] for s in train.samples]),
                        query_paths=np.array([s["image_path"] for s in test.samples]),
                        model_ids=np.array(train.model_ids))
    return refs, queries


def baselines(data_dir: Path | str, output_dir: Path | str, config: RunConfig, *,
              methods=("wa", "ni", "ar"), simsiam_checkpoint: Path | str | None = None):
    methods = list(dict.fromkeys(methods))
    if not methods or set(methods) - {"wa", "ni", "ar"}:
        raise ValueError("Choose at least one of wa, ni, ar")
    if {"wa", "ni"} & set(methods) and simsiam_checkpoint is None:
        raise ValueError("--simsiam-checkpoint is required for wa/ni")
    train = ImageDataset(data_dir, "train")
    test = ImageDataset(data_dir, "test", originals_only=True)
    if train.model_ids != test.model_ids:
        raise ValueError("Test Model IDs differ from training")
    # Preserve CSV precision for exact AP ties; neural supervision tensors are
    # float32 and can collapse distinct AP values into artificial ties.
    frame = pd.read_csv(train.root / "train.csv")
    ap = frame.pivot(index="Image Path", columns="Model ID", values="AP").reindex(
        index=[s["image_path"] for s in train.samples], columns=train.model_ids).fillna(0).to_numpy()
    relevance = torch.stack([s["relevance"] for s in test.samples])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if {"wa", "ni"} & set(methods):
        refs, queries = _extract_features(simsiam_checkpoint, train, test, config, output_dir)
    paths = []
    for method in methods:
        if method == "ar":
            _rank_diagnostic(train)
            scores = np.broadcast_to(average_ranking(ap), (len(test), train.num_models)).copy()
        else:
            scoring = weighted_average if method == "wa" else nearest_image
            scores = np.concatenate([scoring(queries[start:start + config.batch_size], refs, ap)
                                     for start in range(0, len(test), config.batch_size)])
        if not np.isfinite(scores).all():
            raise ValueError(f"{method} produced nonfinite scores")
        values = {k: ndcg(torch.from_numpy(scores), relevance, k).tolist() for k in (3, 5)}
        rows = [{"category": s["category"], "ndcg_3": values[3][i], "ndcg_5": values[5][i]}
                for i, s in enumerate(test.samples)]
        categories = pd.DataFrame(rows).groupby("category", as_index=False, sort=True).mean()
        path = output_dir / f"{method}_test_results.csv"
        category_summary(categories).to_csv(path, index=False)
        paths.append(path)
    return paths
