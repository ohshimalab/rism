"""CSV supervision and deterministic image preprocessing."""

import copy
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def image_transform(image_size: int = 224) -> transforms.Compose:
    """Resize, tensor conversion and ImageNet normalization; no online augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def relevance_labels(ap: np.ndarray) -> np.ndarray:
    """Recompute Sec. 3.3 grades along the last (candidate) dimension."""
    values = np.where(np.isnan(ap), 0.0, np.asarray(ap, dtype=np.float64))
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("AP must be NaN or a finite value in [0, 1]")
    maximum = values.max(axis=-1, keepdims=True)
    relative = np.divide(values, maximum, out=np.zeros_like(values), where=maximum > 0)
    return np.where(relative >= 0.9, 2, np.where(relative >= 0.8, 1, 0)).astype(np.int64)


class ImageDataset(Dataset):
    """One sample per image, with a consistent ascending Model ID candidate axis."""

    def __init__(self, data_dir: Path | str, split: str = "train", *,
                 categories: Iterable[str] | None = None, originals_only: bool = False,
                 image_size: int = 224) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.root = Path(data_dir) / split
        frame = pd.read_csv(self.root / f"{split}.csv")
        required = {"Category", "File Name", "Image Type", "Model ID", "AP",
                    "Image Path", "Relevance Score"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        if frame.empty or frame[list(required - {"AP", "Relevance Score"})].isna().any().any():
            raise ValueError("Empty dataset or missing sample metadata")
        if not frame["Image Type"].isin(["original", "augmented"]).all():
            raise ValueError("Unknown Image Type")
        ids = frame["Model ID"].to_numpy()
        if not np.equal(ids, ids.astype(int)).all():
            raise ValueError("Model IDs must be integers")
        self.model_ids = sorted(frame["Model ID"].astype(int).unique().tolist())
        if self.model_ids != list(range(len(self.model_ids))):
            raise ValueError("Model IDs must be contiguous, starting at zero")
        self.num_models = len(self.model_ids)
        self.samples: list[dict[str, Any]] = []
        disagreements = 0
        # Sec. 3.3: CSV relevance scores are diagnostic only, never supervision.
        for path, group in frame.groupby("Image Path", sort=True):
            group = group.sort_values("Model ID", kind="stable")
            if group["Model ID"].tolist() != self.model_ids:
                raise ValueError(f"Missing/duplicate candidates for {path}")
            for column in ("Category", "File Name", "Image Type"):
                if group[column].nunique() != 1:
                    raise ValueError(f"Inconsistent {column} for {path}")
            ap = group["AP"].to_numpy(dtype=np.float64)
            labels = relevance_labels(ap)
            disagreements += int((labels != group["Relevance Score"].to_numpy()).sum())
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Image Path must stay relative to the split: {path}")
            self.samples.append({
                "image_path": str(path), "category": group["Category"].iloc[0],
                "file_name": group["File Name"].iloc[0],
                "image_type": group["Image Type"].iloc[0],
                "ap": torch.tensor(np.nan_to_num(ap, nan=0.0), dtype=torch.float32),
                "relevance": torch.tensor(labels, dtype=torch.long),
            })
        logging.info("%s: %d rows disagree with CSV Relevance Score", split, disagreements)
        if categories is not None:
            selected = set(categories)
            self.samples = [s for s in self.samples if s["category"] in selected]
        if originals_only:
            self.samples = [s for s in self.samples if s["image_type"] == "original"]
        if not self.samples:
            raise ValueError("Dataset selection contains no images")
        self.transform = image_transform(image_size)

    def subset(self, categories: Iterable[str] | None = None, *,
               originals_only: bool = False, image_size: int | None = None) -> "ImageDataset":
        """Reuse validated CSV supervision without rereading it for every CV run."""
        result = copy.copy(self)
        selected = set(categories) if categories is not None else set(self.categories)
        result.samples = [s for s in self.samples if s["category"] in selected
                          and (not originals_only or s["image_type"] == "original")]
        if not result.samples:
            raise ValueError("Dataset selection contains no images")
        if image_size is not None:
            result.transform = image_transform(image_size)
        return result

    @property
    def categories(self) -> list[str]:
        return sorted({s["category"] for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(self.root / sample["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {**sample, "image": tensor}
