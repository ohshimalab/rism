import os
from pathlib import Path

# The entire suite is offline, including accidental Hugging Face requests.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers import ViTConfig, ViTModel


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_vit():
    return ViTModel(ViTConfig(image_size=32, patch_size=16, hidden_size=32,
                            num_hidden_layers=1, num_attention_heads=2, intermediate_size=64))


@pytest.fixture
def local_vit(tmp_path: Path, tiny_vit):
    path = tmp_path / "tiny_vit"
    tiny_vit.save_pretrained(path)
    return path


@pytest.fixture
def synthetic_data(tmp_path: Path):
    root = tmp_path / "data"
    for split, count in (("train", 4), ("test", 2)):
        rows = []
        for category_idx in range(count):
            category = f"{split}-category-{category_idx}"
            for image_idx in range(3 if split == "train" else 1):
                image_type = "augmented" if image_idx == 2 else "original"
                filename = f"image-{image_idx}.png"
                relative = f"{category}/images/{filename}"
                path = root / split / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (30 * category_idx, 40 * image_idx, 80)).save(path)
                values = np.array([1.0, 0.9, 0.8, 0.3, np.nan])
                if category_idx == 1 and image_idx == 0:
                    values = np.array([0., 0., 0., 0., np.nan])
                for model_id, ap in enumerate(values):
                    rows.append({"Category": category, "File Name": filename, "Image Type": image_type,
                                 "Model ID": model_id, "AP": ap, "Relative AP": 0.,
                                 "AP Rank": model_id + 1, "Relevance Score": 2,
                                 "Image Path": relative})
        # Deliberately unsorted to exercise per-image grouping and Model ID sorting.
        pd.DataFrame(rows).sample(frac=1, random_state=12).to_csv(root / split / f"{split}.csv", index=False)
    return root
