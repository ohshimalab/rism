"""Sec. 6.2 model-token maps, using original test images.

CSV Model IDs and --model-ids are zero-based. Figures and PNG filenames use
one-based Model <ID+1>, matching rism_old and the paper (Model 21 = CSV ID 20).
Raw NPZ weights retain ALL candidates, without normalization or interpolation.
The paper's ViT has 14x14 patches; smaller square grids are supported for tests.
"""
from pathlib import Path
from typing import Iterable
import math

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .config import RunConfig
from .data import ImageDataset
from .plotting import STYLE, plt, save_figure
from .training import image_size, load_checkpoint, make_trainer


def attention_maps(checkpoint: Path | str, data_dir: Path | str, output_dir: Path | str,
                   *, model_ids: Iterable[int] | None = None,
                   categories: Iterable[str] | None = None, batch_size: int = 16,
                   accelerator: str = "auto", devices: str | int | list[int] = "auto") -> Path:
    model, metadata = load_checkpoint(checkpoint)
    if model.variant not in ("ours", "global_te"):
        raise ValueError("Attention maps require an ours or global_te checkpoint; MLP variants have no attention")
    if model.variant == "global_te":
        raise ValueError("global_te has only one image token; spatial attention maps require ours")
    side = math.isqrt(model.token_count)
    if side * side != model.token_count:
        raise ValueError("Attention maps require a square patch grid")
    data = ImageDataset(data_dir, "test", originals_only=True, image_size=image_size(model))
    if data.model_ids != metadata["model_ids"]:
        raise ValueError("Test Model IDs differ from checkpoint")
    ids = list(data.model_ids if model_ids is None else model_ids)
    selected_categories = list(data.categories[:4] if categories is None else categories)
    if not ids or len(set(ids)) != len(ids) or any(i not in data.model_ids for i in ids):
        raise ValueError("Select distinct zero-based CSV Model IDs in the candidate range")
    if (not selected_categories or len(set(selected_categories)) != len(selected_categories)
            or any(c not in data.categories for c in selected_categories)):
        raise ValueError("Select distinct categories present in the original test images")
    config = RunConfig(batch_size=batch_size, num_workers=0, accelerator=accelerator, devices=devices)
    trainer = make_trainer(config, output_dir, epochs=1, prediction=True)
    model.to(trainer.strategy.root_device).eval()
    output_dir = Path(output_dir)
    maps_dir = output_dir / "attention_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    grid_indices = [i for c in selected_categories for i, s in enumerate(data.samples) if s["category"] == c]
    overlays = {}
    weights = []
    filenames = set()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            images = torch.stack([data[i]["image"] for i in range(start, min(start + batch_size, len(data)))])
            raw = model.model_token_attention(images.to(model.device)).cpu().reshape(-1, model.num_models, side, side)
            weights.append(raw.numpy())
            for offset, maps in enumerate(raw):
                index = start + offset
                sample = data.samples[index]
                with Image.open(data.root / sample["image_path"]) as source:
                    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
                for candidate in ids:
                    attn = maps[candidate]
                    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                    up = F.interpolate(attn[None, None], size=rgb.shape[:2], mode="bilinear",
                                       align_corners=False)[0, 0].numpy()
                    overlay = np.clip(0.45 * plt.get_cmap("inferno")(up)[..., :3] + 0.55 * rgb, 0, 1)
                    name = f"{sample['category']}_{Path(sample['file_name']).stem}_model{candidate + 1}.png"
                    if Path(name).name != name or name in filenames:
                        raise ValueError("Attention PNG names must be unique and contain no path separators")
                    filenames.add(name)
                    Image.fromarray((overlay * 255).astype(np.uint8)).save(maps_dir / name)
                    if index in grid_indices:
                        overlays[index, candidate] = overlay
    np.savez_compressed(maps_dir / "attention_weights.npz", attention_weights=np.concatenate(weights),
                        image_names=np.array([s["file_name"] for s in data.samples]),
                        image_paths=np.array([s["image_path"] for s in data.samples]),
                        categories=np.array([s["category"] for s in data.samples]),
                        model_ids=np.array(data.model_ids))
    with plt.rc_context(STYLE):
        # A common row height and aspect-proportional column widths preserve
        # original image geometry, including grids with mixed aspect ratios.
        ratios = [overlays[index, ids[0]].shape[1] / overlays[index, ids[0]].shape[0]
                  for index in grid_indices]
        row_height = 1.5
        fig, axes = plt.subplots(len(ids), len(grid_indices), squeeze=False,
                                 figsize=(row_height * sum(ratios) + 0.7,
                                          row_height * len(ids) + 0.5),
                                 gridspec_kw={"width_ratios": ratios})
        fig.subplots_adjust(left=0.6 / fig.get_figwidth(), right=0.99,
                            bottom=0.1 / fig.get_figheight(),
                            top=1 - 0.35 / fig.get_figheight(), wspace=0.02, hspace=0.02)
        for row, candidate in enumerate(ids):
            for col, index in enumerate(grid_indices):
                ax = axes[row, col]
                ax.imshow(overlays[index, candidate])
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(data.samples[index]["category"])
                if col == 0:
                    ax.set_ylabel(f"Model {candidate + 1}")
        save_figure(fig, output_dir / "attention_grid")
    return maps_dir
