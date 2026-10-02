"""Sec. 6.4 retrieval timing; synthetic candidate embeddings are timing-only.

Each query times one ViT encoding, chunked scoring, and descending argsort.
Decoding and preprocessing are excluded: the fp32 image is already on device.
Default input/embedding sizes are 224 and 768; checkpoint geometry is respected
for tiny offline tests. Pools are generated once per size, outside timed regions.
"""
import math
from pathlib import Path
from typing import Iterable
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_VIT, RunConfig
from .plotting import STYLE, plt, save_figure
from .training import build_model, image_size, load_checkpoint


def plot_latency(csv_path: Path | str, *, chunk_size: int) -> None:
    path = Path(csv_path)
    frame = pd.read_csv(path).sort_values("pool_size")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
        ax.set_xscale("log", base=2)
        ax.errorbar(frame.pool_size, frame.mean_ms, yerr=frame.std_ms, fmt="o-", capsize=3,
                    color="#0072B2", label="Mean ± std")
        lo, hi = min(8, frame.pool_size.min()) / 1.25, max(8192, frame.pool_size.max()) * 1.25
        ax.set_xlim(lo, hi)
        if chunk_size >= lo:
            ax.axvspan(lo, min(chunk_size, hi), color="0.93", zorder=0)
            ax.text(0.02, 0.96, f"Single batch ≤ {chunk_size}", transform=ax.transAxes, va="top")
        if chunk_size < hi:
            ax.axvspan(max(chunk_size, lo), hi, color="#E69F00", alpha=0.10, zorder=0)
            ax.text(0.98, 0.96, "Multiple batches", transform=ax.transAxes, ha="right", va="top")
        single = frame[frame.pool_size <= chunk_size]
        if not single.empty:
            point = single.iloc[-1]
            ax.plot(point.pool_size, point.mean_ms, "o", ms=9, mfc="none", mec="black",
                    label="Largest single-batch pool")
        ax.set_xticks([2**i for i in range(3, 14)], [f"$2^{{{i}}}$" for i in range(3, 14)])
        ax.set_xlabel("Candidate pool size")
        ax.set_ylabel("Retrieval latency (ms)")
        ax.set_ylim(bottom=0, top=max(0.01, float((frame.mean_ms + frame.std_ms).max()) * 1.35))
        ax.legend(loc="best", fontsize=8)
        save_figure(fig, path.with_suffix(""))


def latency(output_dir: Path | str, *, variant: str = "ours", checkpoint: Path | str | None = None,
            vit_name: str = DEFAULT_VIT, pool_sizes: Iterable[int] | None = None,
            chunk_size: int = 2048, warmup: int = 10, repeats: int = 100,
            device: str | None = None, seed: int = 42, nhead: int = 8) -> Path:
    sizes = list(pool_sizes if pool_sizes is not None else [2**i for i in range(3, 14)])
    if not sizes or any(n < 1 for n in sizes) or chunk_size < 1 or warmup < 0 or repeats < 1:
        raise ValueError("Require positive pool/chunk sizes and repeats, and nonnegative warmup")
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Latency measurement supports cpu or cuda")
    if checkpoint is None:
        model = build_model(RunConfig(variant=variant, vit_name=vit_name, nhead=nhead), 1, 0.5, 1)
    else:
        model, _ = load_checkpoint(checkpoint)
        if model.variant != variant:
            raise ValueError("Requested variant differs from checkpoint")
    model.to(device=device, dtype=torch.float32).eval()
    image = torch.rand(1, 3, image_size(model), image_size(model), device=device)
    is_cuda = device.type == "cuda"
    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)
    rows = []
    with torch.inference_mode():
        for size in sizes:
            pool = torch.rand(size, model.model_dim, device=device, dtype=torch.float32)
            def query():
                features = model.encode_image(image)
                scores = model.score_features(features, model_embeddings=pool, chunk_size=chunk_size)
                return torch.argsort(scores, descending=True)
            for _ in range(warmup):
                query()
            synchronize()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                synchronize()
                start = perf_counter()
                query()
                synchronize()
                samples.append((perf_counter() - start) * 1000)
            rows.append({"pool_size": size, "n_batches": math.ceil(size / chunk_size),
                         "mean_ms": np.mean(samples), "std_ms": np.std(samples, ddof=0),
                         "median_ms": np.median(samples), "p95_ms": np.percentile(samples, 95),
                         "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None,
                         "device_name": torch.cuda.get_device_name(device) if is_cuda else "cpu",
                         "torch_version": str(torch.__version__)})
            del pool
    path = Path(output_dir) / f"latency_{variant}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    plot_latency(path, chunk_size=chunk_size)
    return path
