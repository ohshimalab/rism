"""Shared restrained style and paired vector/raster exports (standalone figures)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.3}


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
