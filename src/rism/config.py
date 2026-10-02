"""Execution defaults and serializable run configuration."""

from dataclasses import asdict, dataclass
from typing import Any

VARIANTS = ("ours", "global_te", "local_mlp", "prior")
DEFAULT_ALPHAS = tuple(i / 10 for i in range(11))
DEFAULT_K_LOSSES = (5, 20, 51)
DEFAULT_VIT = "google/vit-base-patch16-224-in21k"


@dataclass(frozen=True)
class RunConfig:
    variant: str = "ours"
    seed: int = 42
    batch_size: int = 16  # Not specified in the paper.
    num_workers: int = 4
    lr: float = 1e-3  # Sec. 5.1.
    max_epochs: int = 100  # Safety cap, not specified in the paper.
    vit_name: str = DEFAULT_VIT
    accelerator: str = "auto"
    devices: str | int | list[int] = "auto"
    nhead: int = 8

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose from {VARIANTS}")
        if min(self.batch_size, self.max_epochs, self.nhead) < 1:
            raise ValueError("batch_size, max_epochs and nhead must be positive")
        if self.num_workers < 0 or self.lr <= 0:
            raise ValueError("num_workers must be nonnegative and lr positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def result_config(config: RunConfig | dict[str, Any]) -> dict[str, Any]:
    """Resume identity excludes only worker/device execution settings."""
    values = config.to_dict() if isinstance(config, RunConfig) else config
    return {k: v for k, v in values.items() if k not in {"num_workers", "accelerator", "devices"}}


def result_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Normalize CV provenance for resume and merging across workers/devices."""
    return {**provenance, "config": result_config(provenance["config"])}


def validate_hparams(alpha: float, k_loss: int, epochs: int = 1) -> None:
    if not 0 <= alpha <= 1 or k_loss < 1 or epochs < 1:
        raise ValueError("Require 0 <= alpha <= 1, k_loss >= 1 and epochs >= 1")
