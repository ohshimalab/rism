"""Typer command-line entry points."""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .config import DEFAULT_VIT, RunConfig
from .cross_validation import cross_validate
from .evaluation import evaluate as evaluate_checkpoint
from .selection import resolve_hparams, select_hparams
from .training import train as train_model

app = typer.Typer(no_args_is_help=True, help="Train and evaluate instance segmentation model retrievers.")


class Variant(str, Enum):
    ours = "ours"
    global_te = "global_te"
    local_mlp = "local_mlp"
    prior = "prior"


DataDir = Annotated[Path, typer.Option(help="Dataset root containing train/ and test/.")]
OutputDir = Annotated[Path, typer.Option(help="Destination for result files.")]
BatchSize = Annotated[int, typer.Option(min=1)]
NumWorkers = Annotated[int, typer.Option(min=0)]
MaxEpochs = Annotated[int, typer.Option(min=1, help="Inner CV training cap (not specified in paper).")]
NHead = Annotated[int, typer.Option(min=1, help="TE attention heads; must divide ViT hidden size.")]
Devices = Annotated[str, typer.Option(
    help="auto, 1, or a single GPU index with trailing comma (e.g. 0,). One device per run.")]


def run_config(variant: Variant, seed: int, batch_size: int, num_workers: int, lr: float,
               max_epochs: int, vit_name: str, accelerator: str, devices: str, nhead: int) -> RunConfig:
    parsed_devices: str | int | list[int]
    if devices == "auto":
        parsed_devices = "auto"
    elif "," in devices:
        parsed_devices = [int(value) for value in devices.split(",") if value.strip()]
    else:
        parsed_devices = int(devices)
    return RunConfig(variant.value, seed, batch_size, num_workers, lr, max_epochs,
                     vit_name, accelerator, parsed_devices, nhead)


@app.callback()
def setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@app.command("cross-validate")
def cv_command(
    data_dir: DataDir = Path("data"), variant: Variant = Variant.ours,
    output_dir: OutputDir = Path("outputs"), seed: int = 42,
    batch_size: BatchSize = 16, num_workers: NumWorkers = 4, lr: float = 1e-3,
    max_epochs: MaxEpochs = 100, vit_name: str = DEFAULT_VIT,
    accelerator: str = "auto",
    devices: Devices = "auto",
    nhead: NHead = 8,
    folds: Annotated[list[int] | None, typer.Option(help="Repeat for each 0-based fold.")] = None,
    alphas: Annotated[list[float] | None, typer.Option(help="Repeat for each alpha.")] = None,
    k_loss: Annotated[list[int] | None, typer.Option(help="Repeat for each loss truncation.")] = None,
    csv_path: Annotated[Path | None, typer.Option(help="Optional CV CSV destination.")] = None,
) -> None:
    config = run_config(variant, seed, batch_size, num_workers, lr, max_epochs,
                        vit_name, accelerator, devices, nhead)
    path = cross_validate(config, data_dir, output_dir, folds=folds, alphas=alphas,
                          k_losses=k_loss, csv_path=csv_path)
    typer.echo(str(path))


@app.command("select-hparams")
def selection_command(
    cv_csv: Annotated[list[Path], typer.Option(exists=True, dir_okay=False,
                                            help="Repeat for each source CV CSV.")],
    variant: Variant = Variant.ours, output_dir: OutputDir = Path("outputs"),
    allow_incomplete: bool = False,
    expected_folds: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    path = output_dir / f"{variant.value}_hparams.json"
    selected = select_hparams(cv_csv, path, variant=variant.value, allow_incomplete=allow_incomplete,
                             expected_folds=expected_folds)
    typer.echo(f"{path}: alpha={selected['alpha']}, k_loss={selected['k_loss']}, epochs={selected['n_epochs']}")


@app.command("train")
def train_command(
    data_dir: DataDir = Path("data"), variant: Variant = Variant.ours,
    output_dir: OutputDir = Path("outputs"), seed: int = 42,
    batch_size: BatchSize = 16, num_workers: NumWorkers = 4, lr: float = 1e-3,
    max_epochs: MaxEpochs = 100, vit_name: str = DEFAULT_VIT,
    accelerator: str = "auto",
    devices: Devices = "auto",
    nhead: NHead = 8,
    alpha: float | None = None, k_loss: int | None = None, epochs: int | None = None,
    hparams: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    config = run_config(variant, seed, batch_size, num_workers, lr, max_epochs,
                        vit_name, accelerator, devices, nhead)
    alpha, k, count = resolve_hparams(hparams, alpha, k_loss, epochs, variant.value)
    typer.echo(str(train_model(config, data_dir, output_dir, alpha=alpha, k_loss=k, epochs=count)))


@app.command("evaluate")
def evaluate_command(
    data_dir: DataDir = Path("data"), variant: Variant = Variant.ours,
    output_dir: OutputDir = Path("outputs"), seed: int = 42,
    batch_size: BatchSize = 16, num_workers: NumWorkers = 4, lr: float = 1e-3,
    max_epochs: MaxEpochs = 100, vit_name: str = DEFAULT_VIT,
    accelerator: str = "auto",
    devices: Devices = "auto",
    nhead: NHead = 8,
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    alpha: float | None = None, k_loss: int | None = None, epochs: int | None = None,
    hparams: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    config = run_config(variant, seed, batch_size, num_workers, lr, max_epochs,
                        vit_name, accelerator, devices, nhead)
    checkpoint = checkpoint or output_dir / f"{variant.value}_checkpoint.pth"
    # Optional supplied hyperparameters verify provenance; checkpoint metadata defines architecture.
    chosen = None
    if hparams is not None or any(value is not None for value in (alpha, k_loss, epochs)):
        chosen = resolve_hparams(hparams, alpha, k_loss, epochs, variant.value)
    typer.echo(str(evaluate_checkpoint(checkpoint, data_dir, output_dir, config, expected_hparams=chosen)))


@app.command("attention-maps")
def attention_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False,
        help="Ours checkpoint for spatial maps; global_te has one token and is rejected.")],
    data_dir: DataDir = Path("data"), output_dir: OutputDir = Path("outputs"),
    model_ids: Annotated[list[int] | None, typer.Option(
        help="Repeat zero-based CSV Model IDs; labels/filenames use Model <ID+1> (paper Model 21 = ID 20). Default all.")] = None,
    categories: Annotated[list[str] | None, typer.Option(
        help="Repeat test categories for the grid; default first four sorted categories.")] = None,
    batch_size: BatchSize = 16, accelerator: str = "auto", devices: Devices = "auto",
) -> None:
    from .attention import attention_maps
    config = run_config(Variant.ours, 42, batch_size, 0, 1e-3, 1, DEFAULT_VIT,
                        accelerator, devices, 8)
    try:
        path = attention_maps(checkpoint, data_dir, output_dir, model_ids=model_ids,
                              categories=categories, batch_size=batch_size,
                              accelerator=accelerator, devices=config.devices)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(str(path))


@app.command("latency")
def latency_command(
    variant: Variant = Variant.ours,
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    vit_name: str = DEFAULT_VIT,
    pool_sizes: Annotated[list[int] | None, typer.Option(min=1, help="Repeat pool sizes; default 2^3 through 2^13.")] = None,
    chunk_size: Annotated[int, typer.Option(min=1, help="Candidate scoring batch limit.")] = 2048,
    warmup: Annotated[int, typer.Option(min=0)] = 10,
    repeats: Annotated[int, typer.Option(min=1)] = 100,
    device: Annotated[str | None, typer.Option(help="Default cuda if available, else cpu.")] = None,
    output_dir: OutputDir = Path("outputs"), seed: int = 42, nhead: NHead = 8,
) -> None:
    """Time encoding, scoring and ranking; image decoding/preprocessing are excluded."""
    from .latency import latency
    typer.echo(str(latency(output_dir, variant=variant.value, checkpoint=checkpoint, vit_name=vit_name,
                           pool_sizes=pool_sizes, chunk_size=chunk_size, warmup=warmup,
                           repeats=repeats, device=device, seed=seed, nhead=nhead)))


@app.command("alpha-sweep")
def alpha_sweep_command(
    hparams: Annotated[Path, typer.Option(exists=True, dir_okay=False,
        help="Selection JSON with per_alpha CV-selected k_loss and n_epochs.")],
    data_dir: DataDir = Path("data"), variant: Variant = Variant.ours,
    output_dir: OutputDir = Path("outputs"), seed: int = 42,
    batch_size: BatchSize = 16, num_workers: NumWorkers = 4, lr: float = 1e-3,
    max_epochs: MaxEpochs = 100, vit_name: str = DEFAULT_VIT,
    accelerator: str = "auto", devices: Devices = "auto", nhead: NHead = 8,
) -> None:
    """Report test performance for fixed alphas; test data never select hyperparameters."""
    from .alpha_sweep import alpha_sweep
    config = run_config(variant, seed, batch_size, num_workers, lr, max_epochs,
                        vit_name, accelerator, devices, nhead)
    typer.echo(str(alpha_sweep(config, hparams, data_dir, output_dir)))


@app.command("plot-alpha")
def plot_alpha_command(
    csv_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path | None, typer.Option(help="Default: directory containing CSV.")] = None,
) -> None:
    """Replot an alpha-sweep CSV without training or test evaluation."""
    from .alpha_sweep import plot_alpha
    typer.echo(str(plot_alpha(csv_path, output_dir)))


class BaselineMethod(str, Enum):
    wa = "wa"
    ni = "ni"
    ar = "ar"


@app.command("train-simsiam")
def simsiam_command(
    data_dir: DataDir = Path("data"), output_dir: OutputDir = Path("outputs"),
    vit_name: str = DEFAULT_VIT, seed: int = 42, batch_size: BatchSize = 32,
    num_workers: NumWorkers = 4, lr: float = 1e-3,
    weight_decay: Annotated[float, typer.Option(min=0)] = 1e-4,
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    accelerator: str = "auto", devices: Devices = "1",
) -> None:
    """Fine-tune SimSiam; defaults follow ISMR where the paper is silent."""
    from .baselines import train_simsiam
    config = run_config(Variant.ours, seed, batch_size, num_workers, lr, epochs,
                        vit_name, accelerator, devices, 8)
    typer.echo(str(train_simsiam(config, data_dir, output_dir, epochs=epochs,
                                weight_decay=weight_decay)))


@app.command("baselines")
def baselines_command(
    data_dir: DataDir = Path("data"), output_dir: OutputDir = Path("outputs"),
    simsiam_checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    methods: Annotated[list[BaselineMethod] | None, typer.Option(help="Repeat for each method; default: wa, ni, ar.")] = None,
    batch_size: BatchSize = 32, num_workers: NumWorkers = 4,
    accelerator: str = "auto", devices: Devices = "1",
) -> None:
    """Evaluate Sec. 5.2 WA/NI/AR on original test images."""
    from .baselines import baselines
    selected = [m.value for m in methods] if methods else ["wa", "ni", "ar"]
    if simsiam_checkpoint is None and {"wa", "ni"} & set(selected):
        raise typer.BadParameter("--simsiam-checkpoint is required for wa/ni")
    config = run_config(Variant.ours, 42, batch_size, num_workers, 1e-3, 1,
                        DEFAULT_VIT, accelerator, devices, 8)
    for path in baselines(data_dir, output_dir, config, methods=selected,
                          simsiam_checkpoint=simsiam_checkpoint):
        typer.echo(str(path))


@app.command("make-tables")
def tables_command(
    results_dir: Annotated[list[Path] | None, typer.Option(help="Repeat evaluation directories.")] = None,
    output_dir: OutputDir = Path("outputs/tables"),
    ours_results: Path | None = None, prior_results: Path | None = None,
    wa_results: Path | None = None, ni_results: Path | None = None, ar_results: Path | None = None,
    global_te_results: Path | None = None, local_mlp_results: Path | None = None,
) -> None:
    """Assemble Tables 1 and 2 from test CSVs (missing methods are skipped)."""
    from .tables import make_tables
    paths = dict(ours=ours_results, prior=prior_results, wa=wa_results, ni=ni_results,
                 ar=ar_results, global_te=global_te_results, local_mlp=local_mlp_results)
    try:
        for path in make_tables(results_dir or [], output_dir, paths=paths):
            typer.echo(str(path))
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
