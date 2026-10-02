"""Image-level ranking evaluation and category-level reporting."""

from pathlib import Path

import pandas as pd

from .config import RunConfig
from .data import ImageDataset
from .metrics import ndcg
from .models import Retriever
from .training import image_size, load_checkpoint, loader, make_trainer


def evaluate_model(model: Retriever, dataset: ImageDataset, config: RunConfig,
                   output_dir: Path | str) -> pd.DataFrame:
    """Return per-category means; use the same nDCG implementation as training."""
    if dataset.model_ids != list(range(model.num_models)):
        raise ValueError("Evaluation candidate pool differs from the trained model")
    dataset = dataset.subset(image_size=image_size(model))
    trainer = make_trainer(config, output_dir, epochs=1, prediction=True)
    predictions = trainer.predict(model, dataloaders=loader(dataset, config))
    rows = []
    for batch in predictions:
        values = {k: ndcg(batch["scores"], batch["relevance"], k).tolist() for k in (3, 5)}
        rows.extend({"category": category, "ndcg_3": values[3][i], "ndcg_5": values[5][i]}
                    for i, category in enumerate(batch["category"]))
    return pd.DataFrame(rows).groupby("category", as_index=False, sort=True).mean()


def category_summary(categories: pd.DataFrame) -> pd.DataFrame:
    """Table 1 convention: mean and population standard deviation across categories."""
    metrics = ["ndcg_3", "ndcg_5"]
    summaries = pd.DataFrame([
        {"category": "Average", **categories[metrics].mean().to_dict()},
        {"category": "Standard Deviation", **categories[metrics].std(ddof=0).to_dict()},
    ])
    return pd.concat([categories, summaries], ignore_index=True)


def evaluate(checkpoint: Path | str, data_dir: Path | str, output_dir: Path | str,
             config: RunConfig, *,
             expected_hparams: tuple[float, int, int] | None = None) -> Path:
    model, metadata = load_checkpoint(checkpoint)
    if expected_hparams is not None and expected_hparams != (
            metadata["alpha"], metadata["k_loss"], metadata["n_epochs"]):
        raise ValueError("Requested hyperparameters differ from checkpoint")
    if model.variant != config.variant:
        raise ValueError("Requested variant differs from checkpoint")
    dataset = ImageDataset(data_dir, "test", originals_only=True)
    if dataset.model_ids != metadata["model_ids"]:
        raise ValueError("Test Model IDs differ from training")
    categories = evaluate_model(model, dataset, config, output_dir)
    path = Path(output_dir) / f"{model.variant}_test_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    category_summary(categories).to_csv(path, index=False)
    return path
