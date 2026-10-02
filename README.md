# Retrieving Instance Segmentation Models using Local Image Features via Transformer Encoder

This repository contains the code for the paper "Retrieving Instance Segmentation Models using Local Image Features via Transformer Encoder" (ICADL 2026). The dataset is from [ohshimalab/ismr](https://github.com/ohshimalab/ismr).

## Repository structure

```text
rism/
├── scripts/                 # Reproduction scripts (one per table/figure)
│   ├── download_data.sh
│   ├── run_cv.sh            # Nested cross-validation
│   ├── select_hparams.sh    # Hyperparameter selection from CV results
│   ├── reproduce_table1.sh
│   ├── reproduce_table2.sh
│   ├── reproduce_fig_alpha.sh
│   ├── reproduce_fig_attention.sh
│   ├── reproduce_fig_latency.sh
│   └── reproduce_all.sh
├── src/rism/
│   ├── cli.py               # Command-line interface (python -m rism)
│   ├── data.py              # Dataset and preprocessing
│   ├── models.py            # Proposed method, prior method and ablation variants
│   ├── losses.py            # NeuralNDCG loss
│   ├── metrics.py           # nDCG@k
│   ├── training.py          # Training and checkpoints
│   ├── cross_validation.py  # Leave-one-category-out cross-validation
│   ├── selection.py         # Hyperparameter selection
│   ├── evaluation.py        # Test-set evaluation
│   ├── baselines.py         # SimSiam, WA, NI and AR baselines
│   ├── alpha_sweep.py       # Effect of alpha
│   ├── attention.py         # Attention maps
│   ├── latency.py           # Retrieval latency
│   └── tables.py            # Paper tables
├── tests/
├── pyproject.toml
└── LICENSE
```

## Commands

Setup:

```bash
pip install -e ".[baselines]"
bash scripts/download_data.sh
```

Reproduce the paper (run cross-validation and selection first, for each of `ours`, `prior`, `global_te`, `local_mlp`):

```bash
bash scripts/run_cv.sh ours
bash scripts/select_hparams.sh ours
bash scripts/reproduce_table1.sh
bash scripts/reproduce_table2.sh
bash scripts/reproduce_fig_alpha.sh
bash scripts/reproduce_fig_attention.sh
bash scripts/reproduce_fig_latency.sh
```

`bash scripts/reproduce_all.sh` runs everything. Full cross-validation is expensive; use `FOLDS`, `ALPHAS` and `K_LOSS` to split `run_cv.sh` into smaller resumable runs.

Individual commands (`python -m rism <command> --help` for options):

| Command | Description |
| --- | --- |
| `cross-validate` | Nested leave-one-category-out cross-validation |
| `select-hparams` | Select hyperparameters from cross-validation results |
| `train` | Train a model on the full training set |
| `evaluate` | Evaluate a trained model on the test set |
| `train-simsiam` | Fine-tune SimSiam for the WA/NI baselines |
| `baselines` | Evaluate the WA, NI and AR baselines |
| `alpha-sweep` | Train and evaluate for each alpha |
| `plot-alpha` | Plot alpha-sweep results |
| `attention-maps` | Create attention maps |
| `latency` | Measure retrieval latency |
| `make-tables` | Build Tables 1 and 2 from evaluation results |

## Citation

```bibtex
@inproceedings{pham2026rism,
  author = {Huu-Long Pham and Masataka Kubouchi and Takuma Nishimoto and Takehiro Yamamoto and Hiroaki Ohshima},
  title = {Retrieving Instance Segmentation Models using Local Image Features via Transformer Encoder},
  booktitle = {Proceedings of the International Conference on Asian Digital Libraries (ICADL 2026)},
  series = {Lecture Notes in Computer Science},
  volume = {17297},
  publisher = {Springer},
  year = {2026}
}
```
