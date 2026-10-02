#!/usr/bin/env bash
# Reproduces all tables/figures, including full CV and data download if missing.
# Prerequisites: git/network, GPU, installed baselines extra; several GB of disk.
# Outputs: DATA_DIR, OUT_DIR/{cv,test,tables,alpha,attention,latency}.
# Cost: on the order of 1,000 GPU-hours for ours CV alone, plus ablations/final runs.
set -euo pipefail
source scripts/_common.sh
echo 'WARNING: FULL REPRODUCTION IS VERY EXPENSIVE (~1,000 GPU-hours for ours CV alone).' >&2
if [[ ! -d "$DATA_DIR" ]]; then bash scripts/download_data.sh; fi
for variant in ours prior global_te local_mlp; do
    bash scripts/run_cv.sh "$variant"
    bash scripts/select_hparams.sh "$variant"
done
bash scripts/reproduce_table1.sh
bash scripts/reproduce_table2.sh
bash scripts/reproduce_fig_alpha.sh
bash scripts/reproduce_fig_attention.sh
bash scripts/reproduce_fig_latency.sh
