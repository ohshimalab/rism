#!/usr/bin/env bash
# Reproduces the alpha-effect figure using CV-selected settings per alpha.
# Prerequisites: data, GPU, ours_hparams.json. Outputs: OUT_DIR/alpha/ CSV, PDF/PNG, checkpoints.
# Cost: one full training per alpha; hours, resumable by completed alpha.
set -euo pipefail
source scripts/_common.sh
require_hparams ours
"$PYTHON" -m rism alpha-sweep --variant ours --data-dir "$DATA_DIR" \
    --hparams "$OUT_DIR/cv/ours_hparams.json" --output-dir "$OUT_DIR/alpha" "${extra_args[@]}"
