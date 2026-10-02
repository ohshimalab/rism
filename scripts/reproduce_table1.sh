#!/usr/bin/env bash
# Reproduces Table 1 (per-category test nDCG).
# Prerequisites: data, GPU, ours/prior selection JSONs, baselines extra (lightly).
# Outputs: OUT_DIR/test checkpoints/CSVs/features; OUT_DIR/tables/table{1,2}.{csv,md}.
# Cost: final ours/prior training plus 30 SimSiam epochs; hours depending on selected epochs.
set -euo pipefail
source scripts/_common.sh
# Check all selections before starting expensive work.
require_hparams ours
require_hparams prior
for variant in ours prior; do train_evaluate "$variant"; done
"$PYTHON" -m rism train-simsiam --data-dir "$DATA_DIR" --output-dir "$OUT_DIR/test" "${extra_args[@]}"
"$PYTHON" -m rism baselines --data-dir "$DATA_DIR" --output-dir "$OUT_DIR/test" \
    --simsiam-checkpoint "$OUT_DIR/test/simsiam_backbone.pth"
"$PYTHON" -m rism make-tables --results-dir "$OUT_DIR/test" --output-dir "$OUT_DIR/tables"
