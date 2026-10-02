#!/usr/bin/env bash
# Reproduces Table 2 (image-feature/fusion ablations).
# Prerequisites: data, GPU, selection JSONs; reuses existing ours/prior test CSVs.
# Outputs: OUT_DIR/test/<variant>_{checkpoint.pth,test_results.csv}, OUT_DIR/tables/.
# Cost: selected final trainings; ours ~5.3 min/epoch, ablations <1 min/epoch on A6000.
set -euo pipefail
source scripts/_common.sh
for variant in global_te local_mlp; do require_hparams "$variant"; done
for variant in ours prior; do
    if [[ ! -f "$OUT_DIR/test/${variant}_test_results.csv" ]]; then require_hparams "$variant"; fi
done
for variant in global_te local_mlp; do train_evaluate "$variant"; done
for variant in ours prior; do
    if [[ ! -f "$OUT_DIR/test/${variant}_test_results.csv" ]]; then train_evaluate "$variant"; fi
done
"$PYTHON" -m rism make-tables --results-dir "$OUT_DIR/test" --output-dir "$OUT_DIR/tables"
