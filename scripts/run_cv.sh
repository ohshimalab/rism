#!/usr/bin/env bash
# Reproduces nested CV selection inputs for Tables 1/2 and the alpha figure.
# Prerequisites: dataset, pretrained ViT/GPU. Output: OUT_DIR/cv/<variant>_cv.csv + provenance/logs.
# Cost: ours full CV ~1,000 GPU-hours on RTX A6000; other variants roughly 10x less.
set -euo pipefail
source scripts/_common.sh
check_variant "${1:-}"
variant=$1
split_args=()
values=()
read -r -a values <<< "${FOLDS:-}"
for value in "${values[@]}"; do split_args+=(--folds "$value"); done
read -r -a values <<< "${ALPHAS:-}"
for value in "${values[@]}"; do split_args+=(--alphas "$value"); done
read -r -a values <<< "${K_LOSS:-}"
for value in "${values[@]}"; do split_args+=(--k-loss "$value"); done
echo 'WARNING: nested CV is the expensive step; completed CSV rows are resumable.' >&2
"$PYTHON" -m rism cross-validate --variant "$variant" --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR/cv" --csv-path "$OUT_DIR/cv/${CSV_NAME:-${variant}_cv.csv}" \
    "${split_args[@]}" "${extra_args[@]}"
