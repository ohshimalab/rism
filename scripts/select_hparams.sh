#!/usr/bin/env bash
# Reproduces CV-based hyperparameter selection for Tables 1/2 and alpha figure.
# Prerequisites: complete variant CV CSV. Output: OUT_DIR/cv/<variant>_hparams.json.
# Cost: seconds on CPU.
set -euo pipefail
source scripts/_common.sh
check_variant "${1:-}"
cv_csvs=()
cv_args=()
read -r -a cv_csvs <<< "${CV_CSVS:-$OUT_DIR/cv/${1}_cv.csv}"
for path in "${cv_csvs[@]}"; do cv_args+=(--cv-csv "$path"); done
"$PYTHON" -m rism select-hparams --variant "$1" "${cv_args[@]}" --output-dir "$OUT_DIR/cv"
