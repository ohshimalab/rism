#!/usr/bin/env bash
# Reproduces attention for paper Models 21, 7, 31 (zero-based CSV IDs 20, 6, 30).
# Prerequisites: data, OUT_DIR/test/ours_checkpoint.pth. Outputs: OUT_DIR/attention/ NPZ, overlays/grid.
# Cost: test-set inference, typically minutes; CATEGORIES is a space-separated list.
set -euo pipefail
source scripts/_common.sh
category_args=()
values=()
read -r -a values <<< "${CATEGORIES:-}"
for value in "${values[@]}"; do category_args+=(--categories "$value"); done
"$PYTHON" -m rism attention-maps --checkpoint "$OUT_DIR/test/ours_checkpoint.pth" \
    --data-dir "$DATA_DIR" --output-dir "$OUT_DIR/attention" \
    --model-ids 20 --model-ids 6 --model-ids 30 "${category_args[@]}"
