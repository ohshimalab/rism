#!/usr/bin/env bash
# Reproduces the candidate-pool latency figure (2^3..2^13, chunks of 2048).
# Prerequisites: GPU and pretrained ViT (first use downloads); no checkpoint needed.
# Outputs: OUT_DIR/latency/ CSV, PDF/PNG. Cost: minutes, device-dependent.
set -euo pipefail
source scripts/_common.sh
"$PYTHON" -m rism latency --variant ours --chunk-size 2048 --output-dir "$OUT_DIR/latency"
