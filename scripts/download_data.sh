#!/usr/bin/env bash
# Reproduces the dataset prerequisite for all paper results.
# Prerequisites: git, network, ~3 GB disk. Outputs: DATA_DIR symlink, CACHE_DIR checkout.
# Cost: ~3 GB download; duration depends on network speed.
set -euo pipefail
source scripts/_common.sh
CACHE_DIR=${CACHE_DIR:-.cache/ismr}
if [[ -e "$DATA_DIR" || -L "$DATA_DIR" ]]; then
    if [[ -d "$DATA_DIR" && ! -L "$DATA_DIR" && -z "$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        rmdir "$DATA_DIR"
    else
        echo "Refusing to overwrite existing non-empty DATA_DIR: $DATA_DIR" >&2
        exit 1
    fi
fi
if [[ ! -d "$CACHE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$CACHE_DIR")"
    git clone --depth 1 --filter=blob:none --sparse https://github.com/ohshimalab/ismr "$CACHE_DIR"
fi
git -C "$CACHE_DIR" sparse-checkout set data
[[ -d "$CACHE_DIR/data" ]] || { echo "Missing $CACHE_DIR/data" >&2; exit 1; }
mkdir -p "$(dirname "$DATA_DIR")"
ln -s "$(cd "$CACHE_DIR/data" && pwd)" "$DATA_DIR"
echo "Data available at $DATA_DIR (~3 GB)."
