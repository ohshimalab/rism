#!/usr/bin/env bash
# Shared reproduction settings; prerequisite: run from repo root with rism importable.
# Outputs: none. Cost: negligible; source this file from the result scripts.
set -euo pipefail
[[ -f pyproject.toml && -d src/rism ]] || { echo 'Run from the rism repository root.' >&2; exit 1; }
DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-outputs}
PYTHON=${PYTHON:-python}
export DATA_DIR OUT_DIR PYTHON
# EXTRA_ARGS is a whitespace-separated option list, without shell quoting or evaluation.
extra_args=()
read -r -a extra_args <<< "${EXTRA_ARGS:-}"
require_hparams() {
    [[ -f "$OUT_DIR/cv/${1}_hparams.json" ]] || {
        echo "Missing $OUT_DIR/cv/${1}_hparams.json; run scripts/run_cv.sh $1 then scripts/select_hparams.sh $1." >&2
        exit 1
    }
}
check_variant() {
    case "${1:-}" in ours|prior|global_te|local_mlp) ;;
        *) echo 'Usage: script VARIANT (ours|prior|global_te|local_mlp)' >&2; exit 1;; esac
}
train_evaluate() {
    local variant=$1
    require_hparams "$variant"
    "$PYTHON" -m rism train --variant "$variant" --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR/test" --hparams "$OUT_DIR/cv/${variant}_hparams.json" "${extra_args[@]}"
    "$PYTHON" -m rism evaluate --variant "$variant" --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR/test" --checkpoint "$OUT_DIR/test/${variant}_checkpoint.pth" \
        --hparams "$OUT_DIR/cv/${variant}_hparams.json"
}
