#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

CODE_DIR="$(dirname "$ROOT_DIR")"
DEFAULT_DATA_ROOT="$CODE_DIR/conf_paper_code/data_downloads/unimol/molecular_property_prediction"

DATA_ROOT="${1:-$DEFAULT_DATA_ROOT}"
OUT_ROOT="${2:-$ROOT_DIR/results/reviewer_fingerprint_checks}"
THREADS="${3:-8}"
REPEATS="${4:-5}"
BASE_SEED="${5:-42}"

declare -a VARIANTS=(
  "2 2048 ecfp4_2048"
  "2 16384 ecfp4_16384"
  "3 2048 ecfp6_2048"
  "3 16384 ecfp6_16384"
)

for variant in "${VARIANTS[@]}"; do
  read -r RADIUS FP_BITS LABEL <<<"$variant"
  uv run python -m baselines.repeat_xgb_ecfp_eval \
    --data-root "$DATA_ROOT" \
    --output-root "$OUT_ROOT/$LABEL" \
    --radius "$RADIUS" \
    --fp-bits "$FP_BITS" \
    --threads "$THREADS" \
    --repeats "$REPEATS" \
    --base-seed "$BASE_SEED"
done
