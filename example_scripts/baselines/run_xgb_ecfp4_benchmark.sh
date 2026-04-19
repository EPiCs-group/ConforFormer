#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATA_ROOT="${1:-$ROOT_DIR/data_downloads/unimol/molecular_property_prediction}"
OUTPUT_DIR="${2:-$ROOT_DIR/results/xgb_ecfp4_1024}"
TASKS="${3:-all}"

uv run python -m baselines.xgb_ecfp_baseline \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --tasks "$TASKS" \
  --radius 2 \
  --fp-bits 1024 \
  --n-estimators 700 \
  --max-depth 6 \
  --learning-rate 0.05 \
  --min-child-weight 1.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --seed 42
