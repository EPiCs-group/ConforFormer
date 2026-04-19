#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATA_ROOT="${1:-$ROOT_DIR/data_downloads/unimol/molecular_property_prediction}"
CLASS_OUT="${2:-$ROOT_DIR/results/ecfp4_xgb_classification_tuning}"
REG_OUT="${3:-$ROOT_DIR/results/ecfp4_xgb_regression_tuning}"
THREADS="${4:-8}"
REPEATS="${5:-5}"
BASE_SEED="${6:-42}"

uv run python -m baselines.tune_xgb_ecfp4_classification \
  --data-root "$DATA_ROOT" \
  --output-root "$CLASS_OUT" \
  --threads "$THREADS" \
  --repeats "$REPEATS" \
  --base-seed "$BASE_SEED"

uv run python -m baselines.tune_xgb_ecfp4_regression \
  --data-root "$DATA_ROOT" \
  --output-root "$REG_OUT" \
  --threads "$THREADS" \
  --repeats "$REPEATS" \
  --base-seed "$BASE_SEED"
