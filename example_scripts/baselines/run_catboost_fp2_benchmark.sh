#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATA_ROOT="${1:-$ROOT_DIR/data_downloads/unimol/molecular_property_prediction}"
OUTPUT_DIR="${2:-$ROOT_DIR/results/catboost_fp2}"
TASKS="${3:-all}"
FEATURE_MODE="${4:-tanimoto}"
FINGERPRINT="${5:-FP2}"

uv run python baselines/catboost_fp2_baseline.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --tasks "$TASKS" \
  --feature-mode "$FEATURE_MODE" \
  --fingerprint "$FINGERPRINT" \
  --n-anchors 256 \
  --iterations 600 \
  --depth 8 \
  --learning-rate 0.05 \
  --seed 42
