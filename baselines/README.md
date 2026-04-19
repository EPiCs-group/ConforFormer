# 2D Fingerprint Baselines

This directory contains the lightweight 2D baselines used for the MoleculeNet
benchmark comparisons.

The repo is self-contained for these runs:

- literature-range CSVs live in `baselines/reference/`
- tuning scripts default to those local files
- wrapper scripts live in `example_scripts/baselines/`

## CatBoost on OpenBabel fingerprints

`baselines/catboost_fp2_baseline.py` trains CatBoost models on OpenBabel
fingerprints (`FP2`, `FP3`, `FP4`, `MACCS`) using either raw bits or
train-anchor Tanimoto similarities.

Run all 14 tasks:

```bash
UV_CACHE_DIR=.uv-cache uv run python baselines/catboost_fp2_baseline.py \
  --data-root data_downloads/unimol/molecular_property_prediction \
  --output-dir results/catboost_fp2 \
  --feature-mode tanimoto \
  --fingerprint FP2 \
  --n-anchors 256
```

Or via wrapper:

```bash
example_scripts/baselines/run_catboost_fp2_benchmark.sh
```

Wrapper arguments:

```bash
example_scripts/baselines/run_catboost_fp2_benchmark.sh [DATA_ROOT] [OUTPUT_DIR] [TASKS] [FEATURE_MODE] [FINGERPRINT]
```

Notes:

- `--feature-mode tanimoto` uses Tanimoto similarities to train-set anchors
- `--feature-mode bits` uses raw fingerprint bit vectors
- outputs:
  - `summary.csv`
  - `per_target.csv`
  - `summary.json`

## XGBoost on RDKit ECFP4_1024

`baselines/xgb_ecfp_baseline.py` trains XGBoost models on RDKit Morgan
fingerprints. The paper baseline uses `radius=2` and `fp-bits=1024`, i.e.
`ECFP4_1024`.

Generic run:

```bash
uv run python -m baselines.xgb_ecfp_baseline \
  --data-root data_downloads/unimol/molecular_property_prediction \
  --tasks all \
  --radius 2 \
  --fp-bits 1024 \
  --output-dir results/xgb_ecfp4_1024
```

Wrapper:

```bash
example_scripts/baselines/run_xgb_ecfp4_benchmark.sh
```

## Reviewer-response tuning workflow

The main-text 2D baseline in the paper comes from two tuning harnesses:

- `baselines/tune_xgb_ecfp4_classification.py`
- `baselines/tune_xgb_ecfp4_regression.py`

These run a fixed config grid, select the best config against a literature-range
normalized score, then rerun the best config for 5 seeds.

Convenience wrapper:

```bash
example_scripts/baselines/run_xgb_ecfp4_reviewer_tuning.sh
```

Key outputs:

- classification: `results/ecfp4_xgb_classification_tuning/`
- regression: `results/ecfp4_xgb_regression_tuning/`
- best 5-seed summaries:
  - `results/ecfp4_xgb_classification_tuning/best_repeats/best_5seed_task_stats.csv`
  - `results/ecfp4_xgb_regression_tuning/best_repeats/best_5seed_task_stats.csv`

Best configs used in the paper:

- classification: `x03`
- regression: `x07`

See `baselines/xgb_ecfp4_config.md` for the exact hyperparameters and output
paths.
