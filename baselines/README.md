# 2D Fingerprint Baselines

This directory contains the lightweight 2D baselines used for the MoleculeNet
benchmark comparisons.

The repo is self-contained for these runs:

- literature-range CSVs live in `baselines/reference/`
- tuning scripts default to those local files
- wrapper scripts live in `example_scripts/baselines/`

## Prerequisites and data layout

From the repository root, install the lightweight baseline environment with:

```bash
uv sync
```

All baseline scripts expect the Uni-Mol molecular-property LMDBs under a data
root like:

```text
data_downloads/unimol/molecular_property_prediction/
  bace/
    train.lmdb
    valid.lmdb
    test.lmdb
  bbbp/
    train.lmdb
    valid.lmdb
    test.lmdb
  ...
```

Each task folder must contain `train.lmdb`, `valid.lmdb`, and `test.lmdb`.
Override the default location with `--data-root` or the first positional
argument of the wrapper scripts.

Unless noted otherwise, all scripts are meant to be run from the repository
root. The shell wrappers set `UV_CACHE_DIR=.uv-cache` automatically.

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

Defaults:

- `DATA_ROOT`: `data_downloads/unimol/molecular_property_prediction`
- `OUTPUT_DIR`: `results/catboost_fp2`
- `TASKS`: `all`
- `FEATURE_MODE`: `tanimoto`
- `FINGERPRINT`: `FP2`

Notes:

- `--feature-mode tanimoto` uses Tanimoto similarities to train-set anchors
- `--feature-mode bits` uses raw fingerprint bit vectors
- outputs written to `OUTPUT_DIR`:
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

Wrapper arguments:

```bash
example_scripts/baselines/run_xgb_ecfp4_benchmark.sh [DATA_ROOT] [OUTPUT_DIR] [TASKS]
```

Defaults:

- `DATA_ROOT`: `data_downloads/unimol/molecular_property_prediction`
- `OUTPUT_DIR`: `results/xgb_ecfp4_1024`
- `TASKS`: `all`

The wrapper pins the paper benchmark hyperparameters:

- `n_estimators=700`
- `max_depth=6`
- `learning_rate=0.05`
- `min_child_weight=1.0`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `seed=42`

Outputs written to `OUTPUT_DIR`:

- `summary.csv`
- `per_target.csv`
- `summary.json`

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

Wrapper arguments:

```bash
example_scripts/baselines/run_xgb_ecfp4_reviewer_tuning.sh [DATA_ROOT] [CLASS_OUT] [REG_OUT] [THREADS] [REPEATS] [BASE_SEED]
```

Defaults:

- `DATA_ROOT`: `data_downloads/unimol/molecular_property_prediction`
- `CLASS_OUT`: `results/ecfp4_xgb_classification_tuning`
- `REG_OUT`: `results/ecfp4_xgb_regression_tuning`
- `THREADS`: `8`
- `REPEATS`: `5`
- `BASE_SEED`: `42`

Direct invocations are also supported:

```bash
uv run python -m baselines.tune_xgb_ecfp4_classification \
  --data-root data_downloads/unimol/molecular_property_prediction \
  --output-root results/ecfp4_xgb_classification_tuning \
  --radius 2 \
  --fp-bits 1024 \
  --threads 8 \
  --repeats 5 \
  --base-seed 42

uv run python -m baselines.tune_xgb_ecfp4_regression \
  --data-root data_downloads/unimol/molecular_property_prediction \
  --output-root results/ecfp4_xgb_regression_tuning \
  --radius 2 \
  --fp-bits 1024 \
  --threads 8 \
  --repeats 5 \
  --base-seed 42
```

Useful flags:

- `--config-ids x03,x01,x10` restricts the tuning grid to a subset of config
  IDs
- `--radius` and `--fp-bits` let the same harness run alternative Morgan
  fingerprints such as `ECFP6_16384`

Key outputs:

- classification: `results/ecfp4_xgb_classification_tuning/`
- regression: `results/ecfp4_xgb_regression_tuning/`
- under each output root:
  - `tuning_configs.csv`
  - `tuning_overview.csv`
  - `tuning_summary_all.csv`
  - `best_config.json`
  - `best_config.txt`
  - `tuning/<config_id>/seed_<seed>/{summary.csv,per_target.csv,summary.json}`
  - `best_repeats/repeat_overview.csv`
  - `best_repeats/repeat_summary_all.csv`
- best 5-seed summaries:
  - `results/ecfp4_xgb_classification_tuning/best_repeats/best_5seed_task_stats.csv`
  - `results/ecfp4_xgb_regression_tuning/best_repeats/best_5seed_task_stats.csv`

Best configs used in the paper:

- classification: `x03`
- regression: `x07`

See `baselines/xgb_ecfp4_config.md` for the exact hyperparameters and output
paths.

## Reviewer-response fingerprint variants

`baselines/repeat_xgb_ecfp_eval.py` reruns the best classification and
regression XGBoost configs on a different Morgan fingerprint without repeating
the full grid search.

Generic example:

```bash
uv run python -m baselines.repeat_xgb_ecfp_eval \
  --data-root data_downloads/unimol/molecular_property_prediction \
  --output-root results/reviewer_fingerprint_checks/ecfp6_16384 \
  --radius 3 \
  --fp-bits 16384 \
  --threads 8 \
  --repeats 5 \
  --base-seed 42
```

By default the script reads the best paper configs from:

- `results/ecfp4_xgb_classification_tuning/best_config.json`
- `results/ecfp4_xgb_regression_tuning/best_config.json`

The reviewer wrapper runs four preset variants:

```bash
example_scripts/baselines/run_xgb_ecfp_reviewer_variants.sh [DATA_ROOT] [OUT_ROOT] [THREADS] [REPEATS] [BASE_SEED]
```

Defaults:

- `DATA_ROOT`: `data_downloads/unimol/molecular_property_prediction`
- `OUT_ROOT`: `results/reviewer_fingerprint_checks`
- `THREADS`: `8`
- `REPEATS`: `5`
- `BASE_SEED`: `42`

Variants covered by the wrapper:

- `ECFP4_2048`
- `ECFP4_16384`
- `ECFP6_2048`
- `ECFP6_16384`

Outputs written below each `OUT_ROOT/<variant>/` directory:

- `classification/best_repeats/seed_<seed>/{summary.csv,per_target.csv,summary.json}`
- `classification/best_repeats/repeat_overview.csv`
- `classification/best_repeats/best_5seed_task_stats.csv`
- `classification/summary.json`
- `regression/best_repeats/seed_<seed>/{summary.csv,per_target.csv,summary.json}`
- `regression/best_repeats/repeat_overview.csv`
- `regression/best_repeats/best_5seed_task_stats.csv`
- `regression/summary.json`
- top-level `summary.json`

For the interpretation of the currently checked-in reviewer screen, see
`baselines/reviewer_fingerprint_summary.md`.
