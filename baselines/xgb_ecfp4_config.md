# XGBoost ECFP4_1024 Baseline

This is the 2D baseline now referenced in the main paper text.

Core representation:

- fingerprint family: RDKit Morgan fingerprint
- radius: `2` (`ECFP4`)
- bit length: `1024`
- chirality: `False`

Classification tuning:

- script: `baselines/tune_xgb_ecfp4_classification.py`
- selected config: `x03`
- parameters:
  - `n_estimators=600`
  - `max_depth=6`
  - `learning_rate=0.05`
  - `min_child_weight=1.0`
  - `subsample=0.85`
  - `colsample_bytree=0.75`
  - `reg_lambda=1.0`

Regression tuning:

- script: `baselines/tune_xgb_ecfp4_regression.py`
- selected config: `x07`
- parameters:
  - `n_estimators=700`
  - `max_depth=8`
  - `learning_rate=0.05`
  - `min_child_weight=10.0`
  - `subsample=0.75`
  - `colsample_bytree=0.65`
  - `reg_lambda=3.0`

5-seed result files:

- classification:
  - `results/ecfp4_xgb_classification_tuning/best_repeats/best_5seed_task_stats.csv`
- regression:
  - `results/ecfp4_xgb_regression_tuning/best_repeats/best_5seed_task_stats.csv`

Combined paper-facing rows were copied into the paper repo from those two files.
