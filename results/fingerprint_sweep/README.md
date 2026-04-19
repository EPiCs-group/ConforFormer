# Fingerprint Sweep (OpenBabel)

This sweep compares standard OpenBabel fingerprints with the same CatBoost setup.

## Shared training config
- feature_mode: `tanimoto`
- n_anchors: `96`
- iterations: `70`
- depth: `5`
- learning_rate: `0.09`
- l2_leaf_reg: `3.5`
- threads: `8`
- seed: `42`
- tasks: all 14 molecular property benchmarks

## Fingerprints tested
- `FP2` (reference from `results/catboost_tuning_runs/c3/summary.csv`)
- `FP3`
- `FP4`
- `MACCS`

## Main outputs
- `fingerprint_literature_overview.csv`
- `fingerprint_literature_detail.csv`
- `fingerprint_per_task_best.csv`
- `best_fingerprint.txt`

## Quick summary
By mean normalized score vs literature ranges, `FP4` performed best in this sweep.
