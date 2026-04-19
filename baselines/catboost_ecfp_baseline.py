#!/usr/bin/env python3
"""Train CatBoost baselines on Uni-Mol property benchmarks using RDKit ECFP fingerprints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor

from baselines.catboost_fp2_baseline import (
    CLASSIFICATION_TASKS,
    MAE_TASKS,
    TASK_ORDER,
    auc_roc_binary,
    canonical_tasks,
    load_lmdb_split,
    mae,
    nanmean,
    rmse,
    write_csv,
)
from baselines.xgb_ecfp_baseline import build_feature_matrix, build_fp_cache, fingerprint_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_downloads/unimol/molecular_property_prediction"),
        help="Root with per-task folders containing train/valid/test.lmdb",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list (e.g., esol,qm9) or 'all'",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius; radius=2 corresponds to ECFP4",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=1024,
        help="Bit-vector width for the Morgan fingerprint",
    )
    parser.add_argument(
        "--use-chirality",
        action="store_true",
        help="Include chirality in Morgan fingerprint generation",
    )
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/catboost_ecfp4_1024"),
        help="Where summary CSV/JSON outputs are written",
    )
    return parser.parse_args()


def train_target_classifier(x_train, y_train, x_valid, y_valid, x_test, y_test, args: argparse.Namespace) -> tuple[float, float]:
    mask_train = (y_train == y_train) & (y_train >= 0)
    mask_valid = (y_valid == y_valid) & (y_valid >= 0)
    mask_test = (y_test == y_test) & (y_test >= 0)

    if mask_train.sum() < 2 or mask_valid.sum() < 2 or mask_test.sum() < 2:
        return float("nan"), float("nan")

    ytr = (y_train[mask_train] > 0).astype("int32")
    yva = (y_valid[mask_valid] > 0).astype("int32")
    yte = (y_test[mask_test] > 0).astype("int32")

    if len(set(ytr.tolist())) < 2:
        return float("nan"), float("nan")

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        thread_count=args.threads,
        verbose=False,
    )

    model.fit(
        x_train[mask_train],
        ytr,
        eval_set=(x_valid[mask_valid], yva),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=False,
    )

    valid_pred = model.predict_proba(x_valid[mask_valid])[:, 1]
    test_pred = model.predict_proba(x_test[mask_test])[:, 1]
    return auc_roc_binary(yva, valid_pred), auc_roc_binary(yte, test_pred)


def train_target_regressor(
    x_train,
    y_train,
    x_valid,
    y_valid,
    x_test,
    y_test,
    metric_kind: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    mask_train = y_train == y_train
    mask_valid = y_valid == y_valid
    mask_test = y_test == y_test

    if mask_train.sum() < 2 or mask_valid.sum() < 2 or mask_test.sum() < 2:
        return float("nan"), float("nan")

    eval_metric = "MAE" if metric_kind == "mae" else "RMSE"
    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        loss_function="RMSE",
        eval_metric=eval_metric,
        random_seed=args.seed,
        thread_count=args.threads,
        verbose=False,
    )

    model.fit(
        x_train[mask_train],
        y_train[mask_train],
        eval_set=(x_valid[mask_valid], y_valid[mask_valid]),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=False,
    )

    valid_pred = model.predict(x_valid[mask_valid])
    test_pred = model.predict(x_test[mask_test])

    if metric_kind == "mae":
        return mae(y_valid[mask_valid], valid_pred), mae(y_test[mask_test], test_pred)
    return rmse(y_valid[mask_valid], valid_pred), rmse(y_test[mask_test], test_pred)


def run_task(task: str, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    task_dir = args.data_root / task
    train_path = task_dir / "train.lmdb"
    valid_path = task_dir / "valid.lmdb"
    test_path = task_dir / "test.lmdb"

    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files under {task_dir}")

    train_smiles, y_train = load_lmdb_split(train_path)
    valid_smiles, y_valid = load_lmdb_split(valid_path)
    test_smiles, y_test = load_lmdb_split(test_path)

    n_targets = int(max(y_train.shape[1], y_valid.shape[1], y_test.shape[1]))

    def pad_width(arr, width: int):
        if arr.shape[1] == width:
            return arr
        import numpy as np

        out = np.full((arr.shape[0], width), np.nan, dtype=np.float32)
        out[:, : arr.shape[1]] = arr
        return out

    y_train = pad_width(y_train, n_targets)
    y_valid = pad_width(y_valid, n_targets)
    y_test = pad_width(y_test, n_targets)

    all_smiles = train_smiles + valid_smiles + test_smiles
    onbits_map, fp_failures = build_fp_cache(all_smiles, args.radius, args.fp_bits, args.use_chirality)

    x_train, keep_train = build_feature_matrix(train_smiles, onbits_map, args.fp_bits)
    x_valid, keep_valid = build_feature_matrix(valid_smiles, onbits_map, args.fp_bits)
    x_test, keep_test = build_feature_matrix(test_smiles, onbits_map, args.fp_bits)

    x_train = x_train[keep_train]
    y_train = y_train[keep_train]
    x_valid = x_valid[keep_valid]
    y_valid = y_valid[keep_valid]
    x_test = x_test[keep_test]
    y_test = y_test[keep_test]

    task_type = "classification" if task in CLASSIFICATION_TASKS else "regression"
    metric_name = "auc" if task_type == "classification" else ("mae" if task in MAE_TASKS else "rmse")

    per_target_rows: list[dict[str, object]] = []
    valid_metrics: list[float] = []
    test_metrics: list[float] = []

    for t in range(n_targets):
        if task_type == "classification":
            v_score, t_score = train_target_classifier(
                x_train,
                y_train[:, t],
                x_valid,
                y_valid[:, t],
                x_test,
                y_test[:, t],
                args,
            )
        else:
            v_score, t_score = train_target_regressor(
                x_train,
                y_train[:, t],
                x_valid,
                y_valid[:, t],
                x_test,
                y_test[:, t],
                metric_name,
                args,
            )

        per_target_rows.append(
            {
                "task": task,
                "target_index": t,
                "metric": metric_name,
                "valid": v_score,
                "test": t_score,
            }
        )
        valid_metrics.append(v_score)
        test_metrics.append(t_score)

    summary = {
        "task": task,
        "task_type": task_type,
        "metric": metric_name,
        "fingerprint": fingerprint_name(args.radius, args.fp_bits),
        "radius": args.radius,
        "fp_bits": args.fp_bits,
        "use_chirality": bool(args.use_chirality),
        "n_targets": n_targets,
        "n_train": int(y_train.shape[0]),
        "n_valid": int(y_valid.shape[0]),
        "n_test": int(y_test.shape[0]),
        "n_feature_failures": int(fp_failures),
        "valid": nanmean(valid_metrics),
        "test": nanmean(test_metrics),
    }
    return summary, per_target_rows


def main() -> None:
    args = parse_args()
    tasks = canonical_tasks(args.tasks)

    if not args.data_root.exists():
        raise FileNotFoundError(f"data-root does not exist: {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running tasks: {', '.join(tasks)}")
    print(f"Data root: {args.data_root}")
    print(f"Fingerprint: {fingerprint_name(args.radius, args.fp_bits)}")

    summaries: list[dict[str, object]] = []
    all_target_rows: list[dict[str, object]] = []

    t0 = time.time()
    for task in tasks:
        task_start = time.time()
        print(f"\n=== {task} ===")
        summary, target_rows = run_task(task, args)
        summary["elapsed_sec"] = round(time.time() - task_start, 3)
        summaries.append(summary)
        all_target_rows.extend(target_rows)
        print(
            f"{task}: valid={summary['valid']:.6f}, test={summary['test']:.6f}, "
            f"targets={summary['n_targets']}, elapsed={summary['elapsed_sec']}s"
        )

    total_sec = round(time.time() - t0, 3)

    summary_path = args.output_dir / "summary.csv"
    target_path = args.output_dir / "per_target.csv"
    json_path = args.output_dir / "summary.json"

    write_csv(summary_path, summaries)
    write_csv(target_path, all_target_rows)

    payload = {
        "config": {
            "data_root": str(args.data_root),
            "tasks": tasks,
            "radius": args.radius,
            "fp_bits": args.fp_bits,
            "use_chirality": bool(args.use_chirality),
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "random_strength": args.random_strength,
            "bagging_temperature": args.bagging_temperature,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
            "threads": args.threads,
        },
        "elapsed_sec": total_sec,
        "summary": summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Per-target: {target_path}")
    print(f"JSON: {json_path}")
    print(f"Total elapsed: {total_sec}s")


if __name__ == "__main__":
    main()
