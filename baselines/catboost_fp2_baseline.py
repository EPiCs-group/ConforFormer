#!/usr/bin/env python3
"""Train CatBoost baselines on Uni-Mol property benchmarks using OpenBabel fingerprints.

Default feature mode is Tanimoto similarities to training anchors.
You can switch to raw fingerprint bit features with `--feature-mode bits`.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
import warnings
from pathlib import Path

import lmdb
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from openbabel import pybel

warnings.filterwarnings(
    "ignore",
    message="numpy.core.numeric is deprecated and has been renamed to numpy._core.numeric.*",
    category=DeprecationWarning,
)

TASK_ORDER = [
    "bbbp",
    "bace",
    "clintox",
    "tox21",
    "toxcast",
    "sider",
    "hiv",
    "muv",
    "esol",
    "freesolv",
    "lipo",
    "qm7dft",
    "qm8dft",
    "qm9dft",
]

TASK_ALIASES = {
    "bbbp": "bbbp",
    "bace": "bace",
    "clintox": "clintox",
    "tox21": "tox21",
    "toxcast": "toxcast",
    "sider": "sider",
    "hiv": "hiv",
    "muv": "muv",
    "esol": "esol",
    "freesolv": "freesolv",
    "lipo": "lipo",
    "qm7": "qm7dft",
    "qm8": "qm8dft",
    "qm9": "qm9dft",
    "qm7dft": "qm7dft",
    "qm8dft": "qm8dft",
    "qm9dft": "qm9dft",
}

CLASSIFICATION_TASKS = {
    "bbbp",
    "bace",
    "clintox",
    "tox21",
    "toxcast",
    "sider",
    "hiv",
    "muv",
}

MAE_TASKS = {"qm7dft", "qm8dft", "qm9dft"}


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
        help="Comma-separated list (e.g., bbbp,bace,qm9) or 'all'",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="tanimoto",
        choices=["tanimoto", "bits"],
        help="tanimoto = similarities to training anchors; bits = fingerprint bit vector",
    )
    parser.add_argument(
        "--fingerprint",
        type=str,
        default="FP2",
        choices=["FP2", "FP3", "FP4", "MACCS"],
        help="OpenBabel fingerprint type",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=1024,
        help="Bit-vector width when --feature-mode=bits",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=256,
        help="Number of train anchors when --feature-mode=tanimoto",
    )
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="CatBoost thread_count, -1 uses all cores",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/catboost_fp2"),
        help="Where summary CSV/JSON outputs are written",
    )
    return parser.parse_args()


def canonical_tasks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return TASK_ORDER[:]
    out: list[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in TASK_ALIASES:
            raise ValueError(f"Unknown task '{token}'.")
        mapped = TASK_ALIASES[key]
        if mapped not in out:
            out.append(mapped)
    if not out:
        raise ValueError("No tasks selected.")
    return out


def to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, float, np.integer, int)):
        return float(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def extract_smiles(record: dict) -> str | None:
    for key in ("smi", "smiles", "SMILES", "mol"):
        val = record.get(key)
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_target_vector(record: dict) -> list[float] | None:
    raw = None
    for key in ("target", "targets", "label", "labels", "y"):
        if key in record:
            raw = record[key]
            break
    if raw is None:
        return None

    if isinstance(raw, np.ndarray):
        arr = raw.reshape(-1).tolist()
    elif isinstance(raw, (list, tuple)):
        arr = list(raw)
    else:
        arr = [raw]

    return [to_float_or_nan(v) for v in arr]


def load_lmdb_split(path: Path) -> tuple[list[str], np.ndarray]:
    smiles: list[str] = []
    targets: list[list[float]] = []

    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = pickle.loads(value)
                if not isinstance(record, dict):
                    continue
                smi = extract_smiles(record)
                if smi is None:
                    continue
                target = extract_target_vector(record)
                if target is None:
                    continue
                smiles.append(smi)
                targets.append(target)
    finally:
        env.close()

    if not targets:
        return smiles, np.empty((0, 0), dtype=np.float32)

    width = max(len(row) for row in targets)
    mat = np.full((len(targets), width), np.nan, dtype=np.float32)
    for i, row in enumerate(targets):
        mat[i, : len(row)] = np.asarray(row, dtype=np.float32)
    return smiles, mat


def fingerprint_bits_and_int(smiles: str, fingerprint: str, n_bits: int) -> tuple[np.ndarray, int] | None:
    try:
        mol = pybel.readstring("smi", smiles)
        fp = mol.calcfp(fptype=fingerprint)
    except Exception:
        return None

    bits = np.zeros(n_bits, dtype=np.float32)
    bit_int = 0
    for bit in getattr(fp, "bits", []):
        if bit <= 0:
            continue
        idx = bit - 1
        bit_int |= 1 << idx
        if idx < n_bits:
            bits[idx] = 1.0
    return bits, bit_int


def build_fp_cache(
    smiles: list[str], fingerprint: str, n_bits: int
) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    bits_map: dict[str, np.ndarray] = {}
    int_map: dict[str, int] = {}
    failures = 0
    uniq = list(dict.fromkeys(smiles))

    for smi in uniq:
        result = fingerprint_bits_and_int(smi, fingerprint, n_bits)
        if result is None:
            failures += 1
            continue
        bits_map[smi], int_map[smi] = result

    return bits_map, int_map, failures


def tanimoto_similarity(a: int, b: int) -> float:
    union = (a | b).bit_count()
    if union == 0:
        return 0.0
    return (a & b).bit_count() / union


def select_anchor_fps(train_smiles: list[str], fp_int_map: dict[str, int], n_anchors: int, seed: int) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for smi in train_smiles:
        fp = fp_int_map.get(smi)
        if fp is None:
            continue
        if fp in seen:
            continue
        seen.add(fp)
        ordered.append(fp)

    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[: min(n_anchors, len(ordered))]


def build_feature_matrix(
    smiles: list[str],
    fp_bits_map: dict[str, np.ndarray],
    fp_int_map: dict[str, int],
    feature_mode: str,
    fp_bits: int,
    anchors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.zeros(len(smiles), dtype=bool)

    if feature_mode == "bits":
        x = np.zeros((len(smiles), fp_bits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            fp = fp_bits_map.get(smi)
            if fp is None:
                continue
            keep[i] = True
            x[i] = fp
        return x, keep

    x = np.zeros((len(smiles), len(anchors)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        fp = fp_int_map.get(smi)
        if fp is None:
            continue
        keep[i] = True
        for j, anchor in enumerate(anchors):
            x[i, j] = tanimoto_similarity(fp, anchor)
    return x, keep


def auc_roc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ys = y_score[order]
    yt = y_true[order]

    sum_pos_ranks = 0.0
    i = 0
    n = len(ys)
    while i < n:
        j = i + 1
        while j < n and ys[j] == ys[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        pos_count = int(yt[i:j].sum())
        sum_pos_ranks += avg_rank * pos_count
        i = j

    return (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def train_target_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    # In Uni-Mol molecular classification LMDBs, missing labels are encoded as -1.
    mask_train = np.isfinite(y_train) & (y_train >= 0)
    mask_valid = np.isfinite(y_valid) & (y_valid >= 0)
    mask_test = np.isfinite(y_test) & (y_test >= 0)

    if mask_train.sum() < 2 or mask_valid.sum() < 2 or mask_test.sum() < 2:
        return float("nan"), float("nan")

    ytr = (y_train[mask_train] > 0).astype(np.int32)
    yva = (y_valid[mask_valid] > 0).astype(np.int32)
    yte = (y_test[mask_test] > 0).astype(np.int32)

    if np.unique(ytr).size < 2:
        return float("nan"), float("nan")

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
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
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metric_kind: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    mask_train = np.isfinite(y_train)
    mask_valid = np.isfinite(y_valid)
    mask_test = np.isfinite(y_test)

    if mask_train.sum() < 2 or mask_valid.sum() < 2 or mask_test.sum() < 2:
        return float("nan"), float("nan")

    eval_metric = "MAE" if metric_kind == "mae" else "RMSE"
    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
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

    def pad_width(arr: np.ndarray, width: int) -> np.ndarray:
        if arr.shape[1] == width:
            return arr
        out = np.full((arr.shape[0], width), np.nan, dtype=np.float32)
        out[:, : arr.shape[1]] = arr
        return out

    y_train = pad_width(y_train, n_targets)
    y_valid = pad_width(y_valid, n_targets)
    y_test = pad_width(y_test, n_targets)

    all_smiles = train_smiles + valid_smiles + test_smiles
    fp_bits_map, fp_int_map, fp_failures = build_fp_cache(all_smiles, args.fingerprint, args.fp_bits)

    anchors: list[int] = []
    if args.feature_mode == "tanimoto":
        anchors = select_anchor_fps(train_smiles, fp_int_map, args.n_anchors, args.seed)
        if not anchors:
            raise RuntimeError(f"No anchor fingerprints available for task '{task}'.")

    x_train, keep_train = build_feature_matrix(
        train_smiles, fp_bits_map, fp_int_map, args.feature_mode, args.fp_bits, anchors
    )
    x_valid, keep_valid = build_feature_matrix(
        valid_smiles, fp_bits_map, fp_int_map, args.feature_mode, args.fp_bits, anchors
    )
    x_test, keep_test = build_feature_matrix(
        test_smiles, fp_bits_map, fp_int_map, args.feature_mode, args.fp_bits, anchors
    )

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
        "feature_mode": args.feature_mode,
        "fingerprint": args.fingerprint,
        "n_targets": n_targets,
        "n_train": int(y_train.shape[0]),
        "n_valid": int(y_valid.shape[0]),
        "n_test": int(y_test.shape[0]),
        "n_feature_failures": int(fp_failures),
        "n_anchors": int(len(anchors)) if args.feature_mode == "tanimoto" else 0,
        "valid": nanmean(valid_metrics),
        "test": nanmean(test_metrics),
    }
    return summary, per_target_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tasks = canonical_tasks(args.tasks)

    if not args.data_root.exists():
        raise FileNotFoundError(f"data-root does not exist: {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running tasks: {', '.join(tasks)}")
    print(f"Data root: {args.data_root}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Fingerprint: {args.fingerprint}")

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
            "feature_mode": args.feature_mode,
            "fingerprint": args.fingerprint,
            "fp_bits": args.fp_bits,
            "n_anchors": args.n_anchors,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
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
