#!/usr/bin/env python3
"""ECFP4_1024 XGBoost hyperparameter tuning + 5-seed repeat evaluation on classification tasks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev

from baselines.catboost_fp2_baseline import write_csv
from baselines.tune_xgb_ecfp4_regression import get_config_space
from baselines.xgb_ecfp_baseline import run_task

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

CLASSIFICATION_TASKS = [
    "bbbp",
    "bace",
    "clintox",
    "tox21",
    "toxcast",
    "sider",
    "hiv",
    "muv",
]

TASK_TO_BENCHMARK = {
    "bbbp": "BBBP",
    "bace": "BACE",
    "clintox": "ClinTox",
    "tox21": "Tox21",
    "toxcast": "ToxCast",
    "sider": "SIDER",
    "hiv": "HIV",
    "muv": "MUV",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_downloads/unimol/molecular_property_prediction"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/ecfp4_xgb_classification_tuning"),
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--classification-ranges",
        type=Path,
        default=REFERENCE_DIR / "classification_literature_range.csv",
    )
    return p.parse_args()


def load_literature_ranges(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = (float(row["lit_min"]), float(row["lit_max"]))
    return out


def make_base_args(
    data_root: Path,
    output_dir: Path,
    threads: int,
    seed: int,
    config: dict[str, float | int | str],
) -> Namespace:
    return Namespace(
        data_root=data_root,
        tasks=",".join(CLASSIFICATION_TASKS),
        radius=2,
        fp_bits=1024,
        use_chirality=False,
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        reg_alpha=float(config["reg_alpha"]),
        gamma=float(config["gamma"]),
        max_bin=256,
        early_stopping_rounds=50,
        seed=seed,
        threads=threads,
        tree_method="hist",
        output_dir=output_dir,
    )


def compute_run_metrics(summary_rows: list[dict[str, object]], lit_ranges: dict[str, tuple[float, float]]) -> dict[str, float | int]:
    norms: list[float] = []
    within = 0

    for row in summary_rows:
        task = str(row["task"])
        val = float(row["test"])
        bench = TASK_TO_BENCHMARK[task]
        lit_min, lit_max = lit_ranges[bench]
        norm = (val - lit_min) / (lit_max - lit_min)
        norms.append(norm)
        if lit_min <= val <= lit_max:
            within += 1

    return {
        "within_range_count": within,
        "mean_normalized_score": mean(norms),
        "mean_norm_classification": mean(norms),
    }


def run_config(
    config: dict[str, float | int | str],
    seed: int,
    data_root: Path,
    output_dir: Path,
    threads: int,
    lit_ranges: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float | int]]:
    args = make_base_args(data_root, output_dir, threads, seed, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] {config['config_id']} seed={seed} -> {output_dir}", flush=True)
    print(
        "[cfg] "
        f"n_estimators={args.n_estimators} depth={args.max_depth} lr={args.learning_rate} "
        f"min_child_weight={args.min_child_weight} subsample={args.subsample} "
        f"colsample={args.colsample_bytree} reg_lambda={args.reg_lambda}",
        flush=True,
    )

    summaries: list[dict[str, object]] = []
    per_target: list[dict[str, object]] = []

    t0 = time.time()
    for task in CLASSIFICATION_TASKS:
        task_t0 = time.time()
        summary, targets = run_task(task, args)
        summary["elapsed_sec"] = round(time.time() - task_t0, 3)
        summary["config_id"] = str(config["config_id"])
        summary["seed"] = seed
        summaries.append(summary)

        for row in targets:
            row["config_id"] = str(config["config_id"])
            row["seed"] = seed
        per_target.extend(targets)

        print(
            f"  {task}: test={float(summary['test']):.6f} elapsed={summary['elapsed_sec']}s",
            flush=True,
        )

    total_elapsed = round(time.time() - t0, 3)
    stats = compute_run_metrics(summaries, lit_ranges)
    stats["elapsed_sec_total"] = total_elapsed

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "per_target.csv", per_target)
    payload = {
        "config": config,
        "seed": seed,
        "stats": stats,
        "summary": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    print(
        f"[done] {config['config_id']} seed={seed} "
        f"norm={float(stats['mean_normalized_score']):.6f} "
        f"within={int(stats['within_range_count'])}/8 total={total_elapsed}s",
        flush=True,
    )
    return summaries, per_target, stats


def aggregate_repeats(repeat_rows: list[dict[str, object]], out_dir: Path) -> list[dict[str, object]]:
    by_task: dict[str, list[float]] = {}
    meta: dict[str, dict[str, str]] = {}
    for row in repeat_rows:
        task = str(row["task"])
        by_task.setdefault(task, []).append(float(row["test"]))
        meta[task] = {
            "task_type": str(row["task_type"]),
            "metric": str(row["metric"]),
            "benchmark": TASK_TO_BENCHMARK[task],
        }

    out: list[dict[str, object]] = []
    for task in CLASSIFICATION_TASKS:
        vals = by_task[task]
        m = mean(vals)
        s = stdev(vals) if len(vals) >= 2 else 0.0
        out.append(
            {
                "task": task,
                "benchmark": meta[task]["benchmark"],
                "task_type": meta[task]["task_type"],
                "metric": meta[task]["metric"],
                "mean_test": m,
                "sd_test": s,
                "n_runs": len(vals),
            }
        )

    write_csv(out_dir / "best_5seed_task_stats.csv", out)
    return out


def main() -> None:
    args = parse_args()
    lit_ranges = load_literature_ranges(args.classification_ranges)

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    configs = get_config_space()
    write_csv(out_root / "tuning_configs.csv", configs)

    tuning_overview: list[dict[str, object]] = []
    tuning_detail: list[dict[str, object]] = []

    print(f"[start] tuning {len(configs)} configs", flush=True)
    for cfg in configs:
        cfg_id = str(cfg["config_id"])
        cfg_out = out_root / "tuning" / cfg_id / f"seed_{args.base_seed}"
        summary_rows, _, stats = run_config(
            cfg,
            seed=args.base_seed,
            data_root=args.data_root,
            output_dir=cfg_out,
            threads=args.threads,
            lit_ranges=lit_ranges,
        )
        ov = {
            "config_id": cfg_id,
            "fingerprint": "ECFP4_1024",
            "model": "XGBoost",
            "seed": args.base_seed,
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "min_child_weight": cfg["min_child_weight"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "reg_lambda": cfg["reg_lambda"],
            "reg_alpha": cfg["reg_alpha"],
            "gamma": cfg["gamma"],
            **stats,
        }
        tuning_overview.append(ov)

        for row in summary_rows:
            detail_row = dict(row)
            detail_row["benchmark"] = TASK_TO_BENCHMARK[str(row["task"])]
            tuning_detail.append(detail_row)

        write_csv(out_root / "tuning_overview.csv", tuning_overview)
        write_csv(out_root / "tuning_summary_all.csv", tuning_detail)

    tuning_overview.sort(
        key=lambda r: (
            float(r["mean_normalized_score"]),
            float(r["within_range_count"]),
            -float(r["elapsed_sec_total"]),
        ),
        reverse=True,
    )
    best = tuning_overview[0]
    best_cfg_id = str(best["config_id"])
    best_cfg = next(c for c in configs if str(c["config_id"]) == best_cfg_id)

    (out_root / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    (out_root / "best_config.txt").write_text(best_cfg_id + "\n")
    print(f"\n[best] {best_cfg_id} -> {best_cfg}", flush=True)

    repeat_root = out_root / "best_repeats"
    repeat_rows: list[dict[str, object]] = []
    repeat_overview: list[dict[str, object]] = []

    print(f"[start] repeats={args.repeats} for best config", flush=True)
    for i in range(args.repeats):
        seed = args.base_seed + i
        run_dir = repeat_root / f"seed_{seed}"
        summary_rows, _, stats = run_config(
            best_cfg,
            seed=seed,
            data_root=args.data_root,
            output_dir=run_dir,
            threads=args.threads,
            lit_ranges=lit_ranges,
        )
        repeat_rows.extend(summary_rows)
        repeat_overview.append(
            {
                "seed": seed,
                "config_id": best_cfg_id,
                **stats,
            }
        )

    write_csv(repeat_root / "repeat_overview.csv", repeat_overview)
    write_csv(repeat_root / "repeat_summary_all.csv", repeat_rows)

    task_stats = aggregate_repeats(repeat_rows, repeat_root)

    print("\n[done] tuning + repeats complete", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print("Top tuning rows:", flush=True)
    for row in tuning_overview[:5]:
        print(row, flush=True)
    print("Task mean/sd (best config, repeats):", flush=True)
    for row in task_stats:
        print(row, flush=True)


if __name__ == "__main__":
    main()
