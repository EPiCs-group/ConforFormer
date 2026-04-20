#!/usr/bin/env python3
"""Repeat evaluation for alternative XGBoost ECFP variants using fixed best configs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev

from baselines.catboost_fp2_baseline import write_csv
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

REGRESSION_TASKS = [
    "esol",
    "freesolv",
    "lipo",
    "qm7dft",
    "qm8dft",
    "qm9dft",
]

CLASSIFICATION_BENCHMARKS = {
    "bbbp": "BBBP",
    "bace": "BACE",
    "clintox": "ClinTox",
    "tox21": "Tox21",
    "toxcast": "ToxCast",
    "sider": "SIDER",
    "hiv": "HIV",
    "muv": "MUV",
}

REGRESSION_BENCHMARKS = {
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipo": "Lipo",
    "qm7dft": "QM7",
    "qm8dft": "QM8",
    "qm9dft": "QM9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_downloads/unimol/molecular_property_prediction"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/xgb_ecfp_variant_eval"),
    )
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--fp-bits", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--classification-config",
        type=Path,
        default=Path("results/ecfp4_xgb_classification_tuning/best_config.json"),
    )
    parser.add_argument(
        "--regression-config",
        type=Path,
        default=Path("results/ecfp4_xgb_regression_tuning/best_config.json"),
    )
    parser.add_argument(
        "--classification-ranges",
        type=Path,
        default=REFERENCE_DIR / "classification_literature_range.csv",
    )
    parser.add_argument(
        "--regression-ranges",
        type=Path,
        default=REFERENCE_DIR / "regression_literature_range.csv",
    )
    return parser.parse_args()


def fingerprint_name(radius: int, fp_bits: int) -> str:
    return f"ECFP{radius * 2}_{fp_bits}"


def load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def load_literature_ranges(path: Path) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            ranges[row["benchmark"]] = (float(row["lit_min"]), float(row["lit_max"]))
    return ranges


def make_base_args(
    data_root: Path,
    output_dir: Path,
    tasks: list[str],
    radius: int,
    fp_bits: int,
    threads: int,
    seed: int,
    config: dict[str, object],
) -> Namespace:
    return Namespace(
        data_root=data_root,
        tasks=",".join(tasks),
        radius=radius,
        fp_bits=fp_bits,
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


def compute_run_metrics(
    summary_rows: list[dict[str, object]],
    benchmark_names: dict[str, str],
    lit_ranges: dict[str, tuple[float, float]],
    higher_is_better: bool,
) -> dict[str, float | int]:
    normalized_scores: list[float] = []
    within_range_count = 0

    for row in summary_rows:
        task = str(row["task"])
        value = float(row["test"])
        benchmark = benchmark_names[task]
        lit_min, lit_max = lit_ranges[benchmark]
        scale = lit_max - lit_min
        if scale == 0:
            norm = 0.0
        elif higher_is_better:
            norm = (value - lit_min) / scale
        else:
            norm = (lit_max - value) / scale
        normalized_scores.append(norm)
        if lit_min <= value <= lit_max:
            within_range_count += 1

    return {
        "within_range_count": within_range_count,
        "mean_normalized_score": mean(normalized_scores),
    }


def run_repeat(
    family: str,
    tasks: list[str],
    benchmark_names: dict[str, str],
    lit_ranges: dict[str, tuple[float, float]],
    higher_is_better: bool,
    config: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float | int]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    task_args = make_base_args(
        data_root=args.data_root,
        output_dir=run_dir,
        tasks=tasks,
        radius=args.radius,
        fp_bits=args.fp_bits,
        threads=args.threads,
        seed=seed,
        config=config,
    )

    print(f"\n[{family}] seed={seed} -> {run_dir}", flush=True)
    summaries: list[dict[str, object]] = []
    per_target: list[dict[str, object]] = []
    t0 = time.time()

    for task in tasks:
        task_t0 = time.time()
        summary, targets = run_task(task, task_args)
        summary["elapsed_sec"] = round(time.time() - task_t0, 3)
        summary["seed"] = seed
        summary["family"] = family
        summaries.append(summary)

        for row in targets:
            row["seed"] = seed
            row["family"] = family
        per_target.extend(targets)

        print(
            f"  {task}: test={float(summary['test']):.6f} elapsed={summary['elapsed_sec']}s",
            flush=True,
        )

    stats = compute_run_metrics(
        summaries,
        benchmark_names=benchmark_names,
        lit_ranges=lit_ranges,
        higher_is_better=higher_is_better,
    )
    stats["elapsed_sec_total"] = round(time.time() - t0, 3)

    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "per_target.csv", per_target)
    payload = {
        "family": family,
        "fingerprint": fingerprint_name(args.radius, args.fp_bits),
        "seed": seed,
        "config": config,
        "stats": stats,
        "summary": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return summaries, per_target, stats


def aggregate_repeats(
    tasks: list[str],
    benchmark_names: dict[str, str],
    repeat_rows: list[dict[str, object]],
    out_path: Path,
) -> list[dict[str, object]]:
    by_task: dict[str, list[float]] = {}
    meta: dict[str, dict[str, str]] = {}

    for row in repeat_rows:
        task = str(row["task"])
        by_task.setdefault(task, []).append(float(row["test"]))
        meta[task] = {
            "task_type": str(row["task_type"]),
            "metric": str(row["metric"]),
            "benchmark": benchmark_names[task],
        }

    stats_rows: list[dict[str, object]] = []
    for task in tasks:
        values = by_task[task]
        stats_rows.append(
            {
                "task": task,
                "benchmark": meta[task]["benchmark"],
                "task_type": meta[task]["task_type"],
                "metric": meta[task]["metric"],
                "mean_test": mean(values),
                "sd_test": stdev(values) if len(values) >= 2 else 0.0,
                "n_runs": len(values),
            }
        )

    write_csv(out_path, stats_rows)
    return stats_rows


def run_family(
    family: str,
    tasks: list[str],
    benchmark_names: dict[str, str],
    lit_ranges: dict[str, tuple[float, float]],
    higher_is_better: bool,
    config: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    family_root = args.output_root / family
    repeat_root = family_root / "best_repeats"
    repeat_rows: list[dict[str, object]] = []
    repeat_overview: list[dict[str, object]] = []

    for offset in range(args.repeats):
        seed = args.base_seed + offset
        run_dir = repeat_root / f"seed_{seed}"
        summary_rows, _, stats = run_repeat(
            family=family,
            tasks=tasks,
            benchmark_names=benchmark_names,
            lit_ranges=lit_ranges,
            higher_is_better=higher_is_better,
            config=config,
            seed=seed,
            args=args,
            run_dir=run_dir,
        )
        repeat_rows.extend(summary_rows)
        repeat_overview.append(
            {
                "seed": seed,
                "fingerprint": fingerprint_name(args.radius, args.fp_bits),
                **stats,
            }
        )

    repeat_root.mkdir(parents=True, exist_ok=True)
    write_csv(repeat_root / "repeat_summary_all.csv", repeat_rows)
    write_csv(repeat_root / "repeat_overview.csv", repeat_overview)

    task_stats = aggregate_repeats(
        tasks=tasks,
        benchmark_names=benchmark_names,
        repeat_rows=repeat_rows,
        out_path=repeat_root / "best_5seed_task_stats.csv",
    )

    summary = {
        "family": family,
        "fingerprint": fingerprint_name(args.radius, args.fp_bits),
        "config": config,
        "repeat_overview": repeat_overview,
        "task_stats": task_stats,
    }
    (family_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"data-root does not exist: {args.data_root}")

    classification_config = load_json(args.classification_config)
    regression_config = load_json(args.regression_config)
    classification_ranges = load_literature_ranges(args.classification_ranges)
    regression_ranges = load_literature_ranges(args.regression_ranges)

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {args.data_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Fingerprint: {fingerprint_name(args.radius, args.fp_bits)}", flush=True)
    print(f"Repeats: {args.repeats}, base seed: {args.base_seed}", flush=True)

    classification_summary = run_family(
        family="classification",
        tasks=CLASSIFICATION_TASKS,
        benchmark_names=CLASSIFICATION_BENCHMARKS,
        lit_ranges=classification_ranges,
        higher_is_better=True,
        config=classification_config,
        args=args,
    )
    regression_summary = run_family(
        family="regression",
        tasks=REGRESSION_TASKS,
        benchmark_names=REGRESSION_BENCHMARKS,
        lit_ranges=regression_ranges,
        higher_is_better=False,
        config=regression_config,
        args=args,
    )

    payload = {
        "fingerprint": fingerprint_name(args.radius, args.fp_bits),
        "radius": args.radius,
        "fp_bits": args.fp_bits,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "classification": classification_summary,
        "regression": regression_summary,
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2))

    print("\nDone.", flush=True)
    print(f"Summary JSON: {args.output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
