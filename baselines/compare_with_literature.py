#!/usr/bin/env python3
"""Compare baseline summary.csv against literature range CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TASK_TO_BENCHMARK = {
    "bbbp": "BBBP",
    "bace": "BACE",
    "clintox": "ClinTox",
    "tox21": "Tox21",
    "toxcast": "ToxCast",
    "sider": "SIDER",
    "hiv": "HIV",
    "muv": "MUV",
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipo": "Lipo",
    "qm7dft": "QM7",
    "qm8dft": "QM8",
    "qm9dft": "QM9",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--classification-ranges", type=Path, required=True)
    p.add_argument("--regression-ranges", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def load_lit_ranges(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = (float(row["lit_min"]), float(row["lit_max"]))
    return out


def main() -> None:
    args = parse_args()

    lit = {}
    lit.update(load_lit_ranges(args.classification_ranges))
    lit.update(load_lit_ranges(args.regression_ranges))

    out_rows = []
    with args.summary_csv.open() as f:
        for row in csv.DictReader(f):
            task = row["task"]
            benchmark = TASK_TO_BENCHMARK[task]
            lit_min, lit_max = lit[benchmark]
            value = float(row["test"])
            kind = "classification" if task in CLASSIFICATION_TASKS else "regression"

            if kind == "classification":
                normalized = (value - lit_min) / (lit_max - lit_min)
                gap_to_best_lit = lit_max - value
            else:
                normalized = (lit_max - value) / (lit_max - lit_min)
                gap_to_best_lit = value - lit_min

            out_rows.append(
                {
                    "task": task,
                    "benchmark": benchmark,
                    "kind": kind,
                    "metric": row["metric"],
                    "test_value": value,
                    "lit_min": lit_min,
                    "lit_max": lit_max,
                    "within_range": lit_min <= value <= lit_max,
                    "normalized_score": normalized,
                    "gap_to_best_lit": gap_to_best_lit,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_within = sum(int(r["within_range"]) for r in out_rows)
    mean_norm = sum(float(r["normalized_score"]) for r in out_rows) / len(out_rows)
    print(f"Wrote {args.output_csv}")
    print(f"Within-range: {n_within}/{len(out_rows)}")
    print(f"Mean normalized score: {mean_norm:.6f}")


if __name__ == "__main__":
    main()
