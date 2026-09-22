"""Repeat the bi-TSP20 HV comparison with multiple algorithm seeds.

By default, this pipeline performs 10 complete runs. Each run evaluates
NSGA-II, SEMO-strict, SEMO-loose, SEMO-2209, and MOEA/D on the same four deterministic
problem instances using a distinct base seed. Hypervolume uses the original
MPaGE reference point ``[20.0, 20.0]`` unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import compare_hv2


SCRIPT_DIR = Path(__file__).resolve().parent
ALGORITHMS = ("NSGA-II", "SEMO-strict", "SEMO-loose", "SEMO-2209", "MOEA/D")
MEAN_KEYS = {
    "NSGA-II": "nsga_hv",
    "SEMO-strict": "semo_strict_hv",
    "SEMO-loose": "semo_loose_hv",
    "SEMO-2209": "semo2209_hv",
    "MOEA/D": "moead_hv",
}


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if args.seed_step < 1:
        raise ValueError("seed-step must be at least 1")

    runs = []
    for run_index in range(args.runs):
        seed = args.base_seed + run_index * args.seed_step
        print(f"Running comparison {run_index + 1}/{args.runs} with base seed {seed}...", flush=True)

        run_args = copy.copy(args)
        run_args.seed = seed
        comparison = compare_hv2.build_comparison(run_args)
        hypervolumes = {
            algorithm: float(comparison["mean"][mean_key])
            for algorithm, mean_key in MEAN_KEYS.items()
        }
        winner = max(hypervolumes, key=hypervolumes.get)
        runs.append(
            {
                "run": run_index + 1,
                "base_seed": seed,
                "instance_seeds": [seed + index for index in range(args.instances)],
                "hypervolumes": hypervolumes,
                "winner": winner,
                "instances": comparison["instances"],
            }
        )

    win_counts = Counter(run["winner"] for run in runs)
    summary = {}
    for algorithm in ALGORITHMS:
        values = np.asarray(
            [run["hypervolumes"][algorithm] for run in runs], dtype=float
        )
        summary[algorithm] = {
            "mean_hypervolume": float(np.mean(values)),
            "sample_std_hypervolume": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median_hypervolume": float(np.median(values)),
            "min_hypervolume": float(np.min(values)),
            "max_hypervolume": float(np.max(values)),
            "wins": int(win_counts[algorithm]),
        }

    overall_winner = max(
        ALGORITHMS, key=lambda algorithm: summary[algorithm]["mean_hypervolume"]
    )
    return {
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "runs": args.runs,
        "base_seed": args.base_seed,
        "seed_step": args.seed_step,
        "reference_point": list(map(float, args.reference_point)),
        "instances_per_run": args.instances,
        "budget_per_instance": {
            "initial_candidates": args.initial_solutions,
            "offspring_attempts": args.iterations,
        },
        "algorithms": list(ALGORITHMS),
        "run_results": runs,
        "summary": summary,
        "overall_winner_by_mean_hypervolume": overall_winner,
        "note": (
            "SEMO-loose reproduces MPaGE's original permissive constraint and may "
            "score paths that are not valid integer TSP permutations."
        ),
    }


def print_run_table(result: dict) -> None:
    header = (
        f"{'Run':>4} | {'Seed':>8} | {'NSGA-II':>12} | {'SEMO-strict':>12} | "
        f"{'SEMO-loose':>12} | {'SEMO-2209':>12} | {'MOEA/D':>12} | Winner"
    )
    print(header)
    print("-" * len(header))
    for run in result["run_results"]:
        hv = run["hypervolumes"]
        print(
            f"{run['run']:>4} | {run['base_seed']:>8} | {hv['NSGA-II']:>12.6f} | "
            f"{hv['SEMO-strict']:>12.6f} | {hv['SEMO-loose']:>12.6f} | "
            f"{hv['SEMO-2209']:>12.6f} | {hv['MOEA/D']:>12.6f} | {run['winner']}"
        )

    print("-" * len(header))
    means = {name: result["summary"][name]["mean_hypervolume"] for name in ALGORITHMS}
    print(
        f"{'Mean':>4} | {'-':>8} | {means['NSGA-II']:>12.6f} | "
        f"{means['SEMO-strict']:>12.6f} | {means['SEMO-loose']:>12.6f} | "
        f"{means['SEMO-2209']:>12.6f} | {means['MOEA/D']:>12.6f} | "
        f"{result['overall_winner_by_mean_hypervolume']}"
    )
    stds = {name: result["summary"][name]["sample_std_hypervolume"] for name in ALGORITHMS}
    print(
        f"{'Std':>4} | {'-':>8} | {stds['NSGA-II']:>12.6f} | "
        f"{stds['SEMO-strict']:>12.6f} | {stds['SEMO-loose']:>12.6f} | "
        f"{stds['SEMO-2209']:>12.6f} | {stds['MOEA/D']:>12.6f} |"
    )


def write_run_csv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "base_seed",
        "nsga_hv",
        "semo_strict_hv",
        "semo_loose_hv",
        "semo2209_hv",
        "moead_hv",
        "winner",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in result["run_results"]:
            hv = run["hypervolumes"]
            writer.writerow(
                {
                    "run": run["run"],
                    "base_seed": run["base_seed"],
                    "nsga_hv": hv["NSGA-II"],
                    "semo_strict_hv": hv["SEMO-strict"],
                    "semo_loose_hv": hv["SEMO-loose"],
                    "semo2209_hv": hv["SEMO-2209"],
                    "moead_hv": hv["MOEA/D"],
                    "winner": run["winner"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--cities", type=int, default=20)
    parser.add_argument("--reference-point", type=float, nargs=2, default=(20.0, 20.0))
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--initial-solutions", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--neighborhood", type=int, default=20)
    parser.add_argument("--neighborhood-mating-probability", type=float, default=0.9)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "hv_pipeline_10_runs.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "hv_pipeline_10_runs.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_run_csv(args.csv_output, result)
    print_run_table(result)
    print(f"Saved JSON to {args.json_output}")
    print(f"Saved CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
