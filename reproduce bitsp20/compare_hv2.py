"""Compare NSGA-II, strict/loose SEMO, and MOEA/D HV on bi-TSP20.

All four variants are rerun on the same deterministic problem instances.
NSGA-II and MOEA/D candidates are guarded as strict integer permutations.
The two SEMO variants use the constraint behavior implemented by their own
``semo_strict.py`` and ``semo_loose.py`` modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pymoo.indicators.hv import HV


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import moead
import nsga  # nsgaii.py is the compatibility entrypoint for this implementation.
import semo_loose
import semo_strict
from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData


def is_strict_tour(candidate: np.ndarray, cities: int) -> bool:
    """Return True only for an integer permutation of all city identifiers."""
    tour = np.asarray(candidate)
    return bool(
        tour.ndim == 1
        and len(tour) == cities
        and np.issubdtype(tour.dtype, np.integer)
        and np.array_equal(np.sort(tour), np.arange(cities))
    )


def require_strict_tour(candidate: np.ndarray, cities: int, algorithm: str) -> None:
    if not is_strict_tour(candidate, cities):
        tour = np.asarray(candidate)
        raise ValueError(
            f"{algorithm} generated an invalid candidate: "
            f"shape={tour.shape}, dtype={tour.dtype}, values={tour.tolist()}"
        )


def run_with_strict_objective_guard(
    module,
    solve: Callable,
    distances: tuple[np.ndarray, np.ndarray],
    cities: int,
    algorithm: str,
    **solve_kwargs,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Run NSGA-II/MOEA-D while validating every evaluated candidate."""
    original_evaluate = module.evaluate_tour
    checked_candidates = 0

    def strict_evaluate(
        tour: np.ndarray, matrices: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        nonlocal checked_candidates
        require_strict_tour(tour, cities, algorithm)
        checked_candidates += 1
        return original_evaluate(tour, matrices)

    module.evaluate_tour = strict_evaluate
    try:
        tours, objectives, evaluations = solve(distances=distances, cities=cities, **solve_kwargs)
    finally:
        module.evaluate_tour = original_evaluate

    if checked_candidates != evaluations:
        raise RuntimeError(
            f"{algorithm} reported {evaluations} evaluations, but the strict guard "
            f"observed {checked_candidates}"
        )
    for tour in tours:
        require_strict_tour(tour, cities, algorithm)
    return tours, objectives, evaluations, checked_candidates


def run_nsga(
    distances: tuple[np.ndarray, np.ndarray], args: argparse.Namespace, seed: int
) -> dict:
    started = time.perf_counter()
    tours, objectives, evaluations, checked = run_with_strict_objective_guard(
        nsga,
        nsga.solve_instance,
        distances,
        args.cities,
        "NSGA-II",
        population_size=args.population,
        generations=args.generations,
        seed=seed,
        mutation_probability=args.mutation_probability,
    )
    return {
        "runtime_seconds": time.perf_counter() - started,
        "evaluations": evaluations,
        "attempted_candidates": checked,
        "valid_candidates": checked,
        "rejected_candidates": 0,
        "pareto_front_size": len(objectives),
        "objectives": np.asarray(objectives, dtype=float),
        "tours": np.asarray(tours),
    }


def run_moead(
    distances: tuple[np.ndarray, np.ndarray], args: argparse.Namespace, seed: int
) -> dict:
    started = time.perf_counter()
    tours, objectives, evaluations, checked = run_with_strict_objective_guard(
        moead,
        moead.solve_instance,
        distances,
        args.cities,
        "MOEA/D",
        population_size=args.population,
        generations=args.generations,
        seed=seed,
        neighborhood_size=args.neighborhood,
        neighborhood_mating_probability=args.neighborhood_mating_probability,
        max_replacements=args.max_replacements,
        mutation_probability=args.mutation_probability,
    )
    return {
        "runtime_seconds": time.perf_counter() - started,
        "evaluations": evaluations,
        "attempted_candidates": checked,
        "valid_candidates": checked,
        "rejected_candidates": 0,
        "pareto_front_size": len(objectives),
        "objectives": np.asarray(objectives, dtype=float),
        "tours": np.asarray(tours),
    }


def run_semo(
    module,
    algorithm: str,
    instance: np.ndarray,
    distances: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
    seed: int,
    require_strict_output: bool,
) -> dict:
    result = module.solve_instance(
        instance,
        distances,
        args.cities,
        args.initial_solutions,
        args.iterations,
        seed,
    )
    tours = np.asarray(result["tours"])
    if require_strict_output:
        for tour in tours:
            require_strict_tour(tour, args.cities, algorithm)
    result["attempted_candidates"] = args.initial_solutions + result["attempted_offspring"]
    result["valid_candidates"] = result["objective_evaluations"]
    result["pareto_front_size"] = len(result["objectives"])
    result["objectives"] = np.asarray(result["objectives"], dtype=float)
    result["tours"] = tours
    return result


def serializable_run(result: dict) -> dict:
    converted = dict(result)
    converted["objectives"] = np.asarray(result["objectives"]).tolist()
    converted["tours"] = np.asarray(result["tours"]).tolist()
    return converted


def build_comparison(args: argparse.Namespace) -> dict:
    if args.population * args.generations != args.iterations:
        raise ValueError(
            "For an equal offspring-attempt budget, population * generations "
            "must equal SEMO iterations"
        )
    if args.population != args.initial_solutions:
        raise ValueError("NSGA-II/MOEA-D population must equal SEMO initial solutions")

    reference_point = np.asarray(args.reference_point, dtype=float)
    hv = HV(ref_point=reference_point)
    datasets = GetData(args.instances, args.cities).generate_instances()
    rows = []
    detailed_runs = []

    for index, (instance, distance_1, distance_2) in enumerate(datasets):
        seed = args.seed + index
        distances = (distance_1, distance_2)
        results = {
            "NSGA-II": run_nsga(distances, args, seed),
            "SEMO-strict": run_semo(
                semo_strict, "SEMO-strict", instance, distances, args, seed, True
            ),
            "SEMO-loose": run_semo(
                semo_loose, "SEMO-loose", instance, distances, args, seed, False
            ),
            "MOEA/D": run_moead(distances, args, seed),
        }
        values = {}
        for name, result in results.items():
            value = float(hv(np.asarray(result["objectives"], dtype=float)))
            result["hypervolume"] = value
            values[name] = value

        rows.append(
            {
                "instance": index,
                "seed": seed,
                "nsga_hv": values["NSGA-II"],
                "semo_strict_hv": values["SEMO-strict"],
                "semo_loose_hv": values["SEMO-loose"],
                "moead_hv": values["MOEA/D"],
                "winner": max(values, key=values.get),
                "nsga_rejected": results["NSGA-II"]["rejected_candidates"],
                "semo_strict_rejected": results["SEMO-strict"]["rejected_offspring"],
                "semo_loose_rejected": results["SEMO-loose"]["rejected_offspring"],
                "moead_rejected": results["MOEA/D"]["rejected_candidates"],
            }
        )
        detailed_runs.append(
            {
                "instance": index,
                "seed": seed,
                "algorithms": {
                    name: serializable_run(result) for name, result in results.items()
                },
            }
        )

    mean = {
        "nsga_hv": float(np.mean([row["nsga_hv"] for row in rows])),
        "semo_strict_hv": float(np.mean([row["semo_strict_hv"] for row in rows])),
        "semo_loose_hv": float(np.mean([row["semo_loose_hv"] for row in rows])),
        "moead_hv": float(np.mean([row["moead_hv"] for row in rows])),
    }
    winner_values = {
        "NSGA-II": mean["nsga_hv"],
        "SEMO-strict": mean["semo_strict_hv"],
        "SEMO-loose": mean["semo_loose_hv"],
        "MOEA/D": mean["moead_hv"],
    }
    mean["winner"] = max(winner_values, key=winner_values.get)
    return {
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "algorithm_seed": args.seed,
        "implementations": {
            "NSGA-II": "nsgaii.py compatibility entrypoint backed by nsga.py",
            "SEMO-strict": "semo_strict.py",
            "SEMO-loose": "semo_loose.py",
            "MOEA/D": "moead.py",
        },
        "reference_point": reference_point.tolist(),
        "constraint_note": (
            "NSGA-II and MOEA/D use strict integer-permutation guards; each SEMO "
            "variant uses the check_constraint implementation in its own module."
        ),
        "budget": {
            "initial_candidates": args.initial_solutions,
            "offspring_attempts": args.iterations,
            "note": "Rejected candidates are not objective evaluations.",
        },
        "instances": rows,
        "mean": mean,
        "details": detailed_runs,
    }


def print_table(result: dict) -> None:
    header = (
        f"{'Instance':>8} | {'NSGA-II':>12} | {'SEMO-strict':>12} | "
        f"{'SEMO-loose':>12} | {'MOEA/D':>12} | Winner"
    )
    print(header)
    print("-" * len(header))
    for row in result["instances"]:
        print(
            f"{row['instance']:>8} | {row['nsga_hv']:>12.6f} | "
            f"{row['semo_strict_hv']:>12.6f} | {row['semo_loose_hv']:>12.6f} | "
            f"{row['moead_hv']:>12.6f} | {row['winner']}"
        )
    mean = result["mean"]
    print("-" * len(header))
    print(
        f"{'Mean':>8} | {mean['nsga_hv']:>12.6f} | "
        f"{mean['semo_strict_hv']:>12.6f} | {mean['semo_loose_hv']:>12.6f} | "
        f"{mean['moead_hv']:>12.6f} | {mean['winner']}"
    )


def write_csv(path: Path, result: dict) -> None:
    fieldnames = [
        "instance",
        "seed",
        "nsga_hv",
        "semo_strict_hv",
        "semo_loose_hv",
        "moead_hv",
        "winner",
        "nsga_rejected",
        "semo_strict_rejected",
        "semo_loose_rejected",
        "moead_rejected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["instances"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--cities", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
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
        default=SCRIPT_DIR / "results" / "hv_comparison_strict.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "hv_comparison_strict.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comparison(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.csv_output, result)
    print_table(result)
    print(f"Saved JSON to {args.json_output}")
    print(f"Saved CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
