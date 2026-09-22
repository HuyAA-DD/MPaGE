"""Evaluate the best strict-evaluator SEMO heuristic from the 2026-09-22 run.

``select_neighbor`` is preserved verbatim from the best-scoring MPaGE program.
The surrounding evaluator uses the same integer-coercing permutation constraint
as ``semo_strict.py`` so the algorithms can be compared under one protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pymoo.indicators.hv import HV

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData


HISTORICAL_MEAN_HV = 232.82370912625106
HISTORICAL_SCORE = [-232.82370912625106, 0.3124297857284546]
HISTORICAL_SOURCE = "../logs/20260922_113201_Problem_MPaGE/population/pop_1.json"
HISTORICAL_RECORD_INDEX = 1


def select_neighbor(archive: List[Tuple[np.ndarray, Tuple[float, float]]], instance: np.ndarray, distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution from it.

    Args:
    archive: List of (solution, objective) pairs. Each solution is a numpy array of node IDs.
             Each objective is a tuple of two float values (cost in each space).
    instance: Numpy array of shape (N, 4). Each row corresponds to a node and contains its coordinates in two 2D spaces: (x1, y1, x2, y2).
    distance_matrix_1: Distance matrix in the first objective space.
    distance_matrix_2: Distance matrix in the second objective space.

    Returns:
    A new neighbor solution (numpy array).
    """
    total_cost = sum(1 / (1 + obj[0] + obj[1]) for _, obj in archive)
    probabilities = [(1 / (1 + obj[0] + obj[1])) / total_cost for _, obj in archive]
    selected_index = np.random.choice(range(len(archive)), p=probabilities)
    current_solution = archive[selected_index][0].copy()
    
    # Step 2: Generate a neighbor solution using a hybrid local search method
    n = len(current_solution)
    if n < 4:  # Need at least 4 nodes to perform a meaningful perturbation
        return current_solution
    
    # Randomly select two indices to swap
    idx1, idx2 = np.random.choice(n, size=2, replace=False)
    
    # Randomly select a segment to reverse
    start, end = sorted(np.random.choice(n, size=2, replace=False))
    segment = current_solution[start:end + 1][::-1]
    
    # Create a new neighbor solution
    neighbor_solution = np.concatenate((current_solution[:start], segment, current_solution[end + 1:]))
    
    return neighbor_solution


def check_constraint(solution: np.ndarray, problem_size: int) -> bool:
    """Use the same integer-coercing permutation constraint as semo_strict.py."""
    try:
        candidate = np.asarray(solution, dtype=int)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        candidate.ndim == 1
        and len(candidate) == problem_size
        and np.array_equal(np.sort(candidate), np.arange(problem_size))
    )


def tour_cost(
    solution: np.ndarray, distances: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    if not check_constraint(solution, distances[0].shape[0]):
        raise ValueError("tour_cost received an invalid TSP permutation after integer coercion")
    solution = np.asarray(solution, dtype=int)
    following = np.roll(solution, -1)
    return tuple(float(matrix[solution, following].sum()) for matrix in distances)


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def random_solution(problem_size: int) -> np.ndarray:
    solution = list(range(problem_size))
    random.shuffle(solution)
    return np.asarray(solution, dtype=int)


def pareto_indices(objectives: list[tuple[float, float]]) -> list[int]:
    return [
        index
        for index, objective in enumerate(objectives)
        if not any(
            other_index != index and dominates(other, objective)
            for other_index, other in enumerate(objectives)
        )
    ]


def solve_instance(
    instance: np.ndarray,
    distances: tuple[np.ndarray, np.ndarray],
    problem_size: int,
    initial_solutions: int,
    iterations: int,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    tours = [random_solution(problem_size) for _ in range(initial_solutions)]
    archive = [(tour, tour_cost(tour, distances)) for tour in tours]
    objective_evaluations = initial_solutions
    accepted_offspring = 0
    rejected_offspring = 0

    started = time.perf_counter()
    for _ in range(iterations):
        candidate = select_neighbor(archive, instance, distances[0], distances[1])
        if not check_constraint(candidate, problem_size):
            rejected_offspring += 1
            continue
        candidate = np.asarray(candidate, dtype=int)
        objective = tour_cost(candidate, distances)
        objective_evaluations += 1
        if not any(dominates(old_objective, objective) for _, old_objective in archive):
            archive = [
                (tour, old_objective)
                for tour, old_objective in archive
                if not dominates(objective, old_objective)
            ]
            archive.append((candidate, objective))
            accepted_offspring += 1
    runtime = time.perf_counter() - started

    objectives = [objective for _, objective in archive]
    front = pareto_indices(objectives)
    return {
        "seed": seed,
        "attempted_offspring": iterations,
        "accepted_offspring": accepted_offspring,
        "rejected_offspring": rejected_offspring,
        "objective_evaluations": objective_evaluations,
        "runtime_seconds": runtime,
        "objectives": [list(objectives[index]) for index in front],
        "tours": [archive[index][0].tolist() for index in front],
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    datasets = GetData(args.instances, args.cities).generate_instances()
    reference_point = np.asarray(args.reference_point, dtype=float)
    hv_indicator = HV(ref_point=reference_point)
    runs = []
    for index, (instance, distance_1, distance_2) in enumerate(datasets):
        run = solve_instance(
            instance,
            (distance_1, distance_2),
            args.cities,
            args.initial_solutions,
            args.iterations,
            args.seed + index,
        )
        run["instance"] = index
        run["hypervolume"] = float(hv_indicator(np.asarray(run["objectives"])))
        run["pareto_front_size"] = len(run["objectives"])
        runs.append(run)

    return {
        "algorithm": "SEMO-2209 with best strict-evaluator MPaGE heuristic",
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "reference_point": reference_point.tolist(),
        "historical_mean_hypervolume": HISTORICAL_MEAN_HV,
        "historical_score": HISTORICAL_SCORE,
        "historical_program_source": HISTORICAL_SOURCE,
        "historical_record_index": HISTORICAL_RECORD_INDEX,
        "initial_solutions": args.initial_solutions,
        "iterations": args.iterations,
        "instances": runs,
        "mean_hypervolume": float(np.mean([run["hypervolume"] for run in runs])),
        "mean_runtime_seconds": float(np.mean([run["runtime_seconds"] for run in runs])),
    }


def print_results(result: dict) -> None:
    for run in result["instances"]:
        print(
            f"Instance {run['instance']}: HV={run['hypervolume']:.6f}, "
            f"accepted={run['accepted_offspring']}, rejected={run['rejected_offspring']}, "
            f"objective_evaluations={run['objective_evaluations']}"
        )
    print(f"Mean HV: {result['mean_hypervolume']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--cities", type=int, default=20)
    parser.add_argument("--initial-solutions", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--reference-point", type=float, nargs=2, default=(20.0, 20.0))
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "results" / "semo2209_bi_tsp20.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_results(result)
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
