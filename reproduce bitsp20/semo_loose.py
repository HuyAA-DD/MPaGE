"""Reproduce MPaGE's historical best-HV SEMO with its original loose checks.

Both ``select_neighbor`` and ``check_constraint`` are copied verbatim from
their MPaGE sources. The original evaluator also converts node identifiers to
``int`` only while calculating tour costs, so this file preserves that behavior.
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


HISTORICAL_MEAN_HV = 271.28645227989284
HISTORICAL_SOURCE = "../logs/20260910_095825_Problem_MPaGE/population/pop_18.json"


def select_neighbor(archive: List[Tuple[np.ndarray, Tuple[float, float]]], instance: np.ndarray, distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> np.ndarray:
    import numpy as np
    import random

    # {The new algorithm first calculates a weighted score for each solution based on its objectives to select a solution with potential for improvement; then, it randomly selects a segment of the tour that includes a consecutive sequence of nodes and reverses that segment to explore a different configuration; subsequently, it computes a new objective value for the modified tour, ensuring that the new configuration remains a valid tour, and finally randomly perturbs the positions of a few other nodes in the tour to maintain diversity without violating the tour integrity.}

    total_cost = sum(1 / (obj[1][0] + obj[1][1]) for obj in archive)
    selection_probs = [(1 / (obj[1][0] + obj[1][1])) / total_cost for obj in archive]
    selected_index = np.random.choice(len(archive), p=selection_probs)
    selected_solution = archive[selected_index][0].copy()

    n = len(selected_solution)

    # Randomly select a segment of the tour (consecutive nodes)
    segment_start = random.randint(0, n - 3)  # Ensure at least 3 nodes to reverse
    segment_end = segment_start + random.randint(2, n - segment_start)  # At least 2 nodes in the segment

    # Reverse the selected segment
    neighbor_solution = selected_solution.tolist()
    neighbor_solution[segment_start:segment_end] = reversed(neighbor_solution[segment_start:segment_end])

    # Applying random perturbation to a few nodes
    perturbation_indices = random.sample(range(n), k=min(3, n))
    for idx in perturbation_indices:
        perturbation = random.uniform(-0.5, 0.5)  # Slight adjustment
        neighbor_solution[idx] = (neighbor_solution[idx] + perturbation) % n  # Ensure it's a valid node ID

    return np.array(neighbor_solution)


def tour_cost(instance, solution, problem_size):

        cost_1 = 0
        cost_2 = 0
        
        for j in range(problem_size - 1):
            node1, node2 = int(solution[j]), int(solution[j + 1])
            
            coord_1_node1, coord_2_node1 = instance[node1][:2], instance[node1][2:]
            coord_1_node2, coord_2_node2 = instance[node2][:2], instance[node2][2:]

            cost_1 += np.linalg.norm(coord_1_node1 - coord_1_node2)
            cost_2 += np.linalg.norm(coord_2_node1 - coord_2_node2)
        
        node_first, node_last = int(solution[0]), int(solution[-1])
        
        coord_1_first, coord_2_first = instance[node_first][:2], instance[node_first][2:]
        coord_1_last, coord_2_last = instance[node_last][:2], instance[node_last][2:]

        cost_1 += np.linalg.norm(coord_1_last - coord_1_first)
        cost_2 += np.linalg.norm(coord_2_last - coord_2_first)

        return cost_1, cost_2  
    

def dominates(a, b):
        """True if a dominates b (minimization)."""
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(problem_size):
        sol = list(range(problem_size))
        random.shuffle(sol)
        return np.array(sol)


def check_constraint(solution, problem_size):
    sol = list(solution)
    if len(sol) != problem_size:
        return False
    if len(set(sol)) != problem_size:
        return False
    if not all(0 <= x < problem_size for x in solution):
        return False
    return True


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
    archive = [(tour, tour_cost(instance, tour, problem_size)) for tour in tours]
    objective_evaluations = initial_solutions
    accepted_offspring = 0
    rejected_offspring = 0

    started = time.perf_counter()
    for _ in range(iterations):
        candidate = select_neighbor(archive, instance, distances[0], distances[1])
        if not check_constraint(candidate, problem_size):
            rejected_offspring += 1
            continue
        objective = tour_cost(instance, candidate, problem_size)
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


def load_baseline_hv(path: Path, expected_algorithm: str) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("algorithm") != expected_algorithm:
        raise ValueError(f"Expected {expected_algorithm} results in {path}")
    return {int(row["instance"]): float(row["hypervolume"]) for row in data["instances"]}


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

    result = {
        "algorithm": "SEMO with original MPaGE loose feasibility",
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "reference_point": reference_point.tolist(),
        "historical_mean_hypervolume_from_original_evaluator": HISTORICAL_MEAN_HV,
        "historical_program_source": HISTORICAL_SOURCE,
        "initial_solutions": args.initial_solutions,
        "iterations": args.iterations,
        "instances": runs,
        "mean_hypervolume": float(np.mean([run["hypervolume"] for run in runs])),
        "mean_runtime_seconds": float(np.mean([run["runtime_seconds"] for run in runs])),
    }

    if args.nsga_results.exists() and args.moead_results.exists():
        nsga = load_baseline_hv(args.nsga_results, "NSGA-II")
        moead_result = load_baseline_hv(args.moead_results, "MOEA/D")
        comparison = []
        for run in runs:
            index = run["instance"]
            values = {
                "SEMO-loose": run["hypervolume"],
                "NSGA-II": nsga[index],
                "MOEA/D": moead_result[index],
            }
            comparison.append({"instance": index, **values, "winner": max(values, key=values.get)})
        result["comparison"] = comparison
        mean_values = {
            name: float(np.mean([row[name] for row in comparison]))
            for name in ("SEMO-loose", "NSGA-II", "MOEA/D")
        }
        result["comparison_mean"] = {
            **mean_values,
            "winner": max(mean_values, key=mean_values.get),
        }
    return result


def print_results(result: dict) -> None:
    for run in result["instances"]:
        print(
            f"Instance {run['instance']}: accepted={run['accepted_offspring']}, "
            f"rejected={run['rejected_offspring']}, objective_evaluations="
            f"{run['objective_evaluations']}"
        )
    print(f"Historical mean HV (original evaluator): {HISTORICAL_MEAN_HV:.6f}")
    print(f"Reproduced loose mean HV: {result['mean_hypervolume']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--cities", type=int, default=20)
    parser.add_argument("--initial-solutions", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--reference-point", type=float, nargs=2, default=(20.0, 20.0))
    parser.add_argument(
        "--nsga-results", type=Path, default=SCRIPT_DIR / "results" / "nsga_bi_tsp20.json"
    )
    parser.add_argument(
        "--moead-results", type=Path, default=SCRIPT_DIR / "results" / "moead_bi_tsp20.json"
    )
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "results" / "semo_loose_bi_tsp20.json"
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
