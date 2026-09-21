"""MOEA/D baseline for MPaGE's default 20-city bi-objective TSP.

Defaults mirror ``BITSPEvaluation``: four deterministic instances, 100 initial
tours, and 2,000 offspring evaluations per instance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData
from nsga import dominates, evaluate_tour, mutate, non_dominated_sort, order_crossover


def update_archive(
    archive_tours: list[np.ndarray],
    archive_objectives: list[np.ndarray],
    tour: np.ndarray,
    objective: np.ndarray,
) -> None:
    """Insert a candidate into an unbounded external Pareto archive."""
    if any(np.array_equal(objective, old) or dominates(old, objective) for old in archive_objectives):
        return
    retained = [
        index for index, old in enumerate(archive_objectives) if not dominates(objective, old)
    ]
    archive_tours[:] = [archive_tours[index] for index in retained]
    archive_objectives[:] = [archive_objectives[index] for index in retained]
    archive_tours.append(tour.copy())
    archive_objectives.append(objective.copy())


def tchebycheff(objective: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    # A tiny positive weight keeps endpoint subproblems sensitive to both objectives.
    return float(np.max(np.maximum(weight, 1e-6) * np.abs(objective - ideal)))


def solve_instance(
    distances: tuple[np.ndarray, np.ndarray],
    cities: int,
    population_size: int,
    generations: int,
    seed: int,
    neighborhood_size: int = 20,
    neighborhood_mating_probability: float = 0.9,
    max_replacements: int = 2,
    mutation_probability: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run permutation MOEA/D with Tchebycheff decomposition."""
    if population_size < 2 or generations < 0 or cities < 3:
        raise ValueError("population_size >= 2, generations >= 0 and cities >= 3 are required")
    if not 1 <= neighborhood_size <= population_size:
        raise ValueError("neighborhood_size must be in [1, population_size]")
    if max_replacements < 1:
        raise ValueError("max_replacements must be positive")

    rng = np.random.default_rng(seed)
    mutation_probability = 1.0 / cities if mutation_probability is None else mutation_probability
    first_weights = np.linspace(0.0, 1.0, population_size)
    weights = np.column_stack((first_weights, 1.0 - first_weights))
    weight_distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhoods = np.argsort(weight_distances, axis=1)[:, :neighborhood_size]

    population = np.array([rng.permutation(cities) for _ in range(population_size)])
    objectives = np.array([evaluate_tour(tour, distances) for tour in population])
    evaluations = population_size
    ideal = objectives.min(axis=0)
    archive_tours: list[np.ndarray] = []
    archive_objectives: list[np.ndarray] = []
    for tour, objective in zip(population, objectives):
        update_archive(archive_tours, archive_objectives, tour, objective)

    for _ in range(generations):
        for subproblem in rng.permutation(population_size):
            use_neighborhood = rng.random() < neighborhood_mating_probability
            pool = neighborhoods[subproblem] if use_neighborhood else np.arange(population_size)
            replace = len(pool) < 2
            parent_indices = rng.choice(pool, size=2, replace=replace)
            child = order_crossover(
                population[parent_indices[0]], population[parent_indices[1]], rng
            )
            child = mutate(child, rng, mutation_probability)
            child_objective = evaluate_tour(child, distances)
            evaluations += 1
            ideal = np.minimum(ideal, child_objective)

            update_pool = neighborhoods[subproblem] if use_neighborhood else np.arange(population_size)
            update_pool = rng.permutation(update_pool)
            replacements = 0
            for neighbor in update_pool:
                if tchebycheff(child_objective, weights[neighbor], ideal) <= tchebycheff(
                    objectives[neighbor], weights[neighbor], ideal
                ):
                    population[neighbor] = child.copy()
                    objectives[neighbor] = child_objective
                    replacements += 1
                    if replacements >= max_replacements:
                        break
            update_archive(archive_tours, archive_objectives, child, child_objective)

    # Defensive filtering also makes the return contract explicit.
    archive_objectives_array = np.asarray(archive_objectives)
    first_front = non_dominated_sort(archive_objectives_array)[0][0]
    return (
        np.asarray(archive_tours)[first_front],
        archive_objectives_array[first_front],
        evaluations,
    )


def run_benchmark(args: argparse.Namespace) -> dict:
    instances = GetData(args.instances, args.cities).generate_instances()
    hv = HV(ref_point=np.asarray(args.reference_point, dtype=float))
    runs = []
    for index, (_, distance_1, distance_2) in enumerate(instances):
        started = time.perf_counter()
        tours, objectives, evaluations = solve_instance(
            (distance_1, distance_2),
            args.cities,
            args.population,
            args.generations,
            args.seed + index,
            args.neighborhood,
            args.neighborhood_mating_probability,
            args.max_replacements,
            args.mutation_probability,
        )
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "instance": index,
                "seed": args.seed + index,
                "evaluations": evaluations,
                "runtime_seconds": elapsed,
                "hypervolume": float(hv(objectives)),
                "pareto_front_size": len(objectives),
                "objectives": objectives.tolist(),
                "tours": tours.tolist(),
            }
        )
    return {
        "algorithm": "MOEA/D",
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "reference_point": list(map(float, args.reference_point)),
        "population_size": args.population,
        "generations": args.generations,
        "neighborhood_size": args.neighborhood,
        "neighborhood_mating_probability": args.neighborhood_mating_probability,
        "max_replacements": args.max_replacements,
        "mutation_probability": (
            1.0 / args.cities if args.mutation_probability is None else args.mutation_probability
        ),
        "instances": runs,
        "mean_hypervolume": float(np.mean([run["hypervolume"] for run in runs])),
        "mean_runtime_seconds": float(np.mean([run["runtime_seconds"] for run in runs])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--cities", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025, help="Algorithm RNG seed")
    parser.add_argument("--neighborhood", type=int, default=20)
    parser.add_argument("--neighborhood-mating-probability", type=float, default=0.9)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--reference-point", type=float, nargs=2, default=(20.0, 20.0))
    parser.add_argument("--output", type=Path, default=Path("results/moead_bi_tsp20.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MOEA/D mean HV: {result['mean_hypervolume']:.6f}")
    print(f"MOEA/D mean runtime: {result['mean_runtime_seconds']:.6f} s")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
