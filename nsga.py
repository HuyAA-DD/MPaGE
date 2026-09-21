"""NSGA-II baseline for MPaGE's default 20-city bi-objective TSP.

The default run uses the same four instances, data seed, reference point, and
number of evaluated tours as ``BITSPEvaluation`` (100 initial tours followed by
2,000 offspring evaluations per instance).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from pymoo.indicators.hv import HV

from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData


def evaluate_tour(tour: np.ndarray, distances: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return the two closed-tour lengths for a permutation."""
    nxt = np.roll(tour, -1)
    return np.array([matrix[tour, nxt].sum() for matrix in distances], dtype=float)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Fast non-dominated sorting; return fronts and zero-based ranks."""
    size = len(objectives)
    dominates_set: list[list[int]] = [[] for _ in range(size)]
    dominated_count = np.zeros(size, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(size):
        for q in range(p + 1, size):
            if dominates(objectives[p], objectives[q]):
                dominates_set[p].append(q)
                dominated_count[q] += 1
            elif dominates(objectives[q], objectives[p]):
                dominates_set[q].append(p)
                dominated_count[p] += 1
    fronts[0] = np.flatnonzero(dominated_count == 0).tolist()

    rank = np.empty(size, dtype=int)
    level = 0
    while level < len(fronts) and fronts[level]:
        rank[fronts[level]] = level
        following: list[int] = []
        for p in fronts[level]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    following.append(q)
        if following:
            fronts.append(following)
        level += 1
    return fronts, rank


def crowding_distance(objectives: np.ndarray, front: Iterable[int]) -> dict[int, float]:
    indices = np.asarray(list(front), dtype=int)
    distance = np.zeros(len(indices), dtype=float)
    if len(indices) <= 2:
        distance[:] = np.inf
    else:
        for objective in range(objectives.shape[1]):
            order = np.argsort(objectives[indices, objective], kind="stable")
            distance[order[[0, -1]]] = np.inf
            low = objectives[indices[order[0]], objective]
            high = objectives[indices[order[-1]], objective]
            if high > low:
                distance[order[1:-1]] += (
                    objectives[indices[order[2:]], objective]
                    - objectives[indices[order[:-2]], objective]
                ) / (high - low)
    return {int(index): float(value) for index, value in zip(indices, distance)}


def rank_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts, ranks = non_dominated_sort(objectives)
    crowding = np.zeros(len(objectives), dtype=float)
    for front in fronts:
        for index, value in crowding_distance(objectives, front).items():
            crowding[index] = value
    return ranks, crowding


def tournament(rng: np.random.Generator, ranks: np.ndarray, crowding: np.ndarray) -> int:
    a, b = rng.integers(0, len(ranks), size=2)
    if ranks[a] != ranks[b]:
        return int(a if ranks[a] < ranks[b] else b)
    if crowding[a] != crowding[b]:
        return int(a if crowding[a] > crowding[b] else b)
    return int(a if rng.random() < 0.5 else b)


def order_crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """OX crossover, preserving a valid TSP permutation."""
    n = len(parent_a)
    left, right = sorted(rng.choice(n, size=2, replace=False))
    right += 1
    child = np.full(n, -1, dtype=int)
    child[left:right] = parent_a[left:right]
    used = set(child[left:right].tolist())
    fill_positions = list(range(right, n)) + list(range(0, left))
    candidates = np.concatenate((parent_b[right:], parent_b[:right]))
    for position, city in zip(fill_positions, (x for x in candidates if x not in used)):
        child[position] = city
    return child


def mutate(tour: np.ndarray, rng: np.random.Generator, probability: float) -> np.ndarray:
    child = tour.copy()
    if rng.random() < probability:
        left, right = sorted(rng.choice(len(child), size=2, replace=False))
        if rng.random() < 0.5:
            child[left : right + 1] = child[left : right + 1][::-1]
        else:
            child[left], child[right] = child[right], child[left]
    return child


def environmental_selection(
    population: np.ndarray, objectives: np.ndarray, population_size: int
) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = non_dominated_sort(objectives)
    selected: list[int] = []
    for front in fronts:
        remaining = population_size - len(selected)
        if len(front) <= remaining:
            selected.extend(front)
        else:
            distance = crowding_distance(objectives, front)
            selected.extend(sorted(front, key=lambda i: distance[i], reverse=True)[:remaining])
            break
    return population[selected].copy(), objectives[selected].copy()


def solve_instance(
    distances: tuple[np.ndarray, np.ndarray],
    cities: int,
    population_size: int,
    generations: int,
    seed: int,
    mutation_probability: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run NSGA-II and return its final non-dominated tours and objectives."""
    if population_size < 2 or generations < 0 or cities < 3:
        raise ValueError("population_size >= 2, generations >= 0 and cities >= 3 are required")
    rng = np.random.default_rng(seed)
    mutation_probability = 1.0 / cities if mutation_probability is None else mutation_probability
    population = np.array([rng.permutation(cities) for _ in range(population_size)])
    objectives = np.array([evaluate_tour(tour, distances) for tour in population])
    evaluations = population_size

    for _ in range(generations):
        ranks, crowding = rank_and_crowding(objectives)
        offspring = []
        while len(offspring) < population_size:
            first = population[tournament(rng, ranks, crowding)]
            second = population[tournament(rng, ranks, crowding)]
            child = order_crossover(first, second, rng)
            offspring.append(mutate(child, rng, mutation_probability))
        offspring_array = np.asarray(offspring)
        offspring_objectives = np.array(
            [evaluate_tour(tour, distances) for tour in offspring_array]
        )
        evaluations += len(offspring_array)
        population, objectives = environmental_selection(
            np.vstack((population, offspring_array)),
            np.vstack((objectives, offspring_objectives)),
            population_size,
        )

    first_front = non_dominated_sort(objectives)[0][0]
    return population[first_front], objectives[first_front], evaluations


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
        "algorithm": "NSGA-II",
        "problem": f"bi-TSP{args.cities}",
        "data_seed": 2025,
        "reference_point": list(map(float, args.reference_point)),
        "population_size": args.population,
        "generations": args.generations,
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
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--reference-point", type=float, nargs=2, default=(20.0, 20.0))
    parser.add_argument("--output", type=Path, default=Path("results/nsga_bi_tsp20.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"NSGA-II mean HV: {result['mean_hypervolume']:.6f}")
    print(f"NSGA-II mean runtime: {result['mean_runtime_seconds']:.6f} s")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
