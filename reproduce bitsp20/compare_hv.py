"""Compare MPaGE, NSGA-II, and MOEA/D hypervolume on each bi-TSP instance.

MPaGE's existing log stores only ``score[0] = -mean_hypervolume`` for a
heuristic, not one hypervolume per instance.  This script selects the logged
heuristic with the best historical mean HV and re-evaluates it on every
instance with deterministic random seeds before comparing it with the saved
NSGA-II and MOEA/D results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
from pymoo.indicators.hv import HV

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4ad.task.optimization.bi_tsp_semo.evaluation import (
    check_constraint,
    dominates,
    random_solution,
    tour_cost,
)
from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData


def latest_mpage_log(logs_dir: Path) -> Path:
    candidates = sorted(path for path in logs_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No MPaGE run directory found in {logs_dir}")
    return candidates[-1]


def iter_log_records(log_dir: Path):
    for path in sorted(log_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = data if isinstance(data, list) else [data]
        for record in records:
            if isinstance(record, dict):
                yield path, record


def find_best_hv_program(log_dir: Path) -> dict[str, Any]:
    """Select the valid logged program with the smallest -mean-HV score."""
    best: dict[str, Any] | None = None
    for source, record in iter_log_records(log_dir):
        score = record.get("score")
        program = record.get("function")
        if not (
            isinstance(score, (list, tuple))
            and score
            and isinstance(score[0], (int, float))
            and np.isfinite(score[0])
            and isinstance(program, str)
            and program.strip()
        ):
            continue
        if best is None or float(score[0]) < best["score"][0]:
            best = {
                "score": [float(value) for value in score],
                "program": program,
                "algorithm_description": record.get("algorithm"),
                "sample_order": record.get("sample_order"),
                "source": str(source),
            }
    if best is None:
        raise ValueError(f"No evaluated MPaGE program with a numeric score found in {log_dir}")
    return best


def compile_select_neighbor(program: str) -> Callable:
    namespace = {
        "np": np,
        "random": random,
        "List": List,
        "Tuple": Tuple,
    }
    exec(program, namespace)  # The program comes from the user's local MPaGE log.
    function = namespace.get("select_neighbor")
    if not callable(function):
        raise ValueError("The selected MPaGE program does not define select_neighbor")
    return function


def evaluate_mpage_instance(
    select_neighbor: Callable,
    instance: np.ndarray,
    distance_1: np.ndarray,
    distance_2: np.ndarray,
    problem_size: int,
    reference_point: np.ndarray,
    initial_solutions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Reproduce the current BITSPEvaluation loop for one instance."""
    random.seed(seed)
    np.random.seed(seed)
    initial = [random_solution(problem_size) for _ in range(initial_solutions)]
    archive = [(tour, tour_cost(instance, tour, problem_size)) for tour in initial]
    rejected = 0
    strictly_invalid = 0

    for _ in range(iterations):
        candidate = select_neighbor(archive, instance, distance_1, distance_2)
        candidate_array = np.asarray(candidate)
        is_strict_permutation = (
            candidate_array.ndim == 1
            and len(candidate_array) == problem_size
            and np.issubdtype(candidate_array.dtype, np.number)
            and np.all(np.isfinite(candidate_array))
            and np.all(candidate_array == np.round(candidate_array))
            and np.array_equal(np.sort(candidate_array.astype(int)), np.arange(problem_size))
        )
        if not is_strict_permutation:
            strictly_invalid += 1
        if not check_constraint(candidate, problem_size):
            rejected += 1
            continue
        objective = tour_cost(instance, candidate, problem_size)
        if not any(dominates(old_objective, objective) for _, old_objective in archive):
            archive = [
                (tour, old_objective)
                for tour, old_objective in archive
                if not dominates(objective, old_objective)
            ]
            archive.append((candidate, objective))

    objectives = np.asarray([objective for _, objective in archive], dtype=float)
    return {
        "hypervolume": float(HV(ref_point=reference_point)(objectives)),
        "archive_size": len(archive),
        "rejected_candidates": rejected,
        "strictly_invalid_candidates": strictly_invalid,
    }


def load_baseline(path: Path, expected_algorithm: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("algorithm") != expected_algorithm:
        raise ValueError(
            f"Expected algorithm {expected_algorithm!r} in {path}, got {data.get('algorithm')!r}"
        )
    instances = data.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"No instance results found in {path}")
    return data


def baseline_hv_by_instance(data: dict[str, Any]) -> dict[int, float]:
    return {int(row["instance"]): float(row["hypervolume"]) for row in data["instances"]}


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    log_dir = args.mpage_log or latest_mpage_log(args.logs_dir)
    best = find_best_hv_program(log_dir)
    select_neighbor = compile_select_neighbor(best["program"])
    nsga = load_baseline(args.nsga, "NSGA-II")
    moead = load_baseline(args.moead, "MOEA/D")

    nsga_hv = baseline_hv_by_instance(nsga)
    moead_hv = baseline_hv_by_instance(moead)
    instance_ids = sorted(set(nsga_hv) & set(moead_hv))
    if not instance_ids:
        raise ValueError("NSGA-II and MOEA/D result files have no common instances")
    expected_ids = list(range(len(instance_ids)))
    if instance_ids != expected_ids:
        raise ValueError(f"Expected consecutive instance IDs {expected_ids}, got {instance_ids}")

    nsga_ref = tuple(map(float, nsga["reference_point"]))
    moead_ref = tuple(map(float, moead["reference_point"]))
    if nsga_ref != moead_ref:
        raise ValueError(f"Reference-point mismatch: NSGA-II {nsga_ref}, MOEA/D {moead_ref}")
    reference_point = np.asarray(nsga_ref, dtype=float)

    problem_size = int(nsga.get("problem", "bi-TSP20").removeprefix("bi-TSP"))
    datasets = GetData(len(instance_ids), problem_size).generate_instances()
    rows = []
    for index, (instance, distance_1, distance_2) in enumerate(datasets):
        mpage = evaluate_mpage_instance(
            select_neighbor,
            instance,
            distance_1,
            distance_2,
            problem_size,
            reference_point,
            args.initial_solutions,
            args.iterations,
            args.seed + index,
        )
        values = {
            "MPaGE": mpage["hypervolume"],
            "NSGA-II": nsga_hv[index],
            "MOEA/D": moead_hv[index],
        }
        winner = max(values, key=values.get)
        rows.append(
            {
                "instance": index,
                "seed": args.seed + index,
                "mpage_hv": values["MPaGE"],
                "nsga_hv": values["NSGA-II"],
                "moead_hv": values["MOEA/D"],
                "winner": winner,
                "mpage_archive_size": mpage["archive_size"],
                "mpage_rejected_candidates": mpage["rejected_candidates"],
                "mpage_strictly_invalid_candidates": mpage["strictly_invalid_candidates"],
            }
        )

    means = {
        "mpage_hv": float(np.mean([row["mpage_hv"] for row in rows])),
        "nsga_hv": float(np.mean([row["nsga_hv"] for row in rows])),
        "moead_hv": float(np.mean([row["moead_hv"] for row in rows])),
    }
    mean_winner = max(
        {"MPaGE": means["mpage_hv"], "NSGA-II": means["nsga_hv"], "MOEA/D": means["moead_hv"]},
        key={"MPaGE": means["mpage_hv"], "NSGA-II": means["nsga_hv"], "MOEA/D": means["moead_hv"]}.get,
    )
    means["winner"] = mean_winner

    return {
        "problem": f"bi-TSP{problem_size}",
        "data_seed": 2025,
        "reference_point": reference_point.tolist(),
        "mpage": {
            "log_directory": str(log_dir),
            "best_program_source": best["source"],
            "sample_order": best["sample_order"],
            "historical_score": best["score"],
            "historical_mean_hv": -best["score"][0],
            "reevaluation_seed": args.seed,
            "initial_solutions_per_instance": args.initial_solutions,
            "iterations_per_instance": args.iterations,
        },
        "baseline_files": {"nsga": str(args.nsga), "moead": str(args.moead)},
        "instances": rows,
        "mean": means,
        "note": (
            "MPaGE per-instance HV is a deterministic re-evaluation of the historically "
            "best-mean-HV program because the original log stores only its mean HV. "
            "mpage_strictly_invalid_candidates counts outputs that the current MPaGE "
            "constraint check accepts but that are not integer TSP permutations."
        ),
    }


def print_table(result: dict[str, Any]) -> None:
    header = f"{'Instance':>8} | {'MPaGE':>12} | {'NSGA-II':>12} | {'MOEA/D':>12} | Winner"
    print(header)
    print("-" * len(header))
    for row in result["instances"]:
        print(
            f"{row['instance']:>8} | {row['mpage_hv']:>12.6f} | "
            f"{row['nsga_hv']:>12.6f} | {row['moead_hv']:>12.6f} | {row['winner']}"
        )
    mean = result["mean"]
    print("-" * len(header))
    print(
        f"{'Mean':>8} | {mean['mpage_hv']:>12.6f} | {mean['nsga_hv']:>12.6f} | "
        f"{mean['moead_hv']:>12.6f} | {mean['winner']}"
    )
    print(f"MPaGE historical best mean HV: {result['mpage']['historical_mean_hv']:.6f}")
    invalid = sum(row["mpage_strictly_invalid_candidates"] for row in result["instances"])
    if invalid:
        print(
            f"WARNING: MPaGE produced {invalid} candidates that are not strict integer "
            "permutations but were accepted by the current evaluator."
        )


def write_csv(path: Path, result: dict[str, Any]) -> None:
    fieldnames = ["instance", "seed", "mpage_hv", "nsga_hv", "moead_hv", "winner"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["instances"])
        writer.writerow(
            {
                "instance": "mean",
                "mpage_hv": result["mean"]["mpage_hv"],
                "nsga_hv": result["mean"]["nsga_hv"],
                "moead_hv": result["mean"]["moead_hv"],
                "winner": result["mean"]["winner"],
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpage-log", type=Path, default=None, help="MPaGE run directory; default: latest")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument(
        "--nsga", type=Path, default=SCRIPT_DIR / "results" / "nsga_bi_tsp20.json"
    )
    parser.add_argument(
        "--moead", type=Path, default=SCRIPT_DIR / "results" / "moead_bi_tsp20.json"
    )
    parser.add_argument("--seed", type=int, default=2025, help="Base seed for MPaGE re-evaluation")
    parser.add_argument("--initial-solutions", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--json-output", type=Path, default=SCRIPT_DIR / "results" / "hv_comparison.json"
    )
    parser.add_argument(
        "--csv-output", type=Path, default=SCRIPT_DIR / "results" / "hv_comparison.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comparison(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.csv_output, result)
    print_table(result)
    print(f"Saved JSON to {args.json_output}")
    print(f"Saved CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
