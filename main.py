import os
from pathlib import Path

from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.method.LLMPFG import MPaGE
from llm4ad.method.LLMPFG import EoHProfiler

# If you want to run the bi_tsp_semo example, uncomment the following line:
from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation as ProblemEvaluation

# If you want to run the bi_tsp_semo example, uncomment the following line:
# from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation as ProblemEvaluation

# If you want to run the bi_cvrp example, uncomment the following line:
# from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation as ProblemEvaluation

# If you want to run the bi_kp example, uncomment the following line:
# from llm4ad.task.optimization.bi_kp import BIKPEvaluation as ProblemEvaluation

PROJECT_ROOT = Path(__file__).resolve().parent


def load_openai_api_key() -> str:
    """Load one OpenAI API key for both paper-specified models."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    secret_path = PROJECT_ROOT / "secret.txt"
    if secret_path.exists():
        api_key = secret_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY or place the key in MPaGE/secret.txt before running."
        )
    return api_key


def main():
    api_key = load_openai_api_key()

    # Paper setup: GPT-4o-mini (temperature 0.7) generates heuristics.
    llm = HttpsApiOpenAI(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        timeout=60,
    )
    # Paper setup: GPT-4o performs semantic assessment and clustering.
    llm_cluster = HttpsApiOpenAI4Cluster(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-4o",
        timeout=60,
    )
    task = ProblemEvaluation()

    method = MPaGE(
        llm=llm,
        llm_cluster=llm_cluster,
        profiler=EoHProfiler(log_dir="logs", log_style="complex"),
        evaluation=task,
        # Reduced outer heuristic-design budget. This does not change the
        # per-heuristic bi-TSP evaluation budget used by the saved baselines.
        max_sample_nums=60,
        max_generations=10,
        pop_size=6,
        selection_num=2,
        pfg_segments=4,
        selection_epsilon=0.9,
        mutation_probability=0.3,
        llm_review=True,
        num_samplers=1,
        num_evaluators=1,
    )

    method.run()


if __name__ == '__main__':
    main()


