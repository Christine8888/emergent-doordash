"""GPQA experiments: all hint type / solver / mode combinations."""

from experiments.base_experiment import Experiment
from environments.gpqa.gpqa import gpqa_diamond, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, prefill, generate
from utils.submitit_utils import launch_experiment
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, GEMMA_MODELS, LLAMA_MODELS
from utils.setup import setup_logging
import os
with open("/sphinx/u/cye/emergent-doordash/hf.tok", 'r') as f:
    os.environ['HF_TOKEN']=f.read().strip()

logger = setup_logging()

BASE_DIR = "/sphinx/u/cye/emergent-doordash/christine_experiments/data"


def make_experiment(hint_type: str, solver_type: str, mode: str = "sequential"):
    _name = f"{hint_type}_{solver_type}_{mode}"
    _data_path = f"{BASE_DIR}/{hint_type}/gpqa.jsonl"

    class _Experiment(Experiment):
        name = _name
        eval_name = "gpqa"
        data_path = _data_path

        def build_task(self, hint_fraction: float, sample_ids: set[str]):
            config = PrefillConfig(
                path=_data_path,
                fraction=hint_fraction,
                mode=mode,
            )
            if solver_type == "intext":
                hint_solver = intext(config, prefix="Here is part of a hint that may be helpful to your solution:\n")
            else:
                hint_solver = prefill(config)

            solver = [
                instructions(DEFAULT_INSTRUCTIONS),
                hint_solver,
                generate(timeout=self.timeout),
            ]
            return gpqa_diamond(sample_ids=sample_ids, solver=solver)

    return _Experiment

EXPERIMENTS = {
    "cot_intext_sequential": make_experiment("cot", "intext", "sequential"),
    "cot_intext_masked": make_experiment("cot", "intext", "masked"),
    "solution_intext_sequential": make_experiment("solution", "intext", "sequential"),
    "solution_intext_masked": make_experiment("solution", "intext", "masked"),
    "cot_prefill_sequential": make_experiment("cot", "prefill", "sequential"),
    "solution_prefill_sequential": make_experiment("solution", "prefill", "sequential"),
}

HINT_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HINT_FRACTIONS += [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def run_experiment(exp_name: str, epochs: int, results_dir: str):
    """Run a single experiment with full retry logic."""
    logger.info(f"Starting {exp_name}...")
    launch_experiment(
        experiment_class=EXPERIMENTS[exp_name],
        models=QWEN3_MODELS + QWEN25_MODELS + GEMMA_MODELS + LLAMA_MODELS,
        hint_fractions=HINT_FRACTIONS,
        epochs=epochs,
        results_dir=results_dir,
        wait=True,
        poll_interval=300,
        max_retries=3,
    )


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    experiments_to_run = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    with ThreadPoolExecutor(max_workers=len(experiments_to_run)) as executor:
        futures = [
            executor.submit(run_experiment, exp_name, args.epochs, args.results_dir)
            for exp_name in experiments_to_run
        ]
        for future in futures:
            future.result()
