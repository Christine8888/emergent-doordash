"""AIME experiments: all hint type / solver / mode combinations."""

from experiments.base_experiment import Experiment
from environments.aime.aime import aime, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, prefill, generate
from utils.submitit_utils import launch_experiment
from dataclasses import replace
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, GEMMA_MODELS, LLAMA_MODELS, LARGE_MODEL_PARTITIONS
from utils.setup import setup_logging

logger = setup_logging()

BASE_DIR = "/sphinx/u/cye/emergent-doordash/christine_experiments/data"


def make_experiment(hint_type: str, solver_type: str, mode: str = "sequential"):
    _name = f"{hint_type}_{solver_type}_{mode}"
    _data_path = f"{BASE_DIR}/{hint_type}/aime.jsonl"

    class _Experiment(Experiment):
        name = _name
        eval_name = "aime"
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
                generate(max_tokens=8192, timeout=self.timeout),
            ]
            return aime(sample_ids=sample_ids, solver=solver)

    return _Experiment

EXPERIMENTS = {
    "solution_intext_sequential": make_experiment("solution", "intext", "sequential"),
    "solution_intext_masked": make_experiment("solution", "intext", "masked"),
    #"solution_prefill_sequential": make_experiment("solution", "prefill", "sequential"),
}

HINT_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HINT_FRACTIONS += [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def _apply_miso(models):
    """Set account=miso on models whose partitions include miso."""
    return [
        replace(m, account="miso") if m.partitions == LARGE_MODEL_PARTITIONS else m
        for m in models
    ]


def run_experiment(exp_name: str, epochs: int, results_dir: str, debug: bool = False, miso: bool = False):
    """Run a single experiment with full retry logic."""
    models = QWEN3_MODELS + QWEN25_MODELS + GEMMA_MODELS + LLAMA_MODELS
    if miso:
        models = _apply_miso(models)
    logger.info(f"Starting {exp_name}...")
    launch_experiment(
        experiment_class=EXPERIMENTS[exp_name],
        models=models,
        hint_fractions=HINT_FRACTIONS,
        epochs=epochs,
        results_dir=results_dir,
        wait=True,
        poll_interval=300,
        max_retries=3,
        debug=debug,
    )


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--debug", action="store_true", help="Enable Inspect HTTP debug logging")
    parser.add_argument("--miso", action="store_true", help="Use account=miso for large model jobs")
    args = parser.parse_args()

    experiments_to_run = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    with ThreadPoolExecutor(max_workers=len(experiments_to_run)) as executor:
        futures = [
            executor.submit(run_experiment, exp_name, args.epochs, args.results_dir, args.debug, args.miso)
            for exp_name in experiments_to_run
        ]
        for future in futures:
            future.result()
