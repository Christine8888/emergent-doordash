"""AIME experiments: all hint type / solver / mode combinations."""

from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.base_experiment import Experiment
from environments.aime.aime import aime, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, prefill, generate
from utils.submitit_utils import launch_experiment
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, GEMMA_MODELS, LLAMA_MODELS, ModelSpec
from utils.submitit_defaults import DEFAULT_CONFIG
from utils.setup import setup_logging

logger = setup_logging()

HF_TOKEN_PATH = "/afs/cs.stanford.edu/u/suzeva/hf.tok"
with open(HF_TOKEN_PATH, "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()

BASE_DIR = str(REPO_ROOT / "suze_experiments" / "data")
MODELS = QWEN3_MODELS + QWEN25_MODELS + GEMMA_MODELS + LLAMA_MODELS

FEWSHOTS = [0]


def make_experiment(hint_type: str, solver_type: str, mode: str = "sequential", *, max_tokens: int = 8192):
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
                generate(max_tokens=max_tokens, timeout=self.timeout),
            ]
            return aime(sample_ids=sample_ids, solver=solver)

    return _Experiment

EXPERIMENT_SPECS: dict[str, tuple[str, str, str]] = {
    # name -> (hint_type, solver_type, mode)
    # "solution_intext_sequential": ("solution", "intext", "sequential"),
    "solution_intext_masked": ("solution", "intext", "masked"),
    # "solution_prefill_sequential": ("solution", "prefill", "sequential"),
}

EXPERIMENTS = {name: make_experiment(*spec) for name, spec in EXPERIMENT_SPECS.items()}

HINT_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HINT_FRACTIONS += [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# Mapping from --cluster argument to SLURM partition name and node-name prefix.
_CLUSTER_PARTITION = {"sphinx": "sphinx", "miso": "miso", "jag": "jag-standard"}
_CLUSTER_NODE_PREFIX = {"sphinx": "sphinx", "miso": "miso", "jag": "jagupard"}
# Low-priority partition names (nlprun uses these as --partition, not --qos).
_LOW_PRIO_PARTITION = {"sphinx": "sphinx-lo", "miso": "miso-lo", "jag-standard": "jag-lo"}
# SLURM account to use per cluster (miso partition requires account=miso).
_CLUSTER_ACCOUNT = {"sphinx": "nlp", "miso": "miso", "jag": "nlp"}


def _apply_low_prio(models: list[ModelSpec]) -> list[ModelSpec]:
    """Swap each model's partitions to their low-priority equivalents."""
    from dataclasses import replace as dc_replace
    result = []
    for model in models:
        low_parts = [_LOW_PRIO_PARTITION.get(p, p) for p in model.partitions.split(",")]
        result.append(dc_replace(model, partitions=",".join(low_parts)))
    return result


def _restrict_models_to_cluster(models: list[ModelSpec], cluster: str) -> list[ModelSpec]:
    """Return a copy of each ModelSpec restricted to only nodes in *cluster*.

    Raises ValueError if a model has no nodes in the requested cluster (e.g. a
    32B model requested on jag which has no H200s).
    """
    from dataclasses import replace as dc_replace
    partition = _CLUSTER_PARTITION[cluster]
    prefix = _CLUSTER_NODE_PREFIX[cluster]
    restricted = []
    for model in models:
        nodes = [n for n in model.nodelist.split(",") if n.startswith(prefix)]
        if not nodes:
            raise ValueError(
                f"Model {os.path.basename(model.path)!r} has no nodes in cluster "
                f"'{cluster}' (nodelist={model.nodelist!r}). Choose a different cluster."
            )
        restricted.append(dc_replace(model, partitions=partition, nodelist=",".join(nodes)))
    return restricted


def _output_path(
    experiment_class: type[Experiment],
    results_dir: str,
    model_name: str,
    fewshot: int,
    hint_fraction: float,
) -> Path:
    # Mirrors Experiment.get_output_filename, but does not create directories.
    output_dir = Path(results_dir) / experiment_class.eval_name / experiment_class.name / f"{fewshot}shot" / model_name
    filename = f"{experiment_class.eval_name}_{experiment_class.name}_{fewshot}shot_{hint_fraction}.json"
    return output_dir / filename


def plan_runs(
    experiment_names: list[str],
    results_dir: str,
):
    total_existing = 0
    total_missing = 0

    for exp_name in experiment_names:
        exp_cls = EXPERIMENTS[exp_name]
        exp_existing = 0
        exp_missing = 0

        for model in MODELS:
            model_name = os.path.basename(model.path)
            missing_hint_fractions: list[float] = []
            existing_for_model = 0
            missing_for_model = 0

            for fewshot in FEWSHOTS:
                for hint_fraction in HINT_FRACTIONS:
                    out = _output_path(
                        experiment_class=exp_cls,
                        results_dir=results_dir,
                        model_name=model_name,
                        fewshot=fewshot,
                        hint_fraction=hint_fraction,
                    )
                    if out.exists():
                        exp_existing += 1
                        existing_for_model += 1
                    else:
                        exp_missing += 1
                        missing_for_model += 1
                        missing_hint_fractions.append(hint_fraction)

            if missing_for_model:
                missing_hint_fractions = sorted(set(missing_hint_fractions))
                if len(missing_hint_fractions) == len(HINT_FRACTIONS):
                    missing_display = "ALL"
                else:
                    missing_display = str(missing_hint_fractions)
                print(
                    f"{exp_name} / {model_name}: "
                    f"existing={existing_for_model} missing={missing_for_model} "
                    f"missing_hint_fractions={missing_display}"
                )

        total_existing += exp_existing
        total_missing += exp_missing
        print(f"{exp_name}: existing={exp_existing} missing={exp_missing} (models_counted={len(MODELS)})")

    print(f"TOTAL: existing={total_existing} missing={total_missing} (models_counted={len(MODELS)})")


def run_experiment(
    exp_name: str,
    epochs: int,
    results_dir: str,
    debug: bool = False,
    max_jobs: int | None = None,
    *,
    max_connections: int = 16,
    max_tokens: int = 8192,
    cluster: str | None = None,
    low_prio: bool = False,
):
    """Run a single experiment with full retry logic."""
    logger.info(f"Starting {exp_name}...")
    hint_type, solver_type, mode = EXPERIMENT_SPECS[exp_name]
    experiment_cls = make_experiment(hint_type, solver_type, mode, max_tokens=max_tokens)
    models = _restrict_models_to_cluster(MODELS, cluster) if cluster is not None else MODELS
    config_overrides = dict(max_connections=max_connections)
    if cluster is not None:
        config_overrides["account"] = _CLUSTER_ACCOUNT.get(cluster, "nlp")
    models = _apply_low_prio(models) if low_prio else models
    launch_experiment(
        experiment_class=experiment_cls,
        models=models,
        hint_fractions=HINT_FRACTIONS,
        epochs=epochs,
        results_dir=results_dir,
        config=DEFAULT_CONFIG.override(**config_overrides),
        wait=True,
        poll_interval=300,
        max_retries=3,
        debug=debug,
        max_jobs=max_jobs,
    )


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plan", action="store_true", help="Print existing vs missing outputs under results_dir, then exit")
    parser.add_argument("--debug", action="store_true", help="Enable Inspect HTTP debug logging")
    parser.add_argument("--max_jobs", type=int, default=None, help="Maximum number of jobs to submit (default: no limit)")
    parser.add_argument("--max_connections", type=int, default=16, help="Inspect max concurrent connections per job (default: 16)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens per generation (default: 8192)")
    parser.add_argument("--cluster", choices=["sphinx", "miso", "jag"], default=None, help="Restrict job scheduling to a specific cluster (default: all clusters)")
    parser.add_argument("--low_prio", action="store_true", help="Submit jobs at low priority QOS (default: standard)")
    args = parser.parse_args()

    experiments_to_run = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    if args.plan:
        plan_runs(
            experiments_to_run,
            args.results_dir,
        )
    else:
        with ThreadPoolExecutor(max_workers=len(experiments_to_run)) as executor:
            futures = [
                executor.submit(
                    run_experiment,
                    exp_name,
                    args.epochs,
                    args.results_dir,
                    args.debug,
                    args.max_jobs,
                    max_connections=args.max_connections,
                    max_tokens=args.max_tokens,
                    cluster=args.cluster,
                    low_prio=args.low_prio,
                )
                for exp_name in experiments_to_run
            ]
            for future in futures:
                future.result()



"""
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results --plan

TOTAL: existing=150 missing=186 (models_counted=16)

TESTING
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 \
  --max_connections 12 

DEBUG
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 --debug

LOW PRIORITY SPHINX
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 2 \
  --max_connections 12 \
  --low_prio \
  --cluster sphinx


MISO
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 8 \
  --max_connections 16 \
  --cluster miso

sacctmgr show assoc user=suzeva format=user,account,partition,qos
"""