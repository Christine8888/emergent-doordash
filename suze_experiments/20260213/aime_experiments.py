"""AIME experiments: all hint type / solver / mode combinations."""

from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SETUP_ENV_SCRIPT = REPO_ROOT / "scripts" / "setup_env_suze.sh"

from experiments.base_experiment import Experiment
from environments.aime.aime import aime, DEFAULT_INSTRUCTIONS
from evals.prefill import PrefillConfig
from evals.solvers import instructions, intext, prefill, generate
from utils.submitit_utils import launch_experiment, run_specs_throttled
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, GEMMA_MODELS, LLAMA_MODELS, ModelSpec, SC_LOPRIO_PARTITION
from utils.submitit_defaults import DEFAULT_CONFIG
from utils.setup import setup_logging

logger = setup_logging()

HF_TOKEN_PATH = "/nlp/scr/suzeva/hf.tok"
with open(HF_TOKEN_PATH, "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()

BASE_DIR = str(REPO_ROOT / "christine_experiments" / "data")
MODELS = QWEN3_MODELS + QWEN25_MODELS + GEMMA_MODELS + LLAMA_MODELS



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
                generate(timeout=self.timeout),
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


def _apply_sc_loprio(models: list[ModelSpec]) -> list[ModelSpec]:
    """Route all models to sc-loprio, using SLURM constraints instead of nodelists."""
    from dataclasses import replace as dc_replace
    return [dc_replace(m, partitions=SC_LOPRIO_PARTITION, nodelist="") for m in models]


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


def _ckpt_path(output_path: Path) -> Path:
    """Return the .ckpt.json path corresponding to an output .json path."""
    return output_path.with_suffix(".ckpt.json")


def _read_ckpt_progress(ckpt_path: Path) -> tuple[int, int] | None:
    """Return (completed_instances, total_instances) from a checkpoint file, or None if unreadable."""
    if not ckpt_path.exists():
        return None
    try:
        import json
        with open(ckpt_path) as f:
            data = json.load(f)
        completed = data.get("completed_samples")
        total = data.get("total_samples")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            return completed, total
    except Exception:
        pass
    return None


def plan_runs(
    experiment_names: list[str],
    results_dir: str,
):
    total_jobs = 0
    total_existing = 0
    total_missing = 0
    total_progress_jobs = 0.0  # each done job = 1.0, each in-progress job = fraction done

    for exp_name in experiment_names:
        exp_cls = EXPERIMENTS[exp_name]
        exp_existing = 0
        exp_missing = 0

        for model in MODELS:
            model_name = os.path.basename(model.path)
            missing_hint_fractions: list[float] = []
            existing_for_model = 0
            missing_for_model = 0
            inprog_completed = 0
            inprog_total = 0
            inprog_count = 0

            for fewshot in [0]:
                for hint_fraction in HINT_FRACTIONS:
                    total_jobs += 1
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
                        total_progress_jobs += 1.0
                    else:
                        exp_missing += 1
                        missing_for_model += 1
                        missing_hint_fractions.append(hint_fraction)
                        progress = _read_ckpt_progress(_ckpt_path(out))
                        if progress is not None:
                            inprog_count += 1
                            inprog_completed += progress[0]
                            inprog_total += progress[1]
                            total_progress_jobs += progress[0] / progress[1]

            if missing_for_model:
                missing_hint_fractions = sorted(set(missing_hint_fractions))
                if len(missing_hint_fractions) == len(HINT_FRACTIONS):
                    missing_display = "ALL"
                else:
                    missing_display = str(missing_hint_fractions)
                inprog_str = ""
                if inprog_count:
                    pct = 100 * inprog_completed / inprog_total if inprog_total else 0
                    inprog_str = f" | in_progress={inprog_count} ({pct:.0f}% each avg)"
                print(
                    f"{exp_name} / {model_name}: "
                    f"done={existing_for_model} missing={missing_for_model}"
                    f"{inprog_str} | "
                    f"missing_hint_fractions={missing_display}"
                )

        total_existing += exp_existing
        total_missing += exp_missing
        print(f"{exp_name}: existing={exp_existing} missing={exp_missing} (models_counted={len(MODELS)})")

    overall_pct = 100 * total_progress_jobs / total_jobs if total_jobs else 0
    print(
        f"TOTAL: {total_progress_jobs:.1f} / {total_jobs} jobs completed "
        f"({overall_pct:.1f}%) | done={total_existing} missing={total_missing} "
        f"(models_counted={len(MODELS)})"
    )


def run_experiment(
    exp_name: str,
    epochs: int,
    results_dir: str,
    debug: bool = False,
    max_jobs: int | None = None,
    *,
    max_connections: int | None = None,
    num_gpus: int | None = None,
    cluster: str | None = None,
    low_prio: bool = False,
    sc_loprio: bool = False,
):
    """Run a single experiment with full retry logic."""
    logger.info(f"Starting {exp_name}...")
    hint_type, solver_type, mode = EXPERIMENT_SPECS[exp_name]
    experiment_cls = make_experiment(hint_type, solver_type, mode)
    models = _restrict_models_to_cluster(MODELS, cluster) if cluster is not None else MODELS
    config_overrides = {}
    # Ensure submitit workers source this repo's setup script (activates conda env "ed").
    config_overrides["setup_commands"] = [f"source {SETUP_ENV_SCRIPT}"]
    if max_connections is not None:
        config_overrides["max_connections"] = max_connections
    if cluster is not None:
        config_overrides["account"] = _CLUSTER_ACCOUNT.get(cluster, "nlp")
    if sc_loprio:
        models = _apply_sc_loprio(models)
    elif low_prio:
        models = _apply_low_prio(models)
    job_config = DEFAULT_CONFIG.override(**config_overrides)
    if max_jobs is not None:
        specs = launch_experiment(
            experiment_class=experiment_cls,
            models=models,
            hint_fractions=HINT_FRACTIONS,
            epochs=epochs,
            results_dir=results_dir,
            config=job_config,
            wait=True,
            poll_interval=300,
            max_retries=3,
            debug=debug,
            submit=False,
            num_gpus=num_gpus,
        )
        logger.info(f"{exp_name}: throttled submit with max_jobs={max_jobs} ({len(specs)} specs)")
        run_specs_throttled(
            specs=specs,
            max_concurrent=max_jobs,
            config=job_config,
            poll_interval=300,
            max_retries=3,
        )
    else:
        launch_experiment(
            experiment_class=experiment_cls,
            models=models,
            hint_fractions=HINT_FRACTIONS,
            epochs=epochs,
            results_dir=results_dir,
            config=job_config,
            wait=True,
            poll_interval=300,
            max_retries=3,
            debug=debug,
            num_gpus=num_gpus,
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
    parser.add_argument(
        "--max_connections",
        type=int,
        default=None,
        help=(
            "Inspect max concurrent connections per job. "
            "Default (when omitted): auto by GPU (H200/H100=96, other=64)."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help=(
            "Requested GPUs per job. vLLM data_parallel_size is inferred as num_gpus / model.tp. "
            "Must be divisible by model.tp for every selected model."
        ),
    )
    parser.add_argument(
        "--checkpoint_chunk_instances",
        type=int,
        default=128,
        help=(
            "Checkpoint chunk size in instances (sample*epoch). "
            "Values below 128 are clamped to 128."
        ),
    )
    parser.add_argument(
        "--enable_checkpoint",
        action="store_true",
        help="Enable resumable checkpointing/chunking (default: disabled).",
    )
    parser.add_argument(
        "--disable_checkpoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cluster", choices=["sphinx", "miso", "jag"], default=None, help="Restrict job scheduling to a specific cluster (default: all clusters)")
    parser.add_argument("--low_prio", action="store_true", help="Submit jobs at low priority QOS (default: standard)")
    parser.add_argument("--sc_loprio", action="store_true", help="Submit pre-emptible jobs to sc-loprio partition using GPU constraints (overrides --cluster/--low_prio routing)")
    args = parser.parse_args()

    experiments_to_run = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    if args.plan:
        plan_runs(
            experiments_to_run,
            args.results_dir,
        )
    else:
        if args.enable_checkpoint and args.disable_checkpoint:
            raise ValueError("Cannot pass both --enable_checkpoint and --disable_checkpoint")
        # Default behavior: no checkpoint chunking unless explicitly enabled.
        if args.enable_checkpoint:
            os.environ["EXPERIMENT_CHECKPOINT_CHUNK_INSTANCES"] = str(max(args.checkpoint_chunk_instances, 128))
            os.environ.pop("EXPERIMENT_DISABLE_CHECKPOINT", None)
        else:
            os.environ["EXPERIMENT_DISABLE_CHECKPOINT"] = "1"

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
                    num_gpus=args.num_gpus,
                    cluster=args.cluster,
                    low_prio=args.low_prio,
                    sc_loprio=args.sc_loprio,
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

TOTAL: 156.8 / 336 jobs completed (46.7%) | done=151 missing=185 (models_counted=16) at feb 18, 23:22


TESTING
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 \
  --max_connections 12 \
  --checkpoint_chunk_instances 128

DEBUG
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 \
  --checkpoint_chunk_instances 128 \
  --debug

LOW PRIORITY SPHINX
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 2 \
  --max_connections 12 \
  --checkpoint_chunk_instances 128 \
  --low_prio \
  --cluster sphinx


MISO
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 8 \
  --max_connections 16 \
  --checkpoint_chunk_instances 128 \
  --cluster miso


SC-LOPRIO (pre-emptible, any cluster, GPU constraint routing)
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 200 \
  --max_connections 16 \
  --checkpoint_chunk_instances 128 \
  --sc_loprio



sacctmgr show assoc user=suzeva format=user,account,partition,qos
"""

"""

MISO NON-PREEMPTIBLE, DP=8
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 \
  --max_connections 96 \
  --cluster miso \
  --num_gpus 8 \
  --disable_checkpoint


USE THIS FOR NON-PREEMPTIBLE
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 1 \
  --cluster sphinx \
  --disable_checkpoint

"""