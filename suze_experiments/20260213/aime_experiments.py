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
# HINT_FRACTIONS += [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# Mapping from --cluster argument to SLURM partition name and node-name prefix.
_CLUSTER_PARTITION = {"sphinx": "sphinx", "miso": "miso", "jag": "jag-standard"}
_CLUSTER_NODE_PREFIX = {"sphinx": "sphinx", "miso": "miso", "jag": "jagupard"}
# Low-priority partition names (nlprun uses these as --partition, not --qos).
_LOW_PRIO_PARTITION = {"sphinx": "sphinx-lo", "miso": "miso-lo", "jag-standard": "jag-lo"}
# SLURM account to use per cluster (miso partition requires account=miso).
_CLUSTER_ACCOUNT = {"sphinx": "nlp", "miso": "miso", "jag": "nlp"}
# Cluster-specific walltime policy (hours). MISO must be capped at 6h.
_CLUSTER_TIME_HOURS = {"miso": 6}


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


def _filter_models(models: list[ModelSpec], model: str | None) -> list[ModelSpec]:
    """Filter models by full path or basename (e.g., Qwen/Qwen3-8B or Qwen3-8B)."""
    if model is None:
        return models
    selected = [m for m in models if m.path == model or os.path.basename(m.path) == model]
    if selected:
        return selected
    available = sorted({os.path.basename(m.path) for m in models})
    raise ValueError(
        f"Unknown model {model!r}. Pass full path (e.g. 'Qwen/Qwen3-8B') "
        f"or basename from: {available}"
    )


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


def _normalize_repo_relative_path(path: str | None) -> str | None:
    """Normalize path to repo-relative form when possible."""
    if not isinstance(path, str) or not path:
        return None
    norm = os.path.normpath(path).replace("\\", "/")
    marker = f"/{REPO_ROOT.name}/"
    if marker in norm:
        return norm.split(marker, 1)[1]
    return norm


def _dataset_path_key(path: str | None) -> str | None:
    """Return a stable dataset key for checkpoint compatibility checks."""
    if not isinstance(path, str) or not path:
        return None
    norm = os.path.normpath(path).replace("\\", "/")
    marker = "/data/"
    if marker in norm:
        return norm.split(marker, 1)[1]
    parts = [p for p in norm.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[0] if parts else None


def _read_ckpt_progress(ckpt_path: Path, *, expected_data_path: str | None = None) -> tuple[int, int] | None:
    """Return (completed_instances, total_instances) from a compatible checkpoint, or None."""
    if not ckpt_path.exists():
        return None
    try:
        import json
        with open(ckpt_path) as f:
            data = json.load(f)

        if expected_data_path is not None:
            meta = data.get("meta")
            ckpt_data_path = meta.get("data_path") if isinstance(meta, dict) else None
            ckpt_rel = _normalize_repo_relative_path(ckpt_data_path)
            exp_rel = _normalize_repo_relative_path(expected_data_path)
            if not (ckpt_rel is not None and exp_rel is not None and ckpt_rel == exp_rel):
                ckpt_key = _dataset_path_key(ckpt_data_path)
                exp_key = _dataset_path_key(expected_data_path)
                if ckpt_key is None or exp_key is None or ckpt_key != exp_key:
                    return None

        completed = data.get("completed_samples")
        total = data.get("total_samples")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            return completed, total
    except Exception:
        pass
    return None


def _normalize_results_dir(path: str | None) -> str | None:
    """Normalize a results_dir path into a stable absolute string."""
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def _get_active_experiment_configs() -> set[tuple[str, str, int, float, str | None]]:
    """Return active (exp_name, model_name, fewshot, hint_fraction, results_dir) tuples."""
    import pickle
    import subprocess

    active_configs: set[tuple[str, str, int, float, str | None]] = set()
    submitit_folder = Path(DEFAULT_CONFIG.submitit_folder)
    if not submitit_folder.is_absolute():
        submitit_folder = REPO_ROOT / submitit_folder

    try:
        user = os.environ.get("USER", "")
        cmd = ["squeue", "-h", "-o", "%i"]
        if user:
            cmd[1:1] = ["-u", user]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            fallback = subprocess.run(["squeue", "-h", "-o", "%i"], capture_output=True, text=True, timeout=5)
            if fallback.returncode != 0:
                logger.warning("Failed to get SLURM queue, proceeding without running/queued filter")
                return active_configs
            result = fallback
        active_job_ids = set(result.stdout.strip().split())
    except Exception as exc:
        logger.warning(f"Failed to query SLURM queue ({exc}); proceeding without running/queued filter")
        return active_configs

    for job_id in active_job_ids:
        submitted_file = submitit_folder / f"{job_id}_submitted.pkl"
        if not submitted_file.exists():
            continue
        try:
            with open(submitted_file, "rb") as f:
                job_info = pickle.load(f)
            if not hasattr(job_info, "kwargs"):
                continue
            kwargs = job_info.kwargs
            model_path = kwargs.get("model_path", "")
            model_name = os.path.basename(model_path)
            fewshot = kwargs.get("fewshot")
            hint_fraction = kwargs.get("hint_fraction")
            experiment_class = kwargs.get("experiment_class")
            exp_name = getattr(experiment_class, "name", None)
            results_dir = _normalize_results_dir(kwargs.get("results_dir"))
            if (
                exp_name
                and model_name
                and isinstance(fewshot, int)
                and isinstance(hint_fraction, (float, int))
            ):
                active_configs.add((exp_name, model_name, fewshot, float(hint_fraction), results_dir))
        except Exception:
            # Best effort only; ignore unreadable pickle entries.
            pass

    return active_configs


def plan_runs(
    experiment_names: list[str],
    results_dir: str,
    model: str | None = None,
):
    models = _filter_models(MODELS, model)
    normalized_results_dir = _normalize_results_dir(results_dir)
    active_configs = _get_active_experiment_configs()
    total_jobs = 0
    total_existing = 0
    total_missing = 0
    total_progress_jobs = 0.0  # each done job = 1.0, each in-progress job = fraction done
    actionable_missing: dict[str, dict[str, list[float]]] = {}

    for exp_name in experiment_names:
        exp_cls = EXPERIMENTS[exp_name]
        exp_existing = 0
        exp_missing = 0

        for model_spec in models:
            model_name = os.path.basename(model_spec.path)
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
                        key = (exp_name, model_name, fewshot, float(hint_fraction), normalized_results_dir)
                        if key not in active_configs:
                            actionable_missing.setdefault(exp_name, {}).setdefault(model_name, []).append(hint_fraction)
                        progress = _read_ckpt_progress(
                            _ckpt_path(out),
                            expected_data_path=exp_cls.data_path,
                        )
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
        print(f"{exp_name}: existing={exp_existing} missing={exp_missing} (models_counted={len(models)})")

    overall_pct = 100 * total_progress_jobs / total_jobs if total_jobs else 0
    print(
        f"TOTAL: {total_progress_jobs:.1f} / {total_jobs} jobs completed "
        f"({overall_pct:.1f}%) | done={total_existing} missing={total_missing} "
        f"(models_counted={len(models)})"
    )

    print()
    print("MISSING AND NOT RUNNING/QUEUED:")
    if not actionable_missing:
        print("None.")
        return
    for exp_name in experiment_names:
        exp_missing = actionable_missing.get(exp_name, {})
        for model_name in sorted(exp_missing.keys()):
            hints = sorted(set(exp_missing[model_name]))
            print(
                f"{exp_name} / {model_name}: "
                f"missing_not_running_hint_fractions={hints}"
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
    cpus_per_task: int | None = None,
    mem_gb: int | None = None,
    cluster: str | None = None,
    low_prio: bool = False,
    sc_loprio: bool = False,
    model: str | None = None,
):
    """Run a single experiment with full retry logic."""
    logger.info(f"Starting {exp_name}...")
    hint_type, solver_type, mode = EXPERIMENT_SPECS[exp_name]
    experiment_cls = make_experiment(hint_type, solver_type, mode)
    models = _filter_models(MODELS, model)
    models = _restrict_models_to_cluster(models, cluster) if cluster is not None else models
    config_overrides = {}
    # Ensure submitit workers source this repo's setup script (activates conda env "ed").
    config_overrides["setup_commands"] = [f"source {SETUP_ENV_SCRIPT}"]
    if max_connections is not None:
        config_overrides["max_connections"] = max_connections
    if cpus_per_task is not None:
        config_overrides["cpus_per_task"] = cpus_per_task
    if mem_gb is not None:
        config_overrides["mem_gb"] = mem_gb
    if cluster is not None:
        config_overrides["account"] = _CLUSTER_ACCOUNT.get(cluster, "nlp")
        if cluster in _CLUSTER_TIME_HOURS:
            config_overrides["time_hours"] = _CLUSTER_TIME_HOURS[cluster]
    if sc_loprio:
        models = _apply_sc_loprio(models)
        # sc-loprio nodes are preemptible and often have less free VRAM at startup.
        # Use a safer vLLM target to reduce startup failures.
        config_overrides["gpu_memory_utilization"] = 0.88
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
            max_retries=1,
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
            max_retries=1,
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
            max_retries=1,
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
        "--cpus_per_task",
        type=int,
        default=None,
        help="Requested SLURM CPUs per task (default: SubmitConfig default).",
    )
    parser.add_argument(
        "--mem_gb",
        type=int,
        default=None,
        help="Requested SLURM memory in GB per job (default: SubmitConfig default).",
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
        "--regular_completion_fraction",
        type=float,
        default=0.9,
        help=(
            "During regular batches, keep retrying only until this fraction of the batch is complete; "
            "remaining stragglers are deferred to a bad-sample queue."
        ),
    )
    parser.add_argument(
        "--failed_retry_batch_size",
        type=int,
        default=5,
        help="Batch size for deferred bad-sample retries (tail phase).",
    )
    parser.add_argument(
        "--failed_retry_max_connections",
        type=int,
        default=4,
        help="Max connections to use during deferred bad-sample retries (tail phase).",
    )
    parser.add_argument(
        "--enable_checkpoint",
        action="store_true",
        help="Enable resumable checkpointing/chunking (default: disabled).",
    )
    parser.add_argument(
        "--resume_no_chunk",
        action="store_true",
        help=(
            "Resume from existing checkpoint (if any), but finish remaining samples "
            "in a single unchunked eval call."
        ),
    )
    parser.add_argument(
        "--disable_checkpoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cluster", choices=["sphinx", "miso", "jag"], default=None, help="Restrict job scheduling to a specific cluster (default: all clusters)")
    parser.add_argument("--low_prio", action="store_true", help="Submit jobs at low priority QOS (default: standard)")
    parser.add_argument("--sc_loprio", action="store_true", help="Submit pre-emptible jobs to sc-loprio partition using GPU constraints (overrides --cluster/--low_prio routing)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Launch only one model (full path or basename, e.g. Qwen/Qwen3-8B or Qwen3-8B).",
    )
    args = parser.parse_args()

    experiments_to_run = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    if args.plan:
        plan_runs(
            experiments_to_run,
            args.results_dir,
            model=args.model,
        )
    else:
        mode_flags = int(args.enable_checkpoint) + int(args.resume_no_chunk) + int(args.disable_checkpoint)
        if mode_flags > 1:
            raise ValueError(
                "Pass at most one of --enable_checkpoint, --resume_no_chunk, --disable_checkpoint"
            )
        # Default behavior: no checkpoint chunking unless explicitly enabled.
        if args.enable_checkpoint:
            os.environ["EXPERIMENT_CHECKPOINT_CHUNK_INSTANCES"] = str(max(args.checkpoint_chunk_instances, 128))
            os.environ.pop("EXPERIMENT_DISABLE_CHECKPOINT", None)
            os.environ.pop("EXPERIMENT_RESUME_NO_CHUNK", None)
        elif args.resume_no_chunk:
            os.environ.pop("EXPERIMENT_DISABLE_CHECKPOINT", None)
            os.environ["EXPERIMENT_RESUME_NO_CHUNK"] = "1"
        else:
            os.environ["EXPERIMENT_DISABLE_CHECKPOINT"] = "1"
            os.environ.pop("EXPERIMENT_RESUME_NO_CHUNK", None)
        os.environ["EXPERIMENT_REGULAR_CHUNK_COMPLETION_FRACTION"] = str(args.regular_completion_fraction)
        os.environ["EXPERIMENT_FAILED_RETRY_BATCH_SIZE"] = str(max(1, args.failed_retry_batch_size))
        os.environ["EXPERIMENT_FAILED_RETRY_MAX_CONNECTIONS"] = str(max(1, args.failed_retry_max_connections))

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
                    cpus_per_task=args.cpus_per_task,
                    mem_gb=args.mem_gb,
                    cluster=args.cluster,
                    low_prio=args.low_prio,
                    sc_loprio=args.sc_loprio,
                    model=args.model,
                )
                for exp_name in experiments_to_run
            ]
            for future in futures:
                future.result()



"""
SC LOPRIO
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 200 \
  --enable_checkpoint \
  --checkpoint_chunk_instances 100 \
  --sc_loprio \
  --max_connections 48 

MISO NON-PREEMPTIBLE, DP=8
python suze_experiments/20260213/aime_experiments.py \
      --experiment all \
      --epochs 10 \
      --results_dir christine_experiments/20251113/results \
      --max_jobs 10 \
      --max_connections 100 \
      --cluster miso \
      --num_gpus 8 \
      --cpus_per_task 120 \
      --mem_gb 1000 \
      --enable_checkpoint \
      --checkpoint_chunk_instances 1200 \
      --model Qwen3-32B \
      --regular_completion_fraction 0.85 \
      --failed_retry_batch_size 20 \
      --failed_retry_max_connections 20


USE THIS FOR NON-PREEMPTIBLE
python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results \
  --max_jobs 10 \
  --cluster sphinx \
  --enable_checkpoint \
  --checkpoint_chunk_instances 200 \
  --max_connections 48 \
  --regular_completion_fraction 0.85 \
  --failed_retry_batch_size 10 \
  --failed_retry_max_connections 10

python suze_experiments/20260213/aime_experiments.py \
  --experiment all \
  --epochs 10 \
  --results_dir christine_experiments/20251113/results --plan


"""
