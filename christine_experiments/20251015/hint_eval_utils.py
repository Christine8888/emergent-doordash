from utils.setup import setup_env
import os
from inspect_ai import eval
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig
import logging
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs
import json
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
setup_env()


def create_base_parser(default_log_dir, default_prefill_path=None, default_fewshot_path=None):
    """Create argument parser with common arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vllm/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    parser.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--fewshot_seed", type=int, default=42, help="Seed for few-shot sampling")
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_connections", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base_port", type=int, default=9000, help="Base port for vLLM server")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--epochs", type=int, default=1)

    if default_prefill_path:
        parser.add_argument("--prefill_path", type=str, default=default_prefill_path,
                          help="Path to eval-time prefill JSONL file")
    if default_fewshot_path:
        parser.add_argument("--fewshot_path", type=str, default=default_fewshot_path,
                          help="Path to few-shot solutions JSONL file")

    return parser


def setup_vllm_env(base_port):
    """Set vLLM environment variables."""
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{base_port}/v1"
    os.environ["VLLM_API_KEY"] = "local"


def check_output_exists(filename):
    """Check if output file exists and exit if it does."""
    if os.path.exists(filename):
        print(f"Output file {filename} already exists. Skipping evaluation.")
        exit(0)


def create_configs(args, id_field="id", question_field="question", response_field="response"):
    """Create prefill and fewshot configs."""
    prefill_path = getattr(args, 'prefill_path', None)
    fewshot_path = getattr(args, 'fewshot_path', None)

    prefill_kwargs = {
        "path": prefill_path,
        "id_field": id_field,
        "response_field": response_field,
        "fraction": args.hint_fraction,
    }
    if question_field:
        prefill_kwargs["question_field"] = question_field

    prefill_config = PrefillConfig(**prefill_kwargs)

    fewshot_config = None
    if args.fewshot > 0:
        fewshot_kwargs = {
            "path": fewshot_path,
            "id_field": id_field,
            "response_field": response_field,
            "num_examples": args.fewshot,
            "seed": args.fewshot_seed,
            "exclude_current": True,
        }
        if question_field:
            fewshot_kwargs["question_field"] = question_field

        fewshot_config = FewShotConfig(**fewshot_kwargs)

    return prefill_config, fewshot_config


def run_eval_and_save(
    task_fn,
    task_kwargs,
    args,
    output_filename,
    bootstrap_metric=None
):
    """Run evaluation and save results."""
    eval_kwargs = {
        "model": args.model,
        "log_dir": args.log_dir,
        "epochs": args.epochs,
        "limit": args.limit,
        "max_connections": args.max_connections,
        "display": "rich",
    }

    if args.max_tokens is not None:
        eval_kwargs["max_tokens"] = args.max_tokens

    log = eval(task_fn(**task_kwargs), **eval_kwargs)

    results = extract_scores_from_log(log[0])

    if args.epochs > 1 and bootstrap_metric:
        results["manual_bootstrap"] = compute_bootstrap_over_epochs(log[0], bootstrap_metric)

    with open(output_filename, "w") as f:
        json.dump(results, f)
