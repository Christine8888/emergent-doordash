from utils.setup import setup_env
import os
from inspect_ai import eval
from environments.gpqa.gpqa import gpqa_diamond
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig
import logging
from utils.inspect_utils import extract_scores_from_log
import json
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
setup_env()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vllm/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./gpqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_connections", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base_port", type=int, default=9000, help="Base port for vLLM server")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    HINT_FRACTION = args.hint_fraction
    FEWSHOT = args.fewshot
    FEWSHOT_SEED = args.fewshot_seed
    LOG_DIR = args.log_dir
    LIMIT = args.limit
    MAX_CONNECTIONS = args.max_connections
    TIMEOUT = args.timeout
    BASE_PORT = args.base_port
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{BASE_PORT}/v1"
    os.environ["VLLM_API_KEY"] = "local"
    data_path = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251006/gpqa_diamond_samples_with_choices.jsonl"

    # Check if output file already exists
    filename = f"{LOG_DIR}/gpqa_diamond_{FEWSHOT}shot_{HINT_FRACTION}.json"
    if os.path.exists(filename):
        print(f"Output file {filename} already exists. Skipping evaluation.")
        exit(0)

    prefill_config = PrefillConfig(
        path=data_path,
        id_field="id",
        question_field="question_with_choices",
        response_field="response",
        fraction=HINT_FRACTION,
    )

    fewshot_config = None
    if FEWSHOT > 0:
        fewshot_config = FewShotConfig(
            path=data_path,
            id_field="id",
            question_field="question_with_choices",
            response_field="response",
            num_examples=FEWSHOT,
            seed=FEWSHOT_SEED,
            exclude_current=True,
        )

    log = eval(
        gpqa_diamond(
            fewshot_config=fewshot_config,
            prefill_config=prefill_config,
            timeout=TIMEOUT
        ),
        model=MODEL,
        log_dir=LOG_DIR,
        limit=LIMIT,
        max_connections=MAX_CONNECTIONS,
        display="rich",
        retry_on_error=5,
    )

    results = extract_scores_from_log(log[0])

    with open(filename, "w") as f:
        json.dump(results, f)