from utils.setup import setup_env
import os
from inspect_ai import eval
from environments.math.math import math
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
    parser.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples (0 or 5)")
    parser.add_argument("--log_dir", type=str, default="./math")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_connections", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base_port", type=int, default=9000, help="Base port for vLLM server")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--prefill_path", type=str, default="/sphinx/u/cye/emergent-doordash/christine_experiments/20251006/math_test_hints.jsonl", help="Path to eval-time prefill JSONL file")
    parser.add_argument("--fewshot_path", type=str, default="/sphinx/u/cye/emergent-doordash/christine_experiments/20251006/math_train_hints.jsonl", help="Path to few-shot solutions JSONL file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    HINT_FRACTION = args.hint_fraction
    FEWSHOT = args.fewshot
    LOG_DIR = args.log_dir
    LIMIT = args.limit
    MAX_CONNECTIONS = args.max_connections
    TIMEOUT = args.timeout
    BASE_PORT = args.base_port
    MAX_TOKENS = args.max_tokens
    PREFILL_PATH = args.prefill_path
    FEWSHOT_PATH = args.fewshot_path

    # Set vLLM environment variables
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{BASE_PORT}/v1"
    os.environ["VLLM_API_KEY"] = "local"

    prefill_config = PrefillConfig(
        path=PREFILL_PATH,
        id_field="id",
        response_field="response",
        fraction=HINT_FRACTION,
    )

    fewshot_config = None
    if FEWSHOT > 0:
        fewshot_config = FewShotConfig(
            path=FEWSHOT_PATH,
            id_field="id",
            response_field="response",
            num_examples=FEWSHOT,
            seed=42,
            exclude_current=True,
        )

    log = eval(
        math(
            fewshot_config=fewshot_config,
            prefill_config=prefill_config,
            split="test"
        ),
        model=MODEL,
        log_dir=LOG_DIR,
        limit=LIMIT,
        max_connections=MAX_CONNECTIONS,
        display="rich",
        max_tokens=MAX_TOKENS,
    )
    results = extract_scores_from_log(log[0])
    with open(f"{LOG_DIR}/math_{FEWSHOT}shot_{HINT_FRACTION}.json", "w") as f:
        json.dump(results, f)
