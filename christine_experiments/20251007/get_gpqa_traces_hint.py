from utils.setup import setup_env
import os
from inspect_ai import eval
from environments.gpqa.gpqa import gpqa_diamond
from evals.prefill import PrefillConfig
import logging
from utils.inspect_utils import extract_scores_from_log
import json
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
setup_env()

os.environ["VLLM_BASE_URL"] = "http://localhost:9000/v1"
os.environ["VLLM_API_KEY"] = "local"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vllm/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hint_fraction", type=float, default=0.8)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./gpqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_connections", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    HINT_FRACTION = args.hint_fraction
    LOG_DIR = args.log_dir
    TEMPLATE = args.template
    LIMIT = args.limit
    MAX_CONNECTIONS = args.max_connections
    TIMEOUT = args.timeout

    prefill_config = PrefillConfig(
        path="/nlp/scr/cye/emergent-doordash/christine_experiments/20251006/gpqa_diamond_samples.jsonl",
        id_field="id",
        response_field="response",
        fraction=HINT_FRACTION,
    )
    log = eval(
        gpqa_diamond(template=TEMPLATE, prefill_config=prefill_config, timeout=TIMEOUT),
        model=MODEL,
        log_dir=LOG_DIR,
        limit=LIMIT,
        max_connections=MAX_CONNECTIONS,
        display="rich",
    )
    results = extract_scores_from_log(log[0])
    with open(f"{LOG_DIR}/gpqa_diamond_{HINT_FRACTION}.json", "w") as f:
        json.dump(results, f)