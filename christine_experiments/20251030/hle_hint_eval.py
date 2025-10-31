from evals.hint_eval_utils import (
    create_base_parser, setup_vllm_env, check_output_exists,
    create_configs, run_eval_and_save
)
from environments.hle.hle import hle

BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/20251030/hle_filtered_long.jsonl"

if __name__ == "__main__":
    parser = create_base_parser(
        default_log_dir="./hle",
        default_prefill_path=DATA_PATH,
        default_fewshot_path=DATA_PATH
    )
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/hle_{args.fewshot}shot_{args.hint_fraction}.json"
    check_output_exists(filename)

    prefill_config, fewshot_config = create_configs(
        args,
        question_field="question"  # HLE uses simple 'question' field
    )

    task_kwargs = {
        "prefill_config": prefill_config,
        "timeout": args.timeout
    }

    run_eval_and_save(
        task_fn=hle,
        task_kwargs=task_kwargs,
        args=args,
        output_filename=filename,
        bootstrap_metric='accuracy'
    )
