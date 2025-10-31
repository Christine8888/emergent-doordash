from evals.hint_eval_utils import (
    create_base_parser, setup_vllm_env, check_output_exists,
    create_configs, run_eval_and_save
)
from environments.gpqa.gpqa import gpqa_diamond

BASE_DIR = "/sphinx/u/cye/emergent-doordash/"
DATA_PATH = f"{BASE_DIR}/christine_experiments/20251030/gpqa_diamond_samples_filtered.jsonl"

if __name__ == "__main__":
    parser = create_base_parser(
        default_log_dir="./gpqa",
        default_prefill_path=DATA_PATH,
        default_fewshot_path=DATA_PATH
    )
    args = parser.parse_args()

    setup_vllm_env(args.base_port)

    filename = f"{args.log_dir}/gpqa_diamond_{args.fewshot}shot_{args.hint_fraction}.json"
    check_output_exists(filename)

    prefill_config, fewshot_config = create_configs(
        args,
        question_field="question_with_choices"  # Existing prefill file uses this field
    )

    task_kwargs = {
        "fewshot_config": fewshot_config,
        "prefill_config": prefill_config,
        "timeout": args.timeout
    }

    run_eval_and_save(
        task_fn=gpqa_diamond,
        task_kwargs=task_kwargs,
        args=args,
        output_filename=filename,
        bootstrap_metric='choice'
    )