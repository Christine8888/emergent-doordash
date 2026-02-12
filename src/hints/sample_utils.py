"""Shared utilities for sampling scripts."""

import sys
import asyncio
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig
from tqdm.asyncio import tqdm

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evals.example import Example
from utils.setup import setup_env, setup_logging

setup_env()
logger = setup_logging()

CHOICE_LABELS = "abcdefghijklmnopqrstuvwxyz"


def sample_input_to_str(input_val) -> str:
    """Convert sample.input to string; handles both str and list-of-messages (e.g. mmlu_5_shot_cot)."""
    if isinstance(input_val, str):
        return input_val
    if isinstance(input_val, list):
        parts = []
        for msg in input_val:
            if hasattr(msg, "content") and msg.content:
                parts.append(str(msg.content))
        return "\n\n".join(parts)
    return str(input_val)


def format_choices_for_prompt(choices: list[str] | None) -> str:
    """Format multiple-choice options as (a) ..., (b) ..., etc. Returns empty string if no choices."""
    if not choices:
        return ""
    lines = []
    for i, choice in enumerate(choices):
        if i < len(CHOICE_LABELS):
            lines.append(f"({CHOICE_LABELS[i]}) {choice}")
    return "\n".join(lines)


def append_choices_to_messages(messages: list, choices: list[str] | None) -> list:
    """Return copy of messages with choices appended to the last user message."""
    choices_str = format_choices_for_prompt(choices)
    if not choices_str:
        return messages
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if hasattr(result[i], "role") and result[i].role == "user":
            result[i] = ChatMessageUser(content=result[i].content + "\n\n" + choices_str)
            break
    return result


def create_base_parser(description: str) -> ArgumentParser:
    """Create argument parser with common arguments for sampling scripts."""
    parser = ArgumentParser(description=description)
    parser.add_argument("--eval", type=str, required=True,
                        choices=[
                            # Internal environments (existing)
                            "gpqa",
                            "aime",
                            "math",
                            "math_level_5",
                            "hle",
                            "arc",
                            # External baselines (new)
                            "hellaswag",
                            "piqa",
                            "mmlu_5_shot_cot",
                            "bbh",
                            "arc_challenge",
                            "winogrande",
                        ],
                        help="Eval name")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str,
                        default="anthropic/claude-sonnet-4-5-20250929",
                        help="Model ID")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=32000,
                        help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25,
                        help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per problem")
    parser.add_argument("--n-per-question", type=int, default=1,
                        help="Number of correct samples to collect per question")
    parser.add_argument("--rationalize", action="store_true",
                        help="Add answer hint to prompt (rationalize mode)")
    parser.add_argument("--prompt-suffix", type=str, default=None,
                        help="Additional text to append to prompt")
    parser.add_argument(
        "--debug-first-problem",
        action="store_true",
        help="Print one representative problem's prompt/target and per-attempt accept/reject logs",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N problems (for debugging)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every API request, response, and grading result")
    return parser


def load_solved_counts(output_path: Path) -> dict[str, int]:
    """Load count of existing samples per question.

    Returns:
        Dictionary mapping question ID to count of existing samples
    """
    if not output_path.exists():
        return {}

    solved_counts = {}
    with open(output_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("hint") and data["hint"].strip():
                    qid = data["id"]
                    solved_counts[qid] = solved_counts.get(qid, 0) + 1
            except:
                pass
    return solved_counts


def log_sample_statistics(solved_counts: dict[str, int], n_total_problems: int):
    """Log statistics about sample counts per problem."""
    from collections import Counter

    if not solved_counts:
        logger.info("No existing samples found")
        return

    count_distribution = Counter(solved_counts.values())

    logger.info(f"Sample statistics ({len(solved_counts)}/{n_total_problems} problems have samples):")
    for count in sorted(count_distribution.keys()):
        n_problems = count_distribution[count]
        logger.info(f"  {count} sample(s): {n_problems} problems")

    logger.info(f"  Total samples: {sum(solved_counts.values())}")


def _to_model_input(model_input: list | str):
    """Convert to format expected by model.generate: list of messages."""
    if isinstance(model_input, list):
        return model_input
    return [ChatMessageUser(content=model_input)]


async def sample_solution(
    model_input: list | str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    *,
    sample_id: str = "",
    verbose: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Sample a single solution from the model.

    model_input: Either a list of ChatMessage (same format as baseline) or a prompt string.
    semaphore: Optional semaphore to limit concurrent API calls.
    """
    messages = _to_model_input(model_input)
    config = GenerateConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        max_connections=max_connections
    )

    # Gate on semaphore so at most max_concurrent requests are in-flight
    if semaphore is not None:
        await semaphore.acquire()

    try:
        if verbose:
            logger.info(f"  [{sample_id}] sending request to {model_id}...")
        async with get_model(model_id, config=config) as model:
            response = await model.generate(input=messages)
    finally:
        if semaphore is not None:
            semaphore.release()

    if verbose:
        n_tokens = response.usage.output_tokens if response.usage else "?"
        logger.info(f"  [{sample_id}] received {n_tokens} output tokens")
    return response.completion


async def try_sample_once(
    sample_id: str,
    question: str,
    prompt: str,
    model_input: list | str,
    target: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    sample_idx: int,
    attempt: int,
    eval_config,
    extra_fields: dict,
    response_to_hint: Callable[[str], any],
    *,
    debug_sample_id: str | None,
    extract_fn: Callable[[str], str] | None = None,
    rationalize: bool = False,
    verbose: bool = False,
    semaphore: asyncio.Semaphore | None = None,
) -> dict | None:
    """Try sampling once. Returns result dict if successful, None if failed."""
    extract = extract_fn if extract_fn is not None else eval_config.extract_answer

    response = await sample_solution(
        model_input, model_id, temperature, max_tokens, max_connections,
        sample_id=sample_id,
        verbose=verbose,
        semaphore=semaphore,
    )

    extracted = extract(response)
    correct = await eval_config.grade_answer(extracted, target)

    if verbose:
        verdict = "CORRECT" if correct else "INCORRECT"
        logger.info(f"  [{sample_id}][attempt {attempt}] {verdict} | extracted={extracted!r} target={target!r}")

    if debug_sample_id is not None and sample_id == debug_sample_id:
        logger.info(f"[debug] response:\n{response}")

    if correct:
        hint = response_to_hint(response)

        if not hint or not hint.strip():
            logger.warning(f"  {sample_id}: Answer correct but hint extraction failed. Response (first 500 chars):\n{response[:500]}")
            return None

        metadata = {"model": model_id}
        if rationalize:
            metadata["rationalize"] = True

        example = Example(
            id=sample_id,
            question=question,
            target=target,
            response=response,
            hint=hint,
            sample_idx=sample_idx,
            prompt=prompt,
            metadata=metadata,
        )
        result = example.to_dict()
        result.update(extra_fields)
        return result
    else:
        return None


async def run_sampling_loop(
    initial_tasks: list[dict],
    output_path: Path,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    max_retries: int,
    eval_config,
    response_to_hint: Callable[[str], any],
    extract_fn: Callable[[str], str] | None,
    rationalize: bool,
    pbar: tqdm,
    *,
    debug_sample_id: str | None,
    verbose: bool = False,
):
    """Run sampling with queue-based retries."""
    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_connections)

    # Track state per problem: {sample_id: {"attempts": int, "collected": int, "next_idx": int, "n_target": int}}
    problem_state = {}
    for task in initial_tasks:
        problem_state[task["sample_id"]] = {
            "attempts": 0,
            "collected": 0,
            "next_idx": task["start_idx"],
            "n_target": task["n_needed"],
        }

    # Create initial coroutines
    pending = set()
    task_info = {}  # Map task to its info for retry

    for task in initial_tasks:
        state = problem_state[task["sample_id"]]
        attempt_num = state["attempts"]
        coro = try_sample_once(
            sample_id=task["sample_id"],
            question=task["question"],
            prompt=task["prompt"],
            model_input=task["model_input"],
            target=task["target"],
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            max_connections=max_connections,
            sample_idx=state["next_idx"],
            attempt=attempt_num,
            eval_config=eval_config,
            extra_fields=task["extra_fields"],
            response_to_hint=response_to_hint,
            debug_sample_id=debug_sample_id,
            extract_fn=extract_fn,
            rationalize=rationalize,
            verbose=verbose,
            semaphore=semaphore,
        )
        t = asyncio.create_task(coro)
        pending.add(t)
        task_info[t] = task
        state["attempts"] += 1

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for t in done:
            task = task_info.pop(t)
            result = t.result()
            state = problem_state[task["sample_id"]]

            if result:
                # Success - write to file
                async with file_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                state["collected"] += 1
                state["next_idx"] += 1
                pbar.update(1)
                logger.info(f"  {task['sample_id']}[{state['next_idx'] - 1}]: collected ({state['collected']}/{state['n_target']})")

            # Check if we need more samples and haven't hit max retries
            needs_more = state["collected"] < state["n_target"]
            can_retry = state["attempts"] < max_retries

            if needs_more and can_retry:
                # Queue another attempt
                attempt_num = state["attempts"]
                coro = try_sample_once(
                    sample_id=task["sample_id"],
                    question=task["question"],
                    prompt=task["prompt"],
                    model_input=task["model_input"],
                    target=task["target"],
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_connections=max_connections,
                    sample_idx=state["next_idx"],
                    attempt=attempt_num,
                    eval_config=eval_config,
                    extra_fields=task["extra_fields"],
                    response_to_hint=response_to_hint,
                    debug_sample_id=debug_sample_id,
                    extract_fn=extract_fn,
                    rationalize=rationalize,
                    verbose=verbose,
                    semaphore=semaphore,
                )
                new_t = asyncio.create_task(coro)
                pending.add(new_t)
                task_info[new_t] = task
                state["attempts"] += 1


async def collect_samples(
    args,
    response_to_hint: Callable[[str], any],
    format_fn: Callable[[any], str] | None = None,
    extract_fn: Callable[[str], str] | None = None,
):
    """Collect samples by sampling until all have correct solutions.

    Args:
        args: Parsed command-line arguments (from create_base_parser)
        response_to_hint: Function to convert response to hint
        format_fn: Optional function to format prompt (overrides eval_config.format_prompt)
        extract_fn: Optional function to extract answer (overrides eval_config.extract_answer)
    """
    import importlib
    from utils.setup import setup_logging

    logger = setup_logging()

    # Load eval config
    eval_module = importlib.import_module(f"environments.{args.eval}.config")
    eval_config = eval_module

    # Add eval-specific args if available
    if hasattr(eval_config, "add_cli_args"):
        # Re-parse with eval-specific args
        parser = create_base_parser("Sample eval problems until all have correct solutions")
        eval_config.add_cli_args(parser)
        args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading {args.eval.upper()} dataset...")
    dataset_kwargs = {}
    if hasattr(eval_config, "get_dataset_kwargs"):
        dataset_kwargs = eval_config.get_dataset_kwargs(args)
    dataset = eval_config.get_dataset(**dataset_kwargs)
    all_samples = list(dataset)

    if args.limit is not None:
        all_samples = all_samples[:args.limit]
        logger.info(f"Limiting to first {args.limit} problems")

    logger.info(f"Loaded {len(all_samples)} problems")

    # Setup output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing sample counts per question
    solved_counts = load_solved_counts(output_path)
    log_sample_statistics(solved_counts, len(all_samples))

    # Use provided format_fn or fall back to eval_config.format_prompt
    format_prompt_fn = format_fn if format_fn is not None else eval_config.format_prompt

    # Create task dicts for problems that need more samples
    tasks = []
    total_samples_needed = 0

    for sample in all_samples:
        sample_id = str(sample.id)
        existing_count = solved_counts.get(sample_id, 0)
        n_needed = args.n_per_question - existing_count

        if n_needed > 0:
            total_samples_needed += n_needed

            fmt_result = format_prompt_fn(sample)
            if isinstance(fmt_result, tuple):
                model_input, prompt = fmt_result
            else:
                prompt = fmt_result
                # Match baseline: use raw message list when eval provides it (e.g. mmlu_5_shot_cot)
                if hasattr(eval_config, "get_model_input"):
                    raw = eval_config.get_model_input(sample)
                    model_input = raw if raw is not None else prompt
                else:
                    model_input = prompt

            if args.prompt_suffix:
                suffix = "\n\n" + args.prompt_suffix
                prompt = prompt + suffix
                if isinstance(model_input, str):
                    model_input = model_input + suffix
                else:
                    model_input = list(model_input) + [ChatMessageUser(content=args.prompt_suffix)]
            if args.rationalize:
                rationalize_hint = f"HINT: The answer is {sample.target}. Please still produce a fully natural chain of thought and solution that does not reveal the answer early, but that reaches this answer after actually properly solving the problem in full detail. Do not mention this hint anywhere in your response."
                prompt = prompt + "\n\n" + rationalize_hint
                if isinstance(model_input, str):
                    model_input = model_input + "\n\n" + rationalize_hint
                else:
                    model_input = list(model_input) + [ChatMessageUser(content=rationalize_hint)]

            extra_fields = {}
            if hasattr(eval_config, "extract_sample_fields"):
                extra_fields = eval_config.extract_sample_fields(sample)

            # Include choices in the question text for display/storage
            question_text = sample_input_to_str(sample.input)
            choices = getattr(sample, "choices", None)
            choices_str = format_choices_for_prompt(choices)
            if choices_str:
                question_text = question_text + "\n\n" + choices_str

            tasks.append({
                "sample_id": sample_id,
                "question": question_text,
                "prompt": prompt,
                "model_input": model_input,
                "target": str(sample.target),
                "extra_fields": extra_fields,
                "start_idx": existing_count,
                "n_needed": n_needed,
            })

    debug_sample_id = tasks[0]["sample_id"] if (args.debug_first_problem and tasks) else None
    if debug_sample_id is not None:
        t0 = tasks[0]
        logger.info(f"[debug] printing debug for sample_id={debug_sample_id!r}")
        logger.info(f"[debug] target: {t0['target']!r}")
        logger.info(f"[debug] question:\n{t0['question']}")
        logger.info(f"[debug] prompt:\n{t0['prompt']}")

    logger.info(f"Processing {total_samples_needed} samples across {len(tasks)} problems")
    logger.info(f"Target: {args.n_per_question} sample(s) per question")
    if args.rationalize:
        logger.info(f"Rationalize mode: ENABLED")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max retries per problem: {args.max_retries}")
    if args.prompt_suffix:
        logger.info(f"Prompt suffix: {args.prompt_suffix}")
    logger.info(f"Output: {output_path}\n")

    pbar = tqdm(total=total_samples_needed, desc="Solving")

    await run_sampling_loop(
        initial_tasks=tasks,
        output_path=output_path,
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_connections=args.max_concurrent,
        max_retries=args.max_retries,
        eval_config=eval_config,
        response_to_hint=response_to_hint,
        extract_fn=extract_fn,
        rationalize=args.rationalize,
        pbar=pbar,
        debug_sample_id=debug_sample_id,
        verbose=getattr(args, "verbose", False),
    )

    pbar.close()
    logger.info("Done!")
