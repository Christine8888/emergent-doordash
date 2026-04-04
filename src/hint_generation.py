from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.datasets import get_dataset_spec
from src.hint_types import get_hint_type_spec
from src.storage import append_jsonl, build_hint_generation_path, make_stable_id, read_jsonl
from src.types import HintGenerationRecord


def _parse_anthropic_message_text(message) -> str:
    texts: list[str] = []
    for block in message.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts).strip()


def query_claude_hint(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for _ in stream.text_stream:
            pass
        response = stream.get_final_message()

    return {
        "model_output": _parse_anthropic_message_text(response),
        "input_token_count": int(response.usage.input_tokens),
        "output_token_count": int(response.usage.output_tokens),
        "stop_reason": getattr(response, "stop_reason", None),
    }


def _existing_rollouts_by_problem(path: str | Path) -> dict[str, set[int]]:
    existing: dict[str, set[int]] = {}
    for row in read_jsonl(path, model_cls=None):
        if not isinstance(row, dict):
            continue
        problem_id = row.get("problem_id")
        rollout_id = row.get("rollout_id")
        if not isinstance(problem_id, str):
            continue
        if not isinstance(rollout_id, int):
            continue
        existing.setdefault(problem_id, set()).add(rollout_id)
    return existing


def _generate_record_for_task(
    *,
    benchmark_name: str,
    hint_type: str,
    prompt_version: str,
    post_process_version: str,
    should_grade_output: bool,
    dataset_spec,
    hint_type_spec,
    problem,
    rollout_id: int,
    hint_id: str,
    generation_context,
    prompt: str,
    first_model: str,
    first_model_attempts: int,
    second_model: str,
    second_model_attempts: int,
    max_tokens: int,
    temperature: float,
) -> HintGenerationRecord | None:
    successful_usage = None
    successful_model = None
    successful_extracted = None
    successful_full_hint = None
    successful_grader_metadata: dict[str, Any] = {}
    attempt_plan = [
        (first_model, first_model_attempts),
        (second_model, second_model_attempts),
    ]
    attempt_idx = 0

    for attempt_model, max_attempts in attempt_plan:
        for _ in range(max_attempts):
            attempt_idx += 1
            print(
                f"[hint_generation] request benchmark={benchmark_name} hint_type={hint_type} "
                f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                f"model={attempt_model}"
            )
            usage = query_claude_hint(
                prompt=prompt,
                model=attempt_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print(
                f"[hint_generation] response benchmark={benchmark_name} hint_type={hint_type} "
                f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                f"model={attempt_model} input_tokens={usage['input_token_count']} "
                f"output_tokens={usage['output_token_count']} stop_reason={usage['stop_reason']}"
            )

            grade_result = hint_type_spec.grade_output(
                model_output=usage["model_output"],
                problem=problem,
                dataset_spec=dataset_spec,
                context=generation_context,
            )
            extracted_answer = grade_result["extracted_answer"]
            grader_metadata: dict[str, Any] = grade_result["metadata"]
            if not grade_result["is_correct"]:
                if extracted_answer is None:
                    print(
                        f"[hint_generation][WARN] grader_rejected benchmark={benchmark_name} "
                        f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                        f"model={attempt_model} metadata={grader_metadata}"
                    )
                else:
                    print(
                        f"[hint_generation][WARN] grader_rejected benchmark={benchmark_name} "
                        f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                        f"model={attempt_model} extracted={extracted_answer!r} correct={problem.answer!r}"
                    )
                continue

            try:
                full_hint_candidate = hint_type_spec.post_process(
                    model_output=usage["model_output"],
                    context=generation_context,
                )
            except Exception as exc:
                print(
                    f"[hint_generation][WARN] invalid_hint_output benchmark={benchmark_name} "
                    f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                    f"model={attempt_model} error={exc}"
                )
                continue

            successful_usage = usage
            successful_model = attempt_model
            successful_extracted = extracted_answer
            successful_grader_metadata = grader_metadata
            successful_full_hint = full_hint_candidate
            break
        if successful_usage is not None:
            break

    if successful_usage is None:
        return None

    full_hint = str(successful_full_hint)
    context_metadata = hint_type_spec.context_metadata(generation_context)

    return HintGenerationRecord(
        hint_id=hint_id,
        problem_id=problem.problem_id,
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        rollout_id=rollout_id,
        generator_model=successful_model,
        question=problem.question,
        answer=problem.answer,
        model_output=successful_usage["model_output"],
        full_hint=full_hint,
        input_token_count=successful_usage["input_token_count"],
        output_token_count=successful_usage["output_token_count"],
        metadata={
            "hint_type_spec": hint_type_spec.name.value,
            "prompt": prompt,
            "prompt_version": prompt_version,
            "post_process_version": post_process_version,
            "grade_model_output": should_grade_output,
            "dataset_spec": dataset_spec.name,
            "problem_source": problem.source,
            "temperature": temperature,
            "extracted_answer": successful_extracted,
            "grader_metadata": successful_grader_metadata,
            "first_model": first_model,
            "first_model_attempts": first_model_attempts,
            "second_model": second_model,
            "second_model_attempts": second_model_attempts,
            "total_attempts_used": attempt_idx,
            "stop_reason": successful_usage["stop_reason"],
            **context_metadata,
        },
    )


def generate_hints(
    *,
    benchmark_name: str,
    hint_type: str,
    first_model: str,
    first_model_attempts: int,
    second_model: str,
    second_model_attempts: int,
    num_rollouts: int,
    limit: int,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    concurrency: int = 1,
) -> str:
    """Generate hint records and append them to a JSONL file."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    dataset_spec = get_dataset_spec(benchmark_name)
    hint_type_spec = get_hint_type_spec(hint_type)
    problems = dataset_spec.load_problems()[:limit]
    prompt_version = hint_type_spec.prompt_version
    post_process_version = hint_type_spec.post_process_version
    should_grade_output = hint_type_spec.grade_model_output

    out_path = str(
        build_hint_generation_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            data_root="data",
        )
    )

    existing_rollouts_by_problem = _existing_rollouts_by_problem(out_path)
    prepared_tasks: list[dict[str, Any]] = []
    would_write = 0
    skipped = 0

    for problem in problems:
        existing_rollouts = existing_rollouts_by_problem.setdefault(problem.problem_id, set())
        for rollout_id in range(num_rollouts):
            if rollout_id in existing_rollouts:
                skipped += 1
                continue

            hint_id = make_stable_id(
                problem.problem_id,
                hint_type,
                rollout_id,
                length=16,
            )

            generation_context = hint_type_spec.build_context(
                benchmark_name=benchmark_name,
                problem=problem,
                rollout_id=rollout_id,
            )
            prompt = hint_type_spec.build_prompt(
                problem=problem,
                context=generation_context,
            )
            prepared_tasks.append(
                {
                    "problem": problem,
                    "rollout_id": rollout_id,
                    "hint_id": hint_id,
                    "generation_context": generation_context,
                    "prompt": prompt,
                }
            )

    if dry_run:
        would_write = len(prepared_tasks)
        print(
            f"[hint_generation] dry_run benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} would_write={would_write} skipped={skipped} "
            f"output={out_path}"
        )
    else:
        written = 0
        failed = 0

        if concurrency == 1:
            for task in prepared_tasks:
                record = _generate_record_for_task(
                    benchmark_name=benchmark_name,
                    hint_type=hint_type,
                    prompt_version=prompt_version,
                    post_process_version=post_process_version,
                    should_grade_output=should_grade_output,
                    dataset_spec=dataset_spec,
                    hint_type_spec=hint_type_spec,
                    problem=task["problem"],
                    rollout_id=task["rollout_id"],
                    hint_id=task["hint_id"],
                    generation_context=task["generation_context"],
                    prompt=task["prompt"],
                    first_model=first_model,
                    first_model_attempts=first_model_attempts,
                    second_model=second_model,
                    second_model_attempts=second_model_attempts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if record is None:
                    failed += 1
                    continue
                append_jsonl(out_path, record)
                written += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _generate_record_for_task,
                        benchmark_name=benchmark_name,
                        hint_type=hint_type,
                        prompt_version=prompt_version,
                        post_process_version=post_process_version,
                        should_grade_output=should_grade_output,
                        dataset_spec=dataset_spec,
                        hint_type_spec=hint_type_spec,
                        problem=task["problem"],
                        rollout_id=task["rollout_id"],
                        hint_id=task["hint_id"],
                        generation_context=task["generation_context"],
                        prompt=task["prompt"],
                        first_model=first_model,
                        first_model_attempts=first_model_attempts,
                        second_model=second_model,
                        second_model_attempts=second_model_attempts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    for task in prepared_tasks
                ]
                for future in as_completed(futures):
                    record = future.result()
                    if record is None:
                        failed += 1
                        continue
                    append_jsonl(out_path, record)
                    written += 1

        print(
            f"[hint_generation] done benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} concurrency={concurrency} "
            f"written={written} skipped={skipped} failed={failed} "
            f"output={out_path}"
        )
    return out_path
