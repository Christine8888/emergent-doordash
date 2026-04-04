from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets import get_dataset_spec
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


def _existing_hint_ids(path: str | Path) -> set[str]:
    existing = set()
    for row in read_jsonl(path, model_cls=None):
        hint_id = row.get("hint_id") if isinstance(row, dict) else None
        if isinstance(hint_id, str):
            existing.add(hint_id)
    return existing


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
) -> str:
    """Generate hint records and append them to a JSONL file."""
    spec = get_dataset_spec(benchmark_name)
    problems = spec.load_problems()[:limit]
    prompt_version = spec.prompt_version(hint_type)
    post_process_version = spec.post_process_version(hint_type)

    out_path = str(
        build_hint_generation_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            data_root="data",
        )
    )

    existing = _existing_hint_ids(out_path)
    written = 0
    would_write = 0
    skipped = 0
    failed = 0

    for problem in problems:
        for rollout_id in range(num_rollouts):
            hint_id = make_stable_id(
                problem.problem_id,
                hint_type,
                rollout_id,
                first_model,
                first_model_attempts,
                second_model,
                second_model_attempts,
                length=16,
            )
            if hint_id in existing:
                skipped += 1
                continue

            prompt = spec.build_hint_prompt(problem, hint_type)
            if dry_run:
                would_write += 1
                continue

            successful_usage = None
            successful_model = None
            successful_extracted = None
            attempt_plan = [
                (first_model, first_model_attempts),
                (second_model, second_model_attempts),
            ]
            attempt_idx = 0

            for attempt_model, max_attempts in attempt_plan:
                for _ in range(max_attempts):
                    attempt_idx += 1
                    usage = query_claude_hint(
                        prompt=prompt,
                        model=attempt_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    extracted_answer = spec.extract_answer(usage["model_output"])
                    is_correct = spec.is_correct(extracted_answer, problem)
                    if is_correct:
                        successful_usage = usage
                        successful_model = attempt_model
                        successful_extracted = extracted_answer
                        break
                    print(
                        f"[hint_generation][WARN] incorrect answer benchmark={benchmark_name} "
                        f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                        f"model={attempt_model} extracted={extracted_answer!r} correct={problem.answer!r}"
                    )
                if successful_usage is not None:
                    break

            if successful_usage is None:
                failed += 1
                continue

            full_hint = spec.post_process_hint(
                problem=problem,
                hint_type=hint_type,
                model_output=successful_usage["model_output"],
            )

            record = HintGenerationRecord(
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
                    "prompt": prompt,
                    "prompt_version": prompt_version,
                    "post_process_version": post_process_version,
                    "dataset_spec": spec.name,
                    "problem_source": problem.source,
                    "temperature": temperature,
                    "extracted_answer": successful_extracted,
                    "first_model": first_model,
                    "first_model_attempts": first_model_attempts,
                    "second_model": second_model,
                    "second_model_attempts": second_model_attempts,
                    "total_attempts_used": attempt_idx,
                    "stop_reason": successful_usage["stop_reason"],
                },
            )
            append_jsonl(out_path, record)
            existing.add(hint_id)
            written += 1

    if dry_run:
        print(
            f"[hint_generation] dry_run benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} would_write={would_write} skipped={skipped} "
            f"output={out_path}"
        )
    else:
        print(
            f"[hint_generation] done benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} written={written} skipped={skipped} failed={failed} "
            f"output={out_path}"
        )
    return out_path
