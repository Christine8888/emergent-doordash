from __future__ import annotations
import base64
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from src.datasets import get_dataset_spec
from src.hint_types import get_hint_type_spec
from src.storage import append_jsonl, build_hint_generation_path, make_stable_id, read_jsonl
from src.types import HintGenerationRecord

ANTHROPIC_MODELS: set[str] = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}

OPENAI_MODELS: set[str] = {
    "gpt-5.4",
}

PromptContent = str | list[dict[str, Any]]


def _is_max_token_stop(*, provider: str, stop_reason: Any) -> bool:
    if not isinstance(stop_reason, str):
        return False
    if provider == "anthropic":
        return stop_reason == "max_tokens"
    if provider == "openai":
        return stop_reason == "length"
    return False


def _provider_for_model_id(model_id: str) -> str:
    model_name = model_id.strip()
    if model_name in ANTHROPIC_MODELS:
        return "anthropic"
    if model_name in OPENAI_MODELS:
        return "openai"
    all_known = sorted(ANTHROPIC_MODELS | OPENAI_MODELS)
    raise ValueError(
        f"Unknown model_id={model_id!r}. Add it to ANTHROPIC_MODELS or OPENAI_MODELS. "
        f"Known models: {all_known}"
    )


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


_load_project_env()


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def _parse_anthropic_message_text(message) -> str:
    texts: list[str] = []
    for block in message.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts).strip()


def _parse_anthropic_message_thinking(message) -> str:
    thoughts: list[str] = []
    for block in message.content:
        thinking = getattr(block, "thinking", None)
        if isinstance(thinking, str):
            thoughts.append(thinking)
    return "\n".join(thoughts).strip()


def _parse_openai_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts).strip()
    return str(content).strip() if content is not None else ""


def _extension_for_media_type(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(media_type.lower(), ".bin")


def _decode_data_image_uri(value: str) -> tuple[str, bytes] | None:
    match = re.fullmatch(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.*)", value.strip(), flags=re.DOTALL)
    if match is None:
        return None
    media_type = match.group(1).lower()
    payload = re.sub(r"\s+", "", match.group(2))
    return media_type, base64.b64decode(payload)


def _materialize_problem_images(problem) -> list[dict[str, Any]]:
    if problem.source != "cais/hle:test":
        return []
    image_value = str(problem.metadata.get("image") or "").strip()
    if not image_value:
        return []
    decoded = _decode_data_image_uri(image_value)
    if decoded is None:
        return [
            {
                "source_field": "image",
                "source_type": "unsupported",
                "saved": False,
                "error": "expected data:image/...;base64,...",
            }
        ]
    media_type, data = decoded
    image_dir = Path("data") / "hle_images" / problem.problem_id
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"question{_extension_for_media_type(media_type)}"
    if not image_path.exists() or image_path.read_bytes() != data:
        image_path.write_bytes(data)
    return [
        {
            "source_field": "image",
            "source_type": "data_uri",
            "saved": True,
            "path": str(image_path),
            "media_type": media_type,
            "byte_count": len(data),
        }
    ]


def _build_model_prompt_content(*, prompt: str, problem) -> tuple[PromptContent, list[dict[str, Any]]]:
    image_metadata = _materialize_problem_images(problem)
    image_parts: list[dict[str, Any]] = []
    for item in image_metadata:
        path = item.get("path")
        media_type = item.get("media_type")
        if item.get("saved") is not True or not isinstance(path, str) or not isinstance(media_type, str):
            continue
        data = Path(path).read_bytes()
        image_parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        )
    if not image_parts:
        return prompt, image_metadata
    return [{"type": "text", "text": prompt}, *image_parts], image_metadata


def query_anthropic_hint(
    prompt: PromptContent,
    model: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    thinking_effort: str,
) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt is not None:
        request["system"] = system_prompt
    thinking_mode = "disabled"
    effort: str | None = None
    if thinking_enabled:
        thinking_mode = "adaptive"
        effort = thinking_effort
        request["thinking"] = {"type": thinking_mode}
        request["output_config"] = {"effort": effort}

    with client.messages.stream(**request) as stream:
        for _ in stream.text_stream:
            pass
        response = stream.get_final_message()

    return {
        "model_output": _parse_anthropic_message_text(response),
        "thinking": _parse_anthropic_message_thinking(response),
        "provider": "anthropic",
        "thinking_enabled": thinking_enabled,
        "thinking_mode": thinking_mode,
        "effort": effort,
        "input_token_count": int(response.usage.input_tokens),
        "output_token_count": int(response.usage.output_tokens),
        "stop_reason": getattr(response, "stop_reason", None),
    }


def query_openai_hint(
    prompt: PromptContent,
    model: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    thinking_effort: str,
) -> dict[str, Any]:
    from openai import OpenAI

    if not isinstance(prompt, str):
        raise ValueError("OpenAI hint generation does not yet support HLE multimodal prompt content.")

    client = OpenAI(timeout=httpx.Timeout(7200, connect=30))
    effort = thinking_effort if thinking_enabled else "none"
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        reasoning_effort=effort,
    )
    choice = completion.choices[0]
    content = getattr(choice.message, "content", None)
    usage = getattr(completion, "usage", None)
    return {
        "model_output": _parse_openai_message_text(content),
        "thinking": "",
        "provider": "openai",
        "thinking_enabled": thinking_enabled,
        "thinking_mode": "openai_reasoning_effort" if thinking_enabled else "disabled",
        "effort": effort,
        "input_token_count": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_token_count": int(getattr(usage, "completion_tokens", 0) or 0),
        "stop_reason": getattr(choice, "finish_reason", None),
    }


def query_model_hint(
    prompt: PromptContent,
    model: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    thinking_effort: str,
) -> dict[str, Any]:
    provider = _provider_for_model_id(model)
    if provider == "anthropic":
        return query_anthropic_hint(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
        )
    if provider == "openai":
        return query_openai_hint(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
        )
    raise RuntimeError(f"Unhandled provider={provider!r} for model={model!r}")


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
    thinking_enabled: bool,
    thinking_effort: str,
) -> tuple[HintGenerationRecord | None, list[dict[str, Any]]]:
    successful_usage = None
    successful_model = None
    successful_extracted = None
    successful_full_hint = None
    successful_grader_metadata: dict[str, Any] = {}
    failed_attempts: list[dict[str, Any]] = []
    attempt_plan = [
        (first_model, first_model_attempts),
        (second_model, second_model_attempts),
    ]
    attempt_idx = 0
    context_metadata = hint_type_spec.context_metadata(generation_context)
    system_prompt = hint_type_spec.system_prompt
    prompt_content, prompt_image_metadata = _build_model_prompt_content(
        prompt=prompt,
        problem=problem,
    )
    prompt_image_count = sum(1 for item in prompt_image_metadata if item.get("saved") is True)

    for attempt_model, max_attempts in attempt_plan:
        provider_name = _provider_for_model_id(attempt_model)
        query_error_thinking_mode = (
            "adaptive"
            if thinking_enabled and provider_name == "anthropic"
            else ("openai_reasoning_effort" if thinking_enabled else "disabled")
        )
        query_error_effort = (
            thinking_effort
            if thinking_enabled
            else ("none" if provider_name == "openai" else None)
        )
        for _ in range(max_attempts):
            attempt_idx += 1
            _log(
                f"[hint_generation] request benchmark={benchmark_name} hint_type={hint_type} "
                f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                f"model={attempt_model} images={prompt_image_count}"
            )
            try:
                usage = query_model_hint(
                    prompt=prompt_content,
                    model=attempt_model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_enabled=thinking_enabled,
                    thinking_effort=thinking_effort,
                )
            except Exception as exc:
                failed_attempts.append(
                    {
                        "hint_id": hint_id,
                        "problem_id": problem.problem_id,
                        "benchmark_name": benchmark_name,
                        "hint_type": hint_type,
                        "rollout_id": rollout_id,
                        "attempt": attempt_idx,
                        "model": attempt_model,
                        "failure_type": "query_error",
                        "failure_error": str(exc),
                        "question": problem.question,
                        "answer": problem.answer,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "prompt_image_metadata": prompt_image_metadata,
                        "provider": provider_name,
                        "thinking_enabled": thinking_enabled,
                        "thinking_mode": query_error_thinking_mode,
                        "effort": query_error_effort,
                        **context_metadata,
                    }
                )
                _log(
                    f"[hint_generation][WARN] query_error benchmark={benchmark_name} "
                    f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                    f"model={attempt_model} error={exc}"
                )
                continue
            _log(
                f"[hint_generation] response benchmark={benchmark_name} hint_type={hint_type} "
                f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                f"model={attempt_model} input_tokens={usage['input_token_count']} "
                f"output_tokens={usage['output_token_count']} stop_reason={usage['stop_reason']}"
            )
            if _is_max_token_stop(provider=usage["provider"], stop_reason=usage["stop_reason"]):
                failed_attempts.append(
                    {
                        "hint_id": hint_id,
                        "problem_id": problem.problem_id,
                        "benchmark_name": benchmark_name,
                        "hint_type": hint_type,
                        "rollout_id": rollout_id,
                        "attempt": attempt_idx,
                        "model": attempt_model,
                        "failure_type": "max_tokens_reached",
                        "question": problem.question,
                        "answer": problem.answer,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "prompt_image_metadata": prompt_image_metadata,
                        "model_output": usage["model_output"],
                        "provider": usage["provider"],
                        "input_token_count": usage["input_token_count"],
                        "output_token_count": usage["output_token_count"],
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "thinking_enabled": usage["thinking_enabled"],
                        "thinking_mode": usage["thinking_mode"],
                        "effort": usage["effort"],
                        "max_tokens": max_tokens,
                        **context_metadata,
                    }
                )
                _log(
                    f"[hint_generation][WARN] max_tokens_reached benchmark={benchmark_name} "
                    f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                    f"model={attempt_model} output_tokens={usage['output_token_count']} "
                    f"stop_reason={usage['stop_reason']}"
                )
                continue

            grade_result = hint_type_spec.grade_output(
                model_output=usage["model_output"],
                problem=problem,
                dataset_spec=dataset_spec,
                context=generation_context,
            )
            extracted_answer = grade_result["extracted_answer"]
            grader_metadata: dict[str, Any] = grade_result["metadata"]
            if not grade_result["is_correct"]:
                failed_attempts.append(
                    {
                        "hint_id": hint_id,
                        "problem_id": problem.problem_id,
                        "benchmark_name": benchmark_name,
                        "hint_type": hint_type,
                        "rollout_id": rollout_id,
                        "attempt": attempt_idx,
                        "model": attempt_model,
                        "failure_type": "grader_rejected",
                        "question": problem.question,
                        "answer": problem.answer,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "prompt_image_metadata": prompt_image_metadata,
                        "model_output": usage["model_output"],
                        "provider": usage["provider"],
                        "input_token_count": usage["input_token_count"],
                        "output_token_count": usage["output_token_count"],
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "thinking_enabled": usage["thinking_enabled"],
                        "thinking_mode": usage["thinking_mode"],
                        "effort": usage["effort"],
                        "extracted_answer": extracted_answer,
                        "grader_metadata": grader_metadata,
                        **context_metadata,
                    }
                )
                if extracted_answer is None:
                    _log(
                        f"[hint_generation][WARN] grader_rejected benchmark={benchmark_name} "
                        f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                        f"model={attempt_model} metadata={grader_metadata}"
                    )
                else:
                    _log(
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
                failed_attempts.append(
                    {
                        "hint_id": hint_id,
                        "problem_id": problem.problem_id,
                        "benchmark_name": benchmark_name,
                        "hint_type": hint_type,
                        "rollout_id": rollout_id,
                        "attempt": attempt_idx,
                        "model": attempt_model,
                        "failure_type": "invalid_hint_output",
                        "failure_error": str(exc),
                        "question": problem.question,
                        "answer": problem.answer,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "prompt_image_metadata": prompt_image_metadata,
                        "model_output": usage["model_output"],
                        "provider": usage["provider"],
                        "input_token_count": usage["input_token_count"],
                        "output_token_count": usage["output_token_count"],
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "thinking_enabled": usage["thinking_enabled"],
                        "thinking_mode": usage["thinking_mode"],
                        "effort": usage["effort"],
                        "extracted_answer": extracted_answer,
                        "grader_metadata": grader_metadata,
                        **context_metadata,
                    }
                )
                _log(
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
        return None, failed_attempts

    full_hint = str(successful_full_hint)

    return (
        HintGenerationRecord(
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
                "problem_answer_type": problem.metadata.get("answer_type"),
                "problem_text_only": problem.metadata.get("text_only"),
                "temperature": temperature,
                "system_prompt": system_prompt,
                "prompt_image_metadata": prompt_image_metadata,
                "extracted_answer": successful_extracted,
                "grader_metadata": successful_grader_metadata,
                "first_model": first_model,
                "first_model_attempts": first_model_attempts,
                "second_model": second_model,
                "second_model_attempts": second_model_attempts,
                "provider": successful_usage["provider"],
                "total_attempts_used": attempt_idx,
                "stop_reason": successful_usage["stop_reason"],
                "thinking": successful_usage["thinking"],
                "thinking_enabled": successful_usage["thinking_enabled"],
                "thinking_mode": successful_usage["thinking_mode"],
                "effort": successful_usage["effort"],
                **context_metadata,
            },
        ),
        failed_attempts,
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
    problem_ids: list[str] | None = None,
    thinking_enabled: bool = True,
    thinking_effort: str = "medium",
    concurrency: int = 1,
) -> str:
    """Generate hint records and append them to a JSONL file."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if thinking_effort not in {"low", "medium", "high", "max"}:
        raise ValueError("thinking_effort must be one of: low, medium, high, max")
    _provider_for_model_id(first_model)
    _provider_for_model_id(second_model)

    dataset_spec = get_dataset_spec(benchmark_name)
    hint_type_spec = get_hint_type_spec(hint_type)
    all_problems = dataset_spec.load_problems()
    if problem_ids:
        cleaned_problem_ids = [pid.strip() for pid in problem_ids if pid.strip()]
        by_id = {problem.problem_id: problem for problem in all_problems}
        missing_problem_ids = [pid for pid in cleaned_problem_ids if pid not in by_id]
        if missing_problem_ids:
            raise ValueError(
                f"Unknown problem_id(s): {missing_problem_ids}. "
                f"Benchmark={benchmark_name!r} has {len(all_problems)} problems."
            )

        seen: set[str] = set()
        ordered_problem_ids: list[str] = []
        for pid in cleaned_problem_ids:
            if pid in seen:
                continue
            seen.add(pid)
            ordered_problem_ids.append(pid)
        problems = [by_id[pid] for pid in ordered_problem_ids]
    else:
        problems = all_problems[:limit]
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
    failed_out_path = str(Path(out_path).with_name(f"{Path(out_path).stem}_failed.jsonl"))

    existing_rollouts_by_problem = _existing_rollouts_by_problem(out_path)
    prepared_tasks: list[dict[str, Any]] = []
    missing_rollouts_by_problem: dict[str, list[int]] = {}
    would_write = 0
    skipped = 0

    for problem in problems:
        existing_rollouts = existing_rollouts_by_problem.setdefault(problem.problem_id, set())
        for rollout_id in range(num_rollouts):
            if rollout_id in existing_rollouts:
                skipped += 1
                continue

            missing_rollouts_by_problem.setdefault(problem.problem_id, []).append(rollout_id)
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
        _log(
            f"[hint_generation] dry_run benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} "
            f"thinking_enabled={thinking_enabled} thinking_effort={thinking_effort if thinking_enabled else 'n/a'} "
            f"would_write={would_write} skipped={skipped} "
            f"output={out_path}"
        )
        if not missing_rollouts_by_problem:
            _log("[hint_generation] dry_run missing_rollouts none")
        else:
            for problem_id in sorted(missing_rollouts_by_problem.keys()):
                missing_rollouts = sorted(missing_rollouts_by_problem[problem_id])
                _log(
                    f"[hint_generation] dry_run missing_rollouts "
                    f"problem_id={problem_id} missing_count={len(missing_rollouts)} "
                    f"rollout_ids={missing_rollouts}"
                )
    else:
        written = 0
        failed = 0
        failed_attempts_written = 0

        if concurrency == 1:
            for task in prepared_tasks:
                record, failed_attempts = _generate_record_for_task(
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
                    thinking_enabled=thinking_enabled,
                    thinking_effort=thinking_effort,
                )
                task_succeeded = record is not None
                for failed_attempt in failed_attempts:
                    append_jsonl(
                        failed_out_path,
                        {
                            **failed_attempt,
                            "task_succeeded": task_succeeded,
                        },
                    )
                    failed_attempts_written += 1
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
                        thinking_enabled=thinking_enabled,
                        thinking_effort=thinking_effort,
                    )
                    for task in prepared_tasks
                ]
                for future in as_completed(futures):
                    record, failed_attempts = future.result()
                    task_succeeded = record is not None
                    for failed_attempt in failed_attempts:
                        append_jsonl(
                            failed_out_path,
                            {
                                **failed_attempt,
                                "task_succeeded": task_succeeded,
                            },
                        )
                        failed_attempts_written += 1
                    if record is None:
                        failed += 1
                        continue
                    append_jsonl(out_path, record)
                    written += 1

        _log(
            f"[hint_generation] done benchmark={benchmark_name} hint_type={hint_type} "
            f"num_problems={len(problems)} rollouts={num_rollouts} concurrency={concurrency} "
            f"thinking_enabled={thinking_enabled} thinking_effort={thinking_effort if thinking_enabled else 'n/a'} "
            f"accepted={written} rejected={failed_attempts_written} "
            f"written={written} skipped={skipped} failed={failed} failed_attempts_logged={failed_attempts_written} "
            f"output={out_path} failed_output={failed_out_path}"
        )
    return out_path
