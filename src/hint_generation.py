from __future__ import annotations
import base64
import httpx
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import re
import threading
from typing import Any

from src.datasets import get_dataset_spec
from src.hint_types import MissingSourceHintError, get_hint_type_spec
from src.storage import append_jsonl, build_hint_generation_path, make_stable_id, read_jsonl
from src.types import HintGenerationRecord

ANTHROPIC_MODELS: set[str] = {
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
}

OPENAI_MODELS: set[str] = {
    "gpt-5.4",
    "gpt-5.5-2026-04-23"
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


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


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


def _append_output_block(blocks: list[dict[str, str]], block_type: str, text: str | None) -> None:
    if not isinstance(text, str) or not text:
        return
    if blocks and blocks[-1].get("type") == block_type:
        blocks[-1]["text"] = blocks[-1].get("text", "") + text
        return
    blocks.append({"type": block_type, "text": text})


def _anthropic_message_output_blocks(message: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            _append_output_block(blocks, "text", getattr(block, "text", None))
        elif block_type == "thinking":
            _append_output_block(blocks, "thinking", getattr(block, "thinking", None))
        elif block_type == "redacted_thinking":
            _append_output_block(blocks, "redacted_thinking", "[redacted thinking]")
    return blocks


def _join_output_blocks(blocks: list[dict[str, str]], block_type: str) -> str:
    return "".join(
        block.get("text", "")
        for block in blocks
        if block.get("type") == block_type and isinstance(block.get("text"), str)
    ).strip()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _dump_model_debug_response(
    *,
    provider: str,
    model: str,
    final_response: Any,
    stream_events: list[Any],
    output_blocks: list[dict[str, str]],
) -> str:
    debug_dir = Path("data") / "debug" / "model_responses"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = model.replace("/", "_")
    path = debug_dir / f"{timestamp}_{threading.get_ident()}_{provider}_{safe_model}.json"
    payload = {
        "provider": provider,
        "model": model,
        "final_response": _jsonable(final_response),
        "stream_events": [_jsonable(event) for event in stream_events],
        "parsed_output_blocks": output_blocks,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return str(path)


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


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _get_attr_or_key(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _token_breakdown_metadata(usage: dict[str, Any]) -> dict[str, Any]:
    output_tokens = _maybe_int(usage.get("output_token_count"))
    reasoning_tokens = _maybe_int(usage.get("reasoning_token_count"))
    thinking_tokens = _maybe_int(usage.get("thinking_token_count"))
    reasoning_output_tokens = reasoning_tokens if reasoning_tokens is not None else thinking_tokens
    thinking_text = usage.get("thinking")
    thinking_observed = isinstance(thinking_text, str) and bool(thinking_text.strip())
    if not thinking_observed:
        for block in usage.get("output_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"thinking", "redacted_thinking"}:
                thinking_observed = True
                break

    normal_output_tokens = None if thinking_observed and reasoning_output_tokens is None else output_tokens
    if output_tokens is not None and reasoning_output_tokens is not None:
        normal_output_tokens = max(0, output_tokens - reasoning_output_tokens)
    metadata = {
        "reasoning_token_count": reasoning_tokens,
        "thinking_token_count": thinking_tokens,
        "normal_output_tokens": normal_output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "thinking_observed": thinking_observed,
    }
    if thinking_observed and reasoning_output_tokens is None:
        metadata["token_breakdown_note"] = (
            "Thinking was observed, but this provider response did not include a separate "
            "reasoning/thinking token count; output_token_count is the total billed output."
        )
    return metadata


def _thinking_summary_available(usage: dict[str, Any]) -> bool:
    thinking = usage.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return True
    for block in usage.get("output_blocks", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "thinking":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            return True
    return False


def _token_usage_log_fields(usage: dict[str, Any]) -> str:
    thinking = "true" if _thinking_summary_available(usage) else "false"
    return f"output_tokens={usage.get('output_token_count')} thinking={thinking}"


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


def _problem_record_metadata(problem) -> dict[str, Any]:
    metadata = dict(problem.metadata or {})
    raw_example = metadata.get("raw_example")
    problem_id_hle = metadata.get("problem_id_hle") or metadata.get("id")
    return {
        "problem_id_hle": problem_id_hle,
        "problem_metadata": {
            "problem_id_hle": problem_id_hle,
            "answer_type": metadata.get("answer_type"),
            "text_only": metadata.get("text_only"),
            "category": metadata.get("category"),
            "raw_subject": metadata.get("raw_subject"),
            "has_image": bool(str(metadata.get("image") or "").strip()),
            "has_image_preview": bool(
                isinstance(raw_example, dict) and raw_example.get("image_preview") is not None
            ),
            "has_rationale_image": bool(
                isinstance(raw_example, dict) and raw_example.get("rationale_image") is not None
            ),
            "author_name": raw_example.get("author_name") if isinstance(raw_example, dict) else None,
            "canary": raw_example.get("canary") if isinstance(raw_example, dict) else None,
        }
    }


def _openai_prompt_content(prompt: PromptContent) -> str | list[dict[str, Any]]:
    if isinstance(prompt, str):
        return prompt

    content: list[dict[str, Any]] = []
    for part in prompt:
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
            continue
        if part_type == "image":
            source = part.get("source")
            if not isinstance(source, dict):
                raise ValueError("Invalid image part: missing source.")
            source_type = source.get("type")
            media_type = source.get("media_type")
            data = source.get("data")
            if source_type != "base64" or not isinstance(media_type, str) or not isinstance(data, str):
                raise ValueError("OpenAI image parts require base64 source data and media_type.")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }
            )
            continue
        raise ValueError(f"Unsupported prompt content part for OpenAI: {part_type!r}")
    return content


def query_anthropic_hint(
    prompt: PromptContent,
    model: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    thinking_effort: str,
    save_debug_responses: bool,
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
        request["thinking"] = {"type": thinking_mode, "display": "summarized"}
        request["extra_body"] = {"output_config": {"effort": effort}}

    streamed_output_blocks: list[dict[str, str]] = []
    stream_events: list[Any] = []
    with client.messages.stream(**request) as stream:
        for event in stream:
            stream_events.append(event)
            event_type = getattr(event, "type", None)
            if event_type != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                _append_output_block(streamed_output_blocks, "text", getattr(delta, "text", None))
            elif delta_type == "thinking_delta":
                _append_output_block(streamed_output_blocks, "thinking", getattr(delta, "thinking", None))
        response = stream.get_final_message()

    final_output_blocks = _anthropic_message_output_blocks(response)
    output_blocks = streamed_output_blocks or final_output_blocks
    model_output = _join_output_blocks(output_blocks, "text")
    thinking = _join_output_blocks(output_blocks, "thinking")
    if not model_output:
        model_output = _parse_anthropic_message_text(response)
    if not thinking:
        thinking = _parse_anthropic_message_thinking(response)

    debug_dump_path = None
    response_output_tokens = int(response.usage.output_tokens)
    response_stop_reason = getattr(response, "stop_reason", None)
    if save_debug_responses:
        debug_dump_path = _dump_model_debug_response(
            provider="anthropic",
            model=model,
            final_response=response,
            stream_events=stream_events,
            output_blocks=output_blocks,
        )

    return {
        "model_output": model_output,
        "thinking": thinking,
        "output_blocks": output_blocks,
        "debug_dump_path": debug_dump_path,
        "provider": "anthropic",
        "thinking_enabled": thinking_enabled,
        "thinking_mode": thinking_mode,
        "effort": effort,
        "input_token_count": int(response.usage.input_tokens),
        "output_token_count": response_output_tokens,
        "reasoning_token_count": None,
        "thinking_token_count": _maybe_int(getattr(response.usage, "thinking_tokens", None)),
        "stop_reason": response_stop_reason,
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
    save_debug_responses: bool,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(timeout=httpx.Timeout(7200, connect=30))
    effort = thinking_effort if thinking_enabled else "none"
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _openai_prompt_content(prompt)})
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
    completion_token_details = _get_attr_or_key(usage, "completion_tokens_details")
    reasoning_token_count = _maybe_int(
        _get_attr_or_key(completion_token_details, "reasoning_tokens")
    )
    model_output = _parse_openai_message_text(content)
    debug_dump_path = None
    if save_debug_responses:
        debug_dump_path = _dump_model_debug_response(
            provider="openai",
            model=model,
            final_response=completion,
            stream_events=[],
            output_blocks=[{"type": "text", "text": model_output}] if model_output else [],
        )
    return {
        "model_output": model_output,
        "thinking": "",
        "output_blocks": [{"type": "text", "text": model_output}] if model_output else [],
        "debug_dump_path": debug_dump_path,
        "provider": "openai",
        "thinking_enabled": thinking_enabled,
        "thinking_mode": "openai_reasoning_effort" if thinking_enabled else "disabled",
        "effort": effort,
        "input_token_count": int(_get_attr_or_key(usage, "prompt_tokens") or 0),
        "output_token_count": int(_get_attr_or_key(usage, "completion_tokens") or 0),
        "reasoning_token_count": reasoning_token_count,
        "thinking_token_count": reasoning_token_count,
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
    save_debug_responses: bool,
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
            save_debug_responses=save_debug_responses,
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
            save_debug_responses=save_debug_responses,
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
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    thinking_effort: str,
    save_debug_responses: bool,
) -> tuple[HintGenerationRecord | None, list[dict[str, Any]]]:
    successful_usage = None
    successful_model = None
    successful_extracted = None
    successful_full_hint = None
    successful_grader_metadata: dict[str, Any] = {}
    failed_attempts: list[dict[str, Any]] = []
    attempt_plan = [(first_model, 1)]
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
                    save_debug_responses=save_debug_responses,
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
                f"{_token_usage_log_fields(usage)} "
                f"stop_reason={usage['stop_reason']}"
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
                        **_token_breakdown_metadata(usage),
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "output_blocks": usage.get("output_blocks", []),
                        "debug_dump_path": usage.get("debug_dump_path"),
                        "thinking_enabled": usage["thinking_enabled"],
                        "thinking_mode": usage["thinking_mode"],
                        "effort": usage["effort"],
                        "max_tokens": max_tokens,
                        **context_metadata,
                    }
                )
                debug_dump_log = (
                    f" debug_dump={usage['debug_dump_path']}"
                    if usage.get("debug_dump_path")
                    else ""
                )
                _log(
                    f"[hint_generation][WARN] max_tokens_reached benchmark={benchmark_name} "
                    f"problem_id={problem.problem_id} rollout_id={rollout_id} attempt={attempt_idx} "
                    f"model={attempt_model} "
                    f"{_token_usage_log_fields(usage)} "
                    f"stop_reason={usage['stop_reason']}"
                    f"{debug_dump_log}"
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
                        **_token_breakdown_metadata(usage),
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "output_blocks": usage.get("output_blocks", []),
                        "debug_dump_path": usage.get("debug_dump_path"),
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
                        **_token_breakdown_metadata(usage),
                        "stop_reason": usage["stop_reason"],
                        "thinking": usage["thinking"],
                        "output_blocks": usage.get("output_blocks", []),
                        "debug_dump_path": usage.get("debug_dump_path"),
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
                **_problem_record_metadata(problem),
                "temperature": temperature,
                "system_prompt": system_prompt,
                "prompt_image_metadata": prompt_image_metadata,
                "extracted_answer": successful_extracted,
                "grader_metadata": successful_grader_metadata,
                "first_model": first_model,
                "provider": successful_usage["provider"],
                **_token_breakdown_metadata(successful_usage),
                "total_attempts_used": attempt_idx,
                "stop_reason": successful_usage["stop_reason"],
                "thinking": successful_usage["thinking"],
                "output_blocks": successful_usage.get("output_blocks", []),
                "debug_dump_path": successful_usage.get("debug_dump_path"),
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
    num_rollouts: int,
    limit: int,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    problem_ids: list[str] | None = None,
    thinking_enabled: bool = True,
    thinking_effort: str = "medium",
    concurrency: int = 1,
    hle_modality: str = "all",
    save_debug_responses: bool = False,
) -> str:
    """Generate hint records and append them to a JSONL file."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    thinking_enabled = _coerce_bool(thinking_enabled)
    save_debug_responses = _coerce_bool(save_debug_responses)
    if thinking_enabled and thinking_effort not in {"low", "medium", "high", "max"}:
        raise ValueError("thinking_effort must be one of: low, medium, high, max")
    if hle_modality not in {"all", "text-only", "with-images"}:
        raise ValueError("hle_modality must be one of: all, text-only, with-images")
    _provider_for_model_id(first_model)

    dataset_spec = get_dataset_spec(benchmark_name)
    hint_type_spec = get_hint_type_spec(hint_type)
    all_problems = dataset_spec.load_problems()
    if benchmark_name == "hle" and hle_modality != "all":
        want_text_only = hle_modality == "text-only"
        all_problems = [
            problem
            for problem in all_problems
            if bool(problem.metadata.get("text_only")) == want_text_only
        ]
    elif hle_modality != "all":
        raise ValueError("--hle-modality can only be used with --benchmark hle")
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
    skipped_missing_source = 0
    missing_source_examples: list[str] = []

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

            try:
                generation_context = hint_type_spec.build_context(
                    benchmark_name=benchmark_name,
                    problem=problem,
                    rollout_id=rollout_id,
                )
            except MissingSourceHintError as exc:
                skipped_missing_source += 1
                if len(missing_source_examples) < 10:
                    missing_source_examples.append(
                        f"problem_id={problem.problem_id} rollout_id={rollout_id} reason={exc}"
                    )
                continue
            prompt = hint_type_spec.build_prompt(
                problem=problem,
                context=generation_context,
            )
            missing_rollouts_by_problem.setdefault(problem.problem_id, []).append(rollout_id)
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
            f"would_write={would_write} skipped={skipped} skipped_missing_source={skipped_missing_source} "
            f"save_debug_responses={save_debug_responses} "
            f"output={out_path}"
        )
        if skipped_missing_source:
            for example in missing_source_examples:
                _log(f"[hint_generation] dry_run skipped_missing_source {example}")
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
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_enabled=thinking_enabled,
                    thinking_effort=thinking_effort,
                    save_debug_responses=save_debug_responses,
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
                future_to_task = {
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
                        max_tokens=max_tokens,
                        temperature=temperature,
                        thinking_enabled=thinking_enabled,
                        thinking_effort=thinking_effort,
                        save_debug_responses=save_debug_responses,
                    ): task
                    for task in prepared_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        record, failed_attempts = future.result()
                    except Exception as exc:
                        problem = task["problem"]
                        append_jsonl(
                            failed_out_path,
                            {
                                "hint_id": task["hint_id"],
                                "problem_id": problem.problem_id,
                                "benchmark_name": benchmark_name,
                                "hint_type": hint_type,
                                "rollout_id": task["rollout_id"],
                                "attempt": None,
                                "model": first_model,
                                "failure_type": "worker_exception",
                                "failure_error": str(exc),
                                "failure_error_type": type(exc).__name__,
                                "question": problem.question,
                                "answer": problem.answer,
                                "prompt": task["prompt"],
                                "system_prompt": hint_type_spec.system_prompt,
                                "task_succeeded": False,
                            },
                        )
                        failed_attempts_written += 1
                        failed += 1
                        _log(
                            f"[hint_generation][WARN] worker_exception benchmark={benchmark_name} "
                            f"problem_id={problem.problem_id} rollout_id={task['rollout_id']} "
                            f"model={first_model} error={exc}"
                        )
                        continue
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
            f"save_debug_responses={save_debug_responses} "
            f"accepted={written} rejected={failed_attempts_written} "
            f"written={written} skipped={skipped} skipped_missing_source={skipped_missing_source} "
            f"failed={failed} failed_attempts_logged={failed_attempts_written} "
            f"output={out_path} failed_output={failed_out_path}"
        )
    return out_path
