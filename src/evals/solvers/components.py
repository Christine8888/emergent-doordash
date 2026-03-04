"""Modular solver components for composing evaluation pipelines.

These components can be composed in any order:
    solver=[instructions(...), fewshot(...), prefill(...), generate()]

Or:
    solver=[fewshot(...), prefill(...), instructions(...), generate()]
"""

import logging
import os
import random
import threading
import time
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem
from inspect_ai.solver import Generate, Solver, solver
from inspect_ai.solver import TaskState

from evals.prefill import PrefillConfig, get_masked_text
from evals.fewshot import FewShotConfig, format_fewshot_examples
from utils.model_config import get_start_prefill

logger = logging.getLogger(__name__)

_LENGTH_WARN_COUNT = 0
_TOKEN_METRICS_LOCK = threading.Lock()
_TOKEN_METRICS_START_TS = time.time()
_TOKEN_METRICS_LAST_PRINT_TS = _TOKEN_METRICS_START_TS
_TOKEN_METRICS_TOTAL_OUTPUT_TOKENS = 0
_TOKEN_METRICS_TOTAL_REQUESTS = 0
_TOKEN_METRICS_TOTAL_WITH_USAGE = 0
_TOKEN_METRICS_LAST_PRINT_TOKENS = 0
_TOKEN_METRICS_LAST_PRINT_REQUESTS = 0
_TOKEN_METRICS_PRINT_INTERVAL_SEC = float(os.environ.get("EXPERIMENT_TOKENS_PER_SEC_PRINT_INTERVAL", "30"))
_MAX_TOKENS_IGNORED_WARNED = False


def _is_length_stop(output) -> bool:
    """Best-effort detection of max_tokens truncation across model providers."""
    if output is None:
        return False

    # Common direct attributes.
    for attr in ("finish_reason", "stop_reason", "reason"):
        val = getattr(output, attr, None)
        if isinstance(val, str) and val.lower() in ("length", "max_tokens", "token_limit"):
            return True

    # OpenAI-style: output.choices[0].finish_reason
    choices = getattr(output, "choices", None)
    if isinstance(choices, list) and choices:
        fr = getattr(choices[0], "finish_reason", None)
        if isinstance(fr, str) and fr.lower() == "length":
            return True

        # Sometimes nested under message / metadata.
        msg = getattr(choices[0], "message", None)
        md = getattr(msg, "metadata", None)
        if isinstance(md, dict):
            fr2 = md.get("finish_reason") or md.get("stop_reason")
            if isinstance(fr2, str) and fr2.lower() == "length":
                return True

    # Metadata dict.
    md = getattr(output, "metadata", None)
    if isinstance(md, dict):
        fr = md.get("finish_reason") or md.get("stop_reason")
        if isinstance(fr, str) and fr.lower() == "length":
            return True

    return False


def _warn_length_stop(state: TaskState, max_tokens: int | None):
    global _LENGTH_WARN_COUNT
    _LENGTH_WARN_COUNT += 1
    ts = time.strftime("%m/%d %H:%M:%S")
    sid = getattr(state, "sample_id", None)
    epoch = getattr(state, "epoch", None)
    # Print to stdout so it appears alongside Inspect progress lines.
    print(
        f"[{ts}] WARNING: generation hit max_tokens (max_tokens={max_tokens}) "
        f"sample_id={sid!r} epoch={epoch!r} (count={_LENGTH_WARN_COUNT})",
        flush=True,
    )


def _extract_output_tokens(output) -> int | None:
    """Best-effort extraction of generated output tokens from model output."""
    if output is None:
        return None

    usage = getattr(output, "usage", None)
    if usage is not None:
        # Dataclass/object usage (Inspect style).
        for attr in ("output_tokens", "completion_tokens"):
            val = getattr(usage, attr, None)
            if isinstance(val, int):
                return val
        # Dict usage (OpenAI-style dict payloads).
        if isinstance(usage, dict):
            for key in ("output_tokens", "completion_tokens"):
                val = usage.get(key)
                if isinstance(val, int):
                    return val

    # Fallbacks in metadata.
    md = getattr(output, "metadata", None)
    if isinstance(md, dict):
        for key in ("output_tokens", "completion_tokens"):
            val = md.get(key)
            if isinstance(val, int):
                return val

    return None


def _record_tokens_per_second_metric(output) -> None:
    """Track rolling output token throughput and print periodic progress."""
    if _TOKEN_METRICS_PRINT_INTERVAL_SEC <= 0:
        return

    out_tokens = _extract_output_tokens(output)
    now = time.time()

    global _TOKEN_METRICS_LAST_PRINT_TS
    global _TOKEN_METRICS_TOTAL_OUTPUT_TOKENS
    global _TOKEN_METRICS_TOTAL_REQUESTS
    global _TOKEN_METRICS_TOTAL_WITH_USAGE
    global _TOKEN_METRICS_LAST_PRINT_TOKENS
    global _TOKEN_METRICS_LAST_PRINT_REQUESTS

    with _TOKEN_METRICS_LOCK:
        _TOKEN_METRICS_TOTAL_REQUESTS += 1
        if isinstance(out_tokens, int):
            _TOKEN_METRICS_TOTAL_OUTPUT_TOKENS += out_tokens
            _TOKEN_METRICS_TOTAL_WITH_USAGE += 1

        since_last = now - _TOKEN_METRICS_LAST_PRINT_TS
        if since_last < _TOKEN_METRICS_PRINT_INTERVAL_SEC:
            return

        elapsed = max(now - _TOKEN_METRICS_START_TS, 1e-6)
        delta_tokens = _TOKEN_METRICS_TOTAL_OUTPUT_TOKENS - _TOKEN_METRICS_LAST_PRINT_TOKENS
        delta_requests = _TOKEN_METRICS_TOTAL_REQUESTS - _TOKEN_METRICS_LAST_PRINT_REQUESTS
        avg_tps = _TOKEN_METRICS_TOTAL_OUTPUT_TOKENS / elapsed
        win_tps = delta_tokens / max(since_last, 1e-6)

        ts = time.strftime("%m/%d %H:%M:%S")
        print(
            f"[{ts}] throughput: output_tokens/s avg={avg_tps:.1f} window={win_tps:.1f} "
            f"total_out_tokens={_TOKEN_METRICS_TOTAL_OUTPUT_TOKENS} "
            f"requests={_TOKEN_METRICS_TOTAL_REQUESTS} "
            f"requests_with_usage={_TOKEN_METRICS_TOTAL_WITH_USAGE} "
            f"window_requests={delta_requests}",
            flush=True,
        )

        _TOKEN_METRICS_LAST_PRINT_TS = now
        _TOKEN_METRICS_LAST_PRINT_TOKENS = _TOKEN_METRICS_TOTAL_OUTPUT_TOKENS
        _TOKEN_METRICS_LAST_PRINT_REQUESTS = _TOKEN_METRICS_TOTAL_REQUESTS


@solver
def instructions(template: str) -> Solver:
    """Add instructions to the beginning of the user prompt.

    Args:
        template: Instruction text to prepend

    Returns:
        Solver that adds instructions
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = template + "\n\n" + state.user_prompt.text
        return state

    return solve


@solver
def fewshot(
    config: FewShotConfig,
    example_template: str = "{question}\n{response}",
) -> Solver:
    """Add few-shot examples to the user prompt.

    **How it works:**
    This solver APPENDS examples to the end of the current prompt text.
    Always excludes the current problem from few-shot selection.

    **Execution order matters:**
    - [instructions(), fewshot()] → [Instructions][Problem][Examples]
    - [fewshot(), instructions()] → [Examples][Instructions][Problem] ← weird!

    **Typical usage:**
    Put fewshot() AFTER instructions() to get natural order:
        solver = [
            instructions("Solve the problem."),
            fewshot(FewShotConfig(
                path="hints.jsonl",
                num_examples=3,
                prefix="Here are some examples:",
                suffix="Now solve:"
            )),
            prefill(config),
            generate()
        ]

    Args:
        config: FewShotConfig with path, num_examples, seed, prefix, suffix
        example_template: Format string with {question} and {response} (default: "{question}\\n{response}")

    Returns:
        Solver that appends few-shot examples

    Example final prompt structure:
        [Instructions from instructions()]

        [Current problem]

        [config.prefix if provided]

        [Example 1]
        [Example 2]

        [config.suffix if provided]
    """
    fewshot_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        examples_text = format_fewshot_examples(
            fewshot_data=fewshot_data,
            n_examples=config.num_examples,
            example_template=example_template,
            current_id=str(state.sample_id) if config.exclude_current else None,
            seed=config.seed,
            prefix=config.prefix,
            suffix=config.suffix,
        )

        # APPEND examples to current prompt (not prepend!)
        if examples_text:
            state.user_prompt.text = state.user_prompt.text + "\n\n" + examples_text

        return state

    return solve


@solver
def prefill(config: PrefillConfig) -> Solver:
    """Add prefill text as an assistant message.

    For masked mode, masking is applied on the fly with an epoch-dependent seed
    so each epoch sees a different mask (variance reduction over mask position
    and hint sample).

    Args:
        config: PrefillConfig with path and fraction settings

    Returns:
        Solver that adds prefill for the current sample

    Raises:
        KeyError: If sample_id is not in prefill data (when fraction > 0.0)
    """
    prefill_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if config.fraction > 0.0:
            sid = str(state.sample_id)
            if sid not in prefill_data:
                raise KeyError(
                    f"Sample '{state.sample_id}' not found in prefill data. "
                    f"Available samples should be filtered using config.get_available_ids()"
                )
            samples = prefill_data[sid]
            rng = random.Random(f"{state.epoch}_{state.sample_id}")
            sample_keys = sorted(samples.keys())
            chosen_idx = rng.choice(sample_keys)
            prefill_text = samples[chosen_idx]

            # For masked mode, apply mask on the fly (unseeded for max variance)
            if config.mode == "masked":
                prefill_text = get_masked_text(
                    prefill_text, fraction=config.fraction,
                    mask_token=config.mask_token,
                )

            # Prepend model-specific start token if applicable (e.g., "<think>" for Qwen3)
            model_name = os.environ.get("INSPECT_EVAL_MODEL", "")
            start_token = get_start_prefill(model_name)
            if start_token:
                prefill_text = start_token + prefill_text

            state.messages.append(ChatMessageAssistant(content=prefill_text))

        return state

    return solve


@solver
def intext(config: PrefillConfig, prefix: str = "Here is part of a hint that may be helpful to your solution:\n") -> Solver:
    """Add hint text inline to the user prompt.

    Similar to prefill() but appends hint text to the user prompt instead of
    adding an assistant message. For masked mode, masking is applied on the fly
    with an epoch-dependent seed for variance reduction.

    Args:
        config: PrefillConfig with path, fraction, mode, and mask_token settings
        prefix: Text to prepend to the hint (default: "Here is part of a hint that may be helpful to your solution:\\n")

    Returns:
        Solver that appends hint text to user prompt for the current sample

    Raises:
        KeyError: If sample_id is not in prefill data (when fraction > 0.0)
    """
    hint_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if config.fraction > 0.0:
            sid = str(state.sample_id)
            if sid not in hint_data:
                raise KeyError(
                    f"Sample '{state.sample_id}' not found in hint data. "
                    f"Available samples should be filtered using config.get_available_ids()"
                )
            samples = hint_data[sid]
            rng = random.Random(f"{state.epoch}_{state.sample_id}")
            sample_keys = sorted(samples.keys())
            chosen_idx = rng.choice(sample_keys)
            hint_text = samples[chosen_idx]

            # For masked mode, apply mask on the fly with epoch-dependent seed
            if config.mode == "masked":
                mask_seed = f"{state.epoch}_{state.sample_id}_{chosen_idx}"
                hint_text = get_masked_text(
                    hint_text, fraction=config.fraction,
                    mask_token=config.mask_token, seed=mask_seed,
                )

            state.user_prompt.text = state.user_prompt.text + "\n\n" + prefix + hint_text

        return state

    return solve


@solver
def system_message(message: str) -> Solver:
    """Add a system message to the conversation.

    Args:
        message: System message content

    Returns:
        Solver that adds the system message
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=message))
        return state

    return solve


@solver
def generate(
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Generate with automatic continuation detection.

    If the last message is an assistant message, enables continue_final_message
    for vLLM to continue from that message.

    Args:
        max_tokens: Deprecated/ignored. Kept for backward compatibility.
        timeout: Timeout in seconds (total across all retries)

    Returns:
        Solver that generates with appropriate configuration
    """
    async def solve(state: TaskState, gen: Generate) -> TaskState:
        # Auto-detect if we should continue from last message
        continue_message = (
            len(state.messages) > 0 and
            isinstance(state.messages[-1], ChatMessageAssistant)
        )

        gen_kwargs = {
            "continue_final_message": continue_message,
        }
        if timeout is not None:
            gen_kwargs["timeout"] = timeout

        global _MAX_TOKENS_IGNORED_WARNED
        if max_tokens is not None and not _MAX_TOKENS_IGNORED_WARNED:
            logger.warning(
                "generate(max_tokens=...) is ignored to preserve provider defaults."
            )
            _MAX_TOKENS_IGNORED_WARNED = True

        # Leave generation params (temperature/top_p/top_k/etc.) unset
        # so provider defaults are used.
        state = await gen(state, **gen_kwargs)

        # Warn if estimated input tokens exceed 80% of the context window.
        try:
            max_model_len_str = os.environ.get("VLLM_MAX_MODEL_LEN")
            if max_model_len_str:
                max_model_len = int(max_model_len_str)
                total_chars = sum(
                    len(m.content) if isinstance(m.content, str) else
                    sum(len(c.text) if hasattr(c, "text") else 0 for c in m.content)
                    for m in state.messages
                )
                est_input_tokens = total_chars // 4  # rough estimate (~4 chars/token)
                threshold = int(0.8 * max_model_len)
                if est_input_tokens > threshold:
                    ts = time.strftime("%m/%d %H:%M:%S")
                    sid = getattr(state, "sample_id", None)
                    epoch = getattr(state, "epoch", None)
                    remaining = max_model_len - est_input_tokens
                    pct = 100 * est_input_tokens / max_model_len
                    print(
                        f"[{ts}] WARNING: estimated input ~{est_input_tokens} tokens "
                        f"({pct:.0f}% of max_model_len={max_model_len}); "
                        f"only ~{remaining} tokens left for output "
                        f"sample_id={sid!r} epoch={epoch!r}",
                        flush=True,
                    )
        except Exception:
            pass

        # Best-effort warning if generation appears to have been truncated by max_tokens.
        try:
            if max_tokens is not None and _is_length_stop(getattr(state, "output", None)):
                _warn_length_stop(state, max_tokens=max_tokens)
        except Exception:
            # Never let warning logic break eval execution.
            pass

        # Best-effort rolling throughput telemetry for live progress monitoring.
        try:
            _record_tokens_per_second_metric(getattr(state, "output", None))
        except Exception:
            # Never let telemetry logic break eval execution.
            pass

        return state

    return solve
