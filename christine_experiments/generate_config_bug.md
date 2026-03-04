# GenerateConfig Bug: Parameters Silently Dropped

## The Bug

In `src/evals/solvers/components.py`, our custom `generate()` solver passes a
`GenerateConfig` object as `config=gen_config` to Inspect's `Generate.__call__()`.
But `Generate.__call__` takes `**kwargs: Unpack[GenerateConfigArgs]`, and `"config"`
is not a valid key in `GenerateConfigArgs`. The kwarg is silently swallowed.

```python
# What we do (BROKEN):
state = await gen(state, config=gen_config)

# What Inspect expects:
state = await gen(state, max_tokens=8192, temperature=0.6, ...)
```

Additionally, `base_experiment.py` does NOT pass `max_tokens` to `inspect_eval()`
in either the checkpoint or non-checkpoint path. (In contrast, `runner.py` does.)

## What Was Affected

| Parameter | Intended Value | Actually Used (vLLM ≥0.8) | Actually Used (vLLM <0.8) | Source |
|---|---|---|---|---|
| **max_tokens** | 8192 | None → unconstrained (up to max_model_len=32768) | Same | `hinted_baselines.py` MAX_TOKENS |
| **temperature** (Qwen3) | 0.6 | 0.6 (from HF generation_config.json) | 1.0 (vLLM hardcoded default) | `model_config.py` |
| **temperature** (others) | 1.0 | 1.0 (vLLM default) | 1.0 (vLLM default) | `model_config.py` (matches) |
| **top_p** (Qwen3) | 0.95 | 0.95 (from HF generation_config.json) | 1.0 (vLLM hardcoded default) | `model_config.py` |
| **top_k** (Qwen3) | 20 | 20 (from HF generation_config.json) | -1 (not applied) | `model_config.py` |
| **presence_penalty** (Qwen3) | 1.0 | 0.0 (NOT in HF generation_config) | 0.0 | `model_config.py` |
| **timeout** | 3000s | None → no Inspect-level timeout | Same | `base_experiment.py` |
| **continue_final_message** | True (for prefill) | Not passed, but auto-detected | Same | components.py |

## Qwen3 HuggingFace generation_config.json

Starting with vLLM v0.8.0, `--generation-config` defaults to `"auto"`, meaning vLLM
loads `generation_config.json` from the HuggingFace model and uses those as server
defaults when parameters are omitted from the API request.

All Qwen3 models (0.6B–32B) ship with identical `generation_config.json`:
```json
{
    "do_sample": true,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95
}
```

**This means**: On vLLM ≥0.8, temperature/top_p/top_k were actually correct for
Qwen3 even without our GenerateConfig, because vLLM used the HF defaults.
Only `presence_penalty=1.0` was truly lost (not in HF generation_config).

Our vLLM server (`vllm_server.py`) does NOT pass `--generation-config`, so it
uses the default `auto` behavior. The vLLM version is unpinned in requirements.txt.

### Notes on continue_final_message

Despite the config not being passed, **prefilling still works correctly**.
Inspect's vLLM provider (`inspect_ai/model/_providers/vllm.py:252`) auto-detects
when the last message is an assistant message and sets `continue_final_message=True`
in `extra_body` automatically. So this parameter was redundant.

### Notes on max_tokens

Our `max_tokens=8192` was NOT enforced anywhere in the `base_experiment.py` path:
1. The generate() solver builds a GenerateConfig with max_tokens, but it's silently dropped
2. `base_experiment.py` does NOT pass `max_tokens` to `inspect_eval()` (lines 413, 544)
3. vLLM generated up to its `max_model_len` (32768)

Note: `runner.py:run_eval()` DOES pass `max_tokens` to `inspect_eval()` (line 194),
but hinted_baselines uses `base_experiment.py`, not `runner.py`.

### Notes on temperature

For non-Qwen3 models (Qwen2.5, Llama, Gemma), the intended default was 1.0
which matches vLLM's default, so **no impact** regardless of vLLM version.

For Qwen3 models on vLLM ≥0.8: temperature, top_p, top_k were correct (from HF
generation_config.json). Only presence_penalty was wrong (0.0 instead of 1.0).

For Qwen3 models on vLLM <0.8: all sampling parameters were wrong.

## Impact by Model Family (assuming vLLM ≥0.8)

| Model Family | Affected Parameters |
|---|---|
| **Qwen3** (0.6B–32B) | presence_penalty (0.0 vs 1.0), max_tokens, timeout |
| **Qwen2.5** (1.5B–32B) | max_tokens, timeout |
| **Llama 3.1** (8B, 70B) | max_tokens, timeout |
| **Gemma 3** (4B–27B) | max_tokens, timeout |

## How vLLM Defaults Work (≥0.8)

When a parameter is `None` in `GenerateConfig`, Inspect omits it from the API request.
vLLM then checks its server defaults in this order:
1. HuggingFace `generation_config.json` (loaded at startup with `--generation-config auto`)
2. vLLM hardcoded defaults (temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0)

## The Fix

Pass parameters as individual kwargs instead of a GenerateConfig object:

```python
# CORRECT:
state = await gen(state, max_tokens=max_tokens, temperature=0.6, ...)
```

Also: `continue_final_message` is not needed (auto-detected by vLLM provider).
