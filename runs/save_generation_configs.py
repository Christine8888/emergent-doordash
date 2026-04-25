from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PARAM_KEYS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "repetition_penalty",
)

MODEL_PATHS = [
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-270m-it",

    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",

    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",

    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",

    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",


]


def _safe_model_name(model_path: str) -> str:
    return model_path.replace("/", "__")


def _extract_effective_params(config_obj: Any) -> dict[str, Any]:
    return {k: getattr(config_obj, k, None) for k in PARAM_KEYS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download generation_config.json for models listed in runs/save_generation_configs.py"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/model_generation_configs",
        help="Directory to save per-model generation config files and a summary JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from huggingface_hub import hf_hub_download
    try:
        from transformers import GenerationConfig
    except Exception:
        GenerationConfig = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any] | None] = {}

    for model_path in MODEL_PATHS:
        safe_name = _safe_model_name(model_path)
        out_path = output_dir / f"{safe_name}.generation_config.json"

        try:
            cached_path = hf_hub_download(
                repo_id=model_path,
                filename="generation_config.json",
            )
            with open(cached_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)

            effective_params: dict[str, Any] | None = None
            if GenerationConfig is not None:
                effective_cfg = GenerationConfig.from_pretrained(model_path)
                effective_params = _extract_effective_params(effective_cfg)

            summary[model_path] = {
                "generation_config": config,
                "effective_params": effective_params,
            }
            print(f"[save_generation_configs] saved model={model_path} -> {out_path}")
        except Exception as exc:
            summary[model_path] = None
            print(f"[save_generation_configs][WARN] failed model={model_path} error={exc}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[save_generation_configs] summary -> {summary_path}")


if __name__ == "__main__":
    # python -m runs.save_generation_configs
    main()
