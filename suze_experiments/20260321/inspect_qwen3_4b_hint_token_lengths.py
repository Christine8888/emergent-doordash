from __future__ import annotations

import heapq
import json
from pathlib import Path

from transformers import AutoTokenizer


# --- Editable constants ---
DATA_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_regraded")
DATASET_FAMILY = "aime_solution"
MODEL = "Qwen3-4B"
SOLVER_FILE = "solution_intext_masked.jsonl"
TOKENIZER_NAME = "Qwen/Qwen3-4B"
LOCAL_TOKENIZER_SNAPSHOT = Path(
    "/nlp/scr/suzeva/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)

# If non-empty, only these hints are analyzed (e.g. [0.0, 0.1]).
# If empty, analyze all available hint directories.
HINT_LEVELS: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TOP_K_LONGEST_BY_CHARS = 10
# --------------------------


def parse_hint_from_dirname(dirname: str) -> float:
    prefix = "hint_fraction_"
    if not dirname.startswith(prefix):
        raise ValueError(f"Unexpected hint dir name: {dirname}")
    return float(dirname[len(prefix) :])


def list_hint_levels(model_dir: Path) -> list[float]:
    hint_values = []
    for p in model_dir.iterdir():
        if p.is_dir() and p.name.startswith("hint_fraction_"):
            hint_values.append(parse_hint_from_dirname(p.name))
    return sorted(hint_values)


def get_selected_hints(model_dir: Path) -> list[float]:
    available = list_hint_levels(model_dir)
    if not available:
        raise ValueError(f"No hint dirs found under: {model_dir}")

    if HINT_LEVELS:
        selected = sorted(HINT_LEVELS)
        missing = [h for h in selected if h not in available]
        if missing:
            raise ValueError(f"Requested hints missing for {MODEL}: {missing}")
        return selected

    return available


def load_tokenizer() -> AutoTokenizer:
    if LOCAL_TOKENIZER_SNAPSHOT.exists():
        print(f"Loading tokenizer from local snapshot: {LOCAL_TOKENIZER_SNAPSHOT}")
        return AutoTokenizer.from_pretrained(
            str(LOCAL_TOKENIZER_SNAPSHOT),
            trust_remote_code=True,
            local_files_only=True,
        )

    print(f"Loading tokenizer from model id: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
    )


def inspect_hint_file(path: Path, tokenizer: AutoTokenizer) -> dict[str, object]:
    rollouts = 0
    missing_output_text = 0
    total_chars = 0
    top_k: list[tuple[int, str, object, object, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_obj = json.loads(line)
            sample_id = sample_obj.get("sample_id")
            sample_rollouts = sample_obj.get("rollouts")
            if not isinstance(sample_rollouts, list):
                continue

            for rollout in sample_rollouts:
                if not isinstance(rollout, dict):
                    continue
                text_obj = rollout.get("output_text")
                if text_obj is None:
                    missing_output_text += 1
                    text = ""
                else:
                    text = str(text_obj)

                char_count = len(text)
                total_chars += char_count
                rollouts += 1

                candidate = (
                    char_count,
                    str(sample_id),
                    rollout.get("epoch"),
                    rollout.get("rollout_id"),
                    text,
                )
                if len(top_k) < TOP_K_LONGEST_BY_CHARS:
                    heapq.heappush(top_k, candidate)
                elif candidate[0] > top_k[0][0]:
                    heapq.heapreplace(top_k, candidate)

    top_k_sorted = sorted(top_k, key=lambda x: x[0], reverse=True)
    top_token_rows: list[dict[str, object]] = []
    max_tokens = -1
    max_meta: dict[str, object] = {}
    for rank, (char_count, sample_id, epoch, rollout_id, text) in enumerate(top_k_sorted, start=1):
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        top_token_rows.append(
            {
                "rank_by_chars": rank,
                "chars": char_count,
                "tokens": token_count,
                "sample_id": sample_id,
                "epoch": epoch,
                "rollout_id": rollout_id,
            }
        )
        if token_count > max_tokens:
            max_tokens = token_count
            max_meta = {
                "sample_id": sample_id,
                "epoch": epoch,
                "rollout_id": rollout_id,
                "chars": char_count,
            }

    avg_chars = (total_chars / rollouts) if rollouts else 0.0
    max_chars = top_k_sorted[0][0] if top_k_sorted else 0
    return {
        "rollouts": rollouts,
        "missing_output_text": missing_output_text,
        "avg_chars": avg_chars,
        "max_chars": max_chars,
        "max_tokens_among_top_k_chars": max_tokens,
        "max_meta": max_meta,
        "top_token_rows": top_token_rows,
    }


def main() -> None:
    model_dir = DATA_ROOT / DATASET_FAMILY / MODEL
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    tokenizer = load_tokenizer()

    hints = get_selected_hints(model_dir)
    print(f"model={MODEL} solver={SOLVER_FILE}")
    print(f"hints={hints}")
    print("\n=== Per-hint token stats (streaming) ===")

    rows: list[dict[str, object]] = []
    for hint in hints:
        hint_dir = model_dir / f"hint_fraction_{hint:.1f}"
        path = hint_dir / SOLVER_FILE
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        stats = inspect_hint_file(path, tokenizer)
        row = {"hint": hint, **stats}
        rows.append(row)
        meta = row["max_meta"]
        print(
            f"hint={row['hint']:.1f} "
            f"rollouts={row['rollouts']} "
            f"avg_chars={row['avg_chars']:.2f} "
            f"max_chars={row['max_chars']} "
            f"max_tokens_among_top_{TOP_K_LONGEST_BY_CHARS}_chars={row['max_tokens_among_top_k_chars']} "
            f"missing_output_text={row['missing_output_text']} "
            f"max_sample_id={meta.get('sample_id')} "
            f"max_epoch={meta.get('epoch')} "
            f"max_rollout_id={meta.get('rollout_id')}",
            flush=True,
        )

    rows = sorted(rows, key=lambda r: float(r["hint"]))

    ordered = sorted(rows, key=lambda r: int(r["max_tokens_among_top_k_chars"]), reverse=True)
    print(f"\n=== Max tokens ranking (highest to lowest, among top {TOP_K_LONGEST_BY_CHARS} chars per hint) ===")
    for row in ordered:
        meta = row["max_meta"]
        print(
            f"hint={row['hint']:.1f} "
            f"max_tokens={row['max_tokens_among_top_k_chars']} "
            f"chars={meta.get('chars')} "
            f"sample_id={meta.get('sample_id')} "
            f"epoch={meta.get('epoch')}"
        )


if __name__ == "__main__":
    # python suze_experiments/20260321/inspect_qwen3_4b_hint_token_lengths.py
    main()
