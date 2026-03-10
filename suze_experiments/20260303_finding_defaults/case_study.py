"""Case study: investigate Qwen3-4B accuracy shifts across owners/hint levels.

Usage:
  python suze_experiments/20260303_finding_defaults/case_study.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

MODEL = "Qwen3-4B"
EVAL_NAME = "aime"
SOLVER = "solution_intext_masked"
CONDITION = "0shot"

SUZE_HINTS = [0.6, 0.75, 0.8]
CHRISTINE_HINTS = [0.0, 0.5, 0.85, 1.0]

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

SUZE_RESULTS_DIR = REPO_ROOT / "christine_experiments/20251113/results" / EVAL_NAME / SOLVER / CONDITION / MODEL
CHRISTINE_REPO = Path("/sphinx/u/cye/emergent-doordash")
CHRISTINE_RESULTS_DIR = CHRISTINE_REPO / "christine_experiments/20251113/results" / EVAL_NAME / SOLVER / CONDITION / MODEL

SUZE_LOG_DIR = REPO_ROOT / "submitit_logs"
CHRISTINE_LOG_DIR = CHRISTINE_REPO / "christine_experiments/20251113/submitit_logs"

OUTPUT_DIR = THIS_FILE.parent
SUMMARY_CSV = OUTPUT_DIR / "case_study_qwen3_4b_summary.csv"
PROMPT_COMPARE_TXT = OUTPUT_DIR / "case_study_qwen3_4b_prompt_compare.txt"


@dataclass
class EvalMeta:
    owner: str
    path: Path
    created: str
    hint: float | None
    task_id: str | None
    model: str | None
    timeout: int | None
    max_connections: int | None
    max_retries: int | None
    inspect_version: str | None
    data_path: str | None
    dataset_samples: int | None
    epochs: int | None
    generate_max_tokens: int | None


def _parse_start(eval_path: Path) -> EvalMeta | None:
    try:
        with ZipFile(eval_path) as zf:
            start = json.loads(zf.read("_journal/start.json"))
    except (BadZipFile, KeyError, OSError, json.JSONDecodeError):
        return None

    e = start.get("eval", {})
    md = e.get("metadata", {})
    mg = e.get("model_generate_config", {})
    pkgs = e.get("packages", {})
    task_solvers = e.get("task_args", {}).get("solver", [])
    generate_params = {}
    for solver in task_solvers:
        if solver.get("name") == "generate":
            generate_params = solver.get("params", {}) or {}
            break

    owner = "suze" if str(eval_path).startswith(str(REPO_ROOT)) else "christine"
    hint_val = md.get("hint_fraction")
    hint = float(hint_val) if hint_val is not None else None

    return EvalMeta(
        owner=owner,
        path=eval_path,
        created=e.get("created", ""),
        hint=hint,
        task_id=e.get("task_id"),
        model=e.get("model"),
        timeout=md.get("timeout"),
        max_connections=mg.get("max_connections"),
        max_retries=mg.get("max_retries"),
        inspect_version=pkgs.get("inspect_ai"),
        data_path=md.get("data_path"),
        dataset_samples=e.get("dataset", {}).get("samples"),
        epochs=e.get("config", {}).get("epochs"),
        generate_max_tokens=generate_params.get("max_tokens"),
    )


def _list_eval_meta(root: Path, owner: str) -> list[EvalMeta]:
    rows: list[EvalMeta] = []
    for p in sorted(root.glob("*.eval")):
        meta = _parse_start(p)
        if not meta:
            continue
        if meta.model != f"vllm/{MODEL}":
            continue
        meta.owner = owner
        rows.append(meta)
    return rows


def _read_result_json(hint: float) -> dict | None:
    json_path = SUZE_RESULTS_DIR / f"aime_solution_intext_masked_0shot_{hint}.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None


def _select_representative(evals: list[EvalMeta], hint: float) -> EvalMeta | None:
    cands = [e for e in evals if e.hint is not None and abs(e.hint - hint) < 1e-9]
    if not cands:
        return None

    # Prefer timeout matching the shared results json metadata when available.
    preferred_timeout = None
    result_json = _read_result_json(hint)
    if result_json:
        preferred_timeout = result_json.get("metadata", {}).get("timeout")

    if preferred_timeout is not None:
        timeout_matched = [e for e in cands if e.timeout == preferred_timeout]
        if timeout_matched:
            cands = timeout_matched

    # Prefer full-dataset eval files when present (932 AIME problems).
    full = [e for e in cands if e.dataset_samples == 932]
    if full:
        cands = full

    return sorted(cands, key=lambda e: e.created)[-1]


def build_config_profile(
    *,
    evals: list[EvalMeta],
    owner: str,
    hints: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for h in hints:
        sub = [e for e in evals if e.hint is not None and abs(e.hint - h) < 1e-9]
        if not sub:
            continue
        grouped: dict[tuple, int] = {}
        for e in sub:
            key = (
                e.timeout,
                e.max_connections,
                e.max_retries,
                e.dataset_samples,
                e.epochs,
                e.generate_max_tokens,
                "sailhome" if e.data_path and "/sailhome/" in e.data_path else
                "juice5b" if e.data_path and "/juice5b/" in e.data_path else
                "sphinx" if e.data_path and "/sphinx/u/cye/" in e.data_path else
                "other",
            )
            grouped[key] = grouped.get(key, 0) + 1
        for key, count in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True):
            rows.append(
                {
                    "owner": owner,
                    "hint": h,
                    "timeout": key[0],
                    "max_connections": key[1],
                    "max_retries": key[2],
                    "dataset_samples": key[3],
                    "epochs": key[4],
                    "generate_max_tokens": key[5],
                    "data_path_root": key[6],
                    "n_eval_files": count,
                }
            )
    return pd.DataFrame(rows)


def _zip_entries(eval_path: Path, prefix: str) -> list[str]:
    with ZipFile(eval_path) as zf:
        return [n for n in zf.namelist() if n.startswith(prefix)]


def _token_and_accuracy_stats(eval_path: Path) -> dict[str, float | int | None]:
    total_tokens: list[int] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    score_vals: list[str] = []

    with ZipFile(eval_path) as zf:
        summary_files = sorted(
            [n for n in zf.namelist() if n.startswith("_journal/summaries/") and n.endswith(".json")],
            key=lambda n: int(Path(n).stem),
        )
        for sf in summary_files:
            entries = json.loads(zf.read(sf))
            for entry in entries:
                usage = entry.get("model_usage", {}).get(f"vllm/{MODEL}", {})
                if isinstance(usage.get("total_tokens"), int):
                    total_tokens.append(usage["total_tokens"])
                if isinstance(usage.get("input_tokens"), int):
                    input_tokens.append(usage["input_tokens"])
                if isinstance(usage.get("output_tokens"), int):
                    output_tokens.append(usage["output_tokens"])
                val = (
                    entry.get("scores", {})
                    .get("aime_scorer", {})
                    .get("value")
                )
                if isinstance(val, str):
                    score_vals.append(val)

    n = len(score_vals)
    acc = None
    if n > 0:
        correct = sum(1 for v in score_vals if v == "C")
        acc = correct / n

    max_total = max(total_tokens) if total_tokens else None
    sat_rate = None
    if max_total is not None and total_tokens:
        sat_rate = sum(1 for t in total_tokens if t == max_total) / len(total_tokens)

    return {
        "n_samples_scored": n,
        "eval_accuracy": acc,
        "mean_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "max_total_tokens": max_total,
        "total_tokens_saturation_rate": sat_rate,
    }


def _sample_ids(eval_path: Path) -> set[str]:
    ids: set[str] = set()
    with ZipFile(eval_path) as zf:
        for n in zf.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                stem = Path(n).stem
                if "_epoch_" in stem:
                    ids.add(stem.split("_epoch_")[0])
    return ids


def _read_prompt_and_output(eval_path: Path, sample_id: str) -> tuple[str, str] | None:
    with ZipFile(eval_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if n.startswith(f"samples/{sample_id}_epoch_") and n.endswith(".json")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: int(Path(n).stem.split("_epoch_")[-1]))
        d = json.loads(zf.read(candidates[0]))

    prompt = ""
    if d.get("messages"):
        prompt = str(d["messages"][0].get("content", ""))

    output = ""
    choices = (d.get("output") or {}).get("choices") or []
    if choices:
        output = str((choices[0].get("message") or {}).get("content", ""))
    return prompt, output


def _find_task_matches_in_logs(task_id: str | None, log_dir: Path, limit: int = 5) -> list[Path]:
    if not task_id or not log_dir.exists():
        return []
    matches: list[Path] = []
    for p in sorted(log_dir.glob("*_log.out")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        if task_id in txt:
            matches.append(p)
            if len(matches) >= limit:
                return matches
    for p in sorted(log_dir.glob("*_log.err")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        if task_id in txt:
            matches.append(p)
            if len(matches) >= limit:
                return matches
    return matches


def _startup_snippets(log_path: Path) -> list[str]:
    lines = log_path.read_text(errors="ignore").splitlines()
    keep: list[str] = []
    for line in lines:
        s = line.lower()
        if "max_connections" in s or "max_retries" in s or "max_model_len" in s:
            keep.append(line)
        elif "using default chat sampling params" in s:
            keep.append(line)
        elif "generation config" in s and "overridden" in s:
            keep.append(line)
    return keep[:25]


def build_case_study_summary() -> pd.DataFrame:
    suze_evals = _list_eval_meta(SUZE_RESULTS_DIR, owner="suze")
    chr_evals = _list_eval_meta(CHRISTINE_RESULTS_DIR, owner="christine")

    reps: list[EvalMeta] = []
    for h in SUZE_HINTS:
        rep = _select_representative(suze_evals, h)
        if rep:
            reps.append(rep)
    for h in CHRISTINE_HINTS:
        rep = _select_representative(chr_evals, h)
        if rep:
            reps.append(rep)

    rows: list[dict] = []
    for rep in reps:
        stats = _token_and_accuracy_stats(rep.path)
        result_json = _read_result_json(rep.hint if rep.hint is not None else -1)
        json_acc = None
        if result_json is not None:
            json_acc = (
                result_json.get("manual_bootstrap", {})
                .get("accuracy")
            )

        rows.append({
            "owner": rep.owner,
            "hint": rep.hint,
            "created": rep.created,
            "task_id": rep.task_id,
            "eval_file": str(rep.path),
            "timeout": rep.timeout,
            "max_connections": rep.max_connections,
            "max_retries": rep.max_retries,
            "inspect_ai_version": rep.inspect_version,
            "data_path": rep.data_path,
            "dataset_samples": rep.dataset_samples,
            "epochs": rep.epochs,
            "generate_max_tokens": rep.generate_max_tokens,
            "json_accuracy": json_acc,
            **stats,
        })

    df = pd.DataFrame(rows).sort_values(["owner", "hint"]).reset_index(drop=True)
    return df


def write_prompt_comparison(reps_df: pd.DataFrame) -> None:
    suze_row = reps_df[reps_df["owner"] == "suze"].sort_values("hint").head(1)
    chr_row = reps_df[reps_df["owner"] == "christine"].sort_values("hint").head(1)
    if suze_row.empty or chr_row.empty:
        PROMPT_COMPARE_TXT.write_text("Could not find representative evals for both owners.\n")
        return

    suze_eval = Path(suze_row.iloc[0]["eval_file"])
    chr_eval = Path(chr_row.iloc[0]["eval_file"])

    with ZipFile(suze_eval) as zf:
        suze_start = json.loads(zf.read("_journal/start.json"))["eval"]
    with ZipFile(chr_eval) as zf:
        chr_start = json.loads(zf.read("_journal/start.json"))["eval"]
    suze_solver = suze_start.get("task_args", {}).get("solver")
    chr_solver = chr_start.get("task_args", {}).get("solver")

    report = []
    report.append(f"solver_config_equal: {suze_solver == chr_solver}")
    report.append("")
    report.append(f"suze_solver: {suze_solver}")
    report.append(f"christine_solver: {chr_solver}")
    report.append("")

    common = sorted(_sample_ids(suze_eval).intersection(_sample_ids(chr_eval)))
    if not common:
        report.append("No common sample_id found between representative eval files.")
        PROMPT_COMPARE_TXT.write_text("\n".join(report))
        return

    sid = common[0]
    suze_pair = _read_prompt_and_output(suze_eval, sid)
    chr_pair = _read_prompt_and_output(chr_eval, sid)
    if not suze_pair or not chr_pair:
        PROMPT_COMPARE_TXT.write_text(f"Could not extract prompt/output for sample_id={sid}.\n")
        return

    suze_prompt, suze_output = suze_pair
    chr_prompt, chr_output = chr_pair

    report.append(f"sample_id: {sid}")
    report.append(f"prompt_equal: {suze_prompt == chr_prompt}")
    report.append(f"suze_prompt_has_hint_prefix: {'Here is part of a hint' in suze_prompt}")
    report.append(f"christine_prompt_has_hint_prefix: {'Here is part of a hint' in chr_prompt}")
    report.append(f"suze_prompt_has_think_tag: {'<think>' in suze_prompt.lower()}")
    report.append(f"christine_prompt_has_think_tag: {'<think>' in chr_prompt.lower()}")
    report.append(f"suze_output_has_think_tag: {'<think>' in suze_output.lower()}")
    report.append(f"christine_output_has_think_tag: {'<think>' in chr_output.lower()}")
    report.append("")
    report.append("suze_prompt_head:")
    report.append(suze_prompt[:1200])
    report.append("")
    report.append("christine_prompt_head:")
    report.append(chr_prompt[:1200])

    PROMPT_COMPARE_TXT.write_text("\n".join(report))


def print_log_startup_comparison(reps_df: pd.DataFrame) -> None:
    print("\n=== Submitit Log Matches / Startup Snippets ===")
    for _, row in reps_df.iterrows():
        owner = row["owner"]
        hint = row["hint"]
        task_id = row["task_id"]
        log_dir = SUZE_LOG_DIR if owner == "suze" else CHRISTINE_LOG_DIR
        matches = _find_task_matches_in_logs(task_id, log_dir)
        print(f"\nowner={owner} hint={hint} task_id={task_id}")
        if not matches:
            print("  no submitit log match found by task_id")
            continue
        for m in matches:
            print(f"  match: {m}")
            snippets = _startup_snippets(m)
            for s in snippets[:8]:
                print(f"    {s}")


def print_bonus_christine_hint0_vs_1(reps_df: pd.DataFrame) -> None:
    print("\n=== Bonus: Christine hint 0.0 vs 1.0 ===")
    sub = reps_df[(reps_df["owner"] == "christine") & (reps_df["hint"].isin([0.0, 1.0]))]
    if sub.empty or len(sub) < 2:
        print("  Could not find both hint=0.0 and hint=1.0 representative evals.")
        return
    cols = [
        "hint",
        "json_accuracy",
        "eval_accuracy",
        "mean_input_tokens",
        "mean_output_tokens",
        "max_total_tokens",
        "total_tokens_saturation_rate",
        "timeout",
        "max_connections",
    ]
    print(sub[cols].sort_values("hint").to_string(index=False))


def main() -> None:
    suze_evals = _list_eval_meta(SUZE_RESULTS_DIR, owner="suze")
    chr_evals = _list_eval_meta(CHRISTINE_RESULTS_DIR, owner="christine")
    df = build_case_study_summary()
    if df.empty:
        raise RuntimeError("No representative evals found. Check paths and model name.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    profile_df = pd.concat(
        [
            build_config_profile(evals=suze_evals, owner="suze", hints=SUZE_HINTS),
            build_config_profile(evals=chr_evals, owner="christine", hints=CHRISTINE_HINTS),
        ],
        ignore_index=True,
    )
    profile_csv = OUTPUT_DIR / "case_study_qwen3_4b_config_profile.csv"
    profile_df.to_csv(profile_csv, index=False)
    write_prompt_comparison(df)

    print("=== Qwen3-4B Case Study Summary ===")
    print(df[[
        "owner",
        "hint",
        "json_accuracy",
        "eval_accuracy",
        "timeout",
        "max_connections",
        "max_retries",
        "generate_max_tokens",
        "dataset_samples",
        "mean_input_tokens",
        "mean_output_tokens",
        "max_total_tokens",
        "total_tokens_saturation_rate",
        "task_id",
    ]].to_string(index=False))

    print_log_startup_comparison(df)
    print("\n=== Config Profile (all eval files for target owner/hints) ===")
    print(profile_df.to_string(index=False))
    print_bonus_christine_hint0_vs_1(df)

    print(f"\nWrote: {SUMMARY_CSV}")
    print(f"Wrote: {profile_csv}")
    print(f"Wrote: {PROMPT_COMPARE_TXT}")


if __name__ == "__main__":
    main()
