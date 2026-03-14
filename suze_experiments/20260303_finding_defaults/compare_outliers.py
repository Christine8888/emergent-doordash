import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile, ZipFile
from inspect_ai.log import read_eval_log


# goal: compare qwen3-14b on aime hint level 0 vs 1




RESULTS_ROOT = "christine_experiments/20251113/results/aime/solution_intext_masked/0shot"
CHRISTINE_EVAL_FILES_QWEN_14B = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-14B"
CACHE_PATH = Path(__file__).with_name("compare_outliers_cache.json")
EVAL_FILES_BY_HINT = {
    "0.0": "2026-01-12T15-02-22-08-00_aime_Yy2oCxE4DZCdXtc7hdAzuT.eval", # 0.0
    # "0.05":"2026-01-13T19-04-00-08-00_aime_hN2ThxL6jcWU5qTASexdXj.eval", # 0.05
    # "1.0": "2026-01-13T16-40-11-08-00_aime_5aV7cnYwQyXhpashnHxQKr.eval", # 1.0
    "0.9": "2026-01-13T15-14-13-08-00_aime_Pxg9HHfPUj7F7PUC7GWbwr.eval",
}

def to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", item)))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def sample_to_record(sample):
    prompt = None
    if getattr(sample, "messages", None):
        prompt = as_text(sample.messages[0].content)

    output = None
    if getattr(sample, "output", None) and getattr(sample.output, "choices", None):
        output = as_text(sample.output.choices[0].message.content)

    scores = getattr(sample, "scores", None)
    if scores is not None:
        if isinstance(scores, dict):
            scores = {k: to_dict(v) for k, v in scores.items()}
        else:
            scores = to_dict(scores)

    return {
        "id": getattr(sample, "id", None),
        "epoch": getattr(sample, "epoch", None),
        "target": getattr(sample, "target", None),
        "sample_idx": getattr(sample, "sample_idx", None),
        "prompt": prompt,
        "output": output,
        "scores": scores,
    }


def print_sample(record):
    print(f"epoch: {record.get('epoch')}")
    print(f"target: {record.get('target')}")
    print("scores:")
    print(json.dumps(record.get("scores"), indent=2, sort_keys=True))
    print("\nprompt:")
    print(record.get("prompt"))
    print("\noutput:")
    print(record.get("output"))


def is_correct(record):
    scores = record.get("scores")

    # Typical AIME scorer shape:
    # {"aime_scorer": {"value": "C" or "I", ...}}
    aime = scores.get("aime_scorer")
    value = aime.get("value")
    if value == "C":
        return True
    if value == "I":
        return False

    return ValueError('no score found')


def build_sample_index(samples):
    # A sample id can appear multiple times (e.g. across epochs),
    # so keep a list of records for each id.
    index = defaultdict(list)
    for sample in samples:
        record = sample_to_record(sample)
        index[record["id"]].append(record)
    for sample_id in index:
        # Deterministic order: first epoch first.
        index[sample_id].sort(
            key=lambda r: (
                r["epoch"],
                r["sample_idx"],
            )
        )
    return index


def build_cache(eval_files):
    cache = {"evals": {}}

    for eval_file in eval_files:
        file_path = Path(CHRISTINE_EVAL_FILES_QWEN_14B) / eval_file
        t0 = perf_counter()
        log = read_eval_log(str(file_path))
        elapsed = perf_counter() - t0

        sample_index = build_sample_index(log.samples)
        cache["evals"][eval_file] = {
            "path": str(file_path),
            "mtime_ns": file_path.stat().st_mtime_ns,
            "status": log.status,
            "num_samples": len(log.samples),
            "index": sample_index,
        }
        print(f"read {eval_file} in {elapsed:.2f}s")

    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    print(f"wrote cache: {CACHE_PATH}")
    return cache


def load_or_build_cache(eval_files):
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        cache_evals = cache.get("evals", {})
        if all(
            eval_file in cache_evals
            and cache_evals[eval_file].get("mtime_ns")
            == (Path(CHRISTINE_EVAL_FILES_QWEN_14B) / eval_file).stat().st_mtime_ns
            for eval_file in eval_files
        ):
            print(f"using cache: {CACHE_PATH}")
            return cache


    return build_cache(eval_files)


def get_hint_fraction(eval_path: Path):
    try:
        with ZipFile(eval_path) as zf:
            start = json.loads(zf.read("_journal/start.json"))
    except (BadZipFile, KeyError, json.JSONDecodeError, OSError):
        return None
    return start.get("eval", {}).get("metadata", {}).get("hint_fraction")

def get_important_hint_fractions():
    # create mapping of eval files to json output file hint levels

    for file in Path(CHRISTINE_EVAL_FILES_QWEN_14B).iterdir():
        if file.suffix == ".eval":
            # check what hint level this eval file belong to
            hint = get_hint_fraction(file)
            print(f"{file.name} -> hint_fraction: {hint}")


def compare_selected_sample_ids(cache, eval_files, sample_ids):
    # Old/manual behavior preserved: inspect chosen sample ids.
    for sample_id in sample_ids:
        print(f"\n=== sample_id: {sample_id} ===")
        for eval_file in eval_files:
            matches = cache["evals"][eval_file]["index"].get(sample_id, [])
            if not matches:
                print(f"[{eval_file}] not found")
                continue

            first_epoch_sample = matches[0]
            print(
                f"[{eval_file}] found {len(matches)} record(s); "
                f"showing first epoch {first_epoch_sample.get('epoch')}"
            )
            print_sample(first_epoch_sample)


def find_0_right_1_wrong_ids(cache, eval_file_0, eval_file_1):
    idx0 = cache["evals"][eval_file_0]["index"]
    idx1 = cache["evals"][eval_file_1]["index"]

    shared_ids = sorted(set(idx0.keys()).intersection(idx1.keys()))
    out = []
    for sample_id in shared_ids:
        r0 = idx0[sample_id][0]  # first epoch
        r1 = idx1[sample_id][0]  # first epoch
        c0 = is_correct(r0)
        c1 = is_correct(r1)
        if c0 and not c1:
            out.append(sample_id)
    return out


def find_1_right_0_wrong_ids(cache, eval_file_0, eval_file_1):
    idx0 = cache["evals"][eval_file_0]["index"]
    idx1 = cache["evals"][eval_file_1]["index"]

    shared_ids = sorted(set(idx0.keys()).intersection(idx1.keys()))
    out = []
    for sample_id in shared_ids:
        r0 = idx0[sample_id][0]  # first epoch
        r1 = idx1[sample_id][0]  # first epoch
        c0 = is_correct(r0)
        c1 = is_correct(r1)
        if c1 and not c0:
            out.append(sample_id)
    return out


def print_0_right_1_wrong_examples(cache, eval_file_0, eval_file_1, limit=2):
    ids = find_0_right_1_wrong_ids(cache, eval_file_0, eval_file_1)
    print(
        f"\nFound {len(ids)} sample_ids where hint=0 is correct and hint=1 is wrong "
        f"(comparing first epoch)."
    )
    if not ids:
        return

    show_ids = ids[:limit]
    print(f"Showing first {len(show_ids)} sample(s): {show_ids}")
    for sample_id in show_ids:
        print(f"\n=== sample_id: {sample_id} ===")
        rec0 = cache["evals"][eval_file_0]["index"][sample_id][0]
        rec1 = cache["evals"][eval_file_1]["index"][sample_id][0]

        print(f"\n[{eval_file_0}] hint=0 | correct={is_correct(rec0)}")
        print_sample(rec0)
        print(f"\n[{eval_file_1}] hint=1 | correct={is_correct(rec1)}")
        print_sample(rec1)


def print_1_right_0_wrong_examples(cache, eval_file_0, eval_file_1, limit=2):
    ids = find_1_right_0_wrong_ids(cache, eval_file_0, eval_file_1)
    print(
        f"\nFound {len(ids)} sample_ids where hint=1 is correct and hint=0 is wrong "
        f"(comparing first epoch)."
    )
    if not ids:
        return

    show_ids = ids[:limit]
    print(f"Showing first {len(show_ids)} sample(s): {show_ids}")
    for sample_id in show_ids:
        print(f"\n=== sample_id: {sample_id} ===")
        rec0 = cache["evals"][eval_file_0]["index"][sample_id][0]
        rec1 = cache["evals"][eval_file_1]["index"][sample_id][0]

        print(f"\n[{eval_file_0}] hint=0 | correct={is_correct(rec0)}")
        print_sample(rec0)
        print(f"\n[{eval_file_1}] hint=1 | correct={is_correct(rec1)}")
        print_sample(rec1)

def look_at_chosen_examples():
    eval_files = list(EVAL_FILES_BY_HINT.values())
    # sample_ids = ['1988-14', '2006-I-14', '2009-II-9', '2012-I-4', '2012-II-6', '2007-I-15', '2012-I-5', '1999-6', '2024-II-7', '1999-8', '1998-14', '2022-I-4', '2017-II-2', '1996-2', '1984-6', '2007-II-13', '2001-II-2', '2011-II-4']
    sample_ids = ['2024-II-7']
    cache = load_or_build_cache(eval_files)
    
    # now look at these eval files; are 0.0 and 1.0 from the same run?
    for hint, eval_file in EVAL_FILES_BY_HINT.items():
        eval_data = cache["evals"][eval_file]
        print(f"evaluating hint={hint} ({eval_file})")
        print(f"finish status: {eval_data['status']}")
        print(f"num samples evaluated: {eval_data['num_samples']}")
        # print(f'eval spec: {log.eval}')
        # both have status success and all samples evaluated, indexed by sample id? sample_ids= sample_ids=['1996-12', '2022-I-8', '2003-II-6', '2005-II-12', ...]
    
    
    # Old/manual inspection path (kept for debugging):
    compare_selected_sample_ids(cache, eval_files, sample_ids)

def main():
    # python suze_experiments/20260303_finding_defaults/compare_outliers.py

    # get_important_hint_fractions()
    # 2026-01-12T15-02-22-08-00_aime_Yy2oCxE4DZCdXtc7hdAzuT.eval -> hint_fraction: 0.0
    # 2026-01-13T19-04-00-08-00_aime_hN2ThxL6jcWU5qTASexdXj.eval -> hint_fraction: 0.05
    # 2026-01-13T16-40-11-08-00_aime_5aV7cnYwQyXhpashnHxQKr.eval -> hint_fraction: 1.0
    # 2026-01-13T15-14-13-08-00_aime_Pxg9HHfPUj7F7PUC7GWbwr.eval -> hint_fraction: 0.9
    # raise ValueError('a')

    look_at_chosen_examples()
    raise ValueError('a')
    
    
    # New helper: find cases where hint=0 beats hint=1.
    cache = load_or_build_cache(list(EVAL_FILES_BY_HINT.values()))
    print_0_right_1_wrong_examples(
        cache,
        eval_file_0=EVAL_FILES_BY_HINT["0.0"],
        eval_file_1=EVAL_FILES_BY_HINT["1.0"],
        limit=1,
    )
    # print_1_right_0_wrong_examples(
    #     cache,
    #     eval_file_0=EVAL_FILES_BY_HINT["0.0"],
    #     eval_file_1=EVAL_FILES_BY_HINT["1.0"],
    #     limit=1,
    # )
        
if __name__ == "__main__":
    # python suze_experiments/20260303_finding_defaults/compare_outliers.py
    main()
