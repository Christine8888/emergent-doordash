import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile, ZipFile
from inspect_ai.log import read_eval_log


# goal: compare qwen3-4b on aime hint level 0 vs 1




RESULTS_ROOT = "christine_experiments/20251113/results/aime/solution_intext_masked/0shot"
CHRISTINE_EVAL_FILES_QWEN_4B = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-4B"

def to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def print_sample(samples, id, epoch):
    matches = [s for s in samples if s.id == "2012-I-5" and s.epoch == 8]
    s = matches[0]
    # print(s.messages[0].content)                  # prompt
    print(s.output.choices[0].message.content)    # output


def get_hint_fraction(eval_path: Path):
    try:
        with ZipFile(eval_path) as zf:
            start = json.loads(zf.read("_journal/start.json"))
    except (BadZipFile, KeyError, json.JSONDecodeError, OSError):
        return None
    return start.get("eval", {}).get("metadata", {}).get("hint_fraction")

def get_important_hint_fractions():
    # create mapping of eval files to json output file hint levels

    for file in Path(CHRISTINE_EVAL_FILES_QWEN_4B).iterdir():
        if file.suffix == ".eval":
            # check what hint level this eval file belong to
            hint = get_hint_fraction(file)
            print(f"{file.name} -> hint_fraction: {hint}")


def main():
    # python suze_experiments/20260303_finding_defaults/looking_at_results.py

    # get_important_hint_fractions()
    # 2026-01-08T16-11-30-08-00_aime_Ad5gTRHYxdnzR9a7hTjG5p.eval -> hint_fraction: 1.0
    # 2026-01-19T13-33-07-08-00_aime_33SoagUcEG3BPYTWNdpF6E.eval -> hint_fraction: 0.0
    # 2026-01-08T04-13-53-08-00_aime_VGu9zJ75gNFCVhubrpK7VK.eval -> hint_fraction: 0.0

    # now look at these eval files; are 0.0 and 1.0 from the same run?
    for eval_file in ['2026-01-19T13-33-07-08-00_aime_33SoagUcEG3BPYTWNdpF6E.eval', '2026-01-08T04-13-53-08-00_aime_VGu9zJ75gNFCVhubrpK7VK.eval']:
        file_path = CHRISTINE_EVAL_FILES_QWEN_4B + '/' + eval_file
        log = read_eval_log(str(file_path))
        print(f'evaluating {eval_file}')
        print(f'finish status: {log.status}')
        print(f'num samples evaluated: {len(log.samples)}')
        print(f'eval spec: {log.eval}')

    # why only 220 samples evaluated?
            






if __name__ == "__main__":
    # python suze_experiments/20260303_finding_defaults/compare_outliers.py
    main()
