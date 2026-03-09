import json
from pathlib import Path

from inspect_ai.log import read_eval_log


# file that errored: 14767776_0 (Llama-3.1-8B-Instruct | hint=0.10)
# want to check; what were the really long samples?

# [03/09 11:42:23] WARNING: estimated input ~28073 tokens (86% of max_model_len=32768); only ~4695 tokens left for output sample_id='2011-II-3' epoch=10
# [03/09 11:42:23] WARNING: estimated input ~32123 tokens (98% of max_model_len=32768); only ~645 tokens left for output sample_id='2010-I-4' epoch=10

# christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-8B-Instruct/2026-03-09T16-53-41+00-00_aime_GzpuYpZw6bmxazYANAuHsH.eval  

EVAL_FILE = "christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-8B-Instruct/2026-03-09T16-53-41+00-00_aime_GzpuYpZw6bmxazYANAuHsH.eval"


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



def main():
    # python suze_experiments/20260303_finding_defaults/looking_at_results.py
    path = Path(EVAL_FILE)
    if not path.exists():
        raise FileNotFoundError(path)

    log = read_eval_log(str(path))
    samples = to_dict(log.samples)
    

    print_sample(samples=samples, id='2011-II-3', epoch=10)

if __name__ == "__main__":
    main()
