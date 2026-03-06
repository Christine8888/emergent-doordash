import json
import re
import numpy as np

data_path = "/Users/christineye/emergent-doordash/christine_experiments/data/solution/aime.jsonl"

entries = []
with open(data_path) as f:
    for line in f:
        entries.append(json.loads(line.strip()))

total = len(entries)
positions_frac_total = []   # first occurrence position as fraction of total hint length
positions_frac_pre = []     # first occurrence position as fraction of pre-ANSWER text length

for entry in entries:
    hint = entry["hint"]
    target = str(entry["target"])

    # Split at the LAST occurrence of "ANSWER:"
    last_idx = hint.rfind("ANSWER:")
    if last_idx == -1:
        # No "ANSWER:" found, skip
        continue

    pre_answer_text = hint[:last_idx]
    total_hint_len = len(hint)
    pre_answer_len = len(pre_answer_text)

    # Check if target appears as a standalone number (word boundary match) in pre-answer text
    pattern = r'\b' + re.escape(target) + r'\b'
    match = re.search(pattern, pre_answer_text)

    if match:
        first_pos = match.start()
        positions_frac_total.append(first_pos / total_hint_len)
        positions_frac_pre.append(first_pos / pre_answer_len)

n_found = len(positions_frac_total)
positions_frac_total = np.array(positions_frac_total)
positions_frac_pre = np.array(positions_frac_pre)

print(f"Total entries: {total}")
print(f"Entries with target appearing before ANSWER:: {n_found} ({n_found/total*100:.1f}%)")
print()

if n_found > 0:
    print("=== First-appearance position as fraction of TOTAL hint length ===")
    print(f"  Median: {np.median(positions_frac_total):.4f}")
    print(f"  p25:    {np.percentile(positions_frac_total, 25):.4f}")
    print(f"  p75:    {np.percentile(positions_frac_total, 75):.4f}")
    print(f"  p10:    {np.percentile(positions_frac_total, 10):.4f}")
    print(f"  p90:    {np.percentile(positions_frac_total, 90):.4f}")
    print()
    print("=== First-appearance position as fraction of PRE-ANSWER text length ===")
    print(f"  Median: {np.median(positions_frac_pre):.4f}")
    print(f"  p25:    {np.percentile(positions_frac_pre, 25):.4f}")
    print(f"  p75:    {np.percentile(positions_frac_pre, 75):.4f}")
    print(f"  p10:    {np.percentile(positions_frac_pre, 10):.4f}")
    print(f"  p90:    {np.percentile(positions_frac_pre, 90):.4f}")
