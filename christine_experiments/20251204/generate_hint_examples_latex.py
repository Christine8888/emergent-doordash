"""Generate LaTeX figures showing masking and truncation examples from real hint data."""

import json
import re
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "christine_experiments" / "data"
SAVE_DIR = _PROJECT_ROOT / "plots"
SAVE_DIR.mkdir(exist_ok=True)

HINT_FRACTION = 0.5
MAX_DISPLAY_CHARS = 450


def _split_preserving_whitespace(text):
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def truncate_sequential(text, fraction):
    tokens, word_indices = _split_preserving_whitespace(text)
    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text
    last_word_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_word_idx + 1]).strip()


def mask_text(text, fraction, seed=42):
    tokens, word_indices = _split_preserving_whitespace(text)
    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text
    rng = random.Random(seed)
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join("[MASK]" if i in mask_indices else t
                   for i, t in enumerate(tokens)).strip()


def pick_example(path, max_chars=MAX_DISPLAY_CHARS):
    """Pick a hint that's reasonably short for display."""
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    candidates = [(d, len(d['hint'])) for d in lines if d.get('hint', '')]
    candidates.sort(key=lambda x: x[1])
    idx = len(candidates) // 8
    d = candidates[idx][0]
    hint = d['hint']
    if len(hint) > max_chars:
        cut = hint[:max_chars].rfind(' ')
        hint = hint[:cut if cut > 0 else max_chars] + ' [...]'
    return hint


def make_box(label, text, color):
    return rf"""\begin{{tcolorbox}}[
    colback=gray!3, colframe={color}, boxrule=0.5pt,
    title={{\scriptsize\textbf{{{label}}}}},
    fonttitle=\sffamily, coltitle=white, colbacktitle={color},
    left=3pt, right=3pt, top=1pt, bottom=1pt
]
\begin{{lstlisting}}
{text}
\end{{lstlisting}}
\end{{tcolorbox}}"""


def generate_latex(source_type):
    gpqa_hint = pick_example(DATA_DIR / source_type / "gpqa.jsonl")
    aime_hint = pick_example(DATA_DIR / source_type / "aime.jsonl")

    frac_pct = int(HINT_FRACTION * 100)
    rows = []
    for bench_name, hint in [("GPQA", gpqa_hint), ("AIME", aime_hint)]:
        for method_name, apply_fn in [("masking", lambda h: mask_text(h, HINT_FRACTION)),
                                       ("truncation", lambda h: truncate_sequential(h, HINT_FRACTION))]:
            applied = apply_fn(hint)
            left = make_box("full hint", hint, color="blue!60!black")
            right = make_box(f"{method_name} applied", applied, color="red!60!black")
            row = rf"""\textbf{{\sffamily\small {bench_name} --- {method_name} ($h = {HINT_FRACTION}$)}} \\[2pt]
{left} & {right} \\[6pt]"""
            rows.append(row)

    source_label = "chain-of-thought" if source_type == "cot" else "solution"
    body = "\n".join(rows)

    latex = rf"""\begin{{figure}}[p]
\caption{{Examples of hint masking and truncation applied to \textbf{{{source_label}}} reasoning traces. Left columns show the full hint; right columns show the result after applying each method at $h = {HINT_FRACTION}$. Masking replaces randomly selected words with \texttt{{[MASK]}}; truncation retains only the first $h \times 100\%$ of words.}}
\label{{fig:{source_type}-hint-examples}}
\vspace{{4pt}}
\begin{{tabular}}{{@{{}}p{{0.48\textwidth}}p{{0.48\textwidth}}@{{}}}}
{body}
\end{{tabular}}
\end{{figure}}"""

    return latex


if __name__ == "__main__":
    for source_type in ["cot", "solution"]:
        latex = generate_latex(source_type)
        out_path = SAVE_DIR / f"hint_examples_{source_type}.tex"
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"wrote {out_path}")
