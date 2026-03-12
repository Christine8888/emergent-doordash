from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from corpus.ingest import IngestConfig, ingest_eval_corpus


# --------------------------
# Settings
# --------------------------
EVAL_ROOTS = {
    # Suze results (local checkout)
    "suzeva": PROJECT_ROOT / "christine_experiments/20251113",
    # Christine results (external path)
    "christine": Path("/sphinx/u/cye/emergent-doordash/christine_experiments/20251113"),
}

OUTPUT_DIR = Path(__file__).resolve().parent / "corpus"
OVERWRITE_OUTPUTS = False
MAX_EVAL_FILES: int | None = None  # Set small int for quick testing.
PROGRESS_EVERY_EVALS = 1000


def main() -> None:
    config = IngestConfig(
        project_root=PROJECT_ROOT,
        eval_roots=EVAL_ROOTS,
        output_dir=OUTPUT_DIR,
        overwrite_outputs=OVERWRITE_OUTPUTS,
        max_eval_files=MAX_EVAL_FILES,
        progress_every_evals=PROGRESS_EVERY_EVALS,
    )
    summary = ingest_eval_corpus(config)
    print("\nIngest summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # python suze_experiments/20260312/build_corpus_from_evals.py
    main()
