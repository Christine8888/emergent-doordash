from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from corpus.regrade import RegradeConfig, regrade_corpus


# --------------------------
# Settings
# --------------------------
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# Add names from src/corpus/regraders/registry.py.
ENABLED_REGRADERS = [
    # "math_extract_fixed_v1",
]

OVERWRITE_REGRADED = True
PROGRESS_EVERY_ROLLOUTS = 100
BENCHMARK_ALLOWLIST: list[str] | None = None  # e.g. ["aime"]


def main() -> None:
    config = RegradeConfig(
        corpus_dir=CORPUS_DIR,
        enabled_regraders=ENABLED_REGRADERS,
        overwrite_regraded=OVERWRITE_REGRADED,
        progress_every_rollouts=PROGRESS_EVERY_ROLLOUTS,
        benchmark_allowlist=BENCHMARK_ALLOWLIST,
    )
    summary = regrade_corpus(config)
    print("\nRegrade summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # python suze_experiments/20260312/run_regraders.py
    main()
