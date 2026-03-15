# Consolidated Eval Viewer

Local Streamlit app backed by DuckDB for browsing consolidated JSONL rollouts.

## Features

- Filters by `benchmark`, `run_type` (`baseline/results`), `path_hint_level`, `model`, `correctness` (`C/I/U`), and scorer.
- Shows grouped problem rows (`sample_id`) with epoch counts and score counts.
- Lets you select one problem and inspect all epochs (prompt/output/extracted answer/explanation).
- Incremental sync from consolidated JSONL files into a local DuckDB cache.
- Handles changed JSONL files by replacing all rows from that source file in the DB on sync.
- Sync progress bar with per-file status.
- Default fast ingest mode stores metadata/scoring in DB and lazily loads full prompt/output text for selected rows.

## Launch

From repo root:

```bash
pip install duckdb streamlit pandas
streamlit run suze_experiments/20260313/view_consolidated_eval.py
```

If you use conda:

```bash
conda run -n ed pip install duckdb streamlit pandas
conda run -n ed streamlit run suze_experiments/20260313/view_consolidated_eval.py
```

## Usage

1. Open the app in your browser.
2. Confirm the data directory points to `suze_experiments/20260313/consolidated_jsonl` (or your target output dir).
3. Keep **Store full prompt/output in DB (slow, large)** unchecked for faster initial sync.
4. Click **Sync JSONL -> DuckDB**.
5. Apply filters and pick a problem to inspect all epochs.

## Notes

- First sync can take time for large corpora.
- DB cache path defaults to `consolidated_jsonl/_viewer_cache.duckdb`.
- Use **Force Rebuild DB** if you need a full re-import.

## Large-Scale Cache Build (Recommended)

For very large files (e.g. `results__aime.jsonl`), build cache outside Streamlit:

```bash
python suze_experiments/20260313/build_viewer_cache.py \
  --data-dir suze_experiments/20260313/consolidated_jsonl \
  --db-path suze_experiments/20260313/consolidated_jsonl/_viewer_cache.duckdb \
  --files results__aime.jsonl \
  --rebuild \
  --batch-size 20000
```

Notes:

- This script logs progress every few seconds with percent/throughput/rows.
- By default, it **does not** store full prompt/output text or scorer explanations (faster).
- Enable full text only if needed:

```bash
python suze_experiments/20260313/build_viewer_cache.py --files results__aime.jsonl --include-full-text --include-explanations
```

After cache build finishes, open the Streamlit app and query the DB normally.

## Re-Scoring via Sidecar (Safe)

To add a new scorer without mutating `results__aime.jsonl`, write a scorer-only sidecar:

```bash
conda run -n ed python suze_experiments/20260313/rescore_aime_extract_answer_fixed.py
```

Default outputs:
- input: `consolidated_jsonl/results__aime.jsonl` (read-only)
- sidecar: `consolidated_jsonl/results__aime.extract_answer_fixed.scorers.jsonl`
- checkpoint: `consolidated_jsonl/_state/results__aime.extract_answer_fixed.state.json`

Resume behavior:
- Re-run the same command to resume from the checkpoint.
- Use `--restart` to start from byte offset 0 and overwrite sidecar/state.

Ingest into viewer DB:

```bash
python suze_experiments/20260313/build_viewer_cache.py \
  --files results__aime.extract_answer_fixed.scorers.jsonl
```

Or from Streamlit, click **Sync JSONL -> DuckDB** (no rebuild needed).
