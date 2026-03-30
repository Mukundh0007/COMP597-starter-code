# BERT E2 Results

This directory stores automated **E2** artifacts (end-to-end energy measurements with `trainer_stats=codecarbon`).

## Protocol implemented

- Duration target: approximately 5 minutes per run (BERT trainer has a 5-minute stop gate).
- Batch/repeat pairs:
  - 32 -> 550
  - 16 -> 1100
  - 8 -> 2000
- Repetitions: 3 runs per batch size.
- Cooldown: 60 seconds between consecutive runs.
- Overhead check: compares E2 timing against E1 averages from [results/E1/e1_time_summary.csv](results/E1/e1_time_summary.csv).

## How to run

From repository root:

```bash
bash scripts/run-bert-e2.sh
```

Optional overrides:

```bash
E2_RUNS_PER_BATCH=3 E2_COOLDOWN_S=60 bash scripts/run-bert-e2.sh
```

## Output files

- `e2_runs.csv`
  - One row per run with measured times, overhead vs E1, pass/fail against 5% threshold, and artifact paths.
- `e2_time_overhead_summary.csv`
  - Batch-level summary including average runtime and average overhead vs E1.
- `batch{BATCH}_run{RUN}/run_metadata.csv`
  - Per-run metadata and computed overhead fields.
- `batch{BATCH}_run{RUN}/codecarbon/*`
  - CodeCarbon CSV outputs (full run, step/substep task files, and losses folder).
- `logs/batch{BATCH}_run{RUN}.log`
  - Full run logs for reproducibility and parsing.

## Notes

- E2 intentionally uses CodeCarbon only (no extra custom per-step timeline plotting).
- This folder is intended to avoid copying artifacts from a separate location: outputs are written directly under `results/E2`.
