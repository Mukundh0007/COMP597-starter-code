# BERT E1 Results

This directory stores automated **E1** experiment artifacts (end-to-end time baseline with `trainer_stats=noop`).

## Protocol implemented

- Duration target: 5 minutes per run (enforced by BERT trainer time gate).
- Batch sizes: 32, 16, 8 by default.
- Repeat values (fixed): 550, 1100, 2000 respectively.
- Repetitions: 3 runs per batch size.
- Cooldown: 60 seconds between consecutive runs.
- Measurement mode: `noop` trainer stats (baseline, minimal overhead).

## How to run

From repository root:

```bash
bash scripts/run-bert-e1.sh
```

Optional environment overrides:

```bash
E1_RUNS_PER_BATCH=3 E1_COOLDOWN_S=60 bash scripts/run-bert-e1.sh
```

## Output files

- `e1_time_summary.csv`
  - Columns: `batch_size,repeat,run1_s,run2_s,run3_s,avg_s,std_s`
- `e1_runs.csv`
  - One line per run with timestamps, train/wall times, and artifact paths.
- `results_batch{BATCH}_run{RUN}.csv`
  - Per-run metadata for E1 (no per-step instrumentation by design).
- `logs/batch{BATCH}_run{RUN}.log`
  - Full command output for reproducibility.

## Notes

- Repeat is fixed to the provided combinations:
  - batch size 32 -> repeat 550
  - batch size 16 -> repeat 1100
  - batch size 8 -> repeat 2000
- With `--data_configs.bert.n 0`, the data loader sets `n=batch_size`, so total step count is approximately controlled by `repeat`.
