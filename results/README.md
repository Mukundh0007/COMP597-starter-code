# Results Folder Guide

This folder is organized by synchronization frequency and experiment variant.

## Experiment groups

- **Sync every 5 steps**
  - `E1/`
  - `E2/`
  - `E3/`

- **Every step**
  - `E1_new/`
  - `E2_new/`
  - `E3_1step/`

- **Every 3 steps**  
  - `E1_step3/`
  - `E2_step3/`
  - `E3_step3/`

## Main CSV files used by plotting scripts

- **E1 plots**
  - `E1/e1_time_summary.csv`
  - Used by: `E1/plot_e1_results.py`

- **E2 plots**
  - `E2/e2_runs.csv` and `E2_new/e2_runs.csv`
  - Per-run CodeCarbon files under each batch folder, for example:
    - `E2/batch16_run1/codecarbon/run_1_cc_full_rank_0.csv`
    - `E2_new/batch32_run1/codecarbon/run_1_cc_full_rank_0.csv`
  - Used by: `E2/plot_e2_analysis.py` and `E2_new/plot_e2_analysis.py`

- **E3 plots**
  - `E3/e3_runs.csv`
  - `E3/batch{bs}/timeline_summary.csv`
  - `E3_1step/batch{bs}/timeline_summary.csv`
  - Used by: `E3/plot_e3_timelines.py`, `E3/plot_memory_comparison.py`, `E3/plot_e3_utilization.py`, and the matching scripts in `E3_1step/`

## Notes

- The `batch{bs}` folders contain the run-level metrics used for the E3 timeline and memory/utilization plots.
- The CodeCarbon CSV files contain the raw energy tracking outputs used in E2 for the power and energy analysis.
- If you are attaching only the main CSVs, start with the summary files above and include the per-run CodeCarbon CSVs for E2 when energy/power figures are needed.
