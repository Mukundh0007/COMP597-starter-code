#!/usr/bin/env python3
"""
Create a 3x3 matrix of stacked area plots for E3 phase timing.

Matrix layout:
- Rows: batch sizes 8, 16, 32
- Cols: run 1..3

Each subplot reads:
results/E3/batch{bs}_run{run}/metrics/bert_run_results.csv

and plots stacked areas of:
- step_s
- forward_s
- backward_s
- optimizer_s

Output:
results/E3/plots/e3_phase_stacked_matrix.png
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_phase_csv(path: Path):
    steps = []
    step_s = []
    forward_s = []
    backward_s = []
    optimizer_s = []

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(_to_float(row.get("step", "0")))
            step_s.append(max(0.0, _to_float(row.get("step_s", "0"))))
            forward_s.append(max(0.0, _to_float(row.get("forward_s", "0"))))
            backward_s.append(max(0.0, _to_float(row.get("backward_s", "0"))))
            optimizer_s.append(max(0.0, _to_float(row.get("optimizer_s", "0"))))

    return steps, step_s, forward_s, backward_s, optimizer_s


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    runs = [1, 2, 3]

    # Requested pastel palette
    colors = {
        "step_s": "#f4a582",      # pastel coral orange
        "forward_s": "#9ecae1",   # pastel light blue
        "backward_s": "#bcbddc",  # pastel violet
        "optimizer_s": "#fbb4c4", # pastel pink
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=False)

    for r_i, bs in enumerate(batch_sizes):
        for c_i, run in enumerate(runs):
            ax = axes[r_i][c_i]
            csv_path = base_dir / f"batch{bs}_run{run}" / "metrics" / "bert_run_results.csv"

            if not csv_path.exists():
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            steps, step_s, forward_s, backward_s, optimizer_s = load_phase_csv(csv_path)

            if len(steps) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # Convert to milliseconds for clearer visual interpretation.
            step_ms = [v * 1000.0 for v in step_s]
            forward_ms = [v * 1000.0 for v in forward_s]
            backward_ms = [v * 1000.0 for v in backward_s]
            optimizer_ms = [v * 1000.0 for v in optimizer_s]

            # Stack only phase components. step_s is total step time, so it is overlaid as a line.
            ax.stackplot(
                steps,
                forward_ms,
                backward_ms,
                optimizer_ms,
                colors=[
                    colors["forward_s"],
                    colors["backward_s"],
                    colors["optimizer_s"],
                ],
                alpha=0.85,
                linewidth=0.0,
            )
            ax.plot(steps, step_ms, color=colors["step_s"], linewidth=1.8, label="step_s (total)")

            if r_i == 0:
                ax.set_title(f"Run {run}", fontweight="bold", fontsize=12)
            ax.grid(alpha=0.22, linestyle=":")
            if r_i == len(batch_sizes) - 1:
                ax.set_xlabel("Step", fontweight="bold")
            else:
                ax.set_xlabel("")

            if c_i == 0:
                ax.set_ylabel(f"Batch Size {bs}\nTime (ms)", fontweight="bold")
            else:
                ax.set_ylabel("")

            stack_top = [f + b + o for f, b, o in zip(forward_ms, backward_ms, optimizer_ms)]
            ymax = max(max(step_ms), max(stack_top)) if step_ms and stack_top else 1.0
            ax.set_ylim(0.0, ymax * 1.15)

    legend_handles = [
        plt.Line2D([0], [0], color=colors["step_s"], lw=2),
        plt.Rectangle((0, 0), 1, 1, color=colors["forward_s"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=colors["backward_s"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=colors["optimizer_s"], alpha=0.85),
    ]
    fig.legend(
        legend_handles,
        ["step_s", "forward_s", "backward_s", "optimizer_s"],
        loc="upper center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle("Phase-Wise Stacked Area Matrix (Batch Sizes x Runs)", fontsize=16, fontweight="bold", y=1.04)
    fig.subplots_adjust(wspace=0.36, hspace=0.36)

    out_png = plots_dir / "e3_phase_stacked_matrix.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
