#!/usr/bin/env python3
"""
Plot E3 CPU/GPU utilization mean ± std versus time for batch sizes 8, 16, 32.

Input files:
  results/E3/batch8/timeline_summary.csv
  results/E3/batch16/timeline_summary.csv
  results/E3/batch32/timeline_summary.csv

Output:
  results/E3/plots/e3_cpu_gpu_utilization_mean_std.png
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def load_timeline(path: Path):
    t = []
    gpu_mean = []
    gpu_std = []
    cpu_mean = []
    cpu_std = []

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(_to_float(row.get("timeline_s", "0")))
            gpu_mean.append(_to_float(row.get("gpu_util_pct_mean", "0")))
            gpu_std.append(_to_float(row.get("gpu_util_pct_std", "0")))
            cpu_mean.append(_to_float(row.get("cpu_util_pct_mean", "0")))
            cpu_std.append(_to_float(row.get("cpu_util_pct_std", "0")))

    return t, gpu_mean, gpu_std, cpu_mean, cpu_std


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    timelines = {}

    for bs in batch_sizes:
        csv_path = base_dir / f"batch{bs}" / "timeline_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing timeline summary: {csv_path}")
        timelines[bs] = load_timeline(csv_path)

    # Pastel colors requested by user
    gpu_color = "#9ecae1"   # pastel light blue
    cpu_color = "#f4a582"   # pastel coral orange

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=False, sharey=False)

    for row_idx, bs in enumerate(batch_sizes):
        ax_gpu = axes[row_idx][0]
        ax_cpu = axes[row_idx][1]

        t, gpu_mean, gpu_std, cpu_mean, cpu_std = timelines[bs]

        gpu_low = [max(0.0, m - s) for m, s in zip(gpu_mean, gpu_std)]
        gpu_high = [min(100.0, m + s) for m, s in zip(gpu_mean, gpu_std)]
        cpu_low = [max(0.0, m - s) for m, s in zip(cpu_mean, cpu_std)]
        cpu_high = [min(100.0, m + s) for m, s in zip(cpu_mean, cpu_std)]

        ax_gpu.plot(t, gpu_mean, color=gpu_color, linewidth=2.1, label="Mean")
        ax_gpu.fill_between(t, gpu_low, gpu_high, color=gpu_color, alpha=0.28, label="Std band")

        ax_cpu.plot(t, cpu_mean, color=cpu_color, linewidth=2.1, label="Mean")
        ax_cpu.fill_between(t, cpu_low, cpu_high, color=cpu_color, alpha=0.28, label="Std band")

        ax_gpu.set_ylim(95.0, 100.5)
        ax_cpu.set_ylim(0.0, 3.0)

        if row_idx == 0:
            ax_gpu.set_title("GPU Utilization (%)", fontweight="bold")
            ax_cpu.set_title("CPU Utilization (%)", fontweight="bold")

        ax_gpu.set_ylabel(f"Batch size {bs}\nUtilization (%)", fontweight="bold")
        ax_cpu.set_ylabel("")

        if row_idx == len(batch_sizes) - 1:
            ax_gpu.set_xlabel("Time (s)", fontweight="bold")
            ax_cpu.set_xlabel("Time (s)", fontweight="bold")
        else:
            ax_gpu.set_xlabel("")
            ax_cpu.set_xlabel("")

        ax_gpu.grid(alpha=0.25, linestyle=":")
        ax_cpu.grid(alpha=0.25, linestyle=":")

    legend_handles = [
        plt.Line2D([0], [0], color="#5f5f5f", lw=2.1, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.28, label="Std deviation"),
    ]
    fig.legend(
        legend_handles,
        ["Mean", "Std deviation"],
        loc="upper center",
        ncol=2,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle("GPU/CPU Utilization vs Time (Mean ± Std)", fontsize=15, fontweight="bold", y=1.04)

    out_png = plots_dir / "e3_cpu_gpu_utilization_mean_std.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
