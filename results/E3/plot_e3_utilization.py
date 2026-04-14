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

    return {
        "t": t,
        "gpu_mean": gpu_mean,
        "gpu_std": gpu_std,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std,
    }


def band(center, std, lo=None, hi=None):
    low, high = [], []
    for c, s in zip(center, std):
        l = c - s
        h = c + s
        if lo is not None:
            l = max(lo, l)
        if hi is not None:
            h = min(hi, h)
        low.append(l)
        high.append(h)
    return low, high


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

    # Pastel colors (match timelines theme)
    colors = {
        "gpu": "#9ecae1",  # pastel blue
        "cpu": "#f4a582",  # pastel coral
    }

    # ===== 2x3 GRID (ROWS = GPU/CPU, COLS = BATCH SIZE) =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)

    for col_idx, bs in enumerate(batch_sizes):
        data = timelines[bs]
        t = data["t"]

        gpu_low, gpu_high = band(data["gpu_mean"], data["gpu_std"], 0, 100)
        cpu_low, cpu_high = band(data["cpu_mean"], data["cpu_std"], 0, 100)

        # GPU row (row 0)
        ax = axes[0][col_idx]
        ax.plot(t, data["gpu_mean"], color=colors["gpu"], lw=2)
        ax.fill_between(t, gpu_low, gpu_high, color=colors["gpu"], alpha=0.28)
        ax.set_ylim(92, 100.5)
        ax.grid(alpha=0.25, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("GPU (%)", fontweight="bold")
        ax.set_title(f"Batch {bs}", fontweight="bold")

        # CPU row (row 1)
        ax = axes[1][col_idx]
        ax.plot(t, data["cpu_mean"], color=colors["cpu"], lw=2)
        ax.fill_between(t, cpu_low, cpu_high, color=colors["cpu"], alpha=0.28)
        ax.set_ylim(0, 3)
        ax.grid(alpha=0.25, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("CPU (%)", fontweight="bold")
        ax.set_xlabel("Time (s)", fontweight="bold")

    # Legend
    handles = [
        plt.Line2D([0], [0], color="#5f5f5f", lw=2, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.28, label="Std deviation"),
    ]
    fig.legend(handles, ["Mean", "Std deviation"], loc="upper right", ncol=2)

    fig.suptitle("GPU - CPU Utilization vs Time (Mean ± Std)", fontweight="bold")

    out_path = plots_dir / "e3_cpu_gpu_utilization_mean_std.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
