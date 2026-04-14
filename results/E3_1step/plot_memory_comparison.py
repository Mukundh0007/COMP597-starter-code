#!/usr/bin/env python3
"""
Plot memory utilization (mean ± std) versus time for all batch sizes on a single plot.

Input files:
  results/E3_1step/batch8/timeline_summary.csv
  results/E3_1step/batch16/timeline_summary.csv
  results/E3_1step/batch32/timeline_summary.csv

Output:
  results/E3_1step/plots/memory_comparison.png
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
    mem_mean = []
    mem_std = []

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(_to_float(row.get("timeline_s", "0")))
            mem_mean.append(_to_float(row.get("gpu_mem_mb_mean", "0")))
            mem_std.append(_to_float(row.get("gpu_mem_mb_std", "0")))

    return t, mem_mean, mem_std


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    
    # Color scheme for batch sizes (phase bar colors)
    colors_map = {
        8: "#e67e22",   # orange
        16: "#3498db",  # blue
        32: "#2ecc71",  # green
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for bs in batch_sizes:
        csv_path = base_dir / f"batch{bs}" / "timeline_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing timeline summary: {csv_path}")
        
        t, mem_mean, mem_std = load_timeline(csv_path)
        
        mem_low = [m - s for m, s in zip(mem_mean, mem_std)]
        mem_high = [m + s for m, s in zip(mem_mean, mem_std)]
        
        color = colors_map[bs]
        ax.plot(t, mem_mean, color=color, linewidth=2.2, label=f"Batch {bs}", marker='')
        ax.fill_between(t, mem_low, mem_high, color=color, alpha=0.2)

    ax.set_xlabel("Time (s)", fontweight="bold", fontsize=12)
    ax.set_ylabel("GPU Memory (MB)", fontweight="bold", fontsize=12)
    ax.set_title("GPU Memory Utilization vs Time (Mean ± Std)", fontweight="bold", fontsize=14)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best", framealpha=0.95, fontsize=11)

    fig.tight_layout()
    out_path = plots_dir / "memory_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
