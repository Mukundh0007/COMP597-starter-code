#!/usr/bin/env python3
"""
Plot E3 CPU/GPU/MEM utilization mean ± std versus time for batch sizes 8, 16, 32.

Layout:
    Rows    -> GPU, CPU, Memory
    Columns -> Batch sizes (8, 16, 32)

Outputs:
  results/E3/plots/e3_cpu_gpu_mem_3x3.png
  results/E3/plots/e3_timelines_batch{bs}.png  (individual plots)
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

    gpu_mean, gpu_std = [], []
    cpu_mean, cpu_std = [], []
    mem_mean, mem_std = [], []

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(_to_float(row.get("timeline_s", "0")))

            gpu_mean.append(_to_float(row.get("gpu_util_pct_mean", "0")))
            gpu_std.append(_to_float(row.get("gpu_util_pct_std", "0")))

            cpu_mean.append(_to_float(row.get("cpu_util_pct_mean", "0")))
            cpu_std.append(_to_float(row.get("cpu_util_pct_std", "0")))

            mem_mean.append(_to_float(row.get("gpu_mem_mb_mean", "0")))
            mem_std.append(_to_float(row.get("gpu_mem_mb_std", "0")))

    return {
        "t": t,
        "gpu_mean": gpu_mean,
        "gpu_std": gpu_std,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std,
        "mem_mean": mem_mean,
        "mem_std": mem_std,
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


def plot_individual(bs, data, out_path, colors):
    t = data["t"]

    gpu_low, gpu_high = band(data["gpu_mean"], data["gpu_std"], 0, 100)
    cpu_low, cpu_high = band(data["cpu_mean"], data["cpu_std"], 0, 100)
    mem_low, mem_high = band(data["mem_mean"], data["mem_std"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # GPU
    axes[0].plot(t, data["gpu_mean"], color=colors["gpu"], lw=2)
    axes[0].fill_between(t, gpu_low, gpu_high, color=colors["gpu"], alpha=0.28)
    axes[0].set_ylabel("GPU (%)", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25, linestyle=":")

    # CPU
    axes[1].plot(t, data["cpu_mean"], color=colors["cpu"], lw=2)
    axes[1].fill_between(t, cpu_low, cpu_high, color=colors["cpu"], alpha=0.28)
    axes[1].set_ylabel("CPU (%)", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25, linestyle=":")

    # MEM
    axes[2].plot(t, data["mem_mean"], color=colors["mem"], lw=2)
    axes[2].fill_between(t, mem_low, mem_high, color=colors["mem"], alpha=0.28)
    axes[2].set_ylabel("Mem (MB)", fontweight="bold")
    axes[2].set_xlabel("Time (s)", fontweight="bold")
    axes[2].grid(alpha=0.25, linestyle=":")

    fig.suptitle(f"Batch {bs} Timeline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    timelines = {}

    for bs in batch_sizes:
        csv_path = base_dir / f"batch{bs}" / "timeline_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing: {csv_path}")
        timelines[bs] = load_timeline(csv_path)

    # Pastel colors (match your theme)
    colors = {
        "gpu": "#9ecae1",  # pastel blue
        "cpu": "#f4a582",  # pastel coral
        "mem": "#b7e4c7",  # pastel green
    }

    # ===== 3x3 GRID (ROWS = METRIC, COLS = BATCH SIZE) =====
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False)

    for col_idx, bs in enumerate(batch_sizes):
        data = timelines[bs]
        t = data["t"]

        gpu_low, gpu_high = band(data["gpu_mean"], data["gpu_std"], 0, 100)
        cpu_low, cpu_high = band(data["cpu_mean"], data["cpu_std"], 0, 100)
        mem_low, mem_high = band(data["mem_mean"], data["mem_std"])

        # GPU row (row 0)
        ax = axes[0][col_idx]
        ax.plot(t, data["gpu_mean"], color=colors["gpu"], lw=2)
        ax.fill_between(t, gpu_low, gpu_high, color=colors["gpu"], alpha=0.28)
        ax.set_ylim(90, 100.5)
        ax.grid(alpha=0.25, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("GPU (%)", fontweight="bold")

        # CPU row (row 1)
        ax = axes[1][col_idx]
        ax.plot(t, data["cpu_mean"], color=colors["cpu"], lw=2)
        ax.fill_between(t, cpu_low, cpu_high, color=colors["cpu"], alpha=0.28)
        ax.set_ylim(0, 3)
        ax.grid(alpha=0.25, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("CPU (%)", fontweight="bold")

        # MEM row (row 2)
        ax = axes[2][col_idx]
        ax.plot(t, data["mem_mean"], color=colors["mem"], lw=2)
        ax.fill_between(t, mem_low, mem_high, color=colors["mem"], alpha=0.28)
        ax.grid(alpha=0.25, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("Mem (MB)", fontweight="bold")
        ax.set_xlabel("Time (s)", fontweight="bold")

        # Column titles
        axes[0][col_idx].set_title(f"Batch {bs}", fontweight="bold")

        # ===== Save individual subplot (per batch) =====
        out_individual = plots_dir / f"e3_timelines_batch{bs}.png"
        plot_individual(bs, data, out_individual, colors)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="#5f5f5f", lw=2, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.28, label="Std deviation"),
    ]
    fig.legend(handles, ["Mean", "Std deviation"], loc="upper right", ncol=2)

    fig.suptitle("GPU - CPU - Memory Utilization vs Time (Mean ± Std)", fontweight="bold")

    out_path = plots_dir / "e3_cpu_gpu_mem_3x3.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()