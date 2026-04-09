#!/usr/bin/env python3
"""
Plot E3 per-batch phase bars (forward/backward/optimizer) with mean and std deviation.

Input files:
  results/E3/batch8/phase_summary.csv
  results/E3/batch16/phase_summary.csv
  results/E3/batch32/phase_summary.csv

Output file:
  results/E3/plots/e3_phase_bars_mean_std.png
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


def load_phase_summary(path: Path):
    phases = {
        "forward": {"mean_s": 0.0, "std_s": 0.0},
        "backward": {"mean_s": 0.0, "std_s": 0.0},
        "optimizer": {"mean_s": 0.0, "std_s": 0.0},
    }

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = (row.get("phase", "") or "").strip().lower()
            if phase not in phases:
                continue
            phases[phase]["mean_s"] = _to_float(row.get("mean_s", "0"))
            phases[phase]["std_s"] = _to_float(row.get("std_s", "0"))

    return phases


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    phase_order = ["forward", "backward", "optimizer"]
    phase_labels = ["Forward", "Backward", "Optimizer"]
    phase_colors = ["#8ecae6", "#f4a261", "#90be6d"]

    data_by_batch = {}
    for bs in batch_sizes:
        csv_path = base_dir / f"batch{bs}" / "phase_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing phase summary: {csv_path}")
        data_by_batch[bs] = load_phase_summary(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)

    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx]
        phase_data = data_by_batch[bs]

        means_ms = [phase_data[p]["mean_s"] * 1000.0 for p in phase_order]
        stds_ms = [phase_data[p]["std_s"] * 1000.0 for p in phase_order]

        x = list(range(len(phase_order)))
        bars = ax.bar(
            x,
            means_ms,
            color=phase_colors,
            edgecolor="#404040",
            linewidth=0.8,
            alpha=0.9,
        )

        for bar, mean_v, std_v in zip(bars, means_ms, stds_ms):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{mean_v:.2f} ms (± {std_v:.2f} ms)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels)
        ax.set_title(f"Batch size {bs}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        if idx == 0:
            ax.set_ylabel("Time (ms)", fontweight="bold")

    fig.suptitle("Phase Timing Bars (Mean ± Std deviation)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = plots_dir / "e3_phase_bars_mean_std2.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
