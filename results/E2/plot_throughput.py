#!/usr/bin/env python3
"""Plot throughput by batch size for E2 runs.

Input:
  results/E2/e2_runs.csv

Output:
  results/E2/plots/06_throughput_distinct.png
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelweight"] = "bold"


def load_runs(runs_csv):
    rows = []
    with open(runs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row["batch_size"])
            steps = int(row["steps"])
            train_time_s = float(row["train_time_s"])
            throughput = (steps * bs) / train_time_s if train_time_s > 0 else 0.0
            rows.append({
                "batch_size": bs,
                "run": int(row["run"]),
                "throughput": throughput,
            })
    return rows


def main():
    results_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    runs_csv = os.path.join(results_dir, "e2_runs.csv")
    runs = load_runs(runs_csv)

    by_batch = {}
    for r in runs:
        by_batch.setdefault(r["batch_size"], []).append(r["throughput"])

    batch_sizes = sorted(by_batch.keys())
    means = [float(np.mean(by_batch[bs])) for bs in batch_sizes]
    stds = [float(np.std(by_batch[bs])) for bs in batch_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    colors = ["#F8C291", "#A9D6F5", "#BFE3C0"]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        alpha=0.9,
        color=colors[: len(batch_sizes)],
        edgecolor="#2c3e50",
        linewidth=1.8,
    )

    for i, bs in enumerate(batch_sizes):
        vals = by_batch[bs]
        jitter = np.linspace(-0.08, 0.08, num=len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            color="#2c3e50",
            s=45,
            zorder=3,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.7,
        )

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds + [0.0]) * 0.25 + 1.0,
            f"{m:.1f} ± {s:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (samples/s)", fontsize=12, fontweight="bold")
    ax.set_title("E2 Throughput by Batch Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes], fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    out_path = os.path.join(plots_dir, "06_throughput_distinct.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
