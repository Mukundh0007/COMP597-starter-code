#!/usr/bin/env python3
"""
Plot E3 phase-time composition as a 1x3 pie-chart figure.

Input files:
  results/E3/batch8/phase_summary.csv
  results/E3/batch16/phase_summary.csv
  results/E3/batch32/phase_summary.csv

Output:
  results/E3/plots/e3_phase_time_share_pies.png
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


def load_phase_means(path: Path):
    phase_means = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "").strip()
            mean_s = _to_float(row.get("mean_s", "0"), default=0.0)
            phase_means[phase] = max(mean_s, 0.0)
    return phase_means


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]

    # Match colors used in plot_e3_phase_matrix.py
    colors = {
        "forward": "#9ecae1",   # pastel light blue
        "backward": "#bcbddc",  # pastel violet
        "optimizer": "#fbb4c4", # pastel pink
        "remaining": "#d9d9d9", # neutral gray
    }

    phase_order = ["forward", "backward", "optimizer", "remaining"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, bs in enumerate(batch_sizes):
        ax = axes[i]
        in_csv = base_dir / f"batch{bs}" / "phase_summary.csv"
        if not in_csv.exists():
            raise FileNotFoundError(f"Missing phase summary CSV: {in_csv}")

        phase_means = load_phase_means(in_csv)
        step_mean = phase_means.get("step", 0.0)

        measured_sum = (
            phase_means.get("forward", 0.0)
            + phase_means.get("backward", 0.0)
            + phase_means.get("optimizer", 0.0)
        )
        remaining = max(step_mean - measured_sum, 0.0)

        values = [
            phase_means.get("forward", 0.0),
            phase_means.get("backward", 0.0),
            phase_means.get("optimizer", 0.0),
            remaining,
        ]
        total_phase = sum(values)

        if total_phase <= 0.0:
            ax.text(0.5, 0.5, "No phase data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # Use normalized phase-only composition for pie slices.
        pct_values = [(v / total_phase) * 100.0 for v in values]

        wedges, texts, autotexts = ax.pie(
            pct_values,
            labels=None,
            colors=[colors[p] for p in phase_order],
            startangle=90,
            autopct="%1.1f%%",
            pctdistance=0.73,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            textprops={"fontsize": 10, "fontweight": "bold", "color": "#1f2933"},
        )

        # Keep most percentages inside, but move "remaining" outside to avoid overlap.
        remaining_idx = phase_order.index("remaining")
        if 0 <= remaining_idx < len(autotexts):
            autotexts[remaining_idx].set_visible(False)
            remaining_pct = pct_values[remaining_idx]
            ax.annotate(
                f"remaining {remaining_pct:.1f}%",
                xy=(0.70, -0.70),
                xytext=(1.20, -1.05),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#444444",
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 1.0},
            )

        ax.set_title(f"Batch Size {bs}", fontsize=12, fontweight="bold")

        # Add one concise per-subplot note showing absolute step mean.
        ax.text(
            0.0,
            -1.20,
            f"step mean: {step_mean * 1000.0:.2f} ms",
            ha="center",
            va="center",
            fontsize=9,
            color="#334e68",
            transform=ax.transData,
        )

    # One shared legend for the entire figure.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[p], markersize=10)
        for p in phase_order
    ]
    fig.legend(
        legend_handles,
        ["forward", "backward", "optimizer", "remaining"],
        loc="upper center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Phase Composition of Step Time", fontsize=14, fontweight="bold", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = plots_dir / "e3_phase_time_share_pies.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
