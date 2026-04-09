#!/usr/bin/env python3
"""
Plot E3 phase-time composition as a 1x3 stacked-bar figure.

Input files:
  results/E3/batch8/phase_summary.csv
  results/E3/batch16/phase_summary.csv
  results/E3/batch32/phase_summary.csv

Output:
  results/E3/plots/e3_phase_time_share_stacked_bars.png
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
            phase_means[phase] = max(_to_float(row.get("mean_s", "0"), 0.0), 0.0)
    return phase_means


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]
    phase_order = ["forward", "backward", "optimizer", "remaining"]

    # Match colors used in your E3 phase plots.
    colors = {
        "forward": "#9ecae1",   # pastel light blue
        "backward": "#bcbddc",  # pastel violet
        "optimizer": "#fbb4c4", # pastel pink
        "remaining": "#d9d9d9", # neutral gray
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

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

        values = {
            "forward": phase_means.get("forward", 0.0),
            "backward": phase_means.get("backward", 0.0),
            "optimizer": phase_means.get("optimizer", 0.0),
            "remaining": remaining,
        }

        total = sum(values[p] for p in phase_order)
        if total <= 0.0:
            ax.text(0.5, 0.5, "No phase data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        pct = {p: (values[p] / total) * 100.0 for p in phase_order}

        x = 0.0
        bar_width = 0.55
        bottom = 0.0
        centers = {}

        for p in phase_order:
            h = pct[p]
            ax.bar(
                x,
                h,
                width=bar_width,
                bottom=bottom,
                color=colors[p],
                edgecolor="white",
                linewidth=1.2,
            )
            centers[p] = bottom + h / 2.0
            bottom += h

        # Put numeric percentages inside forward/backward/optimizer segments.
        for p in ["forward", "backward", "optimizer"]:
            h = pct[p]
            if h >= 3.0:
                ax.text(
                    x,
                    centers[p],
                    f"{h:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1f2933",
                )

        # Keep only remaining as an outside callout.
        rem = pct["remaining"]
        ax.annotate(
            f"remaining {rem:.1f}%",
            xy=(x + bar_width / 2.0 - 0.03, centers["remaining"]),
            xytext=(x + bar_width / 2.0 + 0.30, min(108.0, centers["remaining"] + 3.5)),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#444444",
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 1.0},
        )

        ax.set_title(f"Batch Size {bs}", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.50, 1.45)
        ax.set_ylim(0, 115)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25, linestyle=":")

        ax.text(
            0.0,
            -8.0,
            f"step mean: {step_mean * 1000.0:.2f} ms",
            ha="center",
            va="center",
            fontsize=9,
            color="#334e68",
            transform=ax.transData,
        )

    axes[0].set_ylabel("Share of Step Time (%)", fontweight="bold")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in phase_order
    ]
    fig.legend(
        legend_handles,
        phase_order,
        loc="upper center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Phase Composition of Step Time", fontsize=14, fontweight="bold", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = plots_dir / "e3_phase_time_share_stacked_bars.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
