#!/usr/bin/env python3
"""
Plot E3 energy sampled at coarse timeline cadence and infer per-step energy.

Inputs per batch size:
  results/E3/batch{bs}/timeline_summary.csv
  results/E3/batch{bs}/phase_summary.csv  (uses phase=step mean_s)

Outputs:
  results/E3/plots/e3_energy_sampling_and_inferred_step_energy.png
  results/E3/batch{bs}/energy_inferred_per_step.csv

Inference method:
  For each timeline point i (i >= 1), let dt_i = t_i - t_{i-1} and
  E_i = gpu_energy_delta_j_mean at t_i.
  Estimated steps in the interval ~= dt_i / step_mean_s.
  Estimated energy per step ~= E_i / (dt_i / step_mean_s).
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


def load_step_mean_s(path: Path) -> float:
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("phase", "") or "").strip().lower() == "step":
                return max(1e-12, _to_float(row.get("mean_s", "0"), default=0.0))
    raise ValueError(f"No step phase found in {path}")


def load_timeline(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "timeline_s": _to_float(row.get("timeline_s", "0")),
                    "gpu_energy_delta_j_mean": _to_float(row.get("gpu_energy_delta_j_mean", "0")),
                    "gpu_energy_delta_j_std": _to_float(row.get("gpu_energy_delta_j_std", "0")),
                    "num_samples": int(_to_float(row.get("num_samples", "0"), default=0.0)),
                }
            )
    return rows


def infer_energy_per_step(rows, step_mean_s: float):
    out = []
    prev_t = None
    for row in rows:
        t = row["timeline_s"]
        energy_delta = max(0.0, row["gpu_energy_delta_j_mean"])
        energy_std = max(0.0, row["gpu_energy_delta_j_std"])

        if prev_t is None:
            dt = 0.0
            e_per_step = 0.0
            e_per_step_std = 0.0
        else:
            dt = max(0.0, t - prev_t)
            if dt > 0.0:
                scale = step_mean_s / dt
                e_per_step = energy_delta * scale
                e_per_step_std = energy_std * scale
            else:
                e_per_step = 0.0
                e_per_step_std = 0.0

        out.append(
            {
                "timeline_s": t,
                "dt_s": dt,
                "gpu_energy_delta_j_mean": energy_delta,
                "gpu_energy_delta_j_std": energy_std,
                "inferred_energy_per_step_j": e_per_step,
                "inferred_energy_per_step_j_std": e_per_step_std,
                "num_samples": row["num_samples"],
            }
        )

        prev_t = t

    return out


def write_inferred_csv(path: Path, rows):
    fieldnames = [
        "timeline_s",
        "dt_s",
        "gpu_energy_delta_j_mean",
        "gpu_energy_delta_j_std",
        "inferred_energy_per_step_j",
        "inferred_energy_per_step_j_std",
        "num_samples",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid_intervals(rows):
    return [r["dt_s"] for r in rows if r["dt_s"] > 0.0]


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [8, 16, 32]

    color_energy = "#4c78a8"
    color_step = "#f58518"

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=False, sharey=False)

    for row_i, bs in enumerate(batch_sizes):
        timeline_csv = base_dir / f"batch{bs}" / "timeline_summary.csv"
        phase_csv = base_dir / f"batch{bs}" / "phase_summary.csv"

        if not timeline_csv.exists():
            raise FileNotFoundError(f"Missing timeline summary: {timeline_csv}")
        if not phase_csv.exists():
            raise FileNotFoundError(f"Missing phase summary: {phase_csv}")

        step_mean_s = load_step_mean_s(phase_csv)
        raw_rows = load_timeline(timeline_csv)
        inferred_rows = infer_energy_per_step(raw_rows, step_mean_s)

        inferred_csv = base_dir / f"batch{bs}" / "energy_inferred_per_step.csv"
        write_inferred_csv(inferred_csv, inferred_rows)

        t = [r["timeline_s"] for r in inferred_rows]

        e_mean = [r["gpu_energy_delta_j_mean"] for r in inferred_rows]
        e_std = [r["gpu_energy_delta_j_std"] for r in inferred_rows]
        e_low = [max(0.0, m - s) for m, s in zip(e_mean, e_std)]
        e_high = [m + s for m, s in zip(e_mean, e_std)]

        eps_mean = [r["inferred_energy_per_step_j"] for r in inferred_rows]
        eps_std = [r["inferred_energy_per_step_j_std"] for r in inferred_rows]
        eps_low = [max(0.0, m - s) for m, s in zip(eps_mean, eps_std)]
        eps_high = [m + s for m, s in zip(eps_mean, eps_std)]

        dts = _valid_intervals(inferred_rows)
        dt_min = min(dts) if dts else 0.0
        dt_mean = _mean(dts)
        dt_max = max(dts) if dts else 0.0

        ax_l = axes[row_i][0]
        ax_r = axes[row_i][1]

        ax_l.plot(t, e_mean, color=color_energy, lw=1.9, label="Mean")
        ax_l.fill_between(t, e_low, e_high, color=color_energy, alpha=0.25, label="Std deviation")
        ax_l.grid(alpha=0.25, linestyle=":")
        ax_l.set_ylabel(f"Batch size {bs}\nEnergy (J)", fontweight="bold")
        if row_i == 0:
            ax_l.set_title("GPU energy delta per timeline sample", fontweight="bold")

        ax_l.text(
            0.01,
            0.98,
            f"dt min/mean/max: {dt_min:.3f}/{dt_mean:.3f}/{dt_max:.3f} s",
            transform=ax_l.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#d0d0d0"},
        )

        ax_r.plot(t, eps_mean, color=color_step, lw=1.9, label="Mean")
        ax_r.fill_between(t, eps_low, eps_high, color=color_step, alpha=0.25, label="Std deviation")
        ax_r.grid(alpha=0.25, linestyle=":")
        if row_i == 0:
            ax_r.set_title("Inferred energy per step from coarse sampling", fontweight="bold")

        if row_i == len(batch_sizes) - 1:
            ax_l.set_xlabel("Time (s)", fontweight="bold")
            ax_r.set_xlabel("Time (s)", fontweight="bold")

        ax_r.set_ylabel("Energy per step (J)", fontweight="bold")

    handles = [
        plt.Line2D([0], [0], color="#5f5f5f", lw=2.0, label="Mean"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.25, label="Std deviation"),
    ]
    fig.legend(handles, ["Mean", "Std deviation"], loc="upper center", ncol=2, framealpha=0.95)

    fig.suptitle(
        "E3 Energy Sampling (>= 500 ms cadence) and Inferred Per-Step Energy",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = plots_dir / "e3_energy_sampling_and_inferred_step_energy.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {out_png}")
    for bs in batch_sizes:
        print(f"Saved CSV: {base_dir / f'batch{bs}' / 'energy_inferred_per_step.csv'}")


if __name__ == "__main__":
    main()
