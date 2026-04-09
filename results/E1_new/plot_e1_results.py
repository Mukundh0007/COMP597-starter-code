#!/usr/bin/env python3
# """
# E1 Results Visualization
# Overlay bar plot of runtime vs batch size (no stacking).
# - Preserves original run values (run1/run2/run3 plotted independently).
# - Shows per-batch mean as a horizontal line and black marker.
# - Uses dynamic y-limits based on observed data spread.
# """

# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import os


# # Plot style
# sns.set_style("whitegrid")
# plt.rcParams["figure.figsize"] = (12, 7)


# # Load summary CSV
# base_dir = Path(__file__).parent
# csv_path = base_dir / "e1_time_summary.csv"
# df = pd.read_csv(csv_path)

# # Ensure deterministic display order
# batch_sizes = sorted(df["batch_size"].tolist())
# run_cols = ["run1_s", "run2_s", "run3_s"]
# run_labels = ["Run 1", "Run 2", "Run 3"]

# # Pastel orange, blue, green
# run_colors = {
#     "Run 1": "#F8C291",
#     "Run 2": "#A9D6F5",
#     "Run 3": "#BFE3C0",
# }

# # Build matrix of run values without transforming/summing data
# run_values = np.array([[df.loc[df["batch_size"] == bs, col].values[0] for bs in batch_sizes] for col in run_cols])

# x = np.arange(len(batch_sizes))
# bar_widths = [0.62, 0.46, 0.30]  # widest to narrowest for overlay effect

# fig, ax = plt.subplots(figsize=(12, 7))

# # Draw overlay bars: same center x, different widths, no stacking
# for i, run_label in enumerate(run_labels):
#     ax.bar(
#         x,
#         run_values[i],
#         width=bar_widths[i],
#         color=run_colors[run_label],
#         edgecolor="white",
#         linewidth=1.2,
#         alpha=0.9,
#         antialiased=True,
#         label=run_label,
#         zorder=2 + i,
#     )

# # Per-batch mean and std overlays
# means = df.set_index("batch_size").loc[batch_sizes, "avg_s"].values
# stds = df.set_index("batch_size").loc[batch_sizes, "std_s"].values

# for xi, mean, std in zip(x, means, stds):
#     # Mean line across the cluster width
#     ax.hlines(
#         y=mean,
#         xmin=xi - 0.36,
#         xmax=xi + 0.36,
#         colors="black",
#         linewidth=2.2,
#         zorder=8,
#     )

#     # Mean point
#     ax.scatter(
#         xi,
#         mean,
#         s=90,
#         color="black",
#         zorder=9,
#     )

#     # Std error bars around mean
#     ax.errorbar(
#         xi,
#         mean,
#         yerr=std,
#         fmt="none",
#         ecolor="black",
#         elinewidth=1.8,
#         capsize=6,
#         capthick=1.8,
#         zorder=8,
#     )

# # Dynamic y-limits from raw run values
# all_runs = run_values.flatten()
# y_min_data = float(np.min(all_runs))
# y_max_data = float(np.max(all_runs))
# spread = max(y_max_data - y_min_data, 1e-6)
# pad = max(0.03, spread * 0.45)
# ax.set_ylim(y_min_data - pad, y_max_data + pad)

# # X labels include mean ± std
# xtick_labels = []
# for bs in batch_sizes:
#     row = df[df["batch_size"] == bs].iloc[0]
#     xtick_labels.append(f"{bs}\n{row['avg_s']:.2f}s ± {row['std_s']:.4f}s")

# ax.set_xticks(x)
# ax.set_xticklabels(xtick_labels, fontsize=11)

# ax.set_xlabel("Batch Size (Mean time ± Std Dev)", fontsize=14, fontweight="bold")
# ax.set_ylabel("Runtime (in seconds)", fontsize=14, fontweight="bold")
# ax.set_title(
#     "End-to-End Runtime vs Batch Size",
#     fontsize=16,
#     fontweight="bold",
#     pad=16,
# )

# # Optional global mean reference line
# # global_mean = float(np.mean(all_runs))
# # ax.axhline(
# #     global_mean,
# #     color="#333333",
# #     linestyle="--",
# #     linewidth=1.2,
# #     alpha=0.7,
# #     label=f"Global mean: {global_mean:.3f}s",
# #     zorder=1,
# # )

# ax.grid(True, axis="y", alpha=0.28)
# ax.grid(False, axis="x")
# ax.legend(title="Runs", fontsize=11, title_fontsize=11, framealpha=0.95)

# plt.tight_layout()

# # Save outputs
# plots_dir = base_dir / "plots"
# if not os.path.exists(plots_dir):
#     os.makedirs(plots_dir)
# png_path = plots_dir / "e1_runtime_vs_batchsize.png"
# # pdf_path = plots_dir / "e1_runtime_vs_batchsize.pdf"

# plt.savefig(png_path, dpi=300, bbox_inches="tight")
# # plt.savefig(pdf_path, bbox_inches="tight")

# print(f"Saved PNG: {png_path}")
# # print(f"Saved PDF: {pdf_path}")

# plt.show()
#!/usr/bin/env python3
"""
E1 Results Visualization
Side-by-side bar plot of runtime vs batch size.
- Run 1 / Run 2 / Run 3 bars are equal width, placed next to each other.
- Per-batch mean shown as a dotted horizontal line with an on-plot label.
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plot style
sns.set_style("whitegrid")

# Load summary CSV
base_dir = Path(__file__).parent
csv_path = base_dir / "e1_time_summary.csv"
df = pd.read_csv(csv_path)

# Ensure deterministic display order
batch_sizes = sorted(df["batch_size"].tolist())
run_cols = ["run1_s", "run2_s", "run3_s"]
run_labels = ["Run 1", "Run 2", "Run 3"]

run_colors = {
    "Run 1": "#F8C291",
    "Run 2": "#A9D6F5",
    "Run 3": "#BFE3C0",
}

# Build matrix: shape (n_runs, n_batches)
run_values = np.array(
    [[df.loc[df["batch_size"] == bs, col].values[0] for bs in batch_sizes] for col in run_cols]
)

n_runs = len(run_labels)
n_batches = len(batch_sizes)

bar_width = 0.22
group_spacing = 1.0  # distance between group centers

# Group centers, then offset each run within the group
group_centers = np.arange(n_batches) * group_spacing
offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * bar_width

fig, ax = plt.subplots(figsize=(12, 7))

# Draw side-by-side bars
for i, run_label in enumerate(run_labels):
    ax.bar(
        group_centers + offsets[i],
        run_values[i],
        width=bar_width,
        color=run_colors[run_label],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.92,
        label=run_label,
        zorder=2,
    )

# Per-batch mean: dotted line spanning the group, with on-plot label
means = df.set_index("batch_size").loc[batch_sizes, "avg_s"].values
half_span = (n_runs * bar_width) / 2 + 0.03  # slightly wider than the bar group

for xi, mean in zip(group_centers, means):
    ax.hlines(
        y=mean,
        xmin=xi - half_span,
        xmax=xi + half_span,
        colors="#333333",
        linewidth=1.8,
        linestyles="dotted",
        zorder=5,
    )
    # Label at the right end of the line
    ax.text(
        xi + half_span + 0.02,
        mean,
        f"{mean:.2f}s",
        va="center",
        ha="left",
        fontsize=9,
        color="#333333",
        zorder=6,
    )

# Dynamic y-limits
all_runs = run_values.flatten()
y_min_data = float(np.min(all_runs))
y_max_data = float(np.max(all_runs))
spread = max(y_max_data - y_min_data, 1e-6)
pad = max(0.03, spread * 0.45)
ax.set_ylim(y_min_data - pad, y_max_data + pad)

# Extend x-axis slightly to fit mean labels
ax.set_xlim(group_centers[0] - half_span - 0.1, group_centers[-1] + half_span + 0.35)

# X labels: batch size + mean ± std
xtick_labels = []
for bs in batch_sizes:
    row = df[df["batch_size"] == bs].iloc[0]
    xtick_labels.append(f"{bs}\n{row['avg_s']:.2f}s ± {row['std_s']:.4f}s")

ax.set_xticks(group_centers)
ax.set_xticklabels(xtick_labels, fontsize=11)

ax.set_xlabel("Batch Size (Mean time ± Std Dev)", fontsize=14, fontweight="bold")
ax.set_ylabel("Runtime (in seconds)", fontsize=14, fontweight="bold")
ax.set_title(
    "End-to-End Runtime vs Batch Size",
    fontsize=16,
    fontweight="bold",
    pad=16,
)

ax.grid(True, axis="y", alpha=0.28)
ax.grid(False, axis="x")
ax.legend(title="Runs", fontsize=11, title_fontsize=11, framealpha=0.95)

plt.tight_layout()

# Save outputs
plots_dir = base_dir / "plots"
os.makedirs(plots_dir, exist_ok=True)
png_path = plots_dir / "e1_runtime_vs_batchsize2.png"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved PNG: {png_path}")

plt.show()