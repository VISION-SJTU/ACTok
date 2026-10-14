"""
Dataset scale ablation figure — CVPR style.

Usage:
    python plot_dataset_scale.py
    python plot_dataset_scale.py --out dataset_scale.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication RC params (mirrors plot_conflict_comparison.py) ────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8,
    "axes.titlesize":       8,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "legend.title_fontsize":7,
    "lines.linewidth":      1.4,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.4,
    "grid.linestyle":       "-",
    "axes.axisbelow":       True,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0,
})

# ── Data ───────────────────────────────────────────────────────────────────────
RATIOS = [1, 20, 50, 100]

DATASETS = {
    "Caltech101": {
        "acc":    [63.1, 85.8, 88.0, 94.4],
        "color":  "#2980B9",
        "marker": "o",
        "ls":     "-",
    },
    "CUB": {
        "acc":    [49.4, 80.1, 83.3, 90.8],
        "color":  "#C0392B",
        "marker": "s",
        "ls":     "--",
    },
    "Aircraft": {
        "acc":    [46.1, 76.6, 81.3, 91.1],
        "color":  "#27AE60",
        "marker": "^",
        "ls":     ":",
    },
}

# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(out: str):
    fig, ax = plt.subplots(figsize=(2.8, 1.8))

    for name, cfg in DATASETS.items():
        ax.plot(RATIOS, cfg["acc"],
                color=cfg["color"],
                linestyle=cfg["ls"],
                linewidth=1.4,
                marker=cfg["marker"],
                markersize=4,
                markeredgewidth=0.6,
                markeredgecolor="white",
                label=name,
                zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(RATIOS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.set_xlabel("Training Data Ratio")
    ax.set_ylabel("Accuracy (%)")

    # y-axis: start just below min value, end just above max
    ax.set_ylim(40, 98)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(which="minor", length=1.5, width=0.4)

    ax.legend(loc="upper left", frameon=False,
              handlelength=1.8, handletextpad=0.4, borderpad=0)

    plt.savefig(out, pad_inches=0)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset_scale.pdf")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)
    plot(out_path)
