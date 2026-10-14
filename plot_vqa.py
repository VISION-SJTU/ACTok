"""
VQA comparison figure (POPE / VQAv2) — CVPR style.

Usage:
    python plot_vqa.py
    python plot_vqa.py --out vqa_comparison.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication RC params ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8,
    "axes.titlesize":       8,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
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
    "hatch.linewidth":      0.8,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0,
})

# ── Data  (None = not evaluated) ──────────────────────────────────────────────
TASKS   = ["POPE", "VQAv2"]
METHODS = {
    "E-RADIO-L": {"scores": [85.1, 76.7], "color": "#C0392B", "hatch": "////"},
    "ACTok":     {"scores": [87.0, 76.1], "color": "#2980B9", "hatch": "...."},
}

BAR_W   = 0.32
GAP     = 0.08       # gap between bars in a group
Y_MIN   = 74
Y_MAX   = 90

# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(out: str):
    n_tasks   = len(TASKS)
    n_methods = len(METHODS)
    names     = list(METHODS.keys())

    # group centers
    x = np.arange(n_tasks)
    # offsets so bars are centered on each group
    total_w = n_methods * BAR_W + (n_methods - 1) * GAP
    offsets = np.linspace(-total_w / 2 + BAR_W / 2,
                           total_w / 2 - BAR_W / 2,
                           n_methods)

    fig, ax = plt.subplots(figsize=(2.8, 1.8))

    for offset, (name, cfg) in zip(offsets, METHODS.items()):
        for i, (task, score) in enumerate(zip(TASKS, cfg["scores"])):
            if score is None:
                # draw an empty hatched bar to indicate "not evaluated"
                ax.bar(x[i] + offset, Y_MAX - Y_MIN, bottom=Y_MIN,
                       width=BAR_W,
                       color="none",
                       edgecolor=cfg["color"],
                       linewidth=0.7,
                       linestyle="--",
                       hatch="////",
                       zorder=2)
                ax.text(x[i] + offset, Y_MIN + (Y_MAX - Y_MIN) * 0.45,
                        "N/A", ha="center", va="center",
                        fontsize=6, color=cfg["color"])
            else:
                bar = ax.bar(x[i] + offset, score - Y_MIN, bottom=Y_MIN,
                             width=BAR_W,
                             color=cfg["color"],
                             edgecolor="white",
                             linewidth=0.4,
                             hatch=cfg["hatch"],
                             zorder=2,
                             label=name if i == 0 else "_")
                # value label on top of bar
                ax.text(x[i] + offset, score + 0.25,
                        f"{score:.1f}", ha="center", va="bottom",
                        fontsize=6, color=cfg["color"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(Y_MIN, Y_MAX + 2.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.tick_params(which="minor", length=1.5, width=0.4)
    ax.tick_params(bottom=False)          # no x-axis ticks needed for categories

    ax.legend(loc="upper left", frameon=False,
              handlelength=1.2, handletextpad=0.4, borderpad=0)

    plt.savefig(out, pad_inches=0)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="vqa_comparison.pdf")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)
    plot(out_path)
