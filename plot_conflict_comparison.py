"""
Compare gradient conflict history between DirectAlign and AcToK.
Publication-quality figure in CVPR style.

Usage:
    python plot_conflict_comparison.py
    python plot_conflict_comparison.py --out figure.pdf   # vector PDF for paper
    python plot_conflict_comparison.py --smooth 11
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication RC params ──────────────────────────────────────────────────────
plt.rcParams.update({
    # Font
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8,
    "axes.titlesize":       8,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "legend.title_fontsize":7,
    # Lines
    "lines.linewidth":      1.4,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    # Spines: keep only bottom + left
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Grid
    "axes.grid":            True,
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.4,
    "grid.linestyle":       "-",
    "axes.axisbelow":       True,
    # Figure
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0,
})

# ── Color / style palette (colorblind-safe) ────────────────────────────────────
PALETTE = [
    {"color": "#C0392B", "ls": "-",  "marker": "o"},   # DirectAlign — crimson solid
    {"color": "#2980B9", "ls": "--", "marker": "s"},   # AcToK       — steel-blue dashed
]

METRICS = ["PCD", "GIR", "SCR"]
METRIC_TITLES = {
    "PCD": "PCD",
    "GIR": "GIR",
    "SCR": "SCR",
}
METRIC_YLABELS = {
    "PCD": "PCD",
    "GIR": "GIR",
    "SCR": "SCR",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    steps, pcd, gir, scr = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            pcd.append(float(row["PCD"]))
            gir.append(float(row["GIR"]))
            scr.append(float(row["SCR"]))
    return {"step": np.array(steps), "PCD": np.array(pcd),
            "GIR": np.array(gir),   "SCR": np.array(scr)}


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


# ── Main plot ──────────────────────────────────────────────────────────────────

def plot(log_files: list, labels: list, out: str, smooth_window: int = 9):
    data = [load_csv(f) for f in log_files]

    # CVPR single-column: ~3.33 in; double-column: ~6.875 in
    # 3-panel side-by-side fits nicely at 6.5 × 1.9 in
    _, axes = plt.subplots(1, 3, figsize=(6.5, 1.25),
                             gridspec_kw={"wspace": 0.42})

    for ax, metric in zip(axes, METRICS):
        for d, label, style in zip(data, labels, PALETTE):
            raw = d[metric]
            smo = smooth(raw, smooth_window)
            steps = d["step"]

            # shaded raw signal
            ax.fill_between(steps, raw, alpha=0.10, color=style["color"], linewidth=0)
            # smoothed line
            ax.plot(steps, smo,
                    color=style["color"],
                    linestyle=style["ls"],
                    linewidth=1.4,
                    label=label,
                    zorder=3)

        # sparse markers on smoothed curve (every ~20 points)
        for d, label, style in zip(data, labels, PALETTE):
            smo = smooth(d[metric], smooth_window)
            steps = d["step"]
            stride = max(1, len(steps) // 8)
            ax.plot(steps[::stride], smo[::stride],
                    marker=style["marker"],
                    markersize=3,
                    linestyle="none",
                    color=style["color"],
                    zorder=4)

        ax.set_xlabel("Training Step")
        ax.set_ylabel(METRIC_YLABELS[metric], labelpad=3)
        # ax.set_title(METRIC_TITLES[metric], fontweight="bold", pad=4)
        ax.set_ylim(0.0, 0.70)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax.tick_params(which="minor", length=1.5, width=0.4)

        # x-axis: show step numbers cleanly
        max_step = int(data[0]["step"].max())
        ax.set_xlim(0, max_step)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))

    # legend inside center panel, upper area
    handles, lbl = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, lbl,
                   loc="upper center",
                   frameon=False,
                   handlelength=1.6,
                   handletextpad=0.4,
                   borderpad=0)

    plt.savefig(out, pad_inches=0)
    plt.close()
    print(f"Saved → {out}")

    # summary table
    print(f"\n{'Metric':<6}  {'Method':<12}  {'Mean':>6}  {'Final':>6}  {'Min':>6}  {'Max':>6}")
    print("─" * 50)
    for metric in METRICS:
        for d, label in zip(data, labels):
            v = d[metric]
            print(f"{metric:<6}  {label:<12}  {v.mean():.4f}  {v[-1]:.4f}  "
                  f"{v.min():.4f}  {v.max():.4f}")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs=2,
                        default=["directalign_history.csv", "actok_history.csv"])
    parser.add_argument("--labels", nargs=2, default=["DirectAlign", "AcToK"])
    parser.add_argument("--out", default="conflict_comparison.png")
    parser.add_argument("--smooth", type=int, default=9)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    log_paths = [p if os.path.isabs(p) else os.path.join(base, p) for p in args.logs]
    out_path  = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)

    plot(log_paths, args.labels, out_path, smooth_window=args.smooth)
