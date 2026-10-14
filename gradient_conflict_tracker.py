"""
GradientConflictTracker — drop-in training hook for logging gradient disagreement.

Usage in your training loop:

    from gradient_conflict_tracker import GradientConflictTracker

    # Pick a fixed batch once (before the training loop)
    fixed_batch = next(iter(train_loader))[0][:16]

    tracker = GradientConflictTracker(
        teacher_types=["clip", "dinov3", "mae"],
        fixed_batch=fixed_batch,
        log_file="runs/conflict_history.csv",
        log_interval=200,   # measure every 200 steps
        device=device,
    )

    for step, (x, _) in enumerate(train_loader):
        # ... normal training step ...
        loss.backward()
        optimizer.step()

        tracker.step(model, step)   # no-op on non-interval steps

After training:
    python gradient_conflict_tracker.py --log runs/conflict_history.csv --out conflict_history.png
    # or compare two runs:
    python gradient_conflict_tracker.py \\
        --log runs/agg_conflict.csv runs/direct_conflict.csv \\
        --labels Aggregator DirectAlign \\
        --out comparison.png
"""

import csv
import os
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F


class GradientConflictTracker:
    """
    Measures PCD / GIR / SCR on the student backbone every `log_interval` steps.

    A fixed measurement batch is used for all measurements so that changes
    in the metrics reflect model evolution, not batch-to-batch variance.
    The model is temporarily set to eval() during measurement and restored
    to its original mode afterward.
    """

    def __init__(self, teacher_types: list, fixed_batch: torch.Tensor,
                 log_file: str = "conflict_history.csv",
                 log_interval: int = 100,
                 device: str = "cuda",
                 param_fn=None):
        """
        Args:
            teacher_types:  prefixes matching loss_dict keys, e.g. ["clip", "dinov3", "mae"]
            fixed_batch:    fixed tensor [B, C, H, W] — sampled once, reused every measurement
            log_file:       path to output CSV (created fresh; append if already exists)
            log_interval:   measure every N global steps
            device:         torch device string or torch.device
            param_fn:       callable (unwrapped_model) -> parameter iterable.
                            Defaults to all student backbone parameters.
                            Example: lambda m: m.student.blocks[-1].parameters()
        """
        self.teacher_types = teacher_types
        self.fixed_batch = fixed_batch.to(device)
        self.log_file = log_file
        self.log_interval = log_interval
        self.device = device
        self.param_fn = param_fn or (lambda m: m.student.parameters())
        self._n_calls = 0

        # Write CSV header (truncate if file exists)
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["step", "PCD", "GIR", "SCR"])

    # -------------------------------------------------------------------------
    def step(self, model: torch.nn.Module, global_step: int) -> dict | None:
        """
        Call once per training step. Returns metrics dict on measurement steps,
        None otherwise. Does not interfere with the optimizer state.
        """
        self._n_calls += 1
        if (self._n_calls - 1) % self.log_interval != 0:
            return None

        was_training = model.training
        model.eval()
        try:
            metrics = self._measure(model)
        finally:
            if was_training:
                model.train()

        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [global_step, metrics["PCD"], metrics["GIR"], metrics["SCR"]]
            )
        return metrics

    # -------------------------------------------------------------------------
    def _get_per_teacher_grads(self, model) -> list:
        # unwrap DDP if needed
        unwrapped = model.module if hasattr(model, "module") else model
        backbone_params = [p for p in self.param_fn(unwrapped) if p.requires_grad]
        model.zero_grad()

        with torch.enable_grad():
            loss_dict = model(self.fixed_batch)

        grads = []
        n = len(self.teacher_types)
        for idx, prefix in enumerate(self.teacher_types):
            teacher_loss = sum(
                v for k, v in loss_dict.items() if k.startswith(prefix)
            )
            teacher_loss.backward(retain_graph=(idx < n - 1))

            g = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for p in backbone_params
            ]).detach().clone()
            grads.append(g)

            for p in backbone_params:
                if p.grad is not None:
                    p.grad.zero_()

        model.zero_grad()
        return grads

    def _measure(self, model) -> dict:
        K = len(self.teacher_types)
        grads = self._get_per_teacher_grads(model)
        G = torch.stack(grads)  # [K, P]

        # PCD — Pairwise Cosine Disagreement
        normed = F.normalize(G, dim=1)
        cos_mat = normed @ normed.T  # [K, K]
        off_diag = ~torch.eye(K, dtype=torch.bool, device=self.device)
        pcd = ((1 - cos_mat[off_diag]) / 2).mean().item()

        # GIR — Gradient Interference Ratio
        sum_norm = G.sum(0).norm().item()
        ind_sum = G.norm(dim=1).sum().item()
        cancellation = max(0.0, ind_sum - sum_norm)  # clamp fp rounding
        gir = cancellation / (ind_sum + 1e-8)

        # SCR — Sign Conflict Rate
        signs = G.sign()
        scr_vals = [
            (signs[i] != signs[j]).float().mean().item()
            for i, j in combinations(range(K), 2)
        ]
        scr = float(np.mean(scr_vals))

        return {"PCD": pcd, "GIR": gir, "SCR": scr}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(log_files: list, labels: list = None, out: str = "conflict_history.png"):
    """
    Plot PCD / GIR / SCR over training steps.

    Args:
        log_files:  one or more CSV paths produced by GradientConflictTracker
        labels:     display name for each run (defaults to filename stem)
        out:        output image path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in log_files]

    metrics = ["PCD", "GIR", "SCR"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for log_file, label in zip(log_files, labels):
        steps, pcd_vals, gir_vals, scr_vals = [], [], [], []
        with open(log_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                pcd_vals.append(float(row["PCD"]))
                gir_vals.append(float(row["GIR"]))
                scr_vals.append(float(row["SCR"]))

        for ax, vals, metric in zip(axes, [pcd_vals, gir_vals, scr_vals], metrics):
            ax.plot(steps, vals, label=label, linewidth=1.5)

    for ax, metric in zip(axes, metrics):
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI for plotting saved logs
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot gradient conflict history from CSV logs")
    parser.add_argument("--log", nargs="+", required=True,
                        help="One or more CSV log files from GradientConflictTracker")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels for each log file")
    parser.add_argument("--out", default="conflict_history.png",
                        help="Output image path")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.log):
        parser.error("--labels must have same length as --log")

    plot_history(args.log, args.labels, args.out)
