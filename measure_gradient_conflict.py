"""
Quantify gradient disagreement between teachers on the shared student backbone.

Two conditions:
  - Direct Align  : DirectAlign_ViT.Aggregator  (per-teacher MLP heads, no cross-model blocks)
  - Aggregator    : ModelAggregator_ViT_combine.Aggergator  (CrossModelBlock + InnerModelBlock)

Three metrics on student backbone gradients g_i = dL_i / d theta_s :
  PCD  Pairwise Cosine Disagreement  in [0,1],  0 = perfectly aligned
  GIR  Gradient Interference Ratio   in [0,1],  0 = constructive sum
  SCR  Sign Conflict Rate             in [0,1],  fraction of dims w/ sign flip

Usage:
  python measure_gradient_conflict.py \
      --data /path/to/imagenet_val \
      --teacher_types clip dinov3 mae \
      --ablation

  Add --ckpt_agg / --ckpt_direct to evaluate trained checkpoints.
  Omit them to measure a freshly-initialised model (probes the loss landscape).
"""

import os
import sys
import json
import random
import argparse
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Core gradient utilities
# ─────────────────────────────────────────────────────────────────────────────

def _backbone_param_list(model):
    """Return list of student backbone parameters that require grad."""
    # could be set to parameters in a specific layers 
    return [p for p in model.student.parameters() if p.requires_grad]


def get_all_teacher_grads(model, x: torch.Tensor, teacher_types: list) -> list:
    """
    One forward pass, K backward passes with retain_graph.
    Returns a list of K flattened gradient vectors over student backbone params.

    Each g_i captures the gradient signal teacher i would individually
    send to the shared backbone — independent of what other teachers want.
    """
    backbone_params = _backbone_param_list(model)
    device = x.device

    model.zero_grad()
    loss_dict = model(x)

    grads = []
    for idx, t_prefix in enumerate(teacher_types):
        # Sum all loss terms belonging to this teacher
        t_loss = sum(v for k, v in loss_dict.items() if k.startswith(t_prefix))

        # retain graph for every teacher except the last
        t_loss.backward(retain_graph=(idx < len(teacher_types) - 1))

        g = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in backbone_params
        ]).detach().clone()
        grads.append(g)

        # zero backbone grads only so the next backward starts clean
        for p in backbone_params:
            if p.grad is not None:
                p.grad.zero_()

    model.zero_grad()
    return grads


# ─────────────────────────────────────────────────────────────────────────────
# Disagreement metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_disagreement(grads: list) -> dict:
    """
    Args:
        grads: list of K tensors, each shape [P]

    Returns dict with scalar keys PCD, GIR, SCR and matrix key cos_matrix [K,K].
    """
    K = len(grads)
    assert K >= 2, "Need at least 2 teacher gradients."

    G = torch.stack(grads)          # [K, P]
    dev = G.device

    # ── PCD: mean (1 - cos(g_i, g_j)) / 2 over all i<j pairs ────────────────
    normed = F.normalize(G, dim=1)  # [K, P]
    cos_mat = normed @ normed.T     # [K, K]  symmetric, diagonal = 1
    off_diag = ~torch.eye(K, dtype=torch.bool, device=dev)
    pcd = ((1 - cos_mat[off_diag]) / 2).mean().item()

    # ── GIR: (Σ||g_i|| - ||Σg_i||) / Σ||g_i||  ──────────────────────────────
    # Measures how much gradient magnitude is lost to inter-teacher cancellation.
    # Triangle inequality guarantees Σ||g_i|| >= ||Σg_i|| in exact arithmetic;
    # the max(0,...) on the numerator absorbs floating-point violations of this.
    # 0 = fully constructive; 1 = completely canceling.
    sum_norm = G.sum(0).norm().item()
    ind_sum  = G.norm(dim=1).sum().item()
    cancellation = max(0.0, ind_sum - sum_norm)
    gir = cancellation / (ind_sum + 1e-8)

    # ── SCR: fraction of parameter dims where teachers disagree in sign ───────
    signs = G.sign()  # [K, P]
    scr_vals = [
        (signs[i] != signs[j]).float().mean().item()
        for i, j in combinations(range(K), 2)
    ]
    scr = float(np.mean(scr_vals))

    return {
        "PCD": pcd,
        "GIR": gir,
        "SCR": scr,
        "cos_matrix": cos_mat.cpu().numpy(),   # kept for heatmap
    }


# ─────────────────────────────────────────────────────────────────────────────
# Measurement over a dataloader
# ─────────────────────────────────────────────────────────────────────────────

def measure_conflict(model, dataloader, teacher_types: list,
                     device, n_batches: int = 50) -> dict:
    """
    Average disagreement metrics over n_batches mini-batches.

    Returns:
        {metric: {"mean": float, "std": float}}
        plus "cos_matrix_mean": np.ndarray [K, K]
    """
    model.eval()
    model.to(device)

    scalars = []
    cos_mats = []

    for i, (x, _) in enumerate(dataloader):
        if i >= n_batches:
            break
        x = x.to(device)
        grads = get_all_teacher_grads(model, x, teacher_types)
        d = compute_disagreement(grads)
        cos_mats.append(d.pop("cos_matrix"))
        scalars.append(d)

    summary = {
        k: {"mean": float(np.mean([s[k] for s in scalars])),
            "std":  float(np.std( [s[k] for s in scalars]))}
        for k in scalars[0]
    }
    summary["cos_matrix_mean"] = np.stack(cos_mats).mean(0)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-count ablation  (K = 2 .. N)
# ─────────────────────────────────────────────────────────────────────────────

def teacher_count_ablation(
    model_factory,          # callable(teachers, types, dims) -> nn.Module
    teacher_pool: list,
    type_pool: list,
    dim_pool: list,
    dataloader,
    device,
    n_batches: int = 30,
    max_subsets_per_k: int = 5,
    seed: int = 42,
) -> dict:
    """
    Sweep K from 2 to N = len(teacher_pool).
    For each K, average over up to max_subsets_per_k random subsets of size K.

    Returns:
        {k: {metric: {"mean", "std"}, "cos_matrix_mean": ndarray}}
    """
    rng = random.Random(seed)
    N = len(teacher_pool)
    results = {}

    for k in range(2, N + 1):
        all_combos = list(combinations(range(N), k))
        selected   = rng.sample(all_combos, min(max_subsets_per_k, len(all_combos)))

        subset_summaries = []
        for combo in selected:
            sub_teachers = [teacher_pool[i] for i in combo]
            sub_types    = [type_pool[i]    for i in combo]
            sub_dims     = [dim_pool[i]     for i in combo]
            clean_types  = [t.split("-")[0] for t in sub_types]

            model = model_factory(sub_teachers, sub_types, sub_dims).to(device)
            summary = measure_conflict(model, dataloader, clean_types, device, n_batches)
            subset_summaries.append(summary)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scalar_keys = [k_ for k_ in subset_summaries[0] if k_ != "cos_matrix_mean"]
        results[k] = {
            metric: {
                "mean": float(np.mean([s[metric]["mean"] for s in subset_summaries])),
                "std":  float(np.mean([s[metric]["std"]  for s in subset_summaries])),
            }
            for metric in scalar_keys
        }
        results[k]["cos_matrix_mean"] = np.stack(
            [s["cos_matrix_mean"] for s in subset_summaries]
        ).mean(0)

        row = "  ".join(
            f"{m}={results[k][m]['mean']:.4f}±{results[k][m]['std']:.4f}"
            for m in scalar_keys
        )
        print(f"  K={k}: {row}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {"Direct Align": "#E07B54", "Aggregator": "#5B8DB8"}
_METRIC_LABELS = {
    "PCD": "PCD\n(Pairwise Cosine Disagree.)",
    "GIR": "GIR\n(Gradient Interference Ratio)",
    "SCR": "SCR\n(Sign Conflict Rate)",
}


def plot_comparison_bar(agg: dict, direct: dict, teacher_types: list, save_path: str):
    """Grouped bar chart: all three metrics side-by-side for both methods."""
    metrics = ["PCD", "GIR", "SCR"]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (label, m_dict) in zip(
        [-w / 2, w / 2],
        [("Direct Align", direct), ("Aggregator", agg)],
    ):
        vals = [m_dict[m]["mean"] for m in metrics]
        errs = [m_dict[m]["std"]  for m in metrics]
        ax.bar(x + offset, vals, w, yerr=errs, label=label,
               color=_COLORS[label], capsize=5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Gradient Disagreement: Direct Align vs Aggregator\n"
        f"Teachers: {', '.join(teacher_types)}",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_vs_num_teachers(results_agg: dict, results_direct: dict,
                         metric: str, save_path: str):
    """Line plot: metric vs K for both methods with error bars."""
    ks = sorted(results_agg.keys())
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, res in [("Direct Align", results_direct), ("Aggregator", results_agg)]:
        means = [res[k][metric]["mean"] for k in ks]
        stds  = [res[k][metric]["std"]  for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker="o", label=label,
                    color=_COLORS[label], capsize=4, linewidth=2, markersize=6)

    ax.set_xlabel("Number of Teachers K", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} vs Number of Teachers", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_pairwise_heatmap(agg_cos: np.ndarray, direct_cos: np.ndarray,
                          teacher_types: list, save_path: str):
    """Side-by-side heatmaps of pairwise gradient cosine similarities."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cos_mat, title in zip(
        axes,
        [direct_cos, agg_cos],
        ["Direct Align", "Aggregator"],
    ):
        im = ax.imshow(cos_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(teacher_types)))
        ax.set_yticks(range(len(teacher_types)))
        ax.set_xticklabels(teacher_types, fontsize=10)
        ax.set_yticklabels(teacher_types, fontsize=10)
        ax.set_title(title, fontsize=12)
        for i in range(len(teacher_types)):
            for j in range(len(teacher_types)):
                ax.text(j, i, f"{cos_mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Pairwise Gradient Cosine Similarity on Student Backbone",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_all_metrics_vs_k(results_agg: dict, results_direct: dict,
                          save_path: str):
    """One 1x3 figure with PCD / GIR / SCR vs K, both methods."""
    metrics = ["PCD", "GIR", "SCR"]
    ks = sorted(results_agg.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, metric in zip(axes, metrics):
        for label, res in [("Direct Align", results_direct), ("Aggregator", results_agg)]:
            means = [res[k][metric]["mean"] for k in ks]
            stds  = [res[k][metric]["std"]  for k in ks]
            ax.errorbar(ks, means, yerr=stds, marker="o", label=label,
                        color=_COLORS[label], capsize=4, linewidth=2, markersize=6)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("K (# teachers)", fontsize=11)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle("Gradient Conflict vs Number of Teachers", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_dataloader(data_root: str, batch_size: int, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure teacher gradient disagreement on student backbone."
    )
    parser.add_argument("--data",    type=str, required=True,
                        help="ImageNet-style validation dir (ImageFolder format)")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--n_batches",   type=int, default=50,
                        help="Batches for the fixed-K comparison")
    parser.add_argument("--n_batches_ablation", type=int, default=30,
                        help="Batches per subset in teacher-count ablation")
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="conflict_results")
    parser.add_argument("--ckpt_agg",    type=str, default="",
                        help="Aggregator checkpoint (.pth).  Omit → random init.")
    parser.add_argument("--ckpt_direct", type=str, default="",
                        help="DirectAlign checkpoint (.pth).  Omit → random init.")
    parser.add_argument("--teacher_types", nargs="+",
                        default=["clip", "dinov3", "mae"],
                        choices=["clip", "dinov3", "mae", "sam"],
                        help="Teachers to use for the fixed-K comparison")
    parser.add_argument("--student",  type=str, default="vit_b",
                        choices=["vit_b", "vit_l"])
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="Number of Cross+Inner blocks in Aggregator")
    parser.add_argument("--ablation", action="store_true",
                        help="Also sweep K = 2..N and plot conflict vs K")
    parser.add_argument("--max_subsets", type=int, default=5,
                        help="Max random subsets per K in ablation (caps compute)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Teacher dim registry ──────────────────────────────────────────────────
    TEACHER_DIMS = {"clip": 512, "dinov3": 768, "mae": 768, "sam": 256}

    # ── Load teacher models ───────────────────────────────────────────────────
    from backbone.vanilla_vit import VanillaViT
    from backbone.vit import clip_vit_b_16, DINOv3ViT
    from backbone.vit_mae import mae_vit_b_16

    def load_teacher(name: str):
        n = name.split("-")[0]
        if n == "clip":
            return clip_vit_b_16().to(device).eval()
        if n == "dinov3":
            return DINOv3ViT().to(device).eval()
        if n == "mae":
            return mae_vit_b_16().to(device).eval()
        raise ValueError(f"Unknown teacher: {name}")

    print("Loading teachers ...")
    teacher_pool = [load_teacher(t) for t in args.teacher_types]
    dim_pool     = [TEACHER_DIMS[t.split("-")[0]] for t in args.teacher_types]
    in_dim       = 768 if args.student == "vit_b" else 1024

    # ── Student + model factories ─────────────────────────────────────────────
    from ModelAggregator_ViT_combine import Aggergator
    from DirectAlign_ViT import Aggregator as DirectAlignModel

    def make_student():
        vit_name = ("vit_base_patch16_224" if args.student == "vit_b"
                    else "vit_large_patch16_224")
        return VanillaViT(vit_name, pretrained=False)

    def make_aggregator(teachers, types, dims):
        # CrossModelBlock + InnerModelBlock, cross-teacher communication enabled
        return Aggergator(
            make_student(), teachers,
            num_blocks=args.num_blocks,
            in_dim=in_dim, dim=in_dim,
            out_dims=dims,
            token_dim=512,
            teacher_type=types,
        )

    def make_direct(teachers, types, dims):
        # Per-teacher MLP projections only, no cross-model communication
        return DirectAlignModel(
            make_student(), teachers,
            teacher_type=types,
            out_dims=dims,
            in_dim=in_dim,
        )

    def load_ckpt(model, path: str):
        if path and os.path.exists(path):
            ckp = torch.load(path, map_location="cpu", weights_only=False)
            ckp = ckp.get("model", ckp)
            model.load_state_dict(ckp, strict=False)
            print(f"  Loaded: {path}")
        else:
            print("  No checkpoint — using random initialisation.")
        return model

    # ── Dataloader ────────────────────────────────────────────────────────────
    loader = build_dataloader(args.data, args.batch_size)
    clean_types = [t.split("-")[0] for t in args.teacher_types]

    # ── Fixed-K comparison ────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Fixed-K comparison  (K={len(args.teacher_types)} teachers: {clean_types})")
    print(sep)

    model_agg = load_ckpt(
        make_aggregator(teacher_pool, args.teacher_types, dim_pool).to(device),
        args.ckpt_agg,
    )
    model_direct = load_ckpt(
        make_direct(teacher_pool, args.teacher_types, dim_pool).to(device),
        args.ckpt_direct,
    )

    print(f"Measuring Aggregator   ({args.n_batches} batches) ...")
    agg_metrics = measure_conflict(model_agg, loader, clean_types, device, args.n_batches)
    print(f"Measuring Direct Align ({args.n_batches} batches) ...")
    direct_metrics = measure_conflict(model_direct, loader, clean_types, device, args.n_batches)

    # Print table
    print(f"\n{'Metric':<6}  {'Direct Align':>20}  {'Aggregator':>20}")
    print("-" * 52)
    for m in ["PCD", "GIR", "SCR"]:
        d = direct_metrics[m]
        a = agg_metrics[m]
        print(f"{m:<6}  {d['mean']:.4f} ± {d['std']:.4f}      "
              f"{a['mean']:.4f} ± {a['std']:.4f}")

    # Save plots
    plot_comparison_bar(
        agg_metrics, direct_metrics, clean_types,
        save_path=os.path.join(args.out_dir, "conflict_bar.png"),
    )
    plot_pairwise_heatmap(
        agg_metrics["cos_matrix_mean"],
        direct_metrics["cos_matrix_mean"],
        clean_types,
        save_path=os.path.join(args.out_dir, "conflict_heatmap.png"),
    )

    # Save raw numbers
    with open(os.path.join(args.out_dir, "fixed_k_results.json"), "w") as f:
        json.dump({
            "aggregator":   _to_serializable(agg_metrics),
            "direct_align": _to_serializable(direct_metrics),
            "teacher_types": clean_types,
        }, f, indent=2)
    print(f"  Saved fixed_k_results.json")

    # ── Teacher-count ablation ────────────────────────────────────────────────
    if args.ablation:
        print(f"\n{sep}")
        print(f"Teacher-count ablation  K = 2 .. {len(args.teacher_types)}")
        print(f"Max subsets per K: {args.max_subsets}  |  Batches per subset: {args.n_batches_ablation}")
        print(sep)

        print("Aggregator ablation ...")
        results_agg = teacher_count_ablation(
            make_aggregator, teacher_pool, args.teacher_types, dim_pool,
            loader, device, args.n_batches_ablation, args.max_subsets,
        )

        print("Direct Align ablation ...")
        results_direct = teacher_count_ablation(
            make_direct, teacher_pool, args.teacher_types, dim_pool,
            loader, device, args.n_batches_ablation, args.max_subsets,
        )

        # Individual metric plots
        for metric in ["PCD", "GIR", "SCR"]:
            plot_vs_num_teachers(
                results_agg, results_direct, metric=metric,
                save_path=os.path.join(args.out_dir, f"conflict_vs_k_{metric}.png"),
            )

        # Combined 1×3 figure
        plot_all_metrics_vs_k(
            results_agg, results_direct,
            save_path=os.path.join(args.out_dir, "conflict_vs_k_all.png"),
        )

        # Save raw numbers
        with open(os.path.join(args.out_dir, "ablation_results.json"), "w") as f:
            json.dump({
                "aggregator":   _to_serializable(results_agg),
                "direct_align": _to_serializable(results_direct),
                "teacher_types": args.teacher_types,
            }, f, indent=2)
        print("  Saved ablation_results.json")

    print("\nDone. Results in:", args.out_dir)


if __name__ == "__main__":
    main()
