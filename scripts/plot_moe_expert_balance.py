#!/usr/bin/env python3
"""Parse expert distribution dump (.pt) and generate heatmaps.

Usage:
    python plot_moe_kernel_balance.py <path_to_pt_file> [--output-dir <dir>]

The .pt file is produced by the ExpertDistributionRecorder in "stat" mode.
It contains:
    logical_count:  tensor of shape [buffer_steps, num_layers, num_logical_experts]
                    Token counts per expert per layer, summed across all ranks.

Example:
    python plot_moe_kernel_balance.py /tmp/expert_distribution_recorder_*.pt
    python plot_moe_kernel_balance.py /tmp/expert_distribution_recorder_*.pt -o ./plots
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    logical_count = data["logical_count"].float().numpy()
    print(f"Loaded: {path}")
    print(f"  logical_count shape = {logical_count.shape}  "
          f"(buffer_steps, num_layers, num_logical_experts)")
    if "average_utilization_rate_over_window" in data:
        print(f"  avg utilization     = {data['average_utilization_rate_over_window']}")
    return logical_count


def plot_total_heatmap(counts: np.ndarray, output_dir: Path):
    """Summed token count per (layer, expert) across all buffered steps."""
    total = counts.sum(axis=0)  # [layers, experts]

    fig, ax = plt.subplots(figsize=(max(10, total.shape[1] * 0.08),
                                    max(6, total.shape[0] * 0.15)))
    im = ax.imshow(total, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Logical Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title("Total Token Count per Expert per Layer (summed over steps)")
    fig.colorbar(im, ax=ax, label="Token Count")
    fig.tight_layout()
    fig.savefig(output_dir / "expert_dist_total_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved expert_dist_total_heatmap.png")


def plot_avg_heatmap(counts: np.ndarray, output_dir: Path):
    """Average token count per (layer, expert) per step."""
    nonzero_steps = (counts.sum(axis=(1, 2)) > 0).sum()
    if nonzero_steps == 0:
        print("  Skipping avg heatmap — no nonzero steps.")
        return
    avg = counts.sum(axis=0) / nonzero_steps  # [layers, experts]

    fig, ax = plt.subplots(figsize=(max(10, avg.shape[1] * 0.08),
                                    max(6, avg.shape[0] * 0.15)))
    im = ax.imshow(avg, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Logical Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title(f"Avg Token Count per Expert per Layer (over {nonzero_steps} steps)")
    fig.colorbar(im, ax=ax, label="Avg Token Count")
    fig.tight_layout()
    fig.savefig(output_dir / "expert_dist_avg_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved expert_dist_avg_heatmap.png")


def plot_expert_load_balance(counts: np.ndarray, output_dir: Path):
    """Per-layer coefficient of variation (std/mean) of expert loads.

    Lower = more balanced. A value of 0 means perfectly uniform.
    """
    total = counts.sum(axis=0).astype(np.float64)  # [layers, experts]
    mean = total.mean(axis=1, keepdims=True)
    std = total.std(axis=1)
    mean_flat = mean.squeeze()
    cv = np.where(mean_flat > 0, std / mean_flat, 0.0)

    num_layers = cv.shape[0]
    fig, ax = plt.subplots(figsize=(8, max(5, num_layers * 0.12)))
    layers = np.arange(num_layers)
    colors = plt.cm.RdYlGn_r(cv / max(cv.max(), 1e-6))
    ax.barh(layers, cv, color=colors)
    ax.set_xlabel("Coefficient of Variation (std / mean)")
    ax.set_ylabel("Layer")
    ax.set_title("Expert Load Imbalance per Layer (lower = more balanced)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "expert_dist_imbalance_per_layer.png", dpi=150)
    plt.close(fig)
    print(f"  Saved expert_dist_imbalance_per_layer.png")


def plot_top_bottom_layers(counts: np.ndarray, output_dir: Path, k: int = 4):
    """Bar charts for the most and least balanced layers."""
    total = counts.sum(axis=0).astype(np.float64)  # [layers, experts]
    num_layers, num_experts = total.shape
    mean = total.mean(axis=1)
    std = total.std(axis=1)
    cv = np.where(mean > 0, std / mean, 0.0)

    k = min(k, num_layers)
    sorted_idx = np.argsort(cv)
    best_layers = sorted_idx[:k]
    worst_layers = sorted_idx[-k:][::-1]

    fig, axes = plt.subplots(2, k, figsize=(4 * k, 8), sharey="row")
    if k == 1:
        axes = axes.reshape(2, 1)

    expert_ids = np.arange(num_experts)
    for col, layer_idx in enumerate(best_layers):
        ax = axes[0, col]
        ax.bar(expert_ids, total[layer_idx], width=1.0, color="steelblue", edgecolor="none")
        ax.set_title(f"Layer {layer_idx} (CV={cv[layer_idx]:.3f})", fontsize=9)
        ax.set_xlabel("Expert ID", fontsize=8)
        if col == 0:
            ax.set_ylabel("Token Count")
    axes[0, 0].annotate("Most Balanced", xy=(0, 1.02), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", color="green")

    for col, layer_idx in enumerate(worst_layers):
        ax = axes[1, col]
        ax.bar(expert_ids, total[layer_idx], width=1.0, color="indianred", edgecolor="none")
        ax.set_title(f"Layer {layer_idx} (CV={cv[layer_idx]:.3f})", fontsize=9)
        ax.set_xlabel("Expert ID", fontsize=8)
        if col == 0:
            ax.set_ylabel("Token Count")
    axes[1, 0].annotate("Least Balanced", xy=(0, 1.02), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", color="red")

    fig.suptitle("Expert Distribution: Best vs Worst Balanced Layers", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "expert_dist_best_worst_layers.png", dpi=150)
    plt.close(fig)
    print(f"  Saved expert_dist_best_worst_layers.png")


def plot_normalized_heatmap(counts: np.ndarray, output_dir: Path):
    """Per-layer normalized distribution: each row sums to 1.

    Highlights relative hot/cold experts independent of total load per layer.
    """
    total = counts.sum(axis=0).astype(np.float64)  # [layers, experts]
    row_sum = total.sum(axis=1, keepdims=True)
    normalized = np.where(row_sum > 0, total / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, normalized.shape[1] * 0.08),
                                    max(6, normalized.shape[0] * 0.15)))
    im = ax.imshow(normalized, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Logical Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title("Normalized Expert Distribution per Layer (each row sums to 1)")
    fig.colorbar(im, ax=ax, label="Fraction of Tokens")
    fig.tight_layout()
    fig.savefig(output_dir / "expert_dist_normalized_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved expert_dist_normalized_heatmap.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot expert distribution heatmaps from .pt dump files"
    )
    parser.add_argument("input", nargs="+", help="Path(s) to .pt file(s), supports glob")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for plots (default: same dir as input)")
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        paths.extend(glob.glob(pattern))
    if not paths:
        parser.error("No matching .pt files found")

    for pt_path in sorted(paths):
        pt_path = Path(pt_path)
        output_dir = Path(args.output_dir) if args.output_dir else pt_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        counts = load_data(str(pt_path))

        if counts.sum() == 0:
            print("  All counts are zero, skipping.")
            continue

        print(f"  Generating plots in {output_dir}/")
        plot_total_heatmap(counts, output_dir)
        plot_avg_heatmap(counts, output_dir)
        plot_normalized_heatmap(counts, output_dir)
        plot_expert_load_balance(counts, output_dir)
        plot_top_bottom_layers(counts, output_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
