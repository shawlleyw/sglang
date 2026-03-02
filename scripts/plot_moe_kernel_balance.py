#!/usr/bin/env python3
"""Visualize MoE kernel execution time heatmaps from recorder dumps.

Usage:
    python plot_moe_kernel_balance.py <path_to_pt_file> [--output-dir <dir>]

The .pt file is produced by the MoEKernelBalanceRecorder.
It contains:
    moe_times:  tensor of shape [#decode_steps, num_layers, world_size]
                Execution time (ms) per layer per rank per decode step.

Examples:
    python plot_moe_kernel_balance.py /tmp/moe_kernel_balance_*.pt
    python plot_moe_kernel_balance.py /tmp/moe_kernel_balance_*.pt -o ./plots
    python plot_moe_kernel_balance.py /tmp/moe_kernel_balance_*.pt --step 0
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    moe_times = data["moe_times"].float().numpy()
    print(f"Loaded: {path}")
    print(f"  moe_times shape    = {moe_times.shape}  "
          f"(decode_steps, num_layers, world_size)")
    print(f"  num_total_steps    = {data['num_total_steps']}")
    print(f"  num_decode_steps   = {data['num_decode_steps']}")
    return moe_times


def plot_single_step_heatmap(moe_times: np.ndarray, step: int, output_dir: Path):
    """Heatmap for one decode step: X = layer, Y = rank, color = time (ms)."""
    data = moe_times[step]  # [num_layers, world_size]
    num_layers, world_size = data.shape

    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.15),
                                    max(4, world_size * 0.4)))
    im = ax.imshow(data.T, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Rank")
    ax.set_title(f"MoE Kernel Time — Decode Step {step}")
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 20)))
    ax.set_yticks(np.arange(world_size))
    fig.colorbar(im, ax=ax, label="Time (ms)")
    fig.tight_layout()
    fname = f"moe_kernel_step_{step}.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_avg_heatmap(moe_times: np.ndarray, output_dir: Path):
    """Average execution time across all decode steps: X = layer, Y = rank."""
    avg = moe_times.mean(axis=0)  # [num_layers, world_size]
    num_layers, world_size = avg.shape

    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.15),
                                    max(4, world_size * 0.4)))
    im = ax.imshow(avg.T, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Rank")
    ax.set_title(f"MoE Kernel Time — Avg over {moe_times.shape[0]} Decode Steps")
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 20)))
    ax.set_yticks(np.arange(world_size))
    fig.colorbar(im, ax=ax, label="Time (ms)")
    fig.tight_layout()
    fname = "moe_kernel_avg_heatmap.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_max_across_ranks(moe_times: np.ndarray, output_dir: Path):
    """Per-layer max time across ranks (the bottleneck), averaged over steps."""
    avg = moe_times.mean(axis=0)  # [num_layers, world_size]
    max_per_layer = avg.max(axis=1)  # [num_layers]
    min_per_layer = avg.min(axis=1)
    num_layers = avg.shape[0]

    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.12), 5))
    layers = np.arange(num_layers)
    ax.fill_between(layers, min_per_layer, max_per_layer,
                    alpha=0.3, color="steelblue", label="min–max range")
    ax.plot(layers, max_per_layer, "o-", markersize=3, color="indianred",
            label="max (bottleneck)")
    ax.plot(layers, min_per_layer, "o-", markersize=3, color="seagreen",
            label="min (fastest)")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Avg Time (ms)")
    ax.set_title("MoE Kernel Bottleneck per Layer (max vs min across ranks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "moe_kernel_bottleneck_per_layer.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_rank_imbalance(moe_times: np.ndarray, output_dir: Path):
    """Per-layer coefficient of variation across ranks (higher = more imbalanced)."""
    avg = moe_times.mean(axis=0)  # [num_layers, world_size]
    mean = avg.mean(axis=1)
    std = avg.std(axis=1)
    cv = np.where(mean > 0, std / mean, 0.0)
    num_layers = cv.shape[0]

    fig, ax = plt.subplots(figsize=(8, max(5, num_layers * 0.12)))
    layers = np.arange(num_layers)
    colors = plt.cm.RdYlGn_r(cv / max(cv.max(), 1e-6))
    ax.barh(layers, cv, color=colors)
    ax.set_xlabel("Coefficient of Variation (std / mean)")
    ax.set_ylabel("Layer ID")
    ax.set_title("MoE Kernel Time Imbalance Across Ranks (lower = more balanced)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = "moe_kernel_rank_imbalance.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MoE kernel execution time heatmaps from .pt dump files"
    )
    parser.add_argument("input", nargs="+", help="Path(s) to .pt file(s), supports glob")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for plots (default: same dir as input)")
    parser.add_argument("--step", "-s", type=int, default=None,
                        help="Decode step index to visualize (default: last step)")
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
        moe_times = load_data(str(pt_path))
        num_steps = moe_times.shape[0]

        if moe_times.sum() == 0:
            print("  All times are zero, skipping.")
            continue

        step = args.step if args.step is not None else num_steps - 1
        if step < 0 or step >= num_steps:
            print(f"  Step {step} out of range [0, {num_steps - 1}], using last step.")
            step = num_steps - 1

        print(f"  Generating plots in {output_dir}/")
        plot_single_step_heatmap(moe_times, step, output_dir)
        plot_avg_heatmap(moe_times, output_dir)
        plot_max_across_ranks(moe_times, output_dir)
        plot_rank_imbalance(moe_times, output_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
