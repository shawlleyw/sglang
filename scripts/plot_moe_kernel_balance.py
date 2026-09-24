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

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    moe_times = data["moe_times"].float().numpy()
    batch_sizes = data.get("batch_sizes")
    if batch_sizes is not None:
        batch_sizes = batch_sizes.int().numpy()
    print(f"Loaded: {path}")
    print(f"  moe_times shape    = {moe_times.shape}  "
          f"(decode_steps, num_layers, world_size)")
    if batch_sizes is not None:
        print(f"  batch_sizes shape  = {batch_sizes.shape}  "
              f"(decode_steps, world_size)")
    print(f"  num_total_steps    = {data['num_total_steps']}")
    print(f"  num_decode_steps   = {data['num_decode_steps']}")
    return moe_times, batch_sizes


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


def _plot_extremes(moe_times: np.ndarray, output_dir: Path,
                   upper_percentile: float | None = None):
    """Per-layer extreme time plot across all ranks AND all steps.

    If upper_percentile is None, uses raw max/min.
    Otherwise, clips values outside [100-upper_percentile, upper_percentile]
    before computing extremes, filtering out outliers on both ends.
    """
    num_steps, num_layers, world_size = moe_times.shape

    if upper_percentile is not None:
        lower_percentile = 100 - upper_percentile
        suffix = f"p{upper_percentile:g}"
        title_filter = f" (clipped to p{lower_percentile:g}–p{upper_percentile:g})"
    else:
        suffix = "raw"
        title_filter = ""

    ext_max = np.zeros(num_layers)
    ext_min = np.zeros(num_layers)
    ext_mean = np.zeros(num_layers)
    for layer in range(num_layers):
        vals = moe_times[:, layer, :].ravel()
        if upper_percentile is not None:
            lo = np.percentile(vals, lower_percentile)
            hi = np.percentile(vals, upper_percentile)
            vals = vals[(vals >= lo) & (vals <= hi)]
        ext_max[layer] = vals.max()
        ext_min[layer] = vals.min()
        ext_mean[layer] = vals.mean()

    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.12), 5))
    layers = np.arange(num_layers)
    ax.fill_between(layers, ext_min, ext_max,
                    alpha=0.2, color="steelblue", label="min–max range")
    ax.plot(layers, ext_max, "o-", markersize=3, color="indianred",
            label="max")
    ax.plot(layers, ext_mean, "o-", markersize=3, color="steelblue",
            label="mean")
    ax.plot(layers, ext_min, "o-", markersize=3, color="seagreen",
            label="min")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"MoE Kernel Time Extremes per Layer{title_filter}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"moe_kernel_extremes_per_layer_{suffix}.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_extreme_across_ranks_and_steps(moe_times: np.ndarray, output_dir: Path):
    _plot_extremes(moe_times, output_dir, upper_percentile=None)
    _plot_extremes(moe_times, output_dir, upper_percentile=99)
    _plot_extremes(moe_times, output_dir, upper_percentile=99.9)


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


def print_outlier_steps(moe_times: np.ndarray, batch_sizes: np.ndarray | None,
                        top_k: int = 5):
    """For each layer, find the steps with the highest max-across-ranks time
    and print step_id, time across all ranks, and batch sizes across all ranks."""
    num_steps, num_layers, world_size = moe_times.shape
    # max time across ranks for each (step, layer)
    max_time_per_step = moe_times.max(axis=2)  # [steps, layers]

    print(f"\n  {'='*70}")
    print(f"  Outlier report: top-{top_k} slowest steps per layer")
    print(f"  {'='*70}")

    for layer in range(num_layers):
        layer_times = max_time_per_step[:, layer]  # [steps]
        top_indices = np.argsort(layer_times)[::-1][:top_k]

        print(f"\n  Layer {layer}:")
        for rank_idx, step_idx in enumerate(top_indices):
            times_all_ranks = moe_times[step_idx, layer, :]
            max_t = times_all_ranks.max()
            min_t = times_all_ranks.min()
            time_str = ", ".join(f"{t:.3f}" for t in times_all_ranks)

            line = (f"    #{rank_idx+1}  step={step_idx:<6d}  "
                    f"max={max_t:.3f}ms  min={min_t:.3f}ms  "
                    f"times=[{time_str}]")

            if batch_sizes is not None:
                bsz_all_ranks = batch_sizes[step_idx, :]
                bsz_str = ", ".join(str(b) for b in bsz_all_ranks)
                line += f"  batch_sizes=[{bsz_str}]"

            print(line)


def print_anomaly_analysis(moe_times: np.ndarray):
    """Statistical analysis to determine whether extreme times are anomalies."""
    num_steps, num_layers, world_size = moe_times.shape

    print(f"\n  {'='*70}")
    print(f"  Anomaly analysis")
    print(f"  {'='*70}")

    # --- 1. Per-layer percentile summary ---
    print(f"\n  Per-layer time statistics (ms) across all ranks and steps:")
    print(f"  {'Layer':>5s}  {'mean':>7s}  {'std':>7s}  {'p50':>7s}  "
          f"{'p95':>7s}  {'p99':>7s}  {'p99.9':>7s}  {'max':>7s}  "
          f"{'max/p50':>7s}")
    for layer in range(num_layers):
        vals = moe_times[:, layer, :].ravel()
        mean = vals.mean()
        std = vals.std()
        p50 = np.percentile(vals, 50)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        p999 = np.percentile(vals, 99.9)
        vmax = vals.max()
        ratio = vmax / p50 if p50 > 0 else float("inf")
        print(f"  {layer:>5d}  {mean:>7.3f}  {std:>7.3f}  {p50:>7.3f}  "
              f"{p95:>7.3f}  {p99:>7.3f}  {p999:>7.3f}  {vmax:>7.3f}  "
              f"{ratio:>7.1f}x")

    # --- 2. Which rank is the straggler most often? ---
    print(f"\n  Straggler rank frequency (which rank has the max time per step):")
    # For each (step, layer), find the rank with the highest time
    slowest_rank = moe_times.argmax(axis=2)  # [steps, layers]
    rank_counts = np.zeros(world_size, dtype=int)
    for r in range(world_size):
        rank_counts[r] = (slowest_rank == r).sum()
    total = slowest_rank.size
    print(f"  {'Rank':>6s}  {'Count':>7s}  {'Fraction':>8s}")
    for r in range(world_size):
        print(f"  {r:>6d}  {rank_counts[r]:>7d}  {rank_counts[r]/total:>8.1%}")
    dominant = rank_counts.max() / total
    if dominant > 0.5:
        print(f"  ** Rank {rank_counts.argmax()} is the straggler in "
              f"{dominant:.0%} of cases — likely a systematic issue, not random noise.")
    else:
        print(f"  ** No single rank dominates — stragglers appear random "
              f"(consistent with transient anomalies).")

    # --- 3. How many steps have extreme outliers? ---
    print(f"\n  Outlier step counts (max-across-ranks > threshold):")
    max_per_step_layer = moe_times.max(axis=2)  # [steps, layers]
    for layer in range(num_layers):
        vals = moe_times[:, layer, :].ravel()
        p50 = np.percentile(vals, 50)
        p99 = np.percentile(vals, 99)
        layer_max = max_per_step_layer[:, layer]
        n_over_2x = (layer_max > 2 * p50).sum()
        n_over_3x = (layer_max > 3 * p50).sum()
        n_over_p99 = (layer_max > p99).sum()
        if n_over_2x > 0:
            print(f"    Layer {layer:>3d}: {n_over_2x:>4d} steps > 2x median, "
                  f"{n_over_3x:>4d} steps > 3x median, "
                  f"{n_over_p99:>4d} steps > p99  "
                  f"(median={p50:.3f}ms, p99={p99:.3f}ms)")


def plot_time_distribution(moe_times: np.ndarray, output_dir: Path, top_n_layers: int = 4):
    """Histogram of per-rank times for the layers with the largest max/median ratio,
    showing that extreme values are long-tail outliers."""
    num_steps, num_layers, world_size = moe_times.shape

    # Find layers with the biggest outlier ratio
    ratios = np.zeros(num_layers)
    for layer in range(num_layers):
        vals = moe_times[:, layer, :].ravel()
        p50 = np.percentile(vals, 50)
        ratios[layer] = vals.max() / p50 if p50 > 0 else 0
    top_layers = np.argsort(ratios)[::-1][:top_n_layers]

    fig, axes = plt.subplots(1, len(top_layers),
                             figsize=(5 * len(top_layers), 4), squeeze=False)
    for col, layer in enumerate(top_layers):
        ax = axes[0, col]
        vals = moe_times[:, layer, :].ravel()
        p50 = np.percentile(vals, 50)
        p99 = np.percentile(vals, 99)
        ax.hist(vals, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(p50, color="green", linestyle="--", label=f"median={p50:.3f}")
        ax.axvline(p99, color="orange", linestyle="--", label=f"p99={p99:.3f}")
        ax.axvline(vals.max(), color="red", linestyle="--", label=f"max={vals.max():.3f}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Layer {layer} (max/med={ratios[layer]:.1f}x)")
        ax.legend(fontsize=7)
    fig.suptitle("Time Distribution for Layers with Largest Outliers", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = "moe_kernel_time_distribution.png"
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
    parser.add_argument("--warmup", "-w", type=int, default=10,
                        help="Number of initial steps to skip as warmup (default: 10)")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of outlier steps to report per layer (default: 5)")
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
        moe_times, batch_sizes = load_data(str(pt_path))

        warmup = args.warmup
        if warmup > 0 and moe_times.shape[0] > warmup:
            print(f"  Skipping first {warmup} steps (warmup)")
            moe_times = moe_times[warmup:]
            if batch_sizes is not None:
                batch_sizes = batch_sizes[warmup:]
        elif warmup > 0:
            print(f"  Warning: only {moe_times.shape[0]} steps, cannot skip {warmup} warmup steps")

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
        plot_extreme_across_ranks_and_steps(moe_times, output_dir)
        plot_rank_imbalance(moe_times, output_dir)
        plot_time_distribution(moe_times, output_dir)
        print_outlier_steps(moe_times, batch_sizes, top_k=args.top_k)
        print_anomaly_analysis(moe_times)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
