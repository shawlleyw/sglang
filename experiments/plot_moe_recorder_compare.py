#!/usr/bin/env python3
"""Compare recorder dumps across multiple experiments."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def filter_peak_batch(data: dict, peak_pct: float) -> dict:
    """Keep only the contiguous time range where global batch size >= peak_pct * peak."""
    if peak_pct <= 0:
        return data

    batch_sizes = data["batch_sizes"]
    global_batch = batch_sizes.sum(axis=1)  # [steps]
    peak = global_batch.max()
    threshold = peak_pct * peak
    mask = global_batch >= threshold
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  WARNING: No steps meet {peak_pct:.0%} of peak batch ({peak}). Keeping all.")
        return data

    start, end = indices[0], indices[-1] + 1
    kept = end - start
    total = len(global_batch)
    print(
        f"  Peak filter: global_batch peak={peak}, threshold={threshold:.0f} "
        f"({peak_pct:.0%}), keeping steps [{start}:{end}] ({kept}/{total} steps)"
    )

    filtered = {
        "pt_path": data["pt_path"],
        "batch_sizes": batch_sizes[start:end],
        "moe_times": data["moe_times"][start:end],
    }
    # Rebuild per-rank data
    num_steps = end - start
    ranks = []
    for rank_data in data["ranks"]:
        new_rank = {
            "rank": rank_data["rank"],
            "batch_sizes": rank_data["batch_sizes"][start:end],
            "step_times": rank_data["step_times"][start:end],
        }
        if "local_tokens" in rank_data:
            # local_tokens was reshaped to [steps * layers]; need to slice by step
            num_layers = data["local_token_counts"].shape[1]
            orig_steps = data["local_token_counts"].shape[0]
            # Reshape back to [steps, layers], slice, reshape again
            lt = rank_data["local_tokens"].reshape(orig_steps, num_layers)
            new_rank["local_tokens"] = lt[start:end].reshape(num_steps * num_layers)
        ranks.append(new_rank)
    filtered["ranks"] = ranks

    if "local_token_counts" in data:
        filtered["local_token_counts"] = data["local_token_counts"][start:end]
    if "timestamps" in data:
        filtered["timestamps"] = data["timestamps"][start:end]
    return filtered


def _resolve_kernel_balance_path(path_or_dir: str) -> str | None:
    if path_or_dir.endswith(".pt") and os.path.exists(path_or_dir):
        return path_or_dir
    matches = sorted(glob.glob(os.path.join(path_or_dir, "moe_kernel_balance_*.pt")))
    if not matches:
        return None
    return matches[-1]


def load_experiment(path_or_dir: str):
    pt_path = _resolve_kernel_balance_path(path_or_dir)
    if pt_path is None:
        print(f"  WARNING: No moe_kernel_balance_*.pt found in {path_or_dir}")
        return None

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    moe_times = data["moe_times"].cpu().float().numpy()
    batch_sizes = data["batch_sizes"].cpu().int().numpy()
    local_token_counts = data.get("local_token_counts")
    if local_token_counts is not None:
        local_token_counts = local_token_counts.cpu().int().numpy()

    # moe_times: [steps, ranks] (per-step total forward time)
    num_steps, world_size = moe_times.shape

    ranks = []
    for rank in range(world_size):
        rank_data = {
            "rank": rank,
            "batch_sizes": batch_sizes[:, rank],
            "step_times": moe_times[:, rank],
        }
        if local_token_counts is not None:
            # local_token_counts: [steps, layers, ranks]
            num_layers = local_token_counts.shape[1]
            rank_data["local_tokens"] = local_token_counts[:, :, rank].reshape(
                num_steps * num_layers
            )
        ranks.append(rank_data)

    timestamps = data.get("timestamps")
    if timestamps is not None:
        timestamps = timestamps.cpu().double().numpy()

    result = {
        "pt_path": pt_path,
        "ranks": ranks,
        "batch_sizes": batch_sizes,
        "moe_times": moe_times,
    }
    if local_token_counts is not None:
        result["local_token_counts"] = local_token_counts
    if timestamps is not None:
        result["timestamps"] = timestamps
    return result


def plot_overlaid_cdf(experiments, value_key, xlabel, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, data) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        first = True
        for rank_data in data["ranks"]:
            vals = rank_data[value_key]
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            if first:
                all_vals = np.concatenate(
                    [r[value_key] for r in data["ranks"] if len(r[value_key]) > 0]
                )
                label_text = (
                    f"{label} (P50={np.percentile(all_vals, 50):.1f}, "
                    f"P99={np.percentile(all_vals, 99):.1f}, N={len(all_vals)})"
                )
                first = False
            else:
                label_text = None
            ax.plot(sorted_vals, cdf, linewidth=1.5, color=color, alpha=0.7, label=label_text)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timeline_grid(experiments, output_path, metric="local_token_counts"):
    """Plot timeline grid. metric can be 'local_token_counts' or 'batch_sizes'."""
    n = len(experiments)
    rows, cols = (1, n) if n <= 2 else (2, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    use_local_tokens = metric == "local_token_counts"

    for i, (label, data) in enumerate(experiments):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        color = COLORS[i % len(COLORS)]

        # Determine x-axis: timestamps if available, else step index
        has_ts = "timestamps" in data
        if has_ts:
            ts = data["timestamps"]
            t0 = ts[0, 0]
            x = ts[:, 0] - t0  # rank-0 elapsed seconds
            xlabel = "Time (s)"
        else:
            x = np.arange(data["batch_sizes"].shape[0])
            xlabel = "Decode Step"

        if use_local_tokens and "local_token_counts" in data:
            # local_token_counts shape: [steps, layers, ranks]
            # Average across layers to get per-step per-rank token load
            ltok = data["local_token_counts"].astype(np.float64)
            per_step_rank = ltok.mean(axis=1)  # [steps, ranks]

            for rank in range(per_step_rank.shape[1]):
                ax.plot(x, per_step_rank[:, rank], linewidth=0.7, alpha=0.25, color=color)

            mean_series = per_step_rank.mean(axis=1)
            ax.plot(x, mean_series, linewidth=2, color=color, label="rank mean")
            ylabel = "Local Tokens (avg over layers)"
        else:
            batch_sizes = data["batch_sizes"]

            for rank in range(batch_sizes.shape[1]):
                ax.plot(x, batch_sizes[:, rank], linewidth=0.7, alpha=0.25, color=color)

            mean_series = batch_sizes.mean(axis=1)
            ax.plot(x, mean_series, linewidth=2, color=color, label="rank mean")
            ylabel = "Batch Size"

        if len(mean_series) > 50:
            window = max(len(mean_series) // 50, 10)
            rolling = np.convolve(mean_series, np.ones(window) / window, mode="valid")
            ax.plot(
                x[window - 1 :],
                rolling,
                linewidth=2.5,
                color="black",
                alpha=0.8,
                label=f"rolling avg (w={window})",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    title = "Local Token Count Timeline" if use_local_tokens else "Batch Size Timeline"
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cumulative_tokens_comparison(experiments, output_path):
    """Side-by-side cumulative token timeline for each experiment.

    Each subplot shows per-rank cumulative local MoE tokens over decode steps,
    plus a bottom panel showing the max-min divergence as % of mean.
    """
    n = len(experiments)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), squeeze=False)

    for i, (label, data) in enumerate(experiments):
        ax = axes[i][0]
        if "local_token_counts" not in data:
            ax.text(0.5, 0.5, "No local_token_counts", transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        ltok = data["local_token_counts"].astype(np.float64)
        per_step_rank = ltok.sum(axis=1)  # [steps, ranks]
        cumulative = np.cumsum(per_step_rank, axis=0)
        num_steps, world_size = cumulative.shape

        has_ts = "timestamps" in data
        if has_ts:
            ts = data["timestamps"]
            t0 = ts[0, 0]
            xlabel = "Time (s)"
        else:
            xlabel = "Decode Step"

        cmap = plt.cm.tab10
        for rank in range(world_size):
            if has_ts:
                x_rank = ts[:, rank] - t0
            else:
                x_rank = np.arange(num_steps)
            ax.plot(
                x_rank, cumulative[:, rank],
                linewidth=1.2, alpha=0.8,
                color=cmap(rank / max(world_size - 1, 1)),
                label=f"Rank {rank}",
            )

        # Annotate final divergence
        final = cumulative[-1]
        gap_pct = 100.0 * (final.max() - final.min()) / max(final.mean(), 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Local Tokens")
        ax.set_title(f"{label}  (final max-min gap: {gap_pct:.2f}%)")
        ax.legend(fontsize=7, ncol=min(world_size, 4), loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cumulative Local MoE Tokens per Rank - Comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare recorder dumps across experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help='Each entry is "Label:path_to_experiment_dir_or_pt"',
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--peak-pct",
        type=float,
        default=0,
        help="Keep only the contiguous time range where global batch size "
        ">= this fraction of peak (e.g. 0.9 for 90%%). 0 = disabled.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = []
    for spec in args.experiments:
        if ":" not in spec:
            print(f"ERROR: Expected 'Label:path', got '{spec}'")
            sys.exit(1)
        label, path_or_dir = spec.split(":", 1)
        print(f"Loading {label} from {path_or_dir}...")
        data = load_experiment(path_or_dir)
        if data is not None:
            if args.peak_pct > 0:
                data = filter_peak_batch(data, args.peak_pct)
            experiments.append((label, data))
            print(
                f"  source={data['pt_path']}, steps={data['batch_sizes'].shape[0]}, "
                f"layers={data['moe_times'].shape[1]}, ranks={data['batch_sizes'].shape[1]}"
            )

    if not experiments:
        print("No data loaded.")
        sys.exit(1)

    print("\nPlotting batch size CDF comparison...")
    plot_overlaid_cdf(
        experiments,
        "batch_sizes",
        xlabel="Batch Size",
        title="MoE Recorder Batch Size CDF - All Experiments",
        output_path=os.path.join(args.output_dir, "cdf_moe_step_batch_size.png"),
    )

    print("Plotting execution time CDF comparison...")
    plot_overlaid_cdf(
        experiments,
        "step_times",
        xlabel="Execution Time (ms)",
        title="MoE Recorder Execution Time CDF - All Experiments",
        output_path=os.path.join(args.output_dir, "cdf_moe_step_exec_time.png"),
    )

    # Check if any experiment has local_token_counts
    has_local_tokens = any(
        "local_tokens" in data["ranks"][0] for _, data in experiments
    )

    if has_local_tokens:
        print("Plotting local token count CDF comparison...")
        plot_overlaid_cdf(
            experiments,
            "local_tokens",
            xlabel="Local Token-Expert Pairs",
            title="Local Token Count CDF - All Experiments",
            output_path=os.path.join(args.output_dir, "cdf_local_token_count.png"),
        )

        print("Plotting local token count timeline grid...")
        plot_timeline_grid(
            experiments,
            output_path=os.path.join(args.output_dir, "timeline_local_tokens.png"),
            metric="local_token_counts",
        )

    print("Plotting batch size timeline grid...")
    plot_timeline_grid(
        experiments,
        output_path=os.path.join(args.output_dir, "timeline_batch_size.png"),
        metric="batch_sizes",
    )

    if has_local_tokens:
        print("Plotting cumulative token comparison...")
        plot_cumulative_tokens_comparison(
            experiments,
            output_path=os.path.join(args.output_dir, "cumulative_tokens_comparison.png"),
        )

    print(f"\nAll comparison plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
