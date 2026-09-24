"""Analyze expert/KV sweeps against an empirical nvbandwidth SM-write reference."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

TESTS = ("one_to_all_write_sm", "all_to_one_write_sm")
METHODS = ("nccl", "nccl_overlap", "peer_access")
LABELS = {"nccl": "NCCL", "nccl_overlap": "NCCL overlap", "peer_access": "Direct"}


def read_nvbandwidth(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "nvbandwidth" in value:
            return value["nvbandwidth"]
    raise ValueError(f"No nvbandwidth JSON found: {path}")


def reference_bandwidth(paths, world):
    evidence = []
    by_test = {name: [] for name in TESTS}
    for path in paths:
        run = read_nvbandwidth(path)
        tests = {item["name"]: item for item in run["testcases"]}
        for name in TESTS:
            item = tests[name]
            if item["status"] != "Passed" or item.get("error"):
                raise ValueError(f"Reference test failed: {path} {name}")
            matrix = item["bandwidth_matrix"]
            if len(matrix) != 1 or len(matrix[0]) != world:
                raise ValueError(f"Expected one {world}-GPU reference row")
            bandwidth = [float(value) for value in matrix[0]]
            if not all(math.isfinite(v) and v > 0 for v in bandwidth):
                raise ValueError("Invalid reference bandwidth")
            by_test[name].append(bandwidth)
            evidence.append(
                {"file": path.name, "test": name, "per_gpu_GBps": bandwidth}
            )
    if not paths:
        raise ValueError("No nvbandwidth reference logs")
    best = {
        name: [max(row[i] for row in by_test[name]) for i in range(world)]
        for name in TESTS
    }
    reference = min(min(best[name]) for name in TESTS)
    return {
        "reference_GBps_per_gpu": reference,
        "selection": "For each GPU and each of outbound/inbound SM-write tests, best mean bandwidth over the fixed 256/512/1024 MiB sweep; minimum over GPUs and both tests.",
        "best_per_gpu_GBps": best,
        "evidence": evidence,
        "scope": "Empirical independent outbound/inbound limits. Not simultaneous all-to-all or a proven theoretical optimum. Remote bytes counted once; no duplex sum.",
    }


def analyze(raw):
    manifest = json.loads((raw / "manifest.json").read_text())
    if manifest.get("smoke") or manifest.get("dry_run"):
        raise ValueError("Smoke/dry-run data are not publication measurements")
    jobs = json.loads((raw / "status.json").read_text())
    if len(jobs) != len(manifest["jobs"]) or any(job["returncode"] for job in jobs):
        raise ValueError("The measurement sweep did not complete successfully")
    world = manifest["gpus"]
    reference = reference_bandwidth(sorted(raw.glob("nvbandwidth-*MiB.log")), world)
    results = []
    cache_slot_policies = set()
    cache_layouts = set()
    manifest_layout = manifest.get("cache_layout", "separate")
    if manifest_layout not in ("separate", "overlapping"):
        raise ValueError("Unknown manifest cache_layout")
    scope_layouts = {
        "distinct_uniform_layers_no_swa": "separate",
        "distinct_uniform_layers_overlapping_no_swa": "overlapping",
    }
    for kind, filename in (("weights", "weights.csv"), ("cache", "cache.csv")):
        with (raw / filename).open() as stream:
            source = list(csv.DictReader(stream))
        for r in source:
            if int(r["tp_size"]) != world:
                raise ValueError("Rank count mismatch")
            if kind == "weights":
                if r["kernel"] != "experts":
                    raise ValueError("Expected expert-only weight measurements")
                payload = int(r["expert_payload_bytes_per_rank"])
                remote = int(r["expert_remote_bytes_per_rank"])
                workload = "Experts"
                volume = 0.0
            else:
                layout = scope_layouts.get(r["scope"])
                if layout is None:
                    raise ValueError("Expected distinct-layer uniform cache data")
                if r.get("cache_layout") and r["cache_layout"] != layout:
                    raise ValueError("Cache layout column disagrees with scope")
                cache_layouts.add(layout)
                if len(cache_layouts) > 1:
                    raise ValueError("Cannot compare mixed separate/overlapping cache layouts")
                if layout != manifest_layout:
                    raise ValueError("Cache layout disagrees with manifest cache_layout")
                payload = int(r["resident_bytes_per_rank_all_layers"])
                remote = int(r["remote_bytes_per_rank_all_layers"])
                volume = float(r["resident_cache_gib_requested"])
                workload = f"KV {volume:g} GiB"
                slot_policy = r.get("slot_policy") or "legacy_random_ep_destination"
                cache_slot_policies.add(slot_policy)
            ms = float(r["total_mean_ms"])
            if not math.isfinite(ms) or ms <= 0 or remote <= 0:
                raise ValueError("Invalid latency or byte count")
            bw = remote / (ms * 1e6)
            bound_ms = remote / (reference["reference_GBps_per_gpu"] * 1e6)
            results.append(
                {
                    "workload": workload,
                    "kind": kind,
                    "resident_cache_gib": volume,
                    "method": r["method"],
                    "direction": r["direction"],
                    "cache_slot_policy": slot_policy if kind == "cache" else "",
                    "cache_layout": layout if kind == "cache" else "",
                    "layers": int(r["num_layers"]),
                    "iterations": int(r["n"]),
                    "total_mean_ms": ms,
                    "total_p50_ms": float(r["total_p50_ms"]),
                    "min_ms": float(r["min_ms"]),
                    "max_ms": float(r["max_ms"]),
                    "payload_bytes_per_gpu": payload,
                    "remote_bytes_per_gpu": remote,
                    "effective_remote_GBps_per_gpu": bw,
                    "sm_reference_GBps_per_gpu": reference["reference_GBps_per_gpu"],
                    "sm_reference_time_ms": bound_ms,
                    "sm_reference_efficiency_percent": 100 * bound_ms / ms,
                    "staging_bytes_per_gpu": int(r["staging_bytes"]),
                }
            )
    if len(cache_slot_policies) > 1:
        raise ValueError(
            "Cannot compare cache measurements with different slot policies"
        )
    expected = {
        (kind, volume, method, direction)
        for kind, volume in [("weights", 0.0)]
        + [("cache", float(v)) for v in manifest["cache_gib"]]
        for method in METHODS
        for direction in ("ep_to_tp", "tp_to_ep")
    }
    keys = {
        (r["kind"], r["resident_cache_gib"], r["method"], r["direction"])
        for r in results
    }
    if len(results) != len(expected) or keys != expected:
        raise ValueError("Missing or duplicate method/workload/direction cells")
    return results, reference


def plot(rows, reference, output):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/paras-kernel-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = {"nccl": "#E6A23C", "nccl_overlap": "#4C78A8", "peer_access": "#D65F5F"}
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    lookup = {(r["workload"], r["method"], r["direction"]): r for r in rows}
    for ax, direction, title in zip(
        axes, ("ep_to_tp", "tp_to_ep"), ("EP → TP", "TP → EP")
    ):
        for i, workload in enumerate(workloads):
            for j, method in enumerate(METHODS):
                r = lookup[workload, method, direction]
                x = i + (j - 1) * 0.24
                ax.bar(
                    x,
                    r["total_mean_ms"],
                    width=0.22,
                    color=colors[method],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
            direct = lookup[workload, "peer_access", direction]
            ax.annotate(
                f"{direct['sm_reference_efficiency_percent']:.0f}%",
                (i + 0.24, direct["total_mean_ms"]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#A64040",
            )
            bound = direct["sm_reference_time_ms"]
            ax.plot(
                [i - 0.36, i + 0.36],
                [bound, bound],
                color="#30343B",
                linestyle="--",
                linewidth=1.3,
                zorder=4,
            )
        ax.set_xticks(
            range(len(workloads)), [w.replace("KV ", "KV\n") for w in workloads]
        )
        ax.set_xlabel(title, fontweight="semibold", labelpad=8)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Transfer time (ms)")
    handles = [Patch(facecolor=colors[m], label=LABELS[m]) for m in METHODS]
    handles.append(
        Line2D([0], [0], color="#30343B", linestyle="--", label="SM-copy reference")
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.85, wspace=0.10)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"kernel_ablation.{suffix}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, reference = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "table.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    lines = [
        "# Direct-transfer kernel ablation",
        "",
        "Mean of maximum-rank CUDA-event times; initialization/checking excluded.",
        "",
        "| Workload | Cache layout | Direction | NCCL (ms) | NCCL overlap (ms) | Direct (ms) | Direct GB/s | % SM reference |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for workload in dict.fromkeys(r["workload"] for r in rows):
        for direction in ("ep_to_tp", "tp_to_ep"):
            subset = {
                r["method"]: r
                for r in rows
                if r["workload"] == workload and r["direction"] == direction
            }
            direct = subset["peer_access"]
            lines.append(
                f"| {workload} | {direct['cache_layout'] or 'N/A'} | {direction} | {subset['nccl']['total_mean_ms']:.3f} | {subset['nccl_overlap']['total_mean_ms']:.3f} | {direct['total_mean_ms']:.3f} | {direct['effective_remote_GBps_per_gpu']:.2f} | {direct['sm_reference_efficiency_percent']:.1f}% |"
            )
    lines += [
        "",
        f"Reference: {reference['reference_GBps_per_gpu']:.3f} decimal GB/s per GPU. {reference['selection']}",
        "",
        "Efficiency = (remote bytes / measured SM-copy bandwidth) / measured transfer time. Remote bytes exclude self copies and are counted once. This is an empirical copy reference, not a proven optimal all-to-all kernel. Its fan-in/fan-out tests run separately; actual kernels have simultaneous all-to-all traffic, layout transformation, self copies, and layer fences.",
        "",
        "KV volumes are resident K+V GiB per EP GPU across distinct uniform layers, without SWA. Weights contain w13+w2 only. NCCL staging/packing/unpacking is timed. TP→EP weight overlap currently uses the sequential schedule. One fresh worker group per method/workload job, measuring both directions with multiple timed iterations; not a multiple-run confidence interval.",
        "",
        "Cache layout: "
        + next(r["cache_layout"] for r in rows if r["kind"] == "cache")
        + ". Distinct cache storage is used for every layer ("
        + ", ".join(str(n) for n in sorted({r["layers"] for r in rows if r["kind"] == "cache"}))
        + " layers). Overlapping layout uses ordered layer transfers through overlapping EP/TP views, with source reinitialization outside every timed iteration; separate layout retains disjoint EP/TP allocations. These layouts must not be mixed in one comparison.",
        "",
        "Cache slot policy: "
        + next(r["cache_slot_policy"] for r in rows if r["kind"] == "cache")
        + ".",
    ]
    (args.output / "TABLE.md").write_text("\n".join(lines) + "\n")
    inputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.input.iterdir()
        if p.name.endswith(".csv") or p.name.startswith("nvbandwidth-")
    }
    (args.output / "input_hashes.json").write_text(json.dumps(inputs, indent=2) + "\n")
    plot(rows, reference, args.output)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
