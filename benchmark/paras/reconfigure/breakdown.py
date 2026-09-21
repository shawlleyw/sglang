"""Export additive Weights / Graph / Others from recorded switch trials.

Usage (stdlib only; no GPU access):
  python benchmark/paras/reconfigure/breakdown.py \
    --trials path/to/trials.jsonl --reference-trials path/to/v1/trials.jsonl \
    --output path/to/new-breakdown-directory

All buckets use the rank with the largest total switch duration. Independently
maximizing phases would produce an inconsistent, potentially inflated stack.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def milliseconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid duration: {value}")
    return value


def remainder(total, *parts):
    value = total - sum(parts)
    if value < -1e-5:
        raise ValueError("Recorded phases exceed their enclosing duration")
    return max(0.0, value)


def breakdown_trial(trial, *, reference=False):
    if trial.get("status") != "passed":
        raise ValueError("Only passed trials can be decomposed")
    total = milliseconds(trial["switch_ms"])
    result = {
        "method": trial["method"],
        "direction": trial["direction"],
        "vmm": trial.get(
            "vmm", "not_applicable" if trial["method"] == "rebuild" else "off"
        ),
        "repetition": trial.get("repetition", 1),
        "origin": trial.get("result_origin", "reference" if reference else "measured"),
        "total_ms": total,
        "rank": None,
        "weights_ms": None,
        "graph_ms": None,
        "others_ms": None,
        "runtime_nonweight_ms": None,
        "outside_phases_ms": None,
        "auxiliary_enqueue_ms": None,
        "graph_disposal_ms": None,
        "graph_capture_ms": None,
        "breakdown_available": False,
    }
    if trial["method"] == "rebuild":
        # Its initialization timer includes loading, capture and other setup.
        # Do not call that all "Others", or infer a split from another method.
        return result
    ranks = trial["rank_results"]
    if not ranks or len({x["rank"] for x in ranks}) != len(ranks):
        raise ValueError("Missing or duplicate rank results")
    for rank in ranks:
        milliseconds(rank["switch_ms"])
    critical = max(ranks, key=lambda x: x["switch_ms"])
    if not math.isclose(total, critical["switch_ms"], rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError("Trial total is not the maximum rank duration")
    phases = critical["phases"]
    transfer = milliseconds(phases["weight_transfer_ms"])
    runtime = milliseconds(phases["runtime_switch_ms"])
    auxiliary = (
        milliseconds(phases["auxiliary_reload_ms"])
        if trial["method"] in ("host_reload", "host_model_to")
        else 0.0
    )
    if trial["method"] == "full":
        disposal = milliseconds(phases.get("discard_graphs_ms", 0))
        capture = milliseconds(phases.get("graph_capture_ms", 0))
        if disposal or capture:
            raise ValueError("Full method unexpectedly discarded or captured graphs")
    else:
        disposal = milliseconds(phases["discard_graphs_ms"])
        capture = milliseconds(phases["graph_capture_ms"])
    weights, graph = transfer + auxiliary, disposal + capture
    result.update(
        rank=critical["rank"],
        weights_ms=weights,
        graph_ms=graph,
        others_ms=remainder(total, weights, graph),
        runtime_nonweight_ms=remainder(runtime, transfer),
        outside_phases_ms=remainder(total, auxiliary, disposal, runtime, capture),
        auxiliary_enqueue_ms=auxiliary,
        graph_disposal_ms=disposal,
        graph_capture_ms=capture,
        breakdown_available=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--reference-trials", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    excluded = []
    for path, reference in ((args.trials, False), (args.reference_trials, True)):
        if path is None:
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trial = json.loads(line)
            if trial.get("status") != "passed":
                excluded.append({"source": str(path), "trial": trial})
                continue
            row = breakdown_trial(trial, reference=reference)
            row["source"] = str(path.resolve())
            rows.append(row)
    if not rows:
        raise ValueError("No passed trials found")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "breakdown.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    with (args.output / "breakdown.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Switch-time breakdown",
        "",
        "All times are seconds. Each row uses the rank with the largest total "
        "duration; components sum to that total before rounding. Rows remain "
        "individual trials, not independently aggregated phase statistics.",
        "",
        "| Method | Direction | VMM | Origin | Rank | Weights | Graph | Others | Total |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            "—" if row[key] is None else f"{row[key] / 1000:.4f}"
            for key in ("weights_ms", "graph_ms", "others_ms", "total_ms")
        ]
        rank = row["rank"] if row["rank"] is not None else "—"
        lines.append(
            f"| {row['method']} | {row['direction']} | {row['vmm']} | {row['origin']} | {rank} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "Weights = weight_transfer_ms + auxiliary_reload_ms (both host methods). "
        "This includes expert and attention weight transfer, packing/unpacking, "
        "weight allocations/rebinding/source release performed inside the transfer "
        "adapter, layer fences and its completion wait. Auxiliary reload is "
        "asynchronous enqueue time; its completion can be charged to later phases.",
        "",
        "Graph = discard_graphs_ms + graph_capture_ms. Disposal drops references; "
        "capture includes SGLang's capture GC, warmup, graph construction and "
        "instantiation. Full has zero in this bucket because it retains graphs; "
        "the uninstrumented activation of saved graph state remains in Others.",
        "",
        "Others = total - Weights - Graph. It includes parallel-mode/group and "
        "scheduler/sampler reconfiguration, empty-request/KV metadata bookkeeping "
        "and collectives, pool resizing/reset, attention/backend/workspace setup "
        "and rebinding, saved graph activation (including VMM unmap/map, scratch "
        "reset and synchronization when enabled), production waits outside the "
        "transfer phase, final device synchronization/barrier, and Python/timer "
        "overhead. No live KV payload is transferred in these empty-state trials.",
        "",
        "The CSV further separates Others into runtime_nonweight_ms "
        "(runtime_switch_ms minus weight_transfer_ms) and outside_phases_ms "
        "(time outside the named phases, including the final synchronization/"
        "barrier). These are accounting subdivisions, not additional measurements.",
        "",
        "Phases are host wall times, not isolated GPU operation durations. "
        "Asynchronous work may finish in a later phase, particularly auxiliary "
        "H2D copies enqueued before graph disposal. The recorded data cannot "
        "separate those device costs exactly. Model initialization, host snapshot "
        "preparation, switch warmup and the validation probe are excluded from "
        "the switching-method totals.",
        "",
        "Copied restart data remains total-only: its combined initialization "
        "timer does not isolate weight loading from graph capture and other "
        "startup work. Missing buckets are unavailable, not zero. Its v1 "
        "configuration differs from the fresh v2 configuration.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    print("\n".join(lines[: 8 + len(rows)]))


if __name__ == "__main__":
    main()
