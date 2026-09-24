"""Reproducible expert/KV transport sweep and optional nvbandwidth SM reference.

Run in the serving environment. Uses eight GPUs unless --dry-run is given.
Cache volumes are resident EP K+V GiB per GPU across distinct uniform layers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cache-layout", choices=["overlapping", "separate"], default="overlapping")
    parser.add_argument("--cache-gib", nargs="+", type=float, default=[10, 20, 30])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["peer_access", "nccl", "nccl_overlap"],
        default=["peer_access", "nccl", "nccl_overlap"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--nvbandwidth", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two layers and 64 MiB resident cache; not paper data",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.gpus < 2
        or args.warmup < 0
        or args.iters < 1
        or any(v <= 0 for v in args.cache_gib)
    ):
        parser.error(
            "GPU count >= 2, nonnegative warmup, positive iterations/volumes required"
        )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}/python:{ROOT}/python/sglang/srt/paras/csrc" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    base = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
    ]
    common = [
        "--model",
        args.model,
        "--tp-size",
        str(args.gpus),
        "--direction",
        "both",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    if args.smoke:
        common += ["--num-hidden-layers", "2"]
    jobs = []
    for method in args.methods:
        jobs.append(
            (
                f"weights-{method}",
                base
                + [str(HERE / "bench_weights.py")]
                + common
                + [
                    "--kernel",
                    "experts",
                    "--method",
                    method,
                    "--out-csv",
                    str(out / "weights.csv"),
                ],
            )
        )
    for size in ([0.0625] if args.smoke else args.cache_gib):
        for method in args.methods:
            jobs.append(
                (
                    f"cache-{size:g}GiB-{method}",
                    base
                    + [str(HERE / "bench_cache.py")]
                    + common
                    + [
                        "--resident-cache-gib",
                        str(size),
                        "--cache-layout",
                        args.cache_layout,
                        "--load",
                        "1",
                        "--method",
                        method,
                        "--out-csv",
                        str(out / "cache.csv"),
                    ],
                )
            )
    if args.nvbandwidth:
        for size in (256, 512, 1024):
            jobs.append(
                (
                    f"nvbandwidth-{size}MiB",
                    [
                        str(args.nvbandwidth.resolve()),
                        "-t",
                        "device_to_device_memcpy_write_sm",
                        "one_to_all_write_sm",
                        "all_to_one_write_sm",
                        "-b",
                        str(size),
                        "-i",
                        "5",
                        "-m",
                        "-F",
                        "json",
                    ],
                )
            )
    files = sorted(
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".sh", ".md", ".json")
        and "results" not in p.parts
        and "legacy" not in p.relative_to(HERE).parts
    )
    manifest = {
        "created_at": time.time(),
        "model": args.model,
        "gpus": args.gpus,
        "smoke": args.smoke,
        "cache_gib": args.cache_gib,
        "cache_layout": args.cache_layout,
        "warmup": args.warmup,
        "iters": args.iters,
        "dry_run": args.dry_run,
        "attention_weights_transferred": False,
        "swa_enabled": False,
        "jobs": [{"name": n, "argv": c} for n, c in jobs],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if args.nvbandwidth:
        manifest["nvbandwidth_binary_sha256"] = hashlib.sha256(
            args.nvbandwidth.read_bytes()
        ).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "tracked_changes.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    )
    with tarfile.open(out / "benchmark_source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    if args.dry_run:
        print(f"Prepared {len(jobs)} jobs in {out}; no GPU access")
        return
    for name, command in [
        (
            "gpu_inventory",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
        ),
        ("topology", ["nvidia-smi", "topo", "-m"]),
    ]:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        (out / f"{name}.txt").write_text(result.stdout)
    statuses = []
    for name, command in jobs:
        print(f"RUN {name}", flush=True)
        started = time.time()
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        statuses.append(
            {
                "name": name,
                "returncode": result.returncode,
                "seconds": time.time() - started,
            }
        )
        (out / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(f"{name} failed; see {out/name}.log")
        print(f'PASS {name} ({statuses[-1]["seconds"]:.1f}s)', flush=True)
    print(f"Completed {len(jobs)} jobs: {out}", flush=True)


if __name__ == "__main__":
    main()
