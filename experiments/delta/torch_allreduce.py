#!/usr/bin/env python3
"""Multi-node torch all-reduce benchmark. Mirrors nccl-tests/all_reduce_perf."""
import os
import time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"world_size={world_size}  backend=nccl  dtype=float32")
        print(f"{'size':>12s}  {'count':>12s}  {'time_us':>10s}  {'algBW_GBs':>10s}  {'busBW_GBs':>10s}")

    byte_size = 1 << 20  # 1 MB
    end_size = 1 << 30   # 1 GB
    warmup_iters = 5
    bench_iters = 20

    while byte_size <= end_size:
        nelems = byte_size // 4
        buf = torch.ones(nelems, device="cuda", dtype=torch.float32)

        for _ in range(warmup_iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(bench_iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_s = elapsed / bench_iters
        alg_bw = byte_size / avg_s / 1e9
        bus_bw = alg_bw * 2 * (world_size - 1) / world_size

        if rank == 0:
            print(f"{byte_size:>12d}  {nelems:>12d}  {avg_s*1e6:>10.1f}  {alg_bw:>10.2f}  {bus_bw:>10.2f}")

        byte_size *= 2

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
