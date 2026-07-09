"""
ParaS dual EP/TP CUDA graph orchestration.

All ParaS-specific CUDA graph logic lives here. Functions operate on
ModelRunner, CudaGraphRunner, and FlashInferAttnBackend externally so
those classes require zero ParaS-specific code.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.utils import (
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory breakdown helper
# ---------------------------------------------------------------------------


def paras_memory_breakdown(device: str, gpu_id: int) -> Dict[str, Any]:
    """Return a dict decomposing GPU memory into:

    - driver_used_gb:     cudaMemGetInfo used = total - free
    - torch_reserved_gb:  PyTorch caching allocator's reserved segments
    - torch_allocated_gb: PyTorch caching allocator's live tensors
    - pools: {pool_id_tuple: bytes} from memory_snapshot
        - ``(0, 0)`` is the default pool
        - ``(0, N>0)`` is a graph-private pool
    - graph_pool_total_gb:   sum of all (0, N>0) pools
    - default_pool_total_gb: sum of (0, 0) pool
    - driver_minus_torch_gb: driver_used - torch_reserved. This is memory held
      by things PyTorch's caching allocator does not track: NCCL internal
      buffers, DeepEP buffers, cuBLAS workspaces allocated via raw cudaMalloc,
      etc. TMS's cudaMalloc LD_PRELOAD hook *can* see these, but only if they
      were allocated inside a tms region.

    Used for instrumenting what fraction of CUDA graph capture cost lands in
    the graph-private pool (TMS-reachable) vs elsewhere (not TMS-reachable
    via graph-pool semantics).
    """
    import torch

    torch.cuda.synchronize(gpu_id)
    free, total = torch.cuda.mem_get_info(gpu_id)
    stats = torch.cuda.memory_stats(gpu_id)

    driver_used = total - free
    torch_reserved = stats.get("reserved_bytes.all.current", 0)
    torch_allocated = stats.get("allocated_bytes.all.current", 0)

    pools: Dict[Any, int] = {}
    try:
        snap = torch.cuda.memory_snapshot()
        for seg in snap:
            if seg.get("device", 0) != gpu_id:
                continue
            pid = seg.get("segment_pool_id", (-1, -1))
            if isinstance(pid, list):
                pid = tuple(pid)
            pools[pid] = pools.get(pid, 0) + seg["total_size"]
    except Exception as e:
        logger.warning(f"ParaS[mem]: memory_snapshot failed: {e}")

    graph_pool_total = sum(v for k, v in pools.items()
                           if isinstance(k, tuple) and len(k) == 2 and k[1] > 0)
    default_pool_total = pools.get((0, 0), 0)

    # ------------------------------------------------------------------
    # Decompose non_torch = driver_used - torch_reserved into named buckets:
    #
    #   non_torch = deepep_buffer     (num_nvl_bytes + num_rdma_bytes,
    #                                  cuMemCreate/cuMemMap path)
    #             + deepep_workspace  (32 MiB hard-coded in
    #                                  deep_ep/csrc/deep_ep.cpp:192
    #                                  via plain cudaMalloc)
    #             + nvshmem_heap      (symmetric heap reserved at
    #                                  internode::init(), default 1 GiB
    #                                  per PE; NVSHMEM_SYMMETRIC_SIZE env
    #                                  overrides. Only allocated when
    #                                  DeepEP's low-latency mode or
    #                                  num_rdma_ranks>1 triggered NVSHMEM.)
    #             + nccl_scratch_est  (per-communicator NCCL buffers:
    #                                  ~9 MiB * nChannels * 2 for
    #                                  collectives + ~12 MiB * nChannels
    #                                  * (nRanks-1) for p2p send/recv.
    #                                  We only count communicators here;
    #                                  bytes are an upper-bound estimate.)
    #             + other_non_torch   (residual: NCCL graph-capture VMM,
    #                                  cuBLAS workspace, misc driver mem)
    #
    # All "est" numbers are labeled so they aren't confused with measured
    # allocations. Driver doesn't expose per-library accounting, so
    # NVSHMEM and NCCL are computed from known constants + library state,
    # not measured at the byte level.
    # ------------------------------------------------------------------

    deepep_buffer_bytes = 0
    deepep_workspace_bytes = 0
    nvshmem_heap_bytes = 0
    deepep_buffer_live = False

    try:
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
        buf = getattr(DeepEPBuffer, "_buffer", None)
        if buf is not None:
            deepep_buffer_live = True
            deepep_buffer_bytes = int(getattr(buf, "num_nvl_bytes", 0) or 0) + int(
                getattr(buf, "num_rdma_bytes", 0) or 0
            )
            # DeepEP's csrc/deep_ep.cpp:192 always cudaMalloc's a 32 MiB
            # workspace per Buffer instance, independent of
            # num_nvl_bytes/num_rdma_bytes.
            deepep_workspace_bytes = 32 * 1024 * 1024

            # NVSHMEM is initialized inside Buffer's C++ constructor when
            # (get_num_rdma_ranks() > 1 OR low_latency_mode). When active,
            # it reserves a symmetric heap of NVSHMEM_SYMMETRIC_SIZE bytes
            # per PE (default 1 GiB). We report the *reservation*, not the
            # physical commit. Physical commit is lazy, bounded below by
            # NVSHMEM_CUMEM_GRANULARITY (DeepEP sets 512 MiB).
            low_latency = bool(getattr(buf, "low_latency_mode", False))
            num_rdma_bytes = int(getattr(buf, "num_rdma_bytes", 0) or 0)
            num_rdma_ranks = 1
            runtime = getattr(buf, "runtime", None)
            if runtime is not None:
                try:
                    num_rdma_ranks = int(runtime.get_num_rdma_ranks())
                except Exception:
                    pass
            nvshmem_active = low_latency or num_rdma_ranks > 1
            if nvshmem_active:
                import os as _os
                sym_size = _os.environ.get("NVSHMEM_SYMMETRIC_SIZE")
                if sym_size is not None:
                    try:
                        nvshmem_heap_bytes = int(sym_size)
                    except ValueError:
                        nvshmem_heap_bytes = 1024 * 1024 * 1024
                else:
                    nvshmem_heap_bytes = 1024 * 1024 * 1024  # NVSHMEM default
    except Exception as e:
        logger.debug(f"ParaS[mem]: DeepEPBuffer introspection failed: {e}")

    # Count live NCCL communicators (module-level singletons in sglang's
    # distributed.parallel_state). Each non-None group holds one NCCL
    # communicator that owns its own default buffers.
    nccl_comm_names: list = []
    seen_group_ids: set = set()
    try:
        import sglang.srt.distributed.parallel_state as _ps
        for name in ("_WORLD", "_TP", "_PP", "_MOE_EP", "_MOE_TP",
                     "_PDMUX_PREFILL_TP_GROUP"):
            g = getattr(_ps, name, None)
            if g is not None and id(g) not in seen_group_ids:
                nccl_comm_names.append(name)
                seen_group_ids.add(id(g))
        # Track ParaS-specific groups too, but dedup against aliases of
        # already-counted groups so we don't double-count shared NCCL comms.
        try:
            import sglang.srt.paras.paras_parallel_state as _paras_ps
            for name in ("_PARAS_TP", "_PARAS_DP", "_PARAS_EP", "_PARAS_SELF"):
                g = getattr(_paras_ps, name, None)
                if g is not None and id(g) not in seen_group_ids:
                    nccl_comm_names.append(name)
                    seen_group_ids.add(id(g))
        except Exception as e:
            logger.debug(f"ParaS[mem]: ParaS group count failed: {e}")
    except Exception as e:
        logger.debug(f"ParaS[mem]: NCCL group count failed: {e}")
    # Per-communicator estimate: NCCL Simple (4 MiB) + LL (256 KiB) +
    # LL128 (~4.7 MiB) = ~9 MiB per protocol, × 2 (send+recv) × 8 channels
    # default on A100 NVLink = ~144 MiB per comm. This is an UPPER BOUND
    # (NCCL may use fewer channels; p2p send/recv buffers add 12 MiB ×
    # (nRanks-1) × nChannels lazily on first send/recv).
    nccl_scratch_est_bytes = len(nccl_comm_names) * 144 * 1024 * 1024

    non_torch = driver_used - torch_reserved
    other_non_torch = (
        non_torch
        - deepep_buffer_bytes
        - deepep_workspace_bytes
        - nvshmem_heap_bytes
        - nccl_scratch_est_bytes
    )

    GB = 1024 ** 3
    return {
        "driver_used_gb": driver_used / GB,
        "torch_reserved_gb": torch_reserved / GB,
        "torch_allocated_gb": torch_allocated / GB,
        "graph_pool_total_gb": graph_pool_total / GB,
        "default_pool_total_gb": default_pool_total / GB,
        "driver_minus_torch_gb": non_torch / GB,
        "deepep_buffer_gb": deepep_buffer_bytes / GB,
        "deepep_workspace_gb": deepep_workspace_bytes / GB,
        "nvshmem_heap_gb": nvshmem_heap_bytes / GB,
        "nccl_scratch_est_gb": nccl_scratch_est_bytes / GB,
        "nccl_comm_count": len(nccl_comm_names),
        "nccl_comm_names": nccl_comm_names,
        "deepep_buffer_live": deepep_buffer_live,
        "other_non_torch_gb": other_non_torch / GB,
        "pools": {str(k): v / GB for k, v in pools.items()},
    }


def paras_mem_log_enabled() -> bool:
    """Return True if detailed ParaS memory-breakdown logging is enabled.

    Gated by env var ``SGLANG_PARAS_MEM_LOG=1``. Default is OFF: the detailed
    3-bucket per-pool breakdown is only emitted when the user explicitly opts
    in. The one-line terse summary (driver_used / torch_reserved / non_torch)
    and the one-time KV-budget log remain on unconditionally so users can
    always see the headline memory numbers.
    """
    import os
    return os.environ.get("SGLANG_PARAS_MEM_LOG", "0") == "1"


def paras_log_memory_breakdown(label: str, device: str, gpu_id: int) -> Dict[str, Any]:
    """Compute and (conditionally) log the memory breakdown.

    - Always computes the breakdown and returns the dict (cheap; needed for
      capture-delta arithmetic even when logging is suppressed).
    - Logs the full per-pool detail line only when ``SGLANG_PARAS_MEM_LOG=1``.
    """
    b = paras_memory_breakdown(device, gpu_id)
    if not paras_mem_log_enabled():
        return b
    pool_detail = ", ".join(f"{k}={v:.3f}GB" for k, v in sorted(b["pools"].items()))
    nccl_names = ",".join(b.get("nccl_comm_names", []))
    logger.info(
        f"ParaS[mem-breakdown:{label}]  "
        f"driver_used={b['driver_used_gb']:.3f}GB  "
        f"torch_reserved={b['torch_reserved_gb']:.3f}GB  "
        f"graph_pool={b['graph_pool_total_gb']:.3f}GB  "
        f"default_pool={b['default_pool_total_gb']:.3f}GB  "
        f"non_torch={b['driver_minus_torch_gb']:.3f}GB  "
        f"(deepep_buf={b['deepep_buffer_gb']:.3f}GB  "
        f"deepep_ws={b['deepep_workspace_gb']:.3f}GB  "
        f"nvshmem={b['nvshmem_heap_gb']:.3f}GB  "
        f"nccl_est={b['nccl_scratch_est_gb']:.3f}GB[{b['nccl_comm_count']}:{nccl_names}]  "
        f"other={b['other_non_torch_gb']:.3f}GB)  "
        f"pools={{{pool_detail}}}"
    )
    return b

# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

_SETTINGS_KEYS = (
    "require_gathered_buffer",
    "require_mlp_tp_gather",
    "require_mlp_sync",
    "require_attn_tp_gather",
    "attn_tp_size",
    "attn_tp_rank",
    "tp_size",
    "dp_size",
    "capture_bs",
    "compile_bs",
)


def paras_refresh_cuda_graph_settings(runner: CudaGraphRunner):
    """Recompute mode-dependent settings from current global state."""
    sa = runner.model_runner.server_args
    runner.require_gathered_buffer = require_gathered_buffer(sa)
    runner.require_mlp_tp_gather = require_mlp_tp_gather(sa)
    runner.require_mlp_sync = require_mlp_sync(sa)
    runner.require_attn_tp_gather = require_attn_tp_gather(sa)
    runner.attn_tp_size = get_attention_tp_size()
    runner.attn_tp_rank = get_attention_tp_rank()
    runner.tp_size = sa.tp_size
    runner.dp_size = sa.dp_size


def paras_save_cuda_graph_state(runner: CudaGraphRunner, mode: str):
    """Save graphs, output buffers, DeepEP state, attention-backend
    cuda-graph state, mode-dependent settings, and the graph memory pool
    handle for *mode* ('ep' or 'tp').

    Saving the graph memory pool is required so that EP and TP can each
    own an isolated pool; ``paras_load_cuda_graph_state`` restores it so
    any downstream code reading ``get_global_graph_memory_pool()`` sees
    the pool that belongs to the currently-active mode.

    The attention-backend's per-mode buffer references (kv_indptr family
    for Triton, per-mode metadata dicts for FlashInfer) are captured
    through the generic ``paras_save_cuda_graph_state`` hook on the
    backend; see ``AttentionBackend.paras_save_cuda_graph_state``.
    """
    from sglang.srt.model_executor.cuda_graph_runner import (
        get_global_graph_memory_pool,
    )

    if not hasattr(runner, "_paras_saved"):
        runner._paras_saved = {}

    state: Dict[str, Any] = {
        "graphs": dict(runner.graphs),
        "output_buffers": dict(runner.output_buffers),
        "deepep_mode": runner.deepep_adapter._captured_deepep_mode,
        "graph_memory_pool": get_global_graph_memory_pool(),
        "attn_backend": runner.model_runner.attn_backend.paras_save_cuda_graph_state(),
    }
    for key in _SETTINGS_KEYS:
        state[key] = getattr(runner, key)

    runner._paras_saved[mode] = state


def paras_load_cuda_graph_state(runner: CudaGraphRunner, mode: str):
    """Restore graphs, output buffers, DeepEP state, attention-backend
    cuda-graph state, mode-dependent settings, the graph memory pool
    handle for *mode*.

    Note: req_to_token_pool and token_to_kv_pool_allocator state are NOT
    saved/restored here. The runtime gather/scatter manager has already
    called paras_resize_and_clear + alloc + writes by the time this load
    runs; restoring init-time state would wipe those mutations and double-
    count slots in both free_pages and tree_cache.evictable. The
    req_to_token tensor's data_ptr stability is now guaranteed by the
    pre-allocated _req_to_token_buffer pattern in ReqToTokenPool, so
    captured FA CUDA graph kernels stay valid without save/restore.
    """
    from sglang.srt.model_executor.cuda_graph_runner import (
        set_global_graph_memory_pool,
    )

    state = runner._paras_saved[mode]
    runner.graphs = state["graphs"]
    runner.output_buffers = state["output_buffers"]
    runner.deepep_adapter._captured_deepep_mode = state["deepep_mode"]
    set_global_graph_memory_pool(state["graph_memory_pool"])
    for key in _SETTINGS_KEYS:
        setattr(runner, key, state[key])
    runner.max_bs = max(runner.capture_bs)
    runner.max_num_token = runner.max_bs * runner.num_tokens_per_bs
    runner.model_runner.attn_backend.paras_load_cuda_graph_state(state["attn_backend"])


# ---------------------------------------------------------------------------
# Runtime swap (called from ModelRunner.paras_configure_tp/ep)
# ---------------------------------------------------------------------------


def paras_swap_cuda_graphs(model_runner: ModelRunner, mode: str):
    """Swap to *mode*'s CUDA graph set if dual graphs were captured."""
    gr = model_runner.graph_runner
    if gr and hasattr(gr, "_paras_saved") and mode in gr._paras_saved:
        paras_load_cuda_graph_state(gr, mode)


# ---------------------------------------------------------------------------
# Init-time dual capture
# ---------------------------------------------------------------------------


def paras_init_dual_cuda_graphs(model_runner: ModelRunner):
    """Capture a second set of CUDA graphs for TP mode at init time.

    Sequence: save EP graphs → switch to TP → capture TP graphs → save
    TP graphs → switch back to EP → load EP graphs.

    EP and TP graphs use **isolated** CUDA graph memory pools: the live
    runner state (``graphs``, ``output_buffers``) is cleared and the
    global graph memory pool is reset to ``None`` before capturing TP so
    ``capture_one_batch_size`` allocates a fresh pool via
    ``graph_pool_handle()``. Each mode's pool handle is saved with its
    state and restored on load, so the two modes never share physical
    pages and CUDA cannot alias their intermediate allocations across
    modes.
    """
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.model_executor.cuda_graph_runner import (
        model_capture_mode,
        set_global_graph_memory_pool,
    )
    from sglang.srt.paras.paras_parallel_state import (
        get_paras_tp_rank,
        get_paras_tp_size,
        paras_comm_configure_ep,
        paras_comm_configure_tp,
    )

    gr = model_runner.graph_runner
    paras_tp_size = get_paras_tp_size()
    paras_tp_rank = get_paras_tp_rank()

    device = model_runner.device
    gpu_id = model_runner.gpu_id

    from sglang.srt.model_executor.cuda_graph_runner import (
        get_global_graph_memory_pool,
    )

    ep_pool = get_global_graph_memory_pool()
    logger.info(f"ParaS: saving EP graphs (ep_pool={ep_pool})")

    # 1. Save EP graph state (includes EP's graph memory pool handle)
    paras_save_cuda_graph_state(gr, "ep")

    # 2. Switch to TP mode — temporarily modify server_args & global state
    saved_args = {
        "enable_dp_attention": model_runner.server_args.enable_dp_attention,
        "dp_size": model_runner.server_args.dp_size,
        "ep_size": model_runner.server_args.ep_size,
        "moe_a2a_backend": model_runner.server_args.moe_a2a_backend,
    }
    saved_moe_backend = moe_utils.MOE_A2A_BACKEND

    model_runner.server_args.enable_dp_attention = False
    model_runner.server_args.dp_size = 1
    model_runner.server_args.ep_size = 1
    model_runner.server_args.moe_a2a_backend = "none"
    moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.NONE

    paras_comm_configure_tp()
    model_runner.token_to_kv_pool.paras_configure_tp(paras_tp_size)
    if hasattr(model_runner.attn_backend, "paras_configure_tp"):
        model_runner.attn_backend.paras_configure_tp(
            paras_tp_size, model_runner.req_to_token_pool.req_to_token
        )
    model_runner.model.paras_configure_tp(paras_tp_size, paras_tp_rank)

    # 3. Clear the live graph dicts before capturing TP. The EP graph
    #    objects are still referenced via ``runner._paras_saved["ep"]``
    #    (see ``paras_save_cuda_graph_state`` which stores copies), so
    #    no EP state is lost. This prevents stale EP entries from
    #    leaking into TP's saved state if the two modes end up with
    #    different batch-size capture lists.
    ep_graph_keys = sorted(gr.graphs.keys())
    gr.graphs.clear()
    gr.output_buffers.clear()

    # 4. Force a fresh graph memory pool for TP so its physical pages
    #    are isolated from EP's pool. ``capture_one_batch_size`` will
    #    allocate a new pool via ``device_module.graph_pool_handle()``
    #    on the first batch size and publish it via
    #    ``set_global_graph_memory_pool``.
    set_global_graph_memory_pool(None)

    # 4b. Swap to TP-specific capture batch sizes if configured. The
    #     TP per-rank batch is paras_tp_size x larger than EP for the
    #     same global workload, so TP graphs typically need a wider
    #     range. The runner's input/output buffers were sized for
    #     max(ep_bs, tp_bs) at __init__ so all TP captures fit.
    #     ``capture_bs`` is in _SETTINGS_KEYS, so each mode's value is
    #     saved with its state and restored on ``paras_swap_cuda_graphs``.
    #     ``compile_bs`` likewise — no recompute needed here.
    if getattr(gr, "_paras_tp_capture_bs", None):
        gr.capture_bs = list(gr._paras_tp_capture_bs)

    logger.info(
        f"ParaS: capturing TP graphs (ep-captured bs={ep_graph_keys}, "
        f"tp-target bs={gr.capture_bs})"
    )

    # 5. Refresh settings for TP mode, then capture. The per-capture
    #    memory breakdown is emitted by
    #    ``cuda_graph_runner.capture()`` via ``paras_log_memory_breakdown``.
    paras_refresh_cuda_graph_settings(gr)
    with model_capture_mode():
        gr.capture()

    tp_graph_keys = sorted(gr.graphs.keys())
    tp_pool = get_global_graph_memory_pool()
    logger.info(
        f"ParaS: TP capture done "
        f"(tp_pool={tp_pool}, ep_pool_was={ep_pool}, pools_differ={ep_pool != tp_pool}, "
        f"tp-captured bs={tp_graph_keys})"
    )

    # 6. Save TP graph state (includes TP's fresh pool handle)
    paras_save_cuda_graph_state(gr, "tp")

    # 7. Switch back to EP mode
    model_runner.server_args.enable_dp_attention = saved_args["enable_dp_attention"]
    model_runner.server_args.dp_size = saved_args["dp_size"]
    model_runner.server_args.ep_size = saved_args["ep_size"]
    model_runner.server_args.moe_a2a_backend = saved_args["moe_a2a_backend"]
    moe_utils.MOE_A2A_BACKEND = saved_moe_backend

    paras_comm_configure_ep()
    model_runner.token_to_kv_pool.paras_configure_ep()
    if hasattr(model_runner.attn_backend, "paras_configure_ep"):
        model_runner.attn_backend.paras_configure_ep(
            model_runner.req_to_token_pool.req_to_token
        )
    model_runner.model.paras_configure_ep()

    # 8. Load EP graph state — restores EP's graphs/buffers and the EP
    #    graph memory pool as the global pool. Ready to serve in EP mode.
    paras_load_cuda_graph_state(gr, "ep")
    model_runner.max_total_num_tokens = model_runner.token_to_kv_pool_allocator.size
    model_runner.max_running_requests = model_runner.req_to_token_pool.size

    mem_final = get_available_gpu_memory(device, gpu_id)
    logger.info(
        f"ParaS: dual capture complete "
        f"avail={mem_final:.3f}GB  #EP graphs={len(ep_graph_keys)}  "
        f"#TP graphs={len(tp_graph_keys)}"
    )


# ---------------------------------------------------------------------------
# Measurement utility
# ---------------------------------------------------------------------------


def paras_measure_instantiation_time(runner: CudaGraphRunner):
    """Measure pure cudaGraphInstantiate time by re-capturing each
    batch size with keep_graph=True, then timing instantiate()."""
    from sglang.srt.model_executor.cuda_graph_runner import (
        get_global_graph_memory_pool,
        patch_model,
    )

    saved_graphs = dict(runner.graphs)
    saved_output_buffers = dict(runner.output_buffers)
    inst_times: Dict[int, float] = {}

    for bs in sorted(saved_graphs.keys()):
        with patch_model(
            runner.model_runner.model,
            False,
            num_tokens=bs * runner.num_tokens_per_bs,
            tp_group=runner.model_runner.tp_group,
        ) as forward:
            kg_graph = torch.cuda.CUDAGraph(keep_graph=True)
            orig_create = runner._create_device_graph
            runner._create_device_graph = lambda: kg_graph
            runner.capture_one_batch_size(bs, forward)
            runner._create_device_graph = orig_create

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            kg_graph.instantiate()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            inst_times[bs] = (t1 - t0) * 1000
            del kg_graph

    runner.graphs = saved_graphs
    runner.output_buffers = saved_output_buffers

    total = sum(inst_times.values())
    per_bs = ", ".join(
        f"bs{bs}={inst_times[bs]:.2f}ms" for bs in sorted(inst_times)
    )
    logger.info(
        f"Pure cudaGraphInstantiate time: total={total:.2f}ms "
        f"({len(inst_times)} graphs). Per-bs: {per_bs}"
    )
    return total, inst_times
