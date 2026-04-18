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


def _save_flashinfer_metadata(attn_backend: Any, mode: str):
    if not hasattr(attn_backend, "_paras_cuda_graph_metadata"):
        attn_backend._paras_cuda_graph_metadata = {}
    attn_backend._paras_cuda_graph_metadata[mode] = {
        "decode": dict(attn_backend.decode_cuda_graph_metadata),
        "prefill": dict(attn_backend.prefill_cuda_graph_metadata),
        "draft_extend": dict(attn_backend.draft_extend_cuda_graph_metadata),
    }


def _load_flashinfer_metadata(attn_backend: Any, mode: str):
    meta = attn_backend._paras_cuda_graph_metadata[mode]
    attn_backend.decode_cuda_graph_metadata = meta["decode"]
    attn_backend.prefill_cuda_graph_metadata = meta["prefill"]
    attn_backend.draft_extend_cuda_graph_metadata = meta["draft_extend"]


def paras_save_cuda_graph_state(runner: CudaGraphRunner, mode: str):
    """Save graphs, output buffers, DeepEP state, FlashInfer metadata,
    mode-dependent settings, and the graph memory pool handle for *mode*
    ('ep' or 'tp').

    Saving the graph memory pool is required so that EP and TP can each
    own an isolated pool; ``paras_load_cuda_graph_state`` restores it so
    any downstream code reading ``get_global_graph_memory_pool()`` sees
    the pool that belongs to the currently-active mode.
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
    }
    for key in _SETTINGS_KEYS:
        state[key] = getattr(runner, key)

    runner._paras_saved[mode] = state
    _save_flashinfer_metadata(runner.model_runner.attn_backend, mode)


def paras_load_cuda_graph_state(runner: CudaGraphRunner, mode: str):
    """Restore graphs, output buffers, DeepEP state, FlashInfer metadata,
    mode-dependent settings, and the graph memory pool handle for *mode*."""
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
    _load_flashinfer_metadata(runner.model_runner.attn_backend, mode)


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

    logger.info("ParaS: saving EP CUDA graphs and capturing TP graphs...")

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
    gr.graphs.clear()
    gr.output_buffers.clear()

    # 4. Force a fresh graph memory pool for TP so its physical pages
    #    are isolated from EP's pool. ``capture_one_batch_size`` will
    #    allocate a new pool via ``device_module.graph_pool_handle()``
    #    on the first batch size and publish it via
    #    ``set_global_graph_memory_pool``.
    set_global_graph_memory_pool(None)

    # 5. Refresh settings for TP mode, then capture
    paras_refresh_cuda_graph_settings(gr)
    with model_capture_mode():
        gr.capture()

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

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    logger.info(
        f"ParaS: dual CUDA graph capture complete. avail mem={after_mem:.2f} GB"
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
