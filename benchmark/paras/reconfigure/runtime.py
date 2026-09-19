"""Benchmark adapters around real Scheduler/ModelRunner state transitions.

Patches are installed only in fresh benchmark worker processes. Normal serving
imports no benchmark code. Each trial executes one switch with no live requests.
"""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

from common.model_configs import ModelConfig
from common.weight_bundle import COMPONENTS, weight_name
from sglang.srt.paras.mode import ParaSMode

from .storage import IndependentWeightMemoryManager
from .configuration import METHOD_TRANSPORT
from .transfers import (
    naive_nccl_transfer_layer,
    reload_host_layer,
    snapshot_host_tensors,
)


def dimensions(config):
    return ModelConfig(
        name=config.model_type,
        num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        ),
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        interleaved_w13=config.model_type == "gpt_oss",
    )


def layer_parameters(layer, mode):
    expert = layer.mlp.ep_experts if mode == ParaSMode.EP else layer.mlp.tp_experts
    attention = layer.self_attn
    attr = "full_weight" if mode == ParaSMode.EP else "_paras_tp_weight"
    return {
        "w13": expert.w13_weight,
        "w2": expert.w2_weight,
        "qkv": getattr(attention.qkv_proj, attr),
        "o": getattr(attention.o_proj, attr),
    }


def install_independent_storage():
    """Replace weight storage and transport only inside this worker process."""
    import sglang.srt.model_executor.model_runner as runner_module
    import sglang.srt.paras.paras_cuda_graph as graphs
    from sglang.srt.paras.layers.paras_model import ParaSModelMixin
    from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_group

    runner_module.ParaSMemoryManager = IndependentWeightMemoryManager
    # Non-retaining methods capture only the current mode. Stale saved graphs
    # must not keep source weight allocations alive or execute after rebinding.
    graphs.paras_init_dual_cuda_graphs = lambda runner: None
    graphs.paras_swap_cuda_graphs = lambda runner, mode: None

    def transfer(body, mode, rank):
        manager = get_global_paras_memory_manager()
        if body._unified_weights_mode == mode:
            return
        source_mode = body._unified_weights_mode
        model = manager.benchmark_dimensions
        direction = "ep_to_tp" if mode == ParaSMode.TP else "tp_to_ep"
        group = get_paras_tp_group().device_group
        indices = (
            range(len(body.layers))
            if mode == ParaSMode.TP
            else reversed(range(len(body.layers)))
        )
        for index in indices:
            layer = body.layers[index]
            source = layer_parameters(layer, source_mode)
            if getattr(manager, "benchmark_method", "naive_nccl") == "host_reload":
                target = reload_host_layer(
                    manager.host_snapshots[index],
                    model,
                    manager.world_size,
                    rank,
                    direction,
                    device=source["w13"].device,
                    non_blocking=True,
                )
            else:
                target = naive_nccl_transfer_layer(
                    source, model, manager.world_size, rank, direction, group=group
                )
            for component, parameter in layer_parameters(layer, mode).items():
                parameter.data = target[component]
                manager.replace_weight(
                    weight_name(index, mode, component), target[component]
                )
            if getattr(manager, "benchmark_method", "naive_nccl") != "host_reload":
                dist.all_reduce(body._unified_fence, group=group)
            # PyTorch NCCL records tensor stream use and establishes current-
            # stream dependencies. Source/staging allocations can be released
            # after enqueueing without a device-wide host wait per layer.
            # H2D reload has no remote source reads and needs no rank fence.
            for component, parameter in source.items():
                name = weight_name(index, source_mode, component)
                placeholder = manager.placeholder(name)
                parameter.data = placeholder
                manager.replace_weight(name, placeholder)
            del source, target
        torch.cuda.synchronize()
        body._unified_weights_mode = mode

    ParaSModelMixin.paras_transfer_unified_weights = transfer

    # Dimensions are needed during initial model graph setup. The real plan
    # contains attention dimensions; expert dimensions come from model config.
    original_init = ParaSModelMixin.paras_init_peer_access

    def initialize(body, peer_access_ctx):
        original_init(body, peer_access_ctx)
        manager = get_global_paras_memory_manager()
        first = body.layers[0].mlp
        spec = manager._unified_spec
        manager.benchmark_dimensions = ModelConfig(
            "runtime",
            spec.num_kv_heads,
            spec.head_dim,
            first.num_global_experts,
            first.hidden_size,
            first.moe_intermediate_size,
            len(body.layers),
            num_attention_heads=spec.num_heads,
            interleaved_w13=first._paras_interleaved_w13,
        )

    ParaSModelMixin.paras_init_peer_access = initialize


class Runtime:
    def __init__(self, scheduler, config, method):
        self.scheduler, self.config, self.method = scheduler, config, method
        self.runner = scheduler.tp_worker.model_runner
        self.body = self.runner.model.model
        self.manager = self.runner.model.paras_memory_manager
        self.independent = isinstance(self.manager, IndependentWeightMemoryManager)
        self.phases = {}
        self.auxiliary_snapshots = []
        gr = self.runner.graph_runner
        # Preserve the lists actually resolved by production initialization,
        # including EP request-pool/alignment filtering and ParaS TP scaling.
        self.graph_batches = {
            ParaSMode.EP: list(gr.capture_bs),
            ParaSMode.TP: list(gr._paras_tp_capture_bs),
        }

    @property
    def mode(self):
        return self.scheduler.paras_parallelism_config

    def synchronize(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.scheduler.paras_tp_group.device_group)

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.phases[name] = (time.perf_counter() - start) * 1000

    def reset_requests(self):
        # The probe uses bench_one_batch's independent ScheduleBatch. It never
        # enters Scheduler.running_batch; reclaim all its pages explicitly.
        batches = (
            getattr(self.scheduler, name, None)
            for name in ("running_batch", "last_batch", "cur_batch")
        )
        if (
            any(batch is not None and batch.reqs for batch in batches)
            or self.scheduler.waiting_queue
            or getattr(self.scheduler, "chunked_req", None) is not None
            or getattr(self.scheduler, "result_queue", None)
        ):
            raise RuntimeError("Reconfiguration benchmark requires an empty scheduler")
        self.runner.req_to_token_pool.clear()
        self.runner.token_to_kv_pool_allocator.clear()
        self.scheduler.tree_cache.reset()

    def discard_graphs(self):
        from sglang.srt.model_executor.cuda_graph_runner import (
            set_global_graph_memory_pool,
        )

        gr = self.runner.graph_runner
        if gr is None:
            raise RuntimeError("Reconfiguration benchmark requires a CUDA graph runner")
        gr.graphs.clear()
        gr.output_buffers.clear()
        if hasattr(gr, "_paras_saved"):
            gr._paras_saved.clear()
        set_global_graph_memory_pool(None)

    def capture(self):
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
        from sglang.srt.paras.paras_cuda_graph import paras_refresh_cuda_graph_settings

        gr = self.runner.graph_runner
        gr.capture_bs = list(self.graph_batches[self.mode])
        paras_refresh_cuda_graph_settings(gr)
        with model_capture_mode():
            gr.capture()
        if sorted(gr.graphs) != sorted(gr.capture_bs):
            raise RuntimeError("Recapture did not produce the production graph set")

    def switch(self, target, *, measured=False):
        if target not in (ParaSMode.EP, ParaSMode.TP):
            raise ValueError(f"Unsupported target mode: {target}")
        if self.mode == target:
            return
        recapture = self.independent or (
            measured and self.method == "fixed_buffer_recapture"
        )
        if recapture:
            with self.phase("discard_graphs_ms"):
                self.discard_graphs()
        with self.phase("runtime_switch_ms"):
            if target == ParaSMode.TP:
                self.scheduler.paras_configure_tp()
            else:
                self.scheduler.paras_configure_ep()
        if self.mode != target:
            raise RuntimeError("Requested switch was rejected")
        if recapture:
            with self.phase("graph_capture_ms"):
                self.capture()

    @torch.inference_mode()
    def probe(self):
        from sglang.bench_one_batch import _maybe_prepare_mlp_sync_batch
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.reset_requests()
        prompt = self.config.get("probe_prompt", "The capital of France is")
        tokens = self.scheduler.tokenizer.encode(prompt)
        request = Req(
            rid="reconfigure-probe",
            origin_input_text=prompt,
            origin_input_ids=tokens,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
        )
        request.fill_ids = list(tokens)
        request.extend_input_len = len(tokens)
        request.logprob_start_len = len(tokens) - 1
        # The scheduler's real cache is required for hybrid/SWA allocation;
        # bench_one_batch.extend uses a dummy cache that cannot serve GPT-OSS.
        batch = ScheduleBatch.init_new(
            reqs=[request],
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, self.runner)
        forward = ForwardBatch.init_new(batch.get_model_worker_batch(), self.runner)
        output, _ = self.runner.forward(forward)
        ids = self.runner.sample(output, forward)
        outputs, graph_used = [output.next_token_logits.detach().float().cpu()], []
        for _ in range(self.config.get("probe_decode_steps", 2)):
            batch.output_ids = ids
            batch.prepare_for_decode()
            _maybe_prepare_mlp_sync_batch(batch, self.runner)
            forward = ForwardBatch.init_new(batch.get_model_worker_batch(), self.runner)
            output, used = self.runner.forward(forward)
            ids = self.runner.sample(output, forward)
            outputs.append(output.next_token_logits.detach().float().cpu())
            graph_used.append(bool(used))
        del batch, forward, request, output
        self.reset_requests()
        if not all(graph_used):
            raise RuntimeError("Decode probe did not execute captured CUDA graphs")
        if not all(torch.isfinite(x).all() for x in outputs):
            raise RuntimeError("Probe produced nonfinite logits")
        return outputs

    def prepare(self, source, target):
        self.switch(target)
        reference = self.probe()
        if self.method == "host_reload":
            pin = self.config.get("host_pin_memory", True)
            self.manager.host_snapshots = {
                index: snapshot_host_tensors(
                    layer_parameters(layer, target), pin_memory=pin
                )
                for index, layer in enumerate(self.body.layers)
            }
            # Common auxiliary parameters (norms, embeddings, biases, etc.)
            # remain at stable addresses and are reloaded in place. Their H2D
            # copies are included; the large redistributed tensors allocate anew.
            weight_ids = {
                id(p)
                for layer in self.body.layers
                for mode in (ParaSMode.EP, ParaSMode.TP)
                for p in layer_parameters(layer, mode).values()
            }
            for parameter in self.runner.model.parameters():
                if id(parameter) not in weight_ids:
                    snapshot = snapshot_host_tensors(
                        {"tensor": parameter}, pin_memory=pin
                    )["tensor"]
                    self.auxiliary_snapshots.append((parameter, snapshot))
        self.switch(source)
        self.probe()
        self.phases.clear()
        self.synchronize()
        return reference

    def execute(self, target, reference):
        self.reset_requests()
        self.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if self.method == "host_reload":
            self.manager.benchmark_method = "host_reload"
            with self.phase("auxiliary_reload_ms"):
                for parameter, snapshot in self.auxiliary_snapshots:
                    parameter.data.copy_(snapshot, non_blocking=True)
        original_transfer = self.body.paras_transfer_unified_weights

        def timed_transfer(mode, rank):
            if self.body._unified_weights_mode == mode:
                return original_transfer(mode, rank)
            with self.phase("weight_transfer_ms"):
                return original_transfer(mode, rank)

        self.body.paras_transfer_unified_weights = timed_transfer
        try:
            self.switch(target, measured=True)
        finally:
            self.body.paras_transfer_unified_weights = original_transfer
        self.synchronize()
        switch_ms = (time.perf_counter() - start) * 1000
        actual = self.probe()
        self.synchronize()
        ready_ms = (time.perf_counter() - start) * 1000
        error = max(float((a - b).abs().max()) for a, b in zip(actual, reference))
        for a, b in zip(actual, reference):
            torch.testing.assert_close(
                a,
                b,
                atol=self.config.get("logits_atol", 0.125),
                rtol=self.config.get("logits_rtol", 0.02),
            )
        return {
            "switch_ms": switch_ms,
            "through_probe_ms": ready_ms,
            "phases": self.phases,
            "max_logit_error": error,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_snapshot_bytes": sum(
                t.numel() * t.element_size()
                for layer in getattr(self.manager, "host_snapshots", {}).values()
                for t in layer.values()
            )
            + sum(t.numel() * t.element_size() for _, t in self.auxiliary_snapshots),
            "validation": "passed",
            "weight_transport": METHOD_TRANSPORT[self.method],
            "graph_batch_sizes": list(self.runner.graph_runner.capture_bs),
            "scope": "scheduler_worker_reconfiguration_empty_requests",
        }
