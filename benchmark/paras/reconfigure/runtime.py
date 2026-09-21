"""Benchmark adapters around real Scheduler/ModelRunner state transitions.

Patches are installed only in fresh benchmark worker processes. Normal serving
imports no benchmark code. Each trial executes one switch with no live requests.
"""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

from common.model_configs import ModelConfig
from common.weight_bundle import weight_name
from sglang.srt.paras.mode import ParaSMode

from .configuration import HOST_METHODS, METHOD_TRANSPORT
from .diagnostics import (
    allocator_configuration,
    allocator_delta,
    kv_reservation,
    memory_snapshot,
)
from .storage import IndependentWeightMemoryManager
from .runtime_memory import runtime_memory_report
from .transfers import (
    naive_nccl_transfer_layer,
    reload_host_layer,
    reload_host_model,
    snapshot_host_tensors,
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


def active_auxiliary_parameters(model, bulk_ids):
    """Visit active module aliases, excluding saved EP/TP representations.

    remove_duplicate=False matters: a saved representation can be registered
    before the active alias. Deduplicate exact tensor views after filtering.
    """
    saved = {
        "ep_experts",
        "tp_experts",
        "ep_sinks",
        "tp_sinks",
        "ep_qkv_bias",
        "tp_qkv_bias",
    }
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in bulk_ids or saved.intersection(name.split(".")):
            continue
        key = (
            parameter.device,
            parameter.data_ptr(),
            parameter.dtype,
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        if key not in seen:
            seen.add(key)
            yield parameter


def release_layer_weights(manager, layer, index, mode):
    """Drop both module and manager references to a source layer's storage."""
    for component, parameter in layer_parameters(layer, mode).items():
        name = weight_name(index, mode, component)
        placeholder = manager.placeholder(name)
        parameter.data = placeholder
        manager.replace_weight(name, placeholder)


def bind_layer_weights(manager, layer, index, mode, target):
    for component, parameter in layer_parameters(layer, mode).items():
        parameter.data = target[component]
        manager.replace_weight(weight_name(index, mode, component), target[component])


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
        method = getattr(manager, "benchmark_method", "naive_nccl")
        if method == "host_model_to":
            device = layer_parameters(body.layers[0], source_mode)["w13"].device
            for index, layer in enumerate(body.layers):
                release_layer_weights(manager, layer, index, source_mode)
            target = reload_host_model(
                manager.host_snapshots,
                model,
                manager.world_size,
                rank,
                direction,
                device=device,
            )
            for index, layer in enumerate(body.layers):
                bind_layer_weights(manager, layer, index, mode, target[index])
            del target
            torch.cuda.synchronize()
            body._unified_weights_mode = mode
            return
        group = get_paras_tp_group().device_group
        host_reload = method == "host_reload"
        indices = (
            range(len(body.layers))
            if mode == ParaSMode.TP
            else reversed(range(len(body.layers)))
        )
        for index in indices:
            layer = body.layers[index]
            source = layer_parameters(layer, source_mode)
            if host_reload:
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
            bind_layer_weights(manager, layer, index, mode, target)
            if not host_reload:
                dist.all_reduce(body._unified_fence, group=group)
            # PyTorch NCCL records tensor stream use and establishes current-
            # stream dependencies. Source/staging allocations can be released
            # after enqueueing without a device-wide host wait per layer.
            # H2D reload has no remote source reads and needs no rank fence.
            # Keep this allocation-before-release order for both baselines:
            # target and source weights coexist for the layer being replaced.
            release_layer_weights(manager, layer, index, source_mode)
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

    def vmm_report(self):
        return runtime_memory_report(
            self.runner.graph_runner,
            self.mode,
            enabled=self.config.get("server_args", {}).get(
                "paras_vmm_runtime_states", False
            ),
        )

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        yield
        # Host wall time, including any production waits. Do not insert extra
        # device barriers inside the switch just to obtain phase timings.
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
        # Backends also own graph metadata outside the graph runner. Drop saved
        # references so recapture cannot retain old mode buffers/wrappers.
        backend = self.runner.attn_backend
        for name in (
            "_paras_graph_states",  # Triton
            "decode_cuda_graph_metadata",  # FlashInfer
            "prefill_cuda_graph_metadata",
            "draft_extend_cuda_graph_metadata",
        ):
            if hasattr(backend, name):
                getattr(backend, name).clear()
        set_global_graph_memory_pool(None)

    def capture(self):
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
        from sglang.srt.paras.paras_cuda_graph import paras_refresh_cuda_graph_settings

        gr = self.runner.graph_runner
        gr.capture_bs = list(self.graph_batches[self.mode])
        # Production now allocates graph inputs and backend metadata per mode.
        # EP buffers cannot capture TP's larger batch range. Follow dual-capture
        # initialization, with all target buffer allocation inside this phase.
        gr.max_bs = max(gr.capture_bs)
        gr.max_num_token = gr.max_bs * gr.num_tokens_per_bs
        paras_refresh_cuda_graph_settings(gr)
        self.runner.attn_backend.init_cuda_graph_state(gr.max_bs, gr.max_num_token)
        gr.init_graph_buffers()
        with model_capture_mode():
            gr.capture()
        if sorted(gr.graphs) != sorted(gr.capture_bs):
            raise RuntimeError("Recapture did not produce the production graph set")

    def graph_state_report(self, *, require_discarded=False):
        """Validate graph coverage without replaying or retaining graph objects."""
        gr = self.runner.graph_runner
        expected = self.graph_batches[self.mode]
        if list(gr.capture_bs) != expected or sorted(gr.graphs) != sorted(expected):
            raise RuntimeError("Active graphs differ from the resolved production set")
        if gr.max_bs != max(expected) or gr.max_num_token != (
            max(expected) * gr.num_tokens_per_bs
        ):
            raise RuntimeError("Active CUDA graph replay limits do not match the mode")
        saved = getattr(gr, "_paras_saved", {})
        retained = {
            mode.value: sorted(state["graphs"]) for mode, state in saved.items()
        }
        if self.method == "full":
            expected_retained = {
                mode.value: sorted(sizes) for mode, sizes in self.graph_batches.items()
            }
            if retained != expected_retained:
                raise RuntimeError("Full method must retain both production graph sets")
        elif saved and (self.independent or require_discarded):
            raise RuntimeError("Non-retaining method still owns saved mode graphs")
        return {
            "mode": self.mode.value,
            "active_batch_sizes": list(gr.capture_bs),
            "max_batch_size": gr.max_bs,
            "max_num_tokens": gr.max_num_token,
            "retained_batch_sizes_by_mode": retained,
        }

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
                # Discarding saved graphs bypasses production's graph-state
                # load (which normally activates VMM). Complete the same drained
                # transition before allocating/capturing any target buffers.
                memory = getattr(
                    self.runner.graph_runner, "_paras_runtime_memory", None
                )
                if memory is not None:
                    memory.activate(target)
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
        # Same global workload in both modes: one request per EP rank, or
        # world_size requests in the single TP replica.
        count = self.config["world_size"] if self.mode == ParaSMode.TP else 1
        requests = []
        for index in range(count):
            request = Req(
                rid=f"reconfigure-probe-{index}",
                origin_input_text=prompt,
                origin_input_ids=list(tokens),
                sampling_params=SamplingParams(
                    temperature=0,
                    max_new_tokens=self.config.get("probe_decode_steps", 2) + 1,
                    ignore_eos=True,
                ),
            )
            request.fill_ids = list(tokens)
            request.extend_input_len = len(tokens)
            request.logprob_start_len = len(tokens) - 1
            requests.append(request)
        # The scheduler's real cache is required for hybrid/SWA allocation;
        # bench_one_batch.extend uses a dummy cache that cannot serve GPT-OSS.
        batch = ScheduleBatch.init_new(
            reqs=requests,
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
        del batch, forward, request, requests, output
        self.reset_requests()
        if not all(graph_used):
            raise RuntimeError("Decode probe did not execute captured CUDA graphs")
        if not all(torch.isfinite(x).all() for x in outputs):
            raise RuntimeError("Probe produced nonfinite logits")
        return outputs

    def prepare(self, source, target):
        self.switch(target)
        reference = self.probe()
        if self.method in HOST_METHODS:
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
            for parameter in active_auxiliary_parameters(self.runner.model, weight_ids):
                snapshot = snapshot_host_tensors({"tensor": parameter}, pin_memory=pin)[
                    "tensor"
                ]
                self.auxiliary_snapshots.append((parameter, snapshot))
        self.switch(source)
        self.probe()
        # Exercise the actual measured transport/recapture path, including H2D
        # for host reload. Restoring source uses NCCL because the prepared host
        # snapshot deliberately contains only the target representation.
        for _ in range(self.config.get("warmup_switches", 1)):
            self.execute(target, reference)
            if self.method in HOST_METHODS:
                self.manager.benchmark_method = "naive_nccl"
            self.switch(source, measured=True)
            self.probe()
        self.phases.clear()
        self.synchronize()
        return reference

    def execute(self, target, reference):
        self.reset_requests()
        self.synchronize()
        self.phases.clear()
        torch.cuda.reset_peak_memory_stats()
        memory_before = memory_snapshot()
        vmm_before = self.vmm_report()
        start = time.perf_counter()
        if self.method in HOST_METHODS:
            self.manager.benchmark_method = self.method
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
        memory_after_switch = memory_snapshot()
        vmm_after_switch = self.vmm_report()
        torch.cuda.reset_peak_memory_stats()
        probe_start = time.perf_counter()
        actual = self.probe()
        self.synchronize()
        # Exclude diagnostic memory queries between switch and probe.
        ready_ms = switch_ms + (time.perf_counter() - probe_start) * 1000
        memory_after_probe = memory_snapshot()
        vmm_after_probe = self.vmm_report()
        graph_state = self.graph_state_report(require_discarded=True)
        if len(actual) != len(reference):
            raise RuntimeError("Probe and reference have different numbers of steps")
        if any(a.shape != b.shape for a, b in zip(actual, reference)):
            raise RuntimeError("Probe and reference have different logit shapes")
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
            "phases": dict(self.phases),
            "max_logit_error": error,
            "phase_timing": "host_wall_time_no_added_internal_cuda_sync",
            "vmm": "on" if vmm_after_switch["enabled"] else "off",
            "warmup_switches": self.config.get("warmup_switches", 1),
            "probe_global_requests": self.config["world_size"],
            "peak_allocated_bytes": max(
                memory_after_switch["peak_allocated_bytes"],
                memory_after_probe["peak_allocated_bytes"],
            ),
            "peak_reserved_bytes": max(
                memory_after_switch["peak_reserved_bytes"],
                memory_after_probe["peak_reserved_bytes"],
            ),
            "memory": {
                "before_switch": memory_before,
                "after_switch": memory_after_switch,
                "after_probe": memory_after_probe,
                "runtime_vmm": {
                    "before_switch": vmm_before,
                    "after_switch": vmm_after_switch,
                    "after_probe": vmm_after_probe,
                },
                "switch_allocator_delta": allocator_delta(
                    memory_before["allocator_counters"],
                    memory_after_switch["allocator_counters"],
                ),
                "kv_reservation": kv_reservation(self.manager),
                "allocator": allocator_configuration(),
                "independent_weight_storage": (
                    self.manager.weight_storage_report(self.mode)
                    if self.independent
                    else None
                ),
            },
            "host_snapshot_bytes": sum(
                t.numel() * t.element_size()
                for layer in getattr(self.manager, "host_snapshots", {}).values()
                for t in layer.values()
            )
            + sum(t.numel() * t.element_size() for _, t in self.auxiliary_snapshots),
            "validation": "passed",
            "weight_transport": METHOD_TRANSPORT[self.method],
            "graph_batch_sizes": list(self.runner.graph_runner.capture_bs),
            "graph_state": graph_state,
            "scope": "scheduler_worker_reconfiguration_empty_requests",
        }
