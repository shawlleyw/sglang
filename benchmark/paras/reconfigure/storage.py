"""Benchmark-only independently allocated weights, with retained empty KV pools.

The production UMM planner determines capacities, but its weight arena is never
allocated here. This lets a naive switch reclaim each source layer's allocation.
Inactive weight views are zero-stride placeholders and must never execute.
"""

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
from sglang.srt.paras.workspace import workspace_views


class IndependentWeightMemoryManager(ParaSMemoryManager):
    def materialize(self, plan):
        self._plan = plan
        self._unified_layout = plan.layout
        self._entries = {entry.name: entry for entry in plan.entries}
        self.ep_max_kv_tokens = plan.layout.ep_cache.full_tokens
        self.tp_max_kv_tokens = plan.layout.tp_cache.full_tokens
        swa = next((s for s in plan.layer_specs if s.kind == "swa"), None)
        self.ep_max_kv_tokens_swa = swa.tokens_cap_ep if swa else 0
        self.tp_max_kv_tokens_swa = swa.tokens_cap_tp if swa else 0
        self._weights = {}
        self._aliases = {}
        self._cache_views = {}
        self._cache_buffers = []
        self._workspaces = {}
        for mode in (ParaSMode.EP, ParaSMode.TP):
            for names in self._unified_spec.for_mode(mode).weight_names:
                for name in names:
                    entry = self._entries[name]
                    self._aliases[name] = name
                    self._weights[name] = (
                        torch.empty(entry.shape, dtype=entry.dtype, device=self.device)
                        if mode == ParaSMode.EP
                        else self.placeholder(name)
                    )
        for name, entry in self._entries.items():
            if name in self._aliases or ".kv." in name:
                continue
            # Checkpoint names alias EP experts. Equal UMM spans are NOT an
            # alias contract: distinct EP/TP layers can intentionally occupy
            # exactly the same original address and byte extent.
            canonical = name.replace(".mlp.experts.", ".mlp.ep_experts.")
            if canonical == name or canonical not in self._weights:
                raise ValueError(f"Unknown independent-storage weight alias: {name}")
            target = self._entries[canonical]
            if entry.shape != target.shape or entry.dtype != target.dtype:
                raise ValueError(f"Incompatible weight alias: {name} -> {canonical}")
            self._aliases[name] = canonical
        cache_groups = {}
        for name, entry in self._entries.items():
            if ".kv." in name:
                prefix, suffix = name.split(".kv.")
                kind = suffix.split(".")[-1]
                cache_groups.setdefault((prefix, kind), []).append(entry)
        for entries in cache_groups.values():
            buffer = torch.empty(
                max(e.size_bytes for e in entries),
                dtype=torch.uint8,
                device=self.device,
            )
            self._cache_buffers.append(buffer)
            for entry in entries:
                self._cache_views[entry.name] = (
                    buffer[: entry.size_bytes].view(entry.dtype).view(entry.shape)
                )
        for mode in (ParaSMode.EP, ParaSMode.TP):
            for kind in ("moe", "attention"):
                _, size = self._unified_spec.for_mode(mode).workspaces.region(kind)
                self._workspaces[mode, kind] = torch.empty(
                    size, dtype=torch.uint8, device=self.device
                )
        # Startup may export an IPC handle. No weight/cache transfer uses this
        # anchor; the benchmark routes weights through NCCL and has no live KV.
        self._buffer = torch.empty(256, dtype=torch.uint8, device=self.device)
        self._buffer_start = self._buffer.data_ptr()
        self._buffer_end = self._buffer_start + self._buffer.numel()
        self._total_bytes = self.allocated_bytes()
        self._materialized = True
        return self._total_bytes

    def placeholder(self, name):
        entry = self._entries[name]
        return torch.empty(1, dtype=entry.dtype, device=self.device).expand(entry.shape)

    def allocated_bytes(self):
        tensors = (
            list(self._weights.values())
            + self._cache_buffers
            + list(self._workspaces.values())
        )
        return sum(t.untyped_storage().nbytes() for t in tensors) + 256

    def weight_storage_report(self, active_mode):
        """Check endpoint ownership without allocating, copying or collecting."""
        report = {}
        for mode in (ParaSMode.EP, ParaSMode.TP):
            physical_bytes = logical_bytes = 0
            for names in self._unified_spec.for_mode(mode).weight_names:
                for name in names:
                    tensor = self.get_view(name)
                    entry = self._entries[name]
                    size = tensor.untyped_storage().nbytes()
                    if mode == active_mode:
                        if not tensor.is_contiguous() or size < entry.size_bytes:
                            raise RuntimeError(
                                f"Active weight is not materialized: {name}"
                            )
                    elif size > tensor.element_size():
                        raise RuntimeError(
                            f"Inactive weight still owns storage: {name}"
                        )
                    physical_bytes += size
                    logical_bytes += entry.size_bytes
            report[mode.value] = {
                "active": mode == active_mode,
                "logical_bytes": logical_bytes,
                "backing_bytes": physical_bytes,
            }
        return report

    def get_view(self, name):
        if not self._materialized:
            raise RuntimeError("Buffer not materialized yet.")
        if name in self._aliases:
            return self._weights[self._aliases[name]]
        if name in self._cache_views:
            return self._cache_views[name]
        raise KeyError(f"Unsupported independent-storage view: {name}")

    def get_view_as(self, name, shape, dtype=None):
        tensor = self.get_view(name)
        if dtype is not None and dtype != tensor.dtype:
            tensor = tensor.view(dtype)
        return tensor.reshape(shape)

    def _get_workspace(self, mode, kind, shapes, dtype, device):
        return workspace_views(
            self._workspaces[mode, kind],
            shapes,
            dtype,
            device,
            label=f"benchmark {mode.value} {kind}",
        )

    def replace_weight(self, name, tensor):
        if name not in self._aliases:
            raise KeyError(f"Not an independent weight view: {name}")
        entry = self._entries[name]
        if tensor.shape != entry.shape or tensor.dtype != entry.dtype:
            raise ValueError(f"Replacement shape/dtype mismatch for {name}")
        if tensor.device != self._buffer.device:
            raise ValueError(f"Replacement device mismatch for {name}")
        canonical = self._aliases[name]
        previous = self._weights[canonical]
        self._weights[canonical] = tensor
        self._total_bytes += (
            tensor.untyped_storage().nbytes() - previous.untyped_storage().nbytes()
        )

    def is_managed(self, tensor):
        if not self._materialized:
            return False
        allocations = (
            list(self._weights.values())
            + self._cache_buffers
            + list(self._workspaces.values())
            + [self._buffer]
        )
        return any(
            tensor.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            and value.untyped_storage().nbytes() > 0
            for value in allocations
        )
