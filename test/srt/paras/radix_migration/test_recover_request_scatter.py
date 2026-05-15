"""T18: scatter_manager.recover_request — attach migrated reqs to migrated tree nodes."""
import importlib
import importlib.util
import os
import sys
import types

import pytest
import torch


def _import_scatter_manager():
    mod_name = "sglang.srt.paras.scatter_manager"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    stubs_needed = [
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.mem_cache.base_prefix_cache",
        "sglang.srt.distributed.parallel_state",
        "sglang.srt.paras.paras_memory_manager",
        "sglang.srt.paras.peer_access",
        "sglang.srt.paras.layers.utils",
        "sglang.srt.paras.cache_transfer.mha",
        "sglang.srt.paras.cache_transfer.swa",
        "sglang.srt.paras.gather_manager",
    ]
    for name in stubs_needed:
        stub = types.ModuleType(name)
        stub.__dict__.setdefault("Req", type("Req", (), {}))
        stub.__dict__.setdefault("ReqToTokenPool", type("ReqToTokenPool", (), {}))
        stub.__dict__.setdefault("MHATokenToKVPool", type("MHATokenToKVPool", (), {}))
        stub.__dict__.setdefault("SWAKVPool", type("SWAKVPool", (), {}))
        stub.__dict__.setdefault("TokenToKVPoolAllocator", type("TokenToKVPoolAllocator", (), {}))
        stub.__dict__.setdefault("SWATokenToKVPoolAllocator", type("SWATokenToKVPoolAllocator", (), {}))
        stub.__dict__.setdefault("GroupCoordinator", type("GroupCoordinator", (), {}))
        stub.__dict__.setdefault("BasePrefixCache", type("BasePrefixCache", (), {}))
        stub.__dict__.setdefault("LayerCacheSpec", type("LayerCacheSpec", (), {}))
        stub.__dict__.setdefault("MHACacheTransfer", type("MHACacheTransfer", (), {}))
        stub.__dict__.setdefault("SWACacheTransfer", type("SWACacheTransfer", (), {}))
        stub.__dict__.setdefault("get_global_paras_memory_manager", lambda *a, **kw: None)
        stub.__dict__.setdefault("paras_tp_group_all_gather_reqs", lambda *a, **kw: (None, None))
        sys.modules[name] = stub

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    python_root = os.path.join(repo_root, "python")
    pkg_paths = {
        "sglang": os.path.join(python_root, "sglang"),
        "sglang.srt": os.path.join(python_root, "sglang", "srt"),
        "sglang.srt.paras": os.path.join(python_root, "sglang", "srt", "paras"),
        "sglang.srt.managers": os.path.join(python_root, "sglang", "srt", "managers"),
        "sglang.srt.mem_cache": os.path.join(python_root, "sglang", "srt", "mem_cache"),
        "sglang.srt.distributed": os.path.join(python_root, "sglang", "srt", "distributed"),
        "sglang.srt.model_executor": os.path.join(python_root, "sglang", "srt", "model_executor"),
    }
    for pkg, path in pkg_paths.items():
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [path]
            m.__package__ = pkg
            sys.modules[pkg] = m

    scatter_path = os.path.join(python_root, "sglang", "srt", "paras", "scatter_manager.py")
    spec = importlib.util.spec_from_file_location(mod_name, scatter_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _RadixKeyStub:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = token_ids
        self.extra_key = extra_key


def _install_radix_key_stub():
    name = "sglang.srt.mem_cache.radix_cache"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.RadixKey = _RadixKeyStub
        sys.modules[name] = stub


_install_radix_key_stub()
_scatter_mod = _import_scatter_manager()
recover_request = _scatter_mod.recover_request


class _MatchResult:
    def __init__(self, device_indices, last_device_node, last_host_node=None):
        self.device_indices = device_indices
        self.last_device_node = last_device_node
        self.last_host_node = last_host_node if last_host_node is not None else last_device_node


class _MatchingTreeCache:
    def __init__(self, match_indices, last_node_sentinel=None):
        self.disable = False
        self.root_node = object()
        self._match_indices = match_indices
        self._last_node = last_node_sentinel or object()

    def match_prefix(self, key):
        return _MatchResult(self._match_indices, self._last_node)


class _NoMatchTreeCache:
    def __init__(self):
        self.disable = False
        self.root_node = object()

    def match_prefix(self, key):
        return _MatchResult(torch.tensor([], dtype=torch.int64), None, None)


class _DisabledTreeCache:
    def __init__(self):
        self.disable = True
        self.root_node = None


class _ChunkCacheLike:
    """Mimics ChunkCache: no root_node attribute (falls back to orphan branch)."""

    def __init__(self):
        self.disable = False
        self.root_node = None


class _Req:
    def __init__(self, fill_ids=None, extra_key=None):
        self.fill_ids = fill_ids if fill_ids is not None else [1, 2, 3]
        self.extra_key = extra_key
        self.last_node = "unset"
        self.last_host_node = "unset"
        self.prefix_indices = "unset"
        self.tokenizer = None
        self.tree_orphaned = False


class TestRecoverRequestScatter:
    def test_scatter_attaches_when_match_present(self):
        last_node_sentinel = object()
        cache = _MatchingTreeCache(
            match_indices=torch.tensor([10, 20], dtype=torch.int64),
            last_node_sentinel=last_node_sentinel,
        )
        req = _Req(fill_ids=[1, 2, 3])

        recover_request(req, cache, tokenizer="dummy-tokenizer")

        assert req.tokenizer == "dummy-tokenizer"
        assert req.tree_orphaned is False
        assert req.last_node is last_node_sentinel
        assert req.last_host_node is last_node_sentinel
        assert torch.equal(req.prefix_indices, torch.tensor([10, 20], dtype=torch.int64))

    def test_scatter_orphans_when_no_match(self):
        cache = _NoMatchTreeCache()
        req = _Req(fill_ids=[7, 8])

        recover_request(req, cache, tokenizer="dummy")

        assert req.tree_orphaned is True
        assert req.last_node is cache.root_node
        assert req.last_host_node is cache.root_node
        assert req.prefix_indices == []

    def test_scatter_orphans_when_tree_disabled(self):
        cache = _DisabledTreeCache()
        req = _Req(fill_ids=[5, 6])

        recover_request(req, cache, tokenizer="x")

        assert req.tree_orphaned is True
        assert req.last_node is None
        assert req.last_host_node is None
        assert req.prefix_indices == []

    def test_scatter_orphans_when_no_root_node(self):
        cache = _ChunkCacheLike()
        req = _Req(fill_ids=[1])

        recover_request(req, cache, tokenizer="x")

        assert req.tree_orphaned is True
        assert req.last_node is None
        assert req.last_host_node is None
        assert req.prefix_indices == []

    def test_scatter_tokenizer_set_regardless_of_branch(self):
        for cache in (_NoMatchTreeCache(), _DisabledTreeCache(), _ChunkCacheLike()):
            req = _Req(fill_ids=[1, 2])
            recover_request(req, cache, tokenizer="t-set")
            assert req.tokenizer == "t-set"

    def test_scatter_match_with_explicit_host_node(self):
        device_node = object()
        host_node = object()

        class _CacheHostNode:
            disable = False
            root_node = object()
            def match_prefix(self, key):
                return _MatchResult(torch.tensor([5], dtype=torch.int64), device_node, host_node)

        req = _Req(fill_ids=[1])
        recover_request(req, _CacheHostNode(), tokenizer="x")

        assert req.tree_orphaned is False
        assert req.last_node is device_node
        assert req.last_host_node is host_node


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
