"""T17: recover_request attaches migrated reqs to migrated tree nodes."""
import sys
import types
import pathlib

import pytest
import torch


def _stub_radix_key_module():
    """Provide a lightweight RadixKey so the function can do its inner import."""
    mod_name = "sglang.srt.mem_cache.radix_cache"
    if mod_name in sys.modules and hasattr(sys.modules[mod_name], "RadixKey"):
        return

    pkg = sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    pkg.__path__ = []
    srt = sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
    srt.__path__ = []
    mem = sys.modules.setdefault("sglang.srt.mem_cache", types.ModuleType("sglang.srt.mem_cache"))
    mem.__path__ = []
    rc = sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    class _RadixKey:
        def __init__(self, token_ids, extra_key=None):
            self.token_ids = token_ids
            self.extra_key = extra_key

    rc.RadixKey = _RadixKey


def _load_recover_request():
    """Extract recover_request by parsing the source file and exec()ing only that function.

    Avoids importing the full sglang chain (transformers, configs, etc.) which is
    not available in the lightweight test environment.
    """
    src_path = pathlib.Path(__file__).parents[4] / "python" / "sglang" / "srt" / "paras" / "gather_manager.py"
    text = src_path.read_text()
    start = text.index("def recover_request(")
    after = text[start:]
    next_top_def = after.index("\ndef ", 1)
    func_src = after[: next_top_def]
    _stub_radix_key_module()
    from typing import Any
    ns = {"Req": object, "BasePrefixCache": object, "Any": Any}
    exec(func_src, ns)
    return ns["recover_request"]


class _MockReq:
    def __init__(self, fill_ids, extra_key=None):
        self.fill_ids = fill_ids
        self.extra_key = extra_key
        self.tokenizer = None
        self.tree_orphaned = False
        self.last_node = None
        self.last_host_node = None
        self.prefix_indices = None


class _MockMatch:
    def __init__(self, device_indices=None, last_device_node=None, last_host_node=None):
        self.device_indices = device_indices
        self.last_device_node = last_device_node
        self.last_host_node = last_host_node


class _MockTreeCache:
    def __init__(self, has_match=True, disable=False):
        self.disable = disable
        self.root_node = object()
        self._has_match = has_match
        self._match_node = object() if has_match else None
        self._match_indices = torch.tensor([10, 20, 30]) if has_match else torch.tensor([])

    def match_prefix(self, key):
        if self._has_match:
            return _MockMatch(
                device_indices=self._match_indices,
                last_device_node=self._match_node,
                last_host_node=self._match_node,
            )
        return _MockMatch(
            device_indices=torch.tensor([]),
            last_device_node=None,
            last_host_node=None,
        )


@pytest.fixture(scope="module")
def recover_request():
    return _load_recover_request()


class TestRecoverRequestGather:
    def test_attaches_to_match_when_present(self, recover_request):
        tree = _MockTreeCache(has_match=True)
        req = _MockReq(fill_ids=[1, 2, 3, 4, 5])
        recover_request(req, tree, tokenizer="dummy")
        assert req.last_node is tree._match_node
        assert req.tree_orphaned is False
        assert req.prefix_indices is not None
        assert len(req.prefix_indices) == 3

    def test_orphans_when_no_match(self, recover_request):
        tree = _MockTreeCache(has_match=False)
        req = _MockReq(fill_ids=[7, 8, 9])
        recover_request(req, tree, tokenizer="dummy")
        assert req.last_node is tree.root_node
        assert req.tree_orphaned is True
        assert len(req.prefix_indices) == 0

    def test_orphans_when_chunk_cache(self, recover_request):
        tree = _MockTreeCache(has_match=True, disable=True)
        req = _MockReq(fill_ids=[1, 2])
        recover_request(req, tree, tokenizer="dummy")
        assert req.tree_orphaned is True

    def test_tokenizer_restored(self, recover_request):
        tree = _MockTreeCache(has_match=True)
        req = _MockReq(fill_ids=[1, 2])
        recover_request(req, tree, tokenizer="my-tokenizer")
        assert req.tokenizer == "my-tokenizer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
