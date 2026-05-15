"""T29: kv_event emission during tree migration."""
import pytest


class _FakeAllBlocksCleared:
    def __repr__(self):
        return "AllBlocksCleared()"


class _FakeBlockStored:
    def __init__(
        self,
        token_ids=None,
        block_hashes=None,
        parent_block_hash=None,
        block_size=None,
        lora_id=None,
    ):
        self.token_ids = token_ids
        self.block_hashes = block_hashes
        self.parent_block_hash = parent_block_hash
        self.block_size = block_size
        self.lora_id = lora_id


def _build_fake_tree(enable_events=True):
    class FakeKey:
        def __init__(self, t):
            self.token_ids = list(t)

    class FakeNode:
        def __init__(self, k=None):
            self.key = k
            self.children = {}
            self.parent = None
            self.value = [10, 20]

    class FakeTree:
        def __init__(self):
            self.root_node = FakeNode(FakeKey([]))
            self.kv_event_queue = []
            self.enable_kv_cache_events = enable_events
            self.page_size = 1

    t = FakeTree()
    a = FakeNode(FakeKey([1, 2]))
    a.parent = t.root_node
    a.value = [10, 20]
    t.root_node.children[1] = a
    b = FakeNode(FakeKey([3]))
    b.parent = a
    b.value = [30]
    a.children[3] = b
    return t


def test_no_events_when_disabled():
    from sglang.srt.paras.tree_migration import emit_migration_events

    t = _build_fake_tree(enable_events=False)
    emit_migration_events(t)
    assert t.kv_event_queue == []


def test_no_events_when_event_types_unavailable(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "sglang.srt.disaggregation.kv_events", raising=False)
    monkeypatch.delitem(sys.modules, "sglang.srt.mem_cache.radix_cache", raising=False)
    monkeypatch.delitem(sys.modules, "sglang.srt.mem_cache.swa_radix_cache", raising=False)

    import types

    # Provide stub modules without the event classes so the import succeeds but
    # the getattr lookups for AllBlocksCleared/BlockStored return None.
    for mod_path in (
        "sglang.srt.disaggregation.kv_events",
        "sglang.srt.mem_cache.radix_cache",
        "sglang.srt.mem_cache.swa_radix_cache",
    ):
        stub = types.ModuleType(mod_path)
        monkeypatch.setitem(sys.modules, mod_path, stub)

    from sglang.srt.paras.tree_migration import emit_migration_events

    t = _build_fake_tree(enable_events=True)
    emit_migration_events(t)
    assert t.kv_event_queue == []


def test_events_emitted_with_fake_types(monkeypatch):
    """Inject fake event types via the kv_events module path."""
    import sys
    import types

    fake_mod = types.ModuleType("sglang.srt.disaggregation.kv_events")
    fake_mod.AllBlocksCleared = _FakeAllBlocksCleared
    fake_mod.BlockStored = _FakeBlockStored
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.kv_events", fake_mod)

    from sglang.srt.paras.tree_migration import emit_migration_events

    t = _build_fake_tree(enable_events=True)
    emit_migration_events(t)

    assert len(t.kv_event_queue) >= 1
    assert "Cleared" in repr(t.kv_event_queue[0])

    block_events = t.kv_event_queue[1:]
    assert len(block_events) >= 2
    for ev in block_events:
        assert isinstance(ev, _FakeBlockStored)
        assert ev.token_ids is not None and len(ev.token_ids) > 0


def test_event_sequence_order(monkeypatch):
    """AllBlocksCleared must precede every BlockStored."""
    import sys
    import types

    fake_mod = types.ModuleType("sglang.srt.disaggregation.kv_events")
    fake_mod.AllBlocksCleared = _FakeAllBlocksCleared
    fake_mod.BlockStored = _FakeBlockStored
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.kv_events", fake_mod)

    from sglang.srt.paras.tree_migration import emit_migration_events

    t = _build_fake_tree(enable_events=True)
    emit_migration_events(t)

    cleared_seen = False
    for ev in t.kv_event_queue:
        if isinstance(ev, _FakeAllBlocksCleared):
            assert not cleared_seen, "AllBlocksCleared emitted more than once"
            cleared_seen = True
        elif isinstance(ev, _FakeBlockStored):
            assert cleared_seen, "BlockStored emitted before AllBlocksCleared"


def test_empty_tree_emits_only_cleared(monkeypatch):
    """A tree with only the root node emits AllBlocksCleared and no BlockStored."""
    import sys
    import types

    fake_mod = types.ModuleType("sglang.srt.disaggregation.kv_events")
    fake_mod.AllBlocksCleared = _FakeAllBlocksCleared
    fake_mod.BlockStored = _FakeBlockStored
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.kv_events", fake_mod)

    from sglang.srt.paras.tree_migration import emit_migration_events

    class FakeKey:
        def __init__(self, t):
            self.token_ids = list(t)

    class FakeNode:
        def __init__(self, k=None):
            self.key = k
            self.children = {}
            self.parent = None
            self.value = None

    class FakeTree:
        def __init__(self):
            self.root_node = FakeNode(FakeKey([]))
            self.kv_event_queue = []
            self.enable_kv_cache_events = True
            self.page_size = 1

    t = FakeTree()
    emit_migration_events(t)
    assert len(t.kv_event_queue) == 1
    assert isinstance(t.kv_event_queue[0], _FakeAllBlocksCleared)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
