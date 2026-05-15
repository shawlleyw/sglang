"""Tree migration utilities for ParaS radix-cache.

Records-based serialize/rebuild to migrate RadixCache and SWARadixCache state
across EP<->TP switches. See docs/paras/radix_cache.md for the design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache


@dataclass
class TreeRecord:
    """One record per non-root tree node.

    Self-contained: ``full_token_path`` encodes position in the tree (parent
    chain) so the receiver can rebuild the tree purely via ``insert()`` calls
    without needing the original source tree's node identity.
    """

    full_token_path: List[int]            # tokens from root to this node, concatenated
    extra_key: Optional[Any]              # RadixKey.extra_key (LoRA / cache_salt / multimodal hash)
    value_slots: List[int]                # node.value as list of int slot indices
    swa_tombstone: bool = False           # only relevant for SWARadixCache (T5)
    last_access_time: float = 0.0         # canonicalized on receiver by T12
    host_value: Optional[List[int]] = None  # HiRadix offload (asserted off; defensive None)


def serialize_radix_cache(tree: "RadixCache") -> List[TreeRecord]:
    """Iterative DFS through the radix tree; emit one ``TreeRecord`` per non-root node.

    NEVER recurse: the max tree depth is bounded by ``max_seq_length`` which
    can easily exceed Python's default recursion limit. An explicit stack is
    used to make the traversal robust on pathologically deep trees.
    """
    records: List[TreeRecord] = []
    # Iterative DFS using explicit stack of (node, parent_path_list).
    # The root node is skipped -- only non-root nodes produce records.
    stack: List[tuple] = []
    root = tree.root_node
    for child in root.children.values():
        stack.append((child, []))

    while stack:
        node, parent_path = stack.pop()
        # ``node.key`` is a RadixKey. Fall back to iterating the key directly
        # for defensive handling when a mock or alternative key type is used.
        if hasattr(node.key, "token_ids"):
            token_ids = list(node.key.token_ids)
        else:
            token_ids = list(node.key)
        full_path = parent_path + token_ids
        extra_key = getattr(node.key, "extra_key", None)
        # Non-root nodes always have a tensor value set by ``_insert_helper``.
        # The guard here is defensive against partial / mocked tree states.
        if node.value is None:
            value_slots: List[int] = []
        elif isinstance(node.value, torch.Tensor):
            value_slots = node.value.tolist()
        else:
            value_slots = list(node.value)
        record = TreeRecord(
            full_token_path=full_path,
            extra_key=extra_key,
            value_slots=value_slots,
            swa_tombstone=False,  # T5 overrides for SWA
            last_access_time=getattr(node, "last_access_time", 0.0),
            host_value=None,  # HiRadix asserted off
        )
        records.append(record)
        # Push children with this node's full_path as their parent_path.
        for child in node.children.values():
            stack.append((child, full_path))

    return records


def rebuild_radix_cache(
    tree,
    records: list,
    remap_slot_idx,
    metrics=None,
):
    """Rebuild a radix tree from migration records, applying slot-index remap.

    Sorts records parent-first (by full_token_path length ascending) so each
    insert() resolves into an existing chain rather than forcing later merges.

    Args:
        tree: target RadixCache or SWARadixCache (already reset / empty post-tree.reset()).
        records: List[TreeRecord] from serialize_*_radix_cache on sender.
        remap_slot_idx: Callable[[int], int]. Maps source-pool slot to dest-pool slot.
            Returns -1 to signal "slot dropped" (record skipped, metric incremented).
        metrics: optional MigrationMetrics instance to increment dedup_drop_count.

    Notes:
        - Does NOT call inc_lock_ref (that's T19's job after this function returns).
        - For SWA: records with swa_tombstone=True are inserted via the post-T2
          tombstone-aware insert path (swa_evicted_seqlen=len(full_token_path)).
        - For MHA: swa_evicted_seqlen kwarg is silently ignored by RadixCache.insert.
    """
    import torch

    try:
        from sglang.srt.mem_cache.radix_cache import RadixKey
    except ImportError:
        RadixKey = None

    sorted_records = sorted(
        records,
        key=lambda r: (len(r.full_token_path), tuple(r.full_token_path)),
    )

    has_swa_kwarg = _supports_swa_evicted_seqlen(tree)

    for record in sorted_records:
        new_slots = [remap_slot_idx(s) for s in record.value_slots]
        if any(s < 0 for s in new_slots):
            if metrics is not None:
                metrics.dedup_drop_count += 1
            continue

        value_tensor = torch.tensor(new_slots, dtype=torch.int64)

        if RadixKey is not None:
            key = RadixKey(record.full_token_path, record.extra_key)
        else:
            key = record.full_token_path

        if has_swa_kwarg:
            swa_evicted = len(record.full_token_path) if record.swa_tombstone else 0
            tree.insert(key, value_tensor, swa_evicted_seqlen=swa_evicted)
        else:
            tree.insert(key, value_tensor)


def _supports_swa_evicted_seqlen(tree) -> bool:
    """Detect at runtime whether tree.insert accepts swa_evicted_seqlen kwarg."""
    import inspect
    try:
        sig = inspect.signature(tree.insert)
        return "swa_evicted_seqlen" in sig.parameters
    except (TypeError, ValueError):
        return False


def serialize_swa_radix_cache(tree: "SWARadixCache") -> List[TreeRecord]:
    """Iterative DFS through the SWA radix tree; emit one ``TreeRecord`` per non-root node.

    Mirrors :func:`serialize_radix_cache` but additionally captures the
    ``swa_tombstone`` flag on each SWA tree node. The traversal DOES descend
    through tombstones because their descendants remain meaningful for the
    ``_match_prefix_helper`` W-distance rule (a tombstone's children may still
    hold valid in-window slots).

    NEVER recurse: the maximum tree depth is bounded by ``max_seq_length``
    which can easily exceed Python's default recursion limit. An explicit
    stack is used to make the traversal robust on pathologically deep trees.
    """
    records: List[TreeRecord] = []
    # Iterative DFS using explicit stack of (node, parent_path_list).
    # The root node is skipped -- only non-root nodes produce records.
    stack: List[tuple] = []
    root = tree.root_node
    for child in root.children.values():
        stack.append((child, []))

    while stack:
        node, parent_path = stack.pop()
        # ``node.key`` is a RadixKey. Fall back to iterating the key directly
        # for defensive handling when a mock or alternative key type is used.
        if hasattr(node.key, "token_ids"):
            token_ids = list(node.key.token_ids)
        else:
            token_ids = list(node.key)
        full_path = parent_path + token_ids
        extra_key = getattr(node.key, "extra_key", None)
        # Non-root nodes always have a tensor value set by ``_insert_helper``.
        # The guard here is defensive against partial / mocked tree states
        # and against tombstoned nodes that may have had ``value`` cleared.
        if node.value is None:
            value_slots: List[int] = []
        elif isinstance(node.value, torch.Tensor):
            value_slots = node.value.tolist()
        else:
            value_slots = list(node.value)
        record = TreeRecord(
            full_token_path=full_path,
            extra_key=extra_key,
            value_slots=value_slots,
            # SWA-specific: capture the per-node tombstone flag. The MHA path
            # in serialize_radix_cache hard-codes this to False.
            swa_tombstone=getattr(node, "swa_tombstone", False),
            last_access_time=getattr(node, "last_access_time", 0.0),
            host_value=None,  # HiRadix asserted off
        )
        records.append(record)
        # Push children with this node's full_path as their parent_path.
        # We DO descend through tombstones because their descendants may still
        # hold valid in-window slots that match_prefix needs to find.
        for child in node.children.values():
            stack.append((child, full_path))

    return records
