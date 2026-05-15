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
