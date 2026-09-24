"""Cache transfer infrastructure for ParaS SWA support.

Layout:
    base.py   -- CacheTransferBackend Protocol + CacheTransferBase concrete
                 shared base class.
    utils.py  -- stateless per-layer gather/scatter kernel wrappers
                 shared by MHA and SWA backends.
    mha.py    -- MHACacheTransfer backend (full-attention layers).
    swa.py    -- SWACacheTransfer backend (sliding-window layers).

The shared per-layer schema ``LayerCacheSpec`` and its config helpers live
in ``sglang.srt.paras.layers.utils`` (they are consumed by both planning
and transfer paths).
"""

from sglang.srt.paras.cache_transfer.base import (
    CacheTransferBackend,
    CacheTransferBase,
)

__all__ = [
    "CacheTransferBackend",
    "CacheTransferBase",
]
