"""Cache transfer infrastructure for ParaS SWA support.

Re-exports the abstract types and helpers from ``base.py`` so existing
import paths keep working.

Layout:
    base.py   -- abstract types: LayerCacheSpec, CacheTransferBackend,
                 CacheTransferBase and config classifiers
    utils.py  -- stateless per-layer gather/scatter kernel wrappers
                 shared by MHA and SWA backends
    mha.py    -- MHACacheTransfer backend (full-attention layers)
    swa.py    -- SWACacheTransfer backend (sliding-window layers)
"""

from sglang.srt.paras.cache_transfer.base import (
    CacheTransferBackend,
    CacheTransferBase,
    LayerCacheSpec,
    classify_layers_from_config,
    validate_layer_specs,
)

__all__ = [
    "LayerCacheSpec",
    "CacheTransferBackend",
    "CacheTransferBase",
    "classify_layers_from_config",
    "validate_layer_specs",
]
