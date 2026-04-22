"""Cache transfer infrastructure for ParaS SWA support.

This module provides the core abstractions for managing KV cache transfers
across tensor parallelism (TP) and expert parallelism (EP) boundaries in
hybrid attention scenarios (full + sliding window).
"""

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class LayerCacheSpec:
    """Specification for a single layer's KV cache in ParaS.
    
    Attributes:
        layer_id: Layer index in the model.
        kind: Attention type — "full" for full attention, "swa" for sliding window.
        tokens_cap_ep: Per-layer token capacity for expert parallelism.
        tokens_cap_tp: Per-layer token capacity for tensor parallelism.
        num_kv_heads: Number of key-value heads in this layer.
        head_dim: Dimension of each attention head.
        sliding_window_size: Window size for SWA layers (None for full attention).
    """
    layer_id: int
    kind: Literal["full", "swa"]
    tokens_cap_ep: int
    tokens_cap_tp: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: Optional[int]


@runtime_checkable
class CacheTransferBackend(Protocol):
    """Protocol for cache transfer backends (gather/scatter operations).
    
    Implementations handle moving KV cache data across TP/EP boundaries
    for both full and sliding window attention layers.
    """

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        """Gather KV cache for one layer across TP/EP boundaries.
        
        Args:
            spec: LayerCacheSpec describing the layer.
            **kwargs: Backend-specific arguments (pinned in T3/T7).
        """
        ...

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        """Scatter KV cache for one layer across TP/EP boundaries.
        
        Args:
            spec: LayerCacheSpec describing the layer.
            **kwargs: Backend-specific arguments (pinned in T3/T7).
        """
        ...


def classify_layers_from_config(
    hf_config,
    *,
    tp_size: int,
    ep_tokens_full: int,
    tp_tokens_full: int,
    ep_tokens_swa: int,
    tp_tokens_swa: int,
    ratio: float = 1.0,
) -> list[LayerCacheSpec]:
    """Classify layers and generate LayerCacheSpec list from HF config.
    
    Handles two config styles:
    - GPT-OSS/Gemma: config.layer_types = ["full_attention", "sliding_attention", ...]
    - Default: all layers are full attention
    
    Args:
        hf_config: HuggingFace model config object.
        tp_size: Tensor parallelism size.
        ep_tokens_full: EP token capacity for full attention layers.
        tp_tokens_full: TP token capacity for full attention layers.
        ep_tokens_swa: EP token capacity for SWA layers.
        tp_tokens_swa: TP token capacity for SWA layers.
        ratio: Scaling factor for token capacities (default 1.0).
    
    Returns:
        List of LayerCacheSpec, one per layer.
    
    Raises:
        ValueError: If validation fails (G15: uniform heads, G7: tp_size <= num_kv_heads for SWA).
    """
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    
    # Determine layer types
    if hasattr(hf_config, "layer_types") and hf_config.layer_types is not None:
        layer_types = hf_config.layer_types
    else:
        # Default: all full attention
        layer_types = ["full_attention"] * num_layers
    
    # Build specs
    specs = []
    for layer_id, layer_type in enumerate(layer_types):
        is_swa = layer_type == "sliding_attention"
        kind = "swa" if is_swa else "full"
        
        # Determine token capacities
        if is_swa:
            tokens_cap_ep = int(ep_tokens_swa * ratio)
            tokens_cap_tp = int(tp_tokens_swa * ratio)
            # Derive sliding_window_size from config (same as gpt_oss.py:95)
            sliding_window_size = hf_config.sliding_window - 1
        else:
            tokens_cap_ep = int(ep_tokens_full * ratio)
            tokens_cap_tp = int(tp_tokens_full * ratio)
            sliding_window_size = None
        
        spec = LayerCacheSpec(
            layer_id=layer_id,
            kind=kind,
            tokens_cap_ep=tokens_cap_ep,
            tokens_cap_tp=tokens_cap_tp,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window_size=sliding_window_size,
        )
        specs.append(spec)
    
    # Validation: G15 (uniform heads)
    if not all(s.num_kv_heads == num_kv_heads for s in specs):
        raise ValueError(
            "LayerCacheSpec: num_kv_heads must be uniform across all layers"
        )
    
    # Validation: G7 (tp_size <= num_kv_heads for SWA)
    has_swa = any(s.kind == "swa" for s in specs)
    if has_swa and tp_size > num_kv_heads:
        raise ValueError(
            "LayerCacheSpec: tp_size must be <= num_kv_heads for SWA layers"
        )
    
    return specs


def validate_layer_specs(specs: list[LayerCacheSpec], tp_size: int) -> None:
    """Validate a list of LayerCacheSpec for consistency.
    
    Args:
        specs: List of LayerCacheSpec to validate.
        tp_size: Tensor parallelism size.
    
    Raises:
        ValueError: If validation fails (G15: uniform heads, G7: tp_size <= num_kv_heads for SWA).
    """
    if not specs:
        return
    
    # G15: uniform heads
    num_kv_heads = specs[0].num_kv_heads
    if not all(s.num_kv_heads == num_kv_heads for s in specs):
        raise ValueError(
            "LayerCacheSpec: num_kv_heads must be uniform across all layers"
        )
    
    # G7: tp_size <= num_kv_heads for SWA
    has_swa = any(s.kind == "swa" for s in specs)
    if has_swa and tp_size > num_kv_heads:
        raise ValueError(
            "LayerCacheSpec: tp_size must be <= num_kv_heads for SWA layers"
        )


__all__ = [
    "LayerCacheSpec",
    "CacheTransferBackend",
    "classify_layers_from_config",
    "validate_layer_specs",
]
