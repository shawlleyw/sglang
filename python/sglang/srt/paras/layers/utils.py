"""Shared schema and helpers for ParaS per-layer KV cache configuration.

``LayerCacheSpec`` is the per-layer descriptor consumed by both the planning
side (``ParaSMemoryManager.reserve_kv_cache``, model __init__) and the
transfer side (``cache_transfer.mha``/``swa``, scatter/gather managers). It
lives here rather than under ``cache_transfer/`` because it is shared schema,
not transfer logic.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class LayerCacheSpec:
    """Specification for a single layer's KV cache in ParaS.

    Attributes:
        layer_id: Layer index in the model.
        kind: Attention type -- ``"full"`` for full attention, ``"swa"`` for
            sliding window.
        tokens_cap_ep: Per-layer token capacity for expert parallelism.
        tokens_cap_tp: Per-layer token capacity for tensor parallelism.
        num_kv_heads: Number of key-value heads in this layer.
        head_dim: Dimension of each attention head.
        sliding_window_size: Window size for SWA layers (``None`` for full).
    """
    layer_id: int
    kind: Literal["full", "swa"]
    tokens_cap_ep: int
    tokens_cap_tp: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: Optional[int]


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
    """Classify layers and generate a ``LayerCacheSpec`` list from an HF config.

    Handles two config styles:
      * GPT-OSS / Gemma: ``config.layer_types = ["full_attention", "sliding_attention", ...]``
      * Default (everything else): all layers are full attention.
    """
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )

    if hasattr(hf_config, "layer_types") and hf_config.layer_types is not None:
        layer_types = hf_config.layer_types
    else:
        layer_types = ["full_attention"] * num_layers

    specs = []
    for layer_id, layer_type in enumerate(layer_types):
        is_swa = layer_type == "sliding_attention"
        kind = "swa" if is_swa else "full"

        if is_swa:
            tokens_cap_ep = int(ep_tokens_swa * ratio)
            tokens_cap_tp = int(tp_tokens_swa * ratio)
            sliding_window_size = hf_config.sliding_window - 1
        else:
            tokens_cap_ep = int(ep_tokens_full * ratio)
            tokens_cap_tp = int(tp_tokens_full * ratio)
            sliding_window_size = None

        specs.append(LayerCacheSpec(
            layer_id=layer_id,
            kind=kind,
            tokens_cap_ep=tokens_cap_ep,
            tokens_cap_tp=tokens_cap_tp,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window_size=sliding_window_size,
        ))

    validate_layer_specs(specs, tp_size)
    return specs


def validate_layer_specs(specs: list[LayerCacheSpec], tp_size: int) -> None:
    """Validate a list of ``LayerCacheSpec`` for consistency."""
    if not specs:
        return

    num_kv_heads = specs[0].num_kv_heads
    if not all(s.num_kv_heads == num_kv_heads for s in specs):
        raise ValueError(
            "LayerCacheSpec: num_kv_heads must be uniform across all layers"
        )

    has_swa = any(s.kind == "swa" for s in specs)
    if has_swa and tp_size > num_kv_heads:
        raise ValueError(
            "LayerCacheSpec: tp_size must be <= num_kv_heads for SWA layers"
        )


__all__ = [
    "LayerCacheSpec",
    "classify_layers_from_config",
    "validate_layer_specs",
]
