"""ParaS wiring for GPT-OSS hybrid attention model.

Classifies layers as full-attention or sliding-window based on
``config.layer_types`` and exposes the resulting ``LayerCacheSpec`` list.

The GPT-OSS model uses a heterogeneous layer stack where each layer is
either ``"full_attention"`` or ``"sliding_attention"``
(see ``sglang/srt/models/gpt_oss.py``).  This module reads that
configuration, delegates to ``classify_layers_from_config`` for the
actual classification, and provides a ``wire_attention_mixin`` helper
that grafts ``ParaSAttentionMixin`` onto every ``GptOssAttention``
instance in the model.

Synthetic tests live in the QA scenarios of this task.
120B end-to-end validation is a follow-up.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from sglang.srt.paras.cache_transfer import LayerCacheSpec, classify_layers_from_config
from sglang.srt.paras.layers.paras_attention import ParaSAttentionMixin

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


class GPTOSSParaSWiring:
    """ParaS wiring for GPT-OSS hybrid attention models.

    Reads ``config.layer_types`` (a list of ``"full_attention"`` /
    ``"sliding_attention"`` strings) and produces a per-layer
    ``LayerCacheSpec`` list that downstream ParaS infrastructure
    (memory manager, gather/scatter managers) consumes.

    Args:
        config: HuggingFace-style model config with at least
            ``num_hidden_layers``, ``layer_types``, ``sliding_window``,
            ``num_key_value_heads``, ``num_attention_heads``,
            ``hidden_size``.
        tp_size: Tensor-parallelism world size.
        ep_tokens_full: EP token capacity for full-attention layers.
        tp_tokens_full: TP token capacity for full-attention layers.
        ep_tokens_swa: EP token capacity for SWA layers.
        tp_tokens_swa: TP token capacity for SWA layers.
        ratio: Scaling factor applied to token capacities (default 1.0).

    Attributes:
        layer_specs: ``list[LayerCacheSpec]`` — one entry per layer.
    """

    def __init__(
        self,
        config,
        *,
        tp_size: int,
        ep_tokens_full: int,
        tp_tokens_full: int,
        ep_tokens_swa: int,
        tp_tokens_swa: int,
        ratio: float = 1.0,
    ) -> None:
        self.layer_specs: List[LayerCacheSpec] = classify_layers_from_config(
            config,
            tp_size=tp_size,
            ep_tokens_full=ep_tokens_full,
            tp_tokens_full=tp_tokens_full,
            ep_tokens_swa=ep_tokens_swa,
            tp_tokens_swa=tp_tokens_swa,
            ratio=ratio,
        )

        n_full = sum(1 for s in self.layer_specs if s.kind == "full")
        n_swa = sum(1 for s in self.layer_specs if s.kind == "swa")
        logger.info(
            "GPTOSSParaSWiring: %d layers (%d full, %d swa)",
            len(self.layer_specs),
            n_full,
            n_swa,
        )

    @staticmethod
    def wire_attention_mixin(model: "nn.Module") -> int:
        """Graft ``ParaSAttentionMixin`` onto every ``GptOssAttention`` in *model*.

        This is the GPT-OSS equivalent of the class-swap done in
        ``Qwen3MoeDecoderLayerParaS.__init__`` for Qwen3-MoE.  It walks
        all sub-modules, finds instances of ``GptOssAttention``, and
        changes their ``__class__`` to a dynamically created subclass
        that inherits from both ``ParaSAttentionMixin`` and
        ``GptOssAttention``.

        Returns:
            Number of attention modules rewired.
        """
        # Lazy import to avoid circular dependency — the model file is
        # not needed at module-load time.
        from sglang.srt.models.gpt_oss import GptOssAttention

        # Build the combined class once (MRO: mixin first → methods win).
        _ParaSGptOssAttention = type(
            "GptOssAttentionParaS",
            (ParaSAttentionMixin, GptOssAttention),
            {},
        )

        count = 0
        for module in model.modules():
            if isinstance(module, GptOssAttention):
                module.__class__ = _ParaSGptOssAttention
                count += 1

        logger.info(
            "GPTOSSParaSWiring.wire_attention_mixin: rewired %d GptOssAttention modules",
            count,
        )
        return count
