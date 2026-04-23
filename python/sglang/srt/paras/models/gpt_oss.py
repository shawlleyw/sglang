"""
ParaS (Parallelism Switch) subclasses for GPT-OSS hybrid attention model.

All ParaS-specific logic for GPT-OSS lives here.  The base model file
(sglang/srt/models/gpt_oss.py) stays clean of ParaS code.

GPT-OSS uses heterogeneous layers: each layer is either "full_attention"
or "sliding_attention" (see config.layer_types).  This module mirrors the
Qwen3-MoE ParaS pattern (sglang/srt/paras/models/qwen3_moe.py) while
handling the per-layer attention geometry and MXFP4 weight loading that
are unique to GPT-OSS.
"""

import logging
import os
import time
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.paras.cache_transfer import classify_layers_from_config
from sglang.srt.paras.layers.paras_attention import ParaSAttentionMixin
from sglang.srt.paras.layers.paras_decoder_layer import ParaSDecoderLayerMixin
from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin
from sglang.srt.paras.layers.paras_model import ParaSModelMixin
from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    create_paras_moe_aliases,
    plan_gpt_oss_moe_layout,
    plan_hybrid_kv_budget,
    set_global_paras_memory_manager,
)
from sglang.srt.paras.paras_parallel_state import (
    get_paras_dp_size,
    get_paras_tp_group,
    get_paras_tp_size,
)
from sglang.srt.paras.utils import paras_func
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

# Import base classes from the model file
from sglang.srt.models.gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
    GptOssSparseMoeBlock,
)

logger = logging.getLogger(__name__)


# 1. ParaS MoE block — adds dual experts (EP + TP) and weight redistribution
class GptOssSparseMoeBlockParaS(ParaSMoeBlockMixin, GptOssSparseMoeBlock):
    """ParaS-enabled GPT-OSS MoE block with EP↔TP switching."""

    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        super().__init__(layer_id, config, quant_config, prefix)
        self.paras_init_moe(config, quant_config, prefix, layer_id)

    def forward_normal(
        self,
        hidden_states,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        """Adapter: GPT-OSS base forward_normal lacks use_reduce_scatter."""
        return super().forward_normal(hidden_states, should_allreduce_fusion)

    def forward(
        self,
        hidden_states,
        forward_batch=None,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        return self.paras_forward(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )


# 2. ParaS Attention — adds paras_configure_tp/ep methods
class GptOssAttentionParaS(ParaSAttentionMixin, GptOssAttention):
    """ParaS-enabled attention. Mixin adds paras_configure_tp/ep/helper."""

    pass


# 3. ParaS Decoder Layer — swaps in ParaS MoE block + attention, manages dual communicators
class GptOssDecoderLayerParaS(ParaSDecoderLayerMixin, GptOssDecoderLayer):
    """ParaS-enabled decoder layer with dual EP/TP communicator contexts."""

    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        sliding_window_size=None,
    ):
        super().__init__(config, layer_id, quant_config, prefix, sliding_window_size)

        # Replace base MLP with ParaS MoE block.  The base constructor
        # already created a plain GptOssSparseMoeBlock; we replace it with
        # the ParaS variant that owns dual EP/TP experts.
        self.mlp = GptOssSparseMoeBlockParaS(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        # Swap attention class to add ParaS methods (no reconstruction needed —
        # ParaSAttentionMixin adds no __init__ state, just methods)
        self.self_attn.__class__ = GptOssAttentionParaS

        # Initialize dual communicator state for EP↔TP switching
        # is_previous_layer_sparse=True because all GPT-OSS layers are sparse
        self.paras_init_layer(
            config, layer_id, self.is_layer_sparse, is_previous_layer_sparse=True
        )


# 4. ParaS Model — passes ParaS decoder layer type, adds layer-level conversion
class GptOssModelParaS(ParaSModelMixin, GptOssModel):
    """ParaS-enabled model with layer-by-layer EP→TP conversion."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(
            config,
            quant_config,
            prefix,
            decoder_layer_type=GptOssDecoderLayerParaS,
        )


# 5. ParaS CausalLM — top-level entry point
class GptOssForCausalLMParaS(GptOssForCausalLM):
    """
    ParaS-enabled CausalLM for GPT-OSS.

    Overrides:
    - __init__: Uses ParaS model (which uses ParaS decoder layers);
      owns the full ParaSMemoryManager lifecycle including heterogeneous
      KV budget computation for hybrid full/SWA layers.
    - paras_configure_tp/ep: Delegates to model + cleans up weight buffers
    """

    def __init__(self, config, quant_config=None, prefix=""):
        # Skip parent __init__ to avoid creating the non-ParaS model
        # (creating the full model twice wastes too much memory)
        nn.Module.__init__(self)
        from sglang.srt.distributed import get_pp_group

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # Normalize GPT-OSS config attribute names for ParaS mixin
        # compatibility: GPT-OSS uses num_local_experts / intermediate_size
        # where the generic ParaS mixins expect num_experts /
        # moe_intermediate_size.
        if not hasattr(config, "num_experts"):
            config.num_experts = config.num_local_experts
        if not hasattr(config, "moe_intermediate_size"):
            config.moe_intermediate_size = config.intermediate_size

        # ---- ParaS Memory Manager ----
        manager = ParaSMemoryManager()

        quant_name = None
        fp8_block_size = None
        if quant_config is not None:
            qn = quant_config.get_name()
            if qn == "fp8":
                quant_name = "fp8"
                if (
                    hasattr(quant_config, "weight_block_size")
                    and quant_config.weight_block_size
                ):
                    fp8_block_size = quant_config.weight_block_size[0]

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        moe_tp_size = get_moe_tensor_parallel_world_size()
        dp_size = get_paras_dp_size()

        configure_method = os.environ.get("PARAS_CONFIGURE_METHOD", "peer_access")

        plan_gpt_oss_moe_layout(
            manager,
            num_layers=config.num_hidden_layers,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            ep_size=get_moe_expert_parallel_world_size(),
            tp_size=get_paras_tp_size(),
            dp_size=dp_size,
            moe_tp_size=moe_tp_size,
            quant_name=quant_name,
            fp8_block_size=fp8_block_size,
            num_fused_shared_experts=getattr(config, "num_fused_shared_experts", 0),
            configure_method=configure_method,
            prefix="model",
        )

        # --- Compute heterogeneous KV token budgets -----------------------
        # GPT-OSS has full-attention and sliding-window layers with
        # different KV capacity requirements.  Mirrors the baseline budget
        # semantics from model_runner.py and splits the budget via
        # plan_hybrid_kv_budget.
        from sglang.srt.utils.common import get_available_gpu_memory

        _server_args = get_global_server_args()
        _mem_fraction = _server_args.mem_fraction_static
        _page_size = getattr(_server_args, "page_size", 1)

        _kv_dtype_str = _server_args.kv_cache_dtype
        if _kv_dtype_str == "auto":
            _kv_store_dtype = torch.bfloat16
        elif _kv_dtype_str in ("fp8", "fp8_e4m3fn"):
            _kv_store_dtype = torch.float8_e4m3fn
        else:
            _kv_store_dtype = torch.bfloat16

        _total_gpu_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory

        from sglang.srt.distributed import get_world_group

        _world = get_world_group()
        _avail_now_gib = get_available_gpu_memory(
            "cuda",
            torch.cuda.current_device(),
            distributed=_world.world_size > 1,
            cpu_group=_world.cpu_group,
            empty_cache=True,
        )
        _avail_now_bytes = int(_avail_now_gib * (1 << 30))

        _dynamic_reserve_bytes = int(_total_gpu_bytes * (1.0 - _mem_fraction))
        _umm_budget_bytes = max(0, _avail_now_bytes - _dynamic_reserve_bytes)
        _kv_budget_bytes = max(0, _umm_budget_bytes - manager.weights_only_bytes)

        _num_layers = config.num_hidden_layers
        _total_kv_heads = config.num_key_value_heads
        _kv_elem_size = torch.tensor([], dtype=_kv_store_dtype).element_size()

        # Classify layers as full or SWA
        _layer_types = getattr(config, "layer_types", None) or (
            ["full_attention"] * _num_layers
        )
        _n_full = sum(1 for t in _layer_types if t == "full_attention")
        _n_swa = sum(1 for t in _layer_types if t == "sliding_attention")

        # Per-token-per-layer KV cost (K + V, same for full and SWA)
        _cell_bytes = _total_kv_heads * head_dim * 2 * _kv_elem_size
        _total_token_layers = max(1, int(_kv_budget_bytes // _cell_bytes))

        _paras_tp_size = get_paras_tp_size()

        if _n_swa > 0:
            _swa_ratio = getattr(_server_args, "swa_full_tokens_ratio", 0.5)
            _full_max_tokens, _swa_max_tokens = plan_hybrid_kv_budget(
                _total_token_layers,
                _n_full,
                _n_swa,
                _swa_ratio,
            )
        else:
            _full_max_tokens = max(1, _total_token_layers // _num_layers)
            _swa_max_tokens = 0

        logger.info(
            "ParaS GPT-OSS KV budget: avail_now=%.3fGiB  "
            "total=%.3fGiB  dynamic_reserve=%.3fGiB  "
            "umm_budget=%.3fGiB  weights_only=%.3fGiB  "
            "kv_budget=%.3fGiB  full_max_tokens=%d  swa_max_tokens=%d  "
            "layers=%d (full=%d swa=%d)",
            _avail_now_gib,
            _total_gpu_bytes / (1 << 30),
            _dynamic_reserve_bytes / (1 << 30),
            _umm_budget_bytes / (1 << 30),
            manager.weights_only_bytes / (1 << 30),
            _kv_budget_bytes / (1 << 30),
            _full_max_tokens,
            _swa_max_tokens,
            _num_layers,
            _n_full,
            _n_swa,
        )

        # Build per-layer LayerCacheSpec with heterogeneous budgets
        _layer_specs = classify_layers_from_config(
            config,
            tp_size=_paras_tp_size,
            ep_tokens_full=_full_max_tokens,
            tp_tokens_full=_full_max_tokens * _paras_tp_size,
            ep_tokens_swa=_swa_max_tokens,
            tp_tokens_swa=_swa_max_tokens * _paras_tp_size,
        )

        # Reserve KV in manager (union layout, heterogeneous per-layer sizes)
        manager.reserve_kv_cache(
            num_layers=_num_layers,
            ep_max_tokens=_full_max_tokens,
            tp_max_tokens=_full_max_tokens * _paras_tp_size,
            num_kv_heads=_total_kv_heads,
            head_dim=head_dim,
            kv_dtype=_kv_store_dtype,
            page_size=_page_size,
            prefix="model",
            layer_specs=_layer_specs,
        )
        # --- End KV budget computation ------------------------------------

        total_bytes = manager.materialize()
        create_paras_moe_aliases(manager, config.num_hidden_layers, prefix="model")
        logger.info("ParaSMemoryManager materialized: %s", manager)
        self.paras_memory_manager = manager

        # Set global so create_weights() can find the manager
        set_global_paras_memory_manager(manager)

        # Pre-initialize NVLink peer access during model init to avoid
        # overhead at switch time.
        try:
            from sglang.srt.paras.peer_access import init_peer_access

            self._fused_peer_access_ctx = init_peer_access(
                manager, get_paras_tp_group().device_group, get_paras_tp_size()
            )
            logger.info("ParaS fused peer access pre-initialized.")
        except Exception as e:
            logger.warning(
                "ParaS fused peer access pre-init failed (will retry at switch): %s", e
            )
            self._fused_peer_access_ctx = None

        self.model = GptOssModelParaS(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        # Inject pre-initialized peer access context so the switch
        # doesn't pay 6s init cost
        if self._fused_peer_access_ctx is not None:
            self.model._peer_access_ctx = self._fused_peer_access_ctx

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        weight_name_mapping: dict = None,
    ):
        torch.cuda.synchronize()
        start_loading = time.time()

        super().load_weights(weights, is_nextn=is_nextn, weight_name_mapping=weight_name_mapping)

        end_loading = time.time()
        torch.cuda.synchronize()
        logger.info(
            "GptOssForCausalLMParaS loaded weights in %.2f seconds",
            end_loading - start_loading,
        )

    def paras_configure_helper(self):
        pass

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        method = os.environ.get("PARAS_CONFIGURE_METHOD", "peer_access")
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank, method=method)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()
