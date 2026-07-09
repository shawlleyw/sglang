"""
ParaS (Parallelism Switch) subclasses for GPT-OSS hybrid attention model.

All ParaS-specific logic for GPT-OSS lives here.  The base model file
(sglang/srt/models/gpt_oss.py) stays clean of ParaS code.

GPT-OSS uses heterogeneous layers: each layer is either "full_attention"
or "sliding_attention" (see config.layer_types).  This module mirrors the
Qwen3-MoE ParaS pattern (sglang/srt/paras/models/qwen3_moe.py) while
handling the per-layer attention geometry and MXFP4 weight loading that
are unique to GPT-OSS.

Parameter management under ParaS
--------------------------------

The unified memory manager holds only tensors whose shape differs
between EP and TP modes and therefore need per-switch redistribution.
Everything else is loaded through the normal PyTorch parameter path
and stays untouched across switches.  Concretely for GPT-OSS:

  Managed by the UMM:
    * mlp.{ep,tp}_experts.w13 / w2  (four-anchor unified layout, EP<->TP transport)
    * self_attn.qkv_proj.weight / tp_weight
    * self_attn.o_proj.weight
    * FP8 weight scales (when quant_name == "fp8")
    * Staging buffers (when configure_method != "peer_access")

  Replicated across all ranks, not in the UMM:
    * input_layernorm.weight, post_attention_layernorm.weight
    * self_attn.sinks (full + tp slice view, rebind on switch)
    * self_attn.qkv_proj.bias (full + tp concatenated slice, rebind on switch)
    * self_attn.o_proj.bias  (hidden-dim output, added after AllReduce)
    * mlp.experts.w{13,2}_weight_bias (full + ep/tp views, rebind on switch)
    * mlp.router.weight / router.bias  (router runs on all ranks)
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
from sglang.srt.paras.layers.paras_attention import ParaSAttentionMixin
from sglang.srt.paras.layers.paras_decoder_layer import ParaSDecoderLayerMixin
from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin
from sglang.srt.paras.layers.paras_model import ParaSModelMixin
from sglang.srt.paras.paras_memory_manager import (
    create_paras_moe_aliases,
    get_global_paras_memory_manager,
    plan_gpt_oss_moe_layout,
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
        self.paras_init_moe(
            config,
            quant_config,
            prefix,
            layer_id,
            interleaved_w13=True,
            activation=self.activation,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
        )

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
    """ParaS-enabled attention with GPT-OSS-specific attention-sink handling.

    The model is built in EP mode with ``attn_tp_size=1``, so the
    checkpoint loader populates full-shape tensors for both attention
    sinks (``self.sinks``) and qkv bias (``self.qkv_proj.bias``) on every
    rank.  ``paras_finalize_sinks_views`` and ``paras_finalize_qkv_bias_views``
    capture each full tensor and build the ep / tp Parameters used at
    forward time; switching is a pure rebind.
    """

    def paras_finalize_sinks_views(self):
        from sglang.srt.paras.paras_parallel_state import (
            get_paras_tp_rank,
            get_paras_tp_size,
        )
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        self.ep_sinks = self.sinks
        heads_per_rank = self.total_num_heads // paras_tp_size
        start = paras_tp_rank * heads_per_rank
        end = start + heads_per_rank
        self.tp_sinks = nn.Parameter(
            self.sinks.data[start:end], requires_grad=False
        )

    def paras_finalize_qkv_bias_views(self):
        if self.qkv_proj.bias is None:
            return
        from sglang.srt.paras.paras_parallel_state import (
            get_paras_tp_rank,
            get_paras_tp_size,
        )
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        hs = self.head_dim
        # GQA replication: mirror QKVParallelLinear.paras_finalize_tp_views
        # (linear.py L1204+) — when paras_tp_size > total_num_kv_heads each
        # rank pulls from kv_shard_idx = rank // num_replicas instead of rank.
        if paras_tp_size >= self.total_num_kv_heads:
            tp_num_kv_heads = 1
            paras_num_kv_replicas = paras_tp_size // self.total_num_kv_heads
        else:
            tp_num_kv_heads = self.total_num_kv_heads // paras_tp_size
            paras_num_kv_replicas = 1
        tp_num_heads = self.total_num_heads // paras_tp_size
        kv_shard_idx = paras_tp_rank // paras_num_kv_replicas
        full_data = self.qkv_proj.bias.data
        q_start = paras_tp_rank * tp_num_heads * hs
        k_start = (
            self.total_num_heads * hs
            + kv_shard_idx * tp_num_kv_heads * hs
        )
        v_start = (
            (self.total_num_heads + self.total_num_kv_heads) * hs
            + kv_shard_idx * tp_num_kv_heads * hs
        )
        self.ep_qkv_bias = self.qkv_proj.bias
        self.tp_qkv_bias = nn.Parameter(
            torch.cat([
                full_data[q_start : q_start + tp_num_heads * hs],
                full_data[k_start : k_start + tp_num_kv_heads * hs],
                full_data[v_start : v_start + tp_num_kv_heads * hs],
            ]),
            requires_grad=False,
        )

    @paras_func
    def paras_configure_tp(self, paras_tp_size, paras_tp_rank):
        super().paras_configure_tp(paras_tp_size, paras_tp_rank)
        self.sinks = self.tp_sinks
        if hasattr(self, "tp_qkv_bias"):
            self.qkv_proj.bias = self.tp_qkv_bias

    @paras_func
    def paras_configure_ep(self):
        super().paras_configure_ep()
        self.sinks = self.ep_sinks
        if hasattr(self, "ep_qkv_bias"):
            self.qkv_proj.bias = self.ep_qkv_bias


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

        # The UMM was constructed by model_runner.load_model() before this
        # model was instantiated; pull it via the global accessor.
        manager = get_global_paras_memory_manager()
        assert manager is not None, (
            "ParaS UMM not constructed: model_runner.load_model() should have "
            "created it before get_model() under enable_paras_moe."
        )

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

        plan = manager.plan_hybrid_swa_kv_capacity(
            config=config,
            tp_size=get_paras_tp_size(),
            head_dim=head_dim,
        )

        manager.reserve_kv_cache(
            num_layers=config.num_hidden_layers,
            ep_max_tokens=plan.ep_max_tokens,
            tp_max_tokens=plan.tp_max_tokens,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            tp_size=get_paras_tp_size(),
            kv_dtype=plan.kv_dtype,
            page_size=getattr(get_global_server_args(), "page_size", 1),
            prefix="model",
            layer_specs=plan.layer_specs,
        )

        manager.materialize()
        create_paras_moe_aliases(manager, config.num_hidden_layers, prefix="model")
        logger.info("ParaSMemoryManager materialized: %s", manager)
        self.paras_memory_manager = manager
        self.paras_layer_specs = plan.layer_specs

        # Skip peer access pre-init when using NCCL transfer (no benefit
        # and seems to interact badly with NCCL on A100).  Set
        # PARAS_DISABLE_PEER_ACCESS=0 to re-enable for peer_access path.
        if os.environ.get("PARAS_DISABLE_PEER_ACCESS", "1") == "1":
            self._fused_peer_access_ctx = None
        else:
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

        super().load_weights(
            weights,
            is_nextn=is_nextn,
            weight_name_mapping=weight_name_mapping,
        )

        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "paras_finalize_moe_biases"):
                mlp.paras_finalize_moe_biases()
            if hasattr(mlp, "paras_finalize_moe_scales"):
                mlp.paras_finalize_moe_scales()
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "paras_finalize_sinks_views"):
                    attn.paras_finalize_sinks_views()
                if hasattr(attn, "paras_finalize_qkv_bias_views"):
                    attn.paras_finalize_qkv_bias_views()

        if hasattr(self.model, "paras_finalize_attn_views"):
            self.model.paras_finalize_attn_views()

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
