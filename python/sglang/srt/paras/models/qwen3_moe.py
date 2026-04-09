"""
ParaS (Parallelism Switch) subclasses for Qwen3 MoE.

All ParaS-specific logic for Qwen3 MoE lives here. The base model file
(sglang/srt/models/qwen3_moe.py) stays clean of ParaS code.
"""

import logging
import time
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
)
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.paras.layers.paras_attention import ParaSAttentionMixin
from sglang.srt.paras.layers.paras_decoder_layer import ParaSDecoderLayerMixin
from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin
from sglang.srt.paras.layers.paras_model import ParaSModelMixin
from sglang.srt.paras.layers.utils import paras_load_tp_experts_weight, paras_weight_buffer
from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager, plan_qwen_moe_layout
from sglang.srt.paras.paras_parallel_state import get_paras_tp_size
from sglang.srt.paras.utils import paras_func
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

# Import base classes from the model file
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)

logger = logging.getLogger(__name__)


# 1. ParaS MoE block — adds dual experts (EP + TP) and weight redistribution
class Qwen3MoeSparseMoeBlockParaS(ParaSMoeBlockMixin, Qwen3MoeSparseMoeBlock):
    """ParaS-enabled Qwen3 MoE block with EP↔TP switching."""

    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        # Grab manager ref before super().__init__ (config attr is temporary)
        manager = getattr(config, "_paras_memory_manager", None)
        super().__init__(layer_id, config, quant_config, prefix)
        self.paras_init_moe(
            config, quant_config, prefix, layer_id,
        )

        # Swap EP expert weights to manager-backed views
        if manager is not None and manager.materialized:
            ep_prefix = f"model.layers.{layer_id}.mlp.experts"
            self._swap_ep_expert_weights(self.ep_experts, manager, ep_prefix)

    @staticmethod
    def _swap_ep_expert_weights(ep_experts, manager, prefix):
        """Replace EP expert weight parameters with manager-backed views.

        The manager buffer was pre-allocated in plan_qwen_moe_layout.
        We swap the torch.empty-allocated parameters created by the base
        class with zero-copy views into the contiguous managed buffer.
        """
        for attr_name in ("w13_weight", "w2_weight"):
            entry_name = f"{prefix}.{attr_name}"
            if entry_name not in manager._entries:
                continue
            old_param = getattr(ep_experts, attr_name, None)
            if old_param is None:
                continue
            view = manager.get_view(entry_name)
            new_param = torch.nn.Parameter(view, requires_grad=False)
            # Carry over weight_loader and other custom attrs from old param
            for k in list(vars(old_param)):
                if not k.startswith("_"):
                    setattr(new_param, k, getattr(old_param, k))
            ep_experts.register_parameter(attr_name, new_param)

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
class Qwen3MoeAttentionParaS(ParaSAttentionMixin, Qwen3MoeAttention):
    """ParaS-enabled attention. Mixin adds paras_configure_tp/ep/helper."""

    pass


# 3. ParaS Decoder Layer — swaps in ParaS MoE block + attention, manages dual communicators
class Qwen3MoeDecoderLayerParaS(ParaSDecoderLayerMixin, Qwen3MoeDecoderLayer):
    """ParaS-enabled decoder layer with dual EP/TP communicator contexts."""

    def _create_sparse_moe_block(self, config, layer_id, quant_config, prefix):
        """Override factory to create ParaS MoE block instead of base."""
        return Qwen3MoeSparseMoeBlockParaS(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        super().__init__(config, layer_id, quant_config, prefix, alt_stream)

        # Swap attention class to add ParaS methods (no reconstruction needed —
        # ParaSAttentionMixin adds no __init__ state, just methods)
        self.self_attn.__class__ = Qwen3MoeAttentionParaS

        # Post-init weight swap: the base class creates QKV and o_proj weights
        # via torch.empty. We replace them with views from the contiguous buffer.
        # This is the "post-replace" pattern — unavoidable because we can't
        # modify the base Qwen3MoeAttention constructor.
        manager = getattr(config, "_paras_memory_manager", None)
        if manager is not None and manager.materialized:
            lp = f"model.layers.{layer_id}"

            # QKV projection
            qkv_name = f"{lp}.self_attn.qkv_proj.weight"
            old_qkv = self.self_attn.qkv_proj.weight
            new_qkv = torch.nn.Parameter(
                manager.get_view(qkv_name), requires_grad=False
            )
            for k in list(vars(old_qkv)):
                if not k.startswith("_"):
                    setattr(new_qkv, k, getattr(old_qkv, k))
            self.self_attn.qkv_proj.weight = new_qkv

            # Output projection
            o_name = f"{lp}.self_attn.o_proj.weight"
            old_o = self.self_attn.o_proj.weight
            new_o = torch.nn.Parameter(
                manager.get_view(o_name), requires_grad=False
            )
            for k in list(vars(old_o)):
                if not k.startswith("_"):
                    setattr(new_o, k, getattr(old_o, k))
            self.self_attn.o_proj.weight = new_o

            # Store manager ref on the QKV module so that paras_configure_tp()
            # can later use it to copy q/k/v slices into the managed TP buffer
            # instead of allocating a new tensor via torch.row_stack.
            self.self_attn.qkv_proj._paras_mgr = manager
            self.self_attn.qkv_proj._paras_prefix = (
                f"{lp}.self_attn.qkv_proj"
            )

        # Initialize dual communicator state for EP↔TP switching
        # is_previous_layer_sparse=True because all Qwen3-MoE layers are sparse
        self.paras_init_layer(
            config, layer_id, self.is_layer_sparse, is_previous_layer_sparse=True
        )


# 4. ParaS Model — passes ParaS decoder layer type, adds layer-level conversion
class Qwen3MoeModelParaS(ParaSModelMixin, Qwen3MoeModel):
    """ParaS-enabled model with layer-by-layer EP→TP conversion."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(
            config,
            quant_config,
            prefix,
            decoder_layer_type=Qwen3MoeDecoderLayerParaS,
        )


# 5. ParaS CausalLM — top-level entry point
class Qwen3MoeForCausalLMParaS(Qwen3MoeForCausalLM):
    """
    ParaS-enabled CausalLM for Qwen3 MoE.

    Overrides:
    - __init__: Uses ParaS model (which uses ParaS decoder layers)
    - load_weights: Also loads tp_experts weights
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

        # ---- ParaS Memory Manager ----
        # Create the static weight buffer BEFORE building the model so that
        # submodule constructors can allocate from it.
        #
        # Flow:
        #   1. Create manager + plan layout (reserves all tensor slots)
        #   2. Materialize (allocates one big GPU buffer)
        #   3. Attach to config._paras_memory_manager (temporary carrier)
        #   4. Build model (submodules pick up manager from config)
        #   5. Clean up config attribute
        manager = ParaSMemoryManager()

        quant_name = None
        fp8_block_size = None
        if quant_config is not None:
            qn = quant_config.get_name()
            if qn == "fp8":
                quant_name = "fp8"
                if hasattr(quant_config, "weight_block_size") and quant_config.weight_block_size:
                    fp8_block_size = quant_config.weight_block_size[0]

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        moe_tp_size = get_moe_tensor_parallel_world_size()
        use_triton_kernels = get_moe_runner_backend().is_triton_kernels()

        plan_qwen_moe_layout(
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
            moe_tp_size=moe_tp_size,
            use_triton_kernels=use_triton_kernels,
            quant_name=quant_name,
            fp8_block_size=fp8_block_size,
            num_fused_shared_experts=getattr(config, "num_fused_shared_experts", 0),
            prefix="model",
        )
        total_bytes = manager.materialize()
        logger.info("ParaSMemoryManager materialized: %s", manager)
        self.paras_memory_manager = manager

        # Thread manager through config: this is the only way to pass it
        # to deeply-nested constructors (DecoderLayer → MoeBlock → FusedMoE)
        # without modifying base class signatures. Cleaned up in finally block.
        config._paras_memory_manager = manager
        try:
            self.model = Qwen3MoeModelParaS(
                config, quant_config, prefix=add_prefix("model", prefix)
            )
        finally:
            # Clean the temporary attribute regardless of success/failure
            del config._paras_memory_manager

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Override to also load tp_experts weights and skip EPLB for ParaS."""
        torch.cuda.synchronize()
        start_loading = time.time()

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )

                    # ParaS: also load into tp_experts
                    paras_load_tp_experts_weight(
                        params_dict, name, loaded_weight, shard_id, expert_id
                    )
                    break
                else:
                    if is_expert_weight:
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # ParaS: skip EPLB — weights are handled differently
        # (deliberately NOT building routed_experts_weights_of_layer)

        end_loading = time.time()
        torch.cuda.synchronize()
        logger.info(
            f"Qwen3MoeForCausalLMParaS loaded weights in "
            f"{end_loading - start_loading:.2f} seconds"
        )

    def paras_configure_helper(self):
        torch.cuda.synchronize()
        paras_weight_buffer.release_all()

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()
