"""
Reusable ParaS decoder layer mixin.
"""

from typing import Optional

from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.paras.utils import paras_func


class ParaSDecoderLayerMixin:
    """
    Mixin that manages dual LayerCommunicator contexts (EP and TP) for ParaS.

    The base class must provide:
      - self.self_attn — attention module (optionally with paras_configure_tp/ep)
      - self.mlp — MoE block (ParaSMoeBlockMixin instance)
      - self.layer_communicator — active LayerCommunicator
      - self.layer_scatter_modes — active LayerScatterModes
      - self.input_layernorm — RMSNorm
      - self.post_attention_layernorm — RMSNorm
      - self.layer_id (int)
      - self.config
      - self.is_layer_sparse (bool)
    """

    def paras_init_layer(
        self, config, layer_id: int, is_layer_sparse: bool, is_previous_layer_sparse: bool
    ):
        """
        Initialize dual EP/TP communicator contexts.
        Call this inside __init__ after LayerCommunicator is set up.
        """
        self.paras_ep_layer_communicator = self.layer_communicator
        self.paras_ep_layer_scatter_modes = self.layer_scatter_modes
        self.paras_tp_layer_communicator = None
        self.paras_tp_layer_scatter_modes = None
        # Store for later use in paras_configure_tp
        self._paras_is_previous_layer_sparse = is_previous_layer_sparse

    def paras_configure_helper(self):
        pass

    def paras_configure_tp_attn(self, paras_tp_size: int, paras_tp_rank: int):
        if hasattr(self.self_attn, "paras_configure_tp"):
            self.self_attn.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_mlp(self, paras_tp_size: int, paras_tp_rank: int):
        self.mlp.paras_configure_tp_all_gather()
        self.mlp.paras_configure_tp_all_to_all()

    def paras_configure_tp_mlp_all_gather(self, stream, handles, async_op=False):
        return self.mlp.paras_configure_tp_all_gather(stream, handles, async_op)

    def paras_configure_tp_mlp_all_to_all(self, stream, handles):
        return self.mlp.paras_configure_tp_all_to_all(stream, handles)

    def paras_configure_tp_mlp_peer_access(
        self, peer_ctx, transfer_plans, packed_plans, staging_suffix, stream, handles
    ):
        """Wrapper: delegate to mlp.paras_configure_tp_peer_access."""
        for handle in (handles or []):
            handle.wait()
        return self.mlp.paras_configure_tp_peer_access(
            peer_ctx, transfer_plans, packed_plans, staging_suffix, stream
        )

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        """Switch from EP to TP mode for this layer."""
        # Save current EP state
        self.paras_ep_layer_scatter_modes = self.layer_scatter_modes
        self.paras_ep_layer_communicator = self.layer_communicator

        # Config only: switch MoE experts to TP
        self.mlp.paras_configure_tp(paras_tp_size, paras_tp_rank)

        # Build TP LayerCommunicator if not cached
        if not self.paras_tp_layer_scatter_modes:
            self.paras_tp_layer_scatter_modes = LayerScatterModes.init_new(
                layer_id=self.layer_id,
                num_layers=self.config.num_hidden_layers,
                is_layer_sparse=self.is_layer_sparse,
                is_previous_layer_sparse=self._paras_is_previous_layer_sparse,
            )
            assert not self.paras_tp_layer_communicator
            self.paras_tp_layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.paras_tp_layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
            )

        # Swap to TP communicator
        assert self.paras_tp_layer_scatter_modes
        assert self.paras_tp_layer_communicator
        self.layer_scatter_modes = self.paras_tp_layer_scatter_modes
        self.layer_communicator = self.paras_tp_layer_communicator
        self.attn_tp_size = paras_tp_size
        self.attn_tp_rank = paras_tp_rank

    @paras_func
    def paras_configure_ep(self):
        """Switch from TP back to EP mode for this layer."""
        if hasattr(self.self_attn, "paras_configure_ep"):
            self.self_attn.paras_configure_ep()
        self.mlp.paras_configure_ep()

        # Revert to EP communicator
        assert (
            self.paras_ep_layer_scatter_modes is not None
        ), "EP scatter modes are not initialized"
        assert (
            self.paras_ep_layer_communicator is not None
        ), "EP communication context is not initialized"
        self.layer_scatter_modes = self.paras_ep_layer_scatter_modes
        self.layer_communicator = self.paras_ep_layer_communicator

        self.attn_tp_size = 1
        self.attn_tp_rank = 0
