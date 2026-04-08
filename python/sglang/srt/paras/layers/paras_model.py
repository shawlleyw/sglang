"""
Reusable ParaS model-level mixin.

CausalLM Integration Pattern:
===============================
For a CausalLM class to support ParaS, define these methods:

    def paras_configure_helper(self):
        torch.cuda.synchronize()
        paras_weight_buffer.release_all()

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()

Where ``self.model`` is the transformer body (inheriting ParaSModelMixin).
After the switch completes, paras_configure_helper() is called by @paras_func to
synchronize CUDA and free the temporary weight redistribution buffers.
"""

import torch

from sglang.srt.paras.utils import paras_func


class ParaSModelMixin:
    """
    Mixin that adds model-level ParaS layer-iteration and conversion strategies.

    The base class must provide:
      - self.layers — list/ModuleList of decoder layers supporting paras methods
    """

    def paras_configure_tp_naive(self, paras_tp_size: int, paras_tp_rank: int):
        """Sequential (non-overlapped) EP→TP conversion for all layers."""
        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_overlap(self, paras_tp_size: int, paras_tp_rank: int):
        """
        Overlapped EP→TP conversion using dual CUDA streams for pipelining.
        Overlaps the all-gather of layer i+1 with all-to-all of layer i.
        """
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()

        self.layers[0].paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
        last_layer_handles = self.layers[0].paras_configure_tp_mlp_all_gather(
            stream_1, [], async_op=True
        )
        nlayers = len(self.layers)
        for i, layer in enumerate(self.layers):
            not_last_layer = i < nlayers - 1
            if not_last_layer:
                next_layer = self.layers[i + 1]
                next_layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
                new_handles = next_layer.paras_configure_tp_mlp_all_gather(
                    stream_2, last_layer_handles, async_op=True
                )

            layer.paras_configure_tp_mlp_all_to_all(stream_1, last_layer_handles)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

            if not_last_layer:
                last_layer_handles = new_handles
                stream_1, stream_2 = stream_2, stream_1

    def paras_configure_helper(self):
        pass

    @paras_func
    def paras_configure_tp(
        self, paras_tp_size: int, paras_tp_rank: int, overlap: bool = False
    ):
        """
        Configure the model for tensor parallelism.
        Note: the embedding layer stays in DP mode, which also works for TP.
        """
        if overlap:
            self.paras_configure_tp_overlap(paras_tp_size, paras_tp_rank)
        else:
            self.paras_configure_tp_naive(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        """Configure all layers back to EP mode."""
        for layer in self.layers:
            layer.paras_configure_ep()
