"""
ParaS attention mixin for tensor parallelism switching.

This module provides the ParaSAttentionMixin class, which adds optional
EP↔TP attention switching capabilities to attention modules.
"""

from sglang.srt.paras.utils import paras_func


class ParaSAttentionMixin:
    """
    Optional mixin that adds ParaS EP↔TP attention switching.
    
    The base class must provide:
      - self.qkv_proj (QKVParallelLinear with paras_configure_tp/ep)
      - self.attn (RadixAttention with paras_configure_tp/ep) 
      - self.o_proj (RowParallelLinear with paras_configure_tp/ep)
      - self.total_num_heads (int)
      - self.total_num_kv_heads (int)
      - self.attn_tp_size (int)
    """
    
    def paras_configure_helper(self):
        """Update attention head counts based on current TP size."""
        self.q_size = self.qkv_proj.q_proj_shard_size
        self.kv_size = self.qkv_proj.kv_proj_shard_size
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.attn_tp_size
    
    @paras_func
    def paras_configure_tp(self, paras_tp_size, paras_tp_rank):
        """Configure attention for tensor parallelism mode."""
        self.attn_tp_rank = paras_tp_rank
        self.attn_tp_size = paras_tp_size
        self.qkv_proj.paras_configure_tp(paras_tp_size, paras_tp_rank)
        self.attn.paras_configure_tp(paras_tp_size, paras_tp_rank)
        self.o_proj.paras_configure_tp(paras_tp_size, paras_tp_rank)
    
    @paras_func
    def paras_configure_ep(self):
        """Configure attention for expert parallelism mode."""
        self.attn_tp_size = 1
        self.attn_tp_rank = 0
        self.qkv_proj.paras_configure_ep()
        self.attn.paras_configure_ep()
        self.o_proj.paras_configure_ep()
