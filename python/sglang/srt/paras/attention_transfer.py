"""Direct attention transfers between disjoint per-layer EP/TP views."""

import triton
import triton.language as tl

from sglang.srt.paras.mode import ParaSMode


@triton.jit
def _restore_attention(
    peer_bases,
    output,
    source_offset: tl.constexpr,
    count: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    HEAD: tl.constexpr,
    T: tl.constexpr,
    IS_QKV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < count
    q = Q // T
    kv = tl.maximum(HEAD, KV // T)
    if IS_QKV:
        row, col = x // H, x % H
        is_q = row < Q
        kv_row = (row - Q) % KV
        replica = tl.maximum(1, T // (KV // HEAD))
        rank = tl.where(is_q, row // q, kv_row // kv * replica)
        local_row = tl.where(is_q, row % q, q + (row - Q) // KV * kv + kv_row % kv)
        index = local_row * H + col
    else:
        row, col = x // Q, x % Q
        rank = col // q
        index = row * q + col % q
    base = tl.load(peer_bases + rank, valid, other=0) + source_offset
    source = base.to(tl.pointer_type(tl.bfloat16))
    value = tl.load(source + index, valid, other=0)
    tl.store(output + x, value, valid)


def transfer_attention(manager, layer_id, mode: ParaSMode, rank, peer_bases):
    """Caller fences all ranks after the complete layer, including MoE."""
    spec = manager._unified_spec
    lp = f"{spec.prefix}.layers.{layer_id}.self_attn"
    h, d, t = spec.hidden_size, spec.head_dim, spec.tp_size
    q, kv = spec.num_heads * d, spec.num_kv_heads * d
    qs, ks = q // t, max(d, kv // t)
    kv_rank = rank // max(1, t // spec.num_kv_heads)
    for proj in ("qkv_proj", "o_proj"):
        ep_name, tp_name = f"{lp}.{proj}.weight", f"{lp}.{proj}.tp_weight"
        ep, tp = manager.get_view(ep_name), manager.get_view(tp_name)
        if mode == ParaSMode.TP:
            if proj == "qkv_proj":
                tp[:qs].copy_(ep[rank * qs : (rank + 1) * qs])
                tp[qs : qs + ks].copy_(ep[q + kv_rank * ks : q + (kv_rank + 1) * ks])
                tp[qs + ks :].copy_(
                    ep[q + kv + kv_rank * ks : q + kv + (kv_rank + 1) * ks]
                )
            else:
                tp.copy_(ep[:, rank * qs : (rank + 1) * qs])
        elif mode == ParaSMode.EP:
            _restore_attention[(triton.cdiv(ep.numel(), 1024),)](
                peer_bases,
                ep,
                manager._entries[tp_name].offset_bytes,
                ep.numel(),
                h,
                q,
                kv,
                d,
                t,
                proj == "qkv_proj",
                1024,
            )
        else:
            raise ValueError(mode)
