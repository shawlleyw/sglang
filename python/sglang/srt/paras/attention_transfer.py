"""Direct attention transfers between disjoint per-layer EP/TP views."""

import triton
import triton.language as tl

from sglang.srt.paras.mode import ParaSMode


@triton.jit
def _restore_attention(
    peer_bases,
    output,
    source_offset: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    HEAD: tl.constexpr,
    T: tl.constexpr,
    RANK: tl.constexpr,
    IS_QKV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Interleave source peers instead of exhausting one peer's shard first.
    # A scalar peer base also preserves contiguous/vectorized remote loads.
    program = tl.program_id(0)
    # Rotate each destination's starting peer to avoid synchronized hotspots.
    peer = (program % T + RANK) % T
    x = (program // T) * BLOCK + tl.arange(0, BLOCK)
    q = Q // T
    kv = tl.maximum(HEAD, KV // T)
    replica = tl.maximum(1, T // (KV // HEAD))
    if IS_QKV:
        row, col = x // H, x % H
        is_q = row < q
        # Q uses every shard; duplicated K/V uses only its canonical replica.
        valid = (x < (q + 2 * kv) * H) & (is_q | (peer % replica == 0))
        destination_row = tl.where(
            is_q,
            peer * q + row,
            Q + ((row - q) // kv) * KV + (peer // replica) * kv + (row - q) % kv,
        )
        destination = destination_row * H + col
    else:
        valid = x < H * q
        destination = (x // q) * Q + peer * q + x % q
    base = tl.load(peer_bases + peer) + source_offset
    source = base.to(tl.pointer_type(tl.bfloat16))
    value = tl.load(source + x, valid, other=0)
    tl.store(output + destination, value, valid)


def transfer_attention(manager, layer_id, mode: ParaSMode, rank, peer_bases):
    """Transfer disjoint layer views; caller fences every rank after the layer.

    The CUDA restore pushes each canonical shard to all peers. No rank may
    consume the reconstructed weights until the collective fence completes.
    """
    spec = manager._unified_spec
    lp = f"{spec.prefix}.layers.{layer_id}.self_attn"
    h, d, t = spec.hidden_size, spec.head_dim, spec.tp_size
    q, kv = spec.num_heads * d, spec.num_kv_heads * d
    qs, ks = q // t, max(d, kv // t)
    kv_rank = rank // max(1, t // spec.num_kv_heads)
    for proj in ("qkv_proj", "o_proj"):
        ep_name, tp_name = f"{lp}.{proj}.weight", f"{lp}.{proj}.tp_weight"
        ep, tp = manager.get_view(ep_name), manager.get_view(tp_name)
        ep_offset = manager._entries[ep_name].offset_bytes
        tp_offset = manager._entries[tp_name].offset_bytes
        # The vectorized CUDA path copies eight BF16 elements per instruction.
        # Retain the existing path for shapes without this alignment.
        if (
            ep.element_size() == 2
            and tp.element_size() == 2
            and h % 8 == 0
            and qs % 8 == 0
            and d % 8 == 0
            and ep.data_ptr() % 16 == 0
            and tp.data_ptr() % 16 == 0
            and ep_offset % 16 == 0
            and tp_offset % 16 == 0
        ):
            import paras_peer_access_cuda as cuda

            base = tp.data_ptr() - tp_offset
            if mode == ParaSMode.EP:
                cuda.launch_attention_restore(
                    base, peer_bases, tp_offset, ep_offset, h, q, kv, d, t,
                    rank, proj == "qkv_proj", "push", 16384, 256, 1,
                )
            elif mode == ParaSMode.TP:
                cuda.launch_attention_slice(
                    base, ep_offset, tp_offset, h, q, kv, d, t, rank,
                    proj == "qkv_proj", 16384, 256,
                )
            else:
                raise ValueError(mode)
            continue
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
            _restore_attention[(triton.cdiv(tp.numel(), 4096) * t,)](
                peer_bases,
                ep,
                manager._entries[tp_name].offset_bytes,
                h,
                q,
                kv,
                d,
                t,
                rank,
                proj == "qkv_proj",
                4096,
            )
        else:
            raise ValueError(mode)
