"""Standalone address math for the ParaS unified EP/TP buffer (prototype + proof).

Combined buffer, EP and TP share it (never live at once). Read per mode, each is
``weights | pad | cache`` in address order:
  * EP weights: forward from P.
  * EP cache:   forward from P + Σwe + PAD;  EP_end = end of EP cache.
  * TP weights: first layer at the head (P + we[0]), forward.
  * TP cache:   forward from (EP_end + max(ct) - Σct); last layer ends at EP_end + max(ct).

The cache tail anchor ``ANCHOR = max_i (ct_i + Σ_{k>i}(ct_k - ce_k))`` keeps every
layer's EP and TP cache disjoint, in ANY order (hybrid SWA + full attention). In
every real config ct_i <= ce_i (num_kv_heads divides tp_size or GQA-replicates, and
page_size >= 1), so the suffix sum is <= 0 and ANCHOR reduces to ``max(ct)``. The
general form is defensive headroom for the (non-occurring) ct_i > ce_i case and adds
no cost. Proof: EP cache before TP cache at layer i needs
``ct_i + Σ_{k>i}(ct_k - ce_k) <= ANCHOR``, which holds by construction. PAD is the
seam that also keeps TP weights off TP cache; it is
max(0, tp_w_end - w_end - Σce + Σct - ANCHOR). Uniform layers give overhead
max(SE, CT) over the per-mode total B. Transfer orders: EP->TP cache N-1..0 then
weights N-1..0; TP->EP weights 0..N-1 then cache 0..N-1.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

ALIGN = 256


def _au(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def _ov(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def compute_layout(
    we: List[int], wt: List[int], ce: List[int], ct: List[int],
    align: int = ALIGN, P: int = 0,
) -> dict:
    N = len(we)
    assert len(wt) == len(ce) == len(ct) == N and N > 0
    we = [_au(x, align) for x in we]
    wt = [_au(x, align) for x in wt]
    ce = [_au(x, align) for x in ce]
    ct = [_au(x, align) for x in ct]
    sum_we, sum_wt, sum_ce, sum_ct = sum(we), sum(wt), sum(ce), sum(ct)

    anchor, suffix = 0, 0
    for i in range(N - 1, -1, -1):
        anchor = max(anchor, ct[i] + suffix)
        suffix += ct[i] - ce[i]
    anchor = _au(anchor, align)

    w_end = P + sum_we
    tp_w_end = P + we[0] + sum_wt
    PAD = _au(max(0, tp_w_end - w_end - sum_ce + sum_ct - anchor), align)
    EP_end = w_end + PAD + sum_ce
    tc_end = EP_end + anchor

    addr: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    off = P
    for i in range(N):
        addr[("ep", i, "w")] = (off, we[i]); off += we[i]
    off = P + we[0]
    for i in range(N):
        addr[("tp", i, "w")] = (off, wt[i]); off += wt[i]
    off = w_end + PAD
    for i in range(N):
        addr[("ep", i, "c")] = (off, ce[i]); off += ce[i]
    assert off == EP_end

    tp_c_start = tc_end - sum_ct
    off = tp_c_start
    for i in range(N):
        addr[("tp", i, "c")] = (off, ct[i]); off += ct[i]
    assert off == tc_end

    return {
        "addr": addr, "N": N, "P": P, "PAD": PAD, "anchor": anchor, "EP_end": EP_end,
        "buffer_bytes": tc_end - P,
        "ep_total": sum_we + sum_ce, "tp_total": sum_wt + sum_ct,
        "tp_w_end": tp_w_end, "tp_c_start": tp_c_start,
    }


def check_safe(L: dict) -> None:
    addr, N = L["addr"], L["N"]
    assert L["tp_w_end"] <= L["tp_c_start"], (L["tp_w_end"], L["tp_c_start"])
    for mode in ("ep", "tp"):
        for blk in ("w", "c"):
            prev = None
            for i in range(N):
                o, s = addr[(mode, i, blk)]
                assert o % ALIGN == 0
                if prev is not None:
                    assert o == prev, f"{mode}.{blk} not contiguous at {i}"
                prev = o + s
    assert addr[("tp", 0, "w")][0] == L["P"] + addr[("ep", 0, "w")][1]
    assert addr[("tp", N - 1, "c")][0] + addr[("tp", N - 1, "c")][1] == L["EP_end"] + L["anchor"]
    order = [("c", i) for i in range(N - 1, -1, -1)] + [("w", i) for i in range(N - 1, -1, -1)]
    unread = {(k, i): addr[("ep", i, k)] for (k, i) in order}
    for (k, i) in order:
        dst = addr[("tp", i, k)]
        for key, rng in unread.items():
            assert not _ov(dst, rng), f"EP->TP clobber tp {k}{i}{dst} vs ep {key}{rng}"
        del unread[(k, i)]
    order2 = [("w", i) for i in range(N)] + [("c", i) for i in range(N)]
    unread2 = {(k, i): addr[("tp", i, k)] for (k, i) in order2}
    for (k, i) in order2:
        dst = addr[("ep", i, k)]
        for key, rng in unread2.items():
            assert not _ov(dst, rng), f"TP->EP clobber ep {k}{i}{dst} vs tp {key}{rng}"
        del unread2[(k, i)]


if __name__ == "__main__":
    import random
    import sys

    MB = 1 << 20
    allok = True

    SE, N = 300 * MB, 8
    L = compute_layout([SE] * N, [SE] * N, [SE] * N, [SE] * N)
    a = L["addr"]
    for i in range(N):
        assert a[("ep", i, "w")][0] == i * SE
        assert a[("tp", i, "w")][0] == (i + 1) * SE
    check_safe(L)
    print("proof1  G=1 (N+1)-slot spacing: PASS")

    print("\nproof2  balanced uniform: safe AND buffer == B + max(SE, CT)")
    uniform = {
        "qwen3-30B N48 G2 ": ([300 * MB] * 48, [600 * MB] * 48, [950 * MB] * 48, [650 * MB] * 48),
        "qwen3-235B N94 G2": ([400 * MB] * 94, [800 * MB] * 94, [950 * MB] * 94, [550 * MB] * 94),
        "adversarial SE>CT": ([100 * MB] * 4, [200 * MB] * 4, [120 * MB] * 4, [20 * MB] * 4),
    }
    for name, (we, wt, ce, ct) in uniform.items():
        L = compute_layout(we, wt, ce, ct)
        try:
            check_safe(L); safe = True
        except AssertionError as e:
            safe = False; print(f"  {name} UNSAFE {e}")
        B = L["ep_total"]
        overhead = L["buffer_bytes"] - B
        expect = _au(max(we[0], ct[0]))
        ok = safe and L["ep_total"] == L["tp_total"] and overhead == expect
        allok = allok and ok
        print(f"  {name} {'PASS' if ok else 'FAIL'}  buf={L['buffer_bytes'] >> 20}MB  "
              f"B={B >> 20}MB  overhead={overhead >> 20}MB == max(SE,CT)={expect >> 20}MB")

    print("\nproof3  hybrid SWA+full in NATURAL order (anchor = EP_end + max(ct))")
    def hyb(pat, G=2, we=2 * MB, cf=40 * MB, ctf=30 * MB, cs=8 * MB, cts=6 * MB):
        ce = [cf if p == "F" else cs for p in pat]
        ct = [ctf if p == "F" else cts for p in pat]
        return [we] * len(pat), [we * G] * len(pat), ce, ct
    for pat in ("SFFS", "SSFFSS", "SSSFFFSSS", "FSFSFS", "FFFFSSSS"):
        L = compute_layout(*hyb(pat))
        try:
            check_safe(L)
            print(f"  {pat:10s} PASS  buf={L['buffer_bytes'] >> 20}MB  "
                  f"anchor={L['anchor'] >> 20}MB  overhead={(L['buffer_bytes'] - L['ep_total']) >> 20}MB")
        except AssertionError as e:
            allok = False; print(f"  {pat:10s} FAIL  {e}")

    random.seed(0)
    ok = 0
    for _ in range(20000):
        M = random.randint(1, 40)
        G = random.randint(1, 8)
        se = random.randint(1, 500) * ALIGN
        we = [se] * M
        wt = [se * G] * M
        cf = random.randint(200, 4000) * ALIGN
        ctf = random.randint(1, cf // ALIGN) * ALIGN
        cs = random.randint(1, cf // ALIGN) * ALIGN
        cts = random.randint(1, cs // ALIGN) * ALIGN
        ce, ct = [], []
        for _ in range(M):
            if random.random() < 0.5:
                ce.append(cf); ct.append(ctf)
            else:
                ce.append(cs); ct.append(cts)
        try:
            check_safe(compute_layout(we, wt, ce, ct)); ok += 1
        except AssertionError as e:
            allok = False
            print(f"  FUZZ FAIL {e}\n    ce={ce}\n    ct={ct}")
            break
    print(f"\nproof4  fuzz random SWA+full hybrid (natural order): {ok}/20000 safe")

    random.seed(1)
    ok2 = 0
    for _ in range(20000):
        M = random.randint(1, 40)
        G = random.randint(1, 8)
        se = random.randint(1, 500) * ALIGN
        we = [se] * M
        wt = [se * G] * M
        ce = [random.randint(1, 4000) * ALIGN for _ in range(M)]
        ct = [random.randint(1, c // ALIGN) * ALIGN for c in ce]
        try:
            check_safe(compute_layout(we, wt, ce, ct)); ok2 += 1
        except AssertionError as e:
            allok = False
            print(f"  FUZZ2 FAIL {e}\n    ce={ce}\n    ct={ct}")
            break
    print(f"proof5  fuzz arbitrary per-layer cache, natural order (ct<=ce): {ok2}/20000 safe")

    print("\nOVERALL:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
