#!/usr/bin/env python3
r"""verify_learned.py — end-to-end runtime verification of the `learned` autotune level.

Drives the REAL production path (scorch.matmul under scorch.autotune(level)) with the
SHIPPED per-machine model loaded, and reports, per (matrix, N), the steady-state
(memoized) per-call time under off / analytic / balanced / learned:

  * learned / off        -- speedup vs pure v2 (must be >= ~1: no regression)
  * learned / analytic   -- did the model beat the cost model? (the win)
  * learned / balanced   -- did the model match the probe's quality at no probe stall?

plus bit-correctness at every level and the kernel each level picked. Interleaved
rotated-round median timing (the cold-first / hybrid-turbo gotcha). A final NEUTRALITY
block confirms the 99% (operand<=C: GCN-small / AE) routes to v2 at the learned level
(is_candidate False) exactly like off.

Env: VL_MATS (comma list), VL_NS (comma list). Requires the model at
autotune-levels/models/model_<machine_id>.json (train it first).
"""
from __future__ import annotations
import os
import sys
import platform

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorch  # noqa: E402
import scorch.tiling as T  # noqa: E402
from collect_autotune_data import (  # noqa: E402  (reuse builders + timing)
    m_scatter, m_powerlaw, m_band, m_gcn, m_suitesparse, timed, _stensors, _extract)

LEVELS = ["off", "analytic", "balanced", "learned"]


def _pick(sig, level):
    # The memo keys on (signature, level, baseline_tag); "v2" is the drop-in-SpMM
    # baseline ops.matmul measures against.
    d = T._decision.get((sig, level, "v2"))
    if d is None:
        return "v2"
    k, p = d
    return k if k == "v2" else (f"{k}@{p}")


def bench_matrix(name, csr, N):
    M, J = csr.shape
    nnz = csr.nnz
    torch.manual_seed(0)
    B = torch.rand(J, N, dtype=torch.float32)
    ref = torch.from_numpy(csr.astype(np.float64) @ B.double().numpy())
    rn = ref.norm().item() + 1e-30
    A_st, B_st = _stensors(csr, B)
    sig = T._signature(A_st, N)

    def thunk(lvl):
        scorch.set_autotune(lvl)
        out = scorch.matmul(A_st, B_st)
        return out if isinstance(out, torch.Tensor) else out.to_torch()

    # correctness once per level
    errs = {}
    for lvl in LEVELS:
        scorch.set_autotune(lvl)
        out = thunk(lvl)
        errs[lvl] = (out.double() - ref.double()).norm().item() / rn
    worst = max(errs.values())

    meds = timed([lambda lvl=lvl: thunk(lvl) for lvl in LEVELS])
    t = dict(zip(LEVELS, meds))
    gf = {lvl: 2.0 * nnz * N / t[lvl] / 1e9 for lvl in LEVELS}
    picks = {lvl: _pick(sig, lvl) for lvl in LEVELS}
    print(f"  N={N:5d} | "
          f"off {gf['off']:6.0f}  analytic {gf['analytic']:6.0f}  "
          f"balanced {gf['balanced']:6.0f}  learned {gf['learned']:6.0f} GF/s | "
          f"L/off {t['off']/t['learned']:.2f}x  L/an {t['analytic']/t['learned']:.2f}x  "
          f"L/bal {t['balanced']/t['learned']:.2f}x | pick[learned]={picks['learned']} "
          f"pick[an]={picks['analytic']} | relerr={worst:.0e}", flush=True)
    return dict(matrix=name, N=N, l_over_off=t['off'] / t['learned'],
                l_over_an=t['analytic'] / t['learned'],
                l_over_bal=t['balanced'] / t['learned'], relerr=worst,
                pick_learned=picks['learned'])


def main():
    mid = T._machine_id()
    model = T._load_learned_model()
    print(f"[cfg] machine={mid} platform={platform.system()} "
          f"model={'LOADED' if model else 'MISSING (-> analytic fallback!)'} "
          f"widen={T._LEARNED_WIDEN} LLC={T.query_llc()/1e6:.0f}MB", flush=True)
    if model is None:
        print("  !! no per-machine model; train bench/train_autotune_model.py first", flush=True)

    default_mats = "scatter_deg50,scatter_deg200,powerlaw_avg200,reddit,ogbn-arxiv,cant"
    mats = os.environ.get("VL_MATS", default_mats).split(",")
    Ns = [int(x) for x in os.environ.get("VL_NS", "64,256,1024").split(",")]

    def build(nm):
        if nm.startswith("scatter_deg"):
            return m_scatter(30000 if int(nm[11:]) <= 200 else 16000, int(nm[11:]))
        if nm.startswith("powerlaw_avg"):
            return m_powerlaw(30000, int(nm[12:]))
        if nm.startswith("band_bw"):
            return m_band(40000, int(nm[7:]))
        if nm in ("reddit", "ogbn-arxiv", "ogbn-products"):
            return m_gcn(nm)
        return m_suitesparse(nm)

    rows = []
    for nm in mats:
        try:
            csr = build(nm)
        except Exception as ex:
            print(f"[skip] {nm}: {type(ex).__name__}: {ex}", flush=True); continue
        if csr is None:
            print(f"[skip] {nm}: not found", flush=True); continue
        M, J = csr.shape
        loc = T._locality_ratio(*_loc_args(csr))
        print(f"\n##### {nm}  M={M} J={J} nnz={csr.nnz} ({csr.nnz/M:.0f}/row) #####", flush=True)
        for N in Ns:
            if (J * 4.0 * N + M * 4.0 * N * 2) > 10 * (1 << 30):
                print(f"  N={N}: over mem budget, skip", flush=True); continue
            T._decision.clear()
            rows.append(bench_matrix(nm, csr, N))

    # ---- neutrality: the 99% (operand<=C -> v2 at every level) ----
    print("\n##### NEUTRALITY (operand<=C: GCN-small / AE -> v2 at learned) #####", flush=True)
    g = torch.Generator().manual_seed(0)
    for (M, J, deg, N, tag) in [(2708, 2708, 4, 64, "cora-like"),
                                (784, 256, 100, 256, "AE-like")]:
        A = torch.zeros(M, J, dtype=torch.float32)
        for r in range(M):
            cols = torch.randperm(J, generator=g)[:min(deg, J)]
            A[r, cols] = torch.randn(cols.numel(), generator=g)
        Bt = torch.randn(J, N, generator=g, dtype=torch.float32)
        A_st = scorch.STensor.from_torch(A.to_sparse_csr())
        B_st = scorch.STensor.from_torch(Bt)
        cand = T.is_candidate(A_st, B_st, level="learned")
        print(f"  {tag:10s} M={M} J={J} N={N}: is_candidate(learned)={cand} "
              f"(expect False -> byte-neutral v2)", flush=True)

    if rows:
        import statistics
        gm = lambda k: statistics.geometric_mean([max(1e-9, r[k]) for r in rows])
        print(f"\n===== SUMMARY ({len(rows)} cells) =====", flush=True)
        print(f"  geomean learned/off      = {gm('l_over_off'):.3f}  (>=1: no regression vs v2)", flush=True)
        print(f"  geomean learned/analytic = {gm('l_over_an'):.3f}  (>1: beats the cost model)", flush=True)
        print(f"  geomean learned/balanced = {gm('l_over_bal'):.3f}  (~1: matches the probe)", flush=True)
        print(f"  worst relerr = {max(r['relerr'] for r in rows):.0e}", flush=True)


def _loc_args(csr):
    """Adapt a scipy csr to the STensor _locality_ratio expects (build a throwaway A)."""
    A = scorch.STensor.from_torch(torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)), size=csr.shape))
    return A, int(csr.shape[1])


if __name__ == "__main__":
    main()
