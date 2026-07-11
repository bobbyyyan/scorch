#!/usr/bin/env python3
r"""bench_tiling_noregress.py — the Phase-B no-regression + opportunity grid for
the adaptive tiling selector wired into scorch.matmul (spmm_csr_float_v2 -> tile-j).

For every matrix x N it times scorch.matmul with the selector ON vs OFF (toggling
tiling._ENABLED in-process, so thermal drift cancels) and reports the ratio +
correctness + which route the selector chose. The CLAUDE.md gate: NEUTRAL (~1.0)
on everything the selector should not touch (FEM/banded, low-degree, small-operand
graphs, AE weights), and a WIN only where it fires (high-degree scattered:
reddit-class). A ratio < 0.97 on any 'v2' route is a REGRESSION.

Covers, with no torch_geometric/ogb dependency:
  * SuiteSparse cache (~/.cache/scorch_suitesparse): FEM (bcsstk/crystk/ct20stif)
    + real scattered graphs (ca-*/email-Enron/p2p-*).
  * reddit (loaded straight from data/reddit/raw/*.npz).
  * sparse-AE weights (weights/autoencoder_*.pt) as W @ X SpMMs.
  * synthetic band / scatter16 / scatter200.
"""
from __future__ import annotations
import os, sys, glob, time, statistics
import numpy as np, scipy.sparse, scipy.io, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorch
from scorch import tiling
import bench_tilej_vs_v2 as T
import bench_tiling_autotuner as A

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_st(csr):
    return scorch.STensor.from_torch(torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)), size=csr.shape))


_CONTROL = os.environ.get("NR_CONTROL", "0") == "1"


def bench_ab(call, w=3, r=15):
    """INTERLEAVED rotated-round A/B (cancels thermal/contention drift on a shared
    box — the timing gotcha that makes a sequential all-on-then-all-off comparison
    show phantom regressions on sub-ms kernels). Toggles tiling._ENABLED per call so
    on/off alternate within each round. Returns (t_on_med, t_off_med). The probe
    fires once during the ON warmup, then memoizes -> timed ON calls run the chosen
    route (v2 byte-identical to OFF for a v2-route).

    NR_CONTROL=1 forces BOTH arms to _ENABLED=False (identical v2 code both sides),
    so the reported 'ratio' is the pure measurement NOISE FLOOR for that shape — any
    v2-route 'regression' at or above this floor is noise, not a real slowdown."""
    on_flag = False if _CONTROL else True
    tiling._ENABLED = on_flag
    for _ in range(w):
        call()
    tiling._ENABLED = False
    for _ in range(w):
        call()
    ton, toff = [], []
    for _ in range(r):
        tiling._ENABLED = on_flag
        t0 = time.perf_counter(); call(); ton.append(time.perf_counter() - t0)
        tiling._ENABLED = False
        t0 = time.perf_counter(); call(); toff.append(time.perf_counter() - t0)
    tiling._ENABLED = True
    return statistics.median(ton), statistics.median(toff)


def load_ss(name):
    # portable across machines: A.m_suitesparse probes /scratch/suitesparse and
    # the M5 caches (~/.cache/scorch/ss_full_envelope, ~/.cache/scorch_suitesparse).
    return A.m_suitesparse(name)


def ae_weight_mats():
    out = []
    for pt in sorted(glob.glob(os.path.join(REPO, "weights", "autoencoder_*.pt"))):
        name = os.path.basename(pt)[len("autoencoder_"):-3]
        try:
            sd = torch.load(pt, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(pt, map_location="cpu")
        for k, W in sd.items():
            if not (isinstance(W, torch.Tensor) and W.dim() == 2):
                continue
            if min(W.shape) < 256:
                continue
            # magnitude-prune to 0.99 sparse (the regime AE ships at)
            Wc = W.clone().float().flatten()
            thr = torch.quantile(Wc.abs()[torch.randint(0, Wc.numel(), (min(200000, Wc.numel()),))], 0.99)
            Wm = W.float().clone(); Wm[Wm.abs() < thr] = 0.0
            sp = scipy.sparse.csr_matrix(Wm.numpy()); sp.eliminate_zeros(); sp.sort_indices()
            if sp.nnz == 0:
                continue
            out.append((f"AE:{name}:{k}:{tuple(W.shape)}", sp))
    return out


def main():
    Ns = [int(x) for x in os.environ.get("NR_NS", "64,256").split(",")]
    grid = []
    # synthetics (regime anchors)
    grid.append(("band16", A.m_band(40000, 16)))
    grid.append(("scatter16", A.m_scatter(40000, 16)))
    grid.append(("scatter200", A.m_scatter(30000, 200)))
    # SuiteSparse cache (FEM + real graphs)
    ss_names = os.environ.get("NR_SS", "bcsstk30,bcsstk31,ct20stif,crystk03,"
                              "ca-AstroPh,ca-CondMat,ca-HepPh,email-Enron,"
                              "p2p-Gnutella31,cond-mat,wb-cs-stanford").split(",")
    for nm in ss_names:
        m = load_ss(nm)
        if m is not None and m.shape[0] == m.shape[1]:
            grid.append((f"ss:{nm}", m))
    # AE weights
    if os.environ.get("NR_AE", "1") == "1":
        grid += ae_weight_mats()
    # reddit (the target win)
    try:
        grid.append(("reddit", T.m_reddit_direct()))
    except Exception as ex:
        print(f"[skip reddit] {ex}")

    print(f"C(gate)={tiling.query_llc()/1e6:.1f}MB  DEG_FLOOR={tiling._DEG_FLOOR}  "
          f"LOC_MIN={tiling._LOC_MIN}  Ns={Ns}\n")
    hdr = f"{'matrix':30s}{'M':>9s}{'deg':>6s}{'N':>6s}  {'route':>7s} {'ratio':>7s} {'relerr':>9s}  verdict"
    print(hdr); print("-" * len(hdr))
    worst_v2 = 1e9
    regressions = []
    for nm, csr in grid:
        M, J = csr.shape
        deg = csr.nnz / M
        A_st = to_st(csr)
        for N in Ns:
            if J * N > 400_000_000:   # cap B memory for the giant graphs
                continue
            torch.manual_seed(0); B = torch.rand(J, N, dtype=torch.float32)
            ref = torch.from_numpy(csr @ B.numpy())
            Bs = scorch.STensor.from_torch(B)
            tiling._decision.clear()
            tiling._ENABLED = True
            out = scorch.matmul(A_st, Bs)   # primes the probe/memo + gives route
            route = list(tiling._decision.values())
            route = route[0][0] if route else "v2"
            r = out if isinstance(out, torch.Tensor) else out.to_torch()
            err = (r.double() - ref.double()).norm().item() / (ref.double().norm().item() + 1e-30)
            t_on, t_off = bench_ab(lambda: scorch.matmul(A_st, Bs))
            ratio = t_off / t_on   # >1 = selector faster
            if route == "v2":
                worst_v2 = min(worst_v2, ratio)
                verd = "neutral" if ratio >= 0.97 else "REGRESSION"
                if ratio < 0.97:
                    regressions.append((nm, N, ratio))
            else:
                verd = "WIN" if ratio > 1.03 else ("~tie" if ratio >= 0.97 else "LOSS?")
            ok = "OK" if err < 1e-3 else "CORR-FAIL"
            print(f"{nm[:30]:30s}{M:9d}{deg:6.0f}{N:6d}  {route:>7s} {ratio:6.2f}x {err:9.1e}  {verd} {ok}")
    # ---- WIDE-N tile-ijk opportunity block --------------------------------
    # No current scorch workload has N>=512, so tile-ijk never fires in the grid
    # above (N<=256). To exercise it we run the SCATTERED synthetics (the reddit /
    # general-library analog) at wide N, where tile-j's ~N^2 output re-traffic
    # erodes and the width-panel relayout wins. These matrices are small enough (M
    # ~20-30K) that J*N stays under the memory cap even at N=4096. v2 stays the
    # probe baseline, so this is still no-regression: a 'tileijk'/'tilej' route is
    # only chosen when the probe measured it faster than v2.
    wide_Ns = [int(x) for x in os.environ.get("NR_WIDE_NS", "2048,4096").split(",") if x]
    if wide_Ns:
        wide_grid = [("scatter200 (wide)", A.m_scatter(20000, 200)),
                     ("scatter120 (wide)", A.m_scatter(30000, 120, seed=2))]
        print("\n" + "=" * 60)
        print(f"WIDE-N tile-ijk opportunity (Ns={wide_Ns})")
        print(hdr); print("-" * len(hdr))
        for nm, csr in wide_grid:
            M, J = csr.shape
            deg = csr.nnz / M
            A_st = to_st(csr)
            for N in wide_Ns:
                if J * N > 400_000_000:
                    continue
                torch.manual_seed(0); B = torch.rand(J, N, dtype=torch.float32)
                ref = torch.from_numpy(csr @ B.numpy())
                Bs = scorch.STensor.from_torch(B)
                tiling._decision.clear()
                tiling._ENABLED = True
                out = scorch.matmul(A_st, Bs)
                route = list(tiling._decision.values())
                route = route[0][0] if route else "v2"
                r = out if isinstance(out, torch.Tensor) else out.to_torch()
                err = (r.double() - ref.double()).norm().item() / (ref.double().norm().item() + 1e-30)
                t_on, t_off = bench_ab(lambda: scorch.matmul(A_st, Bs))
                ratio = t_off / t_on
                if route == "v2":
                    worst_v2 = min(worst_v2, ratio)
                    verd = "neutral" if ratio >= 0.97 else "REGRESSION"
                    if ratio < 0.97:
                        regressions.append((nm, N, ratio))
                else:
                    verd = "WIN" if ratio > 1.03 else ("~tie" if ratio >= 0.97 else "LOSS?")
                ok = "OK" if err < 1e-3 else "CORR-FAIL"
                print(f"{nm[:30]:30s}{M:9d}{deg:6.0f}{N:6d}  {route:>7s} {ratio:6.2f}x {err:9.1e}  {verd} {ok}")

    print("\n" + "=" * 60)
    print(f"worst v2-route ratio (must be >=0.97): {worst_v2:.3f}")
    if regressions:
        print("REGRESSIONS:", regressions)
    else:
        print("NO REGRESSIONS on any v2 route.")


if __name__ == "__main__":
    main()
