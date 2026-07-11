#!/usr/bin/env python3
r"""bench_tileijk_vs_tilej.py — the wide-B crossover, measured on the SHIPPED
prebuilt kernels (not the load_inline prototype).

Compares scorch_ops.spmm_csr_float_v2 (none) vs spmm_csr_float_tilej (column-panel)
vs spmm_csr_float_tileijk (B width-panel relayout) on a scattered high-degree
synthetic across a widening free dim N. The point: tile-j's output re-traffic grows
~N^2 so it erodes as N widens; tile-ijk relays B into contiguous Nc-strips so its
C-traffic stays ~N and it holds throughput — the crossover this productionizes.

CRITICAL: the tile-ijk relayout is paid INSIDE the kernel per call here (exactly as
the runtime probe times it), so this is the HONEST one-shot cost, not the prototype's
hoisted relayout. Nc/Jc come from the shipped selector's _ijk_params, so the numbers
are what scorch.matmul would actually route to.

Env: WT_M (rows, default 20000), WT_DEG (default 200),
     WT_NS (default 512,1024,2048,4096,8192).
"""
from __future__ import annotations
import os, sys, time, statistics, random
if os.path.isdir("/scratch/bobbyy"):
    os.environ.setdefault("HOME", "/scratch/bobbyy")
import numpy as np, scipy.sparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorch_ops as ops
from scorch import tiling


def csr_args(csr):
    return (torch.from_numpy(csr.indptr.astype(np.int32)),
            torch.from_numpy(csr.indices.astype(np.int32)),
            torch.from_numpy(csr.data.astype(np.float32)))


def rand_scatter(M, deg, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = rng.integers(0, M, size=M * deg, dtype=np.int64)
    data = rng.standard_normal(M * deg).astype(np.float32)
    c = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    c.sum_duplicates(); c.sort_indices(); return c


def timed(thunks, warmup=2, min_rounds=3, max_rounds=6, budget=3.0):
    K = len(thunks)
    for _ in range(warmup):
        for th in thunks: th()
    T = [[] for _ in range(K)]; order = list(range(K)); rng = random.Random(7); r = 0
    while r < max_rounds:
        rng.shuffle(order)
        for idx in order:
            t0 = time.perf_counter(); thunks[idx](); T[idx].append(time.perf_counter() - t0)
        r += 1
        if r >= min_rounds and sum(sum(t) for t in T) >= budget: break
    return [statistics.median(t) for t in T]


def main():
    M = int(os.environ.get("WT_M", 20000)); deg = int(os.environ.get("WT_DEG", 200))
    Ns = [int(x) for x in os.environ.get("WT_NS", "512,1024,2048,4096,8192").split(",")]
    C = tiling.query_llc()
    csr = rand_scatter(M, deg); J = M; nnz = csr.nnz; rs = [M, J]
    Ap, Ac, Av = csr_args(csr)
    print(f"[cfg] C_LLC={C/1e6:.1f}MB  A: M={M} nnz={nnz} ({nnz/M:.0f}/row) scattered", flush=True)
    print(f"\n{'N':>6}  {'B(MB)':>7}  {'none':>18}  {'tile-j':>18}  {'tile-ijk':>26}  {'ijk/tj':>7}  relerr", flush=True)

    for N in Ns:
        torch.manual_seed(0); B = torch.rand(J, N, dtype=torch.float32)
        ref = torch.from_numpy(csr.astype(np.float64) @ B.double().numpy()); rn = ref.norm().item() + 1e-30
        flops = 2.0 * nnz * N
        Jc_tj = tiling._panel_width(N, C)
        Nc, Jc_ijk = tiling._ijk_params(N, M, J, C)

        def f_none():
            return ops.spmm_csr_float_v2([M, N], [M, J], [[], [Ap, Ac]], Av,
                                         [J, N], [[], []], B, 256, -1, False).storage.value.reshape(M, N)

        def f_tj():
            return ops.spmm_csr_float_tilej([M, N], [M, J], [[], [Ap, Ac]], Av,
                                            [J, N], [[], []], B, Jc_tj, -1).storage.value.reshape(M, N)

        def f_ijk():
            return ops.spmm_csr_float_tileijk([M, N], [M, J], [[], [Ap, Ac]], Av,
                                              [J, N], [[], []], B, Nc, Jc_ijk, -1).storage.value.reshape(M, N)

        worst = max((f().double() - ref).norm().item() / rn for f in (f_none, f_tj, f_ijk))
        t_none, t_tj, t_ijk = timed([f_none, f_tj, f_ijk])
        g_none, g_tj, g_ijk = flops / t_none / 1e9, flops / t_tj / 1e9, flops / t_ijk / 1e9
        wsB = J * N * 4
        print(f"{N:6d}  {wsB/1e6:7.0f}  {g_none:8.1f} GFLOP/s  "
              f"{g_tj:8.1f} GFLOP/s  {g_ijk:8.1f} GFLOP/s (Nc={Nc},Jc={Jc_ijk})  "
              f"{g_ijk/g_tj:6.2f}x  {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()
