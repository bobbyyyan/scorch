#!/usr/bin/env python3
r"""bench_tilej_vs_v2.py — the Phase-B GATE check: does a materialization-free
tile-j SpMM beat the REAL production kernel (spmm_csr_float_v2, with its
regtile/ILP/prefetch), not the naive `none`, on the high-degree graphs?

The autotuner oracle compares tile-j against a *naive* row-major `none`. But
production ships spmm_csr_float_v2 (2-nnz ILP + AVX2/NEON regtile + prefetch +
E-core recruit). tile-j only earns a place in the dispatch if it beats v2 on the
shapes the cheap gate would fire (reddit/products-class: J*4N>C and high degree).
This script measures scorch.matmul (=v2) vs a prototype tile-j across a small grid
and prints the speedup + correctness. Materialization-free tile-j: for each column
panel [j0,j0+Jc) sweep all rows, processing only each row's crd-slice in the panel
(binary search on the column-sorted CSR row) -> B panel [Jc x N] stays cache-hot,
reused across rows. No CSC, no panel matrices. This IS the kernel we would port to
csrc/spmm.h.
"""
from __future__ import annotations
import os, sys, time, math, statistics, platform
import numpy as np, scipy.sparse, torch
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorch  # noqa: E402  (imports torch first internally; loads scorch_ops)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def detect_llc():
    env = os.environ.get("SCORCH_LLC_BYTES")
    if env:
        return int(env)
    if platform.system() == "Darwin":
        return 24 * 1024 * 1024
    return 36 * 1024 * 1024


C_LLC = detect_llc()

CPP = r"""
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#ifndef SCORCH_RESTRICT
#define SCORCH_RESTRICT __restrict__
#endif
static inline int et(int nt){ return nt>0?nt:omp_get_max_threads(); }

// materialization-free tile-j: column-panel blocking straight off CSR.
// C must be pre-zeroed by the caller (accumulate across panels).
// For each panel [j0,j0+Jc): sweep all rows; for row i, process only the nnz
// whose crd in [j0,j0+Jc) (a contiguous slice since CSR rows are col-sorted),
// found by lower_bound. B rows for the panel (<=Jc distinct) stay cache-resident
// and are reused across the M rows -> that's the recovered reuse.
torch::Tensor tilej(std::vector<int> rs, torch::Tensor Ap, torch::Tensor Ac,
    torch::Tensor Av, torch::Tensor Bv, int N, int Jc, int nt){
  const int M=rs[0], J=rs[1];
  const int* SCORCH_RESTRICT pos=Ap.data_ptr<int>();
  const int* SCORCH_RESTRICT crd=Ac.data_ptr<int>();
  const float* SCORCH_RESTRICT av=Av.data_ptr<float>();
  const float* SCORCH_RESTRICT bv=Bv.data_ptr<float>();
  torch::Tensor Ct=torch::zeros({M,N},torch::kFloat32);
  float* SCORCH_RESTRICT C=Ct.data_ptr<float>();
  for(int j0=0;j0<J;j0+=Jc){
    const int j1 = std::min(j0+Jc, J);
    #pragma omp parallel for schedule(dynamic,64) num_threads(et(nt))
    for(int i=0;i<M;++i){
      const int b=pos[i], e=pos[i+1];
      if(b==e) continue;
      // slice [lo,hi) of row i with crd in [j0,j1)
      const int* rb = crd+b; const int* re = crd+e;
      int lo = (int)(std::lower_bound(rb,re,j0)-crd);
      int hi = (int)(std::lower_bound(crd+lo,re,j1)-crd);
      if(lo==hi) continue;
      float* SCORCH_RESTRICT cr = C+(size_t)i*N;
      for(int p=lo;p<hi;++p){
        const float a=av[p];
        const float* SCORCH_RESTRICT br = bv+(size_t)crd[p]*N;
        if(p+1<hi) __builtin_prefetch(bv+(size_t)crd[p+1]*N,0,1);
        for(int k=0;k<N;++k) cr[k]+=a*br[k];
      }
    }
  }
  return Ct;
}
"""


def build():
    from torch.utils.cpp_extension import load_inline
    from scorch.utils import get_extra_cflags, get_extra_ldflags
    cf = get_extra_cflags(["-O3", "-march=native", "-ffast-math", "-funroll-loops"])
    print(f"compiling tile-j prototype... ({platform.system()})", flush=True)
    return load_inline(name="tilej_proto", cpp_sources=[CPP], functions=["tilej"],
        extra_cflags=cf, extra_ldflags=get_extra_ldflags(), verbose=False)


def timed(thunks, warmup=2, min_rounds=3, max_rounds=8, budget=3.0):
    import random
    K = len(thunks)
    for _ in range(warmup):
        for th in thunks: th()
    T = [[] for _ in range(K)]; order = list(range(K)); rng = random.Random(7); r = 0
    while r < max_rounds:
        rng.shuffle(order)
        for idx in order:
            t0 = time.perf_counter(); thunks[idx](); T[idx].append(time.perf_counter()-t0)
        r += 1
        if r >= min_rounds and sum(sum(t) for t in T) >= budget: break
    return [statistics.median(t) for t in T]


def m_scatter(M, deg, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = rng.integers(0, M, size=M * deg, dtype=np.int64)
    data = rng.random(M * deg, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates(); x.sort_indices(); return x


def m_reddit_direct():
    """Load reddit's normalized adjacency straight from the PyG raw npz (COO edge
    list), no torch_geometric needed. Â = D̃^-1/2 (A+I) D̃^-1/2 (standard GCN)."""
    g = np.load(os.path.join(REPO, "data/reddit/raw/reddit_graph.npz"))
    n = int(g["shape"][0])
    row = g["row"].astype(np.int64); col = g["col"].astype(np.int64)
    A = scipy.sparse.csr_matrix((np.ones(row.size, np.float32), (row, col)), shape=(n, n))
    A = A + scipy.sparse.eye(n, dtype=np.float32, format="csr")  # self-loops
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = scipy.sparse.diags(dinv.astype(np.float32))
    Ahat = (D @ A @ D).tocsr().astype(np.float32)
    Ahat.sort_indices(); return Ahat


def m_gcn(name):
    if name == "reddit":
        return m_reddit_direct()
    import bench_gcn as G
    ds = G.load_dataset(name); a = G.compute_normalized_adj(ds.edge_index, ds.num_nodes)[0]
    x = scipy.sparse.csr_matrix((a.values().numpy().astype(np.float32), a.col_indices().numpy(),
        a.crow_indices().numpy().astype(np.int64)), shape=(ds.num_nodes, ds.num_nodes))
    x.sort_indices(); return x


def scorch_matmul_thunk(csr, B):
    # build an STensor CSR once
    import scorch
    A = scorch.STensor.from_torch(torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)),
        size=csr.shape))
    Bt = scorch.STensor.from_torch(B)
    return lambda: scorch.matmul(A, Bt)


def main():
    nt = int(os.environ.get("SCORCH_TILING_NT", "0")) or (os.cpu_count() if platform.system()=="Darwin" else 0)
    mod = build()
    names = os.environ.get("TJV2_MATS", "scatter200,reddit,ogbn-arxiv").split(",")
    Ns = [int(x) for x in os.environ.get("TJV2_NS", "64,128,256").split(",")]
    print(f"[cfg] NT={nt} C_LLC={C_LLC/1024/1024:.0f}MB Ns={Ns}", flush=True)
    for name in names:
        if name == "scatter200":
            csr = m_scatter(30000, 200)
        else:
            try:
                csr = m_gcn(name)
            except Exception as ex:
                print(f"[skip] {name}: {ex}", flush=True); continue
        M, J = csr.shape; nnz = csr.nnz
        Ap = torch.from_numpy(csr.indptr.astype(np.int32)); Ac = torch.from_numpy(csr.indices.astype(np.int32))
        Av = torch.from_numpy(csr.data.astype(np.float32)); rs = [M, J]
        print(f"\n##### {name}  M={M} nnz={nnz} ({nnz/M:.0f}/row) #####", flush=True)
        for N in Ns:
            torch.manual_seed(0); B = torch.rand(J, N, dtype=torch.float32)
            ref = torch.from_numpy(csr @ B.numpy())
            Jc = max(256, int(C_LLC / (4 * N)))
            v2_thunk = scorch_matmul_thunk(csr, B)
            tj_thunk = lambda: mod.tilej(rs, Ap, Ac, Av, B, N, Jc, nt)
            # correctness
            v2_out = v2_thunk(); tj_out = tj_thunk()
            v2r = (v2_out.to_torch() if hasattr(v2_out, "to_torch") else v2_out)
            e_v2 = (v2r.double() - ref.double()).norm().item() / (ref.double().norm().item()+1e-30)
            e_tj = (tj_out.double() - ref.double()).norm().item() / (ref.double().norm().item()+1e-30)
            t_v2, t_tj = timed([v2_thunk, tj_thunk])
            gf = lambda t: 2.0*nnz*N/t/1e9
            spd = t_v2 / t_tj
            flag = "TILE-J WINS" if spd > 1.03 else ("v2 wins" if spd < 0.97 else "~tie")
            wsB = J*4*N
            print(f"  N={N:5d} Jc={Jc:6d} (B/row-band {wsB/1e6:.0f}MB {'OVER' if wsB>C_LLC else 'fits'}-LLC) | "
                  f"v2 {gf(t_v2):6.1f}  tile-j {gf(t_tj):6.1f} GFLOP/s | {spd:.2f}x  {flag} "
                  f"| relerr v2={e_v2:.0e} tj={e_tj:.0e}", flush=True)


if __name__ == "__main__":
    main()
