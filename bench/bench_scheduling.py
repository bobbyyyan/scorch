#!/usr/bin/env python3
"""Find the optimal scheduling strategy across all matrix sizes."""

import time, statistics, os
import numpy as np, scipy.io, scipy.sparse
import torch
from torch.utils.cpp_extension import load_inline

conda_prefix = os.environ.get("CONDA_PREFIX", "")
tbb_include = f"{conda_prefix}/include"
tbb_lib = f"{conda_prefix}/lib"
CFLAGS = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]

BODY = r"""
#include <torch/extension.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <omp.h>

static inline void spmm_row(
    int i, int N, int kTile,
    const int* __restrict__ pos, const int* __restrict__ crd,
    const float* __restrict__ aval, const float* __restrict__ bval,
    float* __restrict__ C, float* __restrict__ ws) {
  const int begin = pos[i], end = pos[i + 1];
  if (begin == end) return;
  float* __restrict__ C_row = C + (size_t)i * N;
  for (int k_out = 0; k_out < N; k_out += kTile) {
    const int kw = std::min(kTile, N - k_out);
    memset(ws, 0, kw * sizeof(float));
    for (int p = begin; p < end; p++) {
      const float a = aval[p];
      const float* __restrict__ B_row = bval + (size_t)crd[p] * N + k_out;
      if (p + 1 < end) __builtin_prefetch(bval + (size_t)crd[p+1] * N + k_out, 0, 1);
      for (int k = 0; k < kw; k++) ws[k] += a * B_row[k];
    }
    memcpy(C_row + k_out, ws, kw * sizeof(float));
  }
}

#define SPMM_VARIANT(NAME, SCHEDULE)                                        \
torch::Tensor NAME(std::vector<int> rs, torch::Tensor Ap, torch::Tensor Ac,\
    torch::Tensor Av, torch::Tensor Bv, int K) {                           \
  const int M = rs[0], N = K, kTile = std::min(N, 256);                    \
  const int* pos = Ap.data_ptr<int>(); const int* crd = Ac.data_ptr<int>();\
  const float* aval = Av.data_ptr<float>();                                 \
  const float* bval = Bv.data_ptr<float>();                                 \
  torch::Tensor Ct = torch::zeros({M, N}, torch::kFloat32);                \
  float* C = Ct.data_ptr<float>();                                          \
  _Pragma("omp parallel") {                                                 \
    float* ws = (float*)aligned_alloc(64, ((kTile+15)&~15)*sizeof(float));  \
    SCHEDULE                                                                \
    for (int i = 0; i < M; i++) spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws); \
    free(ws);                                                               \
  }                                                                         \
  return Ct;                                                                \
}

SPMM_VARIANT(spmm_dyn16,   _Pragma("omp for schedule(dynamic, 16)"))
SPMM_VARIANT(spmm_dyn64,   _Pragma("omp for schedule(dynamic, 64)"))
SPMM_VARIANT(spmm_dyn256,  _Pragma("omp for schedule(dynamic, 256)"))

// Adaptive: chunk = clamp(nnz / (nthreads * 128), 16, 256)
torch::Tensor spmm_adaptive(std::vector<int> rs, torch::Tensor Ap, torch::Tensor Ac,
    torch::Tensor Av, torch::Tensor Bv, int K) {
  const int M = rs[0], N = K, kTile = std::min(N, 256);
  const int* pos = Ap.data_ptr<int>(); const int* crd = Ac.data_ptr<int>();
  const float* aval = Av.data_ptr<float>();
  const float* bval = Bv.data_ptr<float>();
  torch::Tensor Ct = torch::zeros({M, N}, torch::kFloat32);
  float* C = Ct.data_ptr<float>();
  const int nnz = pos[M];
  const int nthreads = omp_get_max_threads();
  const int chunk = std::max(16, std::min(256, nnz / (nthreads * 128)));
  _Pragma("omp parallel") {
    float* ws = (float*)aligned_alloc(64, ((kTile+15)&~15)*sizeof(float));
    _Pragma("omp for schedule(dynamic) nowait")
    for (int i = 0; i < M; i++) spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    free(ws);
  }
  return Ct;
}

// Adaptive with runtime chunk size (can't use schedule(dynamic, chunk) with variable,
// so we implement manual work-stealing with atomics)
torch::Tensor spmm_adaptive2(std::vector<int> rs, torch::Tensor Ap, torch::Tensor Ac,
    torch::Tensor Av, torch::Tensor Bv, int K) {
  const int M = rs[0], N = K, kTile = std::min(N, 256);
  const int* pos = Ap.data_ptr<int>(); const int* crd = Ac.data_ptr<int>();
  const float* aval = Av.data_ptr<float>();
  const float* bval = Bv.data_ptr<float>();
  torch::Tensor Ct = torch::zeros({M, N}, torch::kFloat32);
  float* C = Ct.data_ptr<float>();
  const int nnz = pos[M];
  const int nthreads = omp_get_max_threads();
  const int chunk = std::max(16, std::min(256, nnz / (nthreads * 128)));
  std::atomic<int> next_row{0};
  _Pragma("omp parallel") {
    float* ws = (float*)aligned_alloc(64, ((kTile+15)&~15)*sizeof(float));
    while (true) {
      int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= M) break;
      int end = std::min(start + chunk, M);
      for (int i = start; i < end; i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    }
    free(ws);
  }
  return Ct;
}
"""

TBB_SRC = r"""
#include <torch/extension.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

static inline void spmm_row(
    int i, int N, int kTile,
    const int* __restrict__ pos, const int* __restrict__ crd,
    const float* __restrict__ aval, const float* __restrict__ bval,
    float* __restrict__ C, float* __restrict__ ws) {
  const int begin = pos[i], end = pos[i + 1];
  if (begin == end) return;
  float* __restrict__ C_row = C + (size_t)i * N;
  for (int k_out = 0; k_out < N; k_out += kTile) {
    const int kw = std::min(kTile, N - k_out);
    memset(ws, 0, kw * sizeof(float));
    for (int p = begin; p < end; p++) {
      const float a = aval[p];
      const float* __restrict__ B_row = bval + (size_t)crd[p] * N + k_out;
      if (p + 1 < end) __builtin_prefetch(bval + (size_t)crd[p+1] * N + k_out, 0, 1);
      for (int k = 0; k < kw; k++) ws[k] += a * B_row[k];
    }
    memcpy(C_row + k_out, ws, kw * sizeof(float));
  }
}
""" + r"""
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

torch::Tensor spmm_tbb_best(std::vector<int> rs, torch::Tensor Ap, torch::Tensor Ac,
    torch::Tensor Av, torch::Tensor Bv, int K) {
  const int M = rs[0], N = K, kTile = std::min(N, 256);
  const int* pos = Ap.data_ptr<int>(); const int* crd = Ac.data_ptr<int>();
  const float* aval = Av.data_ptr<float>(); const float* bval = Bv.data_ptr<float>();
  torch::Tensor Ct = torch::zeros({M, N}, torch::kFloat32);
  float* C = Ct.data_ptr<float>();
  const int al = ((kTile+15)&~15);
  tbb::enumerable_thread_specific<float*> tls([&](){return (float*)aligned_alloc(64, al*sizeof(float));});
  tbb::parallel_for(tbb::blocked_range<int>(0, M, 64),
    [&](const tbb::blocked_range<int>& r) {
      float* ws = tls.local();
      for (int i = r.begin(); i < r.end(); i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    }, tbb::simple_partitioner{});
  for (auto& ws : tls) free(ws);
  return Ct;
}
"""

print("Compiling OMP variants...")
omp_mod = load_inline(name="sched_omp", cpp_sources=[BODY],
    functions=["spmm_dyn16","spmm_dyn64","spmm_dyn256","spmm_adaptive","spmm_adaptive2"],
    extra_cflags=CFLAGS+["-fopenmp"], extra_ldflags=["-fopenmp"])

print("Compiling TBB variant...")
tbb_mod = load_inline(name="sched_tbb", cpp_sources=[TBB_SRC],
    functions=["spmm_tbb_best"],
    extra_cflags=CFLAGS+[f"-I{tbb_include}"], extra_ldflags=[f"-L{tbb_lib}","-ltbb"])

MATRICES = [
    "bcsstk17","crystk02","crystk03","bcsstk30","ct20stif",
    "gupta2","mixtank_new","mosfet2","cfd2","pkustk11",
    "finan512","pwtk","crankseg_1","parabolic_fem",
    "pre2","mouse_gene","thermal2","af_shell3",
    "inline_1","ldoor","audikw_1","Flan_1565","bone010",
]

K, WARMUP, REPEATS = 128, 5, 15

variants = [
    ("dyn16",     omp_mod.spmm_dyn16),
    ("dyn64",     omp_mod.spmm_dyn64),
    ("dyn256",    omp_mod.spmm_dyn256),
    ("adaptive",  omp_mod.spmm_adaptive),
    ("adaptive2", omp_mod.spmm_adaptive2),
    ("tbb",       tbb_mod.spmm_tbb_best),
]

def bench(fn, *args):
    for _ in range(WARMUP): fn(*args)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(*args); times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3

header = f"{'Matrix':20s} {'NNZ':>12s} {'MKL':>7s}"
for name, _ in variants:
    header += f" {name:>8s}"
header += f"  {'Best':>8s}"
print(f"\nScheduling Strategy Benchmark (k={K})")
print(header)
print("-" * len(header))

import warnings; warnings.filterwarnings('ignore')
for mname in MATRICES:
    try: mat = scipy.io.mmread(f"/scratch/suitesparse/{mname}/{mname}.mtx")
    except: continue
    csr = scipy.sparse.csr_matrix(mat, dtype=np.float32)
    Ap = torch.from_numpy(csr.indptr.astype(np.int32))
    Ac = torch.from_numpy(csr.indices.astype(np.int32))
    Av = torch.from_numpy(csr.data.astype(np.float32))
    B = torch.rand(csr.shape[1], K, dtype=torch.float32)
    rs = [csr.shape[0], K]
    tcsr = torch.sparse_csr_tensor(Ap, Ac, Av, size=csr.shape)
    pt = bench(lambda: torch.sparse.mm(tcsr, B))

    results = {}
    for vname, fn in variants:
        results[vname] = bench(fn, rs, Ap, Ac, Av, B, K)

    best = min(results, key=results.get)
    row = f"{mname:20s} {csr.nnz:12,d} {pt:7.2f}"
    for vname, _ in variants:
        ms = results[vname]
        marker = " *" if vname == best else "  "
        row += f" {ms:7.2f}{marker}"
    row += f"  {best:>8s}"
    print(row)
