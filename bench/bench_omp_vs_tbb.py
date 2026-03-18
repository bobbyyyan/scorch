#!/usr/bin/env python3
"""Compare OpenMP vs TBB parallelization for SpMM on Intel hardware."""

import time
import statistics
import os
import numpy as np
import scipy.io
import scipy.sparse
import torch
from torch.utils.cpp_extension import load_inline

conda_prefix = os.environ.get("CONDA_PREFIX", "")
tbb_include = f"{conda_prefix}/include"
tbb_lib = f"{conda_prefix}/lib"

COMMON_CFLAGS = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]

# Shared kernel body as a macro to ensure identical computation
KERNEL_BODY = r"""
#include <torch/extension.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

static inline void spmm_row(
    int i, int N, int kTile,
    const int* __restrict__ pos,
    const int* __restrict__ crd,
    const float* __restrict__ aval,
    const float* __restrict__ bval,
    float* __restrict__ C,
    float* __restrict__ ws) {
  const int begin = pos[i];
  const int end = pos[i + 1];
  if (begin == end) return;
  float* __restrict__ C_row = C + (size_t)i * N;
  for (int k_out = 0; k_out < N; k_out += kTile) {
    const int kw = std::min(kTile, N - k_out);
    memset(ws, 0, kw * sizeof(float));
    for (int p = begin; p < end; p++) {
      const float a = aval[p];
      const float* __restrict__ B_row = bval + (size_t)crd[p] * N + k_out;
      if (p + 1 < end)
        __builtin_prefetch(bval + (size_t)crd[p+1] * N + k_out, 0, 1);
      for (int k = 0; k < kw; k++)
        ws[k] += a * B_row[k];
    }
    memcpy(C_row + k_out, ws, kw * sizeof(float));
  }
}
"""

SPMM_OMP = KERNEL_BODY + r"""
torch::Tensor spmm_omp(
    std::vector<int> result_shape,
    torch::Tensor A_pos, torch::Tensor A_crd, torch::Tensor A_val,
    torch::Tensor B_val_t, int B_cols) {
  const int M = result_shape[0], N = B_cols;
  const int kTile = std::min(N, 256);
  const int* pos = A_pos.data_ptr<int>();
  const int* crd = A_crd.data_ptr<int>();
  const float* aval = A_val.data_ptr<float>();
  const float* bval = B_val_t.data_ptr<float>();
  torch::Tensor C_t = torch::zeros({M, N}, torch::kFloat32);
  float* C = C_t.data_ptr<float>();

  #pragma omp parallel
  {
    float* ws = (float*)aligned_alloc(64, ((kTile+15)&~15) * sizeof(float));
    #pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < M; i++)
      spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    free(ws);
  }
  return C_t;
}
"""

SPMM_TBB_SRC = KERNEL_BODY + r"""
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

// TBB variant 1: blocked_range + auto_partitioner (default)
torch::Tensor spmm_tbb_auto(
    std::vector<int> result_shape,
    torch::Tensor A_pos, torch::Tensor A_crd, torch::Tensor A_val,
    torch::Tensor B_val_t, int B_cols) {
  const int M = result_shape[0], N = B_cols;
  const int kTile = std::min(N, 256);
  const int* pos = A_pos.data_ptr<int>();
  const int* crd = A_crd.data_ptr<int>();
  const float* aval = A_val.data_ptr<float>();
  const float* bval = B_val_t.data_ptr<float>();
  torch::Tensor C_t = torch::zeros({M, N}, torch::kFloat32);
  float* C = C_t.data_ptr<float>();
  const int aligned = ((kTile+15)&~15);

  tbb::enumerable_thread_specific<float*> tls([&]() {
    return (float*)aligned_alloc(64, aligned * sizeof(float));
  });
  tbb::parallel_for(
    tbb::blocked_range<int>(0, M, 64),
    [&](const tbb::blocked_range<int>& r) {
      float* ws = tls.local();
      for (int i = r.begin(); i < r.end(); i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    },
    tbb::auto_partitioner{}
  );
  for (auto& ws : tls) free(ws);
  return C_t;
}

// TBB variant 2: simple_partitioner with grain=64 (like OMP dynamic,64)
torch::Tensor spmm_tbb_simple(
    std::vector<int> result_shape,
    torch::Tensor A_pos, torch::Tensor A_crd, torch::Tensor A_val,
    torch::Tensor B_val_t, int B_cols) {
  const int M = result_shape[0], N = B_cols;
  const int kTile = std::min(N, 256);
  const int* pos = A_pos.data_ptr<int>();
  const int* crd = A_crd.data_ptr<int>();
  const float* aval = A_val.data_ptr<float>();
  const float* bval = B_val_t.data_ptr<float>();
  torch::Tensor C_t = torch::zeros({M, N}, torch::kFloat32);
  float* C = C_t.data_ptr<float>();
  const int aligned = ((kTile+15)&~15);

  tbb::enumerable_thread_specific<float*> tls([&]() {
    return (float*)aligned_alloc(64, aligned * sizeof(float));
  });
  tbb::parallel_for(
    tbb::blocked_range<int>(0, M, 64),
    [&](const tbb::blocked_range<int>& r) {
      float* ws = tls.local();
      for (int i = r.begin(); i < r.end(); i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    },
    tbb::simple_partitioner{}
  );
  for (auto& ws : tls) free(ws);
  return C_t;
}

// TBB variant 3: affinity_partitioner (cache reuse across calls)
static tbb::affinity_partitioner ap;
torch::Tensor spmm_tbb_affinity(
    std::vector<int> result_shape,
    torch::Tensor A_pos, torch::Tensor A_crd, torch::Tensor A_val,
    torch::Tensor B_val_t, int B_cols) {
  const int M = result_shape[0], N = B_cols;
  const int kTile = std::min(N, 256);
  const int* pos = A_pos.data_ptr<int>();
  const int* crd = A_crd.data_ptr<int>();
  const float* aval = A_val.data_ptr<float>();
  const float* bval = B_val_t.data_ptr<float>();
  torch::Tensor C_t = torch::zeros({M, N}, torch::kFloat32);
  float* C = C_t.data_ptr<float>();
  const int aligned = ((kTile+15)&~15);

  tbb::enumerable_thread_specific<float*> tls([&]() {
    return (float*)aligned_alloc(64, aligned * sizeof(float));
  });
  tbb::parallel_for(
    tbb::blocked_range<int>(0, M, 64),
    [&](const tbb::blocked_range<int>& r) {
      float* ws = tls.local();
      for (int i = r.begin(); i < r.end(); i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    },
    ap
  );
  for (auto& ws : tls) free(ws);
  return C_t;
}

// TBB variant 4: per-row scheduling (grain=1) — max load balance
torch::Tensor spmm_tbb_fine(
    std::vector<int> result_shape,
    torch::Tensor A_pos, torch::Tensor A_crd, torch::Tensor A_val,
    torch::Tensor B_val_t, int B_cols) {
  const int M = result_shape[0], N = B_cols;
  const int kTile = std::min(N, 256);
  const int* pos = A_pos.data_ptr<int>();
  const int* crd = A_crd.data_ptr<int>();
  const float* aval = A_val.data_ptr<float>();
  const float* bval = B_val_t.data_ptr<float>();
  torch::Tensor C_t = torch::zeros({M, N}, torch::kFloat32);
  float* C = C_t.data_ptr<float>();
  const int aligned = ((kTile+15)&~15);

  tbb::enumerable_thread_specific<float*> tls([&]() {
    return (float*)aligned_alloc(64, aligned * sizeof(float));
  });
  tbb::parallel_for(
    tbb::blocked_range<int>(0, M, 1),
    [&](const tbb::blocked_range<int>& r) {
      float* ws = tls.local();
      for (int i = r.begin(); i < r.end(); i++)
        spmm_row(i, N, kTile, pos, crd, aval, bval, C, ws);
    },
    tbb::auto_partitioner{}
  );
  for (auto& ws : tls) free(ws);
  return C_t;
}
"""

print("Compiling OpenMP kernel...")
omp_module = load_inline(
    name="spmm_omp_bench2",
    cpp_sources=[SPMM_OMP],
    functions=["spmm_omp"],
    extra_cflags=COMMON_CFLAGS + ["-fopenmp"],
    extra_ldflags=["-fopenmp"],
)

print("Compiling TBB kernels...")
tbb_module = load_inline(
    name="spmm_tbb_bench2",
    cpp_sources=[SPMM_TBB_SRC],
    functions=["spmm_tbb_auto", "spmm_tbb_simple", "spmm_tbb_affinity", "spmm_tbb_fine"],
    extra_cflags=COMMON_CFLAGS + [f"-I{tbb_include}"],
    extra_ldflags=[f"-L{tbb_lib}", "-ltbb"],
)

MATRICES = [
    "bcsstk17", "crystk02", "crystk03", "bcsstk30", "ct20stif",
    "gupta2", "mixtank_new", "mosfet2", "cfd2", "pkustk11",
    "finan512", "pwtk", "crankseg_1", "parabolic_fem",
    "pre2", "mouse_gene", "thermal2", "af_shell3",
    "inline_1", "ldoor", "audikw_1", "Flan_1565", "bone010",
]

K = 128
WARMUP = 5
REPEATS = 15

def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times) * 1e3

print(f"\nOpenMP vs TBB SpMM Benchmark (k={K}, warmup={WARMUP}, repeats={REPEATS})")
print(f"{'Matrix':20s} {'NNZ':>12s} {'MKL':>8s} {'OMP':>8s} {'TBB-auto':>8s} {'TBB-simp':>8s} {'TBB-affi':>8s} {'TBB-fine':>8s}  {'Best':>10s}")
print("-" * 110)

for name in MATRICES:
    try:
        mat = scipy.io.mmread(f"/scratch/suitesparse/{name}/{name}.mtx")
    except FileNotFoundError:
        continue
    csr = scipy.sparse.csr_matrix(mat, dtype=np.float32)
    M, N = csr.shape

    A_pos = torch.from_numpy(csr.indptr.astype(np.int32))
    A_crd = torch.from_numpy(csr.indices.astype(np.int32))
    A_val = torch.from_numpy(csr.data.astype(np.float32))
    B = torch.rand(N, K, dtype=torch.float32)
    rs = [M, K]

    torch_csr = torch.sparse_csr_tensor(A_pos, A_crd, A_val, size=csr.shape)
    _, pt_ms = bench(lambda: torch.sparse.mm(torch_csr, B))

    _, omp_ms = bench(omp_module.spmm_omp, rs, A_pos, A_crd, A_val, B, K)
    _, auto_ms = bench(tbb_module.spmm_tbb_auto, rs, A_pos, A_crd, A_val, B, K)
    _, simp_ms = bench(tbb_module.spmm_tbb_simple, rs, A_pos, A_crd, A_val, B, K)
    _, affi_ms = bench(tbb_module.spmm_tbb_affinity, rs, A_pos, A_crd, A_val, B, K)
    _, fine_ms = bench(tbb_module.spmm_tbb_fine, rs, A_pos, A_crd, A_val, B, K)

    results = {"OMP": omp_ms, "TBB-auto": auto_ms, "TBB-simp": simp_ms, "TBB-affi": affi_ms, "TBB-fine": fine_ms}
    best_name = min(results, key=results.get)
    best_ms = results[best_name]

    print(f"{name:20s} {csr.nnz:12,d} {pt_ms:8.2f} {omp_ms:8.2f} {auto_ms:8.2f} {simp_ms:8.2f} {affi_ms:8.2f} {fine_ms:8.2f}  {best_name:>10s}")

print()
