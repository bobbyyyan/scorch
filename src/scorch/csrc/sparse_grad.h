#pragma once

// Kernels for the gradients of a sparse x dense product, used by scorch/autograd.py.
//
// For out = A @ B with A sparse (m x n, CSR) and B dense (n x k):
//   dL/dB          = A^T @ G           an SpMM against A's transpose
//   dL/dA[i, j]    = G[i, :] . B[j, :] at the stored (i, j) only: a sampled dense-dense
//                                      product (SDDMM), one dot product per nonzero
// with G = dL/d(out). The SpMM runs on the ordinary drop-in path (a CSR operand for A^T,
// planned like any other); these are the two pieces that path does not have.
//
// Neither kernel validates index arrays. The caller (autograd.py) screens them through the
// call boundary's verdict memo before the first use and re-checks their identity and version
// counter before every later one, the same contract the plans in plan.h keep.

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "scorch_policy.h"

namespace scorch_native {

// ---------------------------------------------------------------------------------------------
// CSR transpose
//
// (positions, coordinates) of an m x n CSR matrix -> (positions, coordinates, perm) of its
// n x m transpose, where nonzero q of the transpose is nonzero perm[q] of the original, so the
// transpose's values are values[perm]. A counting sort on the column index, filled in row
// order, so it is stable: each row of the transpose lists its column indices (the original's
// rows) in increasing order, which is the sorted CSR the kernels expect, and the result does
// not depend on how the work is split.
//
// Sequential. It runs once per sparsity structure -- autograd.py caches the result against the
// arrays' identity and version -- so its cost is paid on the first backward pass through a
// structure, not on every one.
template <typename I, typename O>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> csr_transpose_typed(
    const torch::Tensor& positions, const torch::Tensor& coordinates, int64_t rows, int64_t cols,
    torch::ScalarType out_index) {
  const I* SCORCH_RESTRICT pos = positions.data_ptr<I>();
  const I* SCORCH_RESTRICT crd = coordinates.data_ptr<I>();
  const int64_t nnz = static_cast<int64_t>(pos[rows]);
  auto index_options = torch::TensorOptions().dtype(out_index);
  torch::Tensor t_positions = torch::zeros({cols + 1}, index_options);
  torch::Tensor t_coordinates = torch::empty({nnz}, index_options);
  torch::Tensor perm = torch::empty({nnz}, torch::TensorOptions().dtype(torch::kInt64));
  O* SCORCH_RESTRICT tpos = t_positions.data_ptr<O>();
  O* SCORCH_RESTRICT tcrd = t_coordinates.data_ptr<O>();
  int64_t* SCORCH_RESTRICT tperm = perm.data_ptr<int64_t>();
  for (int64_t q = 0; q < nnz; ++q) ++tpos[static_cast<int64_t>(crd[q]) + 1];
  for (int64_t j = 0; j < cols; ++j) tpos[j + 1] += tpos[j];
  std::vector<O> cursor(tpos, tpos + cols);
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t end = static_cast<int64_t>(pos[i + 1]);
    for (int64_t q = static_cast<int64_t>(pos[i]); q < end; ++q) {
      const O dst = cursor[static_cast<int64_t>(crd[q])]++;
      tcrd[dst] = static_cast<O>(i);
      tperm[dst] = q;
    }
  }
  return {t_positions, t_coordinates, perm};
}

// The transpose's index arrays are int32 whenever its extents and nonzero count fit, which is
// the width every native kernel reads; int64 otherwise.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> csr_transpose(
    const torch::Tensor& positions, const torch::Tensor& coordinates, int64_t cols) {
  TORCH_CHECK(positions.dim() == 1 && coordinates.dim() == 1 && positions.numel() >= 1,
              "csr_transpose: positions and coordinates must be 1-D");
  TORCH_CHECK(positions.scalar_type() == coordinates.scalar_type() &&
                  (positions.scalar_type() == torch::kInt32 ||
                   positions.scalar_type() == torch::kInt64),
              "csr_transpose: index arrays must both be int32 or both int64");
  TORCH_CHECK(positions.is_contiguous() && coordinates.is_contiguous() &&
                  positions.device().is_cpu() && coordinates.device().is_cpu(),
              "csr_transpose: index arrays must be contiguous CPU tensors");
  TORCH_CHECK(cols >= 0, "csr_transpose: cols must be non-negative");
  const int64_t rows = positions.numel() - 1;
  const int64_t nnz = coordinates.numel();
  const bool narrow = nnz <= std::numeric_limits<int32_t>::max() &&
                      rows <= std::numeric_limits<int32_t>::max() &&
                      cols < std::numeric_limits<int32_t>::max();
  const torch::ScalarType out = narrow ? torch::kInt32 : torch::kInt64;
  if (positions.scalar_type() == torch::kInt32)
    return narrow ? csr_transpose_typed<int32_t, int32_t>(positions, coordinates, rows, cols, out)
                  : csr_transpose_typed<int32_t, int64_t>(positions, coordinates, rows, cols, out);
  return narrow ? csr_transpose_typed<int64_t, int32_t>(positions, coordinates, rows, cols, out)
                : csr_transpose_typed<int64_t, int64_t>(positions, coordinates, rows, cols, out);
}

// ---------------------------------------------------------------------------------------------
// The values of a transposed structure
//
// out[q] = values[perm[q]], with perm from csr_transpose: the values its nonzeros hold.
// autograd.py runs x @ A as (A^T @ x^T)^T and refills these on every call (a write to A's values
// through .data moves no version counter, so a refill only when a stamp moved can serve stale
// values). torch's index_select took 0.26-0.63 ns per element for it on an M5 at any thread
// count, 0.33 ns even for the identity permutation; this loop is a copy with one indirect read.
// It splits across torch's pool, all of it, from SCORCH_POOL_SPLIT_MIN_ELEMS elements up, where
// waking the team is sure to pay for work at least as costly per element as a copy.
//
// perm is trusted, as csr_transpose produced it: a permutation of [0, values.numel()).
template <typename T>
static void gather_values_range(const T* SCORCH_RESTRICT values,
                                const int64_t* SCORCH_RESTRICT perm, T* SCORCH_RESTRICT out,
                                int64_t begin, int64_t end) {
  for (int64_t q = begin; q < end; ++q) out[q] = values[perm[q]];
}

inline void gather_values(const torch::Tensor& values, const torch::Tensor& perm,
                          torch::Tensor& out, int64_t nthreads_override = -1) {
  TORCH_CHECK(values.dim() == 1 && perm.dim() == 1 && out.dim() == 1,
              "gather_values: values, perm and out must be 1-D");
  TORCH_CHECK(values.numel() == perm.numel() && out.numel() == perm.numel(),
              "gather_values: values, perm and out must have one length");
  TORCH_CHECK(perm.scalar_type() == torch::kInt64, "gather_values: perm must be int64");
  TORCH_CHECK(values.scalar_type() == out.scalar_type() &&
                  (values.scalar_type() == torch::kFloat32 ||
                   values.scalar_type() == torch::kFloat64),
              "gather_values: values and out must both be float32 or both float64");
  TORCH_CHECK(values.is_contiguous() && perm.is_contiguous() && out.is_contiguous() &&
                  values.device().is_cpu() && perm.device().is_cpu() && out.device().is_cpu(),
              "gather_values: values, perm and out must be contiguous CPU tensors");
  const int64_t n = perm.numel();
  const int64_t pool = at::get_num_threads();
  const int64_t* p = perm.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "gather_values", [&] {
    const scalar_t* v = values.data_ptr<scalar_t>();
    scalar_t* o = out.data_ptr<scalar_t>();
    if (nthreads_override > 0 && pool > 1 && n >= SCORCH_POOL_SPLIT_MIN_ELEMS) {
      at::parallel_for(0, n, (n + pool - 1) / pool,
                       [&](int64_t b, int64_t e) { gather_values_range(v, p, o, b, e); });
    } else {
      gather_values_range(v, p, o, 0, n);
    }
  });
}

// ---------------------------------------------------------------------------------------------
// SDDMM on a CSR structure: out[q] = x[i, :] . y[crd[q], :] for every stored q of row i.
//
// Its memory traffic is an SpMM's with the direction of the dense rows reversed: each nonzero
// gathers one row of y (an SpMM gathers one row of B) and the row of x is reused across the
// nonzeros of row i (an SpMM's output row is). So it takes the SpMM's thread rule and work
// measure, and launches exactly as the drop-in SpMM does (scorch_spmm_full_team and
// scorch_spmm_on_torch_pool in scorch_policy.h, the launch in spmm_csr_v2_core): on torch's own
// pool when the caller runs a pipeline of torch ops, inline when one worker is enough, a private
// team otherwise. A training step alternates this kernel with torch's ops, and a private team of
// another width there costs a libgomp pool resize per call -- the first version of this kernel,
// an `omp parallel for` of its own width, lost to MKL on MKT for exactly that.
//
// Each worker takes a contiguous range of rows holding an equal share of the nonzeros: a row's
// cost is its nonzero count, the output is written per nonzero, so no worker shares a write and
// the result does not depend on how many workers there were.
template <typename I, typename T>
void sddmm_csr_rows(const I* SCORCH_RESTRICT pos, const I* SCORCH_RESTRICT crd,
                    const T* SCORCH_RESTRICT x, const T* SCORCH_RESTRICT y, T* SCORCH_RESTRICT out,
                    int64_t rows, int64_t k, int nthreads, bool on_torch_pool) {
  const int64_t nnz = static_cast<int64_t>(pos[rows]);
  // The first row of worker w's share: the first row whose nonzeros start at or after w/team of
  // them. The last worker also takes any empty rows at the end.
  auto first_row = [&](int64_t w, int64_t team) -> int64_t {
    if (w >= team) return rows;
    const int64_t target = nnz / team * w + nnz % team * w / team;
    return std::lower_bound(pos, pos + rows + 1, target,
                            [](I p, int64_t t) { return static_cast<int64_t>(p) < t; }) -
           pos;
  };
  auto worker = [&](int w, int team) {
    const int64_t lo = first_row(w, team), hi = first_row(w + 1, team);
    for (int64_t i = lo; i < hi; ++i) {
      const T* SCORCH_RESTRICT xi = x + i * k;
      const int64_t end = static_cast<int64_t>(pos[i + 1]);
      for (int64_t q = static_cast<int64_t>(pos[i]); q < end; ++q) {
        const T* SCORCH_RESTRICT yj = y + static_cast<int64_t>(crd[q]) * k;
        T acc = 0;
#pragma omp simd reduction(+ : acc)
        for (int64_t t = 0; t < k; ++t) acc += xi[t] * yj[t];
        out[q] = acc;
      }
    }
  };
  // The launch of spmm_csr_v2_core, step for step (see the measurements there).
  if (on_torch_pool) {
    if (nthreads >= 2 * at::get_num_threads()) {
#pragma omp parallel num_threads(nthreads)
      worker(omp_get_thread_num(), omp_get_num_threads());
    } else {
      at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {
        for (int64_t w = wbeg; w < wend; ++w) worker(static_cast<int>(w), nthreads);
      });
    }
  } else if (nthreads <= 1) {
    worker(0, 1);
  } else {
#pragma omp parallel num_threads(nthreads)
    worker(omp_get_thread_num(), omp_get_num_threads());
  }
}

inline torch::Tensor sddmm_csr(const torch::Tensor& positions, const torch::Tensor& coordinates,
                               const torch::Tensor& x, const torch::Tensor& y,
                               int64_t nthreads_override, bool atparallel) {
  TORCH_CHECK(positions.dim() == 1 && coordinates.dim() == 1 && positions.numel() >= 1,
              "sddmm_csr: positions and coordinates must be 1-D");
  TORCH_CHECK(positions.scalar_type() == coordinates.scalar_type() &&
                  (positions.scalar_type() == torch::kInt32 ||
                   positions.scalar_type() == torch::kInt64),
              "sddmm_csr: index arrays must both be int32 or both int64");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
              "sddmm_csr: x and y must be 2-D with the same number of columns");
  TORCH_CHECK(x.scalar_type() == y.scalar_type() &&
                  (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64),
              "sddmm_csr: x and y must both be float32 or both float64");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && positions.is_contiguous() &&
                  coordinates.is_contiguous(),
              "sddmm_csr: every operand must be contiguous");
  TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu() && positions.device().is_cpu() &&
                  coordinates.device().is_cpu(),
              "sddmm_csr: every operand must be on the CPU");
  TORCH_CHECK(!x.is_conj() && !x.is_neg() && !y.is_conj() && !y.is_neg(),
              "sddmm_csr: lazy conjugate or negative views must be materialized first");
  const int64_t rows = positions.numel() - 1;
  TORCH_CHECK(x.size(0) == rows, "sddmm_csr: x must have one row per row of the structure");
  TORCH_CHECK(nthreads_override >= -1 && nthreads_override <= std::numeric_limits<int>::max(),
              "sddmm_csr: nthreads must be -1 or a thread count");
  const int64_t nnz = coordinates.numel();
  const int64_t k = x.size(1);
  torch::Tensor out = torch::empty({nnz}, x.options());
  if (nnz == 0 || rows == 0) return out;
  if (k == 0) return out.zero_();

  const long work = scorch_spmm_work((long)nnz, 0, (long)k, (long)k);
  const int nthreads =
      scorch_spmm_nthreads(work, (long)rows, (int)nthreads_override, (long)nnz * (long)k,
                           (long)nnz, scorch_spmm_full_team(atparallel));
  const bool on_pool =
      scorch_spmm_on_torch_pool(atparallel, (int)nthreads_override, nthreads, work, (long)rows);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sddmm_csr", [&] {
    if (positions.scalar_type() == torch::kInt32)
      sddmm_csr_rows<int32_t, scalar_t>(positions.data_ptr<int32_t>(),
                                        coordinates.data_ptr<int32_t>(), x.data_ptr<scalar_t>(),
                                        y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, k,
                                        nthreads, on_pool);
    else
      sddmm_csr_rows<int64_t, scalar_t>(positions.data_ptr<int64_t>(),
                                        coordinates.data_ptr<int64_t>(), x.data_ptr<scalar_t>(),
                                        y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, k,
                                        nthreads, on_pool);
  });
  return out;
}

}  // namespace scorch_native
