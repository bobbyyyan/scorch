#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "header.h"        // scorch_zero_dense (parallel span zero-fill)
#include "prebuilt_types.h"
#include "scorch_policy.h"  // shared scorch_nthreads / scorch_chunk (see header.h)
#include <ATen/Parallel.h>  // at::parallel_for (pipeline-pool composition A/B)
#include <cstdlib>          // std::getenv / std::atol (runtime A/B flag)

#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

// Global constants for optimization
const int kUnrollFactor = 16;

// ---------------------------------------------------------------------------
// Fused SpMM + bias + ReLU:  C[i,k] = max(0, sum_j A[i,j]*B[j,k] + bias[k])
// ---------------------------------------------------------------------------

template <typename scalar_t, bool apply_relu>
Tensor spmm_csr_bias_act(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, torch::Tensor bias_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  scalar_t* SCORCH_RESTRICT A_val = A_values.data_ptr<scalar_t>();

  int B1_size = B_shape[1];
  scalar_t* SCORCH_RESTRICT B_val = B_values.data_ptr<scalar_t>();
  scalar_t* SCORCH_RESTRICT bias_val = bias_values.data_ptr<scalar_t>();

  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch = torch::empty(
      {(long long)C_capacity}, scorch_torch_dtype<scalar_t>());
  scalar_t* SCORCH_RESTRICT C_values =
      C_values_torch.data_ptr<scalar_t>();

  // Work-escalated thread cap, FLOORED at the platform default (omp_get_max_threads
  // = torch's per-platform thread count, i.e. the P-core count on a hybrid P+E CPU).
  // Unlike the non-fused SpMM panel kernel (spmm_csr_float_v2), spmm_csr_bias_act
  // is ONLY ever a GCN/GNN layer: a low-reuse, bandwidth-bound sparse @ dense
  // product. v2's nnz*k throttle (SCORCH_GRAIN_SPMM, tuned to keep small SuiteSparse
  // products off the E-core cliff) is wrong here -- it collapsed the small/narrow-K
  // GCN shapes to ~1 thread and regressed the fused path +14..33% on x86. Flooring
  // at omp_get_max_threads() keeps those shapes at full P-core width (recovering
  // the pre-6a59ea3 numbers) without ever exceeding what the platform already uses
  // for dense ops, so no new E-core cliff. Big graphs still escalate past the floor
  // via by_work up to omp_get_num_procs(), preserving the M5 fix (reddit 6->18
  // threads). k floored at the 64B cache line (16 f32). This kernel is not on the
  // SuiteSparse SpMM panel, so none of this perturbs that panel. Fixed dynamic
  // chunk 16 matches the pre-regression schedule (proven good on all 6 graphs).
  const int total_nnz = A1_pos[A0_size];
  const long k_eff = B1_size < 16 ? 16L : (long)B1_size;
  const long work = (long)total_nnz * k_eff;
  const int nthreads = scorch_nthreads(work, A0_size, SCORCH_GRAIN_SPMM,
                                       omp_get_max_threads());

  #pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
  for (int i = 0; i < A0_size; i++) {
    size_t pC1_base = (size_t)i * (size_t)C1_size;
    int pA1_end = A1_pos[i + 1];

    // Initialize row with bias (avoid separate memset + bias pass)
    for (int k = 0; k < B1_size; k++) {
      C_values[pC1_base + k] = bias_val[k];
    }

    // Accumulate SpMM
    for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
      if (pA1 + 1 < pA1_end)
        __builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);
      int j = A1_crd[pA1];
      scalar_t a_val = A_val[pA1];
      size_t pB0 = (size_t)j * (size_t)B1_size;

      for (int k = 0; k < B1_size; k++) {
        C_values[pC1_base + k] += a_val * B_val[pB0 + k];
      }
    }

    // Apply ReLU in-place (same cache line, no extra memory pass)
    if constexpr (apply_relu) {
      for (int k = 0; k < B1_size; k++) {
        scalar_t val = C_values[pC1_base + k];
        C_values[pC1_base + k] = val > 0 ? val : 0;
      }
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// Pointer-based kernel; see the note on spmm_csr_float_v2_core for why the entry
// is split in two. This one carries the float64 CSR x dense route, which resolves
// to spmm_csr_double.
template <typename scalar_t>
torch::Tensor spmm_csr_typed_core(
                int C0_size, int C1_size, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const scalar_t* SCORCH_RESTRICT A_val,
                int B1_size, const scalar_t* SCORCH_RESTRICT B_val,
                int tile_size) {
  (void)tile_size;  // the reference kernel does not tile; kept for signature parity

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch = torch::empty(
      {(long long)C_capacity}, scorch_torch_dtype<scalar_t>());
  scalar_t* SCORCH_RESTRICT C_values =
      C_values_torch.data_ptr<scalar_t>();
  memset(C_values, 0, sizeof(scalar_t) * C_capacity);

  // Row-parallel loop: i (rows) -> j (sparse) -> k (dense columns).
  // Traverses the sparse structure once per row, and dynamic scheduling
  // handles load imbalance from varying row densities.
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < A0_size; i++) {
    int pC0 = i;
    int pA1_end = A1_pos[i + 1];

    for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
      if (pA1 + 1 < pA1_end)
        __builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);
      int j = A1_crd[pA1];
      scalar_t a_val = A_val[pA1];
      size_t pB0 = (size_t)j * (size_t)B1_size;
      size_t pC1_base = (size_t)pC0 * (size_t)C1_size;

      for (int k = 0; k < B1_size; k++) {
        C_values[pC1_base + k] += a_val * B_val[pB0 + k];
      }
    }
  }

  return C_values_torch;
}

template <typename scalar_t>
Tensor spmm_csr_typed(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 0) {
  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = spmm_csr_typed_core<scalar_t>(
      result_shape[0], result_shape[1], A_shape[0],
      A_mode_indices[1][0].data_ptr<int>(),
      A_mode_indices[1][1].data_ptr<int>(),
      A_values.data_ptr<scalar_t>(),
      B_shape[1], B_values.data_ptr<scalar_t>(),
      tile_size);
  return C;
}

Tensor spmm_csr_float(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 32) {
  return spmm_csr_typed<float>(
      result_shape,
      A_shape,
      A_mode_indices,
      A_values,
      B_shape,
      B_mode_indices,
      B_values,
      tile_size);
}

Tensor spmm_csr_double(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 32) {
  return spmm_csr_typed<double>(
      result_shape,
      A_shape,
      A_mode_indices,
      A_values,
      B_shape,
      B_mode_indices,
      B_values,
      tile_size);
}

Tensor spmm_csr_float_untiled(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array - use size_t to avoid integer overflow
  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  #pragma omp parallel for
  for (int i = 0; i < A0_size; i++) {
    // Resolve index into dense level of values array
    int pC0 = i;
    int pA1_end = A1_pos[i + 1];

    // For each row i, iterate through all columns k directly
    for (int k = 0; k < B1_size; k++) {
      float accum = 0.0f;

      // Iterate through the non-zero elements in row i
      for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int j = A1_crd[pA1];

        // Resolve dense coordinates - use size_t to avoid overflow
        size_t pB1 = (size_t)j * (size_t)B1_size + (size_t)k;
        accum += A_val[pA1] * B_val[pB1];
      }

      // Add to result - use size_t for index calculation
      size_t pC1 = (size_t)pC0 * (size_t)C1_size + (size_t)k;
      C_values[pC1] += accum;
    }
  }

  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_coo_float(std::vector<int> result_shape,
                      std::vector<int> A_shape,
                      std::vector<std::vector<torch::Tensor>> A_mode_indices,
                      torch::Tensor A_values,
                      std::vector<int> B_shape,
                      std::vector<std::vector<torch::Tensor>> B_mode_indices,
                      torch::Tensor B_values) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  torch::Tensor A0_crd_tensor = A_mode_indices[0][0];
  int* A0_crd = A_mode_indices[0][0].data_ptr<int>();
  torch::Tensor A1_crd_tensor = A_mode_indices[1][0];
  int* A1_crd = A_mode_indices[1][0].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Initialize tile sizes
  constexpr int kTile_k = 4096;

  int residual_k_start = (B1_size / kTile_k) * kTile_k;

  for (int k_out = 0; k_out < residual_k_start; k_out += kTile_k) {
    // Initialize iterators
    int pA0_end = A0_crd_tensor.size(0);
    int pA1_end = 0;

    for (int pA0 = 0; pA0 < pA0_end; pA0 = pA1_end) {
      // Resolve coordinates
      int i = A0_crd[pA0];

      // Find iterator end for coordinate level
      pA1_end = pA0 + 1;
      while (pA1_end < pA0_end && A0_crd[pA1_end] == i) {
        pA1_end++;
      }

      // Resolve index into dense level of values array
      int pC0 = i;

      float wksp[kTile_k] = {};

      for (int pA1 = pA0; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int j = A1_crd[pA1];

        // Resolve dense coordinates
        int pB0 = j;

        for (int k_in = 0; k_in < kTile_k; k_in++) {
          // Resolve tiled index var
          int k = k_out + k_in;
          // Resolve dense coordinates
          int pB1 = pB0 * B1_size + k;
          wksp[k_in] += A_val[pA1] * B_val[pB1];
        }
      }

      // Lower consumer CIN
      for (int k_in = 0; k_in < kTile_k; k_in++) {
        int k = k_out + k_in;
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += wksp[k_in];
      }
    }
  }

  if (residual_k_start < B1_size) {
    for (int k = residual_k_start; k < B1_size; k++) {
      // Initialize iterators
      int pA0_end = A0_crd_tensor.size(0);
      int pA1_end = 0;

      for (int pA0 = 0; pA0 < pA0_end; pA0 = pA1_end) {
        // Resolve coordinates
        int i = A0_crd[pA0];

        // Find iterator end for coordinate level
        pA1_end = pA0 + 1;
        while (pA1_end < pA0_end && A0_crd[pA1_end] == i) {
          pA1_end++;
        }

        // Resolve index into dense level of values array
        int pC0 = i;
        // Initialize workspaces
        float wksp = 0.0f;

        for (int pA1 = pA0; pA1 < pA1_end; pA1++) {
          // Resolve coordinates
          int j = A1_crd[pA1];

          // Resolve dense coordinates
          int pB0 = j;
          int pB1 = pB0 * B1_size + k;
          wksp += A_val[pA1] * B_val[pB1];
        }

        // Lower consumer CIN
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += wksp;
      }
    }
  }
  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_csr_float_optimized(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 128) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Use the tile size parameter with a reasonable default
  int kTile_k = tile_size;

  // Compute how many tiles we need
  int num_tiles = static_cast<int>(
      (static_cast<int64_t>(B1_size) + kTile_k - 1) / kTile_k);

  const size_t workspace_stride =
      ((static_cast<size_t>(kTile_k) + 15) / 16) * 16;
  auto thread_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(omp_get_max_threads()), workspace_stride);

  #pragma omp parallel
  {
    float* SCORCH_RESTRICT thread_workspace =
        thread_workspaces.get() +
        static_cast<size_t>(omp_get_thread_num()) * workspace_stride;

    #pragma omp for
    for (int i = 0; i < A0_size; i++) {
      // Resolve index into dense level of values array
      int pC0 = i;
      int pA1_begin = A1_pos[i];
      int pA1_end = A1_pos[i + 1];
      int nnz_in_row = pA1_end - pA1_begin;

      // Skip empty rows
      if (nnz_in_row == 0) continue;

      // Process each tile
      for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int k_out = tile_idx * kTile_k;
        int k_limit = std::min(k_out + kTile_k, B1_size);
        int actual_tile_size = k_limit - k_out;

        // Clear workspace for this tile
        memset(thread_workspace, 0, sizeof(float) * actual_tile_size);

        // Process all non-zeros in this row for current tile
        for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
          int j = A1_crd[pA1];
          float a_val = A_val[pA1];

          // Base index for B values
          int pB1_base = j * B1_size + k_out;

          // Process tile elements with loop unrolling for better vectorization
          int k_in = 0;

          // Process blocks of 4 elements for better vectorization
          for (; k_in + 3 < actual_tile_size; k_in += 4) {
            thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
            thread_workspace[k_in + 1] += a_val * B_val[pB1_base + k_in + 1];
            thread_workspace[k_in + 2] += a_val * B_val[pB1_base + k_in + 2];
            thread_workspace[k_in + 3] += a_val * B_val[pB1_base + k_in + 3];
          }

          // Handle remaining elements
          for (; k_in < actual_tile_size; k_in++) {
            thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
          }
        }

        // Write results to output with direct indexing
        for (int k_in = 0; k_in < actual_tile_size; k_in++) {
          int pC1 = pC0 * C1_size + (k_out + k_in);
          C_values[pC1] = thread_workspace[k_in];
        }
      }
    }
  }

  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_csr_float_turbo(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 128) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Calculate optimal tile size for the current matrix
  // The heuristic here is to use a larger tile size for larger matrices
  // but avoid excessive memory usage per thread
  int max_nnz_per_row = 0;
  for (int i = 0; i < A0_size; i++) {
    int row_nnz = A1_pos[i + 1] - A1_pos[i];
    max_nnz_per_row = std::max(max_nnz_per_row, row_nnz);
  }

  // Adjust tile size based on matrix properties
  int optimal_tile_size = tile_size;
  if (max_nnz_per_row > 1000 && B1_size > 500) {
    optimal_tile_size = std::min(tile_size, 64); // Use smaller tiles for very dense rows
  } else if (max_nnz_per_row < 10 && B1_size > 1000) {
    optimal_tile_size = std::max(tile_size, 256); // Use larger tiles for very sparse rows
  }

  int kTile_k = optimal_tile_size;

  const size_t workspace_stride =
      ((static_cast<size_t>(kTile_k) + 15) / 16) * 16;
  auto thread_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(omp_get_max_threads()), workspace_stride);

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < A0_size; i++) {
    int pC0 = i;
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    int nnz_in_row = pA1_end - pA1_begin;

    // Skip rows with no non-zeros
    if (SCORCH_UNLIKELY(nnz_in_row == 0)) continue;

    float* SCORCH_RESTRICT accum_c =
        thread_workspaces.get() +
        static_cast<size_t>(omp_get_thread_num()) * workspace_stride;

    // Process each row in tiles
    for (int k_out = 0; k_out < B1_size; k_out += kTile_k) {
      // Calculate actual tile width to handle final partial tile
      int k_width = std::min(kTile_k, B1_size - k_out);

      // Clear workspace for this tile
      if (k_width == kTile_k) {
        memset(accum_c, 0, sizeof(float) * kTile_k);
      } else {
        memset(accum_c, 0, sizeof(float) * k_width);
      }

      // For each non-zero element in the current row
      for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
        int j = A1_crd[pA1];
        float a_val = A_val[pA1];
        int pB1_base = j * B1_size + k_out;

        // Full tile processing
        if (SCORCH_LIKELY(k_width == kTile_k)) {
          // Process blocks of 8 elements where possible using SIMD-friendly pattern
          int k_in = 0;
          for (; k_in + 7 < kTile_k; k_in += 8) {
            accum_c[k_in] += a_val * B_val[pB1_base + k_in];
            accum_c[k_in + 1] += a_val * B_val[pB1_base + k_in + 1];
            accum_c[k_in + 2] += a_val * B_val[pB1_base + k_in + 2];
            accum_c[k_in + 3] += a_val * B_val[pB1_base + k_in + 3];
            accum_c[k_in + 4] += a_val * B_val[pB1_base + k_in + 4];
            accum_c[k_in + 5] += a_val * B_val[pB1_base + k_in + 5];
            accum_c[k_in + 6] += a_val * B_val[pB1_base + k_in + 6];
            accum_c[k_in + 7] += a_val * B_val[pB1_base + k_in + 7];
          }

          // Process remaining elements
          for (; k_in < kTile_k; k_in++) {
            accum_c[k_in] += a_val * B_val[pB1_base + k_in];
          }
        } else {
          // Partial tile processing (last tile in row)
          for (int k_in = 0; k_in < k_width; k_in++) {
            accum_c[k_in] += a_val * B_val[pB1_base + k_in];
          }
        }
      }

      // Write accumulated results to output matrix
      for (int k_in = 0; k_in < k_width; k_in++) {
        int pC1 = pC0 * C1_size + (k_out + k_in);
        C_values[pC1] = accum_c[k_in];
      }
    }
  }

  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_csr_float_ultra(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 256) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array - use size_t to avoid integer overflow
  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;

  // Check if allocation size is reasonable - 256GB limit is arbitrary but reasonable
  const size_t MAX_DENSE_ALLOCATION = (size_t)256 * 1024 * 1024 * 1024;
  size_t allocation_size = C_capacity * sizeof(float);

  if (allocation_size > MAX_DENSE_ALLOCATION) {
    std::stringstream ss;
    ss << "Attempted to allocate " << (allocation_size / (1024.0 * 1024.0 * 1024.0))
       << " GB for dense output matrix (" << C0_size << " x " << C1_size
       << "). This exceeds the maximum allocation size of "
       << (MAX_DENSE_ALLOCATION / (1024.0 * 1024.0 * 1024.0)) << " GB.";
    throw std::runtime_error(ss.str());
  }

  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, allocation_size);

  // Calculate statistics for adaptive tiling
  int total_nnz = 0;
  int max_nnz_per_row = 0;
  int rows_with_nnz = 0;

  for (int i = 0; i < A0_size; i++) {
    int row_nnz = A1_pos[i + 1] - A1_pos[i];
    total_nnz += row_nnz;
    if (row_nnz > 0) rows_with_nnz++;
    max_nnz_per_row = std::max(max_nnz_per_row, row_nnz);
  }

  float avg_nnz_per_row = rows_with_nnz > 0 ? (float)total_nnz / rows_with_nnz : 0;

  // Adaptive tile size based on matrix characteristics
  int default_tile_size = tile_size;

  // Adjust default tile size based on sparsity pattern
  if (avg_nnz_per_row < 5 && B1_size > 1000) {
    // Larger tiles for very sparse matrices
    default_tile_size = std::max(512, tile_size);
  } else if (avg_nnz_per_row > 100 || max_nnz_per_row > 1000) {
    // Smaller tiles for matrices with dense rows
    default_tile_size = std::min(tile_size, 128);
  }

  // Ensure tile size is a multiple of 16 for SIMD operations
  default_tile_size = (default_tile_size + 15) & ~15;

  // Determine optimal thread count
  int num_threads = omp_get_max_threads();
  if (A0_size < 1000 || (avg_nnz_per_row < 10 && A0_size < 10000)) {
    // Use fewer threads for small or very sparse matrices
    num_threads = std::min(num_threads, 4);
  }
  omp_set_num_threads(num_threads);

  // Allocate all worker scratch before entering OpenMP so allocation failures
  // unwind normally instead of escaping a parallel structured block.
  const size_t aligned_tile_size =
      ((static_cast<size_t>(default_tile_size) + 15) / 16) * 16;
  auto thread_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(num_threads), aligned_tile_size);

  // Process all rows in a single parallel region with dynamic scheduling
  #pragma omp parallel
  {
    float* SCORCH_RESTRICT thread_workspace =
        thread_workspaces.get() +
        static_cast<size_t>(omp_get_thread_num()) * aligned_tile_size;

    #pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < A0_size; i++) {
        int pC0 = i;
        int pA1_begin = A1_pos[i];
        int pA1_end = A1_pos[i + 1];
        int nnz_in_row = pA1_end - pA1_begin;

        // Skip empty rows
        if (SCORCH_UNLIKELY(nnz_in_row == 0)) continue;

        // Special case for very sparse rows (1-2 non-zeros)
        // Direct computation without tiling is more efficient
        if (SCORCH_UNLIKELY(nnz_in_row <= 2)) {
          for (int k = 0; k < B1_size; k++) {
            float accum = 0.0f;
            for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
              int j = A1_crd[pA1];
              int pB1 = j * B1_size + k;
              accum += A_val[pA1] * B_val[pB1];
            }
            // Use 64-bit index calculation to avoid overflow
            size_t pC1 = (size_t)pC0 * (size_t)C1_size + (size_t)k;
            C_values[pC1] = accum;
          }
          continue;
        }

        // Choose tile size based on row density
        int row_tile_size;
        if (nnz_in_row > 100) {
          // Dense rows: use smaller tiles for better cache utilization
          row_tile_size = std::min(default_tile_size, 128);
          // Ensure it's still a multiple of 16
          row_tile_size = (row_tile_size + 15) & ~15;
        } else {
          // Other rows: use the default tile size (already aligned)
          row_tile_size = default_tile_size;
        }

        // Use the smallest of the aligned tile size and row_tile_size
        int tile_to_use = std::min((int)aligned_tile_size, row_tile_size);

        // Process the row in tiles
        for (int k_out = 0; k_out < B1_size; k_out += tile_to_use) {
          int k_width = std::min(tile_to_use, B1_size - k_out);

          // Clear the workspace for this tile
          memset(thread_workspace, 0, sizeof(float) * k_width);

          // Process all non-zeros in this row for the current tile
          for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
            int j = A1_crd[pA1];
            float a_val = A_val[pA1];
            // Use 64-bit calculation to avoid integer overflow
            size_t pB1_base = (size_t)j * (size_t)B1_size + (size_t)k_out;

            // Simple prefetch of next element's data if available
            if (pA1 + 1 < pA1_end) {
              __builtin_prefetch(&B_val[(size_t)A1_crd[pA1 + 1] * (size_t)B1_size + (size_t)k_out], 0, 3);
            }

            // Full tile processing with manual unrolling for SIMD efficiency
            if (SCORCH_LIKELY(k_width == tile_to_use)) {
              int k_in = 0;

              // Use aggressive unrolling in blocks of 16 for better vectorization
              SCORCH_PRAGMA_UNROLL
              for (; k_in + 15 < tile_to_use; k_in += 16) {
                thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
                thread_workspace[k_in + 1] += a_val * B_val[pB1_base + k_in + 1];
                thread_workspace[k_in + 2] += a_val * B_val[pB1_base + k_in + 2];
                thread_workspace[k_in + 3] += a_val * B_val[pB1_base + k_in + 3];
                thread_workspace[k_in + 4] += a_val * B_val[pB1_base + k_in + 4];
                thread_workspace[k_in + 5] += a_val * B_val[pB1_base + k_in + 5];
                thread_workspace[k_in + 6] += a_val * B_val[pB1_base + k_in + 6];
                thread_workspace[k_in + 7] += a_val * B_val[pB1_base + k_in + 7];
                thread_workspace[k_in + 8] += a_val * B_val[pB1_base + k_in + 8];
                thread_workspace[k_in + 9] += a_val * B_val[pB1_base + k_in + 9];
                thread_workspace[k_in + 10] += a_val * B_val[pB1_base + k_in + 10];
                thread_workspace[k_in + 11] += a_val * B_val[pB1_base + k_in + 11];
                thread_workspace[k_in + 12] += a_val * B_val[pB1_base + k_in + 12];
                thread_workspace[k_in + 13] += a_val * B_val[pB1_base + k_in + 13];
                thread_workspace[k_in + 14] += a_val * B_val[pB1_base + k_in + 14];
                thread_workspace[k_in + 15] += a_val * B_val[pB1_base + k_in + 15];
              }

              // Handle remaining elements
              for (; k_in < tile_to_use; k_in++) {
                thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
              }
            } else {
              // Handle partial tile (last tile in row)
              for (int k_in = 0; k_in < k_width; k_in++) {
                thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
              }
            }
          }

          // Write accumulated results directly to output matrix
          // Use 64-bit index calculation to avoid overflow
          size_t pC1_base = (size_t)pC0 * (size_t)C1_size + (size_t)k_out;
          int k_in = 0;

          // Use block writes for better memory performance
          for (; k_in + 15 < k_width; k_in += 16) {
            C_values[pC1_base + k_in] = thread_workspace[k_in];
            C_values[pC1_base + k_in + 1] = thread_workspace[k_in + 1];
            C_values[pC1_base + k_in + 2] = thread_workspace[k_in + 2];
            C_values[pC1_base + k_in + 3] = thread_workspace[k_in + 3];
            C_values[pC1_base + k_in + 4] = thread_workspace[k_in + 4];
            C_values[pC1_base + k_in + 5] = thread_workspace[k_in + 5];
            C_values[pC1_base + k_in + 6] = thread_workspace[k_in + 6];
            C_values[pC1_base + k_in + 7] = thread_workspace[k_in + 7];
            C_values[pC1_base + k_in + 8] = thread_workspace[k_in + 8];
            C_values[pC1_base + k_in + 9] = thread_workspace[k_in + 9];
            C_values[pC1_base + k_in + 10] = thread_workspace[k_in + 10];
            C_values[pC1_base + k_in + 11] = thread_workspace[k_in + 11];
            C_values[pC1_base + k_in + 12] = thread_workspace[k_in + 12];
            C_values[pC1_base + k_in + 13] = thread_workspace[k_in + 13];
            C_values[pC1_base + k_in + 14] = thread_workspace[k_in + 14];
            C_values[pC1_base + k_in + 15] = thread_workspace[k_in + 15];
          }

          // Handle remaining elements
          for (; k_in < k_width; k_in++) {
            C_values[pC1_base + k_in] = thread_workspace[k_in];
          }
        }
    }
  }

  // Restore default thread count
  omp_set_num_threads(omp_get_max_threads());

  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_csr_float_apex(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 256) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Calculate statistics for adaptive tiling
  int total_nnz = 0;
  int max_nnz_per_row = 0;
  int rows_with_nnz = 0;

  for (int i = 0; i < A0_size; i++) {
    int row_nnz = A1_pos[i + 1] - A1_pos[i];
    total_nnz += row_nnz;
    if (row_nnz > 0) rows_with_nnz++;
    max_nnz_per_row = std::max(max_nnz_per_row, row_nnz);
  }

  float avg_nnz_per_row = rows_with_nnz > 0 ? (float)total_nnz / rows_with_nnz : 0;

  // Adaptive tile size based on matrix characteristics
  int default_tile_size = tile_size;

  // Adjust default tile size based on sparsity pattern
  if (avg_nnz_per_row < 5 && B1_size > 1000) {
    // Larger tiles for very sparse matrices
    default_tile_size = std::max(512, tile_size);
  } else if (avg_nnz_per_row > 100 || max_nnz_per_row > 1000) {
    // Smaller tiles for matrices with dense rows
    default_tile_size = std::min(tile_size, 128);
  }

  // Ensure tile size is aligned to 16 bytes for SIMD operations
  default_tile_size = (default_tile_size + 15) & ~15;

  // Determine optimal thread count
  int num_threads = omp_get_max_threads();
  if (A0_size < 1000 || (avg_nnz_per_row < 10 && A0_size < 10000)) {
    // Use fewer threads for small or very sparse matrices
    num_threads = std::min(num_threads, 4);
  }
  omp_set_num_threads(num_threads);

  auto thread_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(num_threads),
      static_cast<size_t>(default_tile_size));

  // Process all rows in a single parallel region with dynamic scheduling
  #pragma omp parallel
  {
    float* SCORCH_RESTRICT thread_workspace =
        thread_workspaces.get() +
        static_cast<size_t>(omp_get_thread_num()) *
            static_cast<size_t>(default_tile_size);

    #pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < A0_size; i++) {
      int pC0 = i;
      int pA1_begin = A1_pos[i];
      int pA1_end = A1_pos[i + 1];
      int nnz_in_row = pA1_end - pA1_begin;

      // Skip empty rows
      if (SCORCH_UNLIKELY(nnz_in_row == 0)) continue;

      // Special case for very sparse rows (1-2 non-zeros)
      // Direct computation without tiling is more efficient
      if (SCORCH_UNLIKELY(nnz_in_row <= 2)) {
        for (int k = 0; k < B1_size; k++) {
          float accum = 0.0f;
          for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
            int j = A1_crd[pA1];
            int pB1 = j * B1_size + k;
            accum += A_val[pA1] * B_val[pB1];
          }
          int pC1 = pC0 * C1_size + k;
          C_values[pC1] = accum;
        }
        continue;
      }

      // Choose tile size based on row density
      int row_tile_size;
      if (nnz_in_row > 100) {
        // Dense rows: use smaller tiles for better cache utilization
        row_tile_size = std::min(default_tile_size, 128);
      } else {
        // Other rows: use the default tile size
        row_tile_size = default_tile_size;
      }

      // Process the row in tiles
      for (int k_out = 0; k_out < B1_size; k_out += row_tile_size) {
        int k_width = std::min(row_tile_size, B1_size - k_out);

        // Clear the workspace for this tile
        memset(thread_workspace, 0, sizeof(float) * k_width);

        // Process all non-zeros in this row for the current tile
        for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
          int j = A1_crd[pA1];
          float a_val = A_val[pA1];
          int pB1_base = j * B1_size + k_out;

          // Simple prefetch of next element's data if available
          if (pA1 + 1 < pA1_end) {
            __builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size + k_out], 0, 3);
          }

          // Full tile processing with manual unrolling for SIMD efficiency
          if (SCORCH_LIKELY(k_width == row_tile_size)) {
            int k_in = 0;

            // Use aggressive unrolling in blocks of 16 for better vectorization
            SCORCH_PRAGMA_UNROLL
            for (; k_in + 15 < row_tile_size; k_in += 16) {
              thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
              thread_workspace[k_in + 1] += a_val * B_val[pB1_base + k_in + 1];
              thread_workspace[k_in + 2] += a_val * B_val[pB1_base + k_in + 2];
              thread_workspace[k_in + 3] += a_val * B_val[pB1_base + k_in + 3];
              thread_workspace[k_in + 4] += a_val * B_val[pB1_base + k_in + 4];
              thread_workspace[k_in + 5] += a_val * B_val[pB1_base + k_in + 5];
              thread_workspace[k_in + 6] += a_val * B_val[pB1_base + k_in + 6];
              thread_workspace[k_in + 7] += a_val * B_val[pB1_base + k_in + 7];
              thread_workspace[k_in + 8] += a_val * B_val[pB1_base + k_in + 8];
              thread_workspace[k_in + 9] += a_val * B_val[pB1_base + k_in + 9];
              thread_workspace[k_in + 10] += a_val * B_val[pB1_base + k_in + 10];
              thread_workspace[k_in + 11] += a_val * B_val[pB1_base + k_in + 11];
              thread_workspace[k_in + 12] += a_val * B_val[pB1_base + k_in + 12];
              thread_workspace[k_in + 13] += a_val * B_val[pB1_base + k_in + 13];
              thread_workspace[k_in + 14] += a_val * B_val[pB1_base + k_in + 14];
              thread_workspace[k_in + 15] += a_val * B_val[pB1_base + k_in + 15];
            }

            // Handle remaining elements
            for (; k_in < row_tile_size; k_in++) {
              thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
            }
          } else {
            // Handle partial tile (last tile in row)
            for (int k_in = 0; k_in < k_width; k_in++) {
              thread_workspace[k_in] += a_val * B_val[pB1_base + k_in];
            }
          }
        }

        // Write accumulated results directly to output matrix
        int pC1_base = pC0 * C1_size + k_out;
        int k_in = 0;

        // Use block writes for better memory performance
        for (; k_in + 15 < k_width; k_in += 16) {
          C_values[pC1_base + k_in] = thread_workspace[k_in];
          C_values[pC1_base + k_in + 1] = thread_workspace[k_in + 1];
          C_values[pC1_base + k_in + 2] = thread_workspace[k_in + 2];
          C_values[pC1_base + k_in + 3] = thread_workspace[k_in + 3];
          C_values[pC1_base + k_in + 4] = thread_workspace[k_in + 4];
          C_values[pC1_base + k_in + 5] = thread_workspace[k_in + 5];
          C_values[pC1_base + k_in + 6] = thread_workspace[k_in + 6];
          C_values[pC1_base + k_in + 7] = thread_workspace[k_in + 7];
          C_values[pC1_base + k_in + 8] = thread_workspace[k_in + 8];
          C_values[pC1_base + k_in + 9] = thread_workspace[k_in + 9];
          C_values[pC1_base + k_in + 10] = thread_workspace[k_in + 10];
          C_values[pC1_base + k_in + 11] = thread_workspace[k_in + 11];
          C_values[pC1_base + k_in + 12] = thread_workspace[k_in + 12];
          C_values[pC1_base + k_in + 13] = thread_workspace[k_in + 13];
          C_values[pC1_base + k_in + 14] = thread_workspace[k_in + 14];
          C_values[pC1_base + k_in + 15] = thread_workspace[k_in + 15];
        }

        // Handle remaining elements
        for (; k_in < k_width; k_in++) {
          C_values[pC1_base + k_in] = thread_workspace[k_in];
        }
      }
    }
  }

  // Restore default thread count
  omp_set_num_threads(omp_get_max_threads());

  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmm_csr_float_tiled_i_k(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int i_tile_size = 16, int k_tile_size = 32) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array - use size_t to avoid integer overflow
  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Use the tile size parameters
  int kTile_i = i_tile_size;
  int kTile_k = k_tile_size;

  int num_i_tiles = static_cast<int>(
      (static_cast<int64_t>(A0_size) + kTile_i - 1) / kTile_i);
  int residual_k_start = (B1_size / kTile_k) * kTile_k;

  const size_t workspace_stride =
      ((static_cast<size_t>(kTile_k) + 15) / 16) * 16;
  auto thread_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(omp_get_max_threads()), workspace_stride);

  #pragma omp parallel for
  for (int i_tile = 0; i_tile < num_i_tiles; i_tile++) {
    // Calculate the start and end of this i-tile
    int i_start = i_tile * kTile_i;
    int i_end = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(i_start) + kTile_i, A0_size));

    float* SCORCH_RESTRICT accum_c =
        thread_workspaces.get() +
        static_cast<size_t>(omp_get_thread_num()) * workspace_stride;

    for (int k_out = 0; k_out < residual_k_start; k_out += kTile_k) {
      // For each i-tile and k-tile, process the computation

      for (int i = i_start; i < i_end; i++) {
        // Resolve index into dense level of values array
        int pC0 = i;

        memset(accum_c, 0, sizeof(float) * static_cast<size_t>(kTile_k));

        // Initialize iterators
        int pA1_end = A1_pos[i + 1];

        for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
          // Resolve coordinates
          int j = A1_crd[pA1];

          for (int k_in = 0; SCORCH_LIKELY(k_in < kTile_k); k_in++) {
            // Resolve tiled index var
            int k = k_out + k_in;
            // Resolve dense coordinates - use size_t to avoid overflow
            size_t pB1 = (size_t)j * (size_t)B1_size + (size_t)k;
            accum_c[k_in] += A_val[pA1] * B_val[pB1];
          }
        }

        // Lower consumer CIN
        for (int k_in = 0; SCORCH_LIKELY(k_in < kTile_k); k_in++) {
          int k = k_out + k_in;
          // Use size_t for index calculation to prevent overflow
          size_t pC1 = (size_t)pC0 * (size_t)C1_size + (size_t)k;
          C_values[pC1] += accum_c[k_in];
        }
      }
    }
  }

  if (residual_k_start < B1_size) {
    int tile_k_width = B1_size - residual_k_start;

    #pragma omp parallel for
    for (int i_tile = 0; i_tile < num_i_tiles; i_tile++) {
      // Calculate the start and end of this i-tile
      int i_start = i_tile * kTile_i;
      int i_end = std::min(i_start + kTile_i, A0_size);

      float* SCORCH_RESTRICT accum_c =
          thread_workspaces.get() +
          static_cast<size_t>(omp_get_thread_num()) * workspace_stride;

      for (int i = i_start; i < i_end; i++) {
        int pC0 = i;

        memset(accum_c, 0,
               sizeof(float) * static_cast<size_t>(tile_k_width));
        int pA1_end = A1_pos[i + 1];

        for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
          int j = A1_crd[pA1];

          for (int k = residual_k_start; k < B1_size; k++) {
            // Use size_t for index calculation
            size_t pB1 = (size_t)j * (size_t)B1_size + (size_t)k;
            accum_c[k - residual_k_start] += A_val[pA1] * B_val[pB1];
          }
        }

        for (int k = residual_k_start; k < B1_size; k++) {
          // Use size_t for index calculation
          size_t pC1 = (size_t)pC0 * (size_t)C1_size + (size_t)k;
          C_values[pC1] += accum_c[k - residual_k_start];
        }
      }
    }
  }
  // Assemble final result
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// ============================================================================
// Novel SpMM Kernel Variants
// ============================================================================

// Variant 1: Direct-to-C Accumulation (No Workspace)
// Hypothesis: Workspace alloc + memset + copy-back is unnecessary overhead
// since C is pre-zeroed and each row is independent.
Tensor spmm_csr_float_direct(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < A0_size; i++) {
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    if (pA1_begin == pA1_end) continue;

    float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

    for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;

      // Prefetch next B row
      if (pA1 + 1 < pA1_end) {
        __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + 1] * (size_t)B1_size, 0, 1);
      }

      // 16-wide manual unroll
      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        C_row[k]      += a_val * B_row[k];
        C_row[k + 1]  += a_val * B_row[k + 1];
        C_row[k + 2]  += a_val * B_row[k + 2];
        C_row[k + 3]  += a_val * B_row[k + 3];
        C_row[k + 4]  += a_val * B_row[k + 4];
        C_row[k + 5]  += a_val * B_row[k + 5];
        C_row[k + 6]  += a_val * B_row[k + 6];
        C_row[k + 7]  += a_val * B_row[k + 7];
        C_row[k + 8]  += a_val * B_row[k + 8];
        C_row[k + 9]  += a_val * B_row[k + 9];
        C_row[k + 10] += a_val * B_row[k + 10];
        C_row[k + 11] += a_val * B_row[k + 11];
        C_row[k + 12] += a_val * B_row[k + 12];
        C_row[k + 13] += a_val * B_row[k + 13];
        C_row[k + 14] += a_val * B_row[k + 14];
        C_row[k + 15] += a_val * B_row[k + 15];
      }
      for (; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// Variant 2: Explicit ARM NEON Vectorization
// Hypothesis: Explicit NEON FMA intrinsics may beat auto-vectorization
// for the scatter-add inner loop.
Tensor spmm_csr_float_neon(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < A0_size; i++) {
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    if (pA1_begin == pA1_end) continue;

    float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

    for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;

      // Prefetch next B row
      if (pA1 + 1 < pA1_end) {
        __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + 1] * (size_t)B1_size, 0, 1);
      }

#ifdef __ARM_NEON
      float32x4_t va = vdupq_n_f32(a_val);
      int k = 0;
      // Process 16 floats per iteration (4 NEON registers x 4 floats)
      for (; k + 15 < B1_size; k += 16) {
        float32x4_t c0 = vld1q_f32(C_row + k);
        float32x4_t c1 = vld1q_f32(C_row + k + 4);
        float32x4_t c2 = vld1q_f32(C_row + k + 8);
        float32x4_t c3 = vld1q_f32(C_row + k + 12);
        float32x4_t b0 = vld1q_f32(B_row + k);
        float32x4_t b1 = vld1q_f32(B_row + k + 4);
        float32x4_t b2 = vld1q_f32(B_row + k + 8);
        float32x4_t b3 = vld1q_f32(B_row + k + 12);
        c0 = vfmaq_f32(c0, va, b0);
        c1 = vfmaq_f32(c1, va, b1);
        c2 = vfmaq_f32(c2, va, b2);
        c3 = vfmaq_f32(c3, va, b3);
        vst1q_f32(C_row + k, c0);
        vst1q_f32(C_row + k + 4, c1);
        vst1q_f32(C_row + k + 8, c2);
        vst1q_f32(C_row + k + 12, c3);
      }
      // Handle 4-wide remainder
      for (; k + 3 < B1_size; k += 4) {
        float32x4_t c = vld1q_f32(C_row + k);
        float32x4_t b = vld1q_f32(B_row + k);
        c = vfmaq_f32(c, va, b);
        vst1q_f32(C_row + k, c);
      }
      // Scalar remainder
      for (; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
#else
      // Fallback: scalar with 16-wide unroll
      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        C_row[k]      += a_val * B_row[k];
        C_row[k + 1]  += a_val * B_row[k + 1];
        C_row[k + 2]  += a_val * B_row[k + 2];
        C_row[k + 3]  += a_val * B_row[k + 3];
        C_row[k + 4]  += a_val * B_row[k + 4];
        C_row[k + 5]  += a_val * B_row[k + 5];
        C_row[k + 6]  += a_val * B_row[k + 6];
        C_row[k + 7]  += a_val * B_row[k + 7];
        C_row[k + 8]  += a_val * B_row[k + 8];
        C_row[k + 9]  += a_val * B_row[k + 9];
        C_row[k + 10] += a_val * B_row[k + 10];
        C_row[k + 11] += a_val * B_row[k + 11];
        C_row[k + 12] += a_val * B_row[k + 12];
        C_row[k + 13] += a_val * B_row[k + 13];
        C_row[k + 14] += a_val * B_row[k + 14];
        C_row[k + 15] += a_val * B_row[k + 15];
      }
      for (; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
#endif
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// Variant 3: Multi-Row Merge for B-Row Reuse
// Hypothesis: Processing a panel of R rows together lets us load each B-row
// once and scatter to all rows that share that column, reducing total B-row loads.
Tensor spmm_csr_float_row_panel(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  constexpr int PANEL_SIZE = 8;
  int num_panels = (A0_size + PANEL_SIZE - 1) / PANEL_SIZE;

  struct PanelEntry {
    int col;
    int row;   // row index within panel (0 to PANEL_SIZE-1)
    float val;
  };

  size_t max_panel_nnz = 0;
  for (int panel = 0; panel < num_panels; ++panel) {
    const int i_start = panel * PANEL_SIZE;
    const int i_end = std::min(i_start + PANEL_SIZE, A0_size);
    const size_t panel_nnz = static_cast<size_t>(
        A1_pos[i_end] - A1_pos[i_start]);
    max_panel_nnz = std::max(max_panel_nnz, panel_nnz);
  }
  const size_t entries_stride = ((max_panel_nnz + 15) / 16) * 16;
  auto entries_by_thread = scorch_make_unique_array_pool<PanelEntry>(
      static_cast<size_t>(omp_get_max_threads()), entries_stride);

  #pragma omp parallel
  {
    PanelEntry* SCORCH_RESTRICT entries =
        entries_by_thread.get() +
        static_cast<size_t>(omp_get_thread_num()) * entries_stride;

    #pragma omp for schedule(dynamic)
    for (int panel = 0; panel < num_panels; panel++) {
      int i_start = panel * PANEL_SIZE;
      int i_end = std::min(i_start + PANEL_SIZE, A0_size);
      int panel_rows = i_end - i_start;

      // Collect all (col, row_in_panel, a_val) entries
      int total_nnz = 0;
      for (int r = 0; r < panel_rows; r++) {
        total_nnz += A1_pos[i_start + r + 1] - A1_pos[i_start + r];
      }

      if (total_nnz == 0) continue;

      int num_entries = 0;
      for (int r = 0; r < panel_rows; r++) {
        int i = i_start + r;
        for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
          entries[num_entries++] = {A1_crd[pA1], r, A_val[pA1]};
        }
      }

      // Sort by column for B-row reuse
      std::sort(entries, entries + num_entries,
                [](const PanelEntry& a, const PanelEntry& b) {
                  return a.col < b.col;
                });

      // Process sorted entries - B row stays in cache across entries with same col
      int idx = 0;
      while (idx < num_entries) {
        int j = entries[idx].col;
        const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;

        // Prefetch next unique column's B row
        int next_idx = idx + 1;
        while (next_idx < num_entries && entries[next_idx].col == j) next_idx++;
        if (next_idx < num_entries) {
          __builtin_prefetch(
              B_val + (size_t)entries[next_idx].col * (size_t)B1_size, 0, 1);
        }

        // Scatter to all rows in panel that reference this column
        while (idx < num_entries && entries[idx].col == j) {
          float a_val = entries[idx].val;
          float* SCORCH_RESTRICT C_row =
              C_values + (size_t)(i_start + entries[idx].row) * (size_t)C1_size;

          int k = 0;
          for (; k + 15 < B1_size; k += 16) {
            C_row[k]      += a_val * B_row[k];
            C_row[k + 1]  += a_val * B_row[k + 1];
            C_row[k + 2]  += a_val * B_row[k + 2];
            C_row[k + 3]  += a_val * B_row[k + 3];
            C_row[k + 4]  += a_val * B_row[k + 4];
            C_row[k + 5]  += a_val * B_row[k + 5];
            C_row[k + 6]  += a_val * B_row[k + 6];
            C_row[k + 7]  += a_val * B_row[k + 7];
            C_row[k + 8]  += a_val * B_row[k + 8];
            C_row[k + 9]  += a_val * B_row[k + 9];
            C_row[k + 10] += a_val * B_row[k + 10];
            C_row[k + 11] += a_val * B_row[k + 11];
            C_row[k + 12] += a_val * B_row[k + 12];
            C_row[k + 13] += a_val * B_row[k + 13];
            C_row[k + 14] += a_val * B_row[k + 14];
            C_row[k + 15] += a_val * B_row[k + 15];
          }
          for (; k < B1_size; k++) {
            C_row[k] += a_val * B_row[k];
          }
          idx++;
        }
      }
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// Variant 4: K-Parallel with Direct Output
// Hypothesis: K-tile parallelism combined with direct-to-C writes eliminates
// workspace overhead while keeping well-balanced k-parallel structure.
// Each thread owns disjoint C columns, so no workspace or atomics needed.
Tensor spmm_csr_float_k_parallel(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Tile size = 32 floats = 128 bytes = 1 Apple Silicon cache line
  constexpr int kTile = 32;
  int num_k_tiles = (B1_size + kTile - 1) / kTile;

  #pragma omp parallel for schedule(static)
  for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    int k_start = k_tile * kTile;
    int k_end = std::min(k_start + kTile, B1_size);
    int k_width = k_end - k_start;

    for (int i = 0; i < A0_size; i++) {
      int pA1_begin = A1_pos[i];
      int pA1_end = A1_pos[i + 1];
      if (pA1_begin == pA1_end) continue;

      float* SCORCH_RESTRICT C_ptr = C_values + (size_t)i * (size_t)C1_size + k_start;

      for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
        int j = A1_crd[pA1];
        float a_val = A_val[pA1];
        const float* SCORCH_RESTRICT B_ptr =
            B_val + (size_t)j * (size_t)B1_size + k_start;

        // Direct accumulation into C (no workspace needed)
        if (SCORCH_LIKELY(k_width == kTile)) {
          SCORCH_PRAGMA_UNROLL
          for (int k = 0; k < kTile; k++) {
            C_ptr[k] += a_val * B_ptr[k];
          }
        } else {
          for (int k = 0; k < k_width; k++) {
            C_ptr[k] += a_val * B_ptr[k];
          }
        }
      }
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// Variant 5: Row-Sorted by NNZ Count
// Hypothesis: Sorting rows by nnz count improves load balancing and enables
// density-specific code paths without branch misprediction.
Tensor spmm_csr_float_sorted_rows(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Sort row indices by nnz count (descending)
  std::vector<int> sorted_rows(A0_size);
  std::iota(sorted_rows.begin(), sorted_rows.end(), 0);
  std::sort(sorted_rows.begin(), sorted_rows.end(), [&](int a, int b) {
    return (A1_pos[a + 1] - A1_pos[a]) > (A1_pos[b + 1] - A1_pos[b]);
  });

  // Find boundary indices in sorted order
  constexpr int SPARSE_THRESHOLD = 4;
  int empty_start = A0_size;
  int sparse_start = A0_size;
  for (int idx = 0; idx < A0_size; idx++) {
    int nnz = A1_pos[sorted_rows[idx] + 1] - A1_pos[sorted_rows[idx]];
    if (nnz == 0) {
      empty_start = idx;
      if (sparse_start == A0_size) sparse_start = idx;
      break;
    }
    if (nnz <= SPARSE_THRESHOLD && sparse_start == A0_size) {
      sparse_start = idx;
    }
  }

  // Dense rows (nnz > threshold): tiled processing with workspace
  constexpr int kTile = 128;

  #pragma omp parallel
  {
    float workspace[kTile];

    #pragma omp for schedule(static)
    for (int idx = 0; idx < sparse_start; idx++) {
      int i = sorted_rows[idx];
      int pA1_begin = A1_pos[i];
      int pA1_end = A1_pos[i + 1];
      float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

      for (int k_out = 0; k_out < B1_size; k_out += kTile) {
        int k_width = std::min(kTile, B1_size - k_out);
        memset(workspace, 0, sizeof(float) * k_width);

        for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
          int j = A1_crd[pA1];
          float a_val = A_val[pA1];
          const float* SCORCH_RESTRICT B_ptr =
              B_val + (size_t)j * (size_t)B1_size + k_out;

          if (pA1 + 1 < pA1_end) {
            __builtin_prefetch(
                B_val + (size_t)A1_crd[pA1 + 1] * (size_t)B1_size + k_out, 0, 3);
          }

          SCORCH_PRAGMA_UNROLL
          for (int k = 0; k < k_width; k++) {
            workspace[k] += a_val * B_ptr[k];
          }
        }

        for (int k = 0; k < k_width; k++) {
          C_row[k_out + k] = workspace[k];
        }
      }
    }
  }

  // Sparse rows (nnz <= threshold): direct saxpy, no tiling overhead
  #pragma omp parallel for schedule(static)
  for (int idx = sparse_start; idx < empty_start; idx++) {
    int i = sorted_rows[idx];
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

    for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;

      for (int k = 0; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
  }

  // Empty rows: skip entirely (C is already zeroed)

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}


// ════════════════════════════════════════════════════════════════════════════
// Round 2 optimizations: targeting SuiteSparse SpMM with large k
// ════════════════════════════════════════════════════════════════════════════

// Variant 7: NEON 2-NNZ Unroll + Deep Prefetch
// Process pairs of nnz entries per inner loop to halve C reads/writes,
// provide 2 independent FMA chains for ILP, and prefetch 3 B-rows ahead.
Tensor spmm_csr_float_neon2(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < A0_size; i++) {
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    if (pA1_begin >= pA1_end) continue;

    float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;
    int nnz_count = pA1_end - pA1_begin;
    int pA1 = pA1_begin;

    // Process pairs of nnz
    int pair_end = pA1_begin + (nnz_count / 2) * 2;

#ifdef __ARM_NEON
    for (; pA1 < pair_end; pA1 += 2) {
      int j0 = A1_crd[pA1];
      int j1 = A1_crd[pA1 + 1];
      float a_val0 = A_val[pA1];
      float a_val1 = A_val[pA1 + 1];
      const float* SCORCH_RESTRICT B_row0 = B_val + (size_t)j0 * (size_t)B1_size;
      const float* SCORCH_RESTRICT B_row1 = B_val + (size_t)j1 * (size_t)B1_size;

      // Deep prefetch: 3 B-rows ahead
      if (pA1 + 2 < pA1_end) {
        __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + 2] * (size_t)B1_size, 0, 1);
      }
      if (pA1 + 3 < pA1_end) {
        __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + 3] * (size_t)B1_size, 0, 1);
      }

      float32x4_t va0 = vdupq_n_f32(a_val0);
      float32x4_t va1 = vdupq_n_f32(a_val1);

      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        // Load C once
        float32x4_t c0 = vld1q_f32(C_row + k);
        float32x4_t c1 = vld1q_f32(C_row + k + 4);
        float32x4_t c2 = vld1q_f32(C_row + k + 8);
        float32x4_t c3 = vld1q_f32(C_row + k + 12);

        // FMA chain 1: B_row0
        c0 = vfmaq_f32(c0, va0, vld1q_f32(B_row0 + k));
        c1 = vfmaq_f32(c1, va0, vld1q_f32(B_row0 + k + 4));
        c2 = vfmaq_f32(c2, va0, vld1q_f32(B_row0 + k + 8));
        c3 = vfmaq_f32(c3, va0, vld1q_f32(B_row0 + k + 12));

        // FMA chain 2: B_row1
        c0 = vfmaq_f32(c0, va1, vld1q_f32(B_row1 + k));
        c1 = vfmaq_f32(c1, va1, vld1q_f32(B_row1 + k + 4));
        c2 = vfmaq_f32(c2, va1, vld1q_f32(B_row1 + k + 8));
        c3 = vfmaq_f32(c3, va1, vld1q_f32(B_row1 + k + 12));

        // Store C once
        vst1q_f32(C_row + k, c0);
        vst1q_f32(C_row + k + 4, c1);
        vst1q_f32(C_row + k + 8, c2);
        vst1q_f32(C_row + k + 12, c3);
      }
      for (; k + 3 < B1_size; k += 4) {
        float32x4_t c = vld1q_f32(C_row + k);
        c = vfmaq_f32(c, va0, vld1q_f32(B_row0 + k));
        c = vfmaq_f32(c, va1, vld1q_f32(B_row1 + k));
        vst1q_f32(C_row + k, c);
      }
      for (; k < B1_size; k++) {
        C_row[k] += a_val0 * B_row0[k] + a_val1 * B_row1[k];
      }
    }

    // Handle odd trailing nnz
    for (; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;
      float32x4_t va = vdupq_n_f32(a_val);
      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        float32x4_t c0 = vld1q_f32(C_row + k);
        float32x4_t c1 = vld1q_f32(C_row + k + 4);
        float32x4_t c2 = vld1q_f32(C_row + k + 8);
        float32x4_t c3 = vld1q_f32(C_row + k + 12);
        c0 = vfmaq_f32(c0, va, vld1q_f32(B_row + k));
        c1 = vfmaq_f32(c1, va, vld1q_f32(B_row + k + 4));
        c2 = vfmaq_f32(c2, va, vld1q_f32(B_row + k + 8));
        c3 = vfmaq_f32(c3, va, vld1q_f32(B_row + k + 12));
        vst1q_f32(C_row + k, c0);
        vst1q_f32(C_row + k + 4, c1);
        vst1q_f32(C_row + k + 8, c2);
        vst1q_f32(C_row + k + 12, c3);
      }
      for (; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
#else
    // Scalar fallback
    for (; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;
      for (int k = 0; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
#endif
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}


// Variant 8: NEON 4-NNZ Unroll + Deep Prefetch
// Process quads of nnz entries for maximum ILP: 4 independent FMA chains,
// quarter the C reads/writes, prefetch 5 B-rows ahead.
Tensor spmm_csr_float_neon4(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < A0_size; i++) {
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    if (pA1_begin >= pA1_end) continue;

    float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;
    int nnz_count = pA1_end - pA1_begin;
    int pA1 = pA1_begin;
    int quad_end = pA1_begin + (nnz_count / 4) * 4;

#ifdef __ARM_NEON
    // Process quads of nnz
    for (; pA1 < quad_end; pA1 += 4) {
      float a0 = A_val[pA1], a1 = A_val[pA1+1], a2 = A_val[pA1+2], a3 = A_val[pA1+3];
      const float* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA1]   * (size_t)B1_size;
      const float* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA1+1] * (size_t)B1_size;
      const float* SCORCH_RESTRICT B2 = B_val + (size_t)A1_crd[pA1+2] * (size_t)B1_size;
      const float* SCORCH_RESTRICT B3 = B_val + (size_t)A1_crd[pA1+3] * (size_t)B1_size;

      // Deep prefetch: 5 B-rows ahead
      for (int pf = 4; pf < 8 && pA1 + pf < pA1_end; pf++) {
        __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + pf] * (size_t)B1_size, 0, 1);
      }

      float32x4_t va0 = vdupq_n_f32(a0);
      float32x4_t va1 = vdupq_n_f32(a1);
      float32x4_t va2 = vdupq_n_f32(a2);
      float32x4_t va3 = vdupq_n_f32(a3);

      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        float32x4_t c0 = vld1q_f32(C_row + k);
        float32x4_t c1 = vld1q_f32(C_row + k + 4);
        float32x4_t c2 = vld1q_f32(C_row + k + 8);
        float32x4_t c3 = vld1q_f32(C_row + k + 12);

        c0 = vfmaq_f32(c0, va0, vld1q_f32(B0 + k));
        c1 = vfmaq_f32(c1, va0, vld1q_f32(B0 + k + 4));
        c2 = vfmaq_f32(c2, va0, vld1q_f32(B0 + k + 8));
        c3 = vfmaq_f32(c3, va0, vld1q_f32(B0 + k + 12));

        c0 = vfmaq_f32(c0, va1, vld1q_f32(B1 + k));
        c1 = vfmaq_f32(c1, va1, vld1q_f32(B1 + k + 4));
        c2 = vfmaq_f32(c2, va1, vld1q_f32(B1 + k + 8));
        c3 = vfmaq_f32(c3, va1, vld1q_f32(B1 + k + 12));

        c0 = vfmaq_f32(c0, va2, vld1q_f32(B2 + k));
        c1 = vfmaq_f32(c1, va2, vld1q_f32(B2 + k + 4));
        c2 = vfmaq_f32(c2, va2, vld1q_f32(B2 + k + 8));
        c3 = vfmaq_f32(c3, va2, vld1q_f32(B2 + k + 12));

        c0 = vfmaq_f32(c0, va3, vld1q_f32(B3 + k));
        c1 = vfmaq_f32(c1, va3, vld1q_f32(B3 + k + 4));
        c2 = vfmaq_f32(c2, va3, vld1q_f32(B3 + k + 8));
        c3 = vfmaq_f32(c3, va3, vld1q_f32(B3 + k + 12));

        vst1q_f32(C_row + k, c0);
        vst1q_f32(C_row + k + 4, c1);
        vst1q_f32(C_row + k + 8, c2);
        vst1q_f32(C_row + k + 12, c3);
      }
      for (; k + 3 < B1_size; k += 4) {
        float32x4_t c = vld1q_f32(C_row + k);
        c = vfmaq_f32(c, va0, vld1q_f32(B0 + k));
        c = vfmaq_f32(c, va1, vld1q_f32(B1 + k));
        c = vfmaq_f32(c, va2, vld1q_f32(B2 + k));
        c = vfmaq_f32(c, va3, vld1q_f32(B3 + k));
        vst1q_f32(C_row + k, c);
      }
      for (; k < B1_size; k++) {
        C_row[k] += a0 * B0[k] + a1 * B1[k] + a2 * B2[k] + a3 * B3[k];
      }
    }

    // Handle remainder (1-3 nnz) with single-nnz NEON
    for (; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;
      float32x4_t va = vdupq_n_f32(a_val);
      int k = 0;
      for (; k + 15 < B1_size; k += 16) {
        vst1q_f32(C_row + k,      vfmaq_f32(vld1q_f32(C_row + k),      va, vld1q_f32(B_row + k)));
        vst1q_f32(C_row + k + 4,  vfmaq_f32(vld1q_f32(C_row + k + 4),  va, vld1q_f32(B_row + k + 4)));
        vst1q_f32(C_row + k + 8,  vfmaq_f32(vld1q_f32(C_row + k + 8),  va, vld1q_f32(B_row + k + 8)));
        vst1q_f32(C_row + k + 12, vfmaq_f32(vld1q_f32(C_row + k + 12), va, vld1q_f32(B_row + k + 12)));
      }
      for (; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
#else
    for (; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float a_val = A_val[pA1];
      const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size;
      for (int k = 0; k < B1_size; k++) {
        C_row[k] += a_val * B_row[k];
      }
    }
#endif
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}


// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)

// One SIMD vocabulary for both value types, so every register kernel below is
// written once and instantiated twice. The ONLY thing the kernels see differ is
// the lane count -- 8 floats or 4 doubles per YMM register -- and every shape
// constant is expressed in terms of it, so the register budget is identical for
// both types (4 accumulators for the narrow-k kernel, 8 for the wide-k tile) and
// so is the instruction mix. float64 therefore gets the same avoidance of the
// per-nonzero output round-trip that took float32 past MKL, rather than a second
// hand-written kernel that would drift from this one.
//
// The float32 instantiation must compile to what the hand-written float32 kernel
// compiled to: every member here is a one-line `inline` wrapper over the same
// intrinsic, chosen at compile time, so there is nothing left to fold away.
template <typename T> struct scorch_simd;

template <> struct scorch_simd<float> {
  using vec = __m256;
  using mask = __m256i;
  static constexpr int lanes = 8;
  static inline vec zero() { return _mm256_setzero_ps(); }
  static inline vec splat(float x) { return _mm256_set1_ps(x); }
  static inline vec load(const float* p) { return _mm256_loadu_ps(p); }
  static inline vec maskload(const float* p, mask m) { return _mm256_maskload_ps(p, m); }
  static inline void store(float* p, vec v) { _mm256_storeu_ps(p, v); }
  static inline void maskstore(float* p, mask m, vec v) { _mm256_maskstore_ps(p, m, v); }
  static inline vec fma(vec a, vec b, vec c) { return _mm256_fmadd_ps(a, b, c); }
  static inline vec add(vec a, vec b) { return _mm256_add_ps(a, b); }
  // Lanes [0, valid) enabled. Same expression the hand-written kernel inlined.
  static inline mask lane_mask(int valid) {
    return _mm256_setr_epi32(
        valid > 0 ? -1 : 0, valid > 1 ? -1 : 0, valid > 2 ? -1 : 0, valid > 3 ? -1 : 0,
        valid > 4 ? -1 : 0, valid > 5 ? -1 : 0, valid > 6 ? -1 : 0, valid > 7 ? -1 : 0);
  }
};

template <> struct scorch_simd<double> {
  using vec = __m256d;
  using mask = __m256i;                  // 64-bit lanes for the _pd mask forms
  static constexpr int lanes = 4;
  static inline vec zero() { return _mm256_setzero_pd(); }
  static inline vec splat(double x) { return _mm256_set1_pd(x); }
  static inline vec load(const double* p) { return _mm256_loadu_pd(p); }
  static inline vec maskload(const double* p, mask m) { return _mm256_maskload_pd(p, m); }
  static inline void store(double* p, vec v) { _mm256_storeu_pd(p, v); }
  static inline void maskstore(double* p, mask m, vec v) { _mm256_maskstore_pd(p, m, v); }
  static inline vec fma(vec a, vec b, vec c) { return _mm256_fmadd_pd(a, b, c); }
  static inline vec add(vec a, vec b) { return _mm256_add_pd(a, b); }
  static inline mask lane_mask(int valid) {
    return _mm256_setr_epi64x(
        valid > 0 ? -1 : 0, valid > 1 ? -1 : 0, valid > 2 ? -1 : 0, valid > 3 ? -1 : 0);
  }
};

// Register-blocked narrow-k SpMM row kernel: accumulate the whole output row in
// NVEC YMM registers across the row's nonzeros (2-nnz ILP for two independent
// FMA chains), with a masked load/store on the final partial vector. Avoids the
// workspace round-trip (memset + per-nnz ws load/store + memcpy) that dominated
// MKL for narrow k — the 0.5-0.8x gap at k<=16, and the k=32 dense case.
// NVEC = ceil(k/8), specialized 1..4 (k<=32).
//
// FULL_LAST says the last vector is entirely valid, i.e. k is a multiple of 8 --
// the common narrow-k case, k = 8, 16, 24, 32 -- and it used to be handled with
// the same masked load and store as a genuinely ragged tail. On Intel that is not
// free: vmaskmovps takes 2 uops on the load side and cannot fold into the FMA as a
// memory operand, and its store form is worse. At k=16 that made HALF of every
// row's B loads masked for no reason, plus one masked store per row, and on a
// short row the store is amortized over only a handful of FMAs. Both collapse to
// plain vmovups when FULL_LAST holds. Ragged k keeps the mask, unchanged.
template <typename T, int NVEC, bool FULL_LAST>
static inline void scorch_spmm_row_regblock(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    typename scorch_simd<T>::mask mask_last) {
  using V = scorch_simd<T>;
  constexpr int L = V::lanes;
  typename V::vec acc0[NVEC], acc1[NVEC];
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    acc0[v] = V::zero();
    acc1[v] = V::zero();
  }
  int pA = pA_begin;
  for (; pA + 1 < pA_end; pA += 2) {
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const T* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size;
    if (pA + 2 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + 2] * (size_t)B1_size, 0, 1);
    const typename V::vec a0 = V::splat(A_val[pA]);
    const typename V::vec a1 = V::splat(A_val[pA + 1]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool masked = (v == NVEC - 1) && !FULL_LAST;   // compile-time
      const typename V::vec b0 = masked ? V::maskload(B0 + L * v, mask_last)
                                        : V::load(B0 + L * v);
      const typename V::vec b1 = masked ? V::maskload(B1 + L * v, mask_last)
                                        : V::load(B1 + L * v);
      acc0[v] = V::fma(a0, b0, acc0[v]);
      acc1[v] = V::fma(a1, b1, acc1[v]);
    }
  }
  if (pA < pA_end) {
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const typename V::vec a0 = V::splat(A_val[pA]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool masked = (v == NVEC - 1) && !FULL_LAST;   // compile-time
      const typename V::vec b0 = masked ? V::maskload(B0 + L * v, mask_last)
                                        : V::load(B0 + L * v);
      acc0[v] = V::fma(a0, b0, acc0[v]);
    }
  }
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    const typename V::vec r = V::add(acc0[v], acc1[v]);
    if ((v == NVEC - 1) && !FULL_LAST) V::maskstore(C_row + L * v, mask_last, r);
    else V::store(C_row + L * v, r);
  }
}

#if defined(__AVX2__) && defined(__FMA__)
// Narrow-k row kernel that vectorises across NONZEROS instead of across the output
// row: the eight lanes hold eight different nonzeros of the same row, reduced at
// the end, so every lane carries useful work at any k.
//
// The shipped regblock kernel puts the output row in the lanes, which means at
// k=1 seven of eight lanes are mask and each nonzero costs a full-width masked
// load and FMA to produce one float. Measured over the corpus that is where the
// kernel loses: at k<=8 with more than 64 nonzeros per row it runs 0.866 of MKL
// (9604 cells, 84% below parity), and those rows average 195 nonzeros with a
// length spread of 0.13, so a nonzero-axis loop always has a full vector to fill.
//
// The trade is instruction count against gather throughput: one VGATHERDPS is
// ~4-5 cycles and replaces eight masked loads, but K of them are needed to cover K
// output columns while the regblock kernel needs eight masked FMAs whatever K is.
// Measured over 160 matrices spanning mean row 1 to >1000, interleaved arms, A/A
// p95 0.085 at k=1:
//
//   k=1  1.067x  (wins in 8 of 8 degree bins, 118/160 cells)
//   k=2  0.819-1.008
//   k=4  0.686-0.940
//   k=8  0.495-0.824
//
// So it ships for k=1 only -- not a round number picked for tidiness, but where K
// gathers stop being cheaper than eight masked FMAs. At k=1 it moves the kernel
// from 0.714 to 0.762 of MKL and regresses none of the 49 panel cells that were
// already at or above parity. The K=1..8 instantiations stay so the losing half of
// that map can be re-measured from a shipped binary via SCORCH_NARROWK_GATHER.
//
// float only. The double path would need _mm256_i32gather_pd at 4 lanes, which
// halves the instruction saving; it stays on the regblock kernel.
template <int K>
static inline void scorch_spmm_row_gather_f32(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  __m256 acc[K];
  for (int j = 0; j < K; j++) acc[j] = _mm256_setzero_ps();

  int pA = pA_begin;
  for (; pA + 8 <= pA_end; pA += 8) {
    const __m256i idx = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(A1_crd + pA));
    // element offset of column c's row in B is c*K; the gather's scale covers
    // the 4-byte element size, so the index has to carry the K stride.
    const __m256i off = (K == 1) ? idx
                                 : _mm256_mullo_epi32(idx, _mm256_set1_epi32(K));
    const __m256 a = _mm256_loadu_ps(A_val + pA);
    for (int j = 0; j < K; j++)
      acc[j] = _mm256_fmadd_ps(a, _mm256_i32gather_ps(B_val + j, off, 4), acc[j]);
  }

  // Fold each lane set down to one scalar, then finish the row's tail.
  for (int j = 0; j < K; j++) {
    __m128 lo = _mm256_castps256_ps128(acc[j]);
    __m128 hi = _mm256_extractf128_ps(acc[j], 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (int q = pA; q < pA_end; q++)
      sum += A_val[q] * B_val[(size_t)A1_crd[q] * (size_t)K + j];
    C_row[j] = sum;
  }
}
#endif  // AVX2 && FMA
#if defined(__AVX2__) && defined(__FMA__)
// Multi-stream nonzero-axis gather: S independent gather+FMA chains instead of one.
//
// The single-stream kernel above wins at k=1, and the reason it still loses to MKL
// there is written in its own comment -- the profile is memory LATENCY, not FMA
// throughput, and MKL's SpMV-shaped loop keeps eight or more loads in flight. One
// VGATHERDPS is one outstanding memory operation covering eight nonzeros; the
// accumulator it feeds is the loop's only carried dependency, so consecutive
// iterations can overlap in principle, but the measured cost per nonzero on the
// L3-resident band is far above what either gather throughput or FMA latency
// accounts for, which is what an uncovered load latency looks like.
//
// The existing SCORCH_NARROWK_UNROLL hook does NOT test this. It deepens the
// REGBLOCK kernel, and at k=1 float32 the shipped path is the gather kernel, so
// turning it on swaps the kernel family and the stream count in the same step --
// and the regblock family is already 0.903 of the gather at k=1 on the losing band,
// so that arm starts ten percent behind. This changes the stream count and nothing
// else.
//
// S streams need K*S accumulators plus S index vectors and S value vectors live at
// once, so K*S is held to about half the 16 architectural YMM registers.
//
// MEASURED, AND IT IS A NULL. Over the 118 matrices below MKL parity at k<=2, on
// redwood, kernel timer, A/A 1.5% of cells outside +-10%:
//   S=2/4/8 against the shipped single stream
//     k=1  0.968 / 0.972 / 0.968      k=2  0.994 / 0.995 / 0.997
//     k=4  1.001 / 0.994 / 1.001      k=8  1.000 / 1.000 / 0.999
// float64 is 0.994-1.014 throughout. So the deficit at k=1 is NOT an uncovered load
// latency that more outstanding gathers can hide -- the out-of-order window was
// already overlapping consecutive iterations, and the accumulator chain the extra
// streams break was never the limit. The comment on the regblock_deep hook below
// reasons from the same premise and should be read with this result next to it.
//
// Kept, not deleted: it is the only way to re-price stream depth from a shipped
// binary, and the null is the useful part of it. S == 1 is what ships and routes to
// the single-stream kernel above, byte for byte.
template <int K, int S>
static inline void scorch_spmm_row_gather_f32_ms(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  __m256 acc[K][S];
  #pragma unroll
  for (int j = 0; j < K; j++)
    #pragma unroll
    for (int t = 0; t < S; t++) acc[j][t] = _mm256_setzero_ps();

  int pA = pA_begin;
  // The S index loads and value loads are issued before any FMA consumes them, so
  // the gathers they feed are all in flight together rather than one per iteration.
  for (; pA + 8 * S <= pA_end; pA += 8 * S) {
    __m256i off[S];
    __m256 av[S];
    #pragma unroll
    for (int t = 0; t < S; t++) {
      const __m256i idx = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(A1_crd + pA + 8 * t));
      off[t] = (K == 1) ? idx
                        : _mm256_mullo_epi32(idx, _mm256_set1_epi32(K));
      av[t] = _mm256_loadu_ps(A_val + pA + 8 * t);
    }
    #pragma unroll
    for (int j = 0; j < K; j++)
      #pragma unroll
      for (int t = 0; t < S; t++)
        acc[j][t] = _mm256_fmadd_ps(av[t],
                                    _mm256_i32gather_ps(B_val + j, off[t], 4),
                                    acc[j][t]);
  }
  // Whatever is left of the row that still fills one vector, single-stream.
  for (; pA + 8 <= pA_end; pA += 8) {
    const __m256i idx = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(A1_crd + pA));
    const __m256i off0 = (K == 1) ? idx
                                  : _mm256_mullo_epi32(idx, _mm256_set1_epi32(K));
    const __m256 a = _mm256_loadu_ps(A_val + pA);
    #pragma unroll
    for (int j = 0; j < K; j++)
      acc[j][0] = _mm256_fmadd_ps(a, _mm256_i32gather_ps(B_val + j, off0, 4),
                                  acc[j][0]);
  }

  #pragma unroll
  for (int j = 0; j < K; j++) {
    __m256 tot = acc[j][0];
    #pragma unroll
    for (int t = 1; t < S; t++) tot = _mm256_add_ps(tot, acc[j][t]);
    __m128 lo = _mm256_castps256_ps128(tot);
    __m128 hi = _mm256_extractf128_ps(tot, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (int q = pA; q < pA_end; q++)
      sum += A_val[q] * B_val[(size_t)A1_crd[q] * (size_t)K + j];
    C_row[j] = sum;
  }
}
#endif  // AVX2 && FMA
#if defined(__AVX2__) && defined(__FMA__) && defined(SCORCH_TUNE_HOOKS)
// EXACT-WIDTH narrow-k row kernel, for the widths where the register-block kernel's
// lane mask covers the WHOLE output row.
//
// regblock holds the row in NVEC full vector registers and masks the last partial
// one. When NVEC is 1 and k is not the lane count, that mask is on the only vector,
// so every load in the row and the store go through it -- at k=2 float32, two useful
// lanes in eight, and vmaskmovps is two uops on the load side and cannot fold into
// the FMA as a memory operand. (The FULL_LAST specialisation above removes the mask
// only when k IS a multiple of the lane count.) Separately, B1_size is a runtime
// value there, so every nonzero costs an imul to find its B row: three per two
// nonzeros counting the prefetch.
//
// Both disappear once k is a template parameter. Rather than hand-write a load for
// each width, the accumulators are a compile-time-sized scalar array and the compiler
// picks the decomposition -- which it does better than by hand, because a leftover
// element folds into a scalar FMA as a memory operand and costs no instruction at
// all. Cross-compiled and disassembled for every instantiation: ZERO masked
// operations and ZERO multiplies, and no width reads a byte past its own row. k=3
// float32 comes out as
//
//   movslq (%rdi,%rax,4), %r8            ; the column index
//   leaq   (%r8,%r8,2), %r14             ; times three, no imul
//   vbroadcastss (%rsi,%rax,4), %xmm2    ; the A value, load and splat in one
//   vfmadd231ss 0x8(%rdx,%r14,4), %xmm2, %xmm9    ; element 2, folded
//   vmovsd (%rdx,%r14,4), %xmm11         ; elements 0-1, exactly 8 bytes
//   vfmadd231ps %xmm11, %xmm2, %xmm10
//
// six instructions per nonzero for twelve bytes read exactly, against regblock's
// ~10.5 for a masked thirty-two.
//
// The range is exactly the widths where regblock masks the entire row: float k=1..7
// and double k=1..3. At float k=8 and double k=4 the last vector is full and regblock
// is already mask-free, and the grid confirms this kernel is neutral there. Float k=1
// is instantiated but not routed by default -- the nonzero-axis gather kernel owns it
// and measures better -- so the two stay separately attributable.
template <typename T, int K, int UNROLL>
static inline void scorch_spmm_row_narrow_exact(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, T* SCORCH_RESTRICT C_row,
    int pA_begin, int pA_end) {
  T acc[UNROLL][K];
  #pragma unroll
  for (int u = 0; u < UNROLL; u++) {
    #pragma unroll
    for (int j = 0; j < K; j++) acc[u][j] = T(0);
  }
  int pA = pA_begin;
  for (; pA + UNROLL <= pA_end; pA += UNROLL) {
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      // K is a compile-time constant, so this is a shift or an lea, never a multiply.
      const T* SCORCH_RESTRICT Bp = B_val + (size_t)A1_crd[pA + u] * (size_t)K;
      const T a = A_val[pA + u];
      #pragma unroll
      for (int j = 0; j < K; j++) acc[u][j] += a * Bp[j];
    }
  }
  #pragma unroll
  for (int u = 1; u < UNROLL; u++) {
    #pragma unroll
    for (int j = 0; j < K; j++) acc[0][j] += acc[u][j];
  }
  for (; pA < pA_end; pA++) {
    const T* SCORCH_RESTRICT Bp = B_val + (size_t)A1_crd[pA] * (size_t)K;
    const T a = A_val[pA];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[0][j] += a * Bp[j];
  }
  #pragma unroll
  for (int j = 0; j < K; j++) C_row[j] = acc[0][j];
}

// Narrow-k variant that runs UNROLL independent nonzero streams instead of 2.
//
// Why: at k <= lanes the row needs a single vector accumulator (NVEC == 1), so the
// 2-nnz form above keeps only two B loads in flight. Those loads are the random
// part of an SpMM -- one cache line per nonzero, indexed by the column -- and the
// whole-collection sweep shows the deficit against MKL tracking how far B sits
// from the core: with B under 32 KB (L1-resident) the kernel is 0.894 of MKL, but
// 0.591 once B is L2-sized and 0.489 once it is L3-sized. That is a memory-latency
// profile, not an FMA-throughput one, and two outstanding loads cannot cover an L2
// hit. MKL's SpMV-shaped kernel keeps 8 or more in flight.
//
// NVEC * UNROLL accumulators have to stay in registers, so the caller picks UNROLL
// to hold that product at about 8 of the 16 architectural YMM registers, leaving
// room for the B values and the splats. Rows shorter than UNROLL fall entirely to
// the scalar tail, which has LESS instruction-level parallelism than the 2-stream
// form it replaces, so a deep unroll is expected to lose on short rows -- that is
// what the A/B measures, and it is why this is a hook and not yet a policy.
template <typename T, int NVEC, bool FULL_LAST, int UNROLL>
static inline void scorch_spmm_row_regblock_deep(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    typename scorch_simd<T>::mask mask_last) {
  using V = scorch_simd<T>;
  constexpr int L = V::lanes;
  typename V::vec acc[UNROLL][NVEC];
  for (int u = 0; u < UNROLL; u++)
    for (int v = 0; v < NVEC; v++) acc[u][v] = V::zero();

  int pA = pA_begin;
  for (; pA + UNROLL <= pA_end; pA += UNROLL) {
    // Resolve every base pointer and scalar first, so the UNROLL load chains are
    // visibly independent and the out-of-order engine can overlap their misses.
    const T* SCORCH_RESTRICT Bp[UNROLL];
    typename V::vec av[UNROLL];
    for (int u = 0; u < UNROLL; u++) {
      Bp[u] = B_val + (size_t)A1_crd[pA + u] * (size_t)B1_size;
      av[u] = V::splat(A_val[pA + u]);
    }
    for (int u = 0; u < UNROLL; u++) {
      for (int v = 0; v < NVEC; v++) {
        const bool masked = (v == NVEC - 1) && !FULL_LAST;   // compile-time
        const typename V::vec b = masked ? V::maskload(Bp[u] + L * v, mask_last)
                                         : V::load(Bp[u] + L * v);
        acc[u][v] = V::fma(av[u], b, acc[u][v]);
      }
    }
  }
  for (; pA < pA_end; pA++) {
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const typename V::vec a0 = V::splat(A_val[pA]);
    for (int v = 0; v < NVEC; v++) {
      const bool masked = (v == NVEC - 1) && !FULL_LAST;
      const typename V::vec b0 = masked ? V::maskload(B0 + L * v, mask_last)
                                        : V::load(B0 + L * v);
      acc[0][v] = V::fma(a0, b0, acc[0][v]);
    }
  }
  for (int v = 0; v < NVEC; v++) {
    typename V::vec r = acc[0][v];
    for (int u = 1; u < UNROLL; u++) r = V::add(r, acc[u][v]);
    if ((v == NVEC - 1) && !FULL_LAST) V::maskstore(C_row + L * v, mask_last, r);
    else V::store(C_row + L * v, r);
  }
}
#endif  // AVX2 && FMA && SCORCH_TUNE_HOOKS
#if defined(__AVX2__) && defined(__FMA__) && defined(SCORCH_TUNE_HOOKS)
// Narrow-k variant whose only difference from the shipped kernel is how far ahead
// it prefetches B.
//
// The shipped kernel prefetches the row's next-but-one nonzero -- one loop
// iteration, about four cycles of work at two nonzeros an iteration. On the
// kernel-only timer the narrow-k deficit against MKL is confined almost exactly to
// the band where a B row is an L2 miss and an L3 hit: over redwood's whole
// collection at k <= 8, with B under half a megabyte scorch reads 0.93-1.88 of MKL
// (float64) and with B over 8 MB it reads 0.96-1.20, but in between, at 0.5-8 MB,
// it reads 0.51-0.95. Four cycles of run-ahead cannot cover a ~40-cycle L3 hit,
// and a deeper distance is the cheapest thing that could. PFD is in NONZEROS, so
// the prefetch issued while working on pA lands PFD/2 iterations early; it is
// clamped to the row, so a short row issues none.
template <typename T, int NVEC, bool FULL_LAST, int PFD>
static inline void scorch_spmm_row_regblock_pf(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    typename scorch_simd<T>::mask mask_last) {
  using V = scorch_simd<T>;
  constexpr int L = V::lanes;
  typename V::vec acc0[NVEC], acc1[NVEC];
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    acc0[v] = V::zero();
    acc1[v] = V::zero();
  }
  int pA = pA_begin;
  for (; pA + 1 < pA_end; pA += 2) {
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const T* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size;
    if (pA + PFD < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + PFD] * (size_t)B1_size, 0, 1);
    if (pA + PFD + 1 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + PFD + 1] * (size_t)B1_size,
                         0, 1);
    const typename V::vec a0 = V::splat(A_val[pA]);
    const typename V::vec a1 = V::splat(A_val[pA + 1]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool masked = (v == NVEC - 1) && !FULL_LAST;   // compile-time
      const typename V::vec b0 = masked ? V::maskload(B0 + L * v, mask_last)
                                        : V::load(B0 + L * v);
      const typename V::vec b1 = masked ? V::maskload(B1 + L * v, mask_last)
                                        : V::load(B1 + L * v);
      acc0[v] = V::fma(a0, b0, acc0[v]);
      acc1[v] = V::fma(a1, b1, acc1[v]);
    }
  }
  if (pA < pA_end) {
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const typename V::vec a0 = V::splat(A_val[pA]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool masked = (v == NVEC - 1) && !FULL_LAST;   // compile-time
      const typename V::vec b0 = masked ? V::maskload(B0 + L * v, mask_last)
                                        : V::load(B0 + L * v);
      acc0[v] = V::fma(a0, b0, acc0[v]);
    }
  }
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    const typename V::vec r = V::add(acc0[v], acc1[v]);
    if ((v == NVEC - 1) && !FULL_LAST) V::maskstore(C_row + L * v, mask_last, r);
    else V::store(C_row + L * v, r);
  }
}
#endif  // AVX2 && FMA && SCORCH_TUNE_HOOKS
#ifdef SCORCH_TUNE_HOOKS
// A/B toggle for the regtile partial-tile path; refreshed once per SpMM op by
// spmm_csr_float_v2 (SCORCH_REGTILE_BASE=1 -> legacy runtime-nv partial). Only
// exists in the instrumented build; the shipped .so always takes the templated
// path with no branch.
static int g_scorch_regtile_base = 0;
#endif

// Ragged-tail partial tile (kw=K%64 in 1..63), templated on the vector count NV
// so the k-loop is COMPILE-TIME unrolled. The earlier form ran a runtime-nv loop
// with a per-element `v==nv-1 ? maskload : loadu` branch that the compiler could
// not unroll — measured (redwood i9-14900K P-core, standalone perf) at 3x the
// instructions and 7x the branches of the equivalent full-64-tile work: the
// partial tile was FRONT-END/branch-bound (IPC ~3.5, little FMA), NOT FMA-port,
// cache, or masking bound (a full-64-tile is L2-bandwidth-bound at IPC ~1.4 and
// already beats MKL). Dispatching on nv (compile-time NV) unrolls the loop, and
// 2-nnz ILP (two accumulator sets, like scorch_spmm_row_regblock) feeds the FMA
// ports even when nv<8. `full` (ml==8) drops the mask on a complete last vector.
// Result: 9-29% fewer cycles on partial-tile K (K%64 != 0), full-64-multiple K
// untouched (kw==0 -> this path never runs). Correctness bit-identical to the
// runtime-nv form (verified K=33/63/65/100/120/127 checksum-parity + the suite).
template <typename T, int NV, bool ACCUM = false>
static inline void scorch_spmm_row_regtile_partial(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    int k0, typename scorch_simd<T>::mask mask, bool full) {
  using V = scorch_simd<T>;
  constexpr int L = V::lanes;
  typename V::vec acc0[NV], acc1[NV];
  // ACCUM seeds the first set from the row, so this kernel serves both the assigning
  // case (the drop-in SpMM owns the whole row) and the accumulating one (the tiled
  // kernels add each contraction panel's contribution to a row later panels also add
  // to). Only acc0 is seeded; acc1 starts at zero and is folded in at the store, so
  // the row is added in exactly once.
  #pragma unroll
  for (int v = 0; v < NV; v++) {
    if constexpr (ACCUM) {
      const bool m = (v == NV - 1 && !full);
      acc0[v] = m ? V::maskload(C_row + k0 + L * v, mask) : V::load(C_row + k0 + L * v);
    } else {
      acc0[v] = V::zero();
    }
    acc1[v] = V::zero();
  }
  int pA = pA_begin;
  for (; pA + 1 < pA_end; pA += 2) {                        // 2-nnz ILP: 2*NV chains
    const T* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA]     * (size_t)B1_size + k0;
    const T* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0;
    if (pA + 2 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + 2] * (size_t)B1_size + k0, 0, 1);
    const typename V::vec a0 = V::splat(A_val[pA]);
    const typename V::vec a1 = V::splat(A_val[pA + 1]);
    #pragma unroll
    for (int v = 0; v < NV; v++) {
      const bool m = (v == NV - 1 && !full);
      const typename V::vec b0 = m ? V::maskload(B0 + L * v, mask) : V::load(B0 + L * v);
      const typename V::vec b1 = m ? V::maskload(B1 + L * v, mask) : V::load(B1 + L * v);
      acc0[v] = V::fma(a0, b0, acc0[v]);
      acc1[v] = V::fma(a1, b1, acc1[v]);
    }
  }
  for (; pA < pA_end; pA++) {                                // odd tail nnz
    const T* SCORCH_RESTRICT Bp = B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
    const typename V::vec a = V::splat(A_val[pA]);
    #pragma unroll
    for (int v = 0; v < NV; v++) {
      const bool m = (v == NV - 1 && !full);
      const typename V::vec b = m ? V::maskload(Bp + L * v, mask) : V::load(Bp + L * v);
      acc0[v] = V::fma(a, b, acc0[v]);
    }
  }
  #pragma unroll
  for (int v = 0; v < NV; v++) {
    const typename V::vec r = V::add(acc0[v], acc1[v]);
    if (v == NV - 1 && !full) V::maskstore(C_row + k0 + L * v, mask, r);
    else V::store(C_row + k0 + L * v, r);
  }
}

// Wide-k register-TILED SpMM row kernel (k>32). For each 64-wide k-tile, hold 8
// YMM accumulators for the output-row segment across ALL of the row's nonzeros,
// so C never round-trips through the workspace (the per-nnz ws load+store that
// dominates the cache-hot wide-k case — banded/locality dense matrices at k=64
// were 0.59x of MKL). 8 independent FMA chains fully feed the two FMA ports. The
// row's A indices/values are re-read once per k-tile (ceil(k/64) passes) but are
// L1-hot; B is streamed once total (each element used once per output row). The
// ragged last tile (<64) uses the templated scorch_spmm_row_regtile_partial (see
// above), K need not be a multiple of 64.
// Ordinary stores, deliberately. Non-temporal stores (vmovntps) were tried here
// behind a gate on C exceeding the last-level cache, on the theory that they skip
// the read-for-ownership pass an ordinary store pays to bring a line it is about
// to overwrite entirely. Measured on redwood over the 19 cells of a SuiteSparse/GCN
// grid where such a gate fires (C from 41 MiB to 1199 MiB against a 36 MiB L3):
// geometric mean 0.9972, range 0.975-1.028, against a 21-cell null of cells the
// gate cannot touch that read 1.0315. The largest outputs -- where the effect
// should have been biggest -- were the ones at or below 1.0. Whatever this part
// does for a full-line write already costs what the read-for-ownership would have,
// so there was nothing to skip. Not worth an alignment gate, a store fence and a
// fault mode (vmovntps faults on a misaligned address and has no unaligned form).
template <typename T>
static inline void scorch_spmm_row_regtile(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  using V = scorch_simd<T>;
  constexpr int L = V::lanes;
  constexpr int TILE = 8 * L;      // 64 floats or 32 doubles: 8 YMM accumulators
  int k0 = 0;
  for (; k0 + TILE <= B1_size; k0 += TILE) {
    typename V::vec acc[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) acc[v] = V::zero();
    for (int pA = pA_begin; pA < pA_end; pA++) {
      const T* SCORCH_RESTRICT Bp =
          B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
      if (pA + 1 < pA_end)
        __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
      const typename V::vec a = V::splat(A_val[pA]);
      #pragma unroll
      for (int v = 0; v < 8; v++)
        acc[v] = V::fma(a, V::load(Bp + L * v), acc[v]);
    }
    #pragma unroll
    for (int v = 0; v < 8; v++) V::store(C_row + k0 + L * v, acc[v]);
  }
  // Final partial tile (kw in 1..TILE-1): the templated compile-time-nv path (2-nnz
  // ILP). A scalar tail here reintroduced the per-nnz round-trip (-0.5-0.9x at
  // k=48/96); the earlier runtime-nv YMM loop was correct but front-end-bound.
  const int kw = B1_size - k0;
  if (kw > 0) {
    const int nv = (kw + L - 1) / L;          // 1..8 vectors
    const int ml = kw - L * (nv - 1);         // 1..L valid lanes in last vector
    const bool full = (ml == L);              // complete last vector -> no mask
    const typename V::mask mask = V::lane_mask(ml);
#ifdef SCORCH_TUNE_HOOKS
    // A/B hook: SCORCH_REGTILE_BASE=1 forces the legacy runtime-nv partial path
    // (single-nnz, no compile-time unroll) for an in-process old-vs-new delta.
    // Read once per SpMM op into g_scorch_regtile_base -- a per-row getenv would
    // swamp the hot loop; a cached function-local static would latch the FIRST
    // op's value and ignore later toggles. Compiled out of the shipped .so.
    if (g_scorch_regtile_base) {
      typename V::vec acc[8];
      for (int v = 0; v < nv; v++) acc[v] = V::zero();
      for (int pA = pA_begin; pA < pA_end; pA++) {
        const T* SCORCH_RESTRICT Bp =
            B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
        if (pA + 1 < pA_end)
          __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
        const typename V::vec a = V::splat(A_val[pA]);
        for (int v = 0; v < nv; v++) {
          const typename V::vec b = (v == nv - 1) ? V::maskload(Bp + L * v, mask)
                                                  : V::load(Bp + L * v);
          acc[v] = V::fma(a, b, acc[v]);
        }
      }
      for (int v = 0; v < nv; v++) {
        if (v == nv - 1) V::maskstore(C_row + k0 + L * v, mask, acc[v]);
        else V::store(C_row + k0 + L * v, acc[v]);
      }
      return;
    }
#endif
    switch (nv) {
      case 1: scorch_spmm_row_regtile_partial<T, 1>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 2: scorch_spmm_row_regtile_partial<T, 2>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 3: scorch_spmm_row_regtile_partial<T, 3>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 4: scorch_spmm_row_regtile_partial<T, 4>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 5: scorch_spmm_row_regtile_partial<T, 5>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 6: scorch_spmm_row_regtile_partial<T, 6>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 7: scorch_spmm_row_regtile_partial<T, 7>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 8: scorch_spmm_row_regtile_partial<T, 8>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
    }
  }
}
#endif

#if defined(__ARM_NEON)
// ---------------------------------------------------------------------------
// NEON register-resident row kernels for the drop-in SpMM, the ARM counterpart
// of the AVX2 regblock/regtile pair above.
//
// Why these exist: until now the `#if defined(__AVX2__)` block above was the ONLY
// register-resident path in this kernel, so on ARM every row fell through to the
// workspace loop -- memset a tile, accumulate into it with a load-modify-store per
// nonzero, memcpy it out. The fused Linear kernel further down has had a NEON
// register kernel since f130f54 and measured the workspace round-trip at 10-14% of
// its forward, so the drop-in SpMM was paying a cost its own file already knew how
// to avoid, on the one architecture that could not use the AVX2 fix.
//
// NEON has no masked load or store, which is what the AVX2 kernels use for a k that
// is not a whole number of vectors. The strategy here instead: a row is (a) whole
// vectors accumulated in registers, plus (b) a remainder of fewer than `lanes`
// columns accumulated in SCALAR registers -- both updated in the SAME pass over the
// row's nonzeros. No masks, no overread past the end of B, and one walk of A1_crd
// and A_val per row rather than one per column.
//
// That last point is the trap the fused kernel's version still falls into: its tail
// is a per-column scalar loop OUTSIDE the nonzero loop, so a row with k below its
// 32-wide strip re-walks the row once per column -- 8 walks at k=8, which is exactly
// the GCN shape. Migrating the fused kernel onto these is a separate change with its
// own grid; it is not done here.
template <typename T> struct scorch_neon;

template <> struct scorch_neon<float> {
  using vec = float32x4_t;
  static constexpr int lanes = 4;
  static constexpr int strip_vecs = 8;      // 32 floats, 8 of 32 NEON registers
  static inline vec zero() { return vdupq_n_f32(0.f); }
  static inline vec splat(float x) { return vdupq_n_f32(x); }
  static inline vec load(const float* p) { return vld1q_f32(p); }
  static inline void store(float* p, vec v) { vst1q_f32(p, v); }
  static inline vec fma(vec acc, vec a, vec b) { return vfmaq_f32(acc, a, b); }
  static inline vec add(vec a, vec b) { return vaddq_f32(a, b); }
};

template <> struct scorch_neon<double> {
  using vec = float64x2_t;
  static constexpr int lanes = 2;
  // 16 vectors, not 8: a strip is sized by how many ELEMENTS of the output row it
  // covers, not by how many registers it uses. At 8 vectors a double strip covered
  // 16 columns against float's 32, so float64 needed twice the strips for the same
  // k -- and every extra strip re-walks the row's nonzeros, which is pure loss on a
  // low-degree row. Measured: the win degraded monotonically with strips per row
  // (1.446 / 1.323 / 1.276 / 1.193 at 1 / 2 / 4 / 8 strips) and degree-3.0 aeshape
  // at k=128, the only 8-strip low-degree cell, fell to 0.969-0.990. 16 of 32 NEON
  // registers is still half the file.
  static constexpr int strip_vecs = 16;     // 32 doubles
  static inline vec zero() { return vdupq_n_f64(0.0); }
  static inline vec splat(double x) { return vdupq_n_f64(x); }
  static inline vec load(const double* p) { return vld1q_f64(p); }
  static inline void store(double* p, vec v) { vst1q_f64(p, v); }
  static inline vec fma(vec acc, vec a, vec b) { return vfmaq_f64(acc, a, b); }
  static inline vec add(vec a, vec b) { return vaddq_f64(a, b); }
};

// One strip of the output row: NV vector accumulators plus TAIL scalar ones, all
// live across a single pass over [pA_begin, pA_end). NV and TAIL are template
// parameters so the loops fully unroll and the tail costs nothing when TAIL == 0.
template <typename T, int NV, int TAIL, bool ALLOW_DUAL = true, bool ACCUM = false>
static inline void scorch_spmm_row_neon_strip(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end, int k0) {
  using V = scorch_neon<T>;
  constexpr int L = V::lanes;
  // Two accumulator sets when the register budget allows, so two nonzeros are in
  // flight at once. With NV accumulators and one nonzero per iteration each
  // accumulator is a serial FMA chain as long as the row: at NV=2 (float32, k=8) and
  // degree 5 that is ~5 dependent FMAs with only two chains to interleave. The AVX2
  // regblock has carried the same 2-nonzero unroll from the start for that reason;
  // the wide-k regtile does not need it because 8 accumulators already hide the
  // latency. Threshold on 2*NV rather than NV so the doubled set never exceeds half
  // the 32-register file.
  //
  // ALLOW_DUAL exists so the two versions can be compared inside one binary, next to
  // the workspace arm, rather than across two builds.
  //
  // What the unroll is worth, M5, 447 cell-readings over 12 interleaved passes:
  // geomean 1.031 on float32 and 1.014 on float64, largest at narrow k (float32
  // k=4 1.047, k=8 1.045, decaying to 1.015 at k=32) and at longer rows (degree
  // >= 8: 1.044 float32). 13% of float32 readings and 29% of float64 ones land
  // below 1.0, none worse than 0.93, and the four cells that lose in a majority of
  // their passes lose 1.1-2.2% -- inside the p90 of the same-code control (1.035).
  //
  // The threshold makes this provably inert for float64 at k >= 32, where a strip is
  // 16 vectors and a doubled set would want all 32 registers. Those 85 readings are
  // therefore identical code on both arms, and they measure 1.0066 / 1.0002 / 1.0052
  // at k = 32 / 64 / 128 -- the floor against which the float32 numbers above are
  // real. Note that a wide row is cut into strips of strip_vecs, so NV per strip is 8
  // for float32 at every k, not k/lanes; the unroll is never off for float32.
  //
  // Worth recording why this needed measuring twice: the unroll was first added
  // because gcn__cora@8 read below 1.0 in three runs of four, and that reading does
  // not survive a better instrument. That cell is a 20-microsecond call whose time
  // does not move with k at all, so its runtime is per-call fixed cost and not this
  // kernel. Timed one call at a time its A/A control ran to 1.4-1.7 and eight
  // readings of it scattered from 0.875 to 1.440 with no consistent sign. The unroll
  // is worth keeping; the reason first given for it was not a reason.
  // Not gated on NV >= 1, though a scalar-only width looks like it should be: at a
  // free dimension of 1 the whole row is one scalar accumulator, one FMA per nonzero,
  // and the loop's own pointer arithmetic and prefetch plainly dominate. Requiring
  // NV >= 1 there was measured and made that case WORSE, 0.968 -> 0.922 against the
  // kernel this replaced. Two scalar chains still pay. Left alone.
  constexpr bool DUAL = ALLOW_DUAL && (2 * NV <= 16);
  typename V::vec acc[NV > 0 ? NV : 1];
  typename V::vec acc2[(DUAL && NV > 0) ? NV : 1];
  // ACCUM starts from what is already in the row, so this kernel serves both the
  // assigning case (the drop-in SpMM, which owns the whole row) and the accumulating
  // one (the tiled kernels, which add each contraction panel's contribution to a row
  // that later panels will add to as well). Only the first accumulator set is seeded;
  // the second starts at zero and is folded in at the store, so the row is added in
  // exactly once either way.
  #pragma unroll
  for (int v = 0; v < NV; v++)
    acc[v] = ACCUM ? V::load(C_row + k0 + L * v) : V::zero();
  if constexpr (DUAL) {
    #pragma unroll
    for (int v = 0; v < NV; v++) acc2[v] = V::zero();
  }
  T tail[TAIL > 0 ? TAIL : 1];
  T tail2[TAIL > 0 ? TAIL : 1];
  #pragma unroll
  for (int t = 0; t < TAIL; t++) {
    tail[t] = ACCUM ? C_row[k0 + L * NV + t] : T(0);
    tail2[t] = T(0);
  }

  int pA = pA_begin;
  if constexpr (DUAL) {
    for (; pA + 1 < pA_end; pA += 2) {
      const T* SCORCH_RESTRICT B0 =
          B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
      const T* SCORCH_RESTRICT B1 =
          B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0;
      if (pA + 2 < pA_end)
        __builtin_prefetch(B_val + (size_t)A1_crd[pA + 2] * (size_t)B1_size + k0, 0, 1);
      const T a0 = A_val[pA], a1 = A_val[pA + 1];
      const typename V::vec va0 = V::splat(a0), va1 = V::splat(a1);
      #pragma unroll
      for (int v = 0; v < NV; v++) {
        acc[v] = V::fma(acc[v], va0, V::load(B0 + L * v));
        acc2[v] = V::fma(acc2[v], va1, V::load(B1 + L * v));
      }
      #pragma unroll
      for (int t = 0; t < TAIL; t++) {
        tail[t] += a0 * B0[L * NV + t];
        tail2[t] += a1 * B1[L * NV + t];
      }
    }
  }
  for (; pA < pA_end; pA++) {
    const T* SCORCH_RESTRICT B =
        B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
    if (pA + 1 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
    const T a = A_val[pA];
    const typename V::vec va = V::splat(a);
    #pragma unroll
    for (int v = 0; v < NV; v++)
      acc[v] = V::fma(acc[v], va, V::load(B + L * v));
    #pragma unroll
    for (int t = 0; t < TAIL; t++) tail[t] += a * B[L * NV + t];
  }

  #pragma unroll
  for (int v = 0; v < NV; v++) {
    typename V::vec r = acc[v];
    if constexpr (DUAL) r = V::add(r, acc2[v]);
    V::store(C_row + k0 + L * v, r);
  }
  #pragma unroll
  for (int t = 0; t < TAIL; t++) C_row[k0 + L * NV + t] = tail[t] + tail2[t];
}

// Dispatch a strip of `width` columns (width <= 8 * lanes) to the instantiation
// that covers it. Straight-line once selected; the switch is on a value that is
// loop-invariant per call.
template <typename T, bool ALLOW_DUAL = true, bool ACCUM = false>
static inline void scorch_spmm_row_neon_dispatch(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end, int k0, int width) {
  using V = scorch_neon<T>;
  constexpr int L = V::lanes;
  const int nv = width / L;
  const int tail = width - nv * L;
#define SCORCH_NEON_CASE(NVV)                                                     \
  case NVV:                                                                       \
    switch (tail) {                                                               \
      case 0: scorch_spmm_row_neon_strip<T, NVV, 0, ALLOW_DUAL, ACCUM>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0); return; \
      case 1: scorch_spmm_row_neon_strip<T, NVV, 1, ALLOW_DUAL, ACCUM>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0); return; \
      case 2: scorch_spmm_row_neon_strip<T, NVV, 2, ALLOW_DUAL, ACCUM>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0); return; \
      default: scorch_spmm_row_neon_strip<T, NVV, 3, ALLOW_DUAL, ACCUM>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0); return; \
    }
  switch (nv) {
    SCORCH_NEON_CASE(0)
    SCORCH_NEON_CASE(1)
    SCORCH_NEON_CASE(2)
    SCORCH_NEON_CASE(3)
    SCORCH_NEON_CASE(4)
    SCORCH_NEON_CASE(5)
    SCORCH_NEON_CASE(6)
    SCORCH_NEON_CASE(7)
    SCORCH_NEON_CASE(8)
    SCORCH_NEON_CASE(9)
    SCORCH_NEON_CASE(10)
    SCORCH_NEON_CASE(11)
    SCORCH_NEON_CASE(12)
    SCORCH_NEON_CASE(13)
    SCORCH_NEON_CASE(14)
    SCORCH_NEON_CASE(15)
    default: break;
  }
#undef SCORCH_NEON_CASE
  scorch_spmm_row_neon_strip<T, scorch_neon<T>::strip_vecs, 0, ALLOW_DUAL, ACCUM>(
      A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0);
}

// A whole output row. Wide k is cut into 8-vector strips (32 floats / 16 doubles)
// so the accumulators stay in registers; the last, narrower strip carries whatever
// is left, vectors and scalars together. A row narrower than one strip is a single
// dispatch -- one pass over the nonzeros, which is the case the fused kernel's
// per-column scalar tail gets wrong.
template <typename T, bool ALLOW_DUAL = true, bool ACCUM = false>
static inline void scorch_spmm_row_neon(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  constexpr int NV_FULL = scorch_neon<T>::strip_vecs;
  constexpr int STRIP = NV_FULL * scorch_neon<T>::lanes;   // 32 elements, both types
  int k0 = 0;
  // A full strip is NV_FULL vectors and no remainder by construction, so it needs
  // no dispatch -- one instantiation, straight through.
  for (; k0 + STRIP <= B1_size; k0 += STRIP)
    scorch_spmm_row_neon_strip<T, NV_FULL, 0, ALLOW_DUAL, ACCUM>(
        A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0);
  if (k0 < B1_size)
    scorch_spmm_row_neon_dispatch<T, ALLOW_DUAL, ACCUM>(
        A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, B1_size - k0);
}

// The entry spmm_csr_v2_core calls per row. neon_single / neon_nv / neon_tail are
// hoisted above the row loop by the caller because they are functions of B1_size
// alone, so each row runs one straight-line instantiation rather than re-deriving
// the split -- the same reason the AVX2 arm switches on a loop-invariant nvec.
// Templated on ALLOW_DUAL so a release build instantiates exactly one version.
template <typename T, bool ALLOW_DUAL>
static inline void scorch_spmm_row_neon_hoisted(
    const int* SCORCH_RESTRICT A1_crd, const T* SCORCH_RESTRICT A_val,
    const T* SCORCH_RESTRICT B_val, int B1_size,
    T* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    bool neon_single, int neon_nv, int neon_tail) {
  if (neon_single) {
    #define SCORCH_NEON_TAIL(NV)                                          \
      switch (neon_tail) {                                                \
        case 0: scorch_spmm_row_neon_strip<T, NV, 0, ALLOW_DUAL>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, 0); break; \
        case 1: scorch_spmm_row_neon_strip<T, NV, 1, ALLOW_DUAL>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, 0); break; \
        case 2: scorch_spmm_row_neon_strip<T, NV, 2, ALLOW_DUAL>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, 0); break; \
        default: scorch_spmm_row_neon_strip<T, NV, 3, ALLOW_DUAL>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, 0); break; \
      }
    switch (neon_nv) {
      case 0: SCORCH_NEON_TAIL(0); break;
      case 1: SCORCH_NEON_TAIL(1); break;
      case 2: SCORCH_NEON_TAIL(2); break;
      case 3: SCORCH_NEON_TAIL(3); break;
      case 4: SCORCH_NEON_TAIL(4); break;
      case 5: SCORCH_NEON_TAIL(5); break;
      case 6: SCORCH_NEON_TAIL(6); break;
      case 7: SCORCH_NEON_TAIL(7); break;
      case 8: SCORCH_NEON_TAIL(8); break;
      case 9: SCORCH_NEON_TAIL(9); break;
      case 10: SCORCH_NEON_TAIL(10); break;
      case 11: SCORCH_NEON_TAIL(11); break;
      case 12: SCORCH_NEON_TAIL(12); break;
      case 13: SCORCH_NEON_TAIL(13); break;
      case 14: SCORCH_NEON_TAIL(14); break;
      case 15: SCORCH_NEON_TAIL(15); break;
      // strip_vecs, not a literal: nv reaches 16 for float64 at k=32 (16
      // vectors of 2 lanes) and 8 for float32 at k=32. A literal here wrote
      // NV-1 vectors and silently left the last lanes of the row unwritten.
      default: SCORCH_NEON_TAIL(scorch_neon<T>::strip_vecs); break;
    }
    #undef SCORCH_NEON_TAIL
  } else {
    scorch_spmm_row_neon<T, ALLOW_DUAL>(A1_crd, A_val, B_val, B1_size, C_row,
                                        pA_begin, pA_end);
  }
}
#endif  // __ARM_NEON

// Rows per work-stealing chunk for SpMM.
//
// The generic scorch_chunk aims for a fixed number of dynamic chunks per worker
// and clamps the width at SCORCH_CHUNK_MAX rows. It has no term for what the
// stealing itself costs, and that is the dominant cost on a large short-row
// matrix: every chunk is one atomic fetch_add on a line every worker is fighting
// over, and those do not overlap. scircuit (171k rows, 5.6 nonzeros a row) hit the
// 64-row clamp and so took 2672 steals of ~360 nonzeros each -- the atomics cost
// more than the arithmetic, and the kernel ran at 0.68x of Intel MKL at k=8.
//
// A chunk trades two costs against each other:
//   steal stream   (rows / chunk) * c        c = a contended atomic, wall clock
//   tail           chunk * deg * k * w       w = one nonzero of work on one core
// Minimising the sum gives
//   chunk* = rows * sqrt(K * KREF / (nnz * k)),   K = c/w
// K is a property of the machine -- the ratio of a contended atomic to a nonzero
// of work -- not of the matrix; everything matrix-specific is already in rows,
// nnz and k. KREF anchors it to the k at which K was measured.
//
// Two guards, both load-balance:
//   * never below what scorch_chunk already picks. Where a 64-row chunk already
//     carries plenty of work (reddit: 64 rows are 31k nonzeros) the generic value
//     is right and this returns it unchanged.
//   * never so wide that a worker would get fewer than SCORCH_SPMM_CHUNKS_MIN
//     chunks, because past that the tail of one chunk costs more than the steals
//     it saved. On a matrix small enough that 64 rows already breaks that bound,
//     64 stands: there the whole product is a few microseconds and the steal
//     stream is the only cost that matters.
//
// Measured on the redwood i9-14900K over 25 SuiteSparse/GCN matrices x k in
// 8..512, kernel-only against MKL in the same process: 90 of 90 cells at or above
// MKL (worst 1.007, geometric mean 1.333) where the generic chunk lost on 7 of 90
// (worst 0.680, geometric mean 1.240). The residual against a per-cell oracle
// chunk is a median 1.6% and at most 12%, and it tracks the SHAPE of the degree
// distribution, which this formula cannot see -- see the note in tiling.py.
#ifndef SCORCH_SPMM_CHUNK_K            // c/w on this host; see the calibration note
#define SCORCH_SPMM_CHUNK_K 16
#endif
#ifndef SCORCH_SPMM_CHUNK_KREF         // the k at which SCORCH_SPMM_CHUNK_K was read
#define SCORCH_SPMM_CHUNK_KREF 8
#endif
#ifndef SCORCH_SPMM_CHUNKS_MIN         // chunks per worker the tail term tolerates
#define SCORCH_SPMM_CHUNKS_MIN 16
#endif
// Depart from the generic width only when the model asks for a width at least this
// many times wider. Below that the model is not saying anything its own error bars
// support: chunk* moves as sqrt(K), and K measured across cells on two hosts spans
// roughly 11 to 3300 against the 16 written down here, so the recommended width
// carries a factor-of-several uncertainty. A recommendation 6% wider than generic
// (pubmed at k=32 asks for 68 against 64) is inside that, and acting on it is acting
// on noise -- measured as the only cell on the M5 grid to fall below the null band
// that no-op cells establish, at 0.916. Above the threshold the rule is a large win
// on both hosts; below it, this makes the rule a provable no-op rather than a coin
// flip, which is what the performance convention asks for when a lever only helps a
// sub-regime.
//
// The grid rules out 1 (one cell below the null) and cannot separate 1.5 from 3: the
// whole-grid geomean over that range is 1.138 to 1.114, inside the +-3.6% null band.
// 2 is chosen as the middle of the range the data admits rather than the value that
// maximizes this grid.
#ifndef SCORCH_SPMM_CHUNK_MINRATIO
#define SCORCH_SPMM_CHUNK_MINRATIO 2
#endif
// Where the MINRATIO guard reads its width. The guard asks a question about the
// MODEL -- "is this recommendation far enough from generic to be outside the
// model's own error bars?" -- so it has to read the width the model asked for,
// before the two load-balance ceilings below trim it. Reading it afterwards
// conflates two different things: a ceiling saying "you may not have that much"
// gets scored as the model saying "I am not sure enough to ask", and the
// recommendation is thrown away entirely rather than granted up to the ceiling.
// That is not a rare corner. The ceiling binds on exactly the large matrices where
// the model is most confident, because it falls as 1/nthreads: at 32 threads and
// CHUNKS_MIN 16 a 50000-row matrix may have 97 rows a chunk, so a model asking 316
// (a 4.9x departure, far outside its error bars) lands at 97, which is below
// 2*generic, and reverts to 64. Measured over the corpus this fires on 96 of 320
// narrow-k band cells, where forcing 64 is a no-op (0.96) and forcing 256 is 1.49x
// -- the width was being discarded, not chosen.
#ifndef SCORCH_SPMM_CHUNK_GUARD_PRECAP
#define SCORCH_SPMM_CHUNK_GUARD_PRECAP 1
#endif
inline int scorch_spmm_chunk(long rows, long nnz, long k, int nthreads) {
  const long generic = (long)scorch_chunk(rows, nnz * k, SCORCH_GRAIN_SPMM);
  if (rows <= 0 || nnz <= 0 || k <= 0 || nthreads <= 0) return (int)generic;
  long chunks_min = SCORCH_SPMM_CHUNKS_MIN;
  long minratio = SCORCH_SPMM_CHUNK_MINRATIO;
  int guard_precap = SCORCH_SPMM_CHUNK_GUARD_PRECAP;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_CHUNK");
    if (e && *e) { long f = std::atol(e); if (f > 0) return (int)f; } }
  { const char* e = std::getenv("SCORCH_SPMM_CHUNKS_MIN");
    if (e && *e) { long v = std::atol(e); if (v > 0) chunks_min = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CHUNK_MINRATIO");
    if (e && *e) { long v = std::atol(e); if (v > 0) minratio = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CHUNK_GUARD_PRECAP");
    if (e && *e) guard_precap = (int)std::atol(e); }
#endif
  const double ratio = (double)SCORCH_SPMM_CHUNK_K * (double)SCORCH_SPMM_CHUNK_KREF
                       / ((double)nnz * (double)k);
  const long model = (long)((double)rows * std::sqrt(ratio));
  // The confidence guard, on the model's own width. Below the threshold the model
  // is not saying anything it can support, so the generic width stands.
  if (guard_precap && model < minratio * generic) return (int)generic;
  long c = model;
  if (c < generic) c = generic;
  long cap = rows / ((long)nthreads * chunks_min);
  if (cap < generic) cap = generic;
  if (c > cap) c = cap;
  const long per_worker = rows / (long)nthreads;      // every worker gets one
  if (per_worker >= 1 && c > per_worker) c = per_worker;
  if (c < 1) c = 1;
  if (guard_precap) {
    // A ceiling trims the recommendation; it never pushes it below the width we
    // would have used anyway.
    if (c < generic) c = generic;
  } else {
    // The pre-fix order, kept as the control arm: the guard reads the trimmed width.
    if (c < minratio * generic) c = generic;
  }
  return (int)c;
}

// spmm_csr_float_v2 — workspace-based direct accumulation with 2-nnz ILP
//
// Key ideas:
//   1. Per-thread workspace (fits in L1) avoids cold read-modify-write of
//      the output matrix C — accumulate into hot workspace, write-back once.
//   2. Process 2 nonzeros per iteration for instruction-level parallelism
//      (two independent FMA chains keep the execution units busy).
//   3. K-tiling so B accesses stay cache-friendly for large k.
//   4. Prefetch 2 B-rows ahead to hide DRAM latency.
//   5. Simple inner loop — let -O3 -march=native -ffast-math auto-vectorize.
// ---------------------------------------------------------------------------

// One row cursor per worker, each on its own cache line so the owner's fetch_add
// does not invalidate a neighbour's. Used by the home-range row partition in
// spmm_csr_v2_core.
//
// 128, not 64, because at 64 the sentence above is false on one of the two hosts:
// Apple silicon's cache line is 128 bytes, so two neighbouring workers would share
// a line and each fetch_add would invalidate the neighbour's. 128 is also the right
// number on x86, where the L2 spatial prefetcher works on 128-byte sector pairs and
// would pull the neighbour's line in regardless. This is not offered as the
// explanation for anything measured -- the M5's narrow-k behaviour under this
// partition is under investigation and the thread count there argues the other way
// (narrow k resolves to FEWER workers, hence fewer shared lines, and narrow k is
// where the harm is). It is here because the struct should mean what it says.
struct alignas(128) scorch_spmm_cursor {
  // Deliberately NOT brace-initialised: an array of these is declared on the
  // stack for every call that uses the partition, and a default member
  // initialiser would make merely declaring it write every line. The setup
  // stores into exactly the entries it will use.
  //
  // v is the owner's cursor for modes 1 and 2, where a thief also takes from the
  // front and one fetch_add serves both ends. Mode 3 needs the two ends
  // independent, so it uses ht instead: head in the low 32 bits, exclusive tail in
  // the high 32, moved by compare-exchange. Packing them in one word is what makes
  // a claim atomic -- with two separate counters an owner and a thief can each read
  // a stale view of the other and claim the same rows.
  std::atomic<int> v;
  std::atomic<uint64_t> ht;
};

static inline uint64_t scorch_ht_pack(int head, int tail) {
  return (uint64_t)(uint32_t)head | ((uint64_t)(uint32_t)tail << 32);
}
static inline int scorch_ht_head(uint64_t x) {
  return (int)(uint32_t)(x & 0xffffffffu);
}
static inline int scorch_ht_tail(uint64_t x) { return (int)(uint32_t)(x >> 32); }

// Claim up to `chunk` rows from one range. FRONT is the owner's end, BACK is a
// thief's. False means the range is empty, and empty for good: head only rises and
// tail only falls, so no caller has to ask a second time.
static inline bool scorch_ht_claim(std::atomic<uint64_t>& ht, int chunk, bool front,
                                   int* out_begin, int* out_end) {
  uint64_t cur = ht.load(std::memory_order_relaxed);
  for (;;) {
    const int h = scorch_ht_head(cur), t = scorch_ht_tail(cur);
    if (h >= t) return false;
    int b, e;
    uint64_t nxt;
    if (front) {
      b = h; e = (h + chunk < t) ? h + chunk : t;
      nxt = scorch_ht_pack(e, t);
    } else {
      e = t; b = (t - chunk > h) ? t - chunk : h;
      nxt = scorch_ht_pack(h, b);
    }
    if (ht.compare_exchange_weak(cur, nxt, std::memory_order_relaxed,
                                 std::memory_order_relaxed)) {
      *out_begin = b; *out_end = e;
      return true;
    }
    // A failed exchange refreshed cur; retry against the view it handed back.
  }
}

// The kernel proper takes plain pointers and sizes. Two callers reach it: the
// legacy entry below, which unpacks the nested tensor vectors the pybind ABI
// hands over, and SpmmCsrPlan (plan.h), which already holds the unpacked and
// validated structure and so pays none of that per call. Splitting them is what
// lets a warm dispatch be a single Python->C++ hop; the body is unchanged.
template <typename scalar_t>
torch::Tensor spmm_csr_v2_core(
                int C0_size, int C1_size, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const scalar_t* SCORCH_RESTRICT A_val,
                int B1_size, const scalar_t* SCORCH_RESTRICT B_val,
                int tile_size, int nthreads_override, bool atparallel) {
  const size_t C_capacity = (size_t)C0_size * (size_t)C1_size;

  // Output allocation + zeroing. Every code path below (regblock / regtile /
  // workspace memcpy) ASSIGNS all C1_size entries of each NON-empty output row,
  // so only structurally-empty rows -- and any tail rows C0_size>A0_size, which
  // no worker owns -- have to be zeroed at all. The legacy path malloc'd a raw
  // buffer and memset the WHOLE thing single-threaded before the parallel
  // region: a serial O(C0*C1) zero-fault per call (14MB for a 14K-row k=256
  // product) that MKL never pays, since its dense output is written in full by
  // the kernel from a pooled buffer. We allocate via torch::empty (the same CPU
  // allocator MKL's output uses, so scorch and MKL are apples-to-apples on
  // allocation) and write nothing outside the empty rows. Where those rows are
  // zeroed is a performance question, settled below; correctness does not depend
  // on it, since a non-empty row is fully overwritten by the kernel and every
  // empty row is zeroed exactly once.
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, scorch_torch_dtype<scalar_t>());
  scalar_t* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<scalar_t>();
  bool zero_empty_only = true;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook retained for the zeroing policy. Output storage is always owned by
  // Torch; bit1 selects empty-row-only zeroing versus a full memset.
  { const char* e = std::getenv("SCORCH_SPMM_ALLOC");
    if (e && *e) { long v = std::atol(e); zero_empty_only = v & 2; } }
#endif
  // Which mechanism zeroes the structurally-empty output rows.
  //
  //   2 (default) the worker that steals the row zeroes it, merging consecutive
  //               empty rows within the stolen chunk into one memset;
  //   3           the same as 2, one memset per empty row;
  //   6           the workers steal WIDE slices of the row range and zero the runs
  //               of empty rows in them, alternating one such slice with one chunk
  //               of arithmetic;
  //   5           the same, but every zero slice is drained before any worker
  //               starts on arithmetic;
  //   4           the same, but a static slice per worker instead of stealing;
  //   1           one pre-loop parallel span over the whole output, gated on
  //               SCORCH_SPMM_ZERO_SPAN_ELEMS and on three quarters of the rows
  //               being empty;
  //   0           the pre-2026-08 serial per-empty-row memset.
  //
  // 0..3 are kept so all five can be priced against each other in ONE binary;
  // they are compiled out of the shipped .so, where zero_mode is a constant and
  // every branch on it folds.
  //
  // Why the default zeroes a row where the arithmetic loop already visits it, and
  // not in a pass of its own. Every up-front variant -- 4, 5 and 6 -- walks A1_pos
  // a SECOND time looking for runs of empty rows, and the arithmetic loop was going
  // to visit every row anyway. Over redwood's 362-matrix panel that second pass
  // costs, and it costs exactly the way an O(rows) pass should: the loss grows with
  // the row count and shrinks as the degree gives it more arithmetic to hide
  // behind (float64, geomean against the default):
  //
  //             deg<=2   2-8   8-32   32-128   >128
  //   <1K rows   1.025  1.009  1.010   0.996   1.015
  //   1-10K      0.920  0.976  0.995   1.001   1.004
  //   10-100K    0.867  0.951  0.977      --      --
  //   0.1-1M     0.914     --     --      --      --
  //
  // The wide variants exist because on the M5 a 0.8 GB nearly-empty float64 output
  // ran 2.3x faster with one contiguous run per worker than with the arithmetic
  // chunk's 781-row width. That turned out not to be about the zero: on that host a
  // FRESH torch allocation over ~200 MB zeroes at 30-50 GB/s where a reused buffer
  // holds 235, so what the wide runs were driving better was the page-fault path of
  // an allocator that is not reusing its blocks. Redwood, whose allocator does
  // reuse, has no such gap on the same matrices -- higgs-twitter_reply at 892 MB
  // and 1784 MB zeroes at 33 GB/s of its ~56 GB/s achievable, and both wide
  // variants read 0.856 (assigned) and 0.986 (stolen) against the default there.
  // So the mechanism is chosen on the corpus with a warm allocator, and the fault
  // path is recorded rather than tuned against.
#ifdef SCORCH_TUNE_HOOKS
  int zero_mode = 2;
  { const char* e = std::getenv("SCORCH_SPMM_ZERO_MODE");
    if (e && *e) { long v = std::atol(e); if (v >= 0 && v <= 6) zero_mode = (int)v; } }
  { const char* e = std::getenv("SCORCH_SPMM_ZERO_LEGACY");   // the older name for 0
    if (e && *e && std::atol(e) != 0) zero_mode = 0; }
#else
  constexpr int zero_mode = 2;
#endif
  const size_t out_row_bytes = sizeof(scalar_t) * (size_t)C1_size;
  // Empty output rows, counted before anything is zeroed. Feeds the thread
  // policy below, and mode 2's row loop.
  int64_t empty_rows = 0;
  if (zero_empty_only) {
    // The comment above assumes empty rows are rare. That holds for FEM and graph
    // adjacencies, but 27.9% of the SuiteSparse+DLMC corpus has at least one, and
    // where most rows are empty the serial loop degenerates into a strided zero of
    // essentially the whole output, issued as `rows` separate small memsets.
    // Measured on redwood, a 200000x256 output with 40 nonzeros wrote at 4.7-5.6
    // GB/s where a span write on the same host runs at 14.4-20.3, and vs MKL those
    // shapes read 0.81-0.92.
    //
    // The scan is branchless and reads A1_pos, which the row loop reads anyway.
    // When it comes back zero there is nothing to zero at all -- the kernel writes
    // every non-empty row in full -- so the common case makes one pass over
    // A1_pos and no longer makes the branchy pass the old loop made.
    // ... but it is also SERIAL, and it sits ahead of the parallel region, so on a
    // narrow-k product it is Amdahl's serial fraction rather than a rounding error:
    // it costs O(rows) while the parallel work is O(nnz*k), so its share grows as k
    // shrinks.
    //
    // In the shipped mode the count has exactly one consumer -- the zero_work term
    // in the thread policy below -- so it is only worth paying for when it can
    // change the thread count, and that is decidable in O(1) without counting
    // anything. Ask the policy for the count at the two extremes: no empty row, and
    // every row empty. The count is monotone non-decreasing in work, so if the
    // extremes agree then every value between them agrees and the scan cannot affect
    // any decision. (total_nnz and k_eff are declared further down, next to the
    // policy call itself; recomputing them here is two loads.)
    //
    // Measured worth: 1.0 to 2.2% at k<=2, rising with rows-per-nonzero exactly as
    // an O(rows) term against an O(nnz*k) loop should -- 1.0025, 1.0124, 1.0218
    // across the bands under 0.05, 0.05 to 0.25, and over 0.25.
    //
    // Skipping the scan UNCONDITIONALLY is a different and much worse thing, because
    // it also drops the zero_work credit: measured that way the wide-k cells fall to
    // 0.934 / 0.842 / 0.849 of the scanning arm at k = 64 / 256 / 512 on float32 and
    // 0.851 / 0.793 / 0.806 on float64, since an empty-row-heavy output is
    // bandwidth-bound and wants the workers that term buys it. This skips the scan
    // only where the credit provably changes nothing.
    bool do_scan = true;
    if (zero_mode == 2 || zero_mode == 3) {
      const long nnz_all = A0_size > 0 ? (long)A1_pos[A0_size] : 0L;
      const long keff = B1_size < 16 ? 16L : (long)B1_size;
      const long w_lo = nnz_all * keff;                            // no empty row
      const long w_hi = w_lo + (long)A0_size * (long)C1_size;      // every row empty
      const long w_true = nnz_all * (long)B1_size;
      // nnz has to be passed here too: it can widen the row-axis ceiling, and a
      // skip decision taken under a different ceiling than the real call's would
      // skip a scan whose result the real call then needs.
      if (scorch_spmm_nthreads(w_lo, A0_size, nthreads_override, w_true, nnz_all) ==
          scorch_spmm_nthreads(w_hi, A0_size, nthreads_override, w_true, nnz_all))
        do_scan = false;
    }
#ifdef SCORCH_TUNE_HOOKS
    // A/B hook: 1 forces the scan off outright -- which also drops the credit, so it
    // prices the whole term and not this rule -- and 2 forces it always on, which is
    // the pre-rule behaviour. Compiled out of the shipped .so.
    { const char* e = std::getenv("SCORCH_SPMM_SKIP_EMPTY_SCAN");
      if (e && *e) { const long v = std::atol(e);
        if (v == 1) do_scan = false; else if (v == 2) do_scan = true; } }
#endif
    if (do_scan)
      for (int i = 0; i < A0_size; i++)
        empty_rows += (A1_pos[i] == A1_pos[i + 1]) ? 1 : 0;
    if (empty_rows != 0 && zero_mode <= 1) {
      const int64_t empty_elems = empty_rows * (int64_t)C1_size;
      if (zero_mode == 1 && empty_elems >= SCORCH_SPMM_ZERO_SPAN_ELEMS &&
          empty_rows * 4 >= (int64_t)A0_size * 3) {
        // Mode 1: one parallel span over the whole output. Measured 2.10x (f32) /
        // 3.15x (f64) over mode 0 on the 205 cells it fires on, but it spawns a
        // team of omp_get_num_procs() immediately before the kernel's own team of
        // scorch_spmm_nthreads(), and 19 of those 205 float32 cells LOST by up to
        // 1.9x -- the ones with real arithmetic after the zero, which is the
        // signature of the second team running in the wake of the first. The
        // default has no second team, which is why it is the default; it also
        // matches this arm on the enormous nearly-empty outputs where the
        // contiguous span is at its best (0.975 and 1.019 on two 0.8 GB float64
        // cells, against 0.430 and 0.448 for mode 2).
        scorch_zero_dense(C_values, (int64_t)A0_size * (int64_t)C1_size);
      } else {
        for (int i = 0; i < A0_size; i++)
          if (A1_pos[i] == A1_pos[i + 1])
            memset(C_values + (size_t)i * (size_t)C1_size, 0, out_row_bytes);
      }
    }
    if (C0_size > A0_size) {
      // Rows past A's last one: no worker owns them, so they are zeroed here
      // whatever the mode. A contiguous tail, so this is a span write whatever
      // its size, and scorch_zero_dense falls back to memset below the grain.
      if (zero_mode == 0)
        memset(C_values + (size_t)A0_size * (size_t)C1_size, 0,
               out_row_bytes * (size_t)(C0_size - A0_size));
      else
        scorch_zero_dense(C_values + (size_t)A0_size * (size_t)C1_size,
                          (int64_t)(C0_size - A0_size) * (int64_t)C1_size);
    }
  } else {
    memset(C_values, 0, sizeof(scalar_t) * C_capacity);
  }
  // Where the empty rows get zeroed, as booleans the worker reads. All of them
  // fold to constants in the shipped build, where only zero_in_loop and
  // zero_merge_runs are true and the slice paths compile away entirely.
  //
  // empty_rows appears in the slice conditions, not just inside their loops, so a
  // shape with no empty row does not pay even the slice scan.
  const bool zero_static_slice =
      zero_empty_only && zero_mode == 4 && empty_rows != 0;
  const bool zero_stolen_slice =
      zero_empty_only && (zero_mode == 5 || zero_mode == 6) && empty_rows != 0;
  // Mode 6 takes one zero slice per arithmetic chunk instead of draining them
  // first. Same slices, same counter, same total work; only the order differs.
  const bool zero_alternate = zero_mode == 6;
  const bool zero_in_loop = zero_empty_only && (zero_mode == 2 || zero_mode == 3);
  const bool zero_merge_runs = zero_mode == 2;

  // Round tile to multiple of 16 for SIMD alignment
  const int kTile = (tile_size + 15) & ~15;

  // Work-aware thread cap + adaptive schedule chunk from the shared policy
  // (scorch/csrc/scorch_policy.h): work = nnz*k, grain = SCORCH_GRAIN_SPMM. A small
  // A_sparse @ B_dense product (nnz*k below a few million flops) is swamped by
  // OpenMP fork/join across all cores — a 24-row product spent more time in
  // barriers than computing — so scorch_nthreads throttles it; the cap only binds
  // below ~GRAIN*num_procs, so large products keep every core. `chunk` is the
  // number of rows each worker steals per next_row.fetch_add below.
  const int total_nnz = A1_pos[A0_size];
  // Thread-cap work measure: a B-row gather touches a full 64B cache line (16
  // f32) regardless of how narrow k is, so a tall-skinny product (many rows,
  // k<16) is memory-bound at ~one line per nnz. Crediting only nnz*k here
  // throttled such shapes to near-serial (a 19.7K-row k=3 GCN layer capped to 2
  // threads despite abundant row-parallelism). Floor the k term at the line
  // width so row-parallelism isn't starved; k>=16 behavior is unchanged.
  const long k_eff = B1_size < 16 ? 16L : (long)B1_size;
  // The empty rows are output-writing work the nnz*k term cannot see. A
  // 200000-row k=256 output with 40 nonzeros credits 10240 units against a
  // 150000 grain, so scorch_nthreads gave the whole call ONE thread -- and in
  // mode 2 that one thread would also own 200000 row zeros. Credit the span the
  // workers actually zero, at one unit per element. That over-credits a pure
  // store stream against a gather-and-FMA on purpose: the zero is
  // bandwidth-bound, so more workers help it. The term is exactly zero when no
  // row is empty, which is 72% of the corpus, so no shape without an empty row
  // moves its thread count.
  const long work_nnz = (long)total_nnz * k_eff;
  const long zero_work = (zero_in_loop || zero_static_slice || zero_stolen_slice)
      ? (long)(empty_rows * (int64_t)C1_size) : 0L;
  const long work = work_nnz + zero_work;
  // Composition override: when this drop-in SpMM runs inside a host (torch)
  // pipeline, the surrounding dense ops use the host thread count (e.g. 16); the
  // throttled policy count (e.g. 11 for a narrow-k GCN layer) then forces a
  // libgomp thread-team RESHAPE at every op boundary (~15% of a sub-ms GCN
  // forward). The caller (ops.py drop-in matmul) passes the ambient host count so
  // one warm team spans the pipeline. We adopt it ONLY for products that clear
  // the existing fork/join floor (work >= SCORCH_GRAIN_SPMM): tiny products
  // (arc130-class) that the policy throttles to avoid fork/join blowup keep the
  // policy count (the SuiteSparse panel's small cells stay byte-identical; a
  // forced 16 threads made a 130-row product 1.7x slower). max() keeps big
  // graphs on their (possibly >host) policy count so we never under-thread.
  // nthreads_override<=0 => pure policy (the standalone/panel default). The rule
  // itself lives in scorch_policy.h so a harness can ask for the same number
  // instead of recomputing it; see scorch_spmm_nthreads.
  const int nthreads = scorch_spmm_nthreads(work, A0_size, nthreads_override,
                                            (long)total_nnz * (long)B1_size,
                                            (long)total_nnz);
  const long nnz_total = A0_size > 0 ? (long)A1_pos[A0_size] : 0;
  const int chunk = scorch_spmm_chunk(A0_size, nnz_total, B1_size, nthreads);
  std::atomic<int> next_row{0};

  // ROW PARTITION. The loop below hands rows out from ONE global atomic, so which
  // worker gets which rows is decided by whoever calls fetch_add first and comes
  // out different on every call. Measured on redwood with perf counters, that
  // costs inter-call L2 residency of A. At k=1 float32 this kernel takes 5.2x
  // (nemeth09) to 15.3x (ts-palko) as many L2 misses per nonzero as MKL on the
  // same product: on ts-palko 138404 per call, which at 64B a line is 8.86 MB --
  // the whole 8.6 MB A array re-fetched from L3 on every call, against MKL's 579
  // KB, i.e. MKL keeps 93% of A in the cores' L2 across calls and this kernel
  // keeps none of it. It is not a DRAM effect: LLC misses are 404 against MKL's
  // 507.
  //
  // Only the miss counts are quoted because only they are clean. Both runtimes
  // spin-wait their idle workers, and a spin loop retires instructions and burns
  // cycles while touching no memory, so the cycle and instruction deltas from a
  // whole-process perf stat are not attributable to either kernel; the L2 and LLC
  // miss deltas are.
  //
  // Modes 1 and 2 give each worker a HOME RANGE instead: contiguous rows, split so
  // that the work per worker is balanced -- equal ROW counts would not be, since
  // degree varies -- computed once by binary search over A1_pos. The same worker
  // then touches the same rows of A on every call with the same matrix, which is
  // how every real caller uses an SpMM (GCN layers, an iterative solver, a
  // benchmark loop), so A stays in that core's L2.
  //
  // The balanced measure is A1_pos[i] + i, not A1_pos[i]: a row costs its nonzeros
  // PLUS a fixed amount -- the row-pointer pair, the register accumulator's
  // horizontal reduction, the output store -- and at k=1 that fixed part is worth
  // a few nonzeros, so pure nnz mis-balances a ragged matrix. It also makes the
  // prefix STRICTLY increasing, which is what keeps a power-law matrix from
  // collapsing: with pure nnz, a row holding 90% of the nonzeros sends every later
  // split to the same boundary and one worker inherits all the remaining rows.
  // Nothing is tuned here -- one unit per row and one per nonzero.
  //
  //   0  one global counter, any worker takes any chunk
  //   1  home ranges only, no stealing
  //   2  home ranges, then steal from the FRONT of whoever is still working
  //   3  same, but steal from the BACK, so an owner keeps a stable prefix
  //
  // Mode 3 exists because the counters say 1 and 2 recover different amounts. On
  // ts-palko at k=1, L2 misses per call run 133363 (mode 0) -> 5324 (mode 1, below
  // MKL's own 6637) -> 37039 (mode 2): stealing from the front takes exactly the
  // rows the owner would have reached next, so that work migrates between cores
  // call to call and drags a seventh of the misses back. Taking from the back
  // leaves the owner's prefix where it was.
  //
  // Mode 1 isolates the locality effect; mode 2 is the shippable form, because a
  // home range is balanced in nonzeros and NOT in time -- redwood is 8 P-cores
  // plus 16 E-cores, so equal nnz is unequal duration and someone has to absorb
  // the difference. Both are correct for a team SMALLER than nthreads (OpenMP may
  // hand one back): worker w owns splits w, w+team_size, w+2*team_size, ..., which
  // covers every split whatever the team size and reduces to exactly one range in
  // the normal case.
  int partition_mode = SCORCH_SPMM_PARTITION_DEFAULT;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_PARTITION");
    if (e && *e) { long v = std::atol(e);
      if (v >= 0 && v <= 3) partition_mode = (int)v; } }
#endif
  // OUTPUT-SIZE GATE. The partition buys A's inter-call L2 residency and pays for it
  // in the output store stream: with one global counter the workers all drain from a
  // moving frontier, so at any instant their writes are close together in physical
  // address space; with home ranges they write to as many regions as there are
  // workers, tens of megabytes apart, and the memory controller sees that many open
  // DRAM rows instead of a near-sequential stream.
  //
  // Measured over 2376 cells of the main and large-A corpora, back-stealing against
  // the shipped counter, by output bytes -- and the thread count is identical on
  // every one of the harmed cells, so this is not the policy:
  //
  //   output       float32            float64
  //   < 1 MB       1.239             1.220
  //   1-4 MB       1.433             1.305
  //   4-16 MB      1.258             1.217
  //   16-64 MB     1.109             1.030
  //   64-256 MB    1.029             1.022
  //   >= 256 MB    0.988             0.944   (26.9% of float64 cells below 0.95)
  //
  // Monotone decay, negative at the top. The A-bytes-per-output-byte ratio shows no
  // trend at all across the same cells (1.13 to 1.33 in every band), so the scale
  // that matters is absolute output size, not the balance between the two streams.
  //
  // Expressed as a multiple of the last-level cache rather than as a byte count: the
  // decay begins where the output stops being cache-resident, and a fixed byte
  // threshold would mean something different on every machine. Four times the LLC is
  // 144 MB on a 36 MB L3, which is where the measured sign change is.
  if (partition_mode != 0) {          // nothing to gate when the partition is off
    long partition_maxout = SCORCH_SPMM_PARTITION_MAXOUT_LLC * scorch_llc_bytes();
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_PARTITION_MAXOUT_MB");
      if (e && *e) { long v = std::atol(e);
        partition_maxout = v > 0 ? v * 1024L * 1024L : 0L; } }   // 0 = no gate
#endif
    if (partition_maxout > 0 &&
        (long)A0_size * (long)C1_size * (long)sizeof(scalar_t) >= partition_maxout)
      partition_mode = 0;
  }
  int nsplit = nthreads > 0 ? nthreads : 1;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: force the number of ranges. More ranges than workers is safe -- the
  // stride loop hands a worker its own extras and the steal loop reaches every range
  // regardless -- so this is the only way to run the heap allocation path, and the
  // very fine and very coarse splits, on a host whose team never gets that wide.
  { const char* e = std::getenv("SCORCH_SPMM_NSPLIT_FORCE");
    if (e && *e) { long v = std::atol(e);
      if (v > 0 && v <= 4096) nsplit = (int)v; } }
#endif
  // On the stack, because the smallest cells in the corpus are ten microseconds long
  // and two heap allocations per call would be a percent of that.
  // 64 cursors of 128 bytes is 8 KB. At 128 the frame was 17440 bytes; halving it
  // costs nothing and keeps the untouched stack reservation off pages the caller
  // might otherwise be using. (It does not remove the entry stack probe -- clang
  // emits that above 4 KB, and 24 cursors is too few for a 32-worker host.) The
  // heap fallback keeps the bound a performance choice and not a correctness one;
  // SCORCH_SPMM_NSPLIT_FORCE is how it gets correctness coverage, since no host
  // here resolves to enough workers to reach it.
  constexpr int kSplitOnStack = 64;
  int split_stack[kSplitOnStack + 1];
  scorch_spmm_cursor cursor_stack[kSplitOnStack];
  std::unique_ptr<int[]> split_heap;
  std::unique_ptr<scorch_spmm_cursor[]> cursor_heap;
  int* row_split = nullptr;
  scorch_spmm_cursor* cursors = nullptr;
  if (partition_mode != 0) {
    if (nsplit <= kSplitOnStack) {
      row_split = split_stack;
      cursors = cursor_stack;
    } else {
      split_heap.reset(new int[nsplit + 1]);
      cursor_heap.reset(new scorch_spmm_cursor[nsplit]);
      row_split = split_heap.get();
      cursors = cursor_heap.get();
    }
    row_split[0] = 0;
    row_split[nsplit] = A0_size;
    for (int w = 1; w < nsplit; ++w) {
      // First row whose prefix reaches w/nsplit of the total. The prefix is
      // strictly increasing, so this is exact and the boundaries are distinct
      // whenever there are at least nsplit rows.
      const long target = (nnz_total + (long)A0_size) * (long)w / (long)nsplit;
      int lo = row_split[w - 1], hi = A0_size;
      while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if ((long)A1_pos[mid] + (long)mid < target) lo = mid + 1; else hi = mid;
      }
      row_split[w] = lo;
    }
    for (int w = 0; w < nsplit; ++w) {
      if (partition_mode == 3)
        cursors[w].ht.store(scorch_ht_pack(row_split[w], row_split[w + 1]),
                            std::memory_order_relaxed);
      else
        cursors[w].v.store(row_split[w], std::memory_order_relaxed);
    }
  }
  // Width of a stolen ZERO slice, in rows. Several slices per worker so a slow
  // core cannot become the critical path, and never narrower than the arithmetic
  // chunk. SCORCH_SPMM_ZERO_SLICES sets the slices-per-worker target; 4 is where
  // the grid put it.
  int zchunk = chunk;
  if (zero_stolen_slice) {
    long per_worker = 4;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_ZERO_SLICES");
      if (e && *e) { long v = std::atol(e); if (v > 0) per_worker = v; } }
#endif
    long want = (long)A0_size / (per_worker * (nthreads > 0 ? nthreads : 1));
    if (want < (long)chunk) want = (long)chunk;
    if (want > (long)A0_size) want = (long)A0_size;
    if (want < 1) want = 1;
    zchunk = (int)want;
  }
  std::atomic<int> next_zero_row{0};

  // Narrow-k register-blocked path (K<=16): hold the whole output row in YMM
  // accumulators across the row's nonzeros, masked AVX2 load/store, 2-nnz ILP.
  // The workspace path below round-trips through memory (memset + per-nnz ws
  // load/store + memcpy) which dominates when K is tiny and the k-loop is below
  // one SIMD lane — that was the 0.5-0.8x-of-MKL narrow-k gap (GCN k=3/16). For
  // K>16 the workspace path already matches/beats MKL, so it is unchanged.
#if defined(__AVX2__) && defined(__FMA__)
  // Register-block when the whole output row fits in <=4 YMM accumulators. That
  // is k<=32 for float32 and k<=16 for float64: the same REGISTER budget, which is
  // what the bound is really about, rather than the same k.
  using SV = scorch_simd<scalar_t>;
  constexpr int SL = SV::lanes;
  const bool narrow_k = (B1_size >= 1 && B1_size <= 4 * SL);
  const int nvec = (B1_size + SL - 1) / SL;        // 1..4 when narrow_k
  const int mlast = B1_size - SL * (nvec - 1);     // valid lanes in last vec, 1..SL
  bool full_last = (mlast == SL);                  // k % SL == 0 -> no mask needed
  const typename SV::mask mask_last = SV::lane_mask(mlast);
#else
  const bool narrow_k = false;
#endif

#if defined(__ARM_NEON) && !defined(__AVX2__)
  // Loop-invariant shape of a single-strip row, hoisted for the same reason the
  // AVX2 block above hoists nvec/full_last: it is a function of B1_size only.
  constexpr int kNeonLanes = scorch_neon<scalar_t>::lanes;
  // strip_vecs*lanes, and NOT the wider bound a single dispatch could in principle
  // serve. It could take strip_vecs vectors plus lanes-1 scalars -- 35 elements for
  // float32 -- which would make widths 33..35 one pass instead of a strip plus a
  // remainder, and those widths do lose slightly (0.986 at 33). That was tried. It did
  // not help them (0.986 -> 0.981) and it cost their neighbours, because adding
  // NV=strip_vecs-with-a-tail instantiations to the hoisted switch changed register
  // allocation for the cases around them: 24 went 1.688 -> 1.623 and 31 went
  // 1.997 -> 1.942, neither of which the change touches. Reverted; 33..35 losing about
  // 1.5% is left as a measured fact rather than tuned around.
  const bool neon_single =
      (B1_size <= scorch_neon<scalar_t>::strip_vecs * kNeonLanes);
  const int neon_nv = B1_size / kNeonLanes;
  const int neon_tail = B1_size - neon_nv * kNeonLanes;
#endif

  // Independent nonzero streams in the exact-width narrow-k kernel; 0 leaves the
  // masked-row widths on the register-block kernel. Loop-invariant either way, so the
  // row loop's dispatch on it hoists.
  int narrowk_exact = SCORCH_NARROWK_EXACT_UNROLL;
  // Float k=1 is the one width where two kernel families compete: the nonzero-axis
  // gather kernel owns it by default and measures better there.
  bool narrowk_exact_k1 = false;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: force the legacy workspace path instead of whichever register kernel
  // this architecture has -- AVX2's regblock/regtile or NEON's strip kernel. Not
  // guarded on the ISA, because the workspace path is what BOTH register kernels
  // replaced and it is the only way to price either of them in one process against
  // one binary. Compiled out of the shipped .so.
  const char* _wsonly = std::getenv("SCORCH_SPMM_WORKSPACE");
  const bool force_workspace = _wsonly && *_wsonly && std::atol(_wsonly) != 0;
  const char* _nodual = std::getenv("SCORCH_SPMM_NEON_NODUAL");
  const bool neon_no_dual = _nodual && *_nodual && std::atol(_nodual) != 0;
  (void)neon_no_dual;
#endif
#if defined(__AVX2__) && defined(__FMA__) && defined(SCORCH_TUNE_HOOKS)
  // A/B hook: refresh the regtile partial-tile toggle once per op (read by
  // scorch_spmm_row_regtile). SCORCH_REGTILE_BASE=1 -> legacy runtime-nv partial.
  { const char* e = std::getenv("SCORCH_REGTILE_BASE");
    g_scorch_regtile_base = (e && *e && std::atol(e) != 0) ? 1 : 0; }
  // A/B hook: force the masked load/store path even where k is a multiple of 8,
  // which is the only way to time the masked and unmasked instantiations against
  // each other in ONE process against ONE binary. Compiled out of the shipped .so,
  // where full_last is const and the branch on it folds away.
  { const char* e = std::getenv("SCORCH_SPMM_MASKED");
    if (e && *e && std::atol(e) != 0) full_last = false; }
  // A/B hook: number of independent nonzero streams the narrow-k register kernel
  // runs. 0 (unset) keeps the shipped 2-stream form. Loop-invariant, so the row
  // loop's dispatch on it hoists.
  int narrowk_unroll = 0;
  { const char* e = std::getenv("SCORCH_NARROWK_UNROLL");
    if (e && *e) { long v = std::atol(e); if (v > 2 && v <= 8) narrowk_unroll = (int)v; } }
  // A/B hook: the exact-width narrow-k kernel's unroll depth. 0 (unset) keeps the
  // register-block kernel, which at these widths masks every load and the store.
  { const char* e = std::getenv("SCORCH_NARROWK_EXACT");
    if (e && *e) { long v = std::atol(e);
      if (v == 0 || v == 2 || v == 4 || v == 8) narrowk_exact = (int)v; } }
  { const char* e = std::getenv("SCORCH_NARROWK_EXACT_K1");
    if (e && *e) narrowk_exact_k1 = std::atol(e) != 0; }
  // A/B hook: prefetch distance in NONZEROS for the narrow-k register kernel.
  // 0 (unset) keeps the shipped kernel and its next-but-one prefetch.
  int narrowk_pf = 0;
  { const char* e = std::getenv("SCORCH_NARROWK_PF");
    if (e && *e) { long v = std::atol(e);
      if (v == 4 || v == 8 || v == 16 || v == 32) narrowk_pf = (int)v; } }
#endif

#if defined(__AVX2__) && defined(__FMA__)
  // Route k=1 through the nonzero-axis gather kernel. See its definition for the
  // measured map across k: k=1 is where K gathers stay cheaper than eight masked
  // FMAs, and from k=2 up the gather gives back more than it gains.
  int narrowk_gather = (B1_size == 1) ? 1 : 0;
  // Independent gather streams in the nonzero-axis kernel. 1 is the shipped path.
  // Only combinations whose K*S accumulators fit in about half the 16 architectural
  // YMM registers are instantiated; anything else falls back to one stream.
  int narrowk_streams = 1;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_NARROWK_GATHER_STREAMS");
    if (e && *e) { long v = std::atol(e);
      if (v == 2 || v == 4 || v == 8) narrowk_streams = (int)v; } }
#endif
#ifdef SCORCH_TUNE_HOOKS
  // A/B override, authoritative so the whole k=1..8 map is reproducible from this
  // binary: 0 forces the shipped regblock kernel at every width, and a positive
  // value routes at every width whose row needs one vector, subject to that mean
  // row length. A floor buys nothing in the end -- the cells it would exclude were
  // positive too -- but it is how the map was drawn.
  { const char* e = std::getenv("SCORCH_NARROWK_GATHER");
    if (e && *e) {
      const long v = std::atol(e);
      if (v <= 0) {
        narrowk_gather = 0;
      } else {
        const long mean_row = A0_size > 0 ? (long)(A1_pos[A0_size] / A0_size) : 0;
        narrowk_gather = (mean_row >= v) ? 1 : 0;
      }
    } }
#endif
#endif

  // A register-resident row kernel writes the whole row, so the workspace only
  // exists for architectures that have neither (and for the force_workspace hook).
#define SCORCH_SPMM_NEEDS_WORKSPACE                                            \
  (!(defined(__AVX2__) && defined(__FMA__)) && !defined(__ARM_NEON)) ||        \
      defined(SCORCH_TUNE_HOOKS)
  scorch_unique_buffer<scalar_t> worker_workspaces;
#if SCORCH_SPMM_NEEDS_WORKSPACE
  worker_workspaces = scorch_make_aligned_buffer_pool<scalar_t>(
      static_cast<size_t>(nthreads), static_cast<size_t>(kTile));
#endif

  // The per-worker body (atomic row work-stealing). Factored into a lambda so it
  // can be launched EITHER through a private libgomp team (#pragma omp, default)
  // OR through torch's own intra-op pool (at::parallel_for). The latter shares
  // one warm pool with the surrounding torch epilogue (bias/act) so there is no
  // cross-runtime thread-team reformation at each op boundary — the drop-in-
  // pipeline "same thread pool" composition. Work distribution is byte-identical
  // (same next_row atomic, same regblock/regtile kernels); only the launch differs.
  auto scorch_spmm_worker = [&](int worker_id, int team_size) {
    // Per-thread workspace for the fallback path (cache-line aligned, lives in
    // L1). On AVX2 the register kernels (regblock k<=32 / regtile k>32) own every
    // row, and on NEON scorch_spmm_row_neon does, so neither shipped build touches
    // it. Allocated only where there is no register kernel at all, or for the
    // force_workspace hook.
    scalar_t* SCORCH_RESTRICT ws = nullptr;
#if SCORCH_SPMM_NEEDS_WORKSPACE
    ws = worker_workspaces.get() +
        static_cast<size_t>(worker_id) * static_cast<size_t>(kTile);
#else
    (void)worker_id;
#endif

    // Empty output rows first, in wide slices, as long contiguous memsets. Needs
    // no barrier against the arithmetic loop below, and none between the workers:
    // this writes only rows the arithmetic skips, and the arithmetic writes only
    // rows this skips, whatever order the two run in.
    //
    // The A/B arm that ASSIGNS a slice per worker partitions by the ACTUAL team
    // size, not the requested one, because OpenMP may hand back a smaller team and
    // a slice nobody owns would ship uninitialised memory. The stolen form has no
    // such hazard: the counter covers every row whatever the team size.
    // One stolen zero slice, or false when the counter is exhausted.
    auto zero_one_slice = [&]() -> bool {
      const int zs = next_zero_row.fetch_add(zchunk, std::memory_order_relaxed);
      if (zs >= A0_size) return false;
      const int ze = std::min(zs + zchunk, A0_size);
      for (int i = zs; i < ze;) {
        if (A1_pos[i] == A1_pos[i + 1]) {
          int run = i + 1;
          while (run < ze && A1_pos[run] == A1_pos[run + 1]) run++;
          memset(C_values + (size_t)i * (size_t)C1_size, 0,
                 out_row_bytes * (size_t)(run - i));
          i = run;
        } else {
          i++;
        }
      }
      return true;
    };
    bool zero_slices_left = zero_stolen_slice;
    if (zero_stolen_slice && !zero_alternate) {   // empty_rows != 0 is folded in
      while (zero_one_slice()) {
      }
      zero_slices_left = false;
    } else if (zero_static_slice && team_size > 0) {   // A/B arm: assigned, not stolen
      const int64_t lo = (int64_t)worker_id * (int64_t)A0_size / team_size;
      const int64_t hi = (int64_t)(worker_id + 1) * (int64_t)A0_size / team_size;
      for (int64_t i = lo; i < hi;) {
        if (A1_pos[i] == A1_pos[i + 1]) {
          int64_t run = i + 1;
          while (run < hi && A1_pos[run] == A1_pos[run + 1]) run++;
          memset(C_values + (size_t)i * (size_t)C1_size, 0,
                 out_row_bytes * (size_t)(run - i));
          i = run;
        } else {
          i++;
        }
      }
    }

    // Home-range progress for partition modes 1 and 2 (see the setup above).
    int own_w = worker_id;   // next own split to try
    int steal_t = 1;         // next victim offset to try

    // Atomic work-stealing loop with adaptive chunk size
    while (true) {
      // One zero slice per arithmetic chunk (mode 6). Alternating the two keeps a
      // pure store phase from monopolising the write path while the load path
      // idles, which is what draining the zero first does.
      if (zero_slices_left && zero_alternate && !zero_one_slice())
        zero_slices_left = false;
      int start = 0, end = 0;
      if (partition_mode == 0) {
        start = next_row.fetch_add(chunk, std::memory_order_relaxed);
        if (start >= A0_size) break;
        end = std::min(start + chunk, A0_size);
      } else {
        // Own splits first, then -- mode 2 only -- steal from the rest. A cursor
        // only ever advances, so a fetch_add that comes back at or past its range
        // limit means that range is finished for good and this worker never asks
        // again; own_w and steal_t therefore only move forward and the wasted
        // atomics over a whole call are bounded by nsplit. Every chunk is claimed
        // by exactly one fetch_add, on the owner's cursor or a thief's, so no row
        // is computed twice and none is skipped.
        bool got = false;
        const int stride = team_size > 0 ? team_size : 1;
        if (partition_mode == 3) {
          while (!got && own_w < nsplit) {
            if (scorch_ht_claim(cursors[own_w].ht, chunk, true, &start, &end))
              got = true;
            else own_w += stride;
          }
          while (!got && steal_t < nsplit) {
            const int w = (worker_id + steal_t) % nsplit;
            if (scorch_ht_claim(cursors[w].ht, chunk, false, &start, &end))
              got = true;
            else steal_t++;
          }
        } else {
          while (!got && own_w < nsplit) {
            const int lim = row_split[own_w + 1];
            const int st = cursors[own_w].v.fetch_add(chunk, std::memory_order_relaxed);
            if (st < lim) { start = st; end = std::min(st + chunk, lim); got = true; }
            else own_w += stride;
          }
          while (!got && partition_mode == 2 && steal_t < nsplit) {
            const int w = (worker_id + steal_t) % nsplit;
            const int lim = row_split[w + 1];
            const int st = cursors[w].v.fetch_add(chunk, std::memory_order_relaxed);
            if (st < lim) { start = st; end = std::min(st + chunk, lim); got = true; }
            else steal_t++;
          }
        }
        if (!got) break;
      }

      for (int i = start; i < end; i++) {
        const int pA_begin = A1_pos[i];
        const int pA_end   = A1_pos[i + 1];
        if (pA_begin == pA_end) {
          // Structurally empty, so there is nothing to compute -- and this worker
          // has the row in hand, so it zeroes it here rather than in a pass of its
          // own. One team for the whole call, the zero spread over the same rows as
          // the arithmetic and interleaved with it, each row first-touched by the
          // core that will hold it, nothing written outside the empty rows, and no
          // second walk of A1_pos. The fused Linear kernel below already worked
          // this way -- its worker writes act(bias) into a channel with no
          // nonzeros -- so this makes the drop-in SpMM agree with it.
          //
          // Consecutive empty rows are one memset, not one each, bounded by the
          // stolen chunk so no two workers ever write the same bytes. That width is
          // chosen for arithmetic and is narrow for a store stream; widening it is
          // mode 6 above, and the second A1_pos walk that costs more than the
          // streaming gains (see the mode table). Mode 3 drops the merge, to price
          // it: about 3% here.
          if (zero_in_loop) {
            int run = i + 1;
            if (zero_merge_runs)
              while (run < end && A1_pos[run] == A1_pos[run + 1]) run++;
            memset(C_values + (size_t)i * (size_t)C1_size, 0,
                   out_row_bytes * (size_t)(run - i));
            i = run - 1;          // the loop's ++ steps past the run
          }
          continue;
        }

        scalar_t* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

#if defined(__AVX2__) && defined(__FMA__)
        // AVX2 register-resident kernels own every row: narrow k (<=32) holds the
        // whole output row in YMM accumulators (regblock); wide k (>32) is
        // register-TILED in 64-wide k-tiles (regtile). Both avoid the per-nnz
        // workspace round-trip that dominated the cache-hot wide-k case (banded
        // dense k=64 was 0.59x of MKL). The workspace loop below is the non-AVX2
        // fallback (or the force_workspace A/B hook).
#ifdef SCORCH_TUNE_HOOKS
        if (!force_workspace)
#endif
        {
          if (narrow_k) {
            // full_last depends only on k, so it is loop-invariant: the branch
            // hoists and each row runs one straight-line instantiation.
            #define SCORCH_RB(NV) \
              (full_last \
                 ? scorch_spmm_row_regblock<scalar_t, NV, true>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last) \
                 : scorch_spmm_row_regblock<scalar_t, NV, false>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last))
            // The widths where regblock's mask covers the whole row: float k=1..7 and
            // double k=1..3. Float k=1 is instantiated but only routed when asked for
            // separately, because the nonzero-axis gather kernel owns it -- otherwise
            // this arm would swap two kernels in one step and neither would be
            // attributable. Placed ahead of the gather dispatch for that reason.
            if (narrowk_exact) {
              #define SCORCH_RNE(KK, UN) \
                scorch_spmm_row_narrow_exact<scalar_t, KK, UN>( \
                    A1_crd, A_val, B_val, C_row, pA_begin, pA_end)
              #define SCORCH_RNE_U(KK) \
                switch (narrowk_exact) { \
                  case 2: SCORCH_RNE(KK, 2); break; \
                  case 4: SCORCH_RNE(KK, 4); break; \
                  default: SCORCH_RNE(KK, 8); break; \
                }
              bool took = false;
              if constexpr (std::is_same<scalar_t, float>::value) {
                switch (B1_size) {
                  case 1: if (narrowk_exact_k1) { SCORCH_RNE_U(1); took = true; } break;
                  case 2: SCORCH_RNE_U(2); took = true; break;
                  case 3: SCORCH_RNE_U(3); took = true; break;
                  case 4: SCORCH_RNE_U(4); took = true; break;
                  case 5: SCORCH_RNE_U(5); took = true; break;
                  case 6: SCORCH_RNE_U(6); took = true; break;
                  case 7: SCORCH_RNE_U(7); took = true; break;
                  default: break;
                }
              } else {
                switch (B1_size) {
                  case 1: SCORCH_RNE_U(1); took = true; break;
                  case 2: SCORCH_RNE_U(2); took = true; break;
                  case 3: SCORCH_RNE_U(3); took = true; break;
                  default: break;
                }
              }
              #undef SCORCH_RNE_U
              #undef SCORCH_RNE
              if (took) continue;
            }
            if (narrowk_gather && nvec == 1 &&
                std::is_same<scalar_t, float>::value) {
              // nvec==1 means k <= 8 for float, so K is B1_size itself.
              #define SCORCH_RG(KK) \
                scorch_spmm_row_gather_f32<KK>( \
                    A1_crd, (const float*)A_val, (const float*)B_val, \
                    (float*)C_row, pA_begin, pA_end)
              #define SCORCH_RGS(KK, SS) \
                scorch_spmm_row_gather_f32_ms<KK, SS>( \
                    A1_crd, (const float*)A_val, (const float*)B_val, \
                    (float*)C_row, pA_begin, pA_end)
              if constexpr (std::is_same<scalar_t, float>::value) {
                // K*8+S keys the pair; only the register-feasible pairs exist.
                switch (narrowk_streams > 1 ? B1_size * 16 + narrowk_streams : 0) {
                  case 1 * 16 + 2: SCORCH_RGS(1, 2); break;
                  case 1 * 16 + 4: SCORCH_RGS(1, 4); break;
                  case 1 * 16 + 8: SCORCH_RGS(1, 8); break;
                  case 2 * 16 + 2: SCORCH_RGS(2, 2); break;
                  case 2 * 16 + 4: SCORCH_RGS(2, 4); break;
                  case 4 * 16 + 2: SCORCH_RGS(4, 2); break;
                  default:
                    switch (B1_size) {
                      case 1: SCORCH_RG(1); break;
                      case 2: SCORCH_RG(2); break;
                      case 3: SCORCH_RG(3); break;
                      case 4: SCORCH_RG(4); break;
                      case 5: SCORCH_RG(5); break;
                      case 6: SCORCH_RG(6); break;
                      case 7: SCORCH_RG(7); break;
                      case 8: SCORCH_RG(8); break;
                      default: SCORCH_RB(1); break;
                    }
                }
              }
              #undef SCORCH_RGS
              #undef SCORCH_RG
              continue;
            }
#ifdef SCORCH_TUNE_HOOKS
            #define SCORCH_RBD(NV, UN) \
              (full_last \
                 ? scorch_spmm_row_regblock_deep<scalar_t, NV, true, UN>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last) \
                 : scorch_spmm_row_regblock_deep<scalar_t, NV, false, UN>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last))
            if (narrowk_unroll) {
              // NVEC * UNROLL accumulators must stay resident, so the deeper
              // streams are only instantiated where the vector count is small.
              switch (nvec * 16 + narrowk_unroll) {
                case 1 * 16 + 4: SCORCH_RBD(1, 4); break;
                case 1 * 16 + 8: SCORCH_RBD(1, 8); break;
                case 2 * 16 + 4: SCORCH_RBD(2, 4); break;
                case 2 * 16 + 8: SCORCH_RBD(2, 8); break;
                case 3 * 16 + 4: SCORCH_RBD(3, 4); break;
                case 4 * 16 + 4: SCORCH_RBD(4, 4); break;
                default:
                  switch (nvec) {
                    case 1: SCORCH_RB(1); break;
                    case 2: SCORCH_RB(2); break;
                    case 3: SCORCH_RB(3); break;
                    case 4: SCORCH_RB(4); break;
                  }
              }
              continue;
            }
            #undef SCORCH_RBD
            #define SCORCH_RBP(NV, D) \
              (full_last \
                 ? scorch_spmm_row_regblock_pf<scalar_t, NV, true, D>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last) \
                 : scorch_spmm_row_regblock_pf<scalar_t, NV, false, D>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last))
            if (narrowk_pf) {
              switch (nvec * 64 + narrowk_pf) {
                case 1 * 64 + 4:  SCORCH_RBP(1, 4); break;
                case 1 * 64 + 8:  SCORCH_RBP(1, 8); break;
                case 1 * 64 + 16: SCORCH_RBP(1, 16); break;
                case 1 * 64 + 32: SCORCH_RBP(1, 32); break;
                case 2 * 64 + 4:  SCORCH_RBP(2, 4); break;
                case 2 * 64 + 8:  SCORCH_RBP(2, 8); break;
                case 2 * 64 + 16: SCORCH_RBP(2, 16); break;
                case 2 * 64 + 32: SCORCH_RBP(2, 32); break;
                case 3 * 64 + 8:  SCORCH_RBP(3, 8); break;
                case 3 * 64 + 16: SCORCH_RBP(3, 16); break;
                case 4 * 64 + 8:  SCORCH_RBP(4, 8); break;
                case 4 * 64 + 16: SCORCH_RBP(4, 16); break;
                default:
                  switch (nvec) {
                    case 1: SCORCH_RB(1); break;
                    case 2: SCORCH_RB(2); break;
                    case 3: SCORCH_RB(3); break;
                    case 4: SCORCH_RB(4); break;
                  }
              }
              continue;
            }
            #undef SCORCH_RBP
#endif
            switch (nvec) {
              case 1: SCORCH_RB(1); break;
              case 2: SCORCH_RB(2); break;
              case 3: SCORCH_RB(3); break;
              case 4: SCORCH_RB(4); break;
            }
            #undef SCORCH_RB
          } else {
            scorch_spmm_row_regtile<scalar_t>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end);
          }
          continue;
        }
#endif

#if defined(__ARM_NEON) && !defined(__AVX2__)
        // The NEON strip kernel owns every row here, exactly as the AVX2 block above
        // does: accumulators stay in registers for the whole pass over the row, so
        // the workspace round-trip per nonzero is gone. The loop below survives only
        // as the no-SIMD fallback and as the force_workspace A/B hook.
#ifdef SCORCH_TUNE_HOOKS
        if (!force_workspace)
#endif
        {
          // ALLOW_DUAL=false selects the kernel as it was before the 2-nonzero
          // unroll, reachable only under the tune hooks, so the unroll can be priced
          // in this binary against both the workspace arm and its own predecessor
          // rather than across two builds.
#ifdef SCORCH_TUNE_HOOKS
          if (neon_no_dual) {
            scorch_spmm_row_neon_hoisted<scalar_t, false>(
                A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end,
                neon_single, neon_nv, neon_tail);
            continue;
          }
#endif
          scorch_spmm_row_neon_hoisted<scalar_t, true>(
              A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end,
              neon_single, neon_nv, neon_tail);
          continue;
        }
#endif

        for (int k_out = 0; k_out < B1_size; k_out += kTile) {
          const int kw = std::min(kTile, B1_size - k_out);

          memset(ws, 0, kw * sizeof(scalar_t));

          for (int pA = pA_begin; pA < pA_end; pA++) {
            const int j = A1_crd[pA];
            const scalar_t a = A_val[pA];
            const scalar_t* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size + k_out;

            if (pA + 1 < pA_end) {
              __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k_out, 0, 1);
            }

            for (int k = 0; k < kw; k++) {
              ws[k] += a * B_row[k];
            }
          }

          memcpy(C_row + k_out, ws, kw * sizeof(scalar_t));
        }
      }
    }
    // Mode 6 only: the arithmetic ran out before the zero slices did, and
    // whatever is left is still this worker's to finish.
    while (zero_slices_left && zero_one_slice()) {
    }
  };

  // Launch. Default: private libgomp team (#pragma omp) — byte-identical to the
  // pre-composition kernel. When this is a drop-in pipeline op (the caller sets
  // `atparallel` together with nthreads_override), run the SAME workers on torch's
  // own intra-op pool via at::parallel_for so the SpMM shares one warm team with
  // the surrounding torch epilogue (bias/act): no cross-runtime team reformation
  // at each op boundary — the sparse-AE @0.99 pool-transition tax (mid-size @0.99
  // cur/mkl 1.2-1.3 -> 0.9). Gate it to products that (a) clear the fork/join
  // floor (work >= grain, same test that adopts the host count) AND (b) can feed
  // the FULL host pool: a ROW-STARVED product (rows < nthreads_override*
  // ROWS_PER_THREAD) leaves too few rows per worker, so at::parallel_for's task
  // fan-out costs more than a raw omp team. That row-count test is the exact
  // discriminator measured on the SuiteSparse panel: only arc130 (130 rows) is
  // row-starved and it is the ONLY cell at::parallel_for regresses; every AE layer
  // (>=512 rows) and GCN graph clears it and is neutral-to-better. at::parallel_for
  // over [0,nthreads) grain 1 spawns min(nthreads, at::get_num_threads()) workers,
  // each draining rows via the atomic, so a short/excess worker count stays
  // correct. Env SCORCH_SPMM_ATPARALLEL forces the choice for A/B (1/0), bypassing
  // the gate.
  // work_nnz, not work: the zeroing term above sizes the TEAM, and this gate is a
  // different question -- whether the surrounding torch pipeline's warm pool should
  // be shared with this op. Nothing measured says an output span full of empty rows
  // changes that answer, so the launch path every existing shape takes is unchanged.
  bool use_atparallel = atparallel && nthreads_override > 0
      && work_nnz >= SCORCH_GRAIN_SPMM
      && (long)A0_size >= (long)nthreads_override * SCORCH_ROWS_PER_THREAD;
  if (const char* _atpf = std::getenv("SCORCH_SPMM_ATPARALLEL"))
    if (*_atpf) use_atparallel = (std::atol(_atpf) != 0);
  if (use_atparallel) {
    // E-CORE RECRUIT (M5/hybrid-P+E). at::parallel_for runs on torch's intra-op
    // pool, which on Apple M-series is the 6 P-cores and excludes the 12 E-cores.
    // When the work-aware nthreads justifies >= 2x that pool, launch our own omp
    // team to pull in the idle E-cores on this bandwidth-bound SpMM. The 2x-the-pool
    // gate keeps small/row-starved products (incl. the FEM panel's arc130 and the
    // small GCN graphs, whose by-work nthreads stays <= the pool) on the warm pool,
    // and can NEVER fire on an all-physical-cores pool (x86: pool=24, nproc=32 w/
    // SMT, nthreads<=32 < 48) -- so the x86 pipeline-pool launch and the FEM panel
    // guardrail are byte-unchanged. Identical gate to spmm_csr_linear_fused_float.
    const int atpool = at::get_num_threads();
    if (nthreads >= 2 * atpool) {
      #pragma omp parallel num_threads(nthreads)
      {
        scorch_spmm_worker(omp_get_thread_num(), omp_get_num_threads());
      }
    } else {
      // Every id in [0, nthreads) is executed here, so nthreads IS the partition
      // the static zero above can rely on, whatever pool size torch runs it on.
      at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {
        for (int64_t w = wbeg; w < wend; ++w)
          scorch_spmm_worker(static_cast<int>(w), nthreads);
      });
    }
  } else {
    #pragma omp parallel num_threads(nthreads)
    {
      scorch_spmm_worker(omp_get_thread_num(), omp_get_num_threads());
    }
  }

  return C_values_torch;
}

// The float32 entry every existing caller uses. A one-line forwarder, so plan.h and
// spmm_csr_float_v2 are unchanged and the float32 instantiation is the same code the
// hand-written kernel was.
torch::Tensor spmm_csr_float_v2_core(
                int C0_size, int C1_size, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const float* SCORCH_RESTRICT A_val,
                int B1_size, const float* SCORCH_RESTRICT B_val,
                int tile_size, int nthreads_override, bool atparallel) {
  return spmm_csr_v2_core<float>(C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val,
                                 B1_size, B_val, tile_size, nthreads_override,
                                 atparallel);
}

Tensor spmm_csr_float_v2(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 256,
                int nthreads_override = -1, bool atparallel = false) {
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = spmm_csr_float_v2_core(
      result_shape[0], result_shape[1], A_shape[0],
      A_mode_indices[1][0].data_ptr<int>(),
      A_mode_indices[1][1].data_ptr<int>(),
      A_values.data_ptr<float>(),
      B_shape[1], B_values.data_ptr<float>(),
      tile_size, nthreads_override, atparallel);
  return C;
}

// float64 CSR x dense, on the SAME kernel as float32.
//
// The float64 route used to resolve spmm_csr_double, which is spmm_csr_typed_core --
// the original reference kernel. That kernel accumulates straight into C with a
// read-modify-write per nonzero, memsets the whole output serially before the
// parallel region, and hands rows out with a fixed `schedule(dynamic, 16)` and no
// thread policy. Every one of those is a thing the float32 path was measured fixing.
// Removing the per-call index-validation tax (see the top of this file) took float64
// from 0.409x to 0.853x of MKL and left it below parity on 5 of 6 redwood cells,
// which was the tax having hidden a plain kernel-quality gap: float32 beat MKL on
// the same cells with the same taxes removed.
//
// So this is not a new float64 kernel. It is the float32 one, instantiated at double,
// which is why the register kernels above are templated rather than copied. float64
// gets register-resident accumulation, the atomic row work-stealing with the analytic
// chunk width, the work-aware thread count, and the empty-row-only output zeroing,
// and it will get whatever the float32 path gets next.
//
// Kept alongside spmm_csr_double rather than replacing it, mirroring
// spmm_csr_float / spmm_csr_float_v2: the reference kernel stays as the thing the
// tests compare against.
//
// Guarded on having a register-resident row kernel, not on a particular ISA. What
// makes this kernel win is that the whole output row lives in registers for one pass
// over the nonzeros: AVX2's regblock/regtile, or NEON's strip kernel. Where neither
// exists the row falls to the workspace loop, and that loop is NOT better than the
// reference kernel at low degree and narrow k -- so the #else really is the
// measurement, and it is worth keeping the numbers that established it.
//
// Routing float64 through the workspace loop on the M5, over 50 cells: geometric mean
// 1.64x the reference, but pubmed at k=8 (degree 5.5) read 0.69x and citeseer at k=8
// 0.91x. Two attempts to close that failed -- accumulating in place with an assigning
// first nonzero (0.69x, no change) and zeroing C in bulk instead of per empty row
// (0.80x at that cell, and it cost the mid cells more than it gained, dropping the
// grid to 1.27x). Both reverted. Shipping an unexplained 1.45x regression on a real
// GCN shape to collect a 1.64x mean is the trade the performance convention refuses.
//
// The register kernel is what closed it, and ARM has one now. An earlier version of
// this comment said a NEON kernel could not be a specialization of the traits above
// because NEON has no masked load or store, so the ragged tail would need a different
// strategy. The premise is true and the conclusion was wrong: the tail wants scalar
// accumulators updated in the same pass over the row, which needs no mask and no
// overread, and scorch_spmm_row_neon_regtile in this file had been doing exactly that
// for the fused Linear kernel the whole time. See scorch_spmm_row_neon_strip above:
// float64 on the M5 now reads geomean 1.339 against the workspace loop it replaced
// over 188 readings, none below 1.0.
//
// Split in two like the float32 entry, and for one reason beyond that: the choice of
// row kernel below is a POLICY, and pybind is not the only caller that needs it --
// plan.h dispatches this same product without going through the entry above. Both call
// this function, so the policy is written once. A plan that decided it again would be
// free to get it wrong in the one direction that matters, dropping a host to the
// workspace loop that has a register kernel available.
torch::Tensor spmm_csr_double_v2_core(
                int C0_size, int C1_size, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const double* SCORCH_RESTRICT A_val,
                int B1_size, const double* SCORCH_RESTRICT B_val,
                int tile_size, int nthreads_override, bool atparallel) {
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__ARM_NEON)
  return spmm_csr_v2_core<double>(C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val,
                                  B1_size, B_val, tile_size, nthreads_override,
                                  atparallel);
#else
  // The composition hints are accepted and ignored here, so neither the Python side
  // nor plan.h needs a per-platform branch: the reference kernel takes its thread
  // count from the ambient OpenMP team, which is what it did when float64 resolved to
  // it directly.
  (void)nthreads_override;
  (void)atparallel;
  return spmm_csr_typed_core<double>(C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val,
                                     B1_size, B_val, tile_size);
#endif
}

Tensor spmm_csr_double_v2(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 256,
                int nthreads_override = -1, bool atparallel = false) {
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = spmm_csr_double_v2_core(
      result_shape[0], result_shape[1], A_shape[0],
      A_mode_indices[1][0].data_ptr<int>(),
      A_mode_indices[1][1].data_ptr<int>(),
      A_values.data_ptr<double>(),
      B_shape[1], B_values.data_ptr<double>(),
      tile_size, nthreads_override, atparallel);
  return C;
}

// ---------------------------------------------------------------------------
// spmm_csr_float_tilej — column-panel ("tile-j") SpMM for the HIGH-DEGREE,
// OPERAND-OVER-LLC thrash regime (reddit/products-class graphs, and any wide-B
// scattered general-library workload). Reached ONLY when the adaptive tiling
// selector (src/scorch/tiling.py, wired in ops.matmul) fires — i.e. when the
// dense operand B (J*4N bytes) exceeds the last-level cache AND the degree is
// high enough that column-blocking recovers more B-reuse than it costs in output
// re-traffic (the thrash-and-tile rule). On every OTHER shape the selector routes
// to spmm_csr_float_v2 (byte-unchanged), so this kernel can only ever ADD a win.
//
// v2 streams each output row full-width once (row-major). When B thrashes DRAM
// (reddit @k=256: B is 239MB >> 24MB SLC), v2 re-fetches most of B from DRAM per
// row -> bandwidth-bound at ~70 GFLOP/s. tile-j blocks the CONTRACTION dim j into
// panels of width Jc = C/(4N) columns: for each panel it sweeps all M rows, so the
// panel's <=Jc B-rows (Jc*4N ~ C bytes) stay cache-resident and are reused across
// the M rows. Measured M5 reddit: 1.07x/1.41x/1.84x/2.00x over v2 at N=32/64/128/
// 256, bit-exact. No panel materialization: CSR rows are column-sorted, so each
// panel's slice of a row is a contiguous [lower_bound(j0), lower_bound(j1)) range.
//
// C accumulates across panels, so it is FULLY zeroed first (v2 zeros only empty
// rows because it writes each row once); the panels run as barrier-separated
// omp-for loops inside ONE parallel region (panel p finishes before p+1, so the
// per-row accumulation is race-free and each panel's B stays hot). Raw omp
// num_threads(nthreads) engages M5's E-cores directly (no at::parallel_for cap) —
// tile-j only fires on big BW-bound work where all cores are wanted; on x86 this
// is the same all-physical-cores launch v2 uses. Jc is passed from Python (the
// selector knows the queried cache size); Jc<=0 or >=J degenerates to a single
// full-width panel (== a slower v2, never reached because the selector gates it).
// ---------------------------------------------------------------------------
// Pointer-based kernel; see the note on spmm_csr_float_v2_core for why the entry
// is split in two.
torch::Tensor spmm_csr_float_tilej_core(
                int C0_size, int C1_size, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const float* SCORCH_RESTRICT A_val,
                int J, int N, const float* SCORCH_RESTRICT B_val,
                int Jc, int nthreads_override) {
  const size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  // tile-j accumulates across panels, so C starts zeroed. torch::zeros gets a
  // clean buffer from the same allocator MKL uses (the O(rows) empty-only trick
  // v2 uses does not apply here — every row is +='d across panels).
  torch::Tensor C_values_torch = torch::zeros({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();

  // Panel width. Guard degenerate inputs -> single full-width panel.
  if (Jc <= 0 || Jc > J) Jc = J;
  const int npanel = (J + Jc - 1) / Jc;

  // Work-aware thread cap (same policy as v2). tile-j fires only on big thrash
  // work, so this returns every core; raw omp num_threads() then pulls in M5's
  // E-cores (unlike v2's at::parallel_for pipeline pool). Adopt a >policy host
  // count if supplied (never below policy so big graphs keep all cores).
  const long total_nnz = A1_pos[A0_size];
  const long k_eff = N < 16 ? 16L : (long)N;
  const long work = total_nnz * k_eff;
  int nthreads = scorch_nthreads(work, A0_size, SCORCH_GRAIN_SPMM);
  if (nthreads_override > 0) {
    const long hw = (long)omp_get_num_procs();
    long cand = (long)nthreads_override < hw ? (long)nthreads_override : hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }

  // One fresh parallel-for PER PANEL (matches the validated prototype). All
  // threads sweep the SAME panel together, so its <=Jc B-rows stay cache-hot and
  // are reused across the M rows — that is tile-j's recovered reuse. The fork/join
  // between panels (npanel = J/Jc <= ~20) also lets the OS rebalance across M5's
  // P+E clusters; a single persistent team pinned threads and HALVED throughput.
  // Panels run sequentially, and within a panel each row i is owned by one thread,
  // so the per-row accumulation is race-free.
  for (int p = 0; p < npanel; ++p) {
    const int j0 = p * Jc;
    const int j1 = std::min(j0 + Jc, J);
    #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
    for (int i = 0; i < A0_size; ++i) {
      const int rb = A1_pos[i];
      const int re = A1_pos[i + 1];
      if (rb == re) continue;
      // contiguous slice [pb,pe) of row i whose columns fall in [j0,j1)
      const int* SCORCH_RESTRICT lo = std::lower_bound(A1_crd + rb, A1_crd + re, j0);
      const int* SCORCH_RESTRICT hi = std::lower_bound(lo, A1_crd + re, j1);
      int pb = (int)(lo - A1_crd);
      const int pe = (int)(hi - A1_crd);
      if (pb == pe) continue;
      float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;
      for (; pb < pe; ++pb) {
        const float a = A_val[pb];
        const float* SCORCH_RESTRICT B_row = B_val + (size_t)A1_crd[pb] * (size_t)N;
        if (pb + 1 < pe)
          __builtin_prefetch(B_val + (size_t)A1_crd[pb + 1] * (size_t)N, 0, 1);
        // plain loop: -O3 -march=native -ffast-math auto-vectorizes cleanly (a
        // manual 16-wide unroll here measured ~2x SLOWER on ARM/NEON).
        for (int k = 0; k < N; ++k) C_row[k] += a * B_row[k];
      }
    }
  }

  return C_values_torch;
}

Tensor spmm_csr_float_tilej(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int Jc = 0, int nthreads_override = -1) {
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = spmm_csr_float_tilej_core(
      result_shape[0], result_shape[1], A_shape[0],
      A_mode_indices[1][0].data_ptr<int>(),
      A_mode_indices[1][1].data_ptr<int>(),
      A_values.data_ptr<float>(),
      B_shape[0], B_shape[1], B_values.data_ptr<float>(),
      Jc, nthreads_override);
  return C;
}

// ---------------------------------------------------------------------------
// spmm_csr_float_tileijk — 3D-blocked ("tile-ijk") SpMM with a B WIDTH-PANEL
// RELAYOUT, for the SCATTERED + VERY-WIDE-B regime where even tile-j erodes.
//
// tile-j blocks only the contraction dim j; its output C is re-streamed
// P = J*4N/C times, so its C-traffic grows ~N^2 and its throughput collapses as
// N widens (measured redwood scatter deg200: 262 -> 239 -> 96 GFLOP/s at
// N=512/2048/8192). tile-ijk additionally blocks the FREE dim N into width-strips
// of Nc columns. For each strip it:
//   (1) RELAYS that strip of B into a CONTIGUOUS [J x w] buffer (w <= Nc). Every
//       B sub-block read in step (3) is then contiguous -> no HW-prefetch
//       pollution. (A naive strided-Nc slice of row-major B pulls the unused
//       rest-of-row into cache and re-pollutes it, which is exactly why a plain
//       tile-jk on row-major B loses; the relayout is the whole point.)
//   (2) zeros a CONTIGUOUS Cp [C0 x Nc] that stays cache-resident across (3);
//   (3) accumulates every contraction column-panel of A into Cp, reading the
//       relaid strip (materialization-free: CSR rows are col-sorted, so a panel's
//       slice of a row is [lower_bound(j0), lower_bound(j1)) — no CSC/panel mats);
//   (4) writes Cp to the strided C strip C[:, k0:k0+w] ONCE.
// C-traffic is now ~N (linear: each C entry written once per strip); A is
// re-scanned nk = ceil(N/Nc) times. Result: throughput holds ~270-289 GFLOP/s
// across N=512..8192 where tile-j collapses (redwood 9.0x vs none at N=8192;
// M5 band N=16384 tile-ijk 286 vs tile-j 156 GFLOP/s).
//
// HONEST RELAYOUT COST: the relayout is an O(J*N) copy of B, done ONE strip at a
// time (each B element is read exactly once total across all strips; the only
// extra memory is the reused J*Nc strip buffer + the C0*Nc Cp, NOT a full 2nd
// copy of B). It runs INSIDE this kernel per call, so the runtime micro-probe
// (src/scorch/tiling.py) that routes here times the FULL cost (relayout + compute)
// against tile-j and v2 -> the relayout is never hidden. For a general SpMM
// library each call is one-shot (B reuse is not assumed), so paying it per call is
// the honest accounting; the probe only keeps tile-ijk when relayout + linear-C
// compute still beats tile-j's ~N^2 compute. Reached ONLY via the selector's
// wide-N scattered branch (no current scorch workload has N>=1024), so v2/tile-j
// serve everything today -> this can only ever ADD a win.
//
// Threading mirrors tile-j: work-aware scorch_nthreads, one FRESH omp parallel-for
// PER PANEL / per relayout / per write (a single persistent team pinned M5's P+E
// clusters and HALVED throughput; per-region re-fork lets the OS rebalance), plain
// inner loops (a manual unroll was ~2x slower on NEON; -O3 -march=native auto-vec).
// Byte-portable: no intrinsics, compiles on x86 + ARM. Nc/Jc passed from the
// selector; Nc<=0/>=N => single full-width strip, Jc<=0/>=J => single panel
// (both degenerate to a slower tile-j, never reached because the selector gates).
// ---------------------------------------------------------------------------
// Pointer-based kernel; see the note on spmm_csr_float_v2_core for why the entry
// is split in two.
torch::Tensor spmm_csr_float_tileijk_core(
                int C0_size, int N, int A0_size,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const float* SCORCH_RESTRICT A_val,
                int J, const float* SCORCH_RESTRICT B_val,
                int Nc, int Jc, int nthreads_override) {
  const size_t C_capacity = (size_t)C0_size * (size_t)N;
  // Every C entry is written exactly once (Cp is zeroed per strip and copied to C
  // in full, so empty A-rows and any tail rows C0>A0 land as 0), so torch::empty
  // (the MKL-comparable allocator) suffices — no pre-zero of C needed.
  torch::Tensor C_values_torch = torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();

  // Panel/strip widths. Guard degenerate inputs.
  if (Nc <= 0 || Nc > N) Nc = N;
#if defined(SCORCH_TUNE_HOOKS) && defined(__ARM_NEON) && !defined(__AVX2__)
  // A/B hook: the scalar per-nonzero accumulation this kernel used before the
  // register-resident pass, so the two can be timed in one binary. ARM only, because
  // that is the only host the register pass ships on. Compiled out of the shipped .so.
  bool tileijk_scalar_inner = false;
  if (const char* _si = std::getenv("SCORCH_TILEIJK_SCALAR"))
    if (*_si) tileijk_scalar_inner = (std::atol(_si) != 0);
#endif
  if (Jc <= 0 || Jc > J) Jc = J;
  const int nstrip = (N + Nc - 1) / Nc;
  const int npanel = (J + Jc - 1) / Jc;

  // Thread policy identical to tile-j / v2: tile-ijk fires only on big thrash
  // work, so this returns every core; raw omp num_threads() then pulls in M5's
  // E-cores. Adopt a >policy host count if supplied, never below policy.
  const long total_nnz = A1_pos[A0_size];
  const long k_eff = N < 16 ? 16L : (long)N;
  const long work = total_nnz * k_eff;
  int nthreads = scorch_nthreads(work, A0_size, SCORCH_GRAIN_SPMM);
  if (nthreads_override > 0) {
    const long hw = (long)omp_get_num_procs();
    long cand = (long)nthreads_override < hw ? (long)nthreads_override : hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }

  // Reused per-strip scratch: the CONTIGUOUS relaid B strip [J x Nc] and the
  // cache-resident output panel Cp [C0 x Nc]. Allocated once (Nc-wide); each strip
  // fills only its w<=Nc columns. torch::empty => same allocator as v2/MKL.
  torch::Tensor Brelaid_t = torch::empty({(long long)J * (long long)Nc}, torch::kFloat32);
  float* SCORCH_RESTRICT Brelaid = Brelaid_t.data_ptr<float>();
  torch::Tensor Cp_t = torch::empty({(long long)C0_size * (long long)Nc}, torch::kFloat32);
  float* SCORCH_RESTRICT Cp = Cp_t.data_ptr<float>();

  for (int s = 0; s < nstrip; ++s) {
    const int k0 = s * Nc;
    const int w = std::min(Nc, N - k0);             // this strip's actual width

    // (1) RELAYOUT strip s of B into the contiguous [J x w] buffer (row stride w).
    // Reads row-major B[:, k0:k0+w] (each B element read once across all strips).
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int j = 0; j < J; ++j) {
      const float* SCORCH_RESTRICT src = B_val + (size_t)j * (size_t)N + (size_t)k0;
      float* SCORCH_RESTRICT dst = Brelaid + (size_t)j * (size_t)w;
      for (int c = 0; c < w; ++c) dst[c] = src[c];
    }

    // (2) zero Cp[:, 0:w] over all C0 rows (row stride Nc). Empty/tail rows stay 0.
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < C0_size; ++i) {
      float* SCORCH_RESTRICT cr = Cp + (size_t)i * (size_t)Nc;
      for (int c = 0; c < w; ++c) cr[c] = 0.f;
    }

    // (3) accumulate every contraction column-panel into Cp, one FRESH parallel-for
    // per panel (keeps each Jc*w relaid B sub-block cache-hot across the M rows and
    // lets the OS rebalance across P+E clusters).
    for (int p = 0; p < npanel; ++p) {
      const int j0 = p * Jc;
      const int j1 = std::min(j0 + Jc, J);
      #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
      for (int i = 0; i < A0_size; ++i) {
        const int rb = A1_pos[i];
        const int re = A1_pos[i + 1];
        if (rb == re) continue;
        const int* SCORCH_RESTRICT lo = std::lower_bound(A1_crd + rb, A1_crd + re, j0);
        const int* SCORCH_RESTRICT hi = std::lower_bound(lo, A1_crd + re, j1);
        int pb = (int)(lo - A1_crd);
        const int pe = (int)(hi - A1_crd);
        if (pb == pe) continue;
        float* SCORCH_RESTRICT C_row = Cp + (size_t)i * (size_t)Nc;
        // Register-resident accumulation of the panel's contribution, ARM only, and
        // the numbers are why it is ARM only. The scalar loop below issues w L1 loads
        // and w L1 stores PER NONZERO into a row that is already cache-hot. Nc exists
        // so the output panel fits the CACHE; at the widths the cost model picks it
        // also fits REGISTERS, so the row can be loaded once, accumulated over the
        // panel's nonzeros, and stored once -- the same change the drop-in SpMM got,
        // applied to the loop the tiled path kept.
        //
        // The same change on AVX2 (cut the slice into 64-lane tiles, run
        // scorch_spmm_row_regtile_partial with ACCUM) was built and measured on
        // redwood over the same 15 cells, 3 passes: geomean 1.033, but 4-5 cells below
        // 1.0 in every pass, worst 0.971, all of them at Nc=112 on the short-row
        // matrices (degree 16 and 33) where 112 lanes is two register tiles and so two
        // walks of the row per panel. The M5 reads 1.217 on that grid with nothing
        // below 1.122.
        //
        // The asymmetry is the hosts, not the code. What this removes is L1 traffic to
        // the output row; redwood (~56 GB/s achieved) is DRAM-bound streaming the
        // relaid B, so shaving L1 operations buys little, while the M5 (~412 GB/s) is
        // core-bound and it buys a lot. A 3.3% mean does not pay for a 2.4% regression
        // on real shapes, so the AVX2 arm is not taken. If it is ever wanted, the
        // measured option is to gate it on the slice fitting ONE tile (w <= 8*lanes, no
        // re-walk): every Nc 16 and Nc 32 cell won, 1.027-1.058, and every Nc=112 cell
        // lost. scorch_spmm_row_regtile_partial keeps its ACCUM parameter for that.
#if defined(__ARM_NEON) && !defined(__AVX2__)
#ifdef SCORCH_TUNE_HOOKS
        if (!tileijk_scalar_inner)
#endif
        {
          scorch_spmm_row_neon<float, true, /*ACCUM=*/true>(
              A1_crd, A_val, Brelaid, w, C_row, pb, pe);
          continue;
        }
#endif
        for (; pb < pe; ++pb) {
          const float a = A_val[pb];
          const float* SCORCH_RESTRICT B_row = Brelaid + (size_t)A1_crd[pb] * (size_t)w;
          if (pb + 1 < pe)
            __builtin_prefetch(Brelaid + (size_t)A1_crd[pb + 1] * (size_t)w, 0, 1);
          for (int k = 0; k < w; ++k) C_row[k] += a * B_row[k];
        }
      }
    }

    // (4) write Cp[:, 0:w] to the strided C strip C[:, k0:k0+w], once.
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < C0_size; ++i) {
      const float* SCORCH_RESTRICT src = Cp + (size_t)i * (size_t)Nc;
      float* SCORCH_RESTRICT dst = C_values + (size_t)i * (size_t)N + (size_t)k0;
      for (int c = 0; c < w; ++c) dst[c] = src[c];
    }
  }

  return C_values_torch;
}

Tensor spmm_csr_float_tileijk(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int Nc = 0, int Jc = 0,
                int nthreads_override = -1) {
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = spmm_csr_float_tileijk_core(
      result_shape[0], result_shape[1], A_shape[0],
      A_mode_indices[1][0].data_ptr<int>(),
      A_mode_indices[1][1].data_ptr<int>(),
      A_values.data_ptr<float>(),
      B_shape[0], B_values.data_ptr<float>(),
      Nc, Jc, nthreads_override);
  return C;
}

// ---------------------------------------------------------------------------
// spmm_csr_linear_fused_float — feature-major fused Linear layer:
//     Y[o, b] = act( sum_{i in nnz(W[o,:])} W[o,i] * X[i, b]  +  bias[o] )
// for a sparse CSR weight W[out, in] (A) and a dense, FEATURE-MAJOR activation
// X[in, batch] (B, row-major). Output Y is [out, batch], feature-major, so a
// whole autoencoder forward stays feature-major throughout (transpose X once in,
// once out) — no per-layer transpose, and NO separate torch bias/act epilogue.
//
// This IS spmm_csr_float_v2 (same regtile/regblock inner loops, same thread
// policy, same at::parallel_for pool composition) with the Linear epilogue folded
// INTO the parallel region: after a worker computes an output row Y[o,:] (all
// `batch` free-dim entries), it adds the per-OUTPUT-CHANNEL scalar bias[o]
// (broadcast over the batch) and applies the activation, before moving on. One
// warm parallel region does SpMM + bias + act with the row still hot in cache —
// the composition tax the shipped unfused path pays at every SpMM->torch-epilogue
// handoff under passive OMP is gone.
//
// Bias here is PER-ROW (per output channel), a scalar broadcast over the free
// dim — distinct from spmm_csr_bias_act (per-free-dim/per-column bias, correct
// for a GCN [nodes,features] output). act: 0=identity, 1=relu, 2=sigmoid.
// Structurally-empty output channels (W row o all-zero, e.g. svhn enc1 after a
// global 99% prune) yield Y[o,:] = act(bias[o]) — bias/act still apply.
//
// Reached ONLY via scorch.sparse_linear_fm; scorch.matmul (FEM/GCN) routes to v2
// / spmm_csr_bias_act and never reaches this kernel — the guardrail is the API
// boundary.
// ---------------------------------------------------------------------------

// Numerically stable logistic sigmoid. The naive 1/(1+expf(-x)) overflows expf
// to +inf for large-magnitude x; the build uses -ffast-math, which assumes no
// inf/NaN and turns that into a NaN. Clamping x to [-87,87] keeps expf(-x) finite
// (expf(87) < FLT_MAX) so it never overflows — and the clamp is branchless
// (vminps/vmaxps), so -O3 -ffast-math -march=native vectorizes callers of this
// (the dec2 sigmoid layer) instead of the per-element `x>=0` branch blocking them.
// sigmoid saturates to 0/1 far outside [-87,87], so this matches torch within 1e-3.
static inline float scorch_sigmoidf(float x) {
  x = x < -87.f ? -87.f : x;
  x = x > 87.f ? 87.f : x;
  return 1.f / (1.f + expf(-x));
}

// Scalar activation: 0=identity, 1=relu, 2=sigmoid.
static inline float scorch_act_scalar(float x, int act) {
  if (act == 1) return x > 0.f ? x : 0.f;
  if (act == 2) return scorch_sigmoidf(x);
  return x;
}

// Apply per-output-channel bias `bo` (broadcast over the free dim) then the
// activation to a fully-computed output row of length n. -O3/-ffast-math
// vectorize the identity/relu forms; sigmoid uses expf per element (the same
// work torch's sigmoid does, and only the AE's final layer).
static inline void scorch_apply_row_bias_act(float* SCORCH_RESTRICT C_row,
                                             int n, float bo, int act) {
  if (act == 1) {          // relu
    for (int k = 0; k < n; k++) {
      const float v = C_row[k] + bo;
      C_row[k] = v > 0.f ? v : 0.f;
    }
  } else if (act == 2) {   // sigmoid (scorch_sigmoidf inlines to a vectorizable
    for (int k = 0; k < n; k++) {   // clamp + 1/(1+e^-z); see scorch_sigmoidf)
      C_row[k] = scorch_sigmoidf(C_row[k] + bo);
    }
  } else {                 // identity
    for (int k = 0; k < n; k++) C_row[k] += bo;
  }
}

#if defined(__ARM_NEON)
// The fused Linear path's original NEON row kernel, kept only so the kernel that
// replaced it can be priced against it in one binary. Do not call it in a release
// build; scorch_spmm_row_neon does the same job without the defect below.
//
// The 32-wide strip body is fine and is what scorch_spmm_row_neon_strip does with
// eight accumulators. The tail is not: it walks the row's nonzeros ONCE PER REMAINING
// COLUMN, outside the nonzero loop. So a free dimension under 32 re-walks the row
// B1_size times instead of once -- at B1_size = 8 the row is read eight times, and a
// ragged width like 100 pays four extra full walks for its last four columns. The
// replacement carries the remainder in TAIL scalar accumulators updated in the same
// pass over the row, so every width is one pass.
static inline void scorch_spmm_row_neon_regtile_legacy(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val, int B1_size,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  int k0 = 0;
  for (; k0 + 32 <= B1_size; k0 += 32) {
    float32x4_t c0 = vdupq_n_f32(0.f), c1 = vdupq_n_f32(0.f),
                c2 = vdupq_n_f32(0.f), c3 = vdupq_n_f32(0.f),
                c4 = vdupq_n_f32(0.f), c5 = vdupq_n_f32(0.f),
                c6 = vdupq_n_f32(0.f), c7 = vdupq_n_f32(0.f);
    for (int pA = pA_begin; pA < pA_end; pA++) {
      const float* SCORCH_RESTRICT B = B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
      if (pA + 1 < pA_end)
        __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
      const float32x4_t va = vdupq_n_f32(A_val[pA]);
      c0 = vfmaq_f32(c0, va, vld1q_f32(B));
      c1 = vfmaq_f32(c1, va, vld1q_f32(B + 4));
      c2 = vfmaq_f32(c2, va, vld1q_f32(B + 8));
      c3 = vfmaq_f32(c3, va, vld1q_f32(B + 12));
      c4 = vfmaq_f32(c4, va, vld1q_f32(B + 16));
      c5 = vfmaq_f32(c5, va, vld1q_f32(B + 20));
      c6 = vfmaq_f32(c6, va, vld1q_f32(B + 24));
      c7 = vfmaq_f32(c7, va, vld1q_f32(B + 28));
    }
    vst1q_f32(C_row + k0, c0);      vst1q_f32(C_row + k0 + 4, c1);
    vst1q_f32(C_row + k0 + 8, c2);  vst1q_f32(C_row + k0 + 12, c3);
    vst1q_f32(C_row + k0 + 16, c4); vst1q_f32(C_row + k0 + 20, c5);
    vst1q_f32(C_row + k0 + 24, c6); vst1q_f32(C_row + k0 + 28, c7);
  }
  for (; k0 < B1_size; k0++) {
    float acc = 0.f;
    for (int pA = pA_begin; pA < pA_end; pA++)
      acc += A_val[pA] * B_val[(size_t)A1_crd[pA] * (size_t)B1_size + k0];
    C_row[k0] = acc;
  }
}
#endif

Tensor spmm_csr_linear_fused_float(std::vector<int> result_shape,
                std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, torch::Tensor bias_values, int act,
                int tile_size = 256,
                int nthreads_override = -1, bool atparallel = false) {
  const int C0_size = result_shape[0];
  const int C1_size = result_shape[1];

  const int A0_size = A_shape[0];
  const int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  const int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  const float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  const int B1_size = B_shape[1];
  const float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();
  const float* SCORCH_RESTRICT bias_val = bias_values.data_ptr<float>();

  const size_t C_capacity = (size_t)C0_size * (size_t)C1_size;

  // Output allocation. Every output row in [0, A0_size) is FULLY written by the
  // worker below (empty channels -> act(bias); non-empty -> regtile + epilogue),
  // so no pre-zeroing pass is needed. Allocate via torch::empty (same CPU
  // allocator MKL's output uses -> apples-to-apples). A tail C0_size>A0_size
  // (never happens for a Linear layer, where out==A0_size) is zeroed for safety.
  torch::Tensor C_values_torch = torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  if (C0_size > A0_size)
    memset(C_values + (size_t)A0_size * (size_t)C1_size, 0,
           sizeof(float) * (size_t)(C0_size - A0_size) * (size_t)C1_size);

  const int kTile = (tile_size + 15) & ~15;

  // Thread policy — the same rule as spmm_csr_float_v2, and now literally the same
  // code (scorch_policy.h). It used to be a second copy annotated "IDENTICAL to
  // spmm_csr_float_v2", which is a comment asserting a property nothing enforced.
  const int total_nnz = A1_pos[A0_size];
  const long k_eff = B1_size < 16 ? 16L : (long)B1_size;
  const long work = (long)total_nnz * k_eff;
  const int nthreads = scorch_spmm_nthreads(work, A0_size, nthreads_override,
                                            (long)total_nnz * (long)B1_size,
                                            (long)total_nnz);
  // Deliberately the GENERIC chunk, not the SpMM-specific rule the drop-in kernel
  // uses. The fused kernel's workload is the sparse autoencoder grid, and the chunk
  // rule has never been run against it -- so this is a gap to close with its own
  // measurement, not a decision. Sharing the rule here on the strength of the
  // drop-in kernel's grid would be extending a result past what was measured.
  const int chunk = scorch_chunk(A0_size, work, SCORCH_GRAIN_SPMM);
  std::atomic<int> next_row{0};

  // ROW PARTITION, the same mechanism as spmm_csr_v2_core and for the same reason:
  // one shared counter means the worker->row map is decided by arrival order and
  // differs on every call, so each core re-fetches its slice of A from L3 instead of
  // holding it in L2. A fused Linear layer is the clearest case there is for caring
  // -- the sparse weight matrix is the SAME on every batch of every epoch.
  //
  // Deliberately NOT enabled by default here even if it ships in the drop-in kernel.
  // This kernel's workload is the sparse autoencoder grid, which the partition has
  // not been run against, and turning it on off the back of the SuiteSparse grid is
  // exactly the extension past the evidence that the chunk-rule comment above
  // declines to make. Modes are the same: 0 global counter, 1 home ranges, 2 steal
  // from the front, 3 steal from the back.
  int partition_mode = 0;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_FUSED_PARTITION");
    if (e && *e) { long v = std::atol(e);
      if (v >= 0 && v <= 3) partition_mode = (int)v; } }
#endif
  int nsplit = nthreads > 0 ? nthreads : 1;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: force the number of ranges. More ranges than workers is safe -- the
  // stride loop hands a worker its own extras and the steal loop reaches every range
  // regardless -- so this is the only way to run the heap allocation path, and the
  // very fine and very coarse splits, on a host whose team never gets that wide.
  { const char* e = std::getenv("SCORCH_SPMM_NSPLIT_FORCE");
    if (e && *e) { long v = std::atol(e);
      if (v > 0 && v <= 4096) nsplit = (int)v; } }
#endif
  // 64 cursors of 128 bytes is 8 KB. At 128 the frame was 17440 bytes; halving it
  // costs nothing and keeps the untouched stack reservation off pages the caller
  // might otherwise be using. (It does not remove the entry stack probe -- clang
  // emits that above 4 KB, and 24 cursors is too few for a 32-worker host.) The
  // heap fallback keeps the bound a performance choice and not a correctness one;
  // SCORCH_SPMM_NSPLIT_FORCE is how it gets correctness coverage, since no host
  // here resolves to enough workers to reach it.
  constexpr int kSplitOnStack = 64;
  int split_stack[kSplitOnStack + 1];
  scorch_spmm_cursor cursor_stack[kSplitOnStack];
  std::unique_ptr<int[]> split_heap;
  std::unique_ptr<scorch_spmm_cursor[]> cursor_heap;
  int* row_split = nullptr;
  scorch_spmm_cursor* cursors = nullptr;
  if (partition_mode != 0) {
    if (nsplit <= kSplitOnStack) { row_split = split_stack; cursors = cursor_stack; }
    else {
      split_heap.reset(new int[nsplit + 1]);
      cursor_heap.reset(new scorch_spmm_cursor[nsplit]);
      row_split = split_heap.get();
      cursors = cursor_heap.get();
    }
    row_split[0] = 0;
    row_split[nsplit] = A0_size;
    for (int w = 1; w < nsplit; ++w) {
      const long target = ((long)total_nnz + (long)A0_size) * (long)w / (long)nsplit;
      int lo = row_split[w - 1], hi = A0_size;
      while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if ((long)A1_pos[mid] + (long)mid < target) lo = mid + 1; else hi = mid;
      }
      row_split[w] = lo;
    }
    for (int w = 0; w < nsplit; ++w) {
      if (partition_mode == 3)
        cursors[w].ht.store(scorch_ht_pack(row_split[w], row_split[w + 1]),
                            std::memory_order_relaxed);
      else
        cursors[w].v.store(row_split[w], std::memory_order_relaxed);
    }
  }

#if defined(__AVX2__) && defined(__FMA__)
  const bool narrow_k = (B1_size >= 1 && B1_size <= 32);
  const int nvec = (B1_size + 7) / 8;
  const int mlast = B1_size - 8 * (nvec - 1);
  const bool full_last = (mlast == 8);             // k % 8 == 0 -> no mask needed
  const __m256i mask_last = _mm256_setr_epi32(
      mlast>0?-1:0, mlast>1?-1:0, mlast>2?-1:0, mlast>3?-1:0,
      mlast>4?-1:0, mlast>5?-1:0, mlast>6?-1:0, mlast>7?-1:0);
#else
  const bool narrow_k = false;
#endif

#if defined(__ARM_NEON)
  // NEON register-tiled inner kernel is the ARM default; SCORCH_NEON_REGTILE=0
  // forces the scalar workspace loop (A/B escape hatch, mirrors
  // SCORCH_SPMM_ATPARALLEL). Read once per op.
  bool use_neon_regtile = true;
  if (const char* _nr = std::getenv("SCORCH_NEON_REGTILE"))
    if (*_nr) use_neon_regtile = (std::atol(_nr) != 0);
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: the pre-replacement row kernel, whose tail re-walks the row once per
  // remaining column. The only way to price the replacement without comparing two
  // builds. Compiled out of the shipped .so.
  bool fused_legacy_tail = false;
  if (const char* _lt = std::getenv("SCORCH_FUSED_LEGACY_TAIL"))
    if (*_lt) fused_legacy_tail = (std::atol(_lt) != 0);
#endif
#endif

  scorch_unique_buffer<float> worker_workspaces;
#if !defined(__AVX2__) || !defined(__FMA__)
  worker_workspaces = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(nthreads), static_cast<size_t>(kTile));
#endif

  // Per-worker body: atomic row work-stealing, byte-identical distribution to v2.
  // Computes each output row via the AVX2 regblock/regtile kernels (or the non-
  // AVX2 workspace fallback), then folds bias+act into the SAME parallel region.
  auto worker = [&](int worker_id, int team_size) {
    float* SCORCH_RESTRICT ws = nullptr;
#if !defined(__AVX2__) || !defined(__FMA__)
    ws = worker_workspaces.get() +
        static_cast<size_t>(worker_id) * static_cast<size_t>(kTile);
#endif
    int own_w = worker_id;   // next own split to try  (partition modes 1-3)
    int steal_t = 1;         // next victim offset to try
    while (true) {
      int start = 0, end = 0;
      if (partition_mode == 0) {
        start = next_row.fetch_add(chunk, std::memory_order_relaxed);
        if (start >= A0_size) break;
        end = std::min(start + chunk, A0_size);
      } else {
        bool got = false;
        const int stride = team_size > 0 ? team_size : 1;
        if (partition_mode == 3) {
          while (!got && own_w < nsplit) {
            if (scorch_ht_claim(cursors[own_w].ht, chunk, true, &start, &end))
              got = true;
            else own_w += stride;
          }
          while (!got && steal_t < nsplit) {
            const int w = (worker_id + steal_t) % nsplit;
            if (scorch_ht_claim(cursors[w].ht, chunk, false, &start, &end))
              got = true;
            else steal_t++;
          }
        } else {
          while (!got && own_w < nsplit) {
            const int lim = row_split[own_w + 1];
            const int st = cursors[own_w].v.fetch_add(chunk, std::memory_order_relaxed);
            if (st < lim) { start = st; end = std::min(st + chunk, lim); got = true; }
            else own_w += stride;
          }
          while (!got && partition_mode == 2 && steal_t < nsplit) {
            const int w = (worker_id + steal_t) % nsplit;
            const int lim = row_split[w + 1];
            const int st = cursors[w].v.fetch_add(chunk, std::memory_order_relaxed);
            if (st < lim) { start = st; end = std::min(st + chunk, lim); got = true; }
            else steal_t++;
          }
        }
        if (!got) break;
      }

      for (int i = start; i < end; i++) {
        const int pA_begin = A1_pos[i];
        const int pA_end   = A1_pos[i + 1];
        float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;
        const float bo = bias_val[i];

        // Structurally-empty output channel: Y[o,:] = act(bias[o]) (constant).
        if (pA_begin == pA_end) {
          const float ev = scorch_act_scalar(bo, act);
          for (int k = 0; k < C1_size; k++) C_row[k] = ev;
          continue;
        }

        // Compute the raw SpMM output row into C_row.
#if defined(__AVX2__) && defined(__FMA__)
        if (narrow_k) {
          // The fused Linear runs the identical row kernel, so it inherits the
          // unmasked full-vector dispatch; see the drop-in SpMM above.
          #define SCORCH_RB(NV) \
            (full_last \
               ? scorch_spmm_row_regblock<float, NV, true>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last) \
               : scorch_spmm_row_regblock<float, NV, false>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last))
          switch (nvec) {
            case 1: SCORCH_RB(1); break;
            case 2: SCORCH_RB(2); break;
            case 3: SCORCH_RB(3); break;
            case 4: SCORCH_RB(4); break;
          }
          #undef SCORCH_RB
        } else {
          // Ordinary stores: the fused Linear folds bias and activation into the
          // row epilogue, so it re-reads the row it just wrote and a non-temporal
          // store would force that read back from memory. Whether a fused variant
          // wants streaming is a separate question with a separate grid.
          scorch_spmm_row_regtile<float>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end);
        }
#elif defined(__ARM_NEON)
        if (use_neon_regtile) {
          // The same kernel the drop-in SpMM runs, rather than a second one that
          // happened to live next to this loop: one pass over the row at every width,
          // and the 2-nonzero unroll comes with it.
#ifdef SCORCH_TUNE_HOOKS
          if (fused_legacy_tail) {
            scorch_spmm_row_neon_regtile_legacy(A1_crd, A_val, B_val, B1_size, C_row,
                                                pA_begin, pA_end);
          } else
#endif
          scorch_spmm_row_neon<float>(A1_crd, A_val, B_val, B1_size, C_row,
                                      pA_begin, pA_end);
        } else {
          for (int k_out = 0; k_out < B1_size; k_out += kTile) {
            const int kw = std::min(kTile, B1_size - k_out);
            memset(ws, 0, kw * sizeof(float));
            for (int pA = pA_begin; pA < pA_end; pA++) {
              const int j = A1_crd[pA];
              const float a = A_val[pA];
              const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size + k_out;
              if (pA + 1 < pA_end)
                __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k_out, 0, 1);
              for (int k = 0; k < kw; k++) ws[k] += a * B_row[k];
            }
            memcpy(C_row + k_out, ws, kw * sizeof(float));
          }
        }
#else
        for (int k_out = 0; k_out < B1_size; k_out += kTile) {
          const int kw = std::min(kTile, B1_size - k_out);
          memset(ws, 0, kw * sizeof(float));
          for (int pA = pA_begin; pA < pA_end; pA++) {
            const int j = A1_crd[pA];
            const float a = A_val[pA];
            const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size + k_out;
            if (pA + 1 < pA_end)
              __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k_out, 0, 1);
            for (int k = 0; k < kw; k++) ws[k] += a * B_row[k];
          }
          memcpy(C_row + k_out, ws, kw * sizeof(float));
        }
#endif
        // Fused epilogue: bias + activation, row still hot in cache.
        scorch_apply_row_bias_act(C_row, C1_size, bo, act);
      }
    }
  };

  // Launch. Unlike v2 (which gates at::parallel_for on work>=grain AND a row-
  // starvation floor to protect the SuiteSparse FEM panel — arc130 regresses on
  // torch's pool), the fused Linear kernel is reached ONLY via sparse_linear_fm
  // and NEVER by any FEM/GCN SpMM, so neither guard is needed. Here the whole
  // point is that EVERY layer of a fused autoencoder chain shares ONE warm torch
  // pool — including the tiny near-empty layers (svhn enc1=0nnz / enc2), whose
  // private-libgomp team would otherwise force a mid-chain pool transition into
  // the dec1/dec2 at::parallel_for layers. So when the caller opts into pool-
  // sharing (atparallel + host thread count) we launch on torch's intra-op pool
  // unconditionally; that measured faster on svhn @0.99 than v2's gate. Env
  // SCORCH_SPMM_ATPARALLEL still forces the choice for A/B (1/0).
  bool use_atparallel = atparallel && nthreads_override > 0;
  if (const char* _atpf = std::getenv("SCORCH_SPMM_ATPARALLEL"))
    if (*_atpf) use_atparallel = (std::atol(_atpf) != 0);
  if (use_atparallel) {
    // E-CORE RECRUIT (M5/hybrid-P+E fix). at::parallel_for runs on torch's intra-op
    // pool, whose size is torch's per-platform default thread count. On Apple M-series
    // that default is the P-core count (6) and EXCLUDES the 12 E-cores, so a
    // bandwidth-bound Linear-SpMM (which scales with total memory bandwidth, i.e.
    // with EVERY core) is capped at ~1/3 the machine and runs ~2-2.5x slower than it
    // must (measured across the AE grid @<=0.95). The work-aware `nthreads` already
    // escalates to the full hardware core count (scorch_nthreads caps at
    // omp_get_num_procs()); when it justifies at least 2x the ATen pool we launch our
    // own omp team at that count to pull in the idle E-cores. Gating on 2x-the-pool
    // (a) keeps tiny near-empty layers, whose by-work nthreads stays <= the pool, on
    // the warm at::parallel_for pool (no fork/join over-threading — those regressed
    // ~12% under an unconditional omp team) and (b) makes the recruit fire ONLY on a
    // P-core-only-subset pool: on a non-hybrid / all-physical-cores pool (e.g. x86
    // redwood: pool=24 = every physical core, omp_get_num_procs()=32 counts only SMT
    // siblings) nthreads<=32 < 2*24=48 can never trigger, so the x86 pipeline-pool
    // launch (its measured svhn edge) is byte-unchanged. Env SCORCH_SPMM_ATPARALLEL
    // still forces the choice for A/B (1 => this branch; 0 => the omp branch below).
    const int atpool = at::get_num_threads();
    if (nthreads >= 2 * atpool) {
      #pragma omp parallel num_threads(nthreads)
      {
        worker(omp_get_thread_num(), omp_get_num_threads());
      }
    } else {
      at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {
        for (int64_t w = wbeg; w < wend; ++w)
          worker(static_cast<int>(w), nthreads);
      });
    }
  } else {
    #pragma omp parallel num_threads(nthreads)
    {
      worker(omp_get_thread_num(), omp_get_num_threads());
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

// ---------------------------------------------------------------------------
// scorch_transpose_2d_float — fast cache-blocked 2D transpose of a contiguous
// float32 [R, C] matrix into a contiguous [C, R] matrix.
//
// The drop-in Linear path (scorch.sparse_linear) must materialize the natural
// [batch, in] activation as FEATURE-MAJOR [in, batch] to feed the fused kernel.
// torch's `x.T.contiguous()` (identically `empty + copy_`) does a naive element
// scatter -- for each dst[c,r] = src[r,c] it strides the SOURCE by a full row
// (C floats) -- which runs ~5x below DRAM bandwidth on the i9-14900K (~17 GB/s
// vs ~85 GB/s DDR5): cache-hostile, access-pattern-bound (no thread speedup),
// and once the bias/act epilogue is fused away it is 40-66% of the WHOLE fused
// forward on every autoencoder. This blocks the transpose into cache-resident
// tiles and uses an AVX2 8x8 (NEON 4x4 / scalar) in-register transpose per
// micro-tile, converting the long-stride scatter into blocked sequential access
// to approach bandwidth.
//
// Bit-identical to `x.T.contiguous()` (a transpose is exact -- only the
// traversal order differs). Reached only via the Linear path; never touches the
// FEM/GCN SpMM (the guardrail is the API boundary).
// ---------------------------------------------------------------------------

#if defined(__AVX2__) && defined(__FMA__)
// Transpose one 8x8 float tile: reads 8 rows of src (stride `ss` floats apart),
// writes 8 rows of dst (stride `ds` floats apart). Standard AVX2 unpack/shuffle/
// permute transpose (loadu/storeu -- tiles need not be 32B-aligned).
static inline void scorch_transpose_8x8_avx2(const float* SCORCH_RESTRICT src,
                                             int64_t ss,
                                             float* SCORCH_RESTRICT dst,
                                             int64_t ds) {
  __m256 r0 = _mm256_loadu_ps(src + 0 * ss);
  __m256 r1 = _mm256_loadu_ps(src + 1 * ss);
  __m256 r2 = _mm256_loadu_ps(src + 2 * ss);
  __m256 r3 = _mm256_loadu_ps(src + 3 * ss);
  __m256 r4 = _mm256_loadu_ps(src + 4 * ss);
  __m256 r5 = _mm256_loadu_ps(src + 5 * ss);
  __m256 r6 = _mm256_loadu_ps(src + 6 * ss);
  __m256 r7 = _mm256_loadu_ps(src + 7 * ss);
  __m256 t0 = _mm256_unpacklo_ps(r0, r1);
  __m256 t1 = _mm256_unpackhi_ps(r0, r1);
  __m256 t2 = _mm256_unpacklo_ps(r2, r3);
  __m256 t3 = _mm256_unpackhi_ps(r2, r3);
  __m256 t4 = _mm256_unpacklo_ps(r4, r5);
  __m256 t5 = _mm256_unpackhi_ps(r4, r5);
  __m256 t6 = _mm256_unpacklo_ps(r6, r7);
  __m256 t7 = _mm256_unpackhi_ps(r6, r7);
  __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
  __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
  __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
  __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
  __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  _mm256_storeu_ps(dst + 0 * ds, _mm256_permute2f128_ps(s0, s4, 0x20));
  _mm256_storeu_ps(dst + 1 * ds, _mm256_permute2f128_ps(s1, s5, 0x20));
  _mm256_storeu_ps(dst + 2 * ds, _mm256_permute2f128_ps(s2, s6, 0x20));
  _mm256_storeu_ps(dst + 3 * ds, _mm256_permute2f128_ps(s3, s7, 0x20));
  _mm256_storeu_ps(dst + 4 * ds, _mm256_permute2f128_ps(s0, s4, 0x31));
  _mm256_storeu_ps(dst + 5 * ds, _mm256_permute2f128_ps(s1, s5, 0x31));
  _mm256_storeu_ps(dst + 6 * ds, _mm256_permute2f128_ps(s2, s6, 0x31));
  _mm256_storeu_ps(dst + 7 * ds, _mm256_permute2f128_ps(s3, s7, 0x31));
}
#elif defined(__ARM_NEON)
// Transpose one 4x4 float tile (NEON vtrnq + vcombine). src rows stride `ss`
// floats apart, dst rows stride `ds` floats apart.
static inline void scorch_transpose_4x4_neon(const float* SCORCH_RESTRICT src,
                                             int64_t ss,
                                             float* SCORCH_RESTRICT dst,
                                             int64_t ds) {
  float32x4_t r0 = vld1q_f32(src + 0 * ss);
  float32x4_t r1 = vld1q_f32(src + 1 * ss);
  float32x4_t r2 = vld1q_f32(src + 2 * ss);
  float32x4_t r3 = vld1q_f32(src + 3 * ss);
  float32x4x2_t a = vtrnq_f32(r0, r1);
  float32x4x2_t b = vtrnq_f32(r2, r3);
  vst1q_f32(dst + 0 * ds,
            vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0])));
  vst1q_f32(dst + 1 * ds,
            vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1])));
  vst1q_f32(dst + 2 * ds,
            vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])));
  vst1q_f32(dst + 3 * ds,
            vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1])));
}
#endif

// Transpose the tile rows [rB0,rB1) x cols [cB0,cB1) of a contiguous [R,C] source
// into the [C,R] destination. Uses SIMD micro-tiles where both extents allow,
// scalar for the edges. R/C are the full source dims (= dst/src strides).
static inline void scorch_transpose_block(const float* SCORCH_RESTRICT S,
                                          float* SCORCH_RESTRICT D, int64_t R,
                                          int64_t C, int64_t rB0, int64_t rB1,
                                          int64_t cB0, int64_t cB1) {
  int64_t c = cB0;
#if defined(__AVX2__) && defined(__FMA__)
  for (; c + 8 <= cB1; c += 8) {
    int64_t r = rB0;
    for (; r + 8 <= rB1; r += 8)
      scorch_transpose_8x8_avx2(S + r * C + c, C, D + c * R + r, R);
    for (; r < rB1; ++r)  // row tail for this 8-wide column strip
      for (int64_t cc = c; cc < c + 8; ++cc) D[cc * R + r] = S[r * C + cc];
  }
#elif defined(__ARM_NEON)
  for (; c + 4 <= cB1; c += 4) {
    int64_t r = rB0;
    for (; r + 4 <= rB1; r += 4)
      scorch_transpose_4x4_neon(S + r * C + c, C, D + c * R + r, R);
    for (; r < rB1; ++r)
      for (int64_t cc = c; cc < c + 4; ++cc) D[cc * R + r] = S[r * C + cc];
  }
#endif
  for (; c < cB1; ++c)  // column tail (scalar)
    for (int64_t r = rB0; r < rB1; ++r) D[c * R + r] = S[r * C + c];
}

torch::Tensor scorch_transpose_2d_float(torch::Tensor src,
                                        int nthreads_override = -1) {
  TORCH_CHECK(src.dim() == 2, "scorch_transpose_2d_float expects a 2D tensor");
  src = src.contiguous();
  const int64_t R = src.size(0);
  const int64_t C = src.size(1);
  torch::Tensor dst_t = torch::empty({C, R}, src.options());
  const float* SCORCH_RESTRICT S = src.data_ptr<float>();
  float* SCORCH_RESTRICT D = dst_t.data_ptr<float>();
  if (R == 0 || C == 0) return dst_t;

  // Cache-block edge: a BSxBS float tile (64x64 = 16KB) plus its transpose stays
  // resident in L1/L2 so the scattered writes to D hit warm lines.
  constexpr int64_t BS = 64;
  const int64_t ncblk = (C + BS - 1) / BS;

  // Process a range of column blocks -> writes a contiguous band of dst rows.
  auto do_cblks = [&](int64_t b0, int64_t b1) {
    for (int64_t cb = b0; cb < b1; ++cb) {
      const int64_t cB0 = cb * BS;
      const int64_t cB1 = std::min(cB0 + BS, C);
      for (int64_t rB0 = 0; rB0 < R; rB0 += BS) {
        const int64_t rB1 = std::min(rB0 + BS, R);
        scorch_transpose_block(S, D, R, C, rB0, rB1, cB0, cB1);
      }
    }
  };

  // Access-pattern-bound, so threading mainly helps the wide (stl10) cases;
  // share torch's warm intra-op pool (the SAME pool the fused kernel consuming
  // this output runs on) when the caller passes the host thread count.
  //
  // The grain decides everything below full-size inputs. This used to pass 1, which is
  // ATen for "split whenever there is more than one iteration", so an 8 KiB transpose
  // opened a thread team. Opening one costs ~10 us at four torch threads and ~22 us at
  // eight on an Apple M5, and ~2-3 us at four and ~18-20 us at thirty-two on a 32-core
  // x86 -- against a serial transpose of that same data in 0.3-1.3 us. Nothing was bought
  // with it: below the threshold the serial kernel is not merely adequate, it is 2.0-7.8x
  // faster than the `.contiguous()` scatter it exists to replace, so staying serial gives
  // up nothing at all.
  //
  // SCORCH_TRANSPOSE_PARALLEL_ELEMS is that threshold in elements, converted to blocks
  // because ATen's grain is expressed in units of the iteration space: one column block is
  // R*BS elements, so `parallel_for` splits exactly when the whole transpose exceeds the
  // threshold, and each worker then gets at least that much work.
  //
  // The threshold has to clear one bar in particular. This kernel is already shipped and
  // `fast_transpose` / `sparse_linear` call it with the host thread count, so every shape
  // with more than one column block ran threaded before. Raising the grain can only move a
  // shape from threaded to serial, so any shape it moves the wrong way is a regression in a
  // path that has been measured. Candidate rules were therefore replayed against both
  // modes' measured times over a 40-shape grid (R in {8,32,64,256,784}, C in {64..20000})
  // on two hosts at two thread counts each -- 160 cells -- and scored on how many shapes
  // they made slower than always-threaded.
  //
  // `max(32768, 4096 * threads)` is the rule that regresses nothing on any of the 160
  // cells while improving 15-21 shapes per grid, geomean 1.37-3.13x and up to 80x on the
  // small ones. The floor is the work below which no pool size is worth a launch; above it
  // the threshold grows with the pool because the launch cost does. Fixed thresholds tuned
  // for the best average instead (65536, 131072) do buy a little more -- three or four extra
  // shapes per grid -- and pay for it by making two to five shapes up to 1.8x slower than
  // what already ships, which is not a trade this gets to make.
  //
  // What this does not do is pick the better mode everywhere. On the M5 the worst shape
  // lands within a few percent of `.contiguous()` at four threads and about 1.6x below it at
  // eight, where a higher threshold would have fixed both. Element count alone cannot
  // separate these cases -- 784x256 and 256x784 are the same 200704 elements and want
  // opposite modes, because the block geometry differs -- and the two hosts want different
  // constants anyway: per thread, opening the team costs ~2.7 us on the M5 against ~0.6 us
  // on the x86. Suiting both would mean calibrating that cost once per process. Until then
  // the rest is left on the table rather than bought with a regression.
  const int64_t SCORCH_TRANSPOSE_PARALLEL_ELEMS =
      std::max<int64_t>(32768, 4096 * (int64_t)at::get_num_threads());
  const int64_t grain_blocks = std::max<int64_t>(
      1, SCORCH_TRANSPOSE_PARALLEL_ELEMS / std::max<int64_t>(1, R * BS));
  if (nthreads_override > 0) {
    at::parallel_for(0, ncblk, grain_blocks,
                     [&](int64_t b0, int64_t b1) { do_cblks(b0, b1); });
  } else {
    do_cblks(0, ncblk);
  }
  return dst_t;
}

// scorch_sparse_softmax_csr_float — row-wise softmax over the nonzeros of a CSR
// value array, with the attention `scale` folded in (softmax over `scale * v`).
//
// This is the "lever 1" replacement for the sparse-attention softmax that the
// transformer bench previously did in torch as a scatter chain
// (repeat_interleave -> scatter_reduce(amax) -> exp -> scatter_add -> divide),
// run once PER HEAD. That chain is random-access (scatter) and ~6-16x above the
// memory-traffic floor; profiled at ~59% of the whole Scorch attention layer.
//
// Here each row's nonzeros live in a CONTIGUOUS CSR span [crow[i], crow[i+1]),
// so the softmax is three sequential passes over that span (max, exp+sum,
// normalize) with NO scatter and NO intermediate nnz-sized allocations. Rows are
// independent -> parallel over rows on torch's warm intra-op pool (same pool the
// surrounding SpMM/SDDMM share) when the caller passes the host thread count.
//
// Numerics match the torch reference bit-for-bit up to float rounding: subtract
// the row max for stability, exponentiate, divide by the row sum. The row-max is
// seeded from the first element (NOT -INFINITY) so the code is safe under the
// build's -ffast-math (no reliance on inf semantics). Empty rows are left zero.
torch::Tensor scorch_sparse_softmax_csr_float(torch::Tensor crow_indices,
                                              torch::Tensor values,
                                              double scale = 1.0,
                                              int nthreads_override = -1) {
  TORCH_CHECK(crow_indices.dim() == 1, "crow_indices must be 1-D");
  TORCH_CHECK(values.dim() == 1, "values must be 1-D");
  auto crow = crow_indices.to(torch::kInt64).contiguous();
  values = values.contiguous();
  const int64_t nrows = crow.size(0) - 1;
  torch::Tensor out = torch::empty_like(values);
  if (nrows <= 0) return out;

  const int64_t* SCORCH_RESTRICT rp = crow.data_ptr<int64_t>();
  const float* SCORCH_RESTRICT v = values.data_ptr<float>();
  float* SCORCH_RESTRICT o = out.data_ptr<float>();
  const float sc = (float)scale;

  auto do_rows = [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) {
      const int64_t s = rp[i], e = rp[i + 1];
      if (s >= e) continue;
      float m = v[s] * sc;                       // seed from first (ffast-math safe)
      for (int64_t j = s + 1; j < e; ++j) {
        const float x = v[j] * sc;
        if (x > m) m = x;
      }
      float sum = 0.0f;
      for (int64_t j = s; j < e; ++j) {
        const float ex = std::exp(v[j] * sc - m);
        o[j] = ex;
        sum += ex;
      }
      const float inv = 1.0f / sum;
      for (int64_t j = s; j < e; ++j) o[j] *= inv;
    }
  };

  // Parallel over rows only when the caller opts in (host thread count) and there
  // is more than one row; grain of 128 rows keeps per-task overhead negligible.
  if (nthreads_override > 0 && nrows > 1) {
    at::parallel_for(0, nrows, 128,
                     [&](int64_t r0, int64_t r1) { do_rows(r0, r1); });
  } else {
    do_rows(0, nrows);
  }
  return out;
}

// scorch_sparse_attention_csr_float — fused sparse (masked) multi-head attention
// over a shared CSR mask. The "lever 2" ceiling for the sparse-attention bench:
// one native pass computes the whole attention output, replacing the per-head
// three-kernel chain (SDDMM -> softmax -> SpMM) driven by a Python H x L loop with
// CSR round-tripping and per-head .contiguous() copies between stages.
//
// Layout: Q/K/V are dense [S, H, D] row-major (Q[i,h,d] at (i*H + h)*D + d) —
// taken directly from `q_proj(x).view(S, H, D)`, with NO per-head slice/copy. The
// mask is the CSR structure (crow[S+1], col[nnz]) passed ONCE; its stored values
// are the 0/1 attend pattern, so structural presence == attend and we compute the
// raw scaled dot for each nonzero (no per-value multiply). Output is dense
// [S, H, D] (feeds straight into the out-projection after a reshape to [S, H*D]).
//
// Per row i (the parallel unit) and head h, a two-pass softmax over the row's
// nonzero columns j in [crow[i], crow[i+1]) — rows are tiny (~2W+1 for the window,
// S for the few global rows) so a row stays L1/L2-resident:
//   pass 1: score[jj] = scale * dot(Q[i,h], K[j,h]); track the row max m.
//   pass 2: w = exp(score[jj] - m); l += w; acc[d] += w * V[j,h,d].
//   out[i,h,d] = acc[d] / l.
// This is exactly softmax(scale * Q[i,h].K[j,h]) . V over the attended j, i.e. the
// masked-attention math the dense path does with -inf fills — identical up to
// float rounding. The row max is seeded from the first nonzero (NOT -INFINITY) so
// the code is safe under the build's -ffast-math (matches the lever-1 softmax).
//
// One thread-local scratch (score buffer sized to the widest row + a D-wide V
// accumulator) is allocated once per parallel task, so there are NO nnz-sized
// intermediates and NO heap churn per row/head. Rows are independent -> parallel
// over rows on torch's warm intra-op pool (the same pool the projections run on)
// when the caller passes the host thread count.
torch::Tensor scorch_sparse_attention_csr_float(torch::Tensor crow_indices,
                                                torch::Tensor col_indices,
                                                torch::Tensor Q, torch::Tensor K,
                                                torch::Tensor V,
                                                double scale = 1.0,
                                                int nthreads_override = -1) {
  TORCH_CHECK(crow_indices.dim() == 1, "crow_indices must be 1-D");
  TORCH_CHECK(col_indices.dim() == 1, "col_indices must be 1-D");
  TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3,
              "Q/K/V must be [S, H, D]");

  auto crow = crow_indices.to(torch::kInt64).contiguous();
  auto col = col_indices.to(torch::kInt64).contiguous();
  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  const int64_t S = Q.size(0);
  const int64_t H = Q.size(1);
  const int64_t D = Q.size(2);
  TORCH_CHECK(crow.size(0) == S + 1, "crow length must be S+1");
  TORCH_CHECK(K.size(0) == S && V.size(0) == S, "K/V must have S rows");
  TORCH_CHECK(K.size(1) == H && K.size(2) == D && V.size(1) == H &&
                  V.size(2) == D,
              "K/V head/dim must match Q");

  torch::Tensor out = torch::empty({S, H, D}, Q.options());
  if (S == 0) return out;

  const int64_t* SCORCH_RESTRICT rp = crow.data_ptr<int64_t>();
  const int64_t* SCORCH_RESTRICT cp = col.data_ptr<int64_t>();
  const float* SCORCH_RESTRICT Qp = Q.data_ptr<float>();
  const float* SCORCH_RESTRICT Kp = K.data_ptr<float>();
  const float* SCORCH_RESTRICT Vp = V.data_ptr<float>();
  float* SCORCH_RESTRICT Op = out.data_ptr<float>();
  const float sc = (float)scale;
  const int64_t HD = H * D;

  // Widest row -> score-buffer size. The few global rows attend to all S columns,
  // so the buffer must cover S; window rows use only their prefix. One O(S) scan.
  int64_t max_len = 0;
  for (int64_t i = 0; i < S; ++i) {
    const int64_t len = rp[i + 1] - rp[i];
    if (len > max_len) max_len = len;
  }

  TORCH_CHECK(D <= std::numeric_limits<int64_t>::max() - max_len,
              "sparse-attention scratch size overflow");
  const size_t scratch_elements = static_cast<size_t>(max_len + D);
  const size_t scratch_stride = (scratch_elements + 15) & ~size_t{15};
  const int scratch_workers = std::max(1, at::get_num_threads());
  auto scratch_by_worker = scorch_make_aligned_buffer_pool<float>(
      static_cast<size_t>(scratch_workers), scratch_stride);

  auto do_rows = [&](int64_t r0, int64_t r1, int worker_id) {
    // One serially allocated slice per pool worker: [scores | D-wide V accum].
    float* SCORCH_RESTRICT scratch =
        scratch_by_worker.get() + static_cast<size_t>(worker_id) * scratch_stride;
    float* SCORCH_RESTRICT scores = scratch;
    float* SCORCH_RESTRICT acc = scratch + max_len;
    for (int64_t i = r0; i < r1; ++i) {
      const int64_t s = rp[i], e = rp[i + 1];
      float* SCORCH_RESTRICT out_i = Op + i * HD;
      if (s >= e) {  // empty row (never happens for this mask; kept safe)
        for (int64_t t = 0; t < HD; ++t) out_i[t] = 0.0f;
        continue;
      }
      const int64_t len = e - s;
      for (int64_t h = 0; h < H; ++h) {
        const float* SCORCH_RESTRICT q = Qp + i * HD + h * D;

        // Pass 1: scaled Q.K scores over the row's nonzeros + running max. Seed
        // the max from the first nonzero (ffast-math safe, matches lever-1).
        float m;
        {
          const float* SCORCH_RESTRICT k0 = Kp + cp[s] * HD + h * D;
          float dot0 = 0.0f;
          for (int64_t d = 0; d < D; ++d) dot0 += q[d] * k0[d];
          scores[0] = dot0 * sc;
          m = scores[0];
        }
        for (int64_t jj = 1; jj < len; ++jj) {
          const float* SCORCH_RESTRICT krow = Kp + cp[s + jj] * HD + h * D;
          if (jj + 1 < len) {
            __builtin_prefetch(Kp + cp[s + jj + 1] * HD + h * D, 0, 1);
          }
          float dot = 0.0f;
          for (int64_t d = 0; d < D; ++d) dot += q[d] * krow[d];
          const float sco = dot * sc;
          scores[jj] = sco;
          if (sco > m) m = sco;
        }

        // Pass 2: exp(score - max), row sum, and weighted-V accumulation.
        for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        float l = 0.0f;
        for (int64_t jj = 0; jj < len; ++jj) {
          const float* SCORCH_RESTRICT vrow = Vp + cp[s + jj] * HD + h * D;
          if (jj + 1 < len) {
            __builtin_prefetch(Vp + cp[s + jj + 1] * HD + h * D, 0, 1);
          }
          const float w = std::exp(scores[jj] - m);
          l += w;
          for (int64_t d = 0; d < D; ++d) acc[d] += w * vrow[d];
        }
        const float inv = 1.0f / l;
        float* SCORCH_RESTRICT out_ih = out_i + h * D;
        for (int64_t d = 0; d < D; ++d) out_ih[d] = acc[d] * inv;
      }
    }
  };

  // Parallel over rows on torch's warm intra-op pool when the caller opts in
  // (host thread count) and there is more than one row. Small grain: per-row work
  // is heavy (H * len * D), so a few rows per task already amortizes task overhead
  // and keeps the lone heavy global row from stalling the join barrier.
  if (nthreads_override > 0 && S > 1) {
    at::parallel_for(0, S, 8, [&](int64_t r0, int64_t r1) {
      do_rows(r0, r1, at::get_thread_num());
    });
  } else {
    do_rows(0, S, 0);
  }
  return out;
}

// Variant 9: Large-Tile NEON with Direct Accumulation
// Tile k at 128 (outer loop) so B working set per tile fits in L2 (~2.5MB for n=5000).
// Uses NEON inner loop, dynamic scheduling, and direct C accumulation (no workspace copy).
// Re-traverses sparse structure 16x for k=2048 but all B loads are L2 hits.
Tensor spmm_csr_float_tiled_neon(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  size_t C_capacity = (size_t)C0_size * (size_t)C1_size;
  torch::Tensor C_values_torch =
      torch::empty({(long long)C_capacity}, torch::kFloat32);
  float* SCORCH_RESTRICT C_values = C_values_torch.data_ptr<float>();
  memset(C_values, 0, sizeof(float) * C_capacity);

  constexpr int kTile = 128;

  // Tile k (output columns) in the outer loop
  for (int k_out = 0; k_out < B1_size; k_out += kTile) {
    int k_width = std::min(kTile, B1_size - k_out);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < A0_size; i++) {
      int pA1_begin = A1_pos[i];
      int pA1_end = A1_pos[i + 1];
      if (pA1_begin >= pA1_end) continue;

      float* SCORCH_RESTRICT C_ptr = C_values + (size_t)i * (size_t)C1_size + k_out;

#ifdef __ARM_NEON
      for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
        int j = A1_crd[pA1];
        float a_val = A_val[pA1];
        const float* SCORCH_RESTRICT B_ptr = B_val + (size_t)j * (size_t)B1_size + k_out;

        if (pA1 + 1 < pA1_end) {
          __builtin_prefetch(B_val + (size_t)A1_crd[pA1 + 1] * (size_t)B1_size + k_out, 0, 1);
        }

        float32x4_t va = vdupq_n_f32(a_val);
        int k = 0;
        for (; k + 15 < k_width; k += 16) {
          float32x4_t c0 = vld1q_f32(C_ptr + k);
          float32x4_t c1 = vld1q_f32(C_ptr + k + 4);
          float32x4_t c2 = vld1q_f32(C_ptr + k + 8);
          float32x4_t c3 = vld1q_f32(C_ptr + k + 12);
          c0 = vfmaq_f32(c0, va, vld1q_f32(B_ptr + k));
          c1 = vfmaq_f32(c1, va, vld1q_f32(B_ptr + k + 4));
          c2 = vfmaq_f32(c2, va, vld1q_f32(B_ptr + k + 8));
          c3 = vfmaq_f32(c3, va, vld1q_f32(B_ptr + k + 12));
          vst1q_f32(C_ptr + k, c0);
          vst1q_f32(C_ptr + k + 4, c1);
          vst1q_f32(C_ptr + k + 8, c2);
          vst1q_f32(C_ptr + k + 12, c3);
        }
        for (; k + 3 < k_width; k += 4) {
          float32x4_t c = vld1q_f32(C_ptr + k);
          c = vfmaq_f32(c, va, vld1q_f32(B_ptr + k));
          vst1q_f32(C_ptr + k, c);
        }
        for (; k < k_width; k++) {
          C_ptr[k] += a_val * B_ptr[k];
        }
      }
#else
      for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
        int j = A1_crd[pA1];
        float a_val = A_val[pA1];
        const float* SCORCH_RESTRICT B_ptr = B_val + (size_t)j * (size_t)B1_size + k_out;
        for (int k = 0; k < k_width; k++) {
          C_ptr[k] += a_val * B_ptr[k];
        }
      }
#endif
    }
  }

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}
