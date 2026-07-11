#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "prebuilt_types.h"
#include "scorch_policy.h"  // shared scorch_nthreads / scorch_chunk (see header.cpp)
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
  scalar_t* SCORCH_RESTRICT C_values =
      (scalar_t *)malloc(sizeof(scalar_t) * C_capacity);

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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, scorch_torch_dtype<scalar_t>());
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}

template <typename scalar_t>
Tensor spmm_csr_typed(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 0) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  scalar_t* SCORCH_RESTRICT A_val = A_values.data_ptr<scalar_t>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  scalar_t* SCORCH_RESTRICT B_val = B_values.data_ptr<scalar_t>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  scalar_t* SCORCH_RESTRICT C_values =
      (scalar_t *)malloc(sizeof(scalar_t) * C_capacity);
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

  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, scorch_torch_dtype<scalar_t>());
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* C_values = (float*) malloc(sizeof(float) * C_capacity);
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
      // Initialize workspaces
      // float* wksp = new float[kTile_k]();

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

      // delete[] wksp;
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
        float* wksp = new float[1]();

        for (int pA1 = pA0; pA1 < pA1_end; pA1++) {
          // Resolve coordinates
          int j = A1_crd[pA1];

          // Resolve dense coordinates
          int pB0 = j;
          int pB1 = pB0 * B1_size + k;
          wksp[0] += A_val[pA1] * B_val[pB1];
        }

        // Lower consumer CIN
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += wksp[0];
      }
    }
  }
  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void* ptr) {{ free(ptr); }};
  torch::Tensor C_values_torch = torch::from_blob(C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Use the tile size parameter with a reasonable default
  int kTile_k = tile_size;

  // Compute how many tiles we need
  int num_tiles = (B1_size + kTile_k - 1) / kTile_k;

  #pragma omp parallel
  {
    // Pre-allocate thread-local workspace to avoid repeated allocations
    // This is a significant optimization - only allocate once per thread
    float* thread_workspace = new float[kTile_k]();

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

    // Clean up thread-local storage
    delete[] thread_workspace;
  }

  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < A0_size; i++) {
    int pC0 = i;
    int pA1_begin = A1_pos[i];
    int pA1_end = A1_pos[i + 1];
    int nnz_in_row = pA1_end - pA1_begin;

    // Skip rows with no non-zeros
    if (SCORCH_UNLIKELY(nnz_in_row == 0)) continue;

    // Allocate thread-local workspace once per row
    // This avoids repeated allocation/deallocation inside the tile loop
    float* accum_c = new float[kTile_k]();

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

    // Free thread-local workspace
    delete[] accum_c;
  }

  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
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

  float* SCORCH_RESTRICT C_values = (float *)malloc(allocation_size);
  if (!C_values) {
    std::stringstream ss;
    ss << "Failed to allocate " << (allocation_size / (1024.0 * 1024.0))
       << " MB for dense output matrix (" << C0_size << " x " << C1_size << ")";
    throw std::runtime_error(ss.str());
  }
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

  // Process all rows in a single parallel region with dynamic scheduling
  #pragma omp parallel
  {
    // Ensure allocation size is a multiple of the alignment (64 bytes)
    // Each float is 4 bytes, so we need to align to 16 floats (64/4)
    size_t aligned_tile_size = ((default_tile_size + 15) / 16) * 16;
    size_t aligned_bytes = aligned_tile_size * sizeof(float);

    // Each thread allocates its own workspace aligned to cache line (64 bytes)
    float* SCORCH_RESTRICT thread_workspace = nullptr;

    #if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
    // Use posix_memalign which has better error handling
    if (posix_memalign((void**)&thread_workspace, 64, aligned_bytes) != 0) {
      thread_workspace = nullptr; // Ensure it's null on failure
    }
    #else
    // Fallback to aligned_alloc
    thread_workspace = (float*)aligned_alloc(64, aligned_bytes);
    #endif

    // Check if allocation succeeded
    if (thread_workspace == nullptr) {
      // Handle allocation failure gracefully
      #pragma omp critical
      {
        fprintf(stderr, "Failed to allocate thread workspace memory\n");
      }
      // Skip computation in this thread, others can continue
    } else {
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

      // Clean up thread-local workspace safely
      #if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
      free(thread_workspace);
      #else
      free(thread_workspace);
      #endif
    }
  }

  // Restore default thread count
  omp_set_num_threads(omp_get_max_threads());

  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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

  // Process all rows in a single parallel region with dynamic scheduling
  #pragma omp parallel
  {
    // Each thread allocates its own workspace aligned to cache line (64 bytes)
    float* SCORCH_RESTRICT thread_workspace =
        (float*)aligned_alloc(64, sizeof(float) * default_tile_size);

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

    // Clean up thread-local workspace
    free(thread_workspace);
  }

  // Restore default thread count
  omp_set_num_threads(omp_get_max_threads());

  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Use the tile size parameters
  int kTile_i = i_tile_size;
  int kTile_k = k_tile_size;

  int num_i_tiles = (A0_size + kTile_i - 1) / kTile_i;
  int residual_k_start = (B1_size / kTile_k) * kTile_k;

  #pragma omp parallel for
  for (int i_tile = 0; i_tile < num_i_tiles; i_tile++) {
    // Calculate the start and end of this i-tile
    int i_start = i_tile * kTile_i;
    int i_end = std::min(i_start + kTile_i, A0_size);

    for (int k_out = 0; k_out < residual_k_start; k_out += kTile_k) {
      // For each i-tile and k-tile, process the computation

      for (int i = i_start; i < i_end; i++) {
        // Resolve index into dense level of values array
        int pC0 = i;

        // Initialize workspaces
        float *accum_c = new float[kTile_k]();

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

        delete[] accum_c;
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

      for (int i = i_start; i < i_end; i++) {
        int pC0 = i;

        float *accum_c = new float[tile_k_width]();
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
        delete[] accum_c;
      }
    }
  }
  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
  memset(C_values, 0, sizeof(float) * C_capacity);

  constexpr int PANEL_SIZE = 8;
  int num_panels = (A0_size + PANEL_SIZE - 1) / PANEL_SIZE;

  struct PanelEntry {
    int col;
    int row;   // row index within panel (0 to PANEL_SIZE-1)
    float val;
  };

  #pragma omp parallel
  {
    // Thread-local entry buffer to avoid repeated allocation
    std::vector<PanelEntry> entries;

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

      entries.clear();
      entries.reserve(total_nnz);

      for (int r = 0; r < panel_rows; r++) {
        int i = i_start + r;
        for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
          entries.push_back({A1_crd[pA1], r, A_val[pA1]});
        }
      }

      // Sort by column for B-row reuse
      std::sort(entries.begin(), entries.end(),
                [](const PanelEntry& a, const PanelEntry& b) {
                  return a.col < b.col;
                });

      // Process sorted entries - B row stays in cache across entries with same col
      int idx = 0;
      int num_entries = (int)entries.size();
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}


// ---------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
// Register-blocked narrow-k SpMM row kernel: accumulate the whole output row in
// NVEC YMM registers across the row's nonzeros (2-nnz ILP for two independent
// FMA chains), with a masked load/store on the final partial vector. Avoids the
// workspace round-trip (memset + per-nnz ws load/store + memcpy) that dominated
// MKL for narrow k — the 0.5-0.8x gap at k<=16, and the k=32 dense case.
// NVEC = ceil(k/8), specialized 1..4 (k<=32).
template <int NVEC>
static inline void scorch_spmm_row_regblock(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val, int B1_size,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end, __m256i mask_last) {
  __m256 acc0[NVEC], acc1[NVEC];
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    acc0[v] = _mm256_setzero_ps();
    acc1[v] = _mm256_setzero_ps();
  }
  int pA = pA_begin;
  for (; pA + 1 < pA_end; pA += 2) {
    const float* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const float* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size;
    if (pA + 2 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + 2] * (size_t)B1_size, 0, 1);
    const __m256 a0 = _mm256_set1_ps(A_val[pA]);
    const __m256 a1 = _mm256_set1_ps(A_val[pA + 1]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const __m256 b0 = (v == NVEC - 1) ? _mm256_maskload_ps(B0 + 8 * v, mask_last)
                                        : _mm256_loadu_ps(B0 + 8 * v);
      const __m256 b1 = (v == NVEC - 1) ? _mm256_maskload_ps(B1 + 8 * v, mask_last)
                                        : _mm256_loadu_ps(B1 + 8 * v);
      acc0[v] = _mm256_fmadd_ps(a0, b0, acc0[v]);
      acc1[v] = _mm256_fmadd_ps(a1, b1, acc1[v]);
    }
  }
  if (pA < pA_end) {
    const float* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA] * (size_t)B1_size;
    const __m256 a0 = _mm256_set1_ps(A_val[pA]);
    #pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const __m256 b0 = (v == NVEC - 1) ? _mm256_maskload_ps(B0 + 8 * v, mask_last)
                                        : _mm256_loadu_ps(B0 + 8 * v);
      acc0[v] = _mm256_fmadd_ps(a0, b0, acc0[v]);
    }
  }
  #pragma unroll
  for (int v = 0; v < NVEC; v++) {
    const __m256 r = _mm256_add_ps(acc0[v], acc1[v]);
    if (v == NVEC - 1) _mm256_maskstore_ps(C_row + 8 * v, mask_last, r);
    else _mm256_storeu_ps(C_row + 8 * v, r);
  }
}

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
template <int NV>
static inline void scorch_spmm_row_regtile_partial(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val, int B1_size,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end,
    int k0, __m256i mask, bool full) {
  __m256 acc0[NV], acc1[NV];
  #pragma unroll
  for (int v = 0; v < NV; v++) { acc0[v] = _mm256_setzero_ps(); acc1[v] = _mm256_setzero_ps(); }
  int pA = pA_begin;
  for (; pA + 1 < pA_end; pA += 2) {                        // 2-nnz ILP: 2*NV chains
    const float* SCORCH_RESTRICT B0 = B_val + (size_t)A1_crd[pA]     * (size_t)B1_size + k0;
    const float* SCORCH_RESTRICT B1 = B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0;
    if (pA + 2 < pA_end)
      __builtin_prefetch(B_val + (size_t)A1_crd[pA + 2] * (size_t)B1_size + k0, 0, 1);
    const __m256 a0 = _mm256_set1_ps(A_val[pA]);
    const __m256 a1 = _mm256_set1_ps(A_val[pA + 1]);
    #pragma unroll
    for (int v = 0; v < NV; v++) {
      const bool m = (v == NV - 1 && !full);
      const __m256 b0 = m ? _mm256_maskload_ps(B0 + 8 * v, mask) : _mm256_loadu_ps(B0 + 8 * v);
      const __m256 b1 = m ? _mm256_maskload_ps(B1 + 8 * v, mask) : _mm256_loadu_ps(B1 + 8 * v);
      acc0[v] = _mm256_fmadd_ps(a0, b0, acc0[v]);
      acc1[v] = _mm256_fmadd_ps(a1, b1, acc1[v]);
    }
  }
  for (; pA < pA_end; pA++) {                                // odd tail nnz
    const float* SCORCH_RESTRICT Bp = B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
    const __m256 a = _mm256_set1_ps(A_val[pA]);
    #pragma unroll
    for (int v = 0; v < NV; v++) {
      const bool m = (v == NV - 1 && !full);
      const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask) : _mm256_loadu_ps(Bp + 8 * v);
      acc0[v] = _mm256_fmadd_ps(a, b, acc0[v]);
    }
  }
  #pragma unroll
  for (int v = 0; v < NV; v++) {
    const __m256 r = _mm256_add_ps(acc0[v], acc1[v]);
    if (v == NV - 1 && !full) _mm256_maskstore_ps(C_row + k0 + 8 * v, mask, r);
    else _mm256_storeu_ps(C_row + k0 + 8 * v, r);
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
static inline void scorch_spmm_row_regtile(
    const int* SCORCH_RESTRICT A1_crd, const float* SCORCH_RESTRICT A_val,
    const float* SCORCH_RESTRICT B_val, int B1_size,
    float* SCORCH_RESTRICT C_row, int pA_begin, int pA_end) {
  int k0 = 0;
  for (; k0 + 64 <= B1_size; k0 += 64) {
    __m256 acc[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) acc[v] = _mm256_setzero_ps();
    for (int pA = pA_begin; pA < pA_end; pA++) {
      const float* SCORCH_RESTRICT Bp =
          B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
      if (pA + 1 < pA_end)
        __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
      const __m256 a = _mm256_set1_ps(A_val[pA]);
      #pragma unroll
      for (int v = 0; v < 8; v++)
        acc[v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(Bp + 8 * v), acc[v]);
    }
    #pragma unroll
    for (int v = 0; v < 8; v++) _mm256_storeu_ps(C_row + k0 + 8 * v, acc[v]);
  }
  // Final partial tile (kw in 1..63): the templated compile-time-nv path (2-nnz
  // ILP). A scalar tail here reintroduced the per-nnz round-trip (-0.5-0.9x at
  // k=48/96); the earlier runtime-nv YMM loop was correct but front-end-bound.
  const int kw = B1_size - k0;
  if (kw > 0) {
    const int nv = (kw + 7) / 8;              // 1..8 vectors
    const int ml = kw - 8 * (nv - 1);         // 1..8 valid lanes in last vector
    const bool full = (ml == 8);              // complete last vector -> no mask
    const __m256i mask = _mm256_setr_epi32(
        ml > 0 ? -1 : 0, ml > 1 ? -1 : 0, ml > 2 ? -1 : 0, ml > 3 ? -1 : 0,
        ml > 4 ? -1 : 0, ml > 5 ? -1 : 0, ml > 6 ? -1 : 0, ml > 7 ? -1 : 0);
#ifdef SCORCH_TUNE_HOOKS
    // A/B hook: SCORCH_REGTILE_BASE=1 forces the legacy runtime-nv partial path
    // (single-nnz, no compile-time unroll) for an in-process old-vs-new delta.
    // Read once per SpMM op into g_scorch_regtile_base (below) — a per-row getenv
    // would swamp the hot loop; a cached function-local static would latch the
    // FIRST op's value and ignore later toggles. Compiled out of the shipped .so.
    if (g_scorch_regtile_base) {
      __m256 acc[8];
      for (int v = 0; v < nv; v++) acc[v] = _mm256_setzero_ps();
      for (int pA = pA_begin; pA < pA_end; pA++) {
        const float* SCORCH_RESTRICT Bp =
            B_val + (size_t)A1_crd[pA] * (size_t)B1_size + k0;
        if (pA + 1 < pA_end)
          __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k0, 0, 1);
        const __m256 a = _mm256_set1_ps(A_val[pA]);
        for (int v = 0; v < nv; v++) {
          const __m256 b = (v == nv - 1) ? _mm256_maskload_ps(Bp + 8 * v, mask)
                                         : _mm256_loadu_ps(Bp + 8 * v);
          acc[v] = _mm256_fmadd_ps(a, b, acc[v]);
        }
      }
      for (int v = 0; v < nv; v++) {
        if (v == nv - 1) _mm256_maskstore_ps(C_row + k0 + 8 * v, mask, acc[v]);
        else _mm256_storeu_ps(C_row + k0 + 8 * v, acc[v]);
      }
      return;
    }
#endif
    switch (nv) {
      case 1: scorch_spmm_row_regtile_partial<1>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 2: scorch_spmm_row_regtile_partial<2>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 3: scorch_spmm_row_regtile_partial<3>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 4: scorch_spmm_row_regtile_partial<4>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 5: scorch_spmm_row_regtile_partial<5>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 6: scorch_spmm_row_regtile_partial<6>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 7: scorch_spmm_row_regtile_partial<7>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
      case 8: scorch_spmm_row_regtile_partial<8>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, k0, mask, full); break;
    }
  }
}
#endif

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

Tensor spmm_csr_float_v2(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 256,
                int nthreads_override = -1, bool atparallel = false) {
  const int C0_size = result_shape[0];
  const int C1_size = result_shape[1];

  const int A0_size = A_shape[0];
  const int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  const int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  const float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  const int B1_size = B_shape[1];
  const float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  const size_t C_capacity = (size_t)C0_size * (size_t)C1_size;

  // Output allocation + zeroing. Every code path below (regblock / regtile /
  // workspace memcpy) ASSIGNS all C1_size entries of each NON-empty output row,
  // so only structurally-empty rows (and any tail rows C0_size>A0_size) need
  // pre-zeroing. The legacy path malloc'd a raw buffer and memset the WHOLE
  // thing single-threaded before the parallel region — a serial O(C0*C1)
  // zero-fault per call (14MB for a 14K-row k=256 product) that MKL never pays
  // (its dense output is written in full by the kernel, from a pooled buffer).
  // We (a) allocate via torch::empty (the same CPU allocator MKL's output uses,
  // so scorch and MKL are apples-to-apples on allocation) and (b) zero only the
  // rows the kernel won't touch. FEM/graph adjacencies have no empty rows ->
  // the zeroing is a single O(rows) index scan. Correctness is identical: a
  // non-empty row is fully overwritten by the kernel; an empty row is zeroed
  // here.
  torch::Tensor C_values_torch;
  float* SCORCH_RESTRICT C_values;
  bool torch_alloc = true, zero_empty_only = true;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: SCORCH_SPMM_ALLOC bit0=torch_alloc, bit1=zero_empty_only.
  // 0 = malloc + full memset (legacy); 3 = torch::empty + empty-only (default).
  { const char* e = std::getenv("SCORCH_SPMM_ALLOC");
    if (e && *e) { long v = std::atol(e); torch_alloc = v & 1; zero_empty_only = v & 2; } }
#endif
  if (torch_alloc) {
    C_values_torch = torch::empty({(long long)C_capacity}, torch::kFloat32);
    C_values = C_values_torch.data_ptr<float>();
  } else {
    C_values = (float*)malloc(sizeof(float) * C_capacity);
  }
  if (zero_empty_only) {
    for (int i = 0; i < A0_size; i++)
      if (A1_pos[i] == A1_pos[i + 1])
        memset(C_values + (size_t)i * (size_t)C1_size, 0, sizeof(float) * (size_t)C1_size);
    if (C0_size > A0_size)
      memset(C_values + (size_t)A0_size * (size_t)C1_size, 0,
             sizeof(float) * (size_t)(C0_size - A0_size) * (size_t)C1_size);
  } else {
    memset(C_values, 0, sizeof(float) * C_capacity);
  }

  // Round tile to multiple of 16 for SIMD alignment
  const int kTile = (tile_size + 15) & ~15;

  // Work-aware thread cap + adaptive schedule chunk from the shared policy
  // (csrc/scorch_policy.h): work = nnz*k, grain = SCORCH_GRAIN_SPMM. A small
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
  const long work = (long)total_nnz * k_eff;
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
  // nthreads_override<=0 => pure policy (the standalone/panel default).
  const int policy_nt = scorch_nthreads(work, A0_size, SCORCH_GRAIN_SPMM);
  int nthreads = policy_nt;
  if (nthreads_override > 0 && work >= SCORCH_GRAIN_SPMM) {
    // Adopt the host thread count to avoid the pipeline team-reshape, but bound
    // it two ways so the panel can't regress: (1) never beyond the row-
    // parallelism ceiling by_rows = rows/ROWS_PER_THREAD — a 130-row product at
    // wide K clears the work floor yet can't feed 16 workers, so cap at what the
    // rows support; (2) never below the policy (max) so big graphs keep their
    // possibly-higher policy count. This removes only the by_work throttle that
    // starves tall-skinny GCN SpMM (many rows, narrow k), which is the reshape
    // culprit, while keeping every protection that guards small/tiny products.
    const long by_rows = (long)A0_size / SCORCH_ROWS_PER_THREAD;
    long cand = (long)nthreads_override < by_rows ? (long)nthreads_override : by_rows;
    const long hw = (long)omp_get_num_procs();   // never oversubscribe the box
    if (cand > hw) cand = hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }
  const int chunk = scorch_chunk(A0_size, work, SCORCH_GRAIN_SPMM);
  std::atomic<int> next_row{0};

  // Narrow-k register-blocked path (K<=16): hold the whole output row in YMM
  // accumulators across the row's nonzeros, masked AVX2 load/store, 2-nnz ILP.
  // The workspace path below round-trips through memory (memset + per-nnz ws
  // load/store + memcpy) which dominates when K is tiny and the k-loop is below
  // one SIMD lane — that was the 0.5-0.8x-of-MKL narrow-k gap (GCN k=3/16). For
  // K>16 the workspace path already matches/beats MKL, so it is unchanged.
#if defined(__AVX2__) && defined(__FMA__)
  // Register-block when the whole output row fits in <=4 YMM accumulators (k<=32).
  const bool narrow_k = (B1_size >= 1 && B1_size <= 32);
  const int nvec = (B1_size + 7) / 8;              // 1..4 when narrow_k
  const int mlast = B1_size - 8 * (nvec - 1);      // valid lanes in last vec, 1..8
  const __m256i mask_last = _mm256_setr_epi32(
      mlast>0?-1:0, mlast>1?-1:0, mlast>2?-1:0, mlast>3?-1:0,
      mlast>4?-1:0, mlast>5?-1:0, mlast>6?-1:0, mlast>7?-1:0);
#else
  const bool narrow_k = false;
#endif

#if defined(__AVX2__) && defined(__FMA__) && defined(SCORCH_TUNE_HOOKS)
  // A/B hook: force the legacy workspace path instead of the AVX2 register
  // kernels, for wide-k regression re-checks. Compiled out of the shipped .so.
  const char* _wsonly = std::getenv("SCORCH_SPMM_WORKSPACE");
  const bool force_workspace = _wsonly && *_wsonly && std::atol(_wsonly) != 0;
  // A/B hook: refresh the regtile partial-tile toggle once per op (read by
  // scorch_spmm_row_regtile). SCORCH_REGTILE_BASE=1 -> legacy runtime-nv partial.
  { const char* e = std::getenv("SCORCH_REGTILE_BASE");
    g_scorch_regtile_base = (e && *e && std::atol(e) != 0) ? 1 : 0; }
#endif

  // The per-worker body (atomic row work-stealing). Factored into a lambda so it
  // can be launched EITHER through a private libgomp team (#pragma omp, default)
  // OR through torch's own intra-op pool (at::parallel_for). The latter shares
  // one warm pool with the surrounding torch epilogue (bias/act) so there is no
  // cross-runtime thread-team reformation at each op boundary — the drop-in-
  // pipeline "same thread pool" composition. Work distribution is byte-identical
  // (same next_row atomic, same regblock/regtile kernels); only the launch differs.
  auto scorch_spmm_worker = [&]() {
    // Per-thread workspace for the fallback path (cache-line aligned, lives in
    // L1). On AVX2 the register kernels (regblock k<=32 / regtile k>32) own every
    // row, so the shipped build never touches it (nullptr; free(nullptr) is a
    // no-op). Allocated only for the non-AVX2 fallback or the force_workspace hook.
#if defined(__AVX2__) && defined(__FMA__) && !defined(SCORCH_TUNE_HOOKS)
    float* SCORCH_RESTRICT ws = nullptr;
#else
    float* SCORCH_RESTRICT ws = (float*)aligned_alloc(64, kTile * sizeof(float));
#endif

    // Atomic work-stealing loop with adaptive chunk size
    while (true) {
      const int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= A0_size) break;
      const int end = std::min(start + chunk, A0_size);

      for (int i = start; i < end; i++) {
        const int pA_begin = A1_pos[i];
        const int pA_end   = A1_pos[i + 1];
        if (pA_begin == pA_end) continue;

        float* SCORCH_RESTRICT C_row = C_values + (size_t)i * (size_t)C1_size;

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
            switch (nvec) {
              case 1: scorch_spmm_row_regblock<1>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
              case 2: scorch_spmm_row_regblock<2>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
              case 3: scorch_spmm_row_regblock<3>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
              case 4: scorch_spmm_row_regblock<4>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
            }
          } else {
            scorch_spmm_row_regtile(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end);
          }
          continue;
        }
#endif

        for (int k_out = 0; k_out < B1_size; k_out += kTile) {
          const int kw = std::min(kTile, B1_size - k_out);

          memset(ws, 0, kw * sizeof(float));

          for (int pA = pA_begin; pA < pA_end; pA++) {
            const int j = A1_crd[pA];
            const float a = A_val[pA];
            const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * (size_t)B1_size + k_out;

            if (pA + 1 < pA_end) {
              __builtin_prefetch(B_val + (size_t)A1_crd[pA + 1] * (size_t)B1_size + k_out, 0, 1);
            }

            for (int k = 0; k < kw; k++) {
              ws[k] += a * B_row[k];
            }
          }

          memcpy(C_row + k_out, ws, kw * sizeof(float));
        }
      }
    }

    free(ws);
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
  bool use_atparallel = atparallel && nthreads_override > 0
      && work >= SCORCH_GRAIN_SPMM
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
        scorch_spmm_worker();
      }
    } else {
      at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {
        for (int64_t w = wbeg; w < wend; ++w) scorch_spmm_worker();
      });
    }
  } else {
    #pragma omp parallel num_threads(nthreads)
    {
      scorch_spmm_worker();
    }
  }

  Tensor C;
  if (!torch_alloc) {
    auto C_values_deleter = [](void* ptr) { free(ptr); };
    C_values_torch = torch::from_blob(
        C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
  }
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
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
Tensor spmm_csr_float_tilej(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int Jc = 0, int nthreads_override = -1) {
  const int C0_size = result_shape[0];
  const int C1_size = result_shape[1];              // == N
  const int A0_size = A_shape[0];
  const int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  const int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  const float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();
  const int J = B_shape[0];                         // contraction dim
  const int N = B_shape[1];
  const float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

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

  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
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
// NEON register-tiled SpMM output row (ARM analogue of the AVX2 scorch_spmm_row_
// regtile; the default ARM inner kernel for the fused Linear path). Accumulates
// C_row[0..B1_size) = sum_nnz a*B_row directly in NEON registers, tiling the free
// dim into 32-wide strips held in 8 float32x4 accumulators — 8 independent FMA
// chains hide the ~4-cycle FMA latency without a 2-nnz unroll. This avoids the
// scalar workspace loop's per-nnz round-trip (memset + ws L1 load/store + memcpy),
// which measured ~10-14% of the fused forward on M5 across the AE grid. Tail
// free-dim (B1_size % 32) done scalar.
static inline void scorch_spmm_row_neon_regtile(
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

  // Thread policy / schedule chunk — IDENTICAL to spmm_csr_float_v2 (see there for
  // the full rationale): work = nnz*max(k,16), grain = SCORCH_GRAIN_SPMM; adopt
  // the host thread count when the caller passes nthreads_override (avoids the
  // pipeline team-reshape), bounded by the row-parallelism ceiling and floored at
  // the policy count.
  const int total_nnz = A1_pos[A0_size];
  const long k_eff = B1_size < 16 ? 16L : (long)B1_size;
  const long work = (long)total_nnz * k_eff;
  const int policy_nt = scorch_nthreads(work, A0_size, SCORCH_GRAIN_SPMM);
  int nthreads = policy_nt;
  if (nthreads_override > 0 && work >= SCORCH_GRAIN_SPMM) {
    const long by_rows = (long)A0_size / SCORCH_ROWS_PER_THREAD;
    long cand = (long)nthreads_override < by_rows ? (long)nthreads_override : by_rows;
    const long hw = (long)omp_get_num_procs();
    if (cand > hw) cand = hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }
  const int chunk = scorch_chunk(A0_size, work, SCORCH_GRAIN_SPMM);
  std::atomic<int> next_row{0};

#if defined(__AVX2__) && defined(__FMA__)
  const bool narrow_k = (B1_size >= 1 && B1_size <= 32);
  const int nvec = (B1_size + 7) / 8;
  const int mlast = B1_size - 8 * (nvec - 1);
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
#endif

  // Per-worker body: atomic row work-stealing, byte-identical distribution to v2.
  // Computes each output row via the AVX2 regblock/regtile kernels (or the non-
  // AVX2 workspace fallback), then folds bias+act into the SAME parallel region.
  auto worker = [&]() {
#if defined(__AVX2__) && defined(__FMA__)
    float* SCORCH_RESTRICT ws = nullptr;
#else
    float* SCORCH_RESTRICT ws = (float*)aligned_alloc(64, kTile * sizeof(float));
#endif
    while (true) {
      const int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= A0_size) break;
      const int end = std::min(start + chunk, A0_size);

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
          switch (nvec) {
            case 1: scorch_spmm_row_regblock<1>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
            case 2: scorch_spmm_row_regblock<2>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
            case 3: scorch_spmm_row_regblock<3>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
            case 4: scorch_spmm_row_regblock<4>(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end, mask_last); break;
          }
        } else {
          scorch_spmm_row_regtile(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end);
        }
#elif defined(__ARM_NEON)
        if (use_neon_regtile) {
          scorch_spmm_row_neon_regtile(A1_crd, A_val, B_val, B1_size, C_row, pA_begin, pA_end);
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
    free(ws);
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
        worker();
      }
    } else {
      at::parallel_for(0, (int64_t)nthreads, 1, [&](int64_t wbeg, int64_t wend) {
        for (int64_t w = wbeg; w < wend; ++w) worker();
      });
    }
  } else {
    #pragma omp parallel num_threads(nthreads)
    {
      worker();
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
  if (nthreads_override > 0 && ncblk > 1) {
    at::parallel_for(0, ncblk, 1,
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

  auto do_rows = [&](int64_t r0, int64_t r1) {
    // One allocation per parallel task: [max_len score buffer | D-wide V accum].
    std::vector<float> scratch(max_len + D);
    float* SCORCH_RESTRICT scores = scratch.data();
    float* SCORCH_RESTRICT acc = scratch.data() + max_len;
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
    at::parallel_for(0, S, 8, [&](int64_t r0, int64_t r1) { do_rows(r0, r1); });
  } else {
    do_rows(0, S);
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
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {(long long)C_capacity}, C_values_deleter, torch::kFloat32);
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}
