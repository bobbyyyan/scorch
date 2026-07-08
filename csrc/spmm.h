#include <algorithm>
#include <atomic>
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
                int nthreads_override = -1) {
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

  #pragma omp parallel num_threads(nthreads)
  {
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
