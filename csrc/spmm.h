#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "prebuilt_types.h"

#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

// Global constants for optimization
const int kUnrollFactor = 16;

template <typename scalar_t>
Tensor spmm_csr_typed(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, int tile_size = 32) {
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

  // Fast path: fixed tile size that matches the best compiler-generated strategy.
  // This avoids heap allocation in the hot loop and enables full unrolling.
  constexpr int kFastTile = 32;
  int kTile = tile_size > 0 ? tile_size : kFastTile;

  if (kTile == kFastTile) {
    int full_j_end = (B1_size / kFastTile) * kFastTile;

    #pragma omp parallel for schedule(static)
    for (int j_out = 0; j_out < full_j_end; j_out += kFastTile) {
      for (int i = 0; i < A0_size; i++) {
        scalar_t accum_c[kFastTile] = {};
        int pA1_begin = A1_pos[i];
        int pA1_end = A1_pos[i + 1];

        for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
          int j = A1_crd[pA1];
          scalar_t a_val = A_val[pA1];
          size_t pB1_base = (size_t)j * (size_t)B1_size + (size_t)j_out;

          SCORCH_PRAGMA_UNROLL
          for (int j_in = 0; j_in < kFastTile; j_in++) {
            accum_c[j_in] += a_val * B_val[pB1_base + j_in];
          }
        }

        size_t pC1_base = (size_t)i * (size_t)C1_size + (size_t)j_out;
        SCORCH_PRAGMA_UNROLL
        for (int j_in = 0; j_in < kFastTile; j_in++) {
          C_values[pC1_base + j_in] = accum_c[j_in];
        }
      }
    }

    if (full_j_end < B1_size) {
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < A0_size; i++) {
        int pA1_begin = A1_pos[i];
        int pA1_end = A1_pos[i + 1];

        for (int j_out = full_j_end; j_out < B1_size; j_out++) {
          scalar_t accum = static_cast<scalar_t>(0);
          for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
            int j = A1_crd[pA1];
            size_t pB1 = (size_t)j * (size_t)B1_size + (size_t)j_out;
            accum += A_val[pA1] * B_val[pB1];
          }
          size_t pC1 = (size_t)i * (size_t)C1_size + (size_t)j_out;
          C_values[pC1] = accum;
        }
      }
    }
  } else {
    // Generic path for custom tile sizes, still avoiding per-tile heap allocations.
    #pragma omp parallel
    {
      std::vector<scalar_t> accum_c(kTile, static_cast<scalar_t>(0));

      #pragma omp for schedule(static)
      for (int i = 0; i < A0_size; i++) {
        int pA1_begin = A1_pos[i];
        int pA1_end = A1_pos[i + 1];

        for (int j_out = 0; j_out < B1_size; j_out += kTile) {
          int j_width = std::min(kTile, B1_size - j_out);
          memset(accum_c.data(), 0, sizeof(scalar_t) * j_width);

          for (int pA1 = pA1_begin; pA1 < pA1_end; pA1++) {
            int j = A1_crd[pA1];
            scalar_t a_val = A_val[pA1];
            size_t pB1_base = (size_t)j * (size_t)B1_size + (size_t)j_out;

            for (int j_in = 0; j_in < j_width; j_in++) {
              accum_c[j_in] += a_val * B_val[pB1_base + j_in];
            }
          }

          size_t pC1_base = (size_t)i * (size_t)C1_size + (size_t)j_out;
          for (int j_in = 0; j_in < j_width; j_in++) {
            C_values[pC1_base + j_in] = accum_c[j_in];
          }
        }
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
