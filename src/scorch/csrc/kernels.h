#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <vector>
#include <algorithm>
#include <numeric>

#include "prebuilt_types.h"
#include "scorch_policy.h"  // shared scorch_nthreads / scorch_chunk (see header.h)

template <typename scalar_t>
Tensor spmv_csr(
  std::vector<int> result_shape,
  std::vector<int> A_shape,
  std::vector<std::vector<torch::Tensor>> A_mode_indices,
  torch::Tensor A_values,
  std::vector<int> B_shape,
  std::vector<std::vector<torch::Tensor>> B_mode_indices,
  torch::Tensor B_values) {
  (void)B_shape;
  (void)B_mode_indices;

  int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
  scalar_t* A_val = A_values.data_ptr<scalar_t>();
  scalar_t* B_val = B_values.data_ptr<scalar_t>();

  int C0_size = result_shape[0];
  torch::Tensor C_values_torch =
      torch::empty({C0_size}, scorch_torch_dtype<scalar_t>());
  scalar_t* C_values = C_values_torch.data_ptr<scalar_t>();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < C0_size; i++) {
    scalar_t accum = static_cast<scalar_t>(0);
    for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      int j = A1_crd[pA1];
      accum += A_val[pA1] * B_val[j];
    }
    C_values[i] = accum;
  }

  Tensor C;
  C.storage.index.mode_indices = {{}};
  C.storage.value = C_values_torch;
  return C;
}

template <typename scalar_t>
Tensor spmspm_csr(
  std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Get A's level & value arrays
  const int A0_size = A_shape[0];
  const int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  const int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  const scalar_t* SCORCH_RESTRICT A_val = A_values.data_ptr<scalar_t>();

  // Get B's level & value arrays
  const int B0_size = B_shape[0];
  const int* SCORCH_RESTRICT B1_pos = B_mode_indices[1][0].data_ptr<int>();
  const int* SCORCH_RESTRICT B1_crd = B_mode_indices[1][1].data_ptr<int>();
  const scalar_t* SCORCH_RESTRICT B_val = B_values.data_ptr<scalar_t>();

  const int C1_size = result_shape.size() > 1 ? result_shape[1] : B0_size;

  // Estimate the multiply's work (A nonzeros x average B row length) to drive the
  // adaptive threading below.
  const long A_nnz = A1_pos[A0_size];
  const long B_nnz = B1_pos[B0_size];
  const long avg_B_row = B0_size > 0 ? (B_nnz / B0_size) + 1 : 1;
  const long flop_est = A_nnz * avg_B_row;
  // Work-aware thread cap + adaptive schedule chunk from the shared policy
  // (scorch/csrc/scorch_policy.h): work = flop_est (A_nnz*avg_B_row), grain = SCORCH_GRAIN_SPMSPM.
  // scorch_nthreads reproduces this kernel's former inline decision exactly —
  // clamp(min(flop_est/3000, A0_size/16), 1, omp_get_num_procs()) — and scorch_chunk the
  // matching clamp(A0_size/(nthreads*7), 4, 64). The dynamic-schedule chunk is the dominant
  // lever: a coarse fixed chunk starves load-balancing so the join barrier stalls on the
  // slowest cores (e.g. a hybrid Intel P+E CPU's E-cores, a 4-7x cliff).
  const int nthreads = scorch_nthreads(flop_est, A0_size, SCORCH_GRAIN_SPMSPM);
  const int chunk = scorch_chunk(A0_size, flop_est, SCORCH_GRAIN_SPMSPM);
  const size_t column_count = static_cast<size_t>(C1_size);
  const auto cache_line_stride = [nthreads](size_t count, size_t element_size) {
    if (count == 0) return size_t{0};
    if (nthreads == 1) return count;
    const size_t elements_per_line = std::max<size_t>(1, 64 / element_size);
    const size_t padded =
        ((count + elements_per_line - 1) / elements_per_line) *
        elements_per_line;
    // Keep an unused cache line between unaligned new[]-backed worker slices.
    return padded + elements_per_line;
  };
  const size_t next_stride = cache_line_stride(column_count, sizeof(int));
  auto next_by_worker = scorch_make_unique_array_pool<int>(
      static_cast<size_t>(nthreads), next_stride);
  std::fill_n(next_by_worker.get(),
              scorch_checked_size_product(static_cast<size_t>(nthreads),
                                          next_stride),
              -1);

  // Phase 1: Count nnz directly into the eventual position owner. Keeping the
  // counts in slots [1, rows] lets Phase 2 prefix them in place, avoiding a
  // second temporary allocation and copy while ownership remains move-only.
  auto C1_pos_owner = scorch_make_unique_array_pool<int>(
      1, static_cast<size_t>(A0_size) + 1);
  int* C1_pos_data = C1_pos_owner.get();
  C1_pos_data[0] = 0;

  #pragma omp parallel num_threads(nthreads)
  {
    // Thread-local linked-list slice from the serially allocated workspace pool.
    int* SCORCH_RESTRICT next =
        next_by_worker.get() +
        static_cast<size_t>(omp_get_thread_num()) * next_stride;

    #pragma omp for schedule(dynamic, chunk)
    for (int i = 0; i < A0_size; i++) {
      int head = -2;
      int length = 0;

      for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
        int j = A1_crd[pA1];
        for (int pB1 = B1_pos[j]; pB1 < B1_pos[j + 1]; pB1++) {
          int k = B1_crd[pB1];
          if (next[k] == -1) {
            next[k] = head;
            head = k;
            length++;
          }
        }
      }

      C1_pos_data[i + 1] = length;

      // Reset linked list
      while (head >= 0) {
        int temp = head;
        head = next[head];
        next[temp] = -1;
      }
    }
  }

  // Phase 2: Prefix sum to compute row pointers
  int max_row_nnz = 0;
  for (int i = 0; i < A0_size; i++) {
    const int count = C1_pos_data[i + 1];
    max_row_nnz = std::max(max_row_nnz, count);
    if (count > std::numeric_limits<int>::max() - C1_pos_data[i]) {
      throw std::length_error("CSR SpGEMM output exceeds int32 index capacity");
    }
    C1_pos_data[i + 1] = C1_pos_data[i] + count;
  }
  int total_nnz = C1_pos_data[A0_size];

  // Phase 3: Numeric multiply in parallel - each row writes to its own slice
  auto C1_crd_owner = scorch_make_unique_array_pool<int>(
      1, static_cast<size_t>(total_nnz));
  auto C_values_owner = scorch_make_unique_array_pool<scalar_t>(
      1, static_cast<size_t>(total_nnz));
  int* C1_crd_data = C1_crd_owner.get();
  scalar_t* C_values_data = C_values_owner.get();

  struct SortEntry {
    int coordinate;
    scalar_t value;
  };
  const size_t sums_stride = cache_line_stride(column_count, sizeof(scalar_t));
  const size_t entries_stride = max_row_nnz > 32
      ? cache_line_stride(static_cast<size_t>(max_row_nnz), sizeof(SortEntry))
      : 0;
  auto sums_by_worker = scorch_make_unique_array_pool<scalar_t>(
      static_cast<size_t>(nthreads), sums_stride);
  std::unique_ptr<SortEntry[]> entries_by_worker;
  if (entries_stride != 0) {
    entries_by_worker = scorch_make_unique_array_pool<SortEntry>(
        static_cast<size_t>(nthreads), entries_stride);
  }
  std::fill_n(sums_by_worker.get(),
              scorch_checked_size_product(static_cast<size_t>(nthreads),
                                          sums_stride),
              static_cast<scalar_t>(0));

  #pragma omp parallel num_threads(nthreads)
  {
    const size_t worker = static_cast<size_t>(omp_get_thread_num());
    int* SCORCH_RESTRICT next =
        next_by_worker.get() + worker * next_stride;
    scalar_t* SCORCH_RESTRICT sums =
        sums_by_worker.get() + worker * sums_stride;
    SortEntry* SCORCH_RESTRICT entries = entries_stride != 0
        ? entries_by_worker.get() + worker * entries_stride
        : nullptr;

    #pragma omp for schedule(dynamic, chunk)
    for (int i = 0; i < A0_size; i++) {
      int head = -2;
      int length = 0;

      for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
        int j = A1_crd[pA1];
        scalar_t v = A_val[pA1];
        for (int pB1 = B1_pos[j]; pB1 < B1_pos[j + 1]; pB1++) {
          int k = B1_crd[pB1];
          sums[k] += v * B_val[pB1];
          if (next[k] == -1) {
            next[k] = head;
            head = k;
            length++;
          }
        }
      }

      // Collect and sort the row's entries directly into output
      int base = C1_pos_data[i];
      int pos = 0;
      while (head >= 0) {
        C1_crd_data[base + pos] = head;
        C_values_data[base + pos] = sums[head];
        sums[head] = 0;
        int temp = head;
        head = next[head];
        next[temp] = -1;
        pos++;
      }

      // Sort columns within this row
      // Use simple insertion sort for short rows, std::sort for longer ones
      if (length <= 32) {
        for (int a = 1; a < length; a++) {
          int key_c = C1_crd_data[base + a];
          scalar_t key_v = C_values_data[base + a];
          int b = a - 1;
          while (b >= 0 && C1_crd_data[base + b] > key_c) {
            C1_crd_data[base + b + 1] = C1_crd_data[base + b];
            C_values_data[base + b + 1] = C_values_data[base + b];
            b--;
          }
          C1_crd_data[base + b + 1] = key_c;
          C_values_data[base + b + 1] = key_v;
        }
      } else {
        // Sort a worker-local coordinate/value copy so the parallel region never
        // allocates, then copy the ordered row back to its output slice.
        for (int jj = 0; jj < length; jj++) {
          entries[jj] =
              SortEntry{C1_crd_data[base + jj], C_values_data[base + jj]};
        }
        std::sort(entries, entries + length, [](const SortEntry& left,
                                                const SortEntry& right) {
          return left.coordinate < right.coordinate;
        });
        for (int jj = 0; jj < length; jj++) {
          C1_crd_data[base + jj] = entries[jj].coordinate;
          C_values_data[base + jj] = entries[jj].value;
        }
      }
    }
  }

  // Assemble final result
  torch::Tensor C1_pos_torch = scorch_tensor_from_unique_array(
      std::move(C1_pos_owner), static_cast<int64_t>(A0_size) + 1, torch::kInt);
  torch::Tensor C1_crd_torch = scorch_tensor_from_unique_array(
      std::move(C1_crd_owner), static_cast<int64_t>(total_nnz), torch::kInt);
  torch::Tensor C_values_torch = scorch_tensor_from_unique_array(
      std::move(C_values_owner),
      static_cast<int64_t>(total_nnz),
      scorch_torch_dtype<scalar_t>());
  Tensor C;
  C.storage.index.mode_indices = {{}, {C1_pos_torch, C1_crd_torch}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmspm_csr_float(std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int* B1_pos = B_mode_indices[1][0].data_ptr<int>();
  int* B1_crd = B_mode_indices[1][1].data_ptr<int>();
  float* B_val = B_values.data_ptr<float>();

  // Init result level indices
  std::vector<int> C1_pos;
  std::vector<int> C1_crd;
  scorch_vector_set(C1_pos, 0, 0);
  int pC1 = 0;
  int C1_pos_index = 0;

  for (int pC1 = 1; pC1 <= C0_size; pC1++) {
    scorch_vector_set(C1_pos, pC1, 0);
  }
  // Initialize result value array
  std::vector<float> C_values;


  for (int i = 0; i < A0_size; i++) {
    // Assemble COMPRESSED level
    for (; C1_pos_index < i; C1_pos_index++) {
      scorch_vector_set(C1_pos, C1_pos_index + 1, C1_crd.size());
    }
    // Resolve dense coordinates
    int pA0 = i;
    // Resolve index into dense level of values array
    int pC0 = i;
    // Initialize workspaces
    auto wksp = coo_workspace_1d<float, 1>(1024);
    // Initialize iterators
    int pA1_end = A1_pos[i + 1];

    for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
      // Resolve coordinates
      int j = A1_crd[pA1];

      // Resolve dense coordinates
      int pB0 = j;
      // Initialize iterators
      int pB1_end = B1_pos[j + 1];

      for (int pB1 = B1_pos[j]; pB1 < pB1_end; pB1++) {
        // Resolve coordinates
        int k = B1_crd[pB1];

        wksp.insert({k}, A_val[pA1] * B_val[pB1]);
      }
    }

    // Lower consumer CIN
    wksp.sort();
    for (const auto& it : wksp) {
      int64_t k = it.first;
      float wksp_value = it.second;

      scorch_vector_set(C_values, pC1, wksp_value);
      scorch_vector_set(C1_crd, pC1, k);
      pC1++;
    }


    // Assembly compressed _level indices
    scorch_vector_set(C1_pos, C1_pos_index + 1, C1_crd.size());
  }
  // Assemble final result (Do not change this part of the code)
  Tensor C;
  torch::Tensor C1_pos_torch =
      scorch_tensor_from_vector(std::move(C1_pos), torch::kInt);
  torch::Tensor C1_crd_torch =
      scorch_tensor_from_vector(std::move(C1_crd), torch::kInt);
  torch::Tensor C_values_torch =
      scorch_tensor_from_vector(std::move(C_values), torch::kFloat32);
  C.storage.index.mode_indices = {{}, {C1_pos_torch, C1_crd_torch}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmspm_coo_float(
  std::vector<int> result_shape,
  std::vector<int> A_shape,
  std::vector<std::vector<torch::Tensor>> A_mode_indices,
  torch::Tensor A_values,
  std::vector<int> B_shape,
  std::vector<std::vector<torch::Tensor>> B_mode_indices,
  torch::Tensor B_values) {

  // Get A's level & value arrays
  torch::Tensor A0_crd_tensor = A_mode_indices[0][0];
  int* A0_crd = A_mode_indices[0][0].data_ptr<int>();
  torch::Tensor A1_crd_tensor = A_mode_indices[1][0];
  int* A1_crd = A_mode_indices[1][0].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  torch::Tensor B0_crd_tensor = B_mode_indices[0][0];
  int* B0_crd = B_mode_indices[0][0].data_ptr<int>();
  torch::Tensor B1_crd_tensor = B_mode_indices[1][0];
  int* B1_crd = B_mode_indices[1][0].data_ptr<int>();
  float* B_val = B_values.data_ptr<float>();

  // Init result level indices
  std::vector<int> C0_crd;
  int pC0 = 0;

  std::vector<int> C1_crd;
  int pC1 = 0;

  // Initialize result value array
  std::vector<float> C_values;

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

    // Initialize workspaces
    auto wksp = coo_workspace_1d<float, 1>(1024);
    // Initialize iterators
    int pA1 = pA0;
    int pB0 = 0;
    int pB0_end = B0_crd_tensor.size(0);
    int pB1_end = 0;

    while (pA1 < pA1_end && pB0 < pB0_end) {
      // Load coordinates
      int k_A = A1_crd[pA1];
      int k_B = B0_crd[pB0];

      // Resolve coordinates
      int k = std::min({k_A, k_B});

      // Find iterator end for coordinate level
      pB1_end = pB0 + 1;
      while (pB1_end < pB0_end && B0_crd[pB1_end] == k) {
        pB1_end++;
      }

      // Inner loops over child regions
      if (k_A == k && k_B == k) {

        for (int pB1 = pB0; pB1 < pB1_end; pB1++) {
          // Resolve coordinates
          int j = B1_crd[pB1];

          wksp.insert({j}, A_val[pA1] * B_val[pB1]);
        }
      }

      // Advance iterators
      pA1 += (int) k_A == k;
      pB0 += (int) k_B == k;
    }

    // Lower consumer CIN
    wksp.sort();
    for (const auto& it : wksp) {
      int64_t j = it.first;
      float wksp_value = it.second;

      scorch_vector_set(C0_crd, pC1, i);
      scorch_vector_set(C_values, pC1, wksp_value);
      scorch_vector_set(C1_crd, pC1, j);
      pC1++;
    }

  }
  // Assemble final result
  Tensor C;
  torch::Tensor C0_crd_torch =
      scorch_tensor_from_vector(std::move(C0_crd), torch::kInt);
  torch::Tensor C1_crd_torch =
      scorch_tensor_from_vector(std::move(C1_crd), torch::kInt);
  torch::Tensor C_values_torch =
      scorch_tensor_from_vector(std::move(C_values), torch::kFloat32);
  C.storage.index.mode_indices = {{C0_crd_torch}, {C1_crd_torch}};
  C.storage.value = C_values_torch;
  return C;
}

Tensor spmspm_coo_float_opt(
  std::vector<int> result_shape,
  std::vector<int> A_shape,
  std::vector<std::vector<torch::Tensor>> A_mode_indices,
  torch::Tensor A_values,
  std::vector<int> B_shape,
  std::vector<std::vector<torch::Tensor>> B_mode_indices,
  torch::Tensor B_values) {

  // Get A's level & value arrays
  torch::Tensor A0_crd_tensor = A_mode_indices[0][0];
  int* A0_crd = A_mode_indices[0][0].data_ptr<int>();
  torch::Tensor A1_crd_tensor = A_mode_indices[1][0];
  int* A1_crd = A_mode_indices[1][0].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  torch::Tensor B0_crd_tensor = B_mode_indices[0][0];
  int* B0_crd = B_mode_indices[0][0].data_ptr<int>();
  torch::Tensor B1_crd_tensor = B_mode_indices[1][0];
  int* B1_crd = B_mode_indices[1][0].data_ptr<int>();
  float* B_val = B_values.data_ptr<float>();

  // Init result level indices
  std::vector<int> C0_crd;
  std::vector<int> C1_crd;
  std::vector<float> C_values;

  // Initialize iterators
  int pA0_end = A0_crd_tensor.size(0);
  int pA1_end = 0;

  int B1_size = B_shape[1];
  std::vector<int> next(B1_size, -1);
  std::vector<float> sums(B1_size, 0);

  for (int pA0 = 0; pA0 < pA0_end; pA0 = pA1_end) {
    // Resolve coordinates
    int i = A0_crd[pA0];

    // Find iterator end for coordinate level
    pA1_end = pA0 + 1;
    while (pA1_end < pA0_end && A0_crd[pA1_end] == i) {
      pA1_end++;
    }

    int head = -2;
    int length = 0;

    for (int pA1 = pA0; pA1 < pA1_end; pA1++) {
      int j = A1_crd[pA1];
      float v = A_val[pA1];

      for (int pB1 = 0; pB1 < B1_crd_tensor.size(0); pB1++) {
        if (B0_crd[pB1] == j) {
          int k = B1_crd[pB1];

          sums[k] += v * B_val[pB1];

          if (next[k] == -1) {
            next[k] = head;
            head = k;
            length++;
          }
        }
      }
    }

    for (int jj = 0; jj < length; jj++) {
      C0_crd.push_back(i);
      C1_crd.push_back(head);
      C_values.push_back(sums[head]);

      int temp = head;
      head = next[head];

      next[temp] = -1;
      sums[temp] = 0;
    }

    std::vector<std::pair<int, float>> col_val_pairs;
    col_val_pairs.reserve(length);
    for (int jj = 0; jj < length; jj++) {
      const size_t idx = C_values.size() - length + jj;
      col_val_pairs.emplace_back(C1_crd.at(idx), C_values.at(idx));
    }
    std::sort(col_val_pairs.begin(), col_val_pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (int jj = 0; jj < length; jj++) {
      const size_t idx = C_values.size() - length + jj;
      C1_crd.at(idx) = col_val_pairs[jj].first;
      C_values.at(idx) = col_val_pairs[jj].second;
    }
  }

  // Assemble final result
  Tensor C;
  torch::Tensor C0_crd_torch =
      scorch_tensor_from_vector(std::move(C0_crd), torch::kInt);
  torch::Tensor C1_crd_torch =
      scorch_tensor_from_vector(std::move(C1_crd), torch::kInt);
  torch::Tensor C_values_torch =
      scorch_tensor_from_vector(std::move(C_values), torch::kFloat32);
  C.storage.index.mode_indices = {{C0_crd_torch}, {C1_crd_torch}};
  C.storage.value = C_values_torch;
  return C;
}

// ---------------------------------------------------------------------------
// SDDMM: D[i,j] = S[i,j] * dot(A[i,:], B[j,:])
// S is COO (o,o), A and B are dense (d,d) row-major.
// ---------------------------------------------------------------------------

Tensor sddmm_coo_float_prebuilt(
    std::vector<int> result_shape,
    std::vector<int> S_shape,
    std::vector<std::vector<torch::Tensor>> S_mode_indices,
    torch::Tensor S_values,
    std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices,
    torch::Tensor A_values,
    std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices,
    torch::Tensor B_values) {

  const int nnz = S_values.numel();
  const int K = A_shape[1];

  const int* SCORCH_RESTRICT S_row = S_mode_indices[0][0].data_ptr<int>();
  const int* SCORCH_RESTRICT S_col = S_mode_indices[1][0].data_ptr<int>();
  const float* SCORCH_RESTRICT S_val = S_values.data_ptr<float>();
  const float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();
  const float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  torch::Tensor D_values_torch =
      torch::empty({static_cast<long long>(nnz)}, torch::kFloat32);
  float* SCORCH_RESTRICT D_val = D_values_torch.data_ptr<float>();

  const int nthreads = omp_get_max_threads();
  const int chunk = std::max(16, std::min(256, nnz / (nthreads * 128)));
  std::atomic<int> next_p{0};

  #pragma omp parallel
  {
    while (true) {
      const int start = next_p.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= nnz) break;
      const int end = std::min(start + chunk, nnz);

      for (int p = start; p < end; p++) {
        const int i = S_row[p];
        const int j = S_col[p];
        const float s = S_val[p];
        const float* SCORCH_RESTRICT A_row = A_val + (size_t)i * K;
        const float* SCORCH_RESTRICT B_row = B_val + (size_t)j * K;

        if (p + 1 < end) {
          __builtin_prefetch(A_val + (size_t)S_row[p + 1] * K, 0, 1);
          __builtin_prefetch(B_val + (size_t)S_col[p + 1] * K, 0, 1);
        }

        float dot = 0;
        for (int k = 0; k < K; k++) {
          dot += A_row[k] * B_row[k];
        }
        D_val[p] = s * dot;
      }
    }
  }

  Tensor D;
  D.storage.index.mode_indices = S_mode_indices;
  D.storage.value = D_values_torch;
  return D;
}
