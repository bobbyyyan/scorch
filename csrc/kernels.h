#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <vector>
#include <algorithm>
#include <numeric>

#include "prebuilt_types.h"

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
  scalar_t* C_values = (scalar_t*)malloc(sizeof(scalar_t) * C0_size);

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
  auto C_values_deleter = [](void *ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C0_size}, C_values_deleter, scorch_torch_dtype<scalar_t>());
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

  // Phase 1: Count nnz per row in parallel
  int* row_nnz = (int*)calloc(A0_size, sizeof(int));

  #pragma omp parallel
  {
    // Thread-local linked-list workspace for counting
    std::vector<int> next(C1_size, -1);

    #pragma omp for schedule(dynamic, 64)
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

      row_nnz[i] = length;

      // Reset linked list
      while (head >= 0) {
        int temp = head;
        head = next[head];
        next[temp] = -1;
      }
    }
  }

  // Phase 2: Prefix sum to compute row pointers
  int* C1_pos_data = (int*)malloc((A0_size + 1) * sizeof(int));
  C1_pos_data[0] = 0;
  for (int i = 0; i < A0_size; i++) {
    C1_pos_data[i + 1] = C1_pos_data[i] + row_nnz[i];
  }
  int total_nnz = C1_pos_data[A0_size];
  free(row_nnz);

  // Phase 3: Numeric multiply in parallel - each row writes to its own slice
  int* C1_crd_data = (int*)malloc(total_nnz * sizeof(int));
  scalar_t* C_values_data = (scalar_t*)malloc(total_nnz * sizeof(scalar_t));

  #pragma omp parallel
  {
    std::vector<int> next(C1_size, -1);
    std::vector<scalar_t> sums(C1_size, 0);

    #pragma omp for schedule(dynamic, 64)
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
        // Build index array and sort
        std::vector<int> idx(length);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
          return C1_crd_data[base + a] < C1_crd_data[base + b];
        });
        std::vector<int> tmp_crd(length);
        std::vector<scalar_t> tmp_val(length);
        for (int jj = 0; jj < length; jj++) {
          tmp_crd[jj] = C1_crd_data[base + idx[jj]];
          tmp_val[jj] = C_values_data[base + idx[jj]];
        }
        memcpy(C1_crd_data + base, tmp_crd.data(), length * sizeof(int));
        memcpy(C_values_data + base, tmp_val.data(), length * sizeof(scalar_t));
      }
    }
  }

  // Assemble final result
  Tensor C;
  auto int_deleter = [](void* p) { free(p); };
  torch::Tensor C1_pos_torch = torch::from_blob(C1_pos_data, {(long long)(A0_size + 1)}, int_deleter, torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(C1_crd_data, {(long long)total_nnz}, int_deleter, torch::kInt);
  auto val_deleter = [](void* p) { free(p); };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values_data,
      {(long long)total_nnz},
      val_deleter,
      scorch_torch_dtype<scalar_t>());
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
  cvector<int> C1_pos;
  cvector<int> C1_crd;
  C1_pos[0] = 0;
  int pC1 = 0;
  int C1_pos_index = 0;

  for (int pC1 = 1; pC1 <= C0_size; pC1++) {
    C1_pos[pC1] = 0;
  }
  // Initialize result value array
  cvector<float> C_values;


  for (int i = 0; i < A0_size; i++) {
    // Assemble COMPRESSED level
    for (; C1_pos_index < i; C1_pos_index++) {
      C1_pos[C1_pos_index + 1] = C1_crd.size();
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
      int k = it.first;
      float wksp_value = it.second;

      C_values[pC1] = wksp_value;
      C1_crd[pC1] = k;
      pC1++;
    }


    // Assembly compressed _level indices
    C1_pos[C1_pos_index + 1] = C1_crd.size();
  }
  // Assemble final result (Do not change this part of the code)
  Tensor C;
  torch::Tensor C1_pos_torch = torch::from_blob(C1_pos.data(), {C1_pos.size()}, C1_pos.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch = torch::from_blob(C_values.data(), {C_values.size()}, C_values.get_deleter(), torch::kFloat32);
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
  cvector<int> C0_crd;
  int pC0 = 0;

  cvector<int> C1_crd;
  int pC1 = 0;

  // Initialize result value array
  cvector<float> C_values;

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
      int j = it.first;
      float wksp_value = it.second;

      C_values[pC1] = wksp_value;
      C1_crd[pC1] = j;
      pC1++;
    }

  }
  // Assemble final result
  Tensor C;
  torch::Tensor C0_crd_torch = torch::from_blob(C0_crd.data(), {C0_crd.size()}, C0_crd.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch = torch::from_blob(C_values.data(), {C_values.size()}, C_values.get_deleter(), torch::kFloat32);
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
  cvector<int> C0_crd;
  cvector<int> C1_crd;
  cvector<float> C_values;

  // Initialize iterators
  int pA0_end = A0_crd_tensor.size(0);
  int pA1_end = 0;

  int B0_size = B_shape[0];
  std::vector<int> next(B0_size, -1);
  std::vector<float> sums(B0_size, 0);

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
      int idx = C_values.size() - length + jj;
      col_val_pairs.emplace_back(C1_crd[idx], C_values[idx]);
    }
    std::sort(col_val_pairs.begin(), col_val_pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (int jj = 0; jj < length; jj++) {
      int idx = C_values.size() - length + jj;
      C1_crd[idx] = col_val_pairs[jj].first;
      C_values[idx] = col_val_pairs[jj].second;
    }
  }

  // Assemble final result
  Tensor C;
  torch::Tensor C0_crd_torch = torch::from_blob(C0_crd.data(), {C0_crd.size()}, C0_crd.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch = torch::from_blob(C_values.data(), {C_values.size()}, C_values.get_deleter(), torch::kFloat32);
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

  float* SCORCH_RESTRICT D_val = (float*)malloc(sizeof(float) * nnz);

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
  auto deleter = [](void* ptr) { free(ptr); };
  torch::Tensor D_values_torch = torch::from_blob(
      D_val, {(long long)nnz}, deleter, torch::kFloat32);
  D.storage.index.mode_indices = S_mode_indices;
  D.storage.value = D_values_torch;
  return D;
}
