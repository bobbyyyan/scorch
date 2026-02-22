#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <vector>
#include <algorithm>

template <typename scalar_t>
Tensor spmspm_csr(
  std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
  scalar_t* A_val = A_values.data_ptr<scalar_t>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int* B1_pos = B_mode_indices[1][0].data_ptr<int>();
  int* B1_crd = B_mode_indices[1][1].data_ptr<int>();
  scalar_t* B_val = B_values.data_ptr<scalar_t>();

  // Init result level indices
  cvector<int> C1_pos(A0_size + 1);
  memset(C1_pos.data(), 0, (A0_size + 1) * sizeof(int));
  cvector<int> C1_crd;
  cvector<scalar_t> C_values;

  std::vector<int> next(B0_size, -1);
  std::vector<scalar_t> sums(B0_size, 0);

  int nnz = 0;

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

    for (int jj = 0; jj < length; jj++) {
      C1_crd.push_back(head);
      C_values.push_back(sums[head]);
      nnz++;

      int temp = head;
      head = next[head];

      next[temp] = -1;
      sums[temp] = 0;
    }

    std::vector<std::pair<int, scalar_t>> col_val_pairs;
    col_val_pairs.reserve(length);
    for (int jj = 0; jj < length; jj++) {
      col_val_pairs.emplace_back(C1_crd[nnz - length + jj], C_values[nnz - length + jj]);
    }
    std::sort(col_val_pairs.begin(), col_val_pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (int jj = 0; jj < length; jj++) {
      C1_crd[nnz - length + jj] = col_val_pairs[jj].first;
      C_values[nnz - length + jj] = col_val_pairs[jj].second;
    }

    C1_pos[i + 1] = nnz;
  }

  // Assemble final result
  Tensor C;
  torch::Tensor C1_pos_torch = torch::from_blob(C1_pos.data(), {C1_pos.size()}, C1_pos.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch = torch::from_blob(C_values.data(), {C_values.size()}, C_values.get_deleter(), torch::kFloat32);
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
