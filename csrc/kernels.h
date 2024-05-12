#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

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

Tensor spmspm_coo_float_taco(std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {

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

          C_values[pC1] += A_val[pA1] * B_val[pB1];
          C1_crd[pC1] = j;
          C0_crd[pC1] = i;
          pC1++;
        }
      }

      // Advance iterators
      pA1 += (int) k_A == k;
      // pB0 += (int) k_B == k;
      pB0 = pB1_end;
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
