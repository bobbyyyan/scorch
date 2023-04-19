// clang-format off
// taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:dddd -f=C:ssss -print-evaluate
// Modified for Scorch
// clang-format on

Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, std::vector<int> C_shape,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Init result tensor level sizes
  int A0_size = result_shape[0];
  int A1_size = result_shape[1];

  // Get tensor level arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  int B2_size = B_shape[2];
  int B3_size = B_shape[3];
  torch::Tensor C0_pos = C_mode_indices[0][0];
  torch::Tensor C0_crd = C_mode_indices[0][1];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];
  torch::Tensor C2_pos = C_mode_indices[2][0];
  torch::Tensor C2_crd = C_mode_indices[2][1];
  torch::Tensor C3_pos = C_mode_indices[3][0];
  torch::Tensor C3_crd = C_mode_indices[3][1];

  // Init result level indices
  cvector<int> A2_pos;
  cvector<int> A2_crd;
  A2_pos[0] = 0;
  int pA2 = 0;

  cvector<int> A3_pos;
  cvector<int> A3_crd;
  A3_pos[0] = 0;
  int pA3 = 0;

  // Initialize result value array
  cvector<float> A_values;

  for (int pC0 = C0_pos[0]; pC0 < C0_pos[1]; pC0++) {
    int i = C0_crd[pC0];
    for (int pC1 = C1_pos[pC0]; pC1 < C1_pos[(pC0 + 1)]; pC1++) {
      int j = C1_crd[pC1];
      int pA1 = i * A1_size + j;
      int pB1 = i * B1_size + j;
      int pA2_begin = kA;

      for (int pC2 = C2_pos[pC1]; pC2 < C2_pos[(pC1 + 1)]; pC2++) {
        int k = C2_crd[pC2];
        int pB2 = pB1 * B2_size + k;
        int pA3_begin = pA3;

        for (int pC3 = C3_pos[pC2]; pC3 < C3_pos[(pC2 + 1)]; pC3++) {
          int l = C3_crd[pC3];
          int pB3 = pB2 * B3_size + l;
          A_vals[pA3] = B_vals[pB3] * C_vals[pC3];
          A3_crd[pA3] = l;
          pA3++;
        }

        A3_pos[kA + 1] = pA3;
        if (pA3_begin < pA3) {
          A2_crd[kA] = k;
          kA++;
        }
      }

      A2_pos[pA1 + 1] = kA - pA2_begin;
    }
  }

  int csA2 = 0;
  for (int pA20 = 1; pA20 < (A0_size * A1_size + 1); pA20++) {
    csA2 += A2_pos[pA20];
    A2_pos[pA20] = csA2;
  }

  // Assemble result
  Tensor A;
  torch::Tensor A2_pos_torch = torch::from_blob(
      A2_pos.data(), {A2_pos.size()}, A2_pos.get_deleter(), torch::kInt);
  torch::Tensor A2_crd_torch = torch::from_blob(
      A2_crd.data(), {A2_crd.size()}, A2_crd.get_deleter(), torch::kInt);
  torch::Tensor A3_pos_torch = torch::from_blob(
      A3_pos.data(), {A3_pos.size()}, A3_pos.get_deleter(), torch::kInt);
  torch::Tensor A3_crd_torch = torch::from_blob(
      A3_crd.data(), {A3_crd.size()}, A3_crd.get_deleter(), torch::kInt);
  torch::Tensor A_values_torch =
      torch::from_blob(A_values.data(), {A_values.size()},
                       A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {
      {}, {}, {A2_pos_torch, A2_crd_torch}, {A3_pos_torch, A3_crd_torch}};
  A._storage._value = A_values_torch;
  return A;
}
