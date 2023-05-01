// clang-format off
// taco "A(i, j) = B(i) * C(j)" -f=A:ss -f=B:s -f=C:s -print-evaluate
// Modified for Scorch
// clang-format on

Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, std::vector<int> C_shape,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Get tensor level arrays
  torch::Tensor B0_pos = B_mode_indices[0][0];
  torch::Tensor B0_crd = B_mode_indices[0][1];
  torch::Tensor C0_pos = C_mode_indices[0][0];
  torch::Tensor C0_crd = C_mode_indices[0][1];

  // Init result level indices
  cvector<int> A0_pos;
  cvector<int> A0_crd;
  A0_pos[0] = 0;
  int pA0 = 0;

  cvector<int> A1_pos;
  cvector<int> A1_crd;
  A1_pos[0] = 0;
  int pA1 = 0;

  // Initialize result value array
  cvector<float> A_values;

  for (int pB0 = B0_pos[0]; pB0 < B0_pos[1]; pB0++) {
    int i = B0_crd[pB0];
    int pA1_begin = pA1;

    for (int pC0 = C0_pos[0]; pC0 < C0_pos[1]; pC0++) {
      int j = C0_crd[pC0];
      A_values[pA1] = B_vals[pB0] * C_vals[pC0];
      A1_crd[pA1] = j;
      pA1++;
    }

    A1_pos[iA + 1] = pA1;
    if (pA1_begin < pA1) {
      A0_crd[iA] = i;
      iA++;
    }
  }

  A0_pos[1] = iA;

  // Assemble result
  Tensor A;
  torch::Tensor A0_pos_torch = torch::from_blob(
      A0_pos.data(), {A0_pos.size()}, A0_pos.get_deleter(), torch::kInt);
  torch::Tensor A0_crd_torch = torch::from_blob(
      A0_crd.data(), {A0_crd.size()}, A0_crd.get_deleter(), torch::kInt);
  torch::Tensor A1_pos_torch = torch::from_blob(
      A1_pos.data(), {A1_pos.size()}, A1_pos.get_deleter(), torch::kInt);
  torch::Tensor A1_crd_torch = torch::from_blob(
      A1_crd.data(), {A1_crd.size()}, A1_crd.get_deleter(), torch::kInt);
  torch::Tensor A_values_torch =
      torch::from_blob(A_values.data(), {A_values.size()},
                       A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {{A0_pos_torch, A0_crd_torch},
                                    {A1_pos_torch, A1_crd_torch}};
  A._storage._value = A_values_torch;
  return A;
}
