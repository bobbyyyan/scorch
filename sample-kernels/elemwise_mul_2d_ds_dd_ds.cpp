// taco "A(i, j) = B(i, j) * C(i, j)" -f=A:ds -f=B:dd -f=C:ds -print-evaluate
// Modified for Scorch

Tensor evaluate(std::vector<int> result_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Init result tensor level sizes
  int A0_size = result_shape[0];

  // Get tensor level arrays
  int B0_size = B._shape[0];
  int B1_size = B._shape[1];
  int C0_size = C._shape[0];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];

  // Init result level indices
  cvector<int> A1_pos;
  cvector<int> A1_crd;
  A1_pos[0] = 0;
  int pA1 = 0;

  // Initialize result value array
  cvector<float> A_values;

  // Initialize iterators
  int i = 0;

  while (i < B0_size) {
    // Initialize iterators
    int pC1 = C1_pos[i];
    int pC1_end = C1_pos[i + 1];

    while (pC1 < pC1_end) {
      // Resolve coordinates
      int j = C1_crd[pC1].item<int>();
      int pB1 = i * B1_size + j;

      A_values[pA1] = B_values[pB1].item<float>() * C_values[pC1].item<float>();
      // Set coordinates
      A1_crd[pA1] = j;
      pA1++;
      // Advance iterator
      pC1++;
    }

    // Assembly compressed level indices
    A1_pos.push_back(A1_crd.size());

    // Advance iterators
    i++;
  }
  // Assemble result
  Tensor A;
  torch::Tensor A1_pos_torch = torch::from_blob(
      A1_pos.data(), {A1_pos.size()}, A1_pos.get_deleter(), torch::kInt);
  torch::Tensor A1_crd_torch = torch::from_blob(
      A1_crd.data(), {A1_crd.size()}, A1_crd.get_deleter(), torch::kInt);
  torch::Tensor A_values_torch =
      torch::from_blob(A_values.data(), {A_values.size()},
                       A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {{}, {A1_pos_torch, A1_crd_torch}};
  A._storage._value = A_values_torch;
  return A;
}

== == == == == == == == == == == == == == == 1 passed in 7.45s == == == == == ==
    == == == == == == == == ==
    =

        Process finished with exit code 0
