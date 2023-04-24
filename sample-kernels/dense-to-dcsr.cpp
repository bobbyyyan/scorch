// clang-format off
// taco "A(i, j) = B(i, j)" -f=A:ss -f=B:dd -print-evaluate
// Modified for Scorch
// clang-format on

Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Get tensor level arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];

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

  // Initialize iterators
  int i = 0;

  while (i < B0_size) {
    // Initialize iterators
    int j = 0;

    while (j < B1_size) {
      int pB1 = i * B1_size + j;
      if (B_values[pB1].item<float>() != 0) {
        A_values[pA1] = B_values[pB1].item<float>();
        // Set coordinates
        A1_crd[pA1] = j;
        pA1++;
      }

      // Advance iterators
      j++;
    }

    // Assembly compressed level indices
    if (A1_pos.back() < pA1) {
      A0_crd.push_back(i);
    }
    A1_pos[A0_crd.size()] = A1_crd.size();

    // Advance iterators
    i++;
  }

  // Assembly compressed level indices
  A0_pos.push_back(A0_crd.size());
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
