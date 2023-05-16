Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Get tensor level arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];

  // Init result level indices
  cvector<int> A0_crd;
  int pA0 = 0;

  cvector<int> A1_crd;
  int pA1 = 0;

  // Initialize result value array
  cvector<float> A_values;

  // Initialize iterators
  int i = 0;

  while (i < B0_size) {
    // Resolve index into dense level of values array
    // Initialize iterators
    int j = 0;

    while (j < B1_size) {
      int pB1 = i * B1_size + j;
      // Resolve index into dense level of values array
      if (B_values[pB1].item<float>() != 0) {
        A_values[pA1] = B_values[pB1].item<float>();
      }

      // Set coordinates
      A1_crd[pA1] = j;
      A0_crd[pA1] = i;
      pA1++;

      // Advance iterators
      j++;
    }

    // Advance iterators
    i++;
  }
  // Assemble result
  Tensor A;
  torch::Tensor A0_crd_torch = torch::from_blob(
      A0_crd.data(), {A0_crd.size()}, A0_crd.get_deleter(), torch::kInt);
  torch::Tensor A1_crd_torch = torch::from_blob(
      A1_crd.data(), {A1_crd.size()}, A1_crd.get_deleter(), torch::kInt);
  torch::Tensor A_values_torch =
      torch::from_blob(A_values.data(), {A_values.size()},
                       A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {{A0_crd_torch}, {A1_crd_torch}};
  A._storage._value = A_values_torch;
  return A;
}
