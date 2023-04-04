Tensor evaluate(std::vector<int> result_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Get tensor level arrays
  torch::Tensor B0_crd = B_mode_indices[0][0];
  torch::Tensor B1_crd = B_mode_indices[1][0];

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
  int pB0 = 0;
  int pB0_end = B0_crd.size(0);

  while (pB0 < pB0_end) {
    // Resolve coordinates
    int i0 = B0_crd[pB0].item<int>();

    // Find iterator end for coordinate level
    int pB1_end = pB0 + 1;
    while (pB1_end < pB0_end && B0_crd[pB1_end].item<int>() == i0) {
      pB1_end++;
    }

    // Initialize iterators
    int pB1 = pB0;

    while (pB1 < pB1_end) {
      // Resolve coordinates
      int i1 = B1_crd[pB1].item<int>();

      if (B_values[pB1].item<float>() != 0) {
        A_values[pA1] = B_values[pB1].item<float>();
        // Set coordinates
        A1_crd[pA1] = i1;
        pA1++;
      }
      // Advance iterator
      pB1++;
    }

    // Assembly compressed level indices
    if (A1_pos.back() < pA1) {
      A0_crd.push_back(i0);
    }
    A1_pos[A0_crd.size()] = A1_crd.size();
    // Advance iterator
    pB0 = pB1_end;
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
