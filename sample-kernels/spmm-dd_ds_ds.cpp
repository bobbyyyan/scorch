// clang-format off
// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:dd -f=B:ds -f=C:ds -print-evaluate
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
  torch::Tensor B1_pos = B_mode_indices[1][0];
  torch::Tensor B1_crd = B_mode_indices[1][1];
  int C0_size = C_shape[0];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];

  // Initialize result value array
  int A_capacity = A0_size * A1_size;
  cvector<double> A_values = cvector<double>(A_capacity);

#pragma omp parallel for schedule(static)
  for (int pA = 0; pA < A_capacity; pA++) {
    A_values[pA] = 0.0;
  }

#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < B0_size; i++) {
    int pA0 = i;

    for (int pB1 = B1_pos[i]; pB1 < B1_pos[(i + 1)]; pB1++) {
      // Resolve coordinates
      int k = B1_crd[pB1].item<int>();

      for (int pC1 = C1_pos[k]; pC1 < C1_pos[(k + 1)]; pC1++) {
        // Resolve coordinates
        int j = C1_crd[pC1].item<int>();
        int pA1 = pA0 * A1_size + j;

        A_values[pA1] = A_values[pA1] + B_vals[pB1] * C_vals[pC1];
      }
    }
  }

  // Assemble result
  Tensor A;
  torch::Tensor A_values_torch =
      torch::from_blob(A_values.data(), {A_values.size()},
                       A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {{}, {}};
  A._storage._value = A_values_torch;
  return A;
}
