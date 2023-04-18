// taco "A(i, j) = B(i, j) * C(i, j)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
// Modified for scorch

Tensor evaluate(std::vector<int> result_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Init result tensor level sizes
  int A0_size = result_shape[0];
  int A1_size = result_shape[1];

  // Get tensor level arrays
  int B0_size = B._shape[0];
  int B1_size = B._shape[1];
  int C0_size = C._shape[0];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];

  int A_capacity = A0_size * A1_size;
  A_vals = (double*)malloc(sizeof(double) * A_capacity);

  // Initialize result value array
  cvector<float> A_values;

  // Assemble dense result level as needed
  int A_stride = A0_size * A1_size;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < A_stride; i++) {
    A_values[i] = 0;
  }

#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < C0_size; i++) {
    for (int pC1 = C1_pos[i]; pC1 < C1_pos[(i + 1)]; pC1++) {
      int j = C1_crd[pC1];
      int pA1 = i * A1_size + j;
      int pB1 = i * B1_size + j;
      A_values[pA1] = B_vals[pB1] * C_vals[pC1];
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
