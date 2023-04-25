// clang-format off
// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:sd -f=B:ds -f=C:ds -print-evaluate
// Modified for Scorch
// clang-format on

Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, std::vector<int> C_shape,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Init result tensor level sizes
  int A1_size = result_shape[1];

  // Get tensor level arrays
  int B0_size = B_shape[0];
  torch::Tensor B1_pos = B_mode_indices[1][0];
  torch::Tensor B1_crd = B_mode_indices[1][1];
  int C0_size = C_shape[0];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];

  // Init result level indices
  cvector<int> A0_pos;
  cvector<int> A0_crd;
  A0_pos[0] = 0;
  int pA0 = 0;

  for (int i = 0; i < B0_size; i++) {
    for (int pB1 = B1_pos[i]; pB1 < B1_pos[(i + 1)]; pB1++) {
      int k = B1_crd[pB1];

      #pragma omp parallel for schedule(static)
      for (int pA = pA0 * A1_size; pA < ((pA0 + 1) * A1_size); pA++) {
        A_vals[pA] = 0.0;
      }

      for (int pC1 = C1_pos[k]; pC1 < C1_pos[(k + 1)]; pC1++) {
        int j = C1_crd[pC1];
        int pA1 = pA0 * A1_size + j;
        A_vals[pA1] = A_vals[pA1] + B_vals[pB1] * C_vals[pC1];
      }
    }
    A0_crd[pA0] = i;
    pA0++;
  }

  A0_pos[1] = pA0;

  A->indices[0][0] = (uint8_t *)(A0_pos);
  A->indices[0][1] = (uint8_t *)(A0_crd);
  A->vals = (uint8_t *)A_vals;
  return 0;
}

  ~                                                                                                    base 
❯
