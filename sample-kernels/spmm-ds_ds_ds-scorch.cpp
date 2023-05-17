// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate

int cmp(const void* a, const void* b) {
  return *((const int*)a) - *((const int*)b);
}

Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values, std::vector<int> C_shape,
                std::vector<std::vector<torch::Tensor>> C_mode_indices,
                torch::Tensor C_values) {
  // Init result tensor level sizes
  int A0_size = result_shape[0];

  // Get tensor level arrays
  int B0_size = B_shape[0];
  torch::Tensor B1_pos = B_mode_indices[1][0];
  torch::Tensor B1_crd = B_mode_indices[1][1];
  int C0_size = C_shape[0];
  torch::Tensor C1_pos = C_mode_indices[1][0];
  torch::Tensor C1_crd = C_mode_indices[1][1];

  cvector<int> A1_pos = cvector<int>(A0_size + 1);
  A1_pos[0] = 0;
  for (int pA1 = 1; pA1 < (A0_size + 1); pA1++) {
    A1_pos[pA1] = 0;
  }

  cvector<int> A1_crd;
  int pA1 = 0;

  cvector<double> A_vals;

  for (int i = 0; i < B0_size; i++) {
    int pA0 = i;

    coo_workspace<double> wksp = coo_workspace<double>(1);

    for (int pB1 = B1_pos[i]; pB1 < B1_pos[i + 1]; pB1++) {
      int k = B1_crd[pB1];
      for (int pC1 = C1_pos[k]; pC1 < C1_pos[k + 1]; pC1++) {
        int j = C1_crd[pC1];

        wksp.insert({j}, B_vals[pB1] * C_vals[pC1]);
      }
    }

    int pA1_begin = pA1;

    auto wksp_map = wksp.get_map();

    for (auto it = wksp_map.begin(); it != wksp_map.end(); ++it) {
      int j = it->first[0];
      double w_val = it->second;

      A_vals[pA1] = w_val;
      A1_crd[pA1] = j;
      pA1++;
    }

    A1_pos.push_back(A1_crd.size());
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
