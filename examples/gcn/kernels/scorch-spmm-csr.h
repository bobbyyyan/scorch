Tensor evaluate(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Init result tensor _level sizesÏ
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();
  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  // Use Torch API to create output
  auto options = torch::TensorOptions()
                     .dtype(A_values.scalar_type())
                     .device(A_values.device());
  auto C_values = torch::empty({C0_size, C1_size}, options);
  float* C_val = C_values.data_ptr<float>();

  constexpr int kTileN = 4;
  // TODO: Relax this constraint
  TORCH_CHECK((B1_size % kTileN) == 0);

  for (int i = 0; i < A0_size; i++) {
    int pC0 = i;

    // Initialize iterators
    int pA1_start = A1_pos[i];
    int pA1_end = A1_pos[i + 1];

    for (int outer_j = 0; outer_j < B1_size; outer_j += kTileN) {
      float accumulator[kTileN] = {};
      for (int pA1 = pA1_start; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int k = A1_crd[pA1];

        for (int inner_j = 0; inner_j < kTileN; inner_j++) {
          int j = outer_j + inner_j;
          int pB1 = k * B1_size + j;
          accumulator[inner_j] += A_val[pA1] * B_val[pB1];
        }
      }

      // Flush the accumulator
      for (int inner_j = 0; inner_j < kTileN; inner_j++) {
        int j = outer_j + inner_j;
        int pC1 = pC0 * C1_size + j;
        C_val[pC1] = accumulator[inner_j];
      }
    }
  }

  // Assemble final result
  Tensor C;
  C._storage._index.mode_indices = {{}, {}};
  C._storage._value = C_values;
  return C;
}