#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

Tensor spmm_csr_float(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  float* SCORCH_RESTRICT C_values = (float *)malloc(sizeof(float) * C_capacity);
  memset(C_values, 0, sizeof(float) * C_capacity);

  // Initialize tile sizes
  constexpr int kTile_k = 4096;

  int residual_k_start = (B1_size / kTile_k) * kTile_k;

  for (int i = 0; i < A0_size; i++) {
    // Resolve index into dense level of values array
    int pC0 = i;

    for (int k_out = 0; k_out < residual_k_start; k_out += kTile_k) {
      // Initialize workspaces
      float *accum_c = new float[kTile_k]();
      // Initialize iterators
      int pA1_end = A1_pos[i + 1];

      for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int j = A1_crd[pA1];

        SCORCH_PRAGMA_UNROLL
        for (int k_in = 0; SCORCH_LIKELY(k_in < kTile_k); k_in++) {
          // Resolve tiled index var
          int k = k_out + k_in;
          // Resolve dense coordinates
          int pB1 = j * B1_size + k;
          accum_c[k_in] += A_val[pA1] * B_val[pB1];
        }
      }

      // Lower consumer CIN
      for (int k_in = 0; SCORCH_LIKELY(k_in < kTile_k); k_in++) {
        int k = k_out + k_in;
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += accum_c[k_in];
      }
    }
  }

  if (residual_k_start < B1_size) {
    for (int i = 0; i < A0_size; i++) {
      int pC0 = i;
      int tile_k_width = B1_size - residual_k_start;

      float *accum_c = new float[tile_k_width]();
      int pA1_end = A1_pos[i + 1];

      for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
        int j = A1_crd[pA1];

        for (int k = residual_k_start; k < B1_size; k++) {
          int pB1 = j * B1_size + k;
          accum_c[k - residual_k_start] += A_val[pA1] * B_val[pB1];
        }
      }

      for (int k = residual_k_start; k < B1_size; k++) {
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += accum_c[k - residual_k_start];
      }
      delete[] accum_c;
    }
  }
  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void *ptr) {
    { free(ptr); }
  };
  torch::Tensor C_values_torch = torch::from_blob(
      C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = C_values_torch;
  return C;
}
