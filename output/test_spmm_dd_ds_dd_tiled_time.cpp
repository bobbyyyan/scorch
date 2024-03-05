// /Users/guomingfei/Desktop/scorch/tests/test_scorch/test_perf.py:29: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:55.)
//   random_tensor_a_csr = random_tensor_a.to_sparse_csr()
Tensor evaluate(std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Init result tensor level sizes
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
  float* C_values = (float*) malloc(sizeof(float) * C_capacity);
  memset(C_values, 0, sizeof(float) * C_capacity);
   
  // Initialize tile sizes
  constexpr int kTile_k = 1024;
   
   
  for (int i = 0; i < A0_size; i++) {
    // Resolve index into dense level of values array
    int pC0 = i;
     
    for (int k_out = 0; k_out < B1_size; k_out += kTile_k) {
      // Initialize workspaces
      float* accum_c = new float[kTile_k]();
      // Initialize iterators
      int pA1_end = A1_pos[i + 1];
       
      for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int j = A1_crd[pA1];
         
         
        for (int k_in = 0; k_in < kTile_k; k_in++) {
          // Resolve tiled index var
          int k = k_out + k_in;
          // Resolve dense coordinates
          int pB1 = j * B1_size + k;
          accum_c[k_in] += A_val[pA1] * B_val[pB1];
        }
      }
       
      // Lower consumer CIN
      for (int k_in = 0; k_in < kTile_k; k_in++) {
        int k = k_out + k_in;
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += accum_c[k_in];
      }
    }
  }
  // Assemble final result
  Tensor C;
  auto C_values_deleter = [](void* ptr) { free(ptr); };
  torch::Tensor C_values_torch = torch::from_blob(C_values, {C_capacity}, C_values_deleter, torch::kFloat32);
  C._storage._index.mode_indices = {{}, {}};
  C._storage._value = C_values_torch;
  return C;
}
// torch time: 0.5628759860992432
// scorch total time: 5.339833736419678
// scorch eval time: 0.673259973526001
// scorch eval time / torch time: 1.1961071180025356