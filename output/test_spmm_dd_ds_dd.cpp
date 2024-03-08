 Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Init result tensor level sizes
  int A0_size = result_shape[0];
   
  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  float* B_val = B_values.data_ptr<float>();
   
  // Init result level indices
  cvector<int> A1_pos;
  cvector<int> A1_crd;
  A1_pos[0] = 0;
  int pA1 = 0;
  int A1_pos_index = 0;
   
  for (int pA1 = 1; pA1 <= A0_size; pA1++) {
    A1_pos[pA1] = 0;
  }
  // Initialize result value array
  cvector<float> A_values;
   
  // Lower ForAll i
   
  for (int i = 0; i < B0_size; i++) {
    // Assemble COMPRESSED level
    for (; A1_pos_index < i; A1_pos_index++) {
      A1_pos[A1_pos_index + 1] = A1_crd.size();
    }
    // Resolve index into dense level of values array
    int pA0 = i;
    // Lower ForAll j
     
    for (int j = 0; j < B1_size; j++) {
      // Resolve dense coordinates
      int pB1 = i * B1_size + j;
      // Resolve index into dense level of values array
      if (B_val[pB1] != 0) {
        A_values[pA1] = B_val[pB1];
        // Set coordinates
        A1_crd[pA1] = j;
        pA1++;
      }
    }
     
    // Assembly compressed _level indices
    A1_pos[A1_pos_index + 1] = A1_crd.size();
  }
  // Assemble final result
  Tensor A;
  torch::Tensor A1_pos_torch = torch::from_blob(A1_pos.data(), {A1_pos.size()}, A1_pos.get_deleter(), torch::kInt);
  torch::Tensor A1_crd_torch = torch::from_blob(A1_crd.data(), {A1_crd.size()}, A1_crd.get_deleter(), torch::kInt);
  torch::Tensor A_values_torch = torch::from_blob(A_values.data(), {A_values.size()}, A_values.get_deleter(), torch::kFloat32);
  A._storage._index.mode_indices = {{}, {A1_pos_torch, A1_crd_torch}};
  A._storage._value = A_values_torch;
  return A;
}

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
  constexpr int kTile_k = 32;
   
   
  for (int k_out = 0; k_out < B1_size; k_out += kTile_k) {
     
    for (int i = 0; i < A0_size; i++) {
      // Resolve index into dense level of values array
      int pC0 = i;
      // Initialize workspaces
      float* wksp = new float[kTile_k]();
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
          wksp[k_in] += A_val[pA1] * B_val[pB1];
        }
      }
       
      // Lower consumer CIN
      for (int k_in = 0; k_in < kTile_k; k_in++) {
        int k = k_out + k_in;
        int pC1 = pC0 * C1_size + k;
        C_values[pC1] += wksp[k_in];
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


int pB1 = i * B0_stride * B1_size * B1_stride + j * B1_stride;
for (int i0 = 0; i0 < B0_stride; i0++) {
  for (int i1 = 0; i1 < B1_stride; i1++) {
    int delta = i0 * B1_size * B1_stride + i1;
  }