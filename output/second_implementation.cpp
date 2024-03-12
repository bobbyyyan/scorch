Tensor evaluate(std::vector<int> result_shape, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  int A0_stride = 4;
  int A1_stride = 2;
  // Init result tensor level sizes
  int A0_size = result_shape[0]/A0_stride;
   
  // Get B's level & value arrays
  int B0_stride = 4;
  int B1_stride = 2;
  int B0_size = B_shape[0]/B0_stride;
  int B1_size = B_shape[1]/B1_stride;
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
    for (int i_stride = 0; i_stride < B_stride; i_stride++) {
      // here here here! len(iterators) == 0
      // 222
      // Assemble COMPRESSED level
      for (; A1_pos_index < i; A1_pos_index++) {
        A1_pos[A1_pos_index + 1] = A1_crd.size();
      }
      // 333
      // Resolve index into dense level of values array
      int pA0 = i;
      // 444
      // Lower ForAll j
       
      for (int j = 0; j < B1_size; j++) {
        for (int j_stride = 0; j_stride < B_stride; j_stride++) {
          // here here here! len(iterators) == 0
          // 222
          // Resolve dense coordinates
          int pB1 = i * B1_size + j;
          // 333
          // Resolve index into dense level of values array
          // 444
          if (B_val[pB1] != 0) {
            A_values[pA1] = B_val[pB1];
            // Set coordinates
            A1_crd[pA1] = j;
            pA1++;
          }
          // 555
        }
      }
       
      // Assembly compressed _level indices
      A1_pos[A1_pos_index + 1] = A1_crd.size();
      // 555
    }
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