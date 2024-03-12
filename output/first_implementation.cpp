/Users/guomingfei/Desktop/scorch/tests/test_scorch/test_strided.py:112: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:55.)
  random_tensor_a_csr = random_tensor_a.to_sparse_csr()
self.format.get_level_stride_sizes() [4, 2]


tensor tensor tensor cin_stmt:  ∀{i} (∀{j} (A:d,s[i, j] = B:d,d[i, j]))
get_lattice_points_from_cin cin:  B:d,d[i, j]
get_lattice_points_from_cin cin:  B:d,d[i, j]


tensor tensor tensor cpp_code:

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
      // here here here! len(iterators) == 0
      // 222
      // Resolve dense coordinates
      int pB1 = i * B1_size * B0_stride * B1_stride + j * B1_stride;
      bool stride_non_zero = 1;
      cvector<int> B1_stride_array;
      for (int i0 = 0; i0 < B0_stride; i0++) {
        for (int i1 = 0; i1 < B1_stride; i1++) {
          int p_stride = i0 * B1_stride * B1_size + i1;
          int save_to_array_idx = i0 * B1_stride + i1;
          if (B_val[pB1 + p_stride] != 0) {
            stride_non_zero = 1;
          }
          B1_stride_array[save_to_array_idx] = pB1 + p_stride;
        }
      }
      // 333
      // Resolve index into dense level of values array
      // 444
      if (stride_non_zero != 0) {
        int A_stride_area = A0_stride * A1_stride;
        for (int iA_stride_area = 0; iA_stride_area < A_stride_area; iA_stride_area++) {
          A_values[pA1 * A_stride_area + iA_stride_area] = B_val[B1_stride_array[iA_stride_area]];
        }
        // Set coordinates
        A1_crd[pA1] = j;
        pA1++;
      }
      // 555
    }
     
    // Assembly compressed _level indices
    A1_pos[A1_pos_index + 1] = A1_crd.size();
    // 555
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
CIN:
 ∀{i} (∀{j} (∀{k} (C:d,d[i, k] = (A:d,s[i, j] * B:d,d[j, k]))))
Auto-scheduling CIN statement
∀{i} (Where(
        producer=∀{j} (∀{k} (wksp:d(dim=1)[[ivar_k]] = (A:d,s[i, j] * B:d,d[j, k]))), 
        consumer=∀{j} (∀{k} (C:d,d[i, k] = wksp:d(dim=1)[[ivar_k]]))
))
Index vars: [ivar_i, ivar_k, ivar_j]
Tensor accesses: [A:d,s[i, j], B:d,d[j, k], C:d,d[i, k]]
Index vars to tile: [ivar_k]
Auto-scheduled CIN:
 ∀{k_out} (∀{i} (Where(
        producer=∀{j} (∀{k_in} (wksp:d(dim=1)[[ivar_k_in]] = (A:d,s[i, j] * B:d,d[j, k]))), 
        consumer=∀{j} (∀{k_in} (C:d,d[i, k] = wksp:d(dim=1)[[ivar_k_in]]))
)))
get_lattice_points_from_cin cin:  A:d,s[i, j]
get_lattice_points_from_cin cin:  B:d,d[j, k]
get_lattice_points_from_cin cin:  A:d,s[i, j]
get_lattice_points_from_cin cin:  B:d,d[j, k]
get_lattice_points_from_cin cin:  A:d,s[i, j]
get_lattice_points_from_cin cin:  B:d,d[j, k]
get_lattice_points_from_cin cin:  A:d,s[i, j]
get_lattice_points_from_cin cin:  B:d,d[j, k]


 Tensor evaluate(std::vector<int> result_shape, std::vector<int> A_shape, std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values, std::vector<int> B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values) {
  // Init result tensor level sizes
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];
   
  // Get A's level & value arrays
  int A0_stride = 4;
  int A1_stride = 2;
  int A0_size = A_shape[0]/A0_stride;
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
   
  // Lower ForAll k_out
   
  for (int k_out = 0; k_out < B1_size; k_out += kTile_k) {
    // Lower ForAll i
     
    for (int i = 0; i < A0_size; i++) {
      // here here here! len(iterators) == 0
      // 222
      // 333
      // Resolve index into dense level of values array
      int pC0 = i;
      // 444
      // Initialize workspaces
      float* wksp = new float[kTile_k]();
      // Lower ForAll j
      // Initialize iterators
      int pA1_end = A1_pos[i + 1];
       
      for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int j = A1_crd[pA1];
         
        // Lower ForAll k_in
         
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
      // 555
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