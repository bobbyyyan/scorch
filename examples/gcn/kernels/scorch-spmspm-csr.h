// taco "C(i, k) = A(i, j) * B(j, k)" -f=A:ds -f=B:ds -f=C:ds
#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define SCORCH_RESTRICT __restrict__

#include <torch/torch.h>

#include <vector>

int cmp(const void* a, const void* b) {
  return *((const int*)a) - *((const int*)b);
}

Tensor evaluate(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  // Init result tensor _level sizes
  int C0_size = result_shape[0];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  int* SCORCH_RESTRICT B1_pos = B_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT B1_crd = B_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  std::vector<int> C1_pos(C0_size + 1, 0);
  std::vector<int> C1_crd;
  int pC1 = 0;

  for (int pC1 = 1; pC1 < (C0_size + 1); pC1++) {
    C1_pos[pC1] = 0;
  }

  std::vector<float> C_values;

  std::vector<float> w(B1_size);
  std::vector<int> w_index_list(B1_size);
  std::vector<uint8_t> w_already_set(B1_size, 0);

  for (int i = 0; i < A0_size; i++) {
    int w_index_list_size = 0;
    for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      int j = A1_crd[pA1];
      for (int pB1 = B1_pos[j]; pB1 < B1_pos[j + 1]; pB1++) {
        int k = B1_crd[pB1];
        if (!w_already_set[k]) {
          w[k] = A_val[pA1] * B_val[pB1];
          w_index_list[w_index_list_size] = k;
          w_already_set[k] = 1;
          w_index_list_size++;
        } else {
          w[k] = w[k] + A_val[pA1] * B_val[pB1];
        }
      }
    }
    qsort(w_index_list.data(), w_index_list_size, sizeof(int), cmp);
    int pC1_begin = pC1;

    for (int w_idx = 0; w_idx < w_index_list_size; w_idx++) {
      int k = w_index_list[w_idx];

      C_values.push_back(w[k]);
      C1_crd.push_back(k);
      pC1++;
      w_already_set[k] = 0;
    }

    C1_pos[i + 1] = pC1 - pC1_begin;
  }

  int csC1 = 0;
  for (int pC10 = 1; pC10 < (C0_size + 1); pC10++) {
    csC1 += C1_pos[pC10];
    C1_pos[pC10] = csC1;
  }

  // Assemble final result
  Tensor C;
  torch::Tensor C1_pos_torch =
      scorch_tensor_from_vector(std::move(C1_pos), torch::kInt);
  torch::Tensor C1_crd_torch =
      scorch_tensor_from_vector(std::move(C1_crd), torch::kInt);
  torch::Tensor C_values_torch =
      scorch_tensor_from_vector(std::move(C_values), torch::kFloat32);
  C.storage.index.mode_indices = {{}, {C1_pos_torch, C1_crd_torch}};
  C.storage.value = C_values_torch;
  return C;
}
