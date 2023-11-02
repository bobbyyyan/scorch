// taco "C(i, k) = A(i, j) * B(j, k)" -f=A:ds -f=B:ds -f=C:ds
#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define SCORCH_RESTRICT __restrict__

#include <torch/torch.h>

#include <vector>

#include "header.h"

int cmp(const void* a, const void* b) {
  return *((const int*)a) - *((const int*)b);
}

Tensor evaluate(std::vector<int> result_shape, std::vector<int> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values, std::vector<int> B_shape,
                std::vector<std::vector<torch::Tensor>> B_mode_indices,
                torch::Tensor B_values) {
  int C0_size = result_shape[0];

  int A0_size = A_shape[0];
  int* SCORCH_RESTRICT A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT A_val = A_values.data_ptr<float>();

  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  int* SCORCH_RESTRICT B1_pos = B_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT B1_crd = B_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  cvector<int> C1_pos;
  C1_pos[0] = 0;
  for (int pC1 = 1; pC1 < (C0_size + 1); pC1++) {
    C1_pos[pC1] = 0;
  }
  cvector<int> C1_crd;
  int pC1 = 0;
  int C_capacity = 1048576;
  float* C_vals = (float*)malloc(sizeof(float) * C_capacity);

  float* SCORCH_RESTRICT w = 0;
  int* SCORCH_RESTRICT w_index_list = 0;
  w_index_list = (int*)malloc(sizeof(int) * B1_size);
  bool* SCORCH_RESTRICT w_already_set = (bool*)calloc(B1_size, sizeof(bool));
  w = (float*)malloc(sizeof(float) * B1_size);

  for (int i = 0; i < A0_size; i++) {
    int w_index_list_size = 0;
    for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      int j = A1_crd[pA1];
      for (int kB = B1_pos[j]; kB < B1_pos[j + 1]; kB++) {
        int k = B1_crd[kB];
        if (!w_already_set[k]) {
          w[k] = A_val[pA1] * B_val[kB];
          w_index_list[w_index_list_size] = k;
          w_already_set[k] = 1;
          w_index_list_size++;
        } else {
          w[k] = w[k] + A_val[pA1] * B_val[kB];
        }
      }
    }
    qsort(w_index_list, w_index_list_size, sizeof(int), cmp);
    int pC1_begin = pC1;

    for (int w_index_locator = 0; w_index_locator < w_index_list_size;
         w_index_locator++) {
      int k = w_index_list[w_index_locator];
      if (C_capacity <= pC1) {
        C_vals = (float*)realloc(C_vals, sizeof(float) * (C_capacity * 2));
        C_capacity *= 2;
      }
      C_vals[pC1] = w[k];
      C1_crd[pC1] = k;
      pC1++;
      w_already_set[k] = 0;
    }

    C1_pos[i + 1] = pC1 - pC1_begin;
  }

  free(w_index_list);
  free(w_already_set);
  free(w);

  int csC1 = 0;
  for (int pC10 = 1; pC10 < (C0_size + 1); pC10++) {
    csC1 += C1_pos[pC10];
    C1_pos[pC10] = csC1;
  }

  Tensor C;
  int C1_pos_size = C0_size + 1;
  int C1_crd_size = pC1;
  int C_val_size = pC1;
  auto free_deleter = [](void* ptr) { free(ptr); };

  torch::Tensor C1_pos_torch = torch::from_blob(
      C1_pos.data(), {C1_pos.size()}, C1_pos.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(
      C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch =
      torch::from_blob(C_vals, {C_val_size}, free_deleter, torch::kFloat32);
  C._storage._index.mode_indices = {{}, {C1_pos_torch, C1_crd_torch}};
  C._storage._value = C_values_torch;
  return C;
}
