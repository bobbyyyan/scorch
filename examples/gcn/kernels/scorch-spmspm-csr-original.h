// taco "C(i, k) = A(i, j) * B(j, k)" -f=A:ds -f=B:ds -f=C:ds
#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <torch/torch.h>

#include <vector>

#include "cvector.h"
#include "header.h"
#include "workspace.h"

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

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];
  int* SCORCH_RESTRICT B1_pos = B_mode_indices[1][0].data_ptr<int>();
  int* SCORCH_RESTRICT B1_crd = B_mode_indices[1][1].data_ptr<int>();
  float* SCORCH_RESTRICT B_val = B_values.data_ptr<float>();

  // Init result _level indices
  cvector<int> C1_pos;
  cvector<int> C1_crd;
  C1_pos[0] = 0;
  int pC1 = 0;
  int C1_pos_index = 0;

  for (int pC1 = 1; pC1 <= C0_size; pC1++) {
    C1_pos[pC1] = 0;
  }

  // Initialize result value array
  cvector<float> C_values;

  for (int i = 0; i < A0_size; i++) {
    // Assemble COMPRESSED level
    for (; C1_pos_index < i; C1_pos_index++) {
      C1_pos[C1_pos_index + 1] = C1_crd.size();
    }
    // Resolve index into dense _level of values array
    int pC0 = i;

    // Lower Where statement
    // Initialize workspaces
    coo_workspace<float, 1> wksp = coo_workspace<float, 1>(B1_size);

    for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      int j = A1_crd[pA1];

      for (int pB1 = B1_pos[j]; pB1 < B1_pos[j + 1]; pB1++) {
        int k = B1_crd[pB1];

        wksp.insert({k}, A_val[pA1] * B_val[pB1]);
      }
    }

    // Lower consumer CIN
    wksp.sort();

    for (const auto& it : wksp) {
      int k = it.first;
      float wksp_value = it.second;

      C_values[pC1] = wksp_value;
      C1_crd[pC1] = k;
      pC1++;
    }

    // Assembly compressed _level indices
    C1_pos[C1_pos_index + 1] = C1_crd.size();
  }

  // Assemble final result
  Tensor C;
  torch::Tensor C1_pos_torch = torch::from_blob(
      C1_pos.data(), {C1_pos.size()}, C1_pos.get_deleter(), torch::kInt);
  torch::Tensor C1_crd_torch = torch::from_blob(
      C1_crd.data(), {C1_crd.size()}, C1_crd.get_deleter(), torch::kInt);
  torch::Tensor C_values_torch =
      torch::from_blob(C_values.data(), {C_values.size()},
                       C_values.get_deleter(), torch::kFloat32);
  C._storage._index.mode_indices = {{}, {C1_pos_torch, C1_crd_torch}};
  C._storage._value = C_values_torch;
  return C;
}
