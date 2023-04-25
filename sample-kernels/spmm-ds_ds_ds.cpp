// clang-format off
// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
// Modified for Scorch
// clang-format on

int evaluate(taco_tensor_t* A, taco_tensor_t* B, taco_tensor_t* C) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int* restrict C2_pos = (int*)(C->indices[1][0]);
  int* restrict C2_crd = (int*)(C->indices[1][1]);
  double* restrict C_vals = (double*)(C->vals);

  A2_pos = (int32_t*)malloc(sizeof(int32_t) * (A1_dimension + 1));
  A2_pos[0] = 0;
  for (int32_t pA2 = 1; pA2 < (A1_dimension + 1); pA2++) {
    A2_pos[pA2] = 0;
  }
  int32_t A2_crd_size = 1048576;
  A2_crd = (int32_t*)malloc(sizeof(int32_t) * A2_crd_size);
  int32_t jA = 0;
  int32_t A_capacity = 1048576;
  A_vals = (double*)malloc(sizeof(double) * A_capacity);

  double* restrict w = 0;
  int32_t* restrict w_index_list = 0;
  w_index_list = (int32_t*)malloc(sizeof(int32_t) * C2_dimension);
  bool* restrict w_already_set = calloc(C2_dimension, sizeof(bool));
  w = (double*)malloc(sizeof(double) * C2_dimension);

  for (int32_t i = 0; i < B1_dimension; i++) {
    int32_t w_index_list_size = 0;
    for (int32_t kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
      int32_t k = B2_crd[kB];
      for (int32_t jC = C2_pos[k]; jC < C2_pos[(k + 1)]; jC++) {
        int32_t j = C2_crd[jC];
        if (!w_already_set[j]) {
          w[j] = B_vals[kB] * C_vals[jC];
          w_index_list[w_index_list_size] = j;
          w_already_set[j] = 1;
          w_index_list_size++;
        } else {
          w[j] = w[j] + B_vals[kB] * C_vals[jC];
        }
      }
    }
    qsort(w_index_list, w_index_list_size, sizeof(int32_t), cmp);
    int32_t pA2_begin = jA;

    for (int32_t w_index_locator = 0; w_index_locator < w_index_list_size;
         w_index_locator++) {
      int32_t j = w_index_list[w_index_locator];
      if (A_capacity <= jA) {
        A_vals = (double*)realloc(A_vals, sizeof(double) * (A_capacity * 2));
        A_capacity *= 2;
      }
      A_vals[jA] = w[j];
      if (A2_crd_size <= jA) {
        A2_crd = (int32_t*)realloc(A2_crd, sizeof(int32_t) * (A2_crd_size * 2));
        A2_crd_size *= 2;
      }
      A2_crd[jA] = j;
      jA++;
      w_already_set[j] = 0;
    }

    A2_pos[i + 1] = jA - pA2_begin;
  }

  free(w_index_list);
  free(w_already_set);
  free(w);

  int32_t csA2 = 0;
  for (int32_t pA20 = 1; pA20 < (A1_dimension + 1); pA20++) {
    csA2 += A2_pos[pA20];
    A2_pos[pA20] = csA2;
  }

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

  ~                                                                                                    base 
❯
