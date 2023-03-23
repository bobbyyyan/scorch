// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate

int evaluate(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  int A0_size = (int)(A->dimensions[0]);
  int* restrict A1_pos = (int*)(A->indices[1][0]);
  int* restrict A1_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int B0_size = (int)(B->dimensions[0]);
  int* restrict B1_pos = (int*)(B->indices[1][0]);
  int* restrict B1_crd = (int*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  int C0_size = (int)(C->dimensions[0]);
  int C1_size = (int)(C->dimensions[1]);
  int* restrict C1_pos = (int*)(C->indices[1][0]);
  int* restrict C1_crd = (int*)(C->indices[1][1]);
  double* restrict C_vals = (double*)(C->vals);

  cvector<int> A1_pos = cvector<int>(A0_size + 1);
  A1_pos[0] = 0;
  for (int32_t pA1 = 1; pA1 < (A0_size + 1); pA1++) {
    A1_pos[pA1] = 0;
  }

  cvector<int> A1_crd;
  int32_t pA1 = 0;

  cvector<double> A_vals;

  double* restrict w = 0;
  int32_t* restrict w_index_list = 0;
  cvector<int> w_index_list = cvector<int> (C1_size);
  bool* restrict w_already_set = calloc(C1_size, sizeof(bool));
  cvector<double> w = cvector<double>(C1_size);

  for (int32_t i = 0; i < B0_size; i++) {
    int32_t w_index_list_size = 0;
    for (int32_t kB = B1_pos[i]; kB < B1_pos[i + 1]; kB++) {
      int32_t k = B1_crd[kB];
      for (int32_t jC = C1_pos[k]; jC < C1_pos[k + 1]; jC++) {
        int32_t j = C1_crd[jC];
        if (!w_already_set[j]) {
          w[j] = B_vals[kB] * C_vals[jC];
          w_index_list[w_index_list_size] = j;
          w_already_set[j] = 1;
          w_index_list_size++;
        }
        else {
          w[j] = w[j] + B_vals[kB] * C_vals[jC];
        }
      }
    }
    qsort(w_index_list, w_index_list_size, sizeof(int32_t), cmp);
    int32_t pA1_begin = pA1;

    for (int32_t w_index_locator = 0; w_index_locator < w_index_list_size; w_index_locator++) {
      int32_t j = w_index_list[w_index_locator];
      A_vals[pA1] = w[j];
      A1_crd[pA1] = j;
      pA1++;
      w_already_set[j] = 0;
    }

    A1_pos[i + 1] = pA1 - pA1_begin;
  }

  free(w_index_list);
  free(w_already_set);
  free(w);

  int32_t csA1 = 0;
  for (int32_t pA10 = 1; pA10 < (A0_size + 1); pA10++) {
    csA1 += A1_pos[pA10];
    A1_pos[pA10] = csA1;
  }

  A->indices[1][0] = (uint8_t*)(A1_pos);
  A->indices[1][1] = (uint8_t*)(A1_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}
