// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate

int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}

int evaluate(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  int A0_size = (int)(A->dimensions[0]);
  int* A1_pos = (int*)(A->indices[1][0]);
  int* A1_crd = (int*)(A->indices[1][1]);
  double* A_vals = (double*)(A->vals);
  int B0_size = (int)(B->dimensions[0]);
  int* B1_pos = (int*)(B->indices[1][0]);
  int* B1_crd = (int*)(B->indices[1][1]);
  double* B_vals = (double*)(B->vals);
  int C0_size = (int)(C->dimensions[0]);
  int C1_size = (int)(C->dimensions[1]);
  int* C1_pos = (int*)(C->indices[1][0]);
  int* C1_crd = (int*)(C->indices[1][1]);
  double* C_vals = (double*)(C->vals);

  cvector<int> A1_pos = cvector<int>(A0_size + 1);
  A1_pos[0] = 0;
  for (int32_t pA1 = 1; pA1 < (A0_size + 1); pA1++) {
    A1_pos[pA1] = 0;
  }

  cvector<int> A1_crd;
  int32_t pA1 = 0;

  cvector<double> A_vals;

  double* w = 0;
  int32_t* w_index_list = 0;
  cvector<int> w_index_list = cvector<int>(C1_size);
  bool* w_already_set = calloc(C1_size, sizeof(bool));
  cvector<double> w = cvector<double>(C1_size);
  coo_workspace<double> wksp = coo_workspace<double>(1);

  for (int32_t i = 0; i < B0_size; i++) {
    int32_t w_index_list_size = 0;

    for (int32_t pB1 = B1_pos[i]; pB1 < B1_pos[i + 1]; pB1++) {
      int32_t k = B1_crd[pB1];
      for (int32_t pC1 = C1_pos[k]; pC1 < C1_pos[k + 1]; pC1++) {
        int32_t j = C1_crd[pC1];

        wksp.insert({j}, B_vals[pB1] * C_vals[pC1]);
      }
    }

    int32_t pA1_begin = pA1;

    auto wksp_map = wksp.get_map();
    for (auto it=wksp_map.begin(); it != wksp_map.end(); ++it) {
      int32_t j = it->first[0];
      double w_val = it->second;

      A_vals[pA1] = w_val;
      A1_crd[pA1] = j;
      pA1++;
    }
    A1_pos[i + 1] = pA1 - pA1_begin;
  }

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
