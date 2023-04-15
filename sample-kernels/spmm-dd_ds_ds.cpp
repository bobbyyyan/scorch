// taco "A(i, j) = B(i, k) * C(k, j)" -f=A:dd -f=B:ds -f=C:ds -print-evaluate

int evaluate(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  int A0_size = (int)(A->dimensions[0]);
  int A1_size = (int)(A->dimensions[1]);
  double* restrict A_vals = (double*)(A->vals);
  int B0_size = (int)(B->dimensions[0]);
  int* restrict B1_pos = (int*)(B->indices[1][0]);
  int* restrict B1_crd = (int*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  int C0_size = (int)(C->dimensions[0]);
  int* restrict C1_pos = (int*)(C->indices[1][0]);
  int* restrict C1_crd = (int*)(C->indices[1][1]);
  double* restrict C_vals = (double*)(C->vals);

  int32_t A_capacity = A0_size * A1_size;
  cvector<double> A_vals = cvector<double>(A_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pA = 0; pA < A_capacity; pA++) {
    A_vals[pA] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i = 0; i < B0_size; i++) {
    for (int32_t pB1 = B1_pos[i]; pB1 < B1_pos[(i + 1)]; pB1++) {
      int32_t k = B1_crd[pB1];
      for (int32_t pC1 = C1_pos[k]; pC1 < C1_pos[(k + 1)]; pC1++) {
        int32_t j = C1_crd[pC1];
        int32_t jA = i * A1_size + j;
        A_vals[jA] = A_vals[jA] + B_vals[pB1] * C_vals[pC1];
      }
    }
  }

  A->vals = (uint8_t*)A_vals;
  return 0;
}
