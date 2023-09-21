// taco "y(i) = A(i, j) * x(j)" -f=y:d -f=A:ds -f=x:d -print-evaluate

int evaluate(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int y0_size = (int) (y->dimensions[0]);
  double* restrict y_vals = (double*)(y->vals);
  int A0_size = (int) (A->dimensions[0]);
  int* restrict A1_pos = (int*)(A->indices[1][0]);
  int* restrict A1_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int x1_size = (int)(x->dimensions[0]);
  double* restrict x_vals = (double*)(x->vals);

  int32_t y_capacity = y0_size;
  cvector<double> y_vals = cvector<double>(y_capacity);

  #pragma omp parallel for schedule(runtime)
  for (int i = 0; i < A0_size; i++) {
    double tjy_val = 0.0;
    for (int pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      int j = A1_crd[pA1];
      tjy_val += A_vals[pA1] * x_vals[j];
    }
    y_vals[i] = tjy_val;
  }

  y->vals = (uint8_t*)y_vals;
  return 0;
}
