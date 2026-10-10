// The v2 SpMM for a panel call (ScorchPanelCall), in a translation unit of its own for the reason
// spmm_gathered.cpp is one: see the declarations in spmm.h. This file instantiates
// spmm_csr_v2_core only with a ScorchPanelCall, so the ordinary instance ops.cpp compiles and the
// gathered one keep exactly the callers they had.

#define SCORCH_SPMM_KERNELS_ONLY
#include "spmm.h"

#include <cstring>

void spmm_csr_float_v2_panel_core(int C0_size, int C1_size, int A0_size,
                                  const int* SCORCH_RESTRICT A1_pos,
                                  const int* SCORCH_RESTRICT A1_crd,
                                  const float* SCORCH_RESTRICT A_val, int B1_size,
                                  const float* SCORCH_RESTRICT B_val, int tile_size,
                                  int nthreads_override, bool atparallel, float* out) {
  ScorchPanelCall<float> call;
  call.out = out;
  spmm_csr_v2_rowsplit<float, ScorchPanelCall<float>>(C0_size, C1_size, A0_size, A1_pos, A1_crd,
                                                      A_val, B1_size, B_val, tile_size,
                                                      nthreads_override, atparallel, &call);
}

// The same choice of kernel per host as spmm_csr_double_v2_core. Where that is the reference
// kernel, which makes its own output, the panel's rows are copied into place.
void spmm_csr_double_v2_panel_core(int C0_size, int C1_size, int A0_size,
                                   const int* SCORCH_RESTRICT A1_pos,
                                   const int* SCORCH_RESTRICT A1_crd,
                                   const double* SCORCH_RESTRICT A_val, int B1_size,
                                   const double* SCORCH_RESTRICT B_val, int tile_size,
                                   int nthreads_override, bool atparallel, double* out) {
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__ARM_NEON)
  ScorchPanelCall<double> call;
  call.out = out;
  spmm_csr_v2_rowsplit<double, ScorchPanelCall<double>>(C0_size, C1_size, A0_size, A1_pos,
                                                        A1_crd, A_val, B1_size, B_val, tile_size,
                                                        nthreads_override, atparallel, &call);
#else
  const torch::Tensor whole =
      spmm_csr_double_v2_core(C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val, B1_size, B_val,
                              tile_size, nthreads_override, atparallel, nullptr);
  std::memcpy(out, whole.data_ptr<double>(), (size_t)C0_size * (size_t)C1_size * sizeof(double));
#endif
}
