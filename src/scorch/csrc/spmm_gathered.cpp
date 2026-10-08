// The v2 SpMM for a gathered call (ScorchGatherCall), in a translation unit of its own: see the
// declarations in spmm.h for why. This file instantiates spmm_csr_v2_core only with a
// ScorchGatherCall, so the ordinary instance ops.cpp compiles has exactly the callers it had.

#define SCORCH_SPMM_KERNELS_ONLY
#include "spmm.h"

torch::Tensor spmm_csr_float_v2_gathered_core(
    int C0_size, int C1_size, int A0_size, const int* SCORCH_RESTRICT A1_pos,
    const int* SCORCH_RESTRICT A1_crd, float* SCORCH_RESTRICT A_val, int B1_size,
    const float* SCORCH_RESTRICT B_val, int tile_size, int nthreads_override, bool atparallel,
    const ScorchGatherCall<float>* g) {
  return spmm_csr_v2_rowsplit<float, ScorchGatherCall<float>>(
      C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val, B1_size, B_val, tile_size,
      nthreads_override, atparallel, g);
}

// The same choice of kernel per host as spmm_csr_double_v2_core. Where that is the reference
// kernel, there are no workers of the product's to fill on, and the fill is serial.
torch::Tensor spmm_csr_double_v2_gathered_core(
    int C0_size, int C1_size, int A0_size, const int* SCORCH_RESTRICT A1_pos,
    const int* SCORCH_RESTRICT A1_crd, double* SCORCH_RESTRICT A_val, int B1_size,
    const double* SCORCH_RESTRICT B_val, int tile_size, int nthreads_override, bool atparallel,
    const ScorchGatherCall<double>* g) {
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__ARM_NEON)
  return spmm_csr_v2_rowsplit<double, ScorchGatherCall<double>>(
      C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val, B1_size, B_val, tile_size,
      nthreads_override, atparallel, g);
#else
  const long nnz = A0_size > 0 ? (long)A1_pos[A0_size] : 0;
  for (long q = 0; q < nnz; ++q) A_val[q] = g->src[g->perm[q]];
  return spmm_csr_double_v2_core(C0_size, C1_size, A0_size, A1_pos, A1_crd, A_val, B1_size,
                                 B_val, tile_size, nthreads_override, atparallel, nullptr);
#endif
}
