#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <cstdlib>
#include <vector>

void evaluate(std::vector<int> result_shape,
                std::vector<int> A_shape,
                const int* SCORCH_RESTRICT A1_pos,
                const int* SCORCH_RESTRICT A1_crd,
                const float* SCORCH_RESTRICT A_val,
                std::vector<int> B_shape,
                const float* SCORCH_RESTRICT B_val,
                float* SCORCH_RESTRICT C_val
                ) {
  // Init result tensor _level sizesÏ
  int C0_size = result_shape[0];
  int C1_size = result_shape[1];

  // Get A's level & value arrays
  int A0_size = A_shape[0];

  // Get B's level & value arrays
  int B0_size = B_shape[0];
  int B1_size = B_shape[1];

  // Initialize result value array
  int C_capacity = C0_size * C1_size;
  // float* SCORCH_RESTRICT C_val = (float*)malloc(sizeof(float) * C_capacity);

  constexpr int kTileN = 1024;

  for (int i = 0; SCORCH_LIKELY(i < A0_size); i++) {
    int pC0 = i;

    // Initialize iterators
    int pA1_start = A1_pos[i];
    int pA1_end = A1_pos[i + 1];

    for (int outer_j = 0; outer_j < B1_size; outer_j += kTileN) {
      float accumulator[kTileN] = {};
      // float* SCORCH_RESTRICT accumulator = new float[kTileN]();

      for (int pA1 = pA1_start; pA1 < pA1_end; pA1++) {
        // Resolve coordinates
        int k = A1_crd[pA1];

        // Unroll the inner loop
        SCORCH_PRAGMA_UNROLL
        for (int inner_j = 0; inner_j < kTileN; inner_j++) {
          int j = outer_j + inner_j;
          int pB1 = k * B1_size + j;
          accumulator[inner_j] += A_val[pA1] * B_val[pB1];
        }
      }

      // Flush the accumulator
      SCORCH_PRAGMA_UNROLL
      for (int inner_j = 0; inner_j < kTileN; inner_j++) {
        int j = outer_j + inner_j;
        int pC1 = pC0 * C1_size + j;
        C_val[pC1] = accumulator[inner_j];
      }

      // Deallocate memory to prevent memory leak
      // delete[] accumulator;
    }
  }

  // Assemble final result
  // Tensor C;
  // C._storage._index.mode_indices = {{}, {}};
  // C._storage._value = C_val;
  // return C;
}
