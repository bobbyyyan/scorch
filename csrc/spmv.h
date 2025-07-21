#define SCORCH_PRAGMA_UNROLL _Pragma("unroll")
#define SCORCH_LIKELY(x) __builtin_expect(!!(x), 1)
#define SCORCH_UNLIKELY(x) (x)
#define SCORCH_RESTRICT __restrict__

#include <vector>
#include <torch/torch.h>
#include <cstring>
#include <omp.h>
#include <arm_neon.h>


Tensor spmv_csr_float_v4(const std::vector<int>& result_shape,
                      const std::vector<int>& A_shape,
                      const std::vector<std::vector<torch::Tensor>>& A_mode_indices,
                      const torch::Tensor& A_values,
                      const std::vector<int>& x_shape,
                      const std::vector<std::vector<torch::Tensor>>& x_mode_indices,
                      const torch::Tensor& x_values) {
    int y0_size = result_shape[0];
    int A0_size = A_shape[0];
    int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
    int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
    float* A_val = A_values.data_ptr<float>();
    float* x_val = x_values.data_ptr<float>();
    float* y_values = new float[y0_size]();

    #pragma omp parallel for
    for (int i = 0; i < A0_size; i++) {
        float accumulator = 0.0;
        int row_start = A1_pos[i];
        int row_end = A1_pos[i + 1];

        // Software prefetch
        __builtin_prefetch(&A_val[row_start], 0, 3);
        __builtin_prefetch(&x_val[A1_crd[row_start]], 0, 3);

        for (int idx = row_start; idx < row_end; idx++) {
            accumulator += A_val[idx] * x_val[A1_crd[idx]];
            // Manual loop unrolling
            if (idx + 1 < row_end) {
                accumulator += A_val[idx + 1] * x_val[A1_crd[idx + 1]];
                idx++;
            }
        }
        y_values[i] = accumulator;
    }

    torch::Tensor y_values_torch = torch::from_blob(y_values, {y0_size}, torch::kFloat32);
    Tensor y;
    y.storage.value = y_values_torch;
    return y;
}


Tensor spmv_csr_float_v3(const std::vector<int>& result_shape,
                      const std::vector<int>& A_shape,
                      const std::vector<std::vector<torch::Tensor>>& A_mode_indices,
                      const torch::Tensor& A_values,
                      const std::vector<int>& x_shape,
                      const std::vector<std::vector<torch::Tensor>>& x_mode_indices,
                      const torch::Tensor& x_values) {
    // Retrieve sizes from shapes
    int y0_size = result_shape[0];
    int A0_size = A_shape[0];

    // Retrieve pointers to data arrays
    int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
    int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
    float* A_val = A_values.data_ptr<float>();
    float* x_val = x_values.data_ptr<float>();

    // Allocate and initialize result values array
    float* y_values = new float[y0_size](); // Value-initialized to zero

    // Main SpMV computation
    for (int i = 0; i < A0_size; i++) {
        float32x4_t acc_vec = vdupq_n_f32(0.0f); // Initialize vector accumulator
        int row_start = A1_pos[i];
        int row_end = A1_pos[i + 1];
        int idx = row_start;

        // Process four elements at a time
        for (; idx <= row_end - 4; idx += 4) {
            float32x4_t a_vals = vld1q_f32(&A_val[idx]);
            // Manually gather x_vals
            float32x4_t x_vals = {x_val[A1_crd[idx]], x_val[A1_crd[idx+1]], x_val[A1_crd[idx+2]], x_val[A1_crd[idx+3]]};

            acc_vec = vmlaq_f32(acc_vec, a_vals, x_vals);
        }

        // Reduce vector sum to scalar
        float accumulator = vaddvq_f32(acc_vec);

        // Handle remaining elements
        for (; idx < row_end; idx++) {
            accumulator += A_val[idx] * x_val[A1_crd[idx]];
        }

        y_values[i] = accumulator;
    }

    // Create a torch::Tensor from y_values
    auto y_values_deleter = [](void* ptr) { delete[] static_cast<float*>(ptr); };
    torch::Tensor y_values_torch = torch::from_blob(y_values, {y0_size}, y_values_deleter, torch::kFloat32);

    Tensor y;
    y.storage.value = y_values_torch;
    y.storage.index.mode_indices = std::vector<std::vector<torch::Tensor>>();

    return y;
}

Tensor spmv_csr_float_v2(const std::vector<int>& result_shape,
                      const std::vector<int>& A_shape,
                      const std::vector<std::vector<torch::Tensor>>& A_mode_indices,
                      const torch::Tensor& A_values,
                      const std::vector<int>& x_shape,
                      const std::vector<std::vector<torch::Tensor>>& x_mode_indices,
                      const torch::Tensor& x_values) {
    // Retrieve sizes from shapes
    int y0_size = result_shape[0];
    int A0_size = A_shape[0];

    // Retrieve pointers to data arrays
    int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
    int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
    float* A_val = A_values.data_ptr<float>();
    float* x_val = x_values.data_ptr<float>();

    // Allocate and initialize result values array
    float* y_values = new float[y0_size](); // Value-initialized to zero

    // Main SpMV computation
    for (int i = 0; i < A0_size; i++) {
        float accumulator = 0.0f;
        int row_start = A1_pos[i];
        int row_end = A1_pos[i + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int col_idx = A1_crd[idx];
            float a_val = A_val[idx];
            float x_val_at_idx = x_val[col_idx];

            accumulator += a_val * x_val_at_idx;
        }

        y_values[i] = accumulator;
    }

    // Create a torch::Tensor from y_values
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor y_tensor = torch::from_blob(y_values, {y0_size}, options);

    // Create a torch::Tensor from y_values
    auto y_values_deleter = [](void* ptr) { delete[] static_cast<float*>(ptr); };
    torch::Tensor y_values_torch = torch::from_blob(y_values, {y0_size}, y_values_deleter, torch::kFloat32);

    Tensor y;
    y.storage.value = y_values_torch;

    return y;
}

Tensor spmv_csr_float(std::vector<int> result_shape,
                      std::vector<int> A_shape,
                      std::vector<std::vector<torch::Tensor>> A_mode_indices,
                      torch::Tensor A_values,
                      std::vector<int> x_shape,
                      std::vector<std::vector<torch::Tensor>> x_mode_indices,
                      torch::Tensor x_values) {
  // Init result tensor level sizes
  int y0_size = result_shape[0];

  // Get A's level & value arrays
  int A0_size = A_shape[0];
  int* A1_pos = A_mode_indices[1][0].data_ptr<int>();
  int* A1_crd = A_mode_indices[1][1].data_ptr<int>();
  float* A_val = A_values.data_ptr<float>();

  // Get x's level & value arrays
  int x0_size = x_shape[0];
  float* x_val = x_values.data_ptr<float>();

  // Initialize result value array
  int y_capacity = y0_size;
  float* y_values = (float*) malloc(sizeof(float) * y_capacity);
  memset(y_values, 0, sizeof(float) * y_capacity);

  #pragma omp parallel for
  for (int i = 0; i < A0_size; i++) {
    // Resolve dense coordinates
    int pA0 = i;
    // Resolve index into dense level of values array
    int py0 = i;
    // Initialize workspaces
    float wksp = 0;
    // Initialize iterators
    int pA1_end = A1_pos[i + 1];

    for (int pA1 = A1_pos[i]; pA1 < pA1_end; pA1++) {
      // Resolve coordinates
      int j = A1_crd[pA1];

      // Resolve dense coordinates
      int px0 = j;
      wksp += A_val[pA1] * x_val[j];
    }

    // Lower consumer CIN
    y_values[i] = wksp;

  }
  // Assemble final result
  Tensor y;
  auto y_values_deleter = [](void* ptr) {{ free(ptr); }};
  torch::Tensor y_values_torch = torch::from_blob(y_values, {y_capacity}, y_values_deleter, torch::kFloat32);
  y.storage.index.mode_indices = {{}};
  y.storage.value = y_values_torch;
  return y;
}
