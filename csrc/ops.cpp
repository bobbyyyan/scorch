#include "header.h"

#include <string>

#include "spmm.h"
#include "kernels.h"

namespace scorch {
namespace py = pybind11;

namespace {

using BinaryKernelFn = Tensor (*)(std::vector<int>, std::vector<int>,
                                  std::vector<std::vector<torch::Tensor>>,
                                  torch::Tensor, std::vector<int>,
                                  std::vector<std::vector<torch::Tensor>>,
                                  torch::Tensor);

using BinaryKernelWithTileFn = Tensor (*)(std::vector<int>, std::vector<int>,
                                          std::vector<std::vector<torch::Tensor>>,
                                          torch::Tensor, std::vector<int>,
                                          std::vector<std::vector<torch::Tensor>>,
                                          torch::Tensor, int);

using BinaryKernelWithTwoTilesFn = Tensor (*)(
    std::vector<int>, std::vector<int>, std::vector<std::vector<torch::Tensor>>,
    torch::Tensor, std::vector<int>, std::vector<std::vector<torch::Tensor>>,
    torch::Tensor, int, int);

template <typename scalar_t>
Tensor prebuilt_spmm_csr(
    std::vector<int> result_shape, std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values,
    std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices, torch::Tensor B_values,
    int tile_size = 32) {
  return spmm_csr_typed<scalar_t>(result_shape, A_shape, A_mode_indices, A_values,
                                  B_shape, B_mode_indices, B_values, tile_size);
}

template <typename scalar_t>
Tensor prebuilt_spmspm_csr(
    std::vector<int> result_shape, std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values,
    std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices,
    torch::Tensor B_values) {
  return spmspm_csr<scalar_t>(result_shape, A_shape, A_mode_indices, A_values,
                              B_shape, B_mode_indices, B_values);
}

template <typename scalar_t>
Tensor prebuilt_spmv_csr(
    std::vector<int> result_shape, std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices, torch::Tensor A_values,
    std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices,
    torch::Tensor B_values) {
  return spmv_csr<scalar_t>(result_shape, A_shape, A_mode_indices, A_values,
                            B_shape, B_mode_indices, B_values);
}

void bind_binary_kernel(py::module_& m, const char* name, BinaryKernelFn fn,
                        const char* doc) {
  m.def(name, fn, doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"));
}

void bind_binary_kernel_with_tile(py::module_& m, const char* name,
                                  BinaryKernelWithTileFn fn, const char* doc,
                                  int default_tile) {
  m.def(name, fn, doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"),
        py::arg("tile_size") = default_tile);
}

void bind_binary_kernel_with_two_tiles(py::module_& m, const char* name,
                                       BinaryKernelWithTwoTilesFn fn,
                                       const char* doc, int default_i_tile,
                                       int default_k_tile) {
  m.def(name, fn, doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"),
        py::arg("i_tile_size") = default_i_tile,
        py::arg("k_tile_size") = default_k_tile);
}

template <typename scalar_t>
void bind_typed_prebuilt_kernels(py::module_& m) {
  const std::string suffix = scorch_dtype_suffix<scalar_t>();
  const std::string spmm_name = "prebuilt_spmm_csr_" + suffix;
  const std::string spmspm_name = "prebuilt_spmspm_csr_" + suffix;
  const std::string spmv_name = "prebuilt_spmv_csr_" + suffix;

  bind_binary_kernel_with_tile(m, spmm_name.c_str(), &prebuilt_spmm_csr<scalar_t>,
                               "Typed prebuilt SpMM kernel (CSR x dense)", 32);
  bind_binary_kernel(m, spmspm_name.c_str(), &prebuilt_spmspm_csr<scalar_t>,
                     "Typed prebuilt SpGEMM kernel (CSR x CSR)");
  bind_binary_kernel(m, spmv_name.c_str(), &prebuilt_spmv_csr<scalar_t>,
                     "Typed prebuilt SpMV kernel (CSR x dense vector)");
}

void bind_prebuilt_kernel_family(py::module_& m) {
  bind_typed_prebuilt_kernels<float>(m);
  bind_typed_prebuilt_kernels<double>(m);
  bind_typed_prebuilt_kernels<int32_t>(m);
  bind_typed_prebuilt_kernels<int64_t>(m);

  // Legacy aliases retained for compatibility.
  bind_binary_kernel_with_tile(m, "spmm_csr_float", &spmm_csr_float,
                               "Sparse matrix multiplication (CSR)", 32);
  bind_binary_kernel_with_tile(m, "spmm_csr_double", &spmm_csr_double,
                               "Sparse matrix multiplication (CSR, float64)",
                               32);
  bind_binary_kernel(m, "spmspm_csr_float", &spmspm_csr<float>,
                     "Sparse matrix-sparse matrix multiplication (CSR)");
}

void bind_experimental_spmm_variants(py::module_& m) {
  bind_binary_kernel_with_two_tiles(
      m, "spmm_csr_float_tiled_i_k", &spmm_csr_float_tiled_i_k,
      "Sparse matrix multiplication with i and k tiling (CSR)", 16, 32);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_optimized", &spmm_csr_float_optimized,
      "Optimized sparse matrix multiplication (CSR)", 128);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_turbo", &spmm_csr_float_turbo,
      "Turbo-optimized sparse matrix multiplication (CSR)", 128);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_ultra", &spmm_csr_float_ultra,
      "Ultra-optimized sparse matrix multiplication (CSR)", 256);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_apex", &spmm_csr_float_apex,
      "Apex-optimized sparse matrix multiplication (CSR)", 256);

  bind_binary_kernel(m, "spmm_csr_float_untiled", &spmm_csr_float_untiled,
                     "Sparse matrix multiplication (CSR) (untiled)");
  bind_binary_kernel(m, "spmm_coo_float", &spmm_coo_float,
                     "Sparse matrix multiplication (COO)");
  bind_binary_kernel(m, "spmspm_coo_float", &spmspm_coo_float_opt,
                     "Sparse matrix-sparse matrix multiplication (COO)");

  // Novel SpMM variants
  bind_binary_kernel(m, "spmm_csr_float_direct", &spmm_csr_float_direct,
                     "Direct-to-C accumulation SpMM (no workspace)");
  bind_binary_kernel(m, "spmm_csr_float_neon", &spmm_csr_float_neon,
                     "Explicit ARM NEON vectorized SpMM");
  bind_binary_kernel(m, "spmm_csr_float_row_panel", &spmm_csr_float_row_panel,
                     "Multi-row panel SpMM with B-row reuse");
  bind_binary_kernel(m, "spmm_csr_float_k_parallel", &spmm_csr_float_k_parallel,
                     "K-parallel SpMM with direct output");
  bind_binary_kernel(m, "spmm_csr_float_sorted_rows", &spmm_csr_float_sorted_rows,
                     "Row-sorted SpMM with density-specific code paths");
  bind_binary_kernel(m, "spmm_csr_float_neon2", &spmm_csr_float_neon2,
                     "NEON 2-NNZ unroll with deep prefetch");
  bind_binary_kernel(m, "spmm_csr_float_neon4", &spmm_csr_float_neon4,
                     "NEON 4-NNZ unroll with deep prefetch");
  bind_binary_kernel(m, "spmm_csr_float_tiled_neon", &spmm_csr_float_tiled_neon,
                     "Large-tile NEON (128) with direct accumulation");
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_v2", &spmm_csr_float_v2,
      "Workspace + 2-nnz ILP + k-tiling SpMM", 256);
}

// Fused SpMM + bias + ReLU wrappers
Tensor spmm_csr_bias_relu_float(
    std::vector<int> result_shape, std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices,
    torch::Tensor A_values, std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices,
    torch::Tensor B_values, torch::Tensor bias) {
  return spmm_csr_bias_act<float, true>(
      result_shape, A_shape, A_mode_indices, A_values,
      B_shape, B_mode_indices, B_values, bias);
}

Tensor spmm_csr_bias_float(
    std::vector<int> result_shape, std::vector<int> A_shape,
    std::vector<std::vector<torch::Tensor>> A_mode_indices,
    torch::Tensor A_values, std::vector<int> B_shape,
    std::vector<std::vector<torch::Tensor>> B_mode_indices,
    torch::Tensor B_values, torch::Tensor bias) {
  return spmm_csr_bias_act<float, false>(
      result_shape, A_shape, A_mode_indices, A_values,
      B_shape, B_mode_indices, B_values, bias);
}

using FusedKernelFn = Tensor (*)(std::vector<int>, std::vector<int>,
                                  std::vector<std::vector<torch::Tensor>>,
                                  torch::Tensor, std::vector<int>,
                                  std::vector<std::vector<torch::Tensor>>,
                                  torch::Tensor, torch::Tensor);

void bind_fused_kernel(py::module_& m, const char* name, FusedKernelFn fn,
                       const char* doc) {
  m.def(name, fn, doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"), py::arg("bias"));
}

void bind_fused_spmm_variants(py::module_& m) {
  bind_fused_kernel(m, "spmm_csr_bias_relu_float", &spmm_csr_bias_relu_float,
                    "Fused SpMM + bias + ReLU (CSR x dense)");
  bind_fused_kernel(m, "spmm_csr_bias_float", &spmm_csr_bias_float,
                    "Fused SpMM + bias (CSR x dense, no activation)");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    bind_prebuilt_kernel_family(m);
    bind_experimental_spmm_variants(m);
    bind_fused_spmm_variants(m);

    py::class_<Tensor>(m, "Tensor")
      .def(py::init<>())
      .def_readwrite("storage", &Tensor::storage);
    py::class_<TensorStorage>(m, "TensorStorage")
      .def(py::init<>())
      .def_readwrite("value", &TensorStorage::value)
      .def_readwrite("index", &TensorStorage::index);
    py::class_<TensorIndex>(m, "TensorIndex")
      .def(py::init<>())
      .def_readwrite("mode_indices", &TensorIndex::mode_indices);
  }
}
