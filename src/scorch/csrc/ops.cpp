#include "header.h"

#include <limits>
#include <string>

#include "native_abi.h"
#include "spmm.h"
#include "kernels.h"
#include "plan.h"
#include "planned_matmul.h"
#include "sparse_grad.h"

namespace scorch {
namespace py = pybind11;

namespace {

// A CSR x dense product with more nonzeros than an int32 offset holds, run as row panels that each
// fit (native_abi.h, abi_csr_row_panels). Each panel is validated as an ordinary call is, and its
// kernel writes its rows straight into the product's one output (ScorchPanelCall). Out of line,
// so the ordinary binding it is called from is unchanged.
template <typename scalar_t>
__attribute__((noinline)) Tensor spmm_csr_v2_panelled(
    const char* op, const std::vector<int64_t>& result_shape, const std::vector<int64_t>& A_shape,
    const std::vector<std::vector<torch::Tensor>>& A_mode_indices, const torch::Tensor& A_values,
    const std::vector<int64_t>& B_shape, std::vector<std::vector<torch::Tensor>> B_mode_indices,
    const torch::Tensor& B_values, int tile_size, int nthreads_override, bool atparallel) {
  constexpr bool single = std::is_same<scalar_t, float>::value;
  auto run = [&](const std::vector<int64_t>& panel_shape, const std::vector<int64_t>& panel_a_shape,
                 std::vector<std::vector<torch::Tensor>> panel_levels,
                 const torch::Tensor& panel_values, torch::Tensor& into) {
    scorch_native::validate_binary_inputs(
        op, scorch_native::BinaryContract::CsrDenseMatmulInt64Elements,
        single ? torch::kFloat32 : torch::kFloat64, panel_shape, panel_a_shape, panel_levels,
        panel_values, B_shape, B_mode_indices, B_values);
    const int tile = scorch_native::validate_tile_size(tile_size, op, "tile_size", B_shape[1]);
    scorch_native::validate_thread_override(nthreads_override, op);
    const std::vector<int> c = scorch_native::narrow_legacy_shape(panel_shape, op, "result_shape");
    const std::vector<int> a = scorch_native::narrow_legacy_shape(panel_a_shape, op, "A_shape");
    const std::vector<int> b = scorch_native::narrow_legacy_shape(B_shape, op, "B_shape");
    const int* pos = panel_levels[1][0].data_ptr<int>();
    const int* crd = panel_levels[1][1].data_ptr<int>();
    if constexpr (single)
      spmm_csr_float_v2_panel_core(c[0], c[1], a[0], pos, crd, panel_values.data_ptr<float>(),
                                   b[1], B_values.data_ptr<float>(), tile, nthreads_override,
                                   atparallel, into.data_ptr<float>());
    else
      spmm_csr_double_v2_panel_core(c[0], c[1], a[0], pos, crd, panel_values.data_ptr<double>(),
                                    b[1], B_values.data_ptr<double>(), tile, nthreads_override,
                                    atparallel, into.data_ptr<double>());
  };
  Tensor C;
  C.storage.index.mode_indices = {{}, {}};
  C.storage.value = scorch_native::abi_csr_row_panels(op, result_shape, A_shape, A_mode_indices,
                                                      A_values, run);
  return C;
}

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
                        const char* doc,
                        scorch_native::BinaryContract contract,
                        torch::ScalarType dtype) {
  m.def(name,
        [fn, name, contract, dtype](
            std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
            std::vector<std::vector<torch::Tensor>> A_mode_indices,
            torch::Tensor A_values, std::vector<int64_t> B_shape,
            std::vector<std::vector<torch::Tensor>> B_mode_indices,
            torch::Tensor B_values) {
          scorch_native::validate_binary_inputs(
              name, contract, dtype, result_shape, A_shape, A_mode_indices,
              A_values, B_shape, B_mode_indices, B_values);
          return fn(
              scorch_native::narrow_legacy_shape(result_shape, name,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, name, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, name, "B_shape"),
              B_mode_indices, B_values);
        },
        doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"));
}

void bind_binary_kernel_with_tile(py::module_& m, const char* name,
                                  BinaryKernelWithTileFn fn, const char* doc,
                                  int default_tile,
                                  scorch_native::BinaryContract contract,
                                  torch::ScalarType dtype) {
  m.def(name,
        [fn, name, contract, dtype](
            std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
            std::vector<std::vector<torch::Tensor>> A_mode_indices,
            torch::Tensor A_values, std::vector<int64_t> B_shape,
            std::vector<std::vector<torch::Tensor>> B_mode_indices,
            torch::Tensor B_values, int tile_size) {
          scorch_native::validate_binary_inputs(
              name, contract, dtype, result_shape, A_shape, A_mode_indices,
              A_values, B_shape, B_mode_indices, B_values);
          tile_size = scorch_native::validate_tile_size(
              tile_size, name, "tile_size", B_shape[1]);
          return fn(
              scorch_native::narrow_legacy_shape(result_shape, name,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, name, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, name, "B_shape"),
              B_mode_indices, B_values, tile_size);
        },
        doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"),
        py::arg("tile_size") = default_tile);
}

void bind_binary_kernel_with_two_tiles(py::module_& m, const char* name,
                                       BinaryKernelWithTwoTilesFn fn,
                                       const char* doc, int default_i_tile,
                                       int default_k_tile,
                                       scorch_native::BinaryContract contract,
                                       torch::ScalarType dtype) {
  m.def(name,
        [fn, name, contract, dtype](
            std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
            std::vector<std::vector<torch::Tensor>> A_mode_indices,
            torch::Tensor A_values, std::vector<int64_t> B_shape,
            std::vector<std::vector<torch::Tensor>> B_mode_indices,
            torch::Tensor B_values, int i_tile_size, int k_tile_size) {
          scorch_native::validate_binary_inputs(
              name, contract, dtype, result_shape, A_shape, A_mode_indices,
              A_values, B_shape, B_mode_indices, B_values);
          i_tile_size = scorch_native::validate_tile_size(
              i_tile_size, name, "i_tile_size", A_shape[0]);
          k_tile_size = scorch_native::validate_tile_size(
              k_tile_size, name, "k_tile_size", B_shape[1]);
          return fn(
              scorch_native::narrow_legacy_shape(result_shape, name,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, name, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, name, "B_shape"),
              B_mode_indices, B_values, i_tile_size, k_tile_size);
        },
        doc, py::arg("result_shape"), py::arg("A_shape"),
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
                               "Typed prebuilt SpMM kernel (CSR x dense)", 32,
                               scorch_native::BinaryContract::CsrDenseMatmul,
                               scorch_torch_dtype<scalar_t>());
  bind_binary_kernel(m, spmspm_name.c_str(), &prebuilt_spmspm_csr<scalar_t>,
                     "Typed prebuilt SpGEMM kernel (CSR x CSR)",
                     scorch_native::BinaryContract::CsrCsrMatmul,
                     scorch_torch_dtype<scalar_t>());
  bind_binary_kernel(m, spmv_name.c_str(), &prebuilt_spmv_csr<scalar_t>,
                     "Typed prebuilt SpMV kernel (CSR x dense vector)",
                     scorch_native::BinaryContract::CsrDenseMatvec,
                     scorch_torch_dtype<scalar_t>());
}

void bind_prebuilt_kernel_family(py::module_& m) {
  bind_typed_prebuilt_kernels<float>(m);
  bind_typed_prebuilt_kernels<double>(m);
  bind_typed_prebuilt_kernels<int32_t>(m);
  bind_typed_prebuilt_kernels<int64_t>(m);

  // Legacy aliases retained for compatibility.
  bind_binary_kernel_with_tile(m, "spmm_csr_float", &spmm_csr_float,
                               "Sparse matrix multiplication (CSR)", 32,
                               scorch_native::BinaryContract::CsrDenseMatmul,
                               torch::kFloat32);
  bind_binary_kernel_with_tile(m, "spmm_csr_double", &spmm_csr_double,
                               "Sparse matrix multiplication (CSR, float64)",
                               32,
                               scorch_native::BinaryContract::CsrDenseMatmul,
                               torch::kFloat64);
  bind_binary_kernel(m, "spmspm_csr_float", &spmspm_csr<float>,
                     "Sparse matrix-sparse matrix multiplication (CSR)",
                     scorch_native::BinaryContract::CsrCsrMatmul,
                     torch::kFloat32);
}

void bind_experimental_spmm_variants(py::module_& m) {
  bind_binary_kernel_with_two_tiles(
      m, "spmm_csr_float_tiled_i_k", &spmm_csr_float_tiled_i_k,
      "Sparse matrix multiplication with i and k tiling (CSR)", 16, 32,
      scorch_native::BinaryContract::CsrDenseMatmul, torch::kFloat32);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_optimized", &spmm_csr_float_optimized,
      "Optimized sparse matrix multiplication (CSR)", 128,
      scorch_native::BinaryContract::CsrDenseMatmul, torch::kFloat32);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_turbo", &spmm_csr_float_turbo,
      "Turbo-optimized sparse matrix multiplication (CSR)", 128,
      scorch_native::BinaryContract::CsrDenseMatmul, torch::kFloat32);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_ultra", &spmm_csr_float_ultra,
      "Ultra-optimized sparse matrix multiplication (CSR)", 256,
      scorch_native::BinaryContract::CsrDenseMatmul, torch::kFloat32);
  bind_binary_kernel_with_tile(
      m, "spmm_csr_float_apex", &spmm_csr_float_apex,
      "Apex-optimized sparse matrix multiplication (CSR)", 256,
      scorch_native::BinaryContract::CsrDenseMatmul, torch::kFloat32);

  bind_binary_kernel(m, "spmm_csr_float_untiled", &spmm_csr_float_untiled,
                     "Sparse matrix multiplication (CSR) (untiled)",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_coo_float", &spmm_coo_float,
                     "Sparse matrix multiplication (COO)",
                     scorch_native::BinaryContract::CooDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmspm_coo_float", &spmspm_coo_float_opt,
                     "Sparse matrix-sparse matrix multiplication (COO)",
                     scorch_native::BinaryContract::CooCooMatmul,
                     torch::kFloat32);

  // Novel SpMM variants
  bind_binary_kernel(m, "spmm_csr_float_direct", &spmm_csr_float_direct,
                     "Direct-to-C accumulation SpMM (no workspace)",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_neon", &spmm_csr_float_neon,
                     "Explicit ARM NEON vectorized SpMM",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_row_panel", &spmm_csr_float_row_panel,
                     "Multi-row panel SpMM with B-row reuse",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_k_parallel", &spmm_csr_float_k_parallel,
                     "K-parallel SpMM with direct output",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_sorted_rows", &spmm_csr_float_sorted_rows,
                     "Row-sorted SpMM with density-specific code paths",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_neon2", &spmm_csr_float_neon2,
                     "NEON 2-NNZ unroll with deep prefetch",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_neon4", &spmm_csr_float_neon4,
                     "NEON 4-NNZ unroll with deep prefetch",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  bind_binary_kernel(m, "spmm_csr_float_tiled_neon", &spmm_csr_float_tiled_neon,
                     "Large-tile NEON (128) with direct accumulation",
                     scorch_native::BinaryContract::CsrDenseMatmul,
                     torch::kFloat32);
  // Bound explicitly (not via bind_binary_kernel_with_tile) because v2 carries two
  // extra optional composition hints the drop-in matmul dispatch passes: the host
  // thread count (nthreads_override, avoids host<->kernel team reshape) and
  // atparallel (launch the workers on torch's intra-op pool so the SpMM shares one
  // warm team with the torch epilogue in a pipeline).
  m.def("spmm_csr_float_v2",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values, int tile_size, int nthreads_override,
           bool atparallel) {
          constexpr const char* op = "spmm_csr_float_v2";
          // More nonzeros than an int32 offset holds: run as row panels that each fit
          // (native_abi.h, abi_csr_row_panels).
          if (scorch_native::abi_csr_needs_row_panels(A_mode_indices))
            return spmm_csr_v2_panelled<float>(op, result_shape, A_shape, A_mode_indices,
                                              A_values, B_shape, B_mode_indices, B_values,
                                              tile_size, nthreads_override, atparallel);
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmulInt64Elements,
              torch::kFloat32, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          tile_size = scorch_native::validate_tile_size(
              tile_size, op, "tile_size", B_shape[1]);
          scorch_native::validate_thread_override(nthreads_override, op);
          return spmm_csr_float_v2(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values, tile_size, nthreads_override, atparallel);
        },
        "Workspace + 2-nnz ILP + k-tiling SpMM",
        py::arg("result_shape"), py::arg("A_shape"), py::arg("A_mode_indices"),
        py::arg("A_values"), py::arg("B_shape"), py::arg("B_mode_indices"),
        py::arg("B_values"), py::arg("tile_size") = 256,
        py::arg("nthreads_override") = -1, py::arg("atparallel") = false);
  // float64 CSR x dense through spmm_csr_double_v2_core, which is the float32 v2
  // kernel at double where AVX2 makes that worth doing and the reference kernel
  // where it does not -- see the long note at that function. Bound explicitly for
  // the same reason v2 is: it takes the composition hints. The float64 route
  // resolves this instead of spmm_csr_double, which stays as the reference kernel
  // the tests compare against.
  m.def("spmm_csr_double_v2",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values, int tile_size, int nthreads_override,
           bool atparallel) {
          constexpr const char* op = "spmm_csr_double_v2";
          // More nonzeros than an int32 offset holds: run as row panels that each fit
          // (native_abi.h, abi_csr_row_panels).
          if (scorch_native::abi_csr_needs_row_panels(A_mode_indices))
            return spmm_csr_v2_panelled<double>(op, result_shape, A_shape, A_mode_indices,
                                              A_values, B_shape, B_mode_indices, B_values,
                                              tile_size, nthreads_override, atparallel);
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmulInt64Elements,
              torch::kFloat64, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          tile_size = scorch_native::validate_tile_size(
              tile_size, op, "tile_size", B_shape[1]);
          scorch_native::validate_thread_override(nthreads_override, op);
          return spmm_csr_double_v2(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values, tile_size, nthreads_override, atparallel);
        },
        "float64 CSR x dense: the float32 v2 kernel at double on AVX2, the "
        "reference kernel elsewhere",
        py::arg("result_shape"), py::arg("A_shape"), py::arg("A_mode_indices"),
        py::arg("A_values"), py::arg("B_shape"), py::arg("B_mode_indices"),
        py::arg("B_values"), py::arg("tile_size") = 256,
        py::arg("nthreads_override") = -1, py::arg("atparallel") = false);
  // Column-panel ("tile-j") SpMM for the high-degree operand-over-LLC thrash
  // regime (reddit/products-class). Reached only when the adaptive tiling selector
  // (scorch.tiling / ops.matmul) fires; v2 serves every other shape. Jc = panel
  // width in contraction columns (~C/(4N)); Jc<=0 degenerates to full-width.
  m.def("spmm_csr_float_tilej",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values, int Jc, int nthreads_override) {
          constexpr const char* op = "spmm_csr_float_tilej";
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmulInt64Elements,
              torch::kFloat32, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          TORCH_CHECK(Jc >= 0, op, ": Jc must be nonnegative, got ", Jc);
          TORCH_CHECK(B_shape[0] > 0, op,
                      ": zero contraction extent is not supported");
          scorch_native::validate_thread_override(nthreads_override, op);
          return spmm_csr_float_tilej(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values, Jc, nthreads_override);
        },
        "Column-panel (tile-j) SpMM for high-degree operand-over-LLC graphs",
        py::arg("result_shape"), py::arg("A_shape"), py::arg("A_mode_indices"),
        py::arg("A_values"), py::arg("B_shape"), py::arg("B_mode_indices"),
        py::arg("B_values"), py::arg("Jc") = 0,
        py::arg("nthreads_override") = -1);
  // 3D-blocked (tile-ijk) SpMM with a B width-panel relayout, for the scattered +
  // very-wide-B regime where even tile-j erodes (its C re-traffic grows ~N^2).
  // Blocks the free dim N into Nc-wide strips, relays each strip of B contiguous,
  // accumulates into a cache-resident Cp, writes C once (C-traffic ~N). Reached
  // only when the selector's wide-N branch probes it; v2/tile-j serve everything
  // else. Nc=free-dim strip width, Jc=contraction-panel width (both <=0 degenerate).
  m.def("spmm_csr_float_tileijk",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values, int Nc, int Jc, int nthreads_override) {
          constexpr const char* op = "spmm_csr_float_tileijk";
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmulInt64Elements,
              torch::kFloat32, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          TORCH_CHECK(Nc >= 0, op, ": Nc must be nonnegative, got ", Nc);
          TORCH_CHECK(Jc >= 0, op, ": Jc must be nonnegative, got ", Jc);
          TORCH_CHECK(B_shape[0] > 0 && result_shape[1] > 0, op,
                      ": zero contraction/free extents are not supported");
          scorch_native::validate_thread_override(nthreads_override, op);
          return spmm_csr_float_tileijk(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values, Nc, Jc, nthreads_override);
        },
        "Tile-ijk SpMM (B width-panel relayout) for scattered very-wide-B graphs",
        py::arg("result_shape"), py::arg("A_shape"), py::arg("A_mode_indices"),
        py::arg("A_values"), py::arg("B_shape"), py::arg("B_mode_indices"),
        py::arg("B_values"), py::arg("Nc") = 0, py::arg("Jc") = 0,
        py::arg("nthreads_override") = -1);
  // Fused feature-major Linear: Y[out,batch] = act(W @ X[in,batch] + bias[:,None])
  // with per-OUTPUT-CHANNEL (per-row) bias + activation folded into v2's parallel
  // region (see spmm.h). act: 0=identity, 1=relu, 2=sigmoid. Same composition hints
  // as v2. Reached only via scorch.sparse_linear_fm (never FEM/GCN's scorch.matmul).
  m.def("spmm_csr_linear_fused_float",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values, torch::Tensor bias_values, int act,
           int tile_size, int nthreads_override, bool atparallel) {
          constexpr const char* op = "spmm_csr_linear_fused_float";
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmul,
              torch::kFloat32, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          scorch_native::check_flat_values(bias_values, torch::kFloat32,
                                           A_shape[0], op, "bias_values");
          TORCH_CHECK(act >= 0 && act <= 2, op,
                      ": act must be 0 (identity), 1 (relu), or 2 (sigmoid), got ",
                      act);
          tile_size = scorch_native::validate_tile_size(
              tile_size, op, "tile_size", B_shape[1]);
          scorch_native::validate_thread_override(nthreads_override, op);
          return spmm_csr_linear_fused_float(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values, bias_values, act, tile_size,
              nthreads_override, atparallel);
        },
        "Fused feature-major SpMM + per-output-channel bias + activation",
        py::arg("result_shape"), py::arg("A_shape"), py::arg("A_mode_indices"),
        py::arg("A_values"), py::arg("B_shape"), py::arg("B_mode_indices"),
        py::arg("B_values"), py::arg("bias_values"), py::arg("act"),
        py::arg("tile_size") = 256,
        py::arg("nthreads_override") = -1, py::arg("atparallel") = false);
  // Fast cache-blocked [R,C]->[C,R] float32 transpose (AVX2 8x8 / NEON 4x4 /
  // scalar micro-tiles). Materializes the feature-major input for the drop-in
  // sparse_linear path ~2-3x faster than torch's cache-hostile x.T.contiguous().
  // Pass the host thread count to launch on torch's warm intra-op pool.
  m.def("scorch_transpose_2d_float",
        [](torch::Tensor src, int nthreads_override) {
          constexpr const char* op = "scorch_transpose_2d_float";
          scorch_native::check_tensor_common(src, torch::kFloat32, op, "src", 2);
          scorch_native::checked_product({src.size(0), src.size(1)}, op,
                                         "src shape");
          scorch_native::validate_thread_override(nthreads_override, op);
          return scorch_transpose_2d_float(src, nthreads_override);
        },
        "Fast cache-blocked 2D float32 transpose ([R,C]->[C,R])",
        py::arg("src"), py::arg("nthreads_override") = -1);
  // Row-wise CSR softmax (scale folded in): the scatter-free replacement for the
  // torch scatter softmax in the sparse-attention chain. Parallel over rows on
  // torch's warm intra-op pool when the host thread count is passed.
  m.def("scorch_sparse_softmax_csr_float",
        [](torch::Tensor crow_indices, torch::Tensor values, double scale,
           int nthreads_override) {
          constexpr const char* op = "scorch_sparse_softmax_csr_float";
          scorch_native::validate_csr_segments(op, crow_indices, values);
          scorch_native::validate_finite_scale(scale, op);
          scorch_native::validate_thread_override(nthreads_override, op);
          return scorch_sparse_softmax_csr_float(crow_indices, values, scale,
                                                 nthreads_override);
        },
        "Row-wise softmax over CSR value spans (softmax of scale*values)",
        py::arg("crow_indices"), py::arg("values"), py::arg("scale") = 1.0,
        py::arg("nthreads_override") = -1);
  // Fused sparse (masked) multi-head attention over a shared CSR mask: one pass
  // computes softmax(scale * Q.K) . V over each row's nonzero columns, batched
  // over heads, taking Q/K/V as [S,H,D] directly (no per-head slice) and the CSR
  // mask once. Replaces the per-head SDDMM -> softmax -> SpMM chain + CSR
  // round-trip. Pass the host thread count to run on torch's warm intra-op pool.
  m.def("scorch_sparse_attention_csr_float",
        [](torch::Tensor crow_indices, torch::Tensor col_indices,
           torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale,
           int nthreads_override) {
          constexpr const char* op = "scorch_sparse_attention_csr_float";
          scorch_native::validate_attention_inputs(op, crow_indices, col_indices,
                                                   Q, K, V);
          scorch_native::validate_finite_scale(scale, op);
          scorch_native::validate_thread_override(nthreads_override, op);
          return scorch_sparse_attention_csr_float(
              crow_indices, col_indices, Q, K, V, scale, nthreads_override);
        },
        "Fused sparse multi-head attention (SDDMM + row-softmax + weighted-V)",
        py::arg("crow_indices"), py::arg("col_indices"), py::arg("Q"),
        py::arg("K"), py::arg("V"), py::arg("scale") = 1.0,
        py::arg("nthreads_override") = -1);
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
  m.def(name,
        [fn, name](std::vector<int64_t> result_shape,
                   std::vector<int64_t> A_shape,
                   std::vector<std::vector<torch::Tensor>> A_mode_indices,
                   torch::Tensor A_values, std::vector<int64_t> B_shape,
                   std::vector<std::vector<torch::Tensor>> B_mode_indices,
                   torch::Tensor B_values, torch::Tensor bias) {
          scorch_native::validate_binary_inputs(
              name, scorch_native::BinaryContract::CsrDenseMatmul,
              torch::kFloat32, result_shape, A_shape, A_mode_indices, A_values,
              B_shape, B_mode_indices, B_values);
          scorch_native::check_flat_values(bias, torch::kFloat32, B_shape[1],
                                           name, "bias");
          return fn(
              scorch_native::narrow_legacy_shape(result_shape, name,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(A_shape, name, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, name, "B_shape"),
              B_mode_indices, B_values, bias);
        },
        doc, py::arg("result_shape"), py::arg("A_shape"),
        py::arg("A_mode_indices"), py::arg("A_values"), py::arg("B_shape"),
        py::arg("B_mode_indices"), py::arg("B_values"), py::arg("bias"));
}

void bind_fused_spmm_variants(py::module_& m) {
  bind_fused_kernel(m, "spmm_csr_bias_relu_float", &spmm_csr_bias_relu_float,
                    "Fused SpMM + bias + ReLU (CSR x dense)");
  bind_fused_kernel(m, "spmm_csr_bias_float", &spmm_csr_bias_float,
                    "Fused SpMM + bias (CSR x dense, no activation)");
}

void bind_sddmm_variants(py::module_& m) {
  m.def("sddmm_coo_float_prebuilt",
        [](std::vector<int64_t> result_shape, std::vector<int64_t> S_shape,
           std::vector<std::vector<torch::Tensor>> S_mode_indices,
           torch::Tensor S_values, std::vector<int64_t> A_shape,
           std::vector<std::vector<torch::Tensor>> A_mode_indices,
           torch::Tensor A_values, std::vector<int64_t> B_shape,
           std::vector<std::vector<torch::Tensor>> B_mode_indices,
           torch::Tensor B_values) {
          constexpr const char* op = "sddmm_coo_float_prebuilt";
          scorch_native::validate_sddmm_inputs(
              op, result_shape, S_shape, S_mode_indices, S_values, A_shape,
              A_mode_indices, A_values, B_shape, B_mode_indices, B_values);
          return sddmm_coo_float_prebuilt(
              scorch_native::narrow_legacy_shape(result_shape, op,
                                                  "result_shape"),
              scorch_native::narrow_legacy_shape(S_shape, op, "S_shape"),
              S_mode_indices, S_values,
              scorch_native::narrow_legacy_shape(A_shape, op, "A_shape"),
              A_mode_indices, A_values,
              scorch_native::narrow_legacy_shape(B_shape, op, "B_shape"),
              B_mode_indices, B_values);
        },
        "Prebuilt SDDMM kernel (COO x dense x dense)",
        py::arg("result_shape"),
        py::arg("S_shape"), py::arg("S_mode_indices"), py::arg("S_values"),
        py::arg("A_shape"), py::arg("A_mode_indices"), py::arg("A_values"),
        py::arg("B_shape"), py::arg("B_mode_indices"), py::arg("B_values"));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Whether the SpMM A/B tune hooks (SCORCH_SPMM_WORKSPACE and friends) are
    // compiled in. A harness that selects between two kernel paths by environment
    // variable has no way to tell a build without them from a build where the change
    // it is measuring does nothing: both arms take the same path and every ratio
    // reads 1.000 with tight controls. Publishing the flag lets such a harness fail
    // closed instead of reporting the null it manufactured.
#ifdef SCORCH_TUNE_HOOKS
    m.attr("spmm_tune_hooks") = true;
#else
    m.attr("spmm_tune_hooks") = false;
#endif

    bind_prebuilt_kernel_family(m);
    bind_experimental_spmm_variants(m);
    bind_fused_spmm_variants(m);
    bind_sddmm_variants(m);

    // The ABI validation memo, reachable from Python so tests can force the cold
    // path and so a caller under memory pressure can drop the narrowed int32 copies.
    // Scope: THIS extension's memo only. native_abi.h holds its maps in inline
    // function-local statics, and Python dlopens extension modules with RTLD_LOCAL,
    // so every JIT-compiled kernel carries its own copy. Each amortizes over its own
    // calls, which is all the fix needs; the cost is that a tensor used by both a
    // prebuilt and a generated kernel is narrowed once per module.
    m.def("abi_memo_clear", &scorch_native::abi_memo_clear,
          "Drop every cached index-validation verdict and narrowed index copy");
    m.def("abi_memo_size", &scorch_native::abi_memo_size,
          "Entries across both ABI validation memos (dead ones until the next sweep)");
    m.def("abi_narrow_held", &scorch_native::abi_narrow_held,
          "(entries, bytes) the narrowing memo holds, its arrays live or not. For tests");
    // CSR operands past the int32 offset bound run as row panels (native_abi.h). The bound is
    // INT32_MAX; tests lower it so the panelled path runs on matrices they can build.
    m.def("abi_csr_panel_limit",
          [](int64_t limit) {
            const int64_t old = scorch_native::abi_csr_panel_limit();
            if (limit > 0)
              scorch_native::abi_csr_panel_limit() =
                  std::min<int64_t>(limit, std::numeric_limits<int32_t>::max());
            return old;
          },
          py::arg("limit") = -1,
          "The largest nonzero count a CSR operand may pass to an int32 kernel whole; with "
          "limit > 0 (at most INT32_MAX), sets it and returns the previous one. For tests");
    m.def("abi_csr_panelled_calls",
          [] { return scorch_native::abi_csr_panelled_calls().load(); },
          "Calls run as row panels since the module loaded. For tests");

    // Write witnesses (native_abi.h). The capsule is how a JIT-compiled kernel module, which
    // carries its own memo, reads the epochs this module keeps; the functions are what Python's
    // caches over a torch sparse operand's index arrays use.
    m.attr("_abi_watch_api") =
        py::capsule(static_cast<const void*>(scorch_native::abi_watch_owner_api()),
                    "scorch_ops._abi_watch_api");
    m.def("torch_sparse_index_members", &scorch_native::torch_sparse_index_members,
          "A torch sparse CSR or COO operand's index arrays as caches must key them: the "
          "caller's own arrays (detached), with the operand recorded as a witness of their "
          "storages",
          py::arg("a"));
    m.def("abi_epoch", &scorch_native::abi_epoch_of,
          "The write-witness epoch of a tensor's storage, after observing its witnesses",
          py::arg("t"));
    m.def("abi_witness_count", [](const torch::Tensor& t) -> int64_t {
            std::lock_guard<std::mutex> guard(scorch_native::abi_watch_mutex());
            auto watch = scorch_native::abi_watch_locked(t.storage().unsafeGetStorageImpl(), false);
            return watch != nullptr ? (int64_t)watch->witnesses.size() : 0;
          },
          "Witnesses recorded on a tensor's storage. For tests", py::arg("t"));
    // The plan state of a torch sparse CSR operand's structure (torch_structures.h): the dict
    // plan.py keeps its plans, sightings and declines in, shared by every tensor over the same
    // index arrays and dropped when they die. Asking records the operand as a write witness.
    m.def("torch_structure_state",
          [](py::object a, bool create) -> py::object {
            if (!THPVariable_Check(a.ptr()))
              throw py::type_error("torch_structure_state expects a torch.Tensor");
            const at::Tensor& t = THPVariable_Unpack(a.ptr());
            // Only a 2-D CSR tensor is planned; anything else has no state.
            if (t.layout() != c10::kSparseCsr || t.dim() != 2) return py::none();
            PyObject* state = create ? scorch_native::torch_structure_create(t)
                                     : scorch_native::torch_structure_python_state(t);
            if (state == nullptr) return py::none();
            return py::reinterpret_borrow<py::object>(state);
          },
          "The plan state of a 2-D torch sparse CSR tensor's index arrays; None if it has "
          "none or is not one",
          py::arg("a"), py::arg("create") = false);
    m.def("torch_structure_count", &scorch_native::torch_structure_count,
          "Structures with plan state. For tests");
    m.def("torch_structure_sweep", &scorch_native::torch_structure_sweep,
          "Drops the structures whose arrays are gone and returns how many remain. For tests");

    // The cache size the KERNEL gates on, so a test can check it against the one
    // tiling.query_llc gates on. Two gates that disagree about the machine would
    // route a product to a tiled kernel the kernel then declines to stream for, and
    // nothing in either layer would report the disagreement.
    m.def("scorch_llc_bytes", &scorch_llc_bytes,
          "Effective last-level cache in bytes, as the kernels compute it");

    // The performance-core count, as the policy layer derives it. Exported for the same reason as
    // the chunk width and the thread count: chain46 made the resolved thread count the largest
    // single error on the scoreboard, its optimum on that host is exactly the P-core count, and any
    // harness exploring a rule written in those terms has to be able to ask for the number rather
    // than re-derive it from lscpu or sysctl and drift.
    m.def("scorch_pcore_count", &scorch_pcore_count,
          "Performance cores this host has, as the policy layer derives them: on Linux the "
          "physical cores carrying SMT where only some do, otherwise all physical cores; on macOS "
          "hw.perflevel0.physicalcpu. Cached, and overridable with SCORCH_PCORES.");

    // Whether this build carries the A/B tuning hooks. Without them the SCORCH_*
    // environment switches are inert, so a harness that flips one and compares two
    // arms is timing the same code twice and will report a difference of zero as
    // "the change did nothing" -- a vacuous measurement that looks like a result.
    m.def("scorch_tune_hooks", []() {
#ifdef SCORCH_TUNE_HOOKS
      return true;
#else
      return false;
#endif
    }, "True if this build honours the SCORCH_* kernel A/B hooks");

    // The chunk width the SpMM would pick for a given shape. Exposed so a
    // calibration harness reports the formula's own choice rather than a Python
    // restatement of it -- a second implementation of a rule is a second thing that
    // can be wrong, and it would be wrong silently, in the direction that flatters
    // whichever one the harness used.
#ifdef SCORCH_TUNE_HOOKS
    // Per-worker accounting for the SpMM's parallel region, hooks builds only, and only for calls
    // made while SCORCH_SPMM_WSTAT is set. Returns one row per worker id that ran --
    // (calls, nonzeros, rows, claims, busy nanoseconds) -- plus a final row holding the region's own
    // summed wall nanoseconds, the number of regions, and the last team size. A wall-clock time
    // measured from Python says the team was slow; this says which worker was slow, and whether the
    // per-worker times even add up to the wall. See the struct's comment in spmm.h.
    m.def("scorch_spmm_wstat", []() {
      std::vector<std::vector<double>> out;
      const ScorchSpmmWStat* st = scorch_spmm_wstat();
      for (int i = 0; i < 256; ++i) {
        if (st[i].calls == 0) continue;
        out.push_back({(double)i, (double)st[i].calls, (double)st[i].nnz, (double)st[i].rows,
                       (double)st[i].claims, st[i].ns_busy, st[i].ns_entry});
      }
      const double* w = scorch_spmm_wstat_wall();
      out.push_back({-1.0, w[1], 0.0, 0.0, 0.0, w[0], 0.0});
      out.push_back({-2.0, w[2], 0.0, 0.0, 0.0, 0.0, 0.0});
      return out;
    }, "Per-worker (id, calls, nnz, rows, claims, busy_ns, entry_ns) for the SpMM parallel "
       "region; entry_ns is how long after the region opened the worker started; row "
       "id=-1 carries (regions, ..., wall_ns) and id=-2 carries the last team size. Empty "
       "without SCORCH_TUNE_HOOKS.");

    m.def("scorch_spmm_wstat_reset", []() {
      scorch_spmm_wstat_reset();
    }, "Zero the per-worker accounting above.");

#endif  // SCORCH_TUNE_HOOKS

    m.def("scorch_spmm_chunk", &scorch_spmm_chunk,
          "Rows per work-stealing chunk the SpMM would use for (rows, nnz, k, nthreads)",
          py::arg("rows"), py::arg("nnz"), py::arg("k"), py::arg("nthreads"));

    // The GENERIC chunk -- what the SpMM used before the SpMM-specific rule existed.
    // A calibration harness needs it to answer the decision-relevant question, which
    // is not "how far is the rule from a per-cell oracle" but "is the rule better
    // than what it replaces".
    m.def("scorch_chunk_generic", &scorch_chunk,
          "Rows per chunk from the generic policy, for (rows, work, grain)",
          py::arg("rows"), py::arg("work"), py::arg("grain"));

    // The thread count the drop-in SpMM will actually run on. A harness that wants
    // to know which width the chunk rule picks has to ask for this rather than use
    // torch.get_num_threads(): omp_get_num_procs() reports 32 on a 24-physical-core
    // part, so the two differ, and the difference silently reclassified cells.
    m.def("scorch_spmm_nthreads", &scorch_spmm_nthreads,
          "Threads the drop-in SpMM resolves to, for (work, rows, "
          "nthreads_override, work_true, nnz). work is nnz*max(k,16); work_true "
          "is nnz*k and defaults to work, which reproduces the caller that has "
          "only the one number; nnz defaults to -1, which leaves the row-axis "
          "ceiling stated in rows.",
          py::arg("work"), py::arg("rows"), py::arg("nthreads_override"),
          py::arg("work_true") = -1, py::arg("nnz") = -1, py::arg("full_team") = false);
    // ... and the work it is asked about, which the kernel computes from the operands, its
    // non-empty rows and the empty rows it zeroes. A harness that needs the count for a product
    // asks for both.
    m.def("scorch_spmm_work", &scorch_spmm_work,
        "The work the drop-in SpMM states to scorch_spmm_nthreads for (nnz, empty rows it "
        "zeroes, width, output width)",
        py::arg("nnz"), py::arg("empty_rows"), py::arg("width"), py::arg("out_width"));
    // ... and the two composed as the kernel composes them, which is what a plan asks before a
    // gathered call (SpmmCsrPlan::run_gathered).
    m.def("scorch_spmm_v2_workers", &scorch_spmm_v2_workers,
        "The workers the drop-in SpMM runs a call on: scorch_spmm_nthreads over scorch_spmm_work "
        "for (nnz, empty rows it zeroes, width, output width, rows, nthreads_override, "
        "full_team)",
        py::arg("nnz"), py::arg("empty_rows"), py::arg("width"), py::arg("out_width"),
        py::arg("rows"), py::arg("nthreads_override"), py::arg("full_team") = false);

    // Exported so a harness asks for this count instead of recomputing it, for the same reason
    // as the chunk width and the thread count above. It is deliberately NOT a second spelling of
    // scorch_pcore_count (exported further up): that one counts the MACHINE's performance cores
    // and then clamps to omp_get_num_procs(), while this counts distinct physical cores inside
    // the process's affinity mask. Inside a 32-CPU cgroup over 16 cores the first answers 32 and
    // the second 16, and conflating them is what made a cap look inert on a host where it binds.
    // Re-exporting pcore_count here as a "matching pair" was written first and was wrong: two
    // m.def calls with the same name and signature register a second overload the first always
    // shadows, so it was dead registration code that nothing could ever reach.
    // The row-split partition's two decisions, exported for the same reason as the chunk width and
    // the thread count: a harness that restates a production policy is wrong silently and
    // flatteringly. The imbalance probe in particular is easy to get almost right -- it is the
    // longest run of equal-nonzero cuts landing in ONE row, not the max degree over the mean, and
    // the two differ on exactly the matrices this mechanism is for.
    m.def("scorch_spmm_row_imbalance",
          [](torch::Tensor A1_pos, long rows, long refn) {
            TORCH_CHECK(A1_pos.dtype() == torch::kInt32, "A1_pos must be int32");
            TORCH_CHECK(A1_pos.is_contiguous(), "A1_pos must be contiguous");
            TORCH_CHECK(rows >= 0 && rows < A1_pos.numel(),
                        "rows must be less than indptr's length");
            return scorch_spmm_row_imbalance(A1_pos.data_ptr<int>(), rows, refn);
          },
          "How far past its fair share of the nonzeros the widest row is, measured against a "
          "reference width: the longest run of equal-nonzero cuts that land in one row.",
          py::arg("A1_pos"), py::arg("rows"), py::arg("refn"));
    m.def("scorch_spmm_seg_width", &scorch_spmm_seg_width,
          "Nonzero width of one row-split segment for (nnz, k, elem_size). A function of the call "
          "only, never of the worker count, so a split result does not depend on OMP_NUM_THREADS.",
          py::arg("nnz"), py::arg("k"), py::arg("elem_size"));
    m.def("scorch_spmm_split_seg",
          [](torch::Tensor A1_pos, long rows, long k, long elem_size) {
            TORCH_CHECK(A1_pos.dtype() == torch::kInt32, "A1_pos must be int32");
            TORCH_CHECK(A1_pos.is_contiguous(), "A1_pos must be contiguous");
            TORCH_CHECK(rows >= 0 && rows < A1_pos.numel(),
                        "rows must be less than indptr's length");
            return scorch_spmm_split_seg(A1_pos.data_ptr<int>(), rows, k, elem_size);
          },
          "The row-split gate itself: the segment width the SpMM would use for this call, or 0 for "
          "leave the rows whole. Exported so a harness or a test asks the shipped rule instead of "
          "keeping a second copy of it that can drift.",
          py::arg("A1_pos"), py::arg("rows"), py::arg("k"), py::arg("elem_size"));

    m.def("scorch_phys_cores_avail", &scorch_phys_cores_avail,
          "Distinct physical cores in this process's CPU affinity mask. Falls back to "
          "omp_get_num_procs() where the topology cannot be read.");

    m.def("scorch_omp_runtime_path",
          []() -> py::object {
            const char* p = scorch_omp_runtime_path();
            return p ? py::object(py::str(p)) : py::object(py::none());
          },
          "The file of the OpenMP runtime this module's parallel regions run on, as the "
          "loaded process resolved it (None if it cannot be named).");
    m.def("scorch_omp_is_libgomp", &scorch_omp_is_libgomp,
          "Whether that runtime is GNU libgomp: the answer the drop-in SpMM's launch "
          "decisions use.");

    // The two escape-hatch flags that survive into a release object, as explicit process-level
    // policy rather than as an environment read on every call. See the comment above
    // scorch_flags in scorch_policy.h for why the read was moved: std::getenv is a linear scan
    // of environ, so it cost 50-480 nanoseconds per call depending on how many variables the
    // caller's shell happened to export, and an A/B arm that set more variables slowed every
    // arm in the process.
    //
    // A release-build harness that needs to flip one of these between interleaved arms must call
    // the setter; writing to os.environ after the first call no longer has any effect. That is
    // deliberate and it is the safer failure: a missing setter is an AttributeError, while an
    // environment write that is no longer read is silent, and a silently-ignored A/B flag turns
    // two arms into the same configuration -- a comparison that cannot fail. A HOOKS build still
    // reads per call, so hooks harnesses are unaffected.
    //
    // -1 restores "take it from the environment", which is what an unset flag means.
    m.def("scorch_set_spmm_atparallel",
          [](int v) { scorch_flags::atparallel_slot() = (v < 0 ? -1 : (v != 0 ? 1 : 0)); },
          "Force the drop-in SpMM's at::parallel_for launch on (1) or off (0) for this process; "
          "-1 returns to SCORCH_SPMM_ATPARALLEL. Release-build A/B only -- a hooks build reads "
          "the environment per call and ignores this.",
          py::arg("v"));
    m.def("scorch_set_neon_regtile",
          [](int v) { scorch_flags::neon_regtile_slot() = (v < 0 ? -1 : (v != 0 ? 1 : 0)); },
          "Force the NEON register-tiled row kernel on (1) or off (0) for this process; -1 "
          "returns to SCORCH_NEON_REGTILE. ARM only; inert on x86.",
          py::arg("v"));
    m.def("scorch_get_spmm_atparallel", []() { return scorch_flags::atparallel(); },
          "The resolved at::parallel_for flag: 1, 0, or -1 for 'not asked'. A harness that sets "
          "the policy should read it back rather than assume the set took.");
    m.def("scorch_get_neon_regtile", []() { return scorch_flags::neon_regtile(); },
          "The resolved NEON register-tile flag: 1, 0, or -1 for 'not asked'.");

    // The row-handout mode the drop-in SpMM will actually run in. Exported for the same
    // reason as the chunk width: an offline sweep that restated this rule scored the work
    // gate at 3.7% on the ARM tail against the 1.35% the machine measured, because it
    // credited cells the single-worker gate had already forced to mode 0. A harness that
    // asks cannot make that mistake.
    m.def("scorch_spmm_partition_mode", &scorch_spmm_partition_mode,
          "Row-handout mode (0 shared counter, 1 home ranges, 2 front-stealing, "
          "3 back-stealing) the drop-in SpMM resolves to, for (rows, nnz, k, out_cols, "
          "elem_size, nthreads, nthreads_override). nthreads is the resolved worker "
          "count from scorch_spmm_nthreads; nthreads_override is what the caller "
          "passed, which is what the pool ceiling reads.",
          py::arg("rows"), py::arg("nnz"), py::arg("k"), py::arg("out_cols"),
          py::arg("elem_size"), py::arg("nthreads"), py::arg("nthreads_override"));

    // The same fused structural pass, offered to Scorch's own Python-side validator.
    //
    // `_validate_index_storage` in storage.py runs on every STensor built over a
    // compressed level -- from_torch, to_sparse, a relayout, and every generated
    // kernel's result -- and its bounds and sortedness checks are separate whole-array
    // torch operations: about five passes over the index arrays plus a bool temporary,
    // which measured 304 us of the 385 us it took to wrap a 640k-nonzero result. This
    // screen does the same work in ONE pass with no allocation. A clean verdict also
    // covers the two endpoint checks (positions[0] == 0, positions[last] == nnz), and is
    // kept per array identity and version in the call boundary's verdict memo, so a
    // level built again around arrays that already passed is not scanned again.
    //
    // Same contract as every screen here: `true` means "this may be malformed, go and
    // find out", `false` means "no violation exists". Conservative in that exact
    // direction, so the Python caller can trust `false` and must re-run its own checks
    // on `true` -- which is also what keeps every diagnostic message and its precedence
    // byte-identical, since the Python path remains the only thing that reports.
    //
    // Anything unsupported -- a non-CPU or non-contiguous array, an index width that
    // is not int32/int64, mismatched widths, a position array shorter than one entry --
    // returns `true` so the caller simply does the work itself.
    m.def("abi_screen_compressed_level",
          [](const torch::Tensor& positions, const torch::Tensor& coordinates,
             int64_t extent, bool require_sorted, int64_t grain) -> bool {
            try {
              if (!positions.defined() || !coordinates.defined()) return true;
              if (positions.device().type() != torch::kCPU ||
                  coordinates.device().type() != torch::kCPU) return true;
              if (!positions.is_contiguous() || !coordinates.is_contiguous()) return true;
              if (positions.dtype() != coordinates.dtype()) return true;
              const int64_t count = positions.numel();
              if (count < 1) return true;
              const int64_t parents = count - 1;
              const int64_t nnz = coordinates.numel();
              if (positions.dtype() == torch::kInt32 &&
                  nnz > (int64_t)std::numeric_limits<int32_t>::max())
                return true;
              if (positions.dtype() != torch::kInt32 && positions.dtype() != torch::kInt64)
                return true;
              // The two endpoint checks storage.py would otherwise make with an
              // indexing op and a sync each: a clean verdict now covers them too.
              const bool ends_ok = positions.dtype() == torch::kInt32
                  ? positions.data_ptr<int32_t>()[0] == 0 &&
                        positions.data_ptr<int32_t>()[parents] == nnz
                  : positions.data_ptr<int64_t>()[0] == 0 &&
                        positions.data_ptr<int64_t>()[parents] == nnz;
              if (!ends_ok) return true;
              // The parameter layout of the call boundary's CSR check (checked_csr_view).
              scorch_native::AbiStructEntry want;
              want.add(coordinates);
              want.add(positions);
              want.params = {scorch_native::AbiCheckCompressed, parents, extent, nnz,
                             require_sorted ? 1 : 0};
              return scorch_native::abi_screen_through_memo(std::move(want), [&] {
                if (positions.dtype() == torch::kInt32)
                  return scorch_native::abi_screen_compressed_typed<int32_t>(
                      positions.data_ptr<int32_t>(), coordinates.data_ptr<int32_t>(),
                      parents, nnz, extent, require_sorted, grain);
                return scorch_native::abi_screen_compressed_typed<int64_t>(
                    positions.data_ptr<int64_t>(), coordinates.data_ptr<int64_t>(),
                    parents, nnz, extent, require_sorted, grain);
              });
            } catch (...) {
              return true;  // never turn a validation into a crash
            }
          },
          "Screen one compressed level: false = no violation, true = check it yourself",
          // The scan is O(nnz) and thread-split, and touches only tensor metadata and
          // raw data, so holding the interpreter lock across it would stall every other
          // Python thread for no reason.
          py::call_guard<py::gil_scoped_release>(),
          py::arg("positions"), py::arg("coordinates"), py::arg("extent"),
          py::arg("require_sorted") = true,
          py::arg("grain") = SCORCH_ABI_VALIDATE_GRAIN);

    // "every coordinate of this level lies in [0, extent)". Same contract. Stands in
    // for a min() and a max() reduction plus two device syncs in storage.py.
    m.def("abi_screen_bounds_level",
          [](const torch::Tensor& coordinates, int64_t extent,
             int64_t grain) -> bool {
            try {
              if (!coordinates.defined()) return true;
              if (coordinates.device().type() != torch::kCPU) return true;
              if (!coordinates.is_contiguous()) return true;
              const int64_t n = coordinates.numel();
              if (n == 0) return false;  // nothing to violate; matches storage.py
              if (coordinates.dtype() != torch::kInt32 &&
                  coordinates.dtype() != torch::kInt64)
                return true;
              // The parameter layout of the call boundary's per-level COO check.
              scorch_native::AbiStructEntry want;
              want.add(coordinates);
              want.params = {scorch_native::AbiCheckBounds, extent, n};
              return scorch_native::abi_screen_through_memo(std::move(want), [&] {
                if (coordinates.dtype() == torch::kInt32)
                  return scorch_native::abi_screen_bounds_typed<int32_t>(
                      coordinates.data_ptr<int32_t>(), n, extent, grain);
                return scorch_native::abi_screen_bounds_typed<int64_t>(
                    coordinates.data_ptr<int64_t>(), n, extent, grain);
              });
            } catch (...) {
              return true;
            }
          },
          "Screen one level's coordinate bounds: false = every coordinate is in range",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("coordinates"), py::arg("extent"),
          py::arg("grain") = SCORCH_ABI_VALIDATE_GRAIN);

    // "COO coordinates ascend lexicographically across the levels, in level order".
    //
    // This one replaces the worst loop in the validator: the COO branch of
    // `_validate_index_storage` called .tolist() on every mode's index array and then
    // iterated over every NONZERO in Python, building two tuples per iteration. That
    // measured 0.40 us per nonzero -- 159 ms to wrap a 400,000-nonzero COO tensor --
    // and it is per nonzero, where the compressed loop it sat beside was per row.
    //
    // The comparison is the same one the Python loop performs: the first level that
    // differs decides, so this is a lexicographic test over the levels in order, not a
    // per-level ordering test.
    m.def("abi_screen_lex_levels",
          [](const std::vector<torch::Tensor>& levels, int64_t n,
             int64_t grain) -> bool {
            try {
              if (levels.empty()) return false;
              if (n <= 1) return false;
              const auto width = levels.front().dtype();
              for (const auto& level : levels) {
                if (!level.defined()) return true;
                if (level.device().type() != torch::kCPU) return true;
                if (!level.is_contiguous()) return true;
                if (level.dtype() != width) return true;
                if (level.numel() < n) return true;  // never read past an array
              }
              if (width != torch::kInt32 && width != torch::kInt64) return true;
              // The parameter layout of the call boundary's COO order check.
              scorch_native::AbiStructEntry want;
              for (const auto& level : levels) want.add(level);
              want.params = {scorch_native::AbiCheckLex, n};
              return scorch_native::abi_screen_through_memo(std::move(want), [&] {
                if (width == torch::kInt32) {
                  std::vector<const int32_t*> data;
                  data.reserve(levels.size());
                  for (const auto& level : levels)
                    data.push_back(level.data_ptr<int32_t>());
                  return scorch_native::abi_screen_lex_typed<int32_t>(data, n, grain);
                }
                std::vector<const int64_t*> data;
                data.reserve(levels.size());
                for (const auto& level : levels)
                  data.push_back(level.data_ptr<int64_t>());
                return scorch_native::abi_screen_lex_typed<int64_t>(data, n, grain);
              });
            } catch (...) {
              return true;
            }
          },
          "Screen COO lexicographic order: false = the coordinates ascend",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("levels"), py::arg("n"),
          py::arg("grain") = SCORCH_ABI_VALIDATE_GRAIN);

    // A resolved CSR x dense product (see plan.h). Built once per (operand,
    // free dimension) by scorch.plan and then invoked directly, so a repeated
    // matmul is one Python->C++ hop instead of a full re-resolution. `run`
    // returns None rather than raising when the call is outside what the plan
    // was built for; the caller falls back to the ordinary dispatch.
    py::class_<scorch_native::SpmmCsrPlan>(m, "SpmmCsrPlan")
      .def("run", &scorch_native::SpmmCsrPlan::run,
           "Serve this product, or return None to defer to the ordinary path",
           py::arg("a_values"), py::arg("b"), py::arg("nthreads") = -1,
           py::arg("atparallel") = false)
      .def("run_csr", &scorch_native::SpmmCsrPlan::run_csr,
           "run, for a torch sparse CSR operand: its current index arrays must be the "
           "ones the plan was built from, or the plan declines",
           py::arg("positions"), py::arg("coordinates"), py::arg("a_values"),
           py::arg("b"), py::arg("nthreads") = -1, py::arg("atparallel") = false)
      .def("run_gathered", &scorch_native::SpmmCsrPlan::run_gathered,
           "run, with a_values first set to src[perm[q]] for every nonzero q (by the "
           "product's team, where a v2 product has one); None where `run` would decline",
           py::arg("a_values"), py::arg("src"), py::arg("perm"), py::arg("b"),
           py::arg("nthreads") = -1, py::arg("atparallel") = false)
      .def_property_readonly("kind", &scorch_native::SpmmCsrPlan::kind)
      .def_property_readonly("rows", &scorch_native::SpmmCsrPlan::rows)
      .def_property_readonly("cols", &scorch_native::SpmmCsrPlan::cols)
      .def_property_readonly("nnz", &scorch_native::SpmmCsrPlan::nnz)
      .def_property_readonly("free_dim", &scorch_native::SpmmCsrPlan::free_dim)
      .def_property_readonly("served", &scorch_native::SpmmCsrPlan::served)
      .def_property_readonly("empty_rows", &scorch_native::SpmmCsrPlan::empty_rows)
      .def_property_readonly("borrows", &scorch_native::SpmmCsrPlan::borrows)
      .def_property_readonly("team_fills", &scorch_native::SpmmCsrPlan::team_fills)
      .def_property_readonly("wide_rows", &scorch_native::SpmmCsrPlan::wide_rows)
      .def_property_readonly("wide_segments", &scorch_native::SpmmCsrPlan::wide_segments);
    // A torch CSR operand's record that its transposed product A^T @ M is served by a carrier's
    // plan (planned_matmul.h): what the C entry serves x @ A from. Made by planned_matmul_gathered.
    py::class_<scorch_native::TransposedProductPlan>(m, "TransposedProductPlan")
      .def_property_readonly("served", &scorch_native::TransposedProductPlan::served)
      .def_property_readonly("free_dim", &scorch_native::TransposedProductPlan::free_dim)
      .def_property_readonly("buffer", &scorch_native::TransposedProductPlan::buffer);
    // The kernels the gradient of a sparse x dense product needs beyond the SpMM itself
    // (sparse_grad.h, used by scorch/autograd.py). They trust index arrays the caller has
    // screened, and gather_values a permutation csr_transpose produced.
    m.def("csr_transpose", &scorch_native::csr_transpose,
          "The CSR transpose of an m x n CSR structure: (positions, coordinates, perm), where "
          "nonzero q of the transpose is nonzero perm[q] of the original",
          py::call_guard<py::gil_scoped_release>(), py::arg("positions"),
          py::arg("coordinates"), py::arg("cols"));
    m.def("sddmm_csr", &scorch_native::sddmm_csr,
          "out[q] = x[i, :] . y[coordinates[q], :] for every stored q of row i (SDDMM)",
          py::call_guard<py::gil_scoped_release>(), py::arg("positions"),
          py::arg("coordinates"), py::arg("x"), py::arg("y"), py::arg("nthreads") = -1,
          py::arg("atparallel") = false);
    m.def("planned_matmul_gathered", &scorch_native::planned_matmul_gathered,
          "a @ b by a's plan, a's values first set to src[perm[q]] (SpmmCsrPlan::run_gathered); "
          "None when no plan serves it. A torch CSR `owner` whose transpose `a` carries is given "
          "a TransposedProductPlan for a served product",
          py::arg("a"), py::arg("b"), py::arg("src"), py::arg("perm"),
          py::arg("owner") = py::none());
    m.def("gather_values", &scorch_native::gather_values,
          "out[q] = values[perm[q]]: the values of a CSR transpose, with csr_transpose's perm",
          py::call_guard<py::gil_scoped_release>(), py::arg("values"), py::arg("perm"),
          py::arg("out"), py::arg("nthreads") = -1);
    m.def("make_spmm_csr_plan", &scorch_native::make_spmm_csr_plan,
          "Build a CSR x dense plan, or None when one cannot serve this shape",
          py::arg("kind"), py::arg("A_shape"), py::arg("A_mode_indices"),
          py::arg("A_values"), py::arg("free_dim"), py::arg("panel_free") = 0,
          py::arg("panel_contraction") = 0, py::arg("borrow") = false);
    // scorch.matmul's planned fast path in C (planned_matmul.h). ops.py binds what this
    // returns as `matmul`; every call it does not serve goes to `ordinary` unchanged.
    m.def("install_planned_matmul", &scorch_native::install_planned_matmul,
          "Build the C callable ops.py binds as matmul",
          py::arg("ordinary"), py::arg("stensor_type"), py::arg("tensor_type"),
          py::arg("generation"), py::arg("enabled"), py::arg("plans_attr"),
          py::arg("seen_attr"), py::arg("max_plans"), py::arg("probed_kw"),
          py::arg("match_host_threads"), py::arg("atparallel"), py::arg("module"),
          py::arg("doc"));
    // The backward of a planned product the C entry records for autograd (planned_matmul.h,
    // PlannedMatmulGrad); None withdraws it, and such products go to the Python function.
    m.def("set_planned_matmul_backward",
          [](py::object fn) {
            if (!fn.is_none() && !PyCallable_Check(fn.ptr()))
              throw py::type_error("fn must be callable or None");
            scorch_native::planned_matmul_set(scorch_native::planned_matmul_state().mm_backward,
                                              fn.is_none() ? nullptr : fn.ptr());
          },
          "Install autograd.planned_mm_backward for the C entry's recorded products",
          py::arg("fn"));
    // x @ A for a torch CSR A (planned_matmul.h, planned_matmul_torch_right): the backward of the
    // product PlannedMatmulGradT records, and the state key an operand's TransposedProductPlans
    // live under. None withdraws both, and every x @ A goes to the Python function.
    m.def("set_planned_matmul_right",
          [](py::object fn, py::object tplans_attr) {
            scorch_native::PlannedMatmul& s = scorch_native::planned_matmul_state();
            if (fn.is_none()) {
              scorch_native::planned_matmul_set(s.mm_t_backward, nullptr);
              Py_CLEAR(s.tplans_attr);
              return;
            }
            if (!PyCallable_Check(fn.ptr())) throw py::type_error("fn must be callable or None");
            if (!PyUnicode_Check(tplans_attr.ptr())) throw py::type_error("tplans_attr must be a str");
            PyObject* name = tplans_attr.ptr();
            Py_INCREF(name);
            PyUnicode_InternInPlace(&name);
            Py_XSETREF(s.tplans_attr, name);
            scorch_native::planned_matmul_set(s.mm_t_backward, fn.ptr());
          },
          "Install autograd.planned_mm_t_backward and plan.TRANSPOSED_PLANS_ATTR, so the C entry "
          "serves x @ A for a torch CSR A whose transposed product has a plan",
          py::arg("fn"), py::arg("tplans_attr") = py::none());

    py::class_<Tensor>(m, "Tensor")
      .def_readonly("storage", &Tensor::storage);
    py::class_<TensorStorage>(m, "TensorStorage")
      .def_readonly("value", &TensorStorage::value)
      .def_readonly("index", &TensorStorage::index);
    py::class_<TensorIndex>(m, "TensorIndex")
      .def_readonly("mode_indices", &TensorIndex::mode_indices);
  }
}
