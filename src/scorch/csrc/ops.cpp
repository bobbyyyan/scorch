#include "header.h"

#include <limits>
#include <string>

#include "native_abi.h"
#include "spmm.h"
#include "kernels.h"
#include "plan.h"

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
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmul,
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
          scorch_native::validate_binary_inputs(
              op, scorch_native::BinaryContract::CsrDenseMatmul,
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
              op, scorch_native::BinaryContract::CsrDenseMatmul,
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
              op, scorch_native::BinaryContract::CsrDenseMatmul,
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
          "Number of live entries across both ABI validation memos");

    // The cache size the KERNEL gates on, so a test can check it against the one
    // tiling.query_llc gates on. Two gates that disagree about the machine would
    // route a product to a tiled kernel the kernel then declines to stream for, and
    // nothing in either layer would report the disagreement.
    m.def("scorch_llc_bytes", &scorch_llc_bytes,
          "Effective last-level cache in bytes, as the kernels compute it");

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
          "nthreads_override, work_true). work is nnz*max(k,16); work_true is "
          "nnz*k and defaults to work, which reproduces the caller that has only "
          "the one number.",
          py::arg("work"), py::arg("rows"), py::arg("nthreads_override"),
          py::arg("work_true") = -1);

    // The same fused structural pass, offered to Scorch's own Python-side validator.
    //
    // `_validate_index_storage` in storage.py runs on every STensor built over a
    // compressed level -- from_torch, to_sparse, a relayout, and every generated
    // kernel's result -- and its bounds and sortedness checks are separate whole-array
    // torch operations: about five passes over the index arrays plus a bool temporary,
    // which measured 304 us of the 385 us it took to wrap a 640k-nonzero result. This
    // screen does the same work in ONE pass with no allocation.
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
              if (positions.dtype() == torch::kInt32) {
                if (nnz > (int64_t)std::numeric_limits<int32_t>::max()) return true;
                return scorch_native::abi_screen_compressed_typed<int32_t>(
                    positions.data_ptr<int32_t>(), coordinates.data_ptr<int32_t>(),
                    parents, nnz, extent, require_sorted, grain);
              }
              if (positions.dtype() == torch::kInt64) {
                return scorch_native::abi_screen_compressed_typed<int64_t>(
                    positions.data_ptr<int64_t>(), coordinates.data_ptr<int64_t>(),
                    parents, nnz, extent, require_sorted, grain);
              }
              return true;
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
              if (coordinates.dtype() == torch::kInt32) {
                return scorch_native::abi_screen_bounds_typed<int32_t>(
                    coordinates.data_ptr<int32_t>(), n, extent, grain);
              }
              if (coordinates.dtype() == torch::kInt64) {
                return scorch_native::abi_screen_bounds_typed<int64_t>(
                    coordinates.data_ptr<int64_t>(), n, extent, grain);
              }
              return true;
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
              if (width == torch::kInt32) {
                std::vector<const int32_t*> data;
                data.reserve(levels.size());
                for (const auto& level : levels)
                  data.push_back(level.data_ptr<int32_t>());
                return scorch_native::abi_screen_lex_typed<int32_t>(data, n, grain);
              }
              if (width == torch::kInt64) {
                std::vector<const int64_t*> data;
                data.reserve(levels.size());
                for (const auto& level : levels)
                  data.push_back(level.data_ptr<int64_t>());
                return scorch_native::abi_screen_lex_typed<int64_t>(data, n, grain);
              }
              return true;
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
      .def_property_readonly("kind", &scorch_native::SpmmCsrPlan::kind)
      .def_property_readonly("rows", &scorch_native::SpmmCsrPlan::rows)
      .def_property_readonly("cols", &scorch_native::SpmmCsrPlan::cols)
      .def_property_readonly("nnz", &scorch_native::SpmmCsrPlan::nnz)
      .def_property_readonly("free_dim", &scorch_native::SpmmCsrPlan::free_dim)
      .def_property_readonly("served", &scorch_native::SpmmCsrPlan::served);
    m.def("make_spmm_csr_plan", &scorch_native::make_spmm_csr_plan,
          "Build a CSR x dense plan, or None when one cannot serve this shape",
          py::arg("kind"), py::arg("A_shape"), py::arg("A_mode_indices"),
          py::arg("A_values"), py::arg("free_dim"), py::arg("panel_free") = 0,
          py::arg("panel_contraction") = 0);

    py::class_<Tensor>(m, "Tensor")
      .def_readonly("storage", &Tensor::storage);
    py::class_<TensorStorage>(m, "TensorStorage")
      .def_readonly("value", &TensorStorage::value)
      .def_readonly("index", &TensorStorage::index);
    py::class_<TensorIndex>(m, "TensorIndex")
      .def_readonly("mode_indices", &TensorIndex::mode_indices);
  }
}
