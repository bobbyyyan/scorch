#include "header.h"

#include "spmm.h"
#include "kernels.h"

namespace scorch {
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_csr_float", &spmm_csr_float, "Sparse matrix multiplication (CSR)",
         pybind11::arg("result_shape"), pybind11::arg("A_shape"),
         pybind11::arg("A_mode_indices"), pybind11::arg("A_values"),
         pybind11::arg("B_shape"), pybind11::arg("B_mode_indices"),
         pybind11::arg("B_values"), pybind11::arg("tile_size") = 32);
    m.def("spmm_csr_float_optimized", &spmm_csr_float_optimized, "Optimized sparse matrix multiplication (CSR)",
         pybind11::arg("result_shape"), pybind11::arg("A_shape"),
         pybind11::arg("A_mode_indices"), pybind11::arg("A_values"),
         pybind11::arg("B_shape"), pybind11::arg("B_mode_indices"),
         pybind11::arg("B_values"), pybind11::arg("tile_size") = 128);
    m.def("spmm_csr_float_turbo", &spmm_csr_float_turbo, "Turbo-optimized sparse matrix multiplication (CSR)",
         pybind11::arg("result_shape"), pybind11::arg("A_shape"),
         pybind11::arg("A_mode_indices"), pybind11::arg("A_values"),
         pybind11::arg("B_shape"), pybind11::arg("B_mode_indices"),
         pybind11::arg("B_values"), pybind11::arg("tile_size") = 128);
    m.def("spmm_csr_float_ultra", &spmm_csr_float_ultra, "Ultra-optimized sparse matrix multiplication (CSR)",
         pybind11::arg("result_shape"), pybind11::arg("A_shape"),
         pybind11::arg("A_mode_indices"), pybind11::arg("A_values"),
         pybind11::arg("B_shape"), pybind11::arg("B_mode_indices"),
         pybind11::arg("B_values"), pybind11::arg("tile_size") = 256);
    m.def("spmm_csr_float_apex", &spmm_csr_float_apex, "Apex-optimized sparse matrix multiplication (CSR)",
         pybind11::arg("result_shape"), pybind11::arg("A_shape"),
         pybind11::arg("A_mode_indices"), pybind11::arg("A_values"),
         pybind11::arg("B_shape"), pybind11::arg("B_mode_indices"),
         pybind11::arg("B_values"), pybind11::arg("tile_size") = 256);
    m.def("spmm_csr_float_untiled", &spmm_csr_float_untiled, "Sparse matrix multiplication (CSR) (untiled)");
    m.def("spmm_coo_float", &spmm_coo_float, "Sparse matrix multiplication (COO)");
    m.def("spmspm_coo_float", &spmspm_coo_float_opt, "Sparse matrix-sparse matrix multiplication (COO)");
    m.def("spmspm_csr_float", &spmspm_csr<float>, "Sparse matrix-sparse matrix multiplication (CSR)");

    pybind11::class_<Tensor>(m, "Tensor")
      .def(pybind11::init<>())
      .def_readwrite("storage", &Tensor::storage);
    pybind11::class_<TensorStorage>(m, "TensorStorage")
      .def(pybind11::init<>())
      .def_readwrite("value", &TensorStorage::value)
      .def_readwrite("index", &TensorStorage::index);
    pybind11::class_<TensorIndex>(m, "TensorIndex")
      .def(pybind11::init<>())
      .def_readwrite("mode_indices", &TensorIndex::mode_indices);
  }
}
