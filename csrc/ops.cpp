#include "header.h"

#include "spmm.h"
#include "kernels.h"

namespace scorch {
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_csr_float", &spmm_csr_float, "Sparse matrix multiplication (CSR)");
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
