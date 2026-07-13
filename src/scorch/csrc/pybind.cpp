#include "header.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Tensor>(m, "Tensor")
    .def(pybind11::init<>())
    .def_readwrite("storage", &Tensor::storage)
    .def_readwrite("shape", &Tensor::shape);
  pybind11::class_<TensorStorage>(m, "TensorStorage")
    .def(pybind11::init<>())
    .def_readwrite("value", &TensorStorage::value)
    .def_readwrite("index", &TensorStorage::index);
  pybind11::class_<TensorIndex>(m, "TensorIndex")
    .def(pybind11::init<>())
    .def_readwrite("mode_indices", &TensorIndex::mode_indices);
}
