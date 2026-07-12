#include <torch/extension.h>

typedef struct {
  std::vector<std::vector<torch::Tensor>> mode_indices;
} TensorIndex;

typedef struct {
  TensorIndex _index;
  // a list of torch::Tensor
  torch::Tensor _value;

} TensorStorage;

typedef struct {
  TensorStorage _storage;
  // a tuple of ints as shape
  std::vector<int> _shape;
} Tensor;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Tensor>(m, "Tensor")
    .def(pybind11::init<>())
    .def_readwrite("_storage", &Tensor::_storage);
  pybind11::class_<TensorStorage>(m, "TensorStorage")
    .def(pybind11::init<>())
    .def_readwrite("_value", &TensorStorage::_value)
    .def_readwrite("_index", &TensorStorage::_index);
  pybind11::class_<TensorIndex>(m, "TensorIndex")
    .def(pybind11::init<>())
    .def_readwrite("mode_indices", &TensorIndex::mode_indices);
}
