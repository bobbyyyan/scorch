#include <torch/extension.h>

#include <vector>

typedef struct {
  std::vector<torch::Tensor> mode_indices;
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
} TacoTensor;

torch::Tensor add(torch::Tensor x, torch::Tensor y) { return x + y; }

TacoTensor mul(TacoTensor x, TacoTensor y) {
  TacoTensor z;
  return z;
}

// C++ kernel to elementwise multiply two sparse vectors
// input arguments are two list of torch::Tensor's
// output is a list of torch::Tensor's
TacoTensor elemwise_mul(std::vector<torch::Tensor> x_indices,
                                        torch::Tensor x_values,
                                        std::vector<torch::Tensor> y_indices,
                                        torch::Tensor y_values) {
  torch::Tensor x_pos = x_indices[0];
  torch::Tensor x_crd = x_indices[1];
  torch::Tensor y_pos = y_indices[0];
  torch::Tensor y_crd = y_indices[1];

  std::cout << "x_pos: " << x_pos << std::endl;
  std::cout << "x_crd: " << x_crd << std::endl;
  std::cout << "y_pos: " << y_pos << std::endl;
  std::cout << "y_crd: " << y_crd << std::endl;

  torch::Tensor z_values = torch::zeros(2);

  std::vector<torch::Tensor> z_indices;


  torch::Tensor z_pos = torch::zeros(2);

  int i_z = 0;

  int i_x = x_pos[0].item<int>();
  int px_end = x_pos[1].item<int>();
  int i_y = y_pos[0].item<int>();
  int py_end = y_pos[1].item<int>();

  while (i_x < px_end && i_y < py_end) {
    int i_x_crd = x_crd[i_x].item<int>();
    int i_y_crd = y_crd[i_y].item<int>();
    if (i_x_crd == i_y_crd) {
      z_values[i_z] = x_values[i_x] * y_values[i_y];
      i_x++;
      i_y++;
      i_z++;
    } else if (i_x_crd < i_y_crd) {
      i_x++;
    } else {
      i_y++;
    }
  }

  z_pos[1] = i_z;

  z_indices.push_back(x_pos);

  // return {z_pos, z_values};
  // return a TacoTensor with _storage._index.mode_indices = z_indices and _storage._value = z_value
  TacoTensor z;
  z._storage._index.mode_indices = z_indices;
  z._storage._value = z_values;
  return z;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add");
  m.def("mul", &mul, "Mul");
  m.def("elemwise_mul", &elemwise_mul, "Element-wise Mul");
  pybind11::class_<TacoTensor>(m, "TacoTensor")
      .def(pybind11::init<>())
      .def_readwrite("_storage", &TacoTensor::_storage);
  pybind11::class_<TensorStorage>(m, "TensorStorage")
      .def(pybind11::init<>())
      .def_readwrite("_value", &TensorStorage::_value)
      .def_readwrite("_index", &TensorStorage::_index);
  pybind11::class_<TensorIndex>(m, "TensorIndex")
      .def(pybind11::init<>())
      .def_readwrite("mode_indices", &TensorIndex::mode_indices);
}
