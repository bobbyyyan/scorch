#include <torch/extension.h>

#include <vector>

#include "header.cpp"

TacoTensor sparse_vector_mul_manual(
  std::vector<int> result_shape,
  std::vector<std::vector<torch::Tensor>> b_mode_indices, torch::Tensor b_values,
  std::vector<std::vector<torch::Tensor>> c_mode_indices, torch::Tensor c_values
) {
  // Get tensor level arrays
  torch::Tensor b0_pos = b_mode_indices[0][0];
  torch::Tensor b0_crd = b_mode_indices[0][1];
  torch::Tensor c0_pos = c_mode_indices[0][0];
  torch::Tensor c0_crd = c_mode_indices[0][1];

  // Get tensor value arrays
  cvector<float> a_values;

  cvector<int> a0_pos;
  cvector<int> a0_crd;

  // Initialize result value array index
  int i_a = 0;

  // Initialize iterators
  int pb0 = b0_pos[0].item<int>();;
  int pb0_end = b0_pos[1].item<int>();;
  int pc0 = c0_pos[0].item<int>();;
  int pc0_end = c0_pos[1].item<int>();;



  while (pb0 < pb0_end && pc0 < pc0_end) {
    // Load coordinates
    int i_b = b0_crd[pb0].item<int>();;
    int i_c = c0_crd[pc0].item<int>();;
    // Resolve coordinates
    int i = std::min({i_b, i_c});
    if (i_b == i && i_c == i) {
      a_values.push_back(b_values[pb0].item<float>() * c_values[pc0].item<float>());
      a0_crd.push_back(i);
      i_a++;
    }
    // Advance iterators
    pb0 += (int) (i_b == i);
    pc0 += (int) (i_c == i);
  }

  a0_pos.push_back(i_a);

  TacoTensor a;

  torch::Tensor a0_pos_torch = torch::from_blob(a0_pos.data(), {a0_pos.size()}, a0_pos.get_deleter(), torch::kInt);
  torch::Tensor a0_crd_torch = torch::from_blob(a0_crd.data(), {a0_crd.size()}, a0_crd.get_deleter(), torch::kInt);

  a._storage._index.mode_indices = {{a0_pos_torch, a0_crd_torch}};
  a._storage._value = torch::from_blob(a_values.data(), {i_a}, a_values.get_deleter(), torch::kFloat32);
  // a._storage._value = torch::from_blob(a_values.data(), {i_a}, torch::TensorOptions().dtype(torch::kInt));

  return a;

}

TacoTensor elemwise_vector_mul_sss(std::vector<int> result_shape, std::vector<std::vector<torch::Tensor>> b_mode_indices, torch::Tensor b_values, std::vector<std::vector<torch::Tensor>> c_mode_indices, torch::Tensor c_values) {
  // Get tensor level arrays
  torch::Tensor b0_pos = b_mode_indices[0][0];
  torch::Tensor b0_crd = b_mode_indices[0][1];
  torch::Tensor c0_pos = c_mode_indices[0][0];
  torch::Tensor c0_crd = c_mode_indices[0][1];

  // Init result level indices
  cvector<int> a0_pos;
  cvector<int> a0_crd;
  a0_pos[0] = 0;
  int pa0 = 0;

  // Initialize result value array
  cvector<float> a_values;

  // Initialize iterators
  int pb0 = b0_pos[0].item<int>();
  int pb0_end = b0_pos[1].item<int>();
  int pc0 = c0_pos[0].item<int>();
  int pc0_end = c0_pos[1].item<int>();

  while (pb0 < pb0_end && pc0 < pc0_end) {
    // Load coordinates
    int i_b = b0_crd[pb0].item<int>();
    int i_c = c0_crd[pc0].item<int>();
    // Resolve coordinates
    int i = std::min({i_b, i_c});


    // Inner loops over child regions
    if (i_b == i && i_c == i) {
      a_values[pa0] = b_values[pb0].item<float>() * c_values[pc0].item<float>();
      // Set coordinates
      a0_crd[pa0] = i;
      pa0++;
    }

    // Advance iterators
    pb0 += (int) i_b == i;
    pc0 += (int) i_c == i;
  }

  // Set position index
  a0_pos.push_back(a0_crd.size());

  // Assemble result
  TacoTensor a;
  torch::Tensor a0_pos_torch = torch::from_blob(a0_pos.data(), {a0_pos.size()}, a0_pos.get_deleter(), torch::kInt);
  torch::Tensor a0_crd_torch = torch::from_blob(a0_crd.data(), {a0_crd.size()}, a0_crd.get_deleter(), torch::kInt);
  torch::Tensor a_values_torch = torch::from_blob(a_values.data(), {a_values.size()}, a_values.get_deleter(), torch::kFloat32);
  a._storage._index.mode_indices = {{a0_pos_torch, a0_crd_torch}};
  a._storage._value = a_values_torch;
  return a;
}

TacoTensor sparse_vector_mul(
  std::vector<int> result_shape,
  std::vector<std::vector<torch::Tensor>> b_mode_indices, torch::Tensor b_values,
  std::vector<std::vector<torch::Tensor>> c_mode_indices, torch::Tensor c_values
) {
  // Get tensor level arrays
  torch::Tensor b0_pos = b_mode_indices[0][0];
  torch::Tensor b0_crd = b_mode_indices[0][1];
  torch::Tensor c0_pos = c_mode_indices[0][0];
  torch::Tensor c0_crd = c_mode_indices[0][1];

  // Init result level indices
  cvector<int> a0_pos;
  a0_pos[0] = 0;
  cvector<int> a0_crd;

  // Initialize result value arrays and index
  cvector<float> a_values;
  int i_a = 0;

  // Initialize iterators
  int pb0 = b0_pos[0].item<int>();
  int pb0_end = b0_pos[1].item<int>();
  int pc0 = c0_pos[0].item<int>();
  int pc0_end = c0_pos[1].item<int>();

  while (pb0 < pb0_end && pc0 < pc0_end) {
    // Load coordinates
    int i_b = b0_crd[pb0].item<int>();
    int i_c = c0_crd[pc0].item<int>();

    // Resolve coordinates
    int i = std::min({i_b, i_c});

    // Inner loops over child regions
    if (i_b == i && i_c == i) {
      a_values[i_a] = b_values[pb0].item<float>() * c_values[pc0].item<float>();
      i_a++;
    }

    // Advance iterators
    pb0 += (int) i_b == i;
    pc0 += (int) i_c == i;
  }

  a0_pos.push_back(i_a);

  TacoTensor a;

  torch::Tensor a0_pos_torch = torch::from_blob(a0_pos.data(), {a0_pos.size()}, a0_pos.get_deleter(), torch::kInt);
  torch::Tensor a0_crd_torch = torch::from_blob(a0_crd.data(), {a0_crd.size()}, a0_crd.get_deleter(), torch::kInt);

  a._storage._index.mode_indices = {{a0_pos_torch, a0_crd_torch}};
  a._storage._value = torch::from_blob(a_values.data(), {i_a}, a_values.get_deleter(), torch::kFloat32);
  // a._storage._value = torch::from_blob(a_values.data(), {i_a}, torch::TensorOptions().dtype(torch::kInt));

  return a;

}


torch::Tensor add(torch::Tensor x, torch::Tensor y) { return x + y; }

TacoTensor mul(TacoTensor x, TacoTensor y) {
  TacoTensor z;
  return z;
}

torch::Tensor get_rand_matrix(int rows, int cols) {
  cvector<int> v;

  for (int i = 0; i < rows * cols; i++) {
    v.push_back(i);
  }

  int v_size = v.size();

  torch::Tensor torch_tensor = torch::from_blob(v.data(), {rows, cols}, v.get_deleter(), torch::kInt);
  return torch_tensor;
  // return torch::rand({rows, cols});
}

TacoTensor get_rand_matrix_tt(int rows, int cols) {
  // Get a random matrix as TT tensor
  torch::Tensor rand_tensor = get_rand_matrix(rows, cols);

  TacoTensor result;

  TensorStorage storage;

  storage._value = rand_tensor;

  result._storage = storage;

  return result;
}

// C++ kernel to elementwise multiply two sparse vectors
// input arguments are two list of torch::Tensor's
// output is a list of torch::Tensor's
//TacoTensor elemwise_mul(std::vector<torch::Tensor> x_indices,
//                                        torch::Tensor x_values,
//                                        std::vector<torch::Tensor> y_indices,
//                                        torch::Tensor y_values) {
//  torch::Tensor x_pos = x_indices[0];
//  torch::Tensor x_crd = x_indices[1];
//  torch::Tensor y_pos = y_indices[0];
//  torch::Tensor y_crd = y_indices[1];
//
//  std::cout << "x_pos: " << x_pos << std::endl;
//  std::cout << "x_crd: " << x_crd << std::endl;
//  std::cout << "y_pos: " << y_pos << std::endl;
//  std::cout << "y_crd: " << y_crd << std::endl;
//
//  torch::Tensor z_values = torch::zeros(2);
//
//  std::vector<torch::Tensor> z_indices;
//
//
//  torch::Tensor z_pos = torch::zeros(2);
//
//  int i_z = 0;
//
//  int i_x = x_pos[0].item<int>();
//  int px_end = x_pos[1].item<int>();
//  int i_y = y_pos[0].item<int>();
//  int py_end = y_pos[1].item<int>();
//
//  while (i_x < px_end && i_y < py_end) {
//    int i_x_crd = x_crd[i_x].item<int>();
//    int i_y_crd = y_crd[i_y].item<int>();
//    if (i_x_crd == i_y_crd) {
//      z_values[i_z] = x_values[i_x] * y_values[i_y];
//      i_x++;
//      i_y++;
//      i_z++;
//    } else if (i_x_crd < i_y_crd) {
//      i_x++;
//    } else {
//      i_y++;
//    }
//  }
//
//  z_pos[1] = i_z;
//
//  z_indices.push_back(x_pos);
//
//  // return {z_pos, z_values};
//  // return a TacoTensor with _storage._index.mode_indices = z_indices and _storage._value = z_value
//  TacoTensor z;
//  z._storage._index.mode_indices = z_indices;
//  z._storage._value = z_values;
//  return z;
//
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add");
  m.def("mul", &mul, "Mul");
  m.def("get_rand_matrix", &get_rand_matrix, "Get a random matrix of the input shape");
  m.def("get_rand_matrix_tt", &get_rand_matrix_tt, "Get a random matrix of the input shape as TT tensor");
  m.def("sparse_vector_mul", &sparse_vector_mul, "Sparse vector-vector multiplication");
  m.def("elemwise_vector_mul_sss", &elemwise_vector_mul_sss, "Elementwise vector multiplication (SSS)");
//  m.def("elemwise_mul", &elemwise_mul, "Element-wise Mul");
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
