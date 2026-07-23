#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Checked input views shared by the prebuilt extension and JIT-generated kernels.
//
// All Scorch native kernels are CPU kernels and the legacy implementations use
// int32_t level arrays.  The helpers below validate the complete public ABI before
// any kernel indexes a C++ container or dereferences a pointer.  int64 index tensors
// are accepted, range checked, and converted in the *by-value* mode_indices argument
// owned by the native call; caller tensors and Python lists are never modified.
namespace scorch_native {

enum class BinaryContract {
  CsrDenseMatmul,
  CsrCsrMatmul,
  CsrDenseMatvec,
  CooDenseMatmul,
  CooCooMatmul,
};

enum class LevelKind : int {
  Dense = 0,
  Compressed = 1,
  Coordinate = 2,
};

inline std::string argument_name(const char* op, const char* argument) {
  return std::string(op) + ": " + argument;
}

inline std::string shape_string(const std::vector<int64_t>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) result += ", ";
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

template <typename Dim>
inline std::vector<int64_t> checked_shape(const std::vector<Dim>& shape,
                                          int64_t expected_rank,
                                          const char* op,
                                          const char* argument) {
  TORCH_CHECK(static_cast<int64_t>(shape.size()) == expected_rank,
              argument_name(op, argument), " must have rank ", expected_rank,
              ", got ", shape.size());
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = static_cast<int64_t>(shape[i]);
    TORCH_CHECK(dim >= 0, argument_name(op, argument), " has negative dimension ",
                dim, " at mode ", i);
    TORCH_CHECK(dim <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " dimension ", i,
                " exceeds the current native int32 loop bound: ", dim);
    result.push_back(dim);
  }
  return result;
}

template <typename Dim>
inline std::vector<int> narrow_legacy_shape(const std::vector<Dim>& shape,
                                            const char* op,
                                            const char* argument) {
  std::vector<int> result;
  result.reserve(shape.size());
  for (size_t mode = 0; mode < shape.size(); ++mode) {
    const int64_t extent = static_cast<int64_t>(shape[mode]);
    TORCH_CHECK(extent >= 0 && extent <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " extent ", extent, " at mode ",
                mode, " cannot be represented by the legacy int32 kernel ABI");
    result.push_back(static_cast<int>(extent));
  }
  return result;
}

inline int64_t checked_product(const std::vector<int64_t>& shape,
                               const char* op, const char* argument,
                               bool require_legacy_int_capacity = false) {
  int64_t product = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = shape[i];
    TORCH_CHECK(dim == 0 ||
                    product <= std::numeric_limits<int64_t>::max() / dim,
                argument_name(op, argument), " element count overflows int64 at mode ",
                i);
    product *= dim;
  }
  if (require_legacy_int_capacity) {
    TORCH_CHECK(product <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " element count ", product,
                " exceeds the current native int32 allocation bound");
  }
  return product;
}

template <typename Dim>
inline std::vector<int64_t> checked_expected_shape(
    const std::vector<Dim>& shape, const std::vector<int64_t>& expected,
    const char* op, const char* argument,
    bool require_legacy_int_capacity = false) {
  const auto actual = checked_shape(shape, static_cast<int64_t>(expected.size()),
                                    op, argument);
  TORCH_CHECK(actual == expected, argument_name(op, argument), " must equal ",
              shape_string(expected), ", got ", shape_string(actual));
  checked_product(actual, op, argument, require_legacy_int_capacity);
  return actual;
}

inline void check_tensor_common(const torch::Tensor& tensor,
                                torch::ScalarType expected_dtype,
                                const char* op, const char* argument,
                                int64_t expected_rank = -1) {
  TORCH_CHECK(tensor.defined(), argument_name(op, argument), " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), argument_name(op, argument),
              " must be on CPU, got ", tensor.device());
  TORCH_CHECK(tensor.layout() == torch::kStrided,
              argument_name(op, argument), " must use strided storage");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype,
              argument_name(op, argument), " must have dtype ",
              c10::toString(expected_dtype), ", got ",
              c10::toString(tensor.scalar_type()));
  TORCH_CHECK(tensor.is_contiguous(), argument_name(op, argument),
              " must be contiguous");
  TORCH_CHECK(!tensor.is_neg(), argument_name(op, argument),
              " must resolve its lazy negative view bit");
  TORCH_CHECK(!tensor.is_conj(), argument_name(op, argument),
              " must resolve its lazy conjugate view bit");
  if (expected_rank >= 0) {
    TORCH_CHECK(tensor.dim() == expected_rank, argument_name(op, argument),
                " must have rank ", expected_rank, ", got ", tensor.dim());
  }
}

inline void check_flat_values(const torch::Tensor& values,
                              torch::ScalarType expected_dtype,
                              int64_t expected_numel, const char* op,
                              const char* argument) {
  check_tensor_common(values, expected_dtype, op, argument, 1);
  TORCH_CHECK(values.numel() == expected_numel, argument_name(op, argument),
              " must contain ", expected_numel, " elements, got ",
              values.numel());
  TORCH_CHECK(expected_numel == 0 ||
                  static_cast<uint64_t>(expected_numel) <=
                      std::numeric_limits<size_t>::max() / values.element_size(),
              argument_name(op, argument), " byte size overflows size_t");
}

inline torch::Tensor checked_index_tensor(torch::Tensor index,
                                          const char* op,
                                          const std::string& argument) {
  TORCH_CHECK(index.defined(), op, ": ", argument, " must be defined");
  TORCH_CHECK(index.device().is_cpu(), op, ": ", argument,
              " must be on CPU, got ", index.device());
  TORCH_CHECK(index.layout() == torch::kStrided, op, ": ", argument,
              " must use strided storage");
  TORCH_CHECK(index.dim() == 1, op, ": ", argument, " must be 1-D, got rank ",
              index.dim());
  TORCH_CHECK(index.is_contiguous(), op, ": ", argument,
              " must be contiguous");
  TORCH_CHECK(!index.is_neg(), op, ": ", argument,
              " must resolve its lazy negative view bit");
  TORCH_CHECK(!index.is_conj(), op, ": ", argument,
              " must resolve its lazy conjugate view bit");
  TORCH_CHECK(index.scalar_type() == torch::kInt32 ||
                  index.scalar_type() == torch::kInt64,
              op, ": ", argument, " must have dtype int32 or int64, got ",
              c10::toString(index.scalar_type()));
  TORCH_CHECK(index.numel() <= std::numeric_limits<int>::max(), op, ": ",
              argument, " length exceeds the current int32 native bound");

  if (index.scalar_type() == torch::kInt64) {
    const int64_t* data = index.data_ptr<int64_t>();
    for (int64_t i = 0; i < index.numel(); ++i) {
      TORCH_CHECK(data[i] >= std::numeric_limits<int32_t>::min() &&
                      data[i] <= std::numeric_limits<int32_t>::max(),
                  op, ": ", argument, " element ", i, " (", data[i],
                  ") cannot be represented as int32");
    }
    index = index.to(torch::kInt32);
  }
  return index;
}

inline void check_common_index_dtype(const torch::Tensor& index,
                                     bool& has_index_dtype,
                                     torch::ScalarType& index_dtype,
                                     const char* op,
                                     const std::string& argument) {
  if (!has_index_dtype) {
    index_dtype = index.scalar_type();
    has_index_dtype = true;
    return;
  }
  TORCH_CHECK(index.scalar_type() == index_dtype, op, ": ", argument,
              " must use the common index dtype ", c10::toString(index_dtype),
              ", got ", c10::toString(index.scalar_type()));
}

struct DenseInputView {
  std::vector<int64_t> shape;
  torch::Tensor values;
  const void* data;

  DenseInputView(std::vector<int64_t> shape_, torch::Tensor values_)
      : shape(std::move(shape_)),
        values(std::move(values_)),
        data(values.data_ptr()) {}
};

struct CsrInputView {
  std::vector<int64_t> shape;
  torch::Tensor positions;
  torch::Tensor coordinates;
  torch::Tensor values;
  const int32_t* pos;
  const int32_t* crd;
  const void* data;
  int64_t nnz;

  CsrInputView(std::vector<int64_t> shape_, torch::Tensor positions_,
               torch::Tensor coordinates_, torch::Tensor values_)
      : shape(std::move(shape_)),
        positions(std::move(positions_)),
        coordinates(std::move(coordinates_)),
        values(std::move(values_)),
        pos(positions.data_ptr<int32_t>()),
        crd(coordinates.data_ptr<int32_t>()),
        data(values.data_ptr()),
        nnz(values.numel()) {}
};

struct CooInputView {
  std::vector<int64_t> shape;
  std::vector<torch::Tensor> coordinates;
  torch::Tensor values;
  std::vector<const int32_t*> crd;
  const void* data;
  int64_t nnz;

  CooInputView(std::vector<int64_t> shape_,
               std::vector<torch::Tensor> coordinates_,
               torch::Tensor values_)
      : shape(std::move(shape_)),
        coordinates(std::move(coordinates_)),
        values(std::move(values_)),
        data(values.data_ptr()),
        nnz(values.numel()) {
    crd.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
      crd.push_back(coordinate.data_ptr<int32_t>());
    }
  }
};

template <typename Dim>
inline DenseInputView checked_dense_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    int64_t expected_rank, const char* op, const char* argument) {
  const auto logical_shape = checked_shape(shape, expected_rank, op, argument);
  const int64_t numel = checked_product(logical_shape, op, argument, true);
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " mode_indices must contain ",
              logical_shape.size(), " levels, got ", mode_indices.size());
  for (size_t level = 0; level < mode_indices.size(); ++level) {
    TORCH_CHECK(mode_indices[level].empty(), argument_name(op, argument),
                " dense level ", level, " must not contain index tensors");
  }
  check_flat_values(values, expected_dtype, numel, op,
                    (std::string(argument) + " values").c_str());
  return DenseInputView(logical_shape, values);
}

template <typename Dim>
inline CsrInputView checked_csr_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    const char* op, const char* argument, bool require_sorted = true) {
  const auto logical_shape = checked_shape(shape, 2, op, argument);
  TORCH_CHECK(mode_indices.size() == 2, argument_name(op, argument),
              " CSR mode_indices must contain exactly 2 levels, got ",
              mode_indices.size());
  TORCH_CHECK(mode_indices[0].empty(), argument_name(op, argument),
              " CSR dense level 0 must be empty");
  TORCH_CHECK(mode_indices[1].size() == 2, argument_name(op, argument),
              " CSR compressed level 1 must contain [positions, coordinates]");

  const auto raw_positions = mode_indices[1][0];
  const auto raw_coordinates = mode_indices[1][1];
  mode_indices[1][0] = checked_index_tensor(
      raw_positions, op, std::string(argument) + " positions");
  mode_indices[1][1] = checked_index_tensor(
      raw_coordinates, op, std::string(argument) + " coordinates");
  TORCH_CHECK(raw_positions.scalar_type() == raw_coordinates.scalar_type(),
              argument_name(op, argument),
              " positions and coordinates must use one common index dtype");
  auto positions = mode_indices[1][0];
  auto coordinates = mode_indices[1][1];

  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);
  const int64_t rows = logical_shape[0];
  const int64_t cols = logical_shape[1];
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), argument_name(op, argument),
              " nnz exceeds the current int32 native bound");
  TORCH_CHECK(coordinates.numel() == nnz, argument_name(op, argument),
              " coordinate/value nnz mismatch: ", coordinates.numel(), " vs ",
              nnz);
  TORCH_CHECK(positions.numel() == rows + 1, argument_name(op, argument),
              " positions length must be rows + 1 (", rows + 1, "), got ",
              positions.numel());

  const int32_t* pos = positions.data_ptr<int32_t>();
  const int32_t* crd = coordinates.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, argument_name(op, argument),
              " positions[0] must be 0, got ", pos[0]);
  int32_t previous = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t start = pos[row];
    const int32_t end = pos[row + 1];
    TORCH_CHECK(start >= previous && start >= 0 && end >= start && end <= nnz,
                argument_name(op, argument), " invalid CSR span for row ", row,
                ": [", start, ", ", end, ") with nnz ", nnz);
    for (int32_t p = start; p < end; ++p) {
      TORCH_CHECK(crd[p] >= 0 && crd[p] < cols,
                  argument_name(op, argument), " coordinate ", crd[p],
                  " at position ", p, " is outside [0, ", cols, ")");
      if (require_sorted && p > start) {
        TORCH_CHECK(
            crd[p - 1] < crd[p], argument_name(op, argument),
            " coordinates must be strictly increasing within each CSR row; row ",
            row, " is non-increasing at position ", p);
      }
    }
    previous = end;
  }
  TORCH_CHECK(pos[rows] == nnz, argument_name(op, argument),
              " terminal position must equal nnz (", nnz, "), got ", pos[rows]);
  return CsrInputView(logical_shape, positions, coordinates, values);
}

template <typename Dim>
inline CooInputView checked_coo_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    int64_t expected_rank, const char* op, const char* argument,
    bool require_lexicographic_order = true) {
  const auto logical_shape = checked_shape(shape, expected_rank, op, argument);
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " COO mode_indices must contain ",
              logical_shape.size(), " coordinate levels, got ",
              mode_indices.size());
  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), argument_name(op, argument),
              " nnz exceeds the current int32 native bound");

  std::vector<torch::Tensor> coordinates;
  coordinates.reserve(logical_shape.size());
  bool has_index_dtype = false;
  torch::ScalarType index_dtype = torch::kInt32;
  for (size_t level = 0; level < logical_shape.size(); ++level) {
    TORCH_CHECK(mode_indices[level].size() == 1,
                argument_name(op, argument), " COO level ", level,
                " must contain exactly one coordinate tensor");
    const auto raw_coordinate = mode_indices[level][0];
    mode_indices[level][0] = checked_index_tensor(
        raw_coordinate, op,
        std::string(argument) + " coordinate level " + std::to_string(level));
    check_common_index_dtype(raw_coordinate, has_index_dtype, index_dtype, op,
                             std::string(argument) + " coordinate level " +
                                 std::to_string(level));
    TORCH_CHECK(mode_indices[level][0].numel() == nnz,
                argument_name(op, argument), " COO level ", level,
                " length must equal values nnz (", nnz, "), got ",
                mode_indices[level][0].numel());
    const int32_t* crd = mode_indices[level][0].data_ptr<int32_t>();
    for (int64_t p = 0; p < nnz; ++p) {
      TORCH_CHECK(crd[p] >= 0 && crd[p] < logical_shape[level],
                  argument_name(op, argument), " coordinate ", crd[p],
                  " at level ", level, " position ", p, " is outside [0, ",
                  logical_shape[level], ")");
    }
    coordinates.push_back(mode_indices[level][0]);
  }

  if (require_lexicographic_order && nnz > 1) {
    for (int64_t p = 1; p < nnz; ++p) {
      bool greater = false;
      bool different = false;
      for (size_t level = 0; level < coordinates.size(); ++level) {
        const int32_t* crd = coordinates[level].data_ptr<int32_t>();
        if (crd[p] != crd[p - 1]) {
          greater = crd[p] > crd[p - 1];
          different = true;
          break;
        }
      }
      TORCH_CHECK(!different || greater, argument_name(op, argument),
                  " COO coordinates must be lexicographically ordered; order ",
                  "decreases at position ", p);
    }
  }
  return CooInputView(logical_shape, coordinates, values);
}

template <typename Dim>
inline void check_legacy_output_shape(const std::vector<Dim>& result_shape,
                                      int64_t expected_rank, const char* op) {
  const auto shape = checked_shape(result_shape, expected_rank, op, "result_shape");
  checked_product(shape, op, "result_shape", true);
}

template <typename Dim>
inline void validate_binary_inputs(
    const char* op, BinaryContract contract, torch::ScalarType dtype,
    const std::vector<Dim>& result_shape, const std::vector<Dim>& a_shape,
    std::vector<std::vector<torch::Tensor>>& a_mode_indices,
    const torch::Tensor& a_values, const std::vector<Dim>& b_shape,
    std::vector<std::vector<torch::Tensor>>& b_mode_indices,
    const torch::Tensor& b_values) {
  if (contract == BinaryContract::CsrDenseMatvec) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 1,
                                      op, "B");
    check_legacy_output_shape(result_shape, 1, op);
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0], op,
                ": result_shape must be [A.rows]");
    return;
  }

  check_legacy_output_shape(result_shape, 2, op);
  if (contract == BinaryContract::CsrDenseMatmul) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  if (contract == BinaryContract::CsrCsrMatmul) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_csr_view(b_shape, b_mode_indices, b_values, dtype, op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  if (contract == BinaryContract::CooDenseMatmul) {
    const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                    op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                  op, "A");
  const auto b = checked_coo_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                  op, "B");
  TORCH_CHECK(a.shape[1] == b.shape[0], op,
              ": contraction mismatch: A.shape[1]=", a.shape[1],
              " but B.shape[0]=", b.shape[0]);
  TORCH_CHECK(result_shape[0] == a.shape[0] && result_shape[1] == b.shape[1],
              op, ": result_shape must equal [A.rows, B.cols]");
}

inline int validate_tile_size(int value, const char* op, const char* argument,
                              int64_t extent = -1) {
  TORCH_CHECK(value > 0, op, ": ", argument, " must be positive, got ", value);
  TORCH_CHECK(value <= std::numeric_limits<int>::max() - 15, op, ": ", argument,
              " is too large: ", value);
  if (extent >= 0) {
    return static_cast<int>(
        std::min<int64_t>(value, std::max<int64_t>(extent, 1)));
  }
  return value;
}

inline void validate_thread_override(int value, const char* op) {
  TORCH_CHECK(value >= -1, op,
              ": nthreads_override must be -1, 0, or positive, got ", value);
}

inline void validate_finite_scale(double value, const char* op) {
  TORCH_CHECK(std::isfinite(value), op, ": scale must be finite, got ", value);
  TORCH_CHECK(value >= -std::numeric_limits<float>::max() &&
                  value <= std::numeric_limits<float>::max(),
              op, ": scale cannot be represented as float32: ", value);
}

template <typename Dim>
inline void validate_sddmm_inputs(
    const char* op, const std::vector<Dim>& result_shape,
    const std::vector<Dim>& s_shape,
    std::vector<std::vector<torch::Tensor>>& s_mode_indices,
    const torch::Tensor& s_values, const std::vector<Dim>& a_shape,
    std::vector<std::vector<torch::Tensor>>& a_mode_indices,
    const torch::Tensor& a_values, const std::vector<Dim>& b_shape,
    std::vector<std::vector<torch::Tensor>>& b_mode_indices,
    const torch::Tensor& b_values) {
  const auto s = checked_coo_view(s_shape, s_mode_indices, s_values,
                                  torch::kFloat32, 2, op, "S");
  const auto a = checked_dense_view(a_shape, a_mode_indices, a_values,
                                    torch::kFloat32, 2, op, "A");
  const auto b = checked_dense_view(b_shape, b_mode_indices, b_values,
                                    torch::kFloat32, 2, op, "B");
  check_legacy_output_shape(result_shape, 2, op);
  TORCH_CHECK(result_shape[0] == s.shape[0] && result_shape[1] == s.shape[1],
              op, ": result_shape must equal S.shape");
  TORCH_CHECK(a.shape[0] == s.shape[0], op,
              ": A rows must equal S rows");
  TORCH_CHECK(b.shape[0] == s.shape[1], op,
              ": B rows must equal S columns");
  TORCH_CHECK(a.shape[1] == b.shape[1], op,
              ": A and B reduction dimensions must match");
}

inline void validate_csr_segments(
    const char* op, torch::Tensor& positions, const torch::Tensor& values) {
  check_tensor_common(values, torch::kFloat32, op, "values", 1);
  positions = checked_index_tensor(positions, op, "crow_indices");
  TORCH_CHECK(positions.numel() >= 1, op,
              ": crow_indices must contain at least the initial zero");
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), op,
              ": values nnz exceeds the current int32 native bound");
  const int32_t* pos = positions.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, op, ": crow_indices[0] must be 0, got ", pos[0]);
  for (int64_t i = 1; i < positions.numel(); ++i) {
    TORCH_CHECK(pos[i] >= pos[i - 1] && pos[i] <= nnz, op,
                ": invalid crow_indices entry ", pos[i], " at position ", i,
                " for nnz ", nnz);
  }
  TORCH_CHECK(pos[positions.numel() - 1] == nnz, op,
              ": terminal crow index must equal values nnz (", nnz, "), got ",
              pos[positions.numel() - 1]);
}

inline void validate_attention_inputs(
    const char* op, torch::Tensor& positions, torch::Tensor& coordinates,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  check_tensor_common(q, torch::kFloat32, op, "Q", 3);
  check_tensor_common(k, torch::kFloat32, op, "K", 3);
  check_tensor_common(v, torch::kFloat32, op, "V", 3);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), op,
              ": Q, K, and V must have identical [S, H, D] shapes");
  const int64_t sequence = q.size(0);
  const std::vector<int64_t> output_shape = {q.size(0), q.size(1), q.size(2)};
  checked_product(output_shape, op, "Q/K/V shape");

  const auto raw_positions = positions;
  const auto raw_coordinates = coordinates;
  positions = checked_index_tensor(raw_positions, op, "crow_indices");
  coordinates = checked_index_tensor(raw_coordinates, op, "col_indices");
  TORCH_CHECK(raw_positions.scalar_type() == raw_coordinates.scalar_type(), op,
              ": crow_indices and col_indices must use one common index dtype");
  TORCH_CHECK(positions.numel() == sequence + 1, op,
              ": crow_indices length must be S + 1 (", sequence + 1,
              "), got ", positions.numel());
  TORCH_CHECK(coordinates.numel() <= std::numeric_limits<int>::max(), op,
              ": mask nnz exceeds the current int32 native bound");
  const int32_t* pos = positions.data_ptr<int32_t>();
  const int32_t* crd = coordinates.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, op, ": crow_indices[0] must be 0, got ", pos[0]);
  for (int64_t row = 0; row < sequence; ++row) {
    TORCH_CHECK(pos[row] >= 0 && pos[row + 1] >= pos[row] &&
                    pos[row + 1] <= coordinates.numel(),
                op, ": invalid CSR mask span for row ", row);
  }
  TORCH_CHECK(pos[sequence] == coordinates.numel(), op,
              ": terminal crow index must equal col_indices nnz (",
              coordinates.numel(), "), got ", pos[sequence]);
  for (int64_t p = 0; p < coordinates.numel(); ++p) {
    TORCH_CHECK(crd[p] >= 0 && crd[p] < sequence, op, ": column coordinate ",
                crd[p], " at position ", p, " is outside [0, ", sequence,
                ")");
  }
}

// Generic entry validation emitted at the beginning of every JIT evaluate().
// Shapes and layouts are compile-time kernel contracts encoded by CINLowerer.
inline void validate_jit_tensor(
    const char* op, const char* argument, const std::vector<int64_t>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    const std::vector<int>& level_kinds, const std::vector<int>& mode_order,
    const std::vector<int64_t>& expected_shape) {
  const auto logical_shape = expected_shape.empty() && !level_kinds.empty()
                                 ? checked_shape(
                                       shape,
                                       static_cast<int64_t>(level_kinds.size()),
                                       op, argument)
                                 : checked_expected_shape(
                                       shape, expected_shape, op, argument,
                                       true);
  checked_product(logical_shape, op, argument, true);
  TORCH_CHECK(level_kinds.size() == logical_shape.size(),
              argument_name(op, argument), " level contract rank mismatch");
  TORCH_CHECK(mode_order.size() == logical_shape.size(),
              argument_name(op, argument), " mode-order rank mismatch");
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " mode_indices must contain ",
              logical_shape.size(), " levels, got ", mode_indices.size());
  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);

  std::vector<bool> seen_mode(logical_shape.size(), false);
  for (size_t level = 0; level < mode_order.size(); ++level) {
    TORCH_CHECK(mode_order[level] >= 0 &&
                    static_cast<size_t>(mode_order[level]) < logical_shape.size() &&
                    !seen_mode[mode_order[level]],
                argument_name(op, argument), " invalid compiled mode-order entry ",
                mode_order[level], " at level ", level);
    seen_mode[mode_order[level]] = true;
  }

  int64_t storage_count = 1;
  int64_t coordinate_count = -1;
  bool has_index_dtype = false;
  torch::ScalarType index_dtype = torch::kInt32;
  std::vector<torch::Tensor> coordinate_levels;
  const bool all_coordinate =
      !level_kinds.empty() &&
      std::all_of(level_kinds.begin(), level_kinds.end(), [](int kind) {
        return static_cast<LevelKind>(kind) == LevelKind::Coordinate;
      });
  for (size_t level = 0; level < level_kinds.size(); ++level) {
    // Runtime *_shape arguments are already in physical level order (the same
    // convention used by generated ``A_shape[level]`` bound expressions).
    // mode_order is still checked as metadata, but applying it a second time here
    // would reject valid rectangular tensors after a non-identity permutation.
    const int64_t extent = logical_shape[level];
    const auto kind = static_cast<LevelKind>(level_kinds[level]);
    if (kind == LevelKind::Dense) {
      TORCH_CHECK(mode_indices[level].empty(), argument_name(op, argument),
                  " dense level ", level, " must not contain index tensors");
      TORCH_CHECK(extent == 0 ||
                      storage_count <= std::numeric_limits<int64_t>::max() / extent,
                  argument_name(op, argument), " storage size overflows at level ",
                  level);
      storage_count *= extent;
      continue;
    }

    if (kind == LevelKind::Coordinate) {
      TORCH_CHECK(mode_indices[level].size() == 1,
                  argument_name(op, argument), " coordinate level ", level,
                  " must contain exactly one tensor");
      const auto raw_coordinate = mode_indices[level][0];
      mode_indices[level][0] = checked_index_tensor(
          raw_coordinate, op,
          std::string(argument) + " coordinate level " + std::to_string(level));
      check_common_index_dtype(raw_coordinate, has_index_dtype, index_dtype, op,
                               std::string(argument) + " coordinate level " +
                                   std::to_string(level));
      const int64_t count = mode_indices[level][0].numel();
      if (coordinate_count < 0) {
        coordinate_count = count;
        storage_count = count;
      } else {
        TORCH_CHECK(count == coordinate_count, argument_name(op, argument),
                    " coordinate levels have inconsistent nnz");
      }
      const int32_t* crd = mode_indices[level][0].data_ptr<int32_t>();
      for (int64_t p = 0; p < count; ++p) {
        TORCH_CHECK(crd[p] >= 0 && crd[p] < extent,
                    argument_name(op, argument), " coordinate ", crd[p],
                    " at level ", level, " position ", p,
                    " is outside [0, ", extent, ")");
      }
      coordinate_levels.push_back(mode_indices[level][0]);
      continue;
    }

    TORCH_CHECK(kind == LevelKind::Compressed,
                argument_name(op, argument), " unsupported level kind at ", level);
    TORCH_CHECK(mode_indices[level].size() == 2,
                argument_name(op, argument), " compressed level ", level,
                " must contain [positions, coordinates]");
    const auto raw_positions = mode_indices[level][0];
    const auto raw_coordinates = mode_indices[level][1];
    mode_indices[level][0] = checked_index_tensor(
        raw_positions, op,
        std::string(argument) + " positions level " + std::to_string(level));
    mode_indices[level][1] = checked_index_tensor(
        raw_coordinates, op,
        std::string(argument) + " coordinates level " + std::to_string(level));
    check_common_index_dtype(raw_positions, has_index_dtype, index_dtype, op,
                             std::string(argument) + " positions level " +
                                 std::to_string(level));
    check_common_index_dtype(raw_coordinates, has_index_dtype, index_dtype, op,
                             std::string(argument) + " coordinates level " +
                                 std::to_string(level));
    const auto& positions = mode_indices[level][0];
    const auto& coordinates = mode_indices[level][1];
    TORCH_CHECK(positions.numel() == storage_count + 1,
                argument_name(op, argument), " compressed level ", level,
                " positions length must equal parent count + 1 (",
                storage_count + 1, "), got ", positions.numel());
    const int32_t* pos = positions.data_ptr<int32_t>();
    const int32_t* crd = coordinates.data_ptr<int32_t>();
    TORCH_CHECK(pos[0] == 0, argument_name(op, argument),
                " compressed level ", level, " must start at position 0");
    for (int64_t parent = 0; parent < storage_count; ++parent) {
      TORCH_CHECK(pos[parent] >= 0 && pos[parent + 1] >= pos[parent] &&
                      pos[parent + 1] <= coordinates.numel(),
                  argument_name(op, argument), " invalid compressed span at level ",
                  level, " parent ", parent);
      for (int32_t p = pos[parent]; p < pos[parent + 1]; ++p) {
        TORCH_CHECK(crd[p] >= 0 && crd[p] < extent,
                    argument_name(op, argument), " coordinate ", crd[p],
                    " at compressed level ", level, " position ", p,
                    " is outside [0, ", extent, ")");
        if (p > pos[parent]) {
          TORCH_CHECK(
              crd[p - 1] < crd[p], argument_name(op, argument),
              " coordinates must be strictly increasing at compressed level ",
              level, " position ", p);
        }
      }
    }
    TORCH_CHECK(pos[storage_count] == coordinates.numel(),
                argument_name(op, argument), " compressed level ", level,
                " terminal position must equal coordinate count");
    storage_count = coordinates.numel();
  }
  if (all_coordinate && storage_count > 1) {
    for (int64_t p = 1; p < storage_count; ++p) {
      bool greater = false;
      bool different = false;
      for (const auto& coordinate : coordinate_levels) {
        const int32_t* crd = coordinate.data_ptr<int32_t>();
        if (crd[p] != crd[p - 1]) {
          greater = crd[p] > crd[p - 1];
          different = true;
          break;
        }
      }
      TORCH_CHECK(!different || greater, argument_name(op, argument),
                  " COO coordinates must be lexicographically ordered; order ",
                  "decreases at position ", p);
    }
  }
  TORCH_CHECK(values.numel() == storage_count, argument_name(op, argument),
              " values length must equal validated physical storage size ",
              storage_count, ", got ", values.numel());
}

inline void validate_jit_result_shape(
    const std::vector<int64_t>& result_shape,
    const std::vector<int64_t>& expected_shape, int64_t expected_rank,
    const char* op) {
  if (expected_shape.empty() && expected_rank > 0) {
    const auto shape =
        checked_shape(result_shape, expected_rank, op, "result_shape");
    checked_product(shape, op, "result_shape", true);
  } else {
    checked_expected_shape(result_shape, expected_shape, op, "result_shape",
                           true);
  }
}

inline void validate_jit_extra_tensor(const torch::Tensor& tensor,
                                      torch::ScalarType expected_dtype,
                                      const char* op, const char* argument) {
  check_tensor_common(tensor, expected_dtype, op, argument);
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              argument_name(op, argument),
              " element count exceeds the current native int32 bound");
}

}  // namespace scorch_native
