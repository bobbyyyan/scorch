#pragma once

#include <cstdint>
#include <type_traits>

#include <torch/extension.h>

template <typename scalar_t>
inline torch::ScalarType scorch_torch_dtype() {
  static_assert(
      !std::is_same<scalar_t, scalar_t>::value,
      "Unsupported scalar type for prebuilt kernel dtype mapping");
  return torch::kFloat32;
}

template <>
inline torch::ScalarType scorch_torch_dtype<float>() {
  return torch::kFloat32;
}

template <>
inline torch::ScalarType scorch_torch_dtype<double>() {
  return torch::kFloat64;
}

template <>
inline torch::ScalarType scorch_torch_dtype<int32_t>() {
  return torch::kInt32;
}

template <>
inline torch::ScalarType scorch_torch_dtype<int64_t>() {
  return torch::kInt64;
}

template <typename scalar_t>
inline const char* scorch_dtype_suffix() {
  static_assert(
      !std::is_same<scalar_t, scalar_t>::value,
      "Unsupported scalar type for prebuilt kernel suffix mapping");
  return "unknown";
}

template <>
inline const char* scorch_dtype_suffix<float>() {
  return "f32";
}

template <>
inline const char* scorch_dtype_suffix<double>() {
  return "f64";
}

template <>
inline const char* scorch_dtype_suffix<int32_t>() {
  return "i32";
}

template <>
inline const char* scorch_dtype_suffix<int64_t>() {
  return "i64";
}
