cmake_minimum_required(VERSION 3.23)
project(scorch)

set(CMAKE_CXX_STANDARD 14)

#include_directories(venv/lib/python3.9/site-packages/torch/include)
#include_directories(venv/lib/python3.9/site-packages/torch/include/ATen)

set(TORCH_INCLUDE_DIR venv/lib/python3.9/site-packages/torch/include)
file(GLOB_RECURSE TORCH_INCLUDE_SUBDIRS ${TORCH_INCLUDE_DIR}/*)
include_directories(${TORCH_INCLUDE_DIR})

include_directories(venv/lib/python3.9/site-packages/torch)
include_directories(venv/lib/python3.9/site-packages/torch/csrc/api/include)

include_directories(venv/lib/python3.9/site-packages/torch/include/pybind11)
include_directories(venv/lib/python3.9/site-packages/torch/include/pybind11/detail)
include_directories(venv/lib/python3.9/site-packages/torch/include/torch)
include_directories(venv/lib/python3.9/site-packages/torch/include/torch/csrc)
include_directories(venv/lib/python3.9/site-packages/torch/include/torch/csrc/api)
include_directories(venv/lib/python3.9/site-packages/torch/include/torch/csrc/api/include)


add_executable(scorch csrc/ops.cpp csrc/elem_wise_vector_mul.cpp csrc/elem_wise_vector_mul.h)
