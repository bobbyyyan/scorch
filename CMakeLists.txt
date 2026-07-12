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


# Legacy IDE-indexing target only. The supported native-extension build is declared
# in pyproject.toml and customized by scorch_build.py.
add_custom_target(
        scorch_sources
        SOURCES
        src/scorch/csrc/header.cpp
        src/scorch/csrc/header.h
        src/scorch/csrc/kernels.h
        src/scorch/csrc/native_abi.h
        src/scorch/csrc/ops.cpp
        src/scorch/csrc/prebuilt_types.h
        src/scorch/csrc/pybind.cpp
        src/scorch/csrc/scorch_policy.h
        src/scorch/csrc/spmm.h)
