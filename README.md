# Scorch

[![pytest](https://github.com/bobbyyyan/scorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/bobbyyyan/scorch/actions/workflows/pytest.yml)

## Getting started

```
cd scorch
pipenv install --dev
pipenv shell
```

### Setting up development environment

#### Install Ninja

Ninja is required to load C++ extension. To install, use `pip install ninja`.

#### Install libtorch

libtorch is needed to test generated C++ code. See https://pytorch.org/get-started/locally/ for more details. For M1 Mac, see instructions on how to build the latest version of libtorch below.

#### Building libtorch for M1 Mac

```shell
git clone -b main --recurse-submodule https://github.com/pytorch/pytorch.git
cd pytorch
mkdir build && cd build
cmake -D BUILD_SHARED_LIBS:BOOL=ON \
      -D CMAKE_BUILD_TYPE:STRING=Release \
      -D PYTHON_EXECUTABLE:PATH=`which python3` \
      -D CMAKE_INSTALL_PREFIX:PATH=../libtorch  \
      ..
cmake --build . --target install --parallel 20
```

#### Edit includepath

You may want to edit the include path in your IDE to include the libtorch headers.

For VSCode, edit include path to include the following so that intellisense works.

```
${workspaceFolder}/**
~/libtorch/include/**
~/libtorch/include/torch/**
~/libtorch/include/torch/csrc/api/include/**
```
