# 🔥 Scorch

A compiler-based sparse tensor library for PyTorch.

[![CI](https://img.shields.io/github/actions/workflow/status/bobbyyyan/scorch/pytest.yml?branch=main&style=flat)](https://github.com/bobbyyyan/scorch/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/CGO%202026-paper-blueviolet?style=flat)](https://ieeexplore.ieee.org/abstract/document/11394842)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-blue?style=flat)](https://github.com/bobbyyyan/scorch)

## What is Scorch?

Traditional sparse tensor libraries require hand-tuned kernels for every combination of sparse format and operation. Adding a new storage format means writing new kernels from scratch, and the number of format-operation combinations grows quickly. Scorch takes a different approach: instead of hand-writing kernels, you describe tensor formats declaratively and Scorch generates optimized C++ kernels automatically. Expressions are lowered through Compiler Index Notation (CIN) to a Low-Level IR (LLIR), then to C++ source code that is JIT-compiled and cached as a shared library.

Scorch's central abstraction is a format notation where each tensor dimension is described as dense (d), compressed-sparse (cs), coordinate (o), or singleton (s). Familiar formats map directly to this notation: CSR is (d, cs), meaning dense rows with compressed-sparse columns, and COO is (o, o), coordinate storage on both dimensions. Because the compiler generates kernels from these format descriptions, supporting a new sparse format does not require writing new kernel code -- you simply declare the format and the compiler handles the rest.

Scorch integrates directly with PyTorch. STensors wrap standard PyTorch tensors, and operations like matmul and einsum accept torch.Tensor inputs via from_torch. The first call to an operation compiles a specialized C++ kernel with OpenMP parallelization; subsequent calls with the same format combination reuse the cached shared library from disk, so compilation cost is paid only once. In benchmarks from the CGO 2026 paper, Scorch achieved 1.05--5.80x speedups over PyTorch Sparse across sparse matrix and graph neural network workloads.

## Quick Start

Create sparse tensors and run operations with a familiar API:

```python
import torch
import scorch

# Create STensors from PyTorch tensors
A = scorch.from_torch(torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A")
B = scorch.from_torch(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), "B")

# Matrix multiply
C = scorch.matmul(A, B)
print(C)

# Einstein summation
D = scorch.einsum("ij,jk->ik", A, B)
print(D.to_torch())
```

```
tensor([[11., 14.],
        [ 9., 12.],
        [29., 38.]])
tensor([[11., 14.],
        [ 9., 12.],
        [29., 38.]])
```

> **Note:** The first call compiles a specialized C++ kernel via JIT. Subsequent calls reuse the cached kernel and run at full speed.

> Scorch uses a [format notation](#format-system) where each dimension is dense (`d`), compressed (`s`), or coordinate (`o`) -- so CSR is simply `(d,s)`.
