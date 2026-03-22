# 🔥 Scorch

A compiler-based sparse tensor library for PyTorch.

[![CI](https://img.shields.io/github/actions/workflow/status/bobbyyyan/scorch/pytest.yml?branch=main&style=flat)](https://github.com/bobbyyyan/scorch/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/CGO%202026-paper-blueviolet?style=flat)](https://fredrikbk.com/cgo26scorch.html)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-blue?style=flat)](https://github.com/bobbyyyan/scorch)

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
