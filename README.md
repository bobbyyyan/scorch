# Scorch

Scorch is a Python library for sparse machine learning, built on top of PyTorch. It provides sparse implementations of key PyTorch operations, allowing you to work with sparse tensors seamlessly.

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
