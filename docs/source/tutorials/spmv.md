# SpMV: sparse matrix × dense vector

Sparse matrix–vector product — `y = A @ x` — is the workhorse of sparse linear
algebra: iterative solvers, graph propagation, and PageRank-style power iteration
all spend most of their time here. This page shows how to compute it in Scorch,
verify it against a dense PyTorch reference, and points you to the full
SuiteSparse benchmark.

## The math

Given a sparse matrix $A \in \mathbb{R}^{M \times N}$ and a dense vector
$x \in \mathbb{R}^{N}$, SpMV produces a dense vector $y \in \mathbb{R}^{M}$:

$$ y_i = \sum_{j} A_{ij}\, x_j $$

Because $A$ is sparse, the inner sum only touches the stored (nonzero) entries of
row $i$ — that is the whole point. If $A$ has `nnz` nonzeros, SpMV costs
$O(\text{nnz})$ work rather than the $O(MN)$ of a dense matrix–vector product.

## How to call it

Reach SpMV through {func}`~scorch.matmul` with a **2-D sparse left operand** and a
**1-D dense right operand**. Scorch inspects the operand ranks and dispatches the
sparse matrix–vector kernel automatically.

:::{note}
Scorch does have an internal `spmv` function ({func}`~scorch.ops.spmv`), but it is
**not** part of the public API — it is not exported from the `scorch` namespace.
Always go through {func}`~scorch.matmul`; giving it a 1-D right operand is what
selects the SpMV path. This mirrors PyTorch, where `torch.matmul` also folds the
matrix–vector case into one entry point.
:::

## A runnable example

Build a small torch CSR matrix and a dense vector, run
`scorch.matmul(A, x)`, and check the result against a dense `torch.matmul`:

```python
import torch
import scorch

torch.manual_seed(0)

# A small sparse matrix [4 x 5], stored as torch CSR.
dense_A = torch.tensor([
    [0., 2., 0., 0., 1.],
    [0., 0., 0., 3., 0.],
    [4., 0., 0., 0., 0.],
    [0., 0., 5., 0., 6.],
])
A = dense_A.to_sparse_csr()            # torch sparse CSR
x = torch.rand(5, dtype=torch.float32) # dense length-N vector

y_scorch = scorch.matmul(A, x)         # SpMV: sparse A @ dense x
y_ref = torch.matmul(dense_A, x)       # dense reference

# scorch.matmul may return an STensor; bring it back to a torch tensor.
if hasattr(y_scorch, "to_torch"):
    y_scorch = y_scorch.to_dense().to_torch()

assert torch.allclose(y_scorch, y_ref, atol=1e-3, rtol=1e-3)
print("SpMV OK:", y_scorch)
```

The `assert torch.allclose(..., atol=1e-3, rtol=1e-3)` follows Scorch's
correctness convention: every operation is validated against a PyTorch reference at
`atol = rtol = 1e-3`. Use the same tolerance when you write your own checks.

:::{tip}
The `if hasattr(y_scorch, "to_torch")` guard keeps the snippet robust: depending on
the input formats, {func}`~scorch.matmul` may hand back either a plain
`torch.Tensor` or an {class}`~scorch.STensor`. Calling `.to_dense().to_torch()` on
an STensor materializes it as a dense torch tensor you can compare and print.
:::

## Input formats

The example above starts from a torch CSR tensor (`dense_A.to_sparse_csr()`), which
Scorch accepts directly. You can equally build the operand as an
{class}`~scorch.STensor` first — for instance with {func}`~scorch.from_torch` and a
format string (`"ds"` is CSR: dense rows, compressed columns), or from COO with
{func}`~scorch.from_coo`. See {doc}`/user_guide/format_system` for the full format
notation and {doc}`/user_guide/sparse_tensors` for the STensor constructors.

## The full benchmark

The distilled snippet above is the teaching version. The complete example,
`examples/kernels/spmv.py`, is a benchmark harness: it iterates over the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) (downloaded via `ssgetpy`
into `~/.ssgetpy`), loads each matrix as CSR, times `torch.matmul(A, x)` against
`scorch.matmul(A, x)`, and plots runtime versus nonzero count.

To run it you will need the extra benchmark dependencies (`ssgetpy`, `scipy`,
`pandas`, `matplotlib`, `seaborn`, `tqdm`) and a local copy of the SuiteSparse
matrices — the example README ships a `download_all_ss.py` helper for that. Both
the harness and the snippet share one converter for turning a SciPy matrix into a
torch sparse tensor:

```python
import numpy as np
import torch

def scipy_sparse_to_torch_sparse(matrix, format='csr'):
    if format == 'coo':
        matrix = matrix.tocoo()
        indices = np.vstack((matrix.row, matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(matrix.shape))
    elif format == 'csr':
        matrix = matrix.tocsr()
        crow_indices = torch.LongTensor(matrix.indptr)
        col_indices = torch.LongTensor(matrix.indices)
        values = torch.FloatTensor(matrix.data)
        return torch.sparse_csr_tensor(crow_indices, col_indices, values,
                                       torch.Size(matrix.shape))
    raise ValueError("Unsupported format: only 'coo' and 'csr' are supported")
```

Across the SuiteSparse collection, Scorch's compiled kernels deliver
**1.05–5.80× over PyTorch Sparse (CGO 2026)**.

## Next steps

- {doc}`/tutorials/spmm` — the same pattern with a dense **matrix** on the right
  (`y = A @ B`), Scorch's flagship kernel.
- {doc}`/user_guide/operations` — the full menu of sparse operations and the
  `format=` / `output_format=` kwargs that control output layout.
- {doc}`/tutorials/sddmm` and {doc}`/tutorials/spgemm` — sampled dense-dense
  products and sparse × sparse multiplication.
