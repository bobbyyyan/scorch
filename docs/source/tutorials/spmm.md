# SpMM: sparse matrix × dense matrix

**SpMM** — sparse matrix times dense matrix — is the workhorse of sparse deep
learning. Graph convolutions, sparse linear layers, and sparse attention all
bottom out in a sparse `A` multiplied against a dense `B`. It is also Scorch's
**flagship kernel**: most of the compiler's performance work — tiling,
register-blocking, autotune levels, and the fused `Linear` path — targets exactly
this shape.

This page shows how to run SpMM with {func}`~scorch.matmul`, how to verify it
against PyTorch, and where to go next for the tuned and fused variants.

## What it computes

Given a sparse matrix $A \in \mathbb{R}^{M \times N}$ and a dense matrix
$B \in \mathbb{R}^{N \times K}$, SpMM computes the dense product

$$C = A B, \qquad C_{ik} = \sum_{j} A_{ij}\, B_{jk}.$$

Only the nonzeros of $A$ contribute to the sum, so the cost scales with
$\mathrm{nnz}(A) \cdot K$ rather than $M \cdot N \cdot K$ — the reason a sparse
kernel wins when $A$ is sparse enough.

## A first example

Build a small CSR matrix `A`, a dense `B`, and call `scorch.matmul`. We verify
the result against `torch.sparse.mm`, the PyTorch reference for SpMM.

```python
import torch
import scorch

torch.manual_seed(0)

dense_A = torch.tensor([
    [0., 2., 0., 0., 1.],
    [0., 0., 0., 3., 0.],
    [4., 0., 0., 0., 0.],
    [0., 0., 5., 0., 6.],
])
A = dense_A.to_sparse_csr()                # torch sparse CSR  [4 x 5]
B = torch.rand(5, 8, dtype=torch.float32)  # dense            [5 x 8]  (K = 8)

C_scorch = scorch.matmul(A, B)             # SpMM: sparse A @ dense B
C_ref = torch.sparse.mm(A, B)              # PyTorch reference

# scorch may return an STensor; bring it back to a torch tensor if so
if hasattr(C_scorch, "to_torch"):
    C_scorch = C_scorch.to_dense().to_torch()

assert torch.allclose(C_scorch, C_ref, atol=1e-3, rtol=1e-3)
print("SpMM OK, shape:", C_scorch.shape)   # -> [4, 8]
```

The `atol=rtol=1e-3` tolerance is the project's correctness convention — every
Scorch operation is validated against a PyTorch reference at that tolerance rather
than against hardcoded values.

:::{note}
`scorch.matmul` accepts a torch sparse tensor directly — you do not have to wrap
it in an {class}`~scorch.STensor` first. Internally it detects the sparse operand
and dispatches SpMM. To hand it a Scorch-native tensor instead, build one with
{func}`~scorch.from_csr`, {func}`~scorch.from_coo`, or
`` {func}`~scorch.from_torch`(dense).to_sparse("ds") `` (`"ds"` is Scorch's format
string for CSR).
:::

## Choosing the output layout

`matmul` takes an optional `format=` kwarg that selects the output layout as a
Scorch format string (`d` = dense, `s`/`c` = compressed, `o` = coordinate). For
SpMM the result is dense, so `format="dd"` forces a plain dense output — this is
the idiom the GCN example uses on every graph-conv layer:

```python
# sparse adjacency @ dense features, dense result
out = scorch.matmul(adjacency, x, format="dd")
```

Passing `format="dd"` returns a dense tensor directly, saving the
`.to_dense().to_torch()` round-trip. Without it, `matmul` chooses an output format
from the operands and may return an {class}`~scorch.STensor`.

:::{tip}
When you already know the product is dense — which is always true for SpMM —
prefer `format="dd"`. It documents intent and lets the compiler skip building a
sparse output structure it would only densify anyway.
:::

## The full benchmark

The distilled snippet above is the smallest correct program that exercises the
kernel. The shipped example, `examples/kernels/spmm.py`, is a benchmark harness:
it iterates the SuiteSparse Matrix Collection, loads each matrix as CSR, multiplies
it by a dense `[N, 100]` matrix in both PyTorch and Scorch, and plots runtime
against nonzero count. Its two timed lines are exactly:

```python
dense_matrix = torch.rand((torch_sparse_matrix.shape[1], 100), dtype=torch.float32)

result = torch.sparse.mm(torch_sparse_matrix, dense_matrix)  # baseline
result = scorch.matmul(torch_sparse_matrix, dense_matrix)    # scorch
```

Across that collection Scorch runs SpMM at **1.05–5.80× over PyTorch Sparse
(CGO 2026)**. Running the harness yourself requires the SuiteSparse matrices on
disk plus `ssgetpy`, `scipy`, `pandas`, `matplotlib`, and `seaborn`; see the
example's README for the download step.

## Making it faster

SpMM is where Scorch invests the most, and there are two levers you can pull
without changing the call site:

- **Autotuning.** The autotune level chooses the schedule — tiling strategy,
  register-blocking, thread policy — for each SpMM shape. See
  {doc}`/user_guide/autotuning` for the `off` / `analytic` / `balanced` / `max` /
  `learned` ladder and the {func}`~scorch.set_autotune` API.
- **Fused neural-network kernels.** When SpMM is followed by a bias add and an
  activation — the sparse `Linear` pattern — Scorch ships
  {func}`~scorch.sparse_linear` (and the feature-major
  {func}`~scorch.sparse_linear_fm` plus {func}`~scorch.fast_transpose`) that fuse
  the whole epilogue into one pass. See {doc}`/user_guide/neural_network_ops`.

## See also

- {doc}`/tutorials/spmv` — the vector case, `A @ x`, reached through the same
  `matmul` entry point.
- {doc}`/tutorials/spgemm` — SpGEMM (SpMSpM), when **both** operands are sparse.
- {doc}`/user_guide/autotuning` — pick a schedule for your SpMM shapes.
- {doc}`/user_guide/neural_network_ops` — the fused `sparse_linear` successor to
  a bare SpMM.
- {doc}`/tutorials/gcn` — SpMM in a real model, using `format="dd"` per layer.
