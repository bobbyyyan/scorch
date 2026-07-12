# SpGEMM: sparse × sparse

**SpGEMM** — sparse general matrix–matrix multiply, also called **SpMSpM** — is
the product of two sparse matrices, `C = A @ B`, where *both* operands are sparse
and the result is sparse too. It is the workhorse behind graph analytics
(triangle counting, multi-hop reachability, graph coarsening) and any pipeline
that composes sparse operators. This page walks through the canonical case
`C = A @ Aᵀ` with {func}`~scorch.matmul`, verifies it against a dense reference,
and explains how the output format is chosen.

## The computation

Given a sparse matrix $A \in \mathbb{R}^{M \times N}$, the Gram-style product

$$ C = A A^{\top}, \qquad C_{ik} = \sum_{j} A_{ij}\, A_{kj} $$

is itself sparse: $C_{ik}$ is nonzero only when rows $i$ and $k$ of $A$ share a
nonzero column. Because both inputs *and* the output are sparse, Scorch keeps the
whole pipeline sparse — it never densifies $A$, and it emits $C$ in a compressed
format rather than materializing an $M \times M$ dense buffer.

Unlike SpMM (sparse × **dense**, which produces a dense result), the result of an
SpGEMM has a data-dependent sparsity pattern that is only known after the
multiply. Scorch computes that pattern and the values together.

## A runnable example

The example below mirrors `examples/kernels/spmspm.py`: build a small sparse `A`
in COO, form its transpose, multiply sparse × sparse, and check the result
against a dense `torch.matmul` reference.

```python
import torch
import scorch

torch.manual_seed(0)

# Sparse A [3 x 3] in COO
idx = torch.tensor([[0, 0, 1, 2, 2],
                    [0, 2, 1, 0, 2]])
val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
A = torch.sparse_coo_tensor(idx, val, (3, 3)).coalesce()
A_T = A.transpose(0, 1)                       # torch sparse transpose

C_scorch = scorch.matmul(A, A_T)              # SpGEMM: sparse x sparse

# Dense reference
C_ref = torch.matmul(A.to_dense(), A_T.to_dense())

C_dense = C_scorch.to_dense().to_torch()
assert torch.allclose(C_dense, C_ref, atol=1e-3, rtol=1e-3)
print("SpGEMM OK, output format:", C_scorch.format)
```

The result comes back as an {class}`~scorch.STensor`. Use `.to_dense().to_torch()`
to recover a plain dense `torch.Tensor` for comparison, or keep it sparse and feed
it into the next sparse operation.

:::{note}
Scorch follows the project's correctness convention: verify sparse results against
a PyTorch reference with `torch.allclose(..., atol=1e-3, rtol=1e-3)` rather than
hardcoding expected values.
:::

## COO inputs are promoted to CSR

The example passes torch **COO** tensors straight into `scorch.matmul`. Internally
Scorch does not run a coordinate × coordinate kernel — it promotes both operands to
**CSR** (`"ds"`: dense rows, compressed columns) before dispatching the SpGEMM.
This is why passing a coalesced COO matrix works transparently: the promotion is
handled for you inside `ops.py`.

If you already hold a CSR tensor (for example from `torch.Tensor.to_sparse_csr()`
or `from_csr`), you can pass it directly and skip the conversion.

## Choosing the output format

By default the SpGEMM returns a sparse {class}`~scorch.STensor`. You can request a
specific output layout with the `format=` keyword. Requesting `format="ds"` yields
a **CSR** STensor — the standard row-compressed layout, ideal when the result feeds
another row-major sparse consumer:

```python
C_csr = scorch.matmul(A, A_T, format="ds")    # CSR output
print(C_csr.format)                            # -> "d,s"

C_csr_dense = C_csr.to_dense().to_torch()
assert torch.allclose(C_csr_dense, C_ref, atol=1e-3, rtol=1e-3)
```

Recall the format-string shorthand: `"ds"` = CSR, `"ss"` = DCSR (doubly
compressed), `"oo"` = COO, `"dd"` = dense. For SpGEMM you almost always want a
compressed result (`"ds"` or `"ss"`) — passing `"dd"` would force the dense
$M \times M$ buffer that keeping the product sparse is meant to avoid. See
{doc}`the format system </user_guide/format_system>` for the full level-type
reference.

:::{tip}
`C_scorch.format` reports the level types per mode (e.g. `"d,s"` for CSR). If you
plan to chain another sparse op onto the result, pin the format explicitly so the
downstream kernel sees the layout it expects.
:::

## The full benchmark

The distilled snippet above is the smallest correct program that exercises the
API. The complete benchmark harness lives in `examples/kernels/spmspm.py`: it
iterates the SuiteSparse Matrix Collection, times `scorch.matmul(A, A_T)` against
`torch.matmul(A, A_T)`, and plots runtime versus nonzero count.

A couple of details from that script are worth knowing if you adapt it:

- **Square truncation.** Each matrix is truncated to `min_dim × min_dim` before
  the product so that `A @ Aᵀ` is well-formed on the collection's rectangular
  matrices.
- **Size filters.** It keeps matrices with `max(rows, cols) < 100000` and
  `nnz < 10_000_000` — SpGEMM output can blow up superlinearly, so the harness
  caps input size to keep the benchmark tractable.

Across the SuiteSparse suite, Scorch delivers **1.05–5.80× over PyTorch Sparse
(CGO 2026)** on the sparse-kernel benchmarks.

## See also

- {doc}`Benchmarks </performance/benchmarks>` — SpGEMM shows up throughout the
  graph-analytics results (triangle counting, multi-hop link prediction), which
  are SpMSpM under the hood.
- {doc}`SpMM: sparse × dense </tutorials/spmm>` — the sibling kernel with a *dense*
  right operand and a dense result.
- {doc}`SDDMM: sampled dense-dense product </tutorials/sddmm>` — the other
  sparsity-pattern-restricted contraction, expressed with
  {func}`~scorch.einsum`.
- {doc}`The format system </user_guide/format_system>` — how `"ds"`, `"ss"`,
  `"oo"`, and `"dd"` map onto compressed layouts.
