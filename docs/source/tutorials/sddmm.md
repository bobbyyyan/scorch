# SDDMM: sampled dense-dense product

**SDDMM** (Sampled Dense-Dense Matrix Multiplication) evaluates a dense
matrix product *only* at the nonzero positions of a sparse mask. It is the
workhorse behind graph attention, factorization-machine scoring, and the
similarity step of many self-supervised losses — anywhere you need
`A @ B` but only care about a handful of its entries.

Formally, given a sparse pattern `S` and dense factors `A`, `B`:

$$\text{Out}[i,j] = S[i,j] \cdot (A B)[i,j], \qquad \text{only where } S[i,j] \neq 0.$$

The output keeps `S`'s sparsity. Because you never materialize the full dense
product `A @ B` — you evaluate it at the (typically few) sampled positions — the
work scales with `nnz(S)` rather than `M·N`. Scorch expresses this as a single
{func}`~scorch.einsum`.

## The one-liner

The kernel benchmark in `examples/kernels/sddmm.py` samples a **rank-1** product
(`A` is `[M, 1]`, `B` is `[1, N]`) at the pattern of a COO matrix `S`. The Scorch
call is one line:

```python
result = scorch.einsum("ij,ik,kj->ij", S, A, B)
```

Read the einsum string operand-by-operand: `S` is `ij` (the sparse mask and the
output support), `A` is `ik`, `B` is `kj`, and the output is `ij`. The `k` index
is contracted; the sparse operand `S` on the left of `->ij` restricts which `(i,
j)` positions are ever computed. This is the defining move of a sampled product:
the sparse pattern gates the dense contraction.

:::{note}
Scorch's `einsum` requires an **explicit** `->` and output spec. There is no
implicit-output mode — always write the full `"...->ij"` form.
:::

## A complete, verified example

Here is the smallest correct SDDMM you can paste and run. It builds a `[3, 4]`
COO pattern, samples a rank-1 dense product, and checks the result against a
PyTorch reference that computes the full product and masks it.

```python
import torch
import scorch

torch.manual_seed(0)

# Sparse pattern S as COO [3 x 4] — four sampled positions.
idx = torch.tensor([[0, 1, 2, 2],
                    [1, 3, 0, 3]])
val = torch.tensor([1.0, 1.0, 1.0, 1.0])
S = torch.sparse_coo_tensor(idx, val, (3, 4)).coalesce()

A = torch.rand(3, 1, dtype=torch.float32)   # [M, 1]
B = torch.rand(1, 4, dtype=torch.float32)   # [1, N]

out = scorch.einsum("ij,ik,kj->ij", S, A, B)

# torch reference: form the full product, then mask by S's pattern.
ref = torch.mul(S, torch.matmul(A, B)).coalesce()

out_dense = out.to_dense().to_torch() if hasattr(out, "to_dense") else out
assert torch.allclose(out_dense, ref.to_dense(), atol=1e-3, rtol=1e-3)
print("SDDMM OK")
```

The result comes back as an {class}`~scorch.STensor`; call `.to_dense().to_torch()`
(or `.to_torch()`) to compare it against a dense torch tensor.

:::{note}
The `atol=rtol=1e-3` tolerance is Scorch's project-wide correctness convention —
every kernel is validated against a PyTorch reference at that precision, and your
own SDDMM code should follow the same pattern rather than hard-coding expected
values. Here the reference `torch.mul(S, A @ B)` deliberately computes the *whole*
dense product and throws most of it away; Scorch does the same math but only at
`S`'s nonzeros.
:::

`S` above is a 0/1 mask, so `Out` equals the sampled product itself. If `S`
carries real weights, each sampled entry is additionally scaled by `S[i,j]` — the
$S[i,j]\cdot$ factor in the definition — which is exactly what you want for a
weighted-attention or weighted-similarity score.

## Two index orders: `kj` vs `jk`

There is a subtlety worth pinning down. The example above uses the index order
`"ij,ik,kj->ij"`, where the second dense factor `B` is `[k, j]`. Scorch also ships
a **prebuilt fast-path** SDDMM kernel (`sddmm_coo_float_prebuilt`) that fires for
the *transposed* order `"ij,ik,jk->ij"` — where `B` is `[j, k]` — with a COO `S`
and `float32` `A`, `B`.

The two are the same mathematics with `B` laid out differently:

| einsum string | `A` shape | `B` shape | product at `(i,j)` |
|---|---|---|---|
| `"ij,ik,kj->ij"` | `[M, r]` | `[r, N]` | $\sum_k A[i,k]\,B[k,j]$ |
| `"ij,ik,jk->ij"` | `[M, r]` | `[N, r]` | $\sum_k A[i,k]\,B[j,k]$ |

In the `jk` form both dense factors are indexed by their *row* (`i` for `A`, `j`
for `B`) and share the contraction index `k` in the last position — the
row-major, cache-friendly layout the prebuilt kernel is written for. To hit it,
transpose your second factor so it is `[N, r]` instead of `[r, N]`:

```python
import torch
import scorch

torch.manual_seed(1)

idx = torch.tensor([[0, 1, 2, 2],
                    [1, 3, 0, 3]])
val = torch.tensor([1.0, 1.0, 1.0, 1.0])
S = torch.sparse_coo_tensor(idx, val, (3, 4)).coalesce()

r = 3
A = torch.rand(3, r, dtype=torch.float32)   # [M, r]
B = torch.rand(4, r, dtype=torch.float32)   # [N, r]  (note: N rows, r cols)

# Prebuilt fast-path order: B is [j, k], contraction k is last on both factors.
out = scorch.einsum("ij,ik,jk->ij", S, A, B)

# torch reference: mask the full A @ B.T by S.
ref = torch.mul(S, torch.matmul(A, B.t())).coalesce()

out_dense = out.to_dense().to_torch() if hasattr(out, "to_dense") else out
assert torch.allclose(out_dense, ref.to_dense(), atol=1e-3, rtol=1e-3)
print("SDDMM (jk fast-path) OK")
```

:::{warning}
The index letters are load-bearing. `"ij,ik,kj->ij"` and `"ij,ik,jk->ij"` compute
the same values **only if you also transpose `B`** — `[r, N]` for `kj` versus
`[N, r]` for `jk`. Mixing the string with the wrong `B` shape either errors on the
dimension mismatch or silently computes a different contraction. Match the string
to the layout, and verify against `torch.mul(S, A @ B)` (or `A @ B.T`) whenever you
switch orders.
:::

Both orders produce the same result — the reference check passes for each — so the
choice is purely about performance. Prefer the `jk` order with `float32` factors
and a COO `S` when you want the hand-tuned kernel; the general `kj` form goes
through the JIT-generated codegen path and works for any dtype and layout.

## The full benchmark

`examples/kernels/sddmm.py` runs this same operation across the SuiteSparse
Matrix Collection — loading each matrix as COO, sampling a rank-1 product, timing
Scorch against the `torch.mul(S, A @ B)` baseline, and plotting runtime versus
nonzero count. It filters to matrices with `max(rows, cols) < 800000`. See that
script for the reproducible benchmark harness; the distilled snippets above are
the minimal correct programs that exercise the same API.

## See also

- {doc}`Operations </user_guide/operations>` — the full reference for
  {func}`~scorch.matmul` and {func}`~scorch.einsum`, including output-format
  control and the sparse-operand dispatch rules that decide when a prebuilt kernel
  versus the JIT path fires.
- {doc}`SpMM </tutorials/spmm>` — the sparse × dense product, Scorch's flagship
  kernel and the closest cousin of SDDMM.
- {doc}`SpGEMM </tutorials/spgemm>` — sparse × sparse, when both operands carry a
  pattern.
