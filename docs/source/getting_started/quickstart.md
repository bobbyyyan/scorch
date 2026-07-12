# Quickstart

This page gets you from a dense `torch.Tensor` to a running sparse kernel in
about five minutes. You will build a sparse tensor, multiply it by a dense
matrix, express the same computation as an einsum, run a sparse matrix–vector
product, and check every result against a plain PyTorch reference.

Scorch is a drop-in accelerator for sparse linear algebra: you keep using
PyTorch tensors, and hand the sparse operand to Scorch when it pays off. The
snippets below are complete and runnable — paste them into a Python session with
the `scorch` conda environment active.

## Build a sparse tensor

Start from an ordinary dense `torch.Tensor` and convert it. `from_torch` wraps
the tensor (the second argument is a name used internally by the compiler), and
`to_sparse` picks a storage layout — here `"ds"`, which is CSR (dense rows,
compressed columns).

```python
import torch
import scorch

torch.manual_seed(0)

# A 256x256 matrix that is ~10% dense.
A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()

A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # -> STensor (CSR)
print(A.format)   # d,s
```

`A` is now an {class}`~scorch.STensor`, Scorch's sparse tensor. The `"ds"`
format string uses one character per mode: `d` = dense, `s` = compressed. See
{doc}`the format system </user_guide/format_system>` for the full vocabulary
(`"oo"` = COO, `"ss"` = DCSR, `"dd"` = dense).

## Sparse × dense with `matmul`

{func}`~scorch.matmul` is the tuned entry point. Give it a sparse left operand
and a dense right operand and it dispatches a sparse matrix–dense matrix product
(SpMM), returning a **dense `torch.Tensor`** — the result of a CSR × dense
product is dense, so no unwrapping is needed.

```python
X = torch.rand(256, 128)                # dense [256, 128]

Y = scorch.matmul(A, X)                 # -> dense torch.Tensor [256, 128]

assert torch.allclose(Y, A_dense @ X, atol=1e-3, rtol=1e-3)
print(type(Y), Y.shape)                 # <class 'torch.Tensor'> torch.Size([256, 128])
```

The `assert torch.allclose(..., atol=1e-3, rtol=1e-3)` check is Scorch's
correctness convention — every operation is verified against a PyTorch reference
at that tolerance.

:::{note}
**First call compiles, then caches.** The first time you run a given
operation for a specific combination of input/output formats and dtypes, Scorch
either dispatches a prebuilt C++ kernel or JIT-compiles one — so the first call
carries a one-time compile/dispatch cost. Every subsequent call with the same
shapes-class and formats reuses the cached kernel and runs at full speed. Time
the *second* call, not the first, when you benchmark. To warm the common SpMM
and SpGEMM kernels up front, call {func}`~scorch.precompile_kernels`.
:::

## The same thing with `einsum`

{func}`~scorch.einsum` is the general front door to the compiler. The SpMM above
is the contraction `$Y_{ik} = \sum_j A_{ij} X_{jk}$`, which in einsum notation is
`"ij,jk->ik"`. Pass `format="dd"` to request a dense output layout.

```python
Y2 = scorch.einsum("ij,jk->ik", A, X, format="dd")   # -> STensor (dense format)

assert torch.allclose(Y2.to_torch(), A_dense @ X, atol=1e-3, rtol=1e-3)
```

Two things to note:

- `einsum` returns an {class}`~scorch.STensor`, even when the output format is
  dense. Call `.to_torch()` to get a plain `torch.Tensor` back — that is exactly
  what `matmul` does for you internally on the dense-SpMM path.
- The expression **must** include an explicit `->` and output indices. An
  implicit-output form like `"ij,jk"` is not supported and raises an error.

:::{warning}
Format strings passed as a bare string are split one character per mode:
`"dd"` means two dense modes, `"ds"` means dense-then-compressed. This is why a
2-D dense output is `"dd"`, not `"d"`.
:::

## Sparse matrix–vector product (SpMV)

Give `matmul` a **1-D** dense right operand and it runs a sparse
matrix–vector product instead, returning a dense length-`M` vector.

```python
x = torch.rand(256)                     # 1-D dense vector

y = scorch.matmul(A, x)                 # SpMV -> dense torch.Tensor [256]

assert torch.allclose(y, A_dense @ x, atol=1e-3, rtol=1e-3)
```

There is no `scorch.spmv` in the public API — SpMV is reached through `matmul`
with a 1-D right operand, which selects the CSR SpMV kernel automatically.

## Bringing results back to PyTorch

`matmul` already hands you a `torch.Tensor` for dense results. When an operation
returns an {class}`~scorch.STensor` (for example a sparse-format result, or any
`einsum` output), convert it with `.to_torch()` to drop back into ordinary
PyTorch code:

```python
result_torch = Y2.to_torch()            # STensor -> torch.Tensor
```

For sparse outputs you can chain `.to_dense().to_torch()` to materialize the
dense form.

## Next steps

- {doc}`Key concepts </getting_started/key_concepts>` — STensors, format
  strings, the compiler pipeline, and how dispatch chooses a kernel.
- {doc}`Tutorials </tutorials/index>` — worked SpMV, SpMM, SDDMM, and SpGEMM
  kernels, plus end-to-end GCN, autoencoder, and transformer models.
- {doc}`User guide </user_guide/index>` — the format system, neural-network
  ops, `scorch.compile`, and autotuning in depth.
