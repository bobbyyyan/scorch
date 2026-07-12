# Operations

Scorch's public operations are the everyday verbs of sparse linear algebra:
matrix–vector and matrix–matrix products, sampled dense–dense products, and
sparse–sparse products. They live in {func}`~scorch.matmul`,
{func}`~scorch.einsum`, and {func}`~scorch.matmul_wksp`, and they all share one
promise — you write the same call you would in PyTorch, and Scorch picks the
fastest correct path for the formats you hand it.

```python
import torch
import scorch                       # NOT `import scorch as torch`

A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()
x       = torch.rand(256, 128)

A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR STensor
Y = scorch.matmul(A, x)                                # -> dense torch.Tensor

assert torch.allclose(Y, A_dense @ x, atol=1e-3, rtol=1e-3)
```

Across the sparse workloads we track, Scorch runs **1.05–5.80× faster than
PyTorch Sparse** (CGO 2026). This page documents each operation: its signature,
what it returns, and a runnable example verified against a PyTorch reference.

:::{note}
Examples use the explicit `import torch` / `import scorch` form. The
`import scorch as torch` drop-in shim also works — anything Scorch doesn't
define falls through to real PyTorch — and the application tutorials
({doc}`GCN </tutorials/gcn>`, {doc}`autoencoder </tutorials/sparse_autoencoder>`,
{doc}`transformer </tutorials/sparse_transformer>`) use it deliberately. For an
API reference page we keep the two names separate so it's always obvious which
library a call belongs to.
:::

## How dispatch works

`matmul` and `einsum` both resolve a call through three tiers, tried in order.
Understanding them explains why the same `matmul` sometimes returns a dense
`torch.Tensor` and sometimes an {class}`~scorch.STensor`, and why the first call
for a new format combination is slower than the rest.

```{mermaid}
flowchart TD
    call["matmul(a, b) / einsum(expr, ...)"] --> t1{both dense?}
    t1 -- yes --> dense["torch.matmul → dense torch.Tensor"]
    t1 -- no --> t2{prebuilt C++ kernel<br/>for these formats?}
    t2 -- yes --> pre["hand-written scorch_ops kernel"]
    t2 -- no --> t3["generic JIT compiler<br/>CIN → LLIR → C++ → .so"]
```

1. **Both-dense delegate.** If both operands are dense, the call goes straight to
   `torch.matmul` and returns a dense `torch.Tensor` (unless you request a sparse
   output format, in which case it converts).
2. **Prebuilt hand-written C++ kernel.** For the common sparse shapes — CSR ×
   dense, CSR × CSR, COO × COO, COO × dense, CSR × vector — Scorch dispatches a
   hand-optimized kernel in the native `scorch_ops` extension.
3. **Generic JIT compiler.** Anything else is lowered through the compiler
   pipeline (Compiler Index Notation → Low-Level IR → generated C++ → a JIT-
   compiled `.so`) and cached. See {doc}`the compiler pipeline </compiler/pipeline>`.

Every tier is memoized. The prebuilt resolver, the workspace path, and `einsum`
each keep an in-process cache keyed by operand formats and dtypes, backed by a
persistent `.so` cache on disk. The first call for a novel format combination
pays a one-time compile; subsequent calls hit the cache.

:::{tip}
Warm the cache ahead of time with {func}`~scorch.precompile_kernels`, which
compiles the common SpMM/SpGEMM format combinations up front so the first real
call is fast. It is not run at import — call it explicitly.
:::

## `matmul`

```python
scorch.matmul(a, b, **kwargs) -> Union[torch.Tensor, STensor]
```

The general matrix-multiply entry point and the tuned front door to every SpMV,
SpMM, and SpGEMM path. `a` and `b` may each be a `torch.Tensor` or an
{class}`~scorch.STensor`; torch inputs are auto-converted (dense stays dense;
sparse COO inputs are promoted to CSR).

**Keyword arguments**

`format` / `output_format` (str | list | {class}`~scorch.TensorFormat`)
: The requested output layout, e.g. `"dd"` (dense), `"ds"` (CSR), `"oo"` (COO).
  `matmul` reads `format` first, then `output_format`. If omitted, the output
  format is inferred from the operands.

`use_cache` (bool, default `True`)
: When `True`, allow the prebuilt fast path and the adaptive tiling selector.
  Set `False` to force the generic compiler path (mainly useful for testing the
  JIT pipeline).

`time_dict` (dict)
: If supplied, `time_dict["eval_time"]` is filled with the kernel execution wall
  time in seconds.

**Returns.** A dense `torch.Tensor` when the result format is dense (the common
SpMM case, `ds @ dd -> dd`), or an {class}`~scorch.STensor` when the result is
sparse (SpGEMM, `ds @ ds -> ds`). A sparse operand shape drives which prebuilt
kernel fires; a 1-D right operand routes to SpMV.

```python
import torch, scorch

A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()
x       = torch.rand(256, 128)

A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR STensor
Y = scorch.matmul(A, x)                                # dense torch.Tensor [256,128]
assert torch.allclose(Y, A_dense @ x, atol=1e-3, rtol=1e-3)
```

Request a sparse output to keep the result compressed:

```python
B = scorch.from_torch(A_dense, "B").to_sparse("ds")
C = scorch.matmul(A, B, format="ds")                  # STensor (CSR)
assert torch.allclose(C.to_torch(), A_dense @ A_dense, atol=1e-3, rtol=1e-3)
```

:::{note}
Two environment knobs tune the SpMM thread policy for the drop-in CSR kernel,
both default on: `SCORCH_MATCH_HOST_THREADS=0` disables passing
`torch.get_num_threads()` to the kernel, and `SCORCH_ATPARALLEL_PIPELINE=0`
launches the SpMM on a private OpenMP team instead of torch's intra-op pool. You
rarely need to touch these; they exist for GCN-style pipelines that manage their
own thread teams.
:::

## `einsum`

```python
scorch.einsum(expression, *tensors, compile_only=False, **kwargs) -> STensor
```

The compiler front door. `einsum` takes a numpy-style index expression and any
number of operands, and lowers the computation through the full CIN → LLIR →
codegen pipeline (with a fast dispatch cache and dedicated prebuilt fast paths
for a few common patterns). `matmul` is a thin, tuned wrapper that ultimately
emits an `einsum` call for shapes without a prebuilt kernel.

:::{warning}
`einsum` **requires an explicit `->` output specification.** The parser does
`expression.split("->")`, so an implicit-output form like `"ik,kj"` (valid in
`numpy.einsum`) raises `IndexError`. Always write the output, e.g.
`"ik,kj->ij"`.
:::

**Parameters**

`expression` (str)
: A comma-separated list of input index groups, then `->`, then the output
  indices — one character per index. Examples: `"ik,kj->ij"` (matmul),
  `"ij,ij->ij"` (elementwise multiply), `"ij,ik,jk->ij"` (SDDMM).

`*tensors`
: The operands, torch or {class}`~scorch.STensor`. The count must match the
  number of comma-separated input groups.

`compile_only` (bool, default `False`)
: Compile and cache the kernel but return a placeholder without executing — the
  mechanism behind {func}`~scorch.precompile_kernels`.

`format` (str | list | {class}`~scorch.TensorFormat`, keyword)
: The requested output format. If omitted, it is inferred (see below).
  `output_mode_order` (a permutation of the output modes) and `time_dict` are
  also accepted.

**Returns.** An {class}`~scorch.STensor` in the resolved output format. When the
format is dense, call `.to_torch()` to get a `torch.Tensor` (this is exactly what
`matmul` does for you on the dense path).

### Output-format inference

When you omit `format=`, Scorch infers the output layout per mode:

- sparse × anything → sparse;
- dense + anything → dense;
- otherwise compressed.

Ties prefer coordinate over compressed, and a sparse level may not precede a
dense level (a preceding sparse level is forced dense). One special case: a
sparse output with a reduction variable and an input tensor mirroring the output
sparsity resolves to an all-COO output, which enables the scalar-accumulator
SDDMM codegen path.

### The supported subset

The compiler covers binary contractions and elementwise / SDDMM patterns.
Confirmed-working expressions:

| Expression | Operation |
|---|---|
| `"ik,kj->ij"` | matmul / SpMM / SpGEMM (all format combos) |
| `"ij,jk->ik"` | the form `matmul` emits internally |
| `"ij,ij->ij"` | elementwise multiply |
| `"ij,ik,jk->ij"` | SDDMM (has a dedicated prebuilt fast path) |

Transposed and non-default mode-order operands, dense-STensor × dense-STensor
products, and broadcast-vector patterns are all supported too — these live as
green regression tests in `tests/test_scorch/test_known_compiler_gaps*.py`.
Despite the filename, those are *passing* tests documenting historically fragile
combinations that now work; consult them before assuming a format/mode-order
combination is unsupported. There is no enumerated list of *unsupported*
expressions — patterns the compiler can't lower surface as compiler exceptions
rather than documented errors.

```python
import torch, scorch

A = scorch.from_torch((torch.rand(64, 32) < .2).float(), "A").to_sparse("ds")
B = torch.rand(32, 48)
C = scorch.einsum("ik,kj->ij", A, B, format="dd")     # STensor, dense format
assert torch.allclose(C.to_torch(), A.to_torch() @ B, atol=1e-3, rtol=1e-3)
```

## SpMV — sparse matrix × dense vector

Give `matmul` a 2-D sparse left operand and a 1-D right operand and it computes a
sparse matrix–vector product $y_i = \sum_j A_{ij}\,x_j$, returning a dense vector.

```python
import torch, scorch

A = scorch.from_torch((torch.rand(128, 128) < .1).float(), "A").to_sparse("ds")
x = torch.rand(128)

y = scorch.matmul(A, x)                                # dense vector [128]
assert torch.allclose(y, A.to_torch() @ x, atol=1e-3, rtol=1e-3)
```

Under the hood `matmul` tries the prebuilt CSR SpMV kernel first and falls back
to an internal `spmv` function on a miss.

:::{warning}
There is no `scorch.spmv`. The `spmv` function exists in `ops.py` but is **not
exported** — `scorch.spmv` would fall through to `getattr(torch, "spmv")` and
raise `AttributeError`. Reach SpMV through `matmul(A, x)` with a 1-D `x`.
Advanced users can call it directly as {func}`~scorch.ops.spmv`.
:::

## SpMM — sparse matrix × dense matrix

SpMM computes $C_{ik} = \sum_j A_{ij}\,B_{jk}$ with a sparse `A` (typically CSR,
`"ds"`) and a dense `B`, producing a dense result. It is Scorch's flagship
kernel: much of the performance work — adaptive tiling, register-blocking, fused
Linear layers — targets it.

```python
import torch, scorch

A_dense = torch.tensor([
    [0., 2., 0., 0., 1.],
    [0., 0., 0., 3., 0.],
    [4., 0., 0., 0., 0.],
    [0., 0., 5., 0., 6.],
])
A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR [4x5]
B = torch.rand(5, 8)                                   # dense [5x8]

C = scorch.matmul(A, B)                                # dense torch.Tensor [4x8]
assert torch.allclose(C, A_dense @ B, atol=1e-3, rtol=1e-3)
```

For neural-network layers, prefer the fused entry points — `scorch.sparse_linear`
folds SpMM + bias + activation into one kernel — described in
{doc}`neural network operations </user_guide/neural_network_ops>`. To control the
SpMM schedule (tiling, register-blocking, learned cost model), see
{doc}`autotuning </user_guide/autotuning>`.

## SDDMM — sampled dense–dense product

SDDMM samples a dense–dense product only where a sparse pattern `S` is nonzero:
$\text{Out}_{ij} = S_{ij} \cdot \sum_k A_{ik} B_{jk}$. It keeps `S`'s sparsity and
is the natural fit for `einsum`, since the sparse operand on the left of the
output restricts which positions are computed.

```python
import torch, scorch

# sparse pattern S as COO [3x4] (all-ones, so S acts as a mask)
idx = torch.tensor([[0, 1, 2, 2],
                    [1, 3, 0, 3]])
val = torch.tensor([1.0, 1.0, 1.0, 1.0])
S = torch.sparse_coo_tensor(idx, val, (3, 4)).coalesce()

A = torch.rand(3, 1)                                   # [M, 1]
B = torch.rand(1, 4)                                   # [1, N]

out = scorch.einsum("ij,ik,kj->ij", S, A, B)           # sampled A@B on S's pattern
ref = torch.mul(S, torch.matmul(A, B)).coalesce()      # torch reference
assert torch.allclose(out.to_dense().to_torch(),
                      ref.to_dense(), atol=1e-3, rtol=1e-3)
```

:::{important}
Mind the index order. The example above uses `"ij,ik,kj->ij"` with `A` shaped
`[M, r]` and `B` shaped `[r, N]` — here `A @ B` is an ordinary product sampled at
`S`. Scorch also ships a **dedicated prebuilt fast path** for a *different* order,
`"ij,ik,jk->ij"`, where `S` is COO, `A` is `[M, r]`, `B` is `[N, r]`, all float32
— that computes $\text{Out}_{ij} = S_{ij} \cdot \sum_k A_{ik} B_{jk} =
S_{ij}\,(A B^\top)_{ij}$ and hits the hand-written `sddmm_coo_float_prebuilt`
kernel. Pick the order that matches your operands.
:::

Because `S` is a genuine operand in the einsum expression, its *values* multiply
in — a mask of all ones reproduces pure masking, while nonunit `S` values scale
each sampled entry:

```python
import torch, scorch

S = scorch.from_coo(torch.rand(50, 50).to_sparse_coo(), "S")   # nonunit values
A = torch.rand(50, 16)
B = torch.rand(50, 16)

Sd  = scorch.einsum("ij,ik,jk->ij", S, A, B)                   # prebuilt fast path
ref = S.to_torch() * (A @ B.T)                                 # S's values scale in
assert torch.allclose(Sd.to_dense().to_torch(), ref, atol=1e-3, rtol=1e-3)
```

## SpGEMM / SpMSpM — sparse matrix × sparse matrix

When both operands are sparse, `matmul` computes a sparse–sparse product and
returns a sparse {class}`~scorch.STensor`. Request `format="ds"` (CSR) to keep
the result compressed.

```python
import torch, scorch

idx = torch.tensor([[0, 0, 1, 2, 2],
                    [0, 2, 1, 0, 2]])
val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
A   = torch.sparse_coo_tensor(idx, val, (3, 3)).coalesce()
A_T = A.transpose(0, 1)

C = scorch.matmul(A, A_T, format="ds")                 # STensor (CSR)
ref = A.to_dense() @ A_T.to_dense()
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

:::{note}
COO operands are promoted to CSR internally before the sparse–sparse kernel
runs (`matmul` calls `to_sparse_csr()` when both inputs are torch `sparse_coo`).
A native COO × COO → COO prebuilt path also exists; which fires depends on the
operand formats. Either way you get a correct sparse result.
:::

## `matmul_wksp` — the explicit workspace path

```python
scorch.matmul_wksp(a, b, output_format="ds", **kwargs) -> STensor
```

`matmul_wksp` is a workspace-based SpMM/SpGEMM that **always** goes through the
CIN compiler — it never takes a prebuilt kernel. It builds a `Where` with a
{doc}`workspace </compiler/workspaces>` (dense when the output is dense, else
COO-hashed) that accumulates over the contraction indices, and JIT-compiles once
per `(a.format, b.format, output_format)` combination.

Unlike `matmul`, both torch inputs are always converted to sparse, and the result
is **always** an {class}`~scorch.STensor` of shape `(a.shape[0], b.shape[1])` —
it does not auto-convert a dense-format result back to `torch.Tensor`. Use
`matmul` for production (it is the tuned entry point with prebuilt kernels,
tiling, and thread policy); reach for `matmul_wksp` when you specifically want to
exercise the workspace-lowering codegen path.

```python
import torch, scorch

A = scorch.from_torch((torch.rand(64, 64) < .1).float(), "A").to_sparse("ds")
B = scorch.from_torch((torch.rand(64, 64) < .1).float(), "B").to_sparse("ds")

C = scorch.matmul_wksp(A, B, output_format="ds")       # STensor
ref = A.to_torch() @ B.to_torch()
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

## Reference table

| Operation | Scorch call | PyTorch reference |
|---|---|---|
| SpMV (sparse · vector) | `scorch.matmul(A_csr, x_vec)` | `torch.matmul` |
| SpMM (sparse · dense) | `scorch.matmul(A_csr, B_dense)` | `torch.sparse.mm` |
| SpMM, dense output | `scorch.matmul(A, B, format="dd")` | `torch.sparse.mm` |
| SDDMM (sampled) | `scorch.einsum("ij,ik,kj->ij", S, A, B)` | `torch.mul(S, A @ B)` |
| SDDMM (prebuilt order) | `scorch.einsum("ij,ik,jk->ij", S, A, B)` | `S * (A @ B.T)` |
| SpGEMM (sparse · sparse) | `scorch.matmul(A, B, format="ds")` | `torch.matmul` |
| SpGEMM (workspace path) | `scorch.matmul_wksp(A, B, output_format="ds")` | `torch.matmul` |
| Elementwise multiply | `scorch.einsum("ij,ij->ij", A, B)` | `torch.mul` |

Match the correctness convention when checking your own code: compare against a
PyTorch reference with `assert torch.allclose(..., atol=1e-3, rtol=1e-3)`.

## Next steps

- {doc}`Neural network operations </user_guide/neural_network_ops>` — the fused
  `sparse_linear`, `sparse_attention`, and softmax kernels built on SpMM.
- {doc}`Autotuning </user_guide/autotuning>` — controlling the SpMM schedule and
  the learned cost model.
- {doc}`Tutorials </tutorials/index>` — end-to-end SpMV, SpMM, SDDMM, and SpGEMM
  walkthroughs plus full GCN / autoencoder / transformer models.
- {doc}`API reference: operations </api/operations>` — the complete signatures.
