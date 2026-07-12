# Key concepts

Scorch is a sparse-tensor compiler that wears a PyTorch-shaped glove. You build
sparse operands, call familiar operations like {func}`~scorch.matmul` and
{func}`~scorch.einsum`, and get dense `torch.Tensor` results back — while
underneath, Scorch generates and compiles a specialized C++ kernel tailored to
your exact sparsity layout.

You only need three ideas to use it well:

1. **An {class}`~scorch.STensor` is a torch tensor plus a *format*.**
2. **The format notation names the physical layout** — one letter per mode
   (`d`/`s`/`o`), so CSR is `"ds"` and COO is `"oo"`.
3. **Execution is compile-once:** the first call for a given format combination
   JIT-compiles a kernel and caches the `.so`; every later call runs at full
   speed.

The rest of this page unpacks each idea with a tiny example and a pointer to its
deep-dive page.

---

## 1. An STensor is a tensor + a format

A dense `torch.Tensor` stores every element. A sparse tensor stores only the
nonzeros — but *how* it stores them (which index arrays, in which order) is a
choice. Scorch makes that choice explicit and first-class: an
{class}`~scorch.STensor` is a thin logical handle that pairs your nonzero
**values** with a **format** describing where they live.

You almost never construct one by hand. The factory functions
{func}`~scorch.from_torch`, {func}`~scorch.from_coo`, and `scorch.from_csr`
build an STensor from an ordinary torch tensor:

```python
import torch
import scorch

dense = torch.tensor([[0., 2., 0.],
                      [1., 0., 3.]])

A = scorch.from_torch(dense.to_sparse_csr(), "A")  # CSR STensor

print(str(A.format))   # "d,s"   -> dense rows, compressed columns (CSR)
print(A.shape)         # (2, 3)
print(A.dim())         # 2       (note: .dim() is a method, not a .ndim property)
print(A.values)        # the nonzero values, 1-D

# Exit back to a plain dense torch.Tensor at any time:
assert torch.equal(A.to_torch(), dense)
```

The values are the numeric payload; the format is the structure wrapped around
them. Change the format and you change the physical layout without touching the
math — that is what {func}`~scorch.matmul` and the compiler read to decide how to
iterate.

:::{note}
Matmul is a **top-level function**, not an operator. `scorch.matmul(A, B)` is
correct; `A @ B` on two STensors is not supported (STensor defines no
`__matmul__`). Likewise there is no public `.nnz` or `.ndim` — use `.dim()` and
inspect `.shape`, `.format`, and `.values`.
:::

See {doc}`/user_guide/sparse_tensors` for the full data model — factories,
conversions (`to_torch` / `to_dense` / `to_sparse`), and the storage internals.

---

## 2. The format notation names the layout

Scorch describes every tensor's physical layout as **one level type per mode**,
reading left-to-right from the outermost dimension inward. There are three level
types you will use in practice:

| Letter | Level type    | Meaning                                              |
|:------:|:--------------|:-----------------------------------------------------|
| `d`    | dense         | store the whole extent; iterate every index          |
| `s`    | compressed    | store only nonzeros as a CSR-style pos/crd pair      |
| `o`    | coordinate    | store only nonzeros as a flat COO coordinate list    |

(`c` is an accepted synonym for `s`; a fourth type, `singleton`, exists in the
type system but is not yet lowered — do not rely on it.)

Compose those letters, one per dimension, and you get the familiar matrix
formats:

| Format string | Familiar name | Layout                                    |
|:-------------:|:--------------|:------------------------------------------|
| `"dd"`        | dense matrix  | fully dense, row-major                     |
| `"ds"`        | **CSR**       | dense rows + compressed columns            |
| `"oo"`        | **COO**       | coordinate row + coordinate column         |
| `"ss"`        | DCSR          | both modes compressed                      |

You pass these strings to `to_sparse` or as an output `format=`:

```python
import torch, scorch

A = scorch.from_torch(torch.eye(4), "A").to_sparse("ds")  # densify -> CSR
print(str(A.format))                                       # "d,s"
```

:::{warning}
A **bare string is split one character per mode** — `"ds"` becomes
`["d", "s"]`. That is exactly what you want for the single-letter alphabet, but
it means multi-character aliases only work in **list** form:
`TensorFormat(["dense", "compressed"])`, not `TensorFormat("dense")`. Also note
`str(fmt)` inserts commas (`"d,s"`), so the printed form differs from the input
form (`"ds"`).
:::

The format is not cosmetic: the compiler generates a different loop nest for a
dense level (iterate the full extent by arithmetic) than for a compressed or
coordinate level (walk only stored nonzeros via the pos/crd arrays). See
{doc}`/user_guide/format_system` for every level type, the alias table, and the
gotchas.

---

## 3. Execution is compile-once

The first time you call an operation on a new **format combination**, Scorch
runs the whole compiler pipeline — index notation, lowering, C++ codegen, and a
JIT compile with `-O3 -march=native -ffast-math -funroll-loops -fopenmp`. That
first call pays a one-time compile cost (a few seconds). The resulting shared
library is cached both **in-process** and **on disk**, so every subsequent call
— even in a fresh process — loads the cached `.so` and runs at full speed.

```python
import torch, scorch

A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()
x = torch.rand(256, 128)

A = scorch.from_torch(A_dense, "A").to_sparse("ds")  # CSR STensor

Y = scorch.matmul(A, x)   # 1st call: compiles + caches the kernel
Y = scorch.matmul(A, x)   # 2nd call: reuses the cached .so, full speed

assert torch.allclose(Y, A_dense @ x, atol=1e-3, rtol=1e-3)
```

The cache is keyed by the operand and output **formats**, not by shape or
values — so a warmed kernel is reused across every matrix of the same format.
If you want to pay the compile cost up front (e.g. before timing), warm the
common SpMM/SpGEMM combinations once with
{func}`~scorch.precompile_kernels`.

:::{tip}
Because the cache is keyed on formats, mixing formats needlessly (some operands
CSR, some COO) can trigger extra compiles. Pick a format per role and stick with
it across a workload.
:::

Correctness convention: Scorch verifies against a PyTorch reference with
`torch.allclose(..., atol=1e-3, rtol=1e-3)`. Every example on this site follows
that pattern — paste one and check it yourself.

See {doc}`/compiler/pipeline` for the full journey from a call to a compiled
kernel.

---

## How a call is dispatched

When you call {func}`~scorch.matmul` (or {func}`~scorch.einsum`), Scorch tries
three tiers in order and stops at the first that applies:

```{mermaid}
flowchart TD
    call["matmul(a, b)"] --> dense{both operands<br/>dense?}
    dense -- yes --> torch["delegate to torch.matmul"]
    dense -- no --> prebuilt{prebuilt C++ kernel<br/>for these formats?}
    prebuilt -- yes --> hand["hand-written scorch_ops kernel<br/>(e.g. CSR × dense SpMM)"]
    prebuilt -- no --> jit["generic JIT compiler<br/>(CIN -> LLIR -> C++ -> .so)"]
```

1. **Both-dense → PyTorch.** If both operands are dense, the call delegates
   straight to `torch.matmul` — Scorch never gets in the way of the dense path.
2. **Prebuilt hand-written kernel.** For the hot sparse shapes (CSR × dense SpMM,
   CSR × CSR SpGEMM, COO paths, CSR × vector SpMV), Scorch dispatches to a
   hand-optimized C++ kernel in the native `scorch_ops` extension.
3. **Generic compiler.** Everything else — arbitrary einsum strings, less common
   format combinations, higher-rank contractions — flows through the JIT
   compiler pipeline described above.

This is why Scorch stays fast on the common cases without limiting you to them:
the tuned prebuilt kernels handle the workhorse shapes, and the compiler is the
general fallback for anything they don't cover.

:::{admonition} A note on tuning
:class: tip
The CSR × dense SpMM path also has an **autotune** knob — a compiler-style
`-O` ladder (`off` / `analytic` / `balanced` / `max` / `learned`) that picks a
tiling strategy for large, cache-thrashing graph workloads and is a no-op
everywhere else. Most users never touch it. See {doc}`/user_guide/autotuning`.
:::

---

## Next steps

- {doc}`/getting_started/quickstart` — a five-minute end-to-end walkthrough.
- {doc}`/user_guide/sparse_tensors` — the STensor data model in depth.
- {doc}`/user_guide/format_system` — every level type and format string.
- {doc}`/compiler/pipeline` — how a call becomes a compiled kernel.
