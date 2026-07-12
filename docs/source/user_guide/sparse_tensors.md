# Sparse tensors

The {class}`~scorch.STensor` is Scorch's user-facing sparse tensor. It is a thin
*logical handle* — a name, a logical shape, a logical dtype — that delegates all of
its numeric payload to an internal storage object holding one flat array of nonzero
values plus a compact description of where those values live. This page covers how
to build an STensor from ordinary PyTorch data, how to inspect it, how to change its
layout, and how to hand results back to PyTorch.

If you want the details of the layout notation itself (`"ds"` = CSR, `"oo"` = COO,
level types, and their gotchas), read
{doc}`the format system </user_guide/format_system>` alongside this page. For the
operations that consume STensors — `matmul`, `einsum`, `spmv` — see
{doc}`operations </user_guide/operations>`.

:::{note}
Scorch is a **CPU** sparse-tensor compiler. An STensor carries no autograd, and the
device-movement methods (`.cuda()`, `.to()`) are placeholders that raise
`NotImplementedError`. Treat STensors as transient, immutable-ish data you feed into
Scorch ops and then materialize back to PyTorch.
:::

## A quick tour

```python
import torch
import scorch

# Start from an ordinary dense PyTorch matrix.
dense = torch.tensor([[0., 2., 0.],
                      [1., 0., 3.]])

# Compress it to CSR and hand it to Scorch.
A = scorch.from_csr(dense.to_sparse_csr(), name="A")

print(A.shape)                      # (2, 3)
print(str(A.format))                # d,s   (CSR: dense rows, compressed cols)
print(A.values.tolist())            # [2.0, 1.0, 3.0]  -- the nonzeros, flat

# Materialize back to a dense torch.Tensor and check we round-tripped.
back = A.to_torch()
assert torch.equal(back, dense)
```

The rest of this page unpacks each of those steps.

:::{admonition} Import convention
:class: tip
Throughout the user guide we write `import torch` then `import scorch` so it is
always clear which library a symbol comes from. The application tutorials
({doc}`GCN </tutorials/gcn>`, {doc}`autoencoder </tutorials/sparse_autoencoder>`,
{doc}`transformer </tutorials/sparse_transformer>`) instead use
`import scorch as torch` — that drop-in idiom is the whole point there, and it is
explained under [PyTorch interop and the shim](#pytorch-interop-and-the-shim)
below.
:::

## Construction

You almost never call `STensor(...)` directly — the constructor expects a
hand-built storage object. Instead use one of the three factory functions, which are
re-exported at module scope:

| Factory | Accepts | Resulting format |
|---|---|---|
| {func}`~scorch.from_torch` | dense, `torch.sparse_coo`, or `torch.sparse_csr` | inferred from the input layout |
| {func}`~scorch.from_coo` | a torch COO tensor **or** raw `indices`/`values`/`shape` | `"oo…"` (coordinate per mode) |
| `scorch.from_csr` | a 2-D `torch.sparse_csr` tensor | `"ds"` (CSR) |

### `from_torch` — the workhorse

`from_torch(tensor, name=None, mode_order=None)` accepts any of PyTorch's three
common layouts and picks the Scorch format automatically:

```python
import torch, scorch

# Dense input -> every mode is DENSE ("dd" for a matrix).
d = scorch.from_torch(torch.arange(12, dtype=torch.float32).reshape(3, 4), "D")
assert str(d.format) == "d,d"

# torch CSR input -> "ds" (canonical CSR).
csr = torch.tensor([[0., 2., 0.], [1., 0., 3.]]).to_sparse_csr()
c = scorch.from_torch(csr, "C")
assert str(c.format) == "d,s"

# torch COO input -> "oo" (coordinate per mode); coalesced first.
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3., 4., 5.])
o = scorch.from_torch(torch.sparse_coo_tensor(i, v, (2, 3)), "O")
assert str(o.format) == "o,o"
```

**`name`** defaults to `"tensor"` when omitted. It is a label used in generated
kernel code and diagnostics — see [Inspection](#inspection) for why you may want to
set it.

**`mode_order`** is an optional permutation applied to the input *before* storage
(`tensor.permute(*mode_order)`); the inverse is recorded so that
[`to_torch`](#conversion) hands you back the tensor in its original logical axis
order. This is how Scorch represents a transposed or relaid-out operand without
recomputing it — for example, storing a matrix column-major:

```python
import torch, scorch

t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
m = scorch.from_torch(t, "M", mode_order=[1, 0])   # store transposed
assert m.shape == (3, 2)                            # logical shape is permuted
assert torch.equal(m.to_torch(), t)                 # to_torch inverts it back
```

### `from_coo` — coordinate tensors (any rank)

`from_coo` builds an all-`COORDINATE` STensor and, unlike `from_csr`, works for
arbitrary rank. You can call it two ways — with a torch COO tensor, or with raw
arrays:

```python
import torch, scorch

i = torch.tensor([[0, 1, 1], [2, 0, 2]])   # shape [ndim, nnz]
v = torch.tensor([3., 4., 5.])             # shape [nnz]

# (1) from a torch sparse-COO tensor (coalesced internally):
coo = torch.sparse_coo_tensor(i, v, (2, 3))
a = scorch.from_coo(coo, name="S")

# (2) from raw indices / values / shape (no torch sparse tensor needed):
b = scorch.from_coo(indices=i, values=v, shape=(2, 3), name="S")

assert str(a.format) == "o,o"
assert torch.equal(a.to_torch(), b.to_torch())
```

### `from_csr` — 2-D CSR matrices

`from_csr(csr_matrix, name=None)` is the specialized path for 2-D CSR. It asserts
the input `is_sparse_csr` and that it is exactly 2-D, and produces the `"ds"` format
(`from_torch` on a CSR input does the same thing):

```python
import torch, scorch

dense = torch.tensor([[0., 2., 0.], [1., 0., 3.]])
W = scorch.from_csr(dense.to_sparse_csr(), "W")
assert str(W.format) == "d,s"
assert torch.equal(W.to_torch(), dense)
```

:::{note}
Index arrays are coerced to 32-bit integers internally, so an `int64`
`crow`/`col`/coordinate array you pass in becomes `int32` inside the STensor. The
stored **values** keep their dtype.
:::

## Conversion

Three methods move an STensor between layouts or back into PyTorch. Keep the return
types straight: `to_dense` and `to_sparse` return **STensors** (a layout change that
stays inside Scorch), while `to_torch` returns a **dense `torch.Tensor`** (the exit
door back to PyTorch).

### `to_torch` — back to a dense PyTorch tensor

```python
import torch, scorch

A = scorch.from_csr(torch.tensor([[0., 2.], [1., 0.]]).to_sparse_csr(), "A")
back = A.to_torch()                 # dense torch.Tensor, logical axis order
assert isinstance(back, torch.Tensor)
```

`to_torch` always densifies — there is no sparse-to-sparse export. It also undoes any
`mode_order` permutation, so what you get back is in the original logical shape.

:::{warning}
`to_torch(in_place=True)` is the **default**, and it may replace the STensor's
internal storage with the densified version as a side effect. If you want to inspect
`.values` / `.index` *after* materializing — or keep the STensor sparse for a later
op — either inspect **before** calling `to_torch`, or pass `in_place=False`:

```python
d = A.to_torch(in_place=False)      # A stays sparse
```
:::

### `to_dense` / `to_sparse` — layout changes within Scorch

```python
import torch, scorch

s = scorch.from_csr(torch.tensor([[0., 5.], [0., 0.]]).to_sparse_csr(), "S")

# to_dense -> a NEW all-dense STensor (in_place=False is the default).
d = s.to_dense()
assert isinstance(d, scorch.STensor) and d.format.is_dense()
assert str(s.format) == "d,s"       # original is untouched

# to_sparse(fmt) -> MUTATES in place and returns self.
r = s.to_sparse("ss")               # recompress to DCSR
assert r is s
assert str(s.format) == "s,s"
```

The `fmt` argument accepts a format string like `"ss"`, a list like
`["compressed", "compressed"]`, or a {class}`~scorch.TensorFormat`. When omitted,
`to_dense` targets all-dense and `to_sparse` targets an all-compressed form.

:::{note}
`to_sparse` **always mutates in place and returns `self`** (it has no `in_place`
flag), whereas `to_dense` defaults to `in_place=False` and returns a fresh STensor.
Both compile a small C++ kernel on first use, so the first call to a new
shape/format pays a one-time JIT cost that is then cached.
:::

### `change_mode_order` — transpose / relayout

`change_mode_order(mode_order)` permutes the logical axes of an existing tensor,
mutating its storage and shape and returning `self`. As with `mode_order` on
`from_torch`, `to_torch` inverts the permutation on the way out:

```python
import torch, scorch

t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
a = scorch.from_torch(t, "A")
a.change_mode_order([1, 0])         # transpose
assert a.shape == (3, 2)
assert torch.equal(a.to_torch(), t) # to_torch undoes the relayout
```

## Inspection

STensor deliberately does **not** print its contents — `repr`/`str` of any STensor
returns the bare word `Tensor`, and its storage prints `TensorStorage({})`. Do not
rely on `print(x)` for inspection; use the accessors below instead.

| Accessor | Kind | Returns |
|---|---|---|
| `.shape` | property | logical shape tuple (`()` if unset) |
| `.dim()` | method | number of modes (rank) |
| `.dtype` | property | logical `torch.dtype` (default `torch.float32`) |
| `.format` | property | the {class}`~scorch.TensorFormat` |
| `.values` | property | the flat 1-D tensor of stored values |
| `.index` | property | the sparsity structure (format + coordinate arrays) |
| `.has_index` | property | whether an index is attached |
| `.name` | property | the tensor's name (**raises if never set**) |

```python
import torch, scorch
from scorch.format import LevelType

A = scorch.from_csr(torch.tensor([[0., 2., 0.],
                                  [1., 0., 3.]]).to_sparse_csr(), "A")

print(A.shape)                       # (2, 3)
print(A.dim())                       # 2      (a method — there is no .ndim)
print(A.dtype)                       # torch.float32
print(A.has_index)                   # True
print(A.name)                        # A

# Format: str() gives the comma-joined notation; get_level_types() the enums.
print(str(A.format))                 # d,s
print(A.format.get_level_types())    # [LevelType.DENSE, LevelType.COMPRESSED]
assert A.format.get_level_types() == [LevelType.DENSE, LevelType.COMPRESSED]

# The raw payload (inspect BEFORE any to_torch, which may densify in place).
print(A.values.tolist())             # [2.0, 1.0, 3.0]
print([[t.tolist() for t in lvl]     # per-mode index arrays:
       for lvl in A.index.mode_indices])
# -> [[], [[0, 1, 3], [1, 0, 2]]]    ([], [crow, col]) for CSR
```

For a CSR tensor, `.index.mode_indices` is `[[], [crow, col]]`: the dense row mode
stores no coordinates, and the compressed column mode stores a `pos`/row-pointer
array plus a `crd`/column-index array. A COO tensor instead stores one coordinate
list per mode — `[[row], [col], …]`. The exact meaning of these arrays per level
type is covered in {doc}`the format system </user_guide/format_system>`.

:::{warning}
`.name` **asserts** that a name was set — accessing it on an unnamed tensor raises
`AssertionError`. The factory functions default the name to `"tensor"`, so
factory-built tensors are always safe; only a hand-built `STensor(...)` without a
name can trip this.
:::

:::{note}
STensor omits several PyTorch-style members you might reach for: there is no public
`.nnz` (use the private `._nnz()`), no `.ndim` (use `.dim()`), and no `.device`,
`.T`, `.size()`, or `.numel()`. There is also no `__getitem__` — you cannot index or
slice an STensor. Use `copy()` (not the stub `clone()`) to duplicate one.
:::

## Elementwise operators

Elementwise **addition** is implemented: `A + B` JIT-compiles an add kernel. `B` is
first reordered to match `A`'s mode order, and the result takes `A`'s format.

```python
import torch, scorch

A = scorch.from_torch(torch.rand(4, 4), "A")
B = scorch.from_torch(torch.rand(4, 4), "B")

C = A + B
assert torch.allclose(C.to_torch(),
                      A.to_torch(in_place=False) + B.to_torch(in_place=False),
                      atol=1e-3, rtol=1e-3)
```

:::{warning}
Elementwise **multiply** is not implemented — `A * B` raises `NotImplementedError`.
There is also no broadcasting, and the output format is taken from the left operand
rather than inferred. For contractions (matrix products, `einsum`), use the top-level
ops, not operators — see below.
:::

## Matmul is a function, not an operator

STensor has no `__matmul__`, so `A @ B` will **not** work. Matrix multiplication,
`einsum`, and the other contractions are top-level functions in the `scorch`
namespace:

```python
import torch, scorch

A = scorch.from_torch(torch.rand(64, 128).to_sparse_csr(), "A")
B = scorch.from_torch(torch.rand(128, 32), "B")

C = scorch.matmul(A, B)              # correct
# C = A @ B                          # WRONG: STensor defines no @ operator
```

Depending on the operand formats and shapes, an op may return either an STensor or a
dense `torch.Tensor`, and you can request a specific `output_format`. Those details —
and the reference-checked SpMV/SpMM/SDDMM/SpGEMM examples — live in
{doc}`operations </user_guide/operations>`. Across the SuiteSparse suite Scorch runs
**1.05–5.80× faster than PyTorch Sparse (CGO 2026)**.

(pytorch-interop-and-the-shim)=
## PyTorch interop and the shim

Scorch is designed to sit *next to* PyTorch, not replace it. Two mechanisms make the
interop seamless.

**1. The fall-through shim.** The `scorch` module defines a `__getattr__` that
forwards any name it does not define straight to `torch`:

```python
def __getattr__(name):
    return getattr(torch, name)
```

So `scorch.relu`, `scorch.nn`, `scorch.zeros`, `scorch.softmax`, and everything else
Scorch does not override *are* the real PyTorch objects:

```python
import torch, scorch

assert scorch.relu is torch.relu
assert scorch.nn is torch.nn
```

Scorch only shadows the handful of symbols it re-implements sparsely (its factories,
{func}`~scorch.matmul`, {func}`~scorch.einsum`, the sparse-NN ops, and so on); every
other attribute access lands on PyTorch.

**2. `import scorch as torch` as a drop-in.** Because of the shim, you can alias
Scorch as `torch` and an existing model keeps working — dense ops go to PyTorch
unchanged, and the ops Scorch specializes transparently take the sparse path. This is
exactly what the application tutorials do:

```python
import scorch as torch          # drop-in shim

x = torch.relu(torch.randn(8, 8))     # -> real torch.relu / torch.randn
A = torch.from_csr(some_csr, "A")     # -> Scorch's sparse factory
y = torch.matmul(A, x)                # -> Scorch's SpMM
```

**3. Bringing results back.** When you are done in Scorch and want a plain dense
`torch.Tensor` — to feed a `torch.nn` layer, compute a loss, or compare against a
reference — call `.to_torch()` on any STensor result:

```python
import torch, scorch

A = scorch.from_torch(torch.rand(4, 4), "A")
B = scorch.from_torch(torch.rand(4, 4), "B")

C = A + B                          # STensor
dense_C = C.to_torch()             # hand it back to PyTorch
loss = dense_C.sum()               # ordinary torch from here on
```

:::{tip}
The correctness convention used throughout these docs (and Scorch's own test suite)
is to verify a sparse result against a dense PyTorch reference with
`torch.allclose(result, reference, atol=1e-3, rtol=1e-3)`. When in doubt about an
operation, round-trip through `.to_torch()` and compare.
:::

## See also

- {doc}`The format system </user_guide/format_system>` — level types, the format
  notation, and how CSR/COO/DCSR map onto it.
- {doc}`Operations </user_guide/operations>` — `matmul`, `einsum`, `spmv`, output
  formats, and the reference-checked kernel examples.
- {doc}`API · Tensors </api/tensors>` — the full {class}`~scorch.STensor` and
  {class}`~scorch.TensorFormat` reference.
