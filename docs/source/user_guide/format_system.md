# The format system

Every sparse tensor in Scorch carries a **format**: a declarative, per-mode
description of how its coordinates are physically stored. The format is the
central abstraction of the compiler — it is what the pipeline reads to decide
which loops to emit, which arrays to walk, and how to merge operands. Understand
the format system and you understand how Scorch turns `A @ B` into a specialized
C++ kernel.

This page explains what a format is, the four level types it is built from, the
notation for spelling familiar layouts (CSR, COO, DCSR, …), how to declare
formats, how they attach to tensors, how to request them as op outputs, and how
they drive iteration in the generated kernel.

## What a format is

Scorch describes a sparse tensor's physical layout as a **per-mode sequence of
level types** — the same model used by the TACO tensor-algebra compiler. A
rank-`N` tensor's format is an ordered list of `N` *level formats*, one per mode,
in storage order. Reading the list left to right tells you how you descend from
the outermost mode to the innermost, and how each mode's coordinates are stored.

- A **level type** answers one question about one mode: *are all coordinates of
  this mode stored (dense), or only the nonzero ones (compressed / coordinate /
  singleton)?*
- A {class}`~scorch.TensorFormat` is the ordered list of level types for the
  whole tensor.

The familiar names map onto this directly. A CSR matrix is "dense rows,
compressed columns" — `"ds"`. A COO matrix is "coordinate, coordinate" — `"oo"`.
A fully dense matrix is `"dd"`. Nothing more exotic than an ordered list of
per-mode storage choices is going on.

:::{tip}
The mental model: a format is a *word* over a small alphabet of level types, one
letter per dimension. `"ds"`, `"oo"`, `"ss"` are all two-letter words describing
two-dimensional tensors.
:::

## The four level types

`LevelType` (in `scorch.format`) is a four-member enum. Each member decides both
how a mode's coordinates are stored and how the compiler iterates them.

```python
class LevelType(Enum):
    DENSE = "d"
    COMPRESSED = "s"
    SINGLETON = "singleton"
    COORDINATE = "o"
```

| Level type | Storage of this mode's coordinates | Iteration semantics |
|---|---|---|
| `DENSE` (`d`) | No coordinates stored. The mode occupies its full extent; the position is computed by arithmetic (`parent_pos * size + i`). Only the mode size is needed. | Iterate **all** indices `0 .. size-1` — a plain counted `for` loop. |
| `COMPRESSED` (`s`) | A segmented CSR-style pair: a `pos` array (a.k.a. `crow`) indexing into a `crd` (coordinate) array. Only nonzeros are stored; `pos` gives, per parent coordinate, the `[begin, end)` slice of `crd`. | Iterate **only stored coordinates** in the slice `pos[parent] .. pos[parent+1]`, reading each from `crd[p]`. |
| `COORDINATE` (`o`) | A flat COO-style `crd` array. At the root level a global `pos[0], pos[1]` bounds the list; deeper levels reuse the parent's position bound. | Iterate stored coordinates directly from `crd` over the position range. |
| `SINGLETON` | Intended for a mode that stores exactly one coordinate per parent — the classic COO tail that follows a coordinate head. | Declared in the type system but **not yet lowered**. See [Limitations](#limitations). |

The distinction that matters everywhere downstream is **dense vs. everything
else**: a dense level means "visit the whole extent by arithmetic," while a
compressed or coordinate level means "visit only where I have data, by walking a
`pos`/`crd` array." That single split is what [drives iteration](#how-level-types-drive-iteration)
in the generated code.

## String aliases

You rarely write `LevelType.COMPRESSED` by hand — you write a short string alias.
Scorch's alias table accepts several spellings per level type:

| Level type | Accepted string aliases |
|---|---|
| `DENSE` | `"dense"`, `"d"` |
| `COMPRESSED` | `"compressed"`, `"sparse"`, `"c"`, `"s"` |
| `COORDINATE` | `"coordinate"`, `"coord"`, `"o"` |
| `SINGLETON` | `"singleton"`, `"single"` — **no single-character alias** |

Two facts to internalize:

- **`s` and `c` are synonyms.** Both mean `COMPRESSED`; so does `"sparse"`. There
  is no CSR-vs-something distinction hiding behind the two letters — they yield
  the identical level type.
- **Singleton has no single-letter alias.** `DENSE`/`COMPRESSED`/`COORDINATE` all
  have one-character forms (`d`/`s`(or `c`)/`o`), which is what makes the compact
  string notation possible. `singleton` must always be spelled out, which
  matters for how you construct it (see [Declaring formats](#declaring-formats)).

## Familiar formats, in notation

This is the table to keep at hand. The **String** column is the compact
char-per-mode form you pass to `TensorFormat("...")` or an op's
`output_format="..."`; the **List form** is the equivalent long-form spelling.

| String | List form | Familiar name | Storage meaning |
|---|---|---|---|
| `"d"` | `["dense"]` | Dense vector | 1-D dense array — all elements. |
| `"s"` / `"c"` | `["compressed"]` | Sparse vector | `pos`/`crd` pair; only nonzeros. |
| `"dd"` | `["dense", "dense"]` | Dense matrix | Row-major dense 2-D. `is_dense()` is `True`. |
| `"ds"` | `["dense", "compressed"]` | **CSR** | Dense rows + compressed columns (`crow`/`pos` + `col`/`crd` + values). What `from_csr` and `from_torch(csr)` produce. |
| `"oo"` | `["coordinate", "coordinate"]` | **COO** | Two coordinate lists (row, col) + values. What {func}`~scorch.from_coo` and `from_torch(coo)` produce. |
| `"ss"` | `["compressed", "compressed"]` | **DCSR** | Doubly-compressed CSR: only nonempty rows *and* their nonzeros are stored. |
| `"sd"` | `["compressed", "dense"]` | CSC-like | Compressed outer + dense inner. |
| `"d"*N` | `["dense"]*N` | Dense rank-N | Fully dense N-D. What `from_torch(dense)` produces. |
| `"o"*N` | `["coordinate"]*N` | COO rank-N | N coordinate lists. |
| `["coordinate", "singleton"]` | — | COO (TACO-canonical) | Coordinate head + singleton tail. *Type-representable but not lowered.* |

A few clarifications:

- **Scorch's COO is `"oo"`** — coordinate on *every* mode — not the
  TACO-canonical coordinate+singleton pair. `from_coo` / `from_torch` build the
  `"oo"` form, and that is the supported COO in this codebase.
- **CSC is not a distinct level type.** True column-major storage is CSR of the
  transpose: store the transposed matrix as `"ds"`, or use `"sd"` together with a
  swapped `mode_order` (see [How formats attach to tensors](#how-formats-attach-to-tensors)).
  There is no dedicated CSC constructor.
- **Blocked / BCSR layouts are out of scope for the notation.** `LevelType` has
  no block member, so a block format is not expressible as a word over this
  alphabet. Any block-level optimization happens in the scheduler and codegen,
  not in the format.

## Declaring formats

{class}`~scorch.TensorFormat` accepts four input shapes. All four appear here so
you can see how they line up:

```python
from scorch.format import TensorFormat, LevelType, LevelFormat

# 1. A bare string — split one character per mode.
csr = TensorFormat("ds")

# 2. A list of string aliases (long-form allowed).
csr2 = TensorFormat(["dense", "compressed"])

# 3. A list of LevelFormat objects.
csr3 = TensorFormat([LevelFormat(LevelType.DENSE),
                     LevelFormat(LevelType.COMPRESSED)])

# 4. A single LevelFormat (a one-mode format).
vec = TensorFormat(LevelFormat("compressed"))

# All three describe CSR. Compare their level types to confirm:
target = [LevelType.DENSE, LevelType.COMPRESSED]
assert csr.get_level_types() == target
assert csr2.get_level_types() == target
assert csr3.get_level_types() == target

assert csr.get_order() == 2
assert not csr.is_dense()
assert TensorFormat("dd").is_dense()
```

`TensorFormat` exposes `get_level_formats()`, `get_level_types()` (the per-mode
`LevelType` list), `get_order()` (the rank), and `is_dense()` (`True` only when
*every* mode is dense — the flag ops use to pick fully-dense fast paths).

`TensorFormat` and `LevelFormat` are frozen structural values, so separately
constructed equivalent formats compare equal and have the same hash. The parser
also accepts the comma-delimited display form:

```python
assert TensorFormat("ds") == TensorFormat(["dense", "compressed"])
assert TensorFormat("d,s") == TensorFormat("ds")
assert TensorFormat("singleton").get_order() == 1
```

Compact strings use one character per mode, while a recognized long alias is one
mode. `get_level_formats()` returns an immutable tuple; callers cannot mutate the
format through a live internal list. Use `serialize()` for canonical metadata or
cache-key serialization.

### LevelFormat

`LevelFormat` is one mode's format. Its `mode` argument accepts either a
`LevelType` or any string alias (the full alias table, including `"singleton"`),
and it carries an optional `bit_width` reserved for index-width optimization —
stored but not exercised on the mainline path.

```python
str(LevelFormat(LevelType.DENSE))   # "d"
str(LevelFormat("compressed"))      # "s"
```

## How formats attach to tensors

You seldom build a `TensorFormat` by hand — the `STensor` factory methods assign
one for you based on the source layout. Each factory produces a fixed format:

| Factory | Input | Format assigned |
|---|---|---|
| {func}`~scorch.from_torch` | dense `torch.Tensor` | all-`DENSE` of rank `tensor.dim()` (`"dd…d"`) |
| {func}`~scorch.from_torch` | `torch.sparse_coo` tensor | all-`COORDINATE` (`"oo…"`) |
| {func}`~scorch.from_torch` | `torch.sparse_csr` tensor | `[DENSE, COMPRESSED]` — CSR |
| `from_csr` | 2-D torch CSR | `[DENSE, COMPRESSED]` — CSR |
| {func}`~scorch.from_coo` | torch COO, or raw `indices`/`values`/`shape` | `[COORDINATE]*rank` (`"oo…"`) |
| `from_components` / `STensor.from_components` | explicit validated shape/format/index/value components | caller-specified |

```python
import torch
import scorch

A_csr = torch.randn(4, 4).to_sparse_csr()
A = scorch.from_csr(A_csr, "A")
print(str(A.format))               # "d,s"
print(A.format.get_level_types())  # [LevelType.DENSE, LevelType.COMPRESSED]
print(A.format.is_dense())         # False
```

Read a tensor's format back through the `format` property:
`st.format` is a `TensorFormat`, `st.format.get_level_types()` is the per-mode
enum list, and `str(st.format)` is the human-readable spelling.

### mode_order

{func}`~scorch.from_torch` takes an optional `mode_order`: a permutation applied
via `tensor.permute(*mode_order)` before storage, with the inverse applied on
`to_torch`. This is how you obtain column-major / CSC-of-transpose storage
without introducing a new level type — you permute the modes, then pick the
level types you want for the permuted layout.

For converting an *existing* STensor between formats, use `to_sparse(fmt)` (re-lay
out into the requested format; defaults to a compressed form when `fmt` is
`None`) and `to_dense()` (materialize to all-dense). See
{doc}`/user_guide/sparse_tensors` for the tensor lifecycle in full.

## Requesting output formats in ops

The generic ops let you pin the format of the result. `matmul` honors a `format=`
or `output_format=` keyword; `spmv` and `matmul_wksp` take `output_format=`
directly. Anything that is not already a `TensorFormat` is run through
`parse_format`, so a string (`"ds"`), a list (`["dense", "compressed"]`), or a
`TensorFormat` all work.

The defaults encode the library's conventions:

- **SpMV → `"d"`** (a dense result vector).
- **`matmul` → `"ds"`** (a CSR result).

```python
import torch
import scorch

A_dense = torch.randn(128, 256)
A_dense[A_dense.abs() < 1.0] = 0.0                    # sparsify
A = scorch.from_csr(A_dense.to_sparse_csr(), "A")     # CSR STensor
B_dense = torch.randn(256, 64)
B = scorch.from_torch(B_dense, "B")                   # dense STensor

C = scorch.matmul(A, B)                    # SpMM; dense product
C_dense = scorch.matmul(A, B, format="dd") # request a dense output explicitly
C_dense2 = scorch.matmul(A, B, output_format=scorch.TensorFormat("dd"))

# The product of a sparse matrix and a dense one is dense, so matmul hands
# back a torch.Tensor directly (drop-in with PyTorch). Verify it:
ref = A_dense @ B_dense
assert torch.allclose(C, ref, atol=1e-3, rtol=1e-3)
assert torch.allclose(C_dense, ref, atol=1e-3, rtol=1e-3)
```

:::{note}
`matmul` returns a plain `torch.Tensor` whenever the result is dense — a
sparse-times-dense product like the SpMM above densifies, so you get a
`torch.Tensor` you can drop straight into surrounding PyTorch code. When the
result is genuinely sparse, `matmul` returns an {class}`~scorch.STensor`; call
`.to_torch()` on it to materialize a dense `torch.Tensor`. See
{doc}`/user_guide/sparse_tensors` for the STensor interop surface.
:::

When you *don't* pin a format, Scorch infers the result format from the operand
formats — walking the contraction and marking each output level `DENSE` or
sparse (compressed/coordinate) accordingly.

:::{warning}
`parse_format` is the one canonical parser used by `TensorFormat`, layouts, and
operations. It accepts `singleton` as a structural format token, but runtime
storage and compiler specifications reject singleton levels with a Scorch domain
exception because lowering does not implement them yet.
:::

## How level types drive iteration

The point of a format is that it *generates code*. The compiler builds one
`ModeIterator` per `(tensor, mode)`, and its behavior is a switch on the level
type:

- **DENSE** → a full counted loop `for (i = 0; i < size; ++i)`, with the flat
  position computed arithmetically (`p = p_parent * size + i`). No coordinate
  array is touched.
- **COMPRESSED** → `for (p = pos[parent]; p < pos[parent+1]; ++p)`, reading the
  coordinate `crd[p]`. This is exactly the CSR row-pointer walk.
- **COORDINATE** → a loop over the coordinate list within the position bound,
  reading `crd[p]` directly.

When an index variable is shared by several operands (a contraction), the
iteration lattice composes these iterators:

- Two **dense** operands over the same index → iterate the shared full extent
  once.
- A **dense** and a **sparse** operand → drive the loop by the sparse operand's
  stored coordinates, indexing the dense operand arithmetically. You only visit
  where the sparse tensor has data.
- Two **sparse** operands → **merge** their coordinate streams: an intersection
  walk for multiply-style contractions, a union walk for add-style ones,
  advancing the position pointers in lockstep.

The result's format matters too: a dense output level advances its write position
by arithmetic, while a compressed output level appends to `crd` and bumps `pos`.
This is why the output format you request changes the *write-back* code, not just
the read.

The one-line takeaway: **dense levels mean "iterate everything by arithmetic";
compressed and coordinate levels mean "iterate only stored nonzeros via `pos`/`crd`
arrays" — and the format you pick for each operand and for the output is exactly
what the compiler reads to build the loop nest.** The full mechanism is covered
in {doc}`/compiler/lowering`.

## Limitations

The format system is deliberately small. Know these edges:

Singleton is declared but not usable end-to-end
: `LevelType.SINGLETON` exists and `TensorFormat(["singleton"])` constructs, but
  validated runtime storage and compiler specifications reject it because the
  lowering path supports only `DENSE` / `COMPRESSED` / `COORDINATE`. Treat it as
  a reserved structural token.

`s` and `c` are identical
: Both map to `COMPRESSED`. Don't read a semantic difference into the two
  letters.

One parser, two string spellings
: Compact strings such as `"ds"` are read one character per mode, while a known
  long alias such as `"dense"` or `"singleton"` is one mode. Comma-delimited
  canonical strings such as `"d,s"` are accepted too.

`__str__` inserts commas
: `str(TensorFormat("ds"))` is `"d,s"`, not `"ds"`. Both spellings parse to the
  same structural value, so `TensorFormat(str(TensorFormat("ds")))` round-trips.

Fill value is always 0.0
: The implicit value of an unstored entry is fixed at `0.0`. There is no
  non-zero fill and no "sparse ones."

No block / BCSR level type
: Blocked layouts are not expressible in the format alphabet.

## See also

- {doc}`/user_guide/sparse_tensors` — building, inspecting, and converting
  STensors, where formats come from and where they go.
- {doc}`/api/formats` — the `TensorFormat`, `LevelFormat`, and `LevelType` API
  reference.
- {doc}`/compiler/lowering` — how the compiler turns level types into the loop
  nest of a generated kernel.
