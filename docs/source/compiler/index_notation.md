# Compiler Index Notation (CIN)

Compiler Index Notation is Scorch's highest-level intermediate representation: a
small AST DSL that describes *what* a sparse computation produces in
index-notation form, without yet committing to *how* it loops. It is the first
IR the compiler builds for any operation that misses the prebuilt fast paths,
and every later stage — scheduling, lowering, codegen — is a transform on a CIN
tree.

CIN mirrors TACO's *concrete index notation*. If you can write a contraction as
$C_{ik} = \sum_j A_{ij} B_{jk}$, you can express it directly as a handful of CIN
nodes. This page documents each node type and shows how they compose into the
canonical matmul CIN.

For where CIN sits in the whole flow, see
{doc}`the compiler pipeline </compiler/pipeline>`. For how CIN becomes loops and
C++, see {doc}`lowering </compiler/lowering>`.

## Why an index-notation IR

A sparse loop nest is entangled with the *format* of its operands: iterating a
CSR row is a position-array walk, iterating a dense row is a counter, and
co-iterating two compressed operands is a merge. Writing all of that by hand,
per operation and per format combination, does not scale.

CIN separates the two concerns. It records the *mathematical* statement —
indices, tensors, products, reductions — as a format-agnostic tree. The
{doc}`Scheduler and lowerer </compiler/lowering>` then bind that tree to a
concrete loop order and to each operand's format sparsity. One CIN description
lowers to many kernels, one per operand/output-format combination.

## Node taxonomy

Every CIN node derives from a common base and falls into one of two families:
**expressions** (things with a value, like a tensor access or a product) and
**statements** (things that assign or loop). The nodes you will meet most often:

| Node | Role |
|------|------|
| `IndexVar` | A loop/index variable — `i`, `j`, `k`. |
| `TensorVar` | A named tensor operand or result, with a shape and a format. |
| `TensorAccess` | An indexed access into a `TensorVar`, e.g. `A[i, j]`. |
| `TensorAssign` | The assignment statement `A[i,j] = expr` (or a compound `+=`). |
| `ForAll` | Binds an `IndexVar` over its range and runs a nested statement. |
| `Workspace` | A `TensorVar` subclass used as a reduction accumulation buffer. |
| `Where` | A producer/consumer split around a workspace. |

Arithmetic between accesses (`A[i,j] * B[j,k]`) builds `BinaryOp` expression
nodes; a fused epilogue (bias, ReLU, …) is carried as a `PostOps` attachment.
Those are supporting cast — the seven above are what you compose to describe an
operation.

### `IndexVar` — a loop variable

An `IndexVar` is a single index letter that ranges over one mode of one or more
tensors. In $C_{ik} = \sum_j A_{ij} B_{jk}$ the index variables are `i`, `j`,
and `k`. An `IndexVar` on its own does not say *when* it runs — a `ForAll`
supplies the binding, and the {doc}`Scheduler </compiler/lowering>` chooses the
order.

Index variables split into two kinds, computed automatically from where they
appear:

Free variable
: An index that appears on the **result** tensor. `i` and `k` are free — they
  index `C[i,k]`, so the output has one entry per `(i, k)` pair.

Reduction variable
: An index that appears only on **inputs**, never on the result. `j` is a
  reduction variable: the compiler must sum over it to collapse `A[i,j]*B[j,k]`
  down to `C[i,k]`.

This free-versus-reduction classification is what tells the compiler a summation
is required, and it drives workspace insertion (below). Index variables also
carry tiling state — whether they have been split into an outer/inner pair — but
that is populated later by the Scheduler, not by the CIN you write.

### `TensorVar` — a named tensor

A `TensorVar` is a declared tensor: a name, a shape, a `dtype`, a mode order,
and — crucially — a {class}`~scorch.TensorFormat`. The format is what makes CIN
sparse-aware: the *same* index-notation statement over a CSR `A` versus a dense
`A` describes the same math but lowers to different loops.

Indexing a `TensorVar` produces a `TensorAccess`:

```python
Aij = A[i, j]        # TensorVar.__getitem__ -> TensorAccess
```

and assigning into an indexed `TensorVar` produces the assignment that becomes
the root of the computation:

```python
C[i, k] = A[i, j] * B[j, k]   # TensorVar.__setitem__ records the assignment
```

### `TensorAccess` — where format meets index notation

A `TensorAccess` pairs a `TensorVar` with the ordered list of `IndexVar`s used
to index it: `A[i, j]`, `B[j, k]`, `C[i, k]`. This is the node where sparsity
becomes concrete — for each index position it knows the corresponding **level
type** from the tensor's format (dense, compressed, coordinate, singleton), and
therefore how that mode will be iterated. The lowerer reads exactly this to
decide whether a level is a counter loop, a position-array walk, or a merge.

### `TensorAssign` — the compute statement

`TensorAssign` is the leaf statement: a `TensorAccess` on the left, an
expression on the right, and an operation. A plain assignment writes; a compound
form accumulates:

```python
TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD)   # C[i,k] += ...
```

The `op` field is what expresses the reduction: with `op=Operation.ADD`, every
value of the reduction variable `j` accumulates into the same result location.
This is the innermost work of the kernel, and it lowers almost directly to an
LLIR `Assign` with a `+=` — see {doc}`lowering </compiler/lowering>`.

### `ForAll` — bind an index over its range

`ForAll` binds one `IndexVar` and runs a nested statement for each value of that
variable's range. Nesting `ForAll`s builds the loop structure:

```python
ForAll(i, ForAll(j, ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k],
                                            op=Operation.ADD))))
```

:::{important}
A `ForAll` does **not** fix an execution order. It declares that a variable is
iterated, but the actual loop ordering — and any tiling — is chosen later by the
Scheduler's cost model. The CIN you build is a *set* of bound variables, not a
committed loop nest. This is why the same CIN can be reordered freely before
lowering.
:::

### `Workspace` — a reduction accumulation buffer

A `Workspace` is a `TensorVar` subclass that serves as scratch storage for a
reduction. It is what lets Scorch sum over an inner variable while producing an
output of a different shape or format. Workspaces come in two flavours, and the
distinction matters:

Dense workspace
: Format `"d…d"` — a flat array indexed directly by the free variables that
  remain after the reduction. In matmul, `accum_c[k]` is a length-`K` scratch
  row: the `j` loop accumulates products into it, then the result row is copied
  out. Used for **dense outputs**.

Sparse / COO-hashed workspace
: Format `"o…o"` (coordinate) — a hash-of-coordinates accumulator, backed at the
  C++ level by a `coo_workspace` template. Used for **sparse outputs**, where
  you do not know in advance which output coordinates will be non-zero: the
  producer inserts and accumulates by coordinate, and the consumer sorts and
  emits the compressed result.

Workspaces are not written by hand. The Scheduler inserts one when a reduction
is followed by free variables in the loop order; the mechanics live in
{doc}`workspaces </compiler/workspaces>`, which covers when a workspace is
created, which flavour is chosen, and how it is drained.

### `Where` — producer / consumer split

A `Where` node ties a workspace together with the two loops that use it. Its
`producer` computes into the workspace; its `consumer` then reads the finished
workspace and writes the real result:

```python
Where(
    producer = ForAll(j, ForAll(k, TensorAssign(accum_c[k], A[i, j] * B[j, k],
                                                 op=Operation.ADD))),
    consumer = ForAll(k, TensorAssign(C[i, k], accum_c[k])),
)
```

`Where` is how "reduce into a buffer, then emit" is expressed as a single tree
node. The producer fills `accum_c`; the consumer drains it into `C`. When the
lowerer reaches a `Where` it emits the two phases in sequence, and this split is
also the seam Scorch uses for its two-phase parallel and scalar-accumulator
codegen paths.

## The canonical example: matmul in CIN

Putting the pieces together, here is the CIN for
$C_{ik} = \sum_j A_{ij} B_{jk}$ — the shape produced by
`scorch.einsum("ik,kj->ij", A, B)`. Before scheduling, it is a bare accumulate:

```python
# i, j, k are IndexVars; A, B, C are TensorVars.
# free = {i, k}  (index the result C);  reduction = {j}  (inputs only).

ForAll(i,
    ForAll(j,
        ForAll(k,
            TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD))))
```

The Scheduler then chooses a loop order and, because the reduction variable `j`
is followed by the free variable `k`, inserts a dense `Workspace` and wraps the
body in a `Where`:

```python
ForAll(i,
    Where(
        producer = ForAll(j,
                     ForAll(k,
                       TensorAssign(accum_c[k], A[i, j] * B[j, k],
                                    op=Operation.ADD))),
        consumer = ForAll(k,
                     TensorAssign(C[i, k], accum_c[k]))))
```

Reading it top to bottom: for each row `i`, the producer accumulates the full
sum over `j` into the length-`K` dense row `accum_c`, then the consumer copies
that finished row into the output `C`. Every node on this page appears here —
`IndexVar`s (`i`, `j`, `k`), `TensorVar`s (`A`, `B`, `C`), the `Workspace`
`accum_c`, `TensorAccess`es, `TensorAssign`s, `ForAll`s, and the `Where` that
joins the two phases.

## How the CIN gets built

You rarely construct CIN by hand. The {func}`~scorch.einsum` front end assembles
it from the einsum string:

```python
import torch
import scorch

A = torch.randn(128, 256)
A[A.abs() < 1.0] = 0.0                 # make it sparse
As = scorch.from_torch(A.to_sparse_csr(), "A")   # CSR STensor, format "ds"
Bs = scorch.from_torch(torch.randn(256, 64), "B")

# Forces the generic compiler path and a chosen output format.
C = scorch.einsum("ik,kj->ij", As, Bs, format="dd")

ref = A @ Bs.to_torch()
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

Under the hood `einsum` parses the expression into index letters, builds one
`IndexVar` per unique letter and one `TensorVar` per operand, classifies free
versus reduction variables, infers an output format (unless you pass `format=`),
assembles the right-hand side as a product of `TensorAccess`es, assigns it into
the result `TensorVar`, and wraps the whole thing in nested `ForAll`s. The
result is exactly the pre-scheduling tree shown above, ready to hand to the
Scheduler.

:::{tip}
Whether a specific `matmul`/`einsum` call goes through this generic CIN pipeline
or hits a hand-written **prebuilt kernel** depends on dispatch — see
{doc}`the compiler pipeline </compiler/pipeline>`. Only cases that miss the
prebuilt fast paths build a CIN.
:::

## See also

- {doc}`Compiler pipeline </compiler/pipeline>` — where CIN sits in the full
  dispatch-to-machine-code flow.
- {doc}`Lowering </compiler/lowering>` — how CIN is scheduled and lowered
  through the merge lattice into loops and C++.
- {doc}`Workspaces </compiler/workspaces>` — dense versus COO-hashed
  accumulation buffers, and when the Scheduler inserts them.
