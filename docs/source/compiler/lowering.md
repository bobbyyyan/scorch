# Lowering & iteration

Lowering is the stage where Scorch turns an abstract, order-free description of a
computation into a concrete loop nest. The input is **CIN** — Compiler Index
Notation, the index-notation AST that says *what* is computed (see
{doc}`/compiler/index_notation`). The output is **LLIR**, a typed, C++-shaped IR
that says *how* to loop, which coordinates to walk, and which positions to read.
The component that performs this translation is the `CINLowerer`, and its central
job is **iterator analysis**: reading each operand's format and emitting exactly
the loop machinery that format's storage requires.

This page explains the three ideas that make lowering work:

1. the per-`(tensor, mode)` **`ModeIterator`**, which knows how to walk one
   level of one tensor;
2. the **iteration lattice**, which *merges* several `ModeIterator`s when an
   index variable is shared by multiple operands; and
3. the **`Scheduler`**, which chooses the loop order (and inserts workspaces and
   tiling) before lowering ever runs.

Together they answer the load-bearing question: *given the formats of the
operands and the output, what loop nest do we emit?*

## Where lowering sits

```{mermaid}
flowchart LR
    CIN["CIN<br/>(index notation)"] --> SCH["Scheduler<br/>loop order + workspace"]
    SCH --> LOW["CINLowerer<br/>iterator analysis"]
    LOW --> LAT["IterationLattice<br/>merge logic"]
    LAT --> LLIR["LLIR<br/>typed loop nest"]
    LLIR --> CG["codegen → C++"]
```

The `Scheduler` runs first and rewrites the CIN so its `ForAll` nodes sit in the
chosen execution order (a bare `ForAll` in CIN carries *no* order — order is a
scheduling decision). The `CINLowerer` then walks that ordered CIN top-down; for
each `ForAll` it builds an `IterationLattice` over the tensors that use the loop
variable, and asks the lattice to materialize the loop(s). The resulting LLIR
feeds {doc}`codegen </compiler/codegen>`, which prints the C++ string.

## Iterator analysis: one level, one tensor

A tensor's {doc}`format </user_guide/format_system>` is a per-mode list of level
types — `dense` (`d`), `compressed` (`s`/`c`), `coordinate` (`o`), or the
reserved `singleton`. Each level type stores its coordinates differently, so each
demands a different loop shape. The `ModeIterator` (`compiler/iterator.py`) is
built once per `(tensor-access, index-var, level)` and, from that level's type,
derives the LLIR variables that drive iteration.

Think of a `ModeIterator` as answering three questions for one mode:

- **Where do coordinates come from?** (an arithmetic formula, or a `crd` array)
- **What bounds the loop?** (the full extent, or a `pos[...]` slice)
- **How is the flat storage position computed?** (`p = parent*size + i`, or a
  running position pointer)

### Dense — a counted loop with arithmetic position

A `dense` level stores no coordinates at all: the mode simply occupies its full
extent, and the position into the value array is pure arithmetic. The
`ModeIterator` emits a plain counted loop and computes the flat offset as

$$p_{\text{child}} = p_{\text{parent}} \cdot \text{size} + i$$

so for a dense inner mode of size `B1_size`, the position is
`pB1 = pB0 * B1_size + k`. The index variable *is* the loop variable; no `crd`
or `pos` array is touched. Conceptually:

```c++
for (int k = 0; k < B1_size; ++k) {
    int pB1 = pB0 * B1_size + k;   // arithmetic position, no coordinate array
    // ... use B_values[pB1] ...
}
```

### Compressed — a CSR pos/crd walk

A `compressed` level is the CSR pattern: a `pos` array (the row pointer, also
called `crow`) indexes into a `crd` array (the stored coordinates). Only nonzeros
are stored. The `ModeIterator` iterates a *position* variable over the slice
`pos[parent] .. pos[parent+1]` and reads the actual coordinate from `crd`:

```c++
for (int pA1 = A1_pos[pA0]; pA1 < A1_pos[pA0 + 1]; ++pA1) {
    int j = A1_crd[pA1];          // the stored column index
    // ... use A_values[pA1] at column j ...
}
```

This is the walk that visits *only* the nonzeros of a CSR row. The loop bound
depends on the parent position, which is exactly why the outer mode must be
resolved before the inner one.

### Coordinate — a flat coordinate-list walk

A `coordinate` level (COO) stores a flat `crd` list. At the root level a global
`pos[0], pos[1]` pair bounds the whole array; deeper coordinate levels reuse the
parent's position bound. The `ModeIterator` walks the position range and reads
`crd[p]` directly — like the compressed case, but without a per-parent row
pointer.

:::{note}
`singleton` is representable in the type system (`TensorFormat(["coordinate",
"singleton"])`) but has **no lowering branch** — `iterator.py` and
`iter_lattice.py` switch only on `dense` / `compressed` / `coordinate`. Treat
singleton as reserved. See {doc}`the format system </user_guide/format_system>`
for the full limitation.
:::

The `ModeIterator` also produces the initialization statements (`get_init_stmts`,
`get_iterator_end_init_stmts`) that declare and seed these position variables
before the loop, plus the begin/end bound expressions the lattice uses to build
loop conditions.

## The iteration lattice: co-iterating shared indices

A single `ModeIterator` handles one tensor. But a contraction shares index
variables *across* operands — in `C[i,k] = A[i,j] * B[j,k]`, the variable `j`
indexes both `A` and `B`. When the `CINLowerer` lowers a `ForAll`, it must
co-iterate every tensor that uses that loop variable. The **iteration lattice**
(`compiler/iter_lattice.py`, following TACO's *merge lattice*) is the machinery
that decides how.

The class docstring puts it precisely:

> The iteration lattice of an iteration domain contains an ordered set of
> lattice points, in decreasing order of the number of index variables they
> contain.

Each `LatticePoint` bundles the `ModeIterator`s that must be advanced together
for one sub-region of the iteration space, and emits the for/while conditions,
the iterator-advance statements, and the guarded body. `IterationLattice`
composes those points; `lower_ForAll` calls `get_lattice_loops()` to turn them
into the actual LLIR loop nest for that level. The shape of the merge depends
entirely on the level types of the co-iterated operands.

### Dense × dense — iterate the shared extent once

If every operand touching the index is `dense`, they all span the full extent,
so there is nothing to merge: emit a single counted loop and index each operand
arithmetically.

### Dense × sparse — drive by the sparse operand's coordinates

When one operand is sparse (`compressed`/`coordinate`) and the other is `dense`,
the loop is **driven by the sparse operand's stored coordinates**. You visit only
the positions where the sparse tensor actually has data, and use each stored
coordinate to index the dense operand arithmetically. This is the essence of
SpMM: the CSR walk over `A` picks the columns `j`, and `B[j, k]` is a direct
dense lookup — no wasted iterations over structural zeros.

### Sparse × sparse — merge the coordinate streams

When two sparse operands share an index, neither's coordinates are known ahead of
time, so their streams must be **merged** with a coordinated `while` walk that
advances both position pointers. The merge rule follows the arithmetic:

| Contraction | Merge | Why |
|---|---|---|
| multiply (`*`) | **intersection** | a product term is nonzero only where *both* operands are nonzero |
| add (`+`) | **union** | a sum term is nonzero where *either* operand is nonzero |

`gen_lattice_points` builds these via `union_lattice_points` /
`intersect_lattice_points`. Intersection collapses to the smaller,
single-operand-driven loop above whenever one side is dense (a dense operand is
nonzero *everywhere*, so intersecting with it is a no-op) — which is why
dense × sparse degenerates to "drive by the sparse coords."

:::{tip}
The one-line intuition: a **dense** level contributes "iterate the whole
extent," a **compressed/coordinate** level contributes "iterate only where I have
data," and the lattice combines those contributions — intersection for products,
union for sums — into the merge walk.
:::

### The output format shapes the write-back, too

The lattice also initializes and advances the **result** position variables, and
it does so differently per output level type. A `dense` result level gets an
arithmetic index (write straight to a computed offset); a `compressed` result
level *appends* to its `crd` array and bumps its `pos` pointer as coordinates are
discovered. This is why the format you choose for the output is not cosmetic — it
changes the emitted assembly of the write-back, not just the read side. For a
sparse output whose nonzero pattern isn't known in advance, the producer
accumulates into a **workspace** and the consumer drains it into the compressed
result (see {doc}`workspaces </compiler/workspaces>`).

The lattice hosts a few specialized fast paths as well: a vectorizable
scalar-accumulator loop for COO outputs, and `#pragma omp simd` tagging on
reduction loops.

## The Scheduler: choosing the loop order

Iterator analysis tells you *how* to walk a given loop order. The **`Scheduler`**
(`compiler/scheduler.py`) decides *which* loop order to walk in the first place —
and whether to insert a workspace or tile a loop — all before the `CINLowerer`
runs. A CIN `ForAll` deliberately carries no order; the Scheduler supplies it.

Three entry points:

`select_loop_order(cin)`
: Chooses the ordered `List[IndexVar]`. It seeds an initial order, runs a
  cost-model-driven `optimize_loop_order`, applies operand mode-order
  constraints, and — for sparse outputs with reductions — guarantees at least one
  free variable follows the last reduction variable (the lowerer requires the
  innermost loop to map to a result level, and workspace insertion needs trailing
  free vars).

`auto_schedule(cin)`
: The full transform. Roughly:

```python
loop_order = select_loop_order(cin)
cin = _rebuild_loop_nest(cin, loop_order)      # reorder the ForAlls
if should_insert_workspace(cin, loop_order):
    if (not dense_output) or will_tile:
        cin = insert_workspace(cin, allow_dense=True)   # add Where + Workspace
cin = _apply_tiling_heuristics(cin)            # tile-i / tile-k / ...
return cin
```

`apply_schedule(cin, schedule)`
: Applies an immutable, tuner-provided {class}`~scorch.Schedule`. An empty
  schedule delegates to `auto_schedule` exactly; otherwise `loop_order` is an
  exact logical permutation and the listed {class}`~scorch.TileSpec` objects
  replace implicit tiling. Tile width, outer-loop placement, unrolling,
  accumulator policy, and the selected parallel loop are all part of the
  schedule's cache identity.

For example, this emits the register-block geometry
`i -> k_out -> j -> k_in`, with a four-element stack accumulator and guarded
ragged tail:

```python
schedule = scorch.Schedule(
    loop_order=("i", "j", "k"),
    tiles=(
        scorch.TileSpec(
            "k",
            4,
            placement="child_of:i",
            accum="stack",
        ),
    ),
    tag="spmm-k4",
)
C = scorch.matmul(A, B, schedule=schedule)
```

An explicit schedule forces the generic JIT path, so it cannot be silently
ignored by a matching prebuilt kernel. Affine tiling supports dense domains such
as the SpMM row `i` and free dimension `k`; affine reduction tiling is rejected
until an accumulator can span outer reduction tiles. A sparse contraction tile
with `kind="panel", accum="direct"` lowers after iterator construction: it emits
a serial coordinate panel, two `std::lower_bound` calls per CSR row, and a fresh
parallel row loop per panel. This tile-j form requires sorted compressed
coordinates, matching the prebuilt kernel's requirement. Affine `i`/`k` tiles and
panel `j` can be composed into tiled-ijk loop geometry with independent ragged
tails. Adding a {class}`~scorch.RelayoutSpec` for the dense operand stages each
access into a reusable vector-backed buffer and redirects the selected logical
read using LLIR tensor-access provenance. Its `scope_var` is a logical loop
anchor: the panel variable realizes a `Jc × Nc` stage inside each panel, while
the tiled free variable realizes a `J × Nc` stage once at free-tile entry.

The affine free tile's `accum` field independently selects result lifetime.
`"direct"` updates the final dense result. `"stack"` retains the existing
row-local/register-sized workspace. `"heap"` creates a compact tile spanning
all dense result-prefix positions (`M × Nc` for rank-2 SpMM), initializes it at
free-tile entry, accumulates through all enclosed reduction panels, and copies it
to the result once at tile exit. Thus the compiler can express all four useful
staging/result crosses; for example, the handwritten-equivalent structure is:

```python
schedule = scorch.Schedule(
    loop_order=("i", "j", "k"),
    tiles=(
        scorch.TileSpec(
            "k", Nc, placement="outermost", accum="heap", unroll=False
        ),
        scorch.TileSpec(
            "j", Jc, placement="child_of:k_out",
            kind="panel", accum="direct",
        ),
    ),
    relayout=scorch.RelayoutSpec(
        operand="B", pack_var="k", strip_width=Nc, scope_var="k"
    ),
    parallel_loop="i",
)
```

The initial operand-staging lowering deliberately accepts only a structurally
compatible rank-2 CSR-by-dense contraction with a dense result, matching floating
dtypes, an outermost affine free-axis tile, and a serial panel surrounding the
parallel CSR row loop. Heap result tiles are more general and also cover dense
multi-index contractions such as TTM when the tiled free axis is the trailing
dense result level. Unsupported formats, access choices, lifetimes, and unsafe
parallel placements fail during schedule validation, before C++ generation.

The order matters because iteration cost is dominated by *which* variable sits
innermost and whether a reduction can stream into a small accumulator. Ordering a
reduction variable just above its trailing free variable, for example, lets the
producer fill a dense workspace row that the consumer then copies out — the
canonical SpMM schedule.

### The cost model

`optimize_loop_order` scores candidate orders with the calibrated constants in
`_CostModelConstants`:

```python
alpha = 2.975   beta = 0.1005   gamma = 43.55
c_insert = 85.34   c_sort = 1.741   c_trans = 40.61
rho = 0.0014   default_dim_size = 1024
```

Read them as knobs on a cost estimate:

- `alpha`, `beta`, `gamma` weight per-iteration work (the base cost of a loop
  level, how it scales, and a fixed overhead);
- `c_insert`, `c_sort`, `c_trans` price the workspace operations a schedule may
  incur — coordinate **insertion**, the final **sort** of a COO workspace, and a
  **transpose** when the output mode order forces one;
- `rho` is the assumed sparsity density (fraction of stored entries), so orders
  that iterate a sparse level pay proportionally less than iterating it densely;
- `default_dim_size` is the fallback extent when a dimension size isn't known at
  schedule time.

The model trades the cost of *introducing* a workspace (`c_insert` + `c_sort` +
possible `c_trans`) against the per-iteration savings a better order buys — which
is exactly the decision `should_insert_workspace` and the dense-output guard
encode.

:::{note}
This compile-time `Scheduler` is a **different mechanism** from the runtime
prebuilt tiling selector (the `-O` autotune ladder in `tiling.py`, described in
{doc}`autotuning </user_guide/autotuning>`). The Scheduler orders and tiles the
*generic JIT* loop nest; the tiling selector picks among *hand-written* prebuilt
SpMM kernels during dispatch, before the compiler is ever reached. The immutable
schedule API is the handoff boundary a tuner can use to select JIT loop geometry;
`schedule_from_tuner_choice(("tileijk", (Nc, Jc)), ...)` is an opt-in adapter from
the existing normalized tuner choice to the packed JIT schedule. The compiler-only
extended choice `(Nc, Jc, scope_var, accum)` exposes operand staging at either the
panel or free-axis tile and direct or heap-backed compact-result accumulation.
`compiler_schedule_search_space(...)` constructs the Cartesian product of those
four dimensions for an opt-in search. Neither helper replaces the production
selector nor changes its persistent cache, learned model, or default policy.
:::

## Worked example: formats → loop nest

Take the canonical SpMM `C[i,k] = A[i,j] * B[j,k]` with `A` in CSR (`"ds"`) and
`B` dense (`"dd"`), reaching the generic pipeline. Watch how the three level
types drive the emitted nest.

```python
import torch
import scorch

A_dense = torch.randn(128, 256)
A_dense[A_dense.abs() < 1.0] = 0.0                 # sparsify
A = scorch.from_csr(A_dense.to_sparse_csr(), "A")  # CSR, format "d,s"
B = scorch.from_torch(torch.randn(256, 64), "B")   # dense, format "d,d"

# Force the generic compiler and pin a dense output:
C = scorch.einsum("ik,kj->ij", A, B, format="dd")

ref = A_dense @ B.to_torch()
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

Under the hood:

1. **Schedule.** `select_loop_order` returns `[i, j, k]` — the reduction `j`
   before the trailing free `k`, so a workspace row can be filled. `auto_schedule`
   inserts a dense workspace `accum_c[k]` and splits the body into a producer/
   consumer `Where`.
2. **Iterator analysis per level:**
   - `i` indexes `A`'s outer `dense` row level and `C`'s dense row → a counted
     loop, arithmetic position.
   - `j` indexes `A`'s `compressed` column level and `B`'s `dense` row. This is
     the dense × sparse case: the lattice **drives the loop by `A`'s stored
     column coordinates** — a CSR `pos/crd` walk — and reads `j` back from
     `A1_crd` to index `B` arithmetically.
   - `k` indexes `B`'s and `C`'s `dense` inner levels → a counted loop with
     arithmetic offsets.
3. **Emitted nest** (conceptually):

```c++
#pragma omp parallel for
for (int i = 0; i < M; ++i) {
    float accum_c[K] = {0};
    for (int pA = A1_pos[i]; pA < A1_pos[i + 1]; ++pA) {   // compressed: CSR walk
        int j = A1_crd[pA];                                //   stored column
        for (int k = 0; k < K; ++k)                        // dense: counted loop
            accum_c[k] += A_values[pA] * B_values[j * K + k];
    }
    for (int k = 0; k < K; ++k)                            // consumer drains workspace
        C_vals[i * K + k] = accum_c[k];
}
```

Every structural choice traces back to a format: the middle loop is a `pos/crd`
walk *because* `A`'s column mode is `compressed`; the inner loop is a counted loop
with `j * K + k` arithmetic *because* `B`'s modes are `dense`; and the write-back
is a flat store *because* the output is `dense`. Change `A` to DCSR (`"ss"`) and
the `i` loop becomes a compressed walk too; ask for a sparse output and the
consumer appends to a `crd`/`pos` pair instead of storing to a flat array.

## See also

- {doc}`Index notation (CIN) </compiler/index_notation>` — the AST that lowering
  consumes, and how `einsum` builds it.
- {doc}`Codegen </compiler/codegen>` — how the LLIR loop nest becomes a C++
  `evaluate()` string, including OpenMP emission.
- {doc}`The format system </user_guide/format_system>` — level types, format
  notation, and the storage each iterator walks.
- {doc}`Workspaces </compiler/workspaces>` — dense vs COO-hashed accumulation
  buffers and the producer/consumer split.
