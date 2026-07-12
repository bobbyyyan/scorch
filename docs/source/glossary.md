# Glossary

Vocabulary you will meet across the Scorch docs, source, and papers. Terms are
grouped by area — the data model and format system, the compiler pipeline, the
sparse operations, and the autotuner — and alphabetized within each group. Each
entry links to the page where the concept is treated in depth.

## Data model and formats

:::{glossary}
CSC
: Compressed Sparse Column — dense columns, compressed rows. Scorch has **no
  dedicated CSC level type**; you obtain CSC behavior by storing the transpose in
  CSR (`"ds"` over swapped modes) or by using the `"sd"` layout together with a
  swapped {term}`mode order`. See {doc}`the format system </user_guide/format_system>`.

COO
: Coordinate format — every mode is a {term}`coordinate` level, notation `"oo"`.
  A flat list of `(row, col)` index pairs plus values, storing only nonzeros.
  This is what {func}`~scorch.from_coo` and `scorch.from_torch` (on a
  `torch.sparse_coo` tensor) produce.

CSR
: Compressed Sparse Row — dense rows, compressed columns, notation `"ds"`. Stored
  as a `crow`/`pos` row-pointer array indexing into a `crd` column-index array plus
  values. The default sparse-matrix format; produced by `scorch.from_csr` and by
  `scorch.from_torch` on a `torch.sparse_csr` tensor.

DCSR
: Doubly-Compressed Sparse Row — both modes compressed, notation `"ss"`. Only the
  nonempty rows *and* their nonzeros are stored, saving space on matrices with many
  all-zero rows.

compressed
: A {term}`level type` (`s` or `c`, both synonyms for `LevelType.COMPRESSED`) that
  stores only the nonzeros of a mode as a CSR-style `pos`/`crd` pair. Iteration walks
  only the stored coordinates in each parent's `[pos[p], pos[p+1])` slice.

coordinate
: A {term}`level type` (`o`, `LevelType.COORDINATE`) that stores a mode's nonzeros as
  a flat COO-style coordinate list. Iteration reads coordinates directly from the
  `crd` array over the position range.

dense
: A {term}`level type` (`d`, `LevelType.DENSE`) that stores no coordinates: the mode
  occupies its full extent and positions are computed arithmetically
  (`parent_pos * size + i`). Iteration is a plain counted loop over `0 .. size-1`.

fill value
: The implicit value of every element not explicitly stored. In Scorch this is
  **always `0.0`** (`TensorFormat._fill_value`); non-zero fill is not yet supported.

format notation
: Also called a **format string**. The per-mode, one-character-per-dimension string
  describing a tensor's physical layout: `d`=dense, `s`/`c`=compressed, `o`=coordinate.
  E.g. `"ds"`=CSR, `"oo"`=COO, `"dd"`=dense. A bare string is split one character per
  mode, so multi-letter aliases (like `"singleton"`) must be passed in **list** form,
  e.g. `["coordinate", "singleton"]`. See {doc}`the format system </user_guide/format_system>`.

level type
: The layout of a single tensor mode — one of {term}`dense`, {term}`compressed`,
  {term}`coordinate`, or {term}`singleton` (`LevelType` in `scorch.format`). A
  {class}`~scorch.TensorFormat` is an ordered list of level types, one per mode, and
  is the central abstraction the compiler keys code generation off.

mode order
: The permutation mapping logical tensor dimensions to physical storage order. Passing
  `mode_order` to `scorch.from_torch` permutes the tensor before storage (inverse
  applied on `to_torch`) — this is how you get column-major / CSC-of-transpose storage
  without introducing a new level type.

nnz
: Number of nonzeros — the count of explicitly stored (non-{term}`fill value`)
  elements. Only these are stored and iterated in compressed/coordinate levels.

singleton
: A {term}`level type` (`LevelType.SINGLETON`, spelled out — it has **no
  single-character alias**) intended to store exactly one coordinate per parent (the
  COO-tail companion to a coordinate head). It is type-representable but **not yet
  lowered**: the op layer's `parse_format` rejects it and there is no codegen branch.
  Treat it as reserved.

sparsity
: The fraction of a tensor's elements that are {term}`fill value` (zero); equivalently
  `1 - nnz / total_elements`. High sparsity is what makes the compressed formats and
  sparse kernels pay off.

STensor
: Scorch's user-facing sparse tensor ({class}`~scorch.STensor`), wrapping index and
  value storage plus a {class}`~scorch.TensorFormat`. Built with
  {func}`~scorch.from_torch`, {func}`~scorch.from_coo`, or `scorch.from_csr`, and
  converted back with `to_torch`. See {doc}`sparse tensors </user_guide/sparse_tensors>`.

TensorFormat
: The whole-tensor format ({class}`~scorch.TensorFormat`): an ordered list of
  {term}`level type`s, one per mode. Constructible from a {term}`format notation`
  string (`TensorFormat("ds")`), a list of aliases, or `LevelFormat` objects; its
  `__str__` is comma-joined (`"d,s"`). See {doc}`the format system </user_guide/format_system>`.
:::

## Compiler pipeline

:::{glossary}
CIN
: Compiler Index Notation (`compiler/cin.py`) — the compiler's highest IR, an
  index-notation AST DSL (`ForAll` / `Where` / `TensorAssign` / `IndexVar` /
  `TensorVar` / `Workspace`) describing *what* is computed, in the TACO lineage. See
  {doc}`index notation </compiler/index_notation>`.

codegen
: The stage that emits a single C++ source string from {term}`LLIR`
  (`LLIRLowerer.lower_llir` in `compiler/codegen.py`), including all OpenMP pragma
  emission. See {doc}`code generation </compiler/codegen>`.

generated kernel
: A kernel produced by the JIT compiler pipeline (CIN → LLIR → codegen → compile) for
  a specific operation × operand-format × output-format combination, as opposed to a
  {term}`prebuilt kernel`. Also called a *JIT kernel*.

iteration lattice
: The merge-lattice machinery (`IterationLattice` / `LatticePoint` in
  `compiler/iter_lattice.py`) that turns format sparsity into concrete loop code:
  co-iterating two dense operands, driving a loop by one sparse operand's stored
  coordinates, or merging two sparse coordinate streams (intersection for
  multiply-style contractions, union for add-style). See {doc}`lowering </compiler/lowering>`.

JIT
: Just-In-Time compilation — the first call to a generic op compiles a specialized
  C++ `evaluate()` function (`-O3 -march=native -ffast-math -funroll-loops -fopenmp`)
  and caches it to a `.so`; subsequent calls, even across process restarts, load the
  cached library from disk. See {doc}`the compiler pipeline </compiler/pipeline>`.

LLIR
: Low-Level IR (`compiler/llir.py`) — a typed, C++-shaped IR of loops, conditionals,
  and assignments that maps roughly 1:1 to emitted C++. `ForLoop` nodes carry the
  OpenMP parallelization policy. See {doc}`lowering </compiler/lowering>`.

prebuilt kernel
: A hand-written, hand-optimized C++ kernel compiled into the native `scorch_ops`
  extension (`src/scorch/csrc/spmm.h`, `src/scorch/csrc/kernels.h`) and dispatched
  via `resolve_prebuilt_matmul` before the JIT pipeline is ever reached — e.g.
  `spmm_csr_float_v2` for CSR × dense SpMM. Contrast with
  {term}`generated kernel`.

scheduler
: The compile-time pass (`compiler/scheduler.py`) that chooses loop order
  (`select_loop_order`), inserts {term}`workspace`s, and applies tiling heuristics
  (`auto_schedule`), driven by a calibrated cost model. Distinct from the runtime
  {term}`autotune level` selector, which picks among prebuilt kernels.

workspace
: A temporary accumulation buffer for reductions (`Workspace` in CIN). A **dense**
  workspace is a flat array indexed by the free variables (used for dense SpMM output);
  a **sparse / COO-hashed** workspace accumulates by coordinate for sparse outputs
  whose nonzero pattern is unknown in advance. See {doc}`workspaces </compiler/workspaces>`.
:::

## Operations

:::{glossary}
SDDMM
: Sampled Dense-Dense Matrix Multiplication — $S \odot (A B^\top)$, computing a dense
  product only at a sparse mask's nonzeros, `"ij,ik,jk->ij"` in {func}`~scorch.einsum`.
  A prebuilt COO fast path exists. See {doc}`the SDDMM tutorial </tutorials/sddmm>`.

SpGEMM (SpMSpM)
: Sparse General Matrix-Matrix multiply — sparse × sparse producing a sparse result,
  $C_{ik} = \sum_j A_{ij} B_{jk}$ with both operands sparse (e.g. `"ds" @ "ds" -> "ds"`).
  Reached via {func}`~scorch.matmul` with a sparse output format. See
  {doc}`the SpGEMM tutorial </tutorials/spgemm>`.

SpMM
: Sparse Matrix-Matrix multiply — sparse × dense, $C_{ik} = \sum_j A_{ij} B_{jk}$ with
  `A` sparse (CSR) and `B` dense. The workhorse of GCN and autoencoder workloads and the
  path the {term}`autotune level` tunes. See {doc}`the SpMM tutorial </tutorials/spmm>`.

SpMV
: Sparse Matrix-Vector multiply — $y_i = \sum_j A_{ij} x_j$ with `A` sparse and `x`
  dense. Reached through {func}`~scorch.matmul` with a 1-D right operand (the internal
  {func}`~scorch.ops.spmv` is not exported directly). See
  {doc}`the SpMV tutorial </tutorials/spmv>`.
:::

## Autotuning and performance

:::{glossary}
autotune level
: The `-O`-style knob (`off` / `analytic` / `balanced` / `max` / `learned`) controlling
  how the CSR × dense {term}`SpMM` path is dispatched, set via
  {func}`~scorch.set_autotune`, {func}`~scorch.get_autotune`, or the
  {class}`~scorch.autotune` context manager/decorator. It is no-regression by
  construction — `spmm_csr_float_v2` is always a candidate. See
  {doc}`autotuning </user_guide/autotuning>`.

tile-j
: A prebuilt SpMM strategy (`spmm_csr_float_tilej`) that cache-blocks the contraction
  axis `j`, splitting the dense operand `B` into column panels that fit the last-level
  cache to recover cross-row reuse. Chosen only for high-degree matrices that thrash the
  LLC. See {doc}`the tuning guide </performance/tuning_guide>`.

tile-ijk
: A prebuilt SpMM strategy (`spmm_csr_float_tileijk`) that relays `B` into contiguous
  wide free-dim panels so its output traffic stays linear in `N`, for the very-wide-`B`
  tail. Gated to `N >= 512` (`NIJK_MIN`), above every current GCN/autoencoder workload, so
  it is provably inert on them. See {doc}`the tuning guide </performance/tuning_guide>`.
:::

## See also

- {doc}`Key concepts </getting_started/key_concepts>` — the same ideas in tutorial form.
- {doc}`The format system </user_guide/format_system>` — the full treatment of level
  types and format notation.
- {doc}`The compiler pipeline </compiler/pipeline>` — how CIN, LLIR, and codegen fit
  together.
