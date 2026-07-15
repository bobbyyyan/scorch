# Architecture

A map of the Scorch repository for contributors: where each piece lives, how the
public API sits on top of the sparse-tensor compiler, and where to add new code.
If you are here to *understand* the compiler rather than navigate the source tree,
start with {doc}`/compiler/index` and {doc}`the pipeline walk-through
</compiler/pipeline>`; this page is the orientation layer beneath them.

## The shape of the codebase

Scorch is a sparse-tensor compiler in the [TACO](https://tensor-compiler.org)
lineage, wrapped in a drop-in PyTorch-style API. A user writes
`scorch.matmul(A, B)`; Scorch either dispatches to a hand-written C++ kernel or
JIT-compiles a bespoke kernel for that exact `(operation × operand-format ×
output-format)` combination. The repository is organized around that flow:

```text
scorch/
├── src/scorch/            # the Python package (public API + compiler)
│   ├── __init__.py        #   the shim: unknown attrs fall through to real torch
│   ├── stensor.py         #   STensor — the user-facing sparse tensor
│   ├── storage.py         #   TensorStorage — index + values backing STensor
│   ├── format.py          #   TensorFormat + LevelType — per-mode layout
│   ├── ops.py             #   matmul / einsum / spmv dispatch + generic entry
│   ├── prebuilt_kernels.py #  registry mapping ops → hand-written C++ kernels
│   ├── tiling.py          #   runtime autotune selector over prebuilt SpMM
│   ├── trace.py           #   @scorch.compile (torch.fx fusion decorator)
│   ├── utils.py           #   JIT compile + .so cache + compiler flags
│   ├── csrc/              #   packaged native sources, headers, JIT resources
│   └── compiler/          #   the sparse-tensor compiler (the heart)
│       ├── cin.py         #     CIN — Compiler Index Notation (highest IR)
│       ├── scheduler.py   #     loop ordering, workspace insertion, tiling
│       ├── cin_lowerer.py #     CIN → LLIR (largest file, ~180 KB)
│       ├── iterator.py    #     per-mode position/coordinate arithmetic
│       ├── iter_lattice.py #    merge lattices for sparse co-iteration
│       ├── llir.py        #     LLIR — Low-Level IR (typed, C++-shaped)
│       └── codegen.py     #     LLIR → C++ source string
├── tests/                 # pytest suite, mirrors src/ layout
├── examples/              # runnable applications (gcn, autoencoder, transformer)
├── bench/                 # reproducible benchmarks
├── tools/                 # tuning / benchmark utilities
├── docs/                  # this documentation site
├── pyproject.toml         # package metadata + native extension declaration
├── scorch_build.py        # PyTorch/OpenMP build_ext customization
├── setup.sh               # development-environment bootstrap
└── CLAUDE.md / AGENTS.md   # contributor conventions
```

Two facts to anchor on:

- **`import scorch` is a shim.** Anything Scorch does not define falls through to
  real PyTorch via `__getattr__` in `src/scorch/__init__.py`. Only the sparse ops
  Scorch implements — `matmul`, `einsum`, {func}`~scorch.ops.spmv`, and the neural
  primitives — are intercepted.
- **The public API is a thin surface over a compiler.** `ops.py` is the doorway;
  everything under `src/scorch/compiler/` is the machinery that turns an index
  expression into a compiled `.so`.

## The public API layer (`src/scorch`)

These are the files a user's calls flow through, and the first place most feature
work lands.

`stensor.py`
: Defines {class}`~scorch.STensor`, the user-facing sparse tensor. Interop helpers
  (`from_torch`, `from_coo`, `from_csr`, `to_torch`, `to_sparse`, `to_dense`) live
  here. An `STensor` wraps a `TensorStorage` and a `TensorFormat`.

`storage.py`
: `TensorStorage` — the raw index arrays plus the values buffer that back an
  `STensor`. This is the boundary that the compiled C++ `evaluate()` reads and
  writes.

`format.py`
: {class}`~scorch.TensorFormat` and `LevelType`: the declarative, per-mode layout.
  Format strings use one character per mode — `d` = dense, `s`/`c` =
  compressed/sparse, `o` = coordinate — so `"ds"` is CSR, `"oo"` is COO, `"ss"` is
  DCSR, `"dd"` is dense. See {doc}`the format system </user_guide/format_system>`
  for the full notation.

`ops.py`
: The dispatcher and the generic-path entry point. {func}`~scorch.matmul` and
  {func}`~scorch.einsum` first try fast paths (both-dense → real `torch.matmul`;
  then a prebuilt-kernel lookup) and only fall into the full compiler for genuinely
  sparse, non-prebuilt cases. `lower_and_exec_cin()` is the end-to-end generic
  entry; `einsum()` is the richer front-end that *builds* the CIN and runs the
  scheduler. This is Stage 1 of {doc}`the pipeline </compiler/pipeline>`.

`prebuilt_kernels.py`
: The registry that maps an `(op, operand formats)` request to a hand-written C++
  kernel in the native `scorch_ops` extension. `resolve_prebuilt_matmul()` is what
  `ops.py` consults before ever building a CIN.

`tiling.py`
: The runtime autotune selector. Given a resolved prebuilt SpMM, it picks among
  tiling variants (`none` / `tile-i` / `tile-j` / `tile-ijk`) based on matrix
  shape, sparsity, and cache size. This is a *separate* mechanism from the
  compile-time scheduler in `compiler/scheduler.py` — see the note below. User
  control over it is the autotune-level API ({func}`~scorch.set_autotune`,
  {func}`~scorch.get_autotune`); see {doc}`/user_guide/autotuning`.

`trace.py`
: {func}`~scorch.compile` — a `torch.fx`-based decorator that traces a user
  function, finds fusible contraction + elementwise chains, and dispatches to
  prebuilt or JIT-fused kernels. Distinct from the per-op compiler. See
  {doc}`/user_guide/scorch_compile`.

`utils.py`
: The JIT backend. `_load_kernel()` wraps `torch.utils.cpp_extension.load_inline`,
  manages the two-level module + on-disk `.so` cache, and `get_extra_cflags()` /
  `get_extra_ldflags()` assemble the per-platform compile and link flags. This is
  Stage 6 of the pipeline. Build details are covered in
  {doc}`/development/building`.

:::{note}
`import scorch` is the standard import. The `import scorch as torch` drop-in idiom
appears only in the application tutorials ({doc}`/tutorials/gcn`,
{doc}`/tutorials/sparse_autoencoder`, {doc}`/tutorials/sparse_transformer`), where
transparently swapping PyTorch for Scorch is the teaching point.
:::

## The compiler package (`src/scorch/compiler`)

This is where an operation is lowered through progressively lower IRs until it
becomes emittable C++. Each file owns one stage. The full narrative — with worked
examples and an end-to-end trace of `C[i,k] = A[i,j] * B[j,k]` — lives in
{doc}`/compiler/pipeline`, {doc}`/compiler/index_notation`,
{doc}`/compiler/lowering`, and {doc}`/compiler/codegen`. Here we map stage to file.

`cin.py`
: **CIN — Compiler Index Notation.** The highest IR: an index-notation AST DSL of
  `ForAll` / `Where` / `TensorAssign` / `TensorAccess` / `IndexVar` / `TensorVar` /
  `Workspace` / `BinaryOp`. It describes *what* is computed, not *how* it loops. A
  `ForAll` deliberately does **not** fix an execution order — that is the
  scheduler's job.

`scheduler.py`
: The `Scheduler`. `select_loop_order()` chooses the loop nesting via a calibrated
  cost model; `auto_schedule()` reorders the `ForAll`s, inserts a `Workspace`
  (wrapped in a `Where` producer/consumer split) for reductions, and applies tiling
  heuristics. See {doc}`/compiler/workspaces` for the workspace concept.

`cin_lowerer.py`
: The `CINLowerer` — **CIN → LLIR**, the largest file in the repository (~180 KB).
  It walks the CIN, drives the iterator analysis over each tensor's sparsity
  structure, and emits typed LLIR, ultimately wrapping everything into the
  `evaluate` function.

`iterator.py`
: `ModeIterator` — turns one level's `LevelType` into concrete position/coordinate
  arithmetic. A dense level becomes row-major offset math; a compressed (CSR) level
  becomes a `pos`-array walk; a coordinate (COO) level iterates a `crd` array. This
  is where *format* meets *loop code*.

`iter_lattice.py`
: `LatticePoint` and `IterationLattice` — the **merge lattice** (à la TACO) that
  co-iterates multiple sparse operands. For a given loop variable it computes the
  union/intersection of the operands' iteration domains and emits the guarded
  `while`/`for` merge. The conceptually deepest part of the compiler.

`llir.py`
: **LLIR — Low-Level IR.** A typed, C++-shaped IR whose nodes (`ForLoop`,
  `IfThenElse`, `Assign`, `VarInit`, `Function`, …) map roughly 1:1 to emitted C++.
  `ForLoop` carries the OpenMP metadata (thread counts, schedule, chunk expressions)
  that codegen turns into pragmas.

`codegen.py`
: The `LLIRLowerer` — **LLIR → a single C++ source string**. A recursive walker
  that emits the `evaluate()` function, including all OpenMP pragma synthesis.

### Stage → file, at a glance

| Stage | What it does | File(s) |
|-------|--------------|---------|
| 1. Dispatch | fast paths, prebuilt lookup, generic entry | `ops.py`, `prebuilt_kernels.py`, `tiling.py` |
| 2. CIN | build the index-notation AST | `compiler/cin.py` |
| 2b. Schedule | loop order, workspace, tiling | `compiler/scheduler.py` |
| 3. Lower CIN | CIN → LLIR, iterator analysis | `compiler/cin_lowerer.py`, `iterator.py`, `iter_lattice.py` |
| 4. LLIR | typed C++-shaped IR | `compiler/llir.py` |
| 5. Codegen | LLIR → C++ string | `compiler/codegen.py` |
| 6. JIT compile | `load_inline`, `.so` cache, flags | `utils.py` |
| 7. Execute | run `evaluate()`, wrap result back into an STensor | `ops.py` |

```{mermaid}
flowchart TD
    U["scorch.matmul(A, B)<br/>scorch.einsum(...)"] --> D
    D["Dispatch<br/>ops.py"] -->|prebuilt hit| K["scorch_ops C++ kernel<br/>src/scorch/csrc/"]
    D -->|generic sparse| C["CIN<br/>compiler/cin.py"]
    C --> S["Scheduler<br/>compiler/scheduler.py"]
    S --> L["CINLowerer<br/>compiler/cin_lowerer.py"]
    L --> LL["LLIR<br/>compiler/llir.py"]
    LL --> CG["Codegen<br/>compiler/codegen.py"]
    CG --> J["JIT compile + cache<br/>utils.py"]
    J --> E["evaluate() → STensor<br/>ops.py"]
    K --> E
```

:::{note}
Two tiling mechanisms coexist and should not be confused. The **compile-time**
`Scheduler` (`compiler/scheduler.py`) orders and tiles loops in the *generic JIT*
path. The **runtime** selector (`tiling.py`) picks among *hand-written prebuilt*
SpMM kernels during dispatch — it never touches the compiler IRs.
:::

## The native layer (`src/scorch/csrc`)

The `src/scorch/csrc/` directory holds the hand-written C++ declared in
`pyproject.toml` and compiled into the `scorch_ops` pybind extension — the fast
prebuilt kernels — plus the packaged runtime support that JIT-generated kernels
are compiled against. Keeping these files inside `scorch` makes both wheels and
source distributions self-contained.

| File | Role |
|------|------|
| `ops.cpp` | The single compilation unit for `scorch_ops`; hosts the prebuilt kernel entry points. |
| `spmm.h` | Hand-optimized SpMM kernels (the tiling variants, thread policy). |
| `kernels.h` | Other prebuilt kernels (SpMV, SDDMM, fused neural ops). |
| `header.h` | Canonical runtime prepended to every JIT kernel: tensor result types, checked `std::vector` assembly, and move-only workspaces. |
| `scorch_policy.h` | The parallel thread-policy header shared by prebuilt and JIT paths. |
| `prebuilt_types.h` | Shared type declarations for the prebuilt kernels. |

:::{warning}
`src/scorch/csrc/pybind.cpp` and the top-level `CMakeLists.txt` are **legacy /
IDE-indexing only**. The supported build path is the extension declaration in
`pyproject.toml` plus the PyTorch `BuildExtension` subclass in `scorch_build.py`
(`pip install -e . --no-build-isolation` for a fast developer rebuild), not CMake.
Do not add production build logic to either legacy file.
:::

Generated kernels (from `codegen.py`) are compiled *separately at runtime* by
`utils.py`. It reads `header.h` and `scorch_policy.h` from the installed package
with `importlib.resources`, expands the policy include, and passes a self-contained
source string to PyTorch's JIT compiler. No repository-relative include path is
required.

:::{tip}
After editing anything under `src/scorch/csrc/`, rebuild the extension with
`pip install -e . --no-build-isolation`. Native headers are listed as extension
dependencies in `pyproject.toml`, so header-only changes trigger recompilation
without touching `ops.cpp`. Because
compiled JIT kernels are memoized aggressively — in-process operation and
generated-kernel caches keyed by canonical identity *plus* a persistent `.so`
cache — clear the torch extensions build dir when a codegen or template change
seems to have no effect. The test suite sidesteps this by isolating
`TORCH_EXTENSIONS_DIR` to a tmp dir per session (`tests/conftest.py`).
:::

## Supporting directories

`tests/`
: The pytest suite, mirroring the `src/` layout under `tests/test_scorch/`, with
  code-generation cases in `tests/test_scorch/codegen/`. Correctness tests compare
  against a PyTorch reference with `assert_close()` (`atol=rtol=1e-3`) rather than
  hardcoded numerics. Known-limitation tests live in
  `tests/test_scorch/test_known_compiler_gaps*.py` — consult these before assuming
  a format / loop-order combination is supported. See {doc}`/development/testing`.

`examples/`
: Runnable applications — `gcn/`, `sparse_autoencoder/`, `sparse_transformer/`, and
  per-op `kernels/` demos. These back the {doc}`tutorials </tutorials/index>`.

`bench/`
: Reproducible benchmarks. Benchmark result CSVs are run artifacts and are
  gitignored — never commit them.

`tools/`
: Tuning and benchmark utilities.

## Where to add things

Adapted from the project conventions in `CLAUDE.md`. Match the surrounding stage's
patterns and add a regression test near the affected subsystem.

**A new sparse operation**
: Add it to `ops.py`, following the {func}`~scorch.matmul` / {func}`~scorch.einsum`
  pattern — build a CIN, then call `lower_and_exec_cin`. Test in
  `tests/test_scorch/test_kernels*.py`; add a demo under `examples/kernels/`.

**A new format level type**
: Extend `LevelType` in `format.py`, then teach the merge lattice to co-iterate it
  in `compiler/iter_lattice.py` (and `iterator.py` for its position/coordinate
  arithmetic). Test in `tests/test_scorch/test_format_*.py`.

**A new codegen or scheduling optimization**
: Work in `compiler/scheduler.py` (loop order, workspace, tiling) and/or
  `compiler/codegen.py` (emitted C++). Test in `tests/test_scorch/codegen/` and
  `test_codegen_perf_optimizations.py`.

**A new prebuilt C++ kernel**
: Implement it in `src/scorch/csrc/kernels.h` (or `src/scorch/csrc/spmm.h` for
  SpMM variants), register
  it in `prebuilt_kernels.py`, and rebuild the extension. Test in
  `test_prebuilt_kernel_registry.py`.

:::{admonition} Performance changes must generalize
:class: warning
An optimization ships only if it is neutral-or-better across the whole space Scorch
cares about — narrow- and wide-`k` SpMM, small and large row counts, sparse and
near-dense, and the GCN / autoencoder / attention workload families — with no
regressions on any of them. If a change only helps a sub-regime, gate it behind a
runtime condition that provably cannot fire on the shapes it would hurt, and always
measure across a shape / format / density grid before and after. See
{doc}`/development/contributing` and {doc}`/performance/tuning_guide`.
:::

## See also

- {doc}`/compiler/index` — the compiler internals section, and
  {doc}`/compiler/pipeline` for the full end-to-end lowering walk-through.
- {doc}`/development/building` — how the native extension and JIT kernels are
  compiled, and the caching gotchas.
- {doc}`/development/contributing` — coding style, commit conventions, and the
  performance policy.
