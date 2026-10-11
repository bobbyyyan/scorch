# Code generation & JIT compilation

This page follows a Scorch operation through the last three stages of the
compiler: the **Low-Level IR (LLIR)**, the **codegen** pass that prints LLIR as a
C++ source string, and the **JIT compile + cache** step that turns that string
into a running, OpenMP-parallel kernel. If you have read
{doc}`the compiler pipeline </compiler/pipeline>` and
{doc}`lowering </compiler/lowering>`, this is where the abstract loop nest finally
becomes machine code.

The one-sentence summary: **LLIR is a typed, C++-shaped IR; codegen walks it and
emits a string; `torch.utils.cpp_extension.load_inline` compiles that string once
(~7 s) into a `.so` that every later call — even across process restarts — reuses
from disk.**

```{contents}
:local:
:depth: 2
```

## Stage 4 — LLIR (Low-Level IR)

LLIR is the compiler's lowest intermediate representation. Where
{doc}`CIN </compiler/index_notation>` describes *what* is computed in
index-notation form, LLIR describes *how* it executes as concrete typed loops,
conditionals, and assignments — nodes that map **~1:1 to C++**. It is produced by
the `CINLowerer` (see {doc}`lowering </compiler/lowering>`) and consumed by
codegen.

Every LLIR node is a `Node`, split into `Expr` (expressions) and `Stmt`
(statements), defined in `compiler/llir.py`.

### Types

Types are the `DataType` enum — a set of concrete C++ type *strings*: scalars
(`INT64 = "int64_t"`, `FLOAT32 = "float"`), the torch/runtime structs
(`TORCH_TENSOR = "torch::Tensor"`, `TACO_TENSOR = "Tensor"`), pointer families
(`PTR_FLOAT32 = "float*"`), standard vectors used by sparse builders, and
`coo_workspace<T, dim>` that backs Scorch's {doc}`workspaces </compiler/workspaces>`.
Because a `DataType` *is* its C++ spelling, codegen never has to translate types —
it prints them verbatim.

### Statement and expression nodes

The statement nodes are exactly the C++ constructs you would expect a loop nest to
need:

| Node | Emits |
|------|-------|
| `VarDecl(var)` | `type name;` |
| `VarInit(var, value, op, cast)` | `type name = value;` |
| `DirectInit(var, args)` | `type name(args...);` |
| `Assign(var, value, op)` | `name op value;` (`op` includes `+=`, `*=`, …) |
| `Increment(var)` | `x++;` |
| `FixedStackArrayDecl(...)` | `type name[extent] = {};` |
| `ForLoop(...)` | a C `for` loop, carrying OpenMP metadata (below) |
| `ForLoopAuto(var, array, body)` | `for (type x : arr)` |
| `WhileLoop(cond, body)` | `while (…)` — used for sparse merge co-iteration |
| `IfThenElse(...)` | `if` / else-if chains |
| `Function(return_type, name, args, body)` | a full C++ function definition |
| `FunctionCallStmt` / `RawStmt` | call and compatibility statements |
| `Return` / `Break` / `Continue` / `Comment` / `BlankLine` | control flow and formatting |

Expressions cover the arithmetic and memory-access surface: `Var` (with optional
`__restrict__`), `Literal`, `QualifiedName`, `BinOp` / `Add` / `Mul`, `UnaryOp`,
`Cast`, `Sizeof`, `FunctionCall`, `MemberAccess` / `MemberCall`, and `Array` /
`ArrayAccess`.

As a concrete example, the innermost SpMM update `C[i,k] += A[i,j] * B[j,k]`
lowers to a single `Assign`:

```python
llir.Assign(
    var=llir.Var("C_vals[pC1]", DataType.NO_TYPE),
    value=llir.Mul(
        llir.Var("A_vals[pA1]", ...),
        llir.Var("B_vals[pB1]", ...),
    ),
    op=AssignOp.ADD_ASSIGN,   # +=
)
```

Read left to right, that node *is* the C++ line `C_vals[pC1] += A_vals[pA1] *
B_vals[pB1];`. This near-literal correspondence is the whole point of LLIR — by the
time you reach it, all the interesting decisions (iteration order, workspace
insertion, sparse merges) have already been made upstream, and codegen becomes a
mechanical pretty-printer.

### The parallel policy lives on `ForLoop`

The one place LLIR carries more than plain C++ is the OpenMP metadata on
`ForLoop`. These fields encode Scorch's work-aware threading policy, which codegen
turns into `#pragma` directives:

`omp_parallel_for` (bool), `omp_schedule` (e.g. `"dynamic"`), `unroll`, `simd`
: the basic pragma switches.

`omp_num_threads`
: a C++ expression for the `num_threads(...)` clause — e.g.
  `scorch_nthreads(work, rows)`, so thread count adapts to the actual work of the
  matrix at runtime rather than being fixed at compile time.

`omp_chunk_expr`
: a dynamic-schedule chunk-size expression that overrides the static chunk.

`pre_parallel_body` / `post_parallel_body`
: statements placed *inside* `#pragma omp parallel` but outside the `for`, so
  codegen can split a `parallel for` into `parallel { pre; for; post }` — used for
  per-thread setup/teardown around the parallel loop.

## Stage 5 — Codegen (LLIR → C++ string)

Codegen is `LLIRLowerer` in `compiler/codegen.py`. Its docstring is exactly *"lower
LLIR to C++ code (string)."* It is a recursive, **string-emitting** walker —
`lower_llir(ir, indent_level, ...)` — that dispatches on exact Python node types:

- bare strings pass through with indentation applied;
- exact lists and tuples of statements are joined with newlines (empties dropped);
- `VarInit` prints `type name = value;`, `Assign` prints `name op value;`;
- expression nodes route to `lower_expression`;
- loops route to `lower_loop_construct`, conditionals to `lower_conditional`,
  functions to `lower_function_definition`.

The declared-node and traversal registries contain the same exact node families
that this emitter accepts. Unknown subclasses and list/tuple subclasses fail
closed. Scalar strings are an internal formatting path rather than LLIR nodes.

:::{note}
LLIR node definitions live in `compiler/llir.py`. The production C++ emitter is
`LLIRLowerer` in `compiler/codegen.py`; `llir.py` does not contain a second code
generator.
:::

The output of this stage is one big C++ string containing an `evaluate(...)`
function whose arguments are `result_shape` followed by a `(shape, mode_indices,
values)` triple per input tensor. That argument layout is exactly how the caller
packs its arguments at execution time (see
{doc}`the pipeline's execute stage </compiler/pipeline>`).

### How the parallel policy becomes pragmas

`lower_loop_construct` is where the `ForLoop` OpenMP fields turn into text. The
mapping is direct:

- `num_threads(<expr>)` is appended when `omp_num_threads` is set;
- `schedule(dynamic, <chunk>)` is emitted when `omp_chunk_expr` is present,
  otherwise `schedule(<omp_schedule>)`;
- when `pre_parallel_body` / `post_parallel_body` are set, codegen emits the split
  form `#pragma omp parallel { pre; #pragma omp for …; post }`;
- otherwise a plain `#pragma omp parallel for …`, plus optional `#pragma unroll`
  and `#pragma omp simd`.

For the running SpMM example, the emitted body looks roughly like:

```cpp
Tensor evaluate(std::vector<int64_t> result_shape,
                std::vector<int64_t> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values,
                /* B ... */) {
    // ... allocate C ...
    #pragma omp parallel for schedule(dynamic) num_threads(scorch_nthreads(...))
    for (int64_t i = 0; i < M; ++i) {
        float accum_c[K] = {0};
        for (int64_t pA = A1_pos[i]; pA < A1_pos[i + 1]; ++pA) {
            int64_t j = A1_crd[pA];
            for (int64_t k = 0; k < K; ++k)
                accum_c[k] += A_values[pA] * B_values[j * K + k];
        }
        for (int64_t k = 0; k < K; ++k)
            C_vals[i * K + k] = accum_c[k];
    }
    // ... wrap into the runtime Tensor struct and return ...
}
```

The `accum_c[K]` scratch row is the dense {doc}`workspace </compiler/workspaces>`
the scheduler inserted; the `A1_pos` / `A1_crd` arithmetic is the CSR iteration
that {doc}`lowering </compiler/lowering>` generated from A's `"ds"` format.

## Stage 6 — JIT compile & the two-tier cache

Before compilation, the generated `evaluate(...)` string is prepended with the
packaged `src/scorch/csrc/header.h` resource — the runtime support layer that
defines the `Tensor` / `TensorStorage` structs, RAII vector-to-tensor transfer,
`coo_workspace<T, dim>`, and the `#include "scorch_policy.h"` threading policy.
`utils.py` reads both files through `importlib.resources` and expands that include
before handing the self-contained source to `_load_kernel(...)`.

### Compiler and linker flags

Kernels are compiled with `torch.utils.cpp_extension.load_inline`. The default base
compile flags (`get_extra_cflags`) are:

```python
["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
```

OpenMP is added **per platform**, because the two OSes disagree on how to enable and
link it:

- **macOS:** `-Xpreprocessor -fopenmp`, plus a Homebrew `libomp` include path and
  the macOS SDK C++ stdlib include.
- **Linux:** `-fopenmp`.

The policy header is embedded into the generated source, so runtime compilation
does not depend on a repository-relative include directory.

CLAUDE.md summarizes the effective flag set as
`-O3 -march=native -ffast-math -funroll-loops -fopenmp`.

Linking is the platform-specific part most likely to bite you. `get_extra_ldflags`
links **PyTorch's own bundled OpenMP** rather than the system one, to avoid a
dual-runtime conflict: `libomp.dylib` (with an rpath) on macOS, `libgomp*.so*` on
Linux, with Homebrew/system fallbacks. See {doc}`building from source
</development/building>` for the environment setup this depends on.

### Two-tier `.so` cache

The core performance property of the JIT path is that **compilation cost is paid
once**. `_load_kernel` implements two levels of caching:

1. **In-process memo.** A module-level dict `_so_cache` keyed by kernel name — a hit
   returns the already-loaded module immediately, no filesystem access.
2. **On-disk `.so`.** If the module is not in the dict, `_load_kernel` computes the
   build directory and checks whether `<name>.so` already exists. **If it does, the
   module is loaded directly with `importlib` — bypassing ninja and
   `load_inline` entirely.** Only if there is no `.so` does it fall through to
   `load_inline`, which invokes the compiler.

The `importlib` shortcut exists for a specific reason, called out in the
`_load_kernel` docstring: PyTorch's `JIT_EXTENSION_VERSIONER` is *in-memory only*,
so `load_inline` would recompile on the first call of **every** process (~7 s each),
even when a perfectly good `.so` is sitting on disk. Loading it directly sidesteps
that.

The net behavior:

> The first call to an operation compiles a specialized C++ kernel (~7 s); every
> subsequent call with the same format combination — **including after a process
> restart** — loads the cached shared library from disk and runs at full speed.

### The cache key

The kernel name doubles as the cache key. `_kernel_name(*sources)` is the MD5 of
`concat(sources) + policy_header_text + torch.__version__`, truncated to 12 hex
digits (`kernel_<hash>`). Folding those two extra strings into the hash matters:

- **`torch.__version__`** invalidates every cached `.so` across a PyTorch upgrade
  (ABI changes would otherwise silently mismatch).
- **the parallel-policy header text** invalidates the cache when the per-host
  threading policy is re-tuned.

:::{warning}
**Stale caches can mask codegen edits.** Because compiled kernels are memoized
aggressively — the typed operation and generated-kernel caches in `ops.py` plus
the persistent on-disk `.so` cache — a change you make to codegen or to a
`src/scorch/csrc/` template may not take effect until the old cached kernel is
removed. If you edit codegen and your change appears to do nothing, clear the
torch extensions build directory (`TORCH_EXTENSIONS_DIR`) before re-running.

The test suite isolates this automatically: a session fixture in
`tests/conftest.py` points `TORCH_EXTENSIONS_DIR` at a fresh temp directory, so a
`pytest` run never reuses a stale `.so` from a previous edit. `clear_autotune_cache`
is unrelated — it wipes the autotune probe cache, **not** the JIT `.so` cache.
:::

## Observing the JIT in action

You cannot see the C++ string from the public API, but you *can* observe the
compile-once-then-cache behavior directly, and verify the result against a PyTorch
reference. The standard convention is `import torch` then `import scorch`:

```python
import time
import torch
import scorch

# Sparse (CSR) A @ dense B, routed through the compiler pipeline.
A = torch.randn(256, 512)
A[A.abs() < 1.0] = 0.0                        # ~68% zeros
As = scorch.from_torch(A.to_sparse_csr(), "A")  # STensor, "ds" / CSR
B = torch.randn(512, 64)

# First call: compiles (or resolves a prebuilt kernel) — the slow one.
t0 = time.perf_counter()
C1 = scorch.matmul(As, B)
first = time.perf_counter() - t0

# Second call, same format combination: served from cache — much faster.
t0 = time.perf_counter()
C2 = scorch.matmul(As, B)
second = time.perf_counter() - t0

print(f"first={first*1e3:.1f} ms  second={second*1e3:.1f} ms")

# Correctness: match a dense PyTorch reference (project convention).
ref = A @ B
out = C1 if isinstance(C1, torch.Tensor) else C1.to_torch()
assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
```

To time only the compiled `evaluate()` call (excluding dispatch and wrapping), pass
a `time_dict`:

```python
td = {}
_ = scorch.matmul(As, B, time_dict=td)
print(td["eval_time"])        # seconds spent inside module.evaluate()
```

:::{note}
Whether a given `matmul` goes through the **JIT compiler** or a hand-written
**prebuilt kernel** is decided in dispatch — many common SpMM shapes hit a prebuilt
kernel and never build a JIT `.so` at all. Either way the two-tier caching story is
the same; the prebuilt path just skips straight to a shared library that ships with
the extension. See {doc}`the pipeline overview </compiler/pipeline>` for how
dispatch chooses.
:::

## See also

- {doc}`The compiler pipeline </compiler/pipeline>` — the full journey from a
  `matmul` call to a running kernel, of which this page is the tail.
- {doc}`Lowering </compiler/lowering>` — how CIN and iterator/merge-lattice
  analysis produce the LLIR that codegen consumes.
- {doc}`Autotuning </user_guide/autotuning>` — the `-O` level ladder over the
  prebuilt SpMM kernels, and its own separate persistent cache.
- {doc}`Building from source </development/building>` — the toolchain and OpenMP
  setup the JIT path depends on.
