# The compilation pipeline

Scorch is a **sparse-tensor compiler** in the [TACO](https://tensor-compiler.org)
lineage. A single sparse operation — `scorch.matmul`, `scorch.einsum`,
`scorch.ops.spmv`, and friends — is *not* interpreted at runtime. Instead it is
compiled, once per unique *(operation × operand formats × output format)*
combination, into a bespoke C++ `evaluate()` function that is JIT-compiled to a
shared library and cached. This page walks the seven stages that turn a user call
into machine code, and traces the canonical SpMM
$C_{ik} = \sum_j A_{ij}\,B_{jk}$ through all of them.

If you only remember one thing: **the same math produces a *different kernel* for
every format combination**, because the loop structure that is efficient for CSR ×
dense is not the one that is efficient for DCSR × DCSR. Specializing per format is
the whole point.

## The stack at a glance

```{mermaid}
flowchart TD
    U["scorch.matmul(A, B) /<br/>scorch.einsum('ik,kj->ij', A, B)"]
    D["<b>1. Dispatch</b><br/>ops.py"]
    C["<b>2. CIN</b> — Compiler Index Notation<br/>compiler/cin.py"]
    S["<b>2b. Scheduler</b><br/>loop order · workspaces · tiling<br/>compiler/scheduler.py"]
    L["<b>3. CINLowerer</b><br/>iterator + merge-lattice analysis<br/>compiler/cin_lowerer.py"]
    IR["<b>4. LLIR</b> — Low-Level IR<br/>compiler/llir.py"]
    CG["<b>5. Codegen</b> → C++ source string<br/>compiler/codegen.py"]
    J["<b>6. JIT compile</b> → .so (cached)<br/>utils.py"]
    E["<b>7. Execute</b> → wrap into STensor<br/>ops.py"]

    U --> D
    D -->|"both dense →<br/>torch.matmul"| PT["real PyTorch"]
    D -->|"prebuilt hit →<br/>hand-written C++"| PB["scorch_ops kernel"]
    D -->|"generic sparse case"| C
    C --> S --> L --> IR --> CG --> J --> E
```

The generic path's end-to-end entry point is `lower_and_exec_cin()` in `ops.py`;
`einsum()` is the richer front-end that *builds* the CIN and runs the scheduler
before lowering. The two branches that peel off at dispatch — real PyTorch for
fully dense operands, and hand-written **prebuilt kernels** for common sparse
shapes — are the fast paths that most calls actually take. Only genuinely sparse,
non-prebuilt combinations descend into the compiler.

## Stage 1 — Dispatch

Before the compiler ever runs, {func}`~scorch.matmul` / {func}`~scorch.einsum` /
{func}`~scorch.ops.spmv` try progressively cheaper paths and stop at the first that
applies:

1. **Both real dense `torch.Tensor`** → delegate straight to `torch.matmul`.
   Densely stored operands are both faster and more reliable through PyTorch than
   through sparse scheduling.
2. **Any sparse operand** → wrap operands as {class}`~scorch.STensor` (via
   {func}`~scorch.from_torch`), canonicalize the mode order.
3. **Prebuilt kernel lookup** — `resolve_prebuilt_matmul()` checks a registry of
   hand-optimized C++ SpMV/SpMM kernels (e.g. `spmm_csr_float_v2` and the adaptive
   tiling variants selected by the runtime autotuner). A hit runs the hand-written
   kernel and never touches the compiler.
4. **No prebuilt kernel resolves** → fall into the generic compiler via `einsum` /
   `lower_and_exec_cin`.

:::{note}
The **prebuilt** kernels reached here are a separate mechanism from the compiler's
**JIT / generated** kernels. Prebuilts are the shipped fast path for the shapes we
benchmark most heavily; the JIT compiler is the general fallback that handles *any*
format combination. See {doc}`/user_guide/operations` and
{doc}`/performance/tuning_guide` for how the two interact.
:::

To *guarantee* the generic pipeline for study, pick a format/loop-order
combination with no prebuilt registration, or drive the compiler explicitly with
{func}`~scorch.einsum` and a `format=` kwarg.

## Stage 2 — CIN (Compiler Index Notation)

CIN is the compiler's **highest IR**: an index-notation AST that describes *what*
is computed, not *how* it loops. It mirrors TACO's concrete index notation. The
core node types are:

`IndexVar`
: A loop/index variable such as `i`, `j`, `k`. It carries tiling state and knows
  which tensor accesses use it.

`TensorVar` / `TensorAccess`
: A named operand or result (`A`, `B`, `C`) and an access into it (`A[i, j]`).
  `TensorAccess` is where **format meets index notation** — it knows the level
  type (dense / compressed / coordinate) of each mode.

`ForAll`
: Binds an `IndexVar` over its range. Crucially, a `ForAll` **does not fix an
  execution order** — the loop nesting order is chosen later by the scheduler.

`TensorAssign`
: The assignment `C[i,k] = A[i,j] * B[j,k]` (or a compound `+=` update).

`Where`
: A producer/consumer split — the producer computes into a `Workspace` that the
  consumer then drains. This is how a reduction-into-a-buffer is expressed.

`Workspace`
: A temporary accumulation buffer for reductions. See {doc}`/compiler/workspaces`.

**Free vs reduction variables.** In $C_{ik} = \sum_j A_{ij} B_{jk}$, the indices
`i` and `k` appear on the result (**free** variables) while `j` appears only on
the inputs (a **reduction** variable). Scorch computes this split automatically:
free = the result tensor's index vars; reduction = everything else.

For our SpMM, `einsum("ik,kj->ij", A, B)` first builds the naïve nest, and after
the scheduler inserts a row accumulator it becomes:

```python
ForAll(i,
    Where(
        producer = ForAll(j,
                     ForAll(k,
                       TensorAssign(accum_c[k], A[i,j] * B[j,k],
                                    op=Operation.ADD))),   # accum_c[k] += A·B
        consumer = ForAll(k,
                     TensorAssign(C[i,k], accum_c[k]))     # C[i,k] = accum_c[k]
    )
)
```

Here `accum_c[k]` is a dense length-`K` workspace row: the `j`/`k` producer loop
sums into it, then the consumer copies it into `C`.

Full node reference, the `einsum` string grammar, and output-format inference live
in {doc}`/compiler/index_notation`.

## Stage 2b — Scheduler

The `Scheduler` decides *how* the format-agnostic CIN should loop. Its automatic
entry points are `select_loop_order` (returns the chosen ordering of `IndexVar`s)
and `auto_schedule` (the full transform). `apply_schedule` is the explicit tuner
entry point: it accepts a `Schedule` containing an exact logical loop order and
per-axis `TileSpec` strip-mining decisions. Conceptually:

```python
loop_order = select_loop_order(cin)          # cost-model ranks candidates
cin = _rebuild_loop_nest(cin, loop_order)    # reorder the ForAlls
if should_insert_workspace(cin, loop_order):
    cin = insert_workspace(cin)              # add the Where + Workspace
cin = _apply_tiling_heuristics(cin)          # tile i / k where profitable
```

For an explicit schedule, the same rebuild/workspace stages run, followed by the
listed affine tiles in deterministic order. A tile splits a logical variable
into `<var>_out + <var>_in`; its placement can be outermost, beneath a named
loop, or at a loop depth. The complete schedule is included in both JIT caches,
because the CIN string alone does not encode tile widths or pragma choices.
An explicit permutation must preserve the parent-before-child order of result
storage levels; input operands may be physically relaid out to satisfy the
remaining loop order.
Dense affine tiles are encoded in CIN directly. Sparse `kind="panel"` tiles are
finished just after CIN lowering, when the concrete compressed iterator exists:
the full-row position range becomes a pair of coordinate-window
`std::lower_bound` expressions, and the panel loop is placed outside the tagged
parallel row loop. A validated dense-operand relayout is completed in the same
post-lowering pass: it allocates reusable vector-backed storage, inserts the
panel- or enclosing-tile-scoped pack before parallel row computation, redirects
the selected operand read and its prefetch using logical tensor-access metadata,
and rejects any compute read that remains unstaged. An affine tile with
`accum="heap"` independently redirects dense-result updates into a compact
prefix-by-tile buffer, initializes it at tile entry, and copies it out at tile
exit. Both storage choices are validated from CIN formats, accesses, index roles,
and loop dominance before lowering; neither relies on expression text or operand
position.

Loop order is picked by a small **calibrated cost model** that weighs
per-iteration cost, workspace insert/sort/transpose costs, and an estimated
sparsity density. For a sparse output with a reduction, the scheduler also
enforces that at least one *free* variable follows the last *reduction* variable —
the lowerer requires the innermost loop to map to a result level, and workspace
insertion needs a trailing free var to accumulate over.

For our SpMM, the scheduler picks order `[i, j, k]`, sees that free var `k`
follows the last reduction var `j`, and inserts the dense `accum_c[k]` workspace
shown above — wrapping the body in the `Where` node.

Loop-order selection, the cost-model constants, and the tiling heuristics are
covered in {doc}`/compiler/lowering`; workspace insertion (dense vs COO-hashed) in
{doc}`/compiler/workspaces`.

## Stage 3 — CINLowerer (CIN → LLIR)

The `CINLowerer` turns the scheduled, format-aware CIN into **LLIR**. This is
where *format sparsity* becomes *concrete loop code*. Two pieces of machinery do
the heavy lifting:

**Iterator analysis** derives, per *(tensor access, index var, level)*, the exact
position/coordinate arithmetic for that level's layout:

- a **dense** level loops over the full extent with row-major offset arithmetic
  (`pB1 = pB0 * B1_size + k`);
- a **compressed** (CSR-like) level iterates positions
  `B1_pos[pB0] … B1_pos[pB0+1]` and reads coordinates from `B1_crd[pB1]`;
- a **coordinate** (COO) level iterates the crd array directly.

**Merge lattices** handle co-iteration. When a loop variable ranges over several
sparse operands at once, the *iteration lattice* enumerates the sub-regions where
different subsets of operands are present, and emits the corresponding
`while`-loop merges and guards. (For our CSR × dense SpMM only `A`'s row level is
sparse, so this collapses to a simple `for` over `A`'s compressed columns — the
merge lattice earns its keep on sparse × sparse.)

Lowering dispatches by node type: `TensorAssign` becomes the innermost compute
(`accum_c[k] += A_val * B_val`), `ForAll` builds the loop nest from the lattice,
and `Where` splits into a producer that fills the workspace and a consumer that
drains it. The outermost call wraps everything into an LLIR function:

```python
llir.Function(
    return_type=llir.DataType.TACO_TENSOR,   # the csrc/header.cpp `Tensor` struct
    name="evaluate",
    args=kernel_args,   # result_shape, then per input:
                        #   <name>_shape, <name>_mode_indices, <name>_values
    body=body_stmts,
)
```

The argument layout — `result_shape` first, then a `(shape, mode_indices,
values)` triple per RHS tensor — is exactly how the values are packed at call time
in stage 7. The lowerer, iterator arithmetic, and merge-lattice construction are
detailed in {doc}`/compiler/lowering`.

## Stage 4 — LLIR (Low-Level IR)

LLIR is a **typed, C++-shaped IR** that maps roughly 1:1 to the emitted code. Its
job is to be an explicit, inspectable representation of the C++ *before* it becomes
an unstructured string — so that codegen is a simple mechanical walk rather than a
place where logic lives.

Every node is an `Expr` or a `Stmt`. Expressions include `Var`, `Literal`,
`BinOp` / `Add` / `Mul`, `Cast`, `FunctionCall`, and `ArrayAccess`. Statements
include `VarInit`, `Assign` (with `+=`, `*=`, …), `ForLoop`, `WhileLoop`,
`IfThenElse`, and `Function`. Types are concrete C++ strings carried by a
`DataType` enum (`"int64_t"`, `"float"`, `"float*"`, `torch::Tensor`, the custom
`cvector<T>` and `coo_workspace<T, dim>` families, …).

The SpMM inner assignment is, in LLIR:

```python
llir.Assign(
    var=llir.Var("accum_c[k]"),
    value=llir.Mul(llir.Var("A_vals[pA1]"), llir.Var("B_vals[pB1]")),
    op=AssignOp.ADD_ASSIGN,   # +=
)
```

Critically, `ForLoop` carries the whole **parallelization policy** as structured
metadata rather than baked-in text: whether to emit `#pragma omp parallel for`,
the schedule and chunk expression, a `num_threads(...)` expression, and optional
`unroll` / `simd` flags. Codegen turns those fields into pragmas — so the *policy*
is decided during lowering and merely *rendered* later.

## Stage 5 — Codegen (LLIR → C++ string)

Codegen (`LLIRLowerer.lower_llir`) is a recursive walker that emits a single C++
source string. It dispatches on node type: strings pass through with indentation,
lists join with newlines, `VarInit`/`Assign` render as declarations and
assignments, expressions route through an expression printer, and loops and
conditionals render their control structure.

The parallelization metadata from stage 4 becomes real OpenMP here — a plain
`#pragma omp parallel for` with the requested `num_threads(...)` and
`schedule(...)` clauses, or, for the specialized paths, a split
`#pragma omp parallel { … }` block or an atomic work-stealing loop. For our SpMM
the result is roughly:

```cpp
Tensor evaluate(std::vector<int64_t> result_shape,
                std::vector<int64_t> A_shape,
                std::vector<std::vector<torch::Tensor>> A_mode_indices,
                torch::Tensor A_values,
                std::vector<int64_t> B_shape, /* … B … */) {
    // … allocate C_vals …
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i++) {
        float accum_c[K] = {0};
        for (int pA = A1_pos[i]; pA < A1_pos[i+1]; pA++) {
            int j = A1_crd[pA];
            for (int k = 0; k < K; k++)
                accum_c[k] += A_values[pA] * B_values[j*K + k];
        }
        for (int k = 0; k < K; k++) C_vals[i*K + k] = accum_c[k];
    }
    // … wrap into Tensor and return …
}
```

The full codegen walk, the OpenMP emission variants, and fused-epilogue
(post-op) codegen are covered in {doc}`/compiler/codegen`.

## Stage 6 — JIT compile

The generated `evaluate()` string is prepended with `csrc/header.cpp` — the
runtime support (`Tensor` / `TensorStorage` structs, `cvector<T>`,
`coo_workspace<T, dim>`, the parallel-policy header, OpenMP) — and handed to
`_load_kernel`. Compilation uses `torch.utils.cpp_extension.load_inline` with:

```text
-O3 -march=native -ffast-math -funroll-loops -fopenmp
```

plus platform-specific include/link flags (macOS links PyTorch's bundled
`libomp.dylib`; Linux uses the bundled `libgomp`).

**Caching is two-level and aggressive:**

- The **cache key** is an MD5 of the concatenated sources *plus the parallel-policy
  header text plus* `torch.__version__`, truncated to `kernel_<hash>`. Folding in
  the torch version invalidates stale `.so`s across a PyTorch upgrade; folding in
  the policy header invalidates on per-host retuning.
- If the `.so` already exists on disk, it is loaded directly via `importlib` — no
  ninja, no recompile. PyTorch's own JIT versioner is in-memory only and would
  otherwise recompile (~7s) on the first call in each process; Scorch bypasses it.
  In-process module caches then make repeat calls essentially free.

:::{warning}
Because caching keys on source text, a stale cache can **mask** an edit you just
made to codegen or a `csrc/` template. When changing the compiler's output, clear
the torch extensions build dir. The test suite isolates this automatically by
pointing `TORCH_EXTENSIONS_DIR` at a temp directory.
:::

You can pre-warm the cache without executing anything via
{func}`~scorch.precompile_kernels` (or `einsum(..., compile_only=True)`), which
builds and caches the kernel and returns a placeholder.

## Stage 7 — Execute and wrap back to STensor

Finally, the operand values are packed in the exact order the `evaluate` signature
expects, and the compiled function runs:

```python
module_args = [result_shape]
for arg in args:                       # each input STensor
    module_args.append(arg.shape)
    module_args.append(arg.index.mode_indices)
    module_args.append(arg.values)

result_cpp = module.evaluate(*module_args)     # runs the compiled C++
```

`evaluate()` returns the C++ `Tensor` struct; Scorch wraps its shape, indices, and
values back into an {class}`~scorch.STensor` with the inferred (or requested)
output format. Timing of just the `evaluate()` call is optionally captured into
`time_dict["eval_time"]`.

## Putting it together

Here is the complete generic-path trace for
$C_{ik} = \sum_j A_{ij} B_{jk}$ with `A` in CSR and `B` dense:

| Stage | What happens for this SpMM |
|-------|----------------------------|
| **1. Dispatch** | Wrap as STensors; if no prebuilt SpMM resolves, enter the compiler via `einsum("ik,kj->ij", A, B)`. |
| **2. CIN** | `IndexVar`s `i, j, k`; free `{i,k}`, reduction `{j}`; output format inferred `"dd"`. |
| **2b. Scheduler** | Loop order `[i, j, k]`; insert dense workspace `accum_c[k]`; wrap in `Where(producer, consumer)`. |
| **3. CINLowerer** | Iterator analysis emits `A`'s compressed-row pos/crd loops; `Where` → producer fills `accum_c`, consumer copies to `C`; assemble `evaluate` LLIR function. |
| **4. LLIR** | Typed nodes: `ForLoop` (with OpenMP policy), `Assign(accum_c[k], Mul(A_val, B_val), +=)`. |
| **5. Codegen** | Emit the C++ string with `#pragma omp parallel for` over rows. |
| **6. JIT** | Prepend `header.cpp`; key `kernel_<md5>`; load cached `.so` or `load_inline` with `-O3 -march=native …`. |
| **7. Execute** | `module.evaluate(result_shape, A.shape, A.index.mode_indices, A.values, …)` → `Tensor` → wrap into `STensor("dd")`. |

And here is the same op driven end-to-end from Python, verified against a PyTorch
reference:

```python
import torch
import scorch

# A sparse (CSR), B dense
A = torch.randn(128, 256)
A[A.abs() < 1.0] = 0.0                          # ~68% zeros
As = scorch.from_torch(A.to_sparse_csr(), "A")  # STensor, format "ds" (CSR)
B = torch.randn(256, 64)

# Force the generic compiler with an explicit output format.
C = scorch.einsum("ik,kj->ij", As, scorch.from_torch(B, "B"), format="dd")

ref = A @ B
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)
```

:::{tip}
To watch a real kernel being built the first time (and skipped thereafter), time
two identical calls — the first pays the JIT cost, the second reads the cached
`.so`:

```python
td = {}
_ = scorch.matmul(As, B, time_dict=td)
print(td["eval_time"])   # seconds spent inside the compiled evaluate()
```
:::

## Why so many IRs?

Each lowering exists to isolate one concern, so that no single stage has to reason
about everything at once:

- **CIN** is format- and order-*agnostic* — it captures the mathematics so the
  scheduler is free to choose any legal loop order and any output format.
- The **Scheduler** owns all the *policy* decisions (loop order, workspaces,
  tiling) in one place, driven by a cost model rather than scattered heuristics.
- **CINLowerer** is where *format* enters: iterator analysis and merge lattices
  translate abstract co-iteration into concrete pos/crd arithmetic.
- **LLIR** is a typed, inspectable mirror of the eventual C++ — so **codegen** can
  be a dumb, correct string walk instead of a place where logic hides.
- The **JIT + cache** layer amortizes compilation so the cost is paid once per
  unique kernel and never again.

This separation is what lets Scorch generate a *correct, specialized* kernel for
any format combination while shipping speedups of **1.05–5.80× over PyTorch Sparse
(CGO 2026)** on the combinations it targets.

## See also

- {doc}`/compiler/index_notation` — CIN nodes, the `einsum` grammar, and
  output-format inference.
- {doc}`/compiler/lowering` — the scheduler's cost model, iterator analysis, and
  merge lattices.
- {doc}`/compiler/codegen` — the LLIR-to-C++ walk and OpenMP emission.
- {doc}`/compiler/workspaces` — dense vs COO-hashed reduction buffers and when
  each is inserted.
