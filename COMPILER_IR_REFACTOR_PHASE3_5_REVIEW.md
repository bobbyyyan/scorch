# Phase 3.5 Review: Sparse LoopIR Feasibility Spike and Go/No-Go

Date: 2026-07-22 (America/Los_Angeles)

Session commits under review (branch
`refactor/compiler-ir-phase3-std-move-call`, stacked on the Phase-3 closure
at `34a1849`; nothing pushed):

- `71eba38` — `feat(compiler): prototype sparse LoopIR schema and verifier`
- `1e30817` — `feat(compiler): interpret sparse LoopIR programs`
- `1c78633` — `test(compiler): execute sparse LoopIR feasibility cases`

Evidence ledger:
`/Users/bobby/.cache/scorch-codex/phase35-loopir-spike-1c78633/`.

## Verdict

**GO**, with the explicit conditions in the final section. The verdict is
based on the criterion-by-criterion evaluation below, not on the test count:
the Limitations section records exactly what the passing suites do *not*
demonstrate, and the GO stands only together with the listed Phase-4
revision obligations. GO authorizes starting Milestone 4's schema revision;
it does not freeze this schema, promote the spike into production, or start
any Phase-4 work inside this session.

## What the spike is

`src/scorch/compiler/loopir_spike/` is a strictly experimental, deliberately
small package (six modules, 1,879 lines including docstrings):

- `nodes.py` (342 lines) — frozen, tuple-owned generic IR nodes: `Block`,
  `DenseFor`, `SparseCursorDecl`/`SparseFor`, `MergedSparseFor` with UNION
  and INTERSECTION modes, `Load`/`CursorValue`/`IndexValue`/`DimSize`/
  constants, `BinaryExpr`, `DeclAccum`/`Accumulate`/`AccumValue`, `Store`,
  `AppendEntry`, `TensorDecl`, `LoopProgram`. Constructors validate
  nothing; semantics are documented on the owning node.
- `verifier.py` (700 lines) — `verify_program`, the fail-closed authority
  (stable defect codes plus lexical paths).
- `csr.py` (136 lines) — the canonical plain-Python CSR container.
- `interp.py` (436 lines) — the plain-Python interpreter (verifies, then
  executes; Torch-free).
- `programs.py` (246 lines) — the three hand-authored feasibility programs.

Identity reuses production `SymbolId`/`IndexId`; `LoopNodeId` and `CursorId`
are spike-local because production `NodeId`/`AccessId` are documented as CIN
identities and this schema is explicitly revisable. Nothing in production
compilation, JIT, public APIs, caches, LLIR, or codegen imports the package;
that is enforced by automated tests, not convention (see Neutrality below).

## Milestone criteria

### 1. CSR SpMV executes through the generic schema

`build_csr_spmv_program()` is a dense outer loop over `DimSize(A, 0)`, a
scalar `ReduceOp.ADD` accumulator, one `SparseFor` cursor over `A`'s
compressed level with the row bound through `outer_indices`, a
coordinate-addressed `Load` of the dense vector at the bound sparse
coordinate, and a dense `Store`. It passes dedicated empty/ragged/zero-shape
cases and a 5-seed x 4-shape x 4-density randomized grid, exactly equal to a
pure-Python dense reference (the accumulation order matches the reference
term-for-term on stored entries, and adding a zero term never changes an
IEEE partial sum, so exact equality is the honest comparison).

This case has exactly **one** sparse cursor. It demonstrates sparse
position iteration, coordinate binding, reduction scoping, and dense output
writes. It is *not* evidence about sparse-sparse intersection and is not
claimed as such.

### 2. CSR + CSR elementwise addition via UNION

`build_csr_union_add_program()` uses one generic `MergedSparseFor` in UNION
mode over two cursors, appending `CursorValue(cA, default 0.0) +
CursorValue(cB, default 0.0)` at every candidate coordinate. Covered
differentially: disjoint, identical, and partially overlapping support;
one-sided rows; one-sided early exhaustion (one cursor drains while the
other continues emitting); unequal row lengths; empty operands and
zero-row/zero-column shapes; cancellation to an explicit stored zero; and a
5-seed randomized shape/density grid — all exactly equal to the dense
reference. Exhausted iterators, overlapping support, and one-sided
coordinates are therefore all exercised, which is what the milestone asked
this case to prove.

### 3. Two-CSR INTERSECTION genuinely synchronizes two cursors

`build_csr_intersection_multiply_program()` reuses the **same**
`MergedSparseFor` node in INTERSECTION mode — no operation-specific node
was added. The body runs only when both cursors carry the candidate
coordinate; defaults are statically rejected in this mode because they are
unobservable. Coverage: disjoint support (empty result), nested support,
one side empty, early exhaustion, a randomized grid, and a
structural-not-value test (an explicitly stored zero intersects; the merge
synchronizes on stored coordinates, not on values). The union and
intersection differences — body-emission policy and termination — are the
node's `mode`, and the interpreter implements both from one merge loop,
which is the concrete evidence that two-cursor synchronization is generic
in this schema.

### 4. Differential and adversarial coverage

297 focused tests: 70 verifier-boundary tests (every defect code is
exercised from at least one adversarial construction, including forged
cycles via `object.__setattr__`, unknown subclasses, non-tuple children,
bool/int and int/float confusions, forged enum lookalikes and identity
values, signed-zero reduction identities, and both sides of the nesting
bound), 221 execution/differential tests, and 6 automated neutrality
checks.

### 5. No callbacks, no rendered names, no target syntax, no escape hatch

- **Merge progress** is intrinsic to `MergedSparseFor`: each step advances
  every aligned cursor by one position, so the loop terminates after at
  most the sum of segment lengths (the remaining-position sum strictly
  decreases). Neither the verifier nor the interpreter consults the
  lowerer, and nothing recognizes merge state from strings.
- **Coordinate resolution** is the minimum over non-exhausted cursors'
  current coordinates, bound to the loop's `IndexId`; consumers read it via
  `IndexValue` like any dense loop index.
- **Sparse dominance** is structural: a cursor over level `L` must bind
  coordinates for all levels above it (`outer_indices`, arity-checked), and
  compressed values are reachable only through `CursorValue` — coordinate
  `Load` on a compressed tensor is a verifier error, so nothing can bypass
  position iteration to "search" a sparse level.
- **Output assembly** is the ordered `AppendEntry` stream; the canonical
  container is built from coordinates alone. No positions arrays, offsets,
  or target storage arithmetic appear in any node.
- The automated neutrality suite enforces the import closure (stdlib +
  `identity` only, proven in a subprocess without Torch), that production
  never references the spike, that `import scorch` never loads it, and
  that the sources contain no target-syntax tokens.

## Design-criterion discussion

**Genericity.** The schema has no operation nodes: SpMV, union add, and
intersection multiply differ only in program shape, merge mode, and the
`BinaryOp`/`ReduceOp` members they use. `MergedSparseFor` accepts any
cursor count >= 2 and both modes with one semantics definition. The
CursorValue default rule (required exactly in UNION, forbidden elsewhere)
turns the union/intersection asymmetry into a typing rule instead of an
escape hatch. The genericity claim is bounded by what was built — see
Limitations.

**Readability.** The schema reads well; hand-authored programs do not.
Explicit `node_id` threading makes `programs.py` verbose (a ~40-line loop
nest for SpMV). That is acceptable for a spike whose programs are written
once, but Phase 4 must add a small builder layer before anyone authors
LoopIR by hand at scale. This is an ergonomics debt, not a schema defect:
the verbosity lives entirely in construction, not in the node definitions.

**Termination.** Dense loops are bounded by evaluated extents (negative
extents fail closed); sparse loops by finite validated segments; merges by
the strict decrease argument above. The verifier is cycle-guarded and
depth-bounded (64 levels, controlled diagnostic), so adversarial structure
cannot make verification diverge, and the interpreter only runs verified
programs.

**Interpreter independence.** 436 lines, plain Python containers, no Torch,
no compiler-pipeline imports, no environment reads, no caching. It is
executable documentation of the node semantics and is fit to become the
Phase-4 semantic oracle (the milestone's step 3), subject to the same
revision as the schema.

**Output assembly.** The append contract (lexicographically strictly
increasing full coordinates per output) is enough for canonical CSR
assembly in every fixture, fails closed on disorder and duplicates, and
keeps explicit zeros (documented: UNION cancellation stores an exact zero;
merging is structural). This matches how the current compiler's ordered
result emission behaves and needed no target syntax.

## Phase 0-3 gates at the spike commits

- **Generated-kernel byte gates.** The 20-source corpus and 42-source grid,
  regenerated from clean detached worktrees at base `34a1849` and candidate
  `1c78633` with isolated caches and asserted import provenance, are
  byte-identical to each other and to the retained Phase-3 final captures
  (`phase3-review-fix.tyJNVF/{corpus,grid}-final`). The 124-file SHA-256
  manifest is `CAPTURE_SHA256SUMS`. This is the expected outcome of a
  not-imported experiment, and it held.
- **Compiler latency.** Paired 5-warmup/30-sample run (base then candidate,
  same session): small_dense `0.922/0.830`, reduction `1.013/0.975`,
  csr_intersection `0.961/0.988`, sparse_union `0.932/0.896` p50/p95 —
  inside the 1.10 target everywhere.
- **Full suite.** The complete non-performance suite from the clean
  detached candidate worktree (import provenance asserted for `scorch`,
  `tests`, `tools.benchmark_compiler_ir`, and the spike; isolated
  `TORCH_EXTENSIONS_DIR`/`XDG_CACHE_HOME`/basetemp) passes: **2,857
  passed, 14 skipped, 3 deselected (perf-marked), 1 warning in 2,071.23 s**,
  exit 0 (junit: 2,871 collected, 0 failures, 0 errors).
- **Static checks.** Black, Flake8, and mypy over full `src` are in exact
  parity between base and candidate: the identical nine inherited Flake8
  findings, the identical 146 inherited mypy errors in 12 files, and the
  identical single Black finding (`prebuilt_kernels.py`) on both sides —
  the six spike files add zero findings to any tool. `git diff --check`
  was clean before every commit.
- **Identity/CIN/LoopPlan suites.** `test_cin.py`, `test_cin_analysis.py`,
  `test_loop_plan.py`, and `test_llir_string_budget.py` pass from the
  detached candidate worktree (125 tests), so the stable-identity, analysis,
  plan, and RawStmt/AddressOf/Var budget locks are untouched — as expected,
  since no production source was modified.

## Is tile-j/tile-ijk parity still credible?

Credible, with named risks. What the spike adds to the argument: the
sparse-iteration substrate that tiling transforms must preserve — position
iteration, two-cursor merges, coordinate binding, ordered assembly — is now
demonstrated to be structurally representable and executable without any
lowerer callback or rendered-name recovery, which was precisely the
capability Phase 3.5 was gating on. Loop plans already carry tile-j/tile-ijk
decisions as immutable ID-keyed artifacts (`LoopPlan` since Phase 1), and
the production tile-j/tile-ijk kernels iterate the same CSR structures the
spike's cursors model, with tiling changing loop bounds, operand staging,
and accumulation placement rather than the sparse-merge semantics.

What the spike deliberately does not show, and Phase 4 must: affine tile and
panel/relayout region nodes (`OperandRelayout`-class staging), workspace
allocation/lifetime as IR structure rather than a scalar accumulator,
abstract parallel loops, and — the largest open risk — the CIN-to-LoopIR
lowering that must produce these programs from the existing iterator
analysis. None of these contradicts the schema shape; all of them are
unbuilt. The parity goal therefore remains credible as a plan, not proven
as an artifact, and Milestone 4 step 6 (re-expressing representative tile-j
and tile-ijk schedules) remains the checkpoint where credibility converts
to evidence or the investment case is revisited.

## Limitations (what passing tests must not be read as)

1. The interpreter executes rank-1/rank-2 dense tensors and two-level
   (dense, compressed) CSR only; the schema admits more (any compressed
   level with bound outer levels), but nothing beyond the spike's patterns
   has been executed. Cursors at level 0 (DCSR-style outer sparsity) and
   three-level formats are unexecuted schema surface.
2. Merges synchronize cursors within one dense-row context. Multi-level
   sparse-sparse iteration (merging at an outer compressed level while
   iterating inner levels) is representable in principle but untested.
3. No CIN-to-LoopIR lowering exists. Feasibility of *generating* these
   programs from `iter_lattice`/`cin_analysis` products — the Phase-4
   strangler path — is not evidenced by this spike, only unblocked by it.
4. Nothing here measures generated-code performance; the byte gates prove
   non-interference, not that LoopIR-compiled kernels will match the
   current emitter. That remains gated by Phase 4's shadow comparisons.
5. The value system is scalar `float` with three binary and two reduction
   operators; dtype/layout generality is a schema-revision item.
6. Workspaces, parallel loops, tiles, panels, and relayout regions are
   absent by design; their addition is exactly the "revise the schema from
   the spike" step Milestone 4 begins with.

## Conditions attached to GO

1. Phase 4 begins by revising this schema (builder ergonomics, workspace
   and parallel-loop nodes, tile/panel/relayout regions, dtype generality,
   multi-level cursor execution) — the spike schema is input, not contract.
2. The spike interpreter is promoted to the Phase-4 semantic oracle only
   together with that revision, keeping its independence properties (the
   neutrality suite must keep passing as it evolves).
3. The package stays out of production imports until the Milestone-4
   strangler path deliberately introduces LoopIR behind its own gates; the
   automated isolation tests are the enforcement and must not be weakened.
4. Milestone 4 step 6 (tile-j/tile-ijk re-expression) remains a hard
   checkpoint: if those schedules cannot be expressed as `LoopPlan` plus
   typed passes over the revised LoopIR, the parity claim above is void and
   the investment case must be revisited rather than forced.

## Evidence index

Under `/Users/bobby/.cache/scorch-codex/phase35-loopir-spike-1c78633/`:
`corpus-{base,candidate}/`, `grid-{base,candidate}/`, `CAPTURE_SHA256SUMS`,
`latency-{base,candidate}.{json,log}`, `latency-compare.txt`,
`static/{black,flake8,mypy}-{base,candidate}.log`, `run_full_suite.sh`,
`import-provenance.txt`, `full-suite.{log,xml}`, and the detached worktrees
under `worktrees/`.
