# Compiler IR Refactor Handoff

This file is the implementation handoff for the first shippable milestone. The
canonical end-to-end architecture and all migration phases are documented in
[`COMPILER_IR_REFACTOR_DESIGN.md`](COMPILER_IR_REFACTOR_DESIGN.md). Read that
design first when making architectural decisions; use this handoff for the
immediate execution scope and agent prompt.

## Objective

Refactor Scorch's compiler into a production-quality pipeline with immutable
artifacts at pass boundaries, structured intermediate representations, explicit
analyses and scheduling plans, stage verifiers, and pure transformations.

This must be an incremental migration. Do not perform a big-bang rewrite and do
not merely split large files while preserving their current hidden coupling.
Each milestone should be independently reviewable, tested, and behavior
preserving for supported inputs.

Repository: `/Users/bobby/scorch`

Read `/Users/bobby/scorch/AGENTS.md` before making changes. The working tree may
already contain unrelated user changes and untracked research/benchmark files;
preserve them.

## Why This Is P0

The compiler's main maintainability problem is collapsed stage ownership:

- CIN nodes contain hidden mutation, backreferences, and name-based identity.
- Scheduling partly mutates CIN and partly waits until LLIR exists.
- Iteration-lattice construction calls back into `CINLowerer` and mutates its
  phase-dependent private state.
- LLIR is partly structural and partly C++ text embedded in names and raw
  statements.
- Optimization passes rediscover semantics by parsing generated spellings.
- Unsupported or malformed IR sometimes fails open and surfaces only as a C++
  compiler error.

The current effective pipeline is:

```text
mutable CIN
  -> partial CIN scheduling
  -> CINLowerer <-> IterationLattice
  -> LLIR/C++-string hybrid
  -> in-place rewrites
  -> C++
```

The goal is explicit compiler artifacts and contracts, not simply smaller
modules.

## Investment Thesis and North Star

Most production performance currently comes from prebuilt kernels such as the
SpMM variants and their tiling selector. The reason to invest in the generated
path is that schedule innovations such as tile-j and tile-ijk had to become
handwritten C++ kernels because current codegen could not express them
competitively.

The falsifiable north star is: a new schedule is expressed as `LoopPlan` plus
reusable typed passes, without a new algorithm-specific C++ kernel, and reaches
neutral-or-better performance against the handwritten equivalent across the
full codegen-parity grid on both Apple M5 and x86/redwood. One representative
tile-j and one tile-ijk schedule are the initial proof targets. Legacy deletion
is necessary, but it is not by itself the return on this investment.

The north star does not gate cutover. Legacy deletion compares the new pipeline
against legacy generated kernels only; prebuilt parity is validated at the
selector level after cutover (design document Phase 8.5), and a shortfall there
triggers a scope review, never retention of the legacy compiler.

## Concrete Evidence

### CIN ownership and identity

- `src/scorch/compiler/cin.py:18-24` declares class-level mutable defaults such
  as `no_tile_list`; several subclasses do not call `CIN.__init__`, so raw nodes
  can initially inherit shared state.
- `src/scorch/compiler/cin.py:720-735` mutates every referenced `IndexVar` when a
  `TensorAccess` is constructed.
- `src/scorch/compiler/cin.py:422-434` defines `IndexVar` equality and hashing by
  human-readable name rather than stable symbol identity.
- `src/scorch/compiler/cin.py:1012-1024` and `:1040-1053` install mutable parent
  pointers.
- `src/scorch/compiler/scheduler.py:1415` and nearby scheduling code rewrite
  binders and tensor accesses in place.
- `src/scorch/compiler/scheduler.py:1555` and nearby workspace insertion code
  relies on deep copies plus tree surgery.
- `src/scorch/ops.py:1306-1310` explicitly deep-copies because auto-scheduling
  mutates its input.

### Scheduling straddles multiple abstraction levels

- `src/scorch/compiler/scheduler.py:2516-2521` attaches undeclared
  `explicit_schedule`, `panel_bounds`, `relayout_plan`, and `result_tile_plan`
  fields to the CIN root.
- `src/scorch/compiler/cin_lowerer.py:2119-2125` retrieves those fields through
  `getattr`.
- `src/scorch/compiler/cin_lowerer.py:2465-2475` invokes a post-LLIR schedule
  lowering pass from inside CIN lowering.
- `src/scorch/compiler/schedule_lowerer.py:32-69` rediscovers loops and sparse
  coordinate arrays from tags, names, and regexes.

### LLIR is not yet a reliable structured IR

- Many `llir.Var.name` values contain complete C++ array accesses, calls,
  initializer lists, or arithmetic instead of symbol names. Representative
  locations include `src/scorch/compiler/cin_lowerer.py:750-779` and
  `src/scorch/compiler/iterator.py:280-325`.
- `src/scorch/compiler/cin_lowerer.py` contains dozens of `RawStmt` sites and
  many `NO_TYPE` placeholders.
- `src/scorch/compiler/cin_lowerer.py:3517-3577` and nearby passes rewrite
  rendered expressions through string operations.
- `src/scorch/compiler/cin_lowerer.py:4143-4161` renders LLIR back to C++, catches
  all exceptions, and regex-scans the generated text for sparse arrays.
- `src/scorch/compiler/codegen.py:132-136` emits binary expressions without
  preserving AST precedence with parentheses.
- `src/scorch/compiler/codegen.py:110-113` emits literal
  `No code gen implemented...` text for unknown nodes instead of raising.
- `src/scorch/compiler/llir.py:613` contains a second incomplete code generator
  in addition to the active one in `codegen.py`.

### Lowering is a stateful phase bundle

- `src/scorch/compiler/cin_lowerer.py` is 4,777 lines. `CINLowerer` begins around
  line 548 and combines semantic lowering, iteration analysis, result assembly,
  kernel ABI construction, parallel policy, sparse-output strategies, and
  optimization passes.
- `src/scorch/compiler/cin_lowerer.py:2365-2369` runs four in-place LLIR
  optimizations in a hidden fixed order:
  `_insert_sparse_prefetch`, `_hoist_dense_pointers`,
  `_eliminate_single_iteration_loops`, and
  `_hoist_loop_invariant_factors`.
- `src/scorch/compiler/iter_lattice.py:750` stores a `CINLowerer` reference.
- `src/scorch/compiler/iter_lattice.py:456` recursively calls back into lowering.
- `src/scorch/compiler/iter_lattice.py:1040` and nearby code mutates lowerer
  scalar-accumulator state.

### Failure behavior and diagnostics

- `src/scorch/compiler/cin_lowerer.py:826-831` returns an empty list for unknown
  CIN rather than failing.
- `src/scorch/compiler/cin_lowerer.py:607-641` silently ignores unknown post-op
  kinds.
- `src/scorch/compiler/cin_lowerer.py:2132-2144` acknowledges unsupported
  multiple outputs but selects the first one.
- `src/scorch/compiler/codegen.py:342-350` uses `assert [predicate ...]`, which
  checks only that the list is nonempty.
- Compiler invariants rely heavily on `assert`, which disappears under
  `python -O`.

## Chosen Target Design

Use three principal IR levels. Keep analyses and scheduling plans separate from
IR ownership:

```text
OpSpec / EinsumSpec
        |
        v
validated, normalized CIN                 semantic computation
        |
        +----> analyses ----> LoopPlan     immutable side artifacts
        |                         |
        v                         v
iteration lowering ------> LoopIR + plan
                                  |
                                  v
                           ScheduledLoopIR
                                  |
                                  v
                      typed optimization passes
                                  |
                                  v
                    CxxIR / target-specific IR
                                  |
                                  v
                 ABI wrapper -> C++ -> KernelBuildSpec
```

### CIN

CIN describes what is computed:

- tensor accesses and assignments;
- free and reduction variables;
- index semantics;
- semantic dtype and layout information.

CIN must not own parent pointers, use lists, schedule metadata, OpenMP policy,
generated C++ names, or process-global configuration.

Use stable `SymbolId` values. Human names such as `i` and `A` are display
metadata, not identity. Parent relationships, use-def information, free/reduction
sets, and tensor-access collections belong in analysis side tables keyed by IDs.

### LoopPlan

A `LoopPlan` is an immutable scheduling decision referencing stable logical IDs.
It represents loop order, tiling, placement, sparse panels, relayout, result
accumulation, and abstract parallel-loop selection. It is not attached to CIN
through dynamic fields.

The same CIN should be compilable under several plans without copying or
mutating the semantic program.

During migration, use exactly one explicit transitional carrier:

```python
@dataclass(frozen=True)
class ScheduledCIN:
    cin: NormalizedCIN
    plan: LoopPlan
```

`ScheduledCIN` is not a fourth IR. It introduces no nodes, transformations, or
analysis ownership; it only carries normalized CIN and a verified plan through
legacy seams. Delete it once CIN plus `LoopPlan` lower directly to LoopIR.

### LoopIR

LoopIR is the missing abstraction between semantic CIN and target-specific code.
It should structurally represent:

- dense ranges;
- sparse position iteration and coordinate resolution;
- merge/intersection iteration;
- tensor loads and stores with logical provenance;
- workspace allocation and lifetime;
- affine tiles, sparse panels, and relayout regions;
- abstract parallel loops.

Scheduling transforms LoopIR using a `LoopPlan`. Sparse semantics must never be
rediscovered from rendered C++ names.

### CxxIR

The final low-level representation is intentionally C++/Torch/OpenMP-specific;
rename LLIR to `CxxIR` eventually if that remains its role. It needs structured
nodes for symbols, loads, stores, subscripts, member access, calls, casts,
addresses, allocations, blocks, parallel loops, and atomics.

`RawStmt` is a temporary compatibility escape hatch, not the normal
representation. Code emission must be exhaustive and fail closed.

## Immutability Policy

Do not require every construction helper to be frozen. The practical contract
is:

- builders may be locally mutable while constructing a module;
- finalized nodes use frozen dataclasses with tuple children; do not introduce
  an arena in the initial design;
- finalized artifacts passed between stages are immutable;
- passes take an input artifact plus explicit context and return a new artifact;
- transformations rebuild changed paths with `dataclasses.replace`, and unchanged
  nodes may be structurally shared;
- analyses are pure functions over IR that return immutable ID-keyed side tables
  and are recomputed after transformations;
- there is no fingerprint-keyed analysis cache or preserve/invalidate protocol
  unless later stage profiles prove one is materially needed;
- passes never render C++ to recover semantics;
- passes never read environment variables directly.

New nodes created by transformations receive new IDs and provenance such as
`derived_from`. Canonical fingerprints are computed on demand for persistent
kernel-cache identity or explicit differential comparison, not carried eagerly
by every artifact for analysis caching. Serialization should canonically renumber
IDs so hashes do not depend on allocation order.

## Pass and Artifact Contracts

Each compilation should have explicit options and artifacts, for example:

```text
CompileOptions
CompilationContext
Artifact[NormalizedCIN]
AnalysisBundle
LoopPlan
ScheduledCIN
Artifact[LoopIR]
Artifact[CxxIR]
KernelBuildSpec
```

Every pass should identify:

- accepted input IR and produced output IR;
- required analyses;
- legal failure modes;
- verifier run before/after in tests and debug mode;
- a stable name used in diagnostics and stage dumps.

Required analyses are recomputed for the current input artifact. Passes do not
make preservation claims in the initial architecture.

## Verifiers

Add verifier entry points for normalized CIN, LoopIR, and CxxIR. They should
check at least:

- unique and valid symbol/loop IDs;
- no dangling references or shared mutable child lists;
- valid scopes and definitions-before-use;
- tensor rank, access, dtype, and layout consistency;
- sparse parent levels dominating child levels;
- legal reductions and result writes;
- loop bounds, steps, tile placement, and parallel-loop legality;
- workspace allocation, dominance, and lifetime;
- expression types;
- no target-specific C++ strings before CxxIR;
- unsupported output/post-op structures rejected explicitly.

Failures should carry stage, pass, node ID/IR path, and a compact IR fragment.
Distinguish invalid user programs, unsupported valid features, invalid schedules,
and internal compiler invariant failures.

Full internal verifiers run before and after passes in tests, debug mode, and
curated shadow jobs. They are disabled by default in production/release JIT
compilation. Cheap public-input validation, exhaustive node dispatch, and
fail-closed unsupported-feature checks remain enabled in all modes.

## Performance Gates and Latency Monitoring

Do not defer performance attribution to final cutover. The canonical protocol is
in the design document; its binding summary is:

- Phase 0 records p50/p95 Python compiler latency from validated operation to
  `KernelBuildSpec`, excluding external C++ compilation and execution. The
  default target and investigation threshold is production-mode p50 and p95 no
  higher than 1.10 times baseline for every curated corpus category. Correctness
  and generality take priority: a modest, measured, stage-attributed exceedance
  may be accepted for general typed infrastructure, but material, unexplained,
  or compounding regressions may not. Never weaken validation or introduce
  format-, expression-, or kernel-specific shortcuts merely to hit the target.
  Empty pass/artifact plumbing is measured separately and may add at most 1 ms
  at p95.
- Phase 0 archives the full generated-kernel codegen-parity baseline on Apple M5
  and x86/redwood, together with a same-binary A/A control run per machine that
  calibrates the noise floor. Pass/fail is defined against that control, not
  fixed constants: a cell fails when its new/old median ratio exceeds its A/A
  control band, a machine fails when its ratio geomean leaves the A/A geomean
  band, and thresholds must let the A/A control itself pass the gate. There is
  no rerun-until-it-passes; a rerun requires a diagnosed measurement defect and
  replaces an entire grid run, never individual cells. This is the tiling
  selector's v2-vs-v2 noise-floor methodology.
- Prefetch insertion, dense-pointer hoisting, single-iteration elimination,
  invariant-factor hoisting, and parallel zero-fill each land with structural
  activation tests and, where emission differs, full-grid two-machine
  before/after benchmarks in the same PR that extracts or changes them.
  Byte-identical C++ across the grid corpus under identical build flags waives
  the runtime benchmark, because identical binaries can only produce noise; the
  benchmark is mandatory whenever any cell's emitted code differs.
- Debug verifier cost and curated shadow-mode wall time are measured separately.
  Neither is enabled in the ordinary production JIT or correctness suite.

The sparse-prefetch Phase 2 seam at `c8af101` establishes the intended use of
this policy. It has detached ownership, fail-closed general traversal, structural
activation coverage, exact generated-byte preservation, 42/42 byte-identical
grid inputs on M5 and redwood, and isolated sparse-pass manager overhead of
1.709 microseconds p95. End-to-end latency samples were contaminated by a host
defect that also made an unchanged `14b110b` control miss its own archive. The
seam is accepted on correctness, generality, byte identity, and isolated timing;
a quiet-host latency rerun is useful monitoring, not a reason to specialize or
weaken the pass.

The dense-pointer Phase 2 seam is implemented by `ce5adad` and routed in
production by `81b847a`. The version-1 `hoist_dense_pointers` pass takes an exact
statement-list artifact plus a frozen tuple snapshot of the lowerer's current
value-array/C-type mapping, validates and detaches the complete LLIR tree through
the common traversal boundary, and preserves the characterized narrower legacy
analysis and rewrite scopes. Production order is sparse prefetch, managed dense
pointer hoisting, inline single-iteration elimination, inline invariant-factor
hoisting, result/ABI assembly, and managed dynamic-vector rewriting. The focused
common suite is 317 tests and the schedule/codegen matrix remains 82 tests. The
CSR-by-dense, DS, and DSS anchors remain byte-identical at 2,505, 7,117, and 8,660
bytes with their recorded hashes; all 42 M5 and all 42 redwood grid build inputs
are byte-identical to the `c8af101` references, so compiled-kernel A/B comparison
is waived while structural activation remains covered. In a clean committed
worktree, empty-manager p95 was 1.542 microseconds and dense one-pass incremental
manager overhead was 1.958 microseconds. The unchanged `14b110b` latency
preflight again missed its archive (small-dense p95 1.108x and reduction p95
1.170x), so no candidate end-to-end sample was taken from that invalid host
comparison. The remaining sequential inline optimizations are
`_eliminate_single_iteration_loops` and `_hoist_loop_invariant_factors`; extract
only the former next.

## Incremental Migration Plan

### Milestone 0: safety and characterization

1. Add focused tests for current supported CIN, schedules, and generated code.
2. Make unknown CIN, LLIR, and post-op nodes fail closed.
3. Fix precedence-aware C++ expression emission.
4. Fix ineffective assertions such as `assert [predicate ...]`.
5. Add an initial CIN/CxxIR verifier around currently representable invariants.
6. Record the compile-latency corpus and the two-machine generated-kernel
   baseline, including the same-binary A/A noise-floor control runs, used by
   every later phase.

### Milestone 1: explicit ownership contract

1. Ensure every CIN instance owns its state; remove class-level mutable fallbacks.
2. Introduce stable symbol identity without changing human-readable output.
3. Introduce `LoopPlan` and the exact transitional
   `ScheduledCIN(normalized_cin, loop_plan)` carrier to replace the four dynamic
   schedule attributes.
4. Preserve the public `Schedule` API and supported generated C++ behavior.

### Milestone 2: real pass infrastructure

1. Add a common structural walker/rewriter.
2. Extract the four sequential LLIR optimizations from `CINLowerer` into focused
   passes.
3. Add structural pass tests: no-op, positive, malformed input, idempotence where
   appropriate, and verifier-before/verifier-after.
4. Move `ResultTensorAssembler` and kernel prologue/epilogue construction into a
   dedicated Torch/C++ ABI builder.
5. Add stage timing and evaluate production p50/p95 against the default latency
   target, recording attribution for any accepted modest exceedance; keep the
   pure analysis runner cache-free.
6. Run the phase-local generated-kernel gate in the PR for each extraction that
   changes emitted code.

### Milestone 3: structural low-level IR

1. Introduce structured access, member, call, and allocation nodes.
2. Migrate string-encoded `Var.name` expressions incrementally.
3. Remove regex/string rewriting from each migrated pass.
4. Keep only a measured and documented `RawStmt` compatibility budget.
5. Extract parallel zero-fill into a typed pass with structural activation tests
   and same-PR A/B benchmarks.
6. Keep the generated-kernel gate green and compile latency inside the default
   target or an explicitly reviewed modest exception.

### Milestone 3.5: sparse LoopIR feasibility spike and go/no-go

This spike must complete before freezing or productionizing the LoopIR schema:

1. Hand-author candidate LoopIR for CSR SpMV intersection.
2. Hand-author candidate LoopIR for sparse elementwise-add union, including
   exhausted iterators, overlapping support, and one-sided coordinates.
3. Implement a deliberately small test/debug LoopIR interpreter independent of
   the C++ toolchain.
4. Differential-test empty rows, empty inputs, unequal/ragged sparsity, and random
   patterns against PyTorch or a simple reference.
5. Prove that merge progress, coordinate resolution, sparse dominance, and output
   assembly need no callbacks into the general lowerer, rendered-name parsing,
   target syntax, or operation-specific escape hatch.

Then conduct an explicit go/no-go review. Proceed only if both sparse cases pass,
the interpreter remains a simple independent oracle, Phase 0-3 generated-kernel
gates are green, compiler-latency measurements contain no material or unexplained
regression, and the tile-j/tile-ijk parity goal remains credible. If not, retain
the Phase 0-3 hardening, revisit the schema or investment case, and do not force
Phases 4-8 forward.

### Milestone 4: LoopIR strangler path

1. Revise the LoopIR schema from the successful sparse spike, then introduce it
   for one dense operation.
2. Add a CSR-by-dense path with explicit sparse position/coordinate iteration.
3. Promote the spike interpreter into the required test/debug semantic oracle.
4. Shadow-compile only on a curated debug/CI or scheduled two-machine matrix and
   compare stage dumps, generated code, and numerics; keep shadow mode off in
   production and ordinary pytest.
5. Migrate affine tiling, then sparse panel tiling, relayout, workspaces, and
   parallel lowering.
6. Re-express representative tile-j and tile-ijk schedules as `LoopPlan` plus
   typed passes and validate them for correctness through the new pipeline.
   Selector-level validation against handwritten prebuilt equivalents is the
   post-cutover Phase 8.5 milestone and does not block step 7.
7. Delete `schedule_lowerer.py` and phase-dependent lowerer state only after all
   supported paths have migrated.

## Non-Goals and Guardrails

- Do not rewrite the entire compiler in one change.
- Do not combine this work with new kernel optimization experiments.
- Do not change public tensor mutation/return contracts as part of the compiler
  IR refactor unless separately approved.
- Do not clean all existing mypy/formatting debt in the same PR.
- Do not use generated C++ text as an analysis representation.
- Do not identify symbols solely by human names.
- Do not attach undeclared fields dynamically to IR nodes.
- Do not make passes read environment variables or global tuning state.
- Do not add analysis caching or preserve/invalidate declarations without a
  separately profiled and approved design.
- Do not freeze the LoopIR schema before the sparse-union feasibility spike.
- Do not enable unrestricted shadow compilation in production or ordinary
  correctness tests.
- Treat concurrent compilation as a future nice-to-have, not a migration gate.
- Do not delete the legacy path until the replacement has differential coverage.

## Testing and Verification

Use the configured `scorch` conda environment:

```bash
conda run -n scorch pytest -q \
  tests/test_scorch/test_cin.py \
  tests/test_scorch/test_scheduler.py \
  tests/test_scorch/test_schedule_api.py
```

For changes affecting schedule lowering or generated code, also run:

```bash
conda run -n scorch pytest -q \
  tests/test_scorch/test_schedule_generality.py \
  tests/test_scorch/codegen/test_codegen_perf_optimizations.py \
  tests/test_scorch/codegen/test_tuner_schedule_codegen.py
```

Run Black and lint/type checks on changed files. The repository currently has
pre-existing global quality failures, including substantial mypy debt, so report
the distinction between new failures and the existing baseline. Do not run the
unmarked large performance tests as part of ordinary refactor verification.

Useful test additions:

- scheduling does not mutate its input CIN;
- compiling one CIN under two schedules is independent and deterministic;
- stable IDs distinguish same-named variables in different scopes;
- every pass verifies before and after in test/debug configurations;
- unknown nodes fail at the owning stage, not during C++ compilation;
- structural pass assertions replace most generated-string substring checks;
- differential execution remains equal to PyTorch for supported cases.

## Recommended First Implementation Task

The first implementation should be deliberately smaller than the complete
architecture:

1. Add fail-closed codegen/CIN behavior and precedence-correct expression
   emission with focused tests.
2. Introduce stable `SymbolId`/`IndexId` values for every logical entity
   referenced by `LoopPlan`; do not use display names or Python object identity
   as temporary schedule identity.
3. Introduce `LoopPlan` plus a frozen transitional
   `ScheduledCIN(normalized_cin, loop_plan)` carrier replacing the four dynamic
   schedule attributes while preserving the existing public `Schedule` API.
4. Add a minimal verifier for the invariants touched by that boundary.
5. Demonstrate that scheduling does not mutate its input, or characterize and
   isolate any remaining mutation needed for a later milestone.
6. Establish the Phase 0 production-mode compile-latency corpus and the existing
   two-machine generated-kernel baseline without changing kernel behavior.

Do not introduce the complete LoopIR in this first change.

## Copy-Paste Prompt for a New Session/Agent

```text
You are working in /Users/bobby/scorch on a production-quality compiler IR
refactor. Before taking action, read these files in order:

1. /Users/bobby/scorch/AGENTS.md
2. /Users/bobby/scorch/COMPILER_IR_REFACTOR_DESIGN.md
3. /Users/bobby/scorch/COMPILER_IR_REFACTOR_HANDOFF.md

The architectural goal is an incremental pipeline with immutable artifacts at
pass boundaries:

validated/normalized CIN + analyses -> LoopPlan -> LoopIR -> scheduled LoopIR ->
typed optimization passes -> target-specific CxxIR -> ABI/codegen.

Binding architecture decisions:

- IR nodes are frozen dataclasses with tuple children and ID-keyed side tables.
- LoopPlan references stable SymbolId/IndexId values, never display names or
  Python object identity.
- Analyses are pure and recomputed; do not implement analysis caching or
  preserve/invalidate machinery.
- ScheduledCIN is only the frozen transitional pair
  ScheduledCIN(normalized_cin, verified_loop_plan), not another IR.
- Full internal verifiers run before/after passes in test/debug mode and are off
  by default in production JIT compilation; cheap public validation remains on.
- Production LoopIR cannot begin until the mandatory CSR SpMV intersection plus
  sparse elementwise-union interpreter spike passes the Phase 3.5 go/no-go gate.
- Preserve the concrete compile-latency measurement/review policy and
  two-machine kernel-performance gates in the design document. Treat 1.10 as the
  default latency target and investigation threshold, not a reason to compromise
  correctness or generality.
- Cutover compares the new pipeline against legacy generated kernels; parity
  with handwritten prebuilt kernels is a separate post-cutover milestone
  (Phase 8.5) and never gates legacy deletion.
- Kernel-benchmark pass/fail is calibrated against same-binary A/A noise-floor
  controls, never fixed ratio constants; byte-identical emission across the
  grid corpus waives runtime benchmarks, while structural activation tests are
  never waived.

Do not perform a big-bang rewrite. Do not merely split CINLowerer. Preserve
unrelated user changes and inspect git status before editing.

Implement the first shippable milestone described in the handoff:

1. Make CIN/LLIR/codegen fail closed for unknown nodes and post-ops.
2. Fix precedence-correct C++ expression emission and ineffective assertions,
   with focused regression tests.
3. Introduce stable SymbolId/IndexId values for every logical entity referenced
   by LoopPlan. Do not use display names or Python object identity as temporary
   schedule identity.
4. Introduce LoopPlan and the exact frozen transitional
   ScheduledCIN(normalized_cin, verified_loop_plan) carrier that replaces the
   dynamic explicit_schedule, panel_bounds, relayout_plan, and result_tile_plan
   fields currently attached to CIN. Preserve the public Schedule API and
   generated behavior for supported cases.
5. Add a minimal verifier covering symbol/loop references and the new scheduling
   boundary.
6. Add tests proving deterministic behavior and, where feasible, that scheduling
   does not mutate its input. If legacy mutation cannot yet be removed safely,
   isolate and document it rather than hiding it.
7. Record the Phase 0 production-mode compile-latency corpus and the existing
   generated-kernel baseline with its same-binary A/A noise-floor control runs.
   Do not change kernel policy while establishing the baseline.

Keep the change narrowly scoped. Do not introduce the full LoopIR yet, change
public STensor semantics, add kernel optimizations, or attempt to clear all
existing typing debt. Do not add an analysis cache or analysis invalidation
protocol.

Use apply_patch for edits. Run focused tests in the scorch conda environment,
including test_cin.py, test_scheduler.py, test_schedule_api.py, and relevant
schedule/codegen tests. Report exact commands/results, remaining legacy mutation,
and the next recommended migration seam. Do not claim completion unless the
supported paths are verified and no required work remains. Full verifiers are a
test/debug requirement, not production per-pass overhead.
```
