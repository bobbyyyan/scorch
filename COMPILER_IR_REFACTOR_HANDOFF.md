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
comparison. At that point, the remaining sequential inline optimizations were
`_eliminate_single_iteration_loops` and `_hoist_loop_invariant_factors`.

The single-iteration Phase 2 seam is implemented by `0d31882` and routed in
production by `265123a`. The version-1 `eliminate_single_iteration_loops` pass
takes an exact statement-list artifact and a frozen context containing only its
common-traversal identity. It validates and detaches the complete LLIR tree
before applying the characterized post-order analysis and its distinct legacy
generated-string rewrite scope. Legal misses and normal repeated application
return fully detached output; unknown subclasses, malformed typed children,
wrong roots, and mismatched artifacts, descriptors, contexts, or specifications
fail closed. Production order is managed sparse prefetch, managed dense-pointer
hoisting, managed single-iteration elimination, inline invariant-factor
hoisting, result/ABI assembly, and managed dynamic-vector rewriting. Applied
compressed output still precedes those passes with compressed-Where/OpenMP plus
independent count and fill result-write records.

The focused common suite is now 380 tests and the schedule/codegen matrix remains
82 tests. CSR-by-dense, DS, DSS, and the structurally activating all-COO SDDMM
remain byte-identical at 2,505, 7,117, 8,660, and 3,543 bytes with SHA-256
digests `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151`,
`d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f`,
`1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e`,
and `de94b08752077a621c5e411ce0dcbb40e8bcbeacb9bce3824dd6019e2d2bd29d`.
All 42 M5 and all 42 redwood grid build inputs are byte-identical to the
`81b847a` references, so compiled-kernel A/B comparison is waived while the
all-COO activation remains structurally covered.

In a clean committed worktree, empty-manager p95 was 1.542 microseconds and
single-pass incremental manager overhead was 1.917 microseconds. A fresh
unchanged `14b110b` preflight passed its archived same-commit control before the
candidate was sampled. Against the Phase 0 archive, candidate p50/p95 values in
milliseconds were 1.029/1.129 for small dense, 0.954/1.021 for reduction,
1.009/1.087 for CSR intersection, and 0.974/1.073 for sparse union. Their
respective Phase 0 p50/p95 ratios were 1.127/1.017, 1.108/1.116, 1.132/1.070,
and 1.130/1.192, crossing the 1.10 investigation target in several categories.
Direct production-record timing attributes 22.980-68.541 microseconds p50 and
24.000-71.500 microseconds p95 to the new pass across the corpus. This is a
modest absolute, measured exception accepted for the general fail-closed
detachment and ownership boundary; no validation was weakened and no corpus-
specific shortcut was introduced.

The loop-invariant-factor Phase 2 seam is implemented by `112a9b7` and routed
in production by `f2bca1b`. The version-1
`hoist_loop_invariant_factors` pass takes an exact statement-list artifact and a
frozen context containing only its common-traversal identity. It validates and
detaches the complete LLIR tree before applying the characterized legacy
post-order transform and its distinct whole-loop defined-variable analysis.
Legal misses return fully detached output; unknown subclasses, malformed typed
children (including children in semantically omitted containers), wrong roots,
and mismatched artifacts, descriptors, contexts, or specifications fail closed.
The transform preserves raw substring classification, factor and partition
order, left-associated rebuilding, current-sequence-index `_inv_{i}` naming
without collision checks, first-success-only behavior per loop and invocation,
and the non-idempotent multiple-accumulation repeated-application case.

Production order is now managed sparse prefetch, managed dense-pointer
hoisting, managed single-iteration elimination, managed invariant-factor
hoisting, result/ABI assembly, and managed dynamic-vector rewriting. Applied
compressed output still precedes those passes with compressed-Where/OpenMP plus
independent count and fill result-write records. An invariant-factor failure
preserves the earlier ordered records, adds no factor record, and stops result/
ABI assembly, dynamic-vector rewriting, function construction, scheduling, and
code generation.

The focused common suite is now 455 tests and the schedule/codegen matrix
remains 82 tests. CSR-by-dense, DS, DSS, and the structurally activating all-COO
SDDMM remain byte-identical at 2,505, 7,117, 8,660, and 3,543 bytes with the
same four recorded SHA-256 digests. The all-COO kernel emits exactly
`float _inv_17 = Mask_val[pMask0];`, keeps only
`_Query_val_ptr[q] * _Key_val_ptr[q]` in the q-loop accumulation, and emits
`_accum *= _inv_17;` immediately after that loop. It also retains the managed
single-iteration result: no `pMask1_end = pMask0 + 1` derived-bound statement
or inner `pMask1` loop, and direct use of `Mask1_crd[pMask0]` and
`Mask_val[pMask0]`. (The later exit audit below corrects the earlier mistaken
declaration claim.) All 42 M5 and all 42 redwood grid build inputs are
byte-identical to the `265123a` references, so
compiled-kernel candidate-versus-predecessor comparison is waived on both
machines while structural activation remains required and covered.

In a clean committed worktree, empty-manager p95 was 1.625 microseconds and
invariant-factor single-pass incremental manager overhead was 1.916
microseconds. Direct pass and complete managed-call p95 were 2.083 and 3.917
microseconds, respectively. A fresh unchanged `14b110b` preflight missed its
own same-commit archive: small-dense, reduction, CSR-intersection, and sparse-
union p50/p95 ratios were 1.068/1.136, 1.088/1.155, 1.050/1.074, and
1.057/1.108. Per the settled policy, the control was not rerun
opportunistically and the candidate was not sampled, so there is no valid new
end-to-end compiler-latency result to attribute to this seam.

The four sequential Phase 2 LLIR optimizations are now extracted behind the
typed pass infrastructure with their seam-local ownership, failure, structural,
ordering, timing, and two-machine generated-input gates satisfied. Run the
design-canonical Phase 2 exit audit next before declaring the full phase closed;
after that audit, proceed sequentially to the Phase 3 structured-CxxIR/ABI work.

### Design-canonical Phase 2 exit audit (2026-07-13)

This audit started from exact commit `7205fb1` and inspected the canonical
design, source, focused and integration tests, repository history, generated
source, and the retained clean benchmark archives. It does not change any
binding design decision. The audit fixes before this record are `72b8a59`
(`LoopPlan` boundary hardening), `74c7a96` (partial nested pass records), and
`64a5f1e` (compiler-latency threshold policy in the comparison tool).

No canonical Phase 2 deliverable is marked deferred: each item below is
explicitly assigned to Phase 2 by the design.

#### CompileOptions closure follow-up (2026-07-14)

Commits `7c68ac9`, `d8ae9ed`, and `d4a4cb3` close the isolated canonical
`CompileOptions` deliverable. `compiler/compile_options.py` now owns one exact
frozen snapshot for a compilation. Public `ops.py` and `STensor` compilation
boundaries construct it once when ordinary callers do not provide the internal
optional value, then route that same object through normalization, scheduling,
legacy CIN adaptation, CIN/LLIR lowering, nested compressed-output passes, C++
emission, kernel naming, and the build seam; the detached JIT request carries
the exact `options.build` subobject. Direct scheduler,
CIN-analysis, CIN-lowerer, legacy-adapter, and utility entry points retain
compatibility boundaries that snapshot only when no outer snapshot exists.
Standalone renderer and pass APIs are non-compilation compatibility calls and
do not construct a snapshot. A production stage receiving an explicit snapshot
returns or forwards that exact object and does not reread the corresponding
environment or `ContextVar`.

The exact typed ownership is:

- `CompileOptions(build, requested_schedule, scheduler, verification,
  enabled_llir_passes, regblock_dual, emit_comments)`;
- `SchedulerPolicy(regblock_enabled, regblock_max_n, regblock_tile_width,
  auto_tile_width, cost_model)` and
  `SchedulerCostModel(alpha, beta, gamma, c_insert, c_sort, c_trans, rho,
  default_dim_size)`;
- `VerificationPolicy(verify_cin, llir_pass_options)`;
- `KernelBuildOptions(target_os, target_arch, optimization_level, isa_policy,
  fast_math, unroll_loops, parallel_backend, legacy_parallel_policy,
  legacy_lowering_policy, index_width, abi_policy,
  compiler_abi_check_policy, compiler_wrapper_policy, compiler_wrapper_name,
  compiler_wrapper_path, darwin_toolchain, extra_cflags,
  direct_extension_cflags, special_kernel_cflags, extra_ldflags,
  jit_tune_hooks, torch_version, torch_path, torch_include_paths,
  torch_library_path, torch_cxx11_abi, python_executable, python_version,
  python_cache_tag, python_include_path, scorch_python_path, cxx_compiler,
  cxx_compiler_path, cxx_compiler_from_environment,
  executable_search_path, preamble_source)`; and
- `DarwinToolchainOptions(developer_dir, sdk_root, deployment_target)`.

All collections crossing the boundary are converted to tuples or exact frozen
values, including schedules, pass identities, build/link flags, Torch include
paths, source/function lists in the build request, and scheduler costs.
Unsupported pass sequences, targets, flags, toolchains, ABI policies, compiler
commands, unordered carriers, and conflicting legacy arguments fail
closed with frozen `CompileOptionsDiagnostic` values carried by
`CompileOptionsError`. The exact current seven-pass tuple is represented,
validated, and cache-keyed, but this follow-up deliberately does not claim that
the pass manager now owns an options-assembled pipeline; that remains a
separate canonical blocker below.

##### Configuration inventory: compiler configuration

Unless a row names a different owner, environment keys in this table are
declared and read exactly once by `compiler/compile_options.py`; later files in
the source-location column consume only their frozen typed representation.

| Source/input and source location | Current reader and effect | Frozen typed representation or fail-closed treatment |
| --- | --- | --- |
| `SCORCH_REGBLOCK` (`compiler/compile_options.py`; scheduling policy consumed in `compiler/scheduler.py`) | `CompileOptions.from_environment`; scheduling and generated C++ | `scheduler.regblock_enabled: bool`, default `False`, exact `0`/`1` |
| `SCORCH_REGBLOCK_MAX_N` (`compiler/compile_options.py`; consumed in `compiler/scheduler.py` and `ops.py`) | Same boundary; scheduling and dual-path cutoff | `scheduler.regblock_max_n: int`, default `8`, positive int32 |
| `SCORCH_REGBLOCK_T` (`compiler/compile_options.py`; consumed in `compiler/scheduler.py`) | Same boundary; register-block tile width | `scheduler.regblock_tile_width: int`, defaulting to the snapshotted max-N value |
| `_REGBLOCK_FORCE` / `regblock_force` (`compiler/scheduler.py`) | Compatibility `ContextVar`, read only while constructing a direct-boundary snapshot | Folded once into `scheduler.regblock_enabled`; exact bool or `None` |
| Public `schedule` / internal `_schedule` (`ops.py`) | Public compilation boundary; requested loop order, tiling, relayout, and parallel choice | Detached frozen `requested_schedule: Optional[Schedule]`; conflicts fail closed |
| `_SCHEDULE_FORCE` / `schedule_force` (`compiler/scheduler.py`) | Compatibility `ContextVar`, read only at the boundary | Folded once into `requested_schedule` |
| `Scheduler._DEFAULT_COSTS` and direct optional `costs` (`compiler/scheduler.py`) | Auto-scheduling cost policy | Frozen `SchedulerCostModel(alpha=2.975, beta=0.1005, gamma=43.55, c_insert=85.34, c_sort=1.741, c_trans=40.61, rho=0.0014, default_dim_size=1024)`; conflicting direct arguments fail closed |
| Legacy implicit auto-tile width `32` (`compiler/scheduler.py`) | Auto-scheduling policy | `scheduler.auto_tile_width: int = 32` |
| `SCORCH_VERIFY_CIN` (`compiler/compile_options.py`; verification consumed in `compiler/cin_analysis.py`) | Snapshot boundary; debug verification | `verification.verify_cin: bool`, default `False`, exact `0`/`1` |
| `_VERIFY_CIN_CONTEXT` / `full_cin_verification` (`compiler/cin_analysis.py`) | Debug compatibility `ContextVar`, read only at the boundary | ORed once into `verification.verify_cin` |
| `PRODUCTION_LLIR_PASS_OPTIONS` (`compiler/llir_pass_manager.py`) | Production pass verification/timing policy | `verification.llir_pass_options=(False, False, True)` for before/after/timing |
| `DEBUG_LLIR_PASS_OPTIONS` and direct `llir_pass_options` | Debug pass verification/timing policy | Exact `(True, True, True)`; only the production and debug frozen values are supported |
| Seven manager pass descriptors/order | Enabled compiler behavior | `enabled_llir_passes: tuple[LLIRPassId, ...]`; only the current exact seven-pass tuple is supported |
| `SCORCH_REGBLOCK_DUAL` (`compiler/compile_options.py`; consumed in `ops.py`) | Snapshot boundary; dual-path generated LLIR/C++ | `regblock_dual: bool`, default `True`, exact `0`/`1` |
| `LLIRLowerer.no_comments` / `lower_llir(no_comments=...)` (`compiler/codegen.py`) | Generated C++ spelling and bytes | `emit_comments: bool = True`; a conflicting explicit `no_comments=True` fails closed |
| `SCORCH_JIT_TUNE_HOOKS` (`compiler/compile_options.py`; consumed in `utils.py`) | Snapshot boundary; JIT build flags, runtime-hook enablement, and build/name identity | `build.jit_tune_hooks: bool` and exact matching production/direct flag tuples |
| `CXX` (`compiler/compile_options.py`; build validation in `utils.py`) | Snapshot boundary; compiler identity and build behavior | `cxx_compiler`, absolute `cxx_compiler_path`, and `cxx_compiler_from_environment`; whitespace commands are unsupported |
| `PATH` (`compiler/compile_options.py`; frozen child environment assembled in `utils.py`) | Snapshot boundary; CXX and wrapper resolution | Detached `executable_search_path`; absent uses `os.defpath`, and the child receives only the frozen path |
| `TORCH_NO_COMPILER_WRAPPER` (`compiler/compile_options.py`; fail-closed child revalidation in `utils.py`) | Snapshot boundary; PyTorch ccache/sccache policy | `CompilerWrapperPolicy`, optional wrapper name, and absolute wrapper path; any nonempty value disables, matching PyTorch semantics |
| `shutil.which("ccache")` then `shutil.which("sccache")` | Snapshot boundary; wrapper discovery | First resolved supported wrapper name/path is frozen and build-keyed |
| `TORCH_DONT_CHECK_COMPILER_ABI` (`compiler/compile_options.py`) | Snapshot boundary; production compiler ABI policy | `CompilerABICheckPolicy.REQUIRED`; absent/`0` is accepted, `1` is rejected, malformed values fail closed |
| `CPATH` | Snapshot boundary; would change include search | Presence is rejected as `unsupported_compiler_environment` |
| `CPLUS_INCLUDE_PATH` | Same | Presence is rejected |
| `C_INCLUDE_PATH` | Same | Presence is rejected |
| `OBJC_INCLUDE_PATH` | Same | Presence is rejected |
| `OBJCPLUS_INCLUDE_PATH` | Same | Presence is rejected |
| `LIBRARY_PATH` | Snapshot boundary; would change link search | Presence is rejected |
| `COMPILER_PATH` | Snapshot boundary; would change compiler selection | Presence is rejected |
| `GCC_EXEC_PREFIX` | Snapshot boundary; would change GCC tool lookup | Presence is rejected |
| `CCC_OVERRIDE_OPTIONS` | Snapshot boundary; would rewrite compiler arguments | Presence is rejected |
| `SDKROOT` (`compiler/compile_options.py`; frozen child environment assembled in `utils.py`) | Snapshot boundary; Darwin SDK target | `darwin_toolchain.sdk_root`; only the coherent CommandLineTools SDK is accepted |
| `DEVELOPER_DIR` (`compiler/compile_options.py`; frozen child environment assembled in `utils.py`) | Snapshot boundary; Darwin toolchain root | `darwin_toolchain.developer_dir`; only CommandLineTools is accepted |
| `MACOSX_DEPLOYMENT_TARGET` (`compiler/compile_options.py`; frozen child environment assembled in `utils.py`) | Snapshot boundary; Darwin deployment/build identity | Validated optional numeric `darwin_toolchain.deployment_target` |
| `platform.system()` (`compiler/compile_options.py`) | Snapshot boundary; target policy | `target_os: TargetOS`; only Darwin and Linux are supported |
| `platform.machine()` | Snapshot boundary; target/build identity | `target_arch: str` |
| `shutil.which(CXX, path=PATH)` | Snapshot boundary; compiler resolution | Absolute `cxx_compiler_path`; a missing compiler fails closed |
| Fixed `-O3` | Current optimization policy | `optimization_level=3` and exact flag tuples |
| Fixed `-march=native` | Current ISA policy | `isa_policy=ISAPolicy.NATIVE` and exact flag tuples |
| Fixed `-ffast-math` | Current numerical policy | `fast_math=True` and exact flag tuples |
| Fixed `-funroll-loops` | Current optimization policy | `unroll_loops=True` and exact flag tuples |
| Special-kernel `-fno-signed-zeros` | Special-kernel numerical policy | Exact `special_kernel_cflags: tuple[str, ...]` |
| Fixed OpenMP target choice | Parallel target policy | `parallel_backend=OPENMP` and exact compile/link tuples |
| CommandLineTools SDK/libc++ directory existence | Snapshot boundary; Darwin target availability | Frozen `DarwinToolchainOptions`; missing directories fail closed |
| Torch `libomp.dylib` existence | Snapshot boundary; Darwin OpenMP link selection | Exact `extra_ldflags` and rpath tuple |
| Homebrew `/opt/homebrew` include/lib probes | Snapshot boundary; Darwin OpenMP fallback | Selected paths are frozen in compile/link tuples |
| Homebrew `/usr/local` include/lib probes | Same fallback after `/opt/homebrew` | Selected paths are frozen in compile/link tuples |
| `glob(torch/lib/libgomp*.so*)` | Snapshot boundary; Linux OpenMP link selection | First absolute library/rpath or exact `-fopenmp`, frozen in `extra_ldflags` |
| `torch.__version__` | Snapshot boundary; generated name/cache/build identity | `torch_version: str` |
| `torch.__file__` | Snapshot boundary; Torch installation/build identity | `torch_path: str` |
| Torch include paths derived from `torch.__file__` | Snapshot boundary; build specification | `torch_include_paths: tuple[str, ...]` |
| Torch library path derived from `torch.__file__` | Snapshot boundary; build/link specification | `torch_library_path: str` |
| `torch._C._GLIBCXX_USE_CXX11_ABI` | Snapshot boundary; C++ ABI | `torch_cxx11_abi: bool` |
| `sys.executable` | Snapshot boundary; child interpreter/build identity | `python_executable: absolute str` |
| `platform.python_version()` | Snapshot boundary; Python ABI identity | `python_version: str` |
| `sys.implementation.cache_tag` | Snapshot boundary; Python ABI/cache identity | `python_cache_tag: str` |
| `sysconfig.get_path("include", scheme="posix_prefix")` | Snapshot boundary; Python headers | `python_include_path: absolute str` |
| Scorch root derived from `compile_options.py.__file__` | Snapshot boundary; child import identity | `scorch_python_path: absolute str`, supplied as the child's `PYTHONPATH` |
| Packaged `header.h` (`utils.jit_preamble_text`) | Snapshot boundary; generated translation-unit bytes | Included in immutable `preamble_source` |
| Packaged `native_abi.h` | Same; generated ABI helpers | Included in immutable `preamble_source` |
| Packaged `scorch_policy.h` | Same; generated target/runtime policy | Included in immutable `preamble_source` |
| Optional local `scorch_policy_tuned.h` | Same; local policy overrides | Included in immutable `preamble_source` and build identity when present |
| `SCORCH_GRAIN_DEFAULT=500` source default | Embedded generated-kernel runtime policy | Snapshotted textually in `preamble_source`, not reread from process state |
| `SCORCH_GRAIN_CODEGEN_SPGEMM=1500` source default | Embedded generated-kernel work policy | Snapshotted in `preamble_source` |
| `SCORCH_ROWS_PER_THREAD=16` source default | Embedded parallel policy | Snapshotted in `preamble_source` |
| `SCORCH_CHUNKS_PER_THREAD=7` source default | Embedded parallel policy | Snapshotted in `preamble_source` |
| `SCORCH_CHUNK_MIN=4` source default | Embedded parallel policy | Snapshotted in `preamble_source` |
| `SCORCH_CHUNK_MAX=64` source default | Embedded parallel policy | Snapshotted in `preamble_source` |
| `SCORCH_MEMSET_GRAIN_BYTES=262144` source default | Embedded generated zero-fill policy | Snapshotted in `preamble_source` |
| Fixed int32 index behavior | Generated LLIR/C++ and ABI | `index_width=IndexWidthPolicy.INT32` |
| Fixed Torch C++ extension ABI | Wrapper behavior | `abi_policy=ABIPolicy.TORCH_CPP_EXTENSION` |
| Current legacy lowering behavior | Compiler/output policy | `legacy_lowering_policy=LegacyLoweringPolicy.CURRENT` |
| Current legacy parallel behavior | Compiler/output policy | `legacy_parallel_policy=LegacyParallelPolicy.CURRENT` |
| Production versus direct-extension target flags | Build behavior | `extra_cflags` preserves generated-grid inputs; `direct_extension_cflags` separately adds Darwin `-isysroot` for direct legacy `load_inline` callers |
| PyTorch CPU-extension glue (`-fPIC`, `-std=c++20`, include/lib flags, extension-name macro) | Torch-owned implicit build specification | Not a mutable Scorch input; pinned transitively by the frozen Torch/Python/compiler identities, sources, functions, and name |

The isolated build child rechecks Torch/Python ABI identity,
`TORCH_NO_COMPILER_WRAPPER`, and wrapper availability in
`utils._verify_snapshotted_build_runtime`. These are fail-closed trust-boundary
checks against the minimal environment reconstructed from `KernelBuildOptions`;
they do not reselect compiler configuration or consult the mutated parent
environment.

##### Configuration inventory: fixed behavior and program inputs

| Source/input | Classification and reason it is not a separate `CompileOptions` input |
| --- | --- |
| `CompressedWhereOpenMPPolicy("dynamic, 64", "SCORCH_GRAIN_CODEGEN_SPGEMM")` | Fixed versioned implementation policy with no mutable reader; current parallel policy is represented by `parallel_backend` and `legacy_parallel_policy` |
| Other fixed `dynamic, 16`/`static` OpenMP spellings and parallel eligibility predicates | Fixed current lowering implementation, not independently configurable process state |
| Frozen traversal contexts and pass descriptors/specifications | Fixed pass implementation; enablement/order is in `enabled_llir_passes`, while result names, levels, types, and pointer facts are derived typed artifacts |
| Fixed preamble macros `SCORCH_RESTRICT`, `SCORCH_FORCE_INLINE`, `SCORCH_LIKELY`, `SCORCH_UNLIKELY`, and `SCORCH_PRAGMA_UNROLL` | Source-level implementation spellings with no process reader; their exact definitions are frozen in `preamble_source` |
| `SCORCH_TUNE_HOOKS` preprocessor define | Not read from the environment by compiled code; its inclusion is controlled and validated by `build.jit_tune_hooks` |
| `CINLowerer(filter_zeros=False, post_ops=None)` | Explicit operation/lowering semantic input chosen by entry points; existing post-op ownership is unchanged and is not part of `CompileOptions` |
| Output format, mode order, dtype, shape, layout, and tensor contents | Validated program semantics destined for OpSpec/CIN ownership, not compiler policy |
| Result/workspace names, compressed levels, ctypes, and assembly facts | Derived from the verified program/IR, not ambient configuration |
| `compile_only`, `use_cache`, and timing dictionaries | Frontend execution control or observation, not compiler semantics/output policy |
| `get_extra_cflags(base_flags=...)` custom prefix | Legacy direct-extension helper input, not used by the production JIT; the snapshotted target/OpenMP tail is still applied and the returned list is detached |
| Standalone direct pass/`LLIRLowerer` compatibility calls without options | Their own non-production boundary; every nested production renderer now receives the outer snapshot |
| Compiler ABI/version subprocess probe | Validation of the pinned executable at a trust boundary, not a configuration choice |
| Debug/stage dumps | No current debug-dump environment variable or mutable dump registry exists, so there is no current input to snapshot |

##### Configuration inventory: explicitly outside compiler configuration

The `SCORCH_AUTOTUNE*`, `SCORCH_TILING*`, cache-location, and host-query rows
below are read by `tiling.py`; the native runtime-hook rows identify their C++
header reader explicitly in the first column or exclusion text.

| Source/input and current reader | Concrete exclusion |
| --- | --- |
| `SCORCH_MATCH_HOST_THREADS` / `_MATCH_HOST_THREADS` (`ops.py`) | Module-import snapshot selecting a launch argument for already-built CSR-SpMM; never changes generic compiler IR, emitted C++, or JIT build inputs |
| `SCORCH_ATPARALLEL_PIPELINE` / `_ATPARALLEL_PIPELINE` (`ops.py`) | Module-import snapshot selecting the already-built SpMM's private OpenMP versus Torch-pool launch |
| `torch.get_num_threads()` (`ops.py`) | Per-call prebuilt launch input only |
| `SCORCH_AUTOTUNE` (`tiling.py`) | Initial prebuilt SpMM selector level; `tiling.py` documents that it does not touch the general JIT compiler |
| `SCORCH_TILING` | Legacy mapping to the prebuilt selector's `off`/default level |
| `SCORCH_TILING_PROBE` | Legacy mapping to prebuilt analytic/balanced selection |
| `SCORCH_TILING_DEG_FLOOR` | Prebuilt selector eligibility gate only |
| `SCORCH_TILING_NIJK_MIN` | Prebuilt tile-ijk candidate gate only |
| `SCORCH_LLC_BYTES` | Prebuilt selector LLC input only |
| `SCORCH_TILING_LOC_MIN` | Prebuilt selector locality gate only |
| `SCORCH_AUTOTUNE_MARGIN` | Learned prebuilt selector winner margin |
| `SCORCH_AUTOTUNE_CONFIRM` | Learned prebuilt selector runtime confirmation policy |
| `SCORCH_TILING_CV_NSAMP` | Learned prebuilt selector feature-sampling count |
| `SCORCH_AUTOTUNE_WIDEN` | Learned prebuilt selector gate policy |
| `SCORCH_AUTOTUNE_CACHE` | Prebuilt selector persistent-cache path/disable switch |
| `SCORCH_AUTOTUNE_MODEL` | Prebuilt learned-model path/disable switch |
| `LOCALAPPDATA` (`tiling.py`) | Prebuilt selector cache/model storage only |
| `XDG_CACHE_HOME` (`tiling.py`) | Prebuilt selector cache/model storage only |
| `HOME` via `expanduser` (`tiling.py`) | Prebuilt selector cache/model storage fallback only |
| Tiling `_global_level` | Mutable prebuilt dispatch default only |
| Tiling thread-local `_tls.level` | Scoped prebuilt dispatch override only |
| Tiling `_HAS_TILEJ` | Availability of a symbol in already-built `scorch_ops` |
| Tiling `_HAS_TILEIJK` | Availability of a symbol in already-built `scorch_ops` |
| Tiling `platform.system()` | Prebuilt LLC/cache/model path and fingerprint only |
| Tiling `platform.machine()` | Prebuilt machine fingerprint only |
| Tiling `sysctl`/sysfs `/proc/cpuinfo` queries | Prebuilt LLC and machine fingerprint only |
| Tiling `os.cpu_count()` | Prebuilt machine fingerprint only |
| Tiling `_decision` | Prebuilt runtime dispatch memoization only |
| Tiling `_llc_bytes` | Prebuilt runtime LLC memoization only |
| Tiling `_machine_id_val` | Prebuilt persistent-cache/model fingerprint only |
| Tiling `_LEVELS`, `_LOC_NSAMP`, `_FEATURES`, `_CACHE_VERSION`, and `_LEARNED_VERSION` | Fixed prebuilt selector/model implementation constants only |
| Tiling parsed-policy globals `_DEG_FLOOR`, `_NIJK_MIN`, `_LOC_MIN`, `_LEARNED_MARGIN`, `_LEARNED_CONFIRM`, `_CV_NSAMP`, and `_LEARNED_WIDEN` | Module snapshots of the individually inventoried selector-only environment inputs above; never consumed by the generic compiler |
| Tiling `_cache_loaded`, `_persist_cache`, `_learned_loaded`, and `_learned_model` globals | Prebuilt selector service state only |
| `compiler_schedule_search_space` / `schedule_from_tuner_choice` | Opt-in pure helpers; only a returned explicit `Schedule`, if actually passed to compilation, enters `requested_schedule` |
| `SCORCH_BUILD_TUNE_HOOKS` (`scorch_build.py`) | Install/editable-build configuration for the prebuilt `scorch_ops` extension, outside a per-JIT compilation |
| `SCORCH_TUNE_THREADS` (`scorch_policy.h`) | Execution-time hook inside already-compiled code, only when hooks were compiled in |
| `SCORCH_TUNE_CHUNK` (`scorch_policy.h`) | Execution-time hook inside already-compiled code |
| `SCORCH_CODEGEN_ALLOC` (`header.h`) | Execution-time generated-kernel zero-fill A/B hook; enabling the hook code itself is snapshotted by `jit_tune_hooks` |
| `SCORCH_SPMM_ALLOC` (`spmm.h`) | Runtime A/B control inside the prebuilt SpMM |
| `SCORCH_SPMM_WORKSPACE` (`spmm.h`) | Runtime A/B control inside the prebuilt SpMM |
| `SCORCH_REGTILE_BASE` (`spmm.h`) | Runtime A/B control inside the prebuilt SpMM |
| `SCORCH_SPMM_ATPARALLEL` (`spmm.h`) | Runtime A/B control inside the prebuilt SpMM |
| `SCORCH_NEON_REGTILE` (`spmm.h`) | Runtime A/B control inside the prebuilt SpMM |
| `SCORCH_GRAIN_SPMSPM=3000` source default | Primarily prebuilt runtime policy; its incidental presence in the shared preamble is nevertheless frozen textually |
| `SCORCH_GRAIN_SPMM=150000` source default | Primarily prebuilt runtime policy; its incidental presence in the shared preamble is frozen textually |
| `TORCH_EXTENSIONS_DIR` | Extension storage/cache location only, not emitted source, flags, ABI, or semantic build identity |
| Default Torch cache root derived from HOME/XDG state | Storage only |
| `TMPDIR`, `TEMP`, and `TMP` | Temporary storage only |
| `MAX_JOBS` | Native build concurrency only |
| `NINJA_STATUS` | Native build status presentation only |
| `CC` | The supported CPU JIT path does not consume it; setup/prebuilt builds are a separate boundary |
| `CFLAGS` | Not consumed by the isolated CPU Ninja JIT child |
| `CXXFLAGS` | Not consumed by the isolated CPU Ninja JIT child |
| `CPPFLAGS` | Not consumed by the isolated CPU Ninja JIT child |
| `LDFLAGS` | Not consumed by the isolated CPU Ninja JIT child |
| `ARCHFLAGS` | Not consumed by the isolated CPU Ninja JIT child |
| Ambient `PYTHONPATH` | Ignored; the child receives the snapshotted Scorch root |
| `LD_LIBRARY_PATH` | Native loader state, not compiler configuration; link inputs/rpaths are frozen explicitly |
| `DYLD_LIBRARY_PATH` and other `DYLD_*` loader values | Native loader state only |
| `OMP_*` | OpenMP runtime execution policy only; no compiler reader |
| `MKL_*` | Native library execution policy only |
| `OPENBLAS_*` | Native library execution policy only |
| `SCORCH_SUITESPARSE` | Benchmark dataset location in tooling only |
| `SCORCH_SANITIZER_CXX` | Sanitizer-test compiler selection only, outside production compilation |
| `SCORCH_SANITIZER_LOG` | Sanitizer-test log location only |
| `SCORCH_SANITIZER_RUN` | Sanitizer-test opt-in only |
| `SCORCH_CHECKOUT` | Packaging smoke-test checkout location only |
| Existing `_kernel_cache`, `_einsum_dispatch_cache`, and `utils._so_cache` | Existing execution/build memoization, not configuration; semantic/build keys now include canonical option identity, and no new cache was added |
| `time.time`, `time.perf_counter`, and `perf_counter_ns` | Observation/instrumentation only; run durations are non-semantic |
| Prebuilt dispatch specifications, `_ACT_CODES`, and `_EMPTY_MODE_INDICES` (`ops.py`) | Already-built native symbol routing/argument encoding and a fixed empty native-call argument |
| `_UNRESOLVED_SCHEDULE` (`ops.py`) | Private identity sentinel used only while assembling a boundary snapshot; it is not policy state |
| Special-kernel source file text | An explicit source artifact snapshotted into `_JITBuildRequest.cpp_sources`, not configuration |

The parent boundary reads each of the 22 audited environment keys exactly once.
The focused counting-mapping test proves that property. Other tests replace
`CompileOptions.from_environment` after construction and compile successfully,
assert that public `einsum` invokes it once, and assert exact object identity at
the scheduler, lowerer, nested compressed-output renderers, codegen, and build
seams. Mutation-after-snapshot and two-snapshot tests prove that environment
changes, caller-list changes, and a second configuration cannot affect an
in-flight or completed first compilation. Plain CIN lowering now fails closed
instead of ignoring a requested schedule; the verified `ScheduledCIN` route
continues to accept the same snapshot.

#### Manager-owned pipeline and common analysis-runner closure (2026-07-14)

Commits `8e89822` and `cdb12bc` close the isolated canonical manager-owned
pipeline/common analysis-runner blocker without changing a pass implementation,
pass order, result/ABI assembly, scheduling, generated syntax, or optimization
policy. `LLIRPassPipeline.from_compile_options` constructs one exact frozen
`LLIRPassPipeline(compile_options, pass_ids, pass_descriptors, options)` and
`LLIRPassManager.from_compile_options` retains that pipeline plus the identical
`VerificationPolicy.llir_pass_options` object. `CINLowerer` now has one
`run_production_pipeline` call and no individual manager `run_*` call. The
manager spells out the heterogeneous production composition explicitly; it
does not use generic dispatch, reflection, signature inspection, a pass
registry, or a dictionary-of-`Any` configuration.

The complete pre-implementation orchestration inventory was:

- `CINLowerer.lower_ForAll` called `run_compressed_where_openmp` when the
  compressed-output parallel gate selected the transform. The pass itself ran
  `run_result_write` independently in `count` and `fill` modes over the same
  original work-body list; result write was not a direct `CINLowerer` call.
- `CINLowerer.lower_IndexStmt` then called `run_sparse_prefetch`,
  `run_dense_pointer_hoist`, `run_single_iteration_loop_elimination`, and
  `run_loop_invariant_factor_hoist` over the recursively lowered statement
  list, assembled known-nnz/result storage, ABI validation, prologue, and final
  result statements, and finally called `run_dynamic_vector_access` over that
  assembled body. These six lowerer calls were the complete production
  `LLIRPassManager.run_*` inventory.
- The exact stable seven-identity tuple was, and remains,
  `COMPRESSED_WHERE_OPENMP`, `RESULT_WRITE`, `SPARSE_PREFETCH`,
  `DENSE_POINTER_HOIST`, `SINGLE_ITERATION_LOOP_ELIMINATION`,
  `LOOP_INVARIANT_FACTOR_HOIST`, and `DYNAMIC_VECTOR_ACCESS`. Ordinary
  compilations realize records in sparse, dense, single-iteration,
  invariant-factor, dynamic-vector order. An applied compressed compilation
  realizes compressed parent, count, fill, sparse, dense, single-iteration,
  invariant-factor, dynamic-vector records. Result/ABI assembly remains the
  typed lazy barrier between invariant-factor and dynamic-vector rewriting; it
  is not a new pass.
- A count failure completed no nested record and suppressed fill and all later
  work. A fill failure retained only count. Failure while building or verifying
  the compressed parent retained count and fill but no parent. Each later pass
  failure retained all previously completed top-level and nested records and
  suppressed every later pass, assembly step, function construction, schedule
  lowering, and code generation. The new pipeline globally reindexes the same
  completed records and also transports them across an unexpected ordinary
  Python exception from the legacy assembly barrier; `CINLowerer` then raises
  the exact original exception.
- Production `LLIRPassOptions(False, False, True)` still disables full
  before/after walking and records duration. Debug
  `LLIRPassOptions(True, True, True)` still verifies before and after and records
  duration. `CompileOptions` accepts only those two policies; direct managers
  retain their supported custom-policy compatibility. Cheap artifact,
  descriptor, context, root, nested-record, and snapshot-identity validation
  remains fail closed in every mode.
- Before this closure, `CompileOptions.enabled_llir_passes` was frozen,
  validated, and included in semantic/build cache identity but was not consumed
  for production orchestration. It now supplies the exact tuple retained by
  `LLIRPassPipeline`; the manager also retains the matching ordered descriptor
  tuple and rejects a missing, reordered, malformed, or detached production
  pipeline before work. An applied compressed spec must retain the pipeline's
  exact `CompileOptions` object, while standalone compressed/result-write and
  other direct pass APIs retain their existing compatibility behavior.

The complete analysis inventory and classification is:

| Current computation | Input, output, consumer, and recomputation | Canonical classification |
| --- | --- | --- |
| Normalized-CIN ownership/use/access analysis | `IndexStmt` -> fresh frozen `CINAnalysis(root_id, parents, node_scopes, scope_parents, symbol_definitions, symbol_uses, index_definitions, index_uses, accesses, access_occurrences, tensor_accesses, access_order, free_index_ids, reduction_index_ids, diagnostics)`. Consumed by `verify_cin`, `LoopPlan` entity collection/verification, scheduler ID-boundary checks, and diagnostic/display-name lookup. It is recomputed on every request and retains no mutable IR reference. | The sole current **canonical common analysis-runner** computation. `AnalysisRunner.analyze_cin` is explicit and typed; the compatibility `cin_analysis.analyze_cin` entry delegates to the zero-field frozen `COMMON_ANALYSIS_RUNNER`. |
| Dynamic-vector declaration names; dense-pointer `_LoopAnalysis`; single-step bounds and `_LoopMatch`; invariant-factor defined-variable/factor partitions; sparse-prefetch value/coordinate/dense-access facts; result-write target/coordinate matches; compressed-Where loop, bound, sparse-position, work, and parallel-policy facts | Each scan is derived from the exact LLIR artifact/configuration consumed by one transform and is used only while producing that pass's detached result. Normal repeated calls rescan their input. Current mutable LLIR has no stable node-ID side-table boundary. | **Pass-local derived facts**. They deliberately remain inside their pass and are not promoted into the common runner. |
| Schedule-lowerer loop discovery; `Scheduler` cost/selectivity/mode graph and planning; `IterationLattice`/`ModeIterator`; legacy CIN collectors/backlinks and adapter state; `CINLowerer` tensor/workspace/result/ABI state; normalization, verification, traversal, and canonical serialization | These computations either construct/validate an artifact, encode program or lowering state, or depend on mutable legacy ownership and generated spelling. Consumers are the corresponding scheduler, lattice, adapter, lowerer, verifier, or serializer stage. | **Program/lowering state outside the common analysis runner**, not reusable immutable analysis side tables. |

No current LLIR computation satisfies the common-runner contract, so no LLIR
analysis API was invented. No analysis cache is required: current CIN is small,
the typed result is immutable and cheap to recompute, and caching mutable legacy
IR would require the explicitly prohibited fingerprint cache plus
preserve/invalidate protocol. Existing caches remain unchanged: `CompileOptions`
semantic/build/cache keys, `Schedule.cache_key`, operation dispatch and kernel
caches and `utils._so_cache` cache configuration or build
artifacts rather than analysis results.

Ownership remains caller-safe. Normalization detaches CIN before private legacy
mutation; `ScheduledCIN` carries only detached CIN plus `LoopPlan`; the legacy
adapter claims or copies private state; `CINAnalysis` contains frozen IDs,
scalars, tuples, and `FrozenMap` tables rather than mutable CIN/LLIR nodes; and
every managed LLIR output, including no-ops, is detached. The pipeline owns only
frozen configuration and typed artifact carriers around caller-independent
payloads. A first compilation/result remains independent of later caller
mutation and of a second distinct `CompileOptions` snapshot.

The clean latency artifacts are
`/tmp/scorch-phase2-manager-pipeline-final/latency-af16e792aa1-m5.json`
(`382a04de605b50e2280362b26a71060dd89c0e26c21c1e42a7519d4dd4f99a7d`)
and
`/tmp/scorch-phase2-manager-pipeline-final/latency-cdb12bc8402-m5.json`
(`1f859021940efff5d3b48c51e01d5280949591b47c1d0ec8fadc5c55c3e60572`).
Both metadata records have empty git status and exact detached revisions. The
retained `14b110b` control remains invalid against its own archive and was not
rerun or reclassified; this closure used the single fresh current-predecessor
pair recorded below.

#### Canonical deliverable matrix

| Phase 2 deliverable | Status | Exact evidence or impact |
| --- | --- | --- |
| Immutable `CompileOptions` snapshot | **MET** | `7c68ac9` introduces exact frozen compiler, scheduler, verification, build, target, ABI, wrapper, and Darwin-toolchain values; snapshots mutable schedules/flags/pass inputs into tuples; parses the audited environment and compatibility `ContextVar`s once at public boundaries; and routes one object through normalization, scheduling, lowering, emission, naming, and the detached JIT build. `d8ae9ed` separates direct-extension Darwin target flags from predecessor-identical production flags. `d4a4cb3` completes exact identity routing through nested compressed-output/result-write renderers and rejects plain CIN that would ignore a requested schedule. Unsupported values and combinations fail closed with typed diagnostics. Focused executable coverage proves freezing/types, detachment, one-time reads, post-snapshot environment independence, independent snapshots, production/debug verification, public and nested identity, build-child isolation, cache separation, invalid inputs/combinations, ownership, and source anchors. No singleton, mutable registry, callback bag, reflection/signature inspection, `Any` configuration dictionary, new cache, or preserve/invalidate machinery was introduced. |
| Stable `SymbolId`/`IndexId` references at the `LoopPlan` boundary | **MET** | `identity.py`, `cin.py`, and `cin_analysis.py` assign and verify stable typed IDs. `loop_plan.py` represents every addressable loop/tensor decision with `IndexId`, `SymbolId`, or typed `LoopRef`; `scheduler._build_loop_plan` resolves public names once and emits only IDs. `test_cin_analysis.py` and `test_loop_plan.py` cover stable identity, dangling references, detachment, and caller-list snapshotting. No name or object-identity adapter crosses this boundary. Core commits: `9e3690c`, `9dcd06c`, and audit hardening `72b8a59`. |
| Frozen `ScheduledCIN(cin, plan)` transitional carrier | **MET** | `loop_plan.ScheduledCIN` remains an exact frozen two-field dataclass containing only detached normalized CIN and a `LoopPlan`; it owns no analysis bundle and adds no CIN node schema. Its constructor deliberately remains a carrier, while semantic acceptance is enforced at `verify_scheduled_cin` and lowering trust boundaries. Exact carrier, field, structural, and semantic-boundary behavior is covered in `test_loop_plan.py` and the legality closure below. |
| Internal `LoopPlan` translation from public `Schedule` | **MET** | `scheduler._build_loop_plan` translates public name-bearing schedule decisions to stable IDs, and `Scheduler.apply_schedule` returns `ScheduledCIN`. Public scheduling compatibility, translation, and convergence with direct artifact legality are covered by `test_loop_plan.py`, `test_schedule_api.py`, and `test_schedule_generality.py`; the semantic verifier closure is recorded below. |
| Removal of dynamic schedule attributes | **MET** | Public scheduling decisions are carried by `LoopPlan`; tests in `test_loop_plan.py` and `test_schedule_api.py` assert the old scheduling attributes are absent from normalized CIN and that scheduling leaves the input unchanged. Legacy scheduling mutation occurs only on a private copy. |
| Initial pass manager and pure analysis runner; no cache or preserve/invalidate machinery | **MET** | `8e89822` adds the exact frozen `LLIRPassPipeline` assembled from and retaining one `CompileOptions` snapshot, and one explicit manager-owned heterogeneous production entry. `CINLowerer` delegates once; ordered ordinary and compressed records, same-source count/fill siblings, production/debug policy, partial failures, later-work suppression, trust-boundary validation, and direct pass compatibility are executable. `cdb12bc` adds the zero-field frozen `AnalysisRunner`; every typed CIN request recomputes a fresh immutable `CINAnalysis`. No analysis cache, dependency graph, callback registry, reflection, or preserve/invalidate protocol was introduced. |
| Common current-LLIR walker/rewriter | **MET** | `llir_traversal.py` provides exhaustive exact-type walking and detached rewriting for the current LLIR. `test_llir_traversal.py` covers declared-node completeness, deterministic traversal, nested container shape, detachment, malformed children, replacement validation, and unknown-subclass rejection. Commit `14c1f27`. |
| Four existing sequential LLIR optimizations extracted into passes | **MET** | Sparse prefetch (`325547d` implementation, `2bb1ce6` routing, `cfec7c5` traversal cleanup, `c8af101` descriptor/test hardening), dense-pointer hoisting (`ce5adad`, `81b847a`), single-iteration elimination (`0d31882`, `265123a`), and invariant-factor hoisting (`112a9b7`, `f2bca1b`) are typed, production-routed passes with focused structural, ownership, failure, timing, and activation tests. Dynamic-vector, result-write, and compressed-Where are also managed, for seven managed pass descriptors total. |
| Per-stage latency instrumentation in production and debug configurations | **MET** | `8b3b910`/`d985f74`, corrected and completed by `e14dfbf`/`141cbc5`, add one frozen identity-owned `CompilationContext` per compilation. It retains the exact `CompileOptions` object and immutable `CompilerStageRunRecord`/existing `LLIRPassRunRecord` snapshots. Nine stable `CompilerStageId` values cover validated frontend construction, CIN normalization/optional verification, scheduling/`LoopPlan`, distinct legacy adaptation, CIN lowering, the nested result/ABI barrier, schedule lowering, LLIR-to-C++, and validated kernel-name/build-request assembly ending immediately before cache/native loading. Production and debug execute the same complete stage inventory; only the snapshotted verifier policy differs. Failed stages publish no record, make the owner terminal, preserve earlier stage/pass records, and re-raise the exact original exception. See the closure below. |
| Isolated pass/artifact plumbing latency benchmark | **MET** | Focused executable plumbing tests enforce the 1 ms ceiling. The prior clean committed measurement recorded above reports empty-manager p95 1.625 microseconds, incremental manager p95 1.916 microseconds, direct-pass p95 2.083 microseconds, and complete managed-call p95 3.917 microseconds. |

#### Canonical exit-criterion matrix

| Phase 2 exit criterion | Status | Exact evidence or impact |
| --- | --- | --- |
| One CIN can be compiled independently under two schedules | **MET** | `test_loop_plan.py` and `test_schedule_api.py` compile the same normalized CIN with distinct schedules, observe distinct scheduled/codegen results, and verify the input and first result remain unchanged. `apply_schedule` normalizes/detaches before legacy scheduling mutation. |
| Pass order is explicit and observable | **MET** | `LLIRPassPipeline` retains the exact seven option-supplied identities and matching descriptors. `LLIRPassManager.run_production_pipeline` explicitly realizes ordinary sparse, dense, single-iteration, invariant-factor, assembly barrier, dynamic-vector order, and applied compressed parent, count, fill before that sequence. `CINLowerer` contains one production orchestration call and no individual manager `run_*` call. Behavioral tests observe exact record order/indices and inject failures at nested, pre-assembly, assembly, and post-assembly positions. |
| Every extracted pass has focused structural tests | **MET** | The seven pass suites cover positive/no-op structure, exact descriptors/specs/contexts/artifacts/roots, malformed/unknown nodes, detachment, ordering, verification modes, failures, and pass-specific repeated-application semantics: `test_dynamic_vector_access_pass.py`, `test_result_write_pass.py`, `test_compressed_where_openmp_pass.py`, `test_sparse_prefetch_pass.py`, `test_dense_pointer_hoist_pass.py`, `test_single_iteration_loop_pass.py`, and `test_loop_invariant_factor_pass.py`. |
| No schedule metadata is dynamically attached to CIN nodes | **MET** | Schedule decisions reside in frozen `LoopPlan`; source search and `test_loop_plan.py` show no dynamic schedule fields on normalized CIN. Compatibility backlinks are a separate legacy concern described below. |
| Empty and incremental plumbing p95 remain below 1 ms | **MET** | Clean committed measurements are 1.625 and 1.916 microseconds respectively, over 500 times below the ceiling. Focused tests retain the executable 1 ms assertion. |
| Production compiler latency is measured or handled under the settled policy | **MET** | Sparse and dense routing were not assigned candidate results after unchanged `14b110b` controls missed their own clean archive: sparse preflight ratios were 1.054/1.154, 1.093/1.182, 1.121/1.204, and 1.046/1.092; dense preflight ratios were 1.048/1.108, 1.087/1.170, 1.035/1.033, and 1.027/1.017. The single-iteration candidate was validly measured against Phase 0 at 1.127/1.017, 1.108/1.116, 1.132/1.070, and 1.130/1.192. Direct pass attribution was 22.980-68.541 microseconds p50 and 24.000-71.500 microseconds p95; the modest ownership/validation exception was explicitly accepted. For invariant-factor routing, unchanged-control ratios were 1.068/1.136, 1.088/1.155, 1.050/1.074, and 1.057/1.108. It was not rerun opportunistically and no candidate sample was taken; this is not a valid candidate latency result. `64a5f1e` labels 1.10 crossings `INVESTIGATE`, retains a nonzero result, and prints the attribution requirement instead of treating crossings as automatic rejection. The manager closure used one fresh clean committed `af16e79`/`cdb12bc` pair rather than rerunning the invalid control. Baseline -> candidate p50/p95 milliseconds and ratios were: small dense 1.454/1.718 -> 1.475/1.713 (1.014/0.998), reduction 1.337/1.507 -> 1.354/1.551 (1.012/1.029), CSR intersection 1.432/1.540 -> 1.427/1.516 (0.996/0.984), and sparse union 1.356/1.475 -> 1.362/1.464 (1.004/0.993). Every cell is below 1.10, so no exception is required. |
| Every emission-affecting extraction satisfied its phase-local two-machine gate | **MET** | The retained clean archives were inspected directly. On both M5 and Redwood every archive has 42/42 correct cells, and sorted per-cell build inputs are byte-identical through the full chain: sparse `cfec7c5` versus `14b110b`, sparse hardening `c8af101` versus `cfec7c5`, dense `81b847a` versus `c8af101`, single `265123a` versus `81b847a`, and factor `f2bca1b` versus `265123a`. Runtime comparison was therefore correctly waived at each byte-identical seam. Audit fixes `72b8a59`, `74c7a96`, and `64a5f1e` do not change valid emitted kernels. The CompileOptions diagnostic grids and current-toolchain replacement control are recorded below; final `d8ae9ed`/`d4a4cb3` restore predecessor-identical production build inputs and preserve all source anchors, so the binding policy again waives a ceremonial identical-binary rerun. The clean committed manager candidate retains all four source sizes/hashes, and its production C++ flags, linker flags, kernel name, and preamble digest JSON is byte-identical to `af16e79`; both hash to `4342e155548e81f8524b69a84c6e1d1b114f71de10ad6832b49a23e39257023a`. No emitted production build input changed, so the retained M5/Redwood 42-cell archives remain the applicable runtime evidence and no ceremonial two-machine rerun was performed. |

#### Supplemental ownership, verification, and failure audit

- CIN construction owns its forward state; normalized analysis is ID-keyed,
  deterministic, and recomputed without attaching analyses. Production lowering
  constructs `TensorAssign` explicitly. `Scheduler.apply_schedule` operates on
  detached normalized CIN and preserves caller-owned input across independent
  schedules.
- Materialization/restoration of legacy compatibility backlinks on scheduled
  private trees is isolated to `compiler/legacy_cin_adapter.py`; normalization
  clears those legacy fields in `compiler/cin_analysis.py`. No scheduling
  decision crosses the `LoopPlan` boundary through a backlink.
- `72b8a59` made tuple-valued `LoopPlan`, `OperandRelayout`, and `ResultTile`
  fields detach from caller-owned lists; rejects unordered containers; verifies
  exact nested structures and well-formed IDs; requires a complete loop order;
  and rejects duplicate panel bounds, malformed structured values, and conflicting
  parallel selections with structured `VerificationError`s.
- Canonical semantic `LoopPlan` legality is **CLOSED** by `f63f8c5`,
  `f2e693f`, and the audit completions through `fd0ff9a`/`87014ac`. The same
  direct forged SpMM artifacts now fail at both
  `verify_loop_plan` and `verify_scheduled_cin`: result order `k,i,j` reports
  `result_storage_order`, CSR child-before-parent `j,i,k` reports
  `sparse_parent_dominance`, and parallel reduction `j` reports
  `parallel_reduction`. The closure uses immutable canonical `CINAnalysis`
  facts and is recorded in detail below.
- All manager outputs, including legal no-ops and `run_empty`, are detached from
  caller-owned mutable LLIR. Production disables full before/after verification;
  debug enables both, while exact public/trust-boundary validation remains on.
  Exact descriptor, artifact, root, context, and specification checks fail
  closed; aggregate tests now include the result-write and compressed-Where
  combinations that previously lacked durable manager-level evidence.
- Run records and durations are non-semantic (`compare=False`) and are never
  consumed by scheduling or code generation. `74c7a96` adds a typed internal
  partial-failure carrier so a compressed fill failure preserves only the
  completed count record. Every managed-pass failure stops later work and keeps
  exactly earlier records; malformed nested records are rejected before they can
  be carried. The public/direct API still raises the original compiler error.
- Manager documentation and descriptor tests enumerate all seven managed passes.
  There is no analysis/result cache, callback bag, signature inspection,
  `Any`-based dispatch, preserve/invalidate machinery, or benchmark-corpus
  specialization.
- The introduced-machinery check is **MET with qualification**. No pass adds
  generalized reflective or signature-based dispatch. `llir_traversal.py`,
  `dense_pointer_hoist_pass.py`, `result_write_pass.py`, and
  `compressed_where_openmp_pass.py` do retain closed `getattr`/`hasattr`/
  `setattr`/`delattr` probes for known legacy fields and generated spellings.
  Eliminating those compatibility probes is **DEFERRED BY DESIGN** to Phase 3
  structured-CxxIR/string-rewrite work; it is not a Phase 2 exit blocker.

#### Generated-source and retained-archive audit

The current anchors were regenerated exactly:

| Kernel | Bytes | SHA-256 |
| --- | ---: | --- |
| CSR by dense | 2,505 | `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151` |
| DS | 7,117 | `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f` |
| DSS | 8,660 | `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e` |
| All-COO SDDMM | 3,543 | `de94b08752077a621c5e411ce0dcbb40e8bcbeacb9bce3824dd6019e2d2bd29d` |

The all-COO factor region is correct: it declares
`float _inv_17 = Mask_val[pMask0];`, the `q` loop accumulates only
`_Query_val_ptr[q] * _Key_val_ptr[q]`, and `_accum *= _inv_17;` follows
immediately. It retains `Mask1_crd[pMask0]` and `Mask_val[pMask0]`, and there is
no inner `pMask1` loop. However, the anchored 3,543-byte source still declares
`int pMask1_end = 0;`. The earlier progress claim that the declaration was gone
was incorrect; existing coverage only excluded the exact derived-bound
`pMask1_end = pMask0 + 1` statement/use. The explicit
audit requirement that no `pMask1_end` declaration remain is therefore a
**GAP**. Removing it is an emission-affecting change requiring a changed byte
anchor plus the full two-machine gate, so it was not implemented in this
exit-only audit.

The retained two-machine files were inspected directly, not accepted only from
the earlier prose. Clean M5/Redwood pairs are under
`/tmp/scorch-phase2-pass-manager-final/` (`14b110b`),
`/tmp/scorch-phase2-sparse-prefetch-final/` (`cfec7c5` and `c8af101`),
`/tmp/scorch-phase2-dense-pointer-final/` (`81b847a`),
`/tmp/scorch-phase2-single-iteration-final/` (`265123a`), and
`/tmp/scorch-phase2-loop-invariant-final/` (`f2bca1b`). The machine/configuration
metadata matches the required 42-cell grid. All archives are clean and 42/42
correct; sorted per-cell comparisons found 42/42 byte-identical build inputs
for every adjacent pair on both machines.

The invalid sparse, dense, and invariant-factor latency controls were also
reconciled against the clean same-commit `14b110b` archive. The invariant pair
is
`/tmp/scorch-phase2-loop-invariant-final/latency-14b110b-m5-preflight.json` and
`/tmp/scorch-phase2-pass-manager-final/latency-14b110b-m5-final.json`; both are
clean, same-commit `14b110b` archives with matching corpus configuration.

The CompileOptions implementation preserved all four generated-source anchors
and kernel names. Initial commit `7c68ac9` added an explicit Darwin `-isysroot`
to the production generated-kernel flags while diagnosing a concurrent Apple
Command Line Tools update. That changed one class of M5 build input, so clean
committed full-grid diagnostics were run. The exact current-toolchain
`969f3cd` replacement control and `7c68ac9` candidate were both 42/42 correct.
Their M5 comparison had three individual cell-band crossings, while the
machine geomean was `0.995`, inside the candidate same-binary band
`[0.991, 1.009]`; the comparison therefore retained its nonzero diagnostic
status. The exact crossings were: `M=512,N=64,density=0.1`,
`0.1500845 -> 0.1384020` ms, ratio `0.922160` versus
`[0.936667,1.067616]`; `M=20000,N=16,density=0.02`,
`10.0975037 -> 9.9070072` ms, ratio `0.981134` versus
`[0.987261,1.012903]`; and `M=20000,N=8,density=0.1`,
`45.0789928 -> 45.6268787` ms, ratio `1.012154` versus
`[0.990457,1.009635]`. The first two crossings are faster, and the sole slower
cell is a 1.215% increase only 0.252 percentage points above its band. The
source, optimization/ISA flags, and full-grid geomean exclude a systematic
code-generation regression; the changed pair only pins header/SDK selection,
so these three noisy cell crossings are accepted for the narrow direct-helper
toolchain exception rather than hidden by a rerun. On Redwood, `7c68ac9` was
42/42 correct and all 21 build records were exactly predecessor-identical, so
runtime comparison was waived.
The first comparison against the stale retained M5 predecessor archive was
invalid because the generated build inputs differed across toolchains; its
diagnostic geomean was `1.015` with 34/42 cells inside their bands, and it was
not used as an acceptance result.

The Command Line Tools installation recorded in `/var/log/install.log` during
this session made the older retained M5 predecessor binary archive an invalid
cross-toolchain control. Exact `969f3cd` reproduced a mixed
CommandLineTools/Xcode header failure without a coherent target environment;
the declared replacement control pinned `DEVELOPER_DIR` and `SDKROOT` to the
current CommandLineTools installation. Final fix `d8ae9ed` keeps explicit
`-isysroot` only in frozen `direct_extension_cflags` for direct legacy
`load_inline` callers, while the isolated child receives coherent
`DEVELOPER_DIR` and `SDKROOT` for production builds. Production flags, linker
flags, source, functions, and generated names are therefore again
predecessor-identical on M5, and the `7c68ac9` Redwood records were already
identical. The Darwin direct-extension helper flags intentionally differ from
`969f3cd` by that SDK pair: this is the narrow toolchain-compatibility build
input exception. The identical flag spelling was exercised by the full 42-cell
M5 grid at `7c68ac9`, and the final committed focused suite compiles a real C++
syntax probe with `direct_extension_cflags` after removing ambient `SDKROOT` and
`DEVELOPER_DIR`; Redwood's non-Darwin helper flags did not change. `d4a4cb3`
only routes the already-owned default policy through
nested renderers and does not change a supported default output. Under the
binding byte-identity policy, no final ceremonial runtime rerun was performed.
As a final committed-candidate check, isolated clean `969f3cd` and `d4a4cb3`
worktrees produced byte-identical JSON for production C++ flags, linker flags,
kernel name, and preamble SHA-256; both files hash to
`4342e155548e81f8524b69a84c6e1d1b114f71de10ad6832b49a23e39257023a`.
The four anchor tests at clean committed `d4a4cb3` passed in 0.67 seconds and
left that worktree clean.

The retained diagnostics and SHA-256 digests are:

- `/tmp/scorch-phase2-compile-options-final/kernel-aa-7c68ac9ca037-m5.json`,
  `4cf3049e34933075a9d0ef57aaa3350d51683bea8e5eabd303061db42255423c`;
- `/tmp/scorch-phase2-compile-options-final/kernel-aa-7c68ac9ca037-redwood.json`,
  `120c924a19535f96cf279cd47808d66567ff3c15f79d96798065b3bcd4d4b7d9`;
- `/tmp/scorch-phase2-compile-options-final/kernel-aa-969f3cdd7cd-m5-current-clt-control.json`,
  `d10d668b672114a821e81a0b018e697123f6e9a810b25d952471df5d994a207d`;
- retained predecessor M5
  `/tmp/scorch-phase2-loop-invariant-final/kernel-aa-candidate-m5.json`,
  `816430d435d76dbe47277e921228d45c7387fc93d110bc37cc70f173982dfb77`;
  and
- retained predecessor Redwood
  `/tmp/scorch-phase2-loop-invariant-final/kernel-aa-candidate-redwood.json`,
  `31baca05f50d8f51483aa1d7d5b77f19f93550ebb95e46209d8dcbbd935dbc7e`.

No new candidate compiler-latency sample was assigned. The unchanged clean
`14b110b` preflight remained invalid against its own retained archive and was
not rerun merely to seek a passing sample. Archive p50/p95 milliseconds were
`0.9360005/1.0621401`, `0.8618545/0.95682935`,
`0.8447505/0.91348385`, and `0.7487705/0.8238083` for small dense,
reduction, CSR intersection, and sparse union. The unchanged preflight values
were `0.9995/1.20686265`, `0.9375625/1.10532065`,
`0.886896/0.9811416`, and `0.791104/0.9131712`, giving p50/p95 ratios
`1.068/1.136`, `1.088/1.155`, `1.050/1.074`, and `1.057/1.108`.
This is an invalid control result, not a candidate latency result.

#### Exact remaining unmanaged LLIR traversal inventory

1. `compiler/codegen.py`: `LLIRLowerer` expression/statement/loop/conditional/
   function traversal. This is legitimate final emission; exhaustive structured
   replacement is Phase 3 work.
2. `compiler/llir.py`: unused `NodeVisitor`/`CppCodeGenerator` duplicate emitter.
   Repository search finds no production caller; deletion/unification is explicit
   later structured-emitter debt.
3. `compiler/schedule_lowerer.py`: loop/body/tag/declaration discovery, panel
   placement, access rewriting/checking, prefetch redirection, dense-result-zero
   removal, heap result tiling, relayout, and schedule-lowering orchestration.
   These are legitimate current post-LLIR scheduling operations pending LoopIR.
4. `compiler/cin_lowerer.py`: sparse-loop/target-policy traversal, logical-loop
   tags and explicit parallel selection, and COO parallel output/position/string
   rewrites. The unused recursive helpers `_find_val_array_access`,
   `_find_all_sparse_pos_arrays`, and `_spgemm_flop_work_expr` have no external
   callers and are later cleanup debt.
5. `compiler/iter_lattice.py`: `_mark_simd_on_reduction_loops`; legitimate current
   target scheduling.
6. `ops.py`: register-block top-loop discovery, free-size expression traversal,
   and dual-path stitching; legitimate current schedule selection/stitching.

No other production source has an unmanaged recursive or list-scanning consumer
of an already-built LLIR tree outside the common traversal, managed passes, and
the inventory above.

#### Verification and decision

The following results are the preceding exit audit's historical baseline,
before the CompileOptions closure follow-up. The initial unmodified audit
baseline produced 455 passed tests in the focused
Phase 2/common suite and 82 passed tests in the schedule/codegen matrix. Audit
fixes have focused coverage for `LoopPlan` structural boundaries, exact manager
combinations, nested partial failures, later-work suppression, and latency-tool
policy. The final exact focused command passed 464 tests in 0.84 seconds, and the
exact schedule/codegen command passed 82 tests in 352.99 seconds. The full
`pytest -q -m "not perf" tests` run completed with 973 passed, 14 skipped, 3
deselected, and one failure in 1,872.66 seconds. The sole failure is the unmarked
performance-only `test_spmm_dd_ds_dd_tiled_time`: `lower_Where` indexes an empty
dense-access list. Running that exact test from a detached `7205fb1` worktree
reproduced the same `IndexError` at the predecessor line in 7.35 seconds, so it
is an inherited failure, not an audit regression.

Black left all nine changed Python files unchanged. Flake8 reported the same
five inherited `cin_lowerer.py` findings at `7205fb1` and at the audit head.
Mypy reported the same 59 inherited errors in the same three existing files at
both revisions; the new benchmark-policy test and changed benchmark tool are
clean. `git diff --check` is clean. No `csrc` file was changed, and no test or
benchmark output was added to the repository.

The CompileOptions follow-up adds 42 focused executable cases. Its final exact
focused command passed 506 tests in 1.30 seconds, and the exact
schedule/codegen matrix passed 82 tests in 377.87 seconds. The final full
`pytest -q -m "not perf" tests` run completed with 1,015 passed, 14 skipped, 3
deselected, one warning, and the same sole inherited
`tests/test_scorch/test_perf.py::test_spmm_dd_ds_dd_tiled_time` failure in
2,059.65 seconds. The failure is the already-reproduced `IndexError` from
`CINLowerer.lower_Where`; this scoped follow-up did not fix or mark it.

Black left all 13 implementation/test Python files unchanged. Flake8 reported
the same nine inherited findings on the corresponding existing files at exact
`969f3cd` and at `d4a4cb3`, for zero regressions. Mypy reported 128 inherited
errors in five existing files at `d4a4cb3`, versus 131 errors in six existing
files at exact `969f3cd`: zero regressions and three inherited `utils.py` errors
removed. New `compiler/compile_options.py` and
`tests/test_scorch/test_compile_options.py`, plus the newly changed nested-pass
files, are clean. `git diff --check` is clean. No `csrc` or design file,
benchmark output, generated extension, plot, cache, or unrelated file changed;
the user-owned `.gitignore` modification and untracked research/benchmark
material remain untouched.

The manager/analysis-runner follow-up adds 17 focused executable cases. Its
final exact focused command passed 523 tests in 1.53 seconds, and the exact
schedule/codegen matrix passed 82 tests in 397.90 seconds. The final full
`pytest -q -m "not perf" tests` run completed with 1,032 passed, 14 skipped, 3
deselected, one warning, and the same sole inherited
`tests/test_scorch/test_perf.py::test_spmm_dd_ds_dd_tiled_time` failure in
2,096.81 seconds. The traceback is the baseline `IndexError` in
`CINLowerer.lower_Where`; this scoped closure did not fix or mark it.

Black left all nine changed implementation/test Python files unchanged. On the
exact changed-file set, Flake8 reported five findings at both `af16e79` and
`cdb12bc`, all in existing `cin_lowerer.py`, for zero new findings; the existing
`lower_IndexStmt` C901 score changes from 50 to 53 because the manager call's
typed lazy assembly barrier and partial-failure transport remain visibly inside
the legacy lowering method rather than beginning the prohibited Phase 3 ABI
extraction. Mypy reported the same 48 inherited errors in the same two existing
files at both commits: 40 in `cin_lowerer.py` and eight import-typing findings in
`test_cin_analysis.py`. The new `analysis_runner.py` and every other changed
Python file are clean. The broader inherited inventory remains nine Flake8
findings in four files and 128 mypy errors in five files, exactly matching the
`af16e79` baseline. `git diff --check` is clean. No `csrc` or design file,
benchmark output, generated extension, plot, cache, or unrelated file changed;
the user-owned `.gitignore` modification and untracked research/benchmark
material remain untouched.

#### Compiler-stage timing closure (2026-07-14)

Commits `8b3b910`/`d985f74`, followed by corrective implementation and test
commits `e14dfbf`/`141cbc5`, close only the canonical production/debug
compiler-stage timing blocker. They do not change pass order, scheduling,
result/ABI semantics, emitted syntax, kernel names, native flags, or the
all-COO `pMask1_end` declaration.

The pre-implementation audit found four distinct timing classes:

1. There was no canonical compiler-stage record. The only in-compiler records
   were the existing `LLIRPassRunRecord` values, created with
   `perf_counter_ns` for the seven managed pass descriptors (with applied
   compressed-Where count/fill siblings).
2. `time_dict["eval_time"]` used `time.time` at the existing production
   execution sites. It measures native kernel execution and remains a legacy
   compatibility contract outside the compiler-stage seam.
3. `tiling.py` used `time.perf_counter` for native selector calibration and
   execution, while `utils.load_to_kernel_cache` used `time.time` for a
   special/native build. Both remain outside this seam. Commented build timers
   remain inert.
4. The Phase-0 tool timed the public operation through the former
   `_load_kernel` call but had no typed per-stage attribution. It now retains a
   predecessor-compatible marker and also measures the canonical endpoint.

The exact owner is frozen, identity-equality `CompilationContext`, containing
one exact `CompileOptions` object and immutable tuple snapshots of completed
stage and managed-pass records. `CompilerStageRunRecord(sequence_index,
stage_id, nested_within, duration_ns)` is frozen; `duration_ns` is
`field(compare=False)` and comes from `perf_counter_ns`. Opaque frozen
`CompilerStageToken` values enforce exact owner/token identity, serial root
stages, strict LIFO completion, and terminal suppression after the first
failure. The nine stable `CompilerStageId` values, in
`CANONICAL_COMPILER_STAGES` order, are:

1. `frontend_validated_operation_construction`;
2. `cin_normalization_and_verification`;
3. `scheduling_and_loop_plan_construction`;
4. `legacy_cin_adaptation`;
5. `cin_lowering`;
6. `result_abi_assembly`;
7. `schedule_lowering`;
8. `llir_to_cpp_generation`;
9. `kernel_name_and_build_request_assembly`.

Only result/ABI assembly is a nested canonical stage, directly inside CIN
lowering. CIN-lowering duration therefore includes the manager-owned pass
pipeline and the lazy result/ABI continuation. The barrier remains after
invariant-factor hoisting and before dynamic-vector rewriting; the assembled
artifact is validated inside the R record and again at the unchanged manager
trust boundary. Schedule lowering is a root stage after L. Existing managed
pass records remain a distinct nested observation: their seven descriptor
identities and order are unchanged, they are appended to the same context only
while L is active, and compressed-Where parent/count/fill ordering is unchanged.

The audited production boundaries and orchestration are:

| Boundary/path | Actual compiler work and timing behavior |
| --- | --- |
| `einsum` auto path | F; optional full nested relayout compilation(s); prealignment S; N/S for each real scheduled arm; A/L(R) for each real lowered arm; optional L stitch; C; K. A simple non-dual path is F,S,N,S,A,L,R,C,K. |
| `einsum` explicit schedule | F,S,N,S,A,L,R,SL,C,K, with the same genuine nested relayout behavior. The original topological relayout and later selected-order relayout both remain. |
| `lower_and_exec_cin` | N; F for runtime-binding planning; optional full nested relayout compilation(s); F for applying/validating the binding; A,L,R,C,K. The caller's CIN and runtime tensors remain detached/owned as before. |
| `spmv`, `STensor.__add__`, compiled `to_dense`, `to_sparse`, and `change_mode_order` | F,N,A,L,R,C,K. Nested prerequisite conversions or relayouts run the same complete manual sequence on the same owner. |
| `matmul` and wrappers | Dense/prebuilt routes execute no generated compiler stage; generated routes delegate to `einsum` or `spmv` with the same snapshot/context. Dense-to-sparse result conversion, sparse `to_torch`, unsupported sparse-linear fallbacks, and eager-equivalent compile fallbacks reach the existing generated conversion/matmul boundaries. |
| `precompile_kernels` | Five independent compile-only public `einsum` compilations, each with its own snapshot and owner. |

Prebuilt SDDMM/matmul, dense delegates, rank-one sparse conversion, dense
to-dense, mode-order no-ops, core-format fast relayout, and special kernels do
not fabricate records. A dispatch-cache hit cancels the speculative outer F
and records only any genuine input/output relayout compilation. A
single-path kernel-cache hit retains the frontend/normalization/scheduling work
that actually reruns and skips A/L/C/K. `_so_cache` and on-disk extension hits
occur after K and therefore retain all compiler records.

At the time of the timing audit, the subsequently removed prototype
`matmul_wksp` boundary also recorded F,N,A,L,R,C,K on a private module-cache
miss and no main compiler work on a hit. Its later removal deleted that public
boundary and private cache without changing any stage identity, timing rule, or
remaining production path.

K includes source-derived kernel naming and assembly plus validation of the
exact frozen `_PreparedJITBuild(request, cache_key, so_path)` carrier. The
request contains only name, C++ source tuple, function tuple, flag tuples,
build directory, and the snapshotted build options. K completes immediately
before `_load_validated_prepared_kernel`; `_so_cache` lookup, extension import,
subprocess/native compilation, cache insertion, module execution, and result
wrapping are excluded. Torch's `TORCH_EXTENSIONS_DIR` lookup remains the
previously classified non-semantic storage/cache-location boundary; it is not
compiler policy or emitted/build identity.

Production and debug use the same complete stage inventory. Only the exact
snapshotted `verification.verify_cin` and managed-pass verification policy
differs; timing never enables verification. Every timed stage and nested pass
receives the context's identical `CompileOptions` object. Behavioral spies
prove that corresponding environment values and the CIN/schedule/regblock
`ContextVar`s are not read after snapshot construction.

A failed canonical stage publishes no failed-stage record, preserves all
earlier completed records once in deterministic start order, marks the owner
terminal, suppresses every later stage, and re-raises the exact original
exception. Completed R remains if a later dynamic-vector pass fails. Ordinary
compressed-Where failures retain the same partial pass policy: count failure
retains no nested sibling, fill failure retains count, and later parent work
retains count/fill. Unexpected ordinary Python exceptions are transported
without erasing completed records; `BaseException` behavior is unchanged.
Native/cache failures happen after completed K and do not retroactively create
a failed compiler stage.

Timing is non-semantic. The context and records are absent from
`CompileOptions` equality/fingerprints, operation/dispatch/module/kernel/native
cache keys, emitted names/source/flags, build carriers, and public results;
canonical cache serialization rejects them. Deterministically different clock
observations produce identical results, generated source, names, cache
identity, and exact build-request schemas. Two distinct snapshots have
independent owners, records, requests, and result objects; caller-owned CIN,
LLIR, analyses, schedules, tensors, and prior results remain unchanged. Direct
normalizer, scheduler, lowerer, renderer, manager/pass, and legacy loader APIs
retain their supported standalone behavior; passing a context opts the allowed
compatibility seams into the same typed ownership.

Existing public callers and `time_dict` continue to receive only the native
`eval_time` contract. Canonical compiler records are observed by a caller that
holds and routes a `CompilationContext` through the supported internal
compatibility kwarg; they are not attached to public results, `time_dict`, or
native requests. Internally constructed owners remain compilation-local, and
the benchmark observes them only with a temporary test/tool interception.

No stage/timing/analysis cache, invalidation or preservation protocol,
dependency graph, callback registry, dynamic-dispatch bag, reflection,
signature inspection, global singleton, mutable registry, dictionary-of-Any
configuration, dumping, generalized tracing, or telemetry was introduced.

The final focused Phase-2/common command reports 583 passed in 1.42 seconds;
`test_compiler_stage_timing.py` contributes 60 executable cases. The changed
benchmark/value-boundary command additionally reports 40 passed in 12.08
seconds. The required
schedule/codegen matrix reports 82 passed in 354.74 seconds. The four exact
source-anchor tests report 4 passed in 0.49 seconds and preserve:

- CSR-by-dense: 2,505 bytes,
  `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151`;
- DS: 7,117 bytes,
  `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f`;
- DSS: 8,660 bytes,
  `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e`;
- all-COO SDDMM: 3,543 bytes,
  `de94b08752077a621c5e411ce0dcbb40e8bcbeacb9bce3824dd6019e2d2bd29d`.

The production C++ flags, linker flags, source-derived name, and preamble
digest JSON is byte-identical to the clean `cdb12bc` predecessor; both files
hash to
`4342e155548e81f8524b69a84c6e1d1b114f71de10ad6832b49a23e39257023a`.
No emitted production input changed, so the binding policy waives a ceremonial
42-input runtime rerun. The retained clean M5 and Redwood archives remain the
applicable evidence and match their required SHA-256 values exactly:
`816430d435d76dbe47277e921228d45c7387fc93d110bc37cc70f173982dfb77`
and
`31baca05f50d8f51483aa1d7d5b77f19f93550ebb95e46209d8dcbbd935dbc7e`.
The all-COO declaration is unchanged and remains a separate Phase-2 blocker.

The full `pytest -q -m "not perf" tests` command reports 1,094 passed, 14
skipped, three deselected, one warning, and one inherited failure in 1,914.16
seconds. The only failure is
`tests/test_scorch/test_perf.py::test_spmm_dd_ds_dd_tiled_time`, the same
pre-existing `IndexError` in `CINLowerer.lower_Where`; it is not fixed or
marked by this work. There are no new full-suite failures.

The accepted compiler-latency comparison used exactly one final committed
candidate run from detached clean worktree `141cbc5` and the retained control.
The candidate artifact is
`/tmp/scorch-phase2-stage-timing-final/latency-141cbc5-m5.json`, SHA-256
`c611ff9f7be622db04b1ddc16bb13368b6b3bf38aa7bba2344922e4ccb502a05`;
its metadata records exact revision `141cbc59b8104aeb2b66fcdcfdda38088a47dd3b`,
empty status, and the isolated source root. The retained `cdb12bc` predecessor
matched SHA-256
`1f859021940efff5d3b48c51e01d5280949591b47c1d0ec8fadc5c55c3e60572`
and the exact 5-warmup/30-sample configuration. An earlier provisional agent
violated the gate policy before the corrective work: it unnecessarily created
`latency-a3ee6a3-m5-fresh.json` (SHA-256
`9b126c5a3ed4fe64adfd164bb9846c41159153caefd3f4b0f59c437778a72133`)
even though `cdb12bc` and `a3ee6a3` differ only by documentation, then ran the
unchanged superseded `d985f74` candidate twice (`latency-d985f74-m5.json`,
SHA-256
`2ddb9efc87dfba5f561988e9981d01bd836547618271a707902c03bcc28c8b21`,
and `latency-d985f74-m5-quiet.json`, SHA-256
`efb3bcb8ef6dfcf3825fe0ea5bee8fccfbfdddb083c7080106e8be12a80bfbe3`).
All three have clean 5/30 metadata, but the redundant control and repeated
unchanged candidate violate the settled run policy and are invalid/unused for
this closure; they were not reclassified. The corrective final candidate did
not rerun any control and ran `141cbc5` once. The old `14b110b` control also
remains invalid, unused, and unreclassified.

Predecessor -> candidate compatibility p50/p95 milliseconds and candidate
ratios are:

| Case | Predecessor | Candidate | p50/p95 ratio |
| --- | --- | --- | --- |
| small dense | 1.475 / 1.713 | 1.417 / 1.666 | 0.961 / 0.972 |
| reduction | 1.354 / 1.551 | 1.325 / 1.413 | 0.979 / 0.911 |
| CSR intersection | 1.427 / 1.516 | 1.400 / 1.493 | 0.981 / 0.985 |
| sparse union | 1.362 / 1.464 | 1.329 / 1.393 | 0.976 / 0.952 |

Every ratio is below 1.10, so there is no accepted exception. Extending the
legacy marker through validated K produces canonical p50/p95 values of
1.710/1.966, 1.617/1.703, 1.693/1.786, and 1.627/1.688 milliseconds. The
corresponding endpoint-extension p50/p95 values are 0.291/0.324,
0.294/0.306, 0.289/0.310, and 0.292/0.314 milliseconds.

The predecessor-compatible marker is recorded at entry to
`_prepare_jit_build`, after K has begun and after source-derived naming; the
0.289-0.294 ms p50 endpoint extension is therefore the request/carrier suffix
of K, while the K record covers the whole stage. Across the four cases, total
canonical per-compilation stage p50 ranges are F 0.013-0.033 ms, N
0.007-0.020 ms,
S 0.141-0.240 ms where scheduling runs, A 0.023-0.026 ms, L 0.396-0.770 ms,
nested R 0.007-0.010 ms, C 0.028-0.063 ms, and K 0.387-0.394 ms. The auto
corpus has no explicit SL stage. Small dense/reduction genuinely execute three
N and four S runs because the dual-path arms are retained; CSR intersection
executes one N and two S runs; sparse union is the manual path. These totals
include real repeated work and do not merge or reinterpret pass records. The
isolated timing-owner/context plumbing assertion remains below the
1 ms ceiling. A preliminary dirty-worktree microprobe was discarded; the
5,000-sample rerun from a detached clean `141cbc5` worktree measured 2.583
microseconds p50, 2.708 microseconds p95, and 46.708 microseconds maximum for
context construction, begin, complete, and immutable record access.

All 14 changed Python files are Black-clean. Exact `a3ee6a3` comparison gives
eight Flake8 findings at both revisions and zero regressions: CIN-lowerer C901
moves 53 -> 52, einsum 72 -> 70, and scheduler remains 41; the remaining
F841/F401 findings are inherited. Comparable mypy runs report 124 errors in
five current files versus 128 in five baseline files, with zero added
diagnostics and four inherited `ops.py` diagnostics removed. The new timing
owner, timing test file, and benchmark tool are individually mypy- and
Flake8-clean.

No `csrc` or design file, benchmark output, generated extension, plot, cache,
or unrelated repository file changed. The user-owned `.gitignore` modification
and untracked research/benchmark material remain untouched and uncommitted.

#### Legacy workspace-matmul prototype removal (2026-07-14)

Commit `a37371c` removes the 2023 research-prototype `matmul_wksp` boundary,
its top-level export, and its function-owned module cache. No production source
called that boundary: tuned `matmul` already delegates compiler misses to
`einsum`, while workspace insertion and accumulation policy belong to the
scheduler. The removal therefore deletes duplicate public orchestration rather
than a compiler capability.

Workspace CIN (`Workspace`/`Where`), scheduler insertion, legacy adaptation,
CIN/LLIR lowering, compressed-output passes, generated C++, and native
workspace types remain unchanged. Prototype-only runtime/timing tests and the
obsolete compiler lane in the legacy native-variant benchmark were deleted.
Useful sparse-output and nine-pair input-format matrices now execute through
the supported production `einsum` path; direct scheduler, CIN, codegen, and
compressed-Where tests continue to force and inspect exact workspace shapes.
The public API test proves that the old name is absent from the package,
`__all__`, and `scorch.ops`.

Focused API/timing/scheduler/direct-workspace verification reports 88 passed in
192.28 seconds. The full `pytest -q -m "not perf" tests` command reports 1,089
passed, 14 skipped, three deselected, one warning, and one
`test_spmm_dd_ds_dd_tiled_time` failure in 2,047.62 seconds. That failure was
subsequently proven to be a compiler-refactor test-migration regression, not an
inherited baseline failure; its audit and correction are recorded immediately
below. All four generated source anchors remain exact. Black is clean on the
five changed Python files;
comparable Flake8 runs improve from 14 to 11 inherited findings and comparable
mypy runs improve from 85 to 82 inherited errors, with no new diagnostic. A
strict Sphinx build has the same 23 inherited unresolved-reference warnings as
pre-removal `697cb9c` and no removal-related warning. No compiler latency or
runtime gate was rerun because no remaining production compiler path, emitted
source, build input, or native code changed.

#### Tiled SpMM regression correction (2026-07-14)

Commit `a969d65` corrects the omitted repository-caller migration behind
`test_spmm_dd_ds_dd_tiled_time`; the test remains in the ordinary non-performance
suite and is not hidden or reclassified. A source-isolated history audit proves
that this was not an inherited failure and was not introduced by semantic
`LoopPlan` verification. Exact `9dcd06c` passes the full-size test, while its
child `6d199fe` (`refactor(frontend): construct CIN assignments explicitly`) is
the first failing revision. That commit began normalizing/detaching
`lower_and_exec_cin` input before lowering. Exact `9b952d7` and `d7680f4`, both
descendants of `6d199fe`, reproduce the same `CINLowerer.lower_Where`
`IndexError` when their own `src` trees are forced onto `PYTHONPATH`; a historical
pass attributed to `9b952d7` is therefore not reproducible under source-isolated
execution. Editable-install resolution to a different worktree was independently
observed as a concrete hazard during this phase.

The 2024-era test embedded an execution schedule directly in mutable CIN by
constructing `k = k_out + k_in` and relying on `TileSizeVar` side effects on
`k_out`, `k_in`, and the workspace. Canonical normalization correctly removes
those legacy schedule backlinks. The lowerer consequently treated
`accum_c[k_in]` as an untiled workspace and attempted to infer its extent from
a dense access directly indexed by `k_in`; no such access exists because the
dense operand and result use derived `k`, producing the empty-list lookup.
Preserving the backlinks in normalized semantic CIN would violate the canonical
no-schedule-metadata ownership boundary, and guessing an extent in
`lower_Where` would not restore the tile lifetime.

The corrected test expresses the same decision through
`Schedule(loop_order=("i", "j", "k"), tiles=(TileSpec("k", 4096,
placement="child_of:i", accum="stack"),))` and calls supported production
`matmul`. Direct comparison of legacy-manual and canonical-plan lowering shows
the same loop nesting, bounds, stack zero initialization, CSR traversal, derived
coordinate and ragged guards, accumulation, output copy, and OpenMP placement;
only the scheduler-owned workspace spelling changes from `accum_c` to `wksp`.
The exact 4,096 by 4,096 workload passes, including the original `torch.allclose`
check and native timing observation. This is a test-only correction: no
production source, emitted-source anchor, build input, latency result, or
runtime archive changed.

#### Semantic LoopPlan legality closure (2026-07-14)

Commits `f63f8c5`, `f2e693f`, `aaa293b`, `77bad30`, `72f5711`, `138642e`,
`b57c899`, `87014ac`, and `fd0ff9a` close the design-canonical semantic
`LoopPlan` trust-boundary blocker. `verify_loop_plan` and
`verify_scheduled_cin` now prove legality from detached normalized CIN, one
fresh immutable canonical analysis result, and the exact frozen plan. Passing
through the public `Schedule` adapter is no longer part of the correctness
argument.

The pre-implementation audit covered every `LoopPlan`, nested plan-value, and
`ScheduledCIN` constructor; both artifact verifiers; the public
`Schedule`/`TileSpec`/`RelayoutSpec` adapter; auto, explicit, tuned, forced,
fallback, direct, and standalone scheduler entries; plan replay in
`legacy_cin_adapter`; CIN-lowering and code-generation entry points; compiler
stage S/A ownership and failure propagation; schedule/dispatch/kernel/build
cache consumers; and direct callers that retain or replace CIN, plans,
analyses, schedules, or prior results. It also traced the legacy checks in
`scheduler.py`, the structural checks in `loop_plan.py`, all result and operand
formats/mode orders, free/reduction discovery, affine and panel tiling,
workspace insertion, operand packing, result tiling, explicit and implicit
parallel selection, ragged placement, and the compatibility paths that lower
already-built `ScheduledCIN` values.

The audit found additional direct-artifact bypasses beyond the three recorded
in the prior handoff. An explicit affine tile could split a sparse operand
coordinate, and a child panel could be placed under the affine outer tile of
its own parallel CSR row; these now reject as `sparse_affine_tile` and
`panel_parallel_scope`. Stack accumulation and no-tile sparse-output workspace
replay accepted non-additive reductions even though current workspace lowering
zero-initializes and emits additive updates; these now reject as
`stack_reduction_operator`, `workspace_reduction_operator`, or
`auto_reduction_operator` as appropriate.

Adversarial replay review also found unsupported scope/provenance shapes:
existing-workspace CIN accepted non-auto or explicit decisions,
multi-assignment terminal `Where` programs accepted decisions or derived
workspace lifetimes that replay could not represent, loop-free scalar CIN
accepted non-auto provenance, and a root reduction could insert a root
`Where` before applying a tile. These now fail closed as
`workspace_plan_provenance`, `multi_assignment_schedule`,
`scalar_plan_provenance`, and `root_workspace_tiling`. Supported
no-decision auto workspace/scalar paths remain replayable. Canonical additive
auto reduction tiling remains accepted: its derived workspace spans the inner
reduction tile and replay succeeds. Independent review exercised 20,880
dense/SpMM affine combinations (884 accepted, with no replay mismatch), then
rechecked all later counterexamples on `fd0ff9a`; no concrete semantic bypass
or supported-scheduler false rejection remains.

`CINAnalysis` gained only the missing typed immutable facts, through the
existing zero-field frozen `AnalysisRunner`:

- `AccessLayoutInfo` records stable access/tensor IDs, logical and physical
  storage `IndexId` order, typed level kinds, physical extents, lexical scope,
  access role, and workspace role;
- `AssignmentInfo` records stable assignment/LHS/RHS IDs, update operation,
  result indices, assignment-local reductions, and exact multiplicative-access
  provenance; and
- immutable access-layout/assignment side tables and deterministic order tuples
  are returned with the existing analysis. No fact is attached to CIN or a
  plan, and every verifier call recomputes its analysis.

The semantic proof enforces these rules:

- bound loops are exactly the disjoint free/reduction union, appear exactly
  once in the plan, and form the single root scheduling prefix supported by the
  transitional replay seam; loop-free CIN admits only no-decision auto
  provenance;
- result levels follow physical storage/mode order for dense, compressed, and
  coordinate outputs; every physical sparse parent dominates its dependent
  child, including non-default mode orders and nested sparse levels; singleton
  levels fail as an explicit unsupported feature;
- affine targets must be dense in every compatible access, explicit reduction
  tiling is unsupported without a spanning accumulator, and auto reduction
  tiling is accepted only with the existing additive derived-workspace
  lifetime; stack accumulation requires the sole trailing dense free axis and
  an additive reduction operator; sparse and auto derived workspaces likewise
  require additive reductions;
- sequential placement simulation admits only existing logical or already
  derived parents, rejects self/future/out-of-scope/depth violations, and
  accounts for the workspace-truncated common prefix and ragged inner loops;
- explicit parallelism selects a free result-partitioning loop, never a
  reduction or ragged inner, partitions every result write, and has private or
  partitioned workspace ownership; invalid post-reduction derived-workspace
  placement is rejected;
- existing-workspace CIN supports only its no-decision auto compatibility path;
  derived workspace, tiling, or parallel decisions on multi-assignment CIN fail
  closed; and tile replay cannot follow workspace insertion at the loop root;
- one canonical panel must target exactly one compatible compressed access
  with a dense parent, use the exact dense read bound, remain outside its
  parallel row, have legal placement/policy, and consume exactly one matching
  panel bound;
- relayout identifies one exact rank-two dense RHS access and physical axes,
  one compatible CSR contraction partner, the exact multiplicative assignment,
  pack/panel/scope/row loops, tile geometry/placement, matching dtype, result
  axes, and explicit row ownership; and
- heap accumulation and `ResultTile` are a bijection. The metadata must exactly
  describe the unique dense result access, physical prefix, trailing tile,
  additive reduction, outermost serial lifetime, and safe parallel prefix.
  All panel, relayout, result-tile, and parallel decisions are either consumed
  by these checks or rejected.

Malformed CIN, IDs, or typed plan structure still fails as
`VerificationError`. An illegal scheduling choice fails as `InvalidSchedule`;
a well-formed request that current lowering cannot represent fails as
`UnsupportedFeature`. The latter two retain `ValueError` and
`NotImplementedError` compatibility respectively. The public scheduler maps
its pre-existing built-in validation failures to those domain types inside the
same S-stage timing scope, while existing domain and verification exceptions
are re-raised unchanged.

Direct forged `ScheduledCIN` values can no longer bypass the adapter. Both
artifact entry points reject result order `k,i,j` with
`result_storage_order`, CSR child-before-parent `j,i,k` with
`sparse_parent_dominance`, and parallel reduction `j` with
`parallel_reduction`. Direct tests additionally cover dense and `ds`/`do`
sparse result ordering, `dss`/`doo` nested sparse dominance, complete loop
classification, valid auto reduction tiling, invalid explicit reduction
tiling, valid/invalid stack and auto accumulator lifetime, non-additive stack
and derived-workspace rejection, ambiguous/non-prefix/multi-assignment/scalar
scheduling scopes, existing and root workspace replay, result/workspace
ownership, duplicate/conflicting panels, direct/heap relayout axes, and heap
result-tile lifetime. Representative
auto, explicit, tile-j, tile-ijk, stack, heap, direct, relayout, tuned,
regblock, and fallback plans remain accepted.

Scheduling verification failures preserve stage ownership. A public invalid
schedule fails its active S stage; a forged carrier passed directly to lowering
fails its active A stage. In each case all earlier completed records remain
exactly once, the failed stage publishes no record, all lowering/codegen/native
stages are suppressed, the owner becomes terminal, and the original domain
exception is re-raised. Tests also prove that verification does not reread
environment or `ContextVar` state, does not mutate caller-owned CIN, analysis,
plan, schedule, options, or completed results, and shares no state between two
independent plans/snapshots.

Legality diagnostics and local derived facts are non-semantic: they do not
enter plan equality/hash, option fingerprints, dispatch/kernel/build cache
keys, generated names/source, or build requests. No analysis cache,
preserve/invalidate protocol, dependency graph, callback or dynamic-dispatch
registry, reflection, signature inspection, global singleton, mutable
registry, dictionary-of-`Any` configuration, new IR, dynamic CIN metadata, or
new carrier field was introduced. `ScheduledCIN(cin, plan)` remains the exact
frozen two-field transitional carrier.

The exact canonical 18-file Phase-2/common suite reports 619 passed in 1.55
seconds. The final required 11-file scheduler/CIN/codegen matrix reports 295
passed in 370.98 seconds. The final LoopPlan file collects 45 tests. The four
source-anchor tests from the clean committed `fd0ff9a` production worktree
report four passed in 0.60 seconds and retain all recorded byte counts and
SHA-256 digests. After the tiled-SpMM caller correction, the final
`pytest -q -m "not perf" tests` run reports 1,127 passed, 14 skipped, three
deselected, and one warning in 2,088.23 seconds. It has no failure; the
full-size tiled SpMM remains unmarked and executes in that run.

Black reports all nine changed Python files unchanged. Exact `d7680f4`
comparison improves four Flake8 diagnostics to two: the inherited
`Scheduler._apply_schedule_legacy` C901 complexity-41 finding (line 2614 at
baseline, 2676 at candidate) and the unrelated pre-existing
`test_perf.py` unused local remain, while the stale tiled-test local and unused
import are removed. Mypy improves from 22 to 21 inherited `import-untyped`
diagnostics in four files because the migrated test has one fewer internal
module import; the new `loop_plan_legality.py` is clean. Strict Sphinx 8.2.3
builds at baseline and candidate both finish with exactly 23 inherited
unresolved Python-reference warnings (18 class, two exception, two attribute,
one function) and no new category.

The earlier measured `f2e693f`, `77bad30`, and `b57c899` candidates were each
superseded by later audit fixes or the final typed-helper extraction and are not
used for closure. Their retained artifacts are not reclassified. The first
`b57c899` harness launch correctly stopped before warmups or samples because the
editable import still resolved to the main worktree; prepending that detached
worktree's `src` fixed the preflight defect. It was not a measurement run.

Exactly one five-warmup, 30-sample run of final production candidate `fd0ff9a`
was taken from its detached clean worktree. Its artifact is
`/tmp/scorch-phase2-loop-plan-legality-final/latency-fd0ff9a-m5.json`, SHA-256
`ccf5caa742b753248aac0de49fe1f28dae573cb1ba57453c160ae61644c29f28`;
metadata records exact revision `fd0ff9aac4c119a09b96a19ff0fbbcb0b55eec60`,
empty status, and the isolated source root. The retained `141cbc5` predecessor
was never rerun and still matches SHA-256
`c611ff9f7be622db04b1ddc16bb13368b6b3bf38aa7bba2344922e4ccb502a05`.

Predecessor -> candidate compatibility p50/p95 milliseconds and ratios are:

| Case | Predecessor | Candidate | p50/p95 ratio |
| --- | --- | --- | --- |
| small dense | 1.417 / 1.666 | 1.505 / 1.667 | 1.062 / 1.000 |
| reduction | 1.325 / 1.413 | 1.390 / 1.545 | 1.049 / 1.094 |
| CSR intersection | 1.400 / 1.493 | 1.474 / 1.544 | 1.052 / 1.034 |
| sparse union | 1.329 / 1.393 | 1.414 / 1.566 | 1.064 / 1.124 |

The sole 1.10 crossing, sparse-union p95 at 1.124, is investigated rather than
treated as automatic rejection. Sparse union has no scheduling/LoopPlan stage,
so that tail cannot be verifier overhead. Within the scheduling stage,
predecessor -> candidate p50/p95 milliseconds are 0.240/0.281 -> 0.253/0.282
for small dense, 0.227/0.231 -> 0.234/0.247 for reduction, and 0.141/0.148 ->
0.149/0.156 for CSR intersection. The observed S-stage deltas associated with
the semantic proof are 7.5-13.0 microseconds p50 and 0.2-16.5 microseconds p95.
This single-run attribution is not an isolated verifier-cost measurement.
At the direct lowering trust boundary, legacy-CIN adaptation p50/p95 deltas are
1.3/1.0 microseconds for small dense, 0.7/0.4 for reduction, and 0.6/0.9 for
CSR intersection. Sparse union performs no plan verification yet its adapter
stage moves by 2.6/9.3 microseconds, providing an ambient-tail control.
CIN-lowering itself moves by 11.4-39.8 microseconds p50 and 22.2-65.3
microseconds p95, including the no-scheduling sparse-union case. The small
observed scheduling increment is accepted as the correctness tradeoff; the
candidate and retained control were not repeated.

The candidate latency build summaries are exactly equal to the predecessor's.
The standalone production-input JSON was first retained at `77bad30`; the
same payload was recomputed from the clean final `fd0ff9a` worktree as 493
bytes including its newline. Its C++ flags, linker flags, source-derived name,
and preamble digest are byte-identical and retain SHA-256
`4342e155548e81f8524b69a84c6e1d1b114f71de10ad6832b49a23e39257023a`.
Because valid emitted build inputs are unchanged, the two-machine runtime gate
is waived. The retained M5 and Redwood archives each still contain 42/42
correct cells and match SHA-256
`816430d435d76dbe47277e921228d45c7387fc93d110bc37cc70f173982dfb77`
and `31baca05f50d8f51483aa1d7d5b77f19f93550ebb95e46209d8dcbbd935dbc7e`.

No `csrc`, design, generated output, native extension, benchmark result,
plot, cache, or unrelated repository file is part of these commits. The
all-COO `pMask1_end` declaration remains unchanged. The user-owned
`.gitignore` modification and untracked research/benchmark material remain
untouched and uncommitted.

The canonical `CompileOptions`, manager-owned pipeline/common
analysis-runner, production/debug compiler-stage timing, and semantic
`LoopPlan` legality-verification Phase-2 blockers are **closed**. Canonical
Phase 2 is still **not formally exited**. The remaining blocker is the required
all-COO no-`pMask1_end`-declaration invariant, which requires a separately
gated emission-affecting change.

The next Phase-2 blocker is the all-COO `pMask1_end` declaration.
Do not start Phase 3 while this Phase-2 blocker remains.

#### All-COO bound-declaration and canonical Phase 2 closure (2026-07-15)

This section supersedes the immediately preceding status that named
`pMask1_end` as the remaining blocker. Commits `5740d3c` and `641ec81` close
that blocker without beginning Phase 3.

The pre-implementation audit traced the declaration through every relevant
stage. Coordinate-iterator initialization creates `int pMask1_end = 0;` as a
future-level sentinel. Before flat COO transformation, the outer `pMask0` loop
advances through that bound, coordinate resolution assigns
`pMask1_end = pMask0 + 1` and scans the group boundary, and an inner `pMask1`
loop consumes it. Flat scalar-COO lowering removes the original assignment and
scan, installs a loop-local `int64_t pMask1_end = pMask0 + 1`, and retains the
inner loop, but previously copied the zero-initialized outer prefix declaration
unchanged. Sparse-prefetch and dense-pointer passes preserved all three
remaining references. Single-iteration elimination then removed the local
derived declaration and inner loop, rewrote coordinate/value accesses directly
to `pMask0`, and left the outer zero sentinel because its characterized matcher
only consumes a direct same-sequence `base + 1` bound. Invariant-factor
hoisting preserved that now-dead root statement; result/ABI assembly retained
it in the function body, later dynamic-vector rewriting preserved it, and the
mechanical C++ emitter emitted it. That unconditional prefix retention was the
root cause.

The audit also covered generic iterator initialization, iteration-lattice
assignment and scan generation, `CINLowerer`, all managed-pass routing,
single-iteration and invariant-factor behavior, result/ABI assembly, final C++
declaration emission, and scheduler/codegen-created bounds. Every other
representative `*_end` declaration remains live: `pMask0_end` bounds the
all-COO outer loop; CSR compressed ends bound traversal and prefetch guards;
DS and DSS ends bound both compressed-Where count and fill loops; intersection
ends feed merge conditions; union ends additionally feed one-sided tails; and
nested coordinate sentinels feed boundary scans, child merges, and parent
advances. The same proof holds for nested sparse levels and non-default mode
orders. Panel, relayout, row-window, and atomic-emission bounds are created at
different seams and remain consumed. Focused tests now lock the live CSR,
DS/DSS, and non-default COO-intersection cases.

The correction is deliberately local to the successful flat scalar-COO
lowering branch. After it has constructed the replacement flat loop and its
loop-local derived bound, it filters only direct accumulated prefix statements
whose exact node shape is an exact `llir.VarInit` of an exact `llir.Var`, whose
name equals the transform's detected `end_var`, whose type is exactly
`DataType.INT`, and whose value is an exact `llir.Literal` wrapping an exact
Python `int` equal to zero. The generated prefix owns exactly one such
sentinel. The correction performs no recursive liveness walk, does not inspect
arbitrary `*_end` names, does not mutate an LLIR node, and cannot match the
loop-local `INT64` bound. It is lowering-seam lifetime cleanup, not generalized
dead-declaration elimination.

The complete generated-source change is the 22-byte deletion:

```diff
   // Initialize iterators
   int pMask0_end = Mask0_crd_tensor.size(0);
-  int pMask1_end = 0;

   #pragma omp parallel for num_threads(scorch_nthreads(-1, pMask0_end)) schedule(dynamic, scorch_chunk(pMask0_end, -1))
```

Reinserting that one line into the candidate source reconstructs the exact
3,543-byte `211ffef` source and its
`de94b08752077a621c5e411ce0dcbb40e8bcbeacb9bce3824dd6019e2d2bd29d`
SHA-256. Candidate `641ec81` is exactly 3,521 bytes with SHA-256
`53d6faaee132a5d82515235b529d7d88d16cbeefe388eba5cfae9ace5528d667`.
It contains no `pMask1_end` substring, inner `pMask1` loop, or derived-bound
assignment. It retains direct `Mask1_crd[pMask0]` and `Mask_val[pMask0]`
access, the exact `float _inv_17 = Mask_val[pMask0];` spelling, only
`_Query_val_ptr[q] * _Key_val_ptr[q]` inside the SIMD q-loop, and
`_accum *= _inv_17;` immediately afterward.

The affected source-derived identities change exactly as required:

| Identity | `211ffef` | `641ec81` |
| --- | --- | --- |
| Kernel name | `kernel_78191be4a32b` | `kernel_a67e7b1a138e` |
| Request-content digest | `0add35dde92ce72f1311ccb9ddb4234b356c3218ae72c654dd7db71f1cbf817a` | `f29034b8cc06f9817865082839e73f8b02a4a1ddfd65d2c5bec08b9af5065ed8` |
| Build-identity digest | `c680956e76a633cc102bc495ea55ddbefd0ab9731cd3c1765f303727eaf769b0` | `bb7414675cbc55384f59065d2cd6223ad55ddfb0cb50b970f8acb3b2743d10fb` |

The exact `CompileOptions` cache fingerprint remains
`398311f4f3da7e8ac654459d3f214c237e64a56ade4e041971559001a4a01e6e`,
the build-options cache digest remains
`76462d4b5e037d39b09e941b2880adac519014e8719658c6ff1f4c446519bc76`,
and the preamble digest remains
`db29715709809539883f4904c60dd1276cad5af16a5f43549cfc11375321c544`.
Flags, linker flags, function ABI, and every non-source-derived build field are
equal. Semantic frontend/codegen cache keys remain unchanged; only the affected
source-derived kernel name, JIT request, build directory, and shared-object
identity change. The unaffected production build-input comparison retains
SHA-256
`4342e155548e81f8524b69a84c6e1d1b114f71de10ad6832b49a23e39257023a`.

The unaffected source anchors remain byte-identical:

| Kernel | Bytes | SHA-256 |
| --- | ---: | --- |
| CSR×dense | 2,505 | `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151` |
| DS | 7,117 | `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f` |
| DSS | 8,660 | `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e` |

The exact canonical 18-file Phase-2/common suite reports 620 passed in 2.10
seconds. The required 11-file scheduler/CIN/codegen matrix reports 297 passed
in 498.80 seconds. `tests/test_scorch/test_loop_plan.py` remains 45 passed in
0.86 seconds. The explicit M5 all-COO structural and native command reports two
passed in 26.72 seconds; the native case compiles the committed candidate and
matches `mask * (query @ key.T)` at `atol=rtol=1e-3`. Its log SHA-256 is
`e8cd905f71cf32ee30f1988bf15895d12572a9d3b301578c6d76e516e9cf86b0`.
The full non-performance suite reports
1,129 passed, 14 skipped, three deselected, and the one inherited sparse-tensor
invariant warning in 2,419.93 seconds; its log SHA-256 is
`ac638bd7b6c88de5483aba4f8a9b71a21d6c30a5c6f08d53fb9556c22abcff2d`.

The clean detached M5 gate records exact revision
`641ec81208780031a9d6c726ac1a6bf44c9237ca`, empty status,
`macOS-26.4.1-arm64-arm-64bit`, Python 3.11.15, Torch 2.13.0, and six Torch
threads. Its candidate A/A run reports 42/42 correct cells, 21 unique builds,
identical per-cell and top-level build inputs, and machine control band
`[0.9826540709874659, 1.0176521214582699]`. The standard grid remains
byte-identical to the retained predecessor, so it does not substitute for the
separately executed changed all-COO structural/native case. Candidate artifact
`/tmp/scorch-phase2-all-coo-pmask-results/kernel-aa-641ec81-m5.json` has
SHA-256
`3b655e445d130cbfe3e394563f52498bd25675d7bb5f7c775333ef580fa7b246`;
its log, comparison log, and correctness/build summary have SHA-256
`3adb0c6b0889f8c7a960e17958d503b2cc16574dcf8d49eba57e82e038be9fcb`,
`fefddeec80b5ca91efee6e4381400f5a14da40e88081ed5e5650fab1eb92bcc1`,
and `58a8a998232171363ad79f1ce05d1cbf5fbfe07ac9ef19d98281481d16c292f4`.
The retained predecessor artifact remains
`816430d435d76dbe47277e921228d45c7387fc93d110bc37cc70f173982dfb77`.

Exactly one five-warmup, 30-sample candidate compiler-latency run was retained.
Its artifact is
`/tmp/scorch-phase2-all-coo-pmask-results/latency-641ec81-m5.json`, SHA-256
`eb231ae72b6d1ff71406ab49e819a2cf1d8e38907380410a73592ca5d8f9b673`;
the valid `fd0ff9a` predecessor remains
`ccf5caa742b753248aac0de49fe1f28dae573cb1ba57453c160ae61644c29f28`.

| Case | Predecessor p50/p95 ms | Candidate p50/p95 ms | Ratio |
| --- | --- | --- | --- |
| small dense | 1.505 / 1.667 | 2.093 / 2.492 | 1.391 / 1.495 |
| reduction | 1.390 / 1.545 | 1.981 / 2.504 | 1.426 / 1.621 |
| CSR intersection | 1.474 / 1.544 | 2.367 / 3.336 | 1.607 / 2.161 |
| sparse union | 1.414 / 1.566 | 2.282 / 2.737 | 1.614 / 1.748 |

These crossings are retained and investigated, not treated as automatic
rejection. None of the four latency cases reaches the changed flat all-COO
branch, and every candidate latency source byte count and SHA-256 is identical
to its predecessor. Inflation is stage-wide, including frontend construction,
normalization, legacy adaptation, C++ generation, and cases with no COO
scheduling. The full native suite began at 23:52:30 while the latency artifact
was written at 23:53:01. A same-window contention capture records load averages
`10.22/9.65/8.63`, an active Clang native build at 98% CPU, and several other
high-CPU system processes. Its SHA-256 is
`c51f8c55aa7f45a9008678a10a6d1426edac599b14d2be3645a87c4083bc8698`.
The evidence attributes the broad movement to concurrent M5 contention, not to
the inactive one-line emission correction; the single candidate run was not
repeated.

The clean Redwood candidate records the same exact revision and empty status
on `Linux-5.15.0-121-generic-x86_64-with-glibc2.35`, Python 3.11.15, Torch
2.5.1, and 24 Torch threads. Its full gate reports 42/42 correct cells, 21
unique builds, identical per-cell and top-level build inputs, and machine
control band `[0.9692050303205669, 1.031773431540329]`; the comparator exits
zero with the byte-identical runtime waiver. Candidate artifact
`/tmp/scorch-phase2-all-coo-pmask-results/kernel-aa-641ec81-redwood.json` has
SHA-256
`c3a6a110fc98614ca50111adab3b7ea5ee93ba7aff9da5715ee110946e683d8d`;
its log, comparison log, and correctness/build summary have SHA-256
`d782133ae790b2edf08780952c2ac019e3b8880e78c5e321a8ff2036bd91f2f7`,
`3f9efe57eec1c70084c09d082f64eda06b0a302e91058e4840649b5842e1d5f2`,
and `c74652c5e055de49770a070915d2d4dc9de1a5af608d0b4d6679d3d62cdb02a1`.
The retained predecessor remains
`31baca05f50d8f51483aa1d7d5b77f19f93550ebb95e46209d8dcbbd935dbc7e`.
The separately executed Redwood all-COO structural/native command reports two
passed in 28.56 seconds and log SHA-256
`3b7ad0971e310f769862732039ef981516b47d32ceb086fce6a262534f414157`.

Black leaves all five changed Python files unchanged. Exact `211ffef`
comparisons show the same five inherited Flake8 findings in
`cin_lowerer.py` and the same 68 inherited mypy errors across the five checked
files, with normalized error multisets byte-identical and no new diagnostic.
`git diff --check` is clean. Strict Sphinx finishes with exactly the same 23
inherited unresolved-reference warnings and no new category.

No managed LLIR pass changed, so pass identities, order, nesting, repeated
application, legal no-op behavior, failure records, and later-stage suppression
remain unchanged. The lowerer works on its private detached LLIR and the new
code only rebinds a fresh local statement list; caller-owned CIN, plans,
analyses, options, prior results, and independent compilations retain their
existing ownership guarantees. Exact `CompileOptions` identity,
compiler-stage timing ownership, source-derived caching, and failure
short-circuit behavior remain covered by the canonical suite.

No new IR, structured CxxIR, generalized DCE, analysis cache,
preserve/invalidate protocol, dependency graph, callback registry, reflection,
dynamic metadata/configuration, public forcing API, or mutable global state was
introduced. `ScheduledCIN(cin, plan)` remains the exact frozen two-field
carrier, and `matmul_wksp` was not restored. There is no `csrc` or design-file
change, generated output, native artifact, benchmark result, plot, cache, or
unrelated tracked file in the candidate. The user-owned `.gitignore`
modification and untracked research/benchmark material remain untouched and
uncommitted.

With the full-suite and two-machine results above passing, the final all-COO
declaration blocker is **CLOSED** and canonical Phase 2 is formally
**COMPLETE**. Phase 3 has not begun.

### First narrow Phase-3 structured-access slice (2026-07-15)

This is the first shippable **Phase 3** slice, not completion of Phase 3. It
starts exactly from `28dcca51144c7f84008d1e39bb4050c4fb9909f0` on
`refactor/compiler-ir-phase3-structured-access`. Phase 3.5, LoopIR, parallel
zero-fill extraction, Torch/C++ ABI extraction, generalized allocation
migration, and unrelated optimization work have not begun.

The committed compiler candidate is `2e2a30d666d2272afe7174d7ea5f999e167a6cd1`:

- `09acd48` — `refactor(compiler): add typed low-level access structure`;
- `1fec8f8` — `test(compiler): cover structured access migration`;
- `2e2a30d` — `fix(compiler): type schedule traversal roots`.

#### Pre-implementation inventory and seam selection

The audit considered every current low-level producer, structural consumer,
manager pass, schedule transform, result/ABI path, and C++ emitter before
selecting a seam. At the base revision there were 315 syntactic `llir.Var`
construction sites. A conservative direct-fragment classifier identified 90
expression-shaped `Var.name` producers:

| Direct base category | Count |
|---|---:|
| subscript | 62 |
| call | 13 |
| member | 7 |
| initializer | 3 |
| qualified name | 3 |
| ternary | 1 |
| arithmetic | 1 |
| **Total** | **90** |

Those sites were concentrated in `cin_lowerer.py` (70),
`iter_lattice.py` (5), `iterator.py` (6), and `schedule_lowerer.py` (9).
The audit separately followed indirect expression strings, generic rewrite
sinks, and traversal clones so a local variable could not hide from the
budget.

There were 62 `RawStmt` construction calls: 61 semantic producers and the one
common traversal clone. Their exact base/current distribution is unchanged:

| File | Calls | Classification |
|---|---:|---|
| `cin_lowerer.py` | 17 | ABI/prologue, allocation, parallel, and compatibility output |
| `compressed_where_openmp_pass.py` | 22 | characterized count/fill and workspace compatibility output |
| `result_write_pass.py` | 15 | characterized sparse-result assembly/writes |
| `schedule_lowerer.py` | 3 | reusable vector allocation compatibility output |
| `loop_invariant_factor_pass.py` | 2 | generated invariant declaration/update output |
| `dense_pointer_hoist_pass.py` | 1 | generated pointer declaration output |
| `sparse_prefetch_pass.py` | 1 | generated prefetch output |
| `llir_traversal.py` | 1 | detached clone, not a producer |

The audited expression consumers were sparse prefetch, dense-pointer hoisting,
single-iteration elimination, invariant-factor hoisting, dynamic-vector
rewriting, result writes, compressed-Where, schedule access redirection, packed
relayout construction, and final C++ emission. Coordinate/position accesses,
indexed result writes, workspace accesses, member/call/allocation expressions,
and pointer/declaration strings share some consumers but require additional
typed provenance or a typed lvalue seam. Migrating them together would have
expanded this slice into a generalized expression parser or a broad CxxIR.

The selected end-to-end seam is therefore **non-workspace logical tensor value
reads produced centrally by `CINLowerer.lower_TensorAccess`**. This is the
smallest seam that removes a real string-encoded C++ subscript from production,
survives the manager-owned LLIR pipeline, activates existing optimizations, is
consumed by schedule relayout, and reaches the emitter as one representation.
It includes synthetic packed and dense-hoisted reads so the migrated production
path does not fall back to a parallel string representation.

One narrow dependency was required: the existing `TensorAccessMetadata` and
private relayout/result-tile carriers could not keep display names as semantic
identity. They now carry the already-canonical `AccessId`, `SymbolId`, and
`IndexId` values. Occurrence identity is retained by `AccessId`; schedule
selection intentionally matches the logical `SymbolId`/ordered `IndexId` tuple
and role. Display spellings remain separate and byte-identical. No new identity
allocator or registry was introduced.

#### Implemented structural contract

- Existing `llir.ArrayAccess(array, index, tensor_access=None)` is now a frozen
  dataclass with typed expression children and exact construction validation.
- `TensorAccessMetadata(access_id, tensor_id, index_ids, role)` is frozen and
  validates every typed identity field. Common walking and rewriting revalidate
  exact metadata fields on both `ArrayAccess` and the transitional
  metadata-bearing `Var` result-write form, including forged objects.
- Non-workspace value reads lower to
  `ArrayAccess(Var("<tensor>_val"), Var("<physical-position>"), metadata)`.
  Workspace value reads remain the characterized flat compatibility form.
- The common exact-type walker/rewriter owns traversal and detachment. Unknown
  access subclasses, unknown children, invalid metadata, and malformed roots
  fail at the owning traversal/codegen/schedule stage.
- C++ emission owns postfix precedence and renders the new structure with the
  exact prior spelling and whitespace. Provenance is non-emitting. Structural
  expression equality intentionally excludes provenance, as the prior `Var`
  contract did; semantic consumers compare typed IDs explicitly.
- Dense-pointer hoisting maps structured value reads to structured pointer
  reads while retaining its direct legacy compatibility input. It never shares
  mutable replacement children.
- Single-iteration elimination rewrites exact typed access-index symbols rather
  than parsing the whole subscript. Invariant-factor hoisting determines
  dependence from structured pointer/index children. Compressed-Where and the
  common rewriter rebuild frozen accesses and preserve provenance.
- Schedule relayout/result selection uses stable IDs. Access rewriting now uses
  the common exact traversal, is detached and repeatable, clones each
  replacement, and fails closed for unknown children. Packed reads and relayout
  source loads are structured `ArrayAccess` plus typed arithmetic; packed
  destination and result stores remain the explicit indexed-lvalue debt.
- Sparse prefetch already understood `ArrayAccess`; result writes already render
  structured right-hand sides; dynamic-vector rewriting does not own this input
  read seam. No compatibility shim or second production representation was
  added for migrated non-workspace reads.

This is the minimum coherent typed structure: one existing access node, typed
children, and stable provenance. Adding member, call, allocation, pointer, or
indexed-store nodes is neither necessary for this seam nor justified in the
same change.

#### Remaining measured compatibility budget

`tests/test_scorch/test_llir_string_budget.py` locks every current `Var`
constructor, direct category, known indirect sink, generic rewrite sink, and
`RawStmt` producer. After the slice:

- 326 total `Var` constructor calls are inventoried;
- 87 are directly provable expression strings: 59 subscript, 13 call, 7 member,
  3 initializer, 3 qualified, 1 ternary, and 1 arithmetic;
- all remaining 239 constructor arguments are separately counted by file; the
  manual audit identifies eight indirect expression/compatibility sinks plus
  the one common traversal clone among them;
- 11 generic string-rewrite compatibility sites are explicitly locked across
  `cin_lowerer`, compressed-Where, dense-pointer, single-iteration, and
  dynamic-vector rewriting;
- `RawStmt` remains 62 calls / 61 producers with the file budget above.

The direct expression-string budget therefore falls from 90 to 87. The larger
constructor count reflects the intended replacement of one opaque value-read
`Var` with an `ArrayAccess` and typed `Var` children; it is not new generated-C++
string debt. Coordinates/positions, result/workspace lvalues, member/call and
allocation forms, result assembly, raw prefetch/pointer declarations, and
schedule-only stores remain future measured slices.

#### Correctness, ownership, and failure evidence

Focused tests cover exact construction, freezing, type hints, structural and
semantic equality, metadata validation, precedence, byte-exact emission,
common traversal, unknown subclasses/children, forged metadata, detached
rewrites, no-ops, repeated application, replacement non-sharing, and stable-ID
schedule matching. Production regressions cover CSR-by-dense, DS, DSS,
all-COO, sparse intersection/union, compressed-Where, nested sparse levels,
non-default mode order, panel/relayout, and register-blocked paths.

Caller-owned CIN/access objects and source LLIR remain unchanged; each manager
pass works on a detached tree. Existing canonical tests prove plans, analyses,
`CompileOptions`, prior results, and independent compilations retain their
ownership. Two replacements in one schedule rewrite are distinct objects.
Compiler-stage identities/order, exact options identity, timing ownership,
managed-pass records, failure propagation, and later-stage suppression remain
covered by the canonical matrices and full suite.

Exact generated-source anchors remain:

| Path | Bytes | SHA-256 |
|---|---:|---|
| CSR-by-dense | 2,505 | `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151` |
| DS | 7,117 | `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f` |
| DSS | 8,660 | `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e` |
| all-COO SDDMM | 3,521 | `53d6faaee132a5d82515235b529d7d88d16cbeefe388eba5cfae9ace5528d667` |

Native PyTorch-comparison evidence is included in the scheduler/codegen matrix
and full non-performance suite. The full run covers structurally activating
CSR-by-dense, DS/DSS, sparse intersection, sparse union, compressed-Where,
nested sparse, all-COO, non-default mode-order, relayout/panel, and
register-blocked paths.

#### Exact source/build identity and runtime waiver

A deterministic same-path pre-native capture compared `28dcca5` with the
committed compiler candidate `2e2a30d`. The raw C++ files, 68,671-byte preamble,
evaluate signatures, source-derived kernel names, codegen keys, semantic/build/
full `CompileOptions` keys, request fields, compiler/linker flags, ABI/index
policies, build-option keys, request keys, build identity, prepared cache key,
build directory, and `.so` path are byte-for-byte identical for the four anchor
families. `CompileOptions` identity reaching the lowerer is exact.

- preamble SHA-256:
  `db29715709809539883f4904c60dd1276cad5af16a5f43549cfc11375321c544`;
- complete anchor manifest SHA-256:
  `d2dfcf5cb4299be88f2bf5b35a047bdacf2a0e8a65ab03e17d20ca89a0a17024`.

The full 42-cell generated-SpMM corpus was then captured without entering
native compilation. Every cell matches the retained Phase-2 M5 build summary
for source digest/bytes, function list, exact compiler/linker flags, and kernel
name. The richer same-path base/candidate grid capture, including per-source and
request/prepared/build identities, is byte-identical with SHA-256
`204d80f7df45eb222e5308ab72bbaf0aaa326aa0a7e7677d17ba289b535b0dc6`.
The retained M5 and Redwood archives have identical source digest/byte fields in
all cells; their source-derived names differ only by the retained platform/Torch
build identity as expected.

Because every corpus source and build input is unchanged, the canonical
byte-identical runtime waiver applies. No new M5 or Redwood runtime grid was
run. The retained artifacts were re-hashed without reclassification:

- latency predecessor: `ccf5caa742b753248aac0de49fe1f28dae573cb1ba57453c160ae61644c29f28`;
- M5 kernel predecessor: `3b655e445d130cbfe3e394563f52498bd25675d7bb5f7c775333ef580fa7b246`;
- Redwood kernel predecessor: `c3a6a110fc98614ca50111adab3b7ea5ee93ba7aff9da5715ee110946e683d8d`;
- final Phase-2 manifest: `f9509f16f44ec61373b71e24494c066ee60017892b9e807f8a43cb215cdf0460`.

#### Compiler latency and attribution

The valid uncontended production run used the clean committed
`2e2a30d666d2272afe7174d7ea5f999e167a6cd1` worktree, five warmups, and 30
samples. Native build/execution was excluded. Artifact:

`/tmp/scorch-phase3-structured-access-results/latency-2e2a30d-m5.json`

SHA-256:
`2dac0b53298ba2215f92d6e1a500af369869825700127aa8bcf0db7ef5d5288b`.

| Case | Candidate p50/p95 ms | New/old p50 | New/old p95 | Decision |
|---|---:|---:|---:|---|
| small dense | 1.588 / 1.933 | 1.055 | 1.160 | investigate p95 |
| reduction | 1.457 / 1.589 | 1.049 | 1.028 | target |
| CSR intersection | 1.556 / 1.714 | 1.056 | 1.110 | investigate p95 |
| sparse union | 1.580 / 1.864 | 1.118 | 1.190 | investigate p50/p95 |

The 1.10 threshold is an investigation trigger, not automatic rejection. The
crossings are explained primarily by CIN lowering, where one opaque value-read
node becomes an access plus typed children and each managed detached rewrite
must visit, validate, and rebuild that structure. CIN-lowering p50/p95 absolute
deltas are +0.031/+0.059 ms (small dense), +0.033/+0.053 ms (reduction),
+0.069/+0.082 ms (intersection), and +0.105/+0.193 ms (union). LLIR-to-C++
generation changes are only +0.001 to +0.005 ms at p50. Kernel-name/build-request
assembly p50 improves in all four cases, and the canonical endpoint extension
also improves, so the regression is neither source/cache/build-request work nor
an unexplained downstream effect. The worst total absolute p95 increase is
0.298 ms, with every candidate p95 still below 1.94 ms. This is recorded as a
modest, attributed structural-cost exception for the first typed access slice;
future Phase-3 slices must not allow it to compound without review.

#### Verification record

All Python/test/tool commands activated the `scorch` conda environment first.

- changed-file focused suite: **376 passed** in 118.11 s;
- canonical 18-file Phase-2/common suite: **661 passed** in 1.88 s;
- required 11-file scheduler/CIN/codegen matrix: **303 passed** in 388.34 s;
- `pytest -q -m "not perf" tests`: **1,176 passed, 14 skipped, 3 deselected**
  with one inherited PyTorch sparse-invariant warning in 2,059.22 s;
- exact four-source anchor check after final structural code: **4 passed**;
- Black on all 20 changed Python files: clean (the repository targets Python
  3.15 while the required environment is Python 3.11, so Black reports its
  existing target-version safety-check warning);
- Flake8 on every changed test/new file: clean. Changed production files report
  the exact same seven normalized findings as `28dcca5` (two F841, two F401,
  two C901, and one F541), with only shifted line numbers;
- mypy on changed production files: the same 45 normalized inherited findings
  as `28dcca5`; the nine existing changed tests have the same 33 `py.typed`/
  import-stub findings as the base; the new budget test is clean;
- strict Sphinx command completed and reproduced the exact base failure: both
  base and candidate report the same 23 unresolved-reference warnings under
  `-W`; there is no new warning;
- source/build capture and the valid latency run used clean detached committed
  worktrees and `/tmp` outputs only.

No analysis cache, preservation/invalidation protocol, dependency graph,
generalized DCE, reflection, signature inspection, callback/dynamic registry,
dictionary-of-`Any` configuration, mutable global singleton, forcing API,
complete CxxIR, LoopIR, Phase-3.5 interpreter, `csrc` change, design-document
change, generated tracked output, benchmark artifact, or unrelated tracked file
was introduced. `ScheduledCIN(cin, plan)` remains the exact frozen two-field
carrier and `matmul_wksp` remains removed. The user-owned `.gitignore` change
and all untracked research/benchmark material remain untouched and uncommitted.

This closes only the first narrow Phase-3 **structured non-workspace tensor
value-read access** slice. The measured budgets above identify the remaining
string/`RawStmt` families for later independently gated slices. Phase 3 remains
in progress; Phase 3.5 and LoopIR have not begun.

### Phase-3 structured indexed-store slice complete (2026-07-15)

The second independently shippable Phase-3 slice is complete. The compiler
implementation commit is `dd979ef` and the test commit is `d437174`. This slice
starts from `878dd24`, retains the first structured-read slice unchanged, and
does not claim that Phase 3 is complete.

#### Complete pre-implementation inventory and seam selection

The locked budget was re-audited from the AST before implementation. All 326
`llir.Var` constructor sites were accounted for. The 87 directly provable
expression strings remained exactly 59 subscript, 13 call, 7 member, 3
initializer, 3 qualified, 1 ternary, and 1 arithmetic. The other 239
constructors included the eight known indirect compatibility sinks and the one
common traversal clone. The generic rewrite budget was 11 and `RawStmt` was 62
calls / 61 producers.

The 59 direct subscript strings split as follows:

| Producer | Indexed lvalue | Read | Declaration | Total |
|---|---:|---:|---:|---:|
| `cin_lowerer.py` | 22 | 20 | 1 | 43 |
| `iter_lattice.py` | 5 | 0 | 0 | 5 |
| `iterator.py` | 0 | 6 | 0 | 6 |
| `schedule_lowerer.py` | 4 | 1 | 0 | 5 |
| **Total** | **31** | **27** | **1** | **59** |

Twenty-seven lvalues were direct `Assign(var=Var(...))` constructors: 19 in
CIN lowering, 5 in the iteration lattice, and 3 in schedule lowering. The four
other lvalue constructors flowed through the existing post-op/store helpers: 3
in CIN lowering and 1 compact schedule target. The one declaration was the
fixed-size workspace declaration and was not an assignment target.

The 31 lvalues cover logical result-value writes, result coordinate/position
assembly, dense-workspace element stores, dynamic intermediate vectors,
all-COO grouping vectors, post-op output stores, iteration-lattice result
stores, packed-relayout destinations, compact/heap result destinations, and
the final schedule copy to output storage. DS, DSS, CSR-by-dense, sparse
intersection/union, nested sparse levels, all-COO SDDMM, non-default mode
orders, panel/relayout, register-blocked, and atomic-scheduling construction all
reach one or more of those shared producers; none requires a second indexed
target family.

The statement audit also found 13 indexed stores embedded in `RawStmt`: one
dense-workspace `memcpy` destination, five compressed-Where auxiliary/output
stores, and seven result-write assembly stores. Ten standalone stores were in
scope: all seven result-write stores plus `_count[row]`, `_offset[0]`, and
`pos_data[0]` in compressed-Where. Three compound statements remain outside
this seam: the `memcpy` call destination, the prefix-sum loop body, and the
position-copy loop body. Structuring those would require a typed call/nested
statement family rather than an indexed-lvalue node alone.

All regex, string-rewrite, clone, and rendering consumers were traced through
CIN/iteration-lattice lowering, compressed-Where count/fill construction,
result-write assembly, dense-pointer, single-iteration, invariant-factor,
dynamic-vector, schedule relayout/result-tile rewriting, common traversal,
manager verification, and final code generation. The audit showed that the
existing frozen `ArrayAccess` is a legal lvalue carrier; a second store node,
general lvalue hierarchy, expression parser, ABI rewrite, or complete CxxIR was
not necessary. Indexed stores were therefore selected over member/call access
as the minimum coherent seam.

#### Implemented structural contract

- `Assign.var` is now the explicit `AssignmentTarget = Var | ArrayAccess`.
  Exact `Var` targets accept only identifiers or dotted member paths. Exact
  `ArrayAccess` targets require an identifier/member base plus a deliberately
  small typed index grammar; arbitrary rvalue expressions, flat subscript
  strings, malformed children, forged metadata, and unknown subclasses fail
  closed.
- The existing frozen `ArrayAccess` is reused for every migrated production
  indexed target. There is no string fallback for the supported paths and no
  parallel store representation.
- Logical result-value targets carry frozen `TensorAccessMetadata` with typed
  `AccessId`, `SymbolId`, and ordered `IndexId` children and the
  `RESULT_WRITE` role. Result-write matching uses `SymbolId`, not a rendered
  array name. Physical coordinate/position and schedule-local arrays remain
  provenance-free because their scoped storage identity is not logical tensor
  identity.
- `ResultWriteContext` now carries the exact result `SymbolId` and result-value
  pointer `DataType`. Production float, double, C int, int32, and int64 output
  stores receive exact pointer types. A standalone custom legacy C spelling
  remains explicitly `NO_TYPE`; the pass never invents a false type.
- Common traversal validates and detaches assignment targets, recursively walks
  structured indices (including nested typed input reads), clones replacements,
  rebuilds frozen accesses, and rejects unknown structure at the owning stage.
  Codegen validates again and owns precedence-aware lvalue emission.
- Schedule access rewriting preflights every replacement on a detached target
  before mutating the statement tree. Both an invalid outer result replacement
  and an invalid nested read replacement leave caller-owned LLIR unchanged.
- Dense-pointer, single-iteration, invariant-factor, dynamic-vector,
  compressed-Where, result-write, and schedule consumers recognize the typed
  target directly. Coordinate/position, compact, packed, all-COO grouping, and
  dynamic-vector storage bases retain their exact storage/pointer `DataType`.
- Count/fill result-write passes remain independent and repeatable in their
  documented modes. Legal no-ops detach their result; malformed roots,
  contexts, targets, metadata, replacements, and unknown nodes fail before
  partial publication. Manager failure records, later-stage suppression, and
  compiler-stage timing ownership are unchanged.

This is the minimum design because the operation already distinguishes a
scalar/member lvalue from an indexed lvalue. Allowing every `Expr` as an
assignment target would weaken typing, while adding `Store`, `LValue`, member,
call, allocation, and nested-statement families together would exceed this
slice. Reusing `ArrayAccess` plus a narrow target union expresses exactly the
production seam and nothing more.

#### Remaining measured compatibility budget

`tests/test_scorch/test_llir_string_budget.py` now locks the complete post-slice
budget and separately forbids any direct production `Assign` from
reintroducing a string-encoded expression target. The only direct expression
`Var` targets allowed by that seam assertion are the two audited dotted member
paths.

- 371 total `Var` constructor calls;
- 56 direct expression strings: 28 subscript, 13 call, 7 member, 3 initializer,
  3 qualified, 1 ternary, and 1 arithmetic;
- 315 other constructor arguments, including the same nine known indirect
  sinks/clones;
- 10 generic string-rewrite compatibility sites;
- 52 `RawStmt` calls / 51 producers.

The 28 remaining direct subscript strings are exactly 21 in CIN lowering, 6 in
iterator construction, and 1 schedule read. They contain 27 rvalue reads and
the one fixed-workspace declaration; none is a direct indexed assignment
target. The three remaining raw indexed-store statements are the characterized
`memcpy` destination and two compound compressed-Where loops above. Their
future migration requires independently justified call/statement structure.

#### Correctness, ownership, and failure evidence

Focused tests lock exact construction, type hints, freezing, equality,
validation, target/rvalue separation, postfix and arithmetic precedence,
byte-exact emission, common walking/rewriting, replacement ownership,
detachment, legal no-ops, repeated application, malformed input, unknown-child
and subclass failure, stable identity, and exact float32/float64/integer
storage types. Every migrated producer has a structural regression, including
the ten formerly raw standalone stores.

Exact CSR-by-dense, DS, DSS, and all-COO source anchors remain unchanged.
Native PyTorch comparisons in the scheduler/codegen matrix and full suite cover
the activating CSR-by-dense, DS/DSS, intersection, union, compressed-Where,
nested-sparse, all-COO, non-default mode-order, relayout/panel,
register-blocked, workspace, and dynamic-vector paths.

Caller-owned CIN, LLIR, access metadata, analyses, exact `ScheduledCIN(cin,
plan)`, schedules, plans, `CompileOptions`, and prior pass results remain
unchanged. Two independent compilations share no mutable state. The canonical
manager/stage suites retain stage identity/order, exact option identity, timing
ownership, pass records, failure records, cache identity, and later-stage
suppression.

#### Exact source/build identity and runtime waiver

Committed candidate `d437174` was captured in a clean detached worktree. A
same-root comparison against `878dd24` produced byte-identical raw C++,
preamble, signatures, source-derived kernel names, semantic/codegen/build/full
cache keys, request fields, compiler/linker flags, ABI/index policies,
build-option keys, request keys, build identities, prepared cache keys, build
directories, and `.so` paths. The complete same-root manifest is identical on
both sides with SHA-256
`ab6c28584676d963b14713dcb88ec4906aad53fc304730b28d7ce59ae671c169`.

The inherited capture root was then reproduced exactly. The entire candidate
anchor manifest is byte-identical to the retained `2e2a30d` manifest with
SHA-256 `d2dfcf5cb4299be88f2bf5b35a047bdacf2a0e8a65ab03e17d20ca89a0a17024`.
The preamble remains 68,671 bytes with SHA-256
`db29715709809539883f4904c60dd1276cad5af16a5f43549cfc11375321c544`.

| Path | Bytes | SHA-256 |
|---|---:|---|
| CSR-by-dense | 2,505 | `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151` |
| DS | 7,117 | `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f` |
| DSS | 8,660 | `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e` |
| all-COO SDDMM | 3,521 | `53d6faaee132a5d82515235b529d7d88d16cbeefe388eba5cfae9ace5528d667` |

The 42-cell base/candidate SpMM source/build corpus is byte-identical in the
new same-root capture. Reproducing the inherited capture root yields the exact
retained SHA-256
`204d80f7df45eb222e5308ab72bbaf0aaa326aa0a7e7677d17ba289b535b0dc6`.
Every cell also matches the retained M5 build summary for source digest/bytes,
function list, flags, and kernel name.

Because every generated source and build input is unchanged, the canonical
byte-identical runtime waiver applies. No new M5 or Redwood runtime grid was
run. The retained artifacts were re-hashed:

- M5 runtime artifact:
  `3b655e445d130cbfe3e394563f52498bd25675d7bb5f7c775333ef580fa7b246`;
- Redwood runtime artifact:
  `c3a6a110fc98614ca50111adab3b7ea5ee93ba7aff9da5715ee110946e683d8d`.

#### Compiler latency and attribution

The valid uncontended run used committed `d437174`, five warmups, 30 samples,
and excluded native build/execution. Artifact:

`/tmp/scorch-phase3-structured-store-results/latency-d437174-m5.json`

SHA-256:
`b65c724ea39f83fea7dbb277396724a16474041632bba4d28ebe7f59cda1d9f5`.

The predecessor was the retained uncontended `2e2a30d` artifact with SHA-256
`2dac0b53298ba2215f92d6e1a500af369869825700127aa8bcf0db7ef5d5288b`.

| Case | Candidate p50/p95 ms | New/old p50 | New/old p95 | Decision |
|---|---:|---:|---:|---|
| small dense | 1.533 / 1.727 | 0.966 | 0.893 | target |
| reduction | 1.417 / 1.557 | 0.972 | 0.980 | target |
| CSR intersection | 1.662 / 1.804 | 1.068 | 1.052 | target |
| sparse union | 1.644 / 1.929 | 1.040 | 1.035 | target |

No case crosses the 1.10 investigation threshold. Small dense improves by
0.055/0.206 ms at p50/p95 and reduction by 0.040/0.032 ms. Intersection adds
0.106/0.090 ms, attributed primarily to CIN lowering (+0.078/+0.050 ms) with
LLIR emission adding +0.007/+0.010 ms. Union adds 0.063/0.066 ms; CIN lowering
adds +0.129/+0.117 ms, partly offset by kernel-name/request assembly
(-0.021/-0.035 ms) and the canonical endpoint extension
(-0.010/-0.023 ms). This is the expected absolute cost of constructing and
validating typed lvalue children in the structurally largest sparse cases; it
does not propagate into source, cache, request, or build work. Every
compatibility-path candidate p95 remains below 1.93 ms.

#### Verification record

All Python, pytest, lint, documentation, capture, and benchmark commands used
the `scorch` conda environment. Final results on the committed code/test tree:

- focused pass/traversal/codegen/budget set: **236 passed** in 0.96 s;
- added budget/identity/type focus: **76 passed** in 0.92 s;
- canonical 18-file Phase-2/common suite: **719 passed** in 1.64 s;
- required 11-file scheduler/CIN/codegen matrix: **308 passed** in 377.84 s;
- `pytest -q -m "not perf" tests`: **1,237 passed, 14 skipped, 3
  deselected**, with one inherited PyTorch sparse-invariant warning, in
  1,946.55 s;
- Black: all 27 changed Python files clean, with only the existing Python
  3.11-versus-target-3.15 safety warning;
- Flake8: six inherited implementation findings, exactly matching the same six
  base findings; the candidate removes the base's two `iter_lattice.py`
  formatting findings;
- mypy on all 27 changed files: 103 inherited errors in 11 files versus 105 in
  12 at the base; production-only: 56 in 3 files versus 57 in 3 at the base;
  no candidate-only error was introduced;
- strict Sphinx completed the HTML build and reproduced the exact base failure:
  23 unresolved-reference warnings under `-W`, with no new warning;
- `git diff --check`: clean; both independent final reviews found no release
  blocker.

No analysis cache, preservation/invalidation protocol, dependency graph,
generalized DCE, parser, reflection, signature inspection, callback, dynamic
metadata/configuration bag, mutable registry/global singleton, forcing API,
parallel zero-fill extraction, TorchCppABI extraction, generalized allocation
migration, complete CxxIR, `csrc` change, design-document change, generated
tracked output, benchmark artifact, or unrelated tracked file was introduced.
`ScheduledCIN(cin, plan)` remains the exact frozen carrier and `matmul_wksp`
remains removed. The user-owned `.gitignore` modification and all untracked
research/benchmark material remain untouched and uncommitted.

This closes only the second narrow Phase-3 **structured indexed
assignment/store target** slice. Remaining low-level structural debt includes
the 28 characterized subscript strings, three raw compound indexed stores, and
the independently gated member/call/allocation/statement families. Phase 3
remains in progress. Phase 3.5 and LoopIR have not begun.

### Phase-3 compact-result copy-read slice complete (2026-07-15)

The third Phase-3 slice starts exactly at documentation commit `5339f56` and is
implemented by `81850b3` with tests in `6c35b03`. It migrates only the
schedule-local read from a compact heap-result tile during copy-out. The branch
is `refactor/compiler-ir-phase3-compact-result-read`. No earlier Phase-2,
structured-read, or structured-store seam was reopened.

#### Complete pre-implementation audit and seam selection

The locked starting budget contained 371 `llir.Var` constructors, 56 direct
expression strings, 315 other constructor arguments (including nine known
indirect sinks/clones), ten generic string-rewrite sites, and 52 `RawStmt`
calls / 51 producers. The 56 direct expressions were exactly 28 subscripts, 13
calls, 7 members, 3 initializers, 3 qualified names, 1 ternary, and 1
arithmetic expression. No direct indexed `Assign` target remained opaque.

All 28 subscript constructors were audited before selection:

| Classification | Count | Exact inherited sites |
|---|---:|---|
| logical tensor values | 2 | `cin_lowerer.py:736,747`, post-op `<tensor>_val[index]` reads |
| physical positions | 4 | `iterator.py:281,292,306,316`, sparse `*_pos[...]` bounds/positions |
| coordinates | 6 | `iterator.py:300,326`; `cin_lowerer.py:3896,3899,3957,4010`, sparse/all-COO `*_crd[...]` reads |
| shapes/extents | 2 | `cin_lowerer.py:784,2611`, `<tensor>_shape[level]` and `result_shape[i]` |
| workspaces | 6 | `cin_lowerer.py:900,1482,1535,1939,2030,2107`, metadata-free value fallback, `it.first[i]`, and dense/tiled workspace reads |
| mode-index containers | 4 | `cin_lowerer.py:798,812,825,838`, nested `mode_indices[level][slot]` with and without `data_ptr<int>()` |
| schedule/group compatibility | 3 | `cin_lowerer.py:3947,3951`, all-COO `_group_starts[...]`; `schedule_lowerer.py:811`, compact-result copy read |
| declaration | 1 | `cin_lowerer.py:1254`, fixed workspace `wksp[tile_size]` |

This is 27 reads plus the one declaration. The six workspace-family reads are
deliberately counted by workspace ownership, not as ordinary sparse tensor
value/coordinate-array reads. The remaining call/member audit was also
complete:

- the 13 calls were six tensor/value `data_ptr<T>()` spellings
  (`cin_lowerer.py:160,218,855,1617,1656,2827`), five `std::move(...)`
  spellings (`:428,453,476,1597,1634`), and two schedule-storage `.data()`
  spellings (`schedule_lowerer.py:869,1151`);
- the seven members were the two result-storage lvalues
  (`cin_lowerer.py:492,506`), the input `storage.value` compatibility read
  (`:868`), and four workspace-pair reads (`:1504,1554,2093,2120`);
- the rest were three initializer spellings (`cin_lowerer.py:141,200,496`),
  three `torch::kInt` names (`:432,457,1601`), the all-COO ternary at `:3873`,
  and the all-COO arithmetic spelling at `:4037`.

The physical iterator family is shared by sparse-prefetch, compressed-Where,
nested sparse, panel, and all-COO consumers. The mode-index family requires a
postfix member/call hierarchy. Workspace pair reads require member structure,
and post-op, shape, and all-COO reads have different provenance and rewrite
ownership. Migrating any of those together would conflate independently gated
families. The single remaining schedule read was instead a complete producer
family: one scope-local physical buffer, one already-typed pointer, one affine
index, and one existing precedence-aware consumer.

#### Minimum coherent representation and production trace

`_apply_heap_result_tile` already derives the result pointer type from the
generated result-value initialization, accepts only `float*` or `double*`,
creates the compact storage and its structured write targets, and owns the
three typed `INT64` index components. The former opaque RHS

`tiled_C[C_tile_copy * kTile_k + k_tile_copy]`

is now an existing frozen `ArrayAccess(Var(pointer_type), Add(Mul(INT64,
INT64), INT64))`. No new node, widened field, parser, lvalue hierarchy, logical
metadata, or fallback was needed. This physical temporary has no cross-pass
semantic identity, so adding `AccessId`/`SymbolId` provenance would be false
precision.

The path is explicit `Schedule` -> verified `LoopPlan` -> ordinary CIN/managed
LLIR production -> result/ABI assembly -> function construction -> schedule
lowering -> `_apply_heap_result_tile` -> existing `ArrayAccess` codegen. The
managed production pipeline and result/ABI assembly finish before schedule
lowering; this new node therefore does **not** traverse managed LLIR passes.
Relayout runs subsequently for packed schedules but rewrites only its selected
row-loop operand subtree; it does not traverse the compact copy-out loop. There
is no regex or string consumer for the compact RHS. Existing exact-type common
traversal validates, walks, rebuilds, and fails closed for `ArrayAccess`;
codegen owns postfix precedence and emits its typed arithmetic index
byte-for-byte.

Malformed heap schedules still fail at the owning schedule stage for a missing
tile/plan, unsupported placement, missing result write, or unsupported pointer
type. Unknown LLIR nodes/children still fail in common traversal/codegen.
Legal no-ops, repeated scheduling/rewrite application, replacement preflight,
detachment, failure propagation, managed-pass records, and later-stage
suppression are unchanged and remain covered by the inherited manager and
schedule suites. In particular, no latency cost is attributed to managed
passes for this late-created node.

Focused tests lock the exact frozen tree, pointer and index child types,
float32/float64 behavior, precedence-correct expression spelling, the complete
copy statement, and detached ownership across two independent lowerings of the
same frozen `ScheduledCIN`. Existing `ArrayAccess` tests retain exact
construction, type hints, equality, malformed-child validation, freezing,
common walk/rewrite, detachment, and unknown-subclass/unknown-child failure.
The activating packed heap-result native matrix covers both panel and full
relayout scopes, ragged panels, empty rows, zero-sized domains, and float64;
generic TTM heap copy-out is also structurally covered.

#### Locked post-slice compatibility budget

The precise budget is now:

- 374 total `Var` constructors: replacing one opaque `Var` with the
  `ArrayAccess` base plus three index `Var` children is a net increase of three;
- 55 direct expression strings: 27 subscript, 13 call, 7 member, 3 initializer,
  3 qualified, 1 ternary, and 1 arithmetic;
- 319 other constructor arguments, including the unchanged nine known indirect
  sinks/clones;
- ten generic string-rewrite compatibility sites;
- 52 `RawStmt` calls / 51 producers, unchanged.

The remaining 27 direct subscripts are exactly 21 in `cin_lowerer.py` and 6 in
`iterator.py`; `schedule_lowerer.py` has none. They are 26 reads plus the one
fixed-workspace declaration. The three characterized raw indexed stores remain
the dense-workspace `memcpy` destination and the two compressed-Where compound
prefix-sum/position-copy loops. No direct production `Assign` target is a
string-encoded expression.

#### Exact source, cache, request, ABI, flags, and build identity

Clean same-root captures compare `5339f56` with committed code/test candidate
`6c35b03`. The canonical four-anchor manifest and the 42-cell SpMM grid are
byte-identical on both sides and exactly reproduce the inherited structured-
store/`28dcca5` findings:

- anchor manifest SHA-256:
  `d2dfcf5cb4299be88f2bf5b35a047bdacf2a0e8a65ab03e17d20ca89a0a17024`;
- 42-cell grid SHA-256:
  `204d80f7df45eb222e5308ab72bbaf0aaa326aa0a7e7677d17ba289b535b0dc6`;
- preamble: 68,671 bytes,
  `db29715709809539883f4904c60dd1276cad5af16a5f43549cfc11375321c544`.

| Canonical path | Bytes | SHA-256 |
|---|---:|---|
| CSR-by-dense | 2,505 | `36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151` |
| DS | 7,117 | `d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f` |
| DSS | 8,660 | `1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e` |
| all-COO SDDMM | 3,521 | `53d6faaee132a5d82515235b529d7d88d16cbeefe388eba5cfae9ace5528d667` |

Because the canonical corpus auto-schedules and does not activate the explicit
heap seam, a separate no-build capture used both panel-scoped and full-axis
packed heap schedules. Its `28dcca5`/`5339f56`/candidate manifest is
byte-identical with
SHA-256 `4d1620ec67fc38d4403c18b2e3a6b2cd8e57aa593fa99c4db7893eb157abd130`.
Panel heap source is 5,414 bytes with SHA-256
`5f2f59ebcd245c79746129fd50554e24698ccadc1fc30ad879d8503ce6ae02bc`
and kernel name `kernel_d0e556f1c6e6`; full heap source is 5,334 bytes with
SHA-256 `4e65ab9737251480abe63687cc898adc924a1e435d9a2a629c02914662b3a476`
and kernel name `kernel_07f39c22f1ac`. Their build-identity digests are
`3c0bb0b3b440a8e27f389e34214d6383c6c4d57fe153afb6eff737408b6aa71f`
and `e61becb739622cc5b4aed36e592a3c2d36f0641a11f86187f04eb553bd514274`.
The capture script SHA-256 is
`27d5ab5c291d047084689e41a10e1cebda8df51d51fbf69dbd54defdb30205c0`.

Across both captures the generated source, exact `evaluate` signature,
preamble, source-derived kernel name, codegen/semantic/build/full cache keys,
request fields, compiler/linker flags, ABI and index policies, build-option and
request keys, build identity, prepared cache key, build directory, and `.so`
path are identical. Exact `CompileOptions` identity reaches the lowerer.

The existing canonical byte-identical runtime waiver therefore applies; no new
M5 or Redwood runtime grid is required. Retained artifacts re-hash to:

- M5:
  `3b655e445d130cbfe3e394563f52498bd25675d7bb5f7c775333ef580fa7b246`;
- Redwood:
  `c3a6a110fc98614ca50111adab3b7ea5ee93ba7aff9da5715ee110946e683d8d`.

#### Compiler latency and attribution

The valid run used committed `6c35b03` in its clean detached worktree, five
warmups, 30 samples, and no overlapping pytest, native compilation/execution,
or benchmark process. Artifact:

`/tmp/scorch-phase3-compact-read-results/latency-6c35b03-control-m5.json`

SHA-256:
`0076729e891440100dc524ee36b539fa0918dacb9cb6a5c04beb5189e2e36f1e`.
The required `d437174` predecessor re-hashes to
`b65c724ea39f83fea7dbb277396724a16474041632bba4d28ebe7f59cda1d9f5`.

| Case | Candidate p50/p95 ms | New/old p50 | New/old p95 | Decision |
|---|---:|---:|---:|---|
| small dense | 1.635 / 1.816 | 1.066 | 1.051 | target |
| reduction | 1.460 / 1.642 | 1.030 | 1.054 | target |
| CSR intersection | 1.653 / 1.784 | 0.994 | 0.989 | target |
| sparse union | 1.670 / 1.826 | 1.016 | 0.947 | target |

No case crosses 1.10. Absolute p50/p95 changes are +0.102/+0.089 ms,
+0.043/+0.085 ms, -0.009/-0.020 ms, and +0.026/-0.103 ms respectively.
The largest recorded stage change is small-dense CIN lowering at
+0.031/+0.097 ms; reduction CIN lowering is +0.007/+0.025 ms and kernel/request
assembly is +0.007/+0.017 ms. Intersection CIN lowering is -0.012/+0.043 ms,
while union CIN lowering is +0.003/-0.034 ms. The canonical endpoint extension
changes only +0.005/-0.014, +0.003/+0.014, -0.003/-0.021, and
-0.003/+0.006 ms. The latency corpus does not request heap schedules and cannot
execute the new construction, so these small bidirectional stage changes are
measurement variation, not managed-pass or compact-read cost.

One earlier sample taken immediately after the 35-minute native full suite is
retained but rejected as contended:
`/tmp/scorch-phase3-compact-read-results/latency-6c35b03-m5.json`, SHA-256
`c118b0311e57857853304614d4c6c8a384c87a24619f2ebc0980d67c62efe701`.
It crossed 1.10 broadly in unrelated non-activating stages; an immediate host
check showed high CPU in `syspolicyd`, `peopled`, `WindowServer`, `trustd`, and
CallHistory synchronization. A fresh same-host `5339f56` control (SHA-256
`ee57b6ad9278e433389fd671891845d2c8f14952a20f55cdc16ae8255724c701`)
followed by the valid candidate produced p50/p95 ratios of 1.014/0.981,
1.000/1.038, 0.991/1.012, and 1.029/1.056. This paired control confirms the
rejected crossing was host noise and that every candidate case is within the
policy target.

#### Verification record

Every Python, pytest, documentation, capture, and benchmark command activated
the `scorch` conda environment first. On committed `6c35b03` in a clean
detached worktree:

- exact compact structure/budget focus: 6 passed in 1.17 s;
- focused schedule/traversal/codegen/budget/native set: 264 passed in 332.43 s;
- canonical 18-file Phase-2/common suite: 720 passed in 2.76 s (the inherited
  719 plus one new schedule regression);
- required 11-file scheduler/CIN/codegen matrix: 309 passed in 375.22 s (the
  inherited 308 plus that regression);
- `pytest -q -m "not perf" tests`: 1,238 passed, 14 skipped, 3 deselected, and
  the one inherited PyTorch sparse-invariant warning in 2,100.49 s;
- Black on all three changed Python files: clean, with only the existing Python
  3.11-versus-target-3.15 safety warning;
- Flake8 on all three changed Python files: clean at both base and candidate;
- mypy on the production file: clean at both base and candidate; all three
  changed files report the exact same nine inherited missing-`py.typed` test
  import diagnostics at base and candidate, with no candidate-only error;
- strict Sphinx completes HTML generation and reproduces the exact normalized
  base failure: 23 unresolved-reference warnings under `-W`, no new warning;
- `git diff --check`: clean; an independent implementation/requirements audit
  found no release blocker.

The canonical suites retain caller-owned CIN/LLIR, access metadata, analyses,
plans, schedules, exact `CompileOptions`, prior results, and independent-
compilation ownership. They also retain exact stage identity/order, timing
ownership, managed-pass/failure records, cache identity, and failure
short-circuit/later-stage suppression. `ScheduledCIN(cin, plan)` remains the
exact frozen carrier.

No parser, generalized member/call/lvalue/allocation hierarchy, broad ABI
rewrite, generalized DCE, reflection, signature inspection, dynamic metadata,
dictionary-of-`Any` configuration, callback, mutable registry/global singleton,
analysis cache/invalidation protocol, dependency graph, forcing API, parallel
zero-fill extraction, TorchCppABI extraction, generalized allocation migration,
complete CxxIR, generated tracked output, benchmark artifact, `csrc` change,
design-document change, or unrelated tracked file was introduced. The
user-owned `.gitignore` modification and untracked `autotune-levels/`, `bench/`,
`bench/bench_results/`, and `scratchpad/` material remain untouched and
uncommitted.

This closes only the narrow Phase-3 **structured schedule-local compact-result
copy read** slice. Remaining low-level structural debt is the 27 characterized
subscript strings, three raw compound indexed stores, and the independently
gated member/call/initializer/qualified/ternary/arithmetic/allocation/statement
families. Phase 3 remains in progress. Phase 3.5 and LoopIR have not begun.

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
