# Scorch Compiler IR Refactor: Architecture and Migration Design

## Document Status

- Status: proposed architecture and migration plan
- Scope: the complete compiler pipeline, from validated operation semantics to a
  compiled Torch/C++ kernel artifact
- Repository: `/Users/bobby/scorch`
- Latency-policy revision: 2026-07-13; correctness and generality come first,
  with 1.10x kept as the default measurement target and the point at which a
  slowdown must be explained, rather than as an automatic rejection boundary
- Companion implementation handoff:
  [`COMPILER_IR_REFACTOR_HANDOFF.md`](COMPILER_IR_REFACTOR_HANDOFF.md)

This is the design the project is building toward. It describes the final
architecture, what each stage owns and guarantees, the whole migration sequence,
and the conditions under which the legacy compiler can be deleted. It covers
considerably more than the first implementation task.

## Binding Review Decisions

These are settled. They are part of the design, not details left to
implementation:

- Analyses are pure functions over immutable IR and return ID-keyed side tables.
  They are recomputed after transformations. The initial architecture has no
  fingerprint-keyed analysis cache, and no protocol by which a pass declares
  which analyses it preserves or invalidates. Add caching only if stage-latency
  profiles show a material bottleneck.
- LoopIR must be shown to handle sparse merging before its schema is frozen. A
  mandatory experiment hand-writes candidate LoopIR for CSR SpMV intersection
  and for sparse elementwise addition, and runs both through a small reference
  interpreter.
- What justifies the investment is code-generation parity with the handwritten
  prebuilt schedules: a new schedule should be expressible as `LoopPlan` plus
  typed passes and should perform competitively without a new
  algorithm-specific C++ kernel.
- Cutover and deletion of the legacy compiler compare the new pipeline against
  the kernels the legacy compiler generates, and against nothing else. Parity
  with the handwritten prebuilt kernels is checked at the selector level in a
  dedicated milestone after cutover (Phase 8.5); falling short there triggers a
  scope review, never retention of the legacy compiler.
- Correctness, refusing what is unsupported, and general compiler structure all
  take precedence over minimizing Python compiler latency. Compile-latency
  budgets are per-phase regression monitors: the Phase 0 ratio is the default
  target and the threshold for investigation, but a modest, measured increase
  attributed to a named stage may be accepted when it buys general typed
  infrastructure. Do not weaken validation, and do not add shortcuts specific to
  a format, an expression or a kernel, merely to hit the target. A material or
  unexplained slowdown still blocks a phase. Full internal IR verification is on
  by default in tests and debug builds and off by default in the production JIT
  path; cheap validation of public input stays on everywhere.
- Whether a generated kernel passes is decided against a noise floor measured on
  each machine by timing the baseline binary against itself — an A/A control —
  and never against a fixed ratio constant; a rule that cannot pass when applied
  to its own A/A control is measuring noise rather than regressions. If a change
  emits byte-identical C++ for every program in the measurement set under
  identical build flags, no runtime benchmark is required; the tests that prove
  an optimization actually fired are required in every case.
- IR nodes use frozen dataclasses with tuple children and ID-keyed side tables.
  Do not introduce an arena unless later evidence establishes a concrete need.
- `ScheduledCIN` is only a frozen transitional carrier for normalized CIN and a
  verified `LoopPlan`. It is not a fourth IR and has no node model of its own.
- Compiling concurrently is a desirable hardening property, not something a
  migration phase has to demonstrate before it can proceed.
- Phase 3 plus the sparse feasibility experiment ends in an explicit go/no-go
  review. Stopping there must still leave the current compiler safer and more
  maintainable.

## Executive Decision

Rebuild the compiler incrementally around immutable artifacts at pass
boundaries, stable identities, structured IRs, explicit analysis and scheduling
artifacts, per-stage verifiers, and diagnostics that refuse what they cannot
handle.

The final conceptual pipeline is:

```text
Python operation
  -> validated OpSpec / EinsumSpec
  -> normalized semantic CIN
  -> semantic + iteration analyses
  -> LoopPlan
  -> LoopIR
  -> ScheduledLoopIR
  -> typed target-independent optimizations
  -> parallel and target lowering
  -> CxxIR
  -> TorchCppABI wrapper
  -> emitted C++ + KernelBuildSpec
  -> compiled artifact
```

This will be delivered as a strangler migration: the new pipeline is built
beside the old one and takes over one family of programs at a time. The legacy
pipeline stays available until the replacement paths have differential coverage
of structure, generated code, and numerics. There will be no flag-day rewrite.

## Motivation

Scorch's compiler contains valuable sparse-iteration and scheduling logic, but
which stage owns what is implicit:

- CIN owns mutable backreferences and name-based identity.
- Scheduling mutates CIN and attaches metadata that is nowhere declared.
- Sparse scheduling is split between CIN and post-LLIR transformations.
- Iteration analysis and lowering call into each other recursively.
- LLIR stores C++ expressions inside symbol names and raw statements.
- Passes parse generated C++ spelling to recover semantics.
- ABI construction, result assembly, optimization, parallel policy, and
  semantic lowering all live together in `CINLowerer`.
- Unsupported nodes sometimes pass through silently and surface later as C++
  compiler errors.

These properties make local changes non-local in effect. A new schedule, layout,
or code-generation optimization can depend on undocumented mutation order,
rendered variable names, and private state established by a previous phase.

The goal is not abstraction for its own sake. It is to make every compiler
decision inspectable, deterministic, testable in isolation, and owned by exactly
one stage.

### Why the JIT path merits the investment

Most production performance today comes from prebuilt kernels such as the SpMM
implementations and their tiling selector. That usefully limits how much a
compiler change can break during the migration, but it also shows what the
current compiler costs us: schedules such as tile-j and tile-ijk became
handwritten C++ kernels because the generated path could not express them
competitively.

The testable payoff for this refactor is therefore not just deleting legacy
code. After migration, a schedule should be implementable as a `LoopPlan` plus
reusable typed transformations and reach parity with its handwritten
equivalent. One representative tile-j schedule and one tile-ijk schedule are the
initial proof targets. Runtime support and target intrinsics may remain native;
the schedule algorithm itself must not require a new bespoke C++ kernel.

This is what justifies the investment; it is not a condition of cutover.
Deleting the legacy compiler requires only that the new pipeline match the
kernels the legacy compiler generates today, which is one compiler compared
against another. Parity with the handwritten prebuilt kernels is a strictly
higher bar, and keeping the legacy compiler alive contributes nothing toward
closing any remaining gap. Prebuilt parity is also a claim about the whole
system: the runtime tiling selector armed with generated kernels must match the
selector armed with prebuilt kernels, because a prebuilt schedule only runs on
the shapes where the selector's probe picks it. Beating a hand-tuned kernel on
one shape the selector would never route to it is not the target.

## Goals

1. A semantic program can be compiled repeatedly under different schedules
   without copying or mutating the original program.
2. Every transformation has a typed input, typed output, explicit context,
   declared analysis dependencies, and a verifier.
3. Sparse iteration semantics are represented structurally and never recovered
   from generated C++ strings.
4. Symbols and loops use stable identity independent of human-readable names.
5. Unsupported valid programs, invalid programs, invalid schedules, and internal
   compiler failures produce distinct diagnostics.
6. Code emission is mechanical, exhaustive, and precedence correct.
7. Target-specific Torch, C++, OpenMP, and ABI details remain outside semantic
   and loop-level IRs.
8. Cache identity is derived from canonical artifacts and explicit compile
   options, not from incidental object representation.
9. Each migration phase is shippable and preserves supported behavior.
10. A new schedule can be expressed as `LoopPlan` plus typed passes and achieve
    agreed parity with an equivalent handwritten prebuilt implementation.
11. The old compiler can be deleted when objective exit criteria are satisfied.

## Non-Goals

- Replacing PyTorch as the runtime or tensor allocator in this project.
- Adding new sparse formats or kernel optimizations during the IR migration.
  Re-expressing the existing tile-j/tile-ijk schedules to prove codegen parity
  validates the migration; it is not a new optimization experiment.
- Changing public `STensor` mutation or result-type contracts as part of this
  compiler-specific project.
- Making the semantic IR target every possible backend immediately.
- Eliminating every `RawStmt` in the first phases.
- Solving all repository-wide typing and formatting debt in the same changes.
- Keeping exact generated C++ text stable when a change intentionally fixes a
  correctness bug; semantic and performance compatibility are the contract.

## Design Principles

### Immutable at pass boundaries

Builders may mutate locally while constructing an artifact. Once finalized, an
artifact passed to another stage is immutable. A pass returns a new artifact and
cannot modify its input.

This is more practical than requiring every temporary object to be frozen, and
it still prevents hidden aliasing across compilations and passes.
Transformations rebuild only changed paths, normally with `dataclasses.replace`;
unchanged frozen nodes may be shared safely.

### Identity is not spelling

`i`, `j`, `A`, and generated names are presentation. Every symbol, loop, tensor,
workspace, access, and memory region has a stable typed ID. Same-named variables
in distinct scopes are distinct.

Transformations that create new entities assign new IDs and retain provenance
such as `derived_from`, `split_from`, or `lowered_from`.

### IR contains semantics, not rendered code

Array access is an access node, not a string in `Var.name`. A member call is a
member/call node. A parallel loop is a structured parallel construct. Rendering
is the final operation on CxxIR.

### Analyses live beside IR

Parent maps, use-def chains, free/reduction classification, tensor roles, layout
facts, dominance, and cost estimates are analysis results keyed by stable IDs.
They are not mutable backreferences stored on nodes.

### Fail closed

An unsupported or unknown node fails in the stage that owns it. It never becomes
an empty statement list, an ignored post-op, placeholder C++ text, or a silently
degraded schedule.

### Explicit target boundary

CIN and LoopIR do not know about Torch C++ field names, `torch::from_blob`,
OpenMP pragma spelling, or native return structs. Those belong to CxxIR and the
`TorchCppABI` layer.

## Overall Architecture

The pipeline is not purely linear. Analyses and planning are side artifacts used
to construct and transform IR:

```text
                    +--------------------+
                    |   CompileOptions   |
                    +----------+---------+
                               |
                               v
+-----------+   validate   +----------------+   normalize   +---------------+
| Op/Einsum | -----------> | Semantic CIN   | ------------> | NormalizedCIN |
+-----------+              +----------------+               +-------+-------+
                                                                     |
                          +------------------------------------------+
                          |                                          |
                          v                                          v
                 +----------------+                         +----------------+
                 | AnalysisBundle | ----------------------> |    LoopPlan    |
                 +-------+--------+                         +--------+-------+
                         |                                           |
                         +------------------+------------------------+
                                            v
                                      +-----------+
                                      |  LoopIR   |
                                      +-----+-----+
                                            |
                                            v
                                   +-----------------+
                                   | ScheduledLoopIR |
                                   +--------+--------+
                                            |
                                            v
                                     typed IR passes
                                            |
                                            v
                                        +-------+
                                        | CxxIR |
                                        +---+---+
                                            |
                                +-----------+------------+
                                v                        v
                         +-------------+          +---------------+
                         | TorchCppABI |          | C++ emitter   |
                         +------+------+          +-------+-------+
                                |                         |
                                +-----------+-------------+
                                            v
                                    +---------------+
                                    |KernelBuildSpec|
                                    +---------------+
```

## Core Compiler Artifacts

### CompileOptions

`CompileOptions` is an immutable snapshot of all configuration that can affect
compiler behavior or output, including:

- requested schedule or auto-scheduling policy;
- target and parallel backend;
- optimization level and enabled passes;
- numerical/fast-math policy;
- debug verification and stage-dump settings;
- index-width and ABI policy;
- target feature/ISA policy;
- compiler feature flags currently read from environment variables.

Environment variables are parsed once at the API/runtime boundary. Passes do not
read global process state.

### CompilationContext

The context supplies read-only services rather than hidden mutable compiler
state:

- ID allocator and provenance recorder;
- target description;
- diagnostics collector;
- pure analysis runner;
- stage-dump sink;
- pass instrumentation and timing;
- on-demand canonical serializer/fingerprint service.

The context must not become a bag of phase-dependent fields. State that affects
semantics belongs in an explicit artifact.

### Artifact

Every major stage returns an artifact conceptually containing:

```text
Artifact[T]
  ir: T
  stage: StageId
  provenance: ProvenanceMap
  diagnostics: immutable diagnostics
```

The exact Python representation may be simpler initially. What matters is
explicit ownership and deterministic serialization. Fingerprints are computed on
demand, for persistent build/cache identity or to compare two runs; they are not
mandatory fields carried through every pass.

## Stage 0: Operation Specification

### Responsibility

Parse and validate the public operation before constructing compiler IR.

For einsum, `EinsumSpec` owns:

- operand labels;
- result labels;
- operand count and ranks;
- dimension unification;
- reduction/free classification;
- requested result layout and mode order;
- dtype promotion/compatibility policy;
- supported feature checks.

### Invariants

- Every label and operand rank is consistent.
- Result labels are legal and have known sizes.
- Repeated-label semantics are explicit.
- Unsupported broadcasting or output structures are rejected here.
- No tensor or CIN mutation has occurred.

### Output

An immutable `OpSpec`/`EinsumSpec` suitable for deterministic CIN construction.

## Stage 1: Semantic CIN v2

### Responsibility

Represent what is computed, independent of execution order and target code.

Suggested semantic node families:

- `TensorDecl`
- `IndexDecl`
- `TensorRead`
- `TensorWrite`
- `Assign` / `Reduce`
- `Product`, `Add`, and other scalar expressions
- `ForAll` as a semantic index binding where still needed
- `Where` or an explicit producer/consumer semantic region
- `WorkspaceDecl` only when the workspace is semantically required rather than a
  schedule implementation choice

### Ownership

- Nodes are immutable after construction.
- Nodes are frozen dataclasses and children are immutable tuples.
- No node owns mutable parent/use lists.
- IDs determine identity; display names do not.
- Semantic layout facts reference immutable layout values from the core tensor
  model.

### Normalization

Normalization produces a canonical CIN form before scheduling:

- canonical expression and access ordering where semantics permit;
- explicit reductions;
- canonical result assignment;
- normalized mode-order interpretation;
- desugared frontend conveniences;
- no implicit output and no hidden `_assignment` side effects;
- deterministic node ordering and serialization.

Normalization is a pure pass and must be idempotent.

## Stage 2: Analyses

Analyses consume normalized CIN and return immutable side tables.

### Required analyses

#### Symbol and scope analysis

- definition and use sites;
- scope ownership;
- collision-free display-name generation;
- free/dangling reference detection.

#### Tensor access analysis

- input, result, and workspace roles;
- access index order;
- logical-to-physical mode mapping;
- layout level associated with each index;
- read/write/reduction behavior.

#### Reduction analysis

- free and reduction IDs;
- reduction operators and identities;
- legal parallelization boundaries;
- result initialization requirements.

#### Iteration-domain analysis

- dense ranges;
- compressed position and coordinate domains;
- coordinate/COO domains;
- parent-child sparse level relationships;
- union/intersection merge requirements;
- iterator dependencies and dominance.

#### Shape and type analysis

- index extents;
- scalar and index types;
- result shape;
- legal casts and promotions;
- index overflow requirements.

#### Cost analysis

- estimated extent, nnz, work, memory traffic, and workspace cost;
- target-independent facts separated from calibrated policy constants.

### Analysis execution policy

Each analysis is a pure function from an immutable artifact and explicit options
to an immutable, ID-keyed side table. Analyses never mutate the IR they inspect.
The initial implementation recomputes requested analyses after a transformation;
for the small IRs that current Scorch operations produce, this favors
correctness and simplicity over speculative reuse.

There is deliberately no analysis-result cache, and no pass-level declaration of
what an analysis preserves or invalidates. Caching may be proposed later, only
with stage profiles showing that recomputation is material, an explicit cache
key, and tests proving that a stale result cannot be observed.

## Stage 3: LoopPlan

### Responsibility

Represent scheduling decisions independently of both semantic CIN and concrete
C++ spelling.

Suggested contents:

- logical loop order by stable index/loop ID;
- affine split/tile width;
- placement constraints;
- sparse coordinate panel decisions;
- workspace insertion and accumulation strategy;
- operand staging/relayout plan;
- result-tile strategy;
- abstract parallel-loop selection;
- unroll/vectorization preferences;
- schedule provenance: explicit, auto, tuned, or fallback;
- a canonical cache key containing every decision.

### Validation

`verify_loop_plan` checks the plan against normalized CIN and analyses before it
is applied:

- referenced IDs exist;
- loop order is complete and legal;
- sparse parents precede dependent children;
- reduction tiling has a valid accumulator lifetime;
- parallel loops do not introduce result races;
- panel and relayout decisions reference compatible accesses;
- staging and result buffers have legal scope and ownership;
- unsupported plans fail as `InvalidSchedule` or `UnsupportedFeature`.

### Relationship to public Schedule

The current public `Schedule`, `TileSpec`, and `RelayoutSpec` may remain as API
input types. A dedicated adapter translates them into a verified `LoopPlan`.
Internal passes consume `LoopPlan`, never the public object directly.

### Transitional ScheduledCIN carrier

During migration, legacy lowering receives an explicit frozen carrier equivalent
to:

```python
@dataclass(frozen=True)
class ScheduledCIN:
    cin: NormalizedCIN
    plan: LoopPlan
```

`ScheduledCIN` creates no new node kinds, owns no analysis cache, and permits no
schedule metadata on CIN nodes. It exists only to move normalized CIN and its
verified plan together across the places where the new code still hands off to
legacy code. Once CIN plus `LoopPlan` lower directly to LoopIR, this carrier
should disappear rather than grow into another IR.

## Stage 4: LoopIR

LoopIR is the central future design. It represents executable iteration
semantics without committing to C++/OpenMP/Torch ABI spelling.

### Responsibilities

- Make every iteration domain explicit.
- Preserve logical tensor/access provenance.
- Represent sparse merge behavior structurally.
- Provide stable anchors for scheduling transformations.
- Model workspace and memory-region lifetime before target lowering.
- Keep abstract parallel/vector semantics target independent.

### Suggested node families

#### Module and regions

- `LoopModule`
- `FunctionRegion`
- `Block`
- `Yield` / region terminator

#### Values and symbols

- `ValueId`
- `LoopId`
- `TensorId`
- `MemoryRegionId`
- typed scalar/index values

#### Iteration

- `DenseFor(loop_id, lower, upper, step, body)`
- `SparsePosFor(loop_id, level_ref, pos_begin, pos_end, body)`
- `ResolveCoord(level_ref, position)`
- `MergedSparseFor(domains, merge_kind, body)`
- `Filter` / `Guard`
- `TileFor` or affine split representation
- `ParallelFor` as an abstract property or node

#### Tensor and memory operations

- `TensorLoad(access_ref, indices)`
- `TensorStore(access_ref, indices, value)`
- `ReduceStore(access_ref, indices, op, value)`
- `WorkspaceAlloc`
- `WorkspaceReset`
- `WorkspaceLoad` / `WorkspaceStore`
- `RelayoutAlloc`
- `Pack` / `Copy`
- `Dealloc` where explicit lifetime is required

#### Scalar expressions

- typed literals;
- arithmetic and comparisons;
- min/max/select;
- typed calls to approved intrinsics.

### Iteration lowering

Normalized CIN plus iteration-domain analysis lowers into an unscheduled LoopIR.
This stage replaces the current recursive `CINLowerer`/`IterationLattice`
coupling.

The merge lattice becomes an analysis or builder that produces explicit
`MergedSparseFor`, guards, and coordinate resolution. It does not call the
general lowerer or toggle lowerer state.

### LoopIR invariants

- Every loop has a stable ID and explicit domain.
- Every coordinate resolution references a dominating position iterator.
- Sparse child iteration is dominated by its parent level.
- Loads/stores reference typed logical accesses, not rendered positions.
- Reductions have explicit operator and identity.
- Workspace lifetime dominates all uses.
- No C++ spelling or Torch storage field appears.
- Control regions are well formed and terminated.

## Stage 5: ScheduledLoopIR

`ScheduledLoopIR` is LoopIR after applying a verified `LoopPlan` through pure
passes. It is a verified stage of the same LoopIR node model, not a fourth
principal IR with a second schema.

### Scheduling pass families

#### Loop reorder

Rebuild loop regions according to legal dependencies. Preserve stable IDs for
unchanged loops and attach provenance to moved and split loops.

#### Affine tiling

Split one loop into outer and inner loops with explicit bounds and ragged-tail
semantics. New loop IDs reference the original logical loop through provenance.

#### Sparse panel tiling

Transform an explicit sparse coordinate domain into a coordinate window. The
pass uses structured level/coordinate facts; it never finds arrays through
regexes or generated names.

#### Workspace materialization

Insert structured allocation, reset, producer, and consumer regions with
verified lifetime and reduction semantics.

#### Operand relayout/staging

Allocate a typed memory region, insert structured pack loops at the requested
scope, and redirect a specific logical access ID to the staged region.

#### Result tiling

Redirect a result access into compact storage, initialize at the dominating
scope, and copy out at verified lifetime end.

#### Abstract parallelization

Mark a legal loop as parallel and attach target-independent work estimates,
reduction/race information, and scheduling intent. OpenMP spelling is deferred.

### ScheduledLoopIR verifier

In addition to the ordinary LoopIR checks:

- all LoopPlan decisions were consumed exactly once or explicitly declined;
- generated split loops have correct bounds and provenance;
- no direct access remains after a mandatory staging/result redirect;
- parallel regions are race free;
- workspace/staging/result-tile lifetimes are legal;
- schedule transformations preserve tensor access semantics.

## Stage 6: Typed Optimization Passes

Existing post-lowering optimizations migrate into explicit passes over
structured LoopIR or CxxIR, depending on the semantics they require.

### Target-independent candidates

- single-iteration loop elimination;
- loop-invariant expression hoisting;
- common subexpression or address calculation hoisting;
- workspace reset simplification;
- dead allocation/copy elimination;
- canonical loop simplification;
- access provenance preservation.

### Target-aware candidates

- sparse prefetch insertion;
- pointer derivation and restrict qualification;
- SIMD marking;
- target-specific alignment decisions;
- OpenMP/ATen work-distribution selection.

### Pass requirements

Each pass gets its own module and tests. It must define:

- accepted stage;
- required analyses;
- verifier expectations;
- whether it must be idempotent;
- legal no-op behavior;
- exact unsupported conditions.

Required analyses are invoked as pure computations for the current input. In the
initial architecture, passes do not declare what they preserve or invalidate.

A common walker/rewriter replaces separate recursive traversal implementations.

## Stage 7: Parallel and Target Lowering

Abstract parallel intent is lowered using `CompileOptions` and target analysis.

### Inputs

- scheduled and optimized LoopIR;
- race/reduction analysis;
- work estimates;
- target runtime choice: OpenMP, ATen thread pool, or serial;
- target capability profile.

### Outputs

Structured target operations, eventually represented in CxxIR:

- parallel region;
- parallel-for work sharing;
- thread-count expression;
- static/dynamic chunk policy;
- per-thread allocation lifetime;
- atomics or reductions where explicitly supported;
- barriers and synchronization.

No pass calls `omp_set_num_threads` or mutates process-global threading state.

Parallel policy is versioned and participates in the build fingerprint.

## Stage 8: CxxIR

CxxIR is a structured AST for emitted Torch/C++ code.

### Required expression nodes

- symbol reference;
- literal;
- unary/binary expression with explicit precedence;
- cast;
- subscript;
- member access;
- address-of/dereference;
- function and member call;
- initializer list;
- conditional/select.

### Required statement nodes

- declaration and initialization;
- assignment/reduction assignment;
- block;
- `for`, `while`, and conditional;
- break/continue/return;
- allocation/deallocation abstractions where still needed;
- parallel region/loop/pragmas;
- comment and blank line as non-semantic formatting nodes;
- narrowly scoped raw statement escape hatch.

### Type model

Replace a flat enum of rendered spellings with algebraic types where useful:

- scalar type;
- index type;
- pointer/reference/const/restrict qualifiers;
- vector/array/container type;
- Torch tensor and ABI struct types;
- user/runtime support types.

Rendering a type to C++ is a target operation. Unsupported type construction
fails before emission.

### CxxIR verifier

- every reference is declared and in scope;
- expression and assignment types are compatible;
- control-flow nodes are well formed;
- declarations dominate uses;
- return type matches;
- parallel constructs are nested legally;
- raw fragments are restricted to approved categories;
- codegen supports every node present.

## Stage 9: TorchCppABI and Code Generation

### TorchCppABI

This component owns all Torch/native boundary details:

- kernel function signature;
- shape, index, and value argument schema;
- dtype/index-width validation assumptions;
- tensor data-pointer extraction;
- result allocation and ownership;
- sparse index/result assembly;
- return-value construction;
- post-op extra arguments;
- runtime support declarations.

It replaces the ABI construction and result assembly currently embedded in
`CINLowerer` and its `ResultTensorAssembler`.

### C++ emitter

The emitter is a total visitor over verified CxxIR. It performs formatting only:

- precedence-correct expressions;
- deterministic indentation and naming;
- target syntax rendering;
- no semantic analysis;
- no silent fallback for unknown nodes.

### KernelBuildSpec

Emission produces a complete immutable build specification:

- canonical source and included runtime sources;
- Scorch compiler/IR/ABI versions;
- compile and link flags;
- compiler identity;
- Python and Torch ABI/configuration;
- platform, architecture, and ISA policy;
- numerical/fast-math mode;
- parallel runtime policy;
- source artifact fingerprints.

This specification is the input to the centralized runtime cache/compiler,
rather than an ad-hoc hash of selected strings.

## IDs, Names, Provenance, and Serialization

### Stable IDs

Use typed IDs rather than raw integers everywhere possible:

- `NodeId`
- `SymbolId`
- `IndexId`
- `TensorId`
- `AccessId`
- `LoopId`
- `MemoryRegionId`

IDs are unique within a compilation artifact. They do not need to remain stable
across unrelated process executions.

### Provenance

Transformations record relationships:

- normalized from frontend node;
- lowered from CIN node;
- loop split from logical loop;
- access redirected from tensor access;
- target operation lowered from abstract operation.

Diagnostics and stage dumps use provenance to explain how generated code relates
to the user's operation.

### Canonical serialization

Persistent build/cache identity, and comparing two runs against each other, must
not depend on memory addresses, dictionary ordering, or allocation-order IDs.
Canonical serialization is computed when those uses require it, and:

- orders nodes deterministically;
- renumbers IDs by deterministic traversal;
- serializes all semantic and scheduling fields;
- omits non-semantic display/debug fields from semantic fingerprints;
- includes IR schema version.

`str()` and `repr()` are for humans, not for cache identity.

## Pass Manager

The pass manager orchestrates an explicit pipeline assembled from
`CompileOptions`:

```text
verify input (tests/debug)
-> run pass
-> recompute any analysis requested by the next pass
-> verify output (tests/debug)
-> optionally dump artifact
-> record timing/diagnostics
```

Full internal verifiers run before and after every pass in tests and debug mode.
They are off by default in production/release JIT compilation, so the safety
discipline does not become hidden per-compile latency. Cheap validation at
public and trust boundaries stays on in all modes.

### Pass API requirements

A pass should expose:

- stable pass name and version;
- input/output artifact types;
- required analyses;
- deterministic/pure contract;
- configuration schema;
- diagnostic categories.

Pass order is explicit and testable. No required ordering exists only as a list
of method calls hidden inside a lowerer.

## Diagnostics

Define a compiler exception/diagnostic hierarchy:

- `InvalidProgram`: malformed user/frontend program;
- `UnsupportedFeature`: valid program outside current capabilities;
- `InvalidSchedule`: schedule inconsistent with program or target;
- `VerificationError`: malformed IR at a stage boundary;
- `CompilerInvariantError`: internal compiler bug;
- `CodegenError`: verified target IR could not be emitted;
- `BuildError`: external compiler/build failure;
- `ExecutionError`: native invocation failure.

Every compiler diagnostic should include:

- stage and pass;
- operation/expression summary;
- node/loop/access ID and provenance when applicable;
- compact IR fragment;
- actionable reason;
- full stage dump location when debug dumping is enabled.

## Debugging and Observability

Provide a supported mechanism rather than commented-out `print` statements:

- dump normalized CIN, LoopPlan, LoopIR, ScheduledLoopIR, and CxxIR;
- emit canonical textual formats suitable for diffs;
- log pass and analysis timings;
- explain schedule decisions and rejected alternatives;
- expose artifact/cache fingerprints;
- optionally retain generated C++ and build manifest;
- identify the first verifier, or the first compared stage, that diverged.

Debug output must not affect cache identity or generated code.

## Dependency Boundaries

The desired dependency direction is:

```text
core tensor metadata/layout
        ^
frontend OpSpec/CIN
        ^
compiler analyses + LoopPlan + LoopIR
        ^
target CxxIR + TorchCppABI
        ^
runtime build/cache/execution
```

Rules:

- Core format/layout code does not import compiler IR.
- CIN does not import CxxIR or runtime code.
- Analyses do not import code generation.
- LoopPlan validation may use semantic/iteration analyses but not rendered code.
- CxxIR does not inspect `STensor` objects directly.
- Runtime does not mutate compiler artifacts.

## Performance and Latency: What Is Measured, and When

Performance is checked where behavior changes, not postponed until cutover.

### The generated-kernel check

The repository's codegen-parity grid runs on both reference machines: Apple M5
and x86/redwood. The grid is a fixed set of program and shape combinations, one
"cell" per combination, and every cell must remain numerically correct.

Noise is measured before any change is judged. Every run of this check is
accompanied by an A/A control: the baseline binary measured against itself, on
the same grid and machine, with the same interleaved repeated-measurement
method. The control establishes, per cell and per machine, the spread of ratios
that "no change" produces. This is the noise-floor method already used for the
tiling selector's no-regression proof (v2 against v2 on both machines).

Pass and fail are defined against that control, not against fixed constants:

- a cell fails when its new-over-old median runtime ratio exceeds the band its
  A/A control establishes for that cell;
- a machine fails when its ratio geomean falls outside the A/A geomean band;
- thresholds must account for how many cells are compared at once, and must
  satisfy one calibration requirement: applied to the A/A control itself, the
  check passes. A rule that a same-binary comparison cannot pass is measuring
  noise, not regressions.

Any excess over the control band is a failure requiring investigation and
attribution, however small; a quiet cell's 2% regression is real even though a
noisy cell's 2% wobble is not. There is no fixed "inconclusive" band and no
rerunning until it passes: repeating a measurement until a noisy cell dips below
a threshold selects on noise instead of establishing safety. A rerun is
legitimate only to replace a grid run with a diagnosed measurement defect (for
example thermal throttling), is declared as such, and replaces the entire run,
never individual cells.

Every extraction that changes emitted code for any grid cell runs the full
two-machine codegen-parity comparison in the same PR. Extraction of sparse
prefetch insertion, dense-pointer hoisting, single-iteration elimination,
invariant-factor hoisting, and parallel zero-fill is not complete until the
tests proving the optimization fires and — where emission differs — the
two-machine before/after benchmarks land together.

Byte-identical emission removes the need for the runtime benchmark. If a change
produces byte-for-byte identical C++ for every cell of the grid under identical
build flags, the compiled kernels are identical, and timing a binary against
itself can only produce noise that demands investigation of nothing. The PR must
instead include the byte-identical evidence across the whole set, plus the tests
proving the optimization fires, which are never optional. The runtime benchmark
is mandatory whenever any cell's emitted code differs.

### JIT compile-latency monitor and acceptance policy

Phase 0 records p50 and p95 Python compiler latency from validated operation to
`KernelBuildSpec`, excluding external C++ compilation and native execution. The
curated set of programs includes small dense, reduction, CSR intersection, and
sparse union cases. Phase 2 separately measures empty pass/artifact plumbing,
whose p95 incremental overhead must be at most 1 ms. In production/release mode,
every category should target end-to-end Python compiler p50 and p95 at or below
1.10 times its Phase 0 baseline. Crossing 1.10 calls for investigation and is
not an automatic rejection: a modest increase may be accepted when measurements
attribute it to general-purpose correctness, ownership, validation, or typed-pass
infrastructure and its absolute cost is reasonable for the operation being
compiled. Every accepted exception must record absolute and relative
measurements, which stage the cost is in, and the design tradeoff. A material,
unexplained, or compounding slowdown remains a failure. Compiler latency must
not be reduced by weakening correctness checks that belong on the production
trust boundary, or by tailoring dispatch to the benchmark's formats,
expressions, or kernels.

Debug verifier and stage-dump costs are measured separately and do not count
against production/release latency measurements. Full verification is
test/debug-only by default.
Compiling twice to compare the two pipelines against each other is also excluded
from release latency, because it is confined to a curated comparison matrix
rather than the ordinary JIT or correctness suite.

## Full Migration Plan

### Phase 0: Stabilize and fail closed

#### Deliverables

- Characterization tests for supported CIN/schedule/codegen paths.
- Unknown CIN, LLIR, and post-op nodes rejected at their owning stage.
- Precedence-correct C++ expression emission.
- Ineffective assertions corrected.
- Initial diagnostic classes.
- Minimal current-CIN and current-LLIR verifiers.
- Stage dumps for debugging tests.
- Baseline p50/p95 frontend-to-`KernelBuildSpec` latency on the curated set.
- Archived legacy generated-kernel results for the full codegen-parity grid on
  Apple M5 and x86/redwood, together with the same-binary A/A control runs that
  calibrate each machine's noise floor.

#### Exit criteria

- Unsupported nodes cannot reach the external C++ compiler as placeholder text.
- Existing supported targeted tests pass.
- New negative tests identify the correct compiler stage.
- The compile-latency baseline and benchmark invocation are checked in, so later
  phases can be compared consistently.
- Both reference machines have a reproducible generated-kernel baseline and A/A
  noise-floor calibration for per-phase comparisons.

### Phase 1: CIN ownership and stable identity

#### Deliverables

- Per-instance CIN state; no mutable class fallbacks.
- Stable IDs for symbols/index variables/tensors.
- Analysis side tables for parents, uses, free/reduction variables, and
  accesses.
- Removal or deprecation of node-owned backreferences.
- Pure CIN construction without `_assignment`, `exec`, or `eval` side effects.
- Canonical CIN serialization and verifier.

#### Exit criteria

- Constructing accesses does not mutate symbol nodes.
- Same-named symbols in distinct scopes remain distinct.
- Normalization is deterministic and idempotent.
- Scheduling tests can compare input CIN before and after without mutation.

### Phase 2: Explicit schedule artifacts and pass infrastructure

#### Deliverables

- `CompileOptions` snapshot.
- Stable `SymbolId`/`IndexId` references for every logical entity addressable by
  `LoopPlan`; no name- or object-identity adapter at this boundary.
- Frozen `ScheduledCIN(cin, plan)` transitional carrier, with no node schema and
  no attached analyses.
- Internal `LoopPlan` translated from public `Schedule`.
- Replacement of dynamic schedule attributes.
- Initial pass manager and pure analysis runner; no result cache, and no
  preserve/invalidate protocol.
- Common current-IR walker/rewriter.
- Four existing sequential LLIR optimizations extracted into passes.
- Per-stage latency instrumentation in production and debug configurations.
- Isolated pass/artifact plumbing latency benchmark.

#### Exit criteria

- One CIN can be compiled under two schedules independently.
- Pass order is explicit and observable.
- Every extracted pass has focused structural tests.
- No schedule metadata is attached dynamically to CIN nodes.
- Isolated plumbing p95 satisfies the 1 ms ceiling. Production-mode
  per-category p50/p95 is measured against the Phase 0 target; any modest
  accepted exceedance is attributed and recorded under the latency policy
  above.
- Each extracted optimization that changes emitted code passes that phase's
  two-machine generated-kernel check in the same PR.

### Phase 3: Structured CxxIR and ABI extraction

This phase improves the existing lowering path before LoopIR is complete, giving
later LoopIR work a safe target.

#### Deliverables

- Structured symbol, subscript, member, call, allocation, and parallel nodes.
- Progressive removal of expression text from `Var.name`.
- Progressive removal of regex/string-based low-level rewrites.
- Unified exhaustive CxxIR emitter; removal of the duplicate generator.
- `TorchCppABI` and result assembly extracted from `CINLowerer`.
- Raw-statement inventory and compatibility budget.
- Parallel zero-fill extracted as a typed pass with focused structural and
  before/after performance coverage.

#### Exit criteria

- Migrated passes use structured access metadata only.
- Codegen handles every verifier-accepted node.
- Kernel signatures and result assembly are constructed outside semantic
  lowering.
- Every emission-affecting extraction passes the two-machine generated-kernel
  check; production compiler latency remains within the Phase 0 target or has a
  documented, explicitly accepted modest exception under the policy above.

### Phase 3.5: Sparse LoopIR feasibility spike and go/no-go

This phase happens before the production LoopIR schema is frozen. It is a spike:
a throwaway prototype whose only job is to answer the riskiest question in the
architecture, which is whether sparse merge semantics can be represented by a
clean one-directional lowering instead of the current recursive
`CINLowerer`/`IterationLattice` coupling.

#### Deliverables

- Hand-authored candidate LoopIR for CSR SpMV intersection.
- Hand-authored candidate LoopIR for sparse elementwise add, including union
  cases with exhausted iterators and coordinates present on only one side.
- A deliberately small test/debug LoopIR interpreter (roughly 200 lines is a
  target, not a correctness constraint) independent of C++ emission.
- Tests comparing results against PyTorch or a simple reference across empty
  inputs, empty rows, unequal sparsity, ragged structures, and randomized
  patterns.
- A schema review documenting how merge selection, coordinate resolution,
  dominance, output assembly, and iterator exhaustion are expressed without
  calling back into a general lowerer and without parsing rendered code.

#### Go criteria

- Both examples execute correctly through the interpreter.
- The candidate IR contains no C++ spelling, no semantics recovered by regex, no
  mutable lowerer callbacks, and no operation-specific escape hatch.
- The verifier can state sparse parent/child dominance and merge-progress
  invariants locally.
- The interpreter and lowering model remain simple enough to serve as an
  independent semantic oracle during the migration — the authority on what a
  program should compute, independent of the compiler under test.
- The Phase 0-3 generated-kernel checks still pass; compiler-latency
  measurements are reviewed with no material or unexplained slowdown; and the
  tile-j/tile-ijk parity objective still appears technically credible.

If these criteria fail, do not begin Phases 4-8. Keep the Phase 0-3 hardening,
revise the LoopIR model or the investment case, and repeat the review. That is a
successful partial outcome, not pressure to force a flawed schema through.

### Phase 4: LoopIR foundation and dense vertical slice

A vertical slice is one operation carried end to end through every stage, from
normalized CIN to a compiled kernel.

#### Deliverables

- LoopIR schema revised from the successful sparse spike, then frozen for the
  first production vertical slices; builder, verifier, printer, and canonical
  serializer.
- Normalized-CIN-to-LoopIR lowering for scalar/dense elementwise operations.
- Dense reduction/matmul vertical slice.
- The spike interpreter promoted into the required test/debug semantic oracle.
- Curated comparison runs that compile the same program through both the legacy
  and the LoopIR path; disabled by default outside dedicated debug and nightly
  workflows.

#### Exit criteria

- Dense vertical slices compile and execute correctly through CxxIR.
- LoopIR stage dumps are deterministic.
- No C++ spelling appears in LoopIR.
- The comparison covers generated structure and numerics.
- Ordinary correctness-test and release JIT wall time does not include double
  compilation.

### Phase 5: Sparse iteration and merge-lattice migration

#### Deliverables

- Explicit compressed position loops and coordinate resolution.
- Coordinate/COO iteration.
- Production structured sparse merge/intersection representation based on the
  Phase 3.5 spike.
- Iteration analysis separated from lowering.
- CSR SpMV and CSR-by-dense SpMM vertical slices.
- Sparse elementwise union/intersection paths.
- The LoopIR interpreter retained as the independent semantic oracle for sparse
  lowering and transformation tests.

#### Exit criteria

- Iteration-lattice code no longer calls the general lowerer or mutates lowerer
  state for migrated paths.
- Parent/child sparse dominance is verifier enforced.
- Migrated sparse operations match PyTorch across empty, ragged, and random
  structures.

### Phase 6: Scheduling migration to LoopIR

#### Deliverables

- Loop reorder pass.
- Affine tiling and ragged-tail pass.
- Workspace materialization pass.
- Sparse panel tiling pass.
- Operand relayout/staging pass.
- Heap/stack/direct result accumulation passes.
- Abstract parallel-loop selection.
- Public Schedule-to-LoopPlan adapter fully exercised.
- `LoopPlan` encodings and typed transformations for representative tile-j and
  tile-ijk schedules, with no algorithm-specific handwritten C++ in the new
  path.

#### Exit criteria

- Schedule decisions are applied only to LoopIR for migrated operations.
- Panel/relayout/result-tile passes perform no name or regex discovery.
- Schedule cache identity comes from canonical `LoopPlan`.
- Explicit and auto schedules pass structural tests and tests comparing the two
  pipelines' numerical results.
- Tile-j/tile-ijk generated kernels are ready for the Phase 8.5 selector-level
  parity check against the corresponding prebuilt implementations after target
  lowering.

### Phase 7: Parallel and target lowering migration

#### Deliverables

- Structured OpenMP and/or ATen target lowering.
- Work-estimate analysis represented structurally.
- Prefetch, restrict, SIMD, and pointer-hoisting target passes.
- Versioned parallel policy in `CompileOptions` and `KernelBuildSpec`.
- Deterministic repeated-compilation and native parallel-execution tests.

#### Exit criteria

- No dynamic hidden fields on loop nodes.
- No process-global thread mutation.
- Parallel correctness is verified under repeated execution, and production
  compiler latency remains within the default target or an explicitly reviewed
  modest exception.

### Phase 8: Cutover and legacy deletion

#### Deliverables

- LoopIR pipeline becomes default for every declared supported
  operation/format.
- Legacy compiler available temporarily behind a diagnostic fallback flag.
- Measurements or test evidence for how often the fallback is taken.
- Deletion of `schedule_lowerer.py`, duplicated walkers/code generators, dynamic
  schedule attributes, and obsolete lowerer state.
- Documentation updated to the new stage model.

#### Exit criteria

- The supported compatibility matrix passes through the new pipeline.
- No production test requires the legacy pipeline.
- Every migrated path passes the generated-kernel check and the compile-latency
  measurement and review protocol defined above. The cutover comparison is the
  new pipeline against legacy generated kernels, neutral-or-better across the
  full grid on both M5 and x86/redwood; parity with handwritten prebuilt kernels
  is deliberately not a cutover criterion (see Phase 8.5).
- The fallback is never taken across the agreed validation suite.
- Legacy deletion reduces complexity rather than merely relocating it.

### Phase 8.5: Validation against the prebuilt schedules

This milestone measures the payoff from the investment. It runs after cutover
and does not hold up legacy deletion: the legacy compiler is even further from
the handwritten kernels than the new pipeline is, so keeping it cannot help
close any gap found here.

#### Deliverables

- Representative tile-j and tile-ijk schedules expressed as `LoopPlan` plus
  typed passes and generated through the production pipeline, with no
  algorithm-specific handwritten C++.
- Full-grid comparison against the corresponding prebuilt kernels on Apple M5
  and x86/redwood using the repository's interleaved repeated-measurement
  method.
- The primary parity measurement at the system level: the runtime tiling
  selector armed with generated kernels versus the selector armed with prebuilt
  kernels across the grid. A prebuilt schedule only runs on the shapes where the
  selector's probe picks it, so beating it on shapes the selector would never
  route to that schedule is not required.

#### Exit criteria

- Selector-level performance with generated kernels is neutral-or-better against
  selector-level performance with prebuilt kernels, under the same thresholds as
  the generated-kernel check, on both machines.
- Any shortfall is documented per shape regime and triggers an explicit scope
  review: additional target-lowering passes, retaining specific prebuilt kernels
  for the affected regimes, or revising the parity claim. No outcome of this
  review reopens retention of the legacy compiler.

### Phase 9: Production hardening

#### Deliverables

- Fuzz/property generation for valid and invalid CIN/LoopPlan combinations.
- Sanitizer-enabled native/JIT validation.
- Determinism and cache-fingerprint tests.
- Continued compiler-stage latency reporting against the Phase 0 target,
  including attribution for accepted exceptions.
- Optional compiler reentrancy and concurrent-compilation tests where an actual
  production use case supports them; not required for cutover.
- Supported IR schema/versioning policy.
- Contributor documentation for adding an operation, analysis, schedule pass,
  LoopIR node, target pass, and ABI feature.

#### Exit criteria

- New compiler features have documented extension points.
- Failures identify the first invalid stage.
- Cache/build artifacts are reproducible for identical build specifications.

## Shadow Compilation and Rollout

Shadow compilation means compiling one program through both pipelines and
executing only the selected one, so the two can be compared without changing
what runs. During Phases 4 through 8, support three explicit modes for migrated
paths:

1. Legacy: compile and execute the current pipeline.
2. Shadow: compile with both pipelines, execute the selected production path, and
   compare structural fingerprints and optionally numerics in tests/debug mode.
3. New: compile and execute the LoopIR pipeline, with explicit diagnostic
   fallback during rollout only.

Fallback must be observable and categorized. A silent fallback would hide
missing coverage and prevent legacy deletion.

Shadow mode is off in production and in the ordinary correctness test suite. It
runs only for a curated debug/CI matrix and scheduled two-machine validation.
Because it compiles twice, its wall time is reported and budgeted separately; it
must not silently double all JIT compilations or pytest cases.

Comparisons should happen at several levels:

- semantic result and output layout;
- loop/access structure;
- schedule decisions;
- generated C++ where stable enough;
- numerical output against PyTorch;
- performance on the representative benchmark matrix.

## Testing Strategy

### Unit tests

- one module per verifier and pass;
- valid, invalid, and unsupported cases;
- no-op behavior;
- idempotence where required;
- input immutability;
- analysis purity, determinism, and association with the correct source IR;
- stable identity and same-name scopes;
- deterministic serialization.

### Structural integration tests

- normalized CIN to LoopIR;
- LoopPlan application;
- sparse iteration shapes;
- workspace and staging lifetime;
- parallel legality;
- CxxIR ABI shape;
- only a small reviewed set of generated C++ goldens.

### Differential execution tests

These run the same program two ways and compare the results:

- PyTorch reference results;
- format and mode-order combinations;
- empty dimensions and empty sparse rows;
- ragged tile/panel tails;
- dtype and index-width variants;
- multiple explicit schedules from one CIN;
- randomized sparse patterns;
- deterministic repeated compilations of the same and different programs.

### Performance tests

- compiler-stage latency separately from kernel execution;
- generated-kernel performance against the legacy pipeline;
- dedicated benchmark runners, not ordinary correctness pytest;
- performance changes attributed to a pass and schedule artifact;
- the concrete M5 and x86/redwood checks in the protocol above, run in the PR
  that changes emission and not deferred to final cutover.

## Compatibility Policy

During migration:

- preserve the public `Schedule` API through an adapter;
- preserve supported output layout and numerical behavior;
- preserve documented compiler options unless intentionally deprecated;
- allow internal IR APIs to change, because they are not yet a stable public
  contract, but update all repository callers atomically;
- characterize current accidental behavior before deciding whether to preserve
  it;
- do not preserve silent failure or input mutation as an internal compiler
  requirement.

## Success Metrics

The refactor is successful when:

- compiling does not mutate semantic CIN or user tensors;
- all stage artifacts verify in test/debug mode and serialize deterministically;
- no pass parses generated C++ text to recover compiler semantics;
- no dynamic fields are attached to IR nodes;
- CIN/LoopIR contain no target-specific C++ spelling;
- CxxIR emission is exhaustive and precedence correct;
- compiler failures identify stage/pass/IR location;
- each schedule transformation has focused structural tests;
- one semantic program can be compiled safely and deterministically under
  multiple schedules;
- new schedules can be expressed as `LoopPlan` plus typed passes without a new
  algorithm-specific C++ kernel;
- the generated tile-j and tile-ijk proof schedules match the handwritten
  prebuilt kernels at the selector level on both M5 and x86/redwood in the
  Phase 8.5 milestone, independent of cutover;
- production compiler latency remains inside the Phase 0 p50/p95 target, or has
  only modest, measured, explicitly accepted exceptions attributable to general
  compiler infrastructure;
- the legacy lowerer/schedule-lowerer path is deleted;
- generated kernel correctness and the concrete performance checks still pass.

## Current-to-Target Responsibility Map

| Current responsibility | Current location | Target owner |
|---|---|---|
| Einsum parsing and inference | `ops.py` | `OpSpec` / frontend |
| Semantic tensor/index program | `cin.py` | normalized CIN v2 |
| Parent/use/free/reduction facts | CIN node backreferences/visitors | analyses |
| Loop ordering and schedule validation | `scheduler.py` | `LoopPlan` builder/verifier |
| Sparse iteration domains | `iterator.py`, `iter_lattice.py` | iteration analysis + LoopIR lowering |
| Affine tiling | CIN scheduler mutation | LoopIR scheduling pass |
| Sparse panels and relayout | `schedule_lowerer.py` | LoopIR scheduling passes |
| Result/workspace assembly | `cin_lowerer.py` | LoopIR memory passes + `TorchCppABI` |
| LLIR optimizations | `CINLowerer` private methods | typed pass modules |
| Parallel policy | lowerer fields and dynamic loop attrs | parallel analysis/lowering |
| C++ AST | `llir.py` plus embedded strings | CxxIR |
| C++ rendering | `codegen.py` plus duplicate generator | exhaustive CxxIR emitter |
| Kernel signature and return ABI | `CINLowerer` | `TorchCppABI` |
| Build fingerprint | ad-hoc source hash | `KernelBuildSpec` |

## Remaining Open Design Decisions

These should be resolved through small prototypes or design reviews rather than
assumed during implementation. Frozen dataclass trees with tuple children, and a
small LoopIR interpreter, are already binding decisions above.

1. Whether LoopIR uses nested regions, a block/CFG model, or a limited
   structured control-flow form. Structured regions are the preferred starting
   point.
2. Whether a distinct target-neutral KernelIR is needed between LoopIR and
   CxxIR. Do not add it until multiple target lowerings demonstrate the need.
3. How much semantic `Where` remains in normalized CIN, as against becoming an
   explicit workspace/reduction plan.
4. Stable index-width policy, and how int32/int64 choices enter type analysis
   and ABI lowering.
5. Multi-result support: explicitly reject it initially, or design first-class
   result tuples before enabling it.
6. Whether ATen parallel lowering should become the default over OpenMP for some
   kernels; this is a target-policy decision, not a LoopIR concern.

## Immediate Next Step

The immediate implementation remains Milestone/Phase 0 plus the first Phase 2
boundary and its minimal Phase 1 identity prerequisite, described in the
companion handoff:

- fail-closed behavior;
- precedence-correct emission;
- minimal verifiers;
- stable `SymbolId`/`IndexId` values for entities referenced by `LoopPlan`;
- a typed `LoopPlan` plus transitional frozen
  `ScheduledCIN(normalized_cin, loop_plan)` carrier replacing dynamic schedule
  attributes;
- focused immutability and determinism tests.

Do not begin full LoopIR implementation until those contracts are in place, and
do not freeze or productionize LoopIR until the Phase 3.5 sparse-intersection
and sparse-union spike passes its go/no-go review. The LoopIR phases above
define where the project is going and how to recognize when each migration step
is complete.
