# Phase 3.5 Review: Sparse LoopIR Feasibility Spike and Go/No-Go

Date: 2026-07-22 (America/Los_Angeles)

This document supersedes the initial review at `62e94ff`. That review correctly
recorded the CSR execution evidence, but drew a GO conclusion from a schema that
does not meet one of the design's mandatory GO criteria.

## Current verdict

**NO-GO for the general level-based LoopIR foundation.** Do not begin Phase 4.
Revise the spike and repeat this review first.

The narrower result is positive: **the canonical CSR sparse-merge control
algorithm is feasible**. The same target-neutral `MergedSparseFor` correctly
executes UNION and INTERSECTION over sorted, unique CSR row segments, with
minimum-coordinate selection, aligned-cursor advancement, correct exhaustion,
and guaranteed progress. That result is worth retaining, but it is not the
Phase-3.5 GO defined in `COMPILER_IR_REFACTOR_DESIGN.md`.

The design requires every GO criterion to hold and says that, if one fails,
Phases 4-8 must not start. The criterion that the verifier can state sparse
parent/child dominance locally is not met. Calling the missing representation a
Phase-4 revision condition would move a gating design question past its gate.

## The CSR question

The long-term compiler must be based on general physical levels and logical
modes, not on CSR as its compiler model.

`csr.py` and the three CSR fixture builders are appropriate: Phase 3.5
explicitly asks for CSR SpMV and CSR union/intersection examples. The defect is
not that CSR appears in the spike. The defect is that CSR storage assumptions
also live inside `interp.py`, while the candidate nodes omit information that a
format-neutral level interpreter would need:

- `_CSR_LEVELS`, `_CursorState.matrix`, `_segment`, `row_segment`, `indices`,
  `values`, and `_CsrOutputBuilder` make the interpreter CSR-specific;
- `SparseCursorDecl.outer_indices` contains coordinates, not the dominating
  parent storage position needed to select a compressed child segment;
- `TensorDecl.levels` has no physical-level-to-logical-mode mapping or logical
  coordinate-domain identity, so CSR and CSC have the same IR shape;
- `LevelKind` omits Scorch's `COORDINATE` and `SINGLETON` level kinds; and
- merged cursors cannot be proven to iterate the same logical domain.

The CSR interpreter is therefore a useful, independent oracle for the narrow
CSR merge experiment. It is not yet the general level-based semantic oracle
that Phase 4 is supposed to promote.

## Commits reviewed and corrections

Original spike, stacked on the Phase-3 closure at `34a1849`:

- `71eba38` — `feat(compiler): prototype sparse LoopIR schema and verifier`
- `1e30817` — `feat(compiler): interpret sparse LoopIR programs`
- `1c78633` — `test(compiler): execute sparse LoopIR feasibility cases`
- `62e94ff` — `docs(compiler): record Phase-3.5 go-no-go review`

Rigorous-review corrections:

- `de1d1d7` — `fix(compiler): make sparse LoopIR spike fail closed`
- `79c5837` — `test(compiler): cover sparse LoopIR review boundaries`
- `245e673` — `fix(compiler): validate LoopIR extents before allocation`
- `c17636a` — `test(compiler): lock pre-allocation extent validation`
- `8280ad8` — `fix(compiler): snapshot LoopIR input bindings once`
- `5733f30` — `test(compiler): cover stateful LoopIR input mappings`
- `ee58ad9` — `fix(compiler): contain LoopIR mapping lookup failures`
- `7f32141` — `test(compiler): cover disappearing LoopIR bindings`
- `775a408` — `fix(compiler): validate LoopIR mapping key identities`
- `b59491b` — `test(compiler): reject hostile LoopIR mapping keys`
- `64163d8` — `fix(compiler): validate LoopIR keys before hashing`
- `e719be3` — `test(compiler): cover colliding LoopIR key snapshots`

Nothing in the spike or its corrections is imported by production compilation,
JIT, public APIs, caches, LLIR, code generation, or native code.

## Findings

### 1. Parent/child dominance is not representable as claimed

The original review said arity-checked `outer_indices` made sparse dominance
structural and described DCSR/multi-level traversal as representable but
untested. That is false for this schema.

For a compressed child, Scorch's actual level storage indexes the child's
position array with the parent's physical storage position. A parent coordinate
is not interchangeable with that position. Yet `SparseFor` binds only a
coordinate, and `SparseCursorDecl` has no `LevelRef`, `PositionId`, parent cursor,
or position expression.

The old verifier accepted a top-level level-1 cursor on a
`(COMPRESSED, COMPRESSED)` tensor with `outer_indices=(IntConst(0),)` and no
level-0 cursor at all. It also accepted scalar `CursorValue` reads from
non-leaf compressed levels. Both contradict explicit one-directional level
lowering.

The correction now fails closed with `unsupported_sparse_hierarchy` unless a
cursor targets a compressed leaf whose outer levels are all dense. This makes
the current boundary honest; it does not solve hierarchical sparse iteration.

### 2. Logical modes and merge domains are missing

Physical level number is not logical coordinate identity. Production Scorch
supports mode order, so two `(DENSE, COMPRESSED)` layouts may denote CSR-like or
CSC-like traversal depending on the mapping. The candidate cannot distinguish
them.

For the same reason, `MergedSparseFor` cannot locally verify that every cursor
produces coordinates in one shared logical domain. Adding a check that physical
level numbers happen to match would create false confidence. The repeat spike
needs explicit logical dimension/domain identities and physical mode order.

### 3. Fixture shape compatibility was data-dependent

The original interpreter used access-time bounds checks but encoded no static
relationship between operand and output extents. This admitted plausible wrong
results whenever sparsity hid the mismatch. Reproduced examples included:

- a `1 x 3` SpMV matrix storing only column 0 with a length-1 vector; and
- UNION of a one-row left operand with a two-row right operand, where the extra
  right row was silently ignored.

The correction adds target-neutral `ExtentEquality` program preconditions and
checks them after all input/output shapes are registered but before input copies,
output allocation, or execution.
SpMV declares `A[0] == y[0]` and `A[1] == x[0]`; both elementwise fixtures
declare `A[0] == B[0] == C[0]` and `A[1] == B[1] == C[1]`. This removes
sparsity-dependent shape acceptance for the three fixtures. It is deliberately
not a substitute for logical dimension IDs or mode order in the revised schema.

### 4. Malformed stored state leaked Python exceptions

Deleting ordinary dataclass fields such as `Block.statements` or identity
fields such as `LoopNodeId.value` could leak `AttributeError`, despite the
verifier being described as the single fail-closed authority.

The correction preflights every exact node's stored dataclass fields before a
checker reads them and hardens all four identity readers. Missing node state now
raises `LoopIRVerificationError` with `malformed_state` at the exact field path;
missing or invalid identity values retain their specific `invalid_*_id` codes.

The original “every defect code is tested” statement was also inaccurate:
`invalid_index_id` and `invalid_cursor_id` had no tests. The corrected suite
exercises all 31 current defect codes, including those two and the two new
boundaries.

### 5. Position-array wording was too broad

The IR nodes contain no position arrays, offsets, target syntax, or rendered
name recovery. The package as a whole does contain CSR positions: `CsrMatrix`
owns `indptr`, the interpreter reads row offsets, and `_CsrOutputBuilder`
constructs canonical CSR output. That storage belongs to the CSR oracle adapter,
not to LoopIR nodes. Claims that no positions or offsets existed “anywhere in
the package” are superseded by this narrower statement.

### 6. Caller-supplied mappings crossed the validation boundary

The original interpreter repeatedly consulted caller-owned input mappings, so
a stateful mapping could advertise one tensor during key validation and return
another during materialization. Iteration and lookup exceptions also escaped as
arbitrary Python errors. The first key-identity correction still constructed a
set before checking key types; a foreign key colliding with an exact
`SymbolId` could therefore run its `__eq__` callback during set construction.

The corrections snapshot every mapping value once, contain iteration and
lookup failures as `LoopIRInterpreterError`, snapshot advertised keys into a
tuple before hashing, and reject every non-exact or malformed `SymbolId` before
constructing role sets. Regression mappings cover changing/disappearing values,
lookup failures, malformed keys, and a hostile key advertised beside the exact
colliding program key.

## What the spike does establish

- One generic `MergedSparseFor` control rule handles canonical CSR UNION and
  INTERSECTION without operation-specific nodes or callbacks.
- Candidate selection, aligned advancement, exhaustion, and the strict progress
  argument are sound for canonical sorted CSR segments.
- The three CSR fixtures execute differentially against independent dense
  references across empty, disjoint, overlapping, one-sided, early-exhaustion,
  explicit-zero, and randomized cases.
- Ordered `AppendEntry` is sufficient to assemble canonical CSR output in the
  tested identity-mode-order cases.
- The spike remains Torch-free, target-syntax-free, and isolated from production
  imports.
- Phase 0-3 generated-code, latency, quality, and correctness gates were not
  affected by adding the isolated experiment.

These are useful feasibility results. They do not establish general
hierarchical level traversal or a production-ready LoopIR schema.

There is one deliverable wording ambiguity worth preserving. The design asks
for “CSR SpMV intersection.” The fixture implements the sparse-dense
intersection in its degenerate optimized form: one sparse cursor plus a dense
load, so it does not exercise synchronized cursor intersection. A separate
two-CSR elementwise multiply supplies the genuine two-cursor INTERSECTION test.
That is strong evidence for merge control, but the repeat review must either
justify the degenerate SpMV interpretation explicitly or add the literal mixed
domain/intersection representation the design intended.

## Criterion-by-criterion decision

| Phase-3.5 GO criterion | Result | Evidence |
| --- | --- | --- |
| Requested CSR examples execute correctly | Partial/pass with interpretation | SpMV and UNION execute; SpMV uses a one-cursor sparse-dense intersection, while genuine two-cursor INTERSECTION is demonstrated by a separate elementwise multiply. |
| No C++ spelling, parsing, callbacks, or operation escape hatch | Pass | Automated import/target-syntax neutrality checks and direct source review. |
| Verifier states sparse parent/child dominance and merge progress locally | **Fail** | Merge progress is local; compressed-parent position dominance is absent and was falsely accepted. |
| Interpreter/lowering model can serve as an independent semantic oracle | Partial | Independent and useful for CSR; storage access is hard-coded to CSR and cannot execute general levels. |
| Phase 0-3 gates green and parity remains credible | Pass as non-interference; unproven for LoopIR | Original byte/latency/full-suite receipts remain valid; no production path imports the spike. |

Because one mandatory criterion fails, the overall verdict is NO-GO even though
the merge-control subexperiment passes.

## Verification

The original evidence ledger remains at
`/Users/bobby/.cache/scorch-codex/phase35-loopir-spike-1c78633/`. Its factual
receipts were independently checked during this review:

- 297 original focused tests pass;
- the 20-source corpus and 42-source grid are byte-identical between detached
  `34a1849` and `1c78633` worktrees and chain to the retained Phase-3 captures;
- the recorded full suite contains 2,857 passed, 14 skipped, 3 perf-marked
  deselections, and no failures/errors;
- the recorded latency ratios are inside 1.10 everywhere; and
- original static, import-provenance, protected-file, staging, and origin-tip
  claims match their retained receipts.

Review-correction verification through final code/test commit `e719be3`:

- focused spike suites: **329 passed** (92 verifier, 231 execution, 6
  neutrality);
- focused spike plus identity/CIN-analysis/LoopPlan/raw-budget suites:
  **454 passed**;
- fresh 20-source corpus and 42-source grid captures from detached `b59491b`
  are byte-identical to the retained `1c78633` candidate captures, which already
  chain byte-identically to `34a1849` and the Phase-3 finals; the byte waiver
  therefore applies and no runtime benchmark is required; `64163d8..e719be3`
  touch only the isolated interpreter and its tests, which the neutrality suite
  proves are outside production imports;
- all 31 verifier defect codes have direct regression coverage;
- Black and Flake8 are clean on the seven changed source/test files; mypy reports
  success on the five changed production files; and
- `git diff --check` is clean.

The authoritative clean detached-worktree non-performance run at exact final
code/test commit `e719be3`, with import provenance asserted and caches isolated,
passed **2,889 tests with 14 skipped, 3 perf-marked deselections, and one known
warning in 2,033.63 seconds**. It had no failures or errors.

## Requirements before repeating the review

The repeat spike must resolve the failed criterion rather than relabel it as
future work:

1. Add stable logical dimension/domain identities and an explicit mapping from
   physical levels to logical modes.
2. Add explicit level references and physical position bindings. A compressed
   child must reference a dominating parent position; coordinate recovery may
   not be implicit or string-derived.
3. Define when a cursor position owns a scalar value; non-leaf structural
   levels must not expose leaf values.
4. Put sparse storage behind a format-neutral level interface, with CSR as one
   adapter rather than the interpreter's execution model.
5. Represent and verify merge-domain compatibility and shape relationships
   using the logical dimension model.
6. Execute at least one nested compressed case (DCSR or a three-level CSF-like
   fixture) and one permuted-mode case (for example CSC), in addition to the CSR
   regressions. Include wrong-parent and wrong-domain adversarial tests.
7. Decide and document the required Phase-4 surface for `COORDINATE` and
   `SINGLETON`; unsupported kinds must fail explicitly.
8. Keep the neutrality suite green and keep the package outside production
   imports until a repeated review records GO.

Only after those conditions are implemented and the explicit review is repeated
may Phase 4 begin. Phase 0-3 remains complete and valuable regardless of this
NO-GO; no Phase-3 work is reopened by the decision.
