# Phase 3.5 Review: Sparse LoopIR Feasibility Spike and Go/No-Go (Repeated)

Date: 2026-07-22 (America/Los_Angeles)

This document supersedes the corrected NO-GO review recorded at `587cbc1`.
That review found that the original candidate schema could not state sparse
parent/child dominance, logical coordinate domains, or physical mode order,
and it blocked Phase 4 until the spike was revised and this review repeated.
This is that repeated review, conducted over the revised spike at code
commits `a3a473b` and `b2fe883`, followed by independent-review corrections
through `6ca14f5`.

## Current verdict

**GO-with-conditions to begin Phase 4; NO-GO to freeze or productionize this
spike unchanged.** Every mandatory Phase-3.5 feasibility criterion in
`COMPILER_IR_REFACTOR_DESIGN.md` is met by the review-corrected candidate at
`6ca14f5`. The recorded boundaries below are scope statements the schema fails
closed on with stable diagnostics, not unstated assumptions, and none of them
falls inside a mandatory feasibility criterion.

The first repeated-review stopping point at `b2fe883` did **not** yet justify
that sentence. Independent review found that DENSE value-bearing leaves below
sparse levels were storage-representable but unexecutable, caller-owned level
records could be mutated after validation to change a result, and
`StoreReduce(MUL)` silently used the wrong zero identity. A mapped-extent-free
`DenseFor` also deferred failure to execution, and the CSR adapter still
accepted coercible non-numeric scalar objects. Four review-correction commits
close those gaps and add the adversarial evidence recorded here; the GO verdict
applies to the corrected candidate, not to `b2fe883` alone.

Phase 4 was deliberately **not started in this session**; this GO is an
independently auditable stopping point. The spike remains outside production
imports and must stay there until the Phase-4 strangler path deliberately
promotes it.

## Reproduction of the NO-GO counterexamples

Before designing the replacement, every blocking finding of the superseded
review was independently reproduced against the prior HEAD (`587cbc1`);
script and transcript are `reproduce_nogo.{py,out}` in the evidence ledger:

1. **CSR/CSC indistinguishable.** `TensorDecl` had fields
   `(node_id, symbol, name, levels)` only; the CSR and CSC declarations were
   equal objects, so no physical-to-logical mapping existed.
2. **Merge domains unverifiable, with a live wrong result.** An INTERSECTION
   merge of a row-compressed operand with a logically transposed
   column-compressed operand verified and executed, producing
   `[[10, 40], [0, 120]]` where the logical intent was `[[10, 0], [60, 120]]`
   — physical level number silently stood in for coordinate domain.
3. **DCSR unrepresentable.** A level-0 compressed cursor failed closed with
   `unsupported_sparse_hierarchy`; compressed-under-compressed descent could
   not be expressed at all.
4. **The pre-correction acceptance** (a level-1 cursor with a constant outer
   coordinate on a `(COMPRESSED, COMPRESSED)` tensor, no dominating level-0
   cursor) was confirmed rejected — an honest boundary, not a solution.
5. **Value ownership unposeable.** Cursors existed only on compressed
   leaves, so the non-leaf `CursorValue` question could not even be asked.

## What was revised

The revision keeps the proven merge control and fail-closed discipline and
rebuilds the representation around three separated spaces:

- **Logical dimensions.** Programs declare stable `DimensionId` identities
  (`DimensionDecl`); every tensor maps each logical mode to one declared
  dimension. Shared extent and shared coordinate domain are now the same
  identity. The interpreter resolves every dimension's extent from all bound
  inputs and output shapes *before* materializing anything, so incompatible
  shapes still fail independently of stored sparsity. The temporary
  `ExtentEquality` contract is **lowered into shared dimension identities
  and deleted** (the open decision from the superseded review); `DimSize` is
  deleted with it, and `DenseFor` iterates a declared dimension directly.
- **Physical levels with explicit mode order.** `TensorDecl.levels` is now a
  tuple of `LevelDecl(kind, mode)`: physical storage order plus the logical
  mode each level stores, verified to be a permutation
  (`invalid_mode_order`). CSR `(dense@0, compressed@1)` and CSC
  `(dense@1, compressed@0)` are structurally distinct.
- **Physical positions, separate from coordinates.** `RootPosition`,
  `DensePosition(tensor, level, parent, coord)` (dense positions are the
  arithmetic `parent * extent + coord`), and `PositionValue(position)` are
  position-typed expressions. `SparseFor` binds a `PositionId` beside its
  coordinate; `SparseCursorDecl` names its dominating parent position
  explicitly instead of carrying outer coordinates. The verifier's
  expression typing carries (tensor, level) linkage on every position, so a
  compressed child must reference a position of the immediately dominating
  level of the same tensor (`parent_position_mismatch` otherwise), grounded
  at the root. Positions are never recovered from coordinates, rendered
  names, callbacks, or implicit interpreter state.
- **Explicit value ownership.** A compressed leaf cursor exposes its scalar
  through `CursorValue`; `PositionLoad(tensor, position)` reads any
  value-bearing leaf position, including a DENSE leaf below compressed
  structure. Both forms require the last physical level of the exact input
  tensor; structural non-leaf reads are the stable `non_leaf_value` defect,
  and cross-tensor positions are `position_load_mismatch`.
- **Format-neutral level storage.** `levels.py` adds a validated
  `LevelTensorStorage` whose interface (`segment`, `coordinate_at`,
  `leaf_value`) is all the interpreter's execution core consults. CSR is
  exactly one adapter: `from_csr` on the input side, `CsrOutputBuilder` on
  the assembly side. DCSR, CSC, and CSF-like layouts bind the same storage
  class directly, and `from_dense`/`to_dense` round-trip every
  DENSE/COMPRESSED composition under every mode permutation. Binding takes a
  validated deep structural snapshot, so later mutation of caller-owned level
  records cannot redirect execution. Container boundaries accept only exact
  integer/float scalar types and do not invoke arbitrary conversion callbacks.
  The former CSR-specific
  `_CSR_LEVELS`/`_CursorState.matrix`/`_segment`/`row_segment` interpreter
  coupling is gone.
- **Scatter accumulation.** `StoreReduce` adds coordinate-addressed ADD into
  zero-initialized dense outputs (`target += value`), the target-neutral
  primitive a permuted traversal needs (CSC SpMV) without making outputs
  generally readable. Other operators fail with
  `unsupported_store_reduction` until output initialization carries their
  identity explicitly.
- **`COORDINATE`/`SINGLETON` disposition.** Both production level kinds are
  now declared `LevelKind` members, and the verifier fails closed on any
  tensor declaring them with the stable `unsupported_level_kind` defect.
  Their iteration semantics are the recorded Phase-5 surface
  ("Coordinate/COO iteration" is a Phase-5 deliverable in the design); the
  spike makes the gap explicit instead of leaving the kinds unrepresentable.
- **`MergedSparseFor` semantics unchanged.** Minimum-coordinate selection,
  aligned advancement, UNION/INTERSECTION emission and exhaustion, and the
  progress argument are exactly the proven Phase-3.5 semantics; merges gain
  the requirement that all cursors iterate one shared logical dimension
  (`merge_domain_mismatch`).

Each blocking counterexample now has a stable resolution: CSR and CSC are
distinct declarations and the CSC fixture executes the permuted layout
correctly (`test_csc_spmv_is_not_transposed` proves it is not a transpose);
the transposed-operand merge is rejected with `merge_domain_mismatch`; DCSR
descent executes through bound parent positions
(`test_dcsr_spmv_row_positions_differ_from_row_coordinates` discriminates
positions from coordinates); the old falsely-accepted shapes are rejected
with `parent_position_mismatch`/`layout_mismatch`; and non-leaf value reads
are rejected with `non_leaf_value`.

## The SpMV "intersection" wording

The design asks for "CSR SpMV intersection". The SpMV fixture implements the
sparse-dense intersection in its degenerate optimized form — one sparse
cursor plus a coordinate-addressed dense load — which this review accepts
explicitly, with the following justification: intersecting a sparse iterator
with a dense operand whose iteration space is total yields exactly the
sparse iterator, so the canonical lowering of that intersection *is* the
single-cursor loop (the same optimization the production merge lattice
performs). The fixture is therefore deliberately not claimed as sparse-sparse
merge evidence. Genuine synchronized two-cursor INTERSECTION is demonstrated
by the two-CSR elementwise multiply, which reuses the same `MergedSparseFor`
node with no operation-specific machinery, and genuine UNION with exhausted
and one-sided iterators by the two-CSR add.

## Executed evidence

Eight hand-authored programs execute differentially against independent
pure-Python dense references, exactly (accumulation orders match, so no
tolerances are involved anywhere):

- **CSR SpMV**, **CSR+CSR UNION add** (one-sided coordinates, early
  exhaustion, explicit-zero cancellation), and **two-CSR INTERSECTION
  multiply** (structural, not value-based) — the original corpus, preserved
  case-for-case under the revised schema.
- **DCSR SpMV** — compressed-under-compressed parent-position descent; the
  inner cursor's parent is the position bound by the outer sparse loop.
- **CSC SpMV** — the same logical `y = A @ x` over column-major physical
  storage, scattering through `StoreReduce`; proves physical/logical mode
  separation.
- **CSF-like three-level row contraction** — two chained parent-position
  descents over an all-compressed rank-3 tensor.
- **Compressed/dense-leaf SpMV** — compressed physical columns dominate dense
  row leaves under a permuted mode order; `PositionLoad` reads the leaf scalar.
- **Dense/compressed/dense contraction** — a rank-3 mixed hierarchy exercises
  a dense value-bearing leaf below a sparse middle level.

Cross-layout differentials require CSR, DCSR, CSC, and compressed/dense-leaf
storage of one logical matrix to produce identical SpMV results, exactly, on
both hand-built and randomized grids. Position-versus-coordinate
discrimination cases execute correctly for DCSR, CSF, and both dense-leaf
fixtures. Rank-1 compressed iteration executes. Empty inputs, empty
rows/segments, zero-extent shapes (including mixed rank 3), signed zero,
ragged/mis-shaped denses, malformed or subsequently forged storage,
exhaustion, and randomized grids are covered throughout; wrong-parent,
wrong-domain, wrong-mode-order, wrong-tensor position loads, non-leaf values,
malformed state, forged identities, and cycle cases are covered adversarially.
All 45 verifier defect codes have direct regression coverage.

## Recorded boundaries (fail closed, not silently assumed)

- **Merged cursors must target value-bearing leaf levels**
  (`unsupported_sparse_hierarchy`). Hierarchical merge *descent* (merging
  non-leaf levels and descending into children, e.g. DCSR+DCSR union) is
  not represented; it belongs to the Phase-5 production merge/lattice
  migration. Hierarchical descent itself is proven by the single-cursor
  fixtures; merge control is proven at leaf level.
- **Sparse output assembly is canonical CSR only** ("unsupported sparse
  output layout" otherwise); dense outputs are rank ≤ 2. Sparse outputs in
  other layouts are future assembly adapters over the same append stream.
- **`COORDINATE`/`SINGLETON` fail closed** (`unsupported_level_kind`), with
  Phase 5 as the recorded gate for their iteration.
- **All-dense tensors bind owned logical nested lists/tuples or an explicit
  `LevelTensorStorage`**. The explicit form preserves shapes such as `(0, n)`
  that nested values cannot infer. Ordinary coordinate `Load` stays on the
  logical copy; physical storage is materialized lazily only for
  `PositionLoad`.
- The interpreter grew to earn format neutrality: the spike package is now
  3,967 lines across seven modules (verifier 1,136; interpreter 675; level
  storage 627; nodes 533; programs 817; CSR container 152; package metadata
  27). It remains
  plain, Torch-free Python with no compiler-pipeline involvement, and its
  per-node execution logic is still direct enough to audit by reading.

## Criterion-by-criterion decision

| Phase-3.5 GO criterion | Result | Evidence |
| --- | --- | --- |
| Requested CSR examples execute correctly | **Pass** | CSR SpMV and UNION/INTERSECTION execute differentially, exactly; the degenerate sparse-dense SpMV interpretation is justified above and two-cursor merge evidence is separate; DCSR/CSC/CSF and two dense-leaf layouts extend the corpus. |
| No C++ spelling, parsing, callbacks, or operation escape hatch | **Pass** | Automated import/target-syntax neutrality suite (module inventory, AST import whitelist, subprocess closure proof, production-reference scan, token scan) plus direct source review; merge/descent/assembly semantics are intrinsic to nodes. |
| Verifier states sparse parent/child dominance and merge-progress invariants locally | **Pass** | Dominance is typed parent-position linkage checked at every cursor and dense-position expression, grounded at the root; merge progress is intrinsic to `MergedSparseFor`; domains and value ownership are also stated locally. The old counterexamples are now stable defects. |
| Interpreter/lowering model can serve as an independent semantic oracle | **Pass** | Torch-free, pipeline-free, container-validated plain Python; storage access is behind the neutral level interface with CSR as one adapter; executes general DENSE/COMPRESSED levels with mode permutation, including dense value-bearing leaves; snapshots caller storage before execution. Size growth recorded above. |
| Phase 0-3 gates green; latency reviewed; tile-j/tile-ijk parity credible | **Pass** | The original 20 corpus and 42 grid captures at `b2fe883` are byte-identical to retained Phase-3 captures. Corrections through `6ca14f5` touch only the import-neutral spike/tests and the neutrality suite remains green, so no production emission or measured compiler path can change; the byte waiver and retained latency receipts (all inside 1.10) remain operative. No production, native, cache, or emission surface changed, so the parity objective's credibility is unchanged. |

Every mandatory feasibility criterion holds, so Phase 4 may begin under the
conditions below. This is not approval to freeze the spike unchanged.

## Verification

Original evidence ledger:
`/Users/bobby/.cache/scorch-codex/phase35-repeat-b2fe883/`. Final independent-
review ledger: `/Users/bobby/.cache/scorch-codex/phase35-final-6ca14f5.9ZVrpS/`.
Its verified `SHA256SUMS` manifest hashes to
`324ab70763a65308c4d7bc0a28aa07e6e78772b4e3d089aeccaac7a896b60cd5`.
Commits reviewed (stacked on `587cbc1`; nothing amended, reordered, or pushed):

- `a3a473b` — `feat(compiler): add format-neutral level storage to LoopIR spike`
- `b2fe883` — `feat(compiler): rebase LoopIR spike on logical dimensions and positions`
- `7f7af51` — `fix(compiler): complete level-general LoopIR spike`
- `aee7c1f` — `test(compiler): cover LoopIR review corrections`
- `a9610cb` — `fix(compiler): reject coercive CSR scalar inputs`
- `6ca14f5` — `test(compiler): lock exact CSR scalar boundaries`

Receipts:

- focused spike suites at clean detached `6ca14f5`: **647 passed** (136
  verifier, 505 execution/differential, 6 neutrality); commit `a3a473b`'s
  intermediate tree also passes its historical 329 tests from a detached
  worktree;
- spike plus identity/CIN-analysis/LoopPlan/raw-budget adjacency from the
  clean detached `6ca14f5` worktree: **772 passed**;
- fresh 20-source corpus and 42-source grid captures from detached
  `b2fe883` are byte-identical to the retained `1c78633` candidate captures
  and to the `34a1849` base captures (`diff -rq` empty in all four
  comparisons), so the byte waiver applies;
- Black and Flake8 are clean over all changed source/test files; focused mypy
  succeeds over all seven spike modules; `git diff --check` is clean. The
  prior full-source inherited-baseline comparison remains applicable because
  the correction adds no finding in the isolated package;
- the authoritative clean detached-worktree non-performance suite at exact
  final code/test commit `6ca14f5`, with import provenance asserted
  (including the spike interpreter and level-storage modules) and
  caches/basetemp isolated: **3,207 passed, 14 skipped, 3 perf-marked
  deselections, and one known warning in 2,216.69 seconds** (JUnit: 3,221
  selected, 0 failures, 0 errors);
- the five protected tracked files retain their recorded SHA-256 values;
  staging used explicit pathspecs only; no GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, or tooling material was touched; and the
  local remote-tracking ref and live `git ls-remote` tip of
  `origin/refactor/compiler-ir-phase3-std-move-call` both remain at
  `1714df2` — nothing was pushed.

## Conditions carried into Phase 4 (work items, not gate failures)

1. Begin with a production-responsibility/gap audit against normalized CIN and
   the first dense vertical slice. Revise the spike candidate where that audit
   requires production identity, serialization, construction, or target split,
   then freeze only the first production subset (builder API, verifier,
   printer, and canonical serializer). The spike is input to that decision,
   not an unchanged contract.
2. Promote the interpreter as the test/debug semantic oracle; keep the
   neutrality discipline until the strangler path deliberately imports it.
3. Hierarchical merge descent, non-CSR sparse output assembly adapters, and
   `COORDINATE`/`SINGLETON` iteration are Phase-5 surfaces; their fail-closed
   diagnostics (`unsupported_sparse_hierarchy`, "unsupported sparse output
   layout", `unsupported_level_kind`) are the tracked gates.
4. Production schema integration and CIN-to-LoopIR lowering are co-leading
   Phase-4 risks: the semantic core is feasible, but nothing here yet models
   every production responsibility or lowers normalized CIN.
