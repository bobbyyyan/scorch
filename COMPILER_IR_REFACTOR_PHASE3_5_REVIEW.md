# Phase 3.5 Review: Sparse LoopIR Feasibility Spike and Go/No-Go (Repeated)

Date: 2026-07-22 (America/Los_Angeles)

This document supersedes the corrected NO-GO review recorded at `587cbc1`.
That review found that the original candidate schema could not state sparse
parent/child dominance, logical coordinate domains, or physical mode order,
and it blocked Phase 4 until the spike was revised and this review repeated.
This is that repeated review, conducted over the revised spike at code
commits `a3a473b` and `b2fe883`.

## Current verdict

**GO for the general level-based LoopIR foundation.** Every mandatory
Phase-3.5 GO criterion in `COMPILER_IR_REFACTOR_DESIGN.md` is met by the
revised candidate. The recorded boundaries below are scope statements the
schema fails closed on with stable diagnostics, not unstated assumptions, and
none of them falls inside a mandatory criterion.

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
- **Explicit value ownership.** Only a cursor over the value-bearing leaf
  level (the last physical level) may expose a scalar `CursorValue`;
  structural non-leaf reads are the stable `non_leaf_value` defect.
- **Format-neutral level storage.** `levels.py` adds a validated
  `LevelTensorStorage` whose interface (`segment`, `coordinate_at`,
  `leaf_value`) is all the interpreter's execution core consults. CSR is
  exactly one adapter: `from_csr` on the input side, `CsrOutputBuilder` on
  the assembly side. DCSR, CSC, and CSF-like layouts bind the same storage
  class directly, and `from_dense`/`to_dense` round-trip every
  DENSE/COMPRESSED composition under every mode permutation. The former
  CSR-specific `_CSR_LEVELS`/`_CursorState.matrix`/`_segment`/`row_segment`
  interpreter coupling is gone.
- **Scatter accumulation.** `StoreReduce` adds coordinate-addressed
  read-modify-write into dense outputs (`target = target op value`), the
  target-neutral primitive a permuted traversal needs (CSC SpMV) without
  making outputs generally readable.
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

Six hand-authored programs execute differentially against independent
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

Cross-layout differentials require CSR, DCSR, and CSC storage of one logical
matrix to produce identical SpMV results, exactly, on both hand-built and
randomized grids. Position-versus-coordinate discrimination cases (absent
rows/fibers making storage positions diverge from coordinates) execute
correctly for DCSR and CSF. Rank-1 compressed iteration executes. Empty
inputs, empty rows/segments, zero-extent shapes, ragged/mis-shaped denses,
malformed storage, exhaustion, and randomized grids are covered throughout;
wrong-parent, wrong-domain, wrong-mode-order, non-leaf-value, malformed
state, forged identities, and cycle cases are covered adversarially. All 42
verifier defect codes have direct regression coverage.

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
- **All-dense tensors bind logical nested lists**; their declared physical
  mode order does not change interpreter semantics (dense storage order is
  a performance concern with no observable effect in the oracle).
- The interpreter grew to earn format neutrality: the spike package is now
  3,371 lines across seven modules (verifier 1,087; interpreter 592; level
  storage 408; nodes 515; programs 607; CSR container 136). It remains
  plain, Torch-free Python with no compiler-pipeline involvement, and its
  per-node execution logic is still direct enough to audit by reading.

## Criterion-by-criterion decision

| Phase-3.5 GO criterion | Result | Evidence |
| --- | --- | --- |
| Requested CSR examples execute correctly | **Pass** | CSR SpMV and UNION/INTERSECTION execute differentially, exactly; the degenerate sparse-dense SpMV interpretation is justified above and two-cursor merge evidence is separate; DCSR/CSC/CSF extend the corpus. |
| No C++ spelling, parsing, callbacks, or operation escape hatch | **Pass** | Automated import/target-syntax neutrality suite (module inventory, AST import whitelist, subprocess closure proof, production-reference scan, token scan) plus direct source review; merge/descent/assembly semantics are intrinsic to nodes. |
| Verifier states sparse parent/child dominance and merge-progress invariants locally | **Pass** | Dominance is typed parent-position linkage checked at every cursor and dense-position expression, grounded at the root; merge progress is intrinsic to `MergedSparseFor`; domains and value ownership are also stated locally. The old counterexamples are now stable defects. |
| Interpreter/lowering model can serve as an independent semantic oracle | **Pass** | Torch-free, pipeline-free, container-validated plain Python; storage access is behind the neutral level interface with CSR as one adapter; executes general DENSE/COMPRESSED levels with mode permutation. Size growth recorded above. |
| Phase 0-3 gates green; latency reviewed; tile-j/tile-ijk parity credible | **Pass** | All 20 corpus and 42 grid sources regenerated from detached `b2fe883` are byte-identical to the retained `1c78633`/`34a1849` captures (which chain to the Phase-3 finals) — the byte waiver applies and no runtime kernel benchmark is required. Both commits touch only the isolated spike and its tests, which the neutrality suite proves are outside every production import, so the retained paired latency receipts (all inside 1.10) remain the operative measurement. No production, native, cache, or emission surface changed, so the parity objective's credibility is unchanged. |

Every mandatory criterion holds, so the verdict is GO.

## Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase35-repeat-b2fe883/`.
Commits reviewed (stacked on `587cbc1`; nothing amended, nothing pushed):

- `a3a473b` — `feat(compiler): add format-neutral level storage to LoopIR spike`
- `b2fe883` — `feat(compiler): rebase LoopIR spike on logical dimensions and positions`

Receipts:

- focused spike suites at `b2fe883`: **539 passed** (126 verifier, 407
  execution/differential, 6 neutrality); commit `a3a473b`'s intermediate
  tree also passes its 329 tests from a detached worktree;
- spike plus identity/CIN-analysis/LoopPlan/raw-budget adjacency from the
  detached `b2fe883` worktree: **664 passed**;
- fresh 20-source corpus and 42-source grid captures from detached
  `b2fe883` are byte-identical to the retained `1c78633` candidate captures
  and to the `34a1849` base captures (`diff -rq` empty in all four
  comparisons), so the byte waiver applies;
- static parity: Black reports only the single inherited finding
  (`prebuilt_kernels.py`); Flake8 reports exactly the nine inherited
  findings; mypy (`--check-untyped-defs`) reports exactly the 146 inherited
  errors in 12 files with zero findings in the spike; `git diff --check`
  clean before every commit;
- the authoritative clean detached-worktree non-performance suite at exact
  final code/test commit `b2fe883`, with import provenance asserted
  (including the spike interpreter and level-storage modules) and
  caches/basetemp isolated, passed **3,099 tests with 14 skipped, 3
  perf-marked deselections, and one known warning in 2,168.57 seconds**
  (junit: 3,113 collected, 0 failures, 0 errors);
- the five protected tracked files retain their recorded SHA-256 values;
  staging used explicit pathspecs only; no GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, or tooling material was touched; and the
  local remote-tracking ref and live `git ls-remote` tip of
  `origin/refactor/compiler-ir-phase3-std-move-call` both remain at
  `1714df2` — nothing was pushed.

## Conditions carried into Phase 4 (work items, not gate failures)

1. Freeze the production LoopIR schema **from** this candidate: builder API,
   printer, canonical serializer, and the workspace/parallel/tile surface
   are Phase-4 deliverables on top of these nodes.
2. Promote the interpreter as the test/debug semantic oracle; keep the
   neutrality discipline until the strangler path deliberately imports it.
3. Hierarchical merge descent, non-CSR sparse output assembly adapters, and
   `COORDINATE`/`SINGLETON` iteration are Phase-5 surfaces; their fail-closed
   diagnostics (`unsupported_sparse_hierarchy`, "unsupported sparse output
   layout", `unsupported_level_kind`) are the tracked gates.
4. The biggest Phase-4 risk remains CIN-to-LoopIR lowering, not the schema:
   nothing in this spike lowers from normalized CIN yet.
