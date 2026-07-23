# Phase 5 Review: Sparse Iteration and Merge-Lattice Migration

Date: 2026-07-22 (America/Los_Angeles)

This review records the Phase-5 milestone of the compiler IR refactor: the
responsibility audit across the production dense LoopIR, the Phase-3.5
spike, normalized CIN, LoopPlan, the legacy iteration lattice, structured
LLIR, ABI/codegen, and the oracle; the deliberate sparse extension of the
frozen production schema; the pure iteration-domain/merge-lattice analysis
separated from lowering; the four compiled sparse vertical slices with
byte-identical legacy parity; the format-neutral level-storage oracle; and
printing/serialization/stage-timing coverage for every new node.  It builds
on the Phase-4 review (`COMPILER_IR_REFACTOR_PHASE4_REVIEW.md`, including
its §7 corrections) and leaves the closed Phase-3.5 and Phase-4 reviews
unchanged.

## Verdict

**Phase 5 is complete for the migrated sparse level families**, with every
mandatory gate green:

- all four named vertical slices — CSR SpMV, CSR-by-dense SpMM, CSR+CSR
  UNION/add, and CSR·CSR INTERSECTION/multiply — compile and execute
  through normalized CIN → pure iteration-domain analysis → verified LoopIR
  → the existing structured LLIR/CxxIR boundary → ABI/codegen → real
  compiled kernels, and match PyTorch, the production oracle, and (where
  compared) legacy shadow execution;
- for every family member the legacy pipeline supports, the LoopIR
  pipeline's generated C++ is **byte-identical** to the untouched legacy
  pipeline's C++ — an eleven-member sparse parity grid locks it (SpMV
  f32/f64, SpMM, union add → CSR and → dense, intersection multiply → CSR
  and → dense, CSR row sum, sampled dense·CSR elementwise, SpGEMM → dense,
  and a union-times-dense chain);
- the iteration-lattice logic for migrated paths lives in a pure analysis
  module that never calls the lowerer, never mutates phase state, and never
  reads rendered names; parent/child sparse dominance, merge domains, leaf
  value ownership, and default discipline are verifier-enforced with stable
  codes, each directly covered;
- the fresh 20-source corpus and 42-source grid captures are byte-identical
  to the sealed Phase-4 review captures, so no legacy emission changed and
  the byte waiver applies; the retained latency receipts remain operative;
- production neutrality is unchanged: `import scorch`, default compilation,
  legacy correctness paths, and release JIT never load the LoopIR package.

Phase 6 was deliberately not started (see §7).

## 1. Responsibility audit (recorded before nodes were added)

| Responsibility | Owner after Phase 5 |
| --- | --- |
| Sparse node schema (positions, cursors, merges, appends) | `loopir/nodes.py`, revised from the spike: dtype-aware, builder-allocated artifact-local `CursorId`/`PositionId`, no accumulators/`IntConst`/`PositionLoad` (not exercised by the milestone). |
| Structural, dominance, and domain verification | `loopir/verifier.py` remains the single fail-closed authority; spike rules adopted under production codes (16 new, §3). |
| Iteration-domain / merge-lattice analysis | **New** `loopir/iterdomain.py`: a pure function over normalized CIN plus the verified `LoopPlan`, returning an immutable per-loop-variable domain table (dense / single cursor / UNION / INTERSECTION) that fully determines the LoopIR iteration structure. It never imports `CINLowerer`, `iterator.py`, or `iter_lattice.py`. |
| CIN→LoopIR sparse materialization | `loopir/lower_cin.py` consumes the domain table mechanically; the dense families' construction path is unchanged. |
| Level-storage semantics and CSR adapters (oracle side) | **New** `loopir/levels.py`, promoted from the spike's proven level-storage and CSR container modules. |
| Sparse execution semantics (oracle) | `loopir/oracle.py`, extended with the spike interpreter's cursor/merge/append semantics over the level interface. |
| Target emission (position loops, merges, CSR assembly) | `loopir/lower_llir.py`; raw statements mirror the legacy lowering exactly (verified byte-for-byte after the shared passes). |
| Kernel ABI, validation, prologue for sparse inputs | Existing `TorchCppKernelABI`/`KernelTensorABI`, unchanged — they already own compressed `pos`/`crd` bindings. |
| CSR output storage and final assembly | Existing `ResultTensorAssembler`, unchanged — append-built position vectors, counters, and `scorch_tensor_from_vector` assembly. |
| Managed optimization passes | Existing `LLIRPassManager.run_production_pipeline`, unchanged; sparse prefetch and dense-pointer hoisting fire identically because the raw trees match legacy's. |
| Parallel policy | Existing `mark_first_for_loop_parallel`/`apply_parallel_policy`, unchanged; the LoopIR path replicates the legacy gate (dense result written by the outer coordinate) and the legacy merged-nest policy (§4.1). |
| Stage timing | Existing `CompilationContext` stages `CIN_TO_LOOPIR_LOWERING`/`LOOPIR_TO_LLIR_LOWERING`; **no new stage identities** and no legacy sequence change. |
| Runtime marshalling and result wrapping | `loopir/pipeline.py` (test/debug only): sparse results wrap from the program's declared output levels into honestly-typed `ds` STensors. |

General level-based compilation is the audit's governing rule: cursors bind
physical positions beside the coordinates they resolve, parents are named
positions (root, dense arithmetic, or a bound compressed position), and
coordinates are never substituted where parent physical positions are
required — the oracle's DCSR differential (positions ≠ coordinates)
exercises exactly this.

## 2. What was frozen (schema extension)

New in `loopir/nodes.py`: `CursorId`, `PositionId` (builder-allocated,
artifact-local), `MergeMode` {UNION, INTERSECTION}, `FloatConst`,
`RootPosition`, `DensePosition(tensor, level, parent, coord)`,
`PositionValue(position)`, `CursorValue(cursor, default)`,
`SparseCursorDecl(cursor, tensor, level, parent)`,
`SparseFor(cursor, position, coord_index, body)`,
`MergedSparseFor(mode, cursors, coord_index, body)` (min-coordinate
selection, aligned advancement, exhaustion, and guaranteed progress are
intrinsic node semantics), and `AppendEntry(tensor, coords, value)`
(ordered sparse assembly).  `LevelKind.COMPRESSED` became executable.

Deliberately **not** declared (fail-closed, not silently assumed):
`IntConst`, `PositionLoad` (dense value-bearing leaves below compressed
structure are rejected at the CIN boundary), accumulators
(`DeclAccum`/`Accumulate`/`AccumValue`), non-ADD `ReduceOp` members,
workspaces, tiles, and parallel nodes.  COORDINATE/SINGLETON iteration
remains `unsupported_level_kind`.

## 3. Verifier surface

Sixteen stable codes were added, each with a direct adversarial regression:
`invalid_cursor_id`, `invalid_position_id`, `duplicate_cursor_id`,
`duplicate_position_binding`, `unbound_cursor`, `unbound_position`,
`parent_position_mismatch` (typed parent-position linkage per level,
grounded at the root), `layout_mismatch` (coordinate loads/stores only on
all-dense tensors, cursors only on COMPRESSED levels, dense positions only
on DENSE levels, appends only into compressed outputs),
`merge_domain_mismatch`, `degenerate_merge`, `unsupported_sparse_hierarchy`
(merged cursors must target value-bearing leaves; hierarchical merge
descent is unrepresented), `missing_union_default`, `dead_default`,
`default_contains_cursor`, `non_leaf_value`, and
`unsupported_sparse_output` (sparse outputs must be canonical CSR
`(dense@0, compressed@1)`).  The full 44-code surface is locked by a
source-scan test.  Dense-subset behavior is unchanged except the two
boundaries Phase 5 deliberately moved: COMPRESSED input levels are now
executable, and the code-surface lock includes the sparse codes.

## 4. The migrated families and their recorded boundaries

**Lowered and compiled with byte-identical legacy parity** (float32 and
float64; ranks 1–2 outputs over rank-2 sparse operands; identity mode order):

1. CSR SpMV (`y[i] += A[i,j]·x[j]`) and CSR row sums;
2. CSR-by-dense SpMM (`C[i,j] += A[i,k]·B[k,j]`), plus SpGEMM
   `ds@ds → dd` (single-cursor chains at two loop levels);
3. CSR+CSR UNION add into canonical CSR and into dense outputs;
4. CSR·CSR INTERSECTION multiply into canonical CSR and into dense
   outputs, plus sampled `ds·dd → dd` elementwise products and
   `(A+B)·D` union-times-dense chains.

Fail-closed family boundaries (stable codes; regression-locked):

- analysis: unions with dense or loop-invariant operands
  (`unsupported_union_with_dense`/`unsupported_union_operand`), sparse
  subtraction (`unsupported_sparse_subtraction` — its one-sided cases need
  negation; legacy cannot compile SUB at all, recorded Phase-4 erratum),
  merges nested under other merges (`unsupported_nested_merge`),
  COORDINATE/SINGLETON operands (`unsupported_level_type`);
- lowering: non-CSR sparse outputs (`unsupported_sparse_output`), CSR
  outputs with an update operator (`unsupported_sparse_output_reduction`),
  CSR row/column domains outside dense-row/sparse-column
  (`unsupported_sparse_output_domain`), merged reductions and merged
  updates (`unsupported_merged_reduction`/`unsupported_merged_update`),
  dense value-bearing leaves below compressed structure
  (`unsupported_format`);
- target: level-0 (root-parent) cursors and compressed-parent (DCSR)
  descent, merges of more than two cursors, non-innermost merges,
  appends outside a dense-row/merged-column nest, unread inputs, and
  nonzero UNION defaults (`unsupported_program_shape`/
  `unsupported_union_default`).  These verify and (where meaningful)
  oracle-execute, but do not reach codegen — the strangler discipline of
  Phase 4.

Merge arity: the schema and oracle support two or more cursors; the target
lowering supports exactly two (the legacy alignment-case enumeration).
Scheduling facts: the `LoopPlan` consumed is still nest order only; no
scheduling transformations were migrated (Phase 6 surface).

## 5. Legacy observations recorded by the differential work

1. **Merged-nest thread policy is row-count-only.**  For union/intersection
   kernels into dense outputs, the legacy parallel marker runs before its
   nested statement lists are flattened, so `find_sparse_pos_array` cannot
   see the merge's position-array initializers and the applied policy is
   `scorch_nthreads(-1, rows)` / `scorch_chunk(rows, -1)` rather than the
   nnz-aware form the same helper produces for position-loop kernels.  The
   LoopIR target reproduces this policy explicitly (with a comment) because
   byte parity is the gate; improving it is dedicated future work, never a
   silent side effect.
2. **`lower_and_exec_cin` wraps every result as `"dd"`.**  The legacy
   low-level entry mis-wraps CSR results (the compiled kernel's `pos`/`crd`
   tensors are dropped by the dense wrapping); the public einsum path wraps
   correctly.  Sparse-output runtime comparisons against legacy therefore
   ride on byte-identical sources (identical source is the identical cached
   kernel) plus PyTorch/oracle differentials; dense-output families also
   compare via `execute_shadow` bitwise.
3. The four Phase-4 errata are unchanged and none was silently "fixed".

## 6. Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase5-sparse-b068c38/`.

Commits (stacked on `d9c8305`; nothing amended, reordered, or pushed):

- `e71f9a7` — `feat(compiler): extend production LoopIR with the sparse level schema and analyses`
- `fbd89b3` — `feat(compiler): lower the sparse level families with legacy byte parity`
- `b068c38` — `test(compiler): lock the Phase-5 sparse vertical slice`
- (docs commit follows this review)

Receipts:

- focused production LoopIR membership: **280 passed + 4 neutrality**
  (87 verifier, 14 printer, 32 oracle, 14 level-storage, 14
  iteration-domain, 37 CIN lowering, 57 LLIR lowering/parity, 25 pipeline
  execution) — 126 tests beyond the corrected Phase-4 membership of 154,
  including the eleven-member sparse byte-parity grid and real compiled
  PyTorch/oracle/shadow differentials for every migrated family across
  empty rows/inputs, disjoint and overlapping supports, one-sided
  exhaustion, explicit-zero cancellation, zero extents, and randomized
  grids;
- spike suites untouched and green: **647 passed**; adjacency:
  cin-analysis/loop-plan/stage-timing/llir-traversal/cin/scheduler/
  schedule-api **666 passed**, llir-pass-manager + string budget **89
  passed**;
- byte gates: fresh 20-source corpus and 42-source grid captures from the
  working tree are byte-identical to the sealed Phase-4 review captures
  (`diff -qr` empty for both), which chain to the Phase-3 finals — no
  legacy emission changed, the byte waiver applies, and no runtime kernel
  benchmark is required; the new sparse kernels are byte-identical to the
  legacy kernels they migrate, so the compiled artifacts are shared;
  structural activation (prefetch, pointer hoist, position loops, ordered
  assembly, both parallel policies) is asserted directly and never waived;
- latency: no production emission or measured legacy compiler path changed
  (the LoopIR stages remain strangler-path-only and release JIT never
  enters them), so the retained paired latency receipts remain operative
  under the 1.10 policy;
- static parity: Black clean over the package and changed tests; Flake8
  clean over the same; focused mypy over all eleven package modules
  succeeds; full-source `mypy --check-untyped-defs src` reports exactly
  the 146 inherited errors in 12 files, **zero** in `loopir/`;
  `git diff --check` clean before every commit;
- the authoritative clean detached-worktree non-performance suite at exact
  code commit `b068c38`, with isolated pytest/Torch-extension/cache
  directories and import provenance asserted (including the new
  `iterdomain` and `levels` modules): **3,491 passed, 14 skipped, 3
  perf-marked deselections, one known warning, and zero failures/errors
  in 2,281.05 seconds** — exactly the Phase-4 corrected baseline of
  3,365 plus the 126 new LoopIR tests.  The JUnit reports 3,505
  selected tests and has SHA-256
  `d1744f2afa5862367c43f48d710744f58dd6ac15c81d615f206ed47a4fda6926`;
- the five protected tracked files retain their recorded SHA-256 values;
  staging used explicit pathspecs only; no GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, or tooling material was touched; origin
  `refactor/compiler-ir-phase3-std-move-call` remains at `58e8565` and
  nothing was pushed.

## 7. Limitations and Phase-6 gates

- The Phase-6 stretch objective (one scheduled CSR-by-dense SpMM slice)
  was **not started**: `LoopPlan` still carries nest order only, and
  representing an affine-tile schedule as immutable scheduled LoopIR
  requires the tile/split node family the audit deliberately excluded.
  That representation decision belongs to Phase 6 proper.
- The target lowering's fail-closed surface (§4) is the exact boundary of
  the migrated families; DCSR, CSC, and CSF-like layouts are
  verifier-valid and oracle-executable but not compiled.
- Canonical dumps (schema `scorch.loopir.canonical.v2`) remain semantic
  fingerprints that omit display names; the Phase-4 caution stands: they
  are not sufficient target-artifact cache keys, and kernel caching remains
  source-derived.
- The oracle is a Python-float semantic reference; compiled comparisons
  use the repository's standard tolerances, while oracle-versus-pure-Python
  references remain exact.
