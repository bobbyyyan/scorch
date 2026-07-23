# Phase 6 Review: Scheduling Migration to LoopIR (Explicit Reorder + Affine Tiling)

Date: 2026-07-23 (America/Los_Angeles)

This review records the first Phase-6 milestone of the compiler IR refactor:
the ownership audit across the public Schedule surface and both pipelines,
the scheduled-LoopIR representation decision, the affine-split extension of
the frozen production schema, the pure typed loop-reorder and affine-tiling
passes consuming the existing verified `LoopPlan`, the scheduled target
emission with legacy byte parity, and one real strangler entry for migrated
explicit schedules through `CompileOptions`/`CompilationContext`.  It builds
on the Phase-5 review (`COMPILER_IR_REFACTOR_PHASE5_REVIEW.md`, including
its §8 corrections) and leaves the closed Phase-3.5/4/5 reviews unchanged.

## Verdict

**The Phase-6 explicit-schedule vertical milestone is complete for the
migrated schedule families**, with every applicable mandatory gate green:

- explicit loop reorders and affine `accum="direct"` tiles are represented
  as verified structural LoopIR, applied by pure typed passes, and carried
  end to end — schedule application, target lowering, C++ generation,
  compiled execution — for dense elementwise/reduction/matmul and
  CSR-by-dense SpMM programs;
- for every schedule/program cell of the twenty-member scheduled parity
  grid, the LoopIR pipeline's generated C++ is **byte-identical** to the
  production legacy scheduled route (`Scheduler.apply_schedule` followed by
  the legacy lowering of the verified `ScheduledCIN`);
- compiled shadow execution runs both pipelines on real tensors with
  **bitwise-equal** dense results and PyTorch agreement, across ragged,
  exact, oversized, and zero tile extents and empty CSR rows;
- the production oracle executes scheduled programs directly, and exact
  integer-float counting differentials prove every transformed iteration
  point is visited exactly once, with `erase_schedule` verifying the
  semantics-preserving erasure back to the unscheduled program;
- every unsupported schedule family fails closed with a stable code at the
  schedule-application boundary — nothing silently ignores a requested
  schedule — and default production stays on the legacy route untouched;
- fresh 20-source corpus and 42-source grid captures from the working tree
  are byte-identical to detached `8b0955c` captures, which are in turn
  byte-identical to the sealed Phase-5 §8 captures: **no legacy emission
  changed**, so the byte waiver applies and no runtime kernel benchmark is
  required; a fresh paired compiler-latency run is inside the 1.10 target
  in every category regardless.

The Phase-6 workspace/parallel-annotation and selector-adaptation stretch
families were deliberately not started (§7).

## 1. Ownership audit and the representation decision

The audit walked `Schedule`/`TileSpec`/`RelayoutSpec`, `Scheduler`
(`apply_schedule`, `_apply_schedule_legacy`, `add_tile`,
`_placement_depth`), `schedule_lowerer.py`, normalized CIN, `LoopPlan` and
`loop_plan_legality`, the legacy replay adapter
(`legacy_cin_working_copy`), `CINLowerer`/`iter_lattice` tiled emission,
`CompilationContext` stage identity, kernel cache identity, and the
production oracle.  Dispositions:

| Responsibility | Owner after this milestone |
| --- | --- |
| Public `Schedule` validation and Schedule→`LoopPlan` translation | `Scheduler.apply_schedule`, unchanged — the **shared boundary of both pipelines**.  The LoopIR path consumes its verified `LoopPlan`; the pipelines diverge only downstream (legacy replays CIN tree surgery, LoopIR applies typed passes). |
| Schedule facts artifact | The existing verified `LoopPlan` — already immutable, identity-based (`IndexId` + `LoopRef(part)`), and complete for these families.  It was **not** widened: loop order, affine tile width/placement/unroll/accumulation, and provenance were already facts it owns. |
| Scheduled representation | **Structured schedule nodes in the same LoopIR node model** — one `TileOuterFor`/`TileInnerFor` pair per split, linked by an artifact-local `TileId` — not a sidecar and not a second schema, per the design decision that `ScheduledLoopIR` is a verified state of the same node model.  Semantic meaning stays in the nodes (origin iteration plus clamped point iteration with intrinsic ragged-tail coverage); the only schedule-preference field carried is the target-independent `unroll` hint. |
| Schedule application | **New** `loopir/schedule_passes.py`: `reorder_loops` and `apply_affine_tile` as pure typed passes; `apply_schedule_plan` consumes each plan decision exactly once and returns a frozen `ScheduledLoopIR` artifact retaining the unscheduled base program, the exact plan, the scheduled program, and per-loop `(TileId, IndexId, LoopPart)` provenance. |
| Point/outer-loop identity | `(IndexId, LoopPart)` — the same identity scheme `LoopPlan.LoopRef` already uses — plus the owning `TileId`; no name-derived identity anywhere. |
| Scheduled verification | `loopir/verifier.py` remains the single fail-closed authority; six new stable codes (§3). |
| Scheduled target emission | `loopir/lower_llir.py` mirrors the legacy tiled emission statement-for-statement (width constants, stepping origin loop, reconstructed logical coordinate, overshoot break, input-then-result bound resolution) and reuses the untouched managed pass pipeline and parallel policy, which is why prefetch, pointer hoisting, zero-fill, the nnz-aware row policy, and the ceil-trip-count parallel headers all match legacy byte-for-byte. |
| Strangler entry | `loopir/pipeline.py` (test/debug only; production never imports it): `CompileOptions.requested_schedule` now routes the LoopIR pipeline through the shared scheduler boundary and the typed passes, with a new appended `LOOPIR_SCHEDULE_APPLICATION` stage owning partial failure.  Runtime mode-order alignment targets the plan's logical order (`_plan_mode_orders_to_planned_order`), the scheduled twin of the legacy nest-order helper. |
| Cache identity | Unchanged and source-derived.  A schedule affects the generated source; identical source is the identical kernel artifact — exactly the legacy contract.  Canonical LoopIR dumps (schema v3) remain semantic fingerprints, not target cache keys; no plan- or LoopIR-level artifact is cached. |
| Oracle | `loopir/oracle.py` executes `TileOuterFor`/`TileInnerFor` directly under their intrinsic semantics; `erase_schedule` provides the verified erasure direction. |

## 2. What was frozen (schema extension)

New in `loopir/nodes.py`: `TileId` (builder-allocated, artifact-local) and
the statement pair `TileOuterFor(tile, index, dimension, width, body)` /
`TileInnerFor(tile, index, dimension, width, unroll, body)`.  Semantics are
intrinsic to the nodes: the origin loop iterates tile origins
`0, width, 2·width, …` strictly below the dimension extent and binds no
readable coordinate; the point loop executes inside its origin loop's scope
and binds the logical coordinate over
`origin … min(origin + width, extent) − 1` — ragged-tail coverage is node
semantics, not an emission detail, so every coordinate is visited exactly
once across the pair.  `index` records the split logical loop (schedule
provenance); the pair must agree on index, dimension, and width.

`LoopIRBuilder` gains `new_tile_id`, the two constructors, and
`LoopIRBuilder.resuming(program)`: a deterministic continuation allocator
scanning the stored identity values (never object addresses or allocation
history) so pure passes can rebuild changed paths without identity
collisions.

Deliberately **not** declared: workspace/accumulator nodes (stack/heap
accumulation), sparse coordinate windows (panel tiling), operand staging,
parallel nodes, and every other Phase-6 deliverable family listed in the
design — each remains fail-closed at the plan gate (§4).

## 3. Verifier surface

Six stable codes were added, each with direct adversarial regressions:
`invalid_tile_id`, `duplicate_tile_id` (one origin loop per `TileId`),
`unbound_tile` (a point loop needs its dominating origin loop in scope),
`tile_binding_mismatch` (pair agreement on index/dimension/width),
`invalid_tile_width` (positive exact ints; `bool` and floats rejected), and
`tile_index_conflict` (a split owns its logical loop: the index may be
neither bound nor split again in an enclosing scope).  The point loop's
coordinate binding participates in the ordinary `duplicate_index_binding`
discipline, tile dimensions need a tensor-mapped extent source
(`unresolved_dimension`), and the existing cycle/aliasing/depth guards
cover the new nodes.  The 50-code surface is locked by the source-scan
test.  Canonical serialization moved to schema
`scorch.loopir.canonical.v3` (the serialized contract gained the two node
kinds and the tile identity family); dumps remain deterministic under
registry permutation and disjoint global-identity histories, and
distinguish width, unroll, and split-vs-unsplit structure.

## 4. The migrated schedule families and their boundaries

**Migrated (compiled, byte-identical to the legacy scheduled route,
f32/f64):**

- explicit complete loop orders (`Schedule.loop_order`), applied by
  `reorder_loops` — including orders that repair a source nest the dense
  family's storage-order check would otherwise reject (the check now
  validates against the planned order on the scheduled path; the target
  boundary re-enforces it on the program it actually emits);
- affine `accum="direct"` tiles at every placement kind — `outermost`,
  `child_of:<loop>` (logical or derived `_out`/`_in` parts), `at_depth:<n>`
  — with `unroll` on/off, one or several splits per program (applied in
  plan tile order, exactly legacy sequencing), over dense
  elementwise/reduction/matmul rows and columns and over CSR SpMM rows
  (tile-i) and dense free columns (tile-k), including the broadcast
  coordinate case bounded by the result dimension.

**Fail-closed at the schedule-application boundary** (stable
`SchedulePassError` codes; nothing is silently dropped):
`unsupported_schedule_provenance` (auto/tuned/fallback plans — default
production and empty schedules stay on legacy),
`unsupported_schedule_panel` (sparse coordinate windows),
`unsupported_schedule_relayout` (operand staging),
`unsupported_schedule_result_tile` (heap accumulation),
`unsupported_schedule_parallel` (explicit parallel-loop/tile selection),
`unsupported_schedule_accumulation` (stack/heap affine tiles).
Pass-level legality codes: `reorder_incomplete_order`,
`reorder_sparse_dependency` (cursor parent chains must stay dominated),
`reorder_ordered_assembly` (append nests pin their order),
`reorder_split_chain`, `unsupported_schedule_shape`,
`tile_target_missing`, `tile_target_not_dense` (no windowed compressed
iteration), `tile_target_already_split`, `tile_invalid_placement`
(including origin-must-dominate-point), and `tile_invalid_width`.

**Fail-closed at the target boundary:** affine splits over merged
iteration or ordered sparse assembly (`unsupported_program_shape`) —
verifier-legal and oracle-executable, but outside the emitted families,
the same strangler discipline as Phases 4/5.

**Legality model.**  The passes own semantic legality: binding dependence
(sparse parent dominance), append-order preservation, unique split
ownership, complete orders, and placement dominance are checked before
rebuilding, and the fail-closed verifier runs on input and output of every
pass.  ADD-reduction reassociation is the migrated family's explicit
contract (`StoreReduce` ADD into a zero-initialized output) — the same
freedom the legacy scheduler exercises — and the passes may therefore
split a reduction loop, which the oracle differentials verify even though
the legacy Schedule adapter refuses to request it (recorded asymmetry, not
a defect).  Per-tensor storage-order/nest-order compatibility is
deliberately an *emission* constraint owned by the target lowering
(`unsupported_loop_order`), matching where the legacy pipeline's
equivalent boundary lives.

## 5. Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase6-scheduled-c263b24/`.

Commits (stacked on `8b0955c`; nothing amended, reordered, or pushed):

- `1418e88` — `feat(compiler): extend LoopIR with the affine split schedule subset`
- `b4069ea` — `feat(compiler): apply verified LoopPlans as typed LoopIR scheduling passes`
- `6a9f68c` — `feat(compiler): drive scheduled LoopIR end to end with legacy byte parity`
- `28e1fbf` — `test(compiler): lock the Phase-6 scheduled vertical slice`
- `c263b24` — `test(compiler): move the pipeline requested-schedule boundary lock`
  (the Phase-4 wholesale-rejection lock now proves the moved boundary: an
  identity-order schedule reproduces the unscheduled source byte-for-byte,
  illegal orders fail closed at the shared scheduler boundary, and the
  legacy-accepted dense-ijk erratum order fails closed with
  `unsupported_loop_order`)
- (docs commit follows this review)

Receipts:

- scheduled byte-parity grid: **twenty schedule/program cells** generate
  C++ byte-identical to `Scheduler.apply_schedule` + legacy lowering —
  identity and real reorders (including a reorder from an
  out-of-dense-family source order, f64), matmul tile-j at N∈{2,4,6} for
  width 4 with f32/f64/unroll/`child_of`/`at_depth`/two-splits/zero-extent
  variants, SpMM untiled-scheduled plus tile-k at N∈{2,4,6}, tile-k f64,
  tile-i ragged rows, zero rows, and the result-bounded broadcast split;
- compiled shadow execution (both pipelines, real kernels): matmul ragged
  tile, reordered matmul, SpMM tile-k across all three tile regimes with
  an empty CSR row, and SpMM tile-i — all **bitwise-equal** to the legacy
  scheduled kernels and PyTorch-close; f64 tiled matmul and randomized
  scheduled dimensions execute against PyTorch and the production oracle;
- focused production LoopIR membership: **377 passed + 4 neutrality**
  (102 verifier, 18 printer, 38 oracle, 16 level-storage, 16
  iteration-domain, 37 CIN lowering, 26 schedule passes, 57 LLIR
  lowering/parity, 28 pipeline execution, 39 scheduled slice) — 97 tests
  beyond the corrected Phase-5 membership — including every new defect
  code with forged/cyclic/conflict adversarial coverage, printer goldens
  and canonical-stability locks for scheduled programs, oracle
  exactly-once counting and randomized differentials, pass
  purity/determinism/no-op contracts, erasure round-trips, plan-gate
  fail-closed coverage, stage-sequence and stage-failure ownership, and
  structural activation (width constants, stepping origin loops,
  reconstructed coordinates, overshoot guards, `#pragma unroll`, the
  ceil-trip-count parallel policy, the nnz-aware tile-i row policy, and
  prefetch survival) asserted directly — never waived;
- adjacency green: the combined focused sweep of the LoopIR, spike
  (verifier/execution/neutrality untouched), Schedule
  API/Scheduler, stage-timing (98, with the appended stage identity and
  every legacy sequence lock unchanged), LLIR traversal/pass-manager,
  raw-string-budget, native-ABI, value-object, LoopPlan, and CIN-analysis
  suites: **1,832 passed** at the final code state;
- byte gates: fresh 20-source corpus and 42-source grid captures from the
  working tree are **byte-identical** to detached `8b0955c` captures
  (`diff -qr` empty), and those base captures are byte-identical to the
  sealed Phase-5 §8 candidate captures, chaining to the Phase-3 finals.
  No legacy emission changed; the byte waiver applies and no runtime
  kernel benchmark is required.  The Phase-5 §8 first-run per-cell
  failures and their reviewed assembly/measurement exception remain the
  permanent record and were not rerun;
- compiler latency: a fresh paired base(`8b0955c`)/candidate run of the
  four-category corpus is inside the 1.10 target everywhere — p50 ratios
  `1.003/1.005/1.006/0.999`, p95 ratios `0.985/0.999/1.000/0.990`
  (small-dense, reduction, CSR-intersection, sparse-union).  The legacy
  measured path itself is unchanged (the appended stage identity is inert
  on it, the Phase-4 precedent);
- static parity: Black clean over the changed files; Flake8 clean over the
  same; focused mypy over all thirteen package modules succeeds;
  full-source `mypy --check-untyped-defs src` reports exactly the 146
  inherited errors in the same 12 files as the Phase-5 baseline, **zero**
  in `loopir/`; `git diff --check` clean;
- the authoritative clean detached-worktree non-performance suite at the
  exact final code commit, with isolated pytest/Torch-extension/cache
  directories and import provenance asserted (including
  `schedule_passes`): **3,590 passed, 14 skipped, 3 perf-marked deselections, one known warning, and zero failures/errors in 2,234.79 seconds** — exactly the Phase-5 corrected baseline of 3,505 selected tests plus the 85 new Phase-6 tests (JUnit: 3,604 selected, SHA-256 `cb15c37a8a4e3f81fee23f8857f4ee989d07f982441db8bf75baff4671f6c56c`);
- the five protected tracked files retain their recorded SHA-256 values;
  staging used explicit pathspecs only; no GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, or tooling material was touched; origin
  `refactor/compiler-ir-phase3-std-move-call` remains at `58e8565` and
  nothing was pushed.

## 6. Legacy observations recorded by the differential work

1. **The broadcast-coordinate tile guard resolves from the result.**  The
   legacy `_tile_bound_var` walks the lattice's dense accesses, which
   include the result access exactly when the coordinate is
   broadcast-only, so a tiled broadcast dimension is bounded and guarded
   by the result's size.  This is correct behavior (an earlier working
   hypothesis that legacy emits an unguarded overshooting tile was
   disproved by direct capture); the LoopIR target mirrors the shared
   input-then-result bound policy and the parity grid locks the case.
2. **Legacy refuses affine reduction tiles even for direct accumulation.**
   `_apply_schedule_legacy` rejects every affine tile of a reduction
   variable ("requires an accumulator spanning outer tiles"), although
   direct `+=` accumulation makes the split semantically legal under the
   family's ADD-reassociation contract; the typed pass accepts it and the
   oracle proves it, but no legacy comparand exists, so the strangler
   entry cannot receive such a plan (the shared adapter rejects it first).
   Recorded as a legacy capability boundary, not silently widened.
3. The Phase-4 and Phase-5 errata stand unchanged; none was silently
   fixed or reproduced outside the byte-parity contract.

## 7. Limitations and next-milestone gates

- The migrated schedule surface is exactly §4: explicit orders plus
  direct-accumulation affine splits.  Stack/heap accumulation (workspace
  materialization), sparse panel windows, operand relayout/staging, heap
  result tiles, and explicit parallel selection remain on the legacy
  route, fail closed with stable codes.  These are the natural next
  Phase-6 slices — workspace materialization first, since stack
  accumulation is the schedule the production tile-j/regblock families
  actually use, and it requires the workspace node family the schema still
  deliberately lacks.
- The strangler entry lives in the test/debug pipeline module; production
  dispatch (`ops.einsum`) still routes every schedule through the legacy
  path, and release JIT never enters the LoopIR stages.  Import
  neutrality is unchanged and subprocess-enforced.
- `reorder_loops` operates on unsplit chains (reorder-then-tile is the
  canonical order the plan encodes); reordering an already-split chain
  fails closed rather than guessing tile-loop mobility.
- The scheduled families were byte-compared against legacy for f32 and
  f64; the Phase-5 narrowing of general sparse f64 claims to committed
  cases still stands (the scheduled SpMM f64 parity/execution evidence
  here is fresh and its own record).
- Canonical dumps (schema v3) remain semantic fingerprints that omit
  display names; kernel caching remains source-derived; the Phase-4
  caution that dumps are not target cache keys stands.

## 8. Next broad milestone

Migrate the workspace/accumulation schedule family: declare the workspace
node family in the schema (allocation, reset, producer/consumer regions
with verified lifetime), extend `apply_affine_tile` to stack accumulation
(the `wksp[kTile]` producer/consumer pair legacy emits), byte-compare
against the legacy stack-tile kernels (the `spmm-tilek-stack` shape
captured during this milestone's exploration), and only then approach the
panel/relayout families that `schedule_lowerer.py` owns.  Public
Schedule-adapter cutover, selector integration, and legacy deletion remain
out of scope until those families close.
