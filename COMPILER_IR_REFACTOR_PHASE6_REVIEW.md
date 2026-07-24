# Phase 6 Review: Scheduling Migration to LoopIR (Explicit Reorder + Affine Tiling)

Date: 2026-07-23 (America/Los_Angeles); §10 (the workspace/stack milestone)
and §11 (its review) added later the same day; §12 (the panel milestone)
added 2026-07-24.

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
| Scheduled verification | `loopir/verifier.py` verifies the semantic program, including complete affine pairs; `verify_scheduled_loopir` separately verifies the schedule carrier by deterministic replay and exact provenance (§3 and §9). |
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

Seven stable semantic-program codes were added, each with direct adversarial
regressions:
`invalid_tile_id`, `duplicate_tile_id` (one origin loop per `TileId`),
`unbound_tile` (a point loop needs its dominating origin loop in scope),
`missing_tile_inner` (every origin must contain its point loop),
`tile_binding_mismatch` (pair agreement on index/dimension/width),
`invalid_tile_width` (positive exact ints; `bool` and floats rejected), and
`tile_index_conflict` (a split owns its logical loop: the index may be
neither bound nor split again in an enclosing scope).  The point loop's
coordinate binding participates in the ordinary `duplicate_index_binding`
discipline, tile dimensions need a tensor-mapped extent source
(`unresolved_dimension`), and the existing cycle/aliasing/depth guards
cover the new nodes.  The 51-code surface is locked by the source-scan
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
`reorder_invalid_order`, `invalid_schedule_tile`, `invalid_schedule_plan`,
`reorder_sparse_dependency` (cursor parent chains must stay dominated),
`reorder_ordered_assembly` (append nests pin their order),
`reorder_split_chain`, `unsupported_schedule_shape`,
`tile_target_missing`, `tile_target_not_logical`, `tile_target_not_dense`
(no windowed compressed iteration), `tile_target_already_split`,
`tile_invalid_placement`
(including origin-must-dominate-point), and `tile_invalid_width`.

The frozen `ScheduledLoopIR` carrier has its own fail-closed cross-field
verification: `invalid_scheduled_artifact`, `scheduled_base_not_unscheduled`,
`scheduled_program_mismatch`, and `scheduled_provenance_mismatch`.  The
verifier replays the retained plan from the retained unscheduled base and
requires exact equality with both the stored result and its owned provenance;
the carrier is not trusted merely because its individual programs verify.

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

Original milestone receipts at `c263b24` (the independent corrections and
their final receipts are recorded in §9):

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

Complete a substantial remaining Phase-6 memory-scheduling family, not a
schema-only seam.  The required first vertical slice is workspace
materialization plus stack accumulation: allocation, reset, producer and
consumer regions, verified lifetime/reduction semantics, the
`wksp[kTile]` legacy shape, oracle execution, scheduled target lowering,
and compiled byte/numeric parity.  Once that foundation is green, the same
session should continue through the largest coherent adjacent family that
fits it — preferably sparse-panel plus operand-relayout/staging together,
because they share scope and memory-region lifetime — with heap result
tiles or explicit parallel selection as a separately committed stretch.
Public adapter cutover, selector integration, Phase 7 target work, and
legacy deletion remain out of scope until the remaining Phase-6 plan
families are genuinely closed.

## 9. Independent review corrections (2026-07-23)

An independent adversarial review of `1418e88..727d55c` found concrete
contract failures that the original milestone gates did not expose.  They
are fixed in two stacked local commits:

- `e338e98` — `fix(compiler): verify scheduled LoopIR artifacts`
- `e00facd` — `fix(compiler): preserve scheduled runtime contracts`

Nothing was amended, reordered, or pushed.  Both the remote-tracking ref
and live `git ls-remote` remained at `58e8565`; the code/test tip was 27
commits ahead before this docs commit.

### 9.1 Findings and corrections

1. **An affine origin could verify without its point loop.**  The semantic
   verifier checked a point loop's dominating origin but not the reverse.
   `missing_tile_inner` now requires every `TileOuterFor` to contain its
   matching `TileInnerFor`; the existing global index-binding invariant
   independently rejects a second point loop.
2. **The scheduled carrier was frozen but not cross-field verified.**
   A caller could combine a valid base, plan, scheduled program, and
   provenance that did not describe one another, and non-logical tile
   parts or malformed placement state could reach direct pass logic.
   `verify_scheduled_loopir` now checks exact owned carrier state, verifies
   both programs, requires an unsplit base, deterministically reapplies the
   structurally validated plan, and compares the stored program and
   provenance exactly.  The exported reorder/tile passes now reject
   malformed requests with stable diagnostics rather than leaking
   `AttributeError` or container `TypeError`.
3. **Oversized tile widths silently crossed a narrower target boundary.**
   Both CPU routes emit the width as `constexpr int`; on the supported
   compiler, `2**31` becomes a negative value.  Public `TileSpec` and
   `RelayoutSpec`, verified `LoopPlan`, the typed schedule pass, and the
   C++ target now enforce `width <= 2**31 - 1`.  Semantic LoopIR and its
   oracle deliberately continue to accept arbitrary positive Python ints;
   target representability is not misrepresented as target-independent
   semantics.
4. **Forged non-schema object state affected continuation identities.**
   `LoopIRBuilder.resuming` previously scanned every `__dict__` value even
   though verification and canonical serialization recognize declared
   dataclass fields only.  It now scans exactly the schema fields, so a
   non-semantic extra `TileId` cannot perturb deterministic allocation.
5. **Tile-only legacy replay had two opposite failure modes.**  A public
   `Schedule(loop_order=None, tiles=...)` materializes an explicit loop
   order in its verified plan and was then rejected as conflicting during
   private legacy replay.  The adapter now canonicalizes exactly that
   proven-equivalent omission.  It does not overwrite an explicitly
   conflicting requested schedule; that still raises
   `conflicting_schedule`.
6. **Scheduled runtime layout prerequisites were not isolated or owned
   correctly.**  A requested schedule could leak into an auxiliary
   mode-order-relayout compilation, while moving the work to a discarded
   child context initially left its elapsed time unowned and its failures
   non-terminal for the caller context.  Relayout now receives a
   schedule-free options snapshot and independent child context while the
   caller's frontend-binding stage remains active around it, charging the
   prerequisite time and retiring the parent compilation on failure.
7. **Scheduled shadow execution did not compare equivalent runtime
   layouts.**  The legacy route consumed nonidentity physical layouts
   without plan alignment, the result wrapper was hard-coded to rank-two
   `"dd"`, and a tile-only shadow could reselect policy after compiler-owned
   mode-order metadata changed.  Both routes now align to the verified
   plan, dense result wrapping derives format/rank from CIN, and policy
   shorthand is frozen to the first verified plan before any sub-run.
8. **Several broad evidence claims were source-only.**  Committed compiled
   shadows now cover `child_of`, `at_depth`, unroll, two splits, non-square
   nonidentity layouts, rank-one dense results, f64 CSR×dense SpMM with
   double-precision tolerances, and zero logical/result extents under a
   tiled loop.  The stale reference to a legacy unguarded-broadcast erratum
   was removed; direct capture had already disproved it.

### 9.2 Corrected verification

- Contract focus: **309 passed** across the semantic verifier, scheduling
  passes, LoopIR target, LoopPlan, and Schedule API.
- Full scheduled runtime/pipeline files: **80 passed**, including the
  expanded compiled matrix and injected relayout-failure ownership.
- Authoritative clean detached-worktree non-performance suite at
  `e00facd`, with import provenance asserted and isolated caches:
  **3,617 passed, 14 skipped, 3 perf-marked deselections, one known warning,
  and zero failures/errors in 2,427.96 seconds**.  This is the original
  Phase-6 total plus exactly the 27 correction regressions.
- Black and `git diff --check` are clean.  Focused mypy is clean on seven
  changed package modules; the eighth (`scheduler.py`) retains exactly its
  two inherited `import-untyped` findings.  Flake8 has no new finding versus
  `727d55c` (the scheduler's existing C901 remains byte-for-byte inherited).
  Full-source mypy remains at the exact Phase-6 baseline:
  **146 inherited findings in 12 files and zero in `loopir/`**.
- The five protected tracked files retain the recorded SHA-256 values and
  were never staged; unrelated dirty/untracked GPU, CUDA, benchmark,
  packaging, scheduler, research, scratchpad, and tooling material remains
  untouched.

The same clean-suite receipt is recorded in the handoff section committed
with this correction.

## 10. Workspace milestone: stack accumulation on LoopIR (2026-07-23)

The second Phase-6 milestone migrates the stack-accumulation (workspace)
schedule family — the schedule the production tile-j/regblock families
actually use — end to end: schema, verifier, printer/serialization,
oracle, typed pass, target lowering with byte parity, and a compiled
matrix.  It is recorded in four stacked local commits plus the docs
commit that records this section; nothing earlier was amended or
reordered and nothing was pushed:

- `9f431d9` — `feat(compiler): extend LoopIR with the stack workspace region schema`
- `986c773` — `feat(compiler): materialize stack tiles as a typed workspace pass`
- `ba87329` — `feat(compiler): lower workspace regions with legacy byte parity`
- `72e991e` — `test(compiler): lock the Phase-6 workspace vertical slice`

### 10.1 Audit and representation decision

The audit (recorded before nodes were chosen, with captured legacy golden
sources for every emission fact) walked the remaining LoopPlan families —
stack/heap accumulation, panel bounds, operand relayout, result tiles,
explicit parallel selection — against normalized CIN, the legacy owners
(`Scheduler._apply_schedule_legacy` / `insert_workspace` / `add_tile` /
`CINLowerer.lower_Where` / `lower_ConsumerIndexStmt` /
`schedule_lowerer.py`), target emission, stage and cache identity, and
the oracle:

| Plan family | Legacy owner (validate → transform → emit) | Lifetime in emitted code | Disposition |
| --- | --- | --- | --- |
| Stack accumulation (`LoopTile.accumulation="stack"` — the verified plan already carried the fact; it was not widened) | `_tile_target_needs_workspace` + `_validate_stack_workspace_scope` → `insert_workspace` (a `Where{producer, consumer}` rooted at the *last* reduction ForAll with a dense 1-D `wksp`) → `add_tile` (rebinds both branch binders to the point variable; origin inserted at the placement resolved against the loop prefix *above* the Where) → `lower_Where` tiled-dense path (`float wksp[kTile] = {};`, empty parallel-workspace cluster) + producer `wksp[k_in] += value` + the synthesized result-bounded consumer loop | allocation + zero-reset per innermost-prefix-loop iteration by declaration-with-initializer; producer writes it, consumer copies out, gone at block end; no heap traffic and no per-thread pool | **migrated this milestone** |
| Heap accumulation | the result-tile machinery in `schedule_lowerer.py` | outer-tile entry/exit heap buffer | fail-closed (`unsupported_schedule_accumulation` at the direct passes, `unsupported_schedule_result_tile` at the plan gate), unchanged |
| Untiled/auto dense workspace | `should_insert_workspace` on the auto path; per-thread pool-owner slice + per-row memset | per-thread heap pool | not an explicit-plan fact; auto plans already fail closed (`unsupported_schedule_provenance`) |
| Sparse (COO-hash) workspace | the `coo_workspace` heap path of `lower_Where` | loop-scoped heap container | out of family: stack requires the dense-output reduce leaf (`stack_tile_target_invalid`) |
| Panel bounds / operand relayout | `schedule_lowerer.py` post-LLIR completion by rendered-name discovery | coordinate window / staging buffer at the scope loop | fail-closed, unchanged; §10.5 records the audit verdict |
| Result tiles / explicit parallel selection | `schedule_lowerer.py` / `_set_explicit_parallel_loop` | — | fail-closed, unchanged |

Representation: **a structured region in the same LoopIR node model** —
exactly the design's Stage-5 workspace materialization ("allocation,
reset, producer, and consumer regions with verified lifetime and
reduction semantics").  `WorkspaceRegion(workspace, producer, consumer)`
owns a `WorkspaceDecl(workspace: WorkspaceId, name, dtype, tile: TileId)`
whose extent is intrinsic: the region buffers the point domain of one
affine split, allocation and zero-reset (ADD's identity) are intrinsic
region-entry semantics, teardown is region exit, and
`WorkspaceReduce`/`WorkspaceRead` address cells by the owning split's
point coordinate.  No C++ names, callbacks, dynamic fields, or target
strings anywhere; dimension-extent, multi-dimensional, and sparse
workspace forms are deliberately not declared.

One boundary was deliberately moved: the region's producer and consumer
each bind the split's point coordinate once, in disjoint sibling scopes,
so a split may now own *several* ``TileInnerFor`` bindings of the same
``TileId`` (each iterates the same clamped window exactly once).  Nested
rebinding and every other binder kind keep the global once-only
``duplicate_index_binding`` rule, and the previously locked
nested-second-point regression still passes unchanged.

### 10.2 What was frozen (schema extension)

New in `loopir/nodes.py`: `WorkspaceId` (builder-allocated,
artifact-local, scanned by the declared-field-only `resuming`
continuation), `WorkspaceDecl`, `WorkspaceRegion`, `WorkspaceRead`, and
`WorkspaceReduce`.  Canonical serialization moved to schema
`scorch.loopir.canonical.v4` (the three statement/expression kinds plus
the workspace identity family; workspace display names are omitted like
every other display name).  The printer renders regions with explicit
producer/consumer roles.  The oracle executes regions under the intrinsic
semantics — fresh zeroed cells per region execution, cell = coordinate −
current origin with hard bounds, teardown on exit — and fails closed at
runtime on access outside a region or its origin loop.

### 10.3 Verifier surface

Nine stable codes were added, each with direct adversarial regressions:
`invalid_workspace_id`, `duplicate_workspace_id`, `unbound_workspace`,
`workspace_scope_mismatch` (a region needs its tile's origin loop in
scope and must open outside its own point loops),
`workspace_write_scope`, `workspace_read_scope`,
`workspace_coord_mismatch` (cells are addressed only by the owning
split's point coordinate bound inside the region),
`workspace_output_write` (producers never write declared outputs), and
`workspace_dead_region` (a region must accumulate and copy out).  The
60-code surface is locked by the source-scan test.  Workspace dtypes join
the uniform-dtype discipline (`mixed_dtype`) and the existing
cycle/aliasing/forged-state/depth guards cover the new nodes.

### 10.4 Passes, lowering, and the compiled matrix

`apply_stack_tile` is the new pure typed pass: one fused rebuild
mirroring the legacy `insert_workspace` + `add_tile` composition —
producer = the reduction chain from the last reduction loop with the
point loop reducing into the workspace; consumer = the copy-out point
loop; origin inserted at the placement resolved against the loops that
remain above the region (the legacy prefix-of-`Where` rule, including
its `0..len(prefix)` `at_depth` range and prefix-only `child_of`
parents).  The legacy legality boundary is mirrored with stable codes:
`stack_tile_target_invalid` (the target must be the single trailing
dense free loop after the last reduction of a dense-output ADD
reduction; a region-terminated chain refuses further stack tiles) and
`stack_tile_root_scope` (the region may not replace the chain root).
`apply_schedule_plan` consumes `accum="stack"` tiles exactly once
through the pass (`heap` still fails closed at the plan gate), direct
tiles compose with region-terminated chains, the carrier verifier
replays stack plans and rejects region-carrying base programs,
provenance lists prefix → producer → consumer chains in the documented
execution order, and `erase_schedule` erases a region to its
direct-accumulation equivalent (defined for the exact copy-out form the
pass produces; other verified consumers fail closed).  Erasure equality
is proven by canonical dump against the reordered base and by exact
integer-float oracle differentials across ragged, exact, oversized,
non-dividing, unit, and zero extents — counting differentials that
prove reset, lifetime, ragged-tail, and reduction semantics, not merely
loop visitation (a missing per-tile reset, a double copy-out, or a
tail overshoot each change the counted result).

The target lowering emits the region byte-for-byte as legacy does:
`// Initialize workspaces` + `float|double wksp[kTile_<name>] = {};`
(allocation and reset in one `FixedStackArrayDecl` inside the innermost
prefix loop), the producer chain with the input-bounded point loop and
the untyped `wksp[k_in] += <value>;` leaf, then the synthesized
`// Lower consumer CIN` copy-out loop with the consumer's own int64
resolve spelling, the *result*-bounded overshoot break, result-write
access metadata, and the tile's unroll preference on both point loops.
Result positions driven by the split coordinate resolve only in the
consumer — exactly where the legacy consumer lowering resolves them —
so prefetch, pointer hoisting, zero-fill, and both parallel policies
(the ceil-trip-count origin form and the nnz-aware row form) reproduce
byte-for-byte through the untouched managed passes.  The workspace
display name joins the target's identifier-safety and name-collision
checks; region shapes outside the family (bare-point producers,
non-copy-out consumers, merged producers, foreign point loops) fail
closed with `unsupported_program_shape`.

**The compiled matrix.**  A thirteen-member stack byte-parity grid locks
generated C++ equality against `Scheduler.apply_schedule` + the legacy
lowering in every cell: CSR SpMM stack tile-k at N below/equal to/above/
not dividing the width, unroll, `child_of`, `at_depth`, f64, zero free
extent, zero rows; dense matmul stack tile-j f32/f64; and the direct-i
plus stack-k two-split.  Compiled shadow execution (both pipelines, real
kernels) is **bitwise-equal** to the legacy scheduled kernels and
PyTorch-close across the three tile regimes with an empty CSR row, f64
SpMM at 1e-10 tolerances, matmul stack with unroll, the two-split
composition, and zero extents; randomized dimensions execute against
PyTorch and the production oracle.  Structural activation is asserted
directly and never waived: the workspace declaration-with-reset, both
bound spellings (`B1_size` in the producer, `C1_size` in the consumer),
the copy-out statement, prefetch survival in the producer,
`scorch_zero_dense`, `#pragma unroll` on both point loops, the
ceil-trip-count origin policy, and the nnz-aware row policy on the
`child_of` form.

### 10.5 Adjacent-family audit verdict (panels + relayout)

The preferred next family — sparse-panel tiling followed by operand
relayout/staging — was audited and deliberately **not** started in this
session.  Legacy completes their lowering *after* LLIR construction
(`schedule_lowerer.py` rewrites emitted position loops and operand
accesses by rendered-name discovery), so typed migration needs three
representations the schema deliberately lacks — a coordinate-window
iteration node over compressed levels (clamped coordinate ranges with
search-derived position bounds), typed access redirection for staged
operands, and pack-loop/staging-buffer lifetime on top of the new region
machinery.

The original milestone wording incorrectly called the two families
inseparable.  They share machinery, but the dependency is one-way:
panel-only schedules are a supported legacy family and
`_apply_panel_tile` runs whether or not relayout exists; relayout, in
contrast, requires exactly one matching panel.  The next milestone may
therefore close and commit a complete panel vertical slice before
layering relayout on it.  Relayout still needs an explicit logical-access
identity decision: production LoopIR `Load` has no occurrence identity,
while the current LLIR target's access map is per tensor symbol and must
not be mistaken for the design's logical access ID.  Both families
remain fail-closed at the plan gate with stable codes; nothing was
silently declined.  Heap result tiles and abstract parallel selection
likewise remain fail-closed stretch families.

### 10.6 Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase6-workspace-72e991e/`.

- both §9 gates were independently reproduced before any edit:
  the 309-test contract focus and the 80-test runtime focus both passed
  at `084ed4c`;
- contract focus after the milestone (same five files): **341 passed** —
  the 32 new contract regressions cover the workspace verifier codes,
  the stack pass surface, and the target boundaries;
- scheduled runtime focus after the milestone: **102 passed** across
  `test_loopir_scheduled_slice.py` (74) and
  `test_loopir_pipeline_execution.py` (28), including the thirteen
  stack parity cells, the compiled stack shadows, and the structural
  activation locks;
- focused production LoopIR membership: **461 passed + 4 neutrality**
  (119 verifier, 22 printer, 42 oracle, 16 level-storage, 16
  iteration-domain, 37 CIN lowering, 45 schedule passes, 62 LLIR
  lowering/parity, 74 scheduled slice, 28 pipeline execution) — the
  full-suite delta against the §9 baseline is exactly the 62 tests this
  milestone adds;
- combined focused adjacency sweep (LoopIR fast suites + spike
  verifier/execution/neutrality + CIN/CIN-analysis + stage timing +
  LLIR pass-manager/string-budget/traversal + LoopPlan + native ABI +
  Schedule API + Scheduler + value-object boundaries): **1,824
  passed**, plus **332 passed** across the cin_lowerer,
  schedule-generality, and tune-scheduler-harness compiled adjacency
  files;
- byte gates: fresh 20-source corpus and 42-source grid captures from
  the working tree are **byte-identical** to detached `084ed4c` captures
  and to the sealed Phase-6 captures (`diff -qr` empty for all
  comparisons) — no legacy emission changed, the byte waiver applies, no
  runtime kernel benchmark is required, and the Phase-5 §8 first-run
  failures and reviewed exception remain the permanent, un-rerun record;
- compiler latency: a fresh paired base(`084ed4c`)/candidate run of the
  four-category corpus is inside the 1.10 target everywhere — p50
  ratios `1.015/0.992/0.981/1.021`, p95 ratios `0.975/0.976/0.957/1.014`
  (small-dense, reduction, CSR-intersection, sparse-union); the release
  JIT path itself never enters the LoopIR stages;
- static parity: Black clean over the package and changed tests; Flake8
  clean over the same; focused `mypy --check-untyped-defs` over all
  twelve package modules succeeds; full-source mypy reports exactly the
  **146 inherited findings in 12 files, zero in `loopir/`**;
  `git diff --check` clean before every commit;
- the authoritative clean detached-worktree non-performance suite at the
  exact final test commit `72e991e`, with isolated
  pytest/Torch-extension/cache directories and import provenance
  asserted: **3,679 passed, 14 skipped, 3 perf-marked deselections,
  one known warning, and zero failures/errors in 2,474.73 seconds** —
  exactly the §9 baseline of 3,617 passed plus the 62 new milestone
  tests (JUnit: 3,693 selected, SHA-256
  `4da4f0ed07e05cc6ebc2af6ede9ce993a54e7eb489dab786b3130647890fb163`);
- the five protected tracked files retain their recorded SHA-256 values
  and were never staged; staging used explicit pathspecs only; no
  GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, or
  tooling material was touched; origin
  `refactor/compiler-ir-phase3-std-move-call` remains at `58e8565`
  (live `git ls-remote` confirmed at session start) and nothing was
  pushed.

### 10.7 Limitations and the candid Phase-6 exit verdict

- The migrated schedule surface is now: explicit complete loop orders,
  affine `accum="direct"` splits, and the stack-accumulation workspace
  family.  Sparse panels, operand relayout/staging, heap result tiles,
  and explicit parallel selection remain fail-closed on the legacy route
  (§10.5).  **Phase 6 is therefore not exited.**  Against the design's
  Phase-6 deliverables: loop reorder, affine tiling/ragged tails, and
  workspace materialization (stack + direct accumulation) are done;
  sparse panel tiling, operand relayout/staging, heap result
  accumulation, abstract parallel-loop selection, and the representative
  tile-j/tile-ijk `LoopPlan` encodings are still open.  What this
  milestone closes is the complete workspace/stack vertical family — the
  memory-region and lifetime foundation the remaining families share.
- The strangler entry remains test/debug-only; production dispatch,
  release JIT, import neutrality, legacy stage sequences, and
  source-derived kernel cache identity are unchanged and
  subprocess-enforced.
- The stack family's legality mirror is the legacy trailing-free-var
  boundary; verifier-legal generalizations outside it (multi-var or
  sparse workspaces, bare-point producers, non-copy-out consumers) stay
  fail-closed at the pass or target boundary, never silently emitted.
- Canonical dumps (schema v4) remain semantic fingerprints that omit
  display names; kernel caching remains source-derived; the Phase-4
  caution that dumps are not target cache keys stands.
- The Phase-4/5 errata and the §6 observations stand unchanged; none
  was silently fixed or reproduced outside the byte-parity contract.

## 11. Independent workspace-milestone review corrections (2026-07-23)

The post-milestone review independently read and probed
`9f431d9..c0ef584` rather than accepting §10's evidence at face value.
The workspace representation, stack transformation, erasure model, and
legacy byte-parity lowering were otherwise sound, but the review found
and fixed five concrete trust-boundary defects:

1. `bd1cb94` rejects target loop-variable name aliasing.  Distinct
   logical `IndexId` binders may legitimately share a `DimensionId`, but
   the current C++ target names binders from the dimension.  Previously
   that could silently shadow one loop and change a workspace reduction;
   target lowering now raises `generated_name_collision` while preserving
   the legal affine outer/inner reuse of one logical index.
2. `bd1cb94` also makes the semantic oracle's workspace storage sparse.
   A verified semantic tile width is any positive integer, so eagerly
   allocating `[0.0] * width` let a valid huge width raise `OverflowError`
   or exhaust memory even when execution touched one cell.  The oracle
   now stores only touched cells with implicit zero, retaining intrinsic
   reset, bounds, and accumulation semantics.
3. `bd1cb94` / `b2371b6` reject executable `int`/`str` subclasses in
   `LoopPlan`, lock artifact-local `TileId`/`WorkspaceId` continuation and
   canonical renumbering, and cover the target/oracle regressions above.
4. `6f86a5b` / `fc163cd` require exact stored field ownership for
   `LoopPlan`, every nested plan carrier and identity, and `ScheduledCIN`.
   Deleting a default-backed field previously let instance lookup fall
   through to the dataclass class default: a stack plan could lose all
   `tiles`, and an `"auto"` plan could become `"explicit"`.  Missing,
   extra, and malformed state now fails before semantic replay or direct
   pass dispatch.
5. `1e55973` / `c423167` and `08b0b6d` / `6ab6a21` require canonical
   enum members, bound untrusted integer diagnostics, and exact
   `ScheduledLoopIR` provenance identities.  Forged exact-type enums,
   deleted identity payloads, extra state, and enormous IDs/depths now
   reach stable compiler diagnostics instead of `AssertionError`,
   `AttributeError`, Python digit-limit `ValueError`, or incidental
   provenance behavior.

The review also corrected §10.5's architectural record: sparse panels
and operand relayout share coordinate-window machinery, but are not
inseparable.  Panel-only schedules are supported independently; relayout
has a one-way dependency on one matching panel.  Relayout must still
decide how a specific logical access occurrence is identified — the
target's existing per-symbol LLIR `AccessId` is not that identity.

### 11.1 Review verification

Final review evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-workspace-review-6ab6a21/`;
the fresh source captures and full-source mypy log produced during the
same logical review are under the preceding
`phase6-workspace-review-c0ef584/` ledger.

- the five-file Phase-6 contract focus passes **352 tests**;
- the compiled scheduled/runtime focus passes **102 tests** in 456.48
  seconds;
- the authoritative clean detached-worktree non-performance suite at
  `6ab6a21` passes **3,692 tests, 14 skipped, 3 perf-marked
  deselections, one known warning, and zero failures/errors in 2,554.39
  seconds**, with isolated caches and import provenance asserted; this is
  exactly the §10 baseline of 3,679 plus the 13 review regressions
  (log SHA-256 `e7ad5dbeeb7503f59ed8fcfb4834b9b07812938af3cefd7860503228f8cf1808`,
  JUnit SHA-256
  `68da0d7ec41e9f6bf8c23c555c9c64068fa9276931a60f1aba667a96f1a75c5b`);
- fresh 20-source corpus and 42-source grid captures remain byte-identical
  to `084ed4c` and the sealed Phase-6 captures; valid workspace C++
  remains byte-identical, while the new malformed cases fail before
  emission;
- paired compiler latency for base `c0ef584` versus candidate `6ab6a21`
  is inside the 1.10 target in every category and percentile: p50 ratios
  `0.994/0.997/1.009/0.985`, p95 ratios
  `1.033/0.969/1.012/1.044` (small-dense, reduction,
  CSR-intersection, sparse-union), with identical source hashes;
- Black, scoped Flake8, focused mypy, and `git diff --check` are clean;
  full-source mypy remains exactly the inherited **146 findings in 12
  files**, with the finding log byte-identical to the review baseline
  apart from `conda run`'s trailing exit wrapper;
- the five protected tracked files retain their recorded hashes, staging
  used explicit pathspecs, no unrelated dirty/untracked material entered
  a commit, origin remains `58e8565`, and nothing was pushed.

### 11.2 Remaining boundaries and verdict

- The C++ compatibility target still accepts a stack tile width up to
  `INT_MAX` and emits a fixed-size stack array.  That mirrors legacy and
  was not narrowed during a byte-parity review, but widths large enough
  to overflow practical thread stacks remain a real policy boundary for
  a future schedule-resource gate.
- Exact carrier validation now covers the plan and scheduled LoopIR
  artifacts reviewed here.  The older normalized-CIN analyzer still
  trusts the internals of exact identity objects during hashing; hostile
  forged CIN identity payloads are a separate CIN-analysis hardening
  slice and are not claimed fixed by this review.
- The Phase-6 exit verdict remains **not exited**.  The next coherent
  order is a complete standalone sparse-panel vertical slice, then
  relayout/staging on that foundation, with heap result tiles and
  abstract parallel selection as separately gated stretch families.

## 12. Panel milestone: sparse coordinate windows on LoopIR (2026-07-24)

The third Phase-6 milestone migrates the sparse-panel schedule family —
the SpMM tile-j coordinate window, the family relayout builds on — end to
end: schema, verifier, printer/serialization, oracle, typed pass, target
lowering with byte parity, and a compiled matrix.  It is recorded in four
stacked local commits plus the docs commit that records this section;
nothing earlier was amended or reordered and nothing was pushed:

- `68d7eb4` — `feat(compiler): extend LoopIR with the sparse panel window schema`
- `cae69dd` — `feat(compiler): apply sparse panel plans as a typed scheduling pass`
- `57fff36` — `feat(compiler): lower panel windows with legacy byte parity`
- `bb97172` — `test(compiler): lock the Phase-6 panel vertical slice`

Both §11 gates were independently reproduced at `a880e86` before any
edit: the five-file contract focus passed **352** and the compiled
runtime focus passed **102** (446.87 s).  The audit, the captured legacy
goldens (nine sources: five panel forms, both relayout staging scopes,
the heap composition, and the unpaneled explicit reference), and the
responsibility/lifetime table classifying every fact as semantic,
scheduled, target-specific, or compatibility-only were recorded under
`/Users/bobby/.cache/scorch-codex/phase6-panel-audit/` *before* any
schema was written.

### 12.1 Audit and representation decision

The audit walked the complete legacy responsibility chain:
`TileSpec(kind="panel")` validation in `_apply_schedule_legacy`
(single panel, listed last, dense result, serial, `accum="direct"`,
exactly one compressed access with a dense CSR parent, the mandatory
`parallel_loop` equal to that dense-parent row var, row before panel var
in logical order, `outermost`/`child_of:<outermost-affine>_out`
placement only), the `PanelBound` fact built from the *first dense
access* containing the panel var and rendered as
``f"{tensor}{level}_size"`` at `materialize_legacy_schedule`, and the
post-LLIR completion in `schedule_lowerer._apply_panel_tile`
(tag-discovered target loop, marked-parallel ancestor requirement,
`match_mode_position_bounds` row-bound re-derivation, `_crd`-name
coordinate-array scan, `lower_bound` window derivation, origin-loop wrap,
top-of-function width constant).  Two pipeline-position facts govern
byte parity and are confirmed by the goldens: the legacy managed passes
run on the *unpaneled, unmarked* function (explicit parallel selection
suppresses the emission-time auto gate;
`CINLowerer._apply_explicit_parallel_schedule` marks the row loop on the
assembled function, before `apply_schedule_to_llir` windows and wraps
it), and pass behavior is decided there — the dense-pointer hoist fires
in the panel-only golden but not in the child_of golden, and the sparse
prefetch guard fires on the canonical loop shape and survives the
windowing only because the end spelling is reused.

Representation: **a structured origin/window pair in the same LoopIR
node model**, exactly the design's Stage-5 sparse-panel-tiling pass
("a coordinate window over structured level/coordinate facts, never
name/regex discovery").  `PanelOuterFor(tile, index, dimension, width,
bound_tensor, bound_level, body)` iterates the clamped window origins of
the compressed coordinate's dimension (the origin is not a readable
coordinate, like the affine origin); `SparseWindowFor(tile, cursor,
position, coord_index, body)` visits, in storage order, exactly the
stored entries whose coordinate falls inside
``[origin, min(origin + width, extent))`` — the clamped coordinate
window is intrinsic node semantics, and how the position sub-range is
found (coordinate search on a canonical sorted segment) is a target
concern with no spelling in the node.  The pair shares the
artifact-local `TileId` space with affine splits.  One deliberate
redundancy: the plan's `PanelBound` fact is materialized structurally as
the panel's `(bound_tensor, bound_level)` DENSE extent source.  It is
semantically redundant with the dimension identity (extent equality is
the `DimensionId` contract) but required for the exact legacy bound
spelling at the target, so the verifier enforces its consistency
(declared tensor, in-rank DENSE level, storing the panel's own
dimension) instead of trusting it.

### 12.2 What was frozen (schema extension)

New in `loopir/nodes.py`: `PanelOuterFor` and `SparseWindowFor`
(builder methods `panel_outer_for`/`sparse_window_for`; the
declared-field-only `resuming` continuation covers them through the
generic field scan).  Canonical serialization moved to schema
`scorch.loopir.canonical.v5` with the two new statement kinds; the
printer renders the pair with its bound (`panel_outer_for s0 x0 in d0
width 3 bound t1@0`); canonical dumps remain stable across unrelated
global identity histories and renumber raw schedule identities.  The
oracle executes the intrinsic window semantics — panel widths are
semantic integers, never allocation requests; a window executed outside
its panel's origin loop fails closed at runtime — and counting
differentials prove each stored entry is visited exactly once across
ragged windows, empty rows, zero extents, and disjoint supports.

### 12.3 Verifier surface

Four stable codes were added, each with direct adversarial regressions:
`unbound_panel` (a window with no dominating open `PanelOuterFor` of its
tile — an affine origin does not open a panel scope),
`missing_panel_window` (an origin whose body never binds its window),
`panel_binding_mismatch` (pair disagreement: the window must bind the
panel's logical index over a cursor level storing the panel's
dimension), and `panel_bound_mismatch` (a declared bound level that is
not DENSE or stores another dimension; undeclared tensors and
out-of-rank levels keep `undefined_tensor`/`rank_mismatch`).  The pair
reuses the shared tile discipline — `duplicate_tile_id` across both
families, `tile_index_conflict` in both directions (a panel owns its
logical loop against enclosing binders and open splits; an affine origin
may not split an index an open panel owns), `invalid_tile_width` — and
the existing cursor, once-only binding, forged-state, hostile-subclass,
cycle, aliasing, and depth guards cover the new nodes.  A `TileInnerFor`
cannot bind a panel's tile (`unbound_tile`), and a second window of one
panel is `duplicate_index_binding`: the sibling-rebinding boundary stays
owned by the workspace point family.  The locked source-scan surface
grows from 60 to **64** codes.

### 12.4 Passes, lowering, and the compiled matrix

`apply_panel_tile(program, tile, bound, parallel_loop)` is the new pure
typed pass, operating only on cursor, level, position, and dimension
identities.  Its legality mirror of the legacy family: the target must
be a `SparseFor` whose cursor's dominating parent is a dense position
over a directly bound row coordinate (compressed-parent windows fail
closed as `panel_nested_compressed`; dense, merged, append-assembly,
region-terminated, and already-split targets keep stable codes); the
plan's `parallel_loop` is mandatory and must name that dense-parent row
loop (`panel_parallel_scope`) — the legacy family admits no other value,
so exact validation is the fact's consumption; placement resolves as in
legacy `_apply_panel_tile` (`outermost` wraps the chain root, `child_of`
wraps the loop below the named affine origin and must stay strictly
above the row loop, `at_depth` is `panel_placement_invalid`); and the
`PanelBound` is consumed by materialization after compatibility checks.
The plan gate admits exactly the legacy panel plan shape — at most one
panel tile, listed last, direct serial accumulation, exactly one
corresponding bound, mandatory logical `parallel_loop`, `child_of`
parents restricted to outermost-placed affine tiles of the same plan
(`invalid_schedule_panel`/`panel_parallel_scope`/
`panel_placement_invalid`) — and `parallel_loop` remains an unmigrated
family (`unsupported_schedule_parallel`) on every plan without a panel.
The chain machinery covers the new nodes end to end: decomposition,
rebuild, loop keys (origin = OUTER, window = INNER, matching the legacy
`_out` rendering), `ScheduledLoopIR` provenance and replay, the
scheduled-base purity check, reorder/affine/stack refusal of
panel-scheduled chains (panels apply last, as in legacy), and
`erase_schedule`, which drops the origin and restores the plain
`SparseFor` under the family's ADD-reassociation contract, proven by
canonical-dump equality and exact oracle counting differentials across
unit/ragged/exact/oversized/maximum widths with an empty CSR row plus
randomized dimensions.

The target lowering reproduces the legacy pipeline position exactly:
`raw_loop_statements` emits the unpaneled nest with marking suppressed,
the untouched managed passes run, the function is assembled, and
`complete_panel` then (1) marks the structurally identified row loop
with the same `mark_first_for_loop_parallel` call and empty cluster,
(2) rewrites the window bounds (`p*_row_end` capture, `lower_bound`
begin/end through the schedule lowerer's shared typed constructor —
one source for that spelling), (3) wraps the origin loop with its
clamped `std::min` end, and (4) prepends the width constant.  Because
every managed pass returns a *detached* tree, emission-object identity
cannot survive to the assembled function; the completion instead
navigates the preserved loop skeleton — depth-first for-loop order is
exactly the emitted chain order — and cross-checks each located loop
against the retained emission records, failing closed as
`panel_completion_lost` on any disagreement.  No `scorch_index_var`
tags, rendered-name scans, or regexes are consulted.  The target
boundary rejects everything outside the migrated shape (one pair only,
dense result, CSR cursor form, row strictly between origin and window)
and panel-derived names join the collision discipline.

**The compiled matrix.**  A twelve-member panel byte-parity grid locks
generated C++ equality against `Scheduler.apply_schedule` + the legacy
lowering in every cell: CSR SpMM panel widths below/equal to/above/not
dividing the coordinate extent, the unit width, the maximum
constexpr-int width (`2^31 - 1`), f64, `child_of` below an outermost
affine pack tile, the panel-outermost-over-affine-outermost composition,
and zero rows / zero panel extent / zero free extent.  Compiled shadow
execution (both pipelines, real kernels) is **bitwise-equal** to the
legacy scheduled kernels and PyTorch-close across the three window
regimes with an empty CSR row, f64 at 1e-10 tolerances, the
affine+panel composition, and a zero free extent; randomized dimensions
execute against PyTorch and the production oracle.  Structural
activation is asserted directly and never waived: the top-of-function
width constant, the serial origin loop over the declared dense bound
(`B0_size`), the clamped `j_out_end`, both `lower_bound`-derived
position bounds, the windowed loop start (`pA1 = pA1_panel_begin`),
exactly one nnz-aware row pragma placed between origin and row loop,
prefetch survival inside the window, and the hoisted operand pointer.

### 12.5 Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase6-panel-bb97172/`
(audit and goldens under `phase6-panel-audit/`).

- both §11 gates were independently reproduced before any edit: the
  352-test contract focus and the 102-test runtime focus both passed at
  `a880e86`;
- contract focus after the milestone (same five files): **386 passed** —
  the 34 new contract regressions cover the panel verifier codes (12),
  the panel pass surface (17), and the target boundaries (5);
- scheduled runtime focus after the milestone: **123 passed** across
  `test_loopir_scheduled_slice.py` (94) and
  `test_loopir_pipeline_execution.py` (29), including the twelve panel
  parity cells, the compiled panel shadows, the panel stage-timing
  sequence, and the structural activation locks;
- focused production LoopIR membership: **533 passed + 4 neutrality**
  (131 verifier, 27 printer, 48 oracle, 16 level-storage, 16
  iteration-domain, 37 CIN lowering, 67 schedule passes, 68 LLIR
  lowering/parity, 94 scheduled slice, 29 pipeline execution);
- combined focused adjacency sweep (LoopIR fast suites + spike
  verifier/execution/neutrality + CIN/CIN-analysis + stage timing +
  LLIR pass-manager/string-budget/traversal + LoopPlan + native ABI +
  Schedule API + Scheduler + value-object boundaries): **1,880
  passed**, plus **332 passed** across the cin_lowerer,
  schedule-generality, and tune-scheduler-harness compiled adjacency
  files;
- byte gates: fresh 20-source corpus and 42-source grid captures from
  the working tree are **byte-identical** to detached `a880e86` captures
  and to the sealed Phase-6 captures (`diff -qr` empty for all
  comparisons) — no legacy emission changed, the byte waiver applies, no
  runtime kernel benchmark is required, and the Phase-5 §8 first-run
  failures and reviewed exception remain the permanent, un-rerun record;
- compiler latency: a fresh paired base(`a880e86`)/candidate run of the
  four-category corpus is inside the 1.10 target everywhere — p50
  ratios `1.006/1.000/1.020/0.983`, p95 ratios `1.009/1.009/1.018/0.981`
  (small-dense, reduction, CSR-intersection, sparse-union), with
  identical per-case source hashes; the release JIT path itself never
  enters the LoopIR stages;
- static parity: Black clean over every changed module and test; Flake8
  clean over the same; focused `mypy --check-untyped-defs` over all
  twelve package modules succeeds; full-source mypy reports exactly the
  **146 inherited findings in 12 files, zero in `loopir/`**, with the
  finding log identical to the §11 baseline; `git diff --check` clean
  before every commit (the one package-wide Black finding,
  `prebuilt_kernels.py`, and the nine package-wide Flake8 findings
  pre-exist identically at `a880e86` and are untouched inherited state);
- the authoritative clean detached-worktree non-performance suite at the
  exact final test commit `bb97172`, with isolated
  pytest/Torch-extension/cache directories and import provenance
  asserted: **3,756 passed, 14 skipped, 3 perf-marked deselections, one
  known warning, and zero failures/errors in 2,514.26 seconds** —
  exactly the §11 baseline of 3,692 passed plus the 64 new milestone
  tests (JUnit: 3,770 selected, zero failures/errors; log SHA-256
  `cd3af12b791f092b00771e089cf5a40629b118faf9c346f2c96ea87703d7f341`,
  JUnit SHA-256
  `c5e8e275211112d017f3cf21a49dc23a7cac5b4a1e9562d1af6f22eb566ad608`);
- the five protected tracked files retain their recorded SHA-256 values
  and were never staged; staging used explicit pathspecs only; no
  GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, or
  tooling material was touched; origin
  `refactor/compiler-ir-phase3-std-move-call` remains at `58e8565`
  (live `git ls-remote` confirmed at session start) and nothing was
  pushed.

### 12.6 Limitations and the candid Phase-6 exit verdict

- The migrated schedule surface is now: explicit complete loop orders,
  affine `accum="direct"` splits, the stack-accumulation workspace
  family, and sparse panel tiling with its mandatory parallel row loop.
  Operand relayout/staging, heap result tiles, and general explicit
  parallel selection remain fail-closed on the legacy route.  **Phase 6
  is therefore not exited.**  Against the design's Phase-6 deliverables:
  loop reorder, affine tiling/ragged tails, workspace materialization,
  and sparse panel tiling are done; operand relayout/staging, heap
  result accumulation, abstract parallel-loop selection, and the
  representative tile-j/tile-ijk `LoopPlan` encodings are still open
  (the panel-only tile-j *schedule* is now fully migrated; the
  tile-ijk relayout composition is not).
- `parallel_loop` is migrated only in its panel-mandated form, where the
  legacy family fixes its value; validating it against the window's
  dense-parent row loop is therefore its complete consumption.  General
  abstract parallel selection (a free choice of loop) remains the
  fail-closed stretch family.
- A panel tile's `unroll` flag is accepted with both values and has no
  emission effect, exactly as in legacy (`_apply_panel_tile` never reads
  it); the flag is a compatibility field of the shared `TileSpec`
  surface, not a dropped fact.
- The panel completion relies on a structural invariant of the managed
  pass pipeline — passes insert statements and rewrite expressions but
  never add, drop, or reorder the nest's for-loops — and enforces it
  with a fail-closed count and per-loop cross-checks
  (`panel_completion_lost`).  If a future managed pass restructures
  loops, panel compilation fails loudly at that boundary rather than
  emitting divergent bytes.
- Panel composition boundaries are explicit and fail closed: panels do
  not compose with workspace regions (`panel_target_invalid` — legacy
  requires `accum="direct"` for the panel itself, and the stack+panel
  composition has no parity reference), and windows over
  compressed-parent (DCSR) cursors remain `panel_nested_compressed`.
- The strangler entry remains test/debug-only; production dispatch,
  release JIT, import neutrality, legacy stage sequences, and
  source-derived kernel cache identity are unchanged and
  subprocess-enforced.
- Canonical dumps (schema v5) remain semantic fingerprints that omit
  display names; kernel caching remains source-derived; the Phase-4
  caution that dumps are not target cache keys stands.
- The §6 observations and the Phase-4/5 errata stand unchanged; the §11
  residual boundaries (INT_MAX stack widths, forged-CIN identity
  hardening) remain open and are unchanged by this milestone.
- For the relayout slice, the §11 access-identity requirement stands
  unresolved and must be decided before nodes are chosen: production
  LoopIR `Load` has no occurrence identity and the target's per-symbol
  `AccessId` is not a substitute.  The audited legacy relayout family is
  deliberately narrow (exactly the packed tile-ijk contraction with one
  rank-2 dense operand packed on its last level, `width == strip_width`,
  and scope ∈ {panel var, pack var}); within it, the staged operand is
  read by exactly one access occurrence, so *either* an artifact-local
  access identity *or* a verifier-proven unique operand/access-index
  tuple satisfies the design — the next session should decide with the
  goldens (`relayout_panel_scope.cpp` / `relayout_pack_scope.cpp` /
  `relayout_heap_pack.cpp`) in hand.

## 13. Independent review of the sparse-panel milestone (2026-07-24)

The panel milestone was reviewed again from the committed base
`9423d74`, without trusting §12's report.  The review found concrete
correctness, compatibility, and fail-closed-boundary defects.  They are
fixed in two focused commits; no preceding commit was amended or
reordered:

- `ab75d0f` — `fix(compiler): harden sparse panel completion`
- `4a3e269` — `test(compiler): lock sparse panel review boundaries`

### 13.1 Findings and corrections

1. **A verifier-valid bare LoopIR program could race.**  The target
   accepted the panel's dense-parent row as the OpenMP loop without
   proving that row coordinate appeared in the dense result access.
   A program such as a row-reduced `C[k] += ...` could therefore send
   several workers to the same result cells.  The target now requires
   the selected row to partition the single supported dense result
   access, independently of the stronger LoopPlan gate.
2. **Panel completion trusted an ordinal skeleton too far.**  It counted
   post-pass loops and compared only selected initializer variables.
   Header mutations, same-count sibling/reordered loops, a moved or
   changed window-end declaration, and lost compatibility fields could
   survive until completion and be rewired as though they were the
   emitted chain.  The lowerer now snapshots every complete emitted
   loop header and the window-end declaration before managed passes,
   re-identifies one exact direct parent/child chain, requires the
   declaration to retain exact state immediately before the window
   (apart from owned blank separators), and refuses to guess.
3. **Malformed post-pass artifacts were not total.**  Cyclic or shared
   statement ownership, missing nested fields, hostile subclasses, and
   forged expression equality could leak `AttributeError`,
   `RecursionError`, or arbitrary user exceptions from structural
   discovery.  Completion now uses exact registered statement kinds,
   bounded cycle/share/depth checks, direct stored-state validation, and
   a missing-field-safe structural comparator that never calls an
   untrusted `__eq__`.  Every such failure is the stage-owned
   `panel_completion_lost` diagnostic.
4. **Parallel marking was only assumed.**  A no-op or corrupted marker
   could omit the pragma, inject atomic scheduling/pre/post bodies, or
   substitute arbitrary `omp_num_threads` / chunk text.  Completion now
   builds the expected policy from a detached row-loop snapshot and
   requires exact post-mark header and compatibility-marker state,
   while deliberately ignoring the legitimate managed-pass
   `_hoisted_ptr_decls` payload.
5. **`unroll=False` was rejected on only one path.**  Legacy panel
   lowering accepts and ignores both values of the shared `TileSpec`
   field, and the typed panel pass already had the same semantics, but
   the LoopPlan legality gate rejected `False`.  Both values now pass
   Scheduler, LoopPlan, the typed pass, and byte-identical lowering.
6. **Enormous exact integers leaked CPython conversion errors.**  Widths
   could verify and then fail in the printer/JSON serializer under
   CPython's decimal digit limit, while huge levels and IDs could fail
   while formatting diagnostics.  Semantic affine/panel widths now
   have a target-neutral 2,048-bit canonical-print boundary (maximum
   617 decimal digits, below CPython's minimum configurable 640);
   the target retains its much smaller `INT_MAX` limit.  Diagnostic
   rendering is bounded for all exact-integer identities and level/mode
   fields reached by the panel family.
7. **Two empty-domain runtime boundaries were source-only.**  Zero rows
   and zero panel extent now compile and execute through the legacy and
   LoopIR shadow paths, in addition to the retained zero-free-extent
   case.

The review also corrected the `TileId` / builder documentation: the
identity space owns both affine splits and sparse panels, not affine
splits alone.  Adversarial re-review after the fixes found no remaining
concrete defect in the corrected completion boundary.

### 13.2 Verification

Evidence is retained at
`/Users/bobby/.cache/scorch-codex/phase6-panel-review-4a3e269/`.

- the five-file Phase-6 contract focus
  (`test_loopir_verifier.py`, `test_loopir_schedule_passes.py`,
  `test_loopir_llir_lowering.py`, `test_loop_plan.py`, and
  `test_schedule_api.py`) passes **414 tests** (the inherited §12 count
  plus 28 review regressions);
- the compiled runtime focus passes **126 tests** across
  `test_loopir_scheduled_slice.py` (97) and
  `test_loopir_pipeline_execution.py` (29); all thirteen panel
  source-parity cells remain byte-identical, and the two new
  empty-domain compiled shadows pass;
- focused production LoopIR membership is **565 passed + 4
  neutrality**, exactly the §12 membership plus the 32 review tests;
- fresh release-path captures are byte-identical to the sealed §12
  captures across all **20 corpus sources and 42 grid sources**;
- paired same-session compiler latency against `9423d74` is inside the
  1.10 target in every category: p50 ratios
  `0.989 / 0.987 / 0.975 / 0.969` and p95 ratios
  `1.014 / 0.957 / 0.928 / 0.962` (small-dense, reduction,
  CSR-intersection, sparse-union), with identical source hashes;
- Black and Flake8 are clean over every changed module/test; focused
  mypy is clean; full-source `mypy --check-untyped-defs` remains exactly
  the inherited **146 findings in 12 files**, with the normalized log
  byte-identical to §12 and zero findings in `loopir/`; `git diff
  --check` is clean;
- the authoritative clean detached-worktree non-performance suite at
  exact test commit `4a3e269`, with import provenance and caches
  isolated, passes **3,788 tests, 14 skipped, 3 perf-marked
  deselections, one known warning, and zero failures/errors in
  2,733.20 seconds** — exactly §12's 3,756 plus the 32 review tests.
  The log SHA-256 is
  `29f682e2c97c3e95967d55508dee8b566c577f956dd8de0b16adec3e1fff1383`;
  the JUnit SHA-256 is
  `afadef4ee0def33c7117df757820564c3e70769220ee635314eb272d85f6dba6`;
- the five protected tracked files retain their recorded hashes and
  were never staged; all staging used explicit pathspecs; unrelated
  GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, and
  tooling material remains untouched.  Live origin remains at
  `58e8565`; nothing was pushed.

### 13.3 Revised boundary and Phase-6 verdict

The panel slice is now closed under the stronger contract above, but
**Phase 6 remains open** for exactly the §12 remainder: full
operand-relayout/staging, heap result tiles, target-independent abstract
parallel selection, and the representative tile-ijk composition/exit
audit.  Relayout's post-assembly work must extend this corrected
boundary: no rendered names, regexes, dynamic tags, or bare ordinal
matching.  It must use stable artifact identity/provenance where
available or a complete retained structural snapshot with the same
missing/extra/reordered/shared/cyclic/malformed fail-closed discipline.
