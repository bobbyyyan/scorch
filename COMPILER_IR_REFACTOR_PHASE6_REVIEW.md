# Phase 6 Review: Scheduling Migration to LoopIR (Explicit Reorder + Affine Tiling)

Date: 2026-07-23 (America/Los_Angeles); §10 (the workspace/stack milestone)
and §11 (its review) added later the same day; §§12–15 (the panel and
relayout milestones plus their independent reviews) added 2026-07-24.

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

## 14. Relayout milestone: staged operand packing on LoopIR (2026-07-24)

The fourth Phase-6 milestone migrates the operand-relayout/staging
schedule family — the packed tile-ijk contraction's staged dense
operand, the family the audited legacy `_apply_relayout` completes — end
to end for **both staging scopes**: schema, verifier,
printer/serialization, oracle, typed pass, plan gate, erasure,
provenance, target lowering with byte parity, and a compiled matrix.
Four stacked local commits plus the docs commit that records this
section; nothing earlier was amended or reordered and nothing was
pushed:

- `075255f` — `feat(compiler): extend LoopIR with the operand staging region schema`
- `3aa9ec1` — `feat(compiler): apply operand relayout plans as a typed staging pass`
- `e8dd924` — `feat(compiler): lower staged operand relayout with legacy byte parity`
- `61b68be` — `test(compiler): lock the Phase-6 relayout vertical slice`

One candid process note: the test-lock commit was first created as
`85e666c` and, within minutes and before any gate ran against it, was
amended in place to `61b68be` to fold in the string-budget lock update
its own refactor required (an accidental `--amend` in a shell chain,
disclosed rather than repaired with further history surgery).  The
amended commit was this session's own unpushed tip; no pre-existing
commit of the reviewed history was touched, and every §14 gate below ran
at `61b68be`.

Both §13 gates were independently reproduced at `f41fbf7` before any
edit — the five-file contract focus passed **414** and the compiled
runtime focus passed **126** (531.87 s) — the §13 correction diffs were
read in full, and four independent adversarial probes (the maximum
2,048-bit width verifying and canonically printing at CPython's minimum
640-digit limit, the width boundary+1 failing `invalid_tile_width` with
a total message, a huge forged identity rendering a bounded diagnostic,
and the bare-program row-reduction race gate) all passed.  No concrete
§13 defect was found.

### 14.1 Audit and the access-occurrence identity decision

The relayout re-audit and the identity decision were recorded under
`/Users/bobby/.cache/scorch-codex/phase6-relayout-audit/AUDIT.md`
**before any schema was written**, with the nine panel/relayout goldens
re-read and the legacy chain re-checked at this tip
(`Scheduler._validate_relayout`, `_apply_relayout`,
`_packed_storage_declaration`, `_TensorAccessRewriter` and its
preflight, `_redirect_sparse_prefetch`, both staging lifetimes, and the
`apply_schedule_to_llir` ordering: panel → heap → relayout on the
assembled function).

**The decision: no occurrence identity is added to `Load`.  The typed
pass proves, and re-proves after rebuilding, that the staged operand has
exactly one read occurrence — the deliberately narrow verifier-proven
unique operand/index occurrence boundary the §11/§12 reviews permit.**
Rationale, recorded before schema freeze: (1) within the audited family
the staged operand is read by exactly one access occurrence
(`_validate_relayout` admits exactly two RHS accesses in one
multiplicative contraction), and the pass makes that a checked property
(`relayout_target_missing` / `relayout_ambiguous_access`), never an
assumption; (2) the redirected read is structural — `StagedRead`
carries an artifact-local `RelayoutId`, so after the pass the *region*
identity is the stable anchor and nothing downstream re-identifies the
original occurrence; (3) per-occurrence `Load` identity would be a
schema-wide change (every builder call site, canonical serialization,
continuation, goldens) serving one consumer whose family is proven
single-occurrence, and can be introduced later without breaking this
design if a multi-occurrence family ever needs it; (4) the lower-LLIR
per-symbol `AccessId` is used nowhere as a logical occurrence identity —
the emitted-side redirection consumes the typed
`(tensor_id, index_ids, role)` metadata triple exactly as the audited
legacy rewriter does, made unambiguous by the LoopIR-side proof.

### 14.2 What was frozen (schema extension)

New in `loopir/nodes.py`: `RelayoutId` (builder-allocated,
artifact-local, covered by the declared-field-only `resuming` scan),
`RelayoutScope` (`PANEL` | `PACK_AXIS`), `RelayoutDecl(relayout,
operand, panel: TileId, pack: TileId, scope)`, the region statement
`RelayoutStage(decl, body)`, and `StagedRead(relayout, indices)` — the
staged twin of `Load`.  Region semantics are intrinsic: at entry the
operand's current strip is staged (PANEL: the panel's current clamped
window rows; PACK_AXIS: the whole panel axis; columns always the pack
split's current clamped point window), the staged cells hold exactly
`operand[r, c]`, the strip is valid throughout the body, and teardown at
exit means every scope iteration observes a fresh strip.  Buffer
naming, capacity arithmetic, reuse, and pack-loop emission are target
concerns with no spelling in the nodes.  Canonical serialization moved
to schema `scorch.loopir.canonical.v6`; the printer renders the region
(`relayout_stage r0 t1 panel s1 pack s0 scope panel`) and staged reads
(`staged r0[x1, x0]`); dumps stay stable across unrelated global
identity histories and renumber raw relayout identities.  The oracle
executes the intrinsic semantics with fail-closed runtime guards —
region outside its pack origin, PANEL region outside its panel origin,
re-entry, staged reads outside the region or outside the staged
row/column domains — serving values lazily from the operand (staging
copies exactly), so nothing is eagerly allocated from verifier-approved
widths.

### 14.3 Verifier surface

Seven stable codes were added, each with direct adversarial
regressions: `invalid_relayout_id`, `duplicate_relayout_id`,
`unbound_relayout` (a staged read with no enclosing region in scope,
including huge-identity diagnostic totality),
`relayout_scope_mismatch` (no dominating pack origin; PANEL outside its
panel origin; PACK_AXIS inside it; a panel identity naming an affine
split), `relayout_operand_mismatch` (wrong rank/kind or level dimensions
against the panel/pack dimensions), `relayout_read_mismatch` (the row
index must be the panel's window coordinate and the column index the
pack split's point coordinate — a rebound same-dimension coordinate is
rejected), and `relayout_dead_region`.  Hostile
`RelayoutStage`/`StagedRead`/`RelayoutId` subclasses, missing stored
fields, non-member scopes, cyclic bodies, and shared nodes fail through
the existing guards.  The locked source-scan surface grows from 64 to
**71** codes.

### 14.4 Pass, plan gate, erasure, and provenance

`apply_relayout(program, relayout)` is the new pure typed pass,
operating on identities only, applied by `apply_schedule_plan` **after**
every tile — the fully scheduled chain, exactly where the legacy
lowering completes relayout.  Each `OperandRelayout` fact is consumed
exactly once: `pack_loop`/`panel_loop` select the chain's two schedule
pairs, `scope_loop` selects the region scope, `row_loop` is validated
against the window's dense-parent row coordinate (the fact has no other
legal value — the panel `parallel_loop` precedent), `strip_width`
against the pack split's width, the two operand levels against the
declaration's dimension structure, and `access_indices` select the
redirected read.  The pass requires exactly the audited five-loop chain
(pack origin, panel origin directly below it, parallel row, window,
pack point) with a direct dense-result leaf, proves the operand's
unique `Load` occurrence, replaces it structurally with a `StagedRead`
carrying the fresh region identity, wraps the row loop (PANEL) or the
panel origin (PACK_AXIS) in the region, and re-checks that no residual
direct read survived the rebuild.

The plan gate replaces the unconditional `unsupported_schedule_relayout`
rejection with the exact family admission (`invalid_schedule_relayout`:
exactly one outermost affine pack tile at the strip width plus one
panel placed `child_of` the pack origin, `parallel_loop` equal to the
relayout row, scope in the pair, logical loop refs); the
heap-accumulation composition stays fail-closed as the unmigrated heap
family (`unsupported_schedule_result_tile` through the Scheduler's
plan, `unsupported_schedule_accumulation` on a direct plan).
`_decompose_body` treats the region as a chain element only for callers
that pass a relayout sink (provenance, erasure, the carrier's
base-purity check — which now rejects staging regions explicitly);
every scheduling pass keeps the default and refuses already-relayouted
chains with `unsupported_schedule_shape`.  Provenance stays loop-only
(five entries; the region binds no loop) with the region's placement
covered by the carrier's deterministic replay equality.
`erase_schedule` drops the region and restores the plain operand
`Load`, proven equal to the reordered base by canonical dump for both
scopes and by exact oracle differentials (all-ones counting across
ragged panel windows and ragged pack strips in both scopes, randomized
dimensions, zero extents).

### 14.5 Target lowering and the post-assembly completion

The target accepts the staging region on exactly the audited chain
(`_validate_relayout_shape`: the five-loop kinds, shared tile
identities, the scope-consistent region depth, the operand's dimension
structure, and exactly one `StagedRead` of the region in the compute
leaf indexed by the window and point coordinates).  The staged read is
recorded as a synthetic direct-`Load` view, so raw emission — position
resolves, drivers, bounds, metadata — and **every managed pass** see
byte-for-byte the tree the legacy pipeline transforms; the dead
`pB1` resolve and the direct `B_val` read the passes decided on are
byte-preserved.

`complete_relayout` runs on the assembled function immediately after
`complete_panel` — the legacy `apply_schedule_to_llir` order — and
extends the corrected §13 completion boundary: it consumes the panel
completion's retained, already re-identified loop objects (the pack
origin from the verified chain, the created panel origin, the row loop,
the window — never a second discovery pass); redirects the emitted
operand read by the typed metadata triple exactly once with a residual
re-check; adapts the sparse prefetch through the schedule lowerer's
shared constructor while passing the target's own coordinate-array
spelling so the legacy `_find_coordinate_array` name scan never runs on
the typed path; re-identifies the window's resolved coordinate against
a detached pre-pass snapshot (the §13 `_exact_panel_state_matches`
comparator) before inserting the compatibility range guard; and places
the pack loop and reusable storage through helpers extracted from the
legacy `_apply_relayout` (`_panel_range_guard`, `_relayout_pack_loop`,
`_relayout_storage_statements`, plus the widened
`_redirect_sparse_prefetch`) — one source per spelling, the extraction
proven byte-neutral by regenerating all nine retained audit goldens
byte-identically.  Every disagreement is the stage-owned
`relayout_completion_lost` diagnostic, including a lost completion
record and a corrupted coordinate declaration.  The raw-string-budget
lock follows the moved `_packed_storage_declaration` owner into the
shared helper.

**The compiled matrix.**  An eleven-cell relayout byte-parity grid
locks generated C++ equality against `Scheduler.apply_schedule` + the
legacy lowering in every cell: both staging scopes, unit widths, a
panel width above the extent, strips not dividing the extents, f32/f64
in both scopes, zero rows, zero panel extent, zero free extent, and
larger non-square shapes (a ten-cell independent probe sweep during
development also passed byte-identically, including width 64/strip 32).
Compiled shadow execution (both pipelines, real kernels) is
**bitwise-equal** to the legacy scheduled kernels and PyTorch-close for
both scopes across the three window regimes with an empty CSR row, f64
at 1e-10 tolerances, and all three zero-extent boundaries; randomized
dimensions, widths, strips, and scopes execute against PyTorch and the
production oracle.  Structural activation is asserted directly and
never waived: the reusable packed storage and restrict pointer for both
capacities (`(size_t)kTile_j * (size_t)kTile_k` and
`(size_t)B0_size * (size_t)kTile_k`), both scope-specific pack pragmas
and destination spellings, the compatibility range guard, the
redirected compute read, the packed prefetch for both scopes, and the
absence of any residual `B_val[pB1]` read.

### 14.6 Verification

Evidence ledger:
`/Users/bobby/.cache/scorch-codex/phase6-relayout-61b68be/` (audit and
identity decision under `phase6-relayout-audit/`).

- both §13 gates were independently reproduced before any edit: the
  414-test contract focus and the 126-test runtime focus at `f41fbf7`;
- contract focus after the milestone (same five files): **440 passed** —
  the 26 new contract regressions cover the relayout verifier codes,
  the pass surface, the plan gate, and the target boundaries;
- scheduled runtime focus after the milestone: **150 passed** in
  622.80 s across `test_loopir_scheduled_slice.py` (121) and
  `test_loopir_pipeline_execution.py` (29), including the eleven relayout parity cells, the compiled
  relayout shadows, and the structural activation locks;
- focused production LoopIR membership: **625 passed + 4 neutrality**
  (the §13 membership plus the 60 milestone tests across verifier,
  printer, oracle, schedule passes, LLIR lowering, and the scheduled
  slice);
- combined focused adjacency sweep (LoopIR fast suites + spike
  verifier/execution/neutrality + CIN/CIN-analysis + stage timing +
  LLIR pass-manager/string-budget/traversal + LoopPlan + native ABI +
  Schedule API + Scheduler + value-object boundaries): **1,945
  passed**, plus **332 passed** across the cin_lowerer,
  schedule-generality, and tune-scheduler-harness compiled adjacency
  files;
- byte gates: fresh 20-source corpus and 42-source grid captures from
  the working tree are **byte-identical** to detached `f41fbf7`
  captures (`diff -qr` empty), and all nine retained panel/relayout
  audit goldens regenerate byte-identically after the schedule-lowerer
  helper extraction — no legacy emission changed, the byte waiver
  applies, no runtime kernel benchmark is required, and the Phase-5 §8
  first-run failures and reviewed exception remain the permanent,
  un-rerun record;
- compiler latency: a fresh paired base(`f41fbf7`)/candidate run of the
  four-category corpus, measured sequentially on a quiet machine, is
  inside the 1.10 target everywhere — p50 ratios
  `0.978/0.991/0.981/0.973`, p95 ratios `0.933/1.033/0.966/0.998`
  (small-dense, reduction, CSR-intersection, sparse-union), with
  identical per-case source hashes; the release JIT path itself never
  enters the LoopIR stages;
- static parity: Black clean over every changed module and test; Flake8
  clean over the same; focused `mypy --check-untyped-defs` clean over
  all thirteen package modules including `schedule_lowerer.py`;
  full-source mypy reports exactly the **146 inherited findings in 12
  files, zero in `loopir/`**; `git diff --check` clean before every
  commit;
- the authoritative clean detached-worktree non-performance suite at
  the exact final test commit `61b68be`, with isolated
  pytest/Torch-extension/cache directories and import provenance
  asserted: **3,848 passed, 14 skipped, 3 perf-marked
  deselections, one known warning, and zero failures/errors in
  2,538.08 seconds** — exactly the §13 baseline of 3,788 passed plus
  the 60 new milestone tests (JUnit: 3,862 selected; log SHA-256
  `74e6961b50ca9a60cd0ff9b59fbbb4769ef1714219d60673551403b098effda8`,
  JUnit SHA-256
  `0a14d814abd0a024f1f86baeea22dbcd256e7e72e6a156ffcae10e7a8156cbfa`);
- the five protected tracked files retain their recorded SHA-256 values
  and were never staged; staging used explicit pathspecs only; no
  GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, or
  tooling material was touched; origin
  `refactor/compiler-ir-phase3-std-move-call` remains at `58e8565`
  (live `git ls-remote` confirmed at session start) and nothing was
  pushed.

### 14.7 Limitations and the candid Phase-6 exit verdict

- The migrated schedule surface is now: explicit complete loop orders,
  affine `accum="direct"` splits, the stack-accumulation workspace
  family, sparse panel tiling with its mandatory parallel row loop, and
  operand relayout/staging at **both** scopes with direct accumulation.
  Heap result tiles and general abstract parallel selection remain
  fail-closed on the legacy route.  **Phase 6 is therefore not
  exited.**  Against the design's Phase-6 deliverables: loop reorder,
  affine tiling/ragged tails, workspace materialization, sparse panel
  tiling, and operand relayout/staging are done; heap/stack/direct
  result accumulation is done only for stack and direct (heap open);
  abstract parallel-loop selection is open; of the representative
  encodings, panel-only tile-j and the direct-accumulation tile-ijk
  composition (pack + panel + relayout, both staging scopes, through
  the public Schedule → verified LoopPlan adapter) are now migrated and
  byte-locked — the heap-accumulation tile-ijk variant needs the heap
  family first.
- The relayout family is exactly the audited legacy shape: one rank-2
  all-dense operand packed on its contiguous last level under one CSR
  input and a dense `(row, pack)` result.  Verifier-legal
  generalizations (other ranks, permuted levels, multiple regions,
  regions at other depths) stay fail-closed at the pass or target
  boundary, never silently emitted.
- The completion's coordinate re-identification relies on the managed
  passes preserving the window's resolved-coordinate declaration
  byte-for-byte (true today; the prefetch pass inserts above it).  A
  future pass that legally rewrites that declaration will fail loudly
  as `relayout_completion_lost` at this boundary rather than emit
  divergent bytes — the same posture as the §13 panel snapshot.
- The strangler entry remains test/debug-only; production dispatch,
  release JIT, import neutrality, legacy stage sequences, and
  source-derived kernel cache identity are unchanged.
- Canonical dumps (schema v6) remain semantic fingerprints that omit
  display names; kernel caching remains source-derived.
- The §6 observations, the Phase-4/5 errata, and the §11 residual
  boundaries (INT_MAX stack widths, forged-CIN identity hardening)
  stand unchanged.

## 15. Independent review corrections to the relayout milestone (2026-07-24)

Commits `6ac5704` (production fixes), `e4cfa50` (regression lock), and
`bba935e` (independent identity-snapshot lock) follow the §14 milestone
without amending or reordering it.  This review did not change the
LoopIR schema, canonical-v6 serialization, or the 71-code verifier
surface.  It did find several concrete gaps in the post-assembly
compatibility boundary, including one wrong-code path.  This section
supersedes §14 wherever it strengthens the completion contract.

### 15.1 Findings and corrections

1. **The plan gate did not own the exact admitted family.**
   `_check_plan_families` admitted a wrong logical loop order, swapped
   access axes/physical levels, and stack accumulation.  Some invalid
   plans therefore ran multiple typed passes before failing under an
   unrelated late diagnostic.  Relayout preflight now requires the
   exact `(row, panel, pack)` order, `(panel, pack)` operand access,
   physical levels `0/1`, and direct accumulation before replay begins.
2. **The metadata triple was not a physical-occurrence identity.**
   Swapping two individually valid A/B `TensorAccessMetadata` values
   after panel completion made the old code redirect A's physical access
   as though it were B's, leaving the real B read direct.  The resulting
   C++ contained a packed-B read multiplied by `B_val[...]` and was
   accepted instead of failing.
   Target lowering now retains a detached snapshot of the exact emitted
   `ArrayAccess`, including independently rebuilt `AccessId`, `SymbolId`,
   and `IndexId` values.  Completion requires exactly one metadata
   candidate whose entire physical subtree matches that snapshot before
   the existing exact-one rewrite/residual checks run.  This keeps the
   deliberately narrow no-`Load`-occurrence-ID decision while making its
   emitted-side proof real.
3. **The coordinate snapshot did not prove dominance.**  Matching only
   the detached `VarInit` accepted moving the declaration below a use;
   matching the first correction's three-statement context alone still
   accepted moving that whole context.  Completion now re-identifies the
   exact `(Comment, VarInit, BlankLine)` skeleton at its canonical lexical
   position immediately after the rewritten prefetch.  Moving either the
   declaration or the intact context fails as
   `relayout_completion_lost`.
4. **Prefetch cardinality was best-effort on the typed path.**  No
   canonical guard was silently accepted, duplicate canonical guards
   were collapsed, and a nested noncanonical guard for the original
   operand could survive beside the packed guard.  The shared legacy
   helper now reports the number removed (its compatibility caller still
   intentionally ignores that result); typed completion requires exactly
   one and recursively rejects every residual direct prefetch of the
   unstaged operand.
5. **Malformed provenance could escape the stage boundary.**  Candidate
   discovery previously consulted dataclass equality before full
   validation, allowing missing, cyclic, or hostile identity state to
   leak `AttributeError`, `RecursionError`, or user equality behavior.
   Common LLIR traversal now validates the exact stored fields and exact
   integer payload of every tensor-access `AccessId`, `SymbolId`, and
   `IndexId`.  Completion matches only after that validation and compares
   exact-string state keys without invoking forged equality.  Malformed,
   extra-key, hostile-value, cyclic, and shared-snapshot adversaries all
   terminate under the owned diagnostic.

The regression review also corrected three evidence/documentation issues:
the purported “after region exit” verifier case was still inside the
region and now has a real sibling-after-exit adversary; the all-ones oracle
case proves compute visitation, not physical staging freshness; and the
package/node documentation now includes panels, relayout, `RelayoutId`,
and the distinction between intrinsic PACK_AXIS entry semantics and the
typed pass's placement.

### 15.2 Verification

Evidence ledger:
`/Users/bobby/.cache/scorch-codex/phase6-relayout-review-e4cfa50/`.

- exact five-file contract focus at committed test tip `bba935e`:
  **452 passed** (the §14 lock plus 12 new relayout cases);
- common LLIR traversal file: **435 passed**, including 28 new
  walk/rewrite cases over all three identity carriers and missing,
  boolean, hostile-value, and hostile-extra-key state;
- scheduled compiled runtime focus
  (`test_loopir_scheduled_slice.py` plus
  `test_loopir_pipeline_execution.py`): **150 passed**; both relayout
  scopes still lower and execute through the unchanged valid route;
- full legacy schedule-generality file: **45 passed**; the eleven-cell
  relayout source-parity matrix remains byte-identical, and all nine
  retained panel/relayout audit goldens (including
  `relayout_heap_pack.cpp`) regenerated with an empty diff;
- fresh candidate captures contain 20 corpus sources and 42 grid
  sources, each byte-identical to the sealed §14 candidate captures;
  structural relayout activation remains directly tested and is not
  waived;
- Black and Flake8 are clean over every changed production/test file;
  focused mypy is clean.  Fresh clean detached base/candidate
  full-source mypy logs are line-normalized byte-identical at **140
  inherited findings in 11 files, zero in `loopir/`**.  This clean
  committed-tree comparison supersedes §14's stale 146-finding count;
- paired sequential compiler latency retained both execution orders.  The
  first base-then-candidate run kept every p50 inside the 1.10 target but
  crossed at p95 for `csr_intersection` (`1.222`) and `sparse_union`
  (`1.143`).  The required candidate-then-base control did not reproduce
  either crossing: all cells passed, with worst p50 `1.028` and worst p95
  `1.021`.  Source hashes were identical at both revisions, so the
  order-dependent p95 tails are attributed to session-position machine
  drift, not the correction.  Both complete JSON pairs, comparison logs,
  and `latency-attribution.md` are retained;
- the authoritative isolated clean-worktree suite at `bba935e`:
  **3,888 passed, 14 skipped, 3 deselected, 1 known warning, 0 failed**
  in 2,653.72 seconds, with import provenance asserted.  The log SHA-256
  is `c521f4e64219b35c2fb91182c278da54835805fa7cc350f68894ebd102a34f0a`
  and the JUnit SHA-256 is
  `0035bd451789c850dcc53abd4b29e639688ea8273a1caf0385bf8caff7fb401d`;
- `git diff --check` is clean.  The five protected tracked files retain
  their recorded SHA-256 values; only explicit pathspecs were staged;
  unrelated GPU/CUDA, benchmark, packaging, scheduler, research,
  scratchpad, and tooling material remains untouched.  Live origin
  remains `58e8565`; no commit was pushed.

### 15.3 Corrected boundary and Phase-6 verdict

The direct-accumulation relayout slice remains closed, but only under
the stronger exact-family, physical-occurrence, independent-snapshot,
lexical-dominance, exact-prefetch-cardinality, and traversal-totality
contract above.  No occurrence identity was added to `Load`; the
single-occurrence pass proof plus the exact physical LLIR fingerprint is
the deliberately narrow boundary.

**Phase 6 remains open.**  Heap result-tile accumulation and its
heap-relayout tile-ijk composition, target-independent abstract
parallel-loop selection, intended automatic-plan provenance, canonical
LoopPlan schedule cache identity, and the criterion-by-criterion exit
audit remain.  Phase 7 policy/pass migration has not started.

## 16. Heap result-tile milestone: originating report, superseded by §17 (2026-07-25)

The originating report claimed that the fifth Phase-6 milestone
migrated a rank-2 trailing-axis subset of the audited rank>=2 legacy
`_apply_heap_result_tile` family end to end: schema, verifier,
printer/serialization, oracle, typed pass, plan gate, erasure,
provenance, target lowering with byte parity, and a compiled matrix
covering heap alone and the heap tile-ijk composition (pack + panel +
relayout) at **both** staging scopes.  Five stacked local commits plus
the docs commit that records this section; nothing earlier was amended
or reordered and nothing was pushed:

- `39ffa7a` — `feat(compiler): extend LoopIR with the heap result-tile region schema`
- `087f04e` — `feat(compiler): apply heap result-tile plans as a typed accumulation pass`
- `f7d3af6` — `feat(compiler): lower heap result tiles with legacy byte parity`
- `34b933c` — `test(compiler): lock the heap result-tile completion boundaries`
- `554a1eb` — `test(compiler): move the remaining heap boundary locks`

The session first independently reviewed the §15 correction commits
(`6ac5704`/`e4cfa50`/`bba935e`/`4abc8fa`): the exact contract focus
(**452**), common traversal (**435**), compiled runtime focus (**150**),
and schedule generality (**45**) all reproduced at the docs tip, and four
independent adversarial probes beyond the committed lock — a duplicated
coordinate-context decoy, an in-place `IndexId` payload mutation, a
post-pass physical array-spelling mutation, and valid-path both-scope
lowering — all behaved as §15 claims.  **No concrete §15 defect was
found**; no fix commit was needed.

### 16.1 Audit and boundary decisions

The heap audit was recorded under
`/Users/bobby/.cache/scorch-codex/phase6-heap-audit/AUDIT.md` **before
any schema was written**: the responsibility/lifetime table for
`Scheduler._validate_heap_result_tile`, `_ResultTilePlan`,
`_apply_heap_result_tile`, `_remove_dense_result_zero`, the compact
access rewriting, allocation/reset/copy-out, parallel-prefix legality,
and the `relayout_heap_pack.cpp` composition, plus eleven fresh legacy
goldens (exact/ragged/unit/oversized/f64 heap-alone, the rank-3 TTM
multi-prefix reference, heap+panel, and heap+relayout at both scopes in
f32 and f64), with `relayout_heap_pack.cpp` byte-identical to the
§12/§13 retained copy.

Recorded boundary decisions:

- **No occurrence identity on `StoreReduce`** — the identical decision
  to the relayout `Load` boundary.  The pass proves the result's unique
  write occurrence and the region identity anchors everything
  downstream.  Unlike operand reads, result writes are statements the
  verifier's coordinate model fully pins within this family, so the
  exact fact admission subsumes the scan; it is retained as
  checked-property defense in depth.
- **Region nesting**: `ResultTileRegion` wraps the pack origin's entire
  body (after relayout), one uniform shape for heap-alone, heap+panel,
  and both relayout scopes; the PACK_AXIS staging region sits *inside*
  the result-tile region.  The oracle's region-entry order (fresh
  compact tile, then staging) differs from the emitted C++ order (pack
  loop, then init loop) — a deliberate observational equivalence:
  staging reads only the operand and the init writes only the compact
  tile, so the entries commute.
- **The typed implementation is a rank-2 subset of the audited rank>=2
  family** (one dense prefix loop).  Multi-prefix heap (the TTM shape)
  remains legacy-only and
  fails closed at the plan gate as `invalid_schedule_result_tile` —
  a recorded boundary, never a silent emission.  The dense-reduction
  heap-alone chain (dense matmul tile-j heap) is admitted by the pass
  and the target but is not in the byte-locked compiled matrix.
- **Parallel-prefix race legality**: the plan gate requires the
  parallel loop to be a LOGICAL dense result-prefix loop; with one
  prefix level the selection has no degrees of freedom (the panel
  precedent), so heap plans admit `parallel_loop` without a panel and
  the typed target marks the row loop at completion exactly as the
  legacy explicit-parallel schedule does.  The emission-time auto
  parallel gate is suppressed for heap chains, mirroring the panel
  suppression, so raw emission and every managed pass keep seeing the
  legacy unmarked tree.

### 16.2 What was frozen (schema extension)

New in `loopir/nodes.py`: `ResultTileId` (builder-allocated,
artifact-local, covered by the declared-field-only `resuming` scan),
`ResultTileDecl(result_tile, result, pack)`, the region statement
`ResultTileRegion(decl, body)`, and `TiledReduce(result_tile, indices,
op, value)` — the staged twin of `StoreReduce`.  Region semantics are
intrinsic: a fresh all-zero compact tile per pack-origin iteration (one
cell per dense prefix position and clamped window column — ADD's
identity, the reduction-legality contract), accumulation through
`TiledReduce` only, and exactly-once copy-out of every clamped-window
cell at region exit — which is what discharges the result's
whole-tensor zero-initialization contract on the heap route (cells that
received no accumulation copy the entry zero, covering empty rows).
Canonical serialization moved to schema `scorch.loopir.canonical.v7`;
the printer renders `result_tile_region h0 t2 pack s0` and
`tiled_reduce(add) h0[x1, x0]`, stable across unrelated identity
histories with raw `ResultTileId` renumbering.  The verifier grows
eight stable codes (`invalid_result_tile_id`, `duplicate_result_tile_id`,
`unbound_result_tile`, `result_tile_scope_mismatch`,
`result_tile_result_mismatch`, `result_tile_write_mismatch`,
`result_tile_residual_write`, `result_tile_dead_region`), each with
direct adversarial regressions (hostile subclasses, forged and
huge-integer identity payloads, missing fields, cycles, nested
same-result regions, per-point regions, rebound trailing coordinates,
residual direct writes while a region is open); the locked source-scan
surface grows from 71 to **79** codes.  The oracle executes the
intrinsic semantics with fail-closed guards (region outside its pack
origin, re-entry, reduces outside the region or the compact window
domain), keeps only written cells so verifier-approved widths never
become allocation requests, and copy-out enumerates the
caller-allocated result.

### 16.3 Pass, plan gate, erasure, and provenance

`apply_result_tile(program, result_tile)` is the new pure typed pass,
applied by `apply_schedule_plan` **last** — after `apply_relayout`, so
the region wraps the fully staged chain.  Each `ResultTile` fact is
consumed exactly once: `tile_loop` selects the pack schedule pair,
`access_indices` select the redirected write, and
`result_prefix`/`result_level` are validation-is-consumption against
the result declaration.  The pass admits exactly the audited chains
(the pack origin over the dense row loop, one sparse or dense reduction
loop, and the pack point loop — optionally windowed by one panel with
at most one relayout stage), proves the unique `StoreReduce`
occurrence, replaces it with a `TiledReduce` carrying the fresh region
identity, wraps the pack origin's body in the region, and re-checks
that no residual direct write survived.

The plan gate replaces the blanket `unsupported_schedule_result_tile`
rejection with the exact family admission
(`invalid_schedule_result_tile`): one serial outermost heap-accumulation
affine tile targeting the plan's innermost logical loop, paired
one-to-one with the `result_tile` fact compacting a rank-2 dense result
on its trailing storage level, at most one composed panel tile, and the
mandatory parallel dense result-prefix loop.  A heap pack tile in the
relayout composition now requires the matching result-tile fact
(previously `unsupported_schedule_accumulation`).  Chain decomposition
gains a result-tile sink for provenance, erasure, and carrier purity;
the carrier's base-purity check explicitly rejects result-tile regions
and tiled-reduce leaves; provenance stays loop-only across both
transparent regions.  `erase_schedule` restores the plain result
reduction (the region's fresh-zero/copy-out semantics are an ADD
reassociation), proven by canonical-dump equality and all-ones counting
differentials across exact/ragged/unit/oversized strips for heap alone
and both composition scopes.

### 16.4 Target lowering and the post-assembly completion

The nest walk accepts the region and its `TiledReduce` compute leaf and
records the leaf as a synthetic direct `StoreReduce` view, so raw
emission — including the result-write metadata — and **every managed
pass** see byte-for-byte the tree the legacy pipeline transforms.  The
emitted result write is snapshotted with independently rebuilt
`AccessId`/`SymbolId`/`IndexId` payloads — the §15 detached-fingerprint
discipline.

`complete_result_tile` runs on the assembled function **between**
`complete_panel` and `complete_relayout` — exactly the legacy
`apply_schedule_to_llir` order (panel, heap, relayout).  Panel chains
consume the panel completion's retained loop objects; the bare heap
chain re-identifies its direct loops against the detached pre-pass
headers and marks the parallel row with the same
policy/marking/verification block the panel completion uses, under
owned-diagnostic totality.  Completion requires exactly one metadata
candidate whose entire physical subtree matches the detached snapshot,
redirects the write to compact storage exactly once with a residual
re-check, re-proves exactly one generated dense-result zero before
removing it (the exactly-once copy-out coverage proof), and places the
init/copy groups and reusable storage through helpers extracted
byte-neutrally from the legacy `_apply_heap_result_tile`
(`_heap_result_tile_names`, `_heap_compact_access`,
`_heap_result_init_group`, `_heap_result_copy_group`,
`_heap_result_storage_statements` — one source per spelling; all eleven
goldens defined by the available heap audit generator regenerate
byte-identically; no separate 20-case generator exists, as corrected
in §17; the raw-string-budget lock follows the moved storage owner).
The originating report claimed every disagreement became the
stage-owned `result_tile_completion_lost` diagnostic.  Section 17
disproves that claim and records the corrected ownership boundary,
including repeating lifetimes, assignment ownership, hidden effects,
Torch-storage aliases, and malformed top-level state.

**The compiled matrix.**  A fourteen-cell heap byte-parity grid locks
generated C++ equality against `Scheduler.apply_schedule` + the legacy
lowering in every cell: heap alone at exact/ragged/unit/oversized
strips, f32/f64, heap+panel without relayout, the heap tile-ijk
relayout composition at both staging scopes including f64, all three
zero-extent boundaries, and larger non-square shapes.  Structural
activation is asserted directly and never waived: the reusable
`tiled_C` storage and restrict pointer, both init/copy groups with the
legacy `scorch_nthreads((C0_size) * kTile_k, (C0_size))` static policy,
the redirected compute write, the removed `scorch_zero_dense`, the
retained dead `pC1` resolve, and the legacy sparse dynamic policy on
the marked row loop.  Compiled shadow execution is **bitwise-equal** to
the legacy scheduled kernels and PyTorch-close across strip regimes
with an empty CSR row, both relayout scopes, f64 at 1e-10, and all
three zero-extent boundaries; randomized heap and composition rounds
execute against PyTorch and the production oracle.

### 16.5 Verification

Evidence ledger:
`/Users/bobby/.cache/scorch-codex/phase6-heap-34b933c/` (audit and
goldens under `phase6-heap-audit/`).

- §15 gates reproduced before any edit: contract 452, traversal 435,
  runtime 150, generality 45, plus the four independent probes above;
- exact five-file contract focus after the milestone: **485 passed**
  (the §15 lock plus 33 heap contract regressions across the verifier,
  the pass surface, the plan gate, and the target boundaries);
- common LLIR traversal file: **435 passed** (unchanged);
- focused production LoopIR membership (verifier, printer, oracle,
  schedule passes, LLIR lowering, CIN lowering, levels, iterdomain):
  **530 passed + 4 neutrality**;
- scheduled compiled runtime focus
  (`test_loopir_scheduled_slice.py` + `test_loopir_pipeline_execution.py`):
  **not completed by the originating session**; the rigorous review in
  §17 supersedes this missing receipt with an exact 177-test rerun,
  including the expanded heap parity cells, compiled shadows, and
  structural activation locks;
- full legacy schedule-generality file: **45 passed** (the legacy heap
  path is byte-unchanged under the helper extraction);
- byte gates: fresh 20-source corpus and 42-source grid captures from
  detached `554a1eb`-lineage and detached base `4abc8fa` worktrees are
  **byte-identical** (`diff -qr` empty).  The available heap audit
  generator's eleven cases regenerate byte-identically; the originating
  session's claimed separate nine-case addition was not reproducible
  from any retained generator and is superseded by §17;
- compiler latency: **not completed by the originating session**; the
  paired clean-tree review measurement in §17 supersedes this missing
  receipt;
- static parity: Black clean over every changed module and test (the
  single repo-level `black --check src` finding, `prebuilt_kernels.py`,
  is byte-identical at base and candidate and predates the milestone —
  an environment-level Black/Python parsing quirk in an untouched
  file); Flake8 diff between clean base and candidate worktrees is
  empty; focused mypy clean over the changed modules; clean detached
  base and candidate full-source `mypy --check-untyped-defs src` are
  **identical at 146 findings in 12 files, zero in `loopir/`** (the
  §15-recorded 140-in-11 clean-tree count did not reproduce in this
  session's environment on either side; equality between base and
  candidate is the binding no-new-findings gate);
- the authoritative clean detached-worktree non-performance suite at
  the final test tip `554a1eb` was **not completed by the originating
  session**; §17 records the clean-suite result at the corrected test
  tip instead;
- `git diff --check` clean before every commit; the five protected
  tracked files retain their recorded SHA-256 values and were never
  staged; staging used explicit pathspecs only; no GPU/CUDA, benchmark,
  packaging, scheduler, research, scratchpad, or tooling material was
  touched; live origin `refactor/compiler-ir-phase3-std-move-call`
  remains at `58e8565` and nothing was pushed.

### 16.6 Limitations and the candid Phase-6 exit verdict

- The migrated schedule surface is now: explicit complete loop orders,
  affine `accum="direct"` splits, stack workspace accumulation, sparse
  panel tiling, operand relayout/staging at both scopes, and heap
  result-tile accumulation for the rank-2 trailing-axis family — heap
  alone and the heap tile-ijk composition (pack + panel + relayout,
  both staging scopes) through the public Schedule → verified LoopPlan
  adapter, byte-locked.  **Phase 6 is therefore still not exited.**
- Still open, in the order the superseding prompt requires:
  target-independent abstract parallel-loop selection (beyond the
  panel/heap families whose row has no degrees of freedom), intended
  automatic-plan provenance through the typed schedule path (the
  blanket `unsupported_schedule_provenance` rejection stands), canonical
  LoopPlan schedule cache identity for the strangler route, and the
  criterion-by-criterion Phase-6 exit audit.  None of these were
  started in this session.
- The emitted heap slice is the originating implementation's rank-2
  subset of the audited rank>=2 legacy family; TTM-style multi-prefix
  heap plans stay fail-closed at the typed plan gate and continue to
  work through the legacy pipeline.
- The heap completion relies on the managed passes preserving the
  emitted result write and pack-origin header byte-for-byte (true
  today); a future pass that legally rewrites either fails loudly as
  `result_tile_completion_lost` — the §13/§15 snapshot posture.
- The strangler entry remains test/debug-only; production dispatch,
  release JIT, import neutrality, legacy stage sequences, and
  source-derived kernel cache identity are unchanged.

## 17. Heap result-tile rigorous review and soundness corrections (2026-07-25)

The inherited heap milestone was not sound as committed.  This review
audited commits `39ffa7a` through `554a1eb` independently, reproduced
concrete wrong results and incomplete target ownership checks, fixed
them in twelve focused commits, and did not start the remaining Phase-6
features:

- `95ba436` — `fix(compiler): close heap result-tile soundness gaps`
- `c6f465c` — `test(compiler): lock heap result-tile review fixes`
- `2a281b9` — `fix(compiler): reject opaque heap completion effects`
- `1b891d8` — `test(compiler): cover opaque heap completion effects`
- `3ebe0cd` — `fix(compiler): own heap result storage aliases`
- `b201e47` — `test(compiler): cover structured heap result aliases`
- `286fd84` — `fix(compiler): close rendered heap effect routes`
- `fad6c90` — `test(compiler): cover rendered heap effect routes`
- `9e0f770` — `fix(compiler): harden residual heap text fields`
- `99b667d` — `test(compiler): cover residual heap text fields`
- `59760ca` — `fix(compiler): finish heap text boundary audit`
- `f162ddf` — `test(compiler): lock final heap text boundaries`

Nothing was pushed, amended, squashed, or reordered.

### 17.1 Findings reproduced before the fix

1. **Verifier-approved repeating region lifetimes produced wrong
   results.**  A `ResultTileRegion` nested below the dense row loop
   verified, yet each row invocation reset and copied the entire prefix
   space.  A two-row probe returned `[[0, 0], [6, 8]]` instead of
   `[[3, 4], [6, 8]]`.  The same lifetime could repeat with the whole
   pack/region pair under another loop.
2. **Oracle copy-out confused physical and logical mode order.**  The
   verifier correctly allowed the pack dimension on the final physical
   level even when that level mapped to logical mode zero, but the
   oracle assumed the packed mode was logical rank minus one and failed
   on the verified permutation.
3. **Pass discovery walked verifier-invisible instance state.**
   `_collect_operand_loads` and `_collect_result_writes` traversed
   `vars()`, so a hidden alias could manufacture ambiguity and a hidden
   cycle could make the pass non-terminating even though canonical
   serialization, builder continuation, and verification all ignored
   that state.
4. **The “exact pre-replay” heap gate was not exact.**  It admitted
   wrong logical orders and panel placements, performed schedule
   rewrites, and rejected only after replay.
5. **Target completion fingerprinted an access, not the complete
   effect.**  A valid dense-matmul heap route failed because completion
   hard-coded the sparse row policy; changing the owning result
   assignment from `+=` to `=` passed and generated wrong code;
   metadata-free physical writes and nested result zeros survived; and
   malformed top-level LLIR could leak `AttributeError` instead of
   `result_tile_completion_lost`.  A final adversarial probe also proved
   that a raw `C_values[0] = ...` compatibility statement could hide a
   result effect from every structured ownership check.
6. **Structured Torch-storage aliases escaped the physical effect
   proof.**  A second `C_values_torch.data_ptr<float>()` alias could
   write through the result after copy-out; a direct
   `C_values_torch.zero_()` mutation was likewise accepted.
7. **Structured-node checks did not close codegen's verbatim fields.**
   Even after `RawStmt` and structured aliases were rejected, forged
   function-call names and `VarInit.op` could emit complete result
   mutations without creating a structured `C_values_torch` use.
   Independent follow-up probes found the same class in variable names
   and types, non-STRING literal text, comment line splicing, unary
   operators, OpenMP policy/schedule fields, and codegen-active loop and
   conditional flags.  Several equality-spoofing objects rendered
   different text from the value accepted by the completion check.

The review also corrected two evidence claims.  Repeating assignment
cannot be inferred from an all-ones copy-out differential because the
copy is idempotent; exactly-once coverage is now a structural oracle
invariant.  And the retained heap audit generator defines **11** heap
goldens, not a standalone 20-case heap generator; this review reports
only the cases it could actually regenerate.

### 17.2 Corrected contracts

- The verifier tracks exact statement ancestry and requires the region
  to be a direct statement of the root, outermost pack origin's body.
  Reset and whole-prefix copy-out therefore execute exactly once per
  pack origin.  The schema documentation states the same lifetime.
- Oracle copy-out traverses logical output axes and applies the clamped
  pack window wherever the final physical level maps.  It tracks copied
  compact keys and rejects duplicate coverage.
- Pass discovery now walks only declared dataclass fields, once per
  node identity.  Extra instance aliases and cycles are non-semantic,
  matching verification, canonical identity, and
  `LoopIRBuilder.resuming`.
- The heap gate pins the complete rank-2 family before replay:
  prefix/reduction/pack order for heap-alone and
  prefix/panel/pack plus `CHILD_OF(pack OUTER)` for the composed route.
- Completion derives the expected row policy by applying the shared
  structured parallel marker to a detached snapshot.  It validates the
  entire assembled LLIR first, then requires the exact additive
  assignment owner, sole metadata write, canonical ABI result-pointer
  declaration, canonical top-level generated zero, and no other
  physical result occurrence.  It also reconstructs and matches the
  canonical Torch allocation and terminal result assembly, then permits
  exactly the three owned `C_values_torch` occurrences: allocation,
  canonical `data_ptr`, and final storage transfer.  Because opaque
  text cannot take part in such an effect proof, completion now validates
  every LLIR field emitted directly as C++ before reasoning about
  ownership: exact ASCII variable/function/call names and DataTypes,
  non-mutating unary and initializer operators, numeric-only raw
  literal text, non-splicing one-line comments, structured-only
  statements, exact codegen-active flags, and token-bounded OpenMP
  policies over declared non-result identifiers and the sole
  compiler-owned grain macro.  `RawStmt`, nested/effectful policy text,
  string subclasses, unknown macros, and result-owned policy names fail
  closed.  Removal is followed by a residual effect check; every
  malformed, moved, duplicated, hostile, opaque, or aliased case is
  owned as `result_tile_completion_lost`.
- The compiled matrix now includes the previously omitted dense-matmul
  heap route, source-byte-identical to legacy and bitwise-equal in a
  real shadow execution.

### 17.3 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-heap-review-f162ddf/`
(receipts at `phase6-heap-review-c6f465c/` and
`phase6-heap-review-1b891d8/` and
`phase6-heap-review-b201e47/` are retained as predecessor evidence).

- exact five-file Phase-6 contract focus
  (`test_loopir_verifier.py`, `test_loopir_schedule_passes.py`,
  `test_loopir_llir_lowering.py`, `test_loop_plan.py`,
  `test_schedule_api.py`): **509 passed**;
- supplementary verifier/printer/oracle/pass/target membership:
  **486 passed**;
- scheduled compiled runtime focus at exact final code/test tip `f162ddf`
  (`test_loopir_scheduled_slice.py` plus
  `test_loopir_pipeline_execution.py`): **177 passed** in 754.24 s;
  this includes all **16** activating heap source-parity and
  compiled-shadow cases;
- full legacy schedule-generality file: **45 passed** in 195.57 s;
- fresh **20-source corpus** and **42-source grid** are byte-identical
  across clean detached `554a1eb`, `b201e47`, and exact-final
  `f162ddf` worktrees and to the retained heap captures.  Manifest
  SHA-256 values are respectively
  `7e4a9c436e5ed1005874e9ece56847ea4ef88dd6f5a04c4d657c3ca5b37cd6c4`
  and
  `65e68ba19510ab240cf85574aa6be272dd6e5b0fdd7e799f7bcfe9db6d06c094`;
- all **11 available heap audit goldens** regenerate byte-identically
  at all three clean revisions and against the retained copies (manifest
  `aa8be2229bea130833461d6dcf789722837f1a100f2da2a74de9cc4b2dcb4f72`);
  the combined 73-source manifest is
  `b78b355ed9b6d6a39cabc3912269312edd98ea144dc12fe79ec37521fe407f5b`;
- Black and Flake8 pass both newly changed Python files.  Focused
  production mypy is clean; the two-file production/test invocation
  preserves 26 byte-identical inherited test findings against
  `b201e47`.  Clean-tree full-source base/candidate Flake8 logs are
  byte-identical at nine inherited findings, and mypy logs are
  byte-identical at 146 inherited findings in 12 files, with zero new
  LoopIR findings;
- paired same-machine compiler latency at exact detached `554a1eb`
  versus `f162ddf` revisions (5 warmups/30 samples) is inside the 1.10
  threshold for every case and percentile: `small_dense`
  **0.977/0.965**, `reduction` **1.008/0.941**,
  `csr_intersection` **1.009/1.032**, and `sparse_union`
  **1.010/0.958** (p50/p95 candidate/base; worst **1.032**).
  Every base/candidate generated-source hash is identical.  A first
  launch with the wrong working directory was rejected by the
  provenance guard before producing samples or JSON; the retained
  second invocation is the complete successful comparison;
- authoritative clean detached-worktree non-performance suite at exact
  test commit `f162ddf`: **3,983 passed, 14 skipped, 3
  performance-deselected, one known warning, zero failures/errors,
  exit 0** in 2,763.48 s (46:03), with import provenance asserted and
  fresh isolated Torch/XDG/pytest caches.  The log and JUnit SHA-256
  values are
  `7d3516e28d86386f0dcfad73b94a12c49f9a509be0de45f9a38dd189a6f9c89c`
  and
  `055d8ad40c30a262020dd28063326bdbdfb45e4c9b2708ba8c428883e72e7552`;
- `git diff --check` is clean, all five protected tracked files retain
  their recorded hashes, and unrelated GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, and tooling material remains
  untouched.

### 17.4 Scope verdict after review

The corrected rank-2 heap slice is sound under the gates above, but
**Phase 6 remains open**.  The originating audit captured a live
rank-3 TTM multi-prefix heap kernel while the typed pass deliberately
rejects every multi-prefix result; that is an explicit incomplete
boundary, not a completed general heap milestone.  Multi-prefix heap,
target-independent parallel selection, intended automatic-plan
provenance, canonical LoopPlan cache identity, and the
criterion-by-criterion Phase-6 exit audit remain.  Phase 7 must not
start until those Phase-6 decisions are closed or the design criteria
are formally revised.

## 18. Independent re-review of §17 and the multi-prefix heap milestone (2026-07-25)

This session independently audited the twelve §17 correction commits plus
the §17 documentation commit (`95ba436` … `b21d78e`), reproduced all four
focused gates, found and fixed one concrete contract gap, and implemented
the identity-storage multi-prefix heap slice §17.4 left open.  A later
rigorous review found additional gaps; §19 supersedes the completion and
evidence claims here.  Four focused commits:

- `3b1611f` — `fix(compiler): own heap binary-operator text`
- `58a22b5` — `test(compiler): cover heap binary-operator text`
- `52c79fe` — `feat(compiler): generalize heap result tiles to the multi-prefix family`
- `3f17ec6` — `test(compiler): lock the multi-prefix heap result-tile family`

Nothing was pushed, amended, squashed, or reordered.  Origin remains
`58e8565`; all work is local-only.

### 18.1 What the re-review confirmed

The §17 corrections were re-derived from the code, not accepted from the
report.  Each of the seven §17.1 findings has a real fix in the tree:

1. **Lifetime.** `_check_result_tile_region` tracks exact statement
   ancestry through `_Context.statement_stack` (object identity, not
   lexical paths) and requires ancestry `[root Block, pack origin,
   pack.body, region]`.  Any intervening or enclosing repeating scope —
   including a region wrapped one Block deeper, or a pack origin nested
   under another loop — makes the ancestry length wrong and fails closed
   with `result_tile_scope_mismatch`.
2. **Permuted copy-out.** The oracle's `copy_out` walks logical output
   axes, applies the clamped pack window wherever `levels[-1].mode`
   lands, and records copied compact keys to reject duplicate coverage.
   Compact keys are built in logical mode order on both sides
   (`_exec_tiled_reduce` and `copy_out`), so they agree by construction.
3. **Schema-only discovery.** `_walk_schema_nodes` walks declared
   dataclass fields once per node identity, via
   `object.__getattribute__`; extra `__dict__` aliases and cycles are
   ignored and cannot hang or double-count.
4. **Pre-replay admission.** The heap gate pins the family before replay.
   Independently re-derived: `loop_order[-1] == tile_loop` plus the
   heap-only length/prefix constraint leaves exactly one position for the
   reduction, so the "exact order" claim holds rather than only pinning
   position 0.
5–7. **Completion ownership.** The assembled function is validated
   before any name discovery; the row policy is derived structurally from
   a detached snapshot; and ownership requires the exact additive
   assignment owner, the sole metadata write, the canonical ABI pointer
   declaration, the canonical top-level zero, the reconstructed Torch
   allocation and terminal assembly, and exactly the three owned
   `C_values_torch` occurrences.

The common `LLIRWalker` was independently confirmed to dispatch on exact
type (`type(node) not in SUPPORTED_…` → `unknown_llir_node`), so node
subclasses cannot slip past the per-type checks the completion adds, and
`_validate_literal_fields` already rejects `str` subclasses.

Eight adversarial probes were run beyond the retained matrix: a Torch
storage alias through `AddressOf(&C_values[0])`, a Torch member
expression, a `MemberCallStmt` mutation, a `QualifiedName`, a hostile
statement in `ForLoop.pre_parallel_body`, one in `_hoisted_ptr_decls`, a
`GuardedCallStmt` prefetch of the result, and a forged `BinOp` operator.
All but the last were already owned as `result_tile_completion_lost`.

### 18.2 The one gap found: `BinOp.op`

`BinOp.op` is interpolated into the emitted C++ verbatim
(`f"{left} {binary.op} {right}"`), exactly like the unary operator beside
it, and the common walker constrains it only to "a non-empty string" —
`Add` and `Mul` are pinned to `+`/`*`, a plain `BinOp` was not.  A forged
exact-`str` operator
`"+ 0; C_values_torch.zero_(); int decoy2 = 0 +"` inserted at function
scope passed completion's ownership proof untouched and surfaced only
later as an unowned `CodegenError` from the codegen precedence table.

This is a **diagnostic-ownership gap, not a soundness gap**: codegen's
closed `_BINARY_PRECEDENCE` table rejects every unsupported spelling, so
no wrong C++ could be emitted through this route.  But §17.2's claim that
completion "validates every LLIR field emitted directly as C++" was not
yet true.  Completion now validates `BinOp`/`Add`/`Mul` against the exact
set of non-mutating binary spellings, mirroring the
`_RESULT_TILE_UNARY_OPERATORS` precedent (which likewise excludes the
mutating `++`/`--` codegen accepts elsewhere).

The two new matrix members are honest about what each proves:
`opaque_binary_operator` **fails without the fix** and is the regression
lock; `binary_operator_subclass` passes both before and after (the
walker's exact-`str` requirement already covered it) and is retained as a
checked property.

To keep the growing audit under the repository `max-complexity` limit,
the per-node validator moved from a closure-nested class to module-scope
`_ResultTileTextValidator` with one method per node type.  Dispatch is
still exact-type, never `isinstance`.

Two §17 evidence claims were re-checked and stand: the 11 retained heap
goldens regenerate byte-identically, and repeated-assignment coverage is
a structural oracle invariant rather than a numeric inference.

### 18.3 Multi-prefix heap: originating identity-storage slice

§17.4 recorded that the originating audit captured a live rank-3 TTM
multi-prefix heap kernel while the typed pass rejected every multi-prefix
result.  This slice admitted the identity-storage form structurally rather
than special-casing the TTM spelling.  It did **not** correctly preserve the
distinction between physical storage-prefix order and logical tensor-access
order; §19 records and fixes that defect.

The family derives from one number — the result tile's dense prefix rank —
admitted at a single point.  Chain length, prefix loop positions, compact
linearization, and copy-out extent are computed from it.  At this revision,
however, `_result_tile_prefix_rank` equated `result_prefix` (a physical
storage-level fact) with `access_indices[:-1]` (a logical-mode fact), and
the pass and target repeated that assumption.  Identity-storage programs
therefore worked, while a verifier-valid permuted level order was rejected
for the wrong reason.  §19 replaces that admission rule with an explicit
level-to-mode mapping.

**No change was needed in LLIR completion, the oracle, erasure, or the
verifier.**  The compact linearization already used the last prefix
level's position variable (`p{result}{last_level-1}`) and the product of
every prefix extent; the oracle's copy-out already walked logical output
axes; erasure is rank-agnostic; the verifier already required rank >= 2.
Generalizing admission was sufficient — which is itself evidence that the
§17 corrections were the right shape.

Proved for the rank-3 representative:

- **Byte-identical legacy C++.** The LoopIR route reproduces the retained
  `heap_ttm_multi_prefix` golden byte-for-byte (5,160 bytes), and the
  public `Schedule` route is byte-identical to the production legacy
  route across nine grid members: exact/ragged/unit/oversized strips,
  f32/f64, and zero outer-prefix, inner-prefix, reduction, and free
  extents.
- **Once-per-pack reset/copy-out and race legality.** The verifier's
  ancestry rule is rank-independent.  Every compact cell is addressed by
  the linearized dense prefix position and the pack point, so distinct
  iterations of any prefix loop write disjoint cells and the reduction
  loop is enclosed by all of them.
- **Oracle differential.** A five-width randomized differential proves
  scheduled == erased == an independent reference over randomized shapes
  and sparsity; erasure restores the reordered base dump exactly.
- **Compiled shadows.** Five compiled cases (strips 1/3/4/64 in f32 plus
  f64, including an empty stored segment) require bitwise-equal dense
  results against legacy and PyTorch agreement.
- **Public routing.** The artifact carries `result_level == 2` and a
  two-element `result_prefix`.

### 18.4 Two boundaries deliberately left closed, with reasons

These are recorded as **criterion boundaries, not completions**:

1. **The composed sparse panel keeps its audited single-prefix shape.**
   Legacy *does* admit a rank-3 heap+panel chain, but only with the CSR
   dense-parent row (`b`, the **inner** prefix) as `parallel_loop`; with
   `a` it raises `InvalidSchedule`.  The LoopIR target still derives its
   parallel row as a fixed chain position, so it cannot express that
   anchor.  A multi-prefix panel plan now fails closed with
   `invalid_schedule_result_tile`.
2. **The heap parallel anchor is pinned to the outermost dense prefix
   loop.**  Legacy accepts any prefix loop and emits materially different
   source per anchor — measured on the rank-3 TTM kernel, anchor `a`
   gives `scorch_nthreads(-1, Core0_size)` at 5,160 bytes and anchor `b`
   gives `scorch_nthreads(Core2_pos[Core1_size], Core1_size)` at 5,200
   bytes.  Accepting an inner-prefix anchor without an abstract selection
   would silently parallelize a loop the plan does not name, so it is now
   rejected instead.  **The rank-2 family is unaffected** — a single
   prefix has no degrees of freedom, so this pin is exactly the previous
   `in result_prefix` test there.

Both boundaries are lifted by the same next milestone: abstract
parallel-loop selection.

### 18.5 Design constraints established for abstract parallel-loop selection

Recorded here because they were derived by measurement this session and
materially change how that milestone must be built:

- **LoopIR currently carries no parallel fact at all.**  The target
  derives it structurally in three places: the generic rule
  (`result_is_dense and outer_in_result` → mark the first for-loop, with
  a merged-nest special case that deliberately passes an empty statement
  list so the applied policy is the row-count-only
  `scorch_nthreads(-1, rows)` form), the panel route (chain position 2),
  and the heap route (`result_tile_row_position`).
- **`lower_loopir_to_llir` takes a bare `LoopProgram`, not the scheduled
  artifact**, and several tests assert direct structural activation on a
  bare verified program.  The next representation audit must decide whether
  parallel selection is intrinsic LoopIR, a verified target-analysis side
  table passed explicitly to lowering, or another typed carrier that preserves
  bare-program activation.  The evidence rules out silently storing the fact
  only on `ScheduledLoopIR`; it does not by itself prove that adding a schema
  node is the sole valid design.
- **`_loop_key(node) -> (IndexId, LoopPart)` already exists** in
  `schedule_passes` and is the stable loop identity such a fact should
  name.
- **The byte-parity constraint is the hard part.**  For every already
  migrated route with no explicit plan fact, the abstract selection must
  reproduce legacy's implicit rule exactly, or the 20-source corpus, the
  42-source grid, and the 11 heap goldens all move.  Either the fact is
  optional and the target keeps its derivation when absent (routes opt
  in), or every route gains the fact at once and the derivation is
  deleted — the second is cleaner but must be proved against all three
  capture sets in one step.

### 18.6 Verification

Evidence retained under
`/Users/bobby/.cache/scorch-codex/phase6-multiprefix-3f17ec6/`.

- exact five-file Phase-6 contract focus (`test_loopir_verifier.py`,
  `test_loopir_schedule_passes.py`, `test_loopir_llir_lowering.py`,
  `test_loop_plan.py`, `test_schedule_api.py`): **519 passed** (509
  inherited + 2 binary-operator + 8 multi-prefix);
- the originating report claimed **308** supplementary tests, but did not
  retain the exact commands/logs and the named six files collect only 170;
  this count is therefore withdrawn rather than treated as evidence;
- scheduled compiled runtime focus (`test_loopir_scheduled_slice.py`
  plus `test_loopir_pipeline_execution.py`): **192 passed** in 738.41 s
  (177 inherited + 15 multi-prefix), including all activating heap
  source-parity and compiled-shadow cases;
- full legacy schedule-generality file: **45 passed** in 183.35 s, both
  before and after the milestone;
- all **11 available heap audit goldens** regenerate byte-identically at
  clean detached `b21d78e` and `3f17ec6` and against the retained
  `phase6-heap-review-f162ddf` copies; manifest SHA-256
  `7072ca461f94c3268884de76ba3aaaa434745f25f3a79cc69787fb7f7cd76df7`;
- fresh **20-source corpus** and **42-source grid** are byte-identical
  across clean detached `b21d78e` and `3f17ec6`; manifest SHA-256 values
  `e240b53cf646f8380433e64d3bdfb534833ea9480eda447e58a2f44780bc9b0c`
  and
  `3d48634508bbb54c3d2b16578ecb52ef96c6934c2a8642eaf8f03498037d5024`;
- clean-tree full-source base/candidate mypy logs are **byte-identical**
  at 146 inherited findings in 12 files over 60 source files, and flake8
  logs are **byte-identical** at nine inherited findings.  Black over
  `src` reports exactly one file, `src/scorch/prebuilt_kernels.py`, at
  **both** revisions — an inherited finding this milestone neither
  introduced nor touched; every changed `loopir/` file is Black-clean;
- paired same-machine compiler latency at clean detached `b21d78e`
  versus `3f17ec6` is inside the 1.10 threshold for every case and
  percentile: `small_dense` **0.968/1.038**, `reduction`
  **0.992/0.928**, `csr_intersection` **1.006/1.017**, `sparse_union`
  **0.997/0.992** (p50/p95 candidate/base; worst **1.038**);
- no green literal full-suite result exists at `3f17ec6`: two unpartitioned
  attempts reached 4,007 passes and then aborted in a later unrelated COO JIT
  test with macOS libomp `OMP: Error #179` / `pthread_key_create failed`.
  The clean `b21d78e` control passed 3,983 tests; the 25 added tests moved the
  same long-lived Python process past that process-local resource ceiling.
  A later non-overlapping partition passed, but is not represented as an
  authoritative full-suite substitute.  §19 isolates the added native tests
  and reruns the literal suite.
- `git diff --check` is clean, all five protected tracked files retain
  their recorded hashes, and unrelated GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, and tooling material remains
  untouched.

### 18.7 Scope verdict

**Phase 6 remains open.**  This revision proves the identity-storage
multi-prefix heap representative, not the whole rank>=2 level-permutation
family.  The physical/logical admission correction and two boundaries of
§18.4 are recorded in §19.
Abstract parallel-loop selection, intended automatic-plan provenance,
canonical LoopPlan cache identity, and the criterion-by-criterion Phase-6
exit audit remain.  Phase 7, selector integration, production cutover,
and legacy deletion must not start until those are closed or the design
criteria are formally revised.

### 18.8 Limitation recorded outside Phase 6

`STensor.to_sparse("dds")` produces a malformed rank-3 storage: the
compressed leaf's position array is sized for the innermost parent only
(5 entries where 13 are required for a 3x4 dense prefix), so the
conversion raises `TensorIndexError` on its own output.  This is a
pre-existing defect in the public conversion path, independent of the
compiler refactor; it was found because the multi-prefix shadow test
needed a `dds` operand.  That test builds the storage directly and says
so; the defect is **not** fixed here and remains open.

## 19. Rigorous review of the multi-prefix heap slice (2026-07-25)

The next session treated §18 as an untrusted handoff, audited the four
commits and retained evidence independently, reproduced the literal
full-suite failure, and found two correctness defects plus two evidence
gaps.  The corrections are:

- `4234a95` — `fix(compiler): close multi-prefix heap review gaps`
- `858c898` — `test(compiler): lock multi-prefix heap review fixes`
- `2fe8fa3` — `fix(compiler): type narrow heap binding dispatch`
- `539784f` — `refactor(compiler): retire flat heap name inventory`

Nothing was pushed, amended, squashed, or reordered.  The five protected
tracked files retained their session-start hashes and every unrelated
GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, and
tooling path remained untouched.

### 19.1 Physical storage order is not logical access order

The central §18 admission rule was wrong for the level-based model.
`ResultTile.result_prefix` names the leading **physical storage levels**;
`ResultTile.access_indices` names tensor coordinates in **logical mode
order**.  Equating `access_indices[:-1]` with `result_prefix` works only
when every `LevelDecl.mode` is the identity permutation.

An adversarial rank-3 result with physical modes `(1, 0, 2)` made the
problem concrete: the canonical physical prefix is `(b, a)`, while the
logical access remains `(a, b, d)`.  The old rank gate rejected the valid
fact, and a differently forged fact could satisfy that gate without
describing the result's physical prefix.  Production target lowering
still rejected every non-identity tensor globally as
`unsupported_mode_order`, so this did not silently emit a wrong kernel,
but the pass contract and §18's general-family claim were false.

The correction keeps `_result_tile_prefix_rank` responsible only for
declaration-independent arity.  `apply_result_tile` now maps every
logical access coordinate through the output declaration's
`LevelDecl.mode` values, proves that the resulting physical prefix equals
`result_prefix`, proves that the final physical level is the packed loop,
and matches the loop chain in physical order.  Target shape validation
uses the same mapping.  A complete pass-level regression constructs the
`(1, 0, 2)` output, reorders the loop chain to `(b, a, c, d)`, applies the
heap plan, and verifies the scheduled artifact.  The C++ target's
pre-existing global non-identity-layout refusal remains explicit and is
not misrepresented as migrated support.

### 19.2 Heap completion now owns declarations and lexical binding

The assembled-function ownership proof validated text spellings and exact
result occurrences, but did not own C++ declaration identity.  A managed
pass could insert a nested declaration of `kTile_d`; completion would
then advance the pack origin by the canonical width while compact
initialization, compute, and copy-out read the shadowing width.  Likewise,
a nested fixed array named `Projected_values` could shadow the canonical
result pointer.  Both mutations passed completion before this review and
can emit valid C++ with wrong result coverage.

The old flat `known_names` set was also not a binding model: it contained
every `Var` occurrence, not declarations.  Completion accepted an
undeclared variable, a use before its later declaration, a branch-local
declaration used after the branch, and a nested `llir.Function` that
codegen rendered as an invalid function definition inside `evaluate`.
It also counted declarations in non-emitted `pre_parallel_body` or
`_hoisted_ptr_decls` compatibility fields.

The review adds two heap-specific, fail-closed layers before any compact
rewrite:

1. function-wide declaration ownership across arguments, ordinary/direct
   declarations, fixed arrays, loop variables, and codegen-synthesized
   atomic names; and
2. emission-order lexical resolution across sequential statement lists,
   branches, ordinary loops, range loops, while loops, split OpenMP
   regions, and atomic work-stealing scopes.

The no-shadow rule is intentionally stronger than general C++ and remains
local to the heap soundness proof.  An independent inventory of all 24
admitted heap parity cells found globally unique declarations in every
case (39 dense, 41 SpMM, 46 rank-3 TTM, 46 panel, and 51 relayout
declarations, each count fully unique), with no unresolved or out-of-scope
variable.  Common LLIR/codegen may eventually gain a general lexical
verifier, but that rollout is separate: the completed legacy-compatible
heap helper groups still carry typed-to-be-migrated pseudo-`Var` text for
`0.0f` and prefix products.

Regression coverage now includes both semantic shadowing cases,
undeclared and use-before-declaration variables, branch leakage, nested
functions, later-only policy declarations, and declarations stranded in
non-emitted compatibility fields.  Every mutation is owned as
`result_tile_completion_lost`.

### 19.3 Missing zero-extent semantics and the macOS full-suite regression

§18's nine source-parity cells included four zero extents, but its oracle
randomization sampled only positive dimensions and all five compiled
shadows were nonzero.  The review adds independent scheduled/erased/reference
oracle checks and compiled LoopIR/legacy/PyTorch shadows for zero outer
prefix, zero inner prefix, zero reduction, and zero free extent.

The literal `3f17ec6` suite failure was not random machine load.  Both
unpartitioned attempts aborted after 4,007 passes with macOS libomp
`pthread_key_create failed`; the clean predecessor completed 3,983 tests.
The five new rank-3 JIT shadows had raised the number of native extensions
loaded by one long-lived pytest process past a process-local pthread-key
ceiling.  Partitioning the suite proved test behavior but did not make the
canonical gate green.

The corrected tests execute each added rank-3 native differential in a
short-lived subprocess.  This preserves the exact compiled comparison,
keeps the pytest parent at its inherited native-resource footprint, and
lets the literal unpartitioned suite remain the authoritative gate.  Four
compiled zero-extent cases are isolated by the same mechanism.

### 19.4 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-multiprefix-review-539784f/`
(with the superseded `858c898` static audit retained separately).

- exact five-file Phase-6 contract focus: **533 passed** (the §18 total
  plus 14 review regressions);
- direct targeted native proof during repair: one ordinary rank-3 shadow
  plus the zero-reduction shadow, **2 passed**;
- the scheduled runtime and literal full-suite reruns at `539784f` were
  superseded when the independent final audit found the §20 blockers;
- 20-source corpus, 42-cell grid, and 11 heap audit goldens:
  **73/73 byte-identical** between clean `3f17ec6` and `539784f`
  worktrees (combined manifest SHA-256
  `b78b355ed9b6076c336141180c6784c36f83fe359712796162174d8343a05d2b`);
- changed-file and full-source static comparison:
  all five changed Python files Black/Flake8-clean; focused mypy clean;
  full-source Black reproduces only the inherited
  `prebuilt_kernels.py` finding, Flake8 is exactly nine inherited
  findings, and full-source mypy is normalized byte-identical at
  **146 findings in 12 files**;
- no latency result is claimed for the superseded `539784f` candidate;
- the literal unpartitioned suite at `539784f` was stopped while healthy
  at roughly 35 percent after the §20 review made that candidate stale;
- `git diff --check` clean; live origin and protected hashes unchanged.

### 19.5 Candid scope verdict

The identity-storage rank-3 TTM path is byte-compatible, and the scheduling
contract now remains correct for a physical mode permutation even though
the current C++ target deliberately refuses that broader layout family.
Arbitrary positive prefix rank is derived rather than enumerated; the
retained rank-4 source probe is useful supporting evidence, not a claim
that every level permutation compiles.

**Phase 6 remains open.**  Abstract parallel-loop selection, the
multi-prefix panel and inner-prefix heap anchors it unlocks, intended
automatic-plan routing, canonical `LoopPlan` schedule identity, and the
criterion-by-criterion exit audit remain.  Phase 7 must not begin unless
those Phase-6 criteria are genuinely closed or the design is explicitly
revised.

## 20. Final adversarial review of heap completion (2026-07-25)

An independent read-only audit of the §19 corrections found that the new
validation model and the generated-name allocator still had different
views of the emitted function, and that lexical validity alone did not
close result-state effects.  The stale `539784f` full-suite run was
stopped immediately; no receipt from that run is represented as final.
The corrections are committed, local-only, as:

- `909fb91` — `fix(compiler): close heap completion ownership gaps`
- `a24c0c1` — `test(compiler): cover heap completion ownership boundaries`

### 20.1 Generated declarations and exact emission order

`_heap_result_tile_names` still consulted the legacy `_declared_names`
inventory.  That inventory does not see declarations in flattened nested
statement containers, `before_parallel_body` / `pre_parallel_body` /
`post_parallel_body`, or codegen-synthesized atomic names.  The §19
validator did see those declarations, but threw its complete inventory
away.  A retained pre-parallel declaration or atomic counter named
`tiled_C` could therefore pass validation and collide with the compact
pointer selected moments later, producing invalid C++ or shadowing the
storage used by the rewritten result update.

The validator now returns its exhaustive declaration set after the
target-owned parallel marking step, and the allocator unions it with the
legacy inventory before choosing every compact/init/copy spelling.
Regression tests cover flattened containers, split OpenMP declarations,
atomic counters, and the counter's use in `omp_num_threads`.

The emission-order model also now matches three codegen boundaries it
previously approximated:

- `break` and `continue` are legal only in an actual loop body, not at
  function scope or in pre/post parallel regions;
- `IfThenElse` mirrors codegen's required condition/body pairing and
  condition-list/body-list cardinality; and
- an atomic counter is visible to the parallel policy, `_start` is visible
  to the loop-bound expression, and neither synthetic name is visible
  before codegen declares it.

### 20.2 Result-state effects, not only variable visibility

The §19 binding proof resolved names but did not constrain what a valid
statement did with them.  In particular, an assignment to `C0_size`, an
extra increment of `pC1` beside the result write, or
`result_shape.clear()` passed every check and could compile into a
silently incomplete or misallocated result.  The actual mutable ABI
argument is named `result_shape`; the earlier protected set accidentally
named only a nonexistent `C_shape`.

Heap completion now protects the real shape argument plus result extents,
positions, storage, capacity, assembly object, and tensor.  It rejects
unowned assignments/increments, mutating member calls, address escape,
`std::move`, guarded calls, and unknown calls receiving protected state.
The allowlist is deliberately structural and narrow:

- the unique metadata-fingerprinted result `StoreReduce` write;
- a loop's own declared result-position induction update;
- the already fingerprinted terminal result assembly;
- `scorch_native::validate_jit_result_shape`;
- the exact dense zero later removed by completion;
- `torch::empty` over the validated scalar capacity; and
- the canonical tensor `data_ptr` acquisition.

Tests inject each mutation before the declaration, loop, copy, or
assembly operation it would corrupt, rather than relying on a later C++
compile failure.

### 20.3 Deliberately recorded broader LLIR debt

Two adversarial observations are not silently promoted into claims:

1. the heap lexical environment binds C++ names, not full C++ types.
   Existing valid LLIR intentionally spells the same variable with
   `NO_TYPE`, pointer/restrict, `INT`/`INT64`, and
   `CONSTEXPR_INT`/`INT` metadata at different uses.  Exact signature
   equality would reject production.  A future general LLIR type checker
   must classify use form and declaration category rather than compare
   `Var` objects literally;
2. legacy OpenMP policy text is token-, delimiter-, name-, and
   effect-checked but is not parsed as a complete C++ expression grammar.
   Malformed compiler-owned arithmetic can therefore remain a compile
   failure.  It cannot inject punctuation, nested calls, result-owned
   names, or an unbound identifier through this boundary.

Neither limitation permits the result-state mutations or name collisions
closed above.  They remain explicit compatibility debt for the eventual
typed-policy/general-LLIR validation work.

### 20.4 Verification

Final evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-multiprefix-review-a24c0c1/`.

- exact five-file Phase-6 contract focus: **548 passed**;
- exact scheduled runtime focus at `a24c0c1`: **196 passed** in
  **784.97 s** (log SHA-256
  `a2f8270152542211d1a4b9562626a2b2db53cbc3629175bdbfbfe02da44241f6`,
  JUnit SHA-256
  `9141da71af719553eb997d53e01ee72091854a2ce70a7a34e662889c07acf8b7`);
- fresh 20-source corpus, 42-cell grid, and 11 heap audit goldens:
  **73/73 byte-identical** between clean `3f17ec6` and `a24c0c1`
  worktrees (combined manifest SHA-256
  `b78b355ed9b6076c336141180c6784c36f83fe359712796162174d8343a05d2b`);
- changed-file and full-source static comparison: all six changed Python
  files are Black/Flake8-clean and the three production files are
  focused-mypy-clean; full-source Black reproduces only the inherited
  `prebuilt_kernels.py` finding, Flake8 is exactly nine inherited
  findings, and normalized full-source mypy is byte-identical at
  **146 findings in 12 files**;
- paired same-session compiler latency (5 warmups / 30 samples) is inside
  the 1.10 threshold for every category and percentile: `small_dense`
  **1.008/1.032**, `reduction` **1.025/1.011**,
  `csr_intersection` **1.018/1.000**, and `sparse_union`
  **1.015/0.989** (p50/p95 candidate/base; worst **1.032**), with
  identical generated-source hashes;
- literal unpartitioned clean detached-worktree non-performance suite at
  `a24c0c1`, with isolated caches and asserted import provenance:
  **4,041 passed, 14 skipped, 3 performance tests deselected, one known
  sparse-invariant warning, and zero failures** in **2,723.92 s**
  (log SHA-256
  `49cf58338ee10f625de26c49f26541a5717769e157b94dca8d0a134715c7da88`,
  JUnit SHA-256
  `a05b3487c3bc2bd03dbfe188432690de9b423a7c0613864e326faebd63ebe7d8`);
  the literal process crossed the former 4,007-pass libomp abort region
  and exited with no lingering compiler or test process;
- `git diff --check` clean; import provenance, live origin, and all five
  protected hashes verified; no unrelated tracked or untracked material
  staged.

### 20.5 Scope verdict

The final verdict remains intentionally narrow: identity-order rank-3 TTM
and the existing rank-2/panel/relayout heap families retain byte parity;
arbitrary positive prefix rank is derived rather than enumerated; and the
schedule pass represents physical mode permutations while the current
C++ target globally rejects unsupported nonidentity layouts.

**Phase 6 remains open.**  The next work is still the broad sequence of
target-independent parallel-loop selection (including inner-prefix and
rank-3 panel anchors), intended automatic-plan routing, canonical
`LoopPlan` schedule identity, and the criterion-by-criterion exit audit.

## 21. Abstract parallel-loop selection milestone (2026-07-26)

This session first re-reviewed the inherited multi-prefix heap state
(§§18–20) without trusting its reports, closed the concrete gaps that
review found, and then implemented the first §20.5 milestone:
target-independent abstract parallel-loop selection, lifting both §18.4
boundaries with measured legacy byte parity.  Ten focused commits,
nothing pushed, amended, squashed, or reordered:

- `f46c5c9` — `fix(compiler): pin heap allowlist calls and member-call arguments`
- `1eefb2d` — `test(compiler): lock heap allowlist pins and review evidence gaps`
- `f4e459c` — `feat(compiler): declare the abstract parallel-selection schema`
- `0b75481` — `test(compiler): lock the abstract parallel-selection schema`
- `c9dca93` — `feat(compiler): consume and realize explicit parallel selection`
- `4cf57f1` — `test(compiler): lock explicit parallel-selection consumption`
- `758ae63` — `test(compiler): lock explicit-anchor parity, activation, and completion`
- `6242da8` — `fix(compiler): restrict explicit anchors to measured loop kinds`
- `533de08` — `test(compiler): re-anchor fail-closed vehicles on parallel tiles`
- (docs commit follows this review)

### 21.1 Independent review of §§18–20 and the gaps it closed

The inherited gates were reproduced exactly at clean detached `5ef0401`:
the 548-test contract focus, the 196-test compiled focus, all 73
corpus/grid/heap captures byte-identical to the retained
`phase6-multiprefix-review-a24c0c1` finals, the static baselines
(one inherited Black finding, nine Flake8 findings, 146 mypy findings in
12 files), and the literal unpartitioned full suite (4,041 passed,
14 skipped, 3 deselected, 1 warning), matching the §20.4 receipts.
Adversarial probing (31 pass-level probes, 28 completion-level probes,
and a claim-by-claim test-evidence audit) confirmed the §19.1
physical/logical correction, the declaration/lexical ownership model,
and the §20.2 effect boundary — and found:

1. **A real allowlist hole (fixed).**  The binding validator admitted
   `scorch_native::validate_jit_result_shape` by name alone.  A forged
   wrong-arity call over the protected `result_shape` argument rode the
   allowlist into non-compiling C++ that surfaced as an unowned JIT
   failure, and an exact duplicate of the canonical validation compiled
   cleanly while changing runtime behavior.  Completion now pins the name
   to exactly one canonical ABI-generated occurrence at function scope,
   structurally matched against a freshly reconstructed
   `kernel_abi().emit_validation()` statement — the `scorch_zero_dense`
   precedent.  Locks: `forged_result_shape_validation` and
   `duplicate_result_shape_validation`, both failing without the fix.
2. **Member-call arguments escaped the effect boundary (fixed).**
   Function, expression, and guarded calls checked their arguments for
   protected uses; `MemberCall`/`MemberCallStmt` arguments were only
   bound for visibility, so an unknown member call on an unprotected
   receiver could receive protected result state (a potential
   non-const-reference mutation route).  The text validator now rejects
   protected uses in member-call arguments; the canonical owned
   `data_ptr` acquisition carries no arguments, so no production spelling
   changed.  Locks: `member_call_protected_argument` and
   `member_call_expression_protected_argument`, both failing without the
   fix.
3. **Evidence gaps (locked).**  §20.2's guarded-call rejection had no
   test (`guarded_call_protected_argument`, a checked property); the
   atomic emission-order model had only positive coverage
   (`test_heap_binding_rejects_atomic_names_before_declaration` covers
   both hidden names in the negative direction); the rank-3 kernel had no
   mutation-matrix members of its own
   (`test_multi_prefix_heap_completion_owns_protected_state`, four
   mutations); §19.5's "retained rank-4 source probe" citation was
   dangling (no such test existed —
   `test_rank_four_heap_plan_derives_three_prefix_loops` now supplies the
   evidence); and no test executed a non-identity-layout heap program
   (`test_rank_two_nonidentity_heap_oracle_differential_is_exact` runs a
   (1, 0) physical rank-2 result whose packed axis is logical mode zero
   at four widths).
4. **One stale docstring (fixed).**  `_validate_result_tile_shape` still
   said "prefix loops in logical mode order" — the exact wording §19.9
   retired in `schedule_passes`; the code matches physical storage order.

Probes A and B confirmed everything else holds: 31/31 admission probes
(forged, permuted, over/under-ranked, non-dense, subclassed,
duplicate-mode prefixes; non-identity execution at two layouts) and
26/28 completion probes passed as contracts, with the two findings above
the only escapes.  The mutation-matrix, alias-census, policy, and
atomic-conversion boundaries all held.

### 21.2 Representation decision

The §18.5 constraint is decisive: `lower_loopir_to_llir` consumes a bare
`LoopProgram` and the structural-activation locks exercise bare verified
programs, so a fact stored only on `ScheduledLoopIR` cannot reach target
lowering.  The selection is therefore **intrinsic optional program
state**: `LoopProgram.parallel: Optional[ParallelSelection]`.

`ParallelSelection` names the selected loop by the same
`(IndexId, part)` identity `_loop_key` already uses, restricted to
`ParallelPart.LOGICAL` and `ParallelPart.OUTER` — a split's point loop
carries the ragged-tail clamp and is deliberately unrepresentable,
matching the legacy rejection of `*_in` anchors.  It carries the three
target-independent fact families the design's Stage-5 abstract
parallelization requires:

- `ParallelWork(rows, nnz)` — the selected loop's declared trip-count
  dimension plus an optional `SparseWorkSource(tensor, level)` naming the
  compressed level whose stored entries measure one iteration's work;
- `ParallelDiscipline` — the race-freedom argument class the verifier
  re-proves (`RESULT_PARTITION` or `COMPACT_PARTITION`);
- `ParallelIntent` — schedule provenance (`EXPLICIT` only, until the
  automatic-plan migration widens the member set).

No OpenMP spelling, thread count, chunk policy, or rendered name appears
anywhere in the schema; canonical serialization moved to
`scorch.loopir.canonical.v8` with the selection as owned payload, and
`print_program` renders one `parallel` line.

### 21.3 Verifier surface

Four codes (83 total), each with direct adversarial regressions:
`invalid_parallel_selection` (exact stored fields, enum members compared
by identity so forged same-type instances fail, work-fact typing,
undeclared rows dimension, dense/bool/out-of-range work levels),
`parallel_target_missing`, `parallel_work_mismatch` (the rows fact must
restate the resolved loop's dimension; a named sparse work source must
actually be iterated inside it), and `parallel_race` (result writes must
carry the selected coordinate; ordered append assembly is rejected;
workspace state must be private to one selected iteration;
`RESULT_PARTITION` refuses heap-region programs; `COMPACT_PARTITION`
requires the unique region, membership inside it, and compact-cell
addressing by the selected coordinate).  A
`parallel_target_ambiguous` code was deliberately **not** added:
`duplicate_index_binding` and the split-ownership rules already make the
identity unique on any verified program, and a regression documents that
reasoning instead of shipping unreachable code.  A deleted
`parallel` field is `malformed_state` (the stored-field walk), so the
§11-style default-fallback downgrade is closed.  Erasure strips the
selection (schedule state, not base semantics), structural pass rebuilds
carry it, `verify_scheduled_loopir` rejects a base program that carries
one, the oracle executes selection-carrying programs unchanged, and the
builder continuation scans the new nodes through the declared-field
walk.

### 21.4 Consumption and target realization

`select_parallel_loop` runs last in `apply_schedule_plan` and consumes
every explicit `parallel_loop` fact exactly once — the typed twin of
`CINLowerer._apply_explicit_parallel_schedule`, which marks the named
loop on the assembled function after every structural transformation.
Legacy anchor semantics are reproduced on identities: a LOGICAL anchor
naming an affine-split variable selects the split's origin loop (the
`{var}_out` redirect), point loops and sparse results fail closed, the
anchor must resolve to exactly one scheduled loop, and only dense
logical loops and affine origin loops are admissible — compressed,
merged, and panel-origin anchors have no measured legacy comparand
(legacy's tag search finds for-loops only and raises an unowned
`ValueError` on a merged anchor), so they stay outside the migrated
family with a stable code.  The pass derives the work facts (the loop's
dimension; the first document-order compressed source under it, matching
the emission-order derivation the target's policy helper performs) and
the discipline (compact for heap plans), stamps the program, and
re-verifies it, so the verifier's race and work obligations gate every
admitted anchor.  Plan-gate changes: plain explicit-parallel plans left
the `unsupported_schedule_parallel` boundary (parallel **tiles** remain
unmigrated there); the heap gate widened from the outermost-prefix pin
to membership in the dense prefix (the exact legacy envelope
`result_names[:-1]`); the composed sparse panel dropped its
single-prefix restriction with the exact `(*prefix, panel, pack)` order.

Target lowering resolves a present selection to its chain position,
requires the work estimate to restate the resolved loop's dimension, and
routes marking by family. Direct and stack chains suppress the
emission-time auto gate and mark the selected loop on the assembled
function through the new `complete_parallel` step — first among the
completions, exactly the legacy pipeline position — under
detached-header re-identification (single-loop matching, because
workspace regions emit loops outside the direct chain), a structural
work-source cross-check against `find_sparse_pos_array`, and a detached
snapshot policy comparison, all owned as `parallel_completion_lost`.
Panel and relayout routes keep their completion-owned marking and only
prove the selection names the row they mark.  The heap route adopts the
selected prefix position as its parallel row; the composed chain is
generalized to rank ≥ 2 (prefix loops between the panel origin and the
window; `apply_panel_tile` and `_validate_panel_shape` were already
rank-general) with its default anchor at the window's dense parent — the
only anchor legacy admits.  A missing selection preserves every existing
derivation byte-for-byte, so bare verified programs keep direct
structural activation and all retained captures are unchanged.

### 21.5 Both §18.4 boundaries lifted, with measured parity

The lifted envelope was measured against the live legacy route before
implementation (anchor survey over 22 schedule/anchor cells) and locked
as an eleven-cell byte-parity grid plus compiled shadows:

- **inner-prefix heap anchor** (`ttm_heap_anchor_b`): legacy emits
  `scorch_nthreads(Core2_pos[Core1_size], Core1_size)` with the dynamic
  chunk at 5,200 bytes versus anchor `a`'s `scorch_nthreads(-1,
  Core0_size)` at 5,160 bytes; the LoopIR route reproduces both
  byte-for-byte, and the strip init/copy policies stay
  anchor-independent;
- **rank-3 heap+panel composition** (`ttm_heap_panel_anchor_b`): legacy
  admits only the window's dense parent (`b`, the innermost prefix) and
  emits 5,764 bytes; the LoopIR route is byte-identical, and the
  outermost anchor still fails closed (`panel_parallel_scope`), exactly
  as legacy rejects it;
- direct, stack, and redirect anchors: dense-matmul anchors `i`/`j`,
  SpMM anchors `i`/`k`, the stack row anchor (3,038 bytes versus the
  auto route's origin marking at 3,066 bytes — the suppression
  differential), and the affine `j`→`j_out` redirect at both public
  spellings — all byte-identical, plus an f64 inner-anchor variant;
- four subprocess-isolated compiled shadows (inner-prefix heap, composed
  heap+panel, SpMM free-axis anchor, stack row anchor) assert bitwise
  equality against the legacy kernels and Torch agreement while keeping
  the pytest parent at its inherited native-extension footprint
  (§19.3 discipline).

Reduction anchors, pack-loop anchors, missing heap anchors, and
non-prefix anchors keep failing closed with the heap family's message;
`tile.parallel` remains `unsupported_schedule_parallel`.

### 21.6 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-parallel-6242da8/` (gate
battery, captures, statics) with the inherited-gate reproduction under
`phase6-repro-5ef0401/` and the working notes under
`phase6-parallel-selection/`.

Final evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-parallel-533de08/` (gate
battery, latency, statics, `EVIDENCE_SHA256SUMS`), with the
corrected-tip captures under `phase6-parallel-6242da8/captures/` (the
tip's one additional commit is test-only, so the source captures carry
over unchanged), the inherited-gate reproduction under
`phase6-repro-5ef0401/`, and the anchor survey, parity harness, and
working notes under `phase6-parallel-selection/`.

- exact five-file Phase-6 contract focus at clean detached `533de08`:
  **600 passed** (the inherited 548 plus 13 review locks, 31 schema
  locks, and 8 consumption/lift locks);
- exact scheduled runtime focus: **212 passed** in **771.74 s**
  (log SHA-256
  `111056f21dc5720267b46ffd1c32b65b17f62154e945692dadcdf345435cd4fa`,
  JUnit SHA-256
  `638a0ab6a4fd3b6643c76949bd667d10fa918f7b7ff4a2f97e87b83334fe7df8`),
  including the eleven-cell anchor parity grid and the four
  subprocess-isolated anchor shadows;
- full legacy schedule-generality file: **45 passed** in 179.76 s;
- fresh 20-source corpus, 42-cell grid, and 11 heap goldens:
  **73/73 byte-identical** to the retained
  `phase6-multiprefix-review-a24c0c1` finals — no legacy emission and
  no unselected LoopIR emission moved;
- static comparison: every changed file is Black/Flake8-clean and the
  changed production files are focused-mypy-clean; full-source Black
  reproduces only the inherited `prebuilt_kernels.py` finding,
  full-source Flake8 is normalized byte-identical at nine inherited
  findings, and normalized full-source mypy is byte-identical at
  **146 findings in 12 files** over 60 source files;
- paired same-session compiler latency (5 warmups / 30 samples, clean
  detached `5ef0401` base versus `533de08` candidate, per-case
  generated-source SHA-256 identical): `small_dense` **0.976/0.988**,
  `reduction` **1.012/1.055**, `csr_intersection` **0.976/0.971**,
  `sparse_union` **0.982/0.973** (p50/p95 candidate/base; worst
  **1.055**, no 1.10 crossing, no control run required);
- literal unpartitioned clean detached-worktree non-performance suite
  at `533de08` with isolated caches and asserted import provenance:
  **4,111 passed, 14 skipped, 3 performance tests deselected, one known
  sparse-invariant warning, and zero failures** in **2,684.23 s**
  (log SHA-256
  `d99ead73cb0d60b4fdba53918202888a98c73307f03819cd148b95384b35cdfc`,
  JUnit SHA-256
  `7fa857ba8d9378de9977c4cdc6266ef481f54151cb9eaf19810e582d962a982b`)
  — the inherited 4,041 selection plus exactly the 70 tests this
  session added;
- `git diff --check` clean; live origin still `58e8565`; all five
  protected tracked files at their recorded hashes; no unrelated
  tracked or untracked material staged.

### 21.7 Scope verdict and remaining Phase-6 boundaries

Milestone 1 of the §20.5 sequence is complete: every explicit
`parallel_loop` fact is consumed exactly once into a verified,
target-independent selection, both §18.4 boundaries are lifted with
byte parity, and unsupported anchors fail closed with stable owned
diagnostics.  **Phase 6 remains open.**  Intended automatic-plan routing
through verified `LoopPlan` (milestone 2), canonical `LoopPlan` schedule
identity for the strangler path (milestone 3), and the
criterion-by-criterion exit audit (milestone 4) remain; production
dispatch, selector integration, cutover, and legacy deletion stay out of
scope.  The admission envelope deliberately excludes compressed, merged,
and panel-origin anchors (no measured legacy comparand) and parallel
tiles; widening either requires fresh legacy measurement first.

## 22. Independent review corrections to abstract parallel selection

This section is chronologically later than §21 and supersedes §21.3,
§21.4, and §21.7 wherever their completeness claims conflict.  The
eleven-commit inherited range was reviewed claim by claim and with fresh
adversarial programs before the next milestone began.  That review found
that Milestone 1 was not complete at `e68c060`: its broad anchor parity
grid was green, but several stored-work, transformation, and post-assembly
ownership boundaries were not yet exact.

### 22.1 Findings

1. **`ParallelWork` was not a verified program fact.**  The selection pass
   stamped the first sparse cursor below a dense anchor without reproducing
   the legacy policy's actual discovery boundary.  It therefore assigned
   sparse work to merged nests (whose established emission is row-count
   only), to a cursor whose dense parent was a different coordinate, and
   to a cursor whose parent tensor/level was not the exact first dense
   input driver that spells the selected loop bound.  The verifier accepted
   all three.  A supported `X[i,k] * A[i,j]` case with `X` declared first
   demonstrated the consequence: the legacy route emitted
   `scorch_nthreads(-1, X0_size)`, while the typed route emitted
   `A1_pos[X0_size]`.  Directly constructed/forged schema states selecting
   sparse or merged coordinate loops also passed verifier work checks while
   storing a logical dimension as `rows`, even though their physical
   position trip count is not represented by that dimension.  The
   production selection pass had already rejected those anchor kinds since
   `6242da8`; this was a verifier truthfulness/totality defect, not a newly
   admitted public schedule.
2. **Source order and transformation ownership were incomplete.**
   `_walk_declared_schema` traversed sibling statements in reverse despite
   the work contract depending on the first emitted sparse initializer.
   `apply_relayout` silently discarded an existing selection, while
   `apply_result_tile` could reinterpret and discard one.  A plan with no
   `parallel_loop` fact could also accept a program already carrying a
   selection.
3. **Target completion trusted mutable state too far.**  Completion did not
   reverify the program and the primitive selection payload at the point of
   use.  `complete_parallel` matched one detached header anywhere in the
   function rather than the complete owner chain, and it did not reject a
   different loop that arrived pre-marked or atomic.  Panel and heap
   completion re-derived policy instead of realizing the exact owned work
   fact.  The panel-to-heap and panel-to-relayout handoffs then trusted a
   syntactically valid but mutated row policy, and composed heap+panel
   completion did not revalidate a work fact changed after panel completion.
4. **Two heap effects were census-pinned but not position/identity-pinned.**
   Moving the exact result-shape validation after compute passed the
   name/shape census, and other ABI validation calls were not required to
   remain in the prologue.  A second `torch::empty` fed by protected result
   shape state was admitted because the allowlist recognized the callee name
   rather than the one canonical allocation object.

### 22.2 Corrections

`d7f0970` closes those boundaries without a schema-version change.
The verifier and selection pass now share one structural derivation:

- statement traversal is declared/source ordered;
- any merged subtree uses the row-count-only policy;
- a sparse work source is permitted only for the first sparse cursor,
  immediately parented by the selected dense coordinate, when that
  tensor/level is also the exact first dense input driver of the target
  loop bound;
- dense logical and affine-origin loops are the only representable target
  kinds, and the verifier requires the stored source (including `None`) to
  equal the re-derived source exactly.

`None` is consequently rendered as `row_only`, not `uniform`: merged rows
can be highly nonuniform even though compatibility requires the row-only
legacy estimate.

Structural passes now preserve selection deliberately, or reject a
preselected input before a transform that changes its race discipline.
Fact-free plans reject carried selections.  At target completion, a
primitive snapshot of the exact selection/work identities, enums, level,
and intent is compared after a fresh `verify_program`.  The resolved sparse
source is combined with the selected loop's actual target bound.  Direct
and stack completion require the unique complete ancestor prefix, reject
every pre-existing ordinary or atomic mark, independently compare the
legacy marker result, apply the exact owned policy, and require the selected
loop to be the sole final mark.

Panel and heap completions use the same work-fact realization.  Panel keeps
an independently detached snapshot of the exact row header it produced;
both heap and relayout recheck that snapshot at their own entry, and heap
also rechecks the selection after the panel boundary.  Heap completion now
requires the complete freshly reconstructed ABI validation block at the
canonical prologue offset before any allocation or compute, and requires
the only protected `torch::empty` expression to be the canonical generated
result allocation by object identity.

`b03c705` locks the corrections with source-order, cross-driver,
merged-row-only, transform-ordering, malformed post-init state,
ancestor-relocation, ordinary/atomic stray-mark, panel/heap/relayout
handoff, full validation-prologue, and protected-allocation adversaries.
The package and node documentation were corrected to the actual rank>=2
heap and row-only work contracts; canonical serialization remains
`scorch.loopir.canonical.v8` because no serialized field or representation
changed (the human-readable label and contract wording were corrected).

### 22.3 Verification and verdict

The focused contract before detached final gates is **534 passed** across
the verifier, printer, schedule-pass, and LLIR-lowering files.  A fresh
86-case format/operand/order/anchor audit admitted 46 programs and produced
byte-identical legacy/LoopIR sources for all 46; the other 40 retained
stable fail-closed boundaries.  Two independent source-only audit runs are
byte-identical and retained under
`phase6-parallel-review-b03c705/verification/schedule-audit/` (results
SHA-256 `d6c075663f9d8df4a1a0ade293a174be99ab346c96f7a91c3674f45d0b60433a`;
manifest SHA-256
`b49b7f0f53484396a195d375a16e5bc47a80ef359209fa1184f2aa53fa4a5769`).
Focused Black, Flake8, and production mypy are clean, and
`git diff --check` is clean.  The final scheduled-runtime, capture/static,
latency, and clean detached full-suite receipts are recorded in the
chronologically latest handoff section.

With these corrections, **abstract parallel selection is complete for the
measured explicit family**, but Phase 6 remains open.  Automatic-plan
routing, canonical `LoopPlan` schedule identity, and the
criterion-by-criterion exit audit are still required.  No automatic route,
release dispatch, cache, production cutover, or legacy deletion changed in
this review.

## 23. Automatic routing, canonical identity, and the Phase-6 exit audit (2026-07-26)

This session first independently re-reviewed the §22 correction commits
(`d7f0970`, `b03c705`, `ef49d50`) without trusting their report, found no
concrete defect, and then implemented the two remaining Phase-6
milestones and the criterion-by-criterion exit audit.  Six focused
commits, nothing pushed, amended, squashed, or reordered:

- `c745e6b` — `feat(compiler): record automatic scheduling decisions in LoopPlan`
- `2fd563a` — `feat(compiler): admit tile-free automatic plans at the LoopIR gate`
- `76205af` — `test(compiler): lock automatic routing parity and boundaries`
- `ea2ec02` — `feat(compiler): own canonical plan and request identity at the strangler boundary`
- `42b817a` — `test(compiler): lock canonical plan and request identity`
- `f218c2e` — `fix(compiler): keep the request identity inside the strangler package`
  (the release-neutrality battery scans production modules outside
  ``compiler/loopir`` for the strangler package's name; the identity
  module moved into the package unchanged, which is also the more
  faithful placement for a strangler-only boundary)

### 23.1 Independent review of the §22 corrections

The inherited three-commit correction range was inspected diff-by-diff
and its focused gates reproduced exactly: the five-file contract battery
(**534 passed**), the compiled scheduled-slice file (**185 passed**),
and the 86-case schedule audit re-run from the retained harness in a
clean detached worktree at `b03c705` — results JSON byte-identical to
the retained run (SHA-256
`d6c075663f9d8df4a1a0ade293a174be99ab346c96f7a91c3674f45d0b60433a`).
Four fresh adversarial probes beyond the committed locks all held:
in-place mutation of the selection's ``rows`` DimensionId integer after
lowering init and wholesale replacement of ``program.parallel`` with a
structurally equal deep copy both fail closed as
``parallel_completion_lost`` (the primitive-signature and object-identity
pins), and a three-input cross-driver case outside the audit matrix
(``Y[i,m]``/``X[i,k]`` dense with ``A[i,j]`` sparse, both declaration
orders) reproduced legacy byte parity with the correct
``scorch_nthreads(-1, X0_size)`` versus
``scorch_nthreads(A1_pos[A0_size], A0_size)`` policies.  The sole-mark
census was verified identity-pinned (LLIR ``ForLoop`` has no structural
equality), the panel prologue offset was verified against the exact
three-statement panel prepend with result-tile completion ordered before
relayout completion, and canonical serialization was confirmed at
``scorch.loopir.canonical.v8`` with the ``row_only`` label confined to
the human printer.  No fix commit was needed.

### 23.2 Recorded automatic plans (Milestone: automatic routing, part 1)

The Milestone-2 inventory was re-verified against the live tree (F1
scalar, F2 identity, F3 explicit, F4 plan-free production
auto-scheduling; only ``"explicit"`` and ``"auto"`` provenances exist).
`c745e6b` makes the automatic scheduler's standalone workspace insertion
an explicit plan fact — the exact class of hidden replay state the
milestone existed to eliminate:

- ``WorkspaceInsertion(reduction_loop, axis_loops, dense)`` is stored on
  ``LoopPlan.workspace``; the legality boundary re-derives the decision
  from the analyzed CIN, the plan order, and the recorded tiles and
  requires the stored fact (including ``None``) to equal that derivation
  exactly (``auto_workspace_decision``) — the §22 stored-equals-derived
  pattern applied at the plan boundary.  Explicit plans reject the fact
  (``workspace_provenance``).
- the F2 identity path records the fact while scheduling, and
  ``_replay_auto_plan_owned`` now consumes it — cross-checked against
  the replayed nest — instead of re-running
  ``should_insert_workspace`` against scheduler policy at replay time;
- ``Scheduler.auto_schedule_plan`` originates one verified
  ``provenance="auto"`` plan at the plan-free F4 boundary, covering both
  regblock arms (the ``CHILD_OF`` regblock tile family included); its
  replay reproduces ``auto_schedule``/``_auto_schedule_regblock_arm``
  output exactly on every measured non-root case.  Release dispatch does
  not consume it yet.

Two legacy observations were recorded by this work.  First, for a dense
root-scope reduction the plan-free surgery inserts a pure-overhead root
workspace whose candidate tiles never materialize (the ``Where`` root
makes the tiling heuristics bail), while the established
``ScheduledCIN`` replay contract omits it; the recorded fact follows the
replay contract (``None``), keeping both existing behaviors
byte-compatible while making the divergence visible and locked.  Second,
the tiled automatic family strip-mines reduction variables with a
reduce-out consumer — a shape the legacy *explicit* route itself rejects
("Affine reduction tiling requires an accumulator spanning outer
tiles"), so it exists only through the auto heuristics and has no
explicit comparand.

### 23.3 Tile-free automatic routing through LoopIR (Milestone: automatic routing, part 2)

`2fd563a` widens the strangler provenance gate to exactly the recorded
tile-free automatic family: a cost-model loop order with no tiles, no
workspace fact, and no explicit-only facts, flowing through the same
verified reorder pass, stage records, and erasure/oracle machinery as
explicit reorder-only plans.  The tiled automatic family fails closed
with the stable ``unsupported_schedule_auto_family`` code, and every
foreign provenance keeps ``unsupported_schedule_provenance``.

`76205af` locks the family: a seven-case auto parity grid (SpMM,
two-reduction, broadcast row, sparse union add to dense, dense add
2d/3d, vector add) is byte-identical to legacy; the strangler artifact
carries the verified auto plan with the full seven-stage record sequence
and erasure equivalence; and two compiled shadows (automatic SpMM,
automatic dense add) execute bitwise-equal to legacy and numerically
equal to Torch.  F1 remains fail-closed at the base-family boundary with
the stable ``unsupported_statement`` code — the legacy generated-kernel
route itself dies on loop-free CIN with an unowned ``IndexError``, so F1
has no measured comparand at this boundary.

### 23.4 Canonical plan and request identity (Milestone complete)

`ea2ec02` defines ``scorch.loopplan.canonical.v1`` — the versioned
canonical serialization of one verified plan from semantic content only
(order, tiles and placements, accumulation, unroll, panel bounds,
relayout, result tile, parallel selection, workspace insertion,
provenance).  Plan-referenced identities are rewritten through an
artifact-local canonical numbering derived from the normalized CIN
(indices by outer-to-inner nest binding order, symbols by first
appearance in a deterministic assignment walk), so equivalent plans from
fresh builders serialize byte-identically while process-global
allocation order, Python ``hash()``, display names, rendered C++,
mutable scheduler state, and insertion history never enter the bytes.
``plan.tag`` is a presentation-only annotation and is deliberately
outside the identity.

``plan_schedule_digest`` is the separate provenance-free layer for
comparing schedule content across provenances; the request identity
(``scorch.loopir.request.v1``) includes provenance because provenance
selects the gate and replay contract — the explicit cross-provenance
rule.  The strangler compile/shadow request boundary now computes the
canonical request dump and its SHA-256 content key (canonical normalized
CIN + canonical plan or the explicit unscheduled marker + result shape +
runtime bindings) and retains both on the compiled artifact.  No new
compiler stage is added, no artifact cache consumes the identity, and
the release source-derived cache is untouched.  Collisions are handled
by retaining the authoritative dumps beside the content-addressed
digest.

`42b817a` locks: fresh-builder equality (the probe asserts the two
builds allocated different global identities), repeated-dump and
repeated-compilation determinism, inequality for every semantic decision
(order, width, accumulation, unroll, placement, tile presence, panel
bound, explicit parallel selection, workspace fact), the
cross-provenance rule with the digest-layer agreement, the tag
exclusion, malformed/hostile state, shape/dtype/plan-presence coverage,
the digest-of-retained-dump contract, and the unchanged stage-record
sequence.

### 23.5 Phase-6 exit audit

Route trace.  Public explicit scheduling:
``Schedule`` → ``Scheduler.apply_schedule`` (shared validation,
surgery discarded) → verified ``LoopPlan`` → normalized CIN →
verified LoopIR → pure typed passes (``apply_schedule_plan``) →
oracle/erasure proofs → structured LLIR → C++ → compiled execution,
with stage records at every boundary.  Migrated automatic routing:
``Schedule()``/``auto_schedule_plan`` → verified ``provenance="auto"``
LoopPlan (order, workspace fact, tiles recorded) → the same typed path
for the tile-free family; the tiled family fails closed at the gate.

Criterion-by-criterion:

1. **Schedule decisions applied only to LoopIR for migrated
   operations — met.**  Every migrated route consumes one verified plan
   through pure typed passes on verified LoopIR; the legacy tree surgery
   is discarded at ``apply_schedule`` and replayed only inside the
   legacy adapter for the legacy comparand.
2. **No name/regex/rendered-text discovery in panel/relayout/result-tile
   passes — met.**  ``schedule_passes.py`` contains no regex, no name
   matching, and no text traversal (verified by direct audit this
   session); the only regexes in the target lowering are the §20-audited
   *rejection* validators (`_CPP_IDENTIFIER`,
   ``_RESULT_TILE_POLICY_TOKEN``, ``_RESULT_TILE_NUMERIC_LITERAL``),
   which forbid C++ spellings rather than discover targets.
3. **Canonical plan identity owns the strangler request boundary —
   met.**  The canonical request identity (§23.4) is computed and
   retained at the compile/shadow boundary for every strangler request;
   there is deliberately no LoopIR or plan artifact cache, and the
   release source-derived kernel cache stays untouched until the
   cutover phase, exactly as scoped.
4. **Explicit and automatic structural/numerical differentials —
   met for the explicit families and the tile-free automatic family;
   open for the tiled automatic family.**  Explicit: the retained §21/§22
   grids, shadows, and captures plus this session's re-runs.  Automatic:
   the seven-case parity grid and two compiled shadows above.  The tiled
   automatic family (heuristic strip-mines composed with the recorded
   workspace insertion, including reduction-loop strip-mining with a
   reduce-out consumer) has no typed emission twin and fails closed; its
   decisions are now fully recorded and replayable, but its differential
   obligation is unmet by construction.
5. **Representative tile-j, direct tile-ijk, rank-2 heap tile-ijk, and
   rank-3 multi-prefix heap readiness — met** (§21.5/§22 receipts,
   reconfirmed by this session's byte-identical capture and audit
   re-runs).

**Verdict: Phase 6 remains open on exactly one boundary** — the typed
emission twin for the tiled automatic family (criterion 4's automatic
residue).  Everything else holds with evidence.  Because there is no
genuine Phase-6 GO, the Phase-7 stretch was not started; production
dispatch, selector integration, cutover, and legacy deletion stay out of
scope.

### 23.6 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-exit-f218c2e/` (captures,
verification, latency, full-suite ledgers).

- exact nine-file contract focus at clean detached `f218c2e`:
  **691 passed** (verifier, printer, schedule passes, LLIR lowering,
  loop plan, plan identity, schedule API, and both neutrality
  batteries);
- compiled scheduled/pipeline/generality focus at the same detached
  worktree: **269 passed** in
  **970.97 s** across the scheduled slice, pipeline execution, and
  legacy schedule-generality files (log SHA-256
  `250a25c2f3defa81451da32441d61a92b58f89017c58a1ad2560ea750750a4c0`,
  JUnit SHA-256
  `8b6ba2b57604f3c774191bd5f91f9c17faee662e2a3604afe1c733e6718b4d06`);
- fresh 86-case schedule audit at `f218c2e`: **46 admitted and 46
  byte-identical**, 40 stable fail-closed outcomes; two runs
  byte-identical, and every per-case record equals the retained
  `b03c705` results (only the embedded commit id differs);
- fresh captures byte-identical to the retained
  `phase6-parallel-review-b03c705` finals: 20-source corpus, 42-cell
  grid, 11 heap goldens, and the 22-cell anchor survey — **95/95**;
  plus **10/10** explicit-anchor and **11/11** heap LoopIR/legacy
  source comparisons, all byte-identical;
- static comparison at the detached tip: full-source Black reproduces
  only the inherited `prebuilt_kernels.py` finding, full-source Flake8
  is byte-identical at the nine inherited findings, and full-source
  mypy is **146 findings in 12 files** over 61 source files with every
  normalized error line byte-identical to the retained baseline; the
  changed files are Black/Flake8-clean and ``loopir/plan_identity.py``
  is focused-mypy-clean;
- paired same-session compiler latency (5 warmups / 30 samples, clean
  detached `ef49d50` base versus `f218c2e` candidate):
  with identical per-case generated-source hashes: `small_dense`
  **0.982/0.987**, `reduction` **1.001/1.040**, `csr_intersection`
  **0.997/0.982**, and `sparse_union` **0.964/0.941** (p50/p95
  candidate/base; worst **1.040**, no 1.10 crossing and no attribution
  rerun required); base JSON SHA-256
  `d9b04b28b2080bd09c38c6550db15099a5757e7c63538254a7ffe139200c3566`,
  candidate JSON SHA-256
  `8fedb6a435677d8d20fb941904ee168fdb5de60f2cb33b8be15aed7aeff9d308`,
  comparison-log SHA-256
  `cd450535f38716102e3f4183053b7df7e35cefed820c1c658872f889dd164327`;
- literal unpartitioned clean detached-worktree non-performance suite at
  `f218c2e` with isolated caches and asserted import provenance:
  **4,166 passed, 14 skipped, 3 performance tests
  deselected, one known sparse-invariant warning, and zero failures**
  in **2,701.32 s** (wall **45:01**); log SHA-256
  `c393ea197aff538996462d6a32946da55563c6465ca7129825c94b6ff79927f4`,
  JUnit SHA-256
  `26d55eaa6ea31405820e7f94f92032a98005be71fc4239d2f0134719877e4e78`;
  the literal run crossed the historical late-abort region without a
  libomp/resource event, so no partition substitute or base control was
  needed;
- `git diff --check` clean; live origin still `58e8565`; all five
  protected tracked files at their recorded hashes; no unrelated
  tracked or untracked material staged.

## 24. Review corrections to automatic routing and request identity (2026-07-26)

This section supersedes §23's current-state claims.  Section 23 remains
historical evidence for the inherited implementation at `f218c2e`; its
statement that Phase 6 had exactly one remaining boundary is not the
verdict after adversarial review.

The review inspected the inherited automatic-plan and canonical-identity
range through `0c4c6cd`, reproduced its focused and compiled gates, and
then tested the claimed boundaries independently.  Fourteen focused
correction commits were added without amending, reordering, or pushing:

- `6a2a5e0` / `fd91be0` — make the request identity complete and lock
  its first trust boundaries;
- `6702f43` / `3d8f30a` — expose the automatic root-workspace boundary
  and remove recording-only work from release scheduling;
- `dd4d73a` / `21cac4d` — close nested request-state and
  index-expression validation gaps;
- `369cf03` / `3c964bf` and `4a82112` / `befa198` — require distinct,
  fresh, exactly typed auto-plan ownership sinks;
- `a2cc209` / `7afbc1e` — make request validation side-effect-free and
  cover unary RHS binding counts and hostile containers; and
- `25715bf` / `c7e3702` — reject hostile option values before invoking
  equality or hashing protocols.

### 24.1 Canonical request boundary: corrected

The inherited `scorch.loopir.request.v1` key omitted `CompileOptions`,
serialized dtypes through arbitrary `str()`, admitted unbounded Python
integers until JSON conversion, and computed between owned compiler
stages.  Consequently different generated C++ could share an identity,
hostile dtype objects could execute code or collide, and malformed
requests could fail while the compilation context recorded no failed
stage.

The corrected schema is **`scorch.loopir.request.v2`**.  It includes the
exact canonical `CompileOptions` state with the public `Schedule`
spelling replaced by the separately verified plan, closed float32/64
dtype tokens, nonnegative int64 extents, exact runtime containers, and
the existing canonical CIN/plan/binding content.  Construction now
belongs to
`FRONTEND_VALIDATED_OPERATION_CONSTRUCTION`; the compiled artifact
requires a nonempty retained dump and its full lowercase SHA-256.
`scorch.loopplan.canonical.v1` and
`scorch.loopir.canonical.v8` are unchanged.

The second adversarial pass found subtler fail-open behavior and closed
it:

- the cycle/depth preflight now traverses `ForAll.index_var`,
  `TensorAccess.indices`, and `IndexVarAdd` edges;
- every nested options carrier has exact stored fields and pure
  stored-state validation, while constructor-normalized fields must
  already be exact tuples;
- forged one-shot iterators are rejected without consumption and an
  immutable Darwin snapshot is not revalidated through new host
  filesystem probes;
- ABI RHS occurrences are counted by a cycle-safe structural walk
  rather than the legacy visitor that recurses forever on `UnaryOp`;
- list subclasses cannot invoke hostile `__iter__`/`__len__`; non-finite
  JSON is rejected; and
- LLIR-pass entries and compiler-wrapper names are type-checked before
  tuple equality or set membership can invoke caller protocols.

The canonical request remains a strangler-only retained identity.  No
release cache consumes it and the source-derived JIT cache is unchanged.

### 24.2 Automatic workspace ownership: corrected and honestly bounded

The inherited report described `auto_schedule_plan` as a complete F4
recording, but the dense root reduction `C[k] += A[j,k]` disproved that
claim: plan-free production auto-scheduling materializes a root
workspace and emits its zero/copy work, whereas the historical
`Schedule()` replay contract deliberately elides that pure-overhead
workspace.  Returning `workspace=None` from F4 silently identified two
different programs as the same recorded decision.

The correction keeps both established emission paths byte-stable but no
longer claims equivalence:

- plan-free `auto_schedule_plan` fails closed when it reaches that
  materialize-versus-elide boundary;
- ordinary release auto-scheduling calls no recording-only helper;
- the admitted LoopIR family is described as the measured **tile-free
  auto-replay contract**, not as authenticated cost-model origin; and
- complete-plan helper mode validates its exact bool before loop-free
  return and requires distinct, empty, exact tile/workspace sinks.

The root behavior is not "fixed" by hiding it.  It is now an explicit
remaining representation decision.

### 24.3 Corrected Phase-6 exit disposition

Criterion-by-criterion after the review:

1. **Schedule decisions applied only to LoopIR:** met for the migrated
   explicit families and the measured tile-free auto-replay family; not
   claimed for the remaining plan-free automatic families.
2. **No name/regex discovery in typed panel/relayout/result-tile
   passes:** unchanged and met.
3. **Canonical identity owns the strangler request:** met by request v2
   with the fail-closed boundary above.
4. **Explicit and automatic differentials:** explicit and tile-free
   replay evidence holds; open for both (a) dense root-workspace
   materialization versus replay elision and (b) heuristic
   tiled/workspace auto, including `OUTERMOST`, `CHILD_OF`, reduction
   strip-mines, and reduce-out consumers.
5. **Representative tile-j/tile-ijk readiness:** unchanged and met.

The default `regblock_dual` production path also stitches two automatic
arms behind a runtime target branch.  Before exit, Phase 6 must either
include that composition in its production-auto differential evidence
or explicitly assign the target stitch to Phase 7 after proving both
constituent schedules independently.

**Verdict: Phase 6 remains open on two automatic-origin families, with
the `regblock_dual` ownership decision still required.**  There is no
Phase-6 GO and no Phase-7 work, production cutover, selector migration,
cache cutover, or legacy deletion in this review.

### 24.4 Verification

Final evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-review-c7e3702/`.

- the final ten-file contract membership passed **759 tests** at
  `c7e3702`; the dedicated options/identity membership passed **81**,
  including all adversarial probes;
- the inherited compiled schedule/pipeline/generality gate plus the
  review corrections passed **269 tests** before the final
  identity-only hardening; the exact final-tip full-suite receipt below
  contains those tests again;
- paired same-session compiler latency at clean detached `0c4c6cd`
  versus `c7e3702` (5 warmups / 30 samples, native work intercepted)
  remained inside the 1.10 target with identical sources and complete
  build summaries: `small_dense` **0.999/0.993**, reduction
  **1.003/1.069**, `csr_intersection` **1.004/1.023**, and
  `sparse_union` **1.009/0.992** (p50/p95); the canonical-endpoint
  worst case was **1.070**;
- changed files are Black/Flake8-clean except the inherited scheduler
  C901; full-source Black, Flake8, and mypy are exactly at the retained
  baselines (one formatting finding, nine lint findings, and 146 mypy
  findings in 12 files, respectively);
- the clean detached source-only gate reproduced **95/95**
  corpus/grid/heap/anchor captures byte-identically, the 86-case audit
  as **46 admitted / 40 rejected / zero divergent** with every
  environment-independent record equal to the retained result, plus
  **10/10** explicit-anchor and **11/11** heap LoopIR/legacy
  byte-identical comparisons (ledger manifest
  `a869e6e3860ce20ad916ddebce8953538d6f0b490751061b8e7c9e55f342010c`);
- the literal unpartitioned clean detached-worktree non-performance
  suite at exact `c7e3702`, with isolated caches and asserted import
  provenance, passed **4,191 tests, 14 skipped, 3 performance tests
  deselected, one known sparse-invariant warning, and zero
  failures/errors** in **2,852.35 s (47:32)**; it crossed the historical
  late-process region with no libomp/resource event (log SHA-256
  `be5a80306ffb4269b6e93b2328c70b7b0701e27b188f1bee8993c8eece99f8d3`,
  JUnit SHA-256
  `b0687cfaf32d64e713f8dca07f824bc3697d678460248ee128dad1af1c111293`,
  evidence-manifest SHA-256
  `f56d0e1cfaf9509ce9ae27f71ad7961a9c7c0b43244b5c68beb37123cac32266`);
  and
- `git diff --check` is clean, local and live origin remain at
  `58e8565`, and all five protected files retain their recorded hashes.

## 25. Automatic origin, root-workspace resolution, and the dual boundary (2026-07-26)

This section supersedes §24's current-state claims.  Section 24 remains
historical evidence for the tree at `f3dee24`.

The session first independently re-reviewed the fourteen §24 correction
commits without trusting their report and found no concrete defect:
every diff was inspected, the ten-file contract membership (**759
passed**), the options/identity membership (**81 passed**), and the
compiled schedule/pipeline/generality battery (**269 passed in
993.27 s**) were reproduced exactly at `f3dee24`, and seven fresh
adversarial probe families beyond the committed locks all held
(identity side-effect freedom and idempotence, requested-schedule
clearing, exact-int extents, non-consuming generator rejection,
hostile-deep index chains under the depth bound, shared-node DAG
rejection, and hostile forged plans).  Five focused commits were then
added, nothing pushed, amended, squashed, or reordered:

- `e1afa72` — `feat(compiler): type automatic origin and admit the regblock stack form`
- `830fcd2` — `test(compiler): lock automatic origin policy and the stack-form family`
- `1d7a6db` — `fix(compiler): elide the dense root-scope automatic workspace`
- `36cbac4` — `test(compiler): lock the aligned dense root-workspace elision`
- `dba2a22` — `test(compiler): lock the regblock_dual ownership boundary`

### 25.1 Typed automatic origin (Milestone: origin honesty)

`e1afa72` introduces the versioned `AutoOriginPolicy` fact
(`scorch.autopolicy.v1`) on `LoopPlan`: which regblock arm originated
the plan and the tile width that arm applies.  The legality boundary
now re-derives the complete heuristic tile list and the workspace
insertion from the analyzed CIN, the plan order, and that policy —
mirroring the legacy surgery's post-insertion candidate selection,
first-loop exclusion, sparse exclusion, retraversal guard, root-scope
bailout, and per-arm placement and width — and requires the stored
facts to equal the derivation exactly (`auto_tile_decision`,
`auto_workspace_decision`).  A plan carrying the string ``"auto"`` can
no longer justify tiles or a workspace the recorded policy would not
derive, and arm mislabeling (a tile-free plan labeled as the regblock
arm, or vice versa, or a forged policy width) fails closed.  The
cost-model loop order remains an attested decision — admission is
still a replay contract, not cost-model attestation.  The policy is
required on every automatic plan, rejected on explicit plans
(`auto_policy_provenance`), exact in type, schema token, stored
fields, arm flag, and positive C++-representable width, and is
verification state only: it stays outside
`scorch.loopplan.canonical.v1` and the request identity, so the two
arms of a decision-free program serialize byte-identically while the
two arms of an SpMM still separate through their recorded decisions.
`scorch.loopir.request.v2` and `scorch.loopir.canonical.v8` are
unchanged.

### 25.2 The regblock stack-form family (Milestone: heuristic tiled/workspace)

The measured equivalence chain for the production-relevant automatic
tiled subfamily — the regblock arm of the dual route — closed it onto
already-verified machinery.  For `ds` and `ss` SpMM the legacy
automatic regblock surgery (standalone dense workspace plus one direct
serial `CHILD_OF` tile of the workspace axis) is byte-identical to the
legacy *explicit* stack-tile schedule, and the migrated stack-tile
pass already reproduces that explicit form byte-for-byte through
LoopIR.  The strangler gate therefore admits exactly the recorded
stack form — dense workspace over a single trailing free axis, one
direct serial unrolled `CHILD_OF` tile of that axis under the row
loop, width equal to the policy width, regblock arm recorded — and the
typed driver lowers the recorded workspace+tile pair through the
verified stack-tile pass without mutating the plan facts.  Locked
evidence: byte-identical LoopIR-versus-legacy source for the automatic
regblock arm, the artifact carrying the verified plan with the
unchanged seven-stage record sequence, erasure returning the base
program, a compiled shadow executing bitwise-equal to legacy and
numerically equal to Torch, fail-closed coverage for every shape
deviation (missing or wrong-arm policy, forged width, no-unroll,
outermost or wrong-parent placement, sparse or missing or
wrong-axis workspace), and the `ss` operand still failing closed at
target lowering (hierarchical compressed descent).

### 25.3 Dense root-workspace resolution (Milestone: materialize versus elide)

The materialize-versus-elide divergence dissolved under evidence: the
"materialize" disposition never produced an executable kernel.  The
legacy plan-free emission for a dense root-scope reduction mixes the
dense workspace API in the producer (`memset` plus indexed
accumulation) with the sparse coordinate-map API in the consumer
(`.sort()` plus iteration) over one never-declared symbol, and the
public spelling `scorch.einsum("jk->k", dense)` died in release with
clang `use of undeclared identifier 'wksp'` (broken kernel source and
both clang transcripts retained, §25.6).  `1d7a6db` aligns plan-free
production auto-scheduling with the only executable established
semantics — the empty-Schedule replay contract's elision — scoped
exactly to the broken family: dense output with the last in-order
reduction at the nest root.  Sparse-output root insertions still
materialize and record their fact.  After the alignment, production
surgery and replay agree by construction, F4 records the honest
elided decision instead of failing closed, the einsum spelling
compiles and matches `torch.sum`, the root family flows tile-free
through LoopIR byte-identical to legacy with a compiled bitwise
shadow, and every previously-compiling emission is byte-unchanged
(corpus, grid, and audit receipts below).  The complete-plan guard
remains as fail-closed defense in depth.  One honest residue: with a
plain `torch.Tensor` operand (rather than a dense `STensor`) the same
einsum spelling now proceeds past compilation and fails later at a
pre-existing result-wrapping gap ("Could not infer dtype of STensor"),
which is unrelated to the compiler path and remains open.

### 25.4 The regblock_dual ownership boundary (Milestone: dual disposition)

Decision: the two constituent schedules are Phase-6 obligations and
both are discharged — the regblock-off arm through the tile-free
auto-replay contract and the regblock-on arm through the stack-form
contract, each independently byte-identical through LoopIR — while the
runtime free-dim branch that stitches the two lowered arms is
target-level composition whose migration belongs to Phase 7's
parallel/target-lowering work.  `dba2a22` locks the boundary with a
composition differential: reconstructing the stitch from the two arm
lowerings reproduces the production `_build_regblock_dual_path` kernel
byte-for-byte, so the dual route contains no third semantic lowering.
The existing branch-structure, cutoff-width, gate-off, and compiled
both-sides-of-cutoff numerical tests (all green this session, now part
of the compiled battery) cover the stitch's runtime behavior.

### 25.5 Phase-6 exit disposition

Criterion-by-criterion:

1. **Schedule decisions applied only to LoopIR:** met for the migrated
   explicit families, the tile-free auto-replay family (now including
   the aligned dense root-scope reduction), and the regblock
   stack-form family; not claimed for the remaining plan-free
   automatic families.
2. **No name/regex discovery in typed panel/relayout/result-tile
   passes:** unchanged and met; the new admission logic consumes typed
   plan facts only.
3. **Canonical identity owns the strangler request:** met by request
   v2, unchanged; the origin policy is deliberately outside the
   identity.
4. **Explicit and automatic differentials:** met for the explicit
   families, the tile-free family (including the aligned root case),
   the stack-form family, and the dual composition differential.
   Open for (a) the reduce-out strip-mine family — heuristic
   `OUTERMOST`/`CHILD_OF` tiles that strip-mine a dense reduction
   variable with an accumulate copy-out consumer (dense-dense
   matmul/SDDMM-shaped inputs; well-formed legacy C++ captured, no
   typed emission twin) — and (b) the sparse-workspace family
   (`coo_workspace` target API; SpMSpM-shaped and sparse-output root
   cases).  Both stay fail-closed at the gate with the stable
   `unsupported_schedule_auto_family` code.
5. **Representative tile-j/tile-ijk readiness:** unchanged and met.

**Verdict: Phase 6 remains open on the reduce-out strip-mine and
sparse-workspace automatic families (criterion 4).  There is no
Phase-6 GO**, and consequently no Phase-7 work, production cutover,
selector migration, cache cutover, or legacy deletion was started.
The §24 root-workspace and dual-route obligations are closed.

### 25.6 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-auto-origin-f3dee24/`
(capture, final, latency, full-suite, and root-workspace-evidence
subtrees).

- final ten-file contract membership at the code tip: **764 passed**;
  dedicated options/identity membership: **82 passed**;
- compiled schedule/pipeline/generality battery *plus the dual-path
  file* at the code tip: **283 passed in 1,038.70 s** (log SHA-256
  `dcada1e91cc636cf9690db4d0aaaae6e5b763e7c636af5a82b9fe03836c38565`,
  JUnit SHA-256
  `8ce75b46aa6a977a823611c9f2ddbc039a451bc8474fa9b54f19071b12aa615a`);
- fresh 86-case schedule audit at the code tip: **46 admitted / 40
  rejected / zero divergent**, two runs byte-identical, and the full
  result equal to the retained `c7e3702` results after removing only
  the embedded commit id (result SHA-256
  `8946526e3bd64be1a24dc5fdb7eaef4e6d9cb65ea216bf387d825fd3a2862b91`);
- fresh source captures byte-identical to the retained
  `phase6-review-c7e3702` finals: 20-source corpus, 42-cell grid, 11
  heap goldens, and the 22-cell anchor survey — **95/95**; plus
  **10/10** explicit-anchor and **11/11** heap LoopIR/legacy
  byte-identical comparisons re-run from patched copies of the
  retained harnesses;
- automatic root/tiled/dual source grids captured before and after the
  alignment (five CIN families × both regblock arms × F2/F4 routes,
  plus the three dual-path cases), including the broken root kernel,
  both clang failure transcripts, and the post-fix einsum success log
  under `root-workspace-evidence/`;
- paired same-session compiler latency (5 warmups / 30 samples, clean
  detached `f3dee24` base versus `dba2a22` candidate, identical
  per-case source hashes): `small_dense` **0.979/1.020**, `reduction`
  **1.006/1.048**, `csr_intersection` **0.994/1.018**, `sparse_union`
  **1.004/0.973** (p50/p95 new/old) — every ratio inside the 1.10
  target (base JSON SHA-256
  `75b36524a887103a9d6287dc6062d784998974d8cb581e5483d45f340dcff0c0`,
  candidate JSON SHA-256
  `5a06db8d1790011bc26778093edeef944046e29918fb9f7fac6039fb8e4b5f34`,
  comparison-log SHA-256
  `47e74ee9e3bd4a821484c4c50cef1cee9adae86c6a6f4256881175f0fb7882dc`);
- full-source Black reproduces only the inherited
  `prebuilt_kernels.py` finding and full-source Flake8 is
  byte-identical at the nine inherited findings; full-source mypy is
  **140 errors in 11 files**, line-identical between a clean detached
  worktree at base `f3dee24` and the candidate tree under the same
  repository-root invocation (the previously recorded 146/12 figure
  is reproducible only from the earlier session's different invocation
  context, so the honest same-methodology baseline is 140/11 with a
  zero-line delta from this session's commits); the four changed
  compiler files are focused-mypy-clean at both base and candidate;
  changed files are Black/Flake8-clean except the inherited scheduler
  C901;
- literal unpartitioned clean detached-worktree non-performance suite
  at exact `dba2a22` with isolated caches and asserted import
  provenance: **4,204 passed, 14 skipped, 3 performance tests
  deselected, one known sparse-invariant warning, and zero
  failures/errors** in **2,798.22 s (46:38)**; the run crossed the
  historical late-process region with no libomp/resource event, so no
  partition substitute or base control was needed (log SHA-256
  `8e0aaefd1e87e25fc43f45ffb4b612c58f8af1d1ac4807a56592d1ec1d3db124`,
  JUnit SHA-256
  `eabe603cf408de99ef2fff26266b546c2636744e605410090d6158af4f1a0d42`); and
- `git diff --check` is clean, local and live origin remain at
  `58e8565`, and all five protected files retain their recorded
  hashes; the only tracked files staged were the explicit compiler
  and test pathspecs of the five commits.

## 26. Automatic-origin review corrections and the real dual boundary (2026-07-26)

This section supersedes §25's current-state and exit claims.  Section
25 remains the historical receipt for `e1afa72..e4c5cb7`; its root
workspace fix remains valid in the reviewed single-assignment scope,
but its automatic-decision and dual-closure claims were too broad.

The review inspected all six inherited commits and reproduced their
focused gates before writing a fix.  It then used direct cross-route
differentials, hostile environment controls, malformed direct-pass
inputs, format/rank/mode-order matrices, and actual production dual
fixtures rather than accepting the handoff's conclusions.

### 26.1 Concrete findings

Five issues were found.

1. **The automatic stored-equals-derived verifier missed three legacy
   decisions.**  Workspace storage was derived from whole-result
   density rather than the workspace axis's result level; sparse
   workspace insertion's addition of the producer reduction to
   `no_tile_list` was not mirrored; and the post-`Where` sparse
   retraversal test was incorrectly applied below the common direct
   loop prefix.  Valid F2/F4 plans consequently failed with
   `auto_workspace_decision` or `auto_tile_decision` for mixed-level
   results, sparse workspaces, and rank-four post-insertion nests.

2. **The policy trust boundary stopped at semantic verification.**
   `verify_loop_plan` required `AutoOriginPolicy` on automatic plans
   and rejected it on explicit plans, but direct
   `apply_schedule_plan` accepted both missing-policy automatic plans
   and explicit plans carrying the policy.  That pass has a verified
   plan precondition and does not replace semantic verification, but
   the enforceable provenance invariant still belongs at every
   consuming boundary.  Excluding the policy from canonical/request
   identity is sound only after semantic verification.

3. **Two new F2/F4 regressions were environment-dependent.**  They
   forced F4 to the regblock-off arm but allowed F2 to read ambient
   `SCORCH_REGBLOCK`; `SCORCH_REGBLOCK=1` made correct production
   behavior fail the tests.  Both routes now receive the same
   environment-free `CompileOptions`.

4. **The original dual differential was vacuous, and the first
   correction still used a surrogate.**  `dba2a22` reconstructed the
   production dual kernel from the same two legacy lowerings the
   production helper used, so it passed even before either arm was
   migrated.  `7c76b7b` improved this to actual LoopIR-produced arms,
   but silently changed the fixture from the public frontend's
   implicit reduction (`TensorAssign.op is None`) to an explicit
   `Operation.ADD`.  The explicit-ADD `ds` schedule/stitch is a useful
   schema-level proof; it is not evidence that the public dual route
   enters LoopIR.  The real `ds` input still fails closed at
   `unsupported_reduction_without_update`; once represented with
   explicit ADD, dense matmul reaches the unmigrated reduce-out
   schedule family and hierarchical `ss` reaches the unsupported
   target shape.

5. **One positive pass test used a semantically invalid plan.**  Its
   hand-built tile-free dense-matmul plan was accepted by the pass
   precondition but correctly rejected by `verify_loop_plan` because
   the automatic policy derived tiles and a workspace.  The positive
   fixture is now a genuinely tile-free dense elementwise plan and
   explicitly proves semantic verification before pass consumption.

### 26.2 Corrections

Seven focused commits implement and lock the corrections:

- `77a2cef` — `fix(compiler): mirror automatic workspace decisions`
- `bfd6692` — `test(compiler): cover exact automatic workspace derivation`
- `5a4a932` — `fix(compiler): preserve automatic policy at LoopIR passes`
- `b6be7dd` — `test(compiler): lock automatic policy pass boundaries`
- `7c76b7b` — `test(compiler): prove dual composition from LoopIR arms`
- `8b3b45d` — `test(compiler): isolate automatic-origin arm selection`
- `fe6bdb2` — `test(compiler): narrow automatic and dual evidence`

The derivation now has the same two phases as legacy scheduling:
pre-insertion candidates decide whether workspace materialization is
useful; after insertion, workspace representation, sparse
`no_tile_list`, the surviving common prefix, and final tile choices
are re-derived from typed facts.  `apply_schedule_plan` now enforces
the missing/stray policy provenance boundary before family dispatch.
No schema token changed: `scorch.autopolicy.v1`,
`scorch.loopplan.canonical.v1`, `scorch.loopir.request.v2`, and
`scorch.loopir.canonical.v8` remain accurate because no serialized
representation changed.

Independent non-native generality checks found no remaining production
derivation mismatch: a 1,376-case structured rank/layout/arm sweep had
940 successful exact F2/F4 plans and only expected legality
rejections; a separate ranks-two-through-five randomized sweep had
896 successful F2/F4 plus replay/direct-surgery equalities; an
exhaustive rank-three layout/mode-order matrix had 2,448 successful
arm cases; and dense/sparse root matrices preserved dense elision and
sparse materialization.  No sweep produced
`auto_tile_decision`/`auto_workspace_decision` after the fix.

### 26.3 Corrected Phase-6 disposition

The dense root-workspace elision remains closed for the reviewed
single-assignment family.  The typed automatic policy remains viable,
and normalized explicit-ADD `ds` stack-form schedules are migrated.
The explicit-ADD dual stitch is byte-identical when built from actual
LoopIR arm lowerings, but it is only a schema-level target-composition
proof.

**Phase 6 has no GO.**  Its production closure now has four linked
obligations rather than §25's two:

1. normalize or otherwise faithfully bridge the public frontend's
   implicit reduction into typed LoopIR, with a direct public-path
   differential;
2. migrate the reduce-out strip-mine automatic family, which also
   blocks dense-matmul dual arms;
3. migrate the sparse-result/workspace boundary in its two distinct
   subfamilies: mixed-level result representation/assembly with a
   dense trailing workspace (currently `unsupported_format`), and
   true sparse `coo_workspace` allocation/reset/assembly with exact
   sparse `no_tile_list` behavior (sparse output may currently fail
   earlier at `unsupported_sparse_output`, not always at §25's claimed
   schedule-family diagnostic); and
4. disposition the full production dual-helper domain: keep the
   actual-LoopIR, explicit-ADD `ds` arm proof, close dense arms through
   reduce-out, and either migrate hierarchical-compressed `ss` target
   lowering or retain an explicit compatibility/fallback boundary.
   Only after every constituent arm is genuinely migrated can the
   runtime stitch be assigned wholesale to Phase 7.

Accordingly, §25's statements that the dual obligation is closed,
that exactly two families remain, and that all remaining cases reach
one stable `unsupported_schedule_auto_family` gate are withdrawn.

### 26.4 Verification

Exact-tip evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-auto-review-e4c5cb7/`.

- ten-file contract membership: **769 passed**; dedicated
  options/identity membership: **82 passed**; the hostile
  `SCORCH_REGBLOCK=1` replay regression: **2 passed**;
- the compiled schedule/pipeline/generality plus dual-path battery:
  **286 passed in 1,024.28 s (17:04)** (log SHA-256
  `b6940b5faeb6be02191c69b9b2021f6826b970cd421dd974d741f3be091f9918`,
  JUnit SHA-256
  `0450067680a9272f05f55962c8553afc9d15516a8d5e3e662b32c9817e3a2ed0`);
- source-only 86-case audit: **46 admitted / 40 rejected / zero
  divergent**, deterministic across repeat runs and byte-identical to
  the retained result after removing the embedded commit id
  (normalized SHA-256
  `93b6737cef94f8bbb56ba9b339a2cba23bfc68bd2b059129d3a8ae95a5c2b788`);
- clean detached source regeneration was byte-identical to the
  retained captures for all **20 corpus files**, **42 grid cells**,
  and **22 anchor-survey files** (manifest SHA-256 values
  `7e4a9c436e5ed1005874e9ece56847ea4ef88dd6f5a04c4d657c3ca5b37cd6c4`,
  `65e68ba19510ab240cf85574aa6be272dd6e5b0fdd7e799f7bcfe9db6d06c094`,
  and
  `44b39b55fbaea4bccbfac2aca6c217ca6b7ccd7135c8d50950efd96fd7f2dd13`);
  exact automatic-family legacy C++ also remained byte-identical to
  the post-root-fix retained capture; explicit-anchor LoopIR/legacy
  parity remained **10/10**, and heap parity remained **11/11**;
- changed source/test files are Black- and Flake8-clean; focused mypy
  reports no findings in the two changed production modules;
  full-source static results remain at the inherited one Black,
  nine Flake8, and 140-mypy-errors-in-11-files baselines (mypy
  category/content is line-normalized identical, with only the
  inherited scheduler import shifted by two lines); `git diff --check`
  is clean;
- paired same-session compiler latency versus inherited tip `e4c5cb7`
  stayed inside 1.10 with identical per-case source hashes:
  `small_dense` **1.005/0.993**, reduction **0.995/0.973**,
  `csr_intersection` **1.014/1.016**, and `sparse_union`
  **0.997/0.987** (p50/p95; comparison SHA-256
  `e736bd856c66f62661edaaccc06aa323135dc96a88d595cd67897c6dafbc479f`);
- literal clean detached-worktree non-performance suite at exact
  `fe6bdb2`: **4,212 passed, 14 skipped, 3 performance tests
  deselected, one known warning, and zero failures/errors** in
  **2,900.18 s (48:20)** (log SHA-256
  `ba88f6692a02b8b75c9b62471918f341bbeec4a7aeec9b35d1df361be6a7809b`,
  JUnit SHA-256
  `6d1376b8433fff93f7197739d758ca840f74a676815f1b54bdbc28ed0823222e`);
  JUnit records 4,226 selected tests and 14 skips.  The outer zsh
  evidence wrapper exited 1 only after pytest completed because it
  assigned the shell's read-only `status` variable; the complete
  pytest summary and zero-failure/error XML establish pytest exit 0,
  and the postprocessing error is retained rather than hidden or
  papered over with a 48-minute rerun; and
- local and live origin remained `58e8565`; all five protected files
  retained their recorded hashes, all unrelated dirty/untracked work
  remained untouched, and nothing was pushed.

## 27. Phase-6 closure milestone: implicit bridge, reduce-out, dual census, and the sparse boundary (2026-07-26/27)

This section records the broad closure milestone taken on top of §26's
corrected disposition.  Five code commits stack above `78e70b2`:

- `7e5f2ba` — `fix(compiler): derive tile candidates in logical access order`
- `a1b6887` — `feat(compiler): normalize the public implicit reduction at the LoopIR boundary`
- `9c45266` — `feat(compiler): migrate the dense reduce-out automatic family to LoopIR`
- `476bf94` — `test(compiler): census and close the dense production dual domain`
- `7d75a45` — `test(compiler): audit the sparse-result/workspace boundary exactly`

### 27.1 Independent review of the §26 commits

All eight inherited commits (`77a2cef` through `78e70b2`) were
re-reviewed from the diffs, and §26's focused gates reproduced exactly
(ten-file contract membership **769**, options/identity **82**, hostile
`SCORCH_REGBLOCK=1` replay **2**).  An independent randomized
cross-route sweep (ranks 2–5, mixed dense/sparse levels, permuted
physical mode orders, one-to-three operands, both arms; F4 origination,
F2 replay equality, and direct-legacy-surgery canonical equality per
case) found **one concrete defect §26 missed**: the automatic
stored-equals-derived verifier enumerated tile candidates from each
access's physical ``storage_index_ids`` while legacy
``_select_index_vars_to_tile`` walks the access's logical index list.
A permuted ``mode_order`` that swaps two dense candidates therefore
made the derived tile order diverge from the recorded surgery order
(stored ``m, j, k`` versus derived ``m, k, j`` on a rank-5 ``dsd``
result with a mode-order-permuted ``dddd`` operand), rejecting a valid
F2/F4 plan with ``auto_tile_decision``.  Tile order is semantically
meaningful because successive origin insertions stack LIFO.  Fixed in
`7e5f2ba` by walking ``logical_index_ids``; level-type pairing keeps
storage order, which is what ``level_type_of_index_var`` reports.
After the fix, sweeps at three seeds (1,100 randomized cases plus the
earlier structured matrices) report zero derivation mismatches and
zero route mismatches.  Adversarial probes confirmed the rest of §26:
semantic verification rejects arm/width flips wherever a decision is
observable, hostile policy states fail closed at both the LoopPlan and
consuming-pass boundaries, dense root elision and sparse root
materialization are unchanged, and the sparse-workspace no-tile and
retraversal-scope behaviors mirror legacy (the workspace access is
excluded from legacy candidate selection because ``tensor_accesses``
omits workspaces, which the sweep verified against production).

### 27.2 Workstream 1: the public implicit-reduction bridge is closed

The public einsum/matmul frontend deliberately builds
``TensorAssign.op is None`` for reductions; legacy iteration analysis
re-derives the additive update from the assignment's
right-hand-side-only index variables (``iter_lattice.get_simplified_cin``
discards the recorded op entirely) and emits the same C++ as an
explicit ADD update — verified byte-identical for ``ds``, ``dd``, and
``ss`` matmul in both regblock arms.  `a1b6887` owns that
normalization exactly once at the CIN→LoopIR boundary: an op-``None``
assignment whose reduction loops all appear in right-hand-side
accesses lowers as the ADD update, and no later stage re-infers the
fact.  ``verify_cin``'s ``unused_index_binding`` invariant guarantees
every non-lhs loop variable has a right-hand-side use, and the
boundary still fails closed on any unprovable shape.  Elementwise
op-``None`` assignments keep plain overwrite stores (the bridge cannot
manufacture updates), and repeated-operand rejection is unchanged.
Agreement is proven among the public implicit spelling, the normalized
explicit-ADD spelling, legacy C++, LoopIR C++, and compiled execution
(bitwise vs legacy, tolerance vs PyTorch), covering multiple
reductions, repeated operands (still rejected), empty extents,
f32/f64, and non-reduction assignments.

### 27.3 Workstream 2: the dense reduce-out automatic family is migrated

`9c45266` represents the legacy automatic strip-mine composition as
one fused typed pass.  ``apply_reduce_out_tiles`` builds the workspace
region with a strip-mined reduction producer (the reduction point loop
wrapping the axis point loop over one ``WorkspaceReduce``) and an
accumulate copy-out consumer, then inserts each recorded origin loop
at the arm placement with legacy ``add_tile`` LIFO semantics —
OUTERMOST (regblock-off, width 32) stacks origins above the prefix;
``CHILD_OF`` (regblock-on, width 8) stacks them under the row loop.
Prefix candidates split in place, so the rank-3 three-tile family
(TTM) is covered as well as two-tile dense matmul and the
three-operand form.  The plan-family gate admits the dense reduce-out
replay contract (arm-uniform affine/direct/serial/unroll tiles at the
recorded policy width splitting the last reduction loop and the dense
workspace axis exactly once each); the stack form and tile-free
contracts are unchanged and the sparse-workspace family stays
fail-closed.  Target lowering accepts strip-mined reduction point
loops inside a workspace producer when their origin is on the outer
chain; erasure and the oracle cover the new shape without
modification.

Evidence: generated source byte-identical to the legacy automatic
surgery through the real empty-Schedule route for dense matmul
(explicit, implicit, f64), TTM, three-operand matmul, and
ragged/exact/unit/oversized/zero extents in both arms — sixteen of
sixteen cells — including the legacy tile-count parallel work estimate
``(B1_size + kTile_j - 1) / kTile_j`` on the off arm; erasure returns
the exact base program and the oracle agrees with the erased program
and the reference on all twelve extent-class cells; compiled kernels
execute bitwise-identically to the legacy production auto route and
match PyTorch on both arms.  Two honest limitations are recorded: the
``execute_shadow`` utility freezes automatic plans into explicit legacy
schedules and cannot express workspace facts, so the compiled
differentials pair the LoopIR route with the legacy production auto
route directly; and the legacy F2 validator itself crashes
(``Affine reduction tiling requires an accumulator spanning outer
tiles``) for shaped dense matmul whose cost ordering picks
reduction-innermost — a pre-existing legacy limitation invisible in
production because both-dense dispatch routes to ``torch.matmul``.

### 27.4 Workstream 4: the production dual domain is censused and its dense constituents closed

`476bf94` locks the complete ``_build_regblock_dual_path`` census.
The admitted dense constituents — ``ds@dd`` SpMM (both public
spellings), dense matmul, dense rank-3 TTM, and the three-operand
``ds@dd`` form — each reconstruct the production dual kernel
byte-for-byte by stitching the two actual LoopIR-produced arm
lowerings, so no admitted dual kernel's evidence compares legacy
helpers to themselves.  The open boundaries stay locked with their
precise codes in both arms: hierarchical-compressed ``ss`` operands
fail target parent-position descent at ``unsupported_program_shape``,
COO operands fail level lowering at ``unsupported_format``, and
trailing-compressed operands (``dd@ds``, ``dds`` TTM) derive
sparse-workspace-adjacent automatic plans kept on the legacy path at
``unsupported_schedule_auto_family``.  Non-qualifying families (SpMV,
``ds@ds``, dense SDDMM) provably build no dual.  Release behavior is
unchanged everywhere: production dispatch still builds the dual kernel
from the legacy helpers and the strangler path is not live.  The
runtime free-dim branch remains target-lowering state owned by
Phase 7.

### 27.5 Workstream 3: the sparse-result/workspace boundary is audited, not migrated

`7d75a45` locks the two remaining representation seams with exact,
arm-independent codes.  Mixed-level results whose trailing workspace
axis is dense record their dense-workspace F2/F4 plans (closed by
§26/`77a2cef`), but the compressed-parent/dense-leaf result
representation fails at ``unsupported_format`` — and the family's
legacy comparand is defective: its generated C++ reserves but never
sizes the values vector it writes through, never appends row
coordinates or advances the position cursor, its execution fails at
result wrapping (``TensorIndexError``), and the generic-path
``ds``-output SpMSpM configuration crashes with SIGSEGV.  Defect
evidence is retained under
`phase6-ws-closure-7e5f2ba/ws3-boundary/`.  No byte-parity gate can
honestly be widened against a comparand that cannot execute, so the
fail-closed boundary is the correct Phase-6 disposition for this seam.
True sparse ``coo_workspace`` families fail closed at their own stable
codes: row-scope SpMSpM and reduction-to-CSR at
``unsupported_sparse_output_reduction``, merged sparse reductions with
dense outputs at ``unsupported_merged_reduction``, and sparse-output
roots at the early ``unsupported_sparse_output`` boundary the census
audits explicitly.  The public SpMSpM route does not use the defective
generic configuration and remains correct, so this seam is a genuine
migration target for the next milestone.

### 27.6 Corrected Phase-6 disposition

Of §26's four linked obligations: (1) the public implicit-reduction
bridge is **closed**; (2) the reduce-out automatic strip-mine family
is **migrated**; (4) the production dual domain is **censused, its
dense constituents closed from actual LoopIR arms**, and its remaining
arms dispositioned as precise fail-closed boundaries with release
behavior unchanged.  Obligation (3), the sparse-result/workspace
boundary, remains **open**: its plan layer is closed and both
representation seams are exactly audited, but neither seam has a typed
emission twin (one of them has no executable legacy comparand at all).

**Phase 6 therefore still has no GO.**  It is open on exactly one
boundary: the sparse-result/workspace representation (mixed-level
result assembly with a dense trailing axis, and true sparse
``coo_workspace`` allocation/reset/assembly for row-scope SpMSpM and
sparse-output roots).  No Phase-7 work, production cutover, selector
parity, cache cutover, legacy deletion, Phase 8, or Phase 8.5 was
started.

### 27.7 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-ws-closure-7e5f2ba/`.

- ten-file contract membership: **774 passed**; dedicated
  options/identity membership: **82 passed**; hostile
  `SCORCH_REGBLOCK=1` replay of the three environment-sensitive F2/F4
  regressions: **4 passed**;
- expanded cross-route automatic-plan census: 1,900 randomized cases
  (ranks 2–5, mixed levels, permuted physical mode orders, one to
  three operands, both arms) across five post-fix sweep runs at four
  seeds, each case checking F4 origination, F2 replay plan equality,
  and direct-legacy-surgery canonical equality: **zero derivation
  mismatches and zero route mismatches** (only expected
  `sparse_parent_dominance` / `result_storage_order` legality
  rejections);
- complete compiled schedule/pipeline/generality plus dual-path
  battery at exact `28f3424`: **316 passed in 1,052.46 s (17:32)**,
  zero failures (log SHA-256
  `0d6ee88f0fa79627f2ef14e4b05b5a4325c7c8a78778414032f8cce30bd18d3c`,
  JUnit SHA-256
  `6c6fc0dfa6e43fb0c96e21893e6c5372b803249456eb95cfa26b222d203d732d`);
  a first run at `7d75a45` passed 315/316, the single failure being
  the stale scheduled-slice lock of the pre-migration tiled-auto
  boundary, updated in `28f3424` to prove the reduce-out replay
  instead;
- source-only 86-case audit at the tip: **46 admitted / 40 rejected /
  zero divergent**, two runs byte-identical (JSON SHA-256
  `bdce0004d0a76681d31c1e9bc25f383fec98387a9a6f7e51bfcf6fb86d28854f`)
  and equal to the retained `fe6bdb2` result after removing only the
  embedded commit id;
- fresh clean-detached captures at `7d75a45` (src byte-identical to
  `28f3424`; the intervening commit touches tests only): corpus
  **20/20**, grid **42/42**, anchor survey **22/22**, and heap goldens
  **11/11** byte-identical to the retained baselines (manifest
  SHA-256 values
  `725bb5934bf57e04f37122789e8848dfa13ef8fd432d9554bf0d056c6adfa7c5`,
  `08041990376fc87cbd4ddfb91baa8e71ff3ea9646366ebee03c61bc96e92ba7b`,
  `c2a4b9873a8ab34f94bf02e27bc683b3f9c3f69e2275f24c04417a66de53ed8f`,
  `b2a76cb3623a79129379adff9732456c35a984e686094a24d27fadc7c2f0c7d0`);
  explicit-anchor LoopIR/legacy parity **10/10** and heap parity
  **11/11**;
- automatic root/tiled/dual source grids: all **22** legacy CIN/C++
  and dual files byte-identical to the retained capture (manifest
  SHA-256
  `ac5f7f595cb668d93c5ec9395d29dbf353d534191a74d85e7882579fd7b34926`);
  the only report delta is the dual `cache_key_suffix` window, which
  embeds the capture worktree's `scorch_python_path` in the build
  fingerprint — every other build-key component is byte-equal, so the
  delta is the capture location, not a semantic change;
- changed source/test files are Black- and Flake8-clean; focused mypy
  reports no findings in the four changed production modules
  (`loop_plan_legality.py`, `loopir/lower_cin.py`,
  `loopir/schedule_passes.py`, `loopir/lower_llir.py`); full-source
  static results remain at the inherited one-Black, nine-Flake8, and
  **140-mypy-errors-in-11-files** baselines with every line-normalized
  mypy error identical to the retained log; `git diff --check` is
  clean;
- paired same-session compiler latency, clean detached `78e70b2` base
  versus `28f3424` candidate (5 warmups / 30 samples, native work
  excluded, identical per-case source hashes): `small_dense`
  **0.802/0.643**, reduction **0.980/0.884**, `csr_intersection`
  **0.945/0.955**, and `sparse_union` **1.015/1.066** (p50/p95) — all
  inside the 1.10 target (comparison SHA-256
  `5e5f06c58da8369127cc6523000ea69bf494af425ec178270ba9b70777cba5a5`);
- literal clean detached-worktree non-performance suite at exact
  `28f3424` with isolated caches and asserted import provenance:
  **4,250 passed, 2 failed, 14 skipped, 3 performance tests
  deselected, one known warning in 2,954.77 s (49:14)** over 4,266
  selected tests (log SHA-256
  `4bf7a89d8a8fcb8bb8570bfe30c72f06abce57f44c8556a0b47c9fadca45c9f6`,
  JUnit SHA-256
  `357115f21b0cf6de7297bedec8abeaa45a42383aa0a3fe627584a48e98ead3f5`).
  Both failures are the known macOS libomp resource ceiling — the JIT
  build subprocess aborts with ``OMP Error #179: pthread_key_create
  failed: Resource temporarily unavailable`` late in the 49-minute
  JIT-heavy process
  (`test_schedule_generality::test_spmv_and_dense_matmul_default_numerics_are_unchanged`
  and
  `test_value_object_boundaries::test_permuted_coo_sddmm_skips_canonical_native_shortcut`),
  the same environmental mode retained at `3f17ec6`; this session's
  added JIT tests grew the suite from 4,226 to 4,266 selected, and the
  retained green `fe6bdb2` literal receipt is the base control below
  the ceiling.  Attribution and closure: `test_schedule_generality`
  passed completely inside the same-tip compiled battery, and both
  affected files rerun **81/81 green in 203.81 s** in one fresh
  isolated process at the same worktree (log SHA-256
  `a136f89de19eabe31593d679fc5fddec40eeae8c07a8fb8b774d05a94f9c16f6`,
  JUnit SHA-256
  `16be3bdc0b331cbda2528f22bd818dac8dad617b924c3b3426e695a48acc3b1f`),
  so every selected test passes in a clean process and no failure is
  attributable to this session's code;
- local and live origin remained `58e8565`; all five protected files
  retained their recorded hashes before every commit, all unrelated
  dirty/untracked GPU/CUDA, benchmark, packaging, scheduler, research,
  scratchpad, and tooling material remained untouched, and nothing was
  pushed.

## 28. Independent review corrections to the Phase-6 closure milestone (2026-07-27)

This section supersedes §27's closure verdict and evidence claims where
they conflict.  The implementation commits `7e5f2ba`, `a1b6887`, and
`9c45266` remain sound under independent review; the review found no
wrong-result or emission regression in their logical-order automatic
derivation, implicit-reduction normalization, or fused reduce-out pass.
It did find one adjacent compiler trust-boundary defect and several
material gaps in the claimed census and retained evidence.  They are
corrected in:

- `a920e10` — `fix(compiler): reject malformed CIN structure`; and
- `8fb4902` — `test(compiler): close Phase-6 review gaps`.

No serialized schema changed.  `scorch.autopolicy.v1`,
`scorch.loopplan.canonical.v1`, `scorch.loopir.request.v2`, and
`scorch.loopir.canonical.v8` remain current.

### 28.1 Analyzed and LoopIR-owned malformed CIN now fails closed

Fresh adversarial probes found that a forged `ForAll.stmt` self-cycle
hung `_collect_loop_nest`, expression cycles escaped as
`RecursionError`, and missing `ForAll.parallel`, `ForAll.stmt`, or
`TensorAssign.op` fields escaped as raw `AttributeError`.  These are
pre-existing defects, but they sit directly on the newly exercised
CIN→LoopIR boundary.

`a920e10` adds one iterative stored-forward-structure preflight with an
active/complete DFS and a 256-edge depth bound.  It returns stable
`missing_cin_field`, `invalid_cin_field`,
`cyclic_cin_structure`, or `cin_structure_depth_exceeded`
diagnostics before recursive ownership analysis, nest collection,
metadata binding, or kernel preparation.  Direct lowering and the
compile, execute, and shadow entries share the boundary; the latter
preflight before the LoopIR pipeline invokes requested scheduling.
Completed shared subgraphs are not mislabeled as cycles: they continue
to the existing ownership analysis, which retains authority for
`duplicate_node_reference` and `duplicate_access_reference`.

This correction does not claim a repository-wide CIN walker boundary.
Plan-free `normalize_cin` does not call the structural preflight in
normal release mode, so a forged missing field can still escape as
`AttributeError`.  Direct `Scheduler.apply_schedule` has a stronger
gap: its display-name validation runs before normalization and can
leak the same error even when full debug verification is enabled.
Moving the preflight to those shared production boundaries requires
its own latency-gated compatibility decision.

### 28.2 The reduce-out implementation is sound; its oracle receipt was missing

Independent source comparisons extended the fused reduce-out pass to
34 additional programs with two through six tiles in both automatic
arms; placement, target lowering, deterministic identities, and source
parity held.  The §27 claim that the oracle covered all twelve
extent/arm cells was nevertheless not backed by committed tests.

`8fb4902` adds the missing executable oracle/erasure lock: zero
reduction, zero output axis, unit, exact, ragged, and oversized extents
in both arms, plus a rank-three output with two reductions and four
tiles.  Each cell compares scheduled, base, and erased execution with
an independent contraction reference and requires canonical erasure
to recover the exact base program.

One wider dense family is deliberately not admitted.  The legacy
automatic order `a,b,d,c,e,f` for a rank-six contraction emits `pA3 =
pA2 * ...` before declaring `pA2`; the same explicit family was already
invalid before `9c45266`.  LoopIR's `unsupported_loop_order` is the
correct boundary.  Byte parity is not evidence when the comparand is
non-compiling C++.

### 28.3 The dual census was representative, not complete

The literal §27 statement that `476bf94` locked the complete production
dual domain is withdrawn.  Additional identity-layout batched and
four-operand families do reconstruct byte-for-byte from their actual
LoopIR arms and are now locked.  A batched sparse-operand family stays
at `unsupported_schedule_auto_family`.

More importantly, the public-aligned `einsum("ij,kj->ik", ds, dd)`
family reaches the dual helper with the dense operand represented as
logical `B[k,j]` over physical `mode_order=[1,0]`.  Both LoopIR arms
fail closed at `unsupported_loop_order`: the current target does not
own general non-identity dense position-chain lowering.  That is a
real Phase-6 boundary, not part of the sparse-result/workspace
representation.  The committed census is now explicitly described as
a representative identity-layout census instead of a complete
production census.

### 28.4 The sparse audit and retained-evidence statement are corrected

The boundary matrix now separately locks:

- dense-domain and sparse-row writes to CSR at
  `unsupported_sparse_output_domain`;
- merged explicit updates at `unsupported_merged_update`;
- mixed compressed-parent/dense-leaf elementwise and reduction forms
  at `unsupported_format`; and
- the existing sparse-output reduction, merged-reduction, and root
  boundaries.

The retained `ws3-boundary` directory contains one generated mixed-`sd`
source file.  That source proves the unsized values-vector and missing
coordinate/position assembly defects by inspection, and its header
records an observed wrapping error.  It does **not** contain a
standalone reproducer or crash transcript for the separately asserted
generic `ds`-output SpMSpM SIGSEGV.  §27's statement that SIGSEGV
evidence was retained is therefore withdrawn; that observation must be
reproduced before it is used as a gate premise.

Finally, §27 listed five code commits but omitted the later
`28f3424` scheduled-replay test commit.  The historical milestone has
six code/test commits before its documentation commit.

### 28.5 Corrected Phase-6 disposition

**Phase 6 still has no GO, and it is not open on exactly one
boundary.**  At minimum, two independent representation/target
families remain:

1. sparse result/workspace ownership: mixed-level result assembly and
   true sparse `coo_workspace` allocation/reset/insertion/assembly; and
2. general logical-coordinate to physical-position lowering for
   non-identity dense layouts, including the transposed dense dual
   constituent.

The high-rank legacy use-before-declaration case is a separate
correctness boundary until a valid target order is proven; it must not
be "migrated" by reproducing invalid C++.  The runtime dual stitch
cannot be assigned wholesale to Phase 7 until a genuinely complete,
format/layout-aware constituent census is closed or explicitly
dispositioned.

No Phase-7 work, production cutover, selector parity, cache cutover,
legacy deletion, Phase 8, or Phase 8.5 is claimed here.

### 28.6 Verification

The review gates are recorded at the final documentation commit:

- broad pure CIN/LoopPlan/LoopIR/schedule-API membership: **918
  passed**;
- the complete compiled scheduled-slice, pipeline-execution,
  schedule-generality, and dual-path battery: **331 passed in
  1,057.39 s (17:37)**;
- changed production and tests: Black, Flake8, focused mypy, and
  `git diff --check` clean; clean detached full-source base/candidate
  comparison retained the exact inherited baselines (**one** Black
  file, **nine** Flake8 findings, and **140 mypy errors in 11 files**)
  with byte-identical line-normalized Flake8 and mypy output;
- clean detached non-performance collection proved a complete
  non-overlapping union of **4,306 selected tests** (zero missing,
  extra, or duplicate node IDs): sequential fresh processes passed
  **4,292**, skipped **14**, deselected **3** performance tests, and
  failed **zero** in 3,012.31 pytest-seconds; no rerun or libomp
  resource event occurred;
- paired same-session compiler latency versus `fc94ec1` stayed inside
  1.10 with identical source hashes in every case: `small_dense`
  **1.003/0.980**, reduction **1.003/0.637**,
  `csr_intersection` **0.973/0.963**, and `sparse_union`
  **1.009/1.042** (p50/p95); a standalone clean-worktree execution of
  public `einsum("ij,kj->ik", ds, dd)` also matched PyTorch; evidence
  is retained under
  `/Users/bobby/.cache/scorch-codex/phase7-infra-8fb4902-full.sWQ9jZ/`;
  and
- local and live origin remained `58e8565`; protected hashes and all
  unrelated material remained unchanged; nothing was pushed.

## 29. General dense-layout lowering, the shared structural boundary, and the corrected exit disposition (2026-07-27)

This section records the broad milestone taken on top of §28.  Four code
commits stack above `80402b3`:

- `56772e4` — `fix(compiler): own the structural preflight at the shared scheduler/normalize boundary`
- `e8866b9` — `fix(compiler): reject descriptor-diverged CIN structure`
- `c69c839` — `feat(compiler): lower logical coordinates onto permuted dense layouts`
- `4e63500` — `style(compiler): keep the divergence diagnostic on one line`

### 29.1 Independent review of the §28 commits

The diffs of `a920e10`, `8fb4902`, and `80402b3` were re-reviewed without
trusting the handoff, and the focused gates reproduced: the broad pure
CIN/LoopPlan/LoopIR/schedule-API membership passes green (a 13-file set
collects 919; the recorded 918 is a one-test set-definition difference,
not a failure), and the complete compiled scheduled-slice,
pipeline-execution, schedule-generality, and dual-path battery passed
**331 in 1,116.22 s (18:36)** at the inherited tip.  Fresh adversarial
probes beyond the committed locks all held: forged container types,
enum-lookalike operations, `Where` cycles, workspace missing fields,
shared diamonds (still `duplicate_node_reference`, never mislabeled
cyclic), and the exact depth boundary (a 257-edge chain is allowed, a
258-edge chain reports `cin_structure_depth_exceeded` once).

Two findings were material:

1. **The §27 SIGSEGV assertion is now reproduced with retained
   evidence.**  The generic-path `ss@ss->ds` SpMSpM configuration through
   legacy `lower_and_exec_cin` with nest order `i,k,j` exits with signal
   11: the generated kernel declares `std::vector<float> C_values;`
   without sizing it and writes through `C_values[pC1] +=`.  The same
   configuration under nest `i,j,k` leaks
   `ValueError: ivar_k is not in list` from `cin_lowerer.py`
   `level_of_index_var`.  The kernel source and exit transcript are
   retained under `phase6-layout-56772e4/probes/probeA/`
   (`sigsegv-ss-ss-ds-ikj-main.cpp`).  §28.4's demand is satisfied; the
   fail-closed LoopIR boundary for this family is confirmed correct, and
   byte parity there would reproduce a memory-corrupting kernel.
2. **One preflight divergence.**  The structural preflight validated
   stored `__dict__` state while every recursive consumer walks the
   getattr view, so a `__class__`-swapped subclass whose property returns
   a different object (a self-cycle over a benign stored child) passed
   the preflight and leaked `RecursionError` from the ownership analysis.
   `e8866b9` requires every stored structural field to be the exact
   object getattr reports and treats raising descriptors as hostile;
   both now produce the stable `invalid_cin_field` diagnostic.  All
   structural fields of the real CIN classes are plain instance
   attributes, so legitimate programs are unaffected; descriptors that
   execute non-terminating code remain outside the threat model, as they
   are for every consumer.  Cost: about one microsecond per program.

Reduce-out spot probes on three fresh programs outside the committed
tests (rank-3 TTM f64 ragged, three-operand, and a two-reduction
four-tile program) reproduced byte parity and exact erasure in both
arms, 6/6.  One scoping observation is recorded: §27's "bitwise-identical
to the legacy production auto route" generalizes only where the
production cost-model schedule coincides with the empty-Schedule
surgery; the committed source-level byte parity is the valid gate for
the other cells.

### 29.2 The shared scheduler/normalize structural boundary (`56772e4`)

Both §28.1 gaps were first re-confirmed live: plan-free release-mode
`normalize_cin` leaked raw `AttributeError` from the clone walk, and
direct `Scheduler.apply_schedule` leaked from the display-name walk
(`cin.py` `is_workspace`) before normalization even with full debug
verification enabled.

`_normalize_cin_owned` now runs the bounded iterative preflight first,
so every normalization caller fails closed with the stable structural
diagnostics before any recursive work, and the three raw-CIN scheduler
entries (`apply_schedule`, `auto_schedule`, `auto_schedule_plan`)
normalize first and validate legacy display names on the normalized
clone, which preserves every display name and stable ID.  The options
boundary stays first; the display-name failure itself is unchanged.  No
path scans the graph more than twice (pipeline entry plus normalize),
and the pipeline-entry preflights are retained because the committed
monkeypatch contract requires entry-owned failure before
`Scheduler.apply_schedule` is invoked at all.

Measured cost of the added scan: 10.3/14.5 microseconds per normalize
(3-loop SpMM / rank-5 contraction), 0.88%/0.70% of a scheduling-only
`apply_schedule` call, and invisible against any JIT compile; the paired
compiler-latency gate below re-proves end-to-end neutrality.  New
error-ordering locks: structural failure before legacy surgery,
structural failure before display-name conflicts, option mismatch before
the CIN is touched, debug-mode missing-field reporting, and both
automatic entries.

### 29.3 General logical-to-physical dense layout lowering (`c69c839`)

The production activation case is the §28.3 boundary: public
`einsum("ij,kj->ik", ds, dd)` reaches the dual helper with the dense
operand as logical `B[k,j]` over physical `mode_order=[1,0]`, and both
typed arms failed at `unsupported_loop_order`.

The representation was already level-based — `LevelDecl.mode` names the
logical mode each physical level stores — so the change is confined to
construction and validation, with no format-specific shortcut and no
serialized representation change: tensor checking validates the stored
order as an exact permutation and returns the storage modes;
storage-order legality walks the access's logical indices through that
order (reducing exactly to the historical check under identity); both
declaration sites emit real level modes; dense position chains are
driven by the logical coordinate each physical level stores in the CIN
lowering, the iteration-domain analysis, and the target's level-driver
collection; the dimension-extent table binds runtime physical shapes per
level; and the kernel ABI embeds the tensor's real mode order.  The
oracle needed no change: its dense values are logical nested lists and
its position arithmetic was already level-mapped.  Admission is scoped
to all-dense operands.  Permuted compressed structure and permuted
results keep the stable `unsupported_mode_order` boundary, the full CIN
verifier owns non-permutation orders (`tensor_mode_order_mismatch`), and
a forged compressed permutation dies at the verifier's dimension-domain
rules before the target; the target checks remain as defense in depth.
`scorch.autopolicy.v1`, `scorch.loopplan.canonical.v1`,
`scorch.loopir.request.v2`, and `scorch.loopir.canonical.v8` remain
current — canonical program identity already serializes level modes and
request identity already serializes `mode_order`, so permuted layouts
already hash distinctly (now locked by test).

Evidence: generated source byte-identical to legacy in both regblock
arms for transposed SpMM (f32, f64, and zero row/reduction/free
extents), batched `[0,2,1]`, non-involutive `[1,2,0]` elementwise,
multi-operand, and transposed-matvec families — 18 committed cells plus
12 probe cells; compiled execution matches PyTorch under both runtime
marshalings (an identity-layout input relayouted by the binding twin,
and an input already carrying `[1,0]` storage), including every
zero-extent class; the production oracle agrees on the permuted dense
reduce-out program; the public einsum route executes and matches; and
the rank-6 use-before-declaration family stays rejected because the
storage-driver check reduces to the historical check under identity.
The full scheduled-slice byte-lock file passes unchanged (201 tests).

### 29.4 The dual census re-run and the cross-route sweep

`test_dense_dual_constituents_compose_from_actual_loopir_arms` now
covers **seven** constituents: the five §28 identity-layout families
plus `transposed_spmm` (`ds@dd`, B `[1,0]`) and `batched_transposed`
(`ddd@ddd`, B `[0,2,1]`), each reconstructing the production
`_build_regblock_dual_path` kernel byte-for-byte from the two actual
LoopIR-produced arm lowerings.  A derivation test ties both census
layouts to the production alignment formula
(`_bind_frontend_operand_mode_orders`) through both production branches:
the cost-selected branch derives `[1,0]` for `B("kj")` under the
selected `i,j,k` order, and the requested-schedule branch derives
`[0,2,1]` for `B("lkj")` under an explicit `l,i,j,k` order.  Two census
observations are recorded honestly: the all-dense batched spelling
`lij,lkj->lik` cost-selects `l,i,k,j` and therefore aligns to identity
(the permuted batched dense family is release-reachable via the
requested-schedule branch); and the cost-selected batched permuted
family is `dds@ddd`, which is now layout-admitted but stays locked at
`unsupported_schedule_auto_family` because its automatic plan is
sparse-workspace-adjacent — added to the boundary census as
`batched_transposed_sparse_operand`.

A randomized two-seed cross-route census over ranks 1-3 results,
one-to-three operands, sparse-leaf formats, and randomly permuted
all-dense operands (120 attempted cases, both arms per case) reports
**zero derivation and zero parity mismatches**; every rejection is an
expected legality code (`InvalidSchedule`, `unsupported_loop_order`)
except one pre-existing legacy limitation now recorded: a
broadcast-only result axis (`C[i,j] += A[k]*B[i]*D[i]`) crashes legacy
lowering with the raw assert `iter_lattice.py` "No lattice points
generated" while LoopIR lowers it; public einsum cannot express an
output subscript absent from every input, so the family is not
production-reachable, and this session did not change legacy.

### 29.5 The sparse-result/workspace boundary: designed and dispositioned, not implemented

A full implementation-ready audit was produced for the remaining
representation family; the honest scope decision is that its smallest
coherent slice is a full session by itself, and rushing it would have
violated the byte-parity/oracle/erasure/adversarial contract.  The audit
result is recorded for the next session:

- **Slice B1 (first, byte-parity-gated): serial `coo_workspace` plus
  multi-compressed-level append assembly**, activated by the public
  `ss@ss->ss` SpMSpM family.  The production comparand compiles and
  executes; one kernel exhibits every element the representation needs
  (per-row allocation/reset, merging insertion, sorted drain, ordered
  append, parent-linked positions across two compressed levels,
  empty-row parent omission, and empty-output finalize).  The proposed
  typed surface is four node kinds — a sparse workspace declaration over
  one drain dimension, an intrinsic producer/consumer region, a merging
  ADD insertion, and an ordered drain loop with a drain-value
  expression — with result assembly staying the single `AppendEntry`
  stream generalized from canonical-CSR-only to level-general
  (every level DENSE or COMPRESSED, lexicographically increasing
  appends, complete leaf-block coverage under compressed parents).
  Parent positions and coordinates remain derived target-assembly state,
  never IR nodes.  By the v7-to-v8 precedent, adding node kinds requires
  a `scorch.loopir.canonical.v9` bump; `scorch.loopir.request.v2` and
  `scorch.loopplan.canonical.v1` are untouched.  The plan layer is
  already closed (`WorkspaceInsertion.dense=False` with exact
  `no_tile_list` re-derivation), so the work is one new schedule pass,
  the family-gate widening to the sparse-workspace tile-free contract,
  verifier/oracle/erasure ownership, and target emission byte-identical
  to the retained comparand.
- **Slice B2 (separable correctness feature): mixed
  compressed-parent/dense-leaf assembly.**  The legacy comparand is
  defective by inspection: it appends values without assembling the
  compressed-parent coordinates, so it cannot provide valid sparse
  storage.  This migration is gated on the LoopIR oracle, PyTorch, and
  public-route differentials with an explicit no-legacy-comparand
  disposition.  The separately retained generic sparse-output reduction
  owns the SIGSEGV evidence; it is not evidence about this dense-domain
  B2 family.  The `sd`-operand load chain is a distinct adjacent gap.
- **Explicitly outside Phase 6: the two-pass OpenMP count/fill form**
  (public `ds@ds->ds` SpGEMM and `ds`-output requests).  Legacy emits a
  statically parallel two-pass kernel with per-thread workspace pools;
  a typed twin is inseparable from target-owned parallel/runtime-stitch
  policy and belongs to the same Phase-7 bucket as the runtime dual
  stitch, with the working public route as its execution oracle.

### 29.6 Phase-6 exit audit

Criterion by criterion: schedule decisions are typed and verified for
every migrated family (stack, panel, relayout, heap, parallel selection,
reduce-out, and now general dense layouts); the explicit/automatic
differentials hold across format and layout families wherever legacy is
a valid comparand; canonical request identity remains semantic and
distinguishes physical layouts without a version change; the
representative tile-j/tile-ijk compositions remain byte-ready; and
release behavior is unchanged everywhere (production dispatch still
builds the dual kernel from the legacy helpers; the strangler path is
not live; the latency corpus emits byte-identical sources).

**Phase 6 has no GO.**  After this session it is open on exactly one
production-relevant family cluster: the sparse-result/workspace
representation (slices B1 and B2 above).  The general dense-layout
boundary that §28 added is **closed** for the production dense domain.
The adjacent locked boundaries remain explicitly outside this cluster
with their stable codes: hierarchical-compressed `ss` operands
(`unsupported_program_shape`), COO operands (`unsupported_format`),
trailing-compressed automatic families
(`unsupported_schedule_auto_family`), permuted compressed structure and
permuted results (`unsupported_mode_order`), and the rank-6 invalid
legacy emission (`unsupported_loop_order`).  No Phase-7 work, production
cutover, selector parity, cache cutover, legacy deletion, Phase 8, or
Phase 8.5 was started.

### 29.7 Verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-layout-56772e4/`.

- broad pure membership after the changes: **1,127 passed** (13-file
  contract set plus scheduler/compile-options/neutrality/spike-verifier
  files); the 13-file set alone passes 919 before and after with only
  the committed re-scopes;
- compiled battery at the inherited tip: **331 passed in 1,116.22 s**;
  at the layout commit, scheduled-slice **201**, pipeline-execution
  **77**, and dual-path **30** all pass in working-tree runs, and the
  clean-detached battery receipt is recorded below;
- 86-case audit at the tip: **46 admitted / 40 rejected / zero
  divergent**, two runs byte-identical (JSON SHA-256
  `f54528305f2190cbc801c55a9293f54ffb74a191e2d7a75c0775fe54371cc698`),
  every case equal to the retained `phase6-ws-closure` baseline;
- fresh clean-detached captures at `c69c839`: corpus **20/20**, grid
  **42/42**, anchor survey **22/22**, heap goldens **11/11**, and
  auto-grid **23/23** files byte-identical to the retained baselines
  (auto-capture report equal modulo embedded worktree paths);
  explicit-anchor parity **10/10** and heap parity **11/11**
  byte-identical;
- randomized cross-route census: two seeds, 120 attempted cases
  including permuted layouts, both arms — **zero mismatches** (§29.4);
- paired same-session compiler latency, clean detached `80402b3` base
  versus `c69c839` candidate (5 warmups / 30 samples, native work
  excluded, identical per-case source hashes in every run): run 1 —
  `small_dense` **1.047/1.052**, reduction **1.047/1.033**,
  `csr_intersection` **1.009/0.996**, `sparse_union` **0.988/1.096**
  (p50/p95); a swapped-order run showed one `csr_intersection`
  excursion (**1.137/1.223**) that was investigated rather than
  averaged: the same-worktree A/A control swings 0.931-1.078 on this
  machine, a third paired run returned every case to **≤1.053**, and
  the mechanism bound (one 10-15 microsecond scan on a
  2-millisecond path) caps the true added cost under 1%, so the
  excursion is session noise, not a regression;
- full-source static parity at the tip in a clean detached worktree:
  Black **1 file** (the inherited `prebuilt_kernels.py` baseline),
  Flake8 **9 findings**, mypy **140 errors in 11 files** — all equal to
  the inherited baselines; mypy error lines are line-normalized
  identical, with one informational `note:` line relocated from
  `ops.py` to `scheduler.py` because the scheduler edit changed which
  import-untyped occurrence carries it; `git diff --check` is clean;
- clean-detached receipts at exact `c69c839` (detached worktree,
  isolated `TORCH_EXTENSIONS_DIR`/`XDG_CACHE_HOME`, asserted import
  provenance): the complete compiled scheduled-slice,
  pipeline-execution, schedule-generality, and dual-path battery passed
  **353 in 1,142.03 s (19:02)** with zero failures (log SHA-256
  `cba0396aa21dcd040055f348701f364d8060fdddf92b92def777aeff25b98ae0`;
  the growth from 331 is this session's added layout cells); the
  partitioned full non-performance suite over the retained complete
  non-overlapping file partition ran as two sequential fresh processes:
  direct collection selected **4,340** tests, partition A passed
  **3,933** with 14 skipped in 2,770.79 s (46:10), partition B passed
  **393** with 3 performance tests deselected in 296.11 s (4:56) —
  **4,326 passed, 14 skipped, zero failures in 3,066.90
  pytest-seconds**, with the JUnit union proving exactly 4,340 case IDs
  (zero missing, extra, or duplicate) and no libomp resource event
  (log SHA-256 values
  `bdcbe316b52efb22f20bf6670cd5fa05afee8ca16137635b78abccad22c4c24a`
  and
  `71e08f2441c4f446b55bbec3626cfbc16f961e4d6728a0d85c8dbd9ef6685627`);
  the later `4e63500` style commit changes one diagnostic string
  literal only, with the focused membership rerun green at that tip;
- local and live origin remained `58e8565`; all five protected files
  retained their recorded hashes before every commit; all unrelated
  dirty/untracked GPU/CUDA, benchmark, packaging, scheduler, research,
  scratchpad, and tooling material remained untouched; nothing was
  pushed, amended, squashed, or reordered.

## 30. Rigorous review corrections to the dense-layout milestone (2026-07-28)

This section supersedes the boundary and performance claims in §29 where
they conflict.  The review read `56772e4`, `e8866b9`, `c69c839`,
`4e63500`, and `a1d4851` in full, reproduced the focused gates, and then
challenged the admitted boundaries with forged descriptors, stored-state
mutation, non-identity result layouts, malformed runtime tensor metadata,
legacy workspace schedules, linked tile metadata, and deep discarded
compatibility state.  The inherited milestone had several concrete
fail-open and correctness defects.  The exact compiled/full-suite/latency
gates and a final independent adversarial pass then exposed six further
boundary mistakes in the review corrections and adjacent locks.  Fourteen
focused commits correct the complete set:

- `ce39b36` — exact, descriptor-free CIN admission; one-scan
  compiler-owned trust; complete dense-layout propagation; bounded ABI
  and runtime metadata ownership;
- `50fce1e` — adversarial and production-path coverage for that boundary;
- `a356148` — exact legacy-workspace compatibility, safe forward copying,
  and linked schedule-metadata validation;
- `7a8621c` — the corresponding alias, detachment, and raw-exception
  regression lock;
- `66a0817` — keep discarded scheduler compatibility state nonsemantic
  during normalization while validating it at raw legacy lowering; and
- `c0cd7fe` — lock both sides of that ownership boundary;
- `2ebead9` — preserve canonical diagnostics for the one exact admitted
  legacy schedule alias receipt without weakening strict normalization;
- `e82603d` — retain always-on access-rank structural validation; and
- `d14f87e` — align the compatibility and relayout regression contracts
  with those final boundaries;
- `8612679` — prove the compiler-built `einsum` root once across the two
  register-block arms and fallback scheduler, with stage-owned failure;
- `291f64e` — lock one-scan ownership, receipt cleanup, and the unchanged
  three-record normalization surface;
- `42d484f` — require copied optional storage on scheduler-detached
  logical tile parents; and
- `b946b58` — prove valid detached parents still lower while both missing
  fields fail with structured diagnostics; and
- `f13ba79` — extract the shared receipt accounting so `_einsum_owned`
  returns to its exact inherited Flake8 complexity baseline.

No serialized contract changed:
`scorch.loopir.canonical.v8`, `scorch.loopir.request.v2`,
`scorch.loopplan.canonical.v1`, and `scorch.autopolicy.v1` remain
current.

### 30.1 CIN admission is now one exact structural boundary

The inherited preflight mixed normal attribute access with stored-state
inspection and did not cover every structural consumer.  That let hostile
descriptors select a different graph after validation, let malformed
graphs reach recursive consumers through direct analysis/lowering
entries, and made the same compiler-owned root cross the bounded scan
repeatedly.  The §29 statement that release paths perform at most two
scans was not true: probes counted seven scans on the ordinary automatic
path and nine on the dual path.

The correction has one iterative, depth-bounded, descriptor-free
preflight over exact stored fields.  Structural fields must be present,
have the exact admitted built-in class/type/enum, and be read solely from
the admitted object's stored state; compiler admission never invokes
caller-defined descriptors.  Every raw-CIN entry passes through that same
preflight, while scheduler and public normalization entries additionally
normalize the verified graph.  A scoped `ContextVar` receipt lets
synchronous compiler-owned descendants reuse exactly the already-verified
root without weakening any caller-owned entry.  The receipt carries no
authority outside the dynamic compiler call, and direct analyses,
normalization, scheduling, request identity, and lowering remain strict.

This also corrects §29's latency attribution.  The first receipt
implementation removed repeated scans from the LoopIR path but missed
the release `einsum` dual builder: both register-block arms and the
fallback independently normalized the same freshly compiler-built root,
so probes still counted three structural scans per call.  Exact
`d14f87e` latency consequently crossed the 1.10 policy in two independent
orders (worst **1.133 p50 / 1.130 p95**), with the normalization stage
explaining 53-76% of the observed deltas.  `8612679` proves that root once
and scopes its `ContextVar` receipt only across the synchronous internal
scheduling block; all three historical normalization records remain, and
a failed shared proof retires the `CompilationContext`.  The successful
shared scan is included in end-to-end latency but deliberately excluded
from the per-arm normalization durations because its provisional stage
token is cancelled to preserve the published record sequence.  Error
ordering, shared DAG handling, cycle detection, maximum depth, and
receipt cleanup are locked independently.  The initial inline form also
raised `_einsum_owned`'s inherited C901 score from 70 to 72; `f13ba79`
extracts only the receipt/stage lifecycle, restoring 70 without changing
the trust scope or published stage sequence.

### 30.2 Dense layouts are complete across the whole vertical slice

`c69c839` admitted permuted all-dense inputs but did not propagate the
logical-to-physical contract through every owner.  Fresh differentials
found incorrect or rejected behavior for non-identity result layouts,
workspace addressing, relayout staging, scheduled result copies,
runtime result wrapping, and several rank-three and zero-extent
compositions.

The correction makes physical level order the sole storage-order fact
and maps logical coordinates through it at every input, workspace,
relayout, result, oracle, target-lowering, and ABI boundary.  Runtime
wrapping snapshots tensor storage, shapes, dtypes, and mode order before
caller-visible mutation, validates exact input/result cardinality and
rank, checks dimensions at signed-64-bit boundaries, and retires the
compilation stage on every failure.  Permuted compressed levels,
non-permutation orders, and unsupported permuted sparse results remain
fail-closed; the schema remains level-based and no CSR-shaped shortcut
was introduced.

The regression matrix covers f32/f64, unary and multi-input expressions,
rank two and three, direct and scheduled lowering, stack and heap
workspaces, relayout, public result wrapping, and every zero-extent
class.  The generated C++ remains byte-identical on all retained legacy
capture surfaces.

### 30.3 Legacy automatic workspace schedules use a narrow receipt

The first correction exposed a release-reachable compatibility fact:
legacy `Scheduler.auto_schedule` creates one producer/consumer workspace
access pair whose compatibility identities intentionally alias across
the two branches.  Strict normalized CIN must reject that graph, but the
compiler-owned legacy lowerer must still consume it.  A broad
identity-relaxation would have hidden real mutations, while the former
generic `deepcopy` could recurse through arbitrary unowned instance state
and leak `RecursionError`.

The final boundary recognizes only the exact legacy alias topology:

- one workspace access on the producer left-hand side and the paired
  workspace access on the consumer right-hand side;
- every reused identity at matching branch suffixes, with paired binder
  identities and tensor-assignment operation, plus exact workspace
  metadata; non-aliased child fields remain independently validated CIN;
- no additional duplicate symbol, access, node, or index identities; and
- only a compiler-owned synchronous ownership transfer after the
  normalized source was already verified.

That post-transformation graph cannot authenticate the semantic
*provenance* of the intentionally unmatched producer RHS and consumer
LHS: workspace insertion replaced and discarded their branch twins.  A
caller can replace either leaf with fresh, independently valid CIN; the
canonical bytes and generated program then change with it.  This is not
an adapter-safety or cache-identity escape, and production `einsum`
retains a synchronous scheduler-to-lowerer handoff.  Proving that those
leaves came from a particular pre-workspace source requires a retained
typed plan/source receipt, which B1 must own rather than infer from raw
legacy syntax.

Caller-owned debug lowering with verification enabled refuses such a
derived graph with `unverifiable_legacy_schedule_aliases`; it never
silently skips semantic verification.  The public `lower_IndexStmt`
surface no longer exposes an ownership-transfer escape hatch.  The
private `_lower_owned_IndexStmt` documents the synchronous handoff and
does not return the transferred tree.

The adapter now copies only validated forward and
schedule-authoritative fields, then rebuilds reverse links.  Arbitrary
extra or discarded attributes are neither copied nor traversed.  The
finite known scheduler backlink surface is instead validated
relationally before reverse links are rebuilt; that validation covers
logical parent expressions, outer/inner tile flags, `TileSizeVar`
endpoints and base, parent membership, `no_tile_list`, tensor-access
backlinks, and missing optional fields.
Malformed state fails with structured `VerificationError` rather than
`KeyError`, `AssertionError`, `ValueError`, or `RecursionError`.  Exact
untiled, workspace, regblock, and multiple-tile legacy schedules retain
their historical emission.  Rank-zero scalar `TensorVar` values retain
the established no-format contract; higher-rank tensors remain strict.

The first exact compiled battery caught a real regression in this
boundary:
`test_source_comparison_ignores_nonsemantic_legacy_metadata` failed
because the shared preflight rejected a hostile `no_tile_list` before
normalization could discard it.  The fix distinguishes semantic forward
state from mutable legacy schedule state.  Normalization ignores and
resets workspace markers, `no_tile_list`, tile roles/backlinks, and
reverse access lists; the raw legacy adapter validates those same fields
because its forward copier consumes them.  Twelve focused adversaries
cover both directions.  The failed run is retained rather than
overwritten (one failure / 413 passes).

The first clean-detached full-suite partition then caught three further
problems rather than allowing a focused-only result to stand:

- canonical dumping rejected the exact legacy alias receipt that the raw
  compiler-owned adapter admits.  The compatibility fallback now applies
  only when the strict defects are duplicate node/index identities and
  the complete exact legacy receipt validates;
- one intermediate correction accidentally made an access-rank mismatch
  debug-policy-dependent.  Access arity is structural and is again
  rejected at every entry; the debug-policy fixture now uses a genuinely
  semantic dangling-index defect; and
- an identity-layout relayout fixture expected the old blanket
  `invalid_schedule_relayout` result, although the generalized dense
  layout boundary correctly admits either logical access order.  With no
  matching staged read, its stable failure is `relayout_target_missing`.

The failing partition is retained.  Every final receipt below uses exact
`f13ba79`, after all corrections.  A final independent adversarial pass
also found that the detached logical parent admitted for successive
legacy tiles treated missing `_parent` and `tile_size_var` storage as
their valid `None` value.  The forward copier indexes both fields, so
that gap leaked raw `KeyError`.  `42d484f` requires exact presence plus
`None`; the two-case lock at `b946b58` retains valid detached lowering
and structured rejection for either deletion.

### 30.4 Exit disposition

The corrections do not widen the Phase-6 claim.  Phase 6 remains
**NO-GO on exactly one production-relevant cluster**: sparse
result/workspace representation.  §29.5's sequencing remains sound:
B1 is serial `coo_workspace` plus general multi-level ordered sparse
assembly against the executing `ss@ss->ss` comparand; B2 is the mixed
compressed-parent/dense-leaf correctness slice gated on the LoopIR
oracle, PyTorch, and the public route because its legacy comparand is
malformed (values without compressed-parent coordinates).  The distinct
generic sparse-output reduction is the memory-unsafe legacy boundary.
The two-pass OpenMP count/fill form stays assigned to Phase 7.  No
Phase-7 work, cutover, cache/selector change, or legacy deletion was
started.

### 30.5 Exact-revision verification

Evidence is retained under
`/Users/bobby/.cache/scorch-codex/phase6-layout-review-f13ba79/`.
Every command below imported Scorch from the clean detached
`f13ba79` worktree unless explicitly identified as a base control.
The bounded 130-file receipt manifest is
`FINAL_RECEIPTS_SHA256SUMS`, SHA-256
`240f03c6c0b23e38196245096f9465bf5399d97bc216c0806695dd9a1fed9470`;
the earlier root `SHA256SUMS` remains the narrower
audit/capture/census manifest.

- focused review membership after the final corrections: **682 passed**
  across CIN analysis, CIN lowering, schedule API, LoopPlan, compiler
  stage timing, and the formerly failing compiled source-comparison
  case;
- 86-case automatic audit, twice: **46 admitted / 40 rejected / zero
  divergent**; the JSON results are byte-identical (SHA-256
  `f2d004b1f320489d43545346ed2eaf6a994db27f2165f31a17eb1405e390f7ac`);
- fresh source captures: corpus **20/20**, grid **42/42**, anchors
  **22/22**, and heap **11/11** are byte-identical to §29 and chain to
  the sealed Phase-6 baselines.  All **22** auto source/CIN artifacts
  are exact; its twenty-third report differs only in the same two
  previously demonstrated process-dependent cache-key suffix
  characters and is byte-identical after normalizing exactly those two
  evidence-only fields.  Corpus/grid/anchor/heap manifest SHA-256 values
  are `7e4a9c436e5ed1005874e9ece56847ea4ef88dd6f5a04c4d657c3ca5b37cd6c4`,
  `65e68ba19510ab240cf85574aa6be272dd6e5b0fdd7e799f7bcfe9db6d06c094`,
  `44b39b55fbaea4bccbfac2aca6c217ca6b7ccd7135c8d50950efd96fd7f2dd13`,
  and `aa8be2229bea130833461d6dcf789722837f1a100f2da2a74de9cc4b2dcb4f72`;
- full-source static parity: Black reports the one inherited
  `prebuilt_kernels.py` file, Flake8 reports the same nine inherited
  findings, and mypy reports the same **140 errors in 11 files**; the
  normalized Black, Flake8, and mypy logs are byte-identical to the
  inherited baseline (SHA-256
  `0cba9cee6ea6e561b398ea9e56f9ba0ddeb040c6d836e6566adfd0b742c5c777`,
  `8aa1212e0d42a9b8b90e2e0798fd3f2ce5085b389dd75205e0d942d98cfcc6b0`,
  and
  `bf34740270200e2521f2d5f287d788714a8bfd2733d932cfc4c735f1dc6d6681`);
- non-performance collection selects **4,518 of 4,521** tests, with
  exactly three performance tests deselected;
- complete compiled scheduled-slice, pipeline-execution,
  schedule-generality, and dual-path battery: **414 passed / zero
  failures in 1,150.01 s** (log SHA-256
  `3288c79598d02fb9121006f0e92f6a4b6d4a409b3757b60fcda5ab3907a054fb`,
  JUnit SHA-256
  `dafd617a42e97f606c18b5e9dfbbfbd9eb17ec2d664441d54c95d34c0c4795fe`);
- production-derived layout/dual census, two seeds with 60 cases each:
  **120 attempted**, **96 qualified / 24 skipped**, **192 arm
  evaluations**, **82 admitted and byte-identical**, **108 expected
  structured rejections**, and **zero mismatches**.  The only two raw
  exceptions are both arms of the already documented, public-unreachable
  broadcast-only result-axis case (`No lattice points generated`);
  the combined census log remains byte-identical to `d14f87e`, SHA-256
  `f9f81e072e8c361770462bd1df0de42da110dcfa5f3c15bb7ecb0705c9293b0a`;
- paired same-session compiler latency, clean detached `a1d4851` base
  versus `f13ba79` candidate, 5 warmups / 30 samples: the first and third
  base-to-candidate pairs pass every endpoint (worst **1.057 p95**), while
  the reverse-order run has one isolated `small_dense` p95 tail at
  **1.290** with p50 1.063.  Comparing the two byte-identical candidate
  runs reproduces that tail at **1.260 p95**; a fresh back-to-back
  candidate A/A is fully green and the third paired run returns
  `small_dense` to **1.035/1.047**.  The crossing is therefore retained
  and attributed to same-revision tail variance rather than averaged
  away.  All build tuples/source hashes are identical; attribution JSON
  SHA-256
  `a3cce293e0c1f690e03b1df050885bdda59693b11083b9d6dd9512e04b364007`;
- partitioned complete non-performance suite, run sequentially in fresh
  processes and isolated caches to avoid the documented macOS libomp
  pthread-key ceiling: partition A passed **3,765** with **14 skipped**
  and the one inherited PyTorch sparse-invariant warning in
  **1,762.966 s**, and partition B passed **739** with the three
  performance tests deselected in **1,233.772 s** — **4,504 passed, 14
  skipped, zero failures**.  The JUnit union contains exactly all
  **4,518** directly collected non-performance cases with zero missing,
  extra, duplicate, or overlapping identities; the selected list,
  pre-run partition union, and post-run JUnit union share SHA-256
  `daaa4a72c8d4e8d100caab2598370253650504b9ea04dd8a7799eefd3bf6883d`.
  The union summary JSON has SHA-256
  `b65dd4f4700cfa79621c67bf08af81597324654696cc8e368d7c811b55330f0a`.
  Partition log SHA-256 values are
  `49abfce7321710e3fe694515c9d4e50fd7c6acec0e8bf187e81a113f5702d9be`
  and
  `cd21a86fb1e77f8f552ea4c6e5863efac752d7839d578f8efae49b1704bc5198`;
- `git diff --check` is clean; the detached worktree remained clean;
  local and live origin remained `58e8565`; the five protected files
  retained their recorded hashes; all unrelated dirty/untracked
  GPU/CUDA, benchmark, packaging, scheduler, research, scratchpad, and
  tooling material remained untouched; nothing was pushed, amended,
  squashed, or reordered.

## 31. Independent review of the §30 corrections and the shared-object boundary (2026-07-28)

This section records the independent adversarial review of the fourteen
§30 correction commits (`ce39b36..f13ba79` plus the `3d2f42d`
documentation commit) demanded by the §30 routing prompt, and the four
correction commits it produced.  The handoff was not trusted: every
critical gate was reproduced from the retained scripts before any code
was read, and the §30 boundaries were then challenged with fresh
adversaries beyond the committed matrix.

### 31.1 Gate reproduction at the inherited tip

All reproductions ran at `3d2f42d` in the working tree with the `scorch`
conda environment; evidence is retained under
`~/.cache/scorch-codex/phase6-b1-review-93530ce/`.

- focused review membership (six files: CIN analysis, CIN lowering,
  schedule API, LoopPlan, plan identity, stage timing): **720 passed**
  (the recorded 682 used a slightly narrower set definition; no
  failures);
- 86-case automatic audit via the retained
  `audit/run_schedule_audit.py`: **46 admitted / 40 rejected / zero
  divergent**, JSON equal to the retained `audit-1.json` after
  normalizing only the embedded commit field;
- source captures via the retained scripts: corpus **20/20**, grid
  **42/42**, anchors **22/22**, heap **11/11** byte-identical; auto
  **22/23** byte-identical with `report.json` differing in exactly the
  two documented process-dependent cache-key suffix characters;
- randomized cross-route census via the retained `census_sweep.py`:
  seeds 1 and 2 byte-identical to the retained logs; fresh seeds 7 and
  11 (60 cases each) report **zero parity mismatches**, and each fresh
  `AssertionError` was individually confirmed to be the documented
  public-unreachable broadcast-only result-axis legacy limitation
  (`No lattice points generated`);
- complete compiled scheduled-slice, pipeline-execution,
  schedule-generality, and dual-path battery: **414 passed in
  1,208.30 s**, matching the recorded gate.

### 31.2 Fresh adversarial findings

Twenty fresh probes (`probes/fresh_adversaries.py`, post-fix log SHA-256
`98ef055575461f79d1deef44ab1fb2640c5c780522e9747381d56b855d67df17`) and
three independent full-diff reviews of the correction commits produced
five confirmed defect classes that §30 had not closed:

1. **Shared-object raw escapes.**  A same-object `BinaryOp` diamond
   passed every raw entry (the preflight deliberately admitted shared
   completed objects, deferring `duplicate_node_reference` to the
   debug-only ownership analysis) and leaked a raw `ValueError` from
   kernel-ABI argument assembly; a shared `TensorAssign` leaked a raw
   `IndexError`; a shared LHS/RHS access lowered silently; and a
   2^60-path shared-DAG chain reached the unmemoized recursive
   verifier/dump/lowering walks even though the preflight itself is
   memoized and linear.
2. **Unbound split-role tile aliases.**  A forged split-role `IndexVar`
   carrying a real `TileSizeVar` but a foreign detached `_parent` passed
   admission (nothing tied a split-role index to its own TileSizeVar
   endpoint), and the forward copier then leaked raw `KeyError` on
   missing stored fields or `RecursionError` through an unwalked parent
   chain.
3. **Divergent same-identity index twins.**  The legacy index-alias
   admission compared only the `(NodeId, IndexId, name)` triple through
   an access-path disjunct broader than the documented matching-suffix
   contract, so a same-identity twin with divergent tile state could be
   admitted and merged onto one canonical object whose merged state no
   validator approved.
4. **Raw lowering entry modes.**  Public
   `lower_IndexStmt(..., recurse=True)` at an outermost call bypassed
   the entire raw-entry boundary (raw `AttributeError`), a reused
   lowerer instance rode the stale `outermost_stmt` bypass, and
   `lower_IndexExpr`/`lower_CIN` accepted bare expression roots with no
   structural preflight at all (raw `RecursionError` on a cyclic
   expression).
5. **Unbounded iteration-domain walk.**  `analyze_iteration_domains`
   looped without bound on a self-referential `ForAll` before leaf
   validation could reject it.

Two further latent items were fixed alongside: the two index-var
level-metadata sites in the legacy lowerer used the stored mode order as
a logical-to-physical map (`mode_order[i]`) instead of the
storage-position lookup (`mode_order.index(i)`) — invisible at rank two
and for all-dense rank-three families, which is why every capture stayed
byte-identical — and `Schedule` admitted `TileSpec`/`RelayoutSpec`
subclasses whose caller-defined code would execute inside
compiler-trusted scopes such as the scoped receipt window.

### 31.3 The corrections

Five commits close the complete set:

- `8f43cea` — completed-object revisits now diagnose
  `duplicate_node_reference` for every node kind except the
  intrinsically shared symbol leaves (`IndexVar`, `TensorVar`,
  `Workspace`); the one admissible shared occurrence node — the legacy
  workspace producer-LHS/consumer-RHS access pair — is classified by a
  receipt post-pass over recorded occurrence paths; the canonical-dump
  compatibility fallback admits the new code so exact legacy receipts
  still serialize; split-role indices must be the exact outer/inner
  endpoint of their stored `TileSizeVar`; aliased index twins must carry
  equivalent schedule state.  Strict rejection sets for
  scheduler-owned workspace alias graphs now additionally report
  `duplicate_node_reference`; the three existing exact-set assertions
  were aligned.
- `9afe69b` — an active-lowering counter recognizes internal recursive
  re-entry; outermost `recurse=True` fails with
  `invalid_recursive_entry`; outermost expression roots fail with
  `invalid_expression_entry`; a reused lowerer takes the full validated
  boundary; both `mode_order` sites use the storage-position lookup,
  with a non-involutive rank-three regression test.
- `12a9267` — the iteration-domain nest walk fails a cyclic `ForAll`
  closed with `unsupported_statement`.
- `93530ce` — `Schedule.tiles`/`Schedule.relayout` admission is
  exact-type only.
- `ef70023` — the schedule-state equivalence requirement was initially
  stricter than the graphs legacy workspace insertion actually produces:
  under regblock the consumer branch carries the tiled logical index
  while the paired branch clone is plain, and the compile-options
  snapshot memberships caught the release-reachable rejection (the
  compiled battery did not, because production einsum hands the
  scheduler result over synchronously without re-entering the raw
  boundary).  Equivalence now admits exactly that plain/tiled-logical
  pairing while continuing to reject every divergent pairing involving a
  tile component; a companion test locks the legitimate regblock
  workspace lowering.  The forged plain twin of a tile component is
  structurally refused at the adapter display-name boundary, with the
  equivalence check standing behind it as defense in depth.

Emission neutrality was proven before committing: all corpus, grid,
anchor, and heap captures byte-identical (re-proven after `ef70023`),
the 86-case audit unchanged at 46/40/0, and the full-source mypy
baseline exactly 140 errors in 11 files.  A compiled-battery run
launched at the pre-relaxation revision reported 413/414 with exactly
the one twin-regression failure `ef70023` fixes
(`test_spmv_and_dense_matmul_empty_schedule_preserve_default_codegen`),
and that test is green at `ef70023`; the complete battery is re-proven
at this session's final tip as part of the B1/B2 gates.

### 31.4 Verified-sound findings and recorded limitations

The reviews confirmed sound: the scoped receipt lifecycle (set, consume,
cleanup on success and on every injected failure path, CompilationContext
retirement on shared-proof failure), the preservation of the three
historical normalization records with the cancelled provisional token,
the behavior-neutrality of the `f13ba79` extraction, the 42d484f
presence checks for the shapes they admit, the canonical-dump fallback's
no-masking property, the always-on access-rank reconciliation, and the
complete dense-layout propagation table.

Recorded honestly, not fixed (no demonstrated failure):

- the trusted-root receipt stores bare `id()` values with no strong
  reference or epoch; every current window holds the root in a live
  local, so recycling is not reachable today (hardening suggestion:
  a companion strong-reference scope);
- the fallback kernel-cache key renders `str(post_ops)` inside the
  receipt window; it runs strictly after the last trusted consumer and
  the dual path requires `post_ops is None`;
- `legacy_cin_working_copy`'s KeyError-safety is positional (it relies
  on `_verify_legacy_cin_lowering_structure` having run at its one
  production call site);
- `_execute_legacy_scheduled` in the pipeline performs runtime binding
  outside stage tokens, mitigated by its private discarded context;
- a 256-deep admitted graph plus a deep caller stack can still surface
  Python `RecursionError` from the recursive clone/serialize walks
  (the preflight bounds depth, not the caller's stack budget);
- the C3 provenance boundary reproduces exactly as §30.3 documents it:
  fresh independently valid unmatched leaves are admitted by the raw
  adapter and change the emitted program, so B1 must carry a structural
  source receipt in the typed plan.

No serialized contract changed: `scorch.loopir.canonical.v8`,
`scorch.loopir.request.v2`, `scorch.loopplan.canonical.v1`, and
`scorch.autopolicy.v1` remain current.  Phase 6 remains **NO-GO on
exactly the sparse result/workspace cluster** (B1/B2), unchanged by this
review.

## 32. Rigorous review of the partial B1 sparse-workspace milestone (2026-07-28)

### 32.1 Scope and verdict

The inherited B1 checkpoint (`28823db` through `f3f4545`) was reviewed as
an intentionally incomplete vertical slice, not accepted from its session
report.  Its semantic core is sound: the level-based program descends
parent positions through a merged INTERSECTION, accumulates serially by
coordinate, drains in coordinate order, preserves explicit-zero
cancellation, and derives multi-level output storage from the ordered
coordinate stream.  The representation is not CSR-specific.

The checkpoint was nevertheless not ready to hand to target work.  It added
roughly 1,250 production lines with only three small compatibility-test
edits, had no direct B1 oracle/erasure/canonical/adversarial coverage, and
contained concrete correctness and fail-closed defects.  Four local commits
now close the review defects:

- `deb71d4` — production boundary corrections;
- `c12b625` — the first direct semantic/adversarial test lock;
- `121e4a9` — deterministic replay and use-boundary corrections;
- `01de1b2` — replay, hostile-state, and diagnostic regressions.

The target path remains deliberately absent.  Both public automatic
`ss@ss->ss` policy arms now reach verified sparse-workspace schedule
application and then stop at the existing
`unsupported_program_shape` hierarchical-compressed target boundary.  Phase
6 therefore remains **NO-GO** on the same B1/B2 cluster; this review makes no
target-parity or Phase-6 exit claim.

### 32.2 Confirmed defects and corrections

The review found and fixed these material issues:

1. `LevelOutputBuilder.finish()` divided by zero for a zero-extent trailing
   dense level and eagerly materialized the entire dense-suffix Cartesian
   product even for an empty output.  Dense suffixes are now checked lazily,
   zero suffixes are canonical empty outputs, unsupported level kinds and
   excessive ranks fail closed, and `LevelTensor` validates its complete
   DENSE/COMPRESSED storage contract.
2. Sparse-workspace consumers could place drains under dynamically repeated
   control flow, nest a second drain of the same workspace, or never consume
   `SparseWorkspaceValue`.  The verifier now requires one direct ordered
   drain and at least one in-scope consumption while preserving room for
   dense-suffix work inside its body.
3. A noncanonical all-`None` merge-position tuple serialized differently
   from the equivalent empty tuple.  It is now rejected; continuation
   builders also scan identity values stored in tuple fields.
4. Loop reorder and scheduled-carrier admission omitted
   `SparseWorkspaceRegion`, while sparse dependency analysis omitted
   `MergedSparseFor.positions`.  Existing sparse regions can no longer be
   reordered or passed off as an unscheduled base, and child cursor
   dependencies retain their position-binder dominance.
5. `apply_sparse_workspace` accepted malformed workspace facts, broader
   output/loop roles than its B1 implementation, and an extra outer
   reduction.  The latter reset and drained the region repeatedly, producing
   duplicate/out-of-order appends.  Admission is now the exact
   identity-ordered rank-2 C/C result with one final logical INTERSECTION
   reduction and one trailing single-cursor result axis.
6. The generated drain binder came from the process-global `IndexId`
   allocator.  It could collide with an imported artifact and made the
   public `verify_scheduled_loopir` replay differ structurally from the
   artifact it was verifying.  Resumed builders now continue from every
   stored artifact `IndexId`; both automatic policy arms replay exactly.
7. Frozen `Schedule`, `TileSpec`, and `RelayoutSpec` instances were checked
   only at construction.  Post-construction forged fields could execute
   caller hooks in scheduling or cache-key construction.  Compiler and
   cache-key boundaries now revalidate exact stored fields, descriptor
   agreement, nested exact carriers, and canonical owned state before
   comparison or rendering.
8. A reused `CINLowerer` silently treated a second valid program as an
   internal subtree; standalone iteration-domain analysis could recurse or
   leak raw exceptions on hostile CIN.  Lowerers are explicitly single-use
   after an outermost lowering begins, and the public analysis shares the
   bounded structural preflight.
9. Huge exact integer extents could leak CPython's decimal digit-limit
   `ValueError` while formatting storage/oracle diagnostics.  Bounded
   diagnostic rendering now preserves domain-specific errors without
   coercing or accepting the value.
10. The B1 files introduced nine mypy findings.  The typed fixes restore
    zero findings under `src/scorch/compiler/loopir/` and return full-source
    mypy from the inherited B1 checkpoint's 149 findings in 13 files to the
    established 140 findings in 11 files.

Fresh malformed-node sweeps over every field of a valid sparse-workspace
graph produced only controlled verifier diagnostics.  The schedule tests
also lock missing fields, descriptor/stored-state divergence, hostile scalar
subclasses, unowned iterable containers, producer/output role separation,
canonical name exclusion, erasure, empty inputs, cancellation, and
independent carrier replay.

### 32.3 Verification

All commands used the `scorch` conda environment.  The broad pure membership
(levels, oracle, printer, verifier, schedule passes, CIN lowering and
analysis, neutrality, LoopPlan and identity, legacy lowering, schedule API,
stage timing, and raw-string budget) passed **1,313 tests**.  The focused
review selection passed **820 tests**.  A clean detached worktree at
`01de1b2`, with `scorch.__file__` asserted inside that worktree, passed the
complete scheduled-slice, pipeline-execution, schedule-generality, and
dual-path compiled battery: **416 passed in 1,123.59 seconds**.

Black and Flake8 have the same inherited file/finding sets as `f3f4545`
(paths and affected line numbers aside); focused LoopIR mypy is clean.
Full-source mypy improves from **149 errors in 13 files** at the partial B1
checkpoint to the established inherited baseline of **140 errors in 11
files**.  `git diff --check` is clean.  The clean-detached full
non-performance run reached **4,549 passed / 14 skipped / 3 performance
deselected** before nine late JIT subprocess creations hit the documented
macOS libomp pthread-key ceiling (`OMP Error #179`, every failure the same
`SIGABRT` at `pthread_key_create`).  A fresh process reran the complete
failed schedule-generality selection plus adjacent parameter cells:
**11 passed**; a second fresh process reran the one failed value-object case:
**1 passed**.  The proven complete union is therefore **4,558 passed / 14
skipped / 3 performance deselected / zero code failures**.  The independent
compiled battery had already passed all eight affected
schedule-generality cases before the exhausted full-suite process.

The five protected tracked files retained their required SHA-256 values
through every commit, only explicit paths were staged, all unrelated
GPU/benchmark/scheduler/research/scratchpad material remains untouched,
nothing was pushed, and local/live origin was not moved by this review.

### 32.4 Next boundary

The next coherent milestone is still B1 target completion, now on a much
firmer contract.  Implement one narrowly recognized LLIR lowering for the
verified outer sparse row → sparse-workspace region → merged reduction →
child sparse axis → insert/drain/append shape.  Emit the retained serial
`coo_workspace_1d` allocation, merged insertion, sorted drain, and
two-compressed-level assembly byte-for-byte in both automatic arms.  Do not
weaken the general target's existing hierarchical-compressed restrictions.
Then add direct compiled sparse-output execution and PyTorch/oracle
differentials (the dense-only shadow helper is not an honest sparse-output
oracle), completion-loss adversaries, stage timing, and source parity.

If those gates are green, continue in the same session into B2 as the
independent-oracle correctness slice described in §29.5/§31: the legacy
mixed-level comparand produces malformed sparse storage and must not be
used as a parity oracle.  The separately retained sparse-output reduction
is the memory-unsafe boundary.  Regenerate the production-derived census
and repeat the full Phase-6 exit audit only after both slices.  Phase 7
may begin only on a genuine GO.

## 33. B1/B2 closure, regenerated census, and the Phase-6 exit (2026-07-28)

### 33.1 Scope

This session independently re-reviewed the five §32 commits, completed
the B1 sparse-workspace target end to end, implemented the B2 mixed
compressed-parent/dense-leaf correctness slice, regenerated the
production-derived census, and performed the criterion-by-criterion
Phase-6 exit audit.  Two local commits carry the code and tests:

- `12c2079` — the B1 serial sparse-workspace LLIR target and its
  differential battery;
- `ebb243b` — the B2 mixed dense-leaf assembly family and its battery.

### 33.2 Independent review of the §32 commits

`deb71d4`, `c12b625`, `121e4a9`, `01de1b2`, and `ed4b51b` were re-read
diff by diff and probed beyond their own tests: replay under deliberate
process-global identity-allocator interference, erasure round trip plus
re-application canonical identity, a forged `WorkspaceInsertion.dense`
flag carrying hostile `__bool__`, artifact-continuation collision
scanning against every stored `IndexId`, hostile bool-typed level
scalars, incomplete huge dense suffixes, zero-extent compressed levels,
and a ragged oracle differential against an independent dense reference.
All nine probes passed; `_apply_schedule_lowering` was confirmed not to
re-enter the new single-use lowerer boundary, and the one changed
`new_index_id` call site was verified.  No defect was found; the review
stands as recorded.

### 33.3 B1: the serial sparse-workspace target is complete

`lower_llir.py` now routes the exact verified chain — outer level-0
sparse row loop over one `SparseWorkspaceRegion` — to a dedicated
`_SparseWorkspaceLowering` (structural routing only; the general
hierarchical-compressed boundary is untouched and every other placement
of a region still fails closed).  The class admits exactly the shape
`apply_sparse_workspace` produces: two identity-ordered doubly
compressed inputs, the two-cursor INTERSECTION merge with bound-position
descent into the child sparse loop, the ADD insertion at the child
coordinate, and the one ordered drain appending the drained value at the
row and drain coordinates.  Raw emission mirrors the retained serial
`coo_workspace_1d` legacy lowering statement for statement; the shared
managed passes produce the `emplace_back`/`scorch_vector_set` spellings.
The one shared-driver change — the `Init result tensor level sizes`
comment is emitted only when dense result levels exist — is provably
inert for every previously migrated family (all carry at least one dense
result level) and required for the all-compressed result.

Byte parity holds in both automatic policy arms for float32 and float64:
`compare_generated_sources` is identical, the arms are identical to each
other, and the pipeline-generated legacy source matches the retained
5,301-byte comparand modulo tensor/coordinate display names and bound
shapes.  The public pipeline (`execute_cin_via_loopir`) compiles,
executes, and wraps the result as honest `ss` storage with identity mode
order derived from the verified declaration.

### 33.4 B2: the mixed dense-leaf assembly family

The B2 slice admits compressed-parent/dense-suffix RESULTS (`sd`,
`sdd`, ...) for the dense-domain elementwise families.  The format gate
keeps rejecting mixed dense-leaf OPERANDS (`unsupported_format`; the
physical position-load chain stays undeclared), and the new
classification branch requires every result coordinate to iterate a
dense domain, reusing the existing level-based `AppendEntry`
construction — no new node kinds, no schema bump
(`scorch.loopir.canonical.v9` unchanged).  A second dedicated,
structurally routed target lowering emits the assembly directly: the
parent coordinate is appended exactly when the dense suffix has nonzero
runtime extent (the canonical zero-extent contract from §32), values
append in complete leaf blocks through explicit `emplace_back` call
statements, and the root position closes after the nest.

The retained legacy comparand for this family is defective and stays
failure evidence only: its generated kernel appends every dense-leaf
value but never assembles the compressed parent's coordinates, so the
returned storage would carry values with no owning rows.  The battery
locks that shape (`C_values.emplace_back` present, no `C0_crd` append)
and the intentional absence of any byte or execution parity gate.  The
family is proven against the production LoopIR oracle (exact positions,
coordinates, values) and the PyTorch dense reference for rank-2 and
rank-3, float32/float64, binary elementwise, zero-extent and canonically
empty cells, in both automatic arms and on the unscheduled route, with
erasure to the base program, route-stable sources, canonical identity,
replay, hand-built target adversaries, and locked seam codes
(`unsupported_sparse_output_reduction` for reducing into a mixed leaf,
`unsupported_format` for mixed operands,
`unsupported_sparse_output_domain` for sparse domains).

### 33.5 Regenerated census and neutrality

The widened deterministic family census (13 cells over the sparse-result
cluster and its neighbors, both automatic arms) records zero route and
zero arm divergence: B1 at byte parity, the three B2 cells admitted with
the documented no-parity disposition, and every seam at its precise
stable code (`ds@ds->ds` SpGEMM and `ss@ss->ds` row-scope at
`unsupported_sparse_output_reduction`, merged reduction at
`unsupported_merged_reduction`, root sparse output at
`unsupported_sparse_output`, union-CSR and dense matmul at byte parity).
The four retained randomized cross-route census seeds (1, 2, 7, 11; 60
cases each) reproduce their sealed tallies exactly.  Fresh corpus
(20/20), grid (42/42), anchor, and heap captures are byte-identical to
the sealed `f13ba79` baselines; the auto capture is identical after
normalizing exactly the two previously demonstrated process-dependent
cache-key suffix characters.  Paired compiler latency (base `ed4b51b`
versus tip, both orders, sources SHA-identical in all four runs) is
neutral: per-case p50/p95 ratios 0.91–1.05, inside the historical noise
band.

### 33.6 Verification

All commands used the `scorch` conda environment.  At the tip: the
focused fast membership (schedule passes, verifier, levels, oracle,
iterdomain, schedule API, printer, CIN and LLIR lowering, neutrality)
passed **828**; the dedicated sparse-result battery passed **37**
(22 B1 + 15 B2, including compiled execution); the sparse-workspace
pipeline selection passed **24**.  The complete compiled battery
(scheduled slice, pipeline execution, schedule generality, the new
battery, and the regblock dual path) passed **438 in 1,262.73 s** at the
B1 state, with the B2-touched fast suites re-proven at the tip.  Focused
LoopIR mypy is zero findings; full-source mypy is exactly the inherited
**140 errors in 11 files**; Black and Flake8 are clean on every changed
file; `git diff --check` is clean.

The clean detached-worktree full non-performance suite at `ebb243b`
(with `scorch.__file__` asserted inside the worktree) reached **4,560
passed / 14 skipped / 3 performance deselected in 2,777.79 s** before 35
late JIT builds hit the documented macOS libomp pthread-key ceiling (31
literal `OMP: Error #179` / `pthread_key_create` markers retained in the
sealed log; a first run of the same suite had failed 33 nodes at the
same signature with 4,562 passed).  One fresh process reran the complete
35-node failed set: **35 passed in 323.30 s**; the five nodes
recoverable from the first run had already passed a separate fresh
process (5/5).  The proven complete non-overlapping union is therefore
**4,595 passed / 14 skipped / 3 performance deselected / zero code
failures** — the late failures are infrastructure, not code, and are
recorded as such.

The five protected tracked files retained their exact SHA-256 values
through every commit, only explicit paths were staged, all unrelated
GPU/benchmark/scheduler/research/scratchpad material remains untouched,
nothing was pushed, and origin remains `58e8565`.  Evidence is sealed
under `/Users/bobby/.cache/scorch-codex/phase6-b1b2-ebb243b/`
(captures, census, latency, parity, and Phase-7 comparands).

### 33.7 Phase-6 exit audit: GO

Criterion by criterion at `ebb243b`: every migrated family has a typed,
verified schedule decision (stack, panel, relayout, heap, parallel
selection, reduce-out, general dense layouts, the serial sparse
workspace, and the tile-free mixed dense-leaf family); the
explicit/automatic differentials hold wherever legacy is a valid
comparand, and the one family whose comparand is invalid carries an
explicit, evidence-locked no-parity disposition; canonical request
identity remains semantic with no version change; the representative
compositions remain byte-ready (all captures identical); and release
behavior is unchanged everywhere (production imports untouched, the
census and captures byte-stable, paired latency neutral).

The sparse-result/workspace cluster that §29–§32 held open is closed:
B1 with byte parity and honest compiled sparse output, B2 as the
independent-oracle correctness slice.  The remaining boundaries are all
explicitly dispositioned outside Phase 6 with stable codes: the two-pass
OpenMP count/fill form (`ds@ds->ds` SpGEMM, `ss@ss->ds` row-scope, and
the dense-axis reduction into a mixed leaf, all
`unsupported_sparse_output_reduction`) is Phase-7 target-owned
parallel/runtime composition; mixed dense-leaf operands
(`unsupported_format`) are the adjacent load-chain gap; COO operands,
permuted compressed structure, and the trailing-compressed automatic
families keep their locked codes.

**Phase 6 is GO.**  No cutover, legacy deletion, or release
dispatch/cache ownership change was performed or started.

### 33.8 The permitted Phase-7 stretch and the next milestone

As the one small Phase-7 runtime-composition stretch, the two-pass
comparand baseline was captured and characterized (no code change):
both arms of `ds@ds->ds` SpGEMM (5,618 bytes, byte-identical arms) and
`ss@ss->ds` row-scope (6,071 bytes) are sealed under the evidence root.
The kernels are statically parallel two-phase forms: a per-thread
`linked_list_workspace_1d<float>` pool sized by `scorch_nthreads` over
an nnz estimate, per-thread `make_view()` workspaces inside one
`#pragma omp parallel` region, and a dynamic-chunk row loop — exactly
the target-owned parallel policy plus runtime-stitch structure Phase 7
must own as a typed twin.  The next coherent milestone is that Phase-7
slice: a typed parallel sparse-workspace-pool surface with the working
public route as its execution oracle, gated by the same
verifier/oracle/erasure/adversarial discipline, still without touching
release dispatch or cache ownership.

## 34. Rigorous post-GO review of B1/B2 (2026-07-29)

### 34.1 Verdict and concrete findings

The §33 GO was not accepted from its report.  The complete
`12c2079^..abc81cb` range was read from the actual diffs, both target
routes were exercised through the public pipeline, retained evidence
was rechecked, and fresh malformed-pass probes were applied after the
managed LLIR pipeline.  The review found real correctness, fail-closed,
test-oracle, identifier, documentation, and evidence defects:

1. B1 assigned descended/root cursor roles by tuple position.  The
   semantically equivalent commuted RHS `B[k,j] * A[i,k]` therefore
   failed in both automatic arms even though the verified schedule and
   legacy target were valid.
2. B2 did not declare every result-owned broadcast bound and did not
   reject distinct loop binders that lower to one C++ name.  A valid
   broadcast leaf could emit an undeclared extent; a colliding hand-built
   program could shadow its own loop variable.
3. The claimed raw-legacy execution differential compiled the candidate
   source a second time instead of compiling independently generated
   legacy source for the same request.
4. B1 had no adequate post-pass completion boundary.  Identity/partial
   vector rewrites, metadata-hidden duplicate drains, extra appends or
   counter mutations, aliases through calls/address-of/data access,
   moved or wrapped drains, reparented insertions, an early row
   `continue`, raw `C0_pos`/`C1_pos` writes, wrapped position sentinels,
   and removal of an ABI validation could all escape successive partial
   validators.
5. Exact token reservations (`__restrict__`, runtime/type spellings)
   still admitted other implementation-reserved C++ identifiers such as
   `__asm`, `__restrict`, and `__typeof__`, plus display-name edges that
   manufacture double underscores when target suffixes are added.
6. The retained evidence manifest included its own digest and could not
   verify.  The prose also incorrectly attributed a separate generic
   sparse-reduction SIGSEGV to B2; B2's legacy defect is malformed
   storage (values without parent coordinates), not that SIGSEGV.

These findings temporarily invalidated the §33 exit claim.  They are
closed by eight focused local commits, with no amendment or reorder:

- `5dbd8c9` / `189461b`: cursor-role classification, B2 bounds/name
  validation, exact integral capacity, independent legacy execution,
  initial completion ownership, and focused regressions;
- `74b1e11` / `0447dc7`: exact drain-effect census and hidden-duplicate
  adversaries;
- `fb4586b` / `6cf8038`: the emitted restrict-token reservation and
  collision lock;
- `fd1f9a9` / `bcc6dfd`: the final exact completion contract,
  implementation-reserved identifier boundary, and complete regression
  matrix.

### 34.2 Final completion contract and latency correction

The final B1 boundary does not rely on a rendered-name search or a
per-effect allowlist.  `_SparseWorkspaceLowering` reconstructs the one
canonical LLIR function that must exist after the managed pipeline:
all ABI validation and input-prologue statements, both checked
compressed-position sentinels, the exact outer cursor header and complete
producer/drain/row body, the checked root close, and final sparse
assembly.  One exact stored-state comparison rejects every missing,
extra, moved, wrapped, aliased, malformed, cyclic, or shared-mutable
aggregate.  The comparator is iterative and requires fresh node/nonempty
container ownership; CPython's childless immutable empty-tuple singleton
is the sole sharing exemption.

An initial exhaustive multi-walker implementation was correct but added
about 38% to an activating B1 compile.  It was not retained.  The final
single-comparison implementation was gated in a same-session
200-warmup/2,000-sample A/B/A against `6cf8038`:

| run | p50 ms | mean ms | p95 ms |
| --- | ---: | ---: | ---: |
| candidate first | 5.0672 | 5.0957 | 5.3794 |
| base | 4.7375 | 4.7602 | 5.0500 |
| candidate second | 5.0707 | 5.0936 | 5.3939 |

Candidate/base ratios are 1.0696–1.0703 p50, 1.0700–1.0705 mean, and
1.0652–1.0681 p95, all inside the 1.10 compiler-latency gate.

### 34.3 Verification and evidence correction

At committed code/test tip `bcc6dfd`:

- the complete B1/B2 target file passed **81 tests in 184.98 s**,
  including real compilation/execution, exact legacy-source execution,
  commuted f32/f64 source parity and compiled f32 execution,
  broadcast/zero extents, all completion attacks, and identifier
  boundaries;
- the broad pure LoopIR membership (CIN lowering, iteration domains,
  levels, LLIR lowering, neutrality, oracle, printer, schedule passes,
  and verifier) passed **735 tests in 3.62 s**;
- an independent final adversarial selection passed **26**, and the
  cycle/shared-ownership matcher lock passed independently;
- the authoritative clean-detached run at exact `bcc6dfd` asserted
  import provenance and collected 4,656 tests with three performance
  cases deselected.  It reached **4,596 passed / 14 skipped** before the
  documented macOS libomp pthread-key ceiling produced a 43-node
  exhaustion cascade (39 literal `OMP Error #179`/`pthread_key_create`
  markers).  The exact `lastfailed` set was proven complete, unique, and
  non-overlapping across fresh-process partitions of 11+11+11+10; all
  four passed.  The proven union is therefore **4,639 passed / 14
  skipped / 3 deselected / zero code failures**;
- Black and Flake8 are clean on all changed source and test files, focused
  production mypy is clean, full-source mypy remains exactly the
  inherited **140 errors in 11 files**, and `git diff --check` is clean.

The retained §33 evidence tree now has a self-excluding, fully verifying
142-entry `SHA256SUMS`; its digest is
`7582577e49baafc3ce4815b0683988ded389f1243ecb97a489a4d48e42264b8b`.
The authoritative full-suite logs, XML, node partitions, and provenance
receipt are retained under
`~/.cache/scorch-codex/authoritative-bcc6dfd/`; its verifying 28-entry
manifest has SHA-256
`8623471a97ae2d2e9879e56a23174bb10bfaa91a7f2a04a0488782b4037895d8`.
The B2 wording in §§29–33 and the handoff is corrected to distinguish
malformed mixed-leaf storage from the separate memory-unsafe
sparse-reduction boundary.

### 34.4 Exit disposition

With the corrections above, the Phase-6 criterion-by-criterion verdict
returns to **GO**.  No schema/version, release dispatch, cache ownership,
legacy path, cutover, or selector changed.  No Phase-7 implementation
has begun; only the evidence-only comparand capture in §33.8 exists.
The opening coherent milestone remains the typed parallel
sparse-workspace-pool/two-pass runtime composition characterized there,
with the working public route as execution oracle and no use of the
malformed or memory-unsafe comparands as correctness oracles.

## 35. Phase-7 opening milestone: the parallel two-phase CSR SpGEMM slice (2026-07-29)

### 35.1 Scope and the independent §34 re-review

This session first independently re-reviewed the complete
`abc81cb..bcc6dfd` correction range from the actual diffs and reproduced
every §34 claim: the repaired 142-entry §33 evidence manifest and the
28-entry authoritative manifest verify at their recorded digests, the
full B1/B2 battery passed at the tip, the broad pure LoopIR membership
passed 735, commuted f32/f64 source parity plus compiled execution hold
(compiled f64 was added as a stronger fresh probe), B2 broadcast and
zero-extent behavior, the independent legacy-source execution, and the
implementation-reserved identifier boundary all reproduce.

The review then found one concrete defect the §34 corrections had
missed.  `_SparseWorkspaceLowering` retained its emitted outer `ForLoop`
object as the completion snapshot, and the emitted statements are the
exact objects the first managed pass receives: a hostile in-place
rewrite of the shared header at the sparse-prefetch entry (demonstrated
with a widened `<=` row bound) mutated the completion reference too and
compiled cleanly instead of dying at the boundary.  Two commits close
it:

- `70f1066` — the completion reference is now reconstructed from the
  verified program facts by a pure builder both emission and completion
  call independently, so it can never share mutable nodes with anything
  a pass can reach; the fresh-ownership census exemption is narrowed to
  the interned empty tuple (a shared empty *list* is mutable aliased
  state and is now rejected);
- `907ceb8` — fresh hostile probes: in-place header mutations at the
  pipeline entry, an exhaustive id-intersection proof that the
  completion reference owns no pipeline-entry state, aliased and cloned
  duplicate row loops, a purely aliased `BlankLine` no rendered
  comparison could see, matcher-level shared-empty-list and
  self-containing-loop locks, dynamic-vector idempotence (the pass-order
  probe), and the commuted compiled-f64 execution.

The activating B1 A/B/A latency gate (200 warmups / 2,000 samples,
candidate–base–candidate plus a base A/A control) is neutral: p50
ratios 0.995–1.000, mean 0.997–1.027, and the one 1.14 p95 reading sits
inside the demonstrated 3.4–4.4% base-vs-base drift band.  Evidence is
sealed under `~/.cache/scorch-codex/phase6-b1b2-review2-907ceb8/`.

### 35.2 Comparand audit and layer assignment

The two sealed sized automatic comparands were verified at their
recorded digests and audited fact by fact.  Both are two-pass OpenMP
count/fill kernels: a per-thread `linked_list_workspace_1d` pool sized
by a derived thread count, borrowed per-worker `make_view()` views
inside each of two parallel regions, a dynamic-chunk row loop, and an
exact serial prefix-sum/`torch::empty` allocation interlude.  Only the
`ds@ds->ds` comparand has honest final assembly.  The adjacent
`ss@ss->ds` comparand has the same runtime skeleton but malformed
row-scope assembly when the first operand has an empty row, as §35.4
demonstrates.  Responsibilities were assigned explicitly: LoopPlan owns
the automatic decision (both arms already record exactly the tile-free
sparse `WorkspaceInsertion` fact for `ds@ds->ds`); semantic LoopIR owns
the format-neutral base and region forms with no new node kinds and no
schema change (`scorch.loopir.canonical.v9` unchanged); target LLIR
owns the entire parallel/runtime composition; and the runtime ABI owns
the spellings (`linked_list_workspace_1d`, `make_view`,
`insert_unchecked`, `omp_get_thread_num`, `scorch_nthreads`,
`scorch_chunk`, `SCORCH_GRAIN_CODEGEN_SPGEMM`, `torch::empty`).  No
CSR-specific shortcut, rendered-name discovery, raw policy parsing, or
target spelling entered semantic LoopIR.

### 35.3 The ds@ds->ds vertical slice (`64c415a` / `df777cb`)

CIN lowering admits the dense-row/stored-sparse-reduction/stored-sparse
-column CSR reduction as the existing coordinate-merged `StoreReduce`
semantic form (an enum-classified family selection; every adjacent cell
keeps its historical seam code, including `reduce_to_csr`, the
trailing-level reduction, dense reduction domains, and the row-scope
shape).  `apply_sparse_workspace` pairs its admission per result format:
the doubly-compressed result keeps the B1 INTERSECTION-merge chain, and
the dense-row CSR result requires one plain dense row binder over a
single-cursor sparse reduction.  The production verifier, oracle,
erasure, and canonical serialization needed no changes: the scheduled
region program verifies as-is, the oracle returns exact `CsrMatrix`
storage with scheduled/base agreement, erasure round-trips canonically,
and both arms produce one canonical dump.

The dedicated `_ParallelSparseWorkspaceLowering` validates the exact
chain by cursor role (commuted operands are classified structurally),
reserves every runtime and two-phase helper spelling against display
names, emits the exact serial per-row assembly the legacy pipeline
hands to the shared production compressed-`Where`/OpenMP pass, and
supplies that same pass configuration to the shared managed pipeline.
The driver's applied branch mirrors the legacy composition; ownership is
fail-closed in both directions (a family outside the target can never
see two-phase output; a detached pass no-op can never degrade the
family to an unallocated serial assembly).  Byte parity holds in both
automatic arms for float32/float64 and the commuted order, and the
generated source reproduces the sealed comparand digest
`fa1026be…a4791c` exactly.

Two boundary corrections were required.  First, the general target
lowering now rejects any `StoreReduce` leaf into sparse result storage:
the newly admitted semantic form would otherwise reach the generic
route unscheduled and emit the known memory-unsafe unsized-values
kernel (demonstrated before the fix: `C_values[pC1] += …` into an empty
`std::vector`).  Second, the exact completion matcher compares frozen
`TensorAccessMetadata` by validated value outside the fresh-ownership
census, because the production two-phase rewrite legitimately
duplicates a work body whose detached statements retain the same
immutable provenance values.

Completion follows the §34 discipline: `complete_sparse_workspace`
reconstructs the entire expected post-pass function — pool policy in
typed and string form built together from the same structural facts,
both phase loops, the exact interlude through the frozen result ABI
snapshot's first-position/coordinate/value allocations, and the honest
storage epilogue — using only locally owned constructions, never the
managed pass being validated, then requires one exact fresh-ownership
match (`sparse_workspace_completion_lost` otherwise).

### 35.4 The battery and the row-scope disposition

The 61-test dedicated battery proves the slice end to end: byte parity
and the sealed digest in both arms; compiled execution against the
LoopIR oracle (exact storage), PyTorch, generated legacy and sealed
comparand sources, and the public dispatch; empty/disjoint/overlapping/
ragged supports, zero extents, explicit-zero cancellation,
deterministic storage under repeated execution, and fresh-process
`OMP_NUM_THREADS` 1-versus-3 runs proving thread-count invariance; route
ownership, replay, erasure, and canonical-dump stability; and the
adversarial surface (hand-built program parity, forged inserts,
reserved and colliding names, recorded stage loss, pass-ownership loss,
nine post-pass completion attacks, the pipeline-entry header mutation,
dynamic-vector idempotence, and the metadata matcher lock).  The
post-review correction in §36 makes the native differential genuinely
independent: it first checks the exact legacy or sealed bytes, then
appends one inert test-only comment before preparation so the native
kernel name, cache key, build directory, and shared object differ from
the candidate.

The adjacent `ss@ss->ds` row-scope family is deliberately dispositioned
as failure evidence, not migrated.  Its sealed comparand sizes
`C1_pos` by the first operand's *stored*-row count rather than the
result's dense row extent, so any input whose first operand has an
empty row returns malformed storage that silently associates later
rows' values with earlier rows.  The post-review hermetic lock in §36
regenerates and hashes the exact legacy source at
`cf1114aa…97b51`, optionally cross-checks the external sealed copy,
executes it under the independent native build identity, asserts the
exact malformed positions `[0, 5, 8, 11]`, reconstructs the
row-3-to-row-2 shift and wrong dense result, and proves the same kernel
correct on its full-row-support sub-domain.  A byte-parity twin would
knowingly reproduce wrong results; the family keeps its stable
`unsupported_sparse_output_reduction` seam and the battery locks both
the defect and the non-admission.  The mixed dense-leaf operand load
chain remains the next target-neutral slice.

### 35.5 Census, audit, captures, and latency

The regenerated deterministic family census (15 cells, both arms)
records zero route and zero arm divergence: the three Phase-7 SpGEMM
cells (base, float64, wider sizing) at byte parity, B1 at parity, the
three B2 cells admitted, and every seam at its exact prior code.  Four
fresh randomized cross-route sweeps (seeds 1, 2, 7, 11; 150 attempted
cases each through the retained f13ba79 harness) report zero parity
mismatches across 84–110 parity cases per seed; the older 60-case
harness variant behind the sealed v2 tallies was not retained alongside
its logs, so the fresh 150-case sweeps stand as this tip's randomized
census (a strictly larger case set).  The 86-case parallel-selection
audit is unchanged at 46 admitted / 40 rejected / 0 non-identical.
Fresh corpus (20/20), grid (42/42), anchor, heap, and auto captures are
byte-identical to the sealed `ebb243b` baselines — the auto capture
this time with zero cache-key normalization — and eight activating
Phase-7 captures (two arms × two dtypes × both operand orders) are
sealed with arm-identical digests.  Repeated public runtime
differentials pass 10/10.  Activating Phase-7 compile latency (200
warmups / 2,000 interleaved samples) is 1.075–1.080× the legacy route
across p50/mean/p95 — the LoopIR route re-verifies, applies, and
exactly completes on top of everything legacy does — inside the 1.10
gate.

### 35.6 Verification

All commands used the `scorch` conda environment.  At code tip
`df777cb`: the complete compiled batteries passed **493** (the Phase-7
battery, the full B1/B2 battery, pipeline execution, and the scheduled
slice); the broad pure LoopIR membership passed **735**; the
options/plan-identity membership passed **153**; the schedule
generality battery passed clean; Black and Flake8 are clean on every
changed file; focused production mypy is clean; full-source mypy is
exactly the inherited **140 errors in 11 files**; `git diff --check`
is clean.  The authoritative clean-detached run at exact `df777cb` (import
provenance asserted) reached **4,642 passed / 14 skipped / 3
performance deselected** in 2,979.70 s before the documented macOS
libomp pthread-key ceiling produced a 68-node exhaustion cascade (64
literal `OMP: Error #179`/`pthread_key_create` markers retained in the
sealed log).  The exact 68-node failed set was proven complete, unique,
and non-overlapping across fresh-process partitions of 17+17+17+17; all
four passed.  The proven union is therefore **4,710 passed / 14 skipped
/ 3 performance deselected / zero code failures**.

The five protected tracked files retained their exact SHA-256 values
through every commit, only explicit paths were staged, all unrelated
GPU/benchmark/scheduler/research/scratchpad material remains untouched,
nothing was pushed, and origin remains `58e8565`.  Evidence is sealed
under `~/.cache/scorch-codex/phase7-spgemm-df777cb/` (captures, census,
audit, latency, parity notes, and the full-suite receipts) beside the
§35.1 review evidence.  Its self-excluding manifest covers 172 entries
and has SHA-256
`eea1512710338fd770acc043aadaa5699f978929505ea7f2bfc19e607bbf0e6c`.

### 35.7 Phase-7 checkpoint audit

Criterion by criterion at `df777cb`: the typed parallel
sparse-workspace-pool composition exists as a verified target-owned
surface (pool, borrowed view lifetimes, two-phase count/fill, derived
thread/chunk/work-estimate policy) with every responsibility assigned
to its layer; the one admitted runtime-composition family holds byte
parity to its sound sealed comparand in both automatic arms plus
independent oracle/PyTorch/public correctness across the required case
matrix; fail-closed verification carries stable codes at every new
boundary, including both completion-ownership directions; canonical
request identity remains semantic with no version change; erasure is
pure; and release behavior is unchanged everywhere (production imports
untouched, all sealed captures byte-stable, the audit unchanged, and
the family census divergence-free).  The row-scope comparand is
dispositioned with a demonstrated soundness defect rather than
migrated, exactly as the no-unsound-parity rule requires.

**The Phase-7 checkpoint is GO** for this opening slice.  No release
dispatch/cache cutover, selector cutover, legacy deletion, Phase 8, or
Phase 8.5 was performed or started.  The remaining Phase-7 surface, in
order: the mixed dense-leaf operand load chain (target-neutral), a
sound re-derivation of the row-scope family (its typed twin must fix
the empty-row defect legacy ships, which breaks byte parity by
construction and therefore needs its own disposition decision), and the
general multi-compressed-level assembly.

## 36. Phase-7 opening-slice review corrections (2026-07-30)

### 36.1 Independent review findings

The complete `3ab4ee8..7e04b1a` range was re-read from the diffs and
probed beyond its committed locks before continuing Phase 7.  The
semantic family admission, sparse-workspace scheduling, two-phase
target composition, byte-parity cells, unsafe unscheduled quarantine,
and row-scope non-admission were otherwise sound, but the review found
four concrete boundary/evidence defects:

1. pass-visible `TensorAccessMetadata` objects owned fresh outer
   dataclass instances but still shared their nested `AccessId`,
   `SymbolId`, and `IndexId` objects with verified LoopIR state and the
   reconstructed reference.  A hostile pass could mutate a logical
   binder in place, compile successfully, and leave the returned
   scheduled program malformed;
2. actual and expected LLIR trees intentionally share enum singletons.
   Mutating (for example) `DataType.INT._value_` or
   `AssignOp.ADD_ASSIGN._value_` changed emitted C++ while both trees
   drifted together and the exact matcher accepted them;
3. sorting untrusted metadata-dictionary keys leaked raw `TypeError`
   for mixed key types, and the compressed-Where pass context received
   the lowering's own result `SymbolId`;
4. the native legacy differential prepared byte-identical source under
   the candidate's same JIT identity, so it could reuse the candidate
   module rather than independently exercise legacy semantics.  The
   row-scope failure lock also depended on the external evidence tree
   and did not prove the claimed row shift or the full-support
   correctness control numerically.

The review also corrected §35.2's overstatement: only the admitted
`ds@ds->ds` comparand has honest final assembly.  The row-scope
`ss@ss->ds` comparand shares the two-phase runtime skeleton but has the
demonstrated malformed assembly contract.

### 36.2 Production correction (`cb49ff7`)

Every LoopIR-target tensor-access metadata boundary that can cross this
managed-pass/completion boundary now copies the complete provenance
identity graph, including the general input/result metadata paths and
both sparse-workspace append paths.  The separate legacy lowerer is
unchanged.  The two-phase pass receives a fresh result `SymbolId`.  The
completion matcher validates exact stored identity state, requires
exact string dictionary keys without sorting untrusted values, and
converts malformed matcher state to
`sparse_workspace_completion_lost`.

The matcher also compares every LLIR enum singleton it can encounter
against an import-time stored-state snapshot.  Identity alone is no
longer sufficient: altered value/name/class/order state fails closed
even when actual and expected refer to the same singleton.  The
snapshot is limited to the three enum types in structured LLIR
(`AssignOp`, `DataType`, and `TensorAccessRole`); no schema,
serialization, scheduling, runtime ABI, or emitted spelling changed.

### 36.3 Regression and evidence correction (`6777f20` / `82cd687`)

The ownership census now includes LLIR nodes, metadata, every nested
provenance identity, nonempty tuple/list containers, and the verified
LoopIR program.  It proves pairwise disjoint ownership among the
pipeline entry, the independently reconstructed reference, and program
state.  Final-pass attacks cover all three nested identity kinds;
additional locks cover integer and string-subclass metadata keys,
compressed-pass result-ID mutation, and live `DataType`, `AssignOp`,
and `TensorAccessRole` singleton mutation.  All fail with the stable
completion diagnostic.

Legacy execution tests now establish exact source parity first and
then append one inert test-only comment before build preparation.  The
marker changes the native kernel name, cache key, build directory, and
shared object, so the differential cannot be satisfied by the
candidate module.  The row-scope lock is hermetic: it regenerates and
hashes the exact comparand source, optionally cross-checks the sealed
file, asserts malformed positions `[0, 5, 8, 11]`, reconstructs the
later-row shift and wrong dense result, and verifies ordered,
in-bounds, numerically correct storage on the full-support sub-domain.
The follow-up lock prepares the exact and marker-keyed sources without
building them and asserts that all four native identities differ:
kernel name, cache key, build directory, and shared-object path.

### 36.4 Latency regression and correction (`6cb2284` / `9cb2ea8`)

The required idle-machine latency gate found one more concrete defect
after the safety correction.  In two alternating candidate/base pairs,
the unoptimized corrected route measured 1.110–1.121× legacy at p50
and p95 and 1.137–1.141× at the mean, crossing the 1.10 gate; the
inherited base remained at 1.076–1.083×.  Candidate-vs-base LoopIR
latency was 1.023–1.041×, so this was a real correction cost rather
than a legacy-route or session-position effect.

`cProfile` over 2,200 activating compiles attributed 1.772 s and
695,200 calls to repeated enum-state validation: one completion tree
checked 316 enum occurrences but contained only 23 distinct global
singletons.  Metadata detachment itself cost only 0.066 ms per compile
and remains unchanged.  `6cb2284` therefore caches successful enum
validation only inside one synchronous exact-completion comparison,
after exact actual/expected identity.  The cache is discarded before
the next compile, so a later hostile mutation is still observed; no
metadata, alias, cycle, or full-tree comparison check is bypassed.
`9cb2ea8` proves repeated leaves validate once and that a mutation fails
again on the immediately following comparison.

The repeated optimized candidate/base/candidate/base gate is green.
LoopIR-vs-legacy ratios are 1.087–1.090 across p50/mean/p95 in both
candidate positions; candidate-vs-base LoopIR ratios are 1.000–1.010.
The optimization recovers the regression without hiding it in an A/A
band.

### 36.5 Verification and checkpoint disposition

The focused adversarial/evidence battery passed **13 tests**, the
unchanged source parity/boundary selection passed **17 tests**, Black
and Flake8 are clean on all three changed source/test files, focused
production mypy is clean, full-source mypy remains exactly the inherited
**140 errors in 11 files**, and `git diff --check` is clean.  A separate
independent probe run reproduced the disjoint ownership sets and every
stable failure class.  At exact final code/test tip `9cb2ea8`, the
clean-detached affected target/pipeline battery passed **293/293** in
736.51 s; broad pure LoopIR plus options/identity passed **888**, and
schedule generality passed **45**.

The authoritative exact-tip non-performance suite collected 4,735
unique selected nodes and proved its four fresh-process partitions
complete and non-overlapping by an exact sorted-union comparison.  The
result is **4,721 passed / 14 skipped / 3 performance deselected / zero
failures** in 3,431.00 aggregate pytest seconds.  The sole warning is
the inherited PyTorch sparse-invariant warning.  Import provenance and
detached-worktree cleanliness were asserted for collection and every
partition.

The corrections are emission-neutral: exact B1 and Phase-7 source
parity remains green in both automatic arms, including f32/f64 and
commuted operands, and the Phase-7 sealed source digest remains
`fa1026be…a4791c`.  The self-excluding 172-entry evidence manifest
verifies at
`eea1512710338fd770acc043aadaa5699f978929505ea7f2bfc19e607bbf0e6c`.
The activating-latency and final clean-worktree gates are green after
the bounded correction.  The opening Phase-7 checkpoint remains
**GO**.  No new Phase-7 family, release cutover, cache/selector change,
legacy deletion, Phase 8, or Phase 8.5 work was started.

## 37. Phase-7 broad milestone: mixed operand loads, multi-compressed assembly, and the sound row-scope route (2026-07-30)

### 37.1 Independent review of the correction range

The full `7e04b1a..94da378` correction range was independently re-read
from the diffs and reproduced before any new family work: the committed
sparse/parallel batteries pass **165/165** at the inherited tip, the
broad memberships pass **888**, the 92-entry review manifest and the
172-entry milestone manifest verify at their recorded digests, and both
sealed Phase-7 comparands match their recorded SHA-256 values.  Six
fresh hostile probes were then committed (`cb7e84a`), all failing
closed with the stable completion diagnostic: enum member ``_name_``
mutation and hostile attribute injection (the import-time snapshot pins
whole member state, not only values); an enum mutation between two
compiles failing the second compile end-to-end (no validation state
survives one synchronous comparison); a fresh ownership census proving
the pass-returned actual tree and the reconstructed completion
reference own pairwise-disjoint aggregates at the real matcher
boundary; a metadata role swapped to the other valid singleton
(invisible to state pinning, rejected by the exact role comparison);
and a mutation of the frozen module-level two-phase pass policy (the
completion reference owns its policy spellings locally, so only the
pass output drifts).  No concrete defect was found in the range.  One
review observation is recorded: the legacy sparse-prefetch pass forms
its mixed-operand prefetch address from the next stored row's
*coordinate* rather than its position; a prefetch hint never
dereferences, so the quirk is semantically inert and byte parity
remains admissible.

### 37.2 Physical position loads (`3e6e5ea`)

The reviewed spike ``PositionLoad`` node was ported to production
LoopIR: a value read of the scalar owned by one tensor's value-bearing
leaf position, valid for a DENSE leaf below compressed structure, with
no merge-alignment/default semantics.  The verifier admits input-only,
position-typed, tensor-linked, leaf-level reads (the new
``position_load_mismatch`` code plus the existing boundaries); the
oracle serves the read from ``LevelTensorStorage.leaf_value`` with the
spike's lazy dense materialization; canonical serialization moves to
``scorch.loopir.canonical.v10`` by the established node-kind precedent
with request.v2/loopplan.v1/autopolicy.v1 untouched.  Focused locks
cover six verifier defect paths, canonical dump/render stability, and
four oracle differentials.

### 37.3 The mixed dense-leaf operand-load slice (`9e3ce04` / `f172742`)

Input tensors whose compressed structure sits above a dense
value-bearing sub-tree (``sd``, ``sdd``, ``dsd``) now lower through
declared position-load chains: ``position_expr`` builds
the dense spine over the tensor's own levels and grounds it at the
single-cursor bound row position, so cursor/position linkage stays
level-based and format-neutral.  The general target admits a
single-cursor sparse outermost loop, folds root-parent pos subscripts
to the exact legacy integer spellings, records position-load spines in
the level-driver census, and emits the leaf read as the resolved
physical value access.  Byte parity with the legacy pipeline holds in
both automatic arms across the envelope — copy (f32/f64), elementwise
MUL in both operand orders, SpMV, row reduction, rank-3 ``sdd``/``dsd``
copies and the regblock-diverging ``sd@dd`` matmul.  Rank-1 ``s`` is
the compressed-leaf control and retains ``CursorValue``.  The 54-test
battery adds oracle/PyTorch/
independent-legacy-build execution, empty/explicit-zero/ragged/zero-
extent cases, fresh-process thread invariance, hand-built source
parity, and recorded stage losses.  Six adjacent seams stay fail-closed
with exact arm-consistent codes (merged and united mixed operands at
``unsupported_sparse_hierarchy``; hierarchical ``ssd``; mixed operands
into sparse-domain results; the permuted compressed pair; merged
outermost vectors).  One adjacent runtime gap is recorded: the public
``to_sparse("sd")`` conversion does not yet build dense-suffix block
values, so runtime mixed inputs are hand-built through ``TensorIndex``
in the battery.

### 37.4 The multi-compressed intersection-assembly family (`2fedaf4` / `a5b1a56`)

The production-caller census selected the B3 family: elementwise MUL
intersection chains into dense-prefix/multi-compressed-suffix results
(``ss``, ``sss``, ``dss``, ``ssss``) — the output formats production
``einsum`` infers for same-shape MUL over compressed inputs, previously
rejected at the ``unsupported_sparse_output`` seam.  CIN lowering
classifies the family when every dense-prefix coordinate iterates a
dense domain and every compressed-suffix coordinate is an intersected
stored stream, binds both aligned cursor positions at each merge, and
lowers to nested two-cursor INTERSECTION merges over an ordered
``AppendEntry`` leaf.  The dedicated target emits the legacy generic
composition statement-for-statement (per-level iterator groups with
folded root subscripts, ``std::min`` while-merges, initially indexed
leaf appends that the shared dynamic-vector pass rewrote, one conditional
compressed-parent append plus child position close per structural
level, the dense-prefix catch-up, and the root position close); the
family is serial, exactly like the legacy route, and the legacy
comparand is honest here (empty child intersections suppress their
parent coordinates and cascade), so byte parity is the gate.  Parity
holds in both automatic arms across all four formats, f32/f64, and
commuted operands; the 42-test battery adds exact oracle storage,
PyTorch and repeated public ``scorch.einsum`` differentials, the rank-2
and rank-3 empty-intermediate-parent cascades, explicit-zero retention,
deterministic storage, hand-built source parity, and six fail-closed
neighbors (united/single-cursor/copy/TTM streams at
``unsupported_sparse_output``; two dense prefix levels and three-
operand chains at ``unsupported_program_shape``).  Base
generalizations (shared root-subscript helper, level-parameterized
catch-up/close, PositionValue-grounded cursor parent chains, merged
bound-position census) are byte-neutral for every existing family.
Coverage is honest about mode order: nonidentity mode order over
compressed structure remains fail-closed for all migrated families
(``unsupported_mode_order``), so the B3 envelope is identity-order —
which is what default production einsum traffic generates after operand
alignment — and the permuted case stays a locked neighbor, not a
claimed capability.

### 37.5 The sound row-scope stretch (`7d2475a` / `f58643a`)

The ``ss@ss->ds`` row-scope reduction is admitted as the
``CSR_SPARSE_ROW`` semantic form (a stored-sparse row domain over the
B1 doubly-compressed merge shape), the sparse-workspace schedule
accepts the row-scope chain, and a dedicated target subclass reuses the
B1 producer/drain chain while replacing only the result assembly:
``C1_pos`` is sized and closed from the logical result row extent — a
per-row positional catch-up closes every skipped empty row before the
stored row's drain, the stored row closes its own entry, and a final
catch-up closes through ``C0_size`` — so the storage always carries
exactly ``C0_size + 1`` positions and empty rows are preserved.  The
defective legacy comparand sizes ``C1_pos`` by the first operand's
stored-row count; by construction the typed route never byte-matches
it (asserted explicitly), the retained comparand stays hermetic
failure evidence only, and the family is proven under the no-parity
discipline: structural activation in both arms and dtypes, exact
oracle indptr/indices/values with base/scheduled agreement, the
PyTorch reference, empty-row preservation, random grids, f64 disjoint
supports, zero extents, deterministic storage, the quarantined
unscheduled generic route, and a hostile pipeline-entry mutation dying
at the inherited completion boundary (13 tests).  Completion
reconstructs the full expected function locally under the established
fresh-ownership discipline.

### 37.6 Census, audit, captures, and gates

The deterministic family census grew from 15 to **32 cells** across
three regenerations (v4 after the mixed-load slice, v5 after B3, v6
after the row-scope admission), each with **zero route and zero arm
divergence**: ten mixed-operand cells and five B3 cells at byte parity,
the row-scope cell flipped to its no-parity admission, the
merged/hierarchical/single-cursor neighbors at exact rejections, and
every prior cell at its exact prior outcome.  The randomized
cross-route sweep was extended with mixed dense-leaf operand formats
and run at seeds 1/2/7/11/13 (150 cases each): 72-84 parity arms per
seed, **zero mismatches** (the small ``AssertionError`` reject class is
inherited — both routes reject those cases identically, as in every
retained seed log).  The 86-case parallel-selection audit is unchanged
at **46 admitted / 40 rejected / 0 non-identical**.  Fresh corpus
(20/20), grid (42/42), auto (23/23), anchor (22/22), and heap (11/11)
captures are byte-identical to the sealed ``ebb243b`` baselines, and
sixteen activating captures for the three new families are sealed with
per-arm digests (the mixed matmul cells arm-divergent exactly as their
legacy comparands are; every other cell arm-identical).

### 37.7 Verification

This subsection records the inherited attempt's evidence report.  The
independent audit in §38 found that its affected sweep was still red,
its full-suite run was incomplete, its latency harness did not activate
the three new families, and its ledger had no manifest; §38 therefore
supersedes those claims with exact-tip receipts.

All commands used the ``scorch`` conda environment.  At the final code
tip, the combined seven-battery affected sweep at the pre-flip tip
passed **616/619**, surfacing exactly the three stale rejection locks
the newly admitted families had outmoded (the B2 census's operand
probe, the SpGEMM battery's row-scope seam probe, and the pipeline
census's row-scope cell).  All three were flipped to the admitted
dispositions (``410280a``, ``f7de688``) and pass, and every dedicated
battery is individually green (54 mixed-operand, 42 multi-compressed,
13 row-scope, 165 inherited sparse/parallel).  The broad
pure LoopIR plus options/plan-identity memberships pass **899**
(the 888 inherited members plus the eleven new node-port locks);
focused production mypy is clean and full-source mypy is exactly the
inherited **140 errors in 11 files** with zero LoopIR findings; Black,
Flake8, and ``git diff --check`` are clean on every changed file.
Paired activating compile latency (200 warmups / 2,000 samples,
candidate/base/candidate/base against the inherited ``94da378`` tip)
has within-run LoopIR-vs-legacy ratios of **1.079-1.093** across
p50/mean/p95 in every candidate position (base 1.080-1.089), inside the
1.10 gate; the clean alternating pair gives candidate-vs-base LoopIR
ratios of **0.993-0.999**.  One middle pair was disturbed by an unrelated
application at 400% CPU; its 1.40-2.09 A/A absolute drift band is
recorded and the pair is superseded by the clean controls, whose
within-run normalization the drift never touched.  The authoritative
clean-detached full
non-performance suite at the exact inherited tip was incomplete; the
authoritative exact-tip union is recorded in §38.

### 37.8 Phase-7 checkpoint audit

Criterion by criterion: the mixed dense-leaf operand chain is
implementation-complete over the envelope its representation supports,
with the declared physical position load promoted from the reviewed
spike decision and every neighbor fail-closed; the named
production-reachable multi-compressed family (B3) is byte-parity
migrated with rank-3-and-4, parent-position descent, and
empty-intermediate-parent coverage, with nonidentity mode order
recorded as a locked representation boundary rather than claimed; the
row-scope family ships the sound dense-row sizing its legacy comparand
lacks, under the no-parity discipline with the defective kernel
retained as hermetic evidence and the generic unsized-values route
still quarantined; every new boundary carries a stable code; canonical
serialization moved to v10 by the node-kind precedent with request
identity unchanged; erasure stays pure; and release behavior is
unchanged (production imports untouched, sealed captures byte-stable,
the audit unchanged, censuses divergence-free).  **The Phase-7
checkpoint is GO for this broad milestone.**  No release
dispatch/cache/selector cutover, legacy deletion, Phase 8, or Phase 8.5
work was performed or started.

## 38. Independent review corrections for the Phase-7 broad milestone (2026-07-30)

### 38.1 Audited range and evidence corrections

The inherited milestone is ten commits, ``cb7e84a^..f7de688``, on
``94da378``; the earlier handoff inventory omitted the two
admitted-family lock commits ``410280a`` and ``f7de688``.  Every diff
was reviewed before extending the tree, and focused probes covered the
new PositionLoad, mixed dense-leaf reads, multi-compressed assembly,
and the no-parity row-scope route.

The evidence ledger was also audited rather than accepted as reported.
Its combined affected sweep was **616 passed / 3 failed**, not a final
green receipt.  The clean-detached suite collected 4,859 selected nodes
at ``f7de688`` but retained a receipt for only its first partition
(1,201 passed / 14 skipped).  The latency harness exercised the older
``ds@ds->ds`` SpGEMM route, not mixed loads, B3, or row-scope.  Finally,
``phase7-mixed-load-session`` contains no manifest.  Those incomplete
claims are discarded.  Independently usable inherited evidence remains:
the 165-test sparse/parallel battery, 899-test broad membership, the
32-cell zero-divergence census, 46/40/0 schedule audit, zero-byte
standard capture diffs, and the separately verifying 92- and 172-entry
manifests from the preceding Phase-7 correction ledgers.

### 38.2 Concrete defects and fixes

Two material trust-boundary defects were found and committed:

- ``370425d`` makes B3 output construction safe at its owning target
  boundary.  The inherited target emitted indexed writes into empty
  ``C_values``/coordinate/position vectors and relied on the generic
  dynamic-vector pass to rewrite them.  Omitting or misrouting that
  pass therefore produced compilable memory-unsafe C++.  B3 now emits
  ``emplace_back`` for leaf values and coordinates,
  ``scorch_vector_set`` for every compressed-position sentinel and
  close, and the existing safe parent-coordinate ``push_back``.  The
  generic pass still runs, but it is byte-neutral for this family and
  is no longer a correctness prerequisite.
- The same commit closes target and oracle binding TOCTOU windows.
  A caller-controlled ``Mapping.__iter__`` or ``__getitem__`` could
  mutate a frozen PositionLoad after the initial verifier pass and
  expose raw exceptions or untrusted state to lowering/execution.
  Both boundaries now verify before callbacks, snapshot exact unique
  ``SymbolId`` values into fresh keys, own values (including deep
  oracle input snapshots), and reverify after custom callbacks.

``a06ea1a`` locks the corrections: a rank-four pre-pass census proves
zero unchecked B3 result mutations and exact checked-call counts; a
no-op dynamic pass produces byte-exact safe source; target and oracle
mutation attacks cover both iteration and lookup; malformed programs
fail before callbacks; and rank-1 compressed leaves are explicitly
locked to production's existing ``CursorValue`` representation.  The
PositionLoad contract was clarified rather than artificially narrowed:
an independently bound compressed-leaf position is semantically sound,
while cursor-owned reads remain canonical whenever merge alignment or
UNION defaults are required.  Fresh probes confirmed UNION cannot use
PositionLoad to bypass its default-value rule.

A whole-tree B3 completion signature was considered and rejected on
measured grounds.  The recursive version cost about 1.29x, a flat
pickle/memo implementation still cost 1.19-1.30x, and even two raw
serializations cost about 1.106x before validation.  Because checked
construction makes the pass semantically inert for B3, the existing
generic pass-manager detachment contract is the right owner; adding a
family-specific hostile-pass audit to every production compile would
exceed the 1.10 latency budget without improving memory safety.

### 38.3 Exact-tip verification

At code/test tip ``a06ea1a``:

- broad LoopIR plus LoopPlan/options/request identity: **904 passed**
  (751 + 153);
- dedicated B3 battery: **44 passed**; the mixed-load battery's one
  concurrent JIT termination and the row-scope battery's one concurrent
  JIT termination both passed in isolated reruns;
- deterministic census: **32 cells / 0 divergence**; randomized
  seed-13 sweep: **150 cases / 76 parity arms / 0 mismatches**;
- retained schedule audit: **46 admitted / 40 rejected /
  0 non-identical**, normalized byte-equal to the retained result;
- corpus **20/20**, grid **42/42**, auto **23/23**, anchors **22/22**,
  and heap **11/11** are byte-identical to the retained captures, which
  chain to the sealed baseline;
- activating B3 candidate/base/candidate latency, with 200 warmups and
  2,000 samples per ``ss`` and ``ssss`` shape, has a worst ratio of
  **1.04651**.  Sources are exact at both revisions: 3,900-byte ``ss``
  SHA-256 ``5d145bbd0b62dcb527f47338520649479fbcf1667a27cf8477ab88458b5e39a7``
  and 7,148-byte ``ssss`` SHA-256
  ``db08dbf6a24d19123fbaa9d0ada163830a7dcef57722822e9b1344e9931a2d15``;
- changed files are Black/Flake8/mypy clean; full-source static baselines
  are unchanged at one Black file, nine Flake8 findings, and 140 mypy
  errors in 11 files, with zero LoopIR mypy findings; whitespace is
  clean;
- authoritative clean-detached collection: **4,873 nodes / 3
  performance deselected / 4,870 selected**.  The selected
  fresh-process union was **4,856 passed / 14 skipped / 0 failures**
  in eight non-overlapping file partitions (aggregate pytest time
  4,438.94 seconds);

The B3 latency samples and clean-worktree provenance are retained under
``~/.cache/scorch-codex/phase7-broad-review-a06ea1a/``.  The eight
full-suite partition outputs were observed but not durably retained;
the counts above are therefore a session result, not a sealed receipt,
and a future audit that requires raw logs must rerun the gate.  Exact-tip
capture and census commands were rerun directly against the retained
artifacts; no manifest is claimed for the older incomplete ledger.
Origin remained ``58e8565`` and no push occurred.  The five protected
files kept their recorded hashes, and no unrelated tracked or untracked
material was staged.

### 38.4 Checkpoint verdict

The two defects are closed without changing canonical v10, request or
schedule identity, public dispatch, release cache, selector behavior,
the native ABI, or the legacy pipeline.  The Phase-7 broad checkpoint
therefore remains **GO**.  The next milestone should widen ordered
sparse assembly rather than start cutover or Phase 8.

## 39. Phase-7 ordered-sparse-assembly milestone: single-cursor streams, union tails, and the dense-suffix conversion (2026-07-31)

### 39.1 Independent review of the correction range

The ``370425d``/``a06ea1a``/``3570b29`` correction range was
independently reviewed from the diffs before any new work.  Every
committed reproduction is green at the inherited tip: the B3 battery
(44, including the rank-four pre-pass mutation census and the no-op
dynamic-vector byte-identity lock), the mixed-operand, oracle, and
row-scope batteries (143 combined, covering both Mapping mutation
attacks, deep oracle-input ownership, and verify-before-callback
ordering), and the exact B3 source-parity locks in both automatic arms.
The retained activating-latency evidence under
``~/.cache/scorch-codex/phase7-broad-review-a06ea1a/b3-latency/``
verifies against its SHA256SUMS (4/4) at the recorded worst ratio
1.04651.  Origin remained ``58e8565`` and the five protected tracked
files hashed exactly as recorded.

Eleven fresh adversarial probes were then committed (``d6c759b``), all
passing without a defect: a hostile tuple-subclass shape value is
rejected by the exact-type boundary with zero caller-method
invocations; Mappings that iterate duplicate SymbolId keys or mutate
the key object during value lookup die inside both the target and
oracle snapshots; a foreign compressed oracle binding class is rejected
without a single attribute access; nested rank-2 dense containers are
deep-owned before output-shape callbacks; and the B3 checked-mutation
census and no-op-pass byte-neutrality locks extend from the committed
rank-4 cell to ``ss``/``sss``/``dss`` with exact per-format counts.
No concrete defect was found in the range.

### 39.2 The pre-implementation caller/seam/comparand census

Retained under
``~/.cache/scorch-codex/phase7-assembly-session/census/``
(``family_caller_census.py``/``.json``, ``legacy_union_soundness.py``/
``.json``, ``legacy_single_cursor_soundness.py``/``legacy_sc_soundness.json``,
plus per-cell legacy source exemplars):

- every candidate cell of both families rejected on the LoopIR route at
  exactly ``unsupported_sparse_output`` in both arms, while the legacy
  pipeline generated C++ for all of them, arm- and schedule-invariant;
- the single-cursor family is production-reachable through public
  ``scorch.einsum``: identity expressions over compressed inputs infer
  ``ss``/``dss`` outputs (``ij->ij``, ``ijk->ijk``) and mixed MUL
  (``ij,ij->ij`` over ``ss``×``dd``) infers ``ss``; twenty legacy
  execution cells through that route (copies, dense-zero products,
  commuted operands, empty rows, all-empty) match the ordered-stream
  reference exactly, including explicit zeros — the comparand is sound;
- no public operation spells elementwise sparse ADD; the compiler-level
  CIN entry is the union caller.  The legacy union kernels were executed
  directly under independent build identity across 28 cells (four
  formats × overlapping/disjoint/one-sided-rows/b-empty/a-empty/
  column-tails/identical-support) against an independent Python
  ordered-union reference: all sound.  Byte parity is therefore the gate
  for BOTH families — the B1/B3 discipline, not B2's no-parity mode;
- recorded pre-existing limitation: the low-level
  ``scorch.ops.lower_and_exec_cin`` result wrapper cannot wrap
  multi-compressed sparse outputs (it derives a dense result format and
  fails in ``TensorIndex`` validation); public einsum wraps them
  correctly, and the LoopIR ``execute_cin_via_loopir`` wraps from the
  verified declaration;
- neighbors censused with exact codes: SUB
  (``unsupported_sparse_subtraction``; legacy cannot compile SUB at
  all), union-with-dense (``unsupported_union_with_dense``), rank-1
  compressed outputs and dense-domain suffixes
  (``unsupported_sparse_output``), 3-ary chains, and ``sd`` trailing
  dense outputs (``unsupported_sparse_output_domain``).

### 39.3 The single-cursor multi-compressed family (``0881264`` / ``506cfc1``)

Classification widens the multi-compressed suffix from all-INTERSECTION
to per-level stream drivers in {single-cursor SPARSE, two-cursor
INTERSECTION}; united suffixes were left at the historical seam for the
next slice, and dense-domain suffixes keep ``unsupported_sparse_output``.
The construction needs no new machinery: SPARSE domains already bind
positions and build ``SparseFor``, so no node kinds, canonical schema,
or request identity changed.  The dedicated target admits SparseFor
levels in the assembly chain, drains them through the base sparse-loop
emission, and attaches the same conditional parent append and child
close every merged level owns.  Two legacy behaviors are reproduced
structurally: single-operand assemblies pre-size dense-parent position
vectors from the ABI-validated parent extent (the legacy
exact-dense-parent rule, derived from operand count, the operand's
declared layout, and statically bound result cells — the pre-sized
levels keep the raw in-bounds indexed close while every dynamic vector
stays checked), and the shared sparse-prefetch pass now also scans
checked value-append calls (``*_values.emplace_back``) for dense
value-array operands — byte-neutral for legacy trees, which still carry
only indexed assignments at that stage.  Byte parity holds in both arms
across ``ss``/``sss``/``dss``/``ssss``/``dsss`` copies (f32/f64),
``ss*dd``, ``ss*ds``, ``dss*ddd``, ``sss*ddd`` and commuted forms.  The
56-test battery adds exact oracle storage with base/scheduled
agreement, PyTorch and repeated public einsum differentials, pre-sized
empty-row coverage, explicit zeros (dense-zero products and hand-built
stored zeros), empty-intermediate-parent suppression, deterministic
storage, zero extents, the checked-mutation census with the pre-sized
level accounted, dynamic-pass byte neutrality, hand-built source parity
and oracle execution, and fail-closed neighbors (PositionLoad
co-operands at ``unsupported_program_shape``, dense-domain suffixes,
rank-1 compressed outputs, permuted compressed mode order at
``unsupported_mode_order``).

### 39.4 The rank-2+ union assembly family (``363cfed`` / ``6f6f085``)

The homogeneous union chain (elementwise ADD of two same-format
compressed operands into ``ss``/``sss``/``dss``/``ssss``/``dsss``) is
admitted as ``MULTI_COMPRESSED_UNION``.  Representation: the existing
``MergedSparseFor.positions`` field carries per-cursor union anchors —
no new node kinds and no canonical-schema change; the node's documented
semantics gain the union rule (a bound entry anchors descent only while
its cursor is aligned; a child stream chained from an unaligned parent
is the empty segment for that candidate coordinate).  The intended
contract allowed the absent-parent sentinel only for child-segment
selection, but the initial verifier enforced complete position binding
without preserving that optionality; §40 records the correction.
Union positions are allocated only for this family, so rank-1 unions
and CSR union columns keep their exact
prior position-free programs and bytes.  The target emits the legacy
union lattice statement-for-statement — the three-case while-merge,
per-case leaf folding through additive-identity defaults, one-sided
cases and post-exhaustion tails draining the surviving operand's whole
subtree through the same single-cursor stream emission the single-cursor
family owns, and the shared parent-append/close — at byte parity in
both arms for all five formats, f32/f64, and commuted operands.  The
initial 62-test battery adds exact oracle storage and source parity on
every format.  Compiled two-arm reference coverage spans
``ss``/``sss``/``dss`` with separate ``ssss`` execution; ``dsss`` did
not have compiled execution.  One-sided fixtures covered four formats,
whole-operand exhaustion only ``sss``, and the checked-mutation census
``ss``/``dss``/``ssss``; §40 widens all three omitted matrices.  The
battery also covers cancellation to a stored explicit zero, hand-built
explicit zeros and empty-intermediate parents, dense-prefix empty-row
closure, zero extents, repeated byte-stable compiled differentials, the
checked-mutation cases above (zero unchecked mutations), dynamic-pass
byte neutrality,
hand-built source parity and one-sided oracle descent, verifier/target
defect paths, and fail-closed neighbors.  One deliberate boundary was
added during review of a surprise admission: the united leaf envelope
is exactly the sum of the two united operand reads, so ``(A+B)*D``
dense-factor widenings fail closed at ``unsupported_program_shape``;
the 3-ary union moved from the layout seam to the same target code the
3-ary intersection already carries (recorded seam move), and SUB,
union-with-dense, and rank-1 united outputs keep their historical codes.

### 39.5 The dense-suffix conversion stretch (``2d1f436`` / ``607d3e1``)

Public ``to_sparse('sd')``/``to_sparse('sdd')`` (and every ``d``/``s``
layout whose value-bearing suffix is DENSE) previously ran the
per-entry legacy filter kernel whose storage carries values without
parent coordinates — the recorded defective dense-suffix comparand —
and hard-failed at storage validation.  The conversion now materializes
the layout directly (densify, collapse trailing dense levels into
blocks, store one block per prefix path exactly when it contains a
nonzero, build the prefix arrays level by level), identity mode order
only, with other mode orders and invalid formats keeping the historical
path and staged failure bookkeeping.  The 16-test battery locks round
trips (``sd``/``sdd``/``dsd``/``ssd``/``dssd``), exact-storage
equivalence against the hand-built ``TensorIndex`` builders,
interior-zero retention, canonical empty storage, dense-prefix position
closure, f64, sparse-source reconversion, untouched compressed-leaf
formats, the identity-order boundary, and compiled execution through
the mixed-operand route.  The initial fast path nevertheless returned
before compiler-options/context validation, ignored the caller's
context while densifying sparse inputs, admitted wrong-rank formats,
and could mutate a sparse receiver before a later materialization
failure; §40 records the correction.  The mixed dense-leaf runtime
batteries now build their inputs through the public conversion, closing
the recorded hand-built-storage requirement (the hand-built builders
remain as exact comparands).

### 39.6 Census, sweeps, audit, and captures

- Deterministic census v7 (47 cells, the 32 inherited cells plus the
  single-cursor flip, five copy/commuted/mixed cells, five union parity
  cells, and four new seam cells): **zero route and zero arm
  divergence**; every inherited cell at its exact prior outcome.
- Randomized cross-route sweep v3 (sparse result formats and union
  value operators added): seeds 1/2/7/11/13 × 150 cases, 34-42 parity
  arms per seed, **zero mismatches**; reject codes are the expected
  seam census.  The retained v2 sweep at seed 13 reproduces
  byte-identically (76 parity arms, zero mismatches), proving the
  legacy-reachable envelope unchanged.
- The 86-case schedule audit: **46 admitted / 40 rejected /
  0 non-identical**; JSON equal to the retained result after
  normalizing only the embedded commit field.
- Captures: corpus **20/20**, grid **42/42**, auto **23/23**, anchors
  **22/22**, heap **11/11** byte-identical to the retained surfaces,
  which chain to the sealed ``ebb243b`` baselines.  Twenty-two
  activating captures for the two new families are sealed with per-arm
  digests; every cell is arm-identical (both families are serial).

### 39.7 Verification at the exact tip

At code/test tip ``607d3e1`` (all commands in the ``scorch`` conda
environment; evidence under
``~/.cache/scorch-codex/phase7-assembly-session/``):

- full 15-file LoopIR battery sweep: **1220 passed / 0 failed**
  (includes 56 single-cursor, 62 union, 45 B3, 77 mixed+conversion);
- pure-LoopIR membership (12 files): **1401 passed**; options/
  plan-identity membership: **153 passed**;
- repeated compiled public differentials (three rounds × five cells,
  including an ``sd`` operand built by the new public conversion):
  all green;
- activating A/B/A compile latency (200 warmups / 2,000 interleaved
  samples per shape; candidate ``607d3e1`` vs base ``3570b29``
  worktrees): worst candidate within-run LoopIR/legacy ratio
  **1.02979** across p50/mean/p95 on ``dss`` copy, ``ss`` union,
  ``ssss`` union, and the shared B3 control, in all three candidate
  runs including the order-flipped control; the union cells run at
  0.82-0.92 (faster than legacy); base-run control ratios 1.018-1.026
  match the candidate's B3 ratios; the A/A pair agrees within run
  normalization (absolute drift 0.81-1.01 superseded by within-run
  ratios, per the established discipline);
- static parity proven **base-vs-candidate under one invocation**:
  Black one pre-existing file, Flake8 nine pre-existing findings, and
  full-source mypy **byte-identical after line normalization (146
  errors in 12 files at both revisions, zero LoopIR findings)** — the
  previously recorded "140 in 11 files" was the inherited session's
  environment count, and the parity claim here is the exact
  base/candidate equality; ``git diff --check`` clean;
- exact-tip clean-detached full non-performance suite at ``607d3e1``:
  **5,012 nodes collected / 3 performance deselected / 5,009 selected;
  4,995 passed / 14 skipped / 0 failures** (aggregate pytest time
  4,420.24 seconds) in eight file partitions, each a fresh
  process with its raw log retained under
  ``phase7-assembly-session/full-suite/``; the partition union is
  proven complete and non-overlapping against the collected file list
  (``union-proof.txt``).

The historical evidence directory retains the census, soundness
probes, sweep logs, audit, captures, activating digests, latency
samples, static-parity outputs, and full-suite partition logs.  It is
not a valid seal: ``tip-at-docs.txt`` was edited after ``SHA256SUMS``
was created, so 224 of 225 entries verify; §40 supplies the corrected
review ledger.  Origin remains ``58e8565``; nothing was pushed; the
five protected tracked files hash exactly as recorded; only explicit
paths were staged.

### 39.8 Phase-7 checkpoint audit

Criterion by criterion: both named ordered-assembly families are
implementation-complete over their proven envelopes at byte parity in
both automatic arms, with exact oracle and PyTorch differentials and
production-caller coverage where a public route exists; every neighbor
carries a stable fail-closed code, including the two boundaries this
milestone deliberately added (the united leaf shape and union-chain
homogeneity) and the one recorded seam move (3-ary union to the target
code); the representation change is semantic-only over the existing
schema (union positions), with canonical v10, request identity,
schedule identity, and erasure untouched; the runtime dense-suffix
conversion closes the recorded ``to_sparse`` gap without touching any
compiled route; release behavior is unchanged (production imports
untouched, sealed captures byte-stable, the audit unchanged, censuses
divergence-free, latency inside the 1.10 budget).  **The Phase-7
checkpoint is GO for this milestone.**  No release
dispatch/cache/selector cutover, legacy deletion, Phase 8, or Phase 8.5
work was performed or started.  This initial verdict is superseded by
the corrected audit in §40.

## 40. Phase-7 ordered-assembly rigorous review corrections (2026-07-31)

### 40.1 Independent review result

The inherited range was nine commits, not eight: seven code/test
commits from ``d6c759b`` through ``607d3e1`` and two documentation
commits, ``a7c37b0`` and ``a0b88d3``.  The range was reviewed
diff-by-diff before any correction.  The single-cursor lowering held:
an exhaustive rank-2-through-rank-4 ``d``/``s`` operand-layout sweep
found 66 admitted programs with zero oracle or source-parity mismatch.
The UNION target also preserved aligned, one-sided, post-exhaustion,
and empty-child assembly and failed closed on reordered, partial,
heterogeneous, n-ary, and widened forms.

Two material defects remained:

1. A position bound by a UNION merge can be absent for one operand at a
   one-sided coordinate.  The verifier recorded it as an unconditional
   physical position.  A verified forged program could therefore feed
   that binding to ``PositionLoad`` or dense-position arithmetic; the
   oracle then received its private absent-parent sentinel where an
   integer position was required.  This violated verifier/oracle
   totality even though the admitted target envelope rejected the
   forged leaf later.
2. The direct dense-suffix ``to_sparse`` route returned before exact
   ``CompileOptions``/``CompilationContext`` validation, did not require
   the requested rank to match the tensor rank, and densified sparse
   receivers through the mutating default route without the caller's
   context.  A failure during later block materialization could leave
   the source converted to dense, and valid timing owners silently
   recorded no work.

The semantic dense-suffix storage algorithm itself held under an
independent 420-cell sweep: every ``d``/``s`` dense-suffix layout at
ranks 2 through 7, including every zero-axis position, decoded exactly
to its dense reference.  The defects were boundary ownership and
failure atomicity, not the block representation.

### 40.2 Corrections

- ``1a92f50`` tracks optionality in the verifier's position type.
  UNION-bound positions may select only the immediate next COMPRESSED
  child segment; unconditional ``PositionLoad`` and dense-position
  spines fail with ``unsupported_sparse_hierarchy``.  Ordinary
  ``SparseFor``/``SparseWindowFor`` and INTERSECTION bindings remain
  unconditional.  Valid source emission is unchanged.
- ``9d5fb16`` moves dense-suffix admission after exact options/context
  validation, requires format-rank equality, and obtains sparse-source
  snapshots through ``to_dense(in_place=False)`` with the caller's
  exact options and context.  State is committed only after complete
  output construction; wrong-rank requests retain the historical
  staged failure.
- ``f196147`` locks both fixes and closes the inherited test-matrix
  overclaims.  The UNION file now has 76 tests: one-sided fixtures,
  whole-operand exhaustion, and checked-mutation censuses cover all
  five output formats.  The dense-suffix file now has 22 tests,
  including invalid/foreign boundary objects, context propagation,
  wrong-rank stage accounting, and injected post-densification failure
  atomicity.

The target-family docstring now names both INTERSECTION and UNION, and
the verifier overview lists every position binder.  Canonical v10,
request and schedule identities, erasure, valid program structure,
public dispatch, release caches, and legacy emission are unchanged.

### 40.3 Verification

All code/test gates ran at the clean detached code/test tip
``f196147`` with import provenance asserted:

- focused non-overlapping review memberships: **498 LoopIR tests** and
  **123 runtime/stage tests**, all passing;
- schedule audit: **46 admitted / 40 rejected / 0 non-identical**, byte
  equal to the retained JSON after removing only the commit field;
- deterministic census v7: **47 cells / 0 divergence**; fresh randomized
  seed 17 × 150: **40 parity arms / 0 mismatches**;
- regenerated capture surfaces: corpus **20/20**, grid **42/42**,
  anchors **22/22**, and heap **11/11** byte-identical to the retained
  files; every automatic C++/CIN artifact is identical, with only the
  same two process-dependent cache-key characters differing in its
  JSON report;
- target-activating latency, 200 warmups and 2,000 interleaved samples
  per cell in both orderings: every metric is within 1.10; the worst
  ratio is **1.03349**;
- Black and Flake8 add zero findings.  Full-source mypy is exact
  base/candidate parity at the current invocation's **140 inherited
  errors in 11 files** (the same two ``stensor.py`` findings merely
  move by line number), with zero LoopIR findings;
- clean detached full non-performance suite: **5,032 collected / 3
  performance deselected / 5,029 selected / 5,015 passed / 14 skipped /
  0 failures**, in eight file-disjoint fresh processes.  The partition
  union is complete and non-overlapping and no libomp resource event
  occurred.

The inherited ``phase7-assembly-session`` evidence is retained but is
not called sealed: its ``tip-at-docs.txt`` actual digest is
``db383fb5...``, while ``SHA256SUMS`` expects ``2c786f09...``.  The new
review ledger is
``~/.cache/scorch-codex/phase7-assembly-review-f196147/``.  It is sealed
outside Git only after the final documentation tip, excludes generated
``__pycache__`` material, and includes the exact-tip full-suite logs,
capture regenerations, audit, latency samples, and the inherited-seal
failure receipt.

### 40.4 Corrected checkpoint verdict

Both defects are closed without widening the admitted compiler family
or changing valid generated source.  The ordered single-cursor and
UNION assembly envelopes remain sound at byte parity, and the public
dense-suffix materializer now has the same strict ownership and
failure-atomicity boundary as the staged compiler route.  The
Phase-7 checkpoint therefore remains **GO for the corrected milestone**.
No cutover, cache/selector change, legacy deletion, Phase 8, or Phase
8.5 work was started.

The next milestone should close the remaining production-reachable
sparse-assembly compatibility envelope: census first, then the coherent
rank-1 and merged dense-leaf non-reduction families, followed by the
multi-compressed reduction/TTM vertical slice.  A Phase-8 inventory may
begin only after a fresh Phase-7 exit audit reaches a genuine GO.

## 41. Phase-7 compatibility-envelope milestone: conversion defects, the census, and rank-1 assembly (2026-08-07)

### 41.1 Independent review of the inherited range

The ``1a92f50``/``9d5fb16``/``f196147``/``9ce6836``/``a606e11`` range was
reviewed diff-by-diff before any new work, and every reproducible claim
was re-derived rather than trusted.  Origin remained ``58e8565`` and the
five protected tracked files hashed exactly as recorded.  The
replacement review ledger
``~/.cache/scorch-codex/phase7-assembly-review-f196147/`` verifies
**175/175** entries under ``shasum -a 256 -c SHA256SUMS``.

Both corrected boundaries reproduce.  Seven independent probes, written
from the ``nodes.py``/``verifier.py`` contracts rather than from the
committed tests, confirm that:

- a ``PositionLoad`` on a UNION-bound leaf position fails at
  ``unsupported_sparse_hierarchy`` on path ``...value.lhs.position``;
- the dense-child misuse — a ``DensePosition`` whose parent is a
  UNION-bound position — fails at the same code on path
  ``...value.position.parent``;
- the two *sound* shapes stay admitted and total: a nested ``SparseFor``
  chained from a possibly-absent parent, and a nested UNION merge whose
  cursors both chain from possibly-absent parents.  ``_segment`` is the
  single choke point all three loop forms use (``SparseFor``,
  ``SparseWindowFor``, ``MergedSparseFor``), so an absent parent always
  yields the empty segment, the freshly bound child position is never
  observed, and the verifier's unconditional typing of it is correct.  A
  one-sided row present only in the other operand produced no entries and
  no parent coordinate, exactly as the conditional-append discipline
  requires.

The dense-suffix conversion reproduces on all four claims under fifteen
independent probes: a **720-cell** decode sweep against an independently
written block decoder (ranks 2-5, every ``d``/``s`` dense-suffix layout,
three extent shapes, f32/f64, and random/all-zero/interior-zero/dense
patterns), a **48-cell** zero-extent sweep, exact options/context
rejection, rank equality, injected post-densification failure atomicity
for both dense and sparse receivers, and non-aliasing of the produced
values.

Focused counts reproduce exactly: the UNION battery collects **76**, the
dense-suffix battery **22**, the single-cursor battery **56**; run
together they are **154 passed**.

Two documentation defects were found:

1. §39.1 records the B3 battery as ``44`` and §39.7 as ``45``.  The B3
   multi-compressed battery collects **43** tests, at ``607d3e1`` and at
   the inherited tip alike; the range never touched that file.
2. The headline focused memberships "498 LoopIR tests and 123
   runtime/stage tests" name no file set and no command, and the
   replacement ledger's ``focused/`` directory is empty (0 of its 175
   manifest entries lie under it).  ``498`` is reproducible only by
   guessing a membership; ``123`` could not be reproduced from any
   documented file set.  They are recorded here as **unreproducible**,
   not as gates.

**No concrete defect was found inside the reviewed range.**

### 41.2 Four concrete defects found by fresh probes, and their fixes

The census probes surfaced four concrete defects in the public
``to_sparse`` conversion — the same surface ``2d1f436``/``9d5fb16``
corrected for dense suffixes, left unfixed on two neighbouring families.
All four are pre-existing and outside the reviewed range; all four are
fixed here, before any representation change.

**(a) ``530571e`` — multi-dense-parent layouts could not be built at
all.**  Public ``to_sparse`` raised ``TensorIndexError`` for every
``d``/``s`` layout with a compressed value-bearing leaf outside
``d?s+`` — ``dds``, ``sds``, ``ddds``, ``ddss``, ``dsds``, ``sdds``,
``sdss``, ``ssds``.  The per-entry filter kernel sizes a compressed
level's position array from its immediately enclosing level alone, so
``dds`` over ``(4,5,6)`` produced a 6-element position array where 21
were required.  The block materializer already handled these layouts
correctly (with no trailing dense level the block is one scalar and the
walk degenerates to ordinary sparse storage); only its admission
predicate excluded them.  ``_is_dense_suffix_format`` becomes
``_is_directly_materialized_format``, stated structurally as "not
``d?s+``" — the exact family the filter kernel does assemble.  The new
predicate is a strict superset, so every previously routed layout is
byte-unchanged and every ``d?s+`` and all-dense layout keeps the kernel
path.  Verified by an independent decoder over **1,152** routed-family
cells and **69** zero-extent cells.

**(b) ``cbc466f`` — rank-1 re-conversion silently corrupted the
receiver.**  The rank-1 branch filtered ``self.values`` unconditionally,
so ``to_sparse('s')`` on an already-compressed rank-1 tensor
reinterpreted stored positions as dense coordinates: ``[0,1,0,0,2]``
round-tripped to ``[1,2,0,0,0]`` with no exception.

**(c) ``cbc466f`` — rank-1 ignored ``fmt`` and the compiler boundary.**
``to_sparse('d')`` and ``to_sparse('ss')`` both returned compressed
rank-1 storage, and foreign ``_compile_options``/``_compilation_context``
objects were never rejected.  The branch now mirrors the rank>=2
discipline: validate the boundary first, require format-rank equality,
reject a rank-1 request naming no compressed mode, and take a
non-mutating dense snapshot under the caller's exact options and
context.  Unparseable formats keep the historical path.

**(d) ``2017aa0`` — prefix assembly was quadratic.**  The materializer
rescanned the whole stored-path list once per parent at every level, so
cost grew ~3.9x per problem doubling (3.552 s for a 32,744-block ``ssd``
conversion).  ``torch.nonzero`` is row-major, so stored paths are
already lexicographically sorted; the walk now groups each level's
children in one pass and carries enumerated parents forward.  Pure
performance change, proven **byte-identical** over a **789-cell**
differential that runs the verbatim pre-change assembler beside the live
one, including every zero-extent case.  The measured conversion drops to
**0.148 s** and scales linearly.

``79d2afe`` locks all four (42 new cells in the dense-suffix file, 18 in
a new rank-1 conversion file).

### 41.3 The Phase-7 compatibility-envelope census

Census v10 records, for every declared cell and in both automatic arms,
the LoopIR outcome and exact fail-closed code, whether legacy generates
C++, whether the legacy kernel executes soundly with well-formed
identity-ordered storage, and public reachability.  Legacy comparands
execute through an independently keyed direct build, because the
low-level ``lower_and_exec_cin`` wrapper cannot wrap multi-compressed
sparse outputs (the pre-existing limitation recorded in §39.2).  Each
cell runs in its own process so a native crash records a receipt instead
of losing the run.  **55 cells; every cell arm-invariant.**

- **Rank-1 compressed outputs** — copies, unions, intersections, mixed
  ``s*d`` and commuted ``d*s``, and an empty operand — are **admitted**
  after this milestone (§41.5).  ``s-s`` keeps
  ``unsupported_sparse_subtraction`` (legacy cannot compile SUB at all),
  ``s+d`` keeps ``unsupported_union_with_dense``, the 3-ary chain keeps
  ``unsupported_program_shape``, and ``s->d`` was already admitted.
- **Compressed-parent/dense-leaf co-operands** — ``ss*sd`` and its
  commuted, f64, rank-3, rank-4, ragged and empty forms — sit at
  ``unsupported_program_shape`` in both arms.  Legacy generates *and*
  executes them with well-formed storage, so this family **is**
  byte-parity gateable when migrated.  ``ss+sd`` keeps
  ``unsupported_union_with_dense`` and ``sd`` copy keeps
  ``unsupported_sparse_output_domain``.
- **Multiple dense prefixes and interleaved dense levels** — ``dds``,
  ``sds``, ``dds+dds`` at ``unsupported_sparse_output``; ``ddss``,
  ``ddss+ddss`` at ``unsupported_program_shape``.  These inputs can now
  be *built* (§41.2a) but their legacy execution produces malformed
  storage, so a future migration must gate on the oracle, not parity.
- **Multi-compressed reduction/TTM** — §41.4.

Two further pre-existing public-route defects were characterized, both
outside this milestone's scope and neither introduced by it:

- ``einsum('ij->i')`` — a reduction into a rank-1 sparse output — emits
  C++ that **does not compile**: the legacy sparse-workspace drain
  subscripts a scalar workspace key
  (``scorch_vector_set(T0_crd_vec, pT, it.first[0])`` →
  *subscripted value is not an array, pointer, or vector*).
- ``einsum('ijk->ijk')`` into ``dds`` fails at
  ``TensorIndexError: compressed mode 2 position array has 5 elements,
  expected 13``.  The *input* now builds correctly (§41.2a gives it a
  13-element position array); the failure is in the legacy **output**
  assembler, which still sizes a compressed level from one dense extent.
  This defect is pre-existing and latent — fixing the input conversion
  is what made it reachable.

### 41.4 The reduction/TTM comparand is not byte-parity gateable

Eight reduction cells — ``sss@dd`` TTM at f32 and f64, ``dss@dd`` TTM,
``ss@dd`` and ``ss@ss`` into ``ss``, two ``ds``-output nests, and a
ragged TTM — **terminate the process with SIGSEGV (exit 139)** when
their legacy C++ is executed on well-formed identity-ordered inputs.
Compilation succeeds (4,207 characters for the ``sss@dd`` cell) and the
module loads; the crash is inside ``evaluate``.  A standalone
reproduction is retained as an executable receipt.

This is a concrete instance of the memory-safety failure ``lower_cin.py``
already attributes to the sparse-reduction comparand.  It settles the
gating question in advance: the reduction/TTM family **can never be
gated on byte parity** and must use the LoopIR oracle and PyTorch
differentials.

The crash is confined to the legacy *generic comparand* route driven
directly by the census.  Production dispatch is unaffected:
``scorch.matmul`` over ``ss``x``ss`` and ``ds``x dense both return
results matching ``torch.matmul``.  The LoopIR route rejects these
nests at ``unsupported_sparse_output`` (TTM and rank-general
reductions), ``unsupported_sparse_output_domain`` (``ss@dd``),
``unsupported_sparse_output_reduction`` and ``unsupported_program_shape``
(the ``ds``-output nests) — arm-invariantly.

### 41.5 The rank-1 compressed assembly migration (``8e2d279`` / ``04dbe60``)

A rank-1 all-compressed result is the degenerate case of the family the
ordered-assembly target already owns: one stored stream, no dense
prefix, and no parent level to close.  Exactly three sites spelled the
structural rule as "two or more compressed suffix levels" — the CIN
family classifier (``_classify_sparse_output_family``), the structural
router (``_multi_compressed_assembly_chain``, in both its layout test
and its stream count), and the target's chain collector
(``_collect_assembly_chain``).  All three now also name the degenerate
case, through level identities alone.

No new node kinds, no canonical-schema change, and no request- or
schedule-identity change.  No CSR-specific shortcut, runtime-format
sniffing, rendered-name or regex routing, and no operation-specific
target hack: canonical CSR keeps its own dedicated family because
``(DENSE, COMPRESSED)`` is not a rank-1 result.

The legacy comparand is honest here, so the gate is the B1/B3
discipline:

- **byte parity** with ``legacy_generated_cpp`` over **20 cells** —
  ``s`` copy, ``s+s`` union, ``s*s`` intersection, ``s*d`` and the
  commuted ``d*s``, at f32 and f64, in both automatic arms — all
  identical;
- a **PyTorch differential** over **140 compiled cells**: seven fixtures
  (random, disjoint supports, either operand empty, exact cancellation,
  identical supports, hand-built explicit zeros) x two dtypes x five
  shapes x both arms — all matching;
- every excluded neighbour keeps its exact prior code in both arms:
  ``s-s`` at ``unsupported_sparse_subtraction``, ``s+d`` and ``d+s`` at
  ``unsupported_union_with_dense``, 3-ary MUL and ADD at
  ``unsupported_program_shape``;
- the rank-1 dense output and the ``ss`` copy, ``ss+ss``, ``ss*dd`` and
  ``ds+ds`` rank>=2 families are byte-unchanged.

The 93-test battery adds honest identity-ordered storage carrying the
exact ordered support, cancellation retaining a stored explicit zero,
canonical empty storage, repeated byte-stable execution, and the
single-compressed-level source shape (no ``C1_*`` arrays, no dense-size
initializer).

**Recorded seam move.**  Three inherited locks asserted that rank-1
compressed outputs stay at ``unsupported_sparse_output``.  That seam now
belongs to the admitted family, so each lock moves to the neighbour that
still occupies it — a single compressed level under two or more dense
parents, which the one-dense-prefix rule excludes — and names the move
in place.

**Production reachability.**  The family is reachable through public
``scorch.einsum``: ``i->i`` over a compressed vector and ``i,i->i`` over
compressed x compressed and compressed x dense all infer an ``s`` result,
assemble the exact ordered support, and match the dense PyTorch
reference.  (An earlier reading of this session claimed a dense
inference; that reading was wrong because it printed the result format
after calling ``to_torch()``, which densifies the receiver in place.
The corrected measurement reads the format first.)

Two further corrections to expectations formed during the work: the
``ds`` CSR *copy* rejection at ``unsupported_program_shape`` is present
at the inherited tip and is unrelated to this widening; and the
``C6``/``C7`` census cells labelled "CSR control" do not reach the
admitted CSR reduction family, because their loop nest is not that
family's shape.

### 41.6 Verification

All gates ran in the ``scorch`` conda environment; evidence is under
``~/.cache/scorch-codex/phase7-envelope-session/``.

- **Focused batteries**: the widened conversion file **64 passed**, the
  rank-1 conversion file **18 passed**, the rank-1 assembly battery
  **93 passed**, the three updated seam files **181 passed**, the
  eight-file LoopIR battery **758 passed** before the seam update (its
  5 failures were exactly the obsolete rank-1 seam assertions), and the
  conversion-adjacent regression subset **214 passed**.
- **Schedule audit** at the new tip: **46 admitted / 40 rejected / 0
  non-identical**, and its JSON is **equal to the retained baseline after
  removing only the commit field**.
- **Capture surfaces**: corpus **20/20**, grid **42/42**, anchors
  **22/22**, heap **11/11** byte-identical to the retained files; the
  automatic surface's every C++/CIN artifact is identical, with only the
  same **two process-dependent cache-key characters** differing in
  ``report.json``.
- **Static parity**, base ``a606e11`` versus candidate: Black flags the
  same single pre-existing file (``prebuilt_kernels.py``) at both
  revisions, Flake8 reports **9** findings at both, and full-source mypy
  is **140 errors in 11 files at both**, exactly equal after line
  normalization, with zero LoopIR findings.  ``git diff --check`` clean.
- **Census v10**: 55 cells, zero arm divergence.

- **Repeated compiled public differentials**: eight public cells
  (``einsum`` rank-1 copy/product/mixed, ``ss`` copy, ``ss*dd``,
  ``ss*sd`` built by the widened conversion, ``dss`` copy, and
  ``matmul`` over ``ss``x``ss``) match the dense reference and are
  byte-stable across three rounds.
- **Activating A/B/A compile latency**: 200 warmups and
  2,000 interleaved samples per cell, in both orderings, over the three
  newly activating rank-1 cells plus the shared ``ss`` intersection
  control.  Every metric is inside the 1.10 budget: the worst
  within-run LoopIR/legacy ratio is **1.04189** in the primary ordering
  and **1.06632** in the order-flipped control.  The rank-1 union cell
  runs at 0.988-0.994 (faster than legacy), and the shared control reads
  1.021-1.030, matching the 1.018-1.026 range the inherited session
  recorded for it.
- **Exact-tip clean detached full non-performance suite**: at ``04dbe60`` in a
  clean detached worktree: **5,185 collected / 3 performance deselected /
  5,182 selected; 5,168 passed / 14 skipped / 0 failures**, across eight
  file-disjoint fresh processes (649 / 648 / 648 / 648 / 648 / 648 /
  633+14sk / 646), every partition exiting 0.  The partition union is
  proven complete and non-overlapping against the collected file list,
  and no libomp resource event occurred.

### 41.7 Checkpoint disposition

Four pre-existing public-conversion defects are closed, one coherent
compiler family is migrated at byte parity in both automatic arms with a
production caller, and the compatibility envelope is censused with exact
arm-invariant codes.  Release behaviour is unchanged: the schedule audit
is equal to its retained baseline, every sealed capture surface is
byte-identical (modulo two process-dependent cache-key characters), and
static findings are at exact base/candidate parity.

**Phase-7 does not exit on this milestone.**  Two declared families
remain, each with a precise blocker recorded in §41.3-§41.4: the
compressed-parent/dense-leaf co-operands (blocked on the assembly
target's leaf envelope, and byte-parity gateable when unblocked), and
the multi-compressed reduction/TTM family (blocked on an unusable legacy
comparand that segfaults, so permanently oracle-gated).  Multiple dense
prefixes remain rejected as well.  No Phase-8 inventory was started, no
cutover, cache, selector or dispatch change was made, and no legacy code
was deleted.
