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
