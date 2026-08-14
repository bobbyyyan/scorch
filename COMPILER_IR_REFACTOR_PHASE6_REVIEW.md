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

## 0. What the words in this document mean (added 2026-08-13)

Sections 1 through 60 were written across many sessions and grew a private
vocabulary along the way.  A rewording pass on 2026-08-13 rewrote some of those
sections in plain English and deliberately left others alone.  This section is
the dictionary for the parts left alone, and for the file and key names on disk,
which keep the old words permanently because renaming them would break the
record they are part of.

That pass changed prose only.  It changed no number, no claim, no code
identifier, no test name and no cross-reference, and it fixed nothing: where a
sentence looked wrong it was left wrong and reported to Bobby separately.  These
documents are a dated record of what was measured, and a rewording pass that
quietly became a content revision would destroy the only thing they are for.

### 0.1 The vocabulary

| word | what it means |
| --- | --- |
| gate | a check a milestone has to pass, run once and reported with its numbers |
| green | every check in a named set passed |
| release neutrality | proof that a change does not alter the C++ that ships to users: a fixed set of programs is compiled before and after the change and the generated source compared byte for byte |
| the frontier | the survey matrix of programs the compiler is measured over, each compiled in both automatic scheduler settings, recording what the new pipeline accepts and what it refuses and with which error.  The sealed receipt holds 1,139 records over 1,138 distinct programs; §60.10 explains the one duplicate.  §52.5 first ran it at that size, and §55.2 ran the same matrix against two pinned trees |
| cell | one program of that matrix — an operand format combination, an einsum expression, and a result format (see 0.2) |
| cell-arm | one cell compiled under one of the two automatic scheduler settings, so a count in cell-arms is up to twice the same count in cells |
| arm, arm 0, arm 1 | the two automatic scheduler settings every frontier and census run covers.  Arm 0 is register blocking off and arm 1 is register blocking on — ``CompileOptions.with_regblock_enabled(False)`` and ``(True)``, the two halves of the production ``regblock_dual`` path |
| arm-invariant | produces the same result in both of those settings |
| the declared 748 | the original 748-cell subset of the matrix, the one the earlier milestones measured and quoted their numbers over |
| the extension | the 391 cells added to the matrix later, across five axes the record names; 748 plus 391 is the 1139 the later sections use |
| receiver | the result tensor a program assembles into.  A "dense receiver" is one whose result format is all dense, which needs no assembly and no workspace drain; a "sparse receiver" has at least one compressed result level, which is where the new pipeline's own assembly code lives |
| seam | a boundary rule that refuses a whole class of programs, and by extension the code site that raises the refusal |
| seam lock | a test asserting that a named program is still refused, at a named error code |
| ledger | an evidence folder under ``~/.cache/scorch-codex/`` (0.3 lists them) |
| sealed | written into a ledger together with a ``SHA256SUMS`` file, so a later session can prove the file it reads is the file that was written |
| receipt | one JSON file of raw per-record results inside a ledger.  Sections quote them, and where a section and its receipt disagree the receipt is the record |
| typed route | the new compiler pipeline — normalized CIN, then ``LoopPlan``, then LoopIR, then LLIR — entered through ``compile_cin_via_loopir`` and ``execute_cin_via_loopir`` |
| legacy | the old pipeline, the one that ships today |
| comparand | whatever a measurement compares the typed route against.  Which one is chosen matters: legacy under an empty ``Schedule()``, legacy under no requested schedule at all, and production's own ``scorch.einsum`` entry are three different comparands that disagree, and §§55.5, 56.5, 57.2 and 57.3 are largely about that |
| census | a compile-only sweep over many cells recording an outcome per cell rather than timing anything |
| quadrant A/B/C/D | the four combinations of "the new pipeline accepts or refuses" against "the old pipeline emits or refuses", tabulated in §55.5 |
| probe | a throwaway experiment, often in a scratch worktree with a sentinel error code installed, run to measure how far a rule reaches |
| battery | the dedicated test file that proves one migrated family end to end |
| corpus, grid | fixed sets of programs reused across milestones so numbers stay comparable: the 20-source corpus, the 42-source ``ss@dd`` grid, the 86-case schedule audit, the production emission corpus |
| lean sweep, heavy sweep | a narrower or wider grid of measurement configurations.  Twice on this branch a lean sweep missed configuration-dependent behaviour that the heavy version found (§52.8 corrected by §55.4, and §57.4's own first draft) |
| A/A control | the same binary measured against itself on the same grid, which is how the noise floor is established.  Every performance claim on this branch is judged against its own A/A control rather than a fixed ratio |
| oracle | the LoopIR interpreter, used as an independent semantic reference for what a program should compute |
| differential | a test that runs two routes on real tensors and compares their results |
| erasure | ``erase_schedule``, which undoes an applied schedule; a family is expected to erase back to the program it started from |
| envelope | the set of program families a milestone claims to have migrated and proven, as against the ones it merely admits |
| fail closed | refuse with a structured, named error instead of continuing.  Standard vocabulary, kept |
| GO, NO-GO | the verdict of a phase-exit audit, which is six named questions here |
| the sacrifice list | measurements a session declared it was not going to run, with its reasons; the frontier extension sat on it for three milestones |
| carried item | an owed measurement handed from one session to the next, lettered (a), (b) |
| attack matrix | a list of deliberate tampering attempts, each one a committed test that must fail closed and must also prove its tamper landed |
| the seal | §50.2, where the ordered-workspace completion contract was fixed and recorded |
| protected tracked files | five files a session must leave alone; their hashes are checked against ``statics/protected-hashes.txt`` before and after |
| strangler migration | building the new pipeline beside the old one and moving families over one at a time, rather than switching everything at once |

### 0.2 How a cell is named, and what the three blockers are

A cell name is the program, spelled the way the harnesses spell it.
``ss ij->j [s]`` is a rank-2 operand compressed in both modes, reduced over
``i``, into a result whose one mode is compressed.  ``d`` is a dense level and
``s`` a compressed one, so ``ddd ijk->ik [dd]`` is an all-dense rank-3 operand
into an all-dense rank-2 result.  Multi-operand programs are written out:
``TTM dss x ss -> dss`` is a tensor-times-matrix contraction, ``MM ds x ds ->
ds`` a matrix multiply, ``3-factor sss x d x d -> s`` a three-operand
contraction, and ``ss+ss ij->i [s]`` and ``ss*ss ij->i [s]`` are a union and an
intersection of two operands.  Family names (``rank3``, ``rank4``,
``degenerate2``, ``matmul``, ``ttm``, ``union2``, ``nonadd-combiner``) are the
harness's own grouping and appear as the ``family`` field of every receipt.

Blockers 1, 2 and 3 are referred to by number hundreds of times.  They are the
three named obstacles §49.5 left standing:

1. **Blocker 1** — a program whose automatic plan carries both a sparse
   accumulation workspace and an affine tile.  No replay contract exists for
   that combination, so it stops at ``unsupported_schedule_auto_family``.  Its
   family is ``TTM * -> dds``.  §52.7 closed it by decision; §55.3 measured its
   real size; §57.4 and §57.5 reopened and fixed it in the layer where the tile
   was illegal.
2. **Blocker 2** — a reduction whose ordered workspace key is empty (``K == 0``,
   scalar accumulation), which the representation had no node for.  Blocked at
   ``sparse_parent_dominance`` through the automatic reorder.  Built in §53.
3. **Blocker 3** — a dense result prefix level bound by a stored loop, which
   needs the row-scope catch-up against a parent count only known at run time.
   Refused up front with ``unsupported_sparse_output_domain``.  Built in §54.

### 0.3 The words that live on disk and cannot be renamed

Evidence folders and the keys inside their receipts keep the old vocabulary
permanently: a section can be reworded, a sealed file cannot.  Ledgers sit under
``~/.cache/scorch-codex/``; the ones the recent sections read are
``phase8-census-frontier-ext/`` (the frontier extension, the census, the heavy
legacy sweep, the suite and latency runs), ``crosshost-phase8-census/`` (the
same three censuses re-run on ``redwood`` and ``mkt1``),
``orderedkey-completion-seal/``, ``orderedkey-abi-signature-window/``,
``orderedkey-postassembly-window/``, ``boundprefix-blocker2/``,
``kernelperf-step0/`` (the first kernel-runtime harness),
``blocker1-legacy-soundness/``, ``blocker1-tilefix/``,
``compressed-prefix-reach/``, ``dense-domain-semantics/``,
``ttm-density-mechanism/``, ``ttm-parallel-singlepass/`` and
``assembly-strategy/``.  Every harness in them takes a tree root or ledger root
as ``$1``.

The recorded outcome of one cell is its ``route``, and the words in that field
are the ones the tables use:

- ``ADMITTED`` — the typed route compiled the program;
- ``EMITS`` — legacy produced generated source for it;
- a refusal, recorded as its exception class plus a defect code, for example
  ``LoopIRTargetError/unsupported_assembly_host``;
- ``unclassified`` — a refusal carrying no structured code, which is the thing
  the fail-closed gate counts and requires to be zero.

Receipt field names worth knowing before reading a table: ``family`` and
``name`` identify the cell; ``arm0``/``arm1`` and ``arm`` carry the scheduler
setting; ``arm_invariant``, ``kind`` (``defect_code``, ``loop_plan_diagnostic``,
``unclassified``) and ``carries_defect_attr`` carry the disposition;
``typed_digest``, ``typed_chars``, ``legacy_default``, ``legacy_default_same``,
``legacy_empty`` and ``legacy_empty_same`` carry the byte-equivalence
comparison; ``loopir_failed`` and ``loopir_ref_mismatch`` carry the heavy
sweep's correctness result; ``storage_agrees``, ``ref_match``, ``ratios``,
``aa``, ``aa_min``, ``aa_max`` and ``nthreads`` carry the runtime grids.  In the
test suite's own 20-name matrix the labels are ``MIGRATED``,
``AUTO_TILE_BLOCKED`` and their siblings.

### 0.4 What the rewording pass rewrote, and what it left as it is

Left in the original vocabulary on purpose, and covered by the table above
instead:

- the superseded sections of ``COMPILER_IR_REFACTOR_HANDOFF.md``.  That file is
  an append-only log; its older entries are historical record and the tail is
  current status.
- ``HANDOFF.md``, the older and separate file.
- ``COMPILER_IR_REFACTOR_PHASE3_5_REVIEW.md``,
  ``COMPILER_IR_REFACTOR_PHASE4_REVIEW.md`` and
  ``COMPILER_IR_REFACTOR_PHASE5_REVIEW.md``, all closed.

Rewriting closed history is not worth the risk of disturbing a number.
``NEXT_SESSION_PROMPT.md`` was stale and superseded, and was deleted rather than
reworded.

What the pass did rewrite, in this order: ``COMPILER_IR_REFACTOR_DESIGN.md`` in
full; this file from §60 backwards toward §1; and the current tail of
``COMPILER_IR_REFACTOR_HANDOFF.md``.  Each step is its own commit, so
``git log --oneline -- COMPILER_IR_REFACTOR_PHASE6_REVIEW.md`` shows how far it
got, and any section still written in the old vocabulary is one the pass did not
reach.  The check that it changed nothing but words is
``docs-plain-english/harness/doc_invariants.py``, which extracts every number,
code span, identifier, section reference and table shape from a document and
compares the sets before and after; its captures and their comparison output are
sealed beside it.

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
of losing the run.  **54 cells; every cell arm-invariant.**  The original
``55`` report counted the public-reachability heading as though it were a
cell; the four groups contain 12 + 14 + 13 + 15 cells.

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
  identical supports, and dense fixtures containing zeros) x two dtypes x
  five admitted operations x both arms — all matching.  These fixtures did
  not contain hand-built *stored* zeros; §42 adds that missing lock;
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
  5 failures were exactly the obsolete rank-1 seam assertions).  That
  eight-file run is an exploratory red receipt, not a final gate.  The
  reported conversion-adjacent **214 passed** result has no retained command
  or file membership and is therefore not independently reproducible.
- **Schedule audit** at the new tip: **46 admitted / 40 rejected / 0
  non-identical**, and its JSON is **equal to the retained baseline after
  removing only the commit field**.
- **Capture surfaces**: corpus **20/20**, grid **42/42**, anchors
  **22/22**, heap **11/11** byte-identical to the retained files; the
  automatic surface's every C++/CIN artifact is identical, with only the
  same **two process-dependent cache-key characters** differing in
  ``report.json``.
- **Static parity**, base ``a606e11`` versus candidate: the retained Black
  and Flake8 logs cover only the candidate and do not establish the claimed
  base/candidate parity; their one-file/nine-finding counts came from a
  narrower invocation.  Full-source mypy is **140 errors in 11 files at
  both**, exactly equal after line normalization, with zero LoopIR findings.
  §42 replaces all three static receipts with one documented full-tree
  methodology.  ``git diff --check`` was clean.
- **Census v10**: 54 cells, zero arm divergence.

- **Repeated compiled public differentials**: eight public cells
  (``einsum`` rank-1 copy/product/mixed, ``ss`` copy, ``ss*dd``,
  ``ss*sd`` built by the widened conversion, ``dss`` copy, and
  ``matmul`` over ``ss``x``ss``) match the dense reference and are
  byte-stable across three rounds.
- **Activating paired compile latency**: 200 warmups and
  2,000 interleaved samples per cell, in both orderings, over the three
  newly activating rank-1 cells plus the shared ``ss`` intersection
  control.  Every metric is inside the 1.10 budget: the worst
  within-run LoopIR/legacy ratio is **1.04189** in the primary ordering
  and **1.06632** in the order-flipped control.  The rank-1 union cell
  runs at 0.988-0.994 (faster than legacy), and the shared control reads
  1.021-1.030, matching the 1.018-1.026 range the inherited session
  recorded for it.  These are same-tip LoopIR-versus-legacy runs in two
  orders, not a cross-revision A/B/A or self-A/A experiment.
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

## 42. Phase-7 compatibility-envelope rigorous review corrections (2026-08-07)

### 42.1 Review result and evidence corrections

The inherited tip was ``2b36b7b``. Seven local commits were added without
amending, reordering, or pushing:

- ``d2e45c5`` / ``2cef248`` — correct public sparse-format conversion;
- ``efa78fe`` — complete the rank-1 assembly semantic evidence;
- ``29d13b8`` / ``67216cc`` — validate and detach caller-owned formats;
- ``f216e31`` / ``c806db4`` — reject hostile stored-field keys.

The inherited rank-1 LoopIR lowering remains byte-sound, but its runtime and
evidence claims needed correction:

- the compatibility census contains **54**, not 55, cells: 12 + 14 + 13 +
  15, with zero arm divergence;
- the retained **214 passed** line has neither a command nor file membership
  and is not independently reproducible, so it is not an authoritative gate;
- the retained 758-pass/5-failure pre-seam run is an exploratory red receipt:
  all five failures are the obsolete rank-1 seam assertions moved by the
  milestone, not a final green battery;
- the inherited latency evidence is a same-tip LoopIR-versus-legacy paired
  comparison in two orders, not cross-revision A/B/A or self-A/A evidence;
- full-tree base/candidate parity is **15 Black findings**, **47 Flake8
  findings**, and **140 mypy errors in 11 files**. Black and Flake8 are not
  clean: their unchanged findings are inherited, while mypy is equal after
  line-number normalization.

The schedule audit remains **46 admitted / 40 rejected / 0 non-identical**.
Corpus 20, grid 42, anchors 22, and heap 11 remain byte-identical; every
automatic C++/CIN artifact is identical, with only the known two
process-dependent cache-key characters differing in its report.

### 42.2 Runtime corrections and completed evidence

Fresh probes found three runtime boundary defects and one evidence gap:

1. Rank-1 ``to_sparse`` discarded the exact requested format metadata and
   always built COMPRESSED storage. ``d2e45c5`` preserves the requested
   ``TensorFormat`` and bit width, admits both COMPRESSED and COORDINATE
   rank-1 levels, rejects DENSE/SINGLETON requests precisely, and avoids
   routing already-dense vectors through an unnecessary dense clone.
2. The direct block materializer unnecessarily rejected nonidentity
   ``mode_order`` even though its physical-order representation already
   handled permutations correctly. ``d2e45c5`` admits those layouts;
   ``2cef248`` locks dense and sparse-source round trips, exact metadata, all
   supported value dtypes, and the dense fast path.
3. The public boundary could retain caller-owned ``TensorFormat`` and nested
   ``LevelFormat`` objects. Later ``object.__setattr__`` mutation could
   desynchronize the declared layout from its index arrays. ``29d13b8``
   validates exact stored fields, level kinds, containers, and signed-int64
   bit widths, then rebuilds a deeply owned snapshot for both rank-1 and
   rank-2-plus routes. ``67216cc`` locks detachment, hostile subclasses,
   malformed state, invalid widths, and failure atomicity.
4. That first ownership check still compared dictionary keys before proving
   they were exact strings, allowing a hostile ``str`` subclass to execute
   overloaded equality and leak an arbitrary exception. ``f216e31``
   validates key types first; ``c806db4`` locks both outer-format and
   nested-level attacks.

The inherited rank-1 battery also lacked direct oracle coverage, genuinely
stored-zero operands, and zero-extent execution. ``efa78fe`` adds all three:
every admitted family runs through the production LoopIR oracle; copy,
intersection, and union consume and retain hand-built stored zeros; and
zero-extent union is byte-identical and canonical in both automatic arms.
The file now collects **103 tests**.

No LoopIR node, canonical/request/schedule identity, LoopIR pipeline route, or
valid generated C++ changed in these corrections. The public runtime
``to_sparse`` conversion route deliberately changed for the corrected formats.

### 42.3 Deferred compatibility seam

Input-format ownership is now closed, but a returned ``STensor`` still exposes
its retained format through ``tensor.format``. A caller using
``object.__setattr__`` on that returned ``TensorFormat`` or a nested
``LevelFormat`` can still corrupt the tensor's own metadata after construction.
This is pre-existing and repository-wide rather than specific to
``to_sparse``. It remains deferred pending an audit of every public format
exposure and every internal identity consumer; it should not be patched only
at this one conversion call site.

### 42.4 Verification and disposition

Evidence is retained under
``~/.cache/scorch-codex/phase7-envelope-review-efa78fe/``. Full-tree static
parity between base ``a606e11`` and candidate ``c806db4``, the 54-cell census,
schedule audit, captures, and paired latency are green under the corrected
interpretations above. The final focused files collect **111**
public-conversion tests and **103** rank-1 assembly tests; the adjacent seam
membership contains **181**. All are included in the exact-tip suite below.
The exact-tip same-revision LoopIR-versus-legacy latency rerun uses 200 warmups
and 2,000 interleaved samples per cell in both orders; its worst ratio is
**1.06301** (rank-1 copy p95), inside the 1.10 target. This is paired two-order
evidence, not A/B/A or A/A.

The exact-tip clean detached suite at ``c806db4`` collected **5,224**, with
3 performance tests deselected and **5,221 selected**. Eight file-disjoint
fresh processes passed **5,207**, skipped **14**, and failed **0** (637 + 14
skipped / 654 / 653 / 653 / 651 / 653 / 653 / 653). The partition union is
proven complete and non-overlapping over 79 files; every process imported from
the detached tree, every partition exited 0, and no libomp resource event
occurred.

The five protected tracked files retain their recorded hashes; live and local
origin remain ``58e8565``; nothing was pushed and unrelated material was not
staged.

**Phase 7 remains open.** The corrections make the rank-1 milestone honest
but do not migrate compressed-parent/dense-leaf co-operands,
multi-compressed reduction/TTM, or multiple-dense-prefix outputs. No Phase-8
inventory, cutover, LoopIR release-dispatch, cache/selector change, or legacy
deletion was performed.

## 43. Phase-7 closure milestone: two silent-correctness fixes, the format-ownership boundary, and the dense-leaf co-operand family (2026-08-07/08)

### 43.1 Independent review of the inherited range

The inherited tip was ``9ca2212``.  ``2b36b7b..9ca2212`` -- ``d2e45c5``,
``2cef248``, ``efa78fe``, ``29d13b8``, ``67216cc``, ``f216e31``, ``c806db4``
and the documentation tip -- was reviewed before any new work, with every
reproducible claim re-derived from the contracts rather than from the
committed tests.  Origin remains ``58e8565``; the five protected tracked files
hash exactly as recorded.  The replacement ledger
``~/.cache/scorch-codex/phase7-envelope-review-efa78fe/`` verifies
**262/262** entries under ``shasum -a 256 -c SHA256SUMS``.

**No defect was found inside the reviewed range.**  Every runtime contract it
introduced reproduces:

- exact requested rank-1 ``TensorFormat`` and per-level ``bit_width``
  preservation, over widths ``None``/1/8/16/24/32/63/64/2**63-1, from both
  dense and already-sparse receivers, surviving ``copy()`` and serialization;
- rank-1 ``COMPRESSED`` versus ``COORDINATE`` assembly producing exactly
  ``[pos, crd]`` and ``[crd]`` respectively, over an 800-case randomized
  differential against a ``torch.nonzero`` oracle with zero failures;
- precise rejection of ``DENSE``/``SINGLETON`` rank-1 requests and of
  format-rank mismatches, with the receiver unmutated on every path;
- the already-dense fast path, including the removal of its trailing
  ``reshape(-1)``: no reachable rank-1 dense ``STensor`` can carry a non-flat,
  non-contiguous or wrong-length ``values``, because ``SparseStorage``
  rejects ``value.dim() != 1`` and non-contiguous values and the all-dense
  validator forces ``numel() == prod(physical_shape)``.  The path does not
  alias the caller's buffer;
- nonidentity ``mode_order`` round trips through the direct block
  materializer, and deep detachment of caller-owned ``TensorFormat`` and
  ``LevelFormat`` state including hostile ``str``-subclass keys in both the
  outer and nested dictionaries.

Two observations are recorded rather than treated as range defects, because
both are byte-identical at the base revision: ``to_sparse`` parses ``fmt``
twice and routes on the first parse; and a ``bit_width`` at or above 2**63 is
rejected at the public boundary although the ``LevelFormat`` constructor
accepts it, which is the pre-existing repository-wide signed-int64 invariant
applied uniformly.

### 43.2 Two silent-correctness defects found by fresh probes, fixed first

Both are pre-existing, publicly reachable, and outside the reviewed range.
Both are fixed before the migration is extended.

**(a) ``3bc5c07`` -- ``STensor.__add__`` returned silently wrong sums, and
crashed on an empty sparse operand.**  A lattice point selected its case on a
bare ``coord == index`` equality.  Under a *dense universe* -- any elementwise
expression in which some operand's level at that index is DENSE -- the emitted
loop is a plain counted ``for`` over the dense extent and never exhausts the
*sparse* operands' cursors.  Once a segment drained, the cursor still
addressed the next segment's first stored coordinate, so whenever that
coordinate equalled the current index the next row's value was folded into
this row.

A five-seed x three-shape sweep over every supported rank-2 format pair
produced **19 wrong results**, with errors up to 2.3 absolute.  The affected
pairs are ``ds + dd`` (canonical CSR plus a dense matrix), ``ds + sd``,
``ss + dd``, ``ss + sd`` and their commuted forms -- ordinary user code, with
no exception and no warning.  Separately, an all-zero sparse receiver added to
a dense operand **terminated the process with SIGSEGV**, because the very
first coordinate load addressed an empty array.

Each lattice-point case is now conjoined with its own cursor bound
(``pX < pX_end``) before the coordinate comparison, and under a dense universe
the coordinate load itself is selected on the same bound.  The value chosen
when a cursor is spent is never observed, because every consuming case carries
the same guard.  After the fix the same sweep produces **zero** wrong results
and the crash is gone.

**(b) ``2f367a4`` -- every reduction into a sparse rank-1 result emitted C++
that did not compile.**  ``coo_workspace_1d`` dereferences to
``std::pair<int64_t, T>``: its key IS the coordinate.  Every other workspace is
``coo_workspace<T, N>``, whose key is a ``std::vector<int64_t>``.  The result
drain subscripted the key unconditionally, so
``scorch_vector_set(T0_crd_vec, pT, it.first[0])`` failed at build time with
*subscripted value is not an array, pointer, or vector* -- the non-compiling
rank-1 reduction characterized in §41.3.  A third reader of the same key in
the same file already spelled the rule correctly; the two drain readers now
share it, keyed on the workspace class the lowering actually selected.

``scorch.einsum('ij->i')`` and ``('ij->j')`` over ``ss`` and ``ds`` receivers
now execute and match the dense reference.  This exposes a **second, distinct**
pre-existing defect that the compile failure had masked: reducing *two*
indices into a rank-1 vector (``ijk->i``, ``ijk->j``) assembles malformed
storage, because the workspace is rebuilt inside the second contraction loop
and the drain emits one entry per (surviving, outer contracted) pair.  Storage
validation rejects that result rather than returning it.  It is characterized
by an explicit test, not hidden.

**Blast radius, measured.**  Both fixes are confined to the legacy route.  Over
a 1,584-cell ``d``/``s`` layout enumeration at ranks 1-4 in both automatic
arms, every LoopIR-admitted program is byte-identical: **0 regressions, 0
newly rejected, 0 fail-closed code changes, 0 arm divergence**.  Legacy
generated source changes for **506 cells**, and those cells have **zero
overlap** with the 186 arm-instances that carry a byte-parity gate -- every one
of them is rejected by the LoopIR route.  All four sealed capture surfaces
regenerate byte-identically: corpus **20/20**, grid **42/42**, anchors
**22/22**, heap **11/11**.

### 43.3 The systemic returned-format mutability seam (``2626a04`` / ``221ff31``)

§42.3 deferred this pending an audit of every public format exposure and every
identity consumer.  That audit was performed across five independent surfaces
and judged from three independent stances.  It found the seam **wider and
worse** than §42.3 described.

Wider: ``tensor.format`` is one of five retained-object exposures.
``STensor.layout``, ``STensor.metadata`` and ``STensor.storage`` also return
the retained instance, and ``TensorLayout`` is strictly more dangerous than
``TensorFormat`` -- forging ``permutation`` yields a silently transposed
``matmul`` result with no exception and no format object involved at all.

Worse: the damage was **process-global**.  The prebuilt matmul resolution memo
handed its own cached ``TensorFormat`` straight through to result tensors, so
one ``object.__setattr__`` on a *returned result* rewrote that memo and changed
every later ``matmul(..., format='ds')`` in the process -- either raising from
an unrelated call site or returning a bare ``torch.Tensor`` where an
``STensor`` was contracted.

The audit also settled what must *not* be done.  Copy-on-read at
``STensor.format`` alone measured **+10% to +18%** on warm CSR-times-dense
matmul against a +/-1% same-shape control, and would still not close the seam,
because the hostile object enters on the write side.  Making ``parse_format``
always rebuild measured **+7.6%** on the same shape.  Native memory safety is
already closed independently: every prebuilt entry point re-validates through
``csrc/native_abi.h``, and three separate attempted memory-corruption forges
were all rejected with clean errors rather than a crash.

The boundary is therefore installed **on the way in**, at the two sites that
retain a format: ``TensorLayout.__post_init__`` (a tensor's single
authoritative format holder) and ``TensorIndex.__init__``.
``TensorIndex._from_layout`` needs no rebuild, because the layout already owns
what it hands over.  ``format.audit_format_state`` performs the structural
audit -- exact stored fields, key types proven exact ``str`` before any
comparison or hashing, exact ``LevelType``, positive signed-int64 bit widths --
and ``format.owned_format`` rebuilds both container layers.  ``to_sparse``'s
``_owned_sparse_format`` now delegates to the shared audit while keeping the
precise public error messages its locked tests require.  The boundary is
fail-open: anything not provably structurally exact is returned unchanged, so
no argument construction accepted before becomes an error.

Cost, measured as interleaved three-round warm-matmul medians against base:
**0.97-1.01** on csr@dense 64, csr@dense 256 and csr@csr 128 -- neutral within
this machine's noise.  A warm dense-output matmul constructs no
``TensorLayout``; a sparse-output one constructs exactly one.

**What remains open is stated, not hidden.**  Reads are still undefended: a
caller can forge a returned tensor's own retained value objects and
desynchronize that tensor's declared layout from its index arrays.  The damage
is now confined to that tensor -- it no longer escapes into a process-global
memo or into an unrelated tensor built from the same caller value.  Closing it
needs structurally unforgeable value types (a change to ``LevelFormat``,
``TensorFormat``, ``TensorLayout`` and ``TensorMetadata`` covering equality,
hashing, pickling and the dataclass surface), which is not attempted here.
Three tests are labelled CHARACTERIZATION LOCK and record exactly that.

### 43.4 The compressed-parent/dense-leaf co-operand migration (``fda8ef6`` / ``4712f4f``)

An operand whose value-bearing leaf is a DENSE level below compressed
structure -- ``sd``, ``ssd``, ``sdd``, ``dsd``, ``sddd``, ``ssdd``, ``sssd``
and the rank-general forms -- is read through ``PositionLoad`` over a
``DensePosition`` spine rather than through a merge cursor.  The blocker was
localized exactly: the ordered assembly target already built the correct loop
nest, the verifier already typed these programs, the oracle already ran them,
and the shared access machinery already validated the spine and recorded its
level drivers.  Only ``_merged_case_value`` -- the per-alignment-case leaf
evaluation -- refused the node kind, so every ``ss*sd``-shaped chain failed
closed at ``unsupported_program_shape``.

A position load is case-invariant: it addresses the loaded tensor's own
validated dense spine, not a merge cursor.  Partially evaluating it for one
cursor-alignment case is therefore sound *provided the position it grounds at
is bound unconditionally*.  ``SparseFor``/``SparseWindowFor`` bindings and
INTERSECTION merges always bind; a UNION merge's binding is optional at a
one-sided coordinate, where the loaded tensor owns no value-bearing position at
all.  ``_require_unconditional_position_load`` enforces exactly that at the
owning target boundary, independently of the verifier's position typing
(``1a92f50``), which already refuses to type a UNION-bound position for a
position-load spine.  Both boundaries were demonstrated firing on a hand-built
program, with the INTERSECTION and cursor-read controls staying admitted.

No new node kinds, no canonical-schema change, no request- or
schedule-identity change, and no CSR shortcut, runtime-format sniffing,
rendered-name or regex routing, or operation-specific target hack: the
admission is expressed through the existing position and level identities
alone.

The legacy comparand is honest here, so the gate is the B1/B3 discipline:

- **byte parity** over the 1,584-cell layout enumeration: **51 newly admitted
  cells (102 arm-instances)**, every one byte-identical to
  ``legacy_generated_cpp`` in both arms, with 0 regressions, 0 newly rejected,
  0 code changes and 0 arm divergence;
- a **108-cell evidence sweep** in both arms adding compiled execution against
  the dense PyTorch reference, **byte-identical produced storage against an
  independently keyed legacy build**, honest identity-ordered result storage,
  and the production LoopIR oracle.  Fixtures cover random, ragged (an empty
  leading slice), all-empty and fully dense; layouts cover ranks 2-4, both
  operand orders, dense prefixes, interleaved dense levels and a 3-ary chain;
  coverage includes float32 and float64, 16 zero-extent cells and hand-built
  stored explicit zeros.  **103 of 103 non-boundary cells pass every check.**

**Recorded seam move.**  ``ss*sd*dd`` -- a dense-factor widening over a
dense-leaf co-operand -- moves from ``unsupported_program_shape`` into the
admitted family at byte parity, joining the 3-ary intersection the target
already carried.  ``ss+sd`` keeps ``unsupported_union_with_dense`` and ``sd``
copy keeps ``unsupported_sparse_output_domain``.

**Permuted compressed structure is not in this envelope.**  The five permuted
cells in the sweep are recorded as a characterized boundary, not as coverage:
``_validate_layouts`` admits permutation only for all-dense tensors, and a
dense-leaf co-operand does not make a permuted compressed layout lowerable.
The rejection is pre-existing and unchanged.

### 43.5 The two remaining clusters, with newly localized blockers

Neither is migrated.  Both blockers are now more precisely located than §41
recorded, with executable evidence.

**Multiple dense prefixes / interleaved dense levels.**  ``dds``, ``sds`` and
``ddss`` outputs stay rejected.  A prototype confirmed the representation is
*not* the blocker: the LoopIR oracle assembles a ``dds`` result correctly,
producing the required 21-element position array for a 4x5 dense prefix.  Two
concrete blockers were isolated instead.  First, ``_collect_assembly_chain``
admits at most one dense prefix loop and ``_child_stream_statements`` requires
a stream loop below every level, so a second dense prefix has no emission
path.  Second -- and newly found -- the **base ``_TargetLowering`` shares the
legacy assembler's one-dense-extent position-sizing defect**: relaxing only
the CIN classifier let ``dds+dds`` compile through it and produce a 5-element
position array where 21 were required, the exact failure §41.3 attributes to
the legacy output assembler.  The generalization therefore requires the dense
catch-up counter and the result position-vector sizing to be driven by the
flattened dense-prefix index, not by a single loop variable.  That is a
coherent slice of its own and is not attempted here; the classifier is
deliberately left unchanged so the defective route stays unreachable.

**Multi-compressed reduction/TTM.**  Unchanged at
``unsupported_sparse_output``, ``unsupported_sparse_output_domain``,
``unsupported_sparse_output_reduction`` and ``unsupported_program_shape``
depending on the nest.  The legacy comparand still terminates with SIGSEGV, so
this family can never claim byte parity and needs its own oracle-gated
vertical with a workspace/result ownership audit.  §43.2(b) adds one concrete
datum: the workspace *placement* defect it exposed -- a workspace rebuilt
inside the second contraction loop -- is in the same machinery this slice
would have to own.

### 43.6 The regenerated compatibility census

Census v10 was re-run at the tip, one cell per subprocess, with the
numeric-soundness column repaired (§43.7).  **54 cells (12 + 14 + 13 + 15),
all 54 arm-invariant** on their arm-resolved LoopIR and legacy source columns.

- **The B group flips.**  ``ss*sd``, its commuted, f64, rank-3, rank-4, ragged
  and empty forms, the ``ss*ds``/``ss*dd`` controls and the 3-ary
  ``ss*sd*dd`` -- twelve of the fourteen B cells -- are now **admitted**, each
  executing to the dense reference with well-formed identity-ordered storage.
  ``ss+sd`` keeps ``unsupported_union_with_dense``; ``sd`` copy keeps
  ``unsupported_sparse_output_domain``.
- **Eight reduction/TTM cells still terminate with SIGSEGV (exit 139)** during
  legacy execution -- C1-C7 and C13 -- exactly the eight §41.4 recorded.  Their
  source columns are recorded by a separate compile-only pass, because the
  crash prevents the executing pass from writing its per-cell record at all.
- **The repaired numeric column is new evidence.**  Eight cells produce
  malformed legacy storage (``B11``, ``D5``-``D11``), and **four of them are
  also numerically wrong** against the dense reference: ``B11`` (``sd`` copy),
  ``D9`` (``sds`` copy), ``D10`` (``sd+sd``) and ``D11`` (``sdd`` copy).  The
  vacuous column had reported all of these as merely "executed".  This
  strengthens, rather than weakens, the decision to gate the multiple-dense-
  prefix and trailing-dense-output families on the oracle: their legacy
  comparand is not just malformed, it is wrong.

### 43.7 Evidence corrections carried forward

- The compatibility census's numeric-soundness column was **vacuous**: it
  called ``to_torch()`` on the natively built legacy result, which exposes no
  such method, so ``matches_torch`` was unset in every cell while the harness
  still reported the cell as "executed".  The harness now decodes the produced
  level storage directly, and the column is real.  The soundness claims that
  column was cited for were, until now, unverified.
- §42.3's "input-format ownership is now closed" was too strong: three further
  public input-side boundaries retained caller-owned formats.  §43.3 closes
  them.
- The "498 LoopIR / 123 runtime" memberships (§40.3), the "214 passed" line
  and the "758 passed" pre-seam receipt remain unreproducible or exploratory
  and are not gates.

### 43.8 Verification

All gates ran in the ``scorch`` conda environment; evidence is retained under
``~/.cache/scorch-codex/phase7-closure-session/``.

- **Focused batteries** at the tip: dense-universe cursor bounds **29
  passed**; sparse rank-1 reduction drains and the dense-leaf co-operand target
  together **202 passed**; the format-ownership boundary **17 passed**; the
  three inherited format/conversion files **147 passed**; the LLIR string
  budget **41 passed**.  Every one of these files is a member of the exact-tip
  suite below.
- **Exhaustive layout differential**: 1,584 cells x 2 automatic arms at ranks
  1-4, base ``9ca2212`` versus candidate.  **3,066 unchanged, 0 regressions, 0
  newly rejected, 0 fail-closed code changes, 0 arm divergence, 102
  newly-admitted arm-instances (51 cells) every one at byte parity with
  legacy.**  Legacy source drifts on 506 cells, with **zero overlap** with the
  186 base byte-parity arm-instances.
- **Family evidence sweep**: 108 cells, **103 of 103 non-boundary cells pass
  every check** (byte parity in both arms, execution against the dense PyTorch
  reference, honest identity-ordered storage, byte-identical storage against an
  independently keyed legacy build, and the production oracle).  The five
  permuted-compressed cells are recorded as a characterized boundary.
- **Deterministic census v10** at the tip: **54 cells (12 + 14 + 13 + 15), 54
  arm-invariant**; eight reduction/TTM cells terminate with SIGSEGV (exit 139)
  during legacy execution and are recorded by a separate compile-only pass.
- **Schedule audit**: **46 admitted / 40 rejected / 0 non-identical**, its JSON
  equal to the retained baseline after removing only the commit field.
- **Capture surfaces**: corpus **20/20**, grid **42/42**, anchors **22/22**,
  heap **11/11** byte-identical to the retained baselines.
- **Full-tree static parity**, base ``9ca2212`` versus candidate ``18d7a27``,
  one invocation each of ``black --check src tests``, ``flake8 src tests`` and
  ``mypy src``: Black **15 findings at both, identical file set**; Flake8 **47
  at both, identical after line normalization**; mypy **140 errors in 11 files
  at both**, whose only residual difference is a line number embedded in the
  message of a pre-existing ``cin_lowerer.py`` finding.  Black and Flake8 are
  not globally clean; their findings are inherited and unchanged.
  ``git diff --check`` is clean.
- **Exact-tip clean detached full non-performance suite: COMPLETE.**  **5,462
  collected / 3 performance deselected / 5,459 selected; 5,445 passed / 14
  skipped / 0 failed**, in eight file-disjoint fresh processes (684 / 684 /
  682 / 682 / 682 / 682 / 682 / 667+14sk), every partition exiting 0, with
  ``UNION-PROOF: complete and non-overlapping``.  The partition totals sum
  exactly to the selected node count.  Partition 0 ran at ``6436e82`` and
  partitions 1-7 at ``f45a7b1``; those revisions differ only by
  ``test_loopir_single_cursor_assembly_target.py``, a member of partition 1,
  which therefore ran at ``f45a7b1``.

  **This gate earned its keep.**  Partition 1 initially failed two nodes:
  ``test_adjacent_seams_stay_fail_closed[posload_co_operand-*]``.  That
  inherited seam lock named ``ss * sd`` -- exactly the cell this milestone
  migrates -- so it asserted ``unsupported_program_shape`` for a program now
  admitted at byte parity.  The 1,584-cell layout differential could not have
  caught it, because that harness compares compiler outcomes and not test
  expectations.  ``f45a7b1`` moves the lock to the neighbour that still
  occupies the seam and names the move in place; partition 1 then passed
  684/684.  The milestone's "every neighbour keeps its exact code" claim was
  incomplete by exactly one inherited lock until that commit.
- **Activating paired two-order compile latency was not run**, for the same
  reason: a thermally throttled host cannot produce an honest latency receipt.
  The harness is retained with the three newly activating dense-leaf cells
  (``dl_ss_sd_mul``, ``dl_sss_sdd_mul``, ``dl_ssss_sddd_mul``) plus the shared
  ``b3_ss_mul`` control already declared.
- **The five protected tracked files** retain their recorded SHA-256 values;
  live and local origin remain ``58e8565``; nothing was pushed, amended,
  squashed or reordered; only explicit paths were staged.

### 43.9 Phase-7 exit audit

Criterion by criterion.  *Migrated families complete over their proven
envelopes*: the compressed-parent/dense-leaf co-operand family is, at byte
parity in both automatic arms with oracle and PyTorch differentials and legacy
storage identity.  *Every neighbour carries a stable fail-closed code*: yes, in
both arms, with one recorded seam move (``ss*sd*dd``).  *Representation
unchanged*: no node kinds, canonical schema, request identity, schedule
identity or erasure changed.  *Release behaviour unchanged*: the schedule audit
equals its baseline, every sealed capture surface is byte-identical, and no
default dispatch, cache or selector changed.  *The declared matrix is closed*:
**it is not.**

Two declared families remain unmigrated, each with a precise blocker recorded
in §43.5: multiple dense prefixes / interleaved dense levels (blocked on the
dense catch-up counter and result position-vector sizing, and on the base
target's shared one-dense-extent sizing defect), and multi-compressed
reduction/TTM (blocked on a legacy comparand that segfaults, so permanently
oracle-gated).

One gate remains outstanding: the activating paired two-order latency receipt
could not be produced on a thermally throttled host, and is recorded as
outstanding rather than as a pass.  The full suite did complete.

**Phase 7 therefore does not exit on this milestone.**  No Phase-8 inventory
was started, no cutover, cache, selector or default-dispatch change was made,
and no legacy code was deleted.  The milestone's own contribution is
nonetheless larger than a family migration: two pre-existing silent-correctness
defects in public operations are closed, the process-global half of the format
seam is closed, and the census's numeric-soundness column -- which had never
actually run -- is now real.

## 44. Exact-tip review: retained ownership and target integrity (2026-08-08)

This review starts at inherited documentation tip ``8a2a83a`` and audits the
whole 25-commit inclusive span ``2b36b7b^..8a2a83a``.  The preceding report's
count of 16 is not a history error: those are the commits it added above
``9ca2212``; the total also includes the eight commits it reviewed above
``2b36b7b`` and that inclusive base checkpoint itself.  No commit was amended,
squashed or reordered.

### 44.1 Evidence audit and inherited verdict

The 540-entry ``phase7-closure-session`` ledger verifies **540/540** from
scratch.  The older §39 ``phase7-assembly-session`` ledger does not: its
``tip-at-docs.txt`` was changed about 25 seconds after ``SHA256SUMS`` was
written, so that manifest now verifies **224/225**.  The changed file only records a Git
tip and does not alter a code, test, capture or timing receipt; this is an
evidence-sealing defect, not a code-result discrepancy.  The failed check,
expected digest, actual digest and timestamps are retained in this review's
ledger instead of silently resealing the old directory.

The inherited family migration itself remains sound over its stated envelope.
Fresh review probes reproduced rank-1 and dense-suffix conversion, the
``PositionLoad`` UNION boundary, the 54-cell compatibility census and the
sealed capture surfaces.  The review nevertheless found ownership gaps outside
that envelope which could make retained runtime or compiler state diverge after
validation.  They are fixed before any further Phase-7 representation work.

### 44.2 Runtime ownership corrections (``554fdaf`` through ``9bc8c91``)

The construction-side format copy in §43 was necessary but incomplete.  A
caller could still cross a retaining boundary with subclassed or forged
formats, layouts, metadata, sparse-storage objects, tensor indices, index
tensors, shapes, permutations, names, dtypes or devices.  Several paths then
read properties from the caller object after validating different stored state,
or shared structural objects between two tensors.  A broad ``to_sparse`` error
handler also converted an invalid requested format into the default path, and
mixed coordinate/non-coordinate hierarchies could begin materialization before
being rejected.

The corrected boundary now:

- audits exact stored base fields without invoking subclass descriptors;
- canonicalizes integer and string subclasses, shapes, permutations, names,
  dtype/device values and both format-container layers;
- rebuilds layouts, metadata, indices and sparse-storage structure for each
  owner, cloning structural index tensors while preserving normal numeric
  value-buffer aliasing;
- preserves one canonical layout object shared by a tensor's own metadata and
  storage, including metadata setter paths;
- rejects malformed or foreign structural state through Scorch domain errors;
- makes ``to_sparse`` reject invalid requested formats and mixed coordinate
  hierarchies atomically instead of swallowing the request; and
- forms device/dtype diagnostics from safe type information, so a bad value's
  ``str``/``repr`` hook cannot execute while reporting the error.

The tests cover cross-owner mutation, tensor-subclass and ``__dict__``
descriptor interposition, malformed fields, index-tensor subclasses, hostile
scalar rendering, copy detachment and storage/layout identity.  The numeric
payload remains intentionally observable and mutable under the repository's
existing tensor semantics.

One compatibility seam remains explicit: ``STensor.format``, ``layout``,
``metadata`` and ``storage`` still return the tensor's own nominally frozen
Python value objects.  A caller using ``object.__setattr__`` can corrupt that
same tensor.  This review closes cross-owner and post-validation retention; it
does not claim structurally unforgeable public value types.

### 44.3 Compiler ownership corrections (``d067fd7`` through ``f034b69``)

Normalized CIN and verified LoopIR are also frozen only by convention.  Fresh
probes found that a caller could replace or mutate a retained statement,
access, identity, position spine, synthetic view, workspace branch, heap leaf,
target class or target-private cache after verification.  Several successful
emission paths then consumed the changed object without one common integrity
authority; a self-consistent ``PositionLoad`` retarget was the clearest escape.

The target now owns an exact, cycle-bounded signature of the complete program
graph and its constructor-final cache graph, strongly retaining every signed
object so an address cannot be recycled into the signature.  Exact stored node,
enum and identity classes are registry-locked; repeated occurrences, aliasing,
position-to-tensor/level/domain links, input membership, target type and every
specialized synthetic region are checked before raw emission.  Narrow
position/value/owner validators are replayed on the changed path so established
diagnostics remain stable.  The sparse-workspace and heap specializations bind
both semantic branches and their declarations rather than relying on the base
target's fields.

Adjacent correctness fixes discovered by those probes are included in the
same reviewed boundary:

- CIN normalization owns tensor formats, and the adjacent CIN-lowering
  preflight rejects mixed coordinate hierarchies before materialization;
- all-dense co-operands use logical ``LevelDecl.mode`` identity under a
  non-identity physical order;
- a position load must retain its exact access occurrence and validated dense
  spine; and
- rank-1 coordinate-result drains use the scalar workspace-key contract.

No LoopIR node, canonical schema, plan/request identity, automatic
LoopIR/legacy compiler-routing rule or valid generated C++ spelling changes in
this review.

### 44.4 The measured integrity-cost correction

The first fail-closed implementation at ``dbad96b`` was correct but too slow.
On an unloaded Redwood run (200 warmups / 2,000 samples, both orderings), the
rank-2 dense-leaf case and B3 control were repeatedly about **1.112-1.115** at
p50/p95 against the legacy compiler, over the declared 1.10 ceiling.  Profiling
attributed the crossing to rebuilding overlapping graph, binding, value, owner
and target-cache signatures on the successful path.

``0c09ea3`` makes the complete graph signature the unchanged-path proof and
replays narrow scans only after a graph difference.  The next optimization,
``2224d29``, initially classified every narrow snapshot as diagnostic-only;
review immediately disproved that assumption because successful merged
emission reads the ``PositionLoad`` map.  ``aa2a3ef`` restored that map.  A
second independent probe then showed the bound-position map is also read on a
successful path and that recursively signing caller-replaced witnesses could
execute callbacks.

``12be09c`` freezes the coherent four-witness family at construction: the two
mappings become fresh exact read-only proxies; the value and owner signatures
are recursively immutable tuples.  The target cache binds each strongly owned
witness by exact identity without calling ``len``, equality, iteration or
lookup on a replacement.  Dedicated serial and parallel sparse-workspace
targets own none of these witnesses, so the family is optional as a whole
rather than globally required.  Missing, partial, foreign, same-value
replacement and callback-bearing replacement state all fail closed; the
unchanged route does not repeat the narrow walks.

The final independent review did not stop there.  ``58fa714`` / ``730e59d``
move the construction authority outside the forgeable target instance and
cross-anchor its graph and cache snapshots to the retained instance fields.
``4913f46`` / ``2d9e93d`` make the same authority available during
construction without accepting a second seal or an incomplete pre-seal
record.  This closes the otherwise circular case in which rewriting both an
instance cache and the instance's own purported snapshot could bless the
rewrite.

The last review pass found two additional fail-closed gaps in the optimized
boundary itself.  First, ``type(x) is MappingProxyType`` did not prove that a
schema registry was canonical: a proxy can wrap a hostile ``Mapping``, and an
ordinary replacement proxy could omit a required node or enum without being
observed on some successful paths.  Second, exact ``NamedTuple`` type did not
prove tuple arity: ``tuple.__new__`` could manufacture a truncated authority
whose field descriptor leaked ``IndexError``.  ``c18abd6`` / ``bb76e26`` pin
the four exact canonical registry proxies in one tuple-immutable bundle before
*any* lookup, validate both schema and target authority arity before a field
descriptor, prove weak-reference type before invocation, and make retained
loop caches real tuples.  They also share one external authority lookup across
an unchanged emission and replay input validation only on the cold mismatch
path, retaining every narrower diagnostic.  Hostile proxy mappings, missing
entries, truncated and oversized authorities, equal fresh mirrors, weak-ref
lookalikes, cycles, malformed caches and ordinary graph mutations all fail
with controlled target diagnostics in the final adversarial review.

That boundary was correct, but the first exact unloaded-host latency rerun was
still marginal: the rank-2 dense-leaf and B3 paths were about 1.106 in both
fixed orders, while an alternating run was 1.097 at p50 but 1.104 at p95.
Profiling exposed one semantic duplicate rather than a reason to weaken the
guard: every merged alignment case rewalked all position bindings even though
the complete binding result was already an exact target witness.  ``80c9e18``
/ ``1a550d4`` freeze that result immediately in all three constructors and
reuse it only after the complete graph/cache check; changed paths still replay
the stored-state walk.  Independent probes replaced the live graph, installed
an equal fresh proxy, attempted proxy mutation and exercised both base and
multi-compressed merged routes.  Every mutation failed closed and unchanged
source stayed byte-identical.

The exact final Redwood alternating run at ``1a550d4`` (200 warmups /
2,000 samples) is inside the 1.10 p50/p95 target on all four cases.  The
dense-leaf rank-2/3/4 ratios are respectively **1.0886/1.0991**,
**1.0703/1.0748** and **1.0449/1.0452**; B3 is **1.0909/1.0961**.  B3's
untrimmed mean ratio is 1.1097 because of isolated tail samples, so it is
reported rather than silently averaged away; the declared gate is p50/p95.
An immediately adjacent same-tip alternating A/A control is 0.9994-1.0145 at
p50 and 0.9996-1.0065 at p95, attributing that tail behavior to the host rather
than a changed code path.  Every LoopIR source is byte-identical to its legacy
source in both runs; the exact rank-2/3/4/B3 SHA-256 values are retained with
the harnesses and raw JSON.  A retrospective provenance receipt proves all 60
tracked ``src/scorch`` Python files on Redwood byte-equal to the clean detached
``1a550d4`` tree and records the host, Python and exact Scorch import path.  The
original harness processes did not retain separate exit-code receipts; the
completed JSON, source proof and adjacent A/A output are retained without
claiming otherwise.

### 44.5 Verification

All commands use the ``scorch`` conda environment.  Final code/test evidence is
under ``~/.cache/scorch-codex/phase7-closure-review-final/``; exact Redwood
latency receipts and harnesses are copied into ``latency-final/`` and
remain in the corresponding Redwood evidence directory.

- Exact code-tip runtime ownership membership: **228 passed** over the six
  affected runtime and conversion files at detached ``b91b773`` with asserted
  import provenance.
- Exact production-tip compiler memberships at ``1a550d4``: **398 passed**
  over CIN analysis, dense-universe and dense-leaf target tests; **520 passed** over LLIR
  lowering, serial sparse-workspace, rank-1 drain, parallel-workspace and
  multi-compressed target tests.  Final ``b91b773`` changes only the disjoint
  schedule-generality test file.
- The first literal full-suite run at ``1a550d4`` correctly stopped in the
  pre-existing nested workspace-pair test: its ``doo`` result contradicted the
  deliberate mixed-coordinate rejection and, when that guard is bypassed,
  produces malformed assembly.  ``b91b773`` changes no production code.  It
  exercises the same two-coordinate workspace-key source contract with
  format-admitted ``ooo`` output and adds an explicit scheduled-``doo``
  rejection lock; both focused tests pass.  This is structural source coverage,
  not a claim of native rank-greater-than-one workspace correctness.  The exact
  final full suite below runs at that test tip.
- Schedule audit: **46 admitted / 40 rejected / 0 non-identical**.  Corpus
  **20/20**, grid **42/42**, anchor **22/22**, heap **11/11** and automatic
  **23/23** captures are raw byte-identical.  The 54-cell execution census was
  run at ``bb76e26`` and carries forward because ``80c9e18`` changes only
  successful-path integrity work while ``1a550d4`` and ``b91b773`` are
  test-only; the ``1a550d4`` capture set plus the test-only final diff prove
  unchanged production source.  It remains arm-invariant with every
  successful admitted arm byte-equal to legacy; the eight known unsafe legacy
  reduction/TTM cells remain executable failure evidence.
- Full-source base/candidate parity: **one inherited Black finding, nine
  inherited Flake8 findings and 140 mypy errors in 11 files** at both
  ``8a2a83a`` and ``1a550d4``; all three normalized logs are byte-identical,
  the production LoopIR mypy membership is clean, and ``b91b773`` changes no
  source file.
  ``git diff --check`` is clean.
- Exact code-tip clean detached non-performance suite: **5,632 passed / 14
  skipped / 0 failures / 2 known sparse-invariant warnings over all 5,646
  selected nodes** at ``b91b773`` in eight file-disjoint fresh processes
  (5,649 collected, 3 performance tests
  deselected).  A pre-run proof places all 85 tracked pytest modules exactly
  once, the collected-node union is complete and non-overlapping, every JUnit
  load matches its partition receipt, and no libomp ceiling event occurred.
- The five protected tracked files retain their recorded SHA-256 values;
  origin remains ``58e8565``; nothing was pushed; every unrelated tracked and
  untracked path remains untouched.

### 44.6 Phase-7 exit disposition

This review fixes ownership and integrity defects; it does not silently widen
the migrated format matrix.  The two §43.5 clusters remain:

1. multiple dense prefixes / interleaved dense output (``dds``, ``ddss`` and a
   separate ``sds`` decision), requiring a flattened dense-prefix counter and
   correctly sized position vectors; and
2. multi-compressed reduction/TTM, requiring an oracle-gated workspace/result
   vertical because the legacy comparand segfaults.

**Phase 7 therefore remains open.**  No Phase-8 cutover, default-dispatch,
selector, cache or fallback change was made, and no legacy implementation was
deleted.

## 45. Exact-tip review, the flattened dense prefix, and an honest cluster-2 NO-GO (2026-08-09)

This milestone starts at inherited documentation tip ``52d43cc`` and reviews
the whole 23-commit span ``8a2a83a..52d43cc`` before any new work.  It then
migrates one of the two remaining Phase-7 clusters end to end, and records a
NO-GO -- with a localized, independently verified root-cause map -- for the
other.  Origin remains ``58e8565``; nothing was pushed, amended, squashed or
reordered; the five protected tracked files hash exactly as recorded.

### 45.1 Independent review of the inherited range

Every contract was re-derived from the code and from raw artifacts, not from
the review or handoff prose, across five independent boundaries.  Each claimed
defect was then put to independent adversarial verifiers instructed to refute
it.  **No defect inside the reviewed range survived verification.**  What
reproduces:

- **Runtime ownership.**  The audit reads exact stored base fields without
  invoking a subclass descriptor: subclasses of ``TensorFormat``,
  ``LevelFormat``, ``TensorLayout``, ``SparseStorage`` and ``TensorIndex``
  carrying recording ``__dict__`` data descriptors, recording
  ``__getattribute__`` and recording property overrides drive every retaining
  boundary with a **total hook-invocation count of zero**.  Integer and string
  subclasses, shapes, permutations, names, dtypes, devices and both format
  container layers canonicalize.  Structural index tensors are cloned while
  the numeric value buffer stays aliased, and one canonical ``TensorLayout``
  object is shared by a tensor's own metadata and storage across the setter,
  ``copy``, ``clone``, ``deepcopy`` and pickle paths.
- **The hostile-scalar diagnostics.**  Thirty cases drive an object whose
  ``__str__``/``__repr__``/``__format__`` all raise and record, through
  metadata, spec, dtype, device, layout, index, name, shape, permutation and
  value boundaries.  Every one reports a ``scorch.exceptions`` error with
  **no rendering hook fired**.
- **The declared-open read seam is genuinely confined.**  Two tensors built
  from one caller format do not share it; corrupting a returned tensor's own
  format damages only that tensor; and the prebuilt matmul resolution memo is
  provably not reachable -- after ``matmul``, no result-side format object is
  the memo's object, and corrupting the result leaves later ``matmul`` calls
  returning the correct format and values.
- **LoopIR graph/cache integrity, external construction/final target
  authority, canonical registry identity, exact authority arity and the
  sealed position-binding reuse** all reproduce, including the strong
  retention that stops an address being recycled into a signature and the
  circularity closure that stops an instance blessing its own rewrite.
- **``b91b773``'s scheduled-``doo`` rejection and ``ooo`` pair-read lock**
  reproduce.  ``ooo`` remains **structural source coverage only**: its
  generated kernel drains with ``C2_crd.emplace_back(c); C1_crd.emplace_back(r);``
  and never appends to ``C0_crd``, and native execution raises
  ``workspace coordinate rank mismatch``.  The open rank-greater-than-one
  workspace coordinate-rank/shape and result-assembly limitation is preserved
  exactly, not narrowed.

### 45.2 Evidence audit, with every qualification preserved

- The ``phase7-closure-session`` ledger verifies **540/540** from scratch.
- The historical §39 ``phase7-assembly-session`` ledger verifies **224/225**
  and **was not resealed**.  The mechanism is now proven rather than inferred:
  the manifest digest for ``tip-at-docs.txt`` equals the SHA-256 of that
  file's *first line only*, and a second line was appended about 25 seconds
  after the manifest was written.  The failed entry, both digests and the
  timestamps are retained in this milestone's ledger.
- The 54-cell execution census carried forward from ``bb76e26`` is
  **disclosed and correct**: an independent re-run of all 54 cells at
  ``52d43cc``, one fresh subprocess per cell, reproduced the retained
  artifacts byte for byte -- 46/46 written cell files identical, identical
  ``arm-exact`` and ``compile-only`` digests, and an identical exit-code file
  including the same eight exit-139 cells.
- The historical latency harnesses still lack separate process-exit receipts;
  that is restated, not repaired.

Two evidence inaccuracies are recorded rather than resealed.
``target-membership-proof/summary.json`` carries a single ``revision`` key of
``b91b773`` beside the ``398 passed``/``520 passed`` compiler memberships,
which actually executed at ``1a550d4``; §44.5 states the memberships
correctly, so this is a metadata attribution error in one summary file.  And
§44.4's "share one external authority lookup across an unchanged emission and
replay input validation only on the cold mismatch path" is **not true for
merged position-load emission**: measured, that path performs two external
authority lookups and one successful-path input-validation replay, because
``_validated_position_load_spine`` opens with an unconditional
``_require_program_inputs_unchanged``.  The guard is sound; the cost claim was
too strong.

### 45.3 Two runtime ownership defects found by fresh probes, fixed first (``35094ba``)

Both are publicly reachable with ordinary arguments -- no ``object.__setattr__``,
no ``ctypes`` -- and both contradict a contract this boundary states.

**``audit_format_state`` rejected constructor-valid ``int``-subclass bit
widths.**  ``LevelFormat.__init__`` accepts any non-``bool`` ``int`` subclass,
so an ``IntEnum`` width constructs a valid format; the audit then demanded
``type(bit_width) is int`` and ``owned_format`` raised
``tensor format has malformed stored state``.  Every retaining boundary
therefore refused a format the constructor had just accepted, violating both
§43.3's "no argument construction accepted before becomes an error" and
§44.2's "canonicalizes integer and string subclasses".  The audit now mirrors
the constructor's own acceptance test and canonicalizes through
``int.__int__`` -- the same base-descriptor idiom ``layout._normalize_shape``
and ``_normalize_permutation`` already use -- so no subclass ``__int__`` runs.
``bool`` and non-positive widths still fail closed.

**``to_sparse`` did not reject a requested SINGLETON level above rank 1.**
``validate_runtime_contract`` already declares singleton levels unrunnable and
the rank-1 arm rejects them up front, but the rank>=2 arm ran the entire JIT
pipeline and then leaked a bare ``builtins.ValueError`` out of code generation
-- an invalid requested format that was *not* rejected atomically through a
Scorch domain error, contrary to §44.2.  It is now rejected beside the
mixed-hierarchy check, with the receiver provably unmutated at ranks 2 and 3.

A third probe claim -- that the format parser executes a hostile ``__repr__``
-- **did not reproduce** and is not treated as a defect: the parser rejects
foreign objects by exact type before rendering them, returning
``TensorTypeError`` with no hook fired.

### 45.4 The multiple-dense-prefix sparse-output migration (``78e4ca6..a1582a9``)

``dds``, ``ddss``, ``ddds``, ``dddss`` and the rank-general forms are migrated
end to end -- copy, intersection and ordered UNION assembly, in both automatic
arms.

The blocker was never the representation.  It was that a compressed level's
child-segment number was modelled as **one dense loop variable**.  With
several dense parents the segment number is the flattened dense coordinate
``(((i0 * E1) + i1) * E2 + i2) ...``, so a compressed level below a ``2x3``
prefix owns 6 segments and a 7-entry position vector, not 3 and 4.  Two
collaborating sites carried the single-extent assumption, and both are now
expressed through narrow overridable hooks whose base implementations are the
inherited spellings byte for byte:

- ``_assembly_catch_up`` asks ``_assembly_catch_up_bound``; the
  multi-compressed override folds the prefix into the flattened index using
  the **same bound spellings the emitted ``for`` statements use**, so the
  arithmetic cannot disagree with the iteration space.
- ``_lower_dense`` asks ``_dense_loop_owns_result_assembly``; the override
  gives catch-up and close ownership to the **innermost** dense parent, the
  only loop that completes a flattened cell.

``_exact_dense_parent_positions`` returns False for a prefix of two or more:
the pre-sized spelling names exactly one level's ``_size`` variable and cannot
express a product extent, which is precisely the defect that makes the legacy
assembler wrong here.  These results are built through the checked
``scorch_vector_set`` growth path.  The chain collector accepts one dense loop
per dense prefix level, still requiring every dense loop to precede every
stream loop; ``_child_stream_statements`` lets a dense prefix level nest the
next dense prefix loop.  Routing and CIN admission gain the same
``compressed_suffix == 1 and prefix >= 2`` disjunct.

No new node kinds, no canonical-schema change, no request- or
schedule-identity change, and no CSR shortcut, rendered-name discovery, regex
or format-string sniffing: the admission is expressed through level and domain
identities alone.

**The legacy comparand is dishonest for this family, and that is proven, not
asserted.**  Its catch-up is emitted as ``for (; C2_pos_index < j; C2_pos_index++)``
-- bounded by the innermost dense loop variable alone -- so for a ``2x3``
prefix it produces a 4-entry position array where 7 are required.  The
production LoopIR oracle, by contrast, already assembles the correct
``(0, 3, 4, 8, 11, 13, 16)``.  The gate is therefore the oracle plus the dense
PyTorch reference, never byte parity, and a dedicated test pins the legacy
malformation directly from its generated source so the choice cannot silently
decay.

**What "migrated" does and does not mean here, stated exactly.**  The LoopIR
route -- CIN admission, routing, target lowering, scheduling/oracle, erasure
and the produced level storage -- is correct and oracle-verified for this
family.  **Default public dispatch is unchanged**, and that is deliberate:
changing it is a Phase-8 cutover, which this milestone is forbidden to start.
So ``scorch.einsum('ijk,ijk->ijk', a, b)`` over ``dds`` operands still reaches
the legacy assembler and still raises
``TensorIndexError: compressed mode 2 position array has 5 elements, expected 13``
-- byte-for-byte the same failure as at base ``52d43cc``, verified by running
the identical probe against an isolated base tree.  The user-visible behaviour
is therefore unchanged: a validation error, never silent corruption.  Public
``to_sparse('dds')`` and ``to_sparse('ddss')`` do already produce correct
13- and 7-entry position vectors and round-trip exactly.

That failure message is itself independent corroboration of the model this
milestone implements: the runtime storage validator computes the required
position length as the product of the dense extents plus one -- 13 for a
``3x4`` prefix, 7 for a ``2x3`` prefix -- which is exactly what the LoopIR
oracle and the migrated target now produce, and exactly what the legacy
assembler does not.  Three independent components agreed on the flattened
model before this change; only the legacy assembler disagreed.

**Explicit dispositions.**  Interleaved ``sds`` **stays fail-closed** at
``unsupported_sparse_output``, with evidence: its dense level's parent count is
the *dynamic* stored-coordinate count of the compressed level above it, so a
dense loop would have to sit below a stream loop, and a compressed ancestor
that turns out not to materialize would need its speculative per-dense-cell
position closes rolled back -- a semantics no existing node owns.  Its legacy
comparand (census ``D9``) is recorded malformed *and* numerically wrong, so
there is no honest comparand either.  The trailing-dense D10/D11 families
(``sd+sd``, ``sdd``) keep ``unsupported_sparse_output_domain``: their parent
coordinates come from a cursor rather than a dense loop, so the
flattened-prefix model does not reach them.  Permuted compressed structure
keeps its pre-existing pre-LoopIR ``InvalidSchedule`` rejection.

### 45.5 Recorded seam moves

Four inherited seam locks named cells this milestone migrates and are moved to
the neighbour that still occupies each seam, with the move named in place --
the §43.8 discipline.  Two of them are moving for the *second* time, and the
comments record both hops.  A fifth lock, in
``test_loopir_dense_leaf_cooperand_target.py``, was found only by running the
focused suites and is moved to a ``dds`` result fed by a ``dsd`` dense-leaf
operand.  The detached full suite then found a **sixth** lock in
``test_loopir_cin_lowering.py``; ``a1582a9`` moves that lock from the newly
admitted ``dds`` copy to the interleaved ``sds`` neighbour.  The total is six,
not five.

The layout differential additionally shows **144 arm-instances of ``dds``
neighbours whose fail-closed code sharpens** from ``unsupported_sparse_output``
to ``unsupported_sparse_output_domain``.  Every one is still rejected; the
layout is now recognized, so the diagnosis names the actual domain violation
instead of reporting an unknown layout.  All 144 are ``dds`` results; no other
result format changed code.

### 45.6 Cluster 2 (multi-compressed reduction/TTM): an honest NO-GO

**This vertical is not implemented in this milestone, and the reason is stated
rather than disguised.**  Implementing it means giving workspace allocation,
reset, lifetime, producer reduction, drain and ordered result assembly
explicit structured ownership in LoopIR -- a slice comparable in size to the
whole B1 SpGEMM milestone -- and doing it on top of a legacy comparand that is
memory-unsafe.  Delivering a partially verified workspace vertical would be
worse than delivering none, so it was not started.

What this milestone *does* contribute is a localized, independently verified
root-cause map, which is strictly more than the inventory §43.5 left:

1. ``scheduler.py:2084`` -- ``insert_workspace`` anchors the ``Where`` at
   ``reduction_vars_todo[-1]``, the **innermost** reduction variable, and
   derives ``dim_workspace`` from the free variables after it.  Every
   contraction outer to that anchor stays above the ``Where``.  This is the
   root cause of the §43.2(b) rank-1 two-contraction placement defect.
2. ``cin_lowerer.py:~998`` -- the workspace is emitted as
   ``coo_workspace<T, wksp.dim>(1024, result_shape)``: the template arity is
   the workspace **key** rank while the runtime argument is the whole
   **result** shape.
3. ``src/scorch/csrc/header.h:653-655`` -- ``insert`` requires
   ``coord.size() == N && _resultShape.size() == N``, so whenever the key rank
   differs from the result rank *every* insert throws
   ``workspace coordinate rank mismatch``.  Together with (2) this is the
   inherited rank-2-workspace/rank-3-shape mismatch named in the handoff.
4. ``cin_lowerer.py:1923`` -- the nested drain takes its leaf coordinate from
   ``wksp_access.get_index_vars()[0]``, the **first** workspace index, while
   the level it writes is derived from the **last**.  That is only
   coincidentally correct for a rank-1 key; for a rank-2 key it stores the row
   coordinate twice and drops the column.  A companion limit at
   ``cin_lowerer.py:1930`` writes at most two coordinate levels, so a rank-3
   coordinate result never receives level 0.

The publicly reachable consequence is reproduced and bounded:
``scorch.einsum`` over an ``sss`` receiver rejects ``ijk->i``, ``ijk->j`` and
``ijk->k`` with ``TensorIndexError: compressed mode 0 coordinates must be
strictly increasing within parent 0`` -- malformed storage caught by
validation, not returned -- while ``ijk->ij`` and ``ijk->ik`` are correct, and
the rank-2 controls ``ij->i``/``ij->j`` are correct on both ``ss`` and ``ds``.
On the LoopIR route the whole declared matrix is rejected arm-invariantly:
rank-1 and rank-2 sparse reductions out of rank 3 stop at the loop-plan
boundary with ``sparse_parent_dominance``, and every TTM form
(``sss``/``dds``/``dss`` results, dense or sparse second factor) stops at
``unsupported_sparse_output``, with the dense-result control at
``unsupported_program_shape``.  The eight §41.4 cells still terminate with
SIGSEGV under the legacy comparand and remain executable failure evidence
only.

### 45.7 Verification

All gates ran in the ``scorch`` conda environment.  Evidence is retained under
``~/.cache/scorch-codex/phase7-multiprefix-4ce6bca/``.

- **Exhaustive layout differential**, 1,584 cells x 2 automatic arms at ranks
  1-4, base ``52d43cc`` versus candidate, both trees isolated so the only
  delta is the two compiler files: **3,012 unchanged, 0 regressions, 0 legacy
  drift, 0 newly rejected**, 12 newly-admitted arm-instances (6 cells) and 144
  code-sharpening arm-instances, all ``dds``.  None of the newly admitted
  cells is at byte parity, which is the expected and required outcome for a
  family whose legacy assembler is malformed.  Consequently the retained
  ``envelope/diff.txt`` ends in the harness-level text ``VERDICT: FAIL``: that
  verdict means "not every admitted candidate is byte-equal to legacy," not a
  regression in this oracle-gated family.  The file is retained as expected
  non-parity evidence and must not be described as a passing byte-parity gate.
- **Execution sweeps**: a 120-case matrix (ranks 3-5, f32/f64, copy/mul/add,
  zero extents, singleton prefixes, both arms) and a 162-case matrix (mixed
  and commuted operands, random/ragged/empty/explicit-stored-zero fixtures,
  cancellation) both pass storage well-formedness, the dense PyTorch
  differential and base/scheduled oracle agreement.  The only rejections are
  the four pre-existing, unchanged ``ds``-copy cells.  The historical ledger
  retained the scripts but **not** their stdout, exit status, revision, or
  import-provenance receipts, and those scripts hard-code the active checkout;
  these counts therefore remain claims pending the provenance-correct rerun in
  the subsequent exact-tip review, not self-contained historical gates.
- **Independent multi-seed randomized oracle sweep**: **560/560 checks over
  five seeds**, spanning seven layouts (``dds``, ``ddss``, ``ddds``,
  ``dddss`` plus the inherited ``dss``/``sss``/``ss`` controls), MUL and ADD,
  f32 and f64, two densities and both automatic arms.  Each check compares
  **four independent computations** of the same result: the compiled kernel's
  produced level storage, the base-program oracle, the scheduled-program
  oracle, and the dense PyTorch reference -- with exact ``(pos, crd)`` tuple
  equality between compiled storage and oracle storage, not merely a numeric
  match.  Legacy is deliberately absent from this sweep.
- **Deterministic 54-cell census** at the tip, one cell per fresh subprocess
  with a 900-second timeout and RLIMIT_CPU/RLIMIT_CORE isolation: **46 records
  written, 46/46 arm-invariant**, and the same **eight cells terminate with
  SIGSEGV** (``C1``-``C7``, ``C13``) -- an identical exit-code file to the
  retained baseline, zero exit-code changes.  **42 of 46 cell records are byte
  identical**; the only four that move are ``D5`` (``dds`` copy), ``D6``
  (``dds+dds`` union), ``D7`` (``ddss`` copy) and ``D8`` (``ddss+ddss``
  union), each flipping ``reject`` to ``admitted`` in **both** arms with equal
  emitted length across arms.  LoopIR admissions rise 25 -> 29.  **All 25
  inherited admissions remain at byte parity with legacy; the four that are
  not at parity are exactly D5-D8** -- and every one of those four is in the
  independently recorded malformed-legacy-storage set, so the parity loss is
  exactly co-located with the legacy defect and nowhere else.  The legacy
  columns are unchanged: **eight cells produce malformed legacy storage**
  (``B11``, ``D5``-``D11``) and **four are also numerically wrong**
  (``B11``, ``D9``, ``D10``, ``D11``), identical to §43.6.  ``D9``, ``D10``
  and ``D11`` keep their exact rejection codes.
- **Schedule audit**: **46 admitted / 40 rejected / 0 non-identical**, its
  JSON **identical to the retained baseline** after removing only the commit
  field.
- **Capture surfaces**: corpus **20/20**, grid **42/42**, heap **11/11**,
  anchors **22/22** and 22 of 23 automatic capture files raw byte-identical --
  **117/118**.  The single differing file is ``auto/report.json``, whose only
  delta is a ``cache_key_suffix`` field; a control regeneration from the
  unmodified base tree produces a *third* value for that field, proving it is
  environment-derived and not caused by the change.  All 22 non-report
  automatic capture sources are byte-identical between base and candidate.
- **Full-tree static parity**, base ``52d43cc`` versus candidate, one
  invocation each of ``black --check src tests``, ``flake8 src tests`` and
  ``mypy src``: Black **15 findings at both**, Flake8 **47 at both with
  byte-identical normalized logs**, mypy **140 errors in 11 files at both**.
  The only residual differences are Black's count of *unchanged* files
  (134 -> 135, from the one added test file) and two line numbers in the same
  pre-existing ``stensor.py`` finding, shifted by the 14 lines the singleton
  guard adds.  ``git diff --check`` is clean.
- **Exact-tip clean detached full non-performance suite: COMPLETE.**
  **5,734 passed / 14 skipped / 0 failed over all 5,748 selected nodes**
  (5,751 collected, 3 performance deselected), in eight file-disjoint fresh
  processes (720 / 718 / 717 / 717 / 720 / 717 / 720 / 705+14sk), every
  partition exiting 0.  A pre-run proof places all **86** tracked pytest
  modules exactly once and the partition node counts sum exactly to the
  selected total, so the node union is complete and non-overlapping.
  Partitions 0-3 and 5-7 ran at ``4ce6bca`` and partition 4 at ``a1582a9``;
  those revisions differ by exactly one file,
  ``tests/test_scorch/test_loopir_cin_lowering.py``, a member of partition 4,
  which therefore ran at ``a1582a9``.

  **This gate earned its keep again.**  Partition 4 initially failed
  ``test_unsupported_sparse_output_layout``: a *sixth* inherited seam lock,
  and the *third* successive occupant of that particular lock, asserting
  ``unsupported_sparse_output`` for the ``dds`` copy this milestone admits.
  Neither the layout differential nor the census could have caught it -- both
  compare compiler outcomes, not test expectations.  ``a1582a9`` moves it to
  the interleaved ``sds`` neighbour and names the move in place.
- **Activating paired two-order compile latency: NOT SATISFIED ON THIS HOST,
  and not claimed as a pass.**  An immediately preceding alternating A/A
  control on the same idle host is tight -- p50 0.9843-1.0001, p95
  0.9819-0.9990 across all four shapes -- so the harness and the host's
  short-timescale stability are sound.  The paired LoopIR-versus-legacy run
  (200 warmups / 2,000 samples, alternating) nevertheless measures
  ``mp_dds_mul`` **1.1213/1.1268**, ``mp_dds_add`` **1.0661/1.0716**,
  ``mp_ddss_mul`` **1.0886/1.1072** and the shared control ``b3_ss_mul``
  **1.1065/1.1135** at p50/p95 -- three of four over the declared 1.10
  ceiling.

  **The attribution is measured, not asserted.**  The same harness run against
  an isolated **base ``52d43cc``** tree on the same host measures ``b3_ss_mul``
  **1.1013/1.1021** and ``dl_ss_sd_mul`` **1.1053/1.1082** -- both already over
  the ceiling with none of this milestone's code present -- and
  ``dl_sss_sdd_mul`` 1.0881/1.0881.  On the shared control the candidate minus
  base delta is **+0.0052 p50 / +0.0114 p95**, inside the A/A control's own
  +/-0.018 band.  So this Apple M5 host carries a LoopIR-versus-legacy offset
  the declared 1.10 target was not calibrated for; §44.4's passing receipt
  (``b3_ss_mul`` 1.0909/1.0961) was produced on an unloaded Redwood x86 host.
  The honest statement is that **the gate is outstanding and must be re-run on
  unloaded Redwood before any Phase-7 exit**, and that on the evidence
  available the migration itself does not account for the crossing.  The
  failed percentiles are reported, not averaged away.
- **The five protected tracked files** retain their recorded SHA-256 values;
  origin remains ``58e8565``; nothing was pushed, amended, squashed or
  reordered; only explicit paths were staged, and every unrelated tracked and
  untracked GPU/benchmark/scheduler/research path is untouched.

### 45.8 Phase-7 exit audit

Criterion by criterion.

*Migrated families complete over their proven envelopes.*  The
multiple-dense-prefix family is, over ranks 3-5, f32/f64, copy/intersection/
ordered-union, mixed and commuted operands, ragged/empty/zero-extent/
explicit-stored-zero fixtures and cancellation, in both automatic arms, gated
on the LoopIR oracle and the dense PyTorch reference because its legacy
comparand is provably malformed.

*Every neighbour carries a stable fail-closed code.*  Yes, in both arms, with
the moves recorded: six inherited seam locks relocated to the neighbour that
still occupies each seam, and 144 arm-instances of ``dds`` neighbours whose
code sharpens from ``unsupported_sparse_output`` to
``unsupported_sparse_output_domain``.  ``sds`` keeps
``unsupported_sparse_output``; ``sd``/``sdd`` keep
``unsupported_sparse_output_domain``; permuted compressed structure keeps its
pre-LoopIR ``InvalidSchedule``.

*Representation unchanged.*  No node kinds, canonical schema, request
identity, schedule identity or erasure changed.  The canonical dump is
arm-stable and erases to base for every migrated layout.

*Release behaviour unchanged.*  The schedule audit equals its retained
baseline; 117 of 118 capture files are raw byte-identical with the single
difference proven environment-derived by a base-tree control; no default
dispatch, cache or selector changed.

*The declared matrix is closed.*  **It is not.**  One of the two declared
clusters -- multiple dense prefixes / interleaved dense levels -- is closed for
the dense-prefix half, with interleaved ``sds`` deliberately and evidentially
excluded.  The other -- multi-compressed reduction/TTM -- is **not
implemented**.  §45.6 states why and hands over four exact, independently
reproduced root-cause sites instead of an inventory.

*The activating paired latency receipt.*  **Outstanding, not passed.**  Three
of four shapes cross the 1.10 ceiling on this host -- but so do two of three
INHERITED shapes measured on an isolated base ``52d43cc`` tree on the same
host, and the shared control's candidate-minus-base delta is +0.0052 p50 /
+0.0114 p95, inside the A/A control's own band.  The crossing is a property of
this host's LoopIR-versus-legacy baseline, not of the migration; the gate must
be re-run on unloaded Redwood before Phase-7 exit, and until then it counts
against exit.

**Phase 7 therefore does not exit on this milestone.**  No Phase-8 inventory
was started, no cutover, cache, selector or default-dispatch change was made,
and no legacy code was deleted.  The milestone's contribution is nonetheless
concrete: one declared cluster closed at oracle strength, two publicly
reachable runtime ownership defects fixed, one carried-forward evidence
receipt independently re-derived rather than trusted, two evidence
inaccuracies recorded rather than resealed, and the remaining cluster reduced
from "blocked, needs its own vertical" to four named lines of code with a
reproduced public consequence.

## 46. Exact-tip review corrections and cross-host closure (2026-08-09)

This section supersedes §45's continuation instructions.  The code/test tip is
``bb429f4``.  The review first re-read and adversarially exercised the complete
``52d43cc..372b0fc`` milestone; the flattened-dense-prefix target itself is
sound over its declared envelope.  Four defects were instead found in shared
validation and emission boundaries reached by the milestone, and each was
fixed before any continuation work:

1. ``c5657cc`` / ``625c04f`` align ``LevelFormat`` construction, ownership
   audit and CIN normalization for genuine ``int`` subclasses.  Validation
   now obtains the canonical value through the base ``int`` descriptor,
   rejects foreign ``__class__`` spoofs and lying comparisons, retains the
   constructor-valid subclass until the ownership boundary, and stores an
   exact owned integer after normalization.  ``bool`` and non-positive widths
   remain rejected.
2. ``bcdd10d`` / ``d50f446`` remove the redundant successful-path authority
   replay from merged ``PositionLoad`` emission.  The sealed-target and final
   graph checks remain authoritative, cold-path replay still owns its precise
   diagnostics, and an activating count lock pins seal / nested-input lookup /
   explicit replay at exactly **1 / 1 / 0**.
3. ``6e9ef24`` / ``49a5758`` make level-mode validation use actual type/MRO
   state, not caller-controlled ``__class__`` attributes.  Real ``str`` and
   ``LevelFormat`` subclasses remain compatible; foreign lookalikes fail
   closed without invoking their hooks.
4. ``d38d3b8`` / ``bb429f4`` close the remaining rejection-path callbacks:
   actual type names are read through base descriptors and canonicalized
   through ``str.__str__``, and runtime sequence recognition uses
   ``collections.abc.Sequence`` rather than the ``typing`` alias that queried
   a hostile metaclass.  Public failures are controlled ``TensorTypeError``
   instances, never caller exceptions.

The fixes add no LoopIR node, schema, schedule/request identity, dispatch,
cache, selector or C++ spelling change.  The protected files and unrelated
work remain outside every commit.

### 46.1 Evidence corrections

The historical evidence is not rewritten.  The §45 milestone has four
code/test commits through ``a1582a9`` (not three), moved **six** seam locks
(not five), and its retained non-parity envelope correctly ends in
``VERDICT: FAIL`` because the twelve oracle-gated new admissions must differ
from malformed legacy output.  The historical 120/162 scripts lacked their
own revision, import, stdout and exit receipts.  Their exact-tip reruns now
show the distinction that the old prose blurred:

- the raw 120-case harness exits **1**, with **116 PASS / 4 FAIL**; a separate
  exit-0 characterization proves the complete fail set is exactly
  ``ds(3,4)`` copy x ``{float32,float64}`` x both arms, each at the retained
  ``unsupported_program_shape`` seam;
- the wider harness exits 0 at **162/162**, and the randomized oracle/storage
  sweep exits 0 at **560/560 over five seeds**.

Likewise, the exact 54-cell census reproduces **46 records, 46 arm-invariant,
eight SIGSEGV cells (C1-C7 and C13), and zero timeouts**.  Its raw wrapper exits
1 only because it lexicographically sorts ``C13`` before ``C2`` and compares
that list against numeric order; the retained raw result is paired with an
order-insensitive exit-0 set characterization.  The exit map, record names and
all 46 environment-independent semantic projections equal the retained
``4ce6bca`` baseline.  Raw numeric-conversion fields are retained but excluded
from that projection because this host's loaded ``scorch_ops.Tensor`` lacks
``to_torch``; all 40 such probes record the exact inspect error.  Neither raw
nonzero receipt is renamed or averaged into a pass.

The native workspace reference is now written as
``src/scorch/csrc/header.h:653-655``.  The entry point begins at line 653, the
rank invariant is checked at line 654, and its diagnostic is thrown at line
655; the earlier shortened path obscured the actual owner.

### 46.2 Exact-tip verification

Evidence is retained under
``~/.cache/scorch-codex/phase7-multiprefix-review-bb429f4/``.

- Focused format/CIN/conversion membership: **296 passed**.  The complete
  clean detached non-performance suite collected 5,767 nodes, deselected the
  three performance nodes, and completed **5,750 passed / 14 skipped / zero
  failures over all 5,764 selected nodes**.  All 86 tracked pytest modules
  occur exactly once across eight fresh-process partitions; every partition
  ran at exact ``bb429f4`` and exited 0.
- The exact schedule audit was run twice: **46 admitted / 40 rejected / zero
  non-identical**, with raw-identical repeated output and normalized equality
  to the retained baseline.  The historical revision-pinned wrapper rejects
  this newer tip as designed and is retained as an incompatibility receipt.
- The exact 120/162/560 receipts have asserted revision and import provenance,
  empty stderr and a clean detached source tree.  The 20-source corpus and
  42-source grid each regenerate byte-identically to the retained captures;
  their native-build sentinel directories remain empty.  The correctness
  subledger seals 288 retained files (manifest SHA-256
  ``0181d9c77ec1fee68cf3d6b95789731dd53806bb32c6649031286613190d5120``)
  and explicitly excludes worktrees, extension caches, pycache and native
  build products; all 288 entries verify.
- Full static parity against clean ``372b0fc``: Black exits 1/1 with the same
  15 finding files and 15/135 counts (only concurrent output order differs),
  Flake8 is raw-byte-identical at 47 inherited findings, and mypy is
  raw-byte-identical at 140 errors in 11 files plus two notes.  No Flake8 or
  mypy coordinate drift exists.  ``git diff --check`` is clean.
- Redwood ran 200 warmups / 2,000 samples for seven activating or shared
  control shapes.  A fresh alternating repeat is wholly inside 1.10 (worst
  p50 **1.0960**, p95 **1.0986**) and its A/A control is tight (worst p50
  **1.0059**, p95 **1.0136**).  The first and order-flipped runs retain
  marginal crossings (worst **1.1131/1.1125**), so those runs are not averaged
  away and the Redwood evidence remains mixed.
- At the user's suggestion, Slurm job ``16596836`` repeated the gate on MKT
  in job-local ``/scr/u/bobbyy`` storage (AMD EPYC 9334, four allocated CPUs,
  no direct ``mkt1`` login).  Both candidate repeats are wholly inside 1.10,
  worst **1.0960/1.0953** and **1.0950/1.0946** p50/p95; the A/A control is
  inside **1.0013/1.0095**.  The seven emitted-source payloads are byte-equal
  between MKT and Redwood and between ``d50f446`` and ``bb429f4``.  Failed
  setup attempts (missing native extension, then an incorrectly rooted
  package metadata build) remain in the ledger rather than being hidden; the
  scratch-local overlay build used by the successful job exited 0.

### 46.3 Phase disposition

The review corrections are complete, but **Phase 7 remains open**.  The one
remaining production-relevant cluster is still §45.6's multi-compressed
sparse-reduction/TTM workspace vertical.  Its four blockers remain at
``scheduler.py:2084``, ``cin_lowerer.py:~998``,
``src/scorch/csrc/header.h:653-655`` and
``cin_lowerer.py:1923``/``:1930``.  No Phase-8 cutover, default-dispatch,
selector or cache change was made, and no fallback or legacy implementation
was removed.  Cross-host latency evidence now attributes §45's local crossing
away from these emission-neutral corrections, but it cannot close the absent
semantic vertical; Phase 7 therefore has no exit verdict yet.

## 47. The ordered workspace key domain, a fifth cluster-2 blocker, and a NO-GO (2026-08-09)

This milestone starts at inherited documentation tip ``8b4b5fc``.  It reviews
``372b0fc..8b4b5fc``, fixes the two defects that survived adversarial
verification, lands the representational half of the Phase-7 cluster-2
vertical, and records a **NO-GO** with a blocker that the inherited
four-blocker map does not contain.  Origin remains ``58e8565``; nothing was
pushed, amended, squashed or reordered; the five protected tracked files hash
exactly as recorded.

### 47.1 Independent review of the inherited range

Four contracts were re-derived from the code and from live probes, never from
the review or handoff prose, and every claimed defect was then put to
independent adversarial verifiers instructed to refute it.

**Three contracts reproduce.**

- *Valid int subclasses canonicalize at the ownership boundary.*
  ``LevelFormat.__init__`` deliberately RETAINS the caller's object and only
  computes a canonical value for validation (``format.py:191-206``); the
  conversion to an exact ``int`` happens at ``audit_format_state``
  (``format.py:589``), which every one of the five retaining sites reaches.
  An ``IntEnum`` width survives the whole public path and is owned as exact
  ``int``.  A subclass lying in either direction cannot move the verdict:
  ``NegLie(8)`` (whose ``__int__``/``__le__`` claim it is negative) is
  accepted and owned as 8, while ``PosLie(-3)`` (whose ``__int__`` claims 32)
  is rejected.  ``bool``, zero and negative widths fail closed with zero
  caller hooks.
- *Merged ``PositionLoad`` emission performs exactly one seal lookup, one
  nested input-authority lookup and zero successful-path replays.*  Measured
  at 1/1/0 with an independent counting harness, and the cold mismatch path
  keeps its diagnostics.
- *Forged, mutated, aliased and cyclic cold paths retain controlled
  diagnostics.*

**The fourth does not.**  "Actual-type/MRO format validation executes no
caller callbacks" is false, and two independently reproduced defects are the
reason (§47.2).

### 47.2 Two defects found by fresh probes, fixed first (``7c8617e`` / ``a79ac78``)

``_normalize_level_formats`` decided whether a foreign object was a sequence
with ``issubclass(value_type, collections.abc.Sequence)``.  ``ABCMeta``'s
subclass check inserts the candidate into its positive and negative
``WeakSet`` caches, and those inserts call the candidate metaclass's
``__hash__`` and ``__eq__``.  Measured on an ordinary
``class Hostile(metaclass=HostileMeta)``: **eight ``__hash__`` calls** on the
first ``TensorFormat(hostile)`` and **two ``__eq__`` calls** on the second,
with a control confirming ``issubclass(Hostile, (bytes, bytearray))`` fires
none.  With a metaclass whose ``__hash__`` raises, a bare
``builtins.RuntimeError`` **escaped the public ``TensorFormat(...)``
constructor**.  That contradicts §46 item 4's "public failures are controlled
``TensorTypeError`` instances, never caller exceptions" and §46 item 3's
"foreign lookalikes fail closed without invoking their hooks."

The hook also fired on the **success** path: constructing a format from a
genuine ``collections.abc.Sequence`` subclass invoked the caller metaclass
twice, so a raising ``__hash__`` could kill an otherwise-legal construction.
``TensorFormat.from_dict`` carried the same defect at three sites.

This is worth stating plainly: ``d38d3b8`` — the commit written to *close*
metaclass callback gaps — moved ``Sequence`` from the ``typing`` alias to
``collections.abc`` to harden the MRO check.  It closed the ``__class__``
route and left the ``__hash__``/``__eq__`` route open, and the lock it shipped
with (``test_real_sequence_mro_check_bypasses_the_candidate_metaclass``)
counts only ``__class__`` reads, so the surviving route passed unnoticed.

``_derives_from`` replaces every one of these with identity membership in the
real MRO, read through the base ``type.__mro__`` descriptor — the same
base-descriptor idiom ``_actual_type_name``, ``_normalize_shape`` and
``_normalize_permutation`` already use.  The hostile-metaclass probes now
measure **zero hooks** and report ``TensorTypeError``/``TensorFormatError``.

Acceptance is unchanged, and that required care: ``collections.abc``
registers the concrete builtins *virtually* rather than by inheritance
(``Mapping not in dict.__mro__``, ``Sequence not in list.__mro__``), so a
naive MRO recognizer would have silently rejected plain ``dict`` and ``list``
and broken ``from_dict``.  ``_MAPPING_BASES``/``_SEQUENCE_BASES`` name them
explicitly; ``dict``, ``OrderedDict``, ``mappingproxy``, ``list``, ``tuple``,
list subclasses, real ``Sequence`` subclasses and the ``serialize`` round trip
all still work.  Virtual registrations beyond those builtins now fail closed,
which is how this boundary already treats every other foreign lookalike.

``a79ac78`` adds four locks.  **Three fail against ``8b4b5fc``'s
``format.py`` and pass against the fix**, verified by swapping the file in and
back; the fourth is the anti-narrowing lock that would have caught the
builtin-registration trap.

### 47.3 Evidence qualifications, reproduced rather than trusted

All three reproduce at the tip.

- **Six seam locks.**  Five ``Recorded seam move`` annotations were added in
  ``52d43cc..a1582a9`` plus ``a1582a9``'s docstring move in
  ``test_loopir_cin_lowering.py``.  The sixth annotation, in
  ``test_loopir_multi_dense_prefix_target.py``, is the *code-sharpening* cell
  §45.5 describes separately, not a lock move — so the count is six, exactly
  as §46.1 corrects it.
- **The raw 120-case harness exits 1** with **116 PASS and exactly four
  failures**, all ``ds(3, 4)`` copy × ``{float32, float64}`` × both arms at
  the retained ``unsupported_program_shape`` seam, stderr empty.  The wider
  harness exits 0 at **162/162** and the randomized oracle/storage sweep exits
  0 at **560/560 over five seeds**.
- **The 54-cell census** produces **46 records, 46 arm-invariant, eight
  SIGSEGV cells (C1-C7 and C13), zero timeouts**, and **all 46 cell records
  are byte-identical to the retained ``bb429f4`` baseline**.  Its wrapper
  exits 1 solely because it compares the lexicographic list
  ``["C1","C13","C2",...]`` against the numeric ``["C1","C2",...,"C13"]``; the
  sets are equal.

### 47.4 The ordered key domain (``a1fc642`` / ``81d9d7c``)

The serial sparse workspace was declared over exactly one drain dimension, and
``nodes.py`` said so: *"There is deliberately no multi-dimensional drain form
in this subset."*  That is the representational reason cluster 2 has no home
in LoopIR — a rank-2 sparse reduction out of rank 3 needs a two-component key,
and TTM needs a key whose rank differs from the result's.

The three nodes now carry ordered tuples: ``SparseWorkspaceDecl
.key_dimensions``, ``SparseWorkspaceInsert.coords``, ``SparseWorkspaceDrainFor
.indices``.  ``len(key_dimensions) == 1`` is the ``K == 1`` instance of the
same node, not a separate kind, so every migrated family keeps its exact
shape.

Two contracts are now stated structurally rather than implied.

- **The key domain is declared independently of any result layout.**  No
  result level structure, dense-prefix extent or result rank appears in the
  decl.  Conflating them is exactly what makes the legacy path instantiate
  ``coo_workspace<T, key_rank>`` with the whole result shape and then reject
  every insert.
- **The drain visits entries in strictly increasing lexicographic key
  order.**  When the key dimensions are listed in result level order that is
  also the canonical append order, which is what will make multi-level
  ordered assembly correct without any sort in the IR.

**Blocker 3 is not a native limit, and this is worth correcting in the
inherited map.**  ``coo_workspace<T,N>`` is already fully rank-N: its
constructor's shape vector is used *only* to flatten the dedup key, and
``sort()`` (``header.h:692-704``) compares the N coordinate components in
order — a genuine lexicographic comparator, not a flattened index.  Passing
the **key-domain extents** instead of the result shape makes
``coord.size() == _resultShape.size() == N`` hold.  **No C++ change is
required.**  Blocker 3 is a consequence of blocker 2, not an independent
obstacle.

Ownership landed at every site the milestone names except LLIR lowering:
nodes, builder, verifier, canonical printer (schema **v10 → v11**, because the
serialized shape changed), oracle, and schedule application/erasure.
``lower_llir`` is deliberately left as the **sole rank-1 fail-closed
boundary** — both existing targets now assert ``len(key_dimensions) == 1``
explicitly, since they emit ``coo_workspace_1d<T, 1>``.

Two defects in the new representation were caught by the pre-implementation
ownership map rather than by tests: the canonical schema version had not been
bumped despite a changed serialized shape, and the oracle's
``zip(stmt.coords, decl.key_dimensions)`` would have silently truncated a
malformed key to the shorter side instead of failing.  Both are fixed; the
oracle owns an explicit rank check because ``run_program`` is reachable
directly.

### 47.5 Cluster 2: a fifth blocker, and the reachability split (``d6e32f0``)

The inherited four-blocker map is accurate but **incomplete**, and the missing
blocker changes what a migration slice can attempt.

``Scheduler.select_loop_order`` ends with a forced reorder — *"ensure at least
one free variable appears after the last reduction variable"* — that moves the
last free variable to the very end of the loop order whenever none follows the
last reduction, **with no legality check**.  Measured by instrumenting
``_verify_storage_order`` and printing ``plan.loop_order`` beside each access
layout's storage positions:

- ``sss ijk->ij`` declared ``i,j,k`` becomes plan ``i,k,j`` while ``A``'s
  storage order is ``i,j,k``;
- ``ss ij->i`` declared ``i,j`` becomes plan ``j,i``.

Both then die at ``loop_plan_legality.py:288``'s ``sparse_parent_dominance``
**before any LoopIR admission decision is reached**.  That reorder exists
precisely because the legacy ``insert_workspace`` can only key a workspace on
free variables below the *innermost* reduction — blocker 1 seen from the other
side.

**It is deliberately not fixed here.**  ``Scheduler.apply_schedule`` is the
single owner of Schedule-to-LoopPlan translation for *both* pipelines
(``loopir/pipeline.py:250-280``), so there is no LoopIR-only divergence point.
Legacy computes these reductions correctly from the reordered order today —
verified against PyTorch on ``ss``/``ds``/``dd`` rank-2 and ``sss`` rank-3
cells, maximum difference 2.4e-07 — so changing the order would change
generated code on the default dispatch path.  That is a Phase-8 cutover
decision, which this milestone is forbidden to make.

The consequence is a **reachability split**, now pinned as a committed
arm-invariant census (``test_loopir_reduction_ttm_census.py``, 25 tests):

- **Five cells are reorder-blocked**: ``sss ijk->i``, ``sss ijk->j``,
  ``sss ijk->ij``, ``ss ij->i``, ``ds ij->i``.  No LoopIR-side widening can
  reach them.
- **Seven cells are reachable**, with legal plan orders and exact LoopIR
  admission codes: ``unsupported_sparse_output`` (``sss ijk->k``,
  ``ss ij->j``, ``ds ij->j`` and both TTM forms),
  ``sparse_workspace_target_invalid`` (``sss ijk->ik``) and
  ``unsupported_schedule_auto_family`` (``sss ijk->jk``).

The seven reachable cells span every shape the vertical needs — rank-1 and
rank-2 keys, with and without a bound prefix, single-cursor and merged
producers — under one rule: **anchor the region at the outermost reduction and
key it on the result indices at or below that anchor, in result level order.**
B1 SpGEMM is that rule's ``K = 1, prefix = 1`` instance, which is why the
migrated family is the ``K == 1`` case of the ordered key domain rather than a
separate form.  A public probe additionally records that TTM
``ijk,kl->ijl`` over ``sss × ss`` fails with ``compressed mode 1 position
array must start at zero`` — a *distinct* legacy malformation from the
reductions' ``coordinates must be strictly increasing``, not previously noted.

### 47.6 Verification

Evidence retained under ``~/.cache/scorch-codex/phase7-cluster2-8b4b5fc/``.

- **Generated source identity: 62/62 byte-identical.**  The retained 20-source
  corpus and 42-source grid captures regenerate byte-for-byte against the
  ``bb429f4`` baseline from a clean detached candidate worktree, with the
  native-build sentinel holding zero artifacts exactly as the baseline
  records.  This is the direct proof that the representation change is
  emission-neutral.
- **Schedule audit: 46 admitted / 40 rejected / 0 non-identical**, its JSON
  **identical to the retained baseline** after removing only the commit field.
- **Full static parity in isolated base/candidate worktrees**: Black **15
  finding files at both**, identical set; Flake8 **47 at both**, identical
  after path normalization; mypy **140 errors in 11 files at both**,
  identical.  The only movement is Black's count of *unchanged* files
  (135 → 136) from the one added test file.  ``git diff --check`` clean.
- **Exact-tip clean detached full non-performance suite: COMPLETE.**
  **5,796 passed / 14 skipped / 0 failed over all 5,810 selected nodes**
  (5,813 collected, 3 performance deselected), in eight file-disjoint
  fresh-process partitions (728 / 727 / 725 / 727 / 725 / 727 / 727 /
  710+14sk), every partition exiting 0, all at exact ``d6e32f0`` in a clean
  detached worktree.  A pre-run proof places all **85** modules carrying a
  selected node exactly once and shows the partition node counts sum exactly
  to the selected total, so the union is complete and non-overlapping.  Of
  the 87 tracked pytest modules the two remaining ones are accounted for:
  ``test_helpers.py`` defines no test function, and ``test_perf_large.py``'s
  two tests are performance-marked.
- Focused suites: 2,661 LoopIR tests, 291 verifier/oracle tests, 81
  format-ownership tests, 25 census tests, 238 format/CIN/tensor tests.
- Request and cache identity cannot have moved: ``plan_identity`` digests
  ``(cin, plan, result_shape, inputs, compile_options)``, never the LoopIR
  program.
- The five protected tracked files retain their recorded SHA-256 values;
  origin remains ``58e8565``; 119 unrelated untracked GPU/benchmark/scheduler/
  research paths are untouched; only explicit paths were staged.

### 47.7 Phase-7 exit audit

*Migrated families complete over their proven envelopes.*  Unchanged — no new
family was migrated.  The representational half of cluster 2 landed and is
oracle-verified at K = 1, 2 and 3, including a rank-2 key whose insertion
order deliberately disagrees with its key order.

*Every neighbour carries a stable fail-closed code.*  Yes, and cluster 2's
twelve cells are now pinned arm-invariantly rather than re-derived.

*Representation unchanged.*  **No.**  Deliberately: the sparse-workspace nodes
gained an ordered key domain and the canonical schema moved v10 → v11.
Release behaviour is unchanged — 62/62 generated sources byte-identical, the
schedule audit identical, no dispatch/selector/cache change.

*The declared matrix is closed.*  **No.**  Cluster 2's semantic half —
schedule-pass admission, the rank-K LLIR target with multi-level ordered
assembly, and public wrapping — is **not implemented**, and five of its twelve
cells are unreachable behind a shared-scheduler blocker this milestone
declines to touch.

*The activating paired latency receipt.*  Not re-run.  This milestone's
changes are proven emission-neutral by byte-identical generated sources, so
there is no new activating shape to measure; §46.2's cross-host evidence
stands unchanged.

**Phase 7 is NO-GO.**  No Phase-8 inventory, cutover, cache, selector or
default-dispatch change was made, and no legacy code was deleted.

**The exact remaining blocker, stated as precisely as it is now known.**
Cluster 2 needs three things, in order:

1. **Schedule-pass admission** implementing the §47.5 anchoring rule.
   ``apply_sparse_workspace`` still gates on ``len(workspace.axis_loops) == 1``
   and a rank-2 identity-ordered result.  The plan layer is already ready:
   ``WorkspaceInsertion.axis_loops`` is a tuple and the automatic scheduler
   already emits every free variable after the last reduction.
2. **A rank-K LLIR target** with multi-level ordered assembly.  This is the
   bulk: the existing ``_SparseWorkspaceLowering`` is ~1,270 lines for the
   rank-1 CSR case, hard-codes a two-level assembly, and names its single key
   dimension throughout.  It needs the key-domain extents passed to
   ``coo_workspace<T,K>`` (no C++ change) and per-level position tracking with
   parent linking.
3. **CIN admission**, which is one wall: ``lower_cin.py:688``'s
   ``result_levels != (DENSE, COMPRESSED)`` catch-all rejects every rank-2/3
   key and every multi-compressed TTM result, plus ``:599`` for the rank-1-key
   TTM shape.

And, separately gated, the five reorder-blocked cells require a decision about
``Scheduler.select_loop_order``'s forced reorder that changes default-dispatch
generated code.

## 48. Rigorous review corrections to the ordered-key milestone (2026-08-09)

This review starts at inherited documentation tip ``cf8cd44`` and reads the
complete ``8b4b5fc..cf8cd44`` range independently.  The ordered key-domain
representation itself is sound over its stated node/verifier/oracle contract,
and ``coo_workspace<T,N>`` really is rank-general.  Three surrounding
contracts were not sound: standard ``Sequence`` compatibility and public
diagnostic totality, root-region schedule erasure/provenance, and the claimed
cluster-2 census/evidence map.  They are corrected in five focused commits:

```
cf831f3  fix(format): preserve callback-safe sequence compatibility
e8a3f74  test(format): lock sequence compatibility and error translation
f3b1ab3  fix(compiler): erase root-owned workspace regions
cde1b9f  test(compiler): lock root rank-K workspace erasure
e13ecba  test(compiler): correct the cluster-2 frontier census
```

Nothing here admits a new target family, changes generated C++, changes the
canonical v11 schema, or changes dispatch, selector, cache, request identity,
or legacy scheduling.  Phase 7 remains **NO-GO**.

### 48.1 Format recognition was callback-safe but compatibility and diagnostics were not

The real-MRO recognizer from ``7c8617e`` avoids ``ABCMeta`` cache callbacks;
fresh hostile metaclass probes confirm that type recognition itself invokes
no caller ``__hash__``, ``__eq__``, ``__subclasscheck__``,
``__instancecheck__``, ``__class__``, ``__name__`` or ``__mro__`` hook.
Section 47 nevertheless overclaimed two consequences.

First, the previous ABC boundary accepted standard virtual ``Sequence``
implementations that ``_SEQUENCE_BASES`` omitted.  Exact base/candidate probes
showed ``deque(["d", "s"])`` and ``array("u", "ds")`` changing from a valid
``d,s`` format to ``TensorTypeError``, and ``memoryview(b"")`` changing from
the valid empty format to rejection.  ``cf831f3`` names the complete concrete
standard-library set that the prior boundary recognized, restoring those
outcomes without reintroducing an ABC query.

Second, callback-free **recognition** cannot mean callback-free **consumption**
of a genuine user-defined ``Sequence`` or ``Mapping``: its protocol methods
must run to obtain the payload.  Before this review, exceptions from those
methods escaped the direct ``TensorFormat`` constructor and ``from_dict`` as
bare caller ``RuntimeError`` instances.  The public boundaries now translate
such failures to ``TensorFormatError`` while preserving the original cause.
The corrected contract is therefore precise: recognition invokes no caller
hook; consumption may invoke the recognized protocol, but no ordinary caller
exception escapes the format API unclassified.

### 48.2 The generalized root-owned region could not be erased

The rank-K tests introduced exactly the placement the semantic vertical
needs: ``SparseWorkspaceRegion`` owns the program root, above every producer
loop.  The program verified and the oracle executed it, but
``erase_schedule`` failed before reaching the generalized erasure logic
because ``_decompose_chain`` unconditionally required an outer loop.  The
same assumption prevented scheduled-chain provenance from describing the
root-owned region.

``f3b1ab3`` makes empty outer chains an explicit opt-in and enables it only
for provenance and erasure; schedule-construction passes keep their previous
nonempty boundary.  The regression lock covers K = 1, 2 and 3 and a stronger
rank-2 case whose key is rotated against producer order and whose four-way
contraction has non-symmetric values.  Scheduled and erased oracle storage is
exactly equal, the independently computed lexicographic entries agree, and
provenance is producer loops followed by the composite drain identity.

### 48.3 The committed census is a representative frontier, not an exhaustive cluster

The ``d6e32f0`` coverage assertion proved only that its own two lists contained
12 names.  It omitted four of the six TTM layouts explicitly named by §45.6.
The corrected review matrix contains **16 representatives**:

- five cells blocked only under the empty automatic origin;
- five reachable reduction cells; and
- six reachable TTM cells, crossing result/receiver layouts
  ``sss``/``dss``/``dds`` with canonical dense/compressed second factors
  ``dd``/``ss``.

All eleven reachable representatives remain arm-invariant at their recorded
later diagnostics.  The word "representative" is binding: a separate
level-general audit already finds adjacent ``dss`` reduction and mixed
``ds``/``sd`` factor variants not enumerated by the 16-cell lock.  A migration
must derive and expand that frontier; it must not treat the committed list as
proof of format/rank exhaustiveness.

The five automatic-origin failures are also not intrinsically unreachable.
Committed controls supply legal explicit orders for ``sss ijk->ij`` and
``ss ij->i`` and reach later LoopIR target/lowering diagnostics.  A
LoopIR-specific automatic-plan repair is therefore architecturally possible
without changing legacy default emission, although it needs its own gated
design.  Sections 47.5/47.7 were wrong to call this necessarily a Phase-8
default-dispatch decision.

The blanket claim that legacy computes all five blocked reductions correctly
is removed.  It contradicts the existing runtime lock that public
``sss ijk->i`` and ``sss ijk->j`` reject malformed storage, and the quoted
``2.4e-07`` had no retained receipt.  Likewise, the five automatic-origin
failures have stable stage/message locks but no structured ``defect.code``;
only the eleven reachable entries carry codes.

### 48.4 Corrected implementation map

``lower_llir`` is not the sole K = 1 boundary.  The current semantic vertical
has independent restrictions in:

1. ``apply_sparse_workspace`` and ``_check_auto_plan_family`` (schedule fact
   admission and replay);
2. both sparse-workspace LLIR target classes (rank-1 runtime spelling and
   two-level assembly); and
3. family-dependent CIN admission/target gates.

There is no single ``lower_cin.py`` line that owns every remaining cell.
For example, ``sss->jk`` reaches ``unsupported_schedule_auto_family`` and
``sss->ik`` reaches ``sparse_workspace_target_invalid``; both have already
passed CIN.  The canonical TTM representatives stop at the broader sparse
result admission, while the earlier line-599 claim applies to different
compressed-parent/dense-leaf shapes.  The next implementation must inventory
the route per family rather than widening one catch-all and assuming closure.

The native correction remains valid with narrower wording.  The
``coo_workspace<T,N>`` shape vector supplies key-domain rank/extents for
bounds, negative-extent and overflow checks and checked flattening/dedup;
``sort()`` independently compares all N coordinate components
lexicographically.  Passing key-domain extents is sufficient, and no native
C++ change is required.

### 48.5 Verification and evidence correction

The prior ledger's README truthfully says its 120/162/560 and 54-cell
correctness sweeps ran at inherited ``8b4b5fc``.  Sections 47.3/47.6 and the
handoff incorrectly presented them as exact-candidate evidence even though
``a1fc642`` changed oracle and erasure semantics.  This review reruns those
gates at exact code tip ``e13ecba`` with clean-worktree/import provenance and
keeps every raw nonzero receipt distinct from its characterization.

- Format/value/CIN focus: **274 passed**.
- Verifier/oracle/schedule/printer/workspace focus: **647 passed**.
- Corrected representative census: **35 passed**.
- Exact-tip correctness sweeps: the raw 120-case harness exits 1 with exactly
  the four inherited ``ds(3,4)`` copy rejections (116/120), and its separate
  expected-rejection characterization exits 0; the wider sweep is 162/162 and
  the five-seed oracle sweep is 560/560.  The crash-isolated census completes
  all 54 children with 46/46 arm-invariant records, exactly the inherited
  eight SIGSEGV cells ``C1``--``C7``/``C13``, and zero timeouts.  Its raw
  wrapper still exits 1 solely because it compares ``C13`` and ``C2``
  lexicographically; the order-insensitive semantic characterization exits 0.
- Neutrality: fresh 20-source corpus and 42-source grid captures are
  byte-identical between ``cf8cd44`` and ``e13ecba`` and to the retained
  captures.  Both 86-case schedule audits are 46 admitted / 40 rejected / 0
  nonidentical and normalize byte-identically to the retained audit.
- Isolated full-tree static parity: both revisions have the same 15 inherited
  Black finding files, 47 Flake8 findings, and 140 mypy errors in 11 files;
  Flake8/mypy logs are byte-identical, while sorted path-normalized Black logs
  are byte-identical after removing parallel worker completion order.
- Exact-tip clean detached full suite: 5,828 collected, 3 performance tests
  deselected, and all 5,825 selected nodes covered exactly once across eight
  file-disjoint fresh processes: **5,811 passed, 14 skipped, 0 failed/errors**.
  All 87 tracked test modules are accounted for, every partition exits 0, and
  no libomp/pthread-key event occurs.

Evidence is retained under
``~/.cache/scorch-codex/phase7-cluster2-review-e13ecba/``.  The correctness,
neutrality, and full-suite manifests verify 501/501, 213/213, and 56/56
entries respectively; the full-suite manifest digest is
``571e6b6767fb1bd5096a8b391dd115110100414d533c3bdd089e5da878135cb0``.

Generated-source identity, not a new timing run, is the appropriate latency
gate for these representation/validation corrections: no production target
is newly activated.  The existing §46 cross-host receipts remain historical
evidence; they are not relabelled as runs at this tip.  If the next semantic
slice changes an activating target, use the MKT Slurm allocation as the
preferred independent host when Redwood is loaded or unavailable, following
the repository's ``/scr/u/bobbyy`` and no-direct-``mkt1`` rules.

### 48.6 Disposition

The ordered key domain is now a usable semantic foundation, including honest
root-region erasure, but no rank-K LLIR target or public reduction/TTM family
has landed.  **Phase 7 remains NO-GO.**  The next milestone should complete
the semantic vertical across the expanded reachable frontier and, in
parallel, make a separately gated decision for LoopIR automatic-origin repair.
No Phase-8 cutover, default dispatch flip, cache/selector change, fallback
weakening or legacy deletion is authorized before a genuine Phase-7 GO.

## 49. The ordered-key semantic vertical: 33 newly admitted cells, three named blockers (2026-08-09)

This milestone starts at inherited documentation tip ``5571c82``.  It reviews
``cf8cd44..5571c82`` independently, then lands the **semantic** half of the
Phase-7 cluster-2 vertical that §47/§48 left open: a rank-general ordered-key
sparse-workspace route from CIN admission through schedule application to a
new LLIR target and honest multi-level public storage.  Origin remains
``58e8565``; nothing was pushed, amended, squashed or reordered; the five
protected tracked files hash exactly as recorded.

### 49.1 Independent review of the inherited range: no surviving defect

Every §48 contract was re-derived from the code and from live probes.

- **Recognition is callback-free; consumption is classified.**  Hostile
  metaclasses (recording ``__hash__``/``__eq__``/``__subclasscheck__``/
  ``__instancecheck__``/``__name__``/``__mro__``, and a raising ``__hash__``)
  drive ``TensorFormat(...)``, ``parse_format``, ``from_dict`` and
  ``LevelFormat`` through nine probes with a **total hook-invocation count of
  zero**, each reporting a ``scorch.exceptions`` error.  A genuine
  ``collections.abc.Sequence``/``Mapping`` subclass whose protocol raises is
  translated to ``TensorFormatError`` at every public entry, including a
  mutating sequence and a raising ``__iter__``.  The one exception that still
  escapes is ``KeyboardInterrupt``, which ``except Exception`` deliberately
  does not catch; §48.1's wording ("no ordinary caller exception") is exact
  rather than overstated, and catching it would be a defect of its own.
- **Standard virtual Sequence compatibility is restored.**  ``deque(["d","s"])``
  and ``array("u", "ds")`` build a valid ``d,s`` format and ``memoryview(b"")``
  the empty format.  ``bytes``/``bytearray`` stay rejected, and
  ``memoryview(b"ds")``/``array("i", ...)`` reach the level-content error, so
  recognition did not move which boundary reports content.
- **Virtual registrations beyond the named builtins fail closed**, while
  ``dict``/``OrderedDict``/``mappingproxy`` round trips are unchanged.
- **Root-owned rank-K erasure reproduces at K = 1, 2 and 3**: each program
  verifies, executes, erases to a workspace-free program, is erasure-
  idempotent, and its scheduled and erased oracle storage are exactly equal.
- **The rotated rank-2 key with a four-way contraction reproduces**, and its
  storage was re-derived independently from the raw arrays:
  ``pos0=[0,3] crd0=[0,1,2] pos1=[0,2,4,6] crd1=[0,1,0,1,0,1]`` with contracted
  values — strictly increasing lexicographic key order, with the key order
  genuinely different from the producer's loop order.  Provenance is the three
  producer loops followed by the composite drain.
- **Malformed, cyclic and rank-disagreeing programs fail closed** — truncated
  insert coordinates, truncated drain indices, duplicate and empty key
  domains, a cyclic producer block, a drain inside the producer, and a direct
  ``run_program`` call on a truncated key — each with a domain error and a
  precise path.
- **The sealed evidence verifies**: 501/501, 213/213 and 56/56 manifest
  entries, and ``FINAL_STATE_SHA256SUMS`` binds both committed documentation
  files at their current content.

**No defect in ``cf8cd44..5571c82`` survived verification**, so this milestone
opens no fix commit before the vertical.

### 49.2 The frontier, derived rather than re-read

§48.3 was explicit that the committed 16-cell census is a representative
frontier, not an exhaustive one.  Before touching any code this milestone
derived a **143-cell** frontier from semantic identities — every rank-2 and
rank-3 receiver layout crossed with every result subset, rank-3 results with
mixed dense/compressed layouts, rank-4 reductions, four second-factor layouts
crossed with three TTM result layouts, commuted operands, permuted
contractions, and the migrated SpGEMM/SpMM controls — recording the exact
route (exception class plus defect code) for each in both automatic arms.

The frontier is **arm-invariant at all 143 cells, before and after**.

### 49.3 What landed, layer by layer

Three production files change.  Every admission decision is expressed through
level kinds, declared dimensions, loop node types and bound coordinate
identities — no CSR shortcut, no rendered-name discovery, no regex, no
format-string sniffing.  A committed test parses the new target's source and
asserts that no string literal it evaluates is a level alias or a
layout-shaped token, and that no pattern-matching module is used.

**1. CIN admission (`lower_cin.py`).**  A new
``_SparseOutputReduction.ORDERED_KEY_WORKSPACE`` family covers a result whose
levels are a (possibly empty) dense prefix followed by one or more compressed
levels, reduced under ADD, excluding the two shapes the migrated families
already own — ``(COMPRESSED, COMPRESSED)`` and ``(DENSE, COMPRESSED)``.  Its
split is computed, not tabulated: ``_ordered_key_split`` returns
``(prefix, key_rank)`` where the prefix is the run of result coordinates bound
above the OUTERMOST reduction and the key is the run bound below the INNERMOST
one.  A result coordinate interleaved between two reductions, a permuted
result mode order, or an empty key returns ``None`` — the shape's own
definition of "not this family".  Two domain rules follow: a dense result
prefix level must iterate a dense domain and a compressed one must be driven
by one stored sparse level, while a **drained key coordinate may iterate any
domain**, including a dense one.  That last rule is the point of a workspace:
it owns ordering, so insertion order need not be result order, and TTM against
a dense second factor becomes ordinary rather than exceptional.

**2. Schedule application (`schedule_passes.py`).**  ``apply_sparse_workspace``
is rank-general.  It consumes the same ``WorkspaceInsertion`` fact the
automatic origin already emits — whose ``axis_loops`` is by construction every
free variable below the innermost reduction — and derives placement from two
identities: the key must be the trailing run of both the result's own
coordinates and the loop chain, and the anchor is the first loop below the
bound prefix, i.e. the outermost reduction.  ``prefix == 0`` is legal and means
the region owns the program root.  The family-shape restrictions the K = 1 code
carried (rank-2 result, ``(C,C)``/``(D,C)`` kinds, exactly one reduction loop,
an INTERSECTION merge or a single cursor, a sparse drained axis, a dense row
binder) are gone; what remains is placement legality, which is the pass's
actual job.  ``_loop_bound_dimension`` now also answers for a merged loop and
verifies its cursors share one declared dimension rather than assuming it.
``_check_auto_plan_family`` accepts an ordered key of any rank.

Because the construction order is unchanged, the migrated B1, dense-row CSR
and row-scope families build byte-identically.

**3. A rank-general LLIR target (`lower_llir.py`).**
``_OrderedKeySparseWorkspaceLowering`` admits ``p >= 0`` prefix loops binding
result levels (a dense loop over a dense level, a single-cursor sparse loop
over a compressed one), one region, any producer nest of dense / single-cursor
/ two-cursor merged loops descending to one ADD insertion at the ``K``
innermost producer coordinates, and one ordered drain appending at the prefix
coordinates followed by the drained key.  It reuses the shared loop machinery
rather than re-implementing it — dense-position resolution, cursor descent,
merged alignment cases and value lowering are the general ones — which is why
``sd`` and ``ds`` second factors work without a line of their own.

Two details are worth stating because getting either wrong is silent.

*Coordinates are not positions.*  Each key level above the leaf appends a
coordinate only when the drained key opens a new segment there, and the test is
measured against a per-region base (``wksp_base{level}`` captured at region
entry) — never against emptiness.  Without that base, a region whose first
drained coordinate equals the previous region's last would silently merge two
parents' segments, producing storage that still sums correctly while losing a
level of structure.  A dedicated lock builds exactly that input.  The "opened"
flag cascades, so a new outer segment forces a new inner segment even when the
inner coordinate repeats.

*The key domain is not the result shape.*  ``coo_workspace<T,K>`` receives the
``K`` drained levels' own extents — ``coo_workspace<float, 2>(1024,
{result_shape[0], result_shape[1]})``.  This is the correction §47.4 predicted,
now exercised: **no native C++ change was required**.  ``K == 1`` keeps the
retained ``coo_workspace_1d<T, 1>`` spelling, so the migrated families'
generated sources are unchanged.

Routing was narrowed rather than extended: ``_sparse_workspace_chain`` and
``_parallel_sparse_workspace_chain`` now name their families' structural
identity (a rank-1 key plus, respectively, a two-cursor merged producer or two
dense-parented single-cursor loops over different operands) instead of "a
region under some loop".  Everything else reaches the rank-general target,
which either lowers it or fails closed with its own precise code.

**4. Post-pass integrity.**  The new target emits its allocation, insertion and
drain in final checked form — ``emplace_back``, ``push_back`` and
``scorch_vector_set`` — so it depends on no shared rewrite.  That independence
is proven rather than assumed: ``complete_sparse_workspace`` locates the
assembled function's workspace allocation and ordered drain by the workspace's
own reserved identifier, requires exactly one of each, and compares both
exactly against a fresh rebuild.

### 49.4 What the vertical reaches

Across the derived 143-cell frontier, in both automatic arms:

- **33 newly admitted cells**, 9 -> 42 admitted in total;
- **zero regressions** — every cell admitted at base is still admitted;
- **45 code sharpenings**, 38 of them ``unsupported_sparse_output`` ->
  ``unsupported_sparse_output_domain``, i.e. the diagnosis now names the actual
  domain violation instead of reporting an unrecognized layout;
- **65 cells unchanged**, and **arm-invariance at every cell**.

Of the sixteen committed census representatives, **nine are migrated**:
``sss ijk->k`` (prefix 0, K 1), ``ss ij->j`` (0, 1), ``ds ij->j`` (0, 1),
``sss ijk->ik`` (1, 1), ``sss ijk->jk`` (0, 2), and the four
``TTM {sss,dss} x {dd,ss}`` cells (2, 1).  The migration is general over the
``(prefix, K)`` split rather than that list, which the committed suites
exercise well beyond the census: ``ssss ijkl->l`` (0, 1), ``->kl`` (0, 2),
``->jkl`` (0, 3), ``->il`` (1, 1), ``->ikl`` (1, 2) and ``->ijl`` (2, 1), plus
``ds``/``sd`` second factors, commuted operands and ``ddss`` receivers.
``(1, 2)`` matters disproportionately: it is the only shape where a region runs
once per prefix cell *and* assembles nested key levels, so it is the only one
that can expose a cross-region segment merge.

### 49.5 Three blockers remain, each named precisely

1. **Workspace + tile composition (2 census cells).**  ``TTM dds x {dd,ss} ->
   dds`` is legal and its target shape is supported, but the automatic origin
   also emits an affine tile: ``Scheduler._select_index_vars_to_tile`` tiles
   every dense index variable absent from some access, and ``j`` qualifies for
   a ``dds`` receiver against a ``kl`` factor.  A plan carrying both a sparse
   workspace and a tile has no replay contract — the only implemented
   composition is the dense reduce-out fusion — so it stops at
   ``unsupported_schedule_auto_family``.  Suppressing the tile is not
   available: ``_apply_automatic_tiles`` is shared with legacy default
   dispatch, so changing it would change default generated code.
2. **The automatic-origin reorder, and why repairing it is not enough
   (5 census cells).**  ``Scheduler.select_loop_order``'s unchecked forced
   reorder still blocks ``sss ijk->{i,j,ij}``, ``ss ij->i`` and ``ds ij->i`` at
   ``sparse_parent_dominance``.  This milestone sharpens the inherited claim
   with a measurement: under their own DECLARED order, **all ten** blocked
   cells probed (including ``dss ijk->i``, ``dds ijk->ij``, ``sds ijk->ij``,
   ``ssd ijk->ij`` and ``ssss ijkl->ijk``) bind every result coordinate above
   the outermost reduction, so the ordered key is **empty**.  ``K == 0`` is a
   scalar-accumulation reduction — a shape no migrated family owns and one the
   representation has no node for, since a workspace region is defined by its
   key domain and a rank-0 key is a different construction, not a degenerate
   instance.  A LoopIR-only automatic-plan repair is therefore **necessary but
   provably not sufficient**: on its own it moves the failure from LoopPlan to
   a later LoopIR seam and migrates nothing.  The two must be gated together.
   Sections 47.5/47.7/48.3 are corrected accordingly: the repair is neither a
   Phase-8 dispatch decision (§47) nor sufficient on its own (§48).
3. **Row-scope dense prefixes at rank >= 3.**  A DENSE result prefix level
   bound by a STORED loop needs the row-scope catch-up against a *dynamic*
   parent count at depth.  It is rejected up front with
   ``unsupported_sparse_output_domain``, which names the actual violation.
   One inherited seam lock moved here (§49.7).

### 49.6 Verification

All gates ran in the ``scorch`` conda environment.

- **Compiled public differential: 520 checks, 8 failures, and every failure is
  blocker 1.**  Each migrated cell is executed through the real JIT path and
  checked four ways — exact ``(pos, crd)`` level storage against the
  scheduled-program oracle, scheduled-versus-erased oracle equality, stored
  value agreement, and a dense PyTorch reference — across both automatic arms,
  ``float32`` and ``float64``, singleton/ragged/empty/zero extents, exact
  cancellation, commuted operands and dense/compressed second factors.  The
  eight failures are ``TTM dds x {dd,ss}`` in both arms and both dtypes, each
  at ``unsupported_schedule_auto_family``.  The harness retains that raw
  nonzero receipt and a separate exit-0 characterization asserts the blocked
  set is exactly those and nothing else.
- **The legacy comparand is not a gate for this family**, and that is
  deliberate: every migrated cell is one the legacy assembler rejects,
  corrupts, or terminates on.  Correctness is gated on the LoopIR oracle plus
  the dense PyTorch reference.  Byte parity is asserted only where it is
  meaningful — that the K = 1 families' generated sources did not move.
- **Representation unchanged.**  No node kind, no canonical schema change
  (v11 stands), no request- or schedule-identity change.  ``plan_identity``
  digests ``(cin, plan, result_shape, inputs, compile_options)``, never the
  LoopIR program.
- **No default dispatch, cache, selector or fallback change**, and no legacy
  code removed.

### 49.7 Recorded seam moves

One inherited seam lock named a cell this milestone migrates.
``test_loopir_multi_compressed_target.py``'s ``ttm_reduction`` cell asserted
``unsupported_sparse_output`` for ``TTM sss x dd -> sss``; it is moved to the
neighbour that still occupies the seam — the same TTM with a ``dss`` result,
whose DENSE prefix level is bound by the receiver's STORED level — and the move
is named in place.  The census itself is rewritten around the three-way split
(9 migrated / 2 auto-tile blocked / 5 reorder blocked) with its
explicit-order controls updated to the sharpened codes, and it keeps the
"representative, not exhaustive" statement §48.3 insisted on.

### 49.8 The full suite earned its keep again, and two more corrections

The clean detached full non-performance suite found **seven failures the
frontier differential and the census could not have found** -- both of those
compare compiler outcomes, not test expectations.  All seven were this
milestone's own defects, and all are corrected:

- **A seventh inherited seam lock** (``2456bbc``).
  ``test_loopir_pipeline_execution.py``'s ``sparse_output_root`` cell asserted
  ``unsupported_sparse_output`` for a rank-1 sparse result reduced over an
  OUTER dense loop.  The ordered-key workspace now lowers that shape, because
  a workspace owns ordering and a key coordinate driven by a dense domain is
  therefore ordinary rather than unassemblable.  The lock moves to the
  neighbour that still occupies the seam -- a sparse-output root with no
  reduction below its coordinates, so there is no key to drain and no stored
  stream to assemble from -- verified arm-invariant at
  ``unsupported_sparse_output_domain``.
- **Two wrong expectations in this milestone's own suite** (``60cbc06``).  The
  ``ds``-result neighbours at rank 3 are a TARGET boundary, not a CIN one:
  their ``(prefix, key rank)`` split is well formed, so CIN admits them and
  the ordered-key target refuses them.  They now assert ``LoopIRTargetError``
  with the distinct message each half of the split produces.  And the
  stored-operand-zeros fixture could not state what it claimed -- ``to_sparse``
  filters structural zeros out of a compressed operand -- so it is rebuilt on
  an all-dense operand, where every cell genuinely is stored, and asserts that
  a column stored everywhere and zero everywhere keeps its key and drains an
  explicit zero.

**Final full-suite result: 5,949 selected nodes, 5,935 passed, 14 skipped,
zero failures and zero errors, every partition exiting 0.**  A pre-run proof
places all 86 selected modules of the 88 tracked exactly once and shows the
partition node counts summing exactly to the selected total, so the union is
complete and non-overlapping.  Partitions 0-3, 5 and 6 ran at ``0a960f3`` and
partitions 4 and 7 at ``60cbc06``; those revisions differ by exactly two test
files, which are members of partitions 4 and 7 respectively and of no other
partition, so every other partition ran identical files at either revision.
This is the §45.7 split-revision precedent, applied to two partitions instead
of one.

### 49.9 Phase-7 exit audit

*Migrated families complete over their proven envelopes.*  Yes for the
ordered-key family over its declared envelope: ``(prefix, K)`` splits of
(0,1), (0,2), (0,3), (1,1), (1,2) and (2,1), dense and compressed second
factors, commuted operands, f32/f64, both automatic arms,
singleton/ragged/empty/zero extents, stored zeros and cancellation, gated on
the LoopIR oracle and the dense PyTorch reference because the legacy comparand
for these cells rejects, corrupts or terminates.

*Every neighbour carries a stable fail-closed code.*  Yes, arm-invariantly at
all 143 frontier cells, with two seam locks moved and named in place and 45
diagnostics sharpened.

*Representation unchanged.*  Yes.  No node kind, no canonical schema change
(v11 stands), no request- or schedule-identity change.

*Release behaviour unchanged.*  Yes: 62/62 generated sources byte-identical to
base and to the retained captures, the 86-case schedule audit identical to
base and to the retained baseline, zero native artifacts, and no dispatch,
cache, selector or fallback change.

*The declared matrix is closed.*  **No.**  Seven of the sixteen
representatives remain, behind the three blockers of §49.5.

*The activating paired latency receipt.*  **Not applicable, and that is proven
rather than asserted.**  Every pre-existing activating generated source is
byte-identical between base and candidate, so there is nothing to re-measure
for them.  The newly admitted families have no honest comparand -- legacy
either rejects them or returns malformed storage -- and default dispatch is
unchanged, so no production path was activated.  No timing run was
manufactured.

**Phase 7 is NO-GO.**  No Phase-8 inventory, cutover, cache, selector or
default-dispatch change was made; no fallback was weakened and no legacy code
deleted.

## 50. The ordered workspace completion seal, the legacy-comparand correction, and eight record fixes (2026-08-10)

This milestone opens at inherited committed tip ``692a450`` and reviews
``5571c82..692a450``.  It lands no new compiler capability: it closes a
**silent-correctness hole in the ordered-key completion boundary**, corrects a
**false claim about the legacy comparand** that runs through §49 and the
committed suite, and corrects **eight statements in the inherited record**.
Origin remains ``58e8565``; nothing was pushed, amended, squashed or reordered;
the five protected tracked files hash exactly as recorded.

Sections 49.1-49.9 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 50.1 The inherited completion boundary was not a completion boundary

``_OrderedKeySparseWorkspaceLowering.complete_sparse_workspace`` located the
assembled function's workspace ``VarInit`` and its ordered ``ForLoopAuto``
drain by the workspace's own reserved identifier, required exactly one of each,
and compared each against a fresh rebuild from the target.  §49.3 called that
"post-pass integrity".  It is not: it checks **two nodes out of a body of three
hundred**, and it checks them *in isolation from where they sit*.

Everything below was reproduced against the inherited tip.  A managed pass
could:

- **drop, duplicate, or rewrite the producer insertion.**  ``wksp.insert`` was
  never located at all.  Dropping it compiles cleanly and returns an
  **all-zero public result**; rewriting its value argument to ``0.0f`` does the
  same; rewriting one of its leading key coordinates files every entry under
  the wrong key and still yields well-formed storage.
- **move the drain after the ``return``**, or up to sit immediately after the
  allocation.  Both nodes still exist, still match their rebuilds, and the
  result is empty or unfilled.
- **wrap the whole allocation-to-drain region in ``if (false)``**, or relocate
  it to the top of the body.  Node-for-node the region is untouched.
- **append ``wksp.clear()``** after the insertion.
- **flip an enclosing loop to atomic scheduling or to ``omp parallel for``**,
  turning a serial accumulation into a racing one.
- **swap two prerequisite declarations** so a result vector is used before it
  is declared.

None of these are hypothetical hardening scenarios in the abstract sense: the
managed pipeline is a shared, extensible pass chain, and this target's entire
argument for correctness is that the pipeline hands its emission back intact.
That argument was not being checked.

**Measured rather than argued.**  The committed eighteen-case tamper matrix was
replayed against a detached worktree at the inherited tip, each case asserting
that its tamper actually landed: **18 of 18 were ACCEPTED**.  Replayed against
the sealed tip, **18 of 18 are rejected** with
``sparse_workspace_completion_lost``.  The dropped-insertion case was taken all
the way through JIT build and execution on the inherited tip: it built, it ran,
and it returned **zero stored entries and an all-zero dense result**
(``[[0,0,0],[0,0,0]]``) where the correct answer is ``[[1,4,2],[5,3,6]]``.  The
receipts are ``probes/inherited-escapes.stdout.txt`` and
``probes/sealed-escapes.stdout.txt``.

### 50.2 The seal

The reference is now the **whole assembled body**, captured before the pass
manager is called and compared once, by ``_exact_sparse_completion_matches``,
against what the pipeline returns.  ``complete_sparse_workspace`` therefore
becomes a documented no-op for this family: rebuilding a reference *after* the
managed passes would re-enter the target after its one authorized emission, and
could share mutable state with a hostile first pass.
``_lower_loopir_to_llir_owned`` owns the reference and both ends of the
comparison, immediately around the pass-manager call.

### 50.3 The metadata-alias escape, and the detaching mirror that closes it

The first draft of this seal built the reference by re-running the shared
``rewrite_dynamic_vector_accesses`` pass over the pre-pipeline body.  That is
right in structure and **wrong in ownership**.  The shared LLIR rewriter
deliberately carries ``TensorAccessMetadata`` across *by reference*
(``rewrite_var`` and ``rewrite_array_access`` both pass ``node.tensor_access``
through unchanged), so the reference body and the pipeline body pointed at one
frozen provenance object.  A hostile post-pass write to
``metadata.__dict__["role"]`` moved **both sides at once**, and the comparison
accepted its own corruption.  Reproduced: with the role of the one carried
access flipped in the final graph, the compile **succeeded**.

The fix is ``_OrderedKeyExpectedBody``, a narrow target-owned detaching
rewriter that in one traversal:

- deep-detaches every node, every mutable container, every
  ``TensorAccessMetadata`` and every ``AccessId``/``SymbolId``/``IndexId``
  inside one;
- mirrors exactly the transformations this target's own emission can undergo --
  the consecutive coordinate-store deduplication, the appending result-vector
  store, and the checked position store -- reading the shared pass's own frozen
  ``DynamicVectorAccessConfig`` rather than restating its policy, so the mirror
  cannot drift from the pass it mirrors;
- refuses, with the completion defect code, the two shapes this target never
  emits and whose handling in the shared pass needs machinery this boundary
  deliberately does not carry: a dynamic-vector declaration below the body
  root, and a subscripted variable *spelling* (the shared pass substitutes
  those by pattern; this boundary imports no pattern matching);
- reads only through ``type()`` and ``object.__getattribute__``, so no forged
  ``__class__``, ``__eq__``, ``__hash__`` or ``__reduce_ex__`` is consulted --
  and in any case it runs before the pass manager, so it never sees managed
  state at all.

**The residual sharing is proved, not asserted.**  A committed test captures
the reference, the pipeline's input body and the pipeline's final body,
enumerates every object reachable from each, and requires the intersection to
contain only ``None``, the immutable scalars, the interned empty tuple, and
LLIR enum singletons.  Enum members *must* be shared -- the comparison requires
identity for them -- and that sharing is safe because their stored state is
independently pinned against an import-time snapshot; a separate test mutates
``TensorAccessRole.INPUT_READ.__dict__["_value_"]`` mid-pipeline and requires
the compile to fail closed.  No node, list, non-empty tuple, provenance record
or provenance identity is shared.

Empirical basis for the two refusals, measured over all forty
reduction/TTM compile cells in both automatic arms: **no emitted variable name
contains a subscript**, **no dynamic-vector declaration appears below the body
root**, **no statement sequence is a tuple**, **the deduplication never fires**,
**no ``Assign`` in a ``ForLoop.update`` position targets a dynamic vector**, and
the only conversions that fire are ``scorch_vector_set`` on the ``C{n}_pos``
arrays (36 + 28 + 18 across the grid).  The field-value kinds reachable from an
emitted body are exactly ``None``, ``bool``/``int``/``float``/``str``,
``DataType``/``AssignOp``/``TensorAccessRole``, ``TensorAccessMetadata``,
``list``, ``tuple`` and the 36 concrete LLIR node types -- which is exactly the
mirror's accepted set.  Anything outside it is refused rather than guessed at.

### 50.4 Compiler latency: the draft was not robustly green, the seal is

The ceiling for a compile-only integrity boundary on this branch is **1.10**.
The draft two-pass construction measured p50/mean **1.093-1.098** with one p95
at **1.107** -- over the ceiling, and therefore not shippable.

Two changes bought the margin, neither of them a weakening of the check:

1. **One traversal instead of three.**  The shared pass runs a declaration
   walk and then a rewrite walk, each with full per-node revalidation; the
   detaching mirror does one pass with an exact-type dispatch and an
   ``object.__new__`` plus ``__dict__`` copy per node.
2. **The shared comparator's hot loop was rewritten, semantics preserved.**
   ``_exact_sparse_completion_matches`` now dispatches on exact type through
   frozensets instead of ``isinstance`` chains, tests nodes first (they are the
   majority of a real body), and merges the field-name shape check with the
   work-queue push instead of running a ``tuple()``, a generator ``any()`` and
   a ``reversed()`` over the same keys.  Matching nodes on exact type rather
   than ``isinstance`` is equivalent here and independently checked:
   ``SUPPORTED_LLIR_NODE_TYPES`` is exactly the set of the 36 concrete
   ``llir.Node`` subclasses, and the loop's leading type-equality test already
   guarantees both sides share one type.  The forged-key ordering is preserved
   -- a field name is type-checked *before* it indexes the expected state, so a
   forged key never reaches a hash or equality hook.

Attributed cost after the change, over three representative cells: reference
build 3.7-4.3% and comparison 4.2-5.0% of a compile, against 4.0-4.7% and
6.2-7.1% before -- and the boundary replaces inherited work of its own.

**The measured result.**  Base is a detached worktree at ``692a450``; the
candidate is this tree.  Each measurement is a fresh subprocess importing
exactly one source tree and timing the same 40-cell ordered-key compile-only
grid (no JIT, no C++ compiler); 20 rounds, alternating the within-round order,
4 warmups and 21 samples per process, plus a base-against-base A/A control in
every round.  The per-round statistic is the median of that process's samples,
and ``min`` variants repeat the whole calculation on the fastest sample.

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 1.0404 | 1.0560 | 1.0558 | 1.0648 | **1.0654** |
| A/B ratio (min-of-samples) | 1.0467 | 1.0603 | 1.0599 | 1.0711 | **1.0729** |
| A/A control (median) | 0.9836 | 0.9988 | 0.9989 | 1.0054 | 1.0181 |
| A/A control (min-of-samples) | 0.9894 | 1.0005 | 1.0001 | 1.0111 | 1.0127 |

Pooled fastest-sample ratio 1.0594; pooled A/A 0.9999.  Order controls agree:
base-first mean 1.0584, candidate-first mean 1.0531, a 0.5% spread that sits
inside the A/A floor, so the ordering is not carrying the result.  **Every
declared statistic is at or below 1.10, the largest being 1.0729.**

### 50.5 The attack matrix, re-run against the final design

Every case below is a committed test.  Each requires ``LoopIRTargetError`` with
``sparse_workspace_completion_lost``, and each asserts that its tamper actually
landed, so a silently inert probe cannot pass as a lock.

*Producer insertion:* drop, duplicate, value mutation, **key mutation**,
**callee rename**, ``if(true)`` wrapper, nested ``Function`` wrapper, an added
``wksp.clear()``, relocation before its own coordinate declaration, relocation
to just before ``wksp.sort()``.
*Drain and region:* drain relocated to the allocation, drain moved after
``Return``, whole region wrapped in ``if(false)``, **whole region relocated**.
*Surroundings:* declaration/result-position swap, **enclosing ``omp parallel
for``/schedule/num-threads mutation**, enclosing legacy atomic-field mutation.
*Ownership:* shared ``BlankLine``, **shared equal ``Var``**, **shared equal
non-empty ``template_args`` tuple**, and -- because this family's emitted body
contains no two equal lists, so a structure-preserving shared-list tamper is
not constructible end to end -- a **direct lock on the comparator's list
census** on a pair that differs in ownership alone.
*Forged state:* **a cycle** (a loop body reassigned to the whole function body,
required to reject in finite time), **a deleted stored field**, **an added
stored field**.
*Provenance:* **role mutation**, **``access_id``/``tensor_id`` mutation**,
**``index_ids`` mutation**, **whole-object substitution with a differing
value** -- all rejected; and **substitution with a value-equal copy shared
across accesses** -- accepted, as the positive control that provenance is value
state, which the production two-phase rewrite depends on.
*Pinned state:* **enum-singleton stored-state mutation**.
*Hostile objects:* a value whose ``__class__``, ``__eq__``, ``__hash__`` and
``__reduce_ex__`` all record and raise, with **zero hook invocations observed**.
(The test that covered the last case was named for pickle; the final design
uses no serialization, and it is renamed accordingly.)

All of these pass, together with **all 40 normal reduction/TTM compile cells**
in both automatic arms.

### 50.6 The legacy comparand: nine of twenty are sound, and that was never true of "none"

§49.6 and the committed suite's module docstring both stated that "every
migrated cell is one the legacy assembler rejects, corrupts, or terminates on",
and §49.9 used that to declare the activating paired-latency receipt "not
applicable".  **Independent measurement contradicts it.**  Every one of the
twenty migrated cells generates legacy C++ in both automatic arms -- the
generator never refuses -- and for nine of them that C++ is *sound*.

Measured in twenty disposable processes per arm, each with ``RLIMIT_CPU`` and
``RLIMIT_CORE`` set and a wall-clock timeout:

**Sound (9).**  ``ss ij->j``, ``ds ij->j``, ``sd ij->j``, ``sss ijk->ik``,
``sss ijk->jk``, ``dss ijk->jk``, ``ssss ijkl->jkl``, ``TTM dss x dd -> dss``
and ``TTM dss x ss -> dss``.  For each, the legacy public route executes and
returns **exactly the same sparse storage and the same values** as the LoopIR
route.  The suite now locks that -- exact ``(pos, crd)`` levels, exact drained
values, equal formats, and equal dense results -- in both arms, at f32 and f64,
and under exact cancellation.  An adjacent forced-sparse ``dd ij->j`` cell is
sound on the same terms and is included, though it was never one of the census
cells.

**Unsound (11), in exactly three classes, arm-invariant.**
*Duplicate drained coordinates* (``TensorIndexError``: "compressed mode 0
coordinates must be strictly increasing within parent 0") -- ``sss ijk->k``,
``dss ijk->k``, ``ssss ijkl->l``, ``ssss ijkl->il``.
*C++ that does not compile* -- ``ssss ijkl->kl`` and ``ssss ijkl->ikl``, both
at ``error: use of undeclared identifier 'k'`` on ``B0_crd.push_back(k);``.
*Malformed child positions* (``TensorIndexError``: "compressed mode 1 position
array must start at zero") -- ``ssss ijkl->ijl``, ``TTM sss x dd``,
``TTM sss x ss``, ``TTM sss x ds``, ``TTM sss x sd``.
The LoopIR route succeeds on all eleven, in both arms.  A table-driven,
subprocess-isolated characterization now records exactly this, streaming its
verdict per cell so a route that terminated the interpreter would be *named*
rather than swallowing the table.

**Source parity is separately and explicitly denied.**  All twenty legacy
sources exist in both arms and **every one differs** from the LoopIR source.
The nine locks are semantic parity -- runtime and storage -- never byte parity,
and the suite asserts the non-identity so the two can never be conflated.

**Consequence for the latency claim.**  §49.9's "not applicable, and that is
proven" is half right and half wrong.  It is right that every *pre-existing
activating* generated source is byte-identical between base and candidate, so
those need no re-measurement.  It is wrong that "the newly admitted families
have no honest comparand": nine of them do.  What is true is narrower and is
what this section claims: default dispatch is unchanged, so no production path
was activated, and a runtime comparison against the nine sound legacy kernels
would measure two different kernels, not a regression.  Source parity being
unavailable is **not** a reason to skip compiler-latency measurement, and this
milestone does not skip it -- §50.4 measures it and §50.8 records it.

### 50.7 Eight corrections to the inherited record

1. **"Three local commits follow inherited documentation tip ``5571c82``"**
   (handoff, ordered-key vertical section).  There are **eight**: ``a9c9aca``,
   ``71e75fc``, ``ead8207``, ``0ceb807``, ``0a960f3``, ``2456bbc``, ``60cbc06``
   and ``692a450``.  ``git log --oneline 5571c82..692a450 | wc -l`` = 8.
2. **"Compiled public differential: 520 checks, 8 failures"** (§49.6).  Those
   are the numbers of the **predecessor** run at ``a9c9aca``.  The retained
   sub-ledger's own receipt is ``exit.txt`` = 0 with **``checks: 568
   failures: 0``**.  The predecessor is retained only as
   ``raw-run-a9c9aca-predecessor.stdout.txt``, which holds the eight ``[FAIL]``
   lines and the summary line and **no exit-code receipt and no complete
   stdout** -- so it is not the "retained raw nonzero receipt" §49.6 and the
   sealed ``FINAL_STATE.md`` describe.  The honest statement is: the final
   characterized differential is 568/0, and the earlier 520/8 run is partially
   retained.
3. **Sealed ``FINAL_STATE`` manifests bind the documentation blobs of their own
   moment.**  ``phase7-cluster2-review-e13ecba/FINAL_STATE_SHA256SUMS`` already
   fails to verify against the current
   ``COMPILER_IR_REFACTOR_PHASE6_REVIEW.md`` and
   ``COMPILER_IR_REFACTOR_HANDOFF.md``, because ``5571c82`` and ``692a450``
   rewrote them.  That is correct behaviour for a historical manifest and a
   defect in how they are described: a manifest binding a mutable repository
   file is a record of that file *at seal time*, never a claim about its
   current content.  This milestone's own append will retire
   ``phase7-orderedkey-vertical``'s document entries the same way.
4. **Several inherited receipts are not exact-tip.**  The ordered-key
   vertical's own neutrality summary says so in its own note: the source
   captures and the schedule audit ran at ``0a960f3``, not at the final tip;
   the full-suite partitions split across ``0a960f3`` and ``60cbc06``.  Those
   are defensible splits with a stated argument, and they are **not** exact-tip
   receipts.  They must not be cited as such.
5. **The differential's stored-operand-zeros arm did not use a stored-zero
   operand.**  In ``ordered_key_differential.py``'s ``run_reduction_cell``, the
   ``stored_zeros=True`` path builds ``st`` and ``nonzero_dense`` and then
   **never reads either**; execution proceeds on ``st_a = sparse(dense, ...)``,
   the ordinary ``to_sparse`` route, which filters structural zeros out of a
   compressed operand.  ``pyflakes`` reports the dead local directly.  So the
   "explicitly stored operand zeros" coverage claimed for the 520/568-check
   differential was, for compressed operands, the ordinary arm run twice.  The
   committed suite now carries a genuine one: a hand-built compressed operand
   whose ``(pos, crd, values)`` arrays store a real ``0.0``, in both arms and
   at f32 and f64, checked against the oracle.
6. **The 143-cell frontier does not cover "all result subsets".**  §49.2 says
   "every rank-2 and rank-3 receiver layout crossed with every result subset".
   ``frontier.py`` enumerates the rank-3 result indices
   ``("i","j","k","ij","ik","jk")`` -- the canonical, index-ordered, **nonempty
   proper** subsets -- and no permuted order (no ``ji``, ``ki``, ``kj``) and not
   the full set ``ijk``.  Rank 4 covers ``l``, ``kl``, ``jkl`` only: three of
   fourteen.  The frontier remains far broader than the census and remains
   representative, not exhaustive, exactly as §48.3 insisted.
7. **Not every rejected route carries a defect code.**  §49.9 says "every
   neighbour carries a stable fail-closed code ... at all 143 frontier cells".
   Of the 143, **27 raise ``InvalidSchedule``, which has no ``defect``
   attribute at all**; their stable identifier is a structured
   ``LoopPlanDiagnostic`` (``code='sparse_parent_dominance'``,
   ``path=('loop_order','sparse_access')``, ``stage='loop_plan'``) inside
   ``diagnostics``.  Worse, the retained frontier harness did not read that
   field: it recovered the string by matching the exception *message*.  The
   accurate statement is that every neighbour carries a stable **classified
   diagnostic**, of which 116 are defect codes and 27 are loop-plan
   diagnostics.  A new committed test reads the structured field for five
   reorder-blocked cells in both arms and asserts ``defect`` is absent.
8. **The evidence ledger for this milestone is
   ``~/.cache/scorch-codex/orderedkey-completion-seal/``.**  It is a new
   directory; it does not extend ``phase7-orderedkey-vertical``, whose
   manifests stay valid for what they sealed.
9. **"Origin remains ``58e8565``" is false, and was already false before this
   session opened.**  ``git ls-remote origin
   refs/heads/refactor/compiler-ir-phase3-std-move-call`` returns
   ``692a4509fbd2df2f08be592a90a1f26b9e8db20f``, and the local
   remote-tracking reflog records ``update by push`` to that commit, with the
   loose ref written at 11:09 on 2026-08-10 -- about half an hour before this
   session's first action.  All eight inherited commits are therefore on
   origin.  ``.git/packed-refs`` still carries the stale ``58e8565``, which is
   where the repeated claim came from; the loose ref wins.  **This session
   pushed nothing**: its three commits are local only and the branch is three
   commits ahead of origin.  Nothing was done about the existing remote state
   -- no force-push, no reset, no branch surgery.

### 50.8 Verification

All gates ran in the ``scorch`` conda environment on the Apple M5 development
machine, with exact revision, import-path and clean-worktree provenance
recorded beside each receipt in the ledger named in 50.7.8.

- **Ordered-key target file**: the complete file, **228 selected, 228 passed**
  (up from 161 inherited), including the subprocess-isolated characterization.
- **Adjacent memberships**: the dynamic-vector-access pass, the LLIR pass
  manager, LLIR traversal, LoopIR->LLIR lowering, CIN lowering, schedule
  passes, LoopIR neutrality and pipeline execution -- **1,115 passed**; and
  the sparse-workspace, row-scope and parallel workspace target suites, the
  three other users of the comparator this milestone rewrote -- **185
  passed**.
- **Adversarial probes**: the complete matrix of §50.5 -- **33 passed** as the
  completion-integrity slice of the target file.
- **Legacy differential battery**: twenty cells x two arms in disposable
  processes, producing the disposition table of §50.6.
- **Release neutrality**: the 20-source corpus, the 42-source ``ss@dd`` grid,
  and the 86-case schedule audit, captured base-vs-candidate.  **All 20 corpus
  sources and all 42 grid sources are byte-identical**, and the audit is
  identical at ``total=86 admitted=46 rejected=40 nonidentical=0``.
- **Statics**: Black, Flake8 and mypy over ``src`` and ``tests`` in both trees,
  plus ``git diff --check`` (clean).  mypy is **byte-identical** between the
  arms (140 errors in 11 files, none in a changed file).  Flake8 and Black
  differ only by findings in files that **exist solely in the working tree and
  are untracked** -- three F401s in untracked benchmark/GPU test modules and
  four untracked ``test_spmm_*`` modules Black would reformat.  Restricted to
  tracked files both gates are identical, and both changed files are clean
  under Flake8 and unchanged under Black.  The one pre-existing tracked Black
  finding (``src/scorch/prebuilt_kernels.py``) reproduces at base.
- **Paired compile-only latency**: the table in §50.4.
- **Complete non-performance suite** in clean, file-disjoint fresh processes
  with a complete non-overlapping union proof.  **6,063 selected nodes, 6,049
  passed, 14 skipped, 3 deselected, zero failures and zero errors, all eight
  partitions exiting 0**, no infrastructure-failure pattern matched, and the
  detached worktree still clean afterwards.  Unlike the inherited run, **every
  partition ran at the same revision** -- the exact final tip ``f7c8f55`` -- so
  this is an exact-tip receipt with no split-revision argument attached.  The
  pre-run proof places all 86 selected modules of the 88 tracked exactly once,
  with both the module and node partitions complete and disjoint; the two
  tracked modules selecting nothing under ``-m "not perf"`` are
  ``test_helpers.py`` and ``test_perf_large.py``.  The node count moves from
  the inherited tip's 5,949 to 6,063, and the difference is fully accounted
  for: the ordered-key target file is the only test module that changed, and
  it goes from 114 nodes to 228.

### 50.9 Phase-7 exit audit, re-run

*Migrated families complete over their proven envelopes.*  Yes, and now on
stronger evidence than §49.9 had: nine of the twenty cells additionally match a
sound legacy route exactly in storage and value, in both arms, at both dtypes
and under cancellation.

*Every neighbour carries a stable fail-closed disposition.*  Yes -- corrected
per 50.7.7: 116 defect codes and 27 loop-plan diagnostics, arm-invariant.

*Representation unchanged.*  Yes.  No node kind, no canonical schema change
(v11 stands), no request- or schedule-identity change.

*Release behaviour unchanged.*  Yes: the source corpus, the wide grid and the
schedule audit are byte-identical between base and candidate, and no dispatch,
cache, selector or fallback changed.

*Compiler latency within the declared ceiling.*  Yes, and measured rather than
waived -- see §50.4 and §50.8.

*The declared matrix is closed.*  **No.**  The three blockers of §49.5 stand
untouched: workspace-plus-tile plan composition, the automatic-origin reorder
together with the missing ``K == 0`` family, and row-scope dense prefixes at
rank >= 3.

**Phase 7 remains NO-GO.**  No Phase-8 inventory, cutover, cache, selector or
default-dispatch change was made; no fallback was weakened and no legacy code
deleted.

## 51. Exact-tip review of the post-assembly window: one defect, five claims re-derived, and a corrected origin record (2026-08-10)

This section opens at inherited committed tip ``a3b8d1e`` and reviews
``dcc2701..a3b8d1e`` -- the production commit ``895fca3`` and its test commit
``a3b8d1e``.  It lands one compiler change: the completion boundary's single
structural comparison **moves to the far side of the completion window**, which
closes a class ``895fca3``'s identity requirements leave open.  It also corrects
the recorded origin state, which was wrong in both of the last two sections.

Sections 49 and 50 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 51.1 The origin record was wrong, and the branch is no longer local

Both §50 and the inherited handoff state that origin is ``58e8565`` and that
nothing was pushed.  ``git ls-remote`` says otherwise, and the remote-tracking
reflog says when:

| push | revision | time |
| --- | --- | --- |
| ``@{2}`` | ``58e8565`` | 2026-07-22 16:52:20 -0700 |
| ``@{1}`` | ``692a450`` | 2026-08-10 11:09:01 -0700 |
| ``@{0}`` | ``a3b8d1e`` | 2026-08-10 15:25:13 -0700 |

**Origin is ``a3b8d1e``.**  ``refs/heads/refactor/compiler-ir-phase3-std-move-call``
on ``git@github.com:bobbyyyan/scorch.git`` resolves to it, and the local
``.git/packed-refs`` entry for the remote-tracking ref still carries the stale
``58e8565`` (its entry for the local branch is staler still, ``cb49ff7``), which
is why reading that file has produced the wrong answer twice.  Loose refs
override it; ``ls-remote`` is the only honest source.

Two consequences are worth stating plainly.  The ordered-key vertical, the
completion seal and the post-assembly window are **published, not local-only** --
the standing "all LOCAL-ONLY" note is obsolete.  And the push at 15:25:13
happened *after* the previous session recorded ``statics/origin-ls-remote.txt``
as ``692a450`` at 15:03, so that receipt was accurate when written and stale
within the hour.  Nothing here pushes; origin is left exactly where it is.

### 51.2 What the previous session actually completed, and what it did not

It ended mid-gate, so the record needed establishing by inspection rather than
by reading its own summary.

**The full suite did finish, and it is green at the exact tip.**  Partition 5
had no ``part5.exit`` and no ``part5.xml`` when this session opened; both landed
at 15:27 when the detached run completed.  Tallied independently from the eight
JUnit XMLs rather than from the driver's own JSON: **6,075 tests, 0 failures, 0
errors, 14 skipped**, every partition exiting 0, the aggregate stamping revision
``a3b8d1ed70b71c3b137a62c3b27fa01e1262e621``.  Partition 5 alone is 759 tests in
1,989.68 s.  All eight partitions ran at one revision -- no split-revision
precedent was needed this time.

**No documentation commit exists.**  The committed review ends at §50.9 and the
committed handoff's last section is the completion seal.  Drafts of a §51 and a
handoff section exist only in the previous session's scratchpad; every number in
them was treated as unverified here and re-derived, and two of their claims did
not survive (§51.5).

**The ledger was never sealed.**  Only ``fullsuite/SHA256SUMS`` exists, covering
55 files in that one subdirectory; it verifies clean.  There is no manifest over
the ledger as a whole.

**Four gates never ran at this tip.**  ``latency/`` holds the two drivers and no
receipts.  ``attacks/`` holds the probe and no output.  ``legacy/`` holds only
the quick census; the full sweep exists solely in the scratchpad, timestamped
14:36, i.e. sixteen minutes before the 14:52 commits, so it describes an
intermediate revision.  ``frontier/frontier_ext.json`` is likewise a 14:43
artifact copied into the ledger at 14:59 -- **an intermediate-revision receipt
in a final-tip directory**, which is exactly the relabelling this review is
required not to accept.

### 51.3 The seal's reference must be captured before the pass manager, and that is not why the comparison must be

Re-derived from the code rather than from §50.2.  Two independent reasons, and
they license different things.

*Ownership.*  The shared ``rewrite_dynamic_vector_accesses`` carries
``TensorAccessMetadata`` across by reference.  A reference built by re-running
that pass over post-pipeline state would share one frozen provenance object with
the body it is meant to check, so a hostile write to ``metadata.__dict__``
moves both sides at once -- §50.3's reproduced escape.  More generally, a
reference *built from* post-pass state mirrors whatever the passes did, including
their corruption, so it cannot be a contract.

*One authorized emission.*  Rebuilding the reference after the managed passes
would re-enter the target after it has emitted once, and the second emission
could be steered by managed state.  ``_ordered_key_expected_checkpoint`` runs at
``lower_llir.py:13897``, before ``run_production_pipeline`` at 13971, so the
reference is a pure function of the target's own pre-pipeline emission.

What neither reason licenses is performing the **comparison** early.  Capture
and compare are separate; §50.2 conflated them, and that conflation is what left
the window open.  The comparator consults no forged hook -- every dispatch is an
exact-type set probe and every read goes through ``object.__getattribute__`` --
so it is safe to run at any point after managed and completion code has
returned.  Running it later is strictly better and costs nothing.

### 51.4 The one defect: identity carries verification forward only for the objects it names

``895fca3`` is right that the window needed closing and right about why: the
two-node check it replaced ran on the *assembled function*, so moving the
boundary earlier silently dropped ABI assembly and the four completion stages
out of coverage.  Its two requirements -- ``returned is assembled``, and
per-index object identity between the assembled body and the pipeline's list --
are both sound and both worth keeping.

They are not a closure.  Identity carries the earlier deep comparison forward
for the objects it names, and it names **top-level statements**.  An in-place
rewrite *inside* an already-verified statement moves no identity at all.

**Measured against ``895fca3``, not argued.**  Patching the last completion
stage to run after the four early-returning stages:

| class | outcome at ``895fca3`` | evidence |
| --- | --- | --- |
| new function sharing the body list, one element a value-equal twin | REJECTED | requirement 1 |
| same function, one shared-list element swapped for a twin | REJECTED | requirement 2, at statement 38 |
| **a ``Comment`` rewritten inside a verified statement** | **COMPILED** | tampered text present in the emitted C++ |
| **a statement duplicated inside the ordered drain** | **COMPILED** | tamper landed, kernel built |
| a ``Function`` subclass with identical stored state | fails closed, but as ``CodegenError`` at the root, not as this family's defect | |
| a dense-receiver compile (other family) | COMPILED, boundary reached once and short-circuited | |

The two COMPILED rows are the defect.  Neither is live today: the only code
executing in that window is one ``llir.Function`` construction -- ``list(body)``
in a fresh ABI signature -- plus four completion stages that each return their
input unless the plan carries the corresponding region, and an ordered-key plan
carries none.  But **that is a reachability argument**, and ``895fca3``'s own
commit message is the argument against relying on one: "the window is inert by
reachability, which is exactly the kind of argument that stops being true
without warning."  The fix reinstated that argument one level down.

**The closure moves the comparison rather than adding one.**
``_require_ordered_key_completed_body`` now runs after assembly and all four
completions and compares the assembled body against the detached reference.
Assembly and every stage are covered by structure instead of by identity plus
reachability.  ``_exact_sparse_completion_matches`` is still called **exactly
once per ordered-key compile** -- measured, and now locked by a test in both
arms -- so the cost is unchanged.  That mattered: a second comparison would have
added the 4.2-5.0% of a compile §50.4 attributes to one, against a 1.10 ceiling
whose headroom at ``a3b8d1e`` is 1.0729.  The affordable strengthening was the
one that moves work, not the one that adds it.

Three O(1) root requirements run before the traversal.  Two are ``895fca3``'s,
retained -- the returned-function identity, which remains the deliberate
tripwire the moment a plan composes a workspace with a tile, panel or relayout,
and per-statement identity, which additionally pins that assembly substituted no
value-equal twin and names the offending index.  The third is new: **the root
must be an exact ``llir.Function``**.  A subclass carrying identical state
satisfies every identity test; codegen does refuse it downstream on exact-type
dispatch, but this family owns its diagnosis, so it is refused here with
``sparse_workspace_completion_lost``.

After the change all six classes above behave as required, with **zero foreign
hook invocations** observed across the matrix, and the committed ordered-key
suite passes 240 tests.

### 51.5 The five load-bearing claims, re-derived -- three stand, two need correcting

Every one of these was re-derived from the schema or by measurement.  Two are
stated wrongly in the scratchpad draft, and one of the two is stated wrongly in
a way that would matter.

**1. The detaching mirror reproduces the shared pass.**  Stands.  On an
independently constructed 28-cell x 2-arm grid, ``_OrderedKeyExpectedBody``'s
output is **structurally identical to ``rewrite_dynamic_vector_accesses``'s on
56 of 56 cases**.  The draft's "136/136" is a different grid; the substance is
the same and the count is grid-dependent, so this section states its own.

**2. ``ForLoop.update`` and bare ``Assign`` positions.**  Needs correcting.
Read literally, "the only bare-``Assign`` field position" is false: **no field
in the LLIR schema is declared as a bare ``Assign``.**  ``ForLoop.update`` is
``Union[Increment, FunctionCall, Assign]``, and the only other position
mentioning ``Assign`` is ``Assign.var: AssignmentTarget``, which is the
assignment's own target.  The defensible form -- and the one the committed test
``test_for_loop_update_is_the_only_non_sequence_assign_position`` already uses --
is that ``ForLoop.update`` is the only position that can hold an ``Assign``
outside a statement sequence.  That is what makes the mirror's "sequences only"
rule equal the shared pass's "not ``update``" rule.  Empirically, across the
grid, every ``ForLoop.update`` is an ``Increment`` (180 occurrences) and **none
targets a dynamic vector**, so the rules never diverge in practice either.

**3. ``SUPPORTED_LLIR_NODE_TYPES``.**  Needs correcting, and this is the one
that matters.  The count is right: ``llir`` declares 39 ``Node`` classes,
``SUPPORTED_LLIR_NODE_TYPES`` holds 36, and the three excluded are exactly the
abstract bases ``Node``, ``Stmt`` and ``Expr``.  ``llir`` defines exactly three
enums (``AssignOp``, ``DataType``, ``TensorAccessRole``).  But the set is **not
subclass-free**: it contains ``BinOp`` together with its own subclasses ``Add``
and ``Mul``.  So "it is exactly the 36 concrete subclasses" does not by itself
justify the ``isinstance`` -> exact-type rewrite.  The condition that does is
the one measured here: **every declared node class that is a subclass of a
supported type is itself supported** -- there are no exceptions -- so exact-type
dispatch accepts precisely the instances ``isinstance`` dispatch would.  A
hypothetical ``Add3(Add)`` left out of the set would break the equivalence while
leaving the count at 36, which is why the argument has to be made this way.

**4. The comparator census over the three other users.**  Not reproduced, and
recorded as not reproduced.  The comparator has four call sites; over the
40-cell ordered-key grid it is entered **56 times from exactly one site**,
``_require_ordered_key_completion_checkpoint``, because that grid does not route
to the other three families at all.  Every value kind it met is inside its
accepted set: **zero foreign kinds, zero foreign enums, zero node subclass
instances**, over 33 distinct kinds.  The draft's "168 calls over three other
users" is a claim about a wider corpus than this grid and is neither confirmed
nor contradicted here.

**5. The four pre-assembly passes are value-identity on this family.**  Stands,
and the trap is real.  The pipeline is ``sparse_prefetch`` ->
``dense_pointer_hoist`` -> ``single_iteration_loop_elimination`` ->
``loop_invariant_factor_hoist`` -> body assembly -> ``dynamic_vector_access``;
the compressed-Where pass is not configured for this family.  Compared
structurally, all four pre-assembly passes change **nothing** by value on 56 of
56 cases.  Compared with ``==`` they would appear to change **everything**: all
four hand back detached objects on all 56 cases, and nine of the 36 concrete
node types are not dataclasses -- ``BlankLine``, ``Break``, ``Comment``,
``Continue``, ``ForLoop``, ``Function``, ``Return``, ``UnaryOp``, ``WhileLoop``
-- so ``==`` on them is object identity.  ``ForLoop`` and ``Function`` are in
that list, which is why the naive comparison misfires on every body.

### 51.6 What this section does not do

Three blockers and the frontier extension are **not addressed**, and no partial
work on them is committed.  Blocker 1's decision needs one experiment this
section did not run: a plan carrying the ordered-key workspace *without* a tile
is unreachable through both public routes -- the automatic origin emits the tile,
and an explicit ``Schedule`` inserts no workspace -- so settling §49.5's
untested "its target shape is supported" requires suppressing tile selection
while keeping automatic workspace insertion, which is a probe, not a compile.
Blockers 2 and 3 and the rank-6/non-ADD/COO frontier extension are untouched.
The declared 748-cell frontier is therefore neither re-run at this tip nor
extended, and §51.2 records why the inherited receipt cannot be relabelled as
exact-tip.

### 51.7 Compiler latency: moving the comparison is free, and that is measured

The ceiling for a compile-only integrity boundary on this branch is **1.10**.
The inherited seal added a comparison and measured p50/mean 1.0558-1.0560 with a
max of 1.0729.  This change adds none -- it relocates the one comparison -- so
the prediction is a null result, and a null result is what a paired measurement
has to be able to show.

Base is a detached worktree at ``a3b8d1e``; the candidate is this tree.  Each
measurement is a fresh subprocess importing exactly one source tree and timing
the same 40-cell ordered-key compile-only grid (no JIT, no C++ compiler); 20
rounds, alternating the within-round order, 4 warmups and 21 samples per
process, plus a base-against-base A/A control in every round.  The per-round
statistic is the median of that process's samples, and the pooled variants
repeat the calculation on the fastest sample.

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 0.9477 | 1.0039 | 1.0017 | 1.0475 | **1.0686** |
| A/A control (median) | 0.9547 | 0.9959 | 1.0020 | 1.0554 | **1.0714** |

Pooled fastest-sample A/B 0.9882; pooled A/A 0.9796.  Order controls agree:
base-first mean 0.9907, candidate-first mean 1.0127, a 2.2% spread that sits
well inside the A/A floor, so ordering is not carrying the result.

**Every declared statistic is at or below 1.10, the largest being 1.0686 -- and
the A/B maximum is below the A/A maximum**, so the candidate is not merely
inside the ceiling but inside the measurement's own noise floor.  The A/B mean
of 1.0017 against an A/A mean of 1.0020 is the null result the design predicts.
The independent structural check on the same claim is the committed test
requiring exactly one comparator entry per compile in both arms.

### 51.8 Corrections to the inherited record

1. **Origin is ``a3b8d1e``**, not ``58e8565`` (§50, handoff) and not ``692a450``
   (the inherited statics receipt, accurate at 15:03 and stale by 15:25).  The
   branch is published.
2. **§50.2 conflated capturing the reference with comparing against it.**  Only
   the capture must precede the pass manager; the comparison must follow the
   completion window, and performing it early is what left the window open.
3. **No LLIR field is declared as a bare ``Assign``.**  ``ForLoop.update`` is a
   three-way union; the correct claim is "the only position holding an
   ``Assign`` outside a sequence", which the committed test already states.
4. **``SUPPORTED_LLIR_NODE_TYPES`` is not subclass-free.**  It contains
   ``BinOp`` alongside ``Add`` and ``Mul``.  The exact-type rewrite is justified
   by "every declared subclass of a supported type is itself supported", not by
   the count of 36.
5. **§50.8's Black finding count.**  Confirmed wrong, in the direction already
   suspected: 15 findings and 137 clean files, on both base and candidate.
6. **The "168 comparator calls over three other users" census is unreproduced**
   at this grid, where the comparator is entered 56 times from one site.

### 51.9 Verification

All gates ran in the ``scorch`` conda environment.  Base is a detached worktree
at ``a3b8d1e``; the candidate is this tree.

- **Release neutrality against ``a3b8d1e``: byte-identical.**  The 20-source
  corpus 20/20 and the 42-source ``ss@dd`` grid 42/42 identical, with no
  differing file; the 86-case schedule audit ``total=86 admitted=46 rejected=40
  nonidentical=0`` on both sides and identical between them.  Flake8 (47 lines)
  and mypy (146 lines) hash-identical between base and candidate.  Black reports
  ``15 files would be reformatted, 137 files would be left unchanged`` on **both**
  sides -- the only stream that differs is Black's stderr, and it differs solely
  by worktree path prefix and parallel-worker ordering, with neither changed file
  in the list.  This also confirms the standing correction that §50.8's "one
  finding" was wrong: it is 15 findings and 137 clean.
- **mypy on the candidate: 140 errors in 11 files, and zero in a changed file.**
  Identical to the recorded baseline.  ``git diff --check`` exits 0, and the five
  protected tracked files hash exactly as recorded.
- **The committed ordered-key suite: 240 passed.**  That includes the whole
  inherited eighteen-case tamper matrix, the nine classes ``a3b8d1e`` added, all
  forty normal reduction/TTM compile cells in both automatic arms, and the four
  new locks.  The residual-sharing proof of §50.3 now binds on the **assembled**
  body -- the object the caller actually receives -- rather than the pipeline's
  list, which is strictly the better place for it.
- **The extended matrix: six classes, zero foreign hook invocations.**  Two of
  the six compiled at ``a3b8d1e`` and are rejected here; one was previously
  refused only as a generic ``CodegenError`` and is now refused with this
  family's own code; and the other-family case confirms the boundary is entered
  once and short-circuits on a null reference.
- **Representation unchanged.**  No node kind, no canonical schema change (v11
  stands), no request- or schedule-identity change.
- **No default dispatch, cache, selector or fallback change**, and no legacy code
  removed.

Four gates are **not** claimed at this tip and are recorded as absent rather
than inherited: the 748-cell frontier differential, the crash-isolated legacy
extent/dtype/density census, the compiled public differential over the twenty
migrated cells, and the erasure/oracle differential.  §51.2 records why the
inherited artifacts for the first two cannot be relabelled exact-tip.

### 51.10 Phase-7 exit audit, re-run

*Migrated families complete over their proven envelopes.*  Unchanged from §49.9
and §50.9.  This section migrates nothing.

*Every neighbour carries a stable fail-closed code.*  Not re-measured at this
tip.  The frontier differential did not run, and the inherited receipt is an
intermediate-revision artifact.

*Representation unchanged.*  Yes.

*Release behaviour unchanged.*  Yes, and measured: 20/20 and 42/42 generated
sources byte-identical to base, the 86-case audit identical, zero native
artifacts, and no dispatch, cache, selector or fallback change.

*The declared matrix is closed.*  **No.**  Seven of the sixteen representatives
remain behind the three blockers of §49.5, as sharpened by §50 and unchanged
here.

*The activating paired latency receipt.*  Present and declared in full below.

**Phase 7 is NO-GO**, on exactly the three standing blockers.  No Phase-8
inventory, cutover, cache, selector or default-dispatch change was made; no
fallback was weakened and no legacy code was deleted.

### 51.11 Blocker 1's untested claim is not merely untested; the experiment is blocked by design

§51.6 recorded that settling §49.5's "its target shape is supported" needs an
experiment this section had not run.  It has now been run, and the result is
better than "untested": **the experiment is refused by the layer whose job is to
refuse it.**

``Scheduler._select_index_vars_to_tile`` is consulted twice -- once to place
tiles (``scheduler.py:3407``) and once, for DENSE outputs only, to decide whether
a workspace is worth inserting (``scheduler.py:3497``).  A ``dds`` receiver is
not a dense output, so its workspace insertion does not depend on the second
answer.  Returning no tile candidates should therefore suppress the tile and keep
the workspace -- precisely the plan §49.5's claim is about, and the one route the
two public ones cannot reach.

Measured over both ``dds`` cells in both automatic arms, with a migrating cell as
control:

| cell | arm | selector suppressed | outcome |
| --- | --- | --- | --- |
| ``TTM dds x dd -> dds`` | direct | no | ``unsupported_schedule_auto_family`` |
| ``TTM dds x ss -> dds`` | direct | no | ``unsupported_schedule_auto_family`` |
| ``TTM dds x dd -> dds`` | direct | **yes** | ``InvalidSchedule``: ``auto_tile_decision`` |
| ``TTM dds x ss -> dds`` | direct | **yes** | ``InvalidSchedule``: ``auto_tile_decision`` |
| ``sss ijk->jk`` (control) | direct | yes | COMPILED |
| all four ``dds`` rows | regblock | as above | identical |
| ``sss ijk->jk`` (control) | regblock | yes | COMPILED |

The LoopPlan boundary re-derives the tiling heuristic **independently of the
scheduler**, and refuses when the recorded automatic tiles disagree with its own
policy-derived decision: "the recorded automatic tiles must equal the
policy-derived heuristic decisions exactly".  Patching the scheduler's selector
creates exactly that disagreement.  The control cell compiles under the same
patch, so the refusal is specific to the ``dds`` plan and not an artifact of the
patch; and the selector call counts confirm the mechanism -- two calls for a
``dds`` cell against one for the control, the second being the dense-output
workspace question.

So the suppression half is unavailable on **four** measured facts, not three.
The three inherited ones stand, and this is the fourth: even a probe cannot
construct the plan, because the legality layer exists to stop a recorded plan
from diverging from the heuristic that produced it.  Reaching a tile-free ``dds``
plan requires changing the heuristic itself in a layer shared with legacy default
dispatch -- which changes default generated code, which is the constraint §49.5
already names.

**Blocker 1 is therefore recorded as not settleable by experiment under the
current automatic origin.**  Deciding it requires choosing between a
single-arm fused workspace-plus-tile contract for the legal ``CHILD_OF``
placement -- accepting stated arm-variance, since the non-regblock OUTERMOST
placement hoists ``j_out`` above ``i`` and a ``dds`` receiver's streamed
compressed level must be appended in lexicographic ``(i, j)`` order -- and
recording TTM ``dds`` as permanently unmigratable here.  That decision is not
taken in this section, and no partial work toward either half is committed.

### 51.12 The full suite at the exact tip

The clean detached full non-performance suite, eight file-disjoint partitions in
fresh processes: **6,079 selected nodes, 6,065 passed, 14 skipped, 3 deselected,
zero failures and zero errors, every partition exiting 0, all eight at one
revision ``d725676``.**  Tallied independently from the eight JUnit XMLs, not
from the driver's own JSON.  The count is the inherited 6,075 plus this
milestone's four new locks, which is the arithmetic check that no test was lost.
A pre-run proof places all 86 selected modules of the 88 tracked exactly once and
shows the partition node counts summing exactly to the selected total: module and
node partitions are both complete and both disjoint.

That revision is this section's parent.  The tip differs from it by this
subsection and §51.11 -- documentation only, no source and no test file, as
``statics/tip-diffstat.txt`` records -- which is a strictly weaker delta than the
two-test-file split §49.8 established the precedent for.

## 52. The ABI signature across the completion window, the extended frontier, the legacy census, and blocker 1 decided (2026-08-11)

This section opens at inherited committed tip ``ab0c19f`` and reviews
``a3b8d1e..ab0c19f`` — the production commit ``807f7ff``, its test commit
``f2ac9d0``, and the two documentation commits ``d725676`` and ``ab0c19f``.  It
lands one compiler change: the completion boundary's single structural
comparison now covers the assembled function's **whole stored state** rather
than its body alone, which closes a class §51's move leaves open.  It also runs
three of the four gates §51.9 recorded as absent, reproduces the declared
frontier at the exact tip and extends it by 391 cells, and **decides blocker 1**.

Sections 49, 50 and 51 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 52.1 The inherited state, re-established by inspection

- **Origin is ``a3b8d1e`` and the branch is published**, exactly as §51.1
  records.  ``git ls-remote`` resolves
  ``refs/heads/refactor/compiler-ir-phase3-std-move-call`` to
  ``a3b8d1ed70b71c3b137a62c3b27fa01e1262e621``.  ``.git/packed-refs`` still
  carries ``58e8565`` for the remote-tracking ref and ``cb49ff7`` for the local
  branch; it was not read for any claim here.  Nothing was pushed; origin is
  left where it is.
- The five protected tracked files hash exactly as recorded, before and after
  every change in this section.
- **§51.7's latency table was re-derived from the raw driver log rather than
  from the summary.**  ``latency/driver.log``'s twenty per-round lines give A/B
  mean 1.0016777 and max 1.0686494, and A/A mean 1.0020119 and max 1.0713509 —
  the recorded 1.0017 / 1.0686 / 1.0714 to four places.  The declared headroom
  is real.

### 52.2 Comparing the body is not comparing the function

§51.4 is right that the window needed closing and right about how: the single
comparison moved to the far side of assembly and the four completion stages.
Three questions it left as prose were re-derived here, by measurement.

**1. Does comparing the assembled body leave anything in the window
uncovered?  Yes — three fields, and it matters.**  The reference is a
``List[llir.Stmt]``; ``llir.Function`` stores four fields.  Patching the last
completion stage at ``ab0c19f`` and recompiling ``sss ijk->jk`` in both
automatic arms:

| tamper applied after the reference is captured | outcome at ``ab0c19f`` | evidence |
| --- | --- | --- |
| ``name`` rewritten | **COMPILED** | entry point emitted as ``evaluate_TAMPERED`` |
| ``return_type`` rewritten | **COMPILED** | ``float evaluate(...)`` emitted where a tensor is declared |
| one argument dropped | **COMPILED** | public signature one argument short of the body's own references |
| one argument renamed | **COMPILED** | tamper reaches the emitted signature |
| a non-expression appended to ``args`` | refused | but downstream, as a generic ``CodegenError`` |

Codegen does validate that ``Function.args`` holds exact LLIR expressions — that
is what refuses the last row — but it never checks *which* arguments they are,
so it is a type backstop and not a content one.  And no identity requirement in
the window observes any of the three fields, for the reason 52.2.2 gives.  This
is a real gap, and an ABI-signature reference is affordable: the signature is a
pure function of the frozen ABI metadata, so it can be captured from that same
authority before the pass manager, and it is a handful of argument nodes against
a body of some hundreds of statements.

**2. Can ``returned is assembled`` misfire once a legal plan composes a
workspace with a tile?  Yes — in the direction opposite to the one recorded.**
§51.4 and the boundary's own docstring said the requirement "fails closed the
moment a future plan composes a workspace with a tile, panel or relayout".
Measured from the code: **all four completion stages return the object they were
given.**  ``_complete_result_tile_impl`` and ``_complete_relayout_impl`` each
have exactly one return expression, ``function``; ``complete_panel`` and
``complete_parallel`` construct no ``llir.Function`` on any path.  Result-tile
completion does its work by rewriting *nested* loop bodies in place
(``tile_loop.body = ...``, ``tile_loop.body[0:0] = ...``,
``tile_loop.body.extend(...)``).

So the identity requirement cannot detect that a stage ran, and a fused
workspace-plus-tile plan would satisfy it.  What fails closed for such a plan is
the **structural comparison**, because those in-place nested rewrites are not in
the reference.  That is the gate a fused replay contract has to extend
deliberately — and extending it means teaching the detaching mirror to reproduce
a *second* shared completion stage, with all of the drift risk §50.3 had to
solve once for ``rewrite_dynamic_vector_accesses``.  That cost belongs in
blocker 1's decision, and 52.7 takes it there.  The identity requirement is kept
for what it does prove: no stage substituted a different function object.

**3. Should the exact-root requirement live in codegen for every family?  No,
and that is measured rather than argued.**  Handing codegen a ``Function``
subclass carrying identical state on a *dense-receiver* plan already fails
closed: ``CodegenError: No C++ codegen implemented for LLIR node type:
HostileFunction at root``.  Codegen's exact-type dispatch is therefore a
universal barrier already, for every family.  Keeping the check in this family
buys exactly one thing — diagnosis ownership, so the refusal carries
``sparse_workspace_completion_lost`` instead of a generic unknown-node error —
and that is sufficient reason to keep it where it is.  Moving it into codegen
would duplicate an existing refusal inside a layer shared with legacy default
dispatch, for no behavioural gain.

### 52.3 The closure adds no comparison

``TorchCppKernelABI.signature()`` becomes the one source for the spelling
``assemble_function`` emits, and ``assemble_function`` consumes it, so a boundary
that needs to state what the signature must be asks for it instead of restating
the policy.  ``_ordered_key_expected_checkpoint`` captures that signature beside
the detached body in one ``_OrderedKeyCompletionReference``, still before the
pass manager — which covers ABI assembly itself, not only the four stages after
it.

The comparison is folded into the **existing** traversal: the boundary passes
``[body, return_type, name, args]`` against the reference's four members as one
wrapper sequence, so ``_exact_sparse_completion_matches`` is still entered
exactly once per ordered-key compile, and because its work queue pops from the
tail the O(#args) members settle before the body walk starts.  Two O(1) root
requirements are added on the same principle: the root's stored field set must be
exactly the four declared fields, so an added field cannot ride along unread.

### 52.4 The window matrix, re-run and extended: seventeen classes, no gaps

Each class patches the last completion stage (or, where stated, assembly or a
managed pass), asserts its tamper actually landed, and counts every consultation
of a forged ``__class__``/``__eq__``/``__hash__``/``__reduce_ex__``.

*The six inherited classes* (§51.4) — new function sharing the body list, a
shared-list element swapped for a twin, an in-place ``Comment`` rewrite inside a
verified statement, a statement duplicated inside the ordered drain, a
``Function`` subclass from assembly, and a dense-receiver plan that must find the
gate inert — behave exactly as §51.4 requires, unchanged.

*The four new classes.*  A completion stage that mutates ``assembled.args`` in
place: **refused** (it was refused before only downstream, as a
``CodegenError``).  A stage that swaps ``assembled.__dict__`` wholesale for an
equal dict: **accepted**, and correctly — see below.  A stage that returns a
function whose body is a **list subclass**: refused on the root's exact-type
test.  A pass that mutates the captured reference: refused, which is the positive
proof that the comparison genuinely reads the reference rather than comparing the
body with itself.

*The signature questions* — ``name``, ``return_type``, a dropped argument, a
duplicated argument, a renamed argument — all five refused with this family's own
code, in both arms.

**Two non-rejections are recorded as such rather than papered over.**  A
wholesale ``__dict__`` swap that preserves every field the boundary reads,
including a fresh body list holding the same statements in the same order, is
accepted: it changes nothing the emitted C++ can observe, and rejecting it would
pin an implementation detail of assembly rather than the program.  Its
reordering and renaming variants are refused.  And a pass that walks the caller's
frame to rewrite the reference *and* the body consistently **compiles**.  That is
not a closable gap: code that can read and write the checker's own locals defeats
any in-process boundary, and a digest held in a closure is frame-reachable too.
It is the boundary's stated threat-model limit, and stating it is the honest
alternative to a check that would only look stronger.

**Zero foreign hook invocations across all seventeen classes**, and all 40
normal reduction/TTM compile cells still compile in both arms.

### 52.5 The frontier: 748 reproduced at the exact tip, then 1139

§51.2 recorded that the inherited ``frontier_ext.json`` is a 14:43 artifact in a
final-tip directory and refused to relabel it.  The harness was re-run here.

**The declared numbers reproduce exactly**: 748 cells, **98 admitted, 387 defect
codes, 263 loop-plan diagnostics, zero unclassified**, and the three sums add to
748.  ``LoopPlanDiagnostic.code`` is read out of ``diagnostics``, never from
message text, and no loop-plan cell carries a ``defect`` attribute.

**The arm-invariance scoping is honest and stays scoped.**  Exactly **three** of
the 748 are arm-variant, and all three are rank-3 **dense-receiver** cells
outside the ordered-key envelope: ``dsd ijk->k [d]`` (admitted / program shape),
``ddd ijk->k [d]`` and ``ddd ijk->ik [dd]``.  The claim that is true is the
scoped one — arm-invariance across the retained ordered-key-envelope cells — and
the unscoped one is false.

**The extension adds 391 cells across the five axes the record named as
unextended**, for **1139 cells: 199 admitted, 580 defect codes, 360 loop-plan
diagnostics, zero unclassified, and still exactly three arm-variant cells** (the
same three; the extension adds none).

*A defect in the harness itself, which the declared 748 could not have
exposed.*  The LoopPlan legality layer has **two** exits: ``_invalid`` raises
``InvalidSchedule`` and ``_unsupported`` raises ``UnsupportedFeature``.  Both
carry the same structured ``LoopPlanDiagnostic`` and neither carries a
``defect``.  The inherited harness caught only the first.  None of the 748 cells
reaches the second, so reading one exit looked complete; the extension reaches it
33 times, and until the harness was corrected those 33 were reported as
``UNCLASSIFIED``.  The accurate accounting of a rejected neighbour's stable
identifier is therefore **three** classes, not two: a target/lowering
``defect`` code, an ``InvalidSchedule`` diagnostic (327 cells), or an
``UnsupportedFeature`` diagnostic (33 cells).

*Per axis.*  **Rank 6** is admitted at 20 of 39 cells, and the admitted set
reaches key ranks the declared envelope never named — ``ssssss ijklmn->jklmn`` is
a ``(prefix 0, K 5)`` split, and ``->in``/``->imn`` are ``(1, 1)`` and ``(1, 2)``
at rank 6.  This is admission, not verified correctness, and 52.9 says so.
**Non-ADD reduction operators** are admitted at **0 of 36**: 24 fail closed at
``UnsupportedFeature/auto_reduction_operator`` and 12 at
``sparse_parent_dominance``.  A non-ADD *combiner* under an ADD update is a
different thing and is admitted once, ``ss*ss ij->j``, which is an ordinary
additive reduction over an intersection product.  **COO and singleton levels**
are admitted at **0 of 74** — 47 at ``unsupported_format``, 19 at
``sparse_parent_dominance``, 6 at ``UnsupportedFeature/singleton_level``, with
COO receivers refused as well as COO operands.  **Multi-operand contractions
beyond TTM** — MTTKRP, TTMc, a four-factor chain, SDDMM, a masked chain and a
three-factor vector contraction — are admitted at **0 of 16**, each with a
stable code.  **Zero and degenerate extents crossed with every receiver** are
admitted at 80 of 208, with zero unclassified and zero arm-variance, so a zero
or unit extent does not move any cell's disposition off its classified route.

### 52.6 Blocker 2, re-derived and its host identified — but not built

§51.8 does not in fact contain the K == 0 conclusion; it lives in §49.5's second
blocker.  The substance was re-derived here regardless, and it stands, on
stronger grounds than the original statement had.

**Measured at the tip, not inferred.**  All ten cells §49.5 names are blocked in
the automatic arm at ``InvalidSchedule/sparse_parent_dominance``.  Under their
own DECLARED order, six reach ``unsupported_sparse_output_domain`` and two —
``sss ijk->ij`` and ``ssd ijk->ij`` — reach ``LoopIRTargetError/
unsupported_program_shape``, which is the three-way gate §49.5 predicted, now
observed rather than argued.

**``sss ijk->j`` is a different construction**, exactly as claimed: under order
``ijk`` its single result coordinate sits at position 1 between reductions at 0
and 2, so ``_ordered_key_split`` returns ``None`` by *interleaving*, not by an
empty key.  One workspace region cannot own it, and no K == 0 family reaches it.

**The K == 0 family needs no workspace and therefore no schema change, and that
is now checked against the node definitions rather than asserted.**  With an
empty key every result coordinate is bound above the outermost reduction; the
prefix loops are in result-level order (the split's strictly-increasing test) and
each one either iterates a dense domain or is driven by one stored compressed
level whose coordinates are strictly increasing within a parent.  So the prefix
loops already visit result cells in lexicographic order, and a scalar
accumulator plus a direct append is the right shape.  Neither workspace node can
express such an accumulator: ``WorkspaceDecl`` is bound to a ``TileId`` and
buffers one affine split's point domain, and ``SparseWorkspaceDecl`` requires an
ordered key of **one or more** dimensions.  The accumulator is therefore an LLIR
local, as every other family's reduction accumulator already is, and **canonical
v11 stands**.

**Of the nine K == 0 cells, five are reachable and four are not.**  Four —
``sss/dds/sds/ssd ijk->ij`` — have ``(COMPRESSED, COMPRESSED)`` receivers, which
``_classify_sparse_output_family`` routes to the doubly-compressed family
*before* the ordered-key branch is consulted; two of those stop at that family's
row/column domain rules and two at the target's hierarchical compressed descent.
A K == 0 family in the ordered-key branch never sees them.  The five it does
reach are ``ss ij->i``, ``ds ij->i``, ``sss ijk->i``, ``dss ijk->i`` and
``ssss ijkl->ijk`` — three of them census representatives.

**The correct host is not a new target.**  ``MULTI_COMPRESSED_ASSEMBLY`` already
admits exactly these receiver shapes — a dense prefix over a compressed suffix,
*including* the degenerate rank-1 all-compressed result — and
``_MultiCompressedAssemblyLowering`` already emits one dense loop per prefix
level, one stream loop per compressed result level, the conditional
compressed-parent append with child position close per structural level, the
dense-prefix catch-up and the root position close.  Its only bar to K == 0 is
``not reduce_update``.  The missing construction is an optional reduction
sub-nest below the innermost assembly loop whose leaf accumulates into a scalar
the ``AppendEntry`` then reads.  That is a materially smaller and better-founded
change than a second rank-general target, and it is the finding that should
govern the next session's estimate.

**The LoopIR-only repair has a seam, and unlike the tile it is not pinned.**
``Scheduler.auto_schedule_plan`` is the plan-producing automatic origin whose own
docstring records that release dispatch does not consume it, so a repair there
cannot change legacy default generated code.  And the legality layer treats the
loop **order** differently from the tiles: ``_verify_auto_workspace_decision``
and ``_verify_tiling_capabilities`` require the recorded workspace and tiles to
equal a re-derived decision exactly — which is why §51.11's tile suppression was
refused at ``auto_tile_decision`` — whereas the order is checked only for
*legality* (``_verify_storage_order``'s result-storage and
sparse-parent-dominance rules).  A repaired legal order is therefore not
refused the way the suppressed tile was.  The forced reorder itself is one block
in ``Scheduler.select_loop_order`` that moves the last free variable to the
innermost position when no free variable follows the last reduction; for these
cells that is precisely what puts a compressed parent below its child.

**None of this is built.**  The family, the target extension and the repair are
not in this tip, and no partial work toward them is committed.  What is recorded
is the re-derivation, the exact five-cell reach, the host, the seam, and the
schema conclusion checked against the representation.

### 52.7 Blocker 1, decided: TTM ``dds`` is not migratable under this origin

§51.11 left the decision open between a fused workspace-plus-tile contract for
the legal ``CHILD_OF`` placement with stated arm-variance, and recording TTM
``dds`` as unmigratable.  It is decided here, against the fused contract, on four
costs — one of which §51.11 could not have priced.

1. **The fused contract needs a second detaching mirror.**  52.2.2 measured that
   result-tile completion works by rewriting nested loop bodies in place, and
   that the gate which refuses it is the structural comparison, not the identity
   requirement.  A fused contract therefore has to teach
   ``_OrderedKeyExpectedBody`` to reproduce ``_complete_result_tile_impl``'s
   rewrites as faithfully as it already reproduces
   ``rewrite_dynamic_vector_accesses`` — the exact problem §50.3 had to solve
   once, at the cost of the drift-proofing that reads the shared pass's own
   frozen config.  Doing it twice, for a stage an order of magnitude larger, is
   the real price.
2. **It would be the first arm-variant migrated family.**  The non-regblock
   OUTERMOST placement is illegal, not unimplemented: it hoists ``j_out`` above
   ``i`` while a ``dds`` receiver's streamed compressed level must be appended in
   lexicographic ``(i, j)`` order.  Every milestone on this branch has gated on
   arm-invariance, and 52.5 confirms the only arm-variant cells in 1139 are
   dense-receiver neighbours outside the envelope.  A single-arm family cannot be
   declared complete over an envelope, which is the exit-audit question it would
   have to answer.
3. **There is no correctness pressure.**  §50.6 measured legacy's own route for
   the neighbouring TTM ``sss`` cells as unsound in every configuration, and
   52.8's census re-measures ``TTM sss x {dd,ss,ds,sd}`` as unsound at 0 of 6
   configurations each.  Nothing downstream depends on the composition being
   available.
4. **It buys two census cells**, in one of two arms.

**Decision: TTM ``dds`` is recorded as not migratable under the current
automatic origin**, and the fused contract is declined.  The precision matters:
this is not "permanently unmigratable in principle".  It is unmigratable for
exactly as long as the automatic origin emits an affine tile for a ``dds``
receiver against a ``kl`` factor while the tile heuristic lives in a layer shared
with legacy default dispatch.  Making it reachable *without* arm-variance means
changing that heuristic, which changes default generated code — a Phase-8
question, not a refactor-neutrality one.  If Phase 8 ever revisits the heuristic,
this becomes reachable on its merits; until then the honest record is a closed
decision rather than an open blocker.

### 52.8 The absent gates: three of four run

§51.9 recorded four gates as absent.  Three ran here.  The fourth — the
compiled public differential over the twenty migrated cells, and the
erasure/oracle differential it carries — is reported in 52.9.

**The crash-isolated legacy census across extents, dtypes and densities.**
Twenty cells plus the adjacent forced-sparse ``dd ij->j``, each configuration in
its own disposable subprocess with ``RLIMIT_CORE`` 0, an ``RLIMIT_CPU`` of 600 s
and a wall-clock timeout, streaming one JSON line per density so a route that
terminated the interpreter would be named by its missing line.  **666
measurements.**

- **The expensive claim holds.**  All ten cells the committed suite locks as
  sound are SOUND at **60 of 60** configurations each — five shapes (including
  ragged and singleton extents) × two dtypes × both automatic arms × three
  densities.  **No cell claimed SOUND was measured non-sound anywhere.**
- **LoopIR is correct on every one of the 666**: the LoopIR route executed on
  all of them and matched the dense reference on all of them.
- **§50.6 is too strong for three of the eleven unsound cells, and that is a
  correction.**  ``dss ijk->k``, ``sss ijk->k`` and ``ssss ijkl->il`` are sound
  at 2 of their 6 measured configurations.  The pattern is mechanical: they
  become sound exactly when the outer *reduced* extent collapses to 1 —
  ``(1,4,5)`` for the two rank-3 cells, ``(2,1,4,5)`` for the rank-4 one — which
  removes the repeated pass that produces the duplicate drained coordinates.  So
  the disposition of those three is **configuration-dependent**, not a property
  of the cell, and the committed characterization test's assertion that the
  unsound set is exactly those eleven is valid at its own fixture extents rather
  than universally.  The remaining eight are unsound at 0 of 6.
- One further scoping note, stated rather than glossed: the driver gives the
  unsound-claimed cells the lean sweep (arm 0, f32 only), so **§50.6's
  "arm-invariant" for the unsound eleven is not re-confirmed by this sweep**; it
  is re-confirmed for the ten sound cells, which ran both arms.

**A genuine stored-zero fixture.**  Two exist in the committed suite and both
were re-run.  ``test_stored_compressed_zero_survives_both_arms_and_dtypes``
hand-builds a compressed operand whose ``(pos, crd, values)`` arrays store a real
``0.0`` — the form ``to_sparse`` cannot produce, because it filters structural
zeros out — and checks the compiled storage, the drained values and the oracle.
``test_stored_operand_zeros_keep_their_key`` states the same property on an
all-dense operand, where every cell genuinely is stored.  Separately, the
retained differential harness was found to carry a **second** half of the §50.7.5
defect: not only did its ``stored_zeros`` branch build two locals it never read,
**no call site passed ``stored_zeros=True`` at all**, so that arm was never
exercised in any form.  The harness used here builds the operand structure from a
fully stored, zero-free tensor and then zeroes a deterministic stride of the
stored values, asserting that a zero really is stored.

**Release neutrality against ``ab0c19f``: byte-identical.**  The 20-source corpus
20/20 and the 42-source ``ss@dd`` grid 42/42 identical with no differing file;
the 86-case schedule audit ``total=86 admitted=46 rejected=40 nonidentical=0``
on both sides and identical between them.  mypy is hash-identical between the
arms at 146 lines — **140 errors in 11 files, none in a changed file**.  Flake8
differs only by three F401s in modules that exist solely in the working tree and
are untracked.  Black is 15 findings / 137 clean on base and 19 / 174 on the
candidate; the four extra reformat targets are all untracked ``test_spmm_*``
modules, the tracked reformat sets are identical, and neither changed file
appears in either list.  ``git diff --check`` exits 0 and the five protected
files hash exactly as recorded.

### 52.9 Verification

All gates ran in the ``scorch`` conda environment on the Apple M5 development
machine.  Base is a detached worktree at ``ab0c19f``; the candidate is this tree,
and the suite ran in a clean detached worktree at the exact tip.

- **Complete non-performance suite at the exact tip**, eight file-disjoint
  partitions in fresh processes: **6,095 selected nodes, 6,081 passed, 14
  skipped, 3 deselected, zero failures and zero errors, every partition exiting
  0, all eight at one revision ``97d23fe``.**  Tallied independently from the
  eight JUnit XMLs, not from the driver's JSON.  A pre-run proof places all 86
  selected modules of the 88 tracked exactly once and shows the partition node
  counts summing exactly to the selected total; module and node partitions are
  both complete and both disjoint, and the two tracked modules selecting nothing
  under ``-m "not perf"`` are unchanged.  The count moves from the inherited
  6,079 to 6,095, and the difference is fully accounted: the ordered-key target
  file is the only test module that changed, and it collects **244 at ``ab0c19f``
  and 260 here** — exactly +16, which is the arithmetic check that no test was
  lost.  (§51.9's "240 passed" is that module's *passed* count, four of its
  collected nodes being skips; the collected figure is the one that reconciles
  with the suite.)
- **Compiled public differential over the twenty migrated cells, and the
  erasure/oracle differential it carries: 648 checks, zero failures.**  This is
  the third and fourth of §51.9's four absent gates.  Every cell is executed
  through the real JIT path and checked four ways — exact ``(pos, crd)`` level
  storage against the scheduled-program oracle, scheduled-versus-erased oracle
  equality, stored value agreement, and a dense PyTorch reference — across both
  automatic arms, ``float32`` and ``float64``, singleton/ragged/empty/zero
  extents, exact cancellation, commuted operands and dense/compressed second
  factors, with the eight blocked ``TTM dds`` cells asserted at their exact code.
  The count is the inherited 568 plus 80, which is the twenty stored-zero runs
  this session added; each of those asserts that a zero really is stored before
  it measures anything, and all twenty asserts passed.
- **Crash-isolated legacy census: 666 measurements**, detailed in 52.8.
- **Release neutrality against ``ab0c19f``: byte-identical** — corpus 20/20 and
  grid 42/42 with no differing file, schedule audit ``total=86 admitted=46
  rejected=40 nonidentical=0`` on both sides and identical between them.
- **Statics.**  mypy on the candidate is **140 errors in 11 files, none in a
  changed file**, and hash-identical to base at 146 lines.  Flake8 differs
  between the arms only by three F401s in untracked working-tree-only modules.
  Black is 15/137 on base and 19/174 on the candidate, the four extra reformat
  targets all untracked ``test_spmm_*`` modules; restricted to tracked files the
  sets are identical and neither changed file appears.  ``git diff --check``
  exits 0; the five protected tracked files hash exactly as recorded.
- **Representation unchanged.**  No node kind, no canonical schema change (v11
  stands), no request- or schedule-identity change.
- **No default dispatch, cache, selector or fallback change**, and no legacy code
  removed.

**Paired compile-only latency.**  The ceiling for a compile-only integrity
boundary on this branch is **1.10**.  Base is a detached worktree at ``ab0c19f``;
the candidate is the final tip.  Each measurement is a fresh subprocess importing
exactly one source tree and timing the same 40-cell ordered-key compile-only grid
(no JIT, no C++ compiler); 20 rounds, alternating the within-round order, 4
warmups and 21 samples per process, plus a base-against-base A/A control in every
round.  The machine was otherwise idle.

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 0.9534 | 1.0055 | 1.0001 | 1.0131 | **1.0145** |
| A/A control (median) | 0.9517 | 0.9995 | 0.9988 | 1.0239 | **1.0363** |
| A/B ratio (min-of-samples) | 0.9991 | 1.0731 | 1.0628 | 1.0879 | **1.0959** |
| A/A control (min-of-samples) | 0.9430 | 1.0019 | 1.0029 | 1.0339 | **1.0738** |

Pooled fastest-sample A/B 1.0009; pooled A/A 0.9993.  Order controls agree:
base-first mean 0.9960, candidate-first mean 1.0041, a 0.8% spread inside the A/A
floor.

**Every declared statistic is at or below 1.10, the largest being 1.0959** — and
the two rows disagree in a way that should be stated rather than averaged away.
On the median statistic the result is a clean null and *better* than the
inherited candidate: A/B mean 1.0001 against A/A 0.9988, with the A/B maximum
1.0145 below the A/A maximum 1.0363.  The **per-round** min-of-samples row shows
a consistent ~6-7% offset the A/A control does not show — but the **pooled**
fastest-sample statistic, which is the same idea computed over all rounds at once
and the one §51.7 declared, is 1.0009 against 1.0(-0.07)% for A/A.  The honest
reading is a null result with one noisy per-round variant, and the largest number
anywhere is 1.0959 against a 1.10 ceiling — inside it, but with less headroom
than the inherited 1.0729.

One avoidable constant was identified and **not** taken, because taking it would
have invalidated the suite and latency runs above with no gate budget left to
re-run them: ``assemble_function`` now calls ``self._validate()`` and then
``signature()``, which validates again, so it performs one redundant validation
per assembly.  Removing that line is safe (``signature()`` validates before doing
anything, and the body type-check does not depend on it) and is the first thing
the next session should do, re-gating suite and latency behind it.

### 52.10 Phase-7 exit audit, re-run

*Migrated families complete over their proven envelopes.*  Yes over the declared
envelope, and now on stronger evidence for the legacy-comparand half: the ten
sound cells match a sound legacy route at 60 of 60 configurations each.  The
extension shows rank-6 splits up to ``(0, 5)`` are *admitted*, which is not the
same as proven, and this section does not claim them.

*Every neighbour carries a stable fail-closed disposition.*  Yes, and re-measured
at this tip over 1139 cells: 580 defect codes, 327 ``InvalidSchedule``
diagnostics and 33 ``UnsupportedFeature`` diagnostics, **zero unclassified**,
none carrying a ``defect`` attribute.  Corrected per 52.5: the classification is
three-way, not two-way.

*Representation unchanged.*  Yes.  No node kind, no canonical schema change (v11
stands), no request- or schedule-identity change.

*Release behaviour unchanged.*  Yes, and measured: corpus 20/20 and grid 42/42
byte-identical, the 86-case audit identical, zero native artifacts, and no
dispatch, cache, selector or fallback change.

*Compiler latency within the declared ceiling.*  See 52.9.

*The declared matrix is closed.*  **No.**  Blocker 1 is now decided rather than
open, which removes it as a blocker to *decide* but not as a gap in the matrix:
TTM ``dds`` stays unmigrated by decision.  Blockers 2 and 3 are open and
unbuilt.

**Phase 7 is NO-GO.**  No Phase-8 inventory, cutover, cache, selector or
default-dispatch change was made; no fallback was weakened and no legacy code
was deleted.

### 52.11 What this section does not do

- **Blocker 2 is not built.**  52.6 re-derives it, fixes its migration
  accounting at five reachable cells, identifies
  ``_MultiCompressedAssemblyLowering`` as the host and
  ``Scheduler.auto_schedule_plan`` as the repair seam, and settles the schema
  question against the node definitions — but the family, the target extension
  and the plan repair are not in this tip.  They must still be gated together,
  for the reason §49.5 gives.
- **Blocker 3 is untouched.**  Row-scope dense prefixes at rank >= 3 — a DENSE
  result level bound by a STORED loop, needing the row-scope catch-up against a
  dynamic parent count at depth — is where §49.5 left it.
- **No cross-host run.**  Every gate here is the Apple M5 machine.  Scorch does
  not use MKT as a routine CPU test target, and nothing in this change is
  platform-dependent, but the record should not imply a second host confirmed it.

## 53. Blocker 2 built: the rank-0 ordered key, its host, and a repaired plan origin (2026-08-11)

This section opens at inherited committed tip ``4644608`` and reviews
``a3b8d1e..4644608`` — the production commit ``12bd832``, its test commit
``97d23fe``, and the documentation commit ``4644608``.  It **builds blocker 2**:
the ``K == 0`` family, the target extension that hosts it, and the automatic
plan-origin repair, gated together.  It also corrects three claims §52.6 makes
about that work, one of which would have made the repair migrate nothing.

Sections 49 through 52 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 53.1 The inherited state, re-established by inspection

- **Origin is ``a3b8d1e`` and the branch is published.**  ``git ls-remote``
  resolves ``refs/heads/refactor/compiler-ir-phase3-std-move-call`` on origin to
  ``a3b8d1ed70b71c3b137a62c3b27fa01e1262e621``; the local tip ``4644608`` is
  seven commits ahead and zero behind.  ``.git/packed-refs`` still carries the
  stale ``58e8565``/``cb49ff7`` and was not read for any claim here.  Nothing was
  pushed.
- The five protected tracked files hash exactly as
  ``statics/protected-hashes.txt`` records, before and after every change here.
- **§52.6's declared-order accounting is off by two.**  It records that "six
  reach ``unsupported_sparse_output_domain`` and two — ``sss ijk->ij`` and
  ``ssd ijk->ij`` — reach ``LoopIRTargetError/unsupported_program_shape``", which
  accounts for eight of the ten cells.  Measured at the tip, **eight** reach
  ``unsupported_sparse_output_domain`` and two reach
  ``unsupported_program_shape``; the two the record loses are ``dds ijk->ij`` and
  ``sds ijk->ij``.  The three-way gate §49.5 predicted is intact; only the count
  was wrong.

### 53.2 The repair seam §52.6 names is not on the automatic arm's path

This is the correction that matters, because building to §52.6's instruction
would have produced a change that migrates nothing.

§52.6 records that the repair belongs in ``Scheduler.auto_schedule_plan``,
"the plan-producing automatic origin whose own docstring records that release
dispatch does not consume it".  The docstring is accurate and the conclusion
drawn from it is not: **``auto_schedule_plan`` has no production caller at all.**
Across ``src/`` the only occurrence is its own definition and its own
``TypeError`` message; the four modules that call it are all tests
(``test_cin_analysis``, ``test_loop_plan``, ``test_loopir_pipeline_execution``,
``test_schedule_api``).  The automatic arm every gate on this branch measures
reaches its plan
through ``compile_cin_via_loopir`` →
``pipeline._apply_requested_schedule`` → ``Scheduler.apply_schedule(cin,
Schedule())`` → ``Scheduler._apply_schedule_legacy``'s ``is_identity`` branch.

Measured rather than argued: with ``auto_schedule_plan`` replaced by a function
that raises ``AssertionError``, all ten of §49.5's cells route **identically** in
both arms, and one automatic compile makes exactly one
``_apply_schedule_legacy`` call.

So the repair had to land at the origin that is actually consumed.  Doing that
does **not** weaken the neutrality argument, because the argument that carries
the change is not "this entry is unused" but a property of the repair itself:

> It runs only after the recorded plan has been **refused**.  A program whose
> automatic plan verifies today never reaches it, so no program that produces
> generated code today can change.

That is a stronger claim than the seam-based one, and it is what gate 3
measures.  The repair is implemented once, in ``Scheduler._originate_auto_plan``,
and both plan-producing origins — the ``is_identity`` branch and
``auto_schedule_plan`` — now call it, so the two cannot drift.

### 53.3 What the repair is

``select_loop_order`` composes four steps and ends with an unchecked fifth: a
block that moves the last free variable to the innermost position when no free
variable follows the last reduction.  Its purpose is the nest shape legacy
lowering and workspace insertion need.  For a sparse result whose every
coordinate is already bound above the outermost reduction it buys nothing — such
a program needs no workspace at all — and it costs legality: moving one result
coordinate inward puts a compressed physical parent below its child.

The repair is therefore:

* ``select_loop_order`` gains a **keyword-only out-parameter**,
  ``pre_forced_order``, which when given an empty list receives the composed
  order as it stands immediately before that block.  Every existing caller
  passes nothing and executes exactly the statements it executed before; the
  parameter is validated as an exact empty ``list``.
* ``_originate_auto_plan`` builds the plan from the forced order.  If that plan
  is refused with an ``InvalidSchedule`` naming one of the **two rules
  ``_verify_storage_order`` owns** — ``result_storage_order`` and
  ``sparse_parent_dominance``, the only legality rules that are properties of
  the order alone — and the pre-forced order differs, it re-originates the plan
  from the pre-forced order on a fresh copy of the same normalized CIN and
  offers it to the same trust boundary.  If that is refused too, the **original**
  refusal is the one reported.

The retry re-runs the selection on a **fresh deep copy of the same normalized
CIN** rather than reusing the first copy's ``IndexVar`` objects, so no state the
first attempt's surgery mutated can leak into the second.  That re-run sits
outside the inner ``try``, which is deliberate and safe rather than an oversight:
it is the identical call, on an equal pristine copy, that already succeeded
microseconds earlier on the first copy — the selection reads the CIN and mutates
nothing, and it runs before any surgery in both attempts — so it cannot introduce
a failure mode the first attempt would not already have raised.  The inner
``try`` therefore covers exactly what can newly refuse: plan construction and its
verification.

Two things this deliberately does not do.  It does not restate the reorder, and
it does not restate the rules the reorder breaks: the legality layer stays the
sole authority, consulted rather than duplicated.  And it does not touch any
decision that layer *pins* — ``_verify_tiling_capabilities`` and
``_verify_auto_workspace_decision`` require the recorded tiles and workspace to
equal a re-derived heuristic, which is why §51.11's tile suppression was refused
at ``auto_tile_decision``, whereas the order is checked only for legality.  That
asymmetry is what makes an order repair available and a tile repair not.

**Both re-derivations were measured, not assumed.**  For all ten cells, in both
arms: the forced order's plan is refused at ``sparse_parent_dominance``, and the
pre-forced order's plan is **accepted** with ``workspace=None`` and ``tiles=()``
re-derived by the same boundary.  And nothing re-derives the *order*:
``loop_plan_legality``, ``loop_plan``, ``legacy_cin_adapter`` and the whole
``loopir`` package contain no reference to ``select_loop_order`` or
``init_loop_order``, and ``_replay_auto_plan_owned`` rebuilds the legacy nest
from ``plan.loop_order`` itself, so a repaired order replays.

### 53.4 What the family is

``_ordered_key_split`` returns the ``(prefix, key_rank)`` split of one ordered
sparse reduction's result coordinates.  It previously refused ``key_rank == 0``
in the same clause that refuses interleaving.  Those are different facts, and
separating them is the whole CIN change:

* ``prefix + key_rank != len(lhs)`` means some result coordinate is interleaved
  between two reduction loops.  No split exists; this stays ``None``.
* ``key_rank == 0`` means every result coordinate is bound above the outermost
  reduction.  That is a split, and it selects its own family member,
  ``BOUND_PREFIX_ACCUMULATION``.

The prefix domain rules are **unchanged and now apply to every position**: with
``key_rank == 0`` the loop's "a drained key coordinate may iterate any domain"
skip can never fire, so each coordinate is held to its result level's rule — a
dense level must iterate a dense domain, a compressed level must be driven by
one stored sparse level.  The dense-levels-above-the-reduction check is
vacuously satisfied, because ``prefix`` is the whole rank.

The family joins the ``StoreReduce`` set, so it lowers to the same semantic
accumulation leaf every sparse reduction family uses.  Unlike the ordered-key
family, no schedule pass rewrites it: the owning target reads the accumulation
directly.  It is deliberately **not** added to the two merged-domain
continuations, so a merged reduction domain fails closed at CIN with
``unsupported_merged_reduction`` rather than reaching the target.

**No representation change.**  ``WorkspaceDecl`` is bound to a ``TileId`` and
buffers one affine split's point domain; ``SparseWorkspaceDecl`` requires an
ordered key of one or more dimensions.  Neither can express a rank-0
accumulator, and neither needs to: the accumulator is an LLIR local exactly as
every other family's is.  No node kind, no canonical schema change — **v11
stands** — and no request- or schedule-identity change.

### 53.5 The host, and what it grew

§52.6's identification of ``_MultiCompressedAssemblyLowering`` as the host is
correct and it saved the session real work.  The target already admits these
receivers and already emits everything the family needs above the reduction: one
dense loop per prefix level, one stream loop per compressed result level, the
conditional compressed-parent append with child position close per structural
level, the dense-prefix catch-up, and the root close.

What it grew is exactly what §52.6 predicted plus the bookkeeping that makes it
safe:

* ``_bound_prefix_assembly_chain`` routes the new shape.  It independently
  re-derives the receiver — a dense prefix over an all-compressed suffix,
  excluding the canonical-CSR ``(D, C)`` and doubly-compressed ``(C, C)``
  receivers that keep their own families — then walks one loop per result level,
  a non-empty sub-nest of dense and single-cursor sparse reduction loops, and one
  ``StoreReduce`` leaf.  ``_multi_compressed_assembly_chain`` is **untouched**,
  so the existing route is byte-identical by construction, and the two
  predicates are disjoint: one requires an append leaf, the other an
  accumulating one.
* ``_collect_assembly_chain`` collects the optional sub-nest after the assembly
  loops are complete, and ``_require_bound_prefix_leaf`` validates the leaf: one
  additive ``StoreReduce`` into the declared result whose coordinates are
  **exactly the assembly loops' coordinates in order** — which is the program-level
  statement that every result coordinate is bound above the outermost reduction —
  over a receiver outside the doubly-compressed shape.  A merged loop below the
  sub-nest is refused with its own message.
* The three leaf-position tests that read ``len(self.loops) - 1`` now read
  ``self._assembly_depth - 1``, which is the same number whenever there is no
  sub-nest.
* ``_bound_prefix_leaf_statements`` emits the accumulator, and ``_lower_leaf``
  emits ``C_reduction += <value>``.  The reduction sub-nest itself is emitted by
  the **shared** dense/sparse machinery: ``_loop_children`` delegates every
  position at or below the sub-nest to ``_TargetLowering._loop_children``, which
  is why a dense reduction loop inside a stream loop (``sd``, ``sds``, ``ssd``,
  ``sdss``) needed no new code at all.
* The accumulator's C++ identifier is reserved through the same
  ``_reserve_generated_name`` authority as every merge temporary, so a user
  tensor spelled ``C_reduction`` fails closed with
  ``generated_name_collision``.

The emitted kernel for ``ss ij->i`` is one stream loop over the result's stored
rows, a zeroed ``float C_reduction``, the operand's own child segment loop
accumulating into it, and the checked value/coordinate appends — and for
``ssss ijkl->ijk`` the same, with all three structural levels closing
conditionally.

### 53.6 Eight cells migrate, not five, and two of the five do not

The prompt for this session, following §52.6, expected exactly five: ``ss ij->i``,
``ds ij->i``, ``sss ijk->i``, ``dss ijk->i`` and ``ssss ijkl->ijk``.  The measured
answer is eight, and it overlaps the prediction in three.  This is a finding
about the prediction, not a tolerance.

**``ds ij->i`` and ``dss ijk->i`` do not migrate**, and the reason is the
family's own prefix domain rule.  Both have a rank-1 **COMPRESSED** receiver
whose single coordinate iterates a **DENSE** domain (level 0 of ``ds``/``dss`` is
dense), and a compressed result level must be driven by one stored sparse level.
Admitting them would mean appending one entry per row of a dense iteration
space, including rows with nothing stored — the dense-domain assembly seam the
migrated families keep closed by design.  §52.6's five-cell list is a list of
cells the ordered-key **branch reaches**; reaching the branch is not being
admitted by it, and the section did not check the domain rules against the
receivers it named.  Their measured disposition is
``unsupported_sparse_output_domain``, naming the actual violation.

**Five cells the ten-cell list never enumerated do migrate**: ``sd ij->i``,
``ssd ijk->i``, ``sds ijk->i``, ``ssss ijkl->i`` and ``sdss ijkl->i``.  **Four**
of them carry a **dense reduction loop** inside the sub-nest — ``sd`` at depth 1,
``sds``/``ssd`` mixing a dense and a sparse reduction in either order, and
``sdss`` at depth 1 of three — which is why delegating the sub-nest to the shared
loop machinery is load-bearing rather than tidy: not one line of dense-reduction
emission was written for this family.  §49.5 enumerated the ten cells it had
probed, not the family's reach; the frontier is what measures the reach.

One further admitted cell sits outside the 748-cell grid's format conventions
and is therefore not in that count: ``dsss ijkl->ijk`` into a ``dss`` receiver —
a **DENSE result prefix level** over a compressed suffix, which exercises the
dense-prefix catch-up and the pre-sized parent position vector.  It is covered by
the differential.

### 53.7 The repair's scope is 65 cells, and that is deliberate

The repair fires wherever the forced reorder produced an order the storage-order
rules refuse and the unforced order is legal.  Over the 748-cell frontier that
is **exactly 65 cells** (130 cell-arm pairs).  That number was derived *before
any production code was written*, by replaying the ``is_identity`` block's own
plan construction for both orders over every cell, and the built change then
reproduced it exactly:

| | cells |
| --- | --- |
| newly ADMITTED | **8** |
| moved from ``sparse_parent_dominance`` to a shape-specific fail-closed code | **57** |
| still at ``sparse_parent_dominance`` | 64 |
| admitted cells lost | **0** |

The 57 split as 35 ``unsupported_sparse_output_domain``, 10
``unsupported_sparse_output_reduction``, 9 ``unsupported_program_shape`` and 3
``unsupported_union_with_dense``.  Every one stays refused; what changes is that
the refusal now names the shape's actual violation instead of naming the
automatic order.  The 64 that stay are the **permuted-result** cells, whose
pre-forced order is refused by the result's own ``result_storage_order`` rule as
well — the repair correctly finds no legal order to fall back to.

A narrower predicate was available and was rejected.  Conditioning the repair on
the bound-prefix shape itself would have reduced the moved set from 57 to 35 and
made the ``sss ijk->j`` and ``... ijk->ij`` neighbours §52.6 asks to characterize
unreachable at their own codes.  It would also have fitted the mechanism to the
family it happens to enable, which is the failure mode this repository's
engineering standard names first.  The general form repairs a defect in the
heuristic — it can emit an illegal order — and defers to the layer whose job is
to say so.  The 57 moves are reported here as a measured, enumerated consequence
rather than smoothed over.

### 53.8 One release-visible surface degrades its refusal, and it is measured

The inherited neutrality harness covers default dispatch (its corpus and grid)
and explicit non-empty schedules (its 86-case audit).  The repair sits in the
branch reached by an explicitly **empty** ``Schedule()``, which neither covers,
so that surface was captured separately: both pipelines' generated C++ for 118
programs — reductions at rank 2 and 3 over every receiver spelling, plus
elementwise, matmul and TTM controls — on each tree.

**The neutrality property holds exactly.**  On the legacy arm, **100 of 118
programs emit on both trees, the same 100, with identical SHA-256 digests**; the
LoopIR arm is identical on those same 100 and rises from 40 emitting to 45, the
five migrating cells in this case set.  No program that produced C++ produces
different C++.

**A cost that is not byte-visible is recorded rather than glossed.**  For the 18
programs that refuse on both trees, the legacy arm's refusal *kind* changes: it
was ``InvalidSchedule`` carrying a structured ``sparse_parent_dominance``
diagnostic, and it is now ``ValueError: ivar_j is not in list`` raised from
inside the legacy lowerer.  The mechanism is direct: ``apply_schedule``
propagates only the plan, ``legacy_generated_cpp`` replays that plan through the
legacy lowering, and the repaired order is legal but has no legacy form — which
is the same limitation ``_validate_loop_kinds`` states as "the legacy generic
route writes an unsized result vector".  ``InvalidSchedule`` is a ``ValueError``
subclass, so a caller catching ``ValueError`` is unaffected in type; a caller
catching ``InvalidSchedule`` now sees an unstructured error.

This is a genuine degradation and it is not repairable at this boundary: the
branch returns one plan and both consumers read it, so giving the LoopIR arm a
repaired plan while giving the legacy arm the old refusal would need a mode flag
inside a shared layer — worse than the cost it removes.  It is accepted with
three qualifications: it touches only programs that already failed, only on the
legacy *comparison* surface, and the LoopIR arm's disposition for those same
shapes **improves** from "the automatic order is illegal" to a structured defect
code naming the shape's actual violation.
``test_the_legacy_comparand_still_refuses_this_family`` locks the property (no
emission) without pinning the message.

### 53.9 The two halves gate together, and the gating is measured

§49.5 requires the family and the repair to be gated together.  Both halves were
measured **out of process**, against source trees with one half reverted to
``4644608`` and the import asserted to come from the reverted tree:

| arm | the ten cells, automatic | admitted |
| --- | --- | --- |
| family only (CIN + target; scheduler at ``4644608``) | all ten at ``InvalidSchedule/sparse_parent_dominance`` | **0** |
| repair only (scheduler; CIN + target at ``4644608``) | all ten at the family's own refusal — ``unsupported_sparse_output_domain`` or ``unsupported_program_shape`` | **0** |
| both | three of the ten admitted; the other seven at their own codes | 8 over the frontier |

The family-only arm also compiles three of the ten under their **declared**
order, which is the positive control: the family works, and only the automatic
order stands between it and the automatic arm.

### 53.10 The other seven neighbours, each refused for its own reason

Three of §49.5's ten are admitted.  The remaining seven are two plus five: the
two §53.6 accounts for (``ds ij->i`` and ``dss ijk->i``, refused by the family's
own prefix domain rule), and these five.  Measured at the final tip, both arms:

| cell | disposition | why |
| --- | --- | --- |
| ``sss ijk->j`` | ``unsupported_sparse_output_domain`` | interleaving: its single result coordinate sits at position 1 between reductions at 0 and 2, so ``prefix + key_rank = 0 != 1`` and no split exists |
| ``dds ijk->ij`` | ``unsupported_sparse_output_domain`` | ``(C, C)`` receiver → doubly-compressed family, whose row coordinate must be driven by a stored sparse level; ``dds``'s is dense |
| ``sds ijk->ij`` | ``unsupported_sparse_output_domain`` | same family, whose column coordinate must be stored-sparse; ``sds``'s is dense |
| ``sss ijk->ij`` | ``LoopIRTargetError/unsupported_program_shape`` | same family, admitted by CIN and then refused at the target's hierarchical compressed descent |
| ``ssd ijk->ij`` | ``LoopIRTargetError/unsupported_program_shape`` | same |

The four ``(C, C)`` receivers never reach the ordered-key branch at all —
``_classify_sparse_output_family`` routes them to the doubly-compressed family
first — and ``_bound_prefix_assembly_chain`` independently refuses that receiver,
so the exclusion holds at both layers.

### 53.11 Verification

All gates ran in the ``scorch`` conda environment on the Apple M5 development
machine.  Base is a detached worktree at ``4644608``.

- **Release neutrality against ``4644608``: byte-identical.**  The 20-source
  corpus 20/20 and the 42-source ``ss@dd`` grid 42/42 with no differing file; the
  86-case schedule audit ``total=86 admitted=46 rejected=40 nonidentical=0`` on
  both sides and identical between them.  This is the gate that would catch a
  repair leaking into legacy default dispatch, and it does not fire — as the
  refusal-only trigger requires.
- **The 748-cell declared frontier at the final tip, reading BOTH LoopPlan
  exits** (``InvalidSchedule`` and ``UnsupportedFeature``, from
  ``LoopPlanDiagnostic.code``, never message text): **748 cells = 106 admitted +
  444 defect codes + 198 loop-plan diagnostics, zero unclassified**, and three
  arm-variant cells — the same three rank-3 dense-receiver neighbours outside the
  envelope.  The three sums add to 748.  Against the sealed 98/387/263 baseline
  the whole delta is the 65-cell firing set of 53.7 and nothing else, cell by
  cell.  As §52.5 found, **none of the 748 reaches the second exit**, so reading
  both is a property of the harness rather than a difference in these numbers —
  stated so the figure is not mistaken for evidence that the second exit was
  exercised here.
- **Compiled public + erasure/oracle differential: 936 checks, zero failures**,
  extending the inherited 648 by 288 to the new family with the same four checks —
  exact ``(pos, crd)`` level storage against the scheduled-program oracle,
  scheduled-versus-erased oracle equality, stored value agreement, and a dense
  PyTorch reference — across both automatic arms, ``float32`` and ``float64``,
  singleton/ragged/empty/zero extents, exact cancellation, and genuinely stored
  operand zeros.
- **Complete non-performance suite at the final code tip ``a0b5d6f``**,
  eight file-disjoint partitions in fresh processes with per-partition ``TMPDIR``,
  ``XDG_CACHE_HOME`` and ``TORCH_EXTENSIONS_DIR``: **6,203 selected nodes,
  6,188 passed, 15 skipped, 3 deselected,
  0 failures and 0 errors, every partition exiting 0**, all
  eight at one revision.  Tallied independently from the eight JUnit XMLs rather than
  from the driver's own JSON, and the JUnit total equals the pre-run selected count
  exactly.  The pre-run proof places all 89 tracked modules
  exactly once, shows the partition loads [775, 775, 777, 775, 775, 777, 775, 774] summing to
  6,203, and reports module and node partitions both complete and both
  disjoint.  The detached worktree was clean before and after.

  **The node delta is accounted exactly.**  The base at ``4644608`` collects
  **6,095**; this tip collects **6,203**, a net
  **+108** made of **130 added and 22
  removed**.  Added: **+102**
  in the new ``test_loopir_bound_prefix_target``,
  **+20**
  in the reduction/TTM census, and
  **+8**
  in the ordered-key module.  Removed: the
  10
  ordered-key and
  12
  census parametrizations of the cells that migrated or were renamed.  Every removed
  node is a renamed or re-parametrized lock, not a lost test; the ledger lists all
  22 by name.
- **Gates 1 and 2 are read out of those same JUnit XMLs** rather than from a separate
  invocation, which is stricter: the suite ran every module exactly once at one
  revision, so the two gates cannot disagree with the suite total.  Gate 1, the three
  assembly-target files: **403 nodes, 402 passed, 0 failed, 0
  errors, 1 skipped**.  Gate 2, the eleven adjacent memberships — CIN lowering,
  schedule passes, loop-plan legality, LoopIR neutrality, pipeline execution,
  LoopIR->LLIR lowering, scheduler, schedule API, the reduction/TTM census, schedule
  generality and the CIN lowerer: **1180 nodes, 1180 passed, 0 failed,
  0 errors, 0 skipped**.
- **Statics.**  mypy reports **140 errors in 11 files on both arms, none in a
  changed file**, at 146 lines each; the two outputs differ in **exactly one
  line** and it is not an error — "checked 61 source files" on the fresh base
  worktree against "checked 62" in the working tree, which is the untracked
  working-tree-only ``src/scorch/gpu.py``.  (§52.9's "hash-identical" is
  therefore too strong for any base that is a clean worktree; the error set is
  what is identical.)  Flake8 differs between the arms only by the three
  pre-existing F401s in untracked working-tree-only test modules, plus the
  unchanged ``C901`` on ``_apply_schedule_legacy`` at a moved line.  Black is
  clean on every changed file.  ``git diff --check`` exits 0 and the five
  protected tracked files hash exactly as recorded.
- **Representation unchanged.**  No node kind, no canonical schema change (v11
  stands), no request- or schedule-identity change.
- **No default dispatch, cache, selector or fallback change**, and no legacy
  code removed.

Two of the gate's own findings are worth recording because they were caught by
the gate rather than by inspection: the neutrality run's mypy arm caught a new
``"Stmt" has no attribute "body"`` error in the reduction sub-nest collector
(fixed by narrowing inside each branch), and its flake8 arm caught an unused
import in the new test module.  Both are fixed at the final tip.

**One avoidable constant taken.**  §52.9 identified ``assemble_function``'s
redundant ``self._validate()`` — ``signature()`` validates before doing anything
— and left it for this session.  It is removed, with the body type-check moved
behind ``signature()`` so a malformed-metadata call still fails on the metadata
exactly as it did.

**Paired compile-only latency.**  The ceiling for a compile-only integrity boundary on
this branch is **1.10**.  Base is the detached worktree at ``4644608``; the candidate is
the final-tip worktree.  Each measurement is a fresh subprocess importing exactly one
source tree and timing the same 40-cell ordered-key compile-only grid (no JIT, no C++
compiler); 20 rounds alternating the within-round order,
4 warmups and 21 samples per process, plus a
base-against-base A/A control in every round.  The grid checksum is asserted equal across
all three measurements of every round, so the two arms compile byte-identical work
(141162 characters of C++ over 40 cells, verified equal on both trees).  The machine was
otherwise idle: the suite and the differential had both finished.

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 0.9934 | 1.0008 | 1.0005 | 1.0048 | **1.0048** |
| A/A control (median) | 0.9894 | 1.0009 | 1.0004 | 1.0074 | **1.0103** |
| A/B ratio (min-of-samples) | 0.9925 | 1.0001 | 1.0005 | 1.0079 | **1.0089** |
| A/A control (min-of-samples) | 0.9935 | 1.0020 | 1.0011 | 1.0061 | **1.0140** |

Pooled fastest-sample A/B 1.0021; pooled A/A
1.0018.  Order controls: base-first mean
1.0016, candidate-first mean
0.9994.

**Every declared statistic, including the min-of-samples row, is at or below 1.10; the
largest anywhere is 1.0140 — and it is an A/A control number, not an A/B one.**  The
largest A/B statistic is **1.0089**, the min-of-samples maximum, against an A/A
min-of-samples maximum of 1.0140.  Headroom against the ceiling is ~9%, against
the inherited candidate's 0.4%.

**§52.9's min-of-samples offset does not persist, and the causal question it
asked is answerable only in part.**  The inherited candidate showed a consistent
~6-7% per-round min-of-samples offset that its A/A control did not show; here the
A/B min-of-samples row sits *below* its own A/A control at every statistic
(mean 1.0005 against 1.0011, max 1.0089 against 1.0140), so there is no residual
offset left to explain and nothing further to find.  What this run cannot do is
attribute the disappearance: its base is ``4644608``, which already contains the
code that produced the inherited offset, so the A/B ratio measures only this
session's delta — the family, the target extension, the repair, and the removed
redundant validation.  The honest statement is that the offset is absent at this
tip, not that removing ``assemble_function``'s second ``self._validate()`` is
proven to have been its whole cause.  Settling that would need a three-way
measurement against ``ab0c19f``, which is not worth a gate.

### 53.12 Phase-7 exit audit, re-run

*Migrated families complete over their proven envelopes.*  Yes for the new
family over its measured reach: eight frontier cells plus the dense-prefix cell,
all executed through the real JIT path against the oracle and PyTorch in both
arms and both dtypes.  The reach is smaller than §52.6 predicted in two cells
and larger in five, and 53.6 says which and why.

*Every neighbour carries a stable fail-closed disposition.*  Yes, re-measured
over 748 cells: 444 defect codes, 198 loop-plan diagnostics across both exits,
**zero unclassified**, none carrying a ``defect`` attribute.  Fifty-seven
neighbours now carry a *more specific* code than before, which is an improvement
in diagnosis and a change in the record either way.

*Representation unchanged.*  Yes.  v11 stands.

*Release behaviour unchanged.*  Yes, and measured: corpus 20/20 and grid 42/42
byte-identical, the 86-case audit identical, no dispatch, cache, selector or
fallback change.

*Compiler latency within the declared ceiling.*  See 53.11's latency table.

*The declared matrix is closed.*  **No.**  Blocker 2 is now built; blocker 3 —
row-scope dense prefixes at rank >= 3 — is untouched, and blocker 1 stays closed
by decision rather than by migration.

**Phase 7 is NO-GO, on blocker 3.**  No Phase-8 inventory, cutover, cache,
selector or default-dispatch change was made; no fallback was weakened and no
legacy code was deleted.

### 53.13 What this section does not do

- **Blocker 3 is untouched.**  A DENSE result prefix level bound by a STORED
  loop still needs the row-scope catch-up against a dynamic parent count at
  depth, and is still rejected up front with
  ``unsupported_sparse_output_domain``.  Note that the bound-prefix family does
  admit a dense result prefix (``dsss ijkl->ijk`` into ``dss``) — but only when
  that level iterates a *dense* domain, which is the rule blocker 3 is about.
- **The 1139-cell frontier extension was not re-run.**  It is the first item on
  this session's declared sacrifice list, and the 748-cell declared frontier was
  run instead, at the final tip, reading both exits.
- **The heavy legacy sweep for the eleven unsound-claimed cells was not run**,
  so §52.8's scoping note stands unchanged: their arm-invariance is still
  unmeasured and three of them are configuration-dependent.
- **A merged reduction domain is refused, not supported.**  The family's
  sub-nest admits dense and single-cursor sparse loops only; a merged reduction
  fails closed at CIN with ``unsupported_merged_reduction`` and, defensively, at
  the target.
- **No cross-host run.**  Every gate here is the Apple M5 machine.

## 54. Blocker 3 built: the row-scope dense prefix, its two hosts, and a reach the CIN rule cannot measure (2026-08-11)

This section opens at inherited committed tip ``bb7f391`` and **builds blocker 3**
— a DENSE result prefix level bound by a STORED loop, in both the ordered-key and
bound-prefix families.  It also corrects two claims the inherited record and this
session's own prompt make about that work: the host is not either of the two
classes named for it, and the reach cannot be measured from the CIN rule at all.

Sections 49 through 53 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 54.1 The inherited state, re-established by inspection

- **Origin is ``a3b8d1e`` and the branch is published.**  ``git ls-remote``
  resolves ``refs/heads/refactor/compiler-ir-phase3-std-move-call`` on origin to
  ``a3b8d1ed70b71c3b137a62c3b27fa01e1262e621``; the local tip ``bb7f391`` was ten
  commits ahead and zero behind.  ``.git/packed-refs`` still carries the stale
  ``58e8565``/``cb49ff7`` and was not read for any claim here.  Nothing was
  pushed.
- The five protected tracked files hash exactly as
  ``statics/protected-hashes.txt`` records, before and after every change here.
- **The inherited baseline reproduces exactly.**  Re-run at ``bb7f391`` rather
  than taken from §53: the 748-cell declared frontier is 106 admitted + 444
  defect codes + 198 loop-plan diagnostics, zero unclassified, three arm-variant,
  and the three sums add to 748.  The base collects **6,203** nodes, which is the
  figure §53.11 states for this tip.

### 54.2 The reach is 34 cells at CIN, and that is not the reach of the family

§53.6's precedent is that a predicted migration list is a reachability list, so
the reach was measured before anything was built — twice, because one measurement
turned out not to be able to see the whole boundary.

**The CIN rule's own reach: 34 cells.**  The single ``_fail`` implementing "a
dense result prefix level of an ordered-key sparse reduction must iterate a dense
domain" was given a unique sentinel defect code and the 748-cell frontier re-run.
Exactly **34 cells** carry it, in **both** automatic arms, and **no other cell
moves** — so the sentinel isolates that rule and nothing else.

**What lies behind it: 30 of the 34 move when only the STORED case is relaxed.**
A second probe relaxed the rule for ``DomainKind.SPARSE`` alone, leaving UNION and
INTERSECTION refused.  Measured:

| | cells |
| --- | --- |
| reach the TARGET at ``LoopIRTargetError/unsupported_program_shape`` | **26** |
| reach blocker 1's auto-tile seam at ``unsupported_schedule_auto_family`` | 4 |
| stay refused by the **compressed**-prefix rule beside it | 4 |

The four that stay are ``TTM sds x {dd,ds,sd,ss} -> dss``, whose result level 1
is COMPRESSED and whose ``j`` iterates ``sds``'s DENSE level — the dense-domain
assembly seam this session was told not to reopen, and it does not.  The four at
the auto-tile seam are ``TTM sds x {dd,ds,sd,ss} -> dds``: relaxing the prefix
rule lets CIN admit them, and then the automatic origin's affine tile meets a
plan carrying both a sparse workspace and a tile, for which no replay contract
exists.  That is **blocker 1**, closed by decision in §52.7, and it is
characterized rather than absorbed into this family's reach.

**But the migrating set is 29, not 26, and the three extra cells prove the CIN
rule cannot measure this family.**  ``sss ijk->ik [ds]``, ``sds ijk->ik [ds]`` and
``MM ss x ds -> ds`` have a canonical-CSR ``(DENSE, COMPRESSED)`` receiver.  CIN
excludes that receiver from ``ordered_key_shape`` and routes it to
``CSR_SPARSE_ROW`` — a family that *already* permits a stored row domain — so the
prefix rule never sees them.  The schedule pass then builds an ordered-key region
and the **ordered-key target** refuses them, at exactly the require this change
moves.  A probe of the CIN rule is therefore blind to them; only the built change
measured over the frontier finds them.  This is the same class of error §53.6
recorded, in the opposite direction: there a prediction over-counted by naming
cells a branch merely reached, here a probe under-counted by measuring one of the
two gates.

### 54.3 The host is neither class the prompt named, and that was measured

The prompt asks whether ``_RowScopeSparseWorkspaceLowering``,
``_MultiCompressedAssemblyLowering``, or neither is the host.  Measured by
compiling each relaxed cell and reading the CIN family off the live classifier
and the raising frame off the traceback — never message text — the answer is
**neither alone**, and the class that owns the reachable majority is a third one:

| | receives | refusal before this change |
| --- | --- | --- |
| ``_OrderedKeySparseWorkspaceLowering`` | **all 26** reachable frontier cells, and the three ``(D,C)`` cells | ``_collect_ordered_key_chain``'s "a stored prefix loop only above a compressed result level" |
| ``_MultiCompressedAssemblyLowering`` | the bound-prefix half (``key_rank == 0``) | none — ``_bound_prefix_assembly_chain`` returned False, so the program fell through to the generic ``_TargetLowering`` and was refused by whatever it checked first |
| ``_RowScopeSparseWorkspaceLowering`` | nothing | it is rank-2-only by ``_admits_result_layout`` and belongs to a different CIN family (``CSR_SPARSE_ROW``) |

``_RowScopeSparseWorkspaceLowering`` is the **precedent**, not the host: it is
where the row-scope catch-up already exists, for the ``(DENSE, COMPRESSED)``
receiver at rank 2, and this change generalizes its mechanism rather than reusing
its code.  The bound-prefix half's fall-through is worth recording because it is a
diagnosis trap: for an all-compressed operand the generic target refuses at its
hierarchical-compressed-descent rule, so the observed refusal named the *operand*
and said nothing about the prefix at all.

### 54.4 What the change is

**CIN, one rule, shared by two families.**  The ordered-key prefix domain rule now
admits ``DomainKind.SPARSE`` beside ``DomainKind.DENSE`` at a dense result prefix
level.  The argument is a property of dense levels, not of this family: a dense
level stores no coordinates, so the only obligation a prefix loop owes it is that
its child's position array be closed at EVERY logical cell of the level, not
merely at the cells the loop visits.  A dense domain visits them all.  One stored
stream visits a monotone subsequence — the split's own strictly-increasing test
guarantees the order — and the target closes the cells it skips.  A MERGED domain
stays refused: it has no single cursor whose coordinate advances the catch-up, and
neither target chain admits a merged loop above a result level, so refusing it at
CIN keeps the two layers consistent instead of letting one defer to the other.

The **compressed**-prefix rule beside it is untouched, which is what keeps
``ds ij->i`` and ``dss ijk->i`` refused.

**The emission, shared so the two families cannot drift.**  Three module
functions carry it, and the first two are byte-for-byte the group ``_lower_dense``
already emits for the same level when the loop is dense:

* ``_stored_prefix_open_statements`` — the catch-up, placed inside the stream
  loop's body rather than before it, because ``_lower_sparse`` must resolve the
  loop's coordinate before a bound that reads it.
* ``_stored_prefix_close_statements`` — the close after the body.
* ``_stored_prefix_final_statements`` — the one genuinely new group.  A dense
  prefix loop ends at its own extent, so its last cell's close is the level's last
  segment and nothing is left open.  A stored prefix stops at its last stored
  coordinate, so every cell after it — and, with two prefix levels, every cell of
  every skipped outer coordinate — is still open.  One catch-up through the
  prefix's **total cell count** closes them all, spelled as the product of exactly
  the extents ``_assembly_catch_up_bound`` uses for the per-cell numbering, so the
  total cannot disagree with the numbering.  It is gated by
  ``_needs_stored_prefix_final_catch_up``, which carries the same
  ``0 < dense_prefix < len(levels)`` guard the dense twin
  ``_dense_loop_owns_result_assembly`` carries.

``_assembly_catch_up`` gains a keyword-only ``bound`` override for the final
catch-up; every inherited caller omits it and executes what it executed before.

**Three predicates are re-keyed from a loop's driver to the RESULT level**, which
is behaviour-preserving because the two agreed before a stored prefix level
existed, and only the result level is the fact that decides whether a coordinate
is appended: ``_MultiCompressedAssemblyLowering._loop_children``, its
``raw_loop_statements`` root close, and ``_child_stream_statements`` (re-keyed to
the child's own position, because a stored prefix level legitimately nests a dense
prefix level below it).  The stream tallies in ``_collect_assembly_chain`` and
``_require_bound_prefix_leaf`` now count only positions at or past the prefix, so
a prefix stream is not miscounted as a suffix one; and the APPEND leaf
additionally requires a **dense-driven** prefix, so the multi-compressed assembly
family — whose own CIN dense-prefix rule this change does *not* touch — keeps its
boundary by an explicit require rather than by the accident of having no producer.

**No scheduler change, and "no plan repair" is true in two halves.**  The
ordered-key half's automatic plan is already legal: its recorded loop order is
``select_loop_order``'s FORCED order, so blocker 2's retry never fires.  The
bound-prefix half records the PRE-forced order, because a rank-0 key is refused at
``sparse_parent_dominance`` without that repair.  Blocker 3 adds no scheduler code
either way, and both halves are locked as executable facts rather than asserted as
one.

**No representation change.**  Canonical **v11 stands**; no node kind, no request-
or schedule-identity change.  The catch-up cursor is the result assembler's own
``C{level}_pos_index``, which already exists for every compressed level.

### 54.5 Why the catch-up is correct, and the one arm that could see it fail

The per-cell close does not advance the cursor, so the next cell's catch-up
rewrites the slot it just closed.  That is idempotent, not a bug: nothing is
appended between the close and the next catch-up, so the value written is the same
``crd.size()``.  It is also exactly what the rank-2 row-scope family does, whose
``_row_close_statement`` likewise does not increment.

The arm that matters for evidence is the one the inherited differential did not
have.  **A prefix whose every cell is stored never runs a catch-up at all**, so a
differential built from random sparsity can pass with an arbitrarily wrong bound.
This session's differential therefore FORCES prefix cells to be empty — the first
and the last of the outermost level, and one whole inner coordinate across every
outer one, which puts a gap in the FLATTENED numbering rather than only at row
boundaries — and then **asserts, off the built storage, that the operand really
does store fewer prefix cells than the prefix has**.  Every row-scope cell runs
with and without that arm.

### 54.6 Verification

All gates ran in the ``scorch`` conda environment on the Apple M5 development
machine.  Base is a detached worktree at ``bb7f391``; the candidate is a detached
worktree at the final code tip ``460bbf3``.

- **Release neutrality against ``bb7f391``: byte-identical.**  The 20-source
  corpus 20/20 and the 42-source ``ss@dd`` grid 42/42 with no differing file; the
  86-case schedule audit ``total=86 admitted=46 rejected=40 nonidentical=0`` on
  both sides and identical between them.  Because both arms are clean detached
  worktrees, mypy is **hash-identical** (146 lines, 140 errors in 11 files, none
  in a changed file) and so is flake8 (47 lines); black's stdout is identical and
  its stderr differs only in file ORDER, naming the same 15 pre-existing files and
  neither changed file.  (§53.11 had to weaken this to "the error set is
  identical" because its candidate was the working tree.)
- **The empty-``Schedule()`` surface §53.8 added: unchanged, and with no
  degradation this time.**  118 cases on each tree; the legacy arm emits for the
  same **100** on both, with **identical SHA-256 digests**, none gained or lost;
  the LoopIR arm is identical on those same 45 and rises to 52.  **Zero refusal
  records changed** among the cases that refuse on both trees — so unlike blocker
  2, this build costs nothing on that surface, and §53.8's
  ``ValueError: ivar_j is not in list`` degradation is neither widened nor
  repaired by it.
- **The 748-cell declared frontier at the final tip, reading BOTH LoopPlan
  exits** (``InvalidSchedule`` and ``UnsupportedFeature``, from
  ``LoopPlanDiagnostic.code``, never message text): **748 cells = 135 admitted +
  415 defect codes + 198 loop-plan diagnostics, zero unclassified**, the same
  **three** arm-variant cells as the baseline, and the three sums add to 748.
  Against this session's own re-run baseline of 106/444/198 the whole delta is
  **+29 admitted, −29 defect codes, zero admitted cells lost**, enumerated cell by
  cell in the ledger.  As §52.5 found, none of the 748 reaches the second exit, so
  reading both remains a property of the harness rather than a difference in these
  numbers.
- **The final tip's routing and emission are byte-identical to the pre-review
  candidate.**  ``_needs_stored_prefix_final_catch_up`` was introduced after the
  compiled differential began, so its neutrality is measured rather than argued:
  over all 748 cells in both arms — **1,494 cell-arms, 270 of them emitting** —
  the generated C++ digests are **identical**, and the frontier routes differ in
  **zero** cells.  That is what lets the differential's single run stand for the
  final tip.
- **Compiled public + erasure/oracle differential: 1,944 checks, zero failures**,
  extending the inherited 936 by 1,008 to the row-scope prefix in both families --
  the same four checks (exact ``(pos, crd)`` level storage against the scheduled
  oracle, scheduled-versus-erased oracle equality, stored value agreement, and a
  dense PyTorch reference) across both automatic arms, ``float32`` and
  ``float64``, singleton/ragged/empty/zero extents, exact cancellation and stored
  operand zeros -- and, new here, the ``holes`` arm of 54.5.  A ``CsrMatrix``
  branch was added to the oracle comparison because a ``(D,C)`` receiver is
  materialized by the oracle's dedicated CSR builder rather than its per-level
  one.
- **The full non-performance suite earned its keep, and found six nodes nothing
  else could.**  A first pass at the production+test tip returned **six
  failures**, every one an inherited seam lock naming exactly a shape blocker 3
  migrates -- and one of them a test that would have kept PASSING for the wrong
  reason.  All four locks are moved in place per the §49.7 convention and the
  suite was re-run at the corrected tip.  §49.8's finding repeats: the frontier
  and the differential compare compiler outcomes, not test expectations, so only
  the suite finds this class of defect.
- **Complete non-performance suite at the final code tip ``0821799``**, eight
  file-disjoint partitions in fresh processes with per-partition ``TMPDIR``,
  ``XDG_CACHE_HOME`` and ``TORCH_EXTENSIONS_DIR``: **6,282 selected nodes, 6,267
  passed, 15 skipped, 3 deselected, 0 failures and 0 errors, every partition
  exiting 0**, all eight at one revision.  Tallied from the eight JUnit XMLs
  rather than from the driver's own JSON, and the JUnit total equals the pre-run
  selected count exactly.  The pre-run proof places all 88 selected modules of the
  90 tracked exactly once, shows the partition loads
  [787, 787, 785, 784, 784, 784, 784, 787] summing to 6,282, and reports module
  and node partitions both complete and both disjoint.  The detached worktree was
  clean before and after.

  **The node delta is accounted exactly.**  The base at ``bb7f391`` collects
  **6,203**; this tip collects **6,282**, a net **+79** made of **86 added and 7
  removed**.  Added: **+78** in the new ``test_loopir_row_scope_prefix_target``,
  **+3** in the ordered-key module, **+3** in the bound-prefix module and **+2** in
  the multi-compressed module.  Every one of the 7 removed nodes is a seam lock
  MOVED or a test RENAMED -- the ledger names all seven -- and none is a lost test.
- **Gates 1 and 2 are read out of those same JUnit XMLs** rather than from a
  separate invocation, which is stricter: the suite ran every module exactly once
  at one revision, so the two gates cannot disagree with the suite total.  Gate 1,
  the five target files -- both hosts, the new module, the bound-prefix module and
  the rank-2 row-scope precedent: **495 nodes, 494 passed, 0 failed, 0 errors, 1
  skipped**.  Gate 2, the eleven adjacent memberships: **1,180 nodes, 1,180
  passed, 0 failed, 0 errors, 0 skipped**.
- **Representation unchanged.**  No node kind, no canonical schema change (v11
  stands), no request- or schedule-identity change.
- **No default dispatch, cache, selector or fallback change**, and no legacy code
  removed.

One of this gate's findings is worth recording because the gate caught it rather
than inspection: the inherited ``fullsuite/suite_report.py`` hardcoded the
PREVIOUS session's scratchpad path, so running it here printed **that session's**
aggregate -- a copied harness reporting a gate that had not run.  The path is a
parameter now, and every figure above is this run's.

**Paired compile-only latency.**  The ceiling for a compile-only integrity boundary on
this branch is **1.10**.  Base is the detached worktree at ``bb7f391``; the candidate is
the final-tip worktree at ``0821799``.  Each measurement is a fresh subprocess importing
exactly one source tree and timing the same 40-cell ordered-key compile-only grid (no
JIT, no C++ compiler); 20 rounds alternating the within-round order, 4 warmups and 21
samples per process, plus a base-against-base A/A control in every round.  The grid
checksum is asserted equal across all three measurements of every round -- 141162
characters of C++ over 40 cells, in all 60 measurements -- so the two arms compile
byte-identical work.  The machine was otherwise idle: the suite and the differential had
both finished, and their stray processes were reaped first.

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 0.9918 | 0.9999 | 1.0026 | 1.0208 | **1.0338** |
| A/A control (median) | 0.9915 | 0.9996 | 0.9998 | 1.0112 | **1.0164** |
| A/B ratio (min-of-samples) | 0.9922 | 1.0002 | 1.0011 | 1.0159 | **1.0168** |
| A/A control (min-of-samples) | 0.9855 | 0.9999 | 1.0000 | 1.0084 | **1.0171** |

Pooled fastest-sample A/B 1.0085; pooled A/A 1.0009.  Order controls: base-first mean
1.0010, candidate-first mean 1.0042.

**Every declared statistic, including the min-of-samples row, is at or below 1.10; the
largest anywhere is 1.0338.**  Headroom against the ceiling is ~6.4%.

**Unlike §53.11, the largest statistic here is an A/B number, and that is stated rather
than smoothed.**  On the median row the A/B maximum (1.0338) exceeds the A/A maximum
(1.0164), and it comes from a single round -- round 15, candidate-first, whose A/A
control was 0.9972 in the same round, so the round itself was not globally slow.  On the
min-of-samples row, which is the noise-robust one, the A/B maximum (1.0168) sits just
*below* the A/A maximum (1.0171).  The mean offset is +0.3% (A/B 1.0026 against A/A
0.9998).  The honest reading is therefore: no systematic cost beyond about 0.3% is
visible, one round shows a 3.4% median blip that the min-of-samples statistic and that
round's own A/A control both contradict, and nothing approaches the ceiling.  This build
adds two statements to the emitted prefix loop and one loop after the nest, all in
already-compiled Python, so a measurable cost was not expected and none is established.


### 54.7 Phase-7 exit audit, re-run

*Migrated families complete over their proven envelopes.*  Yes for the row-scope
prefix over its measured reach in both families, executed through the real JIT
path against the oracle and PyTorch in both automatic arms and both dtypes, with
forced holes in the dense prefix.

*Every neighbour carries a stable fail-closed disposition.*  Yes, re-measured over
748 cells: 415 defect codes, 198 loop-plan diagnostics across both exits, **zero
unclassified**, none carrying a ``defect`` attribute.  The eight neighbours this
change deliberately leaves refused each keep a code naming their own violation —
four at the compressed-prefix seam, four at blocker 1's auto-tile seam.

*Representation unchanged.*  Yes.  v11 stands.

*Release behaviour unchanged.*  Yes, and measured: corpus 20/20 and grid 42/42
byte-identical, the 86-case audit identical, the empty-``Schedule()`` surface
identical on the legacy arm with zero refusal records changed, no dispatch, cache,
selector or fallback change.

*Compiler latency within the declared ceiling.*  See 54.6's latency table.

*The declared matrix is closed.*  **Yes.**  Blocker 3 was the only open blocker;
blocker 2 is built (§53) and blocker 1 is closed by decision (§52.7).  The declared
matrix of §49.5 is now closed -- **not** by migrating all three, but by migrating two and
deciding the third, which is what 54.9 audits before drawing any Phase-7 conclusion.

### 54.8 The verdict, and the two deferrals that bound it

**Phase 7 is GO on its declared exit criteria.**  All six audit questions above
answer yes, and the §49.5 matrix that has gated Phase 7 since milestone 49 is
closed: blocker 2 built (§53), blocker 3 built here, blocker 1 closed by decision
(§52.7).  This is the first milestone in this sequence able to say that, and it is
said plainly rather than hedged.

It is a GO **on those criteria**, and three things bound what it licenses.  Each
is a fact this session measured or inherited, not a caveat added for safety.

1. **Blocker 1's hole is four cells wider than when it was closed, because of
   this build.**  §52.7 decided ``TTM dds x {dd,ss} -> dds`` is not migratable
   under this origin, when that was two census cells.  Relaxing the prefix domain
   rule makes ``TTM sds x {dd,ds,sd,ss} -> dds`` pass CIN and stop at the same
   ``unsupported_schedule_auto_family``, so the auto-tile seam now refuses more
   shapes than the decision contemplated.  Every one is fail-closed with a
   structured code, so audit question 2 still answers yes -- but a Phase-8
   fallback census must budget for six cells there, not two, and the decision to
   accept the seam was taken against the smaller number.
2. **The 1139-cell frontier extension has not been run for three consecutive
   milestones.**  §52.5 ran it once; §53 and this session both put it first on the
   sacrifice list and ran the declared 748 instead.  The 748 is a strict subset:
   the extension is what covers rank 6, non-ADD update and combiner operators,
   COO and SINGLETON levels, three- and four-operand chains, and zero extents
   crossed with every receiver.  A cutover or fallback census drawn over the 748
   is a census over a knowingly partial frontier.
3. **The eleven unsound-claimed cells are still unmeasured for arm-invariance,
   and §52.8 found three of them configuration-dependent.**  Those are precisely
   the cells a fallback census has to be certain about, because "unsound" is a
   claim about what the legacy comparand does, not about what the typed route
   refuses.

**What this milestone therefore does NOT do, and what the next one may.**  No
Phase-8 inventory, cutover, cache, selector or default-dispatch change was made
here; no fallback was weakened; no legacy code was deleted.  A GO permits the
next session to run a **read-only** Phase-8 cutover/fallback census, and its first
duty is items 2 and 3 above, because both are inputs to that census rather than
follow-ups to it.  An opt-in shadow pilot, if one is run at all, belongs to the
already-proven families and not to anything this milestone added.

**One further limit that no amount of local work removes: every gate in §§49-54
is the Apple M5 machine.**  A cutover decision about release behaviour on a
branch whose whole value is generated-code equivalence should not rest on one
host, and no cross-host run exists.

### 54.9 What this section does not do

- **Blocker 1 stays closed by decision, and this change makes four more cells
  reach it.**  ``TTM sds x {dd,ds,sd,ss} -> dds`` now pass CIN and stop at
  ``unsupported_schedule_auto_family``.  Nothing here changes
  ``_apply_automatic_tiles``, which is shared with legacy default dispatch.
- **The dense-domain assembly seam is untouched.**  A COMPRESSED result level
  still may not be driven by a dense domain, so ``ds ij->i``, ``dss ijk->i`` and
  ``TTM sds x * -> dss`` stay refused.  Admitting them would mean appending one
  entry per row of a dense iteration space; that is a design decision for Bobby,
  not a gap closed on the way past.
- **A MERGED prefix domain is refused, not supported.**
- **The multi-compressed APPEND family's own dense-prefix rule is unchanged**, and
  its target now states that boundary as an explicit require.
- **The 1139-cell frontier extension was not re-run**, and neither was the heavy
  legacy sweep for the eleven unsound-claimed cells, so §52.8's scoping note
  stands unchanged.
- **No cross-host run.**  Every gate here is the Apple M5 machine.

## 55. The extended frontier, the eleven cells' arm-invariance, and a read-only Phase-8 cutover census (2026-08-11)

This section opens at inherited committed tip ``6e8e09c`` and **runs the two
measurements §54.8 named as the next session's opening duties**, then draws the
read-only Phase-8 cutover/fallback census they are inputs to.  It lands **no
production change**: ``git diff 6e8e09c..HEAD -- src/ csrc/`` is empty, so
release neutrality is proven by construction and confirmed by measurement rather
than argued.

It also corrects five inherited or prompted claims: the tip's distance from
origin (55.1), the size of blocker 1's seam (55.3), the measured reach of blocker
2's repair (55.2), the number of configuration-dependent legacy cells and the
mechanism behind them (55.4), and the mechanism §53.8 gives for legacy's
unstructured refusal (55.6).  Sections 49 through 54 are preserved above as
written.  Where this section contradicts them, this section is correct.

### 55.1 The inherited state, re-established by inspection

- **Origin is ``a3b8d1e`` and the branch is published.**  ``git ls-remote``
  resolves ``refs/heads/refactor/compiler-ir-phase3-std-move-call`` on origin to
  ``a3b8d1ed70b71c3b137a62c3b27fa01e1262e621``.  ``.git/packed-refs`` still
  carries the stale ``58e8565``/``cb49ff7`` and was not read for any claim here.
  Nothing was pushed.
- **The inherited tip ``6e8e09c`` is FOURTEEN commits ahead of origin, not
  thirteen.**  ``git rev-list --count a3b8d1e..6e8e09c`` is 14, and the log
  enumerates 14.  §54.1's own arithmetic is right — ``bb7f391`` was ten ahead,
  and ``ce52ff7``/``460bbf3``/``0821799``/``6e8e09c`` are four more.  The
  handoff prompt's "thirteen" is an off-by-one and nothing depends on it.
- The five protected tracked files hash exactly as
  ``statics/protected-hashes.txt`` records, before and after every change here.
- **Phase 7's GO was re-derived rather than inherited**, and 55.8 re-answers all
  six audit questions against this tip.

Evidence ledger: ``~/.cache/scorch-codex/phase8-census-frontier-ext/`` (new).

### 55.2 The 1139-cell frontier extension, run at last

It had been first on the sacrifice list for three consecutive milestones.  It was
run here against two pinned detached worktrees — base ``bb7f391`` and final
``6e8e09c`` — with the harness's import-origin assertion active, reading **both**
LoopPlan exits out of ``LoopPlanDiagnostic.code`` and never message text.

| | cells | admitted | defect codes | loop-plan | unclassified | arm-variant |
| --- | --- | --- | --- | --- | --- | --- |
| final ``6e8e09c`` | 1139 | **248** | 652 | 239 | **0** | 3 |
| base ``bb7f391`` | 1139 | 219 | 681 | 239 | 0 | 3 |
| final, declared 748 subset | 748 | **135** | **415** | **198** | 0 | 3 |
| base, declared 748 subset | 748 | 106 | 444 | 198 | 0 | 3 |
| extension-only 391, **both tips** | 391 | 113 | 237 | 41 | 0 | 0 |

Each row's three sums add to its cell count.

**The 748 subset reproduces the declared 135 / 415 / 198 exactly**, and the base
subset reproduces §54.1's own re-run baseline of 106 / 444 / 198 exactly.  So the
extension harness and the declared harness agree cell for cell where they
overlap, which is the precondition for trusting the other 391.

**The extension-only 391 cells are byte-for-byte identical between the two tips.**
Blocker 3 moved **33 cells, and every one of them is in the declared 748**: 29 to
``ADMITTED`` and 4 to ``unsupported_schedule_auto_family``, with the moved set in
arm 1 identical to arm 0 and **zero admitted cells lost**.  §54.8's second
deferral asked whether the extension hides cells the 748 cannot see; for blocker
3 the measured answer is **no**, and that is a negative result worth having
rather than a formality — it is the first evidence that the declared frontier is
a faithful proxy for that change.

**But the extension was badly stale, and what it was hiding is a correction to
§53.7.**  §52.5's own receipt
(``orderedkey-abi-signature-window/frontier/frontier_extended.json``) was read
directly and diffed against this session's ``bb7f391`` run cell by cell -- the
cell order matches exactly, so the diff is a real pairing and not an arithmetic
subtraction of two summaries.  **121 cells moved between those two tips in arm
0**, and the only production change in that window is blocker 2, which the 748
side confirms: 65 of the 121 are in the declared 748, which is precisely the
"exactly 65 cells" §53.7 derived and reproduced.

The other **56 are in the extension**, every one of them leaving
``sparse_parent_dominance`` -- 19 to ``unsupported_format``, 15 to
``unsupported_update_op``, 8 to ``unsupported_sparse_output_domain``, 2 to
other codes, and **12 to ADMITTED**.  So §53.7's headline is a 748-scoped number
and the repair's measured scope is larger in both columns:

| | §53.7, over the 748 | measured, over the 1139 |
| --- | --- | --- |
| newly ADMITTED | 8 | **20** (8 + 12) |
| re-dispositioned to a shape-specific code | 57 | **101** (57 + 44) |
| admitted cells lost | 0 | **0** |

The twelve extension admissions are worth naming because they are not more of
the same: ``ssssss ijklmn->ijk [sss]`` is a **rank-6** cell, ``3-factor sss x d x
d -> s`` is a **three-operand vector contraction**, and the remaining ten are
``ss ij->i [s]`` and ``sd ij->i [s]`` crossed with the zero and unit extents.
Blocker 2 admitted twelve cells nobody knew it admitted, in families the declared
envelope never names, and no differential covers any of them.

One closure the census supplies for those twelve, since it bears on how much they
matter: **they are exactly the twelve quadrant-B cells of 55.5** — the typed
route admits each one and legacy **refuses** it outright.  The two sets were
compared as sets and are identical.  So blocker 2's hidden reach carries no
equivalence risk at all (there is no legacy emission to differ from); what it
carries is twelve admitted cells with no correctness coverage, in families the
declared envelope never names.

That is the real answer to §54.8's second deferral.  The extension did not need
running for blocker 3 -- it needed running for blocker 2, one milestone earlier,
and the cost of three deferrals is that a repair's admitted set was understated
by a factor of two and a half for two milestones.

### 55.3 The blocker-1 seam is fourteen cells, not six

§52.7 decided blocker 1 closed on the ground, among others, that "it buys two
census cells".  §54.8 and the handoff prompt then instruct the next session to
"budget for SIX cells at ``unsupported_schedule_auto_family``, not two".  Both
numbers are smaller than what the frontier measures, and the two are drawn from
different enumerations, so the arithmetic ``2 + 4 = 6`` mixes them.

Measured over the declared 748, enumerated rather than sampled:

| | arm 0 | arm 1 |
| --- | --- | --- |
| base ``bb7f391`` | 10 | 8 |
| final ``6e8e09c`` | **14** | **12** |

Zero of them are in the extension.  The fourteen are four groups:

- ``TTM dds x {dd,ss,ds,sd} -> dds`` — §52.7's decided family.  Its "two census
  cells" are the two of these four that appear in the 20-cell legacy census; the
  frontier sees all four.
- ``TTM sds x {dd,ss,ds,sd} -> dds`` — the four blocker 3 newly delivers to the
  seam, which is the increment §54.8 correctly identifies.
- ``TTM ddd x {dd,ss,ds,sd} -> dds`` — four all-dense-operand cells that were
  already at the seam and are named in neither §52.7 nor §54.8.
- ``ddd ijk->k [d]`` and ``ddd ijk->ik [dd]`` — arm 0 only; both are
  arm-variant, which is why arm 1 counts 12.

None of this reopens blocker 1, and every one of the fourteen is fail-closed with
a structured code, so audit question 2 still answers yes.  What changes is the
size of the hole a Phase-8 fallback census has to budget for: **fourteen in arm
0 and twelve in arm 1**, not six.

### 55.4 The eleven unsound-claimed cells: arm-invariance measured, and a FOURTH configuration-dependent cell

§52.8 gave the eleven unsound-claimed cells the lean sweep — arm 0, ``f32`` only,
three shapes, densities ``{0.4, 0.9}`` — and said so, scoping its arm-invariance
claim to the ten sound cells.  Here the eleven get the **heavy** sweep the ten
got: five shapes × two dtypes × **both automatic arms** × three densities, each
configuration in its own disposable subprocess with ``RLIMIT_CORE`` 0, an
``RLIMIT_CPU`` and a wall-clock timeout, streaming one JSON line per density so a
route that terminates the interpreter is named by its missing line.  The ten
sound cells were re-run lean at this tip as a control.  **720 measurements.**

- **Arm-invariance holds, and is now measured rather than assumed.**  **330
  paired configurations** — every one of the eleven cells' 30 — and **zero
  arm-variant configurations**.  §50.6's arm-invariance claim for the unsound
  eleven, which §52.8 could not confirm, is confirmed.
- **LoopIR is correct on all 720**: it executed on every one and matched the
  dense reference on every one (``loopir_failed=0``,
  ``loopir_ref_mismatch=0`` for every cell).
- **Zero contradictions**: no cell claimed SOUND measured non-sound, and the ten
  sound cells are 6/6 at this tip.
- **The configuration-dependent set is FOUR cells, not three.**  §52.8 names
  ``dss ijk->k``, ``sss ijk->k`` and ``ssss ijkl->il``.  The heavy sweep adds
  **``ssss ijkl->l``, sound at 4 of 60 configurations**.

| cell | sound at | sound configurations |
| --- | --- | --- |
| ``dss ijk->k`` | 20/60 | ``(1,4,5)`` all densities; ``(3,1,5)`` and ``(3,4,1)`` at density 0.15 only |
| ``sss ijk->k`` | 16/60 | ``(1,4,5)`` all densities; ``(3,4,1)`` at density 0.15 only |
| ``ssss ijkl->il`` | 12/60 | ``(2,1,4,5)`` all densities |
| ``ssss ijkl->l`` | **4/60** | ``(2,1,4,5)`` at **density 0.15 only** |

**Why §52.8 missed it, exactly.**  Its lean sweep's density set is
``{0.4, 0.9}``.  Every sound configuration of ``ssss ijkl->l`` is at density
**0.15**, which the lean set excludes — so the miss is the density axis, not the
shape axis, and the lean sweep would have missed it at any shape coverage.

**And §52.8's mechanism is incomplete in a way that matters more than the count.**
It reads: they become sound "exactly when the outer *reduced* extent collapses to
1 — which removes the repeated pass that produces the duplicate drained
coordinates".  Measured, that describes only the columns that are sound at *every*
density.  Every *additional* sound configuration — ``dss ijk->k`` at ``(3,1,5)``
and ``(3,4,1)``, ``sss ijk->k`` at ``(3,4,1)``, and all four of
``ssss ijkl->l`` — occurs at density 0.15 and nowhere else, at shapes whose outer
reduced extent does not collapse.  At that density the drawn operand simply does
not contain the pattern that produces duplicate drained coordinates.

The honest conclusion is therefore stronger than a count correction: **legacy's
soundness for these cells is data-dependent, not merely shape-dependent.**  A
cell can measure SOUND on a sparse draw and UNSOUND at the same shape on a denser
one.  No finite sweep can certify them, and a Phase-8 fallback that routes any of
them to legacy is relying on a property that measurement can refute but cannot
establish.

### 55.5 The read-only Phase-8 cutover/fallback census

**First, the fact that frames the whole census: production never reaches the
typed route.**  ``compile_cin_via_loopir`` and ``execute_cin_via_loopir`` have
**zero non-test callers** in ``src/``; there is no dynamic import, no entry point
and no environment flag that reaches them; and ``pipeline.py``'s own module
docstring states it.  So "cutover" means introducing a call site that does not
exist, and "fallback" means what legacy does for the cells the typed route
refuses.  Nothing here creates either.

The census runs all 1139 frontier cells, compile-only, in both automatic arms,
through **three** columns: the typed route under the automatic origin; the legacy
comparand under the same empty ``Schedule()``; and the legacy comparand under
**no requested schedule at all**, which is what production default dispatch
actually executes and therefore what a fallback would actually run.

**The four cutover quadrants, against legacy DEFAULT dispatch:**

| quadrant | cells | declared 748 | extension |
| --- | --- | --- | --- |
| A typed admits / legacy emits | 228 | 127 | 101 |
| B typed admits / legacy refuses — cutover strictly ADDS | 20 | 8 | 12 |
| C typed refuses / legacy emits — **the fallback load** | **703** | 533 | 170 |
| D typed refuses / legacy refuses | 188 | 80 | 108 |

Against legacy under the same empty ``Schedule()`` the split is 228 / 20 / 485 /
406, the difference being cells the empty schedule alone pushes into a refusal.

**The load-bearing number is C = 703.**  Under any cutover that keeps a fallback,
production continues to work on 703 of 1139 frontier cells *only because legacy
answers for them* — and legacy's soundness there is characterized for exactly 21
cells, four of which 55.4 has just shown are data-dependent.

**Then the question a cutover actually turns on: on the cells it would MOVE, does
the typed route emit the same C++?**  Over the 496 admitted cell-arms:

| comparand | byte-identical | DIFFERENT | legacy refuses |
| --- | --- | --- | --- |
| legacy under the same empty ``Schedule()`` | **112** | **344** | 40 |
| legacy under default dispatch | 105 | 351 | 40 |

**So a cutover is a byte-neutral no-op on 112 of 496 admitted cell-arms and a
deliberate change of emitted code on 344.**  The differences are structural, not
cosmetic: on ``ss ij->j [s]`` the typed route emits 2,218 characters where legacy
emits 2,845, draining the workspace straight into the result and deleting
legacy's intermediate ``T0_crd_vec``/``T_val_vec`` materialization and its second
pass.  That is a better kernel, and the repository already knows it — 
``test_loopir_sparse_workspace_target`` asserts ``cpp_source != legacy_cpp`` for
such a family on purpose.

**The structure of the 112 is the finding.**  They are, essentially exactly, the
**dense-receiver** cells: 60 ``degenerate2``, 32 ``rank3``, 12 ``rank2`` and 8
``matmul`` arms, every one with an all-dense result format except
``MM ss x ss -> ss`` and ``MM ds x ds -> ds``.  The mechanism is plain — a dense
receiver has no assembly or workspace drain, so both pipelines emit the same
nest; every sparse-receiver family is where the typed route's own assembly lives,
and that is precisely where it differs.

The consequence for Phase 8 is sharp, and it is the census's main product:

> **A cutover is an equivalence-preserving refactor only for dense receivers.
> For every sparse-receiver family it is a behaviour change, and the branch's
> "generated-code equivalence" warrant does not cover it.**  Those families need
> per-family correctness evidence (which the 1,944-check differential supplies
> for the migrated ones and for nothing else) and per-family *performance*
> evidence (which no gate on this branch has ever collected, because compile-only
> latency measures the compiler, not the kernel).

**Two further census facts, recorded because they are cheap to state and
expensive to rediscover.**

- **Legacy's own default path is arm-invariant across all 1139 cells**; the typed
  route has the three known arm-variant cells; and legacy *under an empty
  ``Schedule()``* has **two** — ``ssd ijk->k [d]`` and ``dsd ijk->k [d]``, which
  emit in arm 0 and raise ``TypeError`` in arm 1.
- **An empty ``Schedule()`` is not identity on the legacy path.**  It changes
  legacy's outcome on **662 of 1139 cells**: 367 emit different source, 179 turn
  ``EMITS`` into ``InvalidSchedule``, 33 into ``IndexError``, 25 into
  ``UnsupportedFeature``, and 21 go the other way from ``ValueError`` to
  ``EMITS``.  This is characterization of a surface, not a regression — but it
  means the empty-``Schedule()`` harness measures a route that differs from
  release default dispatch on more than half the frontier, and neutrality
  arguments should not move between the two surfaces.

### 55.6 Carried item (a), decided: the unstructured refusal is not repair-induced

§53.8 records ``ValueError: ivar_j is not in list`` as a degradation the
plan-replay boundary introduced, with a stated mechanism: ``apply_schedule``
propagates only the plan, ``legacy_generated_cpp`` replays it through the legacy
lowering, and the repaired order is legal but has no legacy form.

**That mechanism is wrong, and the correction is what decides the item.**  Read
off the traceback rather than the message, the raising frame is
``compiler/cin.py:841`` in ``level_of_index_var`` — ``sorted_index_vars.index(index)``
— reached from ``compiler/cin_lowerer.py:734`` in ``lower_TensorAssign``.  The
**same frame raises the same error with no requested schedule at all**, on plain
default dispatch, which never consults ``apply_schedule`` and therefore cannot be
replaying a plan.  The lowerer is asking for the level of a *reduced* index
variable, which is by construction absent from the result's sorted index vars.

And it is not eighteen programs on one surface.  Over the 1139-cell frontier it
is **140 cells on legacy DEFAULT dispatch** — the single largest refusal class
legacy has there, ahead of 48 ``UnsupportedFeature`` and 12
``NotImplementedError``.

**Decision: record the limit deliberately, and do not repair this boundary.**
The boundary is not the defect site; repairing it would leave the 140 default-
dispatch cells untouched. A structured refusal belongs in the legacy CIN lowerer,
which is production code shared with release default dispatch and therefore out
of scope for a session licensed to be read-only — and which a cutover would
largely obviate. ``test_the_legacy_unstructured_refusal_is_not_repair_induced``
locks the corrected mechanism as an executable fact, asserting the frame
structurally and never the message, on a cell (``ss ij->i [s]``) where the typed
route *admits* the program — so the contrast a cutover would deliver is visible
in the lock itself.

### 55.7 Verification

All gates ran in the ``scorch`` conda environment on the Apple M5 development
machine.  **No production file changed**, which is the strongest form of the
release-neutrality claim and is stated first because it makes several gates
confirmatory rather than load-bearing.

**Release neutrality against ``6e8e09c``: byte-identical, and this time proven
twice over.**  ``git diff 6e8e09c..3b6b24f -- src/ csrc/`` is **empty**, so
identical source cannot produce different output; the measurement is
confirmation, not the argument.  Both arms are clean detached worktrees.  The
20-source corpus is 20/20 with no differing file, the 42-source ``ss@dd`` grid is
42/42 with no differing file, and the 86-case schedule audit reports
``total=86 admitted=46 rejected=40 nonidentical=0`` on both sides and is
identical between them.  mypy is **hash-identical** (146 lines, same SHA-256) and
so is flake8 (47 lines, same SHA-256).  black's stdout is identical and empty;
its stderr differs **only in the tree-path prefix** — the same 15 pre-existing
reformat targets in the same order, and **the changed test file appears in
neither list**.  (§54.6 could only say the file *order* differed; here nothing
differs but the path.)

**The empty-``Schedule()`` surface: unchanged in every column.**  118 cases on
each tree.  The legacy arm emits for the same **100** on both, with **identical
SHA-256 digests**, none gained or lost; the LoopIR arm emits for the same **52**,
also with identical digests; and **zero refusal records changed**.

**The 1139-cell frontier and the census** are 55.2 and 55.5; both were run
against pinned trees with the import origin asserted, and the harness's pin
caught a relative path on the first attempt rather than silently measuring the
editable install.

**The heavy legacy sweep** is 55.4: 720 crash-isolated measurements, 330 paired
configurations, zero arm-variant, zero contradictions, LoopIR correct on all 720.

**No compiled differential was re-run, and that is deliberate rather than
skipped.**  The 1,944-check public + erasure/oracle differential proves
properties of production code that this session did not touch; re-running it
would re-measure an unchanged tree.  Its result stands as §54.6 recorded it, and
55.9 states plainly which cells it does *not* cover.

**Full non-performance suite at the production+test tip ``3b6b24f``**, eight
file-disjoint partitions in fresh processes with per-partition ``TMPDIR``,
``XDG_CACHE_HOME`` and ``TORCH_EXTENSIONS_DIR``: **6,309 selected nodes, 6,294
passed, 15 skipped, 3 deselected, 0 failures and 0 errors, every partition
exiting 0**, all eight at one revision.  Tallied from the eight JUnit XMLs rather
than from the driver's own JSON, and the JUnit total equals the pre-run selected
count exactly.  The pre-run proof places all 88 selected modules of the 90
tracked exactly once, shows the partition loads
[788, 791, 788, 790, 790, 788, 787, 787] summing to 6,309, and reports module and
node partitions both complete and both disjoint.  The detached worktree was clean
before and after.

**The node delta is accounted exactly, and it is +27 with nothing removed.**
``6e8e09c`` collects **6,282**; ``3b6b24f`` collects **6,309**.  Every one of the
27 is in ``test_loopir_bound_prefix_target``, which goes from 103 to 130 nodes:
**+26** from expanding the re-dispositioned-neighbour parametrize from 2 cells to
28, and **+1** for
``test_the_legacy_unstructured_refusal_is_not_repair_induced``.  **Zero nodes
removed**, so unlike §54.6 there is no moved-or-renamed accounting to do.

Notably, the suite found **nothing** this time — no inherited lock names a shape
this session changes, because this session changed no production behaviour.
§54.6's six-failure first pass is what a production change looks like; this is
what a measurement-only change looks like, and running the suite is what
distinguishes the two rather than asserting it.

**Gates 1 and 2 are read out of those same JUnit XMLs** rather than from a
separate invocation, which is stricter: the suite ran every module exactly once
at one revision, so the two gates cannot disagree with the suite total.  Gate 1,
the five target files — both hosts, the row-scope module, the bound-prefix module
and the rank-2 row-scope precedent: **522 nodes, 521 passed, 0 failed, 0 errors,
1 skipped** (§54.6's 495/494/1 plus this session's 27).  Gate 2, the eleven
adjacent memberships: **1,180 nodes, 1,180 passed, 0 failed, 0 errors, 0
skipped** — identical to §54.6, as it must be for an unchanged production tree.

**Representation unchanged.**  v11 stands; no node kind, no canonical schema
change, no request- or schedule-identity change.

**No default dispatch, cache, selector or fallback change**, and no legacy code
removed.

**Paired compile-only latency.**  The ceiling for a compile-only integrity boundary on
this branch is **1.10**.  Base is the detached worktree at ``6e8e09c``; the candidate is
the final code tip ``3b6b24f``.  Each measurement is a fresh subprocess importing exactly
one source tree and timing the same 40-cell ordered-key compile-only grid (no JIT, no C++
compiler); 20 rounds alternating the within-round order, 4 warmups and 21 samples per
process, plus a base-against-base A/A control in every round.  The grid checksum is
asserted equal across all three measurements of every round -- **141162 over 40 cells, in
all 60 measurements**, the same value §54.6 recorded, which is what an unchanged
production tree requires.  It ran ALONE: the suite, the census and the sweep had all
finished, and their stray pool workers were reaped first (nine were still resident).

| statistic | min | p50 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| A/B ratio (median) | 0.9722 | 1.0009 | 0.9993 | 1.0069 | **1.0133** |
| A/A control (median) | 0.9717 | 1.0003 | 1.0014 | 1.0249 | **1.0301** |
| A/B ratio (min-of-samples) | 0.9882 | 0.9994 | 0.9998 | 1.0052 | **1.0124** |
| A/A control (min-of-samples) | 0.9856 | 1.0017 | 1.0015 | 1.0140 | **1.0259** |

Pooled fastest-sample A/B 0.9967; pooled A/A 0.9977.  Order controls: base-first mean
0.9971, candidate-first mean 1.0015.

**Every declared statistic, including the min-of-samples row, is at or below 1.10; the
largest anywhere is 1.0301.**  Headroom against the ceiling is ~6.7%.

**Unlike §54.6, the largest statistic is an A/A control number, and the reading is
correspondingly simple.**  On both rows the A/B maximum sits *below* its A/A counterpart
(1.0133 against 1.0301; 1.0124 against 1.0259), and the A/B mean is **0.9993** -- below
unity.  That is exactly what should happen, because **the two trees' ``src/`` are
byte-identical**: this comparison is structurally an A/A, and what it measures is this
machine's noise floor rather than a cost of this change.  It is reported that way rather
than presented as a passed cost gate.  Its value to the next session is the calibration:
on this host, a 40-cell compile-only grid carries roughly +-1.5% round-to-round noise at
the median and about +-1.3% at the min-of-samples statistic, so any future A/B under
about 1.03 is indistinguishable from nothing.

### 55.8 Phase-7 exit audit, re-run against this tip

The GO is re-derived here, not inherited.  Each answer is against ``@@TIP@@``.

*Migrated families complete over their proven envelopes.*  **Yes**, and
unchanged: no production file differs from ``6e8e09c``, so the 1,944-check
compiled public + erasure/oracle differential and every envelope it proves stand
exactly as §54.6 recorded them.  The scoping that goes with that answer is now
quantified rather than gestured at: the extension admits **113 cells no
differential covers at all**, which is admission and not verified correctness —
§52.9's note, restated with a number.

*Every neighbour carries a stable fail-closed disposition.*  **Yes, and this is
the strongest form the question has ever been answered in** — over **1139**
cells rather than 748, in both automatic arms, reading both LoopPlan exits: 652
defect codes, 239 loop-plan diagnostics, **zero unclassified**, and no loop-plan
cell carrying a ``defect`` attribute.  The fourteen cells at blocker 1's seam
each keep a structured code.

*Representation unchanged.*  **Yes.**  v11 stands; no node kind, no canonical
schema, no request- or schedule-identity change — trivially, because no
production file changed.

*Release behaviour unchanged.*  **Yes, by construction and by measurement.**
``git diff 6e8e09c..@@TIP@@ -- src/ csrc/`` is empty, so identical source cannot
produce different output; the corpus, grid and audit confirm it.

*Compiler latency within the declared ceiling.*  **Yes** — see 55.7.  The
comparison is structurally an A/A (the two trees' ``src/`` are byte-identical),
so what it measures is this machine's noise floor rather than a cost of this
change, and it is reported that way.

*The declared matrix is closed.*  **Yes**, unchanged from §54.7: blocker 2 built,
blocker 3 built, blocker 1 closed by decision.  55.3's correction resizes the
hole that decision accepted — fourteen cells, not two or six — but every one is
fail-closed with a structured code, so it does not reopen the blocker.

**Phase 7 therefore stays GO**, and nothing the extension or the sweep turned up
fails one of the six questions.  What they turned up bears on Phase 8.

### 55.9 The verdict, split in two because the evidence splits in two

**Phase 7: GO, re-derived.**  All six audit questions answer yes against this
tip, on this session's own measurements rather than on §54's.

**A Phase-8 cutover: NO-GO on this evidence.**  This is not a reversal of the
Phase-7 GO and not a new blocker in the §49.5 sense; it is the census answering
the question it was commissioned to answer.  Three findings block it, each
measured here:

1. **The cutover is not equivalence-preserving where it matters.**  Of 496
   admitted cell-arms it is byte-neutral on **112** — essentially exactly the
   dense receivers — and changes emitted code on **344**.  Every sparse-receiver
   family is on the changed side.  The branch's warrant is generated-code
   equivalence; for those families a cutover forfeits it deliberately, and would
   need per-family correctness evidence (the differential covers the migrated
   families and **113 admitted cells have none**) plus per-family **kernel**
   performance evidence, which **no gate on this branch has ever collected** —
   compile-only latency measures the compiler, not the emitted kernel.
2. **The fallback cannot be certified.**  703 of 1139 cells would fall back to
   legacy.  Legacy's behaviour there is characterized for 21 cells, and 55.4
   shows four of those are **data-dependent** — sound on a sparse draw, unsound
   at the same shape on a denser one.  A fallback census cannot establish
   soundness for such cells by any finite sweep; it can only fail to refute it.
3. **One host.**  Every gate in §§49-55 is the Apple M5 machine, and the
   cross-host run is blocked on interactive authentication rather than declined
   (55.10).  A decision about release behaviour on a generated-code-equivalence
   branch should not rest on one host, and this is the fourth consecutive
   milestone to say so.

**What is licensed next, on this evidence.**  A **dense-receiver-only** shadow
pilot is the one cutover-shaped step the census actually supports: it is the
region where typed and legacy emission are byte-identical, so it is
equivalence-preserving by measurement rather than by argument, and it is 112
cell-arms wide with the enumeration in this ledger.  Everything else needs
evidence that does not exist yet.

### 55.10 What this section does not do

- **No production change of any kind.**  No dispatch, cache, selector or fallback
  change; no legacy code removed; no cutover, no shadow pilot, not even an opt-in
  one.  The census is read-only and the two commits are test and documentation.
- **Blocker 1 stays closed by decision.**  55.3 resizes its hole from the numbers
  §52.7 and §54.8 state; it does not reopen the decision.
- **The dense-domain assembly seam is untouched**, as instructed: ``ds ij->i``,
  ``dss ijk->i`` and ``TTM sds x * -> dss`` stay refused, and whether a
  COMPRESSED result level may be driven by a dense domain remains a design
  decision for Bobby.
- **The legacy CIN lowerer's unstructured refusal is recorded, not repaired.**
  55.6 gives the corrected mechanism and the reason: the replay boundary is not
  the defect site, and the site is production code shared with release default
  dispatch.
- **``unsupported_union_with_dense`` is still unlocked.**  §53.7's fourth code
  needs a two-operand combiner that the reduction helper cannot express.
- **No cross-host run, and this time it is blocked rather than deferred.**
  ``redwood`` is reached through the ``myth`` proxy, which accepts only
  keyboard-interactive authentication (password + Duo); the long-lived control
  master has expired and a valid Kerberos ticket is not sufficient.  Every
  harness in this ledger takes a tree root as its first argument and will run
  there unchanged once a master exists.

## 56. The cross-host run: three hosts, zero divergence, and the comparand the cutover question turns on (2026-08-11)

This section opens at inherited committed tip ``c13b45c`` and takes **carried
item (b)** — the cross-host run that §§52–55 recorded as blocked rather than
declined.  Bobby unblocked both ``redwood`` and the MKT allocation, so it ran.
It is a **read-only measurement session**: no production file changed, no
dispatch, cache, selector or fallback was touched, no legacy code deleted, and
no cutover or shadow pilot was started.

Sections 49 through 55 are preserved above as written.  Where this section
contradicts them, this section is correct.

### 56.1 What the single-host caveat actually was

Every gate in §§49–55 ran on one machine — the Apple M5.  §54.8 stated the limit
plainly ("one further limit that no amount of local work removes"), and §55.9
promoted it to the top of the next session's duties on the grounds that if the
1139-cell dispositions and the byte-equivalence split fail to reproduce on
another host, "that is the most important thing anyone could learn about this
branch".

The caveat is not idle worry.  The frontier and census read their dispositions
off exception types and defect codes produced by a Python compiler pipeline, and
the equivalence census compares *generated C++ text*.  Anything host-dependent
in that chain — dictionary or set iteration order leaking into emitted names, a
``torch`` version changing a dtype or shape inference result, a platform branch
in the ABI or policy layer, floating-point formatting in an emitted literal —
would show up as a different code or a different digest.

### 56.2 The three hosts, and why they are a real contrast

| | M5 (inherited) | redwood | mkt1 |
| --- | --- | --- | --- |
| arch | **arm64** | x86_64 | x86_64 |
| CPU | Apple M5 | Intel i9-14900K | **AMD** EPYC 9334 |
| OS | macOS 26.4.1 | Ubuntu 22.04.4 | Ubuntu 24.04.4 |
| compiler | Apple clang 21.0.0 | g++ 11.4.0 | g++ 13.3.0 |
| Python | 3.11.15 | 3.11.15 | **3.12.3** |
| torch | **2.13.0** | **2.5.1** | **2.10.0**+cu128 |

Two architectures, three CPU vendors, two operating-system families, two
compiler families across three versions, two Python minor versions, and **three
torch versions spanning 2.5.1 to 2.13.0**.  The torch spread matters most: it is
the dependency the pipeline actually consults for dtypes, shapes and the
extension ABI, and an eight-minor-version gap is a far more aggressive test than
a second Linux box would have been.

### 56.3 Provenance, established rather than asserted

- **The measured source is the same source on all three.**  ``redwood`` runs a
  pinned detached ``git worktree`` at ``c13b45c``, clean; ``mkt1`` runs a
  ``git archive`` of the same commit.  Both independently compute a manifest
  digest — sha256 over the sorted sha256 of every ``src/**/*.py`` — and both
  produce ``1258fea6c2626e11cb1e8a75453a501e6fd595808952c6d3437517b7d4423850``.
  The two hosts were reached by different transports and neither digest was
  copied from the other.
- **``c13b45c`` is a valid stand-in for the tip the M5 census ran at.**
  ``git diff 6e8e09c..c13b45c -- src/`` is empty; the two intervening commits are
  test and documentation.
- **Nothing was pushed.**  Origin is still ``a3b8d1e``; commits moved to
  ``redwood`` as a ``git bundle`` and to ``mkt1`` as a source archive.
- **The extension is built from the measured tree on each host**, in place
  (``build_ext --inplace``), so ``PYTHONPATH`` resolves both ``scorch`` and
  ``scorch_ops`` inside the pinned tree.  This is not cosmetic: ``redwood``'s
  shared ``scorch`` env had ``scorch_ops`` built from an unrelated
  ``perf/spmm-fastpath`` checkout, and every harness asserts its import origin,
  so the confound is excluded by construction and then measured.  Bobby's
  ``perf/spmm-fastpath`` checkout and its conda env were left untouched.
- **MKT storage policy respected**: the tree, the build, ``TMPDIR`` and all run
  artifacts live under ``/scr/bobbyy/refactor-crosshost`` on the node, reached
  through Slurm (``sbatch -p mkt --account=mkt``, job 16736984, COMPLETED, exit
  0:0, 1m56s); ``/matx`` holds only the input archives and a copy of the JSON
  receipts.

### 56.4 The result: zero divergence, on the records and not on the summaries

All three harnesses were re-run end to end on both new hosts and compared
**record by record on the full parsed JSON**, not on headline counts — two runs
can agree on every summary and disagree on a defect code.  The positional pairing
is itself justified rather than assumed: the comparator asserts equal length
*and* an identical ``(family, name[, arm])`` sequence before it compares, and
refuses to compare positionally otherwise.

| harness | records | redwood vs M5 | mkt1 vs M5 | redwood vs mkt1 |
| --- | --- | --- | --- | --- |
| frontier (extended 1139) | 1139 | **IDENTICAL** | **IDENTICAL** | **IDENTICAL** |
| dual cutover/fallback census | 1139 | **IDENTICAL** | **IDENTICAL** | **IDENTICAL** |
| byte-equivalence census | 496 | **IDENTICAL** | **IDENTICAL** | **IDENTICAL** |

Zero differing records anywhere: every ``route``, every defect ``code``, every
``paths`` and ``stages`` list, every diagnostic count, every
``carries_defect_attr`` flag, and every generated-source digest.  Consequences
worth stating separately, because each was previously single-host:

- **248 admitted / 652 defect codes / 239 loop-plan diagnostics / 0 unclassified
  / 3 arm-variant** reproduces exactly, on both new hosts.
- The four cutover quadrants **A 228 / B 20 / C 703 / D 188** reproduce exactly.
- The **662-cell** empty-``Schedule()``-is-not-identity result reproduces exactly,
  including its per-transition breakdown.
- The byte-equivalence split reproduces exactly on **both** comparands.

**The single-host caveat carried by §§49–55 and named as duty 1 by §55.9 is
discharged.**  Generated-code equivalence on this branch is not an artifact of
the M5, of macOS, of clang, of arm64, or of one torch version.

### 56.5 A correction that the second comparand forces: the number is 105, and the set is exact

§55.5 reports the byte-equivalence split against two comparands, in a table, and
both numbers reproduce on all three hosts:

| comparand | byte-identical | different | legacy refuses |
| --- | --- | --- | --- |
| legacy under the same empty ``Schedule()`` | 112 | 344 | 40 |
| legacy under **default dispatch** | **105** | 351 | 40 |

The numbers are not in dispute.  What this section corrects is **which one the
conclusion is drawn from**.  §55.5 builds both its headline ("a cutover is a
byte-neutral no-op on 112 of 496 admitted cell-arms") and its structural finding
("the structure of the 112 is the finding … essentially exactly the
dense-receiver cells … every one with an all-dense result format **except**
``MM ss x ss -> ss`` and ``MM ds x ds -> ds``") on the **empty-Schedule()** row.

That is the wrong row for that conclusion, **by §55.5's own rule**.  The same
section establishes that an empty ``Schedule()`` is not identity on the legacy
path — it changes legacy's outcome on 662 of 1139 cells — and concludes that
"neutrality arguments must not migrate between that surface and release default
dispatch".  A cutover changes what **default dispatch** emits.  So the
cutover-relevant comparand is the second row, and the honest headline number is
**105**, not 112.

Measured, the correction makes the finding **stronger and exact** rather than
weaker.  Exactly **7 cell-arms** separate the two rows, and the reverse direction
is empty (nothing is default-identical without being empty-identical):

| cell-arm | receiver |
| --- | --- |
| ``rank3 dsd ijk->ik [dd]`` arm1 | dd |
| ``rank3 ddd ijk->ik [dd]`` arm1 | dd |
| ``matmul MM ss x ss -> ss`` arm0, arm1 | **ss** |
| ``matmul MM ds x ds -> ds`` arm0, arm1 | **ds** |
| ``matmul MM ds x dd -> dd`` arm1 | dd |

**The two cells §55.5 had to name as exceptions to its own dense-receiver rule
are exactly the sparse-receiver members of the 7.**  Drop to the correct
comparand and they drop out with it:

- the **112** has receiver formats ``{d: 87, dd: 21, ss: 2, ds: 2}`` — **4
  sparse-receiver cell-arms**, which is why §55.5 needed the word "essentially"
  and an explicit exception clause;
- the **105** has receiver formats ``{d: 87, dd: 18}`` — **zero** sparse-receiver
  cell-arms.

So the census's main product should be stated without the hedge:

> **Against the comparand a cutover actually moves, the typed route is
> byte-identical to legacy on exactly the dense-receiver cell-arms — 105 of 496 —
> and on no sparse-receiver cell-arm at all.**

This does not change the Phase-8 verdict; it sharpens its basis.  §55.5's
conclusion ("a cutover is an equivalence-preserving refactor only for dense
receivers; for every sparse-receiver family it is a behaviour change") was right,
and was being argued from a surface that contained four counterexamples to it.
On the correct surface there are none.  The practical consequence is for
§55.9 duty 2: **a dense-receiver shadow pilot is 105 cell-arms, not 112**, and
the four ``matmul`` sparse-receiver arms must not be carried into it on the
strength of an empty-``Schedule()`` byte match.

### 56.6 What this section does not do

- **No production file changed.**  ``git diff c13b45c..HEAD -- src/`` is empty.
  The only tracked changes are this section and a handoff section.  (``CLAUDE.md``
  was also corrected — its build instructions still described a ``setup.py`` and a
  top-level ``csrc/``, neither of which exists at this tip — but it is listed on
  line 1 of ``.gitignore`` and is therefore untracked and outside this or any
  commit.)
- **The working tree carries pre-existing uncommitted drift** unrelated to the
  refactor (``src/scorch/__init__.py``, ``gpu.py``, ``src/scorch/csrc/cuda/``,
  packaging and resource tests).  None of it was committed, and none of it can
  have reached a measurement: all three hosts ran **pinned** trees at ``c13b45c``
  — a clean detached worktree on ``redwood`` and a ``git archive`` on ``mkt1`` —
  never this working tree, and both independently reproduce the same ``src/``
  manifest digest.
- **The Phase-8 cutover verdict is unchanged: NO-GO.**  56.4 removes the
  single-host caveat, which was a *reason to distrust the census*, not one of the
  census's reasons to refuse a cutover.  Those are untouched: 703 fallback cells
  resting on a legacy characterized for 21, four of them data-dependent; 344
  cell-arms whose emitted code changes; and no kernel-runtime harness anywhere on
  this branch.
- **Phase 7 stays GO**, on the §54.8 criteria, unchanged.
- **No blocker was reopened, none was closed.**  Blocker 1 in particular is
  untouched.
- **The heavy legacy sweep and the full suite were not re-run cross-host.**  Only
  the three census harnesses were.  The sweep's value is a claim about *legacy's*
  soundness, which 55.4 has already shown is data-dependent and therefore not
  certifiable by adding hosts; the full suite is a claim about *this tree*, worth
  running on x86 but not a census input.  Both are stated as not run rather than
  folded into the reproduction claim.
- **Latency was not measured cross-host**, and should not be read into these
  runs: wall-clock differed widely (frontier 10.6 s on redwood; the whole MKT job
  1m56s) and nothing here is a paired A/B.

**Evidence ledger** ``~/.cache/scorch-codex/crosshost-phase8-census/`` —
``receipts/`` (nine JSONs, three harnesses × three hosts), ``hosts/`` (toolchain
manifests), ``provenance/`` (digests, the sbatch script, the full Slurm job log),
``compare/`` (``compare_hosts.py`` and ``comparand_structure.py``, both taking the
ledger root as ``$1`` and defaulting to their own location — never a hardcoded
path — with their outputs retained), and ``SHA256SUMS``.

## 57. The kernel-runtime measurement, blocker 1 reopened and fixed, and two corrections to what the census compares against (2026-08-11)

Starts from committed tip ``a8f2954``.  Bobby made two decisions that this
session acts on and does not relitigate: the refusal on dense-domain assembly is
lifted under a structural guard, and blocker 1 is to be FIXED in the layer where
it is wrong rather than preserved so that the shipped C++ stays unchanged.  His
words on the second: "it's okay to have different emitted code, but the new
emitted code must be strictly correct and more performant than legacy on a wide
variety of problem sizes."

That changes what these families have to prove.  Byte-identical emission is no
longer the primary standard for them; strict correctness plus measured *kernel
runtime* is.  ``CLAUDE.md`` has been updated to record this, since it is the first
thing a fresh session reads.

### 57.1 The measurement that never existed: a kernel-runtime harness

Every performance check in §§49–56 measured **compile-only latency**, which
measures the compiler.  §55.5's claim that the typed route emits "a better
kernel" is an INSPECTION claim from a character count, and §55.9 duty 2 says it
"should be *run*, not read".  It is now runnable.

Both pipelines converge on one point — ``_load_validated_prepared_kernel`` then
``module.evaluate(*module_args)`` — so the harness patches that single loader,
runs each production entry ONCE, and captures the exact module and argument tuple
production used rather than reconstructing them.  Timing is ABBA-interleaved with
auto-calibrated reps, and every configuration carries a same-binary control **on
each timed column** to establish the noise floor, per the repo convention that a
performance claim is never judged against a fixed ratio constant.  Crash
isolation is not boilerplate: one disposable subprocess per configuration,
because the very first baseline the harness was pointed at segfaults.

**CORRECTION to this subsection, made when the receipts were re-read.**  "Two
controls, one per side" is wrong, and wrong in a way that does not reconcile with
§57.7's own count.  The harness times THREE columns (``legacy_empty``,
``public_einsum``, ``typed``; ``legacy_default`` is excluded after its segfault)
and controls each, so a timed configuration carries three controls.  Of the grid's
44 configurations only **40** time at all — the four ``ss ij->j [d]``
configurations are TYPED-REFUSED, so there is no typed side to pair against — and
40 x 3 is exactly the 120 controls §57.7 reports.  Two per configuration over 44
would be 88, a number that appears nowhere.

**A trap worth recording.**  Every compile-only census harness builds options with
``from_environment(environ={})``.  That is correct for them and part of why they
reproduce across three hosts, but it pins ``executable_search_path`` to
``/bin:/usr/bin``, and ``ninja`` lives in the conda env.  Anything that BUILDS must
use the real environment.

### 57.2 CORRECTION: §55.5's flagship example is drawn from the wrong baseline

§55.5 reports ``ss ij->j [s]`` as typed 2,218 characters against legacy's 2,845,
"deleting legacy's intermediate ``T0_crd_vec``/``T_val_vec`` materialization and
its second pass".  Measured at the frontier's own extents:

| surface | chars | has ``T0_crd_vec``/``T_val_vec`` |
| --- | --- | --- |
| typed | 2,218 | — |
| legacy under empty ``Schedule()`` | 2,845 | yes |
| legacy under **default dispatch** | **1,993** | **no** |

The 2,845 is the **empty-Schedule()** row — the baseline §56.5 itself rules out
for reasoning about a cutover.  On the default-dispatch row legacy emits *fewer*
characters than typed and there is no second pass to delete.  The argument as
written does not survive its own choice of baseline.

**AMENDED BY §57.3, which was measured after this subsection was drafted, and
which partly rehabilitates §55.5.**  The sentence above rests on §56.5's ruling
that default dispatch is what a cutover would move.  §57.3 then measured
production itself and found that ruling too strong: for this very cell
**production emits 2,848 characters WITH a COO workspace**, i.e. within three
characters of the 2,845 row and carrying the materialization the 2,845 row is
marked "yes" for, while the 1,993/1,996 default-dispatch column is not runnable
and is not what production executes.  So the empty-``Schedule()`` row is the
production-faithful one for this cell and the default-dispatch row is not.

What survives is narrower and still worth recording: §55.5 reached the right row
for the wrong reason, having never checked which row production emits, and the
1,993 column shows that "typed deletes a second pass" is not a property of legacy
in general.  What does NOT survive is the claim that §55.5's argument fails
BECAUSE of the row it drew on.  §57.3's own conclusion is the governing one —
neither census column faithfully models production dispatch, and which is closer
depends on the family.

### 57.3 The default-dispatch baseline is not runnable, and not production

Legacy's default-dispatch source for that cell declares ``std::vector<float>
C_values;`` and then executes ``C_values[pC0] += A_val[pA1]`` against it, never
growing it, while ``C0_crd`` grows by ``emplace_back``.  A zero-length vector,
indexed and written on the first iteration.  It **segfaults at every size tried**.
Nothing on this branch had ever executed it, because the census is compile-only
and §50.6's soundness measurements went through the public ``scorch.einsum``
route.

**And it is not what production runs.**  ``legacy_generated_cpp`` with
``requested_schedule=None`` takes its ``else`` branch (``pipeline.py:594``) —
``normalize_cin`` then lower — and **never runs the auto-scheduler**, which
production's ``scorch.einsum`` does.  For this cell production emits 2,848
characters with a COO workspace; the "default dispatch" column emits 1,996 with no
workspace and the out-of-bounds write.  Driving the *public* route under both
option sets emits byte-identical source on 6 of 6 reductions and 72 of 78 cases
across TTM and matmul — so on production's own entry point an empty ``Schedule()``
IS identity, and §55.5's "662 of 1139" is a property of a test-only entry.

This does not simply restore the 112.  For **matmul**, production frequently
resolves a PREBUILT kernel and emits no generated code at all
(``MM ds x ds -> ds``, ``MM ds x dd -> dd``), which neither column describes.  The
defensible statement is: **neither census column is a faithful model of production
dispatch, and which is closer depends on the family.**  A dense-receiver shadow
pilot's membership — 105 or 112 — should be re-derived against production's actual
emission before it is acted on.

### 57.4 Blocker 1: legacy is broken on twelve of the fourteen cells

§52.7 closed blocker 1 partly because changing the heuristic "changes default
generated code" that every user gets.  Nobody had measured that code.  Membership
was re-derived from the sealed frontier receipt (never from prose): 14 arm-0 / 12
arm-1, exactly §55.3's four groups.

The **heavy** sweep — the wide grid, five shapes (ragged, unit and singleton
extents) × two dtypes × both automatic arms × three densities, one disposable
subprocess per configuration, **780 measurements** against the pinned base:

| cells | public route | legacy's lowering of the same CIN |
| --- | --- | --- |
| 11 of the 12 ``TTM * -> dds`` | **0/60 SOUND** each | **0/60** — SIGSEGV |
| ``TTM ddd x ss -> dds`` | **12/60 SOUND** | 0/60 — SIGSEGV |
| ``ddd ijk->k [d]`` | 0/30 (``TypeError``) | **30/30 SOUND** |
| ``ddd ijk->ik [dd]`` | **30/30 SOUND** | 30/30 SOUND |

Totals: ``bare_cin`` is **720 segfaults / 60 sound**; ``public`` is 672
``TensorIndexError``, 42 SOUND, 36 non-compiling, 30 ``TypeError``.

**A CORRECTION to this section's own first draft, which the lean sweep caused.**
It claimed all twelve TTM cells fail "at every extent, dtype, density and arm".
That is drawn from ONE shape and is false for ``TTM ddd x ss -> dds``, which is
sound at 12 of 60 — and **all twelve sound configurations are at shape
``[1, 4, 5, 3]``, i.e. outer extent ``i = 1``**, across every density, both dtypes
and both arms.  This is §52.8 → §55.4 repeating: a narrow sweep missing
configuration-dependence a wide one finds.

The mechanism CONFIRMS the illegality argument rather than weakening it.  The tile
is illegal because hoisting ``j_out`` above ``i`` destroys the ``(i, j)``
lexicographic order the ``dds`` receiver's compressed level is assembled in.  At
``i = 1`` there is no order to destroy, so legacy is accidentally correct exactly
where the illegality cannot manifest, and broken everywhere else.

**The gap this created, and its closure.**  §57.6's affected-cell correctness run
used a single shape, ``(10,12,14)×(14,8)``, which is not ``i = 1`` — so for those
twelve accidentally-correct configurations the candidate's PUBLIC-route behaviour
was initially unverified, and a failure there would have been a genuine
correct→broken regression.  It was then measured directly, all twelve
configurations on both trees:

| | SOUND |
| --- | --- |
| base ``a8f2954`` | **12/12** |
| candidate | **12/12** |

No regression.  The reason is the same mechanism: at ``i = 1`` the guard removes a
tile that was harmless because there was no ordering for it to break, and the
untiled kernel is correct as well.  So the fix is correct→correct exactly where
legacy was accidentally correct, and broken→broken everywhere else.

Setting that cell aside, the §52.7 objection survives for one cell and half of
another, and both have dense receivers where the fix does nothing by
construction.

### 57.5 The fix, and why it needed two layers

``apply_schedule`` already refuses an affine tile over a non-dense result for an
explicitly requested schedule (``scheduler.py:2994``, "tiled sparse-output
assembly is unsupported").  That check reads ``schedule.tiles``, which is empty on
the automatic origin — so it does nothing exactly where the automatic heuristic
then chooses tiles itself.  The automatic path was constructing the artifact the
explicit path forbids, and nothing re-checked it.

Two false starts, both caught by measurement:

- ``_has_dense_output`` is unusable as a post-insertion guard.  It walks the
  ``ForAll`` chain to a ``TensorAssign``, and workspace insertion puts a ``Where``
  there, so past that point it answers **False for every receiver** — using it
  disabled all automatic tiling and broke ``ddd ijk->ik [dd]``, which had been
  SOUND.  Replaced by ``_has_proven_non_dense_receiver``, which reads the single
  non-``Workspace`` receiver and fires only on a proven non-dense format.
- **The heuristic is duplicated.**  ``loop_plan_legality._derive_auto_decisions``
  independently re-derives it to verify the plan.  Fixing only the scheduler landed
  all twelve cells on ``InvalidSchedule/auto_tile_decision``.  The two layers must
  move together, and each now carries a comment saying so.

### 57.6 What the fix measures, on pinned base and candidate worktrees at ``a8f2954``

| check | result |
| --- | --- |
| frontier, 1139 cells × both arms | **24 cell-arms** ``unsupported_schedule_auto_family`` → ADMITTED; **0 admitted lost**, **0 unclassified**, arm-invariant; nothing else moved.  248/652/239 → 260/640/239, base reproducing §55.2 exactly |
| correctness of the newly admitted | **720/720** — 12 cells × 5 shapes × 2 dtypes × 2 arms × 3 densities, all executing, all matching a dense reference, all with well-formed assembled storage |
| production emission, 506 case-arms | 104 changed, **all TTM**; nothing else in the corpus moved |
| were the 104 correct before? | **104/104 broken → broken. Zero regressions.** Zero correct on either tree *at the corpus's configuration* — see the scope note below |
| failure mode of the 104 | base **52 SIGSEGV** / 44 ``TensorIndexError`` / 4 ``RuntimeError`` / 4 build failure → candidate **0 SIGSEGV** / 44 / 4 / 32 build failure / 24 ``IndexError`` |

**A SCOPE NOTE this table needs, and §57.4 forces.**  The 104 verdict is measured
at the production corpus's single configuration, ``(10,12,14)×(14,8)``.  That is
enough to say those 104 case-arms were broken on both trees, and it is NOT enough
to say the underlying cells never worked anywhere: §57.4 measures
``TTM ddd x ss -> dds`` SOUND in 12 of 60 configurations, all at outer extent
``i = 1``, and 12/12 on both trees.  So the honest reading is "no case-arm in this
corpus regressed", not "these cells never worked" — and the ``i = 1`` behaviour is
correct-to-correct, which §57.4 measured directly rather than inferring.

**52 segfaults become zero.**  The emission change is not neutral and is not a
regression: it is a strictly better failure mode on case-arms that did not work at
any configuration measured here, plus
twelve cells the typed route can now compile correctly in both arms — which also
retires §52.7's second cost, that this would be the first arm-variant migrated
family.  It is not; both arms admit identically.

The candidate's ``IndexError`` is unstructured, and that is stated rather than
hidden.  It arises on the LEGACY path, which has no structured-diagnostic contract
at all (§55.6 records 140 cells raising bare ``ValueError`` on default dispatch
today); the requirement to refuse with a structured code governs the typed route,
where the candidate frontier measures **0 unclassified**.

### 57.7 Kernel runtime: real wins, and a real regression

Noise floor across the grid: **0.896–1.046** over 120 same-binary controls.
119 of the 120 span 0.946–1.046 and are typically within ±2%; the outlier is a
single contended excursion — ``sss ijk->ik [ss]`` (256,128,64) at d=0.05,
``legacy_empty`` column, **0.896** — in a record whose status is OK, so it is part
of the floor and not an excluded sample.  An earlier draft of this line quoted
0.946 as the floor, which is the second-lowest control; calibrating against it
would understate the contended run's own noise by 5 points.  No win or regression
reported below is inside even the widened floor.  Ratios are legacy-time ÷
typed-time, so above 1 means the typed route is faster.

Re-run on a QUIET machine (nothing else scheduled), which tightened the floor to
**0.977–1.044** over 120 controls and reproduced the contended run's pattern and
magnitudes:

- **Wins**: ``TTM dss x ss -> dss`` at density 0.001 — **1.749×** and 1.487×;
  ``TTM dss x dd -> dss`` 1.164×; ``ss ij->j [s]`` 1.138–1.202×; ``ds ij->j [s]``
  1.162–1.205×.
- **Regressions**: the same TTM cells at density 0.05 — **0.359×** and **0.398×**,
  i.e. the typed kernel is up to **2.8× slower**, plus 0.804× at a third shape.
- **Neutral**: every rank-3/4 reduction, all of ``sd ij->j``, and the forced-sparse
  ``dd ij->j [s]`` control at 0.999–1.006 — the sanity check that the harness
  reports a same-binary comparison as one.

The pattern is density.  **"The typed route emits better code" does not hold as a
general performance claim**; it holds at low density and inverts as density rises.
Reported as a finding rather than tuned around.  Still single-host: a second host
is owed.

> **SUPERSEDED IN PART BY §58.**  "The pattern is density" is a proxy, and a
> later session identified the variable it proxies for.  The regression tracks
> the thread count legacy's own emitted pragma requests,
> ``scorch_nthreads(stored_ij, rows) = min(rows/16, stored_ij/500)``, which
> orders all eight TTM measurements monotonically and explains the shape
> dependence density alone cannot.  §58 owns the mechanism; the measurements in
> this subsection stand unchanged.

**The tile-legality fix itself is runtime-NEUTRAL.**  Base against candidate on
the same grid: typed median ratios span **0.970–1.036**, entirely inside the
combined noise floor of 0.961–1.044, on the **40 of 44** configurations the typed
route compiles, and legacy's own ratios move no further.  The fix buys correctness
and costs no measured runtime.  The remaining four are ``ss ij->j [d]`` at both
shapes and both densities, which the typed route REFUSES on both trees, so no
typed ratio exists for them; an earlier draft said "all 44", which counted four
configurations that were never timed.

### 57.8 Step 2a: the compressed-prefix rule reaches 58 cells, and admits none

A unique sentinel code (``SENTINEL_compressed_prefix_domain``) was installed in a
throwaway worktree — necessary because the rule shares
``unsupported_sparse_output_domain`` with nine other ``_fail`` sites in the same
file — and the 1139-cell frontier re-run in both arms.

- **58 cells, arm-invariant**, and the sentinel isolates that rule and nothing
  else: zero cells moved for any other reason.  The corpus names six.  That is a
  ~10× understatement, larger than blocker 2's 2.5× (§55.2).
- Families: ttm 32, rank4 10, degenerate2 5, rank6 4, rank3 3, rank2 1, rank5 1,
  union2 1, nonadd-combiner 1 — including a rank-6 cell, a union key and a
  non-additive combiner, none of which the declared envelope names.
- **Relaxing the rule admits ZERO cells.**  56 of the 58 land at the ordered-key
  target's own require; 2 still carry the sentinel at a second prefix level.  The
  CIN rule is entirely shadowed by a second check, so the work is in
  ``lower_llir.py``, not CIN.  Any cost estimate that stops at the CIN rule is
  wrong by construction.

This is §54.2's lesson in the opposite direction: there a CIN probe under-counted
because a second check hid cells; here it shows the rule buys nothing alone.

### 57.9 The dense-domain assembly semantics, stated before building

Recorded in full at ``compressed-prefix-reach/DENSE_DOMAIN_ASSEMBLY_SEMANTICS.md``.
For a compressed result level driven by a dense domain, with ``S(L)`` the levels
strictly below it:

1. **Some level in ``S(L)`` is compressed** — the runtime result-side counter
   already emitted by ``_parent_append_statements``:
   ``if (C{L+1}_pos.back() < pC{L+1}) { C{L}_crd.push_back(coord); ++pC{L}; }``.
2. **All of ``S(L)`` dense, extents statically nonzero** — "non-empty below" is
   statically TRUE (``levels.py:1110-1128``: dense storage materializes every cell
   including ones the stream never touched), so the guard is a tautology and the
   correct emission is an **unconditional append**.  The consequence, stated
   rather than discovered: ``ds ij->i [s]`` appends one entry per row of the dense
   domain and its compressed level becomes fully dense.  §54.9 flagged exactly
   this; it is forced by the rule that structure, not values, decides what is
   stored, and it is the property ``:2181`` already locks for ``dd ij->j [s]``.
3. **All of ``S(L)`` dense, some extent may be zero** — a **compile-time** extent
   predicate, with existing precedent in the B2 family's ``_suffix_guard``
   (``C1_size > 0``, ``lower_llir.py:10544``), never a runtime counter.

Never a value test, in any case.  No operand-side ``A_pos[i+1] > A_pos[i]``
predicate is introduced: no codegen site emits one today, and the result-side
counter answers the same question with machinery already in the tree.

### 57.9a The suite, its control, and eleven locks that encoded the old decision

**Three suite attempts were void before a real one existed, across two method
errors worth recording.**  An earlier draft of this paragraph said "two", having
counted the quarantine's diagnosis file rather than its logs; the quarantine holds
two distinct multi-thousand-line runs, and the third attempt left no artifact at
all.

1. Run as ONE monolithic pytest process over ~6,300 nodes, the suite reported
   **565 failures** — every one ``Fatal Python error: Aborted`` inside
   ``subprocess.py`` ``_execute_child``, i.e. the macOS fork-after-threads abort
   in the PARENT, triggered once the JIT starts forking a build per kernel in a
   process that has accumulated torch/OpenMP threads.  Each failing test PASSES
   run alone.  §55's check 6 ran the suite in **8 file-disjoint partitions**;
   that is the branch's protocol and ignoring it produced 565 phantom failures.
2. The first partitioned attempt used ``mapfile``, which macOS bash 3.2 lacks, so
   it enumerated zero files and reported ``tests 0  failures 0`` — a green-looking
   total over an empty set.  A harness that can report success without running
   anything is worse than one that crashes.

There was also a THIRD void run, and it is the one whose diagnosis is actually on
disk.  ``superseded/cand.log`` records **566 failed / 5728 passed** with the same
``_execute_child`` abort signature, and ``superseded/WHY_SUPERSEDED.md`` attributes
it to a different cause than run 1: it was launched CONCURRENTLY with the
blocker-1 heavy sweep, which spawns a subprocess per configuration that itself
spawns ninja and clang, and together they exhausted the per-uid process budget
(``kern.maxprocperuid`` = 10666).  Same symptom, different mechanism — fork
contention from a neighbour rather than fork-after-threads inside one process —
and the general rule the file draws is the one this branch now runs on.

So the quarantine holds ``cand.log`` (566) and ``cand_monolithic.log`` (565) with
one diagnosis file covering the former; the ``mapfile`` attempt produced no
artifact to quarantine, because its failure was to enumerate nothing.  Saying
"both are quarantined with their diagnoses" was true of neither pair.

**The controlled result, partitioned, one tree at a time:**

| | tests | failures | passed | skipped |
| --- | --- | --- | --- | --- |
| base ``a8f2954`` | 6309 | **0** | 6294 | 15 |
| candidate, before the test updates | 6309 | **11** | 6283 | 15 |
| candidate, after the test updates | 6319 | **0** | 6304 | 15 |

The node count moved, so it was diffed as a SET rather than trusted as a total —
a bigger number with zero failures can hide a silently deleted test.  Measured
over the JUnit XMLs: **8 removed, 18 added.**  The 8 are exactly the locks that
encoded the reversed decision (``test_auto_tile_blocked_cells_keep_their_schedule_code``
×4, ``test_the_auto_tile_neighbours_stay_blocked_on_blocker_one`` ×4) and nothing
else.  The 18 are 8 from the row-scope lock's now-honoured ``b_fmt`` × both arms,
6 from the two ``TTM dds`` cells joining ``MIGRATED`` (arm-source-identity and
arm-invariant compilation), 2 for the successor lock asserting the boundary is
empty and 2 for the new dense-receiver neutrality lock.

Base reproduces §55's recorded 6,309 / 6,294 / 15 exactly, which is what makes
the candidate's 11 attributable.  **Running the base suite as a control was owed
from the first attempt and was not done then**; no candidate suite number means
anything without it.

**None of the eleven demonstrates working behaviour that the change broke.**

- **Eight are the blocker-1 locks** —
  ``test_auto_tile_blocked_cells_keep_their_schedule_code`` (×4) and
  ``test_the_auto_tile_neighbours_stay_blocked_on_blocker_one`` (×4).  They assert
  the cells STAY at ``unsupported_schedule_auto_family``, which is §52.7's
  decision.  The first's own docstring anticipated the move: "If a
  workspace+tile composition is ever implemented, this lock moves to whatever
  still occupies the seam rather than being deleted."
- **Three are plan-shape locks in ``test_loop_plan.py``**, and both were measured
  rather than assumed.  ``test_auto_origin_derives_workspace_storage_from_the_workspace_axis``
  asserts a tile on ``j`` for an **``sd``** receiver; executing that program on the
  unmodified tree, ``einsum('kij->kj', A_ddd, format='sd')`` returns **wrong
  values at every shape tried** — (4,5,6), (8,3,7), (2,9,4), max abs error 3.33 /
  2.10 / 5.76 against a dense reference.  The test only ever inspected the PLAN,
  so it locked a tile whose emitted kernel silently computed the wrong answer.
  ``test_auto_origin_candidate_order_ignores_physical_mode_permutation`` (×2)
  asserts three tiles for a **``dsd``** receiver, a program that fails to build on
  BOTH trees.

**What the updates do.**  The two ``TTM dds`` cells MOVE from
``AUTO_TILE_BLOCKED`` into ``MIGRATED`` (9→11, 2→0), so the census's 20-name
matrix total is unchanged.  The blocked-cell lock is replaced by a successor that
asserts nothing is left blocked for that matrix and names the two cells that still
are over the frontier — ``ddd ijk->k [d]`` and ``ddd ijk->ik [dd]``, both
DENSE receivers.  A new ``test_dense_receivers_keep_their_automatic_tile`` locks
the neutrality half directly: a dense receiver must keep exactly the tile it had.

**A latent test defect fixed in passing**: the row-scope lock parametrized
``b_fmt`` over four formats and then hardcoded ``"dd"`` in the call, so all four
cases exercised the same program.  It now uses ``b_fmt``, over both arms.

**One coverage LOSS, recorded rather than dropped.**  The permutation test's real
purpose is that candidate enumeration follows LOGICAL access order rather than a
permuted physical ``mode_order`` — it guards a regression in which the derived
twin walked ``storage_index_ids``.  That property is no longer observable through
that program, because the tiles whose order it compared are no longer derived for
a non-dense receiver.  Reconstructing it needs a DENSE receiver that still tiles
under a permuted operand ``mode_order``; the obvious candidate
(``ddd ijk->ik [dd]`` with a permuted operand) is refused earlier by
``result_storage_order``.  It is written into the test as an open item.

### 57.10 What this section does not do

- **Nothing is pushed, and nothing wires the typed route into dispatch.**
  ``compile_cin_via_loopir`` and ``execute_cin_via_loopir`` still have zero
  non-test callers, and the suite proving that importing scorch never loads the
  LoopIR package still passes.
- **The dense-domain assembly boundary is NOT moved** — only its reach measured
  and its semantics fixed in writing.
- **The kernel-runtime grid has not been run on a SECOND HOST.**  The quiet-machine
  re-run §57.7 reports is done and recorded with its checksums —
  ``kernelperf-step0/receipts/m5_quick_base.json``,
  120 controls, floor 0.977–1.044 — so what remains outstanding is cross-host
  reproduction on redwood and mkt1, whose transports §56 staged.  (This bullet was
  drafted before §57.7 was amended with the quiet re-run and said the quiet run was
  owed as well; it was not.)
- **The blocker-1 heavy sweep is single-host, not owed.**  Its FIRST run was
  quarantined as tree-contaminated — launched against the live tree, which was then
  edited mid-run, 480 of 780 records, retained with its diagnosis under
  ``blocker1-legacy-soundness/receipts/superseded/``.  It was then re-run in full
  against the PINNED base worktree at ``a8f2954``, and §57.4's verdicts stand on
  that clean **780-measurement** grid (``receipts/base_heavy.json``, checksummed),
  not on the single-shape lean sweep — which §57.4 in fact declares FALSE, and
  whose replacement is where the ``i = 1`` correction comes from.  What is owed is
  a second host.  (Drafted before §57.4 was amended; it said the pinned-base grid
  was "queued".)
- **No blocker other than 1 is touched**, and the Phase-8 cutover verdict is
  unchanged.

**Evidence ledgers** (each with ``SHA256SUMS`` and harnesses taking a ledger root
as ``$1``): ``~/.cache/scorch-codex/kernelperf-step0/`` (runtime harness and
grid), ``blocker1-legacy-soundness/`` (the fourteen cells' legacy verdicts),
``blocker1-tilefix/`` (pinned base/candidate worktrees, frontier, differential,
production emission, affected-cell correctness), and
``compressed-prefix-reach/`` (sentinel probe, relaxed probe, reach and semantics
documents).

## 58. The TTM regression's mechanism, and the dense-domain semantics re-derived operand-side (2026-08-12)

Starts from committed tip ``ed4ce50`` (§57's work, committed in four pieces plus
the correction commit).  Two things land: the §57.7 TTM regression is explained,
and §57.9's dense-domain semantics are replaced with a derivation taken from the
operand side.  No production file changes in this section — ``git diff
ed4ce50..HEAD -- src/`` is empty — and nothing is wired into dispatch.

### 58.1 The regression is one kernel, not two

The emitted C++ does not depend on density.  Verified by emitting at both
densities and comparing SHA-256 across all four TTM cell-shapes and all three
columns, and visible in §57.7's own sealed receipt, whose ``sources`` phase
records the same ``typed`` digest at d=0.001 and d=0.05.  So the question is not
why the typed route emits worse code at high density; it is why one fixed pair of
kernels inverts its ranking as occupancy rises.

### 58.2 The two kernels differ in assembly strategy, and in nothing else

Read side by side, ``typed`` is ONE pass, SERIAL, appending into ``std::vector``;
``legacy_empty`` is TWO passes (count, then fill), **parallel on both** under
``#pragma omp parallel for``, writing into exactly-sized ``torch::empty``.

What is IDENTICAL in both, character for character: the k-merge, the
``A_val[pA2] * B_val[pB1]`` product, the workspace object
``coo_workspace_1d<float, 1>(1024)``, and the ``wksp.sort()`` drain.  That rules
out three of the four suspects **by construction rather than by measurement** —
the sparse-workspace drain's cost, the hash/COO key-domain behaviour, and the
per-``(i,j)`` workspace allocation are paid in equal measure by both routes at
every density.  The tile is ruled out separately: §57.7 measures the tile fix at
0.970–1.036 and these cells have ``dss`` receivers the guard never reaches.

### 58.3 It is a one-bit migration gap, not a heuristic

The two-phase parallel assembly is a SHARED LLIR pass,
``compressed_where_openmp_pass.py``, and both pipelines can run it.  Which one
does is decided by one virtual: ``_TargetLowering.owns_two_phase_output()``
returns False (``lower_llir.py:5484``) and exactly one LoopIR family overrides it
— ``_ParallelSparseWorkspaceLowering`` (``:9511``), and only for
``compressed_levels=(1,)``.  ``lower_llir.py:14504`` couples the two so hard that
a mismatch fails closed.  The regressing cells are hosted by
``_OrderedKeySparseWorkspaceLowering``, which never overrides it, and their
``dss`` receiver has TWO compressed levels.  Legacy's ``cin_lowerer`` runs the
same shared pass on the same programs and gets ``_count1`` AND ``_count2``, so
the pass already supports two compressed levels; only the LoopIR-side opt-in is
missing.

### 58.4 Demonstrated over the corpus, not over the two measured cells

A fix aimed at a mechanism inferred from two cells will appear to work on those
two cells.  So the prediction was stated and then tested compile-only over all 11
cells x 2 shapes: *the typed route loses exactly where legacy's kernel is parallel
and the typed one is not, and nowhere else.*

| group | cell-shapes | §57.7 ratios |
| --- | --- | --- |
| legacy PARALLEL / typed SERIAL | **4** — exactly the four TTM cell-shapes | 0.359, 0.398, 0.804, 1.004 |
| both SERIAL | **18** — every other cell-shape | 0.984–1.241, no loss anywhere |

Zero exceptions in either direction.

### 58.5 CORRECTION to §57.7: the variable is not density

Legacy's pragma requests ``num_threads(scorch_nthreads(A1_pos[A0_size], A0_size))``
and ``scorch_policy.h:123`` defines that as
``min(rows/SCORCH_ROWS_PER_THREAD, work/SCORCH_GRAIN_DEFAULT)`` clamped to
``[1, omp_get_num_procs()]``, with the constants 16 and 500.  **Legacy's own
parallelism is conditional, and the condition is met as density rises.**
Evaluated on the real operands, that thread count orders all eight measurements
monotonically and without exception:

| ``scorch_nthreads`` | §57.7 ratios |
| --- | --- |
| **1** | 1.044, 1.164, 1.487, 1.749 — typed wins every one |
| **2** | 1.004, 0.804 |
| **4** | 0.398, 0.359 |

Density is a proxy for one term.  The other is the OUTER EXTENT, which is why the
regression is shape-dependent in a way density cannot explain — at ``i = 32``
legacy is capped at 2 threads and loses 1.24x, at ``i = 64`` it gets 4 and wins
2.8x — and why the occupancy hypothesis fails outright: ``TTM dss x dd -> dss`` at
(32,64,128,64) d=0.05 has a **100% dense** result (131,072 of 131,072 cells) and
measures NEUTRAL at 1.004, while the 96%-occupied (64,128,64,128) measures 0.398.

### 58.6 The ablation, and a hypothesis of this session's own that it refuted

A third column was built from legacy's OWN emitted source with its two
``#pragma omp parallel for`` lines deleted and nothing else touched — an exact A/B
on one variable, chosen over an env knob because ``scorch_nthreads`` reads
``omp_get_num_procs()``, which ``OMP_NUM_THREADS`` does not change.  All three
columns agreed bit-for-bit on output storage before anything was timed.  A/A floor
0.978–1.013; pinned base worktree; quiet machine.

| cell | shape | d | ``nthr`` | ``legacy/typed`` | ``legacy_serial/typed`` | legacy's parallel speedup |
| --- | --- | --- | --- | --- | --- | --- |
| ``dss x ss`` | (64,128,64,128) | 0.05 | 4 | 0.360 | **1.228** | **3.41x** |
| ``dss x dd`` | (64,128,64,128) | 0.05 | 4 | 0.389 | **1.183** | **3.04x** |
| ``dss x ss`` | (32,64,128,64) | 0.05 | 2 | 0.790 | **1.465** | 1.85x |
| ``dss x dd`` | (32,64,128,64) | 0.05 | 2 | 0.941 | **1.395** | 1.48x |
| ``dss x ss`` | (64,128,64,128) | 0.001 | 1 | 1.437 | 1.321 | 0.92x |
| ``dss x dd`` | (64,128,64,128) | 0.001 | 1 | 0.988 | 0.887 | 0.90x |
| ``dss x ss`` | (32,64,128,64) | 0.001 | 1 | 1.788 | 1.719 | 0.96x |
| ``dss x dd`` | (32,64,128,64) | 0.001 | 1 | 1.070 | 0.985 | 0.92x |

**The regression is 100% parallelism and 0% allocation.**  Delete two pragmas and
every d=0.05 regression inverts into a typed win of 1.18–1.47x.

**This refutes a hypothesis this session had written down before measuring it.**
The mechanism note predicted a residual attributable to ``std::vector`` growth
versus exact allocation, reasoning that a 2.8x loss exceeds the ~2x an ideal "two
passes on four threads" allows.  The residual is real and its sign is the
OPPOSITE: the typed single-pass is *faster* than legacy's serial two-pass, so
legacy must overcome a 1.2–1.5x strategy deficit *and then* win by 2.8x — which is
why its parallel speedup has to be, and is, above 3x.  Recorded because the
arithmetic that looked anomalous was pointing at the right anomaly and the wrong
cause.

At ``nthreads = 1`` the pragma is not merely inert, it **costs 4–10%**: the OpenMP
region is entered and torn down for one thread.  That, plus the redundant counting
pass, is the whole of the typed route's low-density win.

### 58.7 What the measurement says the fix must be

Opting the family into the existing shared pass — the obvious move, and the one
§58.3 makes look easy — is the WRONG target, and the ablation is what shows it.
It would surrender the 1.18–1.47x single-pass advantage and merely TIE legacy at
high density, and adopted unconditionally it would turn the 1.44–1.79x low-density
wins into ties, since ``nthreads = 1`` is exactly where the counting pass buys
nothing.  Bobby's standard for this branch is "more performant than legacy on a
wide variety of problem sizes", not "not worse".

The measurement points at a third option better than either existing kernel:
**keep the typed single-pass strategy and parallelize it**, with per-thread output
buffers concatenated in outer-loop order.  The outer loop is over a dense ``i``
and the receiver's compressed level 1 sits under ``i``, so per-thread buffers
concatenated in ``i`` order reproduce the required lexicographic assembly exactly.
The cost is one ``O(nnz_out)`` memcpy per array in place of legacy's full
recompute — strictly cheaper than the counting pass it replaces, on the evidence
above — under the same ``scorch_nthreads > 1`` condition, so the serial path is
provably untouched at ``nthreads = 1`` and the low-density wins are preserved by
construction.

This option follows from the standard Bobby set for this branch, not from a
separate instruction of his.  That standard is **"more performant than legacy
on a wide variety of problem sizes"**, not "not worse" — the kernel-runtime
standard §54.8 records for Phase 7 and ``CLAUDE.md`` states for migrated
families — together with ``CLAUDE.md``'s rule that a runtime condition must be
*provably* never true on the shapes it must not act on.  Applied to §58.6's three
columns the standard eliminates the other two by measurement: adopting the shared
pass unconditionally surrenders the 1.18–1.47x single-pass advantage AND destroys
the low-density wins, and adopting it under a condition preserves the wins but
reaches only parity where the regression is.  Only the third can be better than
both existing kernels, so it is the one the standard leaves.  **It is not built in
this section.**

### 58.8 The dense-domain semantics, re-derived operand-side

§57.9 keyed its three cases on ``S(L)``, the levels strictly below the RESULT
level.  The question Bobby's decision asks — did the operand have anything here —
is about the OPERAND's structure under the coordinate.  Rewritten at
``compressed-prefix-reach/DENSE_DOMAIN_ASSEMBLY_SEMANTICS.md``, with §57.9's
version kept beside it and marked superseded with the reason.  Three measured
errors:

1. **Wrong answer on the motivating cell.**  ``ds ij->i [s]`` is rank-1 so ``S(L)``
   is empty, so case 2 appends unconditionally and the result densifies.  The
   operand's compressed level 1 makes the structural answer available as
   ``A1_pos[i+1] > A1_pos[i]``.  The densification §57.9 called forced is not.
2. **Its precedent is a different family.**  Measured off ``_ordered_key_split``:
   ``:2181``'s cell ``dd ij->j [s]`` is ``(prefix, key_rank) = (0, 1)``, a drained
   KEY under a workspace, while ``ds ij->i [s]`` is ``(1, 0)``, a bound PREFIX with
   no workspace at all.  The drained key's unconditional append follows from the
   workspace holding an entry per inserted key; the bound-prefix family has no
   workspace to hold one.
3. **Its cost claim does not hold where it was applied.**  It declines an
   operand-side probe because the result-side counter "answers the same question
   with machinery already in the tree".  That counter emits
   ``if (C{L+1}_pos.back() < pC{L+1})``, and for a rank-1 result there is no
   ``C1_pos`` and no ``pC1``.  The machinery is structurally unavailable in exactly
   the case case 2 was covering.

### 58.9 Derivability over all 58 cells, and two the rule does not cover

Method: ``lower_cin._fail`` instrumented to snapshot the raising frame's locals —
``domains``, ``lhs_index_ids``, ``result_levels``, ``prefix``, ``key_rank``,
``position`` — for all 58 cells driven through the real lowering.  Not modelled
from cell names.

**The refusal covers 56 cells, not 58.**  The refused level's actual domain kind
is DENSE on 56, **UNION** on ``ss+ss ij->i [s]`` and **INTERSECTION** on
``ss*ss ij->i [s]``.  The rule's condition is ``domain_kind is not
DomainKind.SPARSE``, which lumps three kinds into one refusal.  This explains
rather than merely records §57.8's leftover: relaxing to admit DENSE left exactly
those two carrying the sentinel because relaxing DENSE admits neither UNION nor
INTERSECTION.  They are a separate refusal over merged domains and a separate
decision, so this document does not owe them a rule — and extending to them by
analogy would have been wrong specifically in the intersection case, whose
non-emptiness is not a function of the streams' extents and cannot be answered by
a position bound at all.

**45 of the 56 need no new machinery.**  They have a compressed result level below
the refused one, so the existing result-side counter answers exactly (ttm 32,
rank4 8, rank6 4, rank5 1).  **11 need the operand-side predicate**, and they are
exactly the rank-1 ``[s]``-on-``i`` family with ``(prefix, key_rank) = (1, 0)``:
``ds ij->i [s]`` and its five degenerate variants, ``dss``/``dds``/``dsd
ijk->i [s]``, and ``dsss``/``ddss ijkl->i [s]``.  **All eleven are derivable**;
nine are a single position-bound comparison, ``dsd`` adds a compile-time extent
conjunct, and ``dds``/``ddss`` index the array at a flattened position because an
intervening dense level makes the child's range contiguous.  No cell in the 58 is
undecidable.  §57.9's case 2 — the unconditional append — applies to **none** of
the 58 cells it was written for.

### 58.10 What the predicate costs, measured against an admitted sibling

``ss ij->i [s]`` is ADMITTED today and is the exact structural sibling of
``ds ij->i [s]``: same family, same ``(1, 0)`` split, same result shape, differing
only in operand level 0.  Its emitted append is unconditional **because its ``i``
loop is a stored stream** — the iteration domain is already carrying the guard.
Make level 0 dense and the domain stops carrying it.  The delta is: give the
loop-init expression ``A1_pos[pA0]`` a name, and wrap the three existing append
statements in one ``if (pA1_end > pA1_begin)``.

Zero new runtime array reads (both operands of the comparison are already read to
open the loop), zero new pointer declarations, no new LLIR node kinds, ~25–40
lines in the bound-prefix target.  It does not disturb the result-side counter
path: case R is selected exactly when ``C{L+1}_pos`` exists and case O exactly when
it does not, so the 45 case-R cells emit byte-identically.  The predicate is
*cheaper* than the mechanism §57.9 preferred, which loads ``C{L+1}_pos.back()``
from a growing vector where this compares two locals.

**Disposition of the two cells the decision was taken for.**  ``ds ij->i [s]`` and
``dss ijk->i [s]`` both take guard ``A1_pos[i+1] > A1_pos[i]``: a row of the dense
``i`` domain with no stored ``j`` gets **no key**, and the compressed result level
stays sparse rather than becoming fully dense.  §54.9's objection is answered
rather than accepted — it is true of unconditional append and false of the
operand-side guard.  ``:2181`` holds under the new rule in both readings, and
unlike §57.9 the rule does not need ``:2181`` to license anything.

### 58.11 What this section does not do

- **No production file changes.**  ``git diff ed4ce50..HEAD -- src/`` is empty;
  ``compile_cin_via_loopir`` and ``execute_cin_via_loopir`` keep zero non-test
  callers.
- **The TTM fix is specified, not built**, and the cross-host re-proof it will need
  is not run.  The kernel-runtime grid is still single-host.
- **The dense-domain refusal is specified, not built** — this is a semantics
  document and a proof that every case is derivable, not an implementation plan.
- **The refusal over merged domains (UNION, INTERSECTION) is named and left to
  Bobby.**
- No blocker other than 1 is touched; the Phase-8 cutover verdict is unchanged; the
  shadow pilot's membership is still un-re-derived.

**Evidence ledgers**, each with ``SHA256SUMS`` and harnesses taking a tree root as
``$1``: ``ttm-density-mechanism/`` (source dump, pragma census, thread-count
prediction, three-column ablation, ``MECHANISM.md``, ``ABLATION.md``) and
``dense-domain-semantics/`` (the state at the refusal over all 58 cells), plus the
amended ``compressed-prefix-reach/``.

## 59. The TTM parallel single-pass assembly, built — and three structural claims the measurements overturned (2026-08-12)

Starts from committed tip ``4a21c57``.  §58.7's fix is BUILT: three production
commits, ``git diff 4a21c57..HEAD -- src/`` is 967 insertions across six files.
Nothing is wired into dispatch; ``compile_cin_via_loopir`` and
``execute_cin_via_loopir`` keep zero non-test callers.

**The verdict, up front: NOT SHIPPABLE as it stands.**  The §57.7 regression is
inverted on three of its four cell-shapes ON THE M5, and the cross-host run
(§59.9) does not reproduce it: on redwood the fix beats legacy on 8 of 16
parallel configurations instead of 13, and **two configurations are slower than
the BASE SERIAL kernel**, which is a regression this change introduces and is
disqualifying on its own under the standard this branch works to.  The
single-host claim was the thing most likely to be wrong and it was; §56 removed a
single-host caveat about ROUTING, and this is the first time the branch has
measured KERNEL RUNTIME on a second host.

The section is longer than the result needs because **four structural arguments
this work made were refuted by its own measurements** -- three of them its own
design's, one of them ``ABLATION.md``'s -- and each refutation matters for
whatever is built next.

### 59.1 What was built

The design is recorded with its checksum at
``ttm-density-mechanism/FIX_DESIGN.md`` — the file ``ABLATION.md`` referenced and
that did not exist — written before any production code and carrying four
predictions measurement could contradict.

Per-CHUNK, not per-thread, output buffers.  Each chunk of the outer dense loop
appends into its own buffers exactly as the serial builder does; the buffers are
concatenated in chunk order, which is outer-loop order, which is the required
lexicographic order because the outer loop binds result level 0 and every
compressed level sits under it.  Chunks rather than threads own the buffers so
that a dynamic schedule cannot reorder the output: one thread may take a low row
range and then a high one, so per-thread buffers would not reproduce the order
while per-chunk buffers do, whatever the schedule did.  Load balancing is
therefore kept rather than traded away.

The first compressed level's ``_pos`` array is shared and pre-sized rather than
chunked, because it is indexed by the outer loop over a statically known range;
every chunk writes a disjoint slice and only its VALUES need the merge's offset.
``owns_two_phase_output()`` stays False and the hard coupling at
``lower_llir.py:14504`` is untouched — this is a THIRD assembly strategy, not
two-phase output.  What is admitted is decided structurally (§59.6).

### 59.2 REFUTED: "byte-identical serial arm" does not mean inert

``FIX_DESIGN.md`` argued that the runtime condition would provably cost nothing
at ``scorch_nthreads == 1``, because the serial arm would be today's statement
list unmodified, with no pragma on it.  That argument is wrong.

Measured: the first build was **up to 34% SLOWER than the base** at
``nthreads == 1``, on shapes where the condition is never true and the arm that
runs is the base's nest character for character.  The penalty grew with the work
the kernel did — ruling out a fixed prologue cost — and at one configuration it
turned a 0.971 tie with legacy into a 0.926 loss.

A three-column source-level ablation isolated it, the same method ``ABLATION.md``
used on legacy's pragmas: a third column built from the candidate's own emitted
source with the PARALLEL arm's body replaced by an unreachable ``throw``,
condition still present, second copy of the nest gone.  **``stubbed`` tracks
``base`` at all twelve one-thread configurations**, to within 0.017 and usually
0.005, including the two worst points where both sit at 0.75 and 0.81 against the
candidate.  So the duplicated body is essentially the whole penalty and the
condition itself is essentially free.

Two consequences.  A ``scorch_assembly_threads`` helper, written to keep
``omp_get_num_procs()`` off the serial path, was **deleted**: the same ablation
shows the condition including that call is free, so it was mechanism introduced
against a hypothesis the measurement had just refuted.  And the general lesson,
recorded so it is not relearned: byte-identical *emission* is a real check;
byte-identical *fragments inside a changed function* carry none of the same
guarantee.  A path that must stay neutral needs a runtime measurement, not an
argument from the source diff.

### 59.3 REFUTED: sharing the body between both arms is worse, and it is not aliasing

The obvious repair — emit the nest once as a lambda both arms call — was built,
and measured **worse**: up to 55% slower than the base at one thread, against
duplication's 34%.

``__restrict__`` on the shared body's reference parameters recovers **none** of
it (``restrict/base`` 1.60 at the worst point), which rules out aliasing between
the buffers and leaves the real cause: passing them by reference makes them
ESCAPE, so an append-heavy drain can no longer keep a vector's internals in
registers.  ``__restrict__`` promises non-aliasing, not non-escape.

What ships: the serial arm keeps the ORIGINAL inline nest over the function's
own locals, and the shared body is used by the PARALLEL arm only — which pays
the escape either way, its buffers being elements of a chunk vector reached by
reference whatever we do.  ``LambdaDef`` earns its place on one arm and not the
other.

### 59.4 The residual removed at compile time, not paid

``scorch_nthreads`` is ``clamp(min(rows/ROWS_PER_THREAD, work/grain), 1, hw)``,
so an outer extent below ``2 * ROWS_PER_THREAD`` cannot reach two threads FOR
ANY OPERAND — the row term alone forces it.  That extent is a compile-time
constant: the kernel is built per shape and the ABI validates it against one.
Those programs are now declined **before anything is emitted** and emit
byte-identically to the base.

This is what makes the cost exactly zero rather than merely small, and §59.2 is
the proof that the distinction matters.  It is also the shape ``CLAUDE.md`` asks
a condition like this to have — provably never true where it must not act —
established by construction rather than by a threshold that happens to hold.  On
the corpus it removes the ten worst one-thread configurations outright:
``(16,256,256,32)`` has 16 rows and now emits the base's kernel character for
character at every density.

### 59.5 The M5 result, scored against the predictions written first

Candidate ``691b46b``, base ``4a21c57``, pinned detached worktrees; four columns
from source through one JIT loader in one process against one preamble;
ABBA-interleaved; one same-binary control per column per configuration.  **All 30
configurations agree on output storage — both index arrays and values — and
match the dense reference, before any ratio was computed.**  A/A floor
0.944–1.117.  Receipt ``ttm-parallel-singlepass/receipts/fix_ablate_m5_final.json``.

| cell | shape | d | ``nthr`` | §57.7 | **now** |
| --- | --- | --- | --- | --- | --- |
| ``dss x ss`` | (64,128,64,128) | 0.05 | 4 | 0.360 | **1.123** |
| ``dss x dd`` | (64,128,64,128) | 0.05 | 4 | 0.389 | **0.967** |
| ``dss x ss`` | (32,64,128,64) | 0.05 | 2 | 0.790 | **1.370** |
| ``dss x dd`` | (32,64,128,64) | 0.05 | 2 | 0.941 | **1.374** |

**13 of 16 parallel configurations beat legacy** (1.032–1.610), and the candidate
is faster than the base serial column at every one of them (1.23x–3.16x).  Both
of those statements are M5-only; §59.9 measures what happens to them on x86.

- **P1** (4 threads, [1.10, 1.35]): all four ``dss x ss`` in band (1.114–1.209);
  ``dss x dd`` gives 1.341 in band and **0.707 / 0.755 / 0.967 MISS**.
- **P2** (2 threads, [1.30, 1.45]): five of eight in or above band, three below
  but still beating legacy.  No configuration below 1.00.
- **P3** (speedup over base ≥ 2.6 at four threads): met on ``dss x ss``
  (2.85–3.16), **missed on every ``dss x dd``** (2.25–2.40).
- **P4** (one thread, inside the A/A floor): nine of fourteen clean; two real
  misses at (64,128,64,128) d=0.001 (+3.6%, +4.1%); three nominal misses of
  +0.4%, +0.6% and +2.5% are on shapes whose emission is byte-identical to the
  base, so those two columns are the same source and those rows are same-binary
  controls rather than measurements.

### 59.6 CORRECTION to ABLATION.md: the single-pass advantage is not uniform

``FIX_DESIGN.md`` chose to keep the single-pass strategy on ``ABLATION.md``'s
finding that it is "1.18–1.47x FASTER than legacy's serial two-pass".  That
finding was measured at two densities, 0.001 and 0.05.  Across five densities
the ``legacy_serial / typed_base`` column says it **inverts**:

| cell | shape | d | ``legacy_serial/base`` | ``legacy/typed`` |
| --- | --- | --- | --- | --- |
| ``dss x dd`` | (64,128,64,128) | 0.005 | **0.857** | 0.707 |
| ``dss x dd`` | (64,128,64,128) | 0.01 | **0.953** | 0.755 |
| ``dss x dd`` | (64,128,64,128) | 0.05 | 1.175 | 0.967 |
| ``dss x dd`` | (64,128,64,128) | 0.2 | 1.561 | 1.341 |

With a dense second operand at mid density, legacy's counting pass is *cheaper*
than the single pass, before any parallelism is involved.  The three losing
configurations are exactly the three where the typed route starts from that
strategy deficit — so they are not a defect of the parallelization.  Legacy also
scales better on that cell (2.84–2.91x against the candidate's 2.25–2.40x)
because its fill pass writes into exactly-sized buffers and has almost no serial
tail, while the candidate pays chunk-buffer growth plus a merge whose
destination sizing is serial.

Stated plainly: **on ``TTM dss x dd -> dss`` at (64,128,64,128) and d in
[0.005, 0.05] the typed route still loses to legacy, by 3% to 29%.**  It lost by
up to 70% before.  Nothing regressed against the base at any parallel
configuration.

### 59.7 What is admitted, and what emits byte-identically

The parallel arm is emitted only when the result's dense prefix is exactly one
with every remaining level compressed, the outermost prefix loop is a
``DenseFor`` binding result level 0, no stored-prefix assembly is owned, no
panel/relayout/result-tile is attached, the outer extent can reach two threads,
and a work estimate is derivable by the same ``sparse_pos_work_expr`` legacy's
pragma uses — so both kernels turn parallel on the SAME condition at the same
operands.

A dense prefix deeper than one is declined rather than guessed at: its first
compressed level's position array is indexed by a FLATTENED dense cell, so the
pre-size, the per-chunk starting index and the shift range become products of
the dense extents.  The mechanism generalizes; that derivation needs its own
statement and its own measurement.

Emission census over the corpus: **16 cell-shapes byte-identical**, 4 carrying
the new condition — exactly §58.4's four TTM cell-shapes — each with exactly one
``#pragma omp``, a serial arm that is the base's nest **verbatim** and carries no
pragma, and a shared parallel body that matches the base's nest; 2 typed-refused,
unchanged.

### 59.8 Frontier (host-independent)

The 1138-cell extended frontier — the survey matrix of programs this branch
measures the compiler over — in both automatic arms, on base and candidate:
**0 lost, 0 gained, 0 route changed, 0 unclassified, 0 NEW arm-variance.**  Three
cells are arm-variant on BOTH sides — inherited, identical sets — and the
comparison is against the base rather than treating inherited state as this
change's failure.

### 59.8a The suite, and the regression the node-set diff caught

Eight file-disjoint partitions in fresh processes, base and candidate, 16 runs.
**6319 node ids on each side, 0 lost, 0 added.**  The node-set diff — rather
than the totals, which matched — caught **one regression**:
``test_walker_and_rewriter_cover_every_declared_node`` went passed → failure,
because ``llir.LambdaDef`` had been added to
``SUPPORTED_LLIR_STATEMENT_NODE_TYPES`` without a sample and an expected
emission in the traversal suite's coverage set.  That test exists to fail
exactly then, and the totals alone would have shown 6319 both sides with a
single failure buried in one partition's exit code.  Fixed in the test commit;
the file now passes 435/435 and the new family's own file 34/34.

### 59.9 CROSS-HOST: redwood does not reproduce the M5, and that is the verdict

Transport per §56: source archives of the two pinned commits, extension built
**in place** in each tree so ``PYTHONPATH`` resolves both ``scorch`` and
``scorch_ops`` inside the measured tree — redwood's shared env carries
``scorch_ops`` from an unrelated ``perf/spmm-fastpath`` checkout, so the
confound is excluded by construction and then asserted by the harness.  Both
hosts independently computed a manifest digest over every ``src/**/*.py`` and
agree: ``0d88487e…`` base, ``73199fdb…`` candidate.  Nothing pushed; redwood's
checkout and env untouched.  Machine quiet (load average 0.44); A/A floor
0.981–1.012; all 30 configurations agree on storage and reference.  Receipt
``receipts/fix_ablate_redwood.json``.  redwood is 32 procs to the M5's 18.

|  | M5 | redwood |
| --- | --- | --- |
| parallel: beats legacy | **13/16** | **8/16** |
| parallel: ``legacy/typed`` | 0.707–1.610 | **0.343–1.376** |
| parallel: ``base/typed`` | 1.23–3.16 | **0.65–2.50** |
| serial arm: ``base/typed`` | 0.960–1.009 | **0.985–1.018** |
| serial arm: ``legacy/typed`` | 0.989–2.531 | 1.106–1.900 |

**What holds on both.**  The serial path.  redwood's ``base/typed`` at one
thread is 0.985–1.018 over all fourteen configurations — TIGHTER than the M5's —
so the compile-time decline and the inline serial arm are neutral on x86 and the
M5's two 3.6%/4.1% residuals do not reproduce (redwood's worst is 1.5%).  The
low-density wins over legacy survive intact, 1.106–1.900.  Correctness holds
everywhere.

**What does not hold.**  Parallel scaling is far weaker on x86: ``base/typed``
at four threads is 1.07–2.50 against 2.25–3.16.  Both hosts pick the SAME thread
count — ``scorch_nthreads`` is capped by ``rows/16`` at these shapes — so this is
not a policy difference.  And **two configurations regress against the BASE**:
``dss x dd`` (32,64,128,64) at d=0.005 and d=0.01 measure 0.647 and 0.766, so
the parallel path is 1.5x and 1.3x slower than the serial kernel it replaced.
The headline cell ``dss x ss`` (64,128,64,128) d=0.05 goes 1.123 → **0.954**.

**Diagnosis.**  The condition asks *is there enough work to be worth threads*.
The measurements say the question is *is there enough work to be worth threads
AND per-chunk buffers AND a concatenation*, and the second has a different answer
on a different host at a different shape.  At (32,64,128,64) the chunk width is 4
rows, so a small problem gets 8 buffer sets, and on x86 their allocation, growth
and concatenation cost more than two threads buy.  The condition lets through
cases the transformation cannot pay for.

That is fixable, and the fix is not a constant to tune.  It needs a second
condition with the same "provably cannot fire" character the extent test has,
derived from the merge's cost rather than from the thread count.  Fitting a
threshold to these sixteen points is the overfitting ``CLAUDE.md`` forbids, so
it is left as a stated defect.

### 59.10 What this section does not do

- **mkt1 is not run.**  Two hosts, not three.
- **The fix is not shippable and is not proposed for shipping.**  Two x86
  configurations regress against the base; that is disqualifying on its own,
  independently of legacy.
- **``TTM dss x dd -> dss`` at (64,128,64,128), d in [0.005, 0.05], loses to
  legacy on both hosts.**  §59.6 measures why on the M5 and it is a strategy
  question, not a parallelism one; no attempt was made to tune around it.
- The M5's two ``nthreads == 1`` residuals of 3.6% and 4.1% are bounded,
  explained and not removed; they do not reproduce on x86.
- The dense-domain seam, the merged-domain seam, the shadow pilot's membership
  and every blocker other than 1 are untouched; the Phase-8 cutover verdict is
  unchanged.

**Evidence ledger**: ``ttm-parallel-singlepass/`` with ``SHA256SUMS`` —
``RESULT.md``, ``DUPLICATION.md``, the four-column timing harness, the
duplication ablation, the emission census, the frontier pair and its diff, the
partitioned suite runner and its node-set differ, and the predictions scorer.
The design it is measured against is ``ttm-density-mechanism/FIX_DESIGN.md``.

## 60. Assembly strategy made a scheduling decision — and three of this design's own claims refuted (2026-08-14)

This section starts from committed tip ``cae4f11``.  Bobby's instruction: *"we
need to make sure our refactored compiler support compiling code for all valid
strategies. then whether to use single pass or two pass etc. whatever becomes a
scheduling decision that our autoscheduler should handle."*  He authorized the
canonical schema bump together with the identity and cache-key changes it
carries, and authorized changing legacy's emission "as long as we are always
moving in a better direction".

Two production commits, ``d9efcf8`` and ``b9f0e2a``; ``git diff cae4f11..HEAD --
src/`` is 1,074 insertions across twelve files, one of them new
(``sparse_assembly.py``, 166 lines).  **``src/scorch/csrc/`` is untouched**, so
the shipping ``scorch_ops`` extension is bit-unchanged by this section — the
csrc work §60.9 still lists as owed is the piece §59's header change left
behind, not a new one.  Nothing is pushed; ``compile_cin_via_loopir`` and
``execute_cin_via_loopir`` keep zero non-test callers.

**The verdict, up front.**  The representation, the split between legality and
cost, the explicit primitive and the structured refusals are BUILT, each behind
the checks that constrain it.  How much can actually be emitted is **partial and
measured**: two of the four strategies emit and are proven bit-exact on the
family the measurements are about, the third emits only on the family that
already owned it, and the fourth has no family that can host it at all.  Three
claims this section's own design made were refuted by its own measurements, and
one premise it inherited from §58.3 was refuted too.

### 60.1 The design, recorded before any code

``~/.cache/scorch-codex/assembly-strategy/DESIGN.md``, written at ``cae4f11``
before any production change, with its SHA-256 and the tip recorded in
``provenance/design_tip.txt`` so every prediction below can be checked to have
been made in advance.  It names the four strategies, separates them into the two
bits that distinguish them, gives a legality domain and a refusal code per pair,
specifies the representation, records how cheaply the output size can be known
for the next milestone's selector, and states nine measurements each with a
prediction that measurement could contradict.

Three corrections it had to make to the inherited record before relying on it:

- **``FIX_DESIGN.md`` exists.**  The task statement said ``ABLATION.md`` "still
  ends by pointing at a ``FIX_DESIGN.md`` that was never written".  It was
  written — 22,191 bytes, listed in ``ttm-density-mechanism/SHA256SUMS``, and
  recorded by §59.1.  The premise "this prevents a third missing design" rests on
  one instance, not two.
- **967 lines across six files is the diff from ``4a21c57``, not from
  ``a8f2954``.**  From ``a8f2954`` it is 1,034 across eight; the extra two are
  §57's blocker-1 tile fix, which is the direct precedent for this milestone's
  step ordering and must not be confused with §59's change.
- **There is no ``schedule_passes._verify_auto_family``.**  The admitted
  automatic replay contracts live in ``_check_auto_plan_family``
  (``schedule_passes.py:2729``) with the independent re-derivation in
  ``loop_plan_legality._derive_auto_decisions`` (``:376``) — the pair §57.5
  measured as duplicated.

### 60.2 The 2x2: right as an inventory, REFUTED as a product of two flags

The four are exactly four emission forms distinguished by two bits, and each has
a measured region where it wins, so listing them as a 2x2 is right.  Treating
them as a *product of two independent flags* is wrong three times over, and the
design says so before building:

1. **The parallelism bit is not a feature added to a serial form.**  ``P1``
   emits a runtime condition whose ``else`` arm is ``S1``'s nest verbatim, while
   ``P2``'s pragma is unconditional and at one thread still enters and tears down
   a region for **4–10%** (``ABLATION.md`` §3).  So "``P1`` with the condition
   false" is ``S1``, and "``P2`` at one thread" is **not** ``S2``.  A product
   would predict the second equality; measurement refutes it.
2. **The axes carry no independent legality.**  All three non-serial strategies
   share ONE receiver contract, and it is the same predicate two implementations
   had reached independently: the two-phase pass enforces
   ``compressed_levels == (1, …, rank-1)``
   (``compressed_where_openmp_pass.py:316``) and the chunk admission requires a
   dense prefix of one with everything below compressed
   (``lower_llir.py:8646``).  Identical sets.
3. **The product's fourth cell has no producer.**  Nothing selects ``S2``; it
   exists only as a column ``ABLATION.md`` built by deleting two lines from
   emitted text.

So the plan field is an enumeration of four tokens, not two booleans — a wider
domain would need a cross-field rule to exclude nothing, which is how a fifth
nonsense state gets added later.

### 60.3 Legality is not cost, and the extent test moved

§59.9 diagnosed the chunk condition as asking "is there enough work to be worth
threads" when the question is "…worth threads AND per-chunk buffers AND a
concatenation".  Under this architecture those are two predicates in two places,
and separating them is the substance of the change:

- **LEGALITY** — can this receiver be assembled this way at all?  Structural,
  provable, **extent-free**, refused with a code.  ``PARTITIONABLE`` plus, for
  the chunked strategy, four program conditions that are genuinely about whether
  the form can be expressed (a ``DenseFor`` outer binding result level 0, no
  stored-prefix assembly, no panel/relayout/result-tile, a derivable work
  estimate).
- **COST** — should we?  A named ``default_assembly()`` per family, which is
  today's choice and the single place a selector would replace.

**The reclassification this forces, and it is the one that matters.**  §59.4's
``2 * ROWS_PER_THREAD`` compile-time decline is COST, not legality: a 16-row
receiver assembles correctly from one chunk.  It therefore moved out of the
admission predicate into ``default_assembly``, where it still keeps every
automatically scheduled kernel byte-identical — and an *explicit* request at that
extent is now honoured, because it is legal.  A test locks both sides, since a
test that only checked the decline would have been satisfied by leaving the rule
where it was.

No constant is tuned in this section.  §59.7's condition keeps its legality half;
its cost half is now a named function the selector inherits.

### 60.4 Representation: two fields, two schemas, and the identity they carry

``LoopProgram.parallel`` is the precedent and the design follows it exactly — "part
of program semantics (canonically serialized, verified, erased with the schedule),
never a target annotation" (``nodes.py:1165``).  So:

| field | type |
| --- | --- |
| ``LoopPlan.assembly`` | ``Optional[str]``, one of four tokens |
| ``LoopProgram.assembly`` | ``Optional[AssemblyStrategy]`` |
| ``Schedule.assembly`` | ``Optional[str]`` — the public request |

Bobby authorized the bump as "v11 → v12".  That is right for the LoopIR schema
and it is not the whole of it: **two** schemas move, and this section records
both rather than performing only the one he named.

| schema | before | after |
| --- | --- | --- |
| ``printer.CANONICAL_SCHEMA`` | ``…canonical.v11`` | **``…canonical.v12``** |
| ``plan_identity.CANONICAL_PLAN_SCHEMA`` | ``…loopplan.canonical.v1`` | **``…v2``** |
| ``CANONICAL_REQUEST_SCHEMA`` | ``…request.v2`` | unchanged; its *bytes* move |

What that does to identity, stated here rather than left for a check to
discover: ``canonical_plan_dump``, ``plan_schedule_digest`` and
``loopir_request_identity`` change for every plan, because ``"assembly":null``
joins the payload — deliberate, so "no decision" stays distinguishable from
"serial by decision".  ``Schedule.cache_key`` changes for every schedule and
**does** reach production, costing one cold cache and no change in behaviour,
because a key is only ever compared with keys from the same build and a v11 dump
differs from a v12 dump in its ``"schema"`` field before it differs anywhere
else.

The tokens and ``PARTITIONABLE`` have exactly ONE definition, in the new
``sparse_assembly`` module, which the four layers that need the rule import.
That is the direct answer to §57.5: two copies of one scheduling rule drifting
apart cost twelve cells there.

### 60.5 REFUTED, by measurement: an automatic plan carrying no strategy makes three of four strategies unreachable

``DESIGN.md`` §4.5 argued that automatic plans should carry **no** token, on the
ground that recording one would require the origin to re-derive receiver legality
— a third copy of a rule whose second copy had already cost twelve cells.  The
reasoning is sound and the conclusion is wrong, and the measurement is
unambiguous: **all 24 legal cells failed** ``unsupported_program_shape`` /
``sparse_parent_dominance``.

The mechanism.  The strategies apply only to sparse-output programs; every such
program needs an accumulation workspace; and ``WorkspaceInsertion`` is a decision
only the automatic path can record, since the explicit path has no way to express
it (``loop_plan.py``: "explicit schedules express workspace lifetime through tile
accumulation instead").  Routing a strategy request down the explicit path
therefore produced a CIN carrying a workspace and a plan without the fact.

What ships instead: a requested strategy rides on the **automatic** plan as a
decision the replay contract verifies — exactly what the standing
``_check_auto_plan_family`` already does for the cost model's loop order, which
is "verified complete and legal, not re-derived".  The automatic **origin** still
chooses nothing, so an ordinary automatic compilation records ``None``; a
recorded strategy always came from a caller, which is what makes it a contract
rather than an unverified degree of freedom.  Byte-neutrality is untouched by the
correction.

The same measurement exposed a second way the code could pass a request through
without acting on it: ``Schedule()`` is the automatic marker, so a schedule
carrying **only** ``assembly`` looks empty to every field-by-field test and its
request was silently dropped.  A caller would have received the default kernel
with no way to find out.  Locked by a test.

### 60.6 REFUTED, inherited: generalizing the two-phase opt-in is NOT wiring

§58.3, ``MECHANISM.md`` §2 and this milestone's own task statement all assert that
the two-phase parallel assembly is a shared pass with one missing opt-in bit, and
that "generalizing this is wiring, not new capability" because the pass "already
supports two compressed levels because legacy drives it that way, with ``_count1``
AND ``_count2``".  That is true of the **level arity** and false of the
**statement vocabulary**, and it was refuted in three measured stages.

**Stage 1 — the completion checkpoint, not the pass.**  Forcing the opt-in, all
six probed cells failed ``sparse_workspace_completion_lost``.  The message is
``_require_ordered_key_completion_checkpoint``'s, **not** the ``:14595``
coupling's — so the coupling passed, meaning the pass *did* take ownership.  The
checkpoint mirrors the body before any pass runs and models exactly one
transformation, so it refuses every other change, including one the pass is
supposed to make.  Reading that failure as "the pass no-op'd" would have been the
natural inference and it would have been wrong.  Fixed by scoping the checkpoint
to the owner whose contract it verifies — keyed on the same bit the coupling
already keys on, which scopes two owners' two contracts rather than loosening
either.  Teaching the mirror to re-derive the two-phase transformation instead
would be a second implementation of the shared pass, which ``FIX_DESIGN.md`` §3.2
already rejected for this exact reason.

**Stage 2 — the pass emits source that cannot compile.**  With the checkpoint
scoped, every cell reached ``applied`` with ``_count1``/``_count2`` and exact
``torch::empty`` — and in the COUNT pass the appends survived unrewritten, so
``_cnt2`` stayed zero, against declarations ``_filtered_prefix`` had dropped.
``result_write_pass`` recognizes result writes in legacy's vocabulary only:

| what | legacy | ordered-key | recognized |
| --- | --- | --- | --- |
| append a coordinate | ``{R}{L}_crd.push_back`` | ``…emplace_back`` (``lower_llir.py:8964``) | **no** |
| append a value | ``{R}_values.push_back`` | ``…emplace_back`` (``:8955``) | **no** |
| close a position | ``Assign`` to ``{R}{L}_pos`` | ``scorch_vector_set(…)`` (``:8986``) | **no** |

And it **failed open** on all three: the pass's final ``return (node,)`` passed
them through to the external C++ compiler, which is the failure mode Phase 0's
exit criteria forbid.  Fixed: both vocabularies recognized, and a result write
this pass does not recognize now raises a structured
``unsupported_result_write_statement`` instead of being retained.  Provably inert
for legacy, which emits the other spellings, and inert on the default typed path,
where the pass does not run.

**Stage 3 — the positions are wrong, and this is where it stops.**  With stages 1
and 2 closed the kernels compile and EXECUTE, and they are incorrect: a ``dss``
receiver produces ``compressed mode 2 position array must be nondecreasing`` and
a ``ds`` receiver a coordinate range ``[0, 2100441888]`` outside ``[0, 32)``.
The two-phase path rebuilds positions from ``_count`` prefix sums indexed by the
phase loop variable; the ordered-key family closes positions through a catch-up
over its dense prefix, and for a stored outer loop the loop variable is a
POSITION rather than a row coordinate.  The two do not coincide.

This is the correctness question §60's own findings file had flagged as "the
probe cannot answer"; the answer is that it is wrong.  **Offering a strategy that
miscompiles is worse than refusing it**, so
``_OrderedKeySparseWorkspaceLowering`` lists the two single-pass strategies and a
two-pass request there fails closed with ``unsupported_assembly_host``.

### 60.7 REFUTED, this design's own: ``unsupported_assembly_host`` is not unreachable

``DESIGN.md`` §4.3 introduced ``unsupported_assembly_host`` as "fail-closed
insurance for a family added later" and P-M2 predicted it would fire **zero**
times over the legal domain.  It fires **58 of 192** cell-arms, and every one is
a real gap in what a family can emit:

- ``_ParallelSparseWorkspaceLowering`` cannot emit either single-pass strategy or
  the region-elided two-pass one.  Measured, not assumed: its own
  ``complete_sparse_workspace`` requires the assembled function to carry the
  two-phase parallel shape, so its body is not a standalone serial builder the
  pass happens to replace.  An earlier draft listed ``single_pass_serial`` there
  on exactly that (wrong) reasoning and the cell failed
  ``sparse_workspace_completion_lost``; listing only what is emittable is what
  makes the refusal name the family instead of surfacing an internal completion
  failure.
- ``_RowScopeSparseWorkspaceLowering`` emits only its serial default.

### 60.8 What the checks measure

| check | result |
| --- | --- |
| **1. Automatic-origin byte-neutrality vs ``cae4f11``** | **PASS.** 1,130 cells x both arms, compile-only: **zero** differing emissions, zero differing refusal codes |
| **2. Frontier**, 1,138 cells x both arms, base and candidate | **PASS.** 0 lost, 0 gained, 0 route changed, 0 unclassified either side, 3 arm-variant both sides (inherited, identical sets), **0 NEW arm-variance**; admitted 260 both sides |
| **3. Correctness per (strategy, cell)** | **PASS** for every pair that can be emitted.  32 configurations (2 TTM cells x 2 shapes x 2 densities x **2 dtypes** x both arms), 96 executed strategy runs: ``AUTO``, ``S1`` and ``P1`` agree **bit-identically** on every index array and every value — not ``allclose`` — and match the dense reference to 6.26e-07.  ``P2`` verified bit-identical on its own family's cell.  Extents were chosen so ``P1``'s runtime condition genuinely fires |
| **4. Refusals enumerated at their exact codes** | **PASS**, with §60.7's prediction missed.  Over 1,130 cells x 4 strategies x both arms — 9,040 compilations — **zero refusals lack a structured code** |
| **6. Schema** | **PASS.** v12 and plan-v2 declared, identity and cache-key consequences recorded in §60.4 and locked by tests |
| **5. Suite** | **IN FLIGHT, not a result.**  8 file-disjoint partitions per side, base and candidate, node set to be DIFFED rather than totals trusted.  Base partition 0 is green (513 passed / 14 skipped / 0 failed); the remaining fifteen runs are owed.  Command: ``run_suite_partitioned.sh <tree> <outdir> 8`` then ``suite_node_diff.py <base_outdir> <cand_outdir>`` |
| **7. Runtime grid, four strategies, three hosts** | **NOT RUN** (§60.9) |
| **8. ``scorch_ops`` built at both tips and compared** | **NOT RUN**, and this section adds nothing to ``csrc/`` (§60.9) |

How much can be emitted, over the 24 frontier cells that are both partitionable
and admitted, in both arms (48 cell-arms per strategy):

| strategy | emitted | refused, and why |
| --- | --- | --- |
| ``single_pass_serial`` | **46** | 2 ``unsupported_assembly_host`` |
| ``single_pass_chunk_parallel`` | **18** | 26 ``unsupported_assembly_strategy`` (stored outer loop — a correct legality refusal), 4 ``unsupported_assembly_host`` |
| ``two_pass_parallel`` | **2** | 46 ``unsupported_assembly_host`` |
| ``two_pass_serial`` | **0** | 48 ``unsupported_assembly_host`` |

Nine cells emit all four strategies' *sources*, and on all nine the four are
byte-DISTINCT — so the representation reaches emission rather than being
decorative.  ``two_pass_serial`` has nothing that can host it: the shared pass
supports it (``CompressedWhereOpenMPContext.parallel``, with the workspace view
relocated out of ``pre_parallel_body`` because codegen silently drops that field
on a non-parallel loop — §60.10), and no family's completion contract accepts the
region-elided shape yet.

### 60.9 What this section does not do

- **The suite check is not finished.**  Eight partitions per side at ~13 minutes
  each; base partition 0 passed and the other fifteen runs are outstanding.  The
  node-set diff is the part that matters — it has caught a real regression the
  totals hid twice — so no claim about the suite is made here beyond that one
  green partition.
- **The runtime oracle grid is NOT run, on any host.**  Check 7 and the whole of
  step 5 are outstanding.  The grid is the next milestone's input and it should
  not be taken until ``two_pass_serial`` has a family that can host it, because
  two of its four columns would be empty.
- **``scorch_ops`` is not built at both tips and compared.**  This section adds
  nothing to ``src/scorch/csrc/`` — ``git diff cae4f11..HEAD -- src/scorch/csrc/``
  is empty — so the obligation is unchanged from §59: the ``header.h`` +231 that
  ``ops.cpp:1`` includes and pyproject's ``depends`` lists is still covered by one
  argument ("the typed route has zero callers") when it needs two.  Still owed.
- **mkt1 is still not run.**  Owed since §59.
- **The merge's serial tail is not closed.**  ``scorch_concat_chunks`` still sizes
  its destination with ``resize``, which value-initializes before the parallel
  ``memcpy`` overwrites it.  Deliberately out of scope: it is a kernel
  optimization inside an architecture milestone and would need its own cross-host
  measurement.  **Consequence for whoever runs the oracle: ``P1``'s column will be
  measured on a kernel with a known, unfixed serial tail, so closing that tail
  later invalidates the column.**
- **The two-pass position reconstruction on the ordered-key family is not
  fixed** (§60.6 stage 3).  Diagnosed, refused at compile time, and stated.
- The dense-domain seam, the merged-domain UNION/INTERSECTION decision, the
  shadow pilot's membership and every blocker other than 1 are untouched; the
  Phase-8 cutover verdict is unchanged.

### 60.10 Defects found and recorded, not fixed here

- **``codegen.py`` silently drops ``pre_parallel_body`` / ``post_parallel_body``
  on a non-parallel ``ForLoop``**, and ``before_parallel_body`` when there is no
  split region: they are emitted only inside the split-region branch or the
  atomic branch.  The typed route refuses that shape
  (``lower_llir.py:12745``, ``:12775``) so this milestone is protected and the
  serial two-pass relocates the statement instead; legacy is not protected, and
  the house rule is fail closed.  Left as a stated defect because raising a
  ``CodegenError`` where text is silently dropped today would change what legacy
  emits, which needs its own proof that nothing shipped changes.
- **The frontier enumerates 1,139 cells over 1,138 distinct ones, and both
  numbers in the record are right.**  §55.2 says 1139 and §59.8 says 1138; the
  sealed receipt holds 1,139 records and ``frontier_diff.py`` keys them by
  ``(family, name)``, which collapses one duplicate —
  ``('rank4-mixed', 'ssss ijkl->l [d]')``, registered twice because
  ``"d" + "s"*0`` and ``"s"*0 + "d"`` are the same one-character format.  Both
  records carry identical routes in both arms, so the diff loses no coverage.
- **A harness defect caught before it informed anything.**  The legality census
  attributed a subclass's cells to its parent by reading the last name in the
  patched ``__init__`` chain; ``_RowScopeSparseWorkspaceLowering`` appeared to
  host **zero** ``ds`` receivers, which cannot be true since it is selected on
  exactly that shape.  Recorded because a census that silently misattributes is
  worse than one that crashes.

### 60.11 How cheaply the output size can be known, recorded for the next milestone's selector

Derivable from the iteration lattice, and **not built into a selector here**.  The
question is how expensive it is to learn the exact output sizes without producing
the output, because two-pass buys exact allocation with a counting pass and that
pass's price is the whole trade:

- **C0 statically countable** — every level below the assembled one is dense with
  known extents, so the count is compile-time arithmetic and no counting pass is
  needed at all;
- **C1 stream-countable** — the count is a position-array difference or a
  coordinate-run length, readable without executing the product or touching
  values;
- **C2 merge-required** — the count is a property of a union/intersection or of a
  workspace's distinct keys, known only after the merge, so counting costs about
  what computing costs.  This is legacy's redundant counting pass and it is why
  the single pass wins at one thread by 1.04–1.79x.

``TTM dss x ss -> dss`` is C2; ``TTM dss x dd -> dss`` is **C1**, because a dense
second operand makes the per-``(i,j)`` count the full ``l`` extent whenever the
``k`` stream is non-empty.  That is the same axis on which the two cells' measured
ratios diverge, which is the first evidence these classes predict anything.

**A second axis the measurements demand, offered as a hypothesis and not a
finding.**  These classes alone do not explain ``RESULT.md``'s five-density
inversion (``legacy_serial/base`` runs 0.857 → 0.953 → 1.175 → 1.561 as density
rises on one cell whose class never changes).  The candidate mechanism: the
single pass's extra cost is ``std::vector`` growth traffic, which scales with
emitted **entries**, while the counting pass's extra cost is recompute, which
scales with **merge steps**.  So the sign of ``S2`` against ``S1`` should be
ordered by ``M/E = merge steps / emitted entries`` — for TTM,
``nnz(A) / stored_ij`` — which is ≈1.2, ≈4 and ≈13 at those densities and orders
all four measurements monotonically with a crossover between 1.2 and 4.  If it
holds, the selector's rule is analytic and needs no fitted constant beyond a
crossover the oracle locates.  It is prediction P-M7c, it is the riskiest claim
in the design, and it is a hypothesis until the grid scores it.

**Evidence ledger**: ``~/.cache/scorch-codex/assembly-strategy/`` with
``SHA256SUMS`` — ``DESIGN.md`` (recorded in advance, digest in
``provenance/``), ``FINDINGS.md`` (M1 and M9a scored), the legality reach census,
the two-phase probe and its dumped sources, the emission/neutrality census, the
cross-strategy correctness run, the frontier pair and its diff, and the
partitioned suite.  Every harness takes a tree root as ``$1``.
