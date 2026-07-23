# Phase 4 Review: LoopIR Foundation and Dense Vertical Slice

Date: 2026-07-22 (America/Los_Angeles)

This review records the Phase-4 milestone of the compiler IR refactor: the
production-responsibility audit of the Phase-3.5 spike, the frozen production
LoopIR dense subset revised from it, normalized-CIN-to-LoopIR lowering for the
dense elementwise and dense reduction/matmul families, LoopIR lowering through
the existing structured LLIR into executed kernels, promotion of the spike
semantics into a production-owned oracle, deterministic printing and canonical
serialization, and strangler-path stage timing.  It builds on the repeated
Phase-3.5 review at `6ca14f5` (GO-with-conditions to begin Phase 4; NO-GO to
freeze the spike unchanged) and leaves that review unchanged.

## Verdict

**The Phase-4 dense vertical slice is complete for the migrated families**,
with every mandatory gate green:

- both dense families compile and execute correctly through the LoopIR
  pipeline and the existing structured LLIR/CxxIR boundary;
- for every family member the legacy pipeline supports, the LoopIR pipeline's
  generated C++ is **byte-identical** to the untouched legacy pipeline's C++,
  and shadow execution is bitwise-equal on real tensors;
- LoopIR stage dumps are deterministic and target-neutral, and canonical
  serialization is stable across independently constructed equivalent
  programs with different global-ID allocation histories;
- normal `import scorch`, default compilation, legacy correctness paths, and
  release JIT never load or execute the new package (subprocess and
  source-scan enforced);
- the 20-source corpus and 42-source grid captures are byte-identical to the
  retained Phase-3.5/Phase-3 captures, so the byte waiver applies and the
  retained latency receipts remain operative.

Phase 5 was deliberately not started.  The spike's fail-closed boundaries
(leaf-only merged cursors, CSR-only sparse output assembly,
COORDINATE/SINGLETON iteration) were not forced open; the production subset
does not even declare their nodes yet.

## 1. Production-responsibility gap audit

The audit compared the Phase-3.5 spike against normalized CIN and every
responsibility the first production dense families need.  Each row records the
spike's state, the production requirement, and the Phase-4 disposition.

| Responsibility | Spike state | Phase-4 disposition |
| --- | --- | --- |
| Identities, ownership, construction | Global module counters allocate `LoopNodeId`/`DimensionId`; no construction API; fixtures hand-assemble nodes. | Production `LoopIRNodeId`/`DimensionId` are artifact-local, allocated deterministically by the new `LoopIRBuilder` (`build.py`), so identical construction sequences allocate identical identities regardless of process history.  Tensors/loops keep production `SymbolId`/`IndexId`; CIN lowering passes CIN identities through unchanged as provenance.  Constructors still perform no validation (adversarial tests forge nodes directly). |
| Verifier authority and pass boundaries | Single fail-closed `verify_program` with 45 codes over the full sparse surface. | Same single-authority discipline, reduced to the dense subset: 28 stable defect codes, each with direct regression coverage, and a test locking the code surface itself.  Unknown node subclasses and forged state fail closed.  Provably dead branches (non-DENSE layouts at store sites, non-ADD `ReduceOp` members) were removed rather than left untestable. |
| dtype/scalar typing | Untyped: Python floats only; no dtype anywhere in the schema. | `ScalarType` (FLOAT32/FLOAT64) on every `TensorDecl`; the verifier requires one uniform scalar type per program (`mixed_dtype`); the CIN boundary maps torch dtypes and fails closed on anything else (`unsupported_dtype`).  Mixed-precision programs are a recorded later surface. |
| Rank, shape, broadcasting | Extents runtime-resolved via shared logical dimensions; shapes never stored in the IR. | Preserved: shapes remain runtime bindings.  The oracle and the LLIR boundary independently re-resolve every dimension's extent across all bound inputs and outputs and fail closed on disagreement.  Broadcast dimensions (a loop variable only in the result) are represented naturally and emit the legacy broadcast form.  Rank is bounded only by the family (1-3 exercised; the oracle's former rank-2 dense-output limit is gone — outputs are arbitrary-rank nested lists). |
| Aliasing and output semantics | Inputs/outputs disjoint; outputs write-only; `Store` overwrite vs `StoreReduce` ADD into zero-initialized outputs. | Preserved verbatim, and the zero-initialization contract is now explicit on the node and realized by the target lowering through the production `ResultTensorAssembler` (`scorch_zero_dense`).  In-place operands fail closed at the CIN boundary (`unsupported_inplace_operand`). |
| Reduction identities and associativity | `DeclAccum`/`Accumulate` scalar accumulators with literal identities, plus ADD-only `StoreReduce`. | The dense slice reduces exactly the way the legacy generated kernels do — `StoreReduce` (ADD) into the zero-initialized output — so the accumulator nodes were deliberately **not** copied into the production subset ("do not invent nodes before a migrated operation needs them").  `ReduceOp` declares only ADD; adding a member requires adding its explicit output-initialization identity contract. |
| Deterministic printing and canonical serialization | None (spike had no printer or serializer). | New `printer.py`: `print_program` (human-readable stage dump) and `canonical_program_dump` (compact JSON, schema `scorch.loopir.canonical.v1`).  Both verify first and renumber all identity families by first appearance; equivalent programs built under different global-ID histories produce byte-identical output; display names are omitted from the canonical form.  No deserializer exists, so no round-trip contract is declared. |
| Target separation and structured-LLIR lowering | Torch-free interpreter only; no target lowering. | The existing structured LLIR **is** the target-specific CxxIR boundary — the audit found no need for another target IR and none was introduced.  `lower_llir.py` reuses the production `TorchCppKernelABI`, `ResultTensorAssembler`, the managed LLIR pass pipeline, and the production parallel-marking policy.  LoopIR itself contains no C++ spelling; a dedicated test asserts dumps are target-neutral. |
| Compiler-stage ownership and timing | None. | Two appended `CompilerStageId` members (`cin_to_loopir_lowering`, `loopir_to_llir_lowering`) recorded through the existing `CompilationContext`.  The legacy default path never begins them; every legacy stage-sequence lock is unchanged; downstream C++ generation and build-request assembly reuse the existing canonical stages. |
| Semantic-oracle integration | Spike interpreter, neutrality-suite-isolated. | Promoted as a production-owned dense oracle (`oracle.py`), Torch-free and fail-closed, loaded only by dedicated tests.  Package-level neutrality is enforced by subprocess checks (plain import and a full legacy dense compilation) plus a production-source scan.  The spike package itself is untouched and remains the Phase-5 sparse reference. |
| Cache identity and versioning | Not applicable (nothing cached). | Kernel cache identity remains source-derived and unchanged.  The migrated families generate byte-identical source, so the LoopIR path honestly shares the legacy kernel artifact (identical source is identical kernel); LoopIR-level artifacts are never cached; canonical serialization carries an explicit schema version for any future persistent use. |

## 2. What was frozen

`src/scorch/compiler/loopir/` — the first production LoopIR subset, revised
from the spike, not copied wholesale:

- `nodes.py` — frozen dataclasses with tuple-owned children:
  `DimensionDecl`, `LevelDecl(kind, mode)`, `TensorDecl(symbol, name, dtype,
  dimensions, levels)`, `LoopProgram`, `Block`, `DenseFor(index, dimension,
  body)`, `IndexValue`, `Load`, `BinaryExpr` ({ADD, SUB, MUL}), `Store`,
  `StoreReduce` (ADD only).  All four production level kinds are declared for
  schema stability; only DENSE is executable (others fail closed with
  `unsupported_level_kind` until Phase 5).  No cursors, positions, merges,
  accumulators, workspaces, tiles, constants, or parallel nodes exist yet.
- `build.py` — `LoopIRBuilder` construction API (identity allocation only).
- `verifier.py` — the single fail-closed authority (28 defect codes).
- `printer.py` — deterministic printing + canonical serialization.
- `oracle.py` — the production-owned test/debug semantic oracle.
- `lower_cin.py` — normalized-CIN-to-LoopIR lowering for the dense families,
  returning the verified program plus a verified `LoopPlan` (nest order).
- `lower_llir.py` — LoopIR + runtime shape bindings → complete structured-LLIR
  `evaluate` function via the reused production target components.
- `pipeline.py` — the test/debug driver (compile, execute, curated shadow
  comparison); production never imports it.

## 3. The migrated families and their recorded boundaries

**Dense elementwise**: a pure `ForAll` nest over one `TensorAssign` with no
update operator; right-hand side an {ADD, SUB, MUL} tree over all-dense
accesses.  **Dense reduction/matmul**: the same shape with the ADD update
operator; loop variables absent from the left-hand side reduce via
`StoreReduce` into the zero-initialized output (row/col sums, matvec, ikj
matmul).  Ranks 1-3, float32/float64, zero extents included.

Fail-closed family boundaries (stable codes, not silent degradation):
identity mode order only (`unsupported_mode_order`); one uniform
float32/float64 scalar type (`unsupported_dtype`, `mixed_dtype`); no
workspaces/`Where` (`unsupported_workspace`, `unsupported_statement`); no
derived index arithmetic (`unsupported_index_expression`); no explicit
parallel marks (`unsupported_explicit_parallel`); ADD as the only update
operator (`unsupported_update_op`); no DIV/unary
(`unsupported_operation`/`unsupported_expression`); reductions require the
update operator (`unsupported_reduction_without_update`); every tensor's
storage-order loop variables must appear in nest order
(`unsupported_loop_order` — the same dependency direction the legacy dense
position chains require); no repeated operands
(`unsupported_repeated_operand`); no repeated per-access indices
(`unsupported_repeated_access_index`); no in-place operands
(`unsupported_inplace_operand`).

## 4. Legacy defects found by the differential work (errata/audit findings)

These are pre-existing legacy-pipeline defects surfaced while constructing
the byte-parity grid; none was introduced or changed by this milestone:

1. **Dense `ijk` matmul emission is invalid C++.**  With loop order (i, j, k)
   over `C[i,j] += A[i,k] * B[k,j]`, the legacy dense lattice emits
   `int pB1 = pB0 * B1_size + j;` before `int pB0 = k;` inside the k loop —
   `pB0` is used before its declaration.  The supported order is `ikj` (the
   codegen tests use it); the LoopIR family rejects storage orders that
   conflict with the nest with a stable code instead.
2. **`A[i, i]` diagonal accesses read the wrong position.**  The legacy
   value lowering resolves the access at the level of the *first* occurrence
   of the index variable (`A_val[pA0]` where `pA0 = i`), silently reading the
   wrong element — a wrong-result risk, not a crash.  The LoopIR family fails
   closed (`unsupported_repeated_access_index`); the shared dense-pointer
   hoist pass also mis-hoists the correct position chain for this pattern,
   so opening it requires a pass fix first.
3. **Dense elementwise SUB is unsupported by the legacy lattice.**
   `IterationLattice.gen_lattice_points` raises `NotImplementedError` on
   `Operation.SUB`, so legacy cannot compile `C = A - B` dense elementwise at
   all.  The LoopIR path supports SUB; its coverage is therefore
   oracle/PyTorch-differential only and is recorded as the one family member
   with no legacy comparand.
4. **Repeated operands break the legacy kernel ABI.**  `C = A * A` produces
   two identical argument triples and the ABI validation raises
   `ValueError: kernel ABI argument names must be unique`.  The LoopIR family
   fails closed with `unsupported_repeated_operand`.

## 5. Verification

Evidence ledger: `/Users/bobby/.cache/scorch-codex/phase4-dense-58e8565/`.

Commits (stacked on `58e8565`; nothing amended, reordered, or pushed):

- `763b73b` — `feat(compiler): freeze production LoopIR dense schema, builder, and verifier`
- `fde87cf` — `feat(compiler): add deterministic LoopIR printing and canonical serialization`
- `0959087` — `feat(compiler): promote the dense semantic oracle into production LoopIR`
- `93444b9` — `feat(compiler): lower normalized dense-family CIN to verified LoopIR`
- `26a84bc` — `feat(compiler): lower LoopIR through structured LLIR with legacy byte parity`
- `788f0e6` — `feat(compiler): drive the LoopIR dense slice end to end with stage timing`
- `6672b23` — `test(compiler): enforce LoopIR production neutrality`

Receipts:

- focused LoopIR suites at the final code commit: **135 passed** (55
  verifier, 8 printer, 16 oracle, 22 CIN lowering, 26 LLIR lowering/parity,
  8 pipeline execution — the execution differentials compile and run real
  kernels) plus 4 neutrality tests;
- byte-parity grid: 14 family members (elementwise add f32/f64, mul, fused
  three-input, vector add, broadcast, rank-3 elementwise, row/col sums,
  matvec, ikj matmul f32/f64, and two zero-extent variants) generate C++
  **byte-identical** to the untouched legacy pipeline, compared in-process
  with no external compilation;
- shadow execution: ikj matmul and elementwise add run through both
  pipelines with byte-identical sources and bitwise-equal results, and match
  the PyTorch reference; SUB, matvec, and float64 matmul match PyTorch and
  the production oracle;
- spike suites unchanged: **647 passed**; adjacency
  (CIN-analysis/LoopPlan/raw-budget: 120 passed; cin/scheduler/schedule-api:
  82 passed); stage-timing suite: 102 passed with the two appended stage
  identities recorded and every legacy sequence lock unchanged;
- byte gates: fresh 20-source corpus and 42-source grid captures from the
  working tree are byte-identical to the retained `phase35-repeat-b2fe883`
  candidate captures (`diff -rq` empty for both), which chain to the Phase-3
  finals — the byte waiver applies and no runtime kernel benchmark is
  required; structural activation tests (OpenMP marking, dense-pointer
  hoisting, zero-fill) are asserted directly and never waived;
- latency: no production emission or measured legacy compiler path changed
  (the two appended enum members are inert on the legacy path), so the
  retained paired latency receipts remain operative under the 1.10 policy;
  the LoopIR stages are strangler-path-only and excluded from release
  latency by construction, since release JIT never enters them;
- static parity against the inherited baselines: Black — exactly the one
  inherited finding (`prebuilt_kernels.py`); Flake8 — exactly the nine
  inherited findings; `mypy --check-untyped-defs src` — exactly the 146
  inherited errors in 12 files, **zero** in `loopir/`; focused mypy over the
  nine package modules succeeds; `git diff --check` clean before every
  commit;
- the authoritative clean detached-worktree non-performance suite at the
  exact final code commit `6672b23`, with import provenance asserted
  (including the LoopIR lowering, oracle, and pipeline modules) and
  caches/basetemp isolated: **3,346 passed, 14 skipped, 3 perf-marked
  deselections, and one known warning in 2,066.14 seconds** (JUnit: 3,360
  selected, 0 failures, 0 errors) — exactly the Phase-3.5 baseline of 3,207
  plus the 139 new LoopIR tests;
- the five protected tracked files retain their session-start SHA-256
  values (`.gitignore` `301c1e74…`, `pyproject.toml` `191c3372…`,
  `src/scorch/__init__.py` `5e2f22c7…`, `tests/packaging/smoke_install.py`
  `f18264fc…`, `tests/test_scorch/test_resources.py` `3d8092cb…`); staging
  used explicit pathspecs only; no GPU/CUDA, benchmark, packaging,
  scheduler, research, scratchpad, or tooling material was touched.  This
  session pushed nothing.  Origin note: the live `git ls-remote` tip of
  `origin/refactor/compiler-ir-phase3-std-move-call` was `1714df2` at
  session start and moved to `58e8565` (the reviewed Phase-3.5 session-start
  HEAD) at 2026-07-22 16:52 -0700 via a push performed outside this session
  by the repository owner; the seven Phase-4 commits remain local-only.

## 6. Limitations and Phase-5 gates

- The production subset is dense-only by construction.  Sparse iteration,
  positions, cursors, merges, COORDINATE/SINGLETON, and non-CSR sparse
  output assembly remain exactly where Phase 3.5 left them: proven in the
  spike, gated behind its fail-closed diagnostics, and not yet declared in
  the production schema.  Phase 5 extends the frozen schema deliberately.
- The family boundaries in §3 are fail-closed scope statements, not silent
  assumptions; each has a stable code and a regression test.
- Dense SUB has no legacy comparand (legacy defect #3); its numerics are
  locked against the oracle and PyTorch instead.
- `LoopIRBuilder` display names must be unique identifiers at the target
  boundary (`invalid_display_name`/`duplicate_display_name`); semantic-level
  name collisions remain legal in the IR itself.
- The LLIR target lowering accepts the family's program shape (one loop nest
  over one store leaf).  Verified-but-unfamiliar LoopIR (permuted storage
  orders, multi-statement blocks) fails closed with stable codes
  (`unsupported_mode_order`, `unsupported_program_shape`).
- The oracle infers input shapes from nested values, so a `(0, n)` leading
  zero extent is uninferable there (inherited spike boundary, tested); the
  compiled path takes explicit shapes and handles zero extents fully.
