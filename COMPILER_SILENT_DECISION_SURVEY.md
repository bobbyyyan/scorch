# What the compiler decides silently: a survey, a ranking, and one measurement

Date: 2026-08-16.  Written at tip `159d6e8` on
`refactor/compiler-ir-phase3-std-move-call`.  **No production code changed to
produce this document.**

Two scheduling decisions have been made explicit in two consecutive sessions —
which assembly strategy builds the sparse result (`sparse_assembly.py`) and which
scratch structure accumulates into it (`sparse_accumulator.py`) — and the second
was found by accident, as a confound while measuring the first.  Nobody went
looking for it.  That is a sampling result, and this document is the attempt to
find out how many more there are before anything is built to choose between them.

Nothing here is a recommendation until §11, which is one paragraph and clearly
labelled as an opinion.

---

## 1. What counts, and how the count was made

A choice was counted only if all three of these hold:

- **(a)** it affects how fast the emitted kernel runs;
- **(b)** more than one answer is legal for the same program — so it is a choice,
  not a correctness requirement;
- **(c)** it is currently made without consulting anything about the program's
  shape, its density, or the machine.

(c) is what makes it silent.  A choice already gated on something measured is
*informed* and appears here only as a contrast — the runtime thread gates in
`src/scorch/csrc/spmm.h`, and the register-block dual path's runtime branch on the
free dimension (`ops.py:1546-1553`), are the two clearest examples of the informed
kind, and neither is a finding.

Scope was CIN through LoopPlan, LoopIR and LLIR to the emitted C++, plus the C++
headers that are textually inlined into every generated translation unit
(`native_abi.h`, `scorch_policy.h`, `header.h` — `jit_preamble_text()` at
`utils.py:59` expands them, so a constant in there is a constant in the kernel).
The prebuilt-kernel dispatch path in `src/scorch/tiling.py` was out of scope
except as precedent, which is §7.

**Method.**  Each stage of the pipeline was read end to end by a finder, and every
candidate it produced was then handed to a second reader whose instructions were
to refute it — check the line, check the quote, check each of (a), (b) and (c)
independently, and correct the claim or throw it out.  339 candidates went in.
The verifier confirmed 134 as written, corrected and downgraded 144, and refuted
61 outright.  After applying the verifier's corrections, **132 records qualify**,
at **124 distinct code sites** (seven sites were found twice by different readers,
and two more describe one rule at two emission points).

**A coverage gap, stated up front.**  Four passes did not run: three
decomposition-blind sweeps (one over every hardcoded constant regardless of which
stage owned it, one over every ordering/tie-break, one over every unconditional
branch) and a completeness critic.  They hit an account limit mid-run.  Those were
precisely the passes aimed at decisions that do not sit tidily inside one stage, so
the reach of what follows is *stage-by-stage reading*, and a constant that is
invisible from inside its own stage could still be missing.  This is the survey's
own known blind spot and it is the first thing a follow-up should close.

## 2. The headline numbers

| | |
| --- | --- |
| candidates read | 339 |
| qualifying under (a) ∧ (b) ∧ (c) | 132 records, 124 distinct sites |
| **not** among the eight known starting points | **109 of 124** |
| never measured against their alternatives, at all | 95 of 124 |
| measured partially or indirectly | 29 of 124 |
| measured properly, against the alternative, on a grid | **0 of 124** |
| cheap to probe | 109 of 124 |
| fires on programs that ship today | 115 of 124 (50 legacy dispatch only, 64 both routes, 1 in a prebuilt C++ kernel) |
| fires only on the caller-less typed route | 9 of 124 |

The distribution by file, for the qualifying set:

```
lower_llir.py 19   scheduler.py 14   cin_lowerer.py 13   nodes.py 8   ops.py 8
iter_lattice.py 7  verifier.py 7     compressed_where_openmp_pass.py 7
parallel_marking_pass.py 7   header.h 6   compile_options.py 6   stensor.py 4
codegen.py 4   sparse_prefetch_pass.py 3   loop_plan_legality.py 3
parallel_chunk_assembly.py 3   plan_identity.py 2   scorch_policy.h 2
llir_pass_manager.py 2   dynamic_vector_access_pass.py 2   … 1 each: result_write_pass.py,
schedule_passes.py, iterdomain.py, oracle.py, native_abi.h, utils.py, llir.py
```

Two facts are worth reading off that table before the ranking.

**The single largest concentration is not in the scheduler.**  It is in the
lowerers and the C++ headers — the places that turn a decided schedule into text.
Both known decisions live there too, which is consistent: the scheduler's choices
are visible because they have a vocabulary, and the lowerers' choices are invisible
because they are spelled as literals.

**Nothing in the qualifying set has ever been measured against its alternative on
a grid.**  Twenty-nine have a partial number attached — usually a commit message
reporting a whole-workload before/after that happens to contain the change, which
is evidence that the surrounding work was worth doing and no evidence at all about
the constant.  Zero have the kind of number §68 produced for the accumulation
structure.

## 3. The ranking

### 3.1 The rubric, stated so the ranking is checkable

Expected value is *size of swing × how often it fires*.  For 95 of 124 the swing
is unknown, so the honest ranking is by:

1. **reach** — does it fire on every shipped kernel, one family, or the caller-less
   route;
2. **mechanism ceiling** — what the change could cost or save if the mechanism is
   taken at face value, which is arithmetic rather than measurement;
3. **whether anyone has a number at all**, in any direction;
4. **cost to probe**, because a cheap probe on an unknown swing beats an expensive
   probe on a known small one.

Where a swing is unmeasured this says so.  It does not guess.

### 3.2 Tier 1 — measure these first

| # | decision | site | reach | swing | probe |
| --- | --- | --- | --- | --- | --- |
| 1 | Every generated `evaluate()` revalidates its whole sparse index, serially, per element, per call | `csrc/native_abi.h:742`, emitted by `torch_cpp_abi.py:409` | every generated kernel, every call | **measured on the sibling path: 1.29–1.82 ns/nonzero** | cheap |
| 2 | The count phase of the two-phase transform re-executes all of the kernel's arithmetic and throws it away | `result_write_pass.py:622` | every compressed-output kernel on the two-phase route | unmeasured; ceiling ≈ halving total flops | expensive |
| 3 | The loop-order cost model assumes every sparse level is 0.14 % dense | `compile_options.py:309` → `scheduler.py:1186`, `:1355` | every automatically scheduled program | unmeasured | cheap |
| 4 | The six cost-model weights that decide which loop order wins | `compile_options.py:303` → `scheduler.py:1609` | same | unmeasured; fitted once, one host, March 2026 | cheap |
| 5 | The loop-order search is a single greedy inward-only pass | `scheduler.py:1640` | same | unmeasured; nests are 2–4 deep, so exhaustive costs microseconds | cheap |
| 6 | The whole automatic tiling rule: which variables are candidates, and the unconditional sparse-retraversal veto on the baseline arm | `scheduler.py:3594`, `:3615`, `:3660` | both arms of every shipped stitched einsum kernel | partially measured — 0.89 geomean on x86 for the narrow-`k` case where the veto is wrong | cheap |
| 7 | Sparse workspace initial capacity is the literal 1024, on a **coordinate-indexed** structure | `cin_lowerer.py:1036`, `lower_llir.py:8044` (and `:9611`, `:9623`) | every sparse-accumulation kernel | unmeasured; regrows per row whenever the receiver is wider than 1024 | cheap |
| 8 | The sparse accumulator is constructed and destroyed once per outer row, not hoisted and cleared | `cin_lowerer.py:1020`, `lower_llir.py:8051`, `:8392`, `:9963` | same | unmeasured; `clear()` exists and the hoisted shape is implemented next door | cheap |
| 9 | Every OpenMP schedule kind emitted anywhere is the literal `dynamic` | `codegen.py:833`; upstream `parallel_marking_pass.py:754`, `lower_llir.py:2858`, `compressed_where_openmp_pass.py:1253`, `parallel_chunk_assembly.py:560` | every parallel generated kernel | unmeasured | cheap |
| 10 | Two work grains for generated kernels (500 for the `A_nnz` measure, 1500 for the SpGEMM flop path), both excluded from the per-host autotune's writable set | `csrc/scorch_policy.h:80`, `:83` | every parallel generated kernel | partially — retuned once as part of a policy A/B, never swept | cheap |

Notes on the top three, because they are the ones that change what a follow-up
session should do first.

**#1 is the highest-expected-value item in the survey and it is not on anyone's
list.**  `validate_jit_tensor` walks every compressed span and every coordinate
with a `TORCH_CHECK` per element, and `torch_cpp_abi.py:409` appends a call for
every input tensor at the top of every generated `evaluate()`.  There is no
memoization and no fast screen on this branch — I grepped `native_abi.h` for
`weak_intrusive`, `StorageImpl`, `memo` and `screen` and got zero hits.  The same
pass on the *prebuilt* path was measured at 1.29–1.82 ns/nonzero, silently
confounded a 195-cell published study, and when removed moved the pooled result
from 0.675× to 1.878× against MKL.  The fix exists — commit `62a5a9e`, "flat
branchless parallel screens + verdict memoized per index tensor" — and
`git merge-base --is-ancestor 62a5a9e HEAD` says **no**: it lives only on
`perf/spmm-beat-mkl`.  So the generated path still pays a cost that has already
been measured, on another branch, to be large enough to invert a study's sign.
This is not a scheduling decision and it does not need a chooser; it needs a port.

**#3–#5 are one decision with three components.**  Loop order is the largest
schedule lever the compiler has, and it is picked by a cost model whose density
input is a constant (`rho = 0.0014`), whose weights (`alpha 2.975, beta 0.1005,
gamma 43.55, c_insert 85.34, c_sort 1.741, c_trans 40.61`) were fitted once on one
host and never re-fitted, and whose search is a single greedy inward-only pass.
The real nnz **is** available at dispatch; the model does not ask for it.  This is
the same shape as the two known decisions — a structural rule standing in for a
cost — except that it already has a cost model, so the work is supplying it with
inputs rather than inventing a vocabulary.

**#7 and #8 are the allocator behaviour of the structure §68 just made
explicit.**  §68 gave the accumulation structure a token and measured the swap.
It did not touch how the chosen structure is sized or how long it lives.  A
coordinate-indexed workspace declared with capacity 1024 regrows on any receiver
wider than that, and it is rebuilt from scratch on every row.  These two are
cheap, they sit in the file the last session was already editing, and their swing
is plausibly the same order as the structure swap that §68 measured at
0.865×–1.571×.

### 3.3 Tier 2 — fires broadly, mechanism is real, nobody has a number

- **Every co-iteration is a linear two-pointer `std::min` merge**, with no gallop
  or binary-search alternative representable anywhere in the IR
  (`iter_lattice.py:612`, `lower_llir.py:4983`, `nodes.py:999`).  On skewed
  intersections this is the classic asymptotic gap; the vocabulary to say
  otherwise does not exist, which is why the cost is "much more" rather than
  "cheap".
- **Which target family lowers a program is decided by `if`/`elif` order over
  non-disjoint predicates** (`lower_llir.py:15591`).  A CSR × CSR SpGEMM matches
  two legal lowerings and source order picks.  Typed route only today.
- **The rank-1 workspace drain always comparison-sorts** (`header.h:800`,
  `nodes.py:775`, `lower_llir.py:7986`), over int64 coordinates bounded by a known
  extent, so counting sort and a bitmap scan are both legal and the r/J ratio that
  decides which wins is exactly what is not consulted.
- **The rank>1 workspace dedups through an unreserved `std::unordered_map`**
  (`header.h:901`) while the constructor already `reserve`s the sibling container
  one line away, and **sorts a permutation indirectly with a per-comparison
  rank-N loop** (`header.h:961`) whose local variable is named `radixComparator`
  and is a lexicographic comparator handed to introsort.  The rank-1 version of
  this exact choice is now a declared scheduling decision; the rank>1 version is
  silent.
- **A rank-K insertion heap-allocates a `std::vector` for its key on every
  nonzero** (`lower_llir.py:9697`).
- **`#pragma omp simd` is stamped on every eligible dense reduction loop**
  (`iter_lattice.py:1235`) — the only writer of that flag in the compiler.  It
  asserts absence of loop-carried dependence, which `-ffast-math` does not, so it
  can help where aliasing blocks the vectorizer and can equally force a bad
  vectorization on a `k=3` reduction.
- **Every automatic tile loop requests unroll** (`scheduler.py:2100`,
  `nodes.py:537`, `loop_plan_legality.py:518`), which also suppresses the SIMD mark
  on that loop.
- **Tile width is 32 on the baseline arm and 8 on the register-block arm**,
  independent of the axis, the dtype, or the target's vector width — 8 is neither
  the AVX2 nor the NEON float32 lane count (`scheduler.py:3717`).
- **Tile placement is `outermost` on one arm and `child_of` the root on the
  other** (`scheduler.py:3731`, defaulted again at `:2247`), keyed on the arm flag
  alone, when `PlacementKind` already enumerates a third legal answer.
- **For a dense result, the workspace is inserted only if something will be
  tiled** (`scheduler.py:3969`, `loop_plan_legality.py:458`), and **tiling is only
  attempted on nests that already contain a workspace** (`scheduler.py:3694`).
  The two gates are mutually recursive: no workspace unless we will tile, no
  tiling unless there is a workspace, so a dense-output nest the tiling rule
  declines once can never be reconsidered.  The stated reason for the second is
  byte-preservation of the legacy scheduler's scope, which is a compatibility
  argument, not a cost one.
- **Every shipped two-phase kernel carries an OpenMP region**
  (`compressed_where_openmp_pass.py:181`): `parallel=True` is a dataclass default
  and the legacy caller at `cin_lowerer.py:4107` constructs the context without
  the field, so the serial two-pass is unreachable from dispatch — even though the
  field's own comment cites it at 0.857/0.953 against the single pass in a region,
  and an unconditional region costs 4–10 % at one thread.  The typed route can
  request it; production cannot.
- **The whole interlude between the two parallel phases is serial and O(number of
  cells)** (`compressed_where_openmp_pass.py:1664`).
- **The count and offset arrays are int32 and int64 respectively**
  (`compressed_where_openmp_pass.py:1302`); index widths are fixed generally at
  `lower_llir.py:4245`.
- **The seven-pass LLIR pipeline is mandatory and its order is frozen**
  (`llir_pass_manager.py:91`, `:1249`), validated twice so it cannot drift, and
  never compared against another order.  At least one adjacent swap is legal and
  would change the emitted C++ — though note the mechanism story in the raw survey
  entry is backwards: moving prefetch after the pointer hoist would emit *fewer*
  prefetches, not more, because the hoist deletes the `X_val[...]` accesses the
  prefetch pass discovers its arrays from.
- **Prefetch distance is exactly one element ahead, one hint per value array
  whatever the row width, locality hint always 1**
  (`sparse_prefetch_pass.py:447`, `:455`, `:467`).  The legality proof is in this
  repo: the hand-written kernels use distances of +2 and +3 and a bounded loop of
  +4..+7 (`csrc/spmm.h:1763`, `:1900`, `:2014`), and locality 3 for the identical
  access (`spmm.h:775`, `:985`, `:1666`) alongside locality 1 elsewhere.  Same
  codebase, same access, different answers.
- **The frontend's format and loop-seed decisions** — the CIN nest is the
  topological order of the einsum label graph (`ops.py:2020`); the result's
  physical mode order is bound to that seed and never re-bound after the scheduler
  picks a loop order, while the *operands* are re-bound (`ops.py:2029` vs
  `:2253`); any compressed input level makes the output level compressed
  (`ops.py:2123`); SDDMM-shaped patterns are forced to an all-coordinate output to
  reach the scalar-accumulation path (`ops.py:2172`); two torch COO tensors are
  promoted to CSR but a COO `STensor` never is (`ops.py:747`); `from_torch` pins
  each torch layout to one scorch format (`stensor.py:1230`); `to_sparse` defaults
  to all-compressed for rank ≥ 2, which misses every prebuilt CSR spec
  (`stensor.py:1764`); the transpose fast path is gated on rank 2 plus three
  literal format strings (`stensor.py:2055`); elementwise add takes the left
  operand's format and always transposes the right (`stensor.py:703`).

### 3.4 Tier 3 — narrower reach, or the typed route only

Per-worker scratch alignment and padding (`cin_lowerer.py:944`,
`parallel_marking_pass.py:816`, `header.h:160`); the atomic work-stealing clamp
bounds 16/256/128 (`parallel_marking_pass.py:610`) and its 32-bit induction
variable (`codegen.py:930`); the dynamic-vector reserve cap of 2048 and the
rank-2-only reserve hint (`cin_lowerer.py:2698`, `:2624`); append-versus-checked-set
chosen by variable-name suffix (`dynamic_vector_access_pass.py:46`); the branchless
first-touch test (`header.h:749`); the serial unglued zero-fill of the shared
position array, twenty lines from a sibling that has a 256 KB gate and a parallel
arm (`header.h:322`); `SCORCH_CHUNKS_PER_THREAD` doing double duty as a
load-balancing knob and an output-buffer budget, where the per-host autotune *can*
move it but scores it against a different kernel (`header.h:282`); per-chunk output
buffers that start empty and grow by reallocation (`parallel_chunk_assembly.py:304`);
panel window boundaries found by `std::lower_bound` on every (row, panel) pair
(`lower_llir.py:6478`); the CSR row-pointer array grown one row at a time through a
bounds-checked helper (`lower_llir.py:12754`); the ragged-tile bound realized as an
in-loop `break` rather than a clamped bound (`cin_lowerer.py:1840`,
`lower_llir.py:5434`); tie-breaks that fall through to alphabetical variable name
(`scheduler.py:1265`) or to Python set iteration order (`utils.py:634` — verified
nondeterministic across processes, but reachable only for einsums with disjoint
reduction chains, so not on matmul, SpMM, SpGEMM, SDDMM or any batched
contraction).

And one that is simply dead: **the COO row-group parallelization strategy is
unreachable** because its gate is a constant `True` at `cin_lowerer.py:3889` — the
sole call site already requires the precondition the branch tests.  Both arms are
written; one has never run outside a unit test that sets the flag by hand.

### 3.5 Three that are not performance findings, and should be handled separately

These qualify on the letter of the definition and would be a mistake to file with
the rest, because their failure mode is wrong answers or silent behaviour changes
rather than slow ones.

1. **The scalar reduction accumulator is hardwired to `float`**
   (`iter_lattice.py:1141`), whatever the tensors' dtype.  The pipeline is
   genuinely fp64-capable — `torch.float64` maps to `FLOAT64`, and
   `schedule_lowerer.py:1360` picks `FLOAT32`/`'0.0f'` versus `FLOAT64`/`'0.0'` off
   the result pointer type one file over.  For an fp32 program this is a speed
   knob.  For an fp64 program with a COO result it truncates every partial sum to
   single precision, and that is a latent precision defect, not a decision.  Worth
   checking whether any test covers float64 with an all-coordinate output.
2. **A requested parallel chunk assembly is silently downgraded to serial**
   (`parallel_chunk_assembly.py:588`) whenever the outer loop is not unit-stride,
   its bound is not a plain `Var`, or an operand prefix does not match — none of
   which `chunk_assembly_legal()` tests.  The sibling `parallel_chunk_context()`
   *does* fail closed with `unsupported_assembly_strategy` for the same class of
   request, so answering silently is a choice.  This is the house's "fail closed,
   zero unclassified" rule, and the path produces a different kernel with no code
   at all.  Typed route only, so it is not shipping today.
3. **A work estimate silently degrades to `-1` on a name mismatch**
   (`parallel_marking_pass.py:750`, and again at
   `compressed_where_openmp_pass.py:1141`).  `sparse_pos_work_expr` accepts the
   pos array only if the loop bound is spelled exactly `<operand><level-1>_size`;
   otherwise `scorch_nthreads` skips its `by_work` clamp entirely and caps by
   `rows/16` alone — which is the over-threading case `scorch_policy.h:19` records
   as having run 4–7× slower than PyTorch for a hand-written kernel.  Reachable in
   production via `cin_lowerer.py:3828` and `:3976`.

### 3.6 Two that make other measurements unsafe until they are settled

Both are cheap, and a follow-up that measures anything in §3.2 or §3.3 without
settling them first will produce numbers that do not transfer between our two
hosts.

- **`#pragma unroll` is emitted in a spelling GCC ignores** (`codegen.py:987`).
  Bare `#pragma unroll` is a clang alias; GCC's is `#pragma GCC unroll N` and it
  warns and ignores the bare form.  So the same scheduling decision is a full
  unroll on the M5 and a no-op on the Linux x86 host, and any unroll measurement
  taken on one does not transfer to the other.  Two tests assert the string
  appears; nothing checks it is honoured.
- **The compiled-kernel cache key does not distinguish microarchitectures under
  `-march=native`** (`compile_options.py:1244`).  `target_arch` is
  `platform.machine()` — `"x86_64"`, `"arm64"` — so an AVX-512 `.so` and an AVX2
  `.so` file under the same identity.  On a single machine this is inert; on a
  shared `TORCH_EXTENSIONS_DIR` spanning heterogeneous nodes, which is exactly the
  MKT cluster layout, it serves the wrong binary — slow in one direction, `SIGILL`
  in the other.

## 4. The eight starting points, checked

The prompt's list was described as a floor, not a frame.  It holds up: seven of
the eight are real qualifying decisions, and they are 15 of the 124 sites.  One
correction, per the standing rule that when the code contradicts the prompt the
code wins:

- `_select_index_vars_to_tile` is at **`scheduler.py:3594`**, with call sites at
  **`:3722`** and **`:3967`** — not `:3303`/`:3411`/`:3634` as the prompt has it,
  and not `:3090` as `COMPILER_IR_REFACTOR_HANDOFF.md:26491` has it.  Both older
  numbers are stale.
- Every other cited line is correct at this tip: `default_assembly` at
  `lower_llir.py:5648`, `:9482`, `:10557`; `default_accumulator` at `:5713`;
  `SCORCH_CHUNKS_PER_THREAD` at `scorch_policy.h:91`.

The eight are covered in the ranking as: the tiling family (#6 and the tile
width/placement entries), loop ordering (#3–#5 plus `ops.py:2020` and
`utils.py:634`), the two automatic arms (tile width, placement, retraversal veto,
and the `regblock_max_n = 8` cutoff that doubles as the tile width at
`ops.py:1618`), whether a workspace is used at all (`scheduler.py:3969`,
`loop_plan_legality.py:458`) and its dense-vs-sparse form
(`loop_plan_legality.py:331`), and chunk width and `SCORCH_CHUNKS_PER_THREAD`
(`header.h:282`, `scorch_policy.h:80`/`:83`).  The two already-explicit decisions
appear only as their "no decision" defaults (`loop_plan.py:240`,
`nodes.py:1247`, `compressed_where_openmp_pass.py:829`).

**109 of 124 were not on the list.**  That ratio is the survey's main result.

## 5. Do the two known decisions interact?

### 5.1 The question, stated precisely

Not "does swapping the accumulation structure change which assembly strategy is
faster."  §68 already answered that: the swap changes the winner on 0 of 64
configurations and shifts magnitude only, median 10.3 % and worst 57.1 %.

The question here is whether the *strategies' relative margins move together with
the structure* — because the existing selector for SpMM tiling is a learned cost
model fitted to magnitudes, not an argmax over winners, so a magnitude that
depends on the other decision is a magnitude the model has to be given both
decisions to predict.

Formally, over a fully crossed 2 × 2 in {`two_pass_serial`, `two_pass_parallel`} ×
{`coordinate_list`, `linked_list`}:

```
I = (b_P / c_P) / (b_S / c_S)   =   (b_P / b_S) / (c_P / c_S)
```

read either as "does the workspace effect depend on the strategy" or as "does the
strategy margin depend on the structure".  The harness computes both and asserts
they agree.  `I = 1` means independent.

### 5.2 The apparatus

No new timing run was needed.  §68's sealed receipt already contains the fully
crossed 2 × 2 — `~/.cache/scorch-codex/workspace-decision/receipts/decision/m5_quick.json`,
digest `74544810ac1a5ee5…`, verified against that ledger's `SHA256SUMS` before any
number was read, measured at asserted-clean commit `0f9b3b0`.  `git diff
0f9b3b0..159d6e8 -- src/` is empty, so the numbers carry to this tip unchanged.

M5, 7 rounds, 25 ms block floor, 80 rows → 40 fully crossed 2 × 2 groups, both
automatic arms, columns interleaved, min-within-round and median-across-rounds.
The null is bootstrapped from the four columns' own same-binary A/A ratios —
200 000 draws, seed 20260816 — in a four-measurement form and a conservative
eight-measurement form, rather than against a fixed ratio constant.  A fifth cell
(TTM, where the transform substitutes nothing, so `b` and `c` are literally the
same program) rides along as an apparatus control: `I` there must be 1.

### 5.3 The answer, with the numbers

**The two decisions interact.**  Over the 32 configurations where the substitution
fires:

| | |
| --- | --- |
| interaction `I` | **0.883× – 1.237×** |
| median \|ln I\| | **4.4 %** |
| worst \|ln I\| | **23.7 %** |
| outside the four-measurement A/A null | **13 of 32** |
| outside the conservative eight-measurement null | **7 of 32** |
| configurations where the faster two-pass strategy flips with structure | **0 of 32** |
| workspace effect under `two_pass_serial` | 0.876× – 1.571× |
| workspace effect under `two_pass_parallel` | 0.865× – 1.535× |
| strategy margin under `coordinate_list` | 0.097× – 0.344× |
| strategy margin under `linked_list` | 0.089× – 0.388× |

The control cell is clean: `I` spans 0.947×–1.014×, median 1.2 %, and **0 of 8**
outside either null.  So the 13-of-32 is not the apparatus.

One caveat that scopes the "0 flips" line, and it should travel with the number:
`two_pass_parallel` is 3–10× faster than `two_pass_serial` on every configuration
here.  A ranking that never flips across a 24 % perturbation of a 3–10× gap is not
surprising, and it is not evidence that rankings are stable in general.  What is
being measured is magnitude, which is the thing that moves.

### 5.4 What the interaction sorts on

It is systematic, not scattered, and it sorts on exactly one axis:

| group | n | `I > 1` on | geomean `I` |
| --- | --- | --- | --- |
| shape = square (compressed extent 256) | 16 | **13/16** | **1.0565** |
| shape = wide-workspace (compressed extent 4096) | 16 | **1/16** | **0.9473** |
| density 0.02 | 16 | 8/16 | 1.0026 |
| density 0.2 | 16 | 6/16 | 0.9982 |
| arm 0 | 16 | 6/16 | 0.9933 |
| arm 1 | 16 | 8/16 | 1.0076 |

Density splits in half.  Both automatic arms split in half and their geomeans are
0.993 and 1.008, so **the interaction is arm-invariant** — which is what the house
rule wants to hear.  The receiver's compressed extent is the whole story: at 256
the parallel strategy pays more for the structure swap, at 4096 it pays less, and
the sign is consistent across four of the five cells.

That is a useful result rather than an inconvenient one, because the receiver's
compressed extent is already one of the two inputs §67.3 said a cost model would
need.  The interaction is not on some quantity nobody was going to collect.

### 5.5 What that means for the grid

Independence would have meant four columns plus two — measure the strategies once,
measure the structures once, add.  It is not independent at the 4.4 %-median /
23.7 %-worst level, on 13 of 32 configurations, and the dependence is on a
structural quantity.  So the next grid has to be **crossed, not additive**, over
shape and density on two hosts.

The cost of that is smaller than the eight columns the prompt budgeted for, and
§6 is why.

## 6. The grid is six columns, not eight

Measured rather than read.  `probe_missing_cell.py` compiles — `compile_cin_via_loopir`
emits C++ and never invokes the external compiler, so nothing here is timed and
nothing here is a performance claim — every (strategy, structure) request over the
five confound cells × both arms, at a pinned worktree asserted clean at
`159d6e8`, and records OK or the exact structured refusal:

| strategy | `coordinate_list` | `linked_list` |
| --- | --- | --- |
| `single_pass_serial` | 10 of 10 | **0 of 10** — `unsupported_accumulator_structure` |
| `single_pass_chunk_parallel` | 4 of 10 | **0 of 10** |
| `two_pass_serial` | 10 of 10 | 8 of 10 |
| `two_pass_parallel` | 10 of 10 | 8 of 10 |

**Six of the eight columns have a producer.**  Both single-pass strategies refuse
the chained accumulator on every cell-arm, and the code cause is one line:
`require_accumulator_without_two_phase` (`lower_llir.py:5754`) refuses
`linked_list` outright, and the shared driver calls it at `:15735` whenever
`compressed_where_pass_spec is None`.  So the missing cells are a lowering
limitation with a structured refusal in front of it, not an accident.

Two other things the probe turned up, recorded because a later census keying on
codes alone would misread them.  `single_pass_chunk_parallel` additionally refuses
`coordinate_list` on 3 of 5 cells with `unsupported_assembly_strategy` — an
assembly-side legality refusal unrelated to the accumulator.  And the TTM cell
refuses `linked_list` under both two-pass strategies, which is the 8-of-10 rather
than 10-of-10 above.

## 7. Is `src/scorch/tiling.py` already a home for this?

**No, not as it stands — and the parts worth reusing are the methodology, not the
mechanism.**

`tiling.py` is a *runtime dispatch* selector.  It is handed a materialized CSR
operand, reads its metadata (M, J, nnz, degree statistics, a locality term `W*`),
and picks among prebuilt kernels; on the `max` level it times a first-call micro
probe with the v2 kernel always a candidate, caches the answer per machine, and
confirms against v2 before shipping the choice.  Every input it uses is a property
of data that exists.

The compiler's decisions are made before any of that exists.  A `TensorVar`
carries shape, format, dtype and mode order — no nnz, no degree distribution, no
measured anything.  That is not an oversight; it is what makes the emitted `.so`
cacheable across calls.  So the ladder's features cannot be lifted across, and
neither can the runtime probe: you cannot time four candidate kernels at compile
time without compiling four kernels, and the compile is the expensive part.

There is already a bridge, and its shape is instructive.
`compiler_schedule_search_space` (`tiling.py:577`) and
`schedule_from_tuner_choice` (`:613`) turn a tuner choice into an explicit
`Schedule` — but the docstrings say plainly "does not participate in the
production native selector, its persistent cache, or its learned-model features"
and "opt-in and is not used by production dispatch", the tile-j arm returns
`None`, and the only caller is `tests/test_scorch/test_autotune_levels.py`.
`COMPILER_IR_REFACTOR_HANDOFF.md:26470` already draws the same line from the other
side: the autotune-level machinery, its cache and its learned model are "kernel
choices, not schedules", explicitly outside the JIT migration.

So these need their own mechanism.  What should be **reused rather than rebuilt**:

- **The level ladder as a user-facing concept and its API.**  `set_autotune` /
  `autotune` / `get_autotune` with `off / analytic / balanced / max / learned` is a
  vocabulary Bobby's users already have.  A second, differently-spelled ladder
  inside the compiler would be the mistake this job was asked about.  One ladder,
  two consumers.
- **The train-equals-serve featurizer discipline.**  The learned model's 17
  canonical features are computed by the same code offline and online.  Whatever
  compile-time features a compiler cost model uses, that property is the one worth
  copying, and the 2026-08-16 ablation is the cautionary half: the models' stamped
  `heldout_geomean` was in-sample, the true held-out numbers were 0.980/0.976, and
  one feature did not pay for itself.
- **The v2 floor and `_confirm_vs_v2`.**  A selector that can only ever be
  neutral-or-better because a known-good default is always a candidate and the
  choice is confirmed against it.  The compiler's equivalent is "legacy's schedule
  is always a candidate", which is also exactly the byte-identity gate stated as a
  cost rule.
- **The per-machine persistent cache**, whose keying and invalidation are already
  solved.
- **The "provably inert on shapes it would hurt" gate design** — the `N ≥ 512`
  gate on tile-ijk is the precedent, and the same construction is what would let a
  compile-time chooser ship without a grid on every family at once.

What cannot be reused: the CSR-metadata features, the first-call runtime probe,
and the argmax-over-kernels framing.  A compile-time chooser has different inputs,
a different cost of being wrong (a cached `.so`, not one call), and — per §5 — at
least two decisions that have to be predicted jointly rather than one at a time.

## 8. What did not qualify, and why

207 of the 339 candidates failed the three-part test.  Grouped by which part
failed, so the survey is checkable rather than anecdotal:

- **62 fail (b) — only one answer is legal.**  Mostly verifier and legality code
  doing its job: tile-width legality is deliberately extent-free
  (`verifier.py:1216`), assembly-strategy legality is checked without extents by
  design (`:3006`), an index may be bound only once (`:1017`), a stack workspace
  region must be fully zeroed because any cell in the clamped window may be read
  (`nodes.py:633`).  Also here: cases where the "alternative" would be wrong —
  `loop_invariant_factor_pass.py:711` refuses to factor-hoist indexed
  accumulations, which is a correctness requirement dressed as a choice.
- **61 fail (a) — no effect on the emitted kernel's speed.**  Compile-time-only
  costs (a linear scan instead of a hash map in `cin_analysis.py:64`), ordering
  that no consumer observes (free/reduction index tuples returned in first-seen
  order and immediately converted to sets), and diagnostics that reject programs
  rather than choosing among them (`MAX_NESTING_DEPTH = 64`).  `oracle.py` is in
  this bucket wholesale: it is a semantic test oracle, not a cost oracle.
- **19 fail (c) — already keyed on something measured**, or are informed choices
  whose inputs simply arrive at runtime rather than compile time.
- **4 fail both (a) and (b).**
- **3 were recorded by their finders as explicit contrasts** and confirmed as
  such: the shipped runtime branch on the free dimension (`ops.py:1579`), the
  non-dense-receiver tiling guard which is a legality rule not a choice
  (`scheduler.py:3714`), and automatic tiles recorded as `accumulation='direct'`,
  which is plan-only.
- **58 had a wrong line, a wrong quote, or a claim that did not survive reading
  the consumer.**  This is the honest cost of a broad sweep, and it is why every
  candidate was verified rather than reported.

## 9. Where the numbers come from

| claim | source |
| --- | --- |
| 339 candidates, 132 qualifying records at 124 sites, verdict split 134/144/61 | this session's survey pass and its adversarial verification pass; per-entry file, line, quote, and verifier note retained |
| the interaction number and everything in §5.3–§5.4 | `~/.cache/scorch-codex/decision-survey/receipts/interaction.json` and `interaction.log`, re-analysing `~/.cache/scorch-codex/workspace-decision/receipts/decision/m5_quick.json` (digest `74544810ac1a5ee5…`, verified against that ledger's `SHA256SUMS`, measured at asserted-clean `0f9b3b0`) |
| `git diff 0f9b3b0..159d6e8 -- src/` is empty, so §68's numbers carry to this tip | verified at this tip |
| six of eight columns have a producer, and the refusal codes | `~/.cache/scorch-codex/decision-survey/receipts/missing_cell.json`, produced by `harness/probe_missing_cell.py` at pinned worktree `worktrees/tip-159d6e8`, asserted clean at `159d6e8` |
| the code cause of the two missing columns | `lower_llir.py:5754`, called at `:15735` |
| §68's own "0 of 64 winners change, median 10.3 %, worst 57.1 %" | review §68.5 |
| the revalidation cost, and that its fix is not on this branch | commit `62a5a9e` message; `git merge-base --is-ancestor 62a5a9e HEAD` returns non-zero; `git branch --contains 62a5a9e` lists only `perf/spmm-beat-mkl` |
| the serial two-pass at 0.857/0.953 | the field comment at `compressed_where_openmp_pass.py:168-180`, citing `ttm-density-mechanism/ABLATION.md` |
| the veto's 0.89 x86 geomean | `ops.py:1454-1457` rollout comment, M5 2026-07-09 and x86 2026-07-10 |
| the prebuilt tiling swings quoted as context (tile-i +1–3.5 %, tile-j 0.67×–6.76×) | the tiling studies in `bench/`, which are about prebuilt kernels, not generated ones |
| `_select_index_vars_to_tile` at `:3594` with call sites `:3722` / `:3967` | grep at this tip |
| `tiling.py`'s bridge is opt-in and test-only | `tiling.py:577`, `:613`, `:630`; the only caller is `tests/test_scorch/test_autotune_levels.py` |
| `compile_cin_via_loopir` / `execute_cin_via_loopir` have zero non-test callers | grep over `src/` and `tests/` at this tip: 24 test modules, no production module |

The ledger is `~/.cache/scorch-codex/decision-survey/`, sealed over 10 files with zero
compiled artifacts and every one verifying; `SHA256SUMS` hashes to
`159d3e3215746cc9…`.  It holds the two harnesses, their three receipts, the four census
files behind §2, and `provenance/tip.txt`.  The pinned worktree both harnesses ran
against is in the ledger and excluded from the seal, as git-backed scratch.

## 10. What this document does not do

- **No production code changed.**  The only tracked file under `src/` that differs
  from `HEAD` is `src/scorch/__init__.py`, which is one of the five files belonging
  to the separate CUDA project and was already modified before this session
  started; the untracked `src/scorch/csrc/cuda/`, `src/scorch/gpu.py` and
  `src/scorch/stensor.py.orig` are the same project's.  Nothing in this session
  touched any of them.  Because `src/` is untouched, the emission, census and
  release-neutrality checks are satisfied by construction and were not re-run.
- **Nothing found here was fixed.**  Several of these are obviously wrong — the
  fp64 accumulator, the GCC unroll spelling, the dead COO branch, the work
  estimate degrading to "unknown" on a name mismatch.  They are written down and
  left alone, because a survey that fixes things as it goes stops being a survey
  and its findings stop being comparable.
- **No decision was made explicit, schedulable or configurable, and no chooser was
  built or extended.**  No default, threshold or heuristic moved.
- **No four-strategy grid was built.**  §5 sizes it; it does not run it.
- **The new pipeline stays unwired.**  `compile_cin_via_loopir` and
  `execute_cin_via_loopir` still have zero non-test callers.
- **The one measurement is a sizing input, not a shipping claim.**  M5 only, one
  receipt, re-analysed rather than re-run.
- **The survey has a stated blind spot** (§1): three decomposition-blind sweeps and
  a completeness critic did not run.

## 11. The one paragraph that is an opinion

If it were mine to sequence: **port `62a5a9e` first, settle the two measurement
hazards second, and only then build the crossed grid** — and I would not start a
chooser this milestone.  The revalidation pass is the highest-expected-value item
in the survey by a wide margin, it is the only one with a measured swing on a
comparable path, the fix is already written on a sibling branch, and every runtime
number anyone takes on a generated kernel before it lands is contaminated by an
O(nnz) serial term that has already inverted one study's conclusion — so measuring
anything else first means measuring it twice.  The unroll spelling and the
`-march=native` cache key are the same argument in miniature: they are each under
a hundred lines, and until they are settled, a number from the M5 and a number
from redwood are not comparable for any decision that touches an unrolled or
vectorized loop, which is most of §3.2.  On the grid itself, §5 says crossed and
§6 says six columns, and I would run it on the receiver's compressed extent as the
primary axis rather than on density, because that is the axis the interaction
actually sorted on and density did not move it at all.  I would resist building a
chooser for the two known decisions even though the grid will make it tempting,
for the reason this session exists: 109 of 124 silent decisions were not on the
list two sessions of measurement produced, so a chooser fitted to two of them is a
chooser that gets rebuilt when the third lands — and the third is now something we
have 122 candidates for rather than a hypothetical.  What I would build instead,
when something gets built, is the thing §7 describes: one ladder with two
consumers, the compiler side keyed on what a `TensorVar` actually carries, with
legacy's schedule as the permanent v2-equivalent floor.
