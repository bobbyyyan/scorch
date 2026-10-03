# Review response: the four shipped dispatch-rule concerns

A reviewing agent raised five concerns about shipped code. #5 (`train_autotune_model.py`
exporting an in-sample figure as `heldout_geomean`) is confirmed by reading and is a labelling
defect; it is tracked separately. This file is the measurement log for #1–#4, which are claims
about dispatch rules I did not change and had not measured.

**Standing rule for this file: a concern is not a defect until it has a size and a null.** Each
entry records what was measured, the null it was measured against, and whether the concern's
*stated consequence* survived — which is not the same question as whether it pointed at real code.

---

## #4 — the tiling plan cache omits thread count (`tiling.py:428`, `:1517`)

**The concern.** `maybe_dispatch`'s memo key is `(signature, level, baseline_tag)` where
`_signature` is `(M, J, nnz, N, 4 sampled index entries)`. `nthreads` reaches the probe
(`tiling.py:1396`) and the dispatch (`:1528`) but is not in the key, so a plan probed at one
thread count is replayed at every other thread count for the same matrix and free dim. The
reviewer paired this with the "never slower by construction" claim at `ops.py:781`.

**The code reads as described.** Verified: `memo_key = (sig, level, baseline_tag)` at
`tiling.py:1523`; `nt = nthreads if nthreads is not None else -1` at `:1517` is passed to
`_dispatch_decision` and never enters the key. `torch.set_num_threads` between calls reaches this
path (the drop-in SpMM follows `torch.get_num_threads()`), so two thread counts in one process is
reachable rather than hypothetical.

**First measurement, and why it was not the answer.** Probing the same cell at nthreads ∈
{2,4,6,8} with the memo cleared between, on three DLMC matrices × N ∈ {1024, 2048} (M5, 16 MB LLC;
at N ≤ 512 **no** DLMC matrix is eligible — `J·4·N > C` fails — so the instrument refused a verdict
on the first run rather than printing six reassuring "STABLE" rows):

| cell | nt=2 | nt=4 | nt=6 | nt=8 | |
|---|---|---|---|---|---|
| `bottleneck_2_block_group4_1_1`, N=1024 | tilej/1024 | tilej/1024 | **tilej/512** | tilej/1024 | moves |
| same, N=2048 | tilej/512 | tilej/512 | **tilej/1024** | tilej/512 | moves |
| `bottleneck_2_block_group4_2_1`, N=1024 | tilej/1024 | tilej/1024 | tilej/1024 | tilej/1024 | stable |
| same, N=2048 | tilej/512 | tilej/512 | tilej/512 | tilej/512 | stable |
| `..._projection_block_group3`, N=2048 | v2 | v2 | v2 | v2 | stable |

2 of 5 eligible cells moved, and the **kind** never moved — only the panel width.

**The null, which overturns that reading — and a second run that overturns my reading of the
null.** A pick that moves across thread counts is evidence of thread dependence only if it is
stable at a *fixed* thread count. Repeating the probe 8 times at each of nt=2 and nt=6:

| run | cell | nt=2 | nt=6 | majority agrees? |
|---|---|---|---|---|
| first | N=1024 | tilej/1024 ×8 — **0/7 transitions** | 1024 ×6, 2048 ×2 — 2/7 | yes (1024) |
| first | N=2048 | tilej/512 ×8 — **0/7 transitions** | 512 ×7, 256 ×1 — 2/7 | yes (512) |
| second | N=1024 | 1024,1024,512,512,1024,512,1024,512 — **5/7 = 0.71** | 3/7 = 0.43 | yes (1024) |
| second | N=2048 | 256,512,256,256,256,512,512,256 — **4/7 = 0.57** | 4/7 = 0.57 | yes (**256**) |

**Three corrections to what the first run let me write, two of them structural.**

- The first version of this entry compared raw **counts** (`within >= across`) across different
  denominators — 14 adjacent pairs against 8 matched-index pairs — and printed "does not exceed"
  for the N=1024 cell where the *rates* are 0.25 against 0.14. The claim was false on that cell and
  the published range "0.12–0.25" hid it at the top end.
- **The two rates were never the same statistic.** Within-condition counts adjacent transitions;
  across-condition counts matched-index disagreements. For a minority set of size *m* in *r*
  contiguous interior runs the numerators are 2*r* and *m*, so when both describe the same events
  their ratio is the denominator ratio and nothing else — on N=1024, 0.25/0.14 is exactly 14/8.
  **The ratio is withdrawn**; it could not have carried information either way.
- **The first run's pooled null averaged a perfectly stable condition with the condition under
  test.** nt=2 was 0/7 on both cells and every switch was at nt=6, so the pooled 0.14 was the mean
  of 0.00 and 0.29 — half its mass was the quantity being explained away, which makes the
  comparison against nt=6 circular and leaves the comparison against nt=2 with no null at all.
  A within-condition null is reported **per condition** when the conditions are the arms of the
  contrast.

**And the second run kills the explanation I had built on the first.** nt=2 is not stable: it is
*more* unstable than nt=6 on the N=1024 cell (0.71 against 0.43) and equal on N=2048. So the
"instability appears at nt=6, which is the M5's fast-core count" reading — the hybrid-core
alternative I had flagged as live — **is dead**, because the first run's nt=2 stability was a draw
and not a property. The N=2048 majority pick also moved between runs, 512 to 256, so even the
surviving leg agrees *within* a run without the majority itself being reproducible *across* runs.

**What is left is simpler than the concern and more actionable.** The pick is substantially
irreproducible at fixed configuration — 0.43 to 0.71 adjacent-transition rate over 8 repeats, two
distinct widths in every condition of every run — and the memo freezes whichever width won the
first call for the process. **Thread count is not the variable.** Adding `nthreads` to the memo key
would partition the cache without changing a decision, because there is no thread-dependent
decision to protect; the defect is the probe's width resolution.

**The cost, with the base named, because the base inverts the ranking.** Timing each pick by
seeding the memo and calling the production entry, two interleaved passes per pick, M5 at nt=6:

| cell | widths in play | span of runtime between them | same-pick pass spread | v2 over best tiled | span as share of runtime | span as share of the gain over v2 |
|---|---|---|---|---|---|---|
| N=1024 | 512 / 1024 / 2048 | 4.66% | 1.15% | +54.87% | **4.66%** | **8.5%** |
| N=2048 | 256 / 512 / 1024 | 6.12% | 2.74% | +94.99% | **6.12%** | **6.4%** |

The span is relative to the **best tiled width**. The first version of this entry called it
"4.7–6.1% of the available tiling gain", which is the span wearing the gain's label: the gain over
v2 is 0.5487 and 0.9499 of the best width's time, so the span is 0.0466/0.5487 = 8.5% and
0.0612/0.9499 = 6.4% of it. Both bases are defensible and the pairing was not. **The choice inverts
which cell is worse** — by share of runtime the second (6.12 > 4.66), by share of the gain the
first (8.5 > 6.4) — so neither number travels without its base.

**The never-slower-than-baseline guarantee is untouched.** v2 is 54.9% and 95.0% slower than the
best tiled width in these cells, so every width the draw chooses between is far above the baseline
the guarantee is about. This costs headroom, not correctness.

**Still one host, and no code change proposed.** The hybrid-core alternative is dead but the
single-host limit is not: everything above is the M5. What a second host should run, in order:
(a) the same per-condition repetition at 2, 8, 16 and 32 workers; (b) whether adjacent candidate
widths are inside the probe's floor **by construction**, which would make the draw a property of
the search rather than of the machine, and is now the leading hypothesis; (c) whether increasing
the probe's repetition count resolves the width, which is the fix if (b) holds.

**Harnesses** (scratchpad; to be filed with the measurement if a second host confirms):
`tile_thread_dep.py` (refuses when no cell is eligible), `tile_thread_null.py` (reports its own
null and exits non-zero only when the signal exceeds it), `tile_width_cost.py` (seeds the memo and
asserts it was not overwritten, so a non-replay cannot read as a replay).

---

## #1 `nthreads >= 2 * pool` (`scorch_policy.h:2277`), #2 `pool <= 16` (`:1583`), #3 mean-degree-16 unroll (`:1420`)

Not measured. All three need an x86 host — #1 and #2 are about pool size and worker counts that the
M5 cannot express, and #3 is an AVX2 threshold on a NEON-only machine. Recorded now so the file
does not read as though the absence of an entry were a verdict.

My position from reading alone, which is not evidence: #2 is the strongest of the three, because a
boundary chosen *between* the two tested machines has no measurement on either side of it. #3's
evidence base (one i9-14900K, 12 matrices, a bin boundary standing in for an optimum) is below the
two-host standard the rest of this branch is held to. #1's 64-CPU/8-worker case is a real
counterexample to the gate's stated intent, which was byte-neutrality on all-physical-core pools.

### #4 continued — the probe measures its own floor on every call and never applies it to the width pick

A peer supplied the structural route statically. **Verified at source rather than taken on report**,
and it holds at every load-bearing point:

- The baseline is entered **twice**, as `cands = [("v2", 0, 0)]` (`tiling.py:1285`) and
  `cands.append(("v2", None, _baseline))` (`:1681`) — two identical arms at opposite ends of the
  candidate list, which is this cell's own A/A control, measured on every call.
- `_clears_noise` (`:1370`) derives `spread` from those two arms and demands the winner beat the
  *faster* of them by more than the spread. Its docstring is explicit about why: *"the probe
  memoizes its verdict permanently, so a coin flip decided by noise is not a one-call mistake, it
  is every subsequent call on that shape."*
- It has **exactly two call sites**, `:1437` and `:1694`. Both compare a winner against the
  **baseline**. Neither compares two tiled candidates to each other.
- The width is chosen by a bare argmin at `:1689-1692`, strict `<`, first wins ties, with no margin.
- Each `times[i]` is a **minimum** over `rounds=2` (`:1311`, `:1357-1365`), so the pick is an argmin
  of a downward-biased statistic over two draws — the candidate that catches one lucky round wins.

**So the quantity needed to detect that two widths are indistinguishable is computed on the same
call, by the same function, and discarded for the pick.** The file states the hazard precisely and
applies the guard to one of the two decisions it governs.

**Measured, because a structural route is not a rate.** Logging the probe's own `times` and
`spread` on each call, 24 calls per cell, M5 at nt=6:

| cell | top-two gap (median) | baseline spread (median) | calls whose top-two gap is INSIDE the measured spread |
|---|---|---|---|
| N=1024 | 1.0093 | 1.0380 | **21/24 = 0.88** |
| N=2048 | 1.0184 | 1.0276 | **15/24 = 0.62** |

The typical gap is well inside the typical floor. **The probe is choosing between candidates it
cannot distinguish, and it holds the number that says so.**

**This changes the fix.** My earlier note suggested a higher repetition count. That is wrong on its
own: more rounds shrink a floor the rule never consults. The matched change is to apply the existing
measured spread to the candidate-versus-candidate comparison and break the resulting tie
deterministically — the analytic `_panel_width` prediction, or the incumbent — rather than by
argmin. The same gap exists one level up: `_LEARNED_MARGIN` (`:1303-1305`) guards the tiled winner
against the predicted v2 and nothing guards it against the other tiled candidates. **Neither level
has a margin between two tiled candidates.**

**A documented claim that the rerun refutes.** `_interleaved_times`' docstring (`:1330-1341`)
defends rotating the start position over drawing a fresh permutation partly on the grounds that *"a
shuffle would also make a dispatch decision nondeterministic across runs, and that decision is
memoized; a verdict that varies run to run is worse than one carrying a bias the margin absorbs."*
The decision **is** nondeterministic across runs — measured above — and the cause is upstream of
ordering, so it survives any ordering discipline. "A bias the margin absorbs" is true only of the
baseline comparison, which is the one comparison that has a margin. The docstring cites real
measurement elsewhere (0.979–1.013 rotated against 1.001–1.051 shuffled) and rests this one clause
on an untested claim about its own output.

### The learned level's determinism prediction is VACUOUS, and declining to tile is the larger finding

The peer offered a cheap falsifiable prediction: `level="learned"` picks from a model prediction
rather than a timing, so its transition rate should be 0.00 where the probed levels flip. Measured,
8 repeats per condition:

| level | N=1024 nt=2 | N=1024 nt=6 | N=2048 nt=2 | N=2048 nt=6 |
|---|---|---|---|---|
| balanced | 0.29 | 0.14 | 0.71 | 0.43 |
| learned | **0.00** | **0.00** | **0.00** | **0.00** |

**The 0.00 is vacuous and the prediction is untested.** The learned level returned `('v2', None)` on
all 32 calls — it never chose a width, so there was nothing for it to be stable about. My first
version of this check verified only that a model was *present*, which is the wrong guard: the
instrument has to confirm the decision under test was actually *made*. Corrected to print the whole
pick rather than the parameter.

**And the reason it is vacuous is a bigger finding than the one it was testing.** The learned model
routes to the baseline on two cells where the probe measures tiling **54.9% and 95.0% faster**. This
is in-domain: `_load_learned_model` (`:1232-1237`) rejects on `machine_id` mismatch, version bump
and feature-schema drift, and it loaded, so this model was trained for this machine. It reached
`('v2', None)` through the prediction path rather than the eligibility gate (`:1510-1514` would have
returned `None` and memoized nothing).

**This is adjacent to concern #5 and should be read with it.** The only quality figure
`train_autotune_model.py` exports is `heldout_geomean`, and that figure is in-sample — `lreg` is
computed over all training rows and the comment above it says "optimistic". A model whose sole
published score is optimistic, leaving a 1.5–1.9× measured win on the table on the first two cells
probed, is the failure mode that mislabel conceals. **Not yet a verdict**: two cells, one host, and
I have not checked whether these shapes are inside the model's training distribution. What would
settle it: the grouped-CV figure that `group_split` already computes at `:215` and never exports,
plus the model's predicted-versus-measured ratio on these two cells.

### #5 — confirmed at source, and the learned level's prediction is wrong by 3–4× in scale and inverted in rank on the two cells probed

**The mislabel, verified.** `bench/train_autotune_model.py:298` reads
`# in-sample cell agreement (sanity, optimistic)`; `:299` computes `lreg` with
`eval_cells(rows, …)` over **all** training rows; `:305` writes exactly that into the shipped spec
as `heldout_geomean`. Two things the concern did not say, both from a peer and both confirmed:

- `heldout_geomean=round(geomean([r for r in lreg]), 4)` — `[r for r in lreg]` is an **identity
  comprehension**, a filter shape with no predicate. That is the construction that lets an unsplit
  list read as a selected one at a glance.
- **The honest figure is not missing. It is computed, printed, and discarded.** `group_split`
  (`:215`) splits by matrix so a matrix's rows never straddle train/test, and `:252-266` runs it
  over multiple seeds and prints `learned held-out geomean = …` under the correct name. The script
  holds both numbers and persists the wrong one. The fix is to write down the figure already
  running, not to add a computation.

**Scope, which bounds it.** `tiling.py`'s loader reads only `version`, `machine_id` and
`feature_names` (`:1233-1235`), so `heldout_geomean` is not consumed by dispatch and the mislabel
cannot misroute anything. It misleads a reader. The shipped value is **0.9896**.

**There is no domain check anywhere on the learned path.** `_eligible_learned` (`:1269-1276`) has a
one-line body: `return _operand_over_cache(J, N, C)`. Its docstring states the delegation —
*"Relax the degree floor + locality pre-exclusion; the model + v2 floor route the newly-admitted
products/arxiv/FEM/banded shapes"* — so the two gates that would have bounded `f_log_degree` and
`f_locality` are the two the widening removed **by name**. The feature vector is **17** wide; one
derived combination of three of them is all that is constrained.

**And the shipped artefact could not support a domain check even if the code wanted one.** The spec
carries `feature_names`, `trees`, `init`, `learning_rate`, `machine_id`, `platform`,
`n_train_rows` (1572) and `heldout_geomean` — and **no feature ranges, quantiles or bounds**. A
consumer cannot tell whether an input is inside the training domain from what ships.

**The measurement that discriminates, run because a structure is not a rate.** Capturing the
model's own predictions and timing the same candidates, M5 at nt=6:

| cell | model's predicted times | measured | |
|---|---|---|---|
| N=1024 | all six candidates in **3.04–3.95 ms** | v2 17.772 ms, best tilej(1024) 11.734 ms | v2 **+51.5%** slower |
| N=2048 | all six in **7.8–10.8 ms** | v2 43.567 ms, best tilej(512) 22.814 ms | v2 **+91.0%** slower |

Two distinct failures:

- **Scale.** Predictions are compressed into a narrow band roughly **3–4× below** the measured
  times. A gradient-boosted tree is piecewise constant and cannot extrapolate: an out-of-domain
  input is clamped into the outermost leaf region and returns whatever the nearest training region
  said, with **no signal in the output** that the input was outside. Compression far from the truth
  is that signature.
- **Rank, and the margin cannot catch it.** At N=1024 the predicted spread between the best
  candidate and v2 is `exp(−5.824 + 5.794) = 0.970` — a **3.0%** predicted improvement, which
  `_LEARNED_MARGIN` correctly rejects, against a **51.5%** measured one. At N=2048 the model's own
  argmin **is v2**, where v2 is measured 91.0% slower: the predicted ranking is inverted.

**This is the structural point made concrete.** At the probed levels the floor is a *measurement* —
two identical baseline arms bracketing the candidates. At the learned level the "v2 floor" is
`_LEARNED_MARGIN` comparing `pred[best]` against `v2_pred` (`:1303-1305`): **two predictions from
the same model.** When the model is wrong it is wrong in both terms, and the margin between two
wrong numbers carries no information about either. The floor cited to justify removing the input
gates is the model grading its own homework — and that is specific to the path the widening opened.

**What is still not established.** Whether these two cells' features lie outside the training rows'
ranges. The shipped spec carries no ranges and the 1572 training rows are not in the repo, so
saturation remains a candidate structure rather than the demonstrated mechanism. Two cells, one
host. **But the consequence does not depend on the mechanism**: on these cells the learned level
routes to the arm it would measure as 1.5–1.9× slower, and nothing in its path can detect that.

## Concerns #1 and #2 — both predicates are REACHABLE on x86; the inertness argument is a statement about one host's core count

Measured 2026-10-02 on mkt1 (AMD EPYC 9334 Zen4, 128 CPUs on the machine; this run saw
64 via its cpuset, and `omp_get_num_procs()` in-process read 64), inside a 64-CPU Slurm
allocation, against `/scr/bobbyy/adg/adg0` — verified to carry the shipped constants
(`SCORCH_SPMM_PARTITION_GATE_MAXTHREADS 16`, `SCORCH_SPMM_NT_CAP -2`,
`SCORCH_SPMM_RECRUIT_MIN_WORK 1e7`). Probe: `harnesses/pfstream/pool_reach.py`.

Line numbers: the review named `:2277` and `:1583`; the predicates are at `:2313` and
`:2612` in the tree as measured. Both were read at both sites to confirm they are the same
two predicates, so this is traceability drift and not a wrong target. The inertness claim
for #1 appears **twice** -- `:333` (the constant's doc comment, which carries the word
"provably") and `:2298` (which does not) -- plus `:2283` asserting the same of the cap, so
a comment fix has three sites.

**This is a reachability result and not a performance result.** No runtime was measured.
What follows says the predicates can fire on x86, not that firing helps or hurts.

The probe asks production rather than restating it: `scorch_spmm_nthreads` and
`scorch_spmm_partition_mode` are both exported to Python for exactly this purpose, and a
harness that recomputed either predicate would be wrong silently and flatteringly.

### Concern #1 — `scorch_policy.h:2313`, `nthreads >= 2 * pool`

The source comment at `:333` reads: *"On an all-physical-cores pool no recruit is EVER at
stake (x86: pool 24, gate at 48, count at most 32), so this condition provably cannot fire
there."* Sweeping the caller's pool over a 175-cell grid of rows x nnz x k:

| pool | gate = 2 x pool | max resolved count | cells reaching the gate, of 175 |
|---|---|---|---|
| 8 | 16 | 64 | 85 |
| 16 | 32 | 64 | 85 |
| 24 | 48 | 64 | **85** |
| 32 | 64 | 64 | 83 |

**At pool 24 -- the exact pool the comment names -- the gate is reachable on 85 of 175
cells.** The comment's premise is that the resolved count is "at most 32". That bound is
`omp_get_num_procs()`, read at `scorch_policy.h:2118` and `:2167` and nowhere else.

**Corrected**: the first version of this section said that bound "is 32 on redwood and 128
on mkt1". 128 is `os.cpu_count()`. libgomp honours the cpuset, so inside the 64-CPU
allocation the run actually used, `omp_get_num_procs()` returned **64** -- which the
`max resolved` column was already showing by reading 64 at pool 8, 16, 24 and 32 alike, a
hard ceiling rather than four coincidences. Three quantities were going by one name:

| quantity | mkt1, this run | what it is |
|---|---|---|
| `os.cpu_count()` | 128 | the machine; what the sentence quoted |
| `len(sched_getaffinity(0))` | 64 | what the process may use |
| `omp_get_num_procs()` in-process | **64** | the predicate's actual bound |

The conclusion was stated in the one quantity the probe did not read, and it named a value
no process in the run ever saw. The probe now prints all three. This **strengthens** the
result rather than weakening it: the predicate fires at an effective processor count of 64,
so an unrestricted node is not needed for it, and 85 of 175 cells at pool 24 against a
ceiling of 64 is a stronger statement than the same count attributed to 128.

It also removes a contradiction between the rule and the probe's own guard: at pool 40,
`128 >= 80` says reachable while `64 >= 80` says not, and `pool_reach.py` already warns
that a pool whose gate exceeds the allocation reports the allocation. With the right
variable the two agree.

So the inertness is real and the reasoning is sound, but what it depends on is the ratio
`omp_get_num_procs() / pool`, not the instruction set:

> the predicate is reachable exactly when `omp_get_num_procs() >= 2 * pool`.

On redwood at its 24-thread pool that is 32 >= 48, false, inert -- which is what the x86
grid measured. On mkt1 at the same 24-thread pool it is 128 >= 48, true. The comment
states a one-host fact as a property of "x86" and of "an all-physical-cores pool", and
mkt1 is both an x86 host and an all-physical-cores pool where it does not hold.

### Concern #2 — `scorch_policy.h:2612`, `pool > gate_maxthreads` disables the work gate

The comment reads: *"Above the ceiling this whole block is unreachable, which is how the
x86 reading is preserved."* That is correct and the boundary reproduces exactly at the
shipped 16. But the converse is what matters: at or below 16 the gate is **active on
x86**, and its sign was established on ARM.

| pool | cells with mode != 0 at the reference pool | of those, forced to mode 0 | share |
|---|---|---|---|
| 1 .. 16 | 165 | 29 | 17.6% |
| 17, 20, 24 | 165 | 0 | 0.0% |
| 32 | 165 | 0 | 0.0% -- IDENTITY, the negative control |

Two corrections to the first version of this table.

**The boundary is located at 16, not merely bracketed.** The original sweep stepped
16 -> 20, which produces an identical table for a ceiling of 16, 17, 18 or 19. A
geometric pool sweep is right for reachability and wrong for localization, and one grid
cannot do both. Pool 17 was added and reads 0 forced, so the transition is between 16 and
17 and matches the shipped constant exactly.

**The pool-32 row is an identity and is now labelled as such.** The reference pool is 32
and 32 was in the sweep, so at that row the two calls being compared are the same call and
`forced` could only ever be 0. The first version banded it with two genuine zeros,
inflating the count of real zeros by one. It is kept rather than deleted, because it is the
only row whose answer is known in advance: a nonzero there would mean the detection is
broken rather than that the gate is inert. The general check is to intersect the sweep list
with every constant the comparison holds fixed, which finds both the vacuous row and the
free negative control in one pass.

The `cells with mode != 0` column is the non-vacuity check: it is 165 at every pool, so the
remaining zero rows are genuine.

Reachable by ordinary means. `torch.set_num_threads(16)` or less is a normal thing for a
caller to do -- it is what a process sharing a node, or running inside a dataloader
worker, will have -- and it puts an ARM-derived gate in charge of 17.6% of the cells in
this grid on an x86 host.

### A defect in my own probe, caught by its own premise check

The first run of this probe reported predicate 1 as "reachable at pool 1 and nowhere
else". It had run inside a default `srun -n 1`, which grants 2 CPUs, so the resolved count
was 2 at every pool and `2 >= 2 x 1` held for reasons having nothing to do with x86 or
with the policy. Every other row read "never fired" because the allocation could not
supply the threads, not because the predicate refused.

A thread-policy probe is bounded by the CPUs its process may use, so it has to state that
number and compare it against the largest threshold it intends to test. The probe now
prints `len(os.sched_getaffinity(0))` beside `2 * max(POOLS)` and says, before any table,
that a "never reached" verdict is uninformative when the allocation is smaller than the
gate. This is the same shape as the harnesses that report DONE having measured nothing:
the output was well-formed, plausible, and about the allocation rather than the subject.

### What is still open

Concern #3 (the mean-degree-16 width-1 unroll at `:1420`) is untouched. Its comment
already carries a redwood grid with confidence intervals by degree band, so the question
there is a second host rather than a first measurement, and the bands are n=22 and n=12.

Neither #1 nor #2 has a runtime measurement on mkt1 yet. The next step for each is
arm-vs-arm at a fixed small pool -- not against MKL, because this build reads env vars
per call and that tax manufactures deficits on short kernels.
