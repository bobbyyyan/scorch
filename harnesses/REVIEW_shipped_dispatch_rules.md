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

### Concern #2, continued — the 17.6% was not attributable, the isolated figure is 16.7%, and the gate is toggleable on MKT

Measured 2026-10-02, same host and allocation. Three corrections and one enabling result.

**WITHDRAWN: the earlier design was not confounded, and this paragraph originally said it
was.** The first sweep varied the caller's pool and recomputed the resolved thread count
from it, so one swept variable reached both arguments of `scorch_spmm_partition_mode`. I
concluded from that structure that the number was not attributable, and named the
single-worker gate as the second consumer.

That gate is `partition_solo_off && nthreads <= 1` and
`SCORCH_SPMM_PARTITION_SOLO_OFF` is **0** at `:531`, so it cannot fire in a shipped build
-- and it is the only one of the four predicates that reads `nthreads` at all. Measured
over 288 cells at a fixed pool, `partition_mode` is identical at nthreads of 1, 2, 8, 32
and 128, at both pools: **zero cells differ**. With the solo gate switched on by its hook,
nthreads=8 against nthreads=1 moves **215** cells, which is the control showing the
comparison could have detected a difference had one existed.

So the second argument the swept variable reached has no effect at shipped settings, the
old figure was not confounded, and the 29-against-35 difference was entirely the grid
(175 cells against 216). I reasoned from the shape of the call rather than measuring
whether the shape mattered, which is the same error as inventing a mechanism for a
misattributed failure -- a structural argument standing in for a measurement.

What survives: holding the count fixed is still the right construction, and running it is
how the independence above is now known rather than assumed. It is a result, not a repair.

Instrument: `harnesses/pfstream/gate_isolate.py`, grid 6 rows x 8 nnz x 6 k = 288 cells,
committed so every figure below is regenerable. The first version of this entry quoted
figures from a script that existed only in a scratchpad, in a commit that touched no code,
so its "grid used" was not the committed probe's 175-cell grid and no number in it could be
reproduced from the repository. That is the defect `scratchpad is not storage` names.

**Isolated, by holding the resolved count fixed and varying only the pool:**

| held nthreads | gate-attributable | live cells | share |
|---|---|---|---|
| 2 | 65 | 280 | 23.2% |
| 4 | 65 | 280 | 23.2% |
| 8 | 65 | 280 | 23.2% |
| 16 | 65 | 280 | 23.2% |
| 32 | 65 | 280 | 23.2% |

Identical at every held count, and every row now reads **exact** against the composed
prediction. The share moved from 16.7% to 23.2% when two cells were added to the grid,
which is the plainest possible demonstration that a share of a synthetic grid is a property
of the grid -- the prevalence section below computes it on a real collection instead.

Identical at every held count, which is the expected result rather than a suspicious one:
the gate's test is `work_proxy = nnz * max(k,16)` against `mingrains * SCORCH_GRAIN_SPMM`,
and neither side contains a thread term. The conclusion of the earlier entry is unchanged
-- the gate is active on x86 at pool <= 16 and its sign was established on ARM -- but the
share is now attributable where before it was not.

**A third leg: the closed form, which the two env routes cannot supply.** `work_proxy`
contains no thread term and no row term, so the fired set is fixed by `(nnz, k)` and the
two committed constants -- `SCORCH_SPMM_PARTITION_MINGRAINS 2L` at `:1581` and
`SCORCH_GRAIN_SPMM 150000L` at `:82`, giving a threshold of 300000. The prediction is 36
cells, from 6 `(nnz,k)` pairs (all at nnz=3000) times 6 row values.

**All four predicates composed, and both columns are predicted from source.** At the large
reference pool the ceiling fires, so the work gate is skipped and the only exclusion is
maxout: 280 live of 288 predicted, 280 measured. At the small pool the ceiling does not
fire, so the work gate applies on top: 65 predicted, 65 measured, at every held thread
count including 1. Previously the ceiling was implicit in the choice of reference pool
rather than part of the prediction.

**The prediction's only host constant, and how much it matters.** Sweeping `llc_bytes`:
16,777,216 gives 272 live / 63 attributable; 33,554,432, 50,000,000 and 64,000,000 all give
280 / 65; 67,108,864 gives 288 / 66. So the prediction is constant exactly across
(32,000,000, 64,000,000] -- the same interval the `elem_size` probe bounds -- and changes
outside it. The result therefore needs `llc_bytes` *bounded* rather than known, which the
probe supplies independently. It is not host-free, and the coincidence of the two intervals
is not luck: both derive from the same two products.

**Composed, the prediction is exact.** A closed form for one predicate in a chain is an
upper bound, not a prediction, and an uncomposed one turns a disagreement into a narrative
instead of a defect -- which is what the first version of this section did, explaining its
one leftover cell in prose afterwards. `scorch_llc_bytes()` is exported, reads 33554432
(32.0 MiB) on mkt1, and `MAXOUT_LLC` is 2L, so maxout is 67108864 exactly. Composing the
work gate with maxout predicts **65** of the 66 the work gate alone predicts, removing
exactly `(rows=1000000, nnz=3000, k=32)` at `1000000 x 32 x 4 = 128000000 >= 67108864`.
Production returns 65. The cell is predicted rather than accounted for.

The measured bound of (64M, 128M] from the elem_size probe contains the exported 67108864,
so the two routes to the constant agree.

**Corrected: as first run, leg 3 BOUNDED the constant rather than confirming it.** The
original grid was geometric, with a 16x gap between nnz=3000 and nnz=20000. Its largest
firing work proxy was 96000 and its smallest non-firing one 320000, so *any* threshold in
(96000, 320000] -- a factor of 3.33, covering `MINGRAINS` of 1 or 2 against `GRAIN_SPMM` of
50000 or 150000 -- predicts the identical 36-cell set. A geometric grid cannot resolve
candidates 2x apart. No cell had work exactly equal to the threshold either, so the
comparison's strictness was untested as well.

Fixed with two bracket points rather than a finer grid. At k=16, nnz=18749 gives a proxy of
299984 and nnz=18750 gives exactly 300000, so under a strict `<` exactly one fires -- and no
other candidate threshold produces that split (100000 and 150000 give zero firing, 450000
and 600000 give two). Re-run with them in: the prediction is 66 cells, production attributes
**65 with zero extra**, so every nnz=18749 cell fires and every nnz=18750 cell does not.
**The product is exactly 300000 and the comparison is strict.** The constant is now
confirmed rather than bracketed.

This leg matters because the two env routes only establish that two readings of the source
agree with each other; both go through the same exported function, so neither can tell
whether that function reads the constants the header declares. A linked object whose
`MINGRAINS` or `GRAIN_SPMM` differs from the header -- which is what a defines-only rebuild
relinking stale objects produces -- would leave the two routes agreeing and both
disagreeing with the closed form.

**The row-dependent exclusion is identified, and confirmed from a second direction.**
`scorch_spmm_partition_mode` holds five predicates, and exactly one of them has a `rows`
term: `rows * out_cols * elem_size >= maxout`. So a row-dependent exclusion can only be
that gate -- an enumeration with a one-member candidate set, not a proposed mechanism.

It predicts a movable boundary, which is what makes it testable rather than merely
plausible. The probe passes `elem_size`, so at 8 bytes the product doubles and the
exclusion must move from k=32 to k=16. Measured at rows=1e6, nnz=3000:

| elem_size | k=8 | k=16 | k=32 |
|---|---|---|---|
| 4 | product 32M, mode 3 | product 64M, mode 3 | product 128M, **mode 0** |
| 8 | product 64M, mode 3 | product 128M, **mode 0** | product 256M, mode 0 |

The exclusion moved exactly as predicted, and the boundary sits at the same product both
times. With `SCORCH_SPMM_PARTITION_MAXOUT_LLC 2L` at `:534` that bounds the host's
`scorch_llc_bytes()`: live at 64,000,000 bytes and excluded at 128,000,000 gives

    maxout in (64,000,000, 128,000,000]   so   llc_bytes in (32,000,000, 64,000,000]

**Corrected, and the first version of this line excluded its own measured value.** It read
"maxout in (64M, 128M], hence an LLC of (32 MiB, 64 MiB]". The products that produced the
bound are decimal, so halving them gives decimal bytes; writing the halves in binary units
makes the interval (33,554,432, 67,108,864], whose lower endpoint is **open at exactly the
33,554,432 that `scorch_llc_bytes()` returns** -- so the published bound excluded the
measured value, and its upper end implied a maxout of 134,217,728 which the probe had
already ruled out. The slip is invisible because 32 and 64 appear on both sides of the
conversion. Stated in bytes from here.

**Two invariances the predicate's own form demands, both run.** Over the held thread count:
identical sets, as required. Over rows: the harness reports **DIFFERS**, and correctly --
6 `(nnz,k)` cells at each of rows 64, 256, 512, 4096, 65536 and only 5 at rows=1000000.
That is the same single cell as above, so the check fired on a real asymmetry and the
asymmetry has an explanation outside the gate. A check that passed silently here would have
been the weaker outcome.

**A defect in my own toggle, which read as a missing feature.** To test whether the build
could switch the gate off I set `SCORCH_SPMM_PARTITION_GATE_MAXTHREADS=0`. That is
backwards. The source is

    if (gate_maxthreads > 0) { if (pool > gate_maxthreads) mingrains = 0; }
    if (mingrains > 0) { if (work_proxy < mingrains * SCORCH_GRAIN_SPMM) partition_mode = 0; }

so 0 skips the *pool ceiling*, which leaves `mingrains` at its default and the work gate
**active at every pool** -- the opposite of off. Zero means "unset this guard" in one
constant and would have to mean "disable this mechanism" in the other, and the knob's name
discloses neither. The probe reported "hooks are NOT live" on a build where they are.

**The gate is toggleable on MKT, so no build is needed.** Ten hooks builds already exist
there; `adg/*` are not among them (0 of 4 knob strings present in any, which is why the
first probe read nothing). `/scr/bobbyy/raisefloor/tree` carries 49 `SCORCH_SPMM_*`
environment strings. With the resolved count held at 8 and the pool at 8 (control row first, because it is the
one that separates a live hook from a live hook whose object carries a stale default):

| arm | cells with mode != 0 | moved vs shipped |
|---|---|---|
| `MAXTHREADS=16` (shipped) | 175 | - |
| `MAXTHREADS=4` (ceiling fires, so the gate is skipped) | 210 | 35 |
| `MINGRAINS=0` (the gate's own documented "leaves it out") | 210 | 35 |
| `MINGRAINS=2` (shipped value, as a control) | 175 | **0** |

The two independent routes to "gate off" agree on all 35 cells and disagree on none, so
the two readings of the source are consistent. The last row is the control that matters:
setting a knob to its compiled default must move nothing, and it moved nothing -- without
it, a live hook and a hook whose default differs from the header would look the same.

Each arm sets exactly one environment variable, so the hooks build's per-call `getenv` tax
is equal on both sides. That tax is why the runtime contrast must be arm-against-arm and
never against MKL.

**What this enables, and what is still not measured.** The runtime question -- does the
ARM-derived gate help or hurt on x86 at pool <= 16 -- is now a single-binary A/B on 35
identified cells, with no second build and so no layout artefact between the arms. Still
unmeasured: any runtime at all, for either concern.

### Concern #2, prevalence on a real collection — 30.2% of DLMC, and it needed no host

Measured 2026-10-02 locally. The grid figures above are on synthetic `(rows, nnz, k)`
tuples, and a share of a synthetic grid is a property of the grid. The decisive question is
how much of a real collection the gate governs, and the closed form answers it from
dimensions alone: the gate forces `partition_mode = 0` when
`nnz * max(k,16) < 300000`, so it needs `nnz < 18750` at k <= 16 and `nnz < 9375` at k=32.

Over the 3820 DLMC matrices **this project measures** -- the screen's corpus and
chain207's, which is the right population here but is not all of DLMC; the same predicate
over chain209's 3846 keys gives 1180 = 30.68%:

| k | matrices in the gate region | of 3820 | share |
|---|---|---|---|
| 1, 2, 4, 8, 16 | 1154 | 3820 | **30.2%** |
| 32 | 724 | 3820 | 19.0% |

The flat share across k <= 16 is the `max(k,16)` in the work proxy, not a coincidence.
The collection's nnz runs min 6, median 52428, max 8.5M, so the region is the lower third
rather than an edge case.

And over the 4594 SuiteSparse matrices cached on mkt1, with **both** predicates applied:

| k | gate alone | + output-bytes | + operands only | share of 4446 operands |
|---|---|---|---|---|
| 1, 2, 4, 8 | 2181 | 2181 | 2069 | **46.5%** |
| 16 | 2181 | 2180 | 2069 | **46.5%** |
| 32 | 1379 | 1378 | 1294 | **29.1%** |

**The operand predicate is symmetric and I verified one half where it cannot fire.** The
first version excluded `shape[1] == 1` on SuiteSparse and reported 0 one-row entries on
DLMC -- the corpus where *both* halves are vacuous. On SuiteSparse, `shape[0] == 1` catches
**23** more entries, and all 23 sit inside the region at every width: a one-row CSR has
`len(indptr)-1 = 1`, so its nnz is tiny and its `rows * k * 4` is `4k`, which never trips
the output-bytes predicate. They inflate the region exactly as column vectors do.

With both halves, non-operators are 125 + 23 = 148, the operand denominator is 4446, and
the shares are 46.5% and 29.1% -- 0.93 and 0.92 pp below the uncomposed 47.47% and 30.02%.

**The shape predicate is better than the filename one and is still not complete.**
Showing a filename match misses 80% of the non-operators does not show that
`min(shape) == 1` catches all of them, and the complement was untested: an entry can be a
non-operator and still have `min(shape) > 1`. A multi-input state-space `B` is n x m, a
multi-output `C` is m x n. Measured on the cache, the complement is **not empty**:

| min(shape) of the 1066 non-square entries | count |
|---|---|
| 1 (the shipped predicate's catch) | 148 |
| 2-4 | 41 |
| 5-16 | 56 |
| 17-64 | 57 |
| 65-256 | 105 |
| >= 257 | 659 |

`boneS01_C` is 3 x 127224, `inlet_B` is 11730 x 2, `rail_1357_B` is 1357 x 7. And the
families confirm the role: 26 base names hold both a degenerate and a thin member, e.g.
`bips07_1693_iv` at (13275, 1) beside `bips07_1693_B` at (13275, 4) and `bips07_1693_C` at
(4, 13275) -- a four-input power-system model, where the single-input members are the n x 1
the shipped predicate already catches.

**No cutoff completes it, which is why the fix is a bracket and not a wider predicate.**
The distribution above is continuous from 2 upward with no gap, so any `min(shape) >= c`
is a chosen constant of exactly the kind this project does not ship, and `bibd_12_4` at
(66, 495) is a combinatorial design matrix -- genuinely sparse, genuinely an operator,
and thin. The share's sensitivity to c is therefore the honest statement:

**And the bracket must vary only the incompleteness.** The first version of this table
swept `min(shape) >= c` over the whole corpus, which also discards every SQUARE matrix
below c x c -- operators this very section argues are legitimate, and small ones, so they
sit preferentially *inside* the small-nnz region being measured. At c = 17 that sweep
removed 120 entries where only 97 thin candidates exist below 17; the excess 23 were
square. The span it produced, 6.14 pp, measures the share and the operator definition
moving together. Restricting the cutoff to non-square entries gives the sensitivity to
the incompleteness alone, with the square count the naive sweep would have taken shown
beside it:

| c | operands | k<=16 region | share | k=32 region | share | squares spared |
|---|---|---|---|---|---|---|
| 2 (shipped) | 4446 | 2069 | **46.536%** | 1294 | **29.105%** | 0 |
| 3 | 4439 | 2062 | 46.452% | 1287 | 28.993% | 1 |
| 5 | 4405 | 2028 | 46.039% | 1253 | 28.445% | 1 |
| 17 | 4349 | 1972 | 45.344% | 1197 | 27.524% | 23 |
| 65 | 4292 | 1918 | 44.688% | 1144 | 26.654% | 102 |
| 257 | 4187 | 1831 | 43.731% | 1060 | 25.316% | 234 |

Over a 128-fold range of c the share moves **2.81 pp** at k<=16 and 3.79 pp at k=32,
against the 6.14 pp the contaminated sweep reported. So the headline does not hinge on the
incompleteness, and the margin is twice what the first version claimed. The divergence
between the two sweeps begins at c = 17 and not before, because nothing square in this
cache has min(shape) below 5. The denominator moves with c in both (an excluded entry
leaves numerator and denominator together), which is stated because the two conventions
differ by over a point at the widest cutoff and a share alone does not say which was used.
Reported at c = 2: the only cutoff needing no chosen constant.

**A suffix list over-counts, which is why it is not the test.** Of the 158 entries matching
`_(b|B|C|E|iv|x|rhs)` with `min(shape) > 1`, **70 are square** -- `LF10_E` at 18 x 18 and
`ted_B` at 10605 x 10605 are descriptor-system mass and input matrices that are perfectly
good SpMM operators. A square matrix is an operator whatever its suffix. The two tests that
carry the finding above use no list: the `min(shape)` histogram and the mixed-family count.

**This does not change the region list, and that is a different question from the share.**
A 4 x 13275 CSR times a 13275 x 16 dense is a valid SpMM and a valid kernel cell; whether a
user would ever call it is workload realism. The denominator matters for the prevalence
claim, where counting a right-hand side as an operand inflates it. The emitted list feeds a
runtime A/B that needs valid shapes, not realistic ones, so it stays at 2069 members.

**The one-row count rests on a property that was implied and is now printed.** All 23
one-row entries leaving the k=32 region requires nnz < 9375 for every one of them. That is
not free: `LF10000_C` is 1 x 19998, and a dense row would give nnz*32 = 639936, outside the
region, so a dense-row `_C` would break the count. Measured max nnz among the 23 is **64**
(`inlet_C`, shape (1, 11730), nnz 64), so the arithmetic holds -- these are sparse output
maps picking out a handful of states, not dense ones.

**The pp shifts are rounded from the difference, not differenced from rounded inputs.**
47.475 - 46.536 = 0.939 and 30.017 - 29.105 = 0.912, so **0.94 and 0.91**; the previously
published 0.93 and 0.92 came from differencing 47.47 - 46.54 and 30.02 - 29.10, off by one
hundredth in *opposite* directions. Pre-rounding an input can move a difference either way,
so it is not a conservative simplification. And the 0.939 pp bundles two corrections of very
different size: composition is **0.022 pp** and operand exclusion is **0.917 pp**.

**The denominator counted vectors as matrices.** SuiteSparse ships the right-hand side of
each linear system alongside the operator, and a one-column entry stored as CSR has
`len(indptr)-1 = rows` with very few nonzeros -- so a predicate selecting on *small nnz*
collects them preferentially. 125 of the 4594 cached entries have `shape[1] == 1`, and 88
of them were inside the k=16 region. Excluding them shrinks numerator and denominator
together, so the published 47.5% and 30.0% become 46.8% and 29.5%: a small move, but the
earlier figures were a share of files rather than a share of operands.

The suffixes say what these are. Of the 125 one-column entries: 34 end `_B`, 29 end `_b`,
11 end `_iv`, and the rest are numbered or bare; the one-row entries are `_C` members like
`LF10_C` at shape (1, 18) and `LF10000_C` at (1, 19998). These are the components of linear
systems and state-space models -- right-hand sides, input matrices, initial values, output
matrices -- shipped alongside the operators. `shape` is the predicate and the filename is
not: a convention matching `_b` alone would have found 29 of 148. It
also supersedes an nnz floor I had noted as a design need after seeing region members with
nnz = 1 over hundreds of thousands of rows -- those are these, and "not an operator" needs
no chosen constant.

The single entry the output-bytes predicate removes is one of them: `Hamrle3_b.npz`, shape
`(1447360, 1)`, nnz 440 -- the right-hand side of Hamrle3, whose operator is
1447360 x 1447360. So that removal was never about a matrix.

DLMC carries no such contamination: over all 3820 keys the minimum column count is well
above 1 and there are no one-column or one-row entries, which is what the collections'
contents would predict -- pruned weight matrices rather than linear systems.

The denominator is the readable count, which here equals the file count -- all 4594 parse,
so the two coincide; an earlier version published the file count while computing the
readable one, which is only correct when they do.

**The composition matters structurally even though it removes one matrix.** The gate is one
predicate of four, and a share computed from it alone is an upper bound -- the same omission
the synthetic-grid section had already corrected one commit earlier, repeated here on a real
corpus. The omitted predicate excludes when `rows * k * 4 >= 2 * llc`, needing rows >=
1048576 at k=16, and SuiteSparse's largest row count is 9,888,048, so it is **live** on this
corpus.

On DLMC it is **vacuous**: the sealed dims ledger gives a maximum of 33288 rows over all
3820 keys, so the largest `rows * 32 * 4` in the collection is 4,260,864 -- a factor of
15.75 under the threshold -- and the predicate excludes 0 of 3820 at every k in
{1,2,4,8,16,32}. So the 30.21% and 18.95% above **are** the composed shares exactly, and
this finding is isolated to SuiteSparse, whose row counts are unbounded.

**So at any caller pool of 16 or fewer, an ARM-derived gate decides the row-handout mode
for about a third of DLMC and nearly half of SuiteSparse on an x86 host.** That is the
reason the runtime question is worth measuring rather than leaving at reachability.

Instrument: `harnesses/pfstream/npz_gate_region.py`. It now reads nnz and rows **exactly**,
from each member's `.npy` header, which states the shape and the dtype -- so there is
nothing to infer. Index dtype over all 4594 readable files is `int32` uniformly.

The first version inferred nnz by dividing the ZIP directory's byte count by a guessed
element size, inside a `for isz in (4, 8)` loop that **could only ever return 4**: dividing
a fixed byte count by a larger element gives a smaller count, so if the first pair passed
the positivity test the others were unreachable. It then discarded the element size before
the filter. Its docstring described a selection the code did not make, and its twelve-matrix
cross-check could not cover a dtype the loop could not reach.

**The row axis is now checked rather than taken from a directory name.** `len(indptr) - 1`
is the row count for CSR and the **column** count for CSC, and the output-bytes predicate
reads rows -- so the composed column rested on the cache being called `csrcache`. The
archives carry `shape`, and `len(indptr) - 1 == shape[0]` holds for all 4594 with zero
disagreements. The number that makes that non-vacuous: **1066 of the 4594 are non-square**,
and only those can discriminate CSR from CSC, since a square matrix satisfies the check
under either layout.

Corpus-wide non-squareness does not certify the axis for the matrices a *figure* is about,
so it is also reported per region: **611 of the 2180** at k=16 are non-square (28.0%) and
478 of 1378 at k=32 (34.7%).

One further confound, and it is bounded rather than removed. `scipy.sparse.csr_matrix`
infers `shape = (len(indptr)-1, max(indices)+1)` when none is given, and an archive written
from such an object satisfies the check *by construction*. The discriminator is
`shape[1] > max(indices)+1`, which can only hold if `shape` carries information the arrays
do not -- a matrix with an empty trailing column is not reconstructible from `indices`
alone. On a 60-entry sample, **16 are strictly greater** and 44 are equal.

**And one witness settles it, not sixteen** -- a correction to how the first version of this
paragraph read the result. Whether `shape` was inferred from `indices` is a property of the
code that *wrote* the archives, not of any individual matrix, so a single entry where
`shape[1] > max(indices)+1` proves the writer did not infer, for every archive that writer
produced. The 44 equal cases are resolved by the same argument rather than left open; the
26.7% rate answers a different question (how often a matrix happens to carry an empty
trailing column), which is a fact about the matrices and says nothing about the check.

The premise is a single writer for the whole cache, and two uniformity counts collected for
other reasons support it: the member set is identical on 4594 of 4594, and the index dtype
is `int32` on 4594 of 4594. A second writer would have had to match both. What would retract
it is evidence of incremental or multi-script construction, and those two counts are the
evidence against.

A `format == b'csr'` check was written first and is not used, because **no file in this
cache carries a `format` member** -- they are not scipy `save_npz` archives, and a check
that exempts a missing key passes on all 4594 while proving nothing. The check I expected
to be weak is the strong one and the one I expected to be decisive is absent.

**The one matrix composition removes is the same matrix at both widths**: `Hamrle3_b.npz`,
nnz=440 over 1,447,360 rows. Its nnz is below the k=32 region's own bound too, so the
counts of 1 and 1 are one matrix and not two -- which the counts alone left open.

**A guard that could not run.** The unreadable-files branch had `%d` on one string and its
argument on another, so it raised `TypeError` before reaching its `return 1` -- and it
executes only when files are unreadable, which is exactly the condition it exists to
report. It fails closed by accident, a traceback also being a non-zero exit, but the named
count and clean exit it claims never happened. Third guard on this branch that could not
fire on its own input.

Two things about that, in opposite directions. It was **one-signed**: a true int64 index
read as int32 doubles the inferred nnz against an upper-bound predicate, so matrices with
real nnz in [9375, 18750) were dropped and none was added -- the published figure was a
lower bound presented as exact. And it turns out to have been **accidentally correct here**,
because the corpus is uniformly int32. That is luck, not design, and the fix removes the
dependence on it.

One design note for the runtime work: the region's smallest members have nnz = 1, some with
hundreds of thousands of rows. Those are not meaningful SpMM subjects and must be excluded
by a stated floor rather than left to dominate a geometric mean.

Provenance: **three** independent records of rows and nnz agree on all 3820 common keys
with zero disagreements -- the sealed `verifier-dlmc-dims` ledger (digest verified),
`dlmc_rowdisp.py`'s own read of the `.smtx` row pointers, and `chain209-dlmc-dims`, sealed
in a different milestone before this thread existed. A prevalence figure computed from one
record of the corpus would be a claim about that record.

The two dims ledgers share the filename `dlmc_dims.csv` and look unrelated on inspection:
different row counts (3846 against 3820), different byte sizes, different digests, `cmp`
reports DIFFER and `diff` reports every line changed. The causes are a `dlmc/` key prefix
on one side and a two-line `#` preamble on the other. Normalized, they are identical. **A
digest mismatch between two records of the same quantity is evidence of nothing until the
comparison is normalized** -- this nearly read as a provenance alarm.

## Concern #2, runtime: the work gate's threshold was calibrated against a different mode

Every earlier probe on this predicate measured which cells it selects. This section is
about time, and about where the gate's original evidence does and does not reach.

### The measurement

One object, arms are env vars read per call at `:2532` and `:2599`, interleaved inside one
process. Pool 8, width 16, float32, mkt1, 120 cells, isolated at `omp_get_num_procs()` = 14
so concern #1's adoption path cannot fire. Readings are tax-free: a caching `getenv`
returns the same pointers and flushes on every write to the environment, so decisions are
unchanged while the hooks build's measured 20.13 us per-call `getenv` cost stops being
per-call.

| route | population | median A-B | sign count | scale | ratio A/B |
|---|---|---|---|---|---|
| R1 gate as shipped | T in-region | **+14.224 us** | **120 of 120** | 29.86 us | **1.6175** |
| R1 | C1 floor, same matrices at k=32 | +0.008 us | 79 of 120 | 26.03 us | 1.0003 |
| R1 | C2 floor, same width, larger nnz | -0.000 us | 60 of 120 | 23.35 us | 1.0000 |

Arm A sets `MINGRAINS=2`, the shipped value, which forces mode 0. So **the gate selects the
slower mode on its entire treatment population**, by 61.75%, on every cell. Under the
theoretical null P(all 120 positive) is 10^-36.1; under C1's own observed rate, 10^-21.8.
Min-of-reps and median-of-reps agree to 0.005 us with comparable arm spreads, so the
estimator choice is not carrying it.

### A proposed explanation, WITHDRAWN, and the error is worth more than the claim was

An earlier version of this section argued that the threshold had been calibrated against a
mode the gate does not gate. The reasoning: `dcf2e1f` states its separator as "the cells
where the partition falls more than 10% behind **the shared counter**", and at that commit
`SCORCH_SPMM_PARTITION_DEFAULT` was **0**, so -- the argument went -- back-stealing was not
what "the partition" meant, and `8ccd104` introduced it when it flipped the default to 3.

**That is wrong, and the policy header at `dcf2e1f` says so in terms.** Its comment on the
same macro reads: "0 = one global atomic counter, which is what ships today ...; 3 =
contiguous home ranges with stealing from the back of a victim's range." Same number, same
semantics, at the calibrating commit. `steal` appears 41 times in that commit's `spmm.h`.
And its `SOLO_OFF` comment eleven lines above describes "what the `p3` arm means on the tiny
cells partway through a study whose other arms are already measured" -- so that study had a
p3 arm and had already measured others. `8ccd104` changed which mode SHIPS. It did not
introduce the mode, and the contrast the gate decides was 0 against 3 before and after.

The error is the generalisable part: **a default flip tells you what shipped, not what was
swept.** A calibration grid studying a feature that is off by default must set it
explicitly, so `PARTITION_DEFAULT 0` carries no information about which arms the study ran.
And the inverted prior is in the same comment, in the past tense: that study is described
as one "whose other arms are already measured". A study with measured arms, at a commit
whose header documents mode 3 by number, makes it likely that mode 3 *was* among them --
the opposite of what the withdrawn argument assumed. (The following sentence there, "priced
as its own arm first, then flipped", is `SOLO_OFF`'s own prospective plan for itself, not a
declared project method, so it is the weaker of the two citations and not the one this
rests on.)

What remains open is narrower and empirical: `dcf2e1f`'s separator column is labelled only
"the partition", so whether it is mode 3, a pooled aggregate over variants, or the
non-stealing partition with p3 priced separately is a question about that grid's arm labels.
Answering it also answers the base-rate question below, since the same grid holds both. The
grid is not readable from this tree.

### And x86 at pool <= 16 was never measured, by that commit's own account

`8ccd104` says, of the pool ceiling: "ungated the work gate costs 7.0% and 8.5% across the
whole x86 corpus, uniformly at every k. Capped at 16 it is **unreachable on a 24-thread
pool**". Its x86 grid is redwood's, at a 24-thread pool -- so the ceiling made the gate
inert for that entire grid, and the headline 1.3242x never exercised it. The gate's only
positive evidence is the next sentence: "On the **6-thread host** the gate recovers a third
of the row partition's only regression", and that host is the ARM one (the same commit
reports "1099 passed, 48 skipped on the ARM host").

So the evidence partitions cleanly:

| configuration | gate fires? | evidence |
|---|---|---|
| x86, pool > 16 | no, ceiling blocks | 1.3242x grid, gate inert |
| x86, pool > 16, ceiling removed | yes | **costs 7.0% / 8.5%** |
| ARM, 6 threads | yes | recovers a third of one regression |
| **x86, pool <= 16** | **yes** | **none** |

The last row is the gate's active range on x86 and it is empty. The 7.0%/8.5% cost figure
already has the sign I measure; the measurement above is the same sign, larger, inside the
range the ceiling deliberately leaves open.

### What this does NOT establish

**The duration band is not evidence either way, and I withdraw leaning on it.** The header
reads "the cells where it falls more than 10% behind the shared counter sit at 19-30
microseconds where the cells it wins sit at 31-110". `dcf2e1f` gives the quartiles those
came from: losing cells at **19 / 27 / 28 / 30** and winning cells at **16 / 31 / 62 / 110**
at min / q1 / median / q3. So "19-30" is min-to-q3 of one population and "31-110" is
q1-to-q3 of the other -- different functionals in one sentence -- and the winning cells'
minimum of 16 is *below* the losing cells' minimum. The two distributions overlap across
the whole of 19-30 and at least a quarter of winning cells sit under 31. A cell measuring
29.86 us is therefore in a region both populations occupy. What survives is only that 29.86
is inside the losing cells' IQR (27-30) and below the winning cells' q1 (31).

**And those microseconds carry their own object.** `dcf2e1f` swept the LLC-divisor ladder
through a hook and closes by noting "a hook that is measured and rejected still costs a
getenv on every call in every instrumented build" -- so its durations were very likely
taken on an instrumented build and carry that build's `getenv` tax, as this concern's own
readings did before the shim. The comparison would need both sides on the same footing.
Not reading, inference: `dcf2e1f` does not name its object, and if the ladder was built
per-value with `-D` rather than swept by env var its durations are tax-free and the bands
stand as written.

**The band is not the criterion in the first place.** `dcf2e1f` scores the shipped
predicate as "catching 74% and 77% of the regressed cells against 14.5% and 13.7% of the
winning ones" -- those are P(in region | outcome). The 120-of-120 above is
P(slower | in region). Converting between them needs the base rates, how many regressed and
how many winning cells that grid held, which the commit never states. So 14.5% is not a
false-positive rate for the gate without those counts, and the right comparison is against
P(outcome | region) measured on the original grid.

### The pool sweep, which is the axis that could have reversed this

The shipped comment at `:1595-1603` says the gate's sign depends on the pool and the
ceiling exists to make that dependence explicit, so a single pool could not speak for the
predicate. Four pools x four widths, 120 cells each, 144 rows parsed with none unplaced.
Positive = the gate selects the slower mode; the ratio is mode 0 over mode 3.

| pool | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| 2 | +1.99 us, x1.068 | +2.44, x1.099 | +2.50, x1.077 | +3.01, x1.086 |
| 4 | +3.75 us, x1.174 | +4.25, x1.232 | +4.27, x1.185 | +4.28, x1.185 |
| 8 | +14.12 us, x1.618 | +12.83, x1.573 | +15.43, x1.623 | +13.71, x1.568 |
| 16 | +12.19 us, x1.534 | +11.03, x1.489 | +12.01, x1.494 | +10.94, x1.460 |

Every cell is positive, every interval excludes zero, and the sign count is 120 of 120 in
15 of the 16 cells and 118 of 120 in the other. Tax-free cell scales run 20.5 to 34.6 us.

**The floor multiple is computed per cell, not from two ranges.** An earlier version of
this section said the floors were "0.01 to 0.029 us against effects of 2 to 15 us, so 100
to 1000 times too small" -- which multiplies the ends of two separate ranges and is only
the range of the ratio if the extremes co-occur. Those inputs in fact permit 69x to 1500x.
Measured per configuration, against the larger of that cell's own two controls:

| pool | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| 2 | 249x | 610x | 227x | 431x |
| 4 | 1877x | 425x | 329x | 329x |
| 8 | 1765x | 6416x | 3856x | 2284x |
| 16 | 12185x | 3675x | 1200x | 377x |

The minimum is **227x** at pool 2, k=8. The upper end is not a stable quantity -- a floor
that happens to land near zero inflates the ratio without saying anything -- so the claim
is the minimum. Note that the published 100-to-1000 figure was wrong in *both* directions:
arrived at by an invalid method, and understating the measured separation by an order of
magnitude at the top. It needed correcting irrespective of which way it moved.

**The sign does not depend on the pool across the gate's whole active range on x86.** But
the magnitude does not simply grow, and saying so pooled two levels that should be
separated. By pool: about 7-10% at pool 2, 17-23% at pool 4, **57-62% at pool 8**, and
**46-53% at pool 16**. Those last two supports are disjoint and pool 16 is lower at all
four widths, 4 of 4, so the shape is a rise to a maximum at pool 8 followed by a fall.

A reversal at the low end was the outcome that would have made the gate defensible with
only its threshold placement in question, and it does not occur.

**The turn matters for the pools still running, so the expectation is registered here.**
Pools 20 and 24 sit on the falling side, so R2 there should read *below* pool 16's 46-53%.
A reading at or above pool 8's level would mean the turn is not a trend. And the header
supplies a prior for it at `:1600`: its pool-24 anchor is 0.785, which in this file's
time-ratio convention is **1.274** -- below pool 16 and consistent with a continuing
decline. Different host and different population, so suggestive rather than matched.
The header's own `:1605-1612` says the same thing a second way: with the ceiling disabled
so the gate fires at pool 24, "that is the configuration x86 measured at 4-9% cost" -- same
sign as this whole table, far below the pool 8 peak.

**One qualification the gate's own criterion supplies.** `dcf2e1f` defines materiality as
"falls more than 10% behind". At pool 2 all four ratios -- 1.068, 1.099, 1.077, 1.086 --
are *below* that bar. So at pool 2 the gate is harmful by a measurement that excludes zero
and unanimous in sign, but not by the standard the gate itself was fit to. From pool 4
upward it clears that bar by 1.7x to 6x.

**And this does not contradict the comment's pool-6 anchor.** That anchor -- the counter
winning at pool 6 -- is an ARM reading, identified by `8ccd104`'s test line and by
`:1595-1603` itself ("The pool the surrounding pipeline runs is 6 there and 24 on
redwood"). Nothing in the header claims the counter wins at low pool on x86, which is the
row that was empty. Pool 6 is not measured here; it lies between pool 4 at +3.75 us and
pool 8 at +14.12 us, both positive.

### The apparatus checked its own premises on these runs

| premise | reading |
|---|---|
| the shim flushed on every arm switch | 32401 flushes against 9720 switches, per width |
| callers pass stable literals | 1 string-compare hit against 1.73e9 by pointer identity |
| the estimator is not carrying the effect | min-of-reps and median-of-reps agree to 0.005 us |
| concern #1's adoption path is inert | `omp_get_num_procs()` capped below 2*pool at every pool |

The one non-literal caller in 1.7 billion lookups is why the cache owns its keys rather
than borrowing them: with a borrowed pointer that single case could have left a dangling
key, and the diagnostic that found it costs nothing.

### Scope

One host, float32, 120 cells per configuration, isolated allocations where concern #1
cannot fire. Pools 20 and 24 are running: above the ceiling R1 becomes a declined argument,
which supplies a floor on the treatment's own cells, and `8ccd104` already measured that
construction at about 1% on redwood at pool 24 -- so those pools are a replication of a
prior reading rather than a new instrument. Two registered expectations for them: the floor
at or below that 1%, where materially above would indict the shim or the affinity cap
rather than the gate; and the R2 effect below pool 16's 46-53%, per the turn above.

**Two ambiguities in the shipped comment itself, both affecting how a later reader uses
it.** At `:1600` the anchor is written "no-partition / partition = 0.785 pooled, 65% of
cells more than 10% behind". Read as a time ratio that says the counter wins, contradicting
the clause it sits inside; read as a performance ratio it agrees. The clause settles it --
"65% of cells more than 10% behind" only parses with the counter behind -- but a later
reader extracts the number, not the clause, and this file's own tables had to state their
direction explicitly for the same reason. Separately, `:1593` gives the ARM host's pipeline
pool as 6 and `:1601` calls it "the M5's 12"; the surrounding text gives that host's
`omp_get_num_procs()` as 18, so three counts are in play under similar phrasing and only
the 6 is the pool the contention argument is about.

Not established: a cause. The explanation attempted above is withdrawn, and the remaining
candidate questions are about `dcf2e1f`'s grid -- its arm labels, and the regressed and
winning cell counts needed to convert its P(region | outcome) scores into the
P(slower | region) measured here. That grid is not in this tree.
