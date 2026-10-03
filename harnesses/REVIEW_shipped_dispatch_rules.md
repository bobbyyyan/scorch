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

**The null, which overturns that reading.** A pick that moves across thread counts is evidence of
thread dependence only if it is stable at a *fixed* thread count. Repeating the probe 8 times at
each of nt=2 and nt=6:

| cell | nt=2, 8 reps | nt=6, 8 reps | within-nt switch (null) | across-nt switch | majority |
|---|---|---|---|---|---|
| N=1024 | tilej/1024 ×8 | tilej/1024 ×6, tilej/2048 ×2 | 2/14 = 0.14 | 2/8 = 0.25 | **same** |
| N=2048 | tilej/512 ×8 | tilej/512 ×7, tilej/256 ×1 | 2/14 = 0.14 | 1/8 = 0.12 | **same** |

The across-thread variation does not exceed the within-thread variation, and the majority pick is
identical at both thread counts on both cells. **The concern's stated consequence does not
reproduce here.** The key's omission of `nthreads` is real, but on these cells it is not what makes
the plan wrong.

**What the null exposed instead, with a size.** The pick is irreproducible at fixed configuration —
three distinct widths appeared at nt=6 alone — and the memo freezes whichever width won the first
call for the whole process. Timing each pick by seeding the memo and calling the production entry,
so production builds its own kernel arguments, two interleaved passes per pick, M5 at nt=6:

| cell | widths the probe flips between | runtime span | worst same-pick pass spread | ratio | v2 baseline |
|---|---|---|---|---|---|
| N=1024 | 512 / 1024 / 2048 | **4.66%** | 1.15% | 4.1× | **+54.87%** |
| N=2048 | 256 / 512 / 1024 | **6.12%** | 2.74% | 2.2× | **+94.99%** |

**So the "never slower by construction" guarantee is not what this breaks.** v2 is 54.9% and 95.0%
slower than the best tiled width in these two cells, so every width the coin flip chooses between
is far above the baseline the guarantee is about. What the irreproducibility costs is **4.7–6.1% of
the available tiling gain**, not a regression. The reviewer pointed at the right code and the wrong
consequence; the defect is smaller in kind and larger in frequency than stated.

**Not yet generalized, and this bounds what the finding is worth.** One host. The instability
appeared at nt=6 and not at nt=2, and 6 is the M5's fast-core count — the cliff already recorded
for OpenMP team width on this machine — so a hybrid-core artefact is a live alternative to a
general property of the probe. Two-host confirmation is the standing requirement for anything
quoting a percentage, and neither x86 host is available: redwood is down and mkt1 is mid-run on
job 17698209. **No code change is proposed on one host.**

**What would settle it**, in the order a second host should run it: (a) the same within-nt
repetition at 2, 8, 16 and 32 workers, where 6 has no special status; (b) whether the probe's
candidate list is dense enough that adjacent widths are inside its floor by construction, which
would make the flip a property of the search rather than of the machine; (c) only then, whether
adding `nt` to the memo key changes any decision — because on this evidence it would partition the
cache without changing a pick.

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
