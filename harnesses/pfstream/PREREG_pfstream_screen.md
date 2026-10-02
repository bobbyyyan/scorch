# PFSTREAM screen: what is fixed before the numbers exist

Job 17680942 on MKT, submitted 2026-10-01 08:36 PDT. Four builds x 418 SuiteSparse matrices x
five widths x {ours, oursb, mkl}, four passes. Nothing in this file is derived from a timing.

## The arms, and why the floor is a fourth build

`ship` is the shipped configuration: prefetch run-ahead 2 nonzeros, clipped at the row boundary.
`strm` sets `SCORCH_SPMM_REGBLOCK_PFSTREAM=1`, clamping the same run-ahead to the end of the
nonzero array instead -- branch-free, a compare and a cmov against a loop-invariant bound.
`nopf` sets `SCORCH_SPMM_REGBLOCK_PFD=0`, no prefetch at all.

`shpb` is `ship`'s configuration built a second time at an equal-length path. It is the floor,
and it has to be a build rather than a second arm inside one process, because the build floor's
whole increment over the arm floor is reproduced by two byte-identical objects. Verified: `ship`
and `shpb` have identical `.text` (`5cea843d...`), asserted by `pfs_build.sh` rather than printed.

## The pre-registered prediction, which is a conjunction and can fail

From `scorch_policy.h:732`, written before this run: `strm` must be **at least as fast as `nopf`
wherever B is under a quarter of the cache** (the guard is gone there and there is no miss for a
prefetch to cover) **and at least as fast as `ship` where the prefetch earns its keep** (the
coverage is extended rather than removed). One object beating both arms.

Failing either clause is a result. If it fails both, the prefetch has no defensible form and
deleting it is what is left -- which is also a result, and not a reason to reach for a width rule.

## The windowing decision, fixed now because deciding afterwards is choosing the analysis the
## result suggests

Four builds in four positions over four passes is a Latin square: each build occupies each
position exactly once, so a position effect cancels exactly. **That property does not survive
being windowed.** Drop one pass and each build has three of four positions with a *different*
position missing per build, so the effect no longer cancels and it biases the arms
differentially. On this host that is not hypothetical: a 48-CPU co-tenant arrived 2h34m into a
previous timing run, and this screen is longer than that.

The decision, fixed before any pass lands:

* **Four passes land** -> analyse all four, position effect cancels by construction.
* **Three land** -> analyse them, and report the position effect's size beside every number.
  The instrument is already in the design and needs no extra arm: `ship` and `shpb` are
  `.text`-byte-identical, so **every ship-vs-shpb difference across positions IS the position
  effect measured on the same code**. If that bound is larger than the effect being claimed, the
  three-pass analysis does not support the claim and says so.
* **Two or fewer land** -> no arm ordering is read. Resubmit.

## Duration: measured, and the extrapolation it replaces was a prefix

The job was sized from a 40-matrix dry run at 82s, giving 856s per object-pass and 3.81h for
sixteen. **That 40 was a prefix of the strided sample, not a stride of it.** A prefix is only a
sound basis for extrapolation if the list's order is uncorrelated with cost; this list is ordered
by SuiteSparse group name, which makes the prefix acceptable by accident rather than by design.
So the figure is superseded by the landed passes' own wall times as soon as the first one closes,
and the ETA is quoted from those with its denominator named.

## Corpus reconciliation, done before the timings rather than after

The cache holds **4594 `.npz` files**; the board's SuiteSparse corpus is **4584 keys**. Those two
numbers do not differ by ten matrices in either direction, and both of the obvious readings are
wrong:

| quantity | value |
|---|---|
| `.npz` files in the MKT cache | 4594 |
| distinct basenames among them | 4578 |
| extra files sharing a basename with another | 16, over 15 basenames |
| board corpus keys (flat, one per basename) | 4584 |
| MKT basenames absent from the board | **0** |
| board basenames with no MKT file | **6** (`bone010_B/C/M`, `boneS10_B/C/M`) |

The 16 extras are **genuinely different matrices that share a name across SuiteSparse groups**,
not duplicate copies: `Nasa/barth4` and `Pothen/barth4` differ in file size (213026 vs 352810
bytes) and in content digest. So the cache holds more FILES and fewer of the board's NAMED
matrices, and the board's flat key `ss/barth4/barth4` names only one of the two -- which one is
not recoverable from the flat list.

Consequences, stated so they are not rediscovered later:

* Resolution here is **exact path**, never basename, so the ambiguity cannot be silently resolved
  the wrong way. All 4594 resolve as `exact`.
* The strided sample is 418 cache entries = **418 distinct matrices under 417 distinct names**
  (both `barth4` files are in it).
* Any later comparison against a board number is across **different populations** and needs the
  intersection taken on exact matrices, not on names.

## AMENDMENT 1: k=2 CANNOT SHOW THE EFFECT, AND IS THEREFORE THE NEGATIVE CONTROL

Found before any pass landed, by reading the dispatch rather than the timings.

`SCORCH_NARROWK_EXACT_HI` ships at **3** (`scorch_policy.h:1518`) and `SCORCH_NARROWK_EXACT_MINDEG`
at **0L** (`scorch_policy.h:562`), so `exact_width = B1_size` is set for every k in {1, 2, 3} on
**every** matrix (`spmm.h:4160`) -- the degree floor at `spmm.h:4183` is guarded by
`narrowk_exact_mindeg > 0` and so never clears it. That the floor is 0 is what makes k=2 a control
on the whole corpus rather than on a subset. At `spmm.h:4979` a nonzero `exact_width` routes the
row to `scorch_spmm_row_narrow_exact` and `continue`s, before the first `SCORCH_RB` invocation.

**The half-vector block is what actually has to be ruled out, and `exact_width` is not what rules
it out.** It sits at `spmm.h:4896`, *before* the `exact_width` branch, and
`halfvec_exact = (B1_size == LH)` at `spmm.h:4899` carries **no `exact_width` test at all**; only
`halfvec_under` at 4903 does. If `halfvec_exact` were true it would preempt the exact branch and
the row would reach `regblock`. What protects float32 k=2 is that `scorch_simd_half<float>` is
`__m128` with `lanes = 4` (`spmm.h:2202`), so `2 == 4` is false. Nothing about `exact_width`
enters it.

This matters for the dtype this screen is not running. `scorch_simd_half<double>` has `lanes = 2`
(`spmm.h:2220`), so at **float64 k=2** `halfvec_exact` is `2 == 2` and the earlier block wins
despite `exact_width == 2` -- the row reaches `regblock` and PFSTREAM applies. It does not fire
today only because `SCORCH_SPMM_HALFVEC_F64` ships at 0 (`scorch_policy.h:1515`). The override at
`spmm.h:4128` is inside `#ifdef SCORCH_TUNE_HOOKS` (opened at `spmm.h:4070`), so in a release
object it does not exist -- **the hazard is one define away, not one env var**, which is a smaller
reach than the first version of this paragraph claimed. So "k=2 cannot fire" holds for **float32 at
release defaults**, and the two dtypes are protected by two different mechanisms, neither of them
`exact_width`.

**And the same block changes what k=4 measures.** `SCORCH_SPMM_HALFVEC_F32` ships at 1
(`scorch_policy.h:1512`) and `LH = 4`, so at float32 k=4 `halfvec_exact` is true and the row takes
`scorch_spmm_row_regblock<scalar_t, 1, true, VH>` with `VH = __m128` -- the **half-vector**
instantiation, which does receive `SCORCH_SPMM_PFLIM`. So the arm applies at k=4, through a
different vector type than at k=8. Both are NVEC 1, so the prediction's NVEC=1 clause spans two
widths with *different vector widths*, and they must be reported separately rather than pooled.

**PFSTREAM lives in `scorch_spmm_row_regblock`. So at k=2 it is not merely small, it is
unreachable.** Of the five widths timed, four can show an effect: 4, 8, 16, 32.

This is worth more than the width it costs. k=2 is a **negative control that the design already
contains**: a width where the arm under test provably cannot fire, measured in the same passes,
on the same matrices, through the same calibration. Any `strm/ship` or `nopf/ship` movement at
k=2 is floor, not effect. Pre-registered predictions, which can fail:

* `strm/ship` and `nopf/ship` at k=2 land inside the `shpb/ship` floor at k=2.
* If either moves beyond that floor at k=2, the floor is understated at every width and no
  k=4/8/16/32 result from this screen is readable. That would be a finding about the instrument,
  not about the prefetch.

## AMENDMENT 2: THE SCOPE LIMIT BELOW WAS WRONG ABOUT NVEC, AND THE REAL HOST DIFFERENCE IS k=1

The original scope limit claimed MKT is AVX-512 with `NVEC = ceil(k/16)`. **There is no AVX-512
code path in this kernel.** `scorch_simd<float>` is unconditionally `using vec = __m256` with
`lanes = 8` (`spmm.h:2155`), with no specialization guarded on `__AVX512F__`; `_mm512` and
`AVX512` appear nowhere under `src/scorch/csrc/` except three gate lines in `scorch_policy.h`,
and the count is computed as `(B1_size + 7) / 8` at `spmm.h:6243`. So **`NVEC = ceil(k/8)` on both
hosts** and widths 2/4/8/16/32 map to NVEC 1,1,1,2,4 on MKT exactly as on redwood. The compiler
may emit zmm for surrounding code; it does not widen this kernel's vector, which is an intrinsic
type fixed at compile time with one specialization.

The hosts do compile different kernels, but at a different place and a narrower one.
`SCORCH_NARROWK_EXACT_UNROLL_W1_HI_MINDEG_F32` is gated on
`defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512F__)` (`scorch_policy.h:1451`), so it is
16 on redwood and 0 on MKT. It is read at `spmm.h:4214` **only when `exact_width == 1`**, and
`exact_width` is `B1_size`, not NVEC -- so that rule fires at **k == 1 alone**. No width in this
screen is 1, so it cannot touch any cell here, and a redwood confirmation measures the same code
path at the widths measured. (`SCORCH_NARROWK_EXACT_K1_MINDEG` is also ISA-conditional, on
`__AVX2__ && __FMA__` rather than on the AVX-512 macro, so it is 8 on both hosts -- and it too is
k=1 only.)

## Scope limits

Single host for now; confirming either clause needs redwood, which is occupied by the c211 board
until ~13:11. float32 only. MKT is shared and not cgroup-partitioned for bandwidth. Four widths
carry the effect and k=2 is the negative control, per Amendment 1.


## AMENDMENT 3: THE WALL-TIME SIGNATURE CLAIM IS WITHDRAWN

Not about this screen, but about the c211 board beside it, and it belongs here because it is the
same instrument. I claimed the deep arm's +14 s per-pass wall-time excess was "the calibration
damping a 5-19% per-call effect down to 1%", and called it an independent signature of the deep
regression. A peer named an identity that makes it falsifiable: the gap should equal the summed
kernel gap in the arms that run the object's own kernel. **It does not.**

Measured on the three landed p1 CSVs, against the +14.0 s matched-position wall gap:

| term | deep − base | A/A (bsb_ − base) |
|---|---|---|
| timed loops, ours+oursb | **+1.9 s** | +0.4 s |
| timed loops, mkl (must be 0) | **+1.2 s** | +0.0 s |
| calibration of ours, 8 unbatched calls/cell | +0.4 s | +0.0 s |
| calibration of mkl | +0.1 s | −0.0 s |
| **accounted** | **+3.6 s of +14.0** | +0.4 s |

So **74% of the gap is in pass overhead the CSVs do not record at all** (per-key `np.load`, tensor
construction, CSV writing: 1352 s of pass against 1220 s of accounted time), and the timed-loop
portion is 0.16%, not a damped 5-19%.

**Why the summed timed loops cannot see a kernel gap, which is the part I had backwards.** The
batch is calibrated per cell to a target, so `batch x ms ~ target_ms` and
`reps x batch x ms ~ reps x target` **independent of how fast the kernel is**. Median `batch x ms`
is 5.665 / 5.666 / 5.641 ms for base / bsb_ / deep against `--target-ms 6`. The quantity is
insensitive to the effect by construction, so it was never going to carry a damped version of it.

**The +14 s is still real** -- matched on position and round, 0 for the byte-identical pair -- but
nothing here attributes it to the kernel, and I withdraw the attribution rather than the
observation.

**One term in that table is a finding on its own.** MKL's timed loops differ by +1.2 s between two
objects that share MKL's code exactly, against a +0.0 s A/A on the same quantity. That is a
cross-object effect on code that did not change, and it is the largest single accounted term.


## AMENDMENT 4: WHAT THE WITHDRAWAL DOES *NOT* TOUCH, measured rather than reasoned

"The summed `ms` cannot see the effect" sits one inch from "the `ms` cannot see the effect", and
they are opposite claims. So this is measured, on the same three landed p1 CSVs.

`ms` is a per-call time; the batch only sets how many calls are averaged into it. The per-cell
ratio is therefore **not** the aggregate a calibrator pins, and the effect is fully present in it:

Bootstrap percentile CIs, 2000 resamples, seed 20261001, n = 3820 per width:

| k | deep/base [95% CI] | A/A bsb_/base [95% CI] | verdict | NVEC / vector |
|---|---|---|---|---|
| 2 | 1.0004 [0.9987, 1.0024] | 1.0017 [1.0004, 1.0028] | **CIs OVERLAP** | *arm unreachable* |
| 4 | 1.0944 [1.0891, 1.0989] | 1.0017 [1.0004, 1.0028] | separated | 1 / `__m128` |
| 8 | **1.1963** [1.1881, 1.2036] | 1.0018 [1.0003, 1.0034] | separated | 1 / `__m256` |
| 16 | 1.0712 [1.0682, 1.0736] | 1.0038 [1.0022, 1.0053] | separated | 2 |
| 32 | 1.0833 [1.0786, 1.0868] | 0.9997 [0.9982, 1.0013] | separated | 4 |

So the per-call effect **is** 5-19% -- 9.4% at k=4, 19.6% at k=8 -- and nothing in the wall-time
withdrawal licenses doubt about the board's per-cell estimator. The withdrawn claim was about a
sum; this is about a ratio.

**THE "x FLOOR" COLUMN THIS REPLACES WAS A CATEGORY ERROR, AND SO IS THE OBVIOUS REPAIR.** The
first version divided `|median(deep/base) - 1|` by `|median(bsb_/base) - 1|` and printed 54.9,
106.6, 19.0, 306.3. The tell is inside the column: k=16 and k=32 have nearly equal numerators
(1.0712, 1.0833) and multiples 16-fold apart, so the swing is entirely denominator -- and it is
worst where a pooled two-sided quantity nearly cancels. 0.9997 is a **centre near zero, not a
spread**, so 306 measures cancellation. A pooled A/A near one understates per-cell scatter, so
every multiple was overstated by an unknown width-dependent factor. Not quotable.

The repair is **not** to divide by the p95 of per-cell `dev(bsb_/base)` either. Measured, that p95
is **0.4987 / 0.6437 / 0.6445 / 0.6514 / 0.6312** by width -- per-cell A/A scatter of +-50-65% --
and dividing the pooled medians by it gives 0.00 / 0.15 / 0.30 / 0.11 / 0.13, i.e. *nothing
clears*. That is the wrong comparison in the opposite direction: **the p95 of per-cell deviations
bounds a SINGLE CELL**, and it is the correct floor for the board's per-cell criterion, which
counts how many of 19100 cells sit beyond it. A **pooled median** needs its own sampling interval,
which is what the bootstrap above gives. Two estimators, two criteria; neither denominator
substitutes for the other.

**k=2 is a NEGATIVE control, and it passes.** Null expected, null observed: that bounds **false
positives** -- the pipeline does not manufacture an effect at a width where the arm provably cannot
fire. The deep arm is bypassed there by the same routing as PFSTREAM (Amendment 1), and k=2 is the
only width of five whose CI overlaps the A/A's. An earlier version of this paragraph called it a
positive control, which it is not: a positive control **injects a known effect** and confirms the
pipeline reports it, bounding false **negatives**.

**So chain211 has a negative control it never had, and still has no positive control.** That gap is
the one that matters for a null result: if a later width or arm reads flat, nothing on this board
yet separates "no effect" from "the instrument cannot see one".

**The cross-object ratio is not batch-matched, which is why the floor must be cross-object.**
Inside one process `ours` and `oursb` share a batch by construction. Across objects each
calibrates its own, so `deep/base` compares two readings taken at different batches:

| k | median batch | batch ratio deep/base | fraction >1.2x apart |
|---|---|---|---|
| 2 | 372 | 0.9911 | 0.382 |
| 4 | 376 | 0.9067 | 0.523 |
| 8 | 383 | **0.8255** | 0.637 |
| 16 | 392 | 0.9366 | 0.448 |
| 32 | 313 | 0.9025 | 0.506 |
| A/A pair, same quantity | | | 0.439 |

At k=8, `1/0.8255 = 1.2114` against a measured `ms` ratio of 1.1963 -- **1.25% apart**
(symmetric `2(a-b)/(a+b)`; the asymmetric readings are 1.2465% on 1.2114 and 1.2622% on 1.1963,
a spread of 0.016 points, so the figure does not depend on the choice), so the batch
tracks `target/ms` and the pinning mechanism is confirmed directly from data. (An earlier version
said "to within 1%", which the two numbers do not support.) The A/A pair's divergence (43.9%) is
comparable to the arms', so **the `bsb_/base` floor contains this term**, which is exactly why it
and not the within-process `oursb/ours` floor is the one an arm must clear.

## AMENDMENT 5: DENOMINATORS FOR AMENDMENT 3

Amendment 3 quoted "the timed portion is 0.16%", which is 1.9/1220 -- the gap against the
*accounted total*, where the claim it refutes is about the timed portion. Corrected:

* ours+oursb timed loops: **+1.9 s of 842.5 s = 0.226%**
* mkl timed loops: **+1.2 s of 370.6 s = 0.324%**, against a +0.0 s A/A on the same base

The conclusion is unchanged -- the sum does not carry a 5-19% effect -- but the ratio as printed
did not support it. And the MKL term is sharper than Amendment 3 put it: it is a deviation in a
quantity the calibration *pins*, so unlike the wall-time gap it cannot be absorbed by unrecorded
overhead. Of the two numbers in that table, the one flagged as a finding is the one whose
instrument is tightest.

## AMENDMENT 6: PFSTREAM IS PROVABLY INERT ON ARM, so "no regressions" is satisfied there by
## construction rather than by measurement

Checked mechanically: one forward pass over `spmm.h` accumulating each line's enclosing
preprocessor stack, with the walk's **own** check that the stack closes at depth 0 at EOF -- without
that, a mismatched `#endif` silently shortens every later stack and the result is spurious in
either direction.

**Counts, each with its predicate, because "12" alone reconstructs to nothing.** The substring
`scorch_spmm_row_regblock` is a **prefix of `scorch_spmm_row_regblock_deep`**, so a substring match
counts calls to two different functions:

| predicate | n | lines |
|---|---|---|
| call expressions matching the substring | **12** | both functions below |
| → calls to `scorch_spmm_row_regblock` | **8** | 4884, 4885, 4968, 4972, 5096, 5097, 6346, 6347 |
| → calls to `scorch_spmm_row_regblock_deep` | **4** | 4948, 4951, 5065, 5066 |
| of the 8, outside `#ifdef SCORCH_TUNE_HOOKS` (release-reachable) | **6** | 4884, 4885, 4968, 4972, 6346, 6347 |
| definitions | 3 | 2267, 2534, 2744 |
| other textual references (comments, macro bodies) | 4 | 2642, 2814, 4923, 4928 |

**Every one of those 23 references is inside `#if defined(__AVX2__) && defined(__FMA__)`, and none
is NEON-guarded.** The 6 is the number that matters for a release object.

**AND THE VECTOR TYPE DOES NOT EXIST ON ARM EITHER, which an earlier version of this paragraph got
backwards.** I wrote that `scorch_simd<float>` has a NEON specialization. It does not:

| template | line | AVX2-guarded | NEON-guarded |
|---|---|---|---|
| `scorch_simd<float>` / `<double>` | 2155 / 2175 | yes | no |
| `scorch_simd_half<float>` / `<double>` | 2202 / 2220 | yes | no |
| `scorch_neon<float>` / `<double>` | 3002 / 3014 | no | yes |

The NEON vectors are a **separate template**. So the register-block kernel's vector type is absent
on ARM and the template could not instantiate even if a call site were reachable.

**That makes the unreachability doubly and independently enforced** -- once by the guard around
every call, once by the guard around the type the kernel is written in. It is a better basis for
"ARM cannot regress" than the call-site walk alone, because a future edit that moved a dispatch out
of the AVX2 block would **fail to compile** rather than silently become reachable.

The belief being corrected is the one that nearly produced the vacuous M5 check: *the kernel is
shared and only the routing differs*. That was never true, and it was not true at the type level
either -- so getting the reason right is what stops the same assumption generating a different
vacuous check later.

Three consequences for the standing "neutral-or-better across the whole space, no regressions"
requirement:

1. **ARM cannot regress.** Not "measured neutral on the M5" -- unreachable. That is the stronger
   form of the gate this project's convention asks for (a condition that *provably cannot fire* on
   the shapes it would hurt), and here it comes free from the existing ISA guard.
2. **Two-host confirmation IS available for this change**, unlike every ARM claim in this
   milestone. redwood is AVX2 and MKT compiles the same AVX2 path (Amendment 2: there is no
   AVX-512 specialization), so both confirmation hosts run the *same* register-block code. The
   usual asymmetry -- one ARM host, so ARM claims never get two hosts -- does not apply.
3. **The M5 is not a third host for this, and no correctness run there would mean anything.** The
   bit-identity check would pass trivially because the arm is absent from the binary's reachable
   code, which is the vacuous-pass shape: a check that cannot fail is not evidence.

The routing map therefore differs by ISA, and the x86 one is the only one that matters here:
k=2 narrow-k exact (arm unreachable), k=4 half-vector regblock (`__m128`), k=8/16/32 full-width
regblock at NVEC 1/2/4.

## AMENDMENT 7: THREE variants, not two -- and "name every kernel this width can dispatch to"

Amendment 6 split the 12 substring matches as 8 + 4. **It is 6 + 4 + 2.** There is a THIRD suffix
variant, `scorch_spmm_row_regblock_pf` (definition `spmm.h:2744`, calls 5096/5097). I published
"match on a word boundary when a `_suffix` variant exists" in the same commit as a count that broke
that rule, because I looked for *the* variant I had just been burned by and there were two.

Enumerated mechanically this time -- the variant set is **discovered** by
`\bscorch_spmm_row_regblock(_[a-z0-9_]+)?\b` rather than assumed:

| variant | calls | definition | comments |
|---|---|---|---|
| `scorch_spmm_row_regblock` | 6 — 4884, 4885, 4968, 4972, 6346, 6347 | 2267 | 2642, 2814, 4923 |
| `..._deep` | 4 — 4948, 4951, 5065, 5066 | 2534 | 4928 |
| `..._pf` | 2 — 5096, 5097 | 2744 | — |
| | **12** | **3** | **4** = 19 total |

**And the release-reachable 6 was right for the wrong reason.** 5096/5097 dropped out of my "8"
because they are hooks-only; they should drop out because they are **a different function**. Two
reasons, one of them wrong, the same answer — which is why the number survived and the derivation
should not have.

### Does the arm exist in every kernel the screen can dispatch to?

The right question is not "is the arm enabled" but **"name every kernel this width can dispatch to,
and does the arm exist in each"**. Tested by reconstructing each **full call expression** (balancing
parentheses forward), because a per-LINE test reported `NO BOUND` for 4968/4972 — whose
`SCORCH_SPMM_PFLIM` sits on continuation lines 4970/4974. That false negative would have read as
"k=4 does not receive the arm", the opposite of the truth.

| call | variant | receives `SCORCH_SPMM_PFLIM` |
|---|---|---|
| 4884, 4885, 4968, 4972, 6346, 6347 | `scorch_spmm_row_regblock` | **yes, all 6** |
| 4948, 4951, 5065, 5066 | `..._deep` | no — separate kernel, default-off |
| 5096, 5097 | `..._pf` | **no, and its signature has no such parameter** |

`..._pf` takes `pA_begin, pA_end, mask_last` and nothing else; `PFLIM`/`pA_limit` appear nowhere in
its body, and its prefetch is row-clipped (`if (pA + PFD < pA_end)`, twice). So **in a hooks build
with `narrowk_pf` set, the arm under test is silently absent**: the clamp is not in the kernel that
runs, the build still compiles, `strm` and `ship` emit the same code on that path, and the result
reads as a plausible null.

That makes `pfs_bits.py`'s `so.scorch_tune_hooks()` refusal load-bearing in a **second and stronger
way** than the per-call getenv tax. The tax distorts a comparison; this removes the arm. Same shape
as the near-miss M5 check — a configuration in which the thing being measured is not in the
reachable code — reached by a different route.

### Screen coverage, and one gap it does not cover

`mrgate` calls `spmm_csr_float_v2` → `spmm_csr_v2_core`. In a release object (no hooks,
`DEEP_UNROLL=0`):

| k | kernel reached | arm present |
|---|---|---|
| 2 | `scorch_spmm_row_narrow_exact` | no — **declared negative control** |
| 4 | `scorch_spmm_row_regblock<…, VH>` via 4968/4972 | yes |
| 8, 16, 32 | `scorch_spmm_row_regblock` via `SCORCH_RB` (4884/4885) | yes |

**THE GAP:** calls 6346/6347 are in `spmm_csr_linear_fused_float` (`spmm.h:6119`), a different entry
point. PFSTREAM changes the **fused Linear-SpMM path too**, and this screen never calls it — so the
screen is silent about a path the change affects. That path carries real workloads (the sparse
autoencoder family). It needs its own measurement before PFSTREAM ships, and "neutral-or-better
everywhere" is not established by this screen alone.

## AMENDMENT 8: COVERAGE IS A QUESTION ABOUT A PRODUCT OF AXES, and the screen covers one cell of three

Amendment 7 enumerated kernels **along width**, holding entry point and dtype at the screen's
values. That finds the fused-path gap only because the fused calls happen to be in a different
function; it cannot see a gap on an axis it never varies. The version that finds all of them:
**name every (entry point, dtype, width) the arm is reachable from, then mark which the screen
visits.**

### The dtype axis, verified from the tree

`scorch_spmm_row_regblock` is templated on `T` and 4884/4885 instantiate it as
`<scalar_t, NV, ...>`. `spmm_csr_double_v2` is an exposed release symbol (`ops.cpp:309`) and its
own comment at `ops.cpp:303` says it goes through "the float32 v2" core -- the same one. The
width bound and the vector count are both **dtype-dependent**, from `spmm.h:3977-3980`:

```
using SV = scorch_simd<scalar_t>;  constexpr int SL = SV::lanes;
const bool narrow_k = (B1_size >= 1 && B1_size <= 4 * SL);
const int nvec = (B1_size + SL - 1) / SL;
```

`scorch_simd<double>` is `__m256d` with `lanes = 4` (`spmm.h:2177`), so:

| dtype | SL | narrow_k holds for | NVEC | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|---|
| float32 | 8 | k ≤ 32 | ceil(k/8) | 1 | 1 | 1 | 2 | 4 |
| float64 | 4 | k ≤ **16** | ceil(k/4) | 1 | 1 | **2** | **4** | *not regblock* |

So PFSTREAM changes the shipped **float64** register kernel at every width up to 16, and this screen
is float32-only by construction. **The f32 grid does not stand in for f64 even where the width
coincides**: f64 k=8 is NVEC 2 where f32 k=8 is NVEC 1, so the (dtype, width, NVEC) triples covered
and uncovered are disjoint in the *middle* of the range, not only at the ends.

The stake is at least as large as the fused one: the f64 register kernel is the one reading
**2.273x geomean against MKL with 0 of 129 cells below parity**. A regression there is not
hypothetical.

### The product, and its actual size

| entry point | dtype | exists? | arm reachable | screen visits |
|---|---|---|---|---|
| `spmm_csr_v2_core` | float32 | yes | yes | **YES** |
| `spmm_csr_v2_core` | float64 | yes (`spmm_csr_double_v2`) | yes, k ≤ 16 | no |
| `spmm_csr_linear_fused_float` | float32 | yes | yes | no |
| fused | float64 | **no such symbol** | — | — |

`grep -c spmm_csr_linear_fused_double` is **0** in both `spmm.h` and `ops.cpp`, so the fourth cell
does not exist and the product is **three reachable cells, of which the screen visits one**. (A peer
put it at a quarter assuming 2x2; it is a third. Checked precisely because that is the direction
that flatters my own coverage.)

**Neither uncovered cell is optional before shipping**, and both carry a live result: the fused path
closed the 0.99-sparsity autoencoder gap against MKL, and f64 is at 2.273x with nothing below
parity.

## AMENDMENT 9: an ETA pooled over builds presumes the null, and the pass wall times are the one
## quantity proved unable to carry the signal

Two estimator notes on the instruments that are *visible* while the real numbers are not.

### The wall times are not an informal preview, and this follows from Amendment 3

A pass's wall time is a mostly-unrecorded quantity whose **measured** portion is constructed to be
insensitive to the effect. On the c211 board only **+3.6 s of a +14.0 s** gap was accountable at
all, with 74% in overhead the CSVs never record, and the accountable part is pinned: `batch x ms`
tracks the calibration target independently of kernel speed. So glancing at the landed durations as
a preview reads the one quantity already *proved* unable to carry the signal. The screen's three
landed durations (840, 841, 839 s — spread ~0.1%) are on screen; the arm's numbers are not. Written
here because availability, not relevance, is what makes a number get read.

### An ETA pooled over builds assumes the arms do not differ

The screen's 840.0 s is a mean over `ship`, `shpb`, `strm` extrapolated to sixteen object-passes of
which four are `nopf` — so it **imputes `nopf` = mean of the other three**, which presumes the null
the screen exists to test. Immaterial in magnitude and free to fix, so: quote it as build-pooled, or
per build.

| estimator | board (7 landed) | screen (3 landed) |
|---|---|---|
| **build-pooled** | 18 × 1357.857 → **13:11:53** | 16 × 840.0 → **12:18:08** (imputes `nopf`) |
| **build-balanced** (each build's own mean × its true share) | 6 × (1351.5 + 1351.5 + 1366.33) → **13:11:28** | *not available — `nopf` unmeasured* |

**On the board the difference has a predictable sign, so it is not merely noise.** `deep` is 3 of 7
landed (0.429) against a true share of 6 of 18 (0.333), and `deep` is the slow build — so the pooled
mean **overstates** and its ETA is late by 25 s. Build-balanced is the correct estimator whenever the
landed composition is unbalanced, which it is at every intermediate point of a rotation.

Screen sensitivity to the imputation, **two-sided because the sign is unknown**: `nopf` at +-2%
moves the ETA +-67 s, at +-5% +-168 s. A first version quoted this one-sided (`+2%`, `+5%`), which
is wrong on a quantity whose direction is the thing being tested: `nopf` has the **fewest**
instructions in the inner loop -- no prefetch at all, `.text` 1488 bytes below `strm` -- so it is
the fastest of the four if instruction count dominates and the slowest if prefetch coverage does.
Which holds is what the screen is for. A one-sided band is blind to exactly one tail, and here that
is the tail where the run finishes **early** and someone looks for output before it exists.

**Henceforth: board 13:11:28 build-balanced; screen 12:18:08 build-pooled with `nopf` imputed.**

## Amendment 10 — the live screen leaves two of the kernel's own branch axes unexercised, and the float64 cell is queued

Writing the float64 width set made me enumerate what the float32 one reaches, and it is less than I
had assumed. With `SL = scorch_simd<float>::lanes == 8`, `nvec = ceil(k/8)` and
`full_last = (k % 8 == 0)` (`spmm.h:3979-3982`), the shipped width set 2,4,8,16,32 reaches:

| k | path | nvec | tail |
|---|---|---|---|
| 2 | `scorch_spmm_row_narrow_exact` — regblock BYPASSED | — | — |
| 4 | regblock at `__m128` (`halfvec_exact`, `LH == 4`) | 1 | full |
| 8 | regblock at `__m256` | 1 | full |
| 16 | regblock | 2 | full |
| 32 | regblock | 4 | full |

So the live screen **never reaches `nvec == 3`** (that needs k ∈ 17..24) and **never reaches a
masked tail** (any k not a multiple of 8). Both are real branches: the call sites instantiate
`scorch_spmm_row_regblock<float, NV, true>` and `<..., false>` on `full_last`, and the `switch
(nvec)` has an arm for 3. Two of the kernel's own template axes are therefore outside the screen's
coverage, and PFSTREAM — which clamps a prefetch run-ahead inside that kernel — is unmeasured on
both.

This is recorded rather than fixed in place. The float32 screen is mid-flight with four passes of a
balanced Latin square; adding widths now would change the design under a running job, and a
balanced rotation does not survive being windowed. The float32 result stands as a result **about
nvec ∈ {1,2,4} with a full tail**, and that qualification travels with every number it produces.

**The float64 cell closes both axes, and that is why its widths are not the float32 set.** With
`SL == 4` (`spmm.h:2178`), `narrow_k` is k ≤ 16 and each width names the branch it exists to reach:

| k | nvec | tail | role |
|---|---|---|---|
| 2 | — | — | **negative control** — `exact_width` routes k ∈ {1,2,3} to `narrow_exact`; `halfvec_exact` would also be true here (`lanes == 2`) and is held off only by `SCORCH_SPMM_HALFVEC_F64` shipping at 0, so two independent mechanisms keep it inert |
| 4 | 1 | full | |
| 6 | 2 | **masked** | |
| 8 | 2 | full | |
| 10 | 3 | **masked** | |
| 12 | 3 | full | |
| 16 | 4 | full | |

k=32 is absent because at float64 it is not the narrow-k path at all, so PFSTREAM is unreachable
there — a coverage statement from `narrow_k = B1_size <= 4*SL`, not a choice. Matching the float32
width *numbers* would have been false comparability: the same k is a different nvec at the two
dtypes, so the covered triples are disjoint in the middle.

**Submitted as job 17681665, held on `--dependency=afterok:17680942`.** afterok rather than afterany
so a failed float32 screen cannot hand this one a build set already in question, and dependency-
gated rather than concurrent because mkt1's memory bandwidth is not cgroup-partitioned: a second
timing job on the same node is a perturbation of the first, not a neighbour.

**Three facts confirmed in a built arm rather than assumed** (`PYTHONPATH=/scr/bobbyy/pfs/strm/src`):
`spmm_csr_double_v2` present, `scorch_tune_hooks()` False (so no getenv tax against MKL), and
`spmm_csr_linear_fused_double` **absent** while `spmm_csr_linear_fused_float` is present. That last
one is the measurement behind amendment 8's claim that the reachable product is three cells and not
four.

**All six release call sites pass `SCORCH_SPMM_PFLIM`, including both inside the fused Linear**
(`spmm.h:6346`, `6347`, inside `spmm_csr_linear_fused_float` defined at 6119). Verified
per-STATEMENT by brace matching, not per line: two sites wrap their argument list, and an earlier
per-line test reported NO BOUND for them — a false negative pointing at absence.

**Provenance closed.** `mrgate.py` and `pfs_screen.sbatch` existed only on the hosts and are now in
the repo, digests matching the producing host: `a55e07bf03e1fb34` for the driver on MKT **and on
redwood** (148 lines each, so the two-host confirmation compares like with like — previously
asserted, now measured) and `23d42aafa6329b84` for the sbatch.

One hazard found while doing that, because it nearly produced a false alarm about the instrument
driving both live runs: **`srun` on `mkt-interactive` defaults to two tasks**, so every observation
command run through it executes twice. Reading the 148-line driver that way produced a 296-line
local copy with a second `#!/usr/bin/env python3` at line 73 — and line 72 sits inside a function
docstring, so it read as a duplicated write swallowed by an unterminated string literal: parses,
runs, plausible. What killed it was digesting on the producing host rather than on the transported
copy. Pass `-n 1`. No published figure was affected, and that was checked rather than assumed: 418
sample keys reconciles as `ceil(4594/11)` and 6270 CSV rows as `418 x 5 widths x 3 arms`.

## Amendment 11 — the two unexercised axes are not symmetric, and amendment 10's digests did not name their function

### Correction to amendment 10's digests

Amendment 10 gave `a55e07bf03e1fb34` and `23d42aafa6329b84`. **A 16-hex digest does not name its
function**, and a verifier had to guess SHA-256-truncated-to-16 and happened to be right. `cksum`
on the same driver is `2321751174`, which is no more self-describing. The full values, with the
function named:

```
sha256 a55e07bf03e1fb34644072da299f4e34645f3d4c1363f45aaa3b4fd382163fc8  mrgate.py            148 lines
sha256 23d42aafa6329b8401f836d477257159c7b49a15af26a2c61a08e195a1490577  pfs_screen.sbatch     49 lines
sha256 509c106a88bd0dd77fc58d01feabd4b125231b96d3fe3c4a00142e3983b4d00c  pfs_screen_f64.sbatch 72 lines
```

And amendment 10's "148 lines each" meant the driver's two host copies, not the two files. The
sbatch is 49 lines. Stated because the ambiguity was real in the sentence as written.

### The asymmetry between the two axes

Amendment 10 recorded that the float32 screen reaches `nvec ∈ {1,2,4}` with a full tail only. The
two gaps are **not closable by the same means**, and that changes what the float64 run is for.

**`full_last` has an existing override.** It is declared non-`const` at `spmm.h:3982`, and
`SCORCH_SPMM_MASKED` at `spmm.h:4296-4297` sets it `false` at any width — its own comment says that
is the only way to time the masked and unmasked instantiations against each other in one process
against one binary. But it sits inside `#if defined(__AVX2__) && defined(__FMA__) &&
defined(SCORCH_TUNE_HOOKS)` (opened at `4287`), so reaching it means a build class `pfs_bits.py`
refuses by design and `pfs_build.sh` never produces. In that class the per-call `getenv` tax is
arm-vs-arm safe and **not** MKL-comparable — it is what manufactured a 1.57× MKL deficit on short
kernels. So the override exists, and using it would cost the MKL comparison that is the whole point.

**`nvec == 3` has no override at all.** `nvec = (B1_size + SL - 1) / SL` is a pure function of the
width, so only a width reaches it.

The consequence: *"both axes are closed by the float64 width set"* is true, and *"both axes could
only be closed by a width set"* is false. One of them could have been closed in a hooks build, at a
price that happens to be the price of this entire screen.

**Both instantiations are emitted at every site, so neither axis is hypothetical.** `FULL_LAST` is
the third template parameter of all three register-block definitions (`2267`, `2534`, `2744`) and
both values are instantiated at every call site; `case 3: SCORCH_RB(3)` exists at `5086`, `5116`,
`5127` and at `6351` inside the fused Linear.

### The float64 control's validity was a derivation, and it holds

`exact_cap_ = exact_f32_ ? 7 : 3` at `spmm.h:4147`, and `narrowk_exact_hi` resolves through
`SCORCH_NARROWK_EXACT_HI_F64` defaulting to `SCORCH_NARROWK_EXACT_HI = 3`. So at float64
`exact_hi_` is 3 and k=2 sits inside the exact band with the register-block arm absent. **Had that
cap been below 2 at float64, k=2 would have been an arm rather than a control** and the design would
have shipped without one. Separately `SCORCH_SPMM_HALFVEC_F64` is 0, so k=4 at float64 is
register-block and not the half-vector path.

### What the float64 set buys beyond coverage

`nvec` 2 and 3 each appear in **both** tail states (k=6/8 and k=10/12), so the run contains a
matched masked-against-full pair at two different vector counts, not one instance of each arm. That
is a within-run contrast on the tail axis rather than a presence check.

### One sharper form of the srun hazard

The doubling is in `srun`'s **stdout**, so an aggregation performed outside the allocation is
doubled while the same aggregation inside it is not: `srun wc -l f` prints the right number twice,
`srun cat f | wc -l` prints twice the number once. That makes the landed-pass count the exposed
quantity, since it is the only authoritative progress counter. **It was not doubled**, and the check
is arithmetic rather than reassurance: at the 8-pass checkpoint, `09:25:31 − 06:24:32 = 10859 s`
against `3(1352.5) + 2(1351.5) + 3(1366.3) = 10859.4 s`. A doubled count would have missed by a
factor near two, not by 0.4 s.

## Amendment 12 — the mechanism's reach is `2/d`, and that is a falsifiable prediction registered before any arm is read

**Registered at 10:40 on 2026-10-01, with the screen at 8 of 16 object-passes and no arm read.**

`SCORCH_SPMM_REGBLOCK_PFD` ships at **2** (`scorch_policy.h:705`), and the two forms of the
run-ahead differ exactly as follows:

| form | site | fires when | prefetches in a row of length L |
|---|---|---|---|
| shipped, row-clipped | `spmm.h:2305` | `pA + 2 < pA_end` | `max(0, L − 2)` |
| PFSTREAM, array-clamped | `spmm.h:2302` | always (the clamp guarantees a target) | `L` |

So the **newly covered fraction of nonzeros is `Σ_rows min(L_i, 2) / nnz`**, which is `≈ 2/d` at mean
degree `d`. This is not an estimate; it is the difference of the two expressions, and the measured
Spearman correlation against `1/d` over the cached SuiteSparse matrices is **1.0000** — as it must
be, because the quantity *is* `2/d`.

**Measured reach, two corpora.** SuiteSparse first, and the sample limitation stated: 16 matrices
cached on the local machine, assembled for earlier panels, a convenience sample and not a draw.

| | newly covered |
|---|---|
| SuiteSparse (16 cached) | min 0.0%, **median 0.9%**, max 38.0% |
| — matrices at ≤ 5% | **9 of 16** |
| — matrices at ≥ 50% | 0 of 16 |

DLMC, read from `~/dlmc` directly rather than from the cache, because the cache stops at pruning
0.9 and the fused path's shipped result lives at 0.95–0.99. 3846 `.smtx` files, up to 30 sampled
per (method, level):

| pruning level | mean degree | newly covered |
|---|---|---|
| 0.5 | 256–286 | 0.7–0.8% |
| 0.8 | 102–104 | 1.9–2.0% |
| 0.9 | 49–63 | 2.8–4.0% |
| 0.95 | 21–28 | 4.5–9.8% |
| 0.98 | 8.6–12.2 | 9.6–21.3% |

**THE PREDICTION, pre-registered.** The screen's per-matrix effect must be **monotone increasing in
that matrix's newly-covered fraction**, and must be **indistinguishable from the floor where the
fraction is near zero**. The screen CSV already records `deg`, so the test needs no new measurement:
compute `2/d` per row and regress the per-cell `strm/ship` log-ratio on it.

**Why this matters more than the earlier prediction.** Amendment 0's conjunction was "beats `nopf`
where B is cache-resident AND beats `ship` on the 115 matrices that need the prefetch". That has no
degree term at all. But in **both** corpora the mechanism touches a *single-digit percentage* of
prefetches at the median. So a large pooled effect would be **disproportionate to its own
mechanism** — and a disproportionate effect is the signature of a layout artefact (two objects with
identical decisions differ 1–2.6% on narrow-k SpMM, largest at the narrowest width) rather than of
the change under test.

**The falsification is therefore sharper than "no effect".** If the effect is large but
*uncorrelated* with `2/d`, the mechanism story is wrong even though the measurement is right — and
PFSTREAM must not ship on it, because what would be shipping is an unexplained difference between
two builds. Stating this now, before any arm is read, is the whole point of registering it.

**One corollary for `nopf`.** Deleting the prefetch entirely was measured at 2.9–14.2% faster while
harming 115 matrices at width 32. Since the prefetch covers `(L−2)/L` of a row, the *cost* side also
scales with degree but in the opposite direction — a high-degree matrix pays the full guard cost on
almost every nonzero and PFSTREAM changes almost nothing there. So `ship`-versus-`nopf` should be
**largest** where `strm`-versus-`ship` is **smallest**. That is a second, independent prediction
from the same arithmetic, and the two must not both be flat.

## Amendment 13 — the windowing rule named the wrong pair, and the bias halves rather than vanishing

Amendment 4's windowing decision said: at three complete rounds, "report the ship-vs-shpb position
effect beside every number". **That names the wrong pair at every window size**, and the position
bias at three rounds is larger than the phrasing implies. From the four landed orders
(`ship shpb strm nopf` / `shpb strm nopf ship` / `strm nopf ship shpb` / `nopf ship shpb strm`),
each build's mean position:

| after | ship | shpb | strm | nopf | spread | worst pair | ship–shpb |
|---|---|---|---|---|---|---|---|
| 2 rounds | 2.500 | 1.500 | 2.500 | 3.500 | **2.000** | nopf–shpb at 2.000 | 1.000 |
| 3 rounds | 2.667 | 2.333 | 2.000 | 3.000 | **1.000** | nopf–strm at 1.000 | 0.333 |
| 4 rounds | 2.500 | 2.500 | 2.500 | 2.500 | **0** | — | 0 |

Three consequences, and the first two are corrections rather than refinements:

1. **The maximally confounded pair is a function of the window.** `nopf–shpb` at two rounds,
   `nopf–strm` at three. `ship–shpb` is 1.000 and 0.333 — so the standing rule pointed at a pair
   carrying half, then a sixth, of the actual confound. The correct statement is *the spread, plus
   which pair carries it*, recomputed for the window actually analysed.
2. **At three rounds the bias halves; it does not nearly vanish.** A per-position drift of `d`
   seconds still buys one full position-unit of build contrast, so a three-round read carries the
   position term with coefficient 1.0 rather than as a caveat.
3. **At two rounds the estimator carries no progress information**, same property as the board at
   3/3/3: with balanced counts the total is `4 × (sum of the four build means)`. The landed sum
   being 6716 s and the remaining estimate also being 6716 s is that identity, not a coincidence,
   and nothing should be read into the symmetry.

This amends amendment 4 and does not change the gate: four complete rounds, which is the only
window where the spread is zero.

## Amendment 14 — amendment 12's validation was a tautology, and its covariate is a proxy that always overstates

**Two hours after registering amendment 12 and still with no arm read.** The derivation in it is
exact and unchanged. Two things around it are not.

### The retraction

Amendment 12 wrote: *"the measured Spearman correlation against `1/d` over the cached SuiteSparse
matrices is 1.0000 — as it must be, because the quantity *is* `2/d`."* That reads as a sanity check
and **it is a tautology that measures nothing**: `2/d` is a monotone function of `1/d`, so their
rank correlation can only be 1. I had computed the correlation of the **approximation** with
`1/d`, not of the **statistic** with anything. This is the same family as a table that can only
print 0 — a check whose output is fixed by construction.

The real statistic, measured over all 3820 nonzero DLMC matrices (and 26 excluded for `nnz == 0`,
which have **zero in this statistic's denominator** — a guard, not a discovery):

```
Spearman( TRUE reach = Σ min(L,2)/nnz , 1/d )  = 0.992354    the real check
Spearman( 2/d                         , 1/d )  = 1.000000    1 by construction
Pearson ( TRUE , 2/d )                         = 0.304235
```

### `2/d` is a proxy, it always overstates, and it leaves the unit interval

`Σ min(L_i, 2) = 2·rows` only when **every** row has degree ≥ 2, so `Σ min(L,2)/nnz ≤ 2/d` always:

| | |
|---|---|
| max TRUE/PROXY ratio | **1.000000** — the bound, so the proxy never understates |
| p5 of TRUE/PROXY | 0.6884 — on 5% of matrices the proxy overstates by ≥ 45% |
| differ by > 1% | 653 of 3820 |
| rows of degree 0 or 1 present | 956 of 3820 |
| `2/d > 1` | **55 of 3820**, max **17067%** |

Most misleading case, ranked by `proxy/true` because that is the shape that manufactures a
violation: `body_encoder_layer_2_ffn_conv1.smtx`, 2048 rows, 159 nonzeros — **true reach 3.77%,
proxy 2576%, overstated 683×**. A covariate an effect is about to be regressed against must not be
able to take the value 2576% when the thing it stands for is a fraction of prefetches.

### What this does to the registered test

Amendment 12 said *"the screen CSV already records `deg`, so the test needs no new measurement."*
**That is false.** From `deg` alone only `2/d` is constructible, and the registered prediction is
monotonicity in the newly-covered **fraction**, which needs `Σ min(L_i, 2)` — a property of the
row-length distribution, not recoverable from the mean. So:

1. **The true reach must be measured per matrix for the screen's own 418 keys before the
   regression is read.** It needs only each matrix's `indptr`, so it is a cheap read — deferred
   until the node is free rather than run beside a live timing job.
2. **If `2/d` is used at all it is a DECLARED PROXY** with the error above stated beside it, not a
   stand-in presented as the statistic.

### The direction of the bias, registered now because it decides how to read the outcome

The proxy **always overstates**. So a low-true-reach matrix can be assigned a high proxy reach, and
its genuinely small effect then reads as a monotonicity **violation**. **The bias works against the
hypothesis**: a confirmation under the proxy is stronger than it looks, and a rejection under the
proxy may be the proxy rather than the mechanism. Amendment 12 exists precisely to separate "the
mechanism story is wrong" from "the measurement is right but unexplained", and without this
asymmetry on the record a proxy-driven rejection would be read as the former.

### The screen's corpus cannot provide the strongest test, and the fused cell's can

On DLMC the true reach is ≥ 5% on 1080 of 3820, ≥ 25% on 156, ≥ 50% on 55, ≥ 90% on 6, with a
maximum of **exactly 100.0000%** — a matrix where every row has degree ≤ 2, so the row clamp issues
**no useful prefetch at all** and PFSTREAM covers every one. Over the 16 cached SuiteSparse
matrices, **none reaches 50%** (median 0.9%).

So the high-reach end of the covariate, where the prediction is most sharply testable, lives in the
corpus the **fused** cell will run on and is absent from the one the **screen** is running on. That
is not a defect in the screen; it is a statement about which corpus can test the mechanism, and it
means a flat screen result is weaker evidence against the mechanism than a flat DLMC result would
be. The 418-key SuiteSparse sample's own reach distribution is unmeasured and is item 1 above.

## Amendment 15 — the dark-row rate is parametric in B, and the faithful B is the one that darkens the check

Registered 2026-10-01, before any fused timing number exists. Measured by the peer session on the
512x512 DLMC matrix with signed `A` values, over its 274 non-empty rows: the rate at which a whole
row goes dark under relu, against `0.5^k`, which is what independent per-output signs would give.

| B | k=2 | k=8 | k=32 |
|---|-----|-----|------|
| uniform[0,1) — what `pfs_bits_fused.py` uses | 0.4562 | 0.3358 (86x `0.5^8` = 0.0039) | 0.2920 |
| standard normal (signed) | 0.2153 | 0.0000 | 0.0000 |

The ELEMENTWISE negative rate is 0.49–0.52 in every case, so the entire excess is within-row
correlation. **It collapses to zero with a signed B**, which locates the cause in B's
non-negativity and not in the matrix: with `B >= 0` a row's `k` outputs are all non-negatively
weighted sums of the same `A[i,:]`, so a row whose values lean negative goes dark across the whole
width. My earlier claim that the dark-row rate is "structural, not a parameter" is correct about
`--bias-scale` (it moves nemeth24 from 49.5% to 49.2%) and **wrong as a general statement**: the
rate is parametric in B, and B is a harness choice that this pre-registration had not declared.

**The declaration, and it is a trade rather than a knob.** A non-negative B is the FAITHFUL choice
for this kernel, because the fused Linear's input in production is post-activation values from a
preceding relu. So the modelling decision that makes the benchmark realistic is the same decision
that makes ~37% of its output unverifiable by a tolerance check. The harness keeps the faithful B
and closes the verification gap on the other axis — the dark-row gate asks whether a `(problem,
width)` was verified by SOME activation, and identity is strong exactly where relu is weak.

**The wrong fix is named here so it is not available later.** On seeing a dark-row refusal the
instinct is to switch B to a signed distribution. That makes the check pass and the workload wrong,
and nothing downstream would record that the inputs no longer resemble a layer's input. If a future
run needs a signed B it must be declared as a separate arm with its own label, never as a repair to
a refusal. `pfs_bits_fused.py` therefore does not expose a B-distribution flag.

## Amendment 16 — the worst pair must be recomputed for the window being read, not fixed in advance

Registered 2026-10-01 at 11:32, with the screen at three complete rounds (12 of 16) and no arm
read. This closes a defect in amendment 13 that is amendment 13's own defect one level down.

**Amendment 13 retracted a pair that had been named in advance.** The standing windowing rule
named `ship–shpb` as the pair carrying the position spread; measurement showed the spread is
2.000 at two rounds (worst pair `nopf–shpb`), 1.000 at three (worst pair `nopf–strm`), and 0 at
four, while `ship–shpb` reads 1.000 then 0.333 — **the worst pair at neither window size**. So
the rule named the wrong pair twice.

**What amendment 13 did not say is that the replacement must be recomputed per window**, and
without that it repeats the error with a different constant. The worst pair is a property of
the landed passes in the window being analysed: it changed between two rounds and three, so
there is no reason to expect the pair at three to be the pair at any other count, and naming
`nopf–strm` in advance would be the same mistake with a fresher number.

**Registered condition.** Any read of a partial window must (a) compute the mean-position
spread for **every** build pair from that window's own landed passes, (b) report the maximum
and name the pair achieving it, and (c) state the position term's coefficient. A pair named
from a different window is not evidence about this one. At four complete rounds the spread is
0 for every pair and this condition is satisfied trivially — which is the reason the gate is
four and not three, stated here as the positive form of the same fact.

**This is not a licence to read at three.** The gate remains four complete rounds. The
condition exists so that a forced partial read (a preempted job, a co-tenant arriving
mid-round) carries the right covariate rather than an inherited one.

## Amendment 17 — the registered negative control FAILED, and the screen's floor is the wrong floor

Read 2026-10-01 at 12:20, after all sixteen passes landed (job 17680942 COMPLETED, 03:44:00,
exit 0:0, last pass closing 12:18:07). Written before any k≥4 number was interpreted.

### The window qualifies, checked first per amendment 16

The pass order read out of `screen.log` is the left-cyclic rotation of (ship, shpb, strm, nopf),
each build occupying each position exactly once. Mean position 2.5000 for all four, and the
spread for **every** one of the six pairs is **0.0000** — so the position term's coefficient is
0 and no pair carries it. That is the four-pass branch of the windowing decision, satisfied by
measurement rather than by assumption.

### Bit-identity holds, and the check is not vacuous

All four builds agree on all **2120** cells (424 keys × 5 widths; 2090 real, 30 degenerate), and
**2120 of 2120 digests are distinct within one build** — so the agreement is not the artefact of
every cell sharing one hash. `pfs_bitcmp.py` **refused**: the producer wrote no `.done` sentinel,
so this is a hand comparison carrying no guarantee that each run closed its file. The row count
being exactly 2120 everywhere bounds that risk to "died after writing every row", which changes
nothing in the comparison, but it does not remove it. **A sentinel is owed by the f64 and fused
producers.**

### The control, and it failed

Amendment 1 registered: `strm/ship` and `nopf/ship` at k=2 must land inside the `shpb/ship`
floor at k=2, and if either moves beyond it, *"the floor is understated at every width and no
k=4/8/16/32 result from this screen is readable"*. Pooled medians over 418 matrices, bootstrap
percentile CIs, 2000 resamples, index matrix sha256 `fe90d5e14bc94d99a833a1464ee98603`:

| ratio at k=2 | pooled median | 95% CI | vs the floor's CI |
|---|---|---|---|
| `strm/ship` | **1.006837** | [1.004488, 1.008271] | **separated** |
| `nopf/ship` | **1.017599** | [1.015588, 1.019778] | **separated** |
| `shpb/ship` (floor) | 0.999359 | [0.998729, 0.999935] | — |

Both arms are slower than `ship` by 0.68% and 1.76% at a width where their differing code
**cannot execute**. The declared floor sits at −0.064%. So the registered consequence applies:
**no k≥4 result from this screen is readable as the arm effect.**

### The control's premise was re-verified, and it holds

Before accepting the failure the premise was re-checked against the source with brace-matched
function spans rather than nearest-preceding-definition (the first attempt mis-attributed a
file-scope macro block to `scorch_spmm_row_narrow_exact` and would have invalidated the control
for the wrong reason). Both macros are confined to `scorch_spmm_row_regblock` — its signature
(`spmm.h:2270`), its body (`2291`–`2295`) and its call sites (`4884`, `4885`, `4970`, `4974`,
`6346`, `6347`). **Nothing in `scorch_spmm_row_narrow_exact`**, which owns float32 k ∈ {1,2,3}.
The control is valid; the failure is real.

### The mechanism, measured

`.text`, read from the four objects:

| build | `.text` bytes | vs ship | `.text` sha |
|---|---|---|---|
| `ship` | 1073586 | — | `d726054c522aaca2` |
| `shpb` | 1073586 | 0 | `d726054c522aaca2` — **identical, as designed** |
| `strm` | 1071954 | −1632 (−0.152%) | `fec561b30165be9e` |
| `nopf` | 1070466 | −3120 (−0.291%) | `238988e56786106e` |

**The floor bounds the wrong thing.** `shpb` differs from `ship` only in build path, with
byte-identical `.text`, so `shpb/ship` measures build-*path* variation. `strm` and `nopf` are
different code. PFSTREAM adds a parameter to a kernel this file's own comment calls "register-starved
enough that one more live value moves code" (`spmm.h:2256-2258`, where the unconditional form cost
67 instructions and 480 bytes). A design whose arms are compile-time macros that change emitted
code needs a floor that **also** differs in emitted code, and this one does not have it.

The displacement is **4–6× the relative code-size change**, which is what a layout effect looks
like: alignment, instruction-cache set conflicts and branch-predictor aliasing are not
proportional to size.

### Two properties that distinguish this from noise

**It is one-signed.** At k=2, `nopf` is slower than `ship` on **359 of 418** cells (85.9%) and
`strm` on **280 of 418** (67.0%), against the floor's **187 of 418** (44.7%). A two-sided
placement wobble would sit near 50%.

**Its size profile matches the known A/A inverted U.** The k=2 displacement peaks in the
1e4–1e5 nonzero band — `strm` +1.766%, `nopf` +2.524% — and falls away at both ends
(0–1e4: −0.113% / +0.691%; 1e6–1e7: −0.327% / +0.663%). That is the shape full-collection A/A
noise already has, which is evidence for a placement mechanism and against a kernel effect.

### What is NOT reported

The k=4/8/16/32 ratios exist and are not the arm result. The largest of them is `strm/ship` =
1.04717 at k=16, and **"PFSTREAM is 4.7% slower at k=16" is not a claim this screen supports**:
the same object carries a +0.68% displacement at a width where its difference cannot run, and
the displacement is not constant in width (`nopf` swings from +1.760% at k=2 to −1.700% at k=8).
Object identity and arm identity are confounded at every width, and nothing in this design
separates them.

### What the next design needs

The floor must be an estimate of **inter-object** variance, not of build-path variance. Two
routes, neither chosen here:

* **≥2 objects per arm**, so the variance between objects of the same configuration is measured
  rather than assumed. This is the honest form and it doubles the build count, not the run time.
* **A floor arm that differs in emitted code by a comparable amount while being semantically
  inert at every width.** Cheaper, and weaker: "comparable" is a judgement, and an inert change
  that happens to land in a different cache set proves nothing about one that does not.

**The f64 cell now carries an independent replication of this question** rather than only a
coverage role. Job 17681665 started 12:19:07 with the same four-build design, and at float64
k=2 the row still takes `narrow_exact` (`exact_hi_` is 3 at f64, and `SCORCH_SPMM_HALFVEC_F64`
ships at 0, so `halfvec_exact` is false). If its k=2 control fails too, the defect is a property
of the build-arm design; if it passes, something specific to these three f32 objects. Either
outcome is informative and neither is the prefetch question.

### The amendment-12 regression stays blocked

Unchanged and for the registered reason: the true reach `Σ min(L_i,2)/nnz` must be measured for
the screen's own 418 keys before the mechanism regression is read, and that read is 418 `.npz`
decompressions on a host now running job 17681665. One bounded 7.3 MB copy of the results was
taken off the live node and is stated here rather than treated as free; a 418-matrix read is not
in the same class and waits.

## Amendment 18 — five corrections to amendment 17, and a criterion registered before the data is touched

Written 2026-10-01 at 12:35. The band-profile comparison in the last section of this amendment
is **registered and not yet computed**; the criterion and its positive control are fixed here
first, because the peer session left the threshold to me and choosing it after seeing the numbers
is the exact failure this file exists to prevent.

### Correction 1 — the repair is RULED OUT, not merely confounded

Amendment 17 said object identity and arm identity are confounded at every width. That is true
and too weak, and it leaves the obvious repair available: treat the k=2 displacement as a
per-object offset and subtract it from the k=16 ratio. **That repair is refuted by data already
in amendment 17.** It assumes the object displacement is constant in width, and `nopf` goes
**+1.760% at k=2 to −1.700% at k=8 — a sign change**. There is no width-invariant object offset
to subtract, so `strm/ship` = 1.04717 at k=16 **cannot be decomposed**, by measurement rather
than by caution.

### Correction 2 — the control's per-cell inertness rests on a default nothing here pinned

`spmm.h:4183-4185` disables the exact kernel per matrix:

```
if (exact_width && narrowk_exact_mindeg > 0 &&
    nnz_total < narrowk_exact_mindeg * (long)A0_size) exact_width = 0;
```

Had `SCORCH_NARROWK_EXACT_MINDEG` shipped nonzero, every matrix below that mean degree would
have escaped `narrow_exact` at k=2 and fallen through to a branch that **does** reach
`regblock` — so the arm would have been live on a low-degree subset of the control population.
It ships at `0L` (`scorch_policy.h:562`) so the branch cannot fire.

**This matters for the one-signed statistic specifically.** "The control is inert" and "the
control is inert on every matrix in it" are different claims, and a per-cell sign test over 418
matrices needs the second. It holds — but because of a default, not because of the kernel's
structure, and amendment 1 did not pin it. Pinned now, with the defaults read at this tip:
`NARROWK_EXACT_HI` 3, `NARROWK_EXACT_MINDEG` 0L, `SPMM_HALFVEC_F32` 1, `SPMM_HALFVEC_F64` 0,
`REGBLOCK_PFSTREAM` 0, `REGBLOCK_PFD` 2. The source comment at 4178-4182 says the constant is
host-conditional and "stays 0", so this is a deliberate default rather than an accident — but a
future run that changes it invalidates the per-cell test and not just the pooled one.

### Correction 3 — k=4 is an ARM, and the plausible reason it is not a second control

`NARROWK_EXACT_HI` is 3, so at f32 k=4 the row leaves `narrow_exact` and takes the half-vector
branch. That branch is **not a different kernel**: `spmm.h:4967-4970` calls
`scorch_spmm_row_regblock<scalar_t, 1, true, VH>` — the same kernel at the half-vector type —
**and passes `SCORCH_SPMM_PFLIM(nnz_total)`**. `halfvec_exact` is `(B1_size == LH)` = (4 == 4)
with `SPMM_HALFVEC_F32` at 1. So PFSTREAM reaches k=4 through the half-vector path.

Registered so nobody later reasons the other way: **"k=4 takes the half-vector path" is true and
"so the register-block arm is inert there" does not follow.** There is no second negative control
in this design.

### Correction 4 — the row-count guarantee is narrower than amendment 17 stated

Amendment 17 said the row count being exactly 2120 bounds the missing-sentinel risk to "died
after writing every row". That holds **only because the comparison is over digests of complete
rows**, so a truncated final row would have changed the count. It does **not** extend to any
statistic computed from partial rows, and it is a strictly narrower guarantee than a sentinel.
The f64 and fused producers owe the sentinel.

### Correction 5 — the band profile does no work where amendment 17 used it

At k=2 the arm provably cannot execute, so the whole k=2 displacement is floor **by
construction**. Inertness already settles that, and no band profile is needed to establish it.
Amendment 17 offered the size profile as evidence for "placement rather than a kernel effect" at
k=2, which is a question the k=2 data neither answers nor needs to.

### THE REGISTERED COMPARISON, fixed before the data is touched

The informative question is the one amendment 17 did not ask: **does the k≥4 displacement share
the k=2 band profile?** Matching shape and magnitude is evidence the k≥4 displacement is floor
too; divergence would be the first thing in this design pointing at an arm effect.

**Statistic.** Per-band median of `num/ship − 1` for `num` in {`strm`, `nopf`, `shpb`} over the
nonzero bands [0,1e4), [1e4,1e5), [1e5,1e6), [1e6,1e7) with n = 129, 181, 67, 36. The
≥1e7 band is **excluded**: n = 5, below the eight-cell minimum already used in this file.

**Interval.** Each band's median gets its own bootstrap percentile CI, 2000 resamples drawn
*within* the band, from a per-band index matrix whose sha256 is reported. Per-width digests must
differ from each other — a one-line control on the seal that this file has failed before.

**Criterion, as a containment test rather than an overlap test.**

* **"Shares the k=2 profile"** = for all four bands, the k≥4 point estimate lies inside the
  k=2 band's 95% CI.
* **"Diverges"** = at least one band where the k≥4 CI and the k=2 CI are **disjoint**.
* Anything else is **"indeterminate"**, and indeterminate is a reportable outcome, not a
  licence to pick the nearer label.

**Positive control, because a "matches" verdict is unfalsifiable without one.** Inject a
multiplicative +2.0% into one build's k≥4 per-cell ratios — comparable to the largest observed
k=2 displacement — and require the test to return **"diverges"**. If it cannot, the test is
underpowered at the magnitude in question and **no "shares the profile" verdict from it is
readable**. The injected effect is applied to a copy and the uninjected run is reported beside
it, so the control reports a magnitude rather than a verdict.

**What neither outcome licenses.** "Shares the profile" does not make the k≥4 ratios readable as
the arm effect — it strengthens the floor interpretation of a quantity that remains confounded,
because correction 1 shows there is no offset to subtract. "Diverges" does not establish an arm
effect either; it identifies a width where object and arm displacement differ in a way the floor
alone does not explain, which is a reason to build the next design's floor, not a result.

## Amendment 19 — the registered comparison is INVALID, because the floor fails it too

Computed 2026-10-01 at 12:40, criterion and controls fixed at `c19277d` beforehand. The outcome
is that **the test cannot discriminate what it was registered to discriminate**, and the evidence
is in the test's own output.

### The verdicts, reported in full including the indeterminate ones

Per-band index matrix digests, required to differ and they do: `927f8b94ef5d5d25`,
`c629d7ba634f1818`, `03fa42a6f80d81dc`, `6319e2d983008d22`.

| build | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|
| `strm` | DIVERGES (0/4 in) | DIVERGES (0/4) | DIVERGES (0/4) | DIVERGES (0/4) |
| `nopf` | DIVERGES (0/4) | DIVERGES (0/4) | DIVERGES (0/4) | DIVERGES (0/4) |
| **`shpb` — the floor** | **INDETERMINATE (2/4)** | **DIVERGES (2/4)** | **INDETERMINATE (1/4)** | **DIVERGES (1/4)** |

**`shpb` is two objects with byte-identical `.text`. There is no arm in it at any width.** It
diverges from its own k=2 band profile at k=8 and k=32, and is indeterminate at k=4 and k=16.
A test its own no-effect case fails is not a test, so `strm` and `nopf` diverging says nothing
about either arm.

### The positive control passed, which makes the failure diagnostic rather than fatal-and-vague

Injecting +2.0% returned DIVERGES in **12 of 12** cells. So the test is **not underpowered — it
is measuring the wrong quantity.** That distinction is the useful part: a passing positive
control beside a failing negative control says *misdirected*, where two failing controls would
only have said *insensitive*.

### What it actually measures, and the floor number that shows it

Comparing a k≥4 band profile against a k=2 band profile tests whether the band profile is
**width-stable**. The floor's is not. `shpb/ship` per-band medians, in percent — a property of
the instrument and not a reading of either arm:

| k | 0–1e4 | 1e4–1e5 | 1e5–1e6 | 1e6–1e7 |
|---|---|---|---|---|
| 2 | +0.257 | −0.538 | −0.064 | −0.170 |
| 4 | +0.312 | −0.763 | −0.482 | −0.445 |
| 8 | +0.270 | **−1.489** | −0.800 | −0.370 |
| 16 | +0.211 | −0.744 | −0.434 | +0.329 |
| 32 | +0.190 | −0.889 | −0.629 | +0.037 |

Largest absolute floor band displacement **1.489% at k=8, band 1e4–1e5**. Width-to-width swing
*within* a band, on byte-identical code, **in band order to match the table above**: 0.122 pp
(0–1e4), **0.951 pp** (1e4–1e5), 0.737 pp (1e5–1e6), 0.775 pp (1e6–1e7). A criterion requiring
the k≥4 profile to sit inside the k=2 profile demands a stability the floor does not have.

*(Corrected by amendment 21: this list was first published sorted descending directly beneath a
band-ordered table, so it read as band-aligned and was not — a reader mapped 0.951 pp to the
0–1e4 band. The values were right and their order was a presentation defect.)*

### My error, and it is a registration error rather than an analysis one

**I registered the positive control and not the negative one.** Amendment 18 fixed the criterion,
the statistic, the intervals and an injected-effect control — and never asked whether the floor
arm passes the criterion when no effect is present. That check needed no new data and no new run:
`shpb`'s width profile was already in the same CSVs. The house rule is that a guard must fire on
its own input and a control must sit inside the population it certifies; the corresponding rule
for a *criterion* is that **both** controls belong in the registration, because a criterion the
no-effect case fails produces a confident verdict on every arm.

Registering the criterion first was still right — it is why this is reportable as a method
failure instead of being quietly reshaped around the numbers.

### Registered for the f64 replication, not applied here

The floor's displacement is bounded in this run: **no floor band at any width exceeds 1.489% in
absolute value.** That suggests a magnitude criterion rather than a shape one, and it is
registered for job 17681665 — whose data does not exist yet — rather than applied to data already
seen:

* **Statistic**: per-band median `|num/ship − 1|`, same bands, same within-band bootstrap form.
* **Floor bound**: the maximum over all widths and bands of `|shpb/ship − 1|`, computed from the
  f64 run's own `shpb`. Never imported from this run — a floor from another run is another run's
  artefact range.
* **Criterion**: a band-width cell counts as beyond floor only if its 95% CI lies entirely
  outside ±(that run's floor bound). Containment, not overlap.
* **Both controls registered this time.** Positive: an injected effect at the floor bound's
  magnitude must be detected. **Negative: `shpb` itself must produce zero beyond-floor cells.**
  If `shpb` produces any, the criterion is invalid for that run and no arm verdict from it is
  readable — stated now so the outcome cannot be re-labelled later.
* The f64 k=2 control remains the gate, and it is checked before any of the above.

**Not applied to the f32 screen, and the reason is not caution.** Correction 1 of amendment 18
shows there is no width-invariant object offset to remove, so even a magnitude comparison on
this run would compare an arm-plus-artefact quantity against a floor measured on objects that
differ from the arms in code. The f32 screen's k≥4 numbers stay unreadable.

## Amendment 20 — three gaps in amendment 18's registration, one of which overstated amendment 19

Raised by the peer session at 12:36, before they had seen amendment 19. Two of the three apply to
what was actually run, and one of those weakens a claim amendment 19 made.

### Gap 1 — the criterion did not say POOLED or PER WIDTH, and the implementation chose silently

Amendment 18 says "the k≥4 point estimate", singular, throughout, and never states whether the
four widths are pooled or tested separately. **The implementation tested per width** (a loop over
k ∈ {4,8,16,32}), so every verdict in amendment 19 is a per-width verdict — but that was an
implementation choice resolving a registration ambiguity, and it went unrecorded until now. A
registration that leaves the analysis unit to the code has not fixed the analysis.

**Registered: per width.** The reason is the peer's and it is directional, not about power.
Correction 1 of amendment 18 established that `nopf` changes sign between k=2 (+1.760%) and k=8
(−1.700%), so the four widths are not estimating a common quantity. A pooled k≥4 estimate
averages opposite-signed displacements toward zero, and "shares the k=2 profile" is the
containment outcome — **so pooling is biased toward "shares" by exactly the mechanism correction 1
established.** Per-width testing carries no such bias and multiplies the comparisons by four,
which is a cost in multiplicity rather than a bias.

### Gap 2 — the positive control was a GLOBAL verdict where the criterion's weakest link is PER BAND

Amendment 18 required an injected +2.0% to return "diverges", and "diverges" is satisfied by any
single disjoint band. So detection in the n=181 band established nothing about the n=36 band —
while "shares" requires **all four** bands to contain, so a band too wide to see a 2% effect
contributes a free pass toward the floor reading. The asymmetry runs against the conclusion
amendment 19 drew.

**Measured per band rather than argued.** Of 48 band-cells (3 builds × 4 widths × 4 bands),
**42 detect an injected +2.0% and 6 do not**:

| band | n | failed to detect +2% |
|---|---|---|
| 0–1e4 | 129 | 2 of 12 |
| 1e4–1e5 | 181 | 1 of 12 |
| 1e5–1e6 | 67 | 0 of 12 |
| 1e6–1e7 | 36 | 3 of 12 |

**So amendment 19's "the test is NOT underpowered — it is measuring the wrong quantity" was
overstated.** The honest form is that it is **both**: misdirected, and partly underpowered at
2.0% in 6 of 48 band-cells. "12 of 12" was a global verdict and is withdrawn as a power claim.

**Amendment 19's conclusion survives the correction, and here is why it is not special pleading.**
The floor arm `shpb` detects the injected effect in **4 of 4 bands at every width**. Its
divergences — bands 1 and 2 at k=8, band 1 at k=32 — therefore sit in bands with *confirmed*
power at this magnitude, so "the no-effect case fails the criterion" is not an artefact of a wide
interval. The invalidation stands; its stated justification is replaced.

### Gap 2b — the peer's n-based scaling prediction is wrong in its ORDERING

They predicted CI half-widths scaling as `1.253·σ/√n` relative to the tightest band: n=181 at
1.00×, 129 at 1.18×, 67 at 1.64×, 36 at 2.24×. Measured half-widths at k=2 (`strm`):

| band | n | half-width % | vs tightest |
|---|---|---|---|
| 0–1e4 | 129 | 0.2191 | 1.00× |
| 1e4–1e5 | 181 | 0.2989 | 1.36× |
| 1e5–1e6 | 67 | 0.2621 | 1.20× |
| 1e6–1e7 | 36 | 0.4129 | 1.88× |

**The largest band has the second-widest interval.** The prediction assumed σ constant across
bands and it is not — the 1e4–1e5 band is the noisiest, which is the full-collection A/A inverted
U this project has already measured. So band power is driven by σ, not by n, and the failure
counts follow: n=129 fails twice while n=67 fails none. Their direction was right and their
ordering was not, and the ordering is what a reader would use to decide which band to distrust.

Detection also depends on the **build**, which neither of us anticipated: `shpb` detects 4/4
everywhere while `nopf` manages only 3/4 at every width, because disjointness is measured against
each build's *own* k=2 interval and `nopf`'s are both wider and shifted.

### Gap 3 — the digest control's subscript was imported from another run

Amendment 18 says "per-width digests must differ from each other". The index matrices in this
design are **per band**, four of them. The control is right and its subscript came from a
different run's phrasing. The implementation asserted the four **per-band** digests are distinct
and they are (`927f8b94ef5d5d25`, `c629d7ba634f1818`, `03fa42a6f80d81dc`, `6319e2d983008d22`), so
the executed check was correct while the registered sentence named the wrong objects. Corrected
here: **the four per-band index digests must differ.**

### What the f64 criterion inherits

Amendment 19's f64 registration is amended by all three before any f64 data exists:

* **per width**, never pooled, for gap 1's reason;
* the positive control is **per band** and its result reported as a per-band table, with the
  consequence that a band failing to detect an effect at the floor bound's magnitude **cannot
  contribute a pass** to any beyond-floor verdict in that band;
* the digest control names the **per-band** matrices;
* and σ is to be reported per band beside n, since gap 2b shows n alone does not order the bands
  by power.

### Two of the peer's checks that found nothing, recorded so they are not re-run

The band partition is complete: 129 + 181 + 67 + 36 = 413, plus the excluded n=5 band = 418,
matching the corpus. And the trichotomy is well-formed — a point estimate inside the k=2 CI means
the intervals share at least that point, so "disjoint" cannot also hold and no cell can satisfy
both labels. The burden is also correctly directed: "shares" is a conjunction over four bands and
is the outcome supporting the floor reading, so the harder test guards the dismissal of an arm
effect.

## Amendment 21 — the f64 positive control as registered was SELF-VOIDING, and three smaller fixes

Raised by the peer session at 12:40. Point 1 is the one that matters: amendment 19's f64
positive control, inherited by amendment 20, would have thrown away a correct and adequately
powered analysis.

### The control was incoherent, not conservative

Amendment 19 registered "positive control an injected effect at the bound's magnitude" together
with "beyond-floor only if the 95% CI lies entirely outside ±(floor bound)". Detection then
requires `estimate − h > b` with `h` the CI half-width; injecting the true effect **at** `b`
centres the estimate on `b`, so detection needs it to exceed `b` by a full half-width. Simulated
at 200000 draws rather than taken on the algebra:

| injected effect | detection |
|---|---|
| `b` — **as registered** | **2.45%** |
| `b` + 1h | 49.89% |
| `b` + 2h | 97.56% |
| `b` + 3h | 100.00% |

The 2.45% is assumption-free in `b` and in the standard error — only the CI's 95% level enters.
So the registered control fails roughly 39 times in 40 on a correctly built test, and amendment
20's consequence ("a band failing to detect at the floor bound's magnitude cannot contribute a
pass to a beyond-floor verdict in that band") would then void the verdict in essentially every
band.

**The error is conceptual and worth naming as such.** The criterion defines an effect at `b` as
*not* beyond floor — it is floor-sized by construction. I asked the control to detect something
the criterion is built to call floor. That is not a cautious threshold; it is a contradiction
between two clauses I registered in the same paragraph.

### The opposite failure, which rules out the obvious repair

Setting the injection to `b + 2h` using **each band's own** half-width would give 97.56%
detection everywhere — and would make the control pass by construction whenever the interval is
well-formed, which is unfalsifiable. A control scaled to the instrument's own resolution reports
nothing about whether that resolution is adequate. So the injection must be a **fixed absolute
magnitude chosen for relevance**, not a multiple of the band's own interval.

### Re-registered: a fixed magnitude, and failure LABELS rather than VOIDS

* **Injection: +3.0%, absolute and fixed**, applied per band. The anchor is prior measurement
  rather than the instrument: chain197 measured deleting the register-block prefetch at
  2.9–14.2%, and the f32 screen's floor bound was 1.489%. So 3.0% is above any floor displacement
  this project has measured and at the low end of an effect worth acting on.
* **Reported per band**, as amendment 20 requires, beside each band's `b + 2h` so a reader can
  see the chosen magnitude against that band's own detectability threshold.
* **Consequence replaced.** A band that fails the control is labelled **UNDERPOWERED AT 3.0%**
  and its beyond-floor verdict in that band is reported carrying that label. It does **not** void
  other bands and does not void the run. "Underpowered" joins shares / diverges / indeterminate
  as a reportable fourth outcome — the same move as making "indeterminate" reportable, for the
  same reason: a label that voids its own analysis will be argued away instead of reported.

### The floor bound must be measured per band, and 13.2× is why

Amendment 19 said the f64 floor bound is "the maximum over all widths and bands of
`|shpb/ship − 1|`, computed from the f64 run's own `shpb`". Point 4 supplies the quantitative
reason that it cannot instead be a scaled pooled figure. On the f32 screen at k=2, the same floor
arm gives:

* pooled `shpb/ship` = 0.999359 [0.998729, 0.999935] — half-width **0.0603 pp**;
* per-band medians +0.257, −0.538, −0.064, −0.170 — spread **0.795 pp**.

**13.2×.** The pooled statement in amendment 17 remains valid as a pooled statement — both arms
pooled over the same 418 cells — but no per-band claim can be judged against a scaled pooled
floor, and the f64 analysis is explicitly per band.

### Correction 1 is now established twice, the second time on byte-identical code

The peer found this in amendment 19's own table and I had not drawn it. The 1e6–1e7 floor band's
displacement across widths is **−0.170, −0.445, −0.370, +0.329, +0.037** — it **changes sign**,
on two objects with byte-identical `.text`. So "there is no width-invariant object offset to
subtract" holds independently of `nopf`: once on an arm, once where there is no arm at all. The
repair is closed for the floor as well as for the arms.

### Amendment 20's gap-2b replacement was one-factor in the other direction

I wrote "band power is driven by σ, not n". That overcorrects. Backing the implied spread out of
each band's half-width as `h·√n` (the 1.2533 cancels in any between-band ratio):

| band | n | half-width % | implied spread | +2% misses |
|---|---|---|---|---|
| 0–1e4 | 129 | 0.2191 | 1.986 | 2 of 12 |
| 1e4–1e5 | 181 | 0.2989 | 3.209 | 1 of 12 |
| 1e5–1e6 | 67 | 0.2621 | 1.712 | 0 of 12 |
| 1e6–1e7 | 36 | 0.4129 | 1.977 | 3 of 12 |

Implied spread ranges 1.87× and peaks at 1e4–1e5, which is the σ effect and is real. But √n
ranges 2.24×, and the **widest** interval — 1e6–1e7 at 0.4129 — is the smallest-n band at an
entirely ordinary spread (1.977 against 0–1e4's 1.986), so n explains that band by itself.
**Neither factor orders the bands alone and they are comparable in magnitude**, so the honest
form is the table, not either single driver. This is the same shape as the rule this project
already holds about decomposing a gap before naming its cause: two non-orthogonal factors have no
unique per-factor share.

And **detection is ordered by neither** — 0.2191 misses twice while 0.2621 misses none — which is
consistent with the build term being dominant for detection and makes three factors, not one.
So σ is reported beside n per amendment 20, and neither is presented as the driver.

## Amendment 22 — the criterion is now code that refuses an incoherent registration

The f64 criterion registered across amendments 19–21 is implemented as
`harnesses/pfstream/pfs_criterion.py`, and the f64 analysis will instantiate it rather than
re-describe it. Three of the four registration failures logged today mechanise, and the peer
session's classification of them is the reason this exists rather than a resolution to be more
careful:

* **"both controls belong in a registration"** is a presence check. `BeyondFloorCriterion.label()`
  raises until `run_negative()` has been **called** — declaring a negative control is not running
  one, and a criterion whose no-effect case was never exercised answers confidently on every arm.
  That is amendment 18's failure, as an exception.
* **"the injected magnitude against a whole-CI criterion"** is arithmetic. Construction refuses
  `injection <= floor_bound` and reports the detection probability, because the criterion defines
  an effect at the bound as floor-sized — so the control would be demanding detection of
  something the criterion is built to call floor. That is amendment 19's failure, at construction
  time.
* **The floor bound must be measured on this run's own floor arm**, and a non-positive bound is
  refused with that sentence attached.

**What does not mechanise, and is therefore written as prose.** The self-voiding diagnosis itself
— "the criterion defines an effect at `b` as floor, so asking the control to detect it is
incoherent" — required reading two clauses of one paragraph against each other. No arithmetic
finds that; the arithmetic only *implements* it once found. The peer priced the threshold instead
of noticing the contradiction, and I wrote both clauses, so the finding belongs to neither of our
checklists and stays in amendment 21's text.

**The pair the implementation names, because only one half was found by either of us.** A control
whose magnitude is a function of the instrument's own noise reports nothing about that noise:

* **self-voiding** — `delta <= b`, detection 2.5% at equality, fails on a correct instrument;
* **self-passing** — `delta = b + 2h` using the band's *own* half-width, detection 97.5%
  whenever the interval is well formed, so it passes without measuring adequacy.

`delta` is therefore an absolute magnitude anchored outside the instrument: **3.0%**, from
chain197's measurement of this same lever at 2.9–14.2% against the f32 screen's measured floor
bound of 1.489%.

**Adequacy is reported, never assumed.** `MIN_DETECT` is declared at 0.80 in the source rather
than implied. A band below it has its non-detection verdict suffixed
`UNDERPOWERED AT 3.00% (power …)`; other bands keep their verdicts and the run is not voided. A
`BEYOND_FLOOR` verdict is never suffixed, because low power is not a caveat on a detection.

**Plants and ablations.** `pfs_criterion_selftest.py`: 7 plants, 20 assertions, and the refusal
population extracted from the target's own AST — every `raise` site, 9 of 9 reached. Six
ablations, each asserting its anchor, its edit and its kept line:

| perturbation | sites | failures |
|---|---|---|
| the self-voiding guard removed | 8 of 9 | 3 |
| the negative-control-was-run guard removed | 8 of 9 | 2 |
| the negative-control-failed guard removed | 8 of 9 | 2 |
| `MIN_DETECT` set to 0 | 8 of 9 | 2 |
| underpowered VOIDS instead of labelling | 10 of 10 | 2 |
| the `−1.0` dropped from the detection formula | 9 of 9 | **5** |

The last is the arithmetic the entire self-voiding diagnosis turns on — detection needs the
estimate to exceed the bound by a full half-width, not merely to exceed the bound — and it is the
one an implementation could get wrong while reading as correct.

### The f32 margin table, as a regression rather than a prediction

Computed from the f32 screen's measured half-widths with `b` = 1.489%, and asserted by the
corpus so the arithmetic cannot drift:

| band | h | b + 2h | margin at 3.0% |
|---|---|---|---|
| 0–1e4 | 0.2191 | 1.927% | +1.073 pp = 4.9 h |
| 1e4–1e5 | 0.2989 | 2.087% | +0.913 pp = 3.1 h |
| 1e5–1e6 | 0.2621 | 2.013% | +0.987 pp = 3.8 h |
| 1e6–1e7 | 0.4129 | 2.315% | +0.685 pp = 1.7 h |

**These are f32 numbers and they are not an f64 prediction.** The f64 band half-widths are
unmeasured and that run's band populations may differ, since corpus composition is a property of
the run. The one expectation they license: if the `UNDERPOWERED` label appears at all, 1e6–1e7 is
where to expect it first, at 1.7 half-widths of margin against 3.1–4.9 elsewhere.

### f64 progress, with the estimator's inputs

Job 17681665: p1 `ship` landed 12:38:49 in 1181 s over 8778 rows (418 keys × 7 widths × 3 arms).
One pass of sixteen. A single landed pass is one observation of one build, so no completion
figure is quoted from it — the four-build means the projector needs do not exist yet.


## Amendment 23 — the floor instrument CHANGED for the f64 population, and the registered one was the wrong functional

Written 2026-10-01 at 18:4x, after the f64 screen completed and the 16-member family finished.

### The change, stated as a change

The registration says the **within-class agreement number of the 16-member padded family** is
read before any arm timing and is what decides whether a contrast is reported as a matched
headline or only as a bound. I honoured the ordering — the family ran to completion (64 CSVs, all
rc=0, 14400 rows) and its number was computed before any arm CSV was opened — **but the number
that ended up bounding the f64 arm contrast is not that one.** It is the `shpb` versus `ship`
spread inside the f64 screen itself. That is a different instrument and it is recorded here as a
substitution rather than absorbed, so the original ordering is not later read as having been
honoured in substance when it was honoured only in sequence.

Two independent reasons, either of which is sufficient:

**1. Host.** The family was built and timed on redwood because MKT was occupied. The f64 screen
ran on MKT. The two hosts compile different code under `-march=native` and differ by roughly 7x in
achieved bandwidth, so placement-induced spread measured on one is not a bound for the other. A
floor is named by what it holds fixed, and this one does not hold the host fixed.

**2. Functional form, which is the deeper reason.** The same full-collection comparison on this
branch gave pooled p95 of 0.0569, 0.0851, 0.1970 and 0.4023 for four different cell statistics —
a **7.1x spread driven by the choice of statistic, not by the data**. The registered number is a
coefficient of variation over **four** objects per cell. An arm contrast is a ratio between
**two** objects. Those are different functionals, so the registered statistic could not have
bounded the registered contrast even on the right host. `shpb` is a second object at the identical
configuration, measured on the same cells, widths, host, run and pass structure, so it is the
matched functional by construction rather than by argument.

The family's own output has been corrected to report the matched form alongside the registered
one — every unordered same-class pair's `|ratio − 1|`, 7200 pairs — so the registered number and
the usable number are both on the page and distinguishable.

### What the family's number actually is, now that it exists

Redwood, 16 members, c4-corrected coefficient of variation per (key, width, class) at n = 4:

| band | cells | median | p95 | max |
|---|---|---|---|---|
| 1e3–1e4 | 240 | 1.279% | 5.707% | 33.247% |
| 1e4–1e5 | 240 | 2.026% | 12.962% | 70.734% |
| 1e5–1e6 | 240 | 2.075% | 32.835% | 93.198% |
| 1e6–1e7 | 240 | 3.991% | 50.659% | 74.596% |
| ≥1e7 | 240 | 5.228% | 14.762% | 22.551% |

The matched two-object form on the same data: median 0.980 / 1.877 / 2.057 / 3.481 / 4.979% and
p95 8.741 / 13.716 / 37.844 / 52.263 / 19.106% over the same bands.

**The within-class column is not positionally confounded** even though this run used the cyclic
step-4 schedule, because that schedule's within-class position range is exactly 0. Its defect was
purely between-class, which is the term reported as an upper bound rather than an estimate.

**Band `<1e3` has zero cells**, and that is not a sampling choice: redwood's cache holds no matrix
below 1e3 nonzeros (pools 477/587/431/250/41 = 1786 over five labels). The f64 arm corpus splits
47/82/181/67/36/5, so **47 of 418 keys (11.24%)** have no family coverage. The direction is
favourable — `<1e3` is the quiet end of the inverted U — so borrowing the 1e3–1e4 floor there is
conservative, and the gap costs sensitivity on 47 cells rather than validity.

### A registered fact about the f64 width grid that changes what the screen can conclude

`scorch_simd<double>::lanes` is 4 and the register-block kernel is specialized NVEC 1..4, so
**float64's register-block range is exactly k = 1..16**. Below it, `scorch_spmm_row_narrow_exact`
owns double k = 1..3 (`exact_cap_ = exact_f32_ ? 7 : 3`), and the `narrowk_exact_mindeg` gate that
could withdraw it ships at 0.

Two consequences, both registered before the arm read was reported:

- **k=2 is a negative control, not a data point.** `SCORCH_SPMM_REGBLOCK_PFD` provably cannot
  change executed code at double k=2, so whatever that width reads is the object pair's layout
  offset rather than an effect.
- **The recorded harm at width 32 is unreachable in float64.** k=32 needs NVEC 8 for doubles, so
  the 115-matrix regression recorded against removing the prefetch is a float32 phenomenon. The
  f64 screen's grid therefore covers the macro's **entire** reachable float64 range, which the
  registration did not previously state and which is the difference between "a width is missing"
  and "there is no missing width".

### Cross-host corpus provenance, registered before the confirmation run

The 418-key list resolves 418 of 418 on both hosts, but 304 of redwood's resolutions are by
**basename** rather than exact path, and a basename match can name a different matrix. Comparing
(rows, nnz) per key across hosts: **417 of 418 agree; `Pothen/barth4/barth4` does not** — MKT has
(6019, 40965), redwood has (6019, 23492), and (40965 + 6019) / 2 = 23492 exactly, so one host
holds the full symmetric expansion and the other the lower triangle including the diagonal. Same
name, different matrix, different work, different time. **It is excluded from the confirmation
corpus (417 keys) and the reason is measured rather than assumed.** A result is invisible to
dataset substitution, so this check is registered as a precondition of any two-host claim here,
not as a diagnostic run afterwards.


## Amendment 24 — the DLMC corpus's absent stratum, measured rather than described

Written 2026-10-02 at 10:4x, while the float32 DLMC screen (job 17693884) runs.

### The claim I made, and what it was missing

I wrote that the 26 files absent from `transformer/variational_dropout/0.98` are "the layers
already thin at 0.95 — median nnz 6602 against 16466 for those that survive — which at 0.98 prune
to nothing", and labelled it as stated without a mechanism. **"Prune to nothing" is the
mechanism**, and the two medians do not support it. Under a uniform per-layer rate, nonzeros scale
by (1 − 0.98) / (1 − 0.95) = 0.4000, so the absent set would read about 2641 nonzeros at 0.98 —
present, not zero. The distance between 2641 and 0 was the whole claim and was unmeasured.

### The quantity that settles it, measured on the files that DO exist

Achieved per-layer sparsity at a nominal 0.98, over the 71 surviving layers:

| min | p05 | median | p95 | max | spread |
|---|---|---|---|---|---|
| 0.8840 | 0.9030 | 0.9895 | 0.9999 | 0.9999 | 0.1160 |

Per-layer nonzero ratio 0.98 / 0.95, against the uniform-rate prediction of 0.4000:

| min | p05 | median | p95 | max |
|---|---|---|---|---|
| 0.0005 | 0.0041 | 0.2075 | 0.8163 | 1.4069 |

**Variational dropout does not allocate sparsity uniformly at a nominal level, and the nominal
level is not a per-layer constraint at all** — the ratio exceeds 1 for some layers (max 1.4069),
meaning a layer can carry MORE nonzeros at 0.98 than at 0.95. And **16 of 71 surviving layers
retain under 5% of their 0.95 nonzeros**, the smallest at 0.0005. A layer reaching exactly zero is
the continuation of that distribution rather than a step outside it, so the claim is supported —
by this measurement, not by the two medians it was originally attached to.

### A coincidence recorded and then closed

6602 / 16466 = 0.4009 and (1 − 0.98) / (1 − 0.95) = 0.4000 agree to 0.24%, and the two quantities
are unrelated: one is a density ratio between two sets of layers at one sparsity, the other a
sparsity ratio between two levels. It was flagged before either session built on it. The
measurement above closes it: the actual median per-layer ratio is 0.2075, so the uniform rate does
not describe this pruner and the agreement is arithmetic only.

### Why this is registered rather than left in a commit message

Four times in two days a tidy structure was asserted over a set that had only been sampled or
reconciled: the symmetric-expansion account of `Pothen/barth4` (an exact identity carrying the
wrong cause), the 97 − 26 pool reconciliation (a difference equal to a remembered count), "two
suffixes across all decoder layers" (a rule read off the first three entries of a sorted list of
26), and "which at 0.98 prune to nothing" (a mechanism attached to a correlation). Each fit its
visible numbers exactly, which is why none looked like a guess.

The refutations were, in order: a predicate enumerable in advance from the key list; a design
invariant that can see absent files; a Cartesian-product comparison; and a measurement of the
quantity the claim was actually about. **The generalisable rule: when a description would explain
a set, state the check that could refute it in the same breath, and prefer a check whose inputs
are not the numbers that suggested the description.**


## Amendment 25 — amendment 24 cited the weaker of two available supports, and barth4 is now confirmed by sealed evidence

Written 2026-10-02 at 10:5x. Both corrections come from a sealed ledger artefact that was readable
all along and that both sessions had been treating as out of reach.

### The artefact, and its seal

`~/.cache/scorch-codex/spmm-beat-mkl/chain209-dlmc-dims/`, verified here:
`shasum -a 256 -c SHA256SUMS` reports OK for all 8 files, and the manifest's digest field is 64
characters wide, so it is a real SHA-256 and not one of the CRC-32 manifests in this ledger that
`shasum -c` passes vacuously. The two files used below:

| file | sha256 (first 16) | content |
|---|---|---|
| `ss_dims.csv` | `195e940572acfd80` | key,rows,cols,nnz for 4584 SuiteSparse keys, from redwood's npz cache |
| `dlmc_dims.csv` | `ff167f71d50abf74` | the same for 3846 DLMC keys |

Its `PROVENANCE.md:29` warns against using `ss_dims.csv` **for density** — 67 of the 150
SuiteSparse matrices at ≥20% have a defect it describes. Nothing below uses density; the uses are
a nonzero count and a key-layout census, so the caveat does not reach them. Anyone citing these
files for a density claim needs that line first.

**A naming hazard beside it, recorded because the filename is what travels.** A second file named
`dlmc_dims.csv` exists at `~/.cache/scorch-codex/verifier-dlmc-dims/`, digest `3f0887e553b61e71`.
It is not a copy and not a contradiction: it holds column counts read from the local `.smtx`
files for chain210's keys, 3822 lines including comments, sharing 3820 keys with the sealed file
and omitting the 26 zero-nonzero ones. Two different measurements under one filename in one
ledger is the configuration in which a citation by name is ambiguous, so both of these must be
cited by digest.

### Correction to amendment 24

Amendment 24 supports "these 26 layers prune to nothing" with the achieved-sparsity distribution:
median per-layer nonzero ratio 0.2075 against the uniform-rate 0.4000, max 1.4069, and 16 of 71
survivors under 5% of their 0.95 nonzeros. **That argument establishes that zero is REACHABLE by
this pruner. It does not establish that these 26 reached it.**

`dlmc_dims.csv` establishes exactly that, by name: 26 of its 3846 rows have `nnz == 0`, every one
under `dlmc/transformer/variational_dropout/0.98`, and `3846 − 26 = 3820` is the DLMC count cached
on both hosts. So the absent-from-cache set IS the zero-nonzero set, by enumeration and by
subtraction — not by two counts of 26 agreeing across runs, which is the form that would have
deserved the suspicion the earlier reconciliation got. The structure also reproduces from the
sealed file independently of my walk over the cache: 2 sides, indices {0,1,3,4,5}, 8 suffixes,
product 80 against 26, split `{decoder: 4, encoder: 22}`.

**Amendment 24's sparsity measurement keeps its own value and stays.** It refutes the uniform-rate
model outright, which is what the objection to it actually required, and the maximum ratio of
1.4069 — a layer carrying MORE nonzeros at a nominal 0.98 than at 0.95 — is a fact about
variational dropout that nothing else on the page carries.

### barth4, confirmed rather than argued

`grep -i barth4` over `ss_dims.csv` returns exactly one line: `ss/barth4/barth4,6019,6019,23492`.
One barth4 in redwood's cache, stored **flat** with no group level, holding 23492 nonzeros —
Nasa's value. Both requested keys therefore resolve to a single file that is Nasa's matrix. The
symmetric-expansion account is dead on direct evidence, not on argument, and the exclusion of
`Pothen/barth4` from the 417-key confirmation corpus is confirmed correct for the stated reason.

**The exposure is bounded by the flat subset, which sharpens the predicate.** Only 1772 of 4584
cache entries have the flat `ss/<X>/<X>` form — 38.66%. The rest are group-qualified
(`ss/AG-Monien/3elt/3elt`) or multi-matrix (`ss/ABACUS_shell_ud/ABACUS_shell_ud_K`), and a
group-qualified entry resolves exactly and cannot collide. So duplicate basenames among requested
keys remains the right predicate, and its exposure is confined to keys whose cache entry is
stored flat. Reporting which member of a colliding pair is stored flat says whether the collision
is reachable at all.

**One sharpening tested and withdrawn, by the measurement rather than by argument.** The
within-cache ambiguity count of 0 of 4584 looked vacuous by construction — a cache keyed
`ss/<X>/<X>` cannot hold two directories of one name. The mixed layout refutes that: the count was
capable of returning nonzero, so it was a measured zero over the wrong population rather than a
vacuous one.

### Corrections to amendment 25

Four, all verified here before adoption.

**The manifest holds 9 entries, not 8.** `grep -c . SHA256SUMS` = 9, all at digest width 64, and
`shasum -a 256 -c SHA256SUMS` prints nine OK lines with rc=0 read directly rather than through a
pipe. The directory lists 11 entries, of which `SHA256SUMS` and `SHA256SUMS.sha256` are not
manifest lines, so 11 − 2 = 9 and no natural subset of 8 exists. My 8 came from counting the lines
a `tail -8` had displayed to me — the same error as reading a rule off the head of a sorted list,
one level down: a count taken from a truncated view of the thing being counted. The direction
understates, so a reader reproducing it gets a larger number and has to work out which file the
record skipped. The seal-of-seal also holds: `-c SHA256SUMS.sha256` prints `SHA256SUMS: OK`, rc=0.

**The second file is 3823 lines, not 3822**, decomposing exactly as 2 comment lines + 1 header +
3820 data rows. My 3822 came from `len(rows) - 1` over a `csv.reader`, which charges one line to a
header when the file has two comment lines above its header.

**The two files are not equally attested, and that asymmetry is what tells a reader which digest
to prefer.** `chain209-dlmc-dims` carries `SHA256SUMS` (9 entries), a verified `SHA256SUMS.sha256`
sealing it, and a 41-line `PROVENANCE.md`. `verifier-dlmc-dims` has **no `SHA256SUMS` and no
`PROVENANCE.md`** — only a one-line file *named* `SHA256SUMS.sha256` which is itself the manifest,
inverting the role that exact name plays in chain209. Two consequences: a checker pinning the
conventional name gets a failed open and reads the directory as unsealed, where the command that
works is `-c SHA256SUMS.sha256`; and with no PROVENANCE.md that directory records no producing
host and no producing session, its directory name being the only identification. The cited digest
is correct — recomputed as `3f0887e553b61e71627246d676035b3577382f314103df11fb36527dd011b028`,
matching both the file and its one-line manifest — so nothing numeric moves. It is also a
**sibling of `spmm-beat-mkl` at the cache root**, not under it; amendment 25's body gives the full
path correctly but the commit body reads as though it were nested.

**`dlmc_dims.csv` is NOT npz-derived, and amendment 25 implied it was** by calling it "the same"
after describing `ss_dims.csv` as coming from redwood's npz cache. Per `PROVENANCE.md:10-17` it is
built from the **first header line of each `.smtx`**, generated independently on redwood
(`/scratch/bobbyy/dlmc`) and on the Mac (`~/dlmc`) and byte-identical between them — which is what
binds the Mac corpus to the corpus chain207 timed. `ss_dims.csv` is the npz-derived one, via
`csrresolve.py` resolving all 4584 keys `exact`, with shapes read from the stored `shape` array.

The distinction has a consequence worth stating: the 3820 keys shared between the sealed and
unsealed DLMC tables agree on rows and nnz with zero disagreements, and that is **not** independent
corroboration — both are reads of the same `.smtx` headers. It confirms the join and the
subsetting, nothing about the numbers' independence. The genuine independence in the sealed file is
**cross-host** (two machines, same source, byte-identical), not cross-source.

Also recorded, since amendment 25 quoted the density caveat without its cause: the 67 of 150
matrices at ≥20% density have **`cols == 1`** — they are right-hand-side vectors, where "density"
is the filled fraction of a single column. **The 0.3347-vs-0.2416 pair is the 60-101% band, not the
≥20% band as a whole.** Measured from `ss_dims.csv`, the three bands partition the 150 exactly
and so does their vector count:

| band | n | `cols == 1` | vectors |
|---|---|---|---|
| 20-35% | 54 | 1 | 1.9% |
| 35-60% | 24 | 2 | 8.3% |
| 60-101% | 72 | 64 | 88.9% |
| (all ≥20%) | 150 | 67 | 44.7% |

So the contrast those two numbers describe sits in a population that is 88.9% right-hand-side
vectors, not one that is 44.7%. The vectors concentrate almost entirely in the top band, which is
the actionable half of the caveat: the 35-60% band is 2 of 24 and is clean. Below 60% is 78 matrices
holding 3 vectors (3.85%), which cannot move the statistic those two numbers report — and that
statistic is now named rather than assumed. `0.3347` and `0.2416` are `S all` and
`S non-vector` from `an_verify5.py` POINT 5: the **geometric median** (`red` = `exp` of the
median of logs) of `ours/mkl` at **k=4**, where S below 1 means Scorch is faster. Being a
median, it cannot be moved by the 3 contaminated members among the 78 matrices below 60%, so the
exemption holds for *this* statistic and is not established for a maximum or a p95 of the same
ratio. The direction also favours us: removing the vectors moves S from 0.3347 to 0.2416, so the
right-hand-side vectors are where our margin is *smallest*, and the split the sealed file asks
for strengthens the result rather than weakening it.

**The exemption is now measured rather than inferred, in both sub-bands, and the producer only
covers one of them.** `an_verify5.py:52` loops over `((.35,.60),(.60,1.01))`, so the artefact
carries no row for 20-35% and "below 60%" spans a band the producer never reports. Recomputed
from chain196's twelve local float32 pass files using the producer's own definitions — by the
verifying session first, and reproduced here independently:

| band | n | vectors | S all | S non-vector |
|---|---|---|---|---|
| 20-35% | 54 | 1 | 0.2435 | 0.2433 |
| 35-60% | 24 | 2 | 0.2371 | 0.2362 |
| 60-101% | 72 | 64 | 0.3347 | 0.2416 |

The bottom two rows reproduce `PROVENANCE.md:32-33` in all four numbers; the top row is new and
is the verifying session's measurement. Both sub-bands below 60% are clean on this statistic, so
the exemption stands on measurement in each, not on the membership fraction alone. The artefact
is sealed and is not being amended to add the row; this table is where it lives.

**A claim withdrawn, from the paragraph this replaces.** I wrote that "`64 of 72`" and S's n have
different denominators — the first counting `ss_dims.csv` rows, the second counting chain196 cells
that survive six passes in both builds. Measured: `len(cells)` and `len(ok)` are **both 22920**, so
no cell in that run fails the pass test, and the k=4 slice is one cell per matrix. 72 *is* S's n
and the two halves share a denominator. The general worry is sound and would bite in a run where
cells do fail; the one-line check that settles it either way is `len(cells) == len(ok)`, and it
should be printed beside any such statistic rather than reasoned about. What pointed at the error
was an internal contradiction in my own paragraph, not the measurement: the robustness argument
treats 78 as the sample size while the next sentence denies that the membership count is the
sample size. Two claims that cannot both hold is a cheaper signal than either one checked alone.

`PROVENANCE.md:33-35` also records that the 5-matrix control in the results doc has zero
vectors because its `nnz >= 2e4` bound makes `cols == 1`
unreachable at `rows <= 4096` — that was luck in the predicate, not foresight, and it does not
transfer to a control chosen on any other bound.


### Corrections to the corrections above (two defects in my own new material)

Both were found by the verifying session, within the hour, in the paragraphs immediately above.

**1. "a 31-line `PROVENANCE.md`" is 41 lines.** `wc -l` reads 41; `grep -c .` reads 31; the file
has 10 blank lines. My census script printed `$(grep -c . "$f") line(s)`, so the number was a
non-blank-line count wearing the label `lines`. This is not the truncated-view error from the
other corrections in this section — the view was complete and the predicate was wrong, which is
why auditing the number against the file it describes could not find it and only recomputing with
the named predicate could. A count in a durable record must name the predicate that produced it;
`line` and `non-blank line` differ by **27.8%** on this file, the symmetric
`2(a-b)/(a+b)`; the asymmetric pair is 24.4% with 41 as the base and 32.3% with 31, and I had
published the 24.4% without naming its base. That is the same order-dependent two-object
relative difference I replaced with a symmetric form in `an_floor_within.py:220` about an hour
earlier, reappearing in prose — a fix to an estimator reaches one analyser, and prose inherits
nothing from it, so after correcting a statistic's form in code the amendments and commit bodies
have to be swept for the same form by hand.

**The sweep's scope was wrong before it was right.** I first swept the amendments only and
reported one instance. Swept over the whole file, the five spellings (`% apart`,
`% difference`, `differ by N`, `understat`, `overstat`) return 16 lines, of which 15 are
directional claims about an estimator or a proxy — "the proxy always overstates", "the floor
is understated" — where there is no second object and no base to name. One is a genuine
two-object ratio: `:264`, "1.3% apart" on 1.2114 against 1.1963, now carrying the symmetric
figure and both asymmetric readings. Its spread is 0.016 percentage points, so nothing it
supports moves. The scoping error matters more than the hit: a hand sweep fails at the
spelling list *and* at per-hit classification, and `% apart` was on my list and present at
:264, so this was the classification half, not a missed spelling.

**2. The 0.3347 / 0.2416 pair is the 60-101% band, not the ≥20% band.** Fixed in place above with
the measured three-band partition. The direction matters: attributing it to ≥20% moves the
population from 88.9% right-hand-side vectors to 44.7%, which understates the concentration and so
runs *against* the caveat's own point. The sealed `PROVENANCE.md` had the band right; the error
entered when I transcribed it, which makes the derived record less reliable than the artefact it
cites — the failure mode that makes transcription worth checking against the source each time
rather than once.


## Amendment 26 — a structural predictor for the 192-cell float64 deficit, registered before it runs

The float64 screen leaves **192 of 2926 cells behind MKL**, and neither prefetch arm touches
them: `PFD=0` wins 1.3-2.9% at k=4..16 but reads **+0.139%** on that selected set at k=2, inside
the floor's own 61% better-count there. So the deficit is not the register-block prefetch, and a
third arm on the same axis would be a guess. This amendment registers a *mechanism* and the
measurement that can refute it, before the measurement is run, because the corpus verdicts are
already known to me and any threshold chosen after looking at them would be fitted.

**The mechanism.** Every nonzero gathers one row of B. At k=2 float64 that row is 16 bytes, inside
a 64-byte cache line, so **four B rows share a line** and whether they are used together is a
property of A's column index pattern, not of the kernel. `harnesses/pfstream/bstream_locality.py`
measures exactly that and nothing else: per row, the number of distinct 64-byte B lines its
columns touch; amortized per nonzero; reported as `lines_per_nnz` and as
`pack = 1 / (lines_per_nnz * brows_per_line)`, the mean fraction of each fetched line the row
consumes. It reads only `indptr` and `indices`, so no arm and no timing can influence it.

**The reach is arithmetic, and it is narrower than I first reasoned.** `brows_per_line` is
`max(1, 64 // (k * itemsize))`. I wrote down the boundary as "`k * 8 < 64`, so k = 2, 4, 6" and the
computation returned k = 2, 4 only: at k=6 a B row is 48 bytes and `64 // 48 == 1`, because two
48-byte rows do not fit in a line. Integer division, not a judgment call, and the self-test pins
it. So for float64:

| k | B row bytes | brows_per_line | structural signal | cells behind MKL |
|---|---|---|---|---|
| 2 | 16 | 4 | yes | 129 |
| 4 | 32 | 2 | yes | 35 |
| 6 | 48 | 1 | **none, constant** | 18 |
| 8 | 64 | 1 | none, constant | 9 |
| 10 | 80 | 1 | none, constant | 1 |
| 12, 16 | 96, 128 | 1 | none, constant | 0 |

The predictor can speak for **164 of 192 cells (85.42%)** and is provably silent on the other 28.
That limit is stated here rather than discovered afterwards. For float32 the same arithmetic gives
a signal at k = 2 and k = 8 and silence from k = 12 up, which is a second, independently reachable
test on the DLMC screen's own width set.

**What would establish it, and what would refute it.** Within k=2 and k=4, the behind-MKL cells
should show **lower `pack`** than the ahead cells — worse line utilization, so more bytes fetched
per useful B row. Two conditions on that, both registered now:

1. **It must survive conditioning on degree.** Mean degree already separates these sets (behind
   6.86 vs ahead 3.98 at k=2). A predictor that only reproduces the degree split has added
   nothing, so the contrast is to be read *within* degree strata, and the registered criterion is
   that the separation persists inside strata, not merely in the pooled comparison.
2. **It must be silent where it is constant.** At k=6 and above, `pack` is pinned at 1.0 for every
   matrix by the table above. An analysis that reports a `pack` contrast at those widths has a bug,
   not a finding. This is a negative control the construction supplies for free.

Refutation: if the behind and ahead `pack` distributions overlap at k=2 and k=4 as much as they do
at k=6, the mechanism is wrong and this amendment closes with a null.

**What it does not license.** No threshold, no cache-size constant, and no per-matrix rule. The
script's output is a continuous structural column and it classifies nothing. If the mechanism
survives, the kernel change it would motivate is a B-side access-order change, and under the
project's performance convention that change ships only gated on a condition that *provably cannot
fire* where it would hurt — which, given the table above, means a gate that is inert for
`k * itemsize >= 32` by construction rather than by measurement.
