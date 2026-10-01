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

At k=8, `1/0.8255 = 1.2114` against a measured `ms` ratio of 1.1963 -- **1.3% apart**, so the batch
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
