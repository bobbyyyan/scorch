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
