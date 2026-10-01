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

| k | deep/base | bsb_/base (A/A) | x floor | NVEC / vector |
|---|---|---|---|---|
| 2 | **1.0004** | 1.0017 | **0.2** | *arm unreachable* |
| 4 | 1.0944 | 1.0017 | 54.9 | 1 / `__m128` |
| 8 | **1.1963** | 1.0018 | 106.6 | 1 / `__m256` |
| 16 | 1.0712 | 1.0038 | 19.0 | 2 |
| 32 | 1.0833 | 0.9997 | 306.3 | 4 |
| all | 1.0768 | 1.0017 | | n = 19100 |

So the per-call effect **is** 5-19% -- 9.4% at k=4, 19.6% at k=8 -- and nothing in the wall-time
withdrawal licenses doubt about the board's per-cell estimator. The withdrawn claim was about a
sum; this is about a ratio.

**k=2 is a positive control for the board's own instrument, and it passes.** The deep arm is
bypassed at k=2 for the same reason PFSTREAM is (Amendment 1), and k=2 reads 1.0004 against a
1.0017 floor -- **0.2x floor, inside it**, the only width of five that is. A width where the arm
provably cannot fire reads flat. That is the control this board was never given explicitly, and it
was in the data all along.

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

At k=8, `1/0.8255 = 1.211` against a measured `ms` ratio of 1.1963 -- the batch is `target/ms` to
within 1%, which confirms the pinning mechanism directly. The A/A pair's divergence (43.9%) is
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
