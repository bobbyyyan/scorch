# Pre-registration — multi-row at the half-vector width, float32 k=4, redwood

Written before any object is built. Nothing below is measured.

## Why this width and no other

redwood's full DLMC float32 board (chain207, 3820 keys × 5 widths, 12 readings per cell) puts the
behind-MKL set almost entirely at one width. Behind in **all 12** readings, split by mean degree:

| width | total | deg<256 | deg≥256 |
|---|---|---|---|
| k=2 | 0 | 0 | 0 |
| **k=4** | **51** | **24** | **27** |
| k=8 | 2 | 2 | 0 |
| k=16 | 0 | 0 | 0 |
| k=32 | 0 | 0 | 0 |

Independently recomputed by a second reader from the same sealed files, agreeing cell for cell.
Two further facts sharpen it:

- **Both k=8 cells are at deg<256.** So in the high-degree band the behind-in-all set is 27 cells
  and **every one is at k=4** — nothing is left over at any other width in that band.
- **k=8's observed 2 is exactly that width's own null** (per-width null `{2:0, 4:1, 8:2, 16:0,
  32:0}`), so k=4 is the only width above its own null.

And at these cells we **beat** MKL at the neighbouring widths: sampled high-degree behind cells read
S(2) 0.79–1.02 and S(8) 0.88–1.13 against S(4) 1.10–1.24. This is a width-specific kernel problem,
not a matrix problem.

## The mechanism, and why it is not a threshold

At float32 k=4 on AVX2 a full 256-bit vector holds 8 floats and the width uses 4 — half the lanes
masked off. `SCORCH_SPMM_HALFVEC_F32=1` serves the width from a 128-bit register instead, which is
mask-free (11.5 emitted instructions per nonzero against the masked path's 13.5) and is worth
**1.1008 (z +14.6)** over it per `scorch_policy.h:1502-1507`. But a 128-bit vector delivers half the
FLOPs per instruction, so what remains is a dependency- and stream-limited loop at full occupancy.

Multi-row adds `ROWS` independent load streams and `ROWS` independent FMA chains. The policy header
is explicit that this is **not** a reuse mechanism — the kernel loads `ROWS` *different* B rows per
iteration — and that the measured gain is roughly flat in width (9.2 / 8.2 / 7.7 / 12.4% at 20, 24,
28, 32 bytes, correlating with the waste factor at **−0.45**, the wrong sign for a reuse story).

So the two fixes at this width are independent: the half-vector drops the mask, multi-row adds
streams. **`SCORCH_SPMM_MULTIROW_HALFVEC` exists precisely to allow both at once** and ships at 0.

## This is NOT the configuration that was measured and rejected

`spmm.h:4400-4407` records: *"Without it multi-row replaces the half-vector path rather than the
masked register block it is meant to replace. Measured at float32 k=4 … ROWS=2 read 0.9425 on the
kernel and 0.9466 on the caller, uniformly across every degree band."*

That configuration is multi-row with the **masked 256-bit** kernel taking the width — trading the
mask-free property away to gain streams, and it lost 5.7%. `SCORCH_SPMM_MULTIROW_HALFVEC=1` routes
instead to `scorch_spmm_multirow_regblock<scalar_t, 1, true, ROWS, VH>` (`spmm.h:4762-4779`) with
`VH = scorch_simd_half<float>` — multi-row **at 128-bit**, keeping both properties. The header says
exactly this: *"Ships at 0: the mechanism is new and unmeasured, and the exclusion it lifts was
measured."*

## Reachability, verified at source before building

Every term of `multirow_ok` (`spmm.h:4410-4425`) at float32 k=4 with both macros set:

| term | value at f32 k=4 | |
|---|---|---|
| `multirow > 1` | 2 | **`=1` is INERT** — an arm set to 1 reads as "no help", not "never fired" |
| `multirow_minnnz <= 0` | ships `0L` (`:741`) | open |
| `multirow_mindeg / maxdeg <= 0` | ship `0` (`:831`, `:929`) | open |
| `narrow_k` | `B1_size <= 4*SL` = 32 | true |
| `!exact_width` | `exact_hi_` ships 3, so k=4 is not exact | true |
| `!(narrowk_gather && nvec==1 && float)` | `narrowk_gather = (B1_size==1)` = 0 (`:4358`) | true |
| `!(spmm_halfvec>0 && !multirow_halfvec && B1_size==half_lanes)` | `multirow_halfvec=1` kills it | true |

**Both macros are required and neither alone suffices**: without `MULTIROW_HALFVEC=1` the last term
excludes the width; without `ROWS_F32>1` the first term excludes it.

## Build

Three objects, same tree, same flags but for the defines:

- `mr0`, `mr0b` — **duplicates of ship.** Required to be **`.text`-identical**; their ratio is this
  run's floor.
- `mrhv` — `-DSCORCH_SPMM_MULTIROW_ROWS_F32=2 -DSCORCH_SPMM_MULTIROW_HALFVEC=1`. Required to
  **differ** from `mr0` in `.text`.

**`_F32` specifically, never the base macro.** `scorch_policy.h:617` states that a single
`-DSCORCH_SPMM_MULTIROW_ROWS=N` sets both dtypes, which would move float64 in a run whose registered
deficit is float32.

**Compiled-in, not hooks.** `spmm.h:4458-4466` records that `-DSCORCH_SPMM_MULTIROW_ROWS=2`
previously **failed to compile** on `sp_row_units was not declared`, repaired by the
`constexpr long sp_row_units = 0` in the split-off `#else`. So a compiled-in three-build is possible
now and was not before — which means `scorch_policy.h:757`'s measured "EXACT_HI_F32=1 with
MULTIROW_ROWS_F32=2 wins 11–20%" must have come through the `SCORCH_TUNE_HOOKS` env path. **Stated
as an inference from those two comments, not as established provenance.** Going compiled-in avoids
both the hooks per-call `getenv` tax — safe arm-vs-arm and **not** safe against MKL, and the deficit
is defined against MKL — and the ~1.1% x86 cost per environment variable an arm sets.

## Controls, both derivable from shipped code

- **k=2 is a NEGATIVE control.** `exact_width` owns float32 k ∈ {1,2,3}, so `!exact_width` fails and
  multi-row provably declines. The arm must be a structural null there.
- **k=8 is a POSITIVE control.** Half-lanes is 4, so `multirow_halfvec` is irrelevant at k=8, but
  `ROWS_F32=2` admits plain multi-row. The header already measured ROWS=2 one width up at
  **1.0652 / 1.0825**. If k=8 does not move, the arm did not fire.

## The four refutation conditions

> Superseded in part. Condition 1, 3 and 4: see amendments 1, 2 and 3. Condition 3 again:
> amendment 2 (question changed) and amendment 6(c) (estimator). Condition 4 again:
> amendment 7 (stratifier withdrawn). Condition 2 stands as
> written. The gated-cell range quoted below and in amendments 2-3 is WITHDRAWN by
> amendment 6(a), and REINSTATED with its unit named by amendment 8: 34.6-54.6% of S-1.
, registered before the run

> **CONDITIONS 1, 3 AND 4 BELOW ARE SUPERSEDED — DO NOT SCORE THEM.** Condition 1 is replaced by
> amendment 1(c) (it certified a width where one macro under test is provably inert). Condition 3
> is replaced by amendment 1(b) and then again by **amendment 2**, which withdraws amendment 1's
> floor-gated primary as structurally unreachable. Condition 4 is replaced by amendment 1(e).
> **Only condition 2 below stands as written.** This marker is here because a reader greps for
> "refutation conditions" and starts at this heading.

1. **Reachability.** If k=8 does not move outside the floor, the arm never fired and **nothing else
   in the run is readable** — not a null result. This is the condition that makes the rest
   interpretable and it is scored first.
2. **Neutrality.** If k=2 moves outside the floor, the build is not neutral where it provably must
   be, and every other reading is confounded by whatever did change.
3. **The question, on reach.** Of the 51 cells behind MKL in all 12 readings at k=4, how many come
   to at least parity? **Passing requires more than half (≥26).** Fewer than that is a bound on
   reach, exactly as the thread-sizing registration treated its own share.
4. **No regression.** If the arm's pooled k=4 ratio on the keys that are **not** behind worsens by
   more than the floor, this is a trade and not a fix, and does not ship whatever condition 3 says.

**Both degree bands are reported separately** — 24 cells below mean degree 256 and 27 at or above.
The bands are not equally clean populations (the high band is 27 of 27 at k=4, the low band 24 of
26), so a pooled refutation condition would mix them.

**No rule, gate or threshold is proposed under any outcome.** This arm flips two existing constants
that already ship at 0; if it wins, what ships is a constant's value, and the no-regression grid
decides that separately.

---

# Amendment 1 — written BEFORE any object was timed, after a second reader checked the conditions

No timing exists yet; the build had not completed when this was written. Nine findings, of which
four change a registered condition. Recorded as an amendment rather than an edit so the original
conditions stay legible.

## (a) The neighbour profile, over the full population instead of a sample

The original quoted "S(2) 0.79–1.02 and S(8) 0.88–1.13 against S(4) 1.10–1.24" from a sample, and
**two endpoints were truncated by it**: S(2)'s max is 1.1151 and S(4)'s is 1.5627. Exact, over all
27 high-degree behind cells:

| width | min | median | max | behind |
|---|---|---|---|---|
| S(2) | 0.7908 | 0.9715 | 1.1151 | 6 of 27 (22%) |
| **S(4)** | **1.0958** | **1.1507** | **1.5627** | **27 of 27 (100%)** |
| S(8) | 0.8554 | 0.9433 | 1.1206 | 8 of 27 (30%) |
| S(16) | 0.6794 | **0.7767** | 0.9347 | **0 of 27 (0%)** |
| S(32) | 0.4710 | 0.7860 | 1.1054 | 1 of 27 (4%) |

**The sentence "at these cells we beat MKL at the neighbouring widths" is false as written** — it
fails for 6 of 27 at k=2 and 8 of 27 at k=8. **Withdrawn** and replaced by the share form, which is
true and stronger: 27 of 27 behind at k=4 against 6 and 8 of 27 at the neighbours.

And the two best facts were missing: on these very 27 matrices we are **22.3% faster than MKL at
k=16 with zero of 27 behind**, and 21.4% faster at k=32. The profile is **0.97 / 1.15 / 0.94 / 0.78
/ 0.79** — k=4 is a single spike in an otherwise winning curve, on the same matrices. That display
supports "a width-specific kernel problem, not a matrix problem" far better than a range does.

## (b) CONDITION 3 IS REPLACED. As written it scores a working mechanism as a failure

With n=51 the 26th-smallest **is** the median, so "≥26 of 51 reach parity" is identically "the gain
exceeds the median S(4) at k=4", which is **1.1475 — a 14.75% gain**. The mechanism's own cited
profile is 9.2 / 8.2 / 7.7 / 12.4%. Implied counts over the 51:

| gain | cells reaching parity |
|---|---|
| 1.092 (9.2%, nearest the target width) | **4 of 51** |
| 1.124 (12.4%, the profile maximum) | **13 of 51** |

**So the likely outcome is 4–13 of 51, which the original condition 3 records as a non-pass even if
the mechanism performs exactly as measured.** That is the defect, and it is the same shape as
amendment 31's "an interval computed for one condition and not applied to another decided by the
same number", committed minutes earlier: **condition 3 was the only condition in this document with
no floor term, and it is the one that decides the arm.**

### Three further reasons the original condition 3 is unreadable

- **The per-cell verdict is inside the floor.** Over the 51, the two-build spread between
  chain207's `.text`-identical replication pair is min 0.10%, p25 1.82%, **median 5.40%**, p75
> **Not transferable (amendment 9(4)):** this is a chain207 6-pass figure. The screen is 4 passes, so it does not transfer as a threshold. The point being made with it, that a count is a threshold in disguise, is unaffected.

  9.92%, p95 26.31%, max 52.80%. 26 of 51 exceed 5%.
- **At the pass line the two byte-identical builds straddle it.** Scoring "does gain g reach
  parity" separately per build: at g = 1.1475, build `aa` says **27** and build `bb` says **20**,
  with **25 of 51 verdicts flipping** — and the registered line is 26. **The verdict would be
  decided by which replication build was measured.**
- **[CORRECTED BY AMENDMENT 3(a): it is 50 of 51, and the two sets are NOT nested — they share one cell.]** This run's own sealed analysis headlines "both builds
  beyond the ratio floor" at per-width `1/8/1/0/0` — **8 at k=4**, not 51. "Behind in all 12
  readings" is a consistent *sign* with no magnitude term. Both predicates are defensible and the
  original named its own, but condition 3 asked the arm to fix cells where the deficit is not
  separated from the replication pair.

### Condition 3, replaced

**3a (primary, floor-gated).** Of the **8** cells at k=4 that are behind MKL beyond the ratio floor
on **both** replication builds, how many reach parity? Scored only on cells whose parity verdict
**agrees across `mr0` and `mr0b`**; cells whose verdict flips are reported as a separate count and
excluded from both numerator and denominator.

**3b (secondary, floor-free).** The same count over the 51 sign-only cells, reported **beside its
own two-build flip count at the measured gain**, never as a bare total.

**3c (the criterion in the mechanism's units).** Report the measured gain at f32 k=4 and the implied
parity count **as a curve**, not against one threshold — the table above, recomputed with the
measured value in it. A criterion whose pass line exceeds every value in the mechanism's own
measured range is not a test.

**Registered now: the one outcome under which a threshold-style reading is available is a measured
gain above 14.75% with per-cell verdicts agreeing across `mr0`/`mr0b`.** Written down here rather
than discovered at scoring time.

## (c) CONDITION 1 IS REPLACED. k=8 certifies the wrong macro and the wrong instantiation

The original made k=8 the positive control. The exclusion at `spmm.h:4422-4425` needs
`spmm_halfvec >= 2` for its second disjunct and `HALFVEC_F32` ships 1, so **`multirow_halfvec` is
provably inert at k=8** — and at k=8 the half-vector kernel does not serve, so multi-row runs the
**full-vector** template, the one already measured at 1.0652/1.0825. The arm's actual new path,
`multirow_regblock<..., scorch_simd_half<float>>` at `:4762-4779`, executes **only at k=4**.

**So condition 1 as written can pass honestly while the code path this run exists to test never
executed** — and a null at k=4 would then be unable to distinguish "ran and did not help" from
"never ran", which is the exact outcome condition 1 exists to exclude.

### Condition 1, replaced — a firing check AT the target width, from defines alone

Add a fourth object **`mrhv4`**: `-DSCORCH_SPMM_MULTIROW_ROWS_F32=4 -DSCORCH_SPMM_MULTIROW_HALFVEC=1`.
`spmm.h:4769-4779` instantiates `case 2` and `case 4` of the half-vector multi-row switch, and the
kernel advances `i += multirow - 1`, so ROWS=2 and ROWS=4 are **different executed code at k=4**.

- If `mrhv` and `mrhv4` differ at k=4 **beyond the floor**, the half-vector multi-row path fired at
  the target width. This tests the instantiation under test, at the width under test, with no
  source instrumentation and no environment variable.
- If they do not differ, the path did not fire and **nothing at k=4 is readable** — unchanged in
  consequence from the original condition 1, but now pointed at the right code.
- k=8 is retained as a **secondary** check that `ROWS_F32` reached the object at all, which is what
  it can actually certify.

## (d) "Measured" is withdrawn from the gain profile

`scorch_policy.h:641-643` pins the unit: the 9.2 / 8.2 / 7.7 / 12.4% figures are at **20, 24, 28
and 32 bytes** of B-row footprint, which at float32 is **k = 5, 6, 7, 8**. The target, f32 k=4, is
**16 bytes — one step below the grid's lower endpoint.** And at k ∈ {5,6,7,8} the half-vector kernel
does not serve, so those are multi-row at **full** vector width. **The arm's instantiation is
unmeasured at any width.**

Both extrapolations still favour the arm — the 20/24/28 trend rises as bytes fall, and the
stream-limited-at-full-occupancy argument predicts more from added streams at 128-bit — so this is
not an objection to the arm. It is an objection to the word *measured*, and it matters only because
condition 3's threshold was set as though the gain were known. Also noted: "roughly flat in width"
is a gloss on a non-monotone sequence whose max is **1.61×** its min, earned by the **sign** of the
−0.45 correlation rather than by a point prediction.

## (e) CONDITION 4 IS REPLACED. Unnamed functional, wrong floor, wrong moment

> **Superseded by amendment 7:** the row-dispersion stratifier registered here is
> withdrawn -- it sorts cells by their own A/A floor (2.7x across terciles) and predicts
> the outcome (rho -0.62). Condition 4 is scored with no stratification.


Three defects:

- **The functional is unnamed.** Geomean, median and mean of a pooled ratio over ~3769 keys are
  different statistics with different floors.
- **It compares a pooled statistic to a per-cell floor.** A pooled number over 3769 keys carries
  roughly `1/sqrt(3769)` of the per-cell dispersion, so "worsens by more than the floor" is close to
  uncrossable. **The floor must be that functional's own floor at that n, from the replication pair.**
- **It stratifies on the wrong moment.** `scorch_policy.h:752-758` already names this mechanism's
  loss regime and it is not mean degree: *"a group of unequal rows pays the register pressure of
  ROWS accumulators and collects the parallelism of one"* — `connectus` 0.602 on redwood against a
  same-code floor of 1.080, `lhr` 0.820–0.845, `cari` 0.850, `lp_osa_14` 0.852. Mean degree is a
  first moment and cannot separate uniform from power-law rows, so it cannot expose a
  dispersion-predicted loss, and pooling dilutes a 15–40% loss on an irregular minority.

**Condition 4, replaced.** Report the **geomean** of the k=4 ratio over the not-behind keys against
**the geomean's own floor at the same n** from `mr0`/`mr0b`; and stratify by **row-length
dispersion** (coefficient of variation of row length), not mean degree, with the most irregular
decile reported separately. Those loss cells are SuiteSparse and this run is DLMC, so the transfer
is not established — this changes how **this** run is read, and shipping still defers to a separate
no-regression grid.

## (f) The per-width null's definition, and one over-precise sentence withdrawn

The original printed the null as a brace literal with no definition. The computation that reproduces
three of five cells is the matched 12-fold arm-pair A/A — cells where `ours > oursb` in all 12
readings, same n and same conjunction depth — which gives **`{2:0, 4:1, 8:2, 16:1, 32:1}`** in one
direction, where the original printed 0 at k=16 and k=32. That makes those two widths strictly
**below** their own null rather than equal to a zero null.

**"k=8's observed 2 is exactly that width's own null" is withdrawn as over-precise.** It rests on a
single draw of a roughly Poisson(0.93) count; the pooled rate over the ten width-by-direction draws
is 0.8, every draw is in {0,1,2}, and `P(X>=2) = 0.23` at that rate. The defensible pair, which does
not depend on a one-draw per-width estimate: **k=4's 51 is about 61× the pooled per-width null
rate**, and **k=8's 2 is inside the null's observed range at every width and not separated from it.**

## (g) The floors row is a FRACTION, so 27.7% at DLMC k=4

> **Not transferable (amendment 9(4)):** this is a chain207 6-pass figure. The screen is 4 passes, so it does not transfer as a threshold. The reasoning about the functional -- that the floors row is a fraction -- is unaffected and stands.


Recomputing p95 of the two-build spread per width against the published ratio-floor row:
`max/min − 1` agrees to ≤0.008 at four of five widths (0.2573/0.2774/0.2930/0.3085/0.2974 against
0.2607/0.2783/0.2858/0.3010/0.2960) while `|r−1|` is low by 0.02–0.05 everywhere. **So the
functional is the two-sided spread and the gate floor at DLMC k=4 is 27.7%, not 0.28%.** The board's
own arithmetic confirms it: 1258 cells behind on `aa` at k=4 with only 8 clearing the floor is
impossible at 0.28%. The SuiteSparse figures (0.0482 / 0.0355 / 0.0449 / 0.0576 / 0.0848) are left
**open**.

## (h) Downstream reconciliation trap, recorded

The "mean beyond ratio floor" row also totals 53 with a **different** per-width split
(`2/40/11/0/0`) than the sign-only criterion's `0/51/2/0/0`. Anyone reconciling by total will
conflate a floor-gated criterion with a floor-free one. **Match the per-width vector, never the
total.**

## (i) The build's floor assertion already refused once, and was right to

The first build script compiled each object in **its own directory** and the floor pair's `.text`
digests differed (`85c717a1…` against `a9d883350…`, identical `.so` sizes). The build path is
embedded in the object. Rebuilt with all objects compiled at **one fixed absolute path**,
sequentially, with the `.so` moved out after each — the discipline `spmm.h`'s own prefetch comment
records ("two builds at one fixed absolute path with an instruction-identical A/A control beside
them"). **No timing was taken from the refused build.**

One consequence for the floor, which cuts against the registration: `mr0`/`mr0b` are *required*
`.text`-identical while `mrhv` is *required* to differ, so the registered floor **excludes the
code-layout component that the `mrhv`-vs-`mr0` contrast carries**. The true floor for that contrast
is at least the measured one. `layout_artefact_by_width.md` records up to −2.61% with a CI excluding
parity on identical-decision objects. That bites conditions 2 and 4, not 3 — a ≤2.6% artefact cannot
touch a 14.75% target — and condition 2's consequence is the largest, since its failure declares the
whole run confounded.

---

# Amendment 2 — amendment 1's floor-gated primary is withdrawn: it can only return zero
> **Implementation corrected by amendment 10:** condition 3 secondary as first coded
> used a two-sided p95 where this amendment registers the geomean, which read "inside
> the floor" across the whole 7.7-12.4% mechanism range.

> **Superseded by amendments 6(a) and 8:** the gated-cell range quoted in this
> amendment is withdrawn, and replaced by 34.6-54.6% of S-1 (median 45.6% of S-1) over
> the sealed population of 8 at k=4. The membership counts 2/8/34 stay withdrawn; the
> sealed count under the geometric middle is 8.


Still before any timing. Amendment 1 replaced condition 3 because its pass line exceeded the
mechanism's measured range. **The replacement is worse: its primary can only return zero, and that
follows from the gate's definition rather than from which cells the gate picks.**

## The structural argument

The ratio gate admits a cell only if `S − 1 > 0.2783` (amendment 1(g): the DLMC k=4 floor is the
two-sided spread at 27.83%). **So every cell the gate admits needs a gain above 27.8% to reach
parity, by definition of the gate.** The mechanism is 7.7–12.4%. The gate's own floor is **2.2×**
the mechanism's maximum.

Membership, estimator choice and replication accuracy are all irrelevant to that conclusion. My
own reconstruction of the gated set at k=4 returns **2** cells, needing 56.5% and 62.7%:

| S pooled | S aa | S bb | gain needed | deg | key |
|---|---|---|---|---|---|
| 1.6274 | 1.3813 | 1.3637 | **62.7%** | 323.3 | `transformer/l0_regularization/0.5/body_decoder_la…` |
| 1.5652 | 1.5603 | 1.3195 | **56.5%** | 256.0 | `transformer/random_pruning/0.5/body_decoder_layer…` |

A second reader's reconstruction returns 8, needing 38.1–56.3%; this run's sealed analysis headlines
8. **Three reconstructions, three populations, one conclusion** — at 9.2%, 12.4%, 14.75% or even
27.83%, the count is **0 of 2** and **0 of 8**.

**And the criterion's null is also zero**, since an A/A moves nothing. Observed = null = 0. Zero
discriminating power, and a reported 0 would read as *"the arm does not work"* when it means *"the
gate and the mechanism are mutually exclusive."*

**The reconstruction gap is recorded, not papered over**: my 2 against the sealed 8 at k=4 means my
floor replication is not byte-exact, and the per-width vector a second reader gets (`1/8/3/0/0`)
differs from the sealed `1/8/1/0/0` at k=8. Whoever scores this must use the sealed definition, and
the discrepancy is itself worth resolving separately.

## The design fact, stated as a fact rather than a bug

**Floor-clearance demands a large deficit; reachability by a bounded mechanism demands a small one.**
At a 27.8% floor those two sets are disjoint, so no threshold on their intersection can carry
information. That is a property of this corpus's floor at this width, not a flaw in either
requirement — scoring only where the deficit clears the replication pair remains the right instinct.

## The lesson, which is cleaner than the original defect

**Narrowing the population to escape a floor problem RAISED the bar, because the gate selects on
deficit size and the bar IS deficit size.** The narrowing and the threshold moved in the same
direction. **When a criterion is unreachable, check whether the proposed narrowing moves the bar in
the same direction as the narrowing itself.**

## Condition 3, replaced a second time — change the question, not the population

**3-PRIMARY (promoted from amendment 1's 3c).** Report the measured gain at float32 k=4 and the
implied parity count **as a curve, with no threshold**. This is measurable, it is what the data can
answer, and it reports a real improvement that does not reach parity as exactly that.

**3-SECONDARY (the answerable displacement question, in condition 4's shape).** Does the arm move
the k=4 geomean by more than **the geomean's own floor at that n**, computed from the `mr0`/`mr0b`
pair? Floor-scaled, answerable, and it does not require the deficit to close.

**3-DESCRIPTIVE.** The parity counts over the 8 gated cells and the 51 sign-only cells, reported
beside their two-build flip counts, as **descriptive counts and not as a pass/fail test**.

**"Does the arm CLOSE the deficit" is unaskable for a 10% mechanism against a 38–63% gap, and is
withdrawn as a question.** A 7.7–12.4% gain on the 51 would be a real improvement; the curve says so
and no threshold needs to.

## Two further items

**The row-dispersion stratifier is not computable from the ledger.** Amendment 1(e) moved condition
4's stratification to row-length coefficient of variation, which is the right covariate — but the
sealed CSVs carry only `build,slice,key,k,rows,nnz,deg,arm,ms,batch`, and `deg` is `nnz/rows`. There
is no row-length distribution in the ledger, so the CV requires re-reading the source `.smtx`
matrices. **A correct stratifier and an available stratifier are different questions, and amending
the condition answered only the first.** Noticed now rather than when the scoring script runs; the
reader is to be written before scoring, against `~/dlmc` on this host and `/scratch/bobbyy/dlmc` on
redwood, and its row counts must be checked against the ledger's `rows` column per key so a
mismatched corpus cannot read as a dispersion result.

**The firing check's one blind direction, registered rather than fixed.** If `mrhv` and `mrhv4` do
not differ at k=4, that conflates *non-firing* with *ROWS=2 and ROWS=4 performing identically*. The
`.text` assertion proves the objects' code differs, not that execution differed. It **fails closed**
— discarding a readable run rather than accepting a confounded one — so it is conservative in the
right direction. Also recorded: `mrhv` against `mr0` could **not** serve as the firing check even
though `mr0` never admits multi-row, because that contrast *is* the effect, so a null would be
indistinguishable from "fired and did not help". A firing check has to vary a parameter the effect
does not depend on, and `multirow_ok` is ROWS-independent beyond `multirow > 1`, so admission is
identical between `mrhv` and `mrhv4`. That is what makes the check valid.

## Build complete, all assertions passed

Four objects at one fixed absolute path: `mr0` and `mr0b` **`.text`-identical** (the floor), `mrhv`
differing from ship by **+4385** instruction lines, `mrhv4` differing from `mrhv` by **−75**.

---

# Amendment 3 — the two criteria select nearly DISJOINT populations, and my gate and my report used different reductions
> **Superseded by amendments 6(a) and 8:** the gated-cell range quoted in this
> amendment is withdrawn, and replaced by 34.6-54.6% of S-1 (median 45.6% of S-1) over
> the sealed population of 8 at k=4. The membership counts 2/8/34 stay withdrawn; the
> sealed count under the geometric middle is 8.


Still before any timing.

## (a) "50 of 51 sit inside the floor (corrected, amendment 3(a))" is wrong — it is 50 of 51, and the sets are not nested

That figure came from a second reader as `51 − 8`, which is arithmetic rather than set algebra, and
I repeated it in amendment 1(b) and amendment 2. **Both sets measured directly, per width:**

| k | sign-only (all 12 behind) | floor-gated (both builds) | intersection |
|---|---|---|---|
| 2 | 0 | 0 | 0 |
| **4** | **51** | **2** | **1** |
| 8 | 2 | 0 | 0 |
| 16 | 0 | 0 | 0 |
| 32 | 0 | 0 | 0 |

**The two criteria share one cell on the whole board.** So **50 of the 51** sign-only cells sit
inside the floor, and half of my gated set is not in the 51 at all. **Corrected in all three places
it appeared.**

The consequence for amendment 1 is bigger than the arithmetic: **its narrowing was a population
SWAP, not a subsetting.** This document's entire motivation section is built on the 51, and
amendment 1's condition 3a then scored a set almost disjoint from it. Amendment 2 withdrew 3a for
unreachability, which stands; **the better reason is that it was a different deficit.**

## (b) The floor gate selects on VARIANCE, not on slowness

| population | n | median CV of the 12 per-pass ratios | median S |
|---|---|---|---|
> **Not transferable (amendment 9(4)):** this is a chain207 6-pass figure. The screen is 4 passes, so it does not transfer as a threshold. The CV contrast is the finding here and still holds; only the absolute levels are 6-pass.

| floor-gated | 2 | **26.8%** | 1.5652 |
| sign-only | 51 | **16.0%** | 1.1510 |
| all k=4 (1200-key sample) | 1200 | **15.9%** | — |

**The sign-only 51 sit at the corpus's own CV; the gated cells sit well above it.** A magnitude
threshold applied to a per-cell median, in a corpus whose per-cell CV is already ~16%,
preferentially admits unstable cells — requiring consistency and requiring magnitude pull in
opposite directions once the floor is comparable to the effect.

Two corrections to how strongly this can be stated. On **my** reconstruction the gated set is n=2, so
26.8% is two cells and the second reader's 37.5% over n=8 is the better-powered figure; and their
"every gated cell above 2.4× per-pass spread" is **not** distinguishing — the sign-only 51 also reach
3.27×, 2.81×, 2.71×. The **CV contrast** is the finding; the spread list is not. Their own flag that
the "0 of 51 ever ahead of MKL" column is true by construction is correct and I am not citing it as
evidence.

**This cuts in favour of the original target.** The 51 — at the corpus CV, median S 1.1510, never
once ahead of MKL across 12 readings — look far more like a genuine width-specific kernel deficit
than the gated cells do. **At this noise level the intersection of "large deficit" and "stable" is
one cell in 3820, which is itself the answer**: the board contains no population that is both, so
the honest target is the stable modest-deficit set with the gain reported as a curve. That is what
amendment 2 already arrived at, and it is better supported than amendment 2 argued.

## (c) A real defect in my own table, which is NOT the one it was reported as

A second reader read the deg-323.3 row as "S_aa 1.6274, S_bb 1.3813, pooled 1.3637" and concluded
all three were single-pass readings with two of them both from `aa`. **The column order was
transposed in the reading.** That cell's readings:

```
aa 6 passes: 4.5076 1.9379 0.7385 1.0066 1.3813 1.6274
bb 6 passes: 2.4504 1.1270 1.3637 2.6324 1.2019 1.8642
pooled(12) = 1.6274    aa(6) = 1.3813    bb(6) = 1.3637
```

The header is `S pooled | S aa | S bb` and the row is `1.6274 | 1.3813 | 1.3637`. **The labels are
correct.** And the general rule offered — "a reduction of twelve values should not BE one of the
twelve" — does not hold here: `pct(·, 0.5)` is a **lower-median order statistic**, copied verbatim
across these harnesses precisely so the convention cannot drift, and an order-statistic median of an
even-sized sample **is** by construction one of the sample values.

**But there is a real defect underneath, and it is mine.** The gate was applied to the **per-build**
medians (1.3813 and 1.3637, both clearing 0.2783) while the "gain needed" column reported the
**pooled** median (1.6274 → 62.7%). **Gate and report must use the same reduction.** Per build the
cell needs **38.1%** and **36.4%**, not 62.7%.

So **"a 38–63% gap" is withdrawn**: its upper endpoint came from a pooled reduction where the gate
used per-build ones. Corrected, the gated cells need **36–57%** against a 7.7–12.4% mechanism — which
changes no conclusion in amendment 2, since its argument is definitional (the gate admits only cells
needing more than 27.8%), but the quoted range was wrong.

**Registered for the scoring path**: every per-cell figure in the run must state which reduction
produced it, and any gate and the number reported beside it must use the **same** one. That is the
defect's transferable form, and it is a different one from what was reported.

## (d) My reconstruction still differs from the sealed analyser, and the sealed definition governs

> **Resolved by amendment 8:** the sealed definition does govern and IS recoverable --
> `an_c207.py:181-189`, `both` beyond the ratio floor over all cells. Transcribing it
> reproduces eleven published figures to the digit.


My gated count at k=4 is **2**; the sealed analysis headlines **8**; a second reader reconstructs
**8**. At k=8 I get 0, they get 3, the sealed says 1. **Three reconstructions, three answers.**
Scoring uses the sealed definition, and the discrepancy is logged as unresolved — it is not
explained by the reduction mismatch in (c), which affects the reported gain and not gate membership.

## (e) Three notes for the driver, all pre-run

- **The gain curve's x-axis anchor is now measured.** The 51's S distribution: min 1.0550, p25
  1.1161, median 1.1475–1.1510, p75 1.2048, max 1.5627. **If the measured gain lands at 9–12% the
  curve reads 4–13 of 51, and that is the expected result rather than a disappointment.**
- **Two different n, therefore two different floors.** The displacement question uses **n=51**; the
  no-regression pool uses **n≈3769**. Using one floor for both would be the pooled-against-per-cell
  error in a new place.
- **Condition 2 is now the only original condition still standing, so it carries more weight** — and
  k=2 is the width where `layout_artefact_by_width.md` records that the artefact's width ordering
  does **not** generalise across host/dtype/corpus. **A condition-2 trip must have the layout
  artefact ruled out before it is read as non-neutrality.**

---

# Amendment 4 — `pct` has no single index convention: it flips with n, and that is the whole 2-against-8
> **Superseded by amendments 6(a) and 8:** the gated-cell range quoted in this
> amendment is withdrawn, and replaced by 34.6-54.6% of S-1 (median 45.6% of S-1) over
> the sealed population of 8 at k=4. The membership counts 2/8/34 stay withdrawn; the
> sealed count under the geometric middle is 8.


Still before any timing. This is an instrument defect in the canonical estimator, not a choice.

## The cause: banker's rounding inside one function

```python
def pct(v, q):                      # an_f64_arms.py:105-109, copied into three other harnesses
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]
```

`round()` in Python is **round-half-to-even**. At `q=0.5` the index is `round((n-1)/2)`, so for even n:

| n | (n−1)/2 | `round` | index is the | |
|---|---|---|---|---|
| 2 | 0.5 | 0 | **lower** median | |
| 4 | 1.5 | 2 | **upper** median | |
| 6 | 2.5 | 2 | **lower** median | ← inside a cell (6 passes per build) |
| 8 | 3.5 | 4 | **upper** median | |
| 12 | 5.5 | 6 | **upper** median | ← pooled across builds |
| 68 | 33.5 | 34 | **upper** median | |
| 78 | 38.5 | 38 | **lower** median | |
| 152 | 75.5 | 76 | **upper** median | |
| 298 | 148.5 | 148 | **lower** median | |

**One function, and its convention is a function of the sample size.** The deg-323.3 row is explained
exactly: `aa(6)` and `bb(6)` are **lower** medians (1.3813, 1.3637) and `pooled(12)` is the **upper**
(1.6274). Two conventions in one row, from one call site.

**So my sentence that `pct` was "copied verbatim across these harnesses so the convention cannot
drift" is falsified by the arithmetic inside the function it describes.** Copying it verbatim pinned
the *code* and not the *convention*. A second reader guessed a second code path; the cause is
narrower and worse — there is one path and it is inconsistent with itself.

## The size, on every number that depends on it

**The gate count at k=4, which is where it is catastrophic:**

| convention | gated cells at k=4 |
|---|---|
| `pct` as published (= lower here) | **2** |
| averaged median | **8** |
| lower median | 2 |
| upper median | **34** |

**A factor of 17 across conventions, on gate membership.** This closes item (d) of amendment 3 and
closes it the opposite way to my reading there: the mismatch moved **membership**, not only the
reported gain. It also identifies the sealed reduction as **averaged-median-like** — averaged
reproduces the sealed headline at four of five widths, with k=8 (3 against 1) still unexplained.

**Why it bites here and nowhere quiet:** an order-statistic median's index convention matters in
proportion to the sample's dispersion. These cells have per-pass spreads of 2.5–6×, so the 3rd and
4th of six sit **17.8%** apart on the deg-323.3 cell, and a threshold between them flips the verdict.
On a low-dispersion cell the three conventions are indistinguishable, which is exactly why this was
never worth naming until a threshold was applied to it.

**The narrow rule: when a threshold is applied to an order statistic of a small, high-dispersion
sample, the index convention is PART OF THE THRESHOLD.** Re-running the gate under both conventions
is a one-line sensitivity check and here it spanned 2 to 34.

**On the width-contrast conclusions, the sensitivity is small.** Recomputing the k=4 contrast under a
single convention throughout shifts it by at most **1.91 pp** (4-vs-32: 40.96% published against
39.04% all-averaged; 4-vs-8: 1.39 pp). Against separations of 9.7–34.6 pp, no conclusion moves.
Amendment 31's bootstrap groups also mix conventions (n=78 lower, n=68 upper, n=152 upper, n=298
lower), but at those n the difference is one adjacent order statistic out of tens to hundreds.

## The reassurance, and it is structural

**The sign-only 51 uses no median at all** — "behind MKL in all 12 readings" is a conjunction over raw
readings. Recomputed: **51, convention-free by construction.** So this document's motivating
population, and amendment 3's conclusion that the 51 are the better-conditioned target, are
**untouched by any of this.**

That is not luck. **The criterion that survived four rounds of review is the one with no reduction
step in it**, and the criteria that kept failing — a magnitude threshold on a per-cell median — are
the ones whose answer depended on an estimator's index.

## What I am changing and what I am not

**Changed:** the scoring path for this run will use one explicitly named convention (averaged median,
`statistics.median`), computed in a new function, with the convention printed in the output header so
it cannot be inferred wrongly from a label.

**Not changed:** `pct` in `an_f64_arms.py`, `an_floor2x2.py` and `an_f64_vs_mkl.py`. Those are
pre-registered estimators in registered screens, and correcting a pre-registered estimator is
Bobby's call, not a verifier's. **Flagged for Bobby with the measured size**, rather than silently
amended. The defect does not invalidate those screens' published conclusions — the ≤1.91 pp
sensitivity above is the bound — but every future threshold applied through `pct` is exposed to it.

## Two corrections accepted, and one range fixed again

- **The 38.1% coincidence is a coincidence**: two different cells under two different reductions
  (deg-314.9 at pooled-averaged 1.3808; deg-323.3 at aa-lower 1.3813). Not the same cell.
- **"The gated cells need 36–57%" was mixed too** — in the very commit that withdrew the previous
  range for being mixed. A cell must clear parity on **both** builds, so the requirement is
  `max(aa, bb)`: **38.1–56.0%**. Third statement of this range, and the lesson is that **an endpoint
  only has to come from somewhere**, which is why ranges are where mixed reductions keep surfacing.
- **The corpus CV is 20.0% over all 3820 cells, not 15.9%** — mine was a 1200-key head-of-list sample
  and is withdrawn as biased by the draw. So the sign-only 51 at 16.0% sit **below** the corpus, not
  at it: the consistently-behind population is **quieter** than the corpus. That strengthens
  amendment 3(b)'s conclusion rather than weakening it. And the CV needs its own functional named for
  the same reason the median did — 26.8% published against 41.4% recomputed as stdev-over-mean.

---

# Amendment 5 — the convention's exposed set in closed form, and the one n in this run that lands in it
> **Superseded in part by amendment 6(c):** the registered reduction for the scoring
> path is the geometric middle `exp(median(log))`, not `statistics.median`. The p95 floor
> registered here is unchanged.


Still before any timing. Verified independently rather than taken on report; all six checks below
reproduce exactly.

## The closed form, which predicts any n instead of enumerating

At `q=0.5`, `pct`'s index is the **upper** median iff `n ≡ 0 (mod 4)` and the **lower** median iff
`n ≡ 2 (mod 4)`. **Zero mismatches over every even n in [2, 4000].** Odd n is unambiguous —
`q*(n−1)` is an integer, zero fractional indices over every odd n in [1, 4001].

Published population sizes: 2 LOWER, 4 UPPER, 6 LOWER, 8 UPPER, 12 UPPER, 68 UPPER, 78 LOWER,
108 UPPER, 152 UPPER, 298 LOWER, 328 UPPER.

## The consequence that matters more than the inconsistency: it flips when the population changes by TWO

| | |
|---|---|
| n=68 UPPER | → n=70 LOWER |
| n=78 LOWER | → n=80 UPPER |
| n=108 UPPER | → n=110 LOWER |
| n=298 LOWER | → n=300 UPPER |
| n=328 UPPER | → n=330 LOWER |

Every one of those is a published population size sitting one step from a boundary. **A filter edit,
a dropped key or a re-grouping silently changes the estimator.**

**One line for the record on amendment 31.** Its replication gate refused at n=328/108 and passed at
n=298/78 — and 328 and 108 are **UPPER** while 298 and 78 are **LOWER**, so **the three attempts
were not estimator-identical**. That does not move amendment 31's result (at n=78/152/298 the
exposure is one adjacent order statistic out of tens to hundreds, already bounded at ≤1.91 pp), but
the whole purpose of that gate was to establish the grouping, so it belongs in the record.

## The exposure is bounded by q, derivably — and the floor side is immune

At `q=0.5` **every** even n hits the half-integer, which is why the medians moved. At `q=0.95` it
requires `0.95(n−1)` to be a half-integer, i.e. **n ≡ 11 (mod 20)**. Verified: all 1000 such n in
[2, 20000], and no others.

| quantity | n | n mod 20 | `0.95(n−1)` frac | |
|---|---|---|---|---|
| DLMC board floor | 3820 | 0 | 0.05 | unambiguous |
| full-board floor | 19100 | 0 | 0.05 | unambiguous |
| no-regression pool | 3769 | 9 | 0.60 | unambiguous |
| **displacement n (the NEW condition)** | **51** | **11** | **0.50** | **EXPOSED** |

> **Not transferable (amendment 9(4)):** this is a chain207 6-pass figure. The screen is 4 passes, so it does not transfer as a threshold. The immunity argument of this amendment concerns the `pct` index convention and is unaffected.

**So the p95 ratio-floor row and the 27.7% gate threshold are provably unaffected.** That is the
reassurance amendment 4 did not have: the floor side — the thing everything was compared *against* —
is immune by arithmetic, not by coincidence. Only the per-cell medians on the other side of the
comparison moved.

## The one forward hazard, and it is exactly where this run is headed

**n=51 is 11 (mod 20).** `0.95 × 50 = 47.5`, a half-integer resolved to index 48 by the rounding mode
rather than by any stated rule. And the replacement secondary condition registers the displacement
floor **at n=51**. So of every population in this run, the single one whose p95 falls in the exposed
class is the one the new condition uses.

The value is determined, so this is not yet an error — but it is determined by a rounding mode
instead of a declared convention, on the one n where that happens to matter.

**Registered now, before the run:** the n=51 displacement floor is computed as the **upper** order
statistic at index `ceil(0.95 × (n−1)) = 48`, stated explicitly, and the scoring output prints both
the index and the convention beside the value. The 0.5 quantiles throughout the scoring path use
`statistics.median`, which averages the two middle values and so **has no index to choose** — naming
it is a real fix and not a different arbitrary choice.

## The design rule, promoted, because it would have pre-empted three failed conditions

**When a threshold must be applied, prefer the formulation that reduces LAST or not at all.**

An estimator's index cannot move a conjunction over raw readings. That is why the sign-only 51
recomputes identically under every convention while every magnitude-threshold-on-a-per-cell-median
criterion in this document failed review — conditions 3, 3a, and the gain-needed column, in one
stroke. The surviving criterion's robustness is a property of **where the reduction sits**, not of
the care taken writing it.

## Scope limits, recorded

The `n mod 4` rule is derived for `q=0.5` specifically, where `q*(n−1)` is a half-integer for all
even n. For an arbitrary quantile the exposed set is whatever n makes `q*(n−1)` land on a
half-integer — the general form of the `n ≡ 11 (mod 20)` result — and **only q=0.5 and q=0.95 have
been checked.** Any other quantile used anywhere needs its own modulus computed.

And the width-contrast blast radius (≤1.91 pp) is from a different ledger and has not been
independently recomputed; it is mine, not confirmed.

# Amendment 6 — the gated-cell range is withdrawn, not corrected; and the scoring path uses the geometric middle
> **(a) IS ITSELF WITHDRAWN by amendment 8.** The range was always reproducible; I
> withdrew it on a wrong conversion of my own (`S = 1/(1-g)` where my column was
> `1 - 1/S`). The sealed predicate is `an_c207.py:181-189` and reproduces eleven
> published figures. (b) and (c) stand.


Written 2026-10-02, with the four-build screen running and no reading from it yet scored.

### (a) The gated-cell range and the gate membership counts are WITHDRAWN as not reproducible

Amendments 2 and 3 quote a range for "gain needed" over the floor-gated cells: first
38–63%, then 36–57%, then 38.1–56.0% as max(aa,bb). A peer session reading the sealed
`an_c207.py` reported it should be 34.6–54.6% (median 45.6%) under that file's estimator.
Amendment 4 quotes gate membership at k=4 as 2 (lower median) / 8 (averaged) / 34 (upper).

I tried to reproduce all of it from the sealed chain207 DLMC ledger before quoting the
peer's correction. **I could not reproduce any of the five numbers.** Reconstructing the
gate as "a cell of the sign-only 51 whose per-build reduction clears the 0.2783 floor,
admitted on max(aa,bb)", over the 12 readings per cell at k=4:

| reduction of the 6 readings per build | gate n | gain needed to reach parity |
|---|---|---|
| lower order statistic | 6 | 22.0% .. 41.4%, median 28.8% |
| geometric middle, `exp(median(log))` | 9 | 23.0% .. 42.2%, median 27.4% |
| arithmetic middle, `statistics.median` | 9 | 23.1% .. 42.2%, median 27.8% |
| upper order statistic | 23 | 21.8% .. 43.1%, median 26.7% |

Dropping the sign-only restriction and gating the whole k=4 population gives n=200 and
21.8–45.8%. No variant I can construct lands on 2/8/34 or on any of the quoted ranges,
and the ranges are all roughly 15 percentage points below every figure ever published
here. A cell needing 38% gain has S = 1.61, i.e. a deficit of 61% — more than twice the
floor the gate is supposed to apply, so the published lower endpoint is not consistent
with a 0.2783 gate at all.

What this means: the definition that produced the published range is not recorded
anywhere I hold, and three successive "corrections" to it — two mine, one a peer's —
were each computed from a different unstated procedure. **The range and the membership
counts are withdrawn outright rather than corrected to a fourth value.** Quoting
34.6–54.6% here would repeat the exact failure amendment 3 recorded, where I passed on a
peer's "43 of 51" while holding the data that said 50 of 51.

The qualitative claim amendment 4 rests on does survive, and is reproducible: the choice
of middle moves gate membership by a factor of 3.8 (6 to 23) on a fixed population and a
fixed floor. That is the point amendment 4 was making, and it does not need the
unreproducible counts to stand.

What is unaffected: the sign-only population is exactly 51, reproducible, and uses no
median at all. Nothing in the live scoring path reads the withdrawn range.

### (b) `an_c207.py` is immune to the `pct` convention defect by construction

Amendment 5 registered which analysers the `pct` half-integer defect reaches. The sealed
board's own analyser is not one of them. `an_c207.py:59` is

    p95 = lambda d: sorted(d)[int(0.95 * (len(d) - 1))] if d else float("nan")

— truncation, not `round`, and it never calls `pct`. So that ledger is immune because of
how it is written, not because its populations happen to sit at n ≡ 0 (mod 20). It should
not be on the list of files for Bobby's `pct` decision; that list is
`an_f64_arms.py`, `an_floor2x2.py`, `an_f64_vs_mkl.py`.

### (c) The scoring path uses the geometric middle — supersedes amendment 5

Amendment 5 registered that "the 0.5 quantiles throughout the scoring path use
`statistics.median`". That is changed, before any reading is scored, to

    red = lambda v: math.exp(statistics.median([math.log(x) for x in v]))

for every reduction of a ratio. Three reasons, in order of weight:

1. S is a ratio. Its natural scale is logarithmic: the middle of {0.5, 2.0} is 1.0, not
   1.25, and only the geometric middle gives an estimator that is invariant when the
   ratio is inverted. Every quantity this arm is scored on is a ratio.
2. The board this arm's result will be set beside reduces the same way
   (`an_c207.py:57`). Comparing a geometric displacement against an arithmetic baseline
   is a units error of the same family as amendment 30's headroom-vs-raw mistake.
3. It is measurable, not free. On the gated cells above, the two middles differ by 0.4 pp
   in the median (27.4% vs 27.8%) — far below the effect under test, but not zero, so the
   choice has to be stated rather than defaulted.

The p95 floor registered in amendment 5 is unchanged and remains the upper order
statistic at index `ceil(0.95 x 50) = 48`, printed with its index and convention beside
its value. The sign-only 51 remains convention-free: it reduces nothing.

# Amendment 7 — condition 4's row-dispersion stratifier is withdrawn: it sorts cells by their own noise floor
> **Mechanism confirmed and two claims corrected by amendment 9.** The floor spread is
> measured on the size axis in 9(1). The reader's guard covered only its redundant
> column (9(2)), and its "only source for mean degree" claim is wrong (9(3)).


Written 2026-10-02 with the screen running and nothing from it scored. The instrument is
`harnesses/pfstream/dlmc_rowdisp.py`, committed with this amendment; every number below
is from the sealed chain207 DLMC ledger plus that reader, which read 3820 of 3820 ledger
keys with zero disagreement on rows or nnz.

Amendment 1(e) replaced condition 4 because its functional "stratified on mean degree
where the header names row DISPERSION". The replacement stratifies on the coefficient of
variation of the per-row nonzero count. Having now measured that quantity, the
replacement is worse than the thing it replaced, for three reasons in order of weight.

**1. The strata have different noise floors, by a factor of 2.7.** Condition 4 compares a
displacement to an A/A floor. Over the 3769 cells it covers, split at the corpus CV
terciles, with the floor as the p95 of the within-cell aa-vs-bb |log| displacement:

| tercile | n | geomean S vs MKL | median mean degree | A/A floor p95 |
|---|---|---|---|---|
| low cv (<= 0.132) | 1239 | 1.0341 | 221.3 | 35.47% |
| mid cv (0.132, 0.310] | 1261 | 0.9259 | 102.4 | 27.40% |
| high cv (> 0.310) | 1269 | 0.8701 | 23.9 | 12.91% |
| pooled | 3769 | 0.9232 | | |

A single pooled floor applied across these strata is too lenient in one and too strict in
another, in a criterion whose only operation is a comparison against a floor. This is the
A/A-floor-is-an-inverted-U-by-size effect arriving through a stratifier that was not
chosen for size: low CV means high mean degree means the noisy middle of the size range.

**2. The stratifier predicts the outcome.** Spearman rho between row CV and S at k=4 is
**-0.622** over all 3820 scored cells. So the strata differ in baseline S by 16.4
percentage points of level (1.034 to 0.870), and a per-stratum displacement is confounded
with the stratum's own level rather than isolated from it.

**3. It is substantially the stratifier it replaced.** Spearman rho between row CV and
mean degree is **-0.780**. Row dispersion in this corpus is largely mean degree inverted,
so amendment 1(e) swapped a rejected variable for 61% of the same variable.

### Condition 4, replaced a second time, with no stratification at all

The defence stratification was meant to provide -- that a no-regression average is not
hiding a hurt subpopulation -- is better had from the pairing that is already in the
design, because a within-cell paired ratio cannot carry the cell's level or its stratum's
level into the answer. That is the same construction that made the k=4 half-vector claim
survive every correction, and it needs no strata.

Condition 4 is now scored as, over all cells not in the sign-only 51:

1. the geometric middle of the per-cell paired displacement, arm object against control
   object, paired by pass index within the cell, against the paired A/A floor computed by
   the identical reduction on the two byte-identical builds;
2. the count of cells displaced worse than the floor, beside the count the floor's own
   definition implies under the null -- a p95 floor is exceeded by 5% of cells when
   nothing is happening, so this criterion reports its own null arithmetically rather
   than needing a separate run;
3. the single worst cell, named, with its matrix dimensions. A regression is a finding to
   report, not a shape to tune around, so the criterion has to be able to produce one.

Any stratified table stays in the readout as description only, and each stratum then
carries its **own** floor from the column above -- never the pooled 0.9232-population
figure. No threshold is applied to a stratified number.

The row-dispersion reader is kept and committed regardless, for row_cv, max_over_mean and
empty_rows, which the descriptive table needs. (The claim that it is the only source for
per-key mean degree is **wrong and corrected in amendment 9(3)** -- the ledger's own `deg`
column already carries it. Amendment 9(2) also corrects what its guard covers.) Its guard earned its place on the first run
by catching that chain207's keys carry a leading `dlmc/` component that a path relative
to `~/dlmc` does not -- without it the run would have produced 3846 perfectly plausible
dispersion values for a population overlapping the ledger in zero keys.

# Amendment 8 — amendment 6(a) is itself withdrawn: the range was always reproducible and I withdrew it on a unit error of my own

Written 2026-10-02, still before any reading from the screen is scored.

Amendment 6(a) withdrew the gated-cell range as "not reproducible from the sealed
ledger". That withdrawal was wrong. The range reproduces exactly, and so do ten other
published figures. The reason I concluded otherwise was a conversion error in the
amendment's own load-bearing sentence.

### The error

Amendment 6(a) argues: *"A cell needing 38% gain has S = 1.61, i.e. a deficit of 61% --
more than twice the floor the gate is supposed to apply, so the published lower endpoint
is not consistent with a 0.2783 gate at all."*

1.6129 is 1/(1 - 0.38). That is the inverse of the convention my own table was computed
in. My column was `1 - 1/S`, so a cell needing 38% by that column has **S = 1.38**, and
1.38 > 1.2783 clears the gate comfortably. The entire inconsistency was manufactured by
applying the wrong direction of my own conversion. The inference also would not have
followed in the other direction: a larger deficit clears a floor more easily, not less.

### The two tables were never in conflict; they were in different units over different populations

The sealed criterion is recorded, at `an_c207.py:181-189`:

    s = [c for c in ok if Sa[c] > 1 and Sb[c] > 1 and f(c)]
    HEAD = run("both beyond RATIO floor  <- THE HEADLINE",
               lambda c: dev(Sa[c]) > FR[c[1]] and dev(Sb[c]) > FR[c[1]])

Transcribing it reproduces, from the sealed CSVs and nothing else: 19100 complete cells
(the file's own `EXPECT_CELLS`), the per-width ratio floors
0.2607 / 0.2783 / 0.2858 / 0.3010 / 0.2960, the headline count 1/8/1/0/0 = 10 cells over
10 matrices, and the board medians, as `median aa`,
0.8864 / 0.9231 / 0.8849 / 0.7907 / 0.7409. Sixteen independent values to the digit. The
procedure is identified, not conjectural.

The medians took a second attempt, and the reason is worth recording because it is the
third instance of one error. `an_c207.py:169-173` prints `red(ra)` and `red(rb)` -- the
two builds in **separate columns** -- and never forms a median across them. My first
reconstruction pooled the two builds into one list, which is a third statistic, and read
0.8854 / 0.9248 / 0.8854 / 0.7926 / 0.7412: wrong in the third digit, plausible in every
digit. Amendment 3(c) recorded the same error in the other direction, where the gate used
per-build medians while "gain needed" reported the pooled one. A reduction across builds
is not a reduction within one, and nothing about either number's appearance says which it
is.

My amendment-6 table differed in two ways, both of which I had stated and neither of
which I accounted for: it restricted to the sign-only 51, and it admitted on
**max(aa, bb)** where the sealed predicate requires **both**. And then the remaining gap
was units: I reported `1 - 1/S` and the figure I was checking against was `S - 1`. The
same eight endpoints appear in both tables once converted.

### The reinstated figures, with the unit inside the range

The sealed k=4 gated cells, n=8, each cell's S being the geometric middle of the two
builds:

| S | S - 1 | 1 - 1/S |
|---|---|---|
| 1.5462 | 54.6% | 35.3% |
| 1.5202 | 52.0% | 34.2% |
| 1.4695 | 46.9% | 31.9% |
| 1.4645 | 46.4% | 31.7% |
| 1.4470 | 44.7% | 30.9% |
| 1.3929 | 39.3% | 28.2% |
| 1.3741 | 37.4% | 27.2% |
| 1.3458 | 34.6% | 25.7% |

**34.6-54.6% of S-1, median 45.6% of S-1.** Equivalently 25.7-35.3% of 1-1/S, median
31.3%. Amendments 2 and 3's "38.1-56.0%" remain withdrawn as a third thing, but the
figure that replaces them is this one, not nothing.

My own population -- cells of the 51 clearing the floor on max(aa,bb) -- is a sensible
set and its table was also correct: 22.0-41.4 / 23.0-42.2 / 23.1-42.2 / 21.8-43.1 percent
of 1-1/S for the lower / geometric / arithmetic / upper middles, at n = 6 / 9 / 9 / 23. It
is simply not the sealed headline's set, and the two differ because the gated set and the
51 are nearly disjoint -- amendment 3(a)'s finding, recurring.

### The unit convention, registered for everything downstream

**A range is the most unit-ambiguous object there is**, because both endpoints read as
plausible under either convention, so neither endpoint looks wrong. From here, every
ratio-derived percentage in this document and in `score_mrhv.py` names its unit *inside*
the number: "45.6% of S-1", never a bare "45.6%".

Which unit is correct is decidable and it is not a matter of taste. A mechanism
delivering a speedup of factor g brings a cell to parity exactly when S <= g, so the
column that may be compared against a mechanism stated as a factor is **S - 1**.
Amendment 6 set `1 - 1/S` ranges against "a 7.7-12.4% mechanism" that was verified as
factors -- a mismatch across the word "against", and the third appearance of the same
conversion in one commit. In `1 - 1/S` units a x1.124 speedup is a 11.03% time reduction,
not 12.4%. This does not move the 0-of-N conclusion of amendment 2, which is definitional
in either unit.

### One self-contradiction, owned

Amendment 6(c) adopts `red = exp(median(log v))` as "`an_c207.py:57`" while amendment
6(a), in the same commit, says the sealed definition "is not recorded anywhere I hold".
The first sentence answers the second. I read the file for one purpose and did not carry
what I had just read across to the other half of the same commit. 6(c) stands; 6(b)
stands; 6(a) is replaced by this amendment.

### What is unaffected

The qualitative claim amendment 4 rests on needs none of this and is unchanged: the
choice of middle moves gate membership by a factor of 3.8 on a fixed population and a
fixed floor, 6 to 23 over the 51; in the sealed population it is 8 under the geometric
middle. The sign-only 51 remains convention-free and reduces nothing. No live scoring
path reads any of the withdrawn or reinstated figures.

# Amendment 9 — the reader's guard certified its one redundant column; and no chain207 floor transfers to this run

Written 2026-10-02 with the screen still on its first pass. Items (1) and (4) arrived from
a peer session; (2), (3) and (5) are measured here.

### 1. Amendment 7's mechanism, now measured on the size axis instead of inferred

Amendment 7 attributed the CV terciles' 2.7x floor spread to the known inverted-U of the
A/A floor by matrix size, arriving through a stratifier not chosen for size. That was an
inference across axes. Measured directly -- p95 of the within-cell aa-vs-bb displacement
at k=4 over all 3820 cells, banded by nonzero count:

> **A 6-pass chain207 floor table, cited as a MECHANISM and not as a bar (amendment 9(4)):** these levels do not transfer as thresholds to the 4-pass screen. What transfers is the SHAPE -- the inverted U and which band peaks.

| nnz band | n | p95 floor, of S-1 |
|---|---|---|
| < 1e4 | 748 | 4.14% |
| 1e4 - 1e5 | 1836 | 22.89% |
| 1e5 - 1e6 | 1202 | 39.27% |
| > 1e6 | 34 | 13.20% |

A clean inverted U peaking at 1e5-1e6 nonzeros, and the CV terciles map onto it through
low CV <-> high mean degree <-> larger matrices. The withdrawal now rests on a
measurement rather than on a cross-axis argument.

### 2. The dimension guard certified the one column that carries no information

`dlmc_rowdisp.py` checked each key's rows and nnz against the ledger and called that
coverage. But once rows and nnz are validated and the pointer is checked to sum to nnz,
`mean_deg` is nnz/rows **exactly** -- a function of two already-validated scalars. The
three columns condition 4 actually acted on (row_cv, max_over_mean, empty_rows) all
depend on the *shape* of the row-length distribution, which nothing independent
validates. So the guard's hazard statement was right and its coverage claim was inverted:
it certified the redundant column and left the load-bearing ones open.

The fix was already inside the file. `cols` was parsed from the header and never used for
anything but output. Every row's nonzero count must lie in [0, cols], since a row's
entries occupy distinct columns -- a free constraint on the distribution rather than on
its mean. Both bounds are now enforced in `row_counts`.

**Demonstrated on two dimensionally-correct corruptions**, each rewriting one key's row
pointer while preserving its rows and nnz exactly, so the ledger check cannot see either:

- *caught.* One row given 600 nonzeros in a 512-column matrix. The reader reports
  `DIMENSION MISMATCH against the ledger: 0` -- the old guard was blind -- and
  `UNREADABLE or internally inconsistent: 1` with the reason, and refuses to write.
- *not caught, and this is the residual gap.* Every row set to the same length, 25 or 26.
  Valid under every check in the file, and wrong: that key's row CV becomes 0. Every
  guard passes and the corpus quantiles move in the fourth digit.

The column bound is weaker than an independent record of row lengths and is not a
substitute for one. No such record exists in this project, and the honest statement is
that the shape columns rest on the corpus files being uncorrupted. Amendment 7's
conclusion does not depend on them: it is a withdrawal, and a withdrawal does not need
the withdrawn numbers to be right.

Separately, `dispersion()` returned `empty_rows = 0` in the `mean == 0.0` branch, where
every row is empty and the answer is n. Unreachable on this corpus -- the smallest key
has 6 nonzeros over 64 rows -- but it would have read "no empty rows" on the one input
where all of them are. Fixed.

### 3. Amendment 7's provenance claim about the reader is wrong

Amendment 7 says the reader is "the only source in this project for per-key mean degree
and empty-row counts checked against an independent record". Mean degree was already in
the ledger: the chain207 CSVs carry a `deg` column, and over all 3820 keys
max |deg - nnz/rows| = 5.00e-03 with only 3 keys above 0.005 -- two-decimal rounding of
the same quantity. The reader's unique contribution is row_cv, max_over_mean and
empty_rows, which per item 2 are exactly the columns nothing independent validates.

### 4. No floor or spread figure from chain207 transfers to this run

chain207 is **6 readings per cell per build**; this screen is **4**. The same reduction
over fewer readings is noisier, so this run's A/A floor will be wider, and every
chain207-derived floor or spread quoted anywhere in this document is a 6-pass estimate
that does not transfer as a threshold -- including the 27.7%-of-S-1 k=4 gate floor of
amendment 1(g), the per-cell two-build spreads, and the CV figures of amendment 3(b).

**Which figures this class contains, because a sweep finds twenty-five and only a few are
omissions.** A chain207 figure used to DESCRIBE chain207 -- its own gate definition, its
own floor row, the arithmetic of a reconstruction, a corpus-CV correction -- is correct
usage and needs no note. The class is figures a reader might take as a BAR FOR THIS RUN.
That is the deciding predicate, and without it the rule is not decidable by a reader: the
sweep returns twenty-five sites with no way to sort them, and the next session applying it
literally will either re-report nineteen false positives or stop trusting the sweep. Notes
are therefore placed only where a figure is offered as a threshold, and a floor cited as a
MECHANISM -- such as amendment 9(1)'s A/A-floor-by-nnz table, which exists to show the
inverted U and not to bound anything -- is outside the class and marked as such at the
table itself.

Verified rather than assumed: `score_mrhv.py` contains no floor constant at all. Every
floor it uses is computed per width from this run's own mr0/mr0b pair, which are
byte-identical objects built at one fixed absolute path.

What *does* transfer, because each is a conjunction or a predicate over raw readings
rather than a reduction: the sign-only 51's membership, the sealed 8's membership, and
the one cell in their intersection.

### 5. The condition 4 null is not an identity, and the reason is worth stating

> **Not transferable (amendment 9(4)):** this is a chain207 6-pass figure. The screen is 4 passes, so it does not transfer as a threshold. Superseded further by amendment 10: the 5.00% is wrong on its own terms (the floor is two-sided, the count one-sided) and is now MEASURED at 2.33-2.76%. The ARGUMENT of this section is unchanged and is what amendment 11(5) cites.

The 5.00% exceedance the p95 floor implies would be a tautology if the floor were
computed on the same readings it is applied to. It is not: the floor comes from the
mr0/mr0b A/A pair and the count comes from the arm contrast, so under no effect the 5% is
a genuine prediction about a different sample. The plant's seven null widths read
2.50-5.83%, straddling it -- a calibration, not a restatement.

# Amendment 10 — condition 3 secondary was two-sided where the effect is one-sided, and read "inside the floor" across the entire registered mechanism range
> **Corrected by amendment 11:** this amendment's verification plant was DEGENERATE
> (zero-width intervals), so it proved unbiasedness and not power; and its defect table
> was computed on a different A/A construction than the scorer builds, understating the
> defect threefold. Amendment 11 carries the power measurement and the corrected table.


Written 2026-10-02, screen still on pass 1, nothing from it scored. Found by a peer
session reading `score_mrhv.py`; reproduced here with my own code before changing anything.

### The defect

Amendment 2 registers condition 3 secondary as the k=4 **geomean** against the geomean's
own floor from the mr0/mr0b pair. The implementation instead built the per-cell
`|log(arm/control)|`, took its p95, and compared that against the same functional on the
A/A pair.

A two-sided tail statistic is not a conservative choice for a one-sided effect. It is
**non-monotone in the effect size**, because a uniform speedup divides every cell's ratio
by the effect, which pulls the cells sitting above 1 down through 1 and *out* of the
two-sided tail before the far side regrows. Planting a uniform speedup onto chain207's own
A/A noise over the 51 cells, floor p95 19.14% of S-1:

| plant | x1.00 | x1.02 | x1.05 | x1.10 | x1.15 | x1.25 |
|---|---|---|---|---|---|---|
| implemented p95, of S-1 | 19.14 | 16.81 | **16.58** | 22.13 | 27.69 | 38.79 |
| implemented verdict | inside | inside | inside | inside | DISPL | DISPL |
| registered geomean CI | contains 1 | **excludes** | excludes | excludes | excludes | excludes |

The statistic falls *below* its own no-effect value at x1.02 and x1.05 and does not
produce a displaced verdict until x1.15. The registered functional separates from x1.02.
**The registered mechanism range is 7.7-12.4%, which is x1.077 to x1.124 -- entirely
inside the dead zone.** A working mechanism would have fired this test less than no
mechanism at all.

I had predicted the opposite sign before measuring, reasoning that `|log r - c|` must
have a growing upper tail. That holds for an `r` symmetric in log, and the A/A
distribution is not: over the 51 its per-cell range is 0.656 to 1.193, so the tail is
dominated by the left side and the ordering of the top three values changes as the plant
shifts the distribution.

### The fix, and what is NOT changed

`boot_sep` now takes the functional as an argument. Condition 3 secondary uses
`stat="center"` -- the geometric middle, compared on the log scale, which is the
registered functional. Conditions 1 and 2 keep `stat="tail"`, where two-sided is
**correct**: a firing difference between ROWS=2 and ROWS=4 has no preferred sign, and
condition 2 is an equivalence test. Sidedness has to match the question, not be chosen
once for the file.

The corrected functional on the same sweep is monotone and recovers the plant exactly:
+2.00 / +5.00 / +10.00 / +15.00 / +25.00 / +50.00 pp of S-1 at the six plants, displaced
from x1.02 on.

### Four further defects in the same file, all fixed

**Condition 4's null was asserted and roughly twice the truth.** The floor is a p95 of
`|log|`, so 5% of cells exceed it on *either* side, while the count tests only the upper
side -- about half. The printed "(null 5.00% by the floor's own definition)" was therefore
too high by a factor of about 1.9, in the direction that makes a clean run look like a
regression. It is now **measured** instead of asserted: the A/A pair's own one-sided
exceedance at the same floor is computed per width and printed beside the arm's. On the
full-corpus plant that reads 2.33-2.76% across widths, and the arm reads 2.15-2.65% at
the seven null widths and 0.00% at k=4 where the effect is. Amendment 9(5)'s *reasoning*
stands untouched -- the floor comes from the A/A pair and the count from the arm, so it
is a prediction about a different sample and not an identity -- only its value was wrong.

**Condition 3 primary set a pooled centre against a per-cell tail over a different
population.** `g - 1.0` is a geometric middle over the 51; the floor column beside it was
a per-cell p95 over all 3820 keys. Two mismatches at once, and it printed a
"beyond/inside" verdict inside a block whose own header says no threshold is applied. The
column is now the A/A **geomean** on the same cells -- the matched functional -- with no
verdict word.

**A dead identity presented as a control.** The line computed
`paired(..., FLOOR_BUILDS[0], FLOOR_BUILDS[0], ...)`, the same build as both arguments,
so every ratio was exactly 1.0; it was never read, and it hardcoded one arm inside a loop
over arms. Deleted. A 1.0000 column labelled "control" is the most convincing artefact a
reader can be handed. It survived because `pre-commit.sh` lints `src` only and never
`harnesses/`, so flake8's unused-variable check could not reach it.

**The key set was derived from the data it certifies.** `keys` came out of the observation
dictionary, so a key missing from every object in every pass is not "short" -- it is
absent, and no per-cell completeness check can see it. The ledger already opened for the
behind set is an independent record of which matrices the corpus holds, so the scorer now
compares its key set against the ledger's and exits if any ledger key is wholly absent.
This fired immediately on the old 171-key plant, which is why the plant is now generated
over all 3820 keys -- so it exercises the tables at the real n, with the real p95 indices.

One further item: the two chain207 figures condition 4's header cites (the 2.7x tercile
spread, rho -0.62) now carry amendment 9(4)'s transferability mark in the output. They
are grounds for a withdrawal, not thresholds, and nothing compares against either.

### The unifying statement, which is the part worth carrying

**There is no such thing as "the floor" for a run.** A floor is a functional over a
population, and this file had one floor object with three different statistics set against
it: condition 4 matched (per-cell against per-cell p95), condition 3 primary mismatched
twice (functional and population), condition 3 secondary building its own and diverging
from the registered word. Every site where a number meets a floor needs checking against
what sits on the other side of the comparison -- the sidedness, the functional, and the
population, each separately.

# Amendment 11 — amendment 10's verification proved the estimator unbiased and said nothing about its power; and the defect was three times worse than recorded

Written 2026-10-02, screen on pass 1, nothing scored. Both items from a peer session,
both reproduced here before changing anything.

### 1. The verification plant was degenerate

Amendment 10 reports the corrected statistic recovering a plant as
+2.00 / +5.00 / +10.00 / +15.00 / +25.00 / +50.00 pp of S-1 and calls that exact
recovery. It is exact recovery, and **every one of those intervals has zero width.** I
printed `[+2.00, +2.00]` and read a collapsed interval as a clean result.

The cause is algebra, not noise. The plant was `arm = floor / g`, so inside every
bootstrap replicate `geo(arm) = geo(floor) / g` identically, every replicate equals
`g - 1`, and the interval is a point. The pairing divides the plant straight back out,
because the plant *was* the floor rescaled. Confirmed: at x1.010 and x1.100 the widths
are 0.00 pp.

**So that table establishes the estimator is unbiased, and nothing whatever about whether
it can detect anything.** A degenerate plant reports certainty at any effect size,
including x1.0001.

### 2. The power, measured the way the screen is actually wired

The screen's contrasts share a denominator and have independent numerators: the floor is
`ours` on mr0b against `ours` on mr0, and the arm is `ours` on mrhv against `ours` on
mr0. Emulated on chain207 by borrowing a different arm for the independent third reading
-- floor `ours_bb/ours_aa`, arm `oursb_bb/ours_aa / g` -- over the 51 at k=4:

| plant | arm geomean | A/A geomean | separation, pp of S-1 | width | verdict |
|---|---|---|---|---|---|
| x1.000 | 0.9854 | 1.0058 | [-0.78, +4.68] | 5.46 | inside -- the null, correctly contains 0 |
| x1.010 | 0.9756 | 1.0058 | [+0.21, +5.73] | 5.52 | DISPLACED |
| x1.030 | 0.9567 | 1.0058 | [+2.19, +7.82] | 5.63 | DISPLACED |
| x1.050 | 0.9384 | 1.0058 | [+4.18, +9.91] | 5.74 | DISPLACED |
| **x1.077** | 0.9149 | 1.0058 | **[+6.86, +12.74]** | 5.88 | DISPLACED -- bottom of the registered range |
| x1.100 | 0.8958 | 1.0058 | [+9.14, +15.15] | 6.01 | DISPLACED |
| **x1.124** | 0.8767 | 1.0058 | **[+11.52, +17.66]** | 6.14 | DISPLACED -- top of the registered range |

**The corrected statistic detects the registered 7.7-12.4% mechanism with its lower bound
at +6.86 pp or better.** The broken one needed x1.50.

The limit, which cuts against the figure: the x1.000 row is centred on +1.95 pp rather
than 0, because `oursb` runs about 2% faster than `ours` throughout that ledger. The
interval contains zero so the null is valid, but a conservative reading subtracts the
offset and puts first detection nearer x1.03-x1.05 -- still inside the registered range.
This emulation carries that offset only because an independent third reading had to be
borrowed from a different arm. The real screen's mr0 and mr0b are the **same** arm on two
byte-identical objects, so it should not carry it, and the floor row of the actual run is
the check on that.

### 3. "The A/A displacement" names four different quantities, and amendment 10 used the wrong one

Over the same 51 cells at k=4, p95 of the per-cell |log|:

| construction | p95, of S-1 |
|---|---|
| object pair on `ours` (bb/aa) -- **what `paired()` builds** | **39.77%** |
| object pair on `oursb` (bb/aa) | 33.29% |
| arm pair inside one build | 21.34% (aa) / 21.60% (bb) |
| object pair on S = ours/mkl -- the sealed `an_c207.py` form | 19.14% |

The sealed form divides MKL out of both sides, so common-mode noise cancels and the floor
is half the size. Amendment 10's table was computed on that last form while the scorer
builds the first, which is why my numbers and the peer's differed with neither being
wrong. **Recomputed on the distribution the scorer actually sees, the defect is about
three times worse than amendment 10 records**: the minimum sits at x1.10, not x1.05, and
nothing fires below x1.50. `score_mrhv.py`'s docstring now carries the corrected table and
the list of four constructions.

### 4. "Reduce last or not at all" does not transfer to a floor

That rule is why the sign-only 51 is estimator-invariant, and it is a rule about a
**predicate**. A floor is a statistic *of* the noise, so reducing shrinks the very thing
being bounded. Same A/A, same 51 cells, same width:

| floor computed over | n | p95, of S-1 |
|---|---|---|
| per-cell geometric middles of 6 pass-paired readings | 51 | 39.77% |
| raw pass-paired readings, no per-cell step | 306 | 128.37% |

A factor of 3.2 from the reduction alone. Both halves have to be stated together: reduce
last in the predicate, and reduce the floor to **exactly** the level of the quantity it
bounds. That is amendment 10's "one floor object, three statistics" generalized -- the
population and the functional have to match, and so does the reduction depth.

### 5. One clause on condition 4's null, recorded as a limit rather than a defect

The measured null is computed from the same mr0/mr0b readings the floor is derived from,
restricted from 3820 cells to 3769, so that row is near-definitional rather than an
independent prediction. What makes the comparison informative is that the **arm** row is
independent of the floor, which is amendment 9(5)'s argument unchanged. The two columns
should not be read as two independent estimates.

---

## Amendment 12: the realized invocation, bound to the driver that ran

`6374caf` committed `mrhv_screen.sh` after a peer found that a 53-hour screen's driver was
not in the tree. That closed the instrument's CONTENT and left its INVOCATION unbound, which
matters more here than it usually would: **the committed default float32 grid does not
contain the width this study is about.** Line 36 reads

    float32) DEFAULT_KS=2,8,12,16,20,24,32 ;;

-- seven widths, no k=4 -- while the whole registration is multi-row at the half-vector
width, float32 k=4. `KS=${3:-$DEFAULT_KS}` takes a positional override, and the row count
proves one was used: `mrgate.py:43` has `ARMS = ["ours","oursb","mkl"]`, so the completed
first unit's 91,680 rows is 3820 x **8** x 3, where the committed default would have given
3820 x 7 x 3 = 80,220. A parameterised script with no arguments recorded is as unbound as no
script, and less obviously so.

**The realized invocation**, read from the live process and corroborated by the log's own
header line:

    bash /scratch/bobbyy/mrhv/mrhv_screen.sh /scratch/bobbyy/mrhv float32 \
         2,4,8,12,16,20,24,32 9

    screen.log: corpus 3820 matrices, dtype float32, widths 2,4,8,12,16,20,24,32, reps 9

So root `/scratch/bobbyy/mrhv`, dtype float32, **eight** widths including k=4, 9 reps.

**Bound to the file that ran**, which the committing commit could not distinguish from a file
edited between the run's start and the fetch:

| | |
|---|---|
| sha256 on redwood | `32c9b0f077040980` |
| sha256 of the committed blob | `32c9b0f077040980` |
| mtime on redwood | 2026-10-02 20:48:12 |
| run start, per `screen.log` | 2026-10-02 20:50:29 |

The mtime precedes the start by 2m17s, so the committed bytes are the bytes that ran. The
driver is deliberately **not** edited to carry this record, because editing it would change
the digest and break the binding; the provenance sits beside the instrument, not inside it.

**The driver is inherited, verified in one command rather than by reading 226 lines.**

    diff <(sed 's/mr0b/shpb/g; s/mrhv4/nopf/g; s/mrhv/strm/g; s/mr0/ship/g' \
           mrhv_screen.sh) arm_screen.sh

is empty, so this is the prefetch screen byte-for-byte modulo the four build names: it
inherits the grid check, the pool guard with its pool-2 comment, the refuse-on-missing-build
loop and the per-unit `--- p$p $tag start/done` echoes. Nothing study-specific was added or
removed. The rotation is therefore the same cyclic Latin square, checked by enumeration:
mr0 at positions 1,4,3,2; mr0b at 2,1,4,3; mrhv at 3,2,1,4; mrhv4 at 4,3,2,1 -- every build
leading exactly once, so each object absorbs one leading unit and a position effect cannot
load onto one arm of the byte-identical floor pair.

**One inherited sentence is inverted for this study**, flagged because it is self-description
rather than measurement. The grid check prints that k=4 "routes to the half-vector kernel on
this dtype ... excluded from the tail census", which is correct for the prefetch screen and
backwards here: k=4 is the only width under test, so the driver's coverage requirement
describes the seven widths that are *not* the subject. The k=4 data is collected regardless,
because the width is in `$KS` and reaches `mrgate.py` the same as any other.

---

## Amendment 13: the invocation is the environment, read rather than inferred

Amendment 12 recorded argv. A peer then pointed out that argv is **four of this driver's
thirteen inputs**, and that two of the other nine default to a different experiment:

    CACHE=${F64_CACHE:-$ROOT/mklsweep/csrcache/ss}      <- SuiteSparse
    KEYS=${F64_KEYS:-$ROOT/f64_keys_screen.txt}         <- a float64 key list

So `mrhv_screen.sh /scratch/bobbyy/mrhv float32 2,4,8,12,16,20,24,32 9` is literally
consistent with a **SuiteSparse** float32 run. The corpus is not in argv, and the corpus is
the one input the driver's own comment block at lines 54-59 argues cannot be recovered
afterwards.

**The repair is to read the environment, not to quote more log lines.** The driver is still
running as pid 2904667, so `/proc/2904667/environ` is the invocation itself rather than a
reconstruction of it:

    F64_CACHE=/scratch/bobbyy/mklsweep/csrcache/dlmc
    F64_KEYS=/scratch/bobbyy/mrhv/dlmc_keys_nopfx.txt
    F64_HARNESS=/scratch/bobbyy
    F64_PFS=/scratch/bobbyy/mrhv/pfs
    F64_OUT=/scratch/bobbyy/mrhv/csv
    F64_ACTIVATE=/scratch/bobbyy/pfs-activate.sh
    F64_THREADS=32
    F64_TREE   unset
    F64_MINPOOL unset

That is a reading of the corpus path, where recovering it from the output filename
(`CORPUS=$(basename "$CACHE")`, `PREFIX=${CORPUS}_${DTYPE}`, so `dlmc_float32_*.csv` implies
basename `dlmc`) is an inference from the naming rule. Both agree; only one is evidence.

### Three inputs that no record binds, now measured

Reading the environment exposes which inputs have no home in any record the driver writes.
They are not incidental -- they are the three that determine **what was measured**: which
matrices, with what code, on which objects.

| input | where the driver records it | identity |
|---|---|---|
| `F64_KEYS` path | **nowhere** | `dlmc_keys_nopfx.txt`, 3820 lines, sha256 `a8132f50d2c44fe4`, mtime 20:49:32 (57s before start) |
| `F64_HARNESS` / `mrgate.py` | **nowhere** | `/scratch/bobbyy/mrgate.py`, sha256 `a55e07bf03e1fb34`, mtime 2026-09-08 19:31:54, `ARMS = ["ours","oursb","mkl"]` at `:43` |
| the four builds' objects | names only, at `builds` | see the floor-pair check below |

The key list's mtime precedes the run start by 57 seconds, so the corpus definition is bound
the same way the driver is. Its first entries are `rn50/extended_magnitude_pruning/0.8/...`,
which is DLMC by content and not only by cardinality.

**The sentinel does not close this, though its field name suggests it does.** Lines 223-225 are

    printf "...\nks %s\nkeys %s\n..."  "$KS" "$NK" ...

where `NK=$(wc -l < "$KEYS")`. The field named `keys` holds a **count**, so the completion
record identifies the corpus by cardinality exactly as the log header does. 3820 is the DLMC
key count, but a count is not a path: any 3820-key list satisfies it.

### The A/A floor pair is byte-identical at the object level, not merely by construction

This was an intention of the design and is now a measurement:

| arm | object sha256 |
|---|---|
| mr0 | `25a5f0c4425671aa` |
| mr0b | `25a5f0c4425671aa` |
| mrhv | `713b9353f62f55c0` |
| mrhv4 | `f2ecbf401439f742` |

mr0 and mr0b are the same bytes, so the floor pair measures process-to-process and
position-in-rotation variation with **zero** code difference -- three distinct objects across
four arms, which is the designed structure. "Built from the same source" would have permitted
the layout artefact that moves narrow-k SpMM 1-2.6% between objects with identical decisions;
identical digests exclude it. Had these two differed, every floor number in this screen would
have been an A/B mislabelled as an A/A, and nothing in the logs would have said so.

### The pool guard was not relaxed, and one latent defect did not fire

`F64_MINPOOL` is unset, so the guard at `:180-186` ran on its default floor `THREADS/2` = 16
and admitted the recorded `pool 24`. That matters because `F64_MINPOOL` can relax the guard
silently, and only the realized POOL shows what was actually admitted.

`F64_TREE` is also unset, so `TREE` resolves to `/scratch/bobbyy/mrhv/scorch-beat-mkl`, which
**does not exist**. It is harmless here only because `TREE` has exactly one use -- line 64,
`HARNESS=${F64_HARNESS:-$TREE/harnesses/pfstream}` -- and `F64_HARNESS` is set. The
resolve-or-refuse loop at `:69` checks `KEYS` and `ACTIVATE` and not `HARNESS`, so with
`F64_HARNESS` unset this driver would have passed its own premise check and failed 130 lines
later at the first `python "$H/mrgate.py"`. Recorded as a gap in the refuse loop's coverage,
not as a defect in this run.

### The general rule

Argv is the invocation only for a script that takes all its inputs there. **Ask which inputs
come from the environment, and bind those too** -- preferring `/proc/<pid>/environ` while the
process lives, because it is the only form that is a reading rather than a reconstruction. A
default that points at a *different experiment* makes the gap worse than a missing record: it
supplies a plausible wrong answer instead of no answer.

---

## Amendment 14: the corpus is now in the tree, and the sentinel has three fields that cannot do their job

### One of amendment 13's three unbound inputs was already bound, and the narrowing sharpens the rest

`mrgate.py` is committed at `harnesses/pfstream/mrgate.py`, sha256
`a55e07bf03e1fb34644072da299f4e34645f3d4c1363f45aaa3b4fd382163fc8` -- the full form of the
digest read on redwood -- and `pfs_screen_dlmc.sbatch:99` **already gates on that exact value**
in its `declare -A WANT`. So the measurement code was bound before this thread started, on two
hosts. "Nowhere" was true of the log and the sentinel and false of the tree, which is the part
amendment 13 did not check. Two unbound inputs, not three.

### The remaining one was worse than "absent": the tree holds a DIFFERENT list under a confusingly close name

`harnesses/pfstream/dlmc_keys.txt` is committed, and it is **392 keys**, sha `24e386cee4371ce9`
-- not the 3820 that ran. Its entries already begin `rn50/...`, so there is no prefix transform
that maps one to the other; stripping a leading `dlmc/` leaves the digest unchanged. A reader
finding a committed `dlmc_keys.txt` beside this screen's record would reasonably take it for the
corpus, and it is a tenth of it. That is the same hazard as `CACHE`'s SuiteSparse default one
layer out: **a plausible wrong answer available where the right one is missing.**

**Fixed by committing the list that ran**, since a 332 KB corpus definition is an input and not a
result:

    harnesses/pfstream/dlmc_keys_full3820.txt
      3820 lines, 3820 distinct
      sha256 a8132f50d2c44fe4acabe673a9394b341453bd636ade9ae45831968612d275d8

The digest of the fetched file equals the digest read on redwood before transport, so the copy is
verified rather than assumed. The relationship between the two lists is also now measured: **the
392 are a strict subset of the 3820** -- `comm -23` gives 0 keys present in the small list and
absent from the large one -- and neither has duplicates. So the small list is a declared subset,
and the hazard was its name, not its content.

### Three sentinel fields cannot attain a value that would contradict them

| `:223-225` field | value | what it can attain |
|---|---|---|
| `nonzero_rc` | `$failed` | **0 only.** `:210` increments it and `:213` exits, with no other assignment but `failed=0`. The printf is unreachable while it is nonzero. |
| `passes 4`, `builds 4` | literals in the format string | the design, asserted |
| `csvs` | `ls "$OUT"/${PREFIX}_*.csv \| wc -l` | derived -- the one number that could contradict the two literals, and **nothing compares them** |
| `keys` | `$NK` = `wc -l < $KEYS` | a cardinality, under a field name that promises a path |

So the record states its design twice as a literal, measures the quantity that could falsify it,
and never does the comparison -- `csvs == passes * builds` is one line away. And `$OUT` is only
ever `mkdir -p`'d (`:171`), with the sole writes under it being the per-unit CSV (`:200`), the
per-unit log (`:205`) and `screen.done` (`:225`); there is no `rm`, `-delete`, `unlink` or
`truncate` anywhere in the file. So a file left by an earlier run under the same prefix counts as
this run's. This project has hit exactly that and resolved it by renaming the directory.

**And the quantity that would have caught the defect this whole thread is about is the one the
sentinel drops.** `rows=$(( $(wc -l < "$out") - 1 ))` is computed at `:207` and echoed at `:208`,
and never reaches `screen.done`. The per-unit row count is what showed an 8-width grid had run
where the committed default was 7 (91680 = 3820 x 8 x 3, against 80220 for seven widths); a
*total* of CSV files cannot do that and a per-unit minimum can. Since `mrhv_screen.sh` is
byte-identical to `arm_screen.sh` modulo four build names, this is the completion record of every
screen in this project.

**Not claimed:** that any of these fields has misled a reading. They are unexercised, and the
outcome that would make them matter is a screen that half-completes and still leaves a sentinel.
The general move is the one worth keeping: **for every field in a status artefact, ask what range
it can attain at the point it is written.** A field whose only attainable value is the good one is
an identity dressed as an observation.

### Mitigation for this run, since the driver cannot be changed mid-flight

The per-unit row counts are being recorded here as each unit closes, which is the quantity the
sentinel omits. Unit 1: `mr0` 91680 rows, rc=0, 20:50:29 to 00:08:43.
