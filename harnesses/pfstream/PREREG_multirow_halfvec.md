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

## The four refutation conditions, registered before the run

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
