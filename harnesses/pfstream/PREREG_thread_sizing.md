# Pre-registration: charging for team width in the SpMM thread rule

Registered 2026-10-02, before any held-out cell was measured. The calibration data is named
below and is explicitly NOT part of the test.

## The gap, in the code's own words

`scorch_policy.h`'s thread rule bounds the worker count two ways -- by work (at least one grain
of work per worker) and by rows (at least sixteen rows per worker) -- and the drop-in path then
adopts the caller's pool so a pipeline does not reshape its OpenMP team at every op boundary.
No term anywhere charges for the cost of opening a wider team. The note in
`memory/omp_team_width_cost.md` states the missing rule exactly: *"A worker is worth adding only
if the arithmetic it takes off the critical path exceeds its arrival delay plus its share of
teardown. Not built."*

## What is measured and already known

Per added worker, 4->12 band, one instrument, three hosts: **MKT 0.06us, redwood 0.26us,
M5 Max 4.75us** (79x and 18x). The M5 has a step at the sixth worker, which is its
perflevel0 core count and tracks the fast-core count rather than a pool boundary. Three
confounds were ruled out when that was measured: the hooks getenv tax is serial and cannot
scale with width, call-by-call `num_threads` reshaping was excluded by re-running in 1500-call
blocks, and ambient laptop load moved the sixth worker's entry offset by 0.5us.

## The model

Let `W` be real arithmetic in nonzero-units (`nnz*k`, NOT the cache-line-floored
`nnz*max(k,16)`), `T1` the single-thread kernel time, and `c` the host's per-worker cost.
Splitting `n` ways leaves `T1/n` on the critical path, so going from `n` to `n+1` removes
`T1/(n(n+1))` and adds `c`. The last worth adding satisfies `T1/(n(n+1)) > c`, so

    n* ~= sqrt(T1 / c),  clamped to [1, min(pool, rows/16)]

`T1` is not known at dispatch and is modelled as `T1 ~= a*nnz + b*nnz*k` -- the first term the
k-independent A-stream and cache-line cost, the second the arithmetic. That decomposition is not
invented here: fitting `ms ~ a + b*nnz*k` to the sealed chain212 DLMC pass-1 data gives an
intercept that explains 99.6-100.8% of our k=2 time in every nonzero band below 1e6 and that
grows with nnz (7.4 -> 11.6 -> 22.7 -> 27.2 -> 189 us), i.e. a cost proportional to nonzeros and
not to k. Three host constants total: `a`, `b`, `c`.

## THE FUNCTIONAL FORM IS THE CLAIM, and it is the part that can fail cheaply

The shipped knob `SCORCH_SPMM_ADOPT_GRAIN` implements `n <= W/G`, which is **linear** in work.
`n* ~= sqrt(T1/c)` is not linear in work. No choice of `G` reproduces a square root, so the two
rules disagree across widths on the same host by construction. **Registered prediction 1: no
single value of `G` in the linear rule matches the measured per-width optimum within +/-1 worker
across k in {2,4,8,12,16,32} on any one host.** If a single `G` does match everywhere, the
sqrt form is unnecessary and this whole line is refuted in favour of the knob that already ships.

## What is CALIBRATION and what is TEST

**Calibration, already measured and therefore not evidence for the model:** a thread sweep
(t = 1,2,4,6 and pure policy) over ten DLMC rn50 matrices with 1e4 <= nnz < 1e5 and rows >= 256,
at k in {2,8,32}, on the M5. It gives `c` a sanity check and fixes `a` and `b` for that host.
Those cells are excluded from every number reported as a result.

**Held out, and the only thing that counts:**
1. **Other widths on the same host:** k in {4,12,16,20,24}. Registered prediction 2: the measured
   argmin worker count agrees with `n*` within +/-1 at every one of those five widths.
2. **Other matrix classes:** the 72-key corpus stratified on DLMC's eight distinct row counts,
   spanning nnz inside each stratum. Registered prediction 3: agreement within +/-1 on at least
   80% of (matrix, width) cells, and no cell off by more than 2.
3. **The two x86 hosts:** same functional form, each host's own `c` from the table above.
   Registered prediction 4: the form transfers with no new term, and in particular the predicted
   counts on MKT (c = 0.06us) are at or near the pool for every cell with nnz >= 1e4, because
   `sqrt(T1/c)` with the smallest `c` is the largest count -- so the rule must be close to
   INERT on MKT. If the rule changes MKT's counts materially at wide k, the model is wrong.

## The regression surface, named in advance

The rule can only be shipped if it does not regress what we already win. On DLMC float32 we beat
MKL at k >= 16 (0.8759 / 0.7457 / 0.6659 / 0.6584 at k = 16/20/24/32). Any rule that narrows a
team at those widths is exposed there, and the shipped `ADOPT_GRAIN` form does: at 32530 nonzeros
and k=16 it takes 6 workers to 3. So **k >= 16 is an exposed surface here, not a negative
control**, and the corpus carries the wide-k small-nonzero corner deliberately.

Two genuine negative controls do exist:
- **Large `W`:** where `sqrt(T1/c)` exceeds `min(pool, rows/16)` the clamp binds and the rule is
  provably inert. The 33288-row DLMC stratum is there to contain such cells. A reading that
  differs from the floor there is an instrument fault, not an effect.
- **The floor pair:** two objects of one configuration at equal-length paths. A same-object
  repeat cannot bound a compile-time arm that changes emitted code -- that is what the PFSTREAM
  screen's k=2 control failure already established (strm +0.68% and nopf +1.76% where the code
  provably cannot run, against a -0.06% same-object floor).

## Refutation

The model is refuted, and reported as refuted, if any of:
- a single linear `G` matches the per-width optima within +/-1 everywhere (prediction 1 fails);
- the held-out widths disagree with `n*` by more than 1 worker at two or more widths;
- the rule is NOT near-inert on MKT at wide k;
- any cell beats the floor pair in the wrong direction at k >= 16 by more than the floor's own p95.

Reporting a refutation is the result. Nothing here is a reason to tune a constant per width:
a per-width table of worker counts with no host property behind it is the overfitting this
project is explicitly measuring against.

## Amendment 1: prediction 4 with numbers, and a competing mechanism named in advance

Registered before any x86 cell was measured. MKT's screen logs record `pool 32`; redwood's
`nproc` is 32 with a torch pool of 24.

**Prediction 4, concretely.** On MKT `c = 0.06us`. For the DLMC deficit class (nnz = 32530,
rows = 512, k = 2) the row ceiling is `min(32, 512/16) = 32`, so the clamp allows the whole pool.
`n* = sqrt(T1/c)` reaches 32 as soon as `T1 >= 61us`, which is 1.9ns a nonzero or about five
cycles each -- comfortably inside what a six-instruction kernel takes single-threaded. So the
rule should name the pool on essentially every cell with nnz >= 1e4 on MKT, i.e. **be inert
there**, and the same arithmetic on redwood (`c = 0.26us`, pool 24) needs `T1 >= 150us` to reach
24, so redwood should be partly bound and the M5 (`c = 4.75us`, pool 6) heavily bound. The
ordering of how much the rule bites is registered as M5 > redwood > MKT.

**A COMPETING MECHANISM, named now because it predicts the same sign on the M5 and a different
one on MKT.** MKT's measured k=2 time is 22.7us at 32 workers on 32530 nonzeros. If `T1` were
61us the ideal split plus team cost is under 4us, so 22.7us is roughly six times what the model
allows. Either `T1` is much larger on MKT than a cycle count suggests, or the parallel scaling at
k=2 is limited by something the model has no term for -- and the obvious candidate is aggregate
bandwidth: at k=2 each nonzero fetches a 64-byte line to use 8 bytes, so 32 workers demand
32 times that line traffic at once where 6 workers do not. A rule expressed in worker count
cannot fix a saturated memory system.

**The two hypotheses are distinguished by the SHAPE of time against worker count, which the
oracle sweep measures directly on every host:** rising with workers means team cost dominates
(the M5 reading, 0.0073 -> 0.0222 ms over 1 -> 6 at k=2); falling means compute-bound; **flat
above some count means saturation**, and in that case the deficit on that host is not a thread
problem and this whole line does not address it. Registered: if MKT's k=2 curve is flat rather
than rising, the thread rule is NOT the fix for MKT's narrow-k deficit, and saying so is the
result. The two hosts may simply be bound by different things at k=2, and one rule in worker
count would then be the wrong shape for both.

## Amendment 2: four corrections, one of which weakens the registered prediction

All four were raised by a reviewing session and all four reproduce. Registered before any
held-out cell was measured; the oracle sweep had not run when this was written.

**(a) The ADOPT_GRAIN family DOES span width-aware rules, and 150000 is the one value that does
not.** Amendment 1 and the commit body for e374854 said the knob "cannot make the decision
width-aware, because all it can do is remove the adoption's raise". That holds only for large G.
The chain is `cand = min(override, rows/16)`, then `cand = min(cand, max(1, work_true/G))`, then
`nthreads = max(nthreads, cand)`, so wherever `policy < by_real <= min(override, rows/16)` the
resolved count IS `by_real = nnz*k/G`, which is proportional to k. At the probe cell (nnz 32530,
rows 512, policy 3 from the object, pool 6) the resolved counts over k = 2,4,8,12,16 are

    G = 400000   3 3 3 3 3     G = 100000   3 3 3 3 5
    G = 150000   3 3 3 3 3     G =  50000   3 3 5 6 6
                               G =  20000   3 6 6 6 6

The flat claim needs `G >= nnz*k_max/policy`, which is 173493 on that cell. **This weakens
registered prediction 1**, which is correct: a family that spans width-aware rules might contain
one that matches everywhere, and that is exactly the refutation the prediction is there to admit.
The oracle sweep scores the whole G ladder, so the comparison is a reading.

**(b) `a` IS NOT A CONSTANT, so "three host constants total" is false as written.** Fitting
`T1 = a*nnz + b*nnz*k` per nonzero decade gives an intercept-per-nonzero that falls by about
three orders of magnitude from the smallest band to the largest -- cache residency, which no
linear term in nnz can carry. Since `n* = sqrt(T1/c)`, a factor f in T1 is sqrt(f) in the count,
so a 850x spread in `a` is up to 29x in the worker count. **Added to the refutation list.** The
repair is to separate two questions the first draft conflated: the FORM `n* = sqrt(T1/c)` is
scored with T1 MEASURED (the sweep's resolved-count-1 reading), which is an upper bound on what
the form can achieve and cannot ship; and the SHIPPABLE rule is scored with T1 modelled. If the
measured-T1 version wins and the modelled one does not, the limit is the T1 model and not the
form, and saying which is the result. `an_ntoracle.py` prints the per-band spread in `a` and
states the premise fails when it exceeds a factor of four.

**(c) 32530 and 22.7 are BAND MEDIANS, not a cell, and median(T) is not T(median(nnz)).**
Amendment 1 said "MKT's measured k=2 time is 22.7us at 32 workers on 32530 nonzeros". 32530 is
the median nnz of the 1e4-1e5 band and 22.7us is the median k=2 time in that band, taken
independently; no row in any sealed CSV has nnz = 32530. So the "roughly six times what the model
allows" factor compares a median time against the model evaluated at a median nnz, which for a
nonlinear T is not a valid composition. The competing-mechanism paragraph stands as a HYPOTHESIS
-- it was always going to be settled by the shape of time against worker count, not by that
factor -- but the factor itself is withdrawn and must not be quoted. The synthetic probe point
(nnz 32530, rows 512) used to interrogate the objects is a constructed input, not a matrix, and
is fine for asking a policy what it resolves to and not fine as evidence about any matrix.

**(d) The intercept-share figure's predicate, stated so it can be checked.** "99.6-100.8% of our
k=2 time" is: `chain212-dlmc-f32-p1/ss64_ship_p1.csv`, build `ship`, arm `ours`, PER MATRIX
ordinary least squares of `ms ~ a + b*(nnz*k)` over that matrix's seven widths, then the MEDIAN
over matrices within a decade band of `a_i / t_i(k=2)`. A band-level OLS with one intercept per
band is a different estimator and gives different numbers (0.87-1.03 on the same data); neither
is wrong, and the two are not comparable. The per-matrix form is what was reported and it is what
the quoted range refers to.

## Amendment 3: a registered control withdrawn, and a decision rule the registration omitted

**(a) THE 33288-ROW NEGATIVE CONTROL IS WITHDRAWN.** It was registered on the ground that where
`sqrt(T1/c)` exceeds `min(pool, rows/16)` the clamp binds and the rule is provably inert, so a
reading that differs from the floor there is an instrument fault. Measured against the object,
57 of that stratum's 72 cells resolve to a single count (18) for every override, because their
policy count reaches twice the caller's pool and `scorch_policy.h` then declines the final cap
outright -- `if ((long)nthreads >= 2 * pool) nt_cap = 0;`. So the override has no effect in
either direction there, and the cells are inert through the cap decline rather than through the
clamp the control names. A control that cannot distinguish its registered reason from a harness
fault cannot detect the thing it exists to detect, and excluding single-configuration cells
empties it rather than cleaning it. Withdrawn rather than reinterpreted.

Two controls replace it, both printed with their own n so an empty one reads as empty:
- **A cell-level A/A by design.** The sweep now times every override TWICE, so each reachable
  count carries a same-object repeat on every live cell. The accidental supply it replaces was
  54 pairs on 18 of 576 cells (3.1%), with 56.8% of cells supplying none -- so the earlier claim
  that duplicate resolutions "turn the defect into an instrument" was true of 3.1% of cells and
  was stated as though it were general.
- **Clamp-bound live cells**, meaning live cells whose largest reachable count already equals
  `min(pool, rows/16)`. A rule cannot ask for more there, so any rule naming a count at or above
  the clamp must read the floor. The analyser warns when there are fewer than twenty.

Override 0 is also added to the sweep, because it is a real configuration and not a probe: the
standalone and panel paths pass no override, so pure policy is something the library actually
runs.

**(b) PREDICTION 1 WAS REGISTERED WITHOUT AN AGGREGATION RULE, and the instrument supplied one.**
The registered text names a per-width quantity -- within +/-1 of the measured optimum -- and says
nothing about what share of cells must satisfy it. The analyser printed "at least 80% of cells",
a constant borrowed from prediction 3, which is a different prediction over matrix classes.
80%, 50% and 100% can disagree about the verdict, so the choice decided the outcome and was
disclosed in a print statement rather than in the registration. **Rather than register a
threshold now, after the instrument exists and with the data about to land, the verdict is
printed at all three thresholds and the threshold-sensitivity is part of the result.** If the
three disagree, that is reported as the finding; picking one retroactively would be choosing the
answer. Prediction 3's own 80% stands, because it was registered with the prediction.

## Amendment 4: every policy-derived figure in this project needs its pool attached

A number produced by querying `scorch_spmm_nthreads` is not a number until the calling process's
pool is recorded with it. Two sessions read the same function on the same corpus and got
incompatible distributions -- one spanning worker counts 7 to 18, the other stopping at 6 -- and
both were right: `torch.set_num_threads` differed. On the cell (rows 128, nnz 44236, k 24) the
same call returns 2, 4, 6, 7, 7, 7, 7 at pools 2, 4, 6, 8, 12, 16, 32. Neither reading was an
arithmetic error and no amount of re-deriving either one would have found it.

**So, retroactively, with the pool named:** the reach figures (36.4% of 3820 DLMC matrices at
k<=16, 31.4% at k=32), the reachable-configuration histogram ({1: 231, 4: 18, 6: 327} over
576 cells), the single-configuration share (231 of 576 = 40.1%, of which 141 pin at 1 and 90 at
18), and the accidental same-count pair supply (54 pairs on 18 cells) were **all measured at
pool 6 on the M5**. They are not comparable to a reading on a host with a different pool, nor to
a rerun on the same host under a different `-c`. The per-host drivers already record the pool;
the prose did not, and that is the gap.

**AND `override = 0` IS NOT THE POLICY COUNT.** With a non-positive override the final cap is
`nt_cap = omp_get_max_threads()` (`scorch_policy.h:2203`), so the reading is the policy count
capped by the process pool, with the `>= 2*pool` recruit decline punching through for large
values. At pool 6 over the tracked corpus, 342 of 576 cells (59.4%) have a saturating count --
the value at an override far above anything -- strictly above the override-0 reading, with gaps
up to 17 (rows 512, nnz 6200, k 32: the override-0 reading is 1 and the saturating count is 18).
Recording the override-0 reading is correct, because the standalone and panel paths pass no
override and it is therefore a configuration worth timing. Calling it `policy` and using it as a
lower bound was not: the bound came out one-signed too small.

That landed on the very null added to make prediction 1 honest, and it biased it toward reporting
that a pass was informative. The null no longer uses any policy bound. It is now the count of
cells where **no tested G** lands within one worker of the measured argmin, computed directly
over the ladder -- exact for the family actually tested, and free of the pool entirely.

---

## Amendment 5 (2026-10-02, before any x86 sweep exists): the grain is one constant and the break-even is not

**The prediction registered in the pending work -- "near-inert on MKT (c=0.06 us)" -- is hereby
replaced, and the replacement predicts the opposite sign.** It was written from the M5's
arithmetic, where widening a team costs 4.75 us per worker and the measured excess headroom is
1.0032 at k=2 (i.e. none). That reasoning does not transfer, and the reason it does not is the
one quantity the two hosts differ in most.

**What provoked it.** Asked at pool 32 what the shipped policy resolves for the shapes that
dominate the MKT DLMC float32 k=2 deficit set:

| rows | nnz | mean deg | resolved at k=2 | measured ours/mkl |
|------|-----|----------|-----------------|-------------------|
| 512  | 9630    | 18.8  | **1** | 1.960x |
| 512  | 13107   | 25.6  | **1** | 1.940x |
| 512  | 26214   | 51.2  | **2** | 1.938x |
| 512  | 58982   | 115.2 | 6     | ~1.47x |
| 512  | 472165  | 922.2 | 18    | 1.481x |

`rows/16` is 32 on every one of them, so the clamp is not binding. The binding term is
`work/grain` with `work = nnz*max(k,16)` and `SCORCH_GRAIN_SPMM 150000`: at nnz 9630 that is
154080, one grain, one worker. **The cells furthest behind MKL are the ones running on a single
thread out of a pool of 32**, and the ordering is monotone in the table above.

**The mechanism, which is not host-specific special-casing.** A worker is worth adding while the
time it removes exceeds what it costs: `T1/(n(n+1)) > c`, where `c` is the per-worker team cost.
`c` is a measured host property and it spans **79x** across our three hosts -- 0.06 us on MKT,
0.26 us on redwood, 4.75 us on the M5. A single work grain encodes one break-even point, so a
grain calibrated where teams are expensive is far too conservative where they are nearly free. On
a 20 us kernel, eight workers cost 0.5 us on MKT (2.4%) and 38 us on the M5 (190%). The same
constant cannot be right for both, and nothing in the shipped rule reads `c`.

### Registered predictions, scored on the x86 sweeps when the screens clear

1. **MKT is NOT inert.** The excess over the winner's-curse null -- median(cur/best) divided by
   its within-cell permutation null, the statistic defined in `nt_k2curve.py` -- will exceed
   **1.05** at k=2 on MKT. The M5 reads 1.0032. Scored on live cells only.
2. **The headroom concentrates where the resolved count is small.** Among MKT k=2 live cells,
   the excess for cells resolving to <= 2 workers will exceed that for cells resolving to >= 8
   workers. Registered as a comparison, not a threshold, because the split sizes are unknown.
3. **redwood sits between the two hosts** on prediction 1's statistic, c being 0.26 us against
   MKT's 0.06 and the M5's 4.75. This is the prediction most likely to fail, because c enters
   the criterion under a square root and a 4.3x difference in c is 2.1x in n*, which the pool
   clamp may absorb entirely on redwood's pool of 24.
4. **The negative control.** At k=32 the work measure `nnz*max(k,16)` is `nnz*32`, so the same
   matrices resolve to roughly twice the count and the criterion is nearer satisfied. Prediction
   1's statistic at k=32 will be below its k=2 value on MKT. If k=32 shows equal or greater
   excess, the account above is wrong and the deficit is not a thread-count story at all.

### What would refute the mechanism rather than the predictions

If MKT's excess at k=2 is at the null (<= 1.01) while single-worker cells remain 1.9x behind MKL,
then the thread count is not the deficit's cause on MKT either, and the single-thread resolution
above is a coincidence of the shapes rather than an explanation. That outcome is to be reported
as the result, exactly as the M5's 1.0032 was.

**No rule is to be written before these are scored.** The marginal criterion gives the form
`n* ~ sqrt(T1/c)`, which no linear grain reproduces, and a host-measured `c` is a defensible
constant rather than a tuned one -- but a defensible form fitted to a noise-selected optimum is
still fitted to noise, which is what the winner's-curse null exists to prevent.

---

## Amendment 6 (2026-10-02, minutes after amendment 5, before any x86 sweep): amendment 5's mechanism is refuted by data already in hand

> **FORWARD MARKER, inserted later the same day, after amendment 7.** The resolved-count table
> in THIS amendment is asked at `override = 0`, which no measurement on this project uses:
> `mrgate.py:123` passes `nthreads_override=POOL`. Its one-worker total of 133 keys is correct
> FOR override 0 and is 45 higher than the 88 the measured configuration gives, because the
> adoption gate at `scorch_policy.h:2139` is conditioned on `override > 0` and raises 45 of
> those keys. Read amendment 7's table instead. The conclusion below is unchanged and is
> sharper at the right override.

**Amendment 5's table was selection on the outcome and the full population reverses it.** I
probed five shapes chosen from the worst-cells table, found the worst ones resolving to one
worker, and proposed that the work grain is too conservative where teams are cheap. The check
that account needed -- the resolved count over ALL cells, with the ahead set as the contrast --
was available without any new measurement and says the opposite.

At k=2, pool 32, over all 392 MKT DLMC float32 keys classified by the 12-reading sign rule:

| class  |   n | resolved to 1 worker | share <= 2 workers | median count | median nnz | median deg |
|--------|-----|----------------------|--------------------|--------------|------------|------------|
| behind | 328 |                   75 |              32.9% |          6.0 |     58 982 |     115.00 |
| ahead  |  53 |               **49** |          **92.5%** |          1.0 |  **1 311** |       7.12 |
| mixed  |  11 |                    9 |              81.8% |          1.0 |          - |          - |

**The ahead cells are the single-worker cells.** 92.5% of them run on at most two workers against
32.9% of the behind cells, so a single worker does not distinguish behind from ahead -- it
anti-correlates with the deficit. Matched within an nnz decade the count does not separate them
either: at 1e3 both sets read a median count of 1.0 (35 behind, 31 ahead), and at 1e5 the ahead
cells resolve HIGHER (18 against 14). The two decades holding most of the behind set, 1e4 and
1e6, contain **no ahead cells at all**, which is why a decade-matched comparison is possible at
only two of four decades and is reported with those gaps named.

**Amendment 5's predictions are NOT withdrawn, because they are about a different quantity** --
the excess over the winner's-curse null on an x86 sweep -- and they remain scoreable. But
prediction 2, "the headroom concentrates where the resolved count is small", now has this
evidence against its direction before the sweep runs, and that is recorded here rather than
discovered afterwards. Prediction 1 and the k=32 control stand unaffected.

**What the same table says instead, and it is a harder fact than the one it replaces.** The
behind set's median is 58 982 nonzeros and the ahead set's is **1 311**. So at k=2 the matrices
we beat MKL on are essentially trivial -- a kernel with 1311 nonzeros on one worker, where MKL's
own call overhead is most of its time -- and we lose on nearly every DLMC matrix of consequential
size. The earlier framing, "we lose on long rows and win on short ones", understated it by
describing degree when the sets also differ 45x in nonzero count and 4x in row count. The honest
statement of the remaining deficit is: **at k=2 we are behind on 328 of 392 DLMC matrices, ahead
on 53 whose median size is 1311 nonzeros, and no thread-count property separates the two.**

**Method note, recorded because it is the second time today.** Both of today's dissolved findings
-- the 169x cell and amendment 5's single-worker account -- came from reading a table sorted by
the outcome and then characterising its top rows. A property found that way is conditioned on the
outcome and cannot be evidence about it. The fix that worked both times was the same: state the
property, then recompute it on the whole population WITH the opposite class as a contrast, before
writing anything down.

---

## Amendment 7 (2026-10-02): amendment 6 reached the right conclusion from the wrong configuration

**Amendment 6's table asked the policy at `override = 0`. No measurement on this project uses
that.** `mrgate.py:123` is `SYM(nthreads_override=POOL, atparallel=False, **native)`, so every
reading in both screens was taken with a positive override, which engages the composition
adoption at `scorch_policy.h:2139` -- `nthreads_override > 0 && gate_work >= grain`, with
`gate_work = nnz*max(k,16)` and `grain = 150000`. The conclusion survives; the table that
supported it named a configuration that never ran.

Recomputed at `override = POOL`, same 392 keys, same classification:

| class  |   n | resolved count, override=POOL  | median | at 1 worker | adoption gate fires |
|--------|-----|--------------------------------|--------|-------------|---------------------|
| behind | 328 | {1:30, 4:1, 8:20, 16:23, 18:254} |   18.0 |  30 (9.1%)  |      298 (90.9%)    |
| ahead  |  53 | {1:49, 18:4}                   |    1.0 |  49 (92.5%) |        4 (7.5%)     |
| mixed  |  11 | {1:9, 18:2}                    |    1.0 |   9 (81.8%) |        2 (18.2%)    |

**The kill condition is met, and more sharply than at override 0.** 92.5% of the cells we BEAT
MKL on run on a single worker, against 9.1% of the cells we lose on. The gate keys on `nnz`
alone at a fixed width, and the behind set is the high-nnz set, so the behind cells pass the gate,
are raised to the host team, **and are still 1.16-1.90x behind**. Amendment 5 required the deficit
cells to be the ones held to few workers. They are the ones given the most.

**One column in that table does not transfer and is marked rather than quietly carried.** The
adoption gate is pool-independent -- at k=2 it is just `nnz >= 9375` -- so the 90.9% / 7.5% split
holds on MKT. The resolved counts do not: the 18 is `min(override=32, rows/16=32,
hardware_concurrency=18)` and 18 is THIS host's core count. On MKT, where that term is 128, the
same behind cells resolve to 32. So the correct transferable statement is the gate split plus the
raise it implies, and the per-count histogram needs re-asking on MKT before it is quoted.

**What this does to amendment 5.** Its provocation is withdrawn: the table said the worst cells
run on one worker out of 32, and under the configuration that produced the 1.960x they run on the
whole team. The `c`-spans-79x argument is untouched by this -- it is about where the break-even
for adding a worker lies, not about which cells are currently starved -- and the four registered
predictions remain scoreable on the x86 sweeps. Prediction 2, "the headroom concentrates where
the resolved count is small", now has its direction contradicted twice: once by amendment 6's
population check and once by this one at the right override.

**The method failure, stated once because it is now three instances in one afternoon.** A
resolved count is a property of four things -- shape, override, pool, and object -- and amendment
5's table named only the first. The 169x cell named no reading index, amendment 5 named no
override, and this correction's own histogram would have named no `hardware_concurrency` if the
value had been 32 instead of 18 and so had looked plausible. The operational form: before
quoting a policy figure, write down all four coordinates; if any is absent the figure is about an
unspecified configuration, and the ones that read plausibly are the dangerous ones.

---

## Amendment 8 (2026-10-02): the thread count is constant where the deficit varies 4.4x

**This closes the thread-count account with an argument that needs no measurement at all, only
the two tables already in hand.** The work measure is `nnz*max(k,16)`, so it is numerically
identical at k=2, 8, 12 and 16. Asked of the default object at `override = POOL` over all 392
MKT DLMC float32 keys:

| k  | resolved count identical to the k=2 count | mean count | cells behind MKL in all 12 readings |
|----|-------------------------------------------|------------|-------------------------------------|
| 2  | **392 of 392 (100.0%)**                   |      13.52 | 328 (83.7%)                         |
| 8  | **392 of 392 (100.0%)**                   |      13.52 | 158 (40.3%)                         |
| 12 | **392 of 392 (100.0%)**                   |      13.52 | 116 (29.6%)                         |
| 16 | **392 of 392 (100.0%)**                   |      13.52 |  74 (18.9%)                         |
| 20 | 386 of 392 (98.5%)                        |      13.68 |   6 (1.5%)                          |
| 24 | 377 of 392 (96.2%)                        |      13.94 |   4 (1.0%)                          |
| 32 | 354 of 392 (90.3%)                        |      14.80 |   2 (0.5%)                          |

Every key runs on exactly the same number of workers at k=2 as at k=16, and the deficit falls
from 83.7% of keys to 18.9% -- a factor of **4.4** -- across that range. A quantity that does not
change cannot explain one that changes 4.4x. The count does rise by about 9% by k=32, which is
the direction that would help, and the deficit there is 2 cells; so the two move together only in
the sense that both improve, and 9% of worker count is not 160x of deficit count.

> **SCOPE MARKER, inserted later the same day, in the form amendment 7 used for the 18.** The
> 392/392 identity is a property of `(override, hardware_concurrency)`, not of the keys. At
> `override = 6` on this host the same object gives 382/392 at k=8, 376/392 at k=12 and
> 373/392 at k=16, and the k=2-to-k=32 mean rises 33.9% rather than 9.5% -- because the raise
> gate reads `work_true = nnz*k`, which is NOT k-invariant, and whether that shows depends on
> where the cap lands. At override 32 here the cap is `hardware_concurrency = 18`; on MKT that
> term is 128, so cells reach 32 and the k-dependence has more room. **Re-ask on MKT before
> quoting 392/392 there.** The argument survives either reading: even at override 6, 95-97% of
> keys are unchanged across k=2..16, and a few percent of keys changing count cannot explain a
> 4.4x change in deficit count.

**Three independent arguments now close it, in both directions and on the choice itself:**

1. **Too few workers** -- refuted by amendments 6 and 7: the cells we BEAT MKL on are the
   single-worker cells (92.5% at one worker against 9.1% of the behind cells), and the behind
   cells pass the adoption gate, are raised to the host team, and lose anyway. Mean 13.52 workers.
2. **Too many workers** -- refuted by the M5 curve relabelled by resolved count: time falls to
   0.8089 at five workers before rising to 0.9615 at six, so six workers are still 3.85% faster
   than one, and the 3.06x figure that said otherwise was labelling arms by the override argument.
3. **The wrong count was chosen** -- refuted by the winner's-curse null: production's count
   against the best reachable count reads 1.1856 at k=2 against a permutation null of 1.1818
   (excess 1.0032), and at k=4..32 the excess is below 1, meaning production BEATS an arbitrary
   count by 4.7-11.4% of the null.

**So the remaining deficit is not a threading question and the prereg's subject is closed.** What
is left is a kernel running on a full host team, at narrow k, on matrices of median 58 982
nonzeros, losing to MKL by 1.16-1.90x, with the deficit monotone in k. The width dependence is
now the primary signature and it is the one thing none of the eliminated mechanisms predicts.

The four predictions of amendment 5 remain registered and will still be scored when the x86
sweeps run, because a registered prediction is not withdrawn for being expected to fail -- but
the expectation is now recorded: predictions 1 and 2 should fail, and if prediction 1 PASSES on
MKT while this table holds, something is wrong with the excess statistic rather than right with
the thread rule.

---

## Amendment 9 (2026-10-02): "the narrow-k deficit" is at least two populations, and one candidate died proving it

**A candidate registered and refuted inside ten minutes, on data already in hand.** The two
constraints from amendments 7 and 8 -- that the class is shape-determined, and that the deficit
is monotone in k while the thread count is not -- point at one quantity: A-bytes streamed per
output element, `8*nnz/(rows*k)` for float32 CSR. It is a function of shape alone and falls as
`1/k`. Registered form: ONE threshold, chosen once on all widths pooled, scored per width against
the majority-class baseline, because a quantity needing a different threshold at each width is a
restatement of k and explains nothing.

| k  | behind |  ahead | threshold accuracy | majority baseline | verdict |
|----|--------|--------|--------------------|-------------------|---------|
| 2  |    328 |     53 |              64.6% |         **86.1%** | worse   |
| 8  |    158 |     96 |              31.5% |         **62.2%** | worse   |
| 12 |    116 |    144 |              48.1% |         **55.4%** | worse   |
| 16 |     74 |    252 |              72.1% |         **77.3%** | worse   |
| 20 |      6 |    362 |              95.1% |         **98.4%** | worse   |
| 24 |      4 |    383 |              97.4% |         **99.0%** | worse   |
| 32 |      2 |    384 |              99.0% |         **99.5%** | worse   |

**It beats the baseline at no width.** Pooled accuracy was 76.0%, which looked respectable and is
the number that would have been quoted without the per-width baseline beside it.

### Why it failed, which is the finding

The association does not keep its sign. Median mean-degree by class:

| k  | behind | ahead | ratio  | we lose on        |
|----|--------|-------|--------|-------------------|
| 2  | 115.00 |  7.12 | 16.14x | **HIGH** degree   |
| 8  |  92.97 | 32.04 |  2.90x | **HIGH** degree   |
| 12 |  51.20 |104.79 |  0.49x | **LOW** degree    |
| 16 |  39.68 |157.95 |  0.25x | **LOW** degree    |
| 20 |  25.61 |102.40 |  0.25x | **LOW** degree    |
| 24 |  15.80 | 93.78 |  0.17x | **LOW** degree    |
| 32 | 216.36 | 92.97 |  2.33x | HIGH, but n=2     |

**The sign reverses between k=8 and k=12.** At k=2 the cells we lose on have 16x the degree of
the cells we win on; at k=16 they have a quarter of it. So the phrase "the narrow-k deficit" has
been naming a set whose composition changes as k rises, and every mechanism search this session
-- team width, thread count, the adoption gate, the work grain, the prefetch, the gather kernel,
A-streaming -- has been looking for a single explanation of two different things. That is why a
candidate satisfying both registered constraints could still fail at every width: the constraints
were derived from the pooled set.

### The split, and which part is the task

- **The high-degree narrow-k population: k=2 and k=8, 486 of the 688 behind cells.** At k=2:
  median 58 982 nonzeros, median degree 115, median 512 rows, and the 53 cells we beat have
  median 1 311 nonzeros. This is the dominant part and the only part worth a mechanism now.
- **The low-degree mid-k population: k=12 through k=24, 200 cells.** Median degree 15-51 against
  an ahead set at 94-158. Its deficit is also falling fast with k (116 to 4), so it may be the
  tail of a different effect entirely and is explicitly NOT in scope until the first is closed.
- **k=32's 2 cells are not a population** and the 216.36 in that row is two matrices. It is in
  the table only so that its sign is not read as corroborating k=2.

### Registered before the next measurement

No mechanism will be proposed for the pooled 688 again. The next measurement addresses the k=2
high-degree set alone, holds the worker count fixed across both libraries, and will be scored
against the stated refutation condition rather than characterised after the fact. A finding on
k=2 that does not also predict the k=8 behaviour of the SAME matrices is not to be reported as
explaining the narrow-k deficit, because those two widths are the part of the set that shares a
composition.

---

## Amendment 10 (2026-10-02): amendment 9's scope included a width whose sign no statistic settles

**k=8 does not belong in the high-degree population, and amendment 9 put it there on a per-set
statistic nothing registered.** Recomputed with three statistics side by side -- the median, the
mean, and the share above a single global degree cut (the corpus median `deg`, 86.22, which makes
that column tail-insensitive by construction):

| k  | behind | ahead | med beh / ahd  | mean beh / ahd | >cut beh / ahd | sign        |
|----|--------|-------|----------------|----------------|----------------|-------------|
| 2  |    328 |    53 | 115.00 /  7.12 | 200.9 /  13.8  | 59.5% /  0.0%  | **all three HIGH** |
| 8  |    158 |    96 |  92.97 / 32.04 | 106.8 / 170.2  | 50.6% / 35.4%  | **SPLIT 2-1**      |
| 12 |    116 |   144 |  51.20 /104.79 |  69.1 / 263.7  | 26.7% / 54.2%  | all three LOW      |
| 16 |     74 |   252 |  39.68 /157.95 |  45.7 / 235.0  |  2.7% / 65.5%  | all three LOW      |
| 20 |      6 |   362 |  25.61 /102.40 |  28.7 / 181.2  |  0.0% / 53.6%  | all three LOW      |
| 24 |      4 |   383 |  15.80 / 93.78 | 114.6 / 173.1  | 25.0% / 50.7%  | all three LOW      |
| 32 |      2 |   384 | 216.36 / 92.97 | 216.4 / 172.7  | 50.0% / 50.5%  | SPLIT 2-1, n=2     |

At k=8 the median and the share say the behind set is the high-degree one and **the mean says the
opposite** -- the ahead set has 1.6x the mean degree. Since the share is tail-insensitive and
agrees with the median, the disagreement is a tail effect: the ahead set at k=8 contains a few
very high-degree matrices. Either reading is defensible and nothing was registered, so k=8 is
**not assignable** and is removed from scope.

### The scope, corrected

- **k=2 alone: 328 cells. This is the task.** All three statistics agree and the separation is
  the cleanest in the table -- **zero of the 53 ahead cells have a degree above the corpus
  median**, against 59.5% of the behind cells. Median 115.00 against 7.12, median 58 982
  nonzeros against 1 311.
- **k=8: 158 cells, sign indeterminate.** A transition width. Admitting it needs a criterion
  registered in advance with its reason, not chosen after seeing which answer it gives. I am not
  registering one now, because the only reason I would is that it enlarges the population I want
  to explain.
- **k=12 through k=24: 200 cells, all three statistics LOW.** A coherent second population,
  out of scope.
- **k=32: 2 cells, no sign.** Under a weaker classification on the sealed pass-1 ledger the same
  width reads 4 keys and the OPPOSITE sign (0.18x against this table's 2.33x) -- same data,
  different split, opposite direction, n in {2,4}. It stays tabulated only with its n attached.

**Amendment 9's "486 of 688" is withdrawn and replaced by 328.** The k=2-plus-k=8 grouping was
the larger number and the one I wanted; it was not the one the data supports. The rule registered
in amendment 9 -- that a k=2 finding must also predict k=8 on the same matrices -- is likewise
withdrawn, since it was premised on those two widths sharing a composition and k=8's composition
is undetermined.

**Carried forward as a rule, since the direction here was decided by a choice I made silently:**
a claim that one set skews higher than another on some quantity must print the median, the mean,
and a share above a single global cut. The share is tail-insensitive, so a split it joins is a
location effect and one it breaks is a tail effect, and the three together say which. A direction
feels non-parametric and so invites picking one functional without noticing a choice was made.

---

## Amendment 11 (2026-10-02): the k=2 structure is a one-sided degree ceiling, and it exists at no other width

**The sharpest statement of the k=2 deficit is a ceiling on the set we WIN, not a floor on the
set we lose.** On the 12-reading classification (328 behind / 53 ahead):

> At k=2, the highest mean degree of any matrix we beat MKL on is **59.92**. **218 of 392 keys
> lie above that ceiling and 217 of them are behind in all twelve readings** -- the remaining one
> is mixed, and none is ahead.

The association is **one-sided**. High degree predicts the deficit almost perfectly; low degree
predicts nothing, because the behind set reaches down to degree **5.12** and overlaps the ahead
set across its whole range. Every earlier framing -- "we lose on long rows", "59.5% of behind
cells exceed the corpus median", "zero of the 53 ahead cells exceed it" -- is the weaker half of
this, and the share-above-a-cut version is also cut-sensitive in a way the ceiling is not.

**It needs a cut-free statistic with its own null, because a maximum is an order statistic of the
data** and 100% precision at exactly that threshold is true by construction -- the same
winner's-curse shape as the regret denominator. Statistic: the percentile rank, in that width's
all-key degree distribution, of the ahead set's maximum. Null: labels shuffled with the ahead-set
SIZE held fixed, 2000 permutations, seed 20261002.

| k  | n ahead | max deg(ahead) | percentile | null median | null p05 | p(null <= obs) |
|----|---------|----------------|------------|-------------|----------|----------------|
| 2  |      53 |      **59.92** |  **43.04** |       98.95 |    95.01 |     **0.0000** |
| 8  |      96 |        1382.40 |     100.00 |      100.00 |    97.64 |     1.0000     |
| 12 |     144 |        2304.00 |     100.00 |      100.00 |    99.23 |     1.0000     |
| 16 |     252 |        2304.00 |     100.00 |      100.00 |    99.39 |     1.0000     |
| 20 |     362 |        2304.00 |     100.00 |      100.00 |   100.00 |     1.0000     |
| 24 |     383 |        2304.00 |     100.00 |      100.00 |   100.00 |     1.0000     |
| 32 |     384 |        2304.00 |     100.00 |      100.00 |   100.00 |     1.0000     |

At k=2 the ceiling sits at the 43rd percentile of a distribution where random labelling puts it
at the 99th. **At every other width the ahead set reaches the corpus maximum degree (2304.00) and
the statistic is exactly null, p = 1.0000.** So the structure does not weaken with width -- it is
absent. This is a third independent reason for amendment 10's k=2-only scope, and the only one of
the three that is cut-free.

### What this constrains about any mechanism

A mechanism for these 328 cells must be something that is **essentially always decisive above
mean degree 60 at k=2 and never decisive at k=8**, where matrices of degree 1382 and 2304 are
ahead. It must also be compatible with the behind set extending down to degree 5.12, so it cannot
be a property that only high-degree matrices have -- it has to be one whose COST at k=2 grows
with degree without bound while its cost at k=8 does not.

Every mechanism eliminated so far fails that test in an instructive way: the thread count, the
adoption gate and the work grain are all identical at k=2 and k=8 by construction (amendment 8,
`nnz*max(k,16)`), so none of them can be decisive at one width and null at the other.

### Cross-criterion status, recorded because the ceiling's VALUE is criterion-dependent

A peer's independent read of sealed pass 1, under the weaker "median behind" split (333/59),
places the ceiling at **88.25** rather than 59.92 -- the difference being a single key,
`rn50/variational_dropout/0.95/initial_conv`, which that criterion admits as ahead and this one
does not. Confirmed here: my ahead set's maximum is 59.92, so that key is not in it. The
CEILING EXISTS under both criteria and the null is p = 0.0000 under both; its numeric value moves
with the classification and must be quoted with it. The shape counts underlying amendment 9 --
266 distinct shapes, 52 repeated, 178 keys covered, max multiplicity 14, and 52 of 52 unanimous
split 42 behind / 10 ahead -- reproduce EXACTLY across the two criteria, and the permutation nulls
differ (35/39/44 against 34/38/42) in the direction the label imbalance predicts (86.1% against
85.0% behind), which is the agreement worth having.

---

## Amendment 12 (2026-10-02): a quantifier corrected, the percentile's convention stated, and the ceiling's axis identified

### 1. A quantifier error in my own restatement, corrected with the measured value

I wrote that redwood's float64 screen found "**0 cells behind MKL in any reading**". The tool
computed something weaker and correct: `an_f64_vs_mkl.py` classifies BEHIND as `all(r > 1.0 for
r in rs)` and prints it as "behind in **every** reading". Those are different claims, and the
stronger one is false by my own numbers in the same sentence -- 81 mixed cells each hold at least
one reading that is not ahead.

Measured rather than bounded: **192 of 46 704 readings (0.4111%) have ours slower than MKL**,
spread over exactly the 81 mixed cells, distributed 39 cells with one such reading, 18 with two,
5 with three, 8 with four, 4 with five, 3 with six, 1 with seven, 2 with eight and 1 with nine.

So the correct statements are: **no cell is consistently behind** (0 of 2919), and **0.41% of
individual readings are behind**, confined to 2.77% of cells. The code had every part of this
right -- it counts the three outcomes separately, and its own comment at `:194-200` explains why
the sum-to-total equality is an identity that must not be asserted as coverage. The failure was
one word of English, in the direction that strengthened the claim.

### 2. The percentile's convention, which was unstated and not re-derivable

Amendment 11's `pctile 43.04` is **164/381**: the denominator is the CLASSIFIED keys at that
width (328 behind + 53 ahead), excluding the 11 mixed, because the permutation null shuffles
labels only among those. A peer reading the same quantity over all 392 keys gets 44.39%
(174/392). Both are defensible and the verdict is identical -- both sit far below the null p05 of
95.01, p = 0.0000 either way -- but the amendment quoted the number without its denominator.

**And the ceiling must not be quoted as a rounded decimal.** Its exact value is **3835/64 =
59.921875**, attained by exactly one key (`rows=64 nnz=3835`). Recomputing any percentile or
precision figure from the printed `59.92` *excludes the very key that attains it*, which shifts
the count by one in every table: 163/381 instead of 164/381, 173/392 instead of 174/392. Quote
it as `3835/64`, or as its (rows, nnz), not as two decimal places.

**Both ceilings are single-key statistics** -- mine attained by one key at 3835/64, a peer's
weaker classification by one key at 88.25 -- so each is one reclassified key from moving. That is
the sharper form of amendment 11's caveat that the VALUE is criterion-dependent while the
existence and the p are not, and it is the reason the percentile-with-its-null is the quotable
form while the raw ceiling and any precision-at-that-threshold figure are not.

### 3. WHICH AXIS the ceiling is on, which amendment 11 assumed rather than tested

The attaining key has only 64 rows, so "degree" could have been standing in for nnz or for rows.
Same one-sided statistic and the same null on each candidate, so they are comparable:

| axis                 | ahead max   | keys above | of those, behind | percentile | null median | p      |
|----------------------|-------------|------------|------------------|------------|-------------|--------|
| **mean degree**      |    59.92    |    **217** |      **217/217** | **43.04%** |      98.95% | 0.0000 |
| nnz                  |  852 171    |         25 |            25/25 |     93.44% |      98.95% | 0.0230 |
| rows                 |   33 288    |          0 |                - |    100.00% |     100.00% | 1.0000 |
| nnz x mean degree    |  2.1816e7   |        104 |          104/104 |     72.70% |      98.95% | 0.0000 |

**Mean degree is the carrier.** It leaves 217 keys above its ceiling with zero exceptions at the
43rd percentile of a distribution where chance puts the ceiling at the 99th. `nnz` reaches only
the 93rd percentile and p = 0.0230, which is marginal and leaves 25 keys above. `rows` is
**exactly null**: the ahead set contains the corpus's widest matrix, so there is no row-count
ceiling at all. The product is real but strictly weaker than degree alone, which means degree is
not a proxy for total work.

**This tightens amendment 11's constraint from a size effect to a per-row one.** Mean degree is
the inner-loop trip count per row. So the mechanism must be a cost **per nonzero within a row**
that is decisive above ~60 nonzeros per row at k=2 and not decisive at k=8, where the ahead set
reaches degree 1382. It is not a function of problem size: `rows` carries nothing and `nnz` is
marginal, so an explanation keyed on working-set size or total traffic has to account for why
degree beats both on the same test.

---

## Amendment 13 (2026-10-02): redwood's sweep scored, two predictions FAIL, and the curve/headroom distinction earns its keep

Object `8192715070225fd3`, pool 24, overrides 0..24 x 8 widths x 2 reps, 28 800 rows, one
`so_sha` throughout. Scored as registered, before MKT's sweep exists.

### The result that matters most: a huge curve and no rule headroom, on the same data

redwood's k=2 time against resolved worker count, geometric mean relative to each cell's own
smallest count:

```
count   1      2      3      4      5      6      7      8      9     ...   24
geo  1.0000 0.6476 0.4993 0.4335 0.3813 0.3497 0.3335 0.3278 0.3147  ...  0.3883
```

**Nine workers run 3.18x faster than one, and twenty-four give back 23% of that.** The endpoint
ratio is 0.5097 against a permutation null of 0.9863 -- a 48% excess, enormous and in the
falling direction. By any reading, worker count matters hugely here. The M5's curve over the same
corpus barely moved (0.8089 at its best).

**And the thread RULE still has nothing to recover.** The headroom statistic -- production's
count against the best reachable count, over its own winner's-curse null -- reads **1.1670
against a null of 1.2176, excess 0.9584**. Production *beats* an arbitrary count by 4.2% of the
null at k=2, and by 4.5 / 2.2 / 0.7% at k=4 / 8 / 12.

So a curve with 3.2x of real structure and a rule with negative headroom coexist on one host.
That is exactly the confusion this script was split off from `an_ntoracle.py` to prevent, and it
is the first time the distinction has been load-bearing rather than hypothetical: *the shape of
the curve says whether the parameter matters; only the headroom says whether the RULE is leaving
anything on the table.* A reader given only the 0.5097 would conclude there was a large win
available, and there is not.

### Scoring, as registered in amendment 5

| # | prediction | verdict | evidence |
|---|------------|---------|----------|
| 1 | MKT's k=2 excess exceeds 1.05 | **NOT SCOREABLE** | MKT's screen still running; no sweep exists |
| 2 | headroom concentrates where the resolved count is small | **FAILS** | k=2: `<=2` workers 0.9584 (n=9) vs `>=8` workers **1.0319** (n=33). k=32: 1.0523 vs 1.0550. Direction REVERSED at both widths |
| 3 | redwood sits between MKT and the M5 | **FAILS** | redwood 0.9584 is BELOW the M5's 1.0032, not between it and a predicted >1.05 |
| 4 | k=32 excess below k=2's (negative control, MKT) | not its host, but **contrary** | redwood k=32 reads 1.0330 against k=2's 0.9584 -- the control shows MORE excess, not less |

Prediction 3 was registered as "the prediction most likely to fail", for the stated reason that
`c` enters the criterion under a square root so a 4.3x difference in `c` is only 2.1x in `n*`,
which the pool clamp may absorb. It failed, and it failed harder than that reasoning allowed --
not absorbed toward the middle but past the M5 in the other direction.

Prediction 2's failure is its third contradiction: amendment 6 (population check at override 0),
amendment 7 (the same at override=POOL), and now a direct measurement of the registered
statistic. It should be considered settled.

### Corrections carried in from a peer's audit, and one of mine applied to my own table

- **The behind-set floor is `2621/512 = 5.119140625`, not 5.12.** Confirmed on my own split: the
  attaining key is `rows=1024 nnz=5242` and it is the same key under both classifications.
  Amendment 11 quoted the rounded decimal and inherits the exact defect amendment 12 filed.
- **The `nnz x degree` ceiling must not be `2.1816e7`.** Its attaining key is `rows=33288
  nnz=852171` and the exact value is `242065137747/11096`. Five significant figures of a
  composite carries a tolerance of several hundred; it happened to shift nothing (104 above it
  either way) but the remedy applies to it as much as to the degree ceiling.
- **The ledger's `deg` column is `nnz/rows` rounded to two decimals** -- 0 of 392 keys deviate
  from `round(nnz/rows, 2)` and only 32 are exact by coincidence. So the exact degree is
  unrecoverable from that column, and `rows` and `nnz` are the only carriers. Every degree figure
  either of us has quoted today came from it.
- **The argument in amendment 12 should lean on the `nnz` row, not the product row.** The claim
  being retired is "degree is a proxy for total work", total work in SpMM is proportional to
  `nnz` times the width, so `nnz` IS the total-work candidate and its marginal p = 0.0230 is what
  rules it out. `nnz x degree` is `nnz^2/rows`, not a natural work measure, so its being weaker
  than degree alone is hard to interpret and is demoted to a footnote.
- **Four candidates were searched, so the carrier test wants a multiplicity correction, and it
  helps.** At Bonferroni `0.05/4 = 0.0125`, `nnz`'s p = 0.0230 does **not** survive while mean
  degree's p = 0.0000 survives trivially. Accounting for the search strengthens the conclusion
  rather than weakening it, which is rare enough to state.

### What is still unscored

Prediction 1 and prediction 4 are both MKT predictions and MKT has three screen passes left. The
conditional registered with them stands: **if prediction 1 PASSES on MKT while amendment 8's
table holds, the excess statistic is wrong rather than the thread rule right.** With prediction 2
now failed on direct measurement and prediction 3 failed, a passing prediction 1 would be the
only surviving leg of amendment 5's account, and the conditional says to distrust it.

---

## Amendment 14 (2026-10-02): the census amendment 13 omitted, and two numbers relabelled

### 1. THE SHAPE CENSUS, which my own instrument says must precede the verdict

`nt_k2curve.py:191-195` states that the rising/falling/interior-min census "decides whether the
registered question is even well posed on this host, so it is printed before any verdict is
read". Amendment 13 reported redwood's verdicts and omitted the census entirely. Here it is, and
redwood is the case the instruction was written for:

| k  | live | rising | falling | **interior min** | flatish |
|----|------|--------|---------|------------------|---------|
| 2  |   53 |      1 |       9 |           **43** |       0 |
| 4  |   53 |      1 |       8 |               44 |       0 |
| 8  |   53 |      0 |      13 |               40 |       0 |
| 12 |   53 |      0 |      16 |               37 |       0 |
| 16 |   53 |      0 |      13 |               40 |       0 |
| 20 |   54 |      0 |      17 |               37 |       0 |
| 24 |   57 |      2 |      16 |               39 |       0 |
| 32 |   59 |      2 |      15 |               42 |       0 |

**81% of redwood's k=2 live cells (43 of 53) have an interior minimum**, against 47% on the M5
(23 of 49). The registration's dichotomy -- "flat in worker count rather than rising" -- has no
true answer on four fifths of them. **That is the primary result for the curve question on this
host**, and it is not a hedge: the pooled curve's own shape shows it, falling to 0.3147 at nine
workers and rising to 0.3883 at twenty-four.

The headroom verdicts in amendment 13 are unaffected -- they do not depend on the curve's shape,
which is the whole reason the headroom statistic exists -- so predictions 2 and 3 remain FAILED.

### 2. Two numbers that cannot sit adjacent unlabelled, and the fix is in the tool

Amendment 13 printed a curve ending at **0.3883** at count 24 with a base of 1.0000, and three
paragraphs later an "endpoint ratio" of **0.5097**. A reader reconstructing the second from the
first fails, and concludes one is wrong. They are different statistics with three differences:

| | the curve's last row | "endpoint hi/lo" |
|---|---|---|
| statistic | geometric mean | **median** |
| upper count | fixed at 24 | **each cell's own maximum** |
| population | only cells reaching 24 | **all live cells** |

`stat_endpoint` takes `min(counts)` and `max(counts)` per cell, and `cand = min(override,
rows/16)` caps small-row matrices low, where the curve is much higher -- which is what lifts the
pooled median from 0.39 to 0.51. Both are sound against their own nulls; neither is derivable
from the other. `nt_k2curve.py` now prints both side by side with the three differences named, so
the next reader cannot make this mistake from the output alone.

**And my host contrast was cross-paired.** I wrote redwood's endpoint (0.5097) against the M5's
minimum (0.8089). Matched:

- minimum against minimum: redwood **0.3147** vs M5 **0.8089**
- endpoint against endpoint: redwood **0.5097** vs M5 **0.9615**

Both favour the point and neither invites the objection, so there was never a reason to cross
them.

### 3. The product ceiling, in exact rationals, with both denominators named

`2.1816e7` overstates the exact `242065137747/11096 = 21815531.5200973...` by **468.5**, and one
key sits in that gap. Recomputed with exact rational arithmetic (`Fraction(nnz*nnz, rows)`, no
floats):

| population | n | above the exact ceiling | of those, behind |
|---|---|---|---|
| all keys | 392 | **105** | 104 |
| classified (behind + ahead) | 381 | **104** | 104 |

The difference is **exactly the one mixed key** above the ceiling, not a coincidence of two
offsets. My published 104 was over the classified population and is correct for it; the all-key
figure is 105. Same structure on the degree row, now stated with both denominators: above
`3835/64`, **218 of 392 all keys** and **217 of 381 classified**, differing by the single mixed
key, and all 217 classified ones behind.

### 4. The pattern, which is now three instances in one day

Each time, the tool printed a qualifier and my restatement dropped it, and each time the headline
number was correct while the clause saying what it was a number ABOUT went missing:

1. `all` became `any` on the float64 verdict (amendment 12) -- the code classifies
   `all(r > 1.0)` and prints "behind in every reading".
2. The per-count `cells` column, printed precisely so a changing denominator is visible, dropped
   from the curve I quoted.
3. The shape census, which the code says to print before any verdict, omitted from amendment 13.

The operational form, and it is not "read your own output more carefully": **diff the sentence
against the tool's own print string, not against its logic.** The logic was right all three
times. What differed was the English, and the print string is the only artefact that holds the
qualifier in the same words the tool uses.

---

## Amendment 15 (2026-10-02): amendment 14's census could not return one of the two registered answers, and its headline is withdrawn

### 1. `flatish` was unreachable by construction

`shape()` read `if i == 0: return "rising" if meds[-1] > meds[0] else "flatish"`. But `i == 0`
means `meds[0]` IS the minimum, so `meds[-1] > meds[0]` is false only when two medians of timing
samples are **bit-identical floats**. There was no tolerance anywhere in the function. The column
duly read 0 at all eight widths on both hosts, and I reported it as a measurement.

That is not cosmetic. **The registration's dichotomy is "flat rather than rising", so FLAT is one
of its two answers, and the artefact built to adjudicate it could only ever return the other.** A
genuinely flat curve came out `rising` when noise put its minimum at the first count, and
`interior-min` otherwise.

The file had the right construction in its *other* notion of flat -- the verdict threshold is
justified against the measured A/A spread so that "FLAT cannot be returned by noise alone" -- and
the census was added without it. Two notions of flat in one file, one with a measured tolerance
and one with exact float equality, and only the second was in the census.

**Fixed with the cell's own permutation null rather than a constant**, because the null of an
extreme widens with the number of draws and cells differ in how many counts they reach: a cell
whose observed max/min does not exceed what shuffling its own readings produces is FLAT, whatever
the position of its minimum. `flatish` is now reachable and non-zero (2 cells at k=2 on redwood).

### 2. The census had no null, and `interior-min` is the null's dominant category

If the worker count did nothing, a cell's minimum lands uniformly among its `n` reachable counts,
so `P(interior) = (n-2)/n` -- 66.7% at n=6, 91.7% at n=24. So a raw interior share is not
evidence, and **amendment 14's "81% on redwood against 47% on the M5" was largely comparing pool
sizes (24 against 6), not curve shapes.**

The null is now measured per cell by the same `perm_null` the other statistics use, because cells
cap at `min(override, rows/16)` and many reach far fewer counts than the pool.

**Unconditionally, per-cell structure is REAL:** observed `flatish` is 0-2 cells against a null of
24-29 on redwood and 0 against 19-25 on the M5. The counts are not exchangeable within a cell.
That much survives and is a genuine finding.

**Conditionally on being non-flat, the interior share is BELOW its null at every width on both
hosts:**

| k | redwood obs / null | M5 obs / null |
|---|--------------------|---------------|
| 2 | 80.4% / **90.4%**  | 46.9% / **66.6%** |
| 8 | 75.5% / **88.9%**  | 54.3% / **67.0%** |
| 16| 74.5% / **88.1%**  | 73.2% / 65.6%     |
| 32| 70.7% / **87.8%**  | 68.4% / 64.1%     |

The conditional comparison is the right one because the null puts most of its own non-flat mass
in `interior-min` by construction, so an unconditional comparison credits the data for a shape
the null prefers.

### 3. What is withdrawn and what stands

**WITHDRAWN: "43 of 53 redwood k=2 cells have an interior minimum" as evidence of anything.** It
is below its own null. Amendment 14 led with it and called it "the primary result for the curve
question on this host". It is not a result.

**STANDS, and is the actual evidence for an interior optimum:** the POOLED curve, which falls to
0.3147 at nine workers and rises 23% to 0.3883 at twenty-four. That is far outside any A/A spread
this corpus has produced and is a different claim with a different null from the per-cell census.
Amendment 14's conclusion -- that the registered dichotomy has no good answer on redwood -- also
stands, but for a different reason than I gave: not because interior minima dominate, but because
the per-cell shape classification cannot distinguish an interior minimum from noise at all. The
curve question is answered by the pooled curve, not by the census.

**UNAFFECTED: predictions 2 and 3 remain FAILED.** They are headroom predictions and the headroom
statistic does not read the curve's shape, which is why it was built separately.

### 4. The pattern, now four instances and a different one

The three in amendment 14 were all the same failure -- the tool printed a qualifier and my
restatement dropped it. This one is not that: the tool printed a column that could only ever hold
zero, and I reported the zero. The general form is the one already on file as "beside a
possibly-vacuous number, report the number that shows it was not" -- a category that cannot be
returned is the vacuous case of a partition, and the check is to ask of every category in a
classification *what input would land here*. For `flatish` the answer was "two bit-identical
timing medians", which is none.

---

## Amendment 16 (2026-10-02): the flat test's null is `live/2` by construction, so the aggregate claim needed a different test

### The defect, and it is in amendment 15's own fix

`perm_null` returns the **median** of a cell's permutation distribution, and amendment 15's new
FLAT test declares flat when the observed statistic is at or below it. But under exchangeability
the observed value is itself a draw from that distribution, so

> **P(FLAT | exchangeable) = 0.5 exactly, for every cell, independent of n, host and width.**

So the `flatish` null column is `live/2` by construction and is not a measurement of anything.
Amendment 15 wrote "observed 0-2 against a null of 24-29", which reads as *noise would have
produced about 27 flat curves*. It is a threshold defined to put half the null mass on each side.

Confirmed empirically -- the realized value tracks `live/2` with no systematic sign, redwood:

| k | live | live/2 | realized | dev |
|---|------|--------|----------|-----|
| 2 | 53 | 26.5 | 26.9 | +0.4 |
| 4 | 53 | 26.5 | 26.2 | −0.3 |
| 8 | 53 | 26.5 | 26.7 | +0.2 |
| 20 | 54 | 27.0 | 26.6 | −0.4 |
| 24 | 57 | 28.5 | 29.3 | +0.8 |
| 32 | 59 | 29.5 | 29.4 | −0.1 |

A proposed diagnostic -- that a strict `<` with ties would under-call FLAT and leave a systematic
shortfall -- does not hold here: the comparison at `:175` is `obs <= nul`, ties count as flat, and
the deviations above alternate in sign within ±0.8. The threshold also uses `perm_null`'s default
**200** permutations, not the 40 that feeds the census tally, so its sampling error is the smaller
one.

### The aggregate claim survives and is much stronger than the number it was quoted against

Both sides share the threshold, so observed-flat against `Binomial(live, 0.5)` is a valid test of
exchangeability, and it is the right one:

| host | k | live | observed flat | Binomial mean / sd | z |
|------|---|------|---------------|--------------------|---|
| M5 | 2 | 49 | 0 | 24.5 / 3.50 | **−7.0** |
| M5 | 8 | 46 | 0 | 23.0 / 3.39 | −6.8 |
| M5 | 32 | 38 | 0 | 19.0 / 3.08 | −6.2 |
| redwood | 2 | 53 | 2 | 26.5 / 3.64 | **−6.7** |

**Worker count matters per cell, at 6+ standard deviations on both hosts.** That part of amendment
15 stands on a far better footing than its own wording implied.

### Two things now stated in the output rather than left implicit

- **The realized null prints against its prediction**, so only a *deviation* can be read from it.
  When a test's threshold is an order statistic of its own null, the null's value becomes a
  property of the threshold; predicting it and checking the realized value is the only way the
  column carries information.
- **The per-cell labels are asserted at the 50% level** and must not be read as each cell being
  individually certified to have real shape. Raising the threshold to the null's p95 makes FLAT
  *easier* and shrinks the non-flat set. The aggregate test is immune because the same threshold
  is applied to both sides; a per-cell count is not.

### And the registered conclusion is restated in the code, not just in an amendment

The script's own conclusion block still said "where interior minima dominate, the registered
question is not well posed". Amendment 15 withdrew that reasoning and the code had not caught up.
It now reads the census as two separate halves: that worker count matters is established at 6+
sd; that the optimum is characteristically interior is **not**, being at or below its null at most
widths on both hosts. The dichotomy has no good answer because the per-cell classification cannot
separate an interior minimum from noise -- and the pooled curve, a different claim with its own
null, is the evidence for an interior optimum.

**A note on where these corrections are landing.** Four of the last six amendments have corrected
an instrument rather than the kernel, and three of those corrected a correction. That is the cost
of the registration being about a curve's *shape*, which is a weaker thing to measure than a
ratio against a baseline. The headroom statistic -- built separately for exactly this reason --
has needed no correction and has carried every verdict that survived: predictions 2 and 3 remain
FAILED on it, and amendment 8's construction argument closes the thread-count account
independently of any curve.

---

## Amendment 17 (2026-10-02): three defects in the instrument that produced amendment 15's wording

Amendment 16 adopted a peer's point that the flat test's null is `live/2` by construction.
Re-running the tool to confirm the fix surfaced three further defects in the same output, two
of which plausibly *caused* the original misreading.

### 1. A column named `flat` that counted cells with no shape at all

The statistics table's third column was `len(cells) - len(live)`: cells reaching exactly **one**
worker count, whose curve shape is **undefined**, not flat. At k=2 on the M5 it reads **23**,
sitting beside a flat-cell null of `live/2` = **24.5**. Two numerically adjacent quantities,
one named `flat` and one actually about flatness, in one output. Amendment 15's "observed 0
against a null of 24-29" is what reading across them produces.

Renamed to `1cnt`, with a line under the header stating what it is and what it is not.

### 2. Two measured numbers hardcoded into explanatory prose

The paragraph introducing the two candidate statistics said "the endpoint ratio reads 0.9615
(FLAT) and max/min reads 1.2362 (RISING)". Both were literals. Neither was computed anywhere in
the file. On redwood that paragraph asserted the M5's numbers as if they were the run's.

Now computed per run and printed as `THIS run at k=%d`, with the agree/disagree conclusion
derived rather than asserted. On the M5 the computed values are **0.9615** and **1.2363** --
the literals were right *for this CSV*, which is exactly why the defect was invisible: it was
checked only against the file it was transcribed from.

### 3. The single-object guard could not fire on the input it was written for

`shas = sorted({r.get("so_sha", "") for r in rows})` followed by `if len(shas) > 1: REFUSE`.
A sweep predating the `so_sha` column yields the one-element set `{''}`, passes, and prints
`object(s): ['']` -- which reads as single-object provenance. The guard existed because an
in-place build in the tree this environment imports from silently becomes every process's
import default, so two objects can produce one indistinguishable CSV. That is precisely the
case it accepted.

**The M5 oracle sweep behind every number in amendments 14-16 is provenance-absent**: 0 of 8064
rows carry an `so_sha`. The within-cell exchangeability result does not depend on which object
answered, but it does require that no single *cell* straddles an object boundary, and this file
cannot establish that. The −6.2 to −7.0 sd figures are therefore **provisional on the M5** and
confirmed on redwood, whose sweep carries one named object, `8192715070225fd3`.

Now: distinct named objects still refuse; a **mix** of named and blank rows refuses
unconditionally, being strictly worse than two named objects; an all-blank file refuses unless
`--allow-unknown-object` is passed, which stamps every pooled-curve line `[PROVENANCE UNKNOWN]`.
Stamping rather than withholding, because a stamp travels with a line that gets copied and a
suppressed number teaches nothing. Row counts per class print beside the verdict, so a vacuous
check cannot read as a passing one.

### What the three have in common

Each is a check or a label that was verified against the single artefact it was written beside.
The guard was written while looking at a CSV that had the column; the literals were transcribed
from the run that produced them; the column name was accurate to an earlier version of the
table. None of the three could be caught by re-reading the code, and all three fell out of
running the tool on a *second* input -- which in two cases was the same host, one sweep older.

---

## Amendment 18 — ALL FOUR registered predictions are now scored, on three hosts with one tool digest. The sqrt model is refuted.

> **CORRECTED BY AMENDMENT 19.** The heading's "all four" is ambiguous inside this document,
> which carries TWO prediction sets both numbered 1-4; it scored the set at `:57-68` and left
> amendment 5's set unscored. And the rows for predictions 2 and 3 report MKT's figures for
> predictions registered on the FIT host. Read amendment 19 for the host-labelled scorecard.

MKT job 17696556 completed, so the last two unscored predictions can be read. Every figure below
comes from `an_ntoracle.py` at digest **`222543063d02c9c1`** run on all three hosts' sweeps — the
same object, because a prediction about an ordering across hosts cannot be scored from statistics
that were redefined between them, and this one nearly was (see below).

### Prediction 4 — FAILS. The registered ordering is violated and the statistic is not monotone in `c`.

Amendment 1 registered: *"The ordering of how much the rule bites is registered as M5 > redwood >
MKT"*, on the reasoning that `n* = sqrt(T1/c)` reaches the whole pool as soon as `T1 >= c·pool²`,
which is 61 µs on MKT against 150 µs on redwood and far more on the M5.

| host | pool | `c` (µs) | LIVE cells | optimum at or above the clamp | strictly above |
|---|---|---|---|---|---|
| M5 | 6 | 4.75 | 345 of 576 | 82 (**23.8%**) | 0 |
| redwood | 24 | 0.26 | 435 of 576 | 150 (**34.5%**) | 68 |
| MKT | 32 | 0.06 | 435 of 576 | 98 (**22.5%**) | 2 |

**Observed ordering: redwood > M5 > MKT.** The registered one puts redwood in the middle; it is
the largest. And the two *extreme* hosts — 4.75 µs and 0.06 µs per worker, a **79× range**, with
pools of 6 and 32 — sit **1.3 points apart**, which is the opposite of a quantity governed by `c`.

The model's premise had already failed its own check: `a` in `T1 = a·nnz + b·nnz·k` spreads **58×
across nnz decades** on the fit half, and since `n* = sqrt(T1/c)` a factor `f` in `T1` is `sqrt(f)`
in the count, so up to 7.6× in `n*`. A model with no constant `a` cannot order hosts by `c`.

### The full scorecard

| prediction | verdict | the figure that decides it |
|---|---|---|
| 1 — per cell within ±1, no single G suffices | **PASSES, UNINFORMATIVE** | 15–22 of 26–29 cells per width are `no-G-can` (50–79%) |
| 2 — within ±1 at every width in {4,12,16,20,24} | **FAILS** | lowest width 17.4%, fails at 50/80/100% |
| 3 — ≥80% within ±1 **and** no cell off by >2 | **FAILS both clauses** | clause A 20.0% of n=160; clause B 127 cells off by >2 |
| 4 — biting order M5 > redwood > MKT | **FAILS** | redwood 34.5% > M5 23.8% > MKT 22.5% |

**Prediction 1's pass carries no information and I am not claiming the registered conditional
fired.** The conditional was: *if prediction 1 passes on MKT while amendment 8's table holds, the
excess statistic is wrong rather than the thread rule right.* But prediction 1 passes when no
single `G` clears the threshold at every width, and on 50–79% of cells **no tested `G` reaches
within 1 of the measured optimum at all**. A width where most cells are unreachable by the whole
family makes "no single G clears it" true for a reason unrelated to linear-versus-sqrt. The
instrument says exactly this in its own text; taking the pass at face value would be reading the
null as the result.

So **the registered prediction set for the sqrt thread-sizing model has failed or is
uninformative in every one of its four parts**, measured on three hosts spanning 6 to 32 workers
and 0.06 to 4.75 µs per worker. Together with amendment 8's construction argument — the resolved
count is identical on 392 of 392 keys at k=2..16 while the deficit falls 4.4× — **thread sizing is
closed as a mechanism for the narrow-k deficit.**

### A near-miss worth recording: the statistic that scores prediction 4 was redefined mid-flight

The REPLACEMENT CONTROL used to read *"LIVE cells whose largest reachable count already equals the
clamp"* and now reads *"whose measured OPTIMUM sits at or above the clamp"*. The M5 figure under
the old definition was **345 of 345, 100.0%**; under the current one it is **82 of 345, 23.8%**.
Had prediction 4 been scored by comparing the M5's archived 100% against MKT's 22.5%, the ordering
would have read as a spectacular confirmation.

**Two numbers from the same named control, differing by a factor of four, because the control was
improved between hosts.** This is why all three were re-run under one digest. A cross-host ordering
is the one claim where a redefinition is indistinguishable from an effect — both appear as a
difference between hosts — and nothing in either output would have flagged it.

### Separately: ADOPT_GRAIN does not ship on MKT

The three-object screen (`adg0`/`adgb`/`adgG`, three passes, 1296 cells) prices
`SCORCH_SPMM_ADOPT_GRAIN`:

- **Placement floor** (`adgb/adg0`, one configuration built twice): median 0.9918–1.0019, p95
  deviation 2.58–5.28%.
- **The arm's pooled CI overlaps the floor's pooled CI at all six widths** (k=2,4,8,12,16,32).
- The gate is **not inert**: it changes the resolved count on **30–31 of 72 cells, 41.7–43.1%**, at
  every width.
- **58 harmed cells**, up to **1.44×**, and **all 58 changed the worker count** — so none of the
  harm is the floor's tail showing through the gate. By row count `{64: 10, 128: 22, 256: 20,
  1024: 3, 2048: 3}`, by nnz decade `{3: 4, 4: 54}`.

A lever that fires on 42% of cells, produces no pooled effect beyond the placement floor, and
harms 58 cells by up to 1.44× is a no-ship. Recorded as a bounded negative, not as noise.

---

## Amendment 19 — amendment 18 scored the wrong set and two rows on the wrong host; amendment 5's prediction 1 is now scored and FAILS

> **PARTLY CORRECTED BY AMENDMENT 20.** This amendment's own scorecard lists three of
> amendment 5's four elements and marks two of those three unscored, when the ledger had
> already scored both and its own table held the fourth element's verdict. Element 2 HOLDS
> on its registered host. Read amendment 20.

Three defects in amendment 18, all found by a reviewing session from inconsistencies between my own
published numbers, and one registered prediction scored for the first time.

### (a) "All four registered predictions" has no fixed domain, because this document carries two sets both numbered 1–4

- The original registration at `:57-68` holds out prediction **2** (other widths {4,12,16,20,24}),
  **3** (other matrix classes, the 72-key corpus), **4** (the two x86 hosts).
- **Amendment 5** at `:288-296` registers a *different* set of four with its own thresholds, to be
  "scored on the x86 sweeps when the screens clear": prediction **1** is *"MKT is NOT inert. The
  excess ... will exceed 1.05 at k=2 on MKT. The M5 reads 1.0032."*

Amendment 13 recorded amendment 5's element 1 as NOT SCOREABLE because "MKT's screen still running;
no sweep exists". **Amendment 18 opens by announcing that exact blocker resolved — and then scores
the other set.** Neither scorecard shows the gap, because each is complete over its own numbering.

**A completeness quantifier with no stated domain cannot be wrong.** "All four of the predictions
registered at `:57-68`" scores one set and leaves the other visibly open; "all four registered
predictions" closes both by assertion. The two numbering schemes are the defect; the heading is
only where it surfaced.

### (b) Amendment 5's prediction 1, scored at last: FAILS

Run on MKT's own sweep, `nt_k2curve.py` at `039bad6ff9b1ef22`, object `09b7b24cbceec93c`, **38016
of 38016 rows carrying an `so_sha`** — the first real grid on which the provenance guard of
amendment 17 has passed rather than refused.

| k | cells | median | null | **excess** | reading |
|---|---|---|---|---|---|
| **2** | 53 | 1.1430 | 1.1435 | **0.9996** | production beats an arbitrary count by 0.0% |
| 4 | 53 | 1.1054 | 1.1220 | 0.9852 | production beats arbitrary by 1.5% |
| 8 | 53 | 1.1343 | 1.1092 | 1.0226 | 2.3% recoverable |
| 12 | 53 | 1.1572 | 1.1117 | 1.0409 | 4.1% recoverable |
| 16 | 53 | 1.1589 | 1.1133 | 1.0409 | 4.1% recoverable |
| 20 | 54 | 1.1468 | 1.0826 | **1.0593** | 5.9% recoverable |
| 24 | 57 | 1.1056 | 1.0860 | 1.0181 | 1.8% recoverable |
| 32 | 59 | 1.0680 | 1.0527 | 1.0146 | 1.5% recoverable |

**Registered: exceeds 1.05 at k=2. Measured: 0.9996** — not merely under the threshold but under
**1.00**, meaning production's count is already at least as good as an arbitrary one. The
prediction FAILS, and it fails in the direction that *supports* amendment 18's closure rather than
undermining it.

**One thing stated precisely because it would otherwise read as a near-miss.** The 1.05 threshold
**is** exceeded — at k=20, reading 1.0593. The registration named **k=2**. So there is recoverable
thread-sizing headroom on MKT, 1.5% to 5.9%, at k=8 through k=32; there is **none at k=2**, which
is the width holding 129 of the 192 float64 deficit cells and 328 of the float32 DLMC deficit. The
closure claim is specifically *"closed as a mechanism for the narrow-k deficit"* and the registered
test now confirms it at the narrow width while refuting inertness at the wide ones.

### (c) Amendment 18's predictions 2 and 3 are MKT's figures for M5 predictions

Prediction 2 is registered as *"Other widths on the **same host**"* — the fit host, the M5.
Amendment 18 reported MKT's numbers for both. Re-run under the same digest:

| prediction | M5 (registered host) | MKT (what amendment 18 printed) |
|---|---|---|
| 2 — within ±1 at {4,12,16,20,24} | lowest width **77.8%** → HOLDS at 50%, FAILS at 80/100 | lowest 17.4% → FAILS at all three |
| 3 clause A — ≥80% within ±1 | **87.5% of n=160 → HOLDS** | 20.0% → FAILS |
| 3 clause B — none off by >2 | **2 cells** → FAILS | 127 cells → FAILS |

**So prediction 3 does not "fail both clauses" on its registered host: clause A holds at 87.5%
with 7.5 points of headroom, and only clause B fails, on two cells.** Amendment 18's row was
67.5 points and 125 cells away from the registered reading, with no note that either figure had
moved — in the amendment whose own thesis is that a statistic must not be redefined between
scorings.

The tool did not change: 105 insertions, 4 deletions, no hunk in the prediction-3 block, and the
M5's clamp figure (82 of 345) is identical across both publications, so the M5 sweep is unchanged
and its 87.5% stands. **The input changed hosts, not the estimator.** Every scorecard row now
carries its host.

(Incidental, found in the same diff: the `sat` assignment occurs at `:49` and `:58` and is never
read, so an earlier "saturating was not the policy count" fix landed on a variable nothing
consumes.)

### The corrected scorecard, with hosts

| set | prediction | host | verdict |
|---|---|---|---|
| orig | 1 — no single G suffices | MKT | passes, **uninformative** (55.2–78.6% of cells unreachable by any G) |
| orig | 2 — within ±1 at five widths | **M5** | HOLDS at 50%, **FAILS** at 80/100 (77.8%) |
| orig | 3 — ≥80% and none off by >2 | **M5** | clause A **HOLDS** 87.5%; clause B **FAILS** (2 cells) |
| orig | 4 — biting order M5 > redwood > MKT | all three | **FAILS** (redwood 34.5% > M5 23.8% > MKT 22.5%) |
| amd 5 | 1 — excess > 1.05 at k=2 on MKT | MKT | **FAILS** (0.9996) |
| amd 5 | 2 — headroom concentrates at small counts | MKT | not computed by this tool |
| amd 5 | 3 — redwood between the hosts | redwood | running |

**The closure stands and now rests on a scored prediction rather than an unexamined one.** What
does not stand is amendment 18's claim to have scored everything registered.

### A prose correction that strengthens prediction 4's refutation

Amendment 18 said `T1 >= c·pool²` needs "61 µs on MKT against 150 µs on redwood and far more on
the M5". The M5 is **4.75 × 36 = 171 µs** against redwood's 150 — **1.14×, not "far more"**. So the
model's own thresholds put the M5 and redwood within 14% of each other while MKT sits 2.8× below
both; the measured clamp shares read M5 and redwood **10.7 points apart** and M5 and MKT **1.3
points apart**. The model's near-tie reads far apart and its far-apart pair reads nearly equal.
That is a stronger refutation than the one amendment 18 wrote, and the refutation is robust to the
sign convention: redwood is the middle term under either direction of "bites" and it reads largest.

---

## Amendment 20 — element 2 HOLDS on its registered host at all eight widths, element 4's control FIRES, and amendment 19 reproduced at n=3 the defect it diagnosed at n=4

Amendment 19 named amendment 5's set as the unscored domain, scored element 1, and gave every
scorecard row its host. It then listed **three** of the four elements and marked **two of those
three unscored** — when the ledger had already scored both, and when its own table one row up held
the fourth element's verdict. Scoring only the element that prompted the finding reproduces the
defect with a smaller n.

All numbers below come from `nt_k2curve.py` at `977c9d0d59f8e7f0`, which now computes the subgroup
split from the **same per-cell triples as the pooled row**, so the two cannot disagree by
construction. MKT: `/scr/bobbyy/adg/csv/oracle.csv`, object `09b7b24cbceec93c`, 38016 of 38016 rows
carrying an `so_sha`. redwood: `oracle_redwood.csv`, object `8192715070225fd3`, 28800 of 28800.

### (a) Element 2 HOLDS on MKT at every width, and the pooled statistics hid it

Element 2, `:292-294`: *"The headroom concentrates where the resolved count is small. **Among MKT
k=2 live cells**, the excess for cells resolving to `<=2` workers will exceed that for cells
resolving to `>=8` workers."*

MKT, the registered host:

| k | ALL n | ALL excess | `<=2` n | `<=2` excess | `3-7` n | `3-7` excess | `>=8` n | `>=8` excess | direction |
|---|---|---|---|---|---|---|---|---|---|
| **2** | 53 | 0.9996 | 9 | **1.3454** | 11 | 0.9243 | 33 | **0.9358** | **AS REGISTERED** |
| 4 | 53 | 0.9852 | 9 | 1.1573 | 11 | 0.8923 | 33 | 0.9292 | as registered |
| 8 | 53 | 1.0226 | 9 | 1.3401 | 11 | 0.9844 | 33 | 0.9431 | as registered |
| 12 | 53 | 1.0409 | 9 | 1.4255 | 11 | 0.9780 | 33 | 0.9462 | as registered |
| 16 | 53 | 1.0409 | 9 | 1.3585 | 11 | 0.9839 | 33 | 0.9362 | as registered |
| 20 | 54 | 1.0593 | 8 | 1.3414 | 12 | 1.0199 | 34 | 0.9964 | as registered |
| 24 | 57 | 1.0181 | 10 | 1.2324 | 10 | 1.0510 | 37 | 0.9670 | as registered |
| 32 | 59 | 1.0146 | 10 | 1.2242 | 9 | 0.8878 | 40 | 0.9977 | as registered |

**Eight of eight, in the registered direction, by margins of 0.23 to 0.48 on a statistic whose
pooled value never leaves 0.9996–1.0593.** At the registered width the small-count cells carry
**34.5% recoverable headroom** against the large-count cells' 0.9358 — production already beating
an arbitrary count by 6.4% there.

**This changes what the closure can claim.** Element 1's pooled 0.9996 is not in conflict with it:
33 of the 53 k=2 cells resolve to `>=8` workers and read 0.9358, so the union median is dominated
by the group with no headroom. (The dominance check confirms the routing: 1.1430 lies inside
[1.0701, 1.3859].) So thread sizing is **closed for the `>=8`-worker majority and OPEN for the
`<=2`-worker minority**, and the prediction that said so was registered in advance, on this host,
at this width, before either sweep ran. That is the opposite of a shape tuned around after the
fact — but it is also **9 cells of 53**, and whether those 9 intersect the deficit sets this
project is trying to close is a separate question that no measurement here answers.

### (b) Element 4 is the registered negative control and it fires

`:297-301`: *"At k=32 the work measure `nnz*max(k,16)` is `nnz*32` ... Prediction 1's statistic at
k=32 will be below its k=2 value on MKT. **If k=32 shows equal or greater excess, the account
above is wrong and the deficit is not a thread-count story at all.**"*

Amendment 19's own table, same row block: **k=2 0.9996, k=32 1.0146.** Greater, not below. The
control fires on the statistic it names — element 1's, which is the pooled one. Amendment 13 had
recorded the same direction on redwood and flagged it *"not its host, but contrary"* (1.0330
against 0.9584), so the control is contrary on **both x86 hosts**, one of them the host it names.

**The direction is favourable, which is why it went unnoticed.** A control that fires unreported
costs the conclusion its pre-registration: the closure then rests on an argument assembled after
the data, and nothing in the write-up shows the difference.

**And one thing the pooled reading does not see.** Inside the `<=2` group — the subpopulation
element 2 identifies — k=32 reads **1.2242 against k=2's 1.3454**, which is *below*, as element 4
registered. So the control fires on the pooled statistic and holds within the small-count group.
Element 4 names "prediction 1's statistic", which is pooled, so **the firing is the verdict**; the
subgroup reading is recorded because element 4's purpose was to test the thread-count account, and
the account survives its own control exactly where element 2 says the headroom is.

### (c) Two of amendment 13's three element-2 figures were wrong, and the verdict they supported was on the wrong host

Amendment 13's row read: *"k=2: `<=2` workers 0.9584 (n=9) vs `>=8` workers **1.0319**. k=32:
1.0523 vs 1.0550. Direction REVERSED at both widths"*, concluding the prediction *"should be
considered settled."*

`0.9584` is redwood's **pooled** k=2 excess — cited as such twenty lines earlier with its own null
(1.2176) beside it, and `1 - 0.9584 = 4.2%` matching the sentence that uses it. A union statistic
landing on a 9-cell minority subgroup's value to four decimals is a transcription. A peer session
derived that from the published table alone, with no access to the run.

Recomputed on redwood under the committed instrument:

| k | ALL excess | `<=2` n / excess | `3-7` n / excess | `>=8` n / excess | direction |
|---|---|---|---|---|---|
| **2** | 0.9584 | 9 / **0.9383** | 11 / 1.0070 | 33 / **0.9771** | **REVERSED** |
| 4 | 0.9547 | 9 / 0.9726 | 11 / 1.0155 | 33 / 0.9337 | as registered |
| 8 | 0.9781 | 9 / 0.9883 | 11 / 0.9285 | 33 / 0.9617 | as registered |
| 12 | 0.9928 | 9 / 1.0261 | 11 / 1.0198 | 33 / 0.9849 | as registered |
| 16 | 1.0037 | 9 / 1.1057 | 11 / 1.0355 | 33 / 0.9602 | as registered |
| 20 | 1.0143 | 8 / 1.0531 | 12 / 1.0122 | 34 / 0.9974 | as registered |
| 24 | 1.0029 | 10 / 1.0780 | 10 / 0.9510 | 37 / 0.9662 | as registered |
| **32** | 1.0330 | 10 / **1.0174** | 9 / 0.9953 | 40 / **1.0727** | **REVERSED** |

`<=2` reads **0.9383**, not 0.9584. `>=8` reads **0.9771**, not 1.0319. At k=32 it is 1.0174 vs
1.0727 — a 0.055 gap where amendment 13 published 0.0027, twenty times narrower. **`1.0319` matches
no value this instrument produces at any width under either subgroup, so its origin is
unrecoverable.** The script that produced all three was never committed; that is the whole reason
they could not be checked.

**Three things follow, and they are different claims.** The redwood k=2 direction is REVERSED, as
amendment 13 said — so on redwood the figures were wrong and the sign was right. The word
*"settled"* is refuted: six of the eight widths go the registered way on redwood, and the two
widths amendment 13 published are the only two that reverse. Those six are outside the
registration, so this is not scoring on the outcome — but a verdict generalized past its registered
width, resting on the only two contrary widths, is contradicted at three quarters of what was
measured.

> **A cleaner statement of the same structure, from a peer session, worth preferring to the one
> above.** Element 2 names k=2 and element 4 names k=32, so **the two widths amendment 13 published
> are the two REGISTERED widths**, not a selection out of eight. The six that go the registered way
> are unregistered scope and do not overturn a k=2 verdict. What the recomputation shows is that
> **the two hosts disagree at exactly the two registered widths and agree with the registered
> direction at the unregistered ones** — 8 of 8 on MKT, 6 of 8 on redwood. The wrong-host point in
> (a) stands on its own and needs none of the width counting. And the verdict itself was **on the wrong host**: element 2 names MKT, where it holds at
all eight.

### (d) Element 3, re-scored on measurements instead of against a prediction

Amendment 13 scored it as *"redwood 0.9584 is BELOW the M5's 1.0032, not between it and a predicted
`>1.05`"* — a measurement against the *prediction* element 1 made, because MKT had no sweep then.
With MKT measured it re-scores on measurements only, three hosts under one tool digest:

| host | `c` (µs/worker) | pool | k=2 pooled excess |
|---|---|---|---|
| M5 | 4.75 | 6 | 1.0032 |
| MKT | 0.06 | 32 | 0.9996 |
| redwood | 0.26 | 24 | **0.9584** |

redwood is below **both**, so element 3 still **FAILS** — and the shape is the finding. The two
hosts **79× apart in `c`** sit **0.0036 apart** while the middle host sits **0.0412 below both,
11× the entire span of the pair it was predicted to fall between.** Non-monotone in `c`, from a
statistic independent of the clamp-share measurement that refuted the original set's prediction 4
in the same shape — an unplanned second corroboration.

The registered failure mode was the opposite one: *"the pool clamp may absorb [it] entirely on
redwood's pool of 24"*, which would pull redwood toward inertness. It went the other way; redwood
is where production's count beats an arbitrary one by the widest margin. **The prediction failed,
and not by the mechanism its registration named.**

### (e) The structural check: what it catches, and the two things it does not

The census prints the union row beside the subgroups and checks the bound that makes a misrouted
pooled figure visible: **a median over a union lies between the smallest and largest subgroup
median** — if it did not, more than half of every subgroup would sit on one side of it, hence more
than half of the union. Checked on the median and on the null **separately**, because the excess is
a ratio of two medians and is not itself a median, so the bound does not transfer. The middle group
`3-7` is printed although the registration does not name it, because a pooled value outside the
range of the *named* groups is explicable only by an unnamed one.

**It fires on its own input.** A planted routing defect — the union drawn from the widest width's
population while the subgroups keep the current width's — gives `rc=1` and six flagged widths.

**Limit one, from the plant itself.** All six failures were on the **null**, none on the median: the
pooled median moves only from 1.1611 to 1.1934 across widths on redwood, so substituting another
width's median usually lands inside the subgroup range. The null bound carried the detection. The
two checks are not redundant and the median's is the weaker.

**Limit two, which is the defect that actually occurred.** Amendment 13's error was a number
transcribed into prose from another row. No check inside the producing script sees that, and
inclusive range-containment admits a pooled value *equal* to a subgroup's. Only recomputation found
it. The sharper reading — that a union statistic *cannot* land on a minority subgroup's value — is
a plausibility argument, not a theorem: with 53 cells split 9/11/33, a union median as low as the
9-cell group's is arithmetically reachable. **The four-decimal coincidence was the tell, not an
impossibility, and the weaker claim is the one that holds.**

### The scorecard, replacing amendment 19's three rows

| set | prediction | registered host | verdict |
|---|---|---|---|
| amd 5 | 1 — pooled excess > 1.05 at k=2 | MKT | **FAILS** (0.9996, below 1.00) |
| amd 5 | 2 — `<=2` workers exceed `>=8` at k=2 | MKT | **HOLDS**, 1.3454 vs 0.9358, and at all 8 widths |
| amd 5 | 3 — redwood between the hosts | redwood | **FAILS** (0.9584, below both; 11× the pair's span) |
| amd 5 | 4 — k=32 pooled excess below k=2's | MKT | **CONTROL FIRES** (1.0146 > 0.9996); contrary on redwood too |

### What the closure may now say, and what it may not

**May:** the pooled thread-sizing headroom on MKT at k=2 is 0.9996 — production already at least as
good as an arbitrary count — the registered negative control fired on its own host, and the one host
constant the model depends on is non-monotone across three hosts on two independent statistics. For
the 33 of 53 k=2 cells resolving to `>=8` workers, thread sizing is closed on measurement, not on
argument.

**May not:** that thread sizing is closed everywhere. A prediction registered before either sweep,
naming where to look, holds at all eight widths on its own host with 34.5% recoverable headroom at
k=2 in the `<=2`-worker group.

### The registered next measurement, and its refutation condition, written before it runs

The question is whether the `<=2`-worker group intersects the deficit sets. **Registered: compute
the overlap between MKT's 9 `<=2`-worker k=2 cells and (i) the 328 float32 DLMC k=2 deficit cells
and (ii) the 129 float64 MKT k=2 deficit cells, by key.** ~~`cand = min(override, rows/16)`, so a cell
resolving to `<=2` has `rows < 48`; every one of the 20 worst float32 DLMC cells reads `rows=512`,
which clamps at 32.~~ **WRONG, corrected in amendment 21: the clamp is a CEILING on the count, not
the count. The nine members read rows 64 to 2048, one of them exactly the 512 this sentence claims
is excluded.** **Refutation condition: if the overlap is zero at both, the `<=2`-worker
headroom is real, registered, and irrelevant to this project's goal, and thread sizing closes for
the deficit on that basis rather than on the pooled number.** If the overlap is non-zero, the group
is named and a fixed-count arm on exactly those keys is the next arm. No mechanism is proposed for
either outcome in advance.

---

## Amendment 21 — amendment 20's row-count argument is wrong, and its registered overlap measurement has almost no power on the existing corpora

Amendment 20 registered a measurement and gave a reason to expect its answer. The reason was wrong
and the measurement is now done.

### (a) The clamp is a ceiling on the worker count, not the count

Amendment 20 wrote: *"`cand = min(override, rows/16)`, so a cell resolving to `<=2` has
`rows < 48`; every one of the 20 worst float32 DLMC cells reads `rows=512`, which clamps at 32."*

The nine members of the `<=2`-worker group at k=2 on MKT, named by `nt_k2curve.py` at
`977c9d0d59f8e7f0`:

| resolved | cur/best | null | rows | nnz | deg | key |
|---|---|---|---|---|---|---|
| 1 | 1.5396 | 1.2775 | **512** | 15612 | 30.49 | `transformer/l0_regularization/0.95/…encdec_attention…output_transform` |
| 2 | 1.5168 | 1.2965 | **2048** | 20971 | 10.24 | `transformer/random_pruning/0.98/…decoder_layer_3_ffn_conv1…` |
| 2 | 1.4197 | 1.3220 | 1024 | 23655 | 23.10 | `rn50/extended_magnitude_pruning/0.91/bottleneck_3_block_group3_5_1` |
| 1 | 1.3986 | 1.0192 | 128 | 15187 | 118.65 | `rn50/variational_dropout/0.95/…projection_block_group2` |
| 1 | 1.3859 | 1.1886 | 1024 | 13107 | 12.80 | `rn50/magnitude_pruning/0.95/…projection_block_group3` |
| 2 | 1.3761 | 1.0201 | 256 | 23655 | 92.40 | `rn50/extended_magnitude_pruning/0.91/bottleneck_1_block_group3_3_1` |
| 1 | 1.3348 | 1.0139 | 128 | 13107 | 102.40 | `rn50/random_pruning/0.8/bottleneck_1_block_group2_3_1` |
| 1 | 1.3122 | 1.0301 | 256 | 11796 | 46.08 | `rn50/magnitude_pruning/0.98/bottleneck_2_block_group3_1_1` |
| 1 | 1.2436 | 1.0087 | 64 | 11059 | 172.80 | `rn50/random_pruning/0.7/bottleneck_2_block_group1_1_1` |

Rows run 64 to 2048, and the **first row reads exactly the 512 that amendment 20 named as the
counter-example.** `min(override, rows/16)` bounds the count from above; production's rule then
chooses below it on total work, so a 2048-row matrix resolves to 2 workers because `nnz` is 20971.
**The group is a small-WORK group — nnz 11059 to 23655, every one of the nine — not a small-rows
group.** Mean degree spans 10.24 to 172.80, so degree does not separate it either.

The line that caught this is the one printed beside the table: *"a cell resolving to `<=2` workers
is a claim about its row count that the shape columns can confirm or contradict."* It contradicted
it on the first row. **A claim stated next to the columns that test it does not survive being
wrong, which is the only reason to put it there.**

### (b) The registered overlap, measured — and nearly powerless

Amendment 20 registered: *"compute the overlap between MKT's 9 `<=2`-worker k=2 cells and (i) the
328 float32 DLMC k=2 deficit cells and (ii) the 129 float64 MKT k=2 deficit cells, by key"*, with
the refutation condition *"if the overlap is zero at both, the `<=2`-worker headroom is real,
registered, and irrelevant to this project's goal."*

**(ii) is zero without a measurement, by corpus disjointness.** The nine keys are DLMC
(`<model>/<method>/<sparsity>/<layer>`); the float64 MKT deficit is SuiteSparse (`ss/<group>/<name>`).
No join is needed and none was run.

**(i) answered by the committed instrument.** `an_f64_vs_mkl.py` at `9c82ae31191662bc` now takes
`--probe-keys` and answers the join itself, because a join answered by an uncommitted script is
exactly how two of amendment 13's three figures became unreproducible. On
`/scr/bobbyy/pfs/csvdlmc` — the MKT float32 DLMC screen, identified by its `k` set
`{2,8,12,16,20,24,32}` and DLMC key shape rather than by its filename, which `POOL_PROVENANCE.txt`
records as colliding with redwood's float64 files — it reproduces the published board exactly:
`readings per cell {16: 2744}`, 662 behind in every reading, per-width `{2:328, 8:149, 12:102,
16:71, 20:6, 24:4, 32:2}`.

```
9 keys requested, 1 present in this screen's corpus
rn50/magnitude_pruning/0.98/bottleneck_2_block_group3_1_1
    k=2 behind   k=8 mixed   k=12 mixed   k=16 ahead   k=20 ahead   k=24 ahead   k=32 ahead
BEHIND-IN-ALL among the probed keys, per width: {2: 1, 8: 0, 12: 0, 16: 0, 20: 0, 24: 0, 32: 0}
```

**The overlap is 1, and the refutation condition does not fire. But the honest verdict is that the
question is not answerable at useful power on the existing corpora.** Eight of the nine keys are
**absent** from the screen corpus — the thread-count oracle ran on `adg_keys.txt` (72 keys) and the
float32 screen on `dlmc_keys.txt` (392 keys), and these two lists share **one** of the nine. So the
measured intersection is 1 of a possible 1, not 1 of a possible 9. "Zero overlap" and "nine of nine"
are both consistent with this measurement, and the instrument prints the denominator for that
reason: *"the denominator is 1 probed key present, NOT the 9 requested; a key absent from this
corpus is unmeasured here, not ahead."*

**The single testable case points the same way as the hypothesis**, which is worth one sentence and
no more at n=1: that key is behind MKL in all 16 readings at k=2 and **only** at k=2 (mixed at 8 and
12, ahead at 16 through 32), and it carries `cur/best 1.3122` against a null of 1.0301 — excess
1.274, so 27% of its apparent headroom survives the winner's curse. A cell that is behind MKL
exactly where it has recoverable thread-sizing headroom, and ahead everywhere else, is the shape the
hypothesis predicts. It is also one cell.

### (c) The registered next measurement, and why this one is not overfitting

The defect is that the oracle and the screen were run on **different corpora**, so the one
population that matters — the 328 — has never had its thread-count curve measured at all. The fix is
to measure the registered statistic on that population instead of inferring it from a 72-key list
that barely intersects it.

**Registered, before it runs:** run `ntoracle.py` on the **328 float32 DLMC k=2 deficit keys**, same
widths, same `--reps 2 --target-ms 10`, and compute `nt_k2curve.py`'s excess and its `<=2` / `3-7` /
`>=8` subgroup split on exactly that set.

**Refutation conditions, all three written now.**

1. **If the pooled k=2 excess on the 328 is at or below 1.00**, thread sizing is closed for the
   deficit on its own population, and amendment 20's `<=2` result is a property of the 72-key
   corpus's small-work tail rather than of the cells this project must fix.
2. **If the `<=2` group on the 328 is smaller than 5% of the set**, the mechanism cannot account for
   the deficit however large its per-cell excess, and the result is reported as a bound on reach
   rather than a candidate.
3. **If the `<=2` group's excess on the 328 does not exceed the `>=8` group's**, the element-2
   direction does not transfer from the 72-key corpus to the deficit population, and the MKT result
   stands as corpus-specific.

**This is the generalizable form of the question and not a tuning step.** The statistic, its null,
the subgroup boundaries and the three refutation conditions are all carried over unchanged from a
registration written before any of this was measured; the only thing that changes is the population,
and it changes to the one the project's goal is stated over. No rule, threshold or gate is proposed
by this measurement under any outcome — a kernel change would need a runtime-testable separating
property, and "resolves to `<=2` workers" is a property of the policy being tested, not an
independent one.

---

## Amendment 22 — the registered reach condition is in the wrong units, the control was missing, and the deficit's own prevalence bounds this mechanism before any of it runs

Three defects in amendment 21(c), raised by a peer session and verified here. The job it registered
(17698162) is cancelled and resubmitted under the amended design, with the thresholds fixed below
**before** any of it is read.

### (a) 83.7% against 17.0% bounds the mechanism with no measurement at all

The float32 DLMC board reads **328 of 392 keys behind MKL in all 16 readings at k=2 = 83.7%.** The
`<=2`-worker group is **9 of 53 live cells = 17.0%** on the only corpus where it has been measured
(9 of 72 keys = 12.5%).

**Even if every `<=2` cell were behind, a 17.0%-prevalence property cannot account for an
83.7%-prevalence deficit.** It is short by a factor of **4.9×** and leaves **66.7 percentage points**
unexplained. That is arithmetic over two already-published numbers.

**A mechanism's prevalence has to be comparable to the prevalence of what it explains.** This is the
reach question, not the effect question, and the two have different comparators: amendment 20's
1.3454 is an effect size and says nothing about how much of the deficit the effect can reach.

### (b) Condition 2's threshold was in the wrong units, so it passes at a value that refutes the claim

Amendment 21(c) registered: *"If the `<=2`-worker group on the 328 is smaller than 5% of the set,
the mechanism cannot account for the deficit however large its per-cell excess."*

Right instinct — bound the reach rather than only detect the effect — **wrong units.** 5% sits
**3.4× below the unselected 17.0%** and **16.7× below the 83.7%** that is the thing to be explained.
**At the only value ever measured, 17.0%, the condition PASSES while the mechanism explains at most
a fifth of the deficit.** A threshold that cannot distinguish "reaches a fifth" from "reaches most"
is not a reach condition.

**CONDITION 2, AMENDED AND FIXED NOW.** Let `s328` be the `<=2` share among the live k=2 cells of
the 328 deficit keys.

- `s328 >= 0.50` — the mechanism reaches the majority of the deficit; a causal claim is available.
- `0.25 <= s328 < 0.50` — reported as a share with **no causal claim**: "reaches `s328` of the
  deficit", and the remaining 1 − `s328` needs a different mechanism.
- `s328 < 0.25` — **reported as a bound on reach, not as a pass.** Three quarters of the deficit
  would remain, so thread sizing cannot be its principal cause whatever the per-cell excess.

The boundaries are stated in units of the explained quantity: a mechanism cannot be the principal
cause of something it reaches in under a quarter of cases, and reaching over half is the weakest
version of "most". **This does not make the measurement vacuous** — `s328` could be far above 17.0%,
which is exactly the question, and the unselected 17.0% is not even a clean prior because
`adg_keys.txt` was stratified on eight row counts and a purposive sample over row counts may
over-represent the small-work tail amendment 21(a) identified.

### (c) The control was missing, and it is free and in the same run

Running the oracle on the 328 alone yields an `s328` **with nothing to compare it against**, and the
existing 17.0% is not a clean comparator for the reason just given. **The 64 keys of the 392 that are
NOT behind at k=2 are the clean control: same corpus, same run, same instrument, and outside the
population being certified** — which is the one property a control inside its own population cannot
have.

**CONDITION 4, REGISTERED NOW.** Let `s64` be the `<=2` share among the live k=2 cells of those 64
keys. The discriminating statistic is **`s328 - s64`**. If a bootstrap CI on that difference contains
zero, **the mechanism does not track the deficit whatever its per-cell excess** — the `<=2` share is
then a property of the corpus and not of being behind MKL. Its power is limited and stated with it:
64 keys, of which the live subset will be smaller, so a null result here is weak and a positive one
is not.

The run therefore sweeps **all 392 keys**, and the split into 328 and 64 happens in analysis, so both
shares come from one process on one object. The 328-only job is cancelled rather than supplemented,
because two jobs would put the arm and its control in different processes — the one difference this
project has repeatedly measured as larger than the effects it looks for.

### (d) The 20-key cross-tabulation needs no new run, and tests the association directly

`comm -12` over sorted copies of the two committed lists: `adg_keys.txt` (72) and `dlmc_keys.txt`
(392) share **exactly 20 keys — ten `rn50/…` and ten `transformer/…`.** So for 20 keys both the
thread-count curve and the ours-vs-MKL verdict already exist on disk.

**Amendment 21(b)'s join lost its power by conditioning on the predictor.** Nine keys drawn from 72
against a 20-key shared set have expected overlap `9 × 20/72 = 2.5`, so observing 1 is unremarkable;
conditioning on `<=2` keeps one cell while conditioning on nothing keeps twenty. The symmetric
question is a cross-tabulation at k=2 of resolved-count group (`<=2` / `3-7` / `>=8`) against
behind-in-all, over those 20 keys.

**Its power is stated before it runs and it is low.** With 83.7% of keys behind at k=2 the marginal
is extreme — roughly 17 of the 20 will be behind — so the test reduces to whether the few non-behind
keys sit at `>=8`. **Registered reading: only a cross-tabulation in which every non-behind key sits
at `>=8` counts as support, because any other arrangement is reachable by chance at this marginal.**
Measured and statable beats a one-cell join, and that is the whole of the claim.

### A correction to amendment 21(a), which overstated how many columns separate the group

Amendment 21(a) reported rows 64–2048, nnz 11059–23655 and degree 10.24–172.80 as three facts. The
degree extremes are **not independent**: `11059/64 = 172.797` and `20971/2048 = 10.2397`, so the
degree range is the same two cells read a second way. **The honest statement is that one shape column
separates the group — `nnz`, tightly, 11059 to 23655 across all nine — and the other two do not.**
And on the clamp: `rows=64` gives `rows/16 = 4`, so the ceiling was 4 and production chose 2 beneath
it, which is the ceiling-not-the-count point in its sharpest form.

---

## Amendment 23 — the only direct test of the association runs OPPOSITE to the hypothesis, and that is a prediction for the job now running

> **VOID. The table was built on the wrong population and the directional prediction is
> WITHDRAWN by amendment 25, before job 17698209 reported.** The `<=2` cell held seven keys
> of which six are NON-LIVE — one reachable count, so the headroom statistic is undefined
> there, not small — and all four non-behind keys the verdict turned on were among them. On
> the effect's own population the sign reverses and n is 1.

Amendment 22(d) registered the 20-key cross-tabulation and its reading before running it.
`an_xtab_resolved_behind.py` at `eb71b2c2f0034df2`, on MKT's existing oracle (38016 rows, 72 keys
with a production count at k=2) against the float32 DLMC board's behind set (328 of 392 = 83.7%,
emitted by `an_f64_vs_mkl.py --dump-behind`):

| resolved group | behind | not | total | behind share |
|---|---|---|---|---|
| `<=2` | 3 | 4 | 7 | **42.9%** |
| `3-7` | 0 | 0 | 0 | n=0 |
| `>=8` | 11 | 2 | 13 | **84.6%** |
| TOTAL | 14 | 6 | 20 | 70.0% |

**The `<=2` cells are HALF as likely to be behind MKL as the `>=8` cells, 42.9% against 84.6%.**
The hypothesis was that the thread-sizing headroom and the deficit live in the same cells. The only
direct test of that association puts them in *different* cells.

**Registered reading, applied: NOT SUPPORT.** Four of the six non-behind keys sit at `<=2`, and
amendment 22(d) registered support as requiring *every* non-behind key at `>=8`.

**Three things said precisely, because the result is weak and favourable to nothing.**

1. **n=20 and Fisher's two-sided exact p on the collapsed 2×2 `[[3, 4], [11, 2]]` is 0.1219.** The
   association is **not established**. What is established is that the point estimate is in the
   opposite direction to the hypothesis, which is a different and smaller claim.
2. **The marginal here is 70.0%, not the corpus's 83.7%.** The power argument registered in
   amendment 22(d) assumed ~17 of 20 behind; the shared 20 are less behind than the corpus, which
   gave the table more discriminating room than registered — so the low-power caveat is *milder*
   than it was written, and the registered reading still returns NOT SUPPORT.
3. **The `3-7` group is empty among the shared 20**, so this is a 2×2 and not the 2×3 the
   registration described. Nothing was collapsed to make it so.

### The directional prediction this makes for job 17698209, registered before it reports

If the `<=2` share is *lower* among behind cells than among non-behind ones, then **condition 4's
statistic `s328 - s64` should be NEGATIVE.** That is registered now, from data independent of the
run: the 20-key table is the old 72-key oracle against the screen's board; the running job sweeps
all 392 keys fresh.

- **`s328 - s64` negative, CI excluding zero** — the `<=2`-worker headroom is *anti*-correlated with
  the deficit. Thread sizing closes for the deficit, not because the headroom is absent but because
  it is somewhere else, and the 20-key table predicted it.
- **CI containing zero** — condition 4's registered verdict applies unchanged: the share is a corpus
  property, not a deficit property.
- **`s328 - s64` POSITIVE, CI excluding zero** — the 20-key table is refuted by a larger sample on
  its own corpus, and the reach question falls back to condition 2's bands on `s328` alone. **This
  is the outcome that would reopen the mechanism**, and naming it now is the point of writing this
  before the job reports.

### And the instrument's own guard, because the first version could not have produced a table

A first version selected production's count by filtering rows on `probe_ovr == 0`. If that column
never holds 0 the filter yields an empty `res` map, every shared key drops, and the output reads
*"all shared keys are behind, NOT SCOREABLE: this is the marginal doing the work"* — a sentence that
is true of an empty table and describes a measurement. It now reads `res_ovr0` from **every** row of
the cell and REFUSES if a key carries more than one value, so the grouping cannot depend on row
order. The Fisher implementation is validated against two published tables before use: the
tea-tasting `[[3,1],[1,3]]` at 0.4857142857 and `[[1,9],[11,3]]` at 0.0027594.

---

## Amendment 24 — condition 2's units were fixed but not its BASE, and condition 4's difference carries a composition effect. Both pinned before job 17698209 reports.

Amendment 22(b) put condition 2 in units of the explained quantity. It did not say which
**denominator** the numerator sits on, and amendment 22(a) names two without choosing: `9/53` live
cells **= 17.0%** and `9/72` keys **= 12.5%**, differing by the live fraction `53/72 = 0.7361`.

**The 83.7% is 328 of 392 KEYS.** So a share computed over the live subset while its thresholds
were calibrated against an all-keys prevalence is **inflated by 1.359×**. Against the registered
boundaries that is decisive, not cosmetic:

| registered boundary (all-keys) | the same number read live-only |
|---|---|
| 0.250 | **0.340** |
| 0.500 | **0.679** |

and inversely, a live-only reading of **0.368** would be an all-keys 0.500. **The base choice alone
moves a reading from "bound on reach" to "share with no causal claim", and from that to "licenses a
causal claim."** The threshold was in the right units and the numerator was not pinned.

### CONDITION 2, BASE FIXED

**`s328` is computed over all 328 deficit keys, with non-live keys in the denominator** — the same
base as the 83.7%. A non-live key — one reaching a single worker count, so its curve is undefined —
can still be behind MKL, so excluding it from the subgroup denominator while the deficit's
denominator includes it is the mismatch. The live count is printed beside the share so the two bases
are both recoverable from the output, and the all-keys one is the one the bands are read against.

### CONDITION 4, COMPOSITION EFFECT NAMED

`s328 - s64` is a difference of prevalences **only if the live fraction is equal in both groups**,
and there is no reason it should be: behind cells may be systematically smaller and less live, which
is the **same small-work axis amendment 21(a) identified** as the one shape column separating the
`<=2` group. **Registered: the live fraction is reported per group beside the difference, and if the
two fractions differ by more than 5 percentage points the difference is reported as carrying a
composition effect shaped like the association under test, not as a difference of prevalences.** In
that case the all-keys form is the primary statistic and the live-only form is reported beside it.

### And the headline shortfall picked the conservative base by accident

Amendment 22(a) quoted **4.9×** from `83.7/17.0`. The base-matched figure is `83.7/12.5` = **6.7×**.
Understating a bound in the mechanism's favour is the safe error, but **a prevalence ratio has to
name its base**, and both of amendment 22(a)'s numbers now do: the shortfall is **6.7× on matched
keys**, 4.9× if the subgroup is read on live cells against an all-keys deficit.

**Timing, stated because it is what makes this an amendment and not a revision.** Job 17698209 was
submitted at the amended design and is running; `sbatch` spools the script at submit, so the four
conditions in its header are fixed as of submission. This amendment pins a **denominator for reading
them**, written before the job has produced an analysable CSV and before any share has been computed
on either base.

---

## Amendment 25 — amendment 23 is VOID: "unconditional" widened the predicate instead of the sample. Apparent power 20, real power 1. The prediction is withdrawn before readout.

Amendment 22(d) diagnosed amendment 21(b)'s join as **conditioning on the predictor** and replaced
it with an unconditional cross-tabulation. The replacement fixed the conditioning and **silently
widened the predicate**, which is the same error one level up.

### The arithmetic, from two numbers already published

The effect — `<=2` excess **1.3454** — was measured on the **nine LIVE `<=2` cells** of the 72-key
corpus at k=2. Amendment 21(b) reports *"9 keys requested, 1 PRESENT in the screen's corpus"*, and
the shared 20 **is** that intersection. **So exactly one of the nine is in the table. The table's
`<=2` cell held seven.** At least six of the seven cannot be in the mechanism's population, and the
registered verdict turned on *"four of the six non-behind keys sit at `<=2`"*.

### Confirmed at the source, and the confounding is near-total

`an_xtab_resolved_behind.py` at `db9b303887050e68` now counts liveness the way `nt_k2curve.py` does —
distinct `resolved` values for the cell across every override row — and prints both tables:

| | `<=2` | `3-7` | `>=8` |
|---|---|---|---|
| all 72 keys | **28** | 11 | 33 |
| LIVE (53) | **9** | 11 | 33 |
| NON-LIVE (19) | **19** | 0 | 0 |

**Every one of the 19 non-live keys resolves to `<=2`, and none to `3-7` or `>=8`.** Liveness and
the `<=2` group are therefore almost entirely confounded on this corpus: a cell resolving to `<=2`
workers is non-live **19 times in 28 (67.9%)**, because a small work measure under
`min(override, rows/16)` leaves only one reachable count. `3-7` and `>=8` are 100% live.

The two tables on the shared 20:

| population | `<=2` behind/not | `>=8` behind/not | registered reading | Fisher p |
|---|---|---|---|---|
| all 20 shared | 3 / 4 → 42.9% | 11 / 2 → 84.6% | **NOT SUPPORT** | 0.1219 |
| **LIVE 14 shared** | **1 / 0 → n=1** | 11 / 2 → 84.6% | **SUPPORT** | **1.0000** |

### Both readings are worthless, in opposite directions, and that is the finding

**The all-shared reading was built on cells where the quantity is undefined.** The four non-behind
`<=2` keys that produced NOT SUPPORT are all non-live: the live `<=2` cell is 1 behind and 0 not.

**The live reading returns SUPPORT vacuously.** With one cell in `<=2` and no non-behind key in it,
amendment 22(d)'s rule — *"support only if every non-behind key sits at `>=8`"* — **cannot fail**.
Fisher's p is exactly 1.0000. **A registered reading with no minimum-n clause can return SUPPORT
from a table that carries no information**, and that is a defect in the registration, not in the
data. The 20-key test is declared **VOID** rather than re-scored: a reading rule repaired after
seeing the table it will score is not a registration.

### Amendment 23's directional prediction is WITHDRAWN

It registered *"`s328 - s64` should be NEGATIVE"* from the all-shared sign (42.9% against 84.6%).
**That sign comes from six non-live cells.** On the effect's own population the direction reverses —
100.0% against 84.6% — on n=1.

**Withdrawal is still free and that is the only reason it is worth anything.** Job 17698209 is
running and has not produced an analysable CSV. Had it landed negative first, the prediction would
have "confirmed" a direction that was never the mechanism's, and nothing in the output could have
shown the difference. **Registering a direction before readout is right; registering it off a
contaminated sign spends the pre-registration for nothing.** Conditions 1 through 4 of amendments
21(c), 22 and 24 are untouched — they are defined on the 392-key sweep's own live cells, not on
this table.

### A second bound on reach, which does not depend on the deficit overlap at all

**The subgroup where element 2 locates the headroom is the subgroup where the headroom is most often
undefined.** Of 72 keys at k=2, the cells where a thread rule can act *and* the `<=2` excess applies
number **9 — 12.5%**. The other 19 `<=2` cells reach one worker count, so there is nothing for any
rule to choose. That is independent of whether the `<=2` cells are behind MKL, and it is the same
12.5% that amendment 24 fixed as the base-matched prevalence, now with a mechanical reason rather
than a coincidence of denominators.

### Two incidental facts worth recording, both weak

**The shared 20's marginal is low because of the non-live keys.** 70.0% behind against the corpus's
83.7% is `z = -1.66` (p ≈ 0.10). Decomposed: the 14 live shared keys are **85.7% behind** and the 6
non-live ones **33.3%**. So the shared set is not a less-behind random subsample — it carries six
single-count cells that are mostly ahead.

**And the `3-7` cell being empty among the 20 is not surprising after all.** `3-7` is 11 of 72 keys
= 15.3%, so `20 × 0.153 = 3.1` expected and 0 observed is unremarkable at this n. Amendment 23
reported the empty cell as a fact about the table; it is a fact about the sample size.

### The reusable form

**A table is only as informative as the cell where the effect could exist.** Widening a group to
include cells where the quantity is undefined is not more power, it is a different question — and
it reads as more power, because the n goes up. The nine-key join kept 1 cell by conditioning on the
predictor; the 20-key table kept 20 by redefining the predictor to a superset of the effect's
domain, and the effect's domain inside it was still 1 cell. **Apparent power 20, real power 1.**
Print the undefined-cell count per group, always, beside the n.

---

## Amendment 26 — a share has TWO degrees of freedom and amendment 24 pinned one. The numerator predicate is worth 3.11×, and it crosses a band boundary in the mechanism's favour.

Amendment 24 registered: *"`s328` is computed over all 328 deficit keys with non-live keys IN the
denominator, the same base as the 83.7%."* **That names the denominator. It does not name the
numerator predicate**, and amendment 25's own census is what decides it. Job 17698209 is still
running and has produced no analysable CSV, so this is still free.

### Three candidate shares on the reference corpus, from amendment 25's census

| reading | value | band it falls in |
|---|---|---|
| all `<=2` over all keys | **28/72 = 38.9%** | a share with **no causal claim** |
| live `<=2` over live cells | 9/53 = 17.0% | a bound on reach |
| **live `<=2` over all keys** | **9/72 = 12.5%** | a bound on reach |

**The unpinned sentence admits the first, which is 3.111× the third** — and the two land in
*different bands*. A registered condition that licenses a causal claim must not have a numerator
whose definition can move the reading across one of its own boundaries, least of all in the
mechanism's favour.

**The intent was unambiguous and only the sentence is defective.** Amendment 24 quotes `9/72 =
12.5%` as *"the base-matched prevalence"*, and amendment 25 derives the same 9 as *"the cells where
a thread rule can act AND the `<=2` excess applies."* Both are live-`<=2` over all keys. But the
sentence is what gets read after the job lands.

### CONDITION 2, NUMERATOR PINNED

**`s328` = (keys among the 328 that resolve to `<=2` AND are live, **both evaluated at
k=2**) / 328.** (The width clause is amendment 27's; the sentence as first committed named
none, and both conjuncts are per-(key, width) properties.) The all-`<=2` count and
the live count are both printed beside it, so all three shares are recoverable from the output and
no later reader has to guess which was meant.

**Why "denominator all keys, numerator live-only" is consistent rather than a contradiction of
amendment 24.** A non-live key reaches one worker count, so **a thread rule has nothing to choose
there and it cannot be within a thread-sizing mechanism's reach — but it can perfectly well be
behind MKL.** The denominator counts **what must be explained**; the numerator counts **where the
mechanism can act**. Amendment 24 fixed the base; this fixes the predicate; a share needs both
written down and neither implies the other.

### CONDITION 4, SAME PREDICATE

`s328 - s64` is a difference of **reach** shares, so **both terms take the live-and-`<=2`
numerator**. Amendment 24's 5-percentage-point live-fraction tripwire catches a composition effect
in the **denominator**; it does not catch a **numerator** that counts cells where the curve is
undefined. If the 392-key corpus shows the same near-total confounding as the 72-key one — 19 of 28
`<=2` cells non-live — an all-`<=2` numerator would report roughly three times the mechanism's
reach on both sides of the difference.

### What this makes the third unpinned degree of freedom found on one statistic

Amendment 22 fixed the **units** (in terms of the explained quantity's prevalence). Amendment 24
fixed the **base** (all keys, not live cells). This fixes the **numerator predicate** (live-and-
`<=2`, not all-`<=2`). **Each was invisible until the one before it was fixed, and each was worth
between 1.36× and 3.11× in the mechanism's favour.** The general question none of the three asked
until now is **on which cells is this quantity defined at all** — the same question amendment 25
answered for the cross-tabulation, one level up, and the one that would have pre-empted all three
had it been asked when the condition was first written.

### Two corrections to the audit trail, offered by the reviewing session and recorded because they change what the earlier findings rest on

**The mechanism proposed for amendment 47's `9.5×` was wrong while the finding was right.** The
review inferred that 9.5 required a comparator of 5.40 pp lying outside both published MKT ranges;
it is `31.27/3.30 = 9.48`, both published endpoints sharing k=2's numerator. **The provable part —
not width-matched, upper end understated — held and drove the correction; the proposed cause did
not.** A right finding from a wrong inference is still a finding, and the inference still has to be
withdrawn.

**And the window factor was verified as arithmetic rather than as a premise.** `sqrt(25/5.69) =
2.096` is correct and 25 ms was never the run's parameter. Checking a number's arithmetic is not
checking where the number came from.

**One strengthening of amendment 47 that follows for free.** The "does not close" conclusion is
robust to *every* possible width-matching, not merely to the bound: the **smallest attainable**
ratio is `31.27/4.89 = 6.394`, and `6.394/3.1266 = 2.045 > 1.9` already, so **no assignment of
MKT's per-width bands can rescue closure.** Publishing MKT float32's four width-matched bands would
narrow the currently 2.4×-wide window — which is set by the comparator's uncertainty and not by
anything physical — without being able to change the verdict. **Deferred rather than done now: the
data is on `mkt1` and job 17698209 is timing there, so even a short read competes with a live
measurement.**

---

## Amendment 27 — the numerator's third degree of freedom is a WIDTH, it does not lean, and counting conjuncts finds the class that units/base/predicate were instances of

Amendment 26's registered sentence — *"`s328` = (keys among the 328 that resolve to `<=2` AND are
live) / 328"* — **names no width**, and both conjuncts are per-`(key, width)` properties. Job
17698209 is still running, so this is still free.

### Pinned: both conjuncts at k=2

**k=2 is the right value and not a choice**: the 328 is itself defined at k=2 over N=16 readings,
and the whole reach question is about the width that carries the deficit. `an_reach328.py` already
takes `--k` as a **required** argument with the comment *"the behind set and the resolved count
both differ by width"*, so the implementation was pinned and the sentence was not. **The sentence is
what gets read.**

### The two conjuncts move OPPOSITELY in width, measured rather than argued

Both directions follow from element 4's own premise — *"at k=32 the work measure `nnz*max(k,16)` is
`nnz*32`, so the same matrices resolve to roughly twice the count"* — and both are visible in the
reference sweep:

| k | keys | `<=2` | live | live & `<=2` | registered share | all-`<=2` share |
|---|---|---|---|---|---|---|
| 2 | 72 | 28 | 53 | 9 | 0.1250 | 0.3889 |
| 4 | 72 | 28 | 53 | 9 | 0.1250 | 0.3889 |
| 8 | 72 | 28 | 53 | 9 | 0.1250 | 0.3889 |
| 12 | 72 | 28 | 53 | 9 | 0.1250 | 0.3889 |
| 16 | 72 | 28 | 53 | 9 | 0.1250 | 0.3889 |
| 20 | 72 | 26 | 54 | 8 | 0.1111 | 0.3611 |
| 24 | 72 | 25 | 57 | 10 | 0.1389 | 0.3472 |
| 32 | 72 | 23 | 59 | 10 | 0.1389 | 0.3194 |

**`<=2` falls monotonically, 28 → 23. Live rises monotonically, 53 → 59.** The conjunction is
therefore **non-monotone** and its span is **0.1111 to 0.1389 — 1.25×, and band-preserving**: every
width reads inside the same *bound on reach* band.

**And the five identical rows are the formula, not a coincidence.** `max(k, 16)` makes the work
measure `nnz*16` for every `k <= 16`, so the resolved count cannot vary there at all. The variation
begins at exactly the width where `k` starts exceeding 16.

### Why the other three leaked in the same direction, and why this one cannot

The numerator is a **conjunction of restrictions**: in the 328 **AND** resolving to `<=2` **AND**
live. For a conjunction, **the looser reading of any conjunct is a superset, and every cell it adds
is one where the effect is absent or undefined.** So a share with a conjunctive numerator is
inflated by every unpinned restriction, **with no coincidence required** — which is also why
amendment 26 could state the damage as exactly 3.11× from the reference census before any new data:
the loosest reading bounds it.

**The audit method that follows: count the conjuncts in the numerator's predicate and require each
to carry a stated value.** For `s328` the count is three — *"among the 328"* (pinned: behind in all
16 readings at k=2), *"resolves to `<=2`"*, *"live"*. Reviewing by diff finds the last repair's
flaw; counting conjuncts finds the class.

**The limit that keeps the rule true, stated because it would otherwise over-promise.** A
**parameter** is not a restriction. The width does not lean, precisely because the two conjuncts
move oppositely in it, so **its direction is not boundable by inspection** and the
loosest-reading argument is unavailable. It needs pinning for a different reason than the other
three: not because it leans, but because nothing lets you reason about it without a value.

> *"Every unpinned degree of freedom leans favourably"* is true of the three found and **false in
> general**. The true version: **every unpinned RESTRICTION leans favourably; parameter choices
> have to be pinned blind.**

### The instruments that score this, written and tested before the sweep reports

- **`nt_k2curve.py`** takes `--keys` so conditions 1 and 3 are scored on the deficit half and the
  control half **without reimplementing the excess estimator** — same binary, same permutation
  null, same seed, different population. It REFUSES when no requested key appears in the sweep, and
  prints the present/absent split so an absent key reads as unmeasured rather than excluded.
- **`an_reach328.py`** at `c8dea455180727e6` computes conditions 2 and 4, printing the registered
  share **and both alternative readings** beside it, plus the non-live count inside the `<=2` group
  and the live fraction per half, so every reading in amendment 26's table is recoverable from the
  output rather than from this document.
- **`an_reach328_selftest.py`**: **14 counted checks, all passing**, including that the registered
  numerator excludes non-live `<=2` cells while the alternative includes them, that the inflation
  equals `n_all/n_live_le2`, that the bootstrap CI excludes zero on an injected +0.40 and covers it,
  that a no-effect control spans zero, and that the **value** digest moves with the data while the
  **index** digest does not.
- **Planted mutation**: substituting the unpinned numerator — the exact defect amendment 26 pins —
  gives **3 of 14 FAILED, rc=1**. The suite fires on the one defect it exists to catch.

---

## Amendment 28 — the excess is a RANK test, not a headroom test. Three numbers refute the gloss I published at every width on three hosts, and the closure language is withdrawn.

A reviewing agent gave a counterexample. It reproduces exactly, through this script's own
`perm_null`:

> Three reachable counts timed at **1.0 / 1.2 / 2.0 µs**, **perfectly repeatable**, production
> picking 1.2. True headroom `cur/best` = **1.2000** — production is 20% off the best count with
> **zero measurement noise**. Measured null = **1.2000**. **Excess = 1.0000** — which the prose
> labelled *"nothing to recover."*

### Why, and it is structural rather than a tuning artefact

The permutation shuffles readings **among** the counts, so the null's support is `{t_i / min(t)}`
— the same set of ratios, reordered. **Its median is the MIDDLE-RANK count's ratio**, so per cell

```
gain / null  ≈  t_production / t_median-rank-count
```

**The excess asks whether production's count beats a TYPICAL reachable count. It never asks how far
production is from the BEST one.** When production sits mid-rank the excess is 1.0 however large the
real curve is.

The permutation destroys the count→time mapping, **which is the signal**, so the null absorbs the
winner's curse and the curve together. The curse was the right thing to correct — `best` is a
minimum over ~6 noisy per-count medians and reads above 1 on pure noise — but this null
**over-corrects by exactly the amount of real structure present.**

### What the numbers actually say, restated

**The headroom is the `median` column, uncorrected.** On MKT at k=2:

| population | `cur/best` (the headroom) | excess (the rank) |
|---|---|---|
| ALL 53 live cells | **1.1430 — 14.3%** | 0.9996 |
| `<=2` workers (n=9) | **1.3859 — 38.6%** | 1.3454 |
| `3-7` (n=11) | 1.1478 — 14.8% | 0.9243 |
| `>=8` (n=33) | **1.0701 — 7.0%** | 0.9358 |

**So "there is no headroom at k=2 on MKT" is wrong. There is 14.3% pooled, 38.6% in the
small-count group and 7.0% in the large-count group** — all inflated by a winner's curse this
statistic does not isolate, and none of it zero.

### What survives and what does not

**The registered verdicts stand as verdicts.** Amendment 5 registered *"the excess over the
winner's-curse null — median(cur/best) divided by its within-cell permutation null, **the statistic
defined in `nt_k2curve.py`**"*. Elements 1–4 were predictions about **that** statistic and they
scored as reported: element 1 FAILS at 0.9996, element 2 HOLDS at all eight widths, element 3
FAILS, element 4's control FIRES. **A prediction about a named statistic is scoreable even when the
statistic is misinterpreted.**

**Every interpretive gloss is withdrawn.** *"Production's count is already at least as good as an
arbitrary one"* is correct and is a **rank** claim. *"There is none at k=2"*, *"nothing to
recover"*, *"34.5% recoverable headroom"* (amendment 20) and *"1.5–5.9% recoverable at k=8..32"*
(amendment 19) are all misreadings of a rank as a magnitude. Element 2's finding survives in better
form: the `<=2` group's **raw** headroom is 1.3859 against the `>=8` group's 1.0701, so the
direction and the ordering hold on the uncorrected quantity too.

### The closure language is withdrawn, and the reviewer's second argument is the one that kills it

Amendments 18–20 said thread sizing is *"closed as a mechanism for the narrow-k deficit."* **The
defensible claim is "these proposed rules have not demonstrated a useful improvement."** Rejecting
the sqrt model does not establish that the current policy is optimal.

**And my structural argument against thread sizing was wrong.** I argued that an unchanged worker
count cannot explain a width-dependent deficit. It can: **if team-setup overhead is roughly fixed
and useful work scales with k, the overhead's share falls as k rises**, so one fixed, too-wide
count hurts proportionally *most* at narrow k — which is the deficit's shape exactly. The
measurement that would separate the two is a fixed-worker-count sweep at matched work, not a
re-reading of this statistic.

### The correct correction, named but NOT applied

The winner's curse needs **sample splitting**: choose `best` on one half of each count's readings
and evaluate `cur/best` on the other. That removes the selection bias **without** destroying the
count ordering. At `--reps 2` it leaves one reading per half, so it is weak on the existing sweeps
and would need `--reps 4`.

**Not swapped here.** Replacing a pre-registered estimator is Bobby's call. The source now carries
the counterexample, the algebra and the alternative at the read point (`nt_k2curve.py` at
`ce4e3f84ad69f88b`), and the printed prose states what the number is rather than what I wanted it
to be.

### The reusable form

**A null built by permuting the treatment labels tests EXCHANGEABILITY, which is a rank question.
If the quantity of interest is a magnitude relative to an optimum, a label permutation absorbs the
very structure being measured and the statistic reads null at any effect size.** Check a candidate
null by feeding it a noise-free case with a known answer — three numbers were enough here, and the
statistic had already been published on three hosts and eight widths.

---

## Amendment 29 — the all-392 sweep reports: reach is bounded at 23.8%, and where it reaches the HEADROOM is monotone in the worker count

Job 17698209, COMPLETED rc=0 in 1:08:44 on mkt1, pool 32, one object `4c11296ce2a68c3d`,
206,976 rows = 392 keys × 8 widths × 33 counts × 2 readings. Transported from `/scr` through a
single-task `srun` and verified by digest on both ends
(`e25356a743d1b93288486070b9a6afc740d8872c12cd8a1fc2c12dabbb82f373`); the two key lists likewise
(`24e386ce…` for the 392, `57eb6834…` for the 328, with all 328 confirmed inside the 392).

This is the first time the thread-count oracle has run on the population the project's goal is
stated over. Every earlier reading was on `adg_keys.txt` (72 keys), which shares only 20 keys with
the deficit.

### The four registered conditions, read as registered

| # | condition | reading | fires? |
|---|---|---|---|
| 1 | pooled k=2 excess ≤ 1.00 → thread sizing closes | **0.9227** | **YES** |
| 2 | s328 ≥ 0.50 causal; 0.25–0.50 no causal claim; **< 0.25 a BOUND ON REACH** | **0.2378** | **bound, not a pass** |
| 3 | `<=2` excess not exceeding `>=8`'s → element 2 does not transfer | 1.1059 vs **0.8380** | **NO** |
| 4 | bootstrap CI on s328 − s64 containing zero → mechanism does not track the deficit | +0.2378 [+0.1921, +0.2835] | **NO — but see below** |

### Condition 1 fires on an instrument this document has already invalidated

Amendment 28 established that the excess is a **rank** test: the permutation null's support is
`{t_i/min(t)}`, so its median is the middle-rank count's ratio, and timings 1.0/1.2/2.0 with
production at 1.2 and zero noise give an excess of exactly 1.0000 against 20% real headroom. **So
0.9227 ≤ 1.00 does not mean there is nothing to recover.** The headroom is the `median` column:

| half | live cells | median (= headroom) | null | excess |
|---|---|---|---|---|
| deficit 328 | 298 | **1.1355 → 13.6%** | 1.2306 | 0.9227 |
| control 64 | **6** | 1.0606 → 6.1% | 1.0908 | 0.9723 |

13.6% of headroom on the deficit's own population, against 6.1% on the control. Condition 1 is
recorded as firing because it was registered, and it is **not** read as closure.

### Condition 3 does not fire, and the reason is stronger than the condition asked for

Element 2 of amendment 5 — "the excess for cells resolving to ≤2 workers will exceed that for cells
resolving to ≥8" — **transfers to the deficit's own population**, now on the right corpus:

| group | n | median (= headroom) | null | excess |
|---|---|---|---|---|
| `<=2` | 78 | **1.3565 → 35.7%** | 1.2265 | 1.1059 |
| `3-7` | 68 | 1.1374 → 13.7% | 1.2338 | 0.9219 |
| `>=8` | 152 | 1.0292 → **2.9%** | 1.2282 | 0.8380 |

Dominance check PASS at every width, on the median and the null separately.

**The ordering is in the HEADROOM, not only in the rank statistic, and that is what makes it
survive amendment 28.** 35.7% / 13.7% / 2.9% is a monotone gradient in a magnitude-valued quantity
against a null that is nearly constant across the three groups (1.2265 / 1.2338 / 1.2282). The
excess column's ordering could have been an artefact of the rank construction; the median column's
cannot, because it is the measured ratio of production's count to the best reachable one.

### Condition 4 nominally does not fire, and is VOID

The difference is +0.2378 with a CI excluding zero, which as registered would mean the mechanism
tracks the deficit. It cannot be read that way. The instrument's own tripwire fired: live fractions
are **0.9085 in the deficit half against 0.0938 in the control**, differing by 81.5 pp against a
5 pp threshold. Only **6 of 64** control keys are live at k=2, and the control's `<=2` group is
**empty (n=0)**. So `s64 = 0` is structurally forced rather than measured, and the positive
difference is a composition effect in the denominator shaped exactly like the association under
test. **Condition 4 is declared void**, on the same grounds that voided amendment 23: the
comparison half is almost entirely cells where the quantity is undefined.

This is a defect in the registration, not in the run. Amendment 22(c) chose the 64 non-behind keys
as "the clean control — same corpus, same run, same instrument, and OUTSIDE the population being
certified". What it did not check is whether the effect is *defined* there. 94% of it is the
clamped-to-one set, which this document already established is exactly the non-live set. **A
control must be outside the population being certified AND inside the domain where the quantity
exists**; this one satisfies the first and fails the second.

### What the sweep establishes

**Thread sizing is a real but minority mechanism for the k=2 float32 DLMC deficit.** Its reach is
bounded at **≤23.8%** of the 328 keys (condition 2, the registered not-a-pass), against the
deficit's own prevalence of 83.7% — 3.52× less. Where it does reach, the headroom is **35.7%** and
falls monotonically to 2.9% as production's resolved count rises. Both halves of that sentence are
needed: the gradient is clean and the population it covers is a quarter of the problem.

**No rule or gate is proposed, as registered.** A mechanism bounded at a quarter of its target
population does not justify a dispatch threshold, and the measurement that would change the picture
is the one named in amendment 28 — a fixed-worker-count sweep at matched work, separating a fixed
per-call overhead share from thread sizing proper. The `<=2` group's 35.7% is the only part of this
worth that run.

**Still one host.** mkt1 only. The equivalent redwood sweep has not run.

---

## Amendment 30 — correcting amendment 29: the nulls are in HEADROOM units, so only the `<=2` group exceeds its own

> **Amendment 29's "clean monotone headroom gradient" is withdrawn as stated.** The ordering is
> real; the levels are not recoverable headroom, and two of three groups sit BELOW their own null.

A peer asked which functional the nulls carry, because amendment 29 put bare ratios on the other
side of "against" from percentages. Settled at source: `perm_null` (`nt_k2curve.py:209-225`) returns
the median of **the same statistic** over permutations, so the null is in the same units as the
`median` column. `1.2265` is **22.65% in headroom units**, and the comparison amendment 29 implied
is the right one to make — it just does not say what that amendment claimed:

| group | n | raw headroom | its own null | observed − null |
|---|---|---|---|---|
| `<=2` | 78 | **35.65%** | 22.65% | **+13.00 pp** |
| `3-7` | 68 | 13.74% | 23.38% | **−9.64 pp** |
| `>=8` | 152 | 2.92% | 22.82% | **−19.90 pp** |
| ALL | 298 | 13.55% | 23.06% | −9.51 pp |

**Only the `<=2` group exceeds its own null.** The ordering 35.65 > 13.74 > 2.92 is, for two of the
three groups, an ordering in how far *under* the null they sit. "A clean monotone headroom gradient
inside the reach" claimed more than this: the gradient is carried by one group of the three.

**What does survive, and why the constancy of the nulls still matters.** The three nulls span
0.73 pp (22.65 / 23.38 / 22.82). So whatever the null measures, it measures the same amount in each
group, and the ordering in the raw column is therefore **not** produced by the groups having
different null levels. The ordering is a real property of the raw statistic.

**And the recoverable amount is bracketed, not point-estimated.** Amendment 28 established that the
null absorbs real between-configuration differences as well as the winner's curse, so `observed −
null` **under**-states; and the raw column is inflated by the winner's curse, so it **over**-states.
For `<=2` the recoverable headroom lies between **+13.00 pp and +35.65 pp**. For `3-7` and `>=8` the
lower end is negative, so **nothing is established** — in rank terms production there already beats
a typical reachable count.

### Three further corrections to amendment 29, all from the same message

**(a) Condition 2 in the units of the thing it must explain.** `0.2378 / 0.837 = 0.2841`. So the
mechanism accounts for **at most 28.4% of the deficit**, which is the sentence the registration is
about; "a quarter of the population" is a different and weaker one. Both are now stated.

**(b) Condition 4 carries no information beyond condition 2, provably.** `s328 − s64 = 0.2378 − 0 =
0.2378` is **numerically identical to s328**. Because `s64` is structurally zero, condition 4's
statistic *is* condition 2's. Amendment 29 reached VOID from the control's empty `<=2` subgroup; the
identity reaches it without needing the subgroup argument, and is the stronger route because it does
not depend on reading a live fraction.

**(c) The design is complete, and the total could not have shown it.** The peer's arithmetic —
206,976 = 392 × 528, and 528 is not divisible by 7 — was right, against a stale premise of seven
widths; this sweep used **eight** (`2,4,8,12,16,20,24,32`), and the factor of 11 is the 33 worker
counts (0..32). The point stands regardless: **a total is not a completeness check, a per-cell
minimum is.** Measured on the CSVs:

- cells `(key, width, nthreads)`: **103,488 of 103,488 expected**
- readings per cell: **min 2, max 2, zero cells short**
- per `(key, width)`: **min 66, max 66, zero short**

That line belongs in the readout and was missing from amendment 29.
