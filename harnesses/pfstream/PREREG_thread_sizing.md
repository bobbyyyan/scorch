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
