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
