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
