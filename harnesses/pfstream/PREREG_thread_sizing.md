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
