# The A/A floor's inverted U is not a timing-window artefact

A null result, recorded with the evidence that the test could have fired.

## Why it was worth asking

Decomposing a shortfall the peer session found -- `mean(ms x batch)` reading 0.433 of `--target-ms`
in one batch group -- produced a sharp sub-result. `mrgate.py` sets `batch = max(1, int(target/m))`
from the median of five calibration calls, so the coarse `int()` floor alone cannot push
`ms x batch / target` below `b/(b+1)`. Measured per cell over chain207's twelve sealed passes:

| batch group | cells | shortfall | flooring bound | implied `ms/m_cal` | window |
|---|---|---|---|---|---|
| 2-10 | 57 | 0.921 | 0.889 | 0.92-1.04 | flooring suffices |
| **11-100** | **6006** | **0.276** | **0.979** | **0.276-0.282** | 1.66 ms, not 6 |
| 101-1000 | 48715 | 0.918 | 0.997 | 0.918-0.921 | 5.5 ms |
| >1000 | 2522 | 0.938 | 0.999 | 0.938-0.939 | 5.6 ms |

(Median over cells, so not comparable to the peer's mean over rows; the estimand differs.)

So 6006 cells are timed in a ~1.66 ms window rather than the intended 6 ms, because the batch was
sized from a calibration median ~3.5x slower than the batched steady state. Their median nnz is
131072 -- **inside the 1e4-1e6 band where the A/A floor is independently known to peak**. Two
suspicious things coinciding in the same band is a reason to suspect one instrument, not two
phenomena: if a short window inflates the floor, the inverted U is partly our own calibration and the
behind-set criterion inherits it.

## The discriminating test, and the refutation

nnz and window length are confounded, so the test conditions on nnz and looks along the window.
Inside nnz 5e4-5e5 (n=9205 cells), split at that band's own median window:

    window <= 5.43 ms   n=4603   p95 dev 0.2489
    window >  5.43 ms   n=4602   p95 dev 0.2688
    ratio short/long    0.93x

The shorter window reads **tighter**, not looser. Two independent confirmations:

- `b=11-100` (window 2.29 ms) and `b=101-1000` (window 5.52 ms) differ 2.4x in window length and
  have essentially the same floor: **0.2117 against 0.2097**.
- Across all ten nnz deciles the floor moves 20x, from 0.0160 to 0.3264 and back to 0.2319, while
  the median window is flat at **5.36-5.62 ms**. A flat candidate cause cannot produce a 20x effect.

**The floor's inverted U is a property of the size regime, not of the timing window.** The existing
characterisation stands unchanged.

## Why this null is informative rather than a failed measurement

The instrument had the dynamic range to show the effect had it existed: the floor varies 20x across
the deciles being compared, the conditioned split had n=4603 against n=4602, and the window contrast
inside it was 2.4x at the group level. A null from an instrument that could not have detected the
effect says nothing; this one could.

## What is NOT concluded

Why `b=11-100` has a calibration median 3.5x above its batched steady state is **not** established
here. Dispersion was my first guess and it is dead: `b=101-1000` has nearly the same per-cell
dispersion (0.640 against 0.734) and only an 8% gap, against 3.5x. The known isolated-versus-batched
cold-start effect on this harness is the right order of magnitude -- chain203 read 6.52x at isolated
call granularity against 1.14x batched -- but that is a candidate, not a test. Left open rather than
narrated, since three mechanisms proposed tonight had three wrong signs.

The measurement consequence is worth stating anyway: for 10.5% of rows the timed window is a third of
the intended length, so a `min` over four reps there rests on fewer calls than the design assumed.
That is a precision cost in a known band, not a bias, and the A/A floor already prices it.

## Correction: "batch carries it" was wrong, in the same output that printed it

The one-way batch grouping dips at 11-100 and the one-way nnz grouping rises monotonically, so the
two disagree about where the shortfall lives while agreeing in aggregate (both weight back to 0.887).
I printed the reading "the 11-100 column is low in EVERY nnz band it occupies, so batch carries it".
The two-way table in that same output refutes it -- the column runs

    nnz <1e3   1e3-1e4   1e4-1e5   1e5-1e6   1e6+
    0.054      0.055     0.144     0.730     0.938

a 17x span. The column is not uniform, so batch does not carry it alone; it is an **interaction**,
visible in the table and invisible in either margin. The peer's reading that this is a grouping
artefact to leave alone is also not right -- an interaction is a real structure, not an artefact of
how the rows were cut.

Where it actually lives: **small nnz AND narrow k, together.** Inside the 11-100 column the shortfall
rises monotonically with width (k=2 0.169, k=4 0.375, k=8 0.429, k=16 0.444, k=32 0.537), and the
extreme cells are matrices under 1e4 nonzeros whose calibration median was 60-550 us -- absurdly slow
for that size. Those 1521 cells are a cold-start population, not a point on a size trend: calibration
ran up to 18x slower than the batched steady state. Neither of us had grouped by width.

## The untimed-call share was understated 1.75x, by both of us, for the reason the table shows

`calibrate()` sets `batch = max(1, int(target/m_cal))`, so `m_cal = target/batch` up to the floor.
The 8 untimed calls therefore cost `8 x target/batch` each cell, **not** `8 x ms`. Pricing them at
`ms` assumes a calibration call costs what a batched call costs -- which is precisely what the
shortfall disproves. Repriced at the interval midpoint of `target/batch`, over all twelve passes:

    at ms            86.44 s   0.511% of wall
    at target/batch  151.22 s  0.894% of wall      ratio 1.75x, stable at 1.72-1.82x per pass

No conclusion moves -- kernel calls go from 85.7% to **86.05%** of wall and the non-kernel remainder
from 14.3% to **13.95%**, so the occupancy bracket *tightens* slightly to **6.045-9.773 of 24 cores
(25.2-40.7%)**, ceiling 2.46-3.97x. But 0.50% was a published figure the peer and I had agreed on,
and agreement between two sessions is not evidence: we had made the same substitution.
