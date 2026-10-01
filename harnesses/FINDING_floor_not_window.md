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

## Amendment: one of the three confirmations is low-power, and the split needs its interval

Both from the peer session, both accepted.

**The group-level contrast is demoted — it is not a third route.** I offered `b=11-100` (window
2.29 ms, floor 0.2117) against `b=101-1000` (window 5.52 ms, floor 0.2097) as independent weight.
But `batch` is a monotone function of per-call time, so the two groups differ about 2x in median nnz
and **both medians sit on the plateau of the inverted U**, where the decile floors are near their
peak. Near-equal floors are therefore what the nnz axis predicts on its own, window or no window.
The test is consistent with the conclusion and could not have contradicted it, which makes it
low-power rather than confirming. (They are not disjoint populations — 767 of 787 distinct nnz values
in the smaller group also appear in the larger — so the confound is in the distributions, not the
support.)

**The conditioned split now carries its uncertainty.** 4000 bootstrap resamples over cells, seed
20261001, same split sizes:

    p95 dev  short 0.2489   long 0.2688   ratio 0.9260
    95% CI on the ratio      [0.8596, 0.9921]

The interval excludes 1.0, so the 7% is not sampling noise. The better framing is still the bound
rather than the reversal: **the artefact hypothesis requires a ratio above 1, and the entire interval
lies below it**, so the window cannot be inflating the floor in this band at all — its contribution
is bounded above at -0.8%. Against a floor that moves 20x across nnz deciles while the median window
stays flat at 5.36-5.62 ms, a 1.05x cause bounded to a non-positive contribution is the result. The
decile argument is the headline because it yields that quantitative bound; the conditioned split with
its interval is the second informative route; the group contrast is neither.

## One group, three published numbers — each needs its pass and its statistic

The shortfall for "batch 11-100" is circulating as 0.276, 0.433 and 0.196, and all three are right:

| figure | population | statistic |
|---|---|---|
| 0.433 | p1 aa rows, n=4934 | mean |
| 0.196 | p1 aa rows, n=4934 | median |
| 0.276 | cells with a median over 12 passes, n=6006 | median of per-cell values |

The group is badly right-skewed, so mean and median differ by 2.2x on the same rows. The same
applies to its median nnz, which reads 131072, 294912 or 104857 depending on whether the population
is all rows, only cells carrying a complete A/A pair, or one pass. **The conclusion survives every
version** -- all three shortfalls sit far below the `b/(b+1)` flooring bound, which at the group's
median batch of 37 is 0.974 -- but no reader can reproduce "0.276" from the phrase "batch 11-100".
Carry the predicate, not the value.

## Amendment 2: the k margin, and a bad endpoint of mine

**The two k margins differ by statistic, not by pass.** The peer read k=2 0.080 / k=4 0.095 /
k=8 0.233 / k=16 0.368 / k=32 0.612 (a 7.7x gradient); I read 0.169 / 0.375 / 0.429 / 0.444 / 0.537
(3.2x, nearly flat above k=8). Both are p1 aa, batch 11-100, n=4934, and both reproduce exactly:
mine is the **mean**, theirs the **median**. The group is skewed enough that the mean compresses the
gradient at both ends. Monotone either way, so the direction holds, but "how much k carries" differs
and the statistic is the whole of it. Third instance tonight of the same lesson.

**"Up to 18x" was wrong, and wrong in the easy direction.** I obtained it by inverting the 0.054
band mean. The reciprocal of a mean of ratios is neither a maximum nor a median of the ratios. On the
1521 cells at nnz < 1e4 inside batch 11-100 the cold-to-warm ratio is **median 21.5x, p90 28.3x,
max 39.0x**. So the true endpoint is more than twice what I published, and the correction enlarges
the pathology rather than shrinking it -- which is the direction that would have flattered my own
argument had I noticed it first.

## The cost of the pairing, stated because it bit twice tonight

Two sessions independently reconstructed 0.50% for the untimed share and called it confirmed; both
had priced calibration calls at `ms`. Earlier, two sessions independently computed a 21-22% timed
share and called it confirmed; both had omitted `reps`. **Agreement between two parties who share a
derivation is one calculation run twice, not a replication.** What broke both was a check against the
producing source -- `timeit`'s loop structure, `calibrate`'s own batch formula -- not a check against
each other. This is the failure mode a checker arrangement actively encourages, so it is worth naming
as its price rather than discovering twice more.
