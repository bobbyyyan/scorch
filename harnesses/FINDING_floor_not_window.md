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
