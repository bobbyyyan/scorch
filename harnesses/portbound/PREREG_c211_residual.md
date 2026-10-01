# Pre-registration: what the k=8 residual is, written before the chain211 board is read

Registered 2026-10-01, with chain211 pass p1 in flight and only the 60-key screen seen. The
screen's deep/base on our own kernel: k=2 1.0177 (negative control, kernel provably inert),
k=4 1.1002, k=8 1.1859, k=16 1.0494, k=32 1.0808.

## The thing to explain

Register pressure is measured and real -- the deep kernel spills 12 times at NVEC=3 and 23 at
NVEC=4 where the shipped kernel spills zero -- but it cannot explain k=8, which is NVEC=1, holds
four accumulators in a sixteen-register file, spills nothing, and is the WORST width on the
screen. A mechanism that predicts the largest regression at the widest register block does not
predict this, so something else is operating at NVEC=1.

## The prediction

**The k=8 residual is prefetch volume, not register pressure.** Read off the same object, same
innermost loops, matched to nonzeros per body, float32:

    kernel   NVEC   prefetch/nnz
    SHIP      1-4           0.50
    deep      3,4           1.00

The deep kernel issues exactly twice the prefetches per nonzero, at every NVEC, and that ratio is
independent of the spill. It is a per-nonzero cost, so unlike the fold it does not amortise on
long rows, and unlike the spill it is present at NVEC=1. chain197 already established that
prefetch volume is a first-order term in this kernel -- deleting the register-block prefetch won
2.9-14.2% while harming 115 matrices at width 32 -- so doubling it is the right order of
magnitude for 18.6%.

## What that predicts on the board, stated so it can fail

1. **Sign and conditioning.** At k=8, `deep/base` is WORSE where the extra prefetches are least
   useful, which is where B is already cache-resident: an unneeded `prefetcht2` still consumes a
   load-port slot and a fill buffer, and on a resident B the kernel has no memory stall to hide it
   in. So `deep/base` at k=8 correlates NEGATIVELY with B footprint (`J * k * 4` bytes) -- small B
   worse, large B closer to 1. If the correlation is positive, or absent, this mechanism is wrong.

2. **An additional penalty at NVEC >= 3, only after conditioning.** The aggregate does not show
   the spill (k=32 reads 1.0808 against k=8's 1.1859), so if both mechanisms are real the spill
   must be masked by the footprint term: k=32's B footprint is 4x k=8's at the same J. Prediction:
   once conditioned on footprint, the NVEC>=3 cells carry a penalty ABOVE the k=8 trend. If
   NVEC>=3 sits on the same footprint curve as NVEC=1, the spill is not costing measurable time
   and the 23 stack moves are being absorbed.

3. **A distinguishing alternative I am naming so it is not adopted later as if predicted.** The
   four prefetches may be hitting four DISTINCT B rows (one per chain) against the shipped
   kernel's two, which is address-translation pressure rather than port pressure. That scales with
   the number of distinct rows of B touched, so it tracks `rows of B` and mean degree, NOT
   footprint. If `deep/base` at k=8 tracks J or degree with footprint partialled out, the
   mechanism is TLB and not the port.

## What would refute all three

`deep/base` at k=8 flat across footprint, J and degree -- a constant penalty. That would point at
something paid per call or per row rather than per nonzero, and both mechanisms above would be
wrong.

## Estimator, fixed now

Per-cell `exp(median(log(deep_ours/base_ours)))` over the six passes, per width. Floor is the p95
of `dev(bsb_/base)` on the same cells at the same width -- the cross-object A/A, not the
within-process arm floor, because deep-vs-base crosses a process and an object. Correlations are
Spearman on log footprint, reported with the floor beside them, and no cell inside the floor
counts toward a sign.

---

# Amendment, same day, before the board is read

Two of this file's premises are withdrawn and the analysis plan is replaced. Written after a peer
session checked the dose-response, and before any chain211 CSV has been opened.

## Withdrawn

1. **"The k=8 residual."** At +18.59% k=8 is the LARGEST effect on the board, not a remainder.
   Framing it as a residual presumed the spill accounted for the rest, which it does not.

2. **"Twice the prefetches per nonzero, at every NVEC."** Measured at NVEC=3 and NVEC=4 only.
   NVEC=1 and NVEC=2 deep instantiations are INLINED and carry no symbol, so no disassembly for
   them exists. `NVEC = ceil(k/8)` puts k=4 and k=8 -- where the effect is largest -- at NVEC=1,
   and NVEC=3 at k=17..24, which is not a board width. The prefetch prediction below therefore has
   no disassembly behind it at the widths it is about, and must be read as a hypothesis rather
   than as an extrapolation from a measurement. Getting the NVEC=1 body requires forcing those
   instantiations out of line, which needs a compile, which must wait until the board is off the
   host.

## P0, the collinearity trap, adopted from the peer

`degree == nnz/rows` exactly, so `{log nnz, log rows, log degree}` has rank two. Never fit all
three. Fit `y` on `log(nnz)` and `log(rows)`; `log(degree)`'s slope is the difference of those two
partials and needs no term of its own. An analysis that reports three independent slopes here is
reporting an artefact of a singular design matrix.

## The candidates and the sign each predicts

Estimand per width k: `y_i = log(deep_i / base_i)` over the 3820 keys, paired within pass.

    (A) per-BODY cost (loop control, fold, spill -- once per UNROLL nonzeros)
        both partials zero inside the floor's CI; y near-constant across keys at fixed k
    (B) per-ROW cost (accumulator init + deeper fold -- once per row)
        slope(log rows) > 0, slope(log nnz) < 0, near-equal magnitude; this IS y ~ c/degree,
        so test directly: R^2 of y on 1/degree exceeds R^2 on log(degree)
    (C) footprint or traffic driven
        slope(log rows) > 0, slope(log nnz) ~ 0, and a THRESHOLD in rows*k*4 against a cache
        level rather than a monotone slope

The null -- both partials inside the floor's CI at every width -- is itself a result: it says the
effect is a property of the width and not of the matrix.

## Registered magnitudes (the peer's, adopted verbatim as predictions)

    P1  at k=8, slope(log rows) > 0 and slope(log nnz) < 0: the per-row family is present.
    P2  it is NOT sufficient. Four extra vector ops per row, over DLMC's median 103 nonzeros at
        roughly 2 cycles each, is about 6%, and under real traffic a nonzero costs more so the
        fraction shrinks. Against +18.59% the per-row account falls at least threefold short. So
        either the fitted c exceeds the structural op count by more than 3x, or a
        degree-independent component of 10 points or more survives at median degree.
    P3  therefore both a negative degree slope AND a floor at high degree: y at the 90th-percentile
        degree (307) stays at or above 5%. If it decays to the floor there, P2 is wrong and the
        effect is purely per-row with a mis-sized c.

All three are conditional on the 60-key screen being representative. If the full board moves the
headline off 18.59%, these are re-read against the new number.

## The layout nuisance, corrected

Quoting "~1.8% is object layout" uniformly across widths is wrong in the direction that matters.
The two-objects-same-decisions artefact is 1-2.6% on narrow-k SpMM and LARGEST AT THE NARROWEST
WIDTH, so k=2 estimates it where it is biggest and over-corrects at k=32 -- the width whose rank
the retraction above turns on. The per-width floor from the byte-identical `base`/`bsb_` pair is
the form that works, it is already in the board, and the ordering must be read against it rather
than against a single number carried across widths. Adjacent-width gaps are 3.1, 1.9 and 3.1
points against a nuisance of unknown size in a 1-2.6% band, so the ordering is not safe until
that floor is in hand.
