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
