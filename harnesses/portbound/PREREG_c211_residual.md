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

**Why I expect P2 to be the one that breaks, named before the data so it is a prediction and not
an excuse.** P2's shortfall rests on "roughly 2 cycles each" for a nonzero, and that figure comes
from an isolated single-row probe with B resident in cache. That premise is the same one that made
the probe's whole chain measurement non-transferable: the probe was latency-bound because its
working set was small, and the board is neither isolated nor resident. If the per-nonzero cost on
the board is materially higher than 2 cycles, the four extra per-row vector ops are a SMALLER
fraction of the row and P2 gets stronger; if it is lower, P2 weakens. The direction is therefore
not obvious from the premise alone, and that is the point -- a denominator imported from a
measurement whose premise failed once should not be trusted to fix its magnitude the second time.

The same shape explains the pass timing and is worth recording as a calibration: the 60-key screen
implied 20.5 minutes per object-pass and the measured rate is 22.3. That is the screen's
extrapolation failing by 9%, not the run misbehaving, and it is the mildest instance of exactly
the inference P1-P3 depend on. Any number in this file derived by scaling the screen to 3820 keys
inherits that failure mode.

## The layout nuisance, corrected

Quoting "~1.8% is object layout" uniformly across widths is wrong in the direction that matters.
The two-objects-same-decisions artefact is 1-2.6% on narrow-k SpMM and LARGEST AT THE NARROWEST
WIDTH, so k=2 estimates it where it is biggest and over-corrects at k=32 -- the width whose rank
the retraction above turns on. The per-width floor from the byte-identical `base`/`bsb_` pair is
the form that works, it is already in the board, and the ordering must be read against it rather
than against a single number carried across widths. Adjacent-width gaps are 3.1, 1.9 and 3.1
points against a nuisance of unknown size in a 1-2.6% band, so the ordering is not safe until
that floor is in hand.

---

# Amendment 2, still before any chain211 CSV is read

The design in Amendment 1 cannot separate two of its own candidates. Caught by the peer session;
every figure below independently reproduced here from this machine's DLMC copy before adoption.

## The confound

The board has no `cols` column -- its header is `build,slice,key,k,rows,nnz,deg,arm,ms,batch` --
so B's footprint, `4 * k * J` for a dense B, is not computable from the board at all. Joining J in
from the collection then exposes the problem: at a FIXED width, footprint is J up to a constant,
and `log J` correlates +0.4104 with `log nnz` while being near-orthogonal to `log rows` (+0.0077).
So in the registered fit of `y` on `{log nnz, log rows}` the footprint signal arrives almost
entirely through the `log(nnz)` partial -- which is the same coefficient P1 predicts to be
NEGATIVE, and the prefetch/footprint story predicts to be negative too.

Two hypotheses, one coefficient, the same sign. As registered, the fit cannot tell them apart and
whichever one I held while reading the number would have looked confirmed. This is P0 again,
inside the amendment that adopted P0.

Nor is it rescued by redefining footprint as the columns actually touched. In a 300-key sample
stratified by `nnz/cols` decile the median `distinct/J` is 1.0000 in deciles 2 through 10, so the
two definitions coincide on roughly 90% of the board; they differ only in the bottom decile, where
25 of 30 sampled keys touch fewer columns than J at a median ratio of 0.9688. (The peer's sample;
not re-run here, and it is a sample rather than a census.)

## The repair, adopted

Fit `y ~ log(nnz) + log(rows) + log(J)`. Demonstrated to be legitimate rather than assumed, by
`harnesses/an_dlmc_dims.py`:

    regressor             residual SS   total SS     VIF     R^2  SE infl
    log J                      1674.0     2136.8   1.276  0.2166    1.130   usable
    log degree  ~0 (< 1e-18 x total SS)     7877.2     inf  1.0000      inf   POSITIVE CONTROL

    rank{log nnz, log rows, log J, 1} = 4      rank{..., log degree, 1} = 3

Reported as VIF and not as residual SS, because residual SS is not scale-free: 1674.0 is in
squared log-J units and would read the same whether log J had a wide spread or a narrow one, and
"0 means collinear" never says how far from 0 is far enough. VIF = total/residual = 1.276 does
say. It is the squared inflation of that coefficient's OWN standard error -- 13% -- and it leaves
the other coefficients' identifiability untouched. Residual SD of log J is 0.6622 against a total
SD of 0.7480, the same statement in log units. A VIF of 1.276 is not a design under strain.

The degree row is the POSITIVE CONTROL for the check rather than a second finding: a collinearity
test that cannot return zero on a genuinely collinear regressor says nothing about the one that
returns 1674. It is reported against a threshold and not as a value, because the residual there is
numerical noise whose magnitude is not reproducible across BLAS paths -- two machines read 7e-25
and 3e-24 on identical data -- so quoting it as a figure would imply precision it does not have.

Both directions of the check are planted and fire: log J made an exact function of nnz and rows
reads collinear and refuses; log degree replaced by noise reads non-collinear and refuses; and the
real log degree = log nnz - log rows reads collinear as it must.

Then the hypotheses separate on a coefficient neither shares:

    per-ROW family      slope(log J) = 0 within the floor's CI
    footprint family    slope(log J) < 0
    flat penalty        all three slopes at zero

**Primary analysis is the stratified version**, which needs no partialling: fit inside `rows==512`
alone -- 2562 keys, 67.1% of the board, rows constant by construction -- and read the J slope
directly. The unstratified three-regressor fit becomes the robustness check rather than the
headline.

## The shape that must be reported beside any J slope

J takes 11 distinct values and 66.2% of keys sit at J=512:

    64:105  128:84  147:17  256:210  512:2527  576:63  1024:147  1152:84  2048:394  2304:126  4608:63

Six levels carry 100 keys or more. So a footprint slope has nothing like n=3820 of leverage behind
it: it is a six-point contrast with two thirds of the mass at one point, and the levels track
DLMC's layer shapes, so the slope is partly a contrast across layer families rather than a
response to footprint. 52.9% of the board is square.

Inside the `rows==512` stratum the J levels are `128:84 256:21 512:2000 1024:21 2048:373 4608:63`
-- effectively a 512-versus-2048 contrast with two small wings. A slope estimated there is a real
slope with a small number of levels, and that sentence goes next to the number.

## Two-copy provenance, which came free

The join was derived independently on this machine from line 1 of each `.smtx` and compared
against the peer's independently obtained copy: **3820 keys x 3 integers = 11460 compared, zero
disagreements, no key present in only one copy.** All 3820 chain210 keys resolve to a local file,
zero unresolved, and rows and nnz agree with the board on every one. Two copies agreeing on 11460
integers is something a substituted or regenerated collection cannot do, which a consistency check
on a single copy cannot establish. The reference copy's digest is
`3f0887e553b61e71627246d676035b3577382f314103df11fb36527dd011b028`.

`an_dlmc_dims.py` REFUSES on any unresolved key rather than fitting on a partial join, since a
silently dropped subset is invisible in every coefficient it would produce.

---

# Amendment 3: the footprint hypothesis is a THRESHOLD, not a slope. Still before any CSV.

The peer supplied the residency table that makes the footprint family testable in a specific form.
Reproduced here independently from the dims join, and the cache sizes read off the host rather
than assumed by either of us.

## redwood's actual cache geometry, from sysfs

    P-core (cpu0)   L1d   48K    L2  2048K     L3 36864K shared
    E-core (cpu16)  L1d   32K    L2  4096K     L3 36864K shared

## B's footprint over the board, `J * k * 4` bytes

      k   min B / median B / max B    <=48K(P L1d)  <=32K(E L1d)  <=2048K(P L2)
      2      512 /    4096 /   36864     100.0%        98.4%          100.0%
      4     1024 /    8192 /   73728      98.4%        95.1%          100.0%
      8     2048 /   16384 /  147456      84.7%        82.5%          100.0%
     16     4096 /   32768 /  294912      78.7%        77.0%          100.0%
     32     8192 /   65536 /  589824      10.9%        10.9%          100.0%

**The only cache boundary this board crosses is L1.** The largest B anywhere is 576 KB against a
2 MB P-core L2, so every key at every width is L2-resident with room to spare. "A resident B
offers no stall to hide the prefetch in" is therefore a statement about L1 and nothing else, and
because the pool spans both core types both the 48 KB and 32 KB boundaries are live.

## Consequence: register the threshold, not the slope

At k=8 the 48 KB boundary falls between J=1152 (36 KB) and J=2048 (64 KB), splitting the board
**3237 resident / 583 non-resident (15.3%)**, the non-resident side being J=2048 (394), 2304 (126)
and 4608 (63). So a slope on log J at k=8 spends most of its leverage on variation from 2 KB to
36 KB -- all inside L1, where the stated mechanism does not vary at all -- while the contrast the
hypothesis actually predicts is the 15.3% on the far side.

**Registered form:** an indicator at `J*k >= 12288` (48 KB) and, reported alongside because the
pool is hybrid, at `J*k >= 8192` (32 KB). The 32 KB boundary splits 3153 / 667 and moves 84 keys
relative to the 48 KB one, all of them the J=1152 level. The monotone log J slope is retained only
as a secondary reading, and if the two disagree the threshold is the registered one.

## A second, independent reason the k=2 layout correction does not transfer

Already withdrawn on the grounds that the two-objects artefact is largest at the narrowest width.
The residency table adds a structural reason that does not depend on that: at k=2 every key on the
board is L1-resident, at k=32 only 10.9% are. The width used as the negative control and the width
whose rank the retraction turned on sit at opposite ends of the only boundary the board crosses.

## Caution carried with the table, the peer's and correct

"Fits L1" is an upper bound on residency, not residency. A's values and column indices stream
through the same L1 and C's output row is live there too, so the effective capacity available to B
is less than 48 KB and the true crossing is at a smaller J than the table says. The direction is
known, so the resident fractions above are OPTIMISTIC -- they overstate how much of the board sits
in the regime where the mechanism is silent.

## Two corrections to the figures as sent

- The L2 was given as 1.25 MB; it is 2 MB on a P-core and 4 MB on an E-core. The conclusion is
  unchanged and in fact stronger: the board clears L2 by a wider margin than stated.
- The 32 KB split was given as 3151 / 669, moving 86 keys. It is **3153 / 667, moving 84** -- the
  J=1152 level holds exactly 84 keys, and `J*8*4 <= 32768` is `J <= 1024`, so the moved set is
  precisely that level.
