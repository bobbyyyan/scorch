# The line-reuse lever reopens at four accumulator chains and is worth 35%

P-core cpu0, attested binary `5b0c0afba493d9a4`, thirteen arms on **matched 8-nonzero machine
bodies**, 61 repetitions, placement floor p95 **0.055%**.

| arm | ns/nnz | cyc/nnz | chains | load ops/nnz | vs V0 | vs V6 | memops/2 |
|-----|--------|---------|--------|--------------|-------|-------|----------|
| V0  | 0.349881 | 1.9943 | 2 | 3 | 0.00% | +26.18% | 1.50 |
| V1  | 0.346255 | 1.9737 | 2 | 2 | −1.04% | +24.87% | 1.00 |
| V3  | 0.344922 | 1.9661 | 2 | 2 | −1.42% | +24.39% | 1.00 |
| V6  | 0.277283 | 1.5805 | 4 | 3 | −20.75% | 0.00% | 1.50 |
| V7  | 0.278617 | 1.5881 | 8 | 3 | −20.37% | +0.48% | 1.50 |
| V8  | 0.179970 | 1.0258 | 4 | 2 | **−48.56%** | **−35.10%** | 1.00 |
| V9  | 0.175055 | 0.9978 | 4 | 2 | **−49.97%** | **−36.87%** | 1.00 |

## The registered prediction fired on the port branch

`probe.cpp` recorded before measuring that the port reading predicts V8 beating V6 by 30–36%, the
slack reading by about 1%, and that nothing between 5% and 25% is explicable by either. **Measured:
35.10%.** The slack reading is dead and "time = load operations / 2" holds across three arms at
two different load counts:

    V6  3 load ops   predicted 1.50 cyc/nnz   measured 1.5805   miss +5.37%
    V8  2 load ops   predicted 1.00           measured 1.0258   miss +2.58%
    V9  2 load ops   predicted 1.00           measured 0.9978   miss −0.22%

**Two effective load slots per cycle, not the three load ports Golden Cove has.** No mechanism is
offered for the shortfall. The model is a description that fits, and the session's record on
mechanisms is six wrong out of six.

## Which load is removed does not matter

V8 removes the index load and walks B sequentially. V9 removes the random column-indexed B load
and holds B in registers. Two completely different operations, and the results land **2.73%
apart**. That is the signature of a count bound rather than of any one operation's cost, and it is
the part of this that no single-arm result could have shown.

## The conditional refutation's condition is now measured, and it was the right condition

When the latency diagnosis was recorded, this qualifier went with it:

> the lever is closed **at two accumulator chains** because latency dominates while the ports sit
> half idle, not closed in general — at four chains the P-core reaches 4/4 = 1.00 and 3 load ops /
> 3 load ports is **also** exactly 1.00, so the load port becomes binding precisely when the
> latency bound is lifted.

The measurement:

    removing one load operation at TWO chains:   −1.04% (V1),  −1.42% (V3)
    removing one load operation at FOUR chains: −35.10% (V8), −36.87% (V9)

**Twenty-five times more.** The lever reported CLOSED is closed only at the shipped accumulator
count. The arithmetic that predicted the reopening was right about the fact and wrong about the
slot count — I wrote three ports giving a 1.00 floor; the measurement says two effective slots
giving 1.50 at three loads, which is why V6 stops at 1.5805 rather than reaching 1.00.

## What this is not

**V8 and V9 do not compute the correct answer.** V8 walks B sequentially with no gather; V9 holds
B in registers. They bound what is available if a load were free and they implement nothing. The
implementable form of "one fewer load per nonzero" is a format or blocking change — consecutive
nonzeros sharing a B row, or pre-scaled row pointers that remove the index load's address
arithmetic. That is real engineering. The 35% is the prize, not the result.

Stacked with the chain deepening the bound is large: V8 is −48.56% from V0, so four chains and two
load operations per nonzero would be about **2× today's inner loop**. chain210's k=4 median
`ours/mkl` is 0.9130 and k=4 carries 19 of its 22 behind-MKL cells, so this is a path at the width
that actually carries the deficit. Two things shrink the realised figure: the inner loop is not the
whole kernel, and the degree sweep says rows below degree 24–32 regress.

## V7's flatness is now explained rather than observed

Eight chains buy +0.48% over four, consistent with every earlier reading. With the load-slot model
confirmed, that follows: at three load operations the bound is 1.50 cyc/nnz for any chain count at
or above four, so V7 has nothing to gain until the load count falls. The prediction to test is
that V7-with-two-loads would also sit at 1.00 and gain nothing over V8.

## Two published numbers moved when the bodies were matched

V4c was +15.253% and is now **+13.266%**; V4 was +12.403% and is now **+7.940%**. Those are
exactly the two arms whose machine bodies were 4-wide and are now 8, so the earlier figures
carried a halved per-nonzero loop control — the peer's diagnosis, confirmed by the correction
moving only those two arms. The clean contrast is now V4c − V4b = +13.195% for exactly +1.00
insn/nnz and +1.00 mem/nnz: one instruction, one memory operation.

## Scope

Frequency resolution is unchanged and still one-sided: 5.700 GHz is read on a 100 MHz grid, so
every `cyc/nnz` above is an upper bound and a lower true clock shrinks all of them — improving
V9's fit and worsening V6's. Single row of 4096 nonzeros, so the long-row limit; the degree sweep
governs production row lengths. B is 128 KB and L2-resident, so the load-slot reading is for that
band. **The E-core scope restriction written here was wrong and is superseded by `THE_2X2.md`.** I
wrote that the model is "P-core only" from V6's absolute level alone, before the E-core's V8 had
landed — and V6's level cannot tell you what V8 will do. The condition replicates on the E-core
*larger*: V8 − V6 is **−47.55%** there against the P-core's −35.10%. What is P-core-only is the
`loads/2` arithmetic, which overshoots the E-core by +50% at two loads and +91% at three; no
integer slot count fits both E-core points. Reading a scope restriction off one cell's level was
the error.
