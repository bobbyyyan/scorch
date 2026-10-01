# RETRACTION: the accumulator-latency model is confirmed, not refuted — it fails at exactly one cell of a 2×2, and the whole gain is in the interaction

I committed a finding titled "V6 REFUTES the accumulator-latency diagnosis on both core types".
**That is wrong and is retracted here.** With the load count added as a second factor, the latency
model is confirmed at both chain counts on both microarchitectures, and the cell where it failed is
the one cell where a second constraint intervenes.

## The 2×2, both cores, attested binary `5b0c0afba493d9a4`, matched 8-nonzero bodies, 61 reps

P-core cpu0, placement floor 0.055%:

|  | 3 loads/nnz | 2 loads/nnz | gain from one fewer load |
|---|---|---|---|
| **2 chains** | 0.349881 (V0) | 0.346255 (V1) | **−1.04%** |
| **4 chains** | 0.277283 (V6) | 0.179970 (V8) | **−35.10%** |
| gain from doubling chains | **−20.75%** | **−48.02%** | |

E-core cpu16, placement floor 0.010%:

|  | 3 loads/nnz | 2 loads/nnz | gain from one fewer load |
|---|---|---|---|
| **2 chains** | 0.679757 (V0) | 0.678071 (V1) | **−0.25%** |
| **4 chains** | 0.651731 (V6) | 0.341847 (V8) | **−47.55%** |
| gain from doubling chains | **−4.12%** | **−49.59%** | |

## The interaction is the entire effect, and on the E-core it is nearly all of it

    P-core: chains alone −20.75%, one fewer load alone −1.04%, additive −21.79%,
            BOTH TOGETHER −48.56%   ->  interaction −26.78 points
    E-core: chains alone  −4.12%, one fewer load alone −0.25%, additive  −4.37%,
            BOTH TOGETHER −49.71%   ->  interaction −45.34 points

On the E-core **neither lever alone does anything** — 4.1% and 0.25% — and together they halve the
time. That is as clean a demonstration as this probe can produce that the two changes have to be
made simultaneously or not at all.

## THIS EXPLAINS THE PRIOR NULLS IN THE CODEBASE, which are the two margins of this table

`spmm.h` records a measured null for stream depth over 118 sub-parity matrices: S = 2/4/8 reading
1.001 / 0.994 / 1.001 at k=4. That is the top row read rightward — more chains, load count
unchanged. And this probe's own earlier result closed the line-reuse lever at −1.0% to −1.4% —
that is the left column read downward, one fewer load, chain count unchanged. **Both margins are
nulls and the interaction is −45 points.** An effect that is invisible in both margins of a 2×2 is
exactly the shape that makes a sequence of single-factor experiments conclude there is nothing
there.

## ONE body of evidence, not three — a correction to how I presented it

I wrote "three independent confirmations, one of which needs no clock". **That count is wrong.**
There is one dataset — ten cells of measured `cycles/nnz` against `FMA latency / chains` — and what
I called confirmations (2) and (3) are exact algebraic rearrangements of it. The peer showed the
identities and they hold to machine precision here:

    (3), the cross-core ratio:  (E_cyc/P_cyc)/1.50 - 1  ==  (1+miss_E)/(1+miss_P) - 1
         because pred_E/pred_P = (6/ch)/(4/ch) = 6/4, so the chain count cancels.
         V0 -0.01844% both ways, V1 +0.77799%, V6 +20.95719%, V8 -2.24967%, V9 +0.51543%
         largest difference 2.2e-16.

    (2), the chain ratio:  (meas_2ch/meas_4ch)/2.00 - 1  ==  (1+miss_2ch)/(1+miss_4ch) - 1
         because pred_2ch/pred_4ch = (lat/2)/(lat/4) = 2, so the LATENCY cancels.
         P V1/V8 -3.80202% both ways, P V0/V6 -36.90904%, E V1/V8 -0.82244%,
         E V0/V6 -47.84988%; largest difference 1.1e-16.

So each projection cancels a nuisance factor — (2) the clock down a column, (3) a common-mode clock
error across cores — and neither adds a datum. Neither can pass where the table fails or fail where
it passes. They look tighter because they drop a nuisance factor, **not a degree of freedom**.

**And my ranking was backwards.** I called (3) "strongest of the three, because the predicted value
is a ratio of two latencies and cannot be fitted". Unfitted it is, but it is the projection *most*
exposed to the clock, because its measured value needs the frequency **ratio**: on the ±100 MHz grid
I adopted, 4.4/5.7 carries ±2.87% in quadrature (±4.03% worst case) against a **3.03% spread** across
the four cells I cited as agreement. On a half-step ±50 MHz reading it is ±1.44% and the agreement
resolves — so whether (3) resolves at all is decided by a grid convention, and I adopted the one on
which it does not. Worse for its use as a test: (3) passes identically whenever both cores miss the
latency floor by the *same* factor, i.e. when the model is wrong in the same way on both. V0's
−0.018% is exactly P's −0.28% and E's −0.30% being nearly equal.

**What the evidence actually is, stated once:** ten cells, two FMA latencies and two chain counts,
both taken from the datasheet and the source with nothing fitted, **eight within 1.32%**. That is
strong parameter-free support and it does not need to be counted three times. The identity remains
useful as a *consistency constraint* — it is how I caught a scale inversion, since V0 agreeing to
−0.30% on its own-core floor made a 68% cross-core miss impossible — and that use is legitimate
precisely because it is an identity.

## What is NOT explained — and it is E-core-only, which is narrower than I said

I put the P-core in this paragraph and should not have. On the P-core, the maximum of the two
parameter-free bounds describes **every** cell:

    arm   max(latency/chains, loads/2)   measured   miss
    V0    max(2.00, 1.50) = 2.00         1.9943    −0.28%
    V1    max(2.00, 1.00) = 2.00         1.9737    −1.32%
    V6    max(1.00, 1.50) = 1.50         1.5805    +5.37%
    V8    max(1.00, 1.00) = 1.00         1.0258    +2.58%
    V9    max(1.00, 1.00) = 1.00         0.9978    −0.22%

All five within 5.4%, including the cell I called unexplained. So the P-core needs no residual: it
is `max(accumulator latency, load supply at two per cycle)` throughout.

**The residual is the E-core's, and it is that its load throughput is not constant.** V8 achieves
2 loads / 1.5041 cycles = **1.330 loads/cycle**; at that rate three loads would cost 2.2562
cycles/nnz, and V6 measures **2.8676**, a **+27.1%** miss. The same core sustains **1.046**
loads/cycle at three loads per nonzero. A single effective-slot count therefore cannot describe the
E-core at both load counts, and I am not offering a mechanism — only narrowing where one is still
owed. Which is: the E-core, at three load operations per nonzero, and nowhere else in this table.

## The V7-with-two-loads arm is not worth building

I listed it as the testable next prediction. It cannot discriminate. At eight chains the latency
floor is 4/8 = 0.50 on the P-core and 6/8 = 0.75 on the E-core, while the load bound at two loads
is 2/2 = 1.00 — and since the binding bound is the larger, `max` gives 1.00 under both readings.
"Gains nothing over V8" is what either predicts, so the arm would cost a build and settle nothing.
The peer caught this before I spent it.

## What this is not

**V8 and V9 compute the wrong answer.** V8 walks B sequentially with no gather; V9 holds B in
registers. They establish that *if* one load per nonzero were removed the chain deepening would
deliver its full factor — they do not remove it. The implementable form is a format or blocking
change: consecutive nonzeros sharing a B row, or pre-scaled row pointers that eliminate the index
load's address arithmetic. That is the work, and the 2× is what it would be worth.

## Scope

Single row of 4096 nonzeros, so the long-row limit; the degree sweep governs production row
lengths and puts the P-core break-even for chains-alone at degree 24–32. B is 8192 × 4 floats =
128 KB, L2-resident. Frequency read on a 100 MHz grid, one-sided downward, which is why
confirmations 2 and 3 are stated alongside 1. float32, redwood only — no second host for any of it.
