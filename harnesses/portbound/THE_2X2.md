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

## Three independent confirmations of the latency model, one of which needs no clock

**1. Per-core latency floors.** The model is `cycles/nnz = FMA latency / chains`, with Golden Cove
at 4 and Gracemont at 6:

    cell                      P-core predicted/measured/miss    E-core predicted/measured/miss
    2 chains, 3 loads         2.00  1.9943   −0.28%             3.00  2.9909   −0.30%
    2 chains, 2 loads         2.00  1.9737   −1.32%             3.00  2.9835   −0.55%
    4 chains, 3 loads         1.00  1.5805  +58.05%             1.50  2.8676  +91.17%
    4 chains, 2 loads         1.00  1.0258   +2.58%             1.50  1.5041   +0.28%
    4 chains, 2 loads (V9)    1.00  0.9978   −0.22%             1.50  1.5044   +0.30%

Eight of ten readings within 1.32%, across two FMA latencies and two chain counts. One cell fails,
on both cores, and it is the same cell.

**2. The chain ratio, which is frequency-free.** Doubling the chains must halve the cycles, so the
*ratio* is 2.00 whatever the clock is:

    V1/V8  (2→4 chains at 2 loads)   P 1.9240 (−3.80%)   E 1.9836 (−0.82%)
    V0/V6  (2→4 chains at 3 loads)   P 1.2618 (−36.9%)   E 1.0430 (−47.9%)

This matters because the frequency read is on a 100 MHz grid, so every `cycles/nnz` above carries
±1.75% / ±2.27%. The ratio carries none of it.

**3. The cross-core cycle ratio, which tests the model's structure rather than a constant.** If
latency binds on both cores, the E/P cycle ratio must be 6/4 = 1.50:

    arm  chains loads   E/P cycles   vs 1.50
    V0      2     3       1.4997     −0.02%
    V1      2     2       1.5117     +0.78%
    V6      4     3       1.8144    +20.96%   <- the same failing cell
    V8      4     2       1.4663     −2.25%
    V9      4     2       1.5077     +0.52%

Four of five within 2.25%. This is the strongest of the three, because the predicted value is a
*ratio of two latencies* rather than a number that could be fitted.

## What is NOT explained, stated rather than papered over

Why the (4 chains, 3 loads) cell sits where it does. On the P-core 1.5805 is within 5.4% of
3 loads / 2 slots = 1.50, which is the hypothesis the peer registered and which V8 confirmed at
2 loads (1.0258 against 1.00). On the E-core the same cell reads 2.8676, which is 1.046
loads/cycle, while V8 in the same core achieves 2 / 1.5041 = **1.33** loads/cycle. **A fixed
effective-slot count therefore does not describe the E-core**, and I am not offering a replacement.
The defensible statement is narrower than a mechanism: at three load operations per nonzero the
load supply, whatever limits it, prevents the chains from paying on both cores; at two it does not.

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
