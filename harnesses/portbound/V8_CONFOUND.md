# V8's machine body is 16 nonzeros where V6's is 8, so the port discriminator is confounded

Written before reading V8's timing. The gate passed it -- 2.00 mem/nnz, 1.00 bcast/nnz, 0.00
vpor/nnz, exactly the registered delta -- and the gate is wrong to have passed it, because machine
body width is not one of the axes it checks.

    arm   nnz/body   insn   mem   insn/nnz   mem/nnz
    V0        8       35     24     4.375     3.000
    V6        8       37     24     4.625     3.000
    V9        8       30     16     3.750     2.000
    V8       16       36     32     2.250     2.000   <- FOUR-fold unrolled from `p += 4`

V8 was built as "V6 with the index load removed", a delta of one memory operation. The emitted
delta is one memory operation **plus a doubled unroll**: 2.250 insn/nnz against V6's 4.625 is a
reduction of 2.375, of which the removed index load and its address arithmetic account for part and
**loop control de-amortised over sixteen nonzeros instead of eight accounts for the rest.** This is
the same defect the peer found in V4 and V4c, which are four-wide where V4b is eight -- and it has
now appeared in the arm I built to settle their hypothesis, which is the strongest argument that
the gate needs the axis rather than that I need to be more careful.

## THE READING IS ASYMMETRIC, WHICH IS WHY THE ARM IS NOT WORTHLESS

The confound works in the port hypothesis's **favour**: a wider unroll makes V8 faster for a reason
that has nothing to do with load ports. So

- **If V8 beats V6 by about 1%**, then even with a free doubled-unroll bonus on top, removing a
  load-port operation at four chains buys nothing. That **refutes the port model more strongly than
  a clean arm would**, because the confound could only have helped it.
- **If V8 beats V6 by 25-36%**, the result is **inconclusive**: port relief and the unroll bonus
  both predict a large gain and this arm cannot separate them.

So one of the two outcomes is decisive and the other is void, and which one it is will be known
before any mechanism is named. Recorded here so that cannot be chosen after the fact.

## V9 IS CLEAN

V9's body is eight nonzeros, matching V6 and V0. So the dependency-versus-memory-parallelism
contrast -- does V6's gain survive when the random column-indexed B load is gone -- is read on
matched unroll widths and needs no qualification beyond the 128 KB footprint already recorded.

## THE GATE FIX

`asmcheck.py` checks mem/nnz, bcast/nnz and vpor/nnz per arm against a registered expectation. It
does not check the machine body width, so a variant whose unroll the compiler changed passes as
long as its per-nonzero counts are right -- and per-nonzero counts are exactly the quantities that
are INVARIANT to unroll width, which is why this axis is invisible to the current check by
construction.

The fix is to register the body width alongside the counts and refuse a mismatch, with an explicit
declaration for the arms that legitimately differ. Its own principle already demands this: "a
variant whose assembly does not match its intended delta is a refusal, not a data point." A doubled
unroll is not the intended delta.

This is the fifth time the gate has refused or should have refused something, and the first time the
missing axis was found by a variant rather than by reading the disassembly on a hunch.
