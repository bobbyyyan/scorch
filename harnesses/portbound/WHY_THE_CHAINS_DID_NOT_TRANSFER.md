# Four accumulator chains cost 4xNVEC registers, and the probe only ever paid for one NVEC

## The claim this replaces

`harnesses/portbound` measured four accumulator chains against two on a single row at -20.7% to
-37.6%, parameter-free on two microarchitectures, with `cycles/nnz = FMA latency / chains` holding
to 1.32% at eight of ten cells. I committed two reasons that measurement might not transfer to the
production kernel -- a fold one level deeper, and a machine body half as wide -- and argued both
pushed the same direction, so the probe's crossing bracket was a lower bound on production's.

Both of those are real and both are small. Neither is why the lever does not transfer. And one of
them is now measured to be no difference at all.

## What the production pair actually does

A 60-key screen of the full DLMC board, three objects, redwood, float32, pool 24, with
`SCORCH_NARROWK_DEEP_UNROLL=4` against the shipped build:

    k     base ours/mkl   deep ours/mkl   deep/base (our kernel)
    2        0.8802          0.8805              1.0177
    4        0.9420          1.0293              1.1002
    8        0.8835          1.0362              1.1859
    16       0.7686          0.8105              1.0494
    32       0.6782          0.7467              1.0808

The deep arm is slower at every width, and it turns k=4 and k=8 from leading MKL into trailing it.
k=2 is a negative control carried inside the same object -- the deep kernel provably does not fire
there, verified by the two builds' outputs being BIT-IDENTICAL at k=1, 2, 3 and 33 and differing
at 4, 8, 16 and 32 -- so roughly 1.8% of each figure is object-layout difference and the rest is
the kernel. This is a screen, not a result; the full 3820-key run is chain211.

## The mechanism, read off the object

Both kernels in one object, `-O3 -march=native -funroll-loops`, g++ 11, innermost loop of each
out-of-line float32 instantiation, matched to nonzeros per body:

    kernel   NVEC  UNROLL  nnz/body  insn  insn/nnz  stack vmov
    SHIP        1       -       4.0    39      9.75           0
    SHIP        2       -       4.0    46     11.50           0
    SHIP        3       -       4.0    46     11.50           0
    SHIP        4       -       4.0    42     10.50           0
    deep        3       4       4.0    72     18.00          12
    deep        4       4       4.0    87     21.75          23

The deep kernel's single most frequent instruction at NVEC=4 is `vmovaps`, 23 of them, every one
stack-referencing -- more than the 14 FMAs in the same loop. The shipped kernel spills zero times
at every NVEC. Instructions per nonzero roughly double.

The reason is a register count and nothing softer. The register block holds NVEC output vectors;
UNROLL accumulator chains multiply that, so the kernel needs `UNROLL * NVEC` vector accumulators
live across the loop, plus registers for the B loads, the A splats and the tail mask. AVX2 has 16
ymm registers. At UNROLL=4 that is:

    NVEC=1   4 accumulators    fits
    NVEC=2   8 accumulators    fits
    NVEC=3  12 accumulators    tight -- 12 stack moves appear
    NVEC=4  16 accumulators    the entire register file, nothing left -- 23 stack moves

## Why the probe could not have seen it, structurally

The probe accumulates into `__m128`/`__m256` scalars -- ONE output vector. So four chains cost it
four registers of sixteen and it has twelve to spare. The production kernel multiplies the chain
count by the register-block width, and the register-block width is the whole point of the kernel.
The probe measured the chain lever at the one value of NVEC where the lever is free.

That is not a defect in the probe's execution. It is a defect in the mapping from the probe to the
kernel, and it is invisible in every quantity the probe reports: the per-nonzero gain, the latency
model, the 2x2 interaction, the crossing degree. All of those are correct about the loop the probe
ran. None of them is a statement about a loop that also has to hold NVEC output vectors live.

## What survives

- The accumulator-latency reading of the SHIPPED kernel. Unaffected: it is a measurement of two
  chains against one on a loop with one output vector.
- The 2x2 and the load-reduction prize. Unaffected for the same reason, and the load-reduction
  half of it was never register-limited.
- The measured fact that four chains beat two by 25-37% per nonzero when registers are free. That
  is a real ceiling, and it is now bounded by a register budget rather than by a row length.

## What does not survive

- `SCORCH_NARROWK_DEEP_MINDEG`'s framing as a ROW-LENGTH question. A row-length minimum cannot fix
  a register spill: the spill is per body, not per row, so it is paid on long rows too -- which is
  exactly where the probe said the lever pays best. The threshold was looking for the crossing on
  the wrong axis.
- The claim that the probe's bracket bounds production's crossing from below. It bounds the
  crossing of a loop with one output vector. For NVEC >= 3 there may be no crossing at all.

## The experiment this makes, and why it is not a tuned threshold

The budget is an architectural constant: `UNROLL * NVEC <= R` for a vector file of R registers,
with headroom for operands. That predicts the chain lever is available only at narrow register
blocks, and that UNROLL must fall as NVEC rises -- UNROLL=4 at NVEC=1, 2 at NVEC=2, 1 (i.e. the
shipped kernel) at NVEC=4 on AVX2, with AVX-512's 32 registers moving every entry one notch.

That is a rule derived from the register file, checkable against the disassembly before it is
timed, and it predicts the sign of its own failures. It is the opposite of a per-shape threshold.
The open question it does NOT answer is k=8, which is NVEC=1, spills nothing, and still regressed
18.6% on the screen -- so register pressure is a mechanism here, not the only one, and the
chain211 board is what separates them.
