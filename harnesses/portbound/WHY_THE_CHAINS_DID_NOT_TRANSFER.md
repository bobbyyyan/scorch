# The four-chain kernel loses on real matrices, and the mechanism I published for it was wrong

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

## The mechanism I published, and why it is not the board's story

Retracted as an explanation on 2026-10-01, within hours of writing it, after a peer session
pointed out that its dose-response runs backwards. Keeping the measurement and the retraction
together, because the measurement is sound and only the inference was wrong.

What is measured, both kernels in one object, `-O3 -march=native -funroll-loops`, g++ 11,
innermost loop of each out-of-line float32 instantiation, matched to nonzeros per body:

    kernel   NVEC  UNROLL  nnz/body  insn  insn/nnz  stack vmov  prefetch/nnz
    SHIP        1       -       4.0    39      9.75           0          0.50
    SHIP        2       -       4.0    46     11.50           0          0.50
    SHIP        3       -       4.0    46     11.50           0          0.50
    SHIP        4       -       4.0    42     10.50           0          0.50
    deep        3       4       4.0    72     18.00          12          1.00
    deep        4       4       4.0    87     21.75          23          1.00

At NVEC=4 the deep kernel's most frequent instruction is a stack `vmovaps`, 23 of them,
outnumbering its own 14 FMAs. The shipped kernel spills nowhere at any NVEC. All of that stands.

**It explains none of the board, for two reasons I should have checked before publishing it.**

First, the dose-response is backwards. `NVEC = ceil(k/8)`, so the board's five widths map to NVEC
1, 1, 1, 2, 4 for k = 2, 4, 8, 16, 32. The spill therefore exists at exactly ONE board width:

    k     NVEC  deep/base   spilled vmovs
    2      1     1.0177     -- does not fire
    4      1     1.1002     not measured
    8      1     1.1859     not measured
    16     2     1.0494     not measured
    32     4     1.0808     23

The only spilling width is third of the four that fire, behind two that spill nothing. The LARGEST
regression, k=8 at +18.6%, sits at NVEC=1 with four accumulators in a sixteen-register file. A
mechanism predicting the worst outcome at the widest register block predicts the reverse of the
observed ordering.

Second, and this is the worse error: I characterised the mechanism where it could not be seen.
Only the NVEC=3 and NVEC=4 deep instantiations survive as out-of-line symbols -- NVEC=1 and NVEC=2
are inlined and carry no symbol, so there is NO disassembly for them. NVEC=3 is k=17..24 and is
not a board width at all. So every number in the `deep` rows above describes configurations the
board either never runs or runs at a single width, and both the spill count and the doubled
prefetch volume were extrapolated to NVEC=1 on no evidence. That is the failure mode of verifying
the smallest-effect cell: the error is definitionally invisible where I looked.

I also framed k=8 as "the residual". At +18.6% it is the largest effect on the board. What is owed
a mechanism is the main effect, not a remainder of one.

## The register budget was already in the source

The rule I offered as the lesson -- `UNROLL * NVEC <= R` with headroom -- is partly implemented
already. The switch that instantiates these kernels carries the comment "NVEC * UNROLL
accumulators must stay resident, so the deeper streams are only instantiated where the vector
count is small", and it admits UNROLL=8 only at NVEC 1 and 2 while NVEC 3 and 4 get UNROLL=4 only.
The 16-accumulator case at NVEC=4, UNROLL=4 is precisely what that rule lets through, and the 23
spills are its price. Tightening it is defensible on the disassembly alone, without any timing.
It is still not an explanation of the board.

## What survives

- The accumulator-latency reading of the SHIPPED kernel. Unaffected: it is a measurement of two
  chains against one on a loop with one output vector.
- The 2x2 and the load-reduction prize. Unaffected for the same reason, and the load-reduction
  half of it was never register-limited.
- The measured fact that four chains beat two by 25-37% per nonzero when registers are free. That
  is a real ceiling, and it is now bounded by a register budget rather than by a row length.

## What does not survive

- The claim that a register spill is why the chain lever does not transfer. The spill is real,
  confined to NVEC=4, and cannot produce the observed ordering.
- `SCORCH_NARROWK_DEEP_MINDEG`'s framing as a ROW-LENGTH question, but for a reason that no longer
  depends on the spill: every candidate still standing is paid per nonzero or per row, and a
  row-length band addresses neither.
- The claim that the probe's bracket bounds production's crossing from below. It bounds the
  crossing of a loop with one output vector.
- My "third reason" that production's machine body is narrower than the probe's. It is -- but the
  SHIPPED kernel's body is also 4.0 nonzeros, so body width is no difference at all between the
  two production kernels, which is the comparison the board measures.

## What is still unexplained, stated plainly

Why the deep kernel loses 10-19% at NVEC=1, where it spills nothing and its per-nonzero loop
control is identical to the shipped kernel's (both bodies 4.0 nonzeros). The structural
differences left at NVEC=1 are per-ROW: four accumulator zeroes against two, and a depth-3 left
fold against a depth-1 add. The peer's arithmetic puts that family at roughly 6% at DLMC's median
degree of 103 and falling as a nonzero gets more expensive under real traffic -- at least
threefold short of 18.6%. So either the per-row cost is far larger than its op count suggests, or
a degree-independent component survives.

Registered predictions that discriminate these, including the collinearity trap that would void
the analysis, are in PREREG_c211_residual.md. The 3820-key board settles it.
