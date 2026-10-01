# The deep kernel's regression is a guarded prefetch issued once per CHAIN, not once per stream

Status: a disassembly result with the axis isolated, on a host that is not the one the board is
running on. **It predicts the timing but is not itself a timing.** The prediction is registered
below before any arm is timed.

## What the earlier spill reading got wrong, and what replaced it

The register spill published this morning was retracted: `NVEC = ceil(k/8)` puts the board's widths
at NVEC 1, 1, 1, 2, 4, so the spill exists only at k=32 (third-worst of the four widths that fire)
and the two worst widths sit at NVEC=1 with four accumulators in a sixteen-register file. Worse,
there was no disassembly at NVEC=1 at all -- those instantiations are inlined on redwood and carry
no symbol.

This closes that gap. Built on MKT with the ISA pinned to `-mavx2 -mfma` (16 ymm, matching
redwood's register file, NOT `-march=native` which would give Zen4 AVX-512 and 32 zmm) and with
inlining defeated on BOTH kernels so the contrast holds outlining fixed.

**Overlap check against redwood's g++ 11.4, which licenses using a different host and compiler:**

    instantiation          redwood (g++ 11.4)      MKT (g++ 13.3)
    SHIP NVEC=3            46 insn                 46 insn
    SHIP NVEC=4            42 insn                 42 insn
    deep NVEC=3            72 insn, 12 stack       75 insn, 15 stack
    deep NVEC=4            87 insn, 23 stack       88 insn, 24 stack

Within 1-3 instructions. The compiler difference does not carry the result.

## At NVEC=1, where the regression lives

    variant      nnz/body  insn  insn/nnz  stack/nnz  pf/nnz
    SHIP              4.0    30      7.50       0.00     0.50
    deep PF=1         4.0    44     11.00       0.00     1.00

**Zero spills**, +47% instructions, prefetch rate doubled. Same 4 FMAs and 4 `vbroadcastss` on both
sides, so the arithmetic is identical and the extra 14 instructions per body are all overhead.

## Isolating the prefetch axis

`SCORCH_NARROWK_DEEP_PF` is already a compile-time axis, defaulting to 1, and the source says why it
exists: "Prefetch is its own axis, so depth and prefetch can be attributed separately." Turning it
off under `-funroll-loops` changes the machine body as well (4 -> 8 nonzeros, because lower register
pressure lets gcc unroll further), so per-nonzero rates there compare two different loops -- the
same dependent-body-width confound as the probe's V8. Rebuilding with `-fno-unroll-loops` makes
every body equal to its source step and moves the axis alone:

    variant      nnz/body  insn  insn/nnz  pf/nnz
    SHIP              2.0    16      8.00     0.50
    deep PF=1         4.0    44     11.00     1.00
    deep PF=0         4.0    20      5.00     0.00

Bodies matched at 4 nonzeros, **the prefetch axis alone costs 24 instructions per body = 6.00
instructions per nonzero.** Per-body delta from PF=1 to PF=0:

    prefetcht2  -4      the prefetches themselves
    movslq      -4      load A1_crd[pA + UNROLL + u]
    shl         -4      multiply by B1_size
    lea         -2      form the address
    add         -1        "
    mov         -1        "
    cmp         -4      the per-prefetch bounds check
    jle         -4      its branch
                ---
                -24      = 6.00 per nonzero, matching 44 - 20 exactly

So each prefetch costs six instructions, only one of which is the prefetch. Note the address
arithmetic is 4 instructions for 4 prefetches on the movslq and shl rows but only 4 across
lea+add+mov -- one address computation is shared, so the compiler has already amortised part of
it. An earlier version of this list omitted `mov -1` and summed to 23 against a measured 24; the
enumeration has to close against the difference of the two instruction counts, because the
memorable sentence is the enumeration's and not the total's.

## The mechanism, from the source

    if (PF) {
      for (int u = 0; u < UNROLL; u++)
        if (pA + UNROLL + u < pA_end)
          __builtin_prefetch(B_val + (size_t)A1_crd[pA + UNROLL + u] * (size_t)B1_size, 0, 1);
    }

`UNROLL` prefetches per body, each with its own bounds check. The shipped kernel issues ONE
prefetch per 2-nonzero body. So the prefetch rate per nonzero went from 0.5 to 1.0 **purely as a
side effect of doubling the chain count** -- the loop prefetches once per chain, and the chain count
is what the deep kernel changes.

That is the generalizable statement: **prefetch coverage per nonzero is the tuned quantity, and
this kernel changed it from the measured 0.5 to 1.0 without that being the change under test.** The
shipped kernel deliberately under-covers (it prefetches one of its next two rows); the deep kernel
scaled to full coverage. It is not a shape-specific threshold, and it is checkable in the
disassembly before anything is timed.

## Registered prediction, before any arm runs

1. `SCORCH_NARROWK_DEEP_PF=0` with `UNROLL=4` is FASTER than the shipped kernel at NVEC=1 widths
   (k=4 and k=8), because at matched bodies it runs 5.00 instructions per nonzero against the
   shipped kernel's 8.00 -- the four-chain gain showing up as fewer loop-control instructions per
   nonzero. If PF=0 is merely less bad than PF=1 and still behind the shipped kernel, the
   instruction count is not what is binding and this account is incomplete.
2. The gain from PF=0 is LARGEST at NVEC=1 and smallest at NVEC=4, because at NVEC=4 the spill
   (24 stack moves) survives PF=0 and is a separate cost.
3. It should NOT be a uniform win. chain197 already measured that deleting the register-block
   prefetch wins 2.9-14.2% on the shipped kernel but HARMS 115 matrices at width 32, so prefetch
   has real value somewhere and PF=0 should show a harmed tail of its own. A clean sweep would be
   evidence the arm is not doing what this document says it does.

## Two fixes this suggests, in order of how well justified they are

- **Keep prefetch coverage per nonzero fixed at the measured rate** rather than letting it scale
  with UNROLL: issue `UNROLL/2` prefetches per body, not `UNROLL`. This restores the quantity that
  was actually tuned and is the change with a mechanism behind it.
- **Remove the per-prefetch bounds check**, in either of two forms, and note which figure goes
  with which. The condition `pA + UNROLL + u < pA_end` is monotone increasing in `u`, so the
  `u = UNROLL-1` test dominates the group.
    *Hoisting* it replaces 4 `cmp` + 4 `jle` with 1 + 1, so it recovers **6** of the 24
    instructions per body -- a group-level check is still a check inside the body.
    *Loop splitting* reaches the full **8**: iterate the bulk while `pA < pA_end - 2*UNROLL + 1`,
    where the guard is statically true, and handle the final group separately, leaving the bulk
    body with zero checks.
  Both are correct independent of what the timing says, and the distinction matters because the 8
  is reachable but hoisting alone does not reach it.

## Scope

Disassembly only, one host, float32, NVEC=1 outlined under duress (both kernels, so the contrast
holds). `-fno-unroll-loops` is NOT the production configuration, so 5.00 against 8.00 is a
matched-body instruction count and not a prediction of production's instruction count. Nothing here
has been timed.
