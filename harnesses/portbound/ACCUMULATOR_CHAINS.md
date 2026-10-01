# The shipped kernel's accumulator-chain count is `2 * NVEC`, and NVEC is a vector count

Written 2026-10-01, while the degree sweep was running. This is the connection between the
probe and the production kernel, and it includes two objections to my own framing.

## The structural fact

`scorch_spmm_row_regblock` declares `typename V::vec acc0[NVEC], acc1[NVEC]` and runs the
nonzero loop two at a time. So the number of independent accumulator chains is **`2 * NVEC`**,
and `NVEC = ceil(k / V::lanes)` is a function of the output width alone:

| k  | NVEC | accumulator chains | chain210 median `ours/mkl` (aa) | chain207, for contrast |
|----|------|--------------------|---------------------------------|------------------------|
| 2  | 1    | **2**              | 0.8815 | 0.8864 |
| 4  | 1    | **2**              | 0.9130 | 0.9231 |
| 8  | 1    | **2**              | 0.8676 | 0.8849 |
| 16 | 2    | 4                  | 0.7596 | 0.7907 |
| 32 | 4    | 8                  | 0.7101 | 0.7409 |

**CORRECTION, and it is the kind this project has a memory file about.** The first version of this
table printed 0.8864 / 0.9231 / 0.8849 / 0.7907 / 0.7409 under the heading "chain210" -- those are
**chain207's values, exactly, to four decimals**, and chain207 is the run whose P4 assay-validity
check FAILED at 4.34% and 4.09%. I carried them out of chain207's `PROVENANCE.md`, which was in
context, and they arrived already believed. The peer caught it; both columns above are recomputed
from the sealed CSVs of each chain independently (12 CSVs, 19100 aa cells each, per-cell
`exp(median(log(ours/mkl)))` then the per-width median).

The argument is slightly **stronger** on the clean run: chain210's step from the two-chain group to
k=16 is 0.8676 -> 0.7596, wider than chain207's 0.8849 -> 0.7907. And a one-value spot check would
not have caught the substitution -- k=2 reads 0.8864 in chain207 and 0.8815 in chain210, a
difference smaller than several of the others -- so it is the whole vector that identifies the
source, not any single entry.

**The trait column has been REMOVED rather than corrected.** It read "k=4 -> `__m128`, 4 lanes" but
`scorch_spmm_row_regblock`'s `V` parameter defaults to the full 8-lane `scorch_simd<T>`, and the
kernel is instantiated BOTH ways -- the default at spmm.h:4877/4878 and the half-vector `VH` at
4939/4943 -- behind the `spmm_halfvec` gate at 4415, which is env-overridable at 4327. So which
trait a k=4 cell runs is a selector decision I had not checked, and asserting it was the last
unverified column in this document. The chain count is unaffected either way, since `ceil(4/4)` and
`ceil(4/8)` are both 1 and k=4 therefore has two chains on both paths -- which is why the argument
survives the removal untouched.

The chain count was never chosen for the dependency structure. It is `2 * NVEC` because the
loop is 2-way unrolled and there is one accumulator per vector, and the vector count follows
from the width. At the three narrowest widths that lands on two chains, which the probe measures
as the slowest configuration available.

## What the probe adds, and it is narrow

At fixed width, fixed load count per nonzero (3.00 mem/nnz attested), and the same instruction
count to within 0.25 insn/nnz, going from 2 to 4 accumulator chains is **-21.259% on the P-core
and -4.032% on the E-core**, and 4 to 8 is nothing (+0.192% / -0.381%, at the placement floor).
That is the chain count isolated from everything else, which is what the production code's own
comment says had never been done at this width:

> "the 302-matrix sweep read nvec=1 at 0.8034 (k=4) and charged it to DEPTH; masking four of
> eight lanes is worth about that on its own ... so depth was never the thing measured there.
> With V supplied by the caller the depth question can be asked at the width the shipped kernel
> actually uses."

## OBJECTION 1 TO MY OWN FRAMING: the chain count does not explain k=4 specifically

k=2, k=4 and k=8 all have two chains, and they read 0.8864, 0.9231, 0.8849 -- within 0.038 of
each other. But the behind-MKL cells do **not** spread across the three: 19 of chain210's 22 are
k=4, and k=4 carries half of chain206's 14 on SuiteSparse. So the chain count is a candidate
explanation for the **step from the 2-chain group to the multi-chain group** -- 0.88-0.92 against
0.7907 and 0.7409, with the two multi-chain widths ordered by chain count -- and it is **not** an
explanation of why k=4 within that group carries the deficits. Something else distinguishes k=4
from its two neighbours, and anyone quoting the table above as explaining the k=4 concentration is
quoting it wrong.

**The candidate I named for that was vector occupancy, and its SIGN depends on the path, so it is
withdrawn until the path is known.** I wrote that k=4 is `__m128` with four of four lanes live
against k=2's two of four masked -- an ordering in which k=4 is the best-off of the three, making
occupancy a poor candidate. On the full-vector path k=4 is instead `ceil(4/8) = 1` with
`FULL_LAST` false, i.e. **four of eight lanes masked**, and this file's own kernel comment calls
that the bad case: the mask lands on the only vector, `vmaskmovps` is two uops on the load side
and cannot fold into the FMA as a memory operand, and spmm.h:2195 says plainly that "at k=4 it
runs `_mm256_maskload_ps` for EVERY nonzero over 4 lanes of 8". On that path k=4 is the
**worst**-off of the three, which is the sign a candidate for 19 of 22 behind-MKL cells needs.

So the same mechanism is a poor candidate or a good one depending on a selector decision, and the
honest state is that it is undetermined. Settling it is one query against the decision export, not
a measurement, and it is the next thing to do on this thread.

## OBJECTION 2: NVEC and k move together, so the table is confounded

Larger k means more work per index load and per A-value broadcast, so the amortisation of the
per-nonzero overhead improves with k independently of the chain count. The table cannot separate
them. What separates them is the probe, which holds k fixed at 4 and varies only the chain count
-- and that is why the probe matters more than the correlation. The table is corroborating
context; the test is the probe.

## A PRIOR NULL ON THE SAME-SOUNDING QUESTION, AND WHY IT DOES NOT TRANSFER

`spmm.h` already records a measured null for stream depth, over 118 matrices below MKL parity, on
redwood, with an A/A of 1.5% of cells outside +-10%:

    S = 2 / 4 / 8 against the shipped single stream
      k=1  0.968 / 0.972 / 0.968     k=2  0.994 / 0.995 / 0.997
      k=4  1.001 / 0.994 / 1.001     k=8  1.000 / 1.000 / 0.999

and concludes "the accumulator chain the extra streams break was never the limit". That is a real
result on real matrices and it is 21 points away from the probe's P-core figure, so it has to be
reconciled rather than ignored.

It reconciles on the kernel family, and the arithmetic is not close. That null is for the
**multi-stream nonzero-axis gather** kernel, where one `VGATHERDPS` plus one FMA covers **eight**
nonzeros. Its accumulator dependency is therefore 4 cycles per 8 nonzeros = **0.5 cycles/nnz**,
which cannot bind against anything -- so adding streams there has no dependency to break, exactly
as measured. The regblock kernel at NVEC=1 issues one FMA per **one** nonzero per chain, giving
4/2 = 2.00 cycles/nnz, which is what the probe's V0 measures at 1.994. The two families'
per-nonzero accumulator depths differ by **8x**, so a null in the gather family is not evidence
about the regblock family. The production comment anticipates this: it says the regblock_deep
comment "reasons from the same premise and should be read with this result next to it."

## WAS THE 21% DEPENDENCY DEPTH OR LOADS IN FLIGHT? The probe's own V3 answers it.

Four chains give four independent load streams as well as four independent accumulator chains, so
the probe's 2-to-4 step changes both and cannot separate them by itself. But V3 already measures
what the B load is worth: V3 removes the random, column-indexed B load entirely and makes B
register-resident, and it is **-1.420% on the P-core and -0.192% on the E-core**. The whole cost
of the random load -- its port slot and its latency together -- is 1.42%. A 21.259% gain cannot be
the covering of a 1.42% cost. So at this footprint the gain is the accumulator dependency and not
memory-level parallelism, and the gather-family null agrees with that for its own structure.

**Scope limit, stated because it is the one that could overturn this.** The probe's B is
`nrowsB = 8192` rows of `K = 4` floats = **128 KB**, which is L2-resident on redwood. The
production comment records the k<=lanes deficit tracking B's distance from the core: 0.894 of MKL
with B under 32 KB, 0.591 at L2 size, 0.489 at L3 size. So V3's 1.42% bounds the load cost **in
the L2 band only**, and at an L3-sized or larger B the load latency is much greater and extra
chains could buy real memory-level parallelism on top of the dependency gain. A footprint sweep
over `nrowsB` is the companion to the degree sweep and is not yet run. Until it is, "the gain is
dependency, not MLP" carries "at a 128 KB B".

## WHAT IS ALREADY BUILT, which changes what remains to be done

`scorch_spmm_row_regblock_deep` already exists: templated on `UNROLL`, on `PF` (prefetch), and on
`V` (so the half-width trait can be supplied), with `UNROLL >= 2` static-asserted and
`NVEC * UNROLL` accumulators held to about half the 16 architectural YMM registers. It is reached
through `SCORCH_NARROWK_UNROLL` and the `SCORCH_NARROWK_DEEP_UNROLL / _PF / _NVEC_LO / _NVEC_HI`
build defines. So no new kernel has to be written to test this end to end on the board -- which
means the next step is a measured grid on real matrices, not more probe work.

Its comment also records, before any of tonight's data, the risk the degree sweep is measuring:
"Rows shorter than UNROLL fall entirely to the scalar tail, which has LESS instruction-level
parallelism than the 2-stream form it replaces, so a deep unroll is expected to lose on short
rows -- that is what the A/B measures, and it is why this is a hook and not yet a policy." The
probe's registered break-even prediction (degree ~9-19 on P, ~33 on E) is a quantitative form of
the same expectation, and the DLMC degree distribution says that band holds the bottom 10-25% of
keys.

And it records that the previous depth rejection was confounded twice over: once by swapping the
half-vector block for a masked 256-bit one, and once by dropping the prefetch the shipped 2-deep
kernel carries, so "a 14.5% harmed tail was attributed to the depth, but no arm ever ran deep
WITH a prefetch". Both confounds are now template parameters. That is why this is worth
re-measuring rather than a settled question.

## What would make this a shipping proposal

1. The degree sweep lands and the break-even is below the degree band the corpus actually holds,
   or a guard exists that provably cannot fire above it. (running)
2. A footprint sweep over `nrowsB`, so the dependency-not-MLP reading is not L2-only.
3. `assert_close` against a PyTorch reference at `atol=rtol=1e-3`, because four chains sum in a
   different order than two.
4. A measured grid on the real board through the existing hook -- both collections, all five
   widths, two hosts -- with no regression anywhere, since the convention makes one regression a
   blocker regardless of the size of the win.
5. Whatever rule picks `UNROLL` has to be justified by the register budget rather than fitted to
   a width, or it is the overfitting the standing task rules out. `NVEC * UNROLL` held near a
   constant is such a rule; a table of per-width unroll factors is not.
