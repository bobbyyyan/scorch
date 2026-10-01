# Per-ROW exposure of a degree-gated four-chain kernel, measured over the whole DLMC collection

The peer's objection was correct and it voided my own inference. The degree sweep puts V6's
break-even at row degree 24–32, so a four-chain kernel needs a guard. I quoted its exposure from
chain210's `deg` column — 1051 of 3820 keys (27.5%) below average degree 32, holding 3.06% of all
nonzeros — then asserted that a per-ROW guard "should have none". That last step was an inference,
not a measurement: `deg` is nnz/rows, a per-matrix average, and chain210's columns
(`build, slice, key, k, rows, nnz, deg, arm, ms, batch`) are per-matrix aggregates only. No
row-length distribution exists anywhere in the `spmm-beat-mkl` ledger, so the per-row form had no
number in either direction. **"Pay a per-row branch and regress nothing" was a claim about rows
backed entirely by numbers about matrices.**

It needed no re-run and no values. A DLMC `.smtx` file is three lines — `rows, cols, nnz`, the CSR
row-pointer array, then the column indices — so every row length is a successive difference of
line 2. `harnesses/an_rowdist.py` reads one line per file.

## The measurement

**3846 files read, 0 malformed, 3402920 rows, 475663122 nonzeros.** The nonzero total matches
chain210's exactly, which is the check that this is the same collection the board was scored on.
(3846 files against chain210's 3820 keys: 26 files the board did not score.)

| threshold | rows below | % of rows | nonzeros in them | **% of nonzeros** |
|---|---|---|---|---|
| 4 | 208128 | 6.1% | 118916 | **0.03%** |
| 8 | 321734 | 9.5% | 764157 | **0.16%** |
| 12 | 473506 | 13.9% | 2204980 | **0.46%** |
| 16 | 591681 | 17.4% | 3785810 | **0.80%** |
| 24 | 777243 | 22.8% | 7414836 | **1.56%** |
| **32** | **958746** | **28.2%** | **12367173** | **2.60%** |
| 48 | 1195781 | 35.1% | 21729586 | **4.57%** |
| 64 | 1423602 | 41.8% | 34199481 | **7.19%** |
| 128 | 1988992 | 58.4% | 88386481 | **18.58%** |

**A per-row guard at degree 32 fires on 28.2% of rows and sends 2.60% of the collection's
arithmetic to the fallback.** Those rows run exactly today's kernel, so this is foregone speedup,
not regression. The remaining **97.4% of the arithmetic** is eligible for the faster kernel.

The per-matrix and per-row aggregates happen to agree closely — 3.06% against 2.60% — but they are
different quantities and the agreement is a coincidence of this collection rather than a reason to
substitute one for the other.

## The per-matrix view, which is what the performance convention counts

Fraction of a matrix's own nonzeros living in rows shorter than 32:

    median 0.0000   p75 0.3525   p90 0.9750   p99 1.0000   max 1.0000
    >= 50% of nonzeros in short rows:  886 of 3846  (23.0%)
    >= 90%:                            493 of 3846  (12.8%)
    >= 99%:                            345 of 3846  ( 9.0%)

The median is **exactly zero**: more than half the collection's matrices have no row shorter than
32 at all, so they take the fast path everywhere. At the other end, **9.0% of matrices are
essentially entirely short rows and would get no speedup whatever** — the most extreme is
`transformer/variational_dropout/0.98/body_encoder_layer_2_self_attention_multihead_attention_v`
at 1.0000. Under a per-row guard those matrices do not regress; they simply do not improve.

So the honest statement of the shipping position, with the quantity each number is about:

- **No matrix regresses**, because the guard is per row and each row independently takes the
  kernel that is faster at its own length.
- **97.4% of the collection's arithmetic** is eligible for the faster kernel.
- **9.0% of matrices gain nothing**, and a further 3.8% (the gap to the 12.8% at the 90% cut) gain
  very little.
- The cost not yet measured is the guard itself: a compare and branch per row, plus the code size
  of two row kernels in the same loop. That is the next thing to price, and it is the only part of
  this that could turn a no-regression claim into a regression.

## Scope

DLMC only — the SuiteSparse collection has its own row-length distribution and this says nothing
about it. The 24–32 bracket is the P-core break-even for chain deepening **alone**; the 2x2 result
shows the chains only pay fully when the load count also drops, and the break-even for that
combination has not been measured at any degree.
