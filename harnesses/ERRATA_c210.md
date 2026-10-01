# chain210 errata, recorded BEFORE the data landed

`an_c210.py` is bound by digest `9c42d96123ce3560d2420ec1ee9cd25408b2aac9ed372aa4e2ede433fdfabef5`
(23026 bytes) in commit `23e9700`. It is therefore **not amended**; every defect found in it is
recorded here and folded into `PROVENANCE.md` at seal time. Reporting the defect at the read point
rather than fixing a pre-registered estimator is deliberate — adopting a fix is Bobby's call.

This file is copied into the ledger as `ERRATA.md` when the tree is sealed.

## 1. The per-pass timing line is printed in (pass, build) order, not chronological order

The timing block iterates `for key2 in sorted(ev)`, which sorts by `(pass index, build)`. The
rotation alternates which object runs first within a pass — `p1: aa bb`, `p2: bb aa`, `p3: aa bb`,
`p4: bb aa`, `p5: aa bb`, `p6: bb aa` — so for every even pass the printed pair is in the reverse
of the order it ran.

**Every aggregate the block computes is order-invariant** (median, stdev, CV, and the sum over the
eleven passes other than `p1 aa`), so no number in the verdict moves. What the defect breaks is
anything a reader derives from the printed *sequence*, and consecutive differences are the obvious
one. On chain207's sealed log:

| | sequence | consecutive deltas |
|---|---|---|
| as printed, `sorted(ev)` | 1370 1399 **1418 1405** 1423 | 29, 19, **−13**, 18 |
| chronological, file order | 1370 1399 **1405 1418** 1423 | 29, 6, 13, 5 |

The multiset is identical. The printed order manufactures a non-monotone step that did not happen:
chain207's first five passes are **monotone increasing**, a 53 s climb (3.87%), and the `−13` is an
artefact of the sort. The chronology is recoverable from the same log — the stage script appends
each line as it happens, so **file order is the chronology**, and `p1 aa` is genuinely first in both
chains, which is what makes holding it out as the conditioning observation correct.

Found by the peer session after I published the wrong delta sequence from it.

## 2. Two ordering errors of my own, in the course of checking item 1

Recorded because they are the same class and one of them would have been a false alarm about the
design's central premise.

**(a) I published the wrong delta sequence.** I read deltas off `sorted(ev)` output and wrote that
chain207's first five "scatter by 13–29 s between consecutive passes". The true range is 5–29 s and
the sequence is monotone. The conclusion it supported — that chain210's passes are far tighter — is
unaffected (the statistic is a variance ratio over an order-invariant multiset), but the sentence
offered as intuition was false.

**(b) My first attempt at a chronological sort was by seconds-of-day, and chain207 ran across
midnight.** It started 22:20 and finished 03:02, so sorting `HH:MM:SS` numerically puts `22:xx`
after `00:xx`. The resulting "chronology" began at `p3 bb` and reported **`ran SECOND: aa 4, bb
2`** — i.e. that the rotation is *unbalanced*, which is the premise the whole aa-vs-bb comparison
rests on. The real counts are 3 and 3 in both chains. Caught because a sequence beginning at `p3
bb` is impossible, not because the counts looked wrong. This is the second midnight-rollover error
of the session; the first was a `date -d` elapsed calculation that returned a negative interval.

## 3. What the pass-timing dispersion does and does not license

chain210's first five passes have stdev 2.70 s against chain207's first five at 20.82 s —
position-matched, F = 59.38 on 4 and 4 df, one-sided p = 0.0008. The shape is sharper than the
ratio: chain207 ramps monotonically 1370 → 1423 over five passes and then plateaus at 1409–1421
(median 1414, stdev 4.96, Spearman against position −0.393), while chain210 is flat from its first
pass and sits 4.5% below that plateau.

Three limits on reading this into the per-cell floor:

- A pass duration is a sum over 19100 cells. A sum of *N* cells with per-cell CV *c* has CV
  `c/sqrt(N)` = `c/138.2`, so explaining chain207's observed 1.020% pass CV by independent per-cell
  noise would require **c = 141%**. Pass-level dispersion is therefore common-mode — frequency,
  thermal settling, co-tenancy, memory state — and not per-cell variance.
- The per-cell statistic is a **ratio** of ours to MKL timed inside the same pass, which cancels
  common mode (measured at the band level earlier: about 4x).
- A drift that is **linear in pass position is cancelled by the balanced rotation** itself: each
  object runs second in exactly three of the six passes, so the later-is-slower penalty falls
  equally on both and cancels to first order in every cross-build ratio. What survives to the floor
  is the non-linear residue, which neither session has bounded.

So the two runs demonstrably differ in common-mode drift, and the own-floor statistic is the one
that absorbs it — which is why the common-floor branch of P0 is the one to read first. That is a
reading order over branches already present in the pinned file, not an amendment. It is **not** a
prediction that the floors differ by anything like 7.71x.

## 4. The blind was broken before the verdict, by me

I published chain210's five landed pass durations (1348, 1350, 1352, 1355, 1353) in a message to the
peer session. Both sessions therefore knew, before the twelfth pass landed, that the partial run's
pace points at H3 (1348.0 s) rather than H1 (1412.5) or H2 (1389.8). What preserves the verdict is
that pre-registration here is **by digest** rather than by blinding: the output is a deterministic
function of sealed inputs and code pinned at `9c42d96123ce3560`, so neither session's expectation
can move it. Recorded so that no reader mistakes the eventual H1/H2/H3 line for a blind adjudication.

## 5. The rotation balance is checked from three sources, and two of them are declarations

The stage log carries the rotation **three** ways: the up-front `rotation (balanced, ...)` print,
each pass header's `order [x y]` field, and the landed sequence of `--- pN t start` lines in file
order. The first two are the **schedule** — both are printed from `$ORDER` before the pass body
runs — and only the third is what happened. chain207's `PROVENANCE.md` already states the
convention: GUARD 4 reads the landed order, never the schedule.

So "prefer a declared field to a derived one" needs its pairing clause to be usable here: the
landed sequence is authoritative for the premise, the declarations are the cheap independent route,
and **requiring all three to agree** is what catches both a lossy derivation and a script that did
not run its own plan. `collect_c210.sh` refuses unless all three agree and each object ran second
in exactly three of the six passes — which is the condition under which a drift linear in pass
position cancels to first order in every cross-build ratio.

Verified on chain207's sealed log: plan, headers and landed order all read
`aabb bbaa aabb bbaa aabb bbaa`, aa second 3, bb second 3. Confirmed independently by the peer
session. For chain210 the same check runs at collection time; the parser was verified against the
real partial log, where it reads plan 6, headers 4, landed 4 and refuses as incomplete.

The planted cases that fire it: all six passes in the same order (aa second 0, bb second 6); a
landed order that differs from declarations which agree with each other; and five passes instead of
six. The second of those is the one a declaration-only check cannot see.
