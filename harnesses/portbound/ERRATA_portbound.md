# Errata: the port-boundedness probe

## 1. The first published figure came from an UNATTESTED binary, and I said otherwise in the present tense

I wrote that "the V5 row now carries 2.12 mem/nnz, 1.00 alu/nnz beside -1.069%". It did not. The
analyzer and `run_probe.sh` had just been written and shipped; `ATTEST.txt` did not yet exist and the
attested run was still in flight. The **-1.069%** figure came from `pb_core0.csv` as produced by the
**eight-arm** binary, which carries no `# binary_sha256` line and predates the current `probe.cpp`.
Any `mem/nnz` obtained by disassembling a later build describes a source edited after those timings
were taken, so the two columns were from different binaries and the table's juxtaposition made it
read as though they were not. Found by the peer session, from file mtimes and the V0 median.

This is the earlier lesson one level up: a gate that rejected three attempts attested nothing about
the fourth, and then the attestation that *did* exist had not been applied to the number printed
beside it.

**Both guards were then confirmed on planted input**: the same data with its digest line stripped is
refused for carrying no `# binary_sha256`, and a mismatched digest is refused by name. The guard
rejects the exact file whose unverifiability prompted it.

**The attested value is -1.013%**, with `dev(V5/V1) = 0.00022` against a 0.00090 floor. The
refutation stands; the figure moved by 0.06 points.

## 2. The absolute level is not reproducible across runs; the within-run ratios are

Three runs of the same probe on the same pinned core:

| run | V0 ns/nnz | V1 - V0 | V5 - V1 |
|---|---|---|---|
| 8 arms, 31 reps | 0.349923 | -1.076% | +0.0071% |
| 8 arms, 61 reps | **0.424528** | -1.076% | +0.0071% |
| 9 arms, 61 reps (attested) | 0.349898 | -1.035% | +0.0217% |

The middle run reads **17.6% slower in absolute terms** while its within-run differences are
identical to three decimal places. So the interleaved randomised-order design does what it was for:
it protects the contrasts against drift in machine state. But **no claim may be made in absolute
ns/nnz**, and in particular a cycles-per-nonzero figure needs a frequency measured inside the same
run -- a carried frequency would inherit a 17.6% error. This is a scope limit the pre-registration
did not state.

## 3. The calibration loop is hoisted, so a timed block is 182 ms and not the documented 5 ms

`for (int i = 0; i < 64; i++) sink += v0(A);` calls a side-effect-free function with the same
`const&` argument 64 times, so the compiler computes it once. The arithmetic from the timed data
confirms it in both directions:

- V0's real per-call time is **1433.2 ns**; calibration measured **39.4 ns**, 36x too small.
- Hoisting predicts `one_call / 64` = 22.4 ns, giving `inner` = 223279. Observed `inner` = 126833 --
  the same order, and *smaller*, which is what a cold first call (39.4 > 22.4 ns) implies. Both
  inequalities point the right way.
- A block is therefore `inner x 1433 ns` = **181.8 ms**, not 5 ms.

**Benign for the verdict**, because the registered requirement is that a block *exceed* 5 ms and it
exceeds it 36-fold. Recorded because a reader deriving the block length from the source would be
wrong by a factor of 36, and because it is why each run takes minutes rather than seconds. Note that
counting call sites in the disassembly does **not** detect this -- a correct 64-iteration loop also
shows exactly one call site -- which is why the evidence here is arithmetic rather than a count.

## 4. V4c costs MORE than V4, so the cost is the load port and not a cache stream

| control | what it adds | P-core cost |
|---|---|---|
| V4 - V4b | one load from a separate 64 KB array | +12.08% |
| **V4c - V4b** | one load from `B0 + K`, same cache line 3 times in 4 | **+14.65%** |

V4c was added because V4 conflates a load-port operation with a new memory stream and so cannot
isolate port pressure. The clean control costs *more* than the confounded one, which settles the
direction: a fourth load-port operation is expensive on its own terms.

**The asymmetry is the unexplained part and is left unexplained.** Adding a fourth load-port
operation costs 14.65%; removing one of three buys 0.60-1.42%, and those three are themselves only
an upper bound because the floor is an arm floor and not a layout floor. A factor of ten between
adding and removing is not accounted for by any mechanism measured here. Candidate causes --
`B0 + K` straddling a 64-byte line one time in four and splitting the load, or a limit that binds
at four operations and not at three -- are untested, and five mechanisms proposed over this session
had wrong signs or wrong shapes. It is recorded as an open asymmetry.

It does not affect the refutation, which rests on V5 sitting at V1 with identical load counts and
identical footprints in both arms.
