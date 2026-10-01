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

The middle run reads **21.33% slower** while its within-run differences are identical to three
decimal places. So the interleaved randomised-order design does what it was for: it protects the
contrasts against drift in machine state. But **no claim may be made in absolute ns/nnz**, and in
particular a cycles-per-nonzero figure needs a frequency measured inside the same run -- a carried
frequency would inherit a 21.3% error. This is a scope limit the pre-registration did not state.

**Correction to this item, which first read 17.6%.** `0.424528 / 0.349898 = 1.2133`, so the slow run
is 21.33% slower; `0.349898 / 0.424528 = 0.8242`, so the other two are 17.58% *below* it. I quoted
the second as "17.6% slower", which normalises a slowdown by the slow run itself. `dev(x) =
exp|ln x| - 1` -- defined in `an_portbound.py:28` and used for every other comparison in this probe
-- returns **0.2133** for the pair. The number I published disagreed with a function in the same
file, and it understated the anomaly by 3.75 points, making the scope limit weaker than the data
supports rather than stronger.

## 3. The calibration loop is hoisted, so a timed block is 182 ms and not the documented 5 ms

The conclusion needs no mechanism and is stated that way: `block = inner x real per-call`, which
requires only the header's `inner` and the timed V0 value and never asks why the calibrator
under-measured.

- V0's real per-call time is **1433.2 ns**; the calibrator measured **39.4 ns** (= 5e6 / 126833), a
  **36.35x** collapse.
- A block is therefore `inner x 1433.2 ns` = **181.8 ms** against the documented 5 ms.

**The hoist story was my sixth proposed mechanism this session and it misses by 1.76x.** A full
hoist of the 64-iteration loop to one call predicts a measured per-call of `1433.2 / 64` = 22.39 ns.
The measurement is 39.42 ns, so the effective surviving call count is `64 / 36.35` = **1.76**, not 1.
My "cold first call makes it smaller" sentence absorbed exactly the gap it needed to, which is the
shape five earlier mechanisms failed on. The hoist is withdrawn as a claim; the collapse factor is
the measurement.

**And a second-order argument, from the peer session, is internal-consistency evidence against the
pure-hoist reading.** A hoist is a property of the binary, so it would predict a FIXED collapse
factor. Across two runs of the same binary:

| run | V0 ns/nnz | real/call | measured/call | collapse | block |
|---|---|---|---|---|---|
| attested | 0.349898 | 1433.2 ns | 39.42 ns | 36.35x | 181.8 ms |
| slow | 0.424528 | 1738.9 ns | 32.31 ns | 53.81x | 269.1 ms |

The factor moves 48% on one binary, so whatever under-measures is at least partly a run-time
quantity rather than a code transformation. Same argument shape as a fixed batch failing to pin
`nnz x k`. It also means **neither "181.8 ms" nor "36x" is a harness property** -- both are per-run;
quote the range, 182-269 ms and 36-54x. The benign verdict is safe either way, since the registered
requirement is that a block *exceed* 5 ms and it exceeds 36 to 54 fold.

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
direction against **both** hypotheses I posed: a fourth load-port operation is expensive on its own
terms, and the extra 64 KB stream is worth **minus 2.2%** -- V4 is faster than V4c despite identical
emitted counts (4.00 mem, 1.00 bcast, 1.00 vpor per nonzero in both). So Amendment 13 is right that
the two are conflated and wrong about which dominates.

On the E-core the same pair reads V4 +17.487% and V4c +22.001%, so the reversal holds on both core
types and is larger on the narrower one. Reported separately and never pooled, per the scope limits.

**The asymmetry is the unexplained part and is left unexplained, and it is nearer twenty than ten.**
The attested count table does not compose: V0 reads 3.00 mem/nnz and V4b = V0 + accumulate reads
3.12, so V4 and V4c should read 4.12 and both read 4.00. An added load is therefore emitted as
**0.88 mem/nnz**, which puts the per-operation cost at `14.636 / 0.88` = **16.6%** rather than
14.6%. Against removals of 0.60-1.42% -- themselves only an upper bound, since the floor is an arm
floor and not a layout floor -- the factor to explain is closer to 20x than 10x. No mechanism
measured here accounts for it. Candidate causes --
`B0 + K` straddling a 64-byte line one time in four and splitting the load, or a limit that binds
at four operations and not at three -- are untested, and five mechanisms proposed over this session
had wrong signs or wrong shapes. It is recorded as an open asymmetry.

It does not affect the refutation, which rests on V5 sitting at V1 with identical load counts and
identical footprints in both arms.
