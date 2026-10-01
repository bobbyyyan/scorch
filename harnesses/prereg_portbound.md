# Pre-registration: is the k=4 float32 SpMM inner loop limited by load-port throughput?

Written 2026-10-01 while chain210 is running, so nothing in it can be tuned to its result. No
measurement in this document has been taken. Nothing here runs until chain210's twelfth
object-pass has landed and the host is free — CPU work beside a pre-registered timing run is the
perturbation.

## Why this question and not another

Every single-constant lever for the k=4 deficit is closed by measurement: the nonzero-axis kernel
(`scorch_spmm_row_gather_f32`, `spmm.h:2373`, measured 0.686–0.940 at k=4), `EXACT_HI_F32 = 4`
(moves k=4 onto a 26–29-instruction scalar kernel), the per-nonzero address multiply (2
instructions of 99, latency-only on chains already independent across nonzeros), the register-block
prefetch (chain197, closed), and MULTIROW (serial tail on the degree band the residues occupy).

What remains is a mechanism rather than a constant: at k=4 a B row is 16 bytes, so one 64-byte line
holds four consecutive B rows, and the measured realised reuse at high density is 2.2–2.7
nonzeros per distinct line — a 53–63% ceiling on B-load count. Translating that into time requires
knowing whether the loop is actually limited by load-port throughput. The honest arithmetic: −55%
on B *loads* is about −18% on a load-port bound (3 ops per nonzero falling to 2.45), and **zero** if
the loop is bound on instruction issue or on the accumulator dependency chain instead.

**That −18% is port-invariant, which I briefly retracted too broadly.** With load-port count `P`,
time per nonzero is `ops/P`, so the ratio is `(2 + 1/reuse)/3` and `P` cancels exactly:

| P | baseline | with reuse 2.226 | ratio |
|---|---|---|---|
| 2 | 1.5000 | 1.2246 | −18.359% |
| 3 | 1.0000 | 0.8164 | −18.359% |
| 4 | 0.7500 | 0.6123 | −18.359% |

It survives a heterogeneous pool too, since each core type carries the same ratio and any weighted
average of per-type times scales by it. What a wrong port count *does* kill is the **absolute**
figure — `3/2 = 1.500` cycles per nonzero was the two-port case — and every cycles-per-nonzero
number, because two clock domains leave no single frequency to divide a pooled ns figure by. The
relative figure stands for any `P`, conditional on the loop being load-port bound at all, and that
conditional is exactly what this document tests. Peer session's correction of my retraction.

**This document decides whether to implement the line-reuse kernel at all.** A refutation closes
the lever; that is the more likely and the more useful outcome.

## Why counters and a port model are both unavailable here

- `perf_event_paranoid = 4` on redwood. That denies unprivileged access to hardware counters
  entirely. I will not change a kernel setting to make a measurement convenient.
- No `llvm-mca` and no IACA on redwood, and none on the Mac either (checked: no `llvm-mca`, no
  `llvm-objdump`, nothing in Homebrew's llvm path, nothing in the `scorch` env). So a static
  port-pressure model is not available without installing a toolchain.
- **The host is hybrid, which breaks a single port model on principle.** redwood is an Intel Core
  i9-14900K: 24 cores / 32 threads, comprising 8 SMT-enabled cores at 5.7 GHz max (cpu0–15, paired
  siblings) and 16 non-SMT cores at 4.4 GHz max (cpu16–31, `core_id` 32–47). AVX-512 is absent from
  `/proc/cpuinfo` flags, which is why `-march=native` emits AVX2. The two core types are different
  microarchitectures with different load-port counts and a 1.30x clock ratio, so a pool-wide
  measurement averages two answers and a cycles-per-nonzero figure derived from a pooled ns figure
  has no single frequency to divide by.

Consequence: the measurement must be an **ablation**, single-threaded and pinned, repeated on each
core type. Not a counter read, not a port model, and not a pooled timing.

## Design

A standalone probe compiled from the shipped header, calling one row kernel in a loop over a
synthetic CSR row with a fixed nonzero count and a fixed column-index pattern. `taskset -c 0` for a
P-core and `taskset -c 16` for an E-core, run separately and reported separately — never pooled.

Variants. Each changes the number of load-port operations per nonzero while holding the FMA count
and the accumulator dependency structure fixed. V1–V3 and V5 are **throughput probes and not
candidate kernels**: they compute the wrong answer by construction, which is stated here so that no
later reader mistakes one for a proposed implementation.

| | variant | load-port ops/nnz | what it tests |
|---|---|---|---|
| V0 | shipped inner loop: index load, A broadcast, folded B load | 3 | baseline |
| V1 | column index replaced by a sequential counter | 2 | remove the index load |
| V2 | A value hoisted out of the loop | 2 | remove the broadcast |
| V3 | B operand register-resident | 2 | remove the B load |
| V4 | V0 plus one dead load whose result is unused | 4 | **positive control** |
| V5 | V1 plus one register-to-register ALU op (`vpxor`) | 2, issue +1 | **the discriminator** |

**V5 is the element that makes this a test rather than an instruction count.** Removing a load
reduces load-port pressure *and* issue pressure, so V1 alone cannot distinguish them. V5 restores
the issue slot without restoring the load. If V5 lands at V0, the binding constraint was the load
port; if V5 lands at V1, it was issue width.

### Both V4 and V5 can be deleted before they measure anything, and the failure points at the lever

Three defects in the first version of this design, found by the peer session:

1. **A dead load and a dead ALU operation are both removable at `-O3`** unless the result is
   consumed. A deleted V4 reads as "an extra load is free", which is the positive control reporting
   success having measured nothing.
2. **`vpxor` with identical source operands is a zeroing idiom**, recognised at register rename on
   Intel since Sandy Bridge, so it retires without consuming an execution port. Written as
   `vpxor %ymmN,%ymmN,%ymmN` it can survive dead-code elimination and *still* not restore the issue
   slot it exists to restore.
3. **The failure direction is the dangerous one.** A vacuous V5 reads identical to V1, and an
   identical V1/V5 pair is precisely the signature that would be read as "the constraint was the
   load port" — the conclusion favourable to the line-reuse lever. The most likely way this probe
   misleads is by confirming the mechanism it was built to test.

Required, therefore, and binding on the implementation:

- The extra operation in V5 must be a **non-idiom** vector ALU operation on **distinct live source
  registers** (not a zeroing or all-ones idiom, not same-register), writing into one of **at least
  four rotating sink registers** whose final values are stored after the loop. Four sinks so the
  sink's own 1-cycle latency chain cannot become the binding constraint, and a post-loop store so
  nothing is dead.
- V4's dead load must likewise be consumed — into a rotating sink, from an address that is live and
  distinct from the B load's.
- **Every variant's emitted assembly is disassembled and checked before any timing is read.** For
  each variant: count memory-operand instructions and vector ALU instructions in the hot loop, and
  assert the variant differs from V0 **in the intended way and only in that way**. A variant whose
  assembly does not match its intended delta is a refusal, not a data point.

This is the rule already in force for planted test inputs, applied to a compiled probe: verify the
plant before trusting the verdict on it. Here the plant is the object code and the planter is the
optimiser.

## Estimand, stated before any data

Per-nonzero time, reduced as `exp(median over repetitions of log(ns per nonzero))`, per variant,
per core type. Deviation `exp|ln x| − 1`. The noise floor is a same-binary A/A: V0 compiled a second
time under a different symbol name in the same object, timed as a seventh arm, floor = p95 of the
per-repetition deviation of V0a/V0b. Nothing inside that floor counts. Repetition count set so each
timed block exceeds 5 ms; arms interleaved in randomised order; the index pattern fixed across
variants so no variant sees a different cache footprint.

## Predictions, and the refutation condition

Let `s` be one load-port slot in ns per nonzero at the pinned core's frequency.

- **If load-port bound:** `t(V0) − t(V1) ≈ t(V0) − t(V2) ≈ t(V0) − t(V3) ≈ s`; `t(V4) − t(V0) ≈ s`
  and is positive beyond the floor; `t(V5) ≈ t(V0)`.
- **If issue bound:** the V1/V2/V3 drops are equal to each other but `t(V4) − t(V0)` is inside the
  floor, and `t(V5) ≈ t(V1)`.
- **If accumulator-chain bound:** none of V1, V2, V3 moves beyond the floor.

**Refutation: if `t(V5)` is at `t(V1)` rather than at `t(V0)`, or if V4 is free, the line-reuse
kernel cannot pay and the lever is closed.** That result gets reported, not worked around.

## Scope limits, declared in advance

A port-structure conclusion is specific to one microarchitecture. This probe measures Raptor Cove
and Gracemont. It does **not** transfer to MKT (Zen4, AVX-512, different load-port count) or to the
M5 (ARM, NEON, the only ARM host so there is no two-host confirmation available for it at all).

So even a positive result does not license shipping a line-reuse kernel unconditionally. It would
license implementing it behind a gate, with the ablation repeated per host before the gate is
allowed to fire there — and a gate keyed on realised reuse, which is a runtime property, provably
cannot fire below about 0.05 density where measured reuse is 1.073 (a 6.8% ceiling) and where we
already run at 0.34–0.52 of MKL across 1083 keys.

### Which core types production actually runs on — measured, not assumed

Reporting per core type raises the question of which type carries production weight. Measured on the
live chain210 run by sampling per-thread `utime+stime` from `/proc/<pid>/task/*/stat` twice, 15 s
apart, so the figure is **residency** rather than a last-CPU snapshot:

- `Cpus_allowed_list: 0-31` — no affinity restriction; placement is the scheduler's.
- **Exactly 24 of 47 threads consumed CPU** in the window, which identifies the OpenMP team and
  shows the other 23 are idle. A last-CPU snapshot mixes the idle ones in and gives 17/30 instead.
- **63.3% of CPU time on E-cores (cpu16–31), 36.7% on P-logical (cpu0–15).**
- Total 8838 ticks in 15 s = **5.9 fully busy cores out of a 24-thread team**, about 25% occupancy,
  consistent with this project's earlier finding that the whole-row balance ceiling is occupancy
  rather than time.

Scope: one 15-second window inside one object-pass of DLMC float32 with widths 2–32 interleaved.
Not established for SuiteSparse or for any single width.

**The per-thread instrument is unusable and was replaced.** A second attempt at the same sample
refused its own stability check: 47 threads before and after, but 32 TIDs differed, so `join` would
have silently dropped most of them. Six further 5-second windows were all void (32, 32, 40, 9, 17
and 8 TIDs changed). The process churns threads continuously; the one window that passed was the
exception, and a property measured in the only window that passed its check is selected on the
instrument.

`/proc/stat`'s per-CPU lines accumulate busy ticks per CPU regardless of which thread ran, so they
are immune to churn, and they are valid here because this process is essentially the only load on
the host. Three 10-second windows:

| window | busy cores of 24 | P-logical | E | P-cores with both siblings busy | E-cores busy |
|---|---|---|---|---|---|
| 1 | 5.93 | 36.0% | 64.0% | 0 of 8 | 16 of 16 |
| 2 | 4.65 | 37.6% | 62.4% | 0 of 8 | 16 of 16 |
| 3 | 4.79 | 39.5% | 60.5% | 0 of 8 | 16 of 16 |

**Two verdicts, and they must not be stated as one.** The sentence that first appeared here —
"reproduces the fragile one (36.0% against 36.7%, 5.93 busy cores against 5.89)" — quoted window 1
for both figures, which is the row nearest the instrument being validated. Separated:

- **The core-type split reproduces.** E share 64.0 / 62.4 / 60.5%, mean **62.3%**, range 3.5 points,
  against the voided instrument's 63.3%. That conclusion holds.
- **The occupancy level does not.** Busy cores 5.93 / 4.65 / 4.79, mean **5.123**, range 1.28 =
  27.5% of the minimum; occupancy 24.7 / 19.4 / 20.0%, mean **21.3%**; the all-idle-is-in-kernel
  ceiling is 4.05x / 5.16x / 5.01x, mean **4.68x** — not the single 4.07x figure published first.
  That spread is five to seven times the −18.4% this probe exists to measure.

A fourth window, taken later while sampling frequencies, was first published as **11.9–13.7 busy
cores** because the sweep overhead inside it was not timed. **My own description of that loop
excludes the slow end.** It is 10 × (32 reads at 1–5 ms, then `sleep 2`), so elapsed is 20.32–21.60
s and the figure is **12.71–13.51 busy cores**, occupancy 52.9–56.3%, ceiling 1.78–1.89x. The
published 11.93 would require 23.01 s, which the loop cannot produce. Peer session's catch; the
tightening came free from a structure I had already described in prose while calling the measurement
untaken.

Three further windows were then taken with the denominator **measured** rather than inferred —
`date +%s%N` either side, elapsed 10.002 / 10.004 / 10.004 s — and with the per-CPU detail retained
for every window rather than reduced away:

| window | busy cores | occupancy | ceiling | sibling share of P-logical | `T_E/T_P` raw | stripped |
|---|---|---|---|---|---|---|
| a | 7.097 | 29.6% | 3.38x | 3.99% | 2.110 | 2.197 |
| b | 6.058 | 25.2% | 3.96x | 7.03% | 1.784 | 1.919 |
| c | 4.581 | 19.1% | 5.24x | 5.70% | 1.614 | 1.712 |

Across all six windows occupancy spans **4.58 to 13.51 busy cores, a factor of 2.9**, and the
ceiling spans **1.78x to 5.24x**. Sampling overhead cannot explain any of it: 320 `cat` forks at
about 1 ms of CPU each is 32 ticks of 27445, 0.12%.

**And the time split is phase-dependent too, which the work-share conversion's input cannot
survive.** `T_E/T_P` reads 1.778 / 1.660 / 1.532 in the first three windows and 2.110 / 1.784 /
1.614 in the second three — a six-window span of **1.532 to 2.110, 38%**. So the "mean 1.656" used
in the conversion was one phase rather than the run, and the resulting E work-share bound moves with
the window set: 57.7% from the old mean unstripped, 58.7% old mean stripped, 61.5% new mean
stripped, 64.4% at the highest single window. **No windowed figure feeds the conversion.**

**The spread is 13x beyond counting noise, which is why it reads as phase and not as scatter.**
Treating tick counts as Poisson, the per-window relative SE on `T` is `sqrt(1/n_E + 1/n_P)` =
2.54–3.04%, mean 2.82%. The observed sd across six windows is 0.2022 = **11.6% of the mean, 4.1x
counting noise**, and the span is **13.3x** it. So "the range is 38%" is not an argument from the
spread looking large; it is four to thirteen times what the counters could produce, from numbers
already in hand. Peer session's point, and the number that makes it was free.

**A whole-run split needs two `/proc/stat` reads, and they must land on pass boundaries.** An
object-pass is a complete sweep of 3820 keys at five widths, so a whole pass *is* the run's
composition — but an interval of part of one pass plus `N` whole ones is missing a specific slice of
the corpus, whichever keys the sweep visits first.

My first anchor, at 02:45:03, was **376 s into p5 aa — 27.9% of the way through it** — so it
reintroduced exactly the selection the design exists to remove. Carried through: the partial pass
takes weight 0.419 at `N = 1` and 0.265 at `N = 2`, giving a bias coefficient on (tail − head) of
0.117 and 0.074, and at the head/tail difference the six windows make plausible (0.578 in `T`) that
is **0.043–0.068 in `T`**, roughly 1.5 points of E work share in an unknown direction. Small against
a 38% phase span, and the only error in this design that was avoidable for free. Peer session's
catch, flagged as time-critical with fourteen minutes to the next boundary.

Re-anchored on the **done-line count incrementing**, not on a clock, since boundary times are
estimates: a bounded script (deadline 04:23:15, at most four snapshots, PID recorded so it is killed
by exact PID and never by a pattern that could match itself) writes a `/proc/stat` snapshot at each
of the four remaining object-pass completions. That yields a one-pass interval and a three-pass
interval, both phase-complete. The 02:45:03 read is kept: against a clean interval it is a positive
control on whether the head/tail difference matters at all.

Note that this quantity is *not* available from the sealed CSVs — they record per-cell kernel time,
not per-core-type attribution — so it is the one input that requires a reading taken while the run
is live. **Conditional on `r` coming entirely from the pinned pair.** If any in-situ frequency
weighting survives into the final conversion, that is live-only too and must be taken **on the same
interval as `T`**, not on a different window, which would be the window-mixing error in a new place.
Peer session's correction to my scope claim.

**And a better `T` does not remove the `r` precondition.** `T` comes from this run's workload on a
loaded host; `r` will come from the pinned V0 pair on a free one. Composing them assumes the pinned
`r` equals the in-situ `r`, which is the occupancy mismatch already adopted: solo rates exceed
in-situ rates for a bandwidth- or latency-limited kernel, so the pinned `r` is an upper bound and E's
share is understated by it. The final figure stays a bound in both directions it was already a bound
in — a whole-run `T` reads like the last loose input and it is not.

**No 10-second window is a property of the run.** The run sweeps 3820 keys at five widths, so a
window samples whichever matrix sizes happen to be in flight; three windows are three different
workloads, not three draws from one occupancy. The windowed occupancy figure is therefore dropped as
a headline. The whole-run computation — per-cell timings summed across the sealed CSVs against the
per-object-pass wall times — is the only phase-free version of the quantity, which is an argument
for the method already chosen rather than a new one. Peer session's correction, in both halves.

**Thread placement: one thread per physical core, across all 24 physical cores, with no meaningful
SMT sharing.** The per-CPU tick counts printed as illustration (even CPUs 0,2,…,14 at 136–547, odd
siblings at 1–22, all 16 E-cores at a uniform 155–193) are **window 3 only**, which is recoverable
from the numbers themselves: w3's 4790 total ticks at a 60.5% E share predicts 2898 E ticks and
181.1 per core against 2897 and 181.1 observed, while w1 predicts 237.2 per core and is excluded.
The peer session found the inconsistency from the published figures alone and could not resolve it
without the raw counters; the resolution is that the detail is one window's and the illustrative
range was then used as though it characterised all three.

The *finding* is not window-limited: the both-siblings-busy count is computed inside each window's
reduction and is 0 of 8 in all three. But it is a **threshold** (>50 ticks of 1000), not an
observation of zero — siblings run at 0.1–2.2% busy in w3, and in a later 20-second window they
reached 1.5–7.7%. The conclusion is robust at any sane threshold; the column should not read as an
exact zero.

**And the exclusion belongs in the denominator too.** Stripping sibling ticks from the numerator
alone gives P 1812/4786 = 37.9%, which answers "what share of all busy ticks landed on P even
cores". The conversion needs "what share of the work was done by P physical cores", so the sibling
time comes out of both sides: P 1812/4709 = **38.5%**, E **61.5%**, which raises w3's `T_E/T_P` from
1.532 to **1.599** and E's bounded share at `r = 1.322` from 53.7% to **54.7%**. Small, and in the
direction that raises E. Peer session's correction, both parts.

**That refutes my own objection to the work-share correction.** I had argued that a P-logical
CPU-second is a second on half a shared core, which would put `r_P/r_E` near 0.78–0.84 and make the
direction unknowable from clocks. No SMT sharing is happening, so the objection does not apply and
the peer session's direction stands: `r_P/r_E` is at least the 1.295x clock ratio and higher once
vector-FMA width differs between the two microarchitectures.

**Time share is not work share, and the probe supplies its own conversion.** With `T = W/r`:
`W_E/W_P = (T_E/T_P)·(r_E/r_P)`. The figure first published here, `T_E/T_P = 1.72`, was **the voided
instrument's number** (63.3/36.7 = 1.7248) carried across an instrument replacement. From the
surviving instrument: 1.778 / 1.660 / 1.532, mean **1.656**. E work share at `r_P/r_E` = 1.295 /
2.0 / 2.5 is then **56.1% / 45.3% / 39.9%**, against the 57 / 46 / 41% published from the voided
number. Direction unchanged and magnitudes barely moved, so this is bookkeeping rather than a
retraction — but a figure should come from the instrument that survived its own check.

`r_P/r_E` does not need a separate experiment: **V0 pinned to cpu0 and to cpu16 measures it
directly**. Until it does, **both arms are reported unweighted** and the time split is stated as a
time split.

**Two preconditions on that conversion, both of which make the converted share a bound.**

1. *Occupancy mismatch.* A solo pin measures `r_P/r_E` at occupancy 1 while the time split it
   converts was measured at occupancy 24. If the kernel is bandwidth- or latency-limited with all
   cores live, per-core rates converge toward the shared limit, so the solo ratio is an **upper**
   bound on the in-situ one. The direction is knowable because E's share falls monotonically in `r`
   (56.1 → 45.3 → 39.9%), so too high an `r` understates E's work share. Either pin the pair at the
   run's own occupancy — 8 threads on P against 16 on E, same total work — or label the converted
   share a bound and name its direction. Peer session's point, and theirs to raise since they
   proposed the conversion.
2. *The clock ratio is not `r`.* `r = (IPC_P/IPC_E) · (f_P/f_E)`, and only the second factor is
   measurable this way. The first is unmeasured and is `≥ 1` for Raptor Cove against Gracemont on a
   kernel of this shape. Demoting the vector-FMA-width argument removed the *justification* for
   `r > f_P/f_E`; it did not turn the clock ratio into an estimate of `r`. So the clock-only input
   yields an **upper bound** on E's work share, because the share falls monotonically in `r`:

   | `r_P/r_E` | 1.322 | 1.5 | 2.0 | 2.5 |
   |---|---|---|---|---|
   | E work share at `T_E/T_P = 1.656` | **55.6%** (bound) | 52.5% | 45.3% | 39.8% |

   And the pinned V0 pair measures `r` **directly, as a rate**, so the clock ratio is not needed as
   a conversion input at all — it is a cross-check on the pin, which is a better use for it. Peer
   session's correction.

   **Read this before quoting anything from the clock subsection below.** Everything there —
   the refuted contamination model, the base-frequency anchor, the coherent operating point, the
   error bar — concerns `f_P/f_E` **only**. `f_P/f_E` was never the factor in doubt. `r` is still
   unmeasured, floored by the clock ratio and with **no upper bound**, and nothing in this document
   pins it to 1.30–1.33. The risk the peer session named is that the strongest-sounding sentence in
   the section is about the factor that was never at issue.

3. *The clock figure itself, and the test its instrument needed.* 1.295 is 5.7/4.4, the
   **single-core boost** ratio, which is the wrong clock for an all-core run. Measured in situ —
   ten sweeps of all 32 CPUs across one window, weighted by the per-CPU busy ticks of that same
   window, `intel_pstate` on `powersave`: **P-logical 3.902 GHz, E 2.952 GHz, ratio 1.3218**.

   Offering "within 2% of the boost ratio" as reassurance was the wrong move, and the peer session's
   objection is the right shape: agreement with a spec number is also what a **duty-contaminated**
   reading produces, since a TSC-normalised `scaling_cur_freq` would report `f_max · duty` and the
   ratio would collapse to `1.2955 · (d_P/d_E)`, needing only `d_P/d_E = 1.020` to land on 1.3218.
   Agreement therefore discriminates nothing; the thing to show is that the reading is busy-only.

   **Tested, and the contamination model is refuted.** An almost-idle sibling, cpu13 at 1.8% busy,
   reads **5.24–5.47 GHz**; duty-inclusive would predict `5.7 × 0.018 = 0.103 GHz`. So the figure is
   C0-only, not TSC-normalised. The test also explains why two of my own sweeps disagreed: five
   back-to-back reads return one cached value (4198243 five times) while reads 2 s apart vary
   (4.198 → 3.890 → 5.289 GHz on the same CPU), so it is an interval-averaged quantity with real
   variance under a bursty workload — the ten-sweep mean averages that, the single sweep caught one
   interval. `cpuinfo_cur_freq` is absent on this kernel, so there is no second hardware route.

   **The replacement reassurance was the same error in a new costume, and the numbers say so.**
   I offered `base_frequency` 3.2/2.4 = 1.3333 as a third route and wrote that "three routes bracket
   the clock ratio at 1.30–1.33". But base and boost are **2.92% apart**: Intel holds the P:E ratio
   nearly constant across the frequency range, so that window was fixed by the product before
   anything was measured, and two of the three "routes" are one source. Computed from my own data —
   per-CPU ten-sweep means, busy-weighted, CPUs as the sampling unit:

   | | value | SE | 95% interval |
   |---|---|---|---|
   | `f_P` | 3.9036 GHz | ±0.1242 (n_eff 8.7) | |
   | `f_E` | 2.9535 GHz | ±0.0629 (n_eff 16.0) | |
   | `f_P/f_E` | **1.3217** | ±0.0506 (**3.8%**) | **[1.223, 1.421]** |

   The ratio's own 95% interval is **15.0% wide and contains the entire 2.92% spec bracket**, so the
   measurement discriminates none of it. The honest statement is `f_P/f_E = 1.32 ± 0.05`, consistent
   with every operating point between base and boost and distinguishing none. "Internally coherent"
   was true and is also exactly what a measurement carrying no information would look like — which
   is the objection to the boost-ratio agreement, arriving one paragraph later in new clothes.

   **And the floor is not independent of the measurement either, which is a third instance.** The
   natural rescue is "you only need a floor on `r`, and 1.29 is one whether the truth is 1.2955 or
   1.3333". But `cpuinfo_min_freq` is **800 MHz for both core types**, so at the bottom of the range
   `f_P/f_E = 1.000`. Intel holds the ratio near 1.30 only *between base and boost*, and what
   establishes that this run sits in that region is the measurement (P 3.90 ∈ [3.2, 5.7], E 2.95 ∈
   [2.4, 4.4]). So the spec cannot supply a floor without the measurement, and the measurement's
   interval reaches down to 1.223.

   **The interior of the range is not constrained at all, which is stronger than "the floor needs
   the measurement".** 1.2955 and 1.3333 are the ratios at two particular operating points. Between
   them each core type's P-state is selected independently by the governor from its own utilisation,
   so no mechanism makes the ratio at an intermediate pair lie between the ratios at the endpoints.
   That the two endpoint ratios agree to 2.92% is a fact about how Intel set four numbers, not a
   constraint on anything in between. So even granting the operating region, the bracket was never a
   bracket, and the floor on `r` is the measurement's own lower limit and nothing else. Peer
   session's strengthening of their own refuted rescue.

   **Use the `t` interval, not the normal one.** With relative variances 1.0123e-3 and 4.5355e-4
   (the P term is 69.1% of the total), Welch–Satterthwaite gives **14.64 df** and
   `t(0.975) = 2.1361`, so the interval is **[1.2136, 1.4298]**, 16.4% of the point estimate rather
   than 15.0%, and the floor on `r` is **1.2136**. Peer session's correction.

   **The two adopted corrections were never composed.** The sibling strip multiplies the time split
   by `1/(1 − sibling share)` and the clock interval supplies the floor on `r`; each was applied once
   and the bound was then computed from the *unstripped* split. Composed, and with the sibling share
   measured per window rather than taken from one — three further windows give 3.99%, 7.03%, 5.70%
   — the bound rises rather than falls. Peer session's catch, and the direction is the one that
   matters: E's share bound is **higher** than published, not lower.

   **One boundary on what the cpu13 test establishes.** It excludes TSC-normalisation; it does not
   by itself exclude a cached *requested* P-state. If the sysfs value were per-physical-core policy
   rather than per-logical `aperf/mperf`, 16 P-logical reads would carry only 8 distinct values and
   busy-tick weighting across 16 would double-count them. Checked from the same data: **all 8
   sibling pairs disagree** (by 0.04 to 0.73 GHz), so the value is per-logical and there is no
   double-counting. Peer session's boundary, answered from data already in hand.

   **The caching observation has a resolution and it belongs with it.** Five back-to-back reads
   returning one value threatens the independence of the ten sweeps the mean rests on. The loop is
   10 × (32 fast reads, then `sleep 2`), so reads *within* a sweep are 1–5 ms apart — that is where
   the cache returns one value — while consecutive reads of the *same* CPU are separated by at least
   2.00 s, exactly the spacing at which the value was shown to move (4.198 → 3.890 → 5.289 GHz). So
   there are about ten independent samples per CPU and the mean stands: caching bounds the usable
   cadence, not the mean. Peer session's rescue.

   **C0 time is not kernel time.** An `aperf/mperf` average covers all C0 time on that CPU — the
   kernel, the Python driver, CSR construction, anything else scheduled there — and weighting by
   busy ticks weights *CPUs*, not *code*. So this figure is the average frequency of everything that
   ran, not of the SpMM. Same fix as for `r`: the frequency taken during the pinned V0 runs is
   kernel-only, because nothing else is on the core.

4. *The two inputs came from different windows.* The frequencies are weighted by window 4's busy
   ticks (11.9–13.7 busy cores) and applied to a time split from windows 1–3 (4.65–5.93). That is
   the window-mixing this document had just finished arguing against, committed one paragraph later.
   P-cores throttle more steeply with active-core count than E-cores do, so a ratio taken at ~13
   busy cores is if anything a floor for a 5-core phase — the same direction as the `r` bound above,
   which is why this is bookkeeping rather than a reversal. Every input is now labelled with its
   window. Peer session's catch.

Finally, "higher once vector-FMA width differs between the microarchitectures" presupposes the
kernel is compute-bound, which is the hypothesis this probe exists to test. It stands as a
prediction, not as an input to any weighting.

### The occupancy question outranks this one, and is settled first

The same measurement puts the 24-thread team at **4.65–13.7 busy cores across four windows, 19–57% occupancy**, mean 21.3%
over the three comparable ones. On the low windows three quarters of the wall time is not in the
arithmetic at all, which is a larger quantity than anything this probe can win: if every idle slot
were in-kernel imbalance, perfect balance would be a 4.05–5.16x lever against −18.4%. **But the
quantity is phase-dependent by a factor of 2.5–2.9 across windows, so no windowed figure is a
property of the run** and the ceiling is quoted only as a range.

**The ceiling is phase-dependent too, and in the high-occupancy phase it is close to this probe's
own effect.** w1–w3 give 4.05x / 5.16x / 5.01x; w4 gives **1.75–2.01x**. So "occupancy outranks
ports" is true of the low-occupancy phases and much weaker in the high one. A whole-run ceiling is
`24 / (time-weighted mean occupancy)`, so the mean is exactly the quantity the whole-run
decomposition produces — which makes it the right estimand rather than a compromise. The per-phase
values are kept alongside it, because a lever that only pays in the low-occupancy phase is
misvalued by the mean. Peer session's framing.

**It is not established that any of it is in-kernel.** The figure is occupancy of the *harness
process*, and it sums at least three components that need different fixes and have different
production meanings:

1. in-kernel thread idling — real, and a kernel-level lever;
2. serial time between calls inside the harness — partly real, since production has per-call cost too;
3. Python and CSR tensor construction between invocations — **an artefact of the measurement
   harness with no production meaning at all.**

A 15-second window spanning 3820 keys at five widths cannot separate them, so the 4.07x is a
ceiling on a quantity whose composition is unknown, and the component that is pure harness overhead
cannot be removed by any kernel change. My reporting 25% occupancy as a finding about SpMM was
therefore too strong; it is a finding about the harness process that *bounds* an SpMM question.

**The priority inversion is accepted.** This probe answers "is the loop load-port bound"; the
occupancy decomposition answers "does the loop matter". A yes to the first with a no to the second
is a kernel change that measures beautifully in isolation and moves nothing in production, so the
decomposition is pre-registered separately and settled first. It needs no new run for its larger
half: the per-cell timings in chain210's twelve sealed CSVs, summed against the per-object-pass wall
times already recorded in `c210_stage.log`, bound component 3 plus 2 directly. That computation
waits for the seal — summing the per-cell timings of a partial run is reducing partial data, which
is the thing this project does not do.

This also sharpens the existing project finding that the whole-row balance ceiling is occupancy
rather than time. That finding closed row-splitting as a lever; it did not establish which of the
three components above produces the occupancy.

One further limit, stated because it is the weakest point of the design: a pinned single-threaded
probe is not the shipped configuration. The production kernel runs on a 24-thread pool spanning both
core types and contends for shared L3 and memory bandwidth. A loop that is load-port bound in
isolation can be bandwidth bound in the pool, in which case removing load-port operations buys
nothing in production. This probe therefore establishes a **necessary** condition for the line-reuse
change to pay, not a sufficient one, and the sufficient test is the full grid against MKL.

## Amendment record

Pre-data throughout: nothing in this document has been measured, and chain210 is still running.

- **Amendment 1**, digest `262a49000111c914` superseded. Retraction of the −18.4% figure was too
  broad; the ratio is port-invariant and only the absolute cycle figures die. Peer's correction.
- **Amendment 2**, same commit. V4 and V5 were deletable at `-O3`, `vpxor` on one register is a
  zeroing idiom that consumes no port, and a vacuous V5 fails *toward* the lever. Now requires
  non-idiom operations on distinct live registers with consumed results, and per-variant assembly
  verification with asserted counts and refusal on mismatch.
- **Amendment 3**, same commit. Added the measured core-type weighting of production, 63.3% E-core
  and 36.7% P-logical by residency, replacing an unstated assumption that the pool is P-core work.
- **Amendment 9**, digest `6449741246a44318` superseded. The whole-run anchor was 27.9% into an
  object-pass rather than on a boundary, reintroducing phase selection worth 0.043–0.068 in `T`;
  re-anchored on done-line increments by a bounded snapshot script, with the misaligned read kept as
  a positive control. The 38% phase span is quantified against counting noise (4.1x on the sd, 13.3x
  on the span). The scope claim is narrowed: a surviving in-situ frequency weighting would also be
  live-only and would have to share `T`'s interval. And a whole-run `T` is recorded as not removing
  the `r` precondition, so the composed figure stays a bound.
- **Amendment 8**, digest `64a566d0f6e8c38b` superseded. The sibling strip and the clock floor were
  each applied once and never composed; composed, with the sibling share measured per window
  (3.99–7.03%), E's work-share bound rises. The `t` interval replaces the normal one (14.64 df,
  [1.2136, 1.4298], floor 1.2136). The spec bracket is shown not to constrain the interior of the
  range at all. The fourth window's denominator is tightened to 12.71–13.51 busy cores by my own
  loop description, and three further windows were taken with the denominator measured and per-CPU
  detail retained. `T_E/T_P` is shown to be phase-dependent over 1.532–2.110, so the work-share
  bound spans 57.7–64.4% depending on the window set and no windowed figure may feed the conversion;
  a whole-pass `/proc/stat` anchor was written instead.
- **Amendment 7**, digest `e929ea9d753438bc` superseded. The "three routes bracket 1.30–1.33"
  reassurance is withdrawn: base and boost are 2.92% apart because the product holds the ratio
  constant, and the measurement's own 95% interval is [1.223, 1.421], 15.0% wide, containing the
  whole bracket. `f_P/f_E = 1.32 ± 0.05` discriminates nothing, the spec cannot supply a floor
  independently of the measurement (`cpuinfo_min_freq` is 800 MHz for both types), and the E
  work-share bound moves to ≤ 57.5%. The cpu13 test's boundary is closed by showing all 8 sibling
  pairs disagree, so the reading is per-logical. The caching observation gains its resolution. C0
  time is restated as not kernel time. A warning added that the whole clock subsection concerns
  `f_P/f_E` and not `r`.
- **Amendment 6**, digest `ea0fd535e9d99ead` superseded. The clock ratio is not `r`: the IPC factor
  is unmeasured and `≥ 1`, so the converted E work share is an upper bound of 55.6% rather than a
  point estimate, and the pinned V0 pair measures `r` directly as a rate. The "within 2% of the
  boost ratio" reassurance is withdrawn as non-discriminating; the instrument was instead tested and
  the duty-contamination model refuted (an idle sibling reads 5.24 GHz where duty-inclusive predicts
  0.103), with `base_frequency` 3.2/2.4 = 1.3333 supplying a third bracketing route. Clock ratio and
  time split labelled with their windows after being mixed. Sibling exclusion moved to both sides of
  the ratio. The occupancy ceiling recorded as phase-dependent, 1.75–2.01x in the high-occupancy
  window against 4.05–5.16x in the low ones.
- **Amendment 5**, digest `17c4d77e22268b83` superseded. The sentence certifying the new instrument
  quoted window 1 twice; the split reproduces and the occupancy level does not, and a fourth window
  widens occupancy to a factor of 2.5–2.9, so the windowed figure is dropped as a headline.
  `T_E/T_P` recomputed from the surviving instrument (1.656, not 1.724). Two preconditions added to
  the conversion, with the in-situ busy-weighted clock ratio measured at 1.322 against the 1.295
  boost ratio. The per-CPU illustration identified as window 3, and "zero SMT sharing" restated as a
  threshold. All five points from the peer session.
- **Amendment 4**, digest `499f35c80f80124b` superseded. The per-thread instrument was replaced
  after it voided itself in 6 of 7 windows; `/proc/stat` per-CPU accounting reproduces the split and
  additionally shows one thread per physical core with **zero SMT sharing**, which refutes my own
  objection to the peer's work-share correction. Time share is restated as not being work share,
  with `r_P/r_E` to be measured by V0 pinned on each core type and both arms reported unweighted
  until then. The occupancy question is accepted as outranking this one and is pre-registered
  separately, with my own 25%-occupancy claim narrowed to a statement about the harness process.
