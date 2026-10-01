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

So the E-core arm carries roughly two thirds of production weight on this corpus and the P-core arm
one third. Both are reported; the weighting is stated and never used to average them into one
number. Scope: one 15-second window inside one object-pass of DLMC float32 with widths 2–32
interleaved. It is not established for SuiteSparse or for any single width.

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
