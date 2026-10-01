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

One further limit, stated because it is the weakest point of the design: a pinned single-threaded
probe is not the shipped configuration. The production kernel runs on a 24-thread pool spanning both
core types and contends for shared L3 and memory bandwidth. A loop that is load-port bound in
isolation can be bandwidth bound in the pool, in which case removing load-port operations buys
nothing in production. This probe therefore establishes a **necessary** condition for the line-reuse
change to pay, not a sufficient one, and the sufficient test is the full grid against MKL.
