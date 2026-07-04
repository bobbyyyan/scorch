// scorch_policy.h — single source of truth for scorch's OpenMP parallel policy.
//
// Two inline helpers compute a work-aware thread cap (`scorch_nthreads`) and an
// adaptive schedule chunk (`scorch_chunk`) from ONE formula. They are shared by
// all three parallel SpGEMM/SpMM code paths, which differ ONLY in their per-call
// work measure and grain:
//
//   * JIT codegen (compiler/codegen.py): the generated kernels call these two
//     helpers. csrc/header.cpp — the text-prepended JIT preamble — #includes this
//     file, and src/scorch/utils.py get_extra_cflags() adds csrc/ to the JIT
//     `-I` path so the include resolves at compile time.
//   * prebuilt spmspm_csr (csrc/kernels.h): work = A_nnz*avg_B_row (flop),
//     grain = SCORCH_GRAIN_SPMSPM.
//   * prebuilt spmm_csr_float_v2 (csrc/spmm.h): work = A_nnz*k, grain = SCORCH_GRAIN_SPMM.
//
// WHY (validated on redwood i9-14900K, a hybrid P+E CPU, back-to-back vs the old
// unconditional-all-cores + coarse-fixed-chunk policy): an unconditional
// `#pragma omp parallel` over-threads small products (fork/join + O(rows)
// per-thread workspace dwarf the work — a 130-row product ran 4-7x SLOWER than
// PyTorch), and a coarse fixed chunk starves load-balancing so the join barrier
// stalls on the slowest cores. So (a) bound the worker count two ways and take
// the smaller — by WORK (>= one grain of work per worker) and by ROWS (>= ~16
// rows per worker) — then (b) size the schedule chunk to ~7 chunks per worker so
// every core, fast or slow, stays fed. omp_get_num_procs() is the stable OS count
// (omp_get_max_threads() is mutated by torch run-to-run).
//
// The constants below are the tuning surface. They are tuned on redwood and are
// robust-but-not-universally-optimal across CPUs (the policy SHAPE transfers with
// no P/E-topology constants; only the CONSTANTS vary by host). Phase 4b (per-host
// install-time autotune) rewrites them for the build host.
//
// NOTE: the codegen grains are emitted as literals by compiler/cin_lowerer.py
// (flop path = 1500; A_nnz sites rely on the 500 default arg below) because they
// are baked into each generated kernel at codegen time. Keep those in sync with
// the intent here; a future single-surface refactor could have the generated code
// reference these named constants instead.

#pragma once

#include <omp.h>

// --- Tunable policy constants (Phase 4b install-time autotune target) --------
// Per-kernel work grain: minimum "work" per worker thread. The work measure
// differs per call site, so the grain does too (flop vs A_nnz vs nnz*k).
constexpr long SCORCH_GRAIN_SPMSPM = 3000;    // prebuilt spmspm_csr; work = A_nnz*avg_B_row
constexpr long SCORCH_GRAIN_SPMM = 150000;    // prebuilt spmm_csr_float_v2; work = A_nnz*k
constexpr long SCORCH_GRAIN_DEFAULT = 500;    // codegen A_nnz-path default (see note above)

constexpr long SCORCH_ROWS_PER_THREAD = 16;   // >= this many rows per worker
constexpr long SCORCH_CHUNKS_PER_THREAD = 7;  // dynamic-schedule chunks per worker
constexpr long SCORCH_CHUNK_MIN = 4;
constexpr long SCORCH_CHUNK_MAX = 64;

// Work-aware OpenMP thread cap. work < 0 means "unknown" -> cap by rows only.
inline int scorch_nthreads(long work, long rows, long grain_default = SCORCH_GRAIN_DEFAULT) {
  int hw = omp_get_num_procs();               // stable; torch mutates omp_get_max_threads
  long n = rows / SCORCH_ROWS_PER_THREAD;     // >= ~16 rows per worker
  if (work >= 0) {
    long by_work = work / grain_default;      // >= grain_default work per worker
    if (by_work < n) n = by_work;
  }
  if (n < 1) n = 1;
  if (n > (long)hw) n = hw;
  return (int)n;
}

// Adaptive schedule chunk: ~SCORCH_CHUNKS_PER_THREAD dynamic chunks per worker.
inline int scorch_chunk(long rows, long work, long grain_default = SCORCH_GRAIN_DEFAULT) {
  int nt = scorch_nthreads(work, rows, grain_default);
  long c = rows / (nt * SCORCH_CHUNKS_PER_THREAD);
  if (c < SCORCH_CHUNK_MIN) c = SCORCH_CHUNK_MIN;
  if (c > SCORCH_CHUNK_MAX) c = SCORCH_CHUNK_MAX;
  return (int)c;
}
