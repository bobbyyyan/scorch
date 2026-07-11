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
//     `-I` path so the include resolves at compile time. The codegen flop path
//     passes SCORCH_GRAIN_CODEGEN_SPGEMM; A_nnz sites use the SCORCH_GRAIN_DEFAULT
//     default arg.
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
// ---- The tuning surface (Phase 4b per-host autotune) -----------------------
// The constants below are the tuning surface. The values written here are tuned
// on redwood and are robust-but-not-universally-optimal across CPUs (the policy
// SHAPE transfers with no P/E-topology constants; only the CONSTANTS vary by
// host). They are declared as #ifndef-guarded macros so a per-host autotune can
// override any subset WITHOUT editing this file:
//
//   * tools/autotune_policy.py measures THIS build host and writes
//     csrc/scorch_policy_tuned.h (gitignored) with `#define`s for the constants
//     it retunes. That file is #included FIRST below (when present), so its
//     defines win over the #ifndef defaults here.
//   * When the tuned header is absent — CI, cross-compile, `pip install` without
//     running the autotune — the redwood-tuned defaults below apply and this
//     header compiles standalone. This is the always-safe fallback.
//
// CACHE NOTE: the JIT kernel cache key (_kernel_name in src/scorch/utils.py) folds
// in the text of THIS file and the tuned header, so retuning busts stale .so's.
// The prebuilt scorch_ops is rebuilt by the autotune, so it picks up new values
// directly.

#pragma once

#include <omp.h>

// Install-time autotune sweep hooks. Compiled ONLY when tools/autotune_policy.py
// builds an instrumented scorch_ops with -DSCORCH_TUNE_HOOKS (see setup.py env
// SCORCH_BUILD_TUNE_HOOKS / utils.get_extra_cflags SCORCH_JIT_TUNE_HOOKS). Then a
// back-to-back threads x chunk sweep can force any cell in-process via env, with
// NO rebuild per cell. The shipped library defines nothing -> these evaporate and
// the helpers are pure computation (zero getenv overhead).
#ifdef SCORCH_TUNE_HOOKS
#include <cstdlib>
#endif

// --- Per-host autotune overrides (optional, generated, gitignored) -----------
#if defined(__has_include)
#  if __has_include("scorch_policy_tuned.h")
#    include "scorch_policy_tuned.h"
#  endif
#endif

// --- Tunable policy constants (redwood-tuned defaults = the safe fallback) ----
// Per-kernel work grain: minimum "work" per worker thread. The work measure
// differs per call site, so the grain does too (flop vs A_nnz vs nnz*k).
#ifndef SCORCH_GRAIN_SPMSPM
#  define SCORCH_GRAIN_SPMSPM 3000L        // prebuilt spmspm_csr; work = A_nnz*avg_B_row
#endif
#ifndef SCORCH_GRAIN_SPMM
#  define SCORCH_GRAIN_SPMM 150000L        // prebuilt spmm_csr_float_v2; work = A_nnz*k
#endif
#ifndef SCORCH_GRAIN_DEFAULT
#  define SCORCH_GRAIN_DEFAULT 500L        // codegen A_nnz-path default arg
#endif
#ifndef SCORCH_GRAIN_CODEGEN_SPGEMM
#  define SCORCH_GRAIN_CODEGEN_SPGEMM 1500L  // codegen 2-phase SpGEMM flop path (heavier
                                             // generic kernel -> smaller grain than prebuilt)
#endif

#ifndef SCORCH_ROWS_PER_THREAD
#  define SCORCH_ROWS_PER_THREAD 16L      // >= this many rows per worker
#endif
#ifndef SCORCH_CHUNKS_PER_THREAD
#  define SCORCH_CHUNKS_PER_THREAD 7L     // dynamic-schedule chunks per worker
#endif
#ifndef SCORCH_CHUNK_MIN
#  define SCORCH_CHUNK_MIN 4L
#endif
#ifndef SCORCH_CHUNK_MAX
#  define SCORCH_CHUNK_MAX 64L
#endif

// Dense-output parallel zero-fill threshold (csrc/header.cpp scorch_zero_dense,
// called by the JIT dense-output kernels): minimum OUTPUT BYTES at which the zero
// is parallelized across all cores. Below this a single memset is used — fork/join
// would exceed the saving. 256 KB keeps >= 2 pages per thread even at 32 threads
// and has a clear no-regression margin above the serial/parallel crossover (~32-64
// KB on redwood); every meaningful win (large outputs) is >= 1 MB. This is a
// zero-fill span threshold, NOT a work grain, so it is bytes rather than flop/nnz.
#ifndef SCORCH_MEMSET_GRAIN_BYTES
#  define SCORCH_MEMSET_GRAIN_BYTES 262144L
#endif

// Work-aware OpenMP thread cap. work < 0 means "unknown" -> cap by rows only.
//
// `nfloor` (default 1 = no-op for the SuiteSparse SpMM / SpMSpM / codegen callers,
// which stay byte-identical) sets a minimum worker count applied BEFORE the hw
// cap. The fused GCN kernel passes nfloor = omp_get_max_threads() so it never
// drops below the platform's default parallelism -- torch's per-platform thread
// count, which on a hybrid P+E CPU is the P-core count. That keeps the
// small/narrow-K GCN shapes at full P-core width (the nnz*k throttle alone
// collapsed them to ~1 thread), while big work still escalates via by_work up to
// omp_get_num_procs() so M5's bandwidth-bound big graphs keep all cores. Because
// this only ever RAISES the count toward what the platform already offers, it
// cannot reintroduce an all-cores E-core cliff beyond torch's own default.
inline int scorch_nthreads(long work, long rows, long grain_default = SCORCH_GRAIN_DEFAULT,
                           int nfloor = 1) {
  int hw = omp_get_num_procs();               // stable; torch mutates omp_get_max_threads
  long n = rows / SCORCH_ROWS_PER_THREAD;     // >= ~16 rows per worker
  if (work >= 0) {
    long by_work = work / grain_default;      // >= grain_default work per worker
    if (by_work < n) n = by_work;
  }
  if (n < (long)nfloor) n = (long)nfloor;     // platform floor (default 1)
  if (n < 1) n = 1;
  if (n > (long)hw) n = hw;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_TUNE_THREADS");
    if (e && *e) { long f = std::atol(e); if (f > 0) n = (f > hw) ? hw : f; } }
#endif
  return (int)n;
}

// Adaptive schedule chunk: ~SCORCH_CHUNKS_PER_THREAD dynamic chunks per worker.
inline int scorch_chunk(long rows, long work, long grain_default = SCORCH_GRAIN_DEFAULT) {
  int nt = scorch_nthreads(work, rows, grain_default);
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_TUNE_CHUNK");
    if (e && *e) { long fc = std::atol(e); if (fc > 0) return (int)fc; } }
#endif
  long c = rows / (nt * SCORCH_CHUNKS_PER_THREAD);
  if (c < SCORCH_CHUNK_MIN) c = SCORCH_CHUNK_MIN;
  if (c > SCORCH_CHUNK_MAX) c = SCORCH_CHUNK_MAX;
  return (int)c;
}
