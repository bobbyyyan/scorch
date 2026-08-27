// scorch_policy.h — single source of truth for scorch's OpenMP parallel policy.
//
// Two inline helpers compute a work-aware thread cap (`scorch_nthreads`) and an
// adaptive schedule chunk (`scorch_chunk`) from ONE formula. They are shared by
// all three parallel SpGEMM/SpMM code paths, which differ ONLY in their per-call
// work measure and grain:
//
//   * JIT codegen (compiler/codegen.py): the generated kernels call these two
//     helpers. scorch/csrc/header.h — the packaged JIT preamble — includes this
//     file, and src/scorch/utils.py expands both resources into one self-contained
//     translation unit. The codegen flop path
//     passes SCORCH_GRAIN_CODEGEN_SPGEMM; A_nnz sites use the SCORCH_GRAIN_DEFAULT
//     default arg.
//   * prebuilt spmspm_csr (scorch/csrc/kernels.h): work = A_nnz*avg_B_row (flop),
//     grain = SCORCH_GRAIN_SPMSPM.
//   * prebuilt spmm_csr_float_v2 (scorch/csrc/spmm.h): work = A_nnz*k, grain = SCORCH_GRAIN_SPMM.
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
//     src/scorch/csrc/scorch_policy_tuned.h (gitignored) with `#define`s for the constants
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
// builds an instrumented scorch_ops with -DSCORCH_TUNE_HOOKS (see scorch_build.py
// SCORCH_BUILD_TUNE_HOOKS / utils.get_extra_cflags SCORCH_JIT_TUNE_HOOKS). Then a
// back-to-back threads x chunk sweep can force any cell in-process via env, with
// NO rebuild per cell. The shipped library defines nothing -> these evaporate and
// the helpers are pure computation (zero getenv overhead).
#include <cstdio>     // scorch_llc_bytes reads sysfs on Linux
#include <cstdlib>
#if defined(__APPLE__)
#include <sys/sysctl.h>
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

// Minimum number of structurally-empty output ELEMENTS before the drop-in SpMM
// zeroes them with one pre-loop parallel span instead of a serial memset per row.
//
// This gates an A/B arm, not the shipped path. The drop-in SpMM zeroes an empty
// output row in the row loop that was going to visit it anyway (spmm.h, zero_mode
// 2), which needs no threshold at all: it spawns no second team and makes no second
// pass, so there is nothing for a size gate to protect. The constant survives
// because the pre-loop span is still one of the arms that path is priced against,
// and that arm needs the gate it was measured with.
//
// The value: measured on redwood, the span arm beats the serial one by 2.099x
// (float32) / 3.152x (float64) on the 205 panel cells it fires on, and LOSES on 19
// of the float32 ones, by up to 1.9x. Those losses sit entirely below an 8 MB
// output span and, within that band, on the cells with the most arithmetic after
// the zero -- the cost of the first team is paid by whatever runs next. 512K
// elements is where the gate was left; it is not the value that makes the span arm
// safe, because no value does.
#ifndef SCORCH_SPMM_ZERO_SPAN_ELEMS
#  define SCORCH_SPMM_ZERO_SPAN_ELEMS 524288L
#endif

#ifndef SCORCH_ROWS_PER_THREAD
#  define SCORCH_ROWS_PER_THREAD 16L      // >= this many rows per worker
#endif
// Rows per worker for the SpMM specifically. 16 is a proxy for "enough work to be
// worth waking a worker", and it is a proxy the SpMM does not need: it knows the
// work exactly (nnz*k) and scorch_nthreads already divides that by the grain. The
// proxy is wrong in the one direction that matters, because a row is not a fixed
// amount of work -- it is deg*k. A pruned ResNet-50 bottleneck layer is 64 rows of
// degree 288, so at k=512 it carries 9.4 M multiply-adds, which is 62 grains of
// work, and 16-rows-per-worker throttles it to 4 workers on a 24-core host. Those
// cells run at 0.53-0.63x of MKL, which splits the free dimension instead. 1 says
// the row axis can feed one worker per row and lets the work term do the bounding
// it was already doing.
//
// This can only change a decision where rows/16 is itself below the core count --
// under 384 rows on a 24-core host -- AND the per-row work clears the grain, which
// is deg*k > 9375. Every matrix with more rows than that gets the identical thread
// count, so the GCN and autoencoder shapes, and reddit, are untouched by
// construction rather than by measurement.
#ifndef SCORCH_SPMM_ROWS_PER_THREAD
#  define SCORCH_SPMM_ROWS_PER_THREAD 1L
#endif
// Minimum NONZEROS per worker, as an alternative statement of the same "enough work
// to be worth waking a worker" requirement that SCORCH_ROWS_PER_THREAD states in
// rows. 0 keeps the row proxy alone, which is what ships.
//
// The row proxy and the raise above between them still leave one class stranded, and
// it is the class the SuiteSparse residual is now made of. Meszaros/kl02 is 71 rows
// holding 212536 nonzeros: rows/16 gives FOUR workers, and the raise cannot lift it
// because the raise is bounded by nnz*k, which at k=2 is 425072 units -- one grain
// and a bit. So a 1.7 MB L3-resident product runs on four threads and reads 0.593 of
// MKL, while per thread we are faster than MKL. Stating the requirement in nonzeros
// instead of rows lifts the ceiling to min(nnz/N, rows), and the work term
// (nnz*max(k,16) / grain) still does the bounding, so a tiny product cannot be
// over-threaded by this.
//
// max() with the row proxy, never min(): this can only ever RAISE the ceiling, so
// nothing that is fast today can be reclassified by it.
#ifndef SCORCH_SPMM_NNZ_PER_THREAD
#  define SCORCH_SPMM_NNZ_PER_THREAD 0L
#endif
// Real arithmetic each worker must get before the COMPOSITION ADOPTION hands it one.
// 0 keeps the adoption ungraded, which is what ships.
//
// The adoption is currently all-or-nothing: clear nnz*max(k,16) >= the grain and the
// whole host team is taken, miss it and the count falls back to the policy's, which
// for anything under one grain is ONE. That cliff is why pricing the gate on real
// arithmetic instead is risky rather than obviously right: a Cora output layer is
// 13264 nonzeros at k=7, which is 92848 multiply-adds, and it would go from 24
// workers to 1 -- the same defect in the other direction as the 12625-unit product
// that takes the whole team today. Grading the adopted count by real arithmetic gives
// it 6 instead, and leaves anything above a handful of grains at the host count.
#ifndef SCORCH_SPMM_ADOPT_GRAIN
#  define SCORCH_SPMM_ADOPT_GRAIN 0L
#endif
// Conditions on the nonzero-expressed row ceiling above. 0 disables a condition, which
// reproduces the ungated rule that measures null. The measured region is rows <= 128 and
// mean degree >= 192 on redwood; a plateau, not an edge -- rows in {96,128,192} crossed
// with degree in {192,256} all read 1.108-1.164 with z of 3.2-3.9.
#ifndef SCORCH_SPMM_CEIL_MAXROWS
#  define SCORCH_SPMM_CEIL_MAXROWS 128L
#endif
#ifndef SCORCH_SPMM_CEIL_MINDEG
#  define SCORCH_SPMM_CEIL_MINDEG 192L
#endif
// Whether the widened count is capped at the caller's thread pool instead of at
// omp_get_num_procs(). Candidate fix for the measured x86/ARM disagreement; off until
// both hosts have run it.
#ifndef SCORCH_SPMM_CEIL_CAP_POOL
#  define SCORCH_SPMM_CEIL_CAP_POOL 0
#endif
// Output size, in multiples of the last-level cache, above which the SpMM's row
// partition is turned off. See the measured table at the gate itself in spmm.h: the
// partition's gain decays monotonically with output size and goes negative past a
// few times the LLC, because home ranges scatter the output store stream across as
// many DRAM regions as there are workers where the global counter keeps it
// near-sequential. 0 disables the gate.
//
// Swept on redwood's large-A corpus (56 matrices, 204 float32 / 183 float64 cells,
// same-code floor 0.996/0.998). The geomean is flat across every threshold to within
// 0.3%; what the threshold buys is the TAIL. Fraction of cells more than 10% slower than
// what ships: 0.0%/0.5% at 1x LLC, 0.0%/0.5% at 2x, 1.5%/1.6% at 4x, 2.5%/3.8% at 8x,
// 2.9%/4.9% ungated. Broken out by output megabytes the partition gains 1.13-1.14x below
// 16 MB and 1.04-1.07x from 16-64 MB, then flattens; the harm is confined to 144-256 MB,
// where 8x reads 0.921/0.954 and ungated 0.915/0.950 while 1x and 2x hold 1.00.
// 2x rather than 1x because 1x also switches the partition off through the 16-64 MB band
// that still pays (1.0573 against 1.0730 on float32, 1.0252 against 1.0407 on float64).
// Whether the row partition is switched off when the policy resolved a single worker.
// With one worker there is no second core to keep A resident for and nothing to steal, so
// the partition can only cost the per-row difference between walking a home range and
// claiming from the counter. Provably inert at two workers or more, and off by default
// anyway -- not because the argument is weak but because flipping it would change what the
// `p3` arm means on the tiny cells partway through a study whose other arms are already
// measured. Priced as its own arm first, then flipped.
#ifndef SCORCH_SPMM_PARTITION_SOLO_OFF
#  define SCORCH_SPMM_PARTITION_SOLO_OFF 0
#endif
#ifndef SCORCH_SPMM_PARTITION_MAXOUT_LLC
#  define SCORCH_SPMM_PARTITION_MAXOUT_LLC 2L
#endif
// Which row-handout the SpMM uses by default. 0 = one global atomic counter, which is
// what ships today and what costs A its inter-call L2 residency; 3 = contiguous home
// ranges with stealing from the back of a victim's range. Compile-time so that
// "shipped" is a build flag: the two-build comparison is then a flag flip on one
// source tree rather than two trees that have to be kept in step.
#ifndef SCORCH_SPMM_PARTITION_DEFAULT
#  define SCORCH_SPMM_PARTITION_DEFAULT 0
#endif
// Independent nonzero streams in the exact-width narrow-k kernel, or 0 to leave those
// widths on the register-block kernel and its whole-row lane mask.
#ifndef SCORCH_NARROWK_EXACT_UNROLL
#  define SCORCH_NARROWK_EXACT_UNROLL 0
#endif
// Whether the exact-width kernel reduces its unroll on rows shorter than it. Off until
// both hosts have measured it; the harm it addresses is ARM-side and measured.
#ifndef SCORCH_NARROWK_EXACT_SHORT
#  define SCORCH_NARROWK_EXACT_SHORT 0
#endif
// Live scalar accumulators the exact-width kernel is allowed to hold. It keeps
// UNROLL*K of them, so at K=6 with UNROLL=4 that is 24, more than the 16 general
// registers x86-64 has; float32 k=6 is the worst cell in the widened grid at 0.9132
// while k=2, which holds 8, reads 1.0666. Nonzero here halves the unroll until
// UNROLL*K fits. 0 leaves the unroll at whatever the width was asked for.
#ifndef SCORCH_NARROWK_EXACT_ACCUM
#  define SCORCH_NARROWK_EXACT_ACCUM 0
#endif
// Widest k the exact-width kernel serves. Measured per width: it wins 6-8% at k=2 and
// k=3 on both dtypes and loses at k=1, 5, 6 and 7, so 3 is where the sign changes. Only
// 1..7 (float) and 1..3 (double) are instantiated and the dispatch clamps to that.
#ifndef SCORCH_NARROWK_EXACT_HI
#  define SCORCH_NARROWK_EXACT_HI 3
#endif
// Grains of work (nnz*max(k,16) over SCORCH_GRAIN_SPMM) the row partition needs before
// its bookkeeping is amortised. Its only measured regression is a short-kernel one: the
// cells where it falls more than 10% behind the shared counter sit at 19-30 microseconds
// where the cells it wins sit at 31-110, and nnz*max(k,16) separates the two better than
// any other feature in the grid, at about two grains on both dtypes. 0 leaves it out.
#ifndef SCORCH_SPMM_PARTITION_MINGRAINS
#  define SCORCH_SPMM_PARTITION_MINGRAINS 0L
#endif
// Grains of REAL arithmetic each worker must get before the row-proxy thread count
// is raised. One grain is not enough: the grain is calibrated for "is more than one
// thread worth it at all", and going from 4 workers to 18 wakes more of them, so it
// has to clear a higher bar than going from 1 to 2.
//
// 2 is where the measurement puts it, over 582 raises on two hosts and both dtypes
// (redwood i9-14900K, Apple M5), scored on whether any admitted cell came out more
// than 10% slower than the un-raised arm on the kernel timer:
//     gate                        redwood                 M5
//     no gate            212 admitted, 30 harmed   370 admitted, 91 harmed
//     1 grain / worker   162 admitted, 18 harmed   320 admitted, 76 harmed
//     2 grains / worker   64 admitted,  0 harmed   156 admitted,  0 harmed
//     4 grains / worker   48 admitted,  0 harmed   116 admitted,  0 harmed
// 2 is the smallest value with no harm on either host, so it is the one that keeps
// the most of the win: +24% (redwood) / +33% (M5) on the cells it admits. Zero
// harmed out of 64 is not luck -- same-code noise alone puts 4-7% of cells below
// 0.9 on this grid, so a merely neutral set of 64 would show three or four.
//
// It is deliberately conservative. It declines 80 redwood cells that were better
// than 1.15x, because the alternative rules that keep those (bounding on the work
// the raise moves off the critical path) each harmed one to three cells, and the
// performance convention here does not trade a regression for an average.
#ifndef SCORCH_SPMM_RAISE_GRAINS
#  define SCORCH_SPMM_RAISE_GRAINS 2L
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

// Dense-output parallel zero-fill threshold (scorch/csrc/header.h scorch_zero_dense,
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
                           int nfloor = 1,
                           long rows_per_thread = SCORCH_ROWS_PER_THREAD) {
  int hw = omp_get_num_procs();               // stable; torch mutates omp_get_max_threads
  if (rows_per_thread < 1) rows_per_thread = 1;
  long n = rows / rows_per_thread;            // row-axis parallel capacity
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

// Effective last-level cache in bytes, queried from the OS -- no hardcoded
// constant. Same sources, same SCORCH_LLC_BYTES override and same per-platform
// fallback as tiling.query_llc on the Python side, so the two layers cannot
// disagree about the machine; a test pins that they return one number. Linux: the
// largest cache level in sysfs (L3 where there is one). macOS: the P-cluster L2,
// which is the binding cache for SpMM on Apple silicon (the SLC is not exposed).
// Cached on first call.
//
// The Python selector gates on this in production. No C++ kernel currently does:
// the one that did -- a non-temporal-store gate on the wide path -- was measured
// at 0.9972 against a 1.0315 null and removed (see scorch_spmm_row_regtile). This
// stays because the selector's number has to be inspectable from a harness without
// the harness restating how it is derived.
inline long scorch_llc_bytes() {
  static const long cached = [] {
    if (const char* e = std::getenv("SCORCH_LLC_BYTES")) {
      if (*e) { long v = std::atol(e); if (v > 0) return v; }
    }
    long best = 0;
#if defined(__APPLE__)
    for (const char* key : {"hw.perflevel0.l2cachesize", "hw.l2cachesize"}) {
      int64_t v = 0; size_t len = sizeof(v);
      if (sysctlbyname(key, &v, &len, nullptr, 0) == 0 && v > 0) {
        best = (long)v; break;
      }
    }
#elif defined(__linux__)
    for (int idx = 0; idx < 10; idx++) {
      char path[128];
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
      FILE* f = std::fopen(path, "r");
      if (!f) continue;
      char buf[32] = {0};
      if (std::fgets(buf, sizeof(buf), f)) {
        long mult = 1;
        for (char* q = buf; *q; ++q) {
          if (*q == 'K') { mult = 1024; *q = 0; break; }
          if (*q == 'M') { mult = 1024 * 1024; *q = 0; break; }
          if (*q == '\n') { *q = 0; break; }
        }
        const long n = std::atol(buf) * mult;
        if (n > best) best = n;
      }
      std::fclose(f);
    }
#endif
    // Same fallback as tiling.query_llc, per platform. A different one here would
    // make the claim above false in exactly the case it matters -- the query
    // failing is when the two gates have nothing but the fallback to agree on.
#if defined(__APPLE__)
    return best > 0 ? best : (long)(16 << 20);
#else
    return best > 0 ? best : (long)(36 << 20);
#endif
  }();
  return cached;
}

// The thread count the drop-in SpMM actually runs on, given the caller's override.
//
// Extracted so there is ONE implementation. The SpMM used to compute this inline
// and a calibration harness recomputed it in Python from torch.get_num_threads(),
// which is not the same number: omp_get_num_procs() reports 32 on a 24-physical-
// core part, so the harness attributed the kernel's chunk to a thread count the
// kernel never used, and then classified cells as "the rule changed nothing" that
// it had in fact changed. A restated policy is a second thing that can be wrong,
// and it is wrong silently.
//
// override <= 0 means pure policy, which is what the standalone and panel paths
// want. Otherwise adopt the host count to avoid a pipeline team reshape, bounded
// two ways so a small product cannot regress: never past the row-parallelism
// ceiling (a 130-row product at wide k clears the work floor but cannot feed 16
// workers), and never below the policy count, so big graphs keep a higher one.
// work_true is nnz*k, the actual arithmetic. `work` is nnz*max(k,16): the floor is
// there because a row of one column still costs a whole cache line, which is the
// right measure for throttling threads on BANDWIDTH, and the wrong one for deciding
// how many threads to WAKE -- at k=1 it overstates the arithmetic sixteenfold. That
// mattered: raising the count off the row proxy on the strength of the floored
// measure made the 20-50 us cells 0.920 (float32, 40% of them more than 10% slower),
// because the extra team's ramp is a large fraction of a 30 us kernel. Callers that
// pass only one number get the old behaviour.
inline int scorch_spmm_nthreads(long work, long rows, int nthreads_override,
                                long work_true = -1, long nnz = -1) {
  if (work_true < 0) work_true = work;
  long rpt = SCORCH_SPMM_ROWS_PER_THREAD;
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: 16 reproduces the pre-change policy exactly (the raise below is then
  // unreachable), which is what the control arm needs. Compiled out of the shipped .so.
  { const char* e = std::getenv("SCORCH_SPMM_ROWS_PER_THREAD");
    if (e && *e) { long v = std::atol(e); if (v > 0) rpt = v; } }
#endif
  // The row-axis capacity. rows/SCORCH_ROWS_PER_THREAD is what ships; where a
  // minimum nonzero count per worker is configured, that requirement is stated in
  // nonzeros instead and the larger of the two wins, so this can only widen.
  long rows_axis = rows / SCORCH_ROWS_PER_THREAD;
  long nnz_per_thread = SCORCH_SPMM_NNZ_PER_THREAD;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_NNZ_PER_THREAD");
    if (e && *e) { long v = std::atol(e); if (v >= 0) nnz_per_thread = v; } }
#endif
  // Whether the nonzero statement is allowed to fire at all. Unconditionally it is a
  // null: over 2172 redwood cells it reads 1.2579 against back-stealing's 1.2670 on
  // float32 and 1.2362 against 1.2374 on float64. That null is a gain and a loss
  // cancelling. Scored against the same-code floor inside each region, it is 1.1109
  // (float32, z=3.38) and 1.1542 (float64, z=3.18) on the 42 cells with few rows and
  // very high degree, and 0.9837 (float32, z=-2.63) on the 276 cells with few rows and
  // ordinary degree. So both conditions are load-bearing: the row cap has to be leaving
  // most of the machine idle AND each row has to still have thousands of nonzeros to
  // chew. kl02 (71 rows, degree 2993) is the shape it is for; a 64-row pruned-ResNet
  // layer at degree 288 with 18432 nonzeros in total is the shape it must not catch.
  //
  // Both thresholds are read off one host and one corpus, so both are hooks and the
  // compiled-in defaults leave the rule OFF. Promote them only once the M5 and the
  // held-out large-A corpus agree.
  long ceil_maxrows = SCORCH_SPMM_CEIL_MAXROWS;
  long ceil_mindeg = SCORCH_SPMM_CEIL_MINDEG;
  bool ceil_cap_pool = SCORCH_SPMM_CEIL_CAP_POOL != 0;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_MAXROWS");
    if (e && *e) { long v = std::atol(e); if (v >= 0) ceil_maxrows = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_MINDEG");
    if (e && *e) { long v = std::atol(e); if (v >= 0) ceil_mindeg = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_CAP_POOL");
    if (e && *e) ceil_cap_pool = std::atol(e) != 0; }
#endif
  if (nnz_per_thread > 0 && nnz > 0 &&
      (ceil_maxrows <= 0 || rows <= ceil_maxrows) &&
      (ceil_mindeg <= 0 || nnz >= ceil_mindeg * rows)) {
    long by_nnz = nnz / nnz_per_thread;
    if (by_nnz > rows) by_nnz = rows;      // one worker per row is the hard ceiling
    // ... and optionally at the pool the CALLER manages rather than at the machine.
    // Inside the gate above the rule reads 1.1109/1.1542 on redwood and 0.934/0.948 on
    // the M5, and the candidate mechanism for that disagreement is here: the widened
    // count is capped by omp_get_num_procs(), which is 32 against torch's 24 on redwood
    // but 18 against torch's 6 on the M5. So the same rule widens kl02 from 4 workers to
    // 22 inside a 24-thread pool on one host and to 18 -- three times the pool, pulling
    // in twelve efficiency cores -- on the other. Capping at the override makes the ARM
    // widening 4 -> 6 and leaves the x86 widening untouched, which is the claim to test.
    if (ceil_cap_pool && nthreads_override > 0 && by_nnz > (long)nthreads_override)
      by_nnz = (long)nthreads_override;
    if (by_nnz > rows_axis) rows_axis = by_nnz;
  }
  // A/B hook: the grain the BASE path divides the work by. 150000 nonzero-units is
  // about sixty-five microseconds of single-thread work, which is a very conservative
  // bar for "is a second worker worth waking" when fork/join is two to five. It is
  // what holds the mostly-empty matrices to one thread: Pd_b is 8081 rows holding
  // 6323 nonzeros, so at k=1 its floored work is 101168 against the grain and it runs
  // single-threaded in 24 us where MKL takes 15.3. Lowering it is NOT obviously right
  // -- raising thread counts off this same floored measure by a different route made
  // the 20-50 us cells 0.920 -- so the point of the hook is to ask rather than assume.
  long grain = SCORCH_GRAIN_SPMM;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_GRAIN");
    if (e && *e) { long v = std::atol(e); if (v > 0) grain = v; } }
#endif
  // The proxy count, exactly as before when nnz_per_thread is 0 and the grain is the
  // default: rows_axis is then rows/SCORCH_ROWS_PER_THREAD and dividing it by 1
  // reproduces the old expression including its truncation.
  int nthreads = scorch_nthreads(work, rows_axis, grain, 1, 1);
  // Then raise it where the ROW proxy, not the work, is what bound it -- a 64-row
  // pruned-ResNet layer at k=512 is 62 grains of arithmetic held to 4 workers -- but
  // only as far as the real arithmetic supports: one grain per worker. Both bounds
  // are needed. rows/rpt alone wakes 31 threads for a k=1 product whose whole
  // kernel is 30 us; work_true/grain alone would ignore that a worker still needs
  // rows to work on.
  if (rpt < SCORCH_ROWS_PER_THREAD) {
    long cand = rows / rpt;
    // The bound is REAL arithmetic on purpose (see the constant's comment). The
    // hook prices the other reading: nnz*max(k,16) is a TIME proxy, and how many
    // workers a product can feed is a question about time, not about multiply-adds.
    // It is what strands the high-degree narrow-k class -- kl02 at k=2 has 425072
    // multiply-adds, one grain, and 3400576 units of the floored measure, eleven.
    long raise_work = work_true;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_RAISE_ON_FLOORED");
      if (e && *e && std::atol(e) != 0) raise_work = work; }
#endif
    const long by_true = raise_work / (SCORCH_SPMM_RAISE_GRAINS * SCORCH_GRAIN_SPMM);
    if (cand > by_true) cand = by_true;
    const long hw = (long)omp_get_num_procs();
    if (cand > hw) cand = hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }
  // Which work measure gates the composition adoption. `work` is nnz*max(k,16) --
  // the k term floored at a cache line, which is right for throttling a
  // bandwidth-bound product and overstates a k=1 product SIXTEENFOLD. Gating the
  // adoption on it means a product with 12625 nonzero-units of real arithmetic
  // reads 202000 against a 150000 grain and gets the whole host team.
  //
  // Measured on the M5 over the 40 matrices where the home-range partition was worst,
  // forcing ONE thread beat the adopted count on 31 of 40 cells at k=1, 28 of 40 at
  // k=2 and 29 of 40 at k=8, geomean 1.197 / 1.156 / 1.142 in favour of one thread;
  // at k=64, where the floor does not bite, the adopted count is right and wins
  // 1.68x. This is the same defect that has already been fixed twice elsewhere --
  // the raise gate above reads work_true for exactly this reason.
  long gate_work = work;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_OVERRIDE_GATE_TRUE");
    if (e && *e && std::atol(e) != 0) gate_work = work_true; }
#endif
  if (nthreads_override > 0 && gate_work >= grain) {
    // Deliberately the 16-rows-per-worker ceiling, not rpt. This is the composition
    // path -- adopt the host team so a pipeline does not reshape at every op
    // boundary -- and widening it too would raise the count on the very k=1 cells
    // the gate above just declined to raise, by a different route.
    // Deliberately rows/SCORCH_ROWS_PER_THREAD and NOT the widened rows_axis. The
    // base path above pairs its ceiling with a work term, so widening it there cannot
    // over-thread a small product; this path has no work term at all, and sharing the
    // widened ceiling here is what made the nonzero-expressed ceiling fail on ARM --
    // 64-row pruned ResNet layers went from 4 workers to 6 on a 20-microsecond kernel
    // and ran 1.5-2x slower, 6.4% of cells more than 10% slower against a 1.4% floor.
    // The ceiling was not choosing 6 workers for those; this line was.
    const long by_rows = rows / SCORCH_ROWS_PER_THREAD;
    long cand = (long)nthreads_override < by_rows ? (long)nthreads_override : by_rows;
    // Graded adoption: cap the adopted count so each worker gets a grain of REAL
    // arithmetic, rather than switching wholesale between the host count and the
    // policy's. Never below one, so this can only lower an adopted count and never
    // decline the adoption -- the pipeline still gets one shared team.
    long adopt_grain = SCORCH_SPMM_ADOPT_GRAIN;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_ADOPT_GRAIN");
      if (e && *e) { long v = std::atol(e); if (v >= 0) adopt_grain = v; } }
#endif
    if (adopt_grain > 0) {
      long by_real = work_true / adopt_grain;
      if (by_real < 1) by_real = 1;
      if (cand > by_real) cand = by_real;
    }
    const long hw = (long)omp_get_num_procs();   // never oversubscribe the box
    if (cand > hw) cand = hw;
    if (cand > (long)nthreads) nthreads = (int)cand;
  }
#ifdef SCORCH_TUNE_HOOKS
  // A/B hook: force the FINAL count, after both the policy and the composition
  // adoption. SCORCH_TUNE_THREADS cannot do this -- it sets the policy count and
  // the adoption path then raises it straight back -- and asking "is this shape
  // better on fewer threads" needs the answer to survive to the launch. Scorch
  // against itself, so it does not inherit the kernel-timer-vs-whole-call
  // asymmetry that the MKL comparison has. Compiled out of the shipped .so.
  { const char* e = std::getenv("SCORCH_SPMM_NT_FORCE");
    if (e && *e) { long v = std::atol(e);
      if (v > 0) { const long hw2 = (long)omp_get_num_procs();
                   nthreads = (int)(v > hw2 ? hw2 : v); } } }
#endif
  return nthreads;
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
