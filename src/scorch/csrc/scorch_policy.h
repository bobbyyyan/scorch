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
// Independent accumulator chains in the prebuilt SpMV row loop (kernels.h). 1 is what ships
// and what the disassembly shows the compiler produces on its own; see scorch_spmv_row for why
// the compiler will not widen it. Raise only once a grid on both hosts says so.
#ifndef SCORCH_SPMV_ACCUM
#  define SCORCH_SPMV_ACCUM 1
#endif

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

// A ceiling on the FINAL resolved worker count, applied after both the policy count and
// the composition adoption. 0 is off; a positive value is that many workers; -1 means
// scorch_pcore_count(), the host's performance-core count.
//
// Every number below is SYNTHESISED from SCORCH_TUNE_THREADS arms, which force the base
// count that scorch_nthreads returns and not the final one -- the row-proxy raise and the
// composition adoption can both put it back, ceilinged at omp_get_num_procs(). So they
// size the effect and identify the value; they are not a measurement of this knob. This
// knob, applied after both raises, is the first instrument here whose value is the count
// the kernel actually launches on.
//
// This is a cap and not a force, and the distinction is most of the effect. Forcing a
// count also RAISES it on the cells that resolved below the value, and on the M5 that is
// ruinous -- over 676 cells, forcing six threads reads 0.726 against the resolved default
// because 216 of those cells resolve to a single thread and six is pure oversubscription
// there. A cap leaves every one of them untouched.
//
// Why the machine needs one at all: both hosts resolve above the pool the surrounding
// framework advertises. On redwood 48 of 231 cells land on 32, which is
// omp_get_num_procs(), inside a 24-thread torch pool; on the M5 128 of 676 land above 6,
// as high as 18, inside a pool of 6. Scored as a cap -- unchanged where it cannot fire,
// the ladder's tC arm where it can -- the P-core count reads, against A/A floors of
// 1.0021/0.9934 (x86 f32), 0.9957/1.0109 (x86 f64) and 1.0003/1.0004 (ARM f32):
//
//     x86 f32, cap 8   191/231 cells fire   1.0865 corpus   1.1055 firing    1 matrix -5%
//     x86 f64, cap 8   191/231 cells fire   1.0700 corpus   1.0852 firing    2 matrices -5%
//     ARM f32, cap 6   128/676 cells fire   1.0486 corpus   1.2850 firing    2 matrices -5%
//
// The P-core count beats the more obvious "never exceed the caller's pool" form: on the
// M5 they are the same number, but on redwood the pool is 24 and capping there is worth
// 1.0470 against 1.0865.
//
// OFF by default and every number above is synthesised from force arms, which is not the
// same experiment: lowering the count also changes what
// scorch_spmm_partition_mode derives from it, and cap 16 costing twelve x86 matrices more
// than 5% where 8 costs one and 24 costs none is a non-monotonicity that parallelism alone
// does not explain. Both corpora also stop at k=8, and on ARM the shipped E-core recruit
// deliberately launches twice the pool for wide, bandwidth-bound products. Promote this
// only once a compiled-in three-build reproduces it and a width sweep says where it must
// stop firing.
#ifndef SCORCH_SPMM_NT_CAP
#  define SCORCH_SPMM_NT_CAP 0L
#endif

// Whether the final cap declines to undo what the nonzero-expressed row ceiling deliberately
// asked for. 1 respects it, 0 lets the cap win.
//
// The two rules correct OPPOSITE errors on populations that do not overlap. The ceiling raises
// the count where the row proxy understated it -- kl02, 71 rows of degree 2993, from 4 workers
// to 22 -- and the cap lowers it where the floored work measure overstated it. On the x86 width
// board they between them cover the whole loser set: 71 of 75 cells for the cap and the
// remaining 4, which are kl02 at k=2,4,8 and nw14 at k=2, for the ceiling.
//
// The cap is applied last on purpose, so the composition adoption cannot raise the count back
// after it, and that placement means it would otherwise pull the ceiling's 22 down to 8. There is
// no measurement saying 8 is acceptable for kl02: the ceiling's 1.1109 (float32, z=3.38) and
// 1.1542 (float64, z=3.18) were both taken at its widened count. Hence the floor, and hence the
// knob -- the alternative composition has to be measurable, not assumed.
#ifndef SCORCH_SPMM_NT_CAP_FLOOR_CEIL
#  define SCORCH_SPMM_NT_CAP_FLOOR_CEIL 1
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
// omp_get_num_procs(). This was the candidate fix for the measured x86/ARM disagreement, and it
// is now the confirmed one -- see the measurement recorded just above the constant.
// Pool FLOOR on the row ceiling: the rule applies only where the caller's pool is at
// least this wide. 0 disables the floor, which is today's behaviour.
//
// This is the mirror of SCORCH_SPMM_PARTITION_GATE_MAXTHREADS. Compiled in, the ceiling
// reads 1.3066 (f32) / 1.4011 (f64) inside its gate on a 24-thread x86 host with the
// harmed tail below the A/A floor, and on a 6-thread ARM host no gain in gate (0.9887 /
// 0.9753) with about 2% off outside it on float64 (per-matrix z -2.15 over 260 matrices).
// The rule takes a few-row product from 4 workers to 22 on the first host and from 4 to 6
// on the second -- there are only six to have -- so the sign is set by how much machine
// the shape was failing to use, not by the shape and not by the ISA. A floor states that
// directly, and on a host below it the whole block is unreachable, so the ARM cost goes
// away as dead code rather than as a tuned constant.
// State the row condition as the MECHANISM instead of as a row count: the rule applies
// where the row proxy is below the width available, rows/SCORCH_ROWS_PER_THREAD < pool,
// which on a 24-thread host is rows < 384 against SCORCH_SPMM_CEIL_MAXROWS's 128.
//
// This form was built, measured and removed once as a null: over the 130 cells (NINE
// matrices) where the two forms differ it read 1.0053, and its pool-capped variant 1.0370,
// against a 1.0015 floor -- z of 0.15 and 1.27 aggregated per matrix. It is back because
// the production-path scoreboard says the three worst ratios in the whole corpus are
// exactly there: rn50's 256-row block at 0.638 gets 16 workers of 24, and kl02 and nw14 at
// 71 and 73 rows get 4 of 24. A 9-matrix null at low power is not evidence against a
// mechanism whose target has since been identified by a different measurement.
#ifndef SCORCH_SPMM_CEIL_ROWBIND
#  define SCORCH_SPMM_CEIL_ROWBIND 0
#endif

#ifndef SCORCH_SPMM_CEIL_MINTHREADS
#  define SCORCH_SPMM_CEIL_MINTHREADS 0L
#endif

// MEASURED on both hosts, 2026-08-28, so this is no longer "off until both hosts have run it".
// M5, 97 matrices stratified on rows x degree, kernel timer, two replicates per dtype, six arms
// each naming the same five knobs: inside the shipped gate the UNCAPPED rule reads 0.9272 and
// 0.9311 on float32 (z -6.4, -6.3) and 0.9822 / 0.9779 on float64, and the CAPPED rule reads
// 1.0025 / 1.0041 and 1.0000 / 1.0001 on the same matrices in the same runs. The cap is therefore
// the entire ARM cost of this rule, exactly by the mechanism this comment predicted: num_procs is
// 18 against a pool of 6 here, so uncapped the rule recruits twelve efficiency cores, and capped
// it widens four workers to six. On x86 the cap costs nothing (1.1059 / 1.1978 against
// 1.1125 / 1.1926).
//
// Default flipped to 1 on that evidence. It changes no emitted code today: `ceil_cap_pool` is
// read only inside `if (nnz_per_thread > 0 && ...)`, and SCORCH_SPMM_NNZ_PER_THREAD ships at 0,
// so the whole block folds away in a release build -- verified by compiling an x86_64 -O3 object
// before and after. What it changes is that a build which turns the ceiling on gets the form that
// is neutral on ARM, rather than the form that costs it 7%.
//
// The pool FLOOR below is what this makes unnecessary: an arm setting MINTHREADS=12 against a
// pool of 6 read the A/A floor as predicted, confirming the floor reads the caller's pool, but
// with the capped rule already neutral there is nothing left for a floor to protect.
#ifndef SCORCH_SPMM_CEIL_CAP_POOL
#  define SCORCH_SPMM_CEIL_CAP_POOL 1
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
#  define SCORCH_SPMM_PARTITION_DEFAULT 3
#endif
// Independent nonzero streams in the exact-width narrow-k kernel, or 0 to leave those
// widths on the register-block kernel and its whole-row lane mask.
#ifndef SCORCH_NARROWK_EXACT_UNROLL
#  define SCORCH_NARROWK_EXACT_UNROLL 4
#endif
// Whether the exact-width kernel reduces its unroll on rows shorter than it. Off until
// both hosts have measured it; the harm it addresses is ARM-side and measured.
// Whether the exact-width kernel's unroll is chosen once per call from the mean row
// length instead of once per ROW from that row's length. The per-row clamp
// (SCORCH_NARROWK_EXACT_SHORT) costs 1.0% pooled on x86 float32 -- 1.0257 -> 1.0157 over
// 2880 padded cells -- and the reason is not the compare: the switch the clamp feeds stops
// being predictable once neighbouring rows take different unrolls. A per-call decision from
// nnz/rows has no per-row branch at all and still gives a degree-1.6 graph an unroll of 1,
// which is the shape the ARM tail is made of. 0 leaves the configured unroll alone.
// Minimum mean degree (nonzeros per row) for the exact-width narrow-k kernel. 0 disables
// the floor. At 1 the kernel is refused on any matrix holding fewer nonzeros than rows,
// where its per-row setup has nothing to amortise -- see the measurement in spmm.h.
#ifndef SCORCH_NARROWK_EXACT_MINDEG
#  define SCORCH_NARROWK_EXACT_MINDEG 0L
#endif

// Width 1 in the exact-width narrow-k band, and the mean degree it is admitted at.
//
// The band {2,3} replaces a register-block tile kernel whose mask wastes 6 lanes of 8 under
// AVX2 and 2 of 4 under NEON. Width 1 replaces something else -- a loop carrying a single
// accumulator, which no mask width describes -- so it is a different trade and gets its own
// admission rather than inheriting the band's floor. On ARM float32, lowering the band floor
// to 1 is worth 1.0243, 1.0460 and 1.1174 at mean degree 8-64, 64-256 and >=256, and costs
// 6.8% below degree 8 (z-3.1 over 32 matrices), so the win is conditional on degree.
//
// The conditioning cannot be expressed with SCORCH_NARROWK_EXACT_MINDEG, which gates the whole
// band: widths 2 and 3 measure 0.90-0.93 under every floor value tried, in every degree band,
// on two independent runs. Hence a second constant rather than a reused one.
//
// 0 disables width 1 entirely, which is what ships.
// Multi-row register blocking: how many consecutive output rows one kernel call takes, and the
// nonzero count below which it declines.
//
// The kernel exists (spmm.h, AVX2+FMA) but its dispatch has only ever been inside
// SCORCH_TUNE_HOOKS, so a release binary does not contain it. Measured on redwood, 302 matrices,
// 1208 cells per dtype, kernel timer, against an A/A floor of 0.995-1.009:
//
//   ROWS=2         float32                    float64
//   nnz 200k-1M    1.1043 (k=4) 1.1171 (k=8)  1.1365 (k=4) 1.0932 (k=8)
//   nnz >1M        1.1332 (k=4) 1.1434 (k=8)  1.1232 (k=4) 1.0862 (k=8)
//   degree 8-32    1.1253 (k=4) 1.1256 (k=8)  1.1243 (k=4) 1.0890 (k=8)
//   degree <8      0.9882 (k=4, z-0.4)        0.9922 (k=4, z-0.5)
//
// Every negative band for ROWS=2 has |z| < 1, i.e. inside the same-code floor, so on this corpus
// ROWS=2 is neutral-or-better everywhere and wins 9-14% where B rows get re-read: mid-to-large
// nonzero counts at moderate degree, which is what amortising the B loads across two output rows
// is for. The cells behind MKL fall 96 -> 59 at float32 k=4, 17 -> 6 at k=8, 38 -> 31 and 11 -> 6
// on float64 -- the largest reduction any single lever has produced.
//
// ROWS=4 is NOT the same answer: it loses at k=4 (0.9730 float32, z-2.0) and only beats ROWS=2 at
// float32 k=16 (1.1604 against 1.0983). So the row count is not "more is better" and 4 is not the
// default even where it is instantiated.
//
// 0 disables the kernel, which is what ships until a compiled-in three-build on both hosts agrees
// with the hooked grid above. MINNNZ 0 means no nonzero gate; it exists because the wins concentrate
// above about 200k nonzeros, so if the three-build finds a small deficit on tiny matrices the fix
// is a gate that already exists rather than a new mechanism.
#ifndef SCORCH_SPMM_MULTIROW_ROWS
#  define SCORCH_SPMM_MULTIROW_ROWS 0
#endif
#ifndef SCORCH_SPMM_MULTIROW_MINNNZ
#  define SCORCH_SPMM_MULTIROW_MINNNZ 0L
#endif

// The deep register-block kernel: the vector counts it serves, its depth, and its prefetch.
//
// It exists (spmm.h, guarded on AVX2+FMA) but has never been reachable outside a
// SCORCH_TUNE_HOOKS build, so a release binary does not contain the dispatch at all. Measured on
// 302 matrices with the kernel timer, against the shipped register block:
//
//   nvec   float32 (8 lanes)      float64 (4 lanes)
//   1      0.8034 (k=4), 0.8124 (k=8)   0.8392 (k=4)
//   2      **1.0396 (k=16)**            **1.0448 (k=8)**
//   4      unmeasured                   1.0154 (k=16)
//
// The two dtypes win at widths a factor of two apart and at the SAME vector count, which is why
// the range below is stated in nvec and not in k -- a rule in k would have been right on one
// dtype and wrong on the other by exactly the lane ratio. At nvec=1 the kernel displaces a
// specialised narrow path (exact-width, or the half-vector block at k=4) and loses 16-20%; at
// nvec=2 it displaces the generic register block and wins.
//
// Decomposed, the gain is the prefetch's ABSENCE: at k=16 float32, depth alone reads 0.9633 and
// turning the deep kernel's prefetch off is worth 1.0792 on top of that.
//
// SCORCH_NARROWK_DEEP_UNROLL 0 disables the kernel, which is what ships. NVEC_HI 0 means "no
// vector-count restriction", so a hooks arm that sets only the depth behaves exactly as it did
// before this range existed.
#ifndef SCORCH_NARROWK_DEEP_UNROLL
#  define SCORCH_NARROWK_DEEP_UNROLL 0
#endif
#ifndef SCORCH_NARROWK_DEEP_PF
#  define SCORCH_NARROWK_DEEP_PF 1
#endif
#ifndef SCORCH_NARROWK_DEEP_NVEC_LO
#  define SCORCH_NARROWK_DEEP_NVEC_LO 0
#endif
#ifndef SCORCH_NARROWK_DEEP_NVEC_HI
#  define SCORCH_NARROWK_DEEP_NVEC_HI 0
#endif

#ifndef SCORCH_NARROWK_EXACT_K1
#  define SCORCH_NARROWK_EXACT_K1 0
#endif

// Minimum mean degree (nonzeros per row) for width 1, when the above is enabled.
// 0 admits it at any degree, which the ARM grid says is a loss below degree 8.
#ifndef SCORCH_NARROWK_EXACT_K1_MINDEG
#  define SCORCH_NARROWK_EXACT_K1_MINDEG 0L
#endif

#ifndef SCORCH_NARROWK_EXACT_DEGUNROLL
#  define SCORCH_NARROWK_EXACT_DEGUNROLL 0
#endif
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
// Serve a width that is exactly the half-vector -- four floats, two doubles -- with 128-bit
// registers instead of a masked 256-bit register. At k=4 float32 the register-block kernel
// runs a masked load per NONZERO over 4 lanes of 8; 128-bit needs no mask and wastes no FMA
// width. This is the width the per-width sweep skipped and where the k=4 float32 fit shows a
// per-nonzero deficit against MKL of 8-23% whichever shapes go into it.
// 0 = off (masked 256-bit, as before). 1 = the exact half-vector width only. 2 = also the
// widths below it that the exact-width scalar kernel does not already claim.
// Deliberately NOT given a combined default. A single SCORCH_SPMM_HALFVEC whose zero value fell
// through to the per-dtype defaults below would be a knob that cannot express "off", so the two
// per-dtype macros are the only compile-time control; -DSCORCH_SPMM_HALFVEC_F32=0 turns it off.
// The environment hook of the same name still overrides both at runtime, and it CAN say 0
// because that path tests whether the variable is set rather than whether it is nonzero.
//
// Per-dtype defaults, because the measurement splits by dtype and one number cannot carry both.
// 128-bit registers are the right shape for a four-lane float row and the wrong shape for a
// two-lane double row, where the halved register width costs more than the halved mask waste
// saves. Over 1510 cells per dtype on the 302-matrix corpus, at the width where the kernel
// fires:
//
//   float32, k=4:  1.1008 (z +14.6) against a 0.9945 same-value control; by group, 1.2143 on
//                  the L2-band family (z +40.6), 1.1463 on SuiteSparse, 1.1305 on short rows,
//                  1.0113 at worst. MKL kernel parity 1.2880 -> 1.4179 and cells below MKL
//                  125/302 -> 70. Taking the widths BELOW the exact one as well adds nothing
//                  (1.0997), so 1 and not 2.
//   float64, k=2:  0.9646 (z -10.7), negative in seven groups of eight, and it turns a column
//                  with 0 of 302 cells below MKL into 21. Stays off.
//
// The kernel is inside the AVX2 block, so both values are inert on ARM by construction.
#ifndef SCORCH_SPMM_HALFVEC_F32
#  define SCORCH_SPMM_HALFVEC_F32 1
#endif
#ifndef SCORCH_SPMM_HALFVEC_F64
#  define SCORCH_SPMM_HALFVEC_F64 0
#endif
#ifndef SCORCH_NARROWK_EXACT_HI
#  define SCORCH_NARROWK_EXACT_HI 3
#endif
// Grains of work (nnz*max(k,16) over SCORCH_GRAIN_SPMM) the row partition needs before
// its bookkeeping is amortised. Its only measured regression is a short-kernel one: the
// cells where it falls more than 10% behind the shared counter sit at 19-30 microseconds
// where the cells it wins sit at 31-110, and nnz*max(k,16) separates the two better than
// any other feature in the grid, at about two grains on both dtypes. 0 leaves it out.
#ifndef SCORCH_SPMM_PARTITION_MINGRAINS
#  define SCORCH_SPMM_PARTITION_MINGRAINS 2L
#endif
// Size of the CALLER'S POOL at or below which that work gate is allowed to fire. It exists
// because the gate's sign depends on the pool, and this makes the dependence explicit
// instead of leaving it to a host-specific default.
//
// The pool, not the resolved worker count. The first version of this compared the resolved
// count, and the ARM grid falsified it: the gated and gated-with-a-ceiling-of-16 arms must
// read identically on a six-thread host, and they differed by 0.7% (1.0154 against 1.0087
// on float32). The resolved count is raised per shape and is bounded by
// omp_get_num_procs(), which is 18 there, so it straddles 16 and the ceiling blocked the
// gate on exactly the shapes whose count had been raised. The pool the surrounding pipeline
// runs is 6 there and 24 on redwood -- a factor of four apart, shape-independent, and it is
// the quantity the contention argument is actually about.
//
// The partition buys A's inter-call L2 residency and pays a setup plus a claim per chunk;
// the shared counter buys perfect load balance and pays contention on one atomic line.
// Contention scales with the number of workers, so which cost binds is a question about
// the pool: on redwood's 24 the counter is the bottleneck and the partition wins even on
// tiny products (no-partition / partition = 0.785 pooled, 65% of cells more than 10%
// behind), while on the M5's 12 it is not, and on the 70 matrices where the partition
// regresses there -- 100 KB of A that was never leaving L2, so no residency to win -- the
// counter wins 1.0303 and every home-range variant loses (no stealing 0.9826,
// front-stealing 0.9682).
//
// A ceiling between the two pools makes the gate PROVABLY INERT on the larger host: it
// cannot fire at a pool of 24, so the x86 measurement stands unchanged by construction
// rather than by a second grid agreeing. 0 disables the ceiling, which lets the gate fire
// at any pool (that is the configuration x86 measured at 4-9% cost, and it is not what
// ships).
#ifndef SCORCH_SPMM_PARTITION_GATE_MAXTHREADS
#  define SCORCH_SPMM_PARTITION_GATE_MAXTHREADS 16
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
// The number of PERFORMANCE cores this host has, queried from the OS and cached on first call,
// the same shape as scorch_llc_bytes below and for the same reason: a harness that wants to know
// what the thread rule is working with has to be able to ask, rather than restate the derivation
// and drift from it.
//
// Nothing reads this yet. It exists because chain46 made the resolved thread count the largest
// single error on the scoreboard -- forcing eight threads took the cells behind MKL from 78/231 to
// 15/231 on float32 and 52/231 to 15/231 on float64, with the optimum flat at a median of 8 in
// every row band above 128 rows while the rule's median ran 9, 13, then 32 -- and eight is exactly
// this host's P-core count. That is a hypothesis about the mechanism, not a decided rule; the ARM
// host has previously wanted its E-cores RECRUITED for a bandwidth-bound kernel, so the two may
// disagree. Either way the count has to be available and inspectable before a rule can be written
// in terms of it.
//
// How it is derived, per platform:
//   Linux -- count physical cores whose thread_siblings_list names more than one CPU. On Intel
//     hybrid parts the P-cores carry SMT and the E-cores do not, so this separates them without
//     reading model numbers or frequencies. On a uniform SMT part every core qualifies and the
//     answer is the physical core count, which is the right answer there.
//   macOS -- hw.perflevel0.physicalcpu, which names the performance cluster directly.
// Both fall back to the physical core count, then to omp_get_num_procs(), so a host whose topology
// cannot be read gets today's bound rather than a wrong smaller one.
inline int scorch_pcore_count() {
  static const int cached = [] {
    if (const char* e = std::getenv("SCORCH_PCORES")) {
      if (*e) { long v = std::atol(e); if (v > 0) return (int)v; }
    }
    int best = 0;
#if defined(__APPLE__)
    // A plain array, not a braced list: a range-for over {...} needs <initializer_list>, and this
    // header deliberately includes almost nothing.
    static const char* const keys[] = {"hw.perflevel0.physicalcpu", "hw.physicalcpu"};
    for (const char* key : keys) {
      int32_t v = 0; size_t len = sizeof(v);
      if (sysctlbyname(key, &v, &len, nullptr, 0) == 0 && v > 0) { best = (int)v; break; }
    }
#elif defined(__linux__)
    // One entry per physical core, keyed by the lowest CPU id in its sibling list, so a core is
    // counted once however many siblings it has.
    int smt_cores = 0, all_cores = 0;
    for (int cpu = 0; cpu < 4096; cpu++) {
      char path[160];
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
      FILE* f = std::fopen(path, "r");
      if (!f) { if (cpu > 0) break; else continue; }
      char buf[256] = {0};
      const bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
      std::fclose(f);
      if (!got) continue;
      // The list is this CPU's siblings; count the core only from its first sibling.
      int first = -1, n = 0;
      for (char* q = buf; *q;) {
        while (*q == ',' || *q == ' ' || *q == '\n') q++;
        if (!*q) break;
        const int id = std::atoi(q);
        // A range "a-b" counts b-a+1 siblings.
        char* dash = q;
        while (*dash && *dash != ',' && *dash != '\n' && *dash != '-') dash++;
        if (*dash == '-') n += std::atoi(dash + 1) - id + 1;
        else n += 1;
        if (first < 0) first = id;
        while (*q && *q != ',' && *q != '\n') q++;
      }
      if (first != cpu) continue;          // not this core's first sibling
      all_cores++;
      if (n > 1) smt_cores++;
    }
    // Hybrid: only some cores carry SMT, and those are the performance ones. Uniform: all or none
    // do, and the physical core count is the answer.
    best = (smt_cores > 0 && smt_cores < all_cores) ? smt_cores : all_cores;
#endif
    if (best > 0) return best;
    const int hw = omp_get_num_procs();
    return hw > 0 ? hw : 1;
  }();
  return cached;
}

inline long scorch_llc_bytes() {
  static const long cached = [] {
    if (const char* e = std::getenv("SCORCH_LLC_BYTES")) {
      if (*e) { long v = std::atol(e); if (v > 0) return v; }
    }
    long best = 0;
#if defined(__APPLE__)
    // A plain array, not a braced list, for the reason given in scorch_pcore_count: a
    // range-for over {...} needs <initializer_list>, which this header does not include
    // and only ever got transitively from whatever torch header preceded it.
    static const char* const keys[] = {"hw.perflevel0.l2cachesize", "hw.l2cachesize"};
    for (const char* key : keys) {
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

// One pool read serving both of the row ceiling's pool-conditioned tests: the floor (the
// rule only applies on a wide enough pool) and the row-bind form (the rule applies where
// the row proxy is below the width available). Reading it once, and only when one of the
// two is enabled, is deliberate -- a pool read hoisted above the cheap tests is how an
// earlier version of this rule charged every call for a branch that could not fire.
//
// Deliberately the caller's POOL and not omp_get_num_procs(): num_procs is 18 on a
// 6-thread ARM host, which straddles any floor set between this project's two machines.
inline bool ceil_pool_ok(int nthreads_override, long minthreads, bool rowbind, long rows) {
  const long pool = nthreads_override > 0 ? (long)nthreads_override
                                          : (long)omp_get_max_threads();
  if (minthreads > 0 && pool < minthreads) return false;
  if (rowbind && rows / SCORCH_ROWS_PER_THREAD >= pool) return false;
  return true;
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
  //
  // Re-measured with equal environment-variable counts across arms, which the first
  // measurement did not have: over 2660 redwood cells the rule reads 1.1125 (float32,
  // z=3.38) and 1.1926 (float64, z=5.31) inside this gate against floors of 1.0180 and
  // 0.9956, and the cells below MKL there fall 53 -> 47 and 48 -> 40. The earlier null
  // was the arm being charged for naming two extra knobs. Capping at the caller's pool
  // costs x86 nothing (1.1059 / 1.1978), so if that is what ARM needs it is free here.
  //
  // What a HOOKED grid cannot decide is whether the rule is inert where it cannot fire.
  // Outside the gate all four variants read 0.9863-0.9962 with z from -6 to -19, ordered
  // by how many variables each arm sets that this function also looks up -- one extra
  // successful getenv and one atol per call. Padding the environment equalises the
  // number of NAMES, not the number of LOOKUPS, so the question needs a compiled-in
  // three-build (rw_stage16.sh / an_ceil3.py) and not another arm.
  long ceil_maxrows = SCORCH_SPMM_CEIL_MAXROWS;
  long ceil_mindeg = SCORCH_SPMM_CEIL_MINDEG;
  bool ceil_cap_pool = SCORCH_SPMM_CEIL_CAP_POOL != 0;
  long ceil_minthreads = SCORCH_SPMM_CEIL_MINTHREADS;
  bool ceil_rowbind = SCORCH_SPMM_CEIL_ROWBIND != 0;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_MINTHREADS");
    if (e && *e) { long v = std::atol(e); if (v >= 0) ceil_minthreads = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_ROWBIND");
    if (e && *e) ceil_rowbind = std::atol(e) != 0; }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_MAXROWS");
    if (e && *e) { long v = std::atol(e); if (v >= 0) ceil_maxrows = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_MINDEG");
    if (e && *e) { long v = std::atol(e); if (v >= 0) ceil_mindeg = v; } }
  { const char* e = std::getenv("SCORCH_SPMM_CEIL_CAP_POOL");
    if (e && *e) ceil_cap_pool = std::atol(e) != 0; }
#endif
  // The row condition stays a row count. Stating it as the mechanism instead -- the row
  // proxy is below the width available, rows/SCORCH_ROWS_PER_THREAD < pool, which is
  // rows < 384 on a 24-thread host against the constant's 128 -- was built and measured
  // and pays nothing: over the region where the two forms differ (129..383 rows at
  // degree >= 192, 130 cells but only NINE matrices) the row-bind form reads 1.0053 and
  // its pool-capped variant 1.0370 against a 1.0015 floor, z of 0.15 and 1.27 aggregated
  // per matrix. At cell level the second of those reads z=3.15, which is what five widths
  // of the same nine matrices look like when they are counted as independent.
  // The floor reads the caller's POOL, not the resolved count and not
  // omp_get_num_procs(): the resolved count is what this rule is about to change, and
  // num_procs is 18 on a 6-thread ARM host, which straddles any floor between the two
  // machines. That distinction already cost one grid -- see the work gate's ceiling.
  //
  // It is evaluated LAST and only when a floor is set, so that with the ceiling off --
  // which is the default -- this costs nothing and no omp call is made. A pool read
  // hoisted above the cheap tests is how an earlier version of this rule charged every
  // call for a branch that could not fire.
  // The row-bind form replaces the row-count test rather than adding to it, and the pool
  // test is evaluated last and only when something needs it, so with both off -- the
  // default -- this reads exactly as it did before and makes no omp call.
  const bool ceil_needs_pool = ceil_rowbind || ceil_minthreads > 0;
#if defined(SCORCH_TUNE_HOOKS) || SCORCH_SPMM_NT_CAP != 0
  long ceil_request = 0;   // see SCORCH_SPMM_NT_CAP_FLOOR_CEIL
#endif
  if (nnz_per_thread > 0 && nnz > 0 &&
      (ceil_rowbind || ceil_maxrows <= 0 || rows <= ceil_maxrows) &&
      (ceil_mindeg <= 0 || nnz >= ceil_mindeg * rows) &&
      (!ceil_needs_pool ||
       ceil_pool_ok(nthreads_override, ceil_minthreads, ceil_rowbind, rows))) {
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
#if defined(SCORCH_TUNE_HOOKS) || SCORCH_SPMM_NT_CAP != 0
    // Remembered so the final cap can decline to undo it. Tracked only where a cap can be
    // live, so a build with the cap off pays neither the store nor the stack slot.
    ceil_request = by_nnz;
#endif
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
  // WHICH work measure bounds the base count. `work` is nnz*max(k,16): a time proxy that
  // floors the k term at a cache line, which is right for throttling a bandwidth-bound product
  // and overstates a k=1 product SIXTEENFOLD. At k=1 the proxy therefore never binds, and
  // rows/SCORCH_ROWS_PER_THREAD alone sets the count -- a 256-row product with 294912 nonzeros
  // gets 16 workers for roughly 4 us of arithmetic, and measures 30.0 us of kernel time against
  // MKL's 22.6, which is 7.2 cycles a nonzero per thread against a bound under one. The raise
  // gate below already reads work_true for exactly this reason; the same correction on the base
  // bound has never been priced. Off by default, and `work` is what ships until a grid on both
  // hosts says otherwise.
  long base_work = work;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_BASE_WORK_TRUE");
    if (e && *e && std::atol(e) != 0) base_work = work_true; }
#endif
  int nthreads = scorch_nthreads(base_work, rows_axis, grain, 1, 1);
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
  // The final ceiling (see SCORCH_SPMM_NT_CAP). Deliberately last: the adoption branch
  // above is what raises the count to the pool or past it, so a ceiling applied before it
  // would be undone. Deliberately a cap: it can only lower the count, never raise one,
  // which is what makes it inert on the cells where forcing the same value is ruinous.
  //
  // The whole block is compiled out when the constant is 0 and the hooks are off. That is
  // not cosmetic -- leaving a dead branch and its locals in place once added ten x86
  // instructions and reshuffled every stack slot in this function, and marking them
  // constexpr did not help.
#if defined(SCORCH_TUNE_HOOKS) || SCORCH_SPMM_NT_CAP != 0
  {
    long nt_cap = SCORCH_SPMM_NT_CAP;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_NT_CAP");
      if (e && *e) nt_cap = std::atol(e); }
#endif
    if (nt_cap < 0) nt_cap = (long)scorch_pcore_count();
    bool floor_ceil = SCORCH_SPMM_NT_CAP_FLOOR_CEIL != 0;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_NT_CAP_FLOOR_CEIL");
      if (e && *e) floor_ceil = std::atol(e) != 0; }
#endif
    // Never below what the row ceiling asked for, when it asked for more. This can only ever
    // RAISE the effective cap, and only on cells where the ceiling fired -- so with the ceiling
    // off, which is the default, ceil_request is 0 and this is exactly the plain cap.
    if (floor_ceil && ceil_request > nt_cap) nt_cap = ceil_request;
    if (nt_cap > 0 && (long)nthreads > nt_cap) nthreads = (int)nt_cap;
  }
#endif
  return nthreads;
}

// The row-handout mode the SpMM will actually run in, for one call's shape.
//
// This is the whole rule, in one place, because the alternative is a harness that
// restates it: the offline threshold sweep for the work gate below scored a 3.7% gain
// on the ARM tail corpus against the 1.35% the machine measured, and the gap was
// entirely cells the solo gate had already put in mode 0 -- where every arm runs the
// same code and the sweep was fitting the difference between two noise draws. A rule
// that can only be read by running the kernel is a rule whose firing set gets guessed.
//
// Exported to Python from the instrumented build (see ops.cpp) so a harness asks for
// the decision instead of reproducing it.
//
// nthreads is the resolved worker count (scorch_spmm_nthreads); nthreads_override is
// what the caller passed, which is what the pool ceiling reads. elem_size is
// sizeof(scalar_t).
inline int scorch_spmm_partition_mode(long rows, long nnz, long k, long out_cols,
                                     long elem_size, int nthreads,
                                     int nthreads_override) {
  int partition_mode = SCORCH_SPMM_PARTITION_DEFAULT;
#ifdef SCORCH_TUNE_HOOKS
  { const char* e = std::getenv("SCORCH_SPMM_PARTITION");
    if (e && *e) { long v = std::atol(e);
      if (v >= 0 && v <= 3) partition_mode = (int)v; } }
#endif
  // OUTPUT-SIZE GATE. The partition buys A's inter-call L2 residency and pays for it
  // in the output store stream: with one global counter the workers all drain from a
  // moving frontier, so at any instant their writes are close together in physical
  // address space; with home ranges they write to as many regions as there are
  // workers, tens of megabytes apart, and the memory controller sees that many open
  // DRAM rows instead of a near-sequential stream.
  //
  // Measured over 2376 cells of the main and large-A corpora, back-stealing against
  // the shipped counter, by output bytes -- and the thread count is identical on
  // every one of the harmed cells, so this is not the policy:
  //
  //   output       float32            float64
  //   < 1 MB       1.239             1.220
  //   1-4 MB       1.433             1.305
  //   4-16 MB      1.258             1.217
  //   16-64 MB     1.109             1.030
  //   64-256 MB    1.029             1.022
  //   >= 256 MB    0.988             0.944   (26.9% of float64 cells below 0.95)
  //
  // Monotone decay, negative at the top. The A-bytes-per-output-byte ratio shows no
  // trend at all across the same cells (1.13 to 1.33 in every band), so the scale
  // that matters is absolute output size, not the balance between the two streams.
  //
  // Expressed as a multiple of the last-level cache rather than as a byte count: the
  // decay begins where the output stops being cache-resident, and a fixed byte
  // threshold would mean something different on every machine. Four times the LLC is
  // 144 MB on a 36 MB L3, which is where the measured sign change is.
  if (partition_mode != 0) {          // nothing to gate when the partition is off
    // A single worker cannot benefit from any of this: there is no second core to keep
    // A resident for and nothing to steal from. What it can still pay is the difference
    // between walking a home range and claiming chunks from the counter, once per row --
    // which is why the matrices that show it are the ones with the most rows per nonzero.
    // On the M5, twenty of the forty-four cells where back-stealing is more than 10%
    // slower than the counter are as-735, 7716 rows of mean degree 1, whose work
    // (7716 * 16 = 123456) is under SCORCH_GRAIN_SPMM and so resolves to exactly one
    // worker. Provably inert for two workers or more.
    bool partition_solo_off = SCORCH_SPMM_PARTITION_SOLO_OFF != 0;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_PARTITION_SOLO_OFF");
      if (e && *e) partition_solo_off = std::atol(e) != 0; }
#endif
    if (partition_solo_off && nthreads <= 1) partition_mode = 0;
  }
  if (partition_mode != 0) {
    // Below a couple of grains of work the partition's bookkeeping is not amortised, and
    // this is where its only measured regression lives. The ARM ladder is what says so.
    // Over 1650 cells the cells where the partition is more than 10% behind the shared
    // counter have a kernel-time distribution of 19 / 27 / 28 / 30 microseconds at min /
    // q1 / median / q3, against 16 / 31 / 62 / 110 for the cells where it wins -- the
    // regression is a SHORT-KERNEL phenomenon, not a shape.
    //
    // A's size was the first candidate and the ladder refuted it: even at a divisor that
    // only fires below about 260 KB of A, a gate on A's bytes gave back 5% of the wins,
    // because home ranges help small matrices too (contiguous ranges narrow each worker's
    // B column band, and workers stop contending on one counter line). Searching every
    // feature the grid carries, in both directions, the cleanest single separator is
    // nnz*max(k,16) -- the same work proxy the thread policy already uses -- at 3.15e5 on
    // float32 and 3.43e5 on float64, which catches 74% and 77% of the regressed cells
    // against 14.5% and 13.7% of the winning ones. Both thresholds are about two
    // SCORCH_GRAIN_SPMM, so that is how this is spelled: in grains, not in a constant.
    long mingrains = SCORCH_SPMM_PARTITION_MINGRAINS;
    long gate_maxthreads = SCORCH_SPMM_PARTITION_GATE_MAXTHREADS;
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_PARTITION_MINGRAINS");
      if (e && *e) { long v = std::atol(e); if (v >= 0) mingrains = v; } }
    { const char* e = std::getenv("SCORCH_SPMM_PARTITION_GATE_MAXTHREADS");
      if (e && *e) { long v = std::atol(e); if (v >= 0) gate_maxthreads = v; } }
#endif
    // The gate only applies where the pool is small enough that the shared counter's
    // contention is not what binds -- see the constant's comment. Above the ceiling this
    // whole block is unreachable, which is how the x86 reading is preserved. Deliberately
    // the caller's POOL and not the resolved worker count: the resolved count is raised per
    // shape and capped by omp_get_num_procs(), so on an 18-processor host it straddles a
    // ceiling of 16 and the gate then fires on some shapes and not others -- which is what
    // the ARM grid caught.
    if (gate_maxthreads > 0) {
      const long pool = nthreads_override > 0 ? (long)nthreads_override
                                              : (long)omp_get_max_threads();
      if (pool > gate_maxthreads) mingrains = 0;
    }
    if (mingrains > 0) {
      const long work_proxy = nnz * (long)(k > 16 ? k : 16);
      if (work_proxy < mingrains * SCORCH_GRAIN_SPMM) partition_mode = 0;
    }
  }
  if (partition_mode != 0) {
    long partition_maxout = SCORCH_SPMM_PARTITION_MAXOUT_LLC * scorch_llc_bytes();
#ifdef SCORCH_TUNE_HOOKS
    { const char* e = std::getenv("SCORCH_SPMM_PARTITION_MAXOUT_MB");
      if (e && *e) { long v = std::atol(e);
        partition_maxout = v > 0 ? v * 1024L * 1024L : 0L; } }   // 0 = no gate
#endif
    if (partition_maxout > 0 &&
        (long)rows * (long)out_cols * elem_size >= partition_maxout)
      partition_mode = 0;
  }
  return partition_mode;
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
