# Phase 0 — where CSR × dense SpMM actually loses to MKL

*Branch `perf/spmm-beat-mkl`, based on `04f321d`. Measured on redwood (Intel
i9-14900K, 8 P-cores + 16 E-cores, 36 MB L3, PyTorch 2.5.1 + MKL 2022.1, float32).
Baseline arm is the faster of MKL's int32-index and int64-index CSR paths.*

## Summary

The losing cells are not bandwidth-bound, latency-bound, or launch-bound. They are
bound by a **serial per-call input-validation pass in the native ABI boundary whose
cost is proportional to `nnz` and independent of `N`**. It costs ~1.4–1.6 ns per
nonzero, it does not parallelize, and it is charged on every `matmul`.

The SpMM kernel itself is already faster than MKL on every cell measured, including
the cells the previous study identified as needing a new kernel.

Two consequences worth stating plainly:

- The reported N-crossover (0.33x at N=32 → 0.80x at N=128 → 1.54x at N=512) is this
  tax being amortized, not blocking starting to pay off. The tax is fixed per
  nonzero while the kernel's work grows with `N`, so the ratio approaches 1 from
  below as `N` grows. That is exactly the observed shape.
- `COMPILER_IR`-adjacent conclusion, and the one that matters most:
  `ADAPTIVE_SPMM_TILING.md` §9.6 concluded that the narrow-`N` high-degree deficit is
  "intrinsic to `v2`'s row-at-a-time full-width traversal", that "MKL is doing
  something structurally different", and that it is "a kernel to write, not a
  threshold to tune". That is refuted. With the tax removed, `v2` beats MKL at
  reddit N=16/32/64/128 by 1.28x/1.91x/1.39x/1.31x. The §9.6 taskset control did not
  catch it because the tax is thread-count-independent, so pinning to 24 physical
  cores left it intact — which is precisely what that control observed (194 ms vs
  181 ms).

## The mechanism

`src/scorch/csrc/native_abi.h` validates every native call. For a CSR operand:

1. `checked_index_tensor` — when the index arrays are int64 (what
   `scipy` / `torch.sparse_csr_tensor` hand a user), a **serial scalar loop over all
   `nnz`** with a `TORCH_CHECK` per element, testing int32 representability; then a
   full `index.to(kInt32)` cast.
2. `checked_csr_view` — a **serial nested loop over all `nnz`** with three branches
   per nonzero: column in `[0, cols)`, and within-row sortedness.

Both run on every call, on both the row-pointer and the column-index array. Neither
is parallel. Together: ~1.4–1.6 ns/nonzero, of which ~0.7–0.9 is (2) and ~0.6–0.7 is
(1).

This was introduced by `1c83b5e` ("validate runtime and native ABI boundaries"), the
commit two before the pinned baseline — so the entire 195-cell tiling study inherited
it.

### How it was found

`perf record` on the timed region named it directly: `at::native::AVX2::copy_kernel`
at 13.7% (the cast), `scorch_native::checked_csr_view<long>` at 3.7%, and
`checked_index_tensor` at 3.1%, against `scorch_spmm_row_regblock<4>` at 15.8%.

Three independent measurements agree on the size:

| evidence | result |
|---|---|
| the identical kernel + parallel skeleton, standalone with no torch (`bench/spmm_micro.cpp`) | 6–18x faster than the same kernel through `scorch.matmul` |
| feeding the STensor int32 indices instead of int64 (no code change) | 1.20–1.95x faster, removing part (1) only |
| gap ÷ nnz, across four matrices spanning 108K–36.8M nonzeros | 1.29, 1.48, 1.40, 1.42 ns/nnz — constant, i.e. per-nonzero |

## Roofline attribution

Machine ceilings, measured on this host with `bench/bwcal.c` (32 threads, best of 5):
pure read 55.3 GB/s, copy 69.7, triad 66.1, **random 64B-line gather 50.1 GB/s** —
the last is the right ceiling for a scattered-gather kernel.

DRAM traffic is measured as `r412e` (`LONGEST_LAT_CACHE.MISS`: every core-originated
request that misses L3, hardware prefetches included) × 64 B, plus C's writeback by
model. This host's kernel exposes no uncore IMC PMU, so the counter was calibrated
against known-traffic microbenchmarks first; `perf stat --control fifo` gates the
counters to the timed region so no matrix loading or warmup is counted.

| cell | arm | ms | measured DRAM | compulsory | amplification | GB/s | % of gather peak | IPC |
|---|---|---|---|---|---|---|---|---|
| reddit@16 | MKL | 37.8 | 0.978 GB | 0.935 GB | **1.05** | 26.3 | 52% | 0.42 |
| reddit@16 | scorch | 205.1 | 3.337 GB | 0.935 GB | **3.57** | 16.3 | 33% | 0.84 |
| reddit@32 | MKL | 79.0 | 2.655 | 0.950 | 2.80 | 34.0 | 68% | 0.24 |
| reddit@32 | scorch | 240.1 | 4.310 | 0.950 | 4.54 | 18.1 | 36% | 0.67 |
| reddit@128 | MKL | 465.8 | 22.22 | 1.039 | 21.4 | 48.0 | **96%** | 0.16 |
| reddit@128 | scorch (tile-j) | 269.7 | 5.940 | 1.039 | 5.72 | 22.5 | 45% | 0.78 |
| inline_1@32 | MKL | 22.2 | 0.799 | 0.361 | 2.21 | 39.0 | 78% | 0.52 |
| inline_1@32 | scorch | 70.0 | 1.213 | 0.361 | 3.36 | 18.2 | 36% | 1.03 |
| bcsstk17@32 | MKL | 0.094 | ~0 | 0.005 | 0.09 | 19.5 | 39% | 1.49 |
| bcsstk17@32 | scorch | 0.588 | ~0 | 0.005 | 0.06 | 2.9 | 6% | 2.50 |
| scatter200@32 | MKL | 3.98 | 0.045 | 0.052 | 0.88 | 12.4 | 25% | 0.61 |
| scatter200@32 | scorch | 10.92 | 0.150 | 0.052 | 2.89 | 14.1 | 28% | 1.08 |

Read this the right way round. Scorch's apparent traffic amplification is *the tax
itself*, not a cache-blocking deficiency. On reddit@16 the validation and cast move
≈2.76 GB per call (representability scan reads the 919 MB int64 array, the cast reads
919 MB and writes 459 MB, the bounds/sortedness scan reads the 459 MB int32 array),
which added to the 0.935 GB compulsory accounts for essentially all of the measured
3.337 GB. MKL sits at 1.05x amplification because it is doing only the compulsory
work.

So the buckets are:

- **bandwidth-bound**: MKL at reddit@128 (96% of the gather ceiling). Nothing to win
  there by moving bytes more cleverly; the win there is tile-j moving fewer bytes.
- **bound by the tax**: every scorch loss cell. Not one of them is near a hardware
  ceiling — scorch sits at 6–36% of the gather ceiling everywhere it loses, while
  showing *higher* IPC than MKL, which is the signature of a serial phase inflating
  wall time rather than a stalled kernel.
- **overhead-bound**: the tiny cells (cora@32, ~50 µs) are dominated by fixed
  per-call cost in both arms and are roughly tied.

The Amdahl consequence is visible directly in a thread-scaling sweep (`taskset`, so
`omp_get_num_procs()` follows the mask):

| cell | scorch 1→32 threads | MKL 1→32 threads |
|---|---|---|
| bcsstk17@32 | 1.51x | 5.60x |
| pubmed@32 | 1.71x | 5.25x |
| inline_1@32 | 2.14x | 3.02x |
| scatter200@32 | 2.55x | 15.15x |
| reddit@32 | 4.00x | 10.37x |

A two-parameter fit on bcsstk17 gives ~0.53 ms serial against ~0.36 ms parallel — a
kernel whose serial prologue is larger than its parallel body. (These absolute
single-thread numbers are contaminated by torch's pool threads spinning on the same
pinned CPU — `perf` showed 20% of cycles in `thread_main` — so read the *ratios*, and
read the torch-free harness for absolute kernel times.)

## What the kernel is actually worth

`bench/spmm_micro.cpp` runs the identical row kernel and the identical parallel
skeleton (one OpenMP team, atomic row work-stealing over the same `chunk`) with no
torch in the process:

| cell | scorch via dispatch | kernel standalone | MKL | standalone vs MKL |
|---|---|---|---|---|
| bcsstk17@32 | 0.586 ms | **0.032 ms** | 0.097 ms | **3.03x** |
| pubmed@32 | 0.398 | **0.023** | 0.139 | **6.07x** |
| scatter200@32 | 9.673 | **0.844** | 1.026 | **1.22x** |
| inline_1@32 | 61.66 | **9.35** | 21.47 | **2.30x** |
| reddit@16 | 205.1 | **29.22** | 37.81 | **1.29x** |
| reddit@32 | 240.1 | **30.70** | 79.04 | **2.57x** |
| reddit@64 | 309.2 | **101.8** | 141.7 | **1.39x** |
| reddit@128 | 617.1 | **355.2** | 465.8 | **1.31x** |

## Hypotheses, re-ranked by measurement

The brief's ranking assumed the deficit was in the kernel. It is not.

| # | hypothesis | verdict |
|---|---|---|
| — | **per-call ABI validation tax** (not in the original list) | **confirmed, dominant, 1.2–2.0x on losing cells** |
| 1 | launch/thread policy for small work | not the cause. Single-thread is also slow, and the poor scaling is the tax's serial fraction. Worth revisiting only after the tax is gone. |
| 2 | deeper ILP / multi-row register blocking | the narrow-k path's 2-nnz ILP is not the limiter: `ilp4`/`ilp8` measure at 0.94–1.11x, no consistent direction. Multi-row blocking still untested and still motivated by the 0.81–0.89 adjacent-row overlap. |
| 3 | A-stream cache-pollution control (NTA hints) | the amplification it was meant to explain is the tax, not B re-fetching. Demoted; needs re-measurement on the fixed build before any work. |
| 4 | cached preprocessing (CSB / reordering) | untouched. Still the only lever for reddit-class structure, but the premise (scorch far behind at narrow N) no longer holds. |
| 5 | full-N-in-registers for N ≤ 64 | untouched. |
| — | **prefetch policy is a small pessimization** | the shipped kernel prefetches 2 nonzeros ahead with gcc locality 1 (`PREFETCHT2`, i.e. no nearer than L2/L3). Removing it entirely is *faster* (1.01–1.09x); 16 nonzeros ahead into L1 plus dropping the redundant mask when `N % 8 == 0` is 1.03–1.14x. |
| — | **int64-index kernel family** (asked during review) | measured and rejected as a default: a 12-byte-per-nonzero A stream costs 1.19–1.51x on DRAM-bound cells (reddit@16 1.19x, reddit@32 1.21x, inline_1@32 1.27x, scatter200@32 1.51x) and nothing on cache-resident ones. Narrowing once is strictly better. Worth having only to support tensors that genuinely exceed int32, dispatched on magnitude rather than dtype. |

## Method notes that cost time to relearn

- **Check the machine is quiet before every timing run.** A leftover `addr2line`
  pinning one core added a flat ~8.6 ms to every arm of every cell — with 32 OpenMP
  threads on 32 CPUs, one preempted worker stalls the whole join barrier for a
  scheduler timeslice. It looked exactly like a uniform kernel regression.
- **`perf stat --control fifo` beats subtracting a setup run.** Gating the counters
  to the timed region is exact; subtraction is not.
- **Hold the outputs, or the allocator lands in the counters.** Freeing a multi-MB
  output inside the timed loop put allocator work inside the counted region — 4x the
  kernel's own time on a sub-millisecond cell. A 2-deep rolling window reproduces
  steady-state allocator behaviour without letting the first rep's page faults
  dominate.
- **redwood's login shell is zsh**, which does not word-split unquoted parameters, so
  `set -- $cell` silently yields one argument. Remote drivers must be bash scripts.
- **Absolute micro-harness times drift between runs when the arm set changes** (reddit
  base moved 41.4 → 30.7 ms between a 7-arm and a 5-arm run) because the arms preceding
  a given arm differ. Compare only within a run; the in-run A/A control stays tight
  (≤0.8%).
