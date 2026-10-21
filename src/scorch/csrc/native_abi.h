#pragma once

#include <torch/extension.h>

#include <ATen/Parallel.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// --------------------------------------------------------------------------- //
// Cost of the ABI boundary
//
// These validators run on EVERY native call, and the CSR ones are O(nnz). Written
// as branchy serial loops with a TORCH_CHECK per element they cost 1.29-1.82 ns per
// nonzero — measured (redwood i9-14900K) at 1.2-2.0x the entire SpMM kernel on
// narrow free dimensions, and, being serial, they put a hard Amdahl ceiling on
// parallel speedup (spmm_csr_float_v2 scaled only 1.5-4.0x over 32 cores because of
// this, against MKL's 3.0-15.2x on the same cells).
//
// The screens below fix that without weakening a single check. Each folds every
// violation into one OR / min / max accumulator, so the loop vectorizes and splits
// across threads; a screen that reports trouble hands off to the ORIGINAL serial
// loop, whose TORCH_CHECKs still produce the byte-identical diagnostic. The screens
// are deliberately CONSERVATIVE (they may flag a valid input, never pass an invalid
// one), so the slow path finding nothing wrong is a legal outcome and simply
// proceeds. Observable behaviour is unchanged; only the valid path got faster.
// --------------------------------------------------------------------------- //

// How many nonzeros a scan needs before it is worth splitting, on the ABI path.
// Deliberately HIGH: with the memo below, a given index array is scanned about once
// rather than once per call, so the split only has to pay off on genuinely large
// arrays, and one million nonzeros is ~4 MB -- past any L3, where threads clearly win.
//
// Set it low and mid-sized inputs paid a parallel region right before the kernel opened
// its own, which measured as a 2-4x REGRESSION on bcsstk17. That measurement stands;
// the explanation once written here -- that the cost was libgomp reshaping the team --
// does not, because nothing ever separated it from the cost of workers still spinning
// after the scan returned. Since the screens moved onto torch's pool the mechanism is
// gone either way, so this value is now a threshold and no longer a guard.
//
// Scorch's own validation path uses its own, lower threshold: see `_WRAP_GRAIN` in
// storage.py, where every wrap scans fresh arrays and the memo does not apply.
#ifndef SCORCH_ABI_VALIDATE_GRAIN
#define SCORCH_ABI_VALIDATE_GRAIN 1048576L
#endif

// LIVE entries retained by the validation memo before it is cleared wholesale (entries whose
// arrays have died are swept, see abi_bound_map). Each verdict is a few dozen bytes; the bound
// exists so a program holding millions of distinct sparse tensors cannot grow it without limit.
#ifndef SCORCH_ABI_MEMO_MAX
#define SCORCH_ABI_MEMO_MAX 4096
#endif

namespace scorch_native {

// Turn "nonzeros per worker" into the grain_size ATen actually wants.
//
// ATen conflates two jobs in one number, and conflating them costs real time. Its
// `parallel_reduce` splits when `n > grain_size`, and then -- see ParallelOpenMP.h --
// opens `#pragma omp parallel` over the WHOLE thread pool and hands work to
// `min(team, ceil(n / grain_size))` of those threads. So the region is opened at full
// width whether one worker or all of them get work. Paying for a full-width region and
// then using two of its threads is the worst of both.
//
// Splitting the number into the two jobs it was doing:
//
//   * `grain` stays the THRESHOLD -- below it the scan is not worth a region at all, so
//     return a grain_size that makes ATen run inline.
//   * above it, every thread in the region that was opened anyway gets a share.
//
// Measured on an M5 at four torch threads, 98,751 nonzeros: as a worker limit the grain
// allowed 2 of the 4 threads and cost 106.0 us; splitting across all 4 cost 78.8 us, for
// the same region. The threshold does the no-regression work and the width is then free.
inline int64_t abi_split(int64_t n, int64_t grain) {
  if (n < 1) return 1;
  if (grain < 1) grain = 1;
  // ATen splits on `n > grain_size`, so returning n itself is how you ask for serial.
  if (n <= grain) return n;
  const int64_t workers = at::get_num_threads();
  if (workers <= 1) return n;
  const int64_t per_worker = (n + workers - 1) / workers;
  return per_worker < 1 ? 1 : per_worker;
}

// at::parallel_reduce over [begin, end), split as abi_split asks -- except that a range it would
// run inline runs here as a plain call. ATen's inline path still sets its thread-id and
// parallel-region guards, which these bodies (loops that call nothing in ATen) do not need.
template <typename R, typename F, typename C>
inline R abi_reduce(int64_t begin, int64_t end, int64_t grain, R ident, const F& body,
                    const C& combine) {
  if (begin >= end) return ident;
  const int64_t split = abi_split(end - begin, grain);
  if (split >= end - begin) return body(begin, end, ident);
  return at::parallel_reduce(begin, end, split, ident, body, combine);
}

// Every screen below splits its scan with `at::parallel_reduce` rather than a raw
// `#pragma omp parallel for num_threads(...)`, so the work runs on TORCH'S thread pool
// at TORCH'S thread count. That is not a stylistic preference; the raw pragma was
// measurably wrong here.
//
// A screen runs between torch operations, so a private team is a team the next torch
// operation has to live with. Asking libgomp for a width torch never asks for makes it
// reshape the team on every crossing, and the workers it spins up keep spinning after
// the scan returns (GOMP's default spin) with nothing to do but compete for cores.
//
// Measured on redwood (i9-14900K, 32 cores, torch at 32 threads, 480k nonzeros): the
// scan itself got FASTER with a private team, 138.5 -> 72.6 us, while a 3.8 MB tensor
// copy standing next to it went 121 -> 1157 us. Net 259.9 -> 1229.4 us, a 4.7x
// regression bought with a 1.9x win. Sharing torch's pool removes the mechanism
// instead of tuning the grain around it: no reshape, no extra spinners, and the host
// program's thread budget is respected for free (`torch.set_num_threads` is the one
// knob, as it is everywhere else in this extension).
//
// `grain` keeps its meaning -- nonzeros per worker -- and is passed straight through as
// ATen's `grain_size`, which also gives the serial case for free: a range shorter than
// one grain, or a screen called from inside an existing parallel region, runs inline.

// --------------------------------------------------------------------------- //
// Structural screens
//
// Each screen replaces a serial loop carrying a TORCH_CHECK per element by folding
// every violation into one OR accumulator, so the loop vectorizes and splits across
// threads. They are deliberately CONSERVATIVE: a screen may report trouble on a valid
// input, never pass an invalid one. The caller's original serial loop therefore stays
// the sole source of diagnostics, and a screen that was merely pessimistic costs one
// wasted walk and proceeds.
// --------------------------------------------------------------------------- //

// Coordinates are int32, so any limit past INT32_MAX admits every representable
// coordinate. Saturating keeps the comparison in int32 and can only make a screen
// more permissive, which the caller's serial loop re-checks exactly.
inline int32_t abi_saturate_limit(int64_t limit) {
  return limit > (int64_t)std::numeric_limits<int32_t>::max()
             ? std::numeric_limits<int32_t>::max()
             : (int32_t)limit;
}

// The same clamp for whichever integer width the index arrays actually use. The
// screens below are templated over that width because Scorch's own storage keeps
// int64 indices when a caller handed them in that way (torch's CSR does), and the
// Python-side validator wants the same single fused pass the ABI boundary gets. The
// int32 entry points keep their exact signatures and delegate, so every existing
// caller is unchanged.
template <typename T>
inline T abi_limit_clamp(int64_t limit) {
  return limit > (int64_t)std::numeric_limits<T>::max()
             ? std::numeric_limits<T>::max()
             : (T)limit;
}

// "every coordinate lies in [0, limit)"
template <typename T>
inline bool abi_screen_bounds_typed(const T* crd, int64_t n, int64_t limit,
                                    int64_t grain = SCORCH_ABI_VALIDATE_GRAIN) {
  if (n <= 0) return false;
  const T lim = abi_limit_clamp<T>(limit);
  const int bad = at::parallel_reduce(
      (int64_t)0, n, abi_split(n, grain), 0,
      [&](int64_t begin, int64_t end, int partial) {
        for (int64_t p = begin; p < end; ++p) {
          const T c = crd[p];
          partial |= (c < 0) | (c >= lim);
        }
        return partial;
      },
      [](int a, int b) { return a | b; });
  return bad != 0;
}

inline bool abi_screen_bounds(const int32_t* crd, int64_t n, int64_t limit) {
  return abi_screen_bounds_typed<int32_t>(crd, n, limit);
}

// "positions do not decrease and stay within [0, nnz]", for the callers that validate
// a position array on its own. O(rows) rather than O(nnz), but a TORCH_CHECK per row
// is what made it show up at all -- and, run on one thread, an O(rows) loop costs more
// than an O(nnz) one split across all of them once rows are short (see
// abi_screen_compressed_typed), so it is split the same way.
template <typename T>
inline bool abi_screen_spans_typed(const T* pos, int64_t count, int64_t nnz,
                                   int64_t grain = SCORCH_ABI_VALIDATE_GRAIN) {
  if (count <= 1) return false;
  const T nnz_t = (T)nnz;  // callers check nnz fits the index width first
  const int bad = abi_reduce(
      (int64_t)1, count, grain, 0,
      [&](int64_t begin, int64_t end, int partial) {
        for (int64_t i = begin; i < end; ++i) {
          partial |= (pos[i] < pos[i - 1]) | (pos[i] > nnz_t);
        }
        return partial;
      },
      [](int a, int b) { return a | b; });
  return bad != 0;
}

inline bool abi_screen_spans(const int32_t* pos, int64_t count, int64_t nnz) {
  return abi_screen_spans_typed<int32_t>(pos, count, nnz);
}

// "positions partition [0, nnz] without decreasing, every coordinate lies in
// [0, limit), and — when required — coordinates STRICTLY ascend inside each
// parent's span."
//
// Both coordinate tests run as FLAT loops over the whole coordinate array rather than
// nested per-parent loops. A per-parent loop cannot vectorize usefully when spans are
// short: a 39-nonzero FEM row spends more on vector peeling and tail than on the
// eight-wide body, ~3.5 cycles per nonzero, i.e. the check cost more than the SpMM it
// was guarding.
//
// Bounds do not care about span structure at all. Ordering does, but only through one
// observation: a NON-INCREASE at position p (crd[p-1] >= crd[p]) is legal exactly when
// p is the first position of a span. So count every non-increase flat, subtract the
// non-increases that sit on a span boundary, and require nothing left.
//
// The comparison is `>=` rather than `>` because this tree requires STRICTLY increasing
// coordinates inside a compressed span, not merely non-decreasing ones (commit 7fd9116:
// a duplicate entry silently miscomputes a merged kernel). A duplicate is a violation
// here, so the screen has to count it as one; counting descents instead would report a
// duplicate clean and the serial loop that names it would never run. Both
// `require_sorted=true` callers — checked_csr_view on the prebuilt route and
// validate_jit_tensor's compressed level on the generated route — are the two sites
// 7fd9116 made strict; the `require_sorted=false` caller (the CSR attention mask) never
// enters this branch.
//
// No loop here runs over the parents on one thread. Run serially, the position checks
// and the boundary count cost more than the whole coordinate scan split across workers
// once spans are short: M5, int64, 1M nonzeros, 65 us in 1024 rows and 640 us in 1M rows
// (8.4M nonzeros: 478 us and 1335 us). The positions are checked by a split scan first;
// that makes them a valid partition, so each worker of the coordinate scan finds the
// first boundary of its range by binary search and walks its boundaries as it goes. It
// scans in blocks of 16 KiB and walks a block's boundaries straight after the block's
// flat loop, so the two coordinates around each boundary are read from L1: where spans
// are a cache line long or shorter, a separate boundary pass would read every line of
// the coordinate array a second time.
//
// Given `pos_out` and `crd_out` (N narrower than T), the same pass also writes each
// position and coordinate converted to N, so arrays seen for the first time are read once
// rather than once to check and once to convert (abi_narrow_csr_first_sight). The copies
// are exact when the screen is clean -- every position then lies in [0, nnz] and every
// coordinate in [0, limit), and callers check both bounds fit N -- and meaningless when not.
template <typename T, typename N = T>
inline bool abi_screen_compressed_typed(
    const T* pos, const T* crd, int64_t parent_count, int64_t nnz, int64_t limit,
    bool require_sorted, int64_t grain = SCORCH_ABI_VALIDATE_GRAIN,
    N* pos_out = nullptr, N* crd_out = nullptr) {
  constexpr bool kCopy = !std::is_same<T, N>::value;
  int bad = 0;
  const T nnz32 = (T)nnz;  // callers check nnz fits the index width first
  if constexpr (kCopy) {
    if (parent_count >= 0) pos_out[0] = (N)pos[0];
  }
  if (parent_count > 0) {
    bad = abi_reduce(
        (int64_t)0, parent_count, grain, 0,
        [&](int64_t begin, int64_t end, int partial) {
          for (int64_t parent = begin; parent < end; ++parent) {
            const T start = pos[parent], stop = pos[parent + 1];
            partial |= (start < 0) | (stop < start) | (stop > nnz32);
            if constexpr (kCopy) pos_out[parent + 1] = (N)stop;
          }
          return partial;
        },
        [](int a, int b) { return a | b; });
  }
  if (bad || nnz <= 0) return bad != 0;

  const T lim = abi_limit_clamp<T>(limit);
  const T c0 = crd[0];
  bad |= (c0 < 0) | (c0 >= lim);
  if constexpr (kCopy) crd_out[0] = (N)c0;
  if (!require_sorted) {
    bad |= abi_reduce(
        (int64_t)1, nnz, grain, 0,
        [&](int64_t begin, int64_t end, int partial) {
          for (int64_t p = begin; p < end; ++p) {
            const T c = crd[p];
            partial |= (c < 0) | (c >= lim);
            if constexpr (kCopy) crd_out[p] = (N)c;
          }
          return partial;
        },
        [](int a, int b) { return a | b; });
    return bad != 0;
  }

  // Out of range is one unsigned comparison -- a negative coordinate converts to a value past
  // any positive limit; a limit that is not positive, under which it would admit everything,
  // has already failed the first coordinate's signed test above -- and both counts are kept at
  // the index type's width, which no count can wrap: a chunk of an int32 array holds fewer than
  // 2^31 coordinates. Together they let the flat loop vectorize at the full lane count. Two
  // signed tests ORed into an int that is read as a condition after the loop kept it scalar
  // ("value that could not be identified as reduction is used outside the loop"). M5, one
  // thread, 8.4M coordinates: int32 1382 -> 467 us, int64 1375 -> 942 us.
  //
  // The ordering count is the strict contract above, a block's non-increases (crd[p - 1] >=
  // crd[p]), taken as the block's length less its increases (crd[p - 1] < crd[p]): the same
  // number, since one of the two holds at every p. Counted directly, the non-increase is a
  // `>=`, which AVX2 has no packed compare for: vpcmpgtd and one more instruction per vector
  // (clang 21 -march=broadwell: 10 instructions to the loop's 9 for int32, 11 to 10 for int64;
  // +692 instructions per first call in GCC's build on sapling's Broadwell, attr3 83817). The
  // increase is one vpcmpgtd. AVX-512's
  // vpcmpd and NEON's cmge take either predicate in one instruction, so on Zen 4 and the M5
  // the two loops are the same length.
  using U = std::make_unsigned_t<T>;
  const U ulim = (U)lim;
  constexpr int64_t kBlock = 16384 / (int64_t)sizeof(T);
  // Span boundaries are pos[1 .. parent_count - 1]; pos[parent_count] is nnz, past every scan.
  const T* const first_boundary = pos + 1;
  const T* const last_boundary = pos + std::max<int64_t>(parent_count, 1);
  const int64_t unclean = abi_reduce(
      (int64_t)1, nnz, grain, (int64_t)0,
      [&](int64_t begin, int64_t end, int64_t partial) {
        // The first boundary at or past `begin`; one before it, if any, lies before `begin`,
        // so the walk below never skips this range's first boundary as a repeat. r never
        // passes last_boundary, so each block's search range is valid.
        int64_t r = std::lower_bound(first_boundary, last_boundary, (T)begin) - pos;
        U out_of_range = 0;
        U unordered = 0;
        for (int64_t b0 = begin; b0 < end; b0 += kBlock) {
          const int64_t b1 = std::min(b0 + kBlock, end);
          U incr = 0;
          // Reads crd[p - 1], which is in range because the scan starts at 1.
          for (int64_t p = b0; p < b1; ++p) {
            const T c = crd[p];
            out_of_range += ((U)c >= ulim);
            incr += (crd[p - 1] < c);
            if constexpr (kCopy) crd_out[p] = (N)c;
          }
          // The block's non-increases. b1 - b0 <= kBlock, so the count fits U.
          U nonincr = (U)(b1 - b0) - incr;
          // The block's boundaries pos[r .. r_end) are the p in [b0, b1), so 1 <= p < nnz.
          // Found by binary search, which makes the walk a counted loop the compiler can
          // vectorize where spans hold one element and every position is a boundary. Parents
          // with empty spans repeat a boundary, which counts once. Each boundary's own
          // non-increase was counted by the flat loop of THIS block, so the subtraction
          // cannot take the count below zero.
          const int64_t r_end = std::lower_bound(pos + r, last_boundary, (T)b1) - pos;
          for (int64_t q = r; q < r_end; ++q) {
            const T p = pos[q];
            nonincr -= (U)((p != pos[q - 1]) & (crd[p - 1] >= crd[p]));
          }
          r = r_end;
          unordered += (nonincr != 0);
        }
        return partial + (int64_t)(out_of_range != 0) + (int64_t)unordered;
      },
      [](int64_t a, int64_t b) { return a + b; });
  bad |= (unclean != 0);
  return bad != 0;
}

inline bool abi_screen_compressed(const int32_t* pos, const int32_t* crd,
                                  int64_t parent_count, int64_t nnz, int64_t limit,
                                  bool require_sorted) {
  return abi_screen_compressed_typed<int32_t>(pos, crd, parent_count, nnz, limit,
                                              require_sorted);
}

// "COO coordinates ascend strictly lexicographically across levels": no entry's coordinate
// tuple is smaller than the one before it, and none is equal to it. An equal tuple is one
// coordinate stored twice, which the kernels do not sum (2026-10-12: refused, and the
// caller calls coalesce()), so it is as much a violation as a descent.
//
// Positions are taken kLexBlock at a time and levels first to last; tie[i] says that
// position i still equals its predecessor on every level read so far. A level is read only
// while some position of the block is still tied, so a block reads no level that a
// position-by-position comparison would not read for one of its positions. The data
// decides a branch once per level of a block (read the next level or not) and once per
// block (stop at the first block with a violation), never per position: each pass inside
// a block is a fixed-count loop over adjacent elements, which the compiler vectorizes.
//
// The loop this replaces compared one position at a time, leaving at the first level that
// differed; it could not be vectorized, and making it strict (`cmp < 0` to `cmp <= 0`)
// cost about 0.25 ns more per entry under gcc (mkt1, 65,536 entries: +16 us). On the
// standalone loops (int32 and int64, 1 to 4 levels, 8,192 to 1,048,576 entries, the first
// level deciding at every entry, at 1 in 8, at random and never: 78 cells) this form takes
// 0.09x to 0.64x the time of the non-strict loop it replaces under gcc 13.4 on Broadwell
// and 0.13x to 0.62x under Apple clang 21 on an M5 Max, and is slower in no cell on
// either. 256 was the fastest of 256, 512 and 1024 on Broadwell.
constexpr int64_t kLexBlock = 256;

template <typename T>
inline bool abi_screen_lex_typed(const std::vector<const T*>& levels, int64_t n,
                                 int64_t grain = SCORCH_ABI_VALIDATE_GRAIN) {
  if (levels.empty() || n <= 1) return false;
  const size_t depth = levels.size();
  const T* const* lv = levels.data();
  const int bad = at::parallel_reduce(
      (int64_t)1, n, abi_split(n - 1, grain), 0,
      [&](int64_t begin, int64_t end, int partial) {
        T tie[kLexBlock];
        for (int64_t first = begin; first < end && partial == 0; first += kLexBlock) {
          const int64_t count = std::min<int64_t>(kLexBlock, end - first);
          T below = 0;  // some position is smaller at the first level where it differs
          T tied = 0;   // some position equals its predecessor on every level so far
          const T* level = lv[0] + first;
          for (int64_t i = 0; i < count; ++i) {
            below |= level[i] < level[i - 1];
            tie[i] = level[i] == level[i - 1];
            tied |= tie[i];
          }
          for (size_t l = 1; l < depth && tied != 0; ++l) {
            level = lv[l] + first;
            tied = 0;
            for (int64_t i = 0; i < count; ++i) {
              below |= tie[i] & (level[i] < level[i - 1]);
              tie[i] &= level[i] == level[i - 1];
              tied |= tie[i];
            }
          }
          // Still tied after the last level: the same tuple twice.
          partial |= (below | tied) != 0;
        }
        return partial;
      },
      [](int a, int b) { return a | b; });
  return bad != 0;
}

inline bool abi_screen_lex(const std::vector<const int32_t*>& levels, int64_t n) {
  return abi_screen_lex_typed<int32_t>(levels, n);
}

// --------------------------------------------------------------------------- //
// Validation memo
//
// Two things were being redone on every native call for index arrays that had not
// changed since the last one:
//
//   1. The O(nnz) structural scans. On reddit at N=16 that is ~9 ms of index
//      re-reading against a 29 ms kernel.
//   2. The int64 -> int32 narrowing, which allocates and fills a fresh array every
//      call. Worse than its own cost: because the narrowed array is new each time,
//      nothing downstream of it can be memoized either.
//
// A tensor's indices do not change between calls, so both are cached. Caching (2) is
// what lets (1) hit at all on the JIT path, where operands arrive as int64.
//
// Recycling safety. Entries are keyed on the array's data_ptr and hold a
// weak_intrusive_ptr to the owning StorageImpl. A weak reference keeps the
// StorageImpl's own allocation alive after its data is released, so while an entry
// exists no *different* StorageImpl can occupy the address recorded in `storage` --
// an expired weak pointer is therefore proof the entry is stale, and a live one plus
// an equal `storage` address is proof this is the array that was validated. nbytes,
// numel, the version counter and the storage's write-witness epoch are compared too, so
// a resize or any mutation routed through torch invalidates -- including one made through
// a torch sparse tensor's index accessor, which moves no counter the array shares (see
// Write witnesses below). Keying on data_ptr rather than on the StorageImpl means
// two distinct views of one storage get an entry each instead of evicting each other
// on every call.
//
// What this does NOT catch: a write straight through a shared raw buffer (numpy
// writing into memory a torch tensor aliases) does not bump the version counter, so a
// buffer corrupted that way can now reach a kernel unchecked; nor do the writes the
// Write witnesses below cannot see (a sparse tensor scorch was never given). That is a deliberate
// trade, made because the per-call scan cost is what stands between these kernels and
// MKL. SCORCH_ABI_VALIDATE_MEMO=0 restores strict per-call validation, narrowing
// included.
//
// One consequence worth stating: the narrowed array is now SHARED between calls, so a
// kernel that wrote through its input index pointers would corrupt every later call
// rather than a private copy. No generated or prebuilt kernel does -- input index
// arrays are read-only by construction, and a test pins that they come back unchanged
// -- but codegen does hand them out as `int*` rather than `const int*`, so this is
// checked rather than guaranteed by the type.
// --------------------------------------------------------------------------- //

// A tensor's version counter, or 0 when it does not track one. Inference-mode tensors
// have their counter disabled and current_version() throws on them, so this must be
// asked rather than assumed — a `with torch.inference_mode():` block would otherwise
// turn every matmul into an exception.
inline uint32_t abi_version_of(const torch::Tensor& t) {
  auto* impl = t.unsafeGetTensorImpl();
  return impl->version_counter().enabled()
             ? impl->version_counter().current_version()
             : 0u;
}

// --------------------------------------------------------------------------- //
// Write witnesses
//
// An index array's version counter does not see every write to it. A torch sparse
// tensor's index arrays are the caller's own tensors (its members:
// get_sparse_csr_impl(a)->plain_indices() is the very TensorImpl the caller passed to
// sparse_csr_tensor), and a write through the caller's array or any view of it moves
// their counter. But the accessors crow_indices(), col_indices(), indices() and
// values() return aliases that share the SPARSE TENSOR's counter, so a write through
// one of them moves only the sparse tensor's, never the array's. (torch 2.13;
// measured.) Any cache keyed on the array's identity and version therefore misses half
// the writes, whichever of the two tensors it was handed.
//
// So every torch sparse tensor whose index arrays scorch reads is recorded, on the
// storage of each array, as a WITNESS: a copy of its version counter
// (c10::VariableVersion shares the counter, so the copy reads writes made through the
// tensor and through every alias it handed out) and the version last seen. observe()
// compares each witness with what it last saw, and when any has moved, advances the
// storage's EPOCH. AbiArrayId carries the epoch, so a memo entry, a plan or a transpose
// built before a write through any witnessed tensor no longer matches after it.
//
// A witness first seen at a version other than 0 has been written through before scorch
// saw it, and advances the epoch at once: what a cache describes may predate that
// write. A witness is dropped once its counter is held by nothing but this table -- the
// tensor and every alias of it are gone, so nothing can write through it again -- after
// a last comparison, by the next registration on the same storage (the only place that
// prunes; see AbiWatchCopy). It is recognised again by the counter itself, so it holds no
// reference to the tensor: a weak one would keep the dead TensorImpl's allocation, and
// with it the counter, alive, and the counter would never be the table's alone.
//
// What this still cannot see: a write through the accessor of a sparse tensor scorch
// never received (its counter is shared with no tensor scorch holds); a write through
// the caller's own array when scorch was handed only an accessor's alias of it, as
// `sparse_softmax_csr( csr.crow_indices(), ...)` is (an alias is not a view: torch
// records no base, so nothing leads from it to the array or the sparse tensor); a write
// through a raw buffer (numpy); and anything written under inference mode (no counter).
// The memo's documented trade above covers the same ground.
//
// The table lives in ONE module, scorch_ops (SCORCH_ABI_WATCH_OWNER), because the
// witnesses are registered there. A JIT-compiled kernel module, which compiles its own
// copy of this header, reads epochs through a capsule scorch_ops exports (abi_epoch_of
// below), so its lookups see the same writes. (Whether that module also has a memo of
// its own depends on the platform: see the note on abi_memo_clear.)
//
// The two builds of this header must not define one external symbol two ways: macOS's
// loader merges weak (inline) definitions across loaded images, and an owner's `static
// const AbiWatchApi api` read as a kernel module's `static const AbiWatchApi* api`
// crashed the first JIT-compiled kernel to run. So the owner's functions have names no
// kernel module defines, and the kernel module's are static (internal linkage), so
// nothing of theirs is merged with anything.
// --------------------------------------------------------------------------- //

struct AbiWatchApi {
  // The epoch of the storage at `storage_impl` (a c10::StorageImpl*), after observing
  // its witnesses; 0 when it has none.
  uint64_t (*epoch_of)(const void* storage_impl);
};

#ifdef SCORCH_ABI_WATCH_OWNER

// The identity of a version counter: the address of the shared object a VariableVersion
// holds, its only member. Stable for as long as the table holds a copy, so it cannot be
// reused while a witness names it.
static_assert(sizeof(c10::VariableVersion) == sizeof(void*),
              "VariableVersion is no longer one shared pointer: change abi_counter_id");
inline const void* abi_counter_id(const c10::VariableVersion& counter) {
  const void* id;
  std::memcpy(&id, &counter, sizeof(id));
  return id;
}

// One copy of a witnessed counter, shared by the watch of every storage the tensor's
// arrays live in (a CSR operand's two), so that "held by nothing but this table" is one
// copy, not one per array: with a copy per array the counter of a dead CSR operand was
// never the table's alone.
struct AbiCounterRef {
  c10::VariableVersion counter;  // shared with the tensor and every alias it returned
  const void* id;
};

struct AbiWitness {
  std::shared_ptr<AbiCounterRef> ref;
  uint32_t seen;  // per storage: each storage's epoch moves on its own observation
};

struct AbiStorageWatch {
  c10::weak_intrusive_ptr<c10::StorageImpl> alive;
  std::atomic<uint64_t> epoch{0};
  // Advanced whenever a witness is added or dropped, so the holder of a copy of the
  // list (a plan, AbiWatchCopy below) can tell without the lock that its copy is still
  // the whole list.
  std::atomic<uint64_t> list_version{0};
  std::vector<AbiWitness> witnesses;  // under abi_watch_mutex()

  explicit AbiStorageWatch(c10::StorageImpl* impl)
      : alive(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(impl)) {}

  bool live() const { return !alive.expired(); }

  // Under abi_watch_mutex(). The epoch after comparing every witness with what it last
  // saw. With `prune`, also drops the witnesses whose counter only this table holds; only
  // registration prunes, and it holds the GIL (see AbiWatchCopy for why that matters).
  uint64_t observe_locked(bool prune) {
    bool moved = false;
    bool dropped = false;
    for (auto it = witnesses.begin(); it != witnesses.end();) {
      const uint32_t now = it->ref->counter.current_version();
      if (now != it->seen) {
        moved = true;
        it->seen = now;
      }
      if (prune && it->ref->counter.unique()) {
        it = witnesses.erase(it);
        dropped = true;
      } else {
        ++it;
      }
    }
    if (moved) epoch.fetch_add(1, std::memory_order_release);
    if (dropped) list_version.fetch_add(1, std::memory_order_release);
    return epoch.load(std::memory_order_acquire);
  }
};

inline std::mutex& abi_watch_mutex() {
  static std::mutex m;
  return m;
}

using AbiWatchMap = std::unordered_map<const void*, std::shared_ptr<AbiStorageWatch>>;
inline AbiWatchMap& abi_watch_map() {
  static AbiWatchMap m;
  return m;
}

// Drops the entries `dead` says are dead once the map has doubled since the last sweep, so
// an insert costs amortized O(1) however many live entries there are. Under the lock.
template <typename Map, typename Dead>
inline void abi_sweep_when_doubled(Map& map, size_t& swept_at, Dead dead) {
  if (map.size() < 2 * std::max<size_t>(swept_at, 32)) return;
  for (auto it = map.begin(); it != map.end();) {
    if (dead(it->second)) it = map.erase(it);
    else ++it;
  }
  swept_at = map.size();
}

// Every witnessed counter, by identity, so the watches of several storages share one
// copy.
using AbiCounterRefs = std::unordered_map<const void*, std::weak_ptr<AbiCounterRef>>;
inline AbiCounterRefs& abi_counter_refs() {
  static AbiCounterRefs m;
  return m;
}

// Under abi_watch_mutex(). The shared copy of `counter`, made if there is none.
inline std::shared_ptr<AbiCounterRef> abi_counter_ref_locked(
    const c10::VariableVersion& counter, const void* id) {
  auto& refs = abi_counter_refs();
  auto it = refs.find(id);
  if (it != refs.end()) {
    if (std::shared_ptr<AbiCounterRef> held = it->second.lock()) return held;
    refs.erase(it);
  }
  static size_t swept_at = 0;
  abi_sweep_when_doubled(refs, swept_at,
                         [](const std::weak_ptr<AbiCounterRef>& r) { return r.expired(); });
  auto ref = std::make_shared<AbiCounterRef>(AbiCounterRef{counter, id});
  refs.emplace(id, ref);
  return ref;
}

// Under abi_watch_mutex(). The watch on this storage, made if `create`; null otherwise.
// An entry whose storage died is replaced: its address may now be another storage's.
// Dead entries are swept once the map has doubled since the last sweep; live ones are
// never cleared, because a plan holding one must keep receiving the witnesses registered
// after it was built.
inline std::shared_ptr<AbiStorageWatch> abi_watch_locked(c10::StorageImpl* impl,
                                                         bool create) {
  auto& map = abi_watch_map();
  auto it = map.find(impl);
  if (it != map.end()) {
    if (it->second->live()) return it->second;
    map.erase(it);
  }
  if (!create) return nullptr;
  static size_t swept_at = 0;
  abi_sweep_when_doubled(map, swept_at, [](const std::shared_ptr<AbiStorageWatch>& w) {
    return !w->live();
  });
  auto watch = std::make_shared<AbiStorageWatch>(impl);
  map.emplace(impl, watch);
  return watch;
}

// The watch on `t`'s storage, made if there is none: what a plan holds for each array
// it reads.
inline std::shared_ptr<AbiStorageWatch> abi_watch_of(const torch::Tensor& t) {
  std::lock_guard<std::mutex> guard(abi_watch_mutex());
  return abi_watch_locked(t.storage().unsafeGetStorageImpl(), true);
}

// A copy of a storage's witness list and of what was last seen of each, taken under the
// lock. It answers "has any witness moved since?" with plain loads, which is what a plan
// asks on every call: observing both arrays under the lock instead cost 22-42 ns per call
// on the M5, 1-4% of a 1-2 us product.
//
// The copy holds the table's AbiCounterRef pointers without owning them, and up to
// kInline of them inside itself, so the check reads no allocation of its own and changes
// no reference count. That matters when the list changes between calls: a new CSR object
// over the same arrays each step adds a witness, and with owning copies every refresh paid
// atomic reference-count traffic and freed the previous step's records inside the next
// product -- about 1 us per cold call on mkt1 (sc2, 2026-10-08).
//
// A pointer is valid while the list is the one copied: an entry leaves the list only by
// being pruned, pruning advances list_version, and the check compares list_version before
// it reads any pointer. That holds only if no prune runs between the comparison and the
// reads, so a prune and a check never run at once: both are made holding the GIL (plan
// runs never release it; registration is called from Python), and observing from anywhere
// else (a validation memo, which may run without the GIL) never prunes.
struct AbiWatchCopy {
  static constexpr size_t kInline = 4;
  struct Entry {
    const AbiCounterRef* ref;
    uint32_t seen;
  };
  // No list has this version, so the first check takes a copy.
  uint64_t list_version = ~uint64_t{0};
  uint64_t epoch = 0;
  size_t count = 0;
  bool overflow = false;  // more witnesses than kInline: every check takes the lock
  Entry entries[kInline];
};

// Without the lock, holding the GIL: whether `copy` is still the storage's whole witness
// list and none of them has moved since it was taken, in which case the epoch is still
// copy.epoch. Counters only grow, so a write through any copied witness shows as a
// version other than the one copied.
inline bool abi_watch_unchanged(const AbiStorageWatch& watch,
                                const AbiWatchCopy& copy) {
  if (copy.overflow ||
      watch.list_version.load(std::memory_order_acquire) != copy.list_version) {
    return false;
  }
  for (size_t i = 0; i < copy.count; ++i) {
    if (copy.entries[i].ref->counter.current_version() != copy.entries[i].seen) {
      return false;
    }
  }
  return watch.epoch.load(std::memory_order_acquire) == copy.epoch;
}

// Holding the GIL: observes the storage under the lock, without pruning, and copies its
// witness list into `copy`. The epoch now.
inline uint64_t abi_watch_refresh(AbiStorageWatch& watch, AbiWatchCopy& copy) {
  std::lock_guard<std::mutex> guard(abi_watch_mutex());
  copy.epoch = watch.observe_locked(/*prune=*/false);
  copy.list_version = watch.list_version.load(std::memory_order_relaxed);
  copy.overflow = watch.witnesses.size() > AbiWatchCopy::kInline;
  copy.count = copy.overflow ? 0 : watch.witnesses.size();
  for (size_t i = 0; i < copy.count; ++i) {
    copy.entries[i] = {watch.witnesses[i].ref.get(), watch.witnesses[i].seen};
  }
  return copy.epoch;
}

// Records the sparse tensor `operand` as a witness of the storage under `array`.
// Nothing for an operand without a version counter (an inference tensor): it can
// witness nothing.
inline void abi_watch_register(const torch::Tensor& array,
                               const torch::Tensor& operand) {
  const c10::VariableVersion& counter =
      operand.unsafeGetTensorImpl()->version_counter();
  if (!counter.enabled()) return;
  const void* id = abi_counter_id(counter);
  std::lock_guard<std::mutex> guard(abi_watch_mutex());
  std::shared_ptr<AbiStorageWatch> watch =
      abi_watch_locked(array.storage().unsafeGetStorageImpl(), true);
  watch->observe_locked(/*prune=*/true);  // called from Python: the GIL is held
  for (const AbiWitness& w : watch->witnesses) {
    if (w.ref->id == id) return;  // already a witness, and just compared
  }
  const uint32_t now = counter.current_version();
  watch->witnesses.push_back(AbiWitness{abi_counter_ref_locked(counter, id), now});
  if (now != 0) watch->epoch.fetch_add(1, std::memory_order_release);
  watch->list_version.fetch_add(1, std::memory_order_release);
}

inline uint64_t abi_epoch_of_impl(const void* storage_impl) {
  auto* impl =
      const_cast<c10::StorageImpl*>(static_cast<const c10::StorageImpl*>(storage_impl));
  std::lock_guard<std::mutex> guard(abi_watch_mutex());
  std::shared_ptr<AbiStorageWatch> watch = abi_watch_locked(impl, false);
  return watch != nullptr ? watch->observe_locked(/*prune=*/false) : 0;
}

// What scorch_ops exports in its capsule.
inline const AbiWatchApi* abi_watch_owner_api() {
  static const AbiWatchApi api{&abi_epoch_of_impl};
  return &api;
}

// The epoch of `t`'s storage (see Write witnesses above).
inline uint64_t abi_epoch_of(const torch::Tensor& t) {
  return t.has_storage() ? abi_epoch_of_impl(t.storage().unsafeGetStorageImpl()) : 0;
}

#else  // not scorch_ops: the table is scorch_ops', reached through its capsule

// Not a lazily initialized static: its initialization guard would be held while waiting
// for the GIL, and a thread holding the GIL could be waiting on that guard. Two threads
// importing the capsule at once is harmless; both find the same one.
static inline const AbiWatchApi* abi_watch_imported_api() {
  static std::atomic<const AbiWatchApi*> imported{nullptr};
  static std::atomic<bool> looked{false};
  if (looked.load(std::memory_order_acquire)) {
    return imported.load(std::memory_order_acquire);
  }
  PyGILState_STATE gil = PyGILState_Ensure();
  const AbiWatchApi* found = static_cast<const AbiWatchApi*>(
      PyCapsule_Import("scorch_ops._abi_watch_api", 0));
  if (found == nullptr) PyErr_Clear();  // no witnesses anywhere, so every epoch is 0
  PyGILState_Release(gil);
  imported.store(found, std::memory_order_release);
  looked.store(true, std::memory_order_release);
  return found;
}

// The epoch of `t`'s storage (see Write witnesses above).
static inline uint64_t abi_epoch_of(const torch::Tensor& t) {
  if (!t.has_storage()) return 0;
  const AbiWatchApi* api = abi_watch_imported_api();
  return api != nullptr ? api->epoch_of(t.storage().unsafeGetStorageImpl()) : 0;
}

#endif  // SCORCH_ABI_WATCH_OWNER

// Identity of one index array, precise enough that a cached verdict about it can be
// trusted. See the recycling-safety note above for why these fields and no others.
struct AbiArrayId {
  c10::weak_intrusive_ptr<c10::StorageImpl> alive;
  const void* storage = nullptr;  // compared, never dereferenced
  const void* data = nullptr;
  int64_t nbytes = 0;
  int64_t numel = 0;
  uint32_t version = 0;
  uint64_t epoch = 0;  // the storage's write-witness epoch (see Write witnesses)

  AbiArrayId() : alive(c10::intrusive_ptr<c10::StorageImpl>()) {}  // null weak ref
};

inline AbiArrayId abi_identify(const torch::Tensor& t) {
  AbiArrayId id;
  auto* impl = t.storage().unsafeGetStorageImpl();
  // reclaim_copy takes a borrowed pointer and returns an owning ref; the temporary
  // strong ref drops at the end of the statement, leaving only the weakcount.
  id.alive = c10::weak_intrusive_ptr<c10::StorageImpl>(
      c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(impl));
  id.storage = impl;
  id.data = t.data_ptr();
  id.nbytes = (int64_t)t.storage().nbytes();
  id.numel = t.numel();
  id.version = abi_version_of(t);
  id.epoch = abi_epoch_of(t);
  return id;
}

inline bool abi_same_array(const AbiArrayId& a, const AbiArrayId& b) {
  return a.storage == b.storage && a.data == b.data && a.nbytes == b.nbytes &&
         a.numel == b.numel && a.version == b.version && a.epoch == b.epoch;
}

inline bool abi_memo_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("SCORCH_ABI_VALIDATE_MEMO");
    return !(e && *e && std::atol(e) == 0);
  }();
  return on;
}

inline std::mutex& abi_memo_mutex() {
  static std::mutex m;
  return m;
}

// The int32 copy made for an int64 caller array, held strongly so it outlives the
// call that made it. It dies with its source: when the source storage goes, the weak
// reference expires and the sweep below drops the entry and the copy with it.
struct AbiNarrowEntry {
  AbiArrayId source;
  torch::Tensor narrowed;

  bool live() const { return !source.alive.expired(); }
};

// Which family of structural check a cached verdict belongs to. It is params[0] of
// every entry, because entries are keyed on an array address and two different checks
// over the same array would otherwise be indistinguishable: a one-level COO view
// records {extent, nnz} for "coordinates in range" and the lexicographic check over
// that same single level would record {extent, nnz} for "coordinates ascend", and one
// would silently satisfy the other. Within a family, matching params SHOULD share —
// a CSR matrix validated through the prebuilt path is validated for a JIT kernel too.
enum AbiCheckKind : int64_t {
  AbiCheckBounds = 1,      // every coordinate in [0, extent)
  AbiCheckCompressed = 2,  // spans partition [0, nnz], coordinates in range and,
                           // when required, strictly increasing inside each span
  // 3 was the order check WITHOUT strictness ("coordinate levels ascend
  // lexicographically"), which admitted one coordinate tuple stored twice. Retired, name
  // and value: never reuse 3, so that no verdict recorded under that rule can ever satisfy
  // the strict check. (A test searches this file for the retired name; keep it out.)
  AbiCheckSpans = 4,       // positions alone: non-decreasing and within [0, nnz]
  AbiCheckLexStrict = 5,   // coordinate levels ascend strictly lexicographically: no
                           // tuple is smaller than, or equal to, the one before it
};

// A verdict that some arrays are structurally valid, together with the shape
// parameters the verdict is conditional on. Comparing `params` is what keeps a
// verdict about (rows, cols) from being reused for a different logical shape over the
// same buffers.
struct AbiStructEntry {
  std::vector<AbiArrayId> arrays;
  std::vector<int64_t> params;

  void add(const torch::Tensor& t) { arrays.push_back(abi_identify(t)); }

  bool live() const {
    for (const auto& a : arrays) {
      if (a.alive.expired()) return false;
    }
    return true;
  }

  bool same(const AbiStructEntry& other) const {
    if (arrays.size() != other.arrays.size() || params != other.params) return false;
    for (size_t i = 0; i < arrays.size(); ++i) {
      if (!abi_same_array(arrays[i], other.arrays[i])) return false;
    }
    return true;
  }
};

inline std::unordered_map<const void*, AbiNarrowEntry>& abi_narrow_map() {
  static std::unordered_map<const void*, AbiNarrowEntry> m;
  return m;
}

// Verdicts are keyed on the first array's address AND the check family. With the address
// alone, two families whose first array is the same buffer evicted each other on every
// call: a COO operand's level-0 bounds verdict and its lexicographic verdict are both
// keyed on level 0's coordinates, so each store overwrote the other and both scans ran
// on every call.
struct AbiStructKey {
  const void* data;
  int64_t kind;
  bool operator==(const AbiStructKey& other) const {
    return data == other.data && kind == other.kind;
  }
};

struct AbiStructKeyHash {
  size_t operator()(const AbiStructKey& key) const {
    return std::hash<const void*>()(key.data) ^ (std::hash<int64_t>()(key.kind) << 1);
  }
};

inline AbiStructKey abi_struct_key(const AbiStructEntry& entry) {
  return {entry.arrays[0].data, entry.params.empty() ? 0 : entry.params[0]};
}

inline std::unordered_map<AbiStructKey, AbiStructEntry, AbiStructKeyHash>& abi_struct_map() {
  static std::unordered_map<AbiStructKey, AbiStructEntry, AbiStructKeyHash> m;
  return m;
}

// When a memo map next sweeps out the entries whose arrays have died, and what that sweep left.
//
// Every insert used to sweep the whole map. An insert happens once per newly seen array, which
// was taken to keep the sweep off any hot path; but a program that keeps many structures alive
// and meets each for the first time in turn -- a dataset of graphs in its first epoch -- inserts
// on every call, and each insert then cost O(live entries): on the M5, 2000 new CSR tensors kept
// alive while each was multiplied once took 64-78 us a call against 25 us when each died after
// its call, and 23-27 us with the memo off.
//
// So a sweep runs once the inserts since the last one reach the entries that sweep left (at
// least 32): an insert costs amortized O(1), and dead entries never outnumber live ones by more
// than that. A narrowed copy is O(nnz) memory, which is why every insert used to sweep: dead
// copies of graph-scale arrays must not pile up. The narrowing memo therefore also sweeps once
// the bytes it has copied since the last sweep reach the live copies' bytes that sweep left (at
// least kAbiMemoSweepBytes): the dead copies it holds never exceed the live ones' bytes, or that
// floor, plus the newest. A program that replaces one large structure per call holds at most
// one dead copy, as before.
constexpr int64_t kAbiMemoSweepBytes = 1 << 20;

struct AbiMemoSweep {
  size_t live = 0;          // entries the last sweep left
  size_t inserted = 0;      // inserts since
  int64_t live_bytes = 0;   // bytes those entries own
  int64_t inserted_bytes = 0;
};

inline int64_t abi_entry_bytes(const AbiNarrowEntry& e) {
  return e.narrowed.defined() ? (int64_t)e.narrowed.nbytes() : 0;
}
inline int64_t abi_entry_bytes(const AbiStructEntry&) { return 0; }

// Called before every insert, with the bytes the new entry owns. Wholesale clearing remains
// the backstop for a program holding more LIVE arrays than SCORCH_ABI_MEMO_MAX: it applies
// only to what a sweep leaves.
template <typename Map>
inline void abi_bound_map(Map& map, AbiMemoSweep& sweep, int64_t incoming_bytes) {
  sweep.inserted += 1;
  sweep.inserted_bytes += incoming_bytes;
  const bool due =
      sweep.inserted >= std::max<size_t>(sweep.live, 32) ||
      (incoming_bytes > 0 &&
       sweep.inserted_bytes >= std::max<int64_t>(sweep.live_bytes, kAbiMemoSweepBytes)) ||
      map.size() >= (size_t)SCORCH_ABI_MEMO_MAX;
  if (!due) return;
  int64_t live_bytes = 0;
  for (auto it = map.begin(); it != map.end();) {
    if (!it->second.live()) {
      it = map.erase(it);
    } else {
      live_bytes += abi_entry_bytes(it->second);
      ++it;
    }
  }
  if (map.size() >= (size_t)SCORCH_ABI_MEMO_MAX) {
    map.clear();
    live_bytes = 0;
  }
  sweep = AbiMemoSweep{map.size(), 0, live_bytes, 0};
}

inline AbiMemoSweep& abi_narrow_sweep() {
  static AbiMemoSweep s;
  return s;
}

inline AbiMemoSweep& abi_struct_sweep() {
  static AbiMemoSweep s;
  return s;
}

// The int32 copy of this array, if one was already made and both still exist
// unchanged.
inline bool abi_narrow_lookup(const AbiArrayId& want, torch::Tensor& out) {
  if (!abi_memo_enabled()) return false;
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  auto& map = abi_narrow_map();
  auto it = map.find(want.data);
  if (it == map.end()) return false;
  if (!it->second.live()) {  // stale: that storage is gone
    map.erase(it);
    return false;
  }
  if (!abi_same_array(it->second.source, want)) return false;
  out = it->second.narrowed;
  return true;
}

inline void abi_narrow_store(AbiArrayId source, torch::Tensor narrowed) {
  if (!abi_memo_enabled()) return;
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  auto& map = abi_narrow_map();
  abi_bound_map(map, abi_narrow_sweep(),
                narrowed.defined() ? (int64_t)narrowed.nbytes() : 0);
  const void* key = source.data;
  AbiNarrowEntry entry;
  entry.source = std::move(source);
  entry.narrowed = std::move(narrowed);
  map.insert_or_assign(key, std::move(entry));
}

// True when exactly these arrays, under exactly these parameters, have already been
// validated and still exist unchanged.
inline bool abi_struct_hit(const AbiStructEntry& want) {
  if (!abi_memo_enabled() || want.arrays.empty()) return false;
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  auto& map = abi_struct_map();
  auto it = map.find(abi_struct_key(want));
  if (it == map.end()) return false;
  if (!it->second.live()) {  // stale: that storage is gone
    map.erase(it);
    return false;
  }
  return it->second.same(want);
}

inline void abi_struct_store(AbiStructEntry entry) {
  if (!abi_memo_enabled() || entry.arrays.empty()) return;
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  auto& map = abi_struct_map();
  abi_bound_map(map, abi_struct_sweep(), 0);
  const AbiStructKey key = abi_struct_key(entry);
  map.insert_or_assign(key, std::move(entry));
}

// A panel's positions, rebased to start at 0, made once per panel and array version: the
// validator's verdict and the narrowing of these positions are memoized on the identity of the
// array they are handed, so a fresh array every call would rescan every panel's coordinates on
// every call. Kept apart from the narrowing memo, which also maps an array's address to a
// derived array: a view starting at the same interior element would otherwise be served the
// rebased copy where it asked for a plain one.
struct AbiPanelPositions {
  AbiArrayId source;  // the slice pos[r0..r1] of the caller's array
  int64_t base = 0;   // pos[r0]
  torch::Tensor rebased;

  bool live() const { return !source.alive.expired(); }
};

inline int64_t abi_entry_bytes(const AbiPanelPositions& e) {
  return e.rebased.defined() ? (int64_t)e.rebased.nbytes() : 0;
}

inline std::unordered_map<const void*, AbiPanelPositions>& abi_panel_positions_map() {
  static std::unordered_map<const void*, AbiPanelPositions> m;
  return m;
}

inline AbiMemoSweep& abi_panel_positions_sweep() {
  static AbiMemoSweep s;
  return s;
}

// Drops every cached verdict, narrowed copy and rebased panel this module can reach.
// Exposed so tests can force the cold path and so a caller under memory pressure can
// reclaim the copies.
//
// How many memos a process has depends on the platform, and nothing may rely on either
// answer. The maps are function-local statics of inline functions. Under gcc on Linux
// such a static is a GNU unique symbol, which the dynamic linker binds to one object for
// the whole process whatever the dlopen flags: there, ONE memo per process, shared by
// the prebuilt extension and every JIT-compiled kernel module (nm -D shows `u` for these
// maps in scorch_ops and in a kernel module; measured 2026-10-12). Where the loader
// keeps the statics apart, each module has its own, and a tensor used on both paths is
// narrowed once per module. macOS has not been measured.
//
// With one memo per process, a generated kernel never rescans an operand that STensor
// construction already scanned through the extension: it finds that verdict. So the
// trade documented under "Validation memo" reaches past a kernel's own calls: a write
// through a raw buffer that does not move the version counter (numpy) is accepted by
// every kernel, one never called on that operand included, until a torch write or a
// write-witness epoch change invalidates the verdict. Sharing is sound only because a
// verdict is specific to its rule: the key and the match carry the check kind and its
// parameters (AbiCheckKind), so a weaker check's verdict cannot satisfy a stricter one.
inline void abi_memo_clear() {
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  abi_narrow_map().clear();
  abi_struct_map().clear();
  abi_panel_positions_map().clear();
  abi_narrow_sweep() = AbiMemoSweep{};
  abi_struct_sweep() = AbiMemoSweep{};
  abi_panel_positions_sweep() = AbiMemoSweep{};
}

// A construction-time screen (storage.py) consulted through the same verdicts as the
// native call boundary. `want` names the arrays and parameters in the layout the boundary
// uses for that check family, so a verdict reached at either place serves the other, and
// a structure rebuilt around index arrays already validated at their current version --
// a torch CSR tensor made afresh each call from the same arrays, say -- is not rescanned.
// `screen` returns true when it found something or cannot tell; only a clean result is
// stored.
template <typename Screen>
inline bool abi_screen_through_memo(AbiStructEntry want, Screen&& screen) {
  if (abi_struct_hit(want)) return false;
  if (screen()) return true;
  abi_struct_store(std::move(want));
  return false;
}

inline size_t abi_memo_size() {
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  return abi_narrow_map().size() + abi_struct_map().size();
}

// The narrowing memo's entries and the bytes their int32 copies own, live or not. For tests.
inline std::pair<int64_t, int64_t> abi_narrow_held() {
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  int64_t bytes = 0;
  for (const auto& kv : abi_narrow_map()) bytes += abi_entry_bytes(kv.second);
  return {(int64_t)abi_narrow_map().size(), bytes};
}

// ---------------------------------------------------------------------------------------------
// CSR operands whose offsets exceed int32.
//
// Every native kernel takes int32 offsets, so a CSR operand with more than INT32_MAX nonzeros
// was refused at this boundary ("positions element ... cannot be represented as int32") while
// torch.sparse.mm computed it. Six SuiteSparse matrices are that large: com-Friendster,
// GAP-kron, GAP-urand, mycielskian20, MOLIERE_2016 and AGATHA_2015. Such an operand is a
// sequence of row panels whose own offsets fit: panel [r0, r1) has the positions
// pos[r0..r1] - pos[r0] and the coordinates and values in [pos[r0], pos[r1]). Each panel goes
// through the ordinary validation and kernel, and its rows are copied into the output. Only the
// offsets need this: columns are an int32 extent, so the coordinates narrow as they always have.
//
// The bound is INT32_MAX. abi_csr_panel_limit lets the tests lower it, so the panelled path runs
// on matrices small enough to build; with it at INT32_MAX no operand an int32 kernel could take
// is ever split.
inline int64_t& abi_csr_panel_limit() {
  static int64_t limit = std::numeric_limits<int32_t>::max();
  return limit;
}

// Calls that ran in panels, for tests that must show the panelled path is the one they ran.
inline std::atomic<int64_t>& abi_csr_panelled_calls() {
  static std::atomic<int64_t> n{0};
  return n;
}

// Whether this CSR operand's last offset -- its nonzero count -- is past the bound. One load,
// taken before validation: anything this cannot read (a malformed level list, a non-CPU or
// non-integer array) answers false and is left to the validator to refuse as before.
inline bool abi_csr_needs_row_panels(const std::vector<std::vector<torch::Tensor>>& levels) {
  if (levels.size() != 2 || levels[1].size() != 2) return false;
  const torch::Tensor& pos = levels[1][0];
  if (!pos.defined() || pos.dim() != 1 || pos.numel() < 1 || !pos.device().is_cpu() ||
      pos.layout() != c10::kStrided)
    return false;
  const int64_t at = (pos.numel() - 1) * pos.stride(0);
  int64_t last;
  if (pos.scalar_type() == torch::kInt64) last = pos.data_ptr<int64_t>()[at];
  else if (pos.scalar_type() == torch::kInt32) last = pos.data_ptr<int32_t>()[at];
  else return false;
  return last > abi_csr_panel_limit();
}

inline torch::Tensor abi_panel_positions(const torch::Tensor& slice, int64_t base) {
  const AbiArrayId want = abi_identify(slice);
  {
    std::lock_guard<std::mutex> guard(abi_memo_mutex());
    auto& map = abi_panel_positions_map();
    auto it = map.find(want.data);
    if (it != map.end() && it->second.live() && it->second.base == base &&
        abi_same_array(it->second.source, want))
      return it->second.rebased;
  }
  torch::Tensor rebased = slice - base;  // same dtype; the validator range-checks it
  if (!abi_memo_enabled()) return rebased;
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  auto& map = abi_panel_positions_map();
  abi_bound_map(map, abi_panel_positions_sweep(), (int64_t)rebased.nbytes());
  AbiPanelPositions entry;
  entry.source = want;
  entry.base = base;
  entry.rebased = rebased;
  map.insert_or_assign(want.data, std::move(entry));
  return rebased;
}

// Runs `run(result_shape, a_shape, a_levels, a_values, into)` -- the ordinary validation and
// kernel of a CSR x dense product, writing the product's rows into `into` -- on each row panel of
// A, and returns the output's flat values, which the panels fill in place. The checks here are the
// ones a panel's own validation cannot see: that the panels tile A (positions length, first and
// last offset, nondecreasing at every cut). Everything inside a panel is the validator's.
template <typename Run>
inline torch::Tensor abi_csr_row_panels(const char* op, const std::vector<int64_t>& result_shape,
                                        const std::vector<int64_t>& a_shape,
                                        const std::vector<std::vector<torch::Tensor>>& a_levels,
                                        const torch::Tensor& a_values, Run&& run) {
  TORCH_CHECK(a_shape.size() == 2 && result_shape.size() == 2, op,
              ": A run as row panels: A and the result must be 2-D");
  const torch::Tensor& raw_pos = a_levels[1][0];
  const torch::Tensor& crd = a_levels[1][1];
  TORCH_CHECK(raw_pos.scalar_type() == crd.scalar_type(), op,
              ": A run as row panels: positions and coordinates must use one common index dtype");
  const int64_t rows = a_shape[0], k = result_shape[1];
  TORCH_CHECK(raw_pos.numel() == rows + 1, op, ": A run as row panels: positions length must be rows + 1 (",
              rows + 1, "), got ", raw_pos.numel());
  const torch::Tensor pos = raw_pos.contiguous();
  const bool wide = pos.scalar_type() == torch::kInt64;
  const int64_t* p64 = wide ? pos.data_ptr<int64_t>() : nullptr;
  const int32_t* p32 = wide ? nullptr : pos.data_ptr<int32_t>();
  auto P = [&](int64_t i) -> int64_t { return wide ? p64[i] : (int64_t)p32[i]; };
  const int64_t nnz = P(rows);
  TORCH_CHECK(P(0) == 0, op, ": A run as row panels: positions[0] must be 0, got ", P(0));
  TORCH_CHECK(crd.numel() == nnz && a_values.numel() == nnz, op,
              ": A run as row panels: terminal position must equal the coordinate and value counts (", nnz,
              " vs ", crd.numel(), " and ", a_values.numel(), ")");

  const int64_t limit = abi_csr_panel_limit();
  torch::Tensor out = torch::empty({rows * k}, a_values.options());
  for (int64_t r0 = 0; r0 < rows;) {
    // The last row whose offset is within `limit` of this panel's first. Binary search assumes
    // nondecreasing positions; where they are not, the cut below or the panel's validator says
    // so, and a cut always advances by at least one row.
    const int64_t base = P(r0);
    int64_t lo = r0 + 1, hi = rows;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo + 1) / 2;
      if (P(mid) - base <= limit) lo = mid;
      else hi = mid - 1;
    }
    const int64_t r1 = lo, end = P(r1);
    TORCH_CHECK(end >= base, op, ": A run as row panels: positions must be nondecreasing; positions[", r1,
                "] = ", end, " < positions[", r0, "] = ", base);
    std::vector<std::vector<torch::Tensor>> levels{
        {}, {abi_panel_positions(pos.narrow(0, r0, r1 - r0 + 1), base),
             crd.narrow(0, base, end - base)}};
    torch::Tensor into = out.narrow(0, r0 * k, (r1 - r0) * k);
    run(std::vector<int64_t>{r1 - r0, result_shape[1]}, std::vector<int64_t>{r1 - r0, a_shape[1]},
        std::move(levels), a_values.narrow(0, base, end - base), into);
    r0 = r1;
  }
  abi_csr_panelled_calls().fetch_add(1, std::memory_order_relaxed);
  return out;
}

}  // namespace scorch_native

// Checked input views shared by the prebuilt extension and JIT-generated kernels.
//
// All Scorch native kernels are CPU kernels and the legacy implementations use
// int32_t level arrays.  The helpers below validate the complete public ABI before
// any kernel indexes a C++ container or dereferences a pointer.  int64 index tensors
// are accepted, range checked, and converted in the *by-value* mode_indices argument
// owned by the native call; caller tensors and Python lists are never modified.
namespace scorch_native {

enum class BinaryContract {
  CsrDenseMatmul,
  CsrCsrMatmul,
  CsrDenseMatvec,
  CooDenseMatmul,
  CooCooMatmul,
  // CsrDenseMatmul into a kernel that addresses B and the result with 64-bit offsets (the v2
  // SpMM and its tiled forms), so their element counts are not held to int32; every extent still
  // is, since the kernels take each one as an int. See the note at checked_dense_view.
  CsrDenseMatmulInt64Elements,
};

// A result's levels as its entry check sees them. DenseBoundedInKernel is a dense level below a
// sparse one whose block the kernel itself holds to the position bound where the block opens
// (scorch_check_dense_block, header.h), so the entry check does not count the sparse levels
// above it by their extents (check_result_positions).
enum class LevelKind : int {
  Dense = 0,
  Compressed = 1,
  Coordinate = 2,
  DenseBoundedInKernel = 3,
};

inline std::string argument_name(const char* op, const char* argument) {
  return std::string(op) + ": " + argument;
}

// The 32-bit position bound.
//
// Every kernel declares a level's positions, its loop variables and its coordinates as `int`.
// What a tensor may hold is therefore bounded per storage level, by the positions that level
// stores, and not by the tensor's logical element count: a 64,000 x 64,000 CSR matrix stores
// 64,001 row offsets and its nonzeros, and no kernel forms rows x columns for it. The bound
// on a logical element count belongs to the kernels that address a dense array with `int`
// offsets, and to no others.
// A refusal under this bound ends with the first code below, and a prebuilt kernel's refusal of
// operands that give one index two extents with the second. scorch.exceptions.raise_if_native_coded
// turns each into its error at every Python call site of a kernel: change both or neither.
#define SCORCH_INDEX_CAPACITY_CODE " [index_position_exceeds_int32]"
#define SCORCH_EXTENT_MISMATCH_CODE " [einsum_index_extent_mismatch]"
[[noreturn]] inline void refuse_contraction_mismatch(const char*, int64_t, int64_t);
constexpr int64_t kMaxIndexPosition = std::numeric_limits<int>::max();

#if defined(__GNUC__) || defined(__clang__)
#define SCORCH_COLD_NOINLINE __attribute__((noinline, cold))
#define SCORCH_NOINLINE __attribute__((noinline))
#else
#define SCORCH_COLD_NOINLINE
#define SCORCH_NOINLINE
#endif

// The refusal a kernel's own growth raises (header.h): kept out of line and cold so that the
// append it guards carries one compare and no message.
[[noreturn]] SCORCH_COLD_NOINLINE inline void refuse_index_capacity(const char* what,
                                                                   int64_t count) {
  TORCH_CHECK(false, what, " would hold ", count, " entries; a 32-bit position addresses ",
              "at most ", kMaxIndexPosition, SCORCH_INDEX_CAPACITY_CODE);
  std::abort();  // not reached: TORCH_CHECK(false) throws
}

// The refusal of a result's dense block that would pass the bound where it opens
// (scorch_check_dense_block, header.h): out of line and cold, as above.
[[noreturn]] SCORCH_COLD_NOINLINE inline void refuse_dense_block_positions(
    int64_t parent_position, int64_t block, bool compressed_below, int first_level,
    int last_level) {
  TORCH_CHECK(false, "result dense levels ", first_level, " to ", last_level,
              ": the block of ", block, " cells under position ", parent_position,
              " of compressed level ", first_level - 1, " would end at position ",
              (parent_position + 1) * block,
              compressed_below ? ", and compressed level " : "",
              compressed_below ? std::to_string(last_level + 1) : std::string(),
              compressed_below ? " below it would have one position entry more" : "",
              "; a 32-bit position addresses at most ", kMaxIndexPosition, " entries",
              SCORCH_INDEX_CAPACITY_CODE);
  std::abort();  // not reached: TORCH_CHECK(false) throws
}

// The refusal of a compressed level that the two-pass assembly counted past the bound
// (scorch_check_level_total, header.h): out of line and cold, as above.
[[noreturn]] SCORCH_COLD_NOINLINE inline void refuse_level_total(int64_t total,
                                                                bool compressed_below,
                                                                int level) {
  TORCH_CHECK(false, "result compressed level ", level, " would hold ", total, " entries",
              compressed_below ? ", and compressed level " : "",
              compressed_below ? std::to_string(level + 1) : std::string(),
              compressed_below ? " below it one position entry more" : "",
              "; a 32-bit position addresses at most ", kMaxIndexPosition, " entries",
              SCORCH_INDEX_CAPACITY_CODE);
  std::abort();  // not reached: TORCH_CHECK(false) throws
}

// The refusal of a sparse workspace drained into more entries than the kernel can address
// when it reads them back (scorch_check_drained_entries, header.h): out of line and cold.
[[noreturn]] SCORCH_COLD_NOINLINE inline void refuse_drained_entries(int64_t entries) {
  TORCH_CHECK(false, "a sparse workspace drained ", entries, " entries, which the kernel ",
              "reads back with 32-bit positions; a 32-bit position addresses at most ",
              kMaxIndexPosition, " entries", SCORCH_INDEX_CAPACITY_CODE);
  std::abort();  // not reached: TORCH_CHECK(false) throws
}

inline std::string shape_string(const std::vector<int64_t>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) result += ", ";
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

template <typename Dim>
inline std::vector<int64_t> checked_shape(const std::vector<Dim>& shape,
                                          int64_t expected_rank,
                                          const char* op,
                                          const char* argument) {
  TORCH_CHECK(static_cast<int64_t>(shape.size()) == expected_rank,
              argument_name(op, argument), " must have rank ", expected_rank,
              ", got ", shape.size());
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = static_cast<int64_t>(shape[i]);
    TORCH_CHECK(dim >= 0, argument_name(op, argument), " has negative dimension ",
                dim, " at mode ", i);
    TORCH_CHECK(dim <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " dimension ", i,
                " exceeds the current native int32 loop bound: ", dim,
                SCORCH_INDEX_CAPACITY_CODE);
    result.push_back(dim);
  }
  return result;
}

template <typename Dim>
inline std::vector<int> narrow_legacy_shape(const std::vector<Dim>& shape,
                                            const char* op,
                                            const char* argument) {
  std::vector<int> result;
  result.reserve(shape.size());
  for (size_t mode = 0; mode < shape.size(); ++mode) {
    const int64_t extent = static_cast<int64_t>(shape[mode]);
    TORCH_CHECK(extent >= 0 && extent <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " extent ", extent, " at mode ",
                mode, " cannot be represented by the legacy int32 kernel ABI",
                SCORCH_INDEX_CAPACITY_CODE);
    result.push_back(static_cast<int>(extent));
  }
  return result;
}

inline int64_t checked_product(const std::vector<int64_t>& shape,
                               const char* op, const char* argument,
                               bool require_legacy_int_capacity = false) {
  int64_t product = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t dim = shape[i];
    TORCH_CHECK(dim == 0 ||
                    product <= std::numeric_limits<int64_t>::max() / dim,
                argument_name(op, argument), " element count overflows int64 at mode ",
                i);
    product *= dim;
  }
  if (require_legacy_int_capacity) {
    TORCH_CHECK(product <= std::numeric_limits<int>::max(),
                argument_name(op, argument), " element count ", product,
                " exceeds the current native int32 allocation bound",
                SCORCH_INDEX_CAPACITY_CODE);
  }
  return product;
}

// A shape a kernel was compiled for, or any shape of the kernel's rank when it was compiled
// for none. Every extent is held to `int` (checked_shape); the element count is not looked
// at, which is what a tensor with a sparse level needs: its logical size bounds nothing.
template <typename Dim>
inline std::vector<int64_t> checked_compiled_shape(
    const std::vector<Dim>& shape, const std::vector<int64_t>& expected,
    int64_t rank, const char* op, const char* argument) {
  if (expected.empty() && rank > 0) {
    return checked_shape(shape, rank, op, argument);
  }
  const auto actual = checked_shape(shape, static_cast<int64_t>(expected.size()),
                                    op, argument);
  TORCH_CHECK(actual == expected, argument_name(op, argument), " must equal ",
              shape_string(expected), ", got ", shape_string(actual));
  return actual;
}

template <typename Dim>
inline std::vector<int64_t> checked_expected_shape(
    const std::vector<Dim>& shape, const std::vector<int64_t>& expected,
    const char* op, const char* argument,
    bool require_legacy_int_capacity = false) {
  const auto actual = checked_compiled_shape(shape, expected, 0, op, argument);
  checked_product(actual, op, argument, require_legacy_int_capacity);
  return actual;
}

// The positions a kernel can form at a dense level of `extent` whose parent level has
// `parent_positions` of them: parent position x extent + coordinate, for every parent
// position and every coordinate. All of them must fit `int`, and so must the count itself,
// which a kernel forms as the end of the last parent's block.
//
// `parent_positions` is 1 at the root, the product so far under dense levels, and the
// stored entries + 1 under a compressed or coordinate level. The + 1 is the level's end
// position: a kernel that merges two operands leaves a finished operand's cursor there and
// may still compute the dense position below it (never reading through it), so that
// product must not overflow either.
inline int64_t checked_dense_positions(int64_t parent_positions, int64_t extent,
                                       size_t level, const char* op,
                                       const char* argument) {
  TORCH_CHECK(extent == 0 || parent_positions <= kMaxIndexPosition / extent,
              argument_name(op, argument), " dense level ", level, " of extent ",
              extent, " under ", parent_positions,
              " parent positions needs more than ", kMaxIndexPosition,
              " positions, the most a 32-bit position addresses",
              SCORCH_INDEX_CAPACITY_CODE);
  return parent_positions * extent;
}

inline void check_tensor_common(const torch::Tensor& tensor,
                                torch::ScalarType expected_dtype,
                                const char* op, const char* argument,
                                int64_t expected_rank = -1) {
  TORCH_CHECK(tensor.defined(), argument_name(op, argument), " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), argument_name(op, argument),
              " must be on CPU, got ", tensor.device());
  TORCH_CHECK(tensor.layout() == torch::kStrided,
              argument_name(op, argument), " must use strided storage");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype,
              argument_name(op, argument), " must have dtype ",
              c10::toString(expected_dtype), ", got ",
              c10::toString(tensor.scalar_type()));
  TORCH_CHECK(tensor.is_contiguous(), argument_name(op, argument),
              " must be contiguous");
  TORCH_CHECK(!tensor.is_neg(), argument_name(op, argument),
              " must resolve its lazy negative view bit");
  TORCH_CHECK(!tensor.is_conj(), argument_name(op, argument),
              " must resolve its lazy conjugate view bit");
  if (expected_rank >= 0) {
    TORCH_CHECK(tensor.dim() == expected_rank, argument_name(op, argument),
                " must have rank ", expected_rank, ", got ", tensor.dim());
  }
}

inline void check_flat_values(const torch::Tensor& values,
                              torch::ScalarType expected_dtype,
                              int64_t expected_numel, const char* op,
                              const char* argument) {
  check_tensor_common(values, expected_dtype, op, argument, 1);
  TORCH_CHECK(values.numel() == expected_numel, argument_name(op, argument),
              " must contain ", expected_numel, " elements, got ",
              values.numel());
  TORCH_CHECK(expected_numel == 0 ||
                  static_cast<uint64_t>(expected_numel) <=
                      std::numeric_limits<size_t>::max() / values.element_size(),
              argument_name(op, argument), " byte size overflows size_t");
}

inline torch::Tensor checked_index_tensor(torch::Tensor index,
                                          const char* op,
                                          const std::string& argument) {
  TORCH_CHECK(index.defined(), op, ": ", argument, " must be defined");
  TORCH_CHECK(index.device().is_cpu(), op, ": ", argument,
              " must be on CPU, got ", index.device());
  TORCH_CHECK(index.layout() == torch::kStrided, op, ": ", argument,
              " must use strided storage");
  TORCH_CHECK(index.dim() == 1, op, ": ", argument, " must be 1-D, got rank ",
              index.dim());
  TORCH_CHECK(index.is_contiguous(), op, ": ", argument,
              " must be contiguous");
  TORCH_CHECK(!index.is_neg(), op, ": ", argument,
              " must resolve its lazy negative view bit");
  TORCH_CHECK(!index.is_conj(), op, ": ", argument,
              " must resolve its lazy conjugate view bit");
  TORCH_CHECK(index.scalar_type() == torch::kInt32 ||
                  index.scalar_type() == torch::kInt64,
              op, ": ", argument, " must have dtype int32 or int64, got ",
              c10::toString(index.scalar_type()));
  TORCH_CHECK(index.numel() <= std::numeric_limits<int>::max(), op, ": ",
              argument, " length exceeds the current int32 native bound",
              SCORCH_INDEX_CAPACITY_CODE);

  if (index.scalar_type() == torch::kInt64) {
    // Already narrowed this array? Then both the scan and the allocate-and-fill below
    // are re-deriving a known answer, and — because the answer is the same tensor
    // rather than a fresh copy — every structural memo downstream of it can hit.
    const AbiArrayId want = abi_identify(index);
    torch::Tensor cached;
    if (abi_narrow_lookup(want, cached)) return cached;

    const int64_t* data = index.data_ptr<int64_t>();
    const int64_t n = index.numel();
    // Screen: "every element is int32-representable" is exactly "min >= INT32_MIN
    // and max <= INT32_MAX", so one branchless min/max reduction replaces n
    // TORCH_CHECKs. 0 is representable, so it is a safe reduction identity.
    const std::pair<int64_t, int64_t> range = at::parallel_reduce(
        (int64_t)0, n, abi_split(n, SCORCH_ABI_VALIDATE_GRAIN),
        std::pair<int64_t, int64_t>(0, 0),
        [&](int64_t begin, int64_t end, std::pair<int64_t, int64_t> partial) {
          for (int64_t i = begin; i < end; ++i) {
            const int64_t v = data[i];
            if (v < partial.first) partial.first = v;
            if (v > partial.second) partial.second = v;
          }
          return partial;
        },
        [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) {
          return std::pair<int64_t, int64_t>(std::min(a.first, b.first),
                                            std::max(a.second, b.second));
        });
    const int64_t lo = range.first, hi = range.second;
    if (lo < (int64_t)std::numeric_limits<int32_t>::min() ||
        hi > (int64_t)std::numeric_limits<int32_t>::max()) {
      // Re-walk serially so the message still names the FIRST offending element.
      for (int64_t i = 0; i < n; ++i) {
        TORCH_CHECK(data[i] >= std::numeric_limits<int32_t>::min() &&
                        data[i] <= std::numeric_limits<int32_t>::max(),
                    op, ": ", argument, " element ", i, " (", data[i],
                    ") cannot be represented as int32");
      }
    }
    index = index.to(torch::kInt32);
    abi_narrow_store(want, index);
  }
  return index;
}

inline void check_common_index_dtype(const torch::Tensor& index,
                                     bool& has_index_dtype,
                                     torch::ScalarType& index_dtype,
                                     const char* op,
                                     const std::string& argument) {
  if (!has_index_dtype) {
    index_dtype = index.scalar_type();
    has_index_dtype = true;
    return;
  }
  TORCH_CHECK(index.scalar_type() == index_dtype, op, ": ", argument,
              " must use the common index dtype ", c10::toString(index_dtype),
              ", got ", c10::toString(index.scalar_type()));
}

struct DenseInputView {
  std::vector<int64_t> shape;
  torch::Tensor values;
  const void* data;

  DenseInputView(std::vector<int64_t> shape_, torch::Tensor values_)
      : shape(std::move(shape_)),
        values(std::move(values_)),
        data(values.data_ptr()) {}
};

struct CsrInputView {
  std::vector<int64_t> shape;
  torch::Tensor positions;
  torch::Tensor coordinates;
  torch::Tensor values;
  const int32_t* pos;
  const int32_t* crd;
  const void* data;
  int64_t nnz;

  CsrInputView(std::vector<int64_t> shape_, torch::Tensor positions_,
               torch::Tensor coordinates_, torch::Tensor values_)
      : shape(std::move(shape_)),
        positions(std::move(positions_)),
        coordinates(std::move(coordinates_)),
        values(std::move(values_)),
        pos(positions.data_ptr<int32_t>()),
        crd(coordinates.data_ptr<int32_t>()),
        data(values.data_ptr()),
        nnz(values.numel()) {}
};

struct CooInputView {
  std::vector<int64_t> shape;
  std::vector<torch::Tensor> coordinates;
  torch::Tensor values;
  std::vector<const int32_t*> crd;
  const void* data;
  int64_t nnz;

  CooInputView(std::vector<int64_t> shape_,
               std::vector<torch::Tensor> coordinates_,
               torch::Tensor values_)
      : shape(std::move(shape_)),
        coordinates(std::move(coordinates_)),
        values(std::move(values_)),
        data(values.data_ptr()),
        nnz(values.numel()) {
    crd.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
      crd.push_back(coordinate.data_ptr<int32_t>());
    }
  }
};

template <typename Dim>
inline DenseInputView checked_dense_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    int64_t expected_rank, const char* op, const char* argument,
    bool int32_elements = true) {
  // `int32_elements` is false only for a kernel that addresses the array with 64-bit offsets.
  // The legacy kernels compute element offsets in int, so for them an array past INT32_MAX
  // elements is refused here. The v2 SpMM computes every offset into B and the result as
  // (size_t)row * (size_t)width, and refusing a B or a result of more than INT32_MAX elements
  // made scorch.matmul raise on products torch.sparse.mm computes: a 9.9M-row matrix at k=256
  // has a 2.5e9-element result.
  const auto logical_shape = checked_shape(shape, expected_rank, op, argument);
  const int64_t numel = checked_product(logical_shape, op, argument, int32_elements);
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " mode_indices must contain ",
              logical_shape.size(), " levels, got ", mode_indices.size());
  for (size_t level = 0; level < mode_indices.size(); ++level) {
    TORCH_CHECK(mode_indices[level].empty(), argument_name(op, argument),
                " dense level ", level, " must not contain index tensors");
  }
  check_flat_values(values, expected_dtype, numel, op,
                    (std::string(argument) + " values").c_str());
  return DenseInputView(logical_shape, values);
}

// The int32 copies the native kernels read, made for int64 index arrays seen for the first time
// in the pass that checks their structure (abi_screen_compressed_typed), in place of a range
// pass and a copy for each array (checked_index_tensor) followed by a structural scan of the
// copies. A clean pass leaves what those would have left -- each copy in the narrowing memo, the
// verdict for the copies in the structural memo -- and the verdict for the int64 arrays, so the
// checks that follow find all three and do no further work. A pair whose structure is already
// known (the construction-time screen, storage.py, records it under the same parameters) is
// only copied. Anything else -- either array already narrowed, a precondition of those checks
// unmet, a pass that is not clean -- leaves the memos as they were, and the checks run as they
// always have and report what they always have.
inline void abi_narrow_csr_first_sight(const torch::Tensor& positions,
                                       const torch::Tensor& coordinates,
                                       const torch::Tensor& values, int64_t rows,
                                       int64_t cols, bool require_sorted) {
  if (!abi_memo_enabled()) return;
  const auto plain = [](const torch::Tensor& t) {
    return t.defined() && t.device().is_cpu() && t.layout() == torch::kStrided &&
           t.dim() == 1 && t.is_contiguous() && !t.is_neg() && !t.is_conj() &&
           t.scalar_type() == torch::kInt64;
  };
  if (!plain(positions) || !plain(coordinates) || !values.defined()) return;
  const int64_t nnz = values.numel();
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  if (rows < 0 || cols < 0 || cols > kMax || nnz > kMax) return;
  if (positions.numel() != rows + 1 || coordinates.numel() != nnz) return;
  const int64_t* pos = positions.data_ptr<int64_t>();
  const int64_t* crd = coordinates.data_ptr<int64_t>();
  if (pos[0] != 0 || pos[rows] != nnz) return;

  AbiArrayId pos_id = abi_identify(positions);
  AbiArrayId crd_id = abi_identify(coordinates);
  torch::Tensor narrowed;
  if (abi_narrow_lookup(pos_id, narrowed) || abi_narrow_lookup(crd_id, narrowed)) return;

  const std::vector<int64_t> params = {AbiCheckCompressed, rows, cols, nnz,
                                       require_sorted ? 1 : 0};
  AbiStructEntry source;
  source.add(coordinates);
  source.add(positions);
  source.params = params;
  torch::Tensor pos32 = torch::empty({rows + 1}, positions.options().dtype(torch::kInt32));
  torch::Tensor crd32 = torch::empty({nnz}, coordinates.options().dtype(torch::kInt32));
  if (abi_struct_hit(source)) {
    pos32.copy_(positions);
    crd32.copy_(coordinates);
  } else {
    // At the grain of ATen's own copy, which this pass replaces: it writes every element.
    if (abi_screen_compressed_typed<int64_t, int32_t>(
            pos, crd, rows, nnz, cols, require_sorted, at::internal::GRAIN_SIZE,
            pos32.data_ptr<int32_t>(), crd32.data_ptr<int32_t>()))
      return;
    abi_struct_store(std::move(source));
  }
  AbiStructEntry copies;
  copies.add(crd32);
  copies.add(pos32);
  copies.params = params;
  abi_narrow_store(std::move(pos_id), pos32);
  abi_narrow_store(std::move(crd_id), crd32);
  abi_struct_store(std::move(copies));
}

// Whether a CSR operand over these arrays is one every check made of it would clear -- the native
// boundary's (checked_csr_view, coordinates strictly increasing within each row, as the drop-in
// SpMM requires)
// and the construction-time screen's (storage.py), which ask the same question -- so that a caller
// can make a call those checks guard knowing none of them will report. The O(1) conditions are
// tested here; the structure through the verdicts both keep, reached by a scan only where neither
// has one yet: for an int64 pair seen for the first time, the pass that also makes the int32
// copies the kernels read (abi_narrow_csr_first_sight); for an int32 pair, the screen at `grain`.
// Never raises and never reports: false means "make the checked call and let it report", as from
// every screen here.
inline bool abi_csr_screen(const torch::Tensor& positions, const torch::Tensor& coordinates,
                           const torch::Tensor& values, int64_t rows, int64_t cols,
                           int64_t grain) {
  const auto plain = [](const torch::Tensor& t) {
    return t.defined() && t.device().is_cpu() && t.layout() == torch::kStrided &&
           t.dim() == 1 && t.is_contiguous() && !t.is_neg() && !t.is_conj();
  };
  if (!plain(positions) || !plain(coordinates) || !plain(values)) return false;
  const auto index_dtype = positions.scalar_type();
  if (coordinates.scalar_type() != index_dtype ||
      (index_dtype != torch::kInt32 && index_dtype != torch::kInt64))
    return false;
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  const int64_t nnz = values.numel();
  if (rows < 0 || cols < 0 || rows > kMax || cols > kMax || nnz > kMax) return false;
  if (positions.numel() != rows + 1 || coordinates.numel() != nnz) return false;
  const std::vector<int64_t> params = {AbiCheckCompressed, rows, cols, nnz, 1};
  if (index_dtype == torch::kInt64) {
    abi_narrow_csr_first_sight(positions, coordinates, values, rows, cols, true);
    torch::Tensor pos32, crd32;
    if (!abi_narrow_lookup(abi_identify(positions), pos32) ||
        !abi_narrow_lookup(abi_identify(coordinates), crd32))
      return false;
    AbiStructEntry copies;
    copies.add(crd32);
    copies.add(pos32);
    copies.params = params;
    return abi_struct_hit(copies);
  }
  const int32_t* pos = positions.data_ptr<int32_t>();
  const int32_t* crd = coordinates.data_ptr<int32_t>();
  if (pos[0] != 0 || pos[rows] != nnz) return false;
  AbiStructEntry want;
  want.add(coordinates);
  want.add(positions);
  want.params = params;
  return !abi_screen_through_memo(std::move(want), [&] {
    return abi_screen_compressed_typed<int32_t>(pos, crd, rows, nnz, cols, true, grain);
  });
}

template <typename Dim>
inline CsrInputView checked_csr_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    const char* op, const char* argument, bool require_sorted = true) {
  const auto logical_shape = checked_shape(shape, 2, op, argument);
  TORCH_CHECK(mode_indices.size() == 2, argument_name(op, argument),
              " CSR mode_indices must contain exactly 2 levels, got ",
              mode_indices.size());
  TORCH_CHECK(mode_indices[0].empty(), argument_name(op, argument),
              " CSR dense level 0 must be empty");
  TORCH_CHECK(mode_indices[1].size() == 2, argument_name(op, argument),
              " CSR compressed level 1 must contain [positions, coordinates]");

  const auto raw_positions = mode_indices[1][0];
  const auto raw_coordinates = mode_indices[1][1];
  abi_narrow_csr_first_sight(raw_positions, raw_coordinates, values, logical_shape[0],
                             logical_shape[1], require_sorted);
  mode_indices[1][0] = checked_index_tensor(
      raw_positions, op, std::string(argument) + " positions");
  mode_indices[1][1] = checked_index_tensor(
      raw_coordinates, op, std::string(argument) + " coordinates");
  TORCH_CHECK(raw_positions.scalar_type() == raw_coordinates.scalar_type(),
              argument_name(op, argument),
              " positions and coordinates must use one common index dtype");
  auto positions = mode_indices[1][0];
  auto coordinates = mode_indices[1][1];

  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);
  const int64_t rows = logical_shape[0];
  const int64_t cols = logical_shape[1];
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), argument_name(op, argument),
              " nnz exceeds the current int32 native bound",
              SCORCH_INDEX_CAPACITY_CODE);
  TORCH_CHECK(coordinates.numel() == nnz, argument_name(op, argument),
              " coordinate/value nnz mismatch: ", coordinates.numel(), " vs ",
              nnz);
  TORCH_CHECK(positions.numel() == rows + 1, argument_name(op, argument),
              " positions length must be rows + 1 (", rows + 1, "), got ",
              positions.numel());

  const int32_t* pos = positions.data_ptr<int32_t>();
  const int32_t* crd = coordinates.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, argument_name(op, argument),
              " positions[0] must be 0, got ", pos[0]);

  // Already validated this exact index pair under this exact shape? Then the O(rows)
  // and O(nnz) structural scans are re-deriving a known answer. Every O(1) check above
  // and below still runs on every call; only the scans are memoized.
  AbiStructEntry memo_want;
  memo_want.add(coordinates);
  memo_want.add(positions);
  memo_want.params = {AbiCheckCompressed, rows, cols, nnz, require_sorted ? 1 : 0};
  const bool memo_valid = abi_struct_hit(memo_want);

  const bool bad = !memo_valid && abi_screen_compressed(pos, crd, rows, nnz, cols,
                                                        require_sorted);
  if (bad) {
    int32_t previous = 0;
    for (int64_t row = 0; row < rows; ++row) {
      const int32_t start = pos[row];
      const int32_t end = pos[row + 1];
      TORCH_CHECK(start >= previous && start >= 0 && end >= start && end <= nnz,
                  argument_name(op, argument), " invalid CSR span for row ", row,
                  ": [", start, ", ", end, ") with nnz ", nnz);
      for (int32_t p = start; p < end; ++p) {
        TORCH_CHECK(crd[p] >= 0 && crd[p] < cols,
                    argument_name(op, argument), " coordinate ", crd[p],
                    " at position ", p, " is outside [0, ", cols, ")");
        if (require_sorted && p > start) {
          TORCH_CHECK(
              crd[p - 1] < crd[p], argument_name(op, argument),
              " coordinates must be strictly increasing within each CSR row; row ",
              row, " is non-increasing at position ", p);
        }
      }
      previous = end;
    }
  }
  TORCH_CHECK(pos[rows] == nnz, argument_name(op, argument),
              " terminal position must equal nnz (", nnz, "), got ", pos[rows]);
  // Record only after every check has passed, so a rejected input is never cached as
  // valid — including the case where the screen flagged nothing but the terminal
  // position check above threw.
  if (!memo_valid) abi_struct_store(std::move(memo_want));
  return CsrInputView(logical_shape, positions, coordinates, values);
}

// The serial check behind abi_screen_lex: run only when the screen says "go and look", so
// that the first violation by position is the one reported, as it always was. A descent and
// a repeat are reported apart. A repeat's text is parsed back into a structured
// RepeatedCoordinateError by scorch.exceptions.raise_if_native_coded at every Python call
// site of a kernel, so its wording is a contract: change both or neither.
inline void check_coo_strictly_ascending(const std::vector<const int32_t*>& levels,
                                         int64_t n, const char* op,
                                         const char* argument) {
  for (int64_t p = 1; p < n; ++p) {
    int cmp = 0;
    for (size_t level = 0; level < levels.size() && cmp == 0; ++level) {
      const int32_t a = levels[level][p - 1], b = levels[level][p];
      cmp = (b > a) - (b < a);
    }
    TORCH_CHECK(cmp >= 0, argument_name(op, argument),
                " COO coordinates must be lexicographically ordered; order ",
                "decreases at position ", p);
    if (cmp == 0) {
      std::string tuple;
      for (size_t level = 0; level < levels.size(); ++level) {
        if (level > 0) tuple += ", ";
        tuple += std::to_string(levels[level][p]);
      }
      TORCH_CHECK(false, argument_name(op, argument), " COO coordinate (", tuple,
                  ") is stored at positions ", p - 1, " and ", p,
                  "; call coalesce() to sum repeated coordinates "
                  "[coo_repeated_coordinate]");
    }
  }
}

template <typename Dim>
inline CooInputView checked_coo_view(
    const std::vector<Dim>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    int64_t expected_rank, const char* op, const char* argument,
    bool require_lexicographic_order = true) {
  const auto logical_shape = checked_shape(shape, expected_rank, op, argument);
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " COO mode_indices must contain ",
              logical_shape.size(), " coordinate levels, got ",
              mode_indices.size());
  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), argument_name(op, argument),
              " nnz exceeds the current int32 native bound",
              SCORCH_INDEX_CAPACITY_CODE);

  std::vector<torch::Tensor> coordinates;
  coordinates.reserve(logical_shape.size());
  bool has_index_dtype = false;
  torch::ScalarType index_dtype = torch::kInt32;
  for (size_t level = 0; level < logical_shape.size(); ++level) {
    TORCH_CHECK(mode_indices[level].size() == 1,
                argument_name(op, argument), " COO level ", level,
                " must contain exactly one coordinate tensor");
    const auto raw_coordinate = mode_indices[level][0];
    mode_indices[level][0] = checked_index_tensor(
        raw_coordinate, op,
        std::string(argument) + " coordinate level " + std::to_string(level));
    check_common_index_dtype(raw_coordinate, has_index_dtype, index_dtype, op,
                             std::string(argument) + " coordinate level " +
                                 std::to_string(level));
    TORCH_CHECK(mode_indices[level][0].numel() == nnz,
                argument_name(op, argument), " COO level ", level,
                " length must equal values nnz (", nnz, "), got ",
                mode_indices[level][0].numel());
    const int32_t* crd = mode_indices[level][0].data_ptr<int32_t>();
    // Memoized per level rather than once for the whole view, so that a rejection is
    // still reported by the same check in the same order as before: an out-of-range
    // coordinate at level 0 must be raised before a dtype problem at level 1.
    AbiStructEntry level_memo;
    level_memo.add(mode_indices[level][0]);
    level_memo.params = {AbiCheckBounds, logical_shape[level], nnz};
    const bool level_known = abi_struct_hit(level_memo);
    if (!level_known && abi_screen_bounds(crd, nnz, logical_shape[level])) {
      for (int64_t p = 0; p < nnz; ++p) {
        TORCH_CHECK(crd[p] >= 0 && crd[p] < logical_shape[level],
                    argument_name(op, argument), " coordinate ", crd[p],
                    " at level ", level, " position ", p, " is outside [0, ",
                    logical_shape[level], ")");
      }
    }
    if (!level_known) abi_struct_store(std::move(level_memo));
    coordinates.push_back(mode_indices[level][0]);
  }

  if (require_lexicographic_order && nnz > 1) {
    AbiStructEntry lex_memo;
    lex_memo.params.push_back(AbiCheckLexStrict);
    std::vector<const int32_t*> crd_levels;
    crd_levels.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
      lex_memo.add(coordinate);
      crd_levels.push_back(coordinate.data_ptr<int32_t>());
    }
    lex_memo.params.push_back(nnz);
    const bool lex_known = abi_struct_hit(lex_memo);
    if (!lex_known && abi_screen_lex(crd_levels, nnz)) {
      check_coo_strictly_ascending(crd_levels, nnz, op, argument);
    }
    if (!lex_known) abi_struct_store(std::move(lex_memo));
  }
  return CooInputView(logical_shape, coordinates, values);
}

// The result's bound that is known before the kernel runs. `result_shape` holds the extents
// by storage level, as an operand's shape does, and `level_kinds` the result's layout.
//
// A kernel addresses a dense level of the result as parent position x extent + coordinate,
// so what must fit `int` is the product of the extents down to the result's deepest dense
// level. For a dense prefix (ds, dss, dds) that is the prefix; a result with no dense level
// (ss, oo) has no bound here at all; and an all-dense result keeps its element count, the
// only case in which the logical size is what a kernel forms. Levels below the deepest
// dense one store what the kernel appends, which is bounded while it runs
// (scorch_vector_set, the chunk merges) and when it returns (the result wrapper, stensor.py).
//
// For a dense level BELOW a sparse one (sd, sdd, dsd) the kernel grows the dense block by
// `resize` and narrows the position that follows to `int`. A kernel that holds each block to
// the bound where it opens (scorch_check_dense_block, header.h) declares the block's levels
// DenseBoundedInKernel: they bound nothing here, and what is left to check is that one block
// fits at all, which the kernel's first stored coordinate needs. A kernel that does not
// declares them Dense, and the product then counts the sparse levels above by their extents,
// as if every coordinate were stored: more than the kernel can form, and the only bound on
// what it could store that holds before it starts.
//
// A compressed level under those extents has one position entry more than it has parents,
// and the kernel that fills it counts parents with `for (int p = 1; p <= parents; p++)`,
// which does not end at INT32_MAX parents: one less is the bound there, for a block too.
//
// refuse_result_positions is the rule with its refusals worded. result_positions_fit is the
// same rule as a yes or no, decided without building a vector or a message, and
// check_result_positions asks it first: this runs once per kernel call, and a call whose
// result fits pays for no allocation. The two are held to one table and one sweep of shapes
// by tests/test_scorch/test_index_position_bound.py.
// Whether a DenseBoundedInKernel level at `level` sits where a kernel's block can: below a
// compressed level or another level of the same block, and above a compressed level, another
// level of the same block, or nothing. Any other place is a malformed contract.
inline bool bounded_in_kernel_level_in_place(const int* level_kinds, size_t level_count,
                                             size_t level) noexcept {
  if (level == 0) return false;
  const auto above = static_cast<LevelKind>(level_kinds[level - 1]);
  if (above != LevelKind::Compressed && above != LevelKind::DenseBoundedInKernel) {
    return false;
  }
  if (level + 1 == level_count) return true;
  const auto below = static_cast<LevelKind>(level_kinds[level + 1]);
  return below == LevelKind::Compressed || below == LevelKind::DenseBoundedInKernel;
}

template <typename Dim>
inline void refuse_result_positions(
    const std::vector<Dim>& result_shape,
    const std::vector<int64_t>& expected_shape,
    const int* level_kinds, size_t level_count, const char* op) {
  const auto shape = checked_compiled_shape(
      result_shape, expected_shape, static_cast<int64_t>(level_count), op,
      "result_shape");
  TORCH_CHECK(level_count == shape.size(), argument_name(op, "result_shape"),
              " level contract rank mismatch");
  size_t bounded_levels = 0;  // one past the deepest dense level the entry check bounds
  for (size_t level = 0; level < level_count; ++level) {
    const auto kind = static_cast<LevelKind>(level_kinds[level]);
    TORCH_CHECK(kind == LevelKind::Dense || kind == LevelKind::Compressed ||
                    kind == LevelKind::Coordinate ||
                    (kind == LevelKind::DenseBoundedInKernel &&
                     bounded_in_kernel_level_in_place(level_kinds, level_count, level)),
                argument_name(op, "result_shape"), " unsupported level kind at ", level);
    if (kind == LevelKind::Dense) bounded_levels = level + 1;
  }
  int64_t positions = 1;
  for (size_t level = 0; level < level_count && level <= bounded_levels; ++level) {
    if (static_cast<LevelKind>(level_kinds[level]) == LevelKind::Compressed) {
      TORCH_CHECK(positions < kMaxIndexPosition, argument_name(op, "result_shape"), " ",
                  shape_string(shape), ": compressed level ", level, " would have ",
                  positions, " parents and one position entry more than that; a 32-bit ",
                  "position addresses at most ", kMaxIndexPosition, " entries",
                  SCORCH_INDEX_CAPACITY_CODE);
    }
    if (level == bounded_levels) break;
    const int64_t extent = shape[level];
    TORCH_CHECK(extent == 0 || positions <= kMaxIndexPosition / extent,
                argument_name(op, "result_shape"), " ", shape_string(shape),
                ": the extents of levels 0 to ", level, " multiply to more than ",
                kMaxIndexPosition, ", the most positions a 32-bit position addresses, ",
                "and level ", bounded_levels - 1, " of the result is dense",
                SCORCH_INDEX_CAPACITY_CODE);
    positions *= extent;
  }
  // Each block the kernel bounds itself (DenseBoundedInKernel levels, a run below a compressed
  // level): the first coordinate stored above it needs one block. The product runs left to
  // right, as the kernel forms the block's positions, so no product the kernel forms from these
  // extents passes int64 either.
  size_t first = 0;
  while (first < level_count) {
    if (static_cast<LevelKind>(level_kinds[first]) != LevelKind::DenseBoundedInKernel) {
      ++first;
      continue;
    }
    size_t end = first;
    int64_t block = 1;
    while (end < level_count &&
           static_cast<LevelKind>(level_kinds[end]) == LevelKind::DenseBoundedInKernel) {
      const int64_t extent = shape[end];
      TORCH_CHECK(extent == 0 || block <= kMaxIndexPosition / extent,
                  argument_name(op, "result_shape"), " ", shape_string(shape),
                  ": the extents of dense levels ", first, " to ", end,
                  " multiply to more than ", kMaxIndexPosition, ", the most positions a ",
                  "32-bit position addresses, and each coordinate stored at level ",
                  first - 1, " owns one block of them", SCORCH_INDEX_CAPACITY_CODE);
      block *= extent;
      ++end;
    }
    TORCH_CHECK(end == level_count || block < kMaxIndexPosition,
                argument_name(op, "result_shape"), " ", shape_string(shape),
                ": the extents of dense levels ", first, " to ", end - 1, " multiply to ",
                kMaxIndexPosition, ", and compressed level ", end, " below them has one ",
                "position entry more than that; a 32-bit position addresses at most ",
                kMaxIndexPosition, " entries", SCORCH_INDEX_CAPACITY_CODE);
    first = end;
  }
}

// True exactly when refuse_result_positions raises nothing for the same arguments.
// `expected` is the shape the kernel was compiled for, or nothing (`expected_count` 0) when
// it was compiled for a rank only: what checked_compiled_shape decides from.
template <typename Dim>
inline bool result_positions_fit(
    const std::vector<Dim>& result_shape, const int64_t* expected,
    size_t expected_count, const int* level_kinds, size_t level_count) noexcept {
  const size_t rank = result_shape.size();
  const bool rank_only = expected_count == 0 && level_count > 0;
  if (rank != (rank_only ? level_count : expected_count)) return false;
  if (rank != level_count) return false;
  size_t bounded_levels = 0;  // one past the deepest dense level the entry check bounds
  bool blocks = false;        // whether any level is DenseBoundedInKernel
  for (size_t level = 0; level < rank; ++level) {
    const int64_t extent = static_cast<int64_t>(result_shape[level]);
    if (extent < 0 || extent > std::numeric_limits<int>::max()) return false;
    if (!rank_only && extent != expected[level]) return false;
    const auto kind = static_cast<LevelKind>(level_kinds[level]);
    if (kind == LevelKind::Dense) {
      bounded_levels = level + 1;
    } else if (kind == LevelKind::DenseBoundedInKernel) {
      if (!bounded_in_kernel_level_in_place(level_kinds, level_count, level)) return false;
      blocks = true;
    } else if (kind != LevelKind::Compressed && kind != LevelKind::Coordinate) {
      return false;
    }
  }
  int64_t positions = 1;
  for (size_t level = 0; level < rank && level <= bounded_levels; ++level) {
    if (static_cast<LevelKind>(level_kinds[level]) == LevelKind::Compressed &&
        positions >= kMaxIndexPosition) {
      return false;
    }
    if (level == bounded_levels) break;
    // `positions` and the extent are each at most INT32_MAX here, so their product is
    // exact in int64, and comparing it is the division refuse_result_positions writes.
    positions *= static_cast<int64_t>(result_shape[level]);
    if (positions > kMaxIndexPosition) return false;
  }
  if (!blocks) return true;
  // Each block the kernel bounds itself, as refuse_result_positions checks it. `block` is at
  // most INT32_MAX before each product, so the product is exact in int64.
  int64_t block = 1;
  for (size_t level = 0; level < rank; ++level) {
    if (static_cast<LevelKind>(level_kinds[level]) != LevelKind::DenseBoundedInKernel) {
      block = 1;
      continue;
    }
    block *= static_cast<int64_t>(result_shape[level]);
    if (block > kMaxIndexPosition) return false;
    // The level below, if any, is compressed or in the same block (checked above).
    if (level + 1 < rank &&
        static_cast<LevelKind>(level_kinds[level + 1]) == LevelKind::Compressed &&
        block >= kMaxIndexPosition) {
      return false;
    }
  }
  return true;
}

template <typename Dim>
inline void check_result_positions(
    const std::vector<Dim>& result_shape, const int64_t* expected,
    size_t expected_count, const int* level_kinds, size_t level_count,
    const char* op) {
  if (result_positions_fit(result_shape, expected, expected_count, level_kinds,
                           level_count)) {
    return;
  }
  refuse_result_positions(
      result_shape, std::vector<int64_t>(expected, expected + expected_count),
      level_kinds, level_count, op);
}

// The compiled shape and the level kinds as braced lists, the form every caller in this
// file and every generated kernel writes. Braced lists bind here without constructing a
// std::vector.
template <typename Dim>
inline void check_result_positions(
    const std::vector<Dim>& result_shape,
    std::initializer_list<int64_t> expected_shape,
    std::initializer_list<int> level_kinds, const char* op) {
  check_result_positions(result_shape, expected_shape.begin(), expected_shape.size(),
                         level_kinds.begin(), level_kinds.size(), op);
}

template <typename Dim>
inline void check_result_positions(
    const std::vector<Dim>& result_shape,
    const std::vector<int64_t>& expected_shape,
    const std::vector<int>& level_kinds, const char* op) {
  check_result_positions(result_shape, expected_shape.data(), expected_shape.size(),
                         level_kinds.data(), level_kinds.size(), op);
}

template <typename Dim>
inline void check_legacy_output_shape(const std::vector<Dim>& result_shape,
                                      int64_t expected_rank, const char* op) {
  const auto shape = checked_shape(result_shape, expected_rank, op, "result_shape");
  checked_product(shape, op, "result_shape", true);
}

template <typename Dim>
inline void validate_binary_inputs(
    const char* op, BinaryContract contract, torch::ScalarType dtype,
    const std::vector<Dim>& result_shape, const std::vector<Dim>& a_shape,
    std::vector<std::vector<torch::Tensor>>& a_mode_indices,
    const torch::Tensor& a_values, const std::vector<Dim>& b_shape,
    std::vector<std::vector<torch::Tensor>>& b_mode_indices,
    const torch::Tensor& b_values) {
  if (contract == BinaryContract::CsrDenseMatvec) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 1,
                                      op, "B");
    check_legacy_output_shape(result_shape, 1, op);
    if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
      refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
    }
    TORCH_CHECK(result_shape[0] == a.shape[0], op,
                ": result_shape must be [A.rows]" SCORCH_EXTENT_MISMATCH_CODE);
    return;
  }

  if (contract == BinaryContract::CsrDenseMatmulInt64Elements) {
    checked_product(checked_shape(result_shape, 2, op, "result_shape"), op, "result_shape",
                    /*require_legacy_int_capacity=*/false);
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B", /*int32_elements=*/false);
    if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
      refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
    }
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]" SCORCH_EXTENT_MISMATCH_CODE);
    return;
  }
  // The logical size of the result bounds only the kernels that address a dense result
  // with `int` offsets. A sparse result is bounded by what its levels store.
  if (contract == BinaryContract::CsrDenseMatmul) {
    check_legacy_output_shape(result_shape, 2, op);
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
      refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
    }
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]" SCORCH_EXTENT_MISMATCH_CODE);
    return;
  }
  if (contract == BinaryContract::CsrCsrMatmul) {
    // A CSR result: rows + 1 row offsets, and the entries spmspm_csr counts before it
    // allocates them (it refuses a count past INT32_MAX itself).
    check_result_positions(
        result_shape, {},
        {static_cast<int>(LevelKind::Dense), static_cast<int>(LevelKind::Compressed)},
        op);
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_csr_view(b_shape, b_mode_indices, b_values, dtype, op, "B");
    if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
      refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
    }
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]" SCORCH_EXTENT_MISMATCH_CODE);
    return;
  }
  if (contract == BinaryContract::CooDenseMatmul) {
    check_legacy_output_shape(result_shape, 2, op);
    const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                    op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
      refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
    }
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]" SCORCH_EXTENT_MISMATCH_CODE);
    return;
  }
  // CooCooMatmul. A COO result: coordinate and value vectors that grow by one entry at a
  // time; the result wrapper refuses more entries than a 32-bit position addresses.
  check_result_positions(
      result_shape, {},
      {static_cast<int>(LevelKind::Coordinate), static_cast<int>(LevelKind::Coordinate)},
      op);
  const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                  op, "A");
  const auto b = checked_coo_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                  op, "B");
  if (C10_UNLIKELY_OR_CONST(!(a.shape[1] == b.shape[0]))) {
    refuse_contraction_mismatch(op, a.shape[1], b.shape[0]);
  }
  TORCH_CHECK(result_shape[0] == a.shape[0] && result_shape[1] == b.shape[1],
              op, ": result_shape must equal [A.rows, B.cols]" SCORCH_EXTENT_MISMATCH_CODE);
}

inline int validate_tile_size(int value, const char* op, const char* argument,
                              int64_t extent = -1) {
  TORCH_CHECK(value > 0, op, ": ", argument, " must be positive, got ", value);
  TORCH_CHECK(value <= std::numeric_limits<int>::max() - 15, op, ": ", argument,
              " is too large: ", value);
  if (extent >= 0) {
    return static_cast<int>(
        std::min<int64_t>(value, std::max<int64_t>(extent, 1)));
  }
  return value;
}

inline void validate_thread_override(int value, const char* op) {
  TORCH_CHECK(value >= -1, op,
              ": nthreads_override must be -1, 0, or positive, got ", value);
}

inline void validate_finite_scale(double value, const char* op) {
  TORCH_CHECK(std::isfinite(value), op, ": scale must be finite, got ", value);
  TORCH_CHECK(value >= -std::numeric_limits<float>::max() &&
                  value <= std::numeric_limits<float>::max(),
              op, ": scale cannot be represented as float32: ", value);
}

template <typename Dim>
inline void validate_sddmm_inputs(
    const char* op, const std::vector<Dim>& result_shape,
    const std::vector<Dim>& s_shape,
    std::vector<std::vector<torch::Tensor>>& s_mode_indices,
    const torch::Tensor& s_values, const std::vector<Dim>& a_shape,
    std::vector<std::vector<torch::Tensor>>& a_mode_indices,
    const torch::Tensor& a_values, const std::vector<Dim>& b_shape,
    std::vector<std::vector<torch::Tensor>>& b_mode_indices,
    const torch::Tensor& b_values) {
  const auto s = checked_coo_view(s_shape, s_mode_indices, s_values,
                                  torch::kFloat32, 2, op, "S");
  // The kernel reads row i of A and row j of B at (size_t)i * K, so neither array's element
  // count is held to int32. Its result is S's stored pattern with new values, one per stored
  // entry of S, whatever rows x columns is.
  const auto a = checked_dense_view(a_shape, a_mode_indices, a_values,
                                    torch::kFloat32, 2, op, "A",
                                    /*int32_elements=*/false);
  const auto b = checked_dense_view(b_shape, b_mode_indices, b_values,
                                    torch::kFloat32, 2, op, "B",
                                    /*int32_elements=*/false);
  check_result_positions(
      result_shape, {},
      {static_cast<int>(LevelKind::Coordinate), static_cast<int>(LevelKind::Coordinate)},
      op);
  TORCH_CHECK(result_shape[0] == s.shape[0] && result_shape[1] == s.shape[1],
              op, ": result_shape must equal S.shape" SCORCH_EXTENT_MISMATCH_CODE);
  TORCH_CHECK(a.shape[0] == s.shape[0], op,
              ": A rows must equal S rows" SCORCH_EXTENT_MISMATCH_CODE);
  TORCH_CHECK(b.shape[0] == s.shape[1], op,
              ": B rows must equal S columns" SCORCH_EXTENT_MISMATCH_CODE);
  TORCH_CHECK(a.shape[1] == b.shape[1], op,
              ": A and B reduction dimensions must match" SCORCH_EXTENT_MISMATCH_CODE);
}

inline void validate_csr_segments(
    const char* op, torch::Tensor& positions, const torch::Tensor& values) {
  check_tensor_common(values, torch::kFloat32, op, "values", 1);
  positions = checked_index_tensor(positions, op, "crow_indices");
  TORCH_CHECK(positions.numel() >= 1, op,
              ": crow_indices must contain at least the initial zero");
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), op,
              ": values nnz exceeds the current int32 native bound",
              SCORCH_INDEX_CAPACITY_CODE);
  const int32_t* pos = positions.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, op, ": crow_indices[0] must be 0, got ", pos[0]);
  AbiStructEntry memo;
  memo.add(positions);
  memo.params = {AbiCheckSpans, positions.numel(), nnz};
  const bool known = abi_struct_hit(memo);
  if (!known && abi_screen_spans(pos, positions.numel(), nnz)) {
    for (int64_t i = 1; i < positions.numel(); ++i) {
      TORCH_CHECK(pos[i] >= pos[i - 1] && pos[i] <= nnz, op,
                  ": invalid crow_indices entry ", pos[i], " at position ", i,
                  " for nnz ", nnz);
    }
  }
  TORCH_CHECK(pos[positions.numel() - 1] == nnz, op,
              ": terminal crow index must equal values nnz (", nnz, "), got ",
              pos[positions.numel() - 1]);
  if (!known) abi_struct_store(std::move(memo));
}

inline void validate_attention_inputs(
    const char* op, torch::Tensor& positions, torch::Tensor& coordinates,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  check_tensor_common(q, torch::kFloat32, op, "Q", 3);
  check_tensor_common(k, torch::kFloat32, op, "K", 3);
  check_tensor_common(v, torch::kFloat32, op, "V", 3);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), op,
              ": Q, K, and V must have identical [S, H, D] shapes");
  const int64_t sequence = q.size(0);
  const std::vector<int64_t> output_shape = {q.size(0), q.size(1), q.size(2)};
  checked_product(output_shape, op, "Q/K/V shape");

  const auto raw_positions = positions;
  const auto raw_coordinates = coordinates;
  positions = checked_index_tensor(raw_positions, op, "crow_indices");
  coordinates = checked_index_tensor(raw_coordinates, op, "col_indices");
  TORCH_CHECK(raw_positions.scalar_type() == raw_coordinates.scalar_type(), op,
              ": crow_indices and col_indices must use one common index dtype");
  TORCH_CHECK(positions.numel() == sequence + 1, op,
              ": crow_indices length must be S + 1 (", sequence + 1,
              "), got ", positions.numel());
  TORCH_CHECK(coordinates.numel() <= std::numeric_limits<int>::max(), op,
              ": mask nnz exceeds the current int32 native bound",
              SCORCH_INDEX_CAPACITY_CODE);
  const int32_t* pos = positions.data_ptr<int32_t>();
  const int32_t* crd = coordinates.data_ptr<int32_t>();
  TORCH_CHECK(pos[0] == 0, op, ": crow_indices[0] must be 0, got ", pos[0]);
  // Spans and coordinate bounds, with no sortedness requirement — exactly what the
  // compressed screen does with require_sorted false. Both serial loops stay put and
  // run in their original order, so a mask with several defects still reports the
  // same one it always did.
  AbiStructEntry memo;
  memo.add(coordinates);
  memo.add(positions);
  memo.params = {AbiCheckCompressed, sequence, sequence, coordinates.numel(), 0};
  const bool known = abi_struct_hit(memo);
  const bool bad = !known && abi_screen_compressed(pos, crd, sequence,
                                                   coordinates.numel(), sequence,
                                                   /*require_sorted=*/false);
  if (bad) {
    for (int64_t row = 0; row < sequence; ++row) {
      TORCH_CHECK(pos[row] >= 0 && pos[row + 1] >= pos[row] &&
                      pos[row + 1] <= coordinates.numel(),
                  op, ": invalid CSR mask span for row ", row);
    }
  }
  TORCH_CHECK(pos[sequence] == coordinates.numel(), op,
              ": terminal crow index must equal col_indices nnz (",
              coordinates.numel(), "), got ", pos[sequence]);
  if (bad) {
    for (int64_t p = 0; p < coordinates.numel(); ++p) {
      TORCH_CHECK(crd[p] >= 0 && crd[p] < sequence, op, ": column coordinate ",
                  crd[p], " at position ", p, " is outside [0, ", sequence,
                  ")");
    }
  }
  if (!known) abi_struct_store(std::move(memo));
}

// Generic entry validation emitted at the beginning of every JIT evaluate().
// Shapes and layouts are compile-time kernel contracts encoded by CINLowerer.
inline void validate_jit_tensor(
    const char* op, const char* argument, const std::vector<int64_t>& shape,
    std::vector<std::vector<torch::Tensor>>& mode_indices,
    const torch::Tensor& values, torch::ScalarType expected_dtype,
    const std::vector<int>& level_kinds, const std::vector<int>& mode_order,
    const std::vector<int64_t>& expected_shape) {
  // The operand is bounded level by level below, by the positions each level stores. Its
  // logical element count is not looked at: for a tensor with a sparse level it bounds
  // nothing a kernel forms, and for an all-dense one the last level's count is that product.
  const auto logical_shape = checked_compiled_shape(
      shape, expected_shape, static_cast<int64_t>(level_kinds.size()), op, argument);
  TORCH_CHECK(level_kinds.size() == logical_shape.size(),
              argument_name(op, argument), " level contract rank mismatch");
  TORCH_CHECK(mode_order.size() == logical_shape.size(),
              argument_name(op, argument), " mode-order rank mismatch");
  TORCH_CHECK(mode_indices.size() == logical_shape.size(),
              argument_name(op, argument), " mode_indices must contain ",
              logical_shape.size(), " levels, got ", mode_indices.size());
  check_tensor_common(values, expected_dtype, op,
                      (std::string(argument) + " values").c_str(), 1);

  std::vector<bool> seen_mode(logical_shape.size(), false);
  for (size_t level = 0; level < mode_order.size(); ++level) {
    TORCH_CHECK(mode_order[level] >= 0 &&
                    static_cast<size_t>(mode_order[level]) < logical_shape.size() &&
                    !seen_mode[mode_order[level]],
                argument_name(op, argument), " invalid compiled mode-order entry ",
                mode_order[level], " at level ", level);
    seen_mode[mode_order[level]] = true;
  }

  int64_t storage_count = 1;
  // The positions a kernel can form at the level above, an end position included: what
  // checked_dense_positions holds a dense level to.
  int64_t parent_positions = 1;
  int64_t coordinate_count = -1;
  bool has_index_dtype = false;
  torch::ScalarType index_dtype = torch::kInt32;
  std::vector<torch::Tensor> coordinate_levels;
  const bool all_coordinate =
      !level_kinds.empty() &&
      std::all_of(level_kinds.begin(), level_kinds.end(), [](int kind) {
        return static_cast<LevelKind>(kind) == LevelKind::Coordinate;
      });
  for (size_t level = 0; level < level_kinds.size(); ++level) {
    // Runtime *_shape arguments are already in physical level order (the same
    // convention used by generated ``A_shape[level]`` bound expressions).
    // mode_order is still checked as metadata, but applying it a second time here
    // would reject valid rectangular tensors after a non-identity permutation.
    const int64_t extent = logical_shape[level];
    const auto kind = static_cast<LevelKind>(level_kinds[level]);
    if (kind == LevelKind::Dense) {
      TORCH_CHECK(mode_indices[level].empty(), argument_name(op, argument),
                  " dense level ", level, " must not contain index tensors");
      parent_positions =
          checked_dense_positions(parent_positions, extent, level, op, argument);
      // Within the bound just checked: storage_count <= parent_positions at every level.
      storage_count *= extent;
      continue;
    }

    if (kind == LevelKind::Coordinate) {
      TORCH_CHECK(mode_indices[level].size() == 1,
                  argument_name(op, argument), " coordinate level ", level,
                  " must contain exactly one tensor");
      const auto raw_coordinate = mode_indices[level][0];
      mode_indices[level][0] = checked_index_tensor(
          raw_coordinate, op,
          std::string(argument) + " coordinate level " + std::to_string(level));
      check_common_index_dtype(raw_coordinate, has_index_dtype, index_dtype, op,
                               std::string(argument) + " coordinate level " +
                                   std::to_string(level));
      const int64_t count = mode_indices[level][0].numel();
      if (coordinate_count < 0) {
        coordinate_count = count;
        storage_count = count;
      } else {
        TORCH_CHECK(count == coordinate_count, argument_name(op, argument),
                    " coordinate levels have inconsistent nnz");
      }
      parent_positions = storage_count + 1;
      const int32_t* crd = mode_indices[level][0].data_ptr<int32_t>();
      AbiStructEntry level_memo;
      level_memo.add(mode_indices[level][0]);
      level_memo.params = {AbiCheckBounds, extent, count};
      const bool level_known = abi_struct_hit(level_memo);
      if (!level_known && abi_screen_bounds(crd, count, extent)) {
        for (int64_t p = 0; p < count; ++p) {
          TORCH_CHECK(crd[p] >= 0 && crd[p] < extent,
                      argument_name(op, argument), " coordinate ", crd[p],
                      " at level ", level, " position ", p,
                      " is outside [0, ", extent, ")");
        }
      }
      if (!level_known) abi_struct_store(std::move(level_memo));
      coordinate_levels.push_back(mode_indices[level][0]);
      continue;
    }

    TORCH_CHECK(kind == LevelKind::Compressed,
                argument_name(op, argument), " unsupported level kind at ", level);
    TORCH_CHECK(mode_indices[level].size() == 2,
                argument_name(op, argument), " compressed level ", level,
                " must contain [positions, coordinates]");
    const auto raw_positions = mode_indices[level][0];
    const auto raw_coordinates = mode_indices[level][1];
    mode_indices[level][0] = checked_index_tensor(
        raw_positions, op,
        std::string(argument) + " positions level " + std::to_string(level));
    mode_indices[level][1] = checked_index_tensor(
        raw_coordinates, op,
        std::string(argument) + " coordinates level " + std::to_string(level));
    check_common_index_dtype(raw_positions, has_index_dtype, index_dtype, op,
                             std::string(argument) + " positions level " +
                                 std::to_string(level));
    check_common_index_dtype(raw_coordinates, has_index_dtype, index_dtype, op,
                             std::string(argument) + " coordinates level " +
                                 std::to_string(level));
    const auto& positions = mode_indices[level][0];
    const auto& coordinates = mode_indices[level][1];
    TORCH_CHECK(positions.numel() == storage_count + 1,
                argument_name(op, argument), " compressed level ", level,
                " positions length must equal parent count + 1 (",
                storage_count + 1, "), got ", positions.numel());
    const int32_t* pos = positions.data_ptr<int32_t>();
    const int32_t* crd = coordinates.data_ptr<int32_t>();
    TORCH_CHECK(pos[0] == 0, argument_name(op, argument),
                " compressed level ", level, " must start at position 0");
    // Same three checks the CSR view runs — spans partition [0, nnz], coordinates in
    // range, coordinates strictly increasing within a span — so the same screen and memo
    // family serve both. (Which modules share a memo depends on the platform: see the
    // note on abi_memo_clear.)
    AbiStructEntry level_memo;
    level_memo.add(coordinates);
    level_memo.add(positions);
    level_memo.params = {AbiCheckCompressed, storage_count, extent,
                         coordinates.numel(), 1};
    const bool level_known = abi_struct_hit(level_memo);
    if (!level_known && abi_screen_compressed(pos, crd, storage_count,
                                              coordinates.numel(), extent,
                                              /*require_sorted=*/true)) {
      for (int64_t parent = 0; parent < storage_count; ++parent) {
        TORCH_CHECK(pos[parent] >= 0 && pos[parent + 1] >= pos[parent] &&
                        pos[parent + 1] <= coordinates.numel(),
                    argument_name(op, argument), " invalid compressed span at level ",
                    level, " parent ", parent);
        for (int32_t p = pos[parent]; p < pos[parent + 1]; ++p) {
          TORCH_CHECK(crd[p] >= 0 && crd[p] < extent,
                      argument_name(op, argument), " coordinate ", crd[p],
                      " at compressed level ", level, " position ", p,
                      " is outside [0, ", extent, ")");
          if (p > pos[parent]) {
            TORCH_CHECK(
                crd[p - 1] < crd[p], argument_name(op, argument),
                " coordinates must be strictly increasing at compressed level ",
                level, " position ", p);
          }
        }
      }
    }
    TORCH_CHECK(pos[storage_count] == coordinates.numel(),
                argument_name(op, argument), " compressed level ", level,
                " terminal position must equal coordinate count");
    // Recorded only after every check on this level has passed, so an input rejected
    // by the terminal-position check above is never cached as valid.
    if (!level_known) abi_struct_store(std::move(level_memo));
    storage_count = coordinates.numel();
    parent_positions = storage_count + 1;
  }
  AbiStructEntry lex_memo;
  bool lex_known = true;  // nothing to record unless the check below actually runs
  if (all_coordinate && storage_count > 1) {
    lex_memo.params.push_back(AbiCheckLexStrict);
    std::vector<const int32_t*> crd_levels;
    crd_levels.reserve(coordinate_levels.size());
    for (const auto& coordinate : coordinate_levels) {
      lex_memo.add(coordinate);
      crd_levels.push_back(coordinate.data_ptr<int32_t>());
    }
    lex_memo.params.push_back(storage_count);
    lex_known = abi_struct_hit(lex_memo);
    if (!lex_known && abi_screen_lex(crd_levels, storage_count)) {
      check_coo_strictly_ascending(crd_levels, storage_count, op, argument);
    }
  }
  TORCH_CHECK(values.numel() == storage_count, argument_name(op, argument),
              " values length must equal validated physical storage size ",
              storage_count, ", got ", values.numel());
  if (!lex_known) abi_struct_store(std::move(lex_memo));
}

// The result bound of a generated kernel (check_result_positions), emitted at the
// beginning of every JIT evaluate() before the operands are validated. A generated kernel
// writes the shape it was compiled for and the level kinds as braced lists,
// `{5, 3}, {0, 1}`, which select the first form and construct no std::vector; the second
// form takes them held in vectors.
//
// Both forms are out of line. Inlined into evaluate(), the check changed the register
// allocation of loops it never touches: in the elementwise-product kernel built on mkt1
// (gcc 13.3, -O3 -march=native), it moved the second operand's merge cursor into a stack
// slot, a store and a reload on every merge step, and that kernel ran 1.18x to 1.28x
// slower on two different patterns. Called out of line, the steps are the base's
// instructions again; the call costs one call per kernel call.
SCORCH_NOINLINE inline void validate_jit_result_shape(
    const std::vector<int64_t>& result_shape,
    std::initializer_list<int64_t> expected_shape,
    std::initializer_list<int> level_kinds, const char* op) {
  check_result_positions(result_shape, expected_shape.begin(), expected_shape.size(),
                         level_kinds.begin(), level_kinds.size(), op);
}

SCORCH_NOINLINE inline void validate_jit_result_shape(
    const std::vector<int64_t>& result_shape,
    const std::vector<int64_t>& expected_shape,
    const std::vector<int>& level_kinds, const char* op) {
  check_result_positions(result_shape, expected_shape.data(), expected_shape.size(),
                         level_kinds.data(), level_kinds.size(), op);
}

inline void validate_jit_extra_tensor(const torch::Tensor& tensor,
                                      torch::ScalarType expected_dtype,
                                      const char* op, const char* argument) {
  check_tensor_common(tensor, expected_dtype, op, argument);
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              argument_name(op, argument),
              " element count exceeds the current native int32 bound",
              SCORCH_INDEX_CAPACITY_CODE);
}

// The refusal of a contraction whose two extents differ (validate_binary_inputs): out of line
// and cold, as refuse_index_capacity, so that each check there is one compare and a branch and
// its message is formed only here. It is defined after the last check in this file, and
// declared in the blank line after the codes, so that no check moves: TORCH_CHECK passes
// __LINE__, and every generated kernel's source holds this file.
[[noreturn]] SCORCH_COLD_NOINLINE inline void refuse_contraction_mismatch(const char* op,
                                                                         int64_t a_cols,
                                                                         int64_t b_rows) {
  TORCH_CHECK(false, op, ": contraction mismatch: A.shape[1]=", a_cols,
              " but B.shape[0]=", b_rows, SCORCH_EXTENT_MISMATCH_CODE);
  std::abort();  // not reached: TORCH_CHECK(false) throws
}

}  // namespace scorch_native
