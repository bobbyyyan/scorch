#pragma once

#include <torch/extension.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
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

// Nonzeros per validation worker. Deliberately HIGH: with the memo below, a given
// index array is scanned once rather than once per call, so the parallel path only
// has to pay off on genuinely large arrays. Set it low and mid-sized inputs spawn a
// small team (e.g. 6 workers for 428K nonzeros) immediately before the kernel spawns
// its own 32 — a libgomp team reshape per call that measured as a 2-4x REGRESSION on
// bcsstk17. One million nonzeros is ~4 MB, past any L3, where threads clearly win.
#ifndef SCORCH_ABI_VALIDATE_GRAIN
#define SCORCH_ABI_VALIDATE_GRAIN 1048576L
#endif

// Entries retained by the validation memo before it is cleared wholesale. Each entry
// is a few dozen bytes; the bound exists so a program churning through millions of
// distinct sparse tensors cannot grow it without limit.
#ifndef SCORCH_ABI_MEMO_MAX
#define SCORCH_ABI_MEMO_MAX 4096
#endif

namespace scorch_native {

// Worker count for a validation scan: one worker per grain of nonzeros, capped by
// the machine. Sized from the same work/grain shape as scorch_policy.h so a small
// input stays serial instead of paying a team launch to read a few kilobytes.
inline int abi_scan_threads(int64_t n) {
#ifdef _OPENMP
  if (n < SCORCH_ABI_VALIDATE_GRAIN) return 1;
  const int64_t by_work = n / SCORCH_ABI_VALIDATE_GRAIN;
  const int64_t hw = omp_get_num_procs();
  return (int)(by_work < hw ? by_work : hw);
#else
  (void)n;
  return 1;
#endif
}

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
inline bool abi_screen_bounds_typed(const T* crd, int64_t n, int64_t limit) {
  if (n <= 0) return false;
  const T lim = abi_limit_clamp<T>(limit);
  int bad = 0;
  const int nt = abi_scan_threads(n);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nt) schedule(static) reduction(| : bad) \
    if (nt > 1)
#endif
  for (int64_t p = 0; p < n; ++p) {
    const T c = crd[p];
    bad |= (c < 0) | (c >= lim);
  }
  return bad != 0;
}

inline bool abi_screen_bounds(const int32_t* crd, int64_t n, int64_t limit) {
  return abi_screen_bounds_typed<int32_t>(crd, n, limit);
}

// "positions do not decrease and stay within [0, nnz]", for the callers that validate
// a position array on its own. O(rows) rather than O(nnz), but a TORCH_CHECK per row
// is what made it show up at all.
template <typename T>
inline bool abi_screen_spans_typed(const T* pos, int64_t count, int64_t nnz) {
  int bad = 0;
  const T nnz_t = (T)nnz;  // callers check nnz fits the index width first
  for (int64_t i = 1; i < count; ++i) {
    bad |= (pos[i] < pos[i - 1]) | (pos[i] > nnz_t);
  }
  return bad != 0;
}

inline bool abi_screen_spans(const int32_t* pos, int64_t count, int64_t nnz) {
  return abi_screen_spans_typed<int32_t>(pos, count, nnz);
}

// "positions partition [0, nnz] without decreasing, every coordinate lies in
// [0, limit), and — when required — coordinates ascend inside each parent's span."
//
// Both coordinate tests run as FLAT loops over the whole coordinate array rather than
// nested per-parent loops. A per-parent loop cannot vectorize usefully when spans are
// short: a 39-nonzero FEM row spends more on vector peeling and tail than on the
// eight-wide body, ~3.5 cycles per nonzero, i.e. the check cost more than the SpMM it
// was guarding.
//
// Bounds do not care about span structure at all. Sortedness does, but only through
// one observation: a DESCENT at position p (crd[p-1] > crd[p]) is legal exactly when p
// is the first position of a span. So count every descent flat, count the descents
// that sit on a span boundary (O(parent_count)), and compare.
template <typename T>
inline bool abi_screen_compressed_typed(const T* pos, const T* crd,
                                        int64_t parent_count, int64_t nnz,
                                        int64_t limit, bool require_sorted) {
  int bad = 0;
  const T nnz32 = (T)nnz;  // callers check nnz fits the index width first
  for (int64_t parent = 0; parent < parent_count; ++parent) {
    const T start = pos[parent], end = pos[parent + 1];
    bad |= (start < 0) | (end < start) | (end > nnz32);
  }
  if (bad || nnz <= 0) return bad != 0;

  const T lim = abi_limit_clamp<T>(limit);
  const T c0 = crd[0];
  bad |= (c0 < 0) | (c0 >= lim);
  const int nt = abi_scan_threads(nnz);
  if (!require_sorted) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nt) schedule(static) reduction(| : bad) \
    if (nt > 1)
#endif
    for (int64_t p = 1; p < nnz; ++p) {
      const T c = crd[p];
      bad |= (c < 0) | (c >= lim);
    }
    return bad != 0;
  }

  int64_t desc = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nt) schedule(static) reduction(| : bad) \
    reduction(+ : desc) if (nt > 1)
#endif
  for (int64_t p = 1; p < nnz; ++p) {
    const T c = crd[p];
    bad |= (c < 0) | (c >= lim);
    desc += (crd[p - 1] > c);
  }
  int64_t allowed = 0;
  for (int64_t parent = 1; parent < parent_count; ++parent) {
    const T p = pos[parent];
    if (p == pos[parent - 1]) continue;  // empty span: same boundary, already counted
    if (p > 0 && p < nnz) allowed += (crd[p - 1] > crd[p]);
  }
  // `allowed` can only ever be a subset of `desc`, so a mismatch means some span is
  // internally unsorted.
  bad |= (desc != allowed);
  return bad != 0;
}

inline bool abi_screen_compressed(const int32_t* pos, const int32_t* crd,
                                  int64_t parent_count, int64_t nnz, int64_t limit,
                                  bool require_sorted) {
  return abi_screen_compressed_typed<int32_t>(pos, crd, parent_count, nnz, limit,
                                              require_sorted);
}

// "COO coordinates ascend lexicographically across levels"
template <typename T>
inline bool abi_screen_lex_typed(const std::vector<const T*>& levels, int64_t n) {
  if (levels.empty() || n <= 1) return false;
  const size_t depth = levels.size();
  const T* const* lv = levels.data();
  int bad = 0;
  const int nt = abi_scan_threads(n);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nt) schedule(static) reduction(| : bad) \
    if (nt > 1)
#endif
  for (int64_t p = 1; p < n; ++p) {
    int cmp = 0;  // first level that differs decides, exactly as the serial loop does
    for (size_t l = 0; l < depth && cmp == 0; ++l) {
      const T a = lv[l][p - 1], b = lv[l][p];
      cmp = (b > a) - (b < a);
    }
    bad |= (cmp < 0);
  }
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
// numel and the version counter are compared too, so a resize or any mutation routed
// through torch invalidates. Keying on data_ptr rather than on the StorageImpl means
// two distinct views of one storage get an entry each instead of evicting each other
// on every call.
//
// What this does NOT catch: a write straight through a shared raw buffer (numpy
// writing into memory a torch tensor aliases) does not bump the version counter, so a
// buffer corrupted that way can now reach a kernel unchecked. That is a deliberate
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

// Identity of one index array, precise enough that a cached verdict about it can be
// trusted. See the recycling-safety note above for why these fields and no others.
struct AbiArrayId {
  c10::weak_intrusive_ptr<c10::StorageImpl> alive;
  const void* storage = nullptr;  // compared, never dereferenced
  const void* data = nullptr;
  int64_t nbytes = 0;
  int64_t numel = 0;
  uint32_t version = 0;

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
  return id;
}

inline bool abi_same_array(const AbiArrayId& a, const AbiArrayId& b) {
  return a.storage == b.storage && a.data == b.data && a.nbytes == b.nbytes &&
         a.numel == b.numel && a.version == b.version;
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
  AbiCheckCompressed = 2,  // spans partition [0, nnz], coordinates in range, sorted
  AbiCheckLex = 3,         // coordinate levels ascend lexicographically
  AbiCheckSpans = 4,       // positions alone: non-decreasing and within [0, nnz]
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

inline std::unordered_map<const void*, AbiStructEntry>& abi_struct_map() {
  static std::unordered_map<const void*, AbiStructEntry> m;
  return m;
}

// Reclaims dead entries and keeps the map bounded. Called before every insert, and it
// sweeps EVERY time rather than only when the map is full, because a narrow entry owns
// an int32 copy of its source array: waiting for 4096 entries to accumulate would hold
// 4096 dead index arrays, which on graph-scale operands is hundreds of megabytes. An
// insert only happens on a miss — once per newly seen array — so an O(entries)
// weak-pointer sweep there is not on any hot path. Wholesale clearing is the backstop
// for a program holding more live arrays than the bound allows.
template <typename Entry>
inline void abi_bound_map(std::unordered_map<const void*, Entry>& map) {
  for (auto it = map.begin(); it != map.end();) {
    if (!it->second.live()) it = map.erase(it);
    else ++it;
  }
  if (map.size() >= (size_t)SCORCH_ABI_MEMO_MAX) map.clear();
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
  abi_bound_map(map);
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
  auto it = map.find(want.arrays[0].data);
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
  abi_bound_map(map);
  const void* key = entry.arrays[0].data;
  map.insert_or_assign(key, std::move(entry));
}

// Drops every cached verdict and every narrowed copy held by THIS module. The maps
// are inline function-local statics and Python dlopens extension modules with
// RTLD_LOCAL, so the prebuilt extension and each JIT-compiled kernel carry their own.
// Every module amortizes over its own calls, which is all that is needed; the cost is
// that a tensor used on both paths is narrowed once per module. Exposed so tests can
// force the cold path and so a caller under memory pressure can reclaim the copies.
inline void abi_memo_clear() {
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  abi_narrow_map().clear();
  abi_struct_map().clear();
}

inline size_t abi_memo_size() {
  std::lock_guard<std::mutex> guard(abi_memo_mutex());
  return abi_narrow_map().size() + abi_struct_map().size();
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
};

enum class LevelKind : int {
  Dense = 0,
  Compressed = 1,
  Coordinate = 2,
};

inline std::string argument_name(const char* op, const char* argument) {
  return std::string(op) + ": " + argument;
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
                " exceeds the current native int32 loop bound: ", dim);
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
                mode, " cannot be represented by the legacy int32 kernel ABI");
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
                " exceeds the current native int32 allocation bound");
  }
  return product;
}

template <typename Dim>
inline std::vector<int64_t> checked_expected_shape(
    const std::vector<Dim>& shape, const std::vector<int64_t>& expected,
    const char* op, const char* argument,
    bool require_legacy_int_capacity = false) {
  const auto actual = checked_shape(shape, static_cast<int64_t>(expected.size()),
                                    op, argument);
  TORCH_CHECK(actual == expected, argument_name(op, argument), " must equal ",
              shape_string(expected), ", got ", shape_string(actual));
  checked_product(actual, op, argument, require_legacy_int_capacity);
  return actual;
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
              argument, " length exceeds the current int32 native bound");

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
    int64_t lo = 0, hi = 0;
    const int nt = abi_scan_threads(n);
    if (nt > 1) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nt) schedule(static) \
    reduction(min : lo) reduction(max : hi)
#endif
      for (int64_t i = 0; i < n; ++i) {
        const int64_t v = data[i];
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
      }
    } else {
      for (int64_t i = 0; i < n; ++i) {
        const int64_t v = data[i];
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
      }
    }
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
    int64_t expected_rank, const char* op, const char* argument) {
  const auto logical_shape = checked_shape(shape, expected_rank, op, argument);
  const int64_t numel = checked_product(logical_shape, op, argument, true);
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
              " nnz exceeds the current int32 native bound");
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
          TORCH_CHECK(crd[p - 1] <= crd[p], argument_name(op, argument),
                      " coordinates must be sorted within each CSR row; row ", row,
                      " decreases at position ", p);
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
              " nnz exceeds the current int32 native bound");

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
    lex_memo.params.push_back(AbiCheckLex);
    std::vector<const int32_t*> crd_levels;
    crd_levels.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
      lex_memo.add(coordinate);
      crd_levels.push_back(coordinate.data_ptr<int32_t>());
    }
    lex_memo.params.push_back(nnz);
    const bool lex_known = abi_struct_hit(lex_memo);
    if (!lex_known && abi_screen_lex(crd_levels, nnz)) {
      for (int64_t p = 1; p < nnz; ++p) {
        bool greater = false;
        bool different = false;
        for (size_t level = 0; level < coordinates.size(); ++level) {
          const int32_t* crd = coordinates[level].data_ptr<int32_t>();
          if (crd[p] != crd[p - 1]) {
            greater = crd[p] > crd[p - 1];
            different = true;
            break;
          }
        }
        TORCH_CHECK(!different || greater, argument_name(op, argument),
                    " COO coordinates must be lexicographically ordered; order ",
                    "decreases at position ", p);
      }
    }
    if (!lex_known) abi_struct_store(std::move(lex_memo));
  }
  return CooInputView(logical_shape, coordinates, values);
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
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0], op,
                ": result_shape must be [A.rows]");
    return;
  }

  check_legacy_output_shape(result_shape, 2, op);
  if (contract == BinaryContract::CsrDenseMatmul) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  if (contract == BinaryContract::CsrCsrMatmul) {
    const auto a = checked_csr_view(a_shape, a_mode_indices, a_values, dtype, op, "A");
    const auto b = checked_csr_view(b_shape, b_mode_indices, b_values, dtype, op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  if (contract == BinaryContract::CooDenseMatmul) {
    const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                    op, "A");
    const auto b = checked_dense_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                      op, "B");
    TORCH_CHECK(a.shape[1] == b.shape[0], op,
                ": contraction mismatch: A.shape[1]=", a.shape[1],
                " but B.shape[0]=", b.shape[0]);
    TORCH_CHECK(result_shape[0] == a.shape[0] &&
                    result_shape[1] == b.shape[1],
                op, ": result_shape must equal [A.rows, B.cols]");
    return;
  }
  const auto a = checked_coo_view(a_shape, a_mode_indices, a_values, dtype, 2,
                                  op, "A");
  const auto b = checked_coo_view(b_shape, b_mode_indices, b_values, dtype, 2,
                                  op, "B");
  TORCH_CHECK(a.shape[1] == b.shape[0], op,
              ": contraction mismatch: A.shape[1]=", a.shape[1],
              " but B.shape[0]=", b.shape[0]);
  TORCH_CHECK(result_shape[0] == a.shape[0] && result_shape[1] == b.shape[1],
              op, ": result_shape must equal [A.rows, B.cols]");
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
  const auto a = checked_dense_view(a_shape, a_mode_indices, a_values,
                                    torch::kFloat32, 2, op, "A");
  const auto b = checked_dense_view(b_shape, b_mode_indices, b_values,
                                    torch::kFloat32, 2, op, "B");
  check_legacy_output_shape(result_shape, 2, op);
  TORCH_CHECK(result_shape[0] == s.shape[0] && result_shape[1] == s.shape[1],
              op, ": result_shape must equal S.shape");
  TORCH_CHECK(a.shape[0] == s.shape[0], op,
              ": A rows must equal S rows");
  TORCH_CHECK(b.shape[0] == s.shape[1], op,
              ": B rows must equal S columns");
  TORCH_CHECK(a.shape[1] == b.shape[1], op,
              ": A and B reduction dimensions must match");
}

inline void validate_csr_segments(
    const char* op, torch::Tensor& positions, const torch::Tensor& values) {
  check_tensor_common(values, torch::kFloat32, op, "values", 1);
  positions = checked_index_tensor(positions, op, "crow_indices");
  TORCH_CHECK(positions.numel() >= 1, op,
              ": crow_indices must contain at least the initial zero");
  const int64_t nnz = values.numel();
  TORCH_CHECK(nnz <= std::numeric_limits<int>::max(), op,
              ": values nnz exceeds the current int32 native bound");
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
              ": mask nnz exceeds the current int32 native bound");
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
  const auto logical_shape = expected_shape.empty() && !level_kinds.empty()
                                 ? checked_shape(
                                       shape,
                                       static_cast<int64_t>(level_kinds.size()),
                                       op, argument)
                                 : checked_expected_shape(
                                       shape, expected_shape, op, argument,
                                       true);
  checked_product(logical_shape, op, argument, true);
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
      TORCH_CHECK(extent == 0 ||
                      storage_count <= std::numeric_limits<int64_t>::max() / extent,
                  argument_name(op, argument), " storage size overflows at level ",
                  level);
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
    // range, coordinates sorted within a span — so the same screen and the same memo
    // family serve both. (Sharing is within one module: see the note on abi_memo_clear
    // for why each loaded kernel carries its own maps.)
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
            TORCH_CHECK(crd[p - 1] <= crd[p], argument_name(op, argument),
                        " coordinates decrease at compressed level ", level,
                        " position ", p);
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
  }
  AbiStructEntry lex_memo;
  bool lex_known = true;  // nothing to record unless the check below actually runs
  if (all_coordinate && storage_count > 1) {
    lex_memo.params.push_back(AbiCheckLex);
    std::vector<const int32_t*> crd_levels;
    crd_levels.reserve(coordinate_levels.size());
    for (const auto& coordinate : coordinate_levels) {
      lex_memo.add(coordinate);
      crd_levels.push_back(coordinate.data_ptr<int32_t>());
    }
    lex_memo.params.push_back(storage_count);
    lex_known = abi_struct_hit(lex_memo);
    if (!lex_known && abi_screen_lex(crd_levels, storage_count)) {
      for (int64_t p = 1; p < storage_count; ++p) {
        bool greater = false;
        bool different = false;
        for (const auto& coordinate : coordinate_levels) {
          const int32_t* crd = coordinate.data_ptr<int32_t>();
          if (crd[p] != crd[p - 1]) {
            greater = crd[p] > crd[p - 1];
            different = true;
            break;
          }
        }
        TORCH_CHECK(!different || greater, argument_name(op, argument),
                    " COO coordinates must be lexicographically ordered; order ",
                    "decreases at position ", p);
      }
    }
  }
  TORCH_CHECK(values.numel() == storage_count, argument_name(op, argument),
              " values length must equal validated physical storage size ",
              storage_count, ", got ", values.numel());
  if (!lex_known) abi_struct_store(std::move(lex_memo));
}

inline void validate_jit_result_shape(
    const std::vector<int64_t>& result_shape,
    const std::vector<int64_t>& expected_shape, int64_t expected_rank,
    const char* op) {
  if (expected_shape.empty() && expected_rank > 0) {
    const auto shape =
        checked_shape(result_shape, expected_rank, op, "result_shape");
    checked_product(shape, op, "result_shape", true);
  } else {
    checked_expected_shape(result_shape, expected_shape, op, "result_shape",
                           true);
  }
}

inline void validate_jit_extra_tensor(const torch::Tensor& tensor,
                                      torch::ScalarType expected_dtype,
                                      const char* op, const char* argument) {
  check_tensor_common(tensor, expected_dtype, op, argument);
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              argument_name(op, argument),
              " element count exceeds the current native int32 bound");
}

}  // namespace scorch_native
