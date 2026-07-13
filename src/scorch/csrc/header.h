#pragma once

#include <ATen/ops/from_blob.h>
#include <torch/extension.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "native_abi.h"

#if defined(__GNUC__) || defined(__clang__)
#define SCORCH_FORCE_INLINE inline __attribute__((always_inline))
#define SCORCH_RESTRICT __restrict__
#else
#define SCORCH_FORCE_INLINE inline
#define SCORCH_RESTRICT
#endif

typedef struct {
  std::vector<std::vector<torch::Tensor>> mode_indices;
} TensorIndex;

typedef struct {
  TensorIndex index;
  torch::Tensor value;
} TensorStorage;

typedef struct {
  TensorStorage storage;
  std::vector<int> shape;
} Tensor;

// Assign into an existing slot or append exactly one new slot. Generated sparse
// builders are sequential unless they pre-size their vectors before an OpenMP
// region. Rejecting gaps turns the old cvector's silent out-of-range behavior
// into a deterministic failure while preserving its amortized append cost.
template <typename T, typename Index, typename U>
SCORCH_FORCE_INLINE void scorch_vector_set(
    std::vector<T>& values, Index index, U&& value) {
  if (index < 0) {
    throw std::out_of_range("negative sparse output index");
  }
  const auto position = static_cast<typename std::vector<T>::size_type>(index);
  if (position == values.size()) {
    values.emplace_back(std::forward<U>(value));
    return;
  }
  values.at(position) = std::forward<U>(value);
}

template <typename T>
inline void scorch_delete_vector_context(void* context) {
  delete static_cast<std::vector<T>*>(context);
}

// Move a dynamic sparse buffer into a Torch storage context without copying.
// TensorMaker takes ownership of the heap vector before make_tensor() can throw;
// the local unique_ptr covers every earlier failure point.
template <typename T>
inline torch::Tensor scorch_tensor_from_vector(
    std::vector<T>&& values, torch::ScalarType dtype) {
  constexpr torch::ScalarType expected_dtype =
      c10::CppTypeToScalarType<typename std::remove_cv<T>::type>::value;
  if (dtype != expected_dtype) {
    throw std::invalid_argument(
        "vector element type does not match requested Torch dtype");
  }
  if (values.empty()) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
  }

  auto owner = std::unique_ptr<std::vector<T>>(
      new std::vector<T>(std::move(values)));
  const int64_t size = static_cast<int64_t>(owner->size());
  const std::array<int64_t, 1> sizes{{size}};
  auto maker = at::for_blob(owner->data(), sizes);
  maker.context(owner.get(), &scorch_delete_vector_context<T>);
  owner.release();
  return maker.options(
                  torch::TensorOptions().dtype(dtype).device(torch::kCPU))
      .make_tensor();
}

template <typename T>
torch::Tensor scorch_tensor_from_vector(
    const std::vector<T>&, torch::ScalarType) = delete;

template <typename T>
inline void scorch_delete_array_context(void* context) {
  delete[] static_cast<T*>(context);
}

// Transfer an exact-size move-only native array into a Torch storage context.
// This is useful for two-pass kernels that know their output size before the
// fill: unique_ptr covers allocation/fill failures, then TensorMaker assumes
// sole ownership before make_tensor() can throw.
template <typename T>
inline torch::Tensor scorch_tensor_from_unique_array(
    std::unique_ptr<T[]>&& values,
    int64_t size,
    torch::ScalarType dtype) {
  constexpr torch::ScalarType expected_dtype =
      c10::CppTypeToScalarType<typename std::remove_cv<T>::type>::value;
  if (dtype != expected_dtype) {
    throw std::invalid_argument(
        "array element type does not match requested Torch dtype");
  }
  if (size < 0) {
    throw std::invalid_argument("negative native array size");
  }
  if (size == 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
  }
  if (!values) {
    throw std::invalid_argument("null native array with nonzero size");
  }

  const std::array<int64_t, 1> sizes{{size}};
  auto maker = at::for_blob(values.get(), sizes);
  maker.context(values.get(), &scorch_delete_array_context<T>);
  values.release();
  return maker.options(
                  torch::TensorOptions().dtype(dtype).device(torch::kCPU))
      .make_tensor();
}

struct scorch_free_deleter {
  void operator()(void* pointer) const noexcept { std::free(pointer); }
};

template <typename T>
using scorch_unique_buffer = std::unique_ptr<T, scorch_free_deleter>;

inline size_t scorch_checked_size_product(size_t left, size_t right) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::length_error("native allocation size overflow");
  }
  return left * right;
}

template <typename T>
inline scorch_unique_buffer<T> scorch_make_aligned_buffer(
    size_t count, size_t alignment = 64) {
  static_assert(std::is_trivial<T>::value,
                "aligned buffers require trivial element types");
  if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("alignment must be a power of two");
  }
  if (alignment < alignof(T)) {
    throw std::invalid_argument("alignment does not satisfy element alignment");
  }
  const size_t actual_count = std::max<size_t>(count, 1);
  if (actual_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::length_error("aligned buffer size overflow");
  }
  void* allocation = nullptr;
  if (posix_memalign(&allocation, alignment, actual_count * sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
  T* elements = static_cast<T*>(allocation);
  for (size_t i = 0; i < actual_count; ++i) {
    ::new (static_cast<void*>(elements + i)) T;
  }
  return scorch_unique_buffer<T>(elements);
}

template <typename T>
inline scorch_unique_buffer<T> scorch_make_aligned_buffer_pool(
    size_t worker_count, size_t elements_per_worker,
    size_t alignment = 64) {
  static_assert(
      std::is_trivial<T>::value,
      "worker scratch elements must have trivial lifetime");
  return scorch_make_aligned_buffer<T>(
      scorch_checked_size_product(worker_count, elements_per_worker),
      alignment);
}

template <typename T>
inline std::unique_ptr<T[]> scorch_make_unique_array_pool(
    size_t worker_count, size_t elements_per_worker) {
  static_assert(
      std::is_trivially_default_constructible<T>::value,
      "worker array elements must be trivially default constructible");
  const size_t count =
      scorch_checked_size_product(worker_count, elements_per_worker);
  return std::unique_ptr<T[]>(new T[std::max<size_t>(count, 1)]);
}

template <typename T>
inline T scorch_relu(T x) {
  return x > T(0) ? x : T(0);
}
template <typename T>
inline T scorch_sigmoid(T x) {
  return T(1) / (T(1) + std::exp(-x));
}
template <typename T>
inline T scorch_tanh(T x) {
  return std::tanh(x);
}
template <typename T>
inline T scorch_gelu(T x) {
  return x * T(0.5) *
      (T(1) + std::tanh(T(0.7978845608) *
                        (x + T(0.044715) * x * x * x)));
}

#include "scorch_policy.h"

template <typename T>
static inline void scorch_zero_dense(T* __restrict__ p, int64_t n) {
  const int64_t bytes = n * static_cast<int64_t>(sizeof(T));
#ifdef SCORCH_TUNE_HOOKS
  {
    const char* setting = std::getenv("SCORCH_CODEGEN_ALLOC");
    if (setting && *setting && std::atol(setting) == 0) {
      std::memset(p, 0, static_cast<size_t>(bytes));
      return;
    }
  }
#endif
  const int thread_count = omp_get_num_procs();
  if (bytes < SCORCH_MEMSET_GRAIN_BYTES || thread_count <= 1) {
    std::memset(p, 0, static_cast<size_t>(bytes));
    return;
  }
#pragma omp parallel num_threads(thread_count)
  {
    const int thread = omp_get_thread_num();
    const int team_size = omp_get_num_threads();
    const int64_t chunk = (n + team_size - 1) / team_size;
    const int64_t begin = static_cast<int64_t>(thread) * chunk;
    const int64_t end = std::min(begin + chunk, n);
    if (end > begin) {
      std::memset(p + begin, 0,
                  static_cast<size_t>(end - begin) * sizeof(T));
    }
  }
}



// ####################################
// ===== BEGIN === COO WKSP IMPL ======
// ####################################

/**
 * This class implements a workspace to store the intermediate results of tensor
 * operations. It keeps track of the intermediate tensor in the coordinate list
 * format.
 * - It provides an interface to insert a coordinate-value pair into the workspace.
 *   - If the coordinate already exists in the workspace, the new value would be
 *     accumulated to the existing value by addition.
 * - It also provides an interface to iterate through the coordinate-value
 *   pairs in the order sorted by the coordinates.
 *   - For example, for a 3-dimensional workspace, the coordinate-value pairs could be:
 *     (0, 0, 0) - 11
 *     (1, 1, 1) - 22
 *     (1, 1, 0) - 33
 *     Then the iterator would return the pairs in the order:
 *     (0, 0, 0) - 11
 *     (1, 1, 0) - 33
 *     (1, 1, 1) - 22
 *
 * @tparam T type of the values stored, e.g. float, double, int, etc.
 */

template <typename T, int N>
class coo_workspace_1d {
  static constexpr int INITIAL_CAPACITY = 1024;
  static constexpr int GROWTH_FACTOR = 2;

  std::unique_ptr<T[]> _values;
  std::unique_ptr<int64_t[]> _indices;
  std::vector<uint8_t> _set_flags;
  int64_t _size = 0;
  int64_t _capacity = 0;

 public:
  explicit coo_workspace_1d(int64_t capacity = INITIAL_CAPACITY)
      : _values(make_uninitialized<T>(capacity)),
        _indices(make_uninitialized<int64_t>(capacity)),
        _set_flags(checked_capacity(capacity), uint8_t{0}),
        _capacity(capacity) {}

  coo_workspace_1d(const coo_workspace_1d&) = delete;
  coo_workspace_1d& operator=(const coo_workspace_1d&) = delete;
  coo_workspace_1d(coo_workspace_1d&&) noexcept = default;
  coo_workspace_1d& operator=(coo_workspace_1d&&) noexcept = default;

  void insert(int64_t coord, T value) {
    if (coord < 0) {
      throw std::out_of_range("negative workspace coordinate");
    }
    if (coord >= _capacity) {
      resize(checked_growth_capacity(coord, _capacity));
    }

    if (!_set_flags[coord]) {
      _values[coord] = value;
      _indices[_size++] = coord;
      _set_flags[coord] = 1;
    } else {
      _values[coord] += value;
    }
  }

  void resize(int64_t new_capacity) {
    const auto capacity = checked_capacity(new_capacity);
    if (new_capacity <= _capacity) {
      return;
    }

    auto values = std::unique_ptr<T[]>(new T[capacity]);
    auto indices = std::unique_ptr<int64_t[]>(new int64_t[capacity]);
    for (int64_t i = 0; i < _size; i++) {
      const int64_t coord = _indices[i];
      values[coord] = std::move(_values[coord]);
      indices[i] = coord;
    }
    _set_flags.resize(capacity, uint8_t{0});
    _values = std::move(values);
    _indices = std::move(indices);
    _capacity = new_capacity;
  }

  void sort() {
    if (_size > 1) {
      std::sort(_indices.get(), _indices.get() + _size);
    }
  }

  void clear() {
    for (int64_t i = 0; i < _size; i++) {
      _set_flags[_indices[i]] = 0;
    }
    _size = 0;
  }

  class iterator {
    int64_t _index;
    T* _values;
    int64_t* _indices;

   public:
    iterator(int64_t index, T* values, int64_t* indices)
        : _index(index), _values(values), _indices(indices) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<int64_t, T> operator*() const {
      int64_t index = _indices[_index];
      return {index, _values[index]};
    }
  };

  iterator begin() { return iterator(0, _values.get(), _indices.get()); }

  iterator end() { return iterator(_size, _values.get(), _indices.get()); }

  int64_t size() const { return _size; }

 private:
  static int64_t checked_growth_capacity(int64_t coord, int64_t capacity) {
    if (coord == std::numeric_limits<int64_t>::max()) {
      throw std::length_error("workspace coordinate exceeds supported capacity");
    }
    const int64_t required = coord + 1;
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    const int64_t grown =
        capacity > maximum / GROWTH_FACTOR
        ? maximum
        : capacity * GROWTH_FACTOR;
    return std::max(required, grown);
  }

  static size_t checked_capacity(int64_t capacity) {
    if (capacity < 0) {
      throw std::invalid_argument("workspace capacity cannot be negative");
    }
    return static_cast<size_t>(capacity);
  }

  template <typename U>
  static std::unique_ptr<U[]> make_uninitialized(int64_t capacity) {
    const auto size = checked_capacity(capacity);
    if (size == 0) {
      return {};
    }
    return std::unique_ptr<U[]>(new U[size]);
  }
};

/**
 * Linked-list workspace for 1D sparse accumulation.
 * Uses two dense arrays (sums + next pointers) sized to the coordinate range.
 * O(1) insert, O(nnz) iterate/clear — no setFlags overhead.
 */
template <typename T>
class linked_list_workspace_view_1d {
  T* _sums = nullptr;
  int64_t* _next = nullptr;
  int64_t* _sorted = nullptr;
  size_t _capacity = 0;
  int64_t _head = -2;
  int64_t _size = 0;
  int64_t _sorted_size = 0;

 public:
  linked_list_workspace_view_1d(
      T* sums, int64_t* next, int64_t* sorted, size_t capacity)
      : _sums(sums),
        _next(next),
        _sorted(sorted),
        _capacity(capacity) {}

  linked_list_workspace_view_1d(
      const linked_list_workspace_view_1d&) = delete;
  linked_list_workspace_view_1d& operator=(
      const linked_list_workspace_view_1d&) = delete;
  linked_list_workspace_view_1d(
      linked_list_workspace_view_1d&&) noexcept = default;
  linked_list_workspace_view_1d& operator=(
      linked_list_workspace_view_1d&&) noexcept = default;

  // Allocation happens serially before OpenMP. Generated kernels defer this
  // non-throwing first touch so each worker initializes its own pages inside
  // the already-established parallel team.
  void initialize_worker_storage() noexcept {
    std::fill_n(_sums, _capacity, T{});
    std::fill_n(_next, _capacity, int64_t{-1});
    _head = -2;
    _size = 0;
    _sorted_size = 0;
  }

  SCORCH_FORCE_INLINE void insert(int64_t coord, T value) {
    if (coord < 0 || static_cast<size_t>(coord) >= _capacity) {
      throw std::out_of_range("workspace coordinate is out of range");
    }
    insert_unchecked(coord, value);
  }

  // Generated kernels call this only after native_abi.h has validated every
  // source coordinate against the same logical extent. Keeping the checked
  // public entry point while removing a redundant branch from every SpGEMM
  // product preserves fail-fast behavior at the ABI boundary.
  SCORCH_FORCE_INLINE void insert_unchecked(int64_t coord, T value) {
    T* SCORCH_RESTRICT sums = _sums;
    int64_t* SCORCH_RESTRICT next = _next;
    sums[coord] += value;
    const int64_t previous = next[coord];
    const bool first = previous == -1;
    const uint64_t first_mask = uint64_t{0} - static_cast<uint64_t>(first);
    next[coord] = static_cast<int64_t>(
        (static_cast<uint64_t>(_head) & first_mask) |
        (static_cast<uint64_t>(previous) & ~first_mask));
    _head = static_cast<int64_t>(
        (static_cast<uint64_t>(coord) & first_mask) |
        (static_cast<uint64_t>(_head) & ~first_mask));
    _size += static_cast<int64_t>(first);
  }

  void sort() {} // no-op: sorting is done during iteration

  void clear() {
    int64_t h = _head;
    while (h >= 0) {
      int64_t tmp = h;
      h = _next[h];
      _next[tmp] = -1;
      _sums[tmp] = T{};
    }
    _head = -2;
    _size = 0;
  }

  int64_t size() const { return _size; }

  class iterator {
    int64_t _index;
    int64_t* _sorted_coords;
    T* _sums;

   public:
    iterator(int64_t index, int64_t* sorted_coords, T* sums)
        : _index(index), _sorted_coords(sorted_coords), _sums(sums) {}

    iterator& operator++() { _index++; return *this; }
    bool operator!=(const iterator& other) const { return _index != other._index; }
    std::pair<int64_t, T> operator*() const {
      int64_t c = _sorted_coords[_index];
      return {c, _sums[c]};
    }
  };

  iterator begin() {
    int64_t pos = 0, h = _head;
    while (h >= 0) {
      _sorted[pos++] = h;
      h = _next[h];
    }
    _sorted_size = pos;
    if (_sorted_size > 1) {
      std::sort(_sorted, _sorted + _sorted_size);
    }
    return iterator(0, _sorted, _sums);
  }

  iterator end() {
    return iterator(_sorted_size, _sorted, _sums);
  }

};

template <typename T>
class linked_list_workspace_1d {
  std::unique_ptr<T[]> _sums_owner;
  std::unique_ptr<int64_t[]> _next_owner;
  std::unique_ptr<int64_t[]> _sorted_owner;
  size_t _capacity;
  linked_list_workspace_view_1d<T> _serial_view;
  bool _worker_storage_initialized = false;

 public:
  explicit linked_list_workspace_1d(
      int64_t capacity, bool defer_worker_initialization = false)
      : _sums_owner(scorch_make_unique_array_pool<T>(
            1, checked_capacity(capacity))),
        _next_owner(scorch_make_unique_array_pool<int64_t>(
            1, checked_capacity(capacity))),
        _sorted_owner(scorch_make_unique_array_pool<int64_t>(
            1, checked_capacity(capacity))),
        _capacity(checked_capacity(capacity)),
        _serial_view(
            _sums_owner.get(),
            _next_owner.get(),
            _sorted_owner.get(),
            _capacity) {
    if (!defer_worker_initialization) {
      initialize_worker_storage();
    }
  }

  linked_list_workspace_1d(const linked_list_workspace_1d&) = delete;
  linked_list_workspace_1d& operator=(const linked_list_workspace_1d&) = delete;
  linked_list_workspace_1d(linked_list_workspace_1d&&) noexcept = default;
  linked_list_workspace_1d& operator=(
      linked_list_workspace_1d&&) noexcept = default;

  linked_list_workspace_view_1d<T> make_view() {
    // OpenMP may choose a different dynamic team size for two consecutive
    // regions. Initialize on the first use of each owner, rather than assuming
    // that an earlier region visited the same worker id. A worker id addresses
    // exactly one owner within a region, and the regions are sequential.
    if (!_worker_storage_initialized) {
      initialize_worker_storage();
    }
    return linked_list_workspace_view_1d<T>(
        _sums_owner.get(),
        _next_owner.get(),
        _sorted_owner.get(),
        _capacity);
  }

  void initialize_worker_storage() noexcept {
    _serial_view.initialize_worker_storage();
    _worker_storage_initialized = true;
  }
  SCORCH_FORCE_INLINE void insert(int64_t coord, T value) {
    _serial_view.insert(coord, value);
  }
  SCORCH_FORCE_INLINE void insert_unchecked(int64_t coord, T value) {
    _serial_view.insert_unchecked(coord, value);
  }
  void sort() { _serial_view.sort(); }
  void clear() { _serial_view.clear(); }
  int64_t size() const { return _serial_view.size(); }
  typename linked_list_workspace_view_1d<T>::iterator begin() {
    return _serial_view.begin();
  }
  typename linked_list_workspace_view_1d<T>::iterator end() {
    return _serial_view.end();
  }

 private:
  static size_t checked_capacity(int64_t capacity) {
    if (capacity < 0) {
      throw std::invalid_argument("workspace capacity cannot be negative");
    }
    return static_cast<size_t>(capacity);
  }
};

template <typename T, int N>
class coo_workspace {
  static_assert(N > 0, "coordinate workspace rank must be positive");
  static constexpr int BLOCK_SIZE = 1024;

  struct Entry {
    std::array<int64_t, N> coords;
    T value;
  };

  std::vector<Entry> _entries;
  std::unordered_map<int64_t, size_t> _existingCoords;
  std::vector<size_t> _sortedIndices;
  std::vector<int64_t> _resultShape;

 public:
  explicit coo_workspace(
      int capacity, const std::vector<int64_t>& result_shape)
      : _resultShape(result_shape) {
    if (capacity < 0) {
      throw std::invalid_argument("workspace capacity cannot be negative");
    }
    _entries.reserve(static_cast<size_t>(capacity));
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE, {}) {}

  coo_workspace(const coo_workspace&) = delete;
  coo_workspace& operator=(const coo_workspace&) = delete;
  coo_workspace(coo_workspace&&) noexcept = default;
  coo_workspace& operator=(coo_workspace&&) noexcept = default;

  void insert(const std::vector<int64_t>& coord, T value) {
    if (coord.size() != N || _resultShape.size() != N) {
      throw std::invalid_argument("workspace coordinate rank mismatch");
    }
    int64_t index = 0;
    for (int i = 0; i < N; i++) {
      const int64_t extent = _resultShape[i];
      const int64_t component = coord[i];
      if (extent < 0) {
        throw std::invalid_argument("workspace extent cannot be negative");
      }
      if (component < 0 || component >= extent) {
        throw std::out_of_range("workspace coordinate is out of range");
      }
      if (i == 0) {
        index = component;
        continue;
      }
      if (index >
          (std::numeric_limits<int64_t>::max() - component) / extent) {
        throw std::length_error("workspace coordinate flattening overflow");
      }
      index = index * extent + component;
    }

    auto existingCoordIt = _existingCoords.find(index);
    if (existingCoordIt != _existingCoords.end()) {
      _entries[existingCoordIt->second].value += value;
      return;
    }

    Entry entry;
    std::copy(coord.begin(), coord.end(), entry.coords.begin());
    entry.value = value;
    _entries.push_back(entry);

    _existingCoords[index] = _entries.size() - 1;
  }

  void sort() {
    _sortedIndices.resize(_entries.size());
    std::iota(_sortedIndices.begin(), _sortedIndices.end(), 0);

    auto radixComparator = [this](size_t a, size_t b) {
      for (int i = 0; i < N; i++) {
        if (_entries[a].coords[i] != _entries[b].coords[i]) {
          return _entries[a].coords[i] < _entries[b].coords[i];
        }
      }
      return false;
    };
    std::sort(_sortedIndices.begin(), _sortedIndices.end(), radixComparator);
  }

  class iterator {
    size_t _index;
    std::vector<Entry>& _entries;
    std::vector<size_t>* _sortedIndices;

   public:
    iterator(size_t index, std::vector<Entry>& entries,
             std::vector<size_t>* sortedIndices)
        : _index(index), _entries(entries), _sortedIndices(sortedIndices) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<std::vector<int64_t>, T> operator*() const {
      size_t sortedIndex = (*_sortedIndices)[_index];
      std::vector<int64_t> coord(_entries[sortedIndex].coords.begin(),
                                 _entries[sortedIndex].coords.end());
      return {coord, _entries[sortedIndex].value};
    }
  };

  iterator begin() {
    return iterator(0, _entries, &_sortedIndices);
  }

  iterator end() {
    return iterator(_entries.size(), _entries, &_sortedIndices);
  }

  int64_t size() const { return static_cast<int64_t>(_entries.size()); }

  int64_t capacity() const { return static_cast<int64_t>(_entries.capacity()); }
};


// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################

#undef SCORCH_FORCE_INLINE
#undef SCORCH_RESTRICT
