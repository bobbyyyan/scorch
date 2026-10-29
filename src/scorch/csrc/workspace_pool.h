// Per-worker sparse workspace pools for generated kernels that drain a sparse workspace into a
// dense result (CINLowerer._lower_dense_result_scatter): coo_workspace_owner_1d, the worker's
// coo_workspace_view_1d and scorch_check_workspace_pool.
//
// Not part of the JIT preamble.  Code generation places this file's text in a kernel's own source,
// after the preamble and ahead of the kernel's function, only when that function names one of
// llir.DataType's coo_workspace_owner_1d pool types (llir.DataType.packaged_header,
// LLIRLowerer._type_text).  Every other kernel's source, and so its name and its cached build, is
// what it would be without this file; a kernel that holds a pool carries this text in its own
// source, so its name changes when the file does.  The text follows the preamble, so header.h's
// definitions (coo_workspace_1d, scorch_make_unique_array_pool) are in scope; it includes nothing
// itself, since load_inline's build directory cannot find the packaged headers
// (utils.packaged_header_text refuses an include directive).  header.h undefines its
// SCORCH_FORCE_INLINE and SCORCH_RESTRICT at its end, so this file defines them the same way
// and undefines them at its own end.
#ifndef SCORCH_WORKSPACE_POOL_H
#define SCORCH_WORKSPACE_POOL_H

#if defined(__GNUC__) || defined(__clang__)
#define SCORCH_FORCE_INLINE inline __attribute__((always_inline))
#define SCORCH_RESTRICT __restrict__
#else
#define SCORCH_FORCE_INLINE inline
#define SCORCH_RESTRICT
#endif

// The same allocation as scorch_make_unique_array_pool for code that must not throw: inside an
// OpenMP region an exception that leaves the region ends the process, so a worker allocating there
// asks for null instead.
template <typename T>
inline std::unique_ptr<T[]> scorch_make_unique_array_nothrow(size_t count) noexcept {
  static_assert(
      std::is_trivially_default_constructible<T>::value,
      "worker array elements must be trivially default constructible");
  return std::unique_ptr<T[]>(new (std::nothrow) T[std::max<size_t>(count, 1)]);
}

/**
 * One worker's coo_workspace_1d over a coo_workspace_owner_1d's arrays, at the owner's fixed
 * capacity: insert, iteration in insertion order and clear(), as coo_workspace_1d<T, 1>, and no
 * growth.  A generated kernel holds the view in a local variable of the worker, so its size and
 * pointers can stay in registers: they are not members of an object other code may reach, which
 * every store through the arrays (the flag array is uint8_t, which may alias any object) would
 * otherwise force the compiler to reload, and no two workers write one cache line.  A pool of
 * coo_workspace_1d objects side by side in one vector had both costs on every insert.  Every
 * member a generated kernel calls (failed, insert_unchecked, begin, end, size, clear) is always
 * inlined, so no call receives the view's address and the view can live in registers.
 */
template <typename T>
class coo_workspace_view_1d {
  T* _values = nullptr;
  int64_t* _indices = nullptr;
  uint8_t* _set_flags = nullptr;
  int64_t _capacity = 0;
  int64_t _size = 0;
  bool _failed = false;

 public:
  using iterator = typename coo_workspace_1d<T, 1>::iterator;

  // The arrays hold capacity elements each and every flag is zero (a view leaves them zero
  // again by clearing after its last drain).
  coo_workspace_view_1d(T* values, int64_t* indices, uint8_t* set_flags, int64_t capacity)
      : _values(values), _indices(indices), _set_flags(set_flags), _capacity(capacity) {}

  // The view a worker gets when its owner could not allocate inside a parallel region: no
  // storage. The worker skips its iterations (failed() is true) and the code after the region
  // throws (scorch_check_workspace_pool).
  static coo_workspace_view_1d failed_view() noexcept {
    coo_workspace_view_1d view(nullptr, nullptr, nullptr, 0);
    view._failed = true;
    return view;
  }

  SCORCH_FORCE_INLINE bool failed() const noexcept { return _failed; }

  coo_workspace_view_1d(const coo_workspace_view_1d&) = delete;
  coo_workspace_view_1d& operator=(const coo_workspace_view_1d&) = delete;
  coo_workspace_view_1d(coo_workspace_view_1d&&) noexcept = default;
  coo_workspace_view_1d& operator=(coo_workspace_view_1d&&) noexcept = default;

  SCORCH_FORCE_INLINE void insert(int64_t coord, T value) {
    if (coord < 0 || coord >= _capacity) {
      throw std::out_of_range("workspace coordinate is out of range");
    }
    insert_unchecked(coord, value);
  }

  // Generated kernels call this only after native_abi.h has validated every source coordinate
  // against the logical extent the capacity is (the scatter after the drain writes the result
  // at the same coordinate, unchecked), as linked_list_workspace_view_1d::insert_unchecked.
  SCORCH_FORCE_INLINE void insert_unchecked(int64_t coord, T value) {
    T* SCORCH_RESTRICT values = _values;
    int64_t* SCORCH_RESTRICT indices = _indices;
    uint8_t* SCORCH_RESTRICT set_flags = _set_flags;
    if (!set_flags[coord]) {
      values[coord] = value;
      indices[_size] = coord;
      set_flags[coord] = 1;
      _size++;
    } else {
      values[coord] += value;
    }
  }

  void sort() {
    if (_size > 1) {
      std::sort(_indices, _indices + _size);
    }
  }

  SCORCH_FORCE_INLINE void clear() {
    for (int64_t i = 0; i < _size; i++) {
      _set_flags[_indices[i]] = 0;
    }
    _size = 0;
  }

  SCORCH_FORCE_INLINE iterator begin() { return iterator(0, _values, _indices); }

  SCORCH_FORCE_INLINE iterator end() { return iterator(_size, _values, _indices); }

  SCORCH_FORCE_INLINE int64_t size() const { return _size; }
};

/**
 * The arrays of one worker's coo_workspace_view_1d, at a fixed capacity (the extent of the
 * result level the workspace's key stands for), for a pool of one owner per worker built before
 * a parallel region.
 * allocate_in_region: the arrays wait for the first make_region_view().  The constructor then
 * allocates nothing, so building a pool of t owners before a region costs O(t) whatever the
 * capacity; each worker allocates its own owner's arrays, and zero-fills the flags, at its first
 * make_region_view(), inside the region, so its pages are first touched by the worker that uses
 * them.  A capacity whose byte size passes size_t still throws here, in serial code, before any
 * region.  Without it (a serial kernel's pool of one) the constructor allocates and zero-fills.
 * An owner is written only when its arrays are allocated: every insert writes the view.
 */
template <typename T>
class coo_workspace_owner_1d {
  std::unique_ptr<T[]> _values;
  std::unique_ptr<int64_t[]> _indices;
  std::unique_ptr<uint8_t[]> _set_flags;
  int64_t _capacity;
  bool _allocation_failed = false;

 public:
  explicit coo_workspace_owner_1d(int64_t capacity, bool allocate_in_region = false)
      : _capacity(checked_capacity(capacity)) {
    // The exception `new T[capacity]` throws for a byte size past size_t, as the eager form does.
    const size_t elements = static_cast<size_t>(_capacity);
    if (elements > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
        elements > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    if (!allocate_in_region) {
      allocate_storage();
    }
  }

  coo_workspace_owner_1d(const coo_workspace_owner_1d&) = delete;
  coo_workspace_owner_1d& operator=(const coo_workspace_owner_1d&) = delete;
  coo_workspace_owner_1d(coo_workspace_owner_1d&&) noexcept = default;
  coo_workspace_owner_1d& operator=(coo_workspace_owner_1d&&) noexcept = default;

  // Serial use (a serial kernel's pool of one): may throw while allocating.
  coo_workspace_view_1d<T> make_view() {
    if (!_values) {
      allocate_storage();
    }
    return storage_view();
  }

  // Inside a parallel region: never throws.  An owner without arrays allocates them here; if that
  // fails it records the failure and returns failed_view(), and the region's code skips the
  // worker's iterations.  Each owner is used by one worker per region and the regions are
  // sequential, so the owner's state needs no synchronization.
  coo_workspace_view_1d<T> make_region_view() noexcept {
    if (!_values) {
      if (_allocation_failed || !try_allocate_storage()) {
        _allocation_failed = true;
        return coo_workspace_view_1d<T>::failed_view();
      }
    }
    return storage_view();
  }

  bool allocation_failed() const noexcept { return _allocation_failed; }

 private:
  coo_workspace_view_1d<T> storage_view() noexcept {
    return coo_workspace_view_1d<T>(
        _values.get(), _indices.get(), _set_flags.get(), _capacity);
  }

  void allocate_storage() {
    const size_t elements = static_cast<size_t>(_capacity);
    auto values = scorch_make_unique_array_pool<T>(1, elements);
    auto indices = scorch_make_unique_array_pool<int64_t>(1, elements);
    auto set_flags = scorch_make_unique_array_pool<uint8_t>(1, elements);
    std::fill_n(set_flags.get(), elements, uint8_t{0});
    _values = std::move(values);
    _indices = std::move(indices);
    _set_flags = std::move(set_flags);
  }

  bool try_allocate_storage() noexcept {
    const size_t elements = static_cast<size_t>(_capacity);
    auto values = scorch_make_unique_array_nothrow<T>(elements);
    auto indices = scorch_make_unique_array_nothrow<int64_t>(elements);
    auto set_flags = scorch_make_unique_array_nothrow<uint8_t>(elements);
    if (!values || !indices || !set_flags) {
      return false;
    }
    std::fill_n(set_flags.get(), elements, uint8_t{0});
    _values = std::move(values);
    _indices = std::move(indices);
    _set_flags = std::move(set_flags);
    return true;
  }

  static int64_t checked_capacity(int64_t capacity) {
    if (capacity < 0) {
      throw std::invalid_argument("workspace capacity cannot be negative");
    }
    return capacity;
  }
};

// After a region whose workers took make_region_view(): a worker whose owner could not allocate
// skipped its iterations, so the region's result is incomplete.  Throw here, in serial code, the
// exception an allocation before the region would have thrown.  Owner is any workspace owner
// with `bool allocation_failed() const noexcept` (coo_workspace_owner_1d; one check for every
// such owner, agreed with row 8c's linked_list_workspace_1d).
template <typename Owner>
inline void scorch_check_workspace_pool(const std::vector<Owner>& pool) {
  for (const auto& owner : pool) {
    if (owner.allocation_failed()) {
      throw std::bad_alloc();
    }
  }
}

#undef SCORCH_FORCE_INLINE
#undef SCORCH_RESTRICT

#endif  // SCORCH_WORKSPACE_POOL_H
