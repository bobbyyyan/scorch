#include <atomic>
#include <omp.h>
#include <cstdlib>
#include <cstdio>

typedef struct {
  std::vector<std::vector<torch::Tensor>> mode_indices;
} TensorIndex;

typedef struct {
  TensorIndex _index;
  // a list of torch::Tensor
  torch::Tensor _value;

} TensorStorage;

typedef struct {
  TensorStorage _storage;
  // a tuple of ints as shape
  std::vector<int> _shape;
} Tensor;

// ####################################
// ===== BEGIN === VECTOR IMPL ========
// ####################################

template<typename T>
class cvector {
    int _size;
    int _capacity;
    T *_data;

public:

    // default constructor
    cvector() {
        _size = 0;
        _capacity = 1;
        _data = (T*) malloc(sizeof(T));
    }

    // constructor with capacity
    cvector(int capacity) {
        _size = 0;
        _capacity = capacity;
        _data = (T*) malloc(sizeof(T) * capacity);
    }

    // destructor is not used
    /**
    ~cvector() {
        free(_data);
    }
    */

    // function to append an element to the vector
    inline void push_back(T element) {
        if (_size == _capacity) {
            T *temp = (T*) malloc(sizeof(T) * 2 * _capacity);
            memcpy(temp, _data, sizeof(T) * _size);
            free(_data);
            _capacity *= 2;
            _data = temp;
        }
        _data[_size] = element;
        _size++;
    }

    // function to get the size of the vector
    inline int size() {
        return _size;
    }

    // function to get the capacity of the vector
    inline int capacity() {
        return _capacity;
    }

    // function to get the element at a particular index
    inline T get(int index) {
        if (index < _size) {
            return _data[index];
        } else {
            return _data[0];
        }
    }

    // function to set the element at a particular index
    inline void set(int index, T element) {
        if (index < _size) {
            _data[index] = element;
        }
    }

    // function to read the element
  // without checking the index
  inline T get_unsafe(int index) { return _data[index]; }

  // function to set the element
  // without checking the index
  inline void set_unsafe(int index, T element) { _data[index] = element; }

    // overload operator [] to get elements
    inline T operator[](int index) const {
        return get(index);
    }

    // overload operator [] to set elements, e.g. vec[0] = 1;
    inline T &operator[](int index) {
        // if index is out of range, resize the vector
        if (index >= _capacity) {
            resize(2 * index);  // Double the requested size
        }
        if (index >= _size) {
            _size = index + 1;
        }
        return _data[index];
    }

    // function to change the capacity of the vector
    inline void resize(int new_capacity) {
        // if new_capacity is smaller than the current size, do nothing
        if (new_capacity <= _capacity) {
            return;
        }
        // if new_capacity is larger than the current size
        // resize the vector
        T *temp = (T*) malloc(sizeof(T) * new_capacity);
        memcpy(temp, _data, sizeof(T) * _size);
        // free the old data
        free(_data);
        _data = temp;
        _capacity = new_capacity;
    }


    // function to remove the last element of the vector
    inline void pop_back() {
        if (_size > 0) {
            _size--;
        }
    }

    // function to remove the element at a particular index
    inline void remove(int index) {
        if (index < _size) {
            // for (int i = index; i < _size - 1; i++) {
            //     _data[i] = _data[i + 1];
            // }
            // use memcpy for better performance
             memcpy(_data + index, _data + index + 1, sizeof(T) * (_size - index - 1));
            _size--;
        }
    }

    // function to insert an element at a particular index
    inline void insert(int index, T element) {
        if (index < _size) {
            if (_size == _capacity) {
                T* temp = (T*) malloc(sizeof(T) * 2 * _capacity);
                memcpy(temp, _data, sizeof(T) * _size);
                free(_data);
                _capacity *= 2;
                _data = temp;
            }
            // for (int i = _size; i > index; i--) {
            //     _data[i] = _data[i - 1];
            // }
            // use memcpy for better performance
            memcpy(_data + index + 1, _data + index, sizeof(T) * (_size - index));
            _data[index] = element;
            _size++;
        }
    }

    // function to clear the vector
    inline void clear() {
        _size = 0;
    }

    // function to check if the vector is empty
    inline bool empty() {
        return _size == 0;
    }

    // function to get the pointer to the data
    inline T* data() {
        return _data;
    }

    // function to return a lambda function
    // std::function<void(void*)>
    // that would deallocate the data
    inline std::function<void(void *)> get_deleter() {
        return [](void *data) {
            free(data);
        };
    }

    // function to get the last element of the vector
    inline T back() {
        return _data[_size - 1];
    }

};


// ####################################
// ====== END ==== VECTOR IMPL ========
// ####################################



// ####################################
// ==== BEGIN === ACTIVATION HELPERS ===
// ####################################

template<typename T> inline T scorch_relu(T x) { return x > T(0) ? x : T(0); }
template<typename T> inline T scorch_sigmoid(T x) { return T(1) / (T(1) + std::exp(-x)); }
template<typename T> inline T scorch_tanh(T x) { return std::tanh(x); }
template<typename T> inline T scorch_gelu(T x) {
  return x * T(0.5) * (T(1) + std::tanh(T(0.7978845608) * (x + T(0.044715) * x * x * x)));
}

// ####################################
// ===== END === ACTIVATION HELPERS ====
// ####################################


// ####################################
// == BEGIN == PARALLEL POLICY HELPERS ==
// ####################################
//
// Work-aware OpenMP thread cap + adaptive schedule chunk, shared by JIT-
// generated kernels (emitted by compiler/codegen.py). Same lesson as the
// prebuilt spmspm/spmm kernels: unconditional all-cores over-threads small
// products and a coarse fixed chunk starves load-balance on hybrid P+E CPUs.
//
// Env override hooks (for tuning; unset = built-in default):
//   SCORCH_CG_THREADS  force nthreads
//   SCORCH_CG_CHUNK    force schedule chunk
//   SCORCH_CG_GRAIN    min work (nnz) per thread (default 500)
//   SCORCH_CG_ROWQ     min rows per thread (default 16)
//   SCORCH_CG_CPT      target chunks per thread (default 7)
//
// Tuning (redwood i9-14900K, ds-path codegen SpGEMM, validated back-to-back vs the
// old all-cores+chunk64 policy): GRAIN=500 + ROWQ=16 + CPT=7 beats the old policy on
// a 7-size dense panel (~0.70x runtime) AND on ultra-sparse ~1nnz/row products
// (0.81-0.92x), capturing ~91% of the per-size threads x chunk oracle. `work` here
// is A_nnz (outer-operand nnz), so GRAIN is ~avg_B_row smaller than the prebuilt
// kernels' flop-based GRAIN=3000. The nnz work bound throttles tiny/ultra-sparse
// products (the documented over-threading risk); rows/16 drives normal matrices to
// ~all cores now that the adaptive fine chunk removed the P+E over-threading cliff.
//   SCORCH_CG_DEBUG    print the (work,rows)->(nt,chunk) decision to stderr

inline long _scorch_env_long(const char* name, long dflt) {
  const char* e = getenv(name);
  if (e && *e) { long v = atol(e); if (v > 0) return v; }
  return dflt;
}

// work < 0 means "unknown" -> cap by rows only.
inline int scorch_nthreads(long work, long rows) {
  const char* forced = getenv("SCORCH_CG_THREADS");
  if (forced && *forced) { int v = atoi(forced); if (v > 0) return v; }
  int hw = omp_get_num_procs();          // stable; torch mutates omp_get_max_threads
  long rowq = _scorch_env_long("SCORCH_CG_ROWQ", 16);
  long n = rows / rowq;
  if (work >= 0) {
    long grain = _scorch_env_long("SCORCH_CG_GRAIN", 500);
    long by_work = work / grain;
    if (by_work < n) n = by_work;
  }
  if (n < 1) n = 1;
  if (n > (long)hw) n = hw;
  return (int)n;
}

inline int scorch_chunk(long rows, long work) {
  int nt = scorch_nthreads(work, rows);
  const char* forced = getenv("SCORCH_CG_CHUNK");
  int chunk;
  if (forced && *forced) { chunk = atoi(forced); if (chunk < 1) chunk = 1; }
  else {
    long cpt = _scorch_env_long("SCORCH_CG_CPT", 7);
    long c = rows / (nt * cpt);
    if (c < 4) c = 4;
    if (c > 64) c = 64;
    chunk = (int)c;
  }
  if (getenv("SCORCH_CG_DEBUG")) {
    fprintf(stderr, "[scorch_cg] work=%ld rows=%ld -> nt=%d chunk=%d\n",
            work, rows, nt, chunk);
  }
  return chunk;
}

// ####################################
// === END === PARALLEL POLICY HELPERS ==
// ####################################


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

#ifndef SPARSE_ML_COO_WORKSPACE_H
#define SPARSE_ML_COO_WORKSPACE_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>

template <typename T, int N>
class coo_workspace_1d {
  static constexpr int INITIAL_CAPACITY = 1024;
  static constexpr int GROWTH_FACTOR = 2;
  static constexpr int BLOCK_SIZE = N;

  T* _values;
  int64_t* _indices;
  bool* _setFlags;
  int64_t _size;
  int64_t _capacity;

 public:
  explicit coo_workspace_1d(int64_t capacity = INITIAL_CAPACITY) : _capacity(capacity) {
    _values = (T*)malloc(sizeof(T) * _capacity);
    if (!_values) throw std::bad_alloc();

    _indices = (int64_t*)malloc(sizeof(int64_t) * _capacity);
    if (!_indices) {
      free(_values);
      throw std::bad_alloc();
    }

    _setFlags = (bool*)calloc(_capacity, sizeof(bool));
    if (!_setFlags) {
      free(_values);
      free(_indices);
      throw std::bad_alloc();
    }

    _size = 0;
  }

  explicit coo_workspace_1d() : coo_workspace_1d(BLOCK_SIZE) {}

  ~coo_workspace_1d() {
    free(_values);
    free(_indices);
    free(_setFlags);
  }

  void insert(int64_t coord, T value) {
    if (coord >= _capacity) {
      resize(std::max(coord + 1, _capacity * (int64_t)GROWTH_FACTOR));
    }

    if (!_setFlags[coord]) {
      _values[coord] = value;
      _indices[_size++] = coord;
      _setFlags[coord] = true;
    } else {
      _values[coord] += value;
    }
  }

  void resize(int64_t new_capacity) {
    _values = (T*)realloc(_values, sizeof(T) * new_capacity);
    _indices = (int64_t*)realloc(_indices, sizeof(int64_t) * new_capacity);
    bool* new_setFlags = (bool*)realloc(_setFlags, sizeof(bool) * new_capacity);

    if (!_values || !_indices || !new_setFlags) {
        throw std::bad_alloc();
    }

    // Initialize the newly allocated memory for _setFlags to false
    std::fill(new_setFlags + _capacity, new_setFlags + new_capacity, false);

    _setFlags = new_setFlags;
    _capacity = new_capacity;
  }

  void sort() {
    std::sort(_indices, _indices + _size, [this](int64_t a, int64_t b) {
      return a < b;
    });
  }

  void clear() {
    for (int64_t i = 0; i < _size; i++) {
      _setFlags[_indices[i]] = false;
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

  iterator begin() { return iterator(0, _values, _indices); }

  iterator end() { return iterator(_size, _values, _indices); }

  int64_t size() const { return _size; }
};

template <typename T>
class linked_list_workspace_1d {
  T* _sums;
  int64_t* _next;
  int64_t _head;
  int64_t _size;
  int64_t _capacity;

 public:
  explicit linked_list_workspace_1d(int64_t capacity)
      : _head(-2), _size(0), _capacity(capacity) {
    _sums = (T*)calloc(_capacity, sizeof(T));
    _next = (int64_t*)malloc(sizeof(int64_t) * _capacity);
    _sorted_buf = (int64_t*)malloc(sizeof(int64_t) * _capacity);
    if (!_sums || !_next || !_sorted_buf) throw std::bad_alloc();
    std::fill_n(_next, _capacity, (int64_t)-1);
  }

  ~linked_list_workspace_1d() {
    free(_sums);
    free(_next);
    free(_sorted_buf);
  }

  inline void insert(int64_t coord, T value) {
    _sums[coord] += value;
    if (_next[coord] == -1) {
      _next[coord] = _head;
      _head = coord;
      _size++;
    }
  }

  void sort() {}

  void clear() {
    int64_t h = _head;
    while (h >= 0) {
      int64_t tmp = h;
      h = _next[h];
      _next[tmp] = -1;
      _sums[tmp] = 0;
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

  int64_t* _sorted_buf;
  int64_t _sorted_size = 0;

  iterator begin() {
    int64_t pos = 0, h = _head;
    while (h >= 0) {
      _sorted_buf[pos++] = h;
      h = _next[h];
    }
    _sorted_size = pos;
    std::sort(_sorted_buf, _sorted_buf + _sorted_size);
    return iterator(0, _sorted_buf, _sums);
  }

  iterator end() { return iterator(_sorted_size, _sorted_buf, _sums); }
};

template <typename T, int N>
class coo_workspace {
  static constexpr int BLOCK_SIZE = 1024;

  struct Entry {
    int64_t coords[N];
    T value;
  };

  std::vector<Entry> _entries;
  std::unordered_map<int64_t, int> _existingCoords;
  std::vector<int> _sortedIndices;
  std::vector<int64_t> _resultShape;

 public:
  explicit coo_workspace(int capacity, const std::vector<int64_t> &result_shape)
        : _resultShape(result_shape) {
    _entries.reserve(capacity);
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE, {}) {}

  void insert(const std::vector<int64_t>& coord, T value) {
    int64_t index = coord[0];
    for (int i = 1; i < N; i++){
        index = index * _resultShape[i] + coord[i];
    }

    auto existingCoordIt = _existingCoords.find(index);
    if (existingCoordIt != _existingCoords.end()) {
      _entries[existingCoordIt->second].value += value;
      return;
    }

    Entry entry;
    std::copy(coord.begin(), coord.end(), entry.coords);
    entry.value = value;
    _entries.push_back(entry);

    _existingCoords[index] = _entries.size() - 1;
  }

  void sort() {
    _sortedIndices.resize(_entries.size());
    std::iota(_sortedIndices.begin(), _sortedIndices.end(), 0);

    auto radixComparator = [this](int a, int b) {
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
    int _index;
    std::vector<Entry>& _entries;
    std::vector<int>* _sortedIndices;

   public:
    iterator(int index, std::vector<Entry>& entries, std::vector<int>* sortedIndices)
        : _index(index), _entries(entries), _sortedIndices(sortedIndices) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<std::vector<int64_t>, T> operator*() const {
      int sortedIndex = (*_sortedIndices)[_index];
      std::vector<int64_t> coord(_entries[sortedIndex].coords, _entries[sortedIndex].coords + N);
      return {coord, _entries[sortedIndex].value};
    }
  };

  iterator begin() {
    return iterator(0, _entries, &_sortedIndices);
  }

  iterator end() {
    return iterator(_entries.size(), _entries, &_sortedIndices);
  }

  int size() const { return _entries.size(); }

  int capacity() const { return _entries.capacity(); }
};

#endif  // SPARSE_ML_COO_WORKSPACE_H


// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################
