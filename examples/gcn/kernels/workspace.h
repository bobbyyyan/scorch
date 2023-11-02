// ####################################
// ===== BEGIN === COO WKSP IMPL ======
// ####################################

/**
 * This class implements a workspace to store the intermediate results of tensor
 * operations. It keeps track of the intermediate tensor in the coordinate list
 * format.
 * - It provides an interface to insert a coordinate-value pair into the
 *   workspace.
 *   - If the coordinate already exists in the workspace, the new value would be
 *     accumulated to the existing value by addition.
 * - It also provides an interface to iterate through the coordinate-value
 *   pairs in the order sorted by the coordinates.
 *   - For example, for a 3-dimensional workspace, the coordinate-value pairs
 *     could be:
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

#ifndef SCORCH_COO_WORKSPACE_H
#define SCORCH_COO_WORKSPACE_H

#include <algorithm>
#include <cstdlib>
#include <vector>

template <typename T, int N>
class coo_workspace {
  static constexpr int BLOCK_SIZE = 1024;

  T* _values;
  int* _indices;
  bool* _setFlags;
  int _size;

 public:
  explicit coo_workspace(int capacity) {
    _values = (T*)malloc(sizeof(T) * capacity);
    _indices = (int*)malloc(sizeof(int) * capacity);
    _setFlags = (bool*)calloc(capacity, sizeof(bool));
    _size = 0;
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE) {}

  ~coo_workspace() {
    free(_values);
    free(_indices);
    free(_setFlags);
  }

  void insert(const std::vector<int>& coord, T value) {
    int index = coord[N - 1];
    for (int i = N - 2; i >= 0; i--) {
      index = index * N + coord[i];
    }

    if (!_setFlags[index]) {
      _values[index] = value;
      _indices[_size] = index;
      _setFlags[index] = true;
      _size++;
    } else {
      _values[index] += value;
    }
  }

  void insert(const int coord, T value) {
    if (!_setFlags[coord]) {
      _values[coord] = value;
      _indices[_size] = coord;
      _setFlags[coord] = true;
      _size++;
    } else {
      _values[coord] += value;
    }
  }

  void sort() {
    std::qsort(_indices, _size, sizeof(int), [](const void* a, const void* b) {
      return *(const int*)a - *(const int*)b;
    });
  }

  void clear() {
    _size = 0;
    std::fill_n(_setFlags, BLOCK_SIZE, false);
  }

  class iterator {
    int _index;
    T* _values;
    int* _indices;

   public:
    iterator(int index, T* values, int* indices)
        : _index(index), _values(values), _indices(indices) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<int, T> operator*() const {
      int index = _indices[_index];
      return {index, _values[index]};
    }
  };

  iterator begin() { return iterator(0, _values, _indices); }

  iterator end() { return iterator(_size, _values, _indices); }

  int size() const { return _size; }
};

#endif

// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################
