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
  int (*_coords)[N];  // Static array for coordinates
  bool* _setFlags;
  int _size;
  int _capacity;

 public:
  explicit coo_workspace(int capacity) {
    _values = (T*)malloc(sizeof(T) * capacity);
    _coords = (int(*)[N])malloc(sizeof(int) * N * capacity);
    _setFlags = (bool*)calloc(capacity, sizeof(bool));
    _size = 0;
    _capacity = capacity;
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE) {}

  ~coo_workspace() {
    free(_values);
    free(_coords);
    free(_setFlags);
  }

  void insert(const std::vector<int>& coord, T value) {
    int index = coord[N - 1]; // simple hash
    for (int i = N - 2; i >= 0; i--) {
      index = index * N + coord[i];
    }

    if (!_setFlags[index]) {
      _values[index] = value;
      for (int i = 0; i < N; i++) {
        _coords[_size][i] = coord[i];
      }
      _setFlags[index] = true;
      _size++;
    } else {
      _values[index] += value;
    }
  }

  void sort() {
    std::vector<int> indices(_size);
    for (int i = 0; i < _size; i++) {
      indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      for (int i = 0; i < N; i++) {
        if (_coords[a][i] < _coords[b][i]) return true;
        if (_coords[a][i] > _coords[b][i]) return false;
      }
      return false;  // Equal
    });

    // Create sorted arrays
    T* sorted_values = (T*)malloc(sizeof(T) * _size);
    int(*sorted_coords)[N] = (int(*)[N])malloc(sizeof(int) * N * _size);

    for (int i = 0; i < _size; i++) {
      sorted_values[i] = _values[indices[i]];
      for (int j = 0; j < N; j++) {
        sorted_coords[i][j] = _coords[indices[i]][j];
      }
    }

    // Replace old arrays with sorted ones
    free(_values);
    free(_coords);
    _values = sorted_values;
    _coords = sorted_coords;
  }

  void clear() {
    _size = 0;
    std::fill_n(_setFlags, BLOCK_SIZE, false);
  }

  class iterator {
    int _index;
    T* _values;
    int (*_coords)[N];

   public:
    iterator(int index, T* values, int (*coords)[N])
        : _index(index), _values(values), _coords(coords) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<std::vector<int>, T> operator*() const {
      std::vector<int> coord(N);
      for (int i = 0; i < N; i++) {
        coord[i] = _coords[_index][i];
      }
      int index = coord[N - 1]; // simple hash
      for (int i = N - 2; i >= 0; i--) {
        index = index * N + coord[i];
      }
      return {coord, _values[index]};
    }
  };

  iterator begin() { return iterator(0, _values, _coords); }

  iterator end() { return iterator(_size, _values, _coords); }

  int size() const { return _size; }
};

#endif

// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################
