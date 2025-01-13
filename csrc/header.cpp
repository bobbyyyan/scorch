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
  int* _indices;
  bool* _setFlags;
  int _size;
  int _capacity;

 public:
  explicit coo_workspace_1d(int capacity = INITIAL_CAPACITY) : _capacity(capacity) {
    _values = (T*)malloc(sizeof(T) * _capacity);
    if (!_values) throw std::bad_alloc();

    _indices = (int*)malloc(sizeof(int) * _capacity);
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

  void insert(const int coord, T value) {
    if (coord >= _capacity) {
      resize(std::max(coord + 1, _capacity * GROWTH_FACTOR));
    }

    if (!_setFlags[coord]) {
      _values[coord] = value;
      _indices[_size++] = coord;
      _setFlags[coord] = true;
    } else {
      _values[coord] += value;
    }
  }

  void resize(int new_capacity) {
    _values = (T*)realloc(_values, sizeof(T) * new_capacity);
    _indices = (int*)realloc(_indices, sizeof(int) * new_capacity);
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
    std::sort(_indices, _indices + _size, [this](int a, int b) {
      return a < b;
    });
  }

  void clear() {
    _size = 0;
    std::fill_n(_setFlags, _capacity, false);
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

template <typename T, int N>
class coo_workspace {
  static constexpr int BLOCK_SIZE = 1024;

  struct Entry {
    int coords[N];
    T value;
  };

  std::vector<Entry> _entries;
  std::unordered_map<int, int> _existingCoords;
  std::vector<int> _sortedIndices;
  std::vector<int> _resultShape;

 public:
  explicit coo_workspace(int capacity, const std::vector<int> &result_shape)
        : _resultShape(result_shape) {
    _entries.reserve(capacity);
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE) {}

  void insert(const std::vector<int>& coord, T value) {
    int index = coord[0];
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

    std::pair<std::vector<int>, T> operator*() const {
      int sortedIndex = (*_sortedIndices)[_index];
      std::vector<int> coord(_entries[sortedIndex].coords, _entries[sortedIndex].coords + N);
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
