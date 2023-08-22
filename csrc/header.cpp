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
 * For a N-dimensional tensor, the workspace stores N lists of indices.
 * The class also provides an interface to iterate through the coordinates-value
 * pairs in the order sorted by the indices (coordinates)
 * e.g. for a 3-dimensional workspace, the coordinates-value pairs could be:
 * (0, 0, 0) - 11
 * (1, 1, 1) - 22
 * (1, 1, 0) - 33
 * Then the iterator would return the pairs in the order:
 * (0, 0, 0) - 11
 * (1, 1, 0) - 33
 * (1, 1, 1) - 22
 *
 * @tparam T type of the values stored, e.g. float, double, int, etc.
 */

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

template <typename T>
class coo_workspace {
  static constexpr int BLOCK_SIZE = 1024;

  int _dim;
  std::vector<T> _values;
  std::vector<int> _indices;
  std::unordered_map<int, int> _existingCoords;
  std::vector<int> _sortedIndices;

 public:
  coo_workspace(int dim, int capacity)
      : _dim(dim) {
    _values.reserve(capacity);
    _indices.reserve(capacity * _dim);
  }

  explicit coo_workspace(int dim) : coo_workspace(dim, BLOCK_SIZE) {}

  void insert(const std::vector<int>& coord, T value) {
    int index = coord[_dim - 1];
    for (int i = _dim - 2; i >= 0; i--) {
      index = index * _dim + coord[i];
    }

    auto existingCoordIt = _existingCoords.find(index);
    if (existingCoordIt != _existingCoords.end()) {
      _values[existingCoordIt->second] += value;
      return;
    }

    _values.push_back(value);
    for (int i = 0; i < _dim; i++) {
      _indices.push_back(coord[i]);
    }

    _existingCoords[index] = _values.size() - 1;
  }

  void sort() {
    _sortedIndices.resize(_values.size());
    for (int i = 0; i < _values.size(); i++) {
      _sortedIndices[i] = i;
    }

    auto radixComparator = [this](int a, int b) {
      for (int i = 0; i < _dim; i++) {
        int coordA = _indices[i * _values.size() + a];
        int coordB = _indices[i * _values.size() + b];
        if (coordA != coordB) {
          return coordA < coordB;
        }
      }
      return false;
    };
    std::sort(_sortedIndices.begin(), _sortedIndices.end(), radixComparator);
  }

  class iterator {
    int _index;
    std::vector<T>& _values;
    std::vector<int>& _indices;
    int _dim;
    std::vector<int>* _sortedIndices;

   public:
    iterator(int index, std::vector<T>& values, std::vector<int>& indices, int dim,
             std::vector<int>* sortedIndices)
        : _index(index), _values(values), _indices(indices), _dim(dim),
          _sortedIndices(sortedIndices) {}

    iterator& operator++() {
      _index++;
      return *this;
    }

    bool operator!=(const iterator& other) const {
      return _index != other._index;
    }

    std::pair<std::vector<int>, T> operator*() const {
      int sortedIndex = (*_sortedIndices)[_index];
      std::vector<int> coord(_dim);
      for (int i = 0; i < _dim; i++) {
        coord[i] = _indices[i * _values.size() + sortedIndex];
      }
      return {coord, _values[sortedIndex]};
    }
  };

  iterator begin() {
    return iterator(0, _values, _indices, _dim, &_sortedIndices);
  }

  iterator end() {
    return iterator(_values.size(), _values, _indices, _dim, &_sortedIndices);
  }

  int size() const { return _values.size(); }

  int capacity() const { return _values.capacity(); }
};

// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################