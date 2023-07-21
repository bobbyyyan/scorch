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
        _data = new T[_capacity];
    }

    // constructor with capacity
    cvector(int capacity) {
        _size = 0;
        _capacity = capacity;
        _data = new T[_capacity];
    }

    // destructor
    /**
    ~cvector() {
        delete[] _data;
    }
    */

    // function to append an element to the vector
    inline void push_back(T element) {
        if (_size == _capacity) {
            T *temp = new T[2 * _capacity];
            for (int i = 0; i < _capacity; i++) {
                temp[i] = _data[i];
            }
            delete[] _data;
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
        // if new_capacity is larger than the current size, resize the vector
        T *temp = new T[new_capacity];
        for (int i = 0; i < _size; i++) {
            temp[i] = _data[i];
        }
        delete[] _data;
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
            for (int i = index; i < _size - 1; i++) {
                _data[i] = _data[i + 1];
            }
            _size--;
        }
    }

    // function to insert an element at a particular index
    inline void insert(int index, T element) {
        if (index < _size) {
            if (_size == _capacity) {
                T *temp = new T[2 * _capacity];
                for (int i = 0; i < _capacity; i++) {
                    temp[i] = _data[i];
                }
                delete[] _data;
                _capacity *= 2;
                _data = temp;
            }
            for (int i = _size; i > index; i--) {
                _data[i] = _data[i - 1];
            }
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
    inline T *data() {
        return _data;
    }

    // function to return a lambda function / std::function<void(void*)> that would deallocate the data
    inline std::function<void(void *)> get_deleter() {
        return [](void *data) {
            delete[] (T *) data;
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

#include <map>
#include <vector>

/**
 * This class implements a workspace to store the intermediate results of tensor
 * operations. It keeps track of the intermediate tensor in the coordinate list
 * format.
 * For a N-dimensional tensor, the workspace stores N lists of indices.
 * The class has a sort function that sorts the indices in the lists.
 *
 * @tparam T type of the values stored
 */
template <typename T>
class coo_workspace {
  // dimension of the tensor
  int _dim;
  // number of non-zero elements in the tensor
  int size = 0;
  // flat array to store the values of the tensor
  std::vector<T> _values;
  // A vector of _dim lists of indices
  std::vector<std::vector<int>> _indices;

  // custom comparator for the map's keys
  // sort the keys, by the array's first element, then the second element, and
  // so on
  struct compare {
    bool operator()(const std::vector<int> &a,
                    const std::vector<int> &b) const {
      for (int i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) {
          return true;
        } else if (a[i] > b[i]) {
          return false;
        }
      }
      return false;
    }
  };

  // map from coordinates to values, should be exposed to the user
  std::map<std::vector<int>, T, compare> _map;

 public:
  // Constructor
  explicit coo_workspace(int dim) : _dim(dim) {
    _dim = dim;
    // Create _dim lists of indices
    for (int i = 0; i < _dim; i++) {
      _indices.emplace_back();
    }
  }
  // Function to insert a coordinate-value pair into the workspace
  void insert(std::vector<int> coord, T value) {
    // if the coordinate is already in the map, add the value to the existing
    // value
    if (_map.count(coord)) {
      _map[coord] += value;
    } else {
      _map[coord] = value;
    }
  }
  // Expose the map to the user
  std::map<std::vector<int>, T, compare> &get_map() { return _map; }
};

// ####################################
// ====== END ==== COO WKSP IMPL ======
// ####################################