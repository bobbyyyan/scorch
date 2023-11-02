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
