#include <unordered_map>
#include <vector>

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

 public:
  explicit coo_workspace(int capacity) {
    _entries.reserve(capacity);
  }

  explicit coo_workspace() : coo_workspace(BLOCK_SIZE) {}

  void insert(const std::vector<int>& coord, T value) {
    int index = coord[N - 1];
    for (int i = N - 2; i >= 0; i--) {
      index = index * N + coord[i];
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
