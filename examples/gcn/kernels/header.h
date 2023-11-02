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
