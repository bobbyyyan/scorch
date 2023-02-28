import torch
from src.scorch import TacoTensor
from src.scorch.format import TensorFormat, LevelFormat, LevelType
from src.scorch.storage import TensorStorage, TensorIndex

A = TacoTensor(name="A", value=torch.randn(3, 4))
B = TacoTensor(name="B", value=torch.randn(3, 4))

# Create a Compressed sparse vector A that has name "vector_a" and shape (4,),
# and the index and value for [1, 0, 2, 4]
vector_a = TacoTensor(
    name="vector_a",
    storage=TensorStorage(
        index=TensorIndex(
            tensor_format=TensorFormat(
                level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
            ),
            mode_indices=[torch.Tensor([0, 3]), torch.Tensor([0, 2, 3])],
        ),
        value=torch.Tensor([1, 2, 4]),
        shape=(4,),
    ),
)

# Create a Compressed sparse vector B that has name "vector_b" and shape (4,),
# and the index and value for [8, 1, 6, 0]
vector_b = TacoTensor(
    name="vector_b",
    storage=TensorStorage(
        index=TensorIndex(
            tensor_format=TensorFormat(
                level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
            ),
            mode_indices=[torch.Tensor([0, 3]), torch.Tensor([0, 1, 2])],
        ),
        value=torch.Tensor([8, 1, 6]),
        shape=(4,),
    ),
)

# Create a Compressed sparse vector C that has name "vector_c" and shape (4,),
# and the index and value that will be computed
vector_c = TacoTensor(
    name="vector_c",
    storage=TensorStorage(
        index=TensorIndex(
            tensor_format=TensorFormat(
                level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
            ),
            mode_indices=[],
        ),
        value=None,
        shape=(4,),
    ),
)

# print(A.value)
# print(B.value)

# TODO generators for random tensors

A_value_t_tensor: torch.Tensor = A.value
# Print A_value_t_tensor's dtype
# print(A_value_t_tensor.dtype)


# Print A_value_t_tensor's dtype
# print(A_value_t_tensor.dtype)


vector_c = vector_a * vector_b

print("vector_c")
print(vector_c)
