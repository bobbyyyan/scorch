import torch

from src.scorch import TacoTensor
from torch.utils import cpp_extension

for path in cpp_extension.include_paths():
    print(f'"{path}",')


def test_torch_tensor_to_tt_tensor():
    torch_tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    tt_tensor = TacoTensor.from_torch(torch_tensor)
    # assert torch.allclose(torch_tensor, tt_tensor.to_torch_tensor())


print("Test passed!")
