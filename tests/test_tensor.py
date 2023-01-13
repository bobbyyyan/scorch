from taco_torch import TacoTensor
from torch.utils import cpp_extension

for path in cpp_extension.include_paths():
    print(f'"{path}",')

print(TacoTensor(name="A"))


print("Test passed!")
