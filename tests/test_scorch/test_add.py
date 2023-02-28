import torch
from src.scorch import Tensor

A = Tensor(name="A", value=torch.randn(3, 4))
B = Tensor(name="B", value=torch.randn(3, 4))

print(A.value)

print(B.value)

# TODO generators for random tensors

# C_i = A_i + B_i
# C = Tensor(name="C")
# Method 1: Use taco_einsum
# taco_einsum("i,i->i", A, B, C)
# OR taco_einsum variant
# C = taco_einsum("i,i->i", A, B)
# Method 2: Use __add__
C = A + B

print(C.value)

print("Test passed!")
