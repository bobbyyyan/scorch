import torch
from taco_torch import TacoTensor, taco_einsum
from taco_torch.cin import TensorAssign, MulOp, TensorAccess, IndexVar

A = TacoTensor(name="A", value=torch.randn(3, 4))
B = TacoTensor(name="B", value=torch.randn(3, 4))

print(A.value)

print(B.value)

# TODO generators for random tensors

# C_i = A_i + B_i
# C = TacoTensor(name="C")
# Method 1: Use taco_einsum
# taco_einsum("i,i->i", A, B, C)
# OR taco_einsum variant
# C = taco_einsum("i,i->i", A, B)
# Method 2: Use __add__
C = A + B

print(C.value)

print("Test passed!")
