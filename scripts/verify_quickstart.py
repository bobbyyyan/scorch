import torch
import scorch

# Create STensors from PyTorch tensors
A = scorch.from_torch(torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A")
B = scorch.from_torch(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), "B")

# Matrix multiply
C = scorch.matmul(A, B)
print(C)

# Einstein summation
D = scorch.einsum("ij,jk->ik", A, B)
print(D.to_torch())
