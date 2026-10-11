import torch

import scorch

A_dense = torch.tensor(
    [
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
    ]
)
B = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)

A = scorch.from_torch(A_dense, "A").to_sparse("ds")

C = scorch.matmul(A, B)
expected = A_dense @ B
assert torch.allclose(C, expected, atol=1e-3, rtol=1e-3)

D = scorch.einsum("ij,jk->ik", A, B, format="dd")
assert torch.allclose(D.to_torch(), expected, atol=1e-3, rtol=1e-3)

print(C)
print(D.to_torch())
