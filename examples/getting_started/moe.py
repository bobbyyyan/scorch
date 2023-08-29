import torch
from torch.nn import Functional as F

# Inputs, (B, D_in)
x = torch.randn(B, D_in)
# Expert embeddings, (N_experts, D_in, D_out)
E = torch.randn(N_experts, D_in, D_out)
# Sparse gating function, (B, N_experts)
gates = torch.rand(B, N_experts)
# Select one expert per input
gates = F.one_hot(gates.argmax(1), N_experts).to_sparse()
# Dispatch inputs to experts, (B, N_experts, D_in)
x_dispatch = torch.rearrange(x, "bd->bnd", n=N_experts)
# Apply experts, (B, N_experts, D_out)
y_experts = torch_einsum("bnd,ndh->bnh", x_dispatch, E)
# Combine expert outputs, (B, D_out)
y = torch.einsum("bnd,bn->bd", y_experts, gates)
