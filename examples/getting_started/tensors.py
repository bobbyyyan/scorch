import scorch as torch

# Creating a sparse vector from a random dense vector,
# stored in coordinate format
dense_vector = torch.rand(100)
sparse_vector = dense_vector.to_sparse("o")

# Creating a sparse matrix from a random dense matrix,
# stored in CSR format (dense, sparse)
dense_matrix = torch.rand(100, 100)
sparse_matrix = dense_matrix.to_sparse("ds")
