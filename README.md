# Scorch

Scorch is a Python library for sparse machine learning, built on top of PyTorch. It provides sparse implementations of key PyTorch operations, allowing you to work with sparse tensors seamlessly.

## Getting Started

To get started with Scorch, follow these steps:

1. Clone the Scorch repository and change into the project directory:
   ```shell
   git clone <repository-url>
   cd scorch
   ```

2. Create a new conda environment and install the required dependencies:
   ```shell
   conda create -n scorch python=3.11
   conda activate scorch
   pip install -r requirements.txt
   ```

3. Install Scorch and its dependencies:
   ```shell
   pip install .
   ```

## Usage

To use Scorch in your PyTorch projects, simply import it as follows:

```python
import scorch as torch
```

With this import statement, you can use Scorch's sparse implementations of PyTorch operations. For example:

```python
# Create sparse tensors
sparse_tensor1 = torch.sparse_coo_tensor(...)
sparse_tensor2 = torch.sparse_coo_tensor(...)

# Perform sparse matrix multiplication
result = torch.matmul(sparse_tensor1, sparse_tensor2)

# Perform sparse einsum operation
result = torch.einsum('ij,jk->ik', sparse_tensor1, sparse_tensor2)
```
