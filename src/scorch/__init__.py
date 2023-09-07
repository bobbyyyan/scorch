from .ops import einsum, matmul, matmul_wksp, precompile_kernels
from .tensor import Tensor
from .format import TensorFormat

from_torch = Tensor.from_torch
from_coo = Tensor.from_coo

precompile_kernels()

__version__ = "0.0.1"

__all__ = [
    "Tensor",
    "TensorFormat",
    "einsum",
    "from_torch",
    "from_coo",
    "matmul",
    "matmul_wksp",
    "__version__",
]
