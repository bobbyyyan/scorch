from .ops import einsum, matmul, matmul_wksp, precompile_kernels
from .tensor import Tensor
from .format import TensorFormat

precompile_kernels()

__version__ = "0.0.1"

__all__ = [
    "Tensor",
    "TensorFormat",
    "einsum",
    "matmul",
    "matmul_wksp",
    "__version__",
]
