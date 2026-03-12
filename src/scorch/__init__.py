import torch
from .ops import einsum, matmul, matmul_wksp, precompile_kernels
from .stensor import STensor
from .format import TensorFormat
from .trace import compile

from_torch = STensor.from_torch
from_coo = STensor.from_coo
from_csr = STensor.from_csr


# precompile_kernels()

def __getattr__(name):
    """
    This function is called when an attribute is not found in the module.
    """
    return getattr(torch, name)


__version__ = "0.0.1"

__all__ = [
    "STensor",
    "TensorFormat",
    "compile",
    "einsum",
    "from_torch",
    "from_coo",
    "matmul",
    "matmul_wksp",
    "__version__",
]
