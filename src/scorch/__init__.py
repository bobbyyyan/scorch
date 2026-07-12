import torch
from .ops import (
    einsum,
    fast_transpose,
    matmul,
    matmul_wksp,
    precompile_kernels,
    sparse_attention,
    sparse_linear,
    sparse_linear_fm,
    sparse_softmax_csr,
)
from .stensor import STensor
from .format import TensorFormat
from .compiler.scheduler import RelayoutSpec, Schedule, TileSpec, schedule_force
from .trace import compile
from .tiling import (
    autotune,
    clear_autotune_cache,
    compiler_schedule_search_space,
    get_autotune,
    schedule_from_tuner_choice,
    set_autotune,
)

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
    "RelayoutSpec",
    "Schedule",
    "TileSpec",
    "autotune",
    "clear_autotune_cache",
    "compiler_schedule_search_space",
    "compile",
    "einsum",
    "fast_transpose",
    "from_torch",
    "from_coo",
    "get_autotune",
    "matmul",
    "matmul_wksp",
    "schedule_force",
    "schedule_from_tuner_choice",
    "set_autotune",
    "sparse_attention",
    "sparse_linear",
    "sparse_linear_fm",
    "sparse_softmax_csr",
    "__version__",
]
