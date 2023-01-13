from typing import Any, Dict, List, Optional, Tuple, Union

from copy import deepcopy

import numpy as numpy
import scipy.sparse
import torch

from .tensor import TacoTensor
from .storage import TensorStorage, TensorIndex, TensorFormat

from torch.utils.cpp_extension import load, load_inline

from pathlib import Path


PROJECT_ROOT_DIR = Path(__file__).parent.parent

ops_cpp = load(
    name="ops_cpp",
    sources=[str(PROJECT_ROOT_DIR / "csrc/ops.cpp")],
)


def taco_einsum(
    expression: str,
    *tensors: Union[torch.Tensor, TacoTensor],
    **kwargs: Any,
) -> TacoTensor:
    """Perform a tensor contraction using the TACO compiler."""
    print(f"Evaluating expression: {expression}")
    print("Tensors:", tensors)
    raise NotImplementedError("TACO einsum is not implemented yet.")


def mul(src: TacoTensor, other: TacoTensor):
    """Multiply two tensors.
    e.g. `mul(a, b)` is equivalent to `a * b`.
    """
    # TODO: Lower to TACO IR
    # TODO: Compile to C++ code
    # TODO: (Inline) load C++ code using PyTorch's C++ extension
    # TODO: Call C++ code
    # TODO: Return result
    ttensor = ops_cpp.TacoTensor

    result = ops_cpp.elemwise_mul(
        src._storage._index.mode_indices,
        src._storage.value,
        other._storage._index.mode_indices,
        other._storage.value,
    )

    print("ops_cpp.TacoTensor", ops_cpp.TacoTensor)

    return TacoTensor(
        storage=TensorStorage(
            index=TensorIndex(
                mode_indices=result._storage._index.mode_indices,
            ),
            value=result._storage._value,
        )
    )

    raise NotImplementedError("TACO mul is not implemented yet.")


def add(src, other):
    # print(ops_cpp.add)
    # TODO: implement this
    # return TacoTensor(
    #     value=ops_cpp.add(src.value, other.value),
    # )
    # return ops_cpp.add(src, other)

    # Lower to TACO IR
    # Compile to C++ code
    # (Inline) load C++ code using PyTorch's C++ extension
    # Call C++ code
    # Return result

    if isinstance(other, TacoTensor):
        raise NotImplementedError("TACO add is not implemented yet.")


TacoTensor.add = add
TacoTensor.__add__ = add
TacoTensor.__mul__ = mul
