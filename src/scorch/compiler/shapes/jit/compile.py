from __future__ import annotations
from scorch.compiler.shapes.jit.ir import *
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.tensor as tensor

import torch


def _compile(func: Callable, region: ScorchRegion, args, kwargs):
    i: int = 0

    def id() -> str:
        nonlocal i
        identifier: int = i
        i += 1
        return f"_T{identifier}"

    def _trace(t: tensor.Tensor | torch.Tensor):
        if isinstance(t, tensor.Tensor):
            return AbstractTensor(t, region)
        assert isinstance(t, torch.Tensor)
        return AbstractTensor(tensor.Tensor.from_torch(t, id()), region)

    def trace(args: Tuple) -> Tuple:
        return tuple(_trace(t) for t in args)

    result: tensor.Tensor = func(*trace(args), **kwargs)
    simplify(region)
    return result


def compile(func):
    """JIT compilation of Scorch functions; this is similar to JAX,
       i.e., we trace and compile over "abstract" values, and then
       use this to evaluate the real input values.
    """
    def wrapper(*args, **kwargs) -> torch.Tensor:
        name: str = func.__name__
        region: ScorchRegion = ScorchRegion(name)
        result: tensor.Tensor = _compile(func, region, args, kwargs)
        return region.torch_evaluate(result)

    return wrapper
