from __future__ import annotations
from scorch.compiler.shapes.jit.ir import *
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.tensor as tensor

import torch


# TODO: This assumes that the function name used in `func` will align with the JIT IR.
# A better approach might be to walk the Python AST, and map these two.
def _compile(func: Callable, region: ScorchRegion, args, kwargs) -> IR:
    i: int = 0

    def id() -> str:
        nonlocal i
        identifier: int = i
        i += 1
        # Generous assumption: no other tensors have this name.
        return f"_T{identifier}"

    def _trace(t: tensor.Tensor | torch.Tensor):
        if isinstance(t, tensor.Tensor):
            return AbstractTensor(t, region)
        assert isinstance(t, torch.Tensor)
        return AbstractTensor(tensor.Tensor.from_torch(t, id()), region)

    def trace(args: Tuple) -> Tuple:
        return tuple(_trace(t) for t in args)

    result: IR = func(*trace(args), **kwargs)
    region.update_result(result)
    region.simplify()
    region.fuse_operations()
    region.dce()
    return result


def compile(func: Optional[Callable]):
    """JIT compilation of Scorch functions; this is similar to JAX,
    i.e., we trace and compile over "abstract" values, and then
    use this to evaluate the real input values.
    """

    def wrapper(*args, **kwargs) -> torch.Tensor:
        name: str = func.__name__
        module: ScorchModule = ScorchModule("module")
        region: ScorchRegion = ScorchRegion(name, module)
        _compile(func, region, args, kwargs)
        return region.torch_evaluate()

    return wrapper
