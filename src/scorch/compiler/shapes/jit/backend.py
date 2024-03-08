import torch
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.tensor as tensor
import operator
from dataclasses import dataclass

# Experimenting with PyTorch2's TorchDynamo.
# Once Sparse layouts are supported in FakeTensor, it will be possible
# to perform more calculated optimizations.


def scorch_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    def f(...): ...
    f_opt = torch.compile(f, backend=scorch_backend)
    """
    print("\n(before).graph:")
    print(gm.graph)
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == operator.mul:
            node.target = operator.add
    print("\n(after).graph:")
    gm.graph.lint()
    gm.recompile()
    print(gm.graph)

    print("\n.code:")
    print(gm.code)

    return gm.forward


@torch.compile(backend=scorch_backend)
def fn(x: tensor.Tensor, y: tensor.Tensor, z):
    return x + y + z


# FakeTensor support for sparsity not completed yet.
a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
b = torch.Tensor([[-1, -1, -1], [-1, -1, -1]])

# a = tensor.Tensor.from_torch(a, name="A1")
# b = tensor.Tensor.from_torch(b, name="B1")
print(fn(a, b, 2))
