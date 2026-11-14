"""``scorch.sparse``: ``torch.sparse`` with the calls Scorch answers itself.

``import scorch as torch`` makes ``torch.sparse`` this module.  A name it does not
define is ``torch.sparse``'s own object, read through the module ``__getattr__``, so
``scorch.sparse.mm is torch.sparse.mm`` and a call of it is torch's call.  It defines
:func:`spsolve`.
"""

from typing import Any, List

import torch

from .exceptions import SparseSolveUnsupportedError, TensorTypeError
from .stensor import STensor

# torch.sparse.spsolve's operator in torch's dispatcher.
_SPSOLVE_OPERATOR = "aten::_spsolve"

# The dispatch keys below BackendSelect are the backend keys (CPU, SparseCPU,
# SparseCsrCPU, SparseCsrCUDA, ...).  The highest of them in a call's key set is the key
# whose kernel the dispatcher runs once the keys above it have passed the call down.
# Those are autograd's, ADInplaceOrView, and the keys between ADInplaceOrView and
# BackendSelect (ZeroTensor, Negative, Conjugate, Functionalize and others), whose
# fallbacks redispatch, so a conjugated operand still runs its backend's kernel.  The
# Python key, also between them, hands the call to Python instead (spsolve checks it
# first).
_BACKEND_KEYS = torch._C._dispatch_keyset_full_after(torch._C.DispatchKey.BackendSelect)
_PYTHON_KEY = torch._C.DispatchKey.Python

# The names ``from scorch.sparse import *`` binds: torch.sparse's ``__all__`` (less
# ``solve``, which torch lists and does not define) and spsolve.
__all__ = sorted(
    {name for name in torch.sparse.__all__ if hasattr(torch.sparse, name)} | {"spsolve"}
)


def __getattr__(name: str) -> Any:
    """``torch.sparse``'s own object for a name this module does not define."""

    return getattr(torch.sparse, name)


def __dir__() -> List[str]:
    """This module's names and ``torch.sparse``'s."""

    return sorted(set(globals()) | set(dir(torch.sparse)))


def _dispatch_key_set(
    input: torch.Tensor, other: torch.Tensor
) -> torch._C.DispatchKeySet:
    """The key set the dispatcher reads for a call on ``input`` and ``other``: the
    operands' keys and the thread's included keys, less the thread's excluded keys."""

    keys = torch._C._dispatch_keys(input) | torch._C._dispatch_keys(other)
    keys = keys | torch._C._dispatch_tls_local_include_set()
    return keys - torch._C._dispatch_tls_local_exclude_set()


def spsolve(input: Any, other: Any, *, left: bool = True) -> torch.Tensor:
    """``torch.sparse.spsolve`` where torch can take the call, else a refusal.

    Scorch has no sparse solver.  The call is
    ``torch.sparse.spsolve(input, other, left=left)`` unchanged, its result and its
    errors torch's, in three cases, checked in this order:

    * torch's override protocol holds the call: an operand defines
      ``__torch_function__`` or a ``TorchFunctionMode`` is active
      (``torch.overrides.has_torch_function``);
    * both operands are torch tensors and the key set the dispatcher reads holds the
      Python key: an operand defines ``__torch_dispatch__`` or a ``TorchDispatchMode``
      is active;
    * both operands are torch tensors and torch's dispatcher holds an ``aten::_spsolve``
      kernel for their backend key (a CSR matrix on a CUDA build with cuDSS).

    Otherwise it raises :class:`~scorch.exceptions.SparseSolveUnsupportedError`
    (``spsolve_unsupported``): for an ``STensor`` operand whatever its format, and for
    torch tensors whose backend has no kernel, which on CPU is every layout.  An operand
    that is neither a torch tensor nor an ``STensor`` raises
    :class:`~scorch.exceptions.TensorTypeError` (``spsolve_operand_type_invalid``).

    The dispatcher is asked, not the device type: a build's kernels decide what torch
    can solve, and where it cannot, ``torch.sparse.spsolve`` raises an uncoded error.
    """

    if torch.overrides.has_torch_function((input, other)):
        return torch.sparse.spsolve(input, other, left=left)
    operands = (("input", input), ("other", other))
    for name, operand in operands:
        if isinstance(operand, STensor):
            raise SparseSolveUnsupportedError(
                f"scorch.sparse.spsolve: {name} is an STensor, and Scorch has no "
                "sparse solver"
            )
    for name, operand in operands:
        if not isinstance(operand, torch.Tensor):
            raise TensorTypeError(
                f"scorch.sparse.spsolve: {name} must be a torch.Tensor or an "
                f"STensor, not {type(operand).__name__}",
                code="spsolve_operand_type_invalid",
            )
    keys = _dispatch_key_set(input, other)
    if keys.has(_PYTHON_KEY):
        return torch.sparse.spsolve(input, other, left=left)
    key = (keys & _BACKEND_KEYS).highestPriorityTypeId()
    if not torch._C._dispatch_has_kernel_for_dispatch_key(_SPSOLVE_OPERATOR, key):
        raise SparseSolveUnsupportedError(
            f"scorch.sparse.spsolve: torch has no spsolve kernel for the {key.name} "
            f"backend (input layout {input.layout}, device {input.device}), and "
            "Scorch has no sparse solver"
        )
    return torch.sparse.spsolve(input, other, left=left)
