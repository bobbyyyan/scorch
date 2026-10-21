"""``torch.matmul``'s rank rules, written as the einsum expression scorch computes.

``scorch.matmul`` computes every product it does not hand to a prebuilt kernel as an
``einsum`` (``ops._generated_matmul``), and ``autograd.py`` records the derivative of such a
product as that einsum's (``einsum_grad``).  Both read the expression here, so the product
and its derivative cannot disagree about what is computed.

For operands of ranks ``p`` and ``q``, as ``torch.matmul`` defines the product:

- ``p = q = 1``: the dot product, ``"j,j->"``;
- ``p = 2, q = 1``: ``"ij,j->i"``;  ``p = 1, q = 2``: ``"j,jk->k"``;  ``p = q = 2``:
  ``"ij,jk->ik"``;
- ``p >= 3, q = 1``: the vector is contracted with the last mode, every other mode kept:
  ``"...ij,j->...i"``;
- ``p = 1, q >= 3``: ``torch.matmul`` prepends a mode of extent one to the vector, takes
  the batched product and removes that mode again, which is the vector contracted with
  the second-to-last mode: ``"j,...jk->...k"``;
- ``p, q >= 2`` with one of them at least 3: the batched product.  The batch modes are
  every mode but the last two, aligned at the right, and the result's are their
  broadcast.  Per aligned position:

  - present in one operand only: that operand's label, in the result;
  - present in both with equal extents: one label, shared by both and the result;
  - present in both, one extent 1 and the other ``n``: the operand of extent ``n``
    holds the result's label, and the operand of extent 1 a label of its own, in no
    other operand and not in the result, so the product sums over that mode's single
    element.  That is the broadcast exactly (a sum of one term), with no copy, view or
    densifying of the operand, and einsum's derivative of a label private to one
    operand is the broadcast's: the gradient summed over the mode it was broadcast
    along;
  - present in both with different extents, neither 1: one shared label, which
    ``einsum`` refuses by code (``einsum_index_extent_mismatch``), the refusal it gives
    every contracted pair of different extents.

The matrix modes keep the rank-2 labels ``i``, ``j`` and ``k``; batch modes take labels from
``_BATCH_LABELS``, which leaves those three out, in order from the left.  A product with
more batch modes than there are labels for, or with a rank-0 operand (which
``torch.matmul`` refuses), has no expression.
"""

from __future__ import annotations

import string
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from .stensor import STensor

#: The labels of batch modes: every ASCII letter einsum accepts except the matrix labels.
_BATCH_LABELS = "".join(c for c in string.ascii_letters if c not in "ijk")

_MATRIX_EXPRESSIONS = {
    (2, 2): "ij,jk->ik",
    (1, 2): "j,jk->k",
    (2, 1): "ij,j->i",
    (1, 1): "j,j->",
}


def matmul_expression(a_shape: Sequence[int], b_shape: Sequence[int]) -> Optional[str]:
    """The einsum expression of ``torch.matmul(a, b)`` for operands of these shapes, or
    None for a pair ``matmul`` refuses: a rank-0 operand, or more batch modes than
    ``_BATCH_LABELS`` holds labels for."""

    p, q = len(a_shape), len(b_shape)
    expression = _MATRIX_EXPRESSIONS.get((p, q))
    if expression is not None or p == 0 or q == 0:
        return expression
    labels = iter(_BATCH_LABELS)
    try:
        a_batch, b_batch, out_batch = _batch_labels(
            a_shape[:-2] if p >= 2 else (), b_shape[:-2] if q >= 2 else (), labels
        )
    except StopIteration:
        return None
    a_matrix = "ij" if p >= 2 else "j"
    b_matrix = "jk" if q >= 2 else "j"
    out_matrix = ("i" if p >= 2 else "") + ("k" if q >= 2 else "")
    return f"{a_batch}{a_matrix},{b_batch}{b_matrix}->{out_batch}{out_matrix}"


def operands_expression(a: Any, b: Any) -> Optional[str]:
    """``matmul_expression`` for the operands ``a`` and ``b`` themselves (STensors or
    torch tensors).  A pair of matrices and vectors, every product a rank-two route
    computes, has its expression by the two ranks alone, so it reads only those, as the
    rank table it replaces did, and builds no shape; any other pair reads the shapes."""

    expression = _MATRIX_EXPRESSIONS.get((a.dim(), b.dim()))
    if expression is not None:
        return expression
    return matmul_expression(operand_shape(a), operand_shape(b))


def operand_shape(x: Union[torch.Tensor, STensor]) -> Tuple[int, ...]:
    """An operand's extents in logical mode order, which the rules read: an STensor's
    ``logical_shape`` (its ``shape`` is the physical one, in storage order), a torch
    tensor's ``shape``.  Every caller has checked that ``x`` is one of the two."""

    if isinstance(x, STensor):
        return tuple(x.logical_shape)
    return tuple(x.shape)


def matmul_result_rank(a_shape: Sequence[int], b_shape: Sequence[int]) -> Optional[int]:
    """The rank of ``torch.matmul(a, b)``'s result, or None where ``matmul_expression``
    has no expression."""

    expression = matmul_expression(a_shape, b_shape)
    return None if expression is None else len(expression.split("->")[1])


def _batch_labels(
    a_batch: Sequence[int], b_batch: Sequence[int], labels: Iterator[str]
) -> tuple[str, str, str]:
    """The labels of each operand's batch modes and of the result's, by the rules in this
    module's docstring.  Raises StopIteration when ``labels`` runs out."""

    n = max(len(a_batch), len(b_batch))
    a_out: List[str] = []
    b_out: List[str] = []
    result: List[str] = []
    for position in range(n):
        ia = position - (n - len(a_batch))
        ib = position - (n - len(b_batch))
        ea = a_batch[ia] if ia >= 0 else None
        eb = b_batch[ib] if ib >= 0 else None
        label = next(labels)
        result.append(label)
        if ea is None:
            b_out.append(label)
        elif eb is None:
            a_out.append(label)
        elif ea == eb or (ea != 1 and eb != 1):
            a_out.append(label)
            b_out.append(label)
        elif ea == 1:
            a_out.append(next(labels))
            b_out.append(label)
        else:
            a_out.append(label)
            b_out.append(next(labels))
    return "".join(a_out), "".join(b_out), "".join(result)
