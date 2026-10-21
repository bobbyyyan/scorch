"""The derivative of scorch's compiler einsum (drop-in note, section 0.2, item d2).

``scorch.einsum`` computes a program with a sparse operand, or a sparse result, with the
compiler, and its result is an STensor.  The program is multilinear in its operands'
values.  For ``C = einsum("s_1,...,s_n->o", X_1, ..., X_n)`` and ``G = dL/dC``::

    dL/dX_k = einsum("o,s_j (j != k)->s_k", G, X_j ...)

When ``X_k`` is sparse the gradient is wanted on its stored coordinates only: a sampled
einsum, computed as ``einsum("s_k,o,s_j (j != k)->s_k", P_k, G, X_j ...)`` with ``P_k``
an STensor over ``X_k``'s index arrays whose values are ones, so the compiler iterates
the intersection with ``X_k``'s structure; its result's values are matched to ``X_k``'s
storage order by coordinate (``stensor_grad._positions``).  When ``X_k`` is dense the
gradient is wanted everywhere, and broadcast along a label no other term names.

``G`` is an STensor over ``C``'s index arrays (a torch tensor when ``C`` is dense): the
compiler stores every coordinate the operands' structures imply, whatever their values,
so ``C`` is zero elsewhere for every value of the operands and has no gradient there.

Each gradient is computed by ``scorch.einsum`` itself -- by ``torch.einsum`` when every
factor is dense, by the compiler otherwise -- which records it when the backward builds
a graph (``create_graph=True``), so the derivative is closed under differentiation.  A
forward-mode tangent is the sum of the program with one operand replaced by its
tangent, matched into ``C``'s storage order the same way.

What the derivative reads is held by reference: each operand's storage (an in-place
conversion of the operand replaces its storage, so the program keeps the one the
forward read), and each operand's values tensor, saved for backward, so torch refuses a
backward after the values were written.  A write to an operand's index arrays in between
is refused with ``stensor_structure_written_before_backward``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.autograd.forward_ad as _fwad

from .compiler.scheduler import call_without_forced_schedule
from .exceptions import (
    EinsumDerivativeError,
    ScorchAssertionError,
    ScorchError,
    ScorchNotImplementedError,
)
from .stensor import STensor
from .stensor_grad import AUTOGRAD_VALUES, _gather, _positions, _Structure, attach

# Keyword arguments that change what the program computes, not how: a call carrying one
# is not the multilinear program above, and its derivative is refused.
_NOT_MULTILINEAR = ("_post_ops", "_post_ops_tensors")
# The keyword arguments the forward's result layout depends on, which a forward-mode
# tangent is computed with so it lands in the same layout.
_RESULT_KEYWORDS = ("format", "output_mode_order")


def _einsum(expression: str, *operands: Any, **kwargs: Any) -> Any:
    """``scorch.einsum`` under the default schedule: bound on use, since ``ops`` imports
    this module's callers.  Every product this module computes is a derivative's, a
    gradient or a forward-mode tangent, and so another expression than the forward a
    forced schedule names: it runs as autograd.py's own products do, with no schedule
    forced, even when ``backward()`` is called inside ``schedule_force``."""

    from . import ops

    return call_without_forced_schedule(ops.einsum, expression, *operands, **kwargs)


def _as_torch(operand: Any) -> torch.Tensor:
    """A dense operand as a torch tensor in its logical layout, with its history."""

    if type(operand) is STensor:
        return cast(torch.Tensor, operand.to_torch(in_place=False))
    return cast(torch.Tensor, operand)


def _is_dense(operand: Any) -> bool:
    if isinstance(operand, torch.Tensor):
        return operand.layout is torch.strided
    return type(operand) is STensor and operand.format.is_dense()


def _tangent_of(tensor: torch.Tensor) -> bool:
    if _fwad._current_level < 0:
        return False
    try:
        return _fwad.unpack_dual(tensor).tangent is not None
    except RuntimeError:
        return False


class _Program:
    """One einsum the forward computed, as its derivative reads it.

    ``terms`` and ``output`` are the expression's subscripts.  Per operand: whether it
    is an STensor (else a dense torch tensor), its storage as the forward read it, an
    STensor over that storage without history (``snapshots``), and its structure with
    version stamps.  The result's layout and structure, and the extent of every label.
    """

    __slots__ = (
        "expression",
        "terms",
        "output",
        "stensor",
        "snapshots",
        "structures",
        "extents",
        "result",
        "result_structure",
        "result_dense",
        "result_keywords",
    )

    def __init__(
        self,
        expression: str,
        operands: Sequence[Any],
        result: Union[STensor, torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> None:
        inputs, output = expression.split("->")
        self.expression = expression
        self.terms = inputs.split(",")
        self.output = output
        self.stensor = [type(x) is STensor for x in operands]
        self.snapshots: List[Optional[STensor]] = [
            x.detach() if type(x) is STensor else None for x in operands
        ]
        self.structures: List[Optional[_Structure]] = [
            _Structure.of(x) if type(x) is STensor else None for x in operands
        ]
        self.extents: Dict[str, int] = {}
        for term, x in zip(self.terms, operands):
            shape = x.logical_shape if type(x) is STensor else tuple(x.shape)
            for label, extent in zip(term, shape):
                self.extents[label] = int(extent)
        # A torch result (``matmul``'s dense product) is its own logical layout and has
        # no structure; an STensor result is held as the forward stored it.
        stored = type(result) is STensor
        self.result: Optional[STensor] = (
            cast(STensor, result).detach() if stored else None
        )
        self.result_structure: Optional[_Structure] = (
            _Structure.of(result) if stored else None
        )
        self.result_dense = not stored or cast(STensor, result).format.is_dense()
        self.result_keywords = {
            key: kwargs[key] for key in _RESULT_KEYWORDS if key in kwargs
        }

    # -- operands as the derivative computes with them -----------------------------

    def operands(self, saved: Sequence[torch.Tensor]) -> List[Any]:
        """The operands over what the forward read, each STensor's values the saved
        tensor (with its history when it requires grad, so a gradient computed under
        ``create_graph`` is recorded as a function of it)."""

        for structure in self.structures + [self.result_structure]:
            if structure is not None:
                structure.check()
        rebuilt: List[Any] = []
        for k, values in enumerate(saved):
            snapshot = self.snapshots[k]
            if snapshot is None:
                rebuilt.append(values)
            elif values.requires_grad:
                rebuilt.append(snapshot._over_structure(values))
            else:
                rebuilt.append(snapshot)
        return rebuilt

    def _with(self, k: int, operands: List[Any], values: torch.Tensor) -> List[Any]:
        """``operands`` with operand ``k`` replaced by ``values`` in its layout."""

        replaced = list(operands)
        snapshot = self.snapshots[k]
        replaced[k] = values if snapshot is None else snapshot._over_structure(values)
        return replaced

    def output_gradient(self, g: torch.Tensor) -> Any:
        """``G``: the gradient of ``C``'s stored values as an operand, logically laid
        out -- a torch tensor for a dense ``C``, else an STensor over ``C``'s arrays."""

        if self.result is None:
            return g
        if not self.result_dense:
            return self.result._over_structure(g)
        layout = self.result._storage.layout
        permutation = layout.permutation
        inverse = [0] * len(permutation)
        for level, mode in enumerate(permutation):
            inverse[mode] = level
        return g.reshape(layout.physical_shape).permute(inverse)

    # -- the derivative --------------------------------------------------------------

    def _refused(self, k: int, error: Exception) -> EinsumDerivativeError:
        return EinsumDerivativeError(
            f"scorch.einsum computed {self.expression!r}, but the gradient of operand "
            f"{k} is a program the compiler refused ({type(error).__name__}: {error})"
        )

    def gradient(self, k: int, grad_output: Any, operands: List[Any]) -> torch.Tensor:
        """``dL/dX_k``: for a dense torch operand, a tensor of its shape; for an
        STensor, a gradient of its stored values in storage order."""

        term = self.terms[k]
        others = [(self.terms[j], operands[j]) for j in range(len(operands)) if j != k]
        if _is_dense(operands[k]):
            return self._dense_gradient(k, term, grad_output, others)
        snapshot = self.snapshots[k]
        structure = self.structures[k]
        if snapshot is None or structure is None:
            raise ScorchAssertionError(
                "snapshot is not None and structure is not None",
                code="einsum_grad_operand_without_structure",
            )
        pattern = snapshot._over_structure(
            torch.ones(
                snapshot._storage._value.numel(),
                dtype=snapshot._storage._value.dtype,
            )
        )
        factors = [(term, pattern), (self.output, grad_output)] + others
        expression = ",".join(t for t, _ in factors) + "->" + term
        try:
            sampled = _einsum(expression, *(x for _, x in factors))
        except ScorchError as error:
            raise self._refused(k, error) from error
        if type(sampled) is not STensor:
            # Only an all-dense call returns a torch tensor, and the pattern is sparse.
            raise self._refused(
                k, TypeError(f"the sampled einsum returned {type(sampled).__name__}")
            )
        values = sampled.__dict__.get(AUTOGRAD_VALUES, sampled._storage._value)
        found, at = _positions(_Structure.of(sampled), structure)
        return _gather(values, found, at)

    def _dense_gradient(
        self,
        k: int,
        term: str,
        grad_output: Any,
        others: List[Tuple[str, Any]],
    ) -> torch.Tensor:
        named = set(self.output)
        for other, _ in others:
            named.update(other)
        target = "".join(label for label in term if label in named)
        factors = [(self.output, grad_output)] + others
        expression = ",".join(t for t, _ in factors) + "->" + target
        operands = [x for _, x in factors]
        if not any(t for t, _ in factors):
            # Every factor is rank 0 (``"ij->"``'s only operand): their product, which
            # an einsum with no label at all ("->") does not compute.
            computed = _as_torch(operands[0])
            for x in operands[1:]:
                computed = computed * _as_torch(x)
            return self._laid_out(k, term, named, computed)
        try:
            if all(_is_dense(x) for x in operands) or not target:
                # torch.einsum's, or a rank-0 result, which is dense in any format.
                computed = _einsum(expression, *operands)
            else:
                computed = _einsum(expression, *operands, format="d" * len(target))
        except ScorchError as error:
            raise self._refused(k, error) from error
        return self._laid_out(k, term, named, _as_torch(computed))

    def _laid_out(
        self, k: int, term: str, named: set, computed: torch.Tensor
    ) -> torch.Tensor:
        """``computed`` (over the labels of ``term`` some other term names, in
        ``term``'s order) broadcast to operand ``k``'s shape, as its gradient."""

        full = computed.reshape(
            [self.extents[label] if label in named else 1 for label in term]
        ).expand([self.extents[label] for label in term])
        snapshot = self.snapshots[k]
        if snapshot is None:
            return full
        # A dense STensor's values, in storage order: physical level i holds logical
        # mode permutation[i].
        permutation = snapshot._storage.layout.permutation
        return full.permute(list(permutation)).reshape(-1)

    def tangent(self, operands: List[Any], tangents: Sequence[Any]) -> torch.Tensor:
        """The forward-mode tangent of ``C``'s stored values: the program with each
        operand that has a tangent replaced by it, summed, in ``C``'s storage order."""

        total: Optional[torch.Tensor] = None
        for k, tangent in enumerate(tangents):
            if tangent is None:
                continue
            replaced = self._with(k, operands, tangent)
            part = _einsum(self.expression, *replaced, **self.result_keywords)
            if self.result_structure is None:
                # A torch result: the tangent in its logical layout.
                part_values = (
                    part.to_torch(in_place=False) if type(part) is STensor else part
                )
            else:
                if type(part) is not STensor:
                    part = STensor.from_torch(part)
                values = part.__dict__.get(AUTOGRAD_VALUES, part._storage._value)
                # Each of C's stored values, from where the tangent program stores
                # its coordinate (zero where it stores none).
                found, at = _positions(_Structure.of(part), self.result_structure)
                part_values = _gather(values, found, at)
            total = part_values if total is None else total + part_values
        assert total is not None  # forward AD asks for this only with some tangent
        return total


class _EinsumValues(torch.autograd.Function):
    """``C``'s stored values as a function of every operand's values (``_Program``)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, values: torch.Tensor, program: _Program, *inputs: torch.Tensor
    ) -> torch.Tensor:
        ctx.program = program
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        return (
            values.detach()
        )  # not ``values`` itself: see the note above autograd._in_layout

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore[override]
        program: _Program = ctx.program
        operands = program.operands(ctx.saved_tensors)
        grad_output = program.output_gradient(g)
        grads: List[Optional[torch.Tensor]] = [None, None]
        for k, needed in enumerate(ctx.needs_input_grad[2:]):
            grads.append(program.gradient(k, grad_output, operands) if needed else None)
        return tuple(grads)

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any, _values: Any, _program: Any, *tangents: Any
    ) -> torch.Tensor:
        program: _Program = ctx.program
        return program.tangent(program.operands(ctx.saved_tensors), tangents)


def _normalized(operand: Any) -> Any:
    """An operand as the derivative reads it: a torch sparse tensor as the STensor the
    compiler converts it to, built here in grad mode so its values carry the torch
    tensor's ``values()`` history (``STensor.from_torch``); anything else as it is."""

    if isinstance(operand, torch.Tensor) and operand.layout is not torch.strided:
        return STensor.from_torch(operand)
    return operand


def recorded(
    expression: Any,
    tensors: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    compute: Callable[..., Any],
    refuse: Callable[[Any, str], Any],
) -> Union[torch.Tensor, STensor]:
    """``compute(expression, *tensors, **kwargs)`` -- the compiler's einsum, for a call
    through which a gradient is wanted -- computed without autograd on detached
    operands, and its result's values recorded as ``_EinsumValues``.

    ``refuse(result, why)`` records a result whose derivative this module does not
    give (a call in another form, or one carrying fused post-operations, which is not
    the multilinear program) with a node that raises when a gradient is asked of it.
    """

    positional = isinstance(expression, str) and not (
        len(tensors) == 1 and isinstance(tensors[0], (list, tuple))
    )
    operands = tensors
    if positional:
        try:
            operands = tuple(_normalized(x) for x in tensors)
        except ScorchError:
            # An operand the compiler cannot read either: the call below raises the
            # same refusal it raises without autograd.
            positional = False

    def detached(x: Any) -> Any:
        return x.detach() if isinstance(x, torch.Tensor) else x

    with torch.no_grad():
        result = compute(expression, *(detached(x) for x in operands), **kwargs)
    dense_result = type(result) is torch.Tensor and result.layout is torch.strided
    if not positional or not (type(result) is STensor or dense_result):
        return refuse(result, "the call is not in the form the derivative reads")
    if any(key in kwargs for key in _NOT_MULTILINEAR):
        return refuse(result, "fused post-operations make it more than a product")
    program = _Program(expression, operands, result, kwargs)
    inputs = [
        x.__dict__.get(AUTOGRAD_VALUES, x._storage._value) if type(x) is STensor else x
        for x in operands
    ]
    if dense_result:
        return cast(torch.Tensor, _EinsumValues.apply(result, program, *inputs))
    tracked = _EinsumValues.apply(result._storage._value, program, *inputs)
    if tracked.requires_grad:
        attach(result, tracked)
    elif _tangent_of(tracked):
        raise ScorchNotImplementedError(
            "scorch.einsum's result is an STensor, which carries a forward-mode "
            "tangent only with the history of its values: make an operand require "
            "grad as well, or ask for a dense torch result",
            code="stensor_forward_tangent_unsupported",
        )
    return result
