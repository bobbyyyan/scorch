"""The derivative of scorch's compiler einsum (drop-in note, section 0.2, item d2).

``scorch.einsum`` computes a program with a sparse operand, or a sparse result, with the
compiler, and its result is an STensor.  The program is multilinear in its operands'
values.  For ``C = einsum("s_1,...,s_n->o", X_1, ..., X_n)`` and ``G = dL/dC``::

    dL/dX_k = einsum("o,s_j (j != k)->s_k", G, X_j ...)

When ``X_k`` is an STensor the gradient is wanted on its stored coordinates only: a
sampled einsum, computed as ``einsum("s_k,o,s_j (j != k)->s_k", P_k, G, X_j ...)`` with
``P_k`` an STensor over ``X_k``'s index arrays whose values are ones, so the compiler
iterates the intersection with ``X_k``'s structure; its result's values are matched to
``X_k``'s storage order by coordinate (``stensor_grad._positions``).  When ``X_k`` is
dense the gradient is wanted everywhere, and broadcast along a label no other term names.

When ``X_k`` is a torch sparse tensor (COO, CSR or CSC) its gradient is torch's: the
unmasked one, everywhere, as torch gives a sparse operand of torch.matmul (Bobby,
2026-10-15, "match torch"; torch.einsum itself raises on a sparse operand, and where torch
raises scorch gives the unmasked gradient). It is dense and strided when every other
operand is dense, which is torch's layout for a sparse times a dense product, and is
otherwise in ``X_k``'s own layout, computed as a sparse program: torch's layout for a
product of two sparse operands of one layout, and the operand's where torch raises. A CSC
operand's gradient is dense and strided in every product: torch's autograd engine accepts no
CSC-layout gradient for a CSC leaf (torch 2.13 and 2.14: AccumulateGrad raises "Sparse CSC
tensors do not have is_contiguous", and torch's own sparse product with a CSC operand raises
in its backward), and dense is torch's layout for a CSC operand of a sparse times a dense
product. The operand itself is the recorded input, so the gradient reaches it in that layout;
``torch.sparse.mm``, which scorch does not define, keeps torch's stored-entries gradient.

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
forward read), and each operand's values tensor -- a torch sparse operand itself, whose
values() share its version counter -- saved for backward, so torch refuses a backward
after the values were written.  A write to an operand's index arrays in between is
refused with ``stensor_structure_written_before_backward``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.autograd.forward_ad as _fwad

import scorch_ops as _native  # type: ignore[import-not-found]

from .compiler.scheduler import call_without_forced_schedule
from .exceptions import (
    EinsumDerivativeError,
    ScorchAssertionError,
    ScorchError,
    ScorchNotImplementedError,
)
from .format import LevelType
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


# --------------------------------------------------------------------------- #
# A torch sparse operand's gradient, in its layout
# --------------------------------------------------------------------------- #


def _sparse_keywords(layout: torch.layout, term: str) -> Dict[str, Any]:
    """The keywords under which the compiler stores a gradient of ``term``'s rank the way
    ``layout`` stores it: a matrix as CSR (``"ds"`` in mode order ``[0, 1]``) or, for COO,
    CSR too -- its stored coordinates are COO's, row-major and each once -- and any other
    rank as coordinates in their own order. A CSC operand never comes here: its gradient is
    dense (see the module's note).

    The mode order is always given: without one, einsum stores the result in the order
    its schedule finds (``temp_mode_order``), which follows the factors' -- so the
    gradient of a CSR operand from a product whose result is CSC (``output_mode_order=
    [1, 0]``, its gradient ``G`` stored columns outermost) came back in mode order
    ``[1, 0]``, which ``_to_torch_layout`` refuses (chk21v3 85360)."""

    if len(term) == 2:
        return {"format": "ds", "output_mode_order": [0, 1]}
    return {"format": "o" * len(term), "output_mode_order": list(range(len(term)))}


def _to_torch_layout(gradient: STensor, layout: torch.layout) -> torch.Tensor:
    """A gradient the compiler stored (:func:`_sparse_keywords`) as a torch tensor of
    ``layout`` over the same index arrays and values -- the tensor autograd tracks for
    them when the gradient was recorded, so a torch constructor's own backward carries it
    on (it gives a values gradient)."""

    values = gradient.__dict__.get(AUTOGRAD_VALUES, gradient._storage._value)
    if values.dtype != gradient.dtype:
        values = values.to(gradient.dtype)
    shape = tuple(int(n) for n in gradient.logical_shape)
    levels = gradient.format.level_types
    order = list(gradient.storage.index.mode_order or range(gradient.dim()))
    indices = gradient._native_mode_indices()
    if levels == (LevelType.DENSE, LevelType.COMPRESSED):
        positions, coordinates = indices[1]
        if order == [0, 1]:
            if layout is torch.sparse_csr:
                return torch.sparse_csr_tensor(positions, coordinates, values, shape)
            if layout is torch.sparse_coo:
                # CSR's coordinates in its order are COO's, coalesced.
                pairs = torch._convert_indices_from_csr_to_coo(
                    positions.long(), coordinates.long()
                )
                return torch.sparse_coo_tensor(pairs, values, shape, is_coalesced=True)
    elif layout is torch.sparse_coo and order == list(range(len(levels))):
        if all(
            level in (LevelType.COORDINATE, LevelType.SINGLETON) for level in levels
        ):
            per_mode = [level[0] for level in indices]
            coalesced = not _native.abi_screen_lex_levels(per_mode, values.numel())
            built = torch.sparse_coo_tensor(
                torch.stack([c.long() for c in per_mode]),
                values,
                shape,
                is_coalesced=coalesced,
            )
            return built if coalesced else built.coalesce()
    raise ScorchAssertionError(
        f"a {layout} gradient stored as {gradient.format} in mode order {order}",
        code="einsum_grad_torch_layout_unexpected",
    )


def _dense_to_layout(dense: torch.Tensor, layout: torch.layout) -> torch.Tensor:
    """A dense gradient in a torch sparse ``layout``, CSR or COO, by torch's own conversion.
    A CSC operand never comes here: its gradient is dense (see the module's note)."""

    if layout is torch.sparse_csr:
        return dense.to_sparse_csr()
    if layout is torch.sparse_coo:
        return dense.to_sparse_coo()
    raise ScorchAssertionError(
        f"a {layout} gradient laid out from a dense one",
        code="einsum_grad_torch_layout_unexpected",
    )


class _Program:
    """One einsum the forward computed, as its derivative reads it.

    ``terms`` and ``output`` are the expression's subscripts.  Per operand: whether it
    is an STensor (else a dense torch tensor), its storage as the forward read it, an
    STensor over that storage without history (``snapshots``), and its structure with
    version stamps; for a torch sparse operand, which the derivative reads through the
    STensor the compiler converted it to, its torch layout (``torch_layouts``, else None).
    The result's layout and structure, and the extent of every label.
    """

    __slots__ = (
        "expression",
        "terms",
        "output",
        "stensor",
        "torch_layouts",
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
        torch_layouts: Optional[Sequence[Optional[torch.layout]]] = None,
    ) -> None:
        inputs, output = expression.split("->")
        self.expression = expression
        self.terms = inputs.split(",")
        self.output = output
        self.stensor = [type(x) is STensor for x in operands]
        self.torch_layouts: List[Optional[torch.layout]] = list(
            torch_layouts or [None] * len(operands)
        )
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
        ``create_graph`` is recorded as a function of it). A torch sparse operand is the
        saved tensor itself while a backward builds a graph through it, so that graph
        reaches it with torch's gradient again, and otherwise the STensor the forward
        computed with."""

        for structure in self.structures + [self.result_structure]:
            if structure is not None:
                structure.check()
        rebuilt: List[Any] = []
        for k, values in enumerate(saved):
            snapshot = self.snapshots[k]
            if snapshot is None:
                rebuilt.append(values)
            elif self.torch_layouts[k] is not None:
                graph = values.requires_grad and torch.is_grad_enabled()
                rebuilt.append(values if graph else snapshot)
            elif values.requires_grad:
                rebuilt.append(snapshot._over_structure(values))
            else:
                rebuilt.append(snapshot)
        return rebuilt

    def _with(self, k: int, operands: List[Any], values: torch.Tensor) -> List[Any]:
        """``operands`` with operand ``k`` replaced by ``values`` in its layout: a torch
        sparse operand's tangent is a torch sparse tensor of its own structure, used as
        it is."""

        replaced = list(operands)
        snapshot = self.snapshots[k]
        if snapshot is None or self.torch_layouts[k] is not None:
            replaced[k] = values
        else:
            replaced[k] = snapshot._over_structure(values)
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
        """``dL/dX_k``: for a dense torch operand, a tensor of its shape; for a torch
        sparse operand, torch's gradient (:meth:`_torch_gradient`); for an STensor, a
        gradient of its stored values in storage order."""

        term = self.terms[k]
        others = [(self.terms[j], operands[j]) for j in range(len(operands)) if j != k]
        layout = self.torch_layouts[k]
        if layout is not None:
            return self._torch_gradient(k, term, layout, grad_output, others)
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
        logical: bool = False,
    ) -> torch.Tensor:
        """The gradient everywhere, as a dense tensor: in operand ``k``'s logical layout
        for a torch operand (or when ``logical``), else a dense STensor's values in its
        storage order."""

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
            return self._laid_out(k, term, named, computed, logical)
        try:
            if all(_is_dense(x) for x in operands) or not target:
                # torch.einsum's, or a rank-0 result, which is dense in any format.
                computed = _einsum(expression, *operands)
            else:
                computed = _einsum(expression, *operands, format="d" * len(target))
        except ScorchError as error:
            raise self._refused(k, error) from error
        return self._laid_out(k, term, named, _as_torch(computed), logical)

    def _torch_gradient(
        self,
        k: int,
        term: str,
        layout: torch.layout,
        grad_output: Any,
        others: List[Tuple[str, Any]],
    ) -> torch.Tensor:
        """``dL/dX_k`` for a torch sparse ``X_k`` of ``layout``: unmasked, as torch gives a
        sparse operand its gradient (see the module's note).

        Dense and strided when every other operand is dense, and for a CSC operand in every
        product (see the module's note). Otherwise in ``layout``,
        computed by the compiler as a sparse program over the factors -- ``G`` and the
        other operands -- so it stores the coordinates their structures imply, as torch's
        sparse product does, and never materializes the dense gradient. A gradient that is
        dense along a label only ``X_k`` names (a sum over it in the forward) is computed
        dense and then laid out, since it is nonzero at every coordinate along that label.
        """

        if layout is torch.sparse_csc or all(_is_dense(x) for _, x in others):
            return self._dense_gradient(k, term, grad_output, others, logical=True)
        named = set(self.output)
        for other, _ in others:
            named.update(other)
        if any(label not in named for label in term) or len(set(term)) != len(term):
            dense = self._dense_gradient(k, term, grad_output, others, logical=True)
            return _dense_to_layout(dense, layout)
        factors = [(self.output, grad_output)] + others
        expression = ",".join(t for t, _ in factors) + "->" + term
        try:
            computed = _einsum(
                expression, *(x for _, x in factors), **_sparse_keywords(layout, term)
            )
        except ScorchError as error:
            raise self._refused(k, error) from error
        if type(computed) is not STensor:
            raise self._refused(
                k,
                TypeError(f"the gradient's program returned {type(computed).__name__}"),
            )
        return _to_torch_layout(computed, layout)

    def _laid_out(
        self,
        k: int,
        term: str,
        named: set,
        computed: torch.Tensor,
        logical: bool = False,
    ) -> torch.Tensor:
        """``computed`` (over the labels of ``term`` some other term names, in
        ``term``'s order) broadcast to operand ``k``'s shape, as its gradient: logically
        laid out for a torch operand or when ``logical``, else in a dense STensor's
        storage order."""

        full = computed.reshape(
            [self.extents[label] if label in named else 1 for label in term]
        ).expand([self.extents[label] for label in term])
        snapshot = self.snapshots[k]
        if snapshot is None or logical:
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


def _torch_sparse_layout(operand: Any) -> Optional[torch.layout]:
    if isinstance(operand, torch.Tensor) and operand.layout is not torch.strided:
        return cast(torch.layout, operand.layout)
    return None


def _normalized(operand: Any) -> Any:
    """An operand as the derivative reads it: a torch sparse tensor as the STensor the
    compiler converts it to, without history -- the tensor itself is the recorded input,
    so that its gradient is torch's (see the module's note); anything else as it is."""

    if _torch_sparse_layout(operand) is not None:
        return STensor.from_torch(operand.detach())
    return operand


def _over_torch_result(result: torch.Tensor) -> Optional[STensor]:
    """The STensor over a torch sparse matrix the forward returned, sharing its index
    arrays and its values in their order; None for any other torch sparse result.

    ``scorch.matmul`` returns the product of two torch compressed operands in torch's type
    (ops._torch_csr_spgemm, ops._as_torch_compressed): a CSR tensor, or a CSC one when
    ``output_mode_order=[1, 0]`` stores its columns outermost. Each row's (column's)
    coordinates are sorted and distinct, so the STensor adopts the arrays as they are. A
    result the construction puts in canonical order instead has its values reordered,
    and is None too: the tensor returned for it (``_torch_result_over``) holds the
    recorded values in the STensor's order over the result's own arrays."""

    if result.dim() != 2:
        return None
    if result.layout is torch.sparse_csr:
        stored = STensor.from_torch(result)
    elif result.layout is torch.sparse_csc:
        # _from_parts takes the stored shape: the columns outermost.
        stored = STensor._from_parts(
            None,
            (result.shape[1], result.shape[0]),
            "ds",
            [[], [result.ccol_indices(), result.row_indices()]],
            result.values(),
            mode_order=[1, 0],
            adopt=True,
        )
    else:
        return None
    values = stored._storage._value
    given = result.values()
    if values.numel() != given.numel() or (
        values.numel() > 0 and values.data_ptr() != given.data_ptr()
    ):
        return None
    return stored


def _torch_result_over(result: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """A torch sparse matrix of ``result``'s layout over its index arrays whose values
    are ``values``, the tensor autograd tracks for them: torch's constructor is
    differentiable in its values, so the gradient reaches them through its own backward.
    The arrays, their index type and the values' bits are the forward's, so the product
    is the same tensor in every grad mode."""

    shape = tuple(result.shape)
    if result.layout is torch.sparse_csr:
        return torch.sparse_csr_tensor(
            result.crow_indices(), result.col_indices(), values, shape
        )
    return torch.sparse_csc_tensor(
        result.ccol_indices(), result.row_indices(), values, shape
    )


def recorded(
    expression: Any,
    tensors: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    compute: Callable[..., Any],
    refuse: Callable[[Any, str], Any],
) -> Union[torch.Tensor, STensor]:
    """``compute(expression, *tensors, **kwargs)`` -- the compiler's einsum, for a call
    through which a gradient is wanted -- computed without autograd on detached
    operands, and its result's values recorded as ``_EinsumValues``. A torch sparse
    result (``scorch.matmul``'s for two torch compressed operands) is returned as a torch
    tensor of its layout over the same arrays, whose values carry the record.

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
    # A torch sparse result (scorch.matmul's for two torch compressed operands) is read through
    # the STensor over its arrays, and returned as a torch tensor of its layout.
    torch_result: Optional[torch.Tensor] = None
    if type(result) is torch.Tensor and result.layout is not torch.strided:
        over = _over_torch_result(result) if positional else None
        if over is None:
            return refuse(result, "its torch sparse result is not one it reads")
        torch_result, result = result, over
    dense_result = type(result) is torch.Tensor and result.layout is torch.strided
    if not positional or not (type(result) is STensor or dense_result):
        return refuse(result, "the call is not in the form the derivative reads")
    if any(key in kwargs for key in _NOT_MULTILINEAR):
        return refuse(
            result if torch_result is None else torch_result,
            "fused post-operations make it more than a product",
        )
    layouts = [_torch_sparse_layout(x) for x in tensors]
    program = _Program(expression, operands, result, kwargs, layouts)
    # Each operand's recorded input: a torch sparse operand itself, an STensor's values
    # (the tensor autograd tracks for them, when it has one), a dense tensor itself.
    inputs = [
        (
            given
            if layout is not None
            else (
                x.__dict__.get(AUTOGRAD_VALUES, x._storage._value)
                if type(x) is STensor
                else x
            )
        )
        for given, x, layout in zip(tensors, operands, layouts)
    ]
    if dense_result:
        return cast(torch.Tensor, _EinsumValues.apply(result, program, *inputs))
    tracked = _EinsumValues.apply(result._storage._value, program, *inputs)
    if torch_result is not None:
        if _tangent_of(tracked):
            raise ScorchNotImplementedError(
                "scorch.matmul returns the product of two torch sparse operands as a "
                "torch sparse tensor, and carries no forward-mode tangent through it: "
                'ask for a dense result (format="dd")',
                code="torch_sparse_forward_tangent_unsupported",
            )
        if tracked.requires_grad:
            return _torch_result_over(torch_result, tracked)
        return torch_result
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
