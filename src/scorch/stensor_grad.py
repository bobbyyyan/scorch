"""Autograd for the stored values of an STensor (drop-in note, section 0.2).

An STensor's values tensor, ``storage._value``, is detached, and stays detached: every
native kernel, plan, cache, validation stamp and autograd's own carrier tensor reads it,
and none of them ever sees autograd's history. An STensor whose values require grad
carries one more attribute, ``_autograd_values``: a tensor over the same memory as
``_value`` that autograd tracks. It is a leaf when the user set ``requires_grad`` (or
the STensor was built from a torch leaf), and the output of a recorded operation
otherwise. It exists exactly when the values require grad, and the metadata flag
``requires_grad`` is written in step with it, through one writer
(``STensor._write_requires_grad``), so that whatever reads the flag -- the plan
installer, the C entry's gradient test, ``autograd.wanted`` -- sees the same answer.

The two tensors share memory and a version counter: an element write through either is
seen by autograd's check on the tensors a backward saved.

No Function here returns a view: each forward returns ``.detach()`` of the values it
stored, so the values an STensor carries take an in-place write (``A.values -= lr *
A.grad.values`` under ``no_grad``, ``relu_`` in grad mode).  Why, and the same rule for
scorch's other Functions: the note above ``autograd._in_layout``.

Conversions move values without arithmetic: each output value is one input value, the
sum of input values stored at one coordinate (``coalesce``), or an implicit zero. Their
derivative is a gather, given by matching the stored coordinates of the input and the
output. ``_Structure`` keeps references to the index arrays a conversion read and wrote,
which exist already, with their version stamps; the coordinates are computed from them
in backward, so a conversion in grad mode holds no O(nnz x rank) table until backward
runs, and a backward over arrays a caller has written since is refused with a code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import torch

from .exceptions import ScorchAssertionError, ScorchRuntimeError, ScorchTypeError
from .layout import TensorLayout
from .storage import stored_coordinates, tensor_version

AUTOGRAD_VALUES = "_autograd_values"
# A user's leaf after an in-place conversion of its STensor: ``(leaf, structure)``, the
# leaf every graph accumulates into and the structure its gradient is laid out in
# (decision 5, option (ii)).
AUTOGRAD_ROOT = "_autograd_root"


def history(tensor: Any) -> Optional[torch.Tensor]:
    """The tensor autograd tracks for ``tensor``'s values, or None."""

    return cast(Optional[torch.Tensor], tensor.__dict__.get(AUTOGRAD_VALUES))


def attach(tensor: Any, values: torch.Tensor) -> None:
    """Give ``tensor`` the history ``values`` (which requires grad), and the flag."""

    tensor.__dict__[AUTOGRAD_VALUES] = values
    if not tensor._metadata.requires_grad:
        tensor._write_requires_grad(True)


def drop(tensor: Any) -> None:
    """Take ``tensor``'s history and flag away."""

    state = tensor.__dict__
    state.pop(AUTOGRAD_VALUES, None)
    state.pop(AUTOGRAD_ROOT, None)
    if tensor._metadata.requires_grad:
        tensor._write_requires_grad(False)


def make_leaf(tensor: Any) -> torch.Tensor:
    """A leaf over ``tensor``'s values, attached; the same memory as ``_value``."""

    leaf = tensor._storage._value.detach().requires_grad_()
    tensor.__dict__.pop(AUTOGRAD_ROOT, None)
    attach(tensor, leaf)
    return leaf


def is_leaf(tensor: Any) -> bool:
    """Whether ``tensor`` is a leaf to the user: its own values are the leaf, or the
    leaf they were converted from in place (``AUTOGRAD_ROOT``)."""

    values = history(tensor)
    if values is None:
        return True
    return values.grad_fn is None or AUTOGRAD_ROOT in tensor.__dict__


def leaf_of(tensor: Any) -> Optional[Tuple[torch.Tensor, Optional["_Structure"]]]:
    """The leaf ``tensor``'s gradient accumulates into and the structure the leaf's
    gradient is laid out in (None when it is the tensor's own current one), or None
    when ``tensor`` is not a leaf."""

    root = tensor.__dict__.get(AUTOGRAD_ROOT)
    if root is not None:
        return cast(Tuple[torch.Tensor, Optional[_Structure]], root)
    values = history(tensor)
    if values is not None and values.grad_fn is None:
        return values, None
    return None


# -- structures and the coordinate match -----------------------------------------------


class _Structure:
    """One STensor's structure at one moment, by reference.

    The layout and the index arrays are the tensor's own (never copied); the version
    stamps say whether they were written afterwards. ``keys()`` is each stored value's
    row-major linear index in the logical shape, computed when asked.
    """

    __slots__ = ("layout", "mode_indices", "count", "stamps", "_keys")

    def __init__(
        self,
        layout: TensorLayout,
        mode_indices: Sequence[Sequence[torch.Tensor]],
        count: int,
    ) -> None:
        self.layout = layout
        self.mode_indices = tuple(tuple(arrays) for arrays in mode_indices)
        self.count = count
        self.stamps = tuple(
            (array.data_ptr(), tensor_version(array))
            for arrays in self.mode_indices
            for array in arrays
        )
        self._keys: Optional[torch.Tensor] = None

    @classmethod
    def of(cls, tensor: Any) -> "_Structure":
        storage = tensor._storage
        return cls(storage.layout, storage._mode_indices, storage._value.numel())

    def same_as(self, other: "_Structure") -> bool:
        """Whether both describe the same arrays in the same layout."""

        return (
            self.layout is other.layout
            and self.count == other.count
            and len(self.mode_indices) == len(other.mode_indices)
            and all(
                len(mine) == len(theirs) and all(a is b for a, b in zip(mine, theirs))
                for mine, theirs in zip(self.mode_indices, other.mode_indices)
            )
        )

    def check(self) -> None:
        """Refuse, with a code, index arrays written since this structure was taken."""

        now = tuple(
            (array.data_ptr(), tensor_version(array))
            for arrays in self.mode_indices
            for array in arrays
        )
        if now != self.stamps:
            raise ScorchRuntimeError(
                "an STensor's index arrays were written after an operation that "
                "autograd recorded read them, so its derivative is not computed",
                code="stensor_structure_written_before_backward",
            )

    def keys(self) -> torch.Tensor:
        if self._keys is not None:
            return self._keys
        self.check()
        coordinates = stored_coordinates(self.layout, self.mode_indices, self.count)
        logical_shape = self.layout.logical_shape
        permutation = self.layout.permutation
        by_logical_mode: List[Optional[torch.Tensor]] = [None] * len(logical_shape)
        for level, coordinate in enumerate(coordinates):
            by_logical_mode[permutation[level]] = coordinate
        key = torch.zeros(self.count, dtype=torch.int64)
        for mode, extent in enumerate(logical_shape):
            along = by_logical_mode[mode]
            if along is None:
                raise ScorchAssertionError(
                    "along is not None", code="stensor_grad_mode_without_level"
                )
            key = key * extent + along
        self._keys = key
        return key


def given_structure(
    tensor: Any, mode_indices: Sequence[Sequence[torch.Tensor]], count: int
) -> _Structure:
    """The structure of values handed to a constructor with the index arrays
    ``mode_indices``, as given (before the storage put them in canonical order), in
    the layout of the STensor ``tensor`` built from them."""

    return _Structure(tensor._storage.layout, mode_indices, count)


def _positions(
    target: Any, source: Any
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """For each stored value of ``source``, whether ``target`` stores its coordinate,
    and where; ``(None, None)`` when the two store the same coordinates in the same
    order. ``target``'s coordinates are distinct (every scorch storage's are)."""

    target_keys = target.keys()
    source_keys = source.keys()
    if target_keys.numel() == 0:
        empty = torch.zeros(source_keys.numel(), dtype=torch.int64)
        return torch.zeros(source_keys.numel(), dtype=torch.bool), empty
    if torch.equal(target_keys, source_keys):
        # The same coordinates in the same order, which is what a conversion that keeps
        # the order and a compiler result laid out like its pattern store: the identity
        # (None, None), found in O(n) and applied as no copy at all, not the sort below.
        return None, None
    ordered, order = torch.sort(target_keys)
    slot = torch.searchsorted(ordered, source_keys).clamp_(max=ordered.numel() - 1)
    found = ordered[slot] == source_keys
    return found, order[slot]


def _gather(
    grad: torch.Tensor, found: Optional[torch.Tensor], at: Optional[torch.Tensor]
) -> torch.Tensor:
    """``grad[at]`` where ``found``, zero elsewhere, recorded by torch; ``grad`` itself
    for the identity match (``None, None``)."""

    if found is None or at is None:
        return grad
    if grad.numel() == 0 and found.numel() > 0:
        # The target stores nothing and the source does (two operands' patterns that
        # share no coordinate, an empty operand or upstream gradient, a conversion
        # that keeps no entry): no ``at`` indexes ``grad``, so ``grad[at]`` would raise,
        # and every source value gathers zero.  ``+ grad.sum()`` (an exact zero) keeps
        # the result recorded through ``grad``, so a second derivative is zero, not
        # missing.
        return grad.new_zeros(found.shape) + grad.sum()
    return torch.where(found, grad[at], grad.new_zeros(()))


def _scatter(
    tangent: torch.Tensor,
    found: Optional[torch.Tensor],
    at: Optional[torch.Tensor],
    count: int,
) -> torch.Tensor:
    """The adjoint of ``_gather``: ``tangent`` summed into ``count`` slots at ``at``;
    ``tangent`` itself for the identity match."""

    if found is None or at is None:
        return tangent
    out = tangent.new_zeros(count)
    return out.index_add(0, at[found], tangent[found])


# -- the Functions ---------------------------------------------------------------------


class _MovedValues(torch.autograd.Function):
    """``values`` (a conversion's output, stored as given) as a function of ``source``
    (its input's history): each output value is the input value stored at its
    coordinate, the sum of those stored at it, or zero."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        values: torch.Tensor,
        source: torch.Tensor,
        source_structure: Any,
        structure: Any,
        identity: bool,
    ) -> torch.Tensor:
        ctx.structures = (source_structure, structure)
        ctx.identity = identity
        ctx.counts = (source.numel(), values.numel())
        return values.detach()  # not ``values`` itself: see the module docstring

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore[override]
        if ctx.identity:
            return None, grad, None, None, None
        source_structure, structure = ctx.structures
        found, at = _positions(structure, source_structure)
        return None, _gather(grad, found, at), None, None, None

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any, values_tangent: Any, source_tangent: Any, *_: Any
    ) -> torch.Tensor:
        if ctx.identity:
            return cast(torch.Tensor, source_tangent)
        source_structure, structure = ctx.structures
        found, at = _positions(structure, source_structure)
        return _scatter(source_tangent, found, at, ctx.counts[1])


class _SummedValues(torch.autograd.Function):
    """``values`` (the stored values of ``a + b``) as a function of the operands'
    histories, either of which may be None."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        values: torch.Tensor,
        a: Optional[torch.Tensor],
        b: Optional[torch.Tensor],
        structures: Tuple[Any, Any, Any],
    ) -> torch.Tensor:
        ctx.structures = structures
        ctx.count = values.numel()
        ctx.present = (a is not None, b is not None)
        return values.detach()  # not ``values`` itself: see the module docstring

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore[override]
        a_structure, b_structure, structure = ctx.structures
        grads: List[Optional[torch.Tensor]] = []
        for present, operand in zip(ctx.present, (a_structure, b_structure)):
            if not present:
                grads.append(None)
                continue
            found, at = _positions(structure, operand)
            grads.append(_gather(grad, found, at))
        return None, grads[0], grads[1], None

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any, values_tangent: Any, a_tangent: Any, b_tangent: Any, *_: Any
    ) -> torch.Tensor:
        a_structure, b_structure, structure = ctx.structures
        total: Optional[torch.Tensor] = None
        for tangent, operand in ((a_tangent, a_structure), (b_tangent, b_structure)):
            if tangent is None:
                continue
            found, at = _positions(structure, operand)
            part = _scatter(tangent, found, at, ctx.count)
            total = part if total is None else total + part
        assert total is not None
        return total


class _CopiedValues(torch.autograd.Function):
    """``values``, a contiguous flat copy of ``source`` made outside autograd (scorch's
    native transpose), as a function of ``source``."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, values: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        ctx.shape = tuple(source.shape)
        return values.detach()  # not ``values`` itself: see the module docstring

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore[override]
        return None, grad.reshape(ctx.shape)

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any, values_tangent: Any, source_tangent: Any
    ) -> torch.Tensor:
        return cast(torch.Tensor, source_tangent).reshape(-1)


# -- the entries -----------------------------------------------------------------------


def link_constructed(
    tensor: Any,
    given: torch.Tensor,
    source_structure: Optional[Any] = None,
) -> None:
    """Give a new STensor built from ``given`` (values that require grad, in grad mode)
    the history of its values: ``given`` itself when the storage kept it, otherwise
    the map from ``given``'s stored coordinates (``source_structure``) to its own."""

    stored = tensor._storage._value
    if (
        given.dim() == 1
        and given.numel() == stored.numel()
        and given.data_ptr() == stored.data_ptr()
        and (given.numel() < 2 or given.stride(0) == 1)
    ):
        attach(tensor, given)
        return
    if source_structure is None:
        raise ScorchRuntimeError(
            "an STensor's storage replaced values that require grad without a map back "
            "to them",
            code="stensor_values_unmapped",
        )
    with torch.enable_grad():
        moved = _MovedValues.apply(
            stored, given, source_structure, _Structure.of(tensor), False
        )
    attach(tensor, moved)


def link_dense_copy(tensor: Any, source: torch.Tensor) -> None:
    """Give a dense STensor whose values are a native copy of ``source`` the history."""

    stored = tensor._storage._value
    if source.is_contiguous() and source.data_ptr() == stored.data_ptr():
        attach(tensor, source.reshape(-1))
        return
    attach(tensor, _CopiedValues.apply(stored, source))


def converted(
    tensor: Any,
    method: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    identity: bool = False,
) -> Any:
    """Run the STensor conversion ``method(tensor, *args, **kwargs)`` with ``tensor``'s
    history set aside, so the conversion itself is today's code on detached values, and
    give its result the history of its values.

    - A new STensor (not in place): in grad mode, its values as a function of
      ``tensor``'s; under ``no_grad``, none.
    - ``tensor`` itself with a new storage (in place): its history follows its
      storage, recorded in any grad mode, since the values autograd tracks must be the
      ones stored. A user's leaf stays the root every graph accumulates into
      (decision 5, option (ii)).
    - ``tensor`` itself with the same storage: its history, unchanged.

    ``identity`` says the result stores the same coordinates in the same order (a
    copy), so the derivative needs no match.
    """

    state = tensor.__dict__
    source = state.pop(AUTOGRAD_VALUES)
    root = state.pop(AUTOGRAD_ROOT, None)
    source_structure = _Structure.of(tensor)
    before = tensor._storage
    try:
        result = method(tensor, *args, **kwargs)
    except BaseException:
        if tensor._storage is before:
            state[AUTOGRAD_VALUES] = source
            if root is not None:
                state[AUTOGRAD_ROOT] = root
        else:
            drop(tensor)
        raise
    if result is not tensor or tensor._storage is before:
        state[AUTOGRAD_VALUES] = source
        if root is not None:
            state[AUTOGRAD_ROOT] = root
        if not tensor._metadata.requires_grad:
            tensor._write_requires_grad(True)
    if result is tensor and tensor._storage is before:
        return result
    if result is not tensor and not torch.is_grad_enabled():
        drop(result)
        return result
    structure = _Structure.of(result)
    same = identity or source_structure.same_as(structure)
    with torch.enable_grad():
        moved = _MovedValues.apply(
            result._storage._value, source, source_structure, structure, same
        )
    if result is tensor and root is None and source.grad_fn is None:
        root = (source, source_structure)
    result.__dict__.pop(AUTOGRAD_ROOT, None)
    attach(result, moved)
    if result is tensor and root is not None:
        state[AUTOGRAD_ROOT] = root
    return result


def summed(a: Any, b: Any, method: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    """``a + b`` (``method``) with both operands' histories set aside, and the result's
    values as a function of both in grad mode."""

    a_values = a.__dict__.pop(AUTOGRAD_VALUES, None)
    a_root = a.__dict__.pop(AUTOGRAD_ROOT, None)
    b_values = b.__dict__.pop(AUTOGRAD_VALUES, None) if b is not a else None
    b_root = b.__dict__.pop(AUTOGRAD_ROOT, None) if b is not a else None
    structures = (_Structure.of(a), _Structure.of(b))
    try:
        result = method(a, b, *args)
    finally:
        for operand, values, root in ((a, a_values, a_root), (b, b_values, b_root)):
            if values is not None:
                operand.__dict__[AUTOGRAD_VALUES] = values
            if root is not None:
                operand.__dict__[AUTOGRAD_ROOT] = root
    if b is a:
        b_values = a_values
    drop(result)
    if not torch.is_grad_enabled() or (a_values is None and b_values is None):
        return result
    summed_values = _SummedValues.apply(
        result._storage._value,
        a_values,
        b_values,
        (structures[0], structures[1], _Structure.of(result)),
    )
    attach(result, summed_values)
    return result


def grad_of(tensor: Any) -> Optional[torch.Tensor]:
    """The gradient of ``tensor``'s values in its current storage order, or None."""

    found = leaf_of(tensor)
    if found is None:
        return None
    leaf, structure = found
    grad = leaf.grad
    if grad is None or structure is None:
        return grad
    current = _Structure.of(tensor)
    hit, at = _positions(structure, current)
    return _gather(grad, hit, at)


def set_grad(tensor: Any, value: Optional[torch.Tensor]) -> None:
    """``tensor.grad = value``: None clears the leaf's gradient; a tensor of the leaf's
    shape in the current storage order is stored as it."""

    found = leaf_of(tensor)
    if found is None:
        raise ScorchTypeError(
            "only an STensor that is a leaf has a gradient to set",
            code="stensor_grad_not_leaf",
        )
    leaf, structure = found
    if value is None:
        leaf.grad = None
        return
    if structure is not None:
        current = _Structure.of(tensor)
        hit, at = _positions(current, structure)
        value = _gather(value, hit, at)
    leaf.grad = value
