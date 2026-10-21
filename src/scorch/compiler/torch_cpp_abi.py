"""Structured Torch/C++ ABI construction shared by compiler lowering stages."""

import math
from dataclasses import dataclass, replace
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, cast

import torch

from . import llir
from .diagnostics import CompilerInvariantError
from .identity import SymbolId
from .llir_traversal import (
    LLIRPath,
    LLIRTraversalContext,
    LLIRValue,
    LLIRWalker,
)
from ..exceptions import (
    IndexCapacityError,
    IndexExtentMismatchError,
    ScorchAssertionError,
    ScorchTypeError,
    ScorchValueError,
    ShapeCapacityKernelError,
    shape_capacity_error,
)
from ..format import MAX_INDEX_POSITION, LevelType
from ..utils import (
    dtype_to_c_datatype,
    get_pytorch_c_dtype_name,
)

_MAX_INT64 = (1 << 63) - 1
_MAX_KERNEL_RANK = 64
_SUPPORTED_KERNEL_DTYPES = (
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
)
#: csrc/native_abi.h, ``LevelKind``: the level kinds a kernel's entry checks know.
_LEVEL_KIND_CODE = {
    LevelType.DENSE: 0,
    LevelType.COMPRESSED: 1,
    LevelType.COORDINATE: 2,
}
#: ``LevelKind::DenseBoundedInKernel``: the code a dense level has in the entry check when
#: the kernel body holds its block to the position bound itself (:func:`dense_block_check`).
_DENSE_BOUNDED_IN_KERNEL_CODE = 3

#: Post-op kinds (``cin.PostOp``) that combine a value with a post-op tensor's entry at
#: the same position, and the update each one emits.
BINARY_POST_OPS = {"add": llir.AssignOp.ADD_ASSIGN, "mul": llir.AssignOp.MUL_ASSIGN}
#: Post-op kinds applied to the value alone, by csrc/header.h's ``scorch_<kind>``.
UNARY_POST_OPS = frozenset({"relu", "gelu", "tanh", "sigmoid"})
#: The unary kinds whose function maps zero to zero (``sigmoid(0)`` is 1/2), the only
#: ones a sparse result can take: its unstored entries are zero and stay so.
ZERO_PRESERVING_POST_OPS = frozenset({"relu", "gelu", "tanh"})

#: csrc/header.h: the call that holds one dense block of the result to the position bound.
DENSE_BLOCK_CHECK = "scorch_check_dense_block"
#: csrc/header.h: the call that holds a compressed level the two-pass assembly counted to
#: the position bound, before the level is allocated (:func:`level_total_check`).
LEVEL_TOTAL_CHECK = "scorch_check_level_total"
#: csrc/header.h: the call that holds a drained sparse workspace to the position bound
#: before the kernel reads it back (:func:`drained_entries_check`).
DRAINED_ENTRIES_CHECK = "scorch_check_drained_entries"


def level_total_check(
    total: llir.Var, level: int, compressed_below: bool
) -> llir.FunctionCallStmt:
    """The check that holds compressed result level ``level``'s entry count to the bound.

    The two-pass assembly emits it once per compressed level, after the prefix sum that
    reads out ``total`` (``_total{level}``, int64_t) and before the level's arrays are
    allocated: the fill pass stores every prefix-sum entry, and every position it fills,
    as ``int``.  ``compressed_below`` is whether a compressed level follows, whose
    position array has one entry more.  The call refuses with
    ``index_position_exceeds_int32`` (csrc/header.h, ``scorch_check_level_total``).  Both
    compilers emit it from here: the shipped two-pass pass
    (compressed_where_openmp_pass.py) and the new compiler's reconstruction of the same
    function (``_ParallelSparseWorkspaceLowering`` in its lower_llir.py).
    """

    if type(total) is not llir.Var or total.type is not llir.DataType.INT64:
        raise ScorchTypeError(
            "a level total check needs the level's int64 total",
            code="internal_argument_type_invalid",
        )
    if type(level) is not int or level < 1:
        raise ScorchValueError(
            "a level total check names a compressed level below level 0",
            code="level_total_check_level_invalid",
        )
    return llir.FunctionCallStmt(
        name=LEVEL_TOTAL_CHECK,
        args=[
            llir.Var(name=total.name, type=total.type),
            llir.Literal(bool(compressed_below), llir.DataType.BOOL),
            llir.Literal(level, llir.DataType.INT),
        ],
    )


def drained_entries_check(entries: llir.Var) -> llir.FunctionCallStmt:
    """The check that holds a drained sparse workspace's entry count to the bound.

    Emitted after a drain that fills a coordinate tensor's arrays through the int64_t
    counter ``entries``, before the kernel reads that tensor back with ``int`` positions
    (``int pT0_end = T0_crd_tensor.size(0)``).  The call refuses with
    ``index_position_exceeds_int32`` (csrc/header.h, ``scorch_check_drained_entries``).
    """

    if type(entries) is not llir.Var or entries.type is not llir.DataType.INT64:
        raise ScorchTypeError(
            "a drained entries check needs the drain's int64 counter",
            code="internal_argument_type_invalid",
        )
    return llir.FunctionCallStmt(
        name=DRAINED_ENTRIES_CHECK,
        args=[llir.Var(name=entries.name, type=entries.type)],
    )


def dense_blocks(level_types: Sequence[LevelType]) -> Tuple[Tuple[int, int], ...]:
    """The result's dense blocks, each as its first and last level.

    A dense block is a run of dense levels directly below a compressed level, and directly
    above a compressed level or nothing.  Each coordinate the compressed level above stores
    owns one block of cells, the product of the run's extents, at consecutive positions
    from the coordinate's position times that product.  These are the levels a kernel can
    hold to the position bound itself, where a block opens (:func:`dense_block_check`).
    """

    blocks: List[Tuple[int, int]] = []
    level = 1
    while level < len(level_types):
        if (
            level_types[level] is not LevelType.DENSE
            or level_types[level - 1] is not LevelType.COMPRESSED
        ):
            level += 1
            continue
        end = level
        while end < len(level_types) and level_types[end] is LevelType.DENSE:
            end += 1
        if end == len(level_types) or level_types[end] is LevelType.COMPRESSED:
            blocks.append((level, end - 1))
        level = end
    return tuple(blocks)


def dense_block_check(
    bounded_levels: Optional[Set[int]],
    result_name: str,
    level_types: Sequence[LevelType],
    first_level: int,
    parent_position: Optional[llir.Expr] = None,
) -> llir.FunctionCallStmt:
    """The check that holds the result's dense block at ``first_level`` to the bound.

    Emitted where the block opens, under the coordinate of compressed level
    ``first_level - 1`` and before the loops that form the block's positions: those are
    multiplied from that coordinate's position -- ``parent_position``, by default its
    pending position ``p<result><first_level - 1>`` -- and narrowed to ``int`` as they
    are formed, before anything is stored.  The call refuses, with
    ``index_position_exceeds_int32``, a block whose end would pass the bound
    (csrc/header.h, ``scorch_check_dense_block``).

    Every site that grows a dense block calls this, and nothing else emits the call.  The
    shipped lowering keeps a record of the levels its body bounds and passes it as
    ``bounded_levels``, to which the block's levels are added; its kernel's ABI declares
    them from that record (:meth:`TorchCppKernelABI.with_levels_bounded_in_kernel`).  The
    new compiler's targets are sealed after construction and keep no record
    (``bounded_levels`` is None): each target class declares the levels it bounds
    (``_TargetLowering._levels_bounded_in_kernel`` in its lower_llir.py).  Either way the
    declaration is checked against the finished body
    (:meth:`TorchCppKernelABI.require_bounded_levels_checked`), in both directions.
    """

    block = next(
        (block for block in dense_blocks(level_types) if block[0] == first_level), None
    )
    if block is None:
        raise ScorchValueError(
            f"level {first_level} does not open a dense block of the result",
            code="dense_block_check_mismatch",
        )
    first, last = block

    def extent(level: int) -> llir.Var:
        return llir.Var(name=f"{result_name}{level}_size", type=llir.DataType.INT64)

    cells: llir.Expr = extent(first)
    for level in range(first + 1, last + 1):
        cells = llir.Mul(left=cells, right=extent(level))
    if bounded_levels is not None:
        bounded_levels.update(range(first, last + 1))
    return llir.FunctionCallStmt(
        name=DENSE_BLOCK_CHECK,
        args=[
            (
                parent_position
                if parent_position is not None
                else llir.Var(
                    name=f"p{result_name}{first - 1}", type=llir.DataType.INT64
                )
            ),
            cells,
            llir.Literal(last + 1 < len(level_types), llir.DataType.BOOL),
            llir.Literal(first, llir.DataType.INT),
            llir.Literal(last, llir.DataType.INT),
        ],
    )


class DenseBlockCheckError(CompilerInvariantError):
    """A kernel's entry check and its body disagree on the dense blocks it bounds.

    The entry check declares a level ``DenseBoundedInKernel`` only because the body checks
    its block where the block opens.  A declared level without the check would let the
    kernel narrow a position past the bound; a check without the declaration would have the
    entry check refuse what the body handles.  ``code`` names the defect.
    """

    code = "dense_block_check_mismatch"


_DENSE_BLOCK_CHECK_CONTEXT = LLIRTraversalContext(
    stage="kernel_abi", pass_name="dense_block_checks"
)


class _DenseBlockChecks(LLIRWalker):
    """The dense blocks a kernel body checks: the levels its check calls name."""

    def __init__(self) -> None:
        super().__init__(_DENSE_BLOCK_CHECK_CONTEXT)
        self.blocks: List[Tuple[int, int]] = []

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        if type(node) is not llir.FunctionCallStmt:
            return
        call = cast(llir.FunctionCallStmt, node)
        if call.name != DENSE_BLOCK_CHECK:
            return
        levels = call.args[3:]
        if len(call.args) != 5 or any(
            type(level) is not llir.Literal
            or type(cast(llir.Literal, level).value) is not int
            for level in levels
        ):
            raise DenseBlockCheckError(
                f"{DENSE_BLOCK_CHECK} at {'/'.join(path)} does not name its levels "
                "as two integer literals"
            )
        first, last = (cast(int, cast(llir.Literal, level).value) for level in levels)
        self.blocks.append((first, last))


def levels_bounded_by_checks(body: LLIRValue) -> FrozenSet[int]:
    """The result levels whose dense blocks ``body`` checks where they open."""

    walker = _DenseBlockChecks()
    walker.walk(body)
    return frozenset(
        level for first, last in walker.blocks for level in range(first, last + 1)
    )


def _validate_known_shape_range(shape: Tuple[int, ...], owner: str) -> None:
    """Reject extents that cannot be represented or safely multiplied."""

    for mode, extent in enumerate(shape):
        if extent > _MAX_INT64:
            raise shape_capacity_error(
                ShapeCapacityKernelError,
                f"known {owner} extents must fit signed int64",
                owner,
                shape,
                mode,
                extent,
            )
    # A zero extent holds no element whatever the extents before it multiply to.
    if 0 not in shape:
        count = math.prod(shape)
        if count > _MAX_INT64:
            raise shape_capacity_error(
                ShapeCapacityKernelError,
                f"known {owner} element count must fit signed int64",
                owner,
                shape,
                None,
                count,
            )


# The two functions below refuse, before a kernel is built, a known shape that the
# kernel's own entry checks would refuse on every call (csrc/native_abi.h,
# ``check_result_positions`` and ``checked_dense_positions``).  The rule lives there, and
# its reasons with it; these repeat the part of it that shapes and layouts decide, with
# the native texts, so that the refusal reads the same whichever side raises it.
# ``tests/test_scorch/test_index_position_bound.py`` holds the two sides to one table.


def _refuse_unaddressable_result(
    shape: Tuple[int, ...],
    level_types: Tuple[LevelType, ...],
    op: str,
    bounded_in_kernel: FrozenSet[int] = frozenset(),
) -> None:
    """Refuse a result whose extents down to its deepest dense level pass ``int``.

    The levels in ``bounded_in_kernel`` are dense blocks the kernel holds to the bound
    itself where each block opens; they bound nothing here except that one block must fit,
    which the first coordinate stored above the block needs.
    """

    bounded_levels = max(
        (
            level + 1
            for level, level_type in enumerate(level_types)
            if level_type is LevelType.DENSE and level not in bounded_in_kernel
        ),
        default=0,
    )
    positions = 1
    for level, level_type in enumerate(level_types[: bounded_levels + 1]):
        if level_type is LevelType.COMPRESSED and positions >= MAX_INDEX_POSITION:
            raise IndexCapacityError(
                f"{op}: result_shape {list(shape)}: compressed level {level} would "
                f"have {positions} parents and one position entry more than that; a "
                f"32-bit position addresses at most {MAX_INDEX_POSITION} entries"
            )
        if level == bounded_levels:
            break
        positions *= shape[level]
        if positions > MAX_INDEX_POSITION:
            raise IndexCapacityError(
                f"{op}: result_shape {list(shape)}: the extents of levels 0 to "
                f"{level} multiply to more than {MAX_INDEX_POSITION}, the most "
                f"positions a 32-bit position addresses, and level "
                f"{bounded_levels - 1} of the result is dense"
            )
    level = 0
    while level < len(level_types):
        if level not in bounded_in_kernel:
            level += 1
            continue
        first = level
        block = 1
        while level < len(level_types) and level in bounded_in_kernel:
            # Left to right, as the kernel forms the block's positions.
            block *= shape[level]
            if block > MAX_INDEX_POSITION:
                raise IndexCapacityError(
                    f"{op}: result_shape {list(shape)}: the extents of dense levels "
                    f"{first} to {level} multiply to more than {MAX_INDEX_POSITION}, "
                    "the most positions a 32-bit position addresses, and each "
                    f"coordinate stored at level {first - 1} owns one block of them"
                )
            level += 1
        if level < len(level_types) and block >= MAX_INDEX_POSITION:
            raise IndexCapacityError(
                f"{op}: result_shape {list(shape)}: the extents of dense levels "
                f"{first} to {level - 1} multiply to {MAX_INDEX_POSITION}, and "
                f"compressed level {level} below them has one position entry more "
                f"than that; a 32-bit position addresses at most {MAX_INDEX_POSITION} "
                "entries"
            )


def _refuse_unaddressable_dense_prefix(
    name: str, shape: Tuple[int, ...], level_types: Tuple[LevelType, ...], op: str
) -> None:
    """Refuse an operand whose leading dense levels pass ``int`` positions.

    What a compressed or coordinate level stores is known only when the kernel runs,
    so the levels from the first sparse one on are left to the kernel's entry check.
    """

    positions = 1
    for level, level_type in enumerate(level_types):
        if level_type is not LevelType.DENSE:
            return
        extent = shape[level]
        if positions * extent > MAX_INDEX_POSITION:
            raise IndexCapacityError(
                f"{op}: {name} dense level {level} of extent {extent} under "
                f"{positions} parent positions needs more than "
                f"{MAX_INDEX_POSITION} positions, the most a 32-bit position "
                "addresses"
            )
        positions *= extent


def tensor_storage_member(tensor_name: str, *members: str) -> llir.MemberAccess:
    """Build a fresh nested member path rooted at one generated ``Tensor``."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise ScorchTypeError(
            "tensor storage root must be a non-empty identifier",
            code="kernel_identifier_invalid",
        )
    if not members:
        raise ScorchTypeError(
            "tensor storage member path cannot be empty",
            code="internal_argument_type_invalid",
        )
    expression: llir.Expr = llir.Var(
        name=tensor_name,
        type=llir.DataType.TACO_TENSOR,
    )
    for member in members:
        expression = llir.MemberAccess(base=expression, member=member)
    return cast(llir.MemberAccess, expression)


def mode_index_tensor(tensor_name: str, level: int, slot: int) -> llir.ArrayAccess:
    """Build a fresh access to one Torch tensor in the mode-index ABI argument."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise ScorchTypeError(
            "mode-index tensor root must be a non-empty identifier",
            code="kernel_identifier_invalid",
        )
    if type(level) is not int or level < 0:
        raise ScorchTypeError(
            "mode-index level must be a non-negative int",
            code="internal_argument_type_invalid",
        )
    if type(slot) is not int or slot < 0:
        raise ScorchTypeError(
            "mode-index slot must be a non-negative int",
            code="internal_argument_type_invalid",
        )
    return llir.ArrayAccess(
        array=llir.ArrayAccess(
            array=llir.Var(
                name=f"{tensor_name}_mode_indices",
                type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
            ),
            index=llir.Literal(level, llir.DataType.INT),
        ),
        index=llir.Literal(slot, llir.DataType.INT),
    )


def tensor_data_ptr(receiver: llir.Expr, data_type: llir.DataType) -> llir.MemberCall:
    """Build a typed ``receiver.data_ptr<T>()`` expression."""

    if not isinstance(receiver, llir.Expr):
        raise ScorchTypeError(
            "tensor data_ptr receiver must be an LLIR Expr",
            code="internal_argument_type_invalid",
        )
    if type(data_type) is not llir.DataType:
        raise ScorchTypeError(
            "tensor data_ptr template argument must be a DataType",
            code="internal_argument_type_invalid",
        )
    return llir.MemberCall(
        base=receiver,
        member="data_ptr",
        template_args=(data_type,),
    )


@dataclass(frozen=True)
class KernelTensorABI:
    """Immutable ABI metadata for one public JIT tensor argument triple."""

    name: str
    level_types: Tuple[LevelType, ...]
    mode_order: Tuple[int, ...]
    shape: Tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.name) is not str or not self.name.isidentifier():
            raise ScorchTypeError(
                "kernel tensor name must be a non-empty identifier",
                code="kernel_identifier_invalid",
            )
        if type(self.level_types) is not tuple or any(
            type(level_type) is not LevelType for level_type in self.level_types
        ):
            raise ScorchTypeError(
                "kernel tensor levels must be an immutable LevelType tuple",
                code="internal_argument_type_invalid",
            )
        if len(self.level_types) > _MAX_KERNEL_RANK:
            raise ScorchValueError(
                f"kernel tensor rank must not exceed {_MAX_KERNEL_RANK}",
                code="kernel_rank_exceeds_limit",
            )
        supported_levels = (
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COORDINATE,
        )
        unsupported_level = next(
            (
                level_type
                for level_type in self.level_types
                if level_type not in supported_levels
            ),
            None,
        )
        if unsupported_level is not None:
            raise ScorchValueError(
                f"unsupported JIT level type {unsupported_level} for ABI validation",
                code="level_type_unknown",
            )
        if type(self.mode_order) is not tuple or any(
            type(level) is not int for level in self.mode_order
        ):
            raise ScorchTypeError(
                "kernel tensor mode order must be an immutable int tuple",
                code="internal_argument_type_invalid",
            )
        rank = len(self.level_types)
        if len(self.mode_order) != rank or set(self.mode_order) != set(range(rank)):
            raise ScorchValueError(
                "kernel tensor mode order must be a rank-sized permutation",
                code="kernel_abi_shape_invalid",
            )
        if type(self.shape) is not tuple or any(
            type(extent) is not int for extent in self.shape
        ):
            raise ScorchTypeError(
                "kernel tensor shape must be an immutable int tuple",
                code="internal_argument_type_invalid",
            )
        if self.shape and len(self.shape) != rank:
            raise ScorchValueError(
                "known kernel tensor shape must match the level rank",
                code="kernel_abi_shape_invalid",
            )
        if any(extent < 0 for extent in self.shape):
            raise ScorchValueError(
                "known kernel tensor extents must be non-negative",
                code="kernel_abi_shape_invalid",
            )
        _validate_known_shape_range(self.shape, "kernel tensor")
        if type(self.dtype) is not torch.dtype:
            raise ScorchTypeError(
                "kernel tensor dtype must be an exact torch.dtype",
                code="internal_argument_type_invalid",
            )
        if self.dtype not in _SUPPORTED_KERNEL_DTYPES:
            raise ScorchValueError(
                f"unsupported kernel tensor dtype {self.dtype}",
                code="dtype_unsupported",
            )

    def _level_kind_codes(self) -> Tuple[int, ...]:
        self._validate()
        return tuple(_LEVEL_KIND_CODE[level] for level in self.level_types)

    def emit_level_array_bindings(self) -> List[llir.Stmt]:
        """Build fresh dense extents and sparse mode-index pointer bindings."""

        self._validate()
        stmts: List[llir.Stmt] = []
        for level, level_type in enumerate(self.level_types):
            if level_type == LevelType.DENSE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_size",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{self.name}_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                            index=llir.Literal(
                                value=level,
                                data_type=llir.DataType.INT64,
                            ),
                        ),
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_pos",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 0),
                            llir.DataType.INT,
                        ),
                    )
                )
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 1),
                            llir.DataType.INT,
                        ),
                    )
                )
            elif level_type == LevelType.COORDINATE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd_tensor",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=mode_index_tensor(self.name, level, 0),
                    )
                )
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 0),
                            llir.DataType.INT,
                        ),
                    )
                )
        return stmts

    def emit_value_pointer(self) -> llir.VarInit:
        """Build a fresh typed pointer binding for this tensor's values."""

        self._validate()
        data_type = dtype_to_c_datatype(self.dtype)
        return llir.VarInit(
            var=llir.Var(
                name=f"{self.name}_val",
                type=llir.DataType.ptr_type(self.dtype),
                is_restrict=True,
            ),
            value=tensor_data_ptr(
                llir.Var(
                    name=f"{self.name}_values",
                    type=llir.DataType.TORCH_TENSOR,
                ),
                data_type,
            ),
        )


def index_extent_bindings(
    arguments: Sequence[Tuple[Sequence[int], Sequence[Sequence[Tuple[object, str]]]]],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int, int], ...]]:
    """A kernel ABI's ``index_names`` and ``index_bindings``, from its accesses.

    ``arguments`` holds, for the result and then each input in ABI order, its mode order
    (``mode_order[level]`` is the logical axis stored at ``level``; empty for the identity)
    and its accesses, each one ``(key, name)`` per logical axis: ``key`` identifies the
    index variable and ``name`` displays it.  An index bound to fewer than two distinct
    argument dimensions is left out, since there is nothing to compare; one bound twice
    in one access (``A(i, i)``) is kept.  An axis the mode order does not hold is skipped,
    as the CIN analysis skips it; that access is refused elsewhere.
    """

    dimensions: dict = {}
    names: dict = {}
    for argument, (mode_order, accesses) in enumerate(arguments):
        for access in accesses:
            for axis, (key, name) in enumerate(access):
                if mode_order and axis not in mode_order:
                    continue
                level = list(mode_order).index(axis) if mode_order else axis
                if key not in dimensions:
                    dimensions[key] = []
                    names[key] = name
                if (argument, level) not in dimensions[key]:
                    dimensions[key].append((argument, level))
    kept = [key for key, bound in dimensions.items() if len(bound) >= 2]
    index_names = tuple(
        names[key] if names[key].isidentifier() else f"index{position}"
        for position, key in enumerate(kept)
    )
    bindings = tuple(
        (position, argument, level)
        for position, key in enumerate(kept)
        for argument, level in sorted(dimensions[key])
    )
    return index_names, bindings


def _int_initializer_list(
    values: Tuple[int, ...],
    element_type: llir.DataType,
) -> llir.Array:
    """Build one typed initializer list from validated ABI integer metadata."""

    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise ScorchTypeError(
            "initializer-list values must be an immutable int tuple",
            code="internal_argument_type_invalid",
        )
    if type(element_type) is not llir.DataType:
        raise ScorchTypeError(
            "initializer-list element type must be a DataType",
            code="internal_argument_type_invalid",
        )
    return llir.Array(
        values=tuple(
            llir.Literal(value=value, data_type=element_type) for value in values
        ),
        data_type=element_type,
    )


@dataclass(frozen=True)
class TorchCppKernelABI:
    """Own one frozen public ``evaluate`` signature and input ABI contract."""

    result_shape: Tuple[int, ...]
    result_level_types: Tuple[LevelType, ...]
    input_tensors: Tuple[KernelTensorABI, ...]
    extra_tensor_names: Tuple[str, ...] = ()
    extra_tensor_dtype: Optional[torch.dtype] = None
    function_name: str = "evaluate"
    #: The result levels the kernel body holds to the position bound itself, whole dense
    #: blocks (:func:`dense_block_check`); the entry check gives them the kind
    #: ``DenseBoundedInKernel``.  ``None`` while the body is being built: construction then
    #: refuses only what no body could address, with every dense block counted as bounded,
    #: and the entry checks cannot be emitted until the body's record is known
    #: (:meth:`with_levels_bounded_in_kernel`).
    result_levels_bounded_in_kernel: Optional[Tuple[int, ...]] = ()
    #: The index variables bound to two or more argument dimensions, and those dimensions:
    #: ``(index, argument, level)`` triples, ``index`` a position in ``index_names``,
    #: ``argument`` 0 for the result and ``k`` for the k-th input, ``level`` a physical
    #: level (shape arguments are in physical level order), an index's triples together and
    #: the indices in order (:func:`index_extent_bindings`).  The extents of an index's
    #: dimensions must agree: refused at construction where every one is known
    #: (:meth:`_refuse_disagreeing_extents`), and at the kernel's entry otherwise.
    index_names: Tuple[str, ...] = ()
    index_bindings: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        self._validate()
        self._refuse_unaddressable_shapes()
        self._refuse_disagreeing_extents()

    @property
    def result_rank(self) -> int:
        return len(self.result_level_types)

    def _bounded_in_kernel(self) -> FrozenSet[int]:
        """The levels the refusals count as bounded in the kernel."""

        if self.result_levels_bounded_in_kernel is None:
            return frozenset(
                level
                for first, last in dense_blocks(self.result_level_types)
                for level in range(first, last + 1)
            )
        return frozenset(self.result_levels_bounded_in_kernel)

    def _require_body_record(self, what: str) -> Tuple[int, ...]:
        if self.result_levels_bounded_in_kernel is None:
            raise ScorchValueError(
                f"{what} needs the dense blocks the kernel body bounds, which are not "
                "known until the body is built (with_levels_bounded_in_kernel)",
                code="abi_body_record_missing",
            )
        return self.result_levels_bounded_in_kernel

    def with_levels_bounded_in_kernel(
        self, levels: Iterable[int]
    ) -> "TorchCppKernelABI":
        """This ABI once the body is built: ``levels`` are the ones its checks bound.

        Construction runs again, so the refusals now count only those levels as bounded.
        """

        return replace(self, result_levels_bounded_in_kernel=tuple(sorted(set(levels))))

    def require_bounded_levels_checked(self, body: LLIRValue) -> None:
        """Refuse a kernel whose entry check and finished body disagree on its blocks.

        Read from the body's LLIR (the check calls :func:`dense_block_check` builds), after
        every pass, so a pass that dropped or copied a check cannot go unnoticed.
        """

        declared = frozenset(self._require_body_record("checking the body"))
        checked = levels_bounded_by_checks(body)
        if declared != checked:
            raise DenseBlockCheckError(
                f"{self.function_name}: the entry check declares result levels "
                f"{sorted(declared)} bounded in the kernel, and the body checks the "
                f"blocks of levels {sorted(checked)}"
            )

    def _result_level_kinds(self) -> Tuple[int, ...]:
        bounded = self._require_body_record("the entry check")
        return tuple(
            (
                _DENSE_BOUNDED_IN_KERNEL_CODE
                if level in bounded
                else _LEVEL_KIND_CODE[level_type]
            )
            for level, level_type in enumerate(self.result_level_types)
        )

    def _refuse_unaddressable_shapes(self) -> None:
        """Refuse known shapes no kernel of this ABI could address, before one is built.

        At construction: the fields are frozen, so the answer cannot change.
        """

        if self.result_shape:
            _refuse_unaddressable_result(
                self.result_shape,
                self.result_level_types,
                self.function_name,
                self._bounded_in_kernel(),
            )
        for tensor in self.input_tensors:
            if tensor.shape:
                _refuse_unaddressable_dense_prefix(
                    tensor.name, tensor.shape, tensor.level_types, self.function_name
                )

    def _argument(self, argument: int) -> Tuple[str, int, Tuple[int, ...]]:
        """Argument ``argument``'s name, rank and known shape (empty when not known)."""

        if argument == 0:
            return "result", self.result_rank, self.result_shape
        tensor = self.input_tensors[argument - 1]
        return tensor.name, len(tensor.level_types), tensor.shape

    def _validate_index_bindings(self) -> None:
        if type(self.index_names) is not tuple or any(
            type(name) is not str or not name.isidentifier()
            for name in self.index_names
        ):
            raise ScorchTypeError(
                "index names must be an immutable identifier tuple",
                code="internal_argument_type_invalid",
            )
        if type(self.index_bindings) is not tuple or any(
            type(binding) is not tuple
            or len(binding) != 3
            or any(type(part) is not int for part in binding)
            for binding in self.index_bindings
        ):
            raise ScorchTypeError(
                "index bindings must be an immutable tuple of (index, argument, level) "
                "int triples",
                code="internal_argument_type_invalid",
            )
        counts = [0] * len(self.index_names)
        previous = -1
        for index, argument, level in self.index_bindings:
            if not 0 <= index < len(self.index_names) or index < previous:
                raise ScorchValueError(
                    "index bindings must name their indices in order",
                    code="kernel_abi_index_binding_invalid",
                )
            if not 0 <= argument <= len(self.input_tensors):
                raise ScorchValueError(
                    f"index binding names argument {argument} of "
                    f"{len(self.input_tensors) + 1}",
                    code="kernel_abi_index_binding_invalid",
                )
            if not 0 <= level < self._argument(argument)[1]:
                raise ScorchValueError(
                    f"index binding names level {level} of argument {argument}, which "
                    f"has {self._argument(argument)[1]}",
                    code="kernel_abi_index_binding_invalid",
                )
            counts[index] += 1
            previous = index
        if any(count < 2 for count in counts):
            raise ScorchValueError(
                "every bound index must be bound to two argument dimensions or more",
                code="kernel_abi_index_binding_invalid",
            )

    def _index_groups(self) -> List[List[Tuple[int, int]]]:
        """Each index's ``(argument, level)`` pairs, in index order."""

        groups: List[List[Tuple[int, int]]] = [[] for _ in self.index_names]
        for index, argument, level in self.index_bindings:
            groups[index].append((argument, level))
        return groups

    def _refuse_disagreeing_extents(self) -> None:
        """Refuse two known extents of one index that differ, before a kernel is built.

        At construction, as for :meth:`_refuse_unaddressable_shapes`: a kernel compiled for
        known shapes would be built only for its entry check to refuse every call.
        """

        for index, group in enumerate(self._index_groups()):
            first: Optional[Tuple[str, int, int]] = None
            for argument, level in group:
                name, _, shape = self._argument(argument)
                if not shape:
                    continue
                if first is None:
                    first = (name, level, shape[level])
                elif shape[level] != first[2]:
                    raise IndexExtentMismatchError(
                        f"{self.function_name}: index '{self.index_names[index]}' has "
                        f"extent {first[2]} in {first[0]} (level {first[1]}) and "
                        f"{shape[level]} in {name} (level {level})"
                    )

    def _entry_checked_bindings(self) -> Tuple[Tuple[int, int, int], ...]:
        """The bindings of the indices one of whose extents is not known: the ones the
        kernel's entry must compare (every known one was compared at construction)."""

        groups = self._index_groups()
        return tuple(
            (index, argument, level)
            for index, argument, level in self.index_bindings
            if any(not self._argument(other)[2] for other, _ in groups[index])
        )

    def _validate(self) -> None:
        if type(self.function_name) is not str or not self.function_name.isidentifier():
            raise ScorchTypeError(
                "kernel function name must be a non-empty identifier",
                code="kernel_identifier_invalid",
            )
        if type(self.result_shape) is not tuple or any(
            type(extent) is not int for extent in self.result_shape
        ):
            raise ScorchTypeError(
                "kernel result shape must be an immutable int tuple",
                code="internal_argument_type_invalid",
            )
        if type(self.result_level_types) is not tuple or any(
            type(level_type) is not LevelType for level_type in self.result_level_types
        ):
            raise ScorchTypeError(
                "kernel result levels must be an immutable LevelType tuple",
                code="internal_argument_type_invalid",
            )
        if self.result_rank > _MAX_KERNEL_RANK:
            raise ScorchValueError(
                f"kernel result rank must not exceed {_MAX_KERNEL_RANK}",
                code="kernel_rank_exceeds_limit",
            )
        unsupported_level = next(
            (
                level_type
                for level_type in self.result_level_types
                if level_type not in _LEVEL_KIND_CODE
            ),
            None,
        )
        if unsupported_level is not None:
            raise ScorchValueError(
                f"unsupported JIT level type {unsupported_level} for ABI validation",
                code="level_type_unknown",
            )
        if self.result_shape and len(self.result_shape) != self.result_rank:
            raise ScorchValueError(
                "known kernel result shape must match the result rank",
                code="kernel_abi_shape_invalid",
            )
        bounded = self.result_levels_bounded_in_kernel
        if bounded is not None:
            if (
                type(bounded) is not tuple
                or any(type(level) is not int for level in bounded)
                or list(bounded) != sorted(set(bounded))
            ):
                raise ScorchTypeError(
                    "result levels bounded in the kernel must be a sorted tuple of "
                    "distinct ints",
                    code="internal_argument_type_invalid",
                )
            unplaced = set(bounded)
            for first, last in dense_blocks(self.result_level_types):
                block = set(range(first, last + 1))
                if block & unplaced and not block <= unplaced:
                    raise ScorchValueError(
                        f"result levels {first} to {last} form one dense block, which "
                        "the kernel bounds whole or not at all",
                        code="dense_block_check_mismatch",
                    )
                unplaced -= block
            if unplaced:
                raise ScorchValueError(
                    f"result levels {sorted(unplaced)} are declared bounded in the "
                    "kernel but are not in a dense block below a compressed level",
                    code="dense_block_check_mismatch",
                )
        if any(extent < 0 for extent in self.result_shape):
            raise ScorchValueError(
                "known kernel result extents must be non-negative",
                code="kernel_abi_shape_invalid",
            )
        _validate_known_shape_range(self.result_shape, "kernel result")
        if type(self.input_tensors) is not tuple or any(
            type(tensor) is not KernelTensorABI for tensor in self.input_tensors
        ):
            raise ScorchTypeError(
                "kernel inputs must be an immutable exact KernelTensorABI tuple",
                code="internal_argument_type_invalid",
            )
        for tensor in self.input_tensors:
            tensor._validate()
        if type(self.extra_tensor_names) is not tuple or any(
            type(name) is not str or not name.isidentifier()
            for name in self.extra_tensor_names
        ):
            raise ScorchTypeError(
                "extra tensor names must be an immutable identifier tuple",
                code="internal_argument_type_invalid",
            )
        if self.extra_tensor_names:
            if type(self.extra_tensor_dtype) is not torch.dtype:
                raise ScorchTypeError(
                    "extra tensor dtype must be an exact torch.dtype when present",
                    code="internal_argument_type_invalid",
                )
            if self.extra_tensor_dtype not in _SUPPORTED_KERNEL_DTYPES:
                raise ScorchValueError(
                    f"unsupported extra tensor dtype {self.extra_tensor_dtype}",
                    code="dtype_unsupported",
                )
        elif self.extra_tensor_dtype is not None:
            raise ScorchTypeError(
                "extra tensor dtype must be None without extra tensors",
                code="internal_argument_type_invalid",
            )
        if self.extra_tensor_names and any(
            level_type is not LevelType.DENSE for level_type in self.result_level_types
        ):
            # The body reads an extra tensor at the result's cell; with a sparse level that
            # is a storage position, whose count no caller can know to size the tensor.
            raise ScorchValueError(
                "a post-op's extra tensor is read at the result's cell, which a result "
                "with a sparse level does not have",
                code="post_op_extra_tensor_sparse_result",
            )
        self._validate_index_bindings()
        argument_names = ["result_shape"]
        for tensor in self.input_tensors:
            argument_names.extend(
                (
                    f"{tensor.name}_shape",
                    f"{tensor.name}_mode_indices",
                    f"{tensor.name}_values",
                )
            )
        argument_names.extend(f"{name}_values" for name in self.extra_tensor_names)
        if len(argument_names) != len(set(argument_names)):
            raise ScorchValueError(
                "kernel ABI argument names must be unique",
                code="kernel_identifier_invalid",
            )

    def emit_arguments(self) -> List[llir.Var]:
        """Build a fresh ordered public function-argument list."""

        self._validate()
        arguments = [
            llir.Var(
                name="result_shape",
                type=llir.DataType.STD_VECTOR_INT,
            )
        ]
        for tensor in self.input_tensors:
            arguments.extend(
                [
                    llir.Var(
                        name=f"{tensor.name}_shape",
                        type=llir.DataType.STD_VECTOR_INT,
                    ),
                    llir.Var(
                        name=f"{tensor.name}_mode_indices",
                        type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                    ),
                    llir.Var(
                        name=f"{tensor.name}_values",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                ]
            )
        for name in self.extra_tensor_names:
            arguments.append(
                llir.Var(
                    name=f"{name}_values",
                    type=llir.DataType.TORCH_TENSOR,
                )
            )
        return arguments

    def emit_validation(self) -> List[llir.FunctionCallStmt]:
        """Build fresh validation calls in exact public-argument order."""

        self._validate()
        validation = [
            llir.FunctionCallStmt(
                name="scorch_native::validate_jit_result_shape",
                args=(
                    llir.Var(
                        name="result_shape",
                        type=llir.DataType.STD_VECTOR_INT,
                    ),
                    _int_initializer_list(
                        self.result_shape,
                        llir.DataType.INT64,
                    ),
                    # The result's layout: its bound depends on which levels are dense,
                    # and on which dense blocks the body bounds itself
                    # (csrc/native_abi.h, check_result_positions).
                    _int_initializer_list(
                        self._result_level_kinds(),
                        llir.DataType.INT,
                    ),
                    llir.Literal(
                        value=self.function_name,
                        data_type=llir.DataType.STRING,
                    ),
                ),
            )
        ]
        for tensor in self.input_tensors:
            validation.append(
                llir.FunctionCallStmt(
                    name="scorch_native::validate_jit_tensor",
                    args=(
                        llir.Literal(
                            value=self.function_name,
                            data_type=llir.DataType.STRING,
                        ),
                        llir.Literal(
                            value=tensor.name,
                            data_type=llir.DataType.STRING,
                        ),
                        llir.Var(
                            name=f"{tensor.name}_shape",
                            type=llir.DataType.STD_VECTOR_INT,
                        ),
                        llir.Var(
                            name=f"{tensor.name}_mode_indices",
                            type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                        ),
                        llir.Var(
                            name=f"{tensor.name}_values",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        llir.QualifiedName(
                            namespace="torch",
                            name=get_pytorch_c_dtype_name(tensor.dtype),
                            data_type=llir.DataType.TORCH_SCALAR_TYPE,
                        ),
                        _int_initializer_list(
                            tensor._level_kind_codes(),
                            llir.DataType.INT,
                        ),
                        _int_initializer_list(
                            tensor.mode_order,
                            llir.DataType.INT,
                        ),
                        _int_initializer_list(
                            tensor.shape,
                            llir.DataType.INT64,
                        ),
                    ),
                )
            )
        if self.extra_tensor_names:
            if self.extra_tensor_dtype is None:
                raise ScorchAssertionError(
                    "self.extra_tensor_dtype is not None",
                    code="extra_tensor_dtype_missing",
                )
            for name in self.extra_tensor_names:
                validation.append(
                    llir.FunctionCallStmt(
                        name="scorch_native::validate_jit_extra_tensor",
                        args=(
                            llir.Var(
                                name=f"{name}_values",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.extra_tensor_dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                            llir.Literal(
                                value=self.function_name,
                                data_type=llir.DataType.STRING,
                            ),
                            llir.Literal(
                                value=f"{name}_values",
                                data_type=llir.DataType.STRING,
                            ),
                            # Read at the result's cell: it must hold that many values.
                            llir.Var(
                                name="result_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                        ),
                    )
                )
        extent_check = self._emit_index_extent_check()
        if extent_check is not None:
            validation.append(extent_check)
        return validation

    def _emit_index_extent_check(self) -> Optional[llir.FunctionCallStmt]:
        """The entry's comparison of the extents its indices are bound to, or ``None``
        when every such extent was compared at construction.  Last of the entry checks:
        each shape argument's rank has been checked by then (the call checks it too)."""

        bindings = self._entry_checked_bindings()
        if not bindings:
            return None
        indices = sorted({index for index, _, _ in bindings})
        renumbered = {index: position for position, index in enumerate(indices)}
        # Each argument's shape parameter is "<argument>_shape" (emit_arguments).
        argument_names = ["result"] + [tensor.name for tensor in self.input_tensors]
        # The shape arguments go last and by reference (the helper is variadic), so the
        # call takes no address.
        return llir.FunctionCallStmt(
            name="scorch_native::validate_jit_index_extents",
            args=(
                llir.Literal(value=self.function_name, data_type=llir.DataType.STRING),
                llir.Array(
                    values=tuple(
                        llir.Literal(value=name, data_type=llir.DataType.STRING)
                        for name in argument_names
                    ),
                    data_type=llir.DataType.STRING,
                ),
                llir.Array(
                    values=tuple(
                        llir.Literal(
                            value=self.index_names[index],
                            data_type=llir.DataType.STRING,
                        )
                        for index in indices
                    ),
                    data_type=llir.DataType.STRING,
                ),
                _int_initializer_list(
                    tuple(
                        part
                        for index, argument, level in bindings
                        for part in (renumbered[index], argument, level)
                    ),
                    llir.DataType.INT,
                ),
                *(
                    llir.Var(
                        name=f"{argument}_shape", type=llir.DataType.STD_VECTOR_INT
                    )
                    for argument in argument_names
                ),
            ),
        )

    def emit_input_prologue(self) -> List[llir.Stmt]:
        """Build fresh shape/index/value bindings for every input tensor."""

        self._validate()
        stmts: List[llir.Stmt] = []
        for tensor in self.input_tensors:
            stmts.append(llir.BlankLine())
            stmts.append(llir.Comment(f"Get {tensor.name}'s level & value arrays"))
            stmts.extend(tensor.emit_level_array_bindings())
            stmts.append(tensor.emit_value_pointer())
        return stmts

    def emit_extra_tensor_prologue(self) -> List[llir.Stmt]:
        """Build fresh typed value pointers for post-op tensor arguments."""

        self._validate()
        if not self.extra_tensor_names:
            return []
        if self.extra_tensor_dtype is None:
            raise ScorchAssertionError(
                "self.extra_tensor_dtype is not None", code="extra_tensor_dtype_missing"
            )
        c_dtype = dtype_to_c_datatype(self.extra_tensor_dtype)
        ptr_type = llir.DataType.ptr_type(self.extra_tensor_dtype)
        return [
            llir.VarInit(
                var=llir.Var(
                    name=f"{name}_val",
                    type=ptr_type,
                    is_restrict=True,
                ),
                value=tensor_data_ptr(
                    llir.Var(
                        name=f"{name}_values",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    c_dtype,
                ),
            )
            for name in self.extra_tensor_names
        ]

    def signature(self) -> Tuple[llir.DataType, str, List[llir.Var]]:
        """The frozen public signature: return type, name, fresh arguments.

        One source for the spelling :meth:`assemble_function` emits, so a
        boundary that needs to state what the signature *must* be can ask for
        it rather than restating the policy.  Every call is a pure function of
        this frozen metadata and returns freshly built, unshared arguments, so
        asking twice cannot alias the assembled function's own state.
        """

        self._validate()
        return (llir.DataType.TACO_TENSOR, self.function_name, self.emit_arguments())

    def assemble_function(self, body: List[llir.Stmt]) -> llir.Function:
        """Wrap one verified body in a fresh ABI-owned function signature.

        :meth:`signature` validates this frozen metadata before doing anything,
        so it is the single validation of one assembly; the body type-check does
        not depend on it and stays behind it, which keeps a malformed-metadata
        call failing on the metadata exactly as it did when this method
        validated a second time itself.
        """

        return_type, name, args = self.signature()
        if type(body) is not list:
            raise ScorchTypeError(
                "kernel function body must be an exact LLIR list",
                code="internal_argument_type_invalid",
            )
        return llir.Function(
            return_type=return_type,
            name=name,
            args=args,
            body=list(body),
        )


@dataclass(frozen=True)
class ResultTensorAssembler:
    """Build result storage and final ABI assembly from an owned metadata snapshot."""

    name: str
    level_types: Tuple[LevelType, ...]
    dtype: torch.dtype
    known_nnz_var: Optional[str] = None
    exact_dense_parent_positions: bool = False
    reserve_hint_var: Optional[str] = None
    #: WHICH result this assembler is assembling, as a logical identity rather
    #: than a name.  Every storage name in this class is built from ``self.name``
    #: and a level number, which says what a statement touches and not whose
    #: storage it is, and ``llir.ResultStorageMetadata`` needs the second.  So a
    #: caller that holds an identity hands it over and this assembler can then
    #: mark the statements it emits; a caller that holds none passes nothing and
    #: every statement is emitted exactly as before.
    #:
    #: Optional rather than required because the two routes differ.  The typed
    #: route's lowering owns a ``SymbolId`` for the result and passes it.  The
    #: legacy route builds this from a ``TensorVar``, whose identity is a name,
    #: and passes nothing -- so nothing legacy emits gains a marker, which is
    #: what keeps this change off the shipping path.
    result_id: Optional[SymbolId] = None
    #: The post-ops this assembler applies to every stored value once the loop
    #: nest has finished, as ``(kind, post-op tensor name or None)`` pairs in
    #: program order (``emit_value_post_ops``).  Empty when there are none, and
    #: when the lowering applied them where it copied each cell out of a dense
    #: workspace; then every statement is emitted exactly as before.
    value_post_ops: Tuple[Tuple[str, Optional[str]], ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the frozen result-ABI metadata snapshot."""

        if type(self.name) is not str or not self.name.isidentifier():
            raise ScorchTypeError(
                "result tensor name must be a non-empty identifier",
                code="kernel_identifier_invalid",
            )
        if type(self.level_types) is not tuple or any(
            type(level_type) is not LevelType for level_type in self.level_types
        ):
            raise ScorchTypeError(
                "result level types must be an immutable LevelType tuple",
                code="internal_argument_type_invalid",
            )
        if len(self.level_types) > _MAX_KERNEL_RANK:
            raise ScorchValueError(
                f"result rank must not exceed {_MAX_KERNEL_RANK}",
                code="kernel_rank_exceeds_limit",
            )
        if type(self.dtype) is not torch.dtype:
            raise ScorchTypeError(
                "result dtype must be an exact torch.dtype",
                code="internal_argument_type_invalid",
            )
        if self.dtype not in _SUPPORTED_KERNEL_DTYPES:
            raise ScorchValueError(
                f"unsupported result dtype {self.dtype}", code="dtype_unsupported"
            )
        if self.known_nnz_var is not None and (
            type(self.known_nnz_var) is not str or not self.known_nnz_var.isidentifier()
        ):
            raise ScorchTypeError(
                "known nnz variable must be an identifier or None",
                code="internal_argument_type_invalid",
            )
        if type(self.exact_dense_parent_positions) is not bool:
            raise ScorchTypeError(
                "exact dense parent positions must be a bool",
                code="internal_argument_type_invalid",
            )
        if self.reserve_hint_var is not None and (
            type(self.reserve_hint_var) is not str
            or not self.reserve_hint_var.isidentifier()
        ):
            raise ScorchTypeError(
                "reserve hint variable must be an identifier or None",
                code="internal_argument_type_invalid",
            )
        if self.result_id is not None and type(self.result_id) is not SymbolId:
            raise ScorchTypeError(
                "result identity must be an exact SymbolId or None",
                code="internal_argument_type_invalid",
            )
        if type(self.value_post_ops) is not tuple or any(
            type(op) is not tuple or len(op) != 2 or type(op[0]) is not str
            for op in self.value_post_ops
        ):
            raise ScorchTypeError(
                "value post-ops must be a tuple of (kind, tensor name) pairs",
                code="internal_argument_type_invalid",
            )
        sparse = any(
            level_type is not LevelType.DENSE for level_type in self.level_types
        )
        for kind, tensor_name in self.value_post_ops:
            if kind in BINARY_POST_OPS:
                if type(tensor_name) is not str or not tensor_name.isidentifier():
                    raise ScorchTypeError(
                        f"post-op {kind!r} needs a tensor name that is an identifier",
                        code="internal_argument_type_invalid",
                    )
                if sparse:
                    raise ScorchValueError(
                        "a post-op's extra tensor is read at the result's cell, "
                        "which a result with a sparse level does not have",
                        code="post_op_extra_tensor_sparse_result",
                    )
            elif kind in UNARY_POST_OPS:
                if tensor_name is not None:
                    raise ScorchTypeError(
                        f"post-op {kind!r} takes no tensor",
                        code="internal_argument_type_invalid",
                    )
                if sparse and kind not in ZERO_PRESERVING_POST_OPS:
                    raise ScorchValueError(
                        f"post-op {kind!r} does not map zero to zero, so it would "
                        "change every entry a sparse result does not store",
                        code="post_op_sparse_result_changes_zero",
                    )
            else:
                raise ScorchValueError(
                    f"unsupported post-op kind {kind!r}",
                    code="post_op_kind_unsupported",
                )

    def emit_value_post_ops(self) -> List[llir.Stmt]:
        """Apply the post-ops to every stored value of the finished result.

        Emitted after the loop nest and before the values become the result's
        storage, where every value is final whatever the loops did: a dense
        result's value tensor holds every cell, a loop that skipped a cell left
        it zero, and a cell a reduction reached several times holds its sum.
        So this is the one place correct for every result.  A post-op tensor
        is read at the same position as the value; its argument check holds it
        to the result's cell count, and einsum hands it over in the result's
        storage order.  A sparse result's unstored entries stay zero, so only
        post-ops that map zero to zero, with no tensor, apply to one
        (``validate``).

        The values are read through a fresh pointer into
        ``{name}_values_torch``, the tensor every assembly path stores as the
        result's values.  The loop's thread count is chosen from its length,
        as for the parallel initialization loops.
        """

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "value post-ops require an exact ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        if not self.value_post_ops:
            return []
        c_datatype = dtype_to_c_datatype(self.dtype)
        values = f"{self.name}_post_op_values"
        count = f"{self.name}_post_op_count"
        position = f"{self.name}_post_op_position"

        def target() -> llir.ArrayAccess:
            return llir.ArrayAccess(
                array=llir.Var(name=values, type=llir.DataType.ptr_type(c_datatype)),
                index=llir.Var(name=position, type=llir.DataType.INT64),
            )

        body: List[llir.Stmt] = []
        for kind, tensor_name in self.value_post_ops:
            if kind in BINARY_POST_OPS:
                body.append(
                    llir.Assign(
                        var=target(),
                        value=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{tensor_name}_val", type=llir.DataType.NO_TYPE
                            ),
                            index=llir.Var(name=position, type=llir.DataType.INT64),
                        ),
                        op=BINARY_POST_OPS[kind],
                    )
                )
            else:
                body.append(
                    llir.Assign(
                        var=target(),
                        value=llir.FunctionCall(
                            name=f"scorch_{kind}",
                            args=[target()],
                        ),
                        op=llir.AssignOp.ASSIGN,
                    )
                )
        values_torch = llir.Var(
            name=f"{self.name}_values_torch", type=llir.DataType.TORCH_TENSOR
        )
        return [
            llir.Comment("Apply the post-ops to every stored value"),
            llir.VarInit(
                var=llir.Var(
                    name=values,
                    type=llir.DataType.ptr_type(c_datatype),
                    is_restrict=True,
                ),
                value=tensor_data_ptr(values_torch, c_datatype),
            ),
            llir.VarInit(
                var=llir.Var(name=count, type=llir.DataType.INT64),
                value=llir.MemberCall(
                    base=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    member="numel",
                    args=(),
                ),
            ),
            llir.ForLoop(
                init=llir.VarInit(
                    var=llir.Var(name=position, type=llir.DataType.INT64),
                    value=llir.Literal(value=0, data_type=llir.DataType.INT64),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=llir.Var(name=position, type=llir.DataType.INT64),
                    right=llir.Var(name=count, type=llir.DataType.INT64),
                ),
                update=llir.Increment(
                    var=llir.Var(name=position, type=llir.DataType.INT64)
                ),
                body=body,
                omp_parallel_for=True,
                omp_schedule="static",
                omp_num_threads=f"scorch_nthreads({count}, {count})",
            ),
        ]

    def result_storage_marker(
        self,
        *references: Tuple[llir.ResultStorageArray, Optional[int], bool],
    ) -> Optional[llir.ResultStorageMetadata]:
        """One emitted statement's result-storage marker, or ``None``.

        This is the per-result identity this file did not have.  A storage name
        here is ``f"{self.name}{level}_pos"`` and the like, which says which
        vector a statement touches; ``result_write_pass`` needs to know whose
        vector it is, and the name cannot say that.  With ``result_id`` set, this
        answers it.

        ``None`` when no identity was handed over, which is the legacy route, and
        an unmarked statement is what that route emits today.  Returning ``None``
        rather than refusing is deliberate: a marker is a claim about identity, and
        a caller with no identity has no claim to make.

        Each reference is ``(array, level, writes)`` -- ``level`` is ``None`` for
        the value vector, which has none, and ``writes`` is carried rather than
        derived because an argument position does not say which way a reference
        goes.  See ``llir.ResultStorageDirection``.
        """

        if self.result_id is None:
            return None
        return llir.ResultStorageMetadata(
            # Rebuilt rather than shared, the discipline every other marker
            # builder in the compiler follows: nothing reachable from an emitted
            # LLIR node aliases a caller's own identity object.
            tensor_id=SymbolId(self.result_id.value),
            references=tuple(
                llir.ResultStorageReference(
                    array,
                    level,
                    (
                        llir.ResultStorageDirection.WRITE
                        if writes
                        else llir.ResultStorageDirection.READ
                    ),
                )
                for array, level, writes in references
            ),
        )

    @property
    def levels(self) -> int:
        return len(self.level_types)

    @property
    def is_dense(self) -> bool:
        return all(level_type == LevelType.DENSE for level_type in self.level_types)

    def _has_fixed_position_count(self, level: int) -> bool:
        """A compressed level below exactly one dense level has that extent + 1 slots.

        Level 1 under a dense level 0 only.  Under several dense levels the
        count is the product of their extents, and under a compressed ancestor
        it is fixed only by assembly; both are append-built, as the typed
        lowering builds them.
        """
        return (
            self.exact_dense_parent_positions
            and level == 1
            and self.level_types[0] == LevelType.DENSE
        )

    def emit_first_compressed_position_allocation(
        self,
        loop_bound: llir.Var,
        offsets: llir.Var,
    ) -> List[llir.Stmt]:
        """Build and initialize first-level position storage from exact offsets."""

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "first compressed position allocation requires an exact "
                "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        if (
            self.levels < 2
            or self.level_types[0] is not LevelType.DENSE
            or any(
                level_type is not LevelType.COMPRESSED
                for level_type in self.level_types[1:]
            )
        ):
            raise ScorchValueError(
                "first compressed position allocation requires one dense level "
                "followed by one or more compressed levels",
                code="result_assembly_allocation_mismatch",
            )

        if type(loop_bound) is not llir.Var:
            raise ScorchTypeError(
                "first compressed position bound must be an exact LLIR Var",
                code="internal_argument_type_invalid",
            )
        if set(vars(loop_bound)) != {
            "name",
            "type",
            "is_ptr",
            "is_restrict",
            "tensor_access",
        }:
            raise ScorchTypeError(
                "first compressed position bound must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            )
        try:
            bound_name = loop_bound.name
            bound_type = loop_bound.type
            bound_is_ptr = loop_bound.is_ptr
            bound_is_restrict = loop_bound.is_restrict
            bound_access = loop_bound.tensor_access
        except AttributeError as failure:
            raise ScorchTypeError(
                "first compressed position bound must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            ) from failure
        if type(bound_name) is not str or not bound_name.isidentifier():
            raise ScorchValueError(
                "first compressed position bound must be an identifier",
                code="result_assembly_allocation_mismatch",
            )
        if bound_type not in (llir.DataType.INT, llir.DataType.INT64):
            raise ScorchTypeError(
                "first compressed position bound must be INT or INT64",
                code="internal_argument_type_invalid",
            )
        if bound_is_ptr is not False:
            raise ScorchTypeError(
                "first compressed position bound cannot be a pointer",
                code="internal_argument_type_invalid",
            )
        if bound_is_restrict is not False:
            raise ScorchTypeError(
                "first compressed position bound cannot be restrict-qualified",
                code="internal_argument_type_invalid",
            )
        if bound_access is not None:
            raise ScorchTypeError(
                "first compressed position bound cannot carry tensor provenance",
                code="internal_argument_type_invalid",
            )

        if type(offsets) is not llir.Var:
            raise ScorchTypeError(
                "first compressed position offsets must be an exact LLIR Var",
                code="internal_argument_type_invalid",
            )
        if set(vars(offsets)) != {
            "name",
            "type",
            "is_ptr",
            "is_restrict",
            "tensor_access",
        }:
            raise ScorchTypeError(
                "first compressed position offsets must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            )
        try:
            offsets_name = offsets.name
            offsets_type = offsets.type
            offsets_is_ptr = offsets.is_ptr
            offsets_is_restrict = offsets.is_restrict
            offsets_access = offsets.tensor_access
        except AttributeError as failure:
            raise ScorchTypeError(
                "first compressed position offsets must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            ) from failure
        if offsets_name != "_offset1":
            raise ScorchValueError(
                "first compressed position offsets must be named _offset1",
                code="result_assembly_allocation_mismatch",
            )
        if offsets_type is not llir.DataType.STD_VECTOR_INT:
            raise ScorchTypeError(
                "first compressed position offsets must have STD_VECTOR_INT type",
                code="internal_argument_type_invalid",
            )
        if offsets_is_ptr is not False:
            raise ScorchTypeError(
                "first compressed position offsets cannot be a pointer",
                code="internal_argument_type_invalid",
            )
        if offsets_is_restrict is not False:
            raise ScorchTypeError(
                "first compressed position offsets cannot be restrict-qualified",
                code="internal_argument_type_invalid",
            )
        if offsets_access is not None:
            raise ScorchTypeError(
                "first compressed position offsets cannot carry tensor provenance",
                code="internal_argument_type_invalid",
            )

        owner_name = f"{self.name}1_pos_torch"
        pointer_name = f"{self.name}1_pos_data"

        def bound_reference() -> llir.Var:
            return llir.Var(name=bound_name, type=bound_type)

        def offsets_reference() -> llir.Var:
            return llir.Var(name=offsets_name, type=offsets_type)

        def owner_reference() -> llir.Var:
            return llir.Var(name=owner_name, type=llir.DataType.TORCH_TENSOR)

        def pointer_reference() -> llir.Var:
            return llir.Var(name=pointer_name, type=llir.DataType.PTR_INT)

        def index_reference() -> llir.Var:
            return llir.Var(name="_i", type=llir.DataType.INT)

        return [
            llir.VarInit(
                var=owner_reference(),
                value=llir.FunctionCall(
                    name="torch::empty",
                    args=(
                        llir.Array(
                            values=(
                                llir.Cast(
                                    expr=llir.Add(
                                        bound_reference(),
                                        llir.Literal(1, llir.DataType.INT),
                                    ),
                                    data_type=llir.DataType.INT64,
                                ),
                            ),
                            data_type=llir.DataType.INT64,
                        ),
                        llir.QualifiedName(
                            namespace="torch",
                            name="kInt",
                            data_type=llir.DataType.TORCH_SCALAR_TYPE,
                        ),
                    ),
                ),
            ),
            llir.VarInit(
                var=pointer_reference(),
                value=tensor_data_ptr(
                    owner_reference(),
                    llir.DataType.INT,
                ),
            ),
            llir.ForLoop(
                init=llir.VarInit(
                    var=index_reference(),
                    value=llir.Literal(0, llir.DataType.INT),
                ),
                cond=llir.BinOp(
                    "<=",
                    index_reference(),
                    bound_reference(),
                ),
                update=llir.Increment(index_reference()),
                body=[
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=pointer_reference(),
                            index=index_reference(),
                        ),
                        value=llir.Cast(
                            expr=llir.ArrayAccess(
                                array=offsets_reference(),
                                index=index_reference(),
                            ),
                            data_type=llir.DataType.INT,
                        ),
                    )
                ],
            ),
        ]

    def emit_compressed_coordinate_allocations(
        self,
        total_vars: Tuple[llir.Var, ...],
    ) -> List[llir.Stmt]:
        """Build coordinate buffers from exact typed compressed cardinalities."""

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "compressed coordinate allocations require an exact "
                "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        if (
            self.levels < 2
            or self.level_types[0] is not LevelType.DENSE
            or any(
                level_type is not LevelType.COMPRESSED
                for level_type in self.level_types[1:]
            )
        ):
            raise ScorchValueError(
                "compressed coordinate allocations require one dense level "
                "followed by one or more compressed levels",
                code="result_assembly_allocation_mismatch",
            )
        if type(total_vars) is not tuple:
            raise ScorchTypeError(
                "compressed coordinate totals must be an immutable Var tuple",
                code="internal_argument_type_invalid",
            )
        if len(total_vars) != self.levels - 1:
            raise ScorchValueError(
                "compressed coordinate totals must match the compressed levels",
                code="result_assembly_allocation_mismatch",
            )

        statements: List[llir.Stmt] = []
        for level, total_var in zip(range(1, self.levels), total_vars):
            if type(total_var) is not llir.Var:
                raise ScorchTypeError(
                    "compressed coordinate totals must contain exact LLIR Vars",
                    code="internal_argument_type_invalid",
                )
            try:
                total_name = total_var.name
                total_type = total_var.type
                total_is_ptr = total_var.is_ptr
                total_is_restrict = total_var.is_restrict
                total_access = total_var.tensor_access
            except AttributeError as failure:
                raise ScorchTypeError(
                    "compressed coordinate totals must contain complete LLIR Vars",
                    code="internal_argument_type_invalid",
                ) from failure
            if total_name != f"_total{level}":
                raise ScorchValueError(
                    "compressed coordinate total name must match its result level",
                    code="result_assembly_allocation_mismatch",
                )
            if total_type is not llir.DataType.INT64:
                raise ScorchTypeError(
                    "compressed coordinate totals must have INT64 type",
                    code="internal_argument_type_invalid",
                )
            if total_is_ptr is not False:
                raise ScorchTypeError(
                    "compressed coordinate totals cannot be pointers",
                    code="internal_argument_type_invalid",
                )
            if total_is_restrict is not False:
                raise ScorchTypeError(
                    "compressed coordinate totals cannot be restrict-qualified",
                    code="internal_argument_type_invalid",
                )
            if total_access is not None:
                raise ScorchTypeError(
                    "compressed coordinate totals cannot carry tensor provenance",
                    code="internal_argument_type_invalid",
                )

            owner_name = f"{self.name}{level}_crd_torch"
            statements.extend(
                (
                    llir.VarInit(
                        var=llir.Var(
                            name=owner_name,
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.FunctionCall(
                            name="torch::empty",
                            args=(
                                llir.Array(
                                    values=(
                                        llir.Var(
                                            name=total_name,
                                            type=total_type,
                                        ),
                                    ),
                                    data_type=llir.DataType.INT64,
                                ),
                                llir.QualifiedName(
                                    namespace="torch",
                                    name="kInt",
                                    data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                ),
                            ),
                        ),
                    ),
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd_data",
                            type=llir.DataType.PTR_INT,
                        ),
                        value=tensor_data_ptr(
                            llir.Var(
                                name=owner_name,
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            llir.DataType.INT,
                        ),
                    ),
                )
            )
        return statements

    def emit_deeper_compressed_position_allocations(
        self,
        total_vars: Tuple[llir.Var, ...],
    ) -> List[llir.Stmt]:
        """Build deeper position buffers from exact parent cardinalities."""

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "deeper compressed position allocations require an exact "
                "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        if (
            self.levels < 3
            or self.level_types[0] is not LevelType.DENSE
            or any(
                level_type is not LevelType.COMPRESSED
                for level_type in self.level_types[1:]
            )
        ):
            raise ScorchValueError(
                "deeper compressed position allocations require one dense level "
                "followed by two or more compressed levels",
                code="result_assembly_allocation_mismatch",
            )
        if type(total_vars) is not tuple:
            raise ScorchTypeError(
                "deeper compressed position totals must be an immutable Var tuple",
                code="internal_argument_type_invalid",
            )
        if len(total_vars) != self.levels - 1:
            raise ScorchValueError(
                "deeper compressed position totals must match the compressed levels",
                code="result_assembly_allocation_mismatch",
            )

        for level, total_var in zip(range(1, self.levels), total_vars):
            if type(total_var) is not llir.Var:
                raise ScorchTypeError(
                    "deeper compressed position totals must contain exact LLIR Vars",
                    code="internal_argument_type_invalid",
                )
            if set(vars(total_var)) != {
                "name",
                "type",
                "is_ptr",
                "is_restrict",
                "tensor_access",
            }:
                raise ScorchTypeError(
                    "deeper compressed position totals must contain complete LLIR Vars",
                    code="internal_argument_type_invalid",
                )
            try:
                total_name = total_var.name
                total_type = total_var.type
                total_is_ptr = total_var.is_ptr
                total_is_restrict = total_var.is_restrict
                total_access = total_var.tensor_access
            except AttributeError as failure:
                raise ScorchTypeError(
                    "deeper compressed position totals must contain complete LLIR Vars",
                    code="internal_argument_type_invalid",
                ) from failure
            if total_name != f"_total{level}":
                raise ScorchValueError(
                    "deeper compressed position total name must match its result level",
                    code="result_assembly_allocation_mismatch",
                )
            if total_type is not llir.DataType.INT64:
                raise ScorchTypeError(
                    "deeper compressed position totals must have INT64 type",
                    code="internal_argument_type_invalid",
                )
            if total_is_ptr is not False:
                raise ScorchTypeError(
                    "deeper compressed position totals cannot be pointers",
                    code="internal_argument_type_invalid",
                )
            if total_is_restrict is not False:
                raise ScorchTypeError(
                    "deeper compressed position totals cannot be restrict-qualified",
                    code="internal_argument_type_invalid",
                )
            if total_access is not None:
                raise ScorchTypeError(
                    "deeper compressed position totals cannot carry tensor provenance",
                    code="internal_argument_type_invalid",
                )

        statements: List[llir.Stmt] = []
        for level, parent_total in zip(range(2, self.levels), total_vars[:-1]):
            owner_name = f"{self.name}{level}_pos_torch"
            pointer_name = f"{self.name}{level}_pos_data"
            statements.extend(
                (
                    llir.VarInit(
                        var=llir.Var(
                            name=owner_name,
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.FunctionCall(
                            name="torch::empty",
                            args=(
                                llir.Array(
                                    values=(
                                        llir.Add(
                                            llir.Var(
                                                name=parent_total.name,
                                                type=parent_total.type,
                                            ),
                                            llir.Literal(1, llir.DataType.INT),
                                        ),
                                    ),
                                    data_type=llir.DataType.INT64,
                                ),
                                llir.QualifiedName(
                                    namespace="torch",
                                    name="kInt",
                                    data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                ),
                            ),
                        ),
                    ),
                    llir.VarInit(
                        var=llir.Var(
                            name=pointer_name,
                            type=llir.DataType.PTR_INT,
                        ),
                        value=tensor_data_ptr(
                            llir.Var(
                                name=owner_name,
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            llir.DataType.INT,
                        ),
                    ),
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=pointer_name,
                                type=llir.DataType.PTR_INT,
                            ),
                            index=llir.Literal(0, llir.DataType.INT),
                        ),
                        value=llir.Literal(0, llir.DataType.INT),
                    ),
                )
            )
        return statements

    def emit_compressed_value_allocation(
        self,
        total_var: llir.Var,
    ) -> List[llir.Stmt]:
        """Build Torch-owned values from one exact compressed cardinality."""

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "compressed value allocation requires an exact "
                "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        if (
            self.levels < 2
            or self.level_types[0] is not LevelType.DENSE
            or any(
                level_type is not LevelType.COMPRESSED
                for level_type in self.level_types[1:]
            )
        ):
            raise ScorchValueError(
                "compressed value allocation requires one dense level followed "
                "by one or more compressed levels",
                code="result_assembly_allocation_mismatch",
            )
        if type(total_var) is not llir.Var:
            raise ScorchTypeError(
                "compressed value total must be an exact LLIR Var",
                code="internal_argument_type_invalid",
            )
        if set(vars(total_var)) != {
            "name",
            "type",
            "is_ptr",
            "is_restrict",
            "tensor_access",
        }:
            raise ScorchTypeError(
                "compressed value total must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            )
        try:
            total_name = total_var.name
            total_type = total_var.type
            total_is_ptr = total_var.is_ptr
            total_is_restrict = total_var.is_restrict
            total_access = total_var.tensor_access
        except AttributeError as failure:
            raise ScorchTypeError(
                "compressed value total must be a complete LLIR Var",
                code="internal_argument_type_invalid",
            ) from failure
        expected_total_name = f"_total{self.levels - 1}"
        if type(total_name) is not str or total_name != expected_total_name:
            raise ScorchValueError(
                "compressed value total name must match the leaf result level",
                code="result_assembly_allocation_mismatch",
            )
        if total_type is not llir.DataType.INT64:
            raise ScorchTypeError(
                "compressed value total must have INT64 type",
                code="internal_argument_type_invalid",
            )
        if total_is_ptr is not False:
            raise ScorchTypeError(
                "compressed value total cannot be a pointer",
                code="internal_argument_type_invalid",
            )
        if total_is_restrict is not False:
            raise ScorchTypeError(
                "compressed value total cannot be restrict-qualified",
                code="internal_argument_type_invalid",
            )
        if total_access is not None:
            raise ScorchTypeError(
                "compressed value total cannot carry tensor provenance",
                code="internal_argument_type_invalid",
            )

        try:
            c_datatype = dtype_to_c_datatype(self.dtype)
            dtype_name = get_pytorch_c_dtype_name(self.dtype)
            pointer_type = llir.DataType.ptr_type(c_datatype)
        except (KeyError, ValueError) as failure:
            raise ScorchValueError(
                "compressed value allocation requires a supported result dtype",
                code="dtype_unsupported",
            ) from failure

        owner_name = f"{self.name}_values_torch"
        pointer_name = f"{self.name}_values_data"

        def total_reference() -> llir.Var:
            return llir.Var(name=total_name, type=total_type)

        def owner_reference() -> llir.Var:
            return llir.Var(name=owner_name, type=llir.DataType.TORCH_TENSOR)

        return [
            llir.VarInit(
                var=owner_reference(),
                value=llir.FunctionCall(
                    name="torch::empty",
                    args=(
                        llir.Array(
                            values=(total_reference(),),
                            data_type=llir.DataType.INT64,
                        ),
                        llir.QualifiedName(
                            namespace="torch",
                            name=dtype_name,
                            data_type=llir.DataType.TORCH_SCALAR_TYPE,
                        ),
                    ),
                ),
            ),
            llir.VarInit(
                var=llir.Var(name=pointer_name, type=pointer_type),
                value=tensor_data_ptr(owner_reference(), c_datatype),
            ),
        ]

    def emit_value_array_init(self) -> List[llir.Stmt]:
        """Emit Torch-owned known-size storage or a dynamic ``std::vector``."""
        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "value array initialization requires an exact " "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        stmts: List[llir.Stmt] = []
        if self.is_dense:
            # capacity = product of all dimension sizes; a rank-0 result has no
            # dimension and holds one value (the empty product).
            res_capacity_expr: llir.Expr = (
                llir.Var(
                    name=f"{self.name}0_size",
                    type=llir.DataType.INT64,
                )
                if self.levels
                else llir.Literal(value=1, data_type=llir.DataType.INT64)
            )
            for i in range(1, self.levels):
                res_capacity_expr = llir.BinOp(
                    left=res_capacity_expr,
                    op="*",
                    right=llir.Var(
                        name=f"{self.name}{i}_size",
                        type=llir.DataType.INT64,
                    ),
                )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_capacity",
                        type=llir.DataType.INT64,
                    ),
                    value=res_capacity_expr,
                )
            )

            c_datatype = dtype_to_c_datatype(self.dtype)
            res_capacity_var = llir.Var(
                name=f"{self.name}_capacity",
                type=llir.DataType.INT64,
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="torch::empty",
                        args=[
                            llir.Array(
                                values=(
                                    llir.Var(
                                        name=f"{self.name}_capacity",
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                data_type=llir.DataType.INT64,
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )
            # DELIBERATELY UNMARKED, and it is a vocabulary question rather than
            # an oversight.  This DECLARES ``{R}_values`` as a pointer into the
            # Torch tensor; it neither reads nor writes the storage's contents,
            # and its value expression names ``{R}_values_torch``, a different
            # thing.  ``ResultStorageArray``'s docstring excludes the two-phase
            # pass's ``_data`` pointers and the ``p{R}{L}`` cursor on the same
            # ground, so a marker here would claim an access the statement does
            # not make.  ``_complete_result_tile_impl`` hand-builds its
            # completion reference for this statement, so leaving it unmarked
            # also keeps the two sides in agreement without needing the
            # vocabulary widened first.
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.ptr_type(c_datatype),
                        is_restrict=True,
                    ),
                    value=tensor_data_ptr(
                        llir.Var(
                            name=f"{self.name}_values_torch",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        c_datatype,
                    ),
                )
            )

            # Zero the whole dense buffer before the parallel += accumulate. The
            # generated kernel accumulates into C, so it needs the full buffer
            # zeroed (not empty-rows-only like the prebuilt SpMM). scorch_zero_dense
            # (scorch/csrc/header.h) parallelizes that zero across cores for large
            # outputs — where the serial memset was a big fraction of runtime — and
            # falls back to a single memset below SCORCH_MEMSET_GRAIN_BYTES. Takes
            # the element count; it computes the byte span internally.
            #
            # MARKED, when this assembler was given an identity.  A free
            # function taking the result's value vector as an ARGUMENT is
            # exactly the shape the callee-name match cannot see -- review
            # section 61.4's gap A -- and the syntax cannot say the argument is
            # written, so the direction is declared here.  This is the second
            # instance of that shape in the tree; section 64.6 named only the
            # position sentinel.  ``_complete_result_tile_impl`` hand-builds the
            # completion reference for this statement, so both sides move
            # together, as they do for the sentinel.
            stmts.append(
                llir.FunctionCallStmt(
                    name="scorch_zero_dense",
                    args=[
                        llir.Var(
                            name=f"{self.name}_values",
                            type=llir.DataType.ptr_type(self.dtype),
                        ),
                        res_capacity_var,
                    ],
                    result_storage=self.result_storage_marker(
                        (llir.ResultStorageArray.VALUES, None, True)
                    ),
                )
            )
        elif self.known_nnz_var:
            c_datatype = dtype_to_c_datatype(self.dtype)
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="torch::empty",
                        args=[
                            llir.Array(
                                values=(
                                    llir.Var(
                                        name=self.known_nnz_var,
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                data_type=llir.DataType.INT64,
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.ptr_type(c_datatype),
                    ),
                    value=tensor_data_ptr(
                        llir.Var(
                            name=f"{self.name}_values_torch",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        c_datatype,
                    ),
                )
            )
        else:
            stmts.append(
                llir.VarDecl(
                    llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.std_vector_type(
                            dtype_to_c_datatype(self.dtype)
                        ),
                    )
                )
            )
            if self.reserve_hint_var:
                stmts.append(
                    llir.FunctionCallStmt(
                        name=f"{self.name}_values.reserve",
                        args=[
                            llir.Var(
                                name=self.reserve_hint_var,
                                type=llir.DataType.INT64,
                            )
                        ],
                    )
                )
        return stmts

    def emit_level_indices_init(self) -> List[llir.Stmt]:
        """Emit per-level index array initialization for COMPRESSED/COORDINATE levels."""
        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "level-index initialization requires an exact " "ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        stmts: List[llir.Stmt] = []
        for i, level_type in enumerate(self.level_types):
            if level_type == LevelType.COMPRESSED:
                if self._has_fixed_position_count(i):
                    # A dense parent fixes the exact position-array length.
                    # Size a standard vector once, then transfer it to Torch
                    # without a copy. Parent coordinates are ABI-validated
                    # against this same extent before the generated loop.
                    stmts.append(
                        llir.DirectInit(
                            var=llir.Var(
                                name=f"{self.name}{i}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            args=(
                                llir.Add(
                                    llir.Cast(
                                        expr=llir.Var(
                                            name=f"{self.name}{i - 1}_size",
                                            type=llir.DataType.INT64,
                                        ),
                                        data_type=llir.DataType.SIZE_T,
                                    ),
                                    llir.Literal(
                                        value=1,
                                        data_type=llir.DataType.INT,
                                    ),
                                ),
                                llir.Literal(
                                    value=0,
                                    data_type=llir.DataType.INT,
                                ),
                            ),
                        )
                    )
                else:
                    # Sparse parents have a dynamically assembled fiber count.
                    stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{self.name}{i}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            )
                        )
                    )
                stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{self.name}{i}_crd",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        )
                    )
                )
                if self.reserve_hint_var:
                    stmts.append(
                        llir.FunctionCallStmt(
                            name=f"{self.name}{i}_crd.reserve",
                            args=[
                                llir.Var(
                                    name=self.reserve_hint_var,
                                    type=llir.DataType.INT64,
                                )
                            ],
                        )
                    )
                if not self._has_fixed_position_count(i):
                    # pos[0] = 0 for the append-built position vector.
                    #
                    # MARKED, when this assembler was given an identity, and this
                    # is the statement that made the identity necessary.  The
                    # dynamic-vector pass rewrites this indexed store into
                    # ``scorch_vector_set(pos, 0, 0)`` and carries the marker
                    # across; the serial sparse-workspace families compare that
                    # rewrite against a completion reference they build
                    # themselves, so the reference must carry the same marker or
                    # the two sides differ and a program that compiles today
                    # fails.  Both sides move together: this one and
                    # ``lower_llir._completed_position_init_statement``.
                    stmts.append(
                        llir.Assign(
                            var=llir.ArrayAccess(
                                array=llir.Var(
                                    name=f"{self.name}{i}_pos",
                                    type=llir.DataType.STD_VECTOR_C_INT,
                                ),
                                index=llir.Literal(0),
                            ),
                            value=llir.Literal(0),
                            result_storage=self.result_storage_marker(
                                (llir.ResultStorageArray.POS, i, True)
                            ),
                        )
                    )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                # int <name><i>_pos_index = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"{self.name}{i}_pos_index",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

            elif level_type == LevelType.COORDINATE:
                if self.known_nnz_var:
                    # Known-size coordinate arrays are Torch-owned from creation.
                    stmts.extend(
                        [
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"{self.name}{i}_crd_torch",
                                    type=llir.DataType.TORCH_TENSOR,
                                ),
                                value=llir.FunctionCall(
                                    name="torch::empty",
                                    args=(
                                        llir.Array(
                                            values=(
                                                llir.Var(
                                                    name=self.known_nnz_var,
                                                    type=llir.DataType.INT64,
                                                ),
                                            ),
                                            data_type=llir.DataType.INT64,
                                        ),
                                        llir.QualifiedName(
                                            namespace="torch",
                                            name="kInt",
                                            data_type=(llir.DataType.TORCH_SCALAR_TYPE),
                                        ),
                                    ),
                                ),
                            ),
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"{self.name}{i}_crd",
                                    type=llir.DataType.PTR_INT,
                                ),
                                value=tensor_data_ptr(
                                    llir.Var(
                                        name=f"{self.name}{i}_crd_torch",
                                        type=llir.DataType.TORCH_TENSOR,
                                    ),
                                    llir.DataType.INT,
                                ),
                            ),
                        ]
                    )
                else:
                    stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{self.name}{i}_crd",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            )
                        )
                    )
                    if self.reserve_hint_var:
                        stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{self.name}{i}_crd.reserve",
                                args=[
                                    llir.Var(
                                        name=self.reserve_hint_var,
                                        type=llir.DataType.INT64,
                                    )
                                ],
                            )
                        )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

        return stmts

    def _get_mode_index_set(self, i: int, level_type: LevelType) -> llir.Array:
        """Return one fresh structured mode-index initializer for a level."""
        tensor_level_name = f"{self.name}{i}"
        if level_type == LevelType.DENSE:
            values: Tuple[llir.Expr, ...] = ()
        elif level_type == LevelType.COMPRESSED:
            values = (
                llir.Var(
                    name=f"{tensor_level_name}_pos_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
                llir.Var(
                    name=f"{tensor_level_name}_crd_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            )
        elif level_type == LevelType.COORDINATE:
            values = (
                llir.Var(
                    name=f"{tensor_level_name}_crd_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            )
        else:
            raise ScorchValueError(
                f"unsupported result level type {level_type}", code="level_type_unknown"
            )
        return llir.Array(
            values=values,
            data_type=llir.DataType.STD_VECTOR_TORCH_TENSOR,
        )

    def emit_result_declaration(self) -> llir.VarDecl:
        """Declare one fresh ABI result owner from this frozen snapshot."""

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "result declaration requires an exact ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        return llir.VarDecl(
            var=llir.Var(
                name=self.name,
                type=llir.DataType.TACO_TENSOR,
            )
        )

    def emit_storage_epilogue(self) -> List[llir.Stmt]:
        """Assign result storage and return without moving dynamic buffers.

        Every assembly path ends here, the compressed two-phase one included, so
        the value post-ops (``emit_value_post_ops``) are emitted here first.
        """

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "storage epilogue requires an exact ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        return [
            *self.emit_value_post_ops(),
            llir.Assign(
                var=tensor_storage_member(
                    self.name,
                    "storage",
                    "index",
                    "mode_indices",
                ),
                value=llir.Array(
                    values=tuple(
                        self._get_mode_index_set(i, level_type)
                        for i, level_type in enumerate(self.level_types)
                    ),
                    data_type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                ),
            ),
            llir.Assign(
                var=tensor_storage_member(self.name, "storage", "value"),
                value=llir.Var(
                    name=f"{self.name}_values_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            ),
            llir.Return(
                value=llir.Var(
                    name=self.name,
                    type=llir.DataType.TACO_TENSOR,
                )
            ),
        ]

    def emit_final_assembly(self) -> List[llir.Stmt]:
        """Move dynamic buffers to Torch, assign indices/values, and return."""
        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "final assembly requires an exact ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        stmts: List[llir.Stmt] = []

        # TacoTensor decl
        stmts.extend(
            [
                llir.Comment("Assemble final result"),
                self.emit_result_declaration(),
            ]
        )

        # Move dynamic index vectors into Torch storage contexts. Known-size
        # coordinate tensors were allocated before the compute loop.
        for i, level_type in enumerate(self.level_types):
            tensor_level_name = f"{self.name}{i}"

            if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                if level_type == LevelType.COMPRESSED:
                    stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{tensor_level_name}_pos_torch",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            value=llir.FunctionCall(
                                name="scorch_tensor_from_vector",
                                args=[
                                    llir.FunctionCall(
                                        name="std::move",
                                        args=[
                                            llir.Var(
                                                name=f"{tensor_level_name}_pos",
                                                type=llir.DataType.STD_VECTOR_C_INT,
                                            )
                                        ],
                                    ),
                                    llir.QualifiedName(
                                        namespace="torch",
                                        name="kInt",
                                        data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                    ),
                                ],
                            ),
                        )
                    )

                if self.known_nnz_var and level_type == LevelType.COORDINATE:
                    continue
                else:
                    stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{tensor_level_name}_crd_torch",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            value=llir.FunctionCall(
                                name="scorch_tensor_from_vector",
                                args=[
                                    llir.FunctionCall(
                                        name="std::move",
                                        args=[
                                            llir.Var(
                                                name=f"{tensor_level_name}_crd",
                                                type=llir.DataType.STD_VECTOR_C_INT,
                                            )
                                        ],
                                    ),
                                    llir.QualifiedName(
                                        namespace="torch",
                                        name="kInt",
                                        data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                    ),
                                ],
                            ),
                        )
                    )

        if not self.is_dense and not self.known_nnz_var:
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="scorch_tensor_from_vector",
                        args=[
                            llir.FunctionCall(
                                name="std::move",
                                args=[
                                    llir.Var(
                                        name=f"{self.name}_values",
                                        type=llir.DataType.std_vector_type(
                                            dtype_to_c_datatype(self.dtype)
                                        ),
                                    )
                                ],
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )

        stmts.extend(self.emit_storage_epilogue())

        return stmts

    def emit_position_tail_close(
        self, extent_names: Optional[Sequence[str]] = None
    ) -> List[llir.Stmt]:
        """Close the position entries no loop reached, for an append-built level.

        The first compressed level below one or more dense levels has one
        position entry per cell of those levels.  A loop over a dense level
        that visits only stored coordinates stops at the last one it has, and
        the catch-up that closes skipped cells runs only when a later cell is
        visited -- so the cells after the last visited one were never closed
        and the array ended short.  This closes them at the final coordinate
        count: they are empty.

        A pre-sized position array (one entry per cell, allocated before the
        loops) needs the same close: its unreached entries hold the zero it was
        allocated with, which leaves the array decreasing.  It is written in
        bounds by construction there, so the indexed store stays as it is; an
        append-built array has it rewritten into a checked set downstream.
        """

        if type(self) is not ResultTensorAssembler:
            raise ScorchTypeError(
                "position tail close requires an exact ResultTensorAssembler",
                code="internal_argument_type_invalid",
            )
        self.validate()
        level = next(
            (
                index
                for index, level_type in enumerate(self.level_types)
                if level_type != LevelType.DENSE
            ),
            0,
        )
        if (
            level == 0
            or self.level_types[level] != LevelType.COMPRESSED
            or self.known_nnz_var is not None
        ):
            return []
        # Each dense level's extent under the name its loop runs to, when the
        # caller knows one; the result's own extent variable otherwise.
        names = [f"{self.name}{above}_size" for above in range(level)]
        if extent_names is not None:
            if len(extent_names) < level or any(
                type(name) is not str or not name.isidentifier()
                for name in extent_names[:level]
            ):
                raise ScorchTypeError(
                    "position tail close extent names must be one identifier "
                    "per dense level above the compressed level",
                    code="internal_argument_type_invalid",
                )
            names = list(extent_names[:level])
        cells: llir.Expr = llir.Var(name=names[0], type=llir.DataType.INT64)
        for above in range(1, level):
            cells = llir.Mul(
                left=cells,
                right=llir.Var(name=names[above], type=llir.DataType.INT64),
            )
        pos_index = llir.Var(
            name=f"{self.name}{level}_pos_index", type=llir.DataType.INT64
        )
        # Spelled as the typed lowering spells the same close
        # (``_stored_prefix_final_statements``), so the two emit one text.
        return [
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            llir.ForLoop(
                init=None,
                cond=llir.BinOp(op="<", left=pos_index, right=cells),
                update=llir.Increment(var=pos_index),
                body=[
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{self.name}{level}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            index=llir.Add(pos_index, llir.Literal(1)),
                        ),
                        value=llir.FunctionCall(
                            name=f"{self.name}{level}_crd.size", args=[]
                        ),
                        result_storage=self.result_storage_marker(
                            (llir.ResultStorageArray.POS, level, True),
                            (llir.ResultStorageArray.CRD, level, False),
                        ),
                    )
                ],
            ),
        ]
