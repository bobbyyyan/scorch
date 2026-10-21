"""Normalized-CIN-to-LoopIR lowering for the migrated families.

This is the strangler-path entry that turns one normalized CIN program into a
verified production LoopIR program plus its verified ``LoopPlan``.  It covers
the coherent migrated families rather than hand-built fixtures:

- **Dense elementwise** — a pure ``ForAll`` nest over one plain
  ``TensorAssign`` (no update operator) whose right-hand side is a
  ``{ADD, SUB, MUL}`` expression tree over all-dense tensor accesses, with
  every loop variable free (bound on the left-hand side).
- **Dense reduction/matmul** — the same shape with an ``ADD`` update
  operator; loop variables absent from the left-hand side are reduction
  variables realized as ``StoreReduce`` (ADD) into the zero-initialized
  dense output.
- **Sparse level families (Phase 5)** — the same nest shapes over
  DENSE/COMPRESSED level compositions.  The pure iteration-domain analysis
  (:mod:`~scorch.compiler.loopir.iterdomain`) classifies every loop
  variable as a dense loop, a single sparse cursor loop, or a structured
  UNION/INTERSECTION merge; this module materializes that table as
  ``SparseFor``/``MergedSparseFor`` iteration with explicit
  parent-position-linked cursors, ``CursorValue`` leaf reads (UNION reads
  carry the explicit additive-identity default), and either dense stores or
  ordered ``AppendEntry`` assembly into a canonical CSR output.

Everything outside the families fails closed with
:class:`LoopIRLoweringError` and a stable code — nothing is silently
degraded to the legacy path from here.  The input CIN is never mutated;
stable ``SymbolId``/``IndexId`` identities flow through unchanged as
provenance, and each bound loop variable becomes one declared logical
dimension.

Recorded family boundaries (deliberate, fail-closed; see the Phase-4 and
Phase-7 reviews): one uniform scalar type (float32, float64, int64, int32,
int8 or uint8), no
workspaces/``Where``/derived index arithmetic/explicit parallel marks,
``ADD`` as the only update operator and ``{ADD, SUB, MUL}`` as the only
value operators, every tensor's storage-order loop variables in nest order,
and DENSE/COMPRESSED level types only.  Dense layouts may permute modes;
compressed structure remains identity-ordered.  Sparse outputs and merged
updates are admitted only through the explicitly classified CSR,
compressed-parent/dense-leaf, and dense-prefix/multi-compressed families;
every neighboring layout/domain combination fails closed rather than
falling through to generic assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple, cast

import torch

from ...format import LevelType
from ..identity import IndexId, SymbolId
from .. import cin as cin_nodes
from ..cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    Where,
    _is_index_stmt_instance,
)
from ..cin_analysis import verify_cin, verify_cin_structure
from ..loop_plan import LoopPlan, verify_loop_plan
from .build import LoopIRBuilder
from .iterdomain import (
    DomainKind,
    IterationDomainError,
    LoopIterationDomain,
    analyze_iteration_domains,
)
from .nodes import (
    BinaryOp,
    CursorId,
    DimensionId,
    Expr,
    LevelKind,
    LoopProgram,
    MergeMode,
    PositionId,
    ReduceOp,
    ScalarType,
    Stmt,
    TensorDecl,
    promoted_scalar_type,
)
from .verifier import verify_program

_CIN_TO_LOOPIR_BINARY: Dict[Operation, BinaryOp] = {
    Operation.ADD: BinaryOp.ADD,
    Operation.SUB: BinaryOp.SUB,
    Operation.MUL: BinaryOp.MUL,
}

_TORCH_TO_SCALAR: Dict[torch.dtype, ScalarType] = {
    torch.float32: ScalarType.FLOAT32,
    torch.float64: ScalarType.FLOAT64,
    torch.int64: ScalarType.INT64,
    torch.int32: ScalarType.INT32,
    torch.int8: ScalarType.INT8,
    torch.uint8: ScalarType.UINT8,
}


@dataclass(frozen=True)
class LoopIRLoweringDefect:
    """One immutable lowering failure: stable code and message."""

    code: str
    message: str


class LoopIRLoweringError(Exception):
    """Normalized CIN is outside the migrated dense LoopIR families."""

    def __init__(self, defect: LoopIRLoweringDefect) -> None:
        super().__init__(f"{defect.code}: {defect.message}")
        self.defect = defect


def _fail(code: str, message: str) -> NoReturn:
    raise LoopIRLoweringError(LoopIRLoweringDefect(code, message))


def supported_scalar_type(tensor: TensorVar) -> ScalarType:
    """The LoopIR scalar type of ``tensor``, or the ``unsupported_dtype`` refusal.

    The one place that decides whether an element type is inside the migrated
    families.  The lowering asks it for every tensor it declares, and the compile
    entry asks it for every bound operand before it serializes the request
    (``pipeline._compile_normalized_cin_via_loopir_owned``): the request
    serialization encodes only the element types named here, and reached first
    it reported an ``int64`` operand as a malformed request, with no code,
    rather than as a program outside the supported set.
    """

    scalar_type = _TORCH_TO_SCALAR.get(tensor.dtype)
    if scalar_type is None:
        _fail(
            "unsupported_dtype",
            f"tensor {tensor.name!r} dtype {tensor.dtype} is outside "
            "float32/float64/int64/int32/int8/uint8",
        )
    assert scalar_type is not None
    return scalar_type


@dataclass(frozen=True)
class LoopIRLoweringResult:
    """One verified LoopIR program plus its verified scheduling artifact.

    ``rhs_access_symbols`` preserves the right-hand-side access occurrence
    order (the order the legacy public entry binds runtime tensors in);
    ``input_symbols`` is the deduplicated declaration order.
    """

    program: LoopProgram
    loop_plan: LoopPlan
    loop_index_ids: Tuple[IndexId, ...]
    input_symbols: Tuple[SymbolId, ...]
    rhs_access_symbols: Tuple[SymbolId, ...]
    result_symbol: SymbolId


def _collect_loop_nest(cin: IndexStmt) -> Tuple[List[IndexVar], TensorAssign]:
    loop_vars: List[IndexVar] = []
    current: IndexStmt = cin
    while isinstance(current, ForAll):
        if current.parallel is True:
            _fail(
                "unsupported_explicit_parallel",
                "explicit parallel loop marks are a schedule decision this "
                "lowering does not consume",
            )
        loop_vars.append(current.index_var)
        current = current.stmt
    if isinstance(current, Where):
        _fail(
            "unsupported_statement",
            "Where/workspace programs are outside the dense LoopIR families",
        )
    if not isinstance(current, TensorAssign):
        _fail(
            "unsupported_statement",
            f"expected a TensorAssign at the nest leaf, got "
            f"{type(current).__name__}",
        )
    if not loop_vars:
        _fail(
            "unsupported_statement",
            "the dense families require at least one ForAll loop",
        )
    return loop_vars, current


def _nest_in_order(
    cin: IndexStmt, order: List[IndexVar], assign: TensorAssign
) -> IndexStmt:
    """``cin``'s perfect nest over the same assignment, with its loops in ``order``.

    A ForAll fixes no execution order, and every family here lowers a
    reduction as an ADD update (the reassociation contract the reorder pass
    relies on), so the permuted nest denotes the same program.  Each loop
    keeps its own parallel mark, and the caller's statement is not mutated.
    """

    parallel: Dict[IndexId, Optional[bool]] = {}
    current: IndexStmt = cin
    while isinstance(current, ForAll):
        parallel[current.index_var.index_id] = current.parallel
        current = current.stmt
    nested: IndexStmt = assign
    for index_var in reversed(order):
        nested = ForAll(index_var, nested, parallel=parallel[index_var.index_id])
    return nested


def _check_index_var(index_var: IndexVar) -> None:
    if index_var._expr is not None:
        _fail(
            "unsupported_index_expression",
            f"index variable {index_var.name!r} carries derived index "
            "arithmetic, which is outside the dense families",
        )


_LEVEL_TYPE_TO_KIND: Dict[LevelType, LevelKind] = {
    LevelType.DENSE: LevelKind.DENSE,
    LevelType.COMPRESSED: LevelKind.COMPRESSED,
    LevelType.COORDINATE: LevelKind.COORDINATE,
}


def _check_tensor(
    tensor: TensorVar,
) -> Tuple[ScalarType, Tuple[LevelType, ...], Tuple[int, ...]]:
    if isinstance(tensor, cin_nodes.Workspace):
        _fail(
            "unsupported_workspace",
            f"workspace {tensor.name!r} is outside the migrated families",
        )
    tensor_format = tensor.format
    if tensor_format is None:
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} declares no format",
        )
    level_types = tuple(tensor_format.get_level_types())
    if not level_types:
        # A rank-0 format is a format (``TensorVar(fmt="")``), and a scalar is a
        # valid tensor outside the migrated families.  Refused here so that it
        # is reported as that, and not by the LoopIR verifier as a malformed
        # ``TensorDecl`` this lowering built.
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} is rank 0; a scalar tensor stays outside "
            "the migrated DENSE/COMPRESSED families",
        )
    if any(level_type not in _LEVEL_TYPE_TO_KIND for level_type in level_types):
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} declares a level type outside the "
            "migrated DENSE/COMPRESSED/COORDINATE families",
        )
    if LevelType.COORDINATE in level_types and any(
        level_type is not LevelType.COORDINATE for level_type in level_types
    ):
        # A coordinate level stores one coordinate per entry and no position
        # array, so beside a level of another kind its first level has no
        # parent segment.  The legacy lowering refuses the same tensors
        # under the same code.
        _fail(
            "unsupported_mixed_coordinate_hierarchy",
            f"tensor {tensor.name!r} mixes a coordinate level with a level "
            "of another kind",
        )
    # Mixed dense-leaf RESULTS are the level-based B2 assembly family
    # (classified structurally below); mixed dense-leaf INPUTS lower to
    # declared physical position loads, so a dense value-bearing leaf
    # below compressed structure is admitted in both roles.
    mode_order = tensor.mode_order
    rank = len(level_types)
    if mode_order is None:
        storage_modes = tuple(range(rank))
    else:
        entries = list(mode_order)
        if (
            len(entries) != rank
            or any(type(entry) is not int for entry in entries)
            or sorted(entries) != list(range(rank))
        ):
            _fail(
                "unsupported_mode_order",
                f"tensor {tensor.name!r} declares a mode order that is not "
                f"a permutation of its rank-{rank} logical modes",
            )
        storage_modes = tuple(entries)
    if storage_modes != tuple(range(rank)) and any(
        level_type is not LevelType.DENSE for level_type in level_types
    ):
        # Permuted compressed structure changes the descent semantics, not
        # only which loop drives each level; it stays fail-closed until a
        # later slice represents it.
        _fail(
            "unsupported_mode_order",
            f"tensor {tensor.name!r} permutes compressed structure, which "
            "stays outside the migrated families",
        )
    return supported_scalar_type(tensor), level_types, storage_modes


def tensor_name_of(accesses: List[TensorAccess], symbol: SymbolId) -> str:
    """The display name of the tensor one of these accesses reads."""

    return next(
        access.tensor.name for access in accesses if access.tensor.symbol_id == symbol
    )


def _level_types_to_lower_on(
    level_types: Dict[SymbolId, Tuple[LevelType, ...]], result_symbol: SymbolId
) -> Dict[SymbolId, Tuple[LevelType, ...]]:
    """The level types a sparse-family program's loops are decided on.

    A result whose levels are all coordinate levels (COO) is assembled as its
    all-compressed twin is: one entry appended per stored cell, in
    lexicographic order of the storage levels.  What differs is what one
    append stores -- a coordinate on every level instead of a coordinate on
    the leaf and position ranges above it -- and that is the target's
    business.  The loops, the family and every ordering rule are therefore
    decided on the twin's level types; the result's declaration keeps the
    coordinate kind.
    """

    declared = level_types[result_symbol]
    if LevelType.COORDINATE not in declared:
        return level_types
    lowered = dict(level_types)
    lowered[result_symbol] = (LevelType.COMPRESSED,) * len(declared)
    return lowered


def input_symbols_of(rhs_accesses: List[TensorAccess]) -> List[SymbolId]:
    """Deduplicated input symbols in right-hand-side occurrence order."""

    symbols: List[SymbolId] = []
    for access in rhs_accesses:
        if access.tensor.symbol_id not in symbols:
            symbols.append(access.tensor.symbol_id)
    return symbols


def _collect_rhs_accesses(expr: object, accesses: List[TensorAccess]) -> None:
    if isinstance(expr, TensorAccess):
        accesses.append(expr)
        return
    if isinstance(expr, CINBinaryOp):
        if expr.op not in _CIN_TO_LOOPIR_BINARY:
            _fail(
                "unsupported_operation",
                f"binary operation {expr.op!r} is outside the dense families",
            )
        _collect_rhs_accesses(expr.left, accesses)
        _collect_rhs_accesses(expr.right, accesses)
        return
    _fail(
        "unsupported_expression",
        f"expression {type(expr).__name__} is outside the dense families",
    )


def _value_scalar_type(
    expr: object, scalar_types: Dict[SymbolId, ScalarType]
) -> Optional[ScalarType]:
    """The promoted type a right-hand side computes, or ``None`` when two of
    its operands promote to a type Scorch does not store."""

    if isinstance(expr, TensorAccess):
        return scalar_types[expr.tensor.symbol_id]
    assert isinstance(expr, CINBinaryOp)
    lhs = _value_scalar_type(expr.left, scalar_types)
    rhs = _value_scalar_type(expr.right, scalar_types)
    if lhs is None or rhs is None:
        return None
    return promoted_scalar_type(lhs, rhs)


def lower_normalized_cin_to_loopir(
    cin: IndexStmt,
    *,
    planned_loop_order: Optional[Tuple[IndexId, ...]] = None,
) -> LoopIRLoweringResult:
    """Lower one normalized dense-family CIN program to verified LoopIR.

    ``planned_loop_order`` is the scheduled path's verified plan order.  When
    it is given, the statement is re-nested in that order before anything
    reads its nest, so the base program's loops, the storage-order family
    check, the plan each family verifies and the iteration-domain analysis all
    see the plan order, and the reorder pass then applies the identity.  The
    written order carries no intent to preserve: the public einsum writes it
    by iterating a set, and a requested schedule's order is the plan.
    Building the base program in the written order instead refused every
    sparse program written in a nest that visits a compressed or result level
    out of storage order, before the reorder pass could apply a plan that
    verifies.  Dimensions are still declared in the written order, as the
    reorder pass leaves them, so a program the written nest already lowered
    keeps its declarations.  The target lowering re-enforces the same
    boundary against the program it actually emits.

    This boundary owns the implicit-reduction normalization: a public
    ``TensorAssign.op is None`` statement whose reduction loops are all
    proven additive by their right-hand-side uses lowers exactly as the
    explicit ADD update, and no later stage re-infers that fact.
    """

    if not _is_index_stmt_instance(cin):
        raise TypeError("lower_normalized_cin_to_loopir expects an IndexStmt")
    verify_cin_structure(cin)
    # Classify the nest shape first so out-of-family statements (Where,
    # workspaces, missing loops) report the family's stable codes; the full
    # CIN verifier then owns reference/ownership validity for family-shaped
    # programs.
    loop_vars, assign = _collect_loop_nest(cin)
    verify_cin(cin)
    for index_var in loop_vars:
        _check_index_var(index_var)
    loop_positions: Dict[IndexId, int] = {}
    for position, index_var in enumerate(loop_vars):
        loop_positions[index_var.index_id] = position
    # ``loop_vars`` keeps the written order, in which dimensions are declared;
    # ``nest_vars`` is the order the loops are nested in.
    nest_vars = loop_vars
    if planned_loop_order is not None:
        if set(planned_loop_order) != set(loop_positions) or len(
            planned_loop_order
        ) != len(loop_positions):
            _fail(
                "unsupported_loop_order",
                "the planned loop order must name every nest loop exactly " "once",
            )
        by_index = {index_var.index_id: index_var for index_var in loop_vars}
        nest_vars = [by_index[index_id] for index_id in planned_loop_order]
        if list(planned_loop_order) != list(loop_positions):
            # The plan each family verifies and the iteration-domain analysis
            # both read the nest from the statement, so it is re-nested too.
            cin = _nest_in_order(cin, nest_vars, assign)
    order_positions = {
        index_var.index_id: position for position, index_var in enumerate(nest_vars)
    }

    if assign.op is None:
        reduce_update = False
    elif assign.op is Operation.ADD:
        reduce_update = True
    else:
        _fail(
            "unsupported_update_op",
            f"update operator {assign.op!r} is outside the dense families",
        )

    lhs = assign.lhs
    if not isinstance(lhs, TensorAccess):
        _fail("unsupported_statement", "assignment left-hand side must be an access")
    rhs_accesses: List[TensorAccess] = []
    _collect_rhs_accesses(assign.rhs, rhs_accesses)

    all_accesses: List[TensorAccess] = [*rhs_accesses, lhs]
    for access in all_accesses:
        for index_var in access.indices:
            _check_index_var(index_var)
            if index_var.index_id not in loop_positions:
                _fail(
                    "unbound_access_index",
                    f"access index {index_var.name!r} is not a nest loop " "variable",
                )

    result_symbol = lhs.tensor.symbol_id
    if any(access.tensor.symbol_id == result_symbol for access in rhs_accesses):
        _fail(
            "unsupported_inplace_operand",
            f"result tensor {lhs.tensor.name!r} also appears as an operand",
        )
    rhs_symbols = [access.tensor.symbol_id for access in rhs_accesses]
    if len(set(rhs_symbols)) != len(rhs_symbols):
        _fail(
            "unsupported_repeated_operand",
            "one operand tensor appears in more than one right-hand-side "
            "access; the legacy kernel ABI binds one runtime argument per "
            "access, so this is outside the migrated families",
        )
    for access in all_accesses:
        if len(set(access.index_ids)) != len(access.index_ids):
            _fail(
                "unsupported_repeated_access_index",
                f"tensor {access.tensor.name!r} repeats a loop variable "
                "within one access; the legacy dense emission resolves such "
                "accesses at the wrong position, so they are outside the "
                "migrated families",
            )

    lhs_index_ids = tuple(lhs.index_ids)
    has_reduction_loops = any(
        index_id not in lhs_index_ids for index_id in loop_positions
    )
    if has_reduction_loops and not reduce_update:
        # The public frontend deliberately leaves ``TensorAssign.op`` unset
        # for reductions; legacy iteration analysis re-derives the additive
        # update from the assignment's right-hand-side-only index variables
        # (``iter_lattice.get_simplified_cin``) and emits the same C++ as an
        # explicit ADD update.  Normalize that proven fact exactly once at
        # this ownership boundary: every reduction loop variable must appear
        # in a right-hand-side access, which ``verify_cin`` already
        # guarantees for used bindings.  A loop variable outside both sides
        # would make the implied accumulation unprovable, so it fails closed
        # instead of manufacturing an update.
        rhs_index_ids = {
            index_id for access in rhs_accesses for index_id in access.index_ids
        }
        if any(
            index_id not in rhs_index_ids
            for index_id in loop_positions
            if index_id not in lhs_index_ids
        ):
            _fail(
                "unsupported_reduction_without_update",
                "a loop variable outside the left-hand side requires the ADD "
                "update operator or a right-hand-side use that proves the "
                "additive reduction",
            )
        reduce_update = True

    checked_tensors = {
        access.tensor.symbol_id: _check_tensor(access.tensor) for access in all_accesses
    }
    scalar_types = {
        symbol: scalar for symbol, (scalar, _, _) in checked_tensors.items()
    }
    level_types = {symbol: levels for symbol, (_, levels, _) in checked_tensors.items()}
    storage_modes = {symbol: modes for symbol, (_, _, modes) in checked_tensors.items()}
    if reduce_update and not has_reduction_loops:
        # An ADD update with no reduction loop accumulates nothing.  Every
        # loop variable binds a result coordinate, and an access may not
        # repeat one (refused above), so the nest visits each result cell at
        # most once, and every result starts empty (sparse) or zeroed (dense):
        # the update is plain assignment, normalized here once, at this
        # ownership boundary.  For a sparse result the operator would select
        # the accumulation families, whose workspaces and coordinate merges
        # exist to combine several contributions per cell and none of which
        # owns a copy.  For a dense result ``+=`` would read every zeroed cell
        # before writing it; legacy lowers the same program as ``=`` on both
        # automatic arms, and so does this.
        reduce_update = False
    value_dtype = _value_scalar_type(assign.rhs, scalar_types)
    if value_dtype is not scalar_types[result_symbol]:
        # Operands may hold different types: the value is computed in their
        # promoted type (PyTorch's rule, which C++ follows for every pair
        # Scorch stores), and the result must hold exactly that type, as
        # ``A + B`` declares it.  Neither a narrowing nor a widening store is
        # admitted.
        _fail(
            "mixed_dtype",
            "the right-hand side computes "
            f"{'no stored type' if value_dtype is None else value_dtype.value}, "
            f"but the result holds {scalar_types[result_symbol].value}",
        )

    # Every tensor's storage-order loop variables must appear in nest order
    # (the planned order on the scheduled path), the same dependency
    # direction the legacy dense position chains use.  Storage order is the
    # access's logical index list viewed through the tensor's mode order, so
    # a permuted dense layout is admitted exactly when its physical levels
    # are nest-consistent.
    for access in all_accesses:
        if all(
            level is LevelType.DENSE for level in level_types[access.tensor.symbol_id]
        ):
            # An all-dense tensor is addressed by its coordinates alone; the
            # target resolves each of its levels where the last coordinate
            # that level needs is bound, whatever the nest order.
            continue
        indices = list(access.indices)
        positions = [
            order_positions[indices[mode].index_id]
            for mode in storage_modes[access.tensor.symbol_id]
        ]
        if positions != sorted(positions):
            _fail(
                "unsupported_loop_order",
                f"tensor {access.tensor.name!r} storage order conflicts with "
                "the loop nest order",
            )

    if any(
        level_type is not LevelType.DENSE
        for levels in level_types.values()
        for level_type in levels
    ):
        return _lower_sparse_family(
            cin,
            nest_vars,
            loop_vars,
            assign,
            reduce_update,
            lhs,
            rhs_accesses,
            input_symbols_of(rhs_accesses),
            result_symbol,
            scalar_types,
            level_types,
            storage_modes,
        )

    builder = LoopIRBuilder()
    dimension_ids: Dict[IndexId, DimensionId] = {}
    dimension_decls = []
    for index_var in loop_vars:
        decl = builder.dimension(index_var.name)
        dimension_ids[index_var.index_id] = decl.dimension
        dimension_decls.append(decl)

    tensor_decls: Dict[SymbolId, TensorDecl] = {}
    input_symbols: List[SymbolId] = []

    def declare_tensor(access: TensorAccess) -> None:
        symbol = access.tensor.symbol_id
        if symbol in tensor_decls:
            return
        tensor_decls[symbol] = builder.tensor(
            symbol,
            access.tensor.name,
            scalar_types[symbol],
            tuple(dimension_ids[index_id] for index_id in access.index_ids),
            tuple(
                builder.level(LevelKind.DENSE, mode) for mode in storage_modes[symbol]
            ),
        )

    for access in rhs_accesses:
        declare_tensor(access)
        if access.tensor.symbol_id not in input_symbols:
            input_symbols.append(access.tensor.symbol_id)
    declare_tensor(lhs)

    def lower_expr(expr: object) -> Expr:
        if isinstance(expr, TensorAccess):
            return builder.load(
                expr.tensor.symbol_id,
                tuple(builder.index_value(index_id) for index_id in expr.index_ids),
            )
        assert isinstance(expr, CINBinaryOp)
        return builder.binary(
            _CIN_TO_LOOPIR_BINARY[expr.op],
            lower_expr(expr.left),
            lower_expr(expr.right),
        )

    value = lower_expr(assign.rhs)
    store_indices = tuple(builder.index_value(index_id) for index_id in lhs_index_ids)
    leaf: Stmt
    if reduce_update:
        leaf = builder.store_reduce(result_symbol, store_indices, ReduceOp.ADD, value)
    else:
        leaf = builder.store(result_symbol, store_indices, value)

    body = builder.block((leaf,))
    for index_var in reversed(nest_vars):
        body = builder.block(
            (
                builder.dense_for(
                    index_var.index_id,
                    dimension_ids[index_var.index_id],
                    body,
                ),
            )
        )

    ordered_tensor_decls = [
        tensor_decls[symbol] for symbol in [*input_symbols, result_symbol]
    ]
    program = builder.program(
        dimension_decls,
        ordered_tensor_decls,
        tuple(input_symbols),
        (result_symbol,),
        body,
    )
    verify_program(program)

    loop_index_ids = tuple(index_var.index_id for index_var in nest_vars)
    plan = verify_loop_plan(
        cin, LoopPlan(loop_order=loop_index_ids), dense_results_any_order=True
    )

    return LoopIRLoweringResult(
        program=program,
        loop_plan=plan,
        loop_index_ids=loop_index_ids,
        input_symbols=tuple(input_symbols),
        rhs_access_symbols=tuple(access.tensor.symbol_id for access in rhs_accesses),
        result_symbol=result_symbol,
    )


class _SparseOutputReduction(Enum):
    """Which semantic sparse-output reduction family one program selects.

    ``NONE`` covers the assembly families (no update operator).  Both
    reduction members lower to the coordinate-merged :class:`StoreReduce`
    leaf; only a sparse-workspace schedule rewrites that semantic form into
    ordered assembly, and target lowering refuses the unscheduled form.
    """

    NONE = "none"
    DOUBLY_COMPRESSED = "doubly_compressed"
    CSR_DENSE_ROW = "csr_dense_row"
    CSR_SPARSE_ROW = "csr_sparse_row"
    ORDERED_KEY_WORKSPACE = "ordered_key_workspace"
    BOUND_PREFIX_ACCUMULATION = "bound_prefix_accumulation"
    MULTI_COMPRESSED_ASSEMBLY = "multi_compressed_assembly"
    MULTI_COMPRESSED_UNION = "multi_compressed_union"


def _ordered_key_split(
    result_levels: Tuple[LevelType, ...],
    lhs_index_ids: Tuple[IndexId, ...],
    reduction_index_ids: Tuple[IndexId, ...],
    loop_positions: Dict[IndexId, int],
) -> Optional[Tuple[int, int]]:
    """The ``(prefix, key_rank)`` split of one ordered sparse reduction result.

    Derived from identities alone: the prefix is exactly the run of result
    coordinates bound above the outermost reduction, and the key is every
    result coordinate bound below it -- after the innermost reduction or
    between two reductions alike.  A key coordinate interleaved with the
    reductions is still owned by the one workspace region that starts at the
    outermost reduction: the region accumulates every contribution to a key
    before its drain appends it, and the drain orders keys itself, so the
    order in which the producer visits them does not matter.  Returns ``None``
    when the program is not an ordered reduction at all: no reduction, a
    result coordinate the nest does not bind, or result modes bound out of
    result level order.

    ``key_rank == 0`` is a legitimate split, not a degenerate one: every result
    coordinate is then bound above the outermost reduction, so the reduction is
    a scalar accumulation under an already-complete result prefix and no
    workspace region participates.  The two splits select different families,
    which is why the caller reads ``key_rank`` rather than treating a rank-0 key
    as an empty instance of a rank->=1 one.
    """

    if not reduction_index_ids or len(lhs_index_ids) != len(result_levels):
        return None
    positions = [loop_positions.get(index_id) for index_id in lhs_index_ids]
    if any(position is None for position in positions):
        return None
    ordered_positions = cast(List[int], positions)
    if any(
        earlier >= later
        for earlier, later in zip(ordered_positions, ordered_positions[1:])
    ):
        # Permuted result modes: the stored order would not be the drain
        # order, and no ordered assembly is defined for that here.
        return None
    reduction_positions = [loop_positions[index_id] for index_id in reduction_index_ids]
    first_reduction = min(reduction_positions)
    prefix = sum(1 for position in ordered_positions if position < first_reduction)
    return prefix, len(lhs_index_ids) - prefix


def _ordered_key_family(
    result_levels: Tuple[LevelType, ...],
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
    reduction_index_ids: Tuple[IndexId, ...],
    loop_positions: Dict[IndexId, int],
    dense_prefix_length: int,
) -> _SparseOutputReduction:
    """Decide an ordered sparse reduction by its split, or refuse with a code.

    The receiver is a run of ``dense_prefix_length`` dense levels followed by
    compressed levels, updated under ADD.  The split and the per-level domain
    rules below are the whole decision, so every receiver branch that decides
    an ordered reduction this way calls this one function rather than keeping
    its own copy of the rules.
    """

    # Two families share this receiver shape -- a dense prefix, then one
    # or more compressed levels, reduced under ADD -- and the split's key
    # rank selects between them.  The key is every result coordinate
    # bound below the outermost reduction, whether after the innermost
    # one or between two of them; every coordinate above the outermost
    # reduction binds a result prefix level directly.  Both runs are
    # derived from loop and level identities -- no layout string, rank
    # table, or format spelling participates.
    #
    # ``key_rank >= 1`` is the ordered-key sparse-workspace family: one
    # workspace region owns the key domain and drains it in order.
    # ``key_rank == 0`` is the bound-prefix accumulation family: the
    # prefix loops already visit result cells in lexicographic order (the
    # split's strictly-increasing test, plus the per-level domain rules
    # below), so the reduction accumulates into a scalar that the ordered
    # append reads and no workspace region exists to own.
    split = _ordered_key_split(
        result_levels, lhs_index_ids, reduction_index_ids, loop_positions
    )
    if split is None:
        _fail(
            "unsupported_sparse_output_domain",
            "an ordered sparse reduction needs every result coordinate "
            "bound by the nest, in result level order",
        )
    prefix, key_rank = cast(Tuple[int, int], split)
    # A drained level may be dense: the drain computes a dense level's position
    # from its parent's and grows the values with zeros over the cells no entry
    # names, so the target assembles any layout of drained levels.
    for position, index_id in enumerate(lhs_index_ids):
        domain_kind = domains[index_id].kind
        if position >= len(result_levels) - key_rank:
            # A drained key coordinate may iterate any domain: the
            # workspace merges and orders its entries, so the assembly
            # does not depend on the insertion order.
            continue
        if result_levels[position] is LevelType.DENSE:
            if domain_kind not in (
                DomainKind.DENSE,
                DomainKind.SPARSE,
                DomainKind.INTERSECTION,
            ):
                # A dense result level stores no coordinates, so the only
                # obligation a prefix loop owes it is that its child's
                # position array be closed at EVERY logical cell of the
                # level, not merely at the cells the loop visits.  A dense
                # domain visits them all and owes nothing further.  One
                # stored stream, or an intersection of two, visits a
                # monotone subsequence, and the target closes the cells it
                # skips with the row-scope catch-up, which advances by the
                # loop's coordinate -- the same obligation, and the same
                # mechanism, the canonical-CSR row-scope family already
                # discharges at rank 2.  A union is refused: its one-sided
                # cases would need the descent below to fill the subtree of
                # one operand alone.
                _fail(
                    "unsupported_sparse_output_domain",
                    "a dense result prefix level of an ordered-key "
                    "sparse reduction must iterate a dense domain, one "
                    "stored sparse level or an intersection",
                )
        elif domain_kind not in (
            DomainKind.DENSE,
            DomainKind.SPARSE,
            DomainKind.INTERSECTION,
        ):
            # A compressed prefix level appends a coordinate when its child
            # stored something, whatever drives its loop; an intersection
            # visits only coordinates both operands store, in order.
            _fail(
                "unsupported_sparse_output_domain",
                "a compressed result prefix level of an ordered-key "
                "sparse reduction must be driven by a dense domain, one "
                "stored sparse level or an intersection",
            )
    if key_rank == 0:
        # Every result coordinate binds a prefix level under the rules
        # just applied, so the whole result access is complete before the
        # outermost reduction opens.  That is the bound-prefix family; it
        # needs no workspace and therefore no workspace node.
        return _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION
    return _SparseOutputReduction.ORDERED_KEY_WORKSPACE


def _multi_compressed_assembly_family(
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
    prefix: int,
) -> _SparseOutputReduction:
    """Decide an ordered sparse assembly by its domains, or refuse with a code.

    The receiver is ``prefix`` dense levels over compressed levels, written
    without a reduction.  The per-level domain rules below are the whole
    decision, so every receiver branch that assembles this way calls this one
    function rather than keeping its own copy of the rules.
    """

    united = (DomainKind.UNION, DomainKind.DENSE_UNION)
    # A union may drive a dense prefix level only above a united compressed
    # suffix: the united assembly's alignment cases then descend into the
    # operands present at each visited cell, and the cells it skips are
    # caught up by coordinate like a stored stream's.
    suffix_united = len(lhs_index_ids) > prefix and all(
        domains[index_id].kind in united or domains[index_id].kind is DomainKind.DENSE
        for index_id in lhs_index_ids[prefix:]
    )
    suffix_kinds = []
    prefix_united = False
    for position, index_id in enumerate(lhs_index_ids):
        domain_kind = domains[index_id].kind
        if position < prefix:
            if domain_kind in united and suffix_united:
                prefix_united = True
                continue
            if domain_kind not in (
                DomainKind.DENSE,
                DomainKind.SPARSE,
                DomainKind.INTERSECTION,
            ):
                # A dense prefix level stores no coordinates; the only
                # obligation a stored stream driving it owes is that
                # the child position array be closed at every cell it
                # skips, which the target's row-scope catch-up
                # discharges -- the same rule, and the same mechanism,
                # as the ordered-key family's dense prefix above.  The
                # catch-up advances by the loop's coordinate, so an
                # intersection's agreed coordinate serves as well.  A
                # union does too, above a united suffix (admitted above);
                # above any other suffix it visits coordinates one operand
                # lacks with no one-sided descent, and is refused.
                _fail(
                    "unsupported_sparse_output_domain",
                    "the dense-prefix coordinates of a multi-"
                    "compressed output must iterate dense domains or "
                    "one stored sparse level in the migrated families",
                )
        else:
            suffix_kinds.append(domain_kind)
    if prefix_united or any(kind in united for kind in suffix_kinds):
        # A DENSE_UNION level is a union one of whose operands is dense
        # there: the united assembly follows it the same way, its dense
        # operand present at every coordinate.  A dense domain (every
        # operand dense there) is visited cell by cell: with both operands
        # present above every united level, and by a one-sided descent
        # through the operand present below one.
        if not all(kind in united or kind is DomainKind.DENSE for kind in suffix_kinds):
            # A union level cannot descend through a non-united
            # child stream: the one-sided branches would need a
            # child iterator the other operand does not carry.
            _fail(
                "unsupported_sparse_output",
                "a united multi-compressed output must unite every "
                "compressed-suffix coordinate in the migrated "
                "families",
            )
        return _SparseOutputReduction.MULTI_COMPRESSED_UNION
    for kind in suffix_kinds:
        if kind not in (
            DomainKind.DENSE,
            DomainKind.SPARSE,
            DomainKind.INTERSECTION,
        ):
            # A suffix coordinate iterates a dense domain, one stored
            # stream, or a two-stream intersection.  A dense domain
            # visits cells the operands may not store under; the
            # target guards such a level by its compressed child's
            # append, and a dense-domain leaf level appends every cell
            # because every operand is dense there and dense storage
            # holds every cell.  A union below a non-united level is
            # refused above.
            _fail(
                "unsupported_sparse_output",
                "a multi-compressed output is assembled only by "
                "dense domains, stored coordinate streams, or "
                "intersections in the migrated families",
            )
    return _SparseOutputReduction.MULTI_COMPRESSED_ASSEMBLY


def _dense_block_accumulation_family(
    result_levels: Tuple[LevelType, ...],
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
) -> _SparseOutputReduction:
    """Decide a reduction below a complete result prefix into a result with a
    dense level below a compressed one, or refuse with a code.

    Every result level is assembled by the loop that binds its coordinate, as
    in the assembly without a reduction; the leaf writes the scalar the
    reduction sub-nest accumulated.  A level's loop is a dense domain, one
    stored stream, or an intersection, which visits the coordinates every
    merged operand stores once each, in order, like one stored stream.  A
    union is refused with the domain code: its one-sided cases would need the
    reduction sub-nest below to iterate one operand alone.
    """

    del result_levels
    for index_id in lhs_index_ids:
        if domains[index_id].kind not in (
            DomainKind.DENSE,
            DomainKind.SPARSE,
            DomainKind.INTERSECTION,
        ):
            _fail(
                "unsupported_sparse_output_domain",
                "a result level above a reduction into a result with a dense "
                "level below a compressed level must iterate a dense domain, "
                "one stored sparse level or an intersection in the migrated "
                "families",
            )
    return _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION


def _dense_block_assembly_family(
    result_levels: Tuple[LevelType, ...],
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
) -> _SparseOutputReduction:
    """Decide an ordered assembly into a result with a dense level below a
    compressed one, or refuse with a code.

    Such a result stores a compressed coordinate exactly when the operands'
    stored structure under it is non-empty, and the dense levels below it hold
    every cell of their block: the cells no operand stores under are zeros, or
    empty segments of a compressed level further down.  The target commits
    each coordinate after its subtree only when the block was written, so any
    driver that visits a monotone subsequence of a dense level's cells is
    sound there, and a compressed level keeps the drivers the ordered families
    already admit.
    """

    prefix = 0
    while prefix < len(result_levels) and result_levels[prefix] is LevelType.DENSE:
        prefix += 1
    united = (DomainKind.UNION, DomainKind.DENSE_UNION)
    kinds = [domains[index_id].kind for index_id in lhs_index_ids]
    if any(kind in united for kind in kinds) and all(
        kind in united or kind is DomainKind.DENSE for kind in kinds
    ):
        # Every level united or dense: the united assembly's alignment cases descend
        # into the operands present at each coordinate, and a one-sided case
        # drains its operand's stream into the dense block the enclosing
        # compressed coordinate opened; the block commits as below any other
        # driver.
        return _SparseOutputReduction.MULTI_COMPRESSED_UNION
    for position, index_id in enumerate(lhs_index_ids):
        domain_kind = domains[index_id].kind
        if domain_kind is DomainKind.UNION:
            # A united level would need its one-sided descent to fill the
            # dense blocks below it; that drain is not defined here.  The
            # code is the one these programs exited with before this family
            # existed: the refusal is about the domain, not the layout.
            _fail(
                "unsupported_sparse_output_domain",
                "a united assembly into a result with a dense level below a "
                "compressed level is outside the migrated families",
            )
        # Every level, dense or compressed, inside a block or in the prefix
        # above the first compressed level, is assembled from the loop's one
        # coordinate: a dense level resolves its position from it and the
        # target catches up the cells it skips, a compressed level appends it.
        # An intersection's agreed coordinate serves exactly like one stored
        # stream's.
        allowed = (DomainKind.DENSE, DomainKind.SPARSE, DomainKind.INTERSECTION)
        if domain_kind not in allowed:
            _fail(
                "unsupported_sparse_output_domain",
                "a level of a result with a dense level below a compressed "
                "level must iterate a dense domain, one stored sparse level "
                "or an intersection in the migrated families",
            )
    return _SparseOutputReduction.MULTI_COMPRESSED_ASSEMBLY


def _classify_sparse_output_family(
    result_sparse: bool,
    reduce_update: bool,
    result_levels: Tuple[LevelType, ...],
    lhs: TensorAccess,
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
    reduction_index_ids: Tuple[IndexId, ...],
    loop_positions: Dict[IndexId, int],
) -> _SparseOutputReduction:
    """Gate the sparse-output families and select the reduction form."""

    sparse_reduction_family = (
        result_sparse
        and reduce_update
        and result_levels == (LevelType.COMPRESSED, LevelType.COMPRESSED)
    )
    dense_prefix_length = 0
    while (
        dense_prefix_length < len(result_levels)
        and result_levels[dense_prefix_length] is LevelType.DENSE
    ):
        dense_prefix_length += 1
    ordered_key_shape = (
        result_sparse
        and reduce_update
        and dense_prefix_length < len(result_levels)
        and all(
            level_type is LevelType.COMPRESSED
            for level_type in result_levels[dense_prefix_length:]
        )
        and result_levels != (LevelType.DENSE, LevelType.COMPRESSED)
    )
    if ordered_key_shape and sparse_reduction_family:
        # The doubly-compressed receiver keeps its own workspace family for a
        # drained key.  When every result coordinate is bound above the
        # outermost reduction there is no key to drain, and the program is the
        # bound-prefix accumulation every other receiver of this shape takes:
        # classify it by the same split rather than by its layout.
        split = _ordered_key_split(
            result_levels, lhs_index_ids, reduction_index_ids, loop_positions
        )
        ordered_key_shape = split is not None and split[1] == 0
        if ordered_key_shape:
            sparse_reduction_family = False
    if ordered_key_shape:
        return _ordered_key_family(
            result_levels,
            lhs_index_ids,
            domains,
            reduction_index_ids,
            loop_positions,
            dense_prefix_length,
        )
    mixed_leaf_family = (
        len(result_levels) >= 2
        and result_levels[0] is LevelType.COMPRESSED
        and all(level_type is LevelType.DENSE for level_type in result_levels[1:])
    )
    first_compressed = next(
        (
            level
            for level, level_type in enumerate(result_levels)
            if level_type is LevelType.COMPRESSED
        ),
        len(result_levels),
    )
    dense_below_compressed = any(
        level_type is LevelType.DENSE
        for level_type in result_levels[first_compressed + 1 :]
    )
    if (
        result_sparse
        and not sparse_reduction_family
        and not reduce_update
        and dense_below_compressed
        and not (
            mixed_leaf_family
            and all(
                domains[index_id].kind is DomainKind.DENSE for index_id in lhs_index_ids
            )
        )
    ):
        # Every layout with a dense level below a compressed one, assembled
        # without a reduction, except the all-dense-domain compressed-parent/
        # dense-leaf form, which keeps its own family below.
        return _dense_block_assembly_family(result_levels, lhs_index_ids, domains)
    if result_sparse and reduce_update and dense_below_compressed:
        split = _ordered_key_split(
            result_levels, lhs_index_ids, reduction_index_ids, loop_positions
        )
        if split is not None and split[1] == 0:
            # Every result coordinate is bound above the outermost reduction,
            # so the reduction is a scalar accumulated under a complete result
            # prefix: the ordered assembly of the layout, with the accumulator
            # written where a copy writes the operand's value.
            return _dense_block_accumulation_family(
                result_levels, lhs_index_ids, domains
            )
        if split is not None:
            # A key is bound below the outermost reduction: the ordered-key
            # workspace drains it, and the target assembles the prefix levels
            # with their dense blocks around the region.
            return _ordered_key_family(
                result_levels,
                lhs_index_ids,
                domains,
                reduction_index_ids,
                loop_positions,
                dense_prefix_length,
            )
    if result_sparse and not sparse_reduction_family:
        if mixed_leaf_family:
            # The B2 compressed-parent/dense-leaf assembly family: a dense
            # iteration domain appends one complete dense-suffix block per
            # materialized parent coordinate.  Its defective legacy
            # comparand produces malformed storage with values but no parent
            # coordinates, so this family is gated on the LoopIR oracle and
            # PyTorch differentials, never legacy parity.  The separately
            # retained sparse-reduction comparand owns the memory-safety
            # failure; it is not evidence about this dense-domain family.
            if reduce_update:
                _fail(
                    "unsupported_sparse_output_reduction",
                    "a compressed-parent/dense-leaf output cannot carry the "
                    "ADD update operator in the migrated families",
                )
            for index_id in lhs_index_ids:
                if domains[index_id].kind is not DomainKind.DENSE:
                    _fail(
                        "unsupported_sparse_output_domain",
                        "compressed-parent/dense-leaf assembly requires "
                        "every result coordinate to iterate a dense domain "
                        "in the migrated families",
                    )
            return _SparseOutputReduction.NONE
        compressed_suffix = 0
        while (
            compressed_suffix < len(result_levels)
            and result_levels[-1 - compressed_suffix] is LevelType.COMPRESSED
        ):
            compressed_suffix += 1
        # An ordered compressed suffix carries two or more compressed levels,
        # or -- degenerately -- is the whole rank-1 result (one stored stream
        # with no dense prefix and no parent level to close), or sits below
        # two or more dense parents.  A single compressed level below exactly
        # one dense parent is canonical CSR, which keeps its own dedicated
        # family below.
        #
        # The multiple-dense-parent form is what makes the flattened prefix
        # explicit: the first compressed level's segment count is the product
        # of every dense extent above it, not one loop's extent.
        ordered_compressed_suffix = (
            compressed_suffix >= 2
            or len(result_levels) == compressed_suffix == 1
            or (compressed_suffix == 1 and len(result_levels) - compressed_suffix >= 2)
        )
        multi_compressed_family = ordered_compressed_suffix and all(
            level_type is LevelType.DENSE
            for level_type in result_levels[: len(result_levels) - compressed_suffix]
        )
        if multi_compressed_family and not reduce_update:
            # Ordered sparse assembly: either a dense prefix over a >=2-level
            # compressed suffix, the rank-1 compressed stream, or several
            # dense parents over one compressed level.  Each suffix coordinate
            # iterates one single-cursor stored stream, one two-cursor
            # intersection, or — when every suffix coordinate is united — one
            # two-cursor ordered union with one-sided descent and tails.  The
            # legacy comparands are honest for the first two forms, which gate
            # on byte parity.  The several-dense-parent comparand closes by
            # only one coordinate and is malformed, so that form gates on the
            # LoopIR oracle and compiled PyTorch differentials instead.
            return _multi_compressed_assembly_family(
                lhs_index_ids,
                domains,
                len(result_levels) - compressed_suffix,
            )
        if result_levels != (LevelType.DENSE, LevelType.COMPRESSED):
            _fail(
                "unsupported_sparse_output",
                f"result {lhs.tensor.name!r} declares a sparse layout other "
                "than canonical CSR, compressed-parent/dense-leaf, or a "
                "dense-prefix multi-compressed suffix; ordered assembly is "
                "defined for those layouts only in the migrated families",
            )
        if reduce_update:
            # The parallel CSR-reduction family (Phase 7): a dense row
            # domain, plain stored-sparse reduction domains that all
            # enclose the trailing column loop, and a stored sparse column
            # domain select the semantic accumulation form.  The sparse-row
            # row-scope shape (a stored-sparse row domain over the same
            # doubly-compressed merge shape B1 admits) selects the
            # row-scope form: its defective legacy comparand sizes the
            # result rows by the first operand's stored-row count, so the
            # typed route is proven against the oracle and PyTorch, never
            # byte parity.  Everything else — merged or dense reduction or
            # column domains, and trailing-level reductions below the
            # column coordinate — keeps the historical seam code.
            column_domain_kind = domains[lhs_index_ids[1]].kind
            if (
                domains[lhs_index_ids[0]].kind is DomainKind.SPARSE
                and column_domain_kind is not DomainKind.DENSE
                and not (
                    column_domain_kind in (DomainKind.INTERSECTION, DomainKind.UNION)
                    and all(
                        loop_positions[index_id] > loop_positions[lhs_index_ids[1]]
                        for index_id in reduction_index_ids
                    )
                )
            ):
                # The row-scope workspace drains the column as its key, so a
                # merged column bound above every reduction is not its shape:
                # each stored (row, column) pair is then one scalar
                # accumulation, which the ordered split below selects.
                return _SparseOutputReduction.CSR_SPARSE_ROW
            column_position = loop_positions[lhs_index_ids[-1]]
            if (
                domains[lhs_index_ids[0]].kind is not DomainKind.DENSE
                or domains[lhs_index_ids[1]].kind is not DomainKind.SPARSE
                or any(
                    domains[index_id].kind is not DomainKind.SPARSE
                    for index_id in reduction_index_ids
                )
                or any(
                    loop_positions[index_id] > column_position
                    for index_id in reduction_index_ids
                )
            ):
                # Not the row-workspace shape.  The receiver is still a dense
                # prefix over compressed levels, so the ordered split decides
                # it as it decides every other such receiver: every result
                # coordinate bound above the outermost reduction is a scalar
                # accumulated under a complete prefix, and a coordinate bound
                # below one is a key the ordered workspace drains.
                return _ordered_key_family(
                    result_levels,
                    lhs_index_ids,
                    domains,
                    reduction_index_ids,
                    loop_positions,
                    dense_prefix_length,
                )
            return _SparseOutputReduction.CSR_DENSE_ROW
        row_domain = domains[lhs_index_ids[0]]
        column_domain = domains[lhs_index_ids[1]]
        if not (
            row_domain.kind is DomainKind.DENSE
            and column_domain.kind in (DomainKind.INTERSECTION, DomainKind.UNION)
        ):
            # Every CSR assembly but a dense row over a merged column is the
            # multi-compressed family at rank 2 (the target-side twin is
            # ``_canonical_csr_copy_leaves_its_route``): a stored or
            # intersected row is caught up, a stored column appends what it
            # visits, a dense column appends every cell.
            return _multi_compressed_assembly_family(lhs_index_ids, domains, 1)
        if row_domain.kind is not DomainKind.DENSE:
            _fail(
                "unsupported_sparse_output_domain",
                "the CSR row coordinate must iterate a dense domain in the "
                "migrated families",
            )
        if column_domain.kind is DomainKind.DENSE:
            _fail(
                "unsupported_sparse_output_domain",
                "the CSR column coordinate must be driven by stored sparse "
                "coordinates; dense-domain assembly is outside the migrated "
                "families",
            )
    elif sparse_reduction_family:
        # The doubly-compressed reduction family lowers to the SEMANTIC
        # accumulation form (a coordinate-merged StoreReduce); only a
        # sparse-workspace schedule can rewrite it into ordered assembly,
        # and target lowering refuses the unscheduled form.
        row_domain = domains[lhs_index_ids[0]]
        column_domain = domains[lhs_index_ids[1]]
        if row_domain.kind not in (
            DomainKind.DENSE,
            DomainKind.SPARSE,
            DomainKind.INTERSECTION,
        ):
            # An intersection row visits the rows both operands store, once
            # each, in order; the rank-general region target assembles it
            # like a stored row (the row-product target takes a stored row
            # only, and the router passes such a program on).
            _fail(
                "unsupported_sparse_output_domain",
                "the doubly-compressed row coordinate must be driven by a "
                "dense domain, one stored sparse level or an intersection in "
                "the migrated families",
            )
    return (
        _SparseOutputReduction.DOUBLY_COMPRESSED
        if sparse_reduction_family
        else _SparseOutputReduction.NONE
    )


def _allocate_cursor_identities(
    builder: LoopIRBuilder,
    domains: Sequence[LoopIterationDomain],
    level_types: Dict[SymbolId, Tuple[LevelType, ...]],
    result_sparse: bool,
    *,
    union_binds_positions: bool,
    intersection_binds_positions: bool,
) -> Tuple[
    Dict[Tuple[SymbolId, int], CursorId],
    Dict[Tuple[SymbolId, int], PositionId],
    set,
]:
    """Cursor and bound-position identities for every iterated compressed level.

    Returns the cursor of each (tensor, level), the position its loop binds
    where one is needed, and the (tensor, level) keys a union iterates (their
    reads take an explicit additive-identity default).
    """

    cursor_ids: Dict[Tuple[SymbolId, int], CursorId] = {}
    position_ids: Dict[Tuple[SymbolId, int], PositionId] = {}
    union_cursors: set = set()
    for domain in domains:
        for ref in domain.cursors:
            key = (ref.tensor, ref.level)
            if key not in cursor_ids:
                cursor_ids[key] = builder.new_cursor_id()
            if domain.kind is DomainKind.SPARSE:
                position_ids[key] = builder.new_position_id()
            elif domain.kind in (DomainKind.UNION, DomainKind.DENSE_UNION):
                union_cursors.add(key)
                if union_binds_positions or (
                    not result_sparse
                    and any(
                        ref_below.level < len(level_types[ref_below.tensor]) - 1
                        for ref_below in domain.cursors
                    )
                ):
                    # Ordered union descent: the merged loop binds every
                    # cursor's position; a child stream chained from a
                    # cursor that did not align at the merged coordinate
                    # is empty in that iteration.  A dense result descends
                    # this way exactly when some merged tensor has a level
                    # below the merged one.
                    position_ids[key] = builder.new_position_id()
            elif domain.kind is DomainKind.INTERSECTION and (
                intersection_binds_positions
                or (
                    not result_sparse
                    and any(
                        ref_below.level < len(level_types[ref_below.tensor]) - 1
                        for ref_below in domain.cursors
                    )
                )
            ):
                # INTERSECTION descent: the merged loop binds each aligned
                # cursor's position so child levels can chain from it.  For
                # a dense result this is needed exactly when some merged
                # tensor has a level below the merged one; an innermost
                # intersection keeps its position-free merge.
                position_ids[key] = builder.new_position_id()
    return cursor_ids, position_ids, union_cursors


def _classify_with_dense_union(
    table_domains: Sequence[LoopIterationDomain],
    result_sparse: bool,
    reduce_update: bool,
    result_levels: Tuple[LevelType, ...],
    lhs: TensorAccess,
    lhs_index_ids: Tuple[IndexId, ...],
    domains: Dict[IndexId, LoopIterationDomain],
    reduction_index_ids: Tuple[IndexId, ...],
    loop_positions: Dict[IndexId, int],
) -> _SparseOutputReduction:
    """``_classify_sparse_output_family``, with one cause named for dense unions.

    A sum with a term present at every coordinate of a dimension, into a
    sparse result, is lowered by the united assembly (an elementwise sum into
    united or dense levels); the other sparse result families -- reductions,
    CSR-shaped prefixes -- do not take it, and their own refusal would name a
    narrower cause than this one.
    """

    dense_union = any(domain.kind is DomainKind.DENSE_UNION for domain in table_domains)
    try:
        reduction_form = _classify_sparse_output_family(
            result_sparse,
            reduce_update,
            result_levels,
            lhs,
            lhs_index_ids,
            domains,
            reduction_index_ids,
            loop_positions,
        )
    except LoopIRLoweringError:
        if not (result_sparse and dense_union):
            raise
        reduction_form = None
    if (
        result_sparse
        and dense_union
        and reduction_form is not _SparseOutputReduction.MULTI_COMPRESSED_UNION
    ):
        _fail(
            "unsupported_union_with_dense",
            "a union of a sparse domain with a dense operand into a sparse "
            "result is lowered only as an elementwise sum in the migrated "
            "families",
        )
    assert reduction_form is not None
    return reduction_form


def _lower_sparse_family(
    cin: IndexStmt,
    loop_vars: List[IndexVar],
    dimension_vars: List[IndexVar],
    assign: TensorAssign,
    reduce_update: bool,
    lhs: TensorAccess,
    rhs_accesses: List[TensorAccess],
    input_symbols: List[SymbolId],
    result_symbol: SymbolId,
    scalar_types: Dict[SymbolId, ScalarType],
    level_types: Dict[SymbolId, Tuple[LevelType, ...]],
    storage_modes: Dict[SymbolId, Tuple[int, ...]],
) -> LoopIRLoweringResult:
    """Materialize the sparse level families from the pure domain analysis.

    ``loop_vars`` is the order the loops are nested in; ``dimension_vars``
    the written order, in which dimensions are declared.
    """

    loop_index_ids = tuple(index_var.index_id for index_var in loop_vars)
    plan = verify_loop_plan(
        cin, LoopPlan(loop_order=loop_index_ids), dense_results_any_order=True
    )
    try:
        table = analyze_iteration_domains(cin, plan)
    except IterationDomainError as error:
        raise LoopIRLoweringError(
            LoopIRLoweringDefect(error.defect.code, error.defect.message)
        ) from error
    domains = {domain.index: domain for domain in table.domains}

    declared_result_levels = level_types[result_symbol]
    level_types = _level_types_to_lower_on(level_types, result_symbol)
    result_levels = level_types[result_symbol]
    result_sparse = any(
        level_type is LevelType.COMPRESSED for level_type in result_levels
    )
    lhs_index_ids = tuple(lhs.index_ids)
    reduction_form = _classify_with_dense_union(
        table.domains,
        result_sparse,
        reduce_update,
        result_levels,
        lhs,
        lhs_index_ids,
        domains,
        tuple(index_id for index_id in loop_index_ids if index_id not in lhs_index_ids),
        {index_id: position for position, index_id in enumerate(loop_index_ids)},
    )
    sparse_reduction_family = reduction_form in (
        _SparseOutputReduction.DOUBLY_COMPRESSED,
        _SparseOutputReduction.CSR_DENSE_ROW,
        _SparseOutputReduction.CSR_SPARSE_ROW,
        _SparseOutputReduction.ORDERED_KEY_WORKSPACE,
        # The bound-prefix family reduces into sparse result storage exactly
        # like the others, so it lowers to the same semantic StoreReduce leaf.
        # Unlike the ordered-key family, no schedule pass rewrites it: the
        # owning target reads the accumulation directly.
        _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION,
    )
    multi_compressed_assembly = (
        reduction_form is _SparseOutputReduction.MULTI_COMPRESSED_ASSEMBLY
    )
    multi_compressed_union = (
        reduction_form is _SparseOutputReduction.MULTI_COMPRESSED_UNION
    )

    for index_id in loop_index_ids:
        domain = domains[index_id]
        if domain.kind in (DomainKind.UNION, DomainKind.INTERSECTION):
            if not result_sparse:
                # A dense result addresses every cell by its coordinates, so a
                # merge may bind a result coordinate or a summed one, under
                # ADD or not; the generic target decides which nests it emits
                # (``_dense_result_merged_nest``) and refuses the rest with a
                # structured code.
                continue
            if index_id not in lhs_index_ids:
                if reduction_form in (
                    _SparseOutputReduction.DOUBLY_COMPRESSED,
                    _SparseOutputReduction.CSR_SPARSE_ROW,
                    _SparseOutputReduction.ORDERED_KEY_WORKSPACE,
                    # The bound-prefix family accumulates its reduction
                    # sub-nest into one scalar per result cell, whatever
                    # domain each reduction loop iterates.
                    _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION,
                ):
                    # The B1, row-scope, and ordered-key families reduce over
                    # the merged reduction dimension by construction; the
                    # semantic StoreReduce form is exactly that reduction.
                    continue
                _fail(
                    "unsupported_merged_reduction",
                    "reducing over a merged sparse domain is outside the "
                    "migrated families",
                )
            if reduce_update:
                if (
                    reduction_form
                    in (
                        _SparseOutputReduction.ORDERED_KEY_WORKSPACE,
                        _SparseOutputReduction.DOUBLY_COMPRESSED,
                        _SparseOutputReduction.CSR_SPARSE_ROW,
                    )
                    and index_id in lhs_index_ids
                    and loop_index_ids.index(index_id)
                    > min(
                        (
                            loop_index_ids.index(reduction)
                            for reduction in loop_index_ids
                            if reduction not in lhs_index_ids
                        ),
                        default=len(loop_index_ids),
                    )
                ):
                    # A merged KEY coordinate (one bound below the outermost
                    # reduction) is ordinary in every family a workspace
                    # drains: the workspace owns ordering, so a coordinate
                    # produced by a merge is inserted exactly like one
                    # produced by a single cursor.
                    continue
                if (
                    reduction_form
                    in (
                        _SparseOutputReduction.ORDERED_KEY_WORKSPACE,
                        _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION,
                        _SparseOutputReduction.DOUBLY_COMPRESSED,
                    )
                    and index_id in lhs_index_ids
                    and domain.kind is DomainKind.INTERSECTION
                ):
                    # An intersection binding a result prefix level visits
                    # each coordinate both operands store once, in order, so
                    # the level is assembled exactly as under one stored
                    # stream (``_ordered_key_family`` admits the domain).
                    continue
                _fail(
                    "unsupported_merged_update",
                    "combining a merged sparse domain with the ADD update "
                    "operator is outside the migrated families",
                )

    builder = LoopIRBuilder()
    dimension_ids: Dict[IndexId, DimensionId] = {}
    dimension_decls = []
    for index_var in dimension_vars:
        decl = builder.dimension(index_var.name)
        dimension_ids[index_var.index_id] = decl.dimension
        dimension_decls.append(decl)

    tensor_decls: Dict[SymbolId, TensorDecl] = {}
    tensor_accesses: Dict[SymbolId, TensorAccess] = {}

    def declare_tensor(access: TensorAccess) -> None:
        symbol = access.tensor.symbol_id
        if symbol in tensor_decls:
            return
        tensor_accesses[symbol] = access
        tensor_decls[symbol] = builder.tensor(
            symbol,
            access.tensor.name,
            scalar_types[symbol],
            tuple(dimension_ids[index_id] for index_id in access.index_ids),
            tuple(
                builder.level(_LEVEL_TYPE_TO_KIND[level_type], mode)
                for mode, level_type in zip(
                    storage_modes[symbol],
                    (
                        declared_result_levels
                        if symbol == result_symbol
                        else level_types[symbol]
                    ),
                )
            ),
        )

    for access in rhs_accesses:
        declare_tensor(access)
    declare_tensor(lhs)

    # Allocate cursor and bound-position identities for every compressed
    # level the domain table iterates; UNION-merged cursors read through an
    # explicit additive-identity default.
    cursor_ids, position_ids, union_cursors = _allocate_cursor_identities(
        builder,
        table.domains,
        level_types,
        result_sparse,
        union_binds_positions=multi_compressed_union,
        intersection_binds_positions=(
            sparse_reduction_family or multi_compressed_assembly
        ),
    )

    def position_expr(
        symbol: SymbolId, level: int, absence_handled: bool = False
    ) -> Expr:
        """The dominating physical position of one tensor level.

        ``absence_handled`` is set where the consumer of the position handles a
        union-bound position that is absent: the address of a load guarded by
        the cursor of the tensor's deepest compressed level (see
        ``guarded_leaf_cursor``), and the parent of a compressed level's
        cursor, whose segment is empty under an absent parent.  The dense
        spine may then hang from a union-bound position.
        """

        if level < 0:
            return builder.root_position()
        kind = _LEVEL_TYPE_TO_KIND[level_types[symbol][level]]
        if kind is LevelKind.DENSE:
            if not absence_handled and any(
                (symbol, above) in union_cursors
                for above in range(level)
                if level_types[symbol][above] is LevelType.COMPRESSED
            ):
                # A united cursor's bound position may be absent (its
                # operand stores nothing at the merged coordinate), so the
                # arithmetic of a dense level below it is meaningful only
                # to a consumer that handles the absence.
                _fail(
                    "unsupported_union_dense_descent",
                    f"tensor {tensor_decls[symbol].name!r} stores dense level "
                    f"{level} below a level iterated by a union; reading it "
                    "would need the union's alignment as a guard",
                )
            driving = tensor_accesses[symbol].index_ids[storage_modes[symbol][level]]
            return builder.dense_position(
                symbol,
                level,
                position_expr(symbol, level - 1, absence_handled),
                builder.index_value(driving),
            )
        bound = position_ids.get((symbol, level))
        if bound is None:
            _fail(
                "unsupported_sparse_hierarchy",
                f"tensor {tensor_decls[symbol].name!r} needs the bound "
                f"position of compressed level {level}, which only a "
                "single-cursor sparse loop binds; hierarchical merge "
                "descent is outside the migrated families",
            )
        return builder.position_value(bound)

    def guarded_leaf_cursor(symbol: SymbolId) -> Optional[CursorId]:
        """The cursor that guards a dense leaf's load, if one is needed.

        A dense leaf below a compressed level that a union iterates has an
        address only where that union's cursor aligned.  The guard is the
        cursor of the deepest compressed level, every level below it dense;
        where it aligned, every united level above it aligned too (a child
        stream under an unaligned cursor is empty).
        """

        compressed = [
            level
            for level, level_type in enumerate(level_types[symbol])
            if level_type is LevelType.COMPRESSED
        ]
        if not compressed or (symbol, compressed[-1]) not in union_cursors:
            return None
        return cursor_ids[(symbol, compressed[-1])]

    def lower_expr(expr: object) -> Expr:
        if isinstance(expr, TensorAccess):
            symbol = expr.tensor.symbol_id
            if all(level_type is LevelType.DENSE for level_type in level_types[symbol]):
                return builder.load(
                    symbol,
                    tuple(builder.index_value(index_id) for index_id in expr.index_ids),
                )
            leaf_level = len(level_types[symbol]) - 1
            if level_types[symbol][leaf_level] is LevelType.DENSE:
                # A dense value-bearing leaf below compressed structure is
                # read at its physical leaf position; ``position_expr``
                # grounds the dense spine at the single-cursor bound
                # position and fails closed on merged sparse structure.
                # Below a union's cursor the read is guarded by that
                # cursor's alignment, with the additive identity elsewhere.
                guard = guarded_leaf_cursor(symbol)
                if guard is None:
                    return builder.position_load(
                        symbol, position_expr(symbol, leaf_level)
                    )
                return builder.position_load(
                    symbol,
                    position_expr(symbol, leaf_level, absence_handled=True),
                    guard,
                    builder.additive_identity(scalar_types[symbol]),
                )
            cursor = cursor_ids.get((symbol, leaf_level))
            if cursor is None:
                _fail(
                    "unsupported_expression",
                    f"tensor {expr.tensor.name!r} has no iterated cursor for "
                    "its value-bearing leaf level",
                )
            default: Optional[Expr] = None
            if (symbol, leaf_level) in union_cursors:
                default = builder.additive_identity(scalar_types[symbol])
            return builder.cursor_value(cursor, default)
        assert isinstance(expr, CINBinaryOp)
        return builder.binary(
            _CIN_TO_LOOPIR_BINARY[expr.op],
            lower_expr(expr.left),
            lower_expr(expr.right),
        )

    value = lower_expr(assign.rhs)
    store_indices = tuple(builder.index_value(index_id) for index_id in lhs_index_ids)
    leaf: Stmt
    if sparse_reduction_family:
        leaf = builder.store_reduce(result_symbol, store_indices, ReduceOp.ADD, value)
    elif result_sparse:
        leaf = builder.append_entry(result_symbol, store_indices, value)
    elif reduce_update:
        leaf = builder.store_reduce(result_symbol, store_indices, ReduceOp.ADD, value)
    else:
        leaf = builder.store(result_symbol, store_indices, value)

    body = builder.block((leaf,))
    for index_var in reversed(loop_vars):
        domain = domains[index_var.index_id]
        if domain.kind is DomainKind.DENSE:
            loop: Stmt = builder.dense_for(
                index_var.index_id,
                dimension_ids[index_var.index_id],
                body,
            )
        elif domain.kind is DomainKind.SPARSE:
            ref = domain.cursors[0]
            cursor_decl = builder.sparse_cursor(
                cursor_ids[(ref.tensor, ref.level)],
                ref.tensor,
                ref.level,
                position_expr(ref.tensor, ref.level - 1, absence_handled=True),
            )
            loop = builder.sparse_for(
                cursor_decl,
                position_ids[(ref.tensor, ref.level)],
                index_var.index_id,
                body,
            )
        else:
            cursor_decls = tuple(
                builder.sparse_cursor(
                    cursor_ids[(ref.tensor, ref.level)],
                    ref.tensor,
                    ref.level,
                    position_expr(ref.tensor, ref.level - 1, absence_handled=True),
                )
                for ref in domain.cursors
            )
            mode = {
                DomainKind.UNION: MergeMode.UNION,
                DomainKind.DENSE_UNION: MergeMode.DENSE_UNION,
                DomainKind.INTERSECTION: MergeMode.INTERSECTION,
            }[domain.kind]
            merge_positions: Tuple[Optional[PositionId], ...] = ()
            if mode is MergeMode.INTERSECTION and any(
                (ref.tensor, ref.level) in position_ids for ref in domain.cursors
            ):
                merge_positions = tuple(
                    position_ids.get((ref.tensor, ref.level)) for ref in domain.cursors
                )
            elif mode in (MergeMode.UNION, MergeMode.DENSE_UNION) and all(
                (ref.tensor, ref.level) in position_ids for ref in domain.cursors
            ):
                # Only the union assembly family allocates union positions,
                # so every other united program keeps its exact prior
                # position-free merge.
                merge_positions = tuple(
                    position_ids[(ref.tensor, ref.level)] for ref in domain.cursors
                )
            loop = builder.merged_sparse_for(
                mode,
                cursor_decls,
                index_var.index_id,
                body,
                merge_positions,
            )
        body = builder.block((loop,))

    ordered_tensor_decls = [
        tensor_decls[symbol] for symbol in [*input_symbols, result_symbol]
    ]
    program = builder.program(
        dimension_decls,
        ordered_tensor_decls,
        tuple(input_symbols),
        (result_symbol,),
        body,
    )
    verify_program(program)

    return LoopIRLoweringResult(
        program=program,
        loop_plan=plan,
        loop_index_ids=loop_index_ids,
        input_symbols=tuple(input_symbols),
        rhs_access_symbols=tuple(access.tensor.symbol_id for access in rhs_accesses),
        result_symbol=result_symbol,
    )
