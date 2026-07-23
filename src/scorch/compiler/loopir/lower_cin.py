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
Phase-5 reviews): identity mode order only, one uniform float32/float64
scalar type, no workspaces/``Where``/derived index arithmetic/explicit
parallel marks, ``ADD`` as the only update operator and ``{ADD, SUB, MUL}``
as the only value operators, every tensor's storage-order loop variables in
nest order, DENSE/COMPRESSED level types only with compressed value-bearing
leaves, canonical CSR as the only sparse output (no update operator, no
merged reductions or merged updates), and no subtraction across sparse
domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NoReturn, Optional, Tuple

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
)
from ..cin_analysis import verify_cin
from ..loop_plan import LoopPlan, verify_loop_plan
from .build import LoopIRBuilder
from .iterdomain import (
    DomainKind,
    IterationDomainError,
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
}


def _check_tensor(tensor: TensorVar) -> Tuple[ScalarType, Tuple[LevelType, ...]]:
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
    if any(level_type not in _LEVEL_TYPE_TO_KIND for level_type in level_types):
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} declares a level type outside the "
            "migrated DENSE/COMPRESSED families",
        )
    if (
        any(level_type is LevelType.COMPRESSED for level_type in level_types)
        and level_types[-1] is not LevelType.COMPRESSED
    ):
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} stores a dense value-bearing leaf below "
            "compressed structure; physical position loads are not declared "
            "in this subset",
        )
    mode_order = tensor.mode_order
    rank = len(level_types)
    if mode_order is not None and list(mode_order) != list(range(rank)):
        _fail(
            "unsupported_mode_order",
            f"tensor {tensor.name!r} uses a non-identity mode order",
        )
    scalar_type = _TORCH_TO_SCALAR.get(tensor.dtype)
    if scalar_type is None:
        _fail(
            "unsupported_dtype",
            f"tensor {tensor.name!r} dtype {tensor.dtype} is outside "
            "float32/float64",
        )
    assert scalar_type is not None
    return scalar_type, level_types


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


def lower_normalized_cin_to_loopir(
    cin: IndexStmt,
    *,
    planned_loop_order: Optional[Tuple[IndexId, ...]] = None,
) -> LoopIRLoweringResult:
    """Lower one normalized dense-family CIN program to verified LoopIR.

    ``planned_loop_order`` is the scheduled path's verified plan order.  The
    base program is always constructed in the CIN's own nest order; the plan
    order only replaces the nest order in the storage-order family check, so
    a schedule that permutes the nest into a storage-consistent order is not
    rejected before the reorder pass can apply it.  The target lowering
    re-enforces the same boundary against the program it actually emits.
    """

    if not isinstance(cin, IndexStmt):
        raise TypeError("lower_normalized_cin_to_loopir expects an IndexStmt")
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
    order_positions = loop_positions
    if planned_loop_order is not None:
        if set(planned_loop_order) != set(loop_positions) or len(
            planned_loop_order
        ) != len(loop_positions):
            _fail(
                "unsupported_loop_order",
                "the planned loop order must name every nest loop exactly " "once",
            )
        order_positions = {
            index_id: position for position, index_id in enumerate(planned_loop_order)
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
        _fail(
            "unsupported_reduction_without_update",
            "a loop variable outside the left-hand side requires the ADD "
            "update operator",
        )

    # Every tensor's storage-order loop variables must appear in nest order
    # (the planned order on the scheduled path), the same dependency
    # direction the legacy dense position chains use.
    for access in all_accesses:
        positions = [
            order_positions[index_var.index_id] for index_var in access.indices
        ]
        if positions != sorted(positions):
            _fail(
                "unsupported_loop_order",
                f"tensor {access.tensor.name!r} storage order conflicts with "
                "the loop nest order",
            )

    checked_tensors = {
        access.tensor.symbol_id: _check_tensor(access.tensor) for access in all_accesses
    }
    scalar_types = {symbol: scalar for symbol, (scalar, _) in checked_tensors.items()}
    level_types = {symbol: levels for symbol, (_, levels) in checked_tensors.items()}
    if len(set(scalar_types.values())) > 1:
        _fail(
            "mixed_dtype",
            "the migrated families require one uniform scalar type",
        )

    if any(
        level_type is not LevelType.DENSE
        for levels in level_types.values()
        for level_type in levels
    ):
        return _lower_sparse_family(
            cin,
            loop_vars,
            assign,
            reduce_update,
            lhs,
            rhs_accesses,
            input_symbols_of(rhs_accesses),
            result_symbol,
            scalar_types,
            level_types,
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
            builder.dense_levels(len(access.index_ids)),
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
    for index_var in reversed(loop_vars):
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

    loop_index_ids = tuple(index_var.index_id for index_var in loop_vars)
    plan = verify_loop_plan(cin, LoopPlan(loop_order=loop_index_ids))

    return LoopIRLoweringResult(
        program=program,
        loop_plan=plan,
        loop_index_ids=loop_index_ids,
        input_symbols=tuple(input_symbols),
        rhs_access_symbols=tuple(access.tensor.symbol_id for access in rhs_accesses),
        result_symbol=result_symbol,
    )


def _lower_sparse_family(
    cin: IndexStmt,
    loop_vars: List[IndexVar],
    assign: TensorAssign,
    reduce_update: bool,
    lhs: TensorAccess,
    rhs_accesses: List[TensorAccess],
    input_symbols: List[SymbolId],
    result_symbol: SymbolId,
    scalar_types: Dict[SymbolId, ScalarType],
    level_types: Dict[SymbolId, Tuple[LevelType, ...]],
) -> LoopIRLoweringResult:
    """Materialize the sparse level families from the pure domain analysis."""

    loop_index_ids = tuple(index_var.index_id for index_var in loop_vars)
    plan = verify_loop_plan(cin, LoopPlan(loop_order=loop_index_ids))
    try:
        table = analyze_iteration_domains(cin, plan)
    except IterationDomainError as error:
        raise LoopIRLoweringError(
            LoopIRLoweringDefect(error.defect.code, error.defect.message)
        ) from error
    domains = {domain.index: domain for domain in table.domains}

    result_levels = level_types[result_symbol]
    result_sparse = any(
        level_type is LevelType.COMPRESSED for level_type in result_levels
    )
    lhs_index_ids = tuple(lhs.index_ids)

    if result_sparse:
        if result_levels != (LevelType.DENSE, LevelType.COMPRESSED):
            _fail(
                "unsupported_sparse_output",
                f"result {lhs.tensor.name!r} declares a sparse layout other "
                "than canonical CSR; ordered assembly is defined for "
                "canonical CSR only in the migrated families",
            )
        if reduce_update:
            _fail(
                "unsupported_sparse_output_reduction",
                "a canonical CSR output cannot carry the ADD update "
                "operator in the migrated families",
            )
        row_domain = domains[lhs_index_ids[0]]
        column_domain = domains[lhs_index_ids[1]]
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

    for index_id in loop_index_ids:
        domain = domains[index_id]
        if domain.kind in (DomainKind.UNION, DomainKind.INTERSECTION):
            if index_id not in lhs_index_ids:
                _fail(
                    "unsupported_merged_reduction",
                    "reducing over a merged sparse domain is outside the "
                    "migrated families",
                )
            if reduce_update:
                _fail(
                    "unsupported_merged_update",
                    "combining a merged sparse domain with the ADD update "
                    "operator is outside the migrated families",
                )

    builder = LoopIRBuilder()
    dimension_ids: Dict[IndexId, DimensionId] = {}
    dimension_decls = []
    for index_var in loop_vars:
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
                for mode, level_type in enumerate(level_types[symbol])
            ),
        )

    for access in rhs_accesses:
        declare_tensor(access)
    declare_tensor(lhs)

    # Allocate cursor and bound-position identities for every compressed
    # level the domain table iterates; UNION-merged cursors read through an
    # explicit additive-identity default.
    cursor_ids: Dict[Tuple[SymbolId, int], CursorId] = {}
    position_ids: Dict[Tuple[SymbolId, int], PositionId] = {}
    union_cursors: set = set()
    for domain in table.domains:
        for ref in domain.cursors:
            key = (ref.tensor, ref.level)
            if key not in cursor_ids:
                cursor_ids[key] = builder.new_cursor_id()
            if domain.kind is DomainKind.SPARSE:
                position_ids[key] = builder.new_position_id()
            elif domain.kind is DomainKind.UNION:
                union_cursors.add(key)

    def position_expr(symbol: SymbolId, level: int) -> Expr:
        """The dominating physical position of one tensor level."""

        if level < 0:
            return builder.root_position()
        kind = _LEVEL_TYPE_TO_KIND[level_types[symbol][level]]
        if kind is LevelKind.DENSE:
            driving = tensor_accesses[symbol].index_ids[level]
            return builder.dense_position(
                symbol,
                level,
                position_expr(symbol, level - 1),
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

    def lower_expr(expr: object) -> Expr:
        if isinstance(expr, TensorAccess):
            symbol = expr.tensor.symbol_id
            if all(level_type is LevelType.DENSE for level_type in level_types[symbol]):
                return builder.load(
                    symbol,
                    tuple(builder.index_value(index_id) for index_id in expr.index_ids),
                )
            leaf_level = len(level_types[symbol]) - 1
            cursor = cursor_ids.get((symbol, leaf_level))
            if cursor is None:
                _fail(
                    "unsupported_expression",
                    f"tensor {expr.tensor.name!r} has no iterated cursor for "
                    "its value-bearing leaf level",
                )
            default: Optional[Expr] = None
            if (symbol, leaf_level) in union_cursors:
                default = builder.float_const(0.0)
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
    if result_sparse:
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
                position_expr(ref.tensor, ref.level - 1),
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
                    position_expr(ref.tensor, ref.level - 1),
                )
                for ref in domain.cursors
            )
            mode = (
                MergeMode.UNION
                if domain.kind is DomainKind.UNION
                else MergeMode.INTERSECTION
            )
            loop = builder.merged_sparse_for(
                mode,
                cursor_decls,
                index_var.index_id,
                body,
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
