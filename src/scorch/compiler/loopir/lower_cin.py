"""Normalized-CIN-to-LoopIR lowering for the Phase-4 dense families.

This is the strangler-path entry that turns one normalized CIN program into a
verified production LoopIR program plus its verified ``LoopPlan``.  It covers
two coherent families rather than hand-built fixtures:

- **Dense elementwise** — a pure ``ForAll`` nest over one plain
  ``TensorAssign`` (no update operator) whose right-hand side is a
  ``{ADD, SUB, MUL}`` expression tree over all-dense tensor accesses, with
  every loop variable free (bound on the left-hand side).
- **Dense reduction/matmul** — the same shape with an ``ADD`` update
  operator; loop variables absent from the left-hand side are reduction
  variables realized as ``StoreReduce`` (ADD) into the zero-initialized
  dense output.

Everything outside the families fails closed with
:class:`LoopIRLoweringError` and a stable code — nothing is silently
degraded to the legacy path from here.  The input CIN is never mutated;
stable ``SymbolId``/``IndexId`` identities flow through unchanged as
provenance, and each bound loop variable becomes one declared logical
dimension.

Recorded family boundaries (deliberate, fail-closed; see the Phase-4
review): identity mode order only, one uniform float32/float64 scalar type,
no workspaces/``Where``/derived index arithmetic/explicit parallel marks,
``ADD`` as the only update operator and ``{ADD, SUB, MUL}`` as the only
value operators, and every tensor's storage-order loop variables must appear
in nest order (the position-chain order the legacy dense emission also
requires).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NoReturn, Tuple

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
from .nodes import (
    BinaryOp,
    DimensionId,
    Expr,
    LoopProgram,
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


def _check_tensor(tensor: TensorVar) -> ScalarType:
    if isinstance(tensor, cin_nodes.Workspace):
        _fail(
            "unsupported_workspace",
            f"workspace {tensor.name!r} is outside the dense families",
        )
    tensor_format = tensor.format
    if tensor_format is None:
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} declares no format",
        )
    if any(
        level_type is not LevelType.DENSE
        for level_type in tensor_format.get_level_types()
    ):
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} is not all-dense",
        )
    mode_order = tensor.mode_order
    rank = len(tensor_format.get_level_types())
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
    return scalar_type


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


def lower_normalized_cin_to_loopir(cin: IndexStmt) -> LoopIRLoweringResult:
    """Lower one normalized dense-family CIN program to verified LoopIR."""

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

    # Every tensor's storage-order loop variables must appear in nest order,
    # the same dependency direction the legacy dense position chains use.
    for access in all_accesses:
        positions = [loop_positions[index_var.index_id] for index_var in access.indices]
        if positions != sorted(positions):
            _fail(
                "unsupported_loop_order",
                f"tensor {access.tensor.name!r} storage order conflicts with "
                "the loop nest order",
            )

    scalar_types = {
        access.tensor.symbol_id: _check_tensor(access.tensor) for access in all_accesses
    }
    if len(set(scalar_types.values())) > 1:
        _fail(
            "mixed_dtype",
            "the dense families require one uniform scalar type",
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
