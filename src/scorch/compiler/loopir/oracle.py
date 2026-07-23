"""Production-owned test/debug semantic oracle for the LoopIR dense subset.

This module promotes the Phase-3.5 spike interpreter's semantics into the
production tree for the migrated dense families.  It executes verified
LoopIR programs over plain Python containers with no Torch, native code, or
compiler-pipeline involvement, and is loaded only by dedicated oracle and
differential tests — never by ``import scorch``, default compilation, legacy
correctness paths, or release JIT.

Semantics (matching the node contracts exactly):

- shape compatibility is the logical dimension model: every bound tensor
  contributes the extent of each dimension its modes store, all extents must
  agree across inputs and outputs before anything executes, and ``DenseFor``
  resolves its trip count from the dimension extent;
- dense outputs are zero-initialized before the body runs — the explicit
  contract ``StoreReduce`` (ADD) depends on;
- all arithmetic is Python-float arithmetic in program order; the oracle is
  a semantic reference, not a bit-accuracy model of any particular scalar
  width, so numeric comparisons against compiled kernels use tolerances
  owned by the caller.

Everything unexpected fails closed with :class:`LoopIROracleError`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from ..identity import IndexId, SymbolId
from .nodes import (
    BinaryExpr,
    BinaryOp,
    Block,
    DenseFor,
    DimensionId,
    Expr,
    IndexValue,
    Load,
    LoopProgram,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)
from .verifier import verify_program

TensorValue = Any


class LoopIROracleError(Exception):
    """Program execution hit a state the LoopIR oracle rejects."""


def _dense_shape(rank: int, value: object, trail: str) -> Tuple[int, ...]:
    """Validate a nested-sequence dense binding and return its shape."""

    shape: List[int] = []
    layer: object = value
    for _level in range(rank):
        if type(layer) not in (list, tuple):
            raise LoopIROracleError(f"{trail} must nest sequences to rank {rank}")
        owned_layer = cast(Sequence[object], layer)
        shape.append(len(owned_layer))
        if len(shape) < rank and not owned_layer:
            raise LoopIROracleError(
                f"{trail} is empty above its innermost mode; its shape "
                "cannot be inferred"
            )
        layer = owned_layer[0] if owned_layer else None
    return tuple(shape)


def _dense_copy(value: object, shape: Tuple[int, ...], trail: str) -> Any:
    if type(value) not in (list, tuple):
        raise LoopIROracleError(f"{trail} is ragged or mis-shaped")
    owned_value = cast(Sequence[object], value)
    if len(owned_value) != shape[0]:
        raise LoopIROracleError(f"{trail} is ragged or mis-shaped")
    if len(shape) == 1:
        row: List[float] = []
        for entry in owned_value:
            if type(entry) is not float and type(entry) is not int:
                raise LoopIROracleError(f"{trail} holds a non-numeric entry")
            try:
                row.append(float(entry))
            except (OverflowError, TypeError, ValueError) as error:
                raise LoopIROracleError(
                    f"{trail} holds an unrepresentable numeric entry"
                ) from error
        return row
    return [
        _dense_copy(entry, shape[1:], f"{trail}[{position}]")
        for position, entry in enumerate(owned_value)
    ]


def _zeros(shape: Tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class _Oracle:
    def __init__(
        self,
        program: LoopProgram,
        inputs: Mapping[SymbolId, object],
        output_shapes: Mapping[SymbolId, Tuple[int, ...]],
    ) -> None:
        self.program = program
        self.decls: Dict[SymbolId, TensorDecl] = {
            decl.symbol: decl for decl in program.tensors
        }
        self.dimension_names: Dict[DimensionId, str] = {
            decl.dimension: decl.name for decl in program.dimensions
        }
        try:
            input_key_snapshot = tuple(inputs)
        except Exception as error:
            raise LoopIROracleError(
                "input binding keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in input_key_snapshot
        ):
            raise LoopIROracleError(
                "input binding keys must be exact int-valued SymbolId values"
            )
        if set(input_key_snapshot) != set(program.inputs):
            raise LoopIROracleError(
                "input bindings must cover exactly the declared inputs"
            )
        try:
            output_key_snapshot = tuple(output_shapes)
        except Exception as error:
            raise LoopIROracleError(
                "output shape keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in output_key_snapshot
        ):
            raise LoopIROracleError(
                "output shape keys must be exact int-valued SymbolId values"
            )
        if set(output_key_snapshot) != set(program.outputs):
            raise LoopIROracleError(
                "output shapes must cover exactly the declared outputs"
            )

        self.values: Dict[SymbolId, Any] = {}
        self.shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        for symbol in program.inputs:
            decl = self.decls[symbol]
            try:
                bound = inputs[symbol]
            except Exception as error:
                raise LoopIROracleError(
                    "input bindings could not be snapshotted"
                ) from error
            shape = _dense_shape(len(decl.levels), bound, f"input {decl.name}")
            self.shapes[symbol] = shape
            self.values[symbol] = _dense_copy(bound, shape, f"input {decl.name}")
        for symbol in program.outputs:
            decl = self.decls[symbol]
            try:
                shape_binding = output_shapes[symbol]
            except Exception as error:
                raise LoopIROracleError(
                    "output shapes could not be snapshotted"
                ) from error
            if (
                type(shape_binding) is not tuple
                or len(shape_binding) != len(decl.levels)
                or any(
                    type(extent) is not int or extent < 0 for extent in shape_binding
                )
            ):
                raise LoopIROracleError(
                    f"output {decl.name} needs a rank-{len(decl.levels)} shape "
                    "of nonnegative ints"
                )
            self.shapes[symbol] = shape_binding
            self.values[symbol] = _zeros(shape_binding) if shape_binding else []

        self.dim_extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        self._resolve_dimension_extents()
        self.indices: Dict[IndexId, int] = {}

    def _resolve_dimension_extents(self) -> None:
        """Bind every dimension's extent before executing anything."""

        for decl in self.program.tensors:
            shape = self.shapes[decl.symbol]
            for mode, dimension in enumerate(decl.dimensions):
                extent = shape[mode]
                known = self.dim_extents.get(dimension)
                if known is None:
                    self.dim_extents[dimension] = (extent, decl.name, mode)
                elif known[0] != extent:
                    name = self.dimension_names.get(
                        dimension, f"<dimension {dimension.value}>"
                    )
                    raise LoopIROracleError(
                        f"dimension extent mismatch for {name!r}: "
                        f"{known[1]}[{known[2]}] is {known[0]} but "
                        f"{decl.name}[{mode}] is {extent}"
                    )

    def _dimension_extent(self, dimension: DimensionId) -> int:
        known = self.dim_extents.get(dimension)
        if known is None:
            name = self.dimension_names.get(dimension, f"<dimension {dimension.value}>")
            raise LoopIROracleError(
                f"unresolved dimension extent for {name!r}: no bound tensor "
                "stores this dimension"
            )
        return known[0]

    def run(self) -> Dict[SymbolId, TensorValue]:
        self._exec_stmt(self.program.body)
        return {symbol: self.values[symbol] for symbol in self.program.outputs}

    def _eval_coord(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIROracleError(
                f"coordinate expression produced {type(result).__name__}"
            )
        return result

    def _eval_value(self, expr: Expr) -> float:
        result = self._eval(expr)
        if type(result) is not float:
            raise LoopIROracleError(
                f"value expression produced {type(result).__name__}"
            )
        return result

    def _eval(self, expr: Expr) -> object:
        if type(expr) is IndexValue:
            return self.indices[expr.index]
        if type(expr) is Load:
            current: Any = self.values[expr.tensor]
            for position, index_expr in enumerate(expr.indices):
                index = self._eval_coord(index_expr)
                if not isinstance(current, list) or not 0 <= index < len(current):
                    raise LoopIROracleError(
                        f"load index {index} out of bounds at mode {position}"
                    )
                current = current[index]
            if type(current) is not float:
                raise LoopIROracleError("load did not resolve to a scalar")
            return current
        if type(expr) is BinaryExpr:
            lhs = self._eval_value(expr.lhs)
            rhs = self._eval_value(expr.rhs)
            if expr.op is BinaryOp.ADD:
                return lhs + rhs
            if expr.op is BinaryOp.SUB:
                return lhs - rhs
            return lhs * rhs
        raise LoopIROracleError(f"unknown expression {type(expr).__name__}")

    def _locate_store(
        self, tensor: SymbolId, indices: Tuple[Expr, ...]
    ) -> Tuple[List[Any], int]:
        coords = [self._eval_coord(index) for index in indices]
        target: Any = self.values[tensor]
        for index in coords[:-1]:
            if not isinstance(target, list) or not 0 <= index < len(target):
                raise LoopIROracleError(f"store index {index} out of bounds")
            target = target[index]
        last = coords[-1]
        if not isinstance(target, list) or not 0 <= last < len(target):
            raise LoopIROracleError(f"store index {last} out of bounds")
        return target, last

    def _exec_stmt(self, stmt: Stmt) -> None:
        if type(stmt) is Block:
            for child in stmt.statements:
                self._exec_stmt(child)
            return
        if type(stmt) is DenseFor:
            extent = self._dimension_extent(stmt.dimension)
            try:
                for coordinate in range(extent):
                    self.indices[stmt.index] = coordinate
                    self._exec_stmt(stmt.body)
            finally:
                self.indices.pop(stmt.index, None)
            return
        if type(stmt) is Store:
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            target[last] = self._eval_value(stmt.value)
            return
        if type(stmt) is StoreReduce:
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            contribution = self._eval_value(stmt.value)
            target[last] = target[last] + contribution
            return
        raise LoopIROracleError(f"unknown statement {type(stmt).__name__}")


def run_program(
    program: LoopProgram,
    inputs: Mapping[SymbolId, object],
    output_shapes: Mapping[SymbolId, Tuple[int, ...]],
) -> Dict[SymbolId, TensorValue]:
    """Verify then execute one LoopIR program over plain-Python containers."""

    verify_program(program)
    return _Oracle(program, inputs, output_shapes).run()
