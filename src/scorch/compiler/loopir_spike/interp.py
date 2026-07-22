"""Small plain-Python interpreter for the Phase-3.5 LoopIR spike.

This is the narrow CSR test/debug oracle the milestone asks for: it executes
verified spike programs over plain Python containers (nested float lists for
dense tensors, :class:`CsrMatrix` for sparse ones) with no Torch, native code,
or compiler-pipeline involvement.  It is not a general level-storage runtime;
the corrected Phase-3.5 review requires that redesign before Phase 4.  Merge
semantics are implemented exactly as documented on :class:`MergedSparseFor`:
candidate coordinates are the minimum over non-exhausted cursors, aligned
cursors advance by one position per step, UNION emits every candidate,
INTERSECTION emits only fully aligned candidates and stops on first exhaustion.
Everything unexpected fails closed with :class:`LoopIRInterpreterError` rather
than being coerced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple, Union

from ..identity import IndexId, SymbolId
from .csr import CsrMatrix
from .nodes import (
    Accumulate,
    AccumValue,
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DeclAccum,
    DenseFor,
    DimSize,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    ReduceOp,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    TensorDecl,
)
from .verifier import verify_program

_CSR_LEVELS = (LevelKind.DENSE, LevelKind.COMPRESSED)

TensorValue = Union[List[float], List[List[float]], CsrMatrix]


class LoopIRInterpreterError(Exception):
    """Program execution hit a state the spike interpreter rejects."""


class _CursorState:
    """Runtime state of one sparse cursor inside its owning loop."""

    __slots__ = ("matrix", "position", "end", "aligned")

    def __init__(self, matrix: CsrMatrix, start: int, end: int, aligned: bool) -> None:
        self.matrix = matrix
        self.position = start
        self.end = end
        self.aligned = aligned

    @property
    def exhausted(self) -> bool:
        return self.position >= self.end

    @property
    def coordinate(self) -> int:
        return self.matrix.indices[self.position]

    @property
    def value(self) -> float:
        return self.matrix.values[self.position]


class _CsrOutputBuilder:
    """Order-checked target-neutral assembly of one rank-2 CSR output."""

    def __init__(self, name: str, shape: Tuple[int, ...]) -> None:
        self.name = name
        self.n_rows, self.n_cols = shape
        self.rows: List[int] = []
        self.columns: List[int] = []
        self.values: List[float] = []

    def append(self, coords: Tuple[int, ...], value: float) -> None:
        row, column = coords
        if not 0 <= row < self.n_rows or not 0 <= column < self.n_cols:
            raise LoopIRInterpreterError(
                f"append to {self.name} at {coords} escapes shape "
                f"({self.n_rows}, {self.n_cols})"
            )
        if self.rows and (row, column) <= (self.rows[-1], self.columns[-1]):
            raise LoopIRInterpreterError(
                f"appends to {self.name} must be lexicographically increasing; "
                f"got {coords} after ({self.rows[-1]}, {self.columns[-1]})"
            )
        self.rows.append(row)
        self.columns.append(column)
        self.values.append(value)

    def finish(self) -> CsrMatrix:
        indptr: List[int] = [0]
        position = 0
        for row in range(self.n_rows):
            while position < len(self.rows) and self.rows[position] == row:
                position += 1
            indptr.append(position)
        return CsrMatrix(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            indptr=tuple(indptr),
            indices=tuple(self.columns),
            values=tuple(self.values),
        )


def _dense_shape(rank: int, value: object, trail: str) -> Tuple[int, ...]:
    """Validate a nested-sequence dense binding and return its shape."""

    shape: List[int] = []
    layer: object = value
    for _level in range(rank):
        if not isinstance(layer, (list, tuple)):
            raise LoopIRInterpreterError(f"{trail} must nest sequences to rank {rank}")
        shape.append(len(layer))
        if len(shape) < rank and not layer:
            raise LoopIRInterpreterError(
                f"{trail} is empty above its innermost mode; its shape "
                "cannot be inferred"
            )
        layer = layer[0] if layer else None
    return tuple(shape)


def _dense_copy(value: object, shape: Tuple[int, ...], trail: str) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        raise LoopIRInterpreterError(f"{trail} is ragged or mis-shaped")
    if len(shape) == 1:
        row: List[float] = []
        for entry in value:
            if type(entry) is not float and type(entry) is not int:
                raise LoopIRInterpreterError(f"{trail} holds a non-numeric entry")
            row.append(float(entry))
        return row
    return [
        _dense_copy(entry, shape[1:], f"{trail}[{position}]")
        for position, entry in enumerate(value)
    ]


class _Interpreter:
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
        try:
            input_keys = set(inputs)
        except Exception as error:
            raise LoopIRInterpreterError(
                "input binding keys could not be snapshotted"
            ) from error
        if input_keys != set(program.inputs):
            raise LoopIRInterpreterError(
                "input bindings must cover exactly the declared inputs"
            )
        try:
            output_keys = set(output_shapes)
        except Exception as error:
            raise LoopIRInterpreterError(
                "output shape keys could not be snapshotted"
            ) from error
        if output_keys != set(program.outputs):
            raise LoopIRInterpreterError(
                "output shapes must cover exactly the declared outputs"
            )
        try:
            input_values = {symbol: inputs[symbol] for symbol in program.inputs}
        except Exception as error:
            raise LoopIRInterpreterError(
                "input bindings could not be snapshotted"
            ) from error
        try:
            registered_output_shapes = {
                symbol: output_shapes[symbol] for symbol in program.outputs
            }
        except Exception as error:
            raise LoopIRInterpreterError(
                "output shapes could not be snapshotted"
            ) from error
        self.values: Dict[SymbolId, Any] = {}
        self.shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        for symbol in program.inputs:
            self._register_input_shape(self.decls[symbol], input_values[symbol])
        self.builders: Dict[SymbolId, _CsrOutputBuilder] = {}
        for symbol in program.outputs:
            self._register_output_shape(
                self.decls[symbol], registered_output_shapes[symbol]
            )
        self._check_extent_equalities()
        for symbol in program.inputs:
            self._materialize_input(self.decls[symbol], input_values[symbol])
        for symbol in program.outputs:
            self._materialize_output(self.decls[symbol])
        self.indices: Dict[IndexId, int] = {}
        self.accums: Dict[SymbolId, Tuple[ReduceOp, float]] = {}
        self.cursors: Dict[CursorId, _CursorState] = {}

    def _register_input_shape(self, decl: TensorDecl, value: object) -> None:
        if all(level is LevelKind.DENSE for level in decl.levels):
            shape = _dense_shape(len(decl.levels), value, f"input {decl.name}")
            self.shapes[decl.symbol] = shape
            return
        if decl.levels == _CSR_LEVELS:
            if type(value) is not CsrMatrix:
                raise LoopIRInterpreterError(
                    f"input {decl.name} must be bound to a CsrMatrix"
                )
            self.shapes[decl.symbol] = (value.n_rows, value.n_cols)
            return
        raise LoopIRInterpreterError(
            f"unsupported layout {tuple(kind.value for kind in decl.levels)} "
            f"for input {decl.name}"
        )

    def _materialize_input(self, decl: TensorDecl, value: object) -> None:
        if all(level is LevelKind.DENSE for level in decl.levels):
            self.values[decl.symbol] = _dense_copy(
                value, self.shapes[decl.symbol], f"input {decl.name}"
            )
        else:
            self.values[decl.symbol] = value

    def _register_output_shape(self, decl: TensorDecl, shape: object) -> None:
        if (
            type(shape) is not tuple
            or len(shape) != len(decl.levels)
            or any(type(extent) is not int or extent < 0 for extent in shape)
        ):
            raise LoopIRInterpreterError(
                f"output {decl.name} needs a rank-{len(decl.levels)} shape "
                "of nonnegative ints"
            )
        self.shapes[decl.symbol] = shape
        if all(level is LevelKind.DENSE for level in decl.levels):
            if len(shape) not in (1, 2):
                raise LoopIRInterpreterError(
                    f"dense outputs above rank 2 are unsupported ({decl.name})"
                )
            return
        if decl.levels == _CSR_LEVELS:
            return
        raise LoopIRInterpreterError(
            f"unsupported layout {tuple(kind.value for kind in decl.levels)} "
            f"for output {decl.name}"
        )

    def _materialize_output(self, decl: TensorDecl) -> None:
        shape = self.shapes[decl.symbol]
        if all(level is LevelKind.DENSE for level in decl.levels):
            if len(shape) == 1:
                self.values[decl.symbol] = [0.0] * shape[0]
            else:
                self.values[decl.symbol] = [[0.0] * shape[1] for _ in range(shape[0])]
        else:
            self.builders[decl.symbol] = _CsrOutputBuilder(decl.name, shape)

    def _check_extent_equalities(self) -> None:
        for position, equality in enumerate(self.program.extent_equalities):
            observed = [
                (
                    self.decls[dimension.tensor].name,
                    dimension.dim,
                    self.shapes[dimension.tensor][dimension.dim],
                )
                for dimension in equality.dimensions
            ]
            if any(extent != observed[0][2] for _, _, extent in observed[1:]):
                detail = ", ".join(
                    f"{name}[{dim}]={extent}" for name, dim, extent in observed
                )
                raise LoopIRInterpreterError(
                    f"extent equality {position} failed: {detail}"
                )

    def run(self) -> Dict[SymbolId, TensorValue]:
        self._exec_block(self.program.body)
        results: Dict[SymbolId, TensorValue] = {}
        for symbol in self.program.outputs:
            if symbol in self.builders:
                results[symbol] = self.builders[symbol].finish()
            else:
                results[symbol] = self.values[symbol]
        return results

    def _eval_coord(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIRInterpreterError(
                f"coordinate expression produced {type(result).__name__}"
            )
        return result

    def _eval_value(self, expr: Expr) -> float:
        result = self._eval(expr)
        if type(result) is not float:
            raise LoopIRInterpreterError(
                f"value expression produced {type(result).__name__}"
            )
        return result

    def _eval(self, expr: Expr) -> object:
        if type(expr) is IntConst:
            return expr.value
        if type(expr) is FloatConst:
            return expr.value
        if type(expr) is DimSize:
            return self.shapes[expr.tensor][expr.dim]
        if type(expr) is IndexValue:
            return self.indices[expr.index]
        if type(expr) is AccumValue:
            return self.accums[expr.accumulator][1]
        if type(expr) is CursorValue:
            state = self.cursors[expr.cursor]
            if state.aligned:
                return state.value
            if expr.default is None:
                raise LoopIRInterpreterError("unaligned cursor read without a default")
            return self._eval_value(expr.default)
        if type(expr) is Load:
            current: Any = self.values[expr.tensor]
            for position, index_expr in enumerate(expr.indices):
                index = self._eval_coord(index_expr)
                if not isinstance(current, list) or not 0 <= index < len(current):
                    raise LoopIRInterpreterError(
                        f"load index {index} out of bounds at mode {position}"
                    )
                current = current[index]
            if type(current) is not float:
                raise LoopIRInterpreterError("load did not resolve to a scalar")
            return current
        if type(expr) is BinaryExpr:
            lhs = self._eval_value(expr.lhs)
            rhs = self._eval_value(expr.rhs)
            if expr.op is BinaryOp.ADD:
                return lhs + rhs
            if expr.op is BinaryOp.SUB:
                return lhs - rhs
            return lhs * rhs
        raise LoopIRInterpreterError(f"unknown expression {type(expr).__name__}")

    def _segment(self, decl: SparseCursorDecl) -> Tuple[CsrMatrix, int, int]:
        tensor_decl = self.decls[decl.tensor]
        if tensor_decl.levels != _CSR_LEVELS or decl.level != 1:
            raise LoopIRInterpreterError(
                f"unsupported cursor layout on {tensor_decl.name}"
            )
        matrix = self.values[decl.tensor]
        if type(matrix) is not CsrMatrix:
            raise LoopIRInterpreterError(
                f"cursor tensor {tensor_decl.name} is not CSR-bound"
            )
        row = self._eval_coord(decl.outer_indices[0])
        if not 0 <= row < matrix.n_rows:
            raise LoopIRInterpreterError(
                f"cursor row {row} outside [0, {matrix.n_rows}) on "
                f"{tensor_decl.name}"
            )
        start, end = matrix.row_segment(row)
        return matrix, start, end

    def _exec_block(self, block: Block) -> None:
        declared_here: List[SymbolId] = []
        for stmt in block.statements:
            self._exec_stmt(stmt)
            if type(stmt) is DeclAccum:
                declared_here.append(stmt.accumulator)
        for accumulator in declared_here:
            del self.accums[accumulator]

    def _exec_stmt(self, stmt: Stmt) -> None:
        if type(stmt) is Block:
            self._exec_block(stmt)
            return
        if type(stmt) is DenseFor:
            extent = self._eval_coord(stmt.extent)
            if extent < 0:
                raise LoopIRInterpreterError(f"negative dense extent {extent}")
            try:
                for coordinate in range(extent):
                    self.indices[stmt.index] = coordinate
                    self._exec_block(stmt.body)
            finally:
                self.indices.pop(stmt.index, None)
            return
        if type(stmt) is SparseFor:
            matrix, start, end = self._segment(stmt.cursor)
            state = _CursorState(matrix, start, end, aligned=True)
            self.cursors[stmt.cursor.cursor] = state
            try:
                while not state.exhausted:
                    self.indices[stmt.coord_index] = state.coordinate
                    self._exec_block(stmt.body)
                    state.position += 1
            finally:
                self.indices.pop(stmt.coord_index, None)
                del self.cursors[stmt.cursor.cursor]
            return
        if type(stmt) is MergedSparseFor:
            self._exec_merge(stmt)
            return
        if type(stmt) is DeclAccum:
            self.accums[stmt.accumulator] = (stmt.op, self._eval_value(stmt.init))
            return
        if type(stmt) is Accumulate:
            op, current = self.accums[stmt.accumulator]
            contribution = self._eval_value(stmt.value)
            combined = (
                current + contribution if op is ReduceOp.ADD else current * contribution
            )
            self.accums[stmt.accumulator] = (op, combined)
            return
        if type(stmt) is Store:
            coords = [self._eval_coord(index) for index in stmt.indices]
            value = self._eval_value(stmt.value)
            target: Any = self.values[stmt.tensor]
            for index in coords[:-1]:
                if not isinstance(target, list) or not 0 <= index < len(target):
                    raise LoopIRInterpreterError(f"store index {index} out of bounds")
                target = target[index]
            last = coords[-1]
            if not isinstance(target, list) or not 0 <= last < len(target):
                raise LoopIRInterpreterError(f"store index {last} out of bounds")
            target[last] = value
            return
        if type(stmt) is AppendEntry:
            coords_tuple = tuple(self._eval_coord(coord) for coord in stmt.coords)
            value = self._eval_value(stmt.value)
            self.builders[stmt.tensor].append(coords_tuple, value)
            return
        raise LoopIRInterpreterError(f"unknown statement {type(stmt).__name__}")

    def _exec_merge(self, stmt: MergedSparseFor) -> None:
        states: List[_CursorState] = []
        try:
            for decl in stmt.cursors:
                matrix, start, end = self._segment(decl)
                state = _CursorState(matrix, start, end, aligned=False)
                self.cursors[decl.cursor] = state
                states.append(state)
            while True:
                if stmt.mode is MergeMode.UNION:
                    active = [state for state in states if not state.exhausted]
                    if not active:
                        break
                else:
                    if any(state.exhausted for state in states):
                        break
                    active = states
                candidate = min(state.coordinate for state in active)
                aligned = [state for state in active if state.coordinate == candidate]
                for state in states:
                    state.aligned = (
                        not state.exhausted and state.coordinate == candidate
                    )
                if stmt.mode is MergeMode.UNION or len(aligned) == len(states):
                    self.indices[stmt.coord_index] = candidate
                    self._exec_block(stmt.body)
                for state in aligned:
                    state.position += 1
        finally:
            self.indices.pop(stmt.coord_index, None)
            for decl in stmt.cursors:
                self.cursors.pop(decl.cursor, None)


def run_program(
    program: LoopProgram,
    inputs: Mapping[SymbolId, object],
    output_shapes: Mapping[SymbolId, Tuple[int, ...]],
) -> Dict[SymbolId, TensorValue]:
    """Verify then execute one spike program over plain-Python containers."""

    verify_program(program)
    return _Interpreter(program, inputs, output_shapes).run()
