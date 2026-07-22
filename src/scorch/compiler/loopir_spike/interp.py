"""Small plain-Python interpreter for the Phase-3.5 LoopIR spike.

This is the milestone's test/debug oracle: it executes verified spike
programs over plain Python containers with no Torch, native code, or
compiler-pipeline involvement.  Sparse traversal is defined entirely over
the format-neutral level interface in :mod:`levels` — the execution core
reads segments, coordinates, and leaf values from a validated
:class:`LevelTensorStorage` and never inspects a concrete container.  CSR
remains exactly one adapter: canonical CSR-declared inputs bind a
:class:`CsrMatrix` (adapted through :func:`levels.from_csr`) and canonical
CSR-declared outputs assemble through :class:`levels.CsrOutputBuilder`;
every other DENSE/COMPRESSED level composition binds a
:class:`LevelTensorStorage` directly.  All-dense tensors bind nested float
lists in logical mode order.

Shape compatibility is the logical dimension model: every bound tensor
contributes the extent of each dimension its modes store, the extents must
agree across all inputs and outputs before anything is materialized or
executed, and loops resolve their trip counts from those extents — so
incompatible shapes fail independently of stored sparsity.

Merge semantics are implemented exactly as documented on
:class:`MergedSparseFor`: candidate coordinates are the minimum over
non-exhausted cursors, aligned cursors advance by one position per step,
UNION emits every candidate, INTERSECTION emits only fully aligned
candidates and stops on first exhaustion.  Everything unexpected fails
closed with :class:`LoopIRInterpreterError` rather than being coerced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple, Union

from ..identity import IndexId, SymbolId
from .csr import CsrMatrix
from .levels import (
    CsrOutputBuilder,
    LevelStorageError,
    LevelTensorStorage,
    from_csr,
)
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
    DensePosition,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PositionId,
    PositionValue,
    ReduceOp,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)
from .verifier import verify_program

_CSR_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)
_IDENTITY_MODES = (0, 1)

TensorValue = Union[List[float], List[List[float]], CsrMatrix]


class LoopIRInterpreterError(Exception):
    """Program execution hit a state the spike interpreter rejects."""


class _CursorState:
    """Runtime state of one sparse cursor inside its owning loop."""

    __slots__ = ("storage", "level", "position", "end", "aligned")

    def __init__(
        self,
        storage: LevelTensorStorage,
        level: int,
        start: int,
        end: int,
        aligned: bool,
    ) -> None:
        self.storage = storage
        self.level = level
        self.position = start
        self.end = end
        self.aligned = aligned

    @property
    def exhausted(self) -> bool:
        return self.position >= self.end

    @property
    def coordinate(self) -> int:
        return self.storage.coordinate_at(self.level, self.position)

    @property
    def value(self) -> float:
        return self.storage.leaf_value(self.position)


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


def _is_canonical_csr(decl: TensorDecl) -> bool:
    kinds = tuple(level.kind for level in decl.levels)
    modes = tuple(level.mode for level in decl.levels)
    return kinds == _CSR_KINDS and modes == _IDENTITY_MODES


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
        self.dimension_names: Dict[DimensionId, str] = {
            decl.dimension: decl.name for decl in program.dimensions
        }
        try:
            input_key_snapshot = tuple(inputs)
        except Exception as error:
            raise LoopIRInterpreterError(
                "input binding keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in input_key_snapshot
        ):
            raise LoopIRInterpreterError(
                "input binding keys must be exact int-valued SymbolId values"
            )
        input_keys = set(input_key_snapshot)
        if input_keys != set(program.inputs):
            raise LoopIRInterpreterError(
                "input bindings must cover exactly the declared inputs"
            )
        try:
            output_key_snapshot = tuple(output_shapes)
        except Exception as error:
            raise LoopIRInterpreterError(
                "output shape keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in output_key_snapshot
        ):
            raise LoopIRInterpreterError(
                "output shape keys must be exact int-valued SymbolId values"
            )
        output_keys = set(output_key_snapshot)
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
        self.storages: Dict[SymbolId, LevelTensorStorage] = {}
        self.shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        for symbol in program.inputs:
            self._register_input_shape(self.decls[symbol], input_values[symbol])
        self.builders: Dict[SymbolId, CsrOutputBuilder] = {}
        for symbol in program.outputs:
            self._register_output_shape(
                self.decls[symbol], registered_output_shapes[symbol]
            )
        self.dim_extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        self._resolve_dimension_extents()
        for symbol in program.inputs:
            self._materialize_input(self.decls[symbol], input_values[symbol])
        for symbol in program.outputs:
            self._materialize_output(self.decls[symbol])
        self.indices: Dict[IndexId, int] = {}
        self.positions: Dict[PositionId, int] = {}
        self.accums: Dict[SymbolId, Tuple[ReduceOp, float]] = {}
        self.cursors: Dict[CursorId, _CursorState] = {}

    def _register_input_shape(self, decl: TensorDecl, value: object) -> None:
        if all(level.kind is LevelKind.DENSE for level in decl.levels):
            shape = _dense_shape(len(decl.levels), value, f"input {decl.name}")
            self.shapes[decl.symbol] = shape
            return
        if _is_canonical_csr(decl):
            if type(value) is not CsrMatrix:
                raise LoopIRInterpreterError(
                    f"input {decl.name} must be bound to a CsrMatrix"
                )
            self.shapes[decl.symbol] = (value.n_rows, value.n_cols)
            return
        if type(value) is not LevelTensorStorage:
            raise LoopIRInterpreterError(
                f"input {decl.name} must be bound to a LevelTensorStorage"
            )
        declared_kinds = tuple(level.kind for level in decl.levels)
        declared_modes = tuple(level.mode for level in decl.levels)
        if value.kinds != declared_kinds or value.modes != declared_modes:
            raise LoopIRInterpreterError(
                f"input {decl.name} storage layout "
                f"({tuple(kind.value for kind in value.kinds)}, "
                f"modes {value.modes}) does not match its declaration "
                f"({tuple(kind.value for kind in declared_kinds)}, "
                f"modes {declared_modes})"
            )
        self.shapes[decl.symbol] = value.shape

    def _materialize_input(self, decl: TensorDecl, value: object) -> None:
        if all(level.kind is LevelKind.DENSE for level in decl.levels):
            self.values[decl.symbol] = _dense_copy(
                value, self.shapes[decl.symbol], f"input {decl.name}"
            )
        elif _is_canonical_csr(decl):
            if type(value) is not CsrMatrix:
                raise LoopIRInterpreterError(
                    f"input {decl.name} must be bound to a CsrMatrix"
                )
            self.storages[decl.symbol] = from_csr(value)
        else:
            if type(value) is not LevelTensorStorage:
                raise LoopIRInterpreterError(
                    f"input {decl.name} must be bound to a LevelTensorStorage"
                )
            self.storages[decl.symbol] = value

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
        if all(level.kind is LevelKind.DENSE for level in decl.levels):
            if len(shape) not in (1, 2):
                raise LoopIRInterpreterError(
                    f"dense outputs above rank 2 are unsupported ({decl.name})"
                )
            return
        if _is_canonical_csr(decl):
            return
        raise LoopIRInterpreterError(
            f"unsupported sparse output layout for {decl.name}; only "
            "canonical CSR outputs are assembled by this spike"
        )

    def _materialize_output(self, decl: TensorDecl) -> None:
        shape = self.shapes[decl.symbol]
        if all(level.kind is LevelKind.DENSE for level in decl.levels):
            if len(shape) == 1:
                self.values[decl.symbol] = [0.0] * shape[0]
            else:
                self.values[decl.symbol] = [[0.0] * shape[1] for _ in range(shape[0])]
        else:
            self.builders[decl.symbol] = CsrOutputBuilder(decl.name, shape)

    def _resolve_dimension_extents(self) -> None:
        """Bind every dimension's extent; disagreement fails sparsity-free."""

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
                    raise LoopIRInterpreterError(
                        f"dimension extent mismatch for {name!r}: "
                        f"{known[1]}[{known[2]}] is {known[0]} but "
                        f"{decl.name}[{mode}] is {extent}"
                    )

    def _dimension_extent(self, dimension: DimensionId) -> int:
        known = self.dim_extents.get(dimension)
        if known is None:
            name = self.dimension_names.get(dimension, f"<dimension {dimension.value}>")
            raise LoopIRInterpreterError(
                f"unresolved dimension extent for {name!r}: no bound tensor "
                "stores this dimension"
            )
        return known[0]

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

    def _eval_position(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIRInterpreterError(
                f"position expression produced {type(result).__name__}"
            )
        return result

    def _eval(self, expr: Expr) -> object:
        if type(expr) is IntConst:
            return expr.value
        if type(expr) is FloatConst:
            return expr.value
        if type(expr) is IndexValue:
            return self.indices[expr.index]
        if type(expr) is RootPosition:
            return 0
        if type(expr) is PositionValue:
            return self.positions[expr.position]
        if type(expr) is DensePosition:
            return self._eval_dense_position(expr)
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

    def _eval_dense_position(self, expr: DensePosition) -> int:
        parent = self._eval_position(expr.parent)
        decl = self.decls[expr.tensor]
        dimension = decl.dimensions[decl.levels[expr.level].mode]
        extent = self._dimension_extent(dimension)
        coord = self._eval_coord(expr.coord)
        if not 0 <= coord < extent:
            raise LoopIRInterpreterError(
                f"dense-level coordinate {coord} outside [0, {extent}) on "
                f"{decl.name}"
            )
        return parent * extent + coord

    def _segment(self, decl: SparseCursorDecl) -> Tuple[LevelTensorStorage, int, int]:
        storage = self.storages.get(decl.tensor)
        if storage is None:
            raise LoopIRInterpreterError(
                f"cursor tensor {self.decls[decl.tensor].name} has no level storage"
            )
        parent = self._eval_position(decl.parent)
        try:
            start, end = storage.segment(decl.level, parent)
        except LevelStorageError as error:
            raise LoopIRInterpreterError(
                f"cursor segment selection failed on "
                f"{self.decls[decl.tensor].name}: {error}"
            ) from error
        return storage, start, end

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
            extent = self._dimension_extent(stmt.dimension)
            try:
                for coordinate in range(extent):
                    self.indices[stmt.index] = coordinate
                    self._exec_block(stmt.body)
            finally:
                self.indices.pop(stmt.index, None)
            return
        if type(stmt) is SparseFor:
            storage, start, end = self._segment(stmt.cursor)
            state = _CursorState(storage, stmt.cursor.level, start, end, aligned=True)
            self.cursors[stmt.cursor.cursor] = state
            try:
                while not state.exhausted:
                    self.positions[stmt.position] = state.position
                    self.indices[stmt.coord_index] = state.coordinate
                    self._exec_block(stmt.body)
                    state.position += 1
            finally:
                self.indices.pop(stmt.coord_index, None)
                self.positions.pop(stmt.position, None)
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
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            target[last] = self._eval_value(stmt.value)
            return
        if type(stmt) is StoreReduce:
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            contribution = self._eval_value(stmt.value)
            if stmt.op is ReduceOp.ADD:
                target[last] = target[last] + contribution
            else:
                target[last] = target[last] * contribution
            return
        if type(stmt) is AppendEntry:
            coords_tuple = tuple(self._eval_coord(coord) for coord in stmt.coords)
            value = self._eval_value(stmt.value)
            try:
                self.builders[stmt.tensor].append(coords_tuple, value)
            except LevelStorageError as error:
                raise LoopIRInterpreterError(str(error)) from error
            return
        raise LoopIRInterpreterError(f"unknown statement {type(stmt).__name__}")

    def _locate_store(
        self, tensor: SymbolId, indices: Tuple[Expr, ...]
    ) -> Tuple[List[Any], int]:
        coords = [self._eval_coord(index) for index in indices]
        target: Any = self.values[tensor]
        for index in coords[:-1]:
            if not isinstance(target, list) or not 0 <= index < len(target):
                raise LoopIRInterpreterError(f"store index {index} out of bounds")
            target = target[index]
        last = coords[-1]
        if not isinstance(target, list) or not 0 <= last < len(target):
            raise LoopIRInterpreterError(f"store index {last} out of bounds")
        return target, last

    def _exec_merge(self, stmt: MergedSparseFor) -> None:
        states: List[_CursorState] = []
        try:
            for decl in stmt.cursors:
                storage, start, end = self._segment(decl)
                state = _CursorState(storage, decl.level, start, end, aligned=False)
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
