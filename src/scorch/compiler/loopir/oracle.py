"""Production-owned test/debug semantic oracle for the LoopIR subset.

This module promotes the Phase-3.5 spike interpreter's semantics into the
production tree for the migrated dense and sparse families.  It executes
verified LoopIR programs over plain Python containers with no Torch, native
code, or compiler-pipeline involvement, and is loaded only by dedicated
oracle and differential tests — never by ``import scorch``, default
compilation, legacy correctness paths, or release JIT.

Semantics (matching the node contracts exactly):

- shape compatibility is the logical dimension model: every bound tensor
  contributes the extent of each dimension its modes store, all extents must
  agree across inputs and outputs before anything executes, and ``DenseFor``
  resolves its trip count from the dimension extent;
- dense outputs are zero-initialized before the body runs — the explicit
  contract ``StoreReduce`` (ADD) depends on;
- sparse iteration executes over the format-neutral level interface of
  :mod:`~scorch.compiler.loopir.levels` (``segment`` / ``coordinate_at`` /
  ``leaf_value``): all-dense inputs bind nested sequences, inputs with a
  COMPRESSED level bind a :class:`LevelTensorStorage` (canonical CSR
  declarations also accept the :class:`CsrMatrix` adapter), and canonical
  CSR outputs are assembled through an order-checked append builder that
  returns a :class:`CsrMatrix`;
- all arithmetic is Python-float arithmetic in program order; the oracle is
  a semantic reference, not a bit-accuracy model of any particular scalar
  width, so numeric comparisons against compiled kernels use tolerances
  owned by the caller.

Everything unexpected fails closed with :class:`LoopIROracleError`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from ..identity import IndexId, SymbolId
from .levels import (
    CsrMatrix,
    CsrOutputBuilder,
    LevelStorageError,
    LevelTensorStorage,
    from_csr,
)
from .nodes import (
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PositionId,
    PositionValue,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)
from .verifier import verify_program

TensorValue = Any
MAX_ORACLE_RANK = 64

_CSR_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)
_CSR_MODES = (0, 1)


class LoopIROracleError(Exception):
    """Program execution hit a state the LoopIR oracle rejects."""


def _is_canonical_csr(decl: TensorDecl) -> bool:
    kinds = tuple(level.kind for level in decl.levels)
    modes = tuple(level.mode for level in decl.levels)
    return kinds == _CSR_KINDS and modes == _CSR_MODES


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


def _snapshot_dense(value: object, remaining_rank: int) -> object:
    """Own every sequence container the rank-bounded dense copy may inspect."""

    if remaining_rank == 0 or type(value) not in (list, tuple):
        return value
    owned = cast(Sequence[object], value)
    return [_snapshot_dense(entry, remaining_rank - 1) for entry in owned]


def _dense_shape(rank: int, value: object, trail: str) -> Tuple[Optional[int], ...]:
    """Validate a dense binding and infer every visible prefix extent.

    An empty sequence fixes its own extent at zero but carries no information
    about deeper modes.  Those suffix extents are resolved later through the
    program's shared :class:`DimensionId` bindings instead of rejecting a
    semantically valid zero-extent tensor prematurely.
    """

    shape: List[int] = []
    layer: object = value
    for _level in range(rank):
        if type(layer) not in (list, tuple):
            raise LoopIROracleError(f"{trail} must nest sequences to rank {rank}")
        owned_layer = cast(Sequence[object], layer)
        shape.append(len(owned_layer))
        if not owned_layer:
            return tuple([*shape, *([None] * (rank - len(shape)))])
        layer = owned_layer[0]
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
        for decl in program.tensors:
            rank = len(decl.levels)
            if rank > MAX_ORACLE_RANK:
                raise LoopIROracleError(
                    f"tensor {decl.name!r} rank {rank} exceeds the oracle "
                    f"limit {MAX_ORACLE_RANK}"
                )

        self.values: Dict[SymbolId, Any] = {}
        self.storages: Dict[SymbolId, LevelTensorStorage] = {}
        self.builders: Dict[SymbolId, CsrOutputBuilder] = {}
        self.shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        input_values: Dict[SymbolId, object] = {}
        partial_input_shapes: Dict[SymbolId, Tuple[Optional[int], ...]] = {}
        for symbol in program.inputs:
            decl = self.decls[symbol]
            try:
                bound = inputs[symbol]
            except Exception as error:
                raise LoopIROracleError(
                    "input bindings could not be snapshotted"
                ) from error
            if any(level.kind is not LevelKind.DENSE for level in decl.levels):
                self.storages[symbol] = self._bind_sparse_input(decl, bound)
                self.shapes[symbol] = self.storages[symbol].shape
                continue
            owned_bound = _snapshot_dense(bound, len(decl.levels))
            input_values[symbol] = owned_bound
            partial_input_shapes[symbol] = _dense_shape(
                len(decl.levels), owned_bound, f"input {decl.name}"
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
            if any(level.kind is not LevelKind.DENSE for level in decl.levels):
                # The verifier admits only canonical CSR sparse outputs.
                self.builders[symbol] = CsrOutputBuilder(decl.name, shape_binding)
            else:
                self.values[symbol] = _zeros(shape_binding) if shape_binding else []

        self.dim_extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        for symbol in program.inputs:
            decl = self.decls[symbol]
            if symbol in self.storages:
                for mode, extent in enumerate(self.shapes[symbol]):
                    self._bind_dimension_extent(
                        decl.dimensions[mode], extent, decl.name, mode
                    )
                continue
            for mode, partial_extent in enumerate(partial_input_shapes[symbol]):
                if partial_extent is not None:
                    self._bind_dimension_extent(
                        decl.dimensions[mode], partial_extent, decl.name, mode
                    )
        for symbol in program.outputs:
            decl = self.decls[symbol]
            for mode, extent in enumerate(self.shapes[symbol]):
                self._bind_dimension_extent(
                    decl.dimensions[mode], extent, decl.name, mode
                )

        for symbol in program.inputs:
            decl = self.decls[symbol]
            if symbol in self.storages:
                continue
            resolved_shape: List[int] = []
            for mode, partial_extent in enumerate(partial_input_shapes[symbol]):
                resolved_extent = partial_extent
                if resolved_extent is None:
                    known = self.dim_extents.get(decl.dimensions[mode])
                    if known is None:
                        raise LoopIROracleError(
                            f"input {decl.name} shape cannot be inferred at "
                            f"mode {mode}; bind another tensor sharing that "
                            "dimension or avoid an empty outer mode"
                        )
                    resolved_extent = known[0]
                resolved_shape.append(resolved_extent)
            shape = tuple(resolved_shape)
            self.shapes[symbol] = shape
            self.values[symbol] = _dense_copy(
                input_values[symbol], shape, f"input {decl.name}"
            )
        self.indices: Dict[IndexId, int] = {}
        self.positions: Dict[PositionId, int] = {}
        self.cursors: Dict[CursorId, _CursorState] = {}

    def _bind_sparse_input(self, decl: TensorDecl, bound: object) -> LevelTensorStorage:
        """Snapshot one compressed-layout input behind the level interface."""

        if type(bound) is CsrMatrix:
            if not _is_canonical_csr(decl):
                raise LoopIROracleError(
                    f"input {decl.name} is not a canonical CSR declaration; "
                    "bind a LevelTensorStorage instead"
                )
            try:
                return from_csr(bound)
            except LevelStorageError as error:
                raise LoopIROracleError(
                    f"input {decl.name} has invalid CSR storage: {error}"
                ) from error
        if type(bound) is not LevelTensorStorage:
            raise LoopIROracleError(
                f"input {decl.name} must be bound to a LevelTensorStorage"
                + (" or CsrMatrix" if _is_canonical_csr(decl) else "")
            )
        try:
            storage = bound.snapshot()
        except LevelStorageError as error:
            raise LoopIROracleError(
                f"input {decl.name} has invalid level storage: {error}"
            ) from error
        declared_kinds = tuple(level.kind for level in decl.levels)
        declared_modes = tuple(level.mode for level in decl.levels)
        if storage.kinds != declared_kinds or storage.modes != declared_modes:
            raise LoopIROracleError(
                f"input {decl.name} storage layout "
                f"({tuple(kind.value for kind in storage.kinds)}, "
                f"modes {storage.modes}) does not match its declaration "
                f"({tuple(kind.value for kind in declared_kinds)}, "
                f"modes {declared_modes})"
            )
        return storage

    def _bind_dimension_extent(
        self, dimension: DimensionId, extent: int, tensor_name: str, mode: int
    ) -> None:
        known = self.dim_extents.get(dimension)
        if known is None:
            self.dim_extents[dimension] = (extent, tensor_name, mode)
        elif known[0] != extent:
            name = self.dimension_names.get(dimension, f"<dimension {dimension.value}>")
            raise LoopIROracleError(
                f"dimension extent mismatch for {name!r}: "
                f"{known[1]}[{known[2]}] is {known[0]} but "
                f"{tensor_name}[{mode}] is {extent}"
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

    def _eval_position(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIROracleError(
                f"position expression produced {type(result).__name__}"
            )
        return result

    def _eval(self, expr: Expr) -> object:
        if type(expr) is IndexValue:
            return self.indices[expr.index]
        if type(expr) is FloatConst:
            return expr.value
        if type(expr) is RootPosition:
            return 0
        if type(expr) is PositionValue:
            return self.positions[expr.position]
        if type(expr) is DensePosition:
            parent = self._eval_position(expr.parent)
            decl = self.decls[expr.tensor]
            dimension = decl.dimensions[decl.levels[expr.level].mode]
            extent = self._dimension_extent(dimension)
            coord = self._eval_coord(expr.coord)
            if not 0 <= coord < extent:
                raise LoopIROracleError(
                    f"dense-level coordinate {coord} outside [0, {extent}) on "
                    f"{decl.name}"
                )
            return parent * extent + coord
        if type(expr) is CursorValue:
            state = self.cursors[expr.cursor]
            if state.aligned:
                return state.value
            if expr.default is None:
                raise LoopIROracleError("unaligned cursor read without a default")
            return self._eval_value(expr.default)
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

    def _segment(self, decl: SparseCursorDecl) -> Tuple[LevelTensorStorage, int, int]:
        storage = self.storages.get(decl.tensor)
        if storage is None:
            raise LoopIROracleError(
                f"cursor tensor {self.decls[decl.tensor].name} has no level storage"
            )
        parent = self._eval_position(decl.parent)
        try:
            start, end = storage.segment(decl.level, parent)
        except LevelStorageError as error:
            raise LoopIROracleError(
                f"cursor segment selection failed on "
                f"{self.decls[decl.tensor].name}: {error}"
            ) from error
        return storage, start, end

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
                    self._exec_stmt(stmt.body)
                for state in aligned:
                    state.position += 1
        finally:
            self.indices.pop(stmt.coord_index, None)
            for decl in stmt.cursors:
                self.cursors.pop(decl.cursor, None)

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
        if type(stmt) is SparseFor:
            storage, start, end = self._segment(stmt.cursor)
            state = _CursorState(storage, stmt.cursor.level, start, end, aligned=True)
            self.cursors[stmt.cursor.cursor] = state
            try:
                while not state.exhausted:
                    self.positions[stmt.position] = state.position
                    self.indices[stmt.coord_index] = state.coordinate
                    self._exec_stmt(stmt.body)
                    state.position += 1
            finally:
                self.indices.pop(stmt.coord_index, None)
                self.positions.pop(stmt.position, None)
                del self.cursors[stmt.cursor.cursor]
            return
        if type(stmt) is MergedSparseFor:
            self._exec_merge(stmt)
            return
        if type(stmt) is AppendEntry:
            coords_tuple = tuple(self._eval_coord(coord) for coord in stmt.coords)
            value = self._eval_value(stmt.value)
            builder = self.builders.get(stmt.tensor)
            if builder is None:
                raise LoopIROracleError(
                    f"append target {self.decls[stmt.tensor].name} has no "
                    "assembly builder"
                )
            try:
                builder.append(coords_tuple, value)
            except LevelStorageError as error:
                raise LoopIROracleError(str(error)) from error
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
