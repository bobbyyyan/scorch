"""LoopIR-to-structured-LLIR lowering for the migrated families.

The existing structured LLIR remains the target-specific CxxIR boundary; no
new target IR is introduced.  This module lowers one verified LoopIR
program into a complete LLIR ``evaluate`` function by reusing the exact
production target components the legacy path uses:

- :class:`~scorch.compiler.torch_cpp_abi.TorchCppKernelABI` /
  :class:`~scorch.compiler.torch_cpp_abi.KernelTensorABI` own the public
  signature, validation, and input prologue (dense extents plus compressed
  ``pos``/``crd`` pointer bindings);
- :class:`~scorch.compiler.torch_cpp_abi.ResultTensorAssembler` owns result
  storage initialization and final assembly for dense and canonical-CSR
  outputs;
- the managed production LLIR pass pipeline
  (:class:`~scorch.compiler.llir_pass_manager.LLIRPassManager`) applies the
  same typed optimization passes (sparse prefetch, dense pointer hoisting,
  single-iteration elimination, invariant hoisting, dynamic-vector
  rewriting, ...);
- :func:`~scorch.compiler.parallel_marking_pass.mark_first_for_loop_parallel`
  applies the same outer-loop parallel policy, under the same gate the
  legacy lowering uses (dense result written by the outer loop variable).

Because the raw loop-nest emission mirrors the legacy lowering
statement-for-statement — dense position chains, sparse position loops,
two-cursor coordinate merges with UNION tail loops, ordered CSR assembly
counters, and the affine-split origin/point loops (width constants, the
stepping origin loop, the reconstructed logical coordinate, and the
ragged-tail overshoot break) — the generated C++ for the migrated families
is byte-identical to the legacy pipeline's output; the differential suites
lock that equality.  LoopIR itself contains none of these target details —
this module is where target spelling begins.

Shapes are runtime bindings, not LoopIR content: callers pass concrete input
shapes and the result shape, and this boundary re-resolves every logical
dimension's extent across all of them, failing closed on any disagreement.

Fail-closed surface: this target lowering accepts the migrated program
shapes only.  Level-0 (root-parent) cursors, compressed-parent descent
(DCSR), merges of more than two cursors, non-innermost merges, merged
reductions, appends outside a dense-row/merged-column nest, nonzero merge
defaults, affine splits over merged iteration or ordered sparse assembly,
and any statement shape outside the single-nest single-leaf form fail with
:class:`LoopIRTargetError` and a stable code rather than being
approximated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Set, Tuple

import torch

from ...format import LevelType
from ..identity import AccessId, SymbolId, new_access_id
from .. import llir
from ..compile_options import CompileOptions
from ..compilation_context import CompilationContext, CompilerStageId
from ..dense_pointer_hoist_pass import DensePointerHoistContext
from ..llir_pass_manager import (
    DensePointerHoistPassSpec,
    LLIRPassPartialFailure,
    LLIRPassManager,
    LLIRRewriteArtifact,
    LLIRStatementListArtifact,
)
from ..parallel_marking_pass import (
    _CPP_KEYWORDS,
    EMPTY_PARALLEL_WORKSPACE_CLUSTER,
    apply_parallel_policy,
    mark_first_for_loop_parallel,
)
from ..torch_cpp_abi import (
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
)
from ...utils import dtype_to_c_datatype
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
    RootPosition,
    ScalarType,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
    TileInnerFor,
    TileOuterFor,
)
from .verifier import verify_program

_SCALAR_TO_TORCH: Dict[ScalarType, torch.dtype] = {
    ScalarType.FLOAT32: torch.float32,
    ScalarType.FLOAT64: torch.float64,
}

_BINARY_TO_CXX: Dict[BinaryOp, str] = {
    BinaryOp.ADD: "+",
    BinaryOp.SUB: "-",
    BinaryOp.MUL: "*",
}

_LEVEL_KIND_TO_LEVEL_TYPE: Dict[LevelKind, LevelType] = {
    LevelKind.DENSE: LevelType.DENSE,
    LevelKind.COMPRESSED: LevelType.COMPRESSED,
}

_CPP_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TARGET_RESERVED_NAMES = frozenset(
    {
        "Tensor",
        "evaluate",
        "result_shape",
        "scorch_chunk",
        "scorch_native",
        "scorch_nthreads",
        "scorch_tensor_from_vector",
        "scorch_vector_set",
        "scorch_zero_dense",
        "std",
        "torch",
    }
)

_DENSE = "dense"
_SPARSE = "sparse"
_MERGED = "merged"
_TILE_OUTER = "tile_outer"
_TILE_INNER = "tile_inner"


@dataclass(frozen=True)
class LoopIRTargetDefect:
    """One immutable target-lowering failure: stable code and message."""

    code: str
    message: str


class LoopIRTargetError(Exception):
    """A verified LoopIR program is outside this target lowering's surface."""

    def __init__(self, defect: LoopIRTargetDefect) -> None:
        super().__init__(f"{defect.code}: {defect.message}")
        self.defect = defect


def _fail(code: str, message: str) -> NoReturn:
    raise LoopIRTargetError(LoopIRTargetDefect(code, message))


@dataclass(frozen=True)
class _Loop:
    """One loop of the family nest, outermost first."""

    kind: str
    index: object
    dimension: DimensionId
    node: Any
    cursors: Tuple[SparseCursorDecl, ...]


class _TargetLowering:
    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self.decls: Dict[SymbolId, TensorDecl] = {
            decl.symbol: decl for decl in program.tensors
        }
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = all(
            level.kind is LevelKind.DENSE for level in self.result_decl.levels
        )
        self.sparse_program = any(
            level.kind is not LevelKind.DENSE
            for decl in program.tensors
            for level in decl.levels
        )
        self.dimension_names: Dict[DimensionId, str] = {}
        self._access_ids: Dict[SymbolId, AccessId] = {}
        self._validate_display_names()
        self._validate_layouts()
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self.loops = self._collect_loop_nest()
        self.loop_positions: Dict[object, int] = {
            loop.index: position for position, loop in enumerate(self.loops)
        }
        self.leaf = self._collect_leaf()
        self.cursor_loops: Dict[CursorId, int] = {}
        for position, loop in enumerate(self.loops):
            for cursor in loop.cursors:
                self.cursor_loops[cursor.cursor] = position
        self.loads, self.cursor_values = self._collect_accesses()
        self.level_drivers = self._compute_level_drivers()
        self._validate_access_orders()
        self._reserve_merge_names()
        self._reserve_tile_names()

    # -- boundary validation -------------------------------------------------

    def _validate_display_names(self) -> None:
        display_names: Dict[str, str] = {}
        for dimension_decl in self.program.dimensions:
            if (
                _CPP_IDENTIFIER.fullmatch(dimension_decl.name) is None
                or dimension_decl.name in _CPP_KEYWORDS
            ):
                _fail(
                    "invalid_display_name",
                    f"dimension name {dimension_decl.name!r} is not a safe "
                    "ASCII C++ identifier",
                )
            if dimension_decl.name in display_names:
                _fail(
                    "duplicate_display_name",
                    f"display name {dimension_decl.name!r} is used more " "than once",
                )
            display_names[dimension_decl.name] = "dimension"
            self.dimension_names[dimension_decl.dimension] = dimension_decl.name
        for decl in self.program.tensors:
            if (
                _CPP_IDENTIFIER.fullmatch(decl.name) is None
                or decl.name in _CPP_KEYWORDS
            ):
                _fail(
                    "invalid_display_name",
                    f"tensor name {decl.name!r} is not a safe ASCII C++ identifier",
                )
            if decl.name in display_names:
                _fail(
                    "duplicate_display_name",
                    f"display name {decl.name!r} is used more " "than once",
                )
            display_names[decl.name] = "tensor"

        generated: Dict[str, str] = {
            name: "the target runtime" for name in _TARGET_RESERVED_NAMES
        }

        def reserve(name: str, owner: str) -> None:
            known = generated.get(name)
            if known is not None:
                _fail(
                    "generated_name_collision",
                    f"generated C++ identifier {name!r} for {owner} conflicts "
                    f"with {known}",
                )
            generated[name] = owner

        self._reserve_generated_name = reserve

        for dimension_decl in self.program.dimensions:
            reserve(
                dimension_decl.name,
                f"dimension {dimension_decl.name!r}",
            )
        for symbol in self.program.inputs:
            decl = self.decls[symbol]
            owner = f"input tensor {decl.name!r}"
            for name in (
                decl.name,
                f"{decl.name}_shape",
                f"{decl.name}_mode_indices",
                f"{decl.name}_values",
                f"{decl.name}_val",
                f"_{decl.name}_val_ptr",
            ):
                reserve(name, owner)
            for level, level_decl in enumerate(decl.levels):
                reserve(f"{decl.name}{level}_size", owner)
                reserve(f"p{decl.name}{level}", owner)
                if level_decl.kind is LevelKind.COMPRESSED:
                    reserve(f"{decl.name}{level}_pos", owner)
                    reserve(f"{decl.name}{level}_crd", owner)
                    reserve(f"p{decl.name}{level}_end", owner)

        output = self.result_decl
        output_owner = f"output tensor {output.name!r}"
        for name in (
            output.name,
            f"{output.name}_capacity",
            f"{output.name}_values",
            f"{output.name}_values_torch",
        ):
            reserve(name, output_owner)
        for level, level_decl in enumerate(output.levels):
            reserve(f"{output.name}{level}_size", output_owner)
            reserve(f"p{output.name}{level}", output_owner)
            if level_decl.kind is LevelKind.COMPRESSED:
                reserve(f"{output.name}{level}_pos", output_owner)
                reserve(f"{output.name}{level}_crd", output_owner)
                reserve(f"{output.name}{level}_pos_index", output_owner)
                reserve(f"{output.name}{level}_pos_torch", output_owner)
                reserve(f"{output.name}{level}_crd_torch", output_owner)

    def _reserve_merge_names(self) -> None:
        """Reserve the per-cursor coordinate temporaries merges generate."""

        for loop in self.loops:
            if loop.kind is not _MERGED:
                continue
            dimension_name = self.dimension_names[loop.dimension]
            for cursor in loop.cursors:
                tensor_name = self.decls[cursor.tensor].name
                self._reserve_generated_name(
                    f"{dimension_name}_{tensor_name}",
                    f"merged coordinate of input tensor {tensor_name!r}",
                )

    def _reserve_tile_names(self) -> None:
        """Reserve the derived loop and width names affine splits generate."""

        for loop in self.loops:
            if loop.kind is not _TILE_OUTER:
                continue
            name = self.dimension_names[loop.dimension]
            owner = f"affine split of dimension {name!r}"
            for generated in (f"{name}_out", f"{name}_in", f"kTile_{name}"):
                self._reserve_generated_name(generated, owner)

    def _validate_layouts(self) -> None:
        if len(self.program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        for decl in self.program.tensors:
            # verify_program already rejected COORDINATE/SINGLETON level
            # kinds and non-CSR sparse outputs, so only the storage
            # permutation needs a target-boundary check.
            modes = tuple(level.mode for level in decl.levels)
            if modes != tuple(range(len(decl.levels))):
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} uses a non-identity storage order, "
                    "which the migrated families do not cover",
                )
        for symbol in self.program.inputs:
            decl = self.decls[symbol]
            compressed = [
                level
                for level, level_decl in enumerate(decl.levels)
                if level_decl.kind is LevelKind.COMPRESSED
            ]
            if not compressed:
                continue
            leaf = len(decl.levels) - 1
            if compressed != [leaf]:
                _fail(
                    "unsupported_program_shape",
                    f"input {decl.name!r} declares compressed structure "
                    "outside the value-bearing leaf level; hierarchical "
                    "compressed descent is outside the migrated families",
                )

    def _validate_shapes(
        self,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> Dict[SymbolId, Tuple[int, ...]]:
        try:
            input_keys = set(input_shapes)
        except Exception as error:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    "invalid_shape_binding",
                    "input shapes could not be snapshotted",
                )
            ) from error
        if input_keys != set(self.program.inputs):
            _fail(
                "invalid_shape_binding",
                "input shapes must cover exactly the declared inputs",
            )
        shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        for symbol in self.program.inputs:
            decl = self.decls[symbol]
            try:
                shape = input_shapes[symbol]
            except Exception as error:
                raise LoopIRTargetError(
                    LoopIRTargetDefect(
                        "invalid_shape_binding",
                        "input shapes could not be snapshotted",
                    )
                ) from error
            if (
                type(shape) is not tuple
                or len(shape) != len(decl.levels)
                or any(type(extent) is not int or extent < 0 for extent in shape)
            ):
                _fail(
                    "invalid_shape_binding",
                    f"input {decl.name!r} needs a rank-{len(decl.levels)} "
                    "shape of nonnegative ints",
                )
            shapes[symbol] = shape
        if (
            type(result_shape) is not tuple
            or len(result_shape) != len(self.result_decl.levels)
            or any(type(extent) is not int or extent < 0 for extent in result_shape)
        ):
            _fail(
                "invalid_shape_binding",
                f"result {self.result_decl.name!r} needs a rank-"
                f"{len(self.result_decl.levels)} shape of nonnegative ints",
            )
        shapes[self.result_symbol] = result_shape

        extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        for decl in self.program.tensors:
            shape = shapes[decl.symbol]
            for mode, dimension in enumerate(decl.dimensions):
                known = extents.get(dimension)
                if known is None:
                    extents[dimension] = (shape[mode], decl.name, mode)
                elif known[0] != shape[mode]:
                    _fail(
                        "dimension_extent_mismatch",
                        f"dimension "
                        f"{self.dimension_names[dimension]!r}: "
                        f"{known[1]}[{known[2]}] is {known[0]} but "
                        f"{decl.name}[{mode}] is {shape[mode]}",
                    )
        return shapes

    def _level_dimension(self, symbol: SymbolId, level: int) -> DimensionId:
        decl = self.decls[symbol]
        return decl.dimensions[decl.levels[level].mode]

    def _collect_loop_nest(self) -> List[_Loop]:
        loops: List[_Loop] = []
        body: Stmt = self.program.body
        while True:
            if type(body) is not Block or len(body.statements) != 1:
                _fail(
                    "unsupported_program_shape",
                    "this target lowering expects a single-statement loop "
                    "nest over one store leaf",
                )
            only = body.statements[0]
            if type(only) is DenseFor:
                loops.append(_Loop(_DENSE, only.index, only.dimension, only, ()))
                body = only.body
                continue
            if type(only) is TileOuterFor:
                # The origin loop binds no readable coordinate, so its
                # position key is a sentinel that can never collide with a
                # logical IndexId; the logical index stays reachable through
                # the node for bound resolution and the parallel gate.
                loops.append(
                    _Loop(
                        _TILE_OUTER,
                        ("tile_outer", only.tile),
                        only.dimension,
                        only,
                        (),
                    )
                )
                body = only.body
                continue
            if type(only) is TileInnerFor:
                loops.append(_Loop(_TILE_INNER, only.index, only.dimension, only, ()))
                body = only.body
                continue
            if type(only) is SparseFor:
                cursor = only.cursor
                loops.append(
                    _Loop(
                        _SPARSE,
                        only.coord_index,
                        self._level_dimension(cursor.tensor, cursor.level),
                        only,
                        (cursor,),
                    )
                )
                body = only.body
                continue
            if type(only) is MergedSparseFor:
                first = only.cursors[0]
                loops.append(
                    _Loop(
                        _MERGED,
                        only.coord_index,
                        self._level_dimension(first.tensor, first.level),
                        only,
                        tuple(only.cursors),
                    )
                )
                body = only.body
                continue
            if type(only) in (Store, StoreReduce, AppendEntry):
                if not loops:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering requires at least one loop",
                    )
                self._validate_loop_kinds(loops, only)
                return loops
            _fail(
                "unsupported_program_shape",
                f"unsupported nest statement {type(only).__name__}",
            )

    def _validate_loop_kinds(self, loops: List[_Loop], leaf: Stmt) -> None:
        if loops[0].kind not in (_DENSE, _TILE_OUTER):
            _fail(
                "unsupported_program_shape",
                "the migrated families require a dense or tile-origin "
                "outermost loop",
            )
        tile_positions = [
            position
            for position, loop in enumerate(loops)
            if loop.kind in (_TILE_OUTER, _TILE_INNER)
        ]
        if tile_positions:
            if any(loop.kind is _MERGED for loop in loops):
                _fail(
                    "unsupported_program_shape",
                    "affine splits over merged iteration are outside the "
                    "migrated schedule families",
                )
            if type(leaf) is AppendEntry:
                _fail(
                    "unsupported_program_shape",
                    "affine splits over ordered sparse assembly are outside "
                    "the migrated schedule families",
                )
        merged_positions = [
            position for position, loop in enumerate(loops) if loop.kind is _MERGED
        ]
        if merged_positions:
            if merged_positions != [len(loops) - 1]:
                _fail(
                    "unsupported_program_shape",
                    "a merged loop is supported only as the innermost loop",
                )
            if len(loops[-1].cursors) != 2:
                _fail(
                    "unsupported_program_shape",
                    "this target lowering merges exactly two sparse cursors",
                )
            if type(leaf) is StoreReduce:
                _fail(
                    "unsupported_program_shape",
                    "merged reductions are outside the migrated families",
                )
        elif type(leaf) is AppendEntry:
            _fail(
                "unsupported_program_shape",
                "ordered sparse assembly requires a merged innermost loop "
                "in the migrated families",
            )
        for loop in loops:
            for cursor in loop.cursors:
                if cursor.level < 1:
                    _fail(
                        "unsupported_program_shape",
                        "level-0 (root-parent) cursors are outside the "
                        "migrated families",
                    )

    def _collect_leaf(self) -> Stmt:
        innermost = self.loops[-1].node.body
        return innermost.statements[0]

    def _index_of(self, expr: Expr, path: str) -> object:
        if type(expr) is not IndexValue:
            _fail(
                "unsupported_program_shape",
                f"{path} must be a directly bound loop coordinate",
            )
        return expr.index

    def _collect_accesses(self) -> Tuple[List[Load], List[CursorValue]]:
        loads: List[Load] = []
        cursor_values: List[CursorValue] = []

        def walk(expr: Expr) -> None:
            if type(expr) is Load:
                loads.append(expr)
                return
            if type(expr) is CursorValue:
                cursor_values.append(expr)
                return
            if type(expr) is BinaryExpr:
                walk(expr.lhs)
                walk(expr.rhs)
                return
            if type(expr) is IndexValue:
                _fail(
                    "unsupported_program_shape",
                    "coordinate values are not value expressions in the "
                    "migrated families",
                )
            _fail(
                "unsupported_program_shape",
                f"unsupported value expression {type(expr).__name__}",
            )

        walk(self.leaf.value)  # type: ignore[attr-defined]
        seen: Set[SymbolId] = set()
        for load in loads:
            if load.tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor {self.decls[load.tensor].name!r} is loaded "
                    "more than once; this target owns one physical "
                    "position chain per input",
                )
            seen.add(load.tensor)
        for cursor_value in cursor_values:
            loop_position = self.cursor_loops.get(cursor_value.cursor)
            if loop_position is None:
                _fail(
                    "unsupported_program_shape",
                    "a cursor value must read a nest loop cursor",
                )
            cursor = self._cursor_decl(cursor_value.cursor)
            if cursor.tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor {self.decls[cursor.tensor].name!r} is read "
                    "more than once; this target owns one physical "
                    "position chain per input",
                )
            seen.add(cursor.tensor)
        return loads, cursor_values

    def _cursor_decl(self, cursor: CursorId) -> SparseCursorDecl:
        position = self.cursor_loops[cursor]
        for decl in self.loops[position].cursors:
            if decl.cursor == cursor:
                return decl
        raise AssertionError("unreachable")

    def _compute_level_drivers(self) -> Dict[SymbolId, Dict[int, object]]:
        """Map every tensor's physical level to the loop index driving it."""

        drivers: Dict[SymbolId, Dict[int, object]] = {}

        def record(symbol: SymbolId, level: int, index: object, what: str) -> None:
            per_tensor = drivers.setdefault(symbol, {})
            known = per_tensor.get(level)
            if known is not None and known != index:
                _fail(
                    "unsupported_program_shape",
                    f"{what} drives level {level} of "
                    f"{self.decls[symbol].name!r} with conflicting loop "
                    "coordinates",
                )
            if index not in self.loop_positions:
                _fail(
                    "unsupported_program_shape",
                    "access coordinates must be nest loop variables",
                )
            per_tensor[level] = index

        def record_parent_chain(cursor: SparseCursorDecl) -> None:
            parent: Expr = cursor.parent
            level = cursor.level - 1
            while True:
                if type(parent) is RootPosition:
                    if level != -1:
                        _fail(
                            "unsupported_program_shape",
                            "a cursor parent chain must ground at the root "
                            "below level 0",
                        )
                    return
                if type(parent) is DensePosition:
                    if parent.tensor != cursor.tensor or parent.level != level:
                        _fail(
                            "unsupported_program_shape",
                            "cursor parent chains must walk this tensor's "
                            "levels in order",
                        )
                    coord = self._index_of(
                        parent.coord,
                        f"the level-{level} parent coordinate of "
                        f"{self.decls[cursor.tensor].name!r}",
                    )
                    record(
                        cursor.tensor,
                        level,
                        coord,
                        "a cursor parent chain",
                    )
                    parent = parent.parent
                    level -= 1
                    continue
                _fail(
                    "unsupported_program_shape",
                    "cursor parents must be dense-position chains over the "
                    "root in the migrated families",
                )

        for load in self.loads:
            for level, index in enumerate(load.indices):
                bound = self._index_of(
                    index, f"access of {self.decls[load.tensor].name!r}"
                )
                record(load.tensor, level, bound, "a coordinate load")
        for position, loop in enumerate(self.loops):
            for cursor in loop.cursors:
                record(cursor.tensor, cursor.level, loop.index, "a sparse loop")
                record_parent_chain(cursor)
        leaf_indices = self._leaf_indices()
        for level, index in enumerate(leaf_indices):
            bound = self._index_of(index, f"access of {self.result_decl.name!r}")
            record(self.result_symbol, level, bound, "the result access")
        return drivers

    def _leaf_indices(self) -> Tuple[Expr, ...]:
        if type(self.leaf) is AppendEntry:
            return self.leaf.coords
        return self.leaf.indices  # type: ignore[attr-defined]

    def _validate_access_orders(self) -> None:
        for symbol, per_tensor in self.level_drivers.items():
            decl = self.decls[symbol]
            if sorted(per_tensor) != list(range(len(decl.levels))):
                _fail(
                    "unsupported_program_shape",
                    f"tensor {decl.name!r} has levels with no driving loop "
                    "coordinate",
                )
            positions = [
                self.loop_positions[per_tensor[level]]
                for level in range(len(decl.levels))
            ]
            if positions != sorted(positions) or len(set(positions)) != len(positions):
                _fail(
                    "unsupported_loop_order",
                    f"tensor {decl.name!r} storage order "
                    "conflicts with the loop nest order",
                )
        leaf_indices = self._leaf_indices()
        leaf_index_ids = [
            self._index_of(index, "the result access") for index in leaf_indices
        ]
        if type(self.leaf) is AppendEntry:
            nest_ids = [loop.index for loop in self.loops]
            if leaf_index_ids != nest_ids:
                _fail(
                    "unsupported_program_shape",
                    "ordered sparse assembly requires the nest loops to be "
                    "exactly the appended coordinates, in order",
                )
        elif type(self.leaf) is Store and any(
            loop.kind is _MERGED for loop in self.loops
        ):
            if set(leaf_index_ids) != {loop.index for loop in self.loops}:
                _fail(
                    "unsupported_program_shape",
                    "a merged store must write every nest coordinate",
                )
        # Inputs not read by the leaf value cannot anchor a position chain.
        read_symbols = {load.tensor for load in self.loads} | {
            self._cursor_decl(cursor_value.cursor).tensor
            for cursor_value in self.cursor_values
        }
        for symbol in self.program.inputs:
            if symbol not in read_symbols and self.decls[symbol].levels:
                if symbol not in self.level_drivers:
                    _fail(
                        "unsupported_program_shape",
                        f"input {self.decls[symbol].name!r} is never read",
                    )

    # -- emission ------------------------------------------------------------

    def _loop_var_name(self, loop: _Loop) -> str:
        return self.dimension_names[loop.dimension]

    def _access_id(self, symbol: SymbolId) -> AccessId:
        known = self._access_ids.get(symbol)
        if known is None:
            known = new_access_id()
            self._access_ids[symbol] = known
        return known

    def _input_metadata(self, symbol: SymbolId) -> llir.TensorAccessMetadata:
        per_tensor = self.level_drivers[symbol]
        index_ids = tuple(
            per_tensor[level] for level in range(len(self.decls[symbol].levels))
        )
        return llir.TensorAccessMetadata(
            access_id=self._access_id(symbol),
            tensor_id=symbol,
            index_ids=index_ids,  # type: ignore[arg-type]
            role=llir.TensorAccessRole.INPUT_READ,
        )

    def _result_metadata(self) -> llir.TensorAccessMetadata:
        leaf_indices = self._leaf_indices()
        return llir.TensorAccessMetadata(
            access_id=self._access_id(self.result_symbol),
            tensor_id=self.result_symbol,
            index_ids=tuple(
                self._index_of(index, "store index")  # type: ignore[misc]
                for index in leaf_indices
            ),
            role=llir.TensorAccessRole.RESULT_WRITE,
        )

    def _loop_logical_index(self, loop: _Loop) -> object:
        if loop.kind in (_TILE_OUTER, _TILE_INNER):
            return loop.node.index
        return loop.index

    def _input_bound_var(self, loop: _Loop) -> Optional[llir.Var]:
        lookup = self._loop_logical_index(loop)
        for symbol in self.program.inputs:
            decl = self.decls[symbol]
            per_tensor = self.level_drivers.get(symbol, {})
            for level in range(len(decl.levels)):
                if (
                    per_tensor.get(level) == lookup
                    and decl.levels[level].kind is LevelKind.DENSE
                ):
                    return llir.Var(
                        name=f"{decl.name}{level}_size", type=llir.DataType.INT64
                    )
        return None

    def _loop_bound_var(self, loop: _Loop) -> llir.Var:
        bound = self._input_bound_var(loop)
        if bound is not None:
            return bound
        leaf_indices = self._leaf_indices()
        lookup = self._loop_logical_index(loop)
        for level, index in enumerate(leaf_indices):
            if self._index_of(index, "store index") == lookup:
                return llir.Var(
                    name=f"{self.result_decl.name}{level}_size",
                    type=llir.DataType.INT64,
                )
        _fail("unsupported_program_shape", "a loop variable is never used")
        raise AssertionError("unreachable")

    def _tile_bound_var(self, loop: _Loop) -> llir.Var:
        """The dimension-size spelling that bounds one affine split.

        The legacy lattice resolves this bound from the first dense access
        containing the split variable — inputs first, and the result access
        for a broadcast coordinate the result alone drives.  That is exactly
        the shared loop-bound policy, so the origin loop's bound, the point
        loop's overshoot guard, and the derived parallel trip count all use
        one spelling.
        """

        return self._loop_bound_var(loop)

    def _position_init(
        self,
        tensor_name: str,
        level: int,
        loop: _Loop,
        result_chain: bool = False,
    ) -> llir.VarInit:
        loop_var = llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT64)
        if level == 0:
            value: llir.Expr = loop_var
        elif result_chain and self.sparse_program:
            # The legacy sparse lowering builds result position chains with
            # plain binary nodes; input chains keep the Add/Mul forms.  The
            # rendered C++ is identical either way — this mirrors the legacy
            # raw statements exactly so every managed pass sees the same
            # tree shapes it sees on the legacy path.
            value = llir.BinOp(
                op="+",
                left=llir.BinOp(
                    op="*",
                    left=llir.Var(
                        name=f"p{tensor_name}{level - 1}",
                        type=llir.DataType.INT,
                    ),
                    right=llir.Var(
                        name=f"{tensor_name}{level}_size",
                        type=llir.DataType.INT,
                    ),
                ),
                right=llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT),
            )
        else:
            value = llir.Add(
                left=llir.Mul(
                    left=llir.Var(
                        name=f"p{tensor_name}{level - 1}",
                        type=llir.DataType.INT64,
                    ),
                    right=llir.Var(
                        name=f"{tensor_name}{level}_size",
                        type=llir.DataType.INT64,
                    ),
                ),
                right=loop_var,
            )
        return llir.VarInit(
            var=llir.Var(name=f"p{tensor_name}{level}", type=llir.DataType.INT),
            value=value,
        )

    def _input_resolves_at(self, loop: _Loop) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        for symbol in self.program.inputs:
            decl = self.decls[symbol]
            per_tensor = self.level_drivers.get(symbol, {})
            for level in range(len(decl.levels)):
                if per_tensor.get(level) != loop.index:
                    continue
                if decl.levels[level].kind is not LevelKind.DENSE:
                    continue
                stmts.append(self._position_init(decl.name, level, loop))
        return stmts

    def _result_resolves_at(self, loop: _Loop) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        leaf_indices = self._leaf_indices()
        for level, index in enumerate(leaf_indices):
            if self._index_of(index, "store index") != loop.index:
                continue
            if self.result_decl.levels[level].kind is not LevelKind.DENSE:
                continue
            stmts.append(
                self._position_init(
                    self.result_decl.name, level, loop, result_chain=True
                )
            )
        return stmts

    def _lower_value(self, expr: Expr) -> llir.Expr:
        if type(expr) is Load:
            decl = self.decls[expr.tensor]
            if len(expr.indices) == 1:
                bound = self._index_of(expr.indices[0], "load index")
                position = self.loops[self.loop_positions[bound]]
                physical_index = self._loop_var_name(position)
            else:
                physical_index = f"p{decl.name}{len(expr.indices) - 1}"
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(name=physical_index, type=llir.DataType.INT),
                tensor_access=self._input_metadata(expr.tensor),
            )
        if type(expr) is CursorValue:
            cursor = self._cursor_decl(expr.cursor)
            loop = self.loops[self.cursor_loops[expr.cursor]]
            if loop.kind is not _SPARSE:
                _fail(
                    "unsupported_program_shape",
                    "merged cursor values are lowered per alignment case, "
                    "not as direct value expressions",
                )
            decl = self.decls[cursor.tensor]
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(
                    name=f"p{decl.name}{cursor.level}", type=llir.DataType.INT
                ),
                tensor_access=self._input_metadata(cursor.tensor),
            )
        if type(expr) is BinaryExpr:
            return llir.BinOp(
                op=_BINARY_TO_CXX[expr.op],
                left=self._lower_value(expr.lhs),
                right=self._lower_value(expr.rhs),
            )
        _fail(
            "unsupported_program_shape",
            f"unsupported value expression {type(expr).__name__}",
        )
        raise AssertionError("unreachable")

    def _lower_leaf(self) -> List[llir.Stmt]:
        leaf = self.leaf
        leaf_indices = leaf.indices  # type: ignore[attr-defined]
        rhs = self._lower_value(leaf.value)  # type: ignore[attr-defined]
        target = llir.ArrayAccess(
            array=llir.Var(
                name=f"{self.result_decl.name}_values",
                type=llir.DataType.NO_TYPE,
            ),
            index=llir.Var(
                name=f"p{self.result_decl.name}{len(leaf_indices) - 1}",
                type=llir.DataType.INT64,
            ),
            tensor_access=self._result_metadata(),
        )
        if type(leaf) is StoreReduce:
            return [llir.Assign(var=target, value=rhs, op=llir.AssignOp.ADD_ASSIGN)]
        return [llir.Assign(var=target, value=rhs)]

    # -- merged-loop case machinery -------------------------------------------

    def _merged_case_value(
        self, expr: Expr, aligned: Set[CursorId]
    ) -> Optional[llir.Expr]:
        """Partially evaluate the leaf value for one cursor-alignment case.

        Returns ``None`` when the case value folds to the additive identity
        (nothing is emitted for that case, exactly as the legacy iteration
        lattice drops it).
        """

        if type(expr) is CursorValue:
            cursor = self._cursor_decl(expr.cursor)
            if expr.cursor in aligned:
                decl = self.decls[cursor.tensor]
                torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
                return llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{decl.name}_val",
                        type=llir.DataType.ptr_type(torch_dtype),
                    ),
                    index=llir.Var(
                        name=f"p{decl.name}{cursor.level}",
                        type=llir.DataType.INT,
                    ),
                    tensor_access=self._input_metadata(cursor.tensor),
                )
            default = expr.default
            if type(default) is not FloatConst or default.value != 0.0:
                _fail(
                    "unsupported_union_default",
                    "the migrated UNION families require the additive "
                    "identity 0.0 as the unaligned-cursor default",
                )
            return None
        if type(expr) is Load:
            return self._lower_value(expr)
        if type(expr) is BinaryExpr:
            left = self._merged_case_value(expr.lhs, aligned)
            right = self._merged_case_value(expr.rhs, aligned)
            if expr.op is BinaryOp.ADD:
                if left is None:
                    return right
                if right is None:
                    return left
                return llir.BinOp(op="+", left=left, right=right)
            if expr.op is BinaryOp.MUL:
                if left is None or right is None:
                    return None
                return llir.BinOp(op="*", left=left, right=right)
            _fail(
                "unsupported_program_shape",
                "subtraction over merged sparse operands is outside the "
                "migrated families",
            )
        _fail(
            "unsupported_program_shape",
            f"unsupported merged value expression {type(expr).__name__}",
        )
        raise AssertionError("unreachable")

    def _merged_case_stmts(
        self, loop: _Loop, aligned: Set[CursorId]
    ) -> Optional[List[llir.Stmt]]:
        value = self._merged_case_value(
            self.leaf.value, aligned  # type: ignore[attr-defined]
        )
        if value is None:
            return None
        dimension_name = self._loop_var_name(loop)
        result_name = self.result_decl.name
        leaf_level = len(self.result_decl.levels) - 1
        target = llir.ArrayAccess(
            array=llir.Var(
                name=f"{result_name}_values",
                type=llir.DataType.NO_TYPE,
            ),
            index=llir.Var(
                name=f"p{result_name}{leaf_level}",
                type=llir.DataType.INT64,
            ),
            tensor_access=self._result_metadata(),
        )
        stmts: List[llir.Stmt] = [llir.Assign(var=target, value=value)]
        if type(self.leaf) is AppendEntry:
            stmts.append(llir.Comment("Set coordinates"))
            stmts.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_name}{leaf_level}_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=f"p{result_name}{leaf_level}",
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.Var(name=dimension_name, type=llir.DataType.NO_TYPE),
                )
            )
            stmts.append(
                llir.Increment(
                    var=llir.Var(
                        name=f"p{result_name}{leaf_level}",
                        type=llir.DataType.INT64,
                    )
                )
            )
        return stmts

    def _merged_coordinate_name(self, loop: _Loop, cursor: SparseCursorDecl) -> str:
        return f"{self._loop_var_name(loop)}_{self.decls[cursor.tensor].name}"

    def _cursor_position_name(self, cursor: SparseCursorDecl) -> str:
        return f"p{self.decls[cursor.tensor].name}{cursor.level}"

    def _cursor_pos_array(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"{name}{cursor.level}_pos", type=llir.DataType.PTR_INT)

    def _cursor_crd_array(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"{name}{cursor.level}_crd", type=llir.DataType.PTR_INT)

    def _cursor_parent_var(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"p{name}{cursor.level - 1}", type=llir.DataType.INT)

    def _iterator_inits(self, loop: _Loop) -> List[llir.Stmt]:
        """The legacy ``Initialize iterators`` group for one sparse loop."""

        stmts: List[llir.Stmt] = []
        if loop.kind is _SPARSE:
            cursor = loop.cursors[0]
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self._cursor_position_name(cursor)}_end",
                        type=llir.DataType.INT,
                    ),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=llir.Add(
                            left=self._cursor_parent_var(cursor),
                            right=llir.Literal(1, llir.DataType.INT),
                        ),
                    ),
                )
            )
            return stmts
        for cursor in loop.cursors:
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=self._cursor_position_name(cursor),
                        type=llir.DataType.INT,
                    ),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=self._cursor_parent_var(cursor),
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self._cursor_position_name(cursor)}_end",
                        type=llir.DataType.INT,
                    ),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=llir.Add(
                            left=self._cursor_parent_var(cursor),
                            right=llir.Literal(1, llir.DataType.INT),
                        ),
                    ),
                )
            )
        return stmts

    def _result_pos_set(self) -> llir.Assign:
        """``C1_pos[C1_pos_index + 1] = C1_crd.size()`` (legacy spelling)."""

        result_name = self.result_decl.name
        leaf_level = len(self.result_decl.levels) - 1
        return llir.Assign(
            var=llir.ArrayAccess(
                array=llir.Var(
                    name=f"{result_name}{leaf_level}_pos",
                    type=llir.DataType.STD_VECTOR_C_INT,
                ),
                index=llir.Add(
                    llir.Var(
                        name=f"{result_name}{leaf_level}_pos_index",
                        type=llir.DataType.INT,
                    ),
                    llir.Literal(1, llir.DataType.INT32),
                ),
            ),
            value=llir.FunctionCall(
                name=f"{result_name}{leaf_level}_crd.size",
                args=[],
            ),
        )

    def _assembly_catch_up(self, loop: _Loop) -> llir.ForLoop:
        """The legacy per-row ``Assemble COMPRESSED level`` catch-up loop."""

        result_name = self.result_decl.name
        leaf_level = len(self.result_decl.levels) - 1
        pos_index = llir.Var(
            name=f"{result_name}{leaf_level}_pos_index",
            type=llir.DataType.INT,
        )
        return llir.ForLoop(
            init=None,
            cond=llir.BinOp(
                op="<",
                left=pos_index,
                right=llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(
                    name=f"{result_name}{leaf_level}_pos_index",
                    type=llir.DataType.INT,
                )
            ),
            body=[self._result_pos_set()],
        )

    def _lower_merged(self, position: int) -> List[llir.Stmt]:
        loop = self.loops[position]
        node = loop.node
        dimension_name = self._loop_var_name(loop)
        cursors = loop.cursors
        result_resolves = self._result_resolves_at(loop)

        def coordinate_loads() -> List[llir.Stmt]:
            stmts: List[llir.Stmt] = [llir.Comment("Load coordinates")]
            for cursor in cursors:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=self._merged_coordinate_name(loop, cursor),
                            type=llir.DataType.INT,
                        ),
                        value=llir.ArrayAccess(
                            array=self._cursor_crd_array(cursor),
                            index=llir.Var(
                                name=self._cursor_position_name(cursor),
                                type=llir.DataType.INT,
                            ),
                        ),
                    )
                )
            return stmts

        def coordinate_var(cursor: SparseCursorDecl) -> llir.Var:
            return llir.Var(
                name=self._merged_coordinate_name(loop, cursor),
                type=llir.DataType.INT,
            )

        def dimension_var() -> llir.Var:
            return llir.Var(name=dimension_name, type=llir.DataType.INT)

        def aligned_guard(cursor: SparseCursorDecl) -> llir.Expr:
            return llir.BinOp(
                op="==", left=coordinate_var(cursor), right=dimension_var()
            )

        # Alignment cases in the legacy lattice order: all cursors aligned
        # first, then each single-cursor case for UNION merges.
        cases: List[Tuple[Set[CursorId], llir.Expr]] = []
        both_guard = llir.BinOp(
            op="&&",
            left=aligned_guard(cursors[0]),
            right=aligned_guard(cursors[1]),
        )
        cases.append(({cursor.cursor for cursor in cursors}, both_guard))
        if node.mode is MergeMode.UNION:
            for cursor in cursors:
                cases.append(({cursor.cursor}, aligned_guard(cursor)))

        cond_list: List[llir.Expr] = []
        then_body_list: List[List[llir.Stmt]] = []
        for aligned, guard in cases:
            case_stmts = self._merged_case_stmts(loop, aligned)
            if case_stmts is None:
                continue
            cond_list.append(guard)
            then_body_list.append(case_stmts)
        if not cond_list:
            _fail(
                "unsupported_program_shape",
                "a merged loop must emit at least one alignment case",
            )

        while_body: List[llir.Stmt] = [
            *coordinate_loads(),
            llir.BlankLine(),
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                value=llir.FunctionCall(
                    name="std::min",
                    args=[
                        llir.Array(
                            values=tuple(coordinate_var(cursor) for cursor in cursors),
                            data_type=llir.DataType.INT,
                        )
                    ],
                ),
            ),
            llir.BlankLine(),
        ]
        input_resolves = self._input_resolves_at(loop)
        if input_resolves:
            while_body.append(llir.Comment("Resolve dense coordinates"))
            while_body.extend(input_resolves)
        if result_resolves:
            while_body.append(
                llir.Comment("Resolve index into dense level of values array")
            )
            while_body.extend(result_resolves)
        while_body.append(llir.Comment("Inner loops over child regions"))
        while_body.append(
            llir.IfThenElse(
                cond_list=cond_list,
                then_body_list=then_body_list,
                make_last_case_else=False,
            )
        )
        while_body.append(llir.BlankLine())
        while_body.append(llir.Comment("Advance iterators"))
        for cursor in cursors:
            while_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=self._cursor_position_name(cursor),
                        type=llir.DataType.INT,
                    ),
                    value=llir.BinOp(
                        op="==",
                        left=coordinate_var(cursor),
                        right=dimension_var(),
                    ),
                    op=llir.AssignOp.ADD_ASSIGN,
                    cast=True,
                )
            )

        def position_cond(cursor: SparseCursorDecl) -> llir.Expr:
            return llir.BinOp(
                op="<",
                left=llir.Var(
                    name=self._cursor_position_name(cursor),
                    type=llir.DataType.INT,
                ),
                right=llir.Var(
                    name=f"{self._cursor_position_name(cursor)}_end",
                    type=llir.DataType.INT,
                ),
            )

        merge_loop = llir.WhileLoop(
            cond=llir.BinOp(
                op="&&",
                left=position_cond(cursors[0]),
                right=position_cond(cursors[1]),
            ),
            body=while_body,
        )
        merge_loop.scorch_index_var = dimension_name

        stmts: List[llir.Stmt] = [
            llir.Comment("Initialize iterators"),
            *self._iterator_inits(loop),
            llir.BlankLine(),
            merge_loop,
        ]

        if node.mode is MergeMode.UNION:
            for cursor in cursors:
                tail_case = self._merged_case_stmts(loop, {cursor.cursor})
                if tail_case is None:
                    continue
                tail_body: List[llir.Stmt] = [
                    llir.Comment("Resolve coordinates"),
                    llir.VarInit(
                        var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                        value=llir.ArrayAccess(
                            array=self._cursor_crd_array(cursor),
                            index=llir.Var(
                                name=self._cursor_position_name(cursor),
                                type=llir.DataType.INT,
                            ),
                        ),
                    ),
                    llir.BlankLine(),
                ]
                tail_input_resolves = self._input_resolves_at(loop)
                if tail_input_resolves:
                    tail_body.append(llir.Comment("Resolve dense coordinates"))
                    tail_body.extend(tail_input_resolves)
                tail_resolves = self._result_resolves_at(loop)
                if tail_resolves:
                    tail_body.append(
                        llir.Comment("Resolve index into dense level of values array")
                    )
                    tail_body.extend(tail_resolves)
                tail_body.extend(tail_case)
                tail_body.append(llir.Comment("Advance iterator"))
                tail_body.append(
                    llir.Increment(
                        var=llir.Var(
                            name=self._cursor_position_name(cursor),
                            type=llir.DataType.INT,
                        )
                    )
                )
                stmts.append(llir.WhileLoop(cond=position_cond(cursor), body=tail_body))
        return stmts

    def _lower_sparse(self, position: int) -> List[llir.Stmt]:
        loop = self.loops[position]
        cursor = loop.cursors[0]
        dimension_name = self._loop_var_name(loop)
        position_name = self._cursor_position_name(cursor)

        body: List[llir.Stmt] = [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
        ]
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        if input_resolves:
            body.append(llir.Comment("Resolve dense coordinates"))
            body.extend(input_resolves)
        if result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))

        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        for_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._cursor_parent_var(cursor),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        for_loop.scorch_index_var = dimension_name
        return [
            llir.Comment("Initialize iterators"),
            *self._iterator_inits(loop),
            llir.BlankLine(),
            for_loop,
        ]

    def _loop_children(self, position: int) -> List[llir.Stmt]:
        """Child-loop or leaf statements appended inside one loop's body."""

        if position + 1 < len(self.loops):
            child = self.loops[position + 1]
            if child.kind is _DENSE:
                return [llir.BlankLine(), self._lower_dense(position + 1)]
            if child.kind is _TILE_OUTER:
                return [llir.BlankLine(), self._lower_tile_outer(position + 1)]
            if child.kind is _TILE_INNER:
                return [llir.BlankLine(), self._lower_tile_inner(position + 1)]
            if child.kind is _SPARSE:
                return self._lower_sparse(position + 1)
            return self._lower_merged(position + 1)
        return self._lower_leaf()

    def _lower_dense(self, position: int) -> llir.ForLoop:
        loop = self.loops[position]
        name = self._loop_var_name(loop)
        result_is_csr_row = (
            not self.result_is_dense
            and type(self.leaf) is AppendEntry
            and self._index_of(self.leaf.coords[0], "the appended row coordinate")
            == loop.index
        )
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        loop_drives_an_input = any(
            per_tensor.get(level) == loop.index
            for symbol, per_tensor in self.level_drivers.items()
            if symbol != self.result_symbol
            for level in per_tensor
        )
        body: List[llir.Stmt] = []
        if result_is_csr_row:
            body.append(llir.Comment("Assemble COMPRESSED level"))
            body.append(self._assembly_catch_up(loop))
        body.append(llir.Comment("Resolve dense coordinates"))
        body.extend(input_resolves)
        if not loop_drives_an_input:
            # A broadcast loop iterates only the result; its position chain
            # is the loop's driving dense iterator, exactly as the legacy
            # dense lattice emits it.
            body.extend(result_resolves)
        elif result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))
        if result_is_csr_row:
            body.append(llir.BlankLine())
            body.append(llir.Comment("Assembly compressed _level indices"))
            body.append(self._result_pos_set())
        loop_var = llir.Var(name=name, type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=self._loop_bound_var(loop)),
            update=llir.Increment(var=loop_var),
            body=body,
        )
        for_loop.scorch_index_var = name
        return for_loop

    def _lower_tile_outer(self, position: int) -> llir.ForLoop:
        """The origin loop of one affine split, exactly as legacy emits it.

        ``for (int64_t k_out = 0; k_out < <bound>; k_out += kTile_k)`` —
        the origin steps by the width constant, the body carries no
        coordinate resolves (the origin is unreadable), and the width
        constant is declared once in the function preamble.
        """

        loop = self.loops[position]
        name = self.dimension_names[loop.dimension]
        bound = self._tile_bound_var(loop)
        loop_var = llir.Var(name=f"{name}_out", type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=bound),
            update=llir.Assign(
                var=loop_var,
                value=llir.Var(name=f"kTile_{name}", type=llir.DataType.INT),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
            body=self._loop_children(position),
        )
        for_loop.scorch_index_var = f"{name}_out"
        return for_loop

    def _lower_tile_inner(self, position: int) -> llir.ForLoop:
        """The point loop of one affine split, exactly as legacy emits it.

        The body first reconstructs the logical coordinate
        (``int64_t k = k_out + k_in;``), breaks past the ragged tail
        (``if (k >= <bound>) break;``), then continues with the ordinary
        dense-body emission for the logical loop.
        """

        loop = self.loops[position]
        name = self.dimension_names[loop.dimension]
        bound = self._tile_bound_var(loop)
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        loop_drives_an_input = any(
            per_tensor.get(level) == loop.index
            for symbol, per_tensor in self.level_drivers.items()
            if symbol != self.result_symbol
            for level in per_tensor
        )
        body: List[llir.Stmt] = [
            llir.Comment("Resolve tiled index var"),
            llir.VarInit(
                var=llir.Var(name=name, type=llir.DataType.INT64),
                value=llir.Add(
                    left=llir.Var(name=f"{name}_out", type=llir.DataType.INT64),
                    right=llir.Var(name=f"{name}_in", type=llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(name=name, type=llir.DataType.INT),
                    right=llir.Var(name=bound.name, type=llir.DataType.INT),
                ),
                then_body=[llir.Break()],
            ),
            llir.Comment("Resolve dense coordinates"),
        ]
        body.extend(input_resolves)
        if not loop_drives_an_input:
            body.extend(result_resolves)
        elif result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))
        loop_var = llir.Var(name=f"{name}_in", type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(
                op="<",
                left=loop_var,
                right=llir.Var(name=f"kTile_{name}", type=llir.DataType.INT),
            ),
            update=llir.Increment(var=loop_var),
            body=body,
            unroll=loop.node.unroll,
        )
        for_loop.scorch_index_var = f"{name}_in"
        return for_loop

    def tile_size_inits(self) -> List[llir.Stmt]:
        """Width-constant declarations, one per split, in nest order."""

        tile_loops = [loop for loop in self.loops if loop.kind is _TILE_OUTER]
        if not tile_loops:
            return []
        stmts: List[llir.Stmt] = [
            llir.BlankLine(),
            llir.Comment("Initialize tile sizes"),
        ]
        for loop in tile_loops:
            name = self.dimension_names[loop.dimension]
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"kTile_{name}",
                        type=llir.DataType.CONSTEXPR_INT,
                    ),
                    value=llir.Literal(loop.node.width),
                )
            )
        return stmts

    def raw_loop_statements(self) -> List[llir.Stmt]:
        """The raw pre-pass loop-nest statements, legacy shape included."""

        first = self.loops[0]
        outer_loop = (
            self._lower_tile_outer(0)
            if first.kind is _TILE_OUTER
            else self._lower_dense(0)
        )
        stmts: List[llir.Stmt] = [llir.BlankLine(), outer_loop]
        leaf_indices = self._leaf_indices()
        outer_index = self._loop_logical_index(first)
        outer_in_result = any(
            self._index_of(index, "store index") == outer_index
            for index in leaf_indices
        )
        if self.result_is_dense and outer_in_result:
            # The legacy gate: parallelize the outer loop only when the
            # result is dense and the outer coordinate addresses it.
            if any(loop.kind is _MERGED for loop in self.loops):
                # Merged nests iterate their cursors through while loops, and
                # the legacy marker runs before its nested statement lists
                # are flattened, so it cannot see the merge's position-array
                # initializers: the applied policy is the row-count-only
                # ``scorch_nthreads(-1, rows)`` form.  Byte parity requires
                # the same policy, so the position-array search is
                # deliberately given no statements to inspect.
                outer = stmts[1]
                assert type(outer) is llir.ForLoop
                outer.omp_parallel_for = True
                apply_parallel_policy(outer, body=())
            else:
                mark_first_for_loop_parallel(stmts, EMPTY_PARALLEL_WORKSPACE_CLUSTER)
        return stmts

    def kernel_abi(self) -> TorchCppKernelABI:
        return TorchCppKernelABI(
            result_shape=self.shapes[self.result_symbol],
            result_rank=len(self.result_decl.levels),
            input_tensors=tuple(
                KernelTensorABI(
                    name=self.decls[symbol].name,
                    level_types=tuple(
                        _LEVEL_KIND_TO_LEVEL_TYPE[level.kind]
                        for level in self.decls[symbol].levels
                    ),
                    mode_order=tuple(range(len(self.decls[symbol].levels))),
                    shape=self.shapes[symbol],
                    dtype=_SCALAR_TO_TORCH[self.decls[symbol].dtype],
                )
                for symbol in self.program.inputs
            ),
        )

    def result_assembler(self) -> ResultTensorAssembler:
        return ResultTensorAssembler(
            name=self.result_decl.name,
            level_types=tuple(
                _LEVEL_KIND_TO_LEVEL_TYPE[level.kind]
                for level in self.result_decl.levels
            ),
            dtype=_SCALAR_TO_TORCH[self.result_decl.dtype],
        )

    def result_size_inits(self) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        for level, level_decl in enumerate(self.result_decl.levels):
            if level_decl.kind is not LevelKind.DENSE:
                continue
            stmts.append(
                llir.VarInit(
                    llir.Var(
                        name=f"{self.result_decl.name}{level}_size",
                        type=llir.DataType.INT64,
                    ),
                    value=llir.ArrayAccess(
                        array=llir.Var(
                            name="result_shape",
                            type=llir.DataType.STD_VECTOR_INT,
                        ),
                        index=llir.Literal(
                            value=level,
                            data_type=llir.DataType.INT64,
                        ),
                    ),
                )
            )
        return stmts

    def value_array_ctypes(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (
                f"{self.decls[symbol].name}_val",
                dtype_to_c_datatype(_SCALAR_TO_TORCH[self.decls[symbol].dtype]).value,
            )
            for symbol in self.program.inputs
        )


def _lower_loopir_to_llir_owned(
    program: LoopProgram,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
) -> llir.Function:
    """Execute target lowering under an already-owned timing boundary."""

    verify_program(program)

    lowering = _TargetLowering(program, input_shapes, result_shape)
    raw_statements = lowering.raw_loop_statements()
    kernel_abi = lowering.kernel_abi()
    assembler = lowering.result_assembler()

    validation_stmts = kernel_abi.emit_validation()
    size_stmts = lowering.result_size_inits()
    prologue_stmts = kernel_abi.emit_input_prologue()
    value_init_stmts = assembler.emit_value_array_init()
    tile_size_stmts = lowering.tile_size_inits()
    level_indices_stmts = assembler.emit_level_indices_init()
    if level_indices_stmts:
        level_indices_stmts = [
            llir.Comment("Init result level indices"),
            *level_indices_stmts,
        ]
    final_assembly_stmts = assembler.emit_final_assembly()

    def assemble_body(
        transformed_body: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact:
        if compressed_output_parallel:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    "unsupported_program_shape",
                    "LoopIR lowering never produces compressed "
                    "two-phase output assembly",
                )
            )
        body_stmts: List[llir.Stmt] = [
            *validation_stmts,
            llir.Comment("Init result tensor level sizes"),
            *size_stmts,
            *prologue_stmts,
            llir.BlankLine(),
            *level_indices_stmts,
            llir.Comment("Initialize result value array"),
            *value_init_stmts,
            *tile_size_stmts,
            llir.BlankLine(),
            *transformed_body.statements,
            *final_assembly_stmts,
        ]
        return LLIRRewriteArtifact(body_stmts)

    manager = LLIRPassManager.from_compile_options(compile_options)
    try:
        pipeline_result = manager.run_production_pipeline(
            LLIRStatementListArtifact(raw_statements),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(
                DensePointerHoistContext(
                    value_array_ctypes=lowering.value_array_ctypes()
                )
            ),
            body_assembler=assemble_body,
        )
    except LLIRPassPartialFailure as failure:
        if compilation_context is not None:
            compilation_context.record_llir_pass_runs(
                failure.completed_run_records,
                compile_options=compile_options,
                stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
            )
        raise failure.failure from None
    if compilation_context is not None:
        compilation_context.record_llir_pass_runs(
            pipeline_result.run_records,
            compile_options=compile_options,
            stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        )
    return kernel_abi.assemble_function(pipeline_result.artifact.value)


def lower_loopir_to_llir(
    program: LoopProgram,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> llir.Function:
    """Lower one verified LoopIR program to a complete LLIR function.

    Runtime shapes are bound and cross-checked here; the returned function is
    the same structured-LLIR artifact the legacy path produces, ready for the
    exhaustive C++ emitter.  When a compilation context is supplied, this
    boundary owns the complete ``LOOPIR_TO_LLIR_LOWERING`` stage and its
    managed-pass records; callers do not pre-open that stage.
    """

    if (
        compilation_context is not None
        and type(compilation_context) is not CompilationContext
    ):
        raise TypeError("compilation_context must be a CompilationContext")
    if compile_options is None:
        compile_options = (
            compilation_context.compile_options
            if compilation_context is not None
            else CompileOptions.from_environment()
        )
    elif type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions snapshot")
    if compilation_context is None:
        return _lower_loopir_to_llir_owned(
            program,
            input_shapes=input_shapes,
            result_shape=result_shape,
            compile_options=compile_options,
            compilation_context=None,
        )

    compilation_context.require_compile_options(
        compile_options,
        stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
    )
    token = compilation_context.begin_stage(
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        compile_options=compile_options,
    )
    try:
        lowered = _lower_loopir_to_llir_owned(
            program,
            input_shapes=input_shapes,
            result_shape=result_shape,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
    except Exception:
        compilation_context.fail_stage(token)
        raise
    compilation_context.complete_stage(token)
    return lowered
