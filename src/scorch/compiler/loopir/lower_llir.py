"""LoopIR-to-structured-LLIR lowering for the Phase-4 dense families.

The existing structured LLIR remains the target-specific CxxIR boundary; no
new target IR is introduced.  This module lowers one verified dense LoopIR
program into a complete LLIR ``evaluate`` function by reusing the exact
production target components the legacy path uses:

- :class:`~scorch.compiler.torch_cpp_abi.TorchCppKernelABI` /
  :class:`~scorch.compiler.torch_cpp_abi.KernelTensorABI` own the public
  signature, validation, and input prologue;
- :class:`~scorch.compiler.torch_cpp_abi.ResultTensorAssembler` owns result
  storage initialization and final assembly;
- the managed production LLIR pass pipeline
  (:class:`~scorch.compiler.llir_pass_manager.LLIRPassManager`) applies the
  same typed optimization passes (dense pointer hoisting, single-iteration
  elimination, invariant hoisting, dynamic-vector rewriting, ...);
- :func:`~scorch.compiler.parallel_marking_pass.mark_first_for_loop_parallel`
  applies the same outer-loop parallel policy.

Because the raw loop-nest emission mirrors the legacy dense lowering
statement-for-statement, the generated C++ for the migrated dense families is
byte-identical to the legacy pipeline's output; the differential suite locks
that equality.  LoopIR itself contains none of these target details — this
module is where target spelling begins.

Shapes are runtime bindings, not LoopIR content: callers pass concrete input
shapes and the result shape, and this boundary re-resolves every logical
dimension's extent across all of them, failing closed on any disagreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, NoReturn, Optional, Tuple

import torch

from ...format import LevelType
from ..identity import SymbolId, new_access_id
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
    mark_first_for_loop_parallel,
)
from ..torch_cpp_abi import (
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
)
from ...utils import dtype_to_c_datatype
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
    ScalarType,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
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

_CPP_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TARGET_RESERVED_NAMES = frozenset(
    {
        "Tensor",
        "evaluate",
        "result_shape",
        "scorch_chunk",
        "scorch_native",
        "scorch_nthreads",
        "scorch_zero_dense",
        "std",
        "torch",
    }
)


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
    """One dense loop of the family nest, outermost first."""

    index: object
    dimension: DimensionId
    node: DenseFor


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
        self.dimension_names: Dict[DimensionId, str] = {}
        self._validate_display_names()
        self._validate_layouts()
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self.loops = self._collect_loop_nest()
        self.loop_positions: Dict[object, int] = {
            loop.index: position for position, loop in enumerate(self.loops)
        }
        self.leaf = self._collect_leaf()
        self.loads = self._collect_loads()
        self._validate_access_orders()

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
                    f"display name {decl.name!r} is used more than once",
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
            for level in range(len(decl.levels)):
                reserve(f"{decl.name}{level}_size", owner)
                reserve(f"p{decl.name}{level}", owner)

        output = self.result_decl
        output_owner = f"output tensor {output.name!r}"
        for name in (
            output.name,
            f"{output.name}_capacity",
            f"{output.name}_values",
            f"{output.name}_values_torch",
        ):
            reserve(name, output_owner)
        for level in range(len(output.levels)):
            reserve(f"{output.name}{level}_size", output_owner)
            reserve(f"p{output.name}{level}", output_owner)

    def _validate_layouts(self) -> None:
        if len(self.program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        for decl in self.program.tensors:
            # verify_program already rejected every non-DENSE level kind, so
            # only the storage permutation needs a target-boundary check.
            modes = tuple(level.mode for level in decl.levels)
            if modes != tuple(range(len(decl.levels))):
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} uses a non-identity storage order, "
                    "which the migrated dense families do not cover",
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
                loops.append(_Loop(only.index, only.dimension, only))
                body = only.body
                continue
            if type(only) in (Store, StoreReduce):
                if not loops:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering requires at least one loop",
                    )
                return loops
            _fail(
                "unsupported_program_shape",
                f"unsupported nest statement {type(only).__name__}",
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

    def _collect_loads(self) -> List[Load]:
        loads: List[Load] = []

        def walk(expr: Expr) -> None:
            if type(expr) is Load:
                loads.append(expr)
                return
            if type(expr) is BinaryExpr:
                walk(expr.lhs)
                walk(expr.rhs)
                return
            if type(expr) is IndexValue:
                _fail(
                    "unsupported_program_shape",
                    "coordinate values are not value expressions in the "
                    "dense families",
                )
            _fail(
                "unsupported_program_shape",
                f"unsupported value expression {type(expr).__name__}",
            )

        walk(self.leaf.value)  # type: ignore[attr-defined]
        seen: set[SymbolId] = set()
        for load in loads:
            if load.tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor {self.decls[load.tensor].name!r} is loaded "
                    "more than once; the dense target owns one physical "
                    "position chain per input",
                )
            seen.add(load.tensor)
        return loads

    def _access_vars(self, tensor: SymbolId, indices: Tuple[Expr, ...]) -> List[object]:
        return [
            self._index_of(index, f"access of {self.decls[tensor].name!r}")
            for index in indices
        ]

    def _validate_access_orders(self) -> None:
        leaf_indices = self.leaf.indices  # type: ignore[attr-defined]
        accesses = [(load.tensor, tuple(load.indices)) for load in self.loads] + [
            (self.result_symbol, tuple(leaf_indices))
        ]
        for tensor, indices in accesses:
            positions = []
            for index in indices:
                bound = self._index_of(index, f"access of {self.decls[tensor].name!r}")
                position = self.loop_positions.get(bound)
                if position is None:
                    _fail(
                        "unsupported_program_shape",
                        "access coordinates must be nest loop variables",
                    )
                positions.append(position)
            if positions != sorted(positions) or len(set(positions)) != len(positions):
                _fail(
                    "unsupported_loop_order",
                    f"tensor {self.decls[tensor].name!r} storage order "
                    "conflicts with the loop nest order",
                )

    # -- emission ------------------------------------------------------------

    def _loop_var_name(self, loop: _Loop) -> str:
        return self.dimension_names[loop.dimension]

    def _loop_bound_var(self, loop: _Loop) -> llir.Var:
        for load in self.loads:
            for level, index in enumerate(load.indices):
                if self._index_of(index, "load index") == loop.index:
                    name = self.decls[load.tensor].name
                    return llir.Var(
                        name=f"{name}{level}_size", type=llir.DataType.INT64
                    )
        leaf_indices = self.leaf.indices  # type: ignore[attr-defined]
        for level, index in enumerate(leaf_indices):
            if self._index_of(index, "store index") == loop.index:
                return llir.Var(
                    name=f"{self.result_decl.name}{level}_size",
                    type=llir.DataType.INT64,
                )
        _fail("unsupported_program_shape", "a loop variable is never used")
        raise AssertionError("unreachable")

    def _position_init(
        self,
        tensor_name: str,
        level: int,
        loop: _Loop,
    ) -> llir.VarInit:
        loop_var = llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT64)
        if level == 0:
            value: llir.Expr = loop_var
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
        emitted: set = set()
        for load in self.loads:
            decl = self.decls[load.tensor]
            for level, index in enumerate(load.indices):
                if self._index_of(index, "load index") != loop.index:
                    continue
                key = (load.tensor, level)
                if key in emitted:
                    continue
                emitted.add(key)
                stmts.append(self._position_init(decl.name, level, loop))
        return stmts

    def _result_resolves_at(self, loop: _Loop) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        leaf_indices = self.leaf.indices  # type: ignore[attr-defined]
        for level, index in enumerate(leaf_indices):
            if self._index_of(index, "store index") != loop.index:
                continue
            stmts.append(self._position_init(self.result_decl.name, level, loop))
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
            metadata = llir.TensorAccessMetadata(
                access_id=new_access_id(),
                tensor_id=expr.tensor,
                index_ids=tuple(
                    self._index_of(index, "load index")  # type: ignore[misc]
                    for index in expr.indices
                ),
                role=llir.TensorAccessRole.INPUT_READ,
            )
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(name=physical_index, type=llir.DataType.INT),
                tensor_access=metadata,
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
        metadata = llir.TensorAccessMetadata(
            access_id=new_access_id(),
            tensor_id=self.result_symbol,
            index_ids=tuple(
                self._index_of(index, "store index")  # type: ignore[misc]
                for index in leaf_indices
            ),
            role=llir.TensorAccessRole.RESULT_WRITE,
        )
        target = llir.ArrayAccess(
            array=llir.Var(
                name=f"{self.result_decl.name}_values",
                type=llir.DataType.NO_TYPE,
            ),
            index=llir.Var(
                name=f"p{self.result_decl.name}{len(leaf_indices) - 1}",
                type=llir.DataType.INT64,
            ),
            tensor_access=metadata,
        )
        if type(leaf) is StoreReduce:
            return [llir.Assign(var=target, value=rhs, op=llir.AssignOp.ADD_ASSIGN)]
        return [llir.Assign(var=target, value=rhs)]

    def _lower_loop(self, position: int) -> llir.ForLoop:
        loop = self.loops[position]
        name = self._loop_var_name(loop)
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        loop_drives_an_input = any(
            self._index_of(index, "load index") == loop.index
            for load in self.loads
            for index in load.indices
        )
        body: List[llir.Stmt] = [llir.Comment("Resolve dense coordinates")]
        body.extend(input_resolves)
        if not loop_drives_an_input:
            # A broadcast loop iterates only the result; its position chain
            # is the loop's driving dense iterator, exactly as the legacy
            # dense lattice emits it.
            body.extend(result_resolves)
        elif result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        if position + 1 < len(self.loops):
            body.append(llir.BlankLine())
            body.append(self._lower_loop(position + 1))
        else:
            body.extend(self._lower_leaf())
        loop_var = llir.Var(name=name, type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=self._loop_bound_var(loop)),
            update=llir.Increment(var=loop_var),
            body=body,
        )
        for_loop.scorch_index_var = name
        return for_loop

    def raw_loop_statements(self) -> List[llir.Stmt]:
        """The raw pre-pass loop-nest statements, legacy shape included."""

        stmts: List[llir.Stmt] = [llir.BlankLine(), self._lower_loop(0)]
        leaf_indices = self.leaf.indices  # type: ignore[attr-defined]
        outer_index = self.loops[0].index
        outer_in_result = any(
            self._index_of(index, "store index") == outer_index
            for index in leaf_indices
        )
        if outer_in_result:
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
                        LevelType.DENSE for _ in self.decls[symbol].levels
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
            level_types=tuple(LevelType.DENSE for _ in self.result_decl.levels),
            dtype=_SCALAR_TO_TORCH[self.result_decl.dtype],
        )

    def result_size_inits(self) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        for level in range(len(self.result_decl.levels)):
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
    final_assembly_stmts = assembler.emit_final_assembly()

    def assemble_body(
        transformed_body: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact:
        if compressed_output_parallel:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    "unsupported_program_shape",
                    "dense LoopIR lowering never produces compressed "
                    "two-phase output assembly",
                )
            )
        body_stmts: List[llir.Stmt] = [
            *validation_stmts,
            llir.Comment("Init result tensor level sizes"),
            *size_stmts,
            *prologue_stmts,
            llir.BlankLine(),
            llir.Comment("Initialize result value array"),
            *value_init_stmts,
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
    """Lower one verified dense LoopIR program to a complete LLIR function.

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
