from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

from . import llir
from .cin import (
    IndexStmt,
    IndexVar,
    TensorVar,
    CINVisitorAccept,
    TensorAssign,
    TensorAccess,
    ForAll,
    IndexExpr,
    BinaryOp,
    CIN,
    Operation,
    Where,
    WorkspaceAccess,
    Workspace,
    PostOp,
    PostOps,
)
from .iter_lattice import IterationLattice
from .iterator import (
    collect_mode_position_arrays,
    match_mode_position_access,
    match_mode_position_begin,
)
from .llir import AssignOp, DataType
from .diagnostics import (
    CompileOptionsDiagnostic,
    CompileOptionsError,
    CompilerInvariantError,
    UnsupportedFeature,
)
from .loop_plan import LoopPlan, ScheduledCIN, verify_scheduled_cin
from .legacy_cin_adapter import (
    claim_legacy_cin_working_tree,
    legacy_cin_working_copy,
)
from .cin_analysis import verify_cin_if_enabled
from .compile_options import CompileOptions
from .compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPPolicy,
)
from .dense_pointer_hoist_pass import DensePointerHoistContext
from .llir_pass_manager import (
    PRODUCTION_LLIR_PASS_OPTIONS,
    CompressedWhereOpenMPPassSpec,
    DensePointerHoistPassSpec,
    LLIRBodyAssembler,
    LLIRPassManager,
    LLIRPassOptions,
    LLIRPassPipeline,
    LLIRPassPartialFailure,
    LLIRPassRunRecord,
    LLIRRewriteArtifact,
    LLIRStatementListArtifact,
)
from .compilation_context import CompilerStageId, CompilationContext
from .llir_traversal import LLIRTraversalContext
from .torch_cpp_abi import (
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
    tensor_data_ptr,
    tensor_storage_member,
)
from ..format import LevelType, TensorFormat, LevelFormat
from ..utils import (
    dtype_to_c_datatype,
    get_pytorch_c_dtype_name,
)

if TYPE_CHECKING:
    from .scheduler import Schedule


_SPARSE_POSITION_DISCOVERY_CONTEXT = LLIRTraversalContext(
    stage="CIN lowering",
    pass_name="collect_sparse_position_arrays",
)


@dataclass(frozen=True)
class _LoweredOuterBody:
    """Raw outer-body LLIR plus an optional deferred compressed-Where pass."""

    statements: List[llir.Stmt]
    compressed_where_pass_spec: Optional[CompressedWhereOpenMPPassSpec] = None


def _result_tensor_abi_assembler(
    tensor_var: TensorVar,
    *,
    known_nnz_var: Optional[str] = None,
    exact_dense_parent_positions: bool = False,
    reserve_hint_var: Optional[str] = None,
) -> ResultTensorAssembler:
    """Snapshot one semantic result at the Torch/C++ ABI ownership boundary."""

    return ResultTensorAssembler(
        name=tensor_var.get_name(),
        level_types=tuple(tensor_var.get_level_types()),
        dtype=tensor_var.dtype,
        known_nnz_var=known_nnz_var,
        exact_dense_parent_positions=exact_dense_parent_positions,
        reserve_hint_var=reserve_hint_var,
    )


def _kernel_tensor_abi(tensor_var: TensorVar) -> KernelTensorABI:
    """Snapshot one semantic input at the Torch/C++ ABI ownership boundary."""

    level_types = tuple(tensor_var.get_level_types())
    mode_order = tuple(tensor_var.get_mode_order() or tuple(range(len(level_types))))
    return KernelTensorABI(
        name=tensor_var.get_name(),
        level_types=level_types,
        mode_order=mode_order,
        shape=tuple(tensor_var.shape or ()),
        dtype=tensor_var.dtype,
    )


def _torch_cpp_kernel_abi(
    result_tensor_var: TensorVar,
    input_tensor_vars: List[TensorVar],
    post_ops: Optional[PostOps],
) -> TorchCppKernelABI:
    """Snapshot one complete public kernel ABI without retaining semantic IR."""

    extra_tensor_names = tuple(post_ops.extra_tensors) if post_ops is not None else ()
    return TorchCppKernelABI(
        result_shape=tuple(result_tensor_var.shape or ()),
        result_rank=len(result_tensor_var.get_level_types()),
        input_tensors=tuple(_kernel_tensor_abi(tensor) for tensor in input_tensor_vars),
        extra_tensor_names=extra_tensor_names,
        extra_tensor_dtype=(result_tensor_var.dtype if extra_tensor_names else None),
    )


class CINLowerer:
    """
    This is a class to lower a CIN to LLIR
    """

    _SUPPORTED_POST_OP_KINDS = frozenset(
        {"add", "mul", "relu", "gelu", "tanh", "sigmoid"}
    )

    def __init__(
        self,
        filter_zeros=False,
        post_ops: Optional[PostOps] = None,
        llir_pass_options: Optional[LLIRPassOptions] = None,
        compile_options: Optional[CompileOptions] = None,
        compilation_context: Optional[CompilationContext] = None,
    ):
        if compile_options is None:
            compile_options = CompileOptions.from_environment(
                llir_pass_options=(
                    llir_pass_options
                    if llir_pass_options is not None
                    else PRODUCTION_LLIR_PASS_OPTIONS
                )
            )
        elif type(compile_options) is not CompileOptions:
            raise TypeError("compile_options must be a CompileOptions instance")
        elif llir_pass_options is not None:
            raise TypeError(
                "llir_pass_options and compile_options cannot both be provided"
            )
        if compilation_context is not None:
            if type(compilation_context) is not CompilationContext:
                raise TypeError(
                    "compilation_context must be a CompilationContext instance"
                )
            if compilation_context.compile_options is not compile_options:
                raise TypeError(
                    "compilation_context must retain this lowering's exact "
                    "CompileOptions snapshot"
                )
        self._compilation_context = compilation_context
        self.compile_options = compile_options
        self.filter_zeros: bool = filter_zeros
        self.post_ops: Optional[PostOps] = post_ops
        self._llir_pass_manager = LLIRPassManager.from_compile_options(compile_options)
        self._validate_post_ops()
        self.defined_index_vars: List[IndexVar] = []

        self.dense_coord_resolve_stmt_to_dep_index_vars: Dict[
            llir.VarInit, List[IndexVar]
        ] = {}

        self.seen_outermost_forall = False
        self.outermost_stmt: Optional[IndexStmt] = None
        self.has_explicit_parallel_loop = False
        self.loop_plan: Optional[LoopPlan] = None
        self.normalized_cin: Optional[IndexStmt] = None

        self.result_value_array_sparse_index_llir = None
        self._scalar_accum_mode = False
        self._used_scalar_accum = False
        self.index_var_to_rhs_tensor_level_type = None
        self.index_var_to_result_tensor_level_type = None

        self._known_nnz_var: Optional[str] = None

        # Two-phase parallel compressed output state
        self._where_producer_stmts: Optional[List[llir.Stmt]] = None
        self._where_consumer_stmts: Optional[List[llir.Stmt]] = None
        self._where_workspace_name: Optional[str] = None
        self._where_workspace_ctype: Optional[str] = None
        self._where_workspace_dim: Optional[int] = None
        self.result_tensor_var: Optional[TensorVar] = None
        self.result_tensor_access: Optional[TensorAccess] = None
        self.result_tensor_value_index_var_dict: Dict[IndexVar, llir.Expr] = {}
        self.final_result_tensor_var: Optional[TensorVar] = None
        self.final_result_tensor_access: Optional[TensorAccess] = None

        self.llir_stmt: Optional[llir.Stmt] = None

        self.need_compute: List[TensorVar] = []
        self.tensor_var_to_llir: Dict[TensorVar, llir.Expr] = {}
        self._value_array_ctypes: Dict[str, str] = {}
        self._llir_pass_run_records: Tuple[LLIRPassRunRecord, ...] = ()

    @property
    def llir_pass_run_records(self) -> Tuple[LLIRPassRunRecord, ...]:
        """Completed managed LLIR passes in exact production execution order."""

        return self._llir_pass_run_records

    @property
    def llir_pass_pipeline(self) -> LLIRPassPipeline:
        """The exact frozen manager-owned pipeline for this compilation."""

        pipeline = self._llir_pass_manager.pipeline
        if type(pipeline) is not LLIRPassPipeline:
            raise CompilerInvariantError(
                "stage=CIN lowering: production manager has no LLIR pass pipeline"
            )
        return pipeline

    def _record_llir_pass_runs(self, records: Tuple[LLIRPassRunRecord, ...]) -> None:
        start = len(self._llir_pass_run_records)
        owned_records = tuple(
            replace(record, sequence_index=start + offset)
            for offset, record in enumerate(records)
        )
        self._llir_pass_run_records += owned_records
        if self._compilation_context is not None:
            self._compilation_context.record_llir_pass_runs(
                owned_records,
                compile_options=self.compile_options,
            )

    def _instrument_body_assembler(
        self,
        body_assembler: LLIRBodyAssembler,
    ) -> LLIRBodyAssembler:
        """Time the existing one-shot result/ABI continuation when owned."""

        compilation_context = self._compilation_context
        if compilation_context is None:
            return body_assembler

        def timed_body_assembler(
            transformed_body: LLIRStatementListArtifact,
            compressed_output_parallel: bool,
        ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
            assembly_token = compilation_context.begin_stage(
                CompilerStageId.RESULT_ABI_ASSEMBLY,
                compile_options=self.compile_options,
                nested_within=CompilerStageId.CIN_LOWERING,
            )
            try:
                assembled = body_assembler(
                    transformed_body,
                    compressed_output_parallel,
                )
                assembled = self._llir_pass_manager.validate_body_assembly_artifact(
                    assembled
                )
            except Exception:
                compilation_context.fail_stage(assembly_token)
                raise
            compilation_context.complete_stage(assembly_token)
            return assembled

        return timed_body_assembler

    def _validate_post_ops(self) -> None:
        """Reject post-ops that this lowering stage cannot represent."""
        if self.post_ops is None:
            return
        for op in self.post_ops.ops:
            if not isinstance(op, PostOp):
                descriptor_type = type(op).__qualname__
                raise CompilerInvariantError(
                    f"stage=CIN lowering: unknown post-op descriptor type "
                    f"'{descriptor_type}'"
                )
            if op.kind not in self._SUPPORTED_POST_OP_KINDS:
                raise UnsupportedFeature(
                    f"stage=CIN lowering: unsupported post-op kind '{op.kind}'"
                )

    @staticmethod
    def _validate_index_stmt(stmt: IndexStmt) -> None:
        """Reject statement nodes with no CIN-to-LLIR lowering rule."""
        if not isinstance(stmt, (TensorAssign, ForAll, Where)):
            node_type = type(stmt).__qualname__
            raise CompilerInvariantError(
                f"stage=CIN lowering: unknown IndexStmt node type '{node_type}'"
            )

    def _emit_post_ops(
        self,
        output_var_name: str,
        index_expr: str,
        tensor_access: Optional[llir.TensorAccessMetadata] = None,
    ) -> List[llir.Stmt]:
        """Emit LLIR statements for post-ops on output_var_name[index_expr]."""
        self._validate_post_ops()
        if not self.post_ops or not self.post_ops.ops:
            return []
        stmts: List[llir.Stmt] = []

        def target() -> llir.ArrayAccess:
            return llir.ArrayAccess(
                array=llir.Var(
                    name=output_var_name,
                    type=llir.DataType.NO_TYPE,
                ),
                index=llir.Var(
                    name=index_expr,
                    type=llir.DataType.INT64,
                ),
                tensor_access=tensor_access,
            )

        for op in self.post_ops.ops:
            if op.kind == "add":
                stmts.append(
                    llir.Assign(
                        var=target(),
                        value=llir.Var(
                            name=f"{op.tensor_name}_val[{index_expr}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        op=AssignOp.ADD_ASSIGN,
                    )
                )
            elif op.kind == "mul":
                stmts.append(
                    llir.Assign(
                        var=target(),
                        value=llir.Var(
                            name=f"{op.tensor_name}_val[{index_expr}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        op=AssignOp.MUL_ASSIGN,
                    )
                )
            elif op.kind in ("relu", "gelu", "tanh", "sigmoid"):
                stmts.append(
                    llir.Assign(
                        var=target(),
                        value=llir.FunctionCall(
                            name=f"scorch_{op.kind}",
                            args=[target()],
                        ),
                        op=AssignOp.ASSIGN,
                    )
                )
        return stmts

    @staticmethod
    def get_value_array_statement(tensor: TensorVar) -> llir.Stmt:
        """
        Get the value array for a tensor
        """
        return llir.VarInit(
            var=llir.Var(name=f"{tensor.name}_values", type=llir.DataType.TORCH_TENSOR),
            value=tensor_storage_member(tensor.name, "storage", "value"),
        )

    def lower_TensorAccess(self, tensor_access: TensorAccess) -> llir.Expr:
        """
        Lower a TensorAccess to LLIR
        """
        tensor_var = tensor_access.get_tensor()
        tensor_access_metadata = self._tensor_access_metadata(
            tensor_access,
            llir.TensorAccessRole.INPUT_READ,
        )
        if tensor_access_metadata is None:
            raise CompilerInvariantError(
                "stage=CIN lowering pass=lower_TensorAccess: "
                f"workspace tensor '{tensor_var.get_name()}' reached generic value "
                "lowering; workspace reads require a workspace-specific consumer"
            )

        sorted_index_vars = tensor_access.get_sorted_index_vars()
        last_index_var = sorted_index_vars[-1]

        # If the level_type corresponding to the last index var is dense, then we can just use
        # the index var as the index into the value array
        level = tensor_access.level_of_index_var(last_index_var)
        level_type = tensor_var.get_level_types()[level]

        if len(tensor_access.indices) == 1 and level_type == LevelType.DENSE:
            physical_index = last_index_var.name
        else:
            physical_index = (
                f"p{tensor_access.tensor.get_name()}"
                f"{tensor_access.level_of_index_var(last_index_var)}"
            )

        return llir.ArrayAccess(
            array=llir.Var(
                name=f"{tensor_access.tensor.name}_val",
                type=llir.DataType.ptr_type(tensor_access.tensor.dtype),
            ),
            index=llir.Var(name=physical_index, type=llir.DataType.INT),
            tensor_access=tensor_access_metadata,
        )

    def lower_BinaryOp(self, bin_op: BinaryOp) -> llir.Expr:
        """
        Lower a BinaryOp to LLIR
        """
        return llir.BinOp(
            op=bin_op.op.value,
            left=self.lower_IndexExpr(bin_op.left),
            right=self.lower_IndexExpr(bin_op.right),
        )

    def lower_IndexExpr(self, index_expr: IndexExpr) -> llir.Expr:
        if isinstance(index_expr, BinaryOp):
            return self.lower_BinaryOp(index_expr)
        elif isinstance(index_expr, TensorAccess):
            return self.lower_TensorAccess(index_expr)
        raise CompilerInvariantError(
            "stage=CIN lowering: unknown IndexExpr node type "
            f"'{type(index_expr).__qualname__}'"
        )

    @staticmethod
    def _tensor_access_metadata(
        tensor_access: TensorAccess,
        role: llir.TensorAccessRole,
    ) -> Optional[llir.TensorAccessMetadata]:
        """Return logical provenance for a non-workspace tensor value access."""
        if tensor_access.is_workspace():
            return None
        return llir.TensorAccessMetadata(
            access_id=tensor_access.access_id,
            tensor_id=tensor_access.tensor_id,
            index_ids=tuple(tensor_access.index_ids),
            role=role,
        )

    def lower_CIN(self, cin: CIN) -> Union[llir.Stmt, List[llir.Stmt], llir.Expr]:
        if isinstance(cin, IndexStmt):
            return self.lower_IndexStmt(cin)
        elif isinstance(cin, IndexExpr):
            return self.lower_IndexExpr(cin)
        node_type = type(cin).__qualname__
        raise CompilerInvariantError(
            f"stage=CIN lowering: unknown CIN node type '{node_type}'"
        )

    def lower_TensorAssign(self, stmt: TensorAssign) -> List[llir.Stmt]:
        """
        Lower a TensorAssign to LLIR
        """
        llir_stmts: List[llir.Stmt] = []

        rhs_llir = self.lower_IndexExpr(stmt.rhs)

        # Scalar accumulation mode: accumulate into local register.
        # Coordinate emission and position increment are handled by
        # the enclosing free-variable ForAll level (see iter_lattice.py).
        if self._scalar_accum_mode:
            llir_stmts.append(
                llir.Assign(
                    var=llir.Var(name="_accum", type=llir.DataType.NO_TYPE),
                    value=rhs_llir,
                    op=AssignOp.ADD_ASSIGN,
                )
            )
            return llir_stmts

        # if we are at the bottommost level, we can emit compute code
        assert self.result_tensor_access, "result tensor access is None"
        is_workspace = self.result_tensor_access.is_workspace()
        result_access_metadata = (
            self._tensor_access_metadata(
                self.result_tensor_access,
                llir.TensorAccessRole.RESULT_WRITE,
            )
            if not is_workspace
            else None
        )
        index_vars = self.result_tensor_access.get_index_vars()
        sorted_index_vars = self.result_tensor_access.get_sorted_index_vars()
        # If index_vars is None (empty), that means we have a scalar workspace
        # Then just do <tensor> += <rhs_llir>
        if not index_vars:
            wksp_name = self.result_tensor_var.get_name()
            assign_stmt = llir.Assign(
                var=llir.Var(name=f"{wksp_name}", type=llir.DataType.NO_TYPE),
                value=rhs_llir,
                op=AssignOp.ADD_ASSIGN,
            )
            llir_stmts.append(assign_stmt)
        else:
            # if index_vars are all in defined_index_vars, then we can emit the compute code
            if all(index_var in self.defined_index_vars for index_var in index_vars):
                assert self.result_tensor_var, "result tensor var is None"

                values_llir_name = self.result_tensor_var.name
                if not is_workspace:
                    values_llir_name = f"{values_llir_name}_values"

                if self.result_value_array_sparse_index_llir:
                    tensor_access_llir = llir.ArrayAccess(
                        array=llir.Var(
                            name=values_llir_name,
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=self.result_value_array_sparse_index_llir,
                        tensor_access=result_access_metadata,
                    )
                else:
                    level = self.result_tensor_access.level_of_index_var(
                        sorted_index_vars[-1]
                    )
                    tensor_access_llir = llir.ArrayAccess(
                        array=llir.Var(
                            name=values_llir_name,
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=f"p{self.result_tensor_var.name}{level}",
                            type=llir.DataType.INT64,
                        ),
                        tensor_access=result_access_metadata,
                    )
                    # tensor_access_llir = llir.Var(
                    #     name=f"{self.result_tensor_var.get_name()}_values"
                    #     + f"[{self.defined_index_vars[-1].name}]",
                    #     type=llir.DataType.NO_TYPE,
                    # )

                if is_workspace:
                    assert isinstance(self.result_tensor_access, WorkspaceAccess)
                    wksp_access: WorkspaceAccess = self.result_tensor_access
                    wksp_index_vars = wksp_access.get_index_vars()
                    sorted_wksp_index_vars = [
                        wksp_index_vars[i] for i in wksp_access.tensor.mode_order
                    ]

                    if wksp_access.is_dense():
                        # <workspace name>[<C++ array of indices>] += <rhs_llir>;
                        assert (
                            len(sorted_wksp_index_vars) == 1
                        ), "dense workspace has more than 1 index var"
                        wksp_index_var = sorted_wksp_index_vars[0]
                        llir_stmts.append(
                            llir.Assign(
                                var=llir.ArrayAccess(
                                    array=llir.Var(
                                        name=self.result_tensor_var.name,
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    index=llir.Var(
                                        name=wksp_index_var.name,
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                value=rhs_llir,
                                op=AssignOp.ADD_ASSIGN,
                            )
                        )

                    else:
                        # <workspace name>.insert(<C++ array of indices>, <rhs_llir>);

                        llir_stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{self.result_tensor_access.get_tensor().get_name()}.insert",
                                args=[
                                    llir.Array(
                                        values=[
                                            llir.Var(
                                                name=ivar.name,
                                                type=llir.DataType.INT64,
                                            )
                                            for ivar in sorted_wksp_index_vars
                                        ],
                                        data_type=llir.DataType.INT64,
                                    ),
                                    rhs_llir,
                                ],
                            )
                        )
                else:
                    if stmt.op == Operation.ADD:
                        # llir_stmts.append(
                        #     llir.Assign(
                        #         var=tensor_access_llir,
                        #         value=llir.BinOp(
                        #             op="+",
                        #             left=tensor_access_llir,
                        #             right=rhs_llir,
                        #         ),
                        #     )
                        # )
                        llir_stmts.append(
                            llir.Assign(
                                var=tensor_access_llir,
                                value=rhs_llir,
                                op=AssignOp.ADD_ASSIGN,
                            )
                        )
                    else:
                        llir_stmts.append(
                            llir.Assign(
                                var=tensor_access_llir,
                                value=rhs_llir,
                            )
                        )
            # If the last _level of the result tensor var is sparse, then we need to set
            # the coordinates
            if not self.result_tensor_access.is_workspace():
                last_ivar = self.defined_index_vars[-1]
                last_level_type = self.result_tensor_access.level_types()[-1]
                if last_level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                    llir_stmts.append(llir.Comment("Set coordinates"))
                    result_tensor_name = self.result_tensor_var.get_name()
                    result_index_name = (
                        f"p{result_tensor_name}{self.result_tensor_var.levels - 1}"
                    )
                    level = self.result_tensor_access.level_of_index_var(last_ivar)

                    llir_stmts.append(
                        llir.Assign(
                            var=llir.ArrayAccess(
                                array=llir.Var(
                                    name=f"{result_tensor_name}{level}_crd",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                index=llir.Var(
                                    name=result_index_name,
                                    type=llir.DataType.INT64,
                                ),
                            ),
                            value=llir.Var(
                                name=last_ivar.name,
                                type=llir.DataType.NO_TYPE,
                            ),
                        )
                    )

                    # if the last _level is COORDINATE, we might need to set the coordinates
                    # or previous levels as well
                    if last_level_type == LevelType.COORDINATE:
                        for defined_ivar in self.defined_index_vars[-2::-1]:
                            level_type = (
                                self.result_tensor_access.level_type_of_index_var(
                                    defined_ivar
                                )
                            )
                            level = self.result_tensor_access.level_of_index_var(
                                defined_ivar
                            )
                            if level_type == LevelType.COORDINATE:
                                llir_stmts.append(
                                    llir.Assign(
                                        var=llir.ArrayAccess(
                                            array=llir.Var(
                                                name=f"{result_tensor_name}{level}_crd",
                                                type=llir.DataType.NO_TYPE,
                                            ),
                                            index=llir.Var(
                                                name=result_index_name,
                                                type=llir.DataType.INT64,
                                            ),
                                        ),
                                        value=llir.Var(
                                            name=defined_ivar.name,
                                            type=llir.DataType.NO_TYPE,
                                        ),
                                    )
                                )
                            else:
                                break

                # if has sparse index for result value array, need to increment
                if self.result_value_array_sparse_index_llir is not None:
                    llir_stmts.append(
                        llir.Increment(
                            var=self.result_value_array_sparse_index_llir,
                        )
                    )

            # If CINLowerer has filter_zeros attribute set to True,
            # we need to wrap llir_stmts in an if block,
            # the condition is whether the input value is non-zero
            if self.filter_zeros:
                llir_stmts = [
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=rhs_llir,
                            right=llir.Literal(value="0", data_type=llir.DataType.INT),
                        ),
                        then_body=llir_stmts,
                    )
                ]

        return llir_stmts

    def lower_Where(self, stmt: Where) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower a Where to LLIR
        """
        workspaces = stmt.get_workspaces()
        workspace_init_stmts: List[llir.Stmt] = [
            llir.Comment("Initialize workspaces"),
        ]
        workspace_cleanup_stmts: List[llir.Stmt] = []
        # Per-thread workspace ownership is constructed serially before OpenMP;
        # each worker selects a disjoint slice inside the region. This keeps
        # allocation failures unwindable through pybind instead of OpenMP.
        self._workspace_alloc_stmts: List[llir.Stmt] = []
        self._workspace_pool_specs: List[tuple[str, str, str]] = []
        self._workspace_free_stmts: List[llir.Stmt] = []
        self._workspace_memset_stmts: List[llir.Stmt] = []
        for wksp in workspaces:
            assert isinstance(wksp, Workspace), "workspace is not a Workspace"
            # coo_workspace<tensor's ctype> <tensor's name> = coo_workspace<tensor's ctype>(<tensor's dim>);
            wksp_ctype = dtype_to_c_datatype(wksp.dtype)
            wksp_ctype_ptr = DataType.ptr_type(wksp_ctype)

            # If the workspace is 0-dimensional, just initialize it with a literal
            if wksp.dim == 0:
                workspace_init_stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=wksp.get_name(),
                            type=wksp_ctype,
                        ),
                        value=llir.Literal(0),
                    )
                )
                continue

            if wksp.dim == 1:
                # Dense workspace: allocate a zeroed array of the appropriate ctype
                if wksp.is_dense():
                    if wksp.is_tiled and wksp.tile_size_var:
                        # Tiled dense workspaces have compile-time bounds. Use stack allocation
                        # so inner kernels avoid heap traffic.
                        workspace_init_stmts.append(
                            llir.FixedStackArrayDecl(
                                name=wksp.get_name(),
                                element_type=wksp_ctype,
                                extent=wksp.tile_size_var.llir_var,
                                initializer=llir.Array(values=[], data_type=wksp_ctype),
                            )
                        )
                    else:
                        # Aligned allocation hoisted to per-thread (outside for loop).
                        # memset per iteration (inside for loop).
                        size_llir = wksp.size_llir_var
                        size_var = size_llir.name
                        # The workspace size variable may reference a dense tensor
                        # dimension (e.g. B1_size). Resolve it to the actual C++ name
                        # so it's available in the hoisted parallel region.
                        wksp_access = [
                            wa
                            for wa in wksp.workspace_accesses
                            if wa.indices and len(wa.indices) == 1
                        ][0]
                        idx_var = wksp_access.indices[0]
                        dense_ta = [
                            ta
                            for ta in idx_var.tensor_accesses
                            if ta.is_dense()
                            and idx_var in ta.indices
                            and not ta.is_workspace()
                        ][0]
                        level = dense_ta.level_of_index_var(idx_var)
                        actual_size = f"{dense_ta.tensor.name}{level}_size"

                        ctype = wksp_ctype.value
                        wname = wksp.get_name()
                        self._workspace_pool_specs.append((wname, ctype, actual_size))
                        self._workspace_alloc_stmts.extend(
                            [
                                # The extent alias is typed; the pool borrow
                                # below stays raw because its casted spelling
                                # `(size_t)omp_get_thread_num()` has no typed
                                # byte-identical Cast form today.
                                llir.VarInit(
                                    var=size_llir,
                                    value=llir.Var(
                                        name=actual_size,
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                llir.RawStmt(
                                    code=(
                                        f"{ctype}* __restrict__ {wname} = "
                                        f"{wname}_pool_owner.get() + "
                                        f"(size_t)omp_get_thread_num() * "
                                        f"(size_t){actual_size}"
                                    ),
                                ),
                            ]
                        )
                        self._workspace_memset_stmts.append(
                            llir.FunctionCallStmt(
                                name="memset",
                                args=[
                                    llir.Var(
                                        name=wname,
                                        type=wksp_ctype_ptr,
                                    ),
                                    llir.Literal(
                                        value=0,
                                        data_type=llir.DataType.INT,
                                    ),
                                    llir.Mul(
                                        llir.Var(
                                            name=size_var,
                                            type=llir.DataType.INT64,
                                        ),
                                        llir.Sizeof(data_type=wksp_ctype),
                                    ),
                                ],
                            )
                        )
                else:
                    # Default: init workspace inside the loop (serial path).
                    # Save metadata for potential parallel hoisting in the transform.
                    wname = wksp.get_name()
                    self._where_workspace_name = wname
                    self._where_workspace_ctype = wksp_ctype.value
                    self._where_workspace_dim = wksp.dim
                    workspace_init_stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=wname,
                                type=llir.DataType.AUTO,
                            ),
                            value=llir.FunctionCall(
                                name=f"coo_workspace_1d<{wksp_ctype.value}, {wksp.dim}>",
                                args=[
                                    llir.Literal(
                                        value="1024",
                                        data_type=llir.DataType.INT,
                                    ),
                                ],
                            ),
                        )
                    )
                continue

            workspace_init_stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=wksp.get_name(),
                        type=llir.DataType.coo_workspace_type_with_dim(
                            wksp_ctype, wksp.dim
                        ),
                    ),
                    value=llir.FunctionCall(
                        name=f"coo_workspace<{wksp_ctype.value}, {wksp.dim}>",
                        args=[
                            llir.Literal(
                                value="1024",
                                data_type=llir.DataType.INT,
                            ),
                            llir.Var(
                                name="result_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                        ],
                    ),
                )
            )

        producer_stmts = self.lower_ProducerIndexStmt(stmt.producer)
        consumer_stmts = self.lower_ConsumerIndexStmt(stmt.consumer)
        self._where_producer_stmts = producer_stmts
        self._where_consumer_stmts = consumer_stmts
        return [
            *workspace_init_stmts,
            *self._workspace_memset_stmts,
            *producer_stmts,
            *consumer_stmts,
            *workspace_cleanup_stmts,
        ]

    def lower_ProducerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower a ProducerIndexStmt to LLIR
        """
        result = self.lower_IndexStmt(stmt)
        if isinstance(result, list):
            return result
        else:
            return [result]

    def lower_outer_ConsumerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower the outermost consumer index statement with mode_order support.

        Two paths:
        - result_is_coord: all result levels are COORDINATE, so we directly
          assign sorted workspace entries to the result tensor arrays.
        - Otherwise: build an intermediate COO tensor from workspace entries,
          move their vectors into Torch storage, then recursively lower a
          conversion CIN to produce the final result format.
        """
        workspaces = stmt.get_workspaces()
        wksp = workspaces[0]
        workspace_accesses = stmt.get_workspace_accesses()
        wksp_access: WorkspaceAccess = workspace_accesses[0]
        wksp_index_vars = wksp_access.get_index_vars()

        result_tensor_accesses = stmt.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]
        result_tensor = result_tensor_access.get_tensor()
        result_index_vars = result_tensor_access.get_index_vars()
        result_tensor_name = result_tensor.get_name()
        result_value_metadata = self._tensor_access_metadata(
            result_tensor_access,
            llir.TensorAccessRole.RESULT_WRITE,
        )

        # Check if result is all-coordinate (fast path: direct assignment)
        result_is_coord = all(
            lt == LevelType.COORDINATE
            for lt in result_tensor.get_format().get_level_types()
        )

        # Create intermediate COO tensor variable
        intermediate_tensor_var = TensorVar(
            name="T",
            fmt=TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.COORDINATE)
                    for _ in range(len(result_index_vars))
                ]
            ),
            dtype=self.result_tensor_var.dtype,
            mode_order=result_tensor.mode_order,
        )

        intermediate_tensor_iterator = llir.Var(
            name=f"p{intermediate_tensor_var.get_name()}",
            type=llir.DataType.INT64,
        )

        # Build intermediate standard vectors (only for non-coordinate path).
        intermediate_crd_vecs = []
        intermediate_val_vec = llir.Var(
            name=f"{intermediate_tensor_var.get_name()}_val_vec",
            type=llir.DataType.std_vector_type(
                dtype_to_c_datatype(intermediate_tensor_var.dtype)
            ),
        )
        vec_decl_stmts = []

        if not result_is_coord:
            for level in range(len(wksp_index_vars)):
                crd_vec = llir.Var(
                    name=f"{intermediate_tensor_var.get_name()}{level}_crd_vec",
                    type=llir.DataType.STD_VECTOR_C_INT,
                )
                intermediate_crd_vecs.append(crd_vec)
                vec_decl_stmts.append(llir.VarDecl(crd_vec))

            vec_decl_stmts.append(llir.VarDecl(intermediate_val_vec))
            vec_decl_stmts.append(
                llir.VarInit(
                    var=intermediate_tensor_iterator,
                    value=llir.Literal(0),
                )
            )

        # Sort workspace
        wksp_name = wksp.get_name()
        wksp_sort_stmt = llir.FunctionCallStmt(
            name=f"{wksp_name}.sort",
            args=[],
        )

        # Build loop: for (const auto& it : wksp) { ... }
        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        loop_array = llir.Var(name=wksp_name, type=llir.DataType.AUTO)
        loop_body: List[llir.Stmt] = []

        if result_is_coord:
            # Direct assignment: A0_crd[pA0] = it.first[0]; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_tensor_name}{i}_crd",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            index=llir.Var(
                                name=f"p{result_tensor_name}{i}",
                                type=llir.DataType.INT64,
                            ),
                        ),
                        value=llir.ArrayAccess(
                            array=llir.MemberAccess(
                                base=llir.Var(
                                    name=loop_var.name,
                                    type=loop_var.type,
                                ),
                                member="first",
                            ),
                            index=llir.Literal(i, llir.DataType.INT64),
                        ),
                    )
                )
            # A_values[pA0] = it.second;
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}_values",
                            type=llir.DataType.std_vector_type(
                                dtype_to_c_datatype(result_tensor.dtype)
                            ),
                        ),
                        index=llir.Var(
                            name=f"p{result_tensor_name}0",
                            type=llir.DataType.INT64,
                        ),
                        tensor_access=result_value_metadata,
                    ),
                    value=llir.MemberAccess(
                        base=llir.Var(
                            name=loop_var.name,
                            type=loop_var.type,
                        ),
                        member="second",
                    ),
                )
            )
            # pA0++; pA1++; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Increment(
                        var=llir.Var(
                            name=f"p{result_tensor_name}{i}",
                            type=llir.DataType.INT64,
                        ),
                    )
                )
        else:
            # Fill intermediate vectors: T0_crd_vec[pT] = it.first[0]; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=intermediate_crd_vecs[i].name,
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            index=llir.Var(
                                name=intermediate_tensor_iterator.name,
                                type=llir.DataType.INT64,
                            ),
                        ),
                        value=llir.ArrayAccess(
                            array=llir.MemberAccess(
                                base=llir.Var(
                                    name=loop_var.name,
                                    type=loop_var.type,
                                ),
                                member="first",
                            ),
                            index=llir.Literal(i, llir.DataType.INT64),
                        ),
                    )
                )
            # T_val_vec[pT] = it.second;
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=intermediate_val_vec.name,
                            type=intermediate_val_vec.type,
                        ),
                        index=llir.Var(
                            name=intermediate_tensor_iterator.name,
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.MemberAccess(
                        base=llir.Var(
                            name=loop_var.name,
                            type=loop_var.type,
                        ),
                        member="second",
                    ),
                )
            )
            # pT++;
            loop_body.append(llir.Increment(var=intermediate_tensor_iterator))

        loop_stmt = llir.ForLoopAuto(
            var=loop_var,
            array=loop_array,
            body=loop_body,
        )

        # For all-coordinate result, we're done after the loop
        if result_is_coord:
            return [
                llir.BlankLine(),
                llir.Comment("Lower outer consumer CIN"),
                *vec_decl_stmts,
                llir.BlankLine(),
                wksp_sort_stmt,
                loop_stmt,
            ]

        # Move intermediate vectors into Torch storage contexts.
        assembly_stmts = []
        intermediate_crd_tensors = []

        for i in range(len(wksp_index_vars)):
            crd_tensor = llir.Var(
                name=f"{intermediate_tensor_var.get_name()}{i}_crd_tensor",
                type=llir.DataType.TORCH_TENSOR,
            )
            intermediate_crd_tensors.append(crd_tensor)

            assembly_stmts.append(
                llir.VarInit(
                    var=crd_tensor,
                    value=llir.FunctionCall(
                        name="scorch_tensor_from_vector",
                        args=[
                            llir.FunctionCall(
                                name="std::move",
                                args=[
                                    llir.Var(
                                        name=intermediate_crd_vecs[i].name,
                                        type=intermediate_crd_vecs[i].type,
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

            # int* T0_crd = T0_crd_tensor.data_ptr<int>();
            assembly_stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{intermediate_tensor_var.get_name()}{i}_crd",
                        type=llir.DataType.PTR_INT,
                    ),
                    value=tensor_data_ptr(
                        llir.Var(
                            name=crd_tensor.name,
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        llir.DataType.INT,
                    ),
                )
            )

        val_tensor = llir.Var(
            name=f"{intermediate_tensor_var.get_name()}_val_tensor",
            type=llir.DataType.TORCH_TENSOR,
        )
        assembly_stmts.append(
            llir.VarInit(
                var=val_tensor,
                value=llir.FunctionCall(
                    name="scorch_tensor_from_vector",
                    args=[
                        llir.FunctionCall(
                            name="std::move",
                            args=[
                                llir.Var(
                                    name=intermediate_val_vec.name,
                                    type=intermediate_val_vec.type,
                                )
                            ],
                        ),
                        llir.QualifiedName(
                            namespace="torch",
                            name=get_pytorch_c_dtype_name(
                                intermediate_tensor_var.dtype
                            ),
                            data_type=llir.DataType.TORCH_SCALAR_TYPE,
                        ),
                    ],
                ),
            )
        )

        # float* T_val = T_val_tensor.data_ptr<float>();
        data_type = dtype_to_c_datatype(intermediate_tensor_var.dtype)
        ptr_type = llir.DataType.ptr_type(intermediate_tensor_var.dtype)
        assembly_stmts.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{intermediate_tensor_var.get_name()}_val",
                    type=ptr_type,
                ),
                value=tensor_data_ptr(
                    llir.Var(
                        name=val_tensor.name,
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    data_type,
                ),
            )
        )

        # Build conversion CIN: result[i,j,...] = T[i,j,...]
        # then wrap in ForAll loops in mode_order and recursively lower
        sorted_result_index_vars = result_tensor_access.get_sorted_index_vars()
        access_key = (
            result_index_vars[0]
            if len(result_index_vars) == 1
            else tuple(result_index_vars)
        )
        rhs_access = intermediate_tensor_var[access_key]
        lhs_access = result_tensor[access_key]
        cin_stmt: IndexStmt = TensorAssign(lhs_access, rhs_access)
        for index_var in reversed(sorted_result_index_vars):
            cin_stmt = ForAll(index_var, cin_stmt)

        # Recursively lower the conversion CIN
        result_conversion_stmts = self.lower_IndexStmt(cin_stmt)
        if not isinstance(result_conversion_stmts, list):
            result_conversion_stmts = [result_conversion_stmts]

        return [
            llir.BlankLine(),
            llir.Comment("Lower outer consumer CIN"),
            *vec_decl_stmts,
            llir.BlankLine(),
            wksp_sort_stmt,
            loop_stmt,
            llir.BlankLine(),
            *assembly_stmts,
            llir.BlankLine(),
            *result_conversion_stmts,
        ]

    def lower_ConsumerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower a ConsumerIndexStmt to LLIR
        """
        if stmt.parent == self.outermost_stmt:
            return self.lower_outer_ConsumerIndexStmt(stmt)

        workspaces = stmt.get_workspaces()
        wksp = workspaces[0]
        workspace_accesses = stmt.get_workspace_accesses()
        wksp_access: WorkspaceAccess = workspace_accesses[0]
        wksp_index_vars = wksp_access.get_index_vars()

        result_tensor_accesses = stmt.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]
        result_tensor_name = result_tensor_access.get_tensor().get_name()
        result_value_metadata = self._tensor_access_metadata(
            result_tensor_access,
            llir.TensorAccessRole.RESULT_WRITE,
        )

        # If the wksp_index_var is None, that means we just have a scalar
        # workspace
        if not wksp_index_vars:
            stmts: List[llir.Stmt] = []

            index_var = result_tensor_access.get_sorted_index_vars()[-1]
            level = result_tensor_access.level_of_index_var(index_var)
            level_type = result_tensor_access.level_type_of_index_var(index_var)

            wksp_var = llir.Var(
                name=f"{wksp.get_name()}",
                type=llir.DataType.NO_TYPE,
            )

            # Guard assignment: if wksp != 0
            if level_type == LevelType.DENSE:
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=[
                            llir.Assign(
                                var=llir.ArrayAccess(
                                    array=llir.Var(
                                        name=f"{result_tensor_name}_values",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    index=llir.Var(
                                        name=f"p{result_tensor_name}{level}",
                                        type=llir.DataType.INT64,
                                    ),
                                    tensor_access=result_value_metadata,
                                ),
                                value=wksp_var,
                            ),
                        ],
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=[
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}{level}_crd.push_back",
                                args=[
                                    llir.Var(
                                        name=index_var.name,
                                        type=llir.DataType.INT64,
                                    )
                                ],
                            ),
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}_values.push_back",
                                args=[wksp_var],
                            ),
                            llir.Increment(
                                var=llir.Var(
                                    name=f"p{result_tensor_name}{level}",
                                    type=llir.DataType.INT64,
                                )
                            ),
                        ],
                    )
                )
            elif level_type == LevelType.COORDINATE:
                # For COO, push coordinates at ALL coordinate levels
                sorted_ivars = result_tensor_access.get_sorted_index_vars()
                push_stmts: List[llir.Stmt] = []
                for lvl, ivar in enumerate(sorted_ivars):
                    lt = result_tensor_access.level_type_of_index_var(ivar)
                    if lt == LevelType.COORDINATE:
                        push_stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}{lvl}_crd.push_back",
                                args=[
                                    llir.Var(
                                        name=ivar.name,
                                        type=llir.DataType.INT64,
                                    )
                                ],
                            )
                        )
                push_stmts.append(
                    llir.FunctionCallStmt(
                        name=f"{result_tensor_name}_values.push_back",
                        args=[wksp_var],
                    )
                )
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=push_stmts,
                    )
                )
            else:
                raise NotImplementedError(
                    f"TODO: need to handle assembly of workspace with {level_type} level"
                )
            return [
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                *stmts,
                llir.BlankLine(),
            ]

        wksp_last_index_var = wksp_index_vars[-1]

        if wksp_last_index_var.has_parent:
            curr_index_var = wksp_last_index_var.parent
        else:
            curr_index_var = wksp_last_index_var

        level = result_tensor_access.level_of_index_var(curr_index_var)
        level_type = result_tensor_access.level_type_of_index_var(curr_index_var)

        parent_index_var = None
        parent_level_type = None
        if level > 0:
            parent_index_var = result_tensor_access.get_parent_index_var(curr_index_var)
            assert parent_index_var is not None, "parent_index_var should not be None"
            parent_level_type = result_tensor_access.level_type_of_index_var(
                parent_index_var
            )

        # p<result tensor's name><result level>
        result_level_iterator_name = f"p{result_tensor_name}{level}"
        result_level_iterator_llir = llir.Var(
            name=result_level_iterator_name,
            type=llir.DataType.NO_TYPE,
        )

        # call .sort() on the workspace
        # <wksp's name>.sort();
        wksp_sort_stmt = llir.FunctionCallStmt(
            name=f"{wksp.get_name()}.sort",
            args=[],
        )

        # Dense accumulator workspace: write to the result tensor.
        # When the result level is dense and contiguous, use memcpy
        # (pure store, avoids cold read-modify-write on large output).
        if wksp_access.is_dense():
            assert (
                len(wksp_index_vars) == 1
            ), "dense workspace has more than 1 index var"
            wksp_index_var = wksp_index_vars[0]

            if not wksp_index_var.tile_size_var:
                # Check if the result level is dense (contiguous layout).
                result_level_type = result_tensor_access.level_type_of_index_var(
                    wksp_index_var
                )
                result_is_dense = (
                    result_level_type is not None and result_level_type.name == "DENSE"
                )

                if result_is_dense:
                    # Emit memcpy: the workspace has the full row, write once.
                    wname = wksp.get_name()
                    size_var = wksp_index_var.size_llir_var.name
                    wksp_write_ctype = dtype_to_c_datatype(wksp.dtype)
                    # The iterator for the row start: pC<level> with j=0
                    # is just pC_prev * C_level_size (which is result_level_iterator_name with j=0).
                    # We can compute it as: &C_values[pC0 * C1_size]
                    row_start: llir.Expr = (
                        llir.Var(
                            name=f"p{result_tensor_name}{level - 1}",
                            type=llir.DataType.INT64,
                        )
                        if level > 0
                        else llir.Literal(value=0, data_type=llir.DataType.INT)
                    )
                    return [
                        llir.BlankLine(),
                        llir.Comment("Write workspace to output (memcpy — pure store)"),
                        llir.FunctionCallStmt(
                            name="memcpy",
                            args=[
                                llir.AddressOf(
                                    operand=llir.ArrayAccess(
                                        array=llir.Var(
                                            name=f"{result_tensor_name}_values",
                                            type=llir.DataType.ptr_type(
                                                wksp_write_ctype
                                            ),
                                        ),
                                        index=llir.Mul(
                                            row_start,
                                            llir.Var(
                                                name=(
                                                    f"{result_tensor_name}"
                                                    f"{level}_size"
                                                ),
                                                type=llir.DataType.INT64,
                                            ),
                                        ),
                                    ),
                                ),
                                llir.Var(
                                    name=wname,
                                    type=llir.DataType.ptr_type(wksp_write_ctype),
                                ),
                                llir.Mul(
                                    llir.Var(
                                        name=size_var,
                                        type=llir.DataType.INT64,
                                    ),
                                    llir.Sizeof(data_type=wksp_write_ctype),
                                ),
                            ],
                        ),
                    ]

                # Fallback: element-by-element assignment (= not +=)
                loop_var = llir.Var(
                    name=f"{wksp_index_var.name}",
                    type=llir.DataType.INT64,
                )

                loop_body: List[llir.Stmt] = []

                loop_body.extend(
                    result_tensor_access.get_level_iterator_resolve_stmts(level=level)
                )

                loop_body.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_tensor_name}_values",
                                type=llir.DataType.NO_TYPE,
                            ),
                            index=llir.Var(
                                name=result_level_iterator_name,
                                type=llir.DataType.INT64,
                            ),
                            tensor_access=result_value_metadata,
                        ),
                        value=llir.Var(
                            name=f"{wksp.get_name()}[{loop_var.name}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        op=AssignOp.ASSIGN,
                    )
                )

                # Inject post-ops (bias add, relu, etc.) after workspace -> output write
                loop_body.extend(
                    self._emit_post_ops(
                        f"{result_tensor_name}_values",
                        result_level_iterator_name,
                        result_value_metadata,
                    )
                )

                for_loop = llir.ForLoop(
                    init=llir.VarInit(
                        var=loop_var,
                        value=llir.Literal(0),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=loop_var,
                        right=wksp_index_var.size_llir_var,
                    ),
                    update=llir.Increment(
                        var=loop_var,
                    ),
                    body=loop_body,
                )
                return [
                    llir.BlankLine(),
                    llir.Comment("Write workspace to output"),
                    for_loop,
                ]

            assert (
                wksp_index_var.tile_size_var and wksp_index_var.is_inner
            ), "Dense accumulator used not for tiling"

            # For loop
            # for (int <wksp index var> = 0; <wksp index var> < <wksp index var bound>; <wksp index var>++) {
            #    <body statement>
            # }
            loop_var = llir.Var(
                name=f"{wksp_index_var.name}",
                type=llir.DataType.INT64,
            )

            loop_body: List[llir.Stmt] = []

            # <result tensor name>_values[<result level iterator>] = <wksp's name>[<wksp index var>];

            loop_body.extend(wksp_index_var.parent.get_resolve_llir_stmts())

            # The producer has the same guard in iter_lattice. This must apply
            # to every tuner-requested ragged tile, not only legacy regblock mode.
            _bound = f"{result_tensor_name}{level}_size"
            loop_body.append(
                llir.IfThenElse(
                    cond=llir.BinOp(
                        op=">=",
                        left=llir.Var(
                            name=wksp_index_var.parent.name,
                            type=llir.DataType.INT,
                        ),
                        right=llir.Var(name=_bound, type=llir.DataType.INT),
                    ),
                    then_body=[llir.Break()],
                )
            )

            loop_body.extend(
                result_tensor_access.get_level_iterator_resolve_stmts(level=level)
            )

            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}_values",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=result_level_iterator_name,
                            type=llir.DataType.INT64,
                        ),
                        tensor_access=result_value_metadata,
                    ),
                    value=llir.ArrayAccess(
                        array=llir.Var(
                            name=wksp.get_name(),
                            type=llir.DataType.ptr_type(wksp.dtype),
                        ),
                        index=llir.Var(
                            name=loop_var.name,
                            type=loop_var.type,
                        ),
                    ),
                    op=AssignOp.ADD_ASSIGN,
                )
            )

            for_loop = llir.ForLoop(
                init=llir.VarInit(
                    var=loop_var,
                    value=llir.Literal(0),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=loop_var,
                    right=llir.Var(
                        name=wksp_index_var.tile_size_var.name,
                        type=llir.DataType.INT64,
                    ),
                ),
                update=llir.Increment(
                    var=loop_var,
                ),
                body=loop_body,
                unroll=wksp_index_var.tile_size_var.unroll,
            )

            return [
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                for_loop,
            ]

        # For loop
        # for (const auto& pair : <wksp's name>) {
        #    <body statement>
        # }

        loop_var = llir.Var(
            name="it",
            type=llir.DataType.CONST_AUTO_REF,
        )

        loop_array = llir.Var(
            name=f"{wksp.get_name()}",
            type=llir.DataType.AUTO,
        )

        loop_body: List[llir.Stmt] = []

        # int <wksp_access's first index var's name> = it->first[0];
        # int <wksp_access's second index var's name> = it->first[1];
        # ...
        # DONE: if the workspace is one dimensional, then just do .first without the index
        # vars
        if len(wksp_access.get_index_vars()) == 1:
            loop_body.append(
                llir.VarInit(
                    var=llir.Var(
                        name=wksp_access.get_index_vars()[0].name,
                        type=llir.DataType.INT64,
                    ),
                    value=llir.MemberAccess(
                        base=llir.Var(
                            name=loop_var.name,
                            type=loop_var.type,
                        ),
                        member="first",
                    ),
                )
            )
        else:
            for i, index_var in enumerate(wksp_access.get_index_vars()):
                loop_body.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=index_var.name,
                            type=llir.DataType.INT64,
                        ),
                        value=llir.ArrayAccess(
                            array=llir.MemberAccess(
                                base=llir.Var(
                                    name=loop_var.name,
                                    type=loop_var.type,
                                ),
                                member="first",
                            ),
                            index=llir.Literal(i, llir.DataType.INT64),
                        ),
                    )
                )
        # <wksp's ctype> <wksp's name>_value = it->second;
        loop_body.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{wksp.get_name()}_value",
                    type=dtype_to_c_datatype(wksp.dtype),
                ),
                value=llir.MemberAccess(
                    base=llir.Var(
                        name=loop_var.name,
                        type=loop_var.type,
                    ),
                    member="second",
                ),
            )
        )

        # Add blank line
        loop_body.append(llir.BlankLine())

        # <result tensor name>_values[<result level iterator>] = <wksp's name>_value;
        loop_body.append(
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_tensor_name}_values",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(
                        name=result_level_iterator_name,
                        type=llir.DataType.INT64,
                    ),
                    tensor_access=result_value_metadata,
                ),
                value=llir.Var(
                    name=f"{wksp.get_name()}_value",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # Set coordinate
        # <result tensor name><level>_crd[<result level iterator>] = <wksp_access's first index var's name>;
        loop_body.append(
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_tensor_name}{level}_crd",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(
                        name=result_level_iterator_name,
                        type=llir.DataType.INT64,
                    ),
                ),
                value=llir.Var(
                    name=f"{wksp_access.get_index_vars()[0].name}",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # If the parent level is COORDINATE, also set the parent level's coordinate
        if parent_level_type == LevelType.COORDINATE:
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{level - 1}_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=result_level_iterator_name,
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.Var(
                        name=f"{parent_index_var.name}",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

        # <result level iterator>++;
        loop_body.append(
            llir.Increment(
                var=result_level_iterator_llir,
            )
        )

        loop_stmt = llir.ForLoopAuto(
            var=loop_var,
            array=loop_array,
            body=loop_body,
        )

        assembly_stmts: List[llir.Stmt] = []

        if level_type == LevelType.COMPRESSED:
            assembly_stmts.extend(
                [
                    llir.BlankLine(),
                    llir.Comment("Assembly compressed _level indices"),
                ]
            )

            # if _level is > 0 and parent _level is also sparse, we need to set
            # the parent _level's crd
            if level > 0:
                assert parent_index_var is not None, "Parent index var is None"
                if parent_level_type == LevelType.COMPRESSED:
                    assembly_stmts.append(
                        # e.g.
                        # if (A1_pos.back() < pA1) {
                        #     A0_crd.push_back(i);
                        # }
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op="<",
                                left=llir.FunctionCall(
                                    name=f"{result_tensor_name}{level}_pos.back",
                                    args=[],
                                ),
                                right=llir.Var(
                                    name=f"p{result_tensor_name}{level}",
                                    type=llir.DataType.INT64,
                                ),
                            ),
                            then_body=[
                                llir.FunctionCallStmt(
                                    name=f"{result_tensor_name}{level - 1}_crd.push_back",
                                    args=[
                                        llir.Var(
                                            name=parent_index_var.name,
                                            type=llir.DataType.INT64,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
            # Assemble pos array for this compressed level:
            # - Dense parent: A1_pos.push_back(A1_crd.size())
            # - Compressed parent: A1_pos[A0_crd.size()] = A1_crd.size()
            assembled_pos_array = False
            if level > 0:
                assert parent_index_var is not None, "Parent index var is None"
                if parent_level_type == LevelType.COMPRESSED:
                    # A1_pos[A0_crd.size()] = A1_crd.size()
                    assembly_stmts.append(
                        llir.Assign(
                            var=llir.ArrayAccess(
                                array=llir.Var(
                                    name=f"{result_tensor_name}{level}_pos",
                                    type=llir.DataType.STD_VECTOR_C_INT,
                                ),
                                index=llir.FunctionCall(
                                    name=(
                                        f"{result_tensor_name}" f"{level - 1}_crd.size"
                                    ),
                                    args=[],
                                ),
                            ),
                            value=llir.FunctionCall(
                                name=f"{result_tensor_name}{level}_crd.size",
                                args=[],
                            ),
                        )
                    )
                    assembled_pos_array = True

            if not assembled_pos_array:
                # e.g. A1_pos[pA1] = A1_crd.size()
                # assembly_stmts.append(
                #     llir.Assign(
                #         var=llir.Var(
                #             name=f"{result_tensor_name}{level}_pos[p{result_tensor_name}{level}]",
                #             type=llir.DataType.INT64,
                #         ),
                #         value=llir.FunctionCall(
                #             name=f"{result_tensor_name}{level}_crd.size",
                #             args=[],
                #         ),
                #     )
                # )
                # e.g. A1_pos[A1_pos_index + 1] = A1_crd.size()
                assembly_stmts.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_tensor_name}{level}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            index=llir.Add(
                                llir.Var(
                                    name=f"{result_tensor_name}{level}_pos_index",
                                    type=llir.DataType.INT64,
                                ),
                                llir.Literal(1),
                            ),
                        ),
                        value=llir.FunctionCall(
                            name=f"{result_tensor_name}{level}_crd.size",
                            args=[],
                        ),
                    )
                )
                # assembly_stmts.append(
                #     # e.g. A1_pos.push_back(pA1))
                #     llir.FunctionCallStmt(
                #         name=f"{result_tensor_name}{level}_pos.push_back",
                #         args=[
                #             llir.Var(
                #                 name=f"{result_tensor_name}{level}_crd.size()",
                #                 # name=f"p{result_tensor_var.name}{_level}",
                #                 type=llir.DataType.INT64,
                #             )
                #         ],
                #     )
                # )

        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            wksp_sort_stmt,
            loop_stmt,
            llir.BlankLine(),
            *assembly_stmts,
        ]

    def _prepare_scheduled_cin(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse: bool,
        ownership_transferred: bool,
    ) -> IndexStmt:
        if recurse or self.outermost_stmt is not None:
            if isinstance(stmt, ScheduledCIN):
                raise CompilerInvariantError(
                    "stage=CIN lowering: ScheduledCIN is valid only at the "
                    "outer scheduling boundary"
                )
            self._validate_index_stmt(stmt)
            return stmt

        plan = None
        if isinstance(stmt, ScheduledCIN):
            verify_scheduled_cin(stmt)
            self.loop_plan = stmt.verified_loop_plan
            plan = stmt.verified_loop_plan
            stmt = stmt.normalized_cin
            self.normalized_cin = stmt
        elif self.compile_options.requested_schedule is not None:
            raise CompileOptionsError(
                (
                    CompileOptionsDiagnostic(
                        code="unsupported_requested_schedule",
                        field="requested_schedule",
                        message=(
                            "plain CIN lowering cannot ignore a requested schedule; "
                            "lower a verified ScheduledCIN"
                        ),
                    ),
                )
            )

        self._validate_index_stmt(stmt)
        verify_cin_if_enabled(
            stmt,
            enabled=self.compile_options.verification.verify_cin,
        )
        if ownership_transferred:
            return claim_legacy_cin_working_tree(
                stmt,
                plan,
                compile_options=self.compile_options,
            )
        return legacy_cin_working_copy(
            stmt,
            plan,
            compile_options=self.compile_options,
        )

    def _lower_owned_IndexStmt(
        self, stmt: Union[IndexStmt, ScheduledCIN]
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Lower a detached compiler-local tree after transferring ownership."""

        return self.lower_IndexStmt(stmt, _ownership_transferred=True)

    def _lower_outer_body(self, stmt: IndexStmt) -> _LoweredOuterBody:
        """Lower the already-validated outer body and retain pass ownership data."""

        if isinstance(stmt, ForAll):
            return self.lower_ForAll(stmt)
        if isinstance(stmt, Where):
            lowered = self.lower_Where(stmt)
            statements = lowered if isinstance(lowered, list) else [lowered]
            return _LoweredOuterBody(statements)
        raise CompilerInvariantError(
            "stage=CIN lowering: outer function body must be ForAll or Where"
        )

    def lower_IndexStmt(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse=False,
        _ownership_transferred: bool = False,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower an IndexStmt to LLIR
        """

        compilation_context = self._compilation_context
        if compilation_context is None or recurse or self.outermost_stmt is not None:
            return self._lower_index_stmt_untimed(
                stmt,
                recurse,
                _ownership_transferred,
            )
        adapter_token = compilation_context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=self.compile_options,
        )
        try:
            prepared_stmt = self._prepare_scheduled_cin(
                stmt,
                recurse,
                ownership_transferred=_ownership_transferred,
            )
        except Exception:
            compilation_context.fail_stage(adapter_token)
            raise
        compilation_context.complete_stage(adapter_token)
        lowering_token = compilation_context.begin_stage(
            CompilerStageId.CIN_LOWERING,
            compile_options=self.compile_options,
        )
        try:
            lowered = self._lower_prepared_index_stmt(prepared_stmt, recurse)
        except Exception:
            compilation_context.fail_stage(lowering_token)
            raise
        compilation_context.complete_stage(lowering_token)
        return self._apply_schedule_lowering(lowered, prepared_stmt)

    def _lower_index_stmt_untimed(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse: bool,
        _ownership_transferred: bool,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        apply_schedule_lowering = not recurse and self.outermost_stmt is None
        stmt = self._prepare_scheduled_cin(
            stmt,
            recurse,
            ownership_transferred=_ownership_transferred,
        )
        lowered = self._lower_prepared_index_stmt(stmt, recurse)
        if apply_schedule_lowering:
            return self._apply_schedule_lowering(lowered, stmt)
        return lowered

    def _lower_prepared_index_stmt(
        self,
        stmt: IndexStmt,
        recurse: bool,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Lower one already adapted compiler-owned CIN tree to current LLIR."""

        if not self.outermost_stmt:
            self.outermost_stmt = stmt
            self.has_explicit_parallel_loop = self._contains_explicit_parallel(stmt)

        if isinstance(stmt, TensorAssign):
            return self.lower_TensorAssign(stmt)

        # loop_order_allow_short_circuit = all_free_var_loops_before_reduction_loops(stmt)

        # Create tensor results and rhs IR variables
        result_tensor_vars: List[TensorVar] = stmt.get_result_tensor_vars()
        # TODO: need to handle multiple result tensors
        self.result_tensor_var = result_tensor_vars[0]
        non_workspace_result_tensor_vars = [
            x for x in result_tensor_vars if not isinstance(x, Workspace)
        ]
        if not self.final_result_tensor_var:
            self.final_result_tensor_var = (
                non_workspace_result_tensor_vars[0]
                if non_workspace_result_tensor_vars
                else None
            )
        result_tensor_accesses = stmt.get_result_tensor_accesses()
        self.result_tensor_access = result_tensor_accesses[0]
        non_workspace_result_tensor_accesses = [
            x for x in result_tensor_accesses if not isinstance(x.tensor, Workspace)
        ]
        if not self.final_result_tensor_access:
            self.final_result_tensor_access = (
                non_workspace_result_tensor_accesses[0]
                if non_workspace_result_tensor_accesses
                else None
            )

        rhs_tensor_vars: List[TensorVar] = stmt.get_rhs_tensor_vars()
        self._value_array_ctypes = {
            f"{tensor.name}_val": dtype_to_c_datatype(tensor.dtype).value
            for tensor in rhs_tensor_vars
        }
        rhs_tensor_accesses: List[TensorAccess] = stmt.get_rhs_tensor_accesses()
        # rhs_tensor_vars_llir: List[llir.Expr] = [
        #     self.lower_TensorVar(tv) for tv in rhs_tensor_vars
        # ]

        tile_size_vars = stmt.get_tile_size_vars()
        tile_size_vars_init_stmts: List[llir.Stmt] = (
            [llir.BlankLine(), llir.Comment("Initialize tile sizes")]
            if tile_size_vars
            else []
        )
        for tile_size_var in tile_size_vars:
            tile_size_vars_init_stmts.append(tile_size_var.llir_var_init)

        self.need_compute.extend(result_tensor_vars)

        if recurse or stmt != self.outermost_stmt:
            if isinstance(stmt, ForAll):
                return self.lower_ForAll(stmt).statements
            if isinstance(stmt, Where):
                return self.lower_Where(stmt)

        result_contract = self.final_result_tensor_var or self.result_tensor_var
        assert result_contract is not None, "No result tensor ABI contract"
        kernel_abi = _torch_cpp_kernel_abi(
            result_contract,
            rhs_tensor_vars,
            self.post_ops,
        )

        tensor_value_array_init_stmts: List[llir.Stmt] = []
        result_level_indices_init_stmts: List[llir.Stmt] = []
        # A compressed level below a dense result parent always has one
        # position slot per parent coordinate plus the sentinel. Sparse-driven
        # conversions benefit from sizing it exactly at every shape. For dense
        # scans, doing so is reserved for large tensors: it relieves enough
        # inner-loop register pressure to matter there, while append-built
        # positions remain faster for small dense conversions.
        result_shape = (
            self.final_result_tensor_var.shape
            if self.final_result_tensor_var is not None
            else None
        )
        result_cells = 0
        if result_shape:
            result_cells = 1
            for extent in result_shape:
                result_cells *= extent
        exact_dense_parent_positions = len(rhs_tensor_vars) == 1 and (
            not rhs_tensor_vars[0].is_dense() or result_cells >= 1024 * 1024
        )
        reserve_hint_var = None
        if (
            len(rhs_tensor_vars) == 1
            and rhs_tensor_vars[0].is_dense()
            and self.final_result_tensor_var is not None
            and not self.final_result_tensor_var.is_dense()
            and self.final_result_tensor_var.levels == 2
        ):
            reserve_hint_var = "_dynamic_reserve"

        for result_tensor_var in non_workspace_result_tensor_vars:
            self.tensor_var_to_llir[result_tensor_var] = self.lower_TensorVar(
                result_tensor_var
            )
            assembler = _result_tensor_abi_assembler(
                result_tensor_var,
                exact_dense_parent_positions=exact_dense_parent_positions,
                reserve_hint_var=reserve_hint_var,
            )
            tensor_value_array_init_stmts.extend(assembler.emit_value_array_init())
            result_level_indices_init_stmts.extend(assembler.emit_level_indices_init())

        if result_level_indices_init_stmts:
            result_level_indices_init_stmts = [
                llir.Comment("Init result level indices"),
                *result_level_indices_init_stmts,
            ]

        # Snapshot-owned public input bindings retain their original construction
        # point before recursive compute lowering and the managed pass pipeline.
        tensor_level_array_assign_stmts = kernel_abi.emit_input_prologue()

        # Generate per-level size variables for each dense level in result tensor
        result_tensor_level_sizes: List[llir.Stmt] = []
        for i, level_type in enumerate(self.result_tensor_var.get_level_types()):
            if level_type == LevelType.DENSE:
                result_tensor_level_sizes.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"{self.result_tensor_var.get_name()}{i}_size",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.ArrayAccess(
                            array=llir.Var(
                                name="result_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                            index=llir.Literal(
                                value=i,
                                data_type=llir.DataType.INT64,
                            ),
                        ),
                    )
                )

        if reserve_hint_var:
            result_tensor_level_sizes.append(
                llir.RawStmt(
                    code=(
                        f"int64_t {reserve_hint_var} = std::min<int64_t>("
                        "scorch_native::checked_product("
                        'result_shape, "evaluate", "result_shape", true), '
                        "2048)"
                    )
                )
            )

        if result_tensor_level_sizes:
            result_tensor_level_sizes = [
                llir.Comment("Init result tensor level sizes"),
                *result_tensor_level_sizes,
            ]

            # A mapping from IndexVar to a list of (TensorVar, _level: int, LevelType) tuples
        self.index_var_to_rhs_tensor_level_type = {}
        for tensor_access in rhs_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            mode_order = tensor_var.get_mode_order()
            for i, index_var in enumerate(index_vars):
                index_var_level = mode_order[i]
                if index_var not in self.index_var_to_rhs_tensor_level_type:
                    self.index_var_to_rhs_tensor_level_type[index_var] = []
                self.index_var_to_rhs_tensor_level_type[index_var].append(
                    [tensor_var, index_var_level, tensor_level_types[index_var_level]]
                )

        self.index_var_to_result_tensor_level_type = {}
        for tensor_access in result_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            if not index_vars:
                continue
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            mode_order = tensor_var.get_mode_order()
            for i, index_var in enumerate(index_vars):
                index_var_level = mode_order[i]
                if index_var not in self.index_var_to_result_tensor_level_type:
                    self.index_var_to_result_tensor_level_type[index_var] = []
                self.index_var_to_result_tensor_level_type[index_var].append(
                    [tensor_var, index_var_level, tensor_level_types[index_var_level]]
                )

        # Initialize index into result if any _level if compressed
        # Find last compressed _level of the result tensor, if any
        result_last_compressed_index_var = None
        for (
            index_var,
            tensor_level_type_list,
        ) in self.index_var_to_result_tensor_level_type.items():
            # TODO: deal with multiple outputs
            tensor_var, level, level_type = tensor_level_type_list[0]
            if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                result_last_compressed_index_var = index_var

        result_index_init_stmts = []

        if result_last_compressed_index_var is not None:
            self.result_value_array_sparse_index_llir = llir.Var(
                # name=f"p{self.result_tensor_var.name}{self.result_tensor_access.level_of_index_var(result_last_compressed_index_var)}",
                name=f"p{self.result_tensor_var.name}{self.result_tensor_var.levels - 1}",
                type=llir.DataType.INT64,
            )
            self.result_tensor_value_index_var_dict[
                result_last_compressed_index_var
            ] = self.result_value_array_sparse_index_llir

            result_index_init_stmts.append(
                llir.VarInit(
                    var=self.result_value_array_sparse_index_llir,
                    value=llir.Literal(value=0, data_type=llir.DataType.INT64),
                )
            )

        # Finally, return function that computes the result
        if stmt == self.outermost_stmt:
            # Validate public by-value arguments before any nested mode-index
            # access or data_ptr call. Validation may replace only these local
            # handles with checked int32 copies; caller-owned tensors stay intact.
            abi_validation_stmts = kernel_abi.emit_validation()
            postop_ptr_stmts = kernel_abi.emit_extra_tensor_prologue()

            # The former ``lower_IndexStmt(..., recurse=True)`` path repeated
            # this bookkeeping before descending.  Keep that observable
            # instance state stable while returning pass ownership explicitly.
            self.need_compute.extend(result_tensor_vars)
            recurse_result = self._lower_outer_body(stmt)

            def assemble_body(
                transformed_body: LLIRStatementListArtifact,
                compressed_output_parallel: bool,
            ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
                recurse_stmts = transformed_body.statements
                body_stmts: List[llir.Stmt] = []
                assembled_value_init_stmts = tensor_value_array_init_stmts
                assembled_level_indices_stmts = result_level_indices_init_stmts

                # Known-nnz detection: if scalar accumulation was used and output is
                # sparse, nnz_out == nnz_in. Re-emit init with Torch-owned storage.
                known_nnz_init_stmts: List[llir.Stmt] = []
                if (
                    self._used_scalar_accum
                    and self.final_result_tensor_var
                    and not self.final_result_tensor_var.is_dense()
                ):
                    # Scalar accumulation preserves the driving sparse leaf. Its
                    # value tensor has the exact output cardinality for either COO
                    # or a compressed-leaf format such as CSR.
                    sparse_values_tensor = None
                    for tensor in rhs_tensor_vars:
                        level_types = tensor.get_level_types()
                        if level_types and level_types[-1] in (
                            LevelType.COMPRESSED,
                            LevelType.COORDINATE,
                        ):
                            sparse_values_tensor = f"{tensor.get_name()}_values"
                            break

                    if sparse_values_tensor:
                        self._known_nnz_var = "_known_nnz"
                        known_nnz_init_stmts.append(
                            llir.VarInit(
                                var=llir.Var(
                                    name="_known_nnz",
                                    type=llir.DataType.INT64,
                                ),
                                value=llir.MemberCall(
                                    base=llir.Var(
                                        name=sparse_values_tensor,
                                        type=llir.DataType.TORCH_TENSOR,
                                    ),
                                    member="size",
                                    args=(
                                        llir.Literal(
                                            value=0,
                                            data_type=llir.DataType.INT64,
                                        ),
                                    ),
                                ),
                            )
                        )

                        # Re-emit init stmts with known_nnz_var
                        assembled_value_init_stmts = []
                        assembled_level_indices_stmts = []
                        for result_tensor_var in non_workspace_result_tensor_vars:
                            assembler = _result_tensor_abi_assembler(
                                result_tensor_var,
                                known_nnz_var=self._known_nnz_var,
                                exact_dense_parent_positions=(
                                    exact_dense_parent_positions
                                ),
                                reserve_hint_var=reserve_hint_var,
                            )
                            assembled_value_init_stmts.extend(
                                assembler.emit_value_array_init()
                            )
                            assembled_level_indices_stmts.extend(
                                assembler.emit_level_indices_init()
                            )
                        if assembled_level_indices_stmts:
                            assembled_level_indices_stmts = [
                                llir.Comment("Init result level indices"),
                                *assembled_level_indices_stmts,
                            ]

                if compressed_output_parallel:
                    # Two-phase parallel transform already emitted Torch-owned
                    # output storage, fill loops, and final assembly.
                    # Only emit tensor level arrays (input pointers) and recurse.
                    body_stmts.extend(
                        [
                            *abi_validation_stmts,
                            *result_tensor_level_sizes,
                            *tensor_level_array_assign_stmts,
                            llir.BlankLine(),
                            *tile_size_vars_init_stmts,
                            *postop_ptr_stmts,
                            llir.BlankLine(),
                            *recurse_stmts,
                        ]
                    )
                else:
                    body_stmts.extend(
                        [
                            *abi_validation_stmts,
                            *result_tensor_level_sizes,
                            *tensor_level_array_assign_stmts,
                            llir.BlankLine(),
                            *known_nnz_init_stmts,
                            *assembled_level_indices_stmts,
                            llir.Comment("Initialize result value array"),
                            *assembled_value_init_stmts,
                            *tile_size_vars_init_stmts,
                            *postop_ptr_stmts,
                            # *result_index_init_stmts,
                            llir.BlankLine(),
                            *recurse_stmts,
                        ]
                    )

                    assert (
                        self.final_result_tensor_var is not None
                    ), "No final result tensor"
                    final_assembler = _result_tensor_abi_assembler(
                        self.final_result_tensor_var,
                        known_nnz_var=self._known_nnz_var,
                        exact_dense_parent_positions=exact_dense_parent_positions,
                        reserve_hint_var=reserve_hint_var,
                    )
                    body_stmts.extend(final_assembler.emit_final_assembly())

                return LLIRRewriteArtifact(body_stmts)

            # Same one-shot continuation at the same lazy barrier position;
            # instrumentation only brackets its existing execution.
            body_assembler = self._instrument_body_assembler(assemble_body)

            try:
                pipeline_result = self._llir_pass_manager.run_production_pipeline(
                    LLIRStatementListArtifact(recurse_result.statements),
                    compressed_where_pass_spec=(
                        recurse_result.compressed_where_pass_spec
                    ),
                    dense_pointer_pass_spec=DensePointerHoistPassSpec(
                        DensePointerHoistContext(
                            value_array_ctypes=tuple(self._value_array_ctypes.items())
                        )
                    ),
                    body_assembler=body_assembler,
                )
            except LLIRPassPartialFailure as failure:
                self._record_llir_pass_runs(failure.completed_run_records)
                raise failure.failure from None
            self._record_llir_pass_runs(pipeline_result.run_records)
            body_stmts = pipeline_result.artifact.value
            return kernel_abi.assemble_function(body_stmts)

        return []

    def _apply_schedule_lowering(
        self,
        lowered: Union[llir.Stmt, List[llir.Stmt]],
        stmt: IndexStmt,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Apply an existing LoopPlan after CIN lowering has completed."""

        if self.loop_plan is None or not isinstance(lowered, llir.Function):
            return lowered
        compilation_context = self._compilation_context
        schedule_lowering_token = (
            compilation_context.begin_stage(
                CompilerStageId.SCHEDULE_LOWERING,
                compile_options=self.compile_options,
            )
            if compilation_context is not None
            else None
        )
        try:
            from .scheduler import materialize_legacy_schedule

            (
                legacy_schedule,
                panel_bounds,
                relayout_plan,
                result_tile_plan,
            ) = materialize_legacy_schedule(self.normalized_cin or stmt, self.loop_plan)
            self._apply_explicit_parallel_schedule(lowered, legacy_schedule)
            from .schedule_lowerer import apply_schedule_to_llir

            scheduled = apply_schedule_to_llir(
                lowered,
                legacy_schedule,
                panel_bounds,
                relayout_plan,
                result_tile_plan,
            )
        except Exception:
            if schedule_lowering_token is not None and compilation_context is not None:
                compilation_context.fail_stage(schedule_lowering_token)
            raise
        if schedule_lowering_token is not None and compilation_context is not None:
            compilation_context.complete_stage(schedule_lowering_token)
        return scheduled

    @staticmethod
    def _contains_explicit_parallel(stmt: IndexStmt) -> bool:
        if isinstance(stmt, ForAll):
            return bool(stmt.parallel) or CINLowerer._contains_explicit_parallel(
                stmt.stmt
            )
        if isinstance(stmt, Where):
            return CINLowerer._contains_explicit_parallel(
                stmt.producer
            ) or CINLowerer._contains_explicit_parallel(stmt.consumer)
        return False

    def _should_parallelize_outer_forall(self, index_var: IndexVar) -> bool:
        if (
            not self.final_result_tensor_var
            or not self.final_result_tensor_var.is_dense()
        ):
            return False
        if not self.final_result_tensor_access:
            return False
        if self.final_result_tensor_access.has_index_var(index_var):
            return True
        if index_var.has_parent and self.final_result_tensor_access.has_index_var(
            index_var.parent
        ):
            return True
        for result_index_var in self.final_result_tensor_access.get_index_vars():
            if result_index_var.has_parent and result_index_var.parent == index_var:
                return True
        return False

    def _should_parallelize_compressed_where(self, index_var: IndexVar) -> bool:
        """Check if the outermost ForAll over a dense dimension can be parallelized
        with two-phase sparse output assembly. Generalizes to any format with
        dense outer + compressed inner level using a sparse workspace."""
        if not self.final_result_tensor_var or not self.final_result_tensor_access:
            return False
        level_types = self.final_result_tensor_var.get_level_types()
        if len(level_types) < 2:
            return False
        if level_types[0] != LevelType.DENSE:
            return False
        if any(lt != LevelType.COMPRESSED for lt in level_types[1:]):
            return False
        if not self.final_result_tensor_access.has_index_var(index_var):
            return False
        if self.final_result_tensor_access.level_of_index_var(index_var) != 0:
            return False
        if not self._where_producer_stmts or not self._where_workspace_name:
            return False
        # Require at least one sparse non-workspace input tensor.
        # Traverse the CIN to find all referenced TensorVars.
        from .cin import Workspace, TensorAccess, CINVisitorAccept

        class _TVCollector(CINVisitorAccept):
            tvars: set = set()

            def visit_TensorAccess(self, node: TensorAccess):
                self.tvars.add(node.get_tensor())

        collector = _TVCollector()
        if self.outermost_stmt:
            collector.visit(self.outermost_stmt)
        has_sparse_input = any(
            not tv.is_dense()
            for tv in collector.tvars
            if tv != self.final_result_tensor_var and not isinstance(tv, Workspace)
        )
        return has_sparse_input

    @staticmethod
    def _is_openmp_compatible_for_loop(for_loop: llir.ForLoop) -> bool:
        if not isinstance(for_loop.init, llir.VarInit):
            return False
        if not isinstance(for_loop.init.var, llir.Var):
            return False
        loop_var = for_loop.init.var

        if isinstance(for_loop.update, llir.Increment):
            if for_loop.update.var.name != loop_var.name:
                return False
        elif isinstance(for_loop.update, llir.Assign):
            if type(for_loop.update.var) is not llir.Var:
                return False
            if for_loop.update.var.name != loop_var.name:
                return False
            if for_loop.update.op not in (AssignOp.ADD_ASSIGN, AssignOp.SUB_ASSIGN):
                return False
        else:
            return False

        if not isinstance(for_loop.cond, llir.BinOp):
            return False
        if for_loop.cond.op not in ("<", "<=", ">", ">="):
            return False
        if not isinstance(for_loop.cond.left, llir.Var):
            return False
        return for_loop.cond.left.name == loop_var.name

    @classmethod
    def _has_sparse_inner_loop(cls, stmts: List[llir.Stmt]) -> bool:
        """Check if any ForLoop in stmts (or nested) iterates over a sparse level
        (identified by init value referencing a _pos array)."""
        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                if (
                    isinstance(stmt.init, llir.VarInit)
                    and match_mode_position_begin(stmt.init.value) is not None
                ):
                    return True
                if cls._has_sparse_inner_loop(stmt.body):
                    return True
        return False

    @staticmethod
    def _rewrite_val_refs(stmts: list, replacements: dict) -> None:
        """Rewrite _val[pos] → _ptr[loop_var] in LLIR statement trees."""
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, llir.Assign):
                rewritten_target = CINLowerer._rewrite_expr_refs(stmt.var, replacements)
                llir._validate_assignment_target(rewritten_target)
                stmt.var = cast(llir.AssignmentTarget, rewritten_target)
                stmt.value = CINLowerer._rewrite_expr_refs(stmt.value, replacements)
            elif isinstance(stmt, llir.VarInit):
                stmt.value = CINLowerer._rewrite_expr_refs(stmt.value, replacements)
            elif isinstance(stmt, llir.MemberCallStmt):
                stmts[i] = llir.MemberCallStmt(
                    base=CINLowerer._rewrite_expr_refs(stmt.base, replacements),
                    member=stmt.member,
                    template_args=stmt.template_args,
                    args=tuple(
                        CINLowerer._rewrite_expr_refs(arg, replacements)
                        for arg in stmt.args
                    ),
                )
            elif isinstance(stmt, llir.FunctionCallStmt):
                rewritten_name = stmt.name
                for old, new in replacements.items():
                    rewritten_name = rewritten_name.replace(old, new)
                stmts[i] = llir.FunctionCallStmt(
                    name=rewritten_name,
                    args=[
                        CINLowerer._rewrite_expr_refs(arg, replacements)
                        for arg in stmt.args
                    ],
                )
            elif isinstance(stmt, llir.ForLoop):
                CINLowerer._rewrite_val_refs(stmt.body, replacements)
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    CINLowerer._rewrite_val_refs(stmt.then_body, replacements)
                if stmt.else_body:
                    CINLowerer._rewrite_val_refs(stmt.else_body, replacements)
                if stmt.then_body_list:
                    for body in stmt.then_body_list:
                        CINLowerer._rewrite_val_refs(body, replacements)
            elif isinstance(stmt, llir.RawStmt):
                for old, new in replacements.items():
                    stmt.code = stmt.code.replace(old, new)

    @staticmethod
    def _rewrite_expr_refs(expr, replacements: dict):
        """Rewrite variable references in an expression."""
        if type(expr) is llir.ArrayAccess:
            access = cast(llir.ArrayAccess, expr)
            rewritten_array = CINLowerer._rewrite_expr_refs(access.array, replacements)
            if type(rewritten_array) is llir.Var:
                array = cast(llir.Var, rewritten_array)
                for old, new in replacements.items():
                    if (
                        old.endswith("[")
                        and new.endswith("[")
                        and array.name == old[:-1]
                    ):
                        array = llir.Var(
                            name=new[:-1],
                            type=array.type,
                            is_ptr=array.is_ptr,
                            is_restrict=array.is_restrict,
                            tensor_access=array.tensor_access,
                        )
                rewritten_array = array
            return llir.ArrayAccess(
                array=rewritten_array,
                index=CINLowerer._rewrite_expr_refs(access.index, replacements),
                tensor_access=access.tensor_access,
            )
        if isinstance(expr, llir.Var):
            for old, new in replacements.items():
                if expr.name == old or old in expr.name:
                    expr = llir.Var(
                        name=expr.name.replace(old, new),
                        type=expr.type,
                        is_ptr=expr.is_ptr,
                        is_restrict=expr.is_restrict,
                        tensor_access=expr.tensor_access,
                    )
            return expr
        if type(expr) in (llir.BinOp, llir.Add, llir.Mul):
            binary = cast(llir.BinOp, expr)
            return llir.rebuild_binary_expression(
                binary,
                CINLowerer._rewrite_expr_refs(binary.left, replacements),
                CINLowerer._rewrite_expr_refs(binary.right, replacements),
            )
        if type(expr) is llir.AddressOf:
            address = cast(llir.AddressOf, expr)
            rewritten_operand = CINLowerer._rewrite_expr_refs(
                address.operand,
                replacements,
            )
            return llir.AddressOf(
                operand=cast(llir.AssignmentTarget, rewritten_operand),
            )
        return expr

    @staticmethod
    def _expr_to_str(expr) -> str:
        """Quick-and-dirty LLIR expr to C++ string."""
        if isinstance(expr, llir.Var):
            return expr.name
        if isinstance(expr, llir.Literal):
            return str(expr.value)
        if isinstance(expr, llir.BinOp):
            return f"({CINLowerer._expr_to_str(expr.left)} {expr.op} {CINLowerer._expr_to_str(expr.right)})"
        return str(expr)

    @staticmethod
    def _find_val_array_access(
        expr: llir.Expr,
        pos_to_stride: Dict[str, str],
    ) -> Optional[tuple]:
        """Recursively search an expression for references to a _val array
        indexed by a position variable in *pos_to_stride*.

        LLIR may represent array accesses either as structured ArrayAccess
        nodes or as flat Var names like ``"B_val[pB1]"``.  This handles both.

        Returns ``(val_array_name, stride_name)`` or *None*.
        """
        import re

        if isinstance(expr, llir.ArrayAccess):
            if (
                isinstance(expr.array, llir.Var)
                and "_val" in expr.array.name
                and isinstance(expr.index, llir.Var)
                and expr.index.name in pos_to_stride
            ):
                return (expr.array.name, pos_to_stride[expr.index.name])
        # Flat Var with name like "B_val[pB1]"
        if isinstance(expr, llir.Var):
            m = re.match(r"^(\w+_val)\[(\w+)\]$", expr.name)
            if m:
                arr_name, pos_var = m.group(1), m.group(2)
                if pos_var in pos_to_stride:
                    return (arr_name, pos_to_stride[pos_var])
        # Recurse into BinOp children
        if isinstance(expr, llir.BinOp):
            left = CINLowerer._find_val_array_access(expr.left, pos_to_stride)
            if left:
                return left
            return CINLowerer._find_val_array_access(expr.right, pos_to_stride)
        return None

    @staticmethod
    def _sparse_pos_work_expr(
        sparse_pos: Optional[str], loop_bound: Optional[str]
    ) -> Optional[str]:
        """Return a safe total-nnz expression for a matching dense parent."""
        import re

        if sparse_pos is None or loop_bound is None:
            return None
        match = re.match(r"([A-Za-z_]\w*?)(\d+)_pos$", sparse_pos)
        if match is None:
            return None
        operand, level_text = match.groups()
        level = int(level_text)
        if level == 0 or loop_bound != f"{operand}{level - 1}_size":
            return None
        return f"{sparse_pos}[{loop_bound}]"

    def _mark_first_for_loop_parallel(self, stmts: List[llir.Stmt]) -> None:
        for llir_stmt in stmts:
            if isinstance(
                llir_stmt, llir.ForLoop
            ) and self._is_openmp_compatible_for_loop(llir_stmt):
                llir_stmt.omp_parallel_for = True
                has_sparse = self._has_sparse_inner_loop(llir_stmt.body)
                # Hoist per-thread workspace alloc/free outside the for loop
                # but inside the OMP parallel region.
                alloc = getattr(self, "_workspace_alloc_stmts", [])
                free = getattr(self, "_workspace_free_stmts", [])

                if has_sparse and alloc:
                    # Use adaptive atomic work-stealing: chunk scales with
                    # total nnz to balance scheduling overhead vs load
                    # imbalance across all matrix sizes.
                    llir_stmt.omp_parallel_for = True
                    llir_stmt.omp_schedule = "dynamic, 64"  # fallback
                    # Find the sparse pos array to compute nnz
                    sparse_pos = self._find_sparse_pos_array(llir_stmt.body)
                    loop_bound = self._extract_loop_bound(llir_stmt)
                    sparse_work = self._sparse_pos_work_expr(sparse_pos, loop_bound)
                    if (
                        sparse_work
                        and loop_bound
                        and isinstance(llir_stmt.update, llir.Increment)
                    ):
                        # Replace the omp for with atomic work-stealing
                        llir_stmt.omp_parallel_for = False
                        adaptive_pre = list(alloc) + [
                            llir.RawStmt(code=f"int _nnz = {sparse_work}"),
                            llir.RawStmt(
                                code="int _chunk = std::max(16, std::min(256, "
                                "_nnz / (omp_get_num_threads() * 128)))"
                            ),
                        ]
                        # The atomic counter is declared BEFORE the parallel region
                        # (shared across threads). We store it as a pre-parallel stmt.
                        self._atomic_counter_decl = llir.RawStmt(
                            code="std::atomic<int> _next_row{0}",
                            add_semicolon=True,
                        )
                        # Wrap the loop body in an atomic work-stealing while loop
                        # We replace the for loop entirely with raw code
                        llir_stmt.pre_parallel_body = adaptive_pre
                        llir_stmt.post_parallel_body = free or None
                        # Mark that the for loop should use atomic scheduling
                        llir_stmt._use_atomic_scheduling = True
                        llir_stmt._atomic_chunk_var = "_chunk"
                        llir_stmt._atomic_counter_var = "_next_row"
                        llir_stmt._loop_bound = loop_bound
                        # Work-aware thread cap; chunk stays the atomic _chunk above.
                        llir_stmt.omp_num_threads = (
                            f"scorch_nthreads({sparse_work}, {loop_bound})"
                        )
                    else:
                        if alloc or free:
                            llir_stmt.pre_parallel_body = alloc or None
                            llir_stmt.post_parallel_body = free or None
                        self._apply_parallel_policy(llir_stmt)
                else:
                    if has_sparse:
                        llir_stmt.omp_schedule = "dynamic, 64"
                    if alloc or free:
                        llir_stmt.pre_parallel_body = alloc or None
                        llir_stmt.post_parallel_body = free or None
                    self._apply_parallel_policy(llir_stmt)
                self._attach_serial_workspace_pools(llir_stmt)
                return

    def _attach_serial_workspace_pools(self, loop: llir.ForLoop) -> None:
        """Allocate per-worker dense workspaces before entering OpenMP."""
        specs = getattr(self, "_workspace_pool_specs", [])
        if not specs:
            return

        thread_expr = loop.omp_num_threads or "omp_get_max_threads()"
        before: List[llir.Stmt] = []
        for workspace_name, ctype, extent in specs:
            before.extend(
                [
                    llir.RawStmt(
                        code=(f"int {workspace_name}_thread_count = " f"{thread_expr}")
                    ),
                    llir.RawStmt(
                        code=(
                            f"auto {workspace_name}_pool_owner = "
                            f"scorch_make_aligned_buffer<{ctype}>("
                            "scorch_checked_size_product("
                            f"(size_t){workspace_name}_thread_count, "
                            f"(size_t){extent}))"
                        )
                    ),
                ]
            )
        loop.before_parallel_body = before

    @staticmethod
    def _tag_first_loop(stmts: List[llir.Stmt], index_var: IndexVar) -> None:
        """Attach a stable logical loop name for post-lowering schedule passes."""
        for stmt in stmts:
            if isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                stmt.scorch_index_var = index_var.name
                return

    @staticmethod
    def _find_tagged_for_loop(
        stmts: List[llir.Stmt], loop_name: str
    ) -> Optional[llir.ForLoop]:
        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                if getattr(stmt, "scorch_index_var", None) == loop_name:
                    return stmt
                nested = CINLowerer._find_tagged_for_loop(stmt.body, loop_name)
                if nested is not None:
                    return nested
            elif isinstance(stmt, llir.WhileLoop):
                nested = CINLowerer._find_tagged_for_loop(stmt.body, loop_name)
                if nested is not None:
                    return nested
            elif isinstance(stmt, llir.IfThenElse):
                bodies = []
                if stmt.then_body:
                    bodies.append(stmt.then_body)
                if stmt.else_body:
                    bodies.append(stmt.else_body)
                if stmt.then_body_list:
                    bodies.extend(stmt.then_body_list)
                for body in bodies:
                    nested = CINLowerer._find_tagged_for_loop(body, loop_name)
                    if nested is not None:
                        return nested
        return None

    def _apply_explicit_parallel_schedule(
        self, function: llir.Function, schedule: "Schedule"
    ) -> None:
        loop_name = schedule.parallel_loop
        if loop_name is None:
            parallel_tiles = [tile for tile in schedule.tiles if tile.parallel]
            if not parallel_tiles:
                return
            loop_name = f"{parallel_tiles[0].index_var}_out"
        elif any(
            tile.index_var == loop_name and tile.kind == "affine"
            for tile in schedule.tiles
        ):
            loop_name = f"{loop_name}_out"

        loop = self._find_tagged_for_loop(function.body, loop_name)
        if loop is None:
            raise ValueError(
                f"Cannot find generated loop {loop_name!r} selected for parallelism"
            )
        if not loop.omp_parallel_for and not getattr(
            loop, "_use_atomic_scheduling", False
        ):
            self._mark_first_for_loop_parallel([loop])

    @staticmethod
    def _find_sparse_pos_array(body: List[llir.Stmt]) -> Optional[str]:
        """Find the name of a sparse pos array (e.g. 'A1_pos') in loop body."""
        import re

        for stmt in body:
            if isinstance(stmt, llir.VarInit):
                matched = match_mode_position_access(stmt.value)
                if matched is not None:
                    return matched
            if isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                result = CINLowerer._find_sparse_pos_array(stmt.body)
                if result:
                    return result
            if isinstance(stmt, llir.RawStmt):
                m = re.search(r"(\w+_pos)\[", stmt.code)
                if m:
                    return m.group(1)
        return None

    @staticmethod
    def _extract_loop_bound(for_loop: llir.ForLoop) -> Optional[str]:
        """Extract the upper bound variable name from a for loop condition."""
        if isinstance(for_loop.cond, llir.BinOp) and for_loop.cond.op == "<":
            right = for_loop.cond.right
            if isinstance(right, llir.Var):
                return right.name
        return None

    @staticmethod
    def _parallel_rows_expr(for_loop: llir.ForLoop, bound: str) -> str:
        """Return the loop trip count, accounting for affine tile strides."""
        update = for_loop.update
        if isinstance(update, llir.Assign) and update.op == llir.AssignOp.ADD_ASSIGN:
            step = CINLowerer._expr_to_str(update.value)
            return f"(({bound} + {step} - 1) / {step})"
        return bound

    # Min work per thread for the SpGEMM 2-phase path, where `work` is the true flop
    # (A_nnz*avg_B_row). Emitted by NAME (not as a literal) so the codegen flop grain
    # lives in scorch/csrc/scorch_policy.h and picks up the Phase 4b
    # per-host autotuned value; the header is prepended to every
    # generated kernel so the macro resolves. Defaults to 1500 there — larger than the
    # A_nnz default (500) because flop is ~avg_B_row bigger; validated on redwood.
    _CG_FLOP_GRAIN = "SCORCH_GRAIN_CODEGEN_SPGEMM"

    def _apply_parallel_policy(
        self, loop, body=None, chunk=True, work_expr=None, grain=None
    ):
        """Attach a work-aware thread cap (+ adaptive schedule chunk) to a parallel
        ForLoop. codegen.py emits these as num_threads(scorch_nthreads(work,rows)) and
        schedule(dynamic, scorch_chunk(rows, work)) (helpers in scorch/csrc/header.h).

        rows = loop trip count; work = the C++ work estimate. When work_expr is
        given it is used verbatim (e.g. the true SpGEMM flop A_nnz*avg_B_row from
        the 2-phase path, where both operands are known); otherwise work = nnz
        (<pos>[<bound>]) for the first sparse pos array found in the body, else -1
        (thread cap by rows only). grain, when given, is emitted as the helpers'
        grain_default arg (the flop path passes _CG_FLOP_GRAIN; A_nnz sites omit
        it and get the header's 500 default). No-op when the bound can't be
        determined. chunk=False keeps the loop's own chunk (e.g. the atomic
        work-stealing _chunk) and only applies the thread cap.
        """
        bound = self._extract_loop_bound(loop)
        if not bound:
            return
        rows = self._parallel_rows_expr(loop, bound)
        if work_expr is not None:
            work = work_expr
        else:
            search_body = body if body is not None else loop.body
            pos = self._find_sparse_pos_array(search_body)
            work = self._sparse_pos_work_expr(pos, bound) or "-1"
        gsuf = f", {grain}" if grain is not None else ""
        loop.omp_num_threads = f"scorch_nthreads({work}, {rows}{gsuf})"
        if chunk:
            loop.omp_chunk_expr = f"scorch_chunk({rows}, {work}{gsuf})"

    @staticmethod
    def _parse_pos(pos_name: str):
        """Split a sparse pos array name into (operand_prefix, level) per the codegen
        naming convention: 'A1_pos' -> ('A', 1), 'B0_pos' -> ('B', 0). None if it
        doesn't parse. A pos array exists only for a COMPRESSED level, so a level-0 pos
        (e.g. 'B0_pos') signals a compressed outer level (no materialised <op>0_size).
        """
        import re

        m = re.match(r"([A-Za-z_]\w*?)(\d+)_pos$", pos_name)
        return (m.group(1), int(m.group(2))) if m else None

    def _find_all_sparse_pos_arrays(self, body) -> List[str]:
        """All distinct sparse pos array names (e.g. ['A1_pos', 'B1_pos']) referenced
        anywhere in `body`, in deterministic structural pre-order. Generalises
        _find_sparse_pos_array to nested loop headers while retaining only the
        explicit RawStmt compatibility escape hatch."""

        return collect_mode_position_arrays(
            list(body),
            _SPARSE_POSITION_DISCOVERY_CONTEXT,
        )

    def _spgemm_flop_work_expr(self, body, bound) -> Optional[str]:
        """True SpGEMM-flop work estimate for the thread cap: A_nnz * avg_B_row, the
        same estimate the prebuilt spmspm_csr kernel uses. A_nnz is the outer (A-side)
        operand's nnz; avg_B_row is the second (B-side) operand's mean row length
        (B_nnz/B0_size + 1). Returns None (caller falls back to the A_nnz-only estimate)
        unless BOTH operands are CSR-like with a DENSE outer level, because only then
        are <op>0_size (the outer dim) and <op><leaf>_pos[<op>0_size] (total nnz) real
        declared vars. In particular a compressed-outer B (format 'ss', 'oo', ...) has
        no <B>0_size var, so we must NOT reference it — that would be undeclared-id.
        """
        # Levels that have a pos array, grouped by operand prefix. A pos array exists
        # only for a compressed level, so `0 in levels[op]` == compressed outer level.
        levels: Dict[str, set] = {}
        for p in self._find_all_sparse_pos_arrays(body):
            parsed = self._parse_pos(p)
            if parsed:
                levels.setdefault(parsed[0], set()).add(parsed[1])

        # A-side prefix from the loop bound (`<A>0_size`). If A had a compressed outer
        # level the loop bound wouldn't be a plain size var, so this already implies A
        # is dense-outer (its <A>0_size is declared).
        if not bound.endswith("0_size"):
            return None
        a_prefix = bound[: -len("0_size")]
        if a_prefix not in levels or 0 in levels[a_prefix]:
            return None

        # B-side: a DIFFERENT operand that is also dense-outer (no level-0 pos), so
        # <B>0_size is a materialised shape var and <B><leaf>_pos[<B>0_size] = B_nnz.
        b_prefix = next(
            (pref for pref, lv in levels.items() if pref != a_prefix and 0 not in lv),
            None,
        )
        if b_prefix is None:
            return None

        a_nnz = f"(long){a_prefix}{max(levels[a_prefix])}_pos[{bound}]"
        b_outer = f"{b_prefix}0_size"
        b_nnz = f"{b_prefix}{max(levels[b_prefix])}_pos[{b_outer}]"
        # (long) forces 64-bit multiply so a big flop can't overflow int (the prebuilt
        # kernel likewise holds A_nnz/flop_est as long). avg_B_row guards B0_size==0
        # (empty contraction dim), matching the prebuilt's `B0_size>0?(B_nnz/B0_size)+1:1`.
        return f"{a_nnz} * ({b_outer} > 0 ? ({b_nnz} / {b_outer}) + 1 : 1)"

    @classmethod
    def _collect_output_arrays(
        cls, stmts: List[llir.Stmt], output_arrays: Dict[str, llir.Var]
    ) -> None:
        """Collect output array roots (e.g., D_values, D0_crd) from Assign stmts.

        The first-seen assignment-target array ``Var`` is retained per name so
        preallocation statements can reuse the tree's own declared metadata.
        """
        import re

        for stmt in stmts:
            if isinstance(stmt, llir.Assign) and type(stmt.var) is llir.ArrayAccess:
                array = cast(llir.ArrayAccess, stmt.var).array
                if type(array) is llir.Var:
                    array_var = cast(llir.Var, array)
                    if not re.match(r"^\w+$", array_var.name):
                        continue
                    if array_var.name not in output_arrays:
                        output_arrays[array_var.name] = array_var
            elif isinstance(stmt, llir.ForLoop):
                cls._collect_output_arrays(stmt.body, output_arrays)
            elif isinstance(stmt, llir.WhileLoop):
                cls._collect_output_arrays(stmt.body, output_arrays)
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    cls._collect_output_arrays(stmt.then_body, output_arrays)
                if stmt.else_body:
                    cls._collect_output_arrays(stmt.else_body, output_arrays)

    @classmethod
    def _replace_output_pos_with_input_pos(
        cls, stmts: List[llir.Stmt], input_iter_var: str
    ) -> None:
        """Replace shared output position variable (pD1) with input iterator position
        for thread-safe parallel output. Finds inner ForLoop over pA1..pA1_end and
        replaces pD<N> references with pA1 in the loop body."""
        import re

        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                # Find the sparse inner loop iterating pA1
                if (
                    isinstance(stmt.init, llir.VarInit)
                    and isinstance(stmt.init.var, llir.Var)
                    and stmt.init.var.name.startswith("p")
                ):
                    inner_pos_var = stmt.init.var.name  # e.g. "pA1"
                    cls._rewrite_output_pos_vars(stmt.body, inner_pos_var)
                else:
                    cls._replace_output_pos_with_input_pos(stmt.body, input_iter_var)
            elif isinstance(stmt, llir.WhileLoop):
                cls._replace_output_pos_with_input_pos(stmt.body, input_iter_var)
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    cls._replace_output_pos_with_input_pos(
                        stmt.then_body, input_iter_var
                    )
                if stmt.else_body:
                    cls._replace_output_pos_with_input_pos(
                        stmt.else_body, input_iter_var
                    )

    @classmethod
    def _rewrite_output_pos_vars(
        cls, stmts: List[llir.Stmt], input_pos_var: str
    ) -> None:
        """Replace output position variables (pD1, pD0) in Assign/Increment stmts
        with the input position variable for thread-safe writes."""
        import re

        to_remove = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, llir.Assign) and type(stmt.var) is llir.ArrayAccess:
                # Replace pD<N> in a typed output index with the input position.
                target = cast(llir.ArrayAccess, stmt.var)
                if type(target.array) is llir.Var and type(target.index) is llir.Var:
                    array = cast(llir.Var, target.array)
                    index = cast(llir.Var, target.index)
                    match = re.match(r"^p([A-Z])\d+$", index.name)
                    if match and array.name.startswith(match.group(1)):
                        stmt.var = llir.ArrayAccess(
                            array=array,
                            index=llir.Var(
                                name=input_pos_var,
                                type=index.type,
                                is_ptr=index.is_ptr,
                                is_restrict=index.is_restrict,
                                tensor_access=index.tensor_access,
                            ),
                            tensor_access=target.tensor_access,
                        )
            elif isinstance(stmt, llir.Increment) and isinstance(stmt.var, llir.Var):
                # Remove pD1++ (no longer needed, position is input-derived)
                if re.match(r"^p[A-Z]\d+$", stmt.var.name):
                    # Check it's an output pos var, not an input one
                    if stmt.var.name != input_pos_var:
                        to_remove.append(i)
            elif isinstance(stmt, llir.ForLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var)
            elif isinstance(stmt, llir.WhileLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var)
        for i in reversed(to_remove):
            stmts.pop(i)

    def _should_parallelize_coo_outer(self, index_var: IndexVar) -> bool:
        """Check if the outermost ForAll iterates a COORDINATE level and
        the output is all-COO, suitable for group-based parallelization."""
        if not self.final_result_tensor_var or not self.final_result_tensor_access:
            return False
        # Output must be all-coordinate (COO)
        for lt in self.final_result_tensor_var.get_level_types():
            if lt != LevelType.COORDINATE:
                return False
        return True

    def _transform_coo_loop_for_openmp(self, stmts: List[llir.Stmt]) -> List[llir.Stmt]:
        """Transform the outer COO WhileLoop into a group-indexed ForLoop
        with OpenMP parallelism.

        Finds the outermost WhileLoop that iterates over COO coordinate
        levels (identified by the pA0 = pA1_end update pattern) and replaces
        it with a pre-scan + parallel for over row groups.
        """
        result: List[llir.Stmt] = []
        transformed = False

        for stmt in stmts:
            if transformed or not isinstance(stmt, (llir.WhileLoop, llir.ForLoop)):
                result.append(stmt)
                continue

            body = stmt.body
            coo_update = None
            iter_var = None
            end_var = None

            # Detect COO outer loop: ForLoop with non-standard update pA0 = pA1_end
            if isinstance(stmt, llir.ForLoop):
                if (
                    isinstance(stmt.update, llir.Assign)
                    and isinstance(stmt.update.var, llir.Var)
                    and isinstance(stmt.update.value, llir.Var)
                    and "_end" in stmt.update.value.name
                ):
                    iter_var = stmt.update.var.name
                    end_var = stmt.update.value.name
                    coo_update = stmt.update  # sentinel, won't be in body
                else:
                    result.append(stmt)
                    continue
            else:
                # WhileLoop: look for pA0 = pA1_end in body
                for body_stmt in body:
                    if (
                        isinstance(body_stmt, llir.Assign)
                        and isinstance(body_stmt.var, llir.Var)
                        and body_stmt.var.name.startswith("p")
                        and isinstance(body_stmt.value, llir.Var)
                        and "_end" in body_stmt.value.name
                        and body_stmt.op == AssignOp.ASSIGN
                    ):
                        coo_update = body_stmt
                        iter_var = body_stmt.var.name
                        end_var = body_stmt.value.name

            if coo_update is None or iter_var is None or end_var is None:
                result.append(stmt)
                continue

            # Find the structured coordinate read from VarInit in body.
            # e.g., i = ArrayAccess(A0_crd, pA0)
            crd_array = None
            coord_var_name = None
            coord_initializer: Optional[llir.VarInit] = None
            for body_stmt in body:
                if type(body_stmt) is not llir.VarInit:
                    continue
                if type(body_stmt.value) is not llir.ArrayAccess:
                    continue
                coordinate_access = cast(llir.ArrayAccess, body_stmt.value)
                if (
                    type(coordinate_access.array) is not llir.Var
                    or type(coordinate_access.index) is not llir.Var
                ):
                    continue
                coordinate_array = cast(llir.Var, coordinate_access.array)
                coordinate_index = cast(llir.Var, coordinate_access.index)
                if (
                    coordinate_access.tensor_access is None
                    and coordinate_array.type is llir.DataType.PTR_INT
                    and coordinate_array.tensor_access is None
                    and type(coordinate_array.name) is str
                    and coordinate_array.name.isidentifier()
                    and coordinate_array.name.endswith("_crd")
                    and coordinate_index.type is llir.DataType.INT
                    and coordinate_index.tensor_access is None
                    and type(coordinate_index.name) is str
                    and coordinate_index.name == iter_var
                ):
                    crd_array = coordinate_array.name
                    coord_var_name = body_stmt.var.name
                    coord_initializer = body_stmt
                    break

            if crd_array is None:
                result.append(stmt)
                continue

            # Extract the end bound from the loop condition
            outer_end_var = None
            if isinstance(stmt.cond, llir.BinOp) and isinstance(
                stmt.cond.right, llir.Var
            ):
                outer_end_var = stmt.cond.right.name

            if outer_end_var is None:
                result.append(stmt)
                continue

            # Build the inner body: everything except the COO advance
            inner_body = [s for s in body if s is not coo_update]

            # Remove from inner body:
            # 1. The coordinate VarInit (we set it from _group_starts)
            # 2. The "find iterator end" WhileLoop (group boundaries
            #    already encode this)
            # 3. The VarInit for pA1_end (already set in group header)
            inner_body_filtered = []
            for s in inner_body:
                # Remove the exact coordinate initializer matched above.
                if s is coord_initializer:
                    continue
                # Remove: while (pA1_end < pA0_end && ...) { pA1_end++; }
                if isinstance(s, llir.WhileLoop):
                    # Check if this is the iterator-end-finding loop
                    if any(isinstance(bs, llir.Increment) for bs in s.body):
                        continue
                # Remove: pA1_end = pA0 + 1 (iterator end init)
                if (
                    isinstance(s, llir.VarInit)
                    and isinstance(s.var, llir.Var)
                    and s.var.name == end_var
                ):
                    continue
                # Remove: pA1_end = pA0 + 1 (as Assign)
                if (
                    isinstance(s, llir.Assign)
                    and isinstance(s.var, llir.Var)
                    and s.var.name == end_var
                ):
                    continue
                inner_body_filtered.append(s)

            # Thread-safety for output position: use input position pA1
            # as output position since nnz_out == nnz_in for SDDMM-like
            # kernels (no filtering). Replace pD1 references in the body.
            CINLowerer._replace_output_pos_with_input_pos(inner_body_filtered, iter_var)

            # Collect output array roots that need pre-allocation
            output_arrays: Dict[str, llir.Var] = {}
            import re as _re

            CINLowerer._collect_output_arrays(inner_body_filtered, output_arrays)

            # Pre-scan code
            pre_scan_stmts: List[llir.Stmt] = [
                llir.Comment("Pre-compute row group boundaries for OpenMP"),
                llir.VarDecl(
                    llir.Var(
                        name="_group_starts",
                        type=llir.DataType.STD_VECTOR_C_INT,
                    )
                ),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name="_group_starts",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        ),
                        index=llir.Literal(0),
                    ),
                    value=llir.Literal(0),
                ),
                llir.VarInit(
                    var=llir.Var(name="_n_groups", type=llir.DataType.INT64),
                    value=llir.Var(
                        name=f"{outer_end_var} > 0 ? 1 : 0",
                        type=llir.DataType.INT64,
                    ),
                ),
                # Scan loop
                llir.ForLoop(
                    init=llir.VarInit(
                        var=llir.Var(name="_p", type=llir.DataType.INT64),
                        value=llir.Literal(1),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=llir.Var(name="_p", type=llir.DataType.INT64),
                        right=llir.Var(name=outer_end_var, type=llir.DataType.INT64),
                    ),
                    update=llir.Increment(
                        var=llir.Var(name="_p", type=llir.DataType.INT64),
                    ),
                    body=[
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op="!=",
                                left=llir.Var(
                                    name=f"{crd_array}[_p]", type=llir.DataType.NO_TYPE
                                ),
                                right=llir.Var(
                                    name=f"{crd_array}[_p - 1]",
                                    type=llir.DataType.NO_TYPE,
                                ),
                            ),
                            then_body=[
                                llir.Assign(
                                    var=llir.ArrayAccess(
                                        array=llir.Var(
                                            name="_group_starts",
                                            type=llir.DataType.STD_VECTOR_C_INT,
                                        ),
                                        index=llir.Var(
                                            name="_n_groups",
                                            type=llir.DataType.INT64,
                                        ),
                                    ),
                                    value=llir.Var(name="_p", type=llir.DataType.INT64),
                                ),
                                llir.Increment(
                                    var=llir.Var(
                                        name="_n_groups", type=llir.DataType.INT64
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name="_group_starts",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        ),
                        index=llir.Var(
                            name="_n_groups",
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.Var(name=outer_end_var, type=llir.DataType.INT64),
                ),
                llir.BlankLine(),
            ]

            # Group loop body
            group_body: List[llir.Stmt] = [
                llir.VarInit(
                    var=llir.Var(name=iter_var, type=llir.DataType.INT64),
                    value=llir.Var(name="_group_starts[_g]", type=llir.DataType.INT64),
                ),
                llir.VarInit(
                    var=llir.Var(name=end_var, type=llir.DataType.INT64),
                    value=llir.Var(
                        name="_group_starts[_g + 1]", type=llir.DataType.INT64
                    ),
                ),
                llir.VarInit(
                    var=llir.Var(name=coord_var_name, type=llir.DataType.INT64),
                    value=llir.ArrayAccess(
                        array=llir.Var(
                            name=crd_array,
                            type=llir.DataType.PTR_INT,
                        ),
                        index=llir.Var(
                            name=cast(str, iter_var),
                            type=llir.DataType.INT64,
                        ),
                    ),
                ),
                *inner_body_filtered,
            ]

            # Group for loop with OpenMP
            group_loop = llir.ForLoop(
                init=llir.VarInit(
                    var=llir.Var(name="_g", type=llir.DataType.INT64),
                    value=llir.Literal(0),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=llir.Var(name="_g", type=llir.DataType.INT64),
                    right=llir.Var(name="_n_groups", type=llir.DataType.INT64),
                ),
                update=llir.Increment(
                    var=llir.Var(name="_g", type=llir.DataType.INT64),
                ),
                body=group_body,
                omp_parallel_for=True,
                omp_schedule="dynamic, 16",
            )
            self._apply_parallel_policy(group_loop, body=group_body)

            # Pre-allocate output arrays for thread-safe parallel writes.
            # Each receiver is a fresh Var carrying the first-seen declared
            # metadata of the assignment-target root it preallocates, without
            # scalar tensor-access provenance: it names the whole array
            # object, not one logical element write.
            prealloc_stmts: List[llir.Stmt] = []
            for array_var in output_arrays.values():
                prealloc_stmts.append(
                    llir.MemberCallStmt(
                        base=llir.Var(
                            name=array_var.name,
                            type=array_var.type,
                            is_ptr=array_var.is_ptr,
                            is_restrict=array_var.is_restrict,
                        ),
                        member="resize",
                        args=(
                            llir.Var(
                                name=outer_end_var,
                                type=llir.DataType.INT64,
                            ),
                        ),
                    )
                )

            # ── Optimization: flat parallel loop when each nonzero is
            # independent (scalar accumulator mode). Skips the serial
            # group-boundary scan entirely. Known-size outputs use Torch
            # storage for zero-overhead array access. ─────────────────
            if self._used_scalar_accum:
                # Detect known-nnz: sparse output + scalar accum → nnz_out == nnz_in
                if (
                    self.final_result_tensor_var
                    and not self.final_result_tensor_var.is_dense()
                ):
                    self._known_nnz_var = "_known_nnz"
                # Build a flat loop body: for each nonzero p, read
                # coordinates inline and execute the inner body.
                flat_body: List[llir.Stmt] = [
                    llir.VarInit(
                        var=llir.Var(name=coord_var_name, type=llir.DataType.INT64),
                        value=llir.ArrayAccess(
                            array=llir.Var(
                                name=crd_array,
                                type=llir.DataType.PTR_INT,
                            ),
                            index=llir.Var(
                                name=cast(str, iter_var),
                                type=llir.DataType.INT64,
                            ),
                        ),
                    ),
                ]
                # The inner body already has the inner loop and accum
                # write. We just need to set the end_var for the inner
                # loop. For a flat loop, each "group" is one nonzero's
                # row segment. We find the end by scanning forward.
                # But for scalar accum, the inner loop (over j within
                # the same row) is already inside inner_body_filtered.
                # We set pA1_end to iter_var+1 to process just this one
                # nonzero if there's no actual inner sparse loop,
                # or keep the original behavior for grouped inner loops.
                #
                # Actually, for SDDMM the inner_body_filtered already
                # contains the for(pA1=pA0; pA1<pA1_end; pA1++) loop
                # which iterates over nonzeros in this row group.
                # For a flat loop, we want pA1=iter_var, pA1_end=iter_var+1
                # so we process exactly one nonzero per flat iteration.
                #
                # We handle this by setting the group boundaries to
                # single-element ranges.
                flat_body.append(
                    llir.VarInit(
                        var=llir.Var(name=end_var, type=llir.DataType.INT64),
                        value=llir.Add(
                            left=llir.Var(
                                name=iter_var,
                                type=llir.DataType.INT64,
                            ),
                            right=llir.Literal(
                                1,
                                data_type=llir.DataType.INT64,
                            ),
                        ),
                    )
                )
                if not self._known_nnz_var:
                    # Rewrite output array accesses to use pre-sized storage directly:
                    # arr[idx] → arr.data()[idx] for pre-allocated arrays.
                    for arr_name in output_arrays:
                        CINLowerer._rewrite_val_refs(
                            inner_body_filtered,
                            {
                                f"{arr_name}[": f"{arr_name}.data()[",
                            },
                        )

                flat_body.extend(inner_body_filtered)

                flat_loop = llir.ForLoop(
                    init=llir.VarInit(
                        var=llir.Var(name=iter_var, type=llir.DataType.INT64),
                        value=llir.Literal(0),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=llir.Var(name=iter_var, type=llir.DataType.INT64),
                        right=llir.Var(name=outer_end_var, type=llir.DataType.INT64),
                    ),
                    update=llir.Increment(
                        var=llir.Var(name=iter_var, type=llir.DataType.INT64),
                    ),
                    body=flat_body,
                    omp_parallel_for=True,
                    omp_schedule="dynamic, 64",
                )
                self._apply_parallel_policy(flat_loop, body=flat_body)

                # The coordinate lattice declares this zero placeholder outside
                # the original grouped loop so its update can advance the parent.
                # Flat COO lowering replaces that update with a loop-local bound,
                # leaving only this exact generated prefix declaration redundant.
                result = [
                    prefix_stmt
                    for prefix_stmt in result
                    if not (
                        type(prefix_stmt) is llir.VarInit
                        and type(prefix_stmt.var) is llir.Var
                        and prefix_stmt.var.name == end_var
                        and prefix_stmt.var.type is llir.DataType.INT
                        and type(prefix_stmt.value) is llir.Literal
                        and type(prefix_stmt.value.value) is int
                        and prefix_stmt.value.value == 0
                    )
                ]
                if not self._known_nnz_var:
                    result.extend(prealloc_stmts)
                result.append(flat_loop)
                transformed = True
            else:
                result.extend(pre_scan_stmts)
                result.extend(prealloc_stmts)
                result.append(group_loop)
                transformed = True

        return result

    def lower_ForAll(self, stmt: ForAll) -> _LoweredOuterBody:
        """
        Lower a ForAll to LLIR
        parent_index_var is the index var of the parent ForAll, if any
        """

        # Get index variable at this forall
        index_var = stmt.get_index_var()
        is_outermost_forall = not self.seen_outermost_forall
        if is_outermost_forall:
            self.seen_outermost_forall = True

        self.defined_index_vars.append(index_var)

        iter_lattice = IterationLattice(for_all_stmt=stmt, cin_lowerer=self)

        stmts: List[llir.Stmt] = []

        # if self.result_tensor_access and not self.result_tensor_access.has_index_var(
        #     index_var
        # ):
        #     stmts.append(llir.Comment(f"{index_var} not in result tensor access"))

        # If the result level for this index_var is dense, need to assemble the result by
        # setting the corresponding values in the result values array to 0
        if (
            self.result_tensor_access
            and self.result_tensor_access.has_index_var(index_var)
            and self.result_tensor_access.level_type_of_index_var(index_var)
            == LevelType.DENSE
        ):
            # If the parent level is not dense or has no parent level,
            # and the next levels are all dense
            # then we need to initialize result value array elements to 0
            level_of_index_var = self.result_tensor_access.level_of_index_var(index_var)
            if (
                (level_of_index_var == 0)
                or (
                    self.result_tensor_access.level_types()[level_of_index_var - 1]
                    != LevelType.DENSE
                )
            ) and all(
                [
                    self.result_tensor_access.level_types()[i] == LevelType.DENSE
                    for i in range(
                        level_of_index_var + 1, self.result_tensor_access.num_levels
                    )
                ]
            ):
                assert self.result_tensor_var, "Result tensor variable not set"

        stmts.extend(
            [
                *iter_lattice.get_iterator_init_stmts(),
                llir.BlankLine(),
                *iter_lattice.get_lattice_loops(),
            ]
        )
        self._tag_first_loop(stmts, index_var)
        if stmt.parallel is True:
            self._mark_first_for_loop_parallel(stmts)
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._should_parallelize_outer_forall(index_var)
        ):
            self._mark_first_for_loop_parallel(stmts)
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._used_scalar_accum
            and self._should_parallelize_coo_outer(index_var)
        ):
            stmts = self._transform_coo_loop_for_openmp(stmts)
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._should_parallelize_compressed_where(index_var)
        ):
            result_tensor = self.final_result_tensor_var
            result_access = self.final_result_tensor_access
            workspace_name = self._where_workspace_name
            workspace_ctype = self._where_workspace_ctype
            if (
                result_tensor is None
                or result_access is None
                or workspace_name is None
                or workspace_ctype is None
            ):
                raise CompilerInvariantError(
                    "stage=CIN lowering pass=compressed Where/OpenMP: "
                    "parallelization gate did not establish result/workspace metadata"
                )
            compressed_levels = tuple(
                level
                for level, level_type in enumerate(result_tensor.get_level_types())
                if level_type == LevelType.COMPRESSED
            )
            return _LoweredOuterBody(
                statements=stmts,
                compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
                    CompressedWhereOpenMPContext(
                        result_name=result_tensor.get_name(),
                        result_id=result_access.tensor_id,
                        compressed_levels=compressed_levels,
                        result_assembler=_result_tensor_abi_assembler(result_tensor),
                        workspace_name=workspace_name,
                        workspace_ctype=workspace_ctype,
                        policy=CompressedWhereOpenMPPolicy(
                            omp_schedule="dynamic, 64",
                            flop_grain=self._CG_FLOP_GRAIN,
                        ),
                        compile_options=self.compile_options,
                    )
                ),
            )

        return _LoweredOuterBody(statements=stmts)

    @staticmethod
    def lower_TensorVar(tensor_var: TensorVar) -> llir.Expr:
        """
        Lower a TensorVar to LLIR
        """
        return llir.Var(
            name=tensor_var.get_name(),
            type=llir.DataType.TACO_TENSOR,
        )

    @staticmethod
    def add_dependent_tensors(
        stmt: IndexStmt, tensor_vars: List[TensorVar]
    ) -> List[TensorVar]:
        """
        Add dependent tensor variables to the list of tensor variables
        Also return the list of dependent tensor variables
        Dependent tensors are those that are used in the RHS of the TensorAssign
        where the tensor variable on the LHS is in TENSOR_VARS
        """
        dependent_tensor_vars: List[TensorVar] = []

        class DependentTensorCollector(CINVisitorAccept):
            def visit_TensorAssign(self, tensor_assign: TensorAssign):
                if tensor_assign.get_lhs_tensor() in tensor_vars:
                    rhs_tensor_vars = tensor_assign.get_rhs_tensor_vars()
                    # add the ones that are not already in tensor_vars
                    for rhs_tensor_var in rhs_tensor_vars:
                        if rhs_tensor_var not in tensor_vars:
                            tensor_vars.append(rhs_tensor_var)
                            dependent_tensor_vars.append(rhs_tensor_var)

        tensor_collector = DependentTensorCollector()
        tensor_collector.visit(stmt)
        return dependent_tensor_vars

    @staticmethod
    def lower_IndexVar(ivar: IndexVar) -> llir.Var:
        """
        Lower an IndexVar to LLIR
        """
        return llir.Var(
            name=ivar.name,
            type=llir.DataType.INT64,
        )
