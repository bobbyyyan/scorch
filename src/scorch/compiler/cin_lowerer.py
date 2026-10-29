from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Iterable,
    List,
    NoReturn,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from . import llir
from .cin import (
    IndexStmt,
    IndexVar,
    IndexVarAdd,
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
    _is_index_expr_instance,
    _is_index_stmt_instance,
    coordinate_result_driver,
)
from .iter_lattice import (
    IterationLattice,
    compressed_child_catch_up,
    dense_block_commit,
    select_vector_marks,
)
from .identity import SymbolId
from .llir import AssignOp, DataType
from .diagnostics import (
    CompileOptionsDiagnostic,
    CompileOptionsError,
    CompilerInvariantError,
    UnsupportedFeature,
    VerificationError,
)
from .loop_plan import LoopPlan, ScheduledCIN, verify_scheduled_cin
from .sparse_accumulator import (
    CHAINED_ACCUMULATOR_STRUCTURE,
    UNSUPPORTED_ACCUMULATOR_STRUCTURE,
)
from .legacy_cin_adapter import (
    claim_legacy_cin_working_tree,
    legacy_cin_working_copy,
)
from .cin_analysis import (
    CINDiagnostic,
    _raise_tile_splits,
    _tile_split_diagnostics,
    _verify_legacy_cin_lowering_structure,
    verify_cin_if_enabled,
)
from .compile_options import CompileOptions
from .compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPPolicy,
    OuterCellDomain,
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
from .scalar_workspace_visit import place_visit_flag, visit_flag, visit_flag_set
from .parallel_marking_pass import (
    EMPTY_PARALLEL_WORKSPACE_CLUSTER,
    POOL_COO_WORKSPACE_1D,
    ParallelWorkspaceCluster,
    ParallelWorkspacePoolSpec,
    apply_parallel_policy,
    mark_first_for_loop_parallel,
    serial_workspace_cluster_placement,
)
from .torch_cpp_abi import (
    BINARY_POST_OPS,
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
    ZERO_PRESERVING_POST_OPS,
    dense_block_check,
    drained_entries_check,
    index_extent_bindings,
    tensor_data_ptr,
    tensor_storage_member,
)
from ..format import LevelType, TensorFormat, LevelFormat
from ..utils import (
    dtype_to_c_datatype,
    get_pytorch_c_dtype_name,
)
from ..exceptions import (
    ScorchAssertionError,
    ScorchNotImplementedError,
    ScorchTypeError,
    ScorchValueError,
)

if TYPE_CHECKING:
    from .scheduler import Schedule


_SPARSE_POSITION_DISCOVERY_CONTEXT = LLIRTraversalContext(
    stage="CIN lowering",
    pass_name="collect_sparse_position_arrays",
)

# Short names for the statement-level result-storage marker's vocabulary; the
# fully qualified enum members bury the fact each marking site is stating.
RESULT_VALUES = llir.ResultStorageArray.VALUES
RESULT_POS = llir.ResultStorageArray.POS
RESULT_CRD = llir.ResultStorageArray.CRD
STORAGE_READ = llir.ResultStorageDirection.READ
STORAGE_WRITE = llir.ResultStorageDirection.WRITE

#: One ``(array, level, direction)`` triple.  ``level`` is ``None`` for the value
#: vector, which has no level.
ResultStorageTriple = Tuple[
    llir.ResultStorageArray, Optional[int], llir.ResultStorageDirection
]


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
    value_post_ops: Optional[Tuple[Tuple[str, Optional[str]], ...]] = None,
) -> ResultTensorAssembler:
    """Snapshot one semantic result at the Torch/C++ ABI ownership boundary.

    The lowering's only way to build a result assembler.  ``value_post_ops``
    is left ``None`` (not decided) by a caller that only allocates storage
    before the loop nest is lowered, when the copy-out has not yet decided
    whether it applies the post-ops; such an assembler refuses to finish the
    result (``result_post_ops_undecided``).  A caller that finishes the
    result passes ``self._value_post_ops()``.
    """

    return ResultTensorAssembler(
        name=tensor_var.get_name(),
        level_types=tuple(tensor_var.get_level_types()),
        dtype=tensor_var.dtype,
        known_nnz_var=known_nnz_var,
        exact_dense_parent_positions=exact_dense_parent_positions,
        reserve_hint_var=reserve_hint_var,
        value_post_ops=value_post_ops,
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


def _abi_index_bindings(
    accesses: Sequence[TensorAccess],
    result_tensor_var: TensorVar,
    input_tensor_vars: List[TensorVar],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int, int], ...]]:
    """The ABI's index bindings from the statement's accesses of its arguments.

    Matched to the arguments by name, which the ABI requires to be unique: the lowering
    works on a copy of the caller's tree, so a ``TensorVar`` is not the caller's object.
    An access of anything else (a workspace) binds no argument dimension.
    """

    argument_names = [result_tensor_var.get_name()] + [
        tensor_var.get_name() for tensor_var in input_tensor_vars
    ]
    per_argument: List[List[List[Tuple[object, str]]]] = [[] for _ in argument_names]
    for access in accesses:
        name = access.get_tensor().get_name()
        if name in argument_names:
            per_argument[argument_names.index(name)].append(
                [(index_var.index_id, index_var.name) for index_var in access.indices]
            )
    tensor_vars = [result_tensor_var] + list(input_tensor_vars)
    return index_extent_bindings(
        [
            (tuple(tensor_var.get_mode_order() or ()), accesses_of)
            for tensor_var, accesses_of in zip(tensor_vars, per_argument)
        ]
    )


def _torch_cpp_kernel_abi(
    result_tensor_var: TensorVar,
    input_tensor_vars: List[TensorVar],
    post_ops: Optional[PostOps],
    result_levels_bounded_in_kernel: Optional[Tuple[int, ...]] = (),
    accesses: Sequence[TensorAccess] = (),
) -> TorchCppKernelABI:
    """Snapshot one complete public kernel ABI without retaining semantic IR.

    ``result_levels_bounded_in_kernel`` is ``None`` before the body is built
    (``TorchCppKernelABI.result_levels_bounded_in_kernel``).  ``accesses`` are the
    statement's accesses, from which the ABI's index bindings come.
    """

    extra_tensor_names = tuple(post_ops.extra_tensors) if post_ops is not None else ()
    index_names, index_bindings = _abi_index_bindings(
        accesses, result_tensor_var, input_tensor_vars
    )
    return TorchCppKernelABI(
        index_names=index_names,
        index_bindings=index_bindings,
        result_shape=tuple(result_tensor_var.shape or ()),
        result_level_types=tuple(result_tensor_var.get_level_types()),
        input_tensors=tuple(_kernel_tensor_abi(tensor) for tensor in input_tensor_vars),
        extra_tensor_names=extra_tensor_names,
        extra_tensor_dtype=(result_tensor_var.dtype if extra_tensor_names else None),
        result_levels_bounded_in_kernel=result_levels_bounded_in_kernel,
    )


def dense_leaf_below_compressed(level_types: Sequence[LevelType]) -> bool:
    """Whether a result's last level is dense with a compressed level above it.

    Such a leaf is one cell of a dense block: each stored position of the
    nearest compressed level above it owns ``prod(extent)`` consecutive
    values, in the dense levels' lexicographic order, so a value is placed at
    its computed position rather than appended.  Cells the loops never visit
    (a stored stream driving a dense level skips them) are zeros of the block.
    """

    return (
        len(level_types) >= 2
        and level_types[-1] == LevelType.DENSE
        and any(level_type == LevelType.COMPRESSED for level_type in level_types[:-1])
    )


def dense_block_leaf_store(
    values_name: str,
    position: llir.Expr,
    value: llir.Expr,
    accumulate: bool,
    result_storage: Optional[llir.ResultStorageMetadata],
    read_storage: Optional[llir.ResultStorageMetadata],
) -> List[llir.Stmt]:
    """Store one dense-block cell of a sparse result's value vector.

    The vector grows with zeros up to the cell first, so skipped cells read as
    zero; the store is then an in-range set.  It is spelled as a call rather
    than an indexed ``Assign`` because the dynamic-vector pass rewrites every
    indexed store into a ``*_values`` vector as an append, which is right only
    for values written one per position in order.
    """

    grow = llir.IfThenElse(
        cond=llir.BinOp(
            op="<=",
            left=llir.FunctionCall(name=f"{values_name}.size", args=[]),
            right=position,
        ),
        then_body=[
            llir.FunctionCallStmt(
                name=f"{values_name}.resize",
                args=[llir.Add(position, llir.Literal(1)), llir.Literal(0)],
                result_storage=result_storage,
            )
        ],
        result_storage=read_storage,
    )
    target = llir.ArrayAccess(
        array=llir.Var(name=values_name, type=llir.DataType.NO_TYPE),
        index=position,
    )
    if accumulate:
        store: llir.Stmt = llir.Assign(
            var=target,
            value=value,
            op=AssignOp.ADD_ASSIGN,
            result_storage=result_storage,
        )
    else:
        store = llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(name=values_name, type=llir.DataType.NO_TYPE),
                position,
                value,
            ],
            result_storage=result_storage,
        )
    return [grow, store]


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
            raise ScorchTypeError(
                "compile_options must be a CompileOptions instance",
                code="internal_argument_type_invalid",
            )
        elif llir_pass_options is not None:
            raise ScorchTypeError(
                "llir_pass_options and compile_options cannot both be provided",
                code="internal_argument_type_invalid",
            )
        if compilation_context is not None:
            if type(compilation_context) is not CompilationContext:
                raise ScorchTypeError(
                    "compilation_context must be a CompilationContext instance",
                    code="internal_argument_type_invalid",
                )
            if compilation_context.compile_options is not compile_options:
                raise ScorchTypeError(
                    "compilation_context must retain this lowering's exact "
                    "CompileOptions snapshot",
                    code="internal_argument_type_invalid",
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
        self._outermost_lowering_started: bool = False
        # Nonzero exactly while _lower_prepared_index_stmt is on the stack:
        # distinguishes compiler-internal recursive re-entry (which may bypass
        # the raw-entry boundary) from a stale reused instance or a hostile
        # caller-supplied recurse flag (which must not).
        self._active_lowerings: int = 0
        self.has_explicit_parallel_loop = False
        self.loop_plan: Optional[LoopPlan] = None
        self.normalized_cin: Optional[IndexStmt] = None

        self.result_value_array_sparse_index_llir = None
        # The visit flag of each scalar workspace whose consumer appends it to a
        # sparse result level, by workspace name, while its Where is being lowered
        # (scalar_workspace_visit.py).
        self._scalar_workspace_visit_flags: Dict[str, llir.Var] = {}
        self._scalar_accum_mode = False
        self._used_scalar_accum = False
        # The input whose stored entries the scalar-accumulation loops walk, one
        # result entry each (``cin.coordinate_result_driver``).
        self._scalar_accum_driver: Optional[TensorVar] = None
        self.index_var_to_rhs_tensor_level_type = None
        self.index_var_to_result_tensor_level_type = None

        self._known_nnz_var: Optional[str] = None

        # The most recent Where lowering's explicit workspace hand-off for
        # the parallel-marking pass. Rebound per Where; empty otherwise.
        self._parallel_workspace_cluster: ParallelWorkspaceCluster = (
            EMPTY_PARALLEL_WORKSPACE_CLUSTER
        )
        # Every Where's cluster that owns a workspace pool and that no parallel
        # marking has placed yet.  A kernel that stays serial places what is
        # left here before its outermost loop; see
        # ``_place_unplaced_workspace_clusters``.
        self._unplaced_workspace_clusters: List[ParallelWorkspaceCluster] = []
        self._kernel_has_parallel_region = False
        # The sparse workspaces a scatter into a dense result drains that are
        # made once per worker; each is a worker's coo_workspace_view_1d, its
        # inserts unchecked, and the scatter clears it after its drain.  See
        # ``_scatter_pool_extent``.
        self._pooled_scatter_workspaces: Set[str] = set()
        # Whether some loop over a DENSE result level visits only stored
        # coordinates (set by the lattice that emits such a loop).
        self.result_dense_level_sparse_driven = False
        # The result levels whose dense blocks the body checks where they open, filled
        # by ``torch_cpp_abi.dense_block_check`` and read by the kernel's ABI.
        self.result_levels_bounded_in_kernel: Set[int] = set()
        # Dense result level -> the bound variable of the dense loop over it,
        # when a dense loop drives it (set by the lattice that emits the loop).
        self.result_dense_level_bound_names: Dict[int, str] = {}
        self._outermost_loop_name: Optional[str] = None
        # The ForAlls whose bodies are being lowered right now, outermost first.
        # ``defined_index_vars`` is not this: it only grows, so it also holds
        # the loops of sibling nests already finished.
        self._enclosing_forall_vars: List[IndexVar] = []
        # Beside it, whether each of those loops visits every coordinate of its
        # extent (its iteration lattice has a point with no sparse iterator), as
        # opposed to only the coordinates its operands store.
        self._enclosing_forall_visits_all: List[bool] = []
        # Whether a dense-workspace copy-out applied the post-ops to every cell
        # (``_copy_out_applies_post_ops``); if not, the result's assembler
        # applies them to its finished values (``_value_post_ops``).
        self._post_ops_applied_in_copy_out = False
        # Dense workspace name -> the flag its producer sets when it writes,
        # for a workspace copied out to a dense level below a compressed one.
        self._workspace_written_flags: Dict[str, str] = {}

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
        # The position arrays the kernel ABI binds from mode-index arguments, set
        # with the ABI snapshot before the body is lowered; the compressed-Where
        # pass reads their entry counts to keep its work estimate in bounds.
        self._bound_position_arrays: FrozenSet[str] = frozenset()
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
                    f"'{descriptor_type}'",
                    code="post_op_descriptor_unknown",
                )
            if op.kind not in self._SUPPORTED_POST_OP_KINDS:
                raise UnsupportedFeature(
                    f"stage=CIN lowering: unsupported post-op kind '{op.kind}'",
                    code="post_op_kind_unsupported",
                )

    @staticmethod
    def _validate_index_stmt(stmt: IndexStmt) -> None:
        """Reject statement nodes with no CIN-to-LLIR lowering rule."""
        if not isinstance(stmt, (TensorAssign, ForAll, Where)):
            node_type = type(stmt).__qualname__
            raise CompilerInvariantError(
                f"stage=CIN lowering: unknown IndexStmt node type '{node_type}'",
                code="cin_node_unknown",
            )

    @staticmethod
    def _validate_coordinate_hierarchies(stmt: IndexStmt) -> None:
        """Reject valid formats the legacy position model cannot lower.

        A pure coordinate hierarchy is the established COO stream.  Mixing a
        coordinate level with DENSE or COMPRESSED levels needs explicit
        parent/child grouping that the legacy CIN-to-LLIR iterator model does
        not represent.  Validate every declared access -- including the
        result -- once at the outer ownership boundary so the gap cannot
        become an undeclared cursor bound or malformed result storage later.
        """

        for tensor_access in stmt.tensor_accesses:
            level_types = tuple(tensor_access.level_types())
            if not any(
                level_type is LevelType.COORDINATE for level_type in level_types
            ) or all(level_type is LevelType.COORDINATE for level_type in level_types):
                continue
            tensor = tensor_access.get_tensor()
            diagnostic = CINDiagnostic(
                code="unsupported_mixed_coordinate_hierarchy",
                message=(
                    "legacy CIN lowering supports coordinate-free hierarchies "
                    "or all-coordinate COO, not a mixed hierarchy for tensor "
                    f"{tensor.get_name()!r}"
                ),
                path=("root",),
                entity_id=tensor.symbol_id,
                stage="legacy_cin_lowering",
                pass_name="prepare_stmt",
            )
            raise UnsupportedFeature(
                "stage=legacy CIN lowering: "
                "unsupported_mixed_coordinate_hierarchy for tensor "
                f"{tensor.get_name()!r}",
                diagnostics=(diagnostic,),
            )

    @staticmethod
    def _validate_loop_nest(stmt: IndexStmt) -> None:
        """Reject a program with no loop, which this lowering has no rule for.

        A rank-0 tensor (``TensorVar(fmt="")``) holds one value, at position 0
        of its value array: the root of its level tree, which has no levels.
        ``lower_TensorAccess`` reads it there and ``lower_TensorAssign`` writes
        it there.  A program whose tensors are all rank 0 can be one assignment
        with no loop around it (``y[] = s[]``), and this lowering emits a
        function body that is a loop nest or a ``Where`` (``_lower_outer_body``).
        Left to run, such a program failed an assertion on its missing result
        access.  Refused here, at the same ownership boundary as the mixed
        coordinate hierarchies, with the code the rank-0 refusal has had since
        this lowering refused every rank-0 tensor.
        """

        if not isinstance(stmt, TensorAssign):
            return
        tensor = stmt.lhs.get_tensor()
        diagnostic = CINDiagnostic(
            code="unsupported_scalar_tensor",
            message=(
                "legacy CIN lowering emits a loop nest and has no rule for the "
                f"program without a loop that assigns the rank-0 tensor "
                f"{tensor.get_name()!r}"
            ),
            path=("root",),
            entity_id=tensor.symbol_id,
            stage="legacy_cin_lowering",
            pass_name="prepare_stmt",
        )
        raise UnsupportedFeature(
            "stage=legacy CIN lowering: "
            f"unsupported_scalar_tensor for tensor {tensor.get_name()!r}",
            diagnostics=(diagnostic,),
        )

    @staticmethod
    def _validate_update_operators(stmt: IndexStmt) -> None:
        """Reject an assignment whose update operator this lowering cannot apply.

        This lowering has a rule for ``=`` and ``+=`` only.  It reads
        ``TensorAssign.op`` in two places, both as ``op == Operation.ADD`` (the
        store into a dense block below a compressed level, and the ordinary
        store), and the iteration lattice rebuilds each assignment it simplifies
        with ``+=`` where the statement reduces and ``=`` where it does not
        (``get_simplified_cin``).  So ``-=``, ``*=`` and ``/=`` were lowered as
        ``+=`` or ``=`` with no message: ``C[i, j] *= A[i, j] * B[i, j]`` emitted
        the kernel of ``C = A * B``, and ``C[i] -= A[i, j]`` summed with the
        wrong sign.  No library entry builds those operators (einsum, matmul,
        ``+`` and the conversions write ``=`` or ``+=``); a raw program that does
        is refused here, at the same ownership boundary as the checks above,
        with the code the new compiler gives it (its ``lower_cin.py``).
        Every assignment of the statement is checked, a workspace's producer and
        consumer included, and the first one found in program order is named.
        """

        pending: List[Tuple[IndexStmt, Tuple[str, ...]]] = [(stmt, ("root",))]
        while pending:
            node, path = pending.pop()
            if isinstance(node, ForAll):
                pending.append((node.stmt, path + ("stmt",)))
            elif isinstance(node, Where):
                # The consumer goes on the stack first so the producer, which
                # runs first, is reported first.
                pending.append((node.consumer, path + ("consumer",)))
                pending.append((node.producer, path + ("producer",)))
            elif isinstance(node, TensorAssign):
                update = node.op
                if update is None or update is Operation.ADD:
                    continue
                spelling = getattr(update, "value", update)
                tensor = node.lhs.get_tensor()
                diagnostic = CINDiagnostic(
                    code="unsupported_update_op",
                    message=(
                        "legacy CIN lowering applies the update operators '=' "
                        f"and '+=' only, not '{spelling}=' (the assignment to "
                        f"{tensor.get_name()!r})"
                    ),
                    path=path + ("op",),
                    entity_id=tensor.symbol_id,
                    stage="legacy_cin_lowering",
                    pass_name="prepare_stmt",
                )
                raise UnsupportedFeature(
                    "stage=legacy CIN lowering: unsupported_update_op "
                    f"'{spelling}=' for tensor {tensor.get_name()!r}",
                    diagnostics=(diagnostic,),
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
                "lowering; workspace reads require a workspace-specific consumer",
                code="workspace_access_unlowered",
            )

        value_array = llir.Var(
            name=f"{tensor_access.tensor.name}_val",
            type=llir.DataType.ptr_type(tensor_access.tensor.dtype),
        )
        if not tensor_access.indices:
            # A rank-0 tensor holds one value, at position 0: the root of its
            # (empty) level tree.
            return llir.ArrayAccess(
                array=value_array,
                index=llir.Literal(value=0, data_type=llir.DataType.INT64),
                tensor_access=tensor_access_metadata,
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
            array=value_array,
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
        if self._active_lowerings == 0:
            # A bare expression root has no statement boundary, so the
            # structural preflight (cycles, depth, fields) never ran; the
            # recursive expression walk below would leak RecursionError on a
            # hostile graph.  Expression lowering is an internal traversal
            # mode used only while a validated statement lowering is active.
            diagnostic = CINDiagnostic(
                code="invalid_expression_entry",
                message=(
                    "expression lowering is an internal traversal mode; "
                    "lower a statement through the validated entry"
                ),
                path=("root",),
                stage="legacy_cin_lowering",
                pass_name="lower_IndexExpr",
            )
            raise VerificationError(
                "stage=legacy CIN lowering: expression roots require an "
                "active statement lowering",
                diagnostics=(diagnostic,),
            )
        expression_type = type(index_expr)
        if expression_type is BinaryOp:
            return self.lower_BinaryOp(cast(BinaryOp, index_expr))
        if expression_type is TensorAccess or expression_type is WorkspaceAccess:
            return self.lower_TensorAccess(cast(TensorAccess, index_expr))
        expression_name = type.__getattribute__(expression_type, "__qualname__")
        raise CompilerInvariantError(
            "stage=CIN lowering: unknown IndexExpr node type " f"'{expression_name}'",
            code="cin_node_unknown",
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

    @staticmethod
    def _result_storage(
        tensor_access: TensorAccess,
        *references: ResultStorageTriple,
    ) -> Optional[llir.ResultStorageMetadata]:
        """One statement's result-storage marker, over the references given.

        ``None`` for a workspace target, mirroring ``_tensor_access_metadata``
        and for the same reason: a workspace's own arrays are not the result's,
        so a marker naming them would be a marker about the wrong tensor.

        A marker describes the references in the statement's OWN expression
        fields -- its target, index, value, condition and call arguments -- and
        not those in a nested statement body, which carries its own.  Without
        that rule the position-boundary conditional and the append inside it
        would both describe the append's coordinate write and neither could be
        checked against the statement it sits on.
        """

        if tensor_access.is_workspace():
            return None
        return llir.ResultStorageMetadata(
            tensor_id=SymbolId(tensor_access.tensor_id.value),
            references=tuple(
                llir.ResultStorageReference(array, level, direction)
                for array, level, direction in references
            ),
        )

    def lower_CIN(self, cin: CIN) -> Union[llir.Stmt, List[llir.Stmt], llir.Expr]:
        if _is_index_stmt_instance(cin):
            return self.lower_IndexStmt(cast(IndexStmt, cin))
        cin_type = type(cin)
        if _is_index_expr_instance(cin):
            return self.lower_IndexExpr(cast(IndexExpr, cin))
        node_type = type.__getattribute__(cin_type, "__qualname__")
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
        if not self.result_tensor_access:
            raise ScorchAssertionError(
                "result tensor access is None", code="cin_lowering_state_missing"
            )
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
        # An access with no index is a scalar workspace or a rank-0 result.  A
        # scalar workspace is a local: just do <tensor> += <rhs_llir>.  A rank-0
        # result is written below like any other result, at position 0.
        if is_workspace and not index_vars:
            wksp_name = self.result_tensor_var.get_name()
            assign_stmt = llir.Assign(
                var=llir.Var(name=f"{wksp_name}", type=llir.DataType.NO_TYPE),
                value=rhs_llir,
                op=AssignOp.ADD_ASSIGN,
            )
            llir_stmts.append(assign_stmt)
            # The producer reached the accumulation: its consumer keeps this entry
            # whatever the sum is (scalar_workspace_visit.py).
            flag = self._scalar_workspace_visit_flags.get(wksp_name)
            if flag is not None:
                llir_stmts.append(visit_flag_set(flag))
        else:
            # if index_vars are all in defined_index_vars, then we can emit the compute code
            if all(index_var in self.defined_index_vars for index_var in index_vars):
                if not self.result_tensor_var:
                    raise ScorchAssertionError(
                        "result tensor var is None", code="cin_lowering_state_missing"
                    )

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
                    # The value's position is the result's position at its last
                    # level.  A rank-0 result has no level: its one value is at
                    # position 0, the root of its (empty) level tree, and every
                    # loop around this statement sums into it.
                    value_position: llir.Expr = llir.Literal(
                        value=0, data_type=llir.DataType.INT64
                    )
                    if sorted_index_vars:
                        level = self.result_tensor_access.level_of_index_var(
                            sorted_index_vars[-1]
                        )
                        value_position = llir.Var(
                            name=f"p{self.result_tensor_var.name}{level}",
                            type=llir.DataType.INT64,
                        )
                    tensor_access_llir = llir.ArrayAccess(
                        array=llir.Var(
                            name=values_llir_name,
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=value_position,
                        tensor_access=result_access_metadata,
                    )
                    # tensor_access_llir = llir.Var(
                    #     name=f"{self.result_tensor_var.get_name()}_values"
                    #     + f"[{self.defined_index_vars[-1].name}]",
                    #     type=llir.DataType.NO_TYPE,
                    # )

                if is_workspace:
                    if not isinstance(self.result_tensor_access, WorkspaceAccess):
                        raise ScorchAssertionError(
                            "isinstance(self.result_tensor_access, WorkspaceAccess)",
                            code="cin_lowering_state_missing",
                        )
                    wksp_access: WorkspaceAccess = self.result_tensor_access
                    wksp_index_vars = wksp_access.get_index_vars()
                    sorted_wksp_index_vars = [
                        wksp_index_vars[i] for i in wksp_access.tensor.mode_order
                    ]

                    if wksp_access.is_dense():
                        # <workspace name>[<C++ array of indices>] += <rhs_llir>;
                        if len(sorted_wksp_index_vars) != 1:
                            raise ScorchAssertionError(
                                "dense workspace has more than 1 index var",
                                code="cin_lowering_state_missing",
                            )
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
                        written_flag = self._workspace_written_flags.get(
                            self.result_tensor_var.name
                        )
                        if written_flag is not None:
                            llir_stmts.append(
                                llir.Assign(
                                    var=llir.Var(
                                        name=written_flag, type=llir.DataType.BOOL
                                    ),
                                    value=llir.Literal(True, llir.DataType.BOOL),
                                )
                            )

                    else:
                        # <workspace name>.insert(<C++ array of indices>, <rhs_llir>);
                        # A pooled scatter workspace is a worker's view, whose
                        # capacity is the extent native_abi.h validated every
                        # coordinate against: insert_unchecked, as the views of
                        # compressed_where_openmp_pass.
                        wksp_tensor_name = (
                            self.result_tensor_access.get_tensor().get_name()
                        )
                        insert_member = (
                            "insert_unchecked"
                            if wksp_tensor_name in self._pooled_scatter_workspaces
                            else "insert"
                        )
                        llir_stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{wksp_tensor_name}.{insert_member}",
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
                elif not self.result_tensor_access.is_workspace() and (
                    dense_leaf_below_compressed(
                        self.result_tensor_var.get_level_types()
                    )
                ):
                    llir_stmts.extend(
                        dense_block_leaf_store(
                            values_llir_name,
                            tensor_access_llir.index,
                            rhs_llir,
                            stmt.op == Operation.ADD,
                            self._result_storage(
                                self.result_tensor_access,
                                (RESULT_VALUES, None, STORAGE_WRITE),
                            ),
                            self._result_storage(
                                self.result_tensor_access,
                                (RESULT_VALUES, None, STORAGE_READ),
                            ),
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
            # the coordinates (a rank-0 result has no level, so it has none to set)
            if not self.result_tensor_access.is_workspace() and index_vars:
                last_ivar = self.defined_index_vars[-1]
                last_level_type = self.result_tensor_access.level_types()[-1]
                if last_level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                    reductions = self._reductions_around_consumer(
                        self.result_tensor_access
                    )
                    if reductions:
                        # An enclosing loop sums over a variable the result
                        # does not have, so appending here would store one
                        # partial sum per iteration of it: under the innermost
                        # loop one entry per summand, under an outer one the same
                        # coordinate once per iteration, out of order.  Only the
                        # scheduler's accumulation workspace makes such a
                        # program lowerable.
                        names = ", ".join(loop.name for loop in reductions)
                        diagnostic = CINDiagnostic(
                            code="unscheduled_sparse_result_reduction",
                            message=(
                                f"the loops over {names} sum over variables the "
                                "sparse result does not have; lower the "
                                "scheduled program (scorch.einsum or "
                                "Scheduler.auto_schedule), whose workspace "
                                "accumulates them"
                            ),
                            path=("root",),
                            stage="legacy_cin_lowering",
                            pass_name="lower_TensorAssign",
                        )
                        raise UnsupportedFeature(
                            "stage=legacy CIN lowering: "
                            "unscheduled_sparse_result_reduction",
                            diagnostics=(diagnostic,),
                        )
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
                            result_storage=self._result_storage(
                                self.result_tensor_access,
                                (RESULT_CRD, level, STORAGE_WRITE),
                            ),
                        )
                    )

                    # if the last _level is COORDINATE, we might need to set the coordinates
                    # or previous levels as well
                    if last_level_type == LevelType.COORDINATE:
                        # ``defined_index_vars`` only grows, and a merge lowers
                        # the loops under it once per branch, so a variable is
                        # listed once per branch already lowered.  Each store
                        # becomes an append, so a variable is stored once: the
                        # walk meets its innermost definition first.
                        stored_ivars = {last_ivar}
                        for defined_ivar in self.defined_index_vars[-2::-1]:
                            if defined_ivar in stored_ivars:
                                continue
                            stored_ivars.add(defined_ivar)
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
                                        result_storage=self._result_storage(
                                            self.result_tensor_access,
                                            (RESULT_CRD, level, STORAGE_WRITE),
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

    def _dense_level_extent_accesses(
        self, stmt: Where, index_var: IndexVar
    ) -> List[TensorAccess]:
        """The tensors a dense workspace over ``index_var`` may size itself from.

        Used only when no all-dense tensor has the variable.  The consumer's
        result comes first, then any other tensor storing the variable at a
        dense level; either declares that level's extent.

        Two refusals, both structured.  None of those tensors exists.  Or the
        producer loops over some operand's COMPRESSED level outside the loop
        over a level above it: a compressed level's segment is read at its
        parent's position, so such a nest would read a position nothing has
        declared.  A dense level looped outside the level above it is lowered:
        its position is the parent's position times its extent plus its
        coordinate, and the dense positions are declared parents first once
        the loops have bound their coordinates (``_parents_first`` in
        ``iter_lattice.py``).
        """

        consumer_results = [
            access
            for access in stmt.consumer.get_result_tensor_accesses()
            if not access.is_workspace()
        ]
        candidates = [
            access
            for access in [*consumer_results, *index_var.tensor_accesses]
            if not access.is_workspace()
            and index_var in access.indices
            and access.level_type_of_index_var(index_var) == LevelType.DENSE
        ]
        if not candidates:
            raise UnsupportedFeature(
                "stage=CIN lowering: "
                "dense_workspace_without_extent for index variable "
                f"{index_var.name!r}",
                diagnostics=(
                    CINDiagnostic(
                        code="dense_workspace_without_extent",
                        message=(
                            f"a dense workspace over {index_var.name!r} has no "
                            "tensor storing that index variable at a dense "
                            "level to take its extent from"
                        ),
                        path=("root",),
                        stage="legacy_cin_lowering",
                        pass_name="lower_Where",
                    ),
                ),
            )
        misplaced = self._compressed_level_outside_its_parent_loop(stmt)
        if misplaced is not None:
            code, tensor_name = misplaced
            raise UnsupportedFeature(
                f"stage=CIN lowering: {code} for tensor {tensor_name!r}",
                diagnostics=(
                    CINDiagnostic(
                        code=code,
                        message=(
                            f"the producer of the dense workspace over "
                            f"{index_var.name!r} reads {tensor_name!r} at a "
                            "level this lowering cannot place: "
                            + (
                                "a compressed level looped outside the loop "
                                "over a level above it"
                                if code == "compressed_level_outside_parent_loop"
                                else "a level no loop binds by its own name"
                            )
                        ),
                        path=("root",),
                        stage="legacy_cin_lowering",
                        pass_name="lower_Where",
                    ),
                ),
            )
        return candidates

    def _compressed_level_outside_its_parent_loop(
        self, stmt: Where
    ) -> Optional[Tuple[str, str]]:
        """``(code, tensor)`` for an operand level this lowering cannot place.

        Read off the lowering's own position: the loops being lowered around
        the Where, outermost first, then the producer's own.  A compressed
        level reads its segment at the position of the level above it, so its
        loop must lie inside the loops of every level above it; ``None`` when
        every operand's compressed levels do.  A dense level may be looped
        anywhere: its position is computed once its own coordinate and the
        levels above it are bound.  An operand level whose variable no loop
        here binds by its own name is reported too, since this lowering cannot
        place its position.

        The enclosing loops come from ``_enclosing_forall_vars`` and not from
        the statement's ``parent`` links, which the scheduled route that
        ``scorch.einsum`` takes does not set.
        """

        order = list(self._enclosing_forall_vars)
        body: IndexStmt = stmt.producer
        while isinstance(body, ForAll):
            order.append(body.get_index_var())
            body = body.stmt
        logical = [
            index_var.parent if index_var.has_parent else index_var
            for index_var in order
        ]
        for access in stmt.producer.get_rhs_tensor_accesses():
            if access.is_workspace():
                continue
            level_vars = access.get_sorted_index_vars()
            if any(level_var not in logical for level_var in level_vars):
                return ("operand_level_unbound_in_producer", access.tensor.get_name())
            depths = [logical.index(level_var) for level_var in level_vars]
            for level, level_type in enumerate(access.level_types()):
                if level_type != LevelType.DENSE and any(
                    depth > depths[level] for depth in depths[:level]
                ):
                    return (
                        "compressed_level_outside_parent_loop",
                        access.tensor.get_name(),
                    )
        return None

    def _dense_block_workspace_flag(
        self, stmt: Where, workspace_name: str, workspace_index_var: IndexVar
    ) -> Optional[str]:
        """Name the flag a dense workspace's producer sets, if its copy-out needs one.

        A dense workspace copied out to a dense result level that has a
        compressed level above it writes into a block whose coordinate is
        stored only when the operands store something under it.  The
        workspace's values cannot say that -- a stored structure may sum to
        zero -- so the producer records that it ran.  Structure, never value.
        """

        consumer_accesses = stmt.consumer.get_result_tensor_accesses()
        if not consumer_accesses:
            return None
        result_access = consumer_accesses[0]
        if result_access.is_workspace() or not result_access.has_index_var(
            workspace_index_var
        ):
            return None
        level = result_access.level_of_index_var(workspace_index_var)
        level_types = result_access.level_types()
        if level != len(level_types) - 1 or not dense_leaf_below_compressed(
            level_types
        ):
            return None
        flag = f"{workspace_name}_written"
        self._workspace_written_flags[workspace_name] = flag
        return flag

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
        workspace_alloc_stmts: List[llir.Stmt] = []
        workspace_pool_specs: List[ParallelWorkspacePoolSpec] = []
        workspace_free_stmts: List[llir.Stmt] = []
        workspace_memset_stmts: List[llir.Stmt] = []
        for wksp in workspaces:
            if not isinstance(wksp, Workspace):
                raise ScorchAssertionError(
                    "workspace is not a Workspace", code="cin_lowering_state_missing"
                )
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
                if self._scalar_workspace_appends(stmt.consumer, wksp):
                    flag = visit_flag(wksp.get_name())
                    workspace_init_stmts.append(
                        llir.VarInit(var=flag, value=llir.Literal(False))
                    )
                    self._scalar_workspace_visit_flags[wksp.get_name()] = flag
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
                        # The extent comes from an all-dense tensor that has
                        # this index variable, as always.  A sparse result
                        # with a dense last level has none to borrow from when
                        # every operand is sparse too, so it then comes from a
                        # tensor that stores the variable at a DENSE level --
                        # the result itself first.  A lookup that found nothing
                        # raised a bare IndexError here before.
                        extent_accesses = [
                            ta
                            for ta in idx_var.tensor_accesses
                            if ta.is_dense()
                            and idx_var in ta.indices
                            and not ta.is_workspace()
                        ]
                        if not extent_accesses:
                            extent_accesses = self._dense_level_extent_accesses(
                                stmt, idx_var
                            )
                        dense_ta = extent_accesses[0]
                        level = dense_ta.level_of_index_var(idx_var)
                        actual_size = f"{dense_ta.tensor.name}{level}_size"

                        wname = wksp.get_name()
                        workspace_pool_specs.append(
                            ParallelWorkspacePoolSpec(
                                name=wname,
                                scalar_type=wksp_ctype,
                                extent=actual_size,
                            )
                        )
                        if size_var != actual_size:
                            # The extent alias.  Skipped when the workspace's
                            # size variable IS the extent's own name: the
                            # alias would then read ``int64_t n = n;``, a
                            # declaration initialized from itself.
                            workspace_alloc_stmts.append(
                                llir.VarInit(
                                    var=size_llir,
                                    value=llir.Var(
                                        name=actual_size,
                                        type=llir.DataType.INT64,
                                    ),
                                )
                            )
                        workspace_alloc_stmts.extend(
                            [
                                # Typed: the restrict-qualified per-worker
                                # slice borrowed from the RAII pool owner
                                # declared by the typed before-parallel
                                # construction.
                                llir.VarInit(
                                    var=llir.Var(
                                        name=wname,
                                        type=wksp_ctype_ptr,
                                        is_restrict=True,
                                    ),
                                    value=llir.Add(
                                        llir.MemberCall(
                                            base=llir.Var(
                                                name=f"{wname}_pool_owner",
                                                type=llir.DataType.NO_TYPE,
                                            ),
                                            member="get",
                                        ),
                                        llir.Mul(
                                            llir.Cast(
                                                expr=llir.FunctionCall(
                                                    "omp_get_thread_num"
                                                ),
                                                data_type=llir.DataType.SIZE_T,
                                            ),
                                            llir.Cast(
                                                expr=llir.Var(
                                                    name=actual_size,
                                                    type=llir.DataType.INT64,
                                                ),
                                                data_type=llir.DataType.SIZE_T,
                                            ),
                                        ),
                                    ),
                                ),
                            ]
                        )
                        written_flag = self._dense_block_workspace_flag(
                            stmt, wname, idx_var
                        )
                        if written_flag is not None:
                            workspace_memset_stmts.append(
                                llir.VarInit(
                                    var=llir.Var(
                                        name=written_flag, type=llir.DataType.BOOL
                                    ),
                                    value=llir.Literal(False, llir.DataType.BOOL),
                                )
                            )
                        workspace_memset_stmts.append(
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
                    pool_extent = self._scatter_pool_extent(stmt, wksp)
                    if pool_extent is not None:
                        # Drained by a scatter into a dense result: one
                        # workspace owner per worker, made before the parallel
                        # region (or once before a serial kernel's outermost
                        # loop), and in the region each worker's view, which
                        # allocates the owner's arrays at the worker's first
                        # make_region_view() and is cleared after each drain,
                        # instead of one workspace made and freed in every
                        # iteration.  The serial placement takes make_view().
                        self._pooled_scatter_workspaces.add(wname)
                        workspace_pool_specs.append(
                            ParallelWorkspacePoolSpec(
                                name=wname,
                                scalar_type=wksp_ctype,
                                extent=pool_extent,
                                container=POOL_COO_WORKSPACE_1D,
                            )
                        )
                        workspace_alloc_stmts.append(
                            llir.VarInit(
                                var=llir.Var(name=wname, type=llir.DataType.AUTO),
                                value=llir.MemberCall(
                                    base=llir.ArrayAccess(
                                        array=llir.Var(
                                            name=f"{wname}_pool_owner",
                                            type=llir.DataType.NO_TYPE,
                                        ),
                                        index=llir.Cast(
                                            expr=llir.FunctionCall(
                                                "omp_get_thread_num"
                                            ),
                                            data_type=llir.DataType.SIZE_T,
                                        ),
                                    ),
                                    member="make_region_view",
                                ),
                            )
                        )
                        continue
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
                            self._sparse_workspace_extents(stmt, wksp),
                        ],
                    ),
                )
            )

        # Freeze this Where's workspace hand-off before recursive producer/
        # consumer lowering so an explicit parallel loop inside either branch
        # observes the completed cluster, exactly as the appended live lists
        # were observed before.
        self._parallel_workspace_cluster = ParallelWorkspaceCluster(
            alloc=tuple(workspace_alloc_stmts),
            free=tuple(workspace_free_stmts),
            pool_specs=tuple(workspace_pool_specs),
        )
        if workspace_pool_specs:
            self._unplaced_workspace_clusters.append(self._parallel_workspace_cluster)

        producer_stmts = self.lower_ProducerIndexStmt(stmt.producer)
        flags = [
            self._scalar_workspace_visit_flags[wksp.get_name()]
            for wksp in workspaces
            if wksp.get_name() in self._scalar_workspace_visit_flags
        ]
        for flag in flags:
            producer_stmts = place_visit_flag(producer_stmts, flag)
        consumer_stmts = self.lower_ConsumerIndexStmt(stmt.consumer)
        for wksp in workspaces:
            self._scalar_workspace_visit_flags.pop(wksp.get_name(), None)
        self._where_producer_stmts = producer_stmts
        self._where_consumer_stmts = consumer_stmts
        return [
            *workspace_init_stmts,
            *workspace_memset_stmts,
            *producer_stmts,
            *consumer_stmts,
            *workspace_cleanup_stmts,
        ]

    @staticmethod
    def _sparse_workspace_extents(stmt: Where, wksp: Workspace) -> llir.Expr:
        """The extents ``coo_workspace`` flattens its keys by, one per key component.

        ``coo_workspace<T, N>`` requires exactly ``N`` extents, in key order.  Both
        sides of that pairing are storage orders.  The key is inserted in the
        workspace's own storage order (its access permuted by its mode order), and
        the kernel's ``result_shape`` holds the result's extents by storage level.
        So a key component's extent is the entry at its variable's level of the
        result.  Reading either side in access order instead is right only while
        the workspace and the result carry the same mode order: a mode-order
        conversion permutes both alike, and ``einsum('abc->cb')`` does not -- it
        stores its result as ``(b, c)`` under a workspace keyed ``(b, c)`` in
        identity order.

        A workspace keyed on the whole result in storage order is given
        ``result_shape`` itself.  One keyed on fewer or reordered result variables
        -- a drain under an enclosing result loop, such as ``einsum('ijkl->ikl')``
        keyed on ``(k, l)`` -- is given those variables' own entries of
        ``result_shape``; the whole shape would fail the workspace's rank check on
        the first insert.
        """

        result_shape = llir.Var(name="result_shape", type=llir.DataType.STD_VECTOR_INT)
        results = [
            access
            for access in stmt.consumer.get_result_tensor_accesses()
            if not isinstance(access.tensor, Workspace)
        ]
        keyed = [
            access
            for access in wksp.workspace_accesses
            if len(access.get_index_vars()) == wksp.dim
        ]
        if len(results) != 1 or not keyed:
            return result_shape
        levels = list(results[0].get_sorted_index_vars())
        key = list(keyed[0].get_sorted_index_vars())
        if any(index_var not in levels for index_var in key):
            return result_shape
        positions = [levels.index(index_var) for index_var in key]
        if positions == list(range(len(levels))):
            return result_shape
        return llir.Array(
            values=[
                llir.ArrayAccess(
                    array=result_shape,
                    index=llir.Literal(position, llir.DataType.INT64),
                )
                for position in positions
            ],
            data_type=llir.DataType.INT64,
        )

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
        # A one-dimensional sparse workspace is the ``coo_workspace_1d``
        # class, whose iterator dereferences to ``std::pair<int64_t, T>``:
        # its key is the scalar coordinate itself, not an indexable
        # container.  Subscripting it emits C++ that does not compile
        # ("subscripted value is not an array, pointer, or vector"), which
        # is what made every sparse rank-1-output reduction -- for example
        # public ``einsum('ij->i')`` -- fail at build time.  Every other
        # workspace is ``coo_workspace<T, N>``, whose key is a
        # ``std::vector<int64_t>`` and is subscripted per level.
        scalar_workspace_key = wksp.dim == 1 and not wksp.is_dense()

        def workspace_key_component(level: int) -> llir.Expr:
            first = llir.MemberAccess(
                base=llir.Var(name=loop_var.name, type=loop_var.type),
                member="first",
            )
            if scalar_workspace_key:
                return first
            return llir.ArrayAccess(
                array=first,
                index=llir.Literal(level, llir.DataType.INT64),
            )

        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        loop_array = llir.Var(name=wksp_name, type=llir.DataType.AUTO)
        loop_body: List[llir.Stmt] = []

        if result_is_coord:
            # Direct assignment: rank 1 uses ``it.first``; higher-rank
            # workspaces use ``it.first[level]``.
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
                        value=workspace_key_component(i),
                        result_storage=self._result_storage(
                            result_tensor_access,
                            (RESULT_CRD, i, STORAGE_WRITE),
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
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_WRITE),
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
            # Fill intermediate vectors from the scalar/vector key component.
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
                        value=workspace_key_component(i),
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
            #
            # DELIBERATELY UNMARKED.  ``T`` is the intermediate COO tensor this
            # method builds, not the declared result, so its coordinate array is
            # not the result's storage and a marker naming it would be a marker
            # about the wrong tensor.  Review section 63.6's census bucketed this
            # site as "result" because ``intermediate_tensor_var`` is on its
            # marker list; for counting a search space that is defensible, for
            # marking it would be false.
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
            # The conversion below reads T back with `int` positions
            # (`int pT0_end = T0_crd_tensor.size(0)`): hold the drained count to the
            # 32-bit position bound first (csrc/header.h, scorch_check_drained_entries).
            drained_entries_check(intermediate_tensor_iterator),
            llir.BlankLine(),
            *assembly_stmts,
            llir.BlankLine(),
            *result_conversion_stmts,
        ]

    def _dense_block_workspace_copy_out(
        self,
        result_tensor_access: TensorAccess,
        wksp: Workspace,
        wksp_index_var: IndexVar,
        level: int,
        parent_index_var: Optional[IndexVar],
        parent_level_type: Optional[LevelType],
        written_flag: str,
    ) -> List[llir.Stmt]:
        """Copy a dense workspace into a result row below a compressed level.

        ``level`` is the result's last level, dense, with a compressed level
        somewhere above it.  The row sits at ``q * E`` in an append-built
        value array, ``q`` the position of the level above and ``E`` this
        level's extent, so the array is grown to hold the row before the copy.
        Nothing is written unless the producer ran: a row the operands store
        nothing under is not stored, whatever the workspace holds.

        When the level above is the compressed one, the row is the
        coordinate's whole block and the copy commits it -- the coordinate is
        appended and its position advances.  When dense levels lie between,
        the block spans several rows and the commit after the loop over the
        block's first dense level pads and appends once for all of them.
        """

        if not (parent_index_var is not None and parent_level_type is not None):
            raise ScorchAssertionError(
                "parent_index_var is not None and parent_level_type is not None",
                code="cin_lowering_state_missing",
            )
        result_name = result_tensor_access.get_tensor().get_name()
        values_name = f"{result_name}_values"
        scalar_type = dtype_to_c_datatype(wksp.dtype)
        parent_position = llir.Var(
            name=f"p{result_name}{level - 1}", type=llir.DataType.INT64
        )
        extent = llir.Var(name=f"{result_name}{level}_size", type=llir.DataType.INT64)
        row_start = llir.Mul(left=parent_position, right=extent)
        row_stop = llir.Mul(
            left=llir.Add(parent_position, llir.Literal(1)), right=extent
        )
        values_write = self._result_storage(
            result_tensor_access, (RESULT_VALUES, None, STORAGE_WRITE)
        )
        grow: llir.Stmt = llir.FunctionCallStmt(
            name=f"{values_name}.resize",
            args=[row_stop, llir.Literal(0)],
            result_storage=values_write,
        )
        if parent_level_type != LevelType.COMPRESSED:
            # Rows of one block arrive in increasing order, so this only ever
            # grows; the test keeps a later block's first row from shrinking
            # nothing and an already-padded array from being cut.
            grow = llir.IfThenElse(
                cond=llir.BinOp(
                    op="<",
                    left=llir.FunctionCall(name=f"{values_name}.size", args=[]),
                    right=row_stop,
                ),
                then_body=[grow],
                result_storage=self._result_storage(
                    result_tensor_access, (RESULT_VALUES, None, STORAGE_READ)
                ),
            )
        # Below the compressed level the row is the whole block, opened here;
        # below a dense level the block opened before the loop over its first
        # level, where the commit is.
        block_opens: List[llir.Stmt] = []
        if parent_level_type == LevelType.COMPRESSED:
            block_opens.append(
                dense_block_check(
                    self.result_levels_bounded_in_kernel,
                    result_name,
                    result_tensor_access.level_types(),
                    level,
                )
            )
        body: List[llir.Stmt] = [
            *block_opens,
            grow,
            llir.FunctionCallStmt(
                name="memcpy",
                args=[
                    llir.AddressOf(
                        operand=llir.ArrayAccess(
                            array=llir.Var(
                                name=values_name,
                                type=llir.DataType.ptr_type(scalar_type),
                            ),
                            index=row_start,
                        ),
                    ),
                    llir.Var(
                        name=wksp.get_name(), type=llir.DataType.ptr_type(scalar_type)
                    ),
                    llir.Mul(
                        llir.Var(
                            name=wksp_index_var.size_llir_var.name,
                            type=llir.DataType.INT64,
                        ),
                        llir.Sizeof(data_type=scalar_type),
                    ),
                ],
                result_storage=values_write,
            ),
        ]
        if parent_level_type == LevelType.COMPRESSED:
            body.extend(
                [
                    llir.FunctionCallStmt(
                        name=f"{result_name}{level - 1}_crd.push_back",
                        args=[
                            llir.Var(name=parent_index_var.name, type=llir.DataType.INT)
                        ],
                        result_storage=self._result_storage(
                            result_tensor_access,
                            (RESULT_CRD, level - 1, STORAGE_WRITE),
                        ),
                    ),
                    llir.Increment(var=parent_position),
                ]
            )
        return [
            llir.BlankLine(),
            llir.Comment("Write workspace to its result row, if the producer ran"),
            llir.IfThenElse(
                cond=llir.Var(name=written_flag, type=llir.DataType.BOOL),
                then_body=body,
            ),
        ]

    def _refuse_consumer_result_cell(
        self,
        result_tensor_access: TensorAccess,
        level: int,
        why: str,
        workspace_kind: str = "dense",
    ) -> NoReturn:
        """Refuse a workspace whose entries have no result cell to go to."""

        result_tensor = result_tensor_access.get_tensor()
        diagnostic = CINDiagnostic(
            code="unsupported_workspace_result_level",
            message=(
                f"a {workspace_kind} workspace is copied into level {level} of "
                f"{result_tensor.get_name()!r}, and {why}"
            ),
            path=("root",),
            entity_id=result_tensor.symbol_id,
            stage="legacy_cin_lowering",
            pass_name="workspace_consumer",
        )
        raise UnsupportedFeature(
            "stage=legacy CIN lowering: unsupported_workspace_result_level for "
            f"result {result_tensor.get_name()!r} (level {level})",
            diagnostics=(diagnostic,),
        )

    def _consumer_result_cell(
        self,
        result_tensor_access: TensorAccess,
        level: int,
        key: Sequence[IndexVar] = (),
    ) -> Tuple[List[llir.Stmt], llir.Var]:
        """Positions of the result cell one entry of a workspace goes to.

        ``level`` is the result level the workspace's variable indexes.  The
        cell's position is the position of the result's LAST level, which is
        ``level`` itself only when the loops bind the result's levels outermost
        first.  A result dense in every level is not held to that order
        (``IterationLattice._dense_result_levels_resolved_at``):
        ``einsum('abc,dc->adb', format='ddd')`` loops ``a, b`` around a
        workspace over ``d``, the result's middle level.  The copy-out wrote
        ``C_values[pC1]`` there, the position of a row, not of a cell.

        ``key`` names the result variables the consumer binds itself, beside
        the enclosing loops: a sparse workspace's key, which its drain reads
        from each entry (:meth:`_lower_dense_result_scatter`), with ``level``
        the first level among them.  A dense workspace's copy-out loops over
        its one variable, the variable of ``level``, and passes none.

        Returned are the declarations of the position at ``level`` and at every
        deeper level, each from the one above it, and the last one's variable.
        With ``level`` the last level these are the declarations this lowering
        always emitted.  The first level's position is its coordinate.

        Refused, with ``unsupported_workspace_result_level``: a level from
        ``level`` down that is not dense (its position is a counter, not
        arithmetic), a level above whose variable no enclosing loop binds, and
        a level below whose variable neither an enclosing loop nor ``key``
        binds.
        """

        level_types = result_tensor_access.level_types()
        level_vars = result_tensor_access.get_sorted_index_vars()
        name = result_tensor_access.get_tensor().get_name()
        bound = [
            loop.parent if loop.has_parent else loop
            for loop in self._enclosing_forall_vars
        ]
        binders = (
            "neither an enclosing loop nor the workspace key binds"
            if key
            else "no enclosing loop binds"
        )

        def refuse(why: str) -> NoReturn:
            self._refuse_consumer_result_cell(
                result_tensor_access,
                level,
                why,
                workspace_kind="sparse" if key else "dense",
            )

        if level_types[level] != LevelType.DENSE:
            refuse(f"that level is {level_types[level].name.lower()}")
        above = level - 1
        while above >= 0 and level_types[above] == LevelType.DENSE:
            if level_vars[above] not in bound:
                refuse(
                    f"no enclosing loop binds {level_vars[above].name}, the "
                    f"variable of level {above} above it"
                )
            above -= 1
        stmts: List[llir.Stmt] = []
        if level == 0:
            stmts.append(
                llir.VarInit(
                    var=llir.Var(name=f"p{name}0", type=llir.DataType.INT64),
                    value=llir.Var(name=level_vars[0].name, type=llir.DataType.INT64),
                )
            )
        else:
            stmts.extend(
                result_tensor_access.get_level_iterator_resolve_stmts(level=level)
            )
        for deeper in range(level + 1, len(level_types)):
            if level_types[deeper] != LevelType.DENSE:
                refuse(f"level {deeper} below it is {level_types[deeper].name.lower()}")
            if level_vars[deeper] not in bound and level_vars[deeper] not in key:
                refuse(
                    f"{binders} {level_vars[deeper].name}, the "
                    f"variable of level {deeper} below it"
                )
            stmts.extend(
                result_tensor_access.get_level_iterator_resolve_stmts(level=deeper)
            )
        return stmts, llir.Var(
            name=f"p{name}{len(level_types) - 1}", type=llir.DataType.INT64
        )

    def _reductions_around_consumer(
        self, result_tensor_access: TensorAccess
    ) -> List[IndexVar]:
        """The loops around a Where whose index variable the result lacks.

        The scheduler inserts a workspace at one reduction loop
        (``Scheduler.insert_workspace``) and leaves the loops above it where
        they were, reductions included.  ``einsum('abce,cef->abf')`` over a
        compressed ``A`` puts the workspace over ``f`` inside the loop over
        ``c``; ``einsum('bce,cef->bf')`` puts it inside the outer loop of a
        tiled ``e``.  The workspace then holds one iteration's share of the sum
        over that variable, and the consumer runs once per iteration.  A tile
        loop counts as the loop over its parent variable.

        Empty for a consumer that writes a workspace: a loop around it may be
        free in the final result and still absent from that workspace.
        """

        if result_tensor_access.is_workspace():
            return []
        result_vars = set(result_tensor_access.get_index_vars())
        return [
            loop
            for loop in self._enclosing_forall_vars
            if (loop.parent if loop.has_parent else loop) not in result_vars
        ]

    def _copy_out_applies_post_ops(
        self,
        result_tensor_access: TensorAccess,
        level: int,
        reductions_around: List[IndexVar],
    ) -> bool:
        """Whether this dense-workspace copy-out is where the post-ops apply.

        The copy-out's element loop writes each cell of ``level`` once per
        iteration of the loops around it, so it can apply the post-ops as it
        writes, one pass over each row while the row is hot.  That is correct
        only where the write gives every cell of the result its final value:

        - No reduction loop encloses the copy-out.  With one, each write adds
          one iteration's partial sums, and the cell is final only after the
          last.
        - Every enclosing loop over another dense level of the result visits
          every coordinate of its extent.  A loop that visits only the
          coordinates its operands store (the rows of a compressed ``A`` in
          ``A @ B``) never reaches the cells of the rows it skips.  They hold
          zero and still owe the post-ops: a bias add makes them nonzero.  A
          compressed result level needs no such loop.  The coordinates it does
          not store are not cells, and a sparse result takes only post-ops
          that keep zero (``_refuse_post_ops_the_result_cannot_take``).

        Otherwise the post-ops go to the result's assembler, which applies them
        to the finished values (``_value_post_ops``).  That pass is correct for
        every program, and costs one more pass over the result.
        """

        if not self.post_ops or not self.post_ops.ops:
            return False
        if reductions_around or result_tensor_access.is_workspace():
            return False
        level_types = result_tensor_access.level_types()
        level_vars = result_tensor_access.get_sorted_index_vars()
        for other in range(len(level_types)):
            if other == level or level_types[other] != LevelType.DENSE:
                continue
            visits = [
                visits_all
                for loop, visits_all in zip(
                    self._enclosing_forall_vars, self._enclosing_forall_visits_all
                )
                if (loop.parent if loop.has_parent else loop) == level_vars[other]
            ]
            if not visits or not all(visits):
                return False
        return True

    def _value_post_ops(self) -> Tuple[Tuple[str, Optional[str]], ...]:
        """The post-ops the result's assembler applies to its finished values.

        All of them, as ``(kind, post-op tensor name or None)``, unless a
        dense-workspace copy-out applied them to every cell
        (``_copy_out_applies_post_ops``).  Read once the loop nest is lowered,
        when the copy-out has decided.
        """

        if (
            self.post_ops is None
            or not self.post_ops.ops
            or self._post_ops_applied_in_copy_out
        ):
            return ()
        return tuple(
            (op.kind, op.tensor_name if op.kind in BINARY_POST_OPS else None)
            for op in self.post_ops.ops
        )

    def _refuse_post_ops_the_result_cannot_take(self) -> None:
        """Refuse a post-op that would change the zeros a sparse result omits.

        A sparse result stores some entries.  Every other entry is zero and is
        never written, so a post-op applied to the stored values agrees with
        one applied to every entry only if it maps zero to zero:
        ``sigmoid(0)`` is 1/2.  A post-op tensor is refused for such a result
        by the kernel's ABI (``post_op_extra_tensor_sparse_result``).  This is
        the matching refusal for the unary kinds, made before the body is
        lowered.
        """

        result = self.final_result_tensor_var
        if (
            self.post_ops is None
            or result is None
            or all(
                level_type == LevelType.DENSE for level_type in result.get_level_types()
            )
        ):
            return
        for op in self.post_ops.ops:
            if op.kind in ZERO_PRESERVING_POST_OPS or op.kind in BINARY_POST_OPS:
                continue
            diagnostic = CINDiagnostic(
                code="post_op_sparse_result_changes_zero",
                message=(
                    f"post-op {op.kind!r} does not map zero to zero, so it would "
                    "change every entry the sparse result "
                    f"{result.get_name()!r} does not store"
                ),
                path=("post_ops",),
                entity_id=result.symbol_id,
                stage="legacy_cin_lowering",
                pass_name="lower_IndexStmt",
            )
            raise UnsupportedFeature(
                "stage=legacy CIN lowering: post_op_sparse_result_changes_zero "
                f"for post-op {op.kind!r} and tensor {result.get_name()!r}",
                diagnostics=(diagnostic,),
            )

    def _refuse_consumer_inside_reduction(
        self,
        result_tensor_access: TensorAccess,
        reductions: List[IndexVar],
        what: str,
    ) -> NoReturn:
        """Refuse a workspace copy-out that cannot add partial sums."""

        result_tensor = result_tensor_access.get_tensor()
        names = ", ".join(loop.name for loop in reductions)
        diagnostic = CINDiagnostic(
            code="unsupported_workspace_inside_reduction",
            message=(
                f"the consumer of a workspace for {result_tensor.get_name()!r} "
                f"runs inside the reduction loops {names} and {what}"
            ),
            path=("root",),
            entity_id=result_tensor.symbol_id,
            stage="legacy_cin_lowering",
            pass_name="workspace_consumer",
        )
        raise UnsupportedFeature(
            "stage=legacy CIN lowering: unsupported_workspace_inside_reduction "
            f"for result {result_tensor.get_name()!r} (loops {names})",
            diagnostics=(diagnostic,),
        )

    def _scalar_workspace_appends(self, consumer: IndexStmt, wksp: Workspace) -> bool:
        """Whether ``consumer`` appends the scalar workspace ``wksp`` to a compressed or
        coordinate result level: the conditions under which lower_ConsumerIndexStmt's
        scalar-workspace branch appends rather than stores or refuses.  Such a consumer
        appends when the producer visited (scalar_workspace_visit.py)."""

        if consumer.parent == self.outermost_stmt:
            return False
        workspaces = consumer.get_workspaces()
        if not workspaces or workspaces[0] is not wksp:
            return False
        if consumer.get_workspace_accesses()[0].get_index_vars():
            return False
        results = consumer.get_result_tensor_accesses()
        if not results or not results[0].get_sorted_index_vars():
            return False
        access = results[0]
        level_type = access.level_type_of_index_var(access.get_sorted_index_vars()[-1])
        sparse = (LevelType.COMPRESSED, LevelType.COORDINATE)
        return level_type in sparse and not self._reductions_around_consumer(access)

    def _scalar_workspace_visit_flag(self, wksp: Workspace) -> llir.Var:
        """The visit flag lower_Where declared for ``wksp``.  Its absence means the
        two compute different conditions for one consumer: an invariant, not a
        program to compile."""

        flag = self._scalar_workspace_visit_flags.get(wksp.get_name())
        if flag is None:
            raise CompilerInvariantError(
                f"the consumer appends the scalar workspace {wksp.get_name()!r} to a "
                "sparse result level, but its Where declared no visit flag",
                code="scalar_workspace_visit_flag_missing",
            )
        return flag

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
        # A reduction loop around the Where runs this consumer once per
        # iteration, each time with that iteration's partial sums, so every
        # copy-out below must add into the zeroed result instead of storing.
        reductions_around = self._reductions_around_consumer(result_tensor_access)

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

            # A dense result skips a zero sum: its cell already holds 0.  A sparse
            # result appends when the producer visited (the branches below).
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
                                op=(
                                    AssignOp.ADD_ASSIGN
                                    if reductions_around
                                    else AssignOp.ASSIGN
                                ),
                                result_storage=self._result_storage(
                                    result_tensor_access,
                                    (RESULT_VALUES, None, STORAGE_WRITE),
                                ),
                            ),
                        ],
                        # The guard tests the WORKSPACE scalar against zero, so
                        # the conditional itself names no result storage.
                    )
                )
            elif reductions_around:
                self._refuse_consumer_inside_reduction(
                    result_tensor_access,
                    reductions_around,
                    f"appends the scalar workspace to {level_type.name.lower()} "
                    f"level {level}",
                )
            elif level_type == LevelType.COMPRESSED:
                # Kept when the producer visited, whatever the sum: a test of the
                # sum against zero would drop a visited entry whose sum is 0.
                stmts.append(
                    llir.IfThenElse(
                        cond=self._scalar_workspace_visit_flag(wksp),
                        then_body=[
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}{level}_crd.push_back",
                                args=[
                                    llir.Var(
                                        name=index_var.name,
                                        type=llir.DataType.INT64,
                                    )
                                ],
                                result_storage=self._result_storage(
                                    result_tensor_access,
                                    (RESULT_CRD, level, STORAGE_WRITE),
                                ),
                            ),
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}_values.push_back",
                                args=[wksp_var],
                                result_storage=self._result_storage(
                                    result_tensor_access,
                                    (RESULT_VALUES, None, STORAGE_WRITE),
                                ),
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
                                result_storage=self._result_storage(
                                    result_tensor_access,
                                    (RESULT_CRD, lvl, STORAGE_WRITE),
                                ),
                            )
                        )
                push_stmts.append(
                    llir.FunctionCallStmt(
                        name=f"{result_tensor_name}_values.push_back",
                        args=[wksp_var],
                        result_storage=self._result_storage(
                            result_tensor_access,
                            (RESULT_VALUES, None, STORAGE_WRITE),
                        ),
                    )
                )
                # Kept when the producer visited, as the compressed branch above.
                stmts.append(
                    llir.IfThenElse(
                        cond=self._scalar_workspace_visit_flag(wksp),
                        then_body=push_stmts,
                    )
                )
            else:
                raise ScorchNotImplementedError(
                    f"TODO: need to handle assembly of workspace with {level_type} level",
                    code="workspace_assembly_level_unsupported",
                )
            return [
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                *stmts,
                llir.BlankLine(),
            ]

        if self._is_dense_result_scatter(wksp_access, result_tensor_access):
            return self._lower_dense_result_scatter(
                wksp, wksp_access, result_tensor_access, reductions_around
            )

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
            if parent_index_var is None:
                raise ScorchAssertionError(
                    "parent_index_var should not be None",
                    code="cin_lowering_state_missing",
                )
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
            if len(wksp_index_vars) != 1:
                raise ScorchAssertionError(
                    "dense workspace has more than 1 index var",
                    code="cin_lowering_state_missing",
                )
            wksp_index_var = wksp_index_vars[0]

            if not wksp_index_var.tile_size_var:
                # Check if the result level is dense (contiguous layout).
                result_level_type = result_tensor_access.level_type_of_index_var(
                    wksp_index_var
                )
                result_is_dense = (
                    result_level_type is not None and result_level_type.name == "DENSE"
                )

                written_flag = self._workspace_written_flags.get(wksp.get_name())
                last_level = result_tensor_access.num_levels - 1
                if written_flag is not None and level != last_level:
                    self._refuse_consumer_result_cell(
                        result_tensor_access,
                        level,
                        "its row is appended below a compressed level and is "
                        "not the result's last level",
                    )
                # Post-ops no longer refuse a copy-out of partial sums: they are
                # then applied to the finished values (``_value_post_ops``).
                if reductions_around and (
                    not result_is_dense or written_flag is not None
                ):
                    self._refuse_consumer_inside_reduction(
                        result_tensor_access,
                        reductions_around,
                        "copies the dense workspace into a result row that is "
                        "appended, which partial sums cannot be",
                    )
                post_ops_here = self._copy_out_applies_post_ops(
                    result_tensor_access, level, reductions_around
                )
                if result_is_dense and written_flag is not None:
                    return self._dense_block_workspace_copy_out(
                        result_tensor_access,
                        wksp,
                        wksp_index_var,
                        level,
                        parent_index_var,
                        parent_level_type,
                        written_flag,
                    )
                if (
                    result_is_dense
                    and not reductions_around
                    and level == last_level
                    and not post_ops_here
                ):
                    # Emit memcpy: the workspace has the full row, write once.
                    # A copy-out that applies post-ops takes the element loop
                    # below, which applies them as it writes.
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
                            # A free function receiving the result's value
                            # storage through an ``AddressOf`` -- the
                            # argument-shaped write no name matcher can see.
                            result_storage=self._result_storage(
                                result_tensor_access,
                                (RESULT_VALUES, None, STORAGE_WRITE),
                            ),
                        ),
                    ]

                # Fallback: element-by-element assignment, ``=`` when the
                # workspace holds the whole sum, ``+=`` when it holds one
                # iteration's share of a reduction around the Where.
                loop_var = llir.Var(
                    name=f"{wksp_index_var.name}",
                    type=llir.DataType.INT64,
                )

                loop_body: List[llir.Stmt] = []

                position_stmts, cell = self._consumer_result_cell(
                    result_tensor_access, level
                )
                loop_body.extend(position_stmts)

                loop_body.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_tensor_name}_values",
                                type=llir.DataType.NO_TYPE,
                            ),
                            index=cell,
                            tensor_access=result_value_metadata,
                        ),
                        value=llir.Var(
                            name=f"{wksp.get_name()}[{loop_var.name}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        op=(
                            AssignOp.ADD_ASSIGN
                            if reductions_around
                            else AssignOp.ASSIGN
                        ),
                        result_storage=self._result_storage(
                            result_tensor_access,
                            (RESULT_VALUES, None, STORAGE_WRITE),
                        ),
                    )
                )

                # Inject post-ops (bias add, relu, etc.) after workspace -> output
                # write, where this copy-out reaches every cell with its final
                # value; elsewhere the result's assembler applies them.
                if post_ops_here:
                    loop_body.extend(
                        self._emit_post_ops(
                            f"{result_tensor_name}_values",
                            cell.name,
                            result_value_metadata,
                        )
                    )
                    self._post_ops_applied_in_copy_out = True

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

            if not (wksp_index_var.tile_size_var and wksp_index_var.is_inner):
                raise ScorchAssertionError(
                    "Dense accumulator used not for tiling",
                    code="cin_lowering_state_missing",
                )

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

            position_stmts, cell = self._consumer_result_cell(
                result_tensor_access, level
            )
            loop_body.extend(position_stmts)

            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}_values",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=cell,
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
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_WRITE),
                    ),
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

        if reductions_around:
            self._refuse_consumer_inside_reduction(
                result_tensor_access,
                reductions_around,
                "drains a sparse workspace into stored coordinates",
            )

        first_key_level = self._nested_ordered_key_level(
            wksp_access, result_tensor_access
        )
        if first_key_level is not None:
            return self._lower_nested_ordered_key_drain(
                wksp, wksp_access, result_tensor_access, first_key_level
            )

        mixed_key_level = self._mixed_key_drain_level(wksp_access, result_tensor_access)
        if mixed_key_level is not None:
            return self._lower_mixed_key_drain(
                wksp, wksp_access, result_tensor_access, mixed_key_level
            )

        if level_type not in (LevelType.COMPRESSED, LevelType.COORDINATE):
            # The drain below appends one value and one coordinate per sorted
            # key at ``level``, which is assembly for a level that stores
            # coordinates.  A dense level stores none: the loop would write a
            # coordinate array the result does not have, and the kernel did
            # not build.
            raise UnsupportedFeature(
                "stage=CIN lowering: a sparse workspace is drained into result "
                f"level {level} of {result_tensor_name!r}, which is "
                f"{level_type.name.lower()}; the drain assembles a compressed or "
                "coordinate level",
                code="unsupported_workspace_result_level",
            )

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
                result_storage=self._result_storage(
                    result_tensor_access,
                    (RESULT_VALUES, None, STORAGE_WRITE),
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
                # The drained level is the workspace key's last variable.  A
                # one-variable key has no other; with a longer key this stored
                # the key's FIRST variable in the last level.
                value=llir.Var(
                    name=f"{wksp_index_vars[-1].name}",
                    type=llir.DataType.NO_TYPE,
                ),
                result_storage=self._result_storage(
                    result_tensor_access,
                    (RESULT_CRD, level, STORAGE_WRITE),
                ),
            )
        )

        # A coordinate level stores one coordinate per entry, so every
        # coordinate level above this one, up to the first that is not, gets
        # its coordinate with each entry: the drained key's when the level's
        # variable is part of the key, the enclosing loop's otherwise.  Only
        # the parent's was set, and a result coordinate in three or more
        # levels -- ``einsum('abc,cd->abd', format='ooo')`` -- came back with
        # no coordinates in its outer levels.
        level_vars = result_tensor_access.get_sorted_index_vars()
        level_types = result_tensor_access.level_types()
        bound = [
            loop.parent if loop.has_parent else loop
            for loop in self._enclosing_forall_vars
        ]
        key_vars = [
            key_var.parent if key_var.has_parent else key_var
            for key_var in wksp_index_vars
        ]
        if len(key_vars) > 1 and level_type == LevelType.COORDINATE:
            # The drain visits keys in sorted order, which is the result's
            # order only when the key's variables are the result's last
            # levels in level order.
            key_levels = [
                result_tensor_access.level_of_index_var(key_var) for key_var in key_vars
            ]
            if key_levels != list(range(level - len(key_vars) + 1, level + 1)):
                diagnostic = CINDiagnostic(
                    code="unsupported_coordinate_drain_key_order",
                    message=(
                        f"the workspace key ({', '.join(v.name for v in key_vars)}) "
                        f"is drained into levels {key_levels} of "
                        f"{result_tensor_name!r}; a sorted drain stores a "
                        "coordinate result in order only when the key's "
                        "variables are its last levels, in level order"
                    ),
                    path=("root",),
                    entity_id=result_tensor_access.get_tensor().symbol_id,
                    stage="legacy_cin_lowering",
                    pass_name="workspace_consumer",
                )
                raise UnsupportedFeature(
                    "stage=legacy CIN lowering: "
                    "unsupported_coordinate_drain_key_order for result "
                    f"{result_tensor_name!r}",
                    diagnostics=(diagnostic,),
                )
        ancestor = level - 1
        while ancestor >= 0 and level_types[ancestor] == LevelType.COORDINATE:
            ancestor_var = level_vars[ancestor]
            if ancestor_var not in bound and ancestor_var not in key_vars:
                diagnostic = CINDiagnostic(
                    code="unsupported_coordinate_drain_unbound_level",
                    message=(
                        f"the drain into coordinate level {level} of "
                        f"{result_tensor_name!r} needs the coordinate of level "
                        f"{ancestor} ({ancestor_var.name}), which no enclosing "
                        "loop binds"
                    ),
                    path=("root",),
                    entity_id=result_tensor_access.get_tensor().symbol_id,
                    stage="legacy_cin_lowering",
                    pass_name="workspace_consumer",
                )
                raise UnsupportedFeature(
                    "stage=legacy CIN lowering: "
                    "unsupported_coordinate_drain_unbound_level for result "
                    f"{result_tensor_name!r}",
                    diagnostics=(diagnostic,),
                )
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{ancestor}_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=result_level_iterator_name,
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.Var(
                        name=(
                            wksp_index_vars[key_vars.index(ancestor_var)].name
                            if ancestor_var in key_vars
                            else ancestor_var.name
                        ),
                        type=llir.DataType.NO_TYPE,
                    ),
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_CRD, ancestor, STORAGE_WRITE),
                    ),
                )
            )
            ancestor -= 1

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

        assembly_stmts = self._drained_segment_assembly(
            result_tensor_access,
            level,
            level_type,
            parent_index_var,
            parent_level_type,
        )

        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            wksp_sort_stmt,
            loop_stmt,
            llir.BlankLine(),
            *assembly_stmts,
        ]

    def _lower_dense_result_scatter(
        self,
        wksp: Workspace,
        wksp_access: WorkspaceAccess,
        result_tensor_access: TensorAccess,
        reductions_around: List[IndexVar],
    ) -> List[llir.Stmt]:
        """Drain a sparse workspace into a result whose levels are all dense.

        A dense result is one array of its full size, zeroed before the loop
        nest runs, and a cell's position is arithmetic: the position of level
        ``l`` is the position of level ``l - 1`` times the extent of ``l`` plus
        the coordinate at ``l``.  So the drain is a scatter.  It reads each
        entry's key and sum, computes the entry's cell from the key and the
        enclosing loops, and stores the sum there.  It appends no coordinate and
        needs no order, and the cells no entry names keep their zero.  Under a
        reduction loop around the Where, each entry is one iteration's partial
        sum, so it is added into its cell.  The other drains assemble a
        compressed or coordinate level, which such a result does not have; the
        kernel they printed for ``dd = oo dd`` did not build.

        The key's components are read in the workspace's storage order, the
        order its producer inserts them in.  A component that is the inner
        variable of a split stands for its parent, the result's variable, which
        is resolved from the outer variable an enclosing loop binds.  The levels
        above the key's first level are bound by enclosing loops, which declare
        their positions; from that level down the positions are declared here
        (:meth:`_consumer_result_cell`).

        ``coo_workspace_1d`` iterates its entries in insertion order without a
        sort.  ``coo_workspace<T, N>``, the workspace of a key with several
        components, iterates through the index its ``sort`` builds, so it is
        sorted first.

        The post-ops are left to the result's assembler, which applies them to
        the finished values (:meth:`_value_post_ops`): the scatter does not reach
        the cells no entry names, and a bias add makes those nonzero.

        Refused, with ``unsupported_workspace_result_level``: a key whose
        components do not stand for distinct variables of the result, one per
        component of the workspace, a split whose outer variable no enclosing
        loop binds, and a result level whose variable neither the key nor an
        enclosing loop binds.
        """

        result_tensor_name = result_tensor_access.get_tensor().get_name()
        result_value_metadata = self._tensor_access_metadata(
            result_tensor_access,
            llir.TensorAccessRole.RESULT_WRITE,
        )
        level_vars = list(result_tensor_access.get_sorted_index_vars())
        key = list(wksp_access.get_sorted_index_vars())
        key_vars = [var.parent if var.has_parent else var for var in key]
        wksp_name = wksp.get_name()

        def refuse(level: int, why: str) -> NoReturn:
            self._refuse_consumer_result_cell(
                result_tensor_access, level, why, workspace_kind="sparse"
            )

        if len(key) != wksp.dim:
            refuse(
                len(level_vars) - 1,
                f"its access names {len(key)} of the workspace's {wksp.dim} "
                "key components",
            )
        for var, key_var in zip(key, key_vars):
            if key_var not in level_vars:
                refuse(
                    len(level_vars) - 1,
                    f"its key component {var.name} stands for no variable of "
                    "the result",
                )
        if len(set(key_vars)) != len(key_vars):
            refuse(
                len(level_vars) - 1,
                "two of its key components stand for one variable of the result",
            )
        enclosing = list(self._enclosing_forall_vars)
        for var in key:
            if not var.has_parent:
                continue
            split = var.parent.expr
            outer = [split.lhs, split.rhs] if isinstance(split, IndexVarAdd) else []
            outer = [operand for operand in outer if operand != var]
            if len(outer) != 1 or outer[0] not in enclosing:
                refuse(
                    level_vars.index(var.parent),
                    f"no enclosing loop binds the outer variable of the split "
                    f"of {var.parent.name} its key component {var.name} is in",
                )

        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        loop_body: List[llir.Stmt] = []
        for key_position, var in enumerate(key):
            entry_key: llir.Expr = llir.MemberAccess(
                base=llir.Var(name=loop_var.name, type=loop_var.type),
                member="first",
            )
            if wksp.dim > 1:
                entry_key = llir.ArrayAccess(
                    array=entry_key,
                    index=llir.Literal(key_position, llir.DataType.INT64),
                )
            loop_body.append(
                llir.VarInit(
                    var=llir.Var(name=var.name, type=llir.DataType.INT64),
                    value=entry_key,
                )
            )
        for var in key:
            if var.has_parent:
                loop_body.extend(var.parent.get_resolve_llir_stmts())
        loop_body.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{wksp_name}_value",
                    type=dtype_to_c_datatype(wksp.dtype),
                ),
                value=llir.MemberAccess(
                    base=llir.Var(name=loop_var.name, type=loop_var.type),
                    member="second",
                ),
            )
        )
        position_stmts, cell = self._consumer_result_cell(
            result_tensor_access,
            min(level_vars.index(key_var) for key_var in key_vars),
            key=key_vars,
        )
        loop_body.extend(position_stmts)
        loop_body.append(
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_tensor_name}_values",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=cell,
                    tensor_access=result_value_metadata,
                ),
                value=llir.Var(name=f"{wksp_name}_value", type=llir.DataType.NO_TYPE),
                op=AssignOp.ADD_ASSIGN if reductions_around else AssignOp.ASSIGN,
                result_storage=self._result_storage(
                    result_tensor_access,
                    (RESULT_VALUES, None, STORAGE_WRITE),
                ),
            )
        )

        stmts: List[llir.Stmt] = [
            llir.BlankLine(),
            llir.Comment("Scatter the sparse workspace into the dense result"),
        ]
        if wksp.dim > 1:
            stmts.append(llir.FunctionCallStmt(name=f"{wksp_name}.sort", args=[]))
        stmts.append(
            llir.ForLoopAuto(
                var=loop_var,
                array=llir.Var(name=wksp_name, type=llir.DataType.AUTO),
                body=loop_body,
            )
        )
        if wksp_name in self._pooled_scatter_workspaces:
            # The worker's workspace outlives this drain: empty it for the
            # next, resetting only the coordinates this drain read.
            stmts.append(
                llir.MemberCallStmt(
                    base=llir.Var(name=wksp_name, type=llir.DataType.NO_TYPE),
                    member="clear",
                )
            )
        return stmts

    @staticmethod
    def _is_dense_result_scatter(
        wksp_access: WorkspaceAccess, result_tensor_access: TensorAccess
    ) -> bool:
        """Whether a consumer drains a sparse workspace by a scatter.

        The route :meth:`lower_ConsumerIndexStmt` sends to
        :meth:`_lower_dense_result_scatter`: a sparse workspace drained into a
        result that is not a workspace and whose levels are all dense.
        """
        return (
            not wksp_access.is_dense()
            and not result_tensor_access.is_workspace()
            and all(
                level_type == LevelType.DENSE
                for level_type in result_tensor_access.level_types()
            )
        )

    def _scatter_pool_extent(self, stmt: Where, wksp: Workspace) -> Optional[str]:
        """The capacity of ``wksp``'s per-worker pool, if a scatter drains it.

        Read before the consumer is lowered, by the predicate the consumer is
        routed by (:meth:`_is_dense_result_scatter`), so that the workspace is
        declared for its drain: one ``coo_workspace_owner_1d`` per worker, of
        capacity the extent of the result level its key stands for, which
        native_abi.h validated every coordinate against, so the worker's view
        inserts unchecked and never grows; the view is cleared after each
        drain.  In a parallel region each worker allocates its owner's arrays
        once, at its first ``make_region_view()``, without throwing.  Made in
        every iteration, the workspace was three allocations and a zeroed flag
        array regrown up to the widest column, freed again at the end of the
        iteration.

        ``None`` keeps the declaration inside the loop: a consumer of the
        outermost statement (lowered by :meth:`lower_outer_ConsumerIndexStmt`),
        another drain, a key that stands for no variable of the result (which
        the scatter refuses), or a scalar type that has no pool.
        """
        consumer = stmt.consumer
        if consumer.parent == self.outermost_stmt:
            return None
        workspaces = consumer.get_workspaces()
        workspace_accesses = consumer.get_workspace_accesses()
        result_accesses = consumer.get_result_tensor_accesses()
        if (
            not workspaces
            or not workspace_accesses
            or not result_accesses
            or workspaces[0].get_name() != wksp.get_name()
        ):
            return None
        wksp_access = workspace_accesses[0]
        result_access = result_accesses[0]
        if not self._is_dense_result_scatter(wksp_access, result_access):
            return None
        key = list(wksp_access.get_sorted_index_vars())
        if len(key) != 1:
            return None
        key_var = key[0].parent if key[0].has_parent else key[0]
        level_vars = list(result_access.get_sorted_index_vars())
        if key_var not in level_vars:
            return None
        try:
            llir.DataType.coo_workspace_1d_pool_type(dtype_to_c_datatype(wksp.dtype))
        except ValueError:
            return None
        result_name = result_access.get_tensor().get_name()
        return f"{result_name}{level_vars.index(key_var)}_size"

    @staticmethod
    def _nested_ordered_key_level(
        wksp_access: WorkspaceAccess, result_tensor_access: TensorAccess
    ) -> Optional[int]:
        """First result level of a nested drain whose key spans several levels.

        The drain under :meth:`_lower_nested_ordered_key_drain` handles a sparse
        workspace keyed on two or more unsplit index variables that are, in
        order, the result's trailing compressed levels, below at least one
        result level bound by an enclosing loop.  None for every other drain,
        which keeps its existing lowering.
        """

        key = list(wksp_access.get_index_vars())
        if len(key) < 2 or wksp_access.is_dense():
            return None
        if any(index_var.has_parent for index_var in key):
            return None
        levels = result_tensor_access.get_sorted_index_vars()
        first = len(levels) - len(key)
        if first < 1 or list(levels[first:]) != key:
            return None
        if any(
            result_tensor_access.level_type_of_index_var(index_var)
            != LevelType.COMPRESSED
            for index_var in key
        ):
            return None
        return first

    def _lower_nested_ordered_key_drain(
        self,
        wksp: Workspace,
        wksp_access: WorkspaceAccess,
        result_tensor_access: TensorAccess,
        first_level: int,
    ) -> List[llir.Stmt]:
        """Drain a multi-level sparse workspace into nested compressed levels.

        The sorted workspace visits its keys in result level order, so each run
        of keys that share a prefix is one segment of the level below that
        prefix.  Every entry appends its leaf coordinate and value.  When the key
        changes at some level, the runs that ended are closed deepest first,
        exactly as the loop over a compressed level closes it
        (``SparseLattice.get_lattice_loops``): the finished segment's parent
        coordinate is appended if the segment is non-empty, and the segment's
        end position is written.  After the last entry every run is closed, and
        :meth:`_drained_segment_assembly` closes the outermost key level against
        the coordinate of the loop that encloses the drain.

        Keeping the closing step in the conditional form the ordinary assembly
        emits keeps it recognizable to the two-phase parallel assembly
        (``result_write_pass._rewrite_if_statement``).
        """

        result_tensor_name = result_tensor_access.get_tensor().get_name()
        result_value_metadata = self._tensor_access_metadata(
            result_tensor_access,
            llir.TensorAccessRole.RESULT_WRITE,
        )
        key = list(wksp_access.get_index_vars())
        leaf_level = first_level + len(key) - 1
        wksp_name = wksp.get_name()
        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)

        def level_count(level: int) -> llir.Var:
            return llir.Var(
                name=f"p{result_tensor_name}{level}", type=llir.DataType.INT64
            )

        def previous(position: int) -> llir.Var:
            return llir.Var(
                name=f"{wksp_name}_prev{position}", type=llir.DataType.INT64
            )

        def close_segment(level: int) -> List[llir.Stmt]:
            # The segment of ``level`` under the previous key's parent coordinate
            # has ended: append that coordinate if the segment is non-empty, then
            # write where the segment ends.
            position = level - 1 - first_level
            return [
                llir.IfThenElse(
                    cond=llir.BinOp(
                        op="<",
                        left=llir.FunctionCall(
                            name=f"{result_tensor_name}{level}_pos.back",
                            args=[],
                        ),
                        right=level_count(level),
                    ),
                    then_body=[
                        llir.FunctionCallStmt(
                            name=f"{result_tensor_name}{level - 1}_crd.push_back",
                            args=[previous(position)],
                            result_storage=self._result_storage(
                                result_tensor_access,
                                (RESULT_CRD, level - 1, STORAGE_WRITE),
                            ),
                        ),
                        llir.Increment(var=level_count(level - 1)),
                    ],
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_POS, level, STORAGE_READ),
                    ),
                ),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{level}_pos",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        ),
                        index=llir.FunctionCall(
                            name=f"{result_tensor_name}{level - 1}_crd.size",
                            args=[],
                        ),
                    ),
                    value=llir.FunctionCall(
                        name=f"{result_tensor_name}{level}_crd.size",
                        args=[],
                    ),
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_POS, level, STORAGE_WRITE),
                        (RESULT_CRD, level - 1, STORAGE_READ),
                        (RESULT_CRD, level, STORAGE_READ),
                    ),
                ),
            ]

        # Coordinates are non-negative, so -1 differs from every first key.
        previous_inits: List[llir.Stmt] = [
            llir.VarInit(var=previous(position), value=llir.Literal(-1))
            for position in range(len(key) - 1)
        ]

        loop_body: List[llir.Stmt] = []
        for position, index_var in enumerate(key):
            loop_body.append(
                llir.VarInit(
                    var=llir.Var(name=index_var.name, type=llir.DataType.INT64),
                    value=llir.ArrayAccess(
                        array=llir.MemberAccess(
                            base=llir.Var(name=loop_var.name, type=loop_var.type),
                            member="first",
                        ),
                        index=llir.Literal(position, llir.DataType.INT64),
                    ),
                )
            )
        loop_body.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{wksp_name}_value",
                    type=dtype_to_c_datatype(wksp.dtype),
                ),
                value=llir.MemberAccess(
                    base=llir.Var(name=loop_var.name, type=loop_var.type),
                    member="second",
                ),
            )
        )
        loop_body.append(llir.BlankLine())

        # Close, deepest first, every segment whose parent prefix changed.
        for level in range(leaf_level, first_level, -1):
            changed: Optional[llir.Expr] = None
            for position in range(level - first_level):
                differs = llir.BinOp(
                    op="!=",
                    left=llir.Var(name=key[position].name, type=llir.DataType.INT64),
                    right=previous(position),
                )
                changed = (
                    differs
                    if changed is None
                    else llir.BinOp(op="||", left=changed, right=differs)
                )
            if changed is None:
                raise ScorchAssertionError(
                    "changed is not None", code="cin_lowering_state_missing"
                )
            loop_body.append(
                llir.IfThenElse(cond=changed, then_body=close_segment(level))
            )
        for position in range(len(key) - 1):
            loop_body.append(
                llir.Assign(
                    var=previous(position),
                    value=llir.Var(name=key[position].name, type=llir.DataType.INT64),
                )
            )

        # The leaf entry, written exactly as the one-level drain writes it.
        loop_body.append(
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_tensor_name}_values",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(
                        name=f"p{result_tensor_name}{leaf_level}",
                        type=llir.DataType.INT64,
                    ),
                    tensor_access=result_value_metadata,
                ),
                value=llir.Var(name=f"{wksp_name}_value", type=llir.DataType.NO_TYPE),
                result_storage=self._result_storage(
                    result_tensor_access,
                    (RESULT_VALUES, None, STORAGE_WRITE),
                ),
            )
        )
        loop_body.append(
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_tensor_name}{leaf_level}_crd",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(
                        name=f"p{result_tensor_name}{leaf_level}",
                        type=llir.DataType.INT64,
                    ),
                ),
                value=llir.Var(name=key[-1].name, type=llir.DataType.NO_TYPE),
                result_storage=self._result_storage(
                    result_tensor_access,
                    (RESULT_CRD, leaf_level, STORAGE_WRITE),
                ),
            )
        )
        loop_body.append(
            llir.Increment(
                var=llir.Var(
                    name=f"p{result_tensor_name}{leaf_level}",
                    type=llir.DataType.NO_TYPE,
                )
            )
        )

        closing: List[llir.Stmt] = []
        for level in range(leaf_level, first_level, -1):
            closing.extend(close_segment(level))

        parent_index_var = result_tensor_access.get_parent_index_var(key[0])
        if parent_index_var is None:
            raise ScorchAssertionError(
                "a nested drain has an enclosing level",
                code="cin_lowering_state_missing",
            )
        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            llir.FunctionCallStmt(name=f"{wksp_name}.sort", args=[]),
            *previous_inits,
            llir.ForLoopAuto(
                var=loop_var,
                array=llir.Var(name=wksp_name, type=llir.DataType.AUTO),
                body=loop_body,
            ),
            *closing,
            llir.BlankLine(),
            *self._drained_segment_assembly(
                result_tensor_access,
                first_level,
                LevelType.COMPRESSED,
                parent_index_var,
                result_tensor_access.level_type_of_index_var(parent_index_var),
            ),
        ]

    @staticmethod
    def _mixed_key_drain_level(
        wksp_access: WorkspaceAccess, result_tensor_access: TensorAccess
    ) -> Optional[int]:
        """First result level of a drain whose key levels are not all compressed.

        The drain under :meth:`_lower_mixed_key_drain` handles a sparse
        workspace keyed on unsplit index variables that are, in order, the
        result's trailing levels, below at least one result level bound by an
        enclosing loop, when at least one of those trailing levels is dense and
        the result has a compressed level.  None for every other drain, which
        keeps its existing lowering.

        The drain stores a dense leaf into the value vector a sparse result
        grows (``dense_block_leaf_store``).  A result whose levels are all
        dense has no such vector: its values are one array of its full size,
        handed to the kernel.  The drain admitted it and emitted vector calls
        on that array, C++ that did not build.  Such a result is scattered
        into by :meth:`_lower_dense_result_scatter`, which the consumer
        lowering takes before it asks this method, so this returns None for it.
        """

        key = list(wksp_access.get_index_vars())
        if not key or wksp_access.is_dense():
            return None
        if any(index_var.has_parent for index_var in key):
            return None
        levels = result_tensor_access.get_sorted_index_vars()
        first = len(levels) - len(key)
        if first < 1 or list(levels[first:]) != key:
            return None
        level_types = result_tensor_access.level_types()
        if any(
            level_type not in (LevelType.DENSE, LevelType.COMPRESSED)
            for level_type in level_types
        ):
            return None
        if LevelType.COMPRESSED not in level_types:
            return None
        if all(
            level_type == LevelType.COMPRESSED for level_type in level_types[first:]
        ):
            return None
        return first

    def _lower_mixed_key_drain(
        self,
        wksp: Workspace,
        wksp_access: WorkspaceAccess,
        result_tensor_access: TensorAccess,
        first_level: int,
    ) -> List[llir.Stmt]:
        """Drain a sorted sparse workspace into trailing levels of mixed kinds.

        The sorted workspace visits its keys in result level order, so the
        drain is the loop nest over those levels with its loops flattened: a
        run of keys sharing a prefix is one subtree below that prefix.  Each
        key level is assembled as the loop over it would assemble it.

        A dense key level stores no coordinates.  Its position is its parent's
        position times its extent plus its coordinate, where a compressed
        parent's position is the count of coordinates that level has stored so
        far -- the position its pending coordinate will take.  A compressed
        level directly below it owns one segment per dense cell, so the cells
        the keys skip are closed before the visited one is appended to.

        A compressed key level above a dense one records its coordinate once
        the keys under it are drained, by the same commit a loop emits
        (:func:`~scorch.compiler.iter_lattice.dense_block_commit`): the block
        of dense cells below is padded to its full width and the coordinate
        appended.  A compressed level above a compressed one closes exactly as
        :meth:`_lower_nested_ordered_key_drain` closes it.

        A dense leaf is one cell of its block and is stored at its position
        with the cells before it zero-filled; a compressed leaf appends.

        Every key came from an insertion, so a subtree the drain opens always
        holds something.  The closing statements are nevertheless the
        conditional ones the loops use, so that they are no-ops before the
        first key and after an empty workspace.
        """

        result_tensor_name = result_tensor_access.get_tensor().get_name()
        level_types = result_tensor_access.level_types()
        result_value_metadata = self._tensor_access_metadata(
            result_tensor_access,
            llir.TensorAccessRole.RESULT_WRITE,
        )
        key = list(wksp_access.get_index_vars())
        leaf_level = first_level + len(key) - 1
        wksp_name = wksp.get_name()
        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)

        def position(level: int) -> llir.Var:
            # A dense level's position, or the count of coordinates a
            # compressed level has stored.
            return llir.Var(
                name=f"p{result_tensor_name}{level}", type=llir.DataType.INT64
            )

        def previous(key_position: int) -> llir.Var:
            return llir.Var(
                name=f"{wksp_name}_prev{key_position}", type=llir.DataType.INT64
            )

        def key_coordinate(key_position: int) -> llir.Var:
            return llir.Var(name=key[key_position].name, type=llir.DataType.INT64)

        def close_node(level: int, coordinate: llir.Expr) -> List[llir.Stmt]:
            # The subtree under ``coordinate`` at ``level`` has been drained.
            child = level + 1
            if level_types[level] != LevelType.COMPRESSED:
                # A dense cell: the compressed level below it closes the cell
                # when the catch-up reaches the next one.
                return []
            if level_types[child] == LevelType.DENSE:
                return dense_block_commit(
                    self, result_tensor_access, child, coordinate
                )[1:]
            return [
                llir.IfThenElse(
                    cond=llir.BinOp(
                        op="<",
                        left=llir.FunctionCall(
                            name=f"{result_tensor_name}{child}_pos.back",
                            args=[],
                        ),
                        right=position(child),
                    ),
                    then_body=[
                        llir.FunctionCallStmt(
                            name=f"{result_tensor_name}{level}_crd.push_back",
                            args=[coordinate],
                            result_storage=self._result_storage(
                                result_tensor_access,
                                (RESULT_CRD, level, STORAGE_WRITE),
                            ),
                        ),
                        llir.Increment(var=position(level)),
                    ],
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_POS, child, STORAGE_READ),
                    ),
                ),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{child}_pos",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        ),
                        index=llir.FunctionCall(
                            name=f"{result_tensor_name}{level}_crd.size",
                            args=[],
                        ),
                    ),
                    value=llir.FunctionCall(
                        name=f"{result_tensor_name}{child}_crd.size",
                        args=[],
                    ),
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_POS, child, STORAGE_WRITE),
                        (RESULT_CRD, level, STORAGE_READ),
                        (RESULT_CRD, child, STORAGE_READ),
                    ),
                ),
            ]

        # The key levels that own a subtree, deepest first, with their closes.
        closes = [
            (level, close_node(level, previous(level - first_level)))
            for level in range(leaf_level - 1, first_level - 1, -1)
        ]
        closes = [(level, stmts) for level, stmts in closes if stmts]
        tracked = max((level - first_level for level, _ in closes), default=-1) + 1

        loop_body: List[llir.Stmt] = []
        for key_position, index_var in enumerate(key):
            entry_key: llir.Expr = llir.MemberAccess(
                base=llir.Var(name=loop_var.name, type=loop_var.type),
                member="first",
            )
            if len(key) > 1:
                entry_key = llir.ArrayAccess(
                    array=entry_key,
                    index=llir.Literal(key_position, llir.DataType.INT64),
                )
            loop_body.append(
                llir.VarInit(
                    var=llir.Var(name=index_var.name, type=llir.DataType.INT64),
                    value=entry_key,
                )
            )
        loop_body.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{wksp_name}_value",
                    type=dtype_to_c_datatype(wksp.dtype),
                ),
                value=llir.MemberAccess(
                    base=llir.Var(name=loop_var.name, type=loop_var.type),
                    member="second",
                ),
            )
        )
        loop_body.append(llir.BlankLine())

        # Close, deepest first, every subtree whose own prefix changed.  A
        # compressed level over a dense block opens the next coordinate's block
        # as it closes the last one, so the block's check follows its commit,
        # before the positions below are resolved.
        for level, stmts in closes:
            if (
                level_types[level] == LevelType.COMPRESSED
                and level_types[level + 1] == LevelType.DENSE
            ):
                stmts = [
                    *stmts,
                    dense_block_check(
                        self.result_levels_bounded_in_kernel,
                        result_tensor_name,
                        level_types,
                        level + 1,
                    ),
                ]
            changed: Optional[llir.Expr] = None
            for key_position in range(level - first_level + 1):
                differs = llir.BinOp(
                    op="!=",
                    left=key_coordinate(key_position),
                    right=previous(key_position),
                )
                changed = (
                    differs
                    if changed is None
                    else llir.BinOp(op="||", left=changed, right=differs)
                )
            if changed is None:
                raise ScorchAssertionError(
                    "changed is not None", code="cin_lowering_state_missing"
                )
            loop_body.append(llir.IfThenElse(cond=changed, then_body=list(stmts)))
        for key_position in range(tracked):
            loop_body.append(
                llir.Assign(
                    var=previous(key_position), value=key_coordinate(key_position)
                )
            )

        # Resolve each dense key level's position, closing the cells the keys
        # skipped in the compressed level below it.
        for level in range(first_level, leaf_level + 1):
            if level_types[level] != LevelType.DENSE:
                continue
            resolve = result_tensor_access.get_level_iterator_resolve_stmts(level=level)
            if not resolve:
                raise ScorchAssertionError(
                    "a dense level below another level resolves its position",
                    code="cin_lowering_state_missing",
                )
            loop_body.extend(resolve)
            if level < leaf_level and level_types[level + 1] == LevelType.COMPRESSED:
                loop_body.extend(
                    compressed_child_catch_up(
                        self, result_tensor_access, level + 1, position(level)
                    )
                )

        if level_types[leaf_level] == LevelType.DENSE:
            loop_body.extend(
                dense_block_leaf_store(
                    f"{result_tensor_name}_values",
                    position(leaf_level),
                    llir.Var(name=f"{wksp_name}_value", type=llir.DataType.NO_TYPE),
                    False,
                    self._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_WRITE),
                    ),
                    self._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_READ),
                    ),
                )
            )
        else:
            # The leaf entry, written exactly as the one-level drain writes it.
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}_values",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=position(leaf_level),
                        tensor_access=result_value_metadata,
                    ),
                    value=llir.Var(
                        name=f"{wksp_name}_value", type=llir.DataType.NO_TYPE
                    ),
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_WRITE),
                    ),
                )
            )
            loop_body.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{leaf_level}_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=position(leaf_level),
                    ),
                    value=llir.Var(name=key[-1].name, type=llir.DataType.NO_TYPE),
                    result_storage=self._result_storage(
                        result_tensor_access,
                        (RESULT_CRD, leaf_level, STORAGE_WRITE),
                    ),
                )
            )
            loop_body.append(llir.Increment(var=position(leaf_level)))

        closing: List[llir.Stmt] = []
        for _, stmts in closes:
            closing.extend(stmts)

        # The outermost key level, against the loop that encloses the drain.
        parent_index_var = result_tensor_access.get_parent_index_var(key[0])
        if parent_index_var is None:
            raise ScorchAssertionError(
                "a mixed drain has an enclosing level",
                code="cin_lowering_state_missing",
            )
        parent_level_type = level_types[first_level - 1]
        outer: List[llir.Stmt]
        # A drain whose first key level opens a dense block under the enclosing
        # loop's coordinate: the block is checked once, before the drain.
        block_opens: List[llir.Stmt] = []
        if level_types[first_level] == LevelType.COMPRESSED:
            outer = self._drained_segment_assembly(
                result_tensor_access,
                first_level,
                LevelType.COMPRESSED,
                parent_index_var,
                parent_level_type,
            )
        elif parent_level_type == LevelType.COMPRESSED:
            outer = dense_block_commit(
                self,
                result_tensor_access,
                first_level,
                llir.Var(name=parent_index_var.name, type=llir.DataType.INT),
            )
            if outer:
                block_opens.append(
                    dense_block_check(
                        self.result_levels_bounded_in_kernel,
                        result_tensor_name,
                        level_types,
                        first_level,
                    )
                )
        else:
            # Dense levels all the way up to the enclosing loop, so the drain
            # itself closes nothing here.  A compressed level below the run
            # has its visited cells closed by the next catch-up.  The cells
            # after the last visited one are closed by the commit of the
            # nearest compressed level above (it pads the block to its end)
            # or, with none above, by the close that follows the whole nest.
            # Writing the last visited cell's end here as well would write
            # position entry 1 of a level that has no cells at all when a
            # dense level above it has extent zero.
            below = first_level
            while below <= leaf_level and level_types[below] == LevelType.DENSE:
                below += 1
            outer = []
            if below <= leaf_level and all(
                level_type == LevelType.DENSE for level_type in level_types[:below]
            ):
                self.result_dense_level_sparse_driven = True

        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            llir.FunctionCallStmt(name=f"{wksp_name}.sort", args=[]),
            *[
                # Coordinates are non-negative, so -1 differs from every first key.
                llir.VarInit(var=previous(key_position), value=llir.Literal(-1))
                for key_position in range(tracked)
            ],
            *block_opens,
            llir.ForLoopAuto(
                var=loop_var,
                array=llir.Var(name=wksp_name, type=llir.DataType.AUTO),
                body=loop_body,
            ),
            *closing,
            *outer,
        ]

    def _drained_segment_assembly(
        self,
        result_tensor_access: TensorAccess,
        level: int,
        level_type: Optional[LevelType],
        parent_index_var: Optional[IndexVar],
        parent_level_type: Optional[LevelType],
    ) -> List[llir.Stmt]:
        """Close the result level a nested workspace drain has just appended to.

        ``level`` is the outermost result level the drain writes; the loop over
        its parent's variable encloses the drain.  For a compressed level this
        appends the parent's coordinate when the drain produced anything and
        writes the level's position for that parent.
        """

        result_tensor_name = result_tensor_access.get_tensor().get_name()
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
                if parent_index_var is None:
                    raise ScorchAssertionError(
                        "Parent index var is None", code="cin_lowering_state_missing"
                    )
                if parent_level_type == LevelType.COMPRESSED:
                    # The enclosing loop over the parent's variable closes the
                    # parent's level with ``if ({R}{L-1}_pos.back() < p{R}{L-1})``
                    # (``SparseLattice.get_lattice_loops``) when the parent itself
                    # has a compressed parent, so the count that test reads must
                    # advance with every coordinate appended here.  Without it an
                    # output with three compressed levels above one drain never
                    # records its outermost coordinates (``einsum('ijkl->ijl')``,
                    # ``sss`` tensor-times-matrix).  Nothing else reads the count.
                    grandparent_index_var = (
                        result_tensor_access.get_parent_index_var(parent_index_var)
                        if level > 1
                        else None
                    )
                    parent_count_stmts: List[llir.Stmt] = (
                        [
                            llir.Increment(
                                var=llir.Var(
                                    name=f"p{result_tensor_name}{level - 1}",
                                    type=llir.DataType.INT64,
                                )
                            )
                        ]
                        if grandparent_index_var is not None
                        and result_tensor_access.level_type_of_index_var(
                            grandparent_index_var
                        )
                        == LevelType.COMPRESSED
                        else []
                    )
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
                                    result_storage=self._result_storage(
                                        result_tensor_access,
                                        (RESULT_CRD, level - 1, STORAGE_WRITE),
                                    ),
                                ),
                                *parent_count_stmts,
                            ],
                            # This is the position-boundary conditional
                            # ``result_write_pass._rewrite_if_statement``
                            # recognizes and REPLACES rather than descending
                            # into, so its own marker covers only the
                            # ``{R}{L}_pos`` read in its condition.
                            result_storage=self._result_storage(
                                result_tensor_access,
                                (RESULT_POS, level, STORAGE_READ),
                            ),
                        )
                    )
            # Assemble pos array for this compressed level:
            # - Dense parent: A1_pos.push_back(A1_crd.size())
            # - Compressed parent: A1_pos[A0_crd.size()] = A1_crd.size()
            assembled_pos_array = False
            if level > 0:
                if parent_index_var is None:
                    raise ScorchAssertionError(
                        "Parent index var is None", code="cin_lowering_state_missing"
                    )
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
                            # One statement, three references, two directions:
                            # it writes this level's position array and reads two
                            # coordinate arrays' sizes to do it.
                            result_storage=self._result_storage(
                                result_tensor_access,
                                (RESULT_POS, level, STORAGE_WRITE),
                                (RESULT_CRD, level - 1, STORAGE_READ),
                                (RESULT_CRD, level, STORAGE_READ),
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
                        # ``{R}{L}_pos_index`` is a scalar index into the
                        # position array, not the array, so it is not one of the
                        # three storage vectors and the marker does not name it.
                        result_storage=self._result_storage(
                            result_tensor_access,
                            (RESULT_POS, level, STORAGE_WRITE),
                            (RESULT_CRD, level, STORAGE_READ),
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

        return assembly_stmts

    def _prepare_scheduled_cin(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse: bool,
        ownership_transferred: bool,
    ) -> IndexStmt:
        scheduled_input = type(stmt) is ScheduledCIN
        if not scheduled_input and not _is_index_stmt_instance(stmt):
            raise ScorchTypeError(
                "CIN lowering expects an IndexStmt or exact ScheduledCIN",
                code="internal_argument_type_invalid",
            )
        plain_cin_input = not scheduled_input
        if recurse or self._active_lowerings > 0:
            if scheduled_input:
                raise CompilerInvariantError(
                    "stage=CIN lowering: ScheduledCIN is valid only at the "
                    "outer scheduling boundary",
                    code="scheduled_cin_misplaced",
                )
            plain_stmt = cast(IndexStmt, stmt)
            self._validate_index_stmt(plain_stmt)
            return plain_stmt

        plan = None
        if scheduled_input:
            scheduled_stmt = cast(ScheduledCIN, stmt)
            verify_scheduled_cin(scheduled_stmt)
            self.loop_plan = scheduled_stmt.verified_loop_plan
            plan = scheduled_stmt.verified_loop_plan
            prepared_stmt = scheduled_stmt.normalized_cin
            self.normalized_cin = prepared_stmt
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
        else:
            prepared_stmt = cast(IndexStmt, stmt)

        if plain_cin_input and (
            not ownership_transferred or self.compile_options.verification.verify_cin
        ):
            # This method is also the direct/public legacy-lowering boundary.
            # Production-owned paths arrive through normalization and transfer
            # ownership. Raw callers receive the strict field/cycle/depth and
            # identity checks plus the scheduler fields the private legacy
            # adapter's forward copier consumes; scheduler output, workspace
            # insertion and tiles included, satisfies them.
            _verify_legacy_cin_lowering_structure(prepared_stmt)
        self._validate_index_stmt(prepared_stmt)
        self._validate_loop_nest(prepared_stmt)
        self._validate_coordinate_hierarchies(prepared_stmt)
        self._validate_update_operators(prepared_stmt)
        verify_cin_enabled = self.compile_options.verification.verify_cin
        if plain_cin_input and verify_cin_enabled:
            # Full CIN verification is defined over normalized CIN, which has no
            # tile, and scheduler output passes it untiled only: a tiled
            # statement's accesses name the sum of its two tile loops, which no
            # loop binds, and the verifier would report it as unbound.  A
            # caller-owned tiled statement is refused with normalization's code.
            # A compiler-owned one is not verified again: the scheduler verified
            # its normalized source before tiling it, and the structural check
            # above still covers the derived statement.
            tile_splits = _tile_split_diagnostics(prepared_stmt, pass_name="verify_cin")
            if tile_splits and not ownership_transferred:
                _raise_tile_splits(tile_splits, stage="legacy CIN lowering")
            if tile_splits:
                verify_cin_enabled = False
        verify_cin_if_enabled(
            prepared_stmt,
            enabled=verify_cin_enabled,
        )
        if ownership_transferred:
            return claim_legacy_cin_working_tree(
                prepared_stmt,
                plan,
                compile_options=self.compile_options,
            )
        return legacy_cin_working_copy(
            prepared_stmt,
            plan,
            compile_options=self.compile_options,
        )

    def _lower_owned_IndexStmt(
        self, stmt: Union[IndexStmt, ScheduledCIN]
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Lower a detached compiler-local tree after transferring ownership.

        This is an internal compiler handoff, not a public validation bypass:
        the caller must have obtained the tree synchronously from normalization
        or scheduling and retained no externally visible alias.
        """

        return self._lower_index_stmt_entry(
            stmt,
            recurse=False,
            ownership_transferred=True,
        )

    def _lower_outer_body(self, stmt: IndexStmt) -> _LoweredOuterBody:
        """Lower the already-validated outer body and retain pass ownership data."""

        if isinstance(stmt, ForAll):
            return self.lower_ForAll(stmt)
        if isinstance(stmt, Where):
            lowered = self.lower_Where(stmt)
            statements = lowered if isinstance(lowered, list) else [lowered]
            return _LoweredOuterBody(statements)
        raise CompilerInvariantError(
            "stage=CIN lowering: outer function body must be ForAll or Where",
            code="cin_outer_body_invalid",
        )

    def lower_IndexStmt(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse=False,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower an IndexStmt to LLIR
        """

        return self._lower_index_stmt_entry(
            stmt,
            recurse=recurse,
            ownership_transferred=False,
        )

    def _lower_index_stmt_entry(
        self,
        stmt: Union[IndexStmt, ScheduledCIN],
        recurse: bool,
        ownership_transferred: bool,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Route one public or explicitly compiler-owned lowering entry."""

        if recurse and self._active_lowerings == 0:
            # ``recurse`` is an internal traversal mode used only while a
            # lowering is already on the stack.  Honoring it at an outermost
            # entry would skip the raw-entry structural boundary entirely.
            diagnostic = CINDiagnostic(
                code="invalid_recursive_entry",
                message=(
                    "recursive lowering is an internal traversal mode; "
                    "outermost callers must use the validated entry"
                ),
                path=("root",),
                stage="legacy_cin_lowering",
                pass_name="prepare_stmt",
            )
            raise VerificationError(
                "stage=legacy CIN lowering: recursive entry requires an "
                "active lowering",
                diagnostics=(diagnostic,),
            )
        compilation_context = self._compilation_context
        if compilation_context is None or recurse or self._active_lowerings > 0:
            return self._lower_index_stmt_untimed(
                stmt,
                recurse,
                ownership_transferred,
            )
        adapter_token = compilation_context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=self.compile_options,
        )
        try:
            prepared_stmt = self._prepare_scheduled_cin(
                stmt,
                recurse,
                ownership_transferred=ownership_transferred,
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
            self._reject_reused_outermost_entry(recurse)
            if not recurse and self._active_lowerings == 0:
                self._outermost_lowering_started = True
            self._active_lowerings += 1
            try:
                lowered = self._lower_prepared_index_stmt(prepared_stmt, recurse)
            finally:
                self._active_lowerings -= 1
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
        self._reject_reused_outermost_entry(recurse)
        if not recurse and self._active_lowerings == 0:
            self._outermost_lowering_started = True
        self._active_lowerings += 1
        try:
            lowered = self._lower_prepared_index_stmt(stmt, recurse)
        finally:
            self._active_lowerings -= 1
        if apply_schedule_lowering:
            return self._apply_schedule_lowering(lowered, stmt)
        return lowered

    def _reject_reused_outermost_entry(self, recurse: bool) -> None:
        """Keep the stateful legacy lowerer explicitly single-use.

        A lowering owns dozens of per-program maps and compatibility flags.
        Reusing an instance used to retain the first program as
        ``outermost_stmt`` and silently treat the second program as an
        internal recursive subtree, returning a list instead of a function
        and consulting stale state.  Resetting only the root would leave the
        rest of that state contaminated, so a second outermost entry fails
        closed after its raw CIN has crossed the ordinary validation boundary.
        """

        if (
            recurse
            or self._active_lowerings > 0
            or not self._outermost_lowering_started
        ):
            return
        diagnostic = CINDiagnostic(
            code="reused_lowerer",
            message=(
                "CINLowerer instances own one program's lowering state and "
                "cannot lower a second outermost program"
            ),
            path=("root",),
            stage="legacy_cin_lowering",
            pass_name="prepare_stmt",
        )
        raise VerificationError(
            "stage=legacy CIN lowering: CINLowerer instances are single-use",
            diagnostics=(diagnostic,),
        )

    def _lower_prepared_index_stmt(
        self,
        stmt: IndexStmt,
        recurse: bool,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Lower one already adapted compiler-owned CIN tree to current LLIR."""

        if not self.outermost_stmt:
            self.outermost_stmt = stmt
            self.has_explicit_parallel_loop = self._contains_explicit_parallel(stmt)
            self._refuse_parallel_loops_with_shared_cells(stmt)

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
        if result_contract is None:
            raise ScorchAssertionError(
                "No result tensor ABI contract", code="cin_lowering_state_missing"
            )
        # The dense blocks the body bounds are known once the body is built; until
        # then the ABI refuses only what no body could address.
        kernel_abi = _torch_cpp_kernel_abi(
            result_contract,
            rhs_tensor_vars,
            self.post_ops,
            result_levels_bounded_in_kernel=None,
            accesses=stmt.tensor_accesses,
        )
        self._refuse_post_ops_the_result_cannot_take()
        # The ABI snapshot is the fail-closed dtype/rank boundary.  Do not
        # index the raw dtype mapping before it has validated every input.
        self._value_array_ctypes = {
            f"{tensor.name}_val": dtype_to_c_datatype(tensor.dtype).value
            for tensor in rhs_tensor_vars
        }
        self._bound_position_arrays = kernel_abi.bound_position_arrays()

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
            # C7: the dynamic reserve hint owns a typed capped checked-product
            # call.  The std::min template argument is a typed DataType, the
            # native product's diagnostic operands are semantic string/bool
            # literals, and the shape vector reference reuses the prologue's
            # declared type.
            result_tensor_level_sizes.append(
                llir.VarInit(
                    var=llir.Var(name=reserve_hint_var, type=llir.DataType.INT64),
                    value=llir.FunctionCall(
                        name="std::min",
                        template_args=(llir.DataType.INT64,),
                        args=[
                            llir.FunctionCall(
                                name="scorch_native::checked_product",
                                args=[
                                    llir.Var(
                                        name="result_shape",
                                        type=llir.DataType.STD_VECTOR_INT,
                                    ),
                                    llir.Literal("evaluate", llir.DataType.STRING),
                                    llir.Literal("result_shape", llir.DataType.STRING),
                                    llir.Literal(True, llir.DataType.BOOL),
                                ],
                            ),
                            llir.Literal(2048, llir.DataType.INT64),
                        ],
                    ),
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
                # mode_order[physical] = logical: the physical level storing
                # logical mode i is its position in the stored order.
                index_var_level = mode_order.index(i)
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
                index_var_level = mode_order.index(i)
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
            postop_ptr_stmts = kernel_abi.emit_extra_tensor_prologue()

            # The former ``lower_IndexStmt(..., recurse=True)`` path repeated
            # this bookkeeping before descending.  Keep that observable
            # instance state stable while returning pass ownership explicitly.
            self.need_compute.extend(result_tensor_vars)
            recurse_result = self._lower_outer_body(stmt)

            # Validate public by-value arguments before any nested mode-index
            # access or data_ptr call. Validation may replace only these local
            # handles with checked int32 copies; caller-owned tensors stay intact.
            # Its result check declares the dense blocks the body just built
            # bounds itself.
            kernel_abi = kernel_abi.with_levels_bounded_in_kernel(
                self.result_levels_bounded_in_kernel
            )
            abi_validation_stmts = kernel_abi.emit_validation()

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
                    # Scalar accumulation stores one result entry per stored
                    # entry of the driving input, so that input's value tensor
                    # has the exact output cardinality when its leaf is
                    # compressed or coordinate.  It is the input the loops were
                    # checked to walk (``_require_scalar_accumulation_driver``),
                    # not the first input with such a leaf: a compressed
                    # operand over the REDUCED index stores an unrelated number
                    # of entries, and sizing the result by it wrote past the
                    # arrays or left their tail unwritten.
                    sparse_values_tensor = None
                    driver = self._scalar_accum_driver
                    if driver is not None:
                        level_types = driver.get_level_types()
                        if level_types and level_types[-1] in (
                            LevelType.COMPRESSED,
                            LevelType.COORDINATE,
                        ):
                            sparse_values_tensor = f"{driver.get_name()}_values"

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

                    if self.final_result_tensor_var is None:
                        raise ScorchAssertionError(
                            "No final result tensor", code="cin_lowering_state_missing"
                        )
                    final_assembler = _result_tensor_abi_assembler(
                        self.final_result_tensor_var,
                        known_nnz_var=self._known_nnz_var,
                        exact_dense_parent_positions=exact_dense_parent_positions,
                        reserve_hint_var=reserve_hint_var,
                        value_post_ops=self._value_post_ops(),
                    )
                    if self.result_dense_level_sparse_driven:
                        body_stmts.extend(
                            final_assembler.emit_position_tail_close(
                                tuple(
                                    self.result_dense_level_extent_name(
                                        final_assembler.name, level
                                    )
                                    for level in range(final_assembler.levels)
                                )
                            )
                        )
                    body_stmts.extend(final_assembler.emit_final_assembly())

                return LLIRRewriteArtifact(body_stmts)

            # Same one-shot continuation at the same lazy barrier position;
            # instrumentation only brackets its existing execution.
            body_assembler = self._instrument_body_assembler(assemble_body)

            self._require_accumulator_carried(recurse_result.compressed_where_pass_spec)
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
            function = kernel_abi.assemble_function(body_stmts)
            kernel_abi.require_bounded_levels_checked(function)
            return function

        return []

    def _require_accumulator_carried(
        self,
        compressed_where_pass_spec: Optional[CompressedWhereOpenMPPassSpec],
    ) -> None:
        """Refuse a chained-accumulator request no transform here can honour.

        The two-phase transform is the only thing on this route that produces the
        chained structure, so with no pass configuration to hand it the request
        the emitted kernel would be the declared coordinate list and the caller
        would have no way to find out.  Refusing is the same treatment the typed
        driver gives the same request, at this route's equivalent single point.

        A request for the coordinate list needs nothing here: it is what this
        route emits when the transform does not substitute.
        """

        plan = self.loop_plan
        if plan is None or plan.accumulator != CHAINED_ACCUMULATOR_STRUCTURE:
            return
        if compressed_where_pass_spec is not None:
            return
        diagnostic = CINDiagnostic(
            code=UNSUPPORTED_ACCUMULATOR_STRUCTURE,
            message=(
                f"accumulation structure {CHAINED_ACCUMULATOR_STRUCTURE!r} is "
                "produced by the two-phase assembly transform, and no two-phase "
                "assembly owns this program's output"
            ),
            path=("plan", "accumulator"),
            stage="legacy_cin_lowering",
            pass_name="lower_IndexStmt",
        )
        raise UnsupportedFeature(
            "stage=legacy CIN lowering: "
            f"{UNSUPPORTED_ACCUMULATOR_STRUCTURE} at plan/accumulator",
            diagnostics=(diagnostic,),
        )

    @staticmethod
    def _apply_vector_loops(
        scheduled: Union[llir.Stmt, List[llir.Stmt]],
        requested: Optional[Tuple[str, ...]],
    ) -> None:
        """Keep the vector marks a plan's ``vector_loops`` names, refuse the rest.

        ``None`` keeps every mark the shared rule placed during lowering.  A
        name that matches no marked loop is a loop the rule did not find safe to
        mark, refused with a structured code rather than ignored.
        """

        body = scheduled.body if isinstance(scheduled, llir.Function) else scheduled
        statements = body if isinstance(body, list) else [body]
        unmarkable = select_vector_marks(statements, requested)
        if not unmarkable:
            return
        from .loop_plan_legality import LoopPlanDiagnostic

        reason = (
            f"{', '.join(map(repr, unmarkable))} carries no vector mark in this "
            "kernel: a mark is placed only on a loop of a sum kept in a register, "
            "not parallel or unrolled, with no early exit and no loop over a "
            "compressed level's positions in its body"
        )
        raise UnsupportedFeature(
            "stage=schedule lowering: unsupported_vector_loop at "
            f"schedule/vector_loops: {reason}",
            diagnostics=(
                LoopPlanDiagnostic(
                    code="unsupported_vector_loop",
                    message=reason,
                    path=("schedule", "vector_loops"),
                    stage="schedule lowering",
                    pass_name="select_vector_marks",
                ),
            ),
        )

    def _require_entries_of(self, requested: Optional[str]) -> None:
        """Accept a plan's ``entries_of`` only where this lowering stores them.

        Scalar accumulation into a sparse result stores one result entry for
        every stored entry of the input its loops walk
        (:meth:`_require_scalar_accumulation_driver`), an entry whose inner sum
        ran zero times included; a request naming that input is what this
        lowering already emits, so the kernel text does not change.  Any other
        request names a structure this lowering does not build and is refused
        with a structured code rather than ignored.
        """

        if requested is None:
            return
        driver = self._scalar_accum_driver
        result = self.final_result_tensor_var
        if (
            self._used_scalar_accum
            and result is not None
            and not result.is_dense()
            and driver is not None
            and driver.get_name() == requested
        ):
            return
        from .loop_plan_legality import LoopPlanDiagnostic

        reason = (
            f"the result stores the entries of input {requested!r} only under "
            "scalar accumulation whose loops walk that input's stored entries; "
            "this kernel does not accumulate that way"
        )
        raise UnsupportedFeature(
            "stage=schedule lowering: unsupported_entries_of at "
            f"schedule/entries_of: {reason}",
            diagnostics=(
                LoopPlanDiagnostic(
                    code="unsupported_entries_of",
                    message=reason,
                    path=("schedule", "entries_of"),
                    stage="schedule lowering",
                    pass_name="require_entries_of",
                ),
            ),
        )

    def _apply_schedule_lowering(
        self,
        lowered: Union[llir.Stmt, List[llir.Stmt]],
        stmt: IndexStmt,
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """Apply an existing LoopPlan after CIN lowering has completed."""

        if not isinstance(lowered, llir.Function):
            return lowered
        if self.loop_plan is None:
            self._place_unplaced_workspace_clusters(lowered)
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
            self._place_unplaced_workspace_clusters(lowered)
            from .schedule_lowerer import apply_schedule_to_llir

            scheduled = apply_schedule_to_llir(
                lowered,
                legacy_schedule,
                panel_bounds,
                relayout_plan,
                result_tile_plan,
            )
            self._apply_vector_loops(scheduled, legacy_schedule.vector_loops)
            self._require_entries_of(legacy_schedule.entries_of)
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

    def _require_scalar_accumulation_driver(
        self, result_tensor_access: TensorAccess, for_all_stmt: ForAll
    ) -> None:
        """Refuse scalar accumulation into a coordinate result that has no driver.

        The path appends one result entry per iteration of the loops over the
        result's indices and sizes the result by one input's stored entries.
        That is the result only when those loops walk exactly that input's
        entries (:func:`~scorch.compiler.cin.coordinate_result_driver`).

        Anything else was lowered all the same.  ``x[a] * B[a,b]`` into a
        coordinate ``C[a]`` with a coordinate ``B`` stored ``a`` once per stored
        entry of ``B`` and never summed over ``b``, through a counter shared by
        the threads of a parallel loop; with a compressed ``B`` the arrays were
        sized by ``B``'s entries and filled with one entry per row.  A raw
        reduction ``A[i,j]`` into a coordinate ``C[i]`` returned its arrays
        untrimmed.  Each is a valid program this path cannot lower; a schedule
        that reduces through a workspace can.
        """

        statement: IndexStmt = for_all_stmt
        while isinstance(statement, ForAll):
            statement = statement.stmt
        driver = (
            coordinate_result_driver(result_tensor_access, statement.rhs)
            if isinstance(statement, TensorAssign)
            else None
        )
        # The loops around the accumulation must be the result's, one each, in
        # its storage order.  A reduction loop outside them appends every entry
        # once per iteration of that loop, and loops out of storage order append
        # coordinates out of order.  An outer tile loop is part of the loop over
        # its parent variable.
        enclosing = [
            loop.parent if loop.has_parent else loop
            for loop in self._enclosing_forall_vars
            if not (loop.tile_size_var and loop.is_outer)
        ]
        if driver is not None and enclosing == list(
            result_tensor_access.get_sorted_index_vars()
        ):
            self._scalar_accum_driver = driver.get_tensor()
            return
        result_tensor = result_tensor_access.get_tensor()
        reason = (
            "no input has exactly the result's indices with every other input "
            "dense along them"
            if driver is None
            else "the enclosing loops are not the result's indices in its "
            "storage order"
        )
        diagnostic = CINDiagnostic(
            code="unsupported_coordinate_result_reduction",
            message=(
                "legacy CIN lowering reduces into the all-coordinate result "
                f"{result_tensor.get_name()!r} one stored entry of a driving "
                f"input at a time, and {reason}"
            ),
            path=("root",),
            entity_id=result_tensor.symbol_id,
            stage="legacy_cin_lowering",
            pass_name="scalar_accumulation",
        )
        raise UnsupportedFeature(
            "stage=legacy CIN lowering: unsupported_coordinate_result_reduction "
            f"for result {result_tensor.get_name()!r}",
            diagnostics=(diagnostic,),
        )

    @staticmethod
    def _loop_selects_cells_of(index_var: IndexVar, access: TensorAccess) -> bool:
        """Whether two iterations of a loop over ``index_var`` always write different
        cells of ``access``: the access names the variable, or names the variable a
        tile split into this loop and its partner loop (each iteration of either
        covers values of that variable no other iteration covers)."""

        if access.has_index_var(index_var):
            return True
        return index_var.has_parent and bool(access.has_index_var(index_var.parent))

    def _refuse_parallel_loops_with_shared_cells(self, stmt: IndexStmt) -> None:
        """Refuse ``stmt`` if a loop it marks parallel can write one cell from two
        iterations (``_refuse_parallel_loop_with_shared_cells``).

        Runs once, on the whole statement, before anything is lowered.  A check in
        ``lower_ForAll`` would miss inner loops: the iteration lattice lowers a
        loop's body from a rebuilt copy (``LatticePoint.get_simplified_cin``) whose
        loops carry no mark.
        """

        if isinstance(stmt, ForAll):
            if stmt.parallel is True:
                self._refuse_parallel_loop_with_shared_cells(stmt)
            self._refuse_parallel_loops_with_shared_cells(stmt.stmt)
        elif isinstance(stmt, Where):
            self._refuse_parallel_loops_with_shared_cells(stmt.producer)
            self._refuse_parallel_loops_with_shared_cells(stmt.consumer)

    def _refuse_parallel_loop_with_shared_cells(self, stmt: ForAll) -> None:
        """Refuse a loop marked parallel whose iterations can write one cell.

        ``omp parallel for`` hands the loop's iterations to workers with no
        synchronization, so every tensor the loop body writes must be written at
        cells the loop's variable selects (``_loop_selects_cells_of``).  A loop over a
        reduction variable fails this: every iteration adds into the same result
        cells, and the workers' unsynchronized updates are lost.  A workspace that a
        Where inside the loop produces is private to one iteration (declared in the
        loop body, or allocated once per worker by the parallel marking), so it is
        not checked; a loop marked inside a producer is checked against the
        workspace that producer writes, which the loop does not enclose.  The
        Schedule and LoopPlan routes refuse the same schedules with the same code
        and exception type (``loop_plan_legality.refuse_parallel_mark``).
        """

        from .loop_plan_legality import PARALLEL_REDUCTION, refuse_parallel_mark

        index_var = stmt.get_index_var()
        private: Set[SymbolId] = set()
        writes: List[TensorAccess] = []

        def collect(node: IndexStmt, in_producer: bool) -> None:
            if isinstance(node, ForAll):
                collect(node.stmt, in_producer)
            elif isinstance(node, Where):
                collect(node.producer, True)
                collect(node.consumer, in_producer)
            elif isinstance(node, TensorAssign):
                writes.append(node.lhs)
                if in_producer and node.lhs.is_workspace():
                    private.add(node.lhs.tensor_id)
            else:
                raise ScorchTypeError(
                    f"unsupported CIN statement {type(node).__name__}",
                    code="cin_node_unknown",
                )

        collect(stmt.stmt, False)
        for access in writes:
            if access.tensor_id in private:
                continue
            if self._loop_selects_cells_of(index_var, access):
                continue
            tensor = access.get_tensor()
            diagnostic = CINDiagnostic(
                code=PARALLEL_REDUCTION,
                message=(
                    f"loop {index_var.name} is marked parallel, and "
                    f"{tensor.get_name()!r} is written at cells that "
                    f"{index_var.name} does not select, so two workers can "
                    "update one cell"
                ),
                path=("root",),
                entity_id=tensor.symbol_id,
                stage="legacy_cin_lowering",
                pass_name="explicit_parallel_loop",
            )
            refuse_parallel_mark(
                PARALLEL_REDUCTION,
                f"stage=legacy CIN lowering: {PARALLEL_REDUCTION} for "
                f"{tensor.get_name()!r} (loop {index_var.name})",
                (diagnostic,),
            )

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
    def _outer_loop_is_counted(iter_lattice: IterationLattice) -> bool:
        """Whether the outermost loop is one counted ``for`` loop.

        The two-pass assembly of a compressed result splits the outermost loop's
        iterations across threads, so that loop has to be a ``for`` loop whose
        iterations are known before it runs: a dense range, or the stored
        coordinates of one operand (:meth:`_compressed_where_outer_cell` says
        which row each of those assembles).

        A merge of several operands' stored coordinates is neither.  It lowers to
        a ``while`` loop that advances one cursor per operand, preceded by one
        group of cursor declarations per operand, and the transform rejected that
        statement list (``invalid_compressed_where_root_member``):
        ``einsum('ij,ik->ij', A, B, format='ds')`` with both operands
        doubly compressed did not compile.  Such a kernel is assembled serially,
        by the statements the loop nest already carries.
        """

        if iter_lattice.dense_index_var_llir is not None:
            return True
        lattice_point = iter_lattice.get_lattice_points()[0]
        return (
            len(lattice_point.get_iterators()) <= 1
            and not lattice_point.parent_lattice_point
        )

    @staticmethod
    def _compressed_where_outer_cell(
        iter_lattice: IterationLattice, result_tensor: TensorVar
    ) -> Optional[OuterCellDomain]:
        """Which result row one iteration of the outermost loop assembles.

        The two-pass assembly keeps one slot per row of the result's dense level
        zero: a count in the first pass, an offset in the second, and one entry
        of the first compressed level's position array.  A loop over a dense
        range numbers the rows with its own variable and bound, which is what
        the pass derives when it is told nothing (``None`` here, and the
        emission of every such kernel is unchanged).

        A loop over one operand's stored coordinates does not.  Its variable is
        a position in that operand's coordinate array and its bound is how many
        coordinates the operand stores.  Numbering rows with them puts row
        ``crd[p]`` in slot ``p`` and gives the position array one entry per
        STORED row, so every row after the first row the operand does not store
        is assembled one row early and the array is too short for the level --
        ``einsum('ij,jk->ik', A, B, format='ds')`` with a doubly-compressed
        ``A`` that has an empty row.  For that loop the row is the coordinate at
        the position, and the row count is the result's own level-zero extent.
        The coordinates of one compressed segment are distinct, so two
        iterations never share a slot.

        The index is spelled as the subscript into the coordinate array rather
        than as the variable the loop body declares for it, because the second
        pass reads its offsets in the body's first statements.
        """

        if iter_lattice.dense_index_var_llir is not None:
            # The loop runs over the dense range and co-iterates the operand.
            return None
        lattice_point = iter_lattice.get_lattice_points()[0]
        iterators = lattice_point.get_iterators()
        if len(iterators) != 1 or lattice_point.parent_lattice_point:
            # A dense range, or a merge of several operands, which lowers to a
            # while loop the pass does not select.
            return None
        iterator = iterators[0]
        if iterator.level_type != LevelType.COMPRESSED:
            return None
        position = iterator.get_iterator_var_llir()
        return OuterCellDomain(
            index=llir.ArrayAccess(
                array=llir.Var(
                    name=f"{iterator.tensor_var.name}{iterator.level}_crd",
                    type=llir.DataType.PTR_INT,
                ),
                index=llir.Var(name=position.name, type=position.type),
            ),
            count=llir.Var(
                name=f"{result_tensor.get_name()}0_size",
                type=llir.DataType.INT64,
            ),
        )

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
                # Rebuilds the same call with rewritten operands, so the
                # statement's result-storage marker travels with it.  Both
                # branches here reconstruct rather than mutate, which is what
                # makes threading the field necessary: a rebuild that omits it
                # silently un-marks a result write, and this file is on the
                # legacy lowering chain that reaches ``result_write_pass``.
                stmts[i] = llir.MemberCallStmt(
                    base=CINLowerer._rewrite_expr_refs(stmt.base, replacements),
                    member=stmt.member,
                    template_args=stmt.template_args,
                    args=tuple(
                        CINLowerer._rewrite_expr_refs(arg, replacements)
                        for arg in stmt.args
                    ),
                    result_storage=stmt.result_storage,
                )
            elif isinstance(stmt, llir.FunctionCallStmt):
                rewritten_name = stmt.name
                for old, new in replacements.items():
                    rewritten_name = rewritten_name.replace(old, new)
                stmts[i] = llir.FunctionCallStmt(
                    name=rewritten_name,
                    template_args=stmt.template_args,
                    args=[
                        CINLowerer._rewrite_expr_refs(arg, replacements)
                        for arg in stmt.args
                    ],
                    result_storage=stmt.result_storage,
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

    def result_dense_level_extent_name(self, result_name: str, level: int) -> str:
        """The variable holding a dense result level's extent, as its loop names it.

        A dense loop over the level runs to its own bound variable, which is
        an operand's when an operand supplies the domain; a loop over stored
        coordinates has none, and the result's own extent variable stands.
        One spelling per level keeps every count of the level's cells -- the
        flattened position and the cell total -- in agreement with the loop.
        """

        return self.result_dense_level_bound_names.get(
            level, f"{result_name}{level}_size"
        )

    def _mark_parallel(self, stmts: List[llir.Stmt]) -> None:
        """Mark the first compatible loop parallel and record what it placed."""

        cluster = self._parallel_workspace_cluster
        if not mark_first_for_loop_parallel(stmts, cluster):
            return
        self._kernel_has_parallel_region = True
        self._unplaced_workspace_clusters = [
            unplaced
            for unplaced in self._unplaced_workspace_clusters
            if unplaced is not cluster
        ]

    def _place_unplaced_workspace_clusters(self, function: llir.Function) -> None:
        """Allocate, serially, each dense workspace no parallel loop placed.

        A dense workspace is allocated by the loop the parallel marking
        selects.  A kernel with a sparse result selects none, so its dense
        workspace was used and never declared, and the kernel did not build.
        Such a kernel runs on one worker; its workspaces are allocated once,
        before its outermost loop.

        A kernel that does have a parallel region and still left a workspace
        unplaced is refused: placed serially, that workspace would be shared
        by every worker of the region.
        """

        if not self._unplaced_workspace_clusters:
            return
        clusters = self._unplaced_workspace_clusters
        self._unplaced_workspace_clusters = []
        if self._kernel_has_parallel_region or self._outermost_loop_name is None:
            raise CompilerInvariantError(
                "stage=CIN lowering pass=workspace placement: a dense workspace "
                "was placed by no parallel loop in a kernel that has a parallel "
                "region or no outermost loop",
                code="workspace_placement_failed",
            )
        before: List[llir.Stmt] = []
        after: List[llir.Stmt] = []
        for cluster in clusters:
            cluster_before, cluster_after = serial_workspace_cluster_placement(cluster)
            before.extend(cluster_before)
            after.extend(cluster_after)
        # At function scope, around the statement that is or holds the
        # outermost loop: the owner then outlives every use, and nothing is
        # allocated per iteration of a loop a schedule wrapped around it.
        for position, candidate in enumerate(function.body):
            if (
                self._find_tagged_loop([candidate], self._outermost_loop_name)
                is not None
            ):
                function.body[position + 1 : position + 1] = after
                function.body[position:position] = before
                return
        raise CompilerInvariantError(
            "stage=CIN lowering pass=workspace placement: the outermost loop "
            "is not in the lowered function",
            code="workspace_placement_failed",
        )

    @staticmethod
    def _tag_first_loop(stmts: List[llir.Stmt], index_var: IndexVar) -> None:
        """Attach a stable logical loop name for post-lowering schedule passes."""
        for stmt in stmts:
            if isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                stmt.scorch_index_var = index_var.name
                return

    @staticmethod
    def _find_tagged_loop(
        stmts: List[llir.Stmt], loop_name: str
    ) -> Optional[Union[llir.ForLoop, llir.WhileLoop]]:
        """The loop ``_tag_first_loop`` named ``loop_name``, of either kind.

        The outermost loop is a ``while`` loop when it merges the coordinates
        of two compressed levels -- ``einsum('bij,bjk->bik')`` with ``b``
        compressed in both operands -- and ``_tag_first_loop`` tags it all the
        same.  ``_find_tagged_for_loop`` looks only for a counted loop, the
        kind a parallel schedule can mark; a placement before or after the
        loop needs the loop whatever its kind.
        """
        for stmt in stmts:
            if isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                if getattr(stmt, "scorch_index_var", None) == loop_name:
                    return stmt
                nested = CINLowerer._find_tagged_loop(stmt.body, loop_name)
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
                    nested = CINLowerer._find_tagged_loop(body, loop_name)
                    if nested is not None:
                        return nested
        return None

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
            raise ScorchValueError(
                f"Cannot find generated loop {loop_name!r} selected for parallelism",
                code="parallel_loop_not_generated",
            )
        if not loop.omp_parallel_for and not getattr(
            loop, "_use_atomic_scheduling", False
        ):
            self._mark_parallel([loop])

    # Min work per thread for the SpGEMM 2-phase path, where `work` is the true flop
    # (A_nnz*avg_B_row). Emitted by NAME (not as a literal) so the codegen flop grain
    # lives in scorch/csrc/scorch_policy.h and picks up the Phase 4b
    # per-host autotuned value; the header is prepended to every
    # generated kernel so the macro resolves. Defaults to 1500 there — larger than the
    # A_nnz default (500) because flop is ~avg_B_row bigger; validated on redwood.
    _CG_FLOP_GRAIN = "SCORCH_GRAIN_CODEGEN_SPGEMM"

    @classmethod
    def _collect_output_arrays(
        cls, stmts: List[llir.Stmt], output_arrays: Dict[str, llir.Var]
    ) -> None:
        """Collect output array roots (e.g., D_values, D0_crd) from Assign stmts.

        The first-seen assignment-target array ``Var`` is retained per name so
        preallocation statements can preserve the reference metadata available
        in this subtree.  Legacy result references may honestly carry NO_TYPE;
        their declarations are assembled outside this loop tree.
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

    @staticmethod
    def _coo_preallocation_receiver(array_var: llir.Var) -> llir.Var:
        """Clone one whole-object receiver admitted by the COO resize seam."""

        if type(array_var) is not llir.Var:
            raise ScorchTypeError(
                "COO preallocation receiver must be an exact metadata-free "
                "std::vector Var",
                code="coo_preallocation_invalid",
            )
        fields = vars(array_var)
        required_fields = {
            "name",
            "type",
            "is_ptr",
            "is_restrict",
            "tensor_access",
        }
        array_name = fields.get("name")
        array_type = fields.get("type")
        is_vector = type(array_type) is llir.DataType and array_type.value.startswith(
            "std::vector<"
        )
        if (
            not required_fields.issubset(fields)
            or type(array_name) is not str
            or not array_name.isidentifier()
            or not is_vector
            or fields["is_ptr"] is not False
            or fields["is_restrict"] is not False
            or fields["tensor_access"] is not None
        ):
            raise ScorchTypeError(
                "COO preallocation receiver must be an exact metadata-free "
                "std::vector Var",
                code="coo_preallocation_invalid",
            )
        return llir.Var(
            name=cast(str, array_name),
            type=cast(llir.DataType, array_type),
            is_ptr=False,
            is_restrict=False,
        )

    @staticmethod
    def _tensor_position_pattern(tensor_names: Iterable[str]) -> Pattern[str]:
        """Match the position variable of one of the named tensors, ``p<name><level>``;
        group 1 is the tensor's name.

        The names are the program's own, so a name of any length is recognised: einsum
        names a 27th tensor ``AA``. A pattern that guessed one capital letter left
        ``pAA1`` shared between the threads of the loop this transform makes parallel.
        """
        import re

        names = sorted(
            {re.escape(name) for name in tensor_names}, key=len, reverse=True
        )
        if not names:
            return re.compile(r"(?!)")
        return re.compile(r"^p(" + "|".join(names) + r")\d+$")

    @staticmethod
    def _is_array_of_tensor(array_name: str, tensor_name: str) -> bool:
        """Whether ``array_name`` is one of ``tensor_name``'s arrays: the tensor's name,
        then nothing, a level digit or an underscore (``D1_crd``, ``D_values``). With
        tensors ``A`` and ``AA``, ``AA1_crd`` is not one of ``A``'s."""
        if not array_name.startswith(tensor_name):
            return False
        rest = array_name[len(tensor_name) :]
        return rest == "" or rest[0].isdigit() or rest[0] == "_"

    @classmethod
    def _replace_output_pos_with_input_pos(
        cls,
        stmts: List[llir.Stmt],
        input_iter_var: str,
        tensor_position: Pattern[str],
    ) -> None:
        """Replace shared output position variable (pD1) with input iterator position
        for thread-safe parallel output. Finds inner ForLoop over pA1..pA1_end and
        replaces pD<N> references with pA1 in the loop body. ``tensor_position`` is
        ``_tensor_position_pattern`` of the program's tensors."""
        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                # Find the sparse inner loop iterating pA1
                if (
                    isinstance(stmt.init, llir.VarInit)
                    and isinstance(stmt.init.var, llir.Var)
                    and stmt.init.var.name.startswith("p")
                ):
                    inner_pos_var = stmt.init.var.name  # e.g. "pA1"
                    cls._rewrite_output_pos_vars(
                        stmt.body, inner_pos_var, tensor_position
                    )
                else:
                    cls._replace_output_pos_with_input_pos(
                        stmt.body, input_iter_var, tensor_position
                    )
            elif isinstance(stmt, llir.WhileLoop):
                cls._replace_output_pos_with_input_pos(
                    stmt.body, input_iter_var, tensor_position
                )
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    cls._replace_output_pos_with_input_pos(
                        stmt.then_body, input_iter_var, tensor_position
                    )
                if stmt.else_body:
                    cls._replace_output_pos_with_input_pos(
                        stmt.else_body, input_iter_var, tensor_position
                    )

    @classmethod
    def _rewrite_output_pos_vars(
        cls,
        stmts: List[llir.Stmt],
        input_pos_var: str,
        tensor_position: Pattern[str],
    ) -> None:
        """Replace output position variables (pD1, pD0) in Assign/Increment stmts
        with the input position variable for thread-safe writes."""
        to_remove = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, llir.Assign) and type(stmt.var) is llir.ArrayAccess:
                # Replace pD<N> in a typed output index with the input position.
                target = cast(llir.ArrayAccess, stmt.var)
                if type(target.array) is llir.Var and type(target.index) is llir.Var:
                    array = cast(llir.Var, target.array)
                    index = cast(llir.Var, target.index)
                    match = tensor_position.match(index.name)
                    if match and cls._is_array_of_tensor(array.name, match.group(1)):
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
                if tensor_position.match(stmt.var.name):
                    # Check it's an output pos var, not an input one
                    if stmt.var.name != input_pos_var:
                        to_remove.append(i)
            elif isinstance(stmt, llir.ForLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var, tensor_position)
            elif isinstance(stmt, llir.WhileLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var, tensor_position)
        for i in reversed(to_remove):
            stmts.pop(i)

    def _should_parallelize_coo_outer(self, index_var: IndexVar) -> bool:
        """Check if the outermost ForAll iterates a COORDINATE level and
        the output is all-COO, suitable for group-based parallelization."""
        if not self.final_result_tensor_var or not self.final_result_tensor_access:
            return False
        # Output must be all-coordinate (COO)
        level_types = self.final_result_tensor_var.get_level_types()
        for lt in level_types:
            if lt != LevelType.COORDINATE:
                return False
        # ``_transform_coo_loop_for_openmp`` flattens the outermost coordinate
        # level and the level below it into one loop over stored entries.  A
        # third coordinate level keeps its own loop, which then reads the
        # positions the flattening removed: ``einsum('abc,abd->abc')`` with a
        # coordinate ``A`` and a coordinate result used ``pA1`` and ``pA1_end``
        # with no declaration and did not build.  A result of three or more
        # levels runs its nested loops serially.
        return len(level_types) <= 2

    def _transform_coo_loop_for_openmp(
        self, stmts: List[llir.Stmt], tensor_names: Iterable[str]
    ) -> List[llir.Stmt]:
        """Transform the outer COO WhileLoop into a group-indexed ForLoop
        with OpenMP parallelism.

        Finds the outermost WhileLoop that iterates over COO coordinate
        levels (identified by the pA0 = pA1_end update pattern) and replaces
        it with a pre-scan + parallel for over row groups. ``tensor_names`` are
        the names of the statement's tensors: they say which variables are tensor
        positions (``_tensor_position_pattern``).
        """
        tensor_position = self._tensor_position_pattern(tensor_names)
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
            outer_end_bound: Optional[llir.Var] = None
            if isinstance(stmt.cond, llir.BinOp) and isinstance(
                stmt.cond.right, llir.Var
            ):
                outer_end_bound = stmt.cond.right

            if outer_end_bound is None:
                result.append(stmt)
                continue
            outer_end_var = outer_end_bound.name

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
            CINLowerer._replace_output_pos_with_input_pos(
                inner_body_filtered, iter_var, tensor_position
            )

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
            apply_parallel_policy(group_loop, body=group_body)

            # Build pre-allocation only on routes that emit it.  The live
            # known-NNZ route uses pointer-backed Torch storage and discards
            # this historical resize block, so it must not be validated as if
            # it were a std::vector receiver.
            def preallocation_statements() -> List[llir.Stmt]:
                bound_fields = vars(outer_end_bound)
                integral_types = {
                    llir.DataType.INT,
                    llir.DataType.UINT8,
                    llir.DataType.INT8,
                    llir.DataType.UINT16,
                    llir.DataType.INT16,
                    llir.DataType.UINT32,
                    llir.DataType.INT32,
                    llir.DataType.UINT64,
                    llir.DataType.INT64,
                    llir.DataType.SIZE_T,
                }
                if (
                    type(outer_end_bound) is not llir.Var
                    or not {
                        "name",
                        "type",
                        "is_ptr",
                        "is_restrict",
                        "tensor_access",
                    }.issubset(bound_fields)
                    or type(bound_fields["name"]) is not str
                    or not bound_fields["name"].isidentifier()
                    or type(bound_fields["type"]) is not llir.DataType
                    or bound_fields["type"] not in integral_types
                    or bound_fields["is_ptr"] is not False
                    or bound_fields["is_restrict"] is not False
                    or bound_fields["tensor_access"] is not None
                ):
                    raise ScorchTypeError(
                        "COO preallocation extent must be an exact metadata-free "
                        "integral Var",
                        code="coo_preallocation_invalid",
                    )
                return [
                    llir.MemberCallStmt(
                        base=CINLowerer._coo_preallocation_receiver(array_var),
                        member="resize",
                        args=(
                            llir.Var(
                                name=outer_end_bound.name,
                                type=outer_end_bound.type,
                                is_ptr=outer_end_bound.is_ptr,
                                is_restrict=outer_end_bound.is_restrict,
                            ),
                        ),
                    )
                    for array_var in output_arrays.values()
                ]

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
                apply_parallel_policy(flat_loop, body=flat_body)

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
                    result.extend(preallocation_statements())
                result.append(flat_loop)
                transformed = True
            else:
                result.extend(pre_scan_stmts)
                result.extend(preallocation_statements())
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
                if not self.result_tensor_var:
                    raise ScorchAssertionError(
                        "Result tensor variable not set",
                        code="cin_lowering_state_missing",
                    )

        self._enclosing_forall_vars.append(index_var)
        self._enclosing_forall_visits_all.append(
            iter_lattice.dense_index_var_llir is not None
        )
        try:
            stmts.extend(
                [
                    *iter_lattice.get_iterator_init_stmts(),
                    llir.BlankLine(),
                    *iter_lattice.get_lattice_loops(),
                ]
            )
        finally:
            self._enclosing_forall_vars.pop()
            self._enclosing_forall_visits_all.pop()
        self._tag_first_loop(stmts, index_var)
        if is_outermost_forall and any(
            isinstance(candidate, (llir.ForLoop, llir.WhileLoop)) for candidate in stmts
        ):
            # The tag ``_tag_first_loop`` just attached; it survives the
            # detaching rewrites between here and the lowered function, which
            # the loop object's identity does not.
            self._outermost_loop_name = index_var.name
        if stmt.parallel is True:
            self._mark_parallel(stmts)
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._should_parallelize_outer_forall(index_var)
        ):
            self._mark_parallel(stmts)
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._used_scalar_accum
            and self._should_parallelize_coo_outer(index_var)
        ):
            self._kernel_has_parallel_region = True
            stmts = self._transform_coo_loop_for_openmp(
                stmts,
                [
                    tensor.name
                    for tensor in (
                        stmt.get_result_tensor_vars() + stmt.get_rhs_tensor_vars()
                    )
                ],
            )
        elif (
            is_outermost_forall
            and not self.has_explicit_parallel_loop
            and self._should_parallelize_compressed_where(index_var)
            and self._outer_loop_is_counted(iter_lattice)
        ):
            self._kernel_has_parallel_region = True
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
                    "parallelization gate did not establish result/workspace metadata",
                    code="parallel_gate_unestablished",
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
                        result_assembler=_result_tensor_abi_assembler(
                            result_tensor, value_post_ops=self._value_post_ops()
                        ),
                        workspace_name=workspace_name,
                        workspace_ctype=workspace_ctype,
                        policy=CompressedWhereOpenMPPolicy(
                            omp_schedule="dynamic, 64",
                            flop_grain=self._CG_FLOP_GRAIN,
                        ),
                        compile_options=self.compile_options,
                        outer_cell=self._compressed_where_outer_cell(
                            iter_lattice, result_tensor
                        ),
                        # The plan's decision, carried rather than re-derived.
                        # ``None`` -- no plan, or a plan that records nothing --
                        # leaves this pass substituting exactly where it
                        # substitutes today, so the shipped kernel is unchanged.
                        # Reading it here rather than at the schedule-lowering
                        # seam is deliberate: that seam runs after this pass has
                        # already transformed the body.
                        accumulator=(
                            None
                            if self.loop_plan is None
                            else self.loop_plan.accumulator
                        ),
                        bound_position_arrays=self._bound_position_arrays,
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
