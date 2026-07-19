"""Typed rewriting of compressed-result writes.

This pass preserves the current compressed-output/OpenMP transformation while
moving its LLIR rewrite behind the common detached ownership boundary.  The
vector operations remain in ``FunctionCallStmt.name`` strings, while indexed
assignment targets and the fill stores produced here stay structured.

Count and fill are independent transformations.  Production applies each mode
once to the same original work body.  Applying one mode to the output of the
other remains outside the supported production contract.
Special position-boundary conditionals are a generated-shape contract for
compressed levels that have a preceding compressed parent; the surrounding
OpenMP transform does not declare ``_prev`` for the first compressed level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from . import llir
from .identity import SymbolId
from .llir_traversal import (
    LLIRPath,
    LLIRRewriteValueT,
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)

if TYPE_CHECKING:
    from .compile_options import CompileOptions

ResultWriteMode = Literal["count", "fill"]

RESULT_WRITE_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR rewrite",
    pass_name="rewrite_result_writes",
)


@dataclass(frozen=True)
class ResultWriteContext:
    """All explicit state required to rewrite one result's writes.

    ``result_id`` is the stable logical identity used to recognize value writes;
    generated storage names are used only for scoped physical coordinate and
    position arrays. Production carries the exact outer compilation snapshot.
    """

    result_name: str
    result_id: SymbolId
    compressed_levels: Tuple[int, ...]
    mode: ResultWriteMode
    value_pointer_type: llir.DataType
    traversal: LLIRTraversalContext = RESULT_WRITE_TRAVERSAL_CONTEXT
    compile_options: Optional["CompileOptions"] = None


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is ResultWriteContext:
        traversal = cast(ResultWriteContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return RESULT_WRITE_TRAVERSAL_CONTEXT


def _raise_result_write_error(
    context: object,
    *,
    code: str,
    message: str,
    path: LLIRPath,
    value: object,
) -> NoReturn:
    traversal = _diagnostic_context(context)
    raise LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code=code,
            message=message,
            path=path,
            node_type=type(value).__name__,
            stage=traversal.stage,
            pass_name=traversal.pass_name,
        )
    )


def _validate_context(context: object) -> ResultWriteContext:
    if type(context) is not ResultWriteContext:
        _raise_result_write_error(
            context,
            code="invalid_result_write_context",
            message="expected an immutable ResultWriteContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(ResultWriteContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_result_write_error(
            RESULT_WRITE_TRAVERSAL_CONTEXT,
            code="invalid_result_write_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )

    if typed_context.compile_options is not None:
        from .compile_options import CompileOptions

        if type(typed_context.compile_options) is not CompileOptions:
            _raise_result_write_error(
                typed_context,
                code="invalid_result_write_compile_options",
                message="compile_options must be an exact CompileOptions snapshot",
                path=("context", "compile_options"),
                value=typed_context.compile_options,
            )

    if type(typed_context.result_name) is not str or not typed_context.result_name:
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_name",
            message="result_name must be a non-empty string",
            path=("context", "result_name"),
            value=typed_context.result_name,
        )
    if type(typed_context.result_id) is not SymbolId:
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_id",
            message="result_id must be an exact SymbolId",
            path=("context", "result_id"),
            value=typed_context.result_id,
        )

    levels = typed_context.compressed_levels
    if type(levels) is not tuple or not levels:
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed_levels must be a non-empty immutable tuple",
            path=("context", "compressed_levels"),
            value=levels,
        )
    if any(type(level) is not int or level < 0 for level in levels):
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed levels must be non-negative exact integers",
            path=("context", "compressed_levels"),
            value=levels,
        )
    if any(current >= following for current, following in zip(levels, levels[1:])):
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed levels must be strictly increasing and unique",
            path=("context", "compressed_levels"),
            value=levels,
        )

    if type(typed_context.mode) is not str or typed_context.mode not in (
        "count",
        "fill",
    ):
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_mode",
            message="mode must be exactly 'count' or 'fill'",
            path=("context", "mode"),
            value=typed_context.mode,
        )
    if type(
        typed_context.value_pointer_type
    ) is not llir.DataType or typed_context.value_pointer_type not in {
        llir.DataType.NO_TYPE,
        llir.DataType.PTR_INT,
        llir.DataType.PTR_INT_32,
        llir.DataType.PTR_INT_64,
        llir.DataType.PTR_FLOAT32,
        llir.DataType.PTR_FLOAT64,
        llir.DataType.PTR_TORCH_FLOAT32,
        llir.DataType.PTR_TORCH_FLOAT64,
        llir.DataType.PTR_TORCH_INT32,
        llir.DataType.PTR_TORCH_INT64,
        llir.DataType.PTR_TORCH_INT8,
        llir.DataType.PTR_TORCH_UINT8,
        llir.DataType.PTR_TORCH_TENSOR,
        llir.DataType.PTR_TENSOR,
        llir.DataType.PTR_VOID,
    }:
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_value_pointer_type",
            message="value_pointer_type must be an exact pointer DataType or NO_TYPE",
            path=("context", "value_pointer_type"),
            value=typed_context.value_pointer_type,
        )
    return typed_context


class _ResultWriteRewriter(LLIRRewriter):
    """Rewrite result assembly statements while preserving legacy regions."""

    _IDENTITY_ONLY_REGIONS = frozenset(
        {
            "before_parallel_body",
            "_hoisted_ptr_decls",
        }
    )

    def __init__(self, context: ResultWriteContext) -> None:
        super().__init__(context.traversal)
        self._result_name = context.result_name
        self._result_id = context.result_id
        self._compressed_levels = context.compressed_levels
        self._leaf = context.compressed_levels[-1]
        self._mode = context.mode
        self._value_pointer_type = context.value_pointer_type
        self._identity = LLIRRewriter(context.traversal)

    def rewrite_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> LLIRStatementSequence:
        # The legacy transform never descended into these loop-owned regions.
        # They still pass through an identity rewrite so the output is detached.
        if path and path[-1] in self._IDENTITY_ONLY_REGIONS:
            return self._identity.rewrite_statement_sequence(statements, path)
        return super().rewrite_statement_sequence(statements, path)

    def rewrite_statement_sequence_member(
        self, node: llir.Stmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        node_type = type(node)
        if node_type is llir.Assign:
            return self._rewrite_assign_statement(cast(llir.Assign, node), path)
        if node_type is llir.Increment:
            return self._rewrite_increment_statement(cast(llir.Increment, node), path)
        if node_type is llir.FunctionCallStmt:
            return self._rewrite_call_statement(cast(llir.FunctionCallStmt, node), path)
        if node_type is llir.VarInit:
            return self._rewrite_var_init_statement(cast(llir.VarInit, node), path)
        if node_type is llir.IfThenElse:
            return self._rewrite_if_statement(cast(llir.IfThenElse, node), path)
        return super().rewrite_statement_sequence_member(node, path)

    @staticmethod
    def _phase_state(prefix: str, level: int) -> llir.Var:
        """Build one fresh, exactly typed mutable count/fill state reference."""

        return llir.Var(name=f"{prefix}{level}", type=llir.DataType.INT)

    @staticmethod
    def _array_name(target: llir.AssignmentTarget) -> Optional[str]:
        if type(target) is not llir.ArrayAccess:
            return None
        array = cast(llir.ArrayAccess, target).array
        if type(array) is not llir.Var:
            return None
        return cast(llir.Var, array).name

    @staticmethod
    def _phase_index(level: int) -> llir.Add:
        return llir.Add(
            llir.Var(name=f"_base{level}", type=llir.DataType.INT64),
            _ResultWriteRewriter._phase_state("_pos", level),
        )

    @classmethod
    def _store(
        cls,
        array_name: str,
        index: llir.Expr,
        value: llir.Expr,
        *,
        array_type: llir.DataType = llir.DataType.NO_TYPE,
    ) -> llir.Assign:
        return llir.Assign(
            var=llir.ArrayAccess(
                array=llir.Var(name=array_name, type=array_type),
                index=index,
            ),
            value=value,
        )

    def _is_result_value_target(self, target: llir.AssignmentTarget) -> bool:
        if type(target) is not llir.ArrayAccess:
            return False
        metadata = cast(llir.ArrayAccess, target).tensor_access
        return bool(
            type(metadata) is llir.TensorAccessMetadata
            and metadata.tensor_id == self._result_id
            and metadata.role is llir.TensorAccessRole.RESULT_WRITE
        )

    def _rewrite_assign_statement(
        self, node: llir.Assign, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        target_name = self._array_name(node.var)

        if self._is_result_value_target(node.var):
            if self._mode == "count":
                return ()
            return (
                self._store(
                    f"{self._result_name}_values_data",
                    self._phase_index(self._leaf),
                    node.value,
                    array_type=self._value_pointer_type,
                ),
            )

        for level in self._compressed_levels:
            if target_name == f"{self._result_name}{level}_crd":
                if self._mode == "count":
                    return (llir.Increment(self._phase_state("_cnt", level)),)
                return (
                    self._store(
                        f"{self._result_name}{level}_crd_data",
                        self._phase_index(level),
                        node.value,
                        array_type=llir.DataType.PTR_INT,
                    ),
                )

        if any(
            target_name == f"{self._result_name}{level}_pos"
            for level in self._compressed_levels
        ):
            return ()
        return (node,)

    def _rewrite_increment_statement(
        self, node: llir.Increment, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        for level in self._compressed_levels:
            if node.var.name == f"p{self._result_name}{level}":
                if self._mode == "fill":
                    return (llir.Increment(self._phase_state("_pos", level)),)
                return ()
        return (node,)

    def _rewrite_call_statement(
        self, node: llir.FunctionCallStmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        for index, level in enumerate(self._compressed_levels):
            if node.name != f"{self._result_name}{level}_crd.push_back":
                continue
            if self._mode == "count":
                return (llir.Increment(self._phase_state("_cnt", level)),)

            coordinate = node.args[0] if node.args else llir.Literal(0)
            replacements = [
                self._store(
                    f"{self._result_name}{level}_crd_data",
                    self._phase_index(level),
                    coordinate,
                    array_type=llir.DataType.PTR_INT,
                ),
                llir.Increment(self._phase_state("_pos", level)),
            ]
            if index + 1 < len(self._compressed_levels):
                next_level = self._compressed_levels[index + 1]
                replacements.append(
                    self._store(
                        f"{self._result_name}{next_level}_pos_data",
                        self._phase_index(level),
                        self._phase_index(next_level),
                        array_type=llir.DataType.PTR_INT,
                    )
                )
            return replacements

        if node.name == f"{self._result_name}_values.push_back":
            if self._mode == "count":
                return ()
            value = node.args[0] if node.args else llir.Literal(0)
            return (
                self._store(
                    f"{self._result_name}_values_data",
                    self._phase_index(self._leaf),
                    value,
                    array_type=self._value_pointer_type,
                ),
            )

        if ".sort" in node.name:
            if self._mode == "fill":
                return (node,)
            return ()
        return (node,)

    def _rewrite_var_init_statement(
        self, node: llir.VarInit, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        if any(
            node.var.name == f"p{self._result_name}{level}"
            for level in self._compressed_levels
        ):
            return ()
        return (node,)

    def _special_if_level(self, node: llir.IfThenElse) -> Optional[int]:
        condition = node.cond
        if (
            type(condition) is not llir.BinOp
            or condition.op != "<"
            or type(condition.left) is not llir.FunctionCall
        ):
            return None
        for level in self._compressed_levels:
            if condition.left.name == f"{self._result_name}{level}_pos.back":
                return level
        return None

    def _rewrite_if_statement(
        self, node: llir.IfThenElse, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        level = self._special_if_level(node)
        if level is None:
            return (node,)

        level_index = self._compressed_levels.index(level)
        parent_level = (
            self._compressed_levels[level_index - 1] if level_index > 0 else None
        )
        if self._mode == "count":
            count_body: List[llir.Stmt] = []
            if parent_level is not None:
                count_body.append(
                    llir.Increment(self._phase_state("_cnt", parent_level))
                )
            count_body.append(
                llir.Assign(
                    self._phase_state("_prev", level),
                    self._phase_state("_cnt", level),
                )
            )
            return (
                llir.IfThenElse(
                    cond=self._progress_condition("_cnt", level),
                    then_body=count_body,
                ),
            )

        coordinate = self._find_serial_coordinate(node)
        fill_body: List[llir.Stmt] = []
        if parent_level is not None and coordinate is not None:
            fill_body.extend(
                [
                    self._store(
                        f"{self._result_name}{parent_level}_crd_data",
                        self._phase_index(parent_level),
                        coordinate,
                        array_type=llir.DataType.PTR_INT,
                    ),
                    llir.Increment(self._phase_state("_pos", parent_level)),
                    self._store(
                        f"{self._result_name}{level}_pos_data",
                        self._phase_index(parent_level),
                        self._phase_index(level),
                        array_type=llir.DataType.PTR_INT,
                    ),
                ]
            )
        fill_body.append(
            llir.Assign(
                self._phase_state("_prev", level),
                self._phase_state("_pos", level),
            )
        )
        return (
            llir.IfThenElse(
                cond=self._progress_condition("_pos", level),
                then_body=fill_body,
            ),
        )

    @staticmethod
    def _progress_condition(prefix: str, level: int) -> llir.BinOp:
        return llir.BinOp(
            op=">",
            left=_ResultWriteRewriter._phase_state(prefix, level),
            right=_ResultWriteRewriter._phase_state("_prev", level),
        )

    def _find_serial_coordinate(self, node: llir.IfThenElse) -> Optional[llir.Expr]:
        if not node.then_body:
            return None
        for statement in node.then_body:
            if (
                type(statement) is llir.FunctionCallStmt
                and ".push_back" in statement.name
                and statement.args
            ):
                return statement.args[0]
        return None

    def rewrite_function(self, node: llir.Function, path: LLIRPath) -> llir.Function:
        return self._identity.rewrite_function(node, path)


def rewrite_result_writes(
    value: LLIRRewriteValueT,
    context: ResultWriteContext,
) -> LLIRRewriteValueT:
    """Return a detached LLIR value with compressed-result writes rewritten.

    Valid contexts use a non-empty result name, a strictly increasing tuple of
    compressed physical levels, and exactly ``"count"`` or ``"fill"`` mode.
    A valid input with no recognized legacy result-write spelling is a detached
    no-op.  Unknown nodes and malformed typed children fail through the shared
    traversal diagnostic.

    A scalar statement root is supported only when its replacement cardinality
    is one.  Deletion and expansion require a statement-list/tuple root so this
    function can preserve the caller's root category.  Count/fill composition
    and a first-level special position-boundary conditional are outside the
    supported production contract.
    """

    checked_context = _validate_context(context)
    LLIRWalker(checked_context.traversal).walk(cast(LLIRValue, value))
    rewriter = _ResultWriteRewriter(checked_context)

    if isinstance(value, llir.Stmt):
        scalar_root: List[llir.Stmt] = [value]
        rewritten = rewriter.rewrite(scalar_root)
        if len(rewritten) != 1:
            _raise_result_write_error(
                checked_context,
                code="unsupported_scalar_result_write_root",
                message=(
                    "a scalar statement root cannot preserve its root category "
                    f"when the rewrite produces {len(rewritten)} statements"
                ),
                path=("root",),
                value=value,
            )
        return cast(LLIRRewriteValueT, rewritten[0])
    return rewriter.rewrite(value)
