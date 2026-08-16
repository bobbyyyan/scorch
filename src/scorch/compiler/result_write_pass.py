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
    FrozenSet,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Set,
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

#: The vector-append member calls this compiler emits for a result array.  Two
#: spellings, both meaning "append one entry", produced by two different
#: lowerings: the legacy CIN lowerer emits ``push_back`` and the LoopIR
#: ordered-key target emits ``emplace_back`` directly (its drain never goes
#: through the dynamic-vector rewrite that would produce one from an indexed
#: assign).  This pass was written against the first spelling only, so the second
#: passed through unrewritten -- the count pass kept appending and its counters
#: stayed zero, against declarations the surrounding transform had already
#: dropped.  Recognizing both is what makes the two-phase strategy available to
#: more than the one family whose body happens to use legacy's vocabulary.
APPEND_METHODS: Tuple[str, ...] = ("push_back", "emplace_back")

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
        llir.DataType.PTR_INT8,
        llir.DataType.PTR_UINT8,
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

    def __init__(
        self,
        context: ResultWriteContext,
        mirror_cursor_levels: FrozenSet[int],
    ) -> None:
        super().__init__(context.traversal)
        self._result_name = context.result_name
        self._result_id = context.result_id
        self._compressed_levels = context.compressed_levels
        self._leaf = context.compressed_levels[-1]
        self._mode = context.mode
        self._value_pointer_type = context.value_pointer_type
        self._context = context
        self._identity = LLIRRewriter(context.traversal)
        #: The levels whose coordinate writes are ALL appends, so ``p{R}{L}`` is
        #: a mirror of the vector's size rather than the write cursor and its
        #: bump is the append's own advance spelled again.  Scanned from the body
        #: this rewriter is about to rewrite; see :class:`_CoordinateWriteForms`.
        self._mirror_cursor_levels = mirror_cursor_levels

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
        """Recognize first, over every statement type; then dispatch a rewrite.

        The two steps are deliberately separate, and the separation is the
        point.  Before this, recognition lived INSIDE
        ``_rewrite_call_statement``, so it could only ever see the statement
        types the dispatch below already names -- which is the whole mechanism
        of review section 61.4's gap B: a ``MemberCallStmt`` on a result array
        falls to the identity path and no guard ever looks at it.  A marker on a
        statement the pass never inspects is a marker that does nothing, so the
        recognizer now runs over the full traversal and the rewrite dispatch is
        a second, narrower thing.
        """

        recognized = self._recognized_result_storage(node, path)

        node_type = type(node)
        replacement: Sequence[llir.Stmt]
        if node_type is llir.Assign:
            replacement = self._rewrite_assign_statement(cast(llir.Assign, node), path)
        elif node_type is llir.Increment:
            replacement = self._rewrite_increment_statement(
                cast(llir.Increment, node), path
            )
        elif node_type is llir.FunctionCallStmt:
            replacement = self._rewrite_call_statement(
                cast(llir.FunctionCallStmt, node), path
            )
        elif node_type is llir.VarInit:
            replacement = self._rewrite_var_init_statement(
                cast(llir.VarInit, node), path
            )
        elif node_type is llir.IfThenElse:
            replacement = self._rewrite_if_statement(cast(llir.IfThenElse, node), path)
        else:
            replacement = super().rewrite_statement_sequence_member(node, path)

        if recognized is not None and recognized.writes():
            self._require_result_write_rewritten(node, replacement, path)
        return replacement

    def _recognized_result_storage(
        self, node: llir.Stmt, path: LLIRPath
    ) -> Optional[llir.ResultStorageMetadata]:
        """This result's storage marker on one statement, or ``None``.

        Three things happen here, and only the first is recognition.

        The marker's ``tensor_id`` decides whether the statement is this
        result's business at all.  A marker naming a DIFFERENT tensor is not
        this result's and is not recognized -- section 63.3's third finding is
        what happens when a foreign marker is allowed to license this result's
        storage name.

        A recognized marker is then checked for TRUTH against the statement it
        sits on, because nothing else checks that a marker is not a lie and
        section 63.3's third finding is the small version of that hazard.

        Finally the name matcher is cross-checked.  It is kept rather than
        replaced, and the disagreement that means something is one-directional:
        ``_touches_result_storage`` is defined only on a ``FunctionCallStmt``'s
        callee name, while the marker is defined on five statement types in any
        position, so the marker legitimately says yes where the name matcher
        says no -- that IS gap A being closed.  The reverse is a defect: the
        name matcher seeing this result's storage where no marker names it means
        a producing lowering omitted one, which is option E's known failure
        mode.  The postcondition already catches the CONSEQUENCE, because an
        unrecognized statement is not rewritten and its references survive; what
        this adds is saying which statement and which recognizer disagreed.
        """

        marker = getattr(node, "result_storage", None)
        recognized: Optional[llir.ResultStorageMetadata] = None
        if type(marker) is llir.ResultStorageMetadata:
            typed_marker = cast(llir.ResultStorageMetadata, marker)
            if typed_marker.tensor_id == self._result_id:
                recognized = typed_marker
                self._require_truthful_marker(node, recognized, path)
                self._require_no_tensor_conflict(node, recognized, path)

        if type(node) is llir.FunctionCallStmt and self._touches_result_storage(
            cast(llir.FunctionCallStmt, node).name
        ):
            if recognized is None:
                _raise_result_write_error(
                    self._context,
                    code="unmarked_result_write_statement",
                    message=(
                        f"statement {cast(llir.FunctionCallStmt, node).name!r} names "
                        f"result {self._result_name}'s own storage in its callee "
                        "name, but carries no result-storage marker for that "
                        "result, and an unmarked result write is one this pass "
                        "cannot recognize by type.  Whoever built this statement "
                        "has to attach one.  If it came from a lowering, mark it "
                        "at the llir.Stmt constructor that builds it; if it is a "
                        "hand-built test body -- which is the likelier reader of "
                        "this message, because production's appends sit inside "
                        "the position-boundary conditional where this callee-name "
                        "match never sees them -- use the _result_marker helper "
                        "in tests/test_scorch/test_compressed_where_openmp_pass.py "
                        "or tests/test_scorch/test_llir_pass_manager.py, whose "
                        "docstrings say what it can and cannot derive"
                    ),
                    path=path,
                    value=cast(llir.FunctionCallStmt, node).name,
                )
        return recognized

    def _require_truthful_marker(
        self,
        node: llir.Stmt,
        marker: llir.ResultStorageMetadata,
        path: LLIRPath,
    ) -> None:
        """Every reference the marker claims must be findable in the statement.

        Deliberately ONE-DIRECTIONAL, and deliberately checked over the whole
        statement including any nested body.  A reference the marker claims and
        the statement does not spell is a lie and is refused.  A reference the
        statement spells and the marker does not claim is NOT refused, because
        a nested statement carries its own marker and this check would otherwise
        demand that the enclosing one restate its children's facts -- which
        would put one reference in two markers, the thing section 63.7's fifth
        row warns against.  Including nested bodies can only make this more
        permissive, never less, so it cannot refuse a truthful marker.

        Direction is checked only where the statement's own spelling decides it.
        A dotted member name does -- ``push_back`` writes, ``size`` reads -- so a
        marker claiming a write on a ``.size`` call is refused.  An argument
        position does not, which is section 3 of the design note's measurement
        and the reason the marker carries the direction at all, so there the
        marker is the only source of truth and this check does not second-guess
        it.
        """

        finder = _ResultStorageNameFinder(
            self._result_name, self._compressed_levels, self._context.traversal
        )
        finder.walk(cast(LLIRValue, node))
        for reference in marker.references:
            name = _result_storage_array_name(self._result_name, reference)
            if name not in finder.mentioned:
                _raise_result_write_error(
                    self._context,
                    code="untruthful_result_storage_marker",
                    message=(
                        f"a result-storage marker claims this statement "
                        f"references {name!r}, which the statement does not "
                        f"name anywhere; the marker describes "
                        f"{[_result_storage_array_name(self._result_name, entry) for entry in marker.references]!r} "
                        f"and the statement names {sorted(finder.mentioned)!r}"
                    ),
                    path=path,
                    value=name,
                )
            written = finder.written_members.get(name)
            if written is None:
                continue
            claims_write = reference.direction is llir.ResultStorageDirection.WRITE
            if claims_write is not written:
                _raise_result_write_error(
                    self._context,
                    code="untruthful_result_storage_marker",
                    message=(
                        f"a result-storage marker claims a "
                        f"{reference.direction.value} of {name!r}, but the "
                        "statement's own member spelling says otherwise"
                    ),
                    path=path,
                    value=name,
                )

    def _require_no_tensor_conflict(
        self,
        node: llir.Stmt,
        marker: llir.ResultStorageMetadata,
        path: LLIRPath,
    ) -> None:
        """The statement marker and the target's access marker must agree.

        The workspace drain's value store is the one statement that carries
        both: ``RESULT_WRITE`` access provenance on its target ``ArrayAccess``,
        saying which ELEMENT is written, and a statement marker saying which
        storage VECTOR the statement names.  They describe different facts at
        different granularity and neither is derivable from the other, so the
        pair is allowed -- but a pair naming DIFFERENT tensors is exactly the
        contradiction section 63.3's third finding refuses, and checking it here
        is what makes the second home a check rather than a hazard.
        """

        if type(node) is not llir.Assign:
            return
        target = cast(llir.Assign, node).var
        if type(target) is not llir.ArrayAccess:
            return
        access = cast(llir.ArrayAccess, target).tensor_access
        if type(access) is not llir.TensorAccessMetadata:
            return
        if access.role is not llir.TensorAccessRole.RESULT_WRITE:
            return
        if access.tensor_id == marker.tensor_id:
            return
        _raise_result_write_error(
            self._context,
            code="result_storage_marker_tensor_conflict",
            message=(
                "this statement's result-storage marker names result "
                f"{self._result_name} while its assignment target's access "
                "metadata names a different tensor; one statement cannot write "
                "two results' storage and one of the two markers is wrong"
            ),
            path=path,
            value=access.tensor_id,
        )

    def _require_result_write_rewritten(
        self,
        node: llir.Stmt,
        replacement: Sequence[llir.Stmt],
        path: LLIRPath,
    ) -> None:
        """A recognized result write must not survive the rewrite unchanged.

        This is what closes both of review section 61.4's structural
        narrownesses, and it closes them by construction rather than by
        spelling.  Nothing this pass CONSTRUCTS carries a result-storage marker
        -- ``_store``, ``_phase_state`` and ``_phase_index`` are its only
        builders and none sets the field -- so a marker naming this result with
        a write, still present in what the rewrite returned, means the rewrite
        left the statement alone.

        Gap A arrives as a ``FunctionCallStmt`` whose callee name is a free
        function, so ``_rewrite_call_statement`` falls through to ``(node,)``.
        Gap B arrives as a ``MemberCallStmt``, which no branch above dispatches,
        so it reaches the identity rewrite.  Both keep their marker and both are
        refused here, naming the statement and its path -- which is the
        diagnosis the postcondition cannot give, since it reports that a
        reference survived rather than which statement was misunderstood.
        """

        for index, statement in enumerate(replacement):
            surviving = getattr(statement, "result_storage", None)
            if type(surviving) is not llir.ResultStorageMetadata:
                continue
            typed_surviving = cast(llir.ResultStorageMetadata, surviving)
            if typed_surviving.tensor_id != self._result_id:
                continue
            if not typed_surviving.writes():
                continue
            _raise_result_write_error(
                self._context,
                code="unrewritten_result_write_statement",
                message=(
                    f"a {type(node).__name__} writes result "
                    f"{self._result_name}'s own storage in a {self._mode} phase "
                    "and this pass has no rewrite for it, so it survived "
                    "unchanged; retaining it would emit a write against a "
                    "declaration the two-phase transform has dropped.  It "
                    f"references "
                    f"{[_result_storage_array_name(self._result_name, entry) for entry in typed_surviving.references]!r}"
                ),
                path=path + (f"[{index}]",),
                value=node,
            )

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
        """Advance the fill cursor, unless the coordinate write already does.

        ``_pos{L}`` counts the coordinates written to ``{R}{L}_crd`` in this
        cell, so it must advance exactly once per coordinate.  Where the body's
        only spelling of that write is an append, the append grows the vector and
        the append's own rewrite carries the advance; this bump then maintains a
        mirror of the same size and emitting it too would move the cursor twice
        per entry, writing past the exactly-sized coordinate and value buffers.
        Everywhere else -- an indexed assignment at this level, or no coordinate
        write at all -- this bump is the advance and is rewritten to one.
        :class:`_CoordinateWriteForms` is where that is decided, from the body.

        Count mode drops it either way: a counting phase has no cursor.
        """

        for level in self._compressed_levels:
            if node.var.name == f"p{self._result_name}{level}":
                if self._mode == "fill" and level not in self._mirror_cursor_levels:
                    return (llir.Increment(self._phase_state("_pos", level)),)
                return ()
        return (node,)

    @staticmethod
    def _append_target(name: str) -> Optional[str]:
        """The array a vector-append call appends to, or ``None``."""

        for method in APPEND_METHODS:
            suffix = f".{method}"
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return None

    def _rewrite_call_statement(
        self, node: llir.FunctionCallStmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        appended = self._append_target(node.name)
        for index, level in enumerate(self._compressed_levels):
            if appended != f"{self._result_name}{level}_crd":
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

        if appended == f"{self._result_name}_values":
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
        if node.name == "scorch_vector_set" and node.args:
            # The position close, in the spelling the LoopIR targets emit
            # directly.  The indexed-assign spelling is dropped by
            # ``_rewrite_assign_statement`` below; this is the same statement
            # reaching the same fate through a different vocabulary.  Both
            # phases rebuild every position array from the exact prefix sums, so
            # a running close inside either phase is dead at best and a write to
            # a dropped declaration at worst.
            target = node.args[0]
            if type(target) is llir.Var and any(
                target.name == f"{self._result_name}{level}_pos"
                for level in self._compressed_levels
            ):
                return ()
        if self._touches_result_storage(node.name):
            _raise_result_write_error(
                self._context,
                code="unsupported_result_write_statement",
                message=(
                    f"statement {node.name!r} writes result storage in a "
                    f"{self._mode} phase and this pass does not recognize it; "
                    "an unrecognized result write would reach the C++ compiler "
                    "against declarations the two-phase transform has dropped"
                ),
                path=path,
                value=node.name,
            )
        return (node,)

    def _touches_result_storage(self, name: str) -> bool:
        """Whether a call statement names one of this result's own arrays.

        The guard is deliberately narrow: only the result's OWN position,
        coordinate and value arrays.  A call touching one of those in a count or
        fill phase either has a rewrite here or is a defect; anything else in the
        body -- operand reads, workspace calls, the drain's sort -- is none of
        this pass's business and passes through as before.
        """

        arrays = [f"{self._result_name}_values"]
        for level in self._compressed_levels:
            arrays.append(f"{self._result_name}{level}_pos")
            arrays.append(f"{self._result_name}{level}_crd")
        return any(name.startswith(f"{array}.") for array in arrays)

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


def _result_storage_array_name(
    result_name: str, reference: llir.ResultStorageReference
) -> str:
    """The generated vector name one marker reference denotes.

    One place spells the marker's three arrays as strings, and it is the same
    spelling ``_touches_result_storage`` and ``_ResultStorageResidueWalker``
    build, so the marker's vocabulary and the postcondition's cannot drift.
    """

    if reference.array is llir.ResultStorageArray.VALUES:
        return f"{result_name}_values"
    if reference.array is llir.ResultStorageArray.POS:
        return f"{result_name}{reference.level}_pos"
    return f"{result_name}{reference.level}_crd"


class _ResultStorageNameFinder(LLIRWalker):
    """Which of this result's storage vectors one statement actually names.

    Used to check that a marker tells the truth about the statement it sits on.
    It reaches the same name positions ``_ResultStorageResidueWalker`` does --
    ``Var``, ``FunctionCall``, ``FunctionCallStmt``, ``QualifiedName``,
    ``FixedStackArrayDecl`` and ``RawStmt`` -- because those are every position a
    generated storage name can be spelled in.

    ``written_members`` records the direction where the statement's own spelling
    decides it: a dotted member name says which way the reference goes, so
    ``C1_crd.push_back`` records ``True`` and ``C1_crd.size`` records ``False``.
    A bare vector name in an argument position records nothing, because the node
    shape genuinely does not say -- which is why the marker has to carry the
    direction rather than have it derived.
    """

    #: ``std::vector`` members that mutate the vector, and members that read it.
    #: A member outside both sets records no direction rather than guessing one.
    _WRITE_MEMBERS = frozenset(
        {"push_back", "emplace_back", "resize", "reserve", "clear", "insert", "assign"}
    )
    _READ_MEMBERS = frozenset(
        {"size", "back", "front", "empty", "data", "at", "begin", "end", "capacity"}
    )

    def __init__(
        self,
        result_name: str,
        compressed_levels: Sequence[int],
        traversal: LLIRTraversalContext,
    ) -> None:
        super().__init__(traversal)
        arrays = {f"{result_name}_values"}
        for level in compressed_levels:
            arrays.add(f"{result_name}{level}_pos")
            arrays.add(f"{result_name}{level}_crd")
        self._arrays = frozenset(arrays)
        self.mentioned: set = set()
        self.written_members: dict = {}

    def _record(self, value: object) -> None:
        if type(value) is not str:
            return
        if value in self._arrays:
            self.mentioned.add(value)
            return
        for name in self._arrays:
            prefix = f"{name}."
            if value.startswith(prefix):
                self.mentioned.add(name)
                member = value[len(prefix) :]
                if member in self._WRITE_MEMBERS:
                    self.written_members[name] = True
                elif member in self._READ_MEMBERS:
                    self.written_members[name] = False
                return

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        node_type = type(node)
        if node_type is llir.Var:
            self._record(cast(llir.Var, node).name)
        elif node_type is llir.FunctionCall:
            self._record(cast(llir.FunctionCall, node).name)
        elif node_type is llir.FunctionCallStmt:
            self._record(cast(llir.FunctionCallStmt, node).name)
        elif node_type is llir.QualifiedName:
            self._record(cast(llir.QualifiedName, node).name)
        elif node_type is llir.FixedStackArrayDecl:
            self._record(cast(llir.FixedStackArrayDecl, node).name)
        elif node_type is llir.RawStmt:
            code = cast(llir.RawStmt, node).code
            if type(code) is str:
                for name in self._arrays:
                    if _mentions_identifier(code, name):
                        self.mentioned.add(name)
        elif node_type in (llir.MemberCall, llir.MemberCallStmt):
            # The receiver is a child expression this hook reaches on its own;
            # the member is a bare identifier, so the DIRECTION it decides has to
            # be picked up here rather than from a dotted string.
            member = getattr(node, "member", None)
            base = getattr(node, "base", None)
            if type(member) is str and type(base) is llir.Var:
                name = cast(llir.Var, base).name
                if name in self._arrays:
                    if member in self._WRITE_MEMBERS:
                        self.written_members[name] = True
                    elif member in self._READ_MEMBERS:
                        self.written_members[name] = False


class _CoordinateWriteForms(LLIRWalker):
    """How each compressed level's coordinate write is spelled in one body.

    Two vocabularies reach this pass and they put the fill phase's cursor
    advance in different statements:

    * ``{R}{L}_crd.push_back(x)`` / ``.emplace_back(x)`` GROWS the vector, so
      the write cursor is the vector's own size and the append is the advance.
      A ``p{R}{L}++`` beside it maintains a mirror of that size, which the
      rewrite has already replaced by ``_pos{L}`` as a value, so honouring it
      as a second advance would move the cursor twice per entry.
    * ``{R}{L}_crd[p{R}{L}] = x`` grows nothing, so ``p{R}{L}`` IS the write
      cursor -- which is why the rewrite may discard the original index and
      store at ``_base{L} + _pos{L}`` -- and the following ``p{R}{L}++`` is the
      advance.

    Both can appear in ONE body for one level, and then they are two entries
    with one advance each rather than two spellings of one advance.  So the only
    case where the explicit bump is redundant is a level whose coordinate writes
    are ALL appends, which is what :meth:`mirror_only_levels` reports.
    """

    def __init__(
        self,
        result_name: str,
        compressed_levels: Sequence[int],
        traversal: LLIRTraversalContext,
    ) -> None:
        super().__init__(traversal)
        self._result_name = result_name
        self._compressed_levels = tuple(compressed_levels)
        self.appended: Set[int] = set()
        self.assigned: Set[int] = set()

    def _level_of(self, array_name: Optional[str]) -> Optional[int]:
        if array_name is None:
            return None
        for level in self._compressed_levels:
            if array_name == f"{self._result_name}{level}_crd":
                return level
        return None

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        node_type = type(node)
        if node_type is llir.FunctionCallStmt:
            appended = _ResultWriteRewriter._append_target(
                cast(llir.FunctionCallStmt, node).name
            )
            level = self._level_of(appended)
            if level is not None:
                self.appended.add(level)
        elif node_type is llir.Assign:
            level = self._level_of(
                _ResultWriteRewriter._array_name(cast(llir.Assign, node).var)
            )
            if level is not None:
                self.assigned.add(level)

    def mirror_only_levels(self) -> FrozenSet[int]:
        """Levels whose coordinate writes are all appends, so a bump is a mirror."""

        return frozenset(self.appended - self.assigned)


def _scan_coordinate_write_forms(
    value: LLIRRewriteValueT,
    context: ResultWriteContext,
) -> FrozenSet[int]:
    """The levels whose explicit cursor bump is the append's advance restated."""

    scan = _CoordinateWriteForms(
        context.result_name, context.compressed_levels, context.traversal
    )
    scan.walk(cast(LLIRValue, value))
    return scan.mirror_only_levels()


@dataclass(frozen=True)
class _ResidueFinding:
    """One surviving reference to storage the pass was supposed to remove."""

    node_type: str
    field: str
    spelling: str
    kind: str
    path: LLIRPath

    def describe(self) -> str:
        location = "/".join(str(part) for part in self.path)
        return f"{self.node_type}.{self.field} = {self.spelling!r} ({self.kind}) at {location}"


class _ResultStorageResidueWalker(LLIRWalker):
    """Find every surviving reference to this result's own bare storage.

    This is the pass's postcondition, and it asks a different question than
    ``_touches_result_storage`` does.  That guard asks, of an input statement,
    "is this a result write I know how to rewrite" -- so it can only refuse a
    shape someone anticipated, and a helper receiving a result array as an
    ARGUMENT or a ``MemberCallStmt`` on one is invisible to it.  This asks, of
    the OUTPUT, "does any reference to the removed storage survive" -- which is
    a question about names, not about statement shapes, so a spelling nobody
    anticipated is caught by construction.

    **Why "no result storage survives" is well formed rather than vacuous.**
    The fill phase legitimately emits stores INTO the result's storage;
    ``_ResultWriteRewriter._store`` is their one constructor.  What separates
    those from residue is that the two vocabularies are disjoint by spelling:

    * every reference this pass CONSTRUCTS is either an exactly-sized pointer
      suffixed ``_data`` (``_store``'s three array names) or a count/fill state
      scalar (``_cnt{L}``, ``_pos{L}``, ``_prev{L}``, ``_base{L}``, from
      ``_phase_state`` and ``_phase_index``);
    * every reference it is supposed to have REMOVED is a bare vector name
      (``{R}_values``, ``{R}{L}_pos``, ``{R}{L}_crd``) or a running cursor into
      one (``p{R}{L}``).

    So this check exempts nothing.  The pass's own emissions fall outside the
    watched set because of how they are spelled, not because they are on an
    allow-list -- which is what keeps the separation from rotting: were a later
    edit to make the pass construct a bare-name reference, this check would
    report it rather than wave it through.

    The two removed vocabularies are checked at different breadths, and
    ``_check_cursor_statement`` gives the reason: an array name is checked
    wherever it appears, a cursor only where the pass has a rewrite for it.

    **Why a surviving READ is a defect too, not an over-refusal.**  The
    surrounding transform removes the declarations of exactly these names from
    the region this output lands in -- ``compressed_where_openmp_pass``'s
    ``_should_drop_prefix_statement`` drops the ``VarDecl`` for ``{R}_values``,
    ``{R}{L}_pos`` and ``{R}{L}_crd``, the ``DirectInit`` for ``{R}{L}_pos``
    and the ``VarInit`` for ``p{R}{L}``.  In the pass's output those names do
    not exist, so a reference to one is a dangling reference whether it reads
    or writes.  ``{R}{L}_crd.size`` as an index and ``{R}{L}_pos.back`` in the
    boundary condition are both legal in the INPUT and both a defect in the
    output, and the check reports the position rather than trying to decide
    from the member name which it is.

    Subclassing the shared walker rather than descending named body fields is
    deliberate: it reaches expression positions as well as statements, reaches
    every body region the walker knows including ``_hoisted_ptr_decls`` and an
    ``IfThenElse``'s ``then_body_list``, and fails closed on an LLIR node type
    the walker does not support instead of silently not descending into it.
    """

    def __init__(self, context: ResultWriteContext) -> None:
        super().__init__(context.traversal)
        self._result_id = context.result_id
        arrays = {f"{context.result_name}_values"}
        for level in context.compressed_levels:
            arrays.add(f"{context.result_name}{level}_pos")
            arrays.add(f"{context.result_name}{level}_crd")
        self._arrays = frozenset(arrays)
        self._cursors = frozenset(
            f"p{context.result_name}{level}" for level in context.compressed_levels
        )
        self.findings: List[_ResidueFinding] = []

    def _flag(
        self,
        node: llir.Node,
        path: LLIRPath,
        field: str,
        spelling: str,
        kind: str,
    ) -> None:
        self.findings.append(
            _ResidueFinding(
                node_type=type(node).__name__,
                field=field,
                spelling=spelling,
                kind=kind,
                path=path,
            )
        )

    def _check_name(
        self, node: llir.Node, path: LLIRPath, field: str, value: object
    ) -> None:
        """One name field, in either spelling a reference to an array takes.

        Exact membership catches a plain reference.  A ``{name}.`` prefix
        catches the spelling that carries its receiver inside the name --
        ``Result1_crd.push_back`` writing, ``Result1_crd.size`` reading -- and
        this pass sees both in ``FunctionCallStmt.name``, in
        ``FunctionCall.name`` and, in at least one lowering, inside a ``Var``
        name.  The prefix must include the dot: the pass's own
        ``{R}{L}_crd_data`` pointers share the bare name's first characters and
        are not references to it.
        """

        if type(value) is not str:
            return
        if value in self._arrays:
            self._flag(node, path, field, value, "bare result array")
            return
        for name in self._arrays:
            if value.startswith(f"{name}."):
                self._flag(
                    node,
                    path,
                    field,
                    value,
                    f"reference to the removed {name}",
                )
                return

    def _check_cursor_statement(
        self, node: llir.Node, path: LLIRPath, var: object
    ) -> None:
        """A running position cursor bumped or initialized as a statement.

        The cursors are checked in these two statement positions only, and NOT
        wherever their name appears, which is the one place this check is
        narrower than it is for the arrays.  The reason is a real difference
        between the two: an array's declaration is always dropped by the
        surrounding transform, so any reference to one is dangling, whereas a
        cursor can be BOUND locally by the header of the loop that walks it
        (``for (pC1 = 0; pC1 < n; pC1++)``), and a reference to a
        locally-bound cursor is well formed.  A loop header is also a position
        the rewriter structurally cannot reach, since ``ForLoop.init`` and
        ``ForLoop.update`` are rewritten as loop fields rather than dispatched
        through ``rewrite_statement_sequence_member``.

        ``Increment`` and ``VarInit`` as sequence members are exactly the two
        forms the pass DOES have a rewrite for -- ``_rewrite_increment_statement``
        turns the bump into ``_pos{L}`` in fill mode and drops it in count, and
        ``_rewrite_var_init_statement`` drops the declaration -- so one
        surviving there means the fill body's position bookkeeping never
        advances, which is silent wrongness rather than dead code.  This is the
        same coverage the F-weak prototype had for the cursors, kept
        deliberately rather than widened.
        """

        if path and path[-1] in ("init", "update"):
            return
        name = getattr(var, "name", None)
        if type(name) is str and name in self._cursors:
            self._flag(node, path, "var", name, "running position cursor")

    def _check_metadata(self, node: llir.Node, path: LLIRPath) -> None:
        """The typed axis: this result's ``RESULT_WRITE`` marker surviving.

        ``_is_result_value_target`` already recognizes the workspace drain's
        value store by this marker rather than by name, so a marked reference
        surviving the rewrite is residue by the pass's own definition: in count
        mode the store should have been dropped, and in fill mode the
        replacement ``_store`` builds carries no metadata at all.
        """

        metadata = getattr(node, "tensor_access", None)
        if type(metadata) is not llir.TensorAccessMetadata:
            return
        if metadata.role is not llir.TensorAccessRole.RESULT_WRITE:
            return
        if metadata.tensor_id != self._result_id:
            return
        self._flag(
            node,
            path,
            "tensor_access",
            llir.TensorAccessRole.RESULT_WRITE.value,
            "surviving result-write marker",
        )

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        node_type = type(node)
        if node_type is llir.Var:
            self._check_name(node, path, "name", cast(llir.Var, node).name)
            self._check_metadata(node, path)
        elif node_type is llir.ArrayAccess:
            self._check_metadata(node, path)
        elif node_type is llir.FunctionCall:
            self._check_name(node, path, "name", cast(llir.FunctionCall, node).name)
        elif node_type is llir.FunctionCallStmt:
            self._check_name(node, path, "name", cast(llir.FunctionCallStmt, node).name)
        elif node_type is llir.QualifiedName:
            qualified = cast(llir.QualifiedName, node)
            self._check_name(node, path, "name", qualified.name)
            self._check_name(node, path, "namespace", qualified.namespace)
        elif node_type is llir.FixedStackArrayDecl:
            self._check_name(
                node, path, "name", cast(llir.FixedStackArrayDecl, node).name
            )
        elif node_type is llir.Increment:
            self._check_cursor_statement(node, path, cast(llir.Increment, node).var)
        elif node_type is llir.VarInit:
            self._check_cursor_statement(node, path, cast(llir.VarInit, node).var)
        elif node_type is llir.RawStmt:
            code = cast(llir.RawStmt, node).code
            if type(code) is str:
                for name in self._arrays:
                    if _mentions_identifier(code, name):
                        self._flag(node, path, "code", name, "verbatim C++ mention")
        # ``Comment`` carries prose, and a comment naming an array writes
        # nothing; refusing one would be a formatting rule wearing a
        # correctness check's clothes.  ``Function.name`` is the enclosing
        # definition's own name.  ``MemberAccess``/``MemberCall``/
        # ``MemberCallStmt``'s ``member`` is validated to be a bare
        # identifier, so a dotted receiver cannot hide there, and the receiver
        # itself is a child expression this hook reaches on its own -- which
        # is how a ``MemberCallStmt`` on a result array is caught even though
        # the rewriter never dispatches that statement type.


def _mentions_identifier(text: str, name: str) -> bool:
    """Whether ``text`` uses ``name`` as a whole identifier."""

    start = 0
    while True:
        found = text.find(name, start)
        if found < 0:
            return False
        before = text[found - 1] if found else ""
        after_index = found + len(name)
        after = text[after_index] if after_index < len(text) else ""
        if not (before.isalnum() or before == "_") and not (
            after.isalnum() or after == "_"
        ):
            return True
        start = found + 1


def _assert_no_surviving_result_storage(
    value: LLIRRewriteValueT,
    context: ResultWriteContext,
) -> None:
    """Refuse an output that still references the storage the pass removed.

    The refusal names every surviving reference and where it sits, because the
    postcondition's known weakness is diagnosis: it reports that something
    survived rather than which input statement was misunderstood.
    """

    walker = _ResultStorageResidueWalker(context)
    walker.walk(cast(LLIRValue, value))
    if not walker.findings:
        return
    described = "; ".join(finding.describe() for finding in walker.findings[:8])
    if len(walker.findings) > 8:
        described += f"; ... {len(walker.findings) - 8} more"
    _raise_result_write_error(
        context,
        code="residual_result_storage_reference",
        message=(
            f"{len(walker.findings)} reference(s) to result {context.result_name}'s "
            f"own storage survive the {context.mode} rewrite; the surrounding "
            "two-phase transform has dropped those declarations, so each one "
            f"is a dangling reference in the generated C++: {described}"
        ),
        path=walker.findings[0].path,
        value=walker.findings[0].spelling,
    )


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

    Every return path is checked against the pass's postcondition: no reference
    to the result's own removed storage may survive in the output.  That check
    is what makes the pass fail closed on a result write it did not recognize,
    independently of whether any recognizer was taught the spelling --
    ``_ResultStorageResidueWalker`` explains why.
    """

    checked_context = _validate_context(context)
    LLIRWalker(checked_context.traversal).walk(cast(LLIRValue, value))
    rewriter = _ResultWriteRewriter(
        checked_context,
        _scan_coordinate_write_forms(value, checked_context),
    )

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
        scalar_result = cast(LLIRRewriteValueT, rewritten[0])
        _assert_no_surviving_result_storage(scalar_result, checked_context)
        return scalar_result
    result = rewriter.rewrite(value)
    _assert_no_surviving_result_storage(result, checked_context)
    return result
