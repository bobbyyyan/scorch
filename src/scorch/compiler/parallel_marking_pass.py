"""Parallel-loop marking and workspace zero-fill placement.

This module is the single owner of the combined parallel-marking/zero-fill
family: given lowered LLIR statements and one explicit frozen
:class:`ParallelWorkspaceCluster`, it selects the first OpenMP-compatible
loop, decides between plain ``omp parallel for`` and adaptive atomic
work-stealing, places the cluster's per-worker allocation borrows and
releases on the loop's ``pre_parallel_body``/``post_parallel_body``, builds
the typed C12/C13 atomic prelude, applies the work-aware thread policy
(pragma text and typed value constructed together from the same structural
pieces), and constructs the serial before-parallel workspace pools.

Ownership boundary. The CIN lowerer resolves CIN workspace metadata into the
typed cluster statements (construction), and decides *whether* a loop family
is parallelized (gating reads CIN tensor-access structure).  This module owns
*how* one loop is configured once that decision is made.  The cluster is the
explicit schema for that hand-off: it replaces the transient per-``Where``
lowerer attributes the family previously read back via ``getattr`` defaults.

This family deliberately remains an in-stage transformation rather than an
eighth registered pipeline pass, so stage-timing records and cache identity
are unchanged.  Registration is blocked by two boundaries: the marking is
invoked from two different compiler stages (``lower_ForAll`` during CIN
lowering — including arbitrary-depth explicit ``ForAll.parallel`` requests —
and ``_apply_explicit_parallel_schedule`` during schedule lowering, after
legacy-schedule materialization), and the loop-to-cluster association is
temporal: nothing in the IR records which workspace cluster belongs to which
loop, so a single deferred pass boundary could only recover it by parsing or
re-deriving lowering decisions.
"""

from dataclasses import dataclass
import re
from typing import Callable, List, Optional, Sequence, Tuple, cast

from . import llir
from .diagnostics import CompilerInvariantError
from .iterator import (
    match_mode_position_access,
    match_mode_position_begin,
)
from .llir_traversal import (
    LLIRRewriter,
    LLIRTraversalContext,
)

_PARALLEL_POOL_SCALAR_TYPES = frozenset(
    {
        llir.DataType.FLOAT32,
        llir.DataType.FLOAT64,
        llir.DataType.INT32,
        llir.DataType.INT64,
        llir.DataType.INT8,
        llir.DataType.UINT8,
    }
)
_ATOMIC_BOUND_TYPES = frozenset(
    {
        llir.DataType.INT,
        llir.DataType.UINT8,
        llir.DataType.INT8,
        llir.DataType.UINT16,
        llir.DataType.INT16,
        llir.DataType.UINT32,
        llir.DataType.INT32,
        llir.DataType.UINT64,
        llir.DataType.INT64,
        llir.DataType.LONG,
        llir.DataType.SIZE_T,
    }
)
_PARALLEL_CLUSTER_STATEMENT_TYPES = (
    llir.VarInit,
    llir.MemberCallStmt,
)
_CPP_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "co_await",
        "co_return",
        "co_yield",
        "compl",
        "concept",
        "const",
        "const_cast",
        "consteval",
        "constexpr",
        "constinit",
        "continue",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    }
)
_MISSING_PARALLEL_FIELD = object()

_PARALLEL_POLICY_VALUE_CONTEXT = LLIRTraversalContext(
    stage="CIN lowering",
    pass_name="typed_parallel_policy_value",
)
_PARALLEL_CLUSTER_PLACEMENT_CONTEXT = LLIRTraversalContext(
    stage="CIN lowering",
    pass_name="place_parallel_workspace_cluster",
)


def _is_cpp_identifier(value: object) -> bool:
    return type(value) is str and value.isidentifier() and value not in _CPP_KEYWORDS


def _is_cpp_qualified_name(value: object) -> bool:
    return type(value) is str and all(
        _is_cpp_identifier(component) for component in value.split("::")
    )


def _cluster_instance_fields(value: object, owner: str) -> dict:
    fields = getattr(value, "__dict__", None)
    if type(fields) is not dict:
        raise TypeError(f"{owner} fields must be stored directly")
    return fields


def _validate_cluster_var(value: object, owner: str) -> None:
    if type(value) is not llir.Var:
        raise TypeError(f"{owner} must be an exact Var")
    fields = _cluster_instance_fields(value, owner)
    if (
        not _is_cpp_identifier(fields.get("name", _MISSING_PARALLEL_FIELD))
        or type(fields.get("type", _MISSING_PARALLEL_FIELD)) is not llir.DataType
        or type(fields.get("is_ptr", _MISSING_PARALLEL_FIELD)) is not bool
        or type(fields.get("is_restrict", _MISSING_PARALLEL_FIELD)) is not bool
        or fields.get("tensor_access", _MISSING_PARALLEL_FIELD) is not None
    ):
        raise TypeError(f"{owner} must be a complete metadata-free C++ identifier Var")


def _validate_cluster_expression(
    value: object,
    owner: str,
    active: Optional[set[int]] = None,
) -> None:
    """Validate the narrow expression grammar used by workspace borrows."""

    if active is None:
        active = set()
    if not isinstance(value, llir.Expr):
        raise TypeError(f"{owner} must be an LLIR expression")
    value_id = id(value)
    if value_id in active:
        raise TypeError(f"{owner} must be acyclic")
    active.add(value_id)
    try:
        if type(value) is llir.Var:
            _validate_cluster_var(value, owner)
            return
        fields = _cluster_instance_fields(value, owner)
        if type(value) in (llir.Add, llir.Mul):
            expected_op = "+" if type(value) is llir.Add else "*"
            if fields.get("op", _MISSING_PARALLEL_FIELD) != expected_op:
                raise TypeError(f"{owner}.op must remain {expected_op!r}")
            _validate_cluster_expression(
                fields.get("left", _MISSING_PARALLEL_FIELD),
                f"{owner}.left",
                active,
            )
            _validate_cluster_expression(
                fields.get("right", _MISSING_PARALLEL_FIELD),
                f"{owner}.right",
                active,
            )
            return
        if type(value) is llir.Cast:
            if (
                type(fields.get("data_type", _MISSING_PARALLEL_FIELD))
                is not llir.DataType
            ):
                raise TypeError(f"{owner}.data_type must be a DataType")
            _validate_cluster_expression(
                fields.get("expr", _MISSING_PARALLEL_FIELD),
                f"{owner}.expr",
                active,
            )
            return
        if type(value) is llir.FunctionCall:
            if not _is_cpp_qualified_name(fields.get("name", _MISSING_PARALLEL_FIELD)):
                raise TypeError(f"{owner}.name must be a qualified C++ name")
        elif type(value) is llir.MemberCall:
            if not _is_cpp_identifier(fields.get("member", _MISSING_PARALLEL_FIELD)):
                raise TypeError(f"{owner}.member must be a C++ identifier")
            _validate_cluster_expression(
                fields.get("base", _MISSING_PARALLEL_FIELD),
                f"{owner}.base",
                active,
            )
        else:
            raise TypeError(f"{owner} must use the workspace-borrow expression grammar")
        template_args = fields.get("template_args", _MISSING_PARALLEL_FIELD)
        if type(template_args) is not tuple or any(
            type(argument) is not llir.DataType for argument in template_args
        ):
            raise TypeError(f"{owner}.template_args must be a DataType tuple")
        args = fields.get("args", _MISSING_PARALLEL_FIELD)
        if type(args) is not tuple:
            raise TypeError(f"{owner}.args must be an expression tuple")
        for index, argument in enumerate(args):
            _validate_cluster_expression(argument, f"{owner}.args[{index}]", active)
    finally:
        active.remove(value_id)


def _validate_pool_spec_fields(spec: object) -> None:
    fields = getattr(spec, "__dict__", None)
    if type(fields) is not dict:
        raise TypeError("ParallelWorkspacePoolSpec fields must be stored directly")
    name = fields.get("name", _MISSING_PARALLEL_FIELD)
    if not _is_cpp_identifier(name):
        raise TypeError("ParallelWorkspacePoolSpec.name must be a C++ identifier")
    scalar_type = fields.get("scalar_type", _MISSING_PARALLEL_FIELD)
    if (
        type(scalar_type) is not llir.DataType
        or scalar_type not in _PARALLEL_POOL_SCALAR_TYPES
    ):
        raise TypeError(
            "ParallelWorkspacePoolSpec.scalar_type must be a supported scalar "
            "DataType"
        )
    extent = fields.get("extent", _MISSING_PARALLEL_FIELD)
    if not _is_cpp_identifier(extent):
        raise TypeError("ParallelWorkspacePoolSpec.extent must be a C++ identifier")


@dataclass(frozen=True)
class ParallelWorkspacePoolSpec:
    """One per-worker dense workspace pool requirement.

    ``name`` is the workspace identifier, ``scalar_type`` the pool element
    type used as the aligned-buffer template argument, and ``extent`` the
    per-worker slice length variable declared by the kernel prologue.
    """

    name: str
    scalar_type: llir.DataType
    extent: str

    def __post_init__(self) -> None:
        _validate_pool_spec_fields(self)


@dataclass(frozen=True)
class ParallelWorkspaceCluster:
    """The explicit workspace hand-off from one ``Where`` lowering.

    ``alloc`` holds typed per-worker borrow templates placed inside the
    parallel region before the work loop, ``free`` the statement templates
    placed after it, and ``pool_specs`` the serial before-parallel pool
    constructions. Every placement detaches fresh statement ownership from
    these templates. An empty cluster is the no-workspace case.
    """

    alloc: Tuple[llir.Stmt, ...] = ()
    free: Tuple[llir.Stmt, ...] = ()
    pool_specs: Tuple[ParallelWorkspacePoolSpec, ...] = ()

    def __post_init__(self) -> None:
        _validate_cluster_fields(self)


def _validate_cluster_fields(cluster: object) -> None:
    """Revalidate a cluster, including forged frozen-dataclass state."""

    if type(cluster) is not ParallelWorkspaceCluster:
        raise TypeError(
            "parallel workspace cluster must be an exact " "ParallelWorkspaceCluster"
        )
    fields = getattr(cluster, "__dict__", None)
    if type(fields) is not dict:
        raise TypeError("ParallelWorkspaceCluster fields must be stored directly")
    for field_name in ("alloc", "free"):
        statements = fields.get(field_name, _MISSING_PARALLEL_FIELD)
        if type(statements) is not tuple or any(
            type(statement) not in _PARALLEL_CLUSTER_STATEMENT_TYPES
            for statement in statements
        ):
            raise TypeError(
                f"ParallelWorkspaceCluster.{field_name} must be a tuple of "
                "VarInit or MemberCallStmt templates"
            )
        for index, statement in enumerate(statements):
            if type(statement) is llir.VarInit:
                statement_fields = _cluster_instance_fields(
                    statement,
                    f"ParallelWorkspaceCluster.{field_name}[{index}]",
                )
                if (
                    type(statement_fields.get("var", _MISSING_PARALLEL_FIELD))
                    is not llir.Var
                    or not isinstance(
                        statement_fields.get("value", _MISSING_PARALLEL_FIELD),
                        llir.Expr,
                    )
                    or type(statement_fields.get("op", _MISSING_PARALLEL_FIELD))
                    is not str
                    or statement_fields["op"] != "="
                    or statement_fields.get("cast", _MISSING_PARALLEL_FIELD)
                    is not False
                ):
                    raise TypeError(
                        f"ParallelWorkspaceCluster.{field_name}[{index}] must "
                        "be a complete plain VarInit template"
                    )
                _validate_cluster_var(
                    statement_fields["var"],
                    f"ParallelWorkspaceCluster.{field_name}[{index}].var",
                )
                _validate_cluster_expression(
                    statement_fields["value"],
                    f"ParallelWorkspaceCluster.{field_name}[{index}].value",
                )
            else:
                statement_fields = _cluster_instance_fields(
                    statement,
                    f"ParallelWorkspaceCluster.{field_name}[{index}]",
                )
                if not _is_cpp_identifier(
                    statement_fields.get("member", _MISSING_PARALLEL_FIELD)
                ):
                    raise TypeError(
                        f"ParallelWorkspaceCluster.{field_name}[{index}].member "
                        "must be a C++ identifier"
                    )
                template_args = statement_fields.get(
                    "template_args", _MISSING_PARALLEL_FIELD
                )
                if type(template_args) is not tuple or any(
                    type(argument) is not llir.DataType for argument in template_args
                ):
                    raise TypeError(
                        f"ParallelWorkspaceCluster.{field_name}[{index}]."
                        "template_args must be a DataType tuple"
                    )
                _validate_cluster_expression(
                    statement_fields.get("base", _MISSING_PARALLEL_FIELD),
                    f"ParallelWorkspaceCluster.{field_name}[{index}].base",
                )
                args = statement_fields.get("args", _MISSING_PARALLEL_FIELD)
                if type(args) is not tuple:
                    raise TypeError(
                        f"ParallelWorkspaceCluster.{field_name}[{index}].args "
                        "must be an expression tuple"
                    )
                for argument_index, argument in enumerate(args):
                    _validate_cluster_expression(
                        argument,
                        f"ParallelWorkspaceCluster.{field_name}[{index}]."
                        f"args[{argument_index}]",
                    )
            try:
                LLIRRewriter(_PARALLEL_CLUSTER_PLACEMENT_CONTEXT).rewrite(statement)
            except (
                AttributeError,
                CompilerInvariantError,
                KeyError,
                RecursionError,
                TypeError,
                ValueError,
            ) as error:
                raise TypeError(
                    f"ParallelWorkspaceCluster.{field_name}[{index}] must be "
                    "a complete acyclic LLIR statement template"
                ) from error
    pool_specs = fields.get("pool_specs", _MISSING_PARALLEL_FIELD)
    if type(pool_specs) is not tuple or any(
        type(spec) is not ParallelWorkspacePoolSpec for spec in pool_specs
    ):
        raise TypeError(
            "ParallelWorkspaceCluster.pool_specs must be a tuple of "
            "ParallelWorkspacePoolSpec values"
        )
    for spec in pool_specs:
        _validate_pool_spec_fields(spec)


def _detach_cluster_statements(
    statements: Sequence[llir.Stmt],
) -> List[llir.Stmt]:
    """Build one independently owned placement from cluster templates."""

    rewriter = LLIRRewriter(_PARALLEL_CLUSTER_PLACEMENT_CONTEXT)
    return [cast(llir.Stmt, rewriter.rewrite(statement)) for statement in statements]


EMPTY_PARALLEL_WORKSPACE_CLUSTER = ParallelWorkspaceCluster()


def is_openmp_compatible_for_loop(for_loop: llir.ForLoop) -> bool:
    """Whether one loop has the canonical shape OpenMP marking supports."""
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
        if for_loop.update.op not in (
            llir.AssignOp.ADD_ASSIGN,
            llir.AssignOp.SUB_ASSIGN,
        ):
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


def has_sparse_inner_loop(stmts: Sequence[llir.Stmt]) -> bool:
    """Check if any ForLoop in stmts (or nested) iterates over a sparse level
    (identified by init value referencing a _pos array)."""
    for stmt in stmts:
        if isinstance(stmt, llir.ForLoop):
            if (
                isinstance(stmt.init, llir.VarInit)
                and match_mode_position_begin(stmt.init.value) is not None
            ):
                return True
            if has_sparse_inner_loop(stmt.body):
                return True
    return False


def find_sparse_pos_array(body: Sequence[llir.Stmt]) -> Optional[str]:
    """Find the name of a sparse pos array (e.g. 'A1_pos') in loop body."""
    for stmt in body:
        if isinstance(stmt, llir.VarInit):
            matched = match_mode_position_access(stmt.value)
            if matched is not None:
                return matched
        if isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
            result = find_sparse_pos_array(stmt.body)
            if result:
                return result
        if isinstance(stmt, llir.RawStmt):
            m = re.search(r"(\w+_pos)\[", stmt.code)
            if m:
                return m.group(1)
    return None


def extract_loop_bound(for_loop: llir.ForLoop) -> Optional[str]:
    """Extract the upper bound variable name from a for loop condition."""
    if isinstance(for_loop.cond, llir.BinOp) and for_loop.cond.op == "<":
        right = for_loop.cond.right
        if isinstance(right, llir.Var):
            return right.name
    return None


def sparse_pos_work_expr(
    sparse_pos: Optional[str], loop_bound: Optional[str]
) -> Optional[str]:
    """Return a safe total-nnz expression for a matching dense parent."""
    if sparse_pos is None or loop_bound is None:
        return None
    match = re.fullmatch(
        r"([A-Za-z_]\w*?)(\d+)_pos",
        sparse_pos,
    )
    if match is None:
        return None
    operand, level_text = match.groups()
    level = int(level_text)
    if level == 0 or loop_bound != f"{operand}{level - 1}_size":
        return None
    return f"{sparse_pos}[{loop_bound}]"


def atomic_work_stealing_prelude(
    sparse_pos: str,
    bound: llir.Var,
) -> List[llir.Stmt]:
    """Typed C12/C13 atomic work-stealing prelude statements.

    The total-nnz initialization owns the same sparse position lookup the
    retained ``omp_num_threads`` pragma string spells, built from the same
    structural pieces (``sparse_pos_work_expr`` validated both names).
    The chunk initialization owns the adaptive clamp expression; its
    division keeps the thread-count product as an explicit right operand
    so emission preserves the required grouping.  The position array stays
    a metadata-free ``NO_TYPE`` reference because the kernel prologue owns
    its declaration, and the bound reference is a fresh typed copy of the
    loop-condition variable.
    """
    if type(sparse_pos) is not str:
        raise TypeError("atomic prelude sparse_pos must be a string")
    if type(bound) is not llir.Var:
        raise TypeError("atomic prelude bound must be an exact Var")
    bound_fields = getattr(bound, "__dict__", None)
    if type(bound_fields) is not dict:
        raise TypeError("atomic prelude bound fields must be stored directly")
    bound_name = bound_fields.get("name", _MISSING_PARALLEL_FIELD)
    bound_type = bound_fields.get("type", _MISSING_PARALLEL_FIELD)
    if (
        type(bound_name) is not str
        or type(bound_type) is not llir.DataType
        or bound_type not in _ATOMIC_BOUND_TYPES
        or bound_fields.get("is_ptr", _MISSING_PARALLEL_FIELD) is not False
        or bound_fields.get("is_restrict", _MISSING_PARALLEL_FIELD) is not False
        or bound_fields.get("tensor_access", _MISSING_PARALLEL_FIELD) is not None
    ):
        raise TypeError(
            "atomic prelude bound must be a metadata-free scalar Var with "
            "complete fields"
        )
    if sparse_pos_work_expr(sparse_pos, bound_name) is None:
        raise TypeError(
            "atomic prelude sparse_pos and bound must form a validated "
            "position-work pair"
        )
    return [
        llir.VarInit(
            var=llir.Var(name="_nnz", type=llir.DataType.INT),
            value=llir.ArrayAccess(
                array=llir.Var(name=sparse_pos, type=llir.DataType.NO_TYPE),
                index=llir.Var(name=bound_name, type=bound_type),
            ),
        ),
        llir.VarInit(
            var=llir.Var(name="_chunk", type=llir.DataType.INT),
            value=llir.FunctionCall(
                "std::max",
                [
                    llir.Literal(16, llir.DataType.INT),
                    llir.FunctionCall(
                        "std::min",
                        [
                            llir.Literal(256, llir.DataType.INT),
                            llir.BinOp(
                                "/",
                                llir.Var(name="_nnz", type=llir.DataType.INT),
                                llir.Mul(
                                    llir.FunctionCall("omp_get_num_threads"),
                                    llir.Literal(128, llir.DataType.INT),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ),
    ]


def typed_thread_policy_factory(
    loop: llir.ForLoop,
    sparse_pos: Optional[str],
) -> Optional[Callable[[], llir.Expr]]:
    """Build fresh typed copies of a loop's applied num_threads policy.

    The factory mirrors the string policy exactly from the same structural
    pieces: the work estimate is the sparse position lookup (or ``-1``),
    and the trip count is the loop bound (or the affine stride division for
    ``ADD_ASSIGN`` updates, whose stride is detached from the live update
    expression on every call).  Position arrays stay metadata-free
    ``NO_TYPE`` references because the kernel prologue owns their
    declarations.
    """
    if not (
        isinstance(loop.cond, llir.BinOp)
        and loop.cond.op == "<"
        and isinstance(loop.cond.right, llir.Var)
    ):
        return None
    bound_name = loop.cond.right.name
    bound_type = loop.cond.right.type
    update = loop.update
    stride = (
        update.value
        if isinstance(update, llir.Assign) and update.op == llir.AssignOp.ADD_ASSIGN
        else None
    )

    def build() -> llir.Expr:
        if sparse_pos is not None:
            work: llir.Expr = llir.ArrayAccess(
                array=llir.Var(name=sparse_pos, type=llir.DataType.NO_TYPE),
                index=llir.Var(name=bound_name, type=bound_type),
            )
        else:
            work = llir.Literal(-1, llir.DataType.INT)
        if stride is None:
            rows: llir.Expr = llir.Var(name=bound_name, type=bound_type)
        else:
            rewriter = LLIRRewriter(_PARALLEL_POLICY_VALUE_CONTEXT)
            rows = llir.BinOp(
                "/",
                llir.BinOp(
                    "-",
                    llir.Add(
                        llir.Var(name=bound_name, type=bound_type),
                        cast(llir.Expr, rewriter.rewrite(stride)),
                    ),
                    llir.Literal(1, llir.DataType.INT),
                ),
                cast(llir.Expr, rewriter.rewrite(stride)),
            )
        return llir.FunctionCall("scorch_nthreads", [work, rows])

    return build


def _expr_to_str(expr: object) -> str:
    """Quick-and-dirty LLIR expr to C++ string."""
    if isinstance(expr, llir.Var):
        return expr.name
    if isinstance(expr, llir.Literal):
        return str(expr.value)
    if isinstance(expr, llir.BinOp):
        return f"({_expr_to_str(expr.left)} {expr.op} {_expr_to_str(expr.right)})"
    return str(expr)


def parallel_rows_expr(for_loop: llir.ForLoop, bound: str) -> str:
    """Return the loop trip count, accounting for affine tile strides."""
    update = for_loop.update
    if isinstance(update, llir.Assign) and update.op == llir.AssignOp.ADD_ASSIGN:
        step = _expr_to_str(update.value)
        return f"(({bound} + {step} - 1) / {step})"
    return bound


def apply_parallel_policy(
    loop: llir.ForLoop,
    body: Optional[Sequence[llir.Stmt]] = None,
    chunk: bool = True,
    work_expr: Optional[str] = None,
    grain: Optional[str] = None,
) -> Optional[Callable[[], llir.Expr]]:
    """Attach a work-aware thread cap (+ adaptive schedule chunk) to a parallel
    ForLoop. codegen.py emits these as num_threads(scorch_nthreads(work,rows)) and
    schedule(dynamic, scorch_chunk(rows, work)) (helpers in scorch/csrc/header.h).

    rows = loop trip count; work = the C++ work estimate. When work_expr is
    given it is used verbatim (e.g. the true SpGEMM flop A_nnz*avg_B_row from
    the 2-phase path, where both operands are known); otherwise work = nnz
    (<pos>[<bound>]) for the first sparse pos array found in the body, else -1
    (thread cap by rows only). grain, when given, is emitted as the helpers'
    grain_default arg (the flop path passes its macro grain; A_nnz sites omit
    it and get the header's 500 default). No-op when the bound can't be
    determined. chunk=False keeps the loop's own chunk (e.g. the atomic
    work-stealing _chunk) and only applies the thread cap.

    Returns a factory building fresh typed copies of the applied
    num_threads value for downstream typed statements, or None when no
    policy was applied or a free-form work_expr/grain string leaves the
    policy without a typed value.
    """
    bound = extract_loop_bound(loop)
    if not bound:
        return None
    rows = parallel_rows_expr(loop, bound)
    sparse_work: Optional[str] = None
    pos: Optional[str] = None
    if work_expr is not None:
        work = work_expr
    else:
        search_body = body if body is not None else loop.body
        pos = find_sparse_pos_array(search_body)
        sparse_work = sparse_pos_work_expr(pos, bound)
        work = sparse_work or "-1"
    gsuf = f", {grain}" if grain is not None else ""
    loop.omp_num_threads = f"scorch_nthreads({work}, {rows}{gsuf})"
    if chunk:
        loop.omp_chunk_expr = f"scorch_chunk({rows}, {work}{gsuf})"
    if work_expr is not None or grain is not None:
        return None
    return typed_thread_policy_factory(
        loop,
        pos if sparse_work is not None else None,
    )


def attach_serial_workspace_pools(
    loop: llir.ForLoop,
    pool_specs: Sequence[ParallelWorkspacePoolSpec],
    thread_policy_factory: Optional[Callable[[], llir.Expr]] = None,
) -> None:
    """Allocate per-worker dense workspaces before entering OpenMP.

    Each spec receives a typed worker-count initialization owning a fresh
    copy of the loop's applied thread policy value (or the
    ``omp_get_max_threads()`` fallback when no policy was applied) and a
    typed RAII owner initialization calling the templated aligned-buffer
    allocator.  A policy string without a typed value would silently
    desynchronize the pragma from the worker count, so that combination
    fails closed.
    """
    if not pool_specs:
        return
    if any(type(spec) is not ParallelWorkspacePoolSpec for spec in pool_specs):
        raise TypeError(
            "attach_serial_workspace_pools requires exact "
            "ParallelWorkspacePoolSpec values"
        )
    for spec in pool_specs:
        _validate_pool_spec_fields(spec)
    if thread_policy_factory is None and loop.omp_num_threads:
        raise CompilerInvariantError(
            "the workspace-pool thread-count policy has no typed value for "
            f"applied policy {loop.omp_num_threads!r}"
        )

    before: List[llir.Stmt] = []
    for spec in pool_specs:
        before.extend(
            [
                llir.VarInit(
                    var=llir.Var(
                        name=f"{spec.name}_thread_count",
                        type=llir.DataType.INT,
                    ),
                    value=(
                        thread_policy_factory()
                        if thread_policy_factory is not None
                        else llir.FunctionCall("omp_get_max_threads")
                    ),
                ),
                llir.VarInit(
                    var=llir.Var(
                        name=f"{spec.name}_pool_owner",
                        type=llir.DataType.AUTO,
                    ),
                    value=llir.FunctionCall(
                        "scorch_make_aligned_buffer",
                        [
                            llir.FunctionCall(
                                "scorch_checked_size_product",
                                [
                                    llir.Cast(
                                        expr=llir.Var(
                                            name=(f"{spec.name}_thread_count"),
                                            type=llir.DataType.INT,
                                        ),
                                        data_type=llir.DataType.SIZE_T,
                                    ),
                                    llir.Cast(
                                        expr=llir.Var(
                                            name=spec.extent,
                                            type=llir.DataType.INT64,
                                        ),
                                        data_type=llir.DataType.SIZE_T,
                                    ),
                                ],
                            )
                        ],
                        template_args=(spec.scalar_type,),
                    ),
                ),
            ]
        )
    loop.before_parallel_body = before


def mark_first_for_loop_parallel(
    stmts: Sequence[llir.Stmt],
    cluster: ParallelWorkspaceCluster,
) -> None:
    """Configure the first OpenMP-compatible loop and place the cluster.

    The plain path marks ``omp parallel for`` and moves the cluster's
    allocation borrows and releases onto the loop's parallel-region
    boundaries.  The adaptive path replaces the marking with atomic
    work-stealing when a sparse inner loop consumes a per-worker workspace:
    the typed C12/C13 prelude, the atomic scheduling markers, and the
    work-aware thread policy are all owned here, and the serial
    before-parallel pool construction closes the placement.
    """
    _validate_cluster_fields(cluster)
    # Cluster members are templates, not live loop children.  Detach them on
    # every application so two markings (or later mutations of one marked
    # loop) cannot couple each other through shared mutable LLIR nodes.
    cluster_alloc = _detach_cluster_statements(cluster.alloc)
    cluster_free = _detach_cluster_statements(cluster.free)
    for llir_stmt in stmts:
        if isinstance(llir_stmt, llir.ForLoop) and is_openmp_compatible_for_loop(
            llir_stmt
        ):
            llir_stmt.omp_parallel_for = True
            has_sparse = has_sparse_inner_loop(llir_stmt.body)
            # Hoist per-thread workspace alloc/free outside the for loop
            # but inside the OMP parallel region.
            alloc = cluster_alloc
            free = cluster_free

            thread_policy_factory: Optional[Callable[[], llir.Expr]] = None
            if has_sparse and alloc:
                # Use adaptive atomic work-stealing: chunk scales with
                # total nnz to balance scheduling overhead vs load
                # imbalance across all matrix sizes.
                llir_stmt.omp_parallel_for = True
                llir_stmt.omp_schedule = "dynamic, 64"  # fallback
                # Find the sparse pos array to compute nnz
                sparse_pos = find_sparse_pos_array(llir_stmt.body)
                loop_bound = extract_loop_bound(llir_stmt)
                sparse_work = sparse_pos_work_expr(sparse_pos, loop_bound)
                if (
                    sparse_work
                    and loop_bound
                    and isinstance(llir_stmt.update, llir.Increment)
                ):
                    # Replace the omp for with atomic work-stealing
                    llir_stmt.omp_parallel_for = False
                    adaptive_pre = alloc + atomic_work_stealing_prelude(
                        cast(str, sparse_pos),
                        cast(llir.Var, cast(llir.BinOp, llir_stmt.cond).right),
                    )
                    # The shared atomic counter is declared before the
                    # parallel region by codegen itself, from the
                    # _atomic_counter_var marker below.
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
                    thread_policy_factory = typed_thread_policy_factory(
                        llir_stmt, sparse_pos
                    )
                else:
                    if alloc or free:
                        llir_stmt.pre_parallel_body = alloc or None
                        llir_stmt.post_parallel_body = free or None
                    thread_policy_factory = apply_parallel_policy(llir_stmt)
            else:
                if has_sparse:
                    llir_stmt.omp_schedule = "dynamic, 64"
                if alloc or free:
                    llir_stmt.pre_parallel_body = alloc or None
                    llir_stmt.post_parallel_body = free or None
                thread_policy_factory = apply_parallel_policy(llir_stmt)
            attach_serial_workspace_pools(
                llir_stmt,
                cluster.pool_specs,
                thread_policy_factory,
            )
            return
