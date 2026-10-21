"""A scalar workspace's result entry is kept on a visit, not on a nonzero sum.

A ``Where`` whose workspace is a scalar and whose consumer appends it to a compressed
or coordinate result level stores one result entry for each consumer iteration in
which the producer accumulated into the workspace at least once.  That is the stored
structure the operands imply.  ``spmv`` into an ``"s"`` result, with ``A`` storing
row 3 at one column where a dense ``x`` is 0, visits row 3; row 3 is kept, holding
the explicit 0 its sum gives.  The consumer used to test ``wksp != 0``, which dropped
that row: a compaction inside the kernel that the stored structure does not imply.

The lowering declares ``bool <wksp>_visited = false`` beside the workspace and sets
it at every accumulation (``CINLowerer.lower_TensorAssign``); the consumer appends
when it is set.  :func:`place_visit_flag` then takes each set out of a loop whose
every iteration reaches it, and tests the loop's entry instead, once, before the
loop::

    wksp_visited |= A1_pos[pA0] < pA1_end;
    for (int pA1 = A1_pos[pA0]; pA1 < pA1_end; pA1++) { ...; wksp += ...; }

The entry test is the loop's own condition with its variable replaced by the initial
value: the first test the loop makes, made one statement earlier, so the flag is set
exactly when the loop runs at least once.  A set that depends on a merge match (an
intersection, ``if (j_A == j && j_x == j) { wksp += ...; }``) stays in the matched
branch, the only place that knows the producer reached the accumulation.  So the
kernel text shows which form a kernel got: ``|=`` before a loop, or ``= true``
inside one.
"""

from typing import Any, Callable, List, Optional, Set, cast

from . import llir
from .llir import AssignOp
from .llir_traversal import LLIRPath, LLIRTraversalContext, LLIRValue, LLIRWalker

VISIT_FLAG_CONTEXT = LLIRTraversalContext(
    stage="CIN lowering",
    pass_name="scalar_workspace_visit_flag",
)

# The comparisons a counted loop's condition is written with.  The entry test reuses
# the operator, so any of them gives the loop's first test.
_ORDER_TESTS = frozenset({"<", "<=", ">", ">=", "!="})

# Expression nodes that only read: an entry test built from them, evaluated one
# statement before the loop's own first test, gives the same result.
_READ_ONLY_EXPRESSIONS = (
    llir.Var,
    llir.Literal,
    llir.ArrayAccess,
    llir.BinOp,
    llir.Add,
    llir.Mul,
    llir.Cast,
    llir.MemberAccess,
    llir.QualifiedName,
    llir.Select,
)

# Statements that can end an iteration before a later statement of the same body
# runs.  A raw statement is opaque, so it counts as one.
_EARLY_EXITS = (llir.Break, llir.Continue, llir.Return, llir.RawStmt)

# Statements that hold statement lists of their own, each under ``body``.
_BODY_HOLDERS = (llir.ForLoop, llir.WhileLoop, llir.ForLoopAuto, llir.LambdaDef)


def visit_flag(workspace_name: str) -> llir.Var:
    """The flag that records whether the producer reached the workspace's
    accumulation in the current consumer iteration."""

    return llir.Var(name=f"{workspace_name}_visited", type=llir.DataType.BOOL)


def visit_flag_set(flag: llir.Var) -> llir.Assign:
    """``<wksp>_visited = true``, emitted beside each accumulation."""

    return llir.Assign(var=flag, value=llir.Literal(True))


def _is_visit_flag_set(statement: Any, name: str) -> bool:
    return (
        type(statement) is llir.Assign
        and type(statement.var) is llir.Var
        and statement.var.name == name
        and statement.op is AssignOp.ASSIGN
        and statement.value == llir.Literal(True)
    )


def _flat(statements: List[Any]) -> List[Any]:
    """A body's own statements, with nested plain statement lists opened."""

    flat: List[Any] = []
    for statement in statements:
        if type(statement) is list or type(statement) is tuple:
            flat.extend(_flat(list(statement)))
        else:
            flat.append(statement)
    return flat


class _ExpressionReads(LLIRWalker):
    """The names an expression reads, with ``pure`` False if it holds anything that
    is not a plain read (a call, an increment)."""

    def __init__(self) -> None:
        super().__init__(VISIT_FLAG_CONTEXT)
        self.pure = True
        self.names: Set[str] = set()

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        if type(node) is llir.UnaryOp:
            if node.op in ("++", "--"):
                self.pure = False
        elif type(node) not in _READ_ONLY_EXPRESSIONS:
            self.pure = False
        elif type(node) is llir.Var:
            self.names.add(node.name)


class _EarlyExit(LLIRWalker):
    """Whether a body holds a statement that can end an iteration early."""

    def __init__(self) -> None:
        super().__init__(VISIT_FLAG_CONTEXT)
        self.found = False

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        if isinstance(node, _EARLY_EXITS):
            self.found = True


def _reads(expression: llir.Expr) -> Optional[Set[str]]:
    walker = _ExpressionReads()
    walker.walk(expression)
    return walker.names if walker.pure else None


def _entry_test(loop: llir.ForLoop, name: str) -> Optional[llir.Expr]:
    """The loop's first test, as an expression to evaluate just before the loop, when
    every iteration reaches a set of the flag; None otherwise."""

    if (
        loop.omp_parallel_for
        or loop.before_parallel_body
        or loop.pre_parallel_body
        or loop.post_parallel_body
    ):
        return None
    if not any(_is_visit_flag_set(statement, name) for statement in _flat(loop.body)):
        return None
    init, cond = loop.init, loop.cond
    if type(init) is not llir.VarInit or init.value is None:
        return None
    if type(cond) is not llir.BinOp or cond.op not in _ORDER_TESTS:
        return None
    if type(cond.left) is not llir.Var or cond.left.name != init.var.name:
        return None
    start, bound = _reads(init.value), _reads(cond.right)
    if start is None or bound is None or init.var.name in bound:
        return None
    exits = _EarlyExit()
    exits.walk(cast(LLIRValue, loop.body))
    if exits.found:
        return None
    return llir.BinOp(op=cond.op, left=init.value, right=cond.right)


def _rewrite_children(
    statement: Any, rewrite: Callable[[List[Any]], List[Any]]
) -> None:
    """Apply ``rewrite`` to every statement list a compound statement holds."""

    if type(statement) in _BODY_HOLDERS:
        statement.body = rewrite(statement.body)
    elif type(statement) is llir.IfThenElse:
        if statement.then_body is not None:
            statement.then_body = rewrite(statement.then_body)
        if statement.else_body is not None:
            statement.else_body = rewrite(statement.else_body)
        if statement.then_body_list is not None:
            statement.then_body_list = [
                rewrite(body) for body in statement.then_body_list
            ]


def _without_sets(statements: List[Any], name: str) -> List[Any]:
    """A loop body with every set of the flag removed, at any depth: once the entry
    test has set it, each of them sets it again in an iteration the test counted."""

    kept: List[Any] = []
    for statement in statements:
        if _is_visit_flag_set(statement, name):
            continue
        if type(statement) is list or type(statement) is tuple:
            kept.append(type(statement)(_without_sets(list(statement), name)))
            continue
        _rewrite_children(statement, lambda body: _without_sets(body, name))
        kept.append(statement)
    return kept


def _place(statements: List[Any], flag: llir.Var) -> List[Any]:
    placed: List[Any] = []
    for statement in statements:
        if type(statement) is list or type(statement) is tuple:
            placed.append(type(statement)(_place(list(statement), flag)))
            continue
        if type(statement) is llir.ForLoop:
            entry = _entry_test(statement, flag.name)
            if entry is not None:
                statement.body = _without_sets(statement.body, flag.name)
                placed.append(llir.Assign(var=flag, value=entry, op=AssignOp.OR_ASSIGN))
                placed.append(statement)
                continue
        _rewrite_children(statement, lambda body: _place(body, flag))
        placed.append(statement)
    return placed


def place_visit_flag(producer: List[llir.Stmt], flag: llir.Var) -> List[llir.Stmt]:
    """The producer's statements with each set of ``flag`` that runs on every
    iteration of its loop replaced by that loop's entry test (module docstring)."""

    return _place(list(producer), flag)
