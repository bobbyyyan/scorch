"""Exhaustive typed traversal and rebuilding for the current LLIR.

The current LLIR is predominantly mutable, while newer structural nodes may be
frozen.  :class:`LLIRRewriter` rebuilds every node and child collection, so a
rewritten value is a detached working tree even when a subclass makes no
semantic change.  Dispatch is deliberately explicit and exact-type based:
adding an LLIR node requires adding its child order here, and an unknown
subclass cannot be mistaken for a supported parent node.

Traversal is deterministic pre-order.  Scalar fields are visited in their
emission-oriented order.  In particular, a ``ForLoop`` visits its optional
before-parallel body, header, optional pre-parallel body, main body, and optional
post-parallel body.  Statement lists and tuples are distinct root values and
retain their container type when rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from . import llir
from .diagnostics import CompilerInvariantError
from .identity import AccessId, IndexId, SymbolId

LLIRPath = Tuple[str, ...]
LLIRStatementValue = Union[
    llir.Stmt,
    List["LLIRStatementValue"],
    Tuple["LLIRStatementValue", ...],
]
LLIRStatementSequence = Union[
    List[LLIRStatementValue],
    Tuple[LLIRStatementValue, ...],
]
LLIRValue = Union[llir.Expr, LLIRStatementValue]
LLIRRewriteValueT = TypeVar(
    "LLIRRewriteValueT",
    llir.Expr,
    llir.Stmt,
    List[llir.Stmt],
    Tuple[llir.Stmt, ...],
    List[LLIRStatementValue],
    Tuple[LLIRStatementValue, ...],
)


SUPPORTED_LLIR_EXPRESSION_NODE_TYPES: Tuple[Type[llir.Expr], ...] = (
    llir.GetTensorProperty,
    llir.Var,
    llir.UnaryOp,
    llir.BinOp,
    llir.Add,
    llir.Mul,
    llir.Literal,
    llir.FunctionCall,
    llir.Array,
    llir.MemberAccess,
    llir.ArrayAccess,
    llir.Cast,
    llir.Sizeof,
)

SUPPORTED_LLIR_STATEMENT_NODE_TYPES: Tuple[Type[llir.Stmt], ...] = (
    llir.Increment,
    llir.Return,
    llir.VarDecl,
    llir.VarInit,
    llir.Assign,
    llir.Allocate,
    llir.Free,
    llir.Print,
    llir.Comment,
    llir.BlankLine,
    llir.RawStmt,
    llir.Continue,
    llir.Break,
    llir.Function,
    llir.FunctionCallStmt,
    llir.ForLoop,
    llir.ForLoopAuto,
    llir.WhileLoop,
    llir.IfThenElse,
    llir.Case,
    llir.Switch,
)

SUPPORTED_LLIR_NODE_TYPES: Tuple[Type[llir.Node], ...] = (
    *SUPPORTED_LLIR_EXPRESSION_NODE_TYPES,
    *SUPPORTED_LLIR_STATEMENT_NODE_TYPES,
)


@dataclass(frozen=True)
class LLIRTraversalContext:
    """Immutable stage identity supplied to a traversal or rewrite."""

    stage: str
    pass_name: str


@dataclass(frozen=True)
class LLIRTraversalDiagnostic:
    """Structured failure emitted by the common LLIR traversal boundary."""

    code: str
    message: str
    path: LLIRPath
    node_type: str
    stage: str
    pass_name: str


class LLIRTraversalError(CompilerInvariantError):
    """An unknown or malformed value reached a typed LLIR traversal."""

    def __init__(self, diagnostic: LLIRTraversalDiagnostic) -> None:
        self.diagnostic = diagnostic
        self.diagnostics = (diagnostic,)
        location = "/".join(diagnostic.path)
        super().__init__(
            f"stage={diagnostic.stage} pass={diagnostic.pass_name}: "
            f"{diagnostic.code} at {location}: {diagnostic.message}"
        )


def _raise_traversal_error(
    context: LLIRTraversalContext,
    *,
    code: str,
    message: str,
    path: LLIRPath,
    value: object,
) -> NoReturn:
    raise LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code=code,
            message=message,
            path=path,
            node_type=type(value).__name__,
            stage=context.stage,
            pass_name=context.pass_name,
        )
    )


def _validate_tensor_access_metadata(
    metadata: object,
    context: LLIRTraversalContext,
    path: LLIRPath,
) -> None:
    if metadata is None:
        return
    if type(metadata) is not llir.TensorAccessMetadata:
        _raise_traversal_error(
            context,
            code="invalid_tensor_access_metadata",
            message="tensor_access must be TensorAccessMetadata or None",
            path=path,
            value=metadata,
        )
    typed_metadata = cast(llir.TensorAccessMetadata, metadata)
    fields = (
        ("access_id", typed_metadata.access_id, AccessId),
        ("tensor_id", typed_metadata.tensor_id, SymbolId),
        ("role", typed_metadata.role, llir.TensorAccessRole),
    )
    for field_name, value, expected_type in fields:
        if type(value) is not expected_type:
            _raise_traversal_error(
                context,
                code="invalid_tensor_access_metadata",
                message=(
                    f"TensorAccessMetadata.{field_name} must be "
                    f"{expected_type.__name__}"
                ),
                path=path + (field_name,),
                value=value,
            )
    if type(typed_metadata.index_ids) is not tuple:
        _raise_traversal_error(
            context,
            code="invalid_tensor_access_metadata",
            message="TensorAccessMetadata.index_ids must be a tuple of IndexId values",
            path=path + ("index_ids",),
            value=typed_metadata.index_ids,
        )
    for index, index_id in enumerate(typed_metadata.index_ids):
        if type(index_id) is not IndexId:
            _raise_traversal_error(
                context,
                code="invalid_tensor_access_metadata",
                message=(
                    "TensorAccessMetadata.index_ids must contain only IndexId values"
                ),
                path=path + ("index_ids", f"[{index}]"),
                value=index_id,
            )


def _validate_assign_fields(
    node: llir.Assign,
    context: LLIRTraversalContext,
    path: LLIRPath,
) -> None:
    if type(node.op) is not llir.AssignOp:
        _raise_traversal_error(
            context,
            code="invalid_assign_op",
            message="Assign.op must be an AssignOp",
            path=path + ("op",),
            value=node.op,
        )
    if type(node.cast) is not bool:
        _raise_traversal_error(
            context,
            code="invalid_assign_cast",
            message="Assign.cast must be a bool",
            path=path + ("cast",),
            value=node.cast,
        )
    if node.cast and type(node.var) is not llir.Var:
        _raise_traversal_error(
            context,
            code="invalid_assign_cast_target",
            message="Assign.cast requires an exact Var target",
            path=path + ("var",),
            value=node.var,
        )


def _validate_binary_expression_fields(
    node: llir.BinOp,
    context: LLIRTraversalContext,
    path: LLIRPath,
) -> None:
    if type(node.op) is not str or not node.op:
        _raise_traversal_error(
            context,
            code="invalid_binary_operator",
            message="binary expression operator must be a non-empty string",
            path=path + ("op",),
            value=node.op,
        )
    if type(node) is llir.Add and node.op != "+":
        _raise_traversal_error(
            context,
            code="invalid_add_operator",
            message="Add.op must remain '+'",
            path=path + ("op",),
            value=node.op,
        )
    if type(node) is llir.Mul and node.op != "*":
        _raise_traversal_error(
            context,
            code="invalid_mul_operator",
            message="Mul.op must remain '*'",
            path=path + ("op",),
            value=node.op,
        )
    for field_name, child in (("left", node.left), ("right", node.right)):
        if not isinstance(child, llir.Expr):
            _raise_traversal_error(
                context,
                code="invalid_binary_child",
                message=f"BinOp.{field_name} must be an LLIR Expr",
                path=path + (field_name,),
                value=child,
            )


def _validate_literal_fields(
    node: llir.Literal,
    context: LLIRTraversalContext,
    path: LLIRPath,
) -> None:
    if type(node.value) not in (bool, int, float, str):
        _raise_traversal_error(
            context,
            code="invalid_literal_value",
            message="Literal.value must be a bool, int, float, or string",
            path=path + ("value",),
            value=node.value,
        )
    if type(node.data_type) is not llir.DataType:
        _raise_traversal_error(
            context,
            code="invalid_literal_data_type",
            message="Literal.data_type must be a DataType",
            path=path + ("data_type",),
            value=node.data_type,
        )


def _validate_cast_fields(
    node: llir.Cast,
    context: LLIRTraversalContext,
    path: LLIRPath,
) -> None:
    if not isinstance(node.expr, llir.Expr):
        _raise_traversal_error(
            context,
            code="invalid_cast_expression",
            message="Cast.expr must be an LLIR Expr",
            path=path + ("expr",),
            value=node.expr,
        )
    if type(node.data_type) is not llir.DataType:
        _raise_traversal_error(
            context,
            code="invalid_cast_data_type",
            message="Cast.data_type must be a DataType",
            path=path + ("data_type",),
            value=node.data_type,
        )


class LLIRWalker:
    """A stateless, exhaustive walker with one typed hook per LLIR node."""

    def __init__(self, context: LLIRTraversalContext) -> None:
        self.context = context

    def walk(self, value: LLIRValue) -> None:
        """Walk a scalar node or a statement list/tuple in deterministic order."""

        self._walk_value(value, ("root",))

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        """Hook invoked before a supported node's typed visitor."""

    def leave_node(self, node: llir.Node, path: LLIRPath) -> None:
        """Hook invoked after a supported node's typed visitor."""

    def enter_statement_sequence(
        self, statements: Sequence[LLIRStatementValue], path: LLIRPath
    ) -> None:
        """Hook invoked before a supported statement list or tuple."""

    def leave_statement_sequence(
        self, statements: Sequence[LLIRStatementValue], path: LLIRPath
    ) -> None:
        """Hook invoked after a supported statement list or tuple."""

    def _walk_value(self, value: LLIRValue, path: LLIRPath) -> None:
        if isinstance(value, llir.Expr):
            self._walk_expr(value, path)
            return
        if isinstance(value, llir.Stmt):
            self._walk_stmt(value, path)
            return
        if type(value) is list or type(value) is tuple:
            self._walk_statement_sequence(cast(LLIRStatementSequence, value), path)
            return
        _raise_traversal_error(
            self.context,
            code="invalid_llir_value",
            message="expected an LLIR node or statement list/tuple",
            path=path,
            value=value,
        )

    def _walk_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> None:
        self.enter_statement_sequence(statements, path)
        for index, statement in enumerate(statements):
            item_path = path + (f"[{index}]",)
            if isinstance(statement, llir.Stmt):
                self._walk_stmt(statement, item_path)
            elif type(statement) is list or type(statement) is tuple:
                self._walk_statement_sequence(
                    cast(LLIRStatementSequence, statement), item_path
                )
            else:
                _raise_traversal_error(
                    self.context,
                    code="invalid_statement_sequence_member",
                    message=(
                        "statement sequences may contain only LLIR statements "
                        "or nested statement lists/tuples"
                    ),
                    path=item_path,
                    value=statement,
                )
        self.leave_statement_sequence(statements, path)

    def _walk_statements(self, statements: object, path: LLIRPath) -> None:
        if type(statements) is not list and type(statements) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_statement_sequence",
                message="required child must be a statement list or tuple",
                path=path,
                value=statements,
            )
        self._walk_statement_sequence(cast(LLIRStatementSequence, statements), path)

    def _walk_optional_statements(
        self,
        statements: Optional[Sequence[llir.Stmt]],
        path: LLIRPath,
    ) -> None:
        if statements is None:
            return
        self._walk_statements(statements, path)

    def _walk_for_loop_update(self, update: object, path: LLIRPath) -> None:
        if type(update) in (llir.Increment, llir.VarInit, llir.Assign):
            self._walk_stmt(cast(llir.Stmt, update), path)
            return
        if type(update) is llir.FunctionCall:
            self._walk_expr(cast(llir.FunctionCall, update), path)
            return
        _raise_traversal_error(
            self.context,
            code="invalid_for_loop_update",
            message=(
                "ForLoop.update must be Increment, VarInit, FunctionCall, or Assign"
            ),
            path=path,
            value=update,
        )

    def _walk_for_loop_init(self, init: object, path: LLIRPath) -> None:
        if type(init) in (llir.VarInit, llir.VarDecl):
            self._walk_stmt(cast(llir.Stmt, init), path)
            return
        _raise_traversal_error(
            self.context,
            code="invalid_for_loop_init",
            message="ForLoop.init must be VarInit, VarDecl, or None",
            path=path,
            value=init,
        )

    def _walk_expr_sequence(
        self, expressions: Sequence[llir.Expr], path: LLIRPath
    ) -> None:
        if type(expressions) is not list and type(expressions) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_expression_sequence",
                message="expected an expression list or tuple",
                path=path,
                value=expressions,
            )
        for index, expression in enumerate(expressions):
            item_path = path + (f"[{index}]",)
            if not isinstance(expression, llir.Expr):
                _raise_traversal_error(
                    self.context,
                    code="invalid_expression_sequence_member",
                    message="expression sequences may contain only LLIR expressions",
                    path=item_path,
                    value=expression,
                )
            self._walk_expr(expression, item_path)

    def _walk_var_child(self, value: object, path: LLIRPath) -> None:
        if not isinstance(value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_var_child",
                message="expected an LLIR Var",
                path=path,
                value=value,
            )
        self._walk_expr(value, path)
        if type(value) is not llir.Var:
            _raise_traversal_error(
                self.context,
                code="invalid_var_child",
                message="expected an LLIR Var",
                path=path,
                value=value,
            )

    def _walk_assignment_target(self, value: object, path: LLIRPath) -> None:
        if not isinstance(value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_assignment_target",
                message="expected an exact LLIR Var or ArrayAccess",
                path=path,
                value=value,
            )
        self._walk_expr(value, path)
        try:
            llir._validate_assignment_target(value)
        except TypeError as error:
            _raise_traversal_error(
                self.context,
                code="invalid_assignment_target",
                message=str(error),
                path=path,
                value=value,
            )

    def _walk_branches(self, branches: object, path: LLIRPath) -> None:
        if type(branches) is not list and type(branches) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_branch_sequence",
                message="expected a branch list or tuple",
                path=path,
                value=branches,
            )
        for index, branch in enumerate(cast(Sequence[object], branches)):
            self._walk_statements(branch, path + (f"[{index}]",))

    def _walk_expr(self, node: llir.Expr, path: LLIRPath) -> None:
        node_type = type(node)
        if node_type not in SUPPORTED_LLIR_EXPRESSION_NODE_TYPES:
            _raise_traversal_error(
                self.context,
                code="unknown_llir_node",
                message=f"unsupported LLIR expression type '{node_type.__name__}'",
                path=path,
                value=node,
            )
        self.enter_node(node, path)
        if node_type is llir.GetTensorProperty:
            self.visit_get_tensor_property(cast(llir.GetTensorProperty, node), path)
        elif node_type is llir.Var:
            self.visit_var(cast(llir.Var, node), path)
        elif node_type is llir.UnaryOp:
            self.visit_unary_op(cast(llir.UnaryOp, node), path)
        elif node_type is llir.BinOp:
            self.visit_bin_op(cast(llir.BinOp, node), path)
        elif node_type is llir.Add:
            self.visit_add(cast(llir.Add, node), path)
        elif node_type is llir.Mul:
            self.visit_mul(cast(llir.Mul, node), path)
        elif node_type is llir.Literal:
            self.visit_literal(cast(llir.Literal, node), path)
        elif node_type is llir.FunctionCall:
            self.visit_function_call(cast(llir.FunctionCall, node), path)
        elif node_type is llir.Array:
            self.visit_array(cast(llir.Array, node), path)
        elif node_type is llir.MemberAccess:
            self.visit_member_access(cast(llir.MemberAccess, node), path)
        elif node_type is llir.ArrayAccess:
            self.visit_array_access(cast(llir.ArrayAccess, node), path)
        elif node_type is llir.Cast:
            self.visit_cast(cast(llir.Cast, node), path)
        elif node_type is llir.Sizeof:
            self.visit_sizeof(cast(llir.Sizeof, node), path)
        else:
            _raise_traversal_error(
                self.context,
                code="unhandled_supported_llir_node",
                message=f"missing walker hook for '{node_type.__name__}'",
                path=path,
                value=node,
            )
        self.leave_node(node, path)

    def _walk_stmt(self, node: llir.Stmt, path: LLIRPath) -> None:
        node_type = type(node)
        if node_type not in SUPPORTED_LLIR_STATEMENT_NODE_TYPES:
            _raise_traversal_error(
                self.context,
                code="unknown_llir_node",
                message=f"unsupported LLIR statement type '{node_type.__name__}'",
                path=path,
                value=node,
            )
        self.enter_node(node, path)
        if node_type is llir.Increment:
            self.visit_increment(cast(llir.Increment, node), path)
        elif node_type is llir.Return:
            self.visit_return(cast(llir.Return, node), path)
        elif node_type is llir.VarDecl:
            self.visit_var_decl(cast(llir.VarDecl, node), path)
        elif node_type is llir.VarInit:
            self.visit_var_init(cast(llir.VarInit, node), path)
        elif node_type is llir.Assign:
            self.visit_assign(cast(llir.Assign, node), path)
        elif node_type is llir.Allocate:
            self.visit_allocate(cast(llir.Allocate, node), path)
        elif node_type is llir.Free:
            self.visit_free(cast(llir.Free, node), path)
        elif node_type is llir.Print:
            self.visit_print(cast(llir.Print, node), path)
        elif node_type is llir.Comment:
            self.visit_comment(cast(llir.Comment, node), path)
        elif node_type is llir.BlankLine:
            self.visit_blank_line(cast(llir.BlankLine, node), path)
        elif node_type is llir.RawStmt:
            self.visit_raw_stmt(cast(llir.RawStmt, node), path)
        elif node_type is llir.Continue:
            self.visit_continue(cast(llir.Continue, node), path)
        elif node_type is llir.Break:
            self.visit_break(cast(llir.Break, node), path)
        elif node_type is llir.Function:
            self.visit_function(cast(llir.Function, node), path)
        elif node_type is llir.FunctionCallStmt:
            self.visit_function_call_stmt(cast(llir.FunctionCallStmt, node), path)
        elif node_type is llir.ForLoop:
            self.visit_for_loop(cast(llir.ForLoop, node), path)
        elif node_type is llir.ForLoopAuto:
            self.visit_for_loop_auto(cast(llir.ForLoopAuto, node), path)
        elif node_type is llir.WhileLoop:
            self.visit_while_loop(cast(llir.WhileLoop, node), path)
        elif node_type is llir.IfThenElse:
            self.visit_if_then_else(cast(llir.IfThenElse, node), path)
        elif node_type is llir.Case:
            self.visit_case(cast(llir.Case, node), path)
        elif node_type is llir.Switch:
            self.visit_switch(cast(llir.Switch, node), path)
        else:
            _raise_traversal_error(
                self.context,
                code="unhandled_supported_llir_node",
                message=f"missing walker hook for '{node_type.__name__}'",
                path=path,
                value=node,
            )
        self.leave_node(node, path)

    def visit_get_tensor_property(
        self, node: llir.GetTensorProperty, path: LLIRPath
    ) -> None:
        self._walk_expr(node.tensor, path + ("tensor",))

    def visit_var(self, node: llir.Var, path: LLIRPath) -> None:
        _validate_tensor_access_metadata(
            node.tensor_access,
            self.context,
            path + ("tensor_access",),
        )

    def visit_unary_op(self, node: llir.UnaryOp, path: LLIRPath) -> None:
        self._walk_expr(node.operand, path + ("operand",))

    def visit_bin_op(self, node: llir.BinOp, path: LLIRPath) -> None:
        _validate_binary_expression_fields(node, self.context, path)
        self._walk_expr(node.left, path + ("left",))
        self._walk_expr(node.right, path + ("right",))

    def visit_add(self, node: llir.Add, path: LLIRPath) -> None:
        self.visit_bin_op(node, path)

    def visit_mul(self, node: llir.Mul, path: LLIRPath) -> None:
        self.visit_bin_op(node, path)

    def visit_literal(self, node: llir.Literal, path: LLIRPath) -> None:
        _validate_literal_fields(node, self.context, path)

    def visit_function_call(self, node: llir.FunctionCall, path: LLIRPath) -> None:
        if type(node.name) is not str or not node.name.strip():
            _raise_traversal_error(
                self.context,
                code="invalid_function_call_name",
                message="FunctionCall.name must be a non-empty string",
                path=path + ("name",),
                value=node.name,
            )
        if type(node.args) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_function_call_args",
                message="FunctionCall.args must be a tuple",
                path=path + ("args",),
                value=node.args,
            )
        self._walk_expr_sequence(node.args, path + ("args",))

    def visit_array(self, node: llir.Array, path: LLIRPath) -> None:
        self._walk_expr_sequence(node.values, path + ("values",))

    def visit_member_access(self, node: llir.MemberAccess, path: LLIRPath) -> None:
        if not isinstance(node.base, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_member_access_base",
                message="MemberAccess.base must be an LLIR Expr",
                path=path + ("base",),
                value=node.base,
            )
        if type(node.member) is not str or not node.member.isidentifier():
            _raise_traversal_error(
                self.context,
                code="invalid_member_access_member",
                message="MemberAccess.member must be a non-empty identifier",
                path=path + ("member",),
                value=node.member,
            )
        self._walk_expr(node.base, path + ("base",))

    def visit_array_access(self, node: llir.ArrayAccess, path: LLIRPath) -> None:
        _validate_tensor_access_metadata(
            node.tensor_access,
            self.context,
            path + ("tensor_access",),
        )
        self._walk_expr(node.array, path + ("array",))
        self._walk_expr(node.index, path + ("index",))

    def visit_cast(self, node: llir.Cast, path: LLIRPath) -> None:
        _validate_cast_fields(node, self.context, path)
        self._walk_expr(node.expr, path + ("expr",))

    def visit_sizeof(self, node: llir.Sizeof, path: LLIRPath) -> None:
        pass

    def visit_increment(self, node: llir.Increment, path: LLIRPath) -> None:
        self._walk_var_child(node.var, path + ("var",))

    def visit_return(self, node: llir.Return, path: LLIRPath) -> None:
        self._walk_expr(node.value, path + ("value",))

    def visit_var_decl(self, node: llir.VarDecl, path: LLIRPath) -> None:
        self._walk_var_child(node.var, path + ("var",))

    def visit_var_init(self, node: llir.VarInit, path: LLIRPath) -> None:
        self._walk_var_child(node.var, path + ("var",))
        self._walk_expr(node.value, path + ("value",))

    def visit_assign(self, node: llir.Assign, path: LLIRPath) -> None:
        _validate_assign_fields(node, self.context, path)
        self._walk_assignment_target(node.var, path + ("var",))
        if not isinstance(node.value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_assign_value",
                message="Assign.value must be an LLIR Expr",
                path=path + ("value",),
                value=node.value,
            )
        self._walk_expr(node.value, path + ("value",))

    def visit_allocate(self, node: llir.Allocate, path: LLIRPath) -> None:
        self._walk_expr(node.var, path + ("var",))
        self._walk_expr(node.num_elements, path + ("num_elements",))

    def visit_free(self, node: llir.Free, path: LLIRPath) -> None:
        self._walk_expr(node.var, path + ("var",))

    def visit_print(self, node: llir.Print, path: LLIRPath) -> None:
        self._walk_expr(node.value, path + ("value",))

    def visit_comment(self, node: llir.Comment, path: LLIRPath) -> None:
        pass

    def visit_blank_line(self, node: llir.BlankLine, path: LLIRPath) -> None:
        pass

    def visit_raw_stmt(self, node: llir.RawStmt, path: LLIRPath) -> None:
        pass

    def visit_continue(self, node: llir.Continue, path: LLIRPath) -> None:
        pass

    def visit_break(self, node: llir.Break, path: LLIRPath) -> None:
        pass

    def visit_function(self, node: llir.Function, path: LLIRPath) -> None:
        self._walk_expr_sequence(node.args, path + ("args",))
        self._walk_statements(node.body, path + ("body",))

    def visit_function_call_stmt(
        self, node: llir.FunctionCallStmt, path: LLIRPath
    ) -> None:
        self._walk_expr_sequence(node.args, path + ("args",))

    def visit_for_loop(self, node: llir.ForLoop, path: LLIRPath) -> None:
        self._walk_optional_statements(
            node.before_parallel_body, path + ("before_parallel_body",)
        )
        if node.init is not None:
            self._walk_for_loop_init(node.init, path + ("init",))
        self._walk_expr(node.cond, path + ("cond",))
        self._walk_for_loop_update(node.update, path + ("update",))
        self._walk_optional_statements(
            node.pre_parallel_body, path + ("pre_parallel_body",)
        )
        self._walk_statements(node.body, path + ("body",))
        self._walk_optional_statements(
            node.post_parallel_body, path + ("post_parallel_body",)
        )
        if hasattr(node, "_hoisted_ptr_decls"):
            self._walk_statements(
                getattr(node, "_hoisted_ptr_decls"),
                path + ("_hoisted_ptr_decls",),
            )

    def visit_for_loop_auto(self, node: llir.ForLoopAuto, path: LLIRPath) -> None:
        self._walk_var_child(node.var, path + ("var",))
        self._walk_expr(node.array, path + ("array",))
        self._walk_statements(node.body, path + ("body",))

    def visit_while_loop(self, node: llir.WhileLoop, path: LLIRPath) -> None:
        self._walk_expr(node.cond, path + ("cond",))
        self._walk_statements(node.body, path + ("body",))

    def visit_if_then_else(self, node: llir.IfThenElse, path: LLIRPath) -> None:
        if node.cond is not None:
            self._walk_expr(node.cond, path + ("cond",))
        self._walk_optional_statements(node.then_body, path + ("then_body",))
        if node.cond_list is not None:
            self._walk_expr_sequence(node.cond_list, path + ("cond_list",))
        if node.then_body_list is not None:
            self._walk_branches(node.then_body_list, path + ("then_body_list",))
        self._walk_optional_statements(node.else_body, path + ("else_body",))

    def visit_case(self, node: llir.Case, path: LLIRPath) -> None:
        self._walk_expr(node.cond, path + ("cond",))
        self._walk_statements(node.body, path + ("body",))

    def visit_switch(self, node: llir.Switch, path: LLIRPath) -> None:
        self._walk_expr(node.cond, path + ("cond",))
        self._walk_statements(node.cases, path + ("cases",))
        self._walk_statements(node.default, path + ("default",))


class LLIRRewriter:
    """Exhaustive ownership-safe identity rewriter for the current LLIR."""

    def __init__(self, context: LLIRTraversalContext) -> None:
        self.context = context

    def rewrite(self, value: LLIRRewriteValueT) -> LLIRRewriteValueT:
        """Return a detached value with the same scalar/container root shape."""

        return cast(
            LLIRRewriteValueT,
            self._rewrite_value(cast(LLIRValue, value), ("root",)),
        )

    def _rewrite_value(self, value: LLIRValue, path: LLIRPath) -> LLIRValue:
        if isinstance(value, llir.Expr):
            return self._rewrite_expr(value, path)
        if isinstance(value, llir.Stmt):
            return self._rewrite_stmt(value, path)
        if type(value) is list or type(value) is tuple:
            return self.rewrite_statement_sequence(
                cast(LLIRStatementSequence, value), path
            )
        _raise_traversal_error(
            self.context,
            code="invalid_llir_value",
            message="expected an LLIR node or statement list/tuple",
            path=path,
            value=value,
        )

    def prepare_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> Sequence[LLIRStatementValue]:
        """Return the statements to rebuild; subclasses may filter, not mutate."""

        return statements

    def rewrite_statement_sequence_member(
        self, node: llir.Stmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        """Return zero or more statements to rebuild in ``node``'s place.

        Subclasses may delete or expand a statement by returning an exact list
        or tuple.  Each returned statement is subsequently dispatched through
        the ordinary exhaustive typed rewriter, so replacement children are
        validated and detached before they enter the rewritten tree.
        """

        return (node,)

    def rewrite_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> LLIRStatementSequence:
        prepared = self.prepare_statement_sequence(statements, path)
        rewritten: List[LLIRStatementValue] = []
        for index, statement in enumerate(prepared):
            item_path = path + (f"[{index}]",)
            if isinstance(statement, llir.Stmt):
                replacements = self.rewrite_statement_sequence_member(
                    statement, item_path
                )
                if type(replacements) is not list and type(replacements) is not tuple:
                    _raise_traversal_error(
                        self.context,
                        code="invalid_statement_rewrite_sequence",
                        message=(
                            "rewrite_statement_sequence_member must return an exact "
                            "statement list or tuple"
                        ),
                        path=item_path,
                        value=replacements,
                    )
                for replacement_index, replacement in enumerate(replacements):
                    replacement_path = item_path
                    if len(replacements) != 1:
                        replacement_path += (f"replacement[{replacement_index}]",)
                    if not isinstance(replacement, llir.Stmt):
                        _raise_traversal_error(
                            self.context,
                            code="invalid_statement_rewrite_member",
                            message=(
                                "rewrite_statement_sequence_member may return only "
                                "LLIR statements"
                            ),
                            path=replacement_path,
                            value=replacement,
                        )
                    rewritten.append(self._rewrite_stmt(replacement, replacement_path))
            elif type(statement) is list or type(statement) is tuple:
                rewritten.append(
                    self.rewrite_statement_sequence(
                        cast(LLIRStatementSequence, statement), item_path
                    )
                )
            else:
                _raise_traversal_error(
                    self.context,
                    code="invalid_statement_sequence_member",
                    message=(
                        "statement sequences may contain only LLIR statements "
                        "or nested statement lists/tuples"
                    ),
                    path=item_path,
                    value=statement,
                )
        if type(statements) is list:
            return rewritten
        if type(statements) is tuple:
            return tuple(rewritten)
        _raise_traversal_error(
            self.context,
            code="invalid_statement_sequence",
            message="expected a statement list or tuple",
            path=path,
            value=statements,
        )

    def _rewrite_statements(
        self, statements: object, path: LLIRPath
    ) -> LLIRStatementSequence:
        if type(statements) is not list and type(statements) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_statement_sequence",
                message="required child must be a statement list or tuple",
                path=path,
                value=statements,
            )
        return self.rewrite_statement_sequence(
            cast(LLIRStatementSequence, statements), path
        )

    def _rewrite_optional_statements(
        self,
        statements: Optional[Sequence[llir.Stmt]],
        path: LLIRPath,
    ) -> Optional[LLIRStatementSequence]:
        if statements is None:
            return None
        return self._rewrite_statements(statements, path)

    def _rewrite_for_loop_init(
        self, init: object, path: LLIRPath
    ) -> Union[llir.VarInit, llir.VarDecl]:
        if type(init) in (llir.VarInit, llir.VarDecl):
            rewritten = self._rewrite_stmt(cast(llir.Stmt, init), path)
            if type(rewritten) in (llir.VarInit, llir.VarDecl):
                return cast(Union[llir.VarInit, llir.VarDecl], rewritten)
            _raise_traversal_error(
                self.context,
                code="invalid_rewritten_for_loop_init",
                message="rewriter produced an invalid ForLoop.init node",
                path=path,
                value=rewritten,
            )
        _raise_traversal_error(
            self.context,
            code="invalid_for_loop_init",
            message="ForLoop.init must be VarInit, VarDecl, or None",
            path=path,
            value=init,
        )

    def _rewrite_for_loop_update(
        self, update: object, path: LLIRPath
    ) -> Union[llir.Increment, llir.VarInit, llir.FunctionCall, llir.Assign]:
        if type(update) in (llir.Increment, llir.VarInit, llir.Assign):
            rewritten = self._rewrite_stmt(cast(llir.Stmt, update), path)
            if type(rewritten) in (llir.Increment, llir.VarInit, llir.Assign):
                return cast(Union[llir.Increment, llir.VarInit, llir.Assign], rewritten)
            _raise_traversal_error(
                self.context,
                code="invalid_rewritten_for_loop_update",
                message="rewriter produced an invalid ForLoop.update node",
                path=path,
                value=rewritten,
            )
        if type(update) is llir.FunctionCall:
            rewritten_call = self._rewrite_expr(cast(llir.FunctionCall, update), path)
            if type(rewritten_call) is llir.FunctionCall:
                return cast(llir.FunctionCall, rewritten_call)
            _raise_traversal_error(
                self.context,
                code="invalid_rewritten_for_loop_update",
                message="rewriter produced an invalid ForLoop.update node",
                path=path,
                value=rewritten_call,
            )
        _raise_traversal_error(
            self.context,
            code="invalid_for_loop_update",
            message=(
                "ForLoop.update must be Increment, VarInit, FunctionCall, or Assign"
            ),
            path=path,
            value=update,
        )

    def _rewrite_expr_sequence(
        self, expressions: Sequence[llir.Expr], path: LLIRPath
    ) -> Sequence[llir.Expr]:
        if type(expressions) is not list and type(expressions) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_expression_sequence",
                message="expected an expression list or tuple",
                path=path,
                value=expressions,
            )
        rewritten: List[llir.Expr] = []
        for index, expression in enumerate(expressions):
            item_path = path + (f"[{index}]",)
            if not isinstance(expression, llir.Expr):
                _raise_traversal_error(
                    self.context,
                    code="invalid_expression_sequence_member",
                    message="expression sequences may contain only LLIR expressions",
                    path=item_path,
                    value=expression,
                )
            rewritten.append(self._rewrite_expr(expression, item_path))
        if type(expressions) is tuple:
            return tuple(rewritten)
        return rewritten

    def _rewrite_var_child(self, value: object, path: LLIRPath) -> llir.Var:
        if not isinstance(value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_var_child",
                message="expected an LLIR Var",
                path=path,
                value=value,
            )
        rewritten = self._rewrite_expr(value, path)
        if type(rewritten) is not llir.Var:
            _raise_traversal_error(
                self.context,
                code="invalid_var_child",
                message="expected an LLIR Var",
                path=path,
                value=rewritten,
            )
        return cast(llir.Var, rewritten)

    def _rewrite_assignment_target(
        self, value: object, path: LLIRPath
    ) -> llir.AssignmentTarget:
        if not isinstance(value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_assignment_target",
                message="expected an exact LLIR Var or ArrayAccess",
                path=path,
                value=value,
            )
        rewritten = self._rewrite_expr(value, path)
        try:
            llir._validate_assignment_target(rewritten)
        except TypeError as error:
            _raise_traversal_error(
                self.context,
                code="invalid_assignment_target",
                message=str(error),
                path=path,
                value=rewritten,
            )
        return cast(llir.AssignmentTarget, rewritten)

    def _rewrite_branches(
        self,
        branches: object,
        path: LLIRPath,
    ) -> Sequence[LLIRStatementSequence]:
        if type(branches) is not list and type(branches) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_branch_sequence",
                message="expected a branch list or tuple",
                path=path,
                value=branches,
            )
        rewritten = [
            self._rewrite_statements(branch, path + (f"[{index}]",))
            for index, branch in enumerate(cast(Sequence[object], branches))
        ]
        if type(branches) is tuple:
            return tuple(rewritten)
        return rewritten

    def _rewrite_expr(self, node: llir.Expr, path: LLIRPath) -> llir.Expr:
        node_type = type(node)
        if node_type is llir.GetTensorProperty:
            return self.rewrite_get_tensor_property(
                cast(llir.GetTensorProperty, node), path
            )
        if node_type is llir.Var:
            return self.rewrite_var(cast(llir.Var, node), path)
        if node_type is llir.UnaryOp:
            return self.rewrite_unary_op(cast(llir.UnaryOp, node), path)
        if node_type is llir.BinOp:
            return self.rewrite_bin_op(cast(llir.BinOp, node), path)
        if node_type is llir.Add:
            return self.rewrite_add(cast(llir.Add, node), path)
        if node_type is llir.Mul:
            return self.rewrite_mul(cast(llir.Mul, node), path)
        if node_type is llir.Literal:
            return self.rewrite_literal(cast(llir.Literal, node), path)
        if node_type is llir.FunctionCall:
            return self.rewrite_function_call(cast(llir.FunctionCall, node), path)
        if node_type is llir.Array:
            return self.rewrite_array(cast(llir.Array, node), path)
        if node_type is llir.MemberAccess:
            return self.rewrite_member_access(cast(llir.MemberAccess, node), path)
        if node_type is llir.ArrayAccess:
            return self.rewrite_array_access(cast(llir.ArrayAccess, node), path)
        if node_type is llir.Cast:
            return self.rewrite_cast(cast(llir.Cast, node), path)
        if node_type is llir.Sizeof:
            return self.rewrite_sizeof(cast(llir.Sizeof, node), path)
        _raise_traversal_error(
            self.context,
            code="unknown_llir_node",
            message=f"unsupported LLIR expression type '{node_type.__name__}'",
            path=path,
            value=node,
        )

    def _rewrite_stmt(self, node: llir.Stmt, path: LLIRPath) -> llir.Stmt:
        node_type = type(node)
        if node_type is llir.Increment:
            return self.rewrite_increment(cast(llir.Increment, node), path)
        if node_type is llir.Return:
            return self.rewrite_return(cast(llir.Return, node), path)
        if node_type is llir.VarDecl:
            return self.rewrite_var_decl(cast(llir.VarDecl, node), path)
        if node_type is llir.VarInit:
            return self.rewrite_var_init(cast(llir.VarInit, node), path)
        if node_type is llir.Assign:
            return self.rewrite_assign(cast(llir.Assign, node), path)
        if node_type is llir.Allocate:
            return self.rewrite_allocate(cast(llir.Allocate, node), path)
        if node_type is llir.Free:
            return self.rewrite_free(cast(llir.Free, node), path)
        if node_type is llir.Print:
            return self.rewrite_print(cast(llir.Print, node), path)
        if node_type is llir.Comment:
            return self.rewrite_comment(cast(llir.Comment, node), path)
        if node_type is llir.BlankLine:
            return self.rewrite_blank_line(cast(llir.BlankLine, node), path)
        if node_type is llir.RawStmt:
            return self.rewrite_raw_stmt(cast(llir.RawStmt, node), path)
        if node_type is llir.Continue:
            return self.rewrite_continue(cast(llir.Continue, node), path)
        if node_type is llir.Break:
            return self.rewrite_break(cast(llir.Break, node), path)
        if node_type is llir.Function:
            return self.rewrite_function(cast(llir.Function, node), path)
        if node_type is llir.FunctionCallStmt:
            return self.rewrite_function_call_stmt(
                cast(llir.FunctionCallStmt, node), path
            )
        if node_type is llir.ForLoop:
            return self.rewrite_for_loop(cast(llir.ForLoop, node), path)
        if node_type is llir.ForLoopAuto:
            return self.rewrite_for_loop_auto(cast(llir.ForLoopAuto, node), path)
        if node_type is llir.WhileLoop:
            return self.rewrite_while_loop(cast(llir.WhileLoop, node), path)
        if node_type is llir.IfThenElse:
            return self.rewrite_if_then_else(cast(llir.IfThenElse, node), path)
        if node_type is llir.Case:
            return self.rewrite_case(cast(llir.Case, node), path)
        if node_type is llir.Switch:
            return self.rewrite_switch(cast(llir.Switch, node), path)
        _raise_traversal_error(
            self.context,
            code="unknown_llir_node",
            message=f"unsupported LLIR statement type '{node_type.__name__}'",
            path=path,
            value=node,
        )

    def rewrite_get_tensor_property(
        self, node: llir.GetTensorProperty, path: LLIRPath
    ) -> llir.GetTensorProperty:
        return llir.GetTensorProperty(
            tensor=self._rewrite_expr(node.tensor, path + ("tensor",)),
            tensor_property=node.tensor_property,
            level=node.level,
            index=node.index,
            name=node.name,
        )

    def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
        _validate_tensor_access_metadata(
            node.tensor_access,
            self.context,
            path + ("tensor_access",),
        )
        return llir.Var(
            name=node.name,
            type=node.type,
            is_ptr=node.is_ptr,
            is_restrict=node.is_restrict,
            tensor_access=node.tensor_access,
        )

    def rewrite_unary_op(self, node: llir.UnaryOp, path: LLIRPath) -> llir.UnaryOp:
        return llir.UnaryOp(
            op=node.op,
            operand=self._rewrite_expr(node.operand, path + ("operand",)),
        )

    def rewrite_bin_op(self, node: llir.BinOp, path: LLIRPath) -> llir.BinOp:
        _validate_binary_expression_fields(node, self.context, path)
        return llir.BinOp(
            op=node.op,
            left=self._rewrite_expr(node.left, path + ("left",)),
            right=self._rewrite_expr(node.right, path + ("right",)),
        )

    def rewrite_add(self, node: llir.Add, path: LLIRPath) -> llir.Add:
        _validate_binary_expression_fields(node, self.context, path)
        return llir.Add(
            left=self._rewrite_expr(node.left, path + ("left",)),
            right=self._rewrite_expr(node.right, path + ("right",)),
        )

    def rewrite_mul(self, node: llir.Mul, path: LLIRPath) -> llir.Mul:
        _validate_binary_expression_fields(node, self.context, path)
        return llir.Mul(
            left=self._rewrite_expr(node.left, path + ("left",)),
            right=self._rewrite_expr(node.right, path + ("right",)),
        )

    def rewrite_literal(self, node: llir.Literal, path: LLIRPath) -> llir.Literal:
        _validate_literal_fields(node, self.context, path)
        return llir.Literal(value=node.value, data_type=node.data_type)

    def rewrite_function_call(
        self, node: llir.FunctionCall, path: LLIRPath
    ) -> llir.FunctionCall:
        if type(node.name) is not str or not node.name.strip():
            _raise_traversal_error(
                self.context,
                code="invalid_function_call_name",
                message="FunctionCall.name must be a non-empty string",
                path=path + ("name",),
                value=node.name,
            )
        if type(node.args) is not tuple:
            _raise_traversal_error(
                self.context,
                code="invalid_function_call_args",
                message="FunctionCall.args must be a tuple",
                path=path + ("args",),
                value=node.args,
            )
        return llir.FunctionCall(
            name=node.name,
            args=cast(
                List[llir.Expr],
                self._rewrite_expr_sequence(node.args, path + ("args",)),
            ),
        )

    def rewrite_array(self, node: llir.Array, path: LLIRPath) -> llir.Array:
        return llir.Array(
            values=cast(
                List[llir.Expr],
                self._rewrite_expr_sequence(node.values, path + ("values",)),
            ),
            data_type=node.data_type,
        )

    def rewrite_member_access(
        self, node: llir.MemberAccess, path: LLIRPath
    ) -> llir.MemberAccess:
        if not isinstance(node.base, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_member_access_base",
                message="MemberAccess.base must be an LLIR Expr",
                path=path + ("base",),
                value=node.base,
            )
        if type(node.member) is not str or not node.member.isidentifier():
            _raise_traversal_error(
                self.context,
                code="invalid_member_access_member",
                message="MemberAccess.member must be a non-empty identifier",
                path=path + ("member",),
                value=node.member,
            )
        return llir.MemberAccess(
            base=self._rewrite_expr(node.base, path + ("base",)),
            member=node.member,
        )

    def rewrite_array_access(
        self, node: llir.ArrayAccess, path: LLIRPath
    ) -> llir.ArrayAccess:
        _validate_tensor_access_metadata(
            node.tensor_access,
            self.context,
            path + ("tensor_access",),
        )
        return llir.ArrayAccess(
            array=self._rewrite_expr(node.array, path + ("array",)),
            index=self._rewrite_expr(node.index, path + ("index",)),
            tensor_access=node.tensor_access,
        )

    def rewrite_cast(self, node: llir.Cast, path: LLIRPath) -> llir.Cast:
        _validate_cast_fields(node, self.context, path)
        return llir.Cast(
            expr=self._rewrite_expr(node.expr, path + ("expr",)),
            data_type=node.data_type,
        )

    def rewrite_sizeof(self, node: llir.Sizeof, path: LLIRPath) -> llir.Sizeof:
        return llir.Sizeof(data_type=node.data_type)

    def rewrite_increment(self, node: llir.Increment, path: LLIRPath) -> llir.Increment:
        return llir.Increment(self._rewrite_var_child(node.var, path + ("var",)))

    def rewrite_return(self, node: llir.Return, path: LLIRPath) -> llir.Return:
        return llir.Return(self._rewrite_expr(node.value, path + ("value",)))

    def rewrite_var_decl(self, node: llir.VarDecl, path: LLIRPath) -> llir.VarDecl:
        return llir.VarDecl(self._rewrite_var_child(node.var, path + ("var",)))

    def rewrite_var_init(self, node: llir.VarInit, path: LLIRPath) -> llir.VarInit:
        rewritten = llir.VarInit(
            var=self._rewrite_var_child(node.var, path + ("var",)),
            value=self._rewrite_expr(node.value, path + ("value",)),
            op=node.op,
            cast=False,
        )
        rewritten.cast = node.cast
        return rewritten

    def rewrite_assign(self, node: llir.Assign, path: LLIRPath) -> llir.Stmt:
        _validate_assign_fields(node, self.context, path)
        if not isinstance(node.value, llir.Expr):
            _raise_traversal_error(
                self.context,
                code="invalid_assign_value",
                message="Assign.value must be an LLIR Expr",
                path=path + ("value",),
                value=node.value,
            )
        rewritten = llir.Assign(
            var=self._rewrite_assignment_target(node.var, path + ("var",)),
            value=self._rewrite_expr(node.value, path + ("value",)),
            op=node.op,
            cast=False,
        )
        rewritten.cast = node.cast
        return rewritten

    def rewrite_allocate(self, node: llir.Allocate, path: LLIRPath) -> llir.Allocate:
        return llir.Allocate(
            var=self._rewrite_expr(node.var, path + ("var",)),
            num_elements=self._rewrite_expr(
                node.num_elements, path + ("num_elements",)
            ),
            is_realloc=node.is_realloc,
            use_calloc=node.use_calloc,
        )

    def rewrite_free(self, node: llir.Free, path: LLIRPath) -> llir.Free:
        return llir.Free(self._rewrite_expr(node.var, path + ("var",)))

    def rewrite_print(self, node: llir.Print, path: LLIRPath) -> llir.Print:
        return llir.Print(self._rewrite_expr(node.value, path + ("value",)))

    def rewrite_comment(self, node: llir.Comment, path: LLIRPath) -> llir.Comment:
        return llir.Comment(node.value)

    def rewrite_blank_line(
        self, node: llir.BlankLine, path: LLIRPath
    ) -> llir.BlankLine:
        return llir.BlankLine()

    def rewrite_raw_stmt(self, node: llir.RawStmt, path: LLIRPath) -> llir.RawStmt:
        return llir.RawStmt(code=node.code, add_semicolon=node.add_semicolon)

    def rewrite_continue(self, node: llir.Continue, path: LLIRPath) -> llir.Continue:
        return llir.Continue()

    def rewrite_break(self, node: llir.Break, path: LLIRPath) -> llir.Break:
        return llir.Break()

    def rewrite_function(self, node: llir.Function, path: LLIRPath) -> llir.Function:
        args = self._rewrite_expr_sequence(node.args, path + ("args",))
        body = self._rewrite_statements(node.body, path + ("body",))
        return llir.Function(
            return_type=node.return_type,
            name=node.name,
            args=args,
            body=cast(List[llir.Stmt], body),
        )

    def rewrite_function_call_stmt(
        self, node: llir.FunctionCallStmt, path: LLIRPath
    ) -> llir.FunctionCallStmt:
        return llir.FunctionCallStmt(
            name=node.name,
            args=cast(
                List[llir.Expr],
                self._rewrite_expr_sequence(node.args, path + ("args",)),
            ),
        )

    def rewrite_for_loop(self, node: llir.ForLoop, path: LLIRPath) -> llir.ForLoop:
        before = self._rewrite_optional_statements(
            node.before_parallel_body, path + ("before_parallel_body",)
        )
        init = (
            None
            if node.init is None
            else self._rewrite_for_loop_init(node.init, path + ("init",))
        )
        cond = self._rewrite_expr(node.cond, path + ("cond",))
        update = self._rewrite_for_loop_update(node.update, path + ("update",))
        pre = self._rewrite_optional_statements(
            node.pre_parallel_body, path + ("pre_parallel_body",)
        )
        body = self._rewrite_statements(node.body, path + ("body",))
        post = self._rewrite_optional_statements(
            node.post_parallel_body, path + ("post_parallel_body",)
        )
        rewritten = llir.ForLoop(
            init=init,
            cond=cond,
            update=update,
            body=cast(List[llir.Stmt], body),
            omp_parallel_for=node.omp_parallel_for,
            omp_schedule=node.omp_schedule,
            unroll=node.unroll,
            simd=node.simd,
            pre_parallel_body=cast(Optional[List[llir.Stmt]], pre),
            post_parallel_body=cast(Optional[List[llir.Stmt]], post),
            omp_num_threads=node.omp_num_threads,
            omp_chunk_expr=node.omp_chunk_expr,
            before_parallel_body=cast(Optional[List[llir.Stmt]], before),
        )
        rewritten.scorch_index_var = node.scorch_index_var

        # These are known legacy compatibility fields, not open-ended
        # ``__dict__`` copying.  They are attached by existing lowerer passes.
        for attribute in (
            "_use_atomic_scheduling",
            "_atomic_chunk_var",
            "_atomic_counter_var",
            "_loop_bound",
        ):
            if hasattr(node, attribute):
                setattr(rewritten, attribute, getattr(node, attribute))
        if hasattr(node, "_hoisted_ptr_decls"):
            declarations = getattr(node, "_hoisted_ptr_decls")
            setattr(
                rewritten,
                "_hoisted_ptr_decls",
                self._rewrite_statements(declarations, path + ("_hoisted_ptr_decls",)),
            )
        return rewritten

    def rewrite_for_loop_auto(
        self, node: llir.ForLoopAuto, path: LLIRPath
    ) -> llir.ForLoopAuto:
        var = self._rewrite_var_child(node.var, path + ("var",))
        array = self._rewrite_expr(node.array, path + ("array",))
        body = self._rewrite_statements(node.body, path + ("body",))
        return llir.ForLoopAuto(
            var=var,
            array=array,
            body=cast(List[llir.Stmt], body),
        )

    def rewrite_while_loop(
        self, node: llir.WhileLoop, path: LLIRPath
    ) -> llir.WhileLoop:
        cond = self._rewrite_expr(node.cond, path + ("cond",))
        body = self._rewrite_statements(node.body, path + ("body",))
        rewritten = llir.WhileLoop(
            cond=cond,
            body=cast(List[llir.Stmt], body),
        )
        rewritten.scorch_index_var = node.scorch_index_var
        return rewritten

    def rewrite_if_then_else(
        self, node: llir.IfThenElse, path: LLIRPath
    ) -> llir.IfThenElse:
        cond = (
            None
            if node.cond is None
            else self._rewrite_expr(node.cond, path + ("cond",))
        )
        then_body = self._rewrite_optional_statements(
            node.then_body, path + ("then_body",)
        )
        cond_list = (
            None
            if node.cond_list is None
            else self._rewrite_expr_sequence(node.cond_list, path + ("cond_list",))
        )
        then_body_list = (
            None
            if node.then_body_list is None
            else self._rewrite_branches(node.then_body_list, path + ("then_body_list",))
        )
        else_body = self._rewrite_optional_statements(
            node.else_body, path + ("else_body",)
        )
        return llir.IfThenElse(
            cond=cond,
            then_body=cast(Optional[List[llir.Stmt]], then_body),
            else_body=cast(Optional[List[llir.Stmt]], else_body),
            cond_list=cast(Optional[List[llir.Expr]], cond_list),
            then_body_list=cast(Optional[List[List[llir.Stmt]]], then_body_list),
            make_last_case_else=node.make_last_case_else,
        )

    def rewrite_case(self, node: llir.Case, path: LLIRPath) -> llir.Case:
        cond = self._rewrite_expr(node.cond, path + ("cond",))
        body = self._rewrite_statements(node.body, path + ("body",))
        return llir.Case(
            cond=cond,
            body=cast(List[llir.Stmt], body),
        )

    def rewrite_switch(self, node: llir.Switch, path: LLIRPath) -> llir.Switch:
        cond = self._rewrite_expr(node.cond, path + ("cond",))
        cases = self._rewrite_statements(node.cases, path + ("cases",))
        default = self._rewrite_statements(node.default, path + ("default",))
        return llir.Switch(
            cond=cond,
            cases=cast(List[llir.Case], cases),
            default=cast(List[llir.Stmt], default),
        )
