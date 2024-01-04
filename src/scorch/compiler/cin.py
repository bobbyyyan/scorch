from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union

import torch

from . import llir
from ..format import TensorFormat, LevelType
from ..utils import parse_format

# Type aliases for type hints
_UnaryOp = Callable[[Any], Any]
_BinaryOp = Callable[[Any, Any], Any]


class CIN:
    def accept(self, visitor: CINVisitor) -> None:
        visitor.visit(self)


class IndexExpr(CIN):
    """
    An expression in the taco IR.
    """

    def __mul__(self, other) -> "BinaryOp":
        return BinaryOp(Operation.MUL, self, other)

    def __add__(self, other) -> "BinaryOp":
        return BinaryOp(Operation.ADD, self, other)

    def __sub__(self, other) -> "BinaryOp":
        return BinaryOp(Operation.SUB, self, other)


class IndexStmt(CIN):
    """A statement is a list of tensor operations.
    e.g. A(i) = B(i) + C(i)
    """

    def __init__(self, lhs: Optional[IndexExpr], rhs: Optional[IndexExpr]):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"IndexStmt(lhs={self.lhs}, rhs={self.rhs})"

    def __repr__(self):
        return f"IndexStmt(lhs={self.lhs}, rhs={self.rhs})"

    def get_result_tensor_accesses(self) -> List["TensorAccess"]:
        class ResultTensorAccessCollector(CINVisitorAccept):
            def __init__(self):
                self.result_tensor_accesses: List[TensorAccess] = []

            def get_result_tensor_accesses(self):
                return self.result_tensor_accesses

            def visit_TensorAssign(self, node: TensorAssign):
                self.result_tensor_accesses.append(node.lhs)

        collector = ResultTensorAccessCollector()
        collector.visit(self)
        # self.accept(collector)
        return collector.get_result_tensor_accesses()

    def get_result_tensor_vars(self) -> List[TensorVar]:
        result_tensor_vars = []
        result_tensor_accesses: List[TensorAccess] = self.get_result_tensor_accesses()
        for tensor_access in result_tensor_accesses:
            if isinstance(tensor_access, TensorAccess):
                result_tensor_vars.append(tensor_access.get_tensor())
        return result_tensor_vars

    def get_rhs_tensor_accesses(self) -> List[TensorAccess]:
        class RHSAccessCollector(CINVisitorAccept):
            def __init__(self):
                self.rhs_tensor_accesses: List[TensorAccess] = []

            def get_rhs_tensor_accesses(self):
                return self.rhs_tensor_accesses

            def visit_TensorAccess(self, node: TensorAccess):
                self.rhs_tensor_accesses.append(node)

            def visit_TensorAssign(self, node: TensorAssign):
                self.visit(node.rhs)

            def visit_Where(self, node: Where):
                # Only want to visit the producer, because visiting the consumer
                # would add the workspace to the list of RHS accesses, which we
                # don't want.
                self.visit(node.producer)

        collector = RHSAccessCollector()
        self.accept(collector)
        return collector.get_rhs_tensor_accesses()

    def get_rhs_tensor_vars(self) -> List["TensorVar"]:
        rhs_tensor_vars = []
        rhs_tensor_accesses: List[TensorAccess] = self.get_rhs_tensor_accesses()
        for tensor_access in rhs_tensor_accesses:
            rhs_tensor_vars.append(tensor_access.get_tensor())
        return rhs_tensor_vars

    def get_rhs_workspaces(self) -> List[Workspace]:
        class WorkspaceGetter(CINVisitorAccept):
            workspaces: List[Workspace] = []

            def visit_Workspace(self, node: Workspace) -> None:
                self.workspaces.append(node)

            def visit_TensorAssign(self, node: TensorAssign) -> None:
                self.visit(node.rhs)

            def visit_Where(self, node: Where) -> None:
                self.visit(node.producer)

            def visit_ForAll(self, node: ForAll) -> None:
                self.visit(node.stmt)

        visitor = WorkspaceGetter()
        visitor.visit(self)
        return visitor.workspaces

    def get_lhs_workspaces(self) -> List[Workspace]:
        class WorkspaceGetter(CINVisitorAccept):
            workspaces: List[Workspace] = []

            def visit_Workspace(self, node: Workspace) -> None:
                self.workspaces.append(node)

            def visit_TensorAssign(self, node: TensorAssign) -> None:
                self.visit(node.lhs)

            def visit_Where(self, node: Where) -> None:
                self.visit(node.consumer)

            def visit_ForAll(self, node: ForAll) -> None:
                self.visit(node.stmt)

        visitor = WorkspaceGetter()
        visitor.visit(self)
        return visitor.workspaces

    def get_workspaces(self) -> List[Workspace]:
        workspaces = self.get_lhs_workspaces() + self.get_rhs_workspaces()
        # Remove duplicates
        return list(set(workspaces))

    def get_workspace_accesses(self) -> List[WorkspaceAccess]:
        class WorkspaceAccessGetter(CINVisitorAccept):
            workspace_accesses: List[WorkspaceAccess] = []

            def visit_WorkspaceAccess(self, node: WorkspaceAccess) -> None:
                self.workspace_accesses.append(node)

            def visit_TensorAssign(self, node: TensorAssign) -> None:
                self.visit(node.lhs)
                self.visit(node.rhs)

            def visit_Where(self, node: Where) -> None:
                self.visit(node.producer)
                self.visit(node.consumer)

            def visit_ForAll(self, node: ForAll) -> None:
                self.visit(node.stmt)

        visitor = WorkspaceAccessGetter()
        visitor.visit(self)
        return visitor.workspace_accesses

    def get_index_vars(self) -> List[IndexVar]:
        class IndexVarCollector(CINVisitorAccept):
            index_vars: List[IndexVar] = []

            def visit_IndexVar(self, node: IndexVar) -> None:
                if node not in self.index_vars:
                    self.index_vars.append(node)

        visitor = IndexVarCollector()
        visitor.visit(self)
        return visitor.index_vars

    def get_tile_size_vars(self) -> List[TileSizeVar]:
        index_vars = self.get_index_vars()
        tile_size_vars = []
        for index_var in index_vars:
            if (
                index_var.tile_size_var
                and index_var.tile_size_var not in tile_size_vars
            ):
                tile_size_vars.append(index_var.tile_size_var)

        return tile_size_vars


class IndexVar(IndexExpr):
    """A tensor index variable.
    e.g. i, j, k, ...
    An index variable is bound to a set of coordinates by a forall statement.
    """

    is_tiled: bool = False
    is_outer: bool = False
    is_inner: bool = False
    tile_size_var: Optional[TileSizeVar] = None

    def __init__(
        self,
        name: str,
        expr: Optional[IndexVarExpr] = None,
        parent: Optional[IndexVar] = None,
    ):
        super().__init__()
        self._name = name
        self._expr = expr
        self._parent = parent
        # if expr, then set parent of expr to self
        if expr:
            expr.set_parent(self)

    @property
    def name(self) -> str:
        assert self._name is not None, "IndexVar name is None"
        return self._name

    @property
    def expr(self) -> IndexVarExpr:
        assert self._expr is not None, "IndexVar expr is None"
        return self._expr

    def get_resolve_llir_stmts(self) -> List[llir.Stmt]:
        if isinstance(self.expr, IndexVarAdd):
            # int self.name = lhs.name + rhs.name
            return [
                llir.VarInit(
                    var=llir.Var(self.name, llir.DataType.INT),
                    value=llir.Add(
                        left=llir.Var(self.expr.lhs.name, llir.DataType.INT),
                        right=llir.Var(self.expr.rhs.name, llir.DataType.INT),
                    ),
                )
            ]

    @property
    def parent(self) -> IndexVar:
        assert self._parent is not None, "IndexVar parent is None"
        return self._parent

    @property
    def has_parent(self) -> bool:
        return self._parent is not None

    @parent.setter
    def parent(self, parent: IndexVar) -> None:
        self._parent = parent

    def set_tile_size_var(
        self,
        tile_size_var: TileSizeVar,
        is_outer: Optional[bool] = None,
        is_inner: Optional[bool] = None,
    ) -> None:
        self.tile_size_var = tile_size_var
        self.parent.is_tiled = True
        if is_outer is not None:
            self.is_outer = is_outer
        if is_inner is not None:
            self.is_inner = is_inner

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"ivar_{self.name}"

    def __eq__(self, other):
        if isinstance(other, IndexVar):
            return self.name == other.name
        return False

    def __copy__(self):
        return IndexVar(deepcopy(self.name))

    def accept(self, visitor: "CINVisitor") -> None:
        return

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other) -> IndexVarExpr:
        return IndexVarAdd(self, other)


@dataclass(frozen=False)
class TileSizeVar(IndexExpr):
    outer_index_var: IndexVar
    inner_index_var: IndexVar
    size: int
    _name: Optional[str] = None
    _index_var: Optional[IndexVar] = None

    @property
    def name(self) -> str:
        assert self._name is not None, "TileSizeVar name is None"
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def index_var(self) -> IndexVar:
        return (
            self._index_var
            or self.inner_index_var.parent
            or self.outer_index_var.parent
        )

    @index_var.setter
    def index_var(self, index_var: IndexVar) -> None:
        self._index_var = index_var

    @property
    def llir_var(self) -> llir.Var:
        return llir.Var(self.name, llir.DataType.CONSTEXPR_INT)

    @property
    def llir_var_init(self) -> llir.VarInit:
        return llir.VarInit(var=self.llir_var, value=llir.Literal(self.size))

    def __post_init__(self):
        # if _name is not given, set it to f"kTile_{index_var.name}"
        if self._name is None:
            self._name = f"kTile_{self.index_var.name}"
        self.outer_index_var.set_tile_size_var(self, is_outer=True)
        self.inner_index_var.set_tile_size_var(self, is_inner=True)


class IndexVarExpr(IndexExpr):
    """An expression involving index variables.
    e.g. 2 * i + j
    """

    def set_parent(self, parent: IndexVar) -> None:
        raise NotImplementedError


class IndexVarAdd(IndexVarExpr):
    """A sum of two index variables.
    e.g. i + j
    """

    def __init__(self, lhs: IndexVar, rhs: IndexVar):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def get_lhs(self) -> IndexVar:
        return self.lhs

    def get_rhs(self) -> IndexVar:
        return self.rhs

    def set_parent(self, parent: IndexVar) -> None:
        self.lhs.parent = parent
        self.rhs.parent = parent

    def __str__(self):
        return f"{self.lhs} + {self.rhs}"

    def __repr__(self):
        return f"ivar_add_{self.lhs}_{self.rhs}"

    def __eq__(self, other):
        if isinstance(other, IndexVarAdd):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __copy__(self):
        return IndexVarAdd(deepcopy(self.lhs), deepcopy(self.rhs))

    def __hash__(self):
        return hash((self.lhs, self.rhs))


class TensorVar(IndexExpr):
    """A tensor variable.
    Tensors are maps from coordinates to scalar values.
    Tensors are only ever used in access expressions, where they are indexed by
    an index variable at each mode.
    Can be either an operand or a result.
    e.g. A, B, C, ...
    """

    _name: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    format: Optional[TensorFormat] = None
    dtype: torch.dtype = torch.float32

    def __init__(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        fmt: Optional[Union[TensorFormat, str, List[str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._name = name
        self.shape = shape

        if fmt:
            self.format = parse_format(fmt)

        self.dtype = dtype

    @property
    def name(self) -> str:
        assert self._name is not None, "TensorVar name is None"
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        assert self.name is not None, "TensorVar name is None"
        return self.name

    def get_format(self) -> TensorFormat:
        assert self.format is not None, "TensorVar format is None"
        return self.format

    def get_level_types(self) -> List[LevelType]:
        return self.get_format().get_level_types()

    @property
    def levels(self) -> int:
        return len(self.get_level_types())

    def is_dense(self) -> bool:
        return self.get_format().is_dense()

    def get_level_size_cpp(self, level: int) -> str:
        return f"(int) ({self.name}._shape[{level}])"

    def __getitem__(self, item) -> TensorAccess:
        return TensorAccess(self, item)

    def __setitem__(self, key, value) -> None:
        """
        Set self._assignment to the processed node.
        """
        self._assignment = TensorAssign(TensorAccess(self, key), value)

    def __str__(self):
        return f"{self.name}:{self.format}"
        # return f'TensorVar("{self.name}", fmt={self.format})'
        # return f"TensorVar(name={self.name}, shape={self.shape}, format={self.format})"

    def __repr__(self):
        return f"tensor_{self.name}:{self.format}"

    def accept(self, visitor: CINVisitor) -> None:
        return


class Workspace(TensorVar):
    name: Optional[str] = None
    dim: int

    def __init__(
        self,
        name: Optional[str] = None,
        dim: int = 1,
        dtype: torch.dtype = torch.float32,
        dense: bool = False,
        tile_size_var: Optional[TileSizeVar] = None,
    ):
        super().__init__()
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.dense = dense
        self._tile_size_var = tile_size_var

    @property
    def is_tiled(self) -> bool:
        return self._tile_size_var is not None

    @property
    def tile_size_var(self) -> TileSizeVar:
        assert self._tile_size_var is not None, "Workspace is not tiled"
        return self._tile_size_var

    def get_format(self) -> TensorFormat:
        return parse_format(["o"] * self.dim)

    def is_dense(self) -> bool:
        return self.dense

    def __str__(self):
        return f"{self.name}({self.dim})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, item) -> WorkspaceAccess:
        return WorkspaceAccess(self, item)

    # Default workspace access for a 0-dimensional/scalar workspace
    def get_default_access(self) -> WorkspaceAccess:
        return WorkspaceAccess(self)


class TensorAccess(IndexExpr):
    """
    A tensor access in the taco IR.
    e.g. A[i, j]
    Access expressions are returned when calling the overloaded
    __getitem__ method on a Tensor.
    Access expressions can be assigned an expression using the overloaded
    __setitem__ method, which happens when they are on the left hand side
    of an assignment.
    """

    def __init__(
        self,
        tensor: TensorVar,
        indices: Union[IndexVar, List[IndexVar]],
    ):
        # TODO
        super(TensorAccess, self).__init__()

        self.tensor = tensor
        if isinstance(indices, IndexVar):
            indices = [indices]
        self.indices = indices

    def is_workspace(self) -> bool:
        return isinstance(self.tensor, Workspace)

    def get_tensor(self) -> TensorVar:
        return self.tensor

    def get_index_vars(self) -> List[IndexVar]:
        return self.indices

    def has_index_var(self, index_var: IndexVar) -> bool:
        return self.indices and index_var in self.indices

    def get_parent_index_var(self, index_var: IndexVar) -> Optional[IndexVar]:
        index_var_index = self.indices.index(index_var)
        return self.indices[index_var_index - 1] if index_var_index > 0 else None

    def level_of_index_var(self, index: IndexVar) -> int:
        return self.indices.index(index)

    def level_type_of_index_var(self, index: IndexVar) -> LevelType:
        return self.tensor.get_level_types()[self.level_of_index_var(index)]

    def parent_level_type_of_index_var(self, index: IndexVar) -> Optional[LevelType]:
        parent_index = self.get_parent_index_var(index)
        if parent_index is None:
            return None
        return self.level_type_of_index_var(parent_index)

    def child_level_type_of_index_var(self, index: IndexVar) -> Optional[LevelType]:
        index_level = self.level_of_index_var(index)
        if index_level == self.tensor.levels - 1:
            return None
        return self.tensor.get_level_types()[index_level + 1]

    def level_types(self) -> List[LevelType]:
        return self.tensor.get_level_types()

    @property
    def num_levels(self) -> int:
        return len(self.indices)

    def get_level_iterator_resolve_stmts(
        self,
        level: Optional[int] = None,
        index_var: Optional[IndexVar] = None,
    ) -> List[llir.Stmt]:
        assert (
            level is not None or index_var is not None
        ), "Either level or index_var must be specified"
        if index_var:
            level = self.level_of_index_var(index_var)
        elif level is not None:
            index_var = self.indices[level]

        level_type = self.level_type_of_index_var(index_var)
        tensor_name = self.tensor.name

        if level_type == LevelType.DENSE:
            if level > 0:
                parent_level = level - 1
                parent_index_var = self.get_parent_index_var(index_var)
                coord_var_llir = llir.Var(
                    name=f"p{tensor_name}{level}",
                    type=llir.DataType.INT,
                )
                # int p{name}{level} = p{name}{level-1} * {name}{level}_size + <index var>
                return [
                    llir.VarInit(
                        var=coord_var_llir,
                        value=llir.Add(
                            left=llir.Mul(
                                left=llir.Var(
                                    name=f"p{tensor_name}{parent_level}",
                                    type=llir.DataType.INT,
                                ),
                                right=llir.Var(
                                    name=f"{tensor_name}{level}_size",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            right=llir.Var(name=index_var.name, type=llir.DataType.INT),
                        ),
                    )
                ]

    def __getitem__(self, index) -> TensorAccess:
        return TensorAccess(self.tensor, self.indices + [index])

    def __str__(self):
        return f"{self.tensor}[{', '.join([str(i) for i in self.indices])}]"
        # return f"TensorAccess(tensor={self.tensor}, indices={self.indices})"

    def __repr__(self):
        return str(self)

    def accept(self, visitor: CINVisitor) -> None:
        visitor.visit(self.tensor)
        for index in self.indices:
            visitor.visit(index)


class WorkspaceAccess(TensorAccess):
    def __init__(
        self,
        wksp: Workspace,
        indices: Optional[Union[IndexVar, List[IndexVar]]] = None,
    ):
        super().__init__(wksp, indices)
        self.wksp: Workspace = wksp

    def is_dense(self) -> bool:
        return self.wksp.is_dense()

    def accept(self, visitor: CINVisitor) -> None:
        visitor.visit(self.tensor)

    def __str__(self):
        return f"{self.tensor}[{self.indices}]"
        # return f"TensorAccess(tensor={self.tensor}, indices={self.indices})"


class Operation(Enum):
    """
    All operations supported by taco.
    e.g. +, -, *, /
    """

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class OpExpr(IndexExpr):
    """
    An operation in the taco IR.
    e.g. A[i, j] = B[i, j] + C[i, j]
    We may have UnaryOp, BinaryOp, TernaryOperation, etc.
    """

    op: Operation


@dataclass(frozen=True)
class UnaryOp(OpExpr):
    """
    A unary expression in the taco IR.
    """

    expr: IndexExpr

    def __str__(self):
        return f"{self.op}({self.expr})"

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class BinaryOp(OpExpr):
    """
    A binary expression in the taco IR.
    """

    op: Operation
    left: IndexExpr
    right: IndexExpr

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self):
        return str(self)

    def accept(self, visitor: "CINVisitor") -> None:
        visitor.visit(self.left)
        visitor.visit(self.right)


class TensorAssign(IndexStmt):
    """
    TensorAssign :=
        | TensorAccess = IndexExpr
        | TensorAccess `op`= IndexExpr
    Can specify an optional operation `op` that turns the assignment
    into an update / compound assignment, e.g. A[i, j] += B[i, j]
    """

    lhs: TensorAccess
    rhs: IndexExpr
    op: Optional[Operation] = None

    def __init__(
        self,
        lhs: TensorAccess,
        rhs: IndexExpr,
        op: Optional[Operation] = None,
    ):
        # TODO
        super(TensorAssign, self).__init__(lhs, rhs)
        self.op = op

    def get_lhs(self) -> TensorAccess:
        return self.lhs

    def get_lhs_tensor(self) -> TensorVar:
        return self.lhs.get_tensor()

    def get_rhs(self) -> IndexExpr:
        return self.rhs

    def __str__(self):
        return f"{self.lhs} {self.op or ''}= {self.rhs}"
        # return f"TensorAssign(lhs={self.lhs}, rhs={self.rhs})"

    def __repr__(self):
        return f"{self.lhs} {self.op or ''}= {self.rhs}"

    def accept(self, visitor: "CINVisitor") -> None:
        visitor.visit(self.lhs)
        visitor.visit(self.rhs)


class ForAll(IndexStmt):
    """
    A forall statement binds an index variable to a set of index values
    (its range) and executes a statement for each index value in the range.

    The range can be inferred from the tensor modes indexed by the index
    variable, and therefore can be omitted.

    A forall statement does not define an execution order.

    ForAll := ForAll_{IndexVar} IndexStmt

    e.g. forall_(i) A[i] = B[i] * C[i]
    """

    def __init__(self, index_var: IndexVar, stmt: IndexStmt):
        # TODO
        super(ForAll, self).__init__(None, None)
        self.index_var = index_var
        self.stmt = stmt

    def get_index_var(self) -> IndexVar:
        return self.index_var

    def __str__(self):
        return f"∀{{{self.index_var}}} ({self.stmt})"

    def __repr__(self):
        return f"ForAll_{{{self.index_var}}} ({self.stmt})"

    def accept(self, visitor: "CINVisitor") -> None:
        visitor.visit(self.index_var)
        visitor.visit(self.stmt)


@dataclass(frozen=True)
class Where(IndexStmt):
    """
    A where statement involves a producer statement and a consumer statement.
    The producer statement binds a tensor variable in the environment of the
    consumer statement.
    """

    consumer: IndexStmt
    producer: IndexStmt

    def __str__(self):
        return f"where ({self.consumer}) ({self.producer})"

    def __repr__(self):
        return str(self)

    def accept(self, visitor: CINVisitor) -> None:
        visitor.visit(self.consumer)
        visitor.visit(self.producer)

    def get_workspaces(self) -> List[Workspace]:
        class WorkspaceGetter(CINVisitorAccept):
            workspaces: List[Workspace] = []

            def visit_Workspace(self, node: Workspace) -> None:
                self.workspaces.append(node)

            def visit_Where(self, node: Where) -> None:
                self.visit(node.producer)

            def visit_ForAll(self, node: ForAll) -> None:
                self.visit(node.stmt)

        visitor = WorkspaceGetter()
        visitor.visit(self)
        return visitor.workspaces


class CINVisitor:
    def visit(self, node: CIN) -> None:
        method_name = "visit_" + node.__class__.__name__
        visitor: Callable[[CIN], None] = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def generic_visit(self, node: CIN) -> None:
        raise Exception(f"No visit_{node.__class__.__name__} method")


class CINVisitorAccept(CINVisitor):
    # A CIN visitor that make non-implemented nodes call their accept method
    # with the visitor as an argument.
    def generic_visit(self, node: CIN) -> None:
        node.accept(self)


class CINIndexVariablesGetter(CINVisitorAccept):
    # free variables are index variables of the result tensor
    free_vars: List[IndexVar] = []
    # input variables are index variables of the input tensors
    input_vars: List[IndexVar] = []

    def __init__(self, stmt: Optional[IndexStmt] = None):
        self.free_vars = []
        self.input_vars = []
        if stmt is not None:
            self.visit(stmt)

    def visit_TensorAssign(self, node: TensorAssign) -> None:
        lhs_index_vars = node.get_lhs().get_index_vars()
        if lhs_index_vars:
            for var in node.get_lhs().get_index_vars():
                self.free_vars.append(var)
        self.visit(node.get_rhs())

    def visit_ForAll(self, node: ForAll) -> None:
        ivar = node.get_index_var()
        if ivar not in self.input_vars:
            self.input_vars.append(ivar)
        self.visit(node.stmt)

    def visit_BinaryOp(self, node: BinaryOp) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_TensorAccess(self, node: TensorAccess) -> None:
        for var in node.get_index_vars():
            if var not in self.input_vars:
                self.input_vars.append(var)

    def get_reduction_vars(self) -> List[IndexVar]:
        return [var for var in self.input_vars if var not in self.free_vars]

    def get_free_vars(self) -> List[IndexVar]:
        return self.free_vars


class LoopOrderGetter(CINVisitor):
    index_vars_ordered: List[IndexVar] = []
    free_vars: List[IndexVar] = []

    def __init__(self, stmt: Optional[CIN] = None):
        self.index_vars_ordered = []
        self.free_vars = []
        if stmt is not None:
            self.visit(stmt)

    def visit_TensorAssign(self, node: TensorAssign) -> None:
        for var in node.get_lhs().get_index_vars():
            self.free_vars.append(var)

    def visit_ForAll(self, node: ForAll) -> None:
        self.index_vars_ordered.append(node.index_var)
        self.visit(node.stmt)


def all_free_var_loops_before_reduction_loops(stmt: IndexStmt) -> bool:
    """
    Free variables are index variables that index into the result tensor.

    This function returns true if all ForAll loops over free variables come
    before the reduction loops.
    """
    loop_order_getter = LoopOrderGetter(stmt)
    index_vars_ordered = loop_order_getter.index_vars_ordered
    free_vars = loop_order_getter.free_vars
    free_var_loops = [var for var in index_vars_ordered if var in free_vars]
    return free_var_loops == index_vars_ordered[: len(free_var_loops)]
