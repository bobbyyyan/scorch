from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union

import torch

from src.scorch.format import TensorFormat, LevelType
from src.scorch.utils import get_format_from_list

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

            def visit_TensorAssign(self, node: "TensorAssign"):
                self.result_tensor_accesses.append(node.lhs)

        collector = ResultTensorAccessCollector()
        self.accept(collector)
        return collector.get_result_tensor_accesses()

    def get_result_tensor_vars(self) -> List["TensorVar"]:
        result_tensor_vars = []
        result_tensor_accesses: List[TensorAccess] = self.get_result_tensor_accesses()
        for tensor_access in result_tensor_accesses:
            result_tensor_vars.append(tensor_access.get_tensor())
        return result_tensor_vars

    def get_rhs_tensor_accesses(self) -> List["TensorAccess"]:
        class RHSAccessCollector(CINVisitorAccept):
            def __init__(self):
                self.rhs_tensor_accesses: List[TensorAccess] = []

            def get_rhs_tensor_accesses(self):
                return self.rhs_tensor_accesses

            def visit_TensorAccess(self, node: "TensorAccess"):
                self.rhs_tensor_accesses.append(node)

            def visit_TensorAssign(self, node: "TensorAssign"):
                self.visit(node.rhs)

        collector = RHSAccessCollector()
        self.accept(collector)
        return collector.get_rhs_tensor_accesses()

    def get_rhs_tensor_vars(self) -> List["TensorVar"]:
        rhs_tensor_vars = []
        rhs_tensor_accesses: List[TensorAccess] = self.get_rhs_tensor_accesses()
        for tensor_access in rhs_tensor_accesses:
            rhs_tensor_vars.append(tensor_access.get_tensor())
        return rhs_tensor_vars


class IndexVar(IndexExpr):
    """A tensor index variable.
    e.g. i, j, k, ...
    An index variable is bound to a set of coordinates by a forall statement.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __str__(self):
        return f"ivar_{self.name}"

    def __repr__(self):
        return str(self)

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


class TensorVar(IndexExpr):
    """A tensor variable.
    Tensors are maps from coordinates to scalar values.
    Tensors are only ever used in access expressions, where they are indexed by
    an index variable at each mode.
    Can be either an operand or a result.
    e.g. A, B, C, ...
    """

    name: Optional[str] = None
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
        self.name = name
        self.shape = shape

        if isinstance(fmt, str):
            fmt = [fmt]

        if isinstance(fmt, TensorFormat):
            self.format = fmt
        elif isinstance(fmt, list):
            self.format = get_format_from_list(fmt)

        self.dtype = dtype

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

    def get_level_size_cpp(self, level: int) -> str:
        return f"(int) ({self.name}._shape[{level}])"

    def __getitem__(self, item) -> "TensorAccess":
        return TensorAccess(self, item)

    def __setitem__(self, key, value) -> None:
        """
        Set self._assignment to the processed node.
        """
        self._assignment = TensorAssign(TensorAccess(self, key), value)

    def __str__(self):
        return f"tensor_{self.name}:{self.format}"
        # return f'TensorVar("{self.name}", fmt={self.format})'
        # return f"TensorVar(name={self.name}, shape={self.shape}, format={self.format})"

    def __repr__(self):
        return str(self)


class TensorAccess(IndexExpr):
    """
    A tensor access in the taco IR.
    e.g. A[i, j]
    Access expressions are returned when calling the overloaded
    __getitem__ method on a TacoTensor.
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

    def get_tensor(self) -> TensorVar:
        return self.tensor

    def get_index_vars(self) -> List[IndexVar]:
        return self.indices

    def get_parent_index_var(self, index_var: IndexVar) -> Optional[IndexVar]:
        index_var_index = self.indices.index(index_var)
        return self.indices[index_var_index - 1] if index_var_index > 0 else None

    def level_of_index_var(self, index: IndexVar) -> int:
        return self.indices.index(index)

    def level_type_of_index_var(self, index: IndexVar) -> LevelType:
        return self.tensor.get_level_types()[self.level_of_index_var(index)]

    def __getitem__(self, index) -> "TensorAccess":
        return TensorAccess(self.tensor, self.indices + [index])

    def __str__(self):
        return f"{self.tensor}[{', '.join([str(i) for i in self.indices])}]"
        # return f"TensorAccess(tensor={self.tensor}, indices={self.indices})"

    def __repr__(self):
        return str(self)


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
        return f"{self.lhs} <- {self.rhs}"
        # return f"TensorAssign(lhs={self.lhs}, rhs={self.rhs})"

    def __repr__(self):
        return str(self)

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
        return f"ForAll_{{{self.index_var}}} ({self.stmt})"

    def __repr__(self):
        return str(self)

    def accept(self, visitor: "CINVisitor") -> None:
        visitor.visit(self.index_var)
        visitor.visit(self.stmt)


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


class LoopOrderGetter(CINVisitor):
    index_vars_ordered: List[IndexVar] = []
    free_vars: List[IndexVar] = []

    def __init__(self, stmt: Optional[IndexStmt] = None):
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
