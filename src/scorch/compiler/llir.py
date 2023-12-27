from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Union, TypeVar, Sequence

import torch

"""
TODO: maybe need this, maybe not
Enum class for different IRNode types.
Literal, Var, Neg, Sqrt,
Add, Sub, Mul, Div, Mod, Rem,
Min, Max,
And, Or, Not,
BitAnd, BitOr,
Eq, Ne, Lt, Le, Gt, Ge,
IfThenElse, For, While, Block,
Function, Call,
VarAssign, VarDecl,
Yield, Allocate, Free, Comment, BlankLine, Print, GetTensorProperty,
Continue, Sort,
Cast, Case, Switch, Load, Malloc, Sizeof, Store, Scope
"""

LLIR_STMT = TypeVar("LLIR_STMT", bound="Stmt")


class Node:
    def accept(self, visitor):
        """Dispatches the visitor to a node."""
        visitor.visit(self)


class Expr(Node):
    """Base class for all expressions."""

    pass


class Stmt(Node):
    """Base class for all statements."""

    pass


class TensorProperty(Enum):
    """Tensor properties."""

    INDICES = "indices"
    VALUES = "values"
    NAME = "name"
    SHAPE = "shape"
    DTYPE = "dtype"


class GetTensorProperty(Expr):
    """Get tensor property."""

    def __init__(
        self,
        tensor: Expr,
        tensor_property: TensorProperty,
        level: int = 0,
        index: int = 0,
        name: Optional[str] = None,
    ):
        self.tensor = tensor
        self.tensor_property = tensor_property
        self.level = level
        self.index = index
        self.name = name


class AssignOp(Enum):
    """Assignment operators."""

    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="
    MOD_ASSIGN = "%="
    REM_ASSIGN = "%="
    AND_ASSIGN = "&="
    OR_ASSIGN = "|="
    XOR_ASSIGN = "^="
    SHL_ASSIGN = "<<="
    SHR_ASSIGN = ">>="


class DataType(Enum):
    """
    All possible data type of a variable in C++.
    """

    AUTO = "auto"
    INT = "int"
    BOOL = "bool"
    UINT8 = "uint8_t"
    INT8 = "int8_t"
    UINT16 = "uint16_t"
    INT16 = "int16_t"
    UINT32 = "uint32_t"
    INT32 = "int32_t"
    UINT64 = "uint64_t"
    INT64 = "int64_t"
    FLOAT32 = "float"
    FLOAT64 = "double"
    VOID = "void"
    STRING = "std::string"
    TORCH_TENSOR = "torch::Tensor"
    TORCH_FLOAT32 = "torch::kFloat32"
    TORCH_FLOAT64 = "torch::kFloat64"
    TORCH_INT32 = "torch::kInt32"
    TORCH_INT64 = "torch::kInt64"
    TORCH_INT8 = "torch::kInt8"
    TORCH_UINT8 = "torch::kUInt8"
    TACO_TENSOR = "Tensor"
    NO_TYPE = "NO_TYPE"
    CVECTOR_INT = "cvector<int>"
    CVECTOR_INT64 = "cvector<int64_t>"
    CVECTOR_FLOAT32 = "cvector<float>"
    CVECTOR_TORCH_FLOAT32 = "cvector<torch::kFloat32>"

    COO_WORKSPACE_INT = "coo_workspace<int>"
    COO_WORKSPACE_FLOAT32 = "coo_workspace<float>"
    COO_WORKSPACE_TORCH_FLOAT32 = "coo_workspace<torch::kFloat32>"
    # "coo_workspace<type, dim_size>"
    COO_WORKSPACE_INT_1 = "coo_workspace<int, 1>"
    COO_WORKSPACE_INT_2 = "coo_workspace<int, 2>"
    COO_WORKSPACE_INT_3 = "coo_workspace<int, 3>"
    COO_WORKSPACE_INT_4 = "coo_workspace<int, 4>"
    COO_WORKSPACE_INT_5 = "coo_workspace<int, 5>"
    COO_WORKSPACE_FLOAT32_1 = "coo_workspace<float, 1>"
    COO_WORKSPACE_FLOAT32_2 = "coo_workspace<float, 2>"
    COO_WORKSPACE_FLOAT32_3 = "coo_workspace<float, 3>"
    COO_WORKSPACE_FLOAT32_4 = "coo_workspace<float, 4>"
    COO_WORKSPACE_FLOAT32_5 = "coo_workspace<float, 5>"

    STD_VECTOR_INT = "std::vector<int>"
    STD_VECTOR_2D_TORCH_TENSOR = "std::vector<std::vector<torch::Tensor>>"
    ARRAY_INT = "int[]"

    # Pointer types
    PTR_INT = "int*"
    PTR_INT_32 = "int32_t*"
    PTR_INT_64 = "int64_t*"
    PTR_FLOAT32 = "float*"
    PTR_TORCH_FLOAT32 = "torch::kFloat32*"
    PTR_TORCH_FLOAT64 = "torch::kFloat64*"
    PTR_TORCH_INT32 = "torch::kInt32*"
    PTR_TORCH_INT64 = "torch::kInt64*"
    PTR_TORCH_INT8 = "torch::kInt8*"
    PTR_TORCH_UINT8 = "torch::kUInt8*"
    PTR_TORCH_TENSOR = "torch::Tensor*"
    PTR_TENSOR = "Tensor*"
    PTR_VOID = "void*"

    CONST_AUTO_REF = "const auto&"

    @classmethod
    def cvector_type(cls, dtype: DataType) -> DataType:
        """
        A custom vector type for C++.
        """
        return DataType(f"cvector<{dtype.value}>")

    @classmethod
    def coo_workspace_type(cls, dtype: DataType) -> DataType:
        """
        A custom vector type for C++.
        """
        return DataType(f"coo_workspace<{dtype.value}>")

    @classmethod
    def coo_workspace_type_with_dim(cls, dtype: DataType, dim: int) -> DataType:
        """
        A custom vector type for C++.
        """
        # if dimension is 0, then simply return the scalar type
        if dim == 0:
            return dtype
        return DataType(f"coo_workspace<{dtype.value}, {dim}>")

    @classmethod
    # pointer type, e.g. int*, float*, etc.
    def ptr_type(cls, dtype: Union[DataType, torch.dtype]) -> DataType:
        if isinstance(dtype, DataType):
            return DataType(f"{dtype.value}*")
        elif isinstance(dtype, torch.dtype):
            data_type = DataType.from_dtype(dtype)
            return DataType(f"{data_type.value}*")

    @classmethod
    def from_dtype(cls, dtype: torch.dtype):
        if dtype == torch.int:
            return cls.INT
        elif dtype == torch.float32:
            return cls.FLOAT32
        elif dtype == torch.float64:
            return cls.FLOAT64
        elif dtype == torch.int32:
            return cls.INT32
        elif dtype == torch.int64:
            return cls.INT64
        elif dtype == torch.int8:
            return cls.INT8
        elif dtype == torch.uint8:
            return cls.UINT8
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

    @classmethod
    def from_python_type(cls, py_type):
        if py_type == bool:
            return cls.BOOL
        elif py_type == int:
            return cls.INT32
        elif py_type == float:
            return cls.FLOAT32
        elif py_type == str:
            return cls.STRING
        else:
            raise NotImplementedError(f"Unsupported type: {py_type}")


"""
Expression nodes
"""


@dataclass(frozen=False)
class Var(Expr):
    """A variable reference."""

    name: str
    type: DataType
    is_ptr: bool = False


class UnaryOp(Expr):
    """Base class for all unary operations."""

    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand


class BinOp(Expr):
    """Base class for all binary operations."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right


class Add(BinOp):
    """Addition."""

    def __init__(self, left: Expr, right: Expr):
        super().__init__("+", left, right)


class Mul(BinOp):
    """Multiplication."""

    def __init__(self, left: Expr, right: Expr):
        super().__init__("*", left, right)


@dataclass(frozen=False)
class Literal(Expr):
    """A literal value."""

    value: Any
    data_type: Optional[DataType] = None

    def __post_init__(self):
        if self.data_type is None:
            self.data_type = DataType.from_python_type(type(self.value))


"""
Statement nodes
"""


class Increment(Stmt):
    """Increment a variable."""

    def __init__(self, var: Var):
        self.var = var


class Return(Stmt):
    """A return statement."""

    def __init__(self, value: Expr):
        self.value = value


@dataclass(frozen=False)
class VarDecl(Stmt):
    """A variable declaration statement."""

    var: Var


@dataclass(frozen=False)
class VarInit(Stmt):
    """A variable initialization statement.
    Declares a variable and assigns a value to it.
    """

    var: Var
    value: Expr
    op: str = "="
    cast: Optional[bool] = False

    def __post_init__(self):
        if self.cast:
            self.value = Cast(self.value, self.var.type)


@dataclass(frozen=False)
class Assign(Stmt):
    """A variable assignment statement."""

    var: Union[Var, Expr]
    value: Expr
    op: AssignOp = AssignOp.ASSIGN
    cast: Optional[bool] = False

    def __post_init__(self):
        if self.cast:
            assert isinstance(self.var, Var)
            self.value = Cast(self.value, self.var.type)


class Allocate(Stmt):
    """Allocate memory for a pointer variable."""

    def __init__(
        self,
        var: Expr,
        num_elements: Expr,
        is_realloc: bool = False,
        use_calloc: bool = False,
    ):
        self.var = var
        self.num_elements = num_elements
        self.is_realloc = is_realloc
        self.use_calloc = use_calloc


class Free(Stmt):
    """Free memory for a pointer variable."""

    def __init__(self, var: Expr):
        self.var = var


class Print(Stmt):
    """A print statement."""

    def __init__(self, value: Expr):
        self.value = value


class Comment(Stmt):
    """A comment statement."""

    def __init__(self, value: str):
        self.value = value


class BlankLine(Stmt):
    """A blank line statement."""

    def __init__(self):
        pass


class Continue(Stmt):
    """A continue statement."""

    def __init__(self):
        pass


class Break(Stmt):
    """A break statement."""

    def __init__(self):
        pass


class Function(Stmt):
    """
    A function definition.
    return_type, function name, argument variables
    """

    def __init__(
        self,
        return_type: DataType,
        name: str,
        args: Sequence[Expr],
        body: List[Stmt],
    ):
        self.return_type: DataType = return_type
        self.name = name
        self.args = args
        self.body = body


class FunctionCall(Expr):
    """A function call expression."""

    def __init__(self, name: str, args: List[Expr] = []):
        self.name = name
        self.args = args


class FunctionCallStmt(Stmt):
    def __init__(self, name: str, args: List[Expr]):
        self.name = name
        self.args = args


class Array(Expr):
    """An array expression."""

    def __init__(self, values: List[Expr], data_type: DataType):
        self.values = values
        self.data_type = data_type


@dataclass(frozen=False)
class ArrayAccess(Expr):
    """An array access expression."""

    array: Expr
    index: Expr


class ForLoop(Stmt):
    """A for loop statement in C/C++."""

    def __init__(
        self,
        init: Optional[Union[VarInit, VarDecl]],
        cond: Expr,
        update: Union[Increment, VarInit, FunctionCall],
        body: List[Stmt],
    ):
        self.init = init
        self.cond = cond
        self.update = update
        self.body = body


# A for loop styled for (auto XXX : YYY) { ... }
@dataclass(frozen=False)
class ForLoopAuto(Stmt):
    """A for loop statement in C/C++."""

    var: Var
    array: Expr
    body: List[Stmt]


class WhileLoop(Stmt):
    """A while loop statement in C/C++."""

    def __init__(self, cond: Expr, body: List[Stmt]):
        self.cond = cond
        self.body = body


@dataclass(frozen=False)
class IfThenElse(Stmt):
    """An if-then-else statement in C/C++."""

    cond: Optional[Expr] = None
    then_body: Optional[List[Stmt]] = None
    else_body: Optional[List[Stmt]] = None

    cond_list: Optional[List[Expr]] = None
    then_body_list: Optional[List[List[Stmt]]] = None

    make_last_case_else: bool = False


class Case(Stmt):
    """A case statement in C/C++."""

    def __init__(self, cond: Expr, body: List[Stmt]):
        self.cond = cond
        self.body = body


class Switch(Stmt):
    """A switch statement in C/C++."""

    def __init__(self, cond: Expr, cases: List[Case], default: List[Stmt]):
        self.cond = cond
        self.cases = cases
        self.default = default


@dataclass(frozen=True)
class Cast(Expr):
    """A cast expression."""

    expr: Expr
    data_type: DataType


@dataclass(frozen=True)
class Sizeof(Expr):
    """A sizeof expression."""

    data_type: DataType


# Node visitor
class NodeVisitor:
    def visit(self, node: Node):
        """Dispatches the visitor to a node."""
        node_name = type(node).__name__
        method_name = "visit_" + node_name
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node):
        """Called if no explicit visitor function exists for a node."""
        raise Exception("No visit_{} method".format(type(node).__name__))


class CppCodeGenerator(NodeVisitor):
    def __init__(self):
        self.indent_level = 0
        self.indent_string = " " * 4
        self.code = ""

    def add_line(self, line: str):
        self.code += self.indent_string * self.indent_level + line + "\n"

    def visit(self, node: Node):
        """Dispatches the visitor to a node."""
        node_name = type(node).__name__
        method_name = "generate_" + node_name
        visitor = getattr(self, method_name, self.generic_generate)
        return visitor(node)

    def generic_generate(self, node: Node):
        """Called if no explicit visitor function exists for a node."""
        raise Exception("No generate_{} method".format(type(node).__name__))

    def generate_BinOp(self, node: BinOp):
        self.add_line(f"({self.visit(node.left)} {node.op} {self.visit(node.right)})")

    def generate_UnaryOp(self, node: UnaryOp):
        self.add_line(f"{node.op}({self.visit(node.operand)})")

    def generate_Return(self, node: Return):
        self.add_line(f"return {self.visit(node.value)};")
