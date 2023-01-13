from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Union

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


class DataType(Enum):
    """
    All possible data type of a variable in C++.
    """

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
    TACO_TENSOR = "TacoTensor"

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


class Var(Expr):
    """A variable reference."""

    name: str
    type: DataType
    is_ptr: bool = False

    def __init__(
        self, name: str, type: Optional[DataType] = None, is_ptr: bool = False
    ):
        super().__init__()
        self.name = name
        self.type = type
        self.is_ptr = is_ptr

    def __str__(self):
        return f"Var(name={self.name}, type={self.type}, is_ptr={self.is_ptr})"

    def __repr__(self):
        return str(self)


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


class Literal(Expr):
    """A literal value."""

    def __init__(self, value: Any, data_type: Optional[DataType] = None):
        self.value = value
        if data_type is None:
            self.type = DataType.from_python_type(type(value))
        else:
            self.type: DataType = data_type

    def __str__(self):
        return f"Literal(value={self.value}, type={self.type})"

    def __repr__(self):
        return str(self)


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


class VarDecl(Stmt):
    """A variable declaration statement."""

    def __init__(self, name: str, value: Expr):
        self.name = name
        self.value = value


@dataclass(frozen=False)
class VarAssign(Stmt):
    """A variable assignment statement.
    Assigns an expression to a variable.
    """

    var: Var
    value: Expr
    op: str = "="
    cast: Optional[bool] = False

    def __post_init__(self):
        if self.cast:
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
        args: List[Expr],
        body: List[Stmt],
    ):
        self.return_type: DataType = return_type
        self.name = name
        self.args = args
        self.body = body


class FunctionCall(Expr):
    """A function call expression."""

    def __init__(self, name: str, args: List[Expr]):
        self.name = name
        self.args = args


class Array(Expr):
    """An array expression."""

    def __init__(self, values: List[Expr], data_type: DataType):
        self.values = values
        self.data_type = data_type


class ForLoop(Stmt):
    """A for loop statement in C/C++."""

    def __init__(
        self,
        init: Union[VarAssign, VarDecl],
        cond: Expr,
        update: Union[VarAssign, FunctionCall],
        body: List[Stmt],
    ):
        self.init = init
        self.cond = cond
        self.update = update
        self.body = body


class WhileLoop(Stmt):
    """A while loop statement in C/C++."""

    def __init__(self, cond: Expr, body: List[Stmt]):
        self.cond = cond
        self.body = body


@dataclass(frozen=False)
class IfThenElse(Stmt):
    """An if-then-else statement in C/C++."""

    cond: Optional[Union[Expr, List[Expr]]] = None
    then_body: Optional[Union[List[Stmt], List[List[Stmt]]]] = None
    else_body: Optional[List[Stmt]] = None
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

    def generate_VarDecl(self, node: VarDecl):
        self.add_line(f"{node.name} = {self.visit(node.value)};")
