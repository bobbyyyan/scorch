from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Union, TypeVar, Sequence, Tuple, cast

import torch

from .identity import AccessId, IndexId, SymbolId

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
Yield, Comment, BlankLine, Continue, Sort,
Cast, Load, Malloc, Sizeof, Store, Scope
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
    LONG = "long"
    SIZE_T = "size_t"
    FLOAT32 = "float"
    FLOAT64 = "double"
    VOID = "void"
    STRING = "std::string"
    TORCH_TENSOR = "torch::Tensor"
    TORCH_SCALAR_TYPE = "torch::ScalarType"
    TORCH_FLOAT32 = "torch::kFloat32"
    TORCH_FLOAT64 = "torch::kFloat64"
    TORCH_INT32 = "torch::kInt32"
    TORCH_INT64 = "torch::kInt64"
    TORCH_INT8 = "torch::kInt8"
    TORCH_UINT8 = "torch::kUInt8"
    TACO_TENSOR = "Tensor"
    NO_TYPE = "NO_TYPE"
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
    COO_WORKSPACE_FLOAT64_1 = "coo_workspace<double, 1>"
    COO_WORKSPACE_FLOAT64_2 = "coo_workspace<double, 2>"
    COO_WORKSPACE_FLOAT64_3 = "coo_workspace<double, 3>"
    COO_WORKSPACE_FLOAT64_4 = "coo_workspace<double, 4>"
    COO_WORKSPACE_FLOAT64_5 = "coo_workspace<double, 5>"
    COO_WORKSPACE_INT32_1 = "coo_workspace<int32_t, 1>"
    COO_WORKSPACE_INT32_2 = "coo_workspace<int32_t, 2>"
    COO_WORKSPACE_INT32_3 = "coo_workspace<int32_t, 3>"
    COO_WORKSPACE_INT32_4 = "coo_workspace<int32_t, 4>"
    COO_WORKSPACE_INT32_5 = "coo_workspace<int32_t, 5>"
    COO_WORKSPACE_INT64_1 = "coo_workspace<int64_t, 1>"
    COO_WORKSPACE_INT64_2 = "coo_workspace<int64_t, 2>"
    COO_WORKSPACE_INT64_3 = "coo_workspace<int64_t, 3>"
    COO_WORKSPACE_INT64_4 = "coo_workspace<int64_t, 4>"
    COO_WORKSPACE_INT64_5 = "coo_workspace<int64_t, 5>"
    COO_WORKSPACE_INT8_1 = "coo_workspace<int8_t, 1>"
    COO_WORKSPACE_INT8_2 = "coo_workspace<int8_t, 2>"
    COO_WORKSPACE_INT8_3 = "coo_workspace<int8_t, 3>"
    COO_WORKSPACE_INT8_4 = "coo_workspace<int8_t, 4>"
    COO_WORKSPACE_INT8_5 = "coo_workspace<int8_t, 5>"
    COO_WORKSPACE_UINT8_1 = "coo_workspace<uint8_t, 1>"
    COO_WORKSPACE_UINT8_2 = "coo_workspace<uint8_t, 2>"
    COO_WORKSPACE_UINT8_3 = "coo_workspace<uint8_t, 3>"
    COO_WORKSPACE_UINT8_4 = "coo_workspace<uint8_t, 4>"
    COO_WORKSPACE_UINT8_5 = "coo_workspace<uint8_t, 5>"

    STD_VECTOR_INT = "std::vector<int64_t>"
    STD_VECTOR_C_INT = "std::vector<int>"
    STD_VECTOR_INT32 = "std::vector<int32_t>"
    STD_VECTOR_FLOAT32 = "std::vector<float>"
    STD_VECTOR_FLOAT64 = "std::vector<double>"
    STD_VECTOR_INT8 = "std::vector<int8_t>"
    STD_VECTOR_UINT8 = "std::vector<uint8_t>"
    STD_VECTOR_TORCH_TENSOR = "std::vector<torch::Tensor>"
    STD_VECTOR_2D_TORCH_TENSOR = "std::vector<std::vector<torch::Tensor>>"
    ARRAY_INT = "int[]"

    # Owned per-worker sparse workspace pools, one member per recognized
    # production scalar spelling of the pooled linked-list workspace.
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_FLOAT32 = (
        "std::vector<linked_list_workspace_1d<float>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_FLOAT64 = (
        "std::vector<linked_list_workspace_1d<double>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_INT = (
        "std::vector<linked_list_workspace_1d<int>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_INT32 = (
        "std::vector<linked_list_workspace_1d<int32_t>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_INT64 = (
        "std::vector<linked_list_workspace_1d<int64_t>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_LONG_LONG = (
        "std::vector<linked_list_workspace_1d<long long>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_INT8 = (
        "std::vector<linked_list_workspace_1d<int8_t>>"
    )
    STD_VECTOR_LINKED_LIST_WORKSPACE_1D_UINT8 = (
        "std::vector<linked_list_workspace_1d<uint8_t>>"
    )

    # Pointer types
    PTR_INT = "int*"
    PTR_INT_32 = "int32_t*"
    PTR_INT_64 = "int64_t*"
    PTR_INT8 = "int8_t*"
    PTR_UINT8 = "uint8_t*"
    PTR_FLOAT32 = "float*"
    PTR_FLOAT64 = "double*"

    # Const-qualified pointers to the recognized production scalar
    # spellings, for read-only input borrows (the D1 hoisted declarations).
    CONST_PTR_INT = "const int*"
    CONST_PTR_INT32 = "const int32_t*"
    CONST_PTR_INT64 = "const int64_t*"
    CONST_PTR_INT8 = "const int8_t*"
    CONST_PTR_UINT8 = "const uint8_t*"
    CONST_PTR_FLOAT32 = "const float*"
    CONST_PTR_FLOAT64 = "const double*"
    PTR_TORCH_FLOAT32 = "torch::kFloat32*"
    PTR_TORCH_FLOAT64 = "torch::kFloat64*"
    PTR_TORCH_INT32 = "torch::kInt32*"
    PTR_TORCH_INT64 = "torch::kInt64*"
    PTR_TORCH_INT8 = "torch::kInt8*"
    PTR_TORCH_UINT8 = "torch::kUInt8*"
    PTR_TORCH_TENSOR = "torch::Tensor*"
    PTR_TENSOR = "Tensor*"
    PTR_VOID = "void*"

    # CONSTEXPR types
    CONSTEXPR_INT = "constexpr int"

    CONST_AUTO_REF = "const auto&"

    @classmethod
    def std_vector_type(cls, dtype: DataType) -> DataType:
        """
        A standard-library vector with the requested C++ element type.
        """
        return DataType(f"std::vector<{dtype.value}>")

    @classmethod
    def coo_workspace_type(cls, dtype: DataType) -> DataType:
        """
        A custom vector type for C++.
        """
        return DataType(f"coo_workspace<{dtype.value}>")

    @classmethod
    def linked_list_workspace_pool_type(cls, element_spelling: str) -> DataType:
        """The owned per-worker linked-list workspace pool for one scalar spelling.

        Enum lookup by value fails closed with :class:`ValueError` for any
        spelling without a dedicated pool member, so free-form legacy C type
        text cannot silently become a typed declaration.
        """
        if type(element_spelling) is not str:
            raise TypeError("linked-list workspace element spelling must be a string")
        return DataType(f"std::vector<linked_list_workspace_1d<{element_spelling}>>")

    @classmethod
    def const_ptr_type(cls, element_spelling: str) -> DataType:
        """The const-qualified pointer for one recognized scalar spelling.

        Enum lookup by value fails closed with :class:`ValueError` for any
        spelling without a dedicated const-pointer member, so free-form
        legacy C type text cannot silently become a typed declaration.
        """
        if type(element_spelling) is not str:
            raise TypeError("const-pointer element spelling must be a string")
        return DataType(f"const {element_spelling}*")

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


_DIRECT_INIT_DATA_TYPES = frozenset(
    {
        DataType.STD_VECTOR_INT,
        DataType.STD_VECTOR_C_INT,
        DataType.STD_VECTOR_INT32,
        DataType.STD_VECTOR_FLOAT32,
        DataType.STD_VECTOR_FLOAT64,
        DataType.STD_VECTOR_INT8,
        DataType.STD_VECTOR_UINT8,
        DataType.STD_VECTOR_TORCH_TENSOR,
        DataType.STD_VECTOR_2D_TORCH_TENSOR,
    }
)


class TensorAccessRole(Enum):
    """The logical role of a tensor value access emitted from CIN."""

    INPUT_READ = "input_read"
    RESULT_WRITE = "result_write"


@dataclass(frozen=True)
class TensorAccessMetadata:
    """Immutable CIN provenance for a generated tensor value access.

    Stable IDs preserve the logical access independently of generated C++
    spelling.  ``access_id`` records occurrence provenance; transformations that
    select every access to a logical tensor/index tuple match ``tensor_id`` and
    ``index_ids`` instead.
    """

    access_id: AccessId
    tensor_id: SymbolId
    index_ids: Tuple[IndexId, ...]
    role: TensorAccessRole

    def __post_init__(self) -> None:
        if type(self.access_id) is not AccessId:
            raise TypeError("TensorAccessMetadata.access_id must be an AccessId")
        if type(self.tensor_id) is not SymbolId:
            raise TypeError("TensorAccessMetadata.tensor_id must be a SymbolId")
        if type(self.index_ids) is not tuple or any(
            type(index_id) is not IndexId for index_id in self.index_ids
        ):
            raise TypeError(
                "TensorAccessMetadata.index_ids must be a tuple of IndexId values"
            )
        if type(self.role) is not TensorAccessRole:
            raise TypeError("TensorAccessMetadata.role must be a TensorAccessRole")


"""
Expression nodes
"""


@dataclass(frozen=False)
class Var(Expr):
    """A variable reference."""

    name: str
    type: DataType
    is_ptr: bool = False
    is_restrict: bool = False
    tensor_access: Optional[TensorAccessMetadata] = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __hash__(self):
        return hash(self.name)


class UnaryOp(Expr):
    """Base class for all unary operations."""

    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand


@dataclass(frozen=True, repr=False)
class BinOp(Expr):
    """An immutable binary expression with structural value semantics."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if type(self.op) is not str or not self.op:
            raise TypeError("BinOp.op must be a non-empty string")
        if not isinstance(self.left, Expr):
            raise TypeError("BinOp.left must be an LLIR Expr")
        if not isinstance(self.right, Expr):
            raise TypeError("BinOp.right must be an LLIR Expr")


@dataclass(frozen=True, repr=False, init=False)
class Add(BinOp):
    """Addition."""

    def __init__(self, left: Expr, right: Expr):
        super().__init__("+", left, right)


@dataclass(frozen=True, repr=False, init=False)
class Mul(BinOp):
    """Multiplication."""

    def __init__(self, left: Expr, right: Expr):
        super().__init__("*", left, right)


def rebuild_binary_expression(
    expression: BinOp,
    left: Expr,
    right: Expr,
) -> BinOp:
    """Rebuild an exact supported binary node with detached child expressions."""

    if type(expression) is BinOp:
        return BinOp(expression.op, left, right)
    if type(expression) is Add:
        if expression.op != "+":
            raise TypeError("Add.op must remain '+'")
        return Add(left, right)
    if type(expression) is Mul:
        if expression.op != "*":
            raise TypeError("Mul.op must remain '*'")
        return Mul(left, right)
    raise TypeError("binary expression must be an exact BinOp, Add, or Mul instance")


@dataclass(frozen=True)
class Literal(Expr):
    """An immutable primitive literal with structural value semantics."""

    value: Any
    data_type: Optional[DataType] = None

    def __post_init__(self) -> None:
        if type(self.value) not in (bool, int, float, str):
            raise TypeError("Literal.value must be a bool, int, float, or string")
        if self.data_type is None:
            object.__setattr__(
                self,
                "data_type",
                DataType.from_python_type(type(self.value)),
            )
        elif type(self.data_type) is not DataType:
            raise TypeError("Literal.data_type must be a DataType or None")


@dataclass(frozen=True, repr=False)
class QualifiedName(Expr):
    """An immutable two-component qualified C++ name expression."""

    namespace: str
    name: str
    data_type: DataType

    def __post_init__(self) -> None:
        if type(self.namespace) is not str or not self.namespace.isidentifier():
            raise TypeError("QualifiedName.namespace must be a non-empty identifier")
        if type(self.name) is not str or not self.name.isidentifier():
            raise TypeError("QualifiedName.name must be a non-empty identifier")
        if type(self.data_type) is not DataType:
            raise TypeError("QualifiedName.data_type must be a DataType")


"""
Statement nodes
"""


@dataclass(frozen=True)
class Increment(Stmt):
    """An immutable post-increment of one exact variable reference."""

    var: Var

    def __post_init__(self) -> None:
        if type(self.var) is not Var:
            raise TypeError("Increment.var must be an exact LLIR Var")


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

    def __hash__(self):
        return hash(self.var)


@dataclass(frozen=True, init=False)
class DirectInit(Stmt):
    """A typed C++ direct-initialization declaration.

    ``args`` must be nonempty so this node cannot accidentally emit the C++
    most-vexing-parse form ``T value()``. Use :class:`VarDecl` for an
    uninitialized declaration.
    """

    var: Var
    args: Tuple[Expr, ...]

    def __init__(self, var: Var, args: Sequence[Expr]) -> None:
        if type(var) is not Var:
            raise TypeError("DirectInit.var must be an exact LLIR Var")
        if type(var.name) is not str or not var.name.isidentifier():
            raise TypeError("DirectInit.var.name must be a non-empty identifier")
        if type(var.type) is not DataType or var.type not in _DIRECT_INIT_DATA_TYPES:
            raise TypeError(
                "DirectInit.var.type must be a supported standard-vector DataType"
            )
        if var.is_ptr is not False:
            raise TypeError("DirectInit.var.is_ptr must be False")
        if var.is_restrict is not False:
            raise TypeError("DirectInit.var.is_restrict must be False")
        if var.tensor_access is not None:
            raise TypeError("DirectInit.var.tensor_access must be None")
        if type(args) is not list and type(args) is not tuple:
            raise TypeError("DirectInit.args must be a list or tuple")
        if not args:
            raise TypeError("DirectInit.args must be non-empty")
        if any(not isinstance(argument, Expr) for argument in args):
            raise TypeError("DirectInit.args must contain only LLIR expressions")
        object.__setattr__(self, "var", var)
        object.__setattr__(self, "args", tuple(args))


@dataclass(frozen=False)
class Assign(Stmt):
    """An assignment to a scalar/member variable or structured subscript."""

    var: "AssignmentTarget"
    value: Expr
    op: AssignOp = AssignOp.ASSIGN
    cast: bool = False

    def __post_init__(self) -> None:
        _validate_assignment_target(self.var)
        if not isinstance(self.value, Expr):
            raise TypeError("Assign.value must be an LLIR Expr")
        if type(self.op) is not AssignOp:
            raise TypeError("Assign.op must be an AssignOp")
        if type(self.cast) is not bool:
            raise TypeError("Assign.cast must be a bool")
        if self.cast:
            if type(self.var) is not Var:
                raise TypeError("Assign.cast requires an exact Var target")
            self.value = Cast(self.value, self.var.type)


class Comment(Stmt):
    """A comment statement."""

    def __init__(self, value: str):
        self.value = value


class BlankLine(Stmt):
    """A blank line statement."""

    def __init__(self):
        pass


@dataclass(frozen=False)
class RawStmt(Stmt):
    """A raw statement emitted verbatim."""

    code: str
    add_semicolon: bool = True


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


def _normalized_call_template_args(
    template_args: Optional[Sequence[DataType]],
    *,
    owner: str,
) -> Tuple[DataType, ...]:
    """Validate and freeze one optional call template-argument sequence."""

    if template_args is None:
        return ()
    if type(template_args) is not list and type(template_args) is not tuple:
        raise TypeError(f"{owner}.template_args must be a list or tuple")
    if any(type(argument) is not DataType for argument in template_args):
        raise TypeError(f"{owner}.template_args must contain only DataType values")
    return tuple(template_args)


@dataclass(frozen=True, init=False, repr=False)
class FunctionCall(Expr):
    """An immutable function call with tuple-owned expression arguments.

    ``template_args`` mirrors :class:`MemberCall` exactly: explicit template
    arguments are typed ``DataType`` values, never spellings embedded in the
    call name.
    """

    name: str
    template_args: Tuple[DataType, ...]
    args: Tuple[Expr, ...]

    def __init__(
        self,
        name: str,
        args: Optional[Sequence[Expr]] = None,
        template_args: Optional[Sequence[DataType]] = None,
    ) -> None:
        if type(name) is not str or not name.strip():
            raise TypeError("FunctionCall.name must be a non-empty string")
        normalized_template_args = _normalized_call_template_args(
            template_args, owner="FunctionCall"
        )
        if args is None:
            normalized_args: Tuple[Expr, ...] = ()
        else:
            if type(args) is not list and type(args) is not tuple:
                raise TypeError("FunctionCall.args must be a list or tuple")
            if any(not isinstance(argument, Expr) for argument in args):
                raise TypeError("FunctionCall.args must contain only LLIR expressions")
            normalized_args = tuple(args)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "template_args", normalized_template_args)
        object.__setattr__(self, "args", normalized_args)


@dataclass(frozen=True, init=False, repr=False)
class FunctionCallStmt(Stmt):
    """An immutable call statement with tuple-owned expression arguments."""

    name: str
    template_args: Tuple[DataType, ...]
    args: Tuple[Expr, ...]

    def __init__(
        self,
        name: str,
        args: Optional[Sequence[Expr]] = None,
        template_args: Optional[Sequence[DataType]] = None,
    ) -> None:
        if type(name) is not str or not name.strip():
            raise TypeError("FunctionCallStmt.name must be a non-empty string")
        normalized_template_args = _normalized_call_template_args(
            template_args, owner="FunctionCallStmt"
        )
        if args is None:
            normalized_args: Tuple[Expr, ...] = ()
        else:
            if type(args) is not list and type(args) is not tuple:
                raise TypeError("FunctionCallStmt.args must be a list or tuple")
            if any(not isinstance(argument, Expr) for argument in args):
                raise TypeError(
                    "FunctionCallStmt.args must contain only LLIR expressions"
                )
            normalized_args = tuple(args)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "template_args", normalized_template_args)
        object.__setattr__(self, "args", normalized_args)


@dataclass(frozen=True, init=False, repr=False)
class Array(Expr):
    """An immutable braced initializer-list expression."""

    values: Tuple[Expr, ...]
    data_type: DataType

    def __init__(self, values: Sequence[Expr], data_type: DataType) -> None:
        if type(values) is not list and type(values) is not tuple:
            raise TypeError("Array.values must be a list or tuple")
        if any(not isinstance(value, Expr) for value in values):
            raise TypeError("Array.values must contain only LLIR expressions")
        if type(data_type) is not DataType:
            raise TypeError("Array.data_type must be a DataType")
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "data_type", data_type)


_FIXED_STACK_ARRAY_ELEMENT_TYPES = frozenset(
    {
        DataType.FLOAT32,
        DataType.FLOAT64,
        DataType.INT32,
        DataType.INT64,
        DataType.INT8,
        DataType.UINT8,
    }
)
_FIXED_STACK_ARRAY_LITERAL_EXTENT_TYPES = frozenset(
    {
        DataType.INT,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT32,
        DataType.UINT64,
    }
)


def _is_fixed_stack_array_extent(value: object) -> bool:
    """Whether ``value`` is one exact supported fixed array extent."""

    if type(value) is Var:
        extent = cast(Var, value)
        return bool(
            type(extent.name) is str
            and extent.name.isidentifier()
            and extent.type is DataType.CONSTEXPR_INT
            and extent.is_ptr is False
            and extent.is_restrict is False
            and extent.tensor_access is None
        )
    if type(value) is Literal:
        extent_literal = cast(Literal, value)
        return bool(
            type(extent_literal.value) is int
            and extent_literal.value > 0
            and type(extent_literal.data_type) is DataType
            and extent_literal.data_type in _FIXED_STACK_ARRAY_LITERAL_EXTENT_TYPES
        )
    return False


@dataclass(frozen=True)
class FixedStackArrayDecl(Stmt):
    """One fixed-size, empty-brace-initialized automatic C++ array.

    This deliberately narrow transitional statement keeps declaration/storage
    separate from :class:`ArrayAccess`.  It does not represent dynamic extents,
    heap storage, multidimensional arrays, alignment, or non-empty initializers.
    """

    name: str
    element_type: DataType
    extent: Expr
    initializer: Array

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.isidentifier():
            raise TypeError("FixedStackArrayDecl.name must be a non-empty identifier")
        if (
            type(self.element_type) is not DataType
            or self.element_type not in _FIXED_STACK_ARRAY_ELEMENT_TYPES
        ):
            raise TypeError(
                "FixedStackArrayDecl.element_type must be a supported scalar "
                "DataType"
            )
        if not _is_fixed_stack_array_extent(self.extent):
            raise TypeError(
                "FixedStackArrayDecl.extent must be an exact metadata-free "
                "constexpr Var or positive integral Literal"
            )
        if type(self.initializer) is not Array:
            raise TypeError("FixedStackArrayDecl.initializer must be an exact Array")
        if type(self.initializer.values) is not tuple:
            raise TypeError("FixedStackArrayDecl.initializer values must be a tuple")
        if self.initializer.values:
            raise TypeError("FixedStackArrayDecl.initializer must be empty")
        if self.initializer.data_type is not self.element_type:
            raise TypeError(
                "FixedStackArrayDecl.initializer type must match element_type"
            )


@dataclass(frozen=True)
class MemberAccess(Expr):
    """An immutable typed dot-member access expression."""

    base: Expr
    member: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, Expr):
            raise TypeError("MemberAccess.base must be an LLIR Expr")
        if type(self.member) is not str or not self.member.isidentifier():
            raise TypeError("MemberAccess.member must be a non-empty identifier")


@dataclass(frozen=True, init=False, repr=False)
class MemberCall(Expr):
    """An immutable call to one member of a structured receiver expression."""

    base: Expr
    member: str
    template_args: Tuple[DataType, ...]
    args: Tuple[Expr, ...]

    def __init__(
        self,
        base: Expr,
        member: str,
        template_args: Optional[Sequence[DataType]] = None,
        args: Optional[Sequence[Expr]] = None,
    ) -> None:
        if not isinstance(base, Expr):
            raise TypeError("MemberCall.base must be an LLIR Expr")
        if type(member) is not str or not member.isidentifier():
            raise TypeError("MemberCall.member must be a non-empty identifier")
        if template_args is None:
            normalized_template_args: Tuple[DataType, ...] = ()
        else:
            if type(template_args) is not list and type(template_args) is not tuple:
                raise TypeError("MemberCall.template_args must be a list or tuple")
            if any(type(argument) is not DataType for argument in template_args):
                raise TypeError(
                    "MemberCall.template_args must contain only DataType values"
                )
            normalized_template_args = tuple(template_args)
        if args is None:
            normalized_args: Tuple[Expr, ...] = ()
        else:
            if type(args) is not list and type(args) is not tuple:
                raise TypeError("MemberCall.args must be a list or tuple")
            if any(not isinstance(argument, Expr) for argument in args):
                raise TypeError("MemberCall.args must contain only LLIR expressions")
            normalized_args = tuple(args)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "member", member)
        object.__setattr__(self, "template_args", normalized_template_args)
        object.__setattr__(self, "args", normalized_args)


@dataclass(frozen=True, init=False, repr=False)
class MemberCallStmt(Stmt):
    """An immutable member-call statement on one structured receiver.

    This mirrors :class:`MemberCall` exactly as :class:`FunctionCallStmt`
    mirrors :class:`FunctionCall`: the receiver is a structured expression
    child, never a dotted spelling embedded in a call name.
    """

    base: Expr
    member: str
    template_args: Tuple[DataType, ...]
    args: Tuple[Expr, ...]

    def __init__(
        self,
        base: Expr,
        member: str,
        template_args: Optional[Sequence[DataType]] = None,
        args: Optional[Sequence[Expr]] = None,
    ) -> None:
        if not isinstance(base, Expr):
            raise TypeError("MemberCallStmt.base must be an LLIR Expr")
        if type(member) is not str or not member.isidentifier():
            raise TypeError("MemberCallStmt.member must be a non-empty identifier")
        if template_args is None:
            normalized_template_args: Tuple[DataType, ...] = ()
        else:
            if type(template_args) is not list and type(template_args) is not tuple:
                raise TypeError("MemberCallStmt.template_args must be a list or tuple")
            if any(type(argument) is not DataType for argument in template_args):
                raise TypeError(
                    "MemberCallStmt.template_args must contain only DataType values"
                )
            normalized_template_args = tuple(template_args)
        if args is None:
            normalized_args: Tuple[Expr, ...] = ()
        else:
            if type(args) is not list and type(args) is not tuple:
                raise TypeError("MemberCallStmt.args must be a list or tuple")
            if any(not isinstance(argument, Expr) for argument in args):
                raise TypeError(
                    "MemberCallStmt.args must contain only LLIR expressions"
                )
            normalized_args = tuple(args)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "member", member)
        object.__setattr__(self, "template_args", normalized_template_args)
        object.__setattr__(self, "args", normalized_args)


@dataclass(frozen=True)
class ArrayAccess(Expr):
    """An immutable typed array/subscript access expression.

    Structural equality describes emitted expression shape.  Optional logical
    provenance is deliberately excluded, as it is for :class:`Var`; consumers
    that need semantic identity must compare its stable typed IDs explicitly.
    """

    array: Expr
    index: Expr
    tensor_access: Optional[TensorAccessMetadata] = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.array, Expr):
            raise TypeError("ArrayAccess.array must be an LLIR Expr")
        if not isinstance(self.index, Expr):
            raise TypeError("ArrayAccess.index must be an LLIR Expr")
        if (
            self.tensor_access is not None
            and type(self.tensor_access) is not TensorAccessMetadata
        ):
            raise TypeError(
                "ArrayAccess.tensor_access must be TensorAccessMetadata or None"
            )


AssignmentTarget = Union[Var, MemberAccess, ArrayAccess]


def _is_assignment_name(name: object, *, allow_member: bool) -> bool:
    """Whether ``name`` is in the narrow identifier/member lvalue grammar."""

    if type(name) is not str or not name:
        return False
    if name.isidentifier():
        return True
    if not allow_member:
        return False
    return all(component.isidentifier() for component in name.split("."))


def _validate_assignment_metadata(
    metadata: object,
    *,
    expected_role: TensorAccessRole,
    owner: str,
) -> None:
    if type(metadata) is not TensorAccessMetadata:
        raise TypeError(f"{owner} metadata must be TensorAccessMetadata")
    missing = _MISSING_ADDRESS_OF_FIELD
    fields = _stored_instance_fields(metadata)
    if fields.get("role", missing) is not expected_role:
        raise TypeError(f"{owner} metadata must have the {expected_role.name} role")
    if type(fields.get("access_id", missing)) is not AccessId:
        raise TypeError(f"{owner} metadata access_id must be an AccessId")
    if type(fields.get("tensor_id", missing)) is not SymbolId:
        raise TypeError(f"{owner} metadata tensor_id must be a SymbolId")
    index_ids = fields.get("index_ids", missing)
    if type(index_ids) is not tuple or any(
        type(index_id) is not IndexId for index_id in index_ids
    ):
        raise TypeError(f"{owner} metadata index_ids must be a tuple of IndexId values")


def _assignment_index_fast_ok(expression: object, budget: int) -> int:
    """Cheaply recognize stored index states the strict validator accepts.

    Returns the remaining depth budget for a provably valid subtree and ``-1``
    for anything uncertain — member-path names, unary/cast/call/sizeof forms,
    metadata-bearing accesses, forged state, or an exhausted budget (which any
    forged cycle produces).  A ``-1`` never rejects: callers rerun the
    authoritative stored-field validator, which accepts the slow-but-valid
    shapes and raises the exact boundary diagnostics for the rest.
    """

    if budget <= 0:
        return -1
    expression_type = type(expression)
    fields = getattr(expression, "__dict__", None)
    if type(fields) is not dict:
        return -1
    if expression_type is Var:
        name = fields.get("name")
        if (
            type(name) is str
            and name.isidentifier()
            and type(fields.get("type")) is DataType
            and type(fields.get("is_ptr")) is bool
            and type(fields.get("is_restrict")) is bool
            and fields.get("tensor_access", False) is None
        ):
            return budget
        return -1
    if expression_type in (BinOp, Add, Mul):
        op = fields.get("op")
        if (
            type(op) is not str
            or op not in ("+", "-", "*", "/", "%")
            or (expression_type is Add and op != "+")
            or (expression_type is Mul and op != "*")
        ):
            return -1
        budget = _assignment_index_fast_ok(fields.get("left"), budget - 1)
        if budget < 0:
            return -1
        return _assignment_index_fast_ok(fields.get("right"), budget - 1)
    if expression_type is Literal:
        if (
            type(fields.get("value")) is int
            and type(fields.get("data_type")) is DataType
        ):
            return budget
        return -1
    if expression_type is ArrayAccess:
        if fields.get("tensor_access", False) is not None:
            return -1
        budget = _assignment_index_fast_ok(fields.get("array"), budget - 1)
        if budget < 0:
            return -1
        return _assignment_index_fast_ok(fields.get("index"), budget - 1)
    return -1


def _validate_assignment_index(expression: object) -> None:
    """Validate the typed expression subset admitted as a subscript index.

    The grammar is the shared stored-field, cycle-hardened index validator;
    only the diagnostic owner prefixes differ from the AddressOf boundary.
    A constructor-hot-path recognizer short-circuits provably valid common
    shapes first.
    """

    if _assignment_index_fast_ok(expression, 64) >= 0:
        return
    _validate_address_of_index(
        expression,
        path=(),
        active=set(),
        owner="assignment index",
        metadata_owner="assignment index ArrayAccess",
    )


def _validate_assignment_target(target: object) -> None:
    """Validate the deliberately small lvalue subset owned by ``Assign``.

    Every field is read from stored instance state and member chains are
    cycle-checked, so forged dataclass state fails with a controlled
    boundary error instead of an attribute fallback, hang, or recursion
    blowup.
    """

    missing = _MISSING_ADDRESS_OF_FIELD
    if type(target) is Var:
        fields = _stored_instance_fields(target)
        if not _is_assignment_name(fields.get("name", missing), allow_member=True):
            raise TypeError("assignment Var.name must be an identifier or member path")
        if type(fields.get("type", missing)) is not DataType:
            raise TypeError("assignment Var.type must be a DataType")
        if fields.get("tensor_access", missing) is not None:
            raise TypeError(
                "scalar/member Assign targets cannot carry tensor access metadata"
            )
        return
    if type(target) is ArrayAccess:
        # Constructor hot path: provably valid common subscript stores return
        # immediately; anything uncertain falls through to the authoritative
        # stored-field checks below.
        fast_fields = getattr(target, "__dict__", None)
        if type(fast_fields) is dict:
            fast_array = fast_fields.get("array")
            if type(fast_array) is Var:
                array_fast_fields = getattr(fast_array, "__dict__", None)
                if type(array_fast_fields) is not dict:
                    array_fast_fields = _EMPTY_INSTANCE_FIELDS
                fast_name = array_fast_fields.get("name")
                if (
                    type(fast_name) is str
                    and fast_name.isidentifier()
                    and type(array_fast_fields.get("type")) is DataType
                    and array_fast_fields.get("tensor_access", False) is None
                    and _assignment_index_fast_ok(fast_fields.get("index"), 64) >= 0
                ):
                    fast_metadata = fast_fields.get("tensor_access", False)
                    if fast_metadata is None:
                        return
                    if type(fast_metadata) is TensorAccessMetadata:
                        _validate_assignment_metadata(
                            fast_metadata,
                            expected_role=TensorAccessRole.RESULT_WRITE,
                            owner="assignment ArrayAccess",
                        )
                        return
    if type(target) is MemberAccess:
        visited_members: set[int] = set()
        base: object = target
        while type(base) is MemberAccess:
            member = cast(MemberAccess, base)
            if id(member) in visited_members:
                raise TypeError("assignment MemberAccess chain must be acyclic")
            visited_members.add(id(member))
            member_fields = _stored_instance_fields(member)
            member_name = member_fields.get("member", missing)
            if type(member_name) is not str or not member_name.isidentifier():
                raise TypeError("assignment MemberAccess members must be identifiers")
            base = member_fields.get("base", missing)
        if type(base) is not Var:
            raise TypeError(
                "assignment MemberAccess must have an exact Var root through exact "
                "MemberAccess bases"
            )
        root_fields = _stored_instance_fields(base)
        if not _is_assignment_name(
            root_fields.get("name", missing), allow_member=False
        ):
            raise TypeError("assignment MemberAccess root name must be an identifier")
        if type(root_fields.get("type", missing)) is not DataType:
            raise TypeError("assignment MemberAccess root type must be a DataType")
        if root_fields.get("tensor_access", missing) is not None:
            raise TypeError(
                "assignment MemberAccess root cannot carry tensor access metadata"
            )
        return
    if type(target) is not ArrayAccess:
        raise TypeError("Assign.var must be an exact Var, MemberAccess, or ArrayAccess")

    access_fields = _stored_instance_fields(target)
    array = access_fields.get("array", missing)
    if type(array) is not Var:
        raise TypeError("assignment ArrayAccess.array must be an exact Var")
    array_fields = _stored_instance_fields(array)
    if not _is_assignment_name(array_fields.get("name", missing), allow_member=True):
        raise TypeError(
            "assignment ArrayAccess.array name must be an identifier or member path"
        )
    if type(array_fields.get("type", missing)) is not DataType:
        raise TypeError("assignment ArrayAccess.array type must be a DataType")
    if array_fields.get("tensor_access", missing) is not None:
        raise TypeError(
            "assignment ArrayAccess.array cannot carry tensor access metadata"
        )
    _validate_assignment_index(access_fields.get("index", missing))
    metadata = access_fields.get("tensor_access", missing)
    if metadata is not None:
        _validate_assignment_metadata(
            metadata,
            expected_role=TensorAccessRole.RESULT_WRITE,
            owner="assignment ArrayAccess",
        )


class ForLoop(Stmt):
    """A for loop statement in C/C++."""

    def __init__(
        self,
        init: Optional[Union[VarInit, VarDecl]],
        cond: Expr,
        update: Union[Increment, FunctionCall, Assign],
        body: List[Stmt],
        omp_parallel_for: bool = False,
        omp_schedule: Optional[str] = None,
        unroll: bool = False,
        simd: bool = False,
        pre_parallel_body: Optional[List[Stmt]] = None,
        post_parallel_body: Optional[List[Stmt]] = None,
        omp_num_threads: Optional[str] = None,
        omp_chunk_expr: Optional[str] = None,
        before_parallel_body: Optional[List[Stmt]] = None,
    ):
        self.init = init
        self.cond = cond
        self.update = update
        self.body = body
        self.omp_parallel_for = omp_parallel_for
        self.omp_schedule = omp_schedule
        self.unroll = unroll
        self.simd = simd
        # Work-aware parallel policy (compiler/codegen.py emits these).
        # omp_num_threads: C++ expr for num_threads(...) (e.g. scorch_nthreads(work,rows)).
        # omp_chunk_expr:  C++ expr for the dynamic schedule chunk; when set the
        #   schedule becomes "dynamic, <omp_chunk_expr>" overriding omp_schedule's chunk.
        self.omp_num_threads = omp_num_threads
        self.omp_chunk_expr = omp_chunk_expr
        # Stable logical-loop identity used by post-CIN schedule lowering.
        self.scorch_index_var: Optional[str] = None
        # Stmts placed before the OpenMP region, or inside it before/after the
        # work loop. Serial pre-region construction lets RAII allocations throw
        # normally instead of terminating while unwinding across OpenMP.
        self.before_parallel_body = before_parallel_body
        self.pre_parallel_body = pre_parallel_body
        self.post_parallel_body = post_parallel_body


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
        self.scorch_index_var: Optional[str] = None


@dataclass(frozen=False)
class IfThenElse(Stmt):
    """An if-then-else statement in C/C++."""

    cond: Optional[Expr] = None
    then_body: Optional[List[Stmt]] = None
    else_body: Optional[List[Stmt]] = None

    cond_list: Optional[List[Expr]] = None
    then_body_list: Optional[List[List[Stmt]]] = None

    make_last_case_else: bool = False


@dataclass(frozen=True)
class Cast(Expr):
    """An immutable typed cast expression."""

    expr: Expr
    data_type: DataType

    def __post_init__(self) -> None:
        if not isinstance(self.expr, Expr):
            raise TypeError("Cast.expr must be an LLIR Expr")
        if type(self.data_type) is not DataType:
            raise TypeError("Cast.data_type must be a DataType")


@dataclass(frozen=True)
class Select(Expr):
    """An immutable C++ conditional selection, spelled ``cond ? a : b``.

    This is the expression-level counterpart of the :class:`IfThenElse`
    statement.  All three children are ordinary expressions; the statement
    form remains the only owner of branch statement lists.
    """

    cond: Expr
    when_true: Expr
    when_false: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.cond, Expr):
            raise TypeError("Select.cond must be an LLIR Expr")
        if not isinstance(self.when_true, Expr):
            raise TypeError("Select.when_true must be an LLIR Expr")
        if not isinstance(self.when_false, Expr):
            raise TypeError("Select.when_false must be an LLIR Expr")


@dataclass(frozen=True)
class Sizeof(Expr):
    """A sizeof expression."""

    data_type: DataType

    def __post_init__(self) -> None:
        if type(self.data_type) is not DataType:
            raise TypeError("Sizeof.data_type must be a DataType")


_MISSING_ADDRESS_OF_FIELD = object()

# Shared read-only stand-in for a missing/forged ``__dict__`` so stored-field
# reads can use one dict fetch per node on validation hot paths.
_EMPTY_INSTANCE_FIELDS: dict = {}


class _AddressOfValidationError(TypeError):
    """A typed lvalue-boundary failure with an operand-relative field path."""

    def __init__(self, message: str, field_path: Tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.field_path = field_path


def _address_of_instance_field(owner: object, name: str) -> object:
    """Read one stored field without falling back to dataclass defaults."""

    instance_fields = getattr(owner, "__dict__", None)
    if type(instance_fields) is not dict or name not in instance_fields:
        return _MISSING_ADDRESS_OF_FIELD
    return instance_fields[name]


def _stored_instance_fields(owner: object) -> dict:
    """All stored fields of one node, or the empty stand-in when forged away."""

    instance_fields = getattr(owner, "__dict__", None)
    if type(instance_fields) is not dict:
        return _EMPTY_INSTANCE_FIELDS
    return instance_fields


def _validate_address_of_metadata(
    metadata: object,
    *,
    path: Tuple[str, ...],
    owner: str = "AddressOf.operand ArrayAccess",
) -> None:
    if metadata is None:
        return
    if type(metadata) is not TensorAccessMetadata:
        raise _AddressOfValidationError(
            f"{owner} metadata must be TensorAccessMetadata or None",
            path,
        )
    typed_metadata = cast(TensorAccessMetadata, metadata)
    if type(_address_of_instance_field(typed_metadata, "access_id")) is not AccessId:
        raise _AddressOfValidationError(
            f"{owner} metadata access_id must be an AccessId",
            path + ("access_id",),
        )
    if type(_address_of_instance_field(typed_metadata, "tensor_id")) is not SymbolId:
        raise _AddressOfValidationError(
            f"{owner} metadata tensor_id must be a SymbolId",
            path + ("tensor_id",),
        )
    index_ids = _address_of_instance_field(typed_metadata, "index_ids")
    if type(index_ids) is not tuple or any(
        type(index_id) is not IndexId for index_id in index_ids
    ):
        raise _AddressOfValidationError(
            f"{owner} metadata index_ids must be a tuple of IndexId values",
            path + ("index_ids",),
        )
    if type(_address_of_instance_field(typed_metadata, "role")) is not TensorAccessRole:
        raise _AddressOfValidationError(
            f"{owner} metadata role must be a TensorAccessRole",
            path + ("role",),
        )


def _validate_address_of_root_var(
    value: object,
    *,
    owner: str,
    path: Tuple[str, ...],
) -> None:
    if type(value) is not Var:
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} must be an exact Var",
            path,
        )
    variable = cast(Var, value)
    name = _address_of_instance_field(variable, "name")
    if type(name) is not str or not name.isidentifier():
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} name must be a non-empty identifier",
            path + ("name",),
        )
    if type(_address_of_instance_field(variable, "type")) is not DataType:
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} type must be a DataType",
            path + ("type",),
        )
    if type(_address_of_instance_field(variable, "is_ptr")) is not bool:
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} is_ptr must be a bool",
            path + ("is_ptr",),
        )
    if type(_address_of_instance_field(variable, "is_restrict")) is not bool:
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} is_restrict must be a bool",
            path + ("is_restrict",),
        )
    if _address_of_instance_field(variable, "tensor_access") is not None:
        raise _AddressOfValidationError(
            f"AddressOf.operand {owner} cannot carry tensor access metadata",
            path + ("tensor_access",),
        )


def _validate_address_of_index(
    expression: object,
    *,
    path: Tuple[str, ...],
    active: set[int],
    owner: str = "AddressOf.operand ArrayAccess index",
    metadata_owner: str = "AddressOf.operand ArrayAccess",
) -> None:
    """Validate one stored, acyclic assignment-index expression tree.

    ``owner``/``metadata_owner`` only parameterize diagnostic prefixes; the
    defaults keep every historical AddressOf message byte-identical while the
    same grammar backs other narrow index-expression boundaries.  The
    positional implementation fetches stored fields once per node and
    brackets only child recursion with cycle bookkeeping (leaves cannot
    participate in a cycle), keeping the shared validator cheap on the
    Assign construction hot path.
    """

    _validate_address_of_index_impl(expression, path, active, owner, metadata_owner)


def _validate_address_of_index_impl(
    expression: object,
    path: Tuple[str, ...],
    active: set[int],
    owner: str,
    metadata_owner: str,
) -> None:
    expression_type = type(expression)
    missing = _MISSING_ADDRESS_OF_FIELD
    fields = getattr(expression, "__dict__", None)
    if type(fields) is not dict:
        fields = _EMPTY_INSTANCE_FIELDS

    if expression_type is Var:
        if not _is_assignment_name(fields.get("name", missing), allow_member=True):
            raise _AddressOfValidationError(
                f"{owner} Var name must be an identifier or member path",
                path + ("name",),
            )
        if type(fields.get("type", missing)) is not DataType:
            raise _AddressOfValidationError(
                f"{owner} Var type must be a DataType",
                path + ("type",),
            )
        if type(fields.get("is_ptr", missing)) is not bool:
            raise _AddressOfValidationError(
                f"{owner} Var is_ptr must be a bool",
                path + ("is_ptr",),
            )
        if type(fields.get("is_restrict", missing)) is not bool:
            raise _AddressOfValidationError(
                f"{owner} Var is_restrict must be a bool",
                path + ("is_restrict",),
            )
        if fields.get("tensor_access", missing) is not None:
            raise _AddressOfValidationError(
                f"{owner} Var cannot carry tensor access metadata",
                path + ("tensor_access",),
            )
        return

    if expression_type is Literal:
        if type(fields.get("value", missing)) is not int:
            raise _AddressOfValidationError(
                f"{owner} Literal value must be an int",
                path + ("value",),
            )
        if type(fields.get("data_type", missing)) is not DataType:
            raise _AddressOfValidationError(
                f"{owner} Literal data_type must be a DataType",
                path + ("data_type",),
            )
        return

    if expression_type is Sizeof:
        if type(fields.get("data_type", missing)) is not DataType:
            raise _AddressOfValidationError(
                f"{owner} Sizeof data_type must be a DataType",
                path + ("data_type",),
            )
        return

    expression_id = id(expression)
    if expression_id in active:
        raise _AddressOfValidationError(
            f"{owner} must be acyclic",
            path,
        )

    if expression_type in (BinOp, Add, Mul):
        op = fields.get("op", missing)
        if (
            type(op) is not str
            or op not in ("+", "-", "*", "/", "%")
            or (expression_type is Add and op != "+")
            or (expression_type is Mul and op != "*")
        ):
            raise _AddressOfValidationError(
                f"{owner} BinOp op must be a supported arithmetic operator",
                path + ("op",),
            )
        active.add(expression_id)
        try:
            _validate_address_of_index_impl(
                fields.get("left", missing),
                path + ("left",),
                active,
                owner,
                metadata_owner,
            )
            _validate_address_of_index_impl(
                fields.get("right", missing),
                path + ("right",),
                active,
                owner,
                metadata_owner,
            )
        finally:
            active.remove(expression_id)
        return

    if expression_type is UnaryOp:
        op = fields.get("op", missing)
        if type(op) is not str or op not in ("+", "-"):
            raise _AddressOfValidationError(
                f"{owner} UnaryOp op must be '+' or '-'",
                path + ("op",),
            )
        active.add(expression_id)
        try:
            _validate_address_of_index_impl(
                fields.get("operand", missing),
                path + ("operand",),
                active,
                owner,
                metadata_owner,
            )
        finally:
            active.remove(expression_id)
        return

    if expression_type is Cast:
        if type(fields.get("data_type", missing)) is not DataType:
            raise _AddressOfValidationError(
                f"{owner} Cast data_type must be a DataType",
                path + ("data_type",),
            )
        active.add(expression_id)
        try:
            _validate_address_of_index_impl(
                fields.get("expr", missing),
                path + ("expr",),
                active,
                owner,
                metadata_owner,
            )
        finally:
            active.remove(expression_id)
        return

    if expression_type is FunctionCall:
        if not _is_assignment_name(fields.get("name", missing), allow_member=True):
            raise _AddressOfValidationError(
                f"{owner} FunctionCall name must be an identifier or member path",
                path + ("name",),
            )
        template_arguments = fields.get("template_args", missing)
        if type(template_arguments) is not tuple or any(
            type(argument) is not DataType for argument in template_arguments
        ):
            raise _AddressOfValidationError(
                f"{owner} FunctionCall template_args must be a tuple of "
                "DataType values",
                path + ("template_args",),
            )
        arguments = fields.get("args", missing)
        if type(arguments) is not tuple:
            raise _AddressOfValidationError(
                f"{owner} FunctionCall args must be a tuple",
                path + ("args",),
            )
        active.add(expression_id)
        try:
            for argument_index, argument in enumerate(arguments):
                _validate_address_of_index_impl(
                    argument,
                    path + ("args", f"[{argument_index}]"),
                    active,
                    owner,
                    metadata_owner,
                )
        finally:
            active.remove(expression_id)
        return

    if expression_type is ArrayAccess:
        metadata = fields.get("tensor_access", missing)
        if metadata is not None:
            _validate_address_of_metadata(
                metadata,
                path=path + ("tensor_access",),
                owner=metadata_owner,
            )
            if (
                _address_of_instance_field(metadata, "role")
                is not TensorAccessRole.INPUT_READ
            ):
                raise _AddressOfValidationError(
                    f"{owner} metadata must have the INPUT_READ role",
                    path + ("tensor_access", "role"),
                )
        active.add(expression_id)
        try:
            _validate_address_of_index_impl(
                fields.get("array", missing),
                path + ("array",),
                active,
                owner,
                metadata_owner,
            )
            _validate_address_of_index_impl(
                fields.get("index", missing),
                path + ("index",),
                active,
                owner,
                metadata_owner,
            )
        finally:
            active.remove(expression_id)
        return

    raise _AddressOfValidationError(
        f"{owner} contains an unsupported LLIR expression",
        path,
    )


def _validate_address_of_operand(operand: object) -> None:
    """Validate the exact syntactic lvalue subset admitted after unary ``&``."""

    if type(operand) is Var:
        _validate_address_of_root_var(operand, owner="Var", path=())
        return

    if type(operand) is MemberAccess:
        member_access = cast(MemberAccess, operand)
        member_path: Tuple[str, ...] = ()
        visited_members: set[int] = set()
        while type(member_access) is MemberAccess:
            if id(member_access) in visited_members:
                raise _AddressOfValidationError(
                    "AddressOf.operand MemberAccess chain must be acyclic",
                    member_path,
                )
            visited_members.add(id(member_access))
            member = _address_of_instance_field(member_access, "member")
            if type(member) is not str or not member.isidentifier():
                raise _AddressOfValidationError(
                    "AddressOf.operand MemberAccess members must be identifiers",
                    member_path + ("member",),
                )
            base = _address_of_instance_field(member_access, "base")
            if type(base) is MemberAccess:
                member_access = cast(MemberAccess, base)
                member_path += ("base",)
                continue
            _validate_address_of_root_var(
                base,
                owner="MemberAccess root",
                path=member_path + ("base",),
            )
            return

    if type(operand) is ArrayAccess:
        access = cast(ArrayAccess, operand)
        _validate_address_of_root_var(
            _address_of_instance_field(access, "array"),
            owner="ArrayAccess root",
            path=("array",),
        )
        index = _address_of_instance_field(access, "index")
        _validate_address_of_index(index, path=("index",), active=set())
        _validate_address_of_metadata(
            _address_of_instance_field(access, "tensor_access"),
            path=("tensor_access",),
        )
        return

    raise _AddressOfValidationError(
        "AddressOf.operand must be an exact Var, MemberAccess, or ArrayAccess",
    )


@dataclass(frozen=True)
class AddressOf(Expr):
    """An immutable address-of expression, spelled ``&operand``."""

    operand: AssignmentTarget

    def __post_init__(self) -> None:
        _validate_address_of_operand(self.operand)


_GUARDED_CALL_COMPARISON_OPS = frozenset({"<", "<=", ">", ">=", "==", "!="})


class _GuardedCallValidationError(TypeError):
    """A guarded-call boundary failure with a node-relative field path."""

    def __init__(self, message: str, field_path: Tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.field_path = field_path


def _validate_guarded_call_condition(
    value: object,
    *,
    path: Tuple[str, ...],
    active: set[int],
) -> None:
    """Validate one stored, acyclic ``&&``-conjunction of index comparisons."""

    value_id = id(value)
    if value_id in active:
        raise _GuardedCallValidationError(
            "GuardedCallStmt.cond must be acyclic",
            path,
        )
    active.add(value_id)
    try:
        if type(value) is not BinOp:
            raise _GuardedCallValidationError(
                "GuardedCallStmt.cond must be an exact comparison or '&&' "
                "conjunction BinOp",
                path,
            )
        op = _address_of_instance_field(value, "op")
        if op == "&&":
            _validate_guarded_call_condition(
                _address_of_instance_field(value, "left"),
                path=path + ("left",),
                active=active,
            )
            _validate_guarded_call_condition(
                _address_of_instance_field(value, "right"),
                path=path + ("right",),
                active=active,
            )
            return
        if type(op) is not str or op not in _GUARDED_CALL_COMPARISON_OPS:
            raise _GuardedCallValidationError(
                "GuardedCallStmt.cond comparisons must use one supported "
                "comparison operator",
                path + ("op",),
            )
        for side in ("left", "right"):
            try:
                _validate_address_of_index(
                    _address_of_instance_field(value, side),
                    path=path + (side,),
                    active=active,
                    owner="GuardedCallStmt.cond comparison operand",
                    metadata_owner="GuardedCallStmt.cond comparison ArrayAccess",
                )
            except _AddressOfValidationError as error:
                raise _GuardedCallValidationError(
                    str(error),
                    error.field_path,
                ) from error
    finally:
        active.remove(value_id)


def _validate_guarded_call_callee(call: object, *, path: Tuple[str, ...]) -> None:
    """Validate the stored single call statement guarded by the condition."""

    if type(call) is not FunctionCallStmt:
        raise _GuardedCallValidationError(
            "GuardedCallStmt.call must be an exact FunctionCallStmt",
            path,
        )
    name = _address_of_instance_field(call, "name")
    if not _is_assignment_name(name, allow_member=True):
        raise _GuardedCallValidationError(
            "GuardedCallStmt.call name must be an identifier or member path",
            path + ("name",),
        )
    template_args = _address_of_instance_field(call, "template_args")
    if type(template_args) is not tuple or any(
        type(argument) is not DataType for argument in template_args
    ):
        raise _GuardedCallValidationError(
            "GuardedCallStmt.call template_args must be a tuple of DataType values",
            path + ("template_args",),
        )
    args = _address_of_instance_field(call, "args")
    if type(args) is not tuple:
        raise _GuardedCallValidationError(
            "GuardedCallStmt.call args must be a tuple",
            path + ("args",),
        )
    for index, argument in enumerate(args):
        argument_path = path + ("args", f"[{index}]")
        if type(argument) is AddressOf:
            try:
                _validate_address_of_operand(
                    _address_of_instance_field(argument, "operand")
                )
            except _AddressOfValidationError as error:
                raise _GuardedCallValidationError(
                    str(error),
                    argument_path + ("operand",) + error.field_path,
                ) from error
            continue
        try:
            _validate_address_of_index(
                argument,
                path=argument_path,
                active=set(),
                owner="GuardedCallStmt.call argument",
                metadata_owner="GuardedCallStmt.call argument ArrayAccess",
            )
        except _AddressOfValidationError as error:
            raise _GuardedCallValidationError(
                str(error),
                error.field_path,
            ) from error


def _validate_guarded_call_fields(node: object) -> None:
    """Revalidate one guarded call from stored fields, never defaults."""

    _validate_guarded_call_condition(
        _address_of_instance_field(node, "cond"),
        path=("cond",),
        active=set(),
    )
    _validate_guarded_call_callee(
        _address_of_instance_field(node, "call"),
        path=("call",),
    )


@dataclass(frozen=True)
class GuardedCallStmt(Stmt):
    """An immutable single-line guarded call, spelled ``if (cond) call;``.

    This deliberately narrow statement owns the legacy inline prefetch-guard
    seams: exactly one call statement guarded by one condition, emitted
    unbraced on a single line.  The condition grammar is restricted to
    ``&&``-conjunctions of comparisons over assignment-index expressions, and
    call arguments are :class:`AddressOf` borrows or the same index grammar.
    :class:`IfThenElse` remains the only owner of general braced conditionals;
    this node does not represent else branches, statement lists, or arbitrary
    guarded statements.
    """

    cond: Expr
    call: FunctionCallStmt

    def __post_init__(self) -> None:
        _validate_guarded_call_fields(self)
