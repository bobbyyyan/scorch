"""Structured Torch/C++ ABI construction shared by compiler lowering stages."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import torch

from . import llir
from ..format import LevelType
from ..utils import (
    dtype_to_c_datatype,
    get_pytorch_c_dtype_name,
    get_pytorch_c_dtype_str,
)


def tensor_storage_member(tensor_name: str, *members: str) -> llir.MemberAccess:
    """Build a fresh nested member path rooted at one generated ``Tensor``."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise TypeError("tensor storage root must be a non-empty identifier")
    if not members:
        raise TypeError("tensor storage member path cannot be empty")
    expression: llir.Expr = llir.Var(
        name=tensor_name,
        type=llir.DataType.TACO_TENSOR,
    )
    for member in members:
        expression = llir.MemberAccess(base=expression, member=member)
    return cast(llir.MemberAccess, expression)


def mode_index_tensor(tensor_name: str, level: int, slot: int) -> llir.ArrayAccess:
    """Build a fresh access to one Torch tensor in the mode-index ABI argument."""

    if type(tensor_name) is not str or not tensor_name.isidentifier():
        raise TypeError("mode-index tensor root must be a non-empty identifier")
    if type(level) is not int or level < 0:
        raise TypeError("mode-index level must be a non-negative int")
    if type(slot) is not int or slot < 0:
        raise TypeError("mode-index slot must be a non-negative int")
    return llir.ArrayAccess(
        array=llir.ArrayAccess(
            array=llir.Var(
                name=f"{tensor_name}_mode_indices",
                type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
            ),
            index=llir.Literal(level, llir.DataType.INT),
        ),
        index=llir.Literal(slot, llir.DataType.INT),
    )


def tensor_data_ptr(receiver: llir.Expr, data_type: llir.DataType) -> llir.MemberCall:
    """Build a typed ``receiver.data_ptr<T>()`` expression."""

    if not isinstance(receiver, llir.Expr):
        raise TypeError("tensor data_ptr receiver must be an LLIR Expr")
    if type(data_type) is not llir.DataType:
        raise TypeError("tensor data_ptr template argument must be a DataType")
    return llir.MemberCall(
        base=receiver,
        member="data_ptr",
        template_args=(data_type,),
    )


@dataclass(frozen=True)
class KernelTensorABI:
    """Immutable ABI metadata for one public JIT tensor argument triple."""

    name: str
    level_types: Tuple[LevelType, ...]
    mode_order: Tuple[int, ...]
    shape: Tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.name) is not str or not self.name.isidentifier():
            raise TypeError("kernel tensor name must be a non-empty identifier")
        if type(self.level_types) is not tuple or any(
            type(level_type) is not LevelType for level_type in self.level_types
        ):
            raise TypeError("kernel tensor levels must be an immutable LevelType tuple")
        supported_levels = (
            LevelType.DENSE,
            LevelType.COMPRESSED,
            LevelType.COORDINATE,
        )
        unsupported_level = next(
            (
                level_type
                for level_type in self.level_types
                if level_type not in supported_levels
            ),
            None,
        )
        if unsupported_level is not None:
            raise ValueError(
                f"unsupported JIT level type {unsupported_level} for ABI validation"
            )
        if type(self.mode_order) is not tuple or any(
            type(level) is not int for level in self.mode_order
        ):
            raise TypeError("kernel tensor mode order must be an immutable int tuple")
        rank = len(self.level_types)
        if len(self.mode_order) != rank or set(self.mode_order) != set(range(rank)):
            raise ValueError(
                "kernel tensor mode order must be a rank-sized permutation"
            )
        if type(self.shape) is not tuple or any(
            type(extent) is not int for extent in self.shape
        ):
            raise TypeError("kernel tensor shape must be an immutable int tuple")
        if self.shape and len(self.shape) != rank:
            raise ValueError("known kernel tensor shape must match the level rank")
        if any(extent < 0 for extent in self.shape):
            raise ValueError("known kernel tensor extents must be non-negative")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("kernel tensor dtype must be a torch.dtype")

    def _level_kind_codes(self) -> Tuple[int, ...]:
        self._validate()
        level_kind_code = {
            LevelType.DENSE: 0,
            LevelType.COMPRESSED: 1,
            LevelType.COORDINATE: 2,
        }
        return tuple(level_kind_code[level] for level in self.level_types)

    def emit_level_array_bindings(self) -> List[llir.Stmt]:
        """Build fresh dense extents and sparse mode-index pointer bindings."""

        self._validate()
        stmts: List[llir.Stmt] = []
        for level, level_type in enumerate(self.level_types):
            if level_type == LevelType.DENSE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_size",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{self.name}_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                            index=llir.Literal(
                                value=level,
                                data_type=llir.DataType.INT64,
                            ),
                        ),
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_pos",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 0),
                            llir.DataType.INT,
                        ),
                    )
                )
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 1),
                            llir.DataType.INT,
                        ),
                    )
                )
            elif level_type == LevelType.COORDINATE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd_tensor",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=mode_index_tensor(self.name, level, 0),
                    )
                )
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=tensor_data_ptr(
                            mode_index_tensor(self.name, level, 0),
                            llir.DataType.INT,
                        ),
                    )
                )
        return stmts

    def emit_value_pointer(self) -> llir.VarInit:
        """Build a fresh typed pointer binding for this tensor's values."""

        self._validate()
        data_type = dtype_to_c_datatype(self.dtype)
        return llir.VarInit(
            var=llir.Var(
                name=f"{self.name}_val",
                type=llir.DataType.ptr_type(self.dtype),
                is_restrict=True,
            ),
            value=tensor_data_ptr(
                llir.Var(
                    name=f"{self.name}_values",
                    type=llir.DataType.TORCH_TENSOR,
                ),
                data_type,
            ),
        )


def _cpp_int_vector(values: Tuple[int, ...]) -> str:
    """Render one validated ABI metadata tuple as a C++ initializer list."""

    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise TypeError("C++ integer vector values must be an immutable int tuple")
    return "{" + ", ".join(str(value) for value in values) + "}"


@dataclass(frozen=True)
class TorchCppKernelABI:
    """Own one frozen public ``evaluate`` signature and input ABI contract."""

    result_shape: Tuple[int, ...]
    result_rank: int
    input_tensors: Tuple[KernelTensorABI, ...]
    extra_tensor_names: Tuple[str, ...] = ()
    extra_tensor_dtype: Optional[torch.dtype] = None
    function_name: str = "evaluate"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.function_name) is not str or not self.function_name.isidentifier():
            raise TypeError("kernel function name must be a non-empty identifier")
        if type(self.result_shape) is not tuple or any(
            type(extent) is not int for extent in self.result_shape
        ):
            raise TypeError("kernel result shape must be an immutable int tuple")
        if type(self.result_rank) is not int or self.result_rank < 0:
            raise TypeError("kernel result rank must be a non-negative int")
        if self.result_shape and len(self.result_shape) != self.result_rank:
            raise ValueError("known kernel result shape must match the result rank")
        if any(extent < 0 for extent in self.result_shape):
            raise ValueError("known kernel result extents must be non-negative")
        if type(self.input_tensors) is not tuple or any(
            type(tensor) is not KernelTensorABI for tensor in self.input_tensors
        ):
            raise TypeError(
                "kernel inputs must be an immutable exact KernelTensorABI tuple"
            )
        for tensor in self.input_tensors:
            tensor._validate()
        if type(self.extra_tensor_names) is not tuple or any(
            type(name) is not str or not name.isidentifier()
            for name in self.extra_tensor_names
        ):
            raise TypeError("extra tensor names must be an immutable identifier tuple")
        if self.extra_tensor_names:
            if not isinstance(self.extra_tensor_dtype, torch.dtype):
                raise TypeError("extra tensor dtype must be a torch.dtype when present")
        elif self.extra_tensor_dtype is not None:
            raise TypeError("extra tensor dtype must be None without extra tensors")
        argument_names = ["result_shape"]
        for tensor in self.input_tensors:
            argument_names.extend(
                (
                    f"{tensor.name}_shape",
                    f"{tensor.name}_mode_indices",
                    f"{tensor.name}_values",
                )
            )
        argument_names.extend(f"{name}_values" for name in self.extra_tensor_names)
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("kernel ABI argument names must be unique")

    def emit_arguments(self) -> List[llir.Var]:
        """Build a fresh ordered public function-argument list."""

        self._validate()
        arguments = [
            llir.Var(
                name="result_shape",
                type=llir.DataType.STD_VECTOR_INT,
            )
        ]
        for tensor in self.input_tensors:
            arguments.extend(
                [
                    llir.Var(
                        name=f"{tensor.name}_shape",
                        type=llir.DataType.STD_VECTOR_INT,
                    ),
                    llir.Var(
                        name=f"{tensor.name}_mode_indices",
                        type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                    ),
                    llir.Var(
                        name=f"{tensor.name}_values",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                ]
            )
        for name in self.extra_tensor_names:
            arguments.append(
                llir.Var(
                    name=f"{name}_values",
                    type=llir.DataType.TORCH_TENSOR,
                )
            )
        return arguments

    def emit_validation(self) -> List[llir.RawStmt]:
        """Build fresh validation calls in exact public-argument order."""

        self._validate()
        validation = [
            llir.RawStmt(
                code=(
                    "scorch_native::validate_jit_result_shape("
                    f"result_shape, {_cpp_int_vector(self.result_shape)}, "
                    f'{self.result_rank}, "{self.function_name}")'
                ),
                add_semicolon=True,
            )
        ]
        for tensor in self.input_tensors:
            validation.append(
                llir.RawStmt(
                    code=(
                        "scorch_native::validate_jit_tensor("
                        f'"{self.function_name}", "{tensor.name}", '
                        f"{tensor.name}_shape, {tensor.name}_mode_indices, "
                        f"{tensor.name}_values, "
                        f"{get_pytorch_c_dtype_str(tensor.dtype)}, "
                        f"{_cpp_int_vector(tensor._level_kind_codes())}, "
                        f"{_cpp_int_vector(tensor.mode_order)}, "
                        f"{_cpp_int_vector(tensor.shape)})"
                    ),
                    add_semicolon=True,
                )
            )
        if self.extra_tensor_names:
            assert self.extra_tensor_dtype is not None
            extra_dtype = get_pytorch_c_dtype_str(self.extra_tensor_dtype)
            for name in self.extra_tensor_names:
                validation.append(
                    llir.RawStmt(
                        code=(
                            "scorch_native::validate_jit_extra_tensor("
                            f"{name}_values, {extra_dtype}, "
                            f'"{self.function_name}", "{name}_values")'
                        ),
                        add_semicolon=True,
                    )
                )
        return validation

    def emit_input_prologue(self) -> List[llir.Stmt]:
        """Build fresh shape/index/value bindings for every input tensor."""

        self._validate()
        stmts: List[llir.Stmt] = []
        for tensor in self.input_tensors:
            stmts.append(llir.BlankLine())
            stmts.append(llir.Comment(f"Get {tensor.name}'s level & value arrays"))
            stmts.extend(tensor.emit_level_array_bindings())
            stmts.append(tensor.emit_value_pointer())
        return stmts

    def emit_extra_tensor_prologue(self) -> List[llir.Stmt]:
        """Build fresh typed value pointers for post-op tensor arguments."""

        self._validate()
        if not self.extra_tensor_names:
            return []
        assert self.extra_tensor_dtype is not None
        c_dtype = dtype_to_c_datatype(self.extra_tensor_dtype)
        ptr_type = llir.DataType.ptr_type(self.extra_tensor_dtype)
        return [
            llir.VarInit(
                var=llir.Var(
                    name=f"{name}_val",
                    type=ptr_type,
                    is_restrict=True,
                ),
                value=tensor_data_ptr(
                    llir.Var(
                        name=f"{name}_values",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    c_dtype,
                ),
            )
            for name in self.extra_tensor_names
        ]

    def assemble_function(self, body: List[llir.Stmt]) -> llir.Function:
        """Wrap one verified body in a fresh ABI-owned function signature."""

        self._validate()
        if type(body) is not list:
            raise TypeError("kernel function body must be an exact LLIR list")
        return llir.Function(
            return_type=llir.DataType.TACO_TENSOR,
            name=self.function_name,
            args=self.emit_arguments(),
            body=list(body),
        )


@dataclass(frozen=True)
class ResultTensorAssembler:
    """Build result storage and final ABI assembly from an owned metadata snapshot."""

    name: str
    level_types: Tuple[LevelType, ...]
    dtype: torch.dtype
    known_nnz_var: Optional[str] = None
    exact_dense_parent_positions: bool = False
    reserve_hint_var: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the frozen result-ABI metadata snapshot."""

        if type(self.name) is not str or not self.name.isidentifier():
            raise TypeError("result tensor name must be a non-empty identifier")
        if type(self.level_types) is not tuple or any(
            type(level_type) is not LevelType for level_type in self.level_types
        ):
            raise TypeError("result level types must be an immutable LevelType tuple")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("result dtype must be a torch.dtype")
        if self.known_nnz_var is not None and (
            type(self.known_nnz_var) is not str or not self.known_nnz_var.isidentifier()
        ):
            raise TypeError("known nnz variable must be an identifier or None")
        if type(self.exact_dense_parent_positions) is not bool:
            raise TypeError("exact dense parent positions must be a bool")
        if self.reserve_hint_var is not None and (
            type(self.reserve_hint_var) is not str
            or not self.reserve_hint_var.isidentifier()
        ):
            raise TypeError("reserve hint variable must be an identifier or None")

    @property
    def levels(self) -> int:
        return len(self.level_types)

    @property
    def is_dense(self) -> bool:
        return all(level_type == LevelType.DENSE for level_type in self.level_types)

    def _has_fixed_position_count(self, level: int) -> bool:
        """A compressed level below a dense parent has parent_size + 1 slots."""
        return (
            self.exact_dense_parent_positions
            and level > 0
            and self.level_types[level - 1] == LevelType.DENSE
        )

    def emit_compressed_coordinate_allocations(
        self,
        total_vars: Tuple[llir.Var, ...],
    ) -> List[llir.Stmt]:
        """Build coordinate buffers from exact typed compressed cardinalities."""

        if type(self) is not ResultTensorAssembler:
            raise TypeError(
                "compressed coordinate allocations require an exact "
                "ResultTensorAssembler"
            )
        self.validate()
        if (
            self.levels < 2
            or self.level_types[0] is not LevelType.DENSE
            or any(
                level_type is not LevelType.COMPRESSED
                for level_type in self.level_types[1:]
            )
        ):
            raise ValueError(
                "compressed coordinate allocations require one dense level "
                "followed by one or more compressed levels"
            )
        if type(total_vars) is not tuple:
            raise TypeError(
                "compressed coordinate totals must be an immutable Var tuple"
            )
        if len(total_vars) != self.levels - 1:
            raise ValueError(
                "compressed coordinate totals must match the compressed levels"
            )

        statements: List[llir.Stmt] = []
        for level, total_var in zip(range(1, self.levels), total_vars):
            if type(total_var) is not llir.Var:
                raise TypeError(
                    "compressed coordinate totals must contain exact LLIR Vars"
                )
            try:
                total_name = total_var.name
                total_type = total_var.type
                total_is_ptr = total_var.is_ptr
                total_is_restrict = total_var.is_restrict
                total_access = total_var.tensor_access
            except AttributeError as failure:
                raise TypeError(
                    "compressed coordinate totals must contain complete LLIR Vars"
                ) from failure
            if total_name != f"_total{level}":
                raise ValueError(
                    "compressed coordinate total name must match its result level"
                )
            if total_type is not llir.DataType.INT64:
                raise TypeError("compressed coordinate totals must have INT64 type")
            if total_is_ptr is not False:
                raise TypeError("compressed coordinate totals cannot be pointers")
            if total_is_restrict is not False:
                raise TypeError(
                    "compressed coordinate totals cannot be restrict-qualified"
                )
            if total_access is not None:
                raise TypeError(
                    "compressed coordinate totals cannot carry tensor provenance"
                )

            owner_name = f"{self.name}{level}_crd_torch"
            statements.extend(
                (
                    llir.VarInit(
                        var=llir.Var(
                            name=owner_name,
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.FunctionCall(
                            name="torch::empty",
                            args=(
                                llir.Array(
                                    values=(
                                        llir.Var(
                                            name=total_name,
                                            type=total_type,
                                        ),
                                    ),
                                    data_type=llir.DataType.INT64,
                                ),
                                llir.QualifiedName(
                                    namespace="torch",
                                    name="kInt",
                                    data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                ),
                            ),
                        ),
                    ),
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{self.name}{level}_crd_data",
                            type=llir.DataType.PTR_INT,
                        ),
                        value=tensor_data_ptr(
                            llir.Var(
                                name=owner_name,
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            llir.DataType.INT,
                        ),
                    ),
                )
            )
        return statements

    def emit_value_array_init(self) -> List[llir.Stmt]:
        """Emit Torch-owned known-size storage or a dynamic ``std::vector``."""
        stmts: List[llir.Stmt] = []
        if self.is_dense:
            # capacity = product of all dimension sizes
            res_capacity_expr: llir.Expr = llir.Var(
                name=f"{self.name}0_size",
                type=llir.DataType.INT64,
            )
            for i in range(1, self.levels):
                res_capacity_expr = llir.BinOp(
                    left=res_capacity_expr,
                    op="*",
                    right=llir.Var(
                        name=f"{self.name}{i}_size",
                        type=llir.DataType.INT64,
                    ),
                )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_capacity",
                        type=llir.DataType.INT64,
                    ),
                    value=res_capacity_expr,
                )
            )

            c_datatype = dtype_to_c_datatype(self.dtype)
            res_capacity_var = llir.Var(
                name=f"{self.name}_capacity",
                type=llir.DataType.INT64,
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="torch::empty",
                        args=[
                            llir.Array(
                                values=(
                                    llir.Var(
                                        name=f"{self.name}_capacity",
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                data_type=llir.DataType.INT64,
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.ptr_type(c_datatype),
                        is_restrict=True,
                    ),
                    value=tensor_data_ptr(
                        llir.Var(
                            name=f"{self.name}_values_torch",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        c_datatype,
                    ),
                )
            )

            # Zero the whole dense buffer before the parallel += accumulate. The
            # generated kernel accumulates into C, so it needs the full buffer
            # zeroed (not empty-rows-only like the prebuilt SpMM). scorch_zero_dense
            # (scorch/csrc/header.h) parallelizes that zero across cores for large
            # outputs — where the serial memset was a big fraction of runtime — and
            # falls back to a single memset below SCORCH_MEMSET_GRAIN_BYTES. Takes
            # the element count; it computes the byte span internally.
            stmts.append(
                llir.FunctionCallStmt(
                    name="scorch_zero_dense",
                    args=[
                        llir.Var(
                            name=f"{self.name}_values",
                            type=llir.DataType.ptr_type(self.dtype),
                        ),
                        res_capacity_var,
                    ],
                )
            )
        elif self.known_nnz_var:
            c_datatype = dtype_to_c_datatype(self.dtype)
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="torch::empty",
                        args=[
                            llir.Array(
                                values=(
                                    llir.Var(
                                        name=self.known_nnz_var,
                                        type=llir.DataType.INT64,
                                    ),
                                ),
                                data_type=llir.DataType.INT64,
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.ptr_type(c_datatype),
                    ),
                    value=tensor_data_ptr(
                        llir.Var(
                            name=f"{self.name}_values_torch",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        c_datatype,
                    ),
                )
            )
        else:
            stmts.append(
                llir.VarDecl(
                    llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.std_vector_type(
                            dtype_to_c_datatype(self.dtype)
                        ),
                    )
                )
            )
            if self.reserve_hint_var:
                stmts.append(
                    llir.FunctionCallStmt(
                        name=f"{self.name}_values.reserve",
                        args=[
                            llir.Var(
                                name=self.reserve_hint_var,
                                type=llir.DataType.INT64,
                            )
                        ],
                    )
                )
        return stmts

    def emit_level_indices_init(self) -> List[llir.Stmt]:
        """Emit per-level index array initialization for COMPRESSED/COORDINATE levels."""
        stmts: List[llir.Stmt] = []
        for i, level_type in enumerate(self.level_types):
            if level_type == LevelType.COMPRESSED:
                if self._has_fixed_position_count(i):
                    # A dense parent fixes the exact position-array length.
                    # Size a standard vector once, then transfer it to Torch
                    # without a copy. Parent coordinates are ABI-validated
                    # against this same extent before the generated loop.
                    stmts.append(
                        llir.DirectInit(
                            var=llir.Var(
                                name=f"{self.name}{i}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            args=(
                                llir.Add(
                                    llir.Cast(
                                        expr=llir.Var(
                                            name=f"{self.name}{i - 1}_size",
                                            type=llir.DataType.INT64,
                                        ),
                                        data_type=llir.DataType.SIZE_T,
                                    ),
                                    llir.Literal(
                                        value=1,
                                        data_type=llir.DataType.INT,
                                    ),
                                ),
                                llir.Literal(
                                    value=0,
                                    data_type=llir.DataType.INT,
                                ),
                            ),
                        )
                    )
                else:
                    # Sparse parents have a dynamically assembled fiber count.
                    stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{self.name}{i}_pos",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            )
                        )
                    )
                stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{self.name}{i}_crd",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        )
                    )
                )
                if self.reserve_hint_var:
                    stmts.append(
                        llir.FunctionCallStmt(
                            name=f"{self.name}{i}_crd.reserve",
                            args=[
                                llir.Var(
                                    name=self.reserve_hint_var,
                                    type=llir.DataType.INT64,
                                )
                            ],
                        )
                    )
                if not self._has_fixed_position_count(i):
                    # pos[0] = 0 for the append-built position vector.
                    stmts.append(
                        llir.Assign(
                            var=llir.ArrayAccess(
                                array=llir.Var(
                                    name=f"{self.name}{i}_pos",
                                    type=llir.DataType.STD_VECTOR_C_INT,
                                ),
                                index=llir.Literal(0),
                            ),
                            value=llir.Literal(0),
                        )
                    )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                # int <name><i>_pos_index = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"{self.name}{i}_pos_index",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

            elif level_type == LevelType.COORDINATE:
                if self.known_nnz_var:
                    # Known-size coordinate arrays are Torch-owned from creation.
                    stmts.extend(
                        [
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"{self.name}{i}_crd_torch",
                                    type=llir.DataType.TORCH_TENSOR,
                                ),
                                value=llir.FunctionCall(
                                    name="torch::empty",
                                    args=(
                                        llir.Array(
                                            values=(
                                                llir.Var(
                                                    name=self.known_nnz_var,
                                                    type=llir.DataType.INT64,
                                                ),
                                            ),
                                            data_type=llir.DataType.INT64,
                                        ),
                                        llir.QualifiedName(
                                            namespace="torch",
                                            name="kInt",
                                            data_type=(llir.DataType.TORCH_SCALAR_TYPE),
                                        ),
                                    ),
                                ),
                            ),
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"{self.name}{i}_crd",
                                    type=llir.DataType.PTR_INT,
                                ),
                                value=tensor_data_ptr(
                                    llir.Var(
                                        name=f"{self.name}{i}_crd_torch",
                                        type=llir.DataType.TORCH_TENSOR,
                                    ),
                                    llir.DataType.INT,
                                ),
                            ),
                        ]
                    )
                else:
                    stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{self.name}{i}_crd",
                                type=llir.DataType.STD_VECTOR_C_INT,
                            )
                        )
                    )
                    if self.reserve_hint_var:
                        stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{self.name}{i}_crd.reserve",
                                args=[
                                    llir.Var(
                                        name=self.reserve_hint_var,
                                        type=llir.DataType.INT64,
                                    )
                                ],
                            )
                        )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT64,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

        return stmts

    def _get_mode_index_set(self, i: int, level_type: LevelType) -> llir.Array:
        """Return one fresh structured mode-index initializer for a level."""
        tensor_level_name = f"{self.name}{i}"
        if level_type == LevelType.DENSE:
            values: Tuple[llir.Expr, ...] = ()
        elif level_type == LevelType.COMPRESSED:
            values = (
                llir.Var(
                    name=f"{tensor_level_name}_pos_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
                llir.Var(
                    name=f"{tensor_level_name}_crd_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            )
        elif level_type == LevelType.COORDINATE:
            values = (
                llir.Var(
                    name=f"{tensor_level_name}_crd_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            )
        else:
            raise ValueError(f"unsupported result level type {level_type}")
        return llir.Array(
            values=values,
            data_type=llir.DataType.STD_VECTOR_TORCH_TENSOR,
        )

    def emit_result_declaration(self) -> llir.VarDecl:
        """Declare one fresh ABI result owner from this frozen snapshot."""

        self.validate()
        return llir.VarDecl(
            var=llir.Var(
                name=self.name,
                type=llir.DataType.TACO_TENSOR,
            )
        )

    def emit_storage_epilogue(self) -> List[llir.Stmt]:
        """Assign result storage and return without moving dynamic buffers."""

        self.validate()
        return [
            llir.Assign(
                var=tensor_storage_member(
                    self.name,
                    "storage",
                    "index",
                    "mode_indices",
                ),
                value=llir.Array(
                    values=tuple(
                        self._get_mode_index_set(i, level_type)
                        for i, level_type in enumerate(self.level_types)
                    ),
                    data_type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                ),
            ),
            llir.Assign(
                var=tensor_storage_member(self.name, "storage", "value"),
                value=llir.Var(
                    name=f"{self.name}_values_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            ),
            llir.Return(
                value=llir.Var(
                    name=self.name,
                    type=llir.DataType.TACO_TENSOR,
                )
            ),
        ]

    def emit_final_assembly(self) -> List[llir.Stmt]:
        """Move dynamic buffers to Torch, assign indices/values, and return."""
        self.validate()
        stmts: List[llir.Stmt] = []

        # TacoTensor decl
        stmts.extend(
            [
                llir.Comment("Assemble final result"),
                self.emit_result_declaration(),
            ]
        )

        # Move dynamic index vectors into Torch storage contexts. Known-size
        # coordinate tensors were allocated before the compute loop.
        for i, level_type in enumerate(self.level_types):
            tensor_level_name = f"{self.name}{i}"

            if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                if level_type == LevelType.COMPRESSED:
                    stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{tensor_level_name}_pos_torch",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            value=llir.FunctionCall(
                                name="scorch_tensor_from_vector",
                                args=[
                                    llir.FunctionCall(
                                        name="std::move",
                                        args=[
                                            llir.Var(
                                                name=f"{tensor_level_name}_pos",
                                                type=llir.DataType.STD_VECTOR_C_INT,
                                            )
                                        ],
                                    ),
                                    llir.QualifiedName(
                                        namespace="torch",
                                        name="kInt",
                                        data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                    ),
                                ],
                            ),
                        )
                    )

                if self.known_nnz_var and level_type == LevelType.COORDINATE:
                    continue
                else:
                    stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{tensor_level_name}_crd_torch",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            value=llir.FunctionCall(
                                name="scorch_tensor_from_vector",
                                args=[
                                    llir.FunctionCall(
                                        name="std::move",
                                        args=[
                                            llir.Var(
                                                name=f"{tensor_level_name}_crd",
                                                type=llir.DataType.STD_VECTOR_C_INT,
                                            )
                                        ],
                                    ),
                                    llir.QualifiedName(
                                        namespace="torch",
                                        name="kInt",
                                        data_type=llir.DataType.TORCH_SCALAR_TYPE,
                                    ),
                                ],
                            ),
                        )
                    )

        if not self.is_dense and not self.known_nnz_var:
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="scorch_tensor_from_vector",
                        args=[
                            llir.FunctionCall(
                                name="std::move",
                                args=[
                                    llir.Var(
                                        name=f"{self.name}_values",
                                        type=llir.DataType.std_vector_type(
                                            dtype_to_c_datatype(self.dtype)
                                        ),
                                    )
                                ],
                            ),
                            llir.QualifiedName(
                                namespace="torch",
                                name=get_pytorch_c_dtype_name(self.dtype),
                                data_type=llir.DataType.TORCH_SCALAR_TYPE,
                            ),
                        ],
                    ),
                )
            )

        stmts.extend(self.emit_storage_epilogue())

        return stmts
