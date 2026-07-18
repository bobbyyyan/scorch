"""Structured Torch/C++ ABI construction shared by compiler lowering stages."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import torch

from . import llir
from ..format import LevelType
from ..utils import dtype_to_c_datatype, get_pytorch_c_dtype_name


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
class ResultTensorAssembler:
    """Build result storage and final ABI assembly from an owned metadata snapshot."""

    name: str
    level_types: Tuple[LevelType, ...]
    dtype: torch.dtype
    known_nnz_var: Optional[str] = None
    exact_dense_parent_positions: bool = False
    reserve_hint_var: Optional[str] = None

    def __post_init__(self) -> None:
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
                        llir.RawStmt(
                            code=(
                                f"std::vector<int> {self.name}{i}_pos("
                                f"(size_t){self.name}{i - 1}_size + 1, 0)"
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
                    stmts.append(
                        llir.RawStmt(
                            code=(
                                f"torch::Tensor {self.name}{i}_crd_torch = "
                                f"torch::empty({{{self.known_nnz_var}}}, torch::kInt);\n"
                                f"  int* {self.name}{i}_crd = "
                                f"{self.name}{i}_crd_torch.data_ptr<int>();"
                            ),
                            add_semicolon=False,
                        )
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

    def emit_final_assembly(self) -> List[llir.Stmt]:
        """Move dynamic buffers to Torch, assign indices/values, and return."""
        stmts: List[llir.Stmt] = []

        # TacoTensor decl
        stmts.extend(
            [
                llir.Comment("Assemble final result"),
                llir.VarDecl(
                    var=llir.Var(
                        name=f"{self.name}",
                        type=llir.DataType.TACO_TENSOR,
                    )
                ),
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

        # mode_indices assignment
        stmts.append(
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
            )
        )

        # _value assignment
        stmts.append(
            llir.Assign(
                var=tensor_storage_member(self.name, "storage", "value"),
                value=llir.Var(
                    name=f"{self.name}_values_torch",
                    type=llir.DataType.TORCH_TENSOR,
                ),
            )
        )

        # return statement
        stmts.append(
            llir.Return(
                value=llir.Var(
                    name=f"{self.name}",
                    type=llir.DataType.TACO_TENSOR,
                )
            )
        )

        return stmts
