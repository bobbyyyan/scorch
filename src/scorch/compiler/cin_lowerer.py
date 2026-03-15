from typing import List, Optional, Dict, Union

from . import llir
from .cin import (
    IndexStmt,
    IndexVar,
    TensorVar,
    CINVisitorAccept,
    TensorAssign,
    TensorAccess,
    ForAll,
    IndexExpr,
    BinaryOp,
    CIN,
    Operation,
    Where,
    WorkspaceAccess,
    Workspace,
)
from .iter_lattice import IterationLattice
from .llir import AssignOp, DataType
from ..format import LevelType, TensorFormat, LevelFormat
from ..utils import dtype_to_c_datatype, get_pytorch_c_dtype_str


class ResultTensorAssembler:
    """Assembles LLIR statements for result tensor initialization and final construction."""

    def __init__(self, tensor_var: TensorVar):
        self.tensor_var = tensor_var
        self.name = tensor_var.get_name()
        self.level_types = tensor_var.get_level_types()
        self.is_dense = tensor_var.is_dense()
        self.dtype = tensor_var.dtype

    def emit_value_array_init(self) -> List[llir.Stmt]:
        """Emit value array initialization: dense malloc+memset or sparse cvector decl."""
        stmts: List[llir.Stmt] = []
        if self.is_dense:
            # capacity = product of all dimension sizes
            res_capacity_expr: llir.Expr = llir.Var(
                name=f"{self.name}0_size",
                type=llir.DataType.INT,
            )
            for i in range(1, self.tensor_var.levels):
                res_capacity_expr = llir.BinOp(
                    left=res_capacity_expr,
                    op="*",
                    right=llir.Var(
                        name=f"{self.name}{i}_size",
                        type=llir.DataType.INT,
                    ),
                )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_capacity",
                        type=llir.DataType.INT,
                    ),
                    value=res_capacity_expr,
                )
            )

            # malloc + cast
            c_datatype = dtype_to_c_datatype(self.dtype)
            sizeof_expr = llir.Sizeof(c_datatype)
            res_capacity_var = llir.Var(
                name=f"{self.name}_capacity",
                type=llir.DataType.INT,
            )
            malloc = llir.FunctionCall(
                name="malloc",
                args=[llir.BinOp(left=sizeof_expr, op="*", right=res_capacity_var)],
            )
            malloc = llir.Cast(
                expr=malloc,
                data_type=llir.DataType.ptr_type(c_datatype),
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.ptr_type(self.dtype),
                        is_restrict=True,
                    ),
                    value=malloc,
                )
            )

            # memset to zero
            stmts.append(
                llir.FunctionCallStmt(
                    name="memset",
                    args=[
                        llir.Var(
                            name=f"{self.name}_values",
                            type=llir.DataType.ptr_type(self.dtype),
                        ),
                        llir.Literal(0),
                        llir.BinOp(
                            left=sizeof_expr,
                            op="*",
                            right=res_capacity_var,
                        ),
                    ],
                )
            )
        else:
            stmts.append(
                llir.VarDecl(
                    llir.Var(
                        name=f"{self.name}_values",
                        type=llir.DataType.cvector_type(
                            dtype_to_c_datatype(self.dtype)
                        ),
                    )
                )
            )
        return stmts

    def emit_level_indices_init(self) -> List[llir.Stmt]:
        """Emit per-level index array initialization for COMPRESSED/COORDINATE levels."""
        stmts: List[llir.Stmt] = []
        for i, level_type in enumerate(self.level_types):
            if level_type == LevelType.COMPRESSED:
                # cvector<int> pos and crd
                stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{self.name}{i}_pos",
                            type=llir.DataType.CVECTOR_INT,
                        )
                    )
                )
                stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{self.name}{i}_crd",
                            type=llir.DataType.CVECTOR_INT,
                        )
                    )
                )
                # pos[0] = 0
                stmts.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{self.name}{i}_pos[0]",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Literal(0),
                    )
                )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Literal(0),
                    )
                )
                # int <name><i>_pos_index = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"{self.name}{i}_pos_index",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

                # Dense-parent pos init loop
                if i > 0 and self.level_types[i - 1] == LevelType.DENSE:
                    loop_var_name = f"p{self.name}{i}"
                    loop_var = llir.Var(
                        name=loop_var_name,
                        type=llir.DataType.INT,
                    )
                    loop = llir.ForLoop(
                        init=llir.VarInit(
                            var=loop_var,
                            value=llir.Literal(1),
                        ),
                        cond=llir.BinOp(
                            left=loop_var,
                            op="<=",
                            right=llir.Var(
                                name=f"{self.name}{i - 1}_size",
                                type=llir.DataType.INT,
                            ),
                        ),
                        update=llir.Increment(
                            var=loop_var,
                        ),
                        body=[
                            llir.Assign(
                                var=llir.Var(
                                    name=f"{self.name}{i}_pos[{loop_var_name}]",
                                    type=llir.DataType.INT,
                                ),
                                value=llir.Literal(0),
                            )
                        ],
                    )
                    stmts.append(loop)

            elif level_type == LevelType.COORDINATE:
                # cvector<int> crd
                stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{self.name}{i}_crd",
                            type=llir.DataType.CVECTOR_INT,
                        )
                    )
                )
                # int p<name><i> = 0
                stmts.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"p{self.name}{i}",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Literal(0),
                    )
                )
                stmts.append(llir.BlankLine())

        return stmts

    def _get_mode_index_set(self, i: int, level_type: LevelType) -> str:
        """Return the mode index set string for a given level."""
        tensor_level_name = f"{self.name}{i}"
        if level_type == LevelType.DENSE:
            return "{}"
        elif level_type == LevelType.COMPRESSED:
            return f"{{{tensor_level_name}_pos_torch, {tensor_level_name}_crd_torch}}"
        elif level_type == LevelType.COORDINATE:
            return f"{{{tensor_level_name}_crd_torch}}"

    def emit_final_assembly(self) -> List[llir.Stmt]:
        """Emit TacoTensor decl, from_blob conversions, index/value assignment, and return."""
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

        # Per-level from_blob for pos/crd
        for i, level_type in enumerate(self.level_types):
            tensor_level_name = f"{self.name}{i}"

            if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                if level_type == LevelType.COMPRESSED:
                    # pos_torch
                    stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{tensor_level_name}_pos_torch",
                                type=llir.DataType.TORCH_TENSOR,
                            ),
                            value=llir.FunctionCall(
                                name="torch::from_blob",
                                args=[
                                    llir.Var(
                                        name=f"{tensor_level_name}_pos.data()",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    llir.Var(
                                        name=f"{{{tensor_level_name}_pos.size()}}",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    llir.Var(
                                        name=f"{tensor_level_name}_pos.get_deleter()",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    llir.Var(
                                        name="torch::kInt",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                ],
                            ),
                        )
                    )

                # crd_torch
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor_level_name}_crd_torch",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.FunctionCall(
                            name="torch::from_blob",
                            args=[
                                llir.Var(
                                    name=f"{tensor_level_name}_crd.data()",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{{{tensor_level_name}_crd.size()}}",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{tensor_level_name}_crd.get_deleter()",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name="torch::kInt",
                                    type=llir.DataType.NO_TYPE,
                                ),
                            ],
                        ),
                    )
                )

        # Values from_blob
        res_values_torch_var = llir.Var(
            name=f"{self.name}_values_torch",
            type=llir.DataType.TORCH_TENSOR,
        )
        if self.is_dense:
            # Lambda deleter + from_blob with capacity
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.name}_values_deleter",
                        type=llir.DataType.AUTO,
                    ),
                    value=llir.Var(
                        name="[](void* ptr) {{ free(ptr); }}",
                        type=llir.DataType.AUTO,
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=res_values_torch_var,
                    value=llir.FunctionCall(
                        name="torch::from_blob",
                        args=[
                            llir.Var(
                                name=f"{self.name}_values",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{{{self.name}_capacity}}",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{self.name}_values_deleter",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=get_pytorch_c_dtype_str(self.dtype),
                                type=llir.DataType.NO_TYPE,
                            ),
                        ],
                    ),
                )
            )
        else:
            # from_blob with cvector data/size/deleter
            stmts.append(
                llir.VarInit(
                    var=res_values_torch_var,
                    value=llir.FunctionCall(
                        name="torch::from_blob",
                        args=[
                            llir.Var(
                                name=f"{self.name}_values.data()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{{{self.name}_values.size()}}",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{self.name}_values.get_deleter()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=get_pytorch_c_dtype_str(self.dtype),
                                type=llir.DataType.NO_TYPE,
                            ),
                        ],
                    ),
                )
            )

        # mode_indices assignment
        stmts.append(
            llir.Assign(
                var=llir.Var(
                    name=f"{self.name}._storage._index.mode_indices",
                    type=llir.DataType.NO_TYPE,
                ),
                value=llir.Var(
                    name=f"{{{', '.join([self._get_mode_index_set(i, lt) for i, lt in enumerate(self.level_types)])}}}",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # _value assignment
        stmts.append(
            llir.Assign(
                var=llir.Var(
                    name=f"{self.name}._storage._value",
                    type=llir.DataType.NO_TYPE,
                ),
                value=llir.Var(
                    name=f"{self.name}_values_torch",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # return statement
        stmts.append(
            llir.Return(
                value=llir.Var(
                    name=f"{self.name}",
                    type=llir.DataType.NO_TYPE,
                )
            )
        )

        return stmts


class CINLowerer:
    """
    This is a class to lower a CIN to LLIR
    """

    def __init__(self, filter_zeros=False):
        self.filter_zeros: bool = filter_zeros
        self.defined_index_vars: List[IndexVar] = []

        self.dense_coord_resolve_stmt_to_dep_index_vars: Dict[
            llir.VarInit, List[IndexVar]
        ] = {}

        self.seen_outermost_forall = False
        self.outermost_stmt: Optional[IndexStmt] = None

        self.result_value_array_sparse_index_llir = None
        self._scalar_accum_mode = False
        self._used_scalar_accum = False
        self.index_var_to_rhs_tensor_level_type = None
        self.index_var_to_result_tensor_level_type = None

        self.result_tensor_var: Optional[TensorVar] = None
        self.result_tensor_access: Optional[TensorAccess] = None
        self.result_tensor_value_index_var_dict: Dict[IndexVar, llir.Expr] = {}
        self.final_result_tensor_var: Optional[TensorVar] = None
        self.final_result_tensor_access: Optional[TensorAccess] = None

        self.llir_stmt: Optional[llir.Stmt] = None

        self.need_compute: List[TensorVar] = []
        self.tensor_var_to_llir: Dict[TensorVar, llir.Expr] = {}

    @staticmethod
    def get_level_arrays(tensor: TensorVar) -> List[llir.Stmt]:
        """
        Generate the bounds variable definitions given a TensorVar
        """
        # Iterate over the levels in tensor, then depending on whether it is sparse or dense, generate the bound
        # variables
        stmts: List[llir.Stmt] = []
        level_types = tensor.get_level_types()
        for level, level_type in enumerate(level_types):
            if level_type == LevelType.DENSE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_size",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}_shape[{level}]",
                            type=llir.DataType.INT,
                        ),
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_pos",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}_mode_indices[{level}][0].data_ptr<int>()",
                            type=llir.DataType.PTR_INT,
                        ),
                    )
                )
                #
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}_mode_indices[{level}][1].data_ptr<int>()",
                            type=llir.DataType.PTR_INT,
                        ),
                    )
                )
            elif level_type == LevelType.COORDINATE:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_crd_tensor",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}_mode_indices[{level}][0]",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                    )
                )
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_crd",
                            type=llir.DataType.PTR_INT,
                            is_restrict=True,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}_mode_indices[{level}][0].data_ptr<int>()",
                            type=llir.DataType.PTR_INT,
                        ),
                    )
                )
        return stmts

    @staticmethod
    def get_val_ptr_stmt(tensor: TensorVar) -> llir.Stmt:
        """
        Get the value pointer for a tensor
        """
        data_type = dtype_to_c_datatype(tensor.dtype)
        ptr_type = llir.DataType.ptr_type(tensor.dtype)
        return llir.VarInit(
            var=llir.Var(name=f"{tensor.name}_val", type=ptr_type, is_restrict=True),
            value=llir.Var(
                name=f"{tensor.name}_values.data_ptr<{data_type.value}>()",
                type=ptr_type,
            ),
        )

    @staticmethod
    def get_value_array_statement(tensor: TensorVar) -> llir.Stmt:
        """
        Get the value array for a tensor
        """
        return llir.VarInit(
            var=llir.Var(name=f"{tensor.name}_values", type=llir.DataType.TORCH_TENSOR),
            value=llir.Var(
                name=f"{tensor.name}._storage._value",
                type=llir.DataType.TORCH_TENSOR,
            ),
        )

    def lower_TensorAccess(self, tensor_access: TensorAccess) -> llir.Expr:
        """
        Lower a TensorAccess to LLIR
        """
        sorted_index_vars = tensor_access.get_sorted_index_vars()
        last_index_var = sorted_index_vars[-1]

        # If the level_type corresponding to the last index var is dense, then we can just use
        # the index var as the index into the value array
        tensor_var = tensor_access.get_tensor()
        level = tensor_access.level_of_index_var(last_index_var)
        level_type = tensor_var.get_level_types()[level]

        if len(tensor_access.indices) == 1 and level_type == LevelType.DENSE:
            return llir.Var(
                name=f"{tensor_access.tensor.name}_val[{last_index_var.name}]",
                type=llir.DataType.NO_TYPE,
            )

        return llir.Var(
            name=f"{tensor_access.tensor.name}_val"
            + f"[p{tensor_access.tensor.get_name()}{tensor_access.level_of_index_var(last_index_var)}]",
            type=llir.DataType.NO_TYPE,
        )

        # if level_type == LevelType.DENSE:
        #     return llir.Var(
        #         name=f"{tensor_access.tensor.name}_values[{last_index_var.name}]",
        #         type=llir.DataType.NO_TYPE,
        #     )
        # elif level_type == LevelType.COMPRESSED:
        #     return llir.Var(
        #         name=f"{tensor_access.tensor.name}_values[{last_index_var.name}_{tensor_access.tensor.name}]",
        #         type=llir.DataType.NO_TYPE,
        #     )
        raise NotImplementedError(f"Level type {level_type} not implemented")

    def lower_BinaryOp(self, bin_op: BinaryOp) -> llir.Expr:
        """
        Lower a BinaryOp to LLIR
        """
        return llir.BinOp(
            op=bin_op.op.value,
            left=self.lower_IndexExpr(bin_op.left),
            right=self.lower_IndexExpr(bin_op.right),
        )

    def lower_IndexExpr(self, index_expr: IndexExpr) -> llir.Expr:
        if isinstance(index_expr, BinaryOp):
            return self.lower_BinaryOp(index_expr)
        elif isinstance(index_expr, TensorAccess):
            return self.lower_TensorAccess(index_expr)
        raise NotImplementedError

    def lower_CIN(self, cin: CIN) -> Union[llir.Stmt, List[llir.Stmt], llir.Expr]:
        if isinstance(cin, IndexStmt):
            return self.lower_IndexStmt(cin)
        elif isinstance(cin, IndexExpr):
            return self.lower_IndexExpr(cin)
        return []

    def lower_TensorAssign(self, stmt: TensorAssign) -> List[llir.Stmt]:
        """
        Lower a TensorAssign to LLIR
        """
        llir_stmts: List[llir.Stmt] = []

        rhs_llir = self.lower_IndexExpr(stmt.rhs)

        # Scalar accumulation mode: accumulate into local register.
        # Coordinate emission and position increment are handled by
        # the enclosing free-variable ForAll level (see iter_lattice.py).
        if self._scalar_accum_mode:
            llir_stmts.append(
                llir.Assign(
                    var=llir.Var(name="_accum", type=llir.DataType.NO_TYPE),
                    value=rhs_llir,
                    op=AssignOp.ADD_ASSIGN,
                )
            )
            return llir_stmts

        # if we are at the bottommost level, we can emit compute code
        assert self.result_tensor_access, "result tensor access is None"
        is_workspace = self.result_tensor_access.is_workspace()
        index_vars = self.result_tensor_access.get_index_vars()
        sorted_index_vars = self.result_tensor_access.get_sorted_index_vars()
        # If index_vars is None (empty), that means we have a scalar workspace
        # Then just do <tensor> += <rhs_llir>
        if not index_vars:
            wksp_name = self.result_tensor_var.get_name()
            assign_stmt = llir.Assign(
                var=llir.Var(name=f"{wksp_name}", type=llir.DataType.NO_TYPE),
                value=rhs_llir,
                op=AssignOp.ADD_ASSIGN,
            )
            llir_stmts.append(assign_stmt)
        else:
            # if index_vars are all in defined_index_vars, then we can emit the compute code
            if all(index_var in self.defined_index_vars for index_var in index_vars):
                assert self.result_tensor_var, "result tensor var is None"

                values_llir_name = self.result_tensor_var.name
                if not is_workspace:
                    values_llir_name = f"{values_llir_name}_values"

                if self.result_value_array_sparse_index_llir:
                    tensor_access_llir = llir.Var(
                        name=f"{values_llir_name}[{self.result_value_array_sparse_index_llir.name}]",
                        type=llir.DataType.NO_TYPE,
                    )
                else:
                    level = self.result_tensor_access.level_of_index_var(sorted_index_vars[-1])
                    tensor_access_llir = llir.Var(
                        name=f"{values_llir_name}[p{self.result_tensor_var.name}{level}]",
                        type=llir.DataType.NO_TYPE,
                    )
                    # tensor_access_llir = llir.Var(
                    #     name=f"{self.result_tensor_var.get_name()}_values"
                    #     + f"[{self.defined_index_vars[-1].name}]",
                    #     type=llir.DataType.NO_TYPE,
                    # )

                if is_workspace:
                    assert isinstance(self.result_tensor_access, WorkspaceAccess)
                    wksp_access: WorkspaceAccess = self.result_tensor_access
                    wksp_index_vars = wksp_access.get_index_vars()
                    sorted_wksp_index_vars = [
                        wksp_index_vars[i] for i in wksp_access.tensor.mode_order
                    ]

                    if wksp_access.is_dense():
                        # <workspace name>[<C++ array of indices>] += <rhs_llir>;
                        assert (
                            len(sorted_wksp_index_vars) == 1
                        ), "dense workspace has more than 1 index var"
                        wksp_index_var = sorted_wksp_index_vars[0]
                        llir_stmts.append(
                            llir.Assign(
                                var=llir.Var(
                                    name=f"{self.result_tensor_var.name}[{wksp_index_var.name}]",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                value=rhs_llir,
                                op=AssignOp.ADD_ASSIGN,
                            )
                        )

                    else:
                        # <workspace name>.insert(<C++ array of indices>, <rhs_llir>);

                        llir_stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{self.result_tensor_access.get_tensor().get_name()}.insert",
                                args=[
                                    llir.Array(
                                        values=[
                                            llir.Var(
                                                name=ivar.name,
                                                type=llir.DataType.INT,
                                            )
                                            for ivar in sorted_wksp_index_vars
                                        ],
                                        data_type=llir.DataType.INT,
                                    ),
                                    rhs_llir,
                                ],
                            )
                        )
                else:
                    if stmt.op == Operation.ADD:
                        # llir_stmts.append(
                        #     llir.Assign(
                        #         var=tensor_access_llir,
                        #         value=llir.BinOp(
                        #             op="+",
                        #             left=tensor_access_llir,
                        #             right=rhs_llir,
                        #         ),
                        #     )
                        # )
                        llir_stmts.append(
                            llir.Assign(
                                var=tensor_access_llir,
                                value=rhs_llir,
                                op=AssignOp.ADD_ASSIGN,
                            )
                        )
                    else:
                        llir_stmts.append(
                            llir.Assign(
                                var=tensor_access_llir,
                                value=rhs_llir,
                            )
                        )
            # If the last _level of the result tensor var is sparse, then we need to set
            # the coordinates
            if not self.result_tensor_access.is_workspace():
                last_ivar = self.defined_index_vars[-1]
                last_level_type = self.result_tensor_access.level_types()[-1]
                if last_level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                    llir_stmts.append(llir.Comment("Set coordinates"))
                    result_tensor_name = self.result_tensor_var.get_name()
                    result_index_name = (
                        f"p{result_tensor_name}{self.result_tensor_var.levels - 1}"
                    )
                    level = self.result_tensor_access.level_of_index_var(last_ivar)

                    llir_stmts.append(
                        llir.Assign(
                            var=llir.Var(
                                name=f"{result_tensor_name}{level}_crd"
                                + f"[{result_index_name}]",
                                type=llir.DataType.NO_TYPE,
                            ),
                            value=llir.Var(
                                name=last_ivar.name,
                                type=llir.DataType.NO_TYPE,
                            ),
                        )
                    )

                    # if the last _level is COORDINATE, we might need to set the coordinates
                    # or previous levels as well
                    if last_level_type == LevelType.COORDINATE:
                        for defined_ivar in self.defined_index_vars[-2::-1]:
                            level_type = (
                                self.result_tensor_access.level_type_of_index_var(
                                    defined_ivar
                                )
                            )
                            level = self.result_tensor_access.level_of_index_var(
                                defined_ivar
                            )
                            if level_type == LevelType.COORDINATE:
                                llir_stmts.append(
                                    llir.Assign(
                                        var=llir.Var(
                                            name=f"{result_tensor_name}{level}_crd"
                                            + f"[{result_index_name}]",
                                            type=llir.DataType.NO_TYPE,
                                        ),
                                        value=llir.Var(
                                            name=defined_ivar.name,
                                            type=llir.DataType.NO_TYPE,
                                        ),
                                    )
                                )
                            else:
                                break

                # if has sparse index for result value array, need to increment
                if self.result_value_array_sparse_index_llir is not None:
                    llir_stmts.append(
                        llir.Increment(
                            var=self.result_value_array_sparse_index_llir,
                        )
                    )

            # If CINLowerer has filter_zeros attribute set to True,
            # we need to wrap llir_stmts in an if block,
            # the condition is whether the input value is non-zero
            if self.filter_zeros:
                llir_stmts = [
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=rhs_llir,
                            right=llir.Literal(value="0"),
                        ),
                        then_body=llir_stmts,
                    )
                ]

        return llir_stmts

    def lower_Where(self, stmt: Where) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower a Where to LLIR
        """
        workspaces = stmt.get_workspaces()
        workspace_init_stmts: List[llir.Stmt] = [
            llir.Comment("Initialize workspaces"),
        ]
        workspace_cleanup_stmts: List[llir.Stmt] = []
        # Per-thread workspace allocation (hoisted outside the for loop but
        # inside the OMP parallel region) and per-iteration memset.
        self._workspace_alloc_stmts: List[llir.Stmt] = []
        self._workspace_free_stmts: List[llir.Stmt] = []
        self._workspace_memset_stmts: List[llir.Stmt] = []
        for wksp in workspaces:
            assert isinstance(wksp, Workspace), "workspace is not a Workspace"
            # coo_workspace<tensor's ctype> <tensor's name> = coo_workspace<tensor's ctype>(<tensor's dim>);
            wksp_ctype = dtype_to_c_datatype(wksp.dtype)
            wksp_ctype_ptr = DataType.ptr_type(wksp_ctype)

            # If the workspace is 0-dimensional, just initialize it with a literal
            if wksp.dim == 0:
                workspace_init_stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=wksp.get_name(),
                            type=wksp_ctype,
                        ),
                        value=llir.Literal(0),
                    )
                )
                continue

            if wksp.dim == 1:
                # Dense workspace: allocate a zeroed array of the appropriate ctype
                if wksp.is_dense():
                    if wksp.is_tiled and wksp.tile_size_var:
                        # Tiled dense workspaces have compile-time bounds. Use stack allocation
                        # so inner kernels avoid heap traffic.
                        workspace_init_stmts.append(
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"{wksp.get_name()}[{wksp.tile_size_var.name}]",
                                    type=wksp_ctype,
                                ),
                                value=llir.Array(
                                    values=[],
                                    data_type=wksp_ctype,
                                ),
                            )
                        )
                    else:
                        # Aligned allocation hoisted to per-thread (outside for loop).
                        # memset per iteration (inside for loop).
                        size_llir = wksp.size_llir_var
                        size_var = size_llir.name
                        # The workspace size variable may reference a dense tensor
                        # dimension (e.g. B1_size). Resolve it to the actual C++ name
                        # so it's available in the hoisted parallel region.
                        wksp_access = [wa for wa in wksp.workspace_accesses
                                       if wa.indices and len(wa.indices) == 1][0]
                        idx_var = wksp_access.indices[0]
                        dense_ta = [ta for ta in idx_var.tensor_accesses
                                    if ta.is_dense() and idx_var in ta.indices
                                    and not ta.is_workspace()][0]
                        level = dense_ta.level_of_index_var(idx_var)
                        actual_size = f"{dense_ta.tensor.name}{level}_size"

                        ctype = wksp_ctype.value
                        wname = wksp.get_name()
                        aligned_size = f"(((size_t){actual_size} + 15) & ~15)"
                        self._workspace_alloc_stmts.extend([
                            llir.RawStmt(
                                code=f"int {size_var} = {actual_size}"
                            ),
                            llir.RawStmt(
                                code=(
                                    f"{ctype}* __restrict__ {wname} = "
                                    f"({ctype}*)aligned_alloc(64, {aligned_size} * sizeof({ctype}))"
                                ),
                            ),
                        ])
                        self._workspace_free_stmts.append(
                            llir.RawStmt(code=f"free({wname})")
                        )
                        self._workspace_memset_stmts.append(
                            llir.RawStmt(
                                code=f"memset({wname}, 0, {size_var} * sizeof({ctype}))"
                            )
                        )
                else:
                    # If the workspace is 1-dimensional, use optimized 1D workspace implementation
                    # coo_workspace_1d<tensor's ctype, tensor's dim>
                    workspace_init_stmts.append(
                        llir.VarInit(
                            var=llir.Var(
                                name=wksp.get_name(),
                                type=llir.DataType.AUTO,
                            ),
                            value=llir.FunctionCall(
                                name=f"coo_workspace_1d<{wksp_ctype.value}, {wksp.dim}>",
                                args=[
                                    llir.Literal(value=f"{1024}"),
                                ],
                            ),
                        )
                    )
                continue

            workspace_init_stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=wksp.get_name(),
                        type=llir.DataType.coo_workspace_type_with_dim(
                            wksp_ctype, wksp.dim
                        ),
                    ),
                    value=llir.FunctionCall(
                        name=f"coo_workspace<{wksp_ctype.value}, {wksp.dim}>",
                        args=[
                            llir.Literal(value=f"{1024}"),
                            llir.Var(
                                name="result_shape",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                        ],
                    ),
                )
            )

        return [
            *workspace_init_stmts,
            *self._workspace_memset_stmts,
            *self.lower_ProducerIndexStmt(stmt.producer),
            *self.lower_ConsumerIndexStmt(stmt.consumer),
            *workspace_cleanup_stmts,
        ]

    def lower_ProducerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower a ProducerIndexStmt to LLIR
        """
        result = self.lower_IndexStmt(stmt)
        if isinstance(result, list):
            return result
        else:
            return [result]

    def lower_outer_ConsumerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower the outermost consumer index statement with mode_order support.

        Two paths:
        - result_is_coord: all result levels are COORDINATE, so we directly
          assign sorted workspace entries to the result tensor arrays.
        - Otherwise: build an intermediate COO tensor from workspace entries,
          assemble torch tensors via from_blob, then recursively lower a
          conversion CIN to produce the final result format.
        """
        workspaces = stmt.get_workspaces()
        wksp = workspaces[0]
        workspace_accesses = stmt.get_workspace_accesses()
        wksp_access: WorkspaceAccess = workspace_accesses[0]
        wksp_index_vars = wksp_access.get_index_vars()

        result_tensor_accesses = stmt.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]
        result_tensor = result_tensor_access.get_tensor()
        result_index_vars = result_tensor_access.get_index_vars()
        result_tensor_name = result_tensor.get_name()

        # Check if result is all-coordinate (fast path: direct assignment)
        result_is_coord = all(
            lt == LevelType.COORDINATE
            for lt in result_tensor.get_format().get_level_types()
        )

        # Create intermediate COO tensor variable
        intermediate_tensor_var = TensorVar(
            name="T",
            fmt=TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.COORDINATE)
                    for _ in range(len(result_index_vars))
                ]
            ),
            dtype=self.result_tensor_var.dtype,
            mode_order=result_tensor.mode_order,
        )

        intermediate_tensor_iterator = llir.Var(
            name=f"p{intermediate_tensor_var.get_name()}",
            type=llir.DataType.INT,
        )

        # Build intermediate cvector declarations (only for non-coord path)
        intermediate_crd_vecs = []
        intermediate_val_vec = llir.Var(
            name=f"{intermediate_tensor_var.get_name()}_val_vec",
            type=llir.DataType.cvector_type(
                dtype_to_c_datatype(intermediate_tensor_var.dtype)
            ),
        )
        vec_decl_stmts = []

        if not result_is_coord:
            for level in range(len(wksp_index_vars)):
                crd_vec = llir.Var(
                    name=f"{intermediate_tensor_var.get_name()}{level}_crd_vec",
                    type=llir.DataType.CVECTOR_INT,
                )
                intermediate_crd_vecs.append(crd_vec)
                vec_decl_stmts.append(llir.VarDecl(crd_vec))

            vec_decl_stmts.append(llir.VarDecl(intermediate_val_vec))
            vec_decl_stmts.append(
                llir.VarInit(
                    var=intermediate_tensor_iterator,
                    value=llir.Literal(0),
                )
            )

        # Sort workspace
        wksp_name = wksp.get_name()
        wksp_sort_stmt = llir.FunctionCallStmt(
            name=f"{wksp_name}.sort",
            args=[],
        )

        # Build loop: for (const auto& it : wksp) { ... }
        loop_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        loop_array = llir.Var(name=wksp_name, type=llir.DataType.AUTO)
        loop_body: List[llir.Stmt] = []

        if result_is_coord:
            # Direct assignment: A0_crd[pA0] = it.first[0]; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{result_tensor_name}{i}_crd[p{result_tensor_name}{i}]",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"{loop_var.name}.first[{i}]",
                            type=llir.DataType.INT,
                        ),
                    )
                )
            # A_values[pA0] = it.second;
            loop_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{result_tensor_name}_values[p{result_tensor_name}0]",
                        type=llir.DataType.INT,
                    ),
                    value=llir.Var(
                        name=f"{loop_var.name}.second",
                        type=llir.DataType.INT,
                    ),
                )
            )
            # pA0++; pA1++; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Increment(
                        var=llir.Var(
                            name=f"p{result_tensor_name}{i}",
                            type=llir.DataType.INT,
                        ),
                    )
                )
        else:
            # Fill intermediate cvectors: T0_crd_vec[pT] = it.first[0]; etc.
            for i in range(len(wksp_index_vars)):
                loop_body.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{intermediate_crd_vecs[i].name}[{intermediate_tensor_iterator.name}]",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"{loop_var.name}.first[{i}]",
                            type=llir.DataType.INT,
                        ),
                    )
                )
            # T_val_vec[pT] = it.second;
            loop_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{intermediate_val_vec.name}[{intermediate_tensor_iterator.name}]",
                        type=llir.DataType.INT,
                    ),
                    value=llir.Var(
                        name=f"{loop_var.name}.second",
                        type=llir.DataType.INT,
                    ),
                )
            )
            # pT++;
            loop_body.append(llir.Increment(var=intermediate_tensor_iterator))

        loop_stmt = llir.ForLoopAuto(
            var=loop_var,
            array=loop_array,
            body=loop_body,
        )

        # For all-coordinate result, we're done after the loop
        if result_is_coord:
            return [
                llir.BlankLine(),
                llir.Comment("Lower outer consumer CIN"),
                *vec_decl_stmts,
                llir.BlankLine(),
                wksp_sort_stmt,
                loop_stmt,
            ]

        # Non-coord path: assemble intermediate torch tensors from cvectors
        assembly_stmts = []
        intermediate_crd_tensors = []

        for i in range(len(wksp_index_vars)):
            crd_tensor = llir.Var(
                name=f"{intermediate_tensor_var.get_name()}{i}_crd_tensor",
                type=llir.DataType.TORCH_TENSOR,
            )
            intermediate_crd_tensors.append(crd_tensor)

            # torch::Tensor T0_crd_tensor = torch::from_blob(...)
            assembly_stmts.append(
                llir.VarInit(
                    var=crd_tensor,
                    value=llir.FunctionCall(
                        name="torch::from_blob",
                        args=[
                            llir.Var(
                                name=f"{intermediate_crd_vecs[i].name}.data()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{{{intermediate_crd_vecs[i].name}.size()}}",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{intermediate_crd_vecs[i].name}.get_deleter()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name="torch::kInt",
                                type=llir.DataType.NO_TYPE,
                            ),
                        ],
                    ),
                )
            )

            # int* T0_crd = T0_crd_tensor.data_ptr<int>();
            assembly_stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{intermediate_tensor_var.get_name()}{i}_crd",
                        type=llir.DataType.PTR_INT,
                    ),
                    value=llir.Var(
                        name=f"{crd_tensor.name}.data_ptr<int>()",
                        type=llir.DataType.PTR_INT,
                    ),
                )
            )

        # torch::Tensor T_val_tensor = torch::from_blob(...)
        val_tensor = llir.Var(
            name=f"{intermediate_tensor_var.get_name()}_val_tensor",
            type=llir.DataType.TORCH_TENSOR,
        )
        assembly_stmts.append(
            llir.VarInit(
                var=val_tensor,
                value=llir.FunctionCall(
                    name="torch::from_blob",
                    args=[
                        llir.Var(
                            name=f"{intermediate_val_vec.name}.data()",
                            type=llir.DataType.NO_TYPE,
                        ),
                        llir.Var(
                            name=f"{{{intermediate_val_vec.name}.size()}}",
                            type=llir.DataType.NO_TYPE,
                        ),
                        llir.Var(
                            name=f"{intermediate_val_vec.name}.get_deleter()",
                            type=llir.DataType.NO_TYPE,
                        ),
                        llir.Var(
                            name=get_pytorch_c_dtype_str(intermediate_tensor_var.dtype),
                            type=llir.DataType.NO_TYPE,
                        ),
                    ],
                ),
            )
        )

        # float* T_val = T_val_tensor.data_ptr<float>();
        data_type = dtype_to_c_datatype(intermediate_tensor_var.dtype)
        ptr_type = llir.DataType.ptr_type(intermediate_tensor_var.dtype)
        assembly_stmts.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{intermediate_tensor_var.get_name()}_val",
                    type=ptr_type,
                ),
                value=llir.Var(
                    name=f"{val_tensor.name}.data_ptr<{data_type.value}>()",
                    type=ptr_type,
                ),
            )
        )

        # Build conversion CIN: result[i,j,...] = T[i,j,...]
        # then wrap in ForAll loops in mode_order and recursively lower
        sorted_result_index_vars = result_tensor_access.get_sorted_index_vars()

        lhs = f'result_tensor[{", ".join(["result_index_vars[{i}]".format(i=i) for i in range(len(result_index_vars))])}]'
        rhs = f'intermediate_tensor_var[{", ".join(["result_index_vars[{i}]".format(i=i) for i in range(len(result_index_vars))])}]'
        exec(f"{lhs} = {rhs}")

        cin_rhs = "result_tensor._assignment"
        for i in range(len(sorted_result_index_vars) - 1, -1, -1):
            cin_rhs = f"ForAll(sorted_result_index_vars[{i}], {cin_rhs})"
        cin_stmt = eval(cin_rhs)

        # Recursively lower the conversion CIN
        result_conversion_stmts = self.lower_IndexStmt(cin_stmt)
        if not isinstance(result_conversion_stmts, list):
            result_conversion_stmts = [result_conversion_stmts]

        return [
            llir.BlankLine(),
            llir.Comment("Lower outer consumer CIN"),
            *vec_decl_stmts,
            llir.BlankLine(),
            wksp_sort_stmt,
            loop_stmt,
            llir.BlankLine(),
            *assembly_stmts,
            llir.BlankLine(),
            *result_conversion_stmts,
        ]

    def lower_ConsumerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower a ConsumerIndexStmt to LLIR
        """
        if stmt.parent == self.outermost_stmt:
            return self.lower_outer_ConsumerIndexStmt(stmt)

        workspaces = stmt.get_workspaces()
        wksp = workspaces[0]
        workspace_accesses = stmt.get_workspace_accesses()
        wksp_access: WorkspaceAccess = workspace_accesses[0]
        wksp_index_vars = wksp_access.get_index_vars()

        result_tensor_accesses = stmt.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]
        result_tensor_name = result_tensor_access.get_tensor().get_name()

        # If the wksp_index_var is None, that means we just have a scalar
        # workspace
        if not wksp_index_vars:
            stmts: List[llir.Stmt] = []

            index_var = result_tensor_access.get_sorted_index_vars()[-1]
            level = result_tensor_access.level_of_index_var(index_var)
            level_type = result_tensor_access.level_type_of_index_var(index_var)

            wksp_var = llir.Var(
                name=f"{wksp.get_name()}",
                type=llir.DataType.NO_TYPE,
            )

            # Guard assignment: if wksp != 0
            if level_type == LevelType.DENSE:
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=[
                            llir.Assign(
                                var=llir.Var(
                                    name=f"{result_tensor_name}_values[p{result_tensor_name}{level}]",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                value=wksp_var,
                            ),
                        ],
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=[
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}{level}_crd.push_back",
                                args=[
                                    llir.Var(
                                        name=index_var.name,
                                        type=llir.DataType.INT,
                                    )
                                ],
                            ),
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}_values.push_back",
                                args=[wksp_var],
                            ),
                            llir.Increment(
                                var=llir.Var(
                                    name=f"p{result_tensor_name}{level}",
                                    type=llir.DataType.INT,
                                )
                            ),
                        ],
                    )
                )
            elif level_type == LevelType.COORDINATE:
                # For COO, push coordinates at ALL coordinate levels
                sorted_ivars = result_tensor_access.get_sorted_index_vars()
                push_stmts: List[llir.Stmt] = []
                for lvl, ivar in enumerate(sorted_ivars):
                    lt = result_tensor_access.level_type_of_index_var(ivar)
                    if lt == LevelType.COORDINATE:
                        push_stmts.append(
                            llir.FunctionCallStmt(
                                name=f"{result_tensor_name}{lvl}_crd.push_back",
                                args=[
                                    llir.Var(
                                        name=ivar.name,
                                        type=llir.DataType.INT,
                                    )
                                ],
                            )
                        )
                push_stmts.append(
                    llir.FunctionCallStmt(
                        name=f"{result_tensor_name}_values.push_back",
                        args=[wksp_var],
                    )
                )
                stmts.append(
                    llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=wksp_var,
                            right=llir.Literal(value=0),
                        ),
                        then_body=push_stmts,
                    )
                )
            else:
                raise NotImplementedError(
                    f"TODO: need to handle assembly of workspace with {level_type} level"
                )
            return [
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                *stmts,
                llir.BlankLine(),
            ]

        wksp_last_index_var = wksp_index_vars[-1]

        if wksp_last_index_var.has_parent:
            curr_index_var = wksp_last_index_var.parent
        else:
            curr_index_var = wksp_last_index_var

        level = result_tensor_access.level_of_index_var(curr_index_var)
        level_type = result_tensor_access.level_type_of_index_var(curr_index_var)

        parent_index_var = None
        parent_level_type = None
        if level > 0:
            parent_index_var = result_tensor_access.get_parent_index_var(curr_index_var)
            assert parent_index_var is not None, "parent_index_var should not be None"
            parent_level_type = result_tensor_access.level_type_of_index_var(
                parent_index_var
            )

        # p<result tensor's name><result level>
        result_level_iterator_name = f"p{result_tensor_name}{level}"
        result_level_iterator_llir = llir.Var(
            name=result_level_iterator_name,
            type=llir.DataType.NO_TYPE,
        )

        # call .sort() on the workspace
        # <wksp's name>.sort();
        wksp_sort_stmt = llir.FunctionCallStmt(
            name=f"{wksp.get_name()}.sort",
            args=[],
        )

        # Dense accumulator workspace: write to the result tensor.
        # When the result level is dense and contiguous, use memcpy
        # (pure store, avoids cold read-modify-write on large output).
        if wksp_access.is_dense():
            assert (
                len(wksp_index_vars) == 1
            ), "dense workspace has more than 1 index var"
            wksp_index_var = wksp_index_vars[0]

            if not wksp_index_var.tile_size_var:
                # Check if the result level is dense (contiguous layout).
                result_level_type = result_tensor_access.level_type_of_index_var(
                    wksp_index_var
                )
                result_is_dense = (result_level_type is not None
                                   and result_level_type.name == "DENSE")

                if result_is_dense:
                    # Emit memcpy: the workspace has the full row, write once.
                    wname = wksp.get_name()
                    size_var = wksp_index_var.size_llir_var.name
                    ctype_str = dtype_to_c_datatype(wksp.dtype).value
                    # Resolve the base pointer for this row in C.
                    resolve_stmts = result_tensor_access.get_level_iterator_resolve_stmts(level=level)
                    # The iterator for the row start: pC<level> with j=0
                    # is just pC_prev * C_level_size (which is result_level_iterator_name with j=0).
                    # We can compute it as: &C_values[pC0 * C1_size]
                    prev_iter = f"p{result_tensor_name}{level - 1}" if level > 0 else "0"
                    c_level_size = f"{result_tensor_name}{level}_size"
                    return [
                        llir.BlankLine(),
                        llir.Comment("Write workspace to output (memcpy — pure store)"),
                        llir.RawStmt(
                            code=(
                                f"memcpy(&{result_tensor_name}_values"
                                f"[{prev_iter} * {c_level_size}], "
                                f"{wname}, {size_var} * sizeof({ctype_str}))"
                            )
                        ),
                    ]

                # Fallback: element-by-element assignment (= not +=)
                loop_var = llir.Var(
                    name=f"{wksp_index_var.name}",
                    type=llir.DataType.INT,
                )

                loop_body: List[llir.Stmt] = []

                loop_body.extend(
                    result_tensor_access.get_level_iterator_resolve_stmts(level=level)
                )

                loop_body.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{result_tensor_name}_values[{result_level_iterator_name}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        value=llir.Var(
                            name=f"{wksp.get_name()}[{loop_var.name}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        op=AssignOp.ASSIGN,
                    )
                )

                for_loop = llir.ForLoop(
                    init=llir.VarInit(
                        var=loop_var,
                        value=llir.Literal(0),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=loop_var,
                        right=wksp_index_var.size_llir_var,
                    ),
                    update=llir.Increment(
                        var=loop_var,
                    ),
                    body=loop_body,
                )
                return [
                    llir.BlankLine(),
                    llir.Comment(
                        "Write workspace to output"
                    ),
                    for_loop,
                ]

            assert (
                wksp_index_var.tile_size_var and wksp_index_var.is_inner
            ), "Dense accumulator used not for tiling"

            # For loop
            # for (int <wksp index var> = 0; <wksp index var> < <wksp index var bound>; <wksp index var>++) {
            #    <body statement>
            # }
            loop_var = llir.Var(
                name=f"{wksp_index_var.name}",
                type=llir.DataType.INT,
            )

            loop_body: List[llir.Stmt] = []

            # <result tensor name>_values[<result level iterator>] = <wksp's name>[<wksp index var>];

            loop_body.extend(wksp_index_var.parent.get_resolve_llir_stmts())

            loop_body.extend(
                result_tensor_access.get_level_iterator_resolve_stmts(level=level)
            )

            loop_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{result_tensor_name}_values[{result_level_iterator_name}]",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{wksp.get_name()}[{loop_var.name}]",
                        type=llir.DataType.NO_TYPE,
                    ),
                    op=AssignOp.ADD_ASSIGN,
                )
            )

            for_loop = llir.ForLoop(
                init=llir.VarInit(
                    var=loop_var,
                    value=llir.Literal(0),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=loop_var,
                    right=llir.Var(
                        name=wksp_index_var.tile_size_var.name,
                        type=llir.DataType.INT,
                    ),
                ),
                update=llir.Increment(
                    var=loop_var,
                ),
                body=loop_body,
                unroll=True,
            )

            return [
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                for_loop,
            ]

        # For loop
        # for (const auto& pair : <wksp's name>) {
        #    <body statement>
        # }

        loop_var = llir.Var(
            name="it",
            type=llir.DataType.CONST_AUTO_REF,
        )

        loop_array = llir.Var(
            name=f"{wksp.get_name()}",
            type=llir.DataType.AUTO,
        )

        loop_body: List[llir.Stmt] = []

        # int <wksp_access's first index var's name> = it->first[0];
        # int <wksp_access's second index var's name> = it->first[1];
        # ...
        # DONE: if the workspace is one dimensional, then just do .first without the index
        # vars
        if len(wksp_access.get_index_vars()) == 1:
            loop_body.append(
                llir.VarInit(
                    var=llir.Var(
                        name=wksp_access.get_index_vars()[0].name,
                        type=llir.DataType.INT,
                    ),
                    value=llir.Var(
                        name=f"{loop_var.name}.first",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )
        else:
            for i, index_var in enumerate(wksp_access.get_index_vars()):
                loop_body.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=index_var.name,
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"{loop_var.name}.first[{i}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                    )
                )
        # <wksp's ctype> <wksp's name>_value = it->second;
        loop_body.append(
            llir.VarInit(
                var=llir.Var(
                    name=f"{wksp.get_name()}_value",
                    type=dtype_to_c_datatype(wksp.dtype),
                ),
                value=llir.Var(
                    name=f"{loop_var.name}.second",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # Add blank line
        loop_body.append(llir.BlankLine())

        # <result tensor name>_values[<result level iterator>] = <wksp's name>_value;
        loop_body.append(
            llir.Assign(
                var=llir.Var(
                    name=f"{result_tensor_name}_values[{result_level_iterator_name}]",
                    type=llir.DataType.NO_TYPE,
                ),
                value=llir.Var(
                    name=f"{wksp.get_name()}_value",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # Set coordinate
        # <result tensor name><level>_crd[<result level iterator>] = <wksp_access's first index var's name>;
        loop_body.append(
            llir.Assign(
                var=llir.Var(
                    name=f"{result_tensor_name}{level}_crd[{result_level_iterator_name}]",
                    type=llir.DataType.NO_TYPE,
                ),
                value=llir.Var(
                    name=f"{wksp_access.get_index_vars()[0].name}",
                    type=llir.DataType.NO_TYPE,
                ),
            )
        )

        # If the parent level is COORDINATE, also set the parent level's coordinate
        if parent_level_type == LevelType.COORDINATE:
            loop_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{result_tensor_name}{level - 1}_crd[{result_level_iterator_name}]",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{parent_index_var.name}",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

        # <result level iterator>++;
        loop_body.append(
            llir.Increment(
                var=result_level_iterator_llir,
            )
        )

        loop_stmt = llir.ForLoopAuto(
            var=loop_var,
            array=loop_array,
            body=loop_body,
        )

        assembly_stmts: List[llir.Stmt] = []

        if level_type == LevelType.COMPRESSED:
            assembly_stmts.extend(
                [
                    llir.BlankLine(),
                    llir.Comment("Assembly compressed _level indices"),
                ]
            )

            # if _level is > 0 and parent _level is also sparse, we need to set
            # the parent _level's crd
            if level > 0:
                assert parent_index_var is not None, "Parent index var is None"
                if parent_level_type == LevelType.COMPRESSED:
                    assembly_stmts.append(
                        # e.g.
                        # if (A1_pos.back() < pA1) {
                        #     A0_crd.push_back(i);
                        # }
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op="<",
                                left=llir.FunctionCall(
                                    name=f"{result_tensor_name}{level}_pos.back",
                                    args=[],
                                ),
                                right=llir.Var(
                                    name=f"p{result_tensor_name}{level}",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            then_body=[
                                llir.FunctionCallStmt(
                                    name=f"{result_tensor_name}{level - 1}_crd.push_back",
                                    args=[
                                        llir.Var(
                                            name=parent_index_var.name,
                                            type=llir.DataType.INT,
                                        )
                                    ],
                                ),
                            ],
                        )
                    )
            # Assemble pos array for this compressed level:
            # - Dense parent: A1_pos.push_back(A1_crd.size())
            # - Compressed parent: A1_pos[A0_crd.size()] = A1_crd.size()
            assembled_pos_array = False
            if level > 0:
                assert parent_index_var is not None, "Parent index var is None"
                if parent_level_type == LevelType.COMPRESSED:
                    # A1_pos[A0_crd.size()] = A1_crd.size()
                    assembly_stmts.append(
                        llir.Assign(
                            var=llir.Var(
                                name=f"{result_tensor_name}{level}_pos[{result_tensor_name}{level - 1}_crd.size()]",
                                type=llir.DataType.INT,
                            ),
                            value=llir.FunctionCall(
                                name=f"{result_tensor_name}{level}_crd.size",
                                args=[],
                            ),
                        )
                    )
                    assembled_pos_array = True

            if not assembled_pos_array:
                # e.g. A1_pos[pA1] = A1_crd.size()
                # assembly_stmts.append(
                #     llir.Assign(
                #         var=llir.Var(
                #             name=f"{result_tensor_name}{level}_pos[p{result_tensor_name}{level}]",
                #             type=llir.DataType.INT,
                #         ),
                #         value=llir.FunctionCall(
                #             name=f"{result_tensor_name}{level}_crd.size",
                #             args=[],
                #         ),
                #     )
                # )
                # e.g. A1_pos[A1_pos_index + 1] = A1_crd.size()
                assembly_stmts.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{result_tensor_name}{level}_pos[{result_tensor_name}{level}_pos_index + 1]",
                            type=llir.DataType.INT,
                        ),
                        value=llir.FunctionCall(
                            name=f"{result_tensor_name}{level}_crd.size",
                            args=[],
                        ),
                    )
                )
                # assembly_stmts.append(
                #     # e.g. A1_pos.push_back(pA1))
                #     llir.FunctionCallStmt(
                #         name=f"{result_tensor_name}{level}_pos.push_back",
                #         args=[
                #             llir.Var(
                #                 name=f"{result_tensor_name}{level}_crd.size()",
                #                 # name=f"p{result_tensor_var.name}{_level}",
                #                 type=llir.DataType.INT,
                #             )
                #         ],
                #     )
                # )

        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            wksp_sort_stmt,
            loop_stmt,
            llir.BlankLine(),
            *assembly_stmts,
        ]

    def lower_IndexStmt(
        self, stmt: IndexStmt, recurse=False
    ) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower an IndexStmt to LLIR
        """

        if not self.outermost_stmt:
            self.outermost_stmt = stmt

        if isinstance(stmt, TensorAssign):
            return self.lower_TensorAssign(stmt)

        # loop_order_allow_short_circuit = all_free_var_loops_before_reduction_loops(stmt)

        # Create tensor results and rhs IR variables
        result_tensor_vars: List[TensorVar] = stmt.get_result_tensor_vars()
        # TODO: need to handle multiple result tensors
        self.result_tensor_var = result_tensor_vars[0]
        non_workspace_result_tensor_vars = [
            x for x in result_tensor_vars if not isinstance(x, Workspace)
        ]
        if not self.final_result_tensor_var:
            self.final_result_tensor_var = (
                non_workspace_result_tensor_vars[0]
                if non_workspace_result_tensor_vars
                else None
            )
        result_tensor_accesses = stmt.get_result_tensor_accesses()
        self.result_tensor_access = result_tensor_accesses[0]
        non_workspace_result_tensor_accesses = [
            x for x in result_tensor_accesses if not isinstance(x.tensor, Workspace)
        ]
        if not self.final_result_tensor_access:
            self.final_result_tensor_access = (
                non_workspace_result_tensor_accesses[0]
                if non_workspace_result_tensor_accesses
                else None
            )

        rhs_tensor_vars: List[TensorVar] = stmt.get_rhs_tensor_vars()
        rhs_tensor_accesses: List[TensorAccess] = stmt.get_rhs_tensor_accesses()
        # rhs_tensor_vars_llir: List[llir.Expr] = [
        #     self.lower_TensorVar(tv) for tv in rhs_tensor_vars
        # ]

        tile_size_vars = stmt.get_tile_size_vars()
        tile_size_vars_init_stmts: List[llir.Stmt] = (
            [llir.BlankLine(), llir.Comment("Initialize tile sizes")]
            if tile_size_vars
            else []
        )
        for tile_size_var in tile_size_vars:
            tile_size_vars_init_stmts.append(tile_size_var.llir_var_init)

        self.need_compute.extend(result_tensor_vars)

        if recurse or stmt != self.outermost_stmt:
            if isinstance(stmt, ForAll):
                return self.lower_ForAll(stmt)
            if isinstance(stmt, Where):
                return self.lower_Where(stmt)

        tensor_value_array_init_stmts: List[llir.Stmt] = []
        result_level_indices_init_stmts: List[llir.Stmt] = []

        for result_tensor_var in non_workspace_result_tensor_vars:
            self.tensor_var_to_llir[result_tensor_var] = self.lower_TensorVar(
                result_tensor_var
            )
            assembler = ResultTensorAssembler(result_tensor_var)
            tensor_value_array_init_stmts.extend(assembler.emit_value_array_init())
            result_level_indices_init_stmts.extend(assembler.emit_level_indices_init())

        if result_level_indices_init_stmts:
            result_level_indices_init_stmts = [
                llir.Comment("Init result level indices"),
                *result_level_indices_init_stmts,
            ]

        # Generate iterator bounds
        tensor_level_array_assign_stmts: List[llir.Stmt] = []

        for tensor in rhs_tensor_vars:
            tensor_level_array_assign_stmts.append(llir.BlankLine())
            tensor_level_array_assign_stmts.append(
                llir.Comment(f"Get {tensor.get_name()}'s level & value arrays")
            )
            tensor_level_array_assign_stmts.extend(self.get_level_arrays(tensor))
            tensor_level_array_assign_stmts.append(self.get_val_ptr_stmt(tensor))

        # Generate per-level size variables for each dense level in result tensor
        result_tensor_level_sizes: List[llir.Stmt] = []
        for i, level_type in enumerate(self.result_tensor_var.get_level_types()):
            if level_type == LevelType.DENSE:
                result_tensor_level_sizes.append(
                    llir.VarInit(
                        llir.Var(
                            name=f"{self.result_tensor_var.get_name()}{i}_size",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"result_shape[{i}]",
                            type=llir.DataType.INT,
                        ),
                    )
                )

        if result_tensor_level_sizes:
            result_tensor_level_sizes = [
                llir.Comment("Init result tensor level sizes"),
                *result_tensor_level_sizes,
            ]

            # A mapping from IndexVar to a list of (TensorVar, _level: int, LevelType) tuples
        self.index_var_to_rhs_tensor_level_type = {}
        for tensor_access in rhs_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            mode_order = tensor_var.get_mode_order()
            for i, index_var in enumerate(index_vars):
                index_var_level = mode_order[i]
                if index_var not in self.index_var_to_rhs_tensor_level_type:
                    self.index_var_to_rhs_tensor_level_type[index_var] = []
                self.index_var_to_rhs_tensor_level_type[index_var].append(
                    [tensor_var, index_var_level, tensor_level_types[index_var_level]]
                )

        self.index_var_to_result_tensor_level_type = {}
        for tensor_access in result_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            if not index_vars:
                continue
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            mode_order = tensor_var.get_mode_order()
            for i, index_var in enumerate(index_vars):
                index_var_level = mode_order[i]
                if index_var not in self.index_var_to_result_tensor_level_type:
                    self.index_var_to_result_tensor_level_type[index_var] = []
                self.index_var_to_result_tensor_level_type[index_var].append(
                    [tensor_var, index_var_level, tensor_level_types[index_var_level]]
                )

        # Initialize index into result if any _level if compressed
        # Find last compressed _level of the result tensor, if any
        result_last_compressed_index_var = None
        for (
            index_var,
            tensor_level_type_list,
        ) in self.index_var_to_result_tensor_level_type.items():
            # TODO: deal with multiple outputs
            tensor_var, level, level_type = tensor_level_type_list[0]
            if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                result_last_compressed_index_var = index_var

        result_index_init_stmts = []

        if result_last_compressed_index_var is not None:
            self.result_value_array_sparse_index_llir = llir.Var(
                # name=f"p{self.result_tensor_var.name}{self.result_tensor_access.level_of_index_var(result_last_compressed_index_var)}",
                name=f"p{self.result_tensor_var.name}{self.result_tensor_var.levels - 1}",
                type=llir.DataType.INT,
            )
            self.result_tensor_value_index_var_dict[
                result_last_compressed_index_var
            ] = self.result_value_array_sparse_index_llir

            result_index_init_stmts.append(
                llir.VarInit(
                    var=self.result_value_array_sparse_index_llir,
                    value=llir.Literal(value=0, data_type=llir.DataType.INT),
                )
            )

        # Finally, return function that computes the result
        if stmt == self.outermost_stmt:
            kernel_args: List[llir.Var] = []

            kernel_args.append(
                llir.Var(
                    name="result_shape",
                    type=llir.DataType.STD_VECTOR_INT,
                )
            )

            for tensor in rhs_tensor_vars:
                kernel_args.append(
                    llir.Var(
                        name=f"{tensor.get_name()}_shape",
                        type=llir.DataType.STD_VECTOR_INT,
                    )
                )
                kernel_args.append(
                    llir.Var(
                        name=f"{tensor.get_name()}_mode_indices",
                        type=llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
                    )
                )
                kernel_args.append(
                    llir.Var(
                        name=f"{tensor.get_name()}_values",
                        type=llir.DataType.TORCH_TENSOR,
                    )
                )

            body_stmts: List[llir.Stmt] = []

            recurse_stmts = self.lower_IndexStmt(stmt, recurse=True)
            if not isinstance(recurse_stmts, list):
                recurse_stmts = [recurse_stmts]

            # Post-lowering optimizations on the LLIR
            self._insert_sparse_prefetch(recurse_stmts)

            body_stmts.extend(
                [
                    *result_tensor_level_sizes,
                    *tensor_level_array_assign_stmts,
                    llir.BlankLine(),
                    *result_level_indices_init_stmts,
                    llir.Comment("Initialize result value array"),
                    *tensor_value_array_init_stmts,
                    *tile_size_vars_init_stmts,
                    # *result_index_init_stmts,
                    llir.BlankLine(),
                    *recurse_stmts,
                ]
            )

            assert self.final_result_tensor_var is not None, "No final result tensor"
            final_assembler = ResultTensorAssembler(self.final_result_tensor_var)
            body_stmts.extend(final_assembler.emit_final_assembly())

            return llir.Function(
                return_type=llir.DataType.TACO_TENSOR,
                name="evaluate",
                args=kernel_args,
                body=body_stmts,
            )

        return []

    def _should_parallelize_outer_forall(self, index_var: IndexVar) -> bool:
        if not self.final_result_tensor_var or not self.final_result_tensor_var.is_dense():
            return False
        if not self.final_result_tensor_access:
            return False
        if self.final_result_tensor_access.has_index_var(index_var):
            return True
        if (
            index_var.has_parent
            and self.final_result_tensor_access.has_index_var(index_var.parent)
        ):
            return True
        for result_index_var in self.final_result_tensor_access.get_index_vars():
            if result_index_var.has_parent and result_index_var.parent == index_var:
                return True
        return False

    @staticmethod
    def _is_openmp_compatible_for_loop(for_loop: llir.ForLoop) -> bool:
        if not isinstance(for_loop.init, llir.VarInit):
            return False
        if not isinstance(for_loop.init.var, llir.Var):
            return False
        loop_var = for_loop.init.var

        if isinstance(for_loop.update, llir.Increment):
            if for_loop.update.var.name != loop_var.name:
                return False
        elif isinstance(for_loop.update, llir.Assign):
            if for_loop.update.var.name != loop_var.name:
                return False
            if for_loop.update.op not in (AssignOp.ADD_ASSIGN, AssignOp.SUB_ASSIGN):
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

    @classmethod
    def _has_sparse_inner_loop(cls, stmts: List[llir.Stmt]) -> bool:
        """Check if any ForLoop in stmts (or nested) iterates over a sparse level
        (identified by init value referencing a _pos array)."""
        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                if (isinstance(stmt.init, llir.VarInit)
                        and isinstance(stmt.init.value, llir.Var)
                        and "_pos[" in stmt.init.value.name):
                    return True
                if cls._has_sparse_inner_loop(stmt.body):
                    return True
        return False

    @staticmethod
    def _insert_sparse_prefetch(stmts: List[llir.Stmt]) -> None:
        """Walk the LLIR tree and insert software prefetch hints in sparse loops.

        When a sparse ForLoop (iterating pA1 from A1_pos[...] to pA1_end)
        contains a dense inner loop that accesses another tensor's values via
        ``B_val[coord * stride + ...]``, insert a prefetch for the *next*
        sparse element's corresponding row:

            if (pA1 + 1 < pA1_end)
              __builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);

        This hides the latency of indirect B-row loads which dominate SpMM.
        """
        import re
        for stmt in stmts:
            if not isinstance(stmt, llir.ForLoop):
                continue
            # Recurse into all ForLoop bodies first
            CINLowerer._insert_sparse_prefetch(stmt.body)

            # Detect sparse loop: init value contains _pos[
            if not (isinstance(stmt.init, llir.VarInit)
                    and isinstance(stmt.init.value, llir.Var)
                    and "_pos[" in stmt.init.value.name):
                continue

            # Extract iter var name (e.g. "pA1")
            iter_var = stmt.init.var.name  # e.g. "pA1"

            # Find the end variable from cond (e.g. "pA1_end")
            if not (isinstance(stmt.cond, llir.BinOp)
                    and isinstance(stmt.cond.right, llir.Var)):
                continue
            end_var = stmt.cond.right.name  # e.g. "pA1_end"

            # Find coordinate array in body: VarInit like k = A1_crd[pA1]
            crd_array = None
            for body_stmt in stmt.body:
                if (isinstance(body_stmt, llir.VarInit)
                        and isinstance(body_stmt.value, llir.Var)):
                    val_name = body_stmt.value.name
                    m = re.match(r'^(\w+_crd)\[' + re.escape(iter_var) + r'\]$',
                                 val_name)
                    if m:
                        crd_array = m.group(1)
                        break
            if not crd_array:
                continue

            # Find a dense values array and its stride by inspecting the inner
            # dense ForLoop.  We look for:
            #   VarInit pB1 = Add(Mul(pB0, B1_size), j)  → stride = B1_size
            #   Assign  C[pC1] += BinOp(*, A_val[pA1], B_val[pB1])
            #                                              → dense_val = B_val
            dense_val_array = None
            dense_stride = None
            for body_stmt in stmt.body:
                if not isinstance(body_stmt, llir.ForLoop):
                    continue
                # Collect position vars and their strides from VarInit nodes
                pos_to_stride: Dict[str, str] = {}
                for inner_stmt in body_stmt.body:
                    if (isinstance(inner_stmt, llir.VarInit)
                            and isinstance(inner_stmt.value, llir.Add)):
                        add = inner_stmt.value
                        # Pattern: Mul(base, stride) + offset
                        if (isinstance(add.left, llir.BinOp)
                                and add.left.op == "*"
                                and isinstance(add.left.right, llir.Var)):
                            pos_to_stride[inner_stmt.var.name] = add.left.right.name
                # Now find Assign nodes that use _val arrays indexed by those pos vars
                for inner_stmt in body_stmt.body:
                    if not isinstance(inner_stmt, llir.Assign):
                        continue
                    # Find _val[pos] references in the expression tree
                    val = inner_stmt.value
                    found = CINLowerer._find_val_array_access(val, pos_to_stride)
                    if found:
                        dense_val_array, dense_stride = found
                        break
                if dense_val_array:
                    break

            if not dense_val_array or not dense_stride:
                continue

            # Insert prefetch for the next sparse element's B-row
            prefetch_code = (
                f"if ({iter_var} + 1 < {end_var}) "
                f"__builtin_prefetch(&{dense_val_array}["
                f"{crd_array}[{iter_var} + 1] * {dense_stride}], 0, 1)"
            )
            stmt.body.insert(0, llir.RawStmt(code=prefetch_code, add_semicolon=True))

    @staticmethod
    def _find_val_array_access(
        expr: llir.Expr, pos_to_stride: Dict[str, str],
    ) -> Optional[tuple]:
        """Recursively search an expression for references to a _val array
        indexed by a position variable in *pos_to_stride*.

        LLIR may represent array accesses either as structured ArrayAccess
        nodes or as flat Var names like ``"B_val[pB1]"``.  This handles both.

        Returns ``(val_array_name, stride_name)`` or *None*.
        """
        import re
        if isinstance(expr, llir.ArrayAccess):
            if (isinstance(expr.array, llir.Var)
                    and "_val" in expr.array.name
                    and isinstance(expr.index, llir.Var)
                    and expr.index.name in pos_to_stride):
                return (expr.array.name, pos_to_stride[expr.index.name])
        # Flat Var with name like "B_val[pB1]"
        if isinstance(expr, llir.Var):
            m = re.match(r'^(\w+_val)\[(\w+)\]$', expr.name)
            if m:
                arr_name, pos_var = m.group(1), m.group(2)
                if pos_var in pos_to_stride:
                    return (arr_name, pos_to_stride[pos_var])
        # Recurse into BinOp children
        if isinstance(expr, llir.BinOp):
            left = CINLowerer._find_val_array_access(expr.left, pos_to_stride)
            if left:
                return left
            return CINLowerer._find_val_array_access(expr.right, pos_to_stride)
        return None

    def _mark_first_for_loop_parallel(self, stmts: List[llir.Stmt]) -> None:
        for llir_stmt in stmts:
            if isinstance(llir_stmt, llir.ForLoop) and self._is_openmp_compatible_for_loop(llir_stmt):
                llir_stmt.omp_parallel_for = True
                # Use dynamic scheduling when body contains sparse loops,
                # since rows have varying nnz counts causing load imbalance.
                if self._has_sparse_inner_loop(llir_stmt.body):
                    llir_stmt.omp_schedule = "dynamic, 64"
                # Hoist per-thread workspace alloc/free outside the for loop
                # but inside the OMP parallel region.
                alloc = getattr(self, '_workspace_alloc_stmts', [])
                free = getattr(self, '_workspace_free_stmts', [])
                if alloc or free:
                    llir_stmt.pre_parallel_body = alloc or None
                    llir_stmt.post_parallel_body = free or None
                return

    @classmethod
    def _collect_output_arrays(cls, stmts: List[llir.Stmt], output_arrays: List[str]) -> None:
        """Collect output array names (e.g., D_values, D0_crd) from Assign stmts."""
        import re
        for stmt in stmts:
            if isinstance(stmt, llir.Assign) and isinstance(stmt.var, llir.Var):
                m = re.match(r'^(\w+)\[', stmt.var.name)
                if m:
                    arr_name = m.group(1)
                    if arr_name not in output_arrays:
                        output_arrays.append(arr_name)
            elif isinstance(stmt, llir.ForLoop):
                cls._collect_output_arrays(stmt.body, output_arrays)
            elif isinstance(stmt, llir.WhileLoop):
                cls._collect_output_arrays(stmt.body, output_arrays)
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    cls._collect_output_arrays(stmt.then_body, output_arrays)
                if stmt.else_body:
                    cls._collect_output_arrays(stmt.else_body, output_arrays)

    @classmethod
    def _replace_output_pos_with_input_pos(cls, stmts: List[llir.Stmt], input_iter_var: str) -> None:
        """Replace shared output position variable (pD1) with input iterator position
        for thread-safe parallel output. Finds inner ForLoop over pA1..pA1_end and
        replaces pD<N> references with pA1 in the loop body."""
        import re
        for stmt in stmts:
            if isinstance(stmt, llir.ForLoop):
                # Find the sparse inner loop iterating pA1
                if (isinstance(stmt.init, llir.VarInit)
                        and isinstance(stmt.init.var, llir.Var)
                        and stmt.init.var.name.startswith("p")):
                    inner_pos_var = stmt.init.var.name  # e.g. "pA1"
                    cls._rewrite_output_pos_vars(stmt.body, inner_pos_var)
                else:
                    cls._replace_output_pos_with_input_pos(stmt.body, input_iter_var)
            elif isinstance(stmt, llir.WhileLoop):
                cls._replace_output_pos_with_input_pos(stmt.body, input_iter_var)
            elif isinstance(stmt, llir.IfThenElse):
                if stmt.then_body:
                    cls._replace_output_pos_with_input_pos(stmt.then_body, input_iter_var)
                if stmt.else_body:
                    cls._replace_output_pos_with_input_pos(stmt.else_body, input_iter_var)

    @classmethod
    def _rewrite_output_pos_vars(cls, stmts: List[llir.Stmt], input_pos_var: str) -> None:
        """Replace output position variables (pD1, pD0) in Assign/Increment stmts
        with the input position variable for thread-safe writes."""
        import re
        to_remove = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, llir.Assign) and isinstance(stmt.var, llir.Var):
                # Replace pD<N> in array index: D_values[pD1] -> D_values[pA1]
                m = re.match(r'^(.+)\[p([A-Z])\d+\]$', stmt.var.name)
                if m and m.group(1).startswith(m.group(2)):
                    # This is an output array write like D_values[pD1] or D0_crd[pD1]
                    stmt.var.name = re.sub(r'\[p[A-Z]\d+\]', f'[{input_pos_var}]', stmt.var.name)
            elif isinstance(stmt, llir.Increment) and isinstance(stmt.var, llir.Var):
                # Remove pD1++ (no longer needed, position is input-derived)
                if re.match(r'^p[A-Z]\d+$', stmt.var.name):
                    # Check it's an output pos var, not an input one
                    if stmt.var.name != input_pos_var:
                        to_remove.append(i)
            elif isinstance(stmt, llir.ForLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var)
            elif isinstance(stmt, llir.WhileLoop):
                cls._rewrite_output_pos_vars(stmt.body, input_pos_var)
        for i in reversed(to_remove):
            stmts.pop(i)

    def _should_parallelize_coo_outer(self, index_var: IndexVar) -> bool:
        """Check if the outermost ForAll iterates a COORDINATE level and
        the output is all-COO, suitable for group-based parallelization."""
        if not self.final_result_tensor_var or not self.final_result_tensor_access:
            return False
        # Output must be all-coordinate (COO)
        for lt in self.final_result_tensor_var.get_level_types():
            if lt != LevelType.COORDINATE:
                return False
        return True

    @classmethod
    def _transform_coo_loop_for_openmp(cls, stmts: List[llir.Stmt]) -> List[llir.Stmt]:
        """Transform the outer COO WhileLoop into a group-indexed ForLoop
        with OpenMP parallelism.

        Finds the outermost WhileLoop that iterates over COO coordinate
        levels (identified by the pA0 = pA1_end update pattern) and replaces
        it with a pre-scan + parallel for over row groups.
        """
        result: List[llir.Stmt] = []
        transformed = False

        for stmt in stmts:
            if transformed or not isinstance(stmt, (llir.WhileLoop, llir.ForLoop)):
                result.append(stmt)
                continue

            body = stmt.body
            coo_update = None
            iter_var = None
            end_var = None

            # Detect COO outer loop: ForLoop with non-standard update pA0 = pA1_end
            if isinstance(stmt, llir.ForLoop):
                if (isinstance(stmt.update, llir.Assign)
                        and isinstance(stmt.update.var, llir.Var)
                        and isinstance(stmt.update.value, llir.Var)
                        and "_end" in stmt.update.value.name):
                    iter_var = stmt.update.var.name
                    end_var = stmt.update.value.name
                    coo_update = stmt.update  # sentinel, won't be in body
                else:
                    result.append(stmt)
                    continue
            else:
                # WhileLoop: look for pA0 = pA1_end in body
                for body_stmt in body:
                    if (isinstance(body_stmt, llir.Assign)
                            and isinstance(body_stmt.var, llir.Var)
                            and body_stmt.var.name.startswith("p")
                            and isinstance(body_stmt.value, llir.Var)
                            and "_end" in body_stmt.value.name
                            and body_stmt.op == AssignOp.ASSIGN):
                        coo_update = body_stmt
                        iter_var = body_stmt.var.name
                        end_var = body_stmt.value.name

            if coo_update is None:
                result.append(stmt)
                continue

            # Find the coordinate array name from VarInit in body
            # e.g., i = A0_crd[pA0]
            crd_array = None
            coord_var_name = None
            for body_stmt in body:
                if (isinstance(body_stmt, llir.VarInit)
                        and isinstance(body_stmt.value, llir.Var)
                        and "_crd[" in body_stmt.value.name):
                    val_name = body_stmt.value.name
                    bracket_pos = val_name.index("[")
                    crd_array = val_name[:bracket_pos]
                    coord_var_name = body_stmt.var.name
                    break

            if crd_array is None:
                result.append(stmt)
                continue

            # Extract the end bound from the loop condition
            outer_end_var = None
            if (isinstance(stmt.cond, llir.BinOp)
                    and isinstance(stmt.cond.right, llir.Var)):
                outer_end_var = stmt.cond.right.name

            if outer_end_var is None:
                result.append(stmt)
                continue

            # Build the inner body: everything except the COO advance
            inner_body = [s for s in body if s is not coo_update]

            # Remove from inner body:
            # 1. The coordinate VarInit (we set it from _group_starts)
            # 2. The "find iterator end" WhileLoop (group boundaries
            #    already encode this)
            # 3. The VarInit for pA1_end (already set in group header)
            inner_body_filtered = []
            for s in inner_body:
                # Remove: int i = A0_crd[pA0]
                if (isinstance(s, llir.VarInit)
                        and isinstance(s.value, llir.Var)
                        and "_crd[" in s.value.name
                        and s.var.name == coord_var_name):
                    continue
                # Remove: while (pA1_end < pA0_end && ...) { pA1_end++; }
                if isinstance(s, llir.WhileLoop):
                    # Check if this is the iterator-end-finding loop
                    if any(isinstance(bs, llir.Increment) for bs in s.body):
                        continue
                # Remove: pA1_end = pA0 + 1 (iterator end init)
                if (isinstance(s, llir.VarInit)
                        and isinstance(s.var, llir.Var)
                        and s.var.name == end_var):
                    continue
                # Remove: pA1_end = pA0 + 1 (as Assign)
                if (isinstance(s, llir.Assign)
                        and isinstance(s.var, llir.Var)
                        and s.var.name == end_var):
                    continue
                inner_body_filtered.append(s)

            # Thread-safety for output position: use input position pA1
            # as output position since nnz_out == nnz_in for SDDMM-like
            # kernels (no filtering). Replace pD1 references in the body.
            cls._replace_output_pos_with_input_pos(inner_body_filtered, iter_var)

            # Collect output array names that need pre-allocation
            output_arrays: List[str] = []
            import re as _re
            cls._collect_output_arrays(inner_body_filtered, output_arrays)

            # Pre-scan code
            pre_scan_stmts: List[llir.Stmt] = [
                llir.Comment("Pre-compute row group boundaries for OpenMP"),
                llir.VarDecl(
                    llir.Var(name="_group_starts", type=llir.DataType.CVECTOR_INT)
                ),
                llir.Assign(
                    var=llir.Var(name="_group_starts[0]", type=llir.DataType.INT),
                    value=llir.Literal(0),
                ),
                llir.VarInit(
                    var=llir.Var(name="_n_groups", type=llir.DataType.INT),
                    value=llir.Literal(1),
                ),
                # Scan loop
                llir.ForLoop(
                    init=llir.VarInit(
                        var=llir.Var(name="_p", type=llir.DataType.INT),
                        value=llir.Literal(1),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=llir.Var(name="_p", type=llir.DataType.INT),
                        right=llir.Var(name=outer_end_var, type=llir.DataType.INT),
                    ),
                    update=llir.Increment(
                        var=llir.Var(name="_p", type=llir.DataType.INT),
                    ),
                    body=[
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op="!=",
                                left=llir.Var(name=f"{crd_array}[_p]", type=llir.DataType.NO_TYPE),
                                right=llir.Var(name=f"{crd_array}[_p - 1]", type=llir.DataType.NO_TYPE),
                            ),
                            then_body=[
                                llir.Assign(
                                    var=llir.Var(name="_group_starts[_n_groups]", type=llir.DataType.INT),
                                    value=llir.Var(name="_p", type=llir.DataType.INT),
                                ),
                                llir.Increment(
                                    var=llir.Var(name="_n_groups", type=llir.DataType.INT),
                                ),
                            ],
                        ),
                    ],
                ),
                llir.Assign(
                    var=llir.Var(name="_group_starts[_n_groups]", type=llir.DataType.INT),
                    value=llir.Var(name=outer_end_var, type=llir.DataType.INT),
                ),
                llir.BlankLine(),
            ]

            # Group loop body
            group_body: List[llir.Stmt] = [
                llir.VarInit(
                    var=llir.Var(name=iter_var, type=llir.DataType.INT),
                    value=llir.Var(name="_group_starts[_g]", type=llir.DataType.INT),
                ),
                llir.VarInit(
                    var=llir.Var(name=end_var, type=llir.DataType.INT),
                    value=llir.Var(name="_group_starts[_g + 1]", type=llir.DataType.INT),
                ),
                llir.VarInit(
                    var=llir.Var(name=coord_var_name, type=llir.DataType.INT),
                    value=llir.Var(name=f"{crd_array}[{iter_var}]", type=llir.DataType.INT),
                ),
                *inner_body_filtered,
            ]

            # Group for loop with OpenMP
            group_loop = llir.ForLoop(
                init=llir.VarInit(
                    var=llir.Var(name="_g", type=llir.DataType.INT),
                    value=llir.Literal(0),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=llir.Var(name="_g", type=llir.DataType.INT),
                    right=llir.Var(name="_n_groups", type=llir.DataType.INT),
                ),
                update=llir.Increment(
                    var=llir.Var(name="_g", type=llir.DataType.INT),
                ),
                body=group_body,
                omp_parallel_for=True,
                omp_schedule="dynamic, 16",
            )

            # Pre-allocate output arrays for thread-safe parallel writes
            prealloc_stmts: List[llir.Stmt] = []
            for arr_name in output_arrays:
                prealloc_stmts.append(llir.RawStmt(
                    code=f"{arr_name}.resize({outer_end_var})",
                    add_semicolon=True,
                ))

            result.extend(pre_scan_stmts)
            result.extend(prealloc_stmts)
            result.append(group_loop)
            transformed = True

        return result

    def lower_ForAll(self, stmt: ForAll) -> List[llir.Stmt]:
        """
        Lower a ForAll to LLIR
        parent_index_var is the index var of the parent ForAll, if any
        """

        # Get index variable at this forall
        index_var = stmt.get_index_var()
        is_outermost_forall = not self.seen_outermost_forall
        if is_outermost_forall:
            self.seen_outermost_forall = True

        self.defined_index_vars.append(index_var)

        iter_lattice = IterationLattice(for_all_stmt=stmt, cin_lowerer=self)

        stmts: List[llir.Stmt] = []

        # if self.result_tensor_access and not self.result_tensor_access.has_index_var(
        #     index_var
        # ):
        #     stmts.append(llir.Comment(f"{index_var} not in result tensor access"))

        # If the result level for this index_var is dense, need to assemble the result by
        # setting the corresponding values in the result values array to 0
        if (
            self.result_tensor_access
            and self.result_tensor_access.has_index_var(index_var)
            and self.result_tensor_access.level_type_of_index_var(index_var)
            == LevelType.DENSE
        ):
            # If the parent level is not dense or has no parent level,
            # and the next levels are all dense
            # then we need to initialize result value array elements to 0
            level_of_index_var = self.result_tensor_access.level_of_index_var(index_var)
            if (
                (level_of_index_var == 0)
                or (
                    self.result_tensor_access.level_types()[level_of_index_var - 1]
                    != LevelType.DENSE
                )
            ) and all(
                [
                    self.result_tensor_access.level_types()[i] == LevelType.DENSE
                    for i in range(
                        level_of_index_var + 1, self.result_tensor_access.num_levels
                    )
                ]
            ):
                assert self.result_tensor_var, "Result tensor variable not set"

        stmts.extend(
            [
                *iter_lattice.get_iterator_init_stmts(),
                llir.BlankLine(),
                *iter_lattice.get_lattice_loops(),
            ]
        )
        if is_outermost_forall and self._should_parallelize_outer_forall(index_var):
            self._mark_first_for_loop_parallel(stmts)
        elif (is_outermost_forall
              and self._used_scalar_accum
              and self._should_parallelize_coo_outer(index_var)):
            stmts = self._transform_coo_loop_for_openmp(stmts)

        return stmts

    @staticmethod
    def lower_TensorVar(tensor_var: TensorVar) -> llir.Expr:
        """
        Lower a TensorVar to LLIR
        """
        return llir.Var(
            name=tensor_var.get_name(),
            type=llir.DataType.TACO_TENSOR,
        )

    @staticmethod
    def add_dependent_tensors(
        stmt: IndexStmt, tensor_vars: List[TensorVar]
    ) -> List[TensorVar]:
        """
        Add dependent tensor variables to the list of tensor variables
        Also return the list of dependent tensor variables
        Dependent tensors are those that are used in the RHS of the TensorAssign
        where the tensor variable on the LHS is in TENSOR_VARS
        """
        dependent_tensor_vars: List[TensorVar] = []

        class DependentTensorCollector(CINVisitorAccept):
            def visit_TensorAssign(self, tensor_assign: TensorAssign):
                if tensor_assign.get_lhs_tensor() in tensor_vars:
                    rhs_tensor_vars = tensor_assign.get_rhs_tensor_vars()
                    # add the ones that are not already in tensor_vars
                    for rhs_tensor_var in rhs_tensor_vars:
                        if rhs_tensor_var not in tensor_vars:
                            tensor_vars.append(rhs_tensor_var)
                            dependent_tensor_vars.append(rhs_tensor_var)

        tensor_collector = DependentTensorCollector()
        tensor_collector.visit(stmt)
        return dependent_tensor_vars

    @staticmethod
    def lower_IndexVar(ivar: IndexVar) -> llir.Var:
        """
        Lower an IndexVar to LLIR
        """
        return llir.Var(
            name=ivar.name,
            type=llir.DataType.INT,
        )
