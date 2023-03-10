from typing import List, Optional, Dict, Union

from src.scorch.compiler import llir
from src.scorch.compiler.cin import (
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
)
from src.scorch.compiler.iter_lattice import IterationLattice
from src.scorch.format import LevelType
from src.scorch.utils import (
    dtype_to_c_datatype,
    get_pytorch_c_dtype_str,
)


class CINLowerer:
    """
    This is a class to lower a CIN to LLIR
    """

    def __init__(self):
        self.defined_index_vars: List[IndexVar] = []
        # dict from IndexVar to a List of llir.Stmt of dense coordinate resolution
        # the index var is the index var that needs to be defined before the coord
        # can be resolved
        self.dep_index_var_to_dense_coord_resolution: Dict[
            IndexVar, List[llir.Stmt]
        ] = {}

        self.seen_outermost_forall = False

        self.result_value_array_sparse_index_llir = None
        self.index_var_to_rhs_tensor_level_type = None
        self.index_var_to_result_tensor_level_type = None

        self.result_tensor_var: Optional[TensorVar] = None
        self.result_tensor_access: Optional[TensorAccess] = None
        self.result_tensor_value_index_var_dict: Dict[IndexVar, llir.Expr] = {}

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
        # TODO: handle COO
        statements: List[llir.Stmt] = []
        level_types = tensor.get_level_types()
        for level, level_type in enumerate(level_types):
            if level_type == LevelType.DENSE:
                statements.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_size",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}._shape[{level}]",
                            type=llir.DataType.INT,
                        ),
                    )
                )
            elif level_type == LevelType.COMPRESSED:
                statements.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_pos",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.Var(
                            # name=f"{tensor.name}._storage._index.mode_indices[{level}][0]",
                            name=f"{tensor.name}_mode_indices[{level}][0]",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                    )
                )
                #
                statements.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_crd",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.Var(
                            # name=f"{tensor.name}._storage._index.mode_indices[{level}][1]",
                            name=f"{tensor.name}_mode_indices[{level}][1]",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                    )
                )
        return statements

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
        last_index_var = tensor_access.indices[-1]

        # If the level_type corresponding to the last index var is dense, then we can just use
        # the index var as the index into the value array
        tensor_var = tensor_access.get_tensor()
        level = tensor_access.level_of_index_var(last_index_var)
        level_type = tensor_var.get_level_types()[level]

        if len(tensor_access.indices) == 1 and level_type == LevelType.DENSE:
            return llir.Var(
                name=f"{tensor_access.tensor.name}_values[{last_index_var.name}]",
                type=llir.DataType.NO_TYPE,
            )

        return llir.Var(
            name=f"{tensor_access.tensor.name}_values"
            + f"[p{tensor_access.tensor.get_name()}{tensor_access.level_of_index_var(last_index_var)}]"
            + f".item<{dtype_to_c_datatype(tensor_var.dtype).value}>()",
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
        # if we are at the bottommost level, we can emit the compute code
        assert self.result_tensor_access, "result tensor access is None"
        if (
            self.result_tensor_access.get_index_vars()[-1]
            == self.defined_index_vars[-1]
        ):
            assert self.result_tensor_var, "result tensor var is None"
            if self.result_value_array_sparse_index_llir:
                tensor_access_llir = llir.Var(
                    name=f"{self.result_tensor_var.get_name()}_values"
                    + f"[{self.result_value_array_sparse_index_llir.name}]",
                    type=llir.DataType.NO_TYPE,
                )
            else:
                tensor_access_llir = llir.Var(
                    name=f"{self.result_tensor_var.get_name()}_values"
                    + f"[{self.defined_index_vars[-1].name}]",
                    type=llir.DataType.NO_TYPE,
                )
            llir_stmts.append(
                llir.Assign(
                    var=tensor_access_llir,
                    value=self.lower_IndexExpr(stmt.rhs),
                )
            )
            # If the last level of the result tensor var is sparse, then we need to set
            # the coordinates
            if (
                self.result_tensor_access.level_type_of_index_var(
                    self.defined_index_vars[-1]
                )
                == LevelType.COMPRESSED
            ):
                llir_stmts.append(llir.Comment("Set coordinates"))
                llir_stmts.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{self.result_tensor_var.get_name()}{self.result_tensor_var.levels - 1}_crd"
                            + f"[p{self.result_tensor_var.get_name()}{self.result_tensor_var.levels - 1}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        value=llir.Var(
                            name=self.defined_index_vars[-1].name,
                            type=llir.DataType.NO_TYPE,
                        ),
                    )
                )

            # if has sparse index for result value array, need to increment
            if self.result_value_array_sparse_index_llir is not None:
                llir_stmts.append(
                    llir.Increment(
                        var=self.result_value_array_sparse_index_llir,
                    )
                )

        return llir_stmts

    def lower_IndexStmt(self, stmt: IndexStmt) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower an IndexStmt to LLIR
        """

        if isinstance(stmt, TensorAssign):
            return self.lower_TensorAssign(stmt)

        # loop_order_allow_short_circuit = all_free_var_loops_before_reduction_loops(stmt)

        # Create tensor results and rhs IR variables
        result_tensor_vars: List[TensorVar] = stmt.get_result_tensor_vars()
        # TODO: need to handle multiple result tensors
        self.result_tensor_var = result_tensor_vars[0]
        result_tensor_accesses = stmt.get_result_tensor_accesses()
        self.result_tensor_access = result_tensor_accesses[0]
        rhs_tensor_vars: List[TensorVar] = stmt.get_rhs_tensor_vars()
        rhs_tensor_accesses: List[TensorAccess] = stmt.get_rhs_tensor_accesses()
        rhs_tensor_vars_llir: List[llir.Expr] = [
            self.lower_TensorVar(tv) for tv in rhs_tensor_vars
        ]

        self.need_compute.extend(result_tensor_vars)

        tensor_value_array_init_stmts = []
        result_level_indices_init_stmts: List[llir.Stmt] = []

        for result_tensor_var in result_tensor_vars:
            self.tensor_var_to_llir[result_tensor_var] = self.lower_TensorVar(
                result_tensor_var
            )
            tensor_value_array_init_stmts.append(
                llir.VarDecl(
                    llir.Var(
                        name=f"{result_tensor_var.get_name()}_values",
                        type=llir.DataType.cvector_type(
                            dtype_to_c_datatype(result_tensor_var.dtype)
                        ),
                    )
                )
            )

            for i, level_type in enumerate(result_tensor_var.get_level_types()):
                if level_type == LevelType.COMPRESSED:
                    # e.g. cvector<int> a0_pos;
                    result_level_indices_init_stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{result_tensor_var.get_name()}{i}_pos",
                                type=llir.DataType.CVECTOR_INT,
                            )
                        )
                    )

                    # e.g. cvector<int> a0_crd;
                    result_level_indices_init_stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{result_tensor_var.get_name()}{i}_crd",
                                type=llir.DataType.CVECTOR_INT,
                            )
                        )
                    )

                    # e.g. a0_pos[0] = 0;
                    result_level_indices_init_stmts.append(
                        llir.Assign(
                            var=llir.Var(
                                name=f"{result_tensor_var.get_name()}{i}_pos[0]",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Literal(0),
                        )
                    )

                    # e.g. int pa0 = 0;
                    result_level_indices_init_stmts.append(
                        llir.VarInit(
                            llir.Var(
                                name=f"p{result_tensor_var.get_name()}{i}",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Literal(0),
                        )
                    )

                    result_level_indices_init_stmts.append(llir.BlankLine())

        if result_level_indices_init_stmts:
            result_level_indices_init_stmts = [
                llir.Comment("Init result level indices"),
                *result_level_indices_init_stmts,
                llir.BlankLine(),
            ]

        # Generate iterator bounds
        tensor_level_array_assign_stmts = []

        tensor_value_array_assign_stmts = []
        for tensor in rhs_tensor_vars:
            tensor_level_array_assign_stmts.extend(self.get_level_arrays(tensor))
            tensor_value_array_assign_stmts.append(
                self.get_value_array_statement(tensor)
            )

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
                llir.Comment("Init result tensor level sizes")
            ] + result_tensor_level_sizes

        # A mapping from IndexVar to a list of (TensorVar, level: int, LevelType) tuples
        self.index_var_to_rhs_tensor_level_type = {}
        for tensor_access in rhs_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            for level, index_var in enumerate(index_vars):
                if index_var not in self.index_var_to_rhs_tensor_level_type:
                    self.index_var_to_rhs_tensor_level_type[index_var] = []
                self.index_var_to_rhs_tensor_level_type[index_var].append(
                    [tensor_var, level, tensor_level_types[level]]
                )

        self.index_var_to_result_tensor_level_type = {}
        for tensor_access in result_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            for level, index_var in enumerate(index_vars):
                if index_var not in self.index_var_to_result_tensor_level_type:
                    self.index_var_to_result_tensor_level_type[index_var] = []
                self.index_var_to_result_tensor_level_type[index_var].append(
                    [tensor_var, level, tensor_level_types[level]]
                )

        # Initialize index into result if any level if compressed
        # Find last compressed level of the result tensor, if any
        result_last_compressed_index_var = None
        for (
            index_var,
            tensor_level_type_list,
        ) in self.index_var_to_result_tensor_level_type.items():
            # TODO: deal with multiple outputs
            tensor_var, level, level_type = tensor_level_type_list[0]
            if level_type == LevelType.COMPRESSED:
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
        if isinstance(stmt, ForAll):
            if self.seen_outermost_forall:
                return self.lower_ForAll(stmt)
            else:
                self.seen_outermost_forall = True

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

            body_stmts.extend(
                [
                    *result_tensor_level_sizes,
                    llir.BlankLine(),
                    llir.Comment("Get tensor level arrays"),
                    *tensor_level_array_assign_stmts,
                    llir.BlankLine(),
                    *result_level_indices_init_stmts,
                    # llir.Comment("Get tensor value arrays"),
                    # *tensor_value_array_assign_stmts,
                    # llir.BlankLine(),
                    llir.Comment("Initialize result value array"),
                    *tensor_value_array_init_stmts,
                    # *result_index_init_stmts,
                    llir.BlankLine(),
                    *self.lower_ForAll(stmt),
                    llir.Comment("Assemble result"),
                    llir.VarDecl(
                        var=llir.Var(
                            name=f"{self.result_tensor_var.get_name()}",
                            type=llir.DataType.TACO_TENSOR,
                        )
                    ),
                ]
            )

            # torch::Tensor a0_pos_torch = torch::from_blob(a0_pos.data(), {a0_pos.size()}, a0_pos.get_deleter(), torch::kInt);
            for i, level_type in enumerate(self.result_tensor_var.get_level_types()):
                if level_type == LevelType.COMPRESSED:
                    tensor_level_name = f"{self.result_tensor_var.get_name()}{i}"
                    # Emit {tensor_level_name}_pos_torch array
                    body_stmts.append(
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
                    # Emit {tensor_level_name}_crd_torch array
                    body_stmts.append(
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

            # Emit result value array
            body_stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self.result_tensor_var.get_name()}_values_torch",
                        type=llir.DataType.TORCH_TENSOR,
                    ),
                    value=llir.FunctionCall(
                        name="torch::from_blob",
                        args=[
                            llir.Var(
                                name=f"{self.result_tensor_var.get_name()}_values.data()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{{{self.result_tensor_var.get_name()}_values.size()}}",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=f"{self.result_tensor_var.get_name()}_values.get_deleter()",
                                type=llir.DataType.NO_TYPE,
                            ),
                            llir.Var(
                                name=get_pytorch_c_dtype_str(
                                    self.result_tensor_var.dtype
                                ),
                                type=llir.DataType.NO_TYPE,
                            ),
                        ],
                    ),
                )
            )

            # Emit result tensor index assignment
            # e.g. A._storage._index.mode_indices = {{A0_pos_torch, A0_crd_torch}, {A1_pos_torch, A1_crd_torch}};

            def get_result_mode_index_set(i, level_type: LevelType):
                assert self.result_tensor_var, "Result tensor variable not set"
                tensor_level_name = f"{self.result_tensor_var.get_name()}{i}"
                if level_type == LevelType.DENSE:
                    return "{}"
                elif level_type == LevelType.COMPRESSED:
                    return f"{{{tensor_level_name}_pos_torch, {tensor_level_name}_crd_torch}}"

            body_stmts.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{self.result_tensor_var.get_name()}._storage._index.mode_indices",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{{{', '.join([get_result_mode_index_set(i, level_type) for i, level_type in enumerate(self.result_tensor_var.get_level_types())])}}}",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

            # Emit result tensor value assignment
            # e.g. A._storage._value = A_values_torch;
            body_stmts.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{self.result_tensor_var.get_name()}._storage._value",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{self.result_tensor_var.get_name()}_values_torch",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

            # Emit return statement
            body_stmts.append(
                llir.Return(
                    value=llir.Var(
                        name=f"{self.result_tensor_var.get_name()}",
                        type=llir.DataType.NO_TYPE,
                    )
                )
            )

            return llir.Function(
                return_type=llir.DataType.TACO_TENSOR,
                name="evaluate",
                args=kernel_args,
                body=body_stmts,
            )

        return []

    def lower_ForAll(self, stmt: ForAll) -> List[llir.Stmt]:
        """
        Lower a ForAll to LLIR
        parent_index_var is the index var of the parent ForAll, if any
        """

        # Get index variable at this forall
        index_var = stmt.get_index_var()

        self.defined_index_vars.append(index_var)

        iter_lattice = IterationLattice(for_all_stmt=stmt, cin_lowerer=self)

        stmts: List[llir.Stmt] = []

        # If the result level for this index_var is dense, need to assemble the result by
        # setting the corresponding values in the result values array to 0
        if (
            self.result_tensor_access
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
                stmts.extend(
                    [
                        llir.Comment("Assemble dense result level as needed"),
                        # initialize a result stride variable = current_level_size * next_level_size * ...
                        llir.VarInit(
                            var=llir.Var(
                                name=f"{self.result_tensor_var.get_name()}_stride",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Var(
                                name=" * ".join(
                                    [
                                        f"{self.result_tensor_var.get_name()}{i}_size"
                                        for i in range(
                                            level_of_index_var,
                                            self.result_tensor_access.num_levels,
                                        )
                                    ]
                                ),
                                type=llir.DataType.INT,
                            ),
                        ),
                        # for (int i = 0; i < <result_tensor_name>_stride; i++) {
                        #   <result_tensor_name>_values[i] = 0;
                        # }
                        llir.ForLoop(
                            init=llir.VarInit(
                                var=llir.Var(
                                    name="i",
                                    type=llir.DataType.INT,
                                ),
                                value=llir.Literal(
                                    value="0",
                                ),
                            ),
                            cond=llir.BinOp(
                                op="<",
                                left=llir.Var(
                                    name="i",
                                    type=llir.DataType.INT,
                                ),
                                right=llir.Var(
                                    name=f"{self.result_tensor_var.get_name()}_stride",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            update=llir.Increment(
                                var=llir.Var(
                                    name="i",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            body=[
                                llir.Assign(
                                    var=llir.Var(
                                        name=f"{self.result_tensor_var.get_name()}_values[i]",
                                        type=llir.DataType.NO_TYPE,
                                    ),
                                    value=llir.Literal(
                                        value="0",
                                    ),
                                )
                            ],
                        ),
                        llir.BlankLine(),
                    ]
                )

        stmts.extend(
            [
                *iter_lattice.get_iterator_init_stmts(),
                llir.BlankLine(),
                *iter_lattice.get_lattice_loops(),
            ]
        )

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
            name=ivar.get_name(),
            type=llir.DataType.INT,
        )
