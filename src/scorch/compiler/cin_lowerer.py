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
from .llir import Assign, AssignOp
from ..format import LevelType
from ..utils import dtype_to_c_datatype, get_pytorch_c_dtype_str


class CINLowerer:
    """
    This is a class to lower a CIN to LLIR
    """

    def __init__(self, filter_zeros=False):
        self.filter_zeros: bool = filter_zeros
        self.defined_index_vars: List[IndexVar] = []
        # dict from IndexVar to a List of llir.Stmt of dense coordinate resolution
        # the index var is the index var that needs to be defined before the coord
        # can be resolved
        self.dep_index_var_to_dense_coord_resolution: Dict[
            IndexVar, List[llir.Stmt]
        ] = {}

        self.seen_outermost_forall = False
        self.outermost_stmt: Optional[IndexStmt] = None

        self.result_value_array_sparse_index_llir = None
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
        # TODO: handle COO
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
            var=llir.Var(name=f"{tensor.name}_val", type=ptr_type),
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
        last_index_var = tensor_access.indices[-1]

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

        # if we are at the bottommost _level, we can emit compute code
        assert self.result_tensor_access, "result tensor access is None"
        is_workspace = self.result_tensor_access.is_workspace()
        index_vars = self.result_tensor_access.get_index_vars()
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
                    level = self.result_tensor_access.level_of_index_var(index_vars[-1])
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

                    if wksp_access.is_dense():
                        # <workspace name>[<C++ array of indices>] += <rhs_llir>;
                        assert (
                            len(wksp_index_vars) == 1
                        ), "dense workspace has more than 1 index var"
                        wksp_index_var = wksp_index_vars[0]
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
                                            for ivar in wksp_index_vars
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
        for wksp in workspaces:
            # coo_workspace<tensor's ctype> <tensor's name> = coo_workspace<tensor's ctype>(<tensor's dim>);
            wksp_ctype = dtype_to_c_datatype(wksp.dtype)

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

            # DONE: If the workspace is 1-dimensional, use optimized 1D workspace implementation
            # class name: coo_workspace_1d<tensor's ctype, tensor's dim>

            if wksp.dim == 1:
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
                        ],
                    ),
                )
            )

        return [
            llir.Comment("Lower Where statement"),
            *workspace_init_stmts,
            *self.lower_ProducerIndexStmt(stmt.producer),
            *self.lower_ConsumerIndexStmt(stmt.consumer),
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

    def lower_ConsumerIndexStmt(self, stmt: IndexStmt) -> List[llir.Stmt]:
        """
        Lower a ConsumerIndexStmt to LLIR
        """
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

            index_var = result_tensor_access.get_index_vars()[0]
            level = result_tensor_access.level_of_index_var(index_var)
            level_type = result_tensor_access.level_type_of_index_var(index_var)

            if level_type == LevelType.DENSE:
                # <result tensor name>_values[<result level iterator>] = <wksp's name>;
                stmts.append(
                    llir.Assign(
                        var=llir.Var(
                            name=f"{result_tensor_name}_values[{index_var.name}]",
                            type=llir.DataType.NO_TYPE,
                        ),
                        value=llir.Var(
                            name=f"{wksp.get_name()}",
                            type=llir.DataType.NO_TYPE,
                        ),
                    )
                )
            else:
                raise NotImplementedError(
                    "TODO: need to handle assembly of workspace with sparse level"
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

        # TODO: handle dense accumulator workspace
        if wksp_access.is_dense():
            assert (
                len(wksp_index_vars) == 1
            ), "dense workspace has more than 1 index var"
            wksp_index_var = wksp_index_vars[0]
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
            # if previous _level is dense: A1_pos.push_back(A1_crd.size())
            # TODO: if previous _level is sparse: A1_pos[A0_crd.size()] = A1_crd.size()
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

        self.need_compute.extend(result_tensor_vars)

        if recurse or stmt != self.outermost_stmt:
            if isinstance(stmt, ForAll):
                return self.lower_ForAll(stmt)
            if isinstance(stmt, Where):
                return self.lower_Where(stmt)

        tensor_value_array_init_stmts = []
        result_level_indices_init_stmts: List[llir.Stmt] = []

        for result_tensor_var in non_workspace_result_tensor_vars:
            self.tensor_var_to_llir[result_tensor_var] = self.lower_TensorVar(
                result_tensor_var
            )
            result_name = result_tensor_var.get_name()
            # If the result tensor var is fully dense, then we use malloc
            # instead of cvector
            if result_tensor_var.is_dense():
                # Initialize a variable <result_name>_capacity to be
                # the product of all the dimensions of the result tensor
                # e.g. int A0_capacity = A0_size * A1_size * A2_size;
                result_capacity_var = llir.Var(
                    name=f"{result_name}_capacity",
                    type=llir.DataType.INT,
                )
                # Base capacity is A0_size
                res_capacity_expr = llir.Var(
                    name=f"{result_name}0_size",
                    type=llir.DataType.INT,
                )
                for i in range(1, result_tensor_var.levels):
                    res_capacity_expr = llir.BinOp(
                        left=res_capacity_expr,
                        op="*",
                        right=llir.Var(
                            name=f"{result_name}{i}_size",
                            type=llir.DataType.INT,
                        ),
                    )
                init_capacity_stmt = llir.VarInit(
                    var=result_capacity_var,
                    value=res_capacity_expr,
                )
                tensor_value_array_init_stmts.append(init_capacity_stmt)

                # use malloc
                # <result c datatype>* restrict <result_name>_values
                #  = (<result c datatype>*)malloc(sizeof(<result c datatype>) * A_capacity);
                result_val_var = llir.Var(
                    name=f"{result_name}_values",
                    type=llir.DataType.ptr_type(result_tensor_var.dtype),
                )
                # malloc_stmt is the RHS
                c_datatype = dtype_to_c_datatype(result_tensor_var.dtype)
                sizeof_expr = llir.Sizeof(c_datatype)
                res_capacity_var = llir.Var(
                    name=f"{result_name}_capacity",
                    type=llir.DataType.INT,
                )

                malloc = llir.FunctionCall(
                    name="malloc",
                    args=[llir.BinOp(left=sizeof_expr, op="*", right=res_capacity_var)],
                )
                # Cast malloc to the correct type
                malloc = llir.Cast(
                    expr=malloc,
                    data_type=llir.DataType.ptr_type(c_datatype),
                )

                tensor_value_array_init_stmts.append(
                    llir.VarInit(
                        var=result_val_var,
                        value=malloc,
                    )
                )

            else:
                tensor_value_array_init_stmts.append(
                    llir.VarDecl(
                        llir.Var(
                            name=f"{result_name}_values",
                            type=llir.DataType.cvector_type(
                                dtype_to_c_datatype(result_tensor_var.dtype)
                            ),
                        )
                    )
                )

            result_level_types = result_tensor_var.get_level_types()
            for i, level_type in enumerate(result_level_types):
                if level_type == LevelType.COMPRESSED:
                    # e.g. cvector<int> a0_pos;
                    result_level_indices_init_stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{result_name}{i}_pos",
                                type=llir.DataType.CVECTOR_INT,
                            )
                        )
                    )

                    # e.g. cvector<int> a0_crd;
                    result_level_indices_init_stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{result_name}{i}_crd",
                                type=llir.DataType.CVECTOR_INT,
                            )
                        )
                    )

                    # e.g. a0_pos[0] = 0;
                    result_level_indices_init_stmts.append(
                        llir.Assign(
                            var=llir.Var(
                                name=f"{result_name}{i}_pos[0]",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Literal(0),
                        )
                    )

                    # e.g. int pa0 = 0;
                    result_level_indices_init_stmts.append(
                        llir.VarInit(
                            llir.Var(
                                name=f"p{result_name}{i}",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Literal(0),
                        )
                    )

                    # e.g. int a0_pos_index = 0;
                    result_level_indices_init_stmts.append(
                        llir.VarInit(
                            llir.Var(
                                name=f"{result_name}{i}_pos_index",
                                type=llir.DataType.INT,
                            ),
                            value=llir.Literal(0),
                        )
                    )

                    result_level_indices_init_stmts.append(llir.BlankLine())

                    # TODO: If parent level of this level is dense, then we also add
                    # a for loop to initialize the pos array to all zeros from the 2nd
                    # element to the last element
                    # e.g. for (int p<tensor_name><i> = 1; p<tensor_name><i> < <tensor_name><i-1>_size + 1; p<tensor_name><i>++) {
                    # e.g.     <tensor_name><i>_pos[p<tensor_name><i>] = 0;
                    # e.g. }
                    if i > 0 and result_level_types[i - 1] == LevelType.DENSE:
                        loop_var_name = f"p{result_name}{i}"
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
                                    name=f"{result_name}{i - 1}_size",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            update=llir.Increment(
                                var=loop_var,
                            ),
                            body=[
                                llir.Assign(
                                    var=llir.Var(
                                        name=f"{result_name}{i}_pos[{loop_var_name}]",
                                        type=llir.DataType.INT,
                                    ),
                                    value=llir.Literal(0),
                                )
                            ],
                        )

                        result_level_indices_init_stmts.append(loop)

                elif level_type == LevelType.COORDINATE:
                    # e.g. cvector<int> a0_crd;
                    result_level_indices_init_stmts.append(
                        llir.VarDecl(
                            llir.Var(
                                name=f"{result_name}{i}_crd",
                                type=llir.DataType.CVECTOR_INT,
                            )
                        )
                    )

                    # e.g. int pa0 = 0;
                    result_level_indices_init_stmts.append(
                        llir.VarInit(
                            llir.Var(
                                name=f"p{result_name}{i}",
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

        for tensor in rhs_tensor_vars:
            tensor_level_array_assign_stmts.append(
                llir.Comment(f"Get {tensor.get_name()}'s level & value arrays")
            )
            tensor_level_array_assign_stmts.extend(self.get_level_arrays(tensor))
            tensor_level_array_assign_stmts.append(self.get_val_ptr_stmt(tensor))

        # Generate per-_level size variables for each dense _level in result tensor
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
            for level, index_var in enumerate(index_vars):
                if index_var not in self.index_var_to_rhs_tensor_level_type:
                    self.index_var_to_rhs_tensor_level_type[index_var] = []
                self.index_var_to_rhs_tensor_level_type[index_var].append(
                    [tensor_var, level, tensor_level_types[level]]
                )

        self.index_var_to_result_tensor_level_type = {}
        for tensor_access in result_tensor_accesses:
            index_vars = tensor_access.get_index_vars()
            if not index_vars:
                continue
            tensor_var = tensor_access.get_tensor()
            tensor_level_types = tensor_var.get_level_types()
            for level, index_var in enumerate(index_vars):
                if index_var not in self.index_var_to_result_tensor_level_type:
                    self.index_var_to_result_tensor_level_type[index_var] = []
                self.index_var_to_result_tensor_level_type[index_var].append(
                    [tensor_var, level, tensor_level_types[level]]
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

            body_stmts.extend(
                [
                    *result_tensor_level_sizes,
                    llir.BlankLine(),
                    *tensor_level_array_assign_stmts,
                    llir.BlankLine(),
                    *result_level_indices_init_stmts,
                    # llir.BlankLine(),
                    llir.Comment("Initialize result value array"),
                    *tensor_value_array_init_stmts,
                    # *result_index_init_stmts,
                    llir.BlankLine(),
                    *recurse_stmts,
                ]
            )

            if self.final_result_tensor_var:
                body_stmts.extend(
                    [
                        llir.Comment("Assemble final result"),
                        llir.VarDecl(
                            var=llir.Var(
                                name=f"{self.final_result_tensor_var.get_name()}",
                                type=llir.DataType.TACO_TENSOR,
                            )
                        ),
                    ]
                )

            # torch::Tensor a0_pos_torch = torch::from_blob(a0_pos.data(), {a0_pos.size()}, a0_pos.get_deleter(), torch::kInt);
            assert self.final_result_tensor_var is not None, "No final result tensor"
            for i, level_type in enumerate(
                self.final_result_tensor_var.get_level_types()
            ):
                tensor_level_name = f"{self.final_result_tensor_var.get_name()}{i}"

                if level_type in [LevelType.COMPRESSED, LevelType.COORDINATE]:
                    if level_type == LevelType.COMPRESSED:
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
            final_result_name = self.final_result_tensor_var.get_name()
            res_values_torch_var = llir.Var(
                name=f"{final_result_name}_values_torch",
                type=llir.DataType.TORCH_TENSOR,
            )
            # If final result tensor var is dense, then first create a lambda deleter
            # function that will delete the result value array
            if self.final_result_tensor_var.is_dense():
                # auto deleter = [](void* ptr) { free(ptr); };
                body_stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=f"{final_result_name}_values_deleter",
                            type=llir.DataType.AUTO,
                        ),
                        value=llir.Var(
                            name=f"[](void* ptr) {{ free(ptr); }}",
                            type=llir.DataType.AUTO,
                        ),
                    )
                )
                body_stmts.append(
                    llir.VarInit(
                        var=res_values_torch_var,
                        value=llir.FunctionCall(
                            name="torch::from_blob",
                            args=[
                                llir.Var(
                                    name=f"{final_result_name}_values",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{{{final_result_name}_capacity}}",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{final_result_name}_values_deleter",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=get_pytorch_c_dtype_str(
                                        self.final_result_tensor_var.dtype
                                    ),
                                    type=llir.DataType.NO_TYPE,
                                ),
                            ],
                        ),
                    )
                )

            else:
                body_stmts.append(
                    llir.VarInit(
                        var=res_values_torch_var,
                        value=llir.FunctionCall(
                            name="torch::from_blob",
                            args=[
                                llir.Var(
                                    name=f"{final_result_name}_values.data()",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{{{final_result_name}_values.size()}}",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=f"{final_result_name}_values.get_deleter()",
                                    type=llir.DataType.NO_TYPE,
                                ),
                                llir.Var(
                                    name=get_pytorch_c_dtype_str(
                                        self.final_result_tensor_var.dtype
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
                assert self.final_result_tensor_var, "Result tensor variable not set"
                tensor_level_name = f"{self.final_result_tensor_var.get_name()}{i}"
                if level_type == LevelType.DENSE:
                    return "{}"
                elif level_type == LevelType.COMPRESSED:
                    return f"{{{tensor_level_name}_pos_torch, {tensor_level_name}_crd_torch}}"
                elif level_type == LevelType.COORDINATE:
                    return f"{{{tensor_level_name}_crd_torch}}"

            body_stmts.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{self.final_result_tensor_var.get_name()}._storage._index.mode_indices",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{{{', '.join([get_result_mode_index_set(i, level_type) for i, level_type in enumerate(self.final_result_tensor_var.get_level_types())])}}}",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

            # Emit result tensor value assignment
            # e.g. A._storage._value = A_values_torch;
            body_stmts.append(
                llir.Assign(
                    var=llir.Var(
                        name=f"{self.final_result_tensor_var.get_name()}._storage._value",
                        type=llir.DataType.NO_TYPE,
                    ),
                    value=llir.Var(
                        name=f"{self.final_result_tensor_var.get_name()}_values_torch",
                        type=llir.DataType.NO_TYPE,
                    ),
                )
            )

            # Emit return statement
            body_stmts.append(
                llir.Return(
                    value=llir.Var(
                        name=f"{self.final_result_tensor_var.get_name()}",
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

        # if self.result_tensor_access and not self.result_tensor_access.has_index_var(
        #     index_var
        # ):
        #     stmts.append(llir.Comment(f"{index_var} not in result tensor access"))

        # If the result _level for this index_var is dense, need to assemble the result by
        # setting the corresponding values in the result values array to 0
        if (
            self.result_tensor_access
            and self.result_tensor_access.has_index_var(index_var)
            and self.result_tensor_access.level_type_of_index_var(index_var)
            == LevelType.DENSE
        ):
            # If the parent _level is not dense or has no parent _level,
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
            name=ivar.name,
            type=llir.DataType.INT,
        )
