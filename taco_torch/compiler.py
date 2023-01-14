from typing import List, Optional, Dict, Union

import taco_torch.llir as llir
from taco_torch.cin import (
    IndexStmt,
    IndexVar,
    all_free_var_loops_before_reduction_loops,
    TensorVar,
    CINVisitorAccept,
    TensorAssign,
    TensorAccess,
    ForAll,
    IndexExpr,
    BinaryOp,
    CIN,
)
from taco_torch.format import LevelType
from taco_torch.iter_lattice import IterationLattice, LatticePoint
from taco_torch.iterator import ModeIterator, ModeAccess


class LLIRLowerer:
    """
    This is a class to lower LLIR to C++ code (string).
    """

    indent_str = "  "
    indent_level = 0

    def lower_llir(
        self,
        ir: Union[llir.Node, List[llir.Node], str, List[str]],
        indent_level: int = 0,
    ) -> str:
        if isinstance(ir, str):
            return indent_level * self.indent_str + ir

        elif isinstance(ir, list):
            return "\n".join([self.lower_llir(node, indent_level) for node in ir])

        elif isinstance(ir, llir.Comment):
            return self.lower_llir(f"// {ir.value}", indent_level)

        elif isinstance(ir, llir.BlankLine):
            return self.lower_llir("", indent_level)

        elif isinstance(ir, llir.Literal):
            return self.lower_llir(str(ir.value), indent_level)

        elif isinstance(ir, llir.VarAssign):
            if ir.var.type:
                return self.lower_llir(
                    f"{ir.var.type.value} {ir.var.name} {ir.op} {self.lower_llir(ir.value)};",
                    indent_level,
                )
            else:
                return self.lower_llir(
                    f"{ir.var.name} {ir.op} {self.lower_llir(ir.value)};", indent_level
                )
        elif isinstance(ir, llir.Cast):
            return self.lower_llir(
                f"({ir.data_type.value}) {self.lower_llir(ir.expr)}", indent_level
            )

        elif isinstance(ir, llir.BinOp):
            return self.lower_llir(
                f"{self.lower_llir(ir.left)} {ir.op} {self.lower_llir(ir.right)}",
                indent_level,
            )
        elif isinstance(ir, llir.FunctionCall):
            return self.lower_llir(
                f"{ir.name}({', '.join([self.lower_llir(arg) for arg in ir.args])})",
                indent_level,
            )
        elif isinstance(ir, llir.Array):
            return self.lower_llir(
                f"{{{', '.join([self.lower_llir(v) for v in ir.values])}}}",
                indent_level,
            )
        elif isinstance(ir, llir.WhileLoop):
            return (
                self.lower_llir(f"while ({self.lower_llir(ir.cond)}) {{", indent_level)
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )
        elif isinstance(ir, llir.IfThenElse):
            # Handle the case where cond is a list of conditions and then_body is a list of list
            # of statements
            result = ""
            if isinstance(ir.cond, list):
                assert isinstance(ir.then_body, list)
                assert len(ir.cond) == len(
                    ir.then_body
                ), f"Number of conditions and then_body's must be the same"

                total_num_conds = len(ir.cond) + (1 if ir.else_body else 0)

                for i, cond in enumerate(ir.cond):
                    if i == 0:
                        result += (
                            self.lower_llir(
                                f"if ({self.lower_llir(cond)}) {{", indent_level
                            )
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
                    elif ir.make_last_case_else and i == total_num_conds - 1:
                        result += (
                            self.lower_llir(f"}} else {{", indent_level)
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
                    else:
                        result += (
                            self.lower_llir(
                                f"}} else if ({self.lower_llir(cond)}) {{", indent_level
                            )
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
            else:
                result += (
                    self.lower_llir(f"if ({self.lower_llir(ir.cond)}) {{", indent_level)
                    + "\n"
                    + self.lower_llir(ir.then_body, indent_level + 1)
                    + "\n"
                )

            if ir.else_body:
                result += (
                    self.lower_llir(f"}} else {{", indent_level)
                    + "\n"
                    + self.lower_llir(ir.else_body, indent_level + 1)
                    + "\n"
                )

            result += self.lower_llir("}", indent_level)
            return result

        elif isinstance(ir, llir.Var):
            return ir.name

        elif isinstance(ir, llir.Increment):
            return self.lower_llir(f"{ir.var.name}++;", indent_level)

        elif isinstance(ir, llir.Function):
            # args must be llir.Var's
            assert [isinstance(arg, llir.Var) for arg in ir.args]
            return (
                self.lower_llir(
                    f"{ir.return_type.value} {ir.name}"
                    + f"({', '.join([f'{arg.type.value} {arg.name}' for arg in ir.args])}) {{",
                    indent_level,
                )
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )
        return self.lower_llir(
            f"No code gen implemented for node type: {ir.__class__.__name__}",
            indent_level,
        )


class CINLowerer:
    """
    This is a class to lower a CIN to LLIR
    """

    def __init__(self):

        self.seen_outermost_forall = False
        self.processed_index_vars: List[IndexVar] = []
        self.result_value_array_sparse_index_llir = None
        self.index_var_to_rhs_tensor_level_type = None
        self.index_var_to_result_tensor_level_type = None

        self.result_tensor_var: Optional[TensorVar] = None
        self.result_tensor_access: Optional[TensorAccess] = None
        self.result_tensor_value_index_var_dict: Dict[IndexVar, llir.Expr] = {}

        self.llir_stmt: Optional[llir.Stmt] = None
        self.defined_index_vars_ordered: List[IndexVar] = []
        self.defined_index_vars: List[IndexVar] = []
        self.need_compute: List[TensorVar] = []
        self.tensor_var_to_llir: Dict[TensorVar, llir.Expr] = {}
        self.level_iterators: Dict[ModeAccess, ModeIterator] = {}

    @staticmethod
    def get_level_arrays(tensor: TensorVar) -> List[llir.Stmt]:
        """
        Generate the bounds variable definitions given a TensorVar
        """
        # Iterate over the levels in tensor, then depending on whether it is sparse or dense, generate the bound
        # variables
        # TODO: handle COO
        statements = []
        level_types = tensor.get_level_types()
        for level, level_type in enumerate(level_types):
            if level_type == LevelType.DENSE:
                statements.append(
                    llir.VarAssign(
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
                    llir.VarAssign(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_pos",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}._storage._index.mode_indices[{level}][0]",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                    )
                )
                statements.append(
                    llir.VarAssign(
                        var=llir.Var(
                            name=f"{tensor.name}{level}_crd",
                            type=llir.DataType.TORCH_TENSOR,
                        ),
                        value=llir.Var(
                            name=f"{tensor.name}._storage._index.mode_indices[{level}][1]",
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
        return llir.VarAssign(
            var=llir.Var(name=f"{tensor.name}_values", type=llir.DataType.TORCH_TENSOR),
            value=llir.Var(
                name=f"{tensor.name}._storage._value",
                type=llir.DataType.TORCH_TENSOR,
            ),
        )

    def lower_IndexExpr(self, index_expr: IndexExpr) -> llir.Expr:
        if isinstance(index_expr, BinaryOp):
            return llir.BinOp(
                op=index_expr.op.value,
                left=self.lower_IndexExpr(index_expr.left),
                right=self.lower_IndexExpr(index_expr.right),
            )
        elif isinstance(index_expr, TensorAccess):
            last_index_var = index_expr.indices[-1]

            return llir.Var(
                name=f"{index_expr.tensor.name}_values[{last_index_var.name}_{index_expr.tensor.name}]",
            )

    def lower_CIN(self, cin: CIN) -> Union[llir.Stmt, List[llir.Stmt], llir.Expr]:
        if isinstance(cin, IndexStmt):
            return self.lower_IndexStmt(cin)
        elif isinstance(cin, IndexExpr):
            return self.lower_IndexExpr(cin)

    def lower_IndexStmt(self, stmt: IndexStmt) -> Union[llir.Stmt, List[llir.Stmt]]:
        """
        Lower an IndexStmt to LLIR
        """

        if isinstance(stmt, TensorAssign):
            llir_stmts = []
            # if we are at the bottommost level, we can emit the compute code
            if (
                self.result_tensor_access.get_index_vars()[-1]
                == self.defined_index_vars[-1]
            ):
                if self.result_value_array_sparse_index_llir:
                    tensor_access_llir = llir.Var(
                        name=f"{self.result_tensor_var.name}_values"
                        + f"[{self.defined_index_vars[-1].name}_{self.result_tensor_var.name}]"
                    )
                else:
                    tensor_access_llir = llir.Var(
                        name=f"{self.result_tensor_var.name}_values"
                        + f"[{self.defined_index_vars[-1].name}]"
                    )
                llir_stmts.append(
                    llir.VarAssign(
                        var=tensor_access_llir,
                        value=self.lower_IndexExpr(stmt.rhs),
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

        loop_order_allow_short_circuit = all_free_var_loops_before_reduction_loops(stmt)

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

        for result_tensor_var in result_tensor_vars:
            self.tensor_var_to_llir[result_tensor_var] = self.lower_TensorVar(
                result_tensor_var
            )

        print()

        # Generate iterator bounds
        tensor_level_array_assign_stmts = []
        tensor_value_array_assign_stmts = []
        for tensor in rhs_tensor_vars + result_tensor_vars:
            tensor_level_array_assign_stmts.extend(self.get_level_arrays(tensor))
            tensor_value_array_assign_stmts.append(
                self.get_value_array_statement(tensor)
            )

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
                name=f"{result_last_compressed_index_var.name}_{self.result_tensor_var.name}",
                type=llir.DataType.INT,
            )
            self.result_tensor_value_index_var_dict[
                result_last_compressed_index_var
            ] = self.result_value_array_sparse_index_llir
            result_index_init_stmts.append(
                llir.VarAssign(
                    var=llir.Var(
                        name=f"{result_last_compressed_index_var.name}_{self.result_tensor_var.name}",
                        type=llir.DataType.INT,
                    ),
                    value=llir.Literal(value=0, data_type=llir.DataType.INT),
                )
            )

        # Finally, return function that computes the result
        if isinstance(stmt, ForAll):
            if self.seen_outermost_forall:
                return self.lower_ForAll(stmt)
            else:
                self.seen_outermost_forall = True
            return llir.Function(
                return_type=llir.DataType.TACO_TENSOR,
                name=f"compute",
                args=rhs_tensor_vars_llir,
                body=[
                    llir.Comment("Get tensor level arrays"),
                    *tensor_level_array_assign_stmts,
                    llir.BlankLine(),
                    llir.Comment("Get tensor value arrays"),
                    *tensor_value_array_assign_stmts,
                    llir.BlankLine(),
                    llir.Comment("Initialize result value array index"),
                    *result_index_init_stmts,
                    llir.BlankLine(),
                    *self.lower_ForAll(stmt),
                ],
            )

    def lower_ForAll(self, stmt: ForAll) -> List[llir.Stmt]:

        """
        Lower a ForAll to LLIR
        parent_index_var is the index var of the parent ForAll, if any
        """

        # Get index variable at this forall
        index_var = stmt.get_index_var()

        self.defined_index_vars.append(index_var)

        iter_lattice = IterationLattice(for_all_stmt=stmt, cin_lowerer=self)

        return [
            *iter_lattice.get_iterator_init_stmts(),
            llir.BlankLine(),
            *iter_lattice.get_lattice_loops(),
        ]

    @staticmethod
    def lower_TensorVar(tensor_var: TensorVar) -> llir.Expr:
        """
        Lower a TensorVar to LLIR
        """
        return llir.Var(
            name=tensor_var.name,
            # type=dtype_to_datatype(tvar.dtype),
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

    def lower_IndexVar(self, ivar: IndexVar) -> llir.Expr:
        """
        Lower an IndexVar to LLIR
        """
        raise NotImplementedError
