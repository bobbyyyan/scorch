import copy

from scorch.compiler.cin import (
    CIN,
    CINIndexVariablesGetter,
    LoopOrderGetter,
    Workspace,
    ForAll,
    TensorAssign,
    WorkspaceAccess,
    Where,
    TensorAccess,
)
from scorch.format import LevelType


class Scheduler:
    """
    Auto-schedules CIN statements.
    """

    def __init__(self):
        pass

    @staticmethod
    def auto_schedule(cin: CIN) -> CIN:
        """
        Auto-schedules a CIN statement.
        Returns a new CIN statement.
        """

        # collect all the reduction variables
        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        result_tensor_accesses = cin.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]

        reduction_vars = cin_ivar_getter.get_reduction_vars()
        free_vars = cin_ivar_getter.get_free_vars()

        loop_order_getter = LoopOrderGetter(cin)
        index_vars_ordered = loop_order_getter.index_vars_ordered

        if len(reduction_vars) == 0:
            return cin
        last_reduction_var = reduction_vars[-1]
        last_reduction_var_index = index_vars_ordered.index(last_reduction_var)

        # List of variables that come after the last reduction variable
        # in the loop order
        vars_after_last_reduction = index_vars_ordered[last_reduction_var_index + 1 :]
        # List of free variables that come after the last reduction variable
        # in the loop order
        free_vars_after_last_reduction = [
            var for var in vars_after_last_reduction if var in free_vars
        ]

        free_vars_after_last_reduction_level_types = [
            result_tensor_access.level_type_of_index_var(var)
            for var in free_vars_after_last_reduction
        ]

        if all(
            [
                level_type == LevelType.DENSE
                for level_type in free_vars_after_last_reduction_level_types
            ]
        ):
            return cin

        dim_workspace = len(free_vars_after_last_reduction)

        new_cin = copy.deepcopy(cin)

        workspace = Workspace(name="wksp", dim=dim_workspace)

        workspace_access = WorkspaceAccess(
            wksp=workspace,
            indices=free_vars_after_last_reduction,
        )

        parent_forall: ForAll = new_cin
        while parent_forall.stmt.index_var != last_reduction_var:
            parent_forall = parent_forall.stmt

        """
        For example, if parent_forall's stmt is:
        
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    B[i, k] * C[k, j],
                ),
            ),
        )
            
        Then it needs to be transformed into:
        
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        )
         
        """

        reduction_forall = parent_forall.stmt

        # Create the producer forall
        producer_forall = copy.deepcopy(reduction_forall)

        producer_forall_tensor_access_parent = producer_forall
        # Iterate until the TensorAssign statement
        while not isinstance(producer_forall_tensor_access_parent.stmt, TensorAssign):
            producer_forall_tensor_access_parent = (
                producer_forall_tensor_access_parent.stmt
            )

        # Replace the TensorAssign's lhs with the workspace
        producer_forall_tensor_access_parent.stmt.lhs = workspace_access

        # Create the consumer forall
        consumer_forall = copy.deepcopy(reduction_forall)

        consumer_forall_tensor_access_parent = consumer_forall
        # Iterate until the TensorAssign statement
        while not isinstance(consumer_forall_tensor_access_parent.stmt, TensorAssign):
            consumer_forall_tensor_access_parent = (
                consumer_forall_tensor_access_parent.stmt
            )

        # Replace the TensorAssign's rhs with the workspace
        consumer_forall_tensor_access_parent.stmt.rhs = workspace_access

        # Create the Where statement
        where_stmt = Where(
            producer=producer_forall,
            consumer=consumer_forall,
        )

        # Replace the reduction forall with the Where statement
        parent_forall.stmt = where_stmt

        return new_cin
