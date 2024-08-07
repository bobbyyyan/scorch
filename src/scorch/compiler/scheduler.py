import copy
import pdb
from typing import List, Set

from scorch.compiler.cin import (
    CIN,
    CINIndexVariablesGetter,
    Workspace,
    ForAll,
    TensorAssign,
    WorkspaceAccess,
    Where,
    TensorAccess,
    IndexVar,
    TileSizeVar,
    CINVisitorAccept,
)
from scorch.format import LevelType


class Scheduler:
    """
    Auto-schedules CIN statements.
    """

    def __init__(self):
        pass

    @staticmethod
    def add_tile(cin: CIN, index_var: IndexVar, tile_size: int) -> CIN:
        """
        Tile the index_var of a CIN statement.
        Specifically,
            1) it stripmines the index_var (i.e. splits it into inner and outer loops)
            2) reorders the outer loop past all other inner loops
        Returns a new CIN statement.

        For example, for SpMM C[i, k] += A[i, j] * B[j, k],

        i = IndexVar("i")
        j = IndexVar("j")
        k = IndexVar("k")

        cin_stmt = ForAll(
            i,
            ForAll(
                j,
                ForAll(
                    k,
                    TensorAssign(
                        C[i, k],
                        A[i, j] * B[j, k],
                        op=Operation.ADD
                    )
                )
            )
        )

        add_tile(cin_stmt, k, 32) will return a new cin that is equivalent to being constructed as follows:

        1) The first step is to insert any necessary dense workspaces:

        cin_stmt = ForAll(
            i,
            Where(
                producer=ForAll(
                    j,
                    ForAll(
                        k,
                        TensorAssign(
                            accum_c[k],
                            A[i, j] * B[j, k],
                            op=Operation.ADD
                        )
                    )
                ),
                consumer=ForAll(
                    k,
                    TensorAssign(
                        C[i, k],
                        accum_c[k],
                    )
                )
            )
        )


        2) The second step is to stripmine the index_var:

        i = IndexVar("i")
        j = IndexVar("j")
        k_out = IndexVar("k_out")
        k_in = IndexVar("k_in")
        k = IndexVar("k", k_out + k_in)

        k_tile_size = 32
        k_tile_var = TileSizeVar(
            outer_index_var=k_out,
            inner_index_var=k_in,
            size=k_tile_size
        )

        accum_c = Workspace(name="accum_c", dim=1, dense=True)

        cin_stmt = ForAll(
            i,
            ForAll(
                k_out,
                Where(
                    producer=ForAll(
                        j,
                        ForAll(
                            k_in,
                            TensorAssign(
                                accum_c[k_in],
                                A[i, j] * B[j, k],
                                op=Operation.ADD,
                            ),
                        ),
                    ),
                    consumer=ForAll(
                        k_in,
                        TensorAssign(
                            C[i, k],
                            accum_c[k_in],
                        )
                    )
                )
            )
        )


        """

        if not cin.inserted_workspace:
            cin = Scheduler.insert_workspace(cin, allow_dense=True)

        # 2) Stripmine the index_var
        # Create inner and outer index variables, e.g.
        #   k_out = IndexVar("k_out")
        #   k_in = IndexVar("k_in")
        inner_index_var = IndexVar(f"{index_var.name}_in")
        outer_index_var = IndexVar(f"{index_var.name}_out")

        # Create a new index variable that is the sum of the inner and outer index variables
        # e.g. k = IndexVar("k", k_out + k_in)
        index_var.expr = outer_index_var + inner_index_var

        # Create a new tile size variable
        #
        # e.g.
        # k_tile_size = 32
        # k_tile_var = TileSizeVar(
        #     outer_index_var=k_out,
        #     inner_index_var=k_in,
        #     size=k_tile_size
        # )
        TileSizeVar(
            outer_index_var=outer_index_var,
            inner_index_var=inner_index_var,
            size=tile_size,
        )

        # Now, we want to go from, for example:
        #
        # new_cin = ForAll(
        #     i,
        #     Where(
        #         producer=ForAll(
        #             j,
        #             ForAll(
        #                 k,
        #                 TensorAssign(
        #                     accum_c[k],
        #                     A[i, j] * B[j, k],
        #                     op=Operation.ADD
        #                 )
        #             )
        #         ),
        #         consumer=ForAll(
        #             k,
        #             TensorAssign(
        #                 C[i, k],
        #                 accum_c[k],
        #             )
        #         )
        #     )
        # )
        #
        # to a new_cin:
        #
        # new_cin = ForAll(
        #     i,
        #     ForAll(
        #         k_out,
        #         Where(
        #             producer=ForAll(
        #                 j,
        #                 ForAll(
        #                     k_in,
        #                     TensorAssign(
        #                         accum_c[k_in],
        #                         A[i, j] * B[j, k],
        #                         op=Operation.ADD,
        #                     ),
        #                 ),
        #             ),
        #             consumer=ForAll(
        #                 k_in,
        #                 TensorAssign(
        #                     C[i, k],
        #                     accum_c[k_in],
        #                 )
        #             )
        #         )
        #     )
        # )
        #
        # This involves several steps:
        # - Replace the index_var in the Where statement indexing into the workspace with the inner index var
        # - Add a new ForAll loop outside/before the Where statement

        # Find the Where statement
        assert isinstance(cin, ForAll), "Expected input CIN to be a ForAll statement."
        parent_forall = cin
        while hasattr(parent_forall, "stmt") and not isinstance(
            parent_forall.stmt, Where
        ):
            assert isinstance(parent_forall, ForAll)
            parent_forall = parent_forall.stmt

        assert isinstance(parent_forall, ForAll)
        assert isinstance(parent_forall.stmt, Where)
        where_stmt = parent_forall.stmt

        class ReplaceIndexVarVisitor(CINVisitorAccept):
            """
            This visitor replaces the index_var in the Where statement
                - indexing into the workspace with the inner index var
                - in the ForAll statement with the inner index var

            """

            def __init__(self, old_index_var: IndexVar, new_index_var: IndexVar):
                self.old_index_var = old_index_var
                self.new_index_var = new_index_var

            def visit_ForAll(self, forall: ForAll):
                if forall.index_var == self.old_index_var:
                    forall.index_var = self.new_index_var
                self.visit(forall.stmt)

            def visit_WorkspaceAccess(self, workspace_access: WorkspaceAccess):
                if workspace_access.indices and workspace_access.indices[-1] == self.old_index_var:
                    workspace_access.indices[-1] = self.new_index_var
                    workspace_access.update_indices(workspace_access.indices)

        replace_index_var_visitor = ReplaceIndexVarVisitor(
            old_index_var=index_var,
            new_index_var=inner_index_var,
        )
        replace_index_var_visitor.visit(where_stmt)

        # Wrap the new_cin in a new ForAll loop, with the outer index var
        cin = ForAll(
            index_var=outer_index_var,
            stmt=cin,
        )

        return cin

    @staticmethod
    def insert_workspace(cin: CIN, allow_dense=False) -> CIN:
        """
        Args:
            cin: CIN statement to insert a workspace into
            allow_dense: If True, then allow dense workspaces to be inserted.

        Returns:
            A new CIN statement with a workspace inserted.

        Insert a workspace into a CIN statement, if necessary.
        Only works on the last reduction variable in the loop order.

        This function should be idempotent.
        """
        # Collect all the reduction variables
        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        assert isinstance(cin, ForAll), "Expected input CIN to be a ForAll statement."

        result_tensor_accesses = cin.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]

        reduction_vars = cin_ivar_getter.get_reduction_vars()
        free_vars = cin_ivar_getter.get_free_vars()

        # loop_order_getter = LoopOrderGetter(cin)
        # index_vars_ordered = loop_order_getter.index_vars_ordered
        index_vars_ordered = cin.loop_order

        if len(reduction_vars) == 0:
            return cin

        reduction_vars_todo = [
            var for var in reduction_vars if var in index_vars_ordered
        ]

        if len(reduction_vars_todo) == 0:
            return cin

        next_reduction_var = reduction_vars_todo[-1]

        last_reduction_var_index = index_vars_ordered.index(next_reduction_var)

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

        are_all_dense_levels = False
        if all(
            [
                level_type == LevelType.DENSE
                for level_type in free_vars_after_last_reduction_level_types
            ]
        ):
            are_all_dense_levels = True

        if not allow_dense and are_all_dense_levels:
            return cin

        dim_workspace = len(free_vars_after_last_reduction)

        new_cin = copy.deepcopy(cin)

        workspace = Workspace(
            name="wksp", dim=dim_workspace, dense=are_all_dense_levels
        )

        workspace_access = WorkspaceAccess(
            wksp=workspace,
            indices=free_vars_after_last_reduction,
        )

        # Note: parent_forall not necessarily ForAll statement at the end
        parent_forall = new_cin
        while (
            isinstance(parent_forall.stmt, ForAll)
            and parent_forall.stmt.index_var != next_reduction_var
            and parent_forall.index_var != next_reduction_var
        ):
            parent_forall = parent_forall.stmt

        assert isinstance(
            parent_forall, ForAll
        ), "Expected parent_forall to be a ForAll statement."

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
        reduction_forall = parent_forall if parent_forall.index_var == next_reduction_var else parent_forall.stmt

        # If we have already inserted a workspace, then we should not insert another one.
        if isinstance(reduction_forall, Where):
            return cin

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

        if not are_all_dense_levels:
            assert isinstance(producer_forall, ForAll)
            new_cin.no_tile_list.append(producer_forall.index_var)

        # Replace the reduction forall with the Where statement
        if parent_forall.stmt.index_var == next_reduction_var:
            parent_forall.stmt = where_stmt
        else:
            new_cin = where_stmt

        new_cin.inserted_workspace = True

        return new_cin

    @staticmethod
    def auto_schedule(cin: CIN) -> CIN:
        # pdb.set_trace()
        cin = Scheduler.insert_workspace(cin, allow_dense=True)
        all_index_vars = cin.index_vars
        tensor_accesses = cin.tensor_accesses
        # pdb.set_trace()

        # print("Auto-scheduling CIN statement" f"\n{cin}")

        # print(f"Index vars: {all_index_vars}")
        # print(f"Tensor accesses: {tensor_accesses}")

        index_vars_to_tile: List[IndexVar] = []
        index_vars_sparse: Set[IndexVar] = set()

        # First, populate the list of index variables to tile by iterating
        # through each of the tensor access; if the tensor access does not
        # use all the index variables, then we add the index variables in
        # that tensor access corresponding to dense levels to the to tile
        # list.
        for tensor_access in tensor_accesses:
            tensor_access_index_vars = tensor_access.index_vars
            for index_var in tensor_access_index_vars:
                if tensor_access.level_type_of_index_var(index_var) != LevelType.DENSE:
                    index_vars_sparse.add(index_var)

            if set(tensor_access_index_vars) != set(all_index_vars):
                for index_var in tensor_access.index_vars:
                    if (
                        index_var not in index_vars_sparse
                        and index_var not in index_vars_to_tile
                    ):
                        index_vars_to_tile.append(index_var)

        first_loop_index_var = cin.loop_order[0]

        # We should remove the first loop index var from the list of index vars to tile
        # because tiling that doesn't help.
        if first_loop_index_var in index_vars_to_tile:
            index_vars_to_tile.remove(first_loop_index_var)

        # Remove the index variables that are sparse
        index_vars_to_tile = [
            index_var
            for index_var in index_vars_to_tile
            if index_var not in index_vars_sparse
        ]

        # Remove the index variables that are already in the no_tile_list
        # TODO: check that the condition for adding to no_tile_list is correct
        index_vars_to_tile = [
            index_var
            for index_var in index_vars_to_tile
            if index_var not in cin.no_tile_list
        ]

        # print(f"Index vars to tile: {index_vars_to_tile}")

        for index_var in index_vars_to_tile:
            cin = Scheduler.add_tile(cin, index_var, 32)

        return cin
