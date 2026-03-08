import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from scorch.compiler.cin import (
    CIN,
    CINIndexVariablesGetter,
    CINVisitorAccept,
    ForAll,
    IndexVar,
    TensorAccess,
    TensorAssign,
    TileSizeVar,
    Where,
    Workspace,
    WorkspaceAccess,
)
from scorch.format import LevelType


@dataclass(frozen=True)
class _CostModelConstants:
    alpha: float = 2.975
    beta: float = 0.1005
    gamma: float = 43.55
    c_insert: float = 85.34
    c_sort: float = 1.741
    c_trans: float = 40.61
    rho: float = 0.0014
    default_dim_size: int = 1024


class Scheduler:
    """
    Auto-schedules CIN statements.
    """

    _DEFAULT_COSTS = _CostModelConstants()

    def __init__(self):
        pass

    @staticmethod
    def _is_sparse_level(level_type: LevelType) -> bool:
        return level_type in (LevelType.COMPRESSED, LevelType.COORDINATE)

    @staticmethod
    def _unique_index_vars(index_vars: List[IndexVar]) -> List[IndexVar]:
        seen: Set[str] = set()
        unique: List[IndexVar] = []
        for index_var in index_vars:
            if index_var.name in seen:
                continue
            seen.add(index_var.name)
            unique.append(index_var)
        return unique

    @staticmethod
    def _has_dense_output(cin: CIN) -> bool:
        body = cin
        while isinstance(body, ForAll):
            body = body.stmt
        if isinstance(body, TensorAssign):
            return body.lhs.get_tensor().is_dense()
        return False

    @staticmethod
    def _extract_loop_chain(cin: CIN) -> Tuple[List[IndexVar], CIN]:
        loop_order: List[IndexVar] = []
        body: CIN = cin
        while isinstance(body, ForAll):
            loop_order.append(body.index_var)
            body = body.stmt
        return loop_order, body

    @staticmethod
    def _rebuild_loop_nest(cin: CIN, loop_order: List[IndexVar]) -> CIN:
        if not isinstance(cin, ForAll):
            return cin

        _, body = Scheduler._extract_loop_chain(cin)
        rebuilt: CIN = body
        for index_var in reversed(loop_order):
            rebuilt = ForAll(index_var=index_var, stmt=rebuilt)

        rebuilt.inserted_workspace = cin.inserted_workspace
        rebuilt.no_tile_list = list(cin.no_tile_list)
        return rebuilt

    @staticmethod
    def get_index_variables(cin: CIN) -> List[IndexVar]:
        loop_order, _ = Scheduler._extract_loop_chain(cin)
        if loop_order:
            return Scheduler._unique_index_vars(loop_order)
        return sorted(
            Scheduler._unique_index_vars(cin.index_vars),
            key=lambda index_var: index_var.name,
        )

    @staticmethod
    def _get_rhs_tensor_accesses(cin: CIN) -> List[TensorAccess]:
        if hasattr(cin, "get_rhs_tensor_accesses"):
            return cin.get_rhs_tensor_accesses()
        return cin.tensor_accesses

    @staticmethod
    def _estimate_index_selectivity(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        selectivity = 1.0
        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                continue
            level_type = tensor_access.level_type_of_index_var(index_var)
            if Scheduler._is_sparse_level(level_type):
                selectivity *= costs.rho
        return max(selectivity, 1e-12)

    @staticmethod
    def _is_sparse_filter(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
    ) -> bool:
        has_sparse_level = False
        has_dense_level = False
        missing_from_some_tensor = False

        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                missing_from_some_tensor = True
                continue

            level_type = tensor_access.level_type_of_index_var(index_var)
            if Scheduler._is_sparse_level(level_type):
                has_sparse_level = True
            else:
                has_dense_level = True

        return has_sparse_level and (has_dense_level or missing_from_some_tensor)

    @staticmethod
    def _sparse_filter_score(
        index_var: IndexVar,
        rhs_tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        if not Scheduler._is_sparse_filter(index_var, rhs_tensor_accesses):
            return 0.0
        selectivity = Scheduler._estimate_index_selectivity(
            index_var=index_var,
            tensor_accesses=rhs_tensor_accesses,
            costs=costs,
        )
        return 1.0 - selectivity

    @staticmethod
    def _mode_position_score(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
    ) -> float:
        levels = [
            tensor_access.level_of_index_var(index_var)
            for tensor_access in tensor_accesses
            if tensor_access.has_index_var(index_var)
        ]
        if not levels:
            return float("inf")
        return sum(levels) / len(levels)

    @staticmethod
    def sort_by_sparsity_descending(
        index_vars: List[IndexVar],
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses
        base_order = {
            index_var: idx
            for idx, index_var in enumerate(Scheduler._unique_index_vars(index_vars))
        }

        sparse_scores = {
            index_var: Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in index_vars
        }

        return sorted(
            index_vars,
            key=lambda index_var: (
                -sparse_scores[index_var],
                Scheduler._mode_position_score(index_var, all_tensor_accesses),
                base_order[index_var],
                index_var.name,
            ),
        )

    @staticmethod
    def init_loop_order(
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        index_vars = Scheduler.get_index_variables(cin)
        return Scheduler.sort_by_sparsity_descending(
            index_vars=index_vars,
            cin=cin,
            costs=costs,
        )

    @staticmethod
    def move_to_position(
        loop_order: List[IndexVar],
        index_var: IndexVar,
        pos: int,
    ) -> List[IndexVar]:
        if index_var not in loop_order:
            return loop_order[:]

        new_loop_order = list(loop_order)
        current_pos = new_loop_order.index(index_var)
        elem = new_loop_order.pop(current_pos)
        new_pos = max(0, min(pos, len(new_loop_order)))
        new_loop_order.insert(new_pos, elem)
        return new_loop_order

    @staticmethod
    def _estimate_index_extent(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        sizes: List[float] = []
        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                continue
            tensor_shape = tensor_access.get_tensor().shape
            if tensor_shape is None:
                continue

            logical_pos = tensor_access.get_index_vars().index(index_var)
            if logical_pos < len(tensor_shape):
                sizes.append(float(tensor_shape[logical_pos]))

        if sizes:
            return max(max(sizes), 1.0)
        return float(costs.default_dim_size)

    @staticmethod
    def _estimate_tensor_nnz(tensor, costs: _CostModelConstants) -> float:
        level_types = tensor.get_level_types()
        if tensor.shape:
            shape = list(tensor.shape)
        else:
            shape = []

        if len(shape) < len(level_types):
            shape = shape + [costs.default_dim_size] * (len(level_types) - len(shape))

        mode_order = tensor.mode_order if tensor.mode_order else list(range(len(level_types)))

        nnz = 1.0
        for level, level_type in enumerate(level_types):
            logical_dim = mode_order[level] if level < len(mode_order) else level
            dim_size = float(shape[logical_dim]) if logical_dim < len(shape) else float(costs.default_dim_size)
            density = costs.rho if Scheduler._is_sparse_level(level_type) else 1.0
            nnz *= max(dim_size, 1.0) * density
        return max(nnz, 1.0)

    @staticmethod
    def _compute_comp_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses

        index_extents = {
            index_var: Scheduler._estimate_index_extent(
                index_var=index_var,
                tensor_accesses=all_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }
        index_selectivities = {
            index_var: Scheduler._estimate_index_selectivity(
                index_var=index_var,
                tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }

        sparse_filters = {
            index_var
            for index_var in loop_order
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        }

        comp_cost = 1.0
        for idx, index_var in enumerate(loop_order):
            if index_var in sparse_filters:
                applicable_filter = index_selectivities[index_var]
            else:
                applicable_filter = 1.0
                for sf in loop_order[:idx]:
                    if sf not in sparse_filters:
                        continue
                    for ta in rhs_tensor_accesses:
                        if (ta.has_index_var(sf) and ta.has_index_var(index_var)
                                and ta.get_parent_index_var(index_var) == sf
                                and Scheduler._is_sparse_level(ta.level_type_of_index_var(index_var))):
                            applicable_filter *= index_selectivities[sf]
                            break
            effective_iters = index_extents[index_var] * applicable_filter
            comp_cost *= max(effective_iters, 1e-12)
        return comp_cost

    @staticmethod
    def _compute_workspace_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        if Scheduler._has_dense_output(cin):
            return 0.0

        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        reduction_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_reduction_vars())
        free_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_free_vars())
        if not reduction_vars:
            return 0.0

        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        reduction_vars_in_loop = [
            index_var for index_var in reduction_vars if index_var in loop_pos
        ]
        if not reduction_vars_in_loop:
            return 0.0

        last_reduction_pos = max(loop_pos[index_var] for index_var in reduction_vars_in_loop)
        free_after_last_reduction = [
            index_var
            for index_var in free_vars
            if index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
        ]
        dim_workspace = len(free_after_last_reduction)
        if dim_workspace == 0:
            return 0.0

        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses
        index_extents = {
            index_var: Scheduler._estimate_index_extent(
                index_var=index_var,
                tensor_accesses=all_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }
        sparse_filters = {
            index_var
            for index_var in loop_order
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        }
        index_selectivities = {
            index_var: Scheduler._estimate_index_selectivity(
                index_var=index_var,
                tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }

        n_insert = 1.0
        for pos, index_var in enumerate(loop_order):
            if index_var in sparse_filters:
                applicable_filter = index_selectivities[index_var]
            else:
                applicable_filter = 1.0
                for sf in loop_order[:pos]:
                    if sf not in sparse_filters:
                        continue
                    for ta in rhs_tensor_accesses:
                        if (ta.has_index_var(sf) and ta.has_index_var(index_var)
                                and ta.get_parent_index_var(index_var) == sf
                                and Scheduler._is_sparse_level(ta.level_type_of_index_var(index_var))):
                            applicable_filter *= index_selectivities[sf]
                            break
            n_insert *= max(index_extents[index_var] * applicable_filter, 1.0)
            if pos >= last_reduction_pos:
                break

        n_entries = 1.0
        for index_var in free_after_last_reduction:
            n_entries *= max(index_extents[index_var], 1.0)

        insert_term = costs.c_insert * n_insert * dim_workspace
        sort_term = (
            costs.c_sort
            * n_entries
            * dim_workspace
            * math.log(max(n_entries, 2.0), 2)
        )
        return insert_term + sort_term

    @staticmethod
    def _compute_transposition_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        needs_transpose: Dict[str, bool] = {}
        tensor_nnz: Dict[str, float] = {}

        for tensor_access in cin.tensor_accesses:
            if tensor_access.is_workspace():
                continue

            sorted_index_vars = [
                index_var
                for index_var in tensor_access.get_sorted_index_vars()
                if index_var in loop_pos
            ]
            if len(sorted_index_vars) < 2:
                continue

            violates = any(
                loop_pos[sorted_index_vars[i]] > loop_pos[sorted_index_vars[i + 1]]
                for i in range(len(sorted_index_vars) - 1)
            )
            tensor = tensor_access.get_tensor()
            tensor_name = tensor.name
            needs_transpose[tensor_name] = needs_transpose.get(tensor_name, False) or violates
            if tensor_name not in tensor_nnz:
                tensor_nnz[tensor_name] = Scheduler._estimate_tensor_nnz(tensor, costs)

        return sum(
            costs.c_trans * tensor_nnz[tensor_name]
            for tensor_name, transpose in needs_transpose.items()
            if transpose
        )

    @staticmethod
    def cost_to_push(
        cin: CIN,
        loop_order: List[IndexVar],
        index_var: IndexVar,
        pos: int,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> float:
        if index_var not in loop_order:
            return 0.0

        new_loop_order = Scheduler.move_to_position(loop_order, index_var, pos)
        if new_loop_order == loop_order:
            return 0.0

        comp_cost_before = Scheduler._compute_comp_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        comp_cost_after = Scheduler._compute_comp_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )
        ws_cost_before = Scheduler._compute_workspace_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        ws_cost_after = Scheduler._compute_workspace_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )
        trans_cost_before = Scheduler._compute_transposition_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        trans_cost_after = Scheduler._compute_transposition_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )

        delta_comp = comp_cost_after - comp_cost_before
        delta_ws = ws_cost_after - ws_cost_before
        delta_trans = trans_cost_after - trans_cost_before
        return (
            costs.alpha * delta_comp
            + costs.beta * delta_ws
            + costs.gamma * delta_trans
        )

    @staticmethod
    def optimize_loop_order(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        sparse_filters_ordered = [
            index_var
            for index_var in Scheduler.sort_by_sparsity_descending(
                index_vars=loop_order,
                cin=cin,
                costs=costs,
            )
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        ]

        optimized_order = list(loop_order)
        for index_var in sparse_filters_ordered:
            if index_var not in optimized_order:
                continue

            current_pos = optimized_order.index(index_var)
            for pos in range(current_pos + 1, len(optimized_order) + 1):
                push_cost = Scheduler.cost_to_push(
                    cin=cin,
                    loop_order=optimized_order,
                    index_var=index_var,
                    pos=pos,
                    costs=costs,
                )
                if push_cost < 0:
                    optimized_order = Scheduler.move_to_position(
                        loop_order=optimized_order,
                        index_var=index_var,
                        pos=pos,
                    )
                    current_pos = optimized_order.index(index_var)

        return optimized_order

    @staticmethod
    def _build_mode_order_graph(
        index_vars: List[IndexVar],
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> Tuple[
        Dict[IndexVar, Set[IndexVar]],
        Dict[Tuple[IndexVar, IndexVar], Set[str]],
        Dict[str, float],
    ]:
        adjacency: Dict[IndexVar, Set[IndexVar]] = {index_var: set() for index_var in index_vars}
        edge_to_tensor_names: Dict[Tuple[IndexVar, IndexVar], Set[str]] = defaultdict(set)
        tensor_nnz: Dict[str, float] = {}
        allowed_index_vars = set(index_vars)

        for tensor_access in tensor_accesses:
            if tensor_access.is_workspace():
                continue

            sorted_index_vars = [
                index_var
                for index_var in tensor_access.get_sorted_index_vars()
                if index_var in allowed_index_vars
            ]
            if len(sorted_index_vars) < 2:
                continue

            tensor = tensor_access.get_tensor()
            tensor_name = tensor.name
            if tensor_name not in tensor_nnz:
                tensor_nnz[tensor_name] = Scheduler._estimate_tensor_nnz(tensor, costs)

            for i in range(len(sorted_index_vars) - 1):
                src = sorted_index_vars[i]
                dst = sorted_index_vars[i + 1]
                if src == dst:
                    continue
                adjacency[src].add(dst)
                edge_to_tensor_names[(src, dst)].add(tensor_name)

        return adjacency, edge_to_tensor_names, tensor_nnz

    @staticmethod
    def _contains_cycles(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
    ) -> bool:
        indegree: Dict[IndexVar, int] = {index_var: 0 for index_var in index_vars}
        for src, dsts in adjacency.items():
            for dst in dsts:
                if dst in indegree:
                    indegree[dst] += 1

        queue = deque([index_var for index_var in index_vars if indegree[index_var] == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for dst in sorted(adjacency.get(node, set()), key=lambda var: var.name):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)

        return visited != len(index_vars)

    @staticmethod
    def _find_cycle_edges(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
    ) -> List[Tuple[IndexVar, IndexVar]]:
        visited: Set[IndexVar] = set()
        in_stack: Set[IndexVar] = set()
        stack: List[IndexVar] = []

        def dfs(node: IndexVar) -> List[Tuple[IndexVar, IndexVar]]:
            visited.add(node)
            in_stack.add(node)
            stack.append(node)

            for neighbor in sorted(adjacency.get(node, set()), key=lambda var: var.name):
                if neighbor not in visited:
                    cycle_edges = dfs(neighbor)
                    if cycle_edges:
                        return cycle_edges
                elif neighbor in in_stack:
                    cycle_start = stack.index(neighbor)
                    cycle_nodes = stack[cycle_start:] + [neighbor]
                    return [
                        (cycle_nodes[i], cycle_nodes[i + 1])
                        for i in range(len(cycle_nodes) - 1)
                    ]

            stack.pop()
            in_stack.remove(node)
            return []

        for index_var in sorted(index_vars, key=lambda var: var.name):
            if index_var in visited:
                continue
            cycle_edges = dfs(index_var)
            if cycle_edges:
                return cycle_edges
        return []

    @staticmethod
    def _remove_cheapest_cycle_edge(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        cycle_edges: List[Tuple[IndexVar, IndexVar]],
        edge_to_tensor_names: Dict[Tuple[IndexVar, IndexVar], Set[str]],
        tensor_nnz: Dict[str, float],
    ) -> Tuple[IndexVar, IndexVar]:
        def edge_cost(edge: Tuple[IndexVar, IndexVar]) -> float:
            tensor_names = edge_to_tensor_names.get(edge, set())
            if not tensor_names:
                return float("inf")
            return sum(tensor_nnz.get(tensor_name, 1.0) for tensor_name in tensor_names)

        edge_to_remove = min(
            cycle_edges,
            key=lambda edge: (edge_cost(edge), edge[0].name, edge[1].name),
        )
        src, dst = edge_to_remove
        adjacency[src].remove(dst)
        return edge_to_remove

    @staticmethod
    def _topological_sort_with_priority(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
        priority: Dict[IndexVar, int],
    ) -> List[IndexVar]:
        indegree: Dict[IndexVar, int] = {index_var: 0 for index_var in index_vars}
        for src, dsts in adjacency.items():
            for dst in dsts:
                if dst in indegree:
                    indegree[dst] += 1

        zero_indegree = [
            index_var for index_var in index_vars if indegree[index_var] == 0
        ]
        order: List[IndexVar] = []
        while zero_indegree:
            zero_indegree.sort(
                key=lambda index_var: (
                    priority.get(index_var, len(priority)),
                    index_var.name,
                )
            )
            node = zero_indegree.pop(0)
            order.append(node)
            for dst in sorted(adjacency.get(node, set()), key=lambda var: var.name):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    zero_indegree.append(dst)

        if len(order) < len(index_vars):
            remaining = [index_var for index_var in index_vars if index_var not in order]
            remaining.sort(
                key=lambda index_var: (
                    priority.get(index_var, len(priority)),
                    index_var.name,
                )
            )
            order.extend(remaining)

        return order

    @staticmethod
    def apply_mode_order_constraints(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        if not loop_order:
            return loop_order

        unique_loop_order = Scheduler._unique_index_vars(loop_order)
        adjacency, edge_to_tensor_names, tensor_nnz = Scheduler._build_mode_order_graph(
            index_vars=unique_loop_order,
            tensor_accesses=cin.tensor_accesses,
            costs=costs,
        )

        while Scheduler._contains_cycles(adjacency, unique_loop_order):
            cycle_edges = Scheduler._find_cycle_edges(adjacency, unique_loop_order)
            if not cycle_edges:
                break
            Scheduler._remove_cheapest_cycle_edge(
                adjacency=adjacency,
                cycle_edges=cycle_edges,
                edge_to_tensor_names=edge_to_tensor_names,
                tensor_nnz=tensor_nnz,
            )

        priority = {
            index_var: pos for pos, index_var in enumerate(unique_loop_order)
        }
        return Scheduler._topological_sort_with_priority(
            adjacency=adjacency,
            index_vars=unique_loop_order,
            priority=priority,
        )

    @staticmethod
    def should_insert_workspace(
        cin: CIN,
        loop_order: List[IndexVar],
    ) -> bool:
        if cin.inserted_workspace:
            return False
        if not isinstance(cin, ForAll):
            return False

        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        reduction_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_reduction_vars())
        if not reduction_vars:
            return False

        free_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_free_vars())
        if not free_vars:
            return False

        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        reductions_in_loop = [
            index_var for index_var in reduction_vars if index_var in loop_pos
        ]
        if not reductions_in_loop:
            return False

        last_reduction_pos = max(loop_pos[index_var] for index_var in reductions_in_loop)
        free_after_last_reduction = [
            index_var
            for index_var in free_vars
            if index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
        ]
        if not free_after_last_reduction:
            return False

        if Scheduler._has_dense_output(cin):
            # For dense outputs, keep workspace insertion conservative:
            # only enable a 1D dense accumulator over a trailing dense axis.
            if len(free_after_last_reduction) != 1:
                return False
            result_tensor_accesses = cin.get_result_tensor_accesses()
            if not result_tensor_accesses:
                return False
            result_tensor_access = result_tensor_accesses[0]
            free_index_var = free_after_last_reduction[0]
            if not result_tensor_access.has_index_var(free_index_var):
                return False
            if (
                result_tensor_access.level_type_of_index_var(free_index_var)
                != LevelType.DENSE
            ):
                return False

        return any(
            index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
            for index_var in free_vars
        )

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
            loop_order, _ = Scheduler._extract_loop_chain(cin)
            if Scheduler.should_insert_workspace(cin, loop_order):
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
        while (
            isinstance(parent_forall, ForAll)
            and hasattr(parent_forall, "stmt")
            and not isinstance(parent_forall.stmt, Where)
        ):
            parent_forall = parent_forall.stmt

        # Tiling transform currently rewrites around a Where(producer, consumer)
        # structure. If workspace insertion was not needed/possible and no Where
        # exists, conservatively skip tiling for this index variable.
        if not (
            isinstance(parent_forall, ForAll)
            and isinstance(parent_forall.stmt, Where)
        ):
            return cin

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
                if (
                    workspace_access.indices
                    and workspace_access.indices[-1] == self.old_index_var
                ):
                    workspace_access.indices[-1] = self.new_index_var
                    workspace_access.update_indices(workspace_access.indices)

        replace_index_var_visitor = ReplaceIndexVarVisitor(
            old_index_var=index_var,
            new_index_var=inner_index_var,
        )
        replace_index_var_visitor.visit(where_stmt)

        # Wrap the new_cin in a new ForAll loop, with the outer index var.
        # Preserve scheduler metadata carried on the root CIN node.
        wrapped = ForAll(
            index_var=outer_index_var,
            stmt=cin,
        )
        wrapped.inserted_workspace = cin.inserted_workspace
        wrapped.no_tile_list = list(cin.no_tile_list)

        return wrapped

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

        reduction_vars_todo = [var for var in reduction_vars if var in index_vars_ordered]

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

        dim_workspace = len(free_vars_after_last_reduction)
        if dim_workspace == 0:
            return cin

        free_vars_after_last_reduction_level_types = [
            result_tensor_access.level_type_of_index_var(var)
            for var in free_vars_after_last_reduction
        ]

        are_all_dense_levels = all(
            level_type == LevelType.DENSE
            for level_type in free_vars_after_last_reduction_level_types
        )
        # Current lowering supports dense workspaces only for 1D accesses.
        # Fall back to sparse workspace representation for higher dimensions.
        if are_all_dense_levels and dim_workspace > 1:
            are_all_dense_levels = False

        if not allow_dense and are_all_dense_levels:
            return cin

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

        reduction_forall = (
            parent_forall if parent_forall.index_var == next_reduction_var else parent_forall.stmt
        )

        # If we have already inserted a workspace, then we should not insert another one.
        if isinstance(reduction_forall, Where):
            return cin

        # Create the producer forall
        producer_forall = copy.deepcopy(reduction_forall)

        producer_forall_tensor_access_parent = producer_forall

        # Iterate until the TensorAssign statement
        while not isinstance(producer_forall_tensor_access_parent.stmt, TensorAssign):
            producer_forall_tensor_access_parent = producer_forall_tensor_access_parent.stmt

        # Replace the TensorAssign's lhs with the workspace
        producer_forall_tensor_access_parent.stmt.lhs = workspace_access

        # Create the consumer forall
        consumer_forall = copy.deepcopy(reduction_forall)

        consumer_forall_tensor_access_parent = consumer_forall
        # Iterate until the TensorAssign statement
        while not isinstance(consumer_forall_tensor_access_parent.stmt, TensorAssign):
            consumer_forall_tensor_access_parent = consumer_forall_tensor_access_parent.stmt

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
        if isinstance(parent_forall.stmt, ForAll) and parent_forall.stmt.index_var == next_reduction_var:
            parent_forall.stmt = where_stmt
        else:
            new_cin = where_stmt

        new_cin.inserted_workspace = True

        return new_cin

    @staticmethod
    def select_loop_order(
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        loop_order = Scheduler.init_loop_order(cin, costs=costs)
        loop_order = Scheduler.optimize_loop_order(cin, loop_order, costs=costs)
        loop_order = Scheduler.apply_mode_order_constraints(
            cin, loop_order, costs=costs
        )

        # For sparse output with reduction variables, ensure at least one free
        # variable appears after the last reduction variable.  The lowerer
        # requires the innermost loop to correspond to a result-tensor level,
        # and workspace insertion needs free variables after the reduction to
        # define the workspace dimensions.
        if not Scheduler._has_dense_output(cin):
            cin_ivar_getter = CINIndexVariablesGetter()
            cin_ivar_getter.visit(cin)
            reduction_vars = set(
                Scheduler._unique_index_vars(cin_ivar_getter.get_reduction_vars())
            )
            if reduction_vars:
                reductions_in_loop = [v for v in loop_order if v in reduction_vars]
                if reductions_in_loop:
                    last_red_pos = max(loop_order.index(v) for v in reductions_in_loop)
                    free_after = [
                        v for v in loop_order[last_red_pos + 1:]
                        if v not in reduction_vars
                    ]
                    if not free_after:
                        free_vars = [v for v in loop_order if v not in reduction_vars]
                        if free_vars:
                            last_free = free_vars[-1]
                            idx = loop_order.index(last_free)
                            loop_order = (
                                loop_order[:idx]
                                + loop_order[idx + 1:]
                                + [last_free]
                            )

        return loop_order

    @staticmethod
    def _select_index_vars_to_tile(cin: CIN) -> List[IndexVar]:
        all_index_vars = cin.index_vars
        tensor_accesses = cin.tensor_accesses

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

        if not cin.loop_order:
            return []

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
        return index_vars_to_tile

    @staticmethod
    def _apply_tiling_heuristics(cin: CIN) -> CIN:
        if not isinstance(cin, ForAll):
            return cin
        for index_var in Scheduler._select_index_vars_to_tile(cin):
            cin = Scheduler.add_tile(cin, index_var, 32)
        return cin

    @staticmethod
    def auto_schedule(
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> CIN:
        if not isinstance(cin, ForAll):
            return cin

        loop_order = Scheduler.select_loop_order(cin, costs=costs)
        cin = Scheduler._rebuild_loop_nest(cin, loop_order)

        if Scheduler.should_insert_workspace(cin, loop_order):
            cin = Scheduler.insert_workspace(cin, allow_dense=True)

        if not isinstance(cin, ForAll):
            return cin

        cin = Scheduler._apply_tiling_heuristics(cin)
        return cin
