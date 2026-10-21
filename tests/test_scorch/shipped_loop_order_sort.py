"""The einsum loop-order sort as it shipped up to patch 0039, kept as a reference.

``topo_sort_characters`` and ``resolve_cycles`` below are ``scorch/utils.py`` as it stood
before that patch, line for line, with one addition marked in ``resolve_cycles``.  The
shipped loop did not return on every graph: on some it comes back to a graph it has
already had, and then repeats the same passes for ever.  Its next pass is a function of
the adjacency lists and of which operands already have an inverted edge, so the addition
records that pair before each pass and raises ``NeverReturns`` on one seen before.  A test
can then tell the three shipped outcomes apart without waiting: a loop order, the bare
``ValueError`` of ``max()`` on an empty list, and no return.

``test_einsum_loop_order_sort.py`` uses it to show that the sort in ``scorch/utils.py``
returns the same order wherever this one returns one.  Nothing else should import it.
"""

import math
from collections import defaultdict, deque


class NeverReturns(Exception):
    """The shipped loop reached a graph it had already had (see the module docstring)."""


def topo_sort_characters(substrings, tensors):
    # Create a directed graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    # Initialize in_degree for all characters
    for substring in substrings:
        for char in substring:
            nodes.add(char)
            if char not in in_degree:
                in_degree[char] = 0

    # Add edges to the graph
    for substring in substrings:
        for i in range(len(substring) - 1):
            if substring[i + 1] not in graph[substring[i]]:
                graph[substring[i]].append(substring[i + 1])
                in_degree[substring[i + 1]] += 1

    def topo_sort():
        zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]
        zero_in_degree_nodes = deque(zero_in_degree_nodes)

        result_ = []
        while zero_in_degree_nodes:
            node = zero_in_degree_nodes.popleft()
            result_.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_nodes.append(neighbor)

            # Sort newly zero in-degree nodes according to priority
            zero_in_degree_nodes = list(zero_in_degree_nodes)
            zero_in_degree_nodes = deque(zero_in_degree_nodes)
        return result_

    # Run topological sort
    result = topo_sort()

    if len(result) < len(nodes):
        resolve_cycles(nodes, graph, in_degree, substrings, tensors)
        # Re-run topo_sort with new graph, in_degree
        result = topo_sort()

        if len(result) < len(nodes):
            raise ValueError("resolve_cycles did not resolve cycles!")

    return result


def resolve_cycles(nodes, graph, in_degree, substrings, tensors):
    # Finds a cycle in the graph (if any) and then returns cycle's edges
    def find_cycle():
        visited = set()
        stack = []
        in_stack = set()

        def dfs(curr_node):
            visited.add(curr_node)
            stack.append(curr_node)
            in_stack.add(curr_node)

            for neighbor in graph[curr_node]:
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:  # If cycle edges were found, propagate them up
                        return result
                elif neighbor in in_stack:
                    # Cycle found, collect the cycle edges
                    cycle_start = stack.index(neighbor)
                    edges = [
                        (stack[i], stack[i + 1])
                        for i in range(cycle_start, len(stack) - 1)
                    ]
                    edges.append((stack[-1], neighbor))
                    return edges

            stack.pop()
            in_stack.remove(curr_node)
            return []

        for node in nodes:
            if node not in visited:
                cycle_edges_ = dfs(node)
                if cycle_edges_:
                    return cycle_edges_

        return []

    # Finds the cheapest edge to invert based on tensor shape and already inverted edges
    def invert_cheapest_edge(edges):
        inverted_tensor_indices = list(inverted_edges.keys())

        # Find edge associated with the smallest tensor, tiebreaker goes to result tensor. Represents cost as a tuple.
        def edge_cost(edge):
            edge_tensor_indices = edges_to_tensor_indices[edge]
            return (
                (0, -(max(edge_tensor_indices)))
                if set(edge_tensor_indices).issubset(set(inverted_tensor_indices))
                else (
                    sum(tensor_index_to_size[index] for index in edge_tensor_indices),
                    -max(edge_tensor_indices),
                )
            )

        min_cost_edge = min(edges, key=edge_cost)
        min_cost_tensor_indices = edges_to_tensor_indices[min_cost_edge]
        return min_cost_edge, min_cost_tensor_indices

    inverted_edges = defaultdict(list)
    edges_to_tensor_indices = defaultdict(list)

    # Build dictionary from edges (tuples) in graph to indices of tensors they appear in
    for tensor_index, substring in enumerate(substrings):
        for i in range(len(substring) - 1):
            edges_to_tensor_indices[(substring[i], substring[i + 1])].append(
                tensor_index
            )

    # Build dictionary from tensor index to size using tensor shapes for operands and result
    tensor_index_to_size = {}
    for i in range(len(tensors)):
        tensor_index_to_size[i] = math.prod(tensors[i].shape)

    result_size = 1
    for char in substrings[-1]:
        shape_index = substrings[0].find(char)
        if shape_index == -1:
            shape_index = substrings[1].find(char)
            result_size *= tensors[1].shape[shape_index]
        else:
            result_size *= tensors[0].shape[shape_index]

    tensor_index_to_size[len(tensors)] = result_size

    seen_states = set()  # the addition

    # Loop while there are cycles, resolving them by inverting edges and storing them in inverted_edges
    while True:
        cycle_edges = find_cycle()
        if not cycle_edges:
            break
        # The addition: a state seen before means no return.
        state = (
            tuple(tuple(graph[node]) for node in sorted(nodes)),
            frozenset(inverted_edges),
        )
        if state in seen_states:
            raise NeverReturns(substrings)
        seen_states.add(state)
        edge_to_invert, tensor_indices = invert_cheapest_edge(cycle_edges)
        graph[edge_to_invert[0]].remove(edge_to_invert[1])
        in_degree[edge_to_invert[1]] -= 1

        # If the cycle is just two edges, we will remove one edge, not invert it
        if edge_to_invert[0] not in graph[edge_to_invert[1]]:
            graph[edge_to_invert[1]].append(edge_to_invert[0])
            in_degree[edge_to_invert[0]] += 1

        for tensor_index in tensor_indices:
            inverted_edges[tensor_index].append(edge_to_invert)
