from typing import List, Optional, TypeVar

from taco_torch.cin import (
    TensorAssign,
    IndexVar,
    TensorAccess,
    IndexExpr,
    BinaryOp,
)

# Type aliases for type hints

# can be IterationGraph or any subclass of IterationGraph
IterationGraphType = TypeVar("IterationGraphType", bound="IterationGraph")


class TensorPath:
    """
    A tensor access expression like A[i, j, k] results in a path in the
    iteration graph through i, j, and k.

    The exact path (i->j->k or j->i->k) is determined by the order of levels in
    the tensor storage. The index variable that indexes into the mode at the
    first level of the tensor storage is the first element of the path, and so
    on.
    """

    path: Optional[List[IndexVar]]
    tensor_access: Optional[TensorAccess]

    def __init__(
        self,
        path: Optional[List[IndexVar]] = None,
        tensor_access: Optional[TensorAccess] = None,
    ):
        self.path = path
        self.tensor_access = tensor_access


class IterationGraph:
    """
    An iteration graph is a directed acyclic graph with nodes being the index
    variables and edges being the tensor paths.

    All tensor paths start from index variables higher in the tree and end at
    index variables lower in the tree.
    """

    # attributes to define a graph
    index_vars: Optional[List[IndexVar]]
    tensor_accesses: Optional[List[TensorAccess]]
    result_tensor_access: Optional[TensorAccess]

    tensor_paths: Optional[List[TensorPath]]
    result_tensor_path: Optional[TensorPath]

    def __init__(
        self,
        tensor_accesses: List[TenersorAccess],
        result_tensor_access: TensorAccess,
    ):
        self.index_vars = []
        self.tensor_accesses = tensor_accesses
        self.result_tensor_access = result_tensor_access

        self.tensor_paths = []
        self.result_tensor_path = None

    @classmethod
    def from_tensor_assignment(cls, tensor_assign: TensorAssign) -> IterationGraphType:
        """
        Construct an iteration graph from a tensor assignment.
        """
        result_tensor_access = tensor_assign.get_lhs()
        expr: IndexExpr = tensor_assign.get_rhs()

        # recursively collect all tensor accesses from the RHS
        tensor_accesses: List[TensorAccess] = []

        def collect_tensor_accesses(expr: IndexExpr):
            if isinstance(expr, TensorAccess):
                tensor_accesses.append(expr)
            elif isinstance(expr, BinaryOp):
                collect_tensor_accesses(expr.left)
                collect_tensor_accesses(expr.right)

        collect_tensor_accesses(expr)

        return cls(
            tensor_accesses=tensor_accesses,
            result_tensor_access=result_tensor_access,
        )

    def get_roots(self) -> List[IndexVar]:
        """
        Get the root index variables (those with no parents).
        """
        raise NotImplementedError

    def get_children(self, index_var: IndexVar) -> List[IndexVar]:
        """
        Get the children of an index variable.
        """
        raise NotImplementedError

    def get_parent(self, index_var: IndexVar) -> Optional[IndexVar]:
        """
        Get the parent of an index variable.
        """
        raise NotImplementedError

    def get_ancestors(self, index_var: IndexVar) -> List[IndexVar]:
        """
        Get the ancestors of an index variable, including itself.
        """
        raise NotImplementedError

    def get_descendants(self, index_var: IndexVar) -> List[IndexVar]:
        """
        Get the descendants of an index variable, including itself.
        """
        raise NotImplementedError

    def get_tensor_paths(self) -> List[TensorPath]:
        """
        Get all tensor paths in the iteration graph.
        """
        raise NotImplementedError

    def get_tensor_path(self, index_var: IndexVar) -> TensorPath:
        """
        Get the tensor path corresponding to a tensor read expression.
        """
        raise NotImplementedError

    def get_result_tensor_path(self) -> TensorPath:
        """
        Get the tensor path corresponding to the result tensor.
        """
        raise NotImplementedError
