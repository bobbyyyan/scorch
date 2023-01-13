from typing import Dict, Optional

from taco_torch import llir
from taco_torch.cin import TensorVar, IndexStmt, TensorAccess, IndexVar
from taco_torch.format import TensorFormat


class ModeIterator:
    def __init__(self):
        self.tensor: Optional[llir.Expr] = None
        self.index_var: Optional[IndexVar] = None
        self.pos_var: Optional[llir.Expr] = None
        self.coord_var: Optional[llir.Expr] = None
        self.begin_var: Optional[llir.Expr] = None
        self.end_var: Optional[llir.Expr] = None
        self.seg_end_var: Optional[llir.Expr] = None
        self.valid_var: Optional[llir.Expr] = None
        self.parent: Optional[ModeIterator] = None

    def from_cin(
        self,
        stmt: IndexStmt,
        tensor_vars_to_llir: Dict[TensorVar, llir.Expr],
    ):
        self.stmt = stmt
        self.tensor_vars_to_llir = tensor_vars_to_llir

    def from_tensor_expr(self, tensor_expr: llir.Expr) -> None:
        self.tensor = tensor_expr
        self.pos_var = llir.Literal(0)
        self.coord_var = llir.Literal(0)
        self.end_var = llir.Literal(1)


class TensorIterators:
    def __init__(self):
        self.level_iterators: Dict[ModeAccess, ModeIterator] = {}
        self.mode_accesses: Dict[ModeIterator, ModeAccess] = {}
        self.mode_iterators: Dict[IndexVar, ModeIterator] = {}

    def create_access_iterators(
        self,
        access: TensorAccess,
        fmt: TensorFormat,
        tensor_var_llir: llir.Expr,
    ) -> None:
        parent_iterator = ModeIterator.from_tensor_expr(tensor_var_llir)
        self.level_iterators[ModeAccess(access, 0)] = parent_iterator


class ModeAccess:
    """
    A ModeAccess is the access of a single mode in a tensor access.
    e.g. A[i, j] consists of two mode accesses, A[1] and A[2].
    """

    def __init__(self, tensor_access: TensorAccess, mode: int):
        self.tensor_access: Optional[TensorAccess] = tensor_access
        self.mode: Optional[int] = mode
