from dataclasses import dataclass
from typing import Dict, Optional, List

from taco_torch import llir
from taco_torch.cin import TensorVar, IndexStmt, TensorAccess, IndexVar
from taco_torch.format import TensorFormat, LevelType


@dataclass(frozen=False)
class ModeIterator:

    tensor_var: Optional[TensorVar] = None
    tensor_access: Optional[TensorAccess] = None
    index_var: Optional[IndexVar] = None
    parent_index_var: Optional[IndexVar] = None
    parent_iterator: Optional["ModeIterator"] = None
    level: Optional[int] = None
    level_type: Optional[LevelType] = None

    iterator_var_llir: Optional[llir.Var] = None
    iterator_var_begin_value_llir: Optional[llir.Expr] = None
    iterator_var_end_var_llir: Optional[llir.Var] = None
    iterator_var_end_value_llir: Optional[llir.Expr] = None

    coord_var_llir: Optional[llir.Var] = None
    coord_var_value_llir: Optional[llir.Expr] = None

    def get_init_stmts(self) -> List[llir.Stmt]:
        if self.level_type == LevelType.COMPRESSED:
            return [
                llir.VarAssign(
                    var=self.iterator_var_llir,
                    value=self.iterator_var_begin_value_llir,
                ),
                llir.VarAssign(
                    var=self.iterator_var_end_var_llir,
                    value=self.iterator_var_end_value_llir,
                ),
            ]

        return []

    def __post_init__(self):
        # IndexVar must be provided
        assert (
            self.index_var is not None
        ), "An IndexVar must be provided to construct a ModeIterator"
        # Either TensorVar or TensorAccess must be provided
        assert (self.tensor_var is not None) or (
            self.tensor_access is not None
        ), "Either a TensorVar or a TensorAccess must be provided to construct a ModeIterator"
        # Either parent_iterator or parent_index_var must be provided, or tensor_access must be provided
        assert (
            (self.parent_iterator is not None)
            or (self.parent_index_var is not None)
            or (self.tensor_access is not None)
        ), (
            "Either parent_iterator or parent_index_var or a TensorAccess"
            + " must be provided to construct a ModeIterator"
        )

        # If TensorVar is none, get it from TensorAccess
        if self.tensor_var is None:
            self.tensor_var = self.tensor_access.get_tensor()

        if self.level is None:
            self.level = self.tensor_access.level_of_index_var(self.index_var)

        if self.level_type is None:
            self.level_type = self.tensor_var.get_level_types()[
                self.tensor_access.level_of_index_var(self.index_var)
            ]

        if self.parent_index_var is None:
            if self.tensor_access is not None:
                self.parent_index_var = self.tensor_access.get_parent_index_var(
                    self.index_var
                )
            elif self.parent_iterator is not None:
                self.parent_index_var = self.parent_iterator.index_var
            else:
                raise Exception("Cannot infer parent_index_var")

        if self.iterator_var_llir is None:
            self.iterator_var_llir = llir.Var(
                name=f"{self.index_var.name}_{self.tensor_var.name}",
                type=llir.DataType.INT,
            )

        if self.level_type == LevelType.COMPRESSED:
            self.iterator_var_begin_value_llir = llir.Var(
                name=f"{self.tensor_var.name}{self.level}_pos[{self.parent_index_var.name}]"
                if self.parent_index_var
                else f"{self.tensor_var.name}{self.level}_pos[0]",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_var_llir = llir.Var(
                name=f"p{self.tensor_var.name}{self.level}_end",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_value_llir = llir.Var(
                name=f"{self.tensor_var.name}{self.level}_pos[{self.parent_index_var.name}+1]"
                if self.parent_index_var
                else f"{self.tensor_var.name}{self.level}_pos[1]",
                type=llir.DataType.INT,
            )
            self.coord_var_llir = llir.Var(
                name=f"{self.index_var.name}_{self.tensor_var.name}{self.level}",
                type=llir.DataType.INT,
            )
            self.coord_var_value_llir = llir.Var(
                name=f"{self.tensor_var.name}{self.level}_crd[{self.iterator_var_llir.name}]",
                type=llir.DataType.INT,
            )


class ModeIteratorOld:
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
