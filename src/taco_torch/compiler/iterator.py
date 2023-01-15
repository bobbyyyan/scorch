from dataclasses import dataclass
from typing import Optional, List

from src.taco_torch.compiler import llir
from src.taco_torch.compiler.cin import TensorVar, TensorAccess, IndexVar
from src.taco_torch.format import LevelType


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

    def get_coord_var_llir(self) -> llir.Var:
        assert self.coord_var_llir is not None, "coord_var_llir is None"
        return self.coord_var_llir

    def get_coord_var_value_llir(self) -> llir.Expr:
        assert self.coord_var_value_llir is not None, "coord_var_value_llir is None"
        return self.coord_var_value_llir

    def get_iterator_var_llir(self) -> llir.Var:
        assert self.iterator_var_llir is not None, "iterator_var_llir is None"
        return self.iterator_var_llir

    def get_iterator_var_begin_value_llir(self) -> llir.Expr:
        assert (
            self.iterator_var_begin_value_llir is not None
        ), "iterator_var_begin_value_llir is None"
        return self.iterator_var_begin_value_llir

    def get_iterator_var_end_var_llir(self) -> llir.Var:
        assert (
            self.iterator_var_end_var_llir is not None
        ), "iterator_var_end_var_llir is None"
        return self.iterator_var_end_var_llir

    def get_iterator_var_end_value_llir(self) -> llir.Expr:
        assert (
            self.iterator_var_end_value_llir is not None
        ), "iterator_var_end_value_llir is None"
        return self.iterator_var_end_value_llir

    def get_init_stmts(self) -> List[llir.Stmt]:
        if self.level_type == LevelType.COMPRESSED:
            return [
                llir.VarInit(
                    var=self.get_iterator_var_llir(),
                    value=self.get_iterator_var_begin_value_llir(),
                ),
                llir.VarInit(
                    var=self.get_iterator_var_end_var_llir(),
                    value=self.get_iterator_var_end_value_llir(),
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
            assert (
                self.tensor_access is not None
            ), "If tensor_var is not provided, tensor_access must be provided"
            self.tensor_var = self.tensor_access.get_tensor()

        if self.level is None:
            assert (
                self.tensor_access is not None
            ), "If level is not provided, tensor_access must be provided"
            self.level = self.tensor_access.level_of_index_var(self.index_var)

        if self.level_type is None:
            assert (
                self.tensor_access is not None
            ), "If level_type is not provided, tensor_access must be provided"
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
