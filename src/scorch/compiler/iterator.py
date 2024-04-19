from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Union

from . import llir
from .cin import TensorVar, TensorAccess, IndexVar
from ..format import LevelType

import pdb
import pprint


@dataclass(frozen=False)
class ModeIterator:
    _tensor_var: Optional[TensorVar] = None
    tensor_access: Optional[TensorAccess] = None
    index_var: Optional[IndexVar] = None
    parent_index_var: Optional[IndexVar] = None
    parent_iterator: Optional[ModeIterator] = None
    _level: Optional[int] = None
    level_type: Optional[LevelType] = None

    iterator_var_llir: Optional[llir.Var] = None
    iterator_var_begin_value_llir: Optional[llir.Expr] = None
    iterator_var_end_var_llir: Optional[llir.Var] = None
    iterator_var_end_value_llir: Optional[llir.Expr] = None

    coord_var_llir: Optional[llir.Var] = None
    coord_var_value_llir: Optional[llir.Expr] = None
    # coord_var_value_depends_on use default factory
    coord_var_value_depends_on: List[IndexVar] = field(default_factory=list)

    @property
    def level(self) -> int:
        assert self._level is not None, "level is None"
        return self._level

    @property
    def tensor_var(self) -> TensorVar:
        assert self._tensor_var is not None, "tensor_var is None"
        return self._tensor_var

    def get_index_var(self) -> IndexVar:
        assert self.index_var is not None, "index_var is None"
        return self.index_var

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

    def get_init_stmt(self) -> llir.VarInit:
        if (
            self.level_type == LevelType.COMPRESSED
            or self.level_type == LevelType.COORDINATE
        ):
            # if this is the parent-most coordinate level,
            # initialize the bounds using the size of the crd array
            if self.level == 0 and self.level_type == LevelType.COORDINATE:
                # int pB0 = 0;
                return llir.VarInit(
                    var=self.get_iterator_var_llir(),
                    value=llir.Literal(0),
                )

            else:
                return llir.VarInit(
                    var=self.get_iterator_var_llir(),
                    value=self.get_iterator_var_begin_value_llir(),
                )

    def get_iterator_end_init_stmts(self) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []

        if (
            self.level_type == LevelType.COMPRESSED
            or self.level_type == LevelType.COORDINATE
        ):
            # if this is the parent-most coordinate level,
            # initialize the bounds using the size of the crd array
            if self.level == 0 and self.level_type == LevelType.COORDINATE:
                # int pB0_end = B0_crd.size(0);
                if self.iterator_var_end_value_llir:
                    stmts.append(
                        llir.VarInit(
                            var=self.get_iterator_var_end_var_llir(),
                            value=llir.FunctionCall(
                                name=f"{self.tensor_var.name}{self.level}_crd_tensor.size",
                                args=[llir.Literal(0)],
                            ),
                        )
                    )
                    # If the next level is also a coordinate _level, then we need to
                    # initialize the next level's end iterator as well
                    if (self.tensor_access.num_levels - 1) > self._level and (
                        self.tensor_access.child_level_type_of_index_var(self.index_var)
                        == LevelType.COORDINATE
                    ):
                        stmts.append(
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"p{self.tensor_var.name}{self.level + 1}_end",
                                    type=llir.DataType.INT,
                                ),
                                value=llir.Literal(0),
                            )
                        )

                return stmts
            else:
                if self.iterator_var_end_value_llir:
                    stmts.append(
                        llir.VarInit(
                            var=self.get_iterator_var_end_var_llir(),
                            value=self.get_iterator_var_end_value_llir(),
                        )
                    )

        return stmts

    def get_init_stmts(self) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []

        if (
            self.level_type == LevelType.COMPRESSED
            or self.level_type == LevelType.COORDINATE
        ):
            # if this is the parent-most coordinate _level,
            # initialize the bounds using the size of the crd array
            if self.level == 0 and self.level_type == LevelType.COORDINATE:
                # int pB0 = 0;
                stmts.append(
                    llir.VarInit(
                        var=self.get_iterator_var_llir(),
                        value=llir.Literal(0),
                    )
                )
                # int pB0_end = B0_crd.size(0);
                if self.iterator_var_end_value_llir:
                    stmts.append(
                        llir.VarInit(
                            var=self.get_iterator_var_end_var_llir(),
                            value=llir.FunctionCall(
                                name=f"{self.tensor_var.name}{self.level}_crd_tensor.size",
                                args=[llir.Literal(0)],
                            ),
                        )
                    )
                    # If the next level is also a coordinate _level, then we need to
                    # initialize the next level's end iterator as well
                    if (self.tensor_access.num_levels - 1) > self._level and (
                        self.tensor_access.child_level_type_of_index_var(self.index_var)
                        == LevelType.COORDINATE
                    ):
                        stmts.append(
                            llir.VarInit(
                                var=llir.Var(
                                    name=f"p{self.tensor_var.name}{self.level + 1}_end",
                                    type=llir.DataType.INT,
                                ),
                                value=llir.Literal(0),
                            )
                        )

                return stmts
            else:
                stmts.append(
                    llir.VarInit(
                        var=self.get_iterator_var_llir(),
                        value=self.get_iterator_var_begin_value_llir(),
                    )
                )
                if self.iterator_var_end_value_llir:
                    stmts.append(
                        llir.VarInit(
                            var=self.get_iterator_var_end_var_llir(),
                            value=self.get_iterator_var_end_value_llir(),
                        )
                    )

        return stmts

    def __post_init__(self):
        # IndexVar must be provided
        assert (
            self.index_var is not None
        ), "An IndexVar must be provided to construct a ModeIterator"
        # Either TensorVar or TensorAccess must be provided
        assert (self._tensor_var is not None) or (
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
        if self._tensor_var is None:
            assert (
                self.tensor_access is not None
            ), "If _tensor_var is not provided, tensor_access must be provided"
            self._tensor_var = self.tensor_access.get_tensor()

        if self._level is None:
            assert (
                self.tensor_access is not None
            ), "If _level is not provided, tensor_access must be provided"
            # TODO: if self.index_var is not in self.tensor_access,
            #  check if the parent index var is in self.tensor_access
            tensor_access_index_vars = self.tensor_access.get_index_vars()
            if self.index_var in tensor_access_index_vars:
                self._level = self.tensor_access.level_of_index_var(self.index_var)
            elif (
                self.index_var.has_parent
                and self.index_var.parent in tensor_access_index_vars
            ):
                self._level = self.tensor_access.level_of_index_var(
                    self.index_var.parent
                )
            else:
                raise Exception(
                    f"IndexVar {self.index_var} not in TensorAccess {self.tensor_access}"
                )

        if self.level_type is None:
            assert (
                self.tensor_access is not None
            ), "If level_type is not provided, tensor_access must be provided"
            self.level_type = self._tensor_var.get_level_types()[
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
            if (
                self.level_type == LevelType.COMPRESSED
                or self.level_type == LevelType.COORDINATE
            ):
                self.iterator_var_llir = llir.Var(
                    name=f"p{self._tensor_var.get_name()}{self._level}",
                    type=llir.DataType.INT,
                )
            else:
                self.iterator_var_llir = llir.Var(
                    name=f"{self.index_var.name}",
                    type=llir.DataType.INT,
                )

        if self.level_type == LevelType.COORDINATE:
            self.iterator_var_begin_value_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level - 1}"
                if self.parent_index_var
                else f"{self._tensor_var.name}{self._level}_pos[0]",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_var_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level}_end",
                type=llir.DataType.INT,
            )
            if not self.parent_index_var:
                self.iterator_var_end_value_llir = llir.Var(
                    f"{self._tensor_var.name}{self._level}_pos[1]",
                    type=llir.DataType.INT,
                )
            self.coord_var_llir = llir.Var(
                name=f"{self.index_var.name}_{self._tensor_var.name}",
                type=llir.DataType.INT,
            )
            self.coord_var_value_llir = llir.Var(
                name=f"{self._tensor_var.name}{self._level}_crd[{self.iterator_var_llir.name}]",
                type=llir.DataType.INT,
            )

        elif self.level_type == LevelType.COMPRESSED:
            self.iterator_var_begin_value_llir = llir.Var(
                name=f"{self._tensor_var.name}{self._level}_pos[{self.parent_index_var.name}]"
                if self.parent_index_var
                else f"{self._tensor_var.name}{self._level}_pos[0]",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_var_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level}_end",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_value_llir = llir.Var(
                name=f"{self._tensor_var.name}{self._level}_pos[{self.parent_index_var.name} + 1]"
                if self.parent_index_var
                else f"{self._tensor_var.name}{self._level}_pos[1]",
                type=llir.DataType.INT,
            )
            self.coord_var_llir = llir.Var(
                name=f"{self.index_var.name}_{self._tensor_var.name}",
                type=llir.DataType.INT,
            )
            self.coord_var_value_llir = llir.Var(
                name=f"{self._tensor_var.name}{self._level}_crd[{self.iterator_var_llir.name}]",
                type=llir.DataType.INT,
            )

        elif self.level_type == LevelType.DENSE:
            # if self._level == 0:
            #     self.coord_var_llir = llir.Var(
            #         name=f"{self.index_var.name}",
            #         type=llir.DataType.INT,
            #     )
            # else:
            if not self.parent_iterator and self.parent_index_var:
                self.parent_iterator = ModeIterator(
                    tensor_access=self.tensor_access,
                    index_var=self.parent_index_var,
                )

            # self.coord_var_llir = llir.Var(
            #     name=f"{self.index_var.name}_{self._tensor_var.name}",
            #     type=llir.DataType.INT,
            # )
            self.coord_var_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level}",
                type=llir.DataType.INT,
            )

            if self.parent_iterator:
                def find_coord_var_value_llir(
                        curr_iterator: ModeIterator,
                        prev_levels_size: Optional[Union[llir.Var, llir.Mul]]
                ):
                    curr_index_var = llir.Var(
                        name=curr_iterator.index_var.name,
                        type=llir.DataType.INT,
                    )

                    if curr_iterator._level == 0:
                        if prev_levels_size is None:
                            return curr_index_var
                        else:
                            return llir.Mul(
                                left=curr_index_var,
                                right=prev_levels_size
                            )
                    else:
                        if prev_levels_size is None:
                            cur_level_size = curr_index_var
                            prev_levels_size = llir.Var(
                                name=f"{curr_iterator.tensor_var.name}{curr_iterator._level}_size",
                                type=llir.DataType.INT
                            )
                        else:
                            cur_level_size = llir.Mul(
                                left=curr_index_var,
                                right=prev_levels_size
                            )
                            prev_levels_size = llir.Mul(
                                left=llir.Var(
                                    name=f"{curr_iterator.tensor_var.name}{curr_iterator._level}_size",
                                    type=llir.DataType.INT
                                ),
                                right=prev_levels_size
                            )

                        return llir.Add(
                            left=find_coord_var_value_llir(curr_iterator.parent_iterator, prev_levels_size),
                            right=cur_level_size
                        )

                self.coord_var_value_llir = find_coord_var_value_llir(self, None)

                self.coord_var_value_depends_on.extend(
                    [self.index_var, self.parent_iterator.index_var]
                )

    def __str__(self) -> str:
        return f"ModeIterator({self.tensor_var}, {self.index_var}, {self.level})"

    def __repr__(self) -> str:
        return str(self)
