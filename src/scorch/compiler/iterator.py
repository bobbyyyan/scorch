from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from . import llir
from .cin import TensorVar, TensorAccess, IndexVar
from ..format import LevelType


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
    iterator_var_stride_size: Optional[int] = None
    iterator_var_begin_value_llir: Optional[llir.Expr] = None
    iterator_var_end_var_llir: Optional[llir.Var] = None
    iterator_var_end_value_llir: Optional[llir.Expr] = None

    coord_var_llir: Optional[llir.Var] = None
    coord_var_value_llir: Optional[llir.Expr] = None
    # coord_var_value_depends_on use default factory
    coord_var_value_depends_on: List[IndexVar] = field(default_factory=list)
    
    coord_var_array_llir: Optional[llir.Var] = None
    coord_var_array_value_llir: Optional[llir.Expr] = None

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

    def is_strided(self) -> bool:
        return any(self.tensor_var.get_level_stride_sizes())
    
    # def get_coord_var_array_llir(self) -> llir.Var:
    #     assert self.coord_var_array_llir is not None, "coord_var_vector_llir is None"
    #     return self.coord_var_array_llir

    def get_coord_var_value_llir(self) -> llir.Expr:
        assert self.coord_var_value_llir is not None, "coord_var_value_llir is None"
        return self.coord_var_value_llir
    
    # def get_coord_var_array_value_llir(self) -> llir.Expr:
    #     assert self.coord_var_array_value_llir is not None, "coord_var_array_value_llir is None"
    #     return self.coord_var_array_value_llir

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
                self.iterator_var_stride_size = llir.Var(
                    name=f"{self._tensor_var.get_name()}{self._level}_stride",
                    type=llir.DataType.INT,
                )
            else:
                self.iterator_var_llir = llir.Var(
                    name=f"{self.index_var.name}",
                    type=llir.DataType.INT,
                )
                self.iterator_var_stride_size = llir.Var(
                    name=f"{self._tensor_var.get_name()}{self._level}_stride",
                    type=llir.DataType.INT,
                )
                
        level_stride_sizes = self._tensor_var.get_level_stride_sizes()
        is_strided = any(level_stride_sizes)

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
            if is_strided:
                self.coord_var_array_llir = llir.Var(
                    name=f"p{self._tensor_var.name}{self._level}_stride_array",
                    type=llir.DataType.CVECTOR_INT,
                )
            if self.parent_iterator:
                # e.g. int pB1 = j * B1_size + k;
                if level_stride_sizes[self._level]:
                    self.coord_var_value_llir = llir.Add(
                        left=llir.Mul(
                            left=llir.Mul(
                                left=llir.Var(
                                    name=self.parent_iterator.get_iterator_var_llir().name,
                                    type=llir.DataType.INT,
                                ),
                                right=llir.Var(
                                    name=f"{self.tensor_var.name}{self.level}_size",
                                    type=llir.DataType.INT,
                                ),
                            ),
                            right=llir.Mul(
                                left=llir.Var(
                                    name=f"{self.tensor_var.name}{self.parent_iterator.level}_stride",
                                    type=llir.DataType.INT,
                                ),
                                right=llir.Var(
                                    name=f"{self.tensor_var.name}{self.level}_stride",
                                    type=llir.DataType.INT,
                                ),
                            )
                        ),
                        right=llir.Mul(
                            left=llir.Var(
                                name=self.index_var.name,
                                type=llir.DataType.INT,
                            ),
                            right=llir.Var(
                                name=f"{self.tensor_var.name}{self.level}_stride",
                                type=llir.DataType.INT,
                            ),
                        ),
                    )
                    
                    self.coord_var_array_value_llir = llir.VarDecl(
                        llir.Var(
                            name=f"{self.tensor_var.name}{self.level}_stride_array",
                            type=llir.DataType.CVECTOR_INT,
                        )
                    )
                    self.stride_non_zero = llir.Var(
                        name="stride_non_zero",
                        type=llir.DataType.BOOL,
                    )
                    self.stride_non_zero_init = llir.VarInit(
                        var=llir.Var(
                            name="stride_non_zero",
                            type=llir.DataType.BOOL,
                        ),
                        value=llir.Literal(1),
                    )
                    self.stride_array_init = llir.VarDecl(
                        llir.Var(
                            name=f"{self.tensor_var.name}{self.level}_stride_array",
                            type=llir.DataType.CVECTOR_INT,
                        )
                    )
                    stride_loop_body = []
                    if_non_zero_assign_true = llir.IfThenElse(
                        cond=llir.BinOp(
                            op="!=",
                            left=llir.Var(
                                name=f"{self.tensor_var.name}_val[p{self._tensor_var.name}{self._level} + p_stride]",
                                type=llir.DataType.INT,
                            ),
                            right=llir.Literal(value="0"),
                        ),
                        then_body=llir.Assign(
                            var=llir.Var(
                                name="stride_non_zero",
                                type=llir.DataType.BOOL,
                            ),
                            value=llir.Literal(1),
                        ),
                    )
                    stride_loop_body.extend([if_non_zero_assign_true])
                    self.stride_for_loops = self.generate_stride_for_loops(level_stride_sizes, 0, stride_loop_body)
                else:
                    self.coord_var_value_llir = llir.Add(
                        left=llir.Mul(
                            left=llir.Var(
                                name=self.parent_iterator.get_iterator_var_llir().name,
                                type=llir.DataType.INT,
                            ),
                            right=llir.Var(
                                name=f"{self.tensor_var.name}{self.level}_size",
                                type=llir.DataType.INT,
                            ),
                        ),
                        right=llir.Var(
                            name=self.index_var.name,
                            type=llir.DataType.INT,
                        ),
                    )

                self.coord_var_value_llir = llir.Add(
                    left=llir.Mul(
                        left=llir.Var(
                            name=self.parent_iterator.get_iterator_var_llir().name,
                            type=llir.DataType.INT,
                        ),
                        right=llir.Var(
                            name=f"{self.tensor_var.name}{self.level}_size",
                            type=llir.DataType.INT,
                        ),
                    ),
                    right=llir.Var(
                        name=self.index_var.name,
                        type=llir.DataType.INT,
                    ),
                )
                self.coord_var_value_depends_on.extend(
                    [self.index_var, self.parent_iterator.index_var]
                )
                
    
    def generate_stride_for_loops(self, level_stride_sizes, cur_level, custom_body, p_stride=None, save_to_array_idx=None):
        # for strided tensors
        # should work for all kinds of strides!
        if cur_level >= len(level_stride_sizes):
            init_p_stride = llir.VarInit(
                var=llir.Var(
                    name=f"p_stride",
                    type=llir.DataType.INT,
                ),
                value=p_stride,
            )
            init_save_to_array_idx = llir.VarInit(
                var=llir.Var(
                    name=f"save_to_array_idx",
                    type=llir.DataType.INT,
                ),
                value=save_to_array_idx,
            )
            # save the stride array
            assign_coord_var_array = llir.Assign(
                var=llir.Var(
                    name=f"{self._tensor_var.name}{self._level}_stride_array[save_to_array_idx]",
                    type=llir.DataType.INT,
                ),
                value=llir.Var(
                    name=f"p{self._tensor_var.name}{self._level} + p_stride",
                    type=llir.DataType.INT,
                ),
            )
            custom_body.extend([assign_coord_var_array])
            return [init_p_stride, init_save_to_array_idx, custom_body]
        
        if p_stride is None:
            p_stride = llir.Var(
                name=f"i{cur_level}",
                type=llir.DataType.INT,
            )
        else:
            p_stride = llir.Mul(
                left=p_stride,
                right=llir.Mul(
                    left=llir.Var(
                        name=f"{self.tensor_var.name}{cur_level}_stride",
                        type=llir.DataType.INT,
                    ),
                    right=llir.Var(
                        name=f"{self.tensor_var.name}{cur_level}_size",
                        type=llir.DataType.INT,
                    )
                ),
            )
            p_stride = llir.Add(
                left=p_stride,
                right=llir.Var(
                    name=f"i{cur_level}",
                    type=llir.DataType.INT,
                )
            )
        
        if save_to_array_idx is None:
            save_to_array_idx = llir.Var(
                name=f"i{cur_level}",
                type=llir.DataType.INT,
            )
        else:
            save_to_array_idx = llir.Mul(
                left=save_to_array_idx,
                right=llir.Var(
                    name=f"{self.tensor_var.name}{cur_level}_stride",
                    type=llir.DataType.INT,
                ),
            )
            save_to_array_idx = llir.Add(
                left=save_to_array_idx,
                right=llir.Var(
                    name=f"i{cur_level}",
                    type=llir.DataType.INT,
                ),
            )
        level_stride_size = level_stride_sizes[cur_level]
        if level_stride_size:
            return llir.ForLoop(
                    init=llir.VarInit(
                        var=llir.Var(
                            name=f"i{cur_level}",
                            type=llir.DataType.INT,
                        ),
                        value=llir.Literal(0),
                    ),
                    cond=llir.BinOp(
                        op="<",
                        left=llir.Var(
                            name=f"i{cur_level}",
                            type=llir.DataType.INT,
                        ),
                        right=llir.Var(
                            name=f"{self.tensor_var.name}{cur_level}_stride",
                            type=llir.DataType.INT,
                        ),
                    ),
                    update=llir.Increment(
                        var=llir.Var(
                            name=f"i{cur_level}",
                            type=llir.DataType.INT,
                        ),
                    ),
                    body=self.generate_stride_for_loops(level_stride_sizes, cur_level + 1, custom_body, p_stride, save_to_array_idx)
                )
        else:
            return self.generate_stride_for_loops(level_stride_sizes, cur_level + 1, custom_body, p_stride, save_to_array_idx)



    def __str__(self) -> str:
        return f"ModeIterator({self.tensor_var}, {self.index_var}, {self.level})"

    def __repr__(self) -> str:
        return str(self)
