from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple, cast

from . import llir
from .cin import TensorVar, TensorAccess, IndexVar
from .llir_traversal import (
    LLIRPath,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from ..format import LevelType

ModePositionBegin = Tuple[str, Optional[str]]
_RAW_POSITION_ACCESS = re.compile(r"(\w+_pos)\[")


def _plain_mode_position_var(
    value: object,
    data_type: llir.DataType,
) -> Optional[llir.Var]:
    if type(value) is not llir.Var:
        return None
    var = cast(llir.Var, value)
    if not (
        type(var.name) is str
        and var.name.isidentifier()
        and var.type is data_type
        and var.is_ptr is False
        and var.is_restrict is False
        and var.tensor_access is None
    ):
        return None
    return var


def _int_mode_position_literal(value: object, expected: int) -> bool:
    return (
        type(value) is llir.Literal
        and type(cast(llir.Literal, value).value) is int
        and cast(llir.Literal, value).value == expected
        and cast(llir.Literal, value).data_type is llir.DataType.INT
    )


def _mode_position_access(
    value: object,
) -> Optional[Tuple[llir.ArrayAccess, llir.Var]]:
    if type(value) is not llir.ArrayAccess:
        return None
    access = cast(llir.ArrayAccess, value)
    array = _plain_mode_position_var(access.array, llir.DataType.PTR_INT)
    if (
        access.tensor_access is not None
        or array is None
        or not array.name.endswith("_pos")
        or array.name == "_pos"
    ):
        return None
    return access, array


def match_mode_position_begin(value: object) -> Optional[ModePositionBegin]:
    """Match one exact structured compressed-mode position begin.

    The optional second tuple member is the parent-position variable name;
    ``None`` denotes the root ``[0]`` form.
    """

    matched = _mode_position_access(value)
    if matched is None:
        return None
    access, array = matched
    parent = _plain_mode_position_var(access.index, llir.DataType.INT)
    if parent is not None:
        return array.name, parent.name
    if _int_mode_position_literal(access.index, 0):
        return array.name, None
    return None


def match_mode_position_access(value: object) -> Optional[str]:
    """Match one exact structured compressed-mode position begin or end."""

    matched = _mode_position_access(value)
    if matched is None:
        return None
    access, array = matched
    if _plain_mode_position_var(access.index, llir.DataType.INT) is not None:
        return array.name
    if _int_mode_position_literal(access.index, 0) or _int_mode_position_literal(
        access.index, 1
    ):
        return array.name
    if type(access.index) is not llir.Add:
        return None
    index = cast(llir.Add, access.index)
    if (
        index.op == "+"
        and _plain_mode_position_var(index.left, llir.DataType.INT) is not None
        and _int_mode_position_literal(index.right, 1)
    ):
        return array.name
    return None


class _ModePositionArrayCollector(LLIRWalker):
    def __init__(self, context: LLIRTraversalContext) -> None:
        super().__init__(context)
        self.position_arrays: List[str] = []

    def _append(self, name: str) -> None:
        if name not in self.position_arrays:
            self.position_arrays.append(name)

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        matched = match_mode_position_access(node)
        if matched is not None:
            self._append(matched)
            return
        if type(node) is not llir.RawStmt:
            return
        code = cast(llir.RawStmt, node).code
        if type(code) is not str:
            raise LLIRTraversalError(
                LLIRTraversalDiagnostic(
                    code="invalid_sparse_position_raw_statement",
                    message="RawStmt.code must be a string during position discovery",
                    path=path + ("code",),
                    node_type=type(code).__name__,
                    stage=self.context.stage,
                    pass_name=self.context.pass_name,
                )
            )
        for raw_match in _RAW_POSITION_ACCESS.finditer(code):
            self._append(raw_match.group(1))


def collect_mode_position_arrays(
    value: LLIRValue,
    context: LLIRTraversalContext,
) -> List[str]:
    """Collect structured mode bounds plus explicit RawStmt compatibility."""

    collector = _ModePositionArrayCollector(context)
    collector.walk(value)
    return collector.position_arrays


def match_mode_position_bounds(begin: object, end: object) -> Optional[str]:
    """Return the exact rendered begin for one coherent position-bound pair."""

    begin_match = match_mode_position_begin(begin)
    end_match = _mode_position_access(end)
    if begin_match is None or end_match is None:
        return None
    array_name, parent_name = begin_match
    end_access, end_array = end_match
    if end_array.name != array_name:
        return None
    if parent_name is None:
        if not _int_mode_position_literal(end_access.index, 1):
            return None
        return f"{array_name}[0]"
    if type(end_access.index) is not llir.Add:
        return None
    index = cast(llir.Add, end_access.index)
    parent = _plain_mode_position_var(index.left, llir.DataType.INT)
    if (
        index.op != "+"
        or parent is None
        or parent.name != parent_name
        or not _int_mode_position_literal(index.right, 1)
    ):
        return None
    return f"{array_name}[{parent_name}]"


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

    def _compressed_position_access(self, offset: int) -> llir.ArrayAccess:
        assert offset in (0, 1), "compressed position offset must be zero or one"
        if self.parent_index_var is None:
            index: llir.Expr = llir.Literal(offset, llir.DataType.INT)
        else:
            parent = llir.Var(
                name=f"p{self.tensor_var.name}{self.level - 1}",
                type=llir.DataType.INT,
            )
            index = (
                parent
                if offset == 0
                else llir.Add(
                    parent,
                    llir.Literal(1, llir.DataType.INT),
                )
            )
        return llir.ArrayAccess(
            array=llir.Var(
                name=f"{self.tensor_var.name}{self.level}_pos",
                type=llir.DataType.PTR_INT,
            ),
            index=index,
        )

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
                    # For all future levels, initialize the end iterator to 0 for all the levels that are also coordinate levels
                    for i in range(1, self.tensor_access.num_levels):
                        if self.tensor_access.level_types()[i] == LevelType.COORDINATE:
                            stmts.append(
                                llir.VarInit(
                                    var=llir.Var(
                                        name=f"p{self.tensor_var.name}{i}_end",
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
            # if this is the parent-most coordinate level,
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
                    # For all future levels, initialize the end iterator to 0 for all the levels that are also coordinate levels
                    for i in range(1, self.tensor_access.num_levels):
                        if self.tensor_access.level_types()[i] == LevelType.COORDINATE:
                            stmts.append(
                                llir.VarInit(
                                    var=llir.Var(
                                        name=f"p{self.tensor_var.name}{i}_end",
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
                name=(
                    f"p{self._tensor_var.name}{self._level - 1}"
                    if self.parent_index_var
                    else f"{self._tensor_var.name}{self._level}_pos[0]"
                ),
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
            self.coord_var_value_llir = llir.ArrayAccess(
                array=llir.Var(
                    name=f"{self._tensor_var.name}{self._level}_crd",
                    type=llir.DataType.PTR_INT,
                ),
                index=llir.Var(
                    name=self.iterator_var_llir.name,
                    type=self.iterator_var_llir.type,
                ),
            )

        elif self.level_type == LevelType.COMPRESSED:
            self.iterator_var_begin_value_llir = self._compressed_position_access(0)
            self.iterator_var_end_var_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level}_end",
                type=llir.DataType.INT,
            )
            self.iterator_var_end_value_llir = self._compressed_position_access(1)
            self.coord_var_llir = llir.Var(
                name=f"{self.index_var.name}_{self._tensor_var.name}",
                type=llir.DataType.INT,
            )
            self.coord_var_value_llir = llir.ArrayAccess(
                array=llir.Var(
                    name=f"{self._tensor_var.name}{self._level}_crd",
                    type=llir.DataType.PTR_INT,
                ),
                index=llir.Var(
                    name=self.iterator_var_llir.name,
                    type=self.iterator_var_llir.type,
                ),
            )

        elif self.level_type == LevelType.DENSE:
            if not self.parent_iterator and self.parent_index_var:
                self.parent_iterator = ModeIterator(
                    tensor_access=self.tensor_access,
                    index_var=self.parent_index_var,
                )

            self.coord_var_llir = llir.Var(
                name=f"p{self._tensor_var.name}{self._level}",
                type=llir.DataType.INT,
            )

            if self.parent_iterator:
                # e.g. int pB1 = pB0 * B1_size + k;
                self.coord_var_value_llir = llir.Add(
                    left=llir.Mul(
                        left=llir.Var(
                            name=f"p{self.tensor_var.name}{self.parent_iterator.level}",
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
            else:
                # e.g. int pB0 = i;
                self.coord_var_value_llir = llir.Var(
                    name=self.index_var.name,
                    type=llir.DataType.INT,
                )

                self.coord_var_value_depends_on.append(self.index_var)

    def __str__(self) -> str:
        return f"ModeIterator({self.tensor_var}, {self.index_var}, {self.level})"

    def __repr__(self) -> str:
        return str(self)
