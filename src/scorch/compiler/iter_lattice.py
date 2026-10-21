from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from itertools import product
from typing import (
    List,
    Optional,
    Dict,
    TYPE_CHECKING,
    Iterable,
    Sequence,
    Set,
    Tuple,
    Union,
)

from . import llir
from .llir_traversal import LLIRPath, LLIRTraversalContext, LLIRWalker
from .torch_cpp_abi import dense_block_check
from ..utils import dtype_to_c_datatype, flatten_2d_list

if TYPE_CHECKING:
    from src.scorch.compiler.cin_lowerer import CINLowerer

from .cin import (
    IndexVar,
    ForAll,
    TensorAccess,
    IndexStmt,
    TensorAssign,
    CIN,
    BinaryOp,
    Operation,
    IndexExpr,
    CINIndexVariablesGetter,
    Where,
)
from ..format import LevelType
from .cin_analysis import CINDiagnostic
from .diagnostics import UnsupportedFeature
from .iterator import ModeIterator
from ..exceptions import ScorchAssertionError, ScorchNotImplementedError

# Short names for the statement-level result-storage marker's vocabulary.  Spelled
# from ``llir`` rather than imported from ``cin_lowerer``, which imports this
# module.
RESULT_VALUES = llir.ResultStorageArray.VALUES
RESULT_POS = llir.ResultStorageArray.POS
RESULT_CRD = llir.ResultStorageArray.CRD
STORAGE_READ = llir.ResultStorageDirection.READ
STORAGE_WRITE = llir.ResultStorageDirection.WRITE

# How a zero of each floating-point element type is spelled in a scalar
# declaration; every integer element type spells it ``0``.
_SCALAR_ZERO_SPELLINGS = {
    llir.DataType.FLOAT32: "0.0f",
    llir.DataType.FLOAT64: "0.0",
}


def compressed_child_catch_up(
    cin_lowerer: Optional["CINLowerer"],
    result_tensor_access: TensorAccess,
    child_level: int,
    target: llir.Expr,
) -> List[llir.Stmt]:
    """Close the position entries of every cell skipped before ``target``.

    ``child_level`` is a compressed result level below a dense one.  Its
    position array has one entry per cell of the dense level, and a loop that
    visits only some cells closes the ones it skipped when it reaches the
    next: ``for (; pos_index < target; pos_index++) pos[pos_index + 1] =
    crd.size();``.  ``target`` is the visited cell's POSITION in the dense
    level.
    """

    result_tensor_name = result_tensor_access.get_tensor().get_name()
    pos_index_var_llir = llir.Var(
        name=f"{result_tensor_name}{child_level}_pos_index",
        type=llir.DataType.INT,
    )
    # Built here rather than spelled inline: the callers read their result
    # access off an optional attribute, and one typed call site is easier to
    # keep right than two.
    catch_up_marker = (
        None
        if cin_lowerer is None or result_tensor_access is None
        else cin_lowerer._result_storage(
            result_tensor_access,
            (RESULT_POS, child_level, STORAGE_WRITE),
            (RESULT_CRD, child_level, STORAGE_READ),
        )
    )
    return [
        llir.Comment("Assemble COMPRESSED level"),
        llir.ForLoop(
            init=None,
            cond=llir.BinOp(
                op="<",
                left=pos_index_var_llir,
                right=target,
            ),
            update=llir.Increment(var=pos_index_var_llir),
            body=[
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_tensor_name}{child_level}_pos",
                            type=llir.DataType.STD_VECTOR_C_INT,
                        ),
                        index=llir.Add(
                            llir.Var(
                                name=pos_index_var_llir.name,
                                type=llir.DataType.INT,
                            ),
                            llir.Literal(1),
                        ),
                    ),
                    value=llir.FunctionCall(
                        name=f"{result_tensor_name}{child_level}_crd.size",
                        args=[],
                    ),
                    result_storage=catch_up_marker,
                )
            ],
        ),
    ]


def dense_block_commit(
    cin_lowerer: "CINLowerer",
    result_tensor_access: TensorAccess,
    level: int,
    coordinate: llir.Expr,
) -> List[llir.Stmt]:
    """Commit a compressed coordinate whose dense block below was written.

    ``coordinate`` is the coordinate to record at level ``level - 1``: the
    enclosing loop's variable, or a drained key's previous value.

    ``level`` is a dense result level directly below a compressed level L.
    The dense levels from here down form one block of ``B`` cells (the
    product of their extents) per stored coordinate of L, at ``q * B`` where
    ``q`` is the coordinate's pending position -- the count of coordinates
    stored so far, which the dense position resolve already multiplies.
    Below the block is either the value array or a compressed level K.

    The subtree wrote exactly when the operands store something under the
    coordinate, because the leaf writes only for stored structure:
    structure, never value.  Written means the value array grew past the
    block's start, or K appended a coordinate since the block opened.
    ``K_pos[q * B]`` is the coordinate count at the block's start, because
    every earlier block was padded to its end when it committed.

    Committed: pad the block to its full width (zero values, or K's
    position entries closed at K's current count), append the coordinate,
    advance ``q``.  Not committed: nothing was appended, and K's position
    entries the visited cells closed are dropped again.
    """

    level_types = result_tensor_access.level_types()
    if level == 0 or level_types[level - 1] != LevelType.COMPRESSED:
        return []
    block_end = level
    while block_end < len(level_types) and level_types[block_end] == LevelType.DENSE:
        block_end += 1
    if block_end < len(level_types) and (
        level_types[block_end] != LevelType.COMPRESSED
    ):
        return []
    result_name = result_tensor_access.get_tensor().get_name()

    def int64(name: str) -> llir.Var:
        return llir.Var(name=name, type=llir.DataType.INT64)

    def storage(
        *references: Tuple[
            llir.ResultStorageArray, Optional[int], llir.ResultStorageDirection
        ]
    ) -> Optional[llir.ResultStorageMetadata]:
        return cin_lowerer._result_storage(result_tensor_access, *references)

    block: llir.Expr = int64(f"{result_name}{level}_size")
    for inner in range(level + 1, block_end):
        block = llir.Mul(left=block, right=int64(f"{result_name}{inner}_size"))
    parent_position = int64(f"p{result_name}{level - 1}")
    block_start = llir.Mul(left=parent_position, right=block)
    block_stop = llir.Mul(left=llir.Add(parent_position, llir.Literal(1)), right=block)

    rollback: List[llir.Stmt] = []
    if block_end == len(level_types):
        values_name = f"{result_name}_values"
        written: llir.Expr = llir.BinOp(
            op=">",
            left=llir.FunctionCall(name=f"{values_name}.size", args=[]),
            right=block_start,
        )
        written_marker = storage((RESULT_VALUES, None, STORAGE_READ))
        pad: List[llir.Stmt] = [
            llir.FunctionCallStmt(
                name=f"{values_name}.resize",
                args=[block_stop, llir.Literal(0)],
                result_storage=storage((RESULT_VALUES, None, STORAGE_WRITE)),
            )
        ]
    else:
        below = block_end
        pos_name = f"{result_name}{below}_pos"
        crd_size = llir.FunctionCall(name=f"{result_name}{below}_crd.size", args=[])
        pos_index = int64(f"{result_name}{below}_pos_index")
        written = llir.BinOp(
            op=">",
            left=crd_size,
            right=llir.ArrayAccess(
                array=llir.Var(name=pos_name, type=llir.DataType.STD_VECTOR_C_INT),
                index=block_start,
            ),
        )
        written_marker = storage(
            (RESULT_CRD, below, STORAGE_READ), (RESULT_POS, below, STORAGE_READ)
        )
        pad = [
            llir.ForLoop(
                init=None,
                cond=llir.BinOp(op="<", left=pos_index, right=block_stop),
                update=llir.Increment(var=pos_index),
                body=[
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=pos_name,
                                type=llir.DataType.STD_VECTOR_C_INT,
                            ),
                            index=llir.Add(pos_index, llir.Literal(1)),
                        ),
                        value=crd_size,
                        result_storage=storage(
                            (RESULT_POS, below, STORAGE_WRITE),
                            (RESULT_CRD, below, STORAGE_READ),
                        ),
                    )
                ],
            )
        ]
        rollback = [
            llir.FunctionCallStmt(
                name=f"{pos_name}.resize",
                args=[llir.Add(block_start, llir.Literal(1))],
                result_storage=storage((RESULT_POS, below, STORAGE_WRITE)),
            ),
            llir.Assign(var=pos_index, value=block_start),
        ]
    return [
        llir.BlankLine(),
        llir.Comment("Commit the compressed coordinate above a written dense block"),
        llir.IfThenElse(
            cond=written,
            then_body=[
                *pad,
                llir.FunctionCallStmt(
                    name=f"{result_name}{level - 1}_crd.push_back",
                    args=[coordinate],
                    result_storage=storage((RESULT_CRD, level - 1, STORAGE_WRITE)),
                ),
                llir.Increment(var=parent_position),
            ],
            else_body=rollback or None,
            result_storage=written_marker,
        ),
    ]


@dataclass(frozen=False)
class LatticePoint:
    """
    iterators: List of tensor accesses to actually loop over
    locators: List of tensor accesses to locate only
    """

    sparse_tensor_accesses: List[TensorAccess] = field(default_factory=list)
    dense_tensor_accesses: List[TensorAccess] = field(default_factory=list)
    iterators: Optional[List[ModeIterator]] = field(default_factory=list)
    dense_iterators: List[ModeIterator] = field(default_factory=list)
    child_lattice_points: List["LatticePoint"] = field(default_factory=list)
    parent_lattice_point: Optional["LatticePoint"] = None  # type: ignore
    index_var: Optional[IndexVar] = None
    index_var_llir: Optional[llir.Var] = None
    _lattice: Optional[IterationLattice] = None

    def __add__(self, other):
        if isinstance(other, LatticePoint):
            return LatticePoint(
                sparse_tensor_accesses=self.get_sparse_tensor_accesses()
                + other.get_sparse_tensor_accesses(),
                dense_tensor_accesses=self.get_dense_tensor_accesses()
                + other.get_dense_tensor_accesses(),
            )
        else:
            return self

    def __radd__(self, other):
        return self.__add__(other)

    def __hash__(self):
        return hash(tuple(self.get_sparse_tensor_accesses())) + hash(
            tuple(self.get_dense_tensor_accesses())
        )

    def set_index_var(self, index_var: IndexVar):
        self.index_var = index_var
        self.index_var_llir = llir.Var(
            name=f"{index_var.name}",
            type=llir.DataType.INT,
        )

    def get_index_var(self) -> IndexVar:
        if self.index_var is None:
            raise ScorchAssertionError(
                "Index var not set", code="iteration_lattice_state_missing"
            )
        return self.index_var

    def get_index_var_llir(self):
        if self.index_var_llir is None:
            raise ScorchAssertionError(
                "Index var LLIR not set", code="iteration_lattice_state_missing"
            )
        return self.index_var_llir

    @property
    def lattice(self) -> IterationLattice:
        assert self._lattice is not None, "Lattice not set"
        return self._lattice

    @lattice.setter
    def lattice(self, lattice: IterationLattice):
        self._lattice = lattice

    @property
    def cin_lowerer(self) -> CINLowerer:
        return self.lattice.cin_lowerer

    def set_index_var_and_gen_iterators(
        self, index_var: IndexVar
    ) -> List[ModeIterator]:
        self.set_index_var(index_var)

        if index_var.has_parent:
            index_var = index_var.parent

        self.iterators = [
            ModeIterator(
                tensor_access=ta,
                index_var=index_var,
            )
            for ta in self.get_sparse_tensor_accesses()
        ]

        self.dense_iterators = [
            ModeIterator(
                tensor_access=ta,
                index_var=index_var,
            )
            for ta in self.get_dense_tensor_accesses()
        ]

        return self.iterators

    def filter_and_set_children(
        self, lattice_points: Iterable[LatticePoint]
    ) -> List[LatticePoint]:
        self.child_lattice_points = [
            lp for lp in lattice_points if lp.is_child_of(self)
        ]
        # For each child, set the parent
        for child in self.child_lattice_points:
            child.parent_lattice_point = self
        return self.child_lattice_points

    def get_for_loop_condition(self, lattice: IterationLattice) -> llir.Expr:
        condition = None

        for it in self.get_iterators():
            this_condition = llir.BinOp(
                op="<",
                left=it.get_iterator_var_llir(),
                right=it.get_iterator_var_end_value_llir(),
            )
            if condition is None:
                condition = this_condition
            else:
                condition = llir.BinOp(op="&&", left=condition, right=this_condition)

        # if domain is dense
        if lattice.dense_index_var_llir:
            if lattice.dense_index_var_end_var_llir is None:
                raise ScorchAssertionError(
                    "Dense index var upper bound not set",
                    code="iteration_lattice_state_missing",
                )
            condition = llir.BinOp(
                op="<",
                left=lattice.dense_index_var_llir,
                right=lattice.dense_index_var_end_var_llir,
            )
        if condition is None:
            raise ScorchAssertionError(
                "Failed to generate while condition",
                code="iteration_lattice_state_missing",
            )
        return condition

    def get_while_condition(self, lattice: IterationLattice) -> llir.Expr:
        condition = None

        for it in self.get_iterators():
            this_condition = llir.BinOp(
                op="<",
                left=it.get_iterator_var_llir(),
                right=it.get_iterator_var_end_var_llir(),
            )
            if condition is None:
                condition = this_condition
            else:
                condition = llir.BinOp(op="&&", left=condition, right=this_condition)

        # if domain is dense
        if lattice.dense_index_var_llir:
            if lattice.dense_index_var_end_var_llir is None:
                raise ScorchAssertionError(
                    "Dense index var upper bound not set",
                    code="iteration_lattice_state_missing",
                )
            condition = llir.BinOp(
                op="<",
                left=lattice.dense_index_var_llir,
                right=lattice.dense_index_var_end_var_llir,
            )
        if condition is None:
            raise ScorchAssertionError(
                "Failed to generate while condition",
                code="iteration_lattice_state_missing",
            )
        return condition

    def get_single_iterator_advance_stmt(
        self, lattice: IterationLattice
    ) -> Union[llir.Increment, llir.Assign]:
        iterators = self.get_iterators()
        if not (
            len(iterators) == 1
            or (len(iterators) == 0 and lattice.dense_index_var_llir)
        ):
            raise ScorchAssertionError(
                "Expected only one iterator", code="iteration_lattice_state_missing"
            )

        if lattice.dense_index_var_llir:
            lattice_ivar = lattice.index_var
            if lattice_ivar.tile_size_var and lattice_ivar.is_outer:
                # k_out += k_tile_size;

                return llir.Assign(
                    var=lattice.dense_index_var_llir,
                    value=llir.Var(
                        name=lattice_ivar.tile_size_var.name,
                        type=llir.DataType.INT,
                    ),
                    op=llir.AssignOp.ADD_ASSIGN,
                )

            else:
                return llir.Increment(
                    var=lattice.dense_index_var_llir,
                )

        it: ModeIterator = iterators[0]
        next_level = it.level + 1
        tensor_var = it.tensor_var

        if (
            next_level < tensor_var.levels
            and it.level_type == LevelType.COORDINATE
            and tensor_var.get_level_types()[next_level] == LevelType.COORDINATE
        ):
            next_level_end_var = llir.Var(
                name=f"p{tensor_var.name}{next_level}_end",
                type=llir.DataType.INT,
            )

            return llir.Assign(
                var=it.get_iterator_var_llir(),
                value=next_level_end_var,
            )
        else:
            return llir.Increment(
                var=it.get_iterator_var_llir(),
            )

    def get_iterators_advance_stmts(
        self, lattice: IterationLattice
    ) -> Sequence[llir.Stmt]:
        stmts: List[llir.Stmt] = []

        iterators = self.get_iterators()

        # Control flows are symmetrical to the initialization of iterators
        if len(iterators) > 1 or lattice.dense_index_var_llir:
            if iterators or lattice.dense_index_var_llir:
                stmts.append(llir.BlankLine())
                stmts.append(llir.Comment("Advance iterators"))
            for it in iterators:
                if (
                    it.level_type == LevelType.COORDINATE
                    and it.level + 1 < it.tensor_var.levels
                ):
                    # e.g. if k_A == k then pA0 = pA1_end;
                    stmts.append(
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op="==",
                                left=it.get_coord_var_llir(),
                                right=self.get_index_var_llir(),
                            ),
                            then_body=[
                                llir.Assign(
                                    var=it.get_iterator_var_llir(),
                                    value=llir.Var(
                                        name=f"p{it.tensor_var.name}{it.level + 1}_end",
                                        type=llir.DataType.INT,
                                    ),
                                )
                            ],
                        )
                    )
                else:
                    # e.g. pA0 += (int)k_A == k;
                    stmts.append(
                        llir.Assign(
                            var=it.get_iterator_var_llir(),
                            value=llir.BinOp(
                                op="==",
                                left=it.get_coord_var_llir(),
                                right=self.get_index_var_llir(),
                            ),
                            op=llir.AssignOp.ADD_ASSIGN,
                            cast=True,
                        )
                    )
            if lattice.dense_index_var_llir:
                lattice_ivar = lattice.index_var
                if lattice_ivar.tile_size_var and lattice_ivar.is_outer:
                    # k_out += k_tile_size;
                    stmts.append(
                        llir.Assign(
                            var=lattice.dense_index_var_llir,
                            value=llir.Var(
                                name=lattice_ivar.tile_size_var.name,
                                type=llir.DataType.INT,
                            ),
                            op=llir.AssignOp.ADD_ASSIGN,
                        )
                    )
                    # stmts.append(
                    #     llir.Assign(
                    #         var=lattice.dense_index_var_llir,
                    #         value=llir.BinOp(
                    #             op="+",
                    #             left=lattice.dense_index_var_llir,
                    #             right=llir.Var(
                    #                 name=lattice_ivar.tile_size_var.name,
                    #                 type=llir.DataType.INT,
                    #             ),
                    #         ),
                    #     )
                    # )
                else:
                    stmts.append(
                        llir.Increment(
                            var=lattice.dense_index_var_llir,
                        )
                    )

        elif len(iterators) == 1:
            stmts.append(llir.Comment("Advance iterator"))

            # If this _level is coordinate and next _level is coordinate
            # then advance by setting
            # p{tensor_name}{current_level}
            # to p{tensor_name}{next_level}_end
            iterator: ModeIterator = iterators[0]
            next_level = iterator.level + 1
            tensor_var = iterator.tensor_var
            if tensor_var is None:
                raise ScorchAssertionError(
                    "Tensor var not set", code="iteration_lattice_state_missing"
                )

            if (
                next_level < tensor_var.levels
                and iterator.level_type == LevelType.COORDINATE
                and tensor_var.get_level_types()[next_level] == LevelType.COORDINATE
            ):
                next_level_end_var = llir.Var(
                    name=f"p{tensor_var.name}{next_level}_end",
                    type=llir.DataType.INT,
                )

                stmts.append(
                    llir.Assign(
                        var=iterator.get_iterator_var_llir(),
                        value=next_level_end_var,
                    )
                )
            else:
                stmts.append(
                    llir.Increment(
                        var=iterators[0].get_iterator_var_llir(),
                    )
                )

        return stmts

    def get_iterators(self) -> Sequence[ModeIterator]:
        if self.iterators is None:
            raise ScorchAssertionError(
                "Iterators not set", code="iteration_lattice_state_missing"
            )
        return self.iterators

    def get_sparse_tensor_accesses(self) -> List[TensorAccess]:
        if self.sparse_tensor_accesses is None:
            raise ScorchAssertionError(
                "Sparse tensor accesses not set", code="iteration_lattice_state_missing"
            )
        return self.sparse_tensor_accesses

    def get_dense_tensor_accesses(self) -> List[TensorAccess]:
        if self.dense_tensor_accesses is None:
            raise ScorchAssertionError(
                "Dense tensor accesses not set", code="iteration_lattice_state_missing"
            )
        return self.dense_tensor_accesses

    def get_simplified_cin(self, cin: CIN) -> Optional[CIN]:
        # Rewrite the CIN to eliminate tensors that have run out of values
        # Based on the lattice_point we are currently at
        if isinstance(cin, TensorAccess):
            if (
                cin in self.get_sparse_tensor_accesses()
                or cin in self.get_dense_tensor_accesses()
            ):
                return cin
            else:
                return None

        elif isinstance(cin, BinaryOp):
            left_new = self.get_simplified_cin(cin.left)
            right_new = self.get_simplified_cin(cin.right)

            if left_new is None and right_new is None:
                return None

            if left_new and right_new:
                if not (
                    isinstance(left_new, IndexExpr) and isinstance(right_new, IndexExpr)
                ):
                    raise ScorchAssertionError(
                        "Expected IndexExpr for left and right",
                        code="iteration_lattice_state_missing",
                    )
                return BinaryOp(
                    op=cin.op,
                    left=left_new,
                    right=right_new,
                )

            # At this point, one of left_new or right_new is None

            if cin.op == Operation.ADD:
                return left_new or right_new
            elif cin.op == Operation.MUL:
                return None

        elif isinstance(cin, ForAll):
            rewritten_inner_stmt = self.get_simplified_cin(cin.stmt)
            if not (isinstance(rewritten_inner_stmt, IndexStmt)):
                raise ScorchAssertionError(
                    "Rewritten inner stmt is not an index stmt",
                    code="iteration_lattice_state_missing",
                )
            return ForAll(
                index_var=cin.index_var,
                stmt=rewritten_inner_stmt,
            )

        elif isinstance(cin, Where):
            producer_rewritten = self.get_simplified_cin(cin.producer)
            consumer_rewritten = self.get_simplified_cin(cin.consumer)
            if not (
                isinstance(producer_rewritten, IndexStmt)
                and isinstance(consumer_rewritten, IndexStmt)
            ):
                raise ScorchAssertionError(
                    "Rewritten producer and consumer are not index stmts",
                    code="iteration_lattice_state_missing",
                )
            return Where(
                producer=producer_rewritten,
                consumer=consumer_rewritten,
            )

        elif isinstance(cin, TensorAssign):
            cin_ivar_getter = CINIndexVariablesGetter()
            cin_ivar_getter.visit(cin)

            reduction_vars = cin_ivar_getter.get_reduction_vars()

            # print("Free vars: ", cin_ivar_getter.free_vars)
            # print("Input vars: ", cin_ivar_getter.input_vars)
            # print("Reduction vars: ", reduction_vars)

            has_reduction = len(reduction_vars) > 0

            rewritten_rhs = self.get_simplified_cin(cin.rhs)
            if not rewritten_rhs:
                rewritten_rhs = cin.rhs
            # print("RHS: ", cin.rhs)
            # print("Rewritten RHS: ", rewritten_rhs)
            if not (isinstance(rewritten_rhs, IndexExpr)):
                raise ScorchAssertionError(
                    "Rewritten RHS is not an IndexExpr",
                    code="iteration_lattice_state_missing",
                )

            return TensorAssign(
                lhs=cin.lhs,
                rhs=rewritten_rhs,
                op=Operation.ADD if has_reduction else None,
            )

        raise ScorchNotImplementedError(
            f"Unhandled CIN type {type(cin)}", code="iteration_lattice_cin_unsupported"
        )

    def get_child_subregion_loops(
        self, cin_lowerer: CINLowerer, cin: IndexStmt, lattice: "IterationLattice"
    ) -> Sequence[llir.Stmt]:
        """
        Iterates over the child lattice points and generate an inner loop over each
        case. The inner loop uses a simplified/rewritten expression of the original
        CIN to ignore the tensors that have run out of values.
        """
        stmts: List[llir.Stmt] = []

        # print("\n========== get_child_subregion_loops =========")
        # print("cin: ", cin)
        # print("self.child_lattice_points:", self.child_lattice_points)

        # print("self.iterators:", self.iterators)

        def lower_cin_and_add_to_list(
            lattice_point: LatticePoint, cin: IndexStmt, lst: List[llir.Stmt]
        ):
            simplified_cin = lattice_point.get_simplified_cin(cin)
            if simplified_cin is None:
                raise ScorchAssertionError(
                    "Simplified CIN is None", code="iteration_lattice_state_missing"
                )
            if not (isinstance(simplified_cin, IndexStmt)):
                raise ScorchAssertionError(
                    "Simplified CIN is not an index stmt",
                    code="iteration_lattice_state_missing",
                )
            cin_lowered = cin_lowerer.lower_IndexStmt(simplified_cin)
            if isinstance(cin_lowered, llir.Stmt):
                lst.append(cin_lowered)
            elif isinstance(cin_lowered, list):
                lst.extend(cin_lowered)

        all_inner_lattice_points = [self]
        if self.child_lattice_points:
            all_inner_lattice_points.extend(self.child_lattice_points)

        if self.child_lattice_points or (self.iterators and len(self.iterators) > 1):
            dense_universe = lattice.dense_index_var_llir is not None
            stmts.append(llir.Comment("Inner loops over child regions"))
            if_conditions: List[llir.Expr] = []
            then_body_list = []
            else_body: List[llir.Stmt] = []

            for child_lp in all_inner_lattice_points:
                if_condition = None
                then_body: List[llir.Stmt] = []

                for iterator in child_lp.get_iterators():
                    this_condition: llir.Expr = llir.BinOp(
                        op="==",
                        left=iterator.get_coord_var_llir(),
                        right=self.get_index_var_llir(),
                    )
                    if dense_universe:
                        # A dense universe drives the loop by the dense
                        # extent, so it never exhausts the sparse cursors:
                        # once a segment is drained the cursor still points
                        # at the NEXT segment's first stored coordinate, and
                        # a bare coordinate equality can spuriously match it
                        # (a silently wrong sum) while the coordinate load
                        # itself reads past the segment.  Bound the cursor by
                        # its own segment end before comparing.
                        this_condition = llir.BinOp(
                            op="&&",
                            left=llir.BinOp(
                                op="<",
                                left=iterator.get_iterator_var_llir(),
                                right=iterator.get_iterator_var_end_var_llir(),
                            ),
                            right=this_condition,
                        )
                    if if_condition is None:
                        if_condition = this_condition
                    else:
                        if_condition = llir.BinOp(
                            op="&&", left=if_condition, right=this_condition
                        )

                # If this lattice point has no (sparse) iterators, then we
                # are at the dense domain, set the would-be then_body as the else_body
                if not child_lp.iterators:
                    lower_cin_and_add_to_list(child_lp, cin, else_body)
                else:
                    lower_cin_and_add_to_list(child_lp, cin, then_body)
                    if if_condition is None:
                        raise ScorchAssertionError(
                            "if_condition cannot be None",
                            code="iteration_lattice_state_missing",
                        )
                    if_conditions.append(if_condition)
                    then_body_list.append(then_body)

            stmts.append(
                llir.IfThenElse(
                    cond_list=if_conditions,
                    then_body_list=then_body_list,
                    else_body=else_body,
                )
            )
        else:
            lower_cin_and_add_to_list(self, cin, stmts)

        return stmts

    # noinspection GrazieInspection
    def get_candidate_coordinate_stmts(
        self, lattice: IterationLattice
    ) -> Sequence[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        iterators = self.get_iterators()

        if len(iterators) > 1 or lattice.dense_index_var_llir:
            # If we have a dense universe, then we still need to resolve
            # the sparse coordinate even if we have a single iterator
            if iterators:
                stmts.append(llir.Comment("Load coordinates"))
                for it in iterators:
                    value = it.get_coord_var_value_llir()
                    if lattice.dense_index_var_llir is not None:
                        # A dense universe drives the loop by the dense
                        # extent, so a sparse cursor can be past its own
                        # segment end here -- including on the very first
                        # iteration when the segment is empty, where the
                        # coordinate array may hold nothing at all.  Reading
                        # it unconditionally is an out-of-bounds load (an
                        # all-zero sparse operand added to a dense one
                        # terminated the process).  Select the stored
                        # coordinate only while the cursor is live; the
                        # value chosen otherwise is never observed, because
                        # every consuming case is itself guarded by the same
                        # bound.
                        value = llir.Select(
                            cond=llir.BinOp(
                                op="<",
                                left=it.get_iterator_var_llir(),
                                right=it.get_iterator_var_end_var_llir(),
                            ),
                            when_true=value,
                            when_false=llir.Literal(0, llir.DataType.INT),
                        )
                    stmts.append(
                        llir.VarInit(
                            var=it.get_coord_var_llir(),
                            value=value,
                        )
                    )
            # Only need to break ties among the resolved sparse coordinates
            # if we don't have a dense domain
            if not lattice.dense_index_var_llir:
                stmts.append(llir.BlankLine())
                stmts.append(llir.Comment("Resolve coordinates"))
                stmts.append(
                    llir.VarInit(
                        var=self.get_index_var_llir(),
                        value=llir.FunctionCall(
                            name="std::min",
                            args=[
                                llir.Array(
                                    values=[
                                        it.get_coord_var_llir() for it in iterators
                                    ],
                                    data_type=llir.DataType.INT,
                                )
                            ],
                        ),
                    )
                )
                stmts.append(llir.BlankLine())

        elif len(iterators) == 1:
            stmts.append(llir.Comment("Resolve coordinates"))
            stmts.append(
                llir.VarInit(
                    var=self.get_index_var_llir(),
                    value=iterators[0].get_coord_var_value_llir(),
                )
            )
            stmts.append(llir.BlankLine())

        result_tensor_access = lattice.cin_lowerer.result_tensor_access
        index_var = self.get_index_var()

        if result_tensor_access.has_index_var(index_var):
            child_level = result_tensor_access.level_of_index_var(index_var) + 1
            curr_level_type = result_tensor_access.level_type_of_index_var(index_var)
            if (
                curr_level_type == LevelType.DENSE
                and lattice.cin_lowerer is not None
                and not result_tensor_access.is_workspace()
            ):
                if lattice.dense_index_var_llir is None and all(
                    above == LevelType.DENSE
                    for above in result_tensor_access.level_types()[:child_level]
                ):
                    # This loop visits only the coordinates its operands
                    # store, so positions of this dense result level can go
                    # unvisited -- the ones after the last visited coordinate
                    # among them, which no later catch-up reaches.  See
                    # ``emit_position_tail_close``.  Only a level of the dense
                    # prefix above the first compressed level owes it: below a
                    # compressed level, the commit pads each block to its end.
                    lattice.cin_lowerer.result_dense_level_sparse_driven = True
                elif lattice.dense_index_var_end_var_llir is not None:
                    # The extent this level's loop actually runs to, by the
                    # name the loop header uses, for every expression that
                    # counts this level's cells.
                    lattice.cin_lowerer.result_dense_level_bound_names[
                        child_level - 1
                    ] = lattice.dense_index_var_end_var_llir.name
            child_level_type = result_tensor_access.child_level_type_of_index_var(
                index_var
            )
            if (
                curr_level_type == LevelType.DENSE
                and child_level_type == LevelType.COMPRESSED
                and (
                    child_level == 1
                    or lattice.dense_index_var_llir is not None
                    or result_tensor_access.is_workspace()
                    or (index_var.tile_size_var and index_var.is_outer)
                    or result_tensor_access in self.get_dense_tensor_accesses()
                )
            ):
                # A loop over STORED coordinates of a dense level below another
                # level is caught up after its position is resolved, by
                # ``get_lattice_loops``.  Every other loop is caught up here,
                # before its body: level 0, whose position is its coordinate;
                # a dense loop, whose coordinate is its loop variable; and the
                # loops that resolve no result position of their own.
                stmts.extend(
                    compressed_child_catch_up(
                        lattice.cin_lowerer,
                        result_tensor_access,
                        child_level,
                        lattice.dense_level_position(
                            result_tensor_access, child_level - 1
                        ),
                    )
                )

        coordinate_level_iterator_end_resolution_stmts: List[llir.Stmt] = []
        for it in iterators:
            if it.level_type == LevelType.COORDINATE:
                if (it.level + 1) < it.tensor_var.levels:
                    next_level_iterator_end_llir = llir.Var(
                        name=f"p{it.tensor_var.name}{it.level + 1}_end",
                        type=llir.DataType.INT,
                    )

                    if next_level_iterator_end_llir is None:
                        raise ScorchAssertionError(
                            "next_level_iterator_end_llir cannot be None",
                            code="iteration_lattice_state_missing",
                        )
                    if it.iterator_var_llir is None:
                        raise ScorchAssertionError(
                            "it.iterator_var_llir cannot be None",
                            code="iteration_lattice_state_missing",
                        )
                    coordinate_level_iterator_end_resolution_stmts.append(
                        llir.Assign(
                            var=next_level_iterator_end_llir,
                            value=llir.BinOp(
                                op="+",
                                left=it.iterator_var_llir,
                                right=llir.Literal(value=1),
                            ),
                        )
                    )

                    if it.iterator_var_end_var_llir is None:
                        raise ScorchAssertionError(
                            "it.iterator_var_end_var_llir cannot be None",
                            code="iteration_lattice_state_missing",
                        )

                    coordinate_level_iterator_end_resolution_stmts.append(
                        llir.WhileLoop(
                            cond=llir.BinOp(
                                op="&&",
                                left=llir.BinOp(
                                    op="<",
                                    left=next_level_iterator_end_llir,
                                    right=it.iterator_var_end_var_llir,
                                ),
                                right=llir.BinOp(
                                    op="==",
                                    left=llir.ArrayAccess(
                                        array=llir.Var(
                                            name=f"{it.tensor_var.name}{it.level}_crd",
                                            type=llir.DataType.ARRAY_INT,
                                        ),
                                        index=next_level_iterator_end_llir,
                                    ),
                                    right=self.get_index_var_llir(),
                                ),
                            ),
                            body=[
                                llir.Increment(
                                    var=next_level_iterator_end_llir,
                                )
                            ],
                        )
                    )
                    coordinate_level_iterator_end_resolution_stmts.append(
                        llir.BlankLine()
                    )

        if coordinate_level_iterator_end_resolution_stmts:
            stmts.append(llir.Comment("Find iterator end for coordinate level"))
            stmts.extend(coordinate_level_iterator_end_resolution_stmts)

        cin_lowerer = lattice.cin_lowerer
        if cin_lowerer is None:
            raise ScorchAssertionError(
                "CIN lowerer is None", code="iteration_lattice_state_missing"
            )

        dense_iterators_resolve_stmts: List[llir.Stmt] = []
        d = cin_lowerer.dense_coord_resolve_stmt_to_dep_index_vars

        if self.dense_iterators:
            for it in self.dense_iterators:
                if it.coord_var_value_llir:
                    dense_coord_resolve_stmt = llir.VarInit(
                        var=it.get_coord_var_llir(),
                        value=it.get_coord_var_value_llir(),
                    )

                    d_stmts: List[llir.VarInit] = list(d.keys())
                    to_resolve_index_var_names = [stmt.var.name for stmt in d_stmts]

                    if (
                        dense_coord_resolve_stmt.var.name
                        not in to_resolve_index_var_names
                    ):
                        d[dense_coord_resolve_stmt] = it.coord_var_value_depends_on

        defined_index_vars = set(cin_lowerer.defined_index_vars)

        for stmt in _declarable_now(d, defined_index_vars):
            dense_iterators_resolve_stmts.append(stmt)
            del d[stmt]

        if dense_iterators_resolve_stmts:
            stmts.append(llir.Comment("Resolve dense coordinates"))
            stmts.extend(_parents_first(dense_iterators_resolve_stmts))

        return stmts

    def is_child_of(self, other: LatticePoint) -> bool:
        """
        A lattice_point is a child of another lattice point if its sparse tensor accesses is a
        strict subset of the parent's
        """
        return set(self.get_sparse_tensor_accesses()).issubset(
            set(other.get_sparse_tensor_accesses())
        )


def mark_simd_on_reduction_loops(stmts: Sequence[llir.Stmt]) -> None:
    """Mark the loops of a sum kept in a register for vector instructions.

    ``stmts`` is the statement list that accumulates one result entry into a
    scalar (the scalar-accumulation subregion of this lowering, the reduction
    sub-nest of the new compiler's bound-prefix family).  A ``for`` loop directly
    in it is marked ``#pragma omp simd`` when its body holds no loop over a
    compressed level's positions, it is not parallel or unrolled, and nothing in
    it breaks out.  One definition, read by both compilers, so the two mark the
    same loops.
    """

    from .parallel_marking_pass import has_sparse_inner_loop

    def contains_break(body: Sequence[llir.Stmt]) -> bool:
        for child in body:
            if isinstance(child, llir.Break):
                return True
            if isinstance(child, (llir.ForLoop, llir.WhileLoop)):
                if contains_break(child.body):
                    return True
            elif isinstance(child, llir.IfThenElse):
                branches = [child.then_body, child.else_body]
                branches.extend(child.then_body_list or [])
                if any(branch and contains_break(branch) for branch in branches):
                    return True
        return False

    for stmt in stmts:
        if isinstance(stmt, llir.ForLoop):
            if (
                not has_sparse_inner_loop(stmt.body)
                and not stmt.omp_parallel_for
                and not stmt.unroll
                and not contains_break(stmt.body)
            ):
                stmt.simd = True


class _VectorMarkedLoops(LLIRWalker):
    """Every ``for`` loop carrying a vector mark, in emission order."""

    def __init__(self) -> None:
        super().__init__(
            LLIRTraversalContext(
                stage="schedule lowering", pass_name="select_vector_marks"
            )
        )
        self.loops: List[llir.ForLoop] = []

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        if type(node) is llir.ForLoop and node.simd:
            self.loops.append(node)


def select_vector_marks(
    stmts: Sequence[llir.Stmt], requested: Optional[Sequence[str]]
) -> Tuple[str, ...]:
    """Keep a vector mark only on the loops a schedule names.

    ``requested`` names loops by their logical-loop identity
    (``ForLoop.scorch_index_var``: an index name, or ``{name}_out`` and
    ``{name}_in`` for the two loops of an affine split).  ``None`` requests
    nothing and keeps every mark :func:`mark_simd_on_reduction_loops` placed.
    Otherwise a marked loop whose identity is not requested loses its mark, and
    the requested names that match no marked loop are returned, in request
    order, for the caller to refuse: a schedule keeps or drops the marks the
    rule places, and cannot mark a loop the rule does not find safe to mark.
    Both compilers call this on their lowered kernel, so one request leaves the
    same loops marked in both.
    """

    if requested is None:
        return ()
    collector = _VectorMarkedLoops()
    collector.walk(list(stmts))
    wanted = set(requested)
    matched: Set[str] = set()
    for loop in collector.loops:
        identity = loop.scorch_index_var
        if identity is not None and identity in wanted:
            matched.add(identity)
        else:
            loop.simd = False
    return tuple(name for name in requested if name not in matched)


def _names_read(expr: object) -> List[str]:
    """The name of every variable an LLIR expression reads."""

    if isinstance(expr, llir.Var):
        return [expr.name]
    if not is_dataclass(expr):
        return []
    names: List[str] = []
    for part in fields(expr):
        names.extend(_names_read(getattr(expr, part.name)))
    return names


def _declarable_now(
    pending: Dict[llir.VarInit, List[IndexVar]], bound: Set[IndexVar]
) -> List[llir.VarInit]:
    """The pending dense position declarations that can be emitted here, in table order.

    A dense level's position is its parent's position times its extent plus
    its coordinate.  Its declaration lists two index variables, its own and
    its parent's, and used to be emitted as soon as both were bound.  But it
    reads the parent's *position*, and when the parent level is dense as well,
    that position has a declaration of its own, which waits for the
    grandparent's index variable.  A dense ``A`` stored ``(k, i, j)`` under
    loops ``j, i, k`` got ``int pA2 = pA1 * A2_size + j;`` in the loop over
    ``i`` and ``int pA1 = pA0 * A1_size + i;`` in the loop over ``k``, which
    does not compile.  Two dense levels never showed it: the list of the
    second names every index variable there is.

    So a declaration whose index variables are bound still waits while a
    position it reads has a pending declaration that cannot be emitted here.
    A position with no pending declaration has been declared by an earlier
    release, or is the iterator of an enclosing loop over a compressed or
    coordinate level, which never enters the table.  A declaration that reads
    no waiting position is emitted where it always was, so a kernel whose
    declarations were already in a valid order is emitted as before.
    """

    ready = [stmt for stmt, needed in pending.items() if set(needed).issubset(bound)]
    waiting = {stmt.var.name for stmt in pending} - {stmt.var.name for stmt in ready}
    while True:
        held = {
            stmt.var.name
            for stmt in ready
            if not waiting.isdisjoint(_names_read(stmt.value))
        }
        if not held:
            return ready
        waiting |= held
        ready = [stmt for stmt in ready if stmt.var.name not in held]


def _parents_first(resolve_stmts: List[llir.Stmt]) -> List[llir.Stmt]:
    """Dense position declarations, each after the one whose variable it reads.

    A dense level's position is its parent's position times its extent plus
    its coordinate, so its declaration reads the parent's.  The declarations
    wait until the loops have bound their index variables and are then emitted
    in the order they were first met.  A loop order that binds a tensor's
    levels inner level first -- ``B[a,c]`` under loops ``c, a`` -- meets the
    inner level's declaration first, and the two came out in that order:
    ``int pB1 = pB0 * B1_size + c; int pB0 = a;``, which does not compile.

    The order is otherwise kept, so a kernel whose declarations were already
    in a valid order is emitted as before.
    """

    declared = {
        stmt.var.name: stmt for stmt in resolve_stmts if isinstance(stmt, llir.VarInit)
    }
    ordered: List[llir.Stmt] = []
    placed: set = set()

    def place(stmt: llir.Stmt) -> None:
        if id(stmt) in placed:
            return
        placed.add(id(stmt))
        if isinstance(stmt, llir.VarInit):
            for name in _names_read(stmt.value):
                parent = declared.get(name)
                if parent is not None and parent is not stmt:
                    place(parent)
        ordered.append(stmt)

    for stmt in resolve_stmts:
        place(stmt)
    return ordered


@dataclass(frozen=False)
class IterationLattice:
    """
    The iteration lattice of an iteration domain contains an ordered set of
    lattice points, in decreasing order of the number of index variables they
    contain.

    """

    for_all_stmt: ForAll
    lattice_points: Optional[List[LatticePoint]] = None
    parent_to_children_lattice_points: Optional[
        Dict[LatticePoint, List[LatticePoint]]
    ] = None
    index_var: Optional[IndexVar] = None
    cin_lowerer: Optional[CINLowerer] = None

    # these are only set if the iteration domain is the universe
    dense_index_var: Optional[IndexVar] = None
    dense_index_var_llir: Optional[llir.Var] = None
    dense_index_var_end_var_llir: Optional[llir.Var] = None

    def gen_lattice_points(self) -> List[LatticePoint]:
        """
        Generate the lattice points for the iteration lattice of the given
        iteration domain.
        """
        curr_index_var = self.for_all_stmt.get_index_var()
        parsed_index_var = curr_index_var
        if parsed_index_var.has_parent:
            parsed_index_var = parsed_index_var.parent

        def union_lattice_points(
            left_lattice_points: List[LatticePoint],
            right_lattice_points: List[LatticePoint],
        ) -> List[LatticePoint]:
            return [
                *left_lattice_points,
                *right_lattice_points,
                *intersect_lattice_points(left_lattice_points, right_lattice_points),
            ]

        def intersect_lattice_points(
            left_lattice_points: List[LatticePoint],
            right_lattice_points: List[LatticePoint],
        ) -> List[LatticePoint]:
            # TODO: double check this
            if not right_lattice_points:
                return left_lattice_points
            if not left_lattice_points:
                return right_lattice_points
            return [
                *map(sum, product(left_lattice_points, right_lattice_points)),  # type: ignore
            ]

        def get_lattice_points_from_cin(cin: CIN) -> List[LatticePoint]:
            if isinstance(cin, Where):
                return get_lattice_points_from_cin(cin.producer)
            if isinstance(cin, ForAll):
                return get_lattice_points_from_cin(cin.stmt)
            if isinstance(cin, TensorAssign):
                return get_lattice_points_from_cin(cin.rhs)
            if isinstance(cin, BinaryOp):
                left_lattice_points: List[LatticePoint] = get_lattice_points_from_cin(
                    cin.left
                )
                right_lattice_points: List[LatticePoint] = get_lattice_points_from_cin(
                    cin.right
                )
                op = cin.op
                if op == Operation.MUL:
                    return intersect_lattice_points(
                        left_lattice_points, right_lattice_points
                    )
                elif op == Operation.ADD:
                    if bool(left_lattice_points) != bool(right_lattice_points):
                        # One addend has this loop's variable and the other
                        # has no access with it.  The addend without it is
                        # defined at every coordinate of the loop, so the sum
                        # is too, but the union below would visit only what
                        # the other addend stores, and ``get_simplified_cin``
                        # would then drop the addend from the body as one that
                        # had run out of values: ``A[i,j] + x[j]`` returned
                        # ``A``.  A loop over the whole extent for such an
                        # addend is not written.
                        diagnostic = CINDiagnostic(
                            code="unsupported_sum_addend_without_loop_variable",
                            message=(
                                "one side of a sum has no tensor indexed by "
                                f"{parsed_index_var.name!r} and the other side "
                                "has; the sum is defined over the whole extent "
                                f"of {parsed_index_var.name!r}, which this "
                                "lowering does not iterate"
                            ),
                            path=("root",),
                            stage="legacy_cin_lowering",
                            pass_name="gen_lattice_points",
                        )
                        raise UnsupportedFeature(
                            "stage=legacy CIN lowering: "
                            "unsupported_sum_addend_without_loop_variable "
                            f"(loop {parsed_index_var.name!r})",
                            diagnostics=(diagnostic,),
                        )
                    return union_lattice_points(
                        left_lattice_points, right_lattice_points
                    )
            if isinstance(cin, TensorAccess):
                # TODO: check Empty list if current_index_var is not in the tensor access
                if parsed_index_var not in cin.indices:
                    return []
                # if index variable correspond to a dense level, put in locators
                if (
                    cin.get_tensor().get_level_types()[
                        cin.get_sorted_index_vars().index(parsed_index_var)
                    ]
                    == LevelType.DENSE
                ):
                    return [LatticePoint(dense_tensor_accesses=[cin])]
                return [LatticePoint(sparse_tensor_accesses=[cin])]
            raise ScorchNotImplementedError(
                f"Unsupported CIN: {cin}", code="iteration_lattice_cin_unsupported"
            )

        lattice_points: List[LatticePoint] = get_lattice_points_from_cin(
            self.for_all_stmt
        )

        # Broadcast dimension: index var appears only in the result tensor,
        # not in any RHS tensor.  Generate a dense lattice point using the
        # result tensor access so the loop iterates over the full range.
        if (
            not lattice_points
            and self.cin_lowerer
            and self.cin_lowerer.result_tensor_access
            and self.cin_lowerer.result_tensor_access.has_index_var(parsed_index_var)
        ):
            lattice_points = [
                LatticePoint(
                    dense_tensor_accesses=[self.cin_lowerer.result_tensor_access]
                )
            ]

        # Set the lattice pointer for each lattice point
        for lattice_point in lattice_points:
            lattice_point.lattice = self

        # Sort lattice_points in descending number of iterators, then descending number of locators
        lattice_points.sort(
            key=lambda lp: (
                len(lp.get_sparse_tensor_accesses()),
                len(lp.get_dense_tensor_accesses()),
            ),
            reverse=True,
        )

        # Remove lattice points that have the same iterators, only keep the one with the most locators
        # TODO: can be optimized
        lattice_points = [
            lattice_point
            for i, lattice_point in enumerate(lattice_points)
            if lattice_point.sparse_tensor_accesses
            not in [lp.sparse_tensor_accesses for lp in lattice_points[:i]]
        ]

        # If any of the lattice points have no iterators, then the iteration domain is the universe
        if any([not lp.sparse_tensor_accesses for lp in lattice_points]):
            self.dense_index_var = curr_index_var
            if not self.cin_lowerer:
                raise ScorchAssertionError(
                    "cin_lowerer must be set", code="iteration_lattice_state_missing"
                )
            self.dense_index_var_llir = self.cin_lowerer.lower_IndexVar(curr_index_var)
            # first get the lattice point with no iterators
            dense_lattice_point = next(
                lp for lp in lattice_points if not lp.sparse_tensor_accesses
            )
            # then pick any of the dense tensor accesses in the lattice point
            # to get the end variable
            first_dense_tensor_access = dense_lattice_point.get_dense_tensor_accesses()[
                0
            ]
            first_dense_tensor_var = first_dense_tensor_access.get_tensor()
            level_of_current_index_var = first_dense_tensor_access.level_of_index_var(
                parsed_index_var
            )
            # end var is the <tensor_var_name><_level>_size, type is int
            if curr_index_var.tile_size_var and curr_index_var.is_inner:
                self.dense_index_var_end_var_llir = llir.Var(
                    name=curr_index_var.tile_size_var.name,
                    type=llir.DataType.INT,
                )
            else:
                self.dense_index_var_end_var_llir = llir.Var(
                    name=f"{first_dense_tensor_var.name}{level_of_current_index_var}_size",
                    type=llir.DataType.INT,
                )

        return lattice_points

    def get_lattice_points(self) -> Sequence[LatticePoint]:
        if self.lattice_points is None:
            raise ScorchAssertionError(
                "Lattice points not generated", code="iteration_lattice_state_missing"
            )
        return self.lattice_points

    def gen_parent_to_children_lattice_points(
        self,
    ) -> Dict[LatticePoint, List[LatticePoint]]:
        """
        Generate the parent to children lattice points mapping for the iteration lattice of the given
        iteration domain.

        """
        lattice_points = self.get_lattice_points()
        parent_to_children_lattice_points = {}
        for i, lattice_point in enumerate(lattice_points):
            children = lattice_point.filter_and_set_children(lattice_points[i + 1 :])
            parent_to_children_lattice_points[lattice_point] = children
            # parent_to_children_lattice_points[lattice_point] = [
            #     lp for lp in lattice_points[i + 1 :] if lp.is_child_of(lattice_point)
            # ]
        self.parent_to_children_lattice_points = parent_to_children_lattice_points
        return parent_to_children_lattice_points

    def __post_init__(self):
        if self.lattice_points is None:
            self.lattice_points = self.gen_lattice_points()
            self.gen_parent_to_children_lattice_points()

        if self.index_var is None:
            # Get the index variable from the ForAll statement
            self.index_var = self.for_all_stmt.get_index_var()

        # Generate the iterators for each lattice point now that we know the index variable
        for lp in self.lattice_points:
            lp.set_index_var_and_gen_iterators(self.index_var)

    def get_iterator_init_stmts(self) -> List[llir.Stmt]:
        """
        Generate the iterator initialization statements for the given iteration domain.

        """
        # We can just use the first lattice point to determine what to initialize
        # since the list is sorted by number of iterators
        # DONE: handle dense iterators
        iterator_init_stmts: List[llir.Stmt] = []
        lattice_points = self.get_lattice_points()
        if not (len(lattice_points) > 0):
            raise ScorchAssertionError(
                "No lattice points generated", code="iteration_lattice_state_missing"
            )
        all_mode_iterators = lattice_points[0].get_iterators()

        # if self.dense_index_var_llir:
        #     iterator_init_stmts.append(llir.Comment("Initialize dense index variable"))
        #     iterator_init_stmts.append(
        #         llir.VarInit(
        #             var=self.dense_index_var_llir,
        #             value=llir.Literal(value=0, data_type=llir.DataType.INT),
        #         )
        #     )

        if len(all_mode_iterators) > 1:
            iterator_init_stmts.extend(
                [*[mode_iterator.get_init_stmts() for mode_iterator in all_mode_iterators]]  # type: ignore
            )
            if self.dense_index_var_llir is not None:
                # Several iterators merged under a dense extent lower to a
                # ``while`` loop that advances the dense variable together with
                # the iterators (``get_iterators_advance_stmts``).  A ``for``
                # loop declares that variable in its header; a ``while`` loop
                # has to be given it, or the kernel names a variable nothing
                # declares (a sum of two compressed operands and a dense one).
                iterator_init_stmts.append(self.get_dense_iterator_init_stmt())

        if len(all_mode_iterators) == 1:
            iterator = all_mode_iterators[0]
            iterator_init_stmts.extend(iterator.get_iterator_end_init_stmts())

        if iterator_init_stmts:
            iterator_init_stmts = [
                llir.Comment("Initialize iterators"),
                *iterator_init_stmts,
            ]

        return iterator_init_stmts

    def get_dense_iterator_init_stmt(self) -> llir.VarInit:
        if self.dense_index_var_llir is None:
            raise ScorchAssertionError(
                "Dense index var not set", code="iteration_lattice_state_missing"
            )
        return llir.VarInit(
            var=self.dense_index_var_llir,
            value=llir.Literal(value=0, data_type=llir.DataType.INT),
        )

    def _tile_bound_var(self, parent_index_var: IndexVar) -> str | None:
        """Return the dimension-size variable name that bounds *parent_index_var*
        in the first dense tensor access that contains it, or None."""
        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "self.cin_lowerer is not None", code="iteration_lattice_state_missing"
            )
        lattice_points = self.get_lattice_points()
        for lp in lattice_points:
            for ta in lp.get_dense_tensor_accesses():
                if ta.has_index_var(parent_index_var):
                    tv = ta.get_tensor()
                    level = ta.level_of_index_var(parent_index_var)
                    return f"{tv.name}{level}_size"
        return None

    def _should_use_scalar_accum(self, index_var: IndexVar) -> bool:
        """Check if this ForAll should use scalar accumulation for child reduction."""
        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "self.cin_lowerer is not None", code="iteration_lattice_state_missing"
            )
        result_tensor_access = self.cin_lowerer.result_tensor_access
        result_tensor_var = self.cin_lowerer.result_tensor_var
        if result_tensor_access is None or result_tensor_var is None:
            return False
        if result_tensor_access.is_workspace():
            return False
        if not result_tensor_access.has_index_var(index_var):
            return False

        # Result must be all-coordinate (COO)
        for lt in result_tensor_var.get_level_types():
            if lt != LevelType.COORDINATE:
                return False

        # This must be the last result level
        if (
            result_tensor_access.level_of_index_var(index_var)
            != result_tensor_var.levels - 1
        ):
            return False

        # Child must be a ForAll with a reduction variable (not in result)
        child_stmt = self.for_all_stmt.stmt
        if not isinstance(child_stmt, ForAll):
            return False
        child_iv = child_stmt.get_index_var()
        return not result_tensor_access.has_index_var(child_iv)

    def _wrap_child_subregion(
        self, lattice_point: "LatticePoint", index_var: IndexVar
    ) -> list:
        """Get child subregion loops, wrapping with scalar accumulation if needed."""
        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "self.cin_lowerer is not None", code="iteration_lattice_state_missing"
            )
        result_tensor_access = self.cin_lowerer.result_tensor_access
        result_tensor_var = self.cin_lowerer.result_tensor_var

        use_scalar_accum = self._should_use_scalar_accum(index_var)
        if use_scalar_accum:
            if result_tensor_access is None:
                raise ScorchAssertionError(
                    "result_tensor_access is not None",
                    code="iteration_lattice_state_missing",
                )
            self.cin_lowerer._require_scalar_accumulation_driver(
                result_tensor_access, self.for_all_stmt
            )
            self.cin_lowerer._scalar_accum_mode = True

        child_subregion = list(
            lattice_point.get_child_subregion_loops(
                self.cin_lowerer, self.for_all_stmt.stmt, self
            )
        )

        if use_scalar_accum:
            self.cin_lowerer._scalar_accum_mode = False
            self.cin_lowerer._used_scalar_accum = True
            if result_tensor_var is None:
                raise ScorchAssertionError(
                    "result_tensor_var is not None",
                    code="iteration_lattice_state_missing",
                )
            if result_tensor_access is None:
                raise ScorchAssertionError(
                    "result_tensor_access is not None",
                    code="iteration_lattice_state_missing",
                )
            result_name = result_tensor_var.name
            n_levels = result_tensor_var.levels
            pos_var = f"p{result_name}{n_levels - 1}"

            # Pre: <element type> _accum = 0;  for example float _accum = 0.0f;
            # or int64_t _accum = 0;.  The accumulator is declared with the
            # result's element type, the type its value array stores, so a
            # partial sum is never rounded to a narrower type than the result
            # holds.  An integer result accumulates exactly, modulo the width
            # of its type, as its value array would.
            accum_type = dtype_to_c_datatype(result_tensor_var.dtype)
            accum_init = llir.VarInit(
                var=llir.Var(name="_accum", type=accum_type),
                value=llir.Literal(
                    value=_SCALAR_ZERO_SPELLINGS.get(accum_type, "0"),
                    data_type=accum_type,
                ),
            )

            # Post: D_values[pD1] = _accum; D0_crd[pD1] = i; D1_crd[pD1] = j; pD1++;
            result_metadata = self.cin_lowerer._tensor_access_metadata(
                result_tensor_access,
                llir.TensorAccessRole.RESULT_WRITE,
            )
            post_stmts: list = []
            post_stmts.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_name}_values",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=pos_var,
                            type=llir.DataType.INT64,
                        ),
                        tensor_access=result_metadata,
                    ),
                    value=llir.Var(name="_accum", type=llir.DataType.NO_TYPE),
                    result_storage=self.cin_lowerer._result_storage(
                        result_tensor_access,
                        (RESULT_VALUES, None, STORAGE_WRITE),
                    ),
                )
            )
            sorted_result_ivars = result_tensor_access.get_sorted_index_vars()
            for level_idx, level_iv in enumerate(sorted_result_ivars):
                post_stmts.append(
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_name}{level_idx}_crd",
                                type=llir.DataType.NO_TYPE,
                            ),
                            index=llir.Var(
                                name=pos_var,
                                type=llir.DataType.INT64,
                            ),
                        ),
                        value=llir.Var(
                            name=level_iv.name,
                            type=llir.DataType.NO_TYPE,
                        ),
                        result_storage=self.cin_lowerer._result_storage(
                            result_tensor_access,
                            (RESULT_CRD, level_idx, STORAGE_WRITE),
                        ),
                    )
                )
            post_stmts.append(
                llir.Increment(
                    var=llir.Var(name=pos_var, type=llir.DataType.INT),
                )
            )

            # Mark SIMD on dense reduction loops inside the child subregion
            self._mark_simd_on_reduction_loops(child_subregion)

            child_subregion = [accum_init] + child_subregion + post_stmts

        return child_subregion

    @staticmethod
    def _mark_simd_on_reduction_loops(stmts: list) -> None:
        """Mark dense reduction loops inside scalar accumulation for SIMD."""
        mark_simd_on_reduction_loops(stmts)

    def _dense_result_levels_resolved_at(
        self, result_tensor_access: TensorAccess, level: int
    ) -> List[int]:
        """The dense result levels whose position the loop over ``level`` resolves.

        A dense level's position is its parent's position times its own extent
        plus its own coordinate.  So it can be computed once its own index
        variable is bound and its parent's position exists.  A parent that is not
        dense keeps its position in a counter declared for the whole kernel; a
        dense parent's position exists only inside the loop that resolved it.  A
        dense level therefore waits for the variable of every level of the run of
        dense levels that ends at it.

        Loops normally bind a result's levels outermost first, and the loop over
        a level then resolves that level and nothing else -- the one statement
        this lowering always emitted there.  A result that is dense in every level
        is not held to that order: the scheduler leaves an all-dense tensor out of
        its mode-order constraints because it is addressed by index arithmetic
        (``Scheduler._build_mode_order_graph``), and public ``einsum`` rebinds its
        dense operands to the selected loop order but not its result.
        ``einsum('ab,ca->cb')`` over a compressed ``A`` loops ``a, b, c`` and
        stores ``(c, b)``.  Emitting level 1's position in the loop over ``b``
        read level 0's position before the loop over ``c`` declared it, and the
        kernel did not compile.

        The loop that binds the last variable a level waits for resolves it, and
        with it every deeper level of the same run that waited only for that
        variable.  They are returned outermost level first, the order in which
        each statement reads the one before it.  Empty when ``level`` still waits
        for a variable an inner loop binds.
        """

        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "cin_lowerer must be set", code="iteration_lattice_state_missing"
            )
        bound = [
            enclosing.parent if enclosing.has_parent else enclosing
            for enclosing in self.cin_lowerer._enclosing_forall_vars
            if not (enclosing.tile_size_var and enclosing.is_outer)
        ]
        level_vars = result_tensor_access.get_sorted_index_vars()
        level_types = result_tensor_access.level_types()
        run_start = level
        while run_start > 0 and level_types[run_start - 1] == LevelType.DENSE:
            run_start -= 1
        resolved: List[int] = []
        for candidate in range(level, len(level_types)):
            if level_types[candidate] != LevelType.DENSE:
                break
            if any(
                level_var not in bound
                for level_var in level_vars[run_start : candidate + 1]
            ):
                break
            resolved.append(candidate)
        return resolved

    def dense_level_position(
        self, result_tensor_access: TensorAccess, level: int
    ) -> llir.Expr:
        """The position of the cell this lattice's loop is at, in a dense result level.

        Level 0: the coordinate.  Under dense levels only: the flattened
        coordinate ``((i0 * E1) + i1) * E2 + i2 ...``, each extent under the
        name its own loop runs to -- the spelling the typed lowering uses
        (``_assembly_catch_up_bound``), so both compilers emit one text.
        Under a compressed level: that level's pending position times this
        level's extent plus the coordinate, which is what the dense position
        resolve assigns.
        """

        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "cin_lowerer must be set", code="iteration_lattice_state_missing"
            )
        result_name = result_tensor_access.get_tensor().get_name()
        level_vars = result_tensor_access.get_sorted_index_vars()
        coordinate = llir.Var(name=level_vars[level].name, type=llir.DataType.INT)
        if level == 0:
            return coordinate
        if self.dense_index_var_end_var_llir is not None:
            self.cin_lowerer.result_dense_level_bound_names[level] = (
                self.dense_index_var_end_var_llir.name
            )
        level_types = result_tensor_access.level_types()
        if any(above != LevelType.DENSE for above in level_types[:level]):
            return llir.Add(
                llir.Mul(
                    left=llir.Var(
                        name=f"p{result_name}{level - 1}", type=llir.DataType.INT64
                    ),
                    right=llir.Var(
                        name=f"{result_name}{level}_size", type=llir.DataType.INT64
                    ),
                ),
                coordinate,
            )
        position: llir.Expr = llir.Var(name=level_vars[0].name, type=llir.DataType.INT)
        for inner in range(1, level + 1):
            position = llir.Add(
                left=llir.Mul(
                    left=position,
                    right=llir.Var(
                        name=self.cin_lowerer.result_dense_level_extent_name(
                            result_name, inner
                        ),
                        type=llir.DataType.INT64,
                    ),
                ),
                right=llir.Var(name=level_vars[inner].name, type=llir.DataType.INT),
            )
        return position

    def _dense_block_commit_stmts(
        self,
        result_tensor_access: TensorAccess,
        index_var: IndexVar,
        level: int,
    ) -> List[llir.Stmt]:
        """The commit of the coordinate above ``index_var``'s dense block.

        Emitted after the loop over a dense result level that sits directly
        below a compressed one; see :func:`dense_block_commit`.
        """

        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "cin_lowerer must be set", code="iteration_lattice_state_missing"
            )
        if level == 0:
            return []
        parent_index_var = result_tensor_access.get_parent_index_var(index_var)
        if parent_index_var is None:
            return []
        return dense_block_commit(
            self.cin_lowerer,
            result_tensor_access,
            level,
            llir.Var(name=parent_index_var.name, type=llir.DataType.INT),
        )

    def get_lattice_loops(self) -> List[llir.Stmt]:
        """
        Generate the outermost loops, one for each lattice point.

        """
        if self.cin_lowerer is None:
            raise ScorchAssertionError(
                "cin_lowerer must be set", code="iteration_lattice_state_missing"
            )

        stmts: List[llir.Stmt] = []

        lattice_points = self.get_lattice_points()
        lattice_point = lattice_points[0]

        result_tensor_var = self.cin_lowerer.result_tensor_var
        if result_tensor_var is None:
            raise ScorchAssertionError(
                "Result tensor var not set", code="iteration_lattice_state_missing"
            )
        result_tensor_name = result_tensor_var.name
        result_tensor_access = self.cin_lowerer.result_tensor_access
        if result_tensor_access is None:
            raise ScorchAssertionError(
                "Result tensor access not set", code="iteration_lattice_state_missing"
            )

        current_index_var = lattice_point.get_index_var()

        # Before the loops over the first level of a dense block below a
        # compressed level: the check that the block the pending coordinate
        # opens ends within the position bound, before the loops form and
        # narrow its positions.  Emitted exactly where the block's commit is.
        block_opens: List[llir.Stmt] = []
        if result_tensor_access.has_index_var(current_index_var):
            level = result_tensor_access.level_of_index_var(current_index_var)
            level_type = result_tensor_access.level_type_of_index_var(current_index_var)

            if level_type == LevelType.DENSE:
                commit = self._dense_block_commit_stmts(
                    result_tensor_access, current_index_var, level
                )
                if commit:
                    block_opens.append(
                        dense_block_check(
                            self.cin_lowerer.result_levels_bounded_in_kernel,
                            result_tensor_name,
                            result_tensor_access.level_types(),
                            level,
                        )
                    )
                stmts.extend(commit)
            if level_type == LevelType.COMPRESSED:
                stmts.extend(
                    [
                        llir.BlankLine(),
                        llir.Comment("Assembly compressed _level indices"),
                    ]
                )

                # if _level is > 0 and parent _level is also sparse, we need to set
                # the parent _level's crd
                if level > 0:
                    parent_index_var = result_tensor_access.get_parent_index_var(
                        current_index_var
                    )
                    if parent_index_var is None:
                        raise ScorchAssertionError(
                            "Parent index var is None",
                            code="iteration_lattice_state_missing",
                        )
                    parent_level_type = result_tensor_access.level_type_of_index_var(
                        parent_index_var
                    )

                    if parent_level_type == LevelType.COMPRESSED:
                        stmts.append(
                            # e.g.
                            # if (A1_pos.back() < pA1) {
                            #     A0_crd.push_back(i);
                            # }
                            llir.IfThenElse(
                                cond=llir.BinOp(
                                    op="<",
                                    left=llir.FunctionCall(
                                        name=f"{result_tensor_name}{level}_pos.back",
                                        args=[],
                                    ),
                                    right=llir.Var(
                                        name=f"p{result_tensor_name}{level}",
                                        type=llir.DataType.INT,
                                    ),
                                ),
                                then_body=[
                                    llir.FunctionCallStmt(
                                        name=f"{result_tensor_name}{level - 1}_crd.push_back",
                                        args=[
                                            llir.Var(
                                                name=parent_index_var.name,
                                                type=llir.DataType.INT,
                                            )
                                        ],
                                        result_storage=(
                                            self.cin_lowerer._result_storage(
                                                result_tensor_access,
                                                (
                                                    RESULT_CRD,
                                                    level - 1,
                                                    STORAGE_WRITE,
                                                ),
                                            )
                                        ),
                                    ),
                                    llir.Increment(
                                        var=llir.Var(
                                            name=f"p{result_tensor_name}{level - 1}",
                                            type=llir.DataType.INT,
                                        )
                                    ),
                                ],
                                result_storage=self.cin_lowerer._result_storage(
                                    result_tensor_access,
                                    (RESULT_POS, level, STORAGE_READ),
                                ),
                            )
                        )
                # Assemble pos array for this compressed level:
                # - Dense parent: A1_pos[A1_pos_index + 1] = A1_crd.size()
                # - Compressed parent: A1_pos[A0_crd.size()] = A1_crd.size()
                assembled_pos_array = False
                if level > 0:
                    parent_index_var = result_tensor_access.get_parent_index_var(
                        lattice_point.get_index_var()
                    )
                    if parent_index_var is None:
                        raise ScorchAssertionError(
                            "Parent index var is None",
                            code="iteration_lattice_state_missing",
                        )
                    if (
                        result_tensor_access.level_type_of_index_var(parent_index_var)
                        == LevelType.COMPRESSED
                    ):
                        # A1_pos[A0_crd.size()] = A1_crd.size()
                        stmts.append(
                            llir.Assign(
                                var=llir.ArrayAccess(
                                    array=llir.Var(
                                        name=f"{result_tensor_name}{level}_pos",
                                        type=llir.DataType.STD_VECTOR_C_INT,
                                    ),
                                    index=llir.FunctionCall(
                                        name=(
                                            f"{result_tensor_var.name}"
                                            f"{level - 1}_crd.size"
                                        ),
                                        args=[],
                                    ),
                                ),
                                value=llir.FunctionCall(
                                    name=f"{result_tensor_name}{level}_crd.size",
                                    args=[],
                                ),
                                result_storage=self.cin_lowerer._result_storage(
                                    result_tensor_access,
                                    (RESULT_POS, level, STORAGE_WRITE),
                                    (RESULT_CRD, level - 1, STORAGE_READ),
                                    (RESULT_CRD, level, STORAGE_READ),
                                ),
                            )
                        )
                        assembled_pos_array = True

                if not assembled_pos_array:
                    stmts.append(
                        # e.g. A1_pos[A1_pos_index + 1] = A1_crd.size()
                        llir.Assign(
                            var=llir.ArrayAccess(
                                array=llir.Var(
                                    name=f"{result_tensor_name}{level}_pos",
                                    type=llir.DataType.STD_VECTOR_C_INT,
                                ),
                                index=llir.Add(
                                    llir.Var(
                                        name=f"{result_tensor_name}{level}_pos_index",
                                        type=llir.DataType.INT,
                                    ),
                                    llir.Literal(1),
                                ),
                            ),
                            value=llir.FunctionCall(
                                name=f"{result_tensor_name}{level}_crd.size",
                                args=[],
                            ),
                            result_storage=self.cin_lowerer._result_storage(
                                result_tensor_access,
                                (RESULT_POS, level, STORAGE_WRITE),
                                (RESULT_CRD, level, STORAGE_READ),
                            ),
                        )
                    )

        def gen_single_lattice_loop(lattice_point: LatticePoint) -> List[llir.Stmt]:
            if not self.cin_lowerer:
                raise ScorchAssertionError(
                    "CINLowerer not set", code="iteration_lattice_state_missing"
                )
            if result_tensor_var is None:
                raise ScorchAssertionError(
                    "Result tensor var is None", code="iteration_lattice_state_missing"
                )
            if not (isinstance(result_tensor_access, TensorAccess)):
                raise ScorchAssertionError(
                    "Result tensor access is None",
                    code="iteration_lattice_state_missing",
                )

            stmts: List[llir.Stmt] = []

            iterators = lattice_point.get_iterators()

            # If we have a sparse level here, we need to set
            # {result_tensor_var}{level}_pos[p{result_tensor_var}{parent_level} + 1] to
            # p{result_tensor_var}{level}

            index_var = lattice_point.get_index_var()
            should_unroll_loop = bool(
                index_var.tile_size_var
                and index_var.is_inner
                and index_var.tile_size_var.unroll
            )
            # Tiled loops bind i_in/k_in while accesses retain the original
            # logical i/k. Result-index generation must use that parent.
            logical_index_var = index_var.parent if index_var.has_parent else index_var

            tiled_index_var_resolve_stmts: List[llir.Stmt] = []

            if index_var.tile_size_var and index_var.is_inner:
                tiled_index_var_resolve_stmts.append(
                    llir.Comment("Resolve tiled index var")
                )
                tiled_index_var_resolve_stmts.extend(
                    index_var.parent.get_resolve_llir_stmts()
                )
                self.cin_lowerer.defined_index_vars.append(index_var.parent)

                # Guard against out-of-bounds when tile size exceeds
                # the remaining dimension.  The unrolled inner loop
                # iterates up to the compile-time tile constant, so we
                # need a runtime break when the resolved index overshoots.
                parent_iv = index_var.parent
                _bound_name = self._tile_bound_var(parent_iv)
                if _bound_name:
                    tiled_index_var_resolve_stmts.append(
                        llir.IfThenElse(
                            cond=llir.BinOp(
                                op=">=",
                                left=llir.Var(
                                    name=parent_iv.name,
                                    type=llir.DataType.INT,
                                ),
                                right=llir.Var(
                                    name=_bound_name,
                                    type=llir.DataType.INT,
                                ),
                            ),
                            then_body=[llir.Break()],
                        )
                    )

            # Skip result index generation when the result tensor access
            # is already used as a lattice point dense access (broadcast
            # dimension) — its position variable is already emitted by
            # get_candidate_coordinate_stmts.
            result_already_in_lattice = (
                result_tensor_access in lattice_point.get_dense_tensor_accesses()
            )

            result_value_index_stmts: List[llir.Stmt] = []
            if (
                not result_tensor_access.is_workspace()
                and not (index_var.tile_size_var and index_var.is_outer)
                and result_tensor_access.has_index_var(logical_index_var)
                and not result_already_in_lattice
            ):
                level = result_tensor_access.level_of_index_var(logical_index_var)
                level_type = result_tensor_access.level_type_of_index_var(
                    logical_index_var
                )

                # Initialize the result index vars this loop resolves, if this
                # _level is dense: p<result tensor var name><_level>, the index
                # into the result value array.
                # If is _level 0, then p<result tensor var name><_level> = index var
                # If _level > 0, then p<result tensor var name><_level> =
                # p<result tensor var name><parent _level> * <size of this _level> + index var
                if level_type == LevelType.DENSE:
                    result_level_vars = result_tensor_access.get_sorted_index_vars()
                    for resolved_level in self._dense_result_levels_resolved_at(
                        result_tensor_access, level
                    ):
                        result_index_var = llir.Var(
                            name=f"p{result_tensor_var.name}{resolved_level}",
                            type=llir.DataType.INT,
                        )
                        resolved_coordinate = llir.Var(
                            name=result_level_vars[resolved_level].name,
                            type=llir.DataType.INT,
                        )
                        if resolved_level == 0:
                            result_value_index_stmts.append(
                                llir.VarInit(
                                    var=result_index_var,
                                    value=resolved_coordinate,
                                )
                            )
                            continue
                        result_value_index_stmts.append(
                            llir.VarInit(
                                var=result_index_var,
                                value=llir.BinOp(
                                    op="+",
                                    left=llir.BinOp(
                                        op="*",
                                        left=llir.Var(
                                            name=(
                                                f"p{result_tensor_var.name}"
                                                f"{resolved_level - 1}"
                                            ),
                                            type=llir.DataType.INT,
                                        ),
                                        # <result tensor name><level>_size
                                        right=llir.Var(
                                            name=(
                                                f"{result_tensor_var.name}"
                                                f"{resolved_level}_size"
                                            ),
                                            type=llir.DataType.INT,
                                        ),
                                    ),
                                    right=resolved_coordinate,
                                ),
                            )
                        )

                if result_value_index_stmts:
                    result_value_index_stmts = [
                        llir.Comment("Resolve index into dense level of values array")
                    ] + result_value_index_stmts
                if (
                    level_type == LevelType.DENSE
                    and level > 0
                    and self.dense_index_var_llir is None
                    and level + 1 < result_tensor_access.num_levels
                    and result_tensor_access.level_types()[level + 1]
                    == LevelType.COMPRESSED
                ):
                    # The catch-up a loop over stored coordinates owes for the
                    # compressed level below, placed after the coordinate and
                    # the position it reads are resolved.
                    result_value_index_stmts.extend(
                        compressed_child_catch_up(
                            self.cin_lowerer,
                            result_tensor_access,
                            level + 1,
                            self.dense_level_position(result_tensor_access, level),
                        )
                    )

            # One emission for every loop.  A loop that resolves no result level --
            # over a reduced index, into a workspace, or the outer loop of a tile
            # -- differs only in having no result positions to resolve.  It used to
            # be a second copy of the three cases below, and the copy had fallen
            # behind: for a single iterator it always re-initialized the iterator,
            # so the loop over one operand's remaining entries that follows a merge
            # of two operands started again at the segment's first entry and added
            # what the merge had already added (a sum reduced over a compressed
            # index, wrong values), and under a dense extent it put the iterator's
            # declaration where the dense variable's belongs (did not build).
            if len(iterators) == 0:
                for_loop = llir.ForLoop(
                    init=self.get_dense_iterator_init_stmt(),
                    cond=lattice_point.get_for_loop_condition(lattice=self),
                    update=lattice_point.get_single_iterator_advance_stmt(lattice=self),
                    body=[
                        *tiled_index_var_resolve_stmts,
                        *lattice_point.get_candidate_coordinate_stmts(lattice=self),
                        *result_value_index_stmts,
                        *self._wrap_child_subregion(lattice_point, index_var),
                    ],
                    unroll=should_unroll_loop,
                )
                stmts.append(for_loop)

            elif len(iterators) == 1 and not lattice_point.parent_lattice_point:
                # Note: we don't want to do this for a child lattice point
                # because we don't want to re-initialize the iterator(s)
                it = iterators[0]

                # When domain is dense but we have a sparse iterator,
                # use the dense variable for the loop and init the
                # sparse iterator separately before the loop.
                # Also generate coordinate iterator advancement in the
                # loop body (the for-loop update only advances the dense var).
                coord_advance_stmts: List[llir.Stmt] = []
                if self.dense_index_var_llir:
                    loop_init = self.get_dense_iterator_init_stmt()
                    stmts.append(it.get_init_stmt())
                    # Advance coordinate iterator: conditional on match
                    if (
                        it.level_type == LevelType.COORDINATE
                        and it.level + 1 < it.tensor_var.levels
                    ):
                        coord_advance_stmts = [
                            llir.IfThenElse(
                                cond=llir.BinOp(
                                    op="==",
                                    left=it.get_coord_var_llir(),
                                    right=lattice_point.get_index_var_llir(),
                                ),
                                then_body=[
                                    llir.Assign(
                                        var=it.get_iterator_var_llir(),
                                        value=llir.Var(
                                            name=f"p{it.tensor_var.name}{it.level + 1}_end",
                                            type=llir.DataType.INT,
                                        ),
                                    )
                                ],
                            )
                        ]
                    else:
                        coord_advance_stmts = [
                            llir.Assign(
                                var=it.get_iterator_var_llir(),
                                value=llir.BinOp(
                                    op="==",
                                    left=it.get_coord_var_llir(),
                                    right=lattice_point.get_index_var_llir(),
                                ),
                                op=llir.AssignOp.ADD_ASSIGN,
                                cast=True,
                            )
                        ]
                else:
                    loop_init = it.get_init_stmt()

                for_loop = llir.ForLoop(
                    init=loop_init,
                    cond=lattice_point.get_while_condition(lattice=self),
                    update=lattice_point.get_single_iterator_advance_stmt(lattice=self),
                    body=[
                        *tiled_index_var_resolve_stmts,
                        *lattice_point.get_candidate_coordinate_stmts(lattice=self),
                        *result_value_index_stmts,
                        *self._wrap_child_subregion(lattice_point, index_var),
                        *coord_advance_stmts,
                    ],
                    unroll=should_unroll_loop,
                )

                stmts.append(for_loop)

            else:
                while_loop = llir.WhileLoop(
                    cond=lattice_point.get_while_condition(lattice=self),
                    body=[
                        *tiled_index_var_resolve_stmts,
                        *lattice_point.get_candidate_coordinate_stmts(lattice=self),
                        *result_value_index_stmts,
                        *self._wrap_child_subregion(lattice_point, index_var),
                        *lattice_point.get_iterators_advance_stmts(lattice=self),
                    ],
                )

                stmts.append(while_loop)

            return stmts

        # When the domain is dense, the first lattice point's for-loop
        # already covers the full range with if/else sub-branches for each
        # sub-case. Subsequent lattice points are redundant (they would
        # generate duplicate remainder loops).
        points_to_lower = (
            [lattice_points[0]] if self.dense_index_var_llir else lattice_points
        )

        return (
            block_opens
            + flatten_2d_list(
                [
                    [
                        *gen_single_lattice_loop(p),
                    ]
                    for p in points_to_lower
                ]
            )
            + stmts
        )
