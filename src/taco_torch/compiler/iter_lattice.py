from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Dict, TYPE_CHECKING

from src.taco_torch.compiler import llir
from src.taco_torch.utils import flatten_2d_list

if TYPE_CHECKING:
    from src.taco_torch.compiler.cin_lowerer import CINLowerer


from src.taco_torch.compiler.cin import (
    IndexVar,
    ForAll,
    TensorAccess,
    IndexStmt,
    TensorAssign,
    CIN,
    BinaryOp,
    Operation,
    IndexExpr,
)
from src.taco_torch.format import LevelType
from src.taco_torch.compiler.iterator import ModeIterator


@dataclass(frozen=False)
class LatticePoint:
    """
    iterators: List of tensor accesses to actually loop over
    locators: List of tensor accesses to locate only
    """

    sparse_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)
    dense_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)
    iterators: Optional[List[ModeIterator]] = field(default_factory=list)
    child_lattice_points: Optional[List["LatticePoint"]] = field(default_factory=list)
    index_var: Optional[IndexVar] = None
    index_var_llir: Optional[llir.Var] = None

    def __add__(self, other):
        if isinstance(other, LatticePoint):
            return LatticePoint(
                sparse_tensor_accesses=self.sparse_tensor_accesses
                + other.sparse_tensor_accesses,
                dense_tensor_accesses=self.dense_tensor_accesses
                + other.dense_tensor_accesses,
            )
        else:
            return self

    def __radd__(self, other):
        return self.__add__(other)

    def __hash__(self):
        return hash(tuple(self.sparse_tensor_accesses)) + hash(
            tuple(self.dense_tensor_accesses)
        )

    def set_index_var(self, index_var: IndexVar):
        self.index_var = index_var
        self.index_var_llir = llir.Var(
            name=f"{index_var.name}",
            type=llir.DataType.INT,
        )

    def set_index_var_and_gen_iterators(
        self, index_var: IndexVar
    ) -> List[ModeIterator]:
        self.set_index_var(index_var)
        self.iterators = [
            ModeIterator(
                tensor_access=ta,
                index_var=index_var,
            )
            for ta in self.sparse_tensor_accesses
        ]
        return self.iterators

    def filter_and_set_children(
        self, lattice_points: List["LatticePoint"]
    ) -> List["LatticePoint"]:
        self.child_lattice_points = [
            lp for lp in lattice_points if lp.is_child_of(self)
        ]
        return self.child_lattice_points

    def get_while_condition(self):
        condition = None
        for it in self.iterators:
            this_condition = llir.BinOp(
                op="<",
                left=it.iterator_var_llir,
                right=it.iterator_var_end_var_llir,
            )
            if condition is None:
                condition = this_condition
            else:
                condition = llir.BinOp(op="&&", left=condition, right=this_condition)
        return condition

    def get_iterators_advance_stmts(self) -> List[llir.Stmt]:
        stmts = []

        if len(self.iterators) > 1:
            stmts.append(llir.Comment("Advance iterators"))
            for it in self.iterators:
                stmts.append(
                    llir.VarAssign(
                        var=it.iterator_var_llir,
                        value=llir.BinOp(
                            op="==",
                            left=it.iterator_var_llir,
                            right=self.index_var_llir,
                        ),
                        op="+=",
                        cast=True,
                    )
                )
        elif len(self.iterators) == 1:
            stmts.append(llir.Comment("Advance iterator"))
            stmts.append(
                llir.Increment(
                    var=self.iterators[0].iterator_var_llir,
                )
            )

        return stmts

    def get_simplified_cin(self, cin: CIN) -> Optional[CIN]:
        # Rewrite the CIN to eliminate tensors that have run out of values
        # Based on the lattice_point we are currently at
        if isinstance(cin, TensorAccess):
            if cin in self.sparse_tensor_accesses or cin in self.dense_tensor_accesses:
                return cin
            else:
                return None

        elif isinstance(cin, BinaryOp):

            left_new = self.get_simplified_cin(cin.left)
            right_new = self.get_simplified_cin(cin.right)

            if left_new is None and right_new is None:
                return None

            if left_new and right_new:
                assert isinstance(left_new, IndexExpr) and isinstance(
                    right_new, IndexExpr
                ), "Expected IndexExpr for left and right"
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
            assert isinstance(
                rewritten_inner_stmt, IndexStmt
            ), "Rewritten inner stmt is not an index stmt"
            return ForAll(
                index_var=cin.index_var,
                stmt=rewritten_inner_stmt,
            )

        elif isinstance(cin, TensorAssign):
            rewritten_rhs = self.get_simplified_cin(cin.rhs)
            assert isinstance(
                rewritten_rhs, IndexExpr
            ), "Rewritten rhs is not an index expr"
            return TensorAssign(
                lhs=cin.lhs,
                rhs=rewritten_rhs,
            )

    def get_child_subregion_loops(
        self, cin_lowerer: CINLowerer, cin: CIN
    ) -> List[llir.Stmt]:
        """
        Iterates over the child lattice points and generate an inner loop over each
        case. The inner loop uses a simplified/rewritten expression of the original
        CIN to ignore the tensors that have run out of values.
        """
        stmts = []
        if self.child_lattice_points:
            if_conditions = []
            then_body_list = []

            for child_lp in [self] + self.child_lattice_points:
                candidate_coord_var_llirs = map(
                    lambda it: it.coord_var_llir, child_lp.iterators
                )
                if_condition = None
                then_body = []

                for coord_var_llir in candidate_coord_var_llirs:
                    this_condition = llir.BinOp(
                        op="==",
                        left=coord_var_llir,
                        right=self.index_var_llir,
                    )
                    if if_condition is None:
                        if_condition = this_condition
                    else:
                        if_condition = llir.BinOp(
                            op="&&", left=if_condition, right=this_condition
                        )

                then_body.append(
                    cin_lowerer.lower_CIN(child_lp.get_simplified_cin(cin))
                )

                if_conditions.append(if_condition)
                then_body_list.append(then_body)

            stmts.append(
                llir.IfThenElse(
                    cond=if_conditions,
                    then_body=then_body_list,
                    else_body=[],
                )
            )
        else:
            stmts.append(cin_lowerer.lower_CIN(self.get_simplified_cin(cin)))
        return stmts

    def get_candidate_coordinate_stmts(self) -> List[llir.Stmt]:

        stmts = []

        if len(self.iterators) > 1:
            stmts.append(llir.Comment("Get candidate coordinates"))
            for it in self.iterators:
                stmts.append(
                    llir.VarAssign(
                        var=it.coord_var_llir,
                        value=it.coord_var_value_llir,
                    )
                )
            stmts.append(llir.Comment("Resolve coordinate"))
            stmts.append(
                llir.VarAssign(
                    var=self.index_var_llir,
                    value=llir.FunctionCall(
                        name="std::min",
                        args=[
                            llir.Array(
                                values=[it.coord_var_llir for it in self.iterators],
                                data_type=llir.DataType.INT,
                            )
                        ],
                    ),
                )
            )
        elif len(self.iterators) == 1:
            stmts.append(llir.Comment("Resolve coordinate"))
            stmts.append(
                llir.VarAssign(
                    var=self.index_var_llir,
                    value=self.iterators[0].coord_var_value_llir,
                )
            )

        return stmts

    def is_child_of(self, other: LatticePoint) -> bool:
        """
        A lattice_point is a child of another lattice point if its sparse tensor accesses is a
        strict subset of the parent's
        """
        return set(self.sparse_tensor_accesses).issubset(
            set(other.sparse_tensor_accesses)
        )


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

    def gen_lattice_points(self) -> List[LatticePoint]:
        """
        Generate the lattice points for the iteration lattice of the given
        iteration domain.
        """
        current_index_var = self.for_all_stmt.get_index_var()

        def intersect_accesses(
            left_accesses: List[List[TensorAccess]],
            right_accesses: List[List[TensorAccess]],
        ) -> List[List[TensorAccess]]:

            unflattened_intersected_accesses = list(
                product(left_accesses, right_accesses)
            )
            flattened_intersected_accesses = [
                flatten_2d_list(accesses)
                for accesses in unflattened_intersected_accesses
            ]
            return flattened_intersected_accesses

        def union_accesses(
            left_accesses: List[List[TensorAccess]],
            right_accesses: List[List[TensorAccess]],
        ) -> List[List[TensorAccess]]:
            return [
                *left_accesses,
                *right_accesses,
                left_accesses + right_accesses,
            ]

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
            return [
                *map(sum, product(left_lattice_points, right_lattice_points)),
            ]

        def get_lattice_points_from_cin(cin: CIN) -> List[LatticePoint]:
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
                    return union_lattice_points(
                        left_lattice_points, right_lattice_points
                    )
            if isinstance(cin, TensorAccess):
                # TODO: check Empty list if current_index_var is not in the tensor access
                if current_index_var not in cin.indices:
                    return []
                # if index variable correspond to a dense level, put in locators
                if (
                    cin.get_tensor().get_level_types()[
                        cin.get_index_vars().index(current_index_var)
                    ]
                    == LevelType.DENSE
                ):
                    return [LatticePoint(dense_tensor_accesses=[cin])]
                return [LatticePoint(sparse_tensor_accesses=[cin])]

        lattice_points = get_lattice_points_from_cin(self.for_all_stmt)

        # Sort lattice_points in descending number of iterators, then descending number of locators
        lattice_points.sort(
            key=lambda lp: (
                len(lp.sparse_tensor_accesses),
                len(lp.dense_tensor_accesses),
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

        return lattice_points

    def gen_parent_to_children_lattice_points(
        self,
    ) -> Dict[LatticePoint, List[LatticePoint]]:
        """
        Generate the parent to children lattice points mapping for the iteration lattice of the given
        iteration domain.

        """
        lattice_points = self.lattice_points
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
        # TODO: need to handle dense iterators
        all_mode_iterators = self.lattice_points[0].iterators

        return [
            llir.Comment("Initialize iterators"),
            *[mode_iterator.get_init_stmts() for mode_iterator in all_mode_iterators],
        ]

    def get_lattice_loops(self) -> List[llir.Stmt]:
        """
        Generate the outermost loops, one for each lattice point.

        """

        def gen_single_lattice_loop(lattice_point: LatticePoint) -> llir.Stmt:

            while_loop = llir.WhileLoop(
                cond=lattice_point.get_while_condition(),
                body=[
                    *lattice_point.get_candidate_coordinate_stmts(),
                    *lattice_point.get_child_subregion_loops(
                        self.cin_lowerer, self.for_all_stmt.stmt
                    ),
                    *lattice_point.get_iterators_advance_stmts(),
                ],
            )

            return while_loop

        return flatten_2d_list(
            [
                [
                    gen_single_lattice_loop(p),
                    llir.BlankLine(),
                ]
                for p in self.lattice_points
            ]
        )
