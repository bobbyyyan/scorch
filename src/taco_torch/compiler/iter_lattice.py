from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Dict, TYPE_CHECKING, Iterable, Sequence

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

    sparse_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)  # type: ignore
    dense_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)  # type: ignore
    iterators: Optional[List[ModeIterator]] = field(default_factory=list)  # type: ignore
    child_lattice_points: Optional[List["LatticePoint"]] = field(default_factory=list)  # type: ignore
    index_var: Optional[IndexVar] = None
    index_var_llir: Optional[llir.Var] = None

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

    def get_index_var_llir(self):
        assert self.index_var_llir is not None, "Index var LLIR not set"
        return self.index_var_llir

    def set_index_var_and_gen_iterators(
        self, index_var: IndexVar
    ) -> List[ModeIterator]:
        self.set_index_var(index_var)
        self.iterators = [
            ModeIterator(
                tensor_access=ta,
                index_var=index_var,
            )
            for ta in self.get_sparse_tensor_accesses()
        ]
        return self.iterators

    def filter_and_set_children(
        self, lattice_points: Iterable[LatticePoint]
    ) -> List[LatticePoint]:
        self.child_lattice_points = [
            lp for lp in lattice_points if lp.is_child_of(self)
        ]
        return self.child_lattice_points

    def get_while_condition(self):
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
        return condition

    def get_iterators_advance_stmts(self) -> Sequence[llir.Stmt]:
        stmts: List[llir.Stmt] = []

        iterators = self.get_iterators()

        if len(iterators) > 1:
            stmts.append(llir.Comment("Advance iterators"))
            for it in iterators:
                stmts.append(
                    llir.VarInit(
                        var=it.get_iterator_var_llir(),
                        value=llir.BinOp(
                            op="==",
                            left=it.get_iterator_var_llir(),
                            right=self.get_index_var_llir(),
                        ),
                        op="+=",
                        cast=True,
                    )
                )
        elif len(iterators) == 1:
            stmts.append(llir.Comment("Advance iterator"))
            stmts.append(
                llir.Increment(
                    var=iterators[0].get_iterator_var_llir(),
                )
            )

        return stmts

    def get_iterators(self) -> Sequence[ModeIterator]:
        assert self.iterators is not None, "Iterators not set"
        return self.iterators

    def get_sparse_tensor_accesses(self) -> List[TensorAccess]:
        assert self.sparse_tensor_accesses is not None, "Sparse tensor accesses not set"
        return self.sparse_tensor_accesses

    def get_dense_tensor_accesses(self) -> List[TensorAccess]:
        assert self.dense_tensor_accesses is not None, "Dense tensor accesses not set"
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

        raise NotImplementedError(f"Unhandled CIN type {type(cin)}")

    def get_child_subregion_loops(
        self, cin_lowerer: CINLowerer, cin: IndexStmt
    ) -> Sequence[llir.Stmt]:
        """
        Iterates over the child lattice points and generate an inner loop over each
        case. The inner loop uses a simplified/rewritten expression of the original
        CIN to ignore the tensors that have run out of values.
        """
        stmts: List[llir.Stmt] = []

        def lower_cin_and_add_to_list(
            lattice_point: LatticePoint, cin: IndexStmt, lst: List[llir.Stmt]
        ):
            simplified_cin = lattice_point.get_simplified_cin(cin)
            assert simplified_cin is not None, "Simplified CIN is None"
            assert isinstance(
                simplified_cin, IndexStmt
            ), "Simplified CIN is not an index stmt"
            cin_lowered = cin_lowerer.lower_IndexStmt(simplified_cin)
            if isinstance(cin_lowered, llir.Stmt):
                lst.append(cin_lowered)
            elif isinstance(cin_lowered, list):
                lst.extend(cin_lowered)

        if self.child_lattice_points:
            if_conditions: List[llir.Expr] = []
            then_body_list = []

            for child_lp in [self] + self.child_lattice_points:
                candidate_coord_var_llirs = map(
                    lambda it: it.get_coord_var_llir(), child_lp.get_iterators()
                )
                if_condition = None
                then_body: List[llir.Stmt] = []

                for coord_var_llir in candidate_coord_var_llirs:
                    this_condition = llir.BinOp(
                        op="==",
                        left=coord_var_llir,
                        right=self.get_index_var_llir(),
                    )
                    if if_condition is None:
                        if_condition = this_condition
                    else:
                        if_condition = llir.BinOp(
                            op="&&", left=if_condition, right=this_condition
                        )

                lower_cin_and_add_to_list(child_lp, cin, then_body)

                if_conditions.append(if_condition)
                then_body_list.append(then_body)

            stmts.append(
                llir.IfThenElse(
                    cond_list=if_conditions,
                    then_body_list=then_body_list,
                    else_body=[],
                )
            )
        else:
            lower_cin_and_add_to_list(self, cin, stmts)

        return stmts

    def get_candidate_coordinate_stmts(self) -> Sequence[llir.Stmt]:

        stmts: List[llir.Stmt] = []
        iterators = self.get_iterators()

        if len(iterators) > 1:
            stmts.append(llir.Comment("Get candidate coordinates"))
            for it in iterators:
                stmts.append(
                    llir.VarInit(
                        var=it.get_coord_var_llir(),
                        value=it.get_coord_var_value_llir(),
                    )
                )
            stmts.append(llir.Comment("Resolve coordinate"))
            stmts.append(
                llir.VarInit(
                    var=self.get_index_var_llir(),
                    value=llir.FunctionCall(
                        name="std::min",
                        args=[
                            llir.Array(
                                values=[it.get_coord_var_llir() for it in iterators],
                                data_type=llir.DataType.INT,
                            )
                        ],
                    ),
                )
            )
        elif len(iterators) == 1:
            stmts.append(llir.Comment("Resolve coordinate"))
            stmts.append(
                llir.VarInit(
                    var=self.get_index_var_llir(),
                    value=iterators[0].get_coord_var_value_llir(),
                )
            )

        return stmts

    def is_child_of(self, other: LatticePoint) -> bool:
        """
        A lattice_point is a child of another lattice point if its sparse tensor accesses is a
        strict subset of the parent's
        """
        return set(self.get_sparse_tensor_accesses()).issubset(
            set(other.get_sparse_tensor_accesses())
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
                *map(sum, product(left_lattice_points, right_lattice_points)),  # type: ignore
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
            raise NotImplementedError(f"Unsupported CIN: {cin}")

        lattice_points = get_lattice_points_from_cin(self.for_all_stmt)

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

        return lattice_points

    def get_lattice_points(self) -> Sequence[LatticePoint]:
        assert self.lattice_points is not None, "Lattice points not generated"
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
        # TODO: need to handle dense iterators
        lattice_points = self.get_lattice_points()
        assert len(lattice_points) > 0, "No lattice points generated"
        all_mode_iterators = lattice_points[0].get_iterators()
        return [
            llir.Comment("Initialize iterators"),
            *[mode_iterator.get_init_stmts() for mode_iterator in all_mode_iterators],  # type: ignore
        ]

    def get_lattice_loops(self) -> List[llir.Stmt]:
        """
        Generate the outermost loops, one for each lattice point.

        """

        def gen_single_lattice_loop(lattice_point: LatticePoint) -> llir.Stmt:

            assert self.cin_lowerer, "CINLowerer not set"

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
                for p in self.get_lattice_points()
            ]
        )
