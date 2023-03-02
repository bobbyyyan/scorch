from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Dict, TYPE_CHECKING, Iterable, Sequence

from src.scorch.compiler import llir
from src.scorch.utils import flatten_2d_list

if TYPE_CHECKING:
    from src.scorch.compiler.cin_lowerer import CINLowerer

from src.scorch.compiler.cin import (
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
from src.scorch.format import LevelType
from src.scorch.compiler.iterator import ModeIterator


@dataclass(frozen=False)
class LatticePoint:
    """
    iterators: List of tensor accesses to actually loop over
    locators: List of tensor accesses to locate only
    """

    sparse_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)  # type: ignore
    dense_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)  # type: ignore
    iterators: Optional[List[ModeIterator]] = field(default_factory=list)  # type: ignore
    dense_iterators: Optional[List[ModeIterator]] = field(default_factory=list)  # type: ignore
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

    def get_index_var(self) -> IndexVar:
        assert self.index_var is not None, "Index var not set"
        return self.index_var

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
        if not self.iterators:
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
        return self.child_lattice_points

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
            assert (
                lattice.dense_index_var_end_var_llir is not None
            ), "Dense index var upper bound not set"
            condition = llir.BinOp(
                op="<",
                left=lattice.dense_index_var_llir,
                right=lattice.dense_index_var_end_var_llir,
            )
        assert condition is not None, "Failed to generate while condition"
        return condition

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
                stmts.append(
                    llir.Increment(
                        var=lattice.dense_index_var_llir,
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

        all_inner_lattice_points = [self] + self.child_lattice_points

        if self.child_lattice_points or (self.iterators and len(self.iterators) > 1):
            stmts.append(llir.Comment("Inner loops over child regions"))
            if_conditions: List[llir.Expr] = []
            then_body_list = []
            else_body: List[llir.Stmt] = []

            for child_lp in all_inner_lattice_points:
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

                # If this lattice point has no (sparse) iterators, then we
                # are at the dense domain, set the would-be then_body as the else_body
                if not child_lp.iterators:
                    lower_cin_and_add_to_list(child_lp, cin, else_body)
                else:
                    lower_cin_and_add_to_list(child_lp, cin, then_body)
                    assert if_condition is not None, "if_condition cannot be None"
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
                    stmts.append(
                        llir.VarInit(
                            var=it.get_coord_var_llir(),
                            value=it.get_coord_var_value_llir(),
                        )
                    )
            # Only need to break ties among the resolved sparse coordinates
            # if we don't have a dense domain
            if not lattice.dense_index_var_llir:
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

        cin_lowerer = lattice.cin_lowerer
        assert cin_lowerer is not None, "CIN lowerer is None"

        if self.dense_iterators:
            for it in self.dense_iterators:
                # TODO: if parent iterator is not yet resolved (get this info from the lowerer)
                # then we need to push this resolution later

                if it.coord_var_value_llir:
                    dense_coord_resolve_stmt = llir.VarInit(
                        var=it.get_coord_var_llir(),
                        value=it.get_coord_var_value_llir(),
                    )
                    if (
                        it.parent_iterator
                        and it.parent_iterator.index_var
                        not in cin_lowerer.defined_index_vars
                    ):
                        dependent_index_var = it.parent_iterator.get_index_var()
                        existing_list = (
                            cin_lowerer.dep_index_var_to_dense_coord_resolution.get(
                                dependent_index_var, []
                            )
                        )
                        existing_list.append(dense_coord_resolve_stmt)
                        cin_lowerer.dep_index_var_to_dense_coord_resolution[
                            dependent_index_var
                        ] = existing_list
                    else:
                        stmts.append(dense_coord_resolve_stmt)

        current_index_var = self.get_index_var()
        if current_index_var in cin_lowerer.dep_index_var_to_dense_coord_resolution:
            stmts.extend(
                cin_lowerer.dep_index_var_to_dense_coord_resolution[current_index_var]
            )

        # stmts = (
        #     [llir.Comment("Resolve dense coordinates"), *stmts, llir.BlankLine()]
        #     if stmts
        #     else []
        # )

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

    # these are only set if the iteration domain is the universe
    dense_index_var: Optional[IndexVar] = None
    dense_index_var_llir: Optional[llir.Var] = None
    dense_index_var_end_var_llir: Optional[llir.Var] = None

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

        # If any of the lattice points have no iterators, then the iteration domain is the universe
        if any([not lp.sparse_tensor_accesses for lp in lattice_points]):
            self.dense_index_var = current_index_var
            assert self.cin_lowerer, "cin_lowerer must be set"
            self.dense_index_var_llir = self.cin_lowerer.lower_IndexVar(
                current_index_var
            )
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
                current_index_var
            )
            # end var is the <tensor_var_name><level>_size, type is int
            self.dense_index_var_end_var_llir = llir.Var(
                name=f"{first_dense_tensor_var.name}{level_of_current_index_var}_size",
                type=llir.DataType.INT,
            )

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
        stmts: List[llir.Stmt] = [llir.Comment("Initialize iterators")]
        lattice_points = self.get_lattice_points()
        assert len(lattice_points) > 0, "No lattice points generated"
        all_mode_iterators = lattice_points[0].get_iterators()
        if self.dense_index_var_llir:
            stmts.append(
                llir.VarInit(
                    var=self.dense_index_var_llir,
                    value=llir.Literal(value=0, data_type=llir.DataType.INT),
                )
            )

        stmts.extend(
            [*[mode_iterator.get_init_stmts() for mode_iterator in all_mode_iterators]]  # type: ignore
        )
        return stmts

    def get_lattice_loops(self) -> List[llir.Stmt]:
        """
        Generate the outermost loops, one for each lattice point.

        """

        def gen_single_lattice_loop(lattice_point: LatticePoint) -> List[llir.Stmt]:
            assert self.cin_lowerer, "CINLowerer not set"

            stmts: List[llir.Stmt] = []

            # If we have a sparse level here, we need to set
            # {result_tensor_var}{level}_pos[p{result_tensor_var}{parent_level} + 1] to
            # p{result_tensor_var}{level}
            result_tensor_var = self.cin_lowerer.result_tensor_var
            assert result_tensor_var, "Result tensor var not set"
            result_tensor_access = self.cin_lowerer.result_tensor_access
            assert result_tensor_access, "Result tensor access not set"
            index_var = lattice_point.get_index_var()
            level = result_tensor_access.level_of_index_var(index_var)
            level_type = result_tensor_access.level_type_of_index_var(index_var)

            while_loop = llir.WhileLoop(
                cond=lattice_point.get_while_condition(lattice=self),
                body=[
                    *lattice_point.get_candidate_coordinate_stmts(lattice=self),
                    *lattice_point.get_child_subregion_loops(
                        self.cin_lowerer, self.for_all_stmt.stmt
                    ),
                    *lattice_point.get_iterators_advance_stmts(lattice=self),
                ],
            )

            stmts.append(while_loop)

            if level_type == LevelType.COMPRESSED:
                stmts.extend(
                    [
                        llir.BlankLine(),
                        llir.Comment("Assembly compressed level indices"),
                    ]
                )

                # if level is > 0 and parent level is also sparse, we need to set
                # the parent level's crd
                if level > 0:
                    parent_index_var = result_tensor_access.get_parent_index_var(
                        lattice_point.get_index_var()
                    )
                    assert parent_index_var is not None, "Parent index var is None"
                    if (
                        result_tensor_access.level_type_of_index_var(parent_index_var)
                        == LevelType.COMPRESSED
                    ):
                        stmts.append(
                            # e.g.
                            # if (A1_pos.back() < pA1) {
                            #     A0_crd.push_back(i);
                            # }
                            llir.IfThenElse(
                                cond=llir.BinOp(
                                    op="<",
                                    left=llir.FunctionCall(
                                        name=f"{result_tensor_var.name}{level}_pos.back",
                                        args=[],
                                    ),
                                    right=llir.Var(
                                        name=f"p{result_tensor_var.name}{level}",
                                        type=llir.DataType.INT,
                                    ),
                                ),
                                then_body=[
                                    llir.FunctionCallStmt(
                                        name=f"{result_tensor_var.name}{level - 1}_crd.push_back",
                                        args=[
                                            llir.Var(
                                                name="i",
                                                type=llir.DataType.INT,
                                            )
                                        ],
                                    ),
                                ],
                            )
                        )

                stmts.append(
                    # e.g. A1_pos.push_back(pA1)
                    llir.FunctionCallStmt(
                        name=f"{result_tensor_var.name}{level}_pos.push_back",
                        args=[
                            llir.Var(
                                name=f"{result_tensor_var.get_name()}{level}_crd.size()",
                                # name=f"p{result_tensor_var.name}{level}",
                                type=llir.DataType.INT,
                            )
                        ],
                    )
                )

            return stmts

        return flatten_2d_list(
            [
                [
                    *gen_single_lattice_loop(p),
                    llir.BlankLine(),
                ]
                for p in self.get_lattice_points()
            ]
        )
