from typing import List, Optional, Any, Iterable
from itertools import chain, product

from taco_torch import llir
from taco_torch.cin import (
    IndexVar,
    ForAll,
    TensorVar,
    TensorAccess,
    IndexStmt,
    TensorAssign,
    CIN,
    BinaryOp,
    Operation,
)
from taco_torch.format import LevelType
from taco_torch.iterator import TensorIterators, ModeIterator
from dataclasses import dataclass, field


@dataclass(frozen=False)
class LatticePoint:
    """
    iterators: List of tensor accesses to actually loop over
    locators: List of tensor accesses to locate only
    """

    sparse_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)
    dense_tensor_accesses: Optional[List[TensorAccess]] = field(default_factory=list)
    iterators: Optional[List[ModeIterator]] = field(default_factory=list)

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

    def gen_iterators(self, index_var: IndexVar) -> List[ModeIterator]:
        self.iterators = [
            ModeIterator(
                tensor_access=ta,
                index_var=index_var,
            )
            for ta in self.sparse_tensor_accesses
        ]
        return self.iterators

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


@dataclass(frozen=False)
class IterationLattice:
    """
    The iteration lattice of an iteration domain contains an ordered set of
    lattice points, in decreasing order of the number of index variables they
    contain.

    """

    for_all_stmt: ForAll
    lattice_points: Optional[List[LatticePoint]] = None

    def gen_lattice_points(self) -> List[LatticePoint]:
        """
        Generate the lattice points for the iteration lattice of the given
        iteration domain.
        """
        current_index_var = self.for_all_stmt.get_index_var()

        def flatten_2d_list(lst: Iterable[List[Any]]) -> List[Any]:
            return list(chain(*lst))

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

    def __post_init__(self):
        if self.lattice_points is None:
            self.lattice_points = self.gen_lattice_points()
